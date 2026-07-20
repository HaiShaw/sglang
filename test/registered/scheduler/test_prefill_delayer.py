import asyncio
import os
import re
import time
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import openai
import requests
import torch

from sglang.benchmark.serving import run_benchmark
from sglang.srt.managers.prefill_delayer import PrefillDelayer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
    run_distributed_test,
)

register_cuda_ci(
    est_time=300,
    stage="base-c",
    runner_config="8-gpu-h200",
    disabled="Temporarily disabled",
)

WORLD_SIZE = os.environ.get("SGLANG_TEST_WORLD_SIZE", "8")

# ============================ Unit Tests ============================
# Exercise the ATOM-style tick state machine (alignment gate / must-fire bounds
# / fill target / stall give-up) over a real cross-DP all_gather.


def _make_server_args(**overrides):
    base = dict(
        enable_dp_attention=True,
        disable_overlap_schedule=False,
        prefill_delayer_target_fill=0.7,
        prefill_delayer_ttft_max_ticks=30,
        prefill_delayer_partial_max_ticks=8,
        prefill_delayer_stall_ticks=3,
        prefill_delayer_kv_high_watermark=0.9,
        prefill_delayer_token_usage_low_watermark=None,
        prefill_delayer_max_queue_ms=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@dataclass
class Tick:
    """Per-tick, per-rank inputs to should_allow_prefill (index by rank)."""

    prefillable: List[bool]
    pending_tokens: List[int]
    running_decode_batch: List[int]
    kv_usage: List[float] = None
    has_partial: List[bool] = None
    oldest_waiting_age_ms: List[float] = None


@dataclass
class DelayerCase:
    name: str
    max_prefill_tokens: int
    ticks: List[Tick]
    expected_allow: bool  # asserted on the LAST tick
    server_args_overrides: dict = field(default_factory=dict)


def _run_delayer_test(rank, test_cases):
    world_size = torch.distributed.get_world_size()
    cpu_group = torch.distributed.new_group(backend="gloo")

    for case in test_cases:
        delayer = PrefillDelayer(
            dp_size=world_size,
            attn_tp_size=1,
            cpu_group=cpu_group,
            server_args=_make_server_args(**case.server_args_overrides),
            max_prefill_tokens=case.max_prefill_tokens,
        )
        result = True
        for tick in case.ticks:
            kv = tick.kv_usage or [0.5] * world_size
            partial = tick.has_partial or [False] * world_size
            age = tick.oldest_waiting_age_ms or [0.0] * world_size
            result = delayer.should_allow_prefill(
                prefillable=tick.prefillable[rank],
                pending_tokens=tick.pending_tokens[rank],
                running_decode_batch=tick.running_decode_batch[rank],
                kv_usage=kv[rank],
                has_partial=partial[rank],
                oldest_waiting_age_ms=age[rank],
            )
        assert result == case.expected_allow, (
            f"Case {case.name} rank {rank}: got {result}, "
            f"expected {case.expected_allow}"
        )


def _all(v, n=4):
    return [v] * n


_DELAYER_TEST_CASES = [
    # First call always fires (warmup seed).
    DelayerCase(
        name="first_fire",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            )
        ],
        expected_allow=True,
    ),
    # Vacuous: no rank prefillable -> allow.
    DelayerCase(
        name="vacuous",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(False),
                pending_tokens=_all(0),
                running_decode_batch=_all(4),
            ),
        ],
        expected_allow=True,
    ),
    # Alignment gate: some ranks lack prefill, decode running, low fill -> HOLD.
    DelayerCase(
        name="alignment_gate_hold",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=[True, True, False, True],
                pending_tokens=[100, 100, 0, 100],
                running_decode_batch=_all(4),
            ),
        ],
        expected_allow=False,
    ),
    # Fill target reached: all prefillable, pending sum >= 0.7*(4*1000) -> FIRE.
    DelayerCase(
        name="fill_target_fire",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(800),
                running_decode_batch=_all(4),
            ),
        ],
        expected_allow=True,
    ),
    # All prefillable but underfilled and no decode -> nodecode must-fire.
    DelayerCase(
        name="nodecode_fire",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(0),
            ),
        ],
        expected_allow=True,
    ),
    # All prefillable, underfilled, decode running -> HOLD (coalesce).
    DelayerCase(
        name="underfilled_hold",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
        ],
        expected_allow=False,
    ),
    # KV-high must-fire: a prefillable rank at/above watermark -> FIRE.
    DelayerCase(
        name="kv_high_fire",
        max_prefill_tokens=1000,
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
                kv_usage=[0.95, 0.5, 0.5, 0.5],
            ),
        ],
        expected_allow=True,
    ),
    # TTFT queue guard: one rank's oldest waiting prefill aged past threshold.
    DelayerCase(
        name="queue_guard_fire",
        max_prefill_tokens=1000,
        server_args_overrides=dict(prefill_delayer_max_queue_ms=100.0),
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
                oldest_waiting_age_ms=[500.0, 0.0, 0.0, 0.0],
            ),
        ],
        expected_allow=True,
    ),
    # Stall give-up: underfilled but pending stops growing for stall_ticks -> FIRE.
    DelayerCase(
        name="stall_fire",
        max_prefill_tokens=1000,
        server_args_overrides=dict(prefill_delayer_stall_ticks=2),
        ticks=[
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(10),
                running_decode_batch=_all(4),
            ),
            # hold (fill low), pending stays flat across the next ticks
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(50),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(50),
                running_decode_batch=_all(4),
            ),
            Tick(
                prefillable=_all(True),
                pending_tokens=_all(50),
                running_decode_batch=_all(4),
            ),
        ],
        expected_allow=True,
    ),
]


class TestPrefillDelayerStateMachine(unittest.TestCase):
    def test_state_machine(self):
        run_distributed_test(
            _run_delayer_test,
            world_size=4,
            backend="gloo",
            test_cases=_DELAYER_TEST_CASES,
        )


# ============================ E2E Tests ============================


class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            self,
            test_name="online_serving",
            other_launch_args=[
                # Not really needed, only to test support non-FCFS algorithms
                "--schedule-policy",
                "lpm",
            ],
            other_benchmark_args=dict(
                num_prompts=500,
                random_input_len=30000,
                random_output_len=256,
                request_rate=32,
            ),
            # TODO: re-enable a throughput-improvement assertion once a
            # workload that reliably exercises PrefillDelayer in online-
            # serving mode is available. The current workload yields run-
            # to-run noise on H200, while the offline test below shows the
            # same code path is healthy (improvement ~+27%). We still
            # validate functionality (server boot, benchmark completion,
            # metrics emission).
            min_improvement_pct=None,
        )


class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            self,
            test_name="offline_gen",
            other_launch_args=["--max-total-tokens", "200000"],
            other_benchmark_args=dict(
                num_prompts=800,
                random_input_len=30000,
                random_output_len=500,
            ),
            token_usage_low_watermark=0.8,
            min_improvement_pct=20,
        )


def _run_throughput_comparison(
    test_case,
    test_name: str,
    other_launch_args,
    other_benchmark_args,
    min_improvement_pct: Optional[float],
    token_usage_low_watermark: float = None,
):
    common_kwargs = dict(
        debug_name=test_name,
        other_launch_args=other_launch_args,
        other_benchmark_args=other_benchmark_args,
        token_usage_low_watermark=token_usage_low_watermark,
    )
    res_enabled = _run_throughput_test(prefill_delayer=True, **common_kwargs)
    res_disabled = _run_throughput_test(prefill_delayer=False, **common_kwargs)

    _assert_throughput_improvement(
        test_case,
        test_name=test_name,
        res_enabled=res_enabled,
        res_disabled=res_disabled,
        min_improvement_pct=min_improvement_pct,
    )


def _run_throughput_test(
    debug_name: str,
    prefill_delayer: bool,
    other_launch_args,
    other_benchmark_args,
    token_usage_low_watermark: float = None,
):
    model = "Qwen/Qwen3-0.6B"
    base_url = DEFAULT_URL_FOR_TEST

    process = _launch_server(
        prefill_delayer=prefill_delayer,
        model=model,
        base_url=base_url,
        other_args=other_launch_args,
        token_usage_low_watermark=token_usage_low_watermark,
    )

    try:
        args = get_benchmark_args(
            base_url=base_url,
            dataset_name="random",
            tokenizer=model,
            **other_benchmark_args,
        )
        res = run_benchmark(args)
        _print_prefill_delayer_metrics(base_url, expect_metrics=prefill_delayer)
    finally:
        kill_process_tree(process.pid)

    print(f"=== {debug_name} ({prefill_delayer=}) ===")
    res["total_throughput"] = res["input_throughput"] + res["output_throughput"]
    print(f"Input throughput: {res['input_throughput']:.2f} token/s")
    print(f"Output throughput: {res['output_throughput']:.2f} token/s")
    print(f"Total throughput: {res['total_throughput']:.2f} token/s")

    return res


def _assert_throughput_improvement(
    test_case,
    test_name: str,
    res_enabled: dict,
    res_disabled: dict,
    min_improvement_pct: Optional[float],
):
    test_case.assertEqual(
        WORLD_SIZE,
        "8",
        f"This test requires 8 GPUs to properly measure throughput improvement, got {WORLD_SIZE}",
    )

    enabled = res_enabled["total_throughput"]
    disabled = res_disabled["total_throughput"]
    improvement_pct = (enabled - disabled) / disabled * 100

    print(f"\n=== {test_name} Throughput Comparison ===")
    print(
        f"Total: enabled={enabled:.2f}, disabled={disabled:.2f}, improvement={improvement_pct:.2f}%"
    )

    if min_improvement_pct is None:
        # Functionality-only mode: skip the perf assertion.
        return

    test_case.assertGreaterEqual(
        improvement_pct,
        min_improvement_pct,
        f"{test_name}: Throughput improvement ({improvement_pct:.2f}%) < {min_improvement_pct}%",
    )


class TestPrefillDelayerTokenUsageLowWatermark(CustomTestCase):
    def test_1_with_low_watermark(self):
        # The kv cache size here is deliberately small, thus we use smaller token usage
        self._run(token_usage_low_watermark=0.5)

    # TODO: re-enable once sglang/sglang#22511 (DP-attention detokenizer
    # hang on H200 in CI) is fixed.
    @unittest.skip("blocked by sgl-project/sglang#22511")
    def test_2_without_low_watermark(self):
        self._run(token_usage_low_watermark=None)

    def _run(self, token_usage_low_watermark):
        model = "Qwen/Qwen3-0.6B"
        base_url = DEFAULT_URL_FOR_TEST
        world_size = int(WORLD_SIZE)

        process = _launch_server(
            model=model,
            base_url=base_url,
            prefill_delayer=True,
            other_args=["--max-total-tokens", "50000"],
            # e.g. gen throughput is 370 tok/s on H200.
            # Will need a different threshold on B200
            max_delay_passes=3000,
            token_usage_low_watermark=token_usage_low_watermark,
        )

        async def run_test():
            client = openai.AsyncClient(base_url=f"{base_url}/v1", api_key="EMPTY")
            long_prompt = "Hello " * 5000

            async def send_blocking_request():
                return await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": long_prompt}],
                    max_tokens=10000,
                    extra_body={"data_parallel_rank": 0},
                )

            async def send_normal_request(dp_rank, req_idx):
                start = time.time()
                await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say hi"}],
                    max_tokens=10,
                    extra_body={"data_parallel_rank": dp_rank},
                )
                elapsed = time.time() - start
                return dp_rank, req_idx, elapsed

            asyncio.create_task(send_blocking_request())
            await asyncio.sleep(3)

            num_reqs_per_rank = 10
            results = await asyncio.gather(
                *[
                    send_normal_request(dp_rank, req_idx)
                    for dp_rank in range(1, world_size)
                    for req_idx in range(num_reqs_per_rank)
                ]
            )

            enabled = token_usage_low_watermark is not None
            thresh = 5
            for dp_rank, req_idx, elapsed in results:
                print(f"DP rank {dp_rank} req {req_idx} completed in {elapsed:.2f}s")
                self.assertTrue(
                    (elapsed < thresh) if enabled else (elapsed > thresh),
                    f"DP rank {dp_rank} req {req_idx}: elapsed={elapsed:.2f}s, thresh={thresh}, enabled={enabled}. "
                    f"Maybe you need a different `max_delay_passes` when using hardware other than H200.",
                )

        try:
            asyncio.run(run_test())

            metrics_text = _print_prefill_delayer_metrics(base_url, expect_metrics=True)
            if token_usage_low_watermark is not None:
                total = _sum_prometheus_metric_values(metrics_text, "kv")
                self.assertGreater(total, 0, "Expected kv-watermark releases > 0")
                print(f"total kv fires: {total}")
        finally:
            kill_process_tree(process.pid)


class TestPrefillDelayerAccuracy(CustomTestCase):
    def test_1_gsm8k_has_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=True)

    def test_2_gsm8k_no_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=False)

    def _run_accuracy_test(self, prefill_delayer: bool):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = _launch_server(
            prefill_delayer=prefill_delayer,
            model=model,
            base_url=base_url,
            other_args=[
                # Not really needed, only to test support non-FCFS algorithms
                "--schedule-policy",
                "lpm",
                # Use this to ensure prefill delayer will be run
                "--max-total-tokens",
                "4096",
            ],
        )
        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="gsm8k",
                num_examples=None,
                num_threads=1024,
            )
            metrics = run_eval(args)
            print(f"=== gsm8k ({prefill_delayer=}) ===")
            print(f"{metrics=}")
            self.assertGreater(metrics["score"], 0.57)
        finally:
            kill_process_tree(process.pid)


def _launch_server(
    *,
    model,
    base_url,
    prefill_delayer: bool,
    other_args,
    max_delay_passes: int = 100,
    token_usage_low_watermark: float = None,
):
    os.environ["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

    return popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[
            "--trust-remote-code",
            "--tp",
            WORLD_SIZE,
            "--enable-dp-attention",
            "--dp",
            WORLD_SIZE,
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.6",
            "--enable-metrics",
            *(["--enable-prefill-delayer"] if prefill_delayer else []),
            *(
                [
                    "--prefill-delayer-token-usage-low-watermark",
                    str(token_usage_low_watermark),
                ]
                if token_usage_low_watermark is not None
                else []
            ),
            *(other_args or []),
        ],
    )


def _print_prefill_delayer_metrics(base_url: str, expect_metrics: bool) -> str:
    metrics_response = requests.get(f"{base_url}/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    prefill_delayer_metrics = [
        line for line in metrics_text.split("\n") if "prefill_delayer" in line
    ]
    print("=== PrefillDelayer Metrics ===")
    for line in prefill_delayer_metrics:
        print(line)
    if expect_metrics:
        assert "sglang:prefill_delayer_outcomes_total" in metrics_text
    return metrics_text


def _sum_prometheus_metric_values(metrics_text: str, label_value: str) -> int:
    matches = re.findall(rf'{label_value}".*?\}} (\d+)', metrics_text)
    return sum(int(m) for m in matches)


if __name__ == "__main__":
    unittest.main()
