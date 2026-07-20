"""
PrefillDelayer — a cross-DP-rank prefill *coalescer* for SGLang.

This is a faithful port of ATOM's prefill coalescer (ROCm/ATOM#1611) to
SGLang's DP-attention scheduler. Under DP-attention each rank schedules
independently; left alone, ranks fire many prefill forwards that each carry
only a handful of tokens. Every prefill forward has ~fixed cost (kernel
launch, pad-to-shape, the lockstep MoE all-to-all), so a nearly-empty forward
wastes most of that cost.

The delayer's single job: hold back prefill admission until the accumulated
prefill is worth a forward, then release -- Nagle's algorithm for prefill.
While it holds, decode keeps running; TTFT is bounded so a held request never
starves. Crucially it preserves cross-DP phase alignment: it only releases
when every rank is prefill-ready (so all ranks enter prefill together and the
MoE collective stays aligned), except when a must-fire bound forces release.

Decision (evaluated every tick, on every rank, in lockstep). Each rank reports
local state; a single cross-DP all_gather reduces it; then every rank computes
the SAME FIRE/HOLD from the reduced values:

  n_prefillable   = #DP ranks with admittable prefill
  G_pending       = total pending prefill tokens across ranks
  G_running_dec   = total decode seqs across ranks
  any_kv_high/low = any prefillable rank at/above / below a KV watermark
  any_partial     = any rank mid-chunked-prefill
  any_queue_hot   = any rank's oldest waiting prefill aged past max_queue_ms

  if n_prefillable == 0:                          FIRE   # nothing to do
  # -- must-fire bounds (release even if unaligned / underfilled) --
  if G_running_dec == 0:                          FIRE   # no decode to hide the wait
  if any_kv_high or any_kv_low:                   FIRE   # KV pressure / starvation
  if any_queue_hot:                               FIRE   # end-to-end TTFT SLA guard
  if hold_ticks >= ttft_max_ticks:                FIRE   # single-hold TTFT bound
  if any_partial and hold_ticks >= partial_max_ticks: FIRE
  # -- alignment gate: never fire while some rank lacks prefill (anti-skew) --
  if n_prefillable < dp_size:                     HOLD
  # -- goal: fire once the aggregate fills a worthwhile forward --
  fill = G_pending / (n_prefillable * budget)
  if fill >= target_fill:                         FIRE
  # -- adaptive give-up: queue stopped growing, waiting longer is futile --
  if G_pending <= prev_G_pending: stall += 1 else stall = 0
  if stall >= stall_ticks:                        FIRE
  HOLD

All timing is tick-based (hold_ticks), deterministic across ranks. The one
wall-clock input (oldest_waiting_age_ms) is compared to the threshold locally
on each rank (disjoint request sets) and only the OR crosses the collective,
so ranks never diverge on a timeout boundary.
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)

# Local-info tensor slot layout (per rank, gathered across DP ranks).
_F_PREFILLABLE = 0
_F_PENDING = 1
_F_RUNNING_DEC = 2
_F_KV_HIGH = 3
_F_KV_LOW = 4
_F_PARTIAL = 5
_F_QUEUE_HOT = 6
_N_FIELDS = 7


class PrefillDelayer:
    def __init__(
        self,
        dp_size: int,
        attn_tp_size: int,
        cpu_group,
        server_args,
        max_prefill_tokens: int,
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
        device: Optional["torch.device"] = "cpu",
        device_group=None,
    ):
        self.dp_size = dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        self.max_num_batched_tokens = max_prefill_tokens

        # Coalescer knobs (ATOM taxonomy). target_fill in (0, 1]; tick bounds >= 1.
        target_fill = server_args.prefill_delayer_target_fill
        if target_fill is None:
            target_fill = 0.7
        self.target_fill = min(max(float(target_fill), 0.05), 1.0)
        self.ttft_max_ticks = self._clamp_ticks(
            "ttft_max_ticks", server_args.prefill_delayer_ttft_max_ticks
        )
        self.partial_max_ticks = self._clamp_ticks(
            "partial_max_ticks", server_args.prefill_delayer_partial_max_ticks
        )
        self.stall_ticks = self._clamp_ticks(
            "stall_ticks", server_args.prefill_delayer_stall_ticks
        )
        self.kv_high_watermark = server_args.prefill_delayer_kv_high_watermark
        self.token_usage_low_watermark = (
            server_args.prefill_delayer_token_usage_low_watermark
        )
        # TTFT SLA guard threshold (ms); None disables it.
        self.max_queue_ms = server_args.prefill_delayer_max_queue_ms

        # Mirror scheduler_dp_attn_mixin's NCCL all-gather path: when the env
        # flag is on (or overlap scheduling is disabled), ride the NCCL device
        # group on `device` instead of gloo on CPU.
        use_nccl = (
            server_args.disable_overlap_schedule
            or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
        )
        if use_nccl:
            assert (
                device_group is not None
            ), "device_group is required when using NCCL for PrefillDelayer all-gather"
            self._gather_group = device_group
            self._gather_device = device
        else:
            self._gather_group = cpu_group
            self._gather_device = "cpu"

        dp_size_dim = dp_size if self.enable_dp_attention else 1
        self._global_info_buffer = torch.empty(
            (dp_size_dim, attn_tp_size, _N_FIELDS),
            dtype=torch.int64,
            device=self._gather_device,
        )

        self._metrics_collector = metrics_collector

        # Episode state. All ticks decide FIRE/HOLD in lockstep, so these evolve
        # identically on every rank (deterministic).
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1
        # First call fires immediately to seed the initial decode batch build-up.
        self._first = True

        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

        logger.info(
            f"PrefillDelayer(coalescer) initialized: dp_size={dp_size} "
            f"max_num_batched_tokens={self.max_num_batched_tokens} "
            f"target_fill={self.target_fill} "
            f"ttft_max_ticks={self.ttft_max_ticks} "
            f"partial_max_ticks={self.partial_max_ticks} "
            f"stall_ticks={self.stall_ticks} "
            f"kv_high_watermark={self.kv_high_watermark} "
            f"token_usage_low_watermark={self.token_usage_low_watermark} "
            f"max_queue_ms={self.max_queue_ms}"
        )

    @staticmethod
    def _clamp_ticks(name: str, value: int) -> int:
        if value is None or value < 1:
            logger.warning(
                f"{name}={value} < 1 would fire on the first tick (bound "
                "disabled); clamping to 1."
            )
            return 1
        return value

    def should_allow_prefill(
        self,
        prefillable: bool,
        pending_tokens: int,
        running_decode_batch: int = 0,
        kv_usage: float = 0.0,
        has_partial: bool = False,
        oldest_waiting_age_ms: float = 0.0,
    ) -> bool:
        """Return True iff this rank may admit new prefills this tick (FIRE).

        MUST be called every tick on every DP rank (it runs a cross-DP
        all_gather) so ranks stay in lockstep.
        """
        # First call fires unconditionally (one-time warmup seed).
        if self._first:
            self._first = False
            self._reset()
            return True

        low = self.token_usage_low_watermark
        kv_high = prefillable and kv_usage >= self.kv_high_watermark
        kv_low = prefillable and low is not None and kv_usage < low
        queue_hot = (
            prefillable
            and self.max_queue_ms is not None
            and oldest_waiting_age_ms >= self.max_queue_ms
        )

        (
            n_prefillable,
            g_pending,
            g_running_dec,
            any_kv_high,
            any_kv_low,
            any_partial,
            any_queue_hot,
        ) = self._gather_reduce(
            prefillable=prefillable,
            pending_tokens=pending_tokens,
            running_decode_batch=running_decode_batch,
            kv_high=kv_high,
            kv_low=kv_low,
            has_partial=has_partial,
            queue_hot=queue_hot,
        )

        # Nothing to prefill anywhere -> allow (vacuous), reset the episode.
        if n_prefillable == 0:
            self._reset()
            return self._observe(True, "vacuous", n_prefillable)

        # ---- must-fire bounds (release even if unaligned / underfilled) ----
        if g_running_dec == 0:
            return self._fire("nodecode", n_prefillable)
        if any_kv_high or any_kv_low:
            return self._fire("kv", n_prefillable)
        if any_queue_hot:
            return self._fire("queue_ms", n_prefillable)
        if self._hold_ticks >= self.ttft_max_ticks:
            return self._fire("ttft", n_prefillable)
        if any_partial and self._hold_ticks >= self.partial_max_ticks:
            return self._fire("partial", n_prefillable)

        # ---- alignment gate: never fire while a rank lacks prefill (anti-skew) ----
        if n_prefillable < self.dp_size:
            return self._hold(n_prefillable)

        # ---- goal: fire once the aggregate fills a worthwhile forward ----
        budget = n_prefillable * self.max_num_batched_tokens
        fill = g_pending / budget if budget > 0 else 1.0
        if fill >= self.target_fill:
            return self._fire("fill", n_prefillable)

        # ---- adaptive give-up: queue stopped growing -> waiting is futile ----
        if g_pending <= self._prev_pending:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._prev_pending = g_pending
        if self._stall_count >= self.stall_ticks:
            return self._fire("stall", n_prefillable)

        return self._hold(n_prefillable)

    def _gather_reduce(
        self,
        prefillable: bool,
        pending_tokens: int,
        running_decode_batch: int,
        kv_high: bool,
        kv_low: bool,
        has_partial: bool,
        queue_hot: bool,
    ):
        local_info = torch.tensor(
            [
                int(prefillable),
                int(pending_tokens),
                int(running_decode_batch),
                int(kv_high),
                int(kv_low),
                int(has_partial),
                int(queue_hot),
            ],
            device=self._gather_device,
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._gather_group,
        )
        # One row per DP rank (attn-tp rank 0 represents each DP group).
        tp0_info = self._global_info_buffer[:, 0, :]
        n_prefillable = int(tp0_info[:, _F_PREFILLABLE].sum().item())
        g_pending = int(tp0_info[:, _F_PENDING].sum().item())
        g_running_dec = int(tp0_info[:, _F_RUNNING_DEC].sum().item())
        any_kv_high = tp0_info[:, _F_KV_HIGH].max().item() > 0
        any_kv_low = tp0_info[:, _F_KV_LOW].max().item() > 0
        any_partial = tp0_info[:, _F_PARTIAL].max().item() > 0
        any_queue_hot = tp0_info[:, _F_QUEUE_HOT].max().item() > 0
        return (
            n_prefillable,
            g_pending,
            g_running_dec,
            any_kv_high,
            any_kv_low,
            any_partial,
            any_queue_hot,
        )

    def _fire(self, reason: str, n_prefillable: int) -> bool:
        hold_ticks = self._hold_ticks
        self._reset()
        if _DEBUG_LOG:
            logger.info(
                f"[PrefillDelayer] FIRE ({reason}): n_prefillable={n_prefillable} "
                f"hold_ticks={hold_ticks}"
            )
        return self._observe(True, reason, n_prefillable, forward_passes=hold_ticks)

    def _hold(self, n_prefillable: int) -> bool:
        self._hold_ticks += 1
        if _DEBUG_LOG:
            logger.info(
                f"[PrefillDelayer] HOLD: n_prefillable={n_prefillable} "
                f"hold_ticks={self._hold_ticks} stall={self._stall_count}"
            )
        return self._observe(False, "delay", n_prefillable)

    def _reset(self) -> None:
        self._hold_ticks = 0
        self._stall_count = 0
        self._prev_pending = -1

    def _observe(
        self,
        output_allow: bool,
        output_reason: str,
        n_prefillable: int,
        forward_passes: int = 0,
    ) -> bool:
        if self._metrics_collector is not None:
            self._metrics_collector.observe_prefill_delayer_outcome(
                forward_passes=forward_passes,
                wait_seconds=0.0,
                input_estimation=str(n_prefillable),
                output_allow=output_allow,
                output_reason=output_reason,
                actual_execution=output_allow,
            )
        return output_allow
