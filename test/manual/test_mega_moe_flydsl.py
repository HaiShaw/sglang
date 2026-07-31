from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.mega_moe_flydsl import (
    _max_workload_tokens,
    _select_mtpr,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@pytest.mark.parametrize(
    ("local_mode", "global_has_extend", "expected_mtpr"),
    [
        (ForwardMode.EXTEND, True, 8192),
        (ForwardMode.DECODE, True, 8192),
        (ForwardMode.IDLE, True, 8192),
        (ForwardMode.DECODE, False, 128),
        (ForwardMode.IDLE, False, 128),
    ],
)
def test_dual_mtpr_uses_rank_global_extend_state(
    monkeypatch, local_mode, global_has_extend, expected_mtpr
):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "8192")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR", "128")
    forward_batch = SimpleNamespace(
        forward_mode=local_mode,
        is_extend_in_batch=global_has_extend,
    )

    selected_mtpr = _select_mtpr(forward_batch)

    assert selected_mtpr == expected_mtpr


def test_dual_mtpr_is_disabled_by_default(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "8192")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR", "0")
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
    )

    selected_mtpr = _select_mtpr(forward_batch)

    assert selected_mtpr == 8192


def test_workload_uses_rank_global_max_tokens():
    forward_batch = SimpleNamespace(global_max_num_tokens=8192)

    assert _max_workload_tokens(forward_batch, local_tokens=1) == 8192
