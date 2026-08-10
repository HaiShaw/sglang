"""Fail-closed adapter for AITER's gfx950 Kimi-K3 FP8 MoE front."""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip


def enabled() -> bool:
    return is_hip() and os.environ.get(
        "SGLANG_K3_AITER_MOE_PREROUTE_FP8", "0"
    ).lower() in ("1", "true")


def _ops():
    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
            kimi_k3_moe_tri_projection_fp8,
            kimi_k3_shared_down_fp8,
            supports_kimi_k3_moe_tri_projection_fp8,
            supports_kimi_k3_shared_down_fp8,
        )
    except (ImportError, ModuleNotFoundError):
        return None, None, None, None
    return (
        kimi_k3_moe_tri_projection_fp8,
        kimi_k3_shared_down_fp8,
        supports_kimi_k3_moe_tri_projection_fp8,
        supports_kimi_k3_shared_down_fp8,
    )


def tri_covered(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    if not enabled():
        return False
    _, _, supports, _ = _ops()
    return bool(
        supports is not None
        and supports(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
        )
    )


def run_tri(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op, _, _, _ = _ops()
    if op is None:
        raise RuntimeError("AITER Kimi-K3 pre-route projection is unavailable")
    return op(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    )


def shared_down_covered(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    if not enabled():
        return False
    _, _, _, supports = _ops()
    return bool(supports is not None and supports(gate_up, weight, scale))


def run_shared_down(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float,
    out: torch.Tensor,
) -> torch.Tensor:
    _, op, _, _ = _ops()
    if op is None:
        raise RuntimeError("AITER Kimi-K3 shared-down is unavailable")
    return op(
        gate_up,
        weight,
        scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        out=out,
    )


def warmup(
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float,
) -> None:
    if not enabled():
        return
    hidden = torch.zeros((1, 7168), dtype=torch.bfloat16, device=routed_weight.device)
    if not tri_covered(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    ):
        return
    _, gate_up, _ = run_tri(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    )
    out = hidden.new_empty((1, 7168))
    if shared_down_covered(gate_up, shared_down_weight, shared_down_scale):
        run_shared_down(
            gate_up,
            shared_down_weight,
            shared_down_scale,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            out=out,
        )
    torch.cuda.synchronize(hidden.device)
