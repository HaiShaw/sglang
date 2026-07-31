# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""AMD FlyDSL MegaMoE adapter for the SGLang MegaMoE hooks."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

_FLYDSL_PATH_READY = False
_MORI_SHMEM_READY = False
_MEGA_MOE_INSTANCE: dict = {}


def _ensure_flydsl_on_path() -> None:
    global _FLYDSL_PATH_READY
    if _FLYDSL_PATH_READY:
        return
    path = envs.SGLANG_AMD_FLYDSL_KERNELS_PATH.get() or os.environ.get(
        "ATOM_FLYDSL_KERNELS_PATH", ""
    )
    if path and path not in sys.path:
        sys.path.insert(0, path)
    _FLYDSL_PATH_READY = True


def _import_flydsl():
    _ensure_flydsl_on_path()
    try:
        try:
            from kernels.mega_moe import MegaMoEV2

            op = MegaMoEV2
            try:
                from kernels.mega_moe import MegaMoEV2Workload

                workload = MegaMoEV2Workload
            except ImportError:
                workload = None
            is_v2 = True
        except ImportError:
            try:
                from kernels.mega_moe import MegaMoE
            except ImportError:
                from kernels.moe.mega_moe import MegaMoE
            op = MegaMoE
            workload = None
            is_v2 = False

        try:
            from tests.kernels.test_mega_moe_v2 import (
                _per_1x32_fp4_quant,
            )
        except ImportError:
            from tests.kernels.test_moe_gemm import _per_1x32_fp4_quant

        try:
            from tests.kernels.utils import gemm_common_utils as fp4_utils
        except ImportError:
            from tests.kernels.utils import fp4_utils
        from tests.utils import shuffle_weight
    except ImportError as exc:
        raise ImportError(
            "FlyDSL MegaMoE requires a pinned FlyDSL source checkout. Set "
            "SGLANG_AMD_FLYDSL_KERNELS_PATH to that checkout. "
            f"Original import error: {exc}"
        ) from exc

    return SimpleNamespace(
        op=op,
        workload=workload,
        is_v2=is_v2,
        per_1x32_fp4_quant=_per_1x32_fp4_quant,
        fp4_utils=fp4_utils,
        shuffle_weight=shuffle_weight,
    )


def _mtpr() -> int:
    mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR.get())
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr} must be a positive power of two"
        )
    return mtpr


def _decode_mtpr() -> int:
    mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR.get())
    if mtpr == 0:
        return _mtpr()
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR={mtpr} must be zero or a positive power of two"
        )
    return mtpr


def _select_mtpr(forward_batch: ForwardBatch | None) -> int:
    global_has_extend = bool(
        forward_batch is not None and forward_batch.is_extend_in_batch
    )
    use_decode = not global_has_extend
    return _decode_mtpr() if use_decode else _mtpr()


def _max_workload_tokens(forward_batch: ForwardBatch | None, local_tokens: int) -> int:
    if forward_batch is not None and forward_batch.global_max_num_tokens is not None:
        return int(forward_batch.global_max_num_tokens)
    global_tokens = get_dp_global_num_tokens()
    return max(global_tokens) if global_tokens else local_tokens


def _ep_rank_world():
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group = get_moe_ep_group().device_group
    return torch.distributed.get_rank(group), torch.distributed.get_world_size(group)


def _ensure_mori_shmem() -> None:
    global _MORI_SHMEM_READY
    if _MORI_SHMEM_READY:
        return

    import mori.shmem

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group_name = "megamoe_flydsl"
    cpu_group = get_moe_ep_group().cpu_group
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
    mori.shmem.shmem_torch_process_group_init(group_name)
    _MORI_SHMEM_READY = True


def build_mega_moe_experts_weights(layer) -> None:
    if getattr(layer, "_mega_moe_weights_built", False):
        return

    fd = _import_flydsl()
    fp4_utils = fd.fp4_utils
    fp4_view = torch.float4_e2m1fn_x2

    def scale_param(name):
        scale = getattr(layer, name, None)
        if scale is None:
            scale = getattr(layer, name + "_inv", None)
        if scale is None:
            raise AttributeError(f"FlyDSL MegaMoE weight scale {name} is missing")
        return scale

    def requant_shuffle(weight, scale, *, gate_up):
        experts, rows, packed_k = weight.shape
        k = packed_k * 2
        values = fp4_utils.mxfp4_to_f32(weight.view(fp4_view))
        scales = fp4_utils.e8m0_to_f32(scale)
        weight_f32 = (
            values.view(experts, rows, k // 32, 32)
            * scales.view(experts, rows, k // 32, 1)
        ).view(experts * rows, k)
        del values, scales
        weight_fp4, weight_scale = fd.per_1x32_fp4_quant(weight_f32)
        del weight_f32

        if gate_up:
            shuffled_weight = fp4_utils.shuffle_weight_w4(
                weight_fp4.view(experts, rows, k // 2),
                NLane=16,
                gate_up=True,
                moe_gemm=True,
            )
            shuffled_scale = fp4_utils.shuffle_scale_w4(
                weight_scale.view(experts * rows, k // 32),
                experts_cnt=experts,
                gate_up=True,
            )
        else:
            shuffled_weight = fd.shuffle_weight(weight_fp4)
            shuffled_scale = fp4_utils.e8m0_shuffle(weight_scale)
        return (
            shuffled_weight.view(torch.uint8).contiguous().view(-1),
            shuffled_scale.view(torch.uint8).contiguous().view(-1),
        )

    w13_scale = scale_param("w13_weight_scale")
    w2_scale = scale_param("w2_weight_scale")
    layer._mega_w1, layer._mega_w1_scale = requant_shuffle(
        layer.w13_weight.data, w13_scale.data, gate_up=True
    )
    layer._mega_w2, layer._mega_w2_scale = requant_shuffle(
        layer.w2_weight.data, w2_scale.data, gate_up=False
    )

    # Keep the shuffled buffers visible through the standard expert parameters.
    # EPLB discovers and migrates expert weights via DeepseekV2MoE.get_moe_weights(),
    # which only returns named parameters and requires expert-major dim 0. These
    # views share storage with the tensors consumed by MegaMoE, so migration
    # updates the live kernel weights without retaining the original checkpoint
    # layout or doubling memory.
    experts = layer.w13_weight.shape[0]
    layer.w13_weight.data = layer._mega_w1.view(experts, -1)
    w13_scale.data = layer._mega_w1_scale.view(experts, -1)
    layer.w2_weight.data = layer._mega_w2.view(experts, -1)
    w2_scale.data = layer._mega_w2_scale.view(experts, -1)
    layer._mega_moe_weights_built = True


def _get_or_build_mega_moe(
    layer,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str,
    mtpr: int | None = None,
):
    _ensure_mori_shmem()
    fd = _import_flydsl()
    rank, world = _ep_rank_world()
    mtpr = _mtpr() if mtpr is None else int(mtpr)
    key = (
        fd.is_v2,
        rank,
        world,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        mtpr,
    )
    mega = _MEGA_MOE_INSTANCE.get(key)
    if mega is not None:
        return mega

    common = {
        "rank": rank,
        "world_size": world,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "quant": quant,
        "w1": layer._mega_w1,
        "w1_scale": layer._mega_w1_scale,
        "w2": layer._mega_w2,
        "w2_scale": layer._mega_w2_scale,
        "max_tok_per_rank": mtpr,
    }
    if fd.is_v2:
        mega = fd.op(**common)
    else:
        mega = fd.op(
            **common,
            gate_mode="interleave",
            gemm2_tile_m=32,
            gemm2_tile_n=128,
            gemm2_tile_k=256,
        )
    _MEGA_MOE_INSTANCE[key] = mega
    return mega


def _swap_layer_weights(mega, layer) -> None:
    mega._s1_w1 = layer._mega_w1
    mega._s1_w1_scale = layer._mega_w1_scale
    mega.w2 = layer._mega_w2
    mega.w2_scale = layer._mega_w2_scale


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        max_tokens = max(global_num_tokens)
    else:
        max_tokens = hidden_states.shape[0]
    return max_tokens <= _mtpr()


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None = None,
    input_ids_global: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    overlap = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )
    if overlap:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        stream_context = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        stream_context = nullcontext()

    with stream_context:
        output = _run_mega_routed(moe, hidden_states, forward_batch, input_ids_global)
    if overlap:
        current_stream.wait_stream(moe.alt_stream)
    if shared_output is not None:
        output.add_(shared_output)
    return output


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None,
    input_ids_global: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = moe.config.hidden_size
    topk = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    experts = moe.experts.num_experts

    if num_tokens:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id
            ),
            **topk_kwargs,
        )
        x_in = hidden_states
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights.to(torch.float32)
    else:
        x_in = hidden_states.new_zeros((1, hidden_size))
        topk_ids = (
            torch.arange(topk, device=hidden_states.device, dtype=torch.int32) % experts
        ).unsqueeze(0)
        topk_weights = hidden_states.new_zeros((1, topk), dtype=torch.float32)

    fd = _import_flydsl()
    selected_mtpr = _select_mtpr(forward_batch)
    assert x_in.shape[0] <= selected_mtpr, (
        f"FlyDSL MegaMoE local tokens {x_in.shape[0]} exceed selected "
        f"MTPR {selected_mtpr}"
    )
    quant = envs.SGLANG_AMD_FLYDSL_MEGA_QUANT.get() or "a8w4"
    if fd.is_v2 and _decode_mtpr() != _mtpr():
        for capacity in (_mtpr(), _decode_mtpr()):
            _get_or_build_mega_moe(
                moe.experts,
                model_dim=hidden_size,
                inter_dim=moe.config.moe_intermediate_size,
                experts=experts,
                topk=topk,
                quant=quant,
                mtpr=capacity,
            )
    mega = _get_or_build_mega_moe(
        moe.experts,
        model_dim=hidden_size,
        inter_dim=moe.config.moe_intermediate_size,
        experts=experts,
        topk=topk,
        quant=quant,
        mtpr=selected_mtpr,
    )
    _swap_layer_weights(mega, moe.experts)
    global_max_tokens = _max_workload_tokens(forward_batch, x_in.shape[0])
    if fd.is_v2 and fd.workload is not None:
        forward_mode = (
            str(forward_batch.forward_mode) if forward_batch is not None else "eager"
        )
        graph_bucket = x_in.shape[0] if get_is_capture_mode() else None
        workload = fd.workload.from_max_tokens(
            global_max_tokens,
            selected_mtpr,
            forward_mode=forward_mode,
            graph_bucket=graph_bucket,
        )
        output = mega.forward(
            x_in,
            topk_weights,
            topk_ids,
            workload=workload,
            slice_output=False,
        )[:num_tokens]
    elif fd.is_v2:
        output = mega.forward(x_in, topk_weights, topk_ids, slice_output=False)[
            :num_tokens
        ]
    else:
        output = mega.forward(x_in, topk_weights, topk_ids, slice_output=False)[
            :num_tokens
        ]

    from sglang.srt.models.deepseek_common.utils import _use_aiter

    if not (moe.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter):
        output.mul_(moe.routed_scaling_factor)
    return output
