"""TP4 tiny-row all-reduce/post using AITER's registered peer buffers."""

from typing import Tuple

import torch
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.jit.core import AITER_CSRC_DIR

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _all_reduce_mhc_module():
    return load_jit(
        "all_reduce_mhc_hip",
        cuda_files=["distributed/all_reduce_mhc_hip.cuh"],
        cuda_wrappers=[("run", "all_reduce_mhc_hip::AllReduceMhcPostKernel::run")],
        # AITER's CustomAllreduce, whose peer buffers and signals the kernel runs on
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        # no FMA contraction: the unfused all-reduce + hc_post rounds every multiply and add
        extra_cuda_cflags=["-ffp-contract=off"],
    )


def all_reduce_mhc_post(
    input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    communicator: CustomAllreduce,
) -> torch.Tensor:
    """TP4 all-reduce of input fused with hc_post onto residual; the kernel checks the
    shapes (1-64 contiguous rows of DeepSeek-V4.1's hidden size)."""
    assert communicator.world_size == 4 and not communicator.disabled
    for name, tensor, dtype in (
        ("input", input, torch.bfloat16),
        ("residual", residual, torch.bfloat16),
        ("post", post, torch.float32),
        ("comb", comb, torch.float32),
    ):
        if tensor.device != communicator.device:
            raise RuntimeError(
                f"{name} must be on {communicator.device}, got {tensor.device}"
            )
        if tensor.dtype != dtype:
            raise RuntimeError(f"{name} must be {dtype}, got {tensor.dtype}")
    capturing = torch.cuda.is_current_stream_capturing()
    # capture() registers the peer addresses when the enclosing graph scope exits.
    assert not capturing or (
        communicator._IS_CAPTURING and communicator.enable_register_for_capturing
    )
    pool = communicator._pool["input"]
    output = torch.empty_like(residual)
    _all_reduce_mhc_module().run(
        communicator._ptr,
        input,
        output,
        residual,
        post,
        comb,
        0 if capturing else pool.data_ptr,
        0 if capturing else pool.max_size,
    )
    return output


@cache_once
def _all_reduce_mhc_stats_module():
    return load_jit(
        "all_reduce_mhc_stats_hip",
        cuda_files=["distributed/all_reduce_mhc_stats_hip.cuh"],
        cuda_wrappers=[
            ("run", "all_reduce_mhc_stats_hip::AllReduceMhcPostStatsKernel::run")
        ],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-ffp-contract=off"],
    )


# slices of the boundary statistics the fused kernel writes: one per 512-wide hidden block
ALL_REDUCE_MHC_STATS_SLICES = 10


def all_reduce_mhc_post_stats(
    input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre_prev: torch.Tensor,
    hc_fn: torch.Tensor,
    communicator: CustomAllreduce,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """all_reduce_mhc_post plus the next mHC boundary without hc_post: returns (residual_out,
    y = sum_k pre_prev[k] * copy_k, part_mix [slices, M, MIX], part_sq [slices, M])."""
    assert communicator.world_size == 4 and not communicator.disabled
    capturing = torch.cuda.is_current_stream_capturing()
    assert not capturing or (
        communicator._IS_CAPTURING and communicator.enable_register_for_capturing
    )
    m = residual.shape[0]
    dev = residual.device
    pool = communicator._pool["input"]
    output = torch.empty_like(residual)
    y = torch.empty_like(input)
    part_mix = torch.empty(
        (ALL_REDUCE_MHC_STATS_SLICES, m, hc_fn.shape[0]), dtype=torch.float32, device=dev
    )
    part_sq = torch.empty(
        (ALL_REDUCE_MHC_STATS_SLICES, m), dtype=torch.float32, device=dev
    )
    _all_reduce_mhc_stats_module().run(
        communicator._ptr,
        input,
        output,
        residual,
        post,
        comb,
        pre_prev.contiguous(),
        hc_fn,
        y,
        part_mix,
        part_sq,
        0 if capturing else pool.data_ptr,
        0 if capturing else pool.max_size,
    )
    return output, y, part_mix, part_sq
