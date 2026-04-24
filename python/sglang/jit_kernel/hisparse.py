from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_transfer_module() -> Module:
    return load_jit(
        "sparse_transfer",
        cuda_files=["deepseek_v4/hisparse_transfer.cuh"],
        cuda_wrappers=[("offload", "hisparse_transfer")],
    )


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    diff_map: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    transfer_tasks_src: torch.Tensor,
    transfer_tasks_dst: torch.Tensor,
    page_size: int,
    layer_id: int,
    item_size_bytes: int,
    *,
    block_size: int = 256,
    num_top_k: int = 512,
    hot_buffer_size: int = 1024,
) -> None:
    # Infer parameters if not provided
    if num_top_k <= 0:
        num_top_k = top_k_tokens.size(-1)
    if hot_buffer_size <= 0:
        hot_buffer_size = device_buffer_tokens.size(-1)

    # Validate that HOT_BUFFER_SIZE >= NUM_TOP_K
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )

    # Create empty tensors for V cache (not used in MLA)
    empty = torch.empty(0)

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        diff_map,
        req_pool_indices,
        seq_lens,
        lru_slots,
        transfer_tasks_src,
        transfer_tasks_dst,
        page_size,
        layer_id,
        item_size_bytes,
    )


def load_cache_to_device_buffer(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache_k: torch.Tensor,
    host_cache_v: torch.Tensor,
    device_buffer_k: torch.Tensor,
    device_buffer_v: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    diff_map: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    transfer_tasks_src: torch.Tensor,
    transfer_tasks_dst: torch.Tensor,
    page_size: int,
    layer_id: int,
    item_size_bytes: int,
    *,
    block_size: int = 256,
    num_top_k: int = 512,
    hot_buffer_size: int = 1024,
) -> None:
    # Infer parameters if not provided
    if num_top_k <= 0:
        num_top_k = top_k_tokens.size(-1)
    if hot_buffer_size <= 0:
        hot_buffer_size = device_buffer_tokens.size(-1)

    # Validate that HOT_BUFFER_SIZE >= NUM_TOP_K
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=False
    )

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        diff_map,
        req_pool_indices,
        seq_lens,
        lru_slots,
        transfer_tasks_src,
        transfer_tasks_dst,
        page_size,
        layer_id,
        item_size_bytes,
    )


def offload_to_host(
    gpu_ptrs: torch.Tensor,
    cpu_ptrs: torch.Tensor,
    gpu_indices: torch.Tensor,
    cpu_indices: torch.Tensor,
) -> None:
    module = _jit_sparse_transfer_module()
    module.offload(gpu_ptrs, cpu_ptrs, gpu_indices, cpu_indices)
