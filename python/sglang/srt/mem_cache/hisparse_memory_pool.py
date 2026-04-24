# mapping on device memory, host memory and memory allocator

import weakref
from typing import Optional

import torch
from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseekv4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)


class DeepSeekV4SingleKVPoolHost:
    # simplified host KV pool for hisparse C4 device pool

    def __init__(
        self,
        device_pool: HiSparseC4DevicePool,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):

        assert host_size > 0, "Host size must be specified and greater than 0"
        # use page size 1 for simplicity
        assert page_size == 1, "Host page size must be 1 for DeepSeekV4SingleKVPoolHost"

        self.device_pool = device_pool
        self.size = host_size
        self.page_size = page_size
        self.num_pages = (self.size + self.page_size - 1) // self.page_size
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.layer_num = device_pool.layer_num
        self.kv_cache_total_dim = device_pool.kv_cache_total_dim

        self.kv_buffer = self.init_kv_buffer()
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.clear()

    def clear(self):
        self.free_slots = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device="cpu"
        )

    def init_kv_buffer(self):
        dims = (self.layer_num, self.size + self.page_size, self.kv_cache_total_dim)
        host_pool = torch.empty(dims, dtype=self.dtype, device=self.device)
        assert self.pin_memory, "DeepSeekV4SingleKVPoolHost requires pin_memory=True"
        if self.pin_memory:
            torch.cuda.cudart().cudaHostRegister(
                host_pool.data_ptr(), host_pool.numel() * host_pool.element_size(), 0
            )
        return host_pool

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices):
        # todo: direct io backend
        # FIXME, page padding to be handled in the custom op
        transfer_kv_all_layer_mla(
            src_layers=device_pool.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=device_indices,
            dst_indices=host_indices,
            item_size=self.kv_cache_total_dim,
            num_layers=self.layer_num,
        )

    def testing_backup_to_device_all_layer(
        self, device_pool, host_indices, device_indices
    ):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=device_pool.data_ptrs,
            src_indices=host_indices,
            dst_indices=device_indices,
            item_size=self.kv_cache_total_dim,
            num_layers=self.layer_num,
        )

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        logical_attn_allocator: BaseTokenToKVPoolAllocator,
    ):
        assert isinstance(logical_attn_allocator._kvcache, DeepSeekV4TokenToKVPool)
        assert isinstance(
            logical_attn_allocator._kvcache.c4_kv_pool, HiSparseC4DevicePool
        )
        self.compress_ratio = 4

        self.hisparse_kvcache = logical_attn_allocator._kvcache.c4_kv_pool
        self._size_full = logical_attn_allocator.size_full
        self._size_hisparse = self.hisparse_kvcache.size

        self.dtype = self.hisparse_kvcache.dtype
        self.device = self.hisparse_kvcache.device
        self.page_size = self.hisparse_kvcache.page_size

        self.logical_attn_allocator = logical_attn_allocator
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            self.hisparse_kvcache,
            logical_attn_allocator.need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_hisparse + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.need_sort = logical_attn_allocator.need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self.hisparse_kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    def full_available_size(self):
        return self.logical_attn_allocator.full_available_size()

    def swa_available_size(self):
        return self.logical_attn_allocator.swa_available_size()

    def free_swa(self, free_indices: torch.Tensor):
        self.logical_attn_allocator.free_swa(free_indices)

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "Page size = 1 is not supported in HiSparse allocator"
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        # disable free group mechanism for device buffer free
        self.hisparse_attn_allocator.is_not_in_free_group = True
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0])

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return (last_locs - 3) // self.compress_ratio

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        hisparse_last_locs = (
            self.hisparse_kvcache._translate_loc_from_compressed_to_hisparse_device(
                self.get_last_loc_compressed(last_locs)
            )
        )
        return hisparse_last_locs

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1
        num_tokens = extend_num_tokens + len(seq_lens) * self.page_size

        if num_tokens > self.available_size():
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens // self.compress_ratio,
            prefix_lens_cpu // self.compress_ratio,
            seq_lens // self.compress_ratio,
            seq_lens_cpu // self.compress_ratio,
            hisparse_last_loc,
            len(compressed_logical_indices),
        )

        assert logical_indices is not None, "Logical allocation failed in alloc_extend"
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"

        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
            hisparse_indices
        )
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        return logical_indices

    def alloc_decode_regular(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        active_reqs = seq_lens % self.compress_ratio == 0
        hisparse_indices = self.hisparse_attn_allocator.alloc_decode(
            seq_lens[active_reqs] // self.compress_ratio,
            seq_lens_cpu[active_reqs.cpu()] // self.compress_ratio,
            hisparse_last_loc[active_reqs],
        )

        if logical_indices is None or hisparse_indices is None:
            return None

        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )

        assert len(compressed_logical_indices) == len(
            hisparse_indices
        ), "Mismatch in allocated indices length in alloc_decode"
        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
            hisparse_indices
        )

        return logical_indices

    def free_compressed(self, compressed_indices: torch.Tensor):
        hisparse_indices = (
            self.hisparse_kvcache.translate_loc_from_compressed_to_hisparse_device(
                compressed_indices
            )
        )
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[compressed_indices] = 0

    def free_hisparse(self, free_indices: torch.Tensor):
        compressed_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(free_indices)
        )
        self.free_compressed(compressed_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
            # free activities will be associated with device buffers
            # self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert (
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )
