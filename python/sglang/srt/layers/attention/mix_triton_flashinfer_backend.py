from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


#======== flashinfer related dependencies ===========
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from sglang.global_config import global_config
from typing import TYPE_CHECKING, List, Optional
import triton
import triton.language as tl
import os 
from enum import Enum, auto
from dataclasses import dataclass


class FlashInferWrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()
    
    
@dataclass
class FlashInferDecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


# ATTENTION: This backend mix triton extend and flashinfer decode
class MixTritonFlashInferAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        #=========triton extend init==============
        # Lazy import of trition op to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.extend_attention_fwd = extend_attention_fwd

        if model_runner.server_args.enable_dp_attention:
            self.num_head = model_runner.model_config.num_attention_heads
        else:
            self.num_head = (
                model_runner.model_config.num_attention_heads // model_runner.tp_size
            )

        self.num_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.forward_metadata = None

        self.cuda_graph_max_seq_len = model_runner.model_config.context_len

        self.device = model_runner.device
        
        #===========flashinfer decode init============
        # currently only group size is conditioned in use_tensor_core function
        self.flashinfer_decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // model_runner.tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                model_runner.tp_size
            ),
        )
        
        # sliding window is supported in flashinfer backend
        # sliding window and cross attention wrappers can't coexit in flashinfer
        # total number of decode wrappers is <= 2
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.flashinfer_num_wrappers = 2
            self.flashinfer_dispatch_reason = FlashInferWrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            # encoder+decoder model requires cross attention 
            self.flashinfer_num_wrappers = 2
            self.flashinfer_dispatch_reason = FlashInferWrapperDispatch.CROSS_ATTENTION
        else:
            self.flashinfer_num_wrappers = 1
            self.flashinfer_dispatch_reason = None
            
        # workspace assigned to flashinfer decoder
        self.flashinfer_workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device=model_runner.device,
        )
        
        self.flashinfer_max_bs = model_runner.req_to_token_pool.size
        # kv indptr for each wrapper
        self.flashinfer_kv_indptr = [
            torch.zeros((self.flashinfer_max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.flashinfer_num_wrappers)
        ]
        # last page len are all initialized to 1, page size is hardcoded to 1 
        self.flashinfer_kv_last_page_len = torch.ones(
            (self.flashinfer_max_bs,), dtype=torch.int32, device=model_runner.device
        )
        # q indptr for each wrapper
        self.flashinfer_qo_indptr = [
            torch.zeros((self.flashinfer_max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.flashinfer_num_wrappers)
        ]
        
        
        # flashinfer decode function
        # will be overwritten if cuda_graph is enabled
        self.flashinfer_decode_wrappers = []
        for _ in range(self.flashinfer_num_wrappers):
            self.flashinfer_decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.flashinfer_workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.flashinfer_decode_use_tensor_cores,
                )
            )

        # one index updater is used for all decode wrappers
        # all decoder indices are updated each time when the decode updater is called
        self.flashinfer_indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)
        
        # flashinfer cuda graph 
        self.flashinfer_max_context_len = model_runner.model_config.context_len
        self.flashinfer_decode_cuda_graph_metadata = {} # wrapper used for each bs, but here only update_single_wrapper is used

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        if forward_batch.forward_mode.is_decode():
            self.flashinfer_indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.flashinfer_decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = FlashInferDecodeMetadata(self.flashinfer_decode_wrappers)
        else:
            attn_logits = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
            self.forward_metadata = attn_logits, max_extend_len
        
    def init_cuda_graph_state(self, max_bs: int):
        self.flashinfer_init_cuda_graph_state(max_bs)
        
    def flashinfer_init_cuda_graph_state(self, max_bs: int):
        cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.flashinfer_max_context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        
        # cuda graph init for each flashinfer wrapper
        self.flashinfer_cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.flashinfer_num_wrappers - 1)
        ]
        
    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        self.flashinfer_init_forward_metadata_capture_cuda_graph(bs, num_tokens, req_pool_indices, seq_lens, encoder_lens, forward_mode, spec_info)
        
    def flashinfer_init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        # cuda graph is captured for each batch size (bs)
        decode_wrappers = []
        for i in range(self.flashinfer_num_wrappers):
            decode_wrappers.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.flashinfer_decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.flashinfer_kv_indptr[i][: num_tokens + 1],
                        paged_kv_indices_buffer=self.flashinfer_cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.flashinfer_kv_last_page_len[
                            :num_tokens
                        ],
                    )
                )
        seq_lens_sum = seq_lens.sum().item()
        self.flashinfer_indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            decode_wrappers=decode_wrappers,
            encoder_lens=encoder_lens,
            spec_info=spec_info,
        )
        self.flashinfer_decode_cuda_graph_metadata[bs] = decode_wrappers
        self.forward_metadata = FlashInferDecodeMetadata(decode_wrappers)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        self.flashinfer_init_forward_metadata_replay_cuda_graph(bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens, forward_mode, spec_info)
        
    def flashinfer_init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        self.flashinfer_indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_sum,
            decode_wrappers=self.flashinfer_decode_cuda_graph_metadata[bs],
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
            spec_info=spec_info,
        )
        
        
    def get_cuda_graph_seq_len_fill_value(self):
        return self.flashinfer_get_cuda_graph_seq_len_fill_value()
    
    def flashinfer_get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o
    
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        return self.flashinfer_forward_decode(q, k, v, layer, forward_batch, save_kv_cache)
    
    
    def flashinfer_forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # out_cache_loc: kv cache tensor VRAM slot indexing table
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
    
    def _get_wrapper_idx(self, layer: RadixAttention):
        # this returns either 0, -1 or 1
        if self.flashinfer_num_wrappers == 1:
            return 0
        if self.flashinfer_dispatch_reason == FlashInferWrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.flashinfer_dispatch_reason == FlashInferWrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.flashinfer_dispatch_reason}")
    
class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.flashinfer_kv_indptr
        self.kv_last_page_len = attn_backend.flashinfer_kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        
        # Dispatch the update function
        if self.attn_backend.flashinfer_dispatch_reason == FlashInferWrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.flashinfer_dispatch_reason == FlashInferWrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.flashinfer_num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
        )
        
    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.minimum(  # TODO: replace this with clamp
                    seq_lens,
                    torch.tensor(self.sliding_window_size + 1),
                )
                paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                kv_start_idx_tmp = None

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
            )
    
    
    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInfo],
    ):
        
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
            )  
    

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInfo],
    ):
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            bs, kv_indices, kv_indptr = spec_info.generate_attn_arg_decode(
                req_pool_indices,
                paged_kernel_lens,
                self.req_to_token,
            )

        wrapper.end_forward()
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
        )



@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
        

def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # Determine based on dtype and GQA group size
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size > 4
    else:
        return False