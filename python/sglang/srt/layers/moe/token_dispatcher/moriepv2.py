from __future__ import annotations

"""MORI EPv2 intranode dispatcher using cco-LSA and FlyDSL kernels."""

import logging
import os
from enum import Enum, auto
from functools import lru_cache
from typing import NamedTuple

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, is_tbo_enabled
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)
_logged_graph_caps = set()


class MoriEPv2NormalDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    origin_topk_ids: torch.Tensor
    origin_topk_weights: torch.Tensor
    out_dtype: torch.dtype

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


assert isinstance(MoriEPv2NormalDispatchOutput, DispatchOutput)


@lru_cache(maxsize=4)
def _init_cco_communicator(group, instance_id: int):
    from mori.cco import Communicator

    parallel = get_parallel()
    world_size = parallel.moe_ep_size
    rank = parallel.moe_ep_rank
    if world_size > 8:
        raise ValueError(
            f"MORI EPv2 is currently intranode-only (world_size<=8); got {world_size}"
        )

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = group.broadcast_object(uid, src=0)
    per_rank_vmm_gb = get_int_env_var("SGLANG_MORI_EPV2_PER_RANK_VMM_GB", 4)
    if per_rank_vmm_gb <= 0:
        raise ValueError("SGLANG_MORI_EPV2_PER_RANK_VMM_GB must be positive")
    comm = Communicator.init(
        world_size,
        rank,
        uid,
        per_rank_vmm=per_rank_vmm_gb * (1 << 30),
    )
    logger.info(
        "[MORI EPv2 init] cco communicator world=%d rank=%d instance=%d "
        "per_rank_vmm_gb=%d",
        world_size,
        rank,
        instance_id,
        per_rank_vmm_gb,
    )
    return comm


@lru_cache(maxsize=4)
def init_mori_epv2_op(
    group,
    router_topk: int,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype,
    max_tokens_per_rank: int,
    instance_id: int = 0,
    max_total_recv_tokens: int = 0,
):
    from mori.ops.dispatch_combine_v2 import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )

    if params_dtype != torch.bfloat16:
        raise ValueError(
            "The first MORI EPv2 serving baseline requires BF16 dispatch/combine; "
            f"got params_dtype={params_dtype}"
        )

    parallel = get_parallel()
    world_size = parallel.moe_ep_size
    rank = parallel.moe_ep_rank
    comm = _init_cco_communicator(group, instance_id)
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_size,
        max_num_inp_token_per_rank=max_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        data_type=torch.bfloat16,
        combine_mode="gather",
        max_total_recv_tokens=max_total_recv_tokens,
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()
    logger.info(
        "[MORI EPv2 init] world=%d rank=%d hidden=%d experts=%d local_experts=%d "
        "topk=%d max_tokens=%d recv_cap=%d schedule=%s",
        world_size,
        rank,
        hidden_size,
        num_experts,
        num_local_experts,
        router_topk,
        max_tokens_per_rank,
        cfg.effective_max_recv,
        cfg.schedule,
    )
    return op


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class MoriEPv2Dispatcher(BaseDispatcher):
    """Synchronous non-TBO MORI EPv2 dispatcher."""

    def __init__(
        self,
        group,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        deepep_mode: DeepEPMode = DeepEPMode.NORMAL,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        instance_id: int = 0,
    ):
        super().__init__()
        del permute_fusion, async_finish, return_recv_hook
        if is_tbo_enabled():
            raise NotImplementedError(
                "MORI EPv2 TBO is gated on non-TBO correctness and serving results"
            )
        if not deepep_mode.enable_normal() or deepep_mode.enable_low_latency():
            raise ValueError("MORI EPv2 currently supports normal mode only")
        try:
            from mori.cco import Communicator  # noqa: F401
            from mori.ops.dispatch_combine_v2 import EpDispatchCombineOp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MORI EPv2 requires a MORI build with cco and dispatch_combine_v2"
            ) from exc

        # EPv2 forwards exactly top-k routed experts and has no fake expert slot.
        os.environ.setdefault("AITER_FLYDSL_EP_NO_FAKE_EXPERT", "1")
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.instance_id = instance_id
        self.max_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )
        self._op = None
        self._stage = _Stage.INITIAL

        parallel = get_parallel()
        self.expert_mask_gpu = torch.zeros(
            num_experts,
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        start = parallel.moe_ep_rank * num_local_experts
        self.expert_mask_gpu[start : start + num_local_experts] = 1
        # CCO communicator/window creation is collective and allocates symmetric
        # VMM resources; it must finish during model construction, before SGLang
        # enters CUDA graph capture for its first decode forward.
        self._op = init_mori_epv2_op(
            self.group,
            self.router_topk,
            self.num_experts,
            self.num_local_experts,
            self.hidden_size,
            self.params_dtype,
            self.max_tokens_per_rank,
            self.instance_id,
            get_int_env_var("SGLANG_MORI_EPV2_PREALLOC_MAX_RECV_TOKENS", 0),
        )
        # Decode graph tiers need rank-consistent logical receive caps. Compile
        # their kernels before capture while reusing this op's one full physical
        # CCO arena; CCO does not support multiple simultaneous arena windows.
        graph_cap_max = get_int_env_var("SGLANG_MORI_EPV2_GRAPH_RECV_CAP_MAX", 8192)
        graph_cap = 32
        while graph_cap <= min(graph_cap_max, self._op.cfg.effective_max_recv):
            self._op.prepare_recv_cap(graph_cap)
            graph_cap *= 2

    @property
    def op(self):
        if self._op is None:
            self._op = init_mori_epv2_op(
                self.group,
                self.router_topk,
                self.num_experts,
                self.num_local_experts,
                self.hidden_size,
                self.params_dtype,
                self.max_tokens_per_rank,
                self.instance_id,
            )
        return self._op

    def _select_recv_cap(self):
        try:
            from sglang.srt.model_executor.runner import get_is_capture_mode
        except ImportError:
            return self.op.cfg.effective_max_recv
        if not get_is_capture_mode():
            return self.op.cfg.effective_max_recv
        try:
            from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
        except ImportError:
            return self.op.cfg.effective_max_recv
        dp_global = get_dp_global_num_tokens()
        if dp_global is None or len(dp_global) <= 1:
            return self.op.cfg.effective_max_recv
        global_capacity = max(int(value) for value in dp_global) * len(dp_global)
        if global_capacity <= 0:
            return self.op.cfg.effective_max_recv
        recv_cap = max(32, 1 << (global_capacity - 1).bit_length())
        if recv_cap > self.op.cfg.effective_max_recv:
            return self.op.cfg.effective_max_recv
        key = (global_capacity, recv_cap)
        if get_parallel().world_rank == 0 and key not in _logged_graph_caps:
            _logged_graph_caps.add(key)
            logger.warning(
                "[MORI EPv2 graph cap] global_capacity=%d recv_cap=%d "
                "physical_cap=%d",
                global_capacity,
                recv_cap,
                self.op.cfg.effective_max_recv,
            )
        return recv_cap

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        # Keep the first matched baseline at BF16 transport. Aiter still applies
        # the model's A8W4 expert GEMM quantization after dispatch.

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> None:
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "MORI EPv2 BF16 baseline received "
                f"hidden_states.dtype={hidden_states.dtype}"
            )
        self._num_tokens = hidden_states.shape[0]
        self._dispatch_intermediate_state = (
            hidden_states,
            topk_output.topk_weights.to(torch.float32),
            topk_output.topk_ids.to(torch.int32),
        )

    def dispatch_b(self) -> DispatchOutput:
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        hidden_states, topk_weights, topk_ids = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        recv_cap = self._select_recv_cap()
        (
            recv_hidden,
            recv_weights,
            recv_scales,
            recv_indices,
            total_recv,
            routing,
        ) = self.op.dispatch(
            hidden_states,
            topk_weights,
            None,
            topk_ids,
            return_routing=True,
            recv_cap=recv_cap,
            clone_routing=False,
        )
        self._routing = routing
        self._recv_topk_ids = recv_indices
        return MoriEPv2NormalDispatchOutput(
            hidden_states=recv_hidden,
            hidden_states_scale=recv_scales,
            topk_ids=recv_indices,
            topk_weights=recv_weights,
            num_recv_tokens_per_expert=total_recv,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=hidden_states.dtype,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()[: self._num_tokens]

    def combine_a(self, combine_input: CombineInput) -> None:
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        self._combine_intermediate_state = tuple(combine_input)

    def combine_b(self) -> torch.Tensor:
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        hidden_states, _topk_ids, _topk_weights = self._combine_intermediate_state
        del self._combine_intermediate_state
        out, _ = self.op.combine(
            hidden_states,
            None,
            self._recv_topk_ids,
            routing=self._routing,
        )
        if not torch.cuda.is_current_stream_capturing():
            del self._routing
        return out

    def _update_stage(self, old_stage: _Stage, new_stage: _Stage) -> None:
        assert self._stage == old_stage, f"stage {self._stage} != expected {old_stage}"
        self._stage = new_stage
