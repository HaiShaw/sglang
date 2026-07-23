from __future__ import annotations

"""FlyDSL intranode EP dispatcher backed by aiter's FlyDSL all-to-all op."""

import logging
import os
from enum import Enum, auto
from functools import lru_cache
from typing import NamedTuple, Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_hip

logger = logging.getLogger(__name__)

FP8_BLOCK_SIZE = 128
MXFP4_BLOCK_SIZE = 32

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


class DispatchDtype(Enum):
    bf16 = "bfloat16"
    fp8 = "float8"
    fp4 = "mxfp4"


class CombineDtype(Enum):
    bf16 = "bfloat16"
    fp8_direct_cast = "float8_direct_cast"


@lru_cache(maxsize=4)
def init_flydsl_op(
    group,
    router_topk,
    num_experts,
    num_local_experts,
    hidden_size,
    params_dtype,
    num_max_dispatch_tokens_per_rank,
    instance_id=0,
    dispatch_dtype=DispatchDtype.bf16,
    combine_dtype=CombineDtype.bf16,
):
    """Initialize one SGLang-vendored FlyDSL dispatch/combine op per config."""
    import mori.shmem as ms
    from sglang.kernels.third_party.flydsl_a2a import (
        FlyDSLDispatchCombineConfig,
        FlyDSLDispatchCombineIntraNodeOp,
    )

    world_size = get_parallel().moe_ep_size
    rank = get_parallel().moe_ep_rank
    if world_size > 8:
        raise ValueError(
            f"FlyDSL a2a is intranode-only (world_size<=8); got {world_size}"
        )

    group_name = "mori"
    try:
        torch._C._distributed_c10d._register_process_group(
            group_name, group.cpu_group
        )
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
        logger.info("[FlyDSL init] process group already registered: %s", exc)
    else:
        ms.shmem_torch_process_group_init(group_name)

    scale_dim = 0
    scale_type_size = 0
    if dispatch_dtype == DispatchDtype.fp8:
        scale_dim = hidden_size // FP8_BLOCK_SIZE
        scale_type_size = torch.float32.itemsize
    elif dispatch_dtype == DispatchDtype.fp4:
        scale_dim = hidden_size // MXFP4_BLOCK_SIZE
        scale_type_size = torch.float8_e8m0fnu.itemsize

    quant_type = (
        "fp8_direct_cast"
        if combine_dtype == CombineDtype.fp8_direct_cast
        else "none"
    )
    logger.info(
        "[FlyDSL init] world=%d rank=%d hidden=%d max_tokens=%d "
        "local_experts=%d topk=%d dispatch=%s combine=%s",
        world_size,
        rank,
        hidden_size,
        num_max_dispatch_tokens_per_rank,
        num_local_experts,
        router_topk,
        dispatch_dtype,
        combine_dtype,
    )
    return FlyDSLDispatchCombineIntraNodeOp(
        FlyDSLDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_size,
            max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=router_topk,
            # Allocate for the largest external row type. dispatch() still
            # specializes on its launch-time bf16/fp8/fp4 dtype.
            data_type=params_dtype,
            max_token_type_size=params_dtype.itemsize,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            quant_type=quant_type,
            enable_std_moe=False,
            max_total_recv_tokens=get_int_env_var(
                "SGLANG_FLYDSL_PREALLOC_MAX_RECV_TOKENS", 0
            ),
        )
    )


class FlyDSLEPNormalDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    origin_topk_ids: torch.Tensor
    origin_topk_weights: torch.Tensor
    out_dtype: torch.dtype

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


assert isinstance(FlyDSLEPNormalDispatchOutput, DispatchOutput)


class FlyDSLEPNormalCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


assert isinstance(FlyDSLEPNormalCombineInput, CombineInput)


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class FlyDSLEPDispatcher(BaseDispatcher):
    """Plain token-major FlyDSL all-to-all for intranode expert parallelism."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        instance_id: int = 0,
    ):
        super().__init__()
        try:
            import aiter  # noqa: F401
            import flydsl  # noqa: F401
            import mori  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FlyDSL EP requires the aiter, flydsl, and mori packages"
            ) from exc

        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.instance_id = instance_id
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_FLYDSL_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )

        self.dispatch_dtype = DispatchDtype.bf16
        self.combine_dtype = CombineDtype.bf16
        self._flydsl_op = None
        self._stage = _Stage.INITIAL

        self.fp8_quant_func = None
        self.fp4_quant_func = None
        self.expert_mask_gpu = None
        if _use_aiter:
            from aiter import QuantType, get_hip_quant

            self.fp8_quant_func = get_hip_quant(QuantType.per_1x128)
            self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)
            if num_experts is not None and num_local_experts is not None:
                ep_rank = get_parallel().moe_ep_rank
                self.expert_mask_gpu = torch.zeros(
                    num_experts,
                    device=torch.cuda.current_device(),
                    dtype=torch.int32,
                )
                start = ep_rank * num_local_experts
                self.expert_mask_gpu[start : start + num_local_experts] = 1

    @property
    def flydsl_op(self):
        if self._flydsl_op is None:
            self._apply_dtype_overrides()
            self._flydsl_op = init_flydsl_op(
                self.group,
                self.router_topk,
                self.num_experts,
                self.num_local_experts,
                self.hidden_size,
                self.params_dtype,
                self.num_max_dispatch_tokens_per_rank,
                self.instance_id,
                self.dispatch_dtype,
                self.combine_dtype,
            )
        return self._flydsl_op

    def _apply_dtype_overrides(self):
        dispatch = os.environ.get("SGLANG_FLYDSL_DISPATCH_DTYPE", "").lower()
        if dispatch == "fp8":
            self.dispatch_dtype = DispatchDtype.fp8
        elif dispatch == "fp4":
            self.dispatch_dtype = DispatchDtype.fp4
        elif dispatch == "bf16":
            self.dispatch_dtype = DispatchDtype.bf16

        combine = os.environ.get("SGLANG_FLYDSL_COMBINE_DTYPE", "").lower()
        if combine == "fp8_direct_cast":
            self.combine_dtype = CombineDtype.fp8_direct_cast
        elif combine == "bf16":
            self.combine_dtype = CombineDtype.bf16

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        weight_dtype = quant_config.get("weight_dtype")
        if weight_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.dispatch_dtype = DispatchDtype.fp8
        elif weight_dtype == torch.float4_e2m1fn_x2:
            self.dispatch_dtype = DispatchDtype.fp4
        else:
            self.dispatch_dtype = DispatchDtype.bf16
        self.combine_dtype = CombineDtype.bf16
        self._apply_dtype_overrides()

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        return self.dispatch_b()

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        self._num_tokens = hidden_states.shape[0]
        self._op_cur_tok = hidden_states.shape[0]
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        output_dtype = hidden_states.dtype
        scale = None
        device = hidden_states.device

        if self.dispatch_dtype == DispatchDtype.fp8 and self.fp8_quant_func:
            from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

            if self._num_tokens:
                hidden_states, scale = self.fp8_quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size), dtype=fp8_dtype, device=device
                )
                scale = torch.empty(
                    (0, self.hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=device,
                )
        elif self.dispatch_dtype == DispatchDtype.fp4 and self.fp4_quant_func:
            if self._num_tokens:
                hidden_states, scale = self.fp4_quant_func(
                    hidden_states, shuffle=False
                )
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=device,
                )
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=device,
                )

        self._dispatch_intermediate_state = (
            hidden_states,
            topk_weights,
            topk_ids,
            scale,
            output_dtype,
        )

    def dispatch_b(self) -> DispatchOutput:
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        hidden_states, topk_weights, topk_ids, scale, output_dtype = (
            self._dispatch_intermediate_state
        )
        del self._dispatch_intermediate_state

        op = self.flydsl_op
        recv_cap = self._resolve_dynamic_recv_cap(op.cfg.effective_max_recv)
        self._op_recv_cap = recv_cap
        out_tok, out_wts, out_scales, out_idx, total_recv = op.dispatch(
            hidden_states,
            topk_weights.to(torch.float32),
            scale,
            topk_ids,
            recv_cap=recv_cap,
        )
        self._recv_topk_ids = out_idx
        return FlyDSLEPNormalDispatchOutput(
            hidden_states=out_tok,
            hidden_states_scale=out_scales,
            topk_ids=out_idx,
            topk_weights=out_wts,
            num_recv_tokens_per_expert=total_recv,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()[: self._num_tokens]

    def combine_a(self, combine_input: CombineInput):
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        self._combine_intermediate_state = tuple(combine_input)

    def combine_b(self) -> torch.Tensor:
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        hidden_states, _topk_ids, _topk_weights = self._combine_intermediate_state
        del self._combine_intermediate_state
        out_tok, _ = self.flydsl_op.combine(
            hidden_states,
            None,
            self._recv_topk_ids,
            cur_tok=self._op_cur_tok,
            recv_cap=self._op_recv_cap,
        )
        return out_tok

    def _resolve_dynamic_recv_cap(self, physical_cap: int) -> int:
        if not get_bool_env_var("SGLANG_FLYDSL_DYNAMIC_RECV_CAP", "false"):
            return physical_cap
        try:
            from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
            from sglang.srt.model_executor.runner import get_is_capture_mode
        except Exception:
            return physical_cap
        if not get_is_capture_mode():
            return physical_cap
        dp_global = get_dp_global_num_tokens()
        if dp_global is None or len(dp_global) <= 1:
            return physical_cap
        global_capacity = max(int(n) for n in dp_global) * len(dp_global)
        if global_capacity <= 0:
            return physical_cap
        recv_cap = max(32, 1 << (global_capacity - 1).bit_length())
        return min(physical_cap, recv_cap)

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage, (
            f"stage {self._stage} != expected {old_stage}"
        )
        self._stage = new_stage
