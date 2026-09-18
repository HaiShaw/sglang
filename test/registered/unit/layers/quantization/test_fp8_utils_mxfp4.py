import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    quantize_block_fp8_weight_to_mxfp4,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestFp8UtilsMxfp4(unittest.TestCase):
    def test_gfx1250_moe_weight_and_scale_shuffle(self):
        from sglang.srt.layers.quantization import fp8

        weight_shuffle = Mock(side_effect=lambda t, **kw: torch.ones_like(t))
        scale_shuffle = Mock(side_effect=lambda t, **kw: torch.ones_like(t))
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.zeros(2, 512, 128)),
            w2_weight=torch.nn.Parameter(torch.zeros(2, 256, 128)),
            w13_weight_scale_inv=torch.nn.Parameter(torch.zeros(2, 512, 8)),
            w2_weight_scale_inv=torch.nn.Parameter(torch.zeros(2, 256, 8)),
        )
        with patch.dict(
            fp8.__dict__,
            _use_aiter=True,
            _is_gfx1250_supported=True,
            _is_shuffle_moe_mxfp4=False,
            _require_fp4_dtype=lambda: torch.float32,
            moe_shuffle_weight=weight_shuffle,
            moe_shuffle_scale=scale_shuffle,
        ), fp8.envs.SGLANG_USE_AITER_MOE_GU_ITLV.override(True):
            fp8.Fp8MoEMethod.process_weights_after_loading_block_quant(
                SimpleNamespace(is_fp4_expert=True), layer
            )
        for shuffle in (weight_shuffle, scale_shuffle):
            calls = shuffle.call_args_list
            self.assertEqual([c.kwargs["gate_up"] for c in calls], [True, False])
            self.assertTrue(all(c.kwargs["is_guinterleave"] for c in calls))
        for call in scale_shuffle.call_args_list:
            self.assertEqual(call.kwargs["experts_cnt"], 2)
            self.assertEqual(call.args[0].ndim, 3)
        for name, tensor in vars(layer).items():
            if isinstance(tensor, torch.Tensor):
                self.assertTrue(torch.all(tensor == 1))
                if name.endswith("weight"):
                    self.assertTrue(tensor.is_shuffled)

    def test_quantize_block_fp8_weight_to_mxfp4_shapes_and_dtype(self):
        fp8_weight = (
            torch.linspace(-2.0, 2.0, 32 * 32, dtype=torch.float32)
            .reshape(32, 32)
            .to(torch.float8_e4m3fn)
        )
        fp8_scale = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)

        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )

        self.assertEqual(fp4_weight.dtype, torch.int8)
        self.assertEqual(fp4_weight.shape, torch.Size([32, 16]))
        self.assertEqual(fp4_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(fp4_scale.shape, torch.Size([32, 1]))

    def test_quantize_block_fp8_weight_to_mxfp4_grouped_weight(self):
        fp8_weight = (
            torch.linspace(-2.0, 2.0, 2 * 32 * 32, dtype=torch.float32)
            .reshape(2, 32, 32)
            .to(torch.float8_e4m3fn)
        )
        fp8_scale = torch.ones(2, 1, 1, dtype=torch.float8_e8m0fnu)

        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )

        self.assertEqual(fp4_weight.dtype, torch.int8)
        self.assertEqual(fp4_weight.shape, torch.Size([2, 32, 16]))
        self.assertEqual(fp4_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(fp4_scale.shape, torch.Size([2, 32, 1]))


if __name__ == "__main__":
    unittest.main()
