import pytest
import torch

from sglang.kernels.ops.moe import moe_route_radix4
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS = 896
TOPK = 16


def _aiter_route(scores, bias, renormalize, scaling=1.0):
    from aiter import biased_grouped_topk

    weights = torch.empty(
        (scores.shape[0], TOPK), dtype=torch.float32, device=scores.device
    )
    ids = torch.empty((scores.shape[0], TOPK), dtype=torch.int32, device=scores.device)
    biased_grouped_topk(
        scores,
        bias,
        weights,
        ids,
        1,
        1,
        renormalize,
        scaling,
    )
    return weights, ids


def _canonical(weights, ids):
    order = ids.argsort(dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


def _assert_matches_aiter(scores, bias, renormalize, scaling=1.0):
    expected = _canonical(*_aiter_route(scores, bias, renormalize, scaling))
    actual = _canonical(
        *moe_route_radix4.route_radix4(scores, bias, TOPK, renormalize, scaling)
    )
    assert torch.equal(actual[1], expected[1])
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-3, atol=2e-4)


@pytest.mark.skipif(not is_hip(), reason="Radix-4 router is ROCm-only")
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_route_radix4_matches_aiter(m, dtype, renormalize):
    generator = torch.Generator(device="cuda").manual_seed(1000 + m)
    backing = torch.randn(
        (m, NUM_EXPERTS + 37),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    scores = backing[:, 19 : 19 + NUM_EXPERTS]
    bias = torch.randn(NUM_EXPERTS, dtype=dtype, device="cuda", generator=generator)
    assert scores.stride(0) == NUM_EXPERTS + 37
    _assert_matches_aiter(scores, bias, renormalize, scaling=2.5)


@pytest.mark.skipif(not is_hip(), reason="Radix-4 router is ROCm-only")
@pytest.mark.parametrize("renormalize", [False, True])
def test_route_radix4_ties_and_extremes(renormalize):
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")

    tied = torch.full((4, NUM_EXPERTS), 0.25, dtype=torch.bfloat16, device="cuda")
    tied[:, 7] = 2.0
    tied[:, 300] = 2.0
    tied[:, 800] = 1.5
    _assert_matches_aiter(tied, bias, renormalize, scaling=2.5)

    extreme = torch.linspace(
        -90, 90, NUM_EXPERTS, dtype=torch.float32, device="cuda"
    ).repeat(4, 1)
    _assert_matches_aiter(extreme.to(torch.bfloat16), bias, renormalize, scaling=2.5)


@pytest.mark.skipif(not is_hip(), reason="Radix-4 router is ROCm-only")
def test_route_radix4_nan_contract():
    scores = torch.randn((4, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    scores[:, 100] = float("nan")
    scores[:, 500] = float("nan")
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    _assert_matches_aiter(scores, bias, True, scaling=1.0)


@pytest.mark.skipif(not is_hip(), reason="Radix-4 router is ROCm-only")
def test_route_radix4_graph_replay():
    scores = torch.randn((32, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    expected = _canonical(*_aiter_route(scores, bias, True, 2.5))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_weights, actual_ids = moe_route_radix4.route_radix4(
            scores, bias, TOPK, True, 2.5
        )
    graph.replay()
    actual = _canonical(actual_weights, actual_ids)

    assert torch.equal(actual[1], expected[1])
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-3, atol=2e-4)


def test_route_radix4_coverage():
    scores = torch.empty((32, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    assert moe_route_radix4.covered(scores, bias, TOPK, 1, 1)
    assert not moe_route_radix4.covered(scores, bias, TOPK, 8, 4)
    assert not moe_route_radix4.covered(scores, bias, TOPK - 1, 1, 1)
    assert not moe_route_radix4.covered(scores[:, :-1], bias[:-1], TOPK, 1, 1)
    assert not moe_route_radix4.covered(
        scores.new_empty((1536, NUM_EXPERTS)), bias, TOPK, 1, 1
    )
