from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mxfp8_epilogue import ue8m0_scale


@triton.jit
def _wo_a_bf16_gemv_kernel(X, W, Y, R: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, D)
    x = tl.load(X + group * D + columns).to(tl.float32)
    w = tl.load(
        W + (group * R + rows[:, None]) * D + columns[None, :],
        rows[:, None] < R,
        0,
    ).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + group * R + rows, result, rows < R)


def wo_a_bf16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for one token."""
    assert x.shape[0] == 1 and x.ndim == weight.ndim == 3
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.is_contiguous() and weight.is_contiguous()
    groups, rows, dim = weight.shape
    assert x.shape[1:] == (groups, dim) and dim == triton.next_power_of_2(dim)
    result = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    # One output row per CTA keeps register use low and exposes enough
    # independent weight loads for single-token decode.
    _wo_a_bf16_gemv_kernel[(rows, groups)](
        x,
        weight,
        result,
        rows,
        dim,
        1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


@triton.jit
def _wo_a_partial(X, W, P, M: tl.constexpr, SX: tl.constexpr):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        w = tl.load(W + (group * 1024 + n[None, :]) * 4096 + offsets[:, None])
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


@triton.jit
def _wo_a_reduce(P, Y, E: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    tl.store(Y + i, tl.sum(values, 0), i < E)


def wo_a_bf16_small_batch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for the TP4 WO-A shape.
    Partial sums stay in FP32 until the final BF16, token-major store."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    _wo_a_reduce[(triton.cdiv(m * 2048, 256),)](partial, result, m * 2048, num_warps=4)
    return result


@triton.jit
def _wo_a_reduce_quant(P, Q, S, M: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    v = tl.load(P + split[:, None] * (M * 2048) + row * 2048 + i[None, :])
    y = tl.sum(v, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    amax = tl.max(tl.abs(y), 1)
    sf, inv = ue8m0_scale(amax)
    quant = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 2048 + i, quant.reshape((256,)))
    col = tile * 8 + tl.arange(0, 8)
    off = (col // 4) * 512 + row * 16 + col % 4
    tl.store(S + off, sf.to(tl.uint8))
    # Zero only padding rows; valid scale bytes have disjoint writers above.
    for z in tl.static_range(triton.cdiv(8192, M * 8 * 256)):
        s = (row * 8 + tile) * 256 + tl.arange(0, 256) + z * (M * 8 * 256)
        sr = (s % 512) // 16 + ((s % 16) // 4) * 32
        tl.store(S + s, 0, (s < 8192) & (sr >= M))


def _quantize_partial(p):
    m = p.shape[1]
    q = torch.empty((m, 2048), device=p.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(8192, device=p.device, dtype=torch.uint8)
    _wo_a_reduce_quant[(m, 8)](p, q, s, m, num_warps=4)
    return q, s


def wo_a_bf16_small_batch_mxfp8(x: torch.Tensor, weight: torch.Tensor):
    """WO-A with BF16 rounding followed by FlashInfer-compatible MXFP8 quantization."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    return _quantize_partial(partial)



# Split-K WO-A for any (G, R, D) with wo_b's MXFP8 quant in the reduce (gfx950 MXFP8 routes).
_SPLIT_K = 8
_SPLIT_K_BN, _SPLIT_K_BK = 64, 128
# Rows of one tile: while every row fits one tile the weight is read once; past that the
# bf16 bmm + quant wins (gfx950, G 1/2/4 x 1024 x 4096). BM 32 measured slower than 64.
WO_A_SPLIT_K_MAX_ROWS = 64


@triton.jit
def _wo_a_split_k_partial(
    X,
    W,
    P,
    M,
    SX,
    G: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    K_STEPS: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
):
    # axis 2 packs (row tile, K split): Triton grids have three axes
    tile, group = tl.program_id(0), tl.program_id(1)
    split, mt = tl.program_id(2) % SPLITS, tl.program_id(2) // SPLITS
    m = mt * BM + tl.arange(0, BM)
    n = tile * 64 + tl.arange(0, 64)
    k = split * (K_STEPS * 128) + tl.arange(0, 128)
    acc = tl.zeros((BM, 64), tl.float32)
    for i in range(K_STEPS):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * D + offsets[None, :], m[:, None] < M, 0
        )
        w = tl.load(W + (group * R + n[None, :]) * D + offsets[:, None])
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * G + group) * R + n[None, :],
        acc,
        m[:, None] < M,
    )


@triton.jit
def _wo_a_split_k_reduce_quant(P, Q, S, M, GR: tl.constexpr, SPLITS: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, SPLITS)
    v = tl.load(P + split[:, None] * (M * GR) + row * GR + i[None, :])
    y = tl.sum(v, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    amax = tl.max(tl.abs(y), 1)
    sf, inv = ue8m0_scale(amax)
    quant = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * GR + i, quant.reshape((256,)))
    tl.store(S + row * (GR // 32) + tile * 8 + tl.arange(0, 8), sf.to(tl.uint8))


def wo_a_split_k_mxfp8_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether wo_a_split_k_mxfp8 tiles x [M, G, D] @ weight [G, R, D]^T."""
    if x.ndim != 3 or weight.ndim != 3:
        return False
    g, r, d = weight.shape
    return (
        x.dtype == weight.dtype == torch.bfloat16
        and 1 <= x.shape[0] <= WO_A_SPLIT_K_MAX_ROWS
        and x.shape[1:] == (g, d)
        and x.stride(2) == 1
        and x.stride(1) == d
        and weight.is_contiguous()
        and r % _SPLIT_K_BN == 0
        and d % (_SPLIT_K * _SPLIT_K_BK) == 0
        and (g * r) % 256 == 0
    )


def wo_a_split_k_mxfp8(x: torch.Tensor, weight: torch.Tensor):
    """einsum('tgd,grd->tgr') rounded to BF16 and MXFP8-quantized in the split-K reduce:
    fp8 e4m3 [M, G * R] and row-major ue8m0 scales [M, G * R / 32]. At G = 2, R = 1024,
    D = 4096 and M <= 8 this is bitwise wo_a_bf16_small_batch + mxfp8_e4m3_quantize."""
    m = x.shape[0]
    g, r, d = weight.shape
    bm = 16 if m <= 16 else WO_A_SPLIT_K_MAX_ROWS
    partial = torch.empty((_SPLIT_K, m, g, r), dtype=torch.float32, device=x.device)
    _wo_a_split_k_partial[(r // _SPLIT_K_BN, g, _SPLIT_K * triton.cdiv(m, bm))](
        x,
        weight,
        partial,
        m,
        x.stride(0),
        g,
        r,
        d,
        d // (_SPLIT_K * _SPLIT_K_BK),
        _SPLIT_K,
        bm,
        num_warps=4,
        num_stages=3,
    )
    q = torch.empty((m, g * r), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((m, g * r // 32), device=x.device, dtype=torch.uint8)
    _wo_a_split_k_reduce_quant[(m, g * r // 256)](
        partial, q, s, m, g * r, _SPLIT_K, num_warps=4
    )
    return q, s
