"""FlyDSL MLA decode stage-1, as a drop-in for ``_paged_decode_split_kernel``.

Env-gated and OFF by default; ``SGLANG_MLA_FLYDSL=1`` selects it. The kernel
itself lives outside the tree (``/shared_nfs/kk/flydsl_mla_decode.py``, see
``claude-skills/agentx/mi355x/FLYDSL_MLA_HANDOFF.md``); this module is only the
glue, and it exists because the two kernels disagree about three things:

1. **Index layout.** ``kv_indptr`` is per TOKEN and every draft token owns its
   own copy of its request's prefix. The FlyDSL kernel folds a request's whole
   verify window into the MFMA M dimension, so it wants one request-shared
   slice plus a per-token length. The slices are prefixes of each other, so the
   request slice *is* the last token's slice -- no new data, just a view.
2. **Tile size.** It contracts PV over ``BLOCK_K``, which must equal the bf16
   atom's K of 32, where the shipped bf16 path uses 16. The reduce kernel
   derives ``act_num_segments`` from the same constexpr, so the reduce has to
   be launched with 32 as well.
3. **Segmentation.** It plans splits from the request's ``kv_len_max``; the
   reduce plans from each token's own ``kv_len``. Near a length boundary the
   two disagree about how many splits hold data, and the partial buffers are
   reused across layers, so a mismatched read returns stale garbage. Handled
   on both sides: the kernel now always writes every split (an empty one gets
   m=-inf, l=0, acc=0), and the reduce is handed an indptr whose per-token
   lengths are rounded up to the request's max so its plan matches.
"""

import functools
import os
import sys

import torch

_ON = os.environ.get("SGLANG_MLA_FLYDSL", "0") == "1"
_CHECK = os.environ.get("SGLANG_MLA_FLYDSL_CHECK", "0") == "1"
_Q_LEN = int(os.environ.get("SGLANG_MLA_FLYDSL_QLEN", "7"))
_PATH = os.environ.get("SGLANG_MLA_FLYDSL_PATH", "/shared_nfs/kk")
_H_PER_WG = int(os.environ.get("SGLANG_MLA_FLYDSL_HW", "16"))


def _parse_ratios(spec):
    # "128" | "0,4,128" | "all" -- a set, because the three decode streams reach
    # this call site with different compress_ratio and per-token mode can serve
    # more than one of them.
    spec = spec.strip()
    if spec in ("all", "*"):
        return None  # admit every stream
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(None if tok in ("none", "None") else int(tok))
    return out


_RATIOS = _parse_ratios(os.environ.get("SGLANG_MLA_FLYDSL_RATIO", "128"))
# Per-token mode (_Q_LEN=1) frees ~50 KB of LDS the folded window needed, which
# is enough for the wider KV row stride that hw=8 measured 3% on.
_KV_PAD = int(os.environ.get("SGLANG_MLA_FLYDSL_KVPAD", "8"))
# Two-phase mode: fold the verify window into M on the invariant production
# actually satisfies -- a shared committed tail plus a union of sliding
# windows -- instead of the prefix assumption, which held on 8 of 56 real
# slices. 0 disables it; otherwise it is the SWA window width.
_WIN = int(os.environ.get("SGLANG_MLA_FLYDSL_WIN", "0"))
# Two-phase ablation arms. It has to reach build_kernel as an argument:
# FlyDSL's JIT cache key covers the kernel source and its closure
# scalars but NOT module globals, so a global flag silently reuses the
# first-compiled arm -- four control arms read identical before this.
_TP_ABL = os.environ.get("SGLANG_MLA_FLYDSL_TP_ABL", "")
# 0 keeps build_kernel's derived split; any other value overrides it. At 1 the
# cross-wave score reduction is skipped, which is what "S in registers" needs.
_QK_DSPLIT = int(os.environ.get("SGLANG_MLA_FLYDSL_QKDSPLIT", "0"))
_QK_ABL = os.environ.get("SGLANG_MLA_FLYDSL_QK_ABL", "")
# Threads cooperating on one row of online-softmax stats; 0/1 = the old scan.
# 8 is worth 10.3 us at the production shape and reads gsm8k 0.940; the cap is
# BLOCK_TH // M_ROWS, and build_kernel clamps to it, so this is safe at any hw.
_COOP = int(os.environ.get("SGLANG_MLA_FLYDSL_COOP", "8"))
_S_PAD = int(os.environ.get("SGLANG_MLA_FLYDSL_SPAD", "0"))
# Reduce a row's partials across lanes (DPP) instead of through LDS scratch.
_XLANE = int(os.environ.get("SGLANG_MLA_FLYDSL_XLANE", "1"))
# Fold the P pass into the stats pass, reusing its exp2. Implies _XLANE.
# Worth 8.4 us at the production shape; gsm8k 0.933 against the off arm's 0.936.
_MERGE_P = int(os.environ.get("SGLANG_MLA_FLYDSL_MERGEP", "1"))
# Ablation ladder rung ("empty"/"gather"/"stage"/"qk"/"softmax"). Output is
# wrong by construction; only the time is readable. Every earlier ladder in
# the handoff was measured in the q-folded geometry, which is not the one that
# ships.
_ABL = os.environ.get("SGLANG_MLA_FLYDSL_ABL", "") or None
BLOCK_K = 32

# Shapes this kernel refuses; counted so a silent fallback is visible.
_skips: dict[str, int] = {}
_hits = 0
# A silent fallback and a real null look identical in a server log, so the
# counters are printed every _LOG_EVERY eligibility checks. 0 disables.
_LOG_EVERY = int(os.environ.get("SGLANG_MLA_FLYDSL_LOG", "5000"))
_calls = 0
# Populated only under SGLANG_MLA_FLYDSL_CHECK=1.
_viol = 0
_checked = 0


def _skip(reason):
    _skips[reason] = _skips.get(reason, 0) + 1
    return False


def _tally(ok):
    global _calls, _hits
    _calls += 1
    if ok:
        _hits += 1
    if _LOG_EVERY and (_calls in (1, 100) or _calls % _LOG_EVERY == 0):
        print(
            f"[flydsl_mla] calls={_calls} hits={_hits} skips={_skips} "
            f"prefix_viol={_viol}/{_checked}",
            flush=True,
        )
    return ok


@functools.lru_cache(maxsize=16)
def _kernel(
    d,
    splits,
    h_per_wg,
    q_len,
    win=0,
    tp_abl="",
    ablate=None,
    qk_dsplit=0,
    qk_abl="",
    coop_stats=0,
    s_pad=0,
    coop_xlane=False,
    merge_p=False,
):
    if _PATH not in sys.path:
        sys.path.insert(0, _PATH)
    import flydsl_mla_decode as proto

    return proto.build_kernel(
        D=d,
        Q_LEN=q_len,
        SPLITS=splits,
        BLOCK_K=BLOCK_K,
        mfma_qk=True,
        mfma_pv=True,
        fast_softmax=True,
        qk_reuse=True,
        kv_single=True,
        alias_p=True,
        q_resident=True,
        kv_vec=8,
        kv_pad=_KV_PAD,
        tok_indptr=True,
        h_per_wg=h_per_wg,
        two_phase=bool(win),
        win=win or 128,
        tp_abl=tp_abl,
        ablate=ablate,
        qk_dsplit=qk_dsplit or None,
        qk_abl=qk_abl,
        coop_stats=coop_stats,
        s_pad=s_pad,
        coop_xlane=coop_xlane,
        merge_p=merge_p,
    )


def eligible(
    q, unified_kv, h, d, t, kv_splits, quant_kv, compress_ratio=None, q_len=None
):
    """Everything the kernel assumes, checked once per call."""
    if not _ON:
        return False
    # Stream gate. Three decode streams share this call site. It was set when
    # the kernel folded the verify window into M on a prefix assumption only
    # the HCA stream (compress_ratio 128) looked like it satisfied; per-token
    # mode (_Q_LEN=1) assumes nothing about index layout, so the gate is now a
    # set and exists only to keep out streams whose shapes are not a win.
    if _RATIOS is not None and compress_ratio not in _RATIOS:
        return _tally(_skip(f"stream ratio {compress_ratio}"))
    # The window width has to come from the caller. Inferring it from T is how
    # a DSpark draft worker's 6-token windows got regrouped as 7 whenever the
    # batch size was a multiple of 7.
    # _Q_LEN == 1 is per-token mode: nothing is folded across the window, so the
    # caller's width carries no information the kernel needs.
    if _Q_LEN != 1 and q_len != _Q_LEN:
        return _tally(_skip(f"window {q_len}"))
    if quant_kv:
        return _tally(_skip("fp8 kv"))
    if q.dtype is not torch.bfloat16 or unified_kv.dtype is not torch.bfloat16:
        return _tally(_skip("dtype"))
    if d != 512:
        return _tally(_skip("head_dim"))
    if t % _Q_LEN or t // _Q_LEN < 1:
        return _tally(_skip("token count not a whole number of verify windows"))
    if h % _H_PER_WG:
        return _tally(_skip("heads not divisible by h_per_wg"))
    if q.stride(2) != 1 or unified_kv.stride(1) != 1:
        return _tally(_skip("non-contiguous D"))
    return _tally(True)


_AUDIT = os.environ.get("SGLANG_MLA_FLYDSL_AUDIT", "0") == "1"
_audits = 0
_degenerate = 0
_worst = 0.0


def audit_on():
    # Capture forbids the sync the comparison needs, and a captured graph would
    # bake the second launch in anyway. Forcing eager decode steps is what
    # PARALLEL above --cuda-graph-max-bs-decode is for.
    return _AUDIT and not torch.cuda.is_current_stream_capturing()


def audit(out, recompute_triton, ctx=None):
    """Compare the merged output against Triton's on the same real inputs.

    Expect ~1.2e-03: both sides round P to bf16 independently, and that is the
    same figure `verify_integration.py` reports offline. A number far above it
    means the kernel is wrong on real metadata; a number at it means the
    accuracy loss is not coming from this kernel.
    """
    global _audits, _degenerate, _worst
    ref = recompute_triton()
    d = (out.float() - ref.float()).pow(2).sum().sqrt().item()
    n = ref.float().pow(2).sum().sqrt().item()
    # The non-captured decode steps include the pre-capture warmup, whose
    # inputs are dummy: both paths then produce all zeros and the ratio is
    # 0/clamp(0) = exactly 0.0, which reads as "bit-identical" and means
    # nothing. Count those separately instead of averaging them in.
    if n < 1e-6:
        _degenerate += 1
        if _degenerate in (1, 500):
            print(
                f"[flydsl_mla] audit skipped {_degenerate} degenerate "
                f"(zero-norm) calls, shape={tuple(out.shape)}",
                flush=True,
            )
        return
    r = d / n
    _audits += 1
    _worst = max(_worst, r)
    if r > 1e-2:
        _report_bad(r, out, ctx)
    if _audits <= 3 or _audits % 200 == 0:
        print(
            f"[flydsl_mla] audit {_audits}: relL2={r:.3e} worst={_worst:.3e} "
            f"ref_norm={n:.3e} shape={tuple(out.shape)}",
            flush=True,
        )


_bad = 0


def _report_bad(rel, out, ctx):
    """Characterise a failing call, and dump the first one for offline repro.

    The failures start ~1,000 audits in, long after the early ones pass at
    1.1e-03, so the discriminator is expected to be a length, not a shape.
    Printing the per-token kv lengths is what tests that.
    """
    global _bad
    _bad += 1
    if ctx is None:
        return
    ind = ctx["kv_indptr"]
    lens = (ind[1:] - ind[:-1]).float()
    print(
        f"[flydsl_mla] BAD {_bad}: relL2={rel:.3e} T={out.shape[0]} "
        f"splits={ctx['kv_splits']} kvlen min/mean/max="
        f"{int(lens.min())}/{int(lens.mean())}/{int(lens.max())} "
        f"indices_numel={ctx['kv_indices'].numel()}",
        flush=True,
    )
    if _bad == 1:
        p = f"{_PATH}/flydsl_bad_case.pt"
        torch.save(
            {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in ctx.items()
            },
            p,
        )
        print(f"[flydsl_mla] dumped failing inputs to {p}", flush=True)


def _check_prefix(kv_indices, kv_indptr, q_len, r):
    """The prefix assumption, verified rather than trusted. Debug-only: this
    is ~r*q_len device syncs, which costs far more than the kernel."""
    global _viol, _checked
    # The syncs below abort graph capture, and --disable-cuda-graph is not an
    # option on this recipe (it dies in an unrelated gemm at M=0). The
    # pre-capture warmup steps are real batches and are not captured, so
    # checking only outside capture gets the answer for free.
    if torch.cuda.is_current_stream_capturing():
        return
    for rr in range(r):
        base = int(kv_indptr[rr * q_len + q_len - 1])
        for i in range(q_len):
            t = rr * q_len + i
            s_, n = int(kv_indptr[t]), int(kv_indptr[t + 1]) - int(kv_indptr[t])
            _checked += 1
            if not torch.equal(kv_indices[s_ : s_ + n], kv_indices[base : base + n]):
                # Counted, not raised: a rate and its first example say far
                # more than dying on the first violation, and the run still
                # produces an accuracy number to correlate against.
                _viol += 1
                if _viol == 1:
                    print(
                        f"[flydsl_mla] PREFIX VIOLATION token {i}/{q_len} of "
                        f"request {rr}: len={n} base_len="
                        f"{int(kv_indptr[rr * q_len + q_len]) - base}",
                        flush=True,
                    )
    if _checked and _viol and _checked % 20000 < r * q_len:
        print(f"[flydsl_mla] prefix: {_viol}/{_checked} violations", flush=True)


def run_split(
    q,
    unified_kv,
    kv_indices,
    kv_indptr,
    m_partial,
    l_partial,
    acc_partial,
    h,
    d,
    t,
    kv_splits,
    qk_scale,
):
    """Fill the same (m, l, acc) partials the Triton split kernel would.

    Returns the verify-window width, which the reduce needs as FOLD_Q_LEN so
    that it plans its splits from the same per-request length this kernel did.

    ``qk_scale`` arrives already multiplied by LOG2E, which is what this kernel
    wants: it exp2s the scaled score directly, so the partials come out in the
    log2 domain the reduce expects.
    """
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from flydsl.compiler.jit_function import Stream

    r = t // _Q_LEN
    if _CHECK:
        _check_prefix(kv_indices, kv_indptr, _Q_LEN, r)
    fn = _kernel(
        d,
        kv_splits,
        _H_PER_WG,
        _Q_LEN,
        _WIN,
        _TP_ABL,
        _ABL,
        _QK_DSPLIT,
        _QK_ABL,
        _COOP,
        _S_PAD,
        bool(_XLANE) or bool(_MERGE_P),
        bool(_MERGE_P),
    )
    # _run_compiled, not fn(...): it caches the CompiledFunction on the launch
    # closure. Dispatching through the tracer on every call is a ~150 us
    # host-side cost that shows up as a batch-size-independent floor.
    _run_compiled(
        fn,
        q,
        int(q.stride(0)),
        int(q.stride(1)),
        unified_kv,
        int(unified_kv.stride(0)),
        kv_indices,
        kv_indptr,  # request slice start: the window's last token
        kv_indptr,  # per-token length: the difference of consecutive entries
        m_partial,
        l_partial,
        acc_partial,
        int(m_partial.stride(0)),
        int(m_partial.stride(1)),
        int(m_partial.stride(2)),
        int(acc_partial.stride(0)),
        int(acc_partial.stride(1)),
        int(acc_partial.stride(2)),
        float(qk_scale),
        int(r),
        int(h) // _H_PER_WG,
        Stream(torch.cuda.current_stream()),
    )
    return _Q_LEN, _WIN
