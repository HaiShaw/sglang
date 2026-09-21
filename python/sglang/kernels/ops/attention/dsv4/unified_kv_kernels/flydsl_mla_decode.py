#!/usr/bin/env python3
"""Stage-A FlyDSL MLA paged-decode prototype: plumbing only, no MFMA.

    PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python python3 flydsl_mla_decode.py [bs]

See /workspace/claude-skills/agentx/mi355x/FLYDSL_MLA_HANDOFF.md, "FlyDSL build
plan". This stage exists to settle the parts that are independent of the MFMA
fragment layout -- paged gather of kv_indices, the request-shared index layout,
a workgroup-shared KV tile in LDS, online softmax with per-row (per draft
token) masking, and the (m, l, acc) partial contract the harness merges. The
inner products are plain FMA loops on purpose; stage B swaps them for bf16 MFMA
atoms behind the same correctness gate.

Geometry: one workgroup owns (request, head, kv-split) and all Q_LEN draft
tokens of that request, so the KV tile is read once for all Q_LEN rows. 512
threads, thread t owning output column d = t.
"""

import itertools
import math
import sys

sys.path.insert(0, "/shared_nfs/kk")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import (  # noqa: E402
    _run_compiled,
    _to_raw,
    ptr_buf_tensor,
)
from flydsl.expr import const_expr, gpu, range_constexpr  # noqa: E402
from flydsl.expr.typing import Int32, Stream, T  # noqa: E402
from flydsl.expr.utils.arith import _to_raw as as_mlir_value  # noqa: E402

_NEG_LARGE = -3.4028234663852886e38


def build_kernel(
    D=512,
    Q_LEN=7,
    BLOCK_K=16,
    SPLITS=4,
    mfma_qk=False,
    mfma_pv=False,
    dbg_scores=False,
    ablate=None,
    fast_softmax=False,
    h_per_wg=1,
    qk_reuse=False,
    kv_single=False,
    alias_p=False,
    q_resident=False,
    q_global=False,
    lds_pool=False,
    pool_split=False,
    pool_pad=0,
    kv_vec=1,
    q_vec=False,
    kv_pad=0,
    s_pad=0,
    tok_indptr=False,
    p_pad=0,
    two_phase=False,
    win=128,
    tp_abl="",
    qk_dsplit=None,
    qk_abl="",
    coop_stats=0,
    coop_xlane=False,
    merge_p=False,
):
    # two_phase: fold the verify window into M the way production's index layout
    # actually permits. Each token's slice is [sliding ring window of
    # min(pos+1, win) entries][committed tail shared by the window], so the
    # window shares a SUFFIX. Phase A walks the shared tail at full M; phase B
    # walks the union of the seven windows, which is win + Q_LEN - 1 entries
    # because window q is window 0 shifted by q. Both invariants are checked by
    # real_meta.check(); the plain prefix assumption this replaces was violated
    # on 48 of every 56 production slices.
    assert not two_phase or (tok_indptr and fast_softmax), (
        "two_phase reads the per-token indptr and masks per row in the fast "
        "softmax; it has no slow-softmax path"
    )
    # fast_softmax: one thread per (draft token, kv) pair instead of all 512
    # threads redundantly computing the whole score block. Row max/sum are done
    # by Q_LEN threads, and m/l live in LDS across the tile loop rather than in
    # every thread's registers. Needs mfma_pv, which is what removed the reason
    # each thread had to hold every p locally.
    assert not fast_softmax or mfma_pv
    # More than one head per workgroup only exists on the B3 path; the
    # earlier arms stay at one head as controls.
    assert (
        h_per_wg == 1
        or (fast_softmax and mfma_pv)
        or ablate
        in (
            "empty",
            "gather",
            "stage",
            "qk",
        )
    )
    # ablate in (None, "stage", "qk", "softmax"): stop the per-tile pipeline
    # early to attribute time. Results are meaningless; only the time is.
    assert ablate in (None, "empty", "gather", "stage", "qk", "softmax")
    # The two LDS savings h_per_wg=16 needs (198 KB -> 158 KB), each a flag:
    #   kv_single: keep only the (kv, d) KV tile and read PV's B fragment out
    #     of it directly, instead of staging a second (d, kv) copy (-32 KB).
    #   alias_p: p shares the score buffer's storage (-7 KB).
    assert not kv_single or mfma_pv or ablate is not None
    # q_resident needs a wave to own a fixed (m_tile, k-chunk set) for the whole
    # tile loop, which is exactly what qk_reuse's assignment gives it.
    assert not q_resident or (mfma_qk and qk_reuse)
    # q_global: with the A fragments resident, Q is read once per workgroup, so
    # it does not need to be staged in LDS at all -- and Q is the term that has
    # kept every arm since B4 at one CTA per CU.
    assert not q_global or q_resident
    stage_q = not q_global
    # lds_pool: with q_resident the Q tile is dead once the pre-loop fragment
    # build is done, so the per-tile buffers are laid over it and the workgroup
    # costs max(Q, working set) instead of the sum. Needs q_resident (nothing
    # may read lds_q inside the tile loop) and keeps the loop-carried state
    # (alpha, m, l) *outside* the pool, since that has to survive every tile.
    assert not lds_pool or q_resident
    # kv_vec: bf16 elements each thread stages per load. At 1 (the arms up to
    # B8) a thread loads ONE bf16 per KV row through a dword-load-and-shift,
    # i.e. BLOCK_K loads and BLOCK_K index loads per tile, all 512 threads
    # fetching the same index. At 8 the load is a single 128-bit dword4 and
    # both counts drop by 8x. Needs the global base to stay 16-byte aligned,
    # which holds because kv_stride_n is D and the column base is a multiple
    # of kv_vec.
    assert kv_vec in (1, 2, 4, 8)
    # kv_pad: extra bf16 columns on the KV tile's LDS row stride. Unpadded the
    # stride is D = 512 bf16 = 1024 B, a multiple of the 128 B the 32 LDS banks
    # cover, so every row starts in bank 0 -- and a QK B fragment has its 16
    # lanes reading 16 different rows, i.e. a 16-way conflict on every
    # ds_read_b128. rocprofv3 measures LDSBankConflict 26.8 % on B11 hw=16
    # against 9.3 % for the shipped kernel. 8 bf16 of padding moves consecutive
    # rows 4 banks apart. Costs BLOCK_K * kv_pad * 2 B of LDS (512 B at 8).
    KV_STRIDE = D + kv_pad
    # s_pad: same story for the score tile. Its row stride is BLOCK_K = 32
    # fp32 = 128 B, exactly the span of the 32 banks, so every row starts in
    # bank 0 and the fast-softmax row scan -- thread d reading row d -- puts
    # all 64 lanes of a wave on one bank. One fp32 of padding makes the bank
    # (d + kk) mod 32, i.e. one per lane. MEASURED REGRESSION, left at 0:
    # the conflict is real but the volume is not (the row scan is ~3.5k
    # element reads per tile against QK's ~131k), and the flat-pair-index
    # decomposition it forces costs more than it saves -- hw=8 155.7 -> 164.3.
    S_STRIDE = BLOCK_K + s_pad
    # p_pad: the P tile has the same defect the KV tile had. Its row stride
    # is BLOCK_K = 32 bf16 = 64 B = 16 banks, so PV's A fragment -- 16 lanes
    # on 16 rows -- is an 8-way conflict, and softmax+PV is 49 % of the
    # kernel once kv_pad has fixed QK. Unlike s_pad this needs no extra
    # index arithmetic (the row is already in hand where p is written) and
    # no extra LDS (alias_p puts p inside the larger s region). MEASURED
    # REGRESSION, left at 0: 144.7 -> 175.6 us, and pad 8 and 16 cost the
    # *same*, so it is not the padding amount -- a non-compact P tile drops
    # the A-fragment copy off its wide vectorised path. Padding pays on a
    # tile that was already strided (KV), not on a compact one (P, s).
    P_STRIDE = BLOCK_K + p_pad
    assert kv_pad % kv_vec == 0, "padding must keep the vector stores aligned"
    stage_kvt = (mfma_pv or ablate is not None) and not kv_single
    do_stage = ablate not in ("empty", "gather")
    BLOCK_TH = D  # one thread per output column
    # KV staging geometry: TH_PER_ROW threads cover one KV row, so each round
    # stages BLOCK_TH // TH_PER_ROW rows and the tile takes KV_ROUNDS of them.
    TH_PER_ROW = D // kv_vec
    ROWS_PER_ROUND = BLOCK_TH // TH_PER_ROW
    assert BLOCK_K % ROWS_PER_ROUND == 0, "KV rows per round must divide BLOCK_K"
    KV_ROUNDS = BLOCK_K // ROWS_PER_ROUND
    # Rows of the attention problem owned by one workgroup: H_PER_WG heads
    # x Q_LEN draft tokens, laid out head-major (row = hh * Q_LEN + q).
    M_ROWS = Q_LEN * h_per_wg
    # MEASURED NULL, left behind its own flag: applying the KV vectorisation to
    # the Q staging loop too is neutral at hw=8 (200.0 vs 200.1 µs) and a
    # regression at hw=16 (234 vs 196). Q is staged once per workgroup, so the
    # load saving is small, while the store side gets worse: thread `d` writing
    # 8 contiguous bf16 puts 64 lanes on a 16-byte stride, which conflicts,
    # where one-column-per-thread is conflict-free. Re-try only together with a
    # 128-bit LDS store.
    Q_ROUNDS = (
        (M_ROWS // ROWS_PER_ROUND)
        if (q_vec and kv_vec > 1 and M_ROWS % ROWS_PER_ROUND == 0)
        else 0
    )
    N_PAIRS = M_ROWS * BLOCK_K
    PAIRS_PER_TH = (N_PAIRS + BLOCK_TH - 1) // BLOCK_TH
    DOT_SPLIT = max(1, BLOCK_TH // N_PAIRS)  # threads cooperating on one score
    # coop_stats: threads cooperating on one row's online-softmax stats. 1 keeps
    # the one-thread-per-row scan, which SGLANG_MLA_FLYDSL_QK_ABL=nostats prices
    # at 21.3 us of a 109 us kernel -- not read volume (that was s_pad's wrong
    # conclusion) but serialisation: M_ROWS of BLOCK_TH threads walk BLOCK_K
    # scores twice while the other NWAVE-1 waves sit at the barrier. Each thread
    # reduces STAT_CH scores against its *own* local max, which is what makes
    # the combine associative: m = max(m_c), l = sum(l_c * exp2(m_c - m)).
    STAT_SPLIT = (
        max(1, min(coop_stats, BLOCK_TH // M_ROWS, BLOCK_K)) if coop_stats else 1
    )
    assert BLOCK_K % STAT_SPLIT == 0, "a row's scores must divide over its threads"
    STAT_CH = BLOCK_K // STAT_SPLIT
    STAT_ELEMS = 2 * BLOCK_TH if STAT_SPLIT > 1 else 1
    # coop_xlane: reduce a row's STAT_SPLIT partials across lanes instead of
    # through LDS scratch. A row's threads are STAT_SPLIT *consecutive* lanes
    # (row = d // STAT_SPLIT), so an aligned power-of-two group never crosses a
    # DPP row or a wave, and the xor ladder below is exact. Reducing the max
    # first means the sum needs no per-thread rescale: STAT_CH exp2 total,
    # against STAT_CH + STAT_SPLIT for the LDS path, and no barrier.
    XL_STEPS = tuple(
        ctrl
        for width, ctrl in (
            (2, 0xB1),  # quad_perm [1,0,3,2]  = lane ^ 1
            (4, 0x4E),  # quad_perm [2,3,0,1]  = lane ^ 2
            (8, 0x141),  # row_half_mirror     = lane ^ 7
            (16, 0x140),  # row_mirror         = lane ^ 15
        )
        if width <= STAT_SPLIT
    )
    assert not coop_xlane or STAT_SPLIT in (1, 2, 4, 8, 16), (
        "the xor ladder needs a power-of-two group of at most 16 lanes"
    )
    # Threads whose row is past M_ROWS exist only when the split does not divide
    # the block evenly; they must not read or write past their tiles.
    STAT_RAGGED = STAT_SPLIT > 1 and M_ROWS * STAT_SPLIT != BLOCK_TH
    # merge_p: the stats pass and the P pass compute exp2 on the *same* elements
    # -- once for l, once as exp2(s - m_new). With coop_xlane every lane already
    # holds m_blk, so every lane can derive m_new = max(m_old, m_blk) with no
    # broadcast, and P = exp2(sv - m_blk) * exp2(m_blk - m_new) reuses the terms
    # the l sum already built, for one scalar exp2. That retires the P pass's
    # BLOCK_K * M_ROWS exp2, its lds_s re-reads and one of the three barriers.
    MERGE_P = bool(merge_p) and coop_xlane and STAT_SPLIT > 1
    assert not merge_p or (coop_xlane and STAT_SPLIT > 1), (
        "merge_p reuses the cross-lane reduction's registers; it needs coop_xlane"
    )
    assert not MERGE_P or fast_softmax, "merge_p replaces the fast-softmax P pass"
    DOT_CHUNK = D // DOT_SPLIT

    # Stage B: both inner products run on the bf16 16x16x32 atom proved in
    # flydsl_mfma_probe.py, which wants each operand held (rows, K) row-major
    # in LDS and computes A @ B.T. The Q/P rows are padded from Q_LEN to MMA_M
    # and the pad rows are ignored; that padding is the whole cost of running
    # at one head per workgroup, and it goes away when stage C folds heads into
    # M (7 -> 112 = 7 x 16 fills seven full atom tiles).
    #
    #   QK: C[q, kv] = Q[MMA_M, dslice] @ KV[BLOCK_K, dslice].T, contraction D
    #       split evenly over the waves and reduced through LDS (same shape of
    #       combine the FMA path does with DOT_SPLIT), BLOCK_K / MMA_N n-tiles.
    #   PV: C[q, d] = P[MMA_M, BLOCK_K] @ KVT[dtile, BLOCK_K].T, contraction
    #       BLOCK_K, so BLOCK_K must equal MMA_K and KV needs a transposed copy.
    MMA_M, MMA_N, MMA_K = 16, 16, 32
    NWAVE = BLOCK_TH // 64
    NREG = MMA_M * MMA_N // 64
    NREG_B = MMA_N * MMA_K // 64
    NREG_A = MMA_M * MMA_K // 64
    NT_N = BLOCK_K // MMA_N  # score n-tiles
    CH_PER_WAVE = (D // MMA_K) // NWAVE  # legacy, kept for the FMA arm
    NT_D_PER_WAVE = (D // MMA_N) // NWAVE  # PV output d-tiles per wave
    # Loop-carried accumulator scalars per lane: one output column per thread
    # on the FMA path, one MFMA accumulator fragment per d-tile on the MFMA one.
    M_TILES = (M_ROWS + MMA_M - 1) // MMA_M
    # QK work assignment: one (m_tile, n_tile) output tile per wave. The D
    # contraction is only split across waves when there are fewer output tiles
    # than waves, which is what used to force a per-wave partial score tile in
    # LDS and cap h_per_wg at 4.
    # qk_reuse: a wave owns an m_tile and *all* NT_N n-tiles, so the A fragment
    # is loaded once per k-chunk and feeds NT_N MFMAs instead of one. Fewer
    # output items per pass, so QK_DSPLIT typically rises by NT_N and the
    # cross-wave reduction comes back.
    QK_ITEMS = M_TILES if qk_reuse else M_TILES * NT_N
    QK_DSPLIT = max(1, min(D // MMA_K, NWAVE // QK_ITEMS))
    # qk_dsplit: override the derived split, to price what the cross-wave score
    # reduction costs. At 1 the MFMA writes scaled scores straight to lds_s and
    # _combine is skipped entirely, at the price of leaving NWAVE - QK_ITEMS
    # waves idle during QK. This is the prize measurement for "S in registers",
    # which can only exist at QK_DSPLIT == 1 -- registers do not cross waves.
    if qk_dsplit is not None:
        QK_DSPLIT = max(1, min(D // MMA_K, qk_dsplit))
    QK_GROUPS = NWAVE // QK_DSPLIT
    QK_PER_WAVE = (QK_ITEMS + QK_GROUPS - 1) // QK_GROUPS
    CH_PER_SLICE = (D // MMA_K) // QK_DSPLIT
    # Independent accumulators per wave, to stop back-to-back MFMAs on one
    # accumulator serialising on the atom's latency. MEASURED NULL: 4 vs 1
    # changed nothing at h_per_wg 4 (359.8 vs 360.5 us) or 8 (270.1 vs 270.8),
    # so the QK rung is not MFMA-latency bound. Left at 1; the loop below is
    # general if a future change makes the atom the critical path again.
    QK_ACCS = 1
    N_ACC = M_TILES * NT_D_PER_WAVE * NREG if mfma_pv else M_ROWS
    if ablate in ("empty", "gather", "stage", "qk"):
        # Fixed-width ablation tail: if it scaled with M_ROWS it would put
        # its own carried-state work into the rung it is meant to measure.
        N_ACC = 4
    Q_ROWS = M_TILES * MMA_M if mfma_qk or mfma_pv else M_ROWS
    # Only the Q_LEN real rows of a score tile are ever read back, and LDS is
    # the binding budget here, so the per-wave partial tile is Q_LEN rows tall
    # rather than MMA_M -- the MFMA store below drops the pad rows.
    if mfma_qk:
        S4_ELEMS = QK_DSPLIT * M_ROWS * BLOCK_K if QK_DSPLIT > 1 else 1
    else:
        S4_ELEMS = N_PAIRS * DOT_SPLIT
    if mfma_qk:
        assert BLOCK_K % MMA_N == 0
    if mfma_pv:
        assert BLOCK_K == MMA_K, "PV contracts over BLOCK_K, which must be MMA_K"

    # alias_p needs the p writes to see none of the s values still wanted by
    # another thread, so every thread holds its own p in registers across a
    # barrier before storing (see _fast_softmax_pv). The atom pad rows are
    # zeroed once before the tile loop on the unaliased path; here s overwrites
    # them every tile, so they are re-zeroed alongside the p stores.
    assert not alias_p or fast_softmax or ablate is not None
    # p lives inside the s region when aliased, so padding it must still fit.
    assert not (alias_p and mfma_pv) or (
        Q_ROWS * P_STRIDE * 2 <= M_ROWS * S_STRIDE * 4
    ), "p_pad overflows the aliased s buffer"
    # alias_p keeps no p array of its own: the bf16 p tile is a recast_iter
    # view of the f32 score buffer, which is 2x its size. (fx.union exists but
    # SharedAllocator cannot peek a Sum-policy type.)
    P_ELEMS = (Q_ROWS * P_STRIDE) if (mfma_pv and not alias_p) else 1

    # Pool offsets, in bf16 elements and all multiples of 8 so every sub-buffer
    # keeps 16-byte alignment for the copy atoms and 4-byte for the f32 views.
    # pool_split is the control arm for lds_pool: same single allocation and
    # the same computed offsets, but the per-tile buffers placed *after* Q
    # instead of on top of it. It saves no LDS, so any time difference between
    # it and the separate-allocation arm is the cost of pooling itself, and any
    # difference between it and lds_pool is what the overlap is worth.
    Q_ELEMS = (Q_ROWS * D) if stage_q else 0
    KV_OFF = Q_ELEMS if pool_split else 0
    KVT_OFF = KV_OFF + BLOCK_K * KV_STRIDE
    S_OFF = KVT_OFF + ((D * BLOCK_K) if stage_kvt else 0)
    S4_OFF = S_OFF + 2 * M_ROWS * S_STRIDE
    WS_ELEMS = S4_OFF + 2 * S4_ELEMS
    # pool_pad buys back LDS the overlap freed, without undoing the overlap:
    # it separates "the reuse is unsafe/slow" from "the smaller LDS changed the
    # occupancy the register allocator targets".
    POOL_ELEMS = max(Q_ELEMS, WS_ELEMS) + pool_pad
    for _off in (KV_OFF, KVT_OFF, S_OFF, S4_OFF):
        assert _off % 8 == 0, "pool sub-buffer would be misaligned"

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.BFloat16, POOL_ELEMS if lds_pool else 1, 16]
        q: fx.Array[
            fx.BFloat16, 1 if lds_pool else ((Q_ROWS * D) if stage_q else 1), 16
        ]
        kv: fx.Array[fx.BFloat16, 1 if lds_pool else (BLOCK_K * KV_STRIDE), 16]
        kvt: fx.Array[
            fx.BFloat16, 1 if (lds_pool or not stage_kvt) else (D * BLOCK_K), 16
        ]
        alpha: fx.Array[fx.Float32, Q_ROWS if mfma_pv else 1, 16]
        mrow: fx.Array[fx.Float32, M_ROWS if fast_softmax else 1, 16]
        lrow: fx.Array[fx.Float32, M_ROWS if fast_softmax else 1, 16]
        p: fx.Array[fx.BFloat16, P_ELEMS, 16]
        s: fx.Array[fx.Float32, 1 if lds_pool else (M_ROWS * S_STRIDE), 16]
        s4: fx.Array[fx.Float32, 1 if lds_pool else S4_ELEMS, 16]
        stat: fx.Array[fx.Float32, STAT_ELEMS, 16]

    stage = (
        ("abl_" + ablate)
        if ablate
        else (
            "B3" if fast_softmax else ("B2" if mfma_pv else ("B1" if mfma_qk else "A"))
        )
    )

    @flyc.kernel(
        name=f"mla_decode_stage{stage}_D{D}_Q{Q_LEN}_K{BLOCK_K}_S{SPLITS}"
        f"_qd{QK_DSPLIT}{qk_abl}_cs{STAT_SPLIT}{'x' if coop_xlane else ''}"
        f"{'mp' if MERGE_P else ''}_flydsl",
        known_block_size=[BLOCK_TH, 1, 1],
    )
    def kernel(
        q_in: fx.Tensor,  # bf16 [N, H, D]
        q_stride_t: Int32,
        q_stride_h: Int32,
        kv_in: fx.Tensor,  # bf16 [NSLOT, D]
        kv_stride_n: Int32,
        kv_indices: fx.Tensor,  # i32, request-shared slices
        req_indptr: fx.Tensor,  # i32 [R+1]
        tok_len: fx.Tensor,  # i32 [N]
        m_out: fx.Tensor,  # f32 [N, SPLITS, H]
        l_out: fx.Tensor,
        acc_out: fx.Tensor,  # f32 [N, SPLITS, H, D]
        mp_t: Int32,
        mp_k: Int32,
        mp_h: Int32,
        ap_t: Int32,
        ap_k: Int32,
        ap_h: Int32,
        qk_scale: fx.Float32,
    ):
        f32 = T.f32
        i32 = T.i32

        r = fx.block_idx.x
        h = fx.block_idx.y * fx.Int32(h_per_wg)
        pid_k = fx.block_idx.z
        tid = fx.thread_idx.x
        d = fx.Int32(tid)
        lane = d % fx.Int32(64)
        wave = d // fx.Int32(64)

        c_neg_large = fx.Float32(_NEG_LARGE)

        def fexp2(x):
            return fx.Float32(fx.rocdl.exp2(f32, _to_raw(x)))

        q_buf = ptr_buf_tensor(fx.get_iter(q_in), fx.Int32)
        kv_buf = ptr_buf_tensor(fx.get_iter(kv_in), fx.Int32)
        idx_buf = ptr_buf_tensor(fx.get_iter(kv_indices), fx.Int32)
        ptr_buf = ptr_buf_tensor(fx.get_iter(req_indptr), fx.Int32)
        len_buf = ptr_buf_tensor(fx.get_iter(tok_len), fx.Int32)

        def load_bf16(buf, off_elems):
            """One bf16 element at an arbitrary (possibly odd) element offset."""
            base_off = fx.Int32(off_elems)
            off_dw = base_off >> 1
            lane_in_dw = base_off & 1
            raw_s = fx.Int32(fx.add_offset(fx.get_iter(buf), off_dw).load(i32))
            hi = fx.Int32((fx.Uint32(raw_s) >> 16).ir_value())
            lo_or_hi = (lane_in_dw == fx.Int32(0)).select(raw_s, hi)
            lo16 = lo_or_hi & 0xFFFF
            pair = fx.Vector.from_elements([lo16], dtype=fx.Int32).bitcast(fx.BFloat16)
            return pair[0].to(fx.Float32)

        def load_bf16_vec(buf, off_elems):
            """kv_vec contiguous bf16 at a kv_vec-aligned element offset.

            At kv_vec=1 this is the dword-plus-bit-extract path (the offset can
            be odd); above that it is one dword2/dword4 load and a bitcast, and
            the values stay bf16 -- the scalar path's round trip through f32
            and back was pure waste for a copy. Idiom copied from
            `fused_compress_attn_hca.py::_load_bf16_vec_to_f32`.
            """
            if const_expr(kv_vec == 1):
                return [load_bf16(buf, off_elems).to(fx.BFloat16)]
            off_dw = fx.Int32(off_elems) >> 1
            raw = fx.Vector(
                fx.add_offset(fx.get_iter(buf), off_dw).load(T.vec(kv_vec // 2, i32))
            )
            return raw.bitcast(fx.BFloat16)

        def load_i32(buf, off):
            return fx.Int32(fx.add_offset(fx.get_iter(buf), fx.Int32(off)).load(i32))

        # -- per-request / per-row metadata -----------------------------
        # tok_indptr: read both from the production per-TOKEN kv_indptr
        # (passed for both arguments) instead of the harness's pre-split
        # start/length arrays. The request's slice is its LAST token's, which
        # is the longest and the one the others are prefixes of. This is what
        # lets the integration hand the kernel production tensors with no
        # host-side index munging -- that glue measured 36 us per call.
        tok0 = fx.Int32(r) * Q_LEN
        if const_expr(tok_indptr):
            kv_start = load_i32(ptr_buf, tok0 + (Q_LEN - 1))
            ends = [load_i32(len_buf, tok0 + q) for q in range_constexpr(Q_LEN + 1)]
            row_len = [ends[q + 1] - ends[q] for q in range_constexpr(Q_LEN)]
        else:
            kv_start = load_i32(ptr_buf, r)
            row_len = [load_i32(len_buf, tok0 + q) for q in range_constexpr(Q_LEN)]

        def dyn_pick(lst, qq):
            """lst[qq] for a runtime qq: a select chain, not a reload."""
            v = lst[Q_LEN - 1]
            for q in range_constexpr(Q_LEN - 1):
                v = (qq == fx.Int32(q)).select(lst[q], v)
            return v

        def dyn_row_len(qq):
            """Length of token qq of this request, qq being a runtime value.

            A select chain over the constexpr row_len list, not a reload: the
            masking paths call this once per (thread, tile) and again per pair
            iteration, so the original single global load was already ~7 loads
            per thread per tile, and deriving lengths from tok_indptr would
            have doubled that. Q_LEN is 7, so this is 6 v_cndmask.
            """
            v = row_len[Q_LEN - 1]
            for q in range_constexpr(Q_LEN - 1):
                v = (qq == fx.Int32(q)).select(row_len[q], v)
            return v

        kv_len_max = row_len[0]
        for q in range_constexpr(Q_LEN - 1):
            nxt = row_len[q + 1]
            kv_len_max = (nxt > kv_len_max).select(nxt, kv_len_max)

        if const_expr(two_phase):
            # win_len[q] = min(len_q, win) is the sliding-window segment; the
            # rest is the committed tail. u0[q] is where row q's window starts
            # on the union axis: q when every token has saturated the window,
            # 0 while it is still growing, and correct in between because
            # u0 = q - (win_q - win_0) is just the difference of the two ring
            # start positions.
            # Only two window lengths have to stay live across the tile loop.
            # Everything per-row is recovered from row_len, which was live
            # already: win_q = min(len_q, win), and the union upper bound
            # collapses to u0_q + win_q = q + win_0. Keeping the per-row lists
            # live instead cost ~30% -- 21 extra loop-carried values on a kernel
            # whose register budget is already the binding constraint.
            def _win_of(rl):
                return (rl > fx.Int32(win)).select(fx.Int32(win), rl)

            win_0 = _win_of(row_len[0])
            win_L = _win_of(row_len[Q_LEN - 1])
            tail_L = row_len[Q_LEN - 1] - win_L
            u0_L = fx.Int32(Q_LEN - 1) + win_0 - win_L
            tail_base = ends[Q_LEN - 1] + win_L
            win0_base = ends[0]
            # union coord u >= win_0 is served by the last token's window, whose
            # entry i sits at u0_L + i.
            winL_base = ends[Q_LEN - 1] - u0_L
            union_len = win_0 + fx.Int32(Q_LEN - 1)
            tail_tiles = (tail_L + (BLOCK_K - 1)) // BLOCK_K
            num_tiles = tail_tiles + (union_len + (BLOCK_K - 1)) // BLOCK_K
            tiles_per_seg = (num_tiles + (SPLITS - 1)) // SPLITS
        else:
            num_tiles = (kv_len_max + (BLOCK_K - 1)) // BLOCK_K
            tiles_per_seg = (kv_len_max + (SPLITS * BLOCK_K - 1)) // (SPLITS * BLOCK_K)
        tile_start = fx.Int32(pid_k) * tiles_per_seg
        tile_end_raw = tile_start + tiles_per_seg
        tile_end_cl = (tile_end_raw > num_tiles).select(num_tiles, tile_end_raw)
        # A split past the end of the sequence gets zero iterations, not a
        # negative trip count -- see the epilogue note at the bottom.
        tile_end = (tile_end_cl < tile_start).select(tile_start, tile_end_cl)
        if const_expr(ablate == "empty"):
            # Zero tile iterations: measures launch + metadata + epilogue only.
            tile_end = tile_start

        def _body():
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            pool = lds.pool.ptr
            lds_q = pool if const_expr(lds_pool) else lds.q.ptr
            lds_kv = (pool + KV_OFF) if const_expr(lds_pool) else lds.kv.ptr
            lds_kvt = (pool + KVT_OFF) if const_expr(lds_pool) else lds.kvt.ptr
            lds_s = (
                fx.recast_iter(fx.Float32, pool + S_OFF)
                if const_expr(lds_pool)
                else lds.s.ptr
            )
            lds_s4 = (
                fx.recast_iter(fx.Float32, pool + S4_OFF)
                if const_expr(lds_pool)
                else lds.s4.ptr
            )
            lds_p = (
                fx.recast_iter(fx.BFloat16, lds_s) if const_expr(alias_p) else lds.p.ptr
            )
            lds_alpha = lds.alpha.ptr
            lds_stat = lds.stat.ptr
            lds_m = lds.mrow.ptr
            lds_l = lds.lrow.ptr

            # const_expr, not a bare `if`: the AST rewriter turns a plain `if`
            # into an scf.if branch function and these names would not escape it.
            if const_expr(mfma_qk or mfma_pv):
                mma_atom = fx.make_mma_atom(
                    fx.rocdl.MFMA(MMA_M, MMA_N, MMA_K, fx.BFloat16)
                )
                tiled_mma = fx.make_tiled_mma(
                    mma_atom,
                    fx.make_layout((1, 1, 1), (1, 1, 0)),
                    fx.make_tile(
                        None, None, fx.make_layout((MMA_K // 4, 4), (1, MMA_K // 4))
                    ),
                )
                copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
                thr_mma = tiled_mma.thr_slice(lane)
                thr_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(lane)
                thr_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(lane)
                cRow = thr_mma.partition_C(
                    fx.make_view(0, fx.make_layout((MMA_M, MMA_N), (1, 0)))
                )
                cCol = thr_mma.partition_C(
                    fx.make_view(0, fx.make_layout((MMA_M, MMA_N), (0, 1)))
                )
                # Each wave reduces its own D slice into its own score tile.
                sScore = fx.make_view(
                    lds_s4 + wave * (Q_LEN * BLOCK_K),
                    fx.make_ordered_layout((Q_LEN, BLOCK_K), (1, 0)),
                )
                # Shape donor for PV accumulators; never stored through.
                sCdonor = fx.make_view(
                    lds_s4, fx.make_ordered_layout((MMA_M, MMA_N), (1, 0))
                )
                sP = fx.make_view(
                    lds_p, fx.make_layout((MMA_M, BLOCK_K), (P_STRIDE, 1))
                )
                # Shape donor and per-lane (row, col) map for a B fragment,
                # read off coordinate views the way cRow/cCol do for C. This is
                # what lets kv_single build PV's B operand from the (kv, d)
                # tile without a second, transposed LDS copy of it.
                sBdonor = fx.make_view(
                    lds_kv, fx.make_ordered_layout((MMA_N, MMA_K), (1, 0))
                )
                sAdonor = fx.make_view(
                    lds_kv, fx.make_ordered_layout((MMA_M, MMA_K), (1, 0))
                )
                aRow = thr_mma.partition_A(
                    fx.make_view(0, fx.make_layout((MMA_M, MMA_K), (1, 0)))
                )
                aCol = thr_mma.partition_A(
                    fx.make_view(0, fx.make_layout((MMA_M, MMA_K), (0, 1)))
                )
                bRow = thr_mma.partition_B(
                    fx.make_view(0, fx.make_layout((MMA_N, MMA_K), (1, 0)))
                )
                bCol = thr_mma.partition_B(
                    fx.make_view(0, fx.make_layout((MMA_N, MMA_K), (0, 1)))
                )

            def lds_ld(ptr, off):
                """Read one bf16 LDS element back as f32."""
                return fx.ptr_load(ptr + off).to(fx.Float32)

            def dpp_f32(val, ctrl):
                """This lane's f32 as seen from the lane `ctrl` selects."""
                raw = as_mlir_value(fx.Float32(val).bitcast(fx.Int32).ir_value())
                zero = as_mlir_value(fx.Int32(0).ir_value())
                moved = fx.rocdl.update_dpp(T.i32, zero, raw, ctrl, 0xF, 0xF, True)
                return fx.Int32(moved).bitcast(fx.Float32)

            # -- stage Q, same vectorisation as the KV tile. Once per workgroup
            # rather than once per tile, but at kv_vec=1 it is still M_ROWS
            # scalar loads per thread, which is the same order as the whole
            # tile loop's KV staging now costs.
            for rnd in range_constexpr(Q_ROUNDS if (do_stage and stage_q) else 0):
                row = fx.Int32(rnd * ROWS_PER_ROUND) + (d // fx.Int32(TH_PER_ROW))
                cbase = (d % fx.Int32(TH_PER_ROW)) * kv_vec
                off = (
                    (tok0 + row % fx.Int32(Q_LEN)) * fx.Int32(q_stride_t)
                    + (fx.Int32(h) + row // fx.Int32(Q_LEN)) * fx.Int32(q_stride_h)
                    + cbase
                )
                vals = load_bf16_vec(q_buf, off)
                for i in range_constexpr(kv_vec):
                    fx.ptr_store(vals[i], lds_q + (row * D + cbase + i))
            for r in range_constexpr(
                M_ROWS if (do_stage and stage_q and not Q_ROUNDS) else 0
            ):
                hh, q = divmod(r, Q_LEN)
                off = (
                    (tok0 + q) * fx.Int32(q_stride_t)
                    + (fx.Int32(h) + hh) * fx.Int32(q_stride_h)
                    + d
                )
                fx.ptr_store(load_bf16(q_buf, off).to(fx.BFloat16), lds_q + (r * D + d))
            # The MFMA tiles are Q_ROWS rows tall; the pad rows feed real lanes,
            # so zero them rather than letting whatever is in LDS reach the atom.
            # (q_global masks them in the fragment load instead.)
            for r in range_constexpr((Q_ROWS - M_ROWS) if stage_q else 0):
                fx.ptr_store(fx.BFloat16(0.0), lds_q + ((M_ROWS + r) * D + d))
            if const_expr(mfma_pv):
                # P and alpha pad rows are never written again, so zero once:
                # a pad row then contributes nothing to its (unread) acc rows.
                # alias_p is the exception -- s overwrites p every tile, so its
                # pad rows are re-zeroed in the loop and doing it here would
                # only write into the pool before Q has been read out of it.
                for r in range_constexpr(0 if alias_p else (Q_ROWS - M_ROWS)):
                    if d < fx.Int32(BLOCK_K):
                        fx.ptr_store(
                            fx.BFloat16(0.0), lds_p + ((M_ROWS + r) * P_STRIDE + d)
                        )
                for r in range_constexpr(Q_ROWS - M_ROWS):
                    if d == fx.Int32(0):
                        fx.ptr_store(fx.Float32(0.0), lds_alpha + (M_ROWS + r))
            for it in range_constexpr(
                ((M_ROWS * S_STRIDE + BLOCK_TH - 1) // BLOCK_TH)
                if qk_abl == "nocombine"
                else 0
            ):
                idx = d + fx.Int32(it * BLOCK_TH)
                if idx < fx.Int32(M_ROWS * S_STRIDE):
                    fx.ptr_store(fx.Float32(0.0), lds_s + idx)
            if const_expr(fast_softmax):
                # m/l live in LDS for the whole tile loop, not in 512 copies of
                # the same two registers. Init m to -inf, per "Method traps".
                if d < fx.Int32(M_ROWS):
                    fx.ptr_store(c_neg_large, lds_m + d)
                    fx.ptr_store(fx.Float32(0.0), lds_l + d)

            # These three depend only on the tiled MMA, so they live outside the
            # tile loop -- q_resident calls _load_A before it.
            def _load_A(sA):
                frag_A = thr_mma.make_fragment_A(sA)
                fx.copy(
                    copy_atom,
                    thr_copy_A.partition_S(sA)[None, None, 0],
                    thr_copy_A.retile(frag_A)[None, None, 0],
                )
                return frag_A

            def _load_B(sB):
                frag_B = thr_mma.make_fragment_B(sB)
                fx.copy(
                    copy_atom,
                    thr_copy_B.partition_S(sB)[None, None, 0],
                    thr_copy_B.retile(frag_B)[None, None, 0],
                )
                return frag_B

            def _mma(frag_C, frag_A, frag_B):
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, 0],
                    frag_B[None, None, 0],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

            def _load_A_global(mt0, koff):
                """QK's A fragment straight from global Q, no LDS copy at all.

                Same partition-coordinate trick kv_single uses for B. Only
                affordable because q_resident makes this a once-per-workgroup
                cost, and it is what takes the workgroup under the LDS wall
                that has held every arm since B4 to one CTA per CU. Pad rows
                are masked here instead of being zeroed in LDS.
                """
                frag_A = thr_mma.make_fragment_A(sAdonor)
                for i in range_constexpr(NREG_A):
                    row = mt0 * MMA_M + fx.Int32(fx.get_scalar(aRow[i]))
                    col = fx.Int32(fx.get_scalar(aCol[i]))
                    live = row < fx.Int32(M_ROWS)
                    safe = live.select(row, fx.Int32(0))
                    off = (
                        (tok0 + safe % fx.Int32(Q_LEN)) * fx.Int32(q_stride_t)
                        + (fx.Int32(h) + safe // fx.Int32(Q_LEN)) * fx.Int32(q_stride_h)
                        + (koff + col)
                    )
                    frag_A[i] = live.select(load_bf16(q_buf, off), fx.Float32(0.0)).to(
                        fx.BFloat16
                    )
                return frag_A

            # q_resident: QK's A fragments are loop-invariant -- with qk_reuse a
            # wave owns one m_tile and a fixed set of k-chunks for the whole
            # tile loop -- so load them once here rather than re-reading Q out
            # of LDS on every one of the ~37 tiles a production kv_len gives.
            if const_expr(q_resident):
                # Q was staged column-per-thread, so the rows this wave wants
                # were written by other waves.
                if const_expr(stage_q):
                    gpu.barrier()
                grp0 = wave // fx.Int32(QK_DSPLIT)
                k_lo0 = (wave % fx.Int32(QK_DSPLIT)) * fx.Int32(CH_PER_SLICE)
                q_frags = []
                for it in range_constexpr(QK_PER_WAVE):
                    raw0 = grp0 + fx.Int32(it * QK_GROUPS)
                    mt0 = (raw0 < fx.Int32(QK_ITEMS)).select(raw0, fx.Int32(0))
                    for c in range_constexpr(CH_PER_SLICE):
                        koff0 = (k_lo0 + fx.Int32(c)) * MMA_K
                        q_frags.append(
                            _load_A_global(mt0, koff0)
                            if const_expr(q_global)
                            else _load_A(
                                fx.make_view(
                                    lds_q + (mt0 * (MMA_M * D) + koff0),
                                    fx.make_layout((MMA_M, MMA_K), (D, 1)),
                                )
                            )
                        )

            m_init = [fx.Float32(_NEG_LARGE) for _ in range_constexpr(Q_LEN)]
            l_init = [fx.Float32(0.0) for _ in range_constexpr(Q_LEN)]
            a_init = [fx.Float32(0.0) for _ in range_constexpr(N_ACC)]

            j_start = fx.Int64(tile_start)
            j_stop = fx.Int64(tile_end)
            j_step = fx.Int64(1)
            for j, state in range(
                j_start, j_stop, j_step, init=m_init + l_init + a_init
            ):
                m_i = list(state[0:Q_LEN])
                l_i = list(state[Q_LEN : 2 * Q_LEN])
                acc = list(state[2 * Q_LEN : 2 * Q_LEN + N_ACC])
                k_base = fx.Int32(j) * BLOCK_K
                if const_expr(two_phase):
                    # Phase A is the shared tail, phase B the windows' union.
                    # A runtime select, never a dynamic `if`: the rewriter runs
                    # an `if` body as a separate function, which cost 53 us per
                    # CTA the one time this kernel wrapped its body in one.
                    is_tail = fx.Int32(j) < tail_tiles
                    u_base = k_base - is_tail.select(fx.Int32(0), tail_tiles * BLOCK_K)
                else:
                    is_tail = None
                    u_base = k_base

                def row_bounds(qq):
                    """[lo, hi) of row qq's valid entries on this tile's axis.

                    One select chain, the same cost as the single-bound version
                    it replaces: the rest is arithmetic on qq.
                    """
                    if const_expr(not two_phase or tp_abl == "oldmask"):
                        return fx.Int32(0), dyn_pick(row_len, qq)
                    rl = dyn_pick(row_len, qq)
                    wl = _win_of(rl)
                    hi_u = qq + win_0
                    return (
                        is_tail.select(fx.Int32(0), hi_u - wl),
                        is_tail.select(rl - wl, hi_u),
                    )

                def row_ok(u, lo, hi):
                    # tp_abl='nolo' times the two-phase axis and index
                    # fetch without the interval mask; results are wrong by
                    # construction, only the time is readable.
                    if const_expr(two_phase and tp_abl == "allmask"):
                        return u < fx.Int32(-1)  # control: mask everything
                    if const_expr(two_phase and tp_abl == "nohi"):
                        return u >= lo
                    if const_expr(not two_phase or tp_abl == "nolo"):
                        return u < hi
                    # Not `(u >= lo) & (u < hi)`: a Boolean `&` here keeps only
                    # the second operand, so the lower bound silently vanished
                    # and the interval mask read exactly like no mask at all.
                    # Folding the lower bound into the value tested keeps one
                    # comparison and no conjunction.
                    return (u >= lo).select(u, hi) < hi

                # -- stage the KV tile, shared by the whole workgroup ----
                gpu.barrier()
                for rnd in range_constexpr(KV_ROUNDS if do_stage else 0):
                    # kv_vec=1 keeps the original one-column-per-thread map;
                    # above that a thread owns kv_vec contiguous columns and
                    # TH_PER_ROW threads share a row, so the row index (and its
                    # kv_indices load) becomes per-round instead of per-column.
                    kk = (
                        fx.Int32(rnd * ROWS_PER_ROUND) + (d // fx.Int32(TH_PER_ROW))
                        if const_expr(kv_vec > 1)
                        else fx.Int32(rnd)
                    )
                    cbase = (
                        (d % fx.Int32(TH_PER_ROW)) * kv_vec
                        if const_expr(kv_vec > 1)
                        else d
                    )
                    if const_expr(two_phase):
                        u = u_base + kk
                        a_union = (u < win_0).select(win0_base + u, winL_base + u)
                        in_tile = is_tail.select(u < tail_L, u < union_len)
                        slot_off = in_tile.select(
                            is_tail.select(tail_base + u, a_union), fx.Int32(0)
                        )
                    else:
                        pos = k_base + kk
                        in_tile = pos < kv_len_max
                        slot_off = in_tile.select(kv_start + pos, fx.Int32(0))
                    slot = load_i32(idx_buf, slot_off)
                    goff = slot * fx.Int32(kv_stride_n) + cbase
                    vals = load_bf16_vec(kv_buf, goff)
                    for i in range_constexpr(kv_vec):
                        fx.ptr_store(vals[i], lds_kv + (kk * KV_STRIDE + cbase + i))
                        if const_expr(stage_kvt):
                            # PV contracts over kv, so it needs V as (d, kv) rows.
                            fx.ptr_store(
                                vals[i], lds_kvt + ((cbase + i) * BLOCK_K + kk)
                            )
                gpu.barrier()

                # -- scores: DOT_SPLIT threads per (draft token, kv position),
                # each covering D // DOT_SPLIT of the contraction, then a
                # DOT_SPLIT-way combine. Keeping the contraction a constexpr
                # unroll avoids a second carried-state loop inside this one.
                def _dot_partials():
                    pair = d // DOT_SPLIT
                    sub = d % DOT_SPLIT
                    qb = (pair // BLOCK_K) * D
                    kb = (pair % BLOCK_K) * KV_STRIDE
                    dbase = sub * DOT_CHUNK
                    part = fx.Float32(0.0)
                    for t in range_constexpr(DOT_CHUNK):
                        di = dbase + t
                        part = part + lds_ld(lds_q, qb + di) * lds_ld(lds_kv, kb + di)
                    fx.ptr_store(part, lds_s4 + d)

                def _mma_step(frag_C, sA, sB):
                    _mma(frag_C, _load_A(sA), _load_B(sB))

                def _qk_mfma_reuse():
                    """One m_tile and all NT_N n-tiles per wave.

                    The A fragment is loaded once per k-chunk and feeds NT_N
                    MFMAs, which is the operand reuse the one-tile-per-wave
                    split above has none of.
                    """
                    grp = wave // fx.Int32(QK_DSPLIT)
                    sub = wave % fx.Int32(QK_DSPLIT)
                    k_lo = sub * fx.Int32(CH_PER_SLICE)
                    for it in range_constexpr(QK_PER_WAVE):
                        raw = grp + fx.Int32(it * QK_GROUPS)
                        live = raw < fx.Int32(QK_ITEMS)
                        mt = live.select(raw, fx.Int32(0))
                        frags = [
                            thr_mma.make_fragment_C(sCdonor)
                            for _ in range_constexpr(NT_N)
                        ]
                        for f in frags:
                            f.fill(0.0)
                        for c in range_constexpr(CH_PER_SLICE):
                            koff = (k_lo + fx.Int32(c)) * MMA_K
                            frag_A = (
                                q_frags[it * CH_PER_SLICE + c]
                                if const_expr(q_resident)
                                else _load_A(
                                    fx.make_view(
                                        lds_q + (mt * (MMA_M * D) + koff),
                                        fx.make_layout((MMA_M, MMA_K), (D, 1)),
                                    )
                                )
                            )
                            for nt in range_constexpr(NT_N):
                                _mma(
                                    frags[nt],
                                    frag_A,
                                    _load_B(
                                        fx.make_view(
                                            lds_kv + (nt * (MMA_N * KV_STRIDE) + koff),
                                            fx.make_layout(
                                                (MMA_N, MMA_K), (KV_STRIDE, 1)
                                            ),
                                        )
                                    ),
                                )
                        for nt in range_constexpr(NT_N):
                            for i in range_constexpr(NREG):
                                row = fx.Int32(fx.get_scalar(cRow[i])) + mt * MMA_M
                                col = fx.Int32(fx.get_scalar(cCol[i])) + nt * MMA_N
                                flat = row * fx.Int32(BLOCK_K) + col
                                flat_s = row * fx.Int32(S_STRIDE) + col
                                val = (
                                    frags[nt][i] * qk_scale
                                    if const_expr(QK_DSPLIT == 1)
                                    else frags[nt][i]
                                )
                                dst = (
                                    lds_s + flat_s
                                    if const_expr(QK_DSPLIT == 1)
                                    else lds_s4 + (sub * (M_ROWS * BLOCK_K) + flat)
                                )
                                if live & (row < fx.Int32(M_ROWS)):
                                    fx.ptr_store(val, dst)

                def _qk_mfma():
                    """One (m_tile, n_tile) score tile per wave.

                    With QK_DSPLIT == 1 there is no cross-wave partial at all:
                    the wave owns the whole D contraction for its tile and
                    writes the scaled score straight into lds_s, which deletes
                    both the s4 buffer and the reduction pass.
                    """
                    grp = wave // fx.Int32(QK_DSPLIT)
                    sub = wave % fx.Int32(QK_DSPLIT)
                    k_lo = sub * fx.Int32(CH_PER_SLICE)
                    for it in range_constexpr(QK_PER_WAVE):
                        raw = grp + fx.Int32(it * QK_GROUPS)
                        live = raw < fx.Int32(QK_ITEMS)
                        item = live.select(raw, fx.Int32(0))
                        mt = item // fx.Int32(NT_N)
                        nt = item % fx.Int32(NT_N)
                        frags = [
                            thr_mma.make_fragment_C(sCdonor)
                            for _ in range_constexpr(QK_ACCS)
                        ]
                        for f in frags:
                            f.fill(0.0)
                        for c in range_constexpr(CH_PER_SLICE):
                            koff = (k_lo + fx.Int32(c)) * MMA_K
                            _mma_step(
                                frags[c % QK_ACCS],
                                fx.make_view(
                                    lds_q + (mt * (MMA_M * D) + koff),
                                    fx.make_layout((MMA_M, MMA_K), (D, 1)),
                                ),
                                fx.make_view(
                                    lds_kv + (nt * (MMA_N * KV_STRIDE) + koff),
                                    fx.make_layout((MMA_N, MMA_K), (KV_STRIDE, 1)),
                                ),
                            )
                        # ptr_store, not sScore[r, c] = v: the AST rewriter reads a
                        # subscript assignment inside a dynamic if as a variable
                        # assignment and rejects it as an scf.if result.
                        for i in range_constexpr(NREG):
                            row = fx.Int32(fx.get_scalar(cRow[i])) + mt * MMA_M
                            col = fx.Int32(fx.get_scalar(cCol[i])) + nt * MMA_N
                            flat = row * fx.Int32(BLOCK_K) + col
                            flat_s = row * fx.Int32(S_STRIDE) + col
                            tot = frags[0][i]
                            for a in range_constexpr(QK_ACCS - 1):
                                tot = tot + frags[a + 1][i]
                            val = tot * qk_scale if const_expr(QK_DSPLIT == 1) else tot
                            dst = (
                                lds_s + flat_s
                                if const_expr(QK_DSPLIT == 1)
                                else lds_s4 + (sub * (M_ROWS * BLOCK_K) + flat)
                            )
                            if live & (row < fx.Int32(M_ROWS)):
                                fx.ptr_store(val, dst)

                if const_expr(ablate not in ("empty", "stage")):
                    if const_expr(mfma_qk and qk_reuse):
                        _qk_mfma_reuse()
                    elif const_expr(mfma_qk):
                        _qk_mfma()
                    elif d < fx.Int32(N_PAIRS * DOT_SPLIT):
                        _dot_partials()
                    gpu.barrier()

                def _combine(idx):
                    if const_expr(mfma_qk):
                        # Pair idx = (row, kk) is element idx of every wave's
                        # M_ROWS x BLOCK_K partial score tile.
                        base, stride, n_part = idx, M_ROWS * BLOCK_K, QK_DSPLIT
                    else:
                        base, stride, n_part = idx * DOT_SPLIT, 1, DOT_SPLIT
                    tot = fx.ptr_load(lds_s4 + base)
                    for sub in range_constexpr(n_part - 1):
                        tot = tot + fx.ptr_load(lds_s4 + (base + (sub + 1) * stride))
                    fx.ptr_store(
                        tot * qk_scale,
                        lds_s
                        + (
                            (
                                (idx // fx.Int32(BLOCK_K)) * S_STRIDE
                                + idx % fx.Int32(BLOCK_K)
                            )
                            if const_expr(s_pad)
                            else idx
                        ),
                    )

                # With QK_DSPLIT == 1 the MFMA already wrote scaled scores
                # into lds_s, so there is nothing to reduce.
                # qk_abl="nocombine": skip the cross-wave score reduction
                # without touching QK's wave assignment, so its cost separates
                # from the idle-wave cost that SGLANG_MLA_FLYDSL_QKDSPLIT=1
                # bundles with it. Scores read as the zero-fill below, so the
                # arm is WRONG BY DESIGN -- time is readable, relL2 is not.
                if const_expr(
                    ablate not in ("empty", "stage")
                    and not (mfma_qk and QK_DSPLIT == 1)
                    and qk_abl != "nocombine"
                ):
                    for it in range_constexpr(PAIRS_PER_TH):
                        idx = d + fx.Int32(it * BLOCK_TH)
                        if idx < fx.Int32(N_PAIRS):
                            _combine(idx)
                    gpu.barrier()

                # -- online softmax + PV, replicated per thread ----------
                def _softmax_core():
                    new_m, new_l = [], []
                    alphas, pvals = [], []
                    for q in range_constexpr(Q_LEN):
                        sv = [
                            fx.ptr_load(lds_s + (q * S_STRIDE + kk))
                            for kk in range_constexpr(BLOCK_K)
                        ]
                        valid = [
                            (k_base + kk) < row_len[q]
                            for kk in range_constexpr(BLOCK_K)
                        ]
                        sv = [
                            valid[kk].select(sv[kk], c_neg_large)
                            for kk in range_constexpr(BLOCK_K)
                        ]
                        m_blk = sv[0]
                        for kk in range_constexpr(BLOCK_K - 1):
                            m_blk = (sv[kk + 1] > m_blk).select(sv[kk + 1], m_blk)
                        m_new = (m_blk > m_i[q]).select(m_blk, m_i[q])
                        alpha = fexp2(m_i[q] - m_new)
                        l_new = l_i[q] * alpha
                        p_q = [
                            valid[kk].select(fexp2(sv[kk] - m_new), fx.Float32(0.0))
                            for kk in range_constexpr(BLOCK_K)
                        ]
                        for kk in range_constexpr(BLOCK_K):
                            l_new = l_new + p_q[kk]
                        new_m.append(m_new)
                        new_l.append(l_new)
                        alphas.append(alpha)
                        pvals.append(p_q)
                    return new_m, new_l, alphas, pvals

                def _pv_fma(alphas, pvals):
                    out = []
                    for q in range_constexpr(Q_LEN):
                        a_new = acc[q] * alphas[q]
                        for kk in range_constexpr(BLOCK_K):
                            a_new = a_new + pvals[q][kk] * lds_ld(
                                lds_kv, kk * KV_STRIDE + d
                            )
                        out.append(a_new)
                    return out

                def _pv_mfma(alphas, pvals):
                    """acc[MMA_M, dtile] = acc * alpha + P @ KVT[dtile, :].T.

                    P is rounded to bf16 to feed the atom -- a deliberate dtype
                    choice, the same one the shipped Triton kernel makes.
                    """
                    # Publish P and the rescale factors: every thread holds all
                    # of both (the softmax is replicated), so row q is written
                    # by the one thread whose id is q. This serialises 7 threads
                    # x BLOCK_K stores behind a barrier and is the reason
                    # fast_softmax exists.
                    for q in range_constexpr(Q_LEN):
                        if d == fx.Int32(q):
                            for kk in range_constexpr(BLOCK_K):
                                fx.ptr_store(
                                    pvals[q][kk].to(fx.BFloat16),
                                    lds_p + (q * P_STRIDE + kk),
                                )
                            fx.ptr_store(alphas[q], lds_alpha + q)
                    gpu.barrier()
                    return _pv_atoms()

                def _pv_load_B(dt):
                    """B[d, kv] for d-tile dt, from whichever KV layout is staged."""
                    frag_B = thr_mma.make_fragment_B(sBdonor)
                    if const_expr(kv_single):
                        # (kv, d) tile: element (row, col) of the fragment is
                        # kv=col, d=dt*MMA_N+row, i.e. lds_kv[col * KV_STRIDE + d].
                        for i in range_constexpr(NREG_B):
                            row = fx.Int32(fx.get_scalar(bRow[i]))
                            col = fx.Int32(fx.get_scalar(bCol[i]))
                            frag_B[i] = fx.ptr_load(
                                lds_kv + (col * KV_STRIDE + dt * MMA_N + row)
                            )
                        return frag_B
                    return _load_B(
                        fx.make_view(
                            lds_kvt + dt * (MMA_N * BLOCK_K),
                            fx.make_ordered_layout((MMA_N, BLOCK_K), (1, 0)),
                        )
                    )

                def _pv_atoms():
                    """The PV atoms themselves: P and alpha are already in LDS.

                    The wave's NT_D_PER_WAVE B fragments are loaded once and
                    reused across all M_TILES, and A once per m_tile -- the
                    same operand reuse qk_reuse gave QK. Append order stays
                    (m_tile, d_tile) so the accumulator indexing is unchanged.
                    """
                    out = []
                    fragB = [
                        _pv_load_B(wave * NT_D_PER_WAVE + t)
                        for t in range_constexpr(NT_D_PER_WAVE)
                    ]
                    for mt in range_constexpr(M_TILES):
                        frag_A = _load_A(
                            fx.make_view(
                                lds_p + mt * (MMA_M * P_STRIDE),
                                fx.make_layout((MMA_M, BLOCK_K), (P_STRIDE, 1)),
                            )
                        )
                        # The accumulator row a register holds depends only on
                        # (mt, i), so read alpha once per m_tile instead of
                        # once per (m_tile, d_tile, register): NREG loads per
                        # tile instead of NREG * NT_D_PER_WAVE.
                        alpha_scales = [
                            fx.ptr_load(
                                lds_alpha
                                + (fx.Int32(fx.get_scalar(cRow[i])) + mt * MMA_M)
                            )
                            for i in range_constexpr(NREG)
                        ]
                        for t in range_constexpr(NT_D_PER_WAVE):
                            frag_C = thr_mma.make_fragment_C(sCdonor)
                            for i in range_constexpr(NREG):
                                frag_C[i] = (
                                    acc[(mt * NT_D_PER_WAVE + t) * NREG + i]
                                    * alpha_scales[i]
                                )
                            _mma(frag_C, frag_A, fragB[t])
                            for i in range_constexpr(NREG):
                                out.append(frag_C[i])
                    return out

                def _fast_softmax_pv():
                    """Softmax with one thread per (draft token, kv) pair.

                    Row stats are done by Q_LEN threads reading their own row of
                    lds_s; every other thread does exactly one exp2. Total exp2
                    per tile is ~2 x N_PAIRS instead of BLOCK_TH x N_PAIRS.
                    """
                    if const_expr(STAT_SPLIT > 1):
                        # STAT_SPLIT consecutive threads share a row, so all
                        # BLOCK_TH threads work and the dependent chain is
                        # STAT_CH + log2(STAT_SPLIT) deep instead of 2 * BLOCK_K.
                        c_row = d // fx.Int32(STAT_SPLIT)
                        c_ch = d % fx.Int32(STAT_SPLIT)
                        c_row_ld = (
                            (c_row < fx.Int32(M_ROWS)).select(c_row, fx.Int32(0))
                            if STAT_RAGGED
                            else c_row
                        )
                        c_lo, c_hi = row_bounds(c_row % fx.Int32(Q_LEN))
                        c_base = c_row_ld * fx.Int32(S_STRIDE) + c_ch * fx.Int32(
                            STAT_CH
                        )
                        u_c = u_base + c_ch * fx.Int32(STAT_CH)
                        sv_c = []
                        m_c = c_neg_large
                        for kk in range_constexpr(STAT_CH):
                            ok = row_ok(u_c + fx.Int32(kk), c_lo, c_hi)
                            sv = ok.select(
                                fx.ptr_load(lds_s + (c_base + fx.Int32(kk))),
                                c_neg_large,
                            )
                            sv_c.append(sv)
                            m_c = (sv > m_c).select(sv, m_c)
                        if const_expr(coop_xlane):
                            # Reduce the max across the group first, so the sum
                            # needs no per-thread rescale and the whole reduction
                            # costs STAT_CH exp2, 2 * log2(STAT_SPLIT) DPP movs
                            # and no barrier.
                            m_blk = m_c
                            for st in range_constexpr(len(XL_STEPS)):
                                o = dpp_f32(m_blk, XL_STEPS[st])
                                m_blk = (o > m_blk).select(o, m_blk)
                            # Keep the terms: merge_p turns them into P.
                            e_c = [
                                (sv_c[kk] > c_neg_large).select(
                                    # Masked to 0, never exp2(-large - -large) = 1.
                                    fexp2(sv_c[kk] - m_blk),
                                    fx.Float32(0.0),
                                )
                                for kk in range_constexpr(STAT_CH)
                            ]
                            l_blk = e_c[0]
                            for kk in range_constexpr(STAT_CH - 1):
                                l_blk = l_blk + e_c[kk + 1]
                            for st in range_constexpr(len(XL_STEPS)):
                                l_blk = l_blk + dpp_f32(l_blk, XL_STEPS[st])
                            # Every lane derives m_new itself. The read has to
                            # happen in all STAT_SPLIT lanes before the c_ch == 0
                            # lane overwrites lds_m, which lockstep gives us only
                            # because it is a separate, earlier instruction.
                            m_old = fx.ptr_load(lds_m + c_row_ld)
                            m_new = (m_blk > m_old).select(m_blk, m_old)
                            alpha = fexp2(m_old - m_new)
                            corr = fexp2(m_blk - m_new)
                            pv_c = (
                                [e_c[kk] * corr for kk in range_constexpr(STAT_CH)]
                                if const_expr(MERGE_P)
                                else []
                            )
                            if c_row < fx.Int32(M_ROWS):
                                if c_ch == fx.Int32(0):
                                    l_new = (
                                        fx.ptr_load(lds_l + c_row) * alpha
                                        + l_blk * corr
                                    )
                                    fx.ptr_store(m_new, lds_m + c_row)
                                    fx.ptr_store(l_new, lds_l + c_row)
                                    fx.ptr_store(alpha, lds_alpha + c_row)
                        else:
                            # Each thread reduces against its own local max, so
                            # the combine stays associative through LDS scratch:
                            # m = max(m_c), l = sum(l_c * exp2(m_c - m)).
                            l_c = fx.Float32(0.0)
                            for kk in range_constexpr(STAT_CH):
                                l_c = l_c + (sv_c[kk] > c_neg_large).select(
                                    fexp2(sv_c[kk] - m_c), fx.Float32(0.0)
                                )
                            fx.ptr_store(m_c, lds_stat + d)
                            fx.ptr_store(l_c, lds_stat + (fx.Int32(BLOCK_TH) + d))
                            gpu.barrier()
                            if d < fx.Int32(M_ROWS):
                                base = d * fx.Int32(STAT_SPLIT)
                                mm = [
                                    fx.ptr_load(lds_stat + (base + fx.Int32(c)))
                                    for c in range_constexpr(STAT_SPLIT)
                                ]
                                m_lds = mm[0]
                                for c in range_constexpr(STAT_SPLIT - 1):
                                    m_lds = (mm[c + 1] > m_lds).select(mm[c + 1], m_lds)
                                m_lold = fx.ptr_load(lds_m + d)
                                m_lnew = (m_lds > m_lold).select(m_lds, m_lold)
                                alpha_lds = fexp2(m_lold - m_lnew)
                                l_lnew = fx.ptr_load(lds_l + d) * alpha_lds
                                for c in range_constexpr(STAT_SPLIT):
                                    l_c2 = fx.ptr_load(
                                        lds_stat
                                        + (fx.Int32(BLOCK_TH) + base + fx.Int32(c))
                                    )
                                    l_lnew = l_lnew + l_c2 * fexp2(mm[c] - m_lnew)
                                fx.ptr_store(m_lnew, lds_m + d)
                                fx.ptr_store(l_lnew, lds_l + d)
                                fx.ptr_store(alpha_lds, lds_alpha + d)
                    else:
                        if d < fx.Int32(M_ROWS):
                            # Row d is (head d // Q_LEN, draft token d % Q_LEN);
                            # masking depends only on the token.
                            r_lo, r_hi = row_bounds(d % fx.Int32(Q_LEN))
                            row0 = d * fx.Int32(S_STRIDE)
                            m_o1 = fx.ptr_load(lds_m + d)
                            m_b1 = c_neg_large
                            STAT_K = 1 if qk_abl == "nostats" else BLOCK_K
                            for kk in range_constexpr(STAT_K):
                                ok = row_ok(u_base + fx.Int32(kk), r_lo, r_hi)
                                sv = ok.select(
                                    fx.ptr_load(lds_s + (row0 + kk)), c_neg_large
                                )
                                m_b1 = (sv > m_b1).select(sv, m_b1)
                            m_n1 = (m_b1 > m_o1).select(m_b1, m_o1)
                            alp1 = fexp2(m_o1 - m_n1)
                            l_n1 = fx.ptr_load(lds_l + d) * alp1
                            for kk in range_constexpr(STAT_K):
                                ok = row_ok(u_base + fx.Int32(kk), r_lo, r_hi)
                                # Masked to 0, never exp2(-large - -large) = 1.
                                l_n1 = l_n1 + ok.select(
                                    fexp2(fx.ptr_load(lds_s + (row0 + kk)) - m_n1),
                                    fx.Float32(0.0),
                                )
                            fx.ptr_store(m_n1, lds_m + d)
                            fx.ptr_store(l_n1, lds_l + d)
                            fx.ptr_store(alp1, lds_alpha + d)
                    gpu.barrier()

                    # alias_p puts p on top of s, so every thread's s reads have
                    # to be done -- by every thread -- before any p store. Hold
                    # the results in registers across a barrier; without the
                    # alias this is the same code with the barrier skipped.
                    pv = []
                    paddr = []
                    for it in range_constexpr(0 if MERGE_P else PAIRS_PER_TH):
                        idx = d + fx.Int32(it * BLOCK_TH)
                        live = idx < fx.Int32(N_PAIRS)
                        safe = live.select(idx, fx.Int32(0))
                        r_i = safe // fx.Int32(BLOCK_K)
                        p_lo, p_hi = row_bounds(r_i % fx.Int32(Q_LEN))
                        ok = row_ok(u_base + (safe % fx.Int32(BLOCK_K)), p_lo, p_hi)
                        paddr.append(
                            (r_i * P_STRIDE + safe % fx.Int32(BLOCK_K))
                            if const_expr(p_pad)
                            else idx
                        )
                        pv.append(
                            ok.select(
                                fexp2(
                                    fx.ptr_load(
                                        lds_s
                                        + (
                                            (r_i * S_STRIDE + safe % fx.Int32(BLOCK_K))
                                            if const_expr(s_pad)
                                            else safe
                                        )
                                    )
                                    - fx.ptr_load(lds_m + r_i)
                                ),
                                fx.Float32(0.0),
                            )
                        )
                    # The merged path read lds_s exactly once, at the top, so
                    # the barrier above already separates those reads from these
                    # writes -- alias_p's second barrier is only needed when the
                    # P pass re-reads lds_s.
                    if const_expr(alias_p and not MERGE_P):
                        gpu.barrier()
                    for it in range_constexpr(0 if MERGE_P else PAIRS_PER_TH):
                        idx = d + fx.Int32(it * BLOCK_TH)
                        if idx < fx.Int32(N_PAIRS):
                            fx.ptr_store(pv[it].to(fx.BFloat16), lds_p + paddr[it])
                    if const_expr(MERGE_P):
                        if c_row < fx.Int32(M_ROWS):
                            p_base = c_row * fx.Int32(P_STRIDE) + c_ch * fx.Int32(
                                STAT_CH
                            )
                            for kk in range_constexpr(STAT_CH):
                                fx.ptr_store(
                                    pv_c[kk].to(fx.BFloat16),
                                    lds_p + (p_base + fx.Int32(kk)),
                                )
                    if const_expr(alias_p):
                        for r in range_constexpr(Q_ROWS - M_ROWS):
                            if d < fx.Int32(BLOCK_K):
                                fx.ptr_store(
                                    fx.BFloat16(0.0),
                                    lds_p + ((M_ROWS + r) * P_STRIDE + d),
                                )
                    gpu.barrier()
                    return list(m_i), list(l_i), _pv_atoms()

                def _abl_cheap():
                    """Ablation tail: consume just enough of what was computed
                    that none of it can be dead-code-eliminated, and nothing
                    more, so the time difference between levels is the work."""
                    a = (
                        [
                            acc[i] + kv_len_max.to(fx.Float32)
                            for i in range_constexpr(N_ACC)
                        ]
                        if const_expr(ablate in ("empty", "gather"))
                        else [
                            acc[i]
                            + lds_ld(lds_q, (i % M_ROWS) * D + d)
                            + lds_ld(lds_kv, (i % BLOCK_K) * KV_STRIDE + d)
                            for i in range_constexpr(N_ACC)
                        ]
                        if const_expr(ablate == "stage")
                        else [
                            acc[i] + fx.ptr_load(lds_s + ((i % M_ROWS) * S_STRIDE))
                            for i in range_constexpr(N_ACC)
                        ]
                    )
                    return list(m_i), list(l_i), a

                def _softmax_then_pv():
                    nm, nl, alphas, pvals = _softmax_core()
                    na = (
                        [acc[q] + alphas[q] for q in range_constexpr(Q_LEN)]
                        if const_expr(ablate == "softmax")
                        else (
                            _pv_mfma(alphas, pvals)
                            if const_expr(mfma_pv)
                            else _pv_fma(alphas, pvals)
                        )
                    )
                    return nm, nl, na

                new_m, new_l, new_a = (
                    _abl_cheap()
                    if const_expr(ablate in ("empty", "gather", "stage", "qk"))
                    else (
                        _fast_softmax_pv()
                        if const_expr(fast_softmax)
                        else _softmax_then_pv()
                    )
                )

                final = yield new_m + new_l + new_a

            a_f = list(final[2 * Q_LEN : 2 * Q_LEN + N_ACC])

            m_buf = ptr_buf_tensor(fx.get_iter(m_out), fx.Float32)
            l_buf = ptr_buf_tensor(fx.get_iter(l_out), fx.Float32)
            a_buf = ptr_buf_tensor(fx.get_iter(acc_out), fx.Float32)

            def ml_offset(hh, qq):
                return (
                    (tok0 + qq) * fx.Int32(mp_t)
                    + fx.Int32(pid_k) * fx.Int32(mp_k)
                    + (fx.Int32(h) + hh) * fx.Int32(mp_h)
                )

            if const_expr(fast_softmax):
                # m and l are already in LDS, one entry per row, so each row
                # writes its own instead of serialising through thread 0.
                if d < fx.Int32(M_ROWS):
                    off = ml_offset(d // fx.Int32(Q_LEN), d % fx.Int32(Q_LEN))
                    fx.add_offset(fx.get_iter(m_buf), off).store(fx.ptr_load(lds_m + d))
                    fx.add_offset(fx.get_iter(l_buf), off).store(fx.ptr_load(lds_l + d))
            else:
                m_f = list(final[0:Q_LEN])
                l_f = list(final[Q_LEN : 2 * Q_LEN])
                for q in range_constexpr(Q_LEN):
                    off = ml_offset(fx.Int32(0), fx.Int32(q))
                    if d == fx.Int32(0):
                        fx.add_offset(fx.get_iter(m_buf), off).store(m_f[q])
                        fx.add_offset(fx.get_iter(l_buf), off).store(l_f[q])
                    if const_expr(not mfma_pv):
                        a_off = (
                            (tok0 + q) * fx.Int32(ap_t)
                            + fx.Int32(pid_k) * fx.Int32(ap_k)
                            + fx.Int32(h) * fx.Int32(ap_h)
                            + d
                        )
                        fx.add_offset(fx.get_iter(a_buf), a_off).store(a_f[q % N_ACC])

            if const_expr(mfma_pv):
                k_off = fx.Int32(pid_k) * fx.Int32(ap_k)
                for mt, t in itertools.product(
                    range_constexpr(M_TILES), range_constexpr(NT_D_PER_WAVE)
                ):
                    dt = wave * NT_D_PER_WAVE + t
                    for i in range_constexpr(NREG):
                        row = fx.Int32(fx.get_scalar(cRow[i])) + mt * MMA_M
                        col = fx.Int32(fx.get_scalar(cCol[i]))
                        if row < fx.Int32(M_ROWS):
                            a_off = (
                                (tok0 + row % fx.Int32(Q_LEN)) * fx.Int32(ap_t)
                                + k_off
                                + (fx.Int32(h) + row // fx.Int32(Q_LEN))
                                * fx.Int32(ap_h)
                                + dt * MMA_N
                                + col
                            )
                            fx.add_offset(fx.get_iter(a_buf), a_off).store(
                                # % N_ACC so the ablation arms, whose tail carries
                                # a fixed 4-register accumulator, can run through
                                # the real epilogue instead of the harness's --
                                # that is what lets the ladder be measured in the
                                # geometry that actually ships. Identity when
                                # ablate is None.
                                a_f[((mt * NT_D_PER_WAVE + t) * NREG + i) % N_ACC]
                            )

            if const_expr(dbg_scores):
                # Debug hook: overwrite acc with the last tile's scaled scores,
                # so two kernels at the same BLOCK_K can be diffed per (draft
                # token, kv) pair without needing a torch score reference.
                if d < fx.Int32(N_PAIRS):
                    dbg_off = (
                        tok0 * fx.Int32(ap_t)
                        + fx.Int32(pid_k) * fx.Int32(ap_k)
                        + fx.Int32(h) * fx.Int32(ap_h)
                        + d
                    )
                    fx.add_offset(fx.get_iter(a_buf), dbg_off).store(
                        fx.ptr_load(lds_s + d)
                    )

        # Always run the epilogue, even for a split with no tiles. The shipped
        # kernel may skip the write because its reduce recomputes the same
        # `act_num_segments` from the same per-TOKEN kv_len and masks the slot
        # out. This kernel plans per REQUEST (kv_len_max), so the two disagree
        # near a length boundary -- e.g. Lmax=129, L_i=123, SPLITS=4: we cover
        # the range in 3 splits and the reduce reads 4. The partial buffers are
        # reused across layers, so the fourth read would be stale garbage.
        # A zero-tile split writes m=-inf, l=0, acc=0, which merges to nothing.
        _body()

    @flyc.jit
    def launch(
        q_in: fx.Tensor,
        q_stride_t: Int32,
        q_stride_h: Int32,
        kv_in: fx.Tensor,
        kv_stride_n: Int32,
        kv_indices: fx.Tensor,
        req_indptr: fx.Tensor,
        tok_len: fx.Tensor,
        m_out: fx.Tensor,
        l_out: fx.Tensor,
        acc_out: fx.Tensor,
        mp_t: Int32,
        mp_k: Int32,
        mp_h: Int32,
        ap_t: Int32,
        ap_k: Int32,
        ap_h: Int32,
        qk_scale: fx.Float32,
        R: Int32,
        H: Int32,
        stream: fx.Stream,
    ):
        k = kernel(
            q_in,
            q_stride_t,
            q_stride_h,
            kv_in,
            kv_stride_n,
            kv_indices,
            req_indptr,
            tok_len,
            m_out,
            l_out,
            acc_out,
            mp_t,
            mp_k,
            mp_h,
            ap_t,
            ap_k,
            ap_h,
            qk_scale,
        )
        k.launch(
            grid=(fx.Int64(R), fx.Int64(H), fx.Int64(SPLITS)),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    launch._h_per_wg = h_per_wg
    return launch


def ref_torch(dd, scale):
    """Independent fp32 reference: exact base-2 softmax attention per token.

    Not derived from either kernel, so it can adjudicate between them -- the
    shipped Triton kernel rounds P to bf16 before PV, this does not.
    """
    q = dd["q"].float()
    kv = dd["kv"].float()
    indptr = dd["tok_indptr"].tolist()
    idx = dd["tok_idx"]
    out = torch.empty_like(q)
    for n in range(dd["N"]):
        k = kv[idx[indptr[n] : indptr[n + 1]].long()]  # [L, D]
        s = (q[n] @ k.T) * scale  # [H, L]
        p = torch.exp2(s - s.amax(dim=-1, keepdim=True))
        out[n] = (p @ k) / p.sum(dim=-1, keepdim=True)
    return out


def main():
    from mla_qfold_proto import SPLITS, H, build, merge, partials, run_shipped, timeit

    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    dd = build(bs)
    for a in sys.argv:
        if a.startswith("--clamp-kv="):
            dd["tok_len"] = dd["tok_len"].clamp(max=int(a.split("=")[1]))
    D = dd["q"].shape[-1]

    variants = {}
    if "--mfma" in sys.argv:
        variants["A  K32   "] = build_kernel(D=D, SPLITS=SPLITS, BLOCK_K=32)
        variants["B1 K16   "] = build_kernel(D=D, SPLITS=SPLITS, mfma_qk=True)
        variants["B1 K32   "] = build_kernel(
            D=D, SPLITS=SPLITS, BLOCK_K=32, mfma_qk=True
        )
        variants["B2 K32   "] = build_kernel(
            D=D, SPLITS=SPLITS, BLOCK_K=32, mfma_qk=True, mfma_pv=True
        )
        b3 = dict(
            D=D,
            SPLITS=SPLITS,
            BLOCK_K=32,
            mfma_qk=True,
            mfma_pv=True,
            fast_softmax=True,
        )
        variants["B3 K32 fs"] = build_kernel(**b3)
        for hw in (4, 8):
            variants[f"B5 hw={hw:<2d} "] = build_kernel(**b3, h_per_wg=hw)
            variants[f"B6 hw={hw:<2d} "] = build_kernel(
                **b3, h_per_wg=hw, qk_reuse=True
            )
        # B7: the two LDS savings that let h_per_wg=16 fit (step 5). Built at
        # hw=8 too, where the budget was never the constraint, so their own
        # cost is separable from the win at 16.
        b7 = dict(b3, qk_reuse=True, kv_single=True, alias_p=True)
        variants["B7 hw=8  "] = build_kernel(**b7, h_per_wg=8)
        variants["B7 hw=16 "] = build_kernel(**b7, h_per_wg=16)
        # B8: B7 + the loop-invariant QK A fragments hoisted into registers.
        # One variable against B7; if it spills instead, the time goes up.
        variants["B8 hw=8  "] = build_kernel(**b7, h_per_wg=8, q_resident=True)
        variants["B8 hw=16 "] = build_kernel(**b7, h_per_wg=16, q_resident=True)
        # B9: B8 + Q out of LDS entirely, which is the occupancy lever.
        b9 = dict(b7, q_resident=True, q_global=True)
        variants["B9 hw=8  "] = build_kernel(**b9, h_per_wg=8)
        variants["B9 hw=4  "] = build_kernel(**b9, h_per_wg=4)
        # B10: B8 + the per-tile buffers laid over the dead Q tile, which is
        # what takes the workgroup off the one-CTA-per-CU LDS wall.
        b10 = dict(b7, q_resident=True, lds_pool=True)
        variants["B10 hw=8 "] = build_kernel(**b10, h_per_wg=8)
        variants["B10 hw=4 "] = build_kernel(**b10, h_per_wg=4)
        # B10s: the pool without the overlap -- no LDS saved, so it isolates
        # the cost of pooling from the value of the second resident CTA.
        variants["B10s hw=8"] = build_kernel(**b10, h_per_wg=8, pool_split=True)
        # Overlapped, but padded back over the 2-CTA/CU LDS threshold.
        variants["B10p hw=8"] = build_kernel(**b10, h_per_wg=8, pool_pad=12288)
        # B11: B8 + 128-bit KV staging. One variable against B8.
        b11 = dict(b7, q_resident=True, kv_vec=8)
        variants["B11 hw=8 "] = build_kernel(**b11, h_per_wg=8)
        variants["B11 hw=16"] = build_kernel(**b11, h_per_wg=16)
        # B12: B11 + KV row-stride padding, against the 16-way bank conflict
        # rocprofv3 found on the QK B-fragment reads.
        variants["B12 hw=16"] = build_kernel(**b11, h_per_wg=16, kv_pad=8)
        variants["B12 hw=8 "] = build_kernel(**b11, h_per_wg=8, kv_pad=8)
        # hw=16 is at the 160 KB LDS wall: kv_pad=8 fits, 16 already launches
        # with hipErrorIllegalState. The sweep therefore runs at hw=8, where
        # Q is 64 KB smaller, to see whether the residual 11 % conflict is
        # worth chasing with a zero-LDS swizzle at hw=16.
        for pad in (16, 32):
            variants[f"B12p{pad:<2d} hw8 "] = build_kernel(
                **b11, h_per_wg=8, kv_pad=pad
            )
        # B14: the same padding on the P tile, aimed at the softmax+PV block
        # that is 49 % of the kernel once kv_pad has fixed QK. Free in LDS:
        # alias_p already puts p inside the larger s region.
        # s_pad is a measured regression and hw=16 cannot afford it alongside
        # kv_pad anyway (hipErrorIllegalState at ~160 KB); see the flag's
        # comment. Left buildable, not in the default sweep.
    else:
        variants["A  K16   "] = build_kernel(D=D, SPLITS=SPLITS)

    for a in sys.argv:
        if a.startswith("--only="):
            pats = a.split("=", 1)[1].split(",")
            variants = {k: v for k, v in variants.items() if any(p in k for p in pats)}

    def run(fn):
        m, l, a = partials(dd["N"])
        _run_compiled(
            fn,
            dd["q"],
            int(dd["q"].stride(0)),
            int(dd["q"].stride(1)),
            dd["kv"],
            int(dd["kv"].stride(0)),
            dd["req_idx"],
            dd["req_indptr"],
            dd["tok_len"],
            m,
            l,
            a,
            int(m.stride(0)),
            int(m.stride(1)),
            int(m.stride(2)),
            int(a.stride(0)),
            int(a.stride(1)),
            int(a.stride(2)),
            float(1.0 / math.sqrt(D)),
            int(dd["R"]),
            int(H) // getattr(fn, "_h_per_wg", 1),
            Stream(torch.cuda.current_stream()),
        )
        return m, l, a

    if "--abl" in sys.argv:
        # Cumulative ladder at B2's geometry: each level adds one stage of the
        # per-tile pipeline, so the difference between adjacent rows is that
        # stage's cost. Outputs are wrong by construction; only time is read.
        hw = 1
        for a in sys.argv:
            if a.startswith("--hw="):
                hw = int(a.split("=")[1])
        cfg = dict(
            D=D,
            SPLITS=SPLITS,
            BLOCK_K=32,
            mfma_qk=True,
            h_per_wg=hw,
            qk_reuse="--reuse" in sys.argv,
            kv_single="--kv1" in sys.argv,
            alias_p="--kv1" in sys.argv,
            q_resident="--qres" in sys.argv,
            q_global="--qg" in sys.argv,
            kv_vec=8 if "--vec8" in sys.argv else 1,
            kv_pad=8 if "--pad8" in sys.argv else 0,
        )
        ladder = [
            ("launch + epilogue only", dict(cfg, ablate="empty")),
            ("+ tile loop (no staging)", dict(cfg, ablate="gather")),
            ("+ Q/KV/KVT -> LDS", dict(cfg, ablate="stage")),
            ("+ QK MFMA + reduce", dict(cfg, ablate="qk")),
        ]
        if hw == 1:
            ladder += [
                ("+ online softmax", dict(cfg, ablate="softmax")),
                ("+ PV MFMA (= B2)", dict(cfg, mfma_pv=True)),
            ]
        ladder += [
            (f"+ fast softmax (hw={hw})", dict(cfg, mfma_pv=True, fast_softmax=True))
        ]
        print(f"ablation ladder, h_per_wg={hw}")
        prev = None
        for name, kw in ladder:
            t = timeit(lambda fn=build_kernel(**kw): run(fn))
            delta = "" if prev is None else f"   ({t - prev:+8.1f})"
            print(f"  {name:26s} {t:8.1f} us{delta}")
            prev = t
        return

    if "--dbg" in sys.argv:
        ref = run(build_kernel(D=D, SPLITS=SPLITS, BLOCK_K=32, dbg_scores=True))[2]
        got = run(
            build_kernel(D=D, SPLITS=SPLITS, BLOCK_K=32, mfma_qk=True, dbg_scores=True)
        )[2]
        # acc[tok0, split, head, 0:224] holds the (q, kv) score tile.
        dr = ref[::7, :, :, :224].reshape(-1, 7, 32)
        dg = got[::7, :, :, :224].reshape(-1, 7, 32)
        bad = (dr - dg).abs().amax(dim=(0, 1))
        print("max |FMA - MFMA| score per kv column:")
        print("  kv 0-15 :", " ".join(f"{v:.1e}" for v in bad[:16].tolist()))
        print("  kv 16-31:", " ".join(f"{v:.1e}" for v in bad[16:].tolist()))
        print(f"  ref magnitude max = {dr.abs().max().item():.3e}")
        for q in range(3):
            print(f"  q{q} FMA :", " ".join(f"{v:+.3f}" for v in dr[0, q, :8].tolist()))
            print(f"  q{q} MFMA:", " ".join(f"{v:+.3f}" for v in dg[0, q, :8].tolist()))
        return

    scale = 1.0 / math.sqrt(D)
    out_ship = merge(*run_shipped(dd))
    out_ref = ref_torch(dd, scale)

    def rel(a, b):
        return (
            (a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-30)
        ).item()

    print(
        f"shipped vs torch  relL2 = {rel(out_ship, out_ref):.3e}  "
        f"time = {timeit(lambda: run_shipped(dd)):8.1f} us"
    )
    for tag, fn in variants.items():
        out_fly = merge(*run(fn))
        t = timeit(lambda: run(fn))
        print(
            f"{tag} vs torch relL2 = {rel(out_fly, out_ref):.3e}  "
            f"vs shipped = {rel(out_fly, out_ship):.3e}  "
            f"finite={bool(torch.isfinite(out_fly).all())}  time = {t:8.1f} us"
        )


if __name__ == "__main__":
    main()
