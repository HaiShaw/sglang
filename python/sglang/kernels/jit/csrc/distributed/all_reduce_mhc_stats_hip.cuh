// all_reduce_mhc_hip.cuh's TP4 all-reduce + hc_post with the next mHC boundary folded in: the
// same blocks also collapse the new residual with pre_prev (y) and take the boundary's mixing
// statistics (one [rows, MIX] partial and one row sum of squares per block's hidden slice), so
// the Triton _hc_boundary_partial launch disappears. The partials feed HcCoefficients like
// _hc_boundary_partial's, with kBlocksPerRow slices instead of H / 64.
#pragma once

#ifndef USE_ROCM
#error "all_reduce_mhc_stats_hip.cuh runs on AITER's ROCm custom all-reduce"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, RankData, start_sync, packed_reduce

namespace sglang {
namespace all_reduce_mhc_stats_hip {

constexpr int kWorldSize = 4;
constexpr int kHc = 4;
constexpr int kHidden = 5120;
constexpr int kMix = (2 + kHc) * kHc;
constexpr int kMaxRows = 64;
constexpr int kLanes = 64;  // one wave owns a block's hidden slice: kPack values of each copy per lane
constexpr int kWaves = 8;   // the waves split the kMix statistics; wave 0 also runs the all-reduce
constexpr int kThreads = kLanes * kWaves;
constexpr int kMixPerWave = kMix / kWaves;
constexpr int kPack = 8;  // bf16 per 16-byte packet
constexpr int kPacks = kHidden / kPack;
constexpr int kBlocksPerRow = kPacks / kLanes;
static_assert(kBlocksPerRow * kLanes == kPacks, "a row's packets split evenly over the blocks");
static_assert(kMixPerWave * kWaves == kMix, "the statistics split evenly over the waves");
// rows beyond kRowGroups loop inside the blocks: the grid stays within AITER's signal slots
constexpr int kRowGroups = aiter::kMaxBlocks / kBlocksPerRow;
static_assert(kRowGroups >= 1, "a row's blocks exceed AITER signal slots");

using P = opus::vector_t<opus::bf16_t, kPack>;
using A = opus::vector_t<opus::fp32_t, kPack>;

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int offset = kLanes / 2; offset > 0; offset >>= 1)
    v += __shfl_xor(v, offset, kLanes);
  return v;
}

__global__ void __launch_bounds__(kThreads) all_reduce_mhc_post_stats_kernel(
    aiter::RankData* peers,
    aiter::RankSignals signals,
    aiter::Signal* local_signals,
    int rank,
    int rows,
    opus::bf16_t* output,
    opus::bf16_t* y,
    float* part_mix,
    float* part_sq,
    const opus::bf16_t* residual,
    const float* post,
    const float* comb,
    const float* pre_prev,
    const float* hc_fn,
    int64_t hc_fn_stride) {
  // a block owns rows first, first + row_step, ...; its waves all-reduce up to kWaves of them at once
  __shared__ A shared_copies[kWaves][kHc][kLanes];
  const int slice = blockIdx.x % kBlocksPerRow;
  const int row_step = gridDim.x / kBlocksPerRow;
  const int wave = threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  const int pack = slice * kLanes + lane;
  // the weights do not depend on the all-reduce: their loads overlap the cross-rank sync
  float4 weights[kMixPerWave][kHc][2];
#pragma unroll
  for (int n = 0; n < kMixPerWave; ++n)
#pragma unroll
    for (int k = 0; k < kHc; ++k) {
      const float4* w =
          reinterpret_cast<const float4*>(hc_fn + (wave * kMixPerWave + n) * hc_fn_stride + k * kHidden + pack * kPack);
      weights[n][k][0] = w[0];
      weights[n][k][1] = w[1];
    }
  aiter::start_sync<kWorldSize>(signals, local_signals, rank);
  for (int first = blockIdx.x / kBlocksPerRow; first < rows; first += row_step * kWaves) {
    // the previous batch's statistics still read shared_copies
    __syncthreads();
    const int row = first + wave * row_step;
    if (row < rows) {
      const P* inputs[kWorldSize];
#pragma unroll
      for (int r = 0; r < kWorldSize; ++r)
        inputs[r] = reinterpret_cast<const P*>(peers->ptrs[r]);
      const P reduced = aiter::packed_reduce<P, kWorldSize, A>(inputs, row * kPacks + pack);
      // hc_post exactly as all_reduce_mhc_post_kernel; the statistics read the bf16 written
      A collapsed;
      float sq = 0.0f;
#pragma unroll
      for (int v = 0; v < kPack; ++v)
        collapsed[v] = 0.0f;
#pragma unroll
      for (int out_h = 0; out_h < kHc; ++out_h) {
        A mixed;
#pragma unroll
        for (int v = 0; v < kPack; ++v)
          mixed[v] = aiter::upcast_s(reduced[v]) * post[row * kHc + out_h];
#pragma unroll
        for (int in_h = 0; in_h < kHc; ++in_h) {
          const P values = reinterpret_cast<const P*>(residual)[(row * kHc + in_h) * kPacks + pack];
          const float coefficient = comb[(row * kHc + in_h) * kHc + out_h];
#pragma unroll
          for (int v = 0; v < kPack; ++v)
            mixed[v] += aiter::upcast_s(values[v]) * coefficient;
        }
        const P rounded = aiter::downcast<P>(mixed);
        reinterpret_cast<P*>(output)[(row * kHc + out_h) * kPacks + pack] = rounded;
        // y = sum_k pre_prev[k] * copy_k, in _hc_boundary_partial_kernel's k order
        const float p = pre_prev[row * kHc + out_h];
#pragma unroll
        for (int v = 0; v < kPack; ++v) {
          const float copy = aiter::upcast_s(rounded[v]);
          shared_copies[wave][out_h][lane][v] = copy;
          collapsed[v] += p * copy;
          sq += copy * copy;
        }
      }
      reinterpret_cast<P*>(y)[row * kPacks + pack] = aiter::downcast<P>(collapsed);
      sq = wave_sum(sq);
      if (lane == 0) part_sq[slice * rows + row] = sq;
    }
    __syncthreads();
    // every wave takes its kMixPerWave statistics of each row in the batch
    for (int w = 0; w < kWaves; ++w) {
      const int stat_row = first + w * row_step;
      if (stat_row >= rows) break;
      float mix[kMixPerWave];
#pragma unroll
      for (int n = 0; n < kMixPerWave; ++n)
        mix[n] = 0.0f;
#pragma unroll
      for (int k = 0; k < kHc; ++k) {
        const A copy = shared_copies[w][k][lane];
#pragma unroll
        for (int n = 0; n < kMixPerWave; ++n) {
          const float4 w0 = weights[n][k][0];
          const float4 w1 = weights[n][k][1];
          mix[n] += copy[0] * w0.x + copy[1] * w0.y + copy[2] * w0.z + copy[3] * w0.w + copy[4] * w1.x +
                    copy[5] * w1.y + copy[6] * w1.z + copy[7] * w1.w;
        }
      }
      float* out_mix = part_mix + (static_cast<int64_t>(slice) * rows + stat_row) * kMix + wave * kMixPerWave;
#pragma unroll
      for (int n = 0; n < kMixPerWave; ++n) {
        mix[n] = wave_sum(mix[n]);
        if (lane == 0) out_mix[n] = mix[n];
      }
    }
  }
  aiter::end_sync<kWorldSize, true>(signals, local_signals, rank);
}

struct AllReduceMhcPostStatsKernel {
  // see AllReduceMhcPostKernel::run; also writes y [M, H], part_mix [kBlocksPerRow, M, kMix] and
  // part_sq [kBlocksPerRow, M] for the boundary after the all-reduce
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post,
      const tvm::ffi::TensorView comb,
      const tvm::ffi::TensorView pre_prev,
      const tvm::ffi::TensorView hc_fn,
      const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView part_mix,
      const tvm::ffi::TensorView part_sq,
      int64_t registered,
      int64_t registered_bytes) {
    using namespace host;
    auto M = SymbolicSize{"num_rows"};
    auto S = SymbolicSize{"hc_fn_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(input).verify(y);
    TensorMatcher({M, kHc, kHidden}).with_dtype<bf16_t>().with_device(device).verify(residual).verify(output);
    TensorMatcher({M, kHc}).with_dtype<float>().with_device(device).verify(post).verify(pre_prev);
    TensorMatcher({M, kHc, kHc}).with_dtype<float>().with_device(device).verify(comb);
    TensorMatcher({kMix, kHc * kHidden}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(hc_fn);
    TensorMatcher({kBlocksPerRow, M, kMix}).with_dtype<float>().with_device(device).verify(part_mix);
    TensorMatcher({kBlocksPerRow, M}).with_dtype<float>().with_device(device).verify(part_sq);
    const int64_t rows = M.unwrap();
    RuntimeCheck(1 <= rows && rows <= kMaxRows, "all_reduce_mhc_post_stats takes 1-", kMaxRows, " rows, got ", rows);
    const int64_t hc_fn_stride = S.unwrap();
    RuntimeCheck(hc_fn_stride % 4 == 0, "hc_fn rows must be 16-byte aligned");

    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorldSize, "all_reduce_mhc_post_stats requires TP", kWorldSize);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    void* values = input.data_ptr();
    if (registered != 0) {
      const int64_t bytes = rows * kHidden * static_cast<int64_t>(sizeof(bf16_t));
      RuntimeCheck(bytes <= registered_bytes, "all-reduce input pool is too small");
      values = reinterpret_cast<void*>(registered);
      RuntimeDeviceCheck(hipMemcpyAsync(values, input.data_ptr(), bytes, hipMemcpyDeviceToDevice, stream));
    }
    aiter::RankData* peers = communicator->get_buffer_RD(stream, values);
    const int64_t row_groups = rows < kRowGroups ? rows : kRowGroups;
    LaunchKernel(static_cast<uint32_t>(row_groups * kBlocksPerRow), kThreads, stream)(
        all_reduce_mhc_post_stats_kernel,
        peers,
        communicator->sg_,
        communicator->self_sg_,
        communicator->rank_,
        static_cast<int>(rows),
        static_cast<opus::bf16_t*>(output.data_ptr()),
        static_cast<opus::bf16_t*>(y.data_ptr()),
        static_cast<float*>(part_mix.data_ptr()),
        static_cast<float*>(part_sq.data_ptr()),
        static_cast<const opus::bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()),
        static_cast<const float*>(pre_prev.data_ptr()),
        static_cast<const float*>(hc_fn.data_ptr()),
        hc_fn_stride);
  }
};

}  // namespace all_reduce_mhc_stats_hip
}  // namespace sglang
