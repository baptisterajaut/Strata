// src/inkling2/ink2_engine.cu - strata-ink2: Inkling-2-Mini's engine path (NEW-MODELS-PLAN.md T2), step 1: the whole
// forward pass on the GPU, one token at a time (the prompt too), checked against strata-ink2ref (T1) and the fp32
// transformers oracle (terminal/tools/inkling2/).
//
//   * weights resident in their checkpoint form, as in strata-ink2ref: BF16 dense matrices, NVFP4 experts (layers 0-19:
//     e2m1 pairs, an e4m3 scale per 16, an fp32 factor amax/denominator per expert), MXFP8 experts (layers 20-23: e4m3,
//     a ue8m0 exponent per 32).  Every matrix-vector product reads them as they are, fp32 activations, fp32 sums
//     ("W exact, A16"): nothing is requantized;
//   * the embedding table stays in host RAM (one row per token);
//   * attention: split over 256-position chunks per head with a log-sum-exp merge that adds the per-token sink logit;
//     the relative bias table [24][1024] built per token; KV in fp32 (CBA: fp32 KV, --ctx 8192 by default - int8 KV is
//     T2 step 3); sliding layers keep a 1024-position ring, full layers the whole context;
//   * convolutions with a 3-row history per layer application (48 applications: each loop pass has its own state).
//
//   * one CUDA graph per token (--graph 1, the default): positions on the device, built at the first token.  Build with
//     --default-stream per-thread (the graph is captured on cudaStreamPerThread).
//
//   * the prompt in chunks of --chunk tokens (default 512; 0 = one token at a time through the decode graph): cuBLAS
//     SGEMM in full FP32 for the dense products (fp32 copy of each BF16 matrix made per use), a grouped expert product
//     decoding NVFP4 / MXFP8 on the fly, tiled fp32 attention.  Link -lcublas.
//
// Modes: --tokens IDS | --tokens-file F [--n N] [--dump-logits F] [--ctx N] [--graph 0|1] [--chunk N]; prints prompt /
// decode speed.
#include "../image/vae/safetensors_lite.hpp"
#include "ink2_iq_tables.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using vec = std::vector<float>;
namespace wmma = nvcuda::wmma;
using strata::image::StFile;
using strata::image::StTensor;

constexpr int H = 2304, NL = 24, NQ = 24, NKV = 8, HD = 128, NREP = NQ / NKV, DREL = 16, EXTENT = 1024, WINDOW = 1024;
constexpr int QD = NQ * HD, KVD = NKV * HD;   // 3072, 1024
constexpr int NE = 128, TOPK = 8, FF = 1152, KC = 4, GATE_RANK = 64, VOCAB = 200058, VOCAB_ROWS = 201024;
constexpr int LOOP_BEGIN = 6, LOOP_END = 18, NLOOP = LOOP_END - LOOP_BEGIN, PASSES = 3, BLOCK = 6;
constexpr int NCALLS = NL + (PASSES - 1) * NLOOP;
constexpr int MXFP8_FIRST = 20;
constexpr float EPS = 1e-6f, QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f, LOG_ALPHA = 0.1f, LOG_FLOOR = 8192.0f;
constexpr float GATE_CAP = 4.0f, UP_CAP = 25.0f, ROUTE_SCALE = 8.0f, LOGIT_CAP = 30.0f;
constexpr int CHUNK = 256;   // attention positions per block
int PF_RING_EXTRA = 512;     // = the prefill chunk: a chunk's keys must not evict the window of its first query
constexpr int OFF_GATE = QD, OFF_REL = 2 * QD, OFF_SINK = OFF_REL + NQ * DREL, OFF_K = OFF_SINK + NQ;   // in proj

bool full_layer(int L) { return L % BLOCK == BLOCK - 1; }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// Kernels

__device__ __forceinline__ float bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ float e4m3_decode(uint32_t v) {
    const int e = (v >> 3) & 15, m = v & 7;
    float r = e == 0 ? (float) m * 0.001953125f : ldexpf(1.0f + (float) m * 0.125f, e - 7);
    if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
    return (v & 0x80) ? -r : r;
}

__device__ __forceinline__ float warp_sum(float a) {
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    return a;
}
__device__ __forceinline__ float warp_max(float a) {
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

// sum over the block (blockDim.x a multiple of 32, <= 1024); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x & 31, nw = blockDim.x / 32;
    __syncthreads();
    if (!l) red[w] = v;
    __syncthreads();
    float t = l < nw ? red[l] : 0.0f;
    return warp_sum(t);
}
__device__ float block_max(float v) {
    __shared__ float red[32];
    v = warp_max(v);
    const int w = threadIdx.x / 32, l = threadIdx.x & 31, nw = blockDim.x / 32;
    __syncthreads();
    if (!l) red[w] = v;
    __syncthreads();
    float t = l < nw ? red[l] : -INFINITY;
    return warp_max(t);
}

// y[row] = W[row] . x  (BF16 W [N, K], K % 8 == 0), one warp per row, x staged in shared memory and read as float4
// (lane stride 32 B: 2-way bank conflicts at most)
// R rows per warp: the staged x serves 8R rows per block, R independent loads in flight per lane.  Each row's sum is
// the same sequence of operations whatever R (bit-identical results).
template <int R>
__device__ __forceinline__ void gemv_bf16_dot(const uint16_t* __restrict__ W, const float* xs, float* __restrict__ y,
                                              int N, int K, int row0) {
    const int lane = threadIdx.x & 31;
    const uint4* w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) w[r] = reinterpret_cast<const uint4*>(W + (size_t) min(row0 + r, N - 1) * K);
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint4 v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = __ldg(w[r] + i);
        const float4 a = x4[2 * i], b = x4[2 * i + 1];
#pragma unroll
        for (int r = 0; r < R; ++r)
            acc[r] += bf16lo(v[r].x) * a.x + bf16hi(v[r].x) * a.y + bf16lo(v[r].y) * a.z + bf16hi(v[r].y) * a.w +
                      bf16lo(v[r].z) * b.x + bf16hi(v[r].z) * b.y + bf16lo(v[r].w) * b.z + bf16hi(v[r].w) * b.w;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[row0 + r] = t;
    }
}
template <int R>
__device__ __forceinline__ void gemv_bf16_rows(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                               float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    gemv_bf16_dot<R>(W, xs, y, N, K, row0);
}
template <int R>
__global__ void k_gemv_bf16(const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N,
                            int K) {
    gemv_bf16_rows<R>(W, x, y, N, K);
}
// q / k / v projections with three inputs in one launch: rows [0, b1) . xq, [b1, b2) . xk, [b2, N) . xv (R divides b1 and
// b2: a warp's rows share an input).  A block stages only the inputs its rows touch (smem 3K).
template <int R>
__global__ void k_gemv_bf16_qkv(const uint16_t* __restrict__ W, const float* __restrict__ xq, const float* __restrict__ xk,
                                const float* __restrict__ xv, int b1, int b2, float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    const int rb = blockIdx.x * (blockDim.x / 32) * R, re = min(rb + (int) (blockDim.x / 32) * R, N);
    if (rb < b1) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xq[i];
    if (rb < b2 && re > b1) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[K + i] = xk[i];
    if (re > b2) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[2 * K + i] = xv[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    gemv_bf16_dot<R>(W, xs + (row0 < b1 ? 0 : row0 < b2 ? K : 2 * K), y, N, K, row0);
}

// Up to 3 independent message gates per launch (gate j = blockIdx.y, or blockIdx.x for the one-block kernels): the
// q / k / v gates of a layer application are independent, 4 launches instead of 12; same arithmetic per gate.
struct GateArgs {
    const uint16_t* down[3];
    const uint16_t* fg[3];
    const float* x[3];
    const float* m[3];
    float* y[3];
    const float* nw[3];
    float* yn[3];
};
// fg GEMV of gate j: gfg[j] [2H] = fg_j [2H, GATE_RANK] . gdown[j]
template <int R>
__global__ void k_gemv_bf16_gate(GateArgs a, const float* __restrict__ gdown, float* __restrict__ gfg, int N, int K) {
    const int j = blockIdx.y;
    gemv_bf16_rows<R>(a.fg[j], gdown + (size_t) j * GATE_RANK, gfg + (size_t) j * N, N, K);
}

// e2m1 pair (one byte: low nibble first) -> fp32 values * 2^-14: the nibble's sign/exponent/mantissa bits placed
// in an fp16 (exact, subnormal 0.5 included); the 2^14 goes into the block scale
__device__ __forceinline__ float2 e2m1x2(uint32_t byte) {
    const uint32_t a = (byte & 0xFu) | ((byte & 0xF0u) << 12);
    const uint32_t h = ((a & 0x00080008u) << 12) | ((a & 0x00070007u) << 9);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
// two e4m3 bytes (in bits 0-7 and 16-23) -> fp32 values * 2^-8 (fp16 placement, exact incl. subnormals)
__device__ __forceinline__ float2 e4m3x2(uint32_t a) {
    const uint32_t h = ((a & 0x00800080u) << 8) | ((a & 0x007F007Fu) << 7);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// The TOPK selected experts at once (blockIdx.y = k): y[k][row] = W_e[row] . x_k, e = ids[k].
// NVFP4: lane reads 4 bytes = 8 columns (half a 16-column block, its e4m3 scale s[i/2]); the sum times factors[e].
// SoftcappedSwiGLU of one (gate, up) pair (row 2j gate j, 2j+1 up j of the gu buffer)
__device__ __forceinline__ float swiglu1(float g, float u) {
    return (tanhf(g / GATE_CAP) * GATE_CAP) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / UP_CAP) * UP_CAP);
}
// SWI: x holds [gate, up] pairs (gu rows, stride x_stride = 2K): the GEMV input is their SwiGLU, computed while staging
template <int R, bool SWI = false>
__global__ void k_gemv_nvfp4_sel(const uint8_t* const* __restrict__ tab,
                                 const float* __restrict__ factors, const int* __restrict__ ids,
                                 const float* __restrict__ x, int x_stride, float* __restrict__ y, int N, int K) {
    extern __shared__ float sm[];
    float* xs = sm;           // K
    float* e4 = sm + K;       // 256 (scale table, * 2^14)
    const float* xk = x + (size_t) blockIdx.y * x_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = SWI ? swiglu1(xk[2 * i], xk[2 * i + 1]) : xk[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4[i] = e4m3_decode(i) * 16384.0f;
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const int e = ids[blockIdx.y];
    const uint8_t* W = tab[e];
    const uint8_t* S = W + (size_t) N * (K / 2);
    const uint32_t* w[R];
    const uint8_t* s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const size_t row = min(row0 + r, N - 1);
        w[r] = reinterpret_cast<const uint32_t*>(W + row * (K / 2));
        s[r] = S + row * (K / 16);
    }
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint32_t v[R];
        uint8_t sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); sc[r] = __ldg(s[r] + (i >> 1)); }
        const float4 a = x4[2 * i], b = x4[2 * i + 1];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 p0 = e2m1x2(v[r] & 0xff), p1 = e2m1x2((v[r] >> 8) & 0xff), p2 = e2m1x2((v[r] >> 16) & 0xff),
                         p3 = e2m1x2(v[r] >> 24);
            const float part = p0.x * a.x + p0.y * a.y + p1.x * a.z + p1.y * a.w + p2.x * b.x + p2.y * b.y +
                               p3.x * b.z + p3.y * b.w;
            acc[r] += part * e4[sc[r]];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[(size_t) blockIdx.y * N + row0 + r] = t * factors[e];
    }
}

// MXFP8: lane reads 4 bytes = 4 columns; the ue8m0 exponent per 32 columns (8 words): 2^(s - 127) * 2^8
template <int R, bool SWI = false>
__global__ void k_gemv_mxfp8_sel(const uint8_t* const* __restrict__ tab, const int* __restrict__ ids, const float* __restrict__ x, int x_stride,
                                 float* __restrict__ y, int N, int K) {
    extern __shared__ float sm[];
    float* xs = sm;
    const float* xk = x + (size_t) blockIdx.y * x_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = SWI ? swiglu1(xk[2 * i], xk[2 * i + 1]) : xk[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const int e = ids[blockIdx.y];
    const uint8_t* W = tab[e];
    const uint8_t* S = W + (size_t) N * K;
    const uint32_t* w[R];
    const uint8_t* s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const size_t row = min(row0 + r, N - 1);
        w[r] = reinterpret_cast<const uint32_t*>(W + row * K);
        s[r] = S + row * (K / 32);
    }
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 4; i += 32) {
        uint32_t v[R];
        uint8_t sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); sc[r] = __ldg(s[r] + (i >> 3)); }
        const float4 a = x4[i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 p01 = e4m3x2((v[r] & 0xffu) | ((v[r] & 0xff00u) << 8));
            const float2 p23 = e4m3x2(((v[r] >> 16) & 0xffu) | ((v[r] & 0xff000000u) >> 8));
            const float part = p01.x * a.x + p01.y * a.y + p23.x * a.z + p23.y * a.w;
            acc[r] += part * exp2f((float) sc[r] - 119.0f);   // 2^(s-127) * 2^8
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[(size_t) blockIdx.y * N + row0 + r] = t;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GGUF weights (--model FILE.gguf, llama.cpp's merged inkling2_loop layout): experts gate_up IQ3_S / down IQ4_NL (or
// Q8_0), dense matrices Q8_0 repacked at load into int8 codes [N][K] + fp16 scales [N][K / 32] (lossless).
enum GgType : int { GG_F32 = 0, GG_F16 = 1, GG_Q8_0 = 8, GG_IQ4_NL = 20, GG_IQ3_S = 21, GG_BF16 = 30 };
__device__ uint32_t d_iq3s_grid[512];
__constant__ int8_t c_iq4nl[16];

// 8 weights of one row (row pointer rp, ggml blocks) at columns [kc, kc + 8), kc % 8 == 0; the tables in shared memory
template <int T>
__device__ __forceinline__ void gg_ld8(const uint8_t* __restrict__ rp, int kc, float* w, const uint32_t* grid,
                                       const int8_t* kv) {
    if constexpr (T == GG_IQ3_S) {   // 110 B per 256: d, qs[64], qh[8], signs[32], scales[4]
        const uint8_t* b = rp + (size_t) (kc >> 8) * 110;
        const int ib = (kc & 255) >> 5, il = (kc & 31) >> 3;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float db = d * (float) (1 + 2 * ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15));
        const uint32_t qh = b[66 + ib], sg = b[74 + 4 * ib + il];
        const uint32_t g1 = grid[b[2 + 8 * ib + 2 * il] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t g2 = grid[b[3 + 8 * ib + 2 * il] | ((qh << (7 - 2 * il)) & 256)];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = db * (float) ((g1 >> (8 * j)) & 255) * (((sg >> j) & 1) ? -1.0f : 1.0f);
            w[4 + j] = db * (float) ((g2 >> (8 * j)) & 255) * (((sg >> (4 + j)) & 1) ? -1.0f : 1.0f);
        }
    } else if constexpr (T == GG_IQ4_NL) {   // 18 B per 32: d, qs[16] (low nibbles values 0..15, high 16..31)
        const uint8_t* b = rp + (size_t) (kc >> 5) * 18;
        const int o = kc & 31;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t by = b[2 + ((o + j) & 15)];
            w[j] = d * (float) kv[o < 16 ? by & 15u : by >> 4];
        }
    } else {   // Q8_0: 34 B per 32: d, qs[32]
        const uint8_t* b = rp + (size_t) (kc >> 5) * 34;
        const int o = kc & 31;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = d * (float) (int8_t) b[2 + o + j];
    }
}
__device__ __forceinline__ void gg_tables(uint32_t* grid, int8_t* kv) {
    for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = d_iq3s_grid[i];
    if (threadIdx.x < 16) kv[threadIdx.x] = c_iq4nl[threadIdx.x];
}
// w13 rows of the GGUF are [gate 0..FF) [up FF..2FF); the engine's gu rows interleave them (2j gate j, 2j+1 up j)
__device__ __forceinline__ int gg_perm(int n, int N) { return (n & 1) ? N / 2 + (n >> 1) : n >> 1; }

// selected experts, GGUF formats: y[k][o] = W_e[row(o)] . x_k (row(o) = gg_perm(o) for PERM); W_e = tab[e] (part A:
// the gate_up rows, part B: the down rows of the expert's slot); rb = bytes per row
template <int T, int R, bool SWI, bool PERM>
__global__ void k_gemv_gg_sel(const uint8_t* const* __restrict__ tab, size_t rb, const int* __restrict__ ids,
                              const float* __restrict__ x, int x_stride, float* __restrict__ y, int N, int K) {
    extern __shared__ float sm[];
    float* xs = sm;
    uint32_t* grid = reinterpret_cast<uint32_t*>(sm + K);
    int8_t* kv = reinterpret_cast<int8_t*>(grid + 512);
    const float* xk = x + (size_t) blockIdx.y * x_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = SWI ? swiglu1(xk[2 * i], xk[2 * i + 1]) : xk[i];
    gg_tables(grid, kv);
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const uint8_t* W = tab[ids[blockIdx.y]];
    const uint8_t* rp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int o = min(row0 + r, N - 1);
        rp[r] = W + (size_t) (PERM ? gg_perm(o, N) : o) * rb;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int kc = lane * 8; kc < K; kc += 256) {
        const float4 a = *reinterpret_cast<const float4*>(xs + kc), b = *reinterpret_cast<const float4*>(xs + kc + 4);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float w[8];
            gg_ld8<T>(rp[r], kc, w, grid, kv);
            acc[r] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[(size_t) blockIdx.y * N + row0 + r] = t;
    }
}

// ---- decode expert GEMV in integer arithmetic (INK2_GG_Q8=1, default): the activations quantized to int8 blocks of 32
// (block_q8_1), the weights dotted 4 bytes at a time (dp4a).  The dot bodies are llama.cpp's vec_dot_iq3_s_q8_1 /
// vec_dot_iq4_nl_q8_1 / vec_dot_q8_0_q8_1 (ggml/src/ggml-cuda/vecdotq.cuh, MIT, Copyright (c) 2023-2026 The ggml
// authors), as vendored in src/split/src/qdot.cuh.
struct __align__(4) gq8_1 { __half2 ds; int8_t qs[32]; };   // ds.x = d, ds.y = d * sum(qs)
static_assert(sizeof(gq8_1) == 36, "block_q8_1");
__constant__ __align__(16) int8_t c_iq4nl_t[16];
__device__ __forceinline__ int gq_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return (int) ((uint32_t) x16[2 * i32] | ((uint32_t) x16[2 * i32 + 1] << 16));
}
__device__ __forceinline__ int2 gq_table16(int q4, const int8_t* table) {
    const uint32_t* t32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | (((uint32_t) q4 & 0x88888888u) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t32[0], t32[1], (uint32_t) q4 >> sh), hi = __byte_perm(t32[2], t32[3], (uint32_t) q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}
// IQ3_S superblock (110 B) kbx of row bq, int offset iqs (0, 2, .., 14): 32 values against q8 block iqs / 2
__device__ __forceinline__ float gq_dot_iq3s(const uint8_t* bq, int kbx, int iqs, const gq8_1* y, const uint32_t* grid) {
    const uint8_t* b = bq + (size_t) kbx * 110;
    const uint8_t *qs8 = b + 2, *qh8 = b + 66, *sg8 = b + 74, *sc8 = b + 106;
    const int2 qsp = make_int2(gq_b2(qs8, iqs), gq_b2(qs8, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qsp;
    const int qh = qh8[iqs / 2];
    const int sgp = gq_b2(sg8, iqs / 2);
    const uint8_t* sg = (const uint8_t*) &sgp;
    const gq8_1& q8 = y[iqs / 2];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int g0 = (int) grid[qs[l0] | ((qh << (8 - l0)) & 0x100)];
        const int g1 = (int) grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const int s0 = __vcmpne4(((sg[l0 / 2] & 0x03) << 7) | ((sg[l0 / 2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sg[l0 / 2] & 0x30) << 3) | ((sg[l0 / 2] & 0xC0) << 17), 0);
        sumi = __dp4a((int) __vsub4(g0 ^ s0, s0), *(const int*) (q8.qs + 4 * l0), sumi);
        sumi = __dp4a((int) __vsub4(g1 ^ s1, s1), *(const int*) (q8.qs + 4 * (l0 + 1)), sumi);
    }
    sumi *= 1 + 2 * ((sc8[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return __half2float(*(const __half*) b) * __low2float(q8.ds) * (float) sumi;
}
// IQ4_NL block (18 B) kbx, int offset iqs (0, 2): 16 values against q8 block kbx
__device__ __forceinline__ float gq_dot_iq4nl(const uint8_t* bq, int kbx, int iqs, const gq8_1* y) {
    const uint8_t* b = bq + (size_t) kbx * 18;
    const gq8_1& q8 = y[kbx];
    const int* q = (const int*) q8.qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int2 v = gq_table16(gq_b2(b + 2, iqs + l), c_iq4nl_t);
        sumi = __dp4a((int) v.x, (int) q[l], sumi);
        sumi = __dp4a((int) v.y, (int) q[l + 4], sumi);
    }
    return __half2float(*(const __half*) b) * __low2float(q8.ds) * (float) sumi;
}
// Q8_0 block (34 B) kbx, int offset iqs (0, 2, 4, 6): 8 values
__device__ __forceinline__ float gq_dot_q8_0(const uint8_t* bq, int kbx, int iqs, const gq8_1* y) {
    const uint8_t* b = bq + (size_t) kbx * 34;
    const gq8_1& q8 = y[kbx];
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) sumi = __dp4a(gq_b2(b + 2, iqs + l), *(const int*) (q8.qs + 4 * (iqs + l)), sumi);
    return __half2float(*(const __half*) b) * __low2float(q8.ds) * (float) sumi;
}
// x [rows][K] (row stride xs; SWI: the SwiGLU of [gate, up] pairs, stride 2K) -> q8 blocks [rows][K / 32]; a warp per block
template <bool SWI>
__global__ void k_quant_q8_1(const float* __restrict__ x, int xs, gq8_1* __restrict__ y, int K) {
    const int nb = K / 32, w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31, r = blockIdx.y;
    if (w >= nb) return;
    const float* xr = x + (size_t) r * xs;
    const int i = 32 * w + lane;
    const float v = SWI ? swiglu1(xr[2 * i], xr[2 * i + 1]) : xr[i];
    const float am = warp_max(fabsf(v));
    const float d = am / 127.0f;
    const int q = am == 0.0f ? 0 : __float2int_rn(v / d);
    const int sum = (int) warp_sum((float) q);
    gq8_1& o = y[(size_t) r * nb + w];
    o.qs[lane] = (int8_t) q;
    if (!lane) o.ds = __floats2half2_rn(d, d * (float) sum);
}
// y[k][o] = W_e[row(o)] . x_k, the input as q8 blocks yq[k] (yq_stride blocks per expert; 0: one input for all); a warp
// per output row, 8 rows per block of 256
template <int T, bool PERM>
__global__ void __launch_bounds__(256) k_gemv_gg_q8(const uint8_t* const* __restrict__ tab, size_t rb,
                                                    const int* __restrict__ ids, const gq8_1* __restrict__ yq,
                                                    int yq_stride, float* __restrict__ y, int N, int K) {
    __shared__ uint32_t grid[512];   // the IQ3_S grid: scattered lookups, kept out of global memory
    if constexpr (T == GG_IQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = d_iq3s_grid[i];
        __syncthreads();
    }
    const int o = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (o >= N) return;
    const uint8_t* rp = tab[ids[blockIdx.y]] + (size_t) (PERM ? gg_perm(o, N) : o) * rb;
    const gq8_1* yk = yq + (size_t) blockIdx.y * yq_stride;
    float acc = 0.0f;
    if constexpr (T == GG_IQ3_S) {   // 8 lanes per superblock (2 ints each): 4 superblocks per warp step
        const int nsb = K / 256, iqs = 2 * (lane & 7);
        for (int kb = lane >> 3; kb < nsb; kb += 4) acc += gq_dot_iq3s(rp, kb, iqs, yk + kb * 8, grid);
    } else if constexpr (T == GG_IQ4_NL) {   // 2 lanes per block: 16 blocks per step
        const int nb = K / 32, iqs = 2 * (lane & 1);
        for (int kb = lane >> 1; kb < nb; kb += 16) acc += gq_dot_iq4nl(rp, kb, iqs, yk);
    } else {   // Q8_0: 4 lanes per block
        const int nb = K / 32, iqs = 2 * (lane & 3);
        for (int kb = lane >> 2; kb < nb; kb += 8) acc += gq_dot_q8_0(rp, kb, iqs, yk);
    }
    acc = warp_sum(acc);
    if (!lane) y[(size_t) blockIdx.y * N + o] = acc;
}

// dense Q8_0 (repacked): y[row] = W[row] . x, as gemv_bf16_dot (one warp per R rows, x staged)
template <int R>
__device__ __forceinline__ void gemv_q8_dot(const int8_t* __restrict__ Wq, const __half* __restrict__ Ws, const float* xs,
                                            float* __restrict__ y, int N, int K, int row0) {
    const int lane = threadIdx.x & 31;
    const uint2* w[R];
    const __half* sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const size_t row = (size_t) min(row0 + r, N - 1);
        w[r] = reinterpret_cast<const uint2*>(Wq + row * K);
        sc[r] = Ws + row * (K / 32);
    }
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint2 v[R];
        float f[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); f[r] = __half2float(sc[r][i >> 2]); }
        const float4 a = x4[2 * i], b = x4[2 * i + 1];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const char4 c0 = *reinterpret_cast<const char4*>(&v[r].x), c1 = *reinterpret_cast<const char4*>(&v[r].y);
            acc[r] += f[r] * (c0.x * a.x + c0.y * a.y + c0.z * a.z + c0.w * a.w + c1.x * b.x + c1.y * b.y + c1.z * b.z +
                              c1.w * b.w);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[row0 + r] = t;
    }
}
template <int R>
__global__ void k_gemv_q8(const int8_t* __restrict__ Wq, const __half* __restrict__ Ws, const float* __restrict__ x,
                          float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    gemv_q8_dot<R>(Wq, Ws, xs, y, N, K, row0);
}
template <int R>
__global__ void k_gemv_q8_qkv(const int8_t* __restrict__ Wq, const __half* __restrict__ Ws, const float* __restrict__ xq,
                              const float* __restrict__ xk, const float* __restrict__ xv, int b1, int b2,
                              float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    const int rb = blockIdx.x * (blockDim.x / 32) * R, re = min(rb + (int) (blockDim.x / 32) * R, N);
    if (rb < b1) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xq[i];
    if (rb < b2 && re > b1) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[K + i] = xk[i];
    if (re > b2) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[2 * K + i] = xv[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    gemv_q8_dot<R>(Wq, Ws, xs + (row0 < b1 ? 0 : row0 < b2 ? K : 2 * K), y, N, K, row0);
}
// Q8 rows -> fp32 / fp16 (prompt products through cuBLAS); n8 = elements / 8
__global__ void kb_q8_to_f32(const int8_t* __restrict__ q, const __half* __restrict__ sc, float* __restrict__ dst, size_t n8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint2 v = reinterpret_cast<const uint2*>(q)[i];
    const float f = __half2float(sc[i >> 2]);
    const char4 c0 = *reinterpret_cast<const char4*>(&v.x), c1 = *reinterpret_cast<const char4*>(&v.y);
    float4* d = reinterpret_cast<float4*>(dst + 8 * i);
    d[0] = make_float4(c0.x * f, c0.y * f, c0.z * f, c0.w * f);
    d[1] = make_float4(c1.x * f, c1.y * f, c1.z * f, c1.w * f);
}
__global__ void kb_q8_to_f16(const int8_t* __restrict__ q, const __half* __restrict__ sc, __half* __restrict__ dst, size_t n8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint2 v = reinterpret_cast<const uint2*>(q)[i];
    const float f = __half2float(sc[i >> 2]);
    const char4 c0 = *reinterpret_cast<const char4*>(&v.x), c1 = *reinterpret_cast<const char4*>(&v.y);
    __half2* d = reinterpret_cast<__half2*>(dst + 8 * i);
    d[0] = __floats2half2_rn(c0.x * f, c0.y * f); d[1] = __floats2half2_rn(c0.z * f, c0.w * f);
    d[2] = __floats2half2_rn(c1.x * f, c1.y * f); d[3] = __floats2half2_rn(c1.z * f, c1.w * f);
}

// ---------------------------------------------------------------------------------------------------------------------
// MTP drafter kernels (--mtp DIR: thinkingmachines' mtp.safetensors, DSpark: 3 dense layers over a block of 8, context
// K/V from the target's hidden states, a recurrent Markov head; see the MTP section before main)

// y[v][row] = W[row] . x[v] for v < nv (<= 8): Q8 W [N][K] read once for all vectors; a warp per row, x staged in K
// chunks of 1024
constexpr int MV = 8, MKC = 1024;
__global__ void __launch_bounds__(256) k_gemv_q8_multi(const int8_t* __restrict__ Wq, const __half* __restrict__ Ws,
                                                       const float* __restrict__ X, int ldx, float* __restrict__ Y, int ldy,
                                                       int nv, int N, int K) {
    __shared__ __align__(16) float xs[MV][MKC];
    constexpr int RR = 4;   // rows per warp: 32 per block share the staged inputs
    const int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * RR, lane = threadIdx.x & 31;
    float acc[RR][MV] = {};
    const uint2* w[RR];
    const __half* sc[RR];
#pragma unroll
    for (int q = 0; q < RR; ++q) {
        w[q] = reinterpret_cast<const uint2*>(Wq + (size_t) min(row0 + q, N - 1) * K);
        sc[q] = Ws + (size_t) min(row0 + q, N - 1) * (K / 32);
    }
    for (int k0 = 0; k0 < K; k0 += MKC) {
        const int kc = min(MKC, K - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < nv * kc; i += blockDim.x) xs[i / kc][i % kc] = X[(size_t) (i / kc) * ldx + k0 + i % kc];
        __syncthreads();
        for (int i = lane; i < kc / 8; i += 32) {
#pragma unroll
            for (int q = 0; q < RR; ++q) {
                const uint2 c = __ldg(w[q] + (k0 / 8) + i);
                const float f = __half2float(sc[q][(k0 + 8 * i) / 32]);
                const char4 c0 = *reinterpret_cast<const char4*>(&c.x), c1 = *reinterpret_cast<const char4*>(&c.y);
#pragma unroll
                for (int v = 0; v < MV; ++v) {
                    if (v >= nv) break;
                    const float4 a = *reinterpret_cast<const float4*>(&xs[v][8 * i]), b = *reinterpret_cast<const float4*>(&xs[v][8 * i + 4]);
                    acc[q][v] += f * (c0.x * a.x + c0.y * a.y + c0.z * a.z + c0.w * a.w + c1.x * b.x + c1.y * b.y + c1.z * b.z + c1.w * b.w);
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < RR; ++q)
#pragma unroll
        for (int v = 0; v < MV; ++v) {
            if (v >= nv) break;
            const float t = warp_sum(acc[q][v]);
            if (!lane && row0 + q < N) Y[(size_t) v * ldy + row0 + q] = t;
        }
}
// per head (128): RMS norm with weight w, then (rope) NeoX rotary at position pos0 + t, base theta; grid (heads, T)
__global__ void k_mtp_norm_rope(float* x, int ld, const float* w, int pos0, float theta, int rope) {
    __shared__ float r[128];
    float* v = x + (size_t) blockIdx.y * ld + blockIdx.x * 128;
    const int i = threadIdx.x;
    const float a = v[i];
    const float ss = block_sum(a * a);
    const float y = a * rsqrtf(ss / 128.0f + EPS) * w[i];
    if (!rope) { v[i] = y; return; }
    r[i] = y;
    __syncthreads();
    const int j = i & 63;
    const float ang = (float) (pos0 + (int) blockIdx.y) * powf(theta, -2.0f * (float) j / 128.0f);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    v[i] = i < 64 ? r[i] * cs - r[i + 64] * sn : r[i - 64] * sn + r[i] * cs;
}
// drafter attention: query t of head h against the context cells [c0, c1) (ring cap, fp16) and the T block keys (all
// visible: the block is non-causal); 4 warps, keys strided over them, lane = 4 dims; grid (24, T)
__global__ void k_mtp_attn(const float* __restrict__ q, const __half* __restrict__ Kc, const __half* __restrict__ Vc,
                           int cap, int c0, int c1, const float* __restrict__ kb, const float* __restrict__ vb, int T,
                           float* __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y, g = h / NREP, wp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const float4 qv = *reinterpret_cast<const float4*>(q + (size_t) t * QD + h * HD + 4 * lane);
    const float scale = rsqrtf((float) HD);
    float m = -INFINITY, l = 0.0f, o[4] = {0, 0, 0, 0};
    const int n = (c1 - c0) + T;
    for (int j = wp; j < n; j += 4) {
        float4 kk, vv;
        if (j < c1 - c0) {
            const size_t row = (size_t) ((c0 + j) % cap) * KVD + g * HD + 4 * lane;
            const __half2 k01 = *reinterpret_cast<const __half2*>(Kc + row), k23 = *reinterpret_cast<const __half2*>(Kc + row + 2);
            const __half2 v01 = *reinterpret_cast<const __half2*>(Vc + row), v23 = *reinterpret_cast<const __half2*>(Vc + row + 2);
            const float2 a = __half22float2(k01), b = __half22float2(k23), c = __half22float2(v01), d = __half22float2(v23);
            kk = make_float4(a.x, a.y, b.x, b.y);
            vv = make_float4(c.x, c.y, d.x, d.y);
        } else {
            const size_t row = (size_t) (j - (c1 - c0)) * KVD + g * HD + 4 * lane;
            kk = *reinterpret_cast<const float4*>(kb + row);
            vv = *reinterpret_cast<const float4*>(vb + row);
        }
        const float s = warp_sum(qv.x * kk.x + qv.y * kk.y + qv.z * kk.z + qv.w * kk.w) * scale;
        const float mn = fmaxf(m, s), cr = expf(m - mn), e = expf(s - mn);
        l = l * cr + e;
        o[0] = o[0] * cr + e * vv.x; o[1] = o[1] * cr + e * vv.y; o[2] = o[2] * cr + e * vv.z; o[3] = o[3] * cr + e * vv.w;
        m = mn;
    }
    __shared__ float sm_m[4], sm_l[4], sm_o[4][HD];
    if (!lane) { sm_m[wp] = m; sm_l[wp] = l; }
    for (int d = 0; d < 4; ++d) sm_o[wp][4 * lane + d] = o[d];
    __syncthreads();
    if (wp) return;
    float M = -INFINITY;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, sm_m[w]);
    float L = 0.0f, acc[4] = {0, 0, 0, 0};
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = expf(sm_m[w] - M);
        L += f * sm_l[w];
        for (int d = 0; d < 4; ++d) acc[d] += f * sm_o[w][4 * lane + d];
    }
    for (int d = 0; d < 4; ++d) out[(size_t) t * QD + h * HD + 4 * lane + d] = acc[d] / L;
}
__global__ void k_mtp_silu_mul(const float* g, const float* u, float* a, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}
// out[t] = sum_j w[j] F[t][j]  (F [n][4][H])
__global__ void k_mtp_mix(const float* F, const float* w, float* out, int n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * H) return;
    const size_t t = i / H, c = i % H;
    const float* f = F + t * 4 * H + c;
    out[i] = w[0] * f[0] + w[1] * f[H] + w[2] * f[2 * H] + w[3] * f[3 * H];
}
// K/V rows (fp32 [n][KVD]) into the fp16 cache cells (p0 + t) % cap
__global__ void k_mtp_store(const float* k, const float* v, __half* Kc, __half* Vc, int cap, int p0, int n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * KVD) return;
    const size_t t = i / KVD, c = i % KVD, row = (size_t) ((p0 + (int) t) % cap) * KVD + c;
    Kc[row] = __float2half_rn(k[i]);
    Vc[row] = __float2half_rn(v[i]);
}
// first maximum of each row (the reference's tie rule); grid rows, block 1024
__global__ void k_argmax_rows(const float* lg, int ld, int n, int* out) {
    const float* x = lg + (size_t) blockIdx.x * ld;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (x[i] > bv) { bv = x[i]; bi = i; }
    __shared__ float sv[32];
    __shared__ int si[32];
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (!(threadIdx.x & 31)) { sv[threadIdx.x / 32] = bv; si[threadIdx.x / 32] = bi; }
    __syncthreads();
    if (threadIdx.x) return;
    for (int w = 1; w < (int) (blockDim.x / 32); ++w)
        if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
    out[blockIdx.x] = bi;
}
// head input [state 256 | w1[prev] 256 | hidden H | context H]; w1 in mapped pinned host memory (BF16 [V][256])
__global__ void k_mtp_joint_in(const float* state, const uint16_t* w1, const int* prev, const float* hid, const float* ctx,
                               float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 512 + 2 * H) return;
    float v;
    if (i < 256) v = state[i];
    else if (i < 512) v = __uint_as_float((uint32_t) w1[(size_t) (*prev) * 256 + (i - 256)] << 16);
    else if (i < 512 + H) v = hid[i - 512];
    else v = ctx[i - 512 - H];
    out[i] = v;
}
// joint (768, bias added here): gate / candidate / latent; state = cand + gate (state - cand); lat = tanh(c2)
__global__ void k_mtp_gates(const float* j, const float* b, float* state, float* lat) {
    const int i = threadIdx.x;   // 256
    const float g = 1.0f / (1.0f + expf(-(j[i] + b[i]))), c = tanhf(j[256 + i] + b[256 + i]);
    state[i] = c + g * (state[i] - c);
    lat[i] = tanhf(j[512 + i] + b[512 + i]);
}
// v = hidden + corr + bias
__global__ void k_mtp_add3(const float* h, const float* c, const float* b, float* v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) v[i] = h[i] + c[i] + b[i];
}
// feature capture: F[t][j] = x[t] (x [T][H]); grid T
__global__ void k_feat_put(const float* x, float* F, int j) {
    const float* src = x + (size_t) blockIdx.x * H;
    float* dst = F + ((size_t) blockIdx.x * 4 + j) * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}
// conv-history repair after a partial accept: entry e = (hist, save, backup, C); rows of the positions q = c_end - 3 ..
// c_end - 1 (ring q % 3) back from the saved inputs (q >= P) or the snapshot (q < P)
struct HistFix { float* hist; const float* save; float* backup; int C; };
__global__ void k_hist_snap(const HistFix* f) {
    const HistFix e = f[blockIdx.x];
    for (int i = threadIdx.x; i < 3 * e.C; i += blockDim.x) e.backup[i] = e.hist[i];
}
__global__ void k_hist_fix(const HistFix* f, int P, int c_end) {
    const HistFix e = f[blockIdx.x];
    for (int q = max(0, c_end - 3); q < c_end; ++q) {
        float* dst = e.hist + (size_t) (q % 3) * e.C;
        const float* src = q >= P ? e.save + (size_t) (q - P) * e.C : e.backup + (size_t) (q % 3) * e.C;
        for (int i = threadIdx.x; i < e.C; i += blockDim.x) dst[i] = src[i];
    }
}
// conv inputs of a verify chunk, kept for k_hist_fix: save[t] = x[t] (row stride sx), T rows
__global__ void k_conv_save(const float* x, int sx, float* save, int C, int T) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    save[i] = x[(i / C) * sx + i % C];
}

// y = rms(x) (* w if w): one block
__global__ void k_rms(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n) {
    x += (size_t) blockIdx.x * n;
    y += (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = w ? w[i] * (x[i] * r) : x[i] * r;
}

// attention residual: stack [S][H] (contiguous), NQ queries [NQ][H] -> out [NQ][H] = sum_j softmax_j(q . n_j / sqrt H)
// n_j, n_j = rms(stack_j) (unweighted); q . n_j = rinv_j * (q . stack_j).  One block of 256 threads (9 columns each),
// S and NQ compile-time (registers only).  With `outn`, also outn[k] = rms(out[k]) * nw[k] (nw[k] null: unweighted);
// `out` may then be null.
constexpr int AR_T = 256, AR_C = H / AR_T;
template <int S, int NQ>
__global__ void k_attnres(const float* __restrict__ stack, size_t sstride, const float* __restrict__ queries,
                          float* __restrict__ out, const float* nw0, const float* nw1, const float* nw2,
                          float* __restrict__ outn) {
    stack += (size_t) blockIdx.x * H;   // a grid of T tokens: stack entry j of token t at stack + j * sstride + t * H
    if (out) out += (size_t) blockIdx.x * NQ * H;
    if (outn) outn += (size_t) blockIdx.x * NQ * H;
    constexpr int NV = S + NQ * S;
    __shared__ float red[AR_T / 32][NV + NQ];
    __shared__ float p[NQ][S], rinv[S], onorm[NQ];
    const int t = threadIdx.x, w = t / 32, l = t & 31;
    float v[S][AR_C];
    float acc[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int c = 0; c < AR_C; ++c) {
        const int i = c * AR_T + t;
        float q[NQ];
#pragma unroll
        for (int k = 0; k < NQ; ++k) q[k] = queries[(size_t) k * H + i];
#pragma unroll
        for (int j = 0; j < S; ++j) {
            v[j][c] = stack[j * sstride + i];
            acc[j] += v[j][c] * v[j][c];
#pragma unroll
            for (int k = 0; k < NQ; ++k) acc[S + k * S + j] += q[k] * v[j][c];
        }
    }
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const float r = warp_sum(acc[i]);
        if (!l) red[w][i] = r;
    }
    __syncthreads();
    if (t < NV) {
        float r = 0.0f;
#pragma unroll
        for (int k = 0; k < AR_T / 32; ++k) r += red[k][t];
        red[0][t] = r;
    }
    __syncthreads();
    if (t < S) rinv[t] = rsqrtf(red[0][t] / H + EPS);
    __syncthreads();
    if (t < NQ) {
        float lg[S], mx = -INFINITY, z = 0.0f;
#pragma unroll
        for (int j = 0; j < S; ++j) { lg[j] = red[0][S + t * S + j] * rinv[j] * rsqrtf((float) H); mx = fmaxf(mx, lg[j]); }
#pragma unroll
        for (int j = 0; j < S; ++j) { lg[j] = expf(lg[j] - mx); z += lg[j]; }
#pragma unroll
        for (int j = 0; j < S; ++j) p[t][j] = lg[j] / z * rinv[j];
    }
    __syncthreads();
    float o[NQ][AR_C], ss[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        ss[k] = 0.0f;
#pragma unroll
        for (int c = 0; c < AR_C; ++c) {
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < S; ++j) a += p[k][j] * v[j][c];
            o[k][c] = a;
            ss[k] += a * a;
            if (out) out[(size_t) k * H + c * AR_T + t] = a;
        }
    }
    if (!outn) return;
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const float r = warp_sum(ss[k]);
        if (!l) red[w][NV + k] = r;
    }
    __syncthreads();
    if (t < NQ) {
        float r = 0.0f;
#pragma unroll
        for (int k = 0; k < AR_T / 32; ++k) r += red[k][NV + t];
        onorm[t] = rsqrtf(r / H + EPS);
    }
    __syncthreads();
    const float* nw[3] = {nw0, nw1, nw2};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const float* wk = k < 3 ? nw[k] : nullptr;
#pragma unroll
        for (int c = 0; c < AR_C; ++c) {
            const int i = c * AR_T + t;
            outn[(size_t) k * H + i] = wk ? wk[i] * (o[k][c] * onorm[k]) : o[k][c] * onorm[k];
        }
    }
}

// rms of two vectors at once (blockIdx.x 0: x, 1: m), unweighted: gate j's [rms(x), rms(m)] into cat[j] [2H]
__global__ void k_rms2(GateArgs g, float* __restrict__ cat) {
    const int j = blockIdx.y;
    const float* a = blockIdx.x ? g.m[j] : g.x[j];
    float* b = cat + (size_t) (2 * j + blockIdx.x) * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += a[i] * a[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) b[i] = a[i] * r;
}

// small-N matrix-vector product: one block per row (BF16 W [N, K]), for N too small to fill the GPU warp-per-row
__device__ __forceinline__ float rowblock_dot(const uint16_t* __restrict__ W, const float* __restrict__ x, int K) {
    const uint4* w = reinterpret_cast<const uint4*>(W + (size_t) blockIdx.x * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < K / 8; i += blockDim.x) {
        const uint4 v = __ldg(w + i);
        const float4 a = __ldg(x4 + 2 * i), b = __ldg(x4 + 2 * i + 1);
        acc += bf16lo(v.x) * a.x + bf16hi(v.x) * a.y + bf16lo(v.y) * a.z + bf16hi(v.y) * a.w +
               bf16lo(v.z) * b.x + bf16hi(v.z) * b.y + bf16lo(v.w) * b.z + bf16hi(v.w) * b.w;
    }
    return block_sum(acc);
}
__global__ void k_gemv_bf16_rowblock(const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                     int K) {
    const float acc = rowblock_dot(W, x, K);
    if (!threadIdx.x) y[blockIdx.x] = acc;
}
// gate j's down projection: gdown[j] [GATE_RANK] = down_j . cat[j]; grid (GATE_RANK, gates)
__global__ void k_gate_down(GateArgs g, const float* __restrict__ cat, float* __restrict__ gdown, int K) {
    const int j = blockIdx.y;
    const float acc = rowblock_dot(g.down[j], cat + (size_t) j * 2 * H, K);
    if (!threadIdx.x) gdown[(size_t) j * GATE_RANK + blockIdx.x] = acc;
}

// message gate tail + the following norm: y = (1 - tanh f) * x + tanh g * m; yn = rms(y) * w.  One block per gate.
__global__ void k_gate_mix_rms(GateArgs ga, const float* gfg) {
    const int j = blockIdx.x;
    const float *x = ga.x[j], *m = ga.m[j], *f = gfg + (size_t) j * 2 * H, *g = f + H, *w = ga.nw[j];
    float *y = ga.y[j], *yn = ga.yn[j];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
        y[i] = a;
        ss += a * a;
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) yn[i] = w[i] * (y[i] * r);
}


__device__ __forceinline__ float log_scale_at(int pos) {
    return 1.0f + LOG_ALPHA * logf(fmaxf((float) (pos + 1) / LOG_FLOOR, 1.0f));
}

// head hd of v: v *= scale; L2 clip to thr; (log scale on full layers).  block = 128 threads, dim <= 128
__device__ __forceinline__ void head_clip1(float* v, const float* scale, int dim, float thr, int log_scaled, int pos,
                                           int hd) {
    float* x = v + (size_t) hd * dim;
    const int i = threadIdx.x;
    float a = i < dim ? x[i] * (scale ? scale[(size_t) hd * dim + i] : 1.0f) : 0.0f;
    const float norm = sqrtf(block_sum(a * a));
    a *= fminf(thr / (norm + 1e-6f), 1.0f);
    if (log_scaled) a *= log_scale_at(pos);
    if (i < dim) x[i] = a;
}

// Inkling2LoopShortConvolution: hist = 3 rows (row p % 3 holds position p's input)
__device__ __forceinline__ float short_conv1(float xc, float* __restrict__ hist, const float* __restrict__ w, int C, int c,
                                             int p) {
    float acc = w[(size_t) c * KC + KC - 1] * xc;
    for (int j = 1; j < KC; ++j)
        if (p - j >= 0) acc += w[(size_t) c * KC + KC - 1 - j] * hist[(size_t) ((p - j) % 3) * C + c];
    hist[(size_t) (p % 3) * C + c] = xc;   // after the reads: lag 3 is the row p % 3
    return __fmul_rn(acc + xc, rsqrtf((float) (KC + 1)));   // _rn: no FMA contraction with the caller's residual add
}
// attention tail: conv of the attention output, then either (block boundary) r = y also stored as the next block state,
// or r = h + y also stored over the layer's stack entry
__global__ void k_attn_tail(const float* __restrict__ x, float* __restrict__ hist, const float* __restrict__ w,
                            const float* __restrict__ h, float* __restrict__ r, float* __restrict__ r_copy, int boundary,
                            const int* d_pos) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    const float y = short_conv1(x[c], hist, w, H, c, *d_pos);
    const float v = boundary ? y : h[c] + y;
    r[c] = v;
    r_copy[c] = v;
}
// MoE tail: combine the TOPK expert outputs, conv, residual: out = r + conv(sum_k y_k w_k)
__global__ void k_moe_tail(const float* __restrict__ ey, const float* __restrict__ wts, float* __restrict__ hist,
                           const float* __restrict__ w, const float* __restrict__ r, float* __restrict__ out,
                           const int* d_pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float a = 0.0f;
    for (int k = 0; k < TOPK; ++k) a += ey[(size_t) k * H + i] * wts[k];
    out[i] = r[i] + short_conv1(a, hist, w, H, i, *d_pos);
}

// Inkling2LoopDynamicShortConvolution: taps static_w[lag][c] + dw[(c / 128) * 4 + lag]
__device__ __forceinline__ float dyn_conv1(float xc, const float* __restrict__ dw, float* __restrict__ hist,
                                           const float* __restrict__ static_w, int C, int c, int p) {
    const int head = c / (C / NKV);
    float acc = 0.0f;
    for (int lag = 0; lag < KC; ++lag) {
        const float tap = static_w[(size_t) lag * C + c] + dw[head * KC + lag];
        if (lag == 0) acc += tap * xc;
        else if (p - lag >= 0) acc += tap * hist[(size_t) ((p - lag) % 3) * C + c];
    }
    hist[(size_t) (p % 3) * C + c] = xc;
    return __fmul_rn(acc + xc, rsqrtf((float) (2 * KC + 1)));
}

// KV cache cells [cap][KVD], format c_kvm (--kv): 0 fp32; 1 fp16 (round to nearest); 2 Strata's int8 layout
// (kv_q8.hpp): codes + one fp16 scale per 64 values [cap][KVD / 64] (ksc / vsc); scale = fp16(max|x| / 127),
// code = clamp(rint(x / scale), -127, 127).  K is L2-clipped and V RMS-normalized per head before the store.
constexpr int KVG = 64;
constexpr int V4G = 32;   // --kv k8v4: V values per fp16 scale
__constant__ int c_kvm;   // 3 = int8 with Walsh-Hadamard rotation (Strata's kv_int8_rotate): K, V, q rotated per head,
                          // the attention output rotated back (H orthonormal and symmetric: scores unchanged);
                          // 4 = k8v4: K as in 2, V rotated as in 3 and stored as 4-bit codes [cap][KVD / 2] (low nibble
                          // first) with one fp16 scale per 32 values [cap][KVD / 32]; q is not rotated (K is not)
                          // 5 = k8v8: K as in 2, V rotated as in 3 and stored int8 as in 2; q not rotated
__host__ __device__ constexpr bool kv_rot_v(int kvm) { return kvm >= 3; }   // the output is rotated back
// orthonormal Walsh-Hadamard transform of one 128-vector held one value per thread by a 128-thread block
__device__ __forceinline__ float wht128(float x) {
    __shared__ float r[128];
    const int t = threadIdx.x;
    __syncthreads();   // r reused across calls
    r[t] = x;
#pragma unroll
    for (int st = 1; st < 128; st <<= 1) {
        __syncthreads();
        const float a = r[t], b = r[t ^ st];
        __syncthreads();
        r[t] = (t & st) ? b - a : a + b;
    }
    __syncthreads();
    return r[t] * 0.08838834764831845f;   // 1 / sqrt(128)
}
// VV: the V cache (k8v4 differs between K and V)
template <bool VV>
__device__ __forceinline__ float4 kv_ld4(const void* B, const __half* sc, size_t row, int e) {
    if (VV && c_kvm == 4) {   // 4 nibbles (e % 4 == 0): 2 bytes, low nibble first
        const uint32_t u = *reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(B) + row * (KVD / 2) + e / 2);
        const float f = __half2float(sc[row * (KVD / V4G) + e / V4G]);
        return make_float4((float) ((int) (u & 15u) - 8) * f, (float) ((int) ((u >> 4) & 15u) - 8) * f,
                           (float) ((int) ((u >> 8) & 15u) - 8) * f, (float) ((int) (u >> 12) - 8) * f);
    }
    if (c_kvm == 0) return *reinterpret_cast<const float4*>(static_cast<const float*>(B) + row * KVD + e);
    if (c_kvm == 1) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __half*>(B) + row * KVD + e);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
    const char4 c = *reinterpret_cast<const char4*>(static_cast<const int8_t*>(B) + row * KVD + e);
    const float f = __half2float(sc[row * (KVD / KVG) + e / KVG]);
    return make_float4(c.x * f, c.y * f, c.z * f, c.w * f);
}
template <bool VV>
__device__ __forceinline__ float kv_ld1(const void* B, const __half* sc, size_t row, int e) {
    if (VV && c_kvm == 4) {
        const uint32_t b = static_cast<const uint8_t*>(B)[row * (KVD / 2) + e / 2];
        return (float) ((int) ((e & 1) ? b >> 4 : b & 15u) - 8) * __half2float(sc[row * (KVD / V4G) + e / V4G]);
    }
    if (c_kvm == 0) return static_cast<const float*>(B)[row * KVD + e];
    if (c_kvm == 1) return __half2float(static_cast<const __half*>(B)[row * KVD + e]);
    return static_cast<const int8_t*>(B)[row * KVD + e] * __half2float(sc[row * (KVD / KVG) + e / KVG]);
}
// one cell value, called by every thread of a 128-thread block, thread i holding element e = g * HD + i (warps 0-1:
// first group of 64, warps 2-3: second)
template <bool VV>
__device__ __forceinline__ void kv_st(void* B, __half* sc, size_t row, int e, float x) {
    if (c_kvm == 0) { static_cast<float*>(B)[row * KVD + e] = x; return; }
    if (c_kvm == 1) { static_cast<__half*>(B)[row * KVD + e] = __float2half_rn(x); return; }
    if (c_kvm == 3 || (VV && c_kvm >= 4)) x = wht128(x);
    if (VV && c_kvm == 4) {   // a warp holds one group of 32: scale = fp16(max|x| / 7), code = clamp(rint(x / s), -7, 7) + 8
        const float m = warp_max(fabsf(x));
        const __half sh = __float2half_rn(m / 7.0f);
        const float f = __half2float(sh);
        const int q = f > 0.0f ? max(-7, min(7, __float2int_rn(x / f))) : 0;
        const int qn = __shfl_down_sync(0xffffffffu, q, 1);
        if (!(threadIdx.x & 1)) static_cast<uint8_t*>(B)[row * (KVD / 2) + e / 2] = (uint8_t) ((q + 8) | ((qn + 8) << 4));
        if (!(threadIdx.x & (V4G - 1))) sc[row * (KVD / V4G) + e / V4G] = sh;
        return;
    }
    __shared__ float gm[4];
    const float m = warp_max(fabsf(x));
    __syncthreads();   // gm is reused by the next call
    if (!(threadIdx.x & 31)) gm[threadIdx.x >> 5] = m;
    __syncthreads();
    const int w = threadIdx.x >> 5;
    const __half sh = __float2half_rn(fmaxf(gm[w & ~1], gm[w | 1]) / 127.0f);
    const float f = __half2float(sh);
    const int q = f > 0.0f ? max(-127, min(127, __float2int_rn(x / f))) : 0;
    static_cast<int8_t*>(B)[row * KVD + e] = (int8_t) q;
    if (!(threadIdx.x & (KVG - 1))) sc[row * (KVD / KVG) + e / KVG] = sh;
}

// keys: L2 clip per KV head; values: unweighted RMS per head; stored at position pos (ring: pos % cap)
// k / v convolutions (static or dynamic taps) fused with the store; grid NKV, block HD
template <bool DYN>
__device__ __forceinline__ void conv_kv_store1(const float* k, const float* v, float* khist, float* vhist, const float* kw,
                                               const float* vw, void* K, void* V, __half* ksc, __half* vsc, int cap,
                                               const int* d_pos, int g) {
    const int i = threadIdx.x, c = g * HD + i, p = *d_pos;
    const float kv = DYN ? dyn_conv1(k[c], k + KVD, khist, kw, KVD, c, p) : short_conv1(k[c], khist, kw, KVD, c, p);
    const float vv = DYN ? dyn_conv1(v[c], v + KVD, vhist, vw, KVD, c, p) : short_conv1(v[c], vhist, vw, KVD, c, p);
    const float kn = sqrtf(block_sum(kv * kv));
    const float vr = rsqrtf(block_sum(vv * vv) / HD + EPS);
    const size_t row = (size_t) ((*d_pos) % cap);
    kv_st<false>(K, ksc, row, c, kv * fminf(QK_CLIP / (kn + 1e-6f), 1.0f));
    kv_st<true>(V, vsc, row, c, vv * vr);
}

// Attention prep, one launch (block = 128): blocks [0, NQ) clip q head b; [NQ, 2 NQ) clip rs head h then
// rel[h][d] = sum_r rs[h][r] proj[r][d]; [2 NQ, 2 NQ + NKV) conv + clip/rms + store of k / v group g.  Same arithmetic as
// the former four kernels (k_head_clip x2, k_rel_logits, k_conv_kv_store).
template <bool DYN>
__global__ void k_attn_prep(float* q, const float* q_scale, float* rs, const float* rel_scale, const float* rel_proj,
                            float* rel, int log_scaled, const float* k, const float* v, float* khist, float* vhist,
                            const float* kw, const float* vw, void* K, void* V, __half* ksc, __half* vsc, int cap,
                            const int* d_pos) {
    const int b = blockIdx.x;
    if (b < NQ) {
        head_clip1(q, q_scale, HD, QK_CLIP, log_scaled, *d_pos, b);
        if (c_kvm == 3) q[b * HD + threadIdx.x] = wht128(q[b * HD + threadIdx.x]);
    } else if (b < 2 * NQ) {
        const int h = b - NQ;
        head_clip1(rs, rel_scale, DREL, REL_CLIP, log_scaled, *d_pos, h);
        __syncthreads();   // the clipped rs of head h, written by this block
        for (int d = threadIdx.x; d < EXTENT; d += blockDim.x) {
            float a = 0.0f;
#pragma unroll
            for (int r = 0; r < DREL; ++r) a += rs[h * DREL + r] * rel_proj[r * EXTENT + d];
            rel[h * EXTENT + d] = a;
        }
    } else {
        conv_kv_store1<DYN>(k, v, khist, vhist, kw, vw, K, V, ksc, vsc, cap, d_pos, b - 2 * NQ);
    }
}

// partial attention: block (head h, chunk c) covers distances [c*CHUNK, c*CHUNK + CHUNK) back from pos; window 0 = full.
// writes m, l (sum of exp) and o[HD] (unnormalized) for the merge.  block = 128 threads (4 warps).
__global__ void k_attn_partial(const float* __restrict__ q, const void* __restrict__ K, const void* __restrict__ V,
                               const __half* __restrict__ ksc, const __half* __restrict__ vsc, const float* __restrict__ rel, int cap, int window, const int* d_pos,
                               float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ po) {
    __shared__ __align__(16) float qs[HD];
    __shared__ float lg[CHUNK];
    const int h = blockIdx.x, c = blockIdx.y, g = h / NREP, tid = threadIdx.x, lane = tid & 31, wp = tid / 32;
    const int pos = *d_pos;
    const int span = window > 0 ? min(pos + 1, window) : pos + 1;   // distances 0 .. span-1
    const int d0 = c * CHUNK, n = min(CHUNK, span - d0);
    const size_t part = (size_t) h * gridDim.y + c;
    if (n <= 0) {
        if (!tid) { pm[part] = -INFINITY; pl[part] = 0.0f; }
        po[part * HD + tid] = 0.0f;
        return;
    }
    qs[tid] = q[h * HD + tid];
    __syncthreads();
    const float scale = rsqrtf((float) HD);
    // ring slot of distance d0 + k: r0 - k, wrapped once (k < CHUNK <= cap)
    const int r0 = (pos - d0) % cap;
    // q.k: 8 lanes per position (4 positions per warp, 16 per block step), each lane 4 float4 of the row
    {
        const int sub = lane & 7;
        const float4* q4 = reinterpret_cast<const float4*>(qs);
        for (int k0 = wp * 4; k0 < n; k0 += 16) {
            const int k = k0 + (lane >> 3);
            float a = 0.0f;
            if (k < n) {
                const int r = r0 - k < 0 ? r0 - k + cap : r0 - k;
                float4 kv[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) kv[u] = kv_ld4<false>(K, ksc, (size_t) r, g * HD + 4 * (sub + 8 * u));
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float4 qv = q4[sub + 8 * u];
                    a += qv.x * kv[u].x + qv.y * kv[u].y + qv.z * kv[u].z + qv.w * kv[u].w;
                }
            }
            a += __shfl_xor_sync(0xffffffffu, a, 4);
            a += __shfl_xor_sync(0xffffffffu, a, 2);
            a += __shfl_xor_sync(0xffffffffu, a, 1);
            const int dist = d0 + k;
            if (!sub && k < n) lg[k] = a * scale + (dist < EXTENT ? rel[h * EXTENT + dist] : 0.0f);
        }
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int k = tid; k < n; k += blockDim.x) mx = fmaxf(mx, lg[k]);
    mx = block_max(mx);
    float s = 0.0f;
    for (int k = tid; k < n; k += blockDim.x) { lg[k] = expf(lg[k] - mx); s += lg[k]; }
    s = block_sum(s);   // block_sum syncs, lg visible after
    __syncthreads();
    // p.V: warp wp takes positions wp, wp + 4, ...; lane holds dims 4 lane .. 4 lane + 3; the 4 partial rows summed
    // through smem (positions spread over the warps: 4x more loads in flight than one position per block step)
    __shared__ float4 red[4][HD / 4];
    float4 o4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int ve = g * HD + 4 * lane;
#pragma unroll 4
    for (int k = wp; k < n; k += 4) {
        const int r = r0 - k < 0 ? r0 - k + cap : r0 - k;
        const float4 v = kv_ld4<true>(V, vsc, (size_t) r, ve);
        const float pk = lg[k];
        o4.x += pk * v.x; o4.y += pk * v.y; o4.z += pk * v.z; o4.w += pk * v.w;
    }
    red[wp][lane] = o4;
    __syncthreads();
    const float* rf = reinterpret_cast<const float*>(red);
    const float o = rf[tid] + rf[HD + tid] + rf[2 * HD + tid] + rf[3 * HD + tid];
    if (!tid) { pm[part] = mx; pl[part] = s; }
    po[part * HD + tid] = o;
}

// merge the chunks of each head with its sink logit (zero value); out[h][HD]
__global__ void k_attn_merge(const float* pm, const float* pl, const float* po, int nchunk, const float* sink,
                             const float* gate, float* out) {
    const int h = blockIdx.x, tid = threadIdx.x;
    float M = sink[h];
    for (int c = 0; c < nchunk; ++c) M = fmaxf(M, pm[h * nchunk + c]);
    float L = expf(sink[h] - M), o = 0.0f;
    for (int c = 0; c < nchunk; ++c) {
        const float m = pm[h * nchunk + c];
        if (m == -INFINITY) continue;
        const float f = expf(m - M);
        L += f * pl[h * nchunk + c];
        o += f * po[((size_t) h * nchunk + c) * HD + tid];
    }
    float a = o / L;   // then the output gate: a * sigmoid(g)
    if (kv_rot_v(c_kvm)) a = wht128(a);   // rotated V: the output comes back with the same transform
    out[h * HD + tid] = a * (1.0f / (1.0f + expf(-gate[h * HD + tid])));
}


// rotate head blockIdx.x of row blockIdx.y in place (--kv i8r): prefill q before attention, output after; block 128
__global__ void kb_wht(float* x, int stride) {
    float* v = x + (size_t) blockIdx.y * stride + blockIdx.x * HD;
    v[threadIdx.x] = wht128(v[threadIdx.x]);
}

// router: sigmoid scores, top-8 on score + bias, weights score / sum * 8.  one block of NE threads
__global__ void k_router(const float* logits, const float* bias, int* ids, float* wts) {
    logits += (size_t) blockIdx.x * NE;
    ids += blockIdx.x * TOPK;
    wts += blockIdx.x * TOPK;
    __shared__ float sel[NE], sc[NE];
    __shared__ int pick[TOPK];
    const int e = threadIdx.x;
    sc[e] = 1.0f / (1.0f + expf(-logits[e]));
    sel[e] = sc[e] + bias[e];
    if (e < TOPK) pick[e] = e;   // only NaN scores leave a rank unfilled: keep the ids in range then
    __syncthreads();
    // rank of e = number of experts ahead of it (higher score, ties: lower index); ranks 0..TOPK-1 are the picks in
    // the order successive argmaxes (ties to the lowest index) would give them
    const float me = sel[e];
    int rank = 0;
    for (int i = 0; i < NE; ++i) {
        const float o = sel[i];
        rank += (o > me) || (o == me && i < e);
    }
    if (rank < TOPK) pick[rank] = e;
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < TOPK; ++k) sum += sc[pick[k]];
        for (int k = 0; k < TOPK; ++k) { ids[k] = pick[k]; wts[k] = sc[pick[k]] / (sum + 1e-20f) * ROUTE_SCALE; }
    }
}



__global__ void k_softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill kernels: T tokens at once (positions pos0 .. pos0 + T - 1), activations row-major [T][dim] with a row stride.
// Same fp32 arithmetic as the one-token kernels; only the summation order of the matrix products differs (cuBLAS SGEMM,
// full FP32, no TF32).

__global__ void kb_bf16_to_f32(const uint16_t* __restrict__ src, float* __restrict__ dst, size_t n8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(src) + i);
    float4* d = reinterpret_cast<float4*>(dst) + 2 * i;
    d[0] = make_float4(bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y));
    d[1] = make_float4(bf16lo(v.z), bf16hi(v.z), bf16lo(v.w), bf16hi(v.w));
}

// [rms(x_t), rms(m_t)] -> cat[t][2H]; grid (2, T)
__global__ void kb_rms2(const float* __restrict__ x, int sx, const float* __restrict__ m, int sm, float* __restrict__ cat) {
    const int t = blockIdx.y;
    const float* a = blockIdx.x ? m + (size_t) t * sm : x + (size_t) t * sx;
    float* b = cat + (size_t) t * 2 * H + (size_t) blockIdx.x * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += a[i] * a[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) b[i] = a[i] * r;
}

// yn_t = rms((1 - tanh f) * x + tanh g * m) * w; fg [T][2H]; grid T
__global__ void kb_gate_mix_rms(const float* x, int sx, const float* m, int sm, const float* fg, const float* w,
                                float* yn) {
    const int t = blockIdx.x;
    x += (size_t) t * sx;
    m += (size_t) t * sm;
    const float* f = fg + (size_t) t * 2 * H;
    const float* g = f + H;
    yn += (size_t) t * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
        ss += a * a;
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
        yn[i] = w[i] * (a * r);
    }
}

// per head and token: v *= scale; L2 clip; (log scale).  grid (heads, T), block 128
__global__ void kb_head_clip(float* v, int sv, const float* scale, int dim, float thr, int log_scaled, int pos0) {
    float* x = v + (size_t) blockIdx.y * sv + (size_t) blockIdx.x * dim;
    const int i = threadIdx.x;
    float a = i < dim ? x[i] * (scale ? scale[(size_t) blockIdx.x * dim + i] : 1.0f) : 0.0f;
    const float norm = sqrtf(block_sum(a * a));
    a *= fminf(thr / (norm + 1e-6f), 1.0f);
    if (log_scaled) a *= log_scale_at(pos0 + blockIdx.y);
    if (i < dim) x[i] = a;
}

// causal depthwise conv over the chunk; inputs before the chunk come from hist (row p % 3).  grid (C / 256, T).
// dw (dynamic taps, full layers): per token at dw + t * sx, [NKV][KC]; static_w then [KC][C]; else w [C][KC]
__global__ void kb_conv(const float* __restrict__ x, int sx, const float* __restrict__ hist, const float* __restrict__ w,
                        const float* __restrict__ dw, float* __restrict__ y, int sy, int C, int pos0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, p = pos0 + t;
    if (c >= C) return;
    const float xc = x[(size_t) t * sx + c];
    float acc;
    if (dw) {
        const float* d = dw + (size_t) t * sx + (c / (C / NKV)) * KC;
        acc = (w[c] + d[0]) * xc;
        for (int lag = 1; lag < KC; ++lag) {
            if (p - lag < 0) break;
            const float v = t - lag >= 0 ? x[(size_t) (t - lag) * sx + c] : hist[(size_t) ((p - lag) % 3) * C + c];
            acc += (w[(size_t) lag * C + c] + d[lag]) * v;
        }
        y[(size_t) t * sy + c] = (acc + xc) * rsqrtf((float) (2 * KC + 1));
    } else {
        acc = w[(size_t) c * KC + KC - 1] * xc;
        for (int j = 1; j < KC; ++j) {
            if (p - j < 0) break;
            const float v = t - j >= 0 ? x[(size_t) (t - j) * sx + c] : hist[(size_t) ((p - j) % 3) * C + c];
            acc += w[(size_t) c * KC + KC - 1 - j] * v;
        }
        y[(size_t) t * sy + c] = (acc + xc) * rsqrtf((float) (KC + 1));
    }
}
// after kb_conv: the chunk's last (up to 3) inputs into hist.  grid (C / 256, min(3, T))
__global__ void kb_hist_store(const float* __restrict__ x, int sx, float* __restrict__ hist, int C, int T, int pos0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = T - 1 - blockIdx.y;
    if (c < C) hist[(size_t) ((pos0 + t) % 3) * C + c] = x[(size_t) t * sx + c];
}

// grid (NKV, T), block HD
__global__ void kb_kv_store(const float* kc, const float* vc, void* K, void* V, __half* ksc, __half* vsc, int cap,
                            int pos0) {
    const int g = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const float kv = kc[(size_t) t * KVD + g * HD + i], vv = vc[(size_t) t * KVD + g * HD + i];
    const float kn = sqrtf(block_sum(kv * kv));
    const float vr = rsqrtf(block_sum(vv * vv) / HD + EPS);
    const size_t row = (size_t) ((pos0 + t) % cap);
    kv_st<false>(K, ksc, row, g * HD + i, kv * fminf(QK_CLIP / (kn + 1e-6f), 1.0f));
    kv_st<true>(V, vsc, row, g * HD + i, vv * vr);
}

// rel[t][h][d] = sum_r rs_t[h][r] proj[r][d]; grid (NQ, T)
__global__ void kb_rel_logits(const float* rs, int sr, const float* proj, float* rel) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* r0 = rs + (size_t) t * sr + h * DREL;
    float* o = rel + ((size_t) t * NQ + h) * EXTENT;
    for (int d = threadIdx.x; d < EXTENT; d += blockDim.x) {
        float a = 0.0f;
#pragma unroll
        for (int r = 0; r < DREL; ++r) a += r0[r] * proj[r * EXTENT + d];
        o[d] = a;
    }
}

// Tiled causal attention, fp32: block = (64 queries, one q head), 256 threads; key tiles of 64 from the cache (ring
// index j % cap).  Online softmax started from the sink (m = sink, l = 1, o = 0: its value is zero).
// Thread (ty = tid / 16, tx = tid % 16): queries ty*4 .. +3; scores for keys tx + 16 m; outputs dims tx + 16 m.
constexpr int FQ = 64, FK = 64, FQ_LD = HD + 4, FK_LD = HD + 1, FP_LD = FK + 1;
constexpr size_t FA_SMEM = (size_t) (FQ * FQ_LD + FK * FK_LD + FQ * FP_LD) * 4;
__global__ void __launch_bounds__(256) kb_attn(const float* __restrict__ P, int sp, const void* __restrict__ K,
                                               const void* __restrict__ V, const __half* __restrict__ ksc,
                                               const __half* __restrict__ vsc, const float* __restrict__ rel, int cap,
                                               int window, int pos0, int T, float* __restrict__ out) {
    extern __shared__ float fsm[];
    float* Qs = fsm;                    // [FQ][FQ_LD]
    float* KVs = Qs + FQ * FQ_LD;       // keys [FK][FK_LD], then values [FK][HD]
    float* Ps = KVs + FK * FK_LD;       // [FQ][FP_LD]
    const int h = blockIdx.y, g = h / NREP, t0 = blockIdx.x * FQ, tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    const int nq = min(FQ, T - t0);
    for (int i = tid; i < FQ * HD; i += 256) {
        const int r = i / HD, d = i % HD;
        Qs[r * FQ_LD + d] = r < nq ? P[(size_t) (t0 + r) * sp + h * HD + d] : 0.0f;
    }
    float m[4], l[4], o[4][8];
    const float scale = rsqrtf((float) HD);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty * 4 + i;
        m[i] = r < nq ? P[(size_t) (t0 + r) * sp + OFF_SINK + h] : 0.0f;
        l[i] = 1.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[i][j] = 0.0f;
    }
    const int pa = pos0 + t0, pb = pos0 + t0 + nq - 1;   // first / last query position
    const int lo = window > 0 ? max(0, pa - window + 1) : 0;
    for (int k0 = lo; k0 <= pb; k0 += FK) {
        __syncthreads();
        for (int i = tid; i < FK * HD; i += 256) {
            const int r = i / HD, d = i % HD, j = k0 + r;
            KVs[r * FK_LD + d] = j <= pb ? kv_ld1<false>(K, ksc, (size_t) (j % cap), g * HD + d) : 0.0f;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) s[i][c] = 0.0f;
#pragma unroll 4
        for (int d = 0; d < HD; ++d) {
            float q[4], k[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) q[i] = Qs[(ty * 4 + i) * FQ_LD + d];
#pragma unroll
            for (int c = 0; c < 4; ++c) k[c] = KVs[(tx + 16 * c) * FK_LD + d];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) s[i][c] += q[i] * k[c];
        }
        // bias, mask, online softmax (a query row lives in the 16 threads of a half warp)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = ty * 4 + i, p = pos0 + t0 + r;
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int j = k0 + tx + 16 * c, dist = p - j;
                const bool ok = r < nq && j <= p && (window == 0 || dist < window);
                s[i][c] = ok ? s[i][c] * scale + (dist < EXTENT ? rel[((size_t) (t0 + r) * NQ + h) * EXTENT + dist] : 0.0f)
                             : -INFINITY;
                mx = fmaxf(mx, s[i][c]);
            }
            for (int w = 8; w; w >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, w));
            const float mn = fmaxf(m[i], mx), f = expf(m[i] - mn);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float e = expf(s[i][c] - mn);
                Ps[r * FP_LD + tx + 16 * c] = e;
                sum += e;
            }
            for (int w = 8; w; w >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, w);
            l[i] = l[i] * f + sum;
            m[i] = mn;
#pragma unroll
            for (int j = 0; j < 8; ++j) o[i][j] *= f;
        }
        __syncthreads();
        for (int i = tid; i < FK * HD; i += 256) {
            const int r = i / HD, d = i % HD, j = k0 + r;
            KVs[r * HD + d] = j <= pb ? kv_ld1<true>(V, vsc, (size_t) (j % cap), g * HD + d) : 0.0f;
        }
        __syncthreads();
        const int kn = min(FK, pb - k0 + 1);
        for (int k = 0; k < kn; ++k) {
            float pv[4], v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) pv[i] = Ps[(ty * 4 + i) * FP_LD + k];
#pragma unroll
            for (int j = 0; j < 8; ++j) v[j] = KVs[k * HD + tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) o[i][j] += pv[i] * v[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty * 4 + i;
        if (r >= nq) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) out[(size_t) (t0 + r) * QD + h * HD + tx + 16 * j] = o[i][j] / l[i];
    }
}

// att[t][i] *= sigmoid(gate_t[i]); grid (QD / 256, T)
__global__ void kb_mul_sigmoid(float* att, const float* gate, int sg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    att[(size_t) t * QD + i] *= 1.0f / (1.0f + expf(-gate[(size_t) t * sg + i]));
}

// MoE plan (one block, thread = expert): rows of the grouped products sorted by expert then (token, k);
// rows[r] = token, slot[t * TOPK + k] = r; tiles of FM rows within one expert.
constexpr int FM = 64, FN = 64, FKK = 32;
// (one block of 1024 threads; the order of the rows within an expert is not deterministic and does not matter: each
// output row depends only on its own input row)
__global__ void kb_moe_plan(const int* __restrict__ ids, int T, int* rows, int* slot, int* tile_e, int* tile_r0,
                            int* ntiles, int* eoff, int* ecnt) {
    __shared__ int cnt[NE], off[NE], cur[NE];
    const int n = T * TOPK;
    for (int e = threadIdx.x; e < NE; e += blockDim.x) { cnt[e] = 0; cur[e] = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int j = 0; j < NE; ++j) {
            off[j] = o;
            for (int r = 0; r < cnt[j]; r += FM) { tile_e[nt] = j; tile_r0[nt] = o + r; ++nt; }
            o += cnt[j];
        }
        *ntiles = nt;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NE; e += blockDim.x) { eoff[e] = off[e]; ecnt[e] = cnt[e]; }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int e = ids[i], r = off[e] + atomicAdd(&cur[e], 1);
        rows[r] = i / TOPK;
        slot[i] = r;
    }
}

// Grouped expert product with the weights decoded on the fly (exact: e2m1 * e4m3 and e4m3 * 2^k are exact in fp32):
// C[r][n] = (factor_e) * sum_k A[arow(r)][k] W_e[n][k] for the rows r of a tile (expert e).  grid (max tiles, N / FN),
// 256 threads, 4x4 outputs per thread.  arow = rows[r] (gather) or r.
template <bool NV>
__global__ void __launch_bounds__(256) kb_moe_gemm(const float* __restrict__ A, int lda, const int* __restrict__ rows,
                                                   const uint8_t* const* __restrict__ tab,
                                                   const float* __restrict__ factors, const int* __restrict__ tile_e,
                                                   const int* __restrict__ tile_r0, const int* __restrict__ ntiles,
                                                   const int* __restrict__ eoff, const int* __restrict__ ecnt,
                                                   float* __restrict__ C, int N, int K) {
    __shared__ __align__(16) float As[FKK][FM + 4], Ws[FKK][FN + 4];
    if ((int) blockIdx.x >= *ntiles) return;
    const int e = tile_e[blockIdx.x], r0 = tile_r0[blockIdx.x], rend = eoff[e] + ecnt[e], n0 = blockIdx.y * FN;
    const uint8_t* W = tab[e];
    if (!W) return;
    const uint8_t* S = W + (size_t) N * (NV ? K / 2 : K);
    const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    // A loads: 2 float4 per thread (row idx / 8, float4 idx % 8)
    const float* asrc[2];
    int arow_l[2], akq[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int idx = tid + 256 * u, row = idx / 8, r = r0 + row;
        arow_l[u] = row;
        akq[u] = idx % 8;
        asrc[u] = r < rend ? A + (size_t) (rows ? rows[r] : r) * lda + akq[u] * 4 : nullptr;
    }
    const int wn = tid / 4, wq = tid % 4;   // W loads: row n0 + wn, 8 columns from wq * 8
    const size_t wrow = (size_t) n0 + wn;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += FKK) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const float4 v = asrc[u] ? *reinterpret_cast<const float4*>(asrc[u] + k0) : make_float4(0, 0, 0, 0);
            const int kk = akq[u] * 4, row = arow_l[u];
            As[kk][row] = v.x; As[kk + 1][row] = v.y; As[kk + 2][row] = v.z; As[kk + 3][row] = v.w;
        }
        const int kc = k0 + wq * 8;
        if (NV) {
            const uint32_t v = __ldg(reinterpret_cast<const uint32_t*>(W + wrow * (K / 2) + kc / 2));
            const float sc = e4m3_decode(__ldg(S + wrow * (K / 16) + kc / 16)) * 16384.0f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float2 p = e2m1x2((v >> (8 * b)) & 0xff);
                Ws[wq * 8 + 2 * b][wn] = p.x * sc;
                Ws[wq * 8 + 2 * b + 1][wn] = p.y * sc;
            }
        } else {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(W + wrow * K + kc));
            const float sc = exp2f((float) __ldg(S + wrow * (K / 32) + k0 / 32) - 119.0f);
            const uint32_t wd[2] = {v.x, v.y};
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const float2 p01 = e4m3x2((wd[b] & 0xffu) | ((wd[b] & 0xff00u) << 8));
                const float2 p23 = e4m3x2(((wd[b] >> 16) & 0xffu) | ((wd[b] & 0xff000000u) >> 8));
                Ws[wq * 8 + 4 * b][wn] = p01.x * sc;
                Ws[wq * 8 + 4 * b + 1][wn] = p01.y * sc;
                Ws[wq * 8 + 4 * b + 2][wn] = p23.x * sc;
                Ws[wq * 8 + 4 * b + 3][wn] = p23.y * sc;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < FKK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Ws[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    const float f = NV ? factors[e] : 1.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + ty * 4 + i;
        if (r < rend)
            *reinterpret_cast<float4*>(C + (size_t) r * N + n0 + tx * 4) =
                make_float4(acc[i][0] * f, acc[i][1] * f, acc[i][2] * f, acc[i][3] * f);
    }
}

// --- tensor-core variant (--tc 1): FP16 inputs, FP32 accumulation.  NVFP4 expert weights are exact in fp16
// (e2m1 * e4m3 <= 2688, >= 2^-10, <= 6 significant bits; the per-expert factor stays an fp32 multiplier); the
// activations are rounded to fp16 (2^-11 relative).  Residual stream, norms, router, gates, attention stay fp32.
// MXFP8 (layers 20-23): e4m3 * 2^(s-127) with s in [113, 118] in this checkpoint (measured, mx_range.py): multiples of
// 2^-23 below 1, exact in fp16; stored * 2^10 (clear of the subnormals), the 2^-10 back as the output factor.
constexpr int TM = 64, TN = 128, TK = 32, TLD = TK + 8, TCLD = TN + 4;
template <bool NV>
__global__ void __launch_bounds__(256) kb_moe_gemm_tc(const half* __restrict__ A, int lda,
                                                      const uint8_t* const* __restrict__ tab,
                                                      const float* __restrict__ factors, const int* __restrict__ tile_e,
                                                      const int* __restrict__ tile_r0, const int* __restrict__ ntiles,
                                                      const int* __restrict__ eoff, const int* __restrict__ ecnt,
                                                      float* __restrict__ C, int N, int K) {
    __shared__ __align__(32) unsigned char smraw[TM * TCLD * 4];   // As + Bs during the loop, C staging after
    half* As = reinterpret_cast<half*>(smraw);
    half* Bs = As + TM * TLD;
    float* Cs = reinterpret_cast<float*>(smraw);
    if ((int) blockIdx.x >= *ntiles) return;
    const int e = tile_e[blockIdx.x], r0 = tile_r0[blockIdx.x], rend = eoff[e] + ecnt[e], n0 = blockIdx.y * TN;
    const uint8_t* W = tab[e];
    if (!W) return;
    const uint8_t* S = W + (size_t) N * (NV ? K / 2 : K);
    const int tid = threadIdx.x, warp = tid / 32, wm = warp / 4, wn = warp % 4;
    // A: fp16 rows in expert order (gathered / converted once per layer, not once per N block): 16 B per thread
    const int arow = tid / 4, akq = tid % 4;
    const half* asrc = r0 + arow < rend ? A + (size_t) (r0 + arow) * lda + akq * 8 : nullptr;
    int wn_[2], wq_[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int idx = tid + 256 * u;
        wn_[u] = idx / 4;
        wq_[u] = idx % 4;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(c[i][j], 0.0f);
    // register prefetch: tile k0 + TK is loaded from global while tile k0 (in smem) goes through the MMAs.  The
    // conversions and the MMA order are unchanged (bit-identical to the single-buffered loop).
    uint4 av;
    uint2 wv[2];
    uint8_t sv[2];
    auto gload = [&](int k0) {
        av = asrc ? *reinterpret_cast<const uint4*>(asrc + k0) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const size_t wrow = (size_t) n0 + wn_[u];
            const int kc = k0 + wq_[u] * 8;
            if (NV) {
                wv[u].x = __ldg(reinterpret_cast<const uint32_t*>(W + wrow * (K / 2) + kc / 2));
                sv[u] = __ldg(S + wrow * (K / 16) + kc / 16);
            } else {
                wv[u] = __ldg(reinterpret_cast<const uint2*>(W + wrow * K + kc));
                sv[u] = __ldg(S + wrow * (K / 32) + k0 / 32);
            }
        }
    };
    gload(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        *reinterpret_cast<uint4*>(As + arow * TLD + akq * 8) = av;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            half2 h[4];
            if (NV) {
                const uint32_t v = wv[u].x;
                const float sc = e4m3_decode(sv[u]) * 16384.0f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float2 p = e2m1x2((v >> (8 * b)) & 0xff);
                    h[b] = __floats2half2_rn(p.x * sc, p.y * sc);
                }
            } else {
                const float sc = exp2f((float) sv[u] - 109.0f);   // 2^(s-127) 2^8 2^10
                const uint32_t wd[2] = {wv[u].x, wv[u].y};
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const float2 p01 = e4m3x2((wd[b] & 0xffu) | ((wd[b] & 0xff00u) << 8));
                    const float2 p23 = e4m3x2(((wd[b] >> 16) & 0xffu) | ((wd[b] & 0xff000000u) >> 8));
                    h[2 * b] = __floats2half2_rn(p01.x * sc, p01.y * sc);
                    h[2 * b + 1] = __floats2half2_rn(p23.x * sc, p23.y * sc);
                }
            }
            *reinterpret_cast<uint4*>(Bs + wn_[u] * TLD + wq_[u] * 8) = *reinterpret_cast<uint4*>(h);
        }
        __syncthreads();
        if (k0 + TK < K) gload(k0 + TK);
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * TLD + kk, TLD);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bs + (wn * 32 + j * 16) * TLD + kk, TLD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * TCLD + wn * 32 + j * 16, c[i][j], TCLD, wmma::mem_row_major);
    __syncthreads();
    const float f = NV ? factors[e] : 1.0f / 1024.0f;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int idx = tid + 256 * u, row = idx / 32, c4 = idx % 32, r = r0 + row;
        if (r >= rend) continue;
        const float4 v = *reinterpret_cast<const float4*>(Cs + row * TCLD + c4 * 4);
        *reinterpret_cast<float4*>(C + (size_t) r * N + n0 + c4 * 4) = make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
    }
}

// Grouped expert products for GGUF experts (prompt): as kb_moe_gemm / kb_moe_gemm_tc, the weights decoded by gg_ld8
// (no per-expert factor), output column n of w13 read from row gg_perm(n) (PERM)
template <int T, bool PERM>
__global__ void __launch_bounds__(256) kb_moe_gemm_gg(const float* __restrict__ A, int lda, const int* __restrict__ rows,
                                                      const uint8_t* const* __restrict__ tab, size_t rb,
                                                      const int* __restrict__ tile_e, const int* __restrict__ tile_r0,
                                                      const int* __restrict__ ntiles, const int* __restrict__ eoff,
                                                      const int* __restrict__ ecnt, float* __restrict__ C, int N, int K) {
    __shared__ __align__(16) float As[FKK][FM + 4], Ws[FKK][FN + 4];
    __shared__ uint32_t grid[512];
    __shared__ int8_t kv[16];
    if ((int) blockIdx.x >= *ntiles) return;
    const int e = tile_e[blockIdx.x], r0 = tile_r0[blockIdx.x], rend = eoff[e] + ecnt[e], n0 = blockIdx.y * FN;
    const uint8_t* W = tab[e];
    if (!W) return;   // not in this batch (offload): another launch has it
    gg_tables(grid, kv);
    const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    const float* asrc[2];
    int arow_l[2], akq[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int idx = tid + 256 * u, row = idx / 8, r = r0 + row;
        arow_l[u] = row;
        akq[u] = idx % 8;
        asrc[u] = r < rend ? A + (size_t) (rows ? rows[r] : r) * lda + akq[u] * 4 : nullptr;
    }
    const int wn = tid / 4, wq = tid % 4, nrow = n0 + wn;
    const uint8_t* wp = W + (size_t) (PERM ? gg_perm(nrow, N) : nrow) * rb;
    float acc[4][4] = {};
    __syncthreads();   // the tables
    for (int k0 = 0; k0 < K; k0 += FKK) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const float4 v = asrc[u] ? *reinterpret_cast<const float4*>(asrc[u] + k0) : make_float4(0, 0, 0, 0);
            const int kk = akq[u] * 4, row = arow_l[u];
            As[kk][row] = v.x; As[kk + 1][row] = v.y; As[kk + 2][row] = v.z; As[kk + 3][row] = v.w;
        }
        float w[8];
        gg_ld8<T>(wp, k0 + wq * 8, w, grid, kv);
#pragma unroll
        for (int j = 0; j < 8; ++j) Ws[wq * 8 + j][wn] = w[j];
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < FKK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Ws[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + ty * 4 + i;
        if (r < rend)
            *reinterpret_cast<float4*>(C + (size_t) r * N + n0 + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

template <int T, bool PERM>
__global__ void __launch_bounds__(256) kb_moe_gemm_tc_gg(const half* __restrict__ A, int lda,
                                                         const uint8_t* const* __restrict__ tab, size_t rb,
                                                         const int* __restrict__ tile_e, const int* __restrict__ tile_r0,
                                                         const int* __restrict__ ntiles, const int* __restrict__ eoff,
                                                         const int* __restrict__ ecnt, float* __restrict__ C, int N, int K) {
    __shared__ __align__(32) unsigned char smraw[TM * TCLD * 4];
    __shared__ uint32_t grid[512];
    __shared__ int8_t kv[16];
    half* As = reinterpret_cast<half*>(smraw);
    half* Bs = As + TM * TLD;
    float* Cs = reinterpret_cast<float*>(smraw);
    if ((int) blockIdx.x >= *ntiles) return;
    const int e = tile_e[blockIdx.x], r0 = tile_r0[blockIdx.x], rend = eoff[e] + ecnt[e], n0 = blockIdx.y * TN;
    const uint8_t* W = tab[e];
    if (!W) return;   // not in this batch (offload): another launch has it
    gg_tables(grid, kv);
    const int tid = threadIdx.x, warp = tid / 32, wm = warp / 4, wn = warp % 4;
    const int arow = tid / 4, akq = tid % 4;
    const half* asrc = r0 + arow < rend ? A + (size_t) (r0 + arow) * lda + akq * 8 : nullptr;
    int wn_[2], wq_[2];
    const uint8_t* wp[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int idx = tid + 256 * u;
        wn_[u] = idx / 4;
        wq_[u] = idx % 4;
        const int nrow = n0 + wn_[u];
        wp[u] = W + (size_t) (PERM ? gg_perm(nrow, N) : nrow) * rb;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(c[i][j], 0.0f);
    __syncthreads();   // the tables
    uint4 av;
    float wv[2][8];
    auto gload = [&](int k0) {
        av = asrc ? *reinterpret_cast<const uint4*>(asrc + k0) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 2; ++u) gg_ld8<T>(wp[u], k0 + wq_[u] * 8, wv[u], grid, kv);
    };
    gload(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        *reinterpret_cast<uint4*>(As + arow * TLD + akq * 8) = av;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            half2 h[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) h[b] = __floats2half2_rn(wv[u][2 * b], wv[u][2 * b + 1]);
            *reinterpret_cast<uint4*>(Bs + wn_[u] * TLD + wq_[u] * 8) = *reinterpret_cast<uint4*>(h);
        }
        __syncthreads();
        if (k0 + TK < K) gload(k0 + TK);
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * TLD + kk, TLD);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bs + (wn * 32 + j * 16) * TLD + kk, TLD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * TCLD + wn * 32 + j * 16, c[i][j], TCLD, wmma::mem_row_major);
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int idx = tid + 256 * u, row = idx / 32, c4 = idx % 32, r = r0 + row;
        if (r >= rend) continue;
        *reinterpret_cast<float4*>(C + (size_t) r * N + n0 + c4 * 4) = *reinterpret_cast<const float4*>(Cs + row * TCLD + c4 * 4);
    }
}

// GGUF experts for a chunk of up to 8 tokens: per expert of the plan (one tile each: <= 8 rows), every output row's
// weights decoded once for all its rows (the tile GEMMs decode a whole expert per 64-row tile).  A rows gathered
// through rows (w13) or direct (w2); grid (N / 8, tiles), a warp per output row
template <int T, bool PERM>
__global__ void __launch_bounds__(256) kb_moe_gemv_gg(const float* __restrict__ A, int lda, const int* __restrict__ rows,
                                                      const uint8_t* const* __restrict__ tab, size_t rb,
                                                      const int* __restrict__ tile_e, const int* __restrict__ tile_r0,
                                                      const int* __restrict__ ntiles, const int* __restrict__ eoff,
                                                      const int* __restrict__ ecnt, float* __restrict__ C, int N, int K) {
    __shared__ __align__(16) float xs[8][MKC];
    __shared__ uint32_t grid[512];
    __shared__ int8_t kv[16];
    if ((int) blockIdx.y >= *ntiles) return;
    const int e = tile_e[blockIdx.y], r0 = tile_r0[blockIdx.y], cnt = min(8, eoff[e] + ecnt[e] - r0);
    const uint8_t* W = tab[e];
    if (!W || cnt <= 0) return;
    gg_tables(grid, kv);
    constexpr int RR = 4;   // rows per warp: 32 per block share the staged inputs and tables
    const int o0 = (blockIdx.x * 8 + threadIdx.x / 32) * RR, lane = threadIdx.x & 31;
    const uint8_t* rp[RR];
#pragma unroll
    for (int q = 0; q < RR; ++q) {
        const int o = min(o0 + q, N - 1);
        rp[q] = W + (size_t) (PERM ? gg_perm(o, N) : o) * rb;
    }
    float acc[RR][8] = {};
    for (int k0 = 0; k0 < K; k0 += MKC) {
        const int kc = min(MKC, K - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < cnt * kc; i += blockDim.x) {
            const int r = i / kc;
            xs[r][i % kc] = A[(size_t) (rows ? rows[r0 + r] : r0 + r) * lda + k0 + i % kc];
        }
        __syncthreads();
        for (int kk = lane * 8; kk < kc; kk += 256) {
#pragma unroll
            for (int q = 0; q < RR; ++q) {
                float w[8];
                gg_ld8<T>(rp[q], k0 + kk, w, grid, kv);
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (r >= cnt) break;
                    const float4 a = *reinterpret_cast<const float4*>(&xs[r][kk]), b = *reinterpret_cast<const float4*>(&xs[r][kk + 4]);
                    acc[q][r] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < RR; ++q)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= cnt) break;
            const float t = warp_sum(acc[q][r]);
            if (!lane && o0 + q < N) C[(size_t) (r0 + r) * N + o0 + q] = t;
        }
}
// big prompt chunks (INK2_DQ_MIN tokens, --tc): an expert's weights decoded once to fp16 [N][K] (row gg_perm(n) for
// PERM), then a cuBLAS tensor-core GEMM per expert over its rows (the tile GEMMs decode the weights per 64-row tile)
template <int T, bool PERM>
__global__ void kb_dequant_gg(const uint8_t* const* __restrict__ tab, int e, size_t rb, half* __restrict__ out, int N, int K) {
    __shared__ uint32_t grid[512];
    __shared__ int8_t kv[16];
    gg_tables(grid, kv);
    __syncthreads();
    const uint8_t* W = tab[e];
    const size_t n8 = (size_t) N * (K / 8);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (size_t) gridDim.x * blockDim.x) {
        const int n = (int) (i / (K / 8)), kc = (int) (i % (K / 8)) * 8;
        float w[8];
        gg_ld8<T>(W + (size_t) (PERM ? gg_perm(n, N) : n) * rb, kc, w, grid, kv);
        half2 h[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) h[b] = __floats2half2_rn(w[2 * b], w[2 * b + 1]);
        *reinterpret_cast<uint4*>(out + (size_t) n * K + kc) = *reinterpret_cast<uint4*>(h);
    }
}
// fp32 [T][K] (row stride ld) -> fp16 [T][K]; bf16 -> fp16 (dense weights: exact for |w| >= 2^-14, the rare smaller
// ones lose low bits as fp16 subnormals)
__global__ void kb_f32_to_f16(const float* __restrict__ x, int ld, int K, half* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / K, k = i % K;
    y[i] = __float2half_rn(x[t * ld + k]);
}
__global__ void kb_bf16_to_f16(const uint16_t* __restrict__ src, half* __restrict__ dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(__uint_as_float((uint32_t) src[i] << 16));
}

// Tensor-core attention (--tc 1): block = 64 queries x one q head, 4 warps (16 query rows each), key tiles of 32.
// Q (fp16, prepared in qh [T][QD]) stays in matrix_a fragments; K / V tiles converted fp32 -> fp16 into smem;
// S = Q K^T (fp32 accumulation) goes through smem for scale + bias + mask + online softmax (fp32), P in fp16;
// O stays in accumulator fragments, rescaled per row with a row map of the fragment elements found at start
// (store_matrix_sync of known values: the element -> row mapping is not specified by the API, so it is measured).
constexpr int AQ = 64, AK = 32, ALD = HD + 8, SLD = AK + 4, PLD = AK + 8, AW = 4 * NREP;
static_assert(16 * PLD * 2 <= 16 * SLD * 4, "P aliases S");
__global__ void __launch_bounds__(128 * NREP, 1) kb_attn_tc(const half* __restrict__ qh, const float* __restrict__ P, int sp,
                                                  const void* __restrict__ K, const void* __restrict__ V,
                                                  const __half* __restrict__ ksc, const __half* __restrict__ vsc,
                                                  const float* __restrict__ rel, int cap, int window, int pos0, int T,
                                                  float* __restrict__ out) {
    // GQA group per block: the NREP q heads of kv head g share each K / V tile (loaded and converted once);
    // warp w: head g * NREP + w / 4, query rows t0 + (w % 4) * 16.  Per warp the computation is unchanged.
    // P (fp16) overwrites the warp's own S rows once read; the row map lives in the K / V tile before the loop.
    __shared__ __align__(32) half KVs[2 * AK * ALD];
    __shared__ __align__(32) float Ss[AW][16 * SLD];
    __shared__ float ms[AW][16], ls[AW][16], fs[AW][16];
    static_assert(AW * 256 * 4 <= 2 * AK * ALD * 2, "row map fits in the K / V tile");
    half* Ks = KVs;
    half* Vs = KVs + AK * ALD;
    const int g = blockIdx.y, t0 = blockIdx.x * AQ, tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int h = g * NREP + w / 4;
    const int tw = t0 + (w % 4) * 16;   // this warp's first query row
    half* Pw = reinterpret_cast<half*>(Ss[w]);
    int* rmw = reinterpret_cast<int*>(KVs) + w * 256;
    using FA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FBc = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
    using FBr = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    // row map of accumulator elements
    FC probe;
    const int ne = probe.num_elements;
    for (int i = 0; i < ne; ++i) probe.x[i] = (float) (lane * ne + i);
    wmma::store_matrix_sync(Ss[w], probe, SLD, wmma::mem_row_major);
    __syncwarp();
    for (int cell = lane; cell < 256; cell += 32) {
        const int v = (int) Ss[w][(cell / 16) * SLD + cell % 16];
        rmw[v] = cell / 16;
    }
    __syncwarp();
    int rowof[8];
    for (int i = 0; i < ne && i < 8; ++i) rowof[i] = rmw[lane * ne + i];
    FA qa[HD / 16];
#pragma unroll
    for (int kd = 0; kd < HD / 16; ++kd) wmma::load_matrix_sync(qa[kd], qh + (size_t) tw * QD + h * HD + kd * 16, QD);
    FC o[HD / 16];
#pragma unroll
    for (int n = 0; n < HD / 16; ++n) wmma::fill_fragment(o[n], 0.0f);
    if (lane < 16) {
        const int r = tw + lane;
        ms[w][lane] = r < T ? P[(size_t) r * sp + OFF_SINK + h] : 0.0f;
        ls[w][lane] = 1.0f;
    }
    const float scale = rsqrtf((float) HD);
    const int nq = min(AQ, T - t0);
    const int pa = pos0 + t0, pb = pos0 + t0 + nq - 1;
    const int lo = window > 0 ? max(0, pa - window + 1) : 0;
    for (int k0 = lo; k0 <= pb; k0 += AK) {
        __syncthreads();
        for (int i = tid; i < AK * HD / 4; i += 128 * NREP) {   // float4 units
            const int r = i / (HD / 4), d = (i % (HD / 4)) * 4, j = k0 + r;
            float4 kv = make_float4(0, 0, 0, 0), vv = kv;
            if (j <= pb) {
                kv = kv_ld4<false>(K, ksc, (size_t) (j % cap), g * HD + d);
                vv = kv_ld4<true>(V, vsc, (size_t) (j % cap), g * HD + d);
            }
            half2 a[2] = {__floats2half2_rn(kv.x, kv.y), __floats2half2_rn(kv.z, kv.w)};
            half2 b[2] = {__floats2half2_rn(vv.x, vv.y), __floats2half2_rn(vv.z, vv.w)};
            *reinterpret_cast<uint2*>(Ks + r * ALD + d) = *reinterpret_cast<uint2*>(a);
            *reinterpret_cast<uint2*>(Vs + r * ALD + d) = *reinterpret_cast<uint2*>(b);
        }
        __syncthreads();
        if (tw >= T) continue;   // whole warp past the chunk (still joins the block barriers)
        FC s[AK / 16];
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) {
            wmma::fill_fragment(s[n], 0.0f);
#pragma unroll
            for (int kd = 0; kd < HD / 16; ++kd) {
                FBc kb;
                wmma::load_matrix_sync(kb, Ks + n * 16 * ALD + kd * 16, ALD);
                wmma::mma_sync(s[n], qa[kd], kb, s[n]);
            }
            wmma::store_matrix_sync(Ss[w] + n * 16, s[n], SLD, wmma::mem_row_major);
        }
        __syncwarp();
        {
            const int r = lane / 2, c0 = (lane % 2) * 16, t = tw + r, p = pos0 + t;
            float sv[16], mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int j = k0 + c0 + c, dist = p - j;
                const bool ok = t < T && j <= p && (window == 0 || dist < window);
                sv[c] = ok ? Ss[w][r * SLD + c0 + c] * scale +
                                 (dist < EXTENT ? rel[((size_t) t * NQ + h) * EXTENT + dist] : 0.0f)
                           : -INFINITY;
                mx = fmaxf(mx, sv[c]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            const float mo = ms[w][r], mn = fmaxf(mo, mx), f = expf(mo - mn);
            float sum = 0.0f;
            __syncwarp();   // every S value of the warp read before P overwrites them
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float e = expf(sv[c] - mn);
                sum += e;
                Pw[r * PLD + c0 + c] = __float2half_rn(e);
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            __syncwarp();
            if (!(lane % 2)) { ms[w][r] = mn; ls[w][r] = ls[w][r] * f + sum; fs[w][r] = f; }
        }
        __syncwarp();
        float fr[8];
        for (int i = 0; i < ne && i < 8; ++i) fr[i] = fs[w][rowof[i]];
#pragma unroll
        for (int n = 0; n < HD / 16; ++n)
            for (int i = 0; i < ne && i < 8; ++i) o[n].x[i] *= fr[i];
#pragma unroll
        for (int ks = 0; ks < AK / 16; ++ks) {
            FA pa_;
            wmma::load_matrix_sync(pa_, Pw + ks * 16, PLD);
#pragma unroll
            for (int n = 0; n < HD / 16; ++n) {
                FBr vb;
                wmma::load_matrix_sync(vb, Vs + ks * 16 * ALD + n * 16, ALD);
                wmma::mma_sync(o[n], pa_, vb, o[n]);
            }
        }
    }
    if (tw >= T) return;
    __syncwarp();
    float il[8];
    for (int i = 0; i < ne && i < 8; ++i) il[i] = 1.0f / ls[w][rowof[i]];
#pragma unroll
    for (int n = 0; n < HD / 16; ++n) {
        for (int i = 0; i < ne && i < 8; ++i) o[n].x[i] *= il[i];
        wmma::store_matrix_sync(out + (size_t) tw * QD + h * HD + n * 16, o[n], QD, wmma::mem_row_major);
    }
}

// SoftcappedSwiGLU on nr rows: gu [nr][2FF] -> act [nr][FF]
__global__ void kb_swiglu(const float* gu, float* act, int nr) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nr * FF) return;
    const size_t k = i / FF, j = i % FF;
    const float g = gu[k * 2 * FF + 2 * j], u = gu[k * 2 * FF + 2 * j + 1];
    act[i] = (tanhf(g / GATE_CAP) * GATE_CAP) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / UP_CAP) * UP_CAP);
}

// expert-ordered fp16 copy of the MoE input: ah[r][k] = fp16(x[rows[r]][k]) (the rounding the TC GEMM applied per tile)
__global__ void kb_gather_h(const float* __restrict__ x, const int* __restrict__ rows, half* __restrict__ ah) {
    const int r = blockIdx.x, src = rows[r];
    for (int k = threadIdx.x * 2; k < H; k += blockDim.x * 2)
        *reinterpret_cast<half2*>(ah + (size_t) r * H + k) =
            __floats2half2_rn(x[(size_t) src * H + k], x[(size_t) src * H + k + 1]);
}
// kb_swiglu with an fp16 output (the down GEMM's A, same values as converting the fp32 act per tile)
__global__ void kb_swiglu_h(const float* gu, half* act, int nr) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nr * FF) return;
    const size_t k = i / FF, j = i % FF;
    const float g = gu[k * 2 * FF + 2 * j], u = gu[k * 2 * FF + 2 * j + 1];
    act[i] = __float2half_rn((tanhf(g / GATE_CAP) * GATE_CAP) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / UP_CAP) * UP_CAP));
}

// out[t][i] = sum_k wts[t][k] ey[slot[t][k]][i] (k in routing order, as k_moe_tail); grid (H / 256, T)
__global__ void kb_combine(const float* ey, const int* slot, const float* wts, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    float a = 0.0f;
    for (int k = 0; k < TOPK; ++k) a += ey[(size_t) slot[t * TOPK + k] * H + i] * wts[t * TOPK + k];
    out[(size_t) t * H + i] = a;
}

// routing record (one-token path): rec[pos][k] = ids[k]
__global__ void k_rec_ids(const int* ids, int* rec, const int* d_pos) {
    rec[(size_t) (*d_pos) * TOPK + threadIdx.x] = ids[threadIdx.x];
}
// Expert cache (offload, --vram-reserve-mib).  Per pool: owner[slot] = l * NE + e or -1, stamp[slot] = last use (one
// clock for both pools); slot_of[l * NE + e] = slot or -1.  k_cache_plan (one block) stamps every id of the call
// first, then gives each miss the least recently used slot (ties: lowest index), so a call never evicts its own
// experts; it points the tables at the slot and queues the copy, which k_fetch does from mapped pinned host memory.
// n ids, duplicates allowed (n = TOPK in decode; T x TOPK for a multi-token verify).
// CBA: 32-bit clock, wraps after ~10M decoded tokens (LRU order then wrong for one pass, results still exact).
struct CacheArgs {
    int *slot_of, *owner;
    unsigned *stamp, *clock;
    int nslot;
    const uint8_t **tabA, **tabB;
    uint8_t* pool;
    size_t eb, partA;
    const uint8_t* host;   // device view of layer l's experts in pinned host memory (NE x eb)
    const uint8_t** fsrc;
    uint8_t** fdst;
    int* nfetch;
    unsigned long long* fbytes;
};
constexpr int FETCH_MAX = 64, FETCH_BLOCKS = 96;
__global__ void k_cache_plan(int l, const int* __restrict__ ids, int n, CacheArgs c) {
    __shared__ int miss[FETCH_MAX], nm;
    __shared__ unsigned clk;
    __shared__ unsigned long long wmin[32];
    if (threadIdx.x == 0) {
        unsigned t = *c.clock;
        int m = 0;
        for (int i = 0; i < n; ++i) {
            const int e = ids[i];
            bool dup = false;
            for (int j = 0; j < i; ++j) dup |= ids[j] == e;
            if (dup) continue;
            const int sl = c.slot_of[l * NE + e];
            if (sl >= 0) c.stamp[sl] = ++t;
            else miss[m++] = e;
        }
        nm = m;
        clk = t;
    }
    __syncthreads();
    for (int m = 0; m < nm; ++m) {
        unsigned long long best = ~0ull;
        for (int i = threadIdx.x; i < c.nslot; i += blockDim.x)
            best = min(best, ((unsigned long long) c.stamp[i] << 32) | (unsigned) i);
#pragma unroll
        for (int o = 16; o; o >>= 1) best = min(best, __shfl_xor_sync(0xffffffffu, best, o));
        if ((threadIdx.x & 31) == 0) wmin[threadIdx.x / 32] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < (int) (blockDim.x / 32); ++w) best = min(best, wmin[w]);
            const int v = (int) (best & 0xffffffffu), e = miss[m], key = l * NE + e, old = c.owner[v];
            if (old >= 0) { c.slot_of[old] = -1; c.tabA[old] = nullptr; c.tabB[old] = nullptr; }
            uint8_t* d = c.pool + (size_t) v * c.eb;
            c.owner[v] = key;
            c.slot_of[key] = v;
            c.stamp[v] = ++clk;
            c.tabA[key] = d;
            c.tabB[key] = d + c.partA;
            c.fsrc[m] = c.host + (size_t) e * c.eb;
            c.fdst[m] = d;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *c.clock = clk;
        *c.nfetch = nm;
        *c.fbytes += (unsigned long long) nm * c.eb;
    }
}
// queued copies, host -> slot: grid (FETCH_BLOCKS, max copies), 16 B per thread and step, 4 loads in flight
__global__ void k_fetch(const uint8_t* const* __restrict__ fsrc, uint8_t* const* __restrict__ fdst,
                        const int* __restrict__ nfetch, size_t bytes) {
    if ((int) blockIdx.y >= *nfetch) return;
    const uint4* src = reinterpret_cast<const uint4*>(fsrc[blockIdx.y]);
    uint4* dst = reinterpret_cast<uint4*>(fdst[blockIdx.y]);
    const size_t n = bytes / 16, st = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * st < n; i += 4 * st) {
        const uint4 a = src[i], b = src[i + st], cc = src[i + 2 * st], d = src[i + 3 * st];
        dst[i] = a; dst[i + st] = b; dst[i + 2 * st] = cc; dst[i + 3 * st] = d;
    }
    for (; i < n; i += st) dst[i] = src[i];
}

// --kv-grow: slots [from, to) leave a pool: their experts out of the tables (the next use fetches them again)
__global__ void k_pool_evict(int* owner, int* slot_of, const uint8_t** tabA, const uint8_t** tabB, unsigned* stamp,
                             int from, int to) {
    const int s = from + (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= to) return;
    const int o = owner[s];
    if (o >= 0) { slot_of[o] = -1; tabA[o] = nullptr; tabB[o] = nullptr; owner[s] = -1; }
    stamp[s] = 0;
}

// teacher-forced routing (verification): the recorded experts, weights from this pass' own scores; grid T, block 32
__global__ void kb_router_forced(const float* logits, const int* forced, int* ids, float* wts) {
    const int t = blockIdx.x, k = threadIdx.x;
    if (k) return;
    float sc[TOPK], sum = 0.0f;
    for (int j = 0; j < TOPK; ++j) {
        sc[j] = 1.0f / (1.0f + expf(-logits[(size_t) t * NE + forced[t * TOPK + j]]));
        sum += sc[j];
    }
    for (int j = 0; j < TOPK; ++j) { ids[t * TOPK + j] = forced[t * TOPK + j]; wts[t * TOPK + j] = sc[j] / (sum + 1e-20f) * ROUTE_SCALE; }
}

__global__ void kb_add(const float* a, const float* b, float* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// Checkpoint and weights

struct Ckpt {
    std::string path;
    StFile st;
    const StTensor& t(const std::string& name) const {
        auto it = st.tensors.find(name);
        if (it == st.tensors.end()) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
        return it->second;
    }
    std::vector<unsigned char> bytes(const std::string& name) const {
        std::vector<unsigned char> b;
        const std::string e = strata::image::st_read_tensor(path, st, t(name), b);
        if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); }
        return b;
    }
    vec f32(const std::string& name) const {
        const StTensor& d = t(name);
        auto b = bytes(name);
        vec out;
        if (d.dtype == "F32") {
            out.resize(b.size() / 4);
            std::memcpy(out.data(), b.data(), b.size());
        } else if (d.dtype == "BF16") {
            out.resize(b.size() / 2);
            for (size_t i = 0; i < out.size(); ++i) {
                uint32_t u = (uint32_t) (b[2 * i] | (b[2 * i + 1] << 8)) << 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else { std::fprintf(stderr, "%s: dtype %s\n", name.c_str(), d.dtype.c_str()); std::exit(1); }
        return out;
    }
};

size_t g_vram = 0;

void* dalloc(size_t bytes) {
    if (!bytes) return nullptr;   // an empty pool (GGUF: no MXFP8 slots)
    void* d = nullptr;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemset(d, 0, bytes));
    g_vram += bytes;
    return d;
}
float* dfloats(size_t n) { return (float*) dalloc(n * 4); }
void* upload(const void* p, size_t bytes) {
    void* d = dalloc(bytes);
    if (!d) return nullptr;
    CK(cudaMemcpy(d, p, bytes, cudaMemcpyHostToDevice));
    return d;
}
float* dvec(const vec& v) { return (float*) upload(v.data(), v.size() * 4); }
// ---------------------------------------------------------------------------------------------------------------------
// Growable device ranges (CUDA virtual memory): an address range reserved once, physical chunks of the allocation
// granularity mapped and unmapped at its end.  The addresses never move.  Driver entry points resolved at run time
// (no link to libcuda), as in src/core/vmm.cpp.
struct Vmm {
    bool ok = false;
    size_t gran = 0;
    int dev = 0;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};
Vmm g_vmm;
template <class F> bool vmm_resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || !p) { (void) cudaGetLastError(); return false; }
    f = (F) p;
    return true;
}
void vmm_init() {
    Vmm& a = g_vmm;
    decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
    CK(cudaGetDevice(&a.dev));
    if (!vmm_resolve("cuMemGetAllocationGranularity", granularity) || !vmm_resolve("cuMemAddressReserve", a.reserve) ||
        !vmm_resolve("cuMemCreate", a.create) || !vmm_resolve("cuMemRelease", a.release) || !vmm_resolve("cuMemMap", a.map) ||
        !vmm_resolve("cuMemUnmap", a.unmap) || !vmm_resolve("cuMemSetAccess", a.access)) {
        std::fprintf(stderr, "--kv-grow: the driver's virtual memory entry points are missing\n");
        std::exit(1);
    }
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    if (granularity(&a.gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || !a.gran) {
        std::fprintf(stderr, "--kv-grow: no allocation granularity\n");
        std::exit(1);
    }
    a.ok = true;
}
struct VRange {
    CUdeviceptr base = 0;
    size_t reserved = 0, mapped = 0;
    std::vector<CUmemGenericAllocationHandle> h;
    void* ptr() const { return (void*) base; }
    static size_t up(size_t b) { return (b + g_vmm.gran - 1) / g_vmm.gran * g_vmm.gran; }
    void init(size_t bytes) {
        reserved = up(std::max<size_t>(bytes, 1));
        if (g_vmm.reserve(&base, reserved, 0, 0, 0) != CUDA_SUCCESS) {
            std::fprintf(stderr, "cuMemAddressReserve of %zu bytes failed\n", reserved);
            std::exit(1);
        }
    }
    size_t need(size_t bytes) const { return std::min(up(bytes), reserved); }   // mapped bytes for `bytes`
    bool map_to(size_t bytes) {   // false: out of memory (what was mapped stays)
        const size_t want = need(bytes);
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = g_vmm.dev;
        CUmemAccessDesc acc{};
        acc.location = prop.location;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (mapped < want) {
            CUmemGenericAllocationHandle hd = 0;
            if (g_vmm.create(&hd, g_vmm.gran, &prop, 0) != CUDA_SUCCESS) return false;
            if (g_vmm.map(base + mapped, g_vmm.gran, 0, hd, 0) != CUDA_SUCCESS ||
                g_vmm.access(base + mapped, g_vmm.gran, &acc, 1) != CUDA_SUCCESS) {
                std::fprintf(stderr, "cuMemMap failed\n");
                std::exit(1);
            }
            h.push_back(hd);
            mapped += g_vmm.gran;
            g_vram += g_vmm.gran;
        }
        return true;
    }
    void unmap_to(size_t bytes) {
        const size_t want = need(bytes);
        while (mapped > want) {
            mapped -= g_vmm.gran;
            g_vmm.unmap(base + mapped, g_vmm.gran);
            g_vmm.release(h.back());
            h.pop_back();
            g_vram -= g_vmm.gran;
        }
    }
};

struct DMat {
    const uint16_t* d = nullptr;   // BF16 [n][k]
    int n = 0, k = 0;
    const int8_t* q = nullptr;     // or Q8_0 (GGUF): codes [n][k], fp16 scales [n][k / 32]
    const __half* qs = nullptr;
};

// Experts live in fixed-size slots of a device pool (one pool per format), each slot one expert as
// [w13 | w13 scales | w2 | w2 scales] (part A = w13 + scales, part B = w2 + scales).  The kernels address an expert
// through tA / tB (device tables, NE part pointers per layer, slices of E.tabA / E.tabB).
struct Experts {
    bool nvfp4 = true;
    const uint8_t* const* tA = nullptr;
    const uint8_t* const* tB = nullptr;
    const float *f13 = nullptr, *f2 = nullptr;
    int gg13 = 0, gg2 = 0;       // GGUF: ggml types of gate_up (part A) and down (part B), 0 = the checkpoint formats
    size_t rb13 = 0, rb2 = 0;    //   bytes per row
};
constexpr size_t NV_A = (size_t) 2 * FF * H / 2 + (size_t) 2 * FF * H / 16, NV_B = (size_t) H * FF / 2 + (size_t) H * FF / 16;
constexpr size_t MX_A = (size_t) 2 * FF * H + (size_t) 2 * FF * H / 32, MX_B = (size_t) H * FF + (size_t) H * FF / 32;
constexpr size_t NV_BYTES = NV_A + NV_B, MX_BYTES = MX_A + MX_B;   // 4478976 / 8211456
static_assert(NV_A % 256 == 0 && NV_BYTES % 256 == 0 && MX_A % 256 == 0 && MX_BYTES % 256 == 0, "slot alignment");

struct Gate { DMat down, fg; };   // fg: [forget; gain] stacked [2H, GATE_RANK]

struct Layer {
    bool full = false, loop = false;
    // proj: [wq 3072; gate_attn 3072; wr 384; sink 24; wk 1024; (k_wproj 32); wv 1024; (v_wproj 32)] stacked, K = H:
    // one launch when the query, key and value inputs are the same vector, three otherwise
    DMat proj, wo, router;
    int off_k = 0, off_v = 0, n_k = 0, n_v = 0;
    const float *q_scale, *rel_scale, *rel_proj, *k_conv = nullptr, *v_conv = nullptr, *k_static = nullptr,
                *v_static = nullptr, *attn_conv, *mlp_conv, *attn_norm = nullptr, *q_norm = nullptr,
                *k_norm = nullptr, *v_norm = nullptr, *mlp_norm, *attn_query = nullptr, *mlp_query, *router_bias;
    Experts ex;
    Gate gates[4];
};

struct Slot {
    void *K, *V;    // [cap][KVD] fp32 / fp16 / int8 codes with ksc / vsc (--kv, see kv_st)
    __half *ksc = nullptr, *vsc = nullptr;   // [cap][KVD / KVG]
    int cap = 0;
    bool full = false;
    float *khist, *vhist, *ahist, *mhist;   // [3][C]
};

struct Engine {
    const Ckpt* ck = nullptr;
    int ctx = 8192, nchunk_full = 0, nchunk_slide = WINDOW / CHUNK;
    std::vector<uint16_t> embed;   // host BF16
    std::vector<uint8_t> embed_q8; // or (GGUF) host Q8_0 rows of H / 32 blocks
    const float *embed_norm, *norm, *final_query, *msg_queries;
    DMat unembed;
    Layer L[NL];
    Slot S[NCALLS];
    int* d_pos;
    // activations
    float *h, *out, *Bbuf, *preB, *pre_h, *prev_stack, *msgs;
    float *a, *t[4], *xq, *xk, *xv, *q, *k, *v, *kc, *vc, *dw, *rs, *rel, *sink, *gate, *att, *attn_out, *att2, *r;
    float *m, *mn, *cat, *gdown, *gf, *gg, *rl, *wts, *gu, *act, *ey, *moe, *y2, *logits;
    float *pm, *pl, *po, *emb_in, *pout, *gfg;
    int* ids;
    bool graph = true;
    cudaGraphExec_t exec = nullptr;
    int* rec = nullptr;   // routing record / forcing [NCALLS][ctx][TOPK] (verification only)
    int rec_mode = 0;     // 1 record (one-token path), 2 force (prefill)
    bool grow = false;    // --kv-grow 1: the full-attention slots' K/V mapped as the context grows (VRange), the expert
    int grow_step = 8192; //   pools (offload) give memory up and take it back (--kv-grow-step cells)
    int kv_cells = 0;     //   cells mapped now
    int kvm = 1;          // --kv f32 (0) | f16 (1, default) | i8 (2: int8 cells + fp16 scale per 64, Strata's kv_q8
                          // layout) | i8r (3: i8 + Walsh-Hadamard rotation)
    // experts: every expert in xhost layout order (NVFP4 layers 0-19 then MXFP8 layers 20-23, NE per layer, one slot
    // each); resident = the device pool holds them all, in that order
    uint8_t *pool_nv = nullptr, *pool_mx = nullptr;
    const uint8_t **tabA = nullptr, **tabB = nullptr;   // device [NL * NE]: part A / part B of expert (l, e)
    std::vector<const uint8_t*> htabA, htabB;            // host mirrors
    // offload (--vram-reserve-mib N: leave N MiB of the card free, the expert pools take what remains): every expert
    // in pinned host memory (xhost), the pools hold nslot_nv / nslot_mx of them, LRU (k_cache_plan)
    int reserve_mib = -1;   // < 0: everything resident
    uint8_t* xhost = nullptr;
    const uint8_t* xhost_d = nullptr;   // its device view
    int nslot_nv = 0, nslot_mx = 0;
    VRange vpool_nv, vpool_mx;   // --kv-grow: the pools as growable ranges (owner / stamp sized for every expert)
    int *slot_of = nullptr, *owner_nv = nullptr, *owner_mx = nullptr, *nfetch = nullptr;
    unsigned *stamp_nv = nullptr, *stamp_mx = nullptr, *clock = nullptr;
    const uint8_t** fsrc = nullptr;
    uint8_t** fdst = nullptr;
    unsigned long long* fbytes = nullptr;
};
// GGUF (--model FILE.gguf): one format for every layer, its slots in pool_nv: [gate_up rows | down rows]
bool g_is_gguf = false;
size_t g_gg_a = 0, g_gg_b = 0;
bool ex_nv(int l) { return g_is_gguf || l < MXFP8_FIRST; }   // the layer's experts live in pool_nv
size_t ex_bytes(int l) { return g_is_gguf ? g_gg_a + g_gg_b : l < MXFP8_FIRST ? NV_BYTES : MX_BYTES; }
size_t ex_partA(int l) { return g_is_gguf ? g_gg_a : l < MXFP8_FIRST ? NV_A : MX_A; }
size_t ex_layer_off(int l) {   // offset of layer l's first expert in the xhost / resident layout
    if (g_is_gguf) return (size_t) l * NE * (g_gg_a + g_gg_b);
    return l < MXFP8_FIRST ? (size_t) l * NE * NV_BYTES
                           : (size_t) MXFP8_FIRST * NE * NV_BYTES + (size_t) (l - MXFP8_FIRST) * NE * MX_BYTES;
}
int g_slot = 0;   // the layer application being built (host side, baked into the launches)
int g_rpw = 4;    // decode GEMV rows per warp, upper bound (INK2_RPW: 1, 2, 4, 8)
int g_sel_rpw = 2;   // expert (selected) GEMV rows per warp (INK2_SEL_RPW: 1, 2, 4, 8)
int g_gemv_min_blocks = 164;   // bf16 GEMV: R halved until the grid has this many blocks (2 per SM; INK2_GEMV_MINB)

Engine E;

DMat dmat(const std::string& name) {
    const StTensor& t = E.ck->t(name);
    DMat m;
    m.n = (int) t.shape[0];
    m.k = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) m.k *= (int) t.shape[i];
    auto b = E.ck->bytes(name);
    m.d = (const uint16_t*) upload(b.data(), b.size());
    return m;
}
const float* fvec(const std::string& name) { return dvec(E.ck->f32(name)); }

DMat dmat_stack(const std::vector<std::string>& names) {   // BF16 matrices with the same K, rows concatenated
    std::vector<unsigned char> all;
    DMat m;
    for (const auto& n : names) {
        const StTensor& t = E.ck->t(n);
        int k = 1;
        for (size_t i = 1; i < t.shape.size(); ++i) k *= (int) t.shape[i];
        if (m.k && k != m.k) { std::fprintf(stderr, "%s: K %d != %d\n", n.c_str(), k, m.k); std::exit(1); }
        m.k = k;
        m.n += (int) t.shape[0];
        auto b = E.ck->bytes(n);
        all.insert(all.end(), b.begin(), b.end());
    }
    m.d = (const uint16_t*) upload(all.data(), all.size());
    return m;
}

void load() {
    const Ckpt& ck = *E.ck;
    const std::string p = "model.llm.";
    E.embed_norm = fvec(p + "embed_norm.weight");
    E.norm = fvec(p + "norm.weight");
    E.final_query = fvec(p + "attn_res.output_query");
    {
        auto b = ck.bytes(p + "embed.weight");
        E.embed.resize(b.size() / 2);
        std::memcpy(E.embed.data(), b.data(), b.size());
    }
    E.unembed = dmat(p + "unembed.weight");
    {
        const vec q = ck.f32(p + "middle_cycle_cross_pass_queries"), kv = ck.f32(p + "middle_cycle_cross_pass_kv_queries");
        vec mq((size_t) NLOOP * 4 * H);
        for (int i = 0; i < NLOOP; ++i) {
            const float* src[4] = {&q[(size_t) (2 * i) * H], &kv[(size_t) (2 * i) * H], &kv[(size_t) (2 * i + 1) * H],
                                   &q[(size_t) (2 * i + 1) * H]};
            for (int j = 0; j < 4; ++j) std::memcpy(&mq[((size_t) i * 4 + j) * H], src[j], H * 4);
        }
        E.msg_queries = dvec(mq);
    }
    if (E.reserve_mib < 0) {
        E.pool_nv = (uint8_t*) dalloc((size_t) MXFP8_FIRST * NE * NV_BYTES);
        E.pool_mx = (uint8_t*) dalloc((size_t) (NL - MXFP8_FIRST) * NE * MX_BYTES);
    } else {
        CK(cudaHostAlloc((void**) &E.xhost, ex_layer_off(NL), cudaHostAllocMapped | cudaHostAllocPortable));
        void* dv = nullptr;
        CK(cudaHostGetDevicePointer(&dv, E.xhost, 0));
        E.xhost_d = (const uint8_t*) dv;
    }
    E.tabA = (const uint8_t**) dalloc((size_t) NL * NE * sizeof(void*));
    E.tabB = (const uint8_t**) dalloc((size_t) NL * NE * sizeof(void*));
    E.htabA.assign((size_t) NL * NE, nullptr);
    E.htabB.assign((size_t) NL * NE, nullptr);
    for (int l = 0; l < NL; ++l) {
        Layer& Y = E.L[l];
        const std::string q = p + "layers." + std::to_string(l) + ".";
        Y.full = full_layer(l);
        Y.loop = l >= LOOP_BEGIN && l < LOOP_END;
        {
            std::vector<std::string> parts = {q + "attn.wq_du.weight", q + "attn.gate_attn.weight", q + "attn.wr_du.weight",
                                              q + "attn.attn_sink_proj.weight", q + "attn.wk_dv.weight"};
            if (Y.full) parts.push_back(q + "attn.k_sconv.weight_proj");
            parts.push_back(q + "attn.wv_dv.weight");
            if (Y.full) parts.push_back(q + "attn.v_sconv.weight_proj");
            Y.proj = dmat_stack(parts);
            Y.off_k = OFF_K;
            Y.n_k = KVD + (Y.full ? NKV * KC : 0);
            Y.off_v = Y.off_k + Y.n_k;
            Y.n_v = Y.n_k;
            if (Y.proj.n != Y.off_v + Y.n_v) { std::fprintf(stderr, "layer %d: stacked rows %d\n", l, Y.proj.n); std::exit(1); }
        }
        Y.wo = dmat(q + "attn.wo_ud.weight");
        Y.router = dmat(q + "mlp.gate.weight");
        Y.router_bias = fvec(q + "mlp.gate.bias");
        Y.q_scale = fvec(q + "attn.q_l2_clip.scale");
        Y.rel_scale = fvec(q + "attn.rel_l2_clip.scale");
        Y.rel_proj = fvec(q + "attn.rel_logits_proj.proj");
        Y.attn_conv = fvec(q + "attn_sconv.weight");
        Y.mlp_conv = fvec(q + "mlp_sconv.weight");
        Y.mlp_norm = fvec(q + "mlp_norm.weight");
        if (Y.full) {
            Y.k_static = fvec(q + "attn.k_sconv.static_w");
            Y.v_static = fvec(q + "attn.v_sconv.static_w");
            Y.q_norm = fvec(q + "q_input_norm.weight");
            Y.k_norm = fvec(q + "k_input_norm.weight");
            Y.v_norm = fvec(q + "v_input_norm.weight");
        } else {
            Y.k_conv = fvec(q + "attn.k_sconv.weight");
            Y.v_conv = fvec(q + "attn.v_sconv.weight");
            Y.attn_norm = fvec(q + "attn_norm.weight");
        }
        if (l > 0) Y.attn_query = fvec(p + "attn_res.attn_queries." + std::to_string(l - 1));
        Y.mlp_query = fvec(p + "attn_res.mlp_queries." + std::to_string(l));
        if (Y.loop) {
            const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int g = 0; g < 4; ++g) {
                const std::string c = q + "controllers." + names[g] + ".";
                Y.gates[g] = {dmat(c + "down_weight"), dmat_stack({c + "forget_weight", c + "gain_weight"})};
            }
        }
        Experts& X = Y.ex;
        X.nvfp4 = l < MXFP8_FIRST;
        {   // the four expert tensors interleaved into one slot per expert, then the layer into its pool range
            const auto w13 = ck.bytes(q + "mlp.experts.w13_weight"), s13 = ck.bytes(q + "mlp.experts.w13_weight_scale");
            const auto w2 = ck.bytes(q + "mlp.experts.w2_weight"), s2 = ck.bytes(q + "mlp.experts.w2_weight_scale");
            const size_t a = w13.size() / NE, b = s13.size() / NE, c = w2.size() / NE, d = s2.size() / NE, eb = ex_bytes(l);
            if (a + b + c + d != eb || a + b != ex_partA(l)) { std::fprintf(stderr, "layer %d: expert bytes %zu\n", l, a + b + c + d); std::exit(1); }
            std::vector<uint8_t> lay(E.xhost ? 0 : (size_t) NE * eb);
            uint8_t* lp = E.xhost ? E.xhost + ex_layer_off(l) : lay.data();
            for (int e = 0; e < NE; ++e) {
                uint8_t* o = lp + (size_t) e * eb;
                std::memcpy(o, w13.data() + e * a, a);
                std::memcpy(o + a, s13.data() + e * b, b);
                std::memcpy(o + a + b, w2.data() + e * c, c);
                std::memcpy(o + a + b + c, s2.data() + e * d, d);
            }
            if (!E.xhost) {   // resident: the layer's range of its pool (offload: setup_offload fills the pools)
                uint8_t* dst = (X.nvfp4 ? E.pool_nv : E.pool_mx) + ex_layer_off(l) - (X.nvfp4 ? 0 : ex_layer_off(MXFP8_FIRST));
                CK(cudaMemcpy(dst, lay.data(), lay.size(), cudaMemcpyHostToDevice));
                for (int e = 0; e < NE; ++e) {
                    E.htabA[l * NE + e] = dst + (size_t) e * eb;
                    E.htabB[l * NE + e] = dst + (size_t) e * eb + ex_partA(l);
                }
            }
            X.tA = E.tabA + (size_t) l * NE;
            X.tB = E.tabB + (size_t) l * NE;
        }
        if (X.nvfp4) {
            const vec a13 = ck.f32(q + "mlp.experts.w13_weight_amax"), d13 = ck.f32(q + "mlp.experts.w13_weight_outer_scale_denominator");
            const vec a2 = ck.f32(q + "mlp.experts.w2_weight_amax"), d2 = ck.f32(q + "mlp.experts.w2_weight_outer_scale_denominator");
            vec f13(NE), f2(NE);
            for (int e = 0; e < NE; ++e) { f13[e] = a13[e] / d13[0]; f2[e] = a2[e] / d2[0]; }
            X.f13 = dvec(f13);
            X.f2 = dvec(f2);
        }
        std::fprintf(stderr, "\rloaded layer %d/%d, %.1f GiB in VRAM", l + 1, NL, g_vram / 1073741824.0);
    }
    std::fprintf(stderr, "\n");
    if (!E.xhost) {
        CK(cudaMemcpy(E.tabA, E.htabA.data(), E.htabA.size() * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(E.tabB, E.htabB.data(), E.htabB.size() * sizeof(void*), cudaMemcpyHostToDevice));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GGUF reader (one file, mapped read-only) and the loader for llama.cpp's merged inkling2_loop layout
struct GgTensor { uint32_t type = 0; int64_t ne[4] = {1, 1, 1, 1}; int nd = 0; const uint8_t* data = nullptr; size_t bytes = 0; };
struct Gguf {
    const uint8_t* base = nullptr;
    size_t size = 0;
    std::map<std::string, GgTensor> t;
    std::map<std::string, int64_t> ints;
    std::string arch;
    const GgTensor& get(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "GGUF: missing tensor %s\n", n.c_str()); std::exit(1); }
        return it->second;
    }
    bool has(const std::string& n) const { return t.count(n) != 0; }
};
size_t gg_row_bytes(uint32_t type, int64_t ne0) {
    switch (type) {
    case GG_F32: return (size_t) ne0 * 4;
    case GG_F16: case GG_BF16: return (size_t) ne0 * 2;
    case GG_Q8_0: return (size_t) ne0 / 32 * 34;
    case GG_IQ4_NL: return (size_t) ne0 / 32 * 18;
    case GG_IQ3_S: return (size_t) ne0 / 256 * 110;
    default: return 0;
    }
}
std::string gg_open(const std::string& path, Gguf& g) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return "cannot open " + path;
    struct stat st{};
    fstat(fd, &st);
    g.size = (size_t) st.st_size;
    void* m = mmap(nullptr, g.size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return "cannot map " + path;
    g.base = (const uint8_t*) m;
    size_t p = 0;
    auto rd = [&](void* dst, size_t n) { if (p + n > g.size) { std::fprintf(stderr, "GGUF: truncated\n"); std::exit(1); } std::memcpy(dst, g.base + p, n); p += n; };
    auto u32 = [&] { uint32_t v; rd(&v, 4); return v; };
    auto u64 = [&] { uint64_t v; rd(&v, 8); return v; };
    auto str = [&] { const uint64_t n = u64(); std::string r((const char*) g.base + p, n); p += n; return r; };
    if (u32() != 0x46554747u) return path + ": not a GGUF file";
    const uint32_t ver = u32();
    if (ver < 2) return path + ": GGUF version 1";
    const uint64_t nt = u64(), nkv = u64();
    const size_t sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    uint64_t align = 32;
    std::function<void(uint32_t, const std::string&)> skip = [&](uint32_t ty, const std::string& key) {
        if (ty == 8) { (void) str(); return; }
        if (ty == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            for (uint64_t i = 0; i < n; ++i) skip(et, "");
            return;
        }
        if (ty > 12) { std::fprintf(stderr, "GGUF: value type %u\n", ty); std::exit(1); }
        int64_t v = 0;
        if (ty == 4 || ty == 5) { uint32_t x; rd(&x, 4); v = ty == 5 ? (int32_t) x : x; }
        else if (ty == 10 || ty == 11) { uint64_t x; rd(&x, 8); v = (int64_t) x; }
        else p += sz[ty];
        if (!key.empty() && (ty == 4 || ty == 5 || ty == 10 || ty == 11)) g.ints[key] = v;
    };
    for (uint64_t i = 0; i < nkv; ++i) {
        const std::string key = str();
        const uint32_t ty = u32();
        if (key == "general.architecture" && ty == 8) { g.arch = str(); continue; }
        skip(ty, key);
        if (key == "general.alignment") align = (uint64_t) g.ints[key];
    }
    std::vector<std::pair<std::string, uint64_t>> offs;
    for (uint64_t i = 0; i < nt; ++i) {
        const std::string name = str();
        GgTensor x;
        x.nd = (int) u32();
        for (int d = 0; d < x.nd; ++d) x.ne[d] = (int64_t) u64();
        x.type = u32();
        const uint64_t off = u64();
        const size_t rb = gg_row_bytes(x.type, x.ne[0]);
        if (!rb) return name + ": ggml type " + std::to_string(x.type) + " is not supported";
        x.bytes = rb * (size_t) (x.ne[1] * x.ne[2] * x.ne[3]);
        g.t[name] = x;
        offs.push_back({name, off});
    }
    const size_t data = (p + align - 1) / align * align;
    for (auto& o : offs) {
        GgTensor& x = g.t[o.first];
        if (data + o.second + x.bytes > g.size) return o.first + ": data past the end of the file";
        x.data = g.base + data + o.second;
    }
    return "";
}

const Gguf* g_gg = nullptr;

uint16_t f32_to_bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// any supported 1-D / 2-D GGUF tensor as fp32 (row-major, ne0 fastest)
vec gg_f32(const std::string& n) {
    const GgTensor& x = g_gg->get(n);
    const size_t ne = (size_t) (x.ne[0] * x.ne[1] * x.ne[2] * x.ne[3]);
    vec out(ne);
    const uint8_t* b = x.data;
    switch (x.type) {
    case GG_F32: std::memcpy(out.data(), b, ne * 4); break;
    case GG_F16: for (size_t i = 0; i < ne; ++i) { uint16_t h; std::memcpy(&h, b + 2 * i, 2); out[i] = __half2float(*reinterpret_cast<__half*>(&h)); } break;
    case GG_BF16: for (size_t i = 0; i < ne; ++i) { uint16_t h; std::memcpy(&h, b + 2 * i, 2); uint32_t u = (uint32_t) h << 16; std::memcpy(&out[i], &u, 4); } break;
    case GG_Q8_0:
        for (size_t j = 0; j < ne / 32; ++j) {
            const uint8_t* q = b + j * 34;
            uint16_t h;
            std::memcpy(&h, q, 2);
            const float d = __half2float(*reinterpret_cast<__half*>(&h));
            for (int i = 0; i < 32; ++i) out[32 * j + i] = d * (float) (int8_t) q[2 + i];
        }
        break;
    default: std::fprintf(stderr, "%s: type %u where a dense tensor was expected\n", n.c_str(), x.type); std::exit(1);
    }
    return out;
}
// BF16 matrix from F32 / BF16 (exact) or another type (rounded; the loop gates, the router if not exact)
DMat gg_bf16_stack(const std::vector<std::string>& names) {
    DMat m;
    std::vector<uint16_t> all;
    size_t inexact = 0;
    for (const auto& n : names) {
        const GgTensor& x = g_gg->get(n);
        if (m.k && x.ne[0] != m.k) { std::fprintf(stderr, "%s: K %lld != %d\n", n.c_str(), (long long) x.ne[0], m.k); std::exit(1); }
        m.k = (int) x.ne[0];
        m.n += (int) (x.ne[1] * x.ne[2]);
        for (float f : gg_f32(n)) {
            const uint16_t h = f32_to_bf16(f);
            uint32_t u = (uint32_t) h << 16;
            float back;
            std::memcpy(&back, &u, 4);
            inexact += back != f;
            all.push_back(h);
        }
    }
    if (inexact && std::getenv("INK2_VERBOSE")) std::fprintf(stderr, "%s: %zu values rounded to BF16\n", names[0].c_str(), inexact);
    m.d = (const uint16_t*) upload(all.data(), all.size() * 2);
    return m;
}
// Q8_0 matrix (rows of the named tensors concatenated) repacked into codes [n][k] + scales [n][k / 32]
DMat gg_q8_stack(const std::vector<std::string>& names) {
    DMat m;
    std::vector<int8_t> q;
    std::vector<__half> sc;
    for (const auto& n : names) {
        const GgTensor& x = g_gg->get(n);
        if (x.type != GG_Q8_0) return gg_bf16_stack(names);   // a part in another type: the whole stack as BF16
        if (m.k && x.ne[0] != m.k) { std::fprintf(stderr, "%s: K differs\n", n.c_str()); std::exit(1); }
        m.k = (int) x.ne[0];
        const int64_t rows = x.ne[1] * x.ne[2];
        m.n += (int) rows;
        const size_t nb = (size_t) rows * (m.k / 32);
        for (size_t j = 0; j < nb; ++j) {
            const uint8_t* b = x.data + j * 34;
            __half h;
            std::memcpy(&h, b, 2);
            sc.push_back(h);
            for (int i = 0; i < 32; ++i) q.push_back((int8_t) b[2 + i]);
        }
    }
    m.q = (const int8_t*) upload(q.data(), q.size());
    m.qs = (const __half*) upload(sc.data(), sc.size() * 2);
    return m;
}
const float* gg_fvec(const std::string& n) { return dvec(gg_f32(n)); }
// a [rows][cols] tensor (ne0 = cols) transposed
vec gg_transpose(const std::string& n) {
    const GgTensor& x = g_gg->get(n);
    const vec a = gg_f32(n);
    const int64_t c = x.ne[0], r = x.ne[1];
    vec o(a.size());
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < c; ++j) o[(size_t) j * r + i] = a[(size_t) i * c + j];
    return o;
}

void load_gguf() {
    const Gguf& g = *g_gg;
    if (g.arch != "inkling2_loop") { std::fprintf(stderr, "GGUF architecture %s, expected inkling2_loop\n", g.arch.c_str()); std::exit(1); }
    auto need = [&](const char* k, int64_t v) {
        auto it = g.ints.find(std::string("inkling2_loop.") + k);
        if (it == g.ints.end() || it->second != v) { std::fprintf(stderr, "GGUF: %s is not %lld\n", k, (long long) v); std::exit(1); }
    };
    need("block_count", NL); need("embedding_length", H); need("expert_count", NE); need("expert_used_count", TOPK);
    need("loop.start", LOOP_BEGIN); need("loop.end", LOOP_END); need("loop.count", PASSES); need("loop.gate_rank", GATE_RANK);
    need("attention.sliding_window", WINDOW); need("expert_feed_forward_length", FF);
    E.embed_norm = gg_fvec("token_embd_norm.weight");
    E.norm = gg_fvec("output_norm.weight");
    E.final_query = gg_fvec("output_res_query.weight");
    {
        const GgTensor& x = g.get("token_embd.weight");
        if (x.type != GG_Q8_0 || x.ne[0] != H) { std::fprintf(stderr, "token_embd: Q8_0 [%d] expected\n", H); std::exit(1); }
        E.embed_q8.assign(x.data, x.data + x.bytes);
    }
    E.unembed = gg_q8_stack({"output.weight"});
    {
        vec mq((size_t) NLOOP * 4 * H);
        for (int i = 0; i < NLOOP; ++i) {
            const vec q = gg_f32("blk." + std::to_string(LOOP_BEGIN + i) + ".loop_msg_query.weight");
            if (q.size() != (size_t) 4 * H) { std::fprintf(stderr, "loop_msg_query: size\n"); std::exit(1); }
            std::memcpy(&mq[(size_t) i * 4 * H], q.data(), q.size() * 4);
        }
        E.msg_queries = dvec(mq);
    }
    {   // expert types: the same in every layer (one pool); slot = [gate_up rows | down rows], 256-byte multiples
        for (int l = 0; l < NL; ++l)
            for (const char* n : {"ffn_gate_up_exps.weight", "ffn_down_exps.weight"}) {
                const GgTensor& x = g.get("blk." + std::to_string(l) + "." + n);
                if (x.type != GG_IQ3_S && x.type != GG_IQ4_NL && x.type != GG_Q8_0) {
                    std::fprintf(stderr, "layer %d: expert type %u (IQ3_S, IQ4_NL or Q8_0)\n", l, x.type);
                    std::exit(1);
                }
            }
        g_gg_a = g.get("blk.0.ffn_gate_up_exps.weight").bytes / NE;
        g_gg_b = g.get("blk.0.ffn_down_exps.weight").bytes / NE;
        if (g_gg_a % 16 || g_gg_b % 16) { std::fprintf(stderr, "GGUF experts: %zu / %zu bytes, not 16-byte multiples\n", g_gg_a, g_gg_b); std::exit(1); }
        if (E.reserve_mib < 0) {
            E.pool_nv = (uint8_t*) dalloc(ex_layer_off(NL));
        } else {
            CK(cudaHostAlloc((void**) &E.xhost, ex_layer_off(NL), cudaHostAllocMapped | cudaHostAllocPortable));
            void* dv = nullptr;
            CK(cudaHostGetDevicePointer(&dv, E.xhost, 0));
            E.xhost_d = (const uint8_t*) dv;
        }
        E.tabA = (const uint8_t**) dalloc((size_t) NL * NE * sizeof(void*));
        E.tabB = (const uint8_t**) dalloc((size_t) NL * NE * sizeof(void*));
        E.htabA.assign((size_t) NL * NE, nullptr);
        E.htabB.assign((size_t) NL * NE, nullptr);
    }
    for (int l = 0; l < NL; ++l) {
        Layer& Y = E.L[l];
        const std::string b = "blk." + std::to_string(l) + ".";
        Y.full = full_layer(l);
        Y.loop = l >= LOOP_BEGIN && l < LOOP_END;
        {
            std::vector<std::string> parts = {b + "attn_q.weight", b + "attn_gate.weight", b + "attn_rel.weight",
                                              b + "attn_sink_proj.weight", b + "attn_k.weight"};
            if (Y.full) parts.push_back(b + "attn_k_conv_proj.weight");
            parts.push_back(b + "attn_v.weight");
            if (Y.full) parts.push_back(b + "attn_v_conv_proj.weight");
            Y.proj = gg_q8_stack(parts);
            Y.off_k = OFF_K;
            Y.n_k = KVD + (Y.full ? NKV * KC : 0);
            Y.off_v = Y.off_k + Y.n_k;
            Y.n_v = Y.n_k;
            if (Y.proj.n != Y.off_v + Y.n_v) { std::fprintf(stderr, "layer %d: stacked rows %d\n", l, Y.proj.n); std::exit(1); }
        }
        Y.wo = gg_q8_stack({b + "attn_output.weight"});
        Y.router = gg_bf16_stack({b + "ffn_gate_inp.weight"});
        Y.router_bias = gg_fvec(b + "exp_probs_b.bias");
        Y.q_scale = gg_fvec(b + "attn_q_clip.weight");
        Y.rel_scale = gg_fvec(b + "attn_rel_clip.weight");
        Y.rel_proj = dvec(gg_transpose(b + "attn_rel_bias.weight"));   // the converter stored proj^T
        Y.attn_conv = gg_fvec(b + "attn_post_conv.weight");
        Y.mlp_conv = gg_fvec(b + "ffn_post_conv.weight");
        Y.mlp_norm = gg_fvec(b + "ffn_norm.weight");
        if (Y.full) {   // static_w was stored as flip(0).T: transpose back, then flip the taps
            for (int kv = 0; kv < 2; ++kv) {
                vec t = gg_transpose(b + (kv ? "attn_v_conv.weight" : "attn_k_conv.weight"));
                const size_t taps = t.size() / KVD;
                vec f(t.size());
                for (size_t r = 0; r < taps; ++r) std::memcpy(&f[r * KVD], &t[(taps - 1 - r) * KVD], KVD * 4);
                (kv ? Y.v_static : Y.k_static) = dvec(f);
            }
            Y.q_norm = gg_fvec(b + "attn_q_inp_norm.weight");
            Y.k_norm = gg_fvec(b + "attn_k_inp_norm.weight");
            Y.v_norm = gg_fvec(b + "attn_v_inp_norm.weight");
        } else {
            Y.k_conv = gg_fvec(b + "attn_k_conv.weight");
            Y.v_conv = gg_fvec(b + "attn_v_conv.weight");
            Y.attn_norm = gg_fvec(b + "attn_norm.weight");
        }
        if (l > 0) Y.attn_query = gg_fvec(b + "attn_res_query.weight");
        Y.mlp_query = gg_fvec(b + "ffn_res_query.weight");
        if (Y.loop) {
            const char* names[4] = {"q", "k", "v", "ffn"};
            for (int k = 0; k < 4; ++k) {
                const std::string c = b + "loop_" + names[k] + "_ctrl_";
                Y.gates[k] = {gg_bf16_stack({c + "down.weight"}), gg_bf16_stack({c + "forget.weight", c + "gain.weight"})};
            }
        }
        Experts& X = Y.ex;
        const GgTensor& gu = g.get(b + "ffn_gate_up_exps.weight");
        const GgTensor& dn = g.get(b + "ffn_down_exps.weight");
        if (gu.ne[0] != H || gu.ne[1] != 2 * FF || gu.ne[2] != NE || dn.ne[0] != FF || dn.ne[1] != H || dn.ne[2] != NE) {
            std::fprintf(stderr, "layer %d: expert shapes\n", l);
            std::exit(1);
        }
        X.nvfp4 = false;
        X.gg13 = (int) gu.type;
        X.gg2 = (int) dn.type;
        X.rb13 = gg_row_bytes(gu.type, gu.ne[0]);
        X.rb2 = gg_row_bytes(dn.type, dn.ne[0]);
        if (gu.bytes / NE != g_gg_a || dn.bytes / NE != g_gg_b) { std::fprintf(stderr, "layer %d: expert sizes differ\n", l); std::exit(1); }
        {   // one slot per expert: [gate_up rows | down rows], then the layer into its pool range (offload: xhost)
            const size_t eb = ex_bytes(l);
            std::vector<uint8_t> lay(E.xhost ? 0 : (size_t) NE * eb);
            uint8_t* lp = E.xhost ? E.xhost + ex_layer_off(l) : lay.data();
            for (int e = 0; e < NE; ++e) {
                std::memcpy(lp + (size_t) e * eb, gu.data + (size_t) e * g_gg_a, g_gg_a);
                std::memcpy(lp + (size_t) e * eb + g_gg_a, dn.data + (size_t) e * g_gg_b, g_gg_b);
            }
            if (!E.xhost) {
                uint8_t* dst = E.pool_nv + ex_layer_off(l);
                CK(cudaMemcpy(dst, lay.data(), lay.size(), cudaMemcpyHostToDevice));
                for (int e = 0; e < NE; ++e) {
                    E.htabA[l * NE + e] = dst + (size_t) e * eb;
                    E.htabB[l * NE + e] = dst + (size_t) e * eb + g_gg_a;
                }
            }
            X.tA = E.tabA + (size_t) l * NE;
            X.tB = E.tabB + (size_t) l * NE;
        }
        std::fprintf(stderr, "\rloaded layer %d/%d, %.1f GiB in VRAM", l + 1, NL, g_vram / 1073741824.0);
    }
    std::fprintf(stderr, "\n");
    if (!E.xhost) {
        CK(cudaMemcpy(E.tabA, E.htabA.data(), E.htabA.size() * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(E.tabB, E.htabB.data(), E.htabB.size() * sizeof(void*), cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpyToSymbol(d_iq3s_grid, k_iq3s_grid, sizeof(k_iq3s_grid)));
    CK(cudaMemcpyToSymbol(c_iq4nl, k_iq4nl_values, sizeof(k_iq4nl_values)));
    CK(cudaMemcpyToSymbol(c_iq4nl_t, k_iq4nl_values, sizeof(k_iq4nl_values)));
}

// offload: once everything else is allocated, the pools take the free VRAM minus --vram-reserve-mib and a margin
// (decode graph, cuBLAS workspaces), split 80 / 20 between NVFP4 and MXFP8 bytes (two-pool LRU replay of corpus12k,
// lru2.py: 0.20 best at 6, 7 and 8 GiB).  Initial content: looped layers first, then the others, expert order.
// CBA: fixed 384 MiB margin and 0.20 split; per-card measurement / a routing profile if the first tokens matter.
int g_off_margin_mib = 128;   // INK2_OFF_MARGIN_MIB: VRAM kept beside --vram-reserve-mib (the decode graph; cuBLAS is warmed up at load)
void setup_offload() {
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t keep = ((size_t) E.reserve_mib << 20) + ((size_t) g_off_margin_mib << 20);
    const size_t budget = fr > keep ? fr - keep : 0;
    if (g_is_gguf) {   // one format, one pool
        E.nslot_mx = 0;
        E.nslot_nv = (int) std::min<size_t>(budget / ex_bytes(0), (size_t) NL * NE);
    } else {
        E.nslot_mx = (int) std::min<size_t>((size_t) (budget * 0.2) / MX_BYTES, (size_t) (NL - MXFP8_FIRST) * NE);
        E.nslot_nv = (int) std::min<size_t>((budget - (size_t) E.nslot_mx * MX_BYTES) / NV_BYTES, (size_t) MXFP8_FIRST * NE);
    }
    if (E.nslot_nv < FETCH_MAX || (!g_is_gguf && E.nslot_mx < FETCH_MAX)) {
        std::fprintf(stderr, "offload: %.2f GiB free, reserve %d MiB: room for %d + %d expert slots (need >= %d each)\n",
                     fr / 1073741824.0, E.reserve_mib, E.nslot_nv, E.nslot_mx, FETCH_MAX);
        std::exit(1);
    }
    const int cap_nv = g_is_gguf ? NL * NE : MXFP8_FIRST * NE, cap_mx = g_is_gguf ? 0 : (NL - MXFP8_FIRST) * NE;
    if (E.grow) {   // growable: --kv-grow takes slots back as the context grows (pool_resize)
        E.vpool_nv.init((size_t) cap_nv * ex_bytes(0));
        if (!E.vpool_nv.map_to((size_t) E.nslot_nv * ex_bytes(0))) { std::fprintf(stderr, "offload: pool mapping failed\n"); std::exit(1); }
        E.pool_nv = (uint8_t*) E.vpool_nv.ptr();
        if (cap_mx) {
            E.vpool_mx.init((size_t) cap_mx * MX_BYTES);
            if (!E.vpool_mx.map_to((size_t) E.nslot_mx * MX_BYTES)) { std::fprintf(stderr, "offload: pool mapping failed\n"); std::exit(1); }
            E.pool_mx = (uint8_t*) E.vpool_mx.ptr();
        }
    } else {
        E.pool_nv = (uint8_t*) dalloc((size_t) E.nslot_nv * ex_bytes(0));
        if (E.nslot_mx) E.pool_mx = (uint8_t*) dalloc((size_t) E.nslot_mx * MX_BYTES);
    }
    std::vector<int> slot_of((size_t) NL * NE, -1), own_nv(E.grow ? cap_nv : E.nslot_nv, -1),
                     own_mx(E.grow ? cap_mx : E.nslot_mx, -1);
    std::vector<int> order;
    for (int l = LOOP_BEGIN; l < LOOP_END; ++l) order.push_back(l);
    for (int l = 0; l < NL; ++l) if (l < LOOP_BEGIN || l >= LOOP_END) order.push_back(l);
    int used_nv = 0, used_mx = 0;
    for (int l : order)
        for (int e = 0; e < NE; ++e) {
            const bool nv = ex_nv(l);
            int& used = nv ? used_nv : used_mx;
            if (used >= (nv ? E.nslot_nv : E.nslot_mx)) continue;
            const size_t eb = ex_bytes(l);
            uint8_t* d = (nv ? E.pool_nv : E.pool_mx) + (size_t) used * eb;
            CK(cudaMemcpy(d, E.xhost + ex_layer_off(l) + (size_t) e * eb, eb, cudaMemcpyHostToDevice));
            slot_of[l * NE + e] = used;
            (nv ? own_nv : own_mx)[used] = l * NE + e;
            E.htabA[l * NE + e] = d;
            E.htabB[l * NE + e] = d + ex_partA(l);
            ++used;
        }
    CK(cudaMemcpy(E.tabA, E.htabA.data(), E.htabA.size() * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(E.tabB, E.htabB.data(), E.htabB.size() * sizeof(void*), cudaMemcpyHostToDevice));
    auto ints = [](const std::vector<int>& v) { return (int*) upload(v.data(), v.size() * 4); };
    E.slot_of = ints(slot_of);
    E.owner_nv = ints(own_nv);
    E.owner_mx = ints(own_mx);
    E.stamp_nv = (unsigned*) dalloc(own_nv.size() * 4);
    E.stamp_mx = (unsigned*) dalloc(own_mx.size() * 4);
    E.clock = (unsigned*) dalloc(4);
    E.nfetch = (int*) dalloc(4);
    E.fsrc = (const uint8_t**) dalloc(FETCH_MAX * sizeof(void*));
    E.fdst = (uint8_t**) dalloc(FETCH_MAX * sizeof(void*));
    E.fbytes = (unsigned long long*) dalloc(8);
    std::fprintf(stderr, "offload: %.2f GiB free after the rest, reserve %d MiB -> %d NVFP4 + %d MXFP8 expert slots "
                         "(%.2f GiB, %.0f %% of the experts)\n", fr / 1073741824.0, E.reserve_mib, E.nslot_nv, E.nslot_mx,
                 ((size_t) E.nslot_nv * ex_bytes(0) + (size_t) E.nslot_mx * MX_BYTES) / 1073741824.0,
                 100.0 * ((size_t) E.nslot_nv * ex_bytes(0) + (size_t) E.nslot_mx * MX_BYTES) / ex_layer_off(NL));
}

// bytes per cell of a slot's K (0), V (1), K scales (2), V scales (3) in format E.kvm
size_t kv_cell_bytes(int a) {
    const int m = E.kvm;
    switch (a) {
    case 0: return (size_t) KVD * (m == 0 ? 4 : m == 1 ? 2 : 1);
    case 1: return m == 4 ? KVD / 2 : (size_t) KVD * (m == 0 ? 4 : m == 1 ? 2 : 1);
    case 2: return m >= 2 ? (KVD / KVG) * 2 : 0;
    default: return m >= 2 ? (KVD / (m == 4 ? V4G : KVG)) * 2 : 0;
    }
}
std::vector<VRange> g_kvr;      // --kv-grow: the full slots' arrays
std::vector<size_t> g_kvr_bpc;  // their bytes per cell
size_t kv_mapped_for(int cells) {
    size_t t = 0;
    for (size_t i = 0; i < g_kvr.size(); ++i) t += g_kvr[i].need((size_t) cells * g_kvr_bpc[i]);
    return t;
}

// --kv-grow with offload: a pool's slot count changes.  Nothing may be running on the device; the decode graph (the
// slot count is one of its launch arguments) is rebuilt at the next token.
size_t vram_free() { size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot)); return fr; }
size_t vram_keep() { return ((size_t) std::max(E.reserve_mib, 0) << 20) + ((size_t) g_off_margin_mib << 20); }   // as setup_offload
void pool_resize(bool nv, int n) {
    int& ns = nv ? E.nslot_nv : E.nslot_mx;
    VRange& r = nv ? E.vpool_nv : E.vpool_mx;
    const size_t eb = nv ? ex_bytes(0) : MX_BYTES;
    n = std::max(0, std::min(n, (int) (r.reserved / eb)));
    if (n < ns) {
        k_pool_evict<<<(ns - n + 255) / 256, 256>>>(nv ? E.owner_nv : E.owner_mx, E.slot_of, E.tabA, E.tabB,
                                                    nv ? E.stamp_nv : E.stamp_mx, n, ns);
        CK(cudaDeviceSynchronize());
        r.unmap_to((size_t) n * eb);
    } else if (n > ns) {
        if (!r.map_to((size_t) n * eb)) n = std::max(ns, (int) (r.mapped / eb));
    }
    ns = n;
    if (E.exec) { CK(cudaGraphExecDestroy(E.exec)); E.exec = nullptr; }
}
double pool_bytes(bool nv) { return (double) (nv ? E.nslot_nv : E.nslot_mx) * (nv ? ex_bytes(0) : MX_BYTES); }
// the pools give up `bytes`, in proportion to their sizes
void pools_release(double bytes, const char* what) {
    const double deficit = bytes + 4.0 * g_vmm.gran, used = pool_bytes(true) + pool_bytes(false);
    for (bool nv : {true, false}) {
        const int ns = nv ? E.nslot_nv : E.nslot_mx;
        if (!ns) continue;
        const int drop = (int) std::ceil(deficit * pool_bytes(nv) / used / (nv ? ex_bytes(0) : MX_BYTES));
        if (ns - drop < FETCH_MAX) { std::fprintf(stderr, "%s would leave the expert pool %d slots\n", what, ns - drop); std::exit(1); }
        pool_resize(nv, ns - drop);
    }
}
// the pools take what is free beyond the margin, in proportion to their sizes
void pools_fill() {
    if (!E.xhost || !E.grow) return;
    const size_t fr = vram_free(), keep = vram_keep() + 4 * g_vmm.gran;
    if (fr <= keep) return;
    const double spare = (double) (fr - keep), used = pool_bytes(true) + pool_bytes(false);
    for (bool nv : {true, false}) {
        const int ns = nv ? E.nslot_nv : E.nslot_mx;
        if (!ns) continue;
        pool_resize(nv, ns + (int) (spare * pool_bytes(nv) / used / (nv ? ex_bytes(0) : MX_BYTES)));
    }
}
// the full slots hold `need` cells (grown in --kv-grow-step steps, from the expert pools when VRAM is short)
void kv_fit(int need) {
    if (!E.grow || need <= E.kv_cells) return;
    const int nc = std::min(E.ctx, (need + E.grow_step - 1) / E.grow_step * E.grow_step);
    CK(cudaDeviceSynchronize());
    const size_t add = kv_mapped_for(nc) - kv_mapped_for(E.kv_cells), keep = vram_keep(), fr = vram_free();
    if (fr < add + keep && E.xhost) pools_release((double) (add + keep - fr), "the K/V");
    for (size_t i = 0; i < g_kvr.size(); ++i)
        if (!g_kvr[i].map_to((size_t) nc * g_kvr_bpc[i])) { std::fprintf(stderr, "--kv-grow: out of VRAM at %d cells\n", nc); std::exit(1); }
    E.kv_cells = nc;
    std::fprintf(stderr, "kv-grow: %d cells (%.2f GiB K/V), expert slots %d + %d\n", nc, kv_mapped_for(nc) / 1073741824.0,
                 E.nslot_nv, E.nslot_mx);
}
// a request that needs only `need` cells: the rest of the mapped K/V back to the pools
void kv_trim(int need) {
    if (!E.grow) return;
    const int nc = std::min(E.ctx, std::max(E.grow_step, (need + E.grow_step - 1) / E.grow_step * E.grow_step));
    if (E.kv_cells <= nc + E.grow_step) return;
    CK(cudaDeviceSynchronize());
    for (size_t i = 0; i < g_kvr.size(); ++i) g_kvr[i].unmap_to((size_t) nc * g_kvr_bpc[i]);
    E.kv_cells = nc;
    pools_fill();
    std::fprintf(stderr, "kv-grow: back to %d cells, expert slots %d + %d\n", nc, E.nslot_nv, E.nslot_mx);
}

void alloc_state() {
    if (E.grow) g_kvr.reserve(4 * NCALLS);   // VRange addresses are handed out: no reallocation
    CK(cudaMemcpyToSymbol(c_kvm, &E.kvm, sizeof(int)));
    E.nchunk_full = (E.ctx + CHUNK - 1) / CHUNK;
    for (int s = 0; s < NCALLS; ++s) {
        const int l = s < NL ? s : LOOP_BEGIN + (s - NL) % NLOOP;
        Slot& S = E.S[s];
        S.full = full_layer(l);
        S.cap = S.full ? E.ctx : WINDOW + PF_RING_EXTRA;
        void** arr[4] = {&S.K, &S.V, (void**) &S.ksc, (void**) &S.vsc};
        for (int a = 0; a < 4; ++a) {
            const size_t b = kv_cell_bytes(a);
            if (!b) continue;
            if (E.grow && S.full) {
                g_kvr.emplace_back();
                VRange& r = g_kvr.back();
                r.init((size_t) S.cap * b);
                g_kvr_bpc.push_back(b);
                *arr[a] = r.ptr();
            } else {
                *arr[a] = dalloc((size_t) S.cap * b);
            }
        }
        S.khist = dfloats(3 * KVD);
        S.vhist = dfloats(3 * KVD);
        S.ahist = dfloats(3 * H);
        S.mhist = dfloats(3 * H);
    }
    if (E.grow) {
        E.kv_cells = std::min(E.ctx, E.grow_step);
        for (size_t i = 0; i < g_kvr.size(); ++i)
            if (!g_kvr[i].map_to((size_t) E.kv_cells * g_kvr_bpc[i])) { std::fprintf(stderr, "--kv-grow: no VRAM for %d cells\n", E.kv_cells); std::exit(1); }
    }
    E.d_pos = (int*) dalloc(4);
    E.h = dfloats(H); E.out = dfloats(H); E.Bbuf = dfloats(8 * H); E.preB = dfloats(8 * H); E.pre_h = dfloats(H);
    E.prev_stack = dfloats(8 * H); E.msgs = dfloats(4 * H); E.a = dfloats(3 * H);
    for (auto& t : E.t) t = dfloats(H);
    E.xq = dfloats(H); E.xk = dfloats(H); E.xv = dfloats(H);
    E.q = dfloats(QD); E.k = dfloats(KVD); E.v = dfloats(KVD); E.kc = dfloats(KVD); E.vc = dfloats(KVD); E.dw = dfloats(64);
    E.rs = dfloats(NQ * DREL); E.rel = dfloats(NQ * EXTENT); E.sink = dfloats(NQ); E.gate = dfloats(QD);
    E.att = dfloats(QD); E.attn_out = dfloats(H); E.att2 = dfloats(H); E.r = dfloats(H);
    E.m = dfloats(H); E.mn = dfloats(H); E.cat = dfloats(3 * 2 * H); E.gdown = dfloats(3 * GATE_RANK); E.gf = dfloats(H);
    E.gg = dfloats(H); E.rl = dfloats(NE); E.wts = dfloats(TOPK); E.ids = (int*) dalloc(TOPK * 4);
    E.gu = dfloats((size_t) TOPK * 2 * FF); E.act = dfloats((size_t) TOPK * FF); E.ey = dfloats((size_t) TOPK * H);
    E.moe = dfloats(H); E.y2 = dfloats(H); E.logits = dfloats(VOCAB_ROWS); E.emb_in = dfloats(H); E.pout = dfloats(OFF_K + 2 * (KVD + NKV * KC)); E.gfg = dfloats(3 * 2 * H);
    const int nc = std::max(E.nchunk_full, E.nchunk_slide);
    E.pm = dfloats((size_t) NQ * nc); E.pl = dfloats((size_t) NQ * nc); E.po = dfloats((size_t) NQ * nc * HD);
}

// ---------------------------------------------------------------------------------------------------------------------
// The forward pass (device pointers; the block-state count is static per call, kept on the host)

void gemv(const DMat& W, const float* x, float* y, int rows = -1, int first = 0) {   // rows [first, first + rows)
    const int n = rows < 0 ? W.n - first : rows;
    // small N (k / v projections: 1024 rows, wo: 2304) gave 32-72 blocks for 82 SMs at R = 4 (420-670 GB/s): fewer rows
    // per warp there, more blocks; per-row arithmetic is the same whatever R (bit-identical)
    int R = g_rpw;
    while (R > 1 && (n + 8 * R - 1) / (8 * R) < g_gemv_min_blocks) R /= 2;
    const unsigned g = (unsigned) ((n + 8 * R - 1) / (8 * R));
    const size_t sm = (size_t) W.k * 4;
    if (W.q) {
        const int8_t* wq = W.q + (size_t) first * W.k;
        const __half* ws = W.qs + (size_t) first * (W.k / 32);
        switch (R) {
        case 1: k_gemv_q8<1><<<g, 256, sm>>>(wq, ws, x, y, n, W.k); break;
        case 2: k_gemv_q8<2><<<g, 256, sm>>>(wq, ws, x, y, n, W.k); break;
        case 8: k_gemv_q8<8><<<g, 256, sm>>>(wq, ws, x, y, n, W.k); break;
        default: k_gemv_q8<4><<<g, 256, sm>>>(wq, ws, x, y, n, W.k);
        }
        return;
    }
    const uint16_t* w = W.d + (size_t) first * W.k;
    switch (R) {
    case 1: k_gemv_bf16<1><<<g, 256, sm>>>(w, x, y, n, W.k); break;
    case 2: k_gemv_bf16<2><<<g, 256, sm>>>(w, x, y, n, W.k); break;
    case 8: k_gemv_bf16<8><<<g, 256, sm>>>(w, x, y, n, W.k); break;
    default: k_gemv_bf16<4><<<g, 256, sm>>>(w, x, y, n, W.k);
    }
}
void rms(const float* x, const float* w, float* y, int n = H) { k_rms<<<1, 512>>>(x, w, y, n); }
void copy(float* dst, const float* src, size_t n) { CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice)); }

// Inkling2LoopMessageGate, then rms(.) * nw into yn (every gated stream is normalized next)
// n (<= 3) independent gates in one launch per stage
void message_gates(int n, const Gate* const g[], const float* const x[], const float* const m[], float* const y[],
                   const float* const nw[], float* const yn[]) {
    GateArgs a{};
    for (int j = 0; j < n; ++j) {
        a.down[j] = g[j]->down.d; a.fg[j] = g[j]->fg.d; a.x[j] = x[j]; a.m[j] = m[j]; a.y[j] = y[j]; a.nw[j] = nw[j];
        a.yn[j] = yn[j];
    }
    k_rms2<<<dim3(2, n), 512>>>(a, E.cat);
    k_gate_down<<<dim3(GATE_RANK, n), 256>>>(a, E.cat, E.gdown, g[0]->down.k);
    const int N = g[0]->fg.n, K = g[0]->fg.k;   // as gemv(): R halved while the grid is short
    int R = g_rpw;
    while (R > 1 && (N + 8 * R - 1) / (8 * R) < g_gemv_min_blocks) R /= 2;
    const dim3 grid((unsigned) ((N + 8 * R - 1) / (8 * R)), n);
    const size_t sm = (size_t) K * 4;
    switch (R) {
    case 1: k_gemv_bf16_gate<1><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K); break;
    case 2: k_gemv_bf16_gate<2><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K); break;
    case 8: k_gemv_bf16_gate<8><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K); break;
    default: k_gemv_bf16_gate<4><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K);
    }
    k_gate_mix_rms<<<n, 512>>>(a, E.gfg);
}

// attnres dispatch on (S, NQ)
void attnres(const float* stack, int S, const float* queries, int nq, float* out, const float* nw0 = nullptr,
             const float* nw1 = nullptr, const float* nw2 = nullptr, float* outn = nullptr, int T = 1) {
    const size_t ss = (size_t) T * H;   // T tokens: entries [S][T][H]
#define AR_CASE(SS, QQ) if (S == SS && nq == QQ) { k_attnres<SS, QQ><<<T, AR_T>>>(stack, ss, queries, out, nw0, nw1, nw2, outn); return; }
    AR_CASE(1, 1) AR_CASE(2, 1) AR_CASE(3, 1) AR_CASE(4, 1) AR_CASE(5, 1) AR_CASE(6, 1)
    AR_CASE(2, 3) AR_CASE(3, 3) AR_CASE(4, 3) AR_CASE(5, 3)
    AR_CASE(2, 4) AR_CASE(3, 4)
#undef AR_CASE
    std::fprintf(stderr, "attnres: no kernel for S=%d nq=%d\n", S, nq);
    std::exit(1);
}

void attention(const Layer& Y, Slot& S, const float* xq, const float* xk, const float* xv) {
    float* P = E.pout;
    if (xq == xk && xk == xv) {
        gemv(Y.proj, xq, P);
    } else {
        const int N = Y.off_v + Y.n_v, K = Y.proj.k;   // one launch, three inputs (rows: q.. | k | v)
        int R = g_rpw;
        while (R > 1 && ((N + 8 * R - 1) / (8 * R) < g_gemv_min_blocks || Y.off_k % R || Y.off_v % R)) R /= 2;
        const unsigned g = (unsigned) ((N + 8 * R - 1) / (8 * R));
        const size_t sm = (size_t) 3 * K * 4;
        if (Y.proj.q) {
            switch (R) {
            case 1: k_gemv_q8_qkv<1><<<g, 256, sm>>>(Y.proj.q, Y.proj.qs, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
            case 2: k_gemv_q8_qkv<2><<<g, 256, sm>>>(Y.proj.q, Y.proj.qs, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
            case 8: k_gemv_q8_qkv<8><<<g, 256, sm>>>(Y.proj.q, Y.proj.qs, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
            default: k_gemv_q8_qkv<4><<<g, 256, sm>>>(Y.proj.q, Y.proj.qs, xq, xk, xv, Y.off_k, Y.off_v, P, N, K);
            }
        } else switch (R) {
        case 1: k_gemv_bf16_qkv<1><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
        case 2: k_gemv_bf16_qkv<2><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
        case 8: k_gemv_bf16_qkv<8><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
        default: k_gemv_bf16_qkv<4><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K);
        }
    }
    float *q = P, *k = P + Y.off_k, *v = P + Y.off_v;
    float *rs = P + OFF_REL, *sink = P + OFF_SINK, *gate = P + OFF_GATE;
    static_assert(HD == 128, "k_attn_prep: 128-thread blocks for q heads, rs heads and k / v groups");
    if (Y.full)
        k_attn_prep<true><<<2 * NQ + NKV, 128>>>(q, Y.q_scale, rs, Y.rel_scale, Y.rel_proj, E.rel, 1, k, v, S.khist, S.vhist,
                                                Y.k_static, Y.v_static, S.K, S.V, S.ksc, S.vsc, S.cap, E.d_pos);
    else
        k_attn_prep<false><<<2 * NQ + NKV, 128>>>(q, Y.q_scale, rs, Y.rel_scale, Y.rel_proj, E.rel, 0, k, v, S.khist,
                                                 S.vhist, Y.k_conv, Y.v_conv, S.K, S.V, S.ksc, S.vsc, S.cap, E.d_pos);
    const int nch = S.full ? E.nchunk_full : E.nchunk_slide;
    k_attn_partial<<<dim3(NQ, nch), 128>>>(q, S.K, S.V, S.ksc, S.vsc, E.rel, S.cap, S.full ? 0 : WINDOW, E.d_pos, E.pm, E.pl, E.po);
    k_attn_merge<<<NQ, HD>>>(E.pm, E.pl, E.po, nch, sink, gate, E.att);
    gemv(Y.wo, E.att, E.attn_out);
}

// offload: make layer l's experts ids[0..n) resident (plan + fetch, device side, graph-capturable)
void expert_cache(int l, const int* ids, int n, cudaStream_t st = cudaStreamPerThread, int blocks = FETCH_BLOCKS) {
    const bool nv = ex_nv(l);
    CacheArgs c{E.slot_of, nv ? E.owner_nv : E.owner_mx, nv ? E.stamp_nv : E.stamp_mx, E.clock,
                nv ? E.nslot_nv : E.nslot_mx, E.tabA, E.tabB, nv ? E.pool_nv : E.pool_mx, ex_bytes(l), ex_partA(l),
                E.xhost_d + ex_layer_off(l), E.fsrc, E.fdst, E.nfetch, E.fbytes};
    k_cache_plan<<<1, 1024, 0, st>>>(l, ids, n, c);
    k_fetch<<<dim3(blocks, std::min(n, FETCH_MAX)), 256, 0, st>>>(E.fsrc, E.fdst, E.nfetch, ex_bytes(l));
}

// Decode prefetch (offload, INK2_PREFETCH=1, default): after a layer application has routed, the next application's
// router is applied to the same MoE input and its predicted experts are fetched on a side stream while this layer's
// experts run; the next layer waits for the side stream before its own plan.  A wrong guess costs a copy and an LRU
// stamp, never a result: the next layer still plans and fetches exactly what it routes to.  The victims are the least
// recently used slots, never the running layer's experts (just stamped).
__global__ void k_pred_stat(const int* pred, const int* ids, unsigned long long* st) {
    if (threadIdx.x) return;
    int m = 0;
    for (int i = 0; i < TOPK; ++i) for (int j = 0; j < TOPK; ++j) m += pred[i] == ids[j];
    atomicAdd(st, (unsigned long long) m);
    atomicAdd(st + 1, (unsigned long long) TOPK);
}
unsigned long long* g_pred_st = nullptr;   // INK2_PREFETCH_STAT=1: [matched, predicted]
bool g_dec_prefetch = false;   // INK2_PREFETCH=1: measured slower on PCIe 3.0 x8 (the link is the bottleneck)
int g_dec_pf_blocks = 32;          // INK2_PREFETCH_BLOCKS: k_fetch blocks of the side copy (SMs left to the compute)
cudaStream_t g_dec_ps = nullptr;
cudaEvent_t g_dec_fork = nullptr, g_dec_join = nullptr;
bool g_dec_pending = false;
int g_dec_app = 0;
std::vector<int> g_dec_seq;        // the layer of each application, in order
float* g_dec_rl = nullptr;         // [NE] predicted router logits
int* g_dec_ids = nullptr;          // [TOPK]
float* g_dec_wts = nullptr;
void dec_prefetch_init() {
    for (int l = 0; l < LOOP_END; ++l) g_dec_seq.push_back(l);
    for (int p = 1; p < PASSES; ++p) for (int l = LOOP_BEGIN; l < LOOP_END; ++l) g_dec_seq.push_back(l);
    for (int l = LOOP_END; l < NL; ++l) g_dec_seq.push_back(l);
    CK(cudaStreamCreateWithFlags(&g_dec_ps, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g_dec_fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&g_dec_join, cudaEventDisableTiming));
    g_dec_rl = dfloats(NE);
    g_dec_wts = dfloats(TOPK);
    g_dec_ids = (int*) dalloc(TOPK * 4);
    if (std::getenv("INK2_PREFETCH_STAT")) g_pred_st = (unsigned long long*) dalloc(16);
}

// GGUF expert GEMV launches (runtime ggml type -> kernel), INK2_GG_RPW rows per warp (2, 4, 8; default 4)
int g_gg_rpw = 4;
template <bool SWI, bool PERM, int R>
void gg_sel_r(int t, const uint8_t* const* tab, size_t rb, const float* x, int xs, float* y, int N, int K) {
    const dim3 g((unsigned) (N / (8 * R)), TOPK);
    const size_t sm = (size_t) (K + 512 + 4) * 4;
    switch (t) {
    case GG_IQ3_S: k_gemv_gg_sel<GG_IQ3_S, R, SWI, PERM><<<g, 256, sm>>>(tab, rb, E.ids, x, xs, y, N, K); break;
    case GG_IQ4_NL: k_gemv_gg_sel<GG_IQ4_NL, R, SWI, PERM><<<g, 256, sm>>>(tab, rb, E.ids, x, xs, y, N, K); break;
    default: k_gemv_gg_sel<GG_Q8_0, R, SWI, PERM><<<g, 256, sm>>>(tab, rb, E.ids, x, xs, y, N, K);
    }
}
bool g_gg_q8 = false;  // INK2_GG_Q8=1: the integer (dp4a) expert GEMV (+6% decode, KL 1e-2 from the int8 inputs)
gq8_1* g_yq = nullptr;  // decode: the experts' inputs as q8 blocks [TOPK][max(H, FF) / 32]
template <bool SWI, bool PERM>
void gg_sel(int t, const uint8_t* const* tab, size_t rb, const float* x, int xs, float* y, int N, int K) {
    if (g_gg_q8) {
        const int nb = K / 32, rows = SWI ? TOPK : 1;   // gate_up: one input for every expert
        k_quant_q8_1<SWI><<<dim3((nb + 7) / 8, rows), 256>>>(x, xs, g_yq, K);
        const dim3 g((unsigned) ((N + 7) / 8), TOPK);
        const int st = SWI ? nb : 0;
        switch (t) {
        case GG_IQ3_S: k_gemv_gg_q8<GG_IQ3_S, PERM><<<g, 256>>>(tab, rb, E.ids, g_yq, st, y, N, K); break;
        case GG_IQ4_NL: k_gemv_gg_q8<GG_IQ4_NL, PERM><<<g, 256>>>(tab, rb, E.ids, g_yq, st, y, N, K); break;
        default: k_gemv_gg_q8<GG_Q8_0, PERM><<<g, 256>>>(tab, rb, E.ids, g_yq, st, y, N, K);
        }
        return;
    }
    switch (g_gg_rpw) {
    case 2: gg_sel_r<SWI, PERM, 2>(t, tab, rb, x, xs, y, N, K); break;
    case 8: gg_sel_r<SWI, PERM, 8>(t, tab, rb, x, xs, y, N, K); break;
    default: gg_sel_r<SWI, PERM, 4>(t, tab, rb, x, xs, y, N, K);
    }
}

void moe(const Layer& Y, const float* x, const float* r, float* mhist, const float* mlp_conv, float* out) {
    k_gemv_bf16_rowblock<<<NE, 256>>>(Y.router.d, x, E.rl, H);
    k_router<<<1, NE>>>(E.rl, Y.router_bias, E.ids, E.wts);
    if (E.rec_mode == 1) k_rec_ids<<<1, TOPK>>>(E.ids, E.rec + (size_t) g_slot * E.ctx * TOPK, E.d_pos);
    const bool pf = E.xhost && g_dec_prefetch && !g_dec_seq.empty();
    if (pf && g_dec_pending) {
        CK(cudaStreamWaitEvent(cudaStreamPerThread, g_dec_join, 0));
        g_dec_pending = false;
        if (g_pred_st) k_pred_stat<<<1, 32>>>(g_dec_ids, E.ids, g_pred_st);
    }
    if (E.xhost) expert_cache(int(&Y - E.L), E.ids, TOPK);
    if (pf && g_dec_app + 1 < (int) g_dec_seq.size()) {   // guess the next application's experts from this input
        const Layer& Yn = E.L[g_dec_seq[g_dec_app + 1]];
        k_gemv_bf16_rowblock<<<NE, 256>>>(Yn.router.d, x, g_dec_rl, H);
        k_router<<<1, NE>>>(g_dec_rl, Yn.router_bias, g_dec_ids, g_dec_wts);
        CK(cudaEventRecord(g_dec_fork, cudaStreamPerThread));
        CK(cudaStreamWaitEvent(g_dec_ps, g_dec_fork, 0));
        expert_cache(g_dec_seq[g_dec_app + 1], g_dec_ids, TOPK, g_dec_ps, g_dec_pf_blocks);
        CK(cudaEventRecord(g_dec_join, g_dec_ps));
        g_dec_pending = true;
    }
    ++g_dec_app;
    const Experts& X = Y.ex;
    if (X.gg13) {
        gg_sel<false, true>(X.gg13, X.tA, X.rb13, x, 0, E.gu, 2 * FF, H);
        gg_sel<true, false>(X.gg2, X.tB, X.rb2, E.gu, 2 * FF, E.ey, H, FF);
    } else if (X.nvfp4) {
#define SEL(R) \
        k_gemv_nvfp4_sel<R><<<dim3(2 * FF / (8 * R), TOPK), 256, (H + 256) * 4>>>(X.tA, X.f13, E.ids, x, 0, E.gu, 2 * FF, H); \
        k_gemv_nvfp4_sel<R, true><<<dim3(H / (8 * R), TOPK), 256, (FF + 256) * 4>>>(X.tB, X.f2, E.ids, E.gu, 2 * FF, E.ey, H, FF);
        switch (g_sel_rpw) { case 1: SEL(1) break; case 2: SEL(2) break; case 8: SEL(8) break; default: SEL(4) }
#undef SEL
    } else {
#define SEL(R) \
        k_gemv_mxfp8_sel<R><<<dim3(2 * FF / (8 * R), TOPK), 256, H * 4>>>(X.tA, E.ids, x, 0, E.gu, 2 * FF, H); \
        k_gemv_mxfp8_sel<R, true><<<dim3(H / (8 * R), TOPK), 256, FF * 4>>>(X.tB, E.ids, E.gu, 2 * FF, E.ey, H, FF);
        switch (g_sel_rpw) { case 1: SEL(1) break; case 2: SEL(2) break; case 8: SEL(8) break; default: SEL(4) }
#undef SEL
    }
    k_moe_tail<<<(H + 255) / 256, 256>>>(E.ey, E.wts, mhist, mlp_conv, r, out, E.d_pos);
}

// one layer application: input E.h, block states Bbuf[0..nB), output E.out; msgs = 4 x H or nullptr
void layer_forward(int l, Slot& S, int nB, const float* msgs) {
    const Layer& Y = E.L[l];
    float* stack_h = E.Bbuf + (size_t) nB * H;
    copy(stack_h, E.h, H);   // stack [B..., h]
    const float *xq, *xk, *xv;
    if (Y.full) {
        if (!msgs) {   // l > 0 always (full layers are 5, 11, 17, 23)
            attnres(E.Bbuf, nB + 1, Y.attn_query, 3, nullptr, Y.q_norm, Y.k_norm, Y.v_norm, E.a);
            xq = E.a; xk = E.a + H; xv = E.a + 2 * H;
        } else {
            attnres(E.Bbuf, nB + 1, Y.attn_query, 3, E.a);
            const Gate* g3[3] = {&Y.gates[0], &Y.gates[1], &Y.gates[2]};
            const float* x3[3] = {E.a, E.a + H, E.a + 2 * H};
            const float* m3[3] = {msgs, msgs + H, msgs + 2 * H};
            float* y3[3] = {E.t[0], E.t[1], E.t[2]};
            const float* nw3[3] = {Y.q_norm, Y.k_norm, Y.v_norm};
            float* yn3[3] = {E.xq, E.xk, E.xv};
            message_gates(3, g3, x3, m3, y3, nw3, yn3);
            xq = E.xq; xk = E.xk; xv = E.xv;
        }
    } else if (!msgs) {
        if (l > 0) attnres(E.Bbuf, nB + 1, Y.attn_query, 1, nullptr, Y.attn_norm, nullptr, nullptr, E.xq);
        else rms(E.h, Y.attn_norm, E.xq);
        xq = xk = xv = E.xq;
    } else {
        attnres(E.Bbuf, nB + 1, Y.attn_query, 1, E.a);
        const Gate* g3[3] = {&Y.gates[0], &Y.gates[1], &Y.gates[2]};
        const float* x3[3] = {E.a, E.a, E.a};
        const float* m3[3] = {msgs, msgs + H, msgs + 2 * H};
        float* y3[3] = {E.t[0], E.t[1], E.t[2]};
        const float* nw3[3] = {Y.attn_norm, Y.attn_norm, Y.attn_norm};
        float* yn3[3] = {E.xq, E.xk, E.xv};
        message_gates(3, g3, x3, m3, y3, nw3, yn3);
        xq = E.xq; xk = E.xk; xv = E.xv;
    }
    attention(Y, S, xq, xk, xv);
    int ns;
    const float* r;
    if (l % BLOCK == 0) {   // block boundary: the residual restarts, B gains h (already at Bbuf[nB])
        k_attn_tail<<<H / 256, 256>>>(E.attn_out, S.ahist, Y.attn_conv, E.h, E.att2, E.Bbuf + (size_t) (nB + 1) * H, 1, E.d_pos);
        r = E.att2;
        ns = nB + 2;
    } else {
        k_attn_tail<<<H / 256, 256>>>(E.attn_out, S.ahist, Y.attn_conv, E.h, E.r, stack_h, 0, E.d_pos);
        r = E.r;
        ns = nB + 1;
    }
    if (!msgs) {
        attnres(E.Bbuf, ns, Y.mlp_query, 1, nullptr, Y.mlp_norm, nullptr, nullptr, E.mn);
    } else {
        attnres(E.Bbuf, ns, Y.mlp_query, 1, E.m);
        const Gate* g1[1] = {&Y.gates[3]};
        const float* x1[1] = {E.m};
        const float* m1[1] = {msgs + 3 * H};
        float* y1[1] = {E.t[3]};
        const float* nw1[1] = {Y.mlp_norm};
        float* yn1[1] = {E.mn};
        message_gates(1, g1, x1, m1, y1, nw1, yn1);
    }
    moe(Y, E.mn, r, S.mhist, Y.mlp_conv, E.out);
    if (l % BLOCK == 0) copy(stack_h, E.h, H);   // B gets the layer's input (the caller counts it)
}

// token tok's embedding row (host table, BF16 or Q8_0) as fp32
void embed_row(int tok, float* out) {
    if (!E.embed_q8.empty()) {
        const uint8_t* b = E.embed_q8.data() + (size_t) tok * (H / 32) * 34;
        for (int j = 0; j < H / 32; ++j, b += 34) {
            uint16_t hb;
            std::memcpy(&hb, b, 2);
            const float d = __half2float(*reinterpret_cast<const __half*>(&hb));
            for (int i = 0; i < 32; ++i) out[32 * j + i] = d * (float) (int8_t) b[2 + i];
        }
        return;
    }
    for (int i = 0; i < H; ++i) {
        uint32_t u = (uint32_t) E.embed[(size_t) tok * H + i] << 16;
        std::memcpy(&out[i], &u, 4);
    }
}

// the token's embedding row (host table) into E.emb_in
void upload_embed(int tok) {
    vec e(H);
    embed_row(tok, e.data());
    CK(cudaMemcpyAsync(E.emb_in, e.data(), H * 4, cudaMemcpyHostToDevice));
}

// one token: embedding in E.emb_in, position in E.d_pos; writes E.logits.  Device work only (captured in a graph):
// the host-side pointer swaps are the same on every call (an even number), so a replay sees the same buffers.
void forward_device() {
    g_dec_app = 0;
    rms(E.emb_in, E.embed_norm, E.h);
    int nB = 0;
    auto run = [&](int l, int slot, const float* msgs) {
        g_slot = slot;
        layer_forward(l, E.S[slot], nB, msgs);
        if (l % BLOCK == 0) ++nB;
        std::swap(E.h, E.out);
    };
    for (int l = 0; l < LOOP_BEGIN; ++l) run(l, l, nullptr);
    const int npre = nB;
    copy(E.pre_h, E.h, H);
    copy(E.preB, E.Bbuf, (size_t) npre * H);
    for (int l = LOOP_BEGIN; l < LOOP_END; ++l) run(l, l, nullptr);
    for (int p = 1; p < PASSES; ++p) {
        const int nprev = nB - (npre + 1);   // previous pass' block states after its first one, then its output h
        copy(E.prev_stack, E.Bbuf + (size_t) (npre + 1) * H, (size_t) nprev * H);
        copy(E.prev_stack + (size_t) nprev * H, E.h, H);
        copy(E.h, E.pre_h, H);
        copy(E.Bbuf, E.preB, (size_t) npre * H);
        nB = npre;
        for (int i = 0; i < NLOOP; ++i) {
            attnres(E.prev_stack, nprev + 1, E.msg_queries + (size_t) i * 4 * H, 4, E.msgs);
            run(LOOP_BEGIN + i, NL + (p - 1) * NLOOP + i, E.msgs);
        }
    }
    for (int l = LOOP_END; l < NL; ++l) run(l, l, nullptr);
    copy(E.Bbuf + (size_t) nB * H, E.h, H);
    attnres(E.Bbuf, nB + 1, E.final_query, 1, nullptr, E.norm, nullptr, nullptr, E.mn);
    gemv(E.unembed, E.mn, E.logits, VOCAB);
    k_softcap<<<(VOCAB + 255) / 256, 256>>>(E.logits, VOCAB, LOGIT_CAP);
    if (g_dec_pending) { CK(cudaStreamWaitEvent(cudaStreamPerThread, g_dec_join, 0)); g_dec_pending = false; }
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill: the prompt in chunks of up to TP tokens.  Same math as forward_device, row-batched: dense products as
// cuBLAS SGEMM (full FP32) on an fp32 copy of the BF16 matrix made per use, experts as kb_moe_gemm (weights decoded on
// the fly), attention tiled (kb_attn).  It fills the same KV / conv state as the one-token path, which then decodes.

struct Prefill {
    int TP = 512;   // chunk capacity
    bool tc = false;   // --tc 1: fp16 tensor-core products (NVFP4 experts, attention proj / wo)
    cublasHandle_t cb = nullptr, cbh = nullptr;   // fp32 (pedantic) / fp16 tensor-op handles
    half *wh = nullptr, *xh = nullptr, *qh = nullptr;   // fp16 weight / activation / query scratch
    float* wf = nullptr;   // fp32 copy of one dense matrix (or a slice of the unembedding)
    size_t wf_cap = 0;
    float *x0, *h, *out, *B, *preB, *pre_h, *prev, *msgs, *a, *xq, *xk, *xv, *P, *kc, *vc, *rel, *att, *attn_out, *att2,
          *r, *m, *mn, *moe, *y2, *cat, *gdown, *gfg, *rl, *wts, *gu, *act, *ey, *lg;
    int *ids, *rows, *slot, *tile_e, *tile_r0, *ntiles, *eoff, *ecnt;
    int np_max = 0;
    // offload (expert_parts): copy stream, two staging halves with their batch tables (pinned host / device)
    cudaStream_t cs = nullptr;
    cudaEvent_t ev_free[2], ev_ready[2];
    uint8_t* stg[2];
    size_t stg_half = 192ull << 20;
    unsigned stg_turn = 0;
    const uint8_t** ptab[2];
    const uint8_t** dtab[2];
    std::vector<int> nonres[NL];   // per layer, the experts not in the pools (refreshed at each prefill call)
    size_t fbytes = 0;
};
Prefill PF;

// Elastic prompt buffers (--kv-grow with offload): the chunk-sized buffers over 8 MiB and the staging halves live in
// growable ranges, mapped while a prompt is read (memory taken from the expert pools) and unmapped after.
half* g_dq[2] = {nullptr, nullptr};   // decoded expert weights (fp16), two in turn
int g_dq_min = 1024;                  // INK2_DQ_MIN: prompt chunks from this size take the decode-once path (0: never)
bool g_pf_routed = false;   // this chunk runs through the decode cache (short chunks): no streaming
int g_pf_routed_max = 64;   // INK2_ROUTED_MAX: chunks of up to this many tokens fetch only the experts they route to
bool g_pf_elastic = false;
std::vector<VRange> g_pfr;
std::vector<int> g_pfr_whole;    // 1: parts at offsets of the full --chunk (the region, the fp16 input thirds), mapped
                                 // whole; 2: the staging halves (whole, and not at all for a routed short prompt)
int g_pf_mapped_T = 0;
void* pf_alloc(size_t bytes, int whole = 0) {
    if (!g_pf_elastic || bytes < ((size_t) 8 << 20)) return dalloc(bytes);
    g_pfr.emplace_back();
    g_pfr.back().init(bytes);
    g_pfr_whole.push_back(whole);
    return g_pfr.back().ptr();
}
float* pf_floats(size_t n) { return (float*) pf_alloc(n * 4); }

// the shared region's phases laid out for a chunk of T tokens (P / rel, gu / ey / act, lg)
float* g_region = nullptr;
int g_cur_T = 0;
void pf_layout(int T) {
    g_cur_T = T;
    PF.P = g_region; PF.rel = g_region + (size_t) T * PF.np_max;
    PF.gu = g_region; PF.ey = g_region; PF.act = g_region + (size_t) T * TOPK * 2 * FF;
    PF.lg = g_region;
}

void alloc_prefill() {
    g_pf_elastic = E.grow && E.xhost;
    g_pfr.reserve(64);
    const size_t T = PF.TP, TH = T * H;
    CK(cublasCreate(&PF.cb) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown);
    cublasSetStream(PF.cb, cudaStreamPerThread);
    cublasSetMathMode(PF.cb, CUBLAS_PEDANTIC_MATH);   // full FP32, no TF32 / reduced-precision shortcuts
    for (int l = 0; l < NL; ++l) PF.np_max = std::max(PF.np_max, E.L[l].proj.n);
    PF.wf_cap = std::max((size_t) PF.np_max * H, (size_t) 8192 * H);
    PF.wf = dfloats(PF.wf_cap);
    PF.x0 = pf_floats(TH); PF.h = pf_floats(TH); PF.out = pf_floats(TH); PF.B = pf_floats(8 * TH); PF.preB = pf_floats(8 * TH);
    PF.pre_h = pf_floats(TH); PF.prev = pf_floats(8 * TH); PF.msgs = pf_floats(4 * TH); PF.a = pf_floats(3 * TH);
    PF.xq = pf_floats(TH); PF.xk = pf_floats(TH); PF.xv = pf_floats(TH);
    PF.kc = pf_floats(T * KVD); PF.vc = pf_floats(T * KVD); PF.att = pf_floats(((T + 63) / 64 * 64) * QD);
    PF.attn_out = pf_floats(TH); PF.att2 = pf_floats(TH); PF.r = pf_floats(TH); PF.m = pf_floats(TH); PF.mn = pf_floats(TH);
    PF.moe = pf_floats(TH); PF.y2 = pf_floats(TH); PF.cat = pf_floats(2 * TH); PF.gdown = pf_floats(T * GATE_RANK);
    PF.gfg = pf_floats(2 * TH); PF.rl = pf_floats(T * NE); PF.wts = pf_floats(T * TOPK);
    // One region, phases with disjoint lifetimes (saves ~217 KB per chunk token, 445 MB at chunk 2048):
    //   attention: P [T][np_max], rel [T][NQ][EXTENT];  MoE: gu [T*TOPK][2FF] then ey (same rows, H = 2FF) after the
    //   SwiGLU has read gu, act [T*TOPK][FF] after gu;  after the last layer: lg [T][8192] (all-logits dumps only)
    static_assert(2 * FF == H, "ey aliases gu");
    const size_t att_n = T * ((size_t) PF.np_max + NQ * EXTENT), moe_n = T * TOPK * (size_t) 3 * FF;
    // (sub-buffers at offsets of the chunk being read, pf_layout: the region maps in proportion to the chunk)
    g_region = (float*) pf_alloc(std::max(std::max(att_n, moe_n), T * (size_t) 8192) * 4);
    pf_layout((int) T);
    auto ints = [](size_t n) { return (int*) dalloc(n * 4); };
    PF.ids = ints(T * TOPK); PF.rows = ints(T * TOPK); PF.slot = ints(T * TOPK);
    PF.tile_e = ints(T * TOPK / FM + NE); PF.tile_r0 = ints(T * TOPK / FM + NE); PF.ntiles = ints(1);
    PF.eoff = ints(NE); PF.ecnt = ints(NE);
    // the fp32 prompt attention needs 83 KB of shared memory: more than Turing's 64 KB (--tc 1 does not use it)
    if (!PF.tc && cudaFuncSetAttribute(kb_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) FA_SMEM) != cudaSuccess) {
        std::fprintf(stderr, "the fp32 prompt attention needs %zu bytes of shared memory, more than this card has: use --tc 1\n", FA_SMEM);
        std::exit(1);
    }
    if (E.xhost) {
        CK(cudaStreamCreateWithFlags(&PF.cs, cudaStreamNonBlocking));
        for (int h = 0; h < 2; ++h) {
            CK(cudaEventCreateWithFlags(&PF.ev_free[h], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&PF.ev_ready[h], cudaEventDisableTiming));
            PF.stg[h] = (uint8_t*) pf_alloc(PF.stg_half, 2);
            CK(cudaHostAlloc((void**) &PF.ptab[h], NE * sizeof(void*), cudaHostAllocDefault));
            PF.dtab[h] = (const uint8_t**) dalloc(NE * sizeof(void*));
        }
    }
    if (PF.tc) {
        CK(cublasCreate(&PF.cbh) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown);
        cublasSetStream(PF.cbh, cudaStreamPerThread);
        PF.wh = (half*) dalloc((size_t) PF.np_max * H * 2);
        for (auto& d : g_dq) d = (half*) dalloc((size_t) 2 * FF * H * 2);   // one expert part (w13: 2FF x H)
        PF.xh = (half*) pf_alloc(3 * T * QD * 2);
        PF.qh = (half*) pf_alloc(((T + 63) / 64 * 64) * QD * 2);
    }
    // cuBLAS loads its kernels and workspace at the first call of each shape class: run them once now, before the
    // expert pools take the free memory (a first prompt after that would find only the reserve)
    {
        const int Tw[3] = {16, 256, PF.TP}, Nw[3] = {NE, H, QD};   // the buffers hold QD x QD and TP x QD
        const size_t big = (size_t) PF.TP * QD;
        float *a = nullptr, *w = nullptr, *c = nullptr;
        CK(cudaMalloc(&a, big * 4)); CK(cudaMalloc(&w, (size_t) QD * QD * 4)); CK(cudaMalloc(&c, big * 4));
        const float one = 1.0f, zero = 0.0f;
        for (int ti : Tw)
            for (int nw : Nw) {
                cublasSgemm(PF.cb, CUBLAS_OP_T, CUBLAS_OP_N, nw, ti, H, &one, w, H, a, H, &zero, c, nw);
                if (PF.tc)
                    cublasGemmEx(PF.cbh, CUBLAS_OP_T, CUBLAS_OP_N, nw, ti, H, &one, w, CUDA_R_16F, H, a, CUDA_R_16F, H, &zero, c,
                                 CUDA_R_32F, nw, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            }
        CK(cudaDeviceSynchronize());
        CK(cudaFree(a)); CK(cudaFree(w)); CK(cudaFree(c));
    }
}

// the prompt buffers for chunks of T tokens (T <= --chunk): mapped, from the pools if VRAM is short
void pf_map(int T) {
    if (g_pfr.empty() || T <= g_pf_mapped_T) return;
    CK(cudaDeviceSynchronize());
    size_t add = 0;
    std::vector<size_t> want(g_pfr.size());
    for (size_t i = 0; i < g_pfr.size(); ++i) {
        const VRange& r = g_pfr[i];
        want[i] = g_pfr_whole[i] == 2 ? (T <= g_pf_routed_max ? 0 : r.reserved)
                : g_pfr_whole[i] ? r.reserved : r.need(std::min(r.reserved, (size_t) ((double) r.reserved * (T + 64) / PF.TP)));
        if (want[i] > r.mapped) add += want[i] - r.mapped;
    }
    const size_t keep = vram_keep(), fr = vram_free();
    if (fr < add + keep) pools_release((double) (add + keep - fr), "the prompt buffers");
    for (size_t i = 0; i < g_pfr.size(); ++i)
        if (!g_pfr[i].map_to(want[i])) { std::fprintf(stderr, "prompt buffers: out of VRAM\n"); std::exit(1); }
    g_pf_mapped_T = T;
}
// the largest chunk (<= T, halving down to 256) whose prompt buffers and K/V growth leave the pools 2 x FETCH_MAX
// slots: long contexts on a small card read the prompt in smaller chunks instead of stopping
int pf_fit(int T, int pos0) {
    if (g_pfr.empty() || !E.xhost) return T;
    for (;; T /= 2) {
        size_t want = 0, have = 0;
        for (size_t i = 0; i < g_pfr.size(); ++i) {
            const VRange& r = g_pfr[i];
            want += g_pfr_whole[i] == 2 ? (T <= g_pf_routed_max ? 0 : r.reserved)
                  : g_pfr_whole[i] ? r.reserved : r.need(std::min(r.reserved, (size_t) ((double) r.reserved * (T + 64) / PF.TP)));
            have += r.mapped;
        }
        const int nc = std::min(E.ctx, (pos0 + T + E.grow_step - 1) / E.grow_step * E.grow_step);
        const double kv_add = E.grow && nc > E.kv_cells ? (double) (kv_mapped_for(nc) - kv_mapped_for(E.kv_cells)) : 0.0;
        const double deficit = kv_add + (double) want - (double) have + (double) vram_keep() - (double) vram_free() + 8.0 * g_vmm.gran;
        const double slots = (double) E.nslot_nv + E.nslot_mx - (deficit > 0 ? deficit / (double) ex_bytes(0) : 0.0);
        if (slots >= 2 * FETCH_MAX || T <= 256) return T;
    }
}
// after a prompt: the buffers' memory back to the pools
void pf_unmap() {
    if (g_pfr.empty() || !g_pf_mapped_T) return;
    CK(cudaDeviceSynchronize());
    for (auto& r : g_pfr) r.unmap_to(0);
    g_pf_mapped_T = 0;
    pools_fill();
}

// tensor-core dense product: Y[t][0..n) = fp16(X[t]) . fp16(W rows [first, first + n))^T, fp32 accumulation.
// wcvt: convert W (whole matrix) into PF.wh first; xslot: which third of PF.xh holds X's fp16 copy
void hgemm(const DMat& W, bool wcvt, int first, int n, const float* X, int ldx, int K, int xslot, float* Y, int ldy, int T) {
    if (W.q && T <= MV) {   // a chunk of up to 8 tokens (MTP verify): the Q8 rows read once, no fp16 copy of the matrix
        k_gemv_q8_multi<<<(n + 31) / 32, 256>>>(W.q + (size_t) first * K, W.qs + (size_t) first * (K / 32), X, ldx, Y, ldy, T, n, K);
        return;
    }
    if (wcvt) {
        const size_t nw = (size_t) W.n * W.k;
        if (W.q) kb_q8_to_f16<<<(unsigned) ((nw / 8 + 255) / 256), 256>>>(W.q, W.qs, PF.wh, nw / 8);
        else kb_bf16_to_f16<<<(unsigned) ((nw + 255) / 256), 256>>>(W.d, PF.wh, nw);
    }
    half* xh = PF.xh + (size_t) xslot * g_cur_T * QD;   // (thirds of the chunk being read)
    const size_t nx = (size_t) T * K;
    kb_f32_to_f16<<<(unsigned) ((nx + 255) / 256), 256>>>(X, ldx, K, xh, nx);
    const float one = 1.0f, zero = 0.0f;
    if (cublasGemmEx(PF.cbh, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &one, PF.wh + (size_t) first * K, CUDA_R_16F, K, xh,
                     CUDA_R_16F, K, &zero, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasGemmEx failed (%d x %d x %d)\n", n, T, K);
        std::exit(1);
    }
}

// rows [first, first + n) of W (BF16) into PF.wf as fp32
const float* wf32(const DMat& W, int first = 0, int n = -1) {
    if (n < 0) n = W.n - first;
    const size_t n8 = (size_t) n * W.k / 8;
    if ((size_t) n * W.k > PF.wf_cap) { std::fprintf(stderr, "wf32: %d x %d over the scratch\n", n, W.k); std::exit(1); }
    if (W.q) kb_q8_to_f32<<<(unsigned) ((n8 + 255) / 256), 256>>>(W.q + (size_t) first * W.k, W.qs + (size_t) first * (W.k / 32), PF.wf, n8);
    else kb_bf16_to_f32<<<(unsigned) ((n8 + 255) / 256), 256>>>(W.d + (size_t) first * W.k, PF.wf, n8);
    return PF.wf;
}
// Y[t][0..n) (row stride ldy) = X[t] (row stride ldx) . Wf[0..n)^T, Wf fp32 [n][K]
void sgemm(const float* Wf, int n, int K, const float* X, int ldx, float* Y, int ldy, int T) {
    const float one = 1.0f, zero = 0.0f;
    if (cublasSgemm(PF.cb, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &one, Wf, K, X, ldx, &zero, Y, ldy) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSgemm failed (%d x %d x %d)\n", n, T, K);
        std::exit(1);
    }
}
void gemm(const DMat& W, const float* X, int ldx, float* Y, int ldy, int T) { sgemm(wf32(W), W.n, W.k, X, ldx, Y, ldy, T); }

void message_gate_b(const Gate& g, const float* x, int sx, const float* m, int sm, const float* nw, float* yn, int T) {
    kb_rms2<<<dim3(2, T), 512>>>(x, sx, m, sm, PF.cat);
    gemm(g.down, PF.cat, 2 * H, PF.gdown, GATE_RANK, T);
    gemm(g.fg, PF.gdown, GATE_RANK, PF.gfg, 2 * H, T);
    kb_gate_mix_rms<<<T, 512>>>(x, sx, m, sm, PF.gfg, nw, yn);
}

// MTP verify: the conv inputs of the chunk are kept per history (k_hist_fix rebuilds the history of a partial accept)
bool g_verify = false;
std::map<const float*, float*> g_hist_save;
void conv_b(const float* x, int sx, float* hist, const float* w, const float* dw, float* y, int sy, int C, int T, int pos0) {
    if (g_verify) {
        auto it = g_hist_save.find(hist);
        if (it != g_hist_save.end())
            k_conv_save<<<(unsigned) (((size_t) T * C + 255) / 256), 256>>>(x, sx, it->second, C, T);
    }
    kb_conv<<<dim3(C / 256, T), 256>>>(x, sx, hist, w, dw, y, sy, C, pos0);
    kb_hist_store<<<dim3(C / 256, std::min(3, T)), 256>>>(x, sx, hist, C, T, pos0);
}

void attention_b(const Layer& Y, Slot& S, const float* xq, int sq, const float* xk, int sk, const float* xv, int sv,
                 int T, int pos0) {
    const int np = Y.proj.n;
    float* P = PF.P;
    if (PF.tc) {
        if (xq == xk && xk == xv && sq == sk && sk == sv) {
            hgemm(Y.proj, true, 0, np, xq, sq, H, 0, P, np, T);
        } else {
            hgemm(Y.proj, true, 0, OFF_K, xq, sq, H, 0, P, np, T);
            hgemm(Y.proj, false, Y.off_k, Y.n_k, xk, sk, H, 1, P + Y.off_k, np, T);
            hgemm(Y.proj, false, Y.off_v, Y.n_v, xv, sv, H, 2, P + Y.off_v, np, T);
        }
    }
    const float* W = PF.tc ? nullptr : wf32(Y.proj);
    if (PF.tc) {
    } else if (xq == xk && xk == xv && sq == sk && sk == sv) {
        sgemm(W, np, H, xq, sq, P, np, T);
    } else {
        sgemm(W, OFF_K, H, xq, sq, P, np, T);
        sgemm(W + (size_t) Y.off_k * H, Y.n_k, H, xk, sk, P + Y.off_k, np, T);
        sgemm(W + (size_t) Y.off_v * H, Y.n_v, H, xv, sv, P + Y.off_v, np, T);
    }
    kb_head_clip<<<dim3(NQ, T), 128>>>(P, np, Y.q_scale, HD, QK_CLIP, Y.full, pos0);
    if (E.kvm == 3) kb_wht<<<dim3(NQ, T), 128>>>(P, np);
    float *k = P + Y.off_k, *v = P + Y.off_v;
    if (Y.full) {
        conv_b(k, np, S.khist, Y.k_static, k + KVD, PF.kc, KVD, KVD, T, pos0);
        conv_b(v, np, S.vhist, Y.v_static, v + KVD, PF.vc, KVD, KVD, T, pos0);
    } else {
        conv_b(k, np, S.khist, Y.k_conv, nullptr, PF.kc, KVD, KVD, T, pos0);
        conv_b(v, np, S.vhist, Y.v_conv, nullptr, PF.vc, KVD, KVD, T, pos0);
    }
    kb_kv_store<<<dim3(NKV, T), HD>>>(PF.kc, PF.vc, S.K, S.V, S.ksc, S.vsc, S.cap, pos0);
    kb_head_clip<<<dim3(NQ, T), 128>>>(P + OFF_REL, np, Y.rel_scale, DREL, REL_CLIP, Y.full, pos0);
    kb_rel_logits<<<dim3(NQ, T), 256>>>(P + OFF_REL, np, Y.rel_proj, PF.rel);
    if (PF.tc) {
        const size_t nqh = (size_t) T * QD;
        kb_f32_to_f16<<<(unsigned) ((nqh + 255) / 256), 256>>>(P, np, QD, PF.qh, nqh);
        kb_attn_tc<<<dim3((T + AQ - 1) / AQ, NKV), 128 * NREP>>>(PF.qh, P, np, S.K, S.V, S.ksc, S.vsc, PF.rel, S.cap, S.full ? 0 : WINDOW, pos0,
                                                         T, PF.att);
    } else {
        kb_attn<<<dim3((T + FQ - 1) / FQ, NQ), 256, FA_SMEM>>>(P, np, S.K, S.V, S.ksc, S.vsc, PF.rel, S.cap, S.full ? 0 : WINDOW, pos0,
                                                             T, PF.att);
    }
    if (kv_rot_v(E.kvm)) kb_wht<<<dim3(NQ, T), 128>>>(PF.att, QD);
    kb_mul_sigmoid<<<dim3(QD / 256, T), 256>>>(PF.att, P + OFF_GATE, np);
    if (PF.tc) hgemm(Y.wo, true, 0, H, PF.att, QD, QD, 0, PF.attn_out, H, T);
    else gemm(Y.wo, PF.att, QD, PF.attn_out, H, T);
}

// Offloaded prefill: the expert cache is frozen during a prefill (a chunk touches nearly every expert, an LRU would
// only thrash).  A grouped product runs once over the resident experts (the decode tables, null = not resident:
// tiles skipped), then over the others in batches streamed into two staging halves on PF.cs (copy of batch b + 1
// overlapping the product of batch b).  Each row still sees exactly its expert's weights: bit-identical.
// part B (w2) is streamed after the SwiGLU, separately from part A (w13): each byte crosses PCIe once per layer call.
// CBA: streams every non-resident expert of the layer, used by the chunk or not (a 1024-token chunk uses 119-127 of 128).
template <class F> void expert_parts(int l, bool partB, F launch) {
    const Experts& X = E.L[l].ex;
    launch(partB ? X.tB : X.tA);
    if (!E.xhost || g_pf_routed) return;
    const auto& nonres = PF.nonres[l];
    const size_t eb = ex_bytes(l), part = partB ? eb - ex_partA(l) : ex_partA(l), off = partB ? ex_partA(l) : 0;
    const int per = (int) (PF.stg_half / part);
    for (int b0 = 0, bi = 0; b0 < (int) nonres.size(); b0 += per, ++bi) {
        const int h = PF.stg_turn++ & 1, n = std::min(per, (int) nonres.size() - b0);
        CK(cudaStreamWaitEvent(PF.cs, PF.ev_free[h], 0));
        CK(cudaEventSynchronize(PF.ev_ready[h]));   // the previous table copy out of ptab[h] is done
        const uint8_t** tab = PF.ptab[h];
        std::fill(tab, tab + NE, nullptr);
        for (int j = 0; j < n; ++j) {
            const int e = nonres[b0 + j];
            uint8_t* d = PF.stg[h] + (size_t) j * part;
            CK(cudaMemcpyAsync(d, E.xhost + ex_layer_off(l) + (size_t) e * eb + off, part, cudaMemcpyHostToDevice, PF.cs));
            tab[e] = d;
        }
        // the batch table rides on the copy stream too (pinned slot per half, reused once ev_free[h] has fired)
        CK(cudaMemcpyAsync(PF.dtab[h], PF.ptab[h], NE * sizeof(void*), cudaMemcpyHostToDevice, PF.cs));
        CK(cudaEventRecord(PF.ev_ready[h], PF.cs));
        CK(cudaStreamWaitEvent(cudaStreamPerThread, PF.ev_ready[h], 0));
        launch(PF.dtab[h]);
        CK(cudaEventRecord(PF.ev_free[h], cudaStreamPerThread));
        PF.fbytes += (size_t) n * part;
    }
}

// GGUF grouped products (prompt), runtime type -> kernel
template <bool PERM>
void gg_gemm(int t, bool tc, int maxt, const void* A, int lda, const int* rows, const uint8_t* const* tab, size_t rb,
             float* C, int N, int K) {
#define GGA PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K
    if (tc) {
        const dim3 g((unsigned) maxt, (unsigned) (N / TN));
        const half* a = (const half*) A;
        switch (t) {
        case GG_IQ3_S: kb_moe_gemm_tc_gg<GG_IQ3_S, PERM><<<g, 256>>>(a, lda, tab, rb, GGA); break;
        case GG_IQ4_NL: kb_moe_gemm_tc_gg<GG_IQ4_NL, PERM><<<g, 256>>>(a, lda, tab, rb, GGA); break;
        default: kb_moe_gemm_tc_gg<GG_Q8_0, PERM><<<g, 256>>>(a, lda, tab, rb, GGA);
        }
    } else {
        const dim3 g((unsigned) maxt, (unsigned) (N / FN));
        const float* a = (const float*) A;
        switch (t) {
        case GG_IQ3_S: kb_moe_gemm_gg<GG_IQ3_S, PERM><<<g, 256>>>(a, lda, rows, tab, rb, GGA); break;
        case GG_IQ4_NL: kb_moe_gemm_gg<GG_IQ4_NL, PERM><<<g, 256>>>(a, lda, rows, tab, rb, GGA); break;
        default: kb_moe_gemm_gg<GG_Q8_0, PERM><<<g, 256>>>(a, lda, rows, tab, rb, GGA);
        }
    }
#undef GGA
}

void moe_b(const Layer& Y, const float* x, float* out, int T, int pos0) {
    gemm(Y.router, x, H, PF.rl, NE, T);
    if (E.rec_mode == 2)
        kb_router_forced<<<T, 32>>>(PF.rl, E.rec + ((size_t) g_slot * E.ctx + pos0) * TOPK, PF.ids, PF.wts);
    else
        k_router<<<T, NE>>>(PF.rl, Y.router_bias, PF.ids, PF.wts);
    kb_moe_plan<<<1, 1024>>>(PF.ids, T, PF.rows, PF.slot, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt);
    const Experts& X = Y.ex;
    const int l = int(&Y - E.L), nr = T * TOPK, maxt = nr / FM + NE;
    // A short chunk uses few experts: it makes the ones it routes to resident through the decode cache (groups of 8
    // tokens: at most 64 misses per plan) instead of streaming every expert the pools do not hold.
    g_pf_routed = E.xhost && T <= g_pf_routed_max;
    if (g_pf_routed)
        for (int t0 = 0; t0 < T; t0 += 8) expert_cache(l, PF.ids + (size_t) t0 * TOPK, std::min(8, T - t0) * TOPK);
    if (X.gg13 && T <= 8) {   // a chunk of up to 8 tokens (MTP verify): each used expert decoded once (routed cache)
        auto gv = [&](int t, bool perm, const float* A, int lda, const int* rws, const uint8_t* const* tb, size_t rb, float* C, int N, int K) {
            const dim3 g((unsigned) ((N + 31) / 32), (unsigned) maxt);
            if (perm) {
                if (t == GG_IQ3_S) kb_moe_gemv_gg<GG_IQ3_S, true><<<g, 256>>>(A, lda, rws, tb, rb, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K);
                else if (t == GG_IQ4_NL) kb_moe_gemv_gg<GG_IQ4_NL, true><<<g, 256>>>(A, lda, rws, tb, rb, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K);
                else kb_moe_gemv_gg<GG_Q8_0, true><<<g, 256>>>(A, lda, rws, tb, rb, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K);
            } else {
                if (t == GG_IQ3_S) kb_moe_gemv_gg<GG_IQ3_S, false><<<g, 256>>>(A, lda, rws, tb, rb, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K);
                else if (t == GG_IQ4_NL) kb_moe_gemv_gg<GG_IQ4_NL, false><<<g, 256>>>(A, lda, rws, tb, rb, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K);
                else kb_moe_gemv_gg<GG_Q8_0, false><<<g, 256>>>(A, lda, rws, tb, rb, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, C, N, K);
            }
        };
        expert_parts(l, false, [&](const uint8_t* const* t) { gv(X.gg13, true, x, H, PF.rows, t, X.rb13, PF.gu, 2 * FF, H); });
        kb_swiglu<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, PF.act, nr);
        expert_parts(l, true, [&](const uint8_t* const* t) { gv(X.gg2, false, PF.act, FF, nullptr, t, X.rb2, PF.ey, H, FF); });
    } else if (X.gg13 && PF.tc && g_dq_min > 0 && T >= g_dq_min) {
        // decode-once: per expert with rows, its part decoded to fp16 and one cuBLAS GEMM over its rows (the row
        // counts are read back).  The non-resident experts' copies into the two staging halves are queued before the
        // read-back (they do not depend on it), so they overlap the layer's attention.
        half* ah = reinterpret_cast<half*>(PF.act);
        kb_gather_h<<<nr, 256>>>(x, PF.rows, ah);
        const auto& nr_list = PF.nonres[l];
        const bool stream = E.xhost && !g_pf_routed && !nr_list.empty();
        const size_t eb = ex_bytes(l), partA = ex_partA(l), partB = eb - partA;
        const int perA = (int) (PF.stg_half / partA), perB = (int) (PF.stg_half / partB);
        const bool early = stream && (int) nr_list.size() <= perA && (int) nr_list.size() <= perB;
        int hA = -1, hB = -1;
        auto stage = [&](bool pb, int b0, int n) {   // queue the copies of nr_list[b0 .. b0 + n) into a half
            const int h = PF.stg_turn++ & 1;
            const size_t part = pb ? partB : partA, poff = pb ? partA : 0;
            CK(cudaStreamWaitEvent(PF.cs, PF.ev_free[h], 0));
            CK(cudaEventSynchronize(PF.ev_ready[h]));
            const uint8_t** tab = PF.ptab[h];
            std::fill(tab, tab + NE, nullptr);
            for (int j = 0; j < n; ++j) {
                const int e = nr_list[b0 + j];
                uint8_t* d = PF.stg[h] + (size_t) j * part;
                CK(cudaMemcpyAsync(d, E.xhost + ex_layer_off(l) + (size_t) e * eb + poff, part, cudaMemcpyHostToDevice, PF.cs));
                tab[e] = d;
            }
            CK(cudaMemcpyAsync(PF.dtab[h], PF.ptab[h], NE * sizeof(void*), cudaMemcpyHostToDevice, PF.cs));
            CK(cudaEventRecord(PF.ev_ready[h], PF.cs));
            PF.fbytes += (size_t) n * part;
            return h;
        };
        if (early) { hA = stage(false, 0, (int) nr_list.size()); hB = stage(true, 0, (int) nr_list.size()); }
        std::vector<int> cnt(NE), off(NE);
        CK(cudaMemcpyAsync(cnt.data(), PF.ecnt, NE * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpyAsync(off.data(), PF.eoff, NE * 4, cudaMemcpyDeviceToHost));
        CK(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<bool> nonres(NE, false);
        if (E.xhost) for (int e : nr_list) nonres[e] = true;
        int turn = 0;
        auto run_parts = [&](bool pb, int hpre) {
            const int t = pb ? X.gg2 : X.gg13, N = pb ? H : 2 * FF, K = pb ? FF : H;
            const size_t rb = pb ? X.rb2 : X.rb13;
            float* C = pb ? PF.ey : PF.gu;
            auto one = [&](const uint8_t* const* tab, int e) {
                if (!cnt[e]) return;
                half* wbuf = g_dq[turn++ & 1];
                const unsigned gb = 2 * 34 * 4;
                if (!pb) {
                    if (t == GG_IQ3_S) kb_dequant_gg<GG_IQ3_S, true><<<gb, 256>>>(tab, e, rb, wbuf, N, K);
                    else if (t == GG_IQ4_NL) kb_dequant_gg<GG_IQ4_NL, true><<<gb, 256>>>(tab, e, rb, wbuf, N, K);
                    else kb_dequant_gg<GG_Q8_0, true><<<gb, 256>>>(tab, e, rb, wbuf, N, K);
                } else {
                    if (t == GG_IQ3_S) kb_dequant_gg<GG_IQ3_S, false><<<gb, 256>>>(tab, e, rb, wbuf, N, K);
                    else if (t == GG_IQ4_NL) kb_dequant_gg<GG_IQ4_NL, false><<<gb, 256>>>(tab, e, rb, wbuf, N, K);
                    else kb_dequant_gg<GG_Q8_0, false><<<gb, 256>>>(tab, e, rb, wbuf, N, K);
                }
                const float one_f = 1.0f, zero = 0.0f;
                if (cublasGemmEx(PF.cbh, CUBLAS_OP_T, CUBLAS_OP_N, N, cnt[e], K, &one_f, wbuf, CUDA_R_16F, K,
                                 ah + (size_t) off[e] * K, CUDA_R_16F, K, &zero, C + (size_t) off[e] * N, CUDA_R_32F, N,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "cublasGemmEx (expert) failed\n");
                    std::exit(1);
                }
            };
            const auto& tabd = pb ? X.tB : X.tA;
            for (int e = 0; e < NE; ++e) if (!nonres[e]) one(tabd, e);
            if (!stream) return;
            auto use = [&](int h, int b0, int n) {
                CK(cudaStreamWaitEvent(cudaStreamPerThread, PF.ev_ready[h], 0));
                for (int j = 0; j < n; ++j) one(PF.dtab[h], nr_list[b0 + j]);
                CK(cudaEventRecord(PF.ev_free[h], cudaStreamPerThread));
            };
            if (hpre >= 0) { use(hpre, 0, (int) nr_list.size()); return; }
            const int per = pb ? perB : perA;
            for (int b0 = 0; b0 < (int) nr_list.size(); b0 += per) {
                const int n = std::min(per, (int) nr_list.size() - b0);
                use(stage(pb, b0, n), b0, n);
            }
        };
        run_parts(false, hA);
        kb_swiglu_h<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, ah, nr);
        run_parts(true, hB);
    } else if (X.gg13) {   // GGUF experts: the same streaming of part A (gate_up) then part B (down)
        half* ah = reinterpret_cast<half*>(PF.act);
        const int mt = PF.tc ? nr / TM + NE : maxt;
        if (PF.tc) kb_gather_h<<<nr, 256>>>(x, PF.rows, ah);
        expert_parts(l, false, [&](const uint8_t* const* t) {
            if (PF.tc) gg_gemm<true>(X.gg13, true, mt, ah, H, nullptr, t, X.rb13, PF.gu, 2 * FF, H);
            else gg_gemm<true>(X.gg13, false, mt, x, H, PF.rows, t, X.rb13, PF.gu, 2 * FF, H);
        });
        if (PF.tc) kb_swiglu_h<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, ah, nr);
        else kb_swiglu<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, PF.act, nr);
        expert_parts(l, true, [&](const uint8_t* const* t) {
            if (PF.tc) gg_gemm<false>(X.gg2, true, mt, ah, FF, nullptr, t, X.rb2, PF.ey, H, FF);
            else gg_gemm<false>(X.gg2, false, mt, PF.act, FF, nullptr, t, X.rb2, PF.ey, H, FF);
        });
    } else if (PF.tc) {
        const int maxt_tc = nr / TM + NE;
        auto g = X.nvfp4 ? kb_moe_gemm_tc<true> : kb_moe_gemm_tc<false>;
        // fp16 A copies live in the act region: the gathered input [nr][H] halves (same bytes as act [nr][FF] floats)
        // until the SwiGLU, then the fp16 act [nr][FF]
        half* ah = reinterpret_cast<half*>(PF.act);
        kb_gather_h<<<nr, 256>>>(x, PF.rows, ah);
        expert_parts(l, false, [&](const uint8_t* const* t) {
            g<<<dim3(maxt_tc, 2 * FF / TN), 256>>>(ah, H, t, X.f13, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt,
                                                   PF.gu, 2 * FF, H);
        });
        kb_swiglu_h<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, ah, nr);
        expert_parts(l, true, [&](const uint8_t* const* t) {
            g<<<dim3(maxt_tc, H / TN), 256>>>(ah, FF, t, X.f2, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, PF.ey,
                                              H, FF);
        });
    } else {
        auto g = X.nvfp4 ? kb_moe_gemm<true> : kb_moe_gemm<false>;
        expert_parts(l, false, [&](const uint8_t* const* t) {
            g<<<dim3(maxt, 2 * FF / FN), 256>>>(x, H, PF.rows, t, X.f13, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff,
                                                PF.ecnt, PF.gu, 2 * FF, H);
        });
        kb_swiglu<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, PF.act, nr);
        expert_parts(l, true, [&](const uint8_t* const* t) {
            g<<<dim3(maxt, H / FN), 256>>>(PF.act, FF, nullptr, t, X.f2, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff,
                                           PF.ecnt, PF.ey, H, FF);
        });
    }
    kb_combine<<<dim3(H / 256, T), 256>>>(PF.ey, PF.slot, PF.wts, out);
}

void copy_b(float* dst, const float* src, size_t n) { copy(dst, src, n); }
void add_b(const float* a, const float* b, float* y, size_t n) { kb_add<<<(unsigned) ((n + 255) / 256), 256>>>(a, b, y, n); }

// layer_forward for T tokens: input PF.h, block states PF.B[0..nB) ([S][T][H]), output PF.out; msgs [T][4][H] or null
void layer_forward_b(int l, Slot& S, int nB, const float* msgs, int T, int pos0) {
    const Layer& Y = E.L[l];
    const size_t ES = (size_t) T * H;
    float* stack_h = PF.B + nB * ES;
    copy_b(stack_h, PF.h, ES);
    const float *xq, *xk, *xv;
    int sq = H, sk = H, sv = H;
    if (Y.full) {
        if (!msgs) {
            attnres(PF.B, nB + 1, Y.attn_query, 3, nullptr, Y.q_norm, Y.k_norm, Y.v_norm, PF.a, T);
            xq = PF.a; xk = PF.a + H; xv = PF.a + 2 * H;
            sq = sk = sv = 3 * H;
        } else {
            attnres(PF.B, nB + 1, Y.attn_query, 3, PF.a, nullptr, nullptr, nullptr, nullptr, T);
            message_gate_b(Y.gates[0], PF.a, 3 * H, msgs, 4 * H, Y.q_norm, PF.xq, T);
            message_gate_b(Y.gates[1], PF.a + H, 3 * H, msgs + H, 4 * H, Y.k_norm, PF.xk, T);
            message_gate_b(Y.gates[2], PF.a + 2 * H, 3 * H, msgs + 2 * H, 4 * H, Y.v_norm, PF.xv, T);
            xq = PF.xq; xk = PF.xk; xv = PF.xv;
        }
    } else if (!msgs) {
        if (l > 0) attnres(PF.B, nB + 1, Y.attn_query, 1, nullptr, Y.attn_norm, nullptr, nullptr, PF.xq, T);
        else k_rms<<<T, 512>>>(PF.h, Y.attn_norm, PF.xq, H);
        xq = xk = xv = PF.xq;
    } else {
        attnres(PF.B, nB + 1, Y.attn_query, 1, PF.a, nullptr, nullptr, nullptr, nullptr, T);
        message_gate_b(Y.gates[0], PF.a, H, msgs, 4 * H, Y.attn_norm, PF.xq, T);
        message_gate_b(Y.gates[1], PF.a, H, msgs + H, 4 * H, Y.attn_norm, PF.xk, T);
        message_gate_b(Y.gates[2], PF.a, H, msgs + 2 * H, 4 * H, Y.attn_norm, PF.xv, T);
        xq = PF.xq; xk = PF.xk; xv = PF.xv;
    }
    attention_b(Y, S, xq, sq, xk, sk, xv, sv, T, pos0);
    conv_b(PF.attn_out, H, S.ahist, Y.attn_conv, nullptr, PF.att2, H, H, T, pos0);
    int ns;
    const float* r;
    if (l % BLOCK == 0) {
        r = PF.att2;
        copy_b(PF.B + (nB + 1) * ES, PF.att2, ES);
        ns = nB + 2;
    } else {
        add_b(PF.h, PF.att2, PF.r, ES);
        r = PF.r;
        copy_b(stack_h, PF.r, ES);
        ns = nB + 1;
    }
    if (!msgs) {
        attnres(PF.B, ns, Y.mlp_query, 1, nullptr, Y.mlp_norm, nullptr, nullptr, PF.mn, T);
    } else {
        attnres(PF.B, ns, Y.mlp_query, 1, PF.m, nullptr, nullptr, nullptr, nullptr, T);
        message_gate_b(Y.gates[3], PF.m, H, msgs + 3 * H, 4 * H, Y.mlp_norm, PF.mn, T);
    }
    moe_b(Y, PF.mn, PF.moe, T, pos0);
    conv_b(PF.moe, H, S.mhist, Y.mlp_conv, nullptr, PF.y2, H, H, T, pos0);
    add_b(r, PF.y2, PF.out, ES);
    if (l % BLOCK == 0) copy_b(stack_h, PF.h, ES);
}

// tokens ids[pos0 .. pos0 + T) -> logits of the last one in E.logits (all of them into `all` [T][VOCAB] if non-null)
float* g_feat = nullptr;              // MTP: [chunk][4][H] target features of the chunk being read
int g_feat_visit[3] = {5, 23, 41};    // mtp_config target_layer_ids (the 4th, 48, is the normalized output)
float* g_vlog = nullptr;              // verify: logits [8][VOCAB_ROWS]
int* g_vids = nullptr;                // verify: greedy token of each position
void forward_prefill(const int* toks, int T, int pos0, float* all) {
    kv_fit(pos0 + T);
    pf_layout(T);   // (before the pools are read below: a growth can take slots)
    const size_t ES = (size_t) T * H;
    if (E.xhost) {   // what decode left in the pools
        std::vector<const uint8_t*> t((size_t) NL * NE);
        CK(cudaMemcpy(t.data(), E.tabA, t.size() * sizeof(void*), cudaMemcpyDeviceToHost));
        for (int l = 0; l < NL; ++l) {
            PF.nonres[l].clear();
            for (int e = 0; e < NE; ++e) if (!t[(size_t) l * NE + e]) PF.nonres[l].push_back(e);
        }
    }
    {
        vec e(ES);
        for (int t = 0; t < T; ++t) embed_row(toks[t], &e[(size_t) t * H]);
        CK(cudaMemcpyAsync(PF.x0, e.data(), ES * 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CK(cudaStreamSynchronize(cudaStreamPerThread));
    }
    k_rms<<<T, 512>>>(PF.x0, E.embed_norm, PF.h, H);
    int nB = 0;
    int visit = 0;   // MTP: the inputs of the target visits are the drafter's features
    auto run = [&](int l, int slot, const float* msgs) {
        g_slot = slot;
        if (g_feat) for (int j = 0; j < 3; ++j) if (visit == g_feat_visit[j]) k_feat_put<<<T, 256>>>(PF.h, g_feat, j);
        ++visit;
        layer_forward_b(l, E.S[slot], nB, msgs, T, pos0);
        if (l % BLOCK == 0) ++nB;
        std::swap(PF.h, PF.out);
    };
    for (int l = 0; l < LOOP_BEGIN; ++l) run(l, l, nullptr);
    const int npre = nB;
    copy_b(PF.pre_h, PF.h, ES);
    copy_b(PF.preB, PF.B, npre * ES);
    for (int l = LOOP_BEGIN; l < LOOP_END; ++l) run(l, l, nullptr);
    for (int p = 1; p < PASSES; ++p) {
        const int nprev = nB - (npre + 1);
        copy_b(PF.prev, PF.B + (npre + 1) * ES, nprev * ES);
        copy_b(PF.prev + nprev * ES, PF.h, ES);
        copy_b(PF.h, PF.pre_h, ES);
        copy_b(PF.B, PF.preB, npre * ES);
        nB = npre;
        for (int i = 0; i < NLOOP; ++i) {
            attnres(PF.prev, nprev + 1, E.msg_queries + (size_t) i * 4 * H, 4, PF.msgs, nullptr, nullptr, nullptr,
                    nullptr, T);
            run(LOOP_BEGIN + i, NL + (p - 1) * NLOOP + i, PF.msgs);
        }
    }
    for (int l = LOOP_END; l < NL; ++l) run(l, l, nullptr);
    copy_b(PF.B + nB * ES, PF.h, ES);
    attnres(PF.B, nB + 1, E.final_query, 1, nullptr, E.norm, nullptr, nullptr, PF.mn, T);
    if (g_feat) k_feat_put<<<T, 256>>>(PF.mn, g_feat, 3);   // visit 48: the normalized output
    if (g_verify) {   // every position's greedy token
        for (int t0 = 0; t0 < T; t0 += MV) {
            const int nv = std::min(MV, T - t0);
            k_gemv_q8_multi<<<(VOCAB + 31) / 32, 256>>>(E.unembed.q, E.unembed.qs, PF.mn + (size_t) t0 * H, H, g_vlog, VOCAB_ROWS, nv, VOCAB, H);
            k_argmax_rows<<<nv, 1024>>>(g_vlog, VOCAB_ROWS, VOCAB, g_vids + t0);
        }
    }
    gemv(E.unembed, PF.mn + (size_t) (T - 1) * H, E.logits, VOCAB);
    k_softcap<<<(VOCAB + 255) / 256, 256>>>(E.logits, VOCAB, LOGIT_CAP);
    if (all) {   // every position (verification): the unembedding in slices of 8192 rows
        for (int r0 = 0; r0 < VOCAB; r0 += 8192) {
            const int n = std::min(8192, VOCAB - r0);
            sgemm(wf32(E.unembed, r0, n), n, H, PF.mn, H, PF.lg, n, T);
            k_softcap<<<(unsigned) (((size_t) T * n + 255) / 256), 256>>>(PF.lg, T * n, LOGIT_CAP);
            CK(cudaMemcpy2DAsync(all + r0, (size_t) VOCAB * 4, PF.lg, (size_t) n * 4, (size_t) n * 4, T,
                                 cudaMemcpyDeviceToHost, cudaStreamPerThread));
        }
    }
    CK(cudaStreamSynchronize(cudaStreamPerThread));
}

std::vector<int> parse_ids(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p) break;   // trailing whitespace / newline: no number, nothing pushed
        v.push_back((int) x);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// sampling on the host (the engine samples; Strata's server only forwards the keys): greedy without a temperature
struct Sampler {
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    std::mt19937_64 rng{std::random_device{}()};
};
int sample(const vec& lg, Sampler& s) {
    const int best = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    if (s.temp <= 0.0f) return best;
    const float mx = lg[best], thr = mx - 20.8f * s.temp;   // exp(-20.8) ~ 1e-9 of the top probability: never matters
    std::vector<std::pair<float, int>> c;
    for (int i = 0; i < VOCAB; ++i)
        if (lg[i] >= thr) c.push_back({lg[i], i});
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if (s.top_k > 0 && (int) c.size() > s.top_k) c.resize(s.top_k);
    std::vector<double> p(c.size());
    double z = 0.0;
    for (size_t i = 0; i < c.size(); ++i) z += p[i] = std::exp((double) (c[i].first - mx) / s.temp);
    size_t keep = c.size();
    if (s.min_p > 0.0f)
        for (size_t i = 0; i < keep; ++i) if (p[i] < s.min_p * p[0]) { keep = i; break; }
    double acc = 0.0, total = 0.0;
    for (size_t i = 0; i < keep; ++i) total += p[i];
    if (s.top_p < 1.0f) {
        for (size_t i = 0; i < keep; ++i) { acc += p[i]; if (acc >= s.top_p * total) { keep = i + 1; break; } }
        total = 0.0;
        for (size_t i = 0; i < keep; ++i) total += p[i];
    }
    double u = std::uniform_real_distribution<double>(0.0, total)(s.rng);
    for (size_t i = 0; i < keep; ++i) { u -= p[i]; if (u <= 0.0) return c[i].second; }
    return c[keep - 1].second;
}

constexpr int EOS_ID = 200006;   // <|content_model_end_sampling|> (generation_config.json)

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// MTP (--mtp DIR with mtp.safetensors; greedy decode): the DSpark drafter of Inkling-2-Mini, as in llama.cpp PR #32218
// (src/models/dspark.cpp graph_inkling2 / build_inkling2_head, common/speculative.cpp).
//   * features: the target's hidden states at the inputs of visits 5, 23, 41 and its normalized output, per position;
//   * inject (committed positions): per drafter layer, context = rms(softmax(fuse[l]) . features) * dflare_norm[l],
//     K/V = k_proj_ctx / v_proj_ctx (K: head RMS norm, NeoX RoPE theta 1e6) into the drafter's cache; the head's
//     context f = rms(softmax(fuse[3]) . features) * markov_head.dflare_norm of the last committed position;
//   * draft: the block [last token, 7 x mask_embed] at positions P .. P + 7 through 3 dense layers (q/k head norms,
//     RoPE, attention over the context window (layers 0, 2: the 1024 cells before P; layer 1: all) plus the whole block,
//     SiLU MLP), norm; then for i = 1..7: joint = rnn_proj . [state, w1[prev], hidden_i, f] + b -> gate / candidate /
//     latent, state = cand + gate (state - cand), logits_i = unembed . (hidden_i + hidden_out . tanh(latent) + b)
//     (the reference adds unembed . hidden_i and unembed . correction: the same sum), draft_i = first argmax;
//   * verify: the target reads [last, d1..d7] as one prompt chunk; d_i is accepted while it equals the target's greedy
//     token after d_(i-1); the target's token after the last accepted one is the next "last".  The conv histories are
//     rebuilt for the accepted prefix (k_hist_fix); K/V cells past it are rewritten later.
// The drafter's matrices are quantized to Q8_0 at load (k_gemv_q8_multi); markov_w1 stays in mapped host memory.
struct MtpLayer { DMat q, k, v, o, kc, vc, g, u, d; const float *attn_norm, *ffn_norm, *dfl, *qn, *kn, *kcn; bool swa; __half *Kc, *Vc; int cap; };
struct Mtp {
    bool on = false;
    std::string dir;
    MtpLayer L[3];
    float fuse[4][4];
    const float *norm, *head_norm, *mask_embed, *rnn_b, *hid_b;
    DMat rnn, hid;
    const uint16_t* w1 = nullptr;   // device view of the mapped host table
    float *cur, *hbuf, *q, *k, *v, *att, *o, *gb, *ub, *ab, *ctx, *mix, *frnn, *state, *jin, *jout, *lat, *corr, *vv, *lg;
    int *prev, *drafts;
    float *feat;   // [TP][4][H]
    std::vector<HistFix> fixes;
    HistFix* d_fixes = nullptr;
    long long steps = 0, accepted = 0;
    double t_draft = 0, t_verify = 0, t_inject = 0;   // ms (INK2_MTP_TIME=1: synchronized timing)
    bool timing = false;
    long long hist[8] = {};   // steps by accepted count
    int nd = 7;               // --mtp-draft: drafts verified per step (1..7; verify reads nd + 1 tokens)
};
Mtp MT;

DMat q8_from_f32(const vec& f, int n, int k) {   // per 32: d = max|x| / 127, q = rint(x / d)
    std::vector<int8_t> q((size_t) n * k);
    std::vector<__half> sc((size_t) n * (k / 32));
    for (size_t b = 0; b < sc.size(); ++b) {
        float am = 0.0f;
        for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(f[b * 32 + i]));
        const float d = am / 127.0f;
        sc[b] = __float2half_rn(d);
        const float dd = __half2float(sc[b]);
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) (dd > 0 ? std::max(-127, std::min(127, (int) std::lrint(f[b * 32 + i] / dd))) : 0);
    }
    DMat m;
    m.n = n; m.k = k;
    m.q = (const int8_t*) upload(q.data(), q.size());
    m.qs = (const __half*) upload(sc.data(), sc.size() * 2);
    return m;
}
void mtp_load(int ctx) {
    Ckpt ck;
    ck.path = MT.dir + "/mtp.safetensors";
    const std::string e = strata::image::st_read_header(ck.path, ck.st);
    if (!e.empty()) { std::fprintf(stderr, "--mtp: %s\n", e.c_str()); std::exit(1); }
    auto mat = [&](const std::string& n) {
        const StTensor& t = ck.t(n);
        return q8_from_f32(ck.f32(n), (int) t.shape[0], (int) t.shape[1]);
    };
    auto fv = [&](const std::string& n) { return dvec(ck.f32(n)); };
    for (int l = 0; l < 3; ++l) {
        MtpLayer& Y = MT.L[l];
        const std::string p = "layers." + std::to_string(l) + ".";
        Y.q = mat(p + "self_attn.q_proj.weight"); Y.k = mat(p + "self_attn.k_proj.weight"); Y.v = mat(p + "self_attn.v_proj.weight");
        Y.o = mat(p + "self_attn.o_proj.weight"); Y.kc = mat(p + "self_attn.k_proj_ctx.weight"); Y.vc = mat(p + "self_attn.v_proj_ctx.weight");
        Y.g = mat(p + "mlp.gate_proj.weight"); Y.u = mat(p + "mlp.up_proj.weight"); Y.d = mat(p + "mlp.down_proj.weight");
        Y.attn_norm = fv(p + "input_layernorm.weight"); Y.ffn_norm = fv(p + "post_attention_layernorm.weight");
        Y.dfl = fv(p + "dflare_norm.weight"); Y.qn = fv(p + "self_attn.q_norm.weight"); Y.kn = fv(p + "self_attn.k_norm.weight");
        Y.kcn = fv(p + "self_attn.k_norm_ctx.weight");
        Y.swa = l != 1;   // mtp_config local_layer_ids [0, 2]
        Y.cap = Y.swa ? 2 * WINDOW : ctx;
        if (!Y.swa && E.grow) {   // the full layer's cache grows with the target's (kv_fit)
            for (int a = 0; a < 2; ++a) {
                g_kvr.emplace_back();
                VRange& r = g_kvr.back();
                r.init((size_t) Y.cap * KVD * 2);
                g_kvr_bpc.push_back((size_t) KVD * 2);
                if (!r.map_to((size_t) E.kv_cells * KVD * 2)) { std::fprintf(stderr, "--mtp: no VRAM for the context cache\n"); std::exit(1); }
                (a ? Y.Vc : Y.Kc) = (__half*) r.ptr();
            }
        } else {
            Y.Kc = (__half*) dalloc((size_t) Y.cap * KVD * 2);
            Y.Vc = (__half*) dalloc((size_t) Y.cap * KVD * 2);
        }
    }
    const vec fw = ck.f32("fuse_weight");   // [4 rows: layers 0..2, head][4 targets]: softmax per row
    for (int r = 0; r < 4; ++r) {
        float mx = -1e30f, z = 0;
        for (int j = 0; j < 4; ++j) mx = std::max(mx, fw[r * 4 + j]);
        for (int j = 0; j < 4; ++j) z += std::exp(fw[r * 4 + j] - mx);
        for (int j = 0; j < 4; ++j) MT.fuse[r][j] = std::exp(fw[r * 4 + j] - mx) / z;
    }
    MT.norm = fv("norm.weight");
    MT.head_norm = fv("markov_head.dflare_norm.weight");
    MT.mask_embed = fv("mask_embed");
    MT.rnn = mat("markov_head.rnn_proj.weight");
    MT.rnn_b = fv("markov_head.rnn_proj.bias");
    MT.hid = mat("markov_head.hidden_out.weight");
    MT.hid_b = fv("markov_head.hidden_out.bias");
    {
        auto b = ck.bytes("markov_head.markov_w1.weight");
        void* h = nullptr;
        CK(cudaHostAlloc(&h, b.size(), cudaHostAllocMapped));
        std::memcpy(h, b.data(), b.size());
        void* dv = nullptr;
        CK(cudaHostGetDevicePointer(&dv, h, 0));
        MT.w1 = (const uint16_t*) dv;
    }
    const int B = 8, FFM = 9216;
    MT.cur = dfloats(B * H); MT.hbuf = dfloats(B * H); MT.q = dfloats(B * QD); MT.k = dfloats((size_t) PF.TP * KVD);
    MT.v = dfloats((size_t) PF.TP * KVD); MT.att = dfloats(B * QD); MT.o = dfloats(B * H); MT.gb = dfloats(B * FFM);
    MT.ub = dfloats(B * FFM); MT.ab = dfloats(B * FFM); MT.ctx = dfloats((size_t) PF.TP * H); MT.mix = dfloats((size_t) PF.TP * H);
    MT.frnn = dfloats(H); MT.state = dfloats(256); MT.jin = dfloats(512 + 2 * H); MT.jout = dfloats(768); MT.lat = dfloats(256);
    MT.corr = dfloats(H); MT.vv = dfloats(H); MT.lg = dfloats(VOCAB_ROWS);
    MT.prev = (int*) dalloc(4); MT.drafts = (int*) dalloc(8 * 4);
    MT.feat = dfloats((size_t) PF.TP * 4 * H);
    g_vlog = dfloats((size_t) 8 * VOCAB_ROWS);
    g_vids = (int*) dalloc(8 * 4);
    // conv histories of every slot: verify saves the chunk's inputs, k_hist_fix rebuilds a partial accept
    for (int sl = 0; sl < NCALLS; ++sl) {
        Slot& S = E.S[sl];
        float* h4[4] = {S.khist, S.vhist, S.ahist, S.mhist};
        const int C4[4] = {KVD, KVD, H, H};
        for (int a = 0; a < 4; ++a) {
            float* sv = dfloats((size_t) 8 * C4[a]);
            g_hist_save[h4[a]] = sv;
            MT.fixes.push_back({h4[a], sv, dfloats((size_t) 3 * C4[a]), C4[a]});
        }
    }
    MT.d_fixes = (HistFix*) upload(MT.fixes.data(), MT.fixes.size() * sizeof(HistFix));
    std::fprintf(stderr, "mtp: drafter loaded (3 layers, Q8_0), %.2f GiB in VRAM\n", g_vram / 1073741824.0);
}
void mgemv(const DMat& W, const float* X, int ldx, float* Y, int ldy, int nv) {   // nv vectors (any count)
    for (int v0 = 0; v0 < nv; v0 += MV)
        k_gemv_q8_multi<<<(W.n + 31) / 32, 256>>>(W.q, W.qs, X + (size_t) v0 * ldx, ldx, Y + (size_t) v0 * ldy, ldy, std::min(MV, nv - v0), W.n, W.k);
}
// the drafter's context for n committed positions p0 .. p0 + n - 1 (features in MT.feat rows f0 ..)
void mtp_inject(const float* F, int n, int p0) {
    const unsigned g = (unsigned) (((size_t) n * H + 255) / 256);
    for (int l = 0; l < 3; ++l) {
        MtpLayer& Y = MT.L[l];
        float* w = (float*) MT.jout;   // 4 weights (scratch)
        CK(cudaMemcpyAsync(w, MT.fuse[l], 16, cudaMemcpyHostToDevice));
        k_mtp_mix<<<g, 256>>>(F, w, MT.mix, n);
        k_rms<<<n, 512>>>(MT.mix, Y.dfl, MT.ctx, H);
        mgemv(Y.kc, MT.ctx, H, MT.k, KVD, n);
        mgemv(Y.vc, MT.ctx, H, MT.v, KVD, n);
        k_mtp_norm_rope<<<dim3(NKV, n), 128>>>(MT.k, KVD, Y.kcn, p0, 1e6f, 1);
        k_mtp_store<<<(unsigned) (((size_t) n * KVD + 255) / 256), 256>>>(MT.k, MT.v, Y.Kc, Y.Vc, Y.cap, p0, n);
    }
    float* w = (float*) MT.jout;
    CK(cudaMemcpyAsync(w, MT.fuse[3], 16, cudaMemcpyHostToDevice));
    k_mtp_mix<<<(H + 255) / 256, 256>>>(F + (size_t) (n - 1) * 4 * H, w, MT.mix, 1);
    k_rms<<<1, 512>>>(MT.mix, MT.head_norm, MT.frnn, H);
}
// 7 drafts after token `last` at position P (the context holds positions < P); host ids in out[0..6]
void mtp_draft(int last, int P, int* out) {
    const int B = 8;
    {
        vec x((size_t) B * H);
        embed_row(last, x.data());
        CK(cudaMemcpy(MT.hbuf, x.data(), H * 4, cudaMemcpyHostToDevice));
        k_rms<<<1, 512>>>(MT.hbuf, E.embed_norm, MT.cur, H);
        for (int i = 1; i < B; ++i) CK(cudaMemcpyAsync(MT.cur + (size_t) i * H, MT.mask_embed, H * 4, cudaMemcpyDeviceToDevice));
    }
    for (int l = 0; l < 3; ++l) {
        MtpLayer& Y = MT.L[l];
        k_rms<<<B, 512>>>(MT.cur, Y.attn_norm, MT.hbuf, H);
        mgemv(Y.q, MT.hbuf, H, MT.q, QD, B);
        mgemv(Y.k, MT.hbuf, H, MT.k, KVD, B);
        mgemv(Y.v, MT.hbuf, H, MT.v, KVD, B);
        k_mtp_norm_rope<<<dim3(NQ, B), 128>>>(MT.q, QD, Y.qn, P, 1e6f, 1);
        k_mtp_norm_rope<<<dim3(NKV, B), 128>>>(MT.k, KVD, Y.kn, P, 1e6f, 1);
        const int c0 = Y.swa ? std::max(0, P - WINDOW) : 0;
        k_mtp_attn<<<dim3(NQ, B), 128>>>(MT.q, Y.Kc, Y.Vc, Y.cap, c0, P, MT.k, MT.v, B, MT.att);
        mgemv(Y.o, MT.att, QD, MT.o, H, B);
        kb_add<<<(B * H + 255) / 256, 256>>>(MT.cur, MT.o, MT.cur, (size_t) B * H);
        k_rms<<<B, 512>>>(MT.cur, Y.ffn_norm, MT.hbuf, H);
        mgemv(Y.g, MT.hbuf, H, MT.gb, Y.g.n, B);
        mgemv(Y.u, MT.hbuf, H, MT.ub, Y.u.n, B);
        k_mtp_silu_mul<<<(unsigned) (((size_t) B * Y.g.n + 255) / 256), 256>>>(MT.gb, MT.ub, MT.ab, (size_t) B * Y.g.n);
        mgemv(Y.d, MT.ab, Y.g.n, MT.o, H, B);
        kb_add<<<(B * H + 255) / 256, 256>>>(MT.cur, MT.o, MT.cur, (size_t) B * H);
    }
    k_rms<<<B, 512>>>(MT.cur, MT.norm, MT.hbuf, H);   // hidden_i = MT.hbuf[i]
    CK(cudaMemsetAsync(MT.state, 0, 256 * 4));
    CK(cudaMemcpyAsync(MT.prev, &last, 4, cudaMemcpyHostToDevice));
    for (int i = 1; i <= MT.nd; ++i) {   // (the block always holds 8: later masks shape the earlier rows)
        const float* hid = MT.hbuf + (size_t) i * H;
        k_mtp_joint_in<<<(512 + 2 * H + 255) / 256, 256>>>(MT.state, MT.w1, MT.prev, hid, MT.frnn, MT.jin);
        gemv(MT.rnn, MT.jin, MT.jout);
        k_mtp_gates<<<1, 256>>>(MT.jout, MT.rnn_b, MT.state, MT.lat);
        gemv(MT.hid, MT.lat, MT.corr);
        k_mtp_add3<<<(H + 255) / 256, 256>>>(hid, MT.corr, MT.hid_b, MT.vv);
        gemv(E.unembed, MT.vv, MT.lg, VOCAB);
        k_argmax_rows<<<1, 1024>>>(MT.lg, VOCAB_ROWS, VOCAB, MT.prev);
        CK(cudaMemcpyAsync(MT.drafts + (i - 1), MT.prev, 4, cudaMemcpyDeviceToDevice));
    }
    CK(cudaMemcpy(out, MT.drafts, MT.nd * 4, cudaMemcpyDeviceToHost));
}
// one speculative step from `last` at position P: emits the accepted drafts and the target's next token into out,
// returns their count; *next = the new last
int mtp_step(int last, int P, std::vector<int>& out, int* next) {
    int d[7];
    double tA = 0;
    if (MT.timing) { CK(cudaDeviceSynchronize()); tA = now_ms(); }
    mtp_draft(last, P, d);
    if (MT.timing) { CK(cudaDeviceSynchronize()); MT.t_draft += now_ms() - tA; tA = now_ms(); }
    int toks[8] = {last, d[0], d[1], d[2], d[3], d[4], d[5], d[6]};
    const int nv = MT.nd + 1;
    pf_map(8);   // (elastic prompt buffers: an 8-token chunk's share, kept between steps)
    k_hist_snap<<<(unsigned) MT.fixes.size(), 256>>>(MT.d_fixes);
    g_verify = true;
    g_feat = MT.feat;
    forward_prefill(toks, nv, P, nullptr);
    g_verify = false;
    g_feat = nullptr;
    int t[8];
    CK(cudaMemcpy(t, g_vids, nv * 4, cudaMemcpyDeviceToHost));
    if (MT.timing) { MT.t_verify += now_ms() - tA; tA = now_ms(); }
    int a = 0;
    while (a < MT.nd && d[a] == t[a]) ++a;
    const int c = a + 1;   // committed positions P .. P + a
    k_hist_fix<<<(unsigned) MT.fixes.size(), 256>>>(MT.d_fixes, P, P + c);
    mtp_inject(MT.feat, c, P);
    if (MT.timing) { CK(cudaDeviceSynchronize()); MT.t_inject += now_ms() - tA; }
    for (int i = 0; i < a; ++i) out.push_back(d[i]);
    out.push_back(t[a]);
    *next = t[a];
    ++MT.steps;
    MT.accepted += a;
    ++MT.hist[a];
    return c;
}

int main(int argc, char** argv) {
    if (const char* r = std::getenv("INK2_RPW")) g_rpw = std::atoi(r);
    if (const char* r = std::getenv("INK2_SEL_RPW")) g_sel_rpw = std::atoi(r);
    if (const char* r = std::getenv("INK2_GEMV_MINB")) g_gemv_min_blocks = std::atoi(r);
    if (const char* r = std::getenv("INK2_GG_RPW")) g_gg_rpw = std::atoi(r);
    if (const char* r = std::getenv("INK2_GG_Q8")) g_gg_q8 = std::atoi(r) != 0;
    if (const char* r = std::getenv("INK2_PREFETCH")) g_dec_prefetch = std::atoi(r) != 0;
    if (const char* r = std::getenv("INK2_PREFETCH_BLOCKS")) g_dec_pf_blocks = std::atoi(r);
    if (const char* r = std::getenv("INK2_ROUTED_MAX")) g_pf_routed_max = std::atoi(r);
    if (const char* r = std::getenv("INK2_DQ_MIN")) g_dq_min = std::atoi(r);
    if (const char* r = std::getenv("INK2_OFF_MARGIN_MIB")) g_off_margin_mib = std::atoi(r);
    std::string dir, dump_logits, rec_path;
    std::vector<int> ids;
    int n_gen = 32;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--serve") { serve = true; continue; }
        if (i + 1 >= argc) { std::fprintf(stderr, "option %s needs a value\n", a.c_str()); return 2; }
        const char* v = argv[++i];
        if (a == "--model") dir = v;
        else if (a == "--tokens") ids = parse_ids(v);
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(v, "r");
            if (!f) { std::fprintf(stderr, "cannot open %s\n", v); return 2; }
            std::string text;
            for (int c; (c = std::fgetc(f)) != EOF;) text += (char) c;
            std::fclose(f);
            ids = parse_ids(text.c_str());
        }
        else if (a == "--n") n_gen = std::atoi(v);
        else if (a == "--ctx" || a == "--max-context") E.ctx = std::atoi(v);
        else if (a == "--dump-logits") dump_logits = v;
        else if (a == "--graph") E.graph = std::atoi(v) != 0;
        else if (a == "--chunk") PF.TP = std::atoi(v);
        else if (a == "--tc") PF.tc = std::atoi(v) != 0;
        else if (a == "--vram-reserve-mib") E.reserve_mib = std::atoi(v);
        else if (a == "--kv-grow") E.grow = std::atoi(v) != 0;
        else if (a == "--mtp") { MT.on = true; MT.dir = v; MT.timing = std::getenv("INK2_MTP_TIME") != nullptr; }
        else if (a == "--mtp-draft") MT.nd = std::max(1, std::min(7, std::atoi(v)));
        else if (a == "--kv-grow-step") E.grow_step = std::max(CHUNK, std::atoi(v) / CHUNK * CHUNK);
        else if (a == "--kv") {
            const std::string k = v;
            E.kvm = k == "k8v8" ? 5 : k == "k8v4" ? 4 : k == "i8r" ? 3 : k == "i8" ? 2 : k == "f32" ? 0 : k == "f16" ? 1 : -1;
            if (E.kvm < 0) { std::fprintf(stderr, "--kv f16|f32|i8|i8r|k8v4|k8v8\n"); return 2; }
        }
        else if (a == "--record-routing") { rec_path = v; E.rec_mode = 1; }
        else if (a == "--force-routing") { rec_path = v; E.rec_mode = 2; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (E.grow) vmm_init();
    if (dir.empty() || (ids.empty() && !serve)) {
        std::fprintf(stderr, "usage: strata-ink2 --model DIR (--serve | --tokens IDS | --tokens-file F) [--n N] [--ctx N] "
                             "[--chunk N] [--tc 0|1] [--kv f16|f32|i8|i8r|k8v4|k8v8] [--kv-grow 1 [--kv-grow-step N]] [--vram-reserve-mib N] [--dump-logits F]\n");
        return 2;
    }
    if (serve && PF.TP <= 0) { std::fprintf(stderr, "--serve needs --chunk > 0\n"); return 2; }
    if (!serve && (int) ids.size() + n_gen > E.ctx) { std::fprintf(stderr, "prompt + n exceeds --ctx %d\n", E.ctx); return 2; }
    Ckpt ck;
    Gguf gg;
    g_is_gguf = dir.size() > 5 && dir.compare(dir.size() - 5, 5, ".gguf") == 0;
    if (g_is_gguf) {   // --model FILE.gguf: llama.cpp's merged inkling2_loop layout
        const std::string e = gg_open(dir, gg);
        if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        g_gg = &gg;
    } else {
        ck.path = dir + "/model-00001-of-00001.safetensors";
        const std::string e = strata::image::st_read_header(ck.path, ck.st);
        if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        E.ck = &ck;
    }
    const double t_load = now_ms();
    if (g_is_gguf) load_gguf();
    else load();
    if (PF.TP > 0) PF_RING_EXTRA = PF.TP;
    alloc_state();
    if (g_is_gguf) g_yq = (gq8_1*) dalloc((size_t) TOPK * (std::max(H, FF) / 32) * sizeof(gq8_1));   // (not inside a capture)
    if (PF.TP > 0) alloc_prefill();
    const size_t rec_n = (size_t) NCALLS * E.ctx * TOPK;
    if (E.rec_mode) E.rec = (int*) dalloc(rec_n * 4);
    if (E.rec_mode == 2) {
        std::vector<int> r(rec_n);
        FILE* f = std::fopen(rec_path.c_str(), "rb");
        if (!f || std::fread(r.data(), 4, rec_n, f) != rec_n) { std::fprintf(stderr, "cannot read %s (same --ctx?)\n", rec_path.c_str()); return 1; }
        std::fclose(f);
        CK(cudaMemcpy(E.rec, r.data(), rec_n * 4, cudaMemcpyHostToDevice));
    }
    if (MT.on) {   // before the pools take the free memory
        if (!g_is_gguf || !E.unembed.q || PF.TP < 8) { std::fprintf(stderr, "--mtp needs a GGUF target (Q8_0 output) and --chunk >= 8\n"); return 2; }
        mtp_load(E.ctx);
    }
    if (E.xhost) setup_offload();
    if (E.xhost && g_dec_prefetch) dec_prefetch_init();

    vec logits(VOCAB);
    auto step = [&](int tok, int pos, bool want_logits) {
        kv_fit(pos + 1);
        CK(cudaMemcpyAsync(E.d_pos, &pos, 4, cudaMemcpyHostToDevice));
        upload_embed(tok);
        if (!E.graph) {
            forward_device();
        } else {
            if (!E.exec) {   // built once: every launch reads the position from E.d_pos
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
                forward_device();
                CK(cudaStreamEndCapture(cudaStreamPerThread, &g));
                CK(cudaGraphInstantiate(&E.exec, g, 0));
            }
            CK(cudaGraphLaunch(E.exec, cudaStreamPerThread));
        }
        if (want_logits) CK(cudaMemcpy(logits.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
    };
    // warm-up at position 0 (builds the decode graph outside any timing; position 0 is rewritten by the first prompt)
    step(0, 0, false);
    CK(cudaDeviceSynchronize());
    std::fprintf(stderr, "load %.1f s, %.2f GiB in VRAM (ctx %d)\n", (now_ms() - t_load) / 1000, g_vram / 1073741824.0, E.ctx);

    if (serve) {
        // Strata engine protocol (terminal/serve/server.py): READY <ctx> stop; GEN <max_new>[ k=v ...] <ids>;
        // T <id> per token; DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish>; PP <read> <total> <ms> <tok/s>;
        // STOP (read by its own thread; cleared when the next GEN starts); EOF / QUIT ends.
        std::atomic<bool> stop{false};
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> q;
        bool eof = false;
        std::thread reader([&] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line == "STOP") { stop.store(true); continue; }
                { std::lock_guard<std::mutex> lk(mu); q.push_back(line); }
                cv.notify_one();
            }
            { std::lock_guard<std::mutex> lk(mu); eof = true; }
            cv.notify_one();
        });
        reader.detach();
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::vector<int> hist;   // tokens whose KV / conv state is in the slots; last_logits follow hist.back()
        vec last_logits;
        Sampler smp;
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !q.empty(); });
                if (q.empty()) break;
                line = q.front();
                q.pop_front();
            }
            if (line == "QUIT") break;
            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            stop.store(false);
            std::vector<std::string> f;
            for (size_t p = 0; p < line.size();) {
                const size_t s2 = line.find(' ', p);
                f.push_back(line.substr(p, s2 == std::string::npos ? std::string::npos : s2 - p));
                if (s2 == std::string::npos) break;
                p = s2 + 1;
            }
            if (f.size() < 3) { std::printf("ERR malformed GEN\n"); std::fflush(stdout); continue; }
            int max_new = std::atoi(f[1].c_str());
            std::vector<int> pr = parse_ids(f.back().c_str());
            Sampler s;
            s.rng = std::move(smp.rng);
            for (size_t k = 2; k + 1 < f.size(); ++k) {   // unknown keys are ignored
                const size_t eq = f[k].find('=');
                if (eq == std::string::npos) continue;
                const std::string key = f[k].substr(0, eq), val = f[k].substr(eq + 1);
                if (key == "temperature") s.temp = std::strtof(val.c_str(), nullptr);
                else if (key == "top_p") s.top_p = std::strtof(val.c_str(), nullptr);
                else if (key == "top_k") s.top_k = std::atoi(val.c_str());
                else if (key == "min_p") s.min_p = std::strtof(val.c_str(), nullptr);
                else if (key == "seed") s.rng.seed(std::strtoull(val.c_str(), nullptr, 10));
            }
            if (pr.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); smp.rng = std::move(s.rng); continue; }
            if ((int) pr.size() >= E.ctx) {
                std::printf("ERR prompt of %zu tokens exceeds the context %d\n", pr.size(), E.ctx);
                std::fflush(stdout);
                smp.rng = std::move(s.rng);
                continue;
            }
            max_new = std::max(0, std::min(max_new, E.ctx - (int) pr.size()));
            const bool reuse = !hist.empty() && pr.size() >= hist.size() && std::equal(hist.begin(), hist.end(), pr.begin());
            kv_trim((int) pr.size() + 1);
            const size_t start = reuse ? hist.size() : 0;
            const double t0 = now_ms();
            if (start == pr.size()) {
                logits = last_logits;
            } else {
                for (size_t p0 = start, Tz = 0; p0 < pr.size(); p0 += Tz) {
                    const int T = pf_fit((int) std::min((size_t) PF.TP, pr.size() - p0), (int) p0);
                    Tz = (size_t) T;
                    if (T < g_pf_mapped_T) pf_unmap();
                    pf_map(T);
                    forward_prefill(&pr[p0], T, (int) p0, nullptr);
                    const double ms = now_ms() - t0;
                    std::printf("PP %zu %zu %.0f %.1f\n", p0 + T, pr.size(), ms, 1000.0 * (p0 + T - start) / std::max(ms, 1.0));
                    std::fflush(stdout);
                }
                CK(cudaMemcpy(logits.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
                pf_unmap();
            }
            const double t1 = now_ms();
            std::vector<int> gen;
            const char* finish = "length";
            int pos = (int) pr.size();
            for (int k = 0; k < max_new; ++k) {
                if (stop.load()) { finish = "cancel"; break; }
                const int next = sample(logits, s);
                gen.push_back(next);
                std::printf("T %d\n", next);
                std::fflush(stdout);
                if (next == EOS_ID) { finish = "stop"; break; }
                if (k + 1 < max_new) step(next, pos++, true);
            }
            const double t2 = now_ms();
            hist = pr;
            if (!gen.empty()) hist.insert(hist.end(), gen.begin(), gen.end() - 1);   // the last token was never fed
            last_logits = logits;
            smp.rng = std::move(s.rng);
            std::printf("DONE %zu %zu %.0f %.0f %s\n", gen.size(), pr.size(), t1 - t0, t2 - t1, finish);
            std::fflush(stdout);
        }
        return 0;
    }

    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    const double t0 = now_ms();
    if (PF.TP > 0) {   // chunked prefill
        std::vector<float> all(fg ? (size_t) PF.TP * VOCAB : 0);
        for (size_t p0 = 0, T = 0; p0 < ids.size(); p0 += T) {
            T = (size_t) pf_fit((int) std::min((size_t) PF.TP, ids.size() - p0), (int) p0);
            if ((int) T < g_pf_mapped_T) pf_unmap();
            pf_map((int) T);
            if (MT.on) g_feat = MT.feat;
            forward_prefill(&ids[p0], T, (int) p0, fg ? all.data() : nullptr);
            if (MT.on) { g_feat = nullptr; mtp_inject(MT.feat, T, (int) p0); }
            if (fg) std::fwrite(all.data(), 4, (size_t) T * VOCAB, fg);
        }
        CK(cudaMemcpy(logits.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
        pf_unmap();
    } else {
        for (size_t t = 0; t < ids.size(); ++t) {
            step(ids[t], (int) t, fg != nullptr || t + 1 == ids.size());
            if (fg) std::fwrite(logits.data(), 4, logits.size(), fg);
        }
    }
    CK(cudaDeviceSynchronize());
    const double t1 = now_ms();
    if (fg) std::fclose(fg);
    unsigned long long fb1 = 0, fb2 = 0;
    if (E.xhost) CK(cudaMemcpy(&fb1, E.fbytes, 8, cudaMemcpyDeviceToHost));
    std::printf("gen:");
    int pos = (int) ids.size();
    int n_out = n_gen;
    if (MT.on) {   // speculative: draft 7, verify 8, accept the matching prefix + the target's next token
        int last = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::printf(" %d", last);   // the prompt's own next token
        n_out = 1;
        while (n_out < n_gen) {
            std::vector<int> o;
            int nx = 0;
            const int c = mtp_step(last, pos, o, &nx);
            bool eos = false;
            for (int tk : o) { std::printf(" %d", tk); ++n_out; eos |= tk == EOS_ID; }
            pos += c;
            last = nx;
            if (eos) break;
        }
        std::fprintf(stderr, "mtp: %lld steps, %.2f drafts accepted per step (%.1f tokens per step)\n", MT.steps,
                     MT.steps ? (double) MT.accepted / MT.steps : 0.0, MT.steps ? 1.0 + (double) MT.accepted / MT.steps : 0.0);
        std::fprintf(stderr, "mtp: accepted 0..7:");
        for (long long h : MT.hist) std::fprintf(stderr, " %lld", h);
        std::fprintf(stderr, "\n");
        if (MT.timing && MT.steps)
            std::fprintf(stderr, "mtp: per step draft %.1f ms, verify %.1f ms, inject %.1f ms\n", MT.t_draft / MT.steps,
                         MT.t_verify / MT.steps, MT.t_inject / MT.steps);
    } else
    for (int k = 0; k < n_gen; ++k) {
        const int next = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::printf(" %d", next);
        step(next, pos++, true);
    }
    CK(cudaDeviceSynchronize());
    const double t2 = now_ms();
    if (E.xhost) {
        CK(cudaMemcpy(&fb2, E.fbytes, 8, cudaMemcpyDeviceToHost));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (g_pred_st) {
            unsigned long long st[2];
            CK(cudaMemcpy(st, g_pred_st, 16, cudaMemcpyDeviceToHost));
            std::printf("\nprefetch guesses: %.1f%% of the next layer's experts\n", st[1] ? 100.0 * st[0] / st[1] : 0.0);
        }
        std::printf("\nexpert fetch: prompt %.1f MB/token, decode %.1f MB/token; VRAM used %.2f GiB of %.2f\n",
                    (fb1 + PF.fbytes) / 1e6 / std::max<size_t>(1, ids.size()), (fb2 - fb1) / 1e6 / std::max(1, n_gen),
                    (tot - fr) / 1073741824.0, tot / 1073741824.0);
    }
    if (E.rec_mode == 1) {
        std::vector<int> r(rec_n);
        CK(cudaMemcpy(r.data(), E.rec, rec_n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen(rec_path.c_str(), "wb");
        std::fwrite(r.data(), 4, rec_n, f);
        std::fclose(f);
    }
    std::printf("\nprompt %zu tokens %.0f ms (%.1f tok/s), decode %d tokens %.0f ms (%.1f tok/s)\n", ids.size(), t1 - t0,
                1000.0 * ids.size() / (t1 - t0), n_out, t2 - t1, n_out > 0 ? 1000.0 * n_out / (t2 - t1) : 0.0);
    return 0;
}
