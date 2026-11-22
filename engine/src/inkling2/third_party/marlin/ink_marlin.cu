// strata-ink2 glue over the vendored Marlin MoE kernels: launcher and weight repack.
//
// The launcher wraps marlin_mm of moe/ops.cu (vLLM, Apache-2.0).  repack_part is derived from vLLM's
// csrc/quantization/marlin/gptq_marlin_repack.cu (weights) and from marlin_permute_scales /
// nvfp4_marlin_process_scales of vllm/model_executor/layers/quantization/utils/marlin_utils{,_fp4}.py (scales),
// Apache-2.0, rewritten as two gather kernels that read the checkpoint's row-major layout directly.
// See README.md in this directory for the source commit and the list of modifications.
#include "ink_marlin.h"

#include "moe/ops.cu"

namespace ink_marlin {

size_t c_tmp_floats(int block, int max_padded_rows, int N) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    size_t n = std::min((size_t) N * max_padded_rows, (size_t) sms * 4 * block * MARLIN_NAMESPACE_NAME::max_thread_n);
    return block == 8 ? 2 * n : n;
}

void gemm(const Gemm& g, cudaStream_t stream) {
    static int sms = 0, dev = 0;
    if (!sms) { cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); }
    MARLIN_NAMESPACE_NAME::marlin_mm(
        g.A, g.base, g.C, g.c_tmp, nullptr, nullptr, const_cast<float*>(g.gscale), nullptr,
        const_cast<int*>(g.sorted_ids), const_cast<int*>(g.expert_ids), const_cast<int*>(g.num_padded), nullptr, g.block,
        0, g.top_k, false, g.m, g.N, g.K, g.locks, vllm::kFloat16, vllm::kFE2M1f, vllm::kFloat16, vllm::kFE4M3fn, false,
        false, 16, dev, stream, -1, -1, sms, -1, false, true, false, g.tab);
}

namespace {

// Weights: output word w of tile (k_tile, n_tile) holds the 8 nibbles Marlin's dequant expects for one tensor-core
// thread (gptq_marlin_repack_kernel: th_id = w / 4, warp_id = w % 4).
__global__ void k_repack_w(const uint8_t* __restrict__ W, uint32_t* __restrict__ out, int N, int K) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int n_tiles = N / 64, k_tiles = K / 16;
    if (idx >= (size_t) n_tiles * k_tiles * 128) return;
    const int w = (int) (idx % 128), tile = (int) (idx / 128), n_tile = tile % n_tiles, k_tile = tile / n_tiles;
    const int th_id = w / 4, warp_id = w % 4, tc_col = th_id / 4, tc_row = (th_id % 4) * 2;
    const int n = n_tile * 64 + warp_id * 16 + tc_col, k0 = k_tile * 16;
    const int tc_offsets[4] = {0, 1, 8, 9}, pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = k0 + tc_row + tc_offsets[i];
        vals[i] = (W[(size_t) n * (K / 2) + k / 2] >> (4 * (k & 1))) & 15u;
        vals[4 + i] = (W[(size_t) (n + 8) * (K / 2) + k / 2] >> (4 * (k & 1))) & 15u;
    }
    uint32_t res = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) res |= vals[pack_idx[i]] << (i * 4);
    out[idx] = res;
}

// Scales: [N][K / 16] e4m3 -> [K / 16][N], permuted within every 64 columns (marlin_permute_scales, then the
// [0, 2, 1, 3] order of nvfp4_marlin_process_scales), each as (scale * 2^7) in S0E5M3: e4m3 byte + 120, or 0 below
// 2^-6 (the e4m3 subnormals; negative scales do not occur).
__global__ void k_repack_s(const uint8_t* __restrict__ S, uint8_t* __restrict__ out, int N, int K) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int G = K / 16;
    if (idx >= (size_t) G * N) return;
    const int g = (int) (idx / N), col = (int) (idx % N), c = col / 64, p = col % 64;
    const int order[4] = {0, 2, 1, 3};
    const int p1 = (p / 4) * 4 + order[p % 4];        // position after the first permutation
    const int src = (p1 / 8) + 8 * (p1 % 8);          // scale_perm[8 i + j] = i + 8 j
    const uint8_t b = S[(size_t) (c * 64 + src) * G + g];
    out[idx] = (b & 0x80) || b < 8 ? 0 : (uint8_t) (b + 120);
}

}   // namespace

void repack_part(const uint8_t* src, uint8_t* dst, int N, int K, cudaStream_t stream) {
    const size_t words = (size_t) N * K / 8, sc = (size_t) N * K / 16;
    k_repack_w<<<(unsigned) ((words + 255) / 256), 256, 0, stream>>>(src, reinterpret_cast<uint32_t*>(dst), N, K);
    k_repack_s<<<(unsigned) ((sc + 255) / 256), 256, 0, stream>>>(src + (size_t) N * K / 2, dst + (size_t) N * K / 2, N, K);
}

}   // namespace ink_marlin
