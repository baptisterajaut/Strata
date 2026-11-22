// strata-ink2 glue over the vendored Marlin MoE kernels (vLLM, Apache-2.0; see README.md in this directory):
// NVFP4 expert weights (e2m1, one e4m3 scale per 16 columns, one global scale per expert), fp16 activations and
// outputs, fp32 accumulation.  Standalone CUDA, no torch.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ink_marlin {

// One grouped product: for every row r listed in sorted_ids (blocks of `block` rows of one expert, padded with
// values >= m * top_k), C[r] = gscale[e] * W_e . A[r / top_k].
struct Gemm {
    const void* A = nullptr;                 // fp16 [m][K]
    void* C = nullptr;                       // fp16 [m * top_k][N]
    float* c_tmp = nullptr;                  // fp32 scratch, c_tmp_floats() floats
    const uint8_t* base = nullptr;           // a 16-byte aligned address the parts lie within +-32 GiB of (the pool)
    const uint8_t* const* tab = nullptr;     // [expert] -> its part in the Marlin layout (repack_part)
    const float* gscale = nullptr;           // [expert] the global scale * GSCALE_MUL
    const int* sorted_ids = nullptr;         // [nblocks * block]
    const int* expert_ids = nullptr;         // [nblocks]
    const int* num_padded = nullptr;         // [1] nblocks * block
    int* locks = nullptr;                    // LOCKS ints, zero
    int block = 8;                           // 8, 16, 32, 48 or 64 rows
    int top_k = 1, m = 0, N = 0, K = 0;
};
constexpr int LOCKS = 4096;
constexpr float GSCALE_MUL = 128.0f;         // Marlin's NVFP4 scales are stored * 2^7 (fp16 activations)
size_t c_tmp_floats(int block, int max_padded_rows, int N);
void gemm(const Gemm& g, cudaStream_t stream);

// One expert part, checkpoint layout (N rows of K / 2 bytes, two e2m1 per byte, low nibble first, then N rows of
// K / 16 e4m3 scales) -> Marlin layout of the same size (weights in 16 x 64 tensor-core tiles, K-major; scales
// [K / 16][N] in Marlin's permuted order and S0E5M3 encoding).  src and dst must not overlap.
// e4m3 scales below 2^-6 (subnormal) are not representable and become 0: count them first (count_subnormal_scales).
void repack_part(const uint8_t* src, uint8_t* dst, int N, int K, cudaStream_t stream);

}   // namespace ink_marlin
