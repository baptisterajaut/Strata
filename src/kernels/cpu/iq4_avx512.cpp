// src/kernels/cpu/iq4_avx512.cpp - IQ4_XS gate/up and IQ4_NL down expert rows in 512-bit lanes, up to 8 tokens a pass.
//
// For the CPUs with AVX-512 F/BW/VL/DQ + VNNI and no VBMI (Cascade Lake: Xeon Gold 6226 / 6230R), where cpu_avx512_ok()
// is false and the experts of a Flash-Next IQ4_XS pack ran on the AVX-2 rows (iq_avx2.cpp).  Same decode as those: the
// nibbles through the IQ4_NL codebook with one pshufb, |w| as the unsigned operand, w's sign moved onto the activation -
// here as a masked subtract (AVX512BW has no vpsignb).  64 values (two 32-value blocks) per vector instead of 32, and
// vpdpbusd / vpdpwssd fold the multiply and the add.  Per block the integer sums are ggml's; only the order of the float
// additions differs (two blocks share a vector, the 16 lanes are summed at the end), so the rows are not bit-equal to
// the AVX-2 ones, but every token's rows are the same at any group size (#152): nothing a token computes depends on NT.
//
// Nothing here runs before native_expert.cpp's iq4_512_on(): no initializer at namespace scope (the TU is compiled for
// AVX-512, and code the compiler emits for it may use AVX-512 anywhere).
#include "strata/kernels/cpu/iq4_avx512.hpp"
#include "strata/kernels/cpu/expert.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// software prefetch distance in bytes, as the AVX-2 rows (STRATA_IQ_PREFETCH; 0 = off)
int prefetch_distance() {
    static const int d = [] { const char* v = std::getenv("STRATA_IQ_PREFETCH"); return v ? std::atoi(v) : 2048; }();
    return d;
}

// Two 32-value blocks' nibbles (16 bytes each) -> 64 codebook values in block order: lo0 hi0 lo1 hi1 (a block's values
// 0..15 are its low nibbles, 16..31 its high ones, as the activation bytes come).  t holds [b0, b0, b1, b1].
inline __m512i decode64(__m512i t, __m512i values) {
    t = _mm512_mask_srli_epi16(t, 0xFF00FF00u, t, 4);   // the high nibbles down in 128-bit lanes 1 and 3
    return _mm512_shuffle_epi8(values, _mm512_and_si512(t, _mm512_set1_epi8(0x0f)));
}

// ---- IQ4_XS (type 23): 256-value blocks of 136 bytes (f16 d, scales_h u16, scales_l[4], qs[128]), eight 32-value
// sub-blocks with signed 6-bit scales (ls - 32), against Q8_K.  The sub-block scale folds into vpdpwssd's int16 operand
// as in iq_avx2's Fmt32<23>; a pair of sub-blocks shares a vector, the scale vector holds each one's in its half.
template <int NT>
inline void xs_row(const uint8_t* row, int nb, const block_q8_K* const* y, float* res, __m512i values, int pf) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * sizeof(block_iq4_xs);
        if (pf > 0) _mm_prefetch((const char*) blk + pf, _MM_HINT_T0);
        const uint16_t sh = u16(blk + 2);
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int p = 0; p < 4; ++p) {   // sub-blocks 2p and 2p+1
            const __m256i b = _mm256_loadu_si256((const __m256i*) (blk + 8 + 32 * p));
            const __m512i t2 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 3, 2, 1, 0, 1, 0), _mm512_castsi256_si512(b));
            const __m512i q4 = decode64(t2, values);
            const __m512i g = _mm512_abs_epi8(q4);        // |w|, vpdpbusd's / maddubs' unsigned operand
            const __mmask64 neg = _mm512_movepi8_mask(q4);
            const int ls0 = (blk[4 + p] & 0xf) | (((sh >> (4 * p)) & 3) << 4);
            const int ls1 = (blk[4 + p] >> 4) | (((sh >> (4 * p + 2)) & 3) << 4);
            const __m512i sc = _mm512_inserti64x4(_mm512_set1_epi16((short) (ls0 - 32)), _mm256_set1_epi16((short) (ls1 - 32)), 1);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * p));
                const __m512i ys = _mm512_mask_sub_epi8(yv, neg, _mm512_setzero_si512(), yv);
                acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int NT>
void xs_gu(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff, int r0,
           int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const __m512i values = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const int nb = n / QK_K, pf = prefetch_distance();
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        xs_row<NT>(blob + (size_t) r * gu_row, nb, y, g, values, pf);
        xs_row<NT>(blob + up_off + (size_t) r * gu_row, nb, y, u, values, pf);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // native_gu_rows' SwiGLU
    }
}

// ---- IQ4_NL (type 20): 32-value blocks of 18 bytes (f16 d, qs[16]) against Q8_0.  The activations are unpacked once
// per call (their qs contiguous, their scales as [d0 x8 | d1 x8] for each block pair): every row reads them again.
template <int NT>
void nl_rows(const uint8_t* w, size_t row_bytes, int n, const int8_t* aq, const float* ad, float* const* out, int r0,
             int r1) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const int nb = n / QK4_NL, pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m512 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
        for (int ib = 0; ib < nb; ib += 2) {
            const uint8_t* b0 = row + (size_t) ib * sizeof(block_iq4_nl);
            const uint8_t* b1 = b0 + sizeof(block_iq4_nl);
            if (pf > 0) _mm_prefetch((const char*) b0 + pf, _MM_HINT_T0);
            const __m512i t2 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (b0 + 2)))),
                _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (b1 + 2))), 1);
            const __m512i q4 = decode64(t2, values);
            const __m512i g = _mm512_abs_epi8(q4);
            const __mmask64 neg = _mm512_movepi8_mask(q4);
            const __m512 dx = _mm512_insertf32x8(_mm512_set1_ps(h2f(u16(b0))), _mm256_set1_ps(h2f(u16(b1))), 1);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (aq + (size_t) t * n + 32 * ib));
                const __m512i ys = _mm512_mask_sub_epi8(yv, neg, _mm512_setzero_si512(), yv);
                const __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, ys);
                const __m512 s = _mm512_mul_ps(dx, _mm512_loadu_ps(ad + ((size_t) t * nb + ib) * 8));
                accf[t] = _mm512_fmadd_ps(s, _mm512_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = _mm512_reduce_add_ps(accf[t]);
    }
}

}  // namespace

void iq4xs512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                      float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: xs_gu<1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: xs_gu<2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: xs_gu<3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: xs_gu<4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: xs_gu<5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: xs_gu<6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: xs_gu<7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 8)
                if (nt - t0 >= 8) xs_gu<8>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
                else iq4xs512_gu_rows(blob, gu_row, up_off, n, act + t0, nt - t0, ff + t0, r0, r1);
    }
}

void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    if (nt > 8) {
        for (int t0 = 0; t0 < nt; t0 += 8)
            iq4nl512_down_rows(w, row_bytes, n, hq + t0, std::min(8, nt - t0), out + t0, r0, r1);
        return;
    }
    // the activations unpacked: qs contiguous per token, each block's d as 8 lanes (one block pair = 16 floats)
    const int nb = n / QK4_NL;
    thread_local std::vector<int8_t> aq;
    thread_local std::vector<float> ad;
    aq.resize((size_t) nt * n);
    ad.resize((size_t) nt * nb * 8);
    for (int t = 0; t < nt; ++t) {
        const block_q8_0* y = (const block_q8_0*) hq[t];
        for (int ib = 0; ib < nb; ++ib) {
            std::memcpy(aq.data() + (size_t) t * n + 32 * ib, y[ib].qs, 32);
            const float d = h2f(u16((const uint8_t*) &y[ib].d));
            std::fill_n(ad.data() + ((size_t) t * nb + ib) * 8, 8, d);
        }
    }
    switch (nt) {
        case 1: nl_rows<1>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        case 2: nl_rows<2>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        case 3: nl_rows<3>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        case 4: nl_rows<4>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        case 5: nl_rows<5>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        case 6: nl_rows<6>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        case 7: nl_rows<7>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
        default: nl_rows<8>(w, row_bytes, n, aq.data(), ad.data(), out, r0, r1); break;
    }
}

}  // namespace strata::kernels::cpu
