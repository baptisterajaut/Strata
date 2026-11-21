// src/inkling2/ink2_cpu.cpp - CPU expert products for strata-ink2's host-LRU decode (INK2_CPU_MISS=1): fp32 row dots
// over ggml IQ3_S / IQ4_NL / Q8_0 blocks (AVX2 + FMA) and a small spin thread pool.  Built with -mavx2 -mfma -mf16c.
#include <immintrin.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "ink2_iq_tables.h"

namespace {

constexpr int GG_Q8_0 = 8, GG_IQ4_NL = 20, GG_IQ3_S = 21;

inline float f16(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}
inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
// IQ3_S superblock (110 B / 256): d, qs[64], qh[8], signs[32], scales[4].  The grid as 4 floats per entry (8 KB,
// L1-resident) and the 16 sign patterns as masks: 4 weights = one 128-bit load and one xor, fp32 FMA with x.
alignas(64) float g_grid4[512][4];
alignas(64) uint32_t g_sign4[16][4];
struct TabInit {
    TabInit() {
        for (int i = 0; i < 512; ++i)
            for (int j = 0; j < 4; ++j) g_grid4[i][j] = (float) ((k_iq3s_grid[i] >> (8 * j)) & 255);
        for (int n = 0; n < 16; ++n)
            for (int j = 0; j < 4; ++j) g_sign4[n][j] = (n >> j) & 1 ? 0x80000000u : 0u;
    }
} g_tab_init;
float dot_iq3s(const uint8_t* row, const float* x, int K) {
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < K / 256; ++sb) {
        const uint8_t* b = row + (size_t) sb * 110;
        const float d = f16(b);
        for (int ib = 0; ib < 8; ++ib) {
            const float db = d * (float) (1 + 2 * ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15));
            const uint32_t qh = b[66 + ib];
            const uint8_t* qs = b + 2 + 8 * ib;
            const uint8_t* sg = b + 74 + 4 * ib;
            const float* xx = x + sb * 256 + ib * 32;
            __m256 s = _mm256_setzero_ps();
            for (int il = 0; il < 4; ++il) {
                const int i1 = qs[2 * il] | ((qh << (8 - 2 * il)) & 256), i2 = qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256);
                const __m256 w = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(g_grid4[i1])), _mm_load_ps(g_grid4[i2]), 1);
                const __m256 m = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps((const float*) g_sign4[sg[il] & 15])),
                                                      _mm_load_ps((const float*) g_sign4[sg[il] >> 4]), 1);
                s = _mm256_fmadd_ps(_mm256_xor_ps(w, m), _mm256_loadu_ps(xx + 8 * il), s);
            }
            acc = _mm256_fmadd_ps(s, _mm256_set1_ps(db), acc);
        }
    }
    return hsum8(acc);
}
// IQ4_NL block (18 B / 32): d, qs[16] (low nibbles values 0..15, high 16..31); the 16-entry book looked up with two
// 8-entry permutes
float dot_iq4nl(const uint8_t* row, const float* x, int K) {
    __m256 acc = _mm256_setzero_ps();
    alignas(32) float book[16];
    for (int i = 0; i < 16; ++i) book[i] = (float) k_iq4nl_values[i];
    const __m256 b_lo = _mm256_load_ps(book), b_hi = _mm256_load_ps(book + 8);
    for (int bk = 0; bk < K / 32; ++bk) {
        const uint8_t* b = row + (size_t) bk * 18;
        uint64_t q0, q1;
        std::memcpy(&q0, b + 2, 8);
        std::memcpy(&q1, b + 10, 8);
        const __m256i n[4] = {
            _mm256_and_si256(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q0)), _mm256_set1_epi32(15)),
            _mm256_and_si256(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q1)), _mm256_set1_epi32(15)),
            _mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q0)), 4),
            _mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) q1)), 4)};
        __m256 s = _mm256_setzero_ps();
        for (int q = 0; q < 4; ++q) {
            const __m256 lo = _mm256_permutevar8x32_ps(b_lo, n[q]), hi = _mm256_permutevar8x32_ps(b_hi, n[q]);
            const __m256 w = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_slli_epi32(n[q], 28)));
            s = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + bk * 32 + 8 * q), s);
        }
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(f16(b)), acc);
    }
    return hsum8(acc);
}
float dot_q8(const uint8_t* row, const float* x, int K) {
    float acc = 0.0f;
    for (int bk = 0; bk < K / 32; ++bk) {
        const uint8_t* b = row + (size_t) bk * 34;
        float s = 0.0f;
        for (int j = 0; j < 32; ++j) s += (float) (int8_t) b[2 + j] * x[bk * 32 + j];
        acc += f16(b) * s;
    }
    return acc;
}

// The pool's state is one 64-bit word, generation << 32 | next index: a task is claimed by compare-and-swap only while
// the word still carries the claimer's generation, so a worker late from the last run can never take (or skip) an
// index of the next one, nor read its job half-written.  Jobs sit in a ring of 4 by generation.
struct Job { void (*fn)(int, void*) = nullptr; void* ctx = nullptr; int n = 0; };
struct Pool {
    std::vector<std::thread> th;
    std::atomic<uint64_t> state{0};
    std::atomic<int> done{0};
    uint32_t gen = 0;   // the host thread's
    Job job[4];
};
Pool P;

}  // namespace

float ink2_cpu_dot(int type, const uint8_t* row, const float* x, int K) {
    return type == GG_IQ3_S ? dot_iq3s(row, x, K) : type == GG_IQ4_NL ? dot_iq4nl(row, x, K) : dot_q8(row, x, K);
}
static void pool_claim(uint32_t g) {   // run generation g's tasks until none is left (or g is over)
    uint64_t v = P.state.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (v >> 32) != g) return;
        const Job& j = P.job[g & 3];
        const int i = (int) (uint32_t) v;
        if (i >= j.n) return;
        if (P.state.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            j.fn(i, j.ctx);
            P.done.fetch_add(1, std::memory_order_release);
            v = P.state.load(std::memory_order_acquire);
        }
    }
}
void ink2_cpu_pool_start(int workers) {
    for (int w = 0; w < workers; ++w)
        P.th.emplace_back([] {
            uint32_t seen = (uint32_t) (P.state.load() >> 32);
            for (;;) {
                uint32_t g;
                while ((g = (uint32_t) (P.state.load(std::memory_order_acquire) >> 32)) == seen) _mm_pause();
                seen = g;
                pool_claim(g);
            }
        });
    for (auto& t : P.th) t.detach();
}
void ink2_cpu_pool_run(int n, void (*fn)(int, void*), void* ctx) {
    const uint32_t g = ++P.gen;
    P.job[g & 3] = Job{fn, ctx, n};
    P.done.store(0, std::memory_order_relaxed);
    P.state.store((uint64_t) g << 32, std::memory_order_release);
    pool_claim(g);
    while (P.done.load(std::memory_order_acquire) < n) _mm_pause();
}
