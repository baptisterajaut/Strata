// Fused relative-bias flash attention for the strata-ink2 prefill with an int8 K/V cache (--kv i8 / i8r, --tc 1), gfx11 (RDNA3) WMMA.
// Opt-in: INK2_RELFA=1 (HIP builds only); without it the prefill attention is kb_attn_tc as before.
//
// What kb_rel_logits + kb_attn_tc compute, in one kernel:
//   * K / V: int8 codes [cap][KVD] + one fp16 scale per 64 values [cap][KVD / 64], ring row = position % cap
//   * the same fp16 operands as kb_attn_tc: q, K, V rounded to fp16, fp32 accumulation, fp16 probabilities; the dequantisation
//     (int8 * scale) is rounded once to fp16, the same value as kb_attn_tc's fp32 product rounded to fp16
//   * bias(h, t, dist) = sum_r rs[t][h][r] * rbT[dist][r] for dist < extent, computed in the kernel: no [T][NQ][EXTENT] table, which
//     is what keeps a 1M-token prompt in memory; the sink logit starts the softmax (m = sink, l = 1) as in kb_attn_tc
//   * sliding-window layers: window > 0 limits the keys to the last `window` positions (distance < window)
//   s(h,t,j) = scale * q_t . k_j + bias(h,t,pos_t - pos_j), softmax over {sink, s(j)}; out = sum_j p_j v_j / (1 + sum_j p_j)
//
// Transposed formulation: S^T = K Q^T and O^T = V^T P^T.  A WMMA C tile then holds one query per lane (column lc, the keys 2v + lh in
// its 8 registers), so the softmax state (running max, sum, position, sink, the bias row) is one scalar or 16 registers per lane, the
// row max needs one cross-row exchange, and P goes from the C layout into the B operand of the next product through a register
// exchange with lane (lane ^ 16) instead of a round trip through LDS.  The two halves of a 16-key slab are ordered [even keys | odd
// keys] in that B operand, so V^T is stored with the same key permutation inside every 16-key slab (slot e < 8: key 2e, slot 8 + e:
// key 2e + 1).  Only the order of the sums (softmax denominator, keys inside a WMMA) differs from kb_attn_tc.
// The verify blocks (T <= SMALL_T) and the other KV formats keep kb_attn_tc.
#pragma once
#ifdef STRATA_USE_HIP

namespace inkfa {

constexpr int D = 128, D_REL = 16, QT = 16, Q_BLOCK = 32, BLOCKS_HD = D / Q_BLOCK;
constexpr int K_STR = D + 8;   // K tile row stride in halves (16-byte rows, bank spread)

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v16h_a16 __attribute__((ext_vector_type(16), aligned(16)));   // LDS rows are only 16-byte aligned
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

#if defined(__GFX11__)
#define INKFA_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))
#else
#define INKFA_WMMA(a, b, c) (c)   // device pass of another target: never launched (the engine is built for gfx1100)
#endif

struct Params {
    const float* q; int q_s1;                    // q[t * q_s1 + h * D + d]  (the engine's P buffer)
    const int8_t* K; const int8_t* V;            // int8 codes [cap][NKV * D]
    const uint16_t* ksc; const uint16_t* vsc;    // fp16 scale bits [cap][NKV * D / 64]
    int cap;
    const float* R; int R_s1;                    // rs[t * R_s1 + h * D_REL + r]  (clipped, in P)
    const float* rbT;                            // [extent][D_REL]  (rel_proj transposed)
    const float* sink; int sink_s1;              // sink[t * sink_s1 + h]
    float* out;                                  // [T][H][D]
    int T, pos0, window, H, Hkv, extent;         // row t of the chunk is position pos0 + t
    float scale;
};

// int8 q in a byte u8: (u8 ^ 0x80) | 0x6400 is the fp16 number 1024 + 128 + q, so one packed subtract of 1152 gives q exactly and
// one packed multiply gives q * d with a single rounding (= the fp32 product rounded to fp16)
__device__ __forceinline__ void deq4(uint32_t w, half2_t d2, uint32_t& o0, uint32_t& o1) {
    const uint32_t x = w ^ 0x80808080u;
    const uint32_t a = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u);   // bytes {x0, 0x64, x1, 0x64}
    const uint32_t b = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u);   // bytes {x2, 0x64, x3, 0x64}
    const half2_t c = {(_Float16) 1152.0f, (_Float16) 1152.0f};
    const half2_t ha = (__builtin_bit_cast(half2_t, a) - c) * d2;
    const half2_t hb = (__builtin_bit_cast(half2_t, b) - c) * d2;
    o0 = __builtin_bit_cast(uint32_t, ha);
    o1 = __builtin_bit_cast(uint32_t, hb);
}

typedef uint32_t v8u __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint32_t dpp_swap_pairs2(uint32_t x) {   // value of lane (lane ^ 2): quad_perm [2, 3, 0, 1]
    return (uint32_t) __builtin_amdgcn_mov_dpp((int) x, 0x4E, 0xF, 0xF, true);
}
__device__ __forceinline__ uint32_t swap_rows16(uint32_t x) {   // value of lane (lane ^ 16): the two 16-lane rows of a wave32 trade places
    return (uint32_t) __builtin_amdgcn_permlanex16(0, (int) x, 0x76543210u, 0xFEDCBA98u, false, false);
}

template <int NG, int NQT, int TK>   // NG q heads per KV head, NQT query tiles of 16 per block, TK keys per LDS tile (32 or 64)
__global__ void __launch_bounds__(32 * NG * NQT) relpos_fa_i8(const Params p) {
    constexpr int NWAVE = NG * NQT, NTHR = 32 * NWAVE, VS = TK + 8;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    _Float16* Klds = (_Float16*) smem_raw;                         // [TK][K_STR]
    _Float16* Vt = Klds + TK * K_STR;                          // [D][VS]  dim-major, keys permuted inside each 16-key slab

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int lc = lane & 15;   // WMMA column: the query of this lane
    const int lh = lane >> 4;   // which half of the 16 keys of a sub-tile this lane C registers hold (key 2v + lh)
    const int g = wave % NG;
    const int tq = wave / NG;
    const int hkv = blockIdx.y, h = hkv * NG + g;
    const int q0 = (blockIdx.x * NQT + tq) * QT;
    const int n_kv = p.pos0 + p.T;
    const int t_l = q0 + lc;   // the query row of this lane

    // B-operand fragments of Q^T: lane = query lc, element e = dim dk * 16 + e
    v16h qf[D / 16];
#pragma unroll
    for (int dk = 0; dk < D / 16; ++dk) {
        v16h f;
#pragma unroll
        for (int e = 0; e < 16; ++e) f[e] = (t_l < p.T) ? (_Float16) p.q[(int64_t) t_l * p.q_s1 + (int64_t) h * D + dk * 16 + e] : (_Float16) 0;
        qf[dk] = f;
    }
    const int qpos = (t_l < p.T) ? p.pos0 + t_l : -1;
    const float sink = (t_l < p.T) ? p.sink[(int64_t) t_l * p.sink_s1 + h] : 0.0f;
    float rr[D_REL];   // the bias row of this query and head (clipped rel scores), constant for every key
#pragma unroll
    for (int c = 0; c < D_REL; ++c) rr[c] = (t_l < p.T) ? p.R[(int64_t) t_l * p.R_s1 + h * D_REL + c] : 0.0f;

    const int blk_q0 = blockIdx.x * NQT * QT;
    const int blk_qlast = min(blk_q0 + NQT * QT - 1, p.T - 1);
    const int kv_end = min(n_kv, p.pos0 + blk_qlast + 1);
    const int n_tiles = (kv_end + TK - 1) / TK;
    const int kt_first = p.window > 0 ? max(0, p.pos0 + blk_q0 - p.window + 1) / TK : 0;
    const int qmin_w = (q0 < p.T) ? p.pos0 + q0 : 0x3fffffff;   // wave-level causal bound (positions ascend with t)

    v8f o[D / 16];   // O^T: lane = query lc, register v of block db = dim db * 16 + 2v + lh
#pragma unroll
    for (int db = 0; db < D / 16; ++db) o[db] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m = sink;                      // the sink logit opens the softmax
    float l = (lh == 0) ? 1.0f : 0.0f;   // one of the two lanes of a query counts the sink; the two partial sums are added at the end

    for (int kt = kt_first; kt < n_tiles; ++kt) {
        const int key0 = kt * TK;
        __syncthreads();   // previous tile fully consumed before LDS is overwritten

        // ---- cooperative int8 -> fp16 tile load (as in relpos_fa_i8, K rows as 16-byte vectors).  V^T: the keys of a 16-key slab are stored
        // [even keys | odd keys]; two keys four apart (4a, 4a + 2 and 4a + 1, 4a + 3) are adjacent slots, so lanes (l, l ^ 2) jointly write the dwords.
        for (int u = tid; u < 2 * BLOCKS_HD * TK; u += NTHR) {
            const int grp = u / TK, jj = u - grp * TK, T = grp / BLOCKS_HD, b = grp - T * BLOCKS_HD;
            const int key = key0 + jj;
            const bool valid = key < n_kv;
            uint32_t q[8];
            half2_t d2;
            if (valid) {
                const int row = key % p.cap;
                const int8_t* cp = (T == 0 ? p.K : p.V) + (size_t) row * (p.Hkv * D) + hkv * D + b * Q_BLOCK;   // 32-byte aligned
                const uint4 u0 = *(const uint4*) cp, u1 = *((const uint4*) cp + 1);
                q[0] = u0.x; q[1] = u0.y; q[2] = u0.z; q[3] = u0.w; q[4] = u1.x; q[5] = u1.y; q[6] = u1.z; q[7] = u1.w;
                const uint32_t sb = (T == 0 ? p.ksc : p.vsc)[(size_t) row * (p.Hkv * D / 64) + hkv * (D / 64) + (b >> 1)];
                d2 = __builtin_bit_cast(half2_t, sb * 0x00010001u);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) q[j] = 0;
                d2 = __builtin_bit_cast(half2_t, 0u);
            }
            uint32_t o16[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) deq4(q[j], d2, o16[2 * j], o16[2 * j + 1]);

            if (T == 0) {
                uint4* dst = (uint4*) (Klds + jj * K_STR + b * Q_BLOCK);
#pragma unroll
                for (int k = 0; k < 4; ++k) dst[k] = make_uint4(o16[4 * k], o16[4 * k + 1], o16[4 * k + 2], o16[4 * k + 3]);
            } else {
                const int par = (lane >> 1) & 1;   // 0: the lower key of the pair (4a or 4a + 1), 1: the upper one (+ 2)
                uint32_t keep[8], send[8];
#pragma unroll
                for (int mm = 0; mm < 4; ++mm) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const uint32_t ev = o16[2 * (2 * mm) + e], od = o16[2 * (2 * mm + 1) + e];
                        keep[2 * mm + e] = par ? od : ev;
                        send[2 * mm + e] = par ? ev : od;
                    }
                }
                uint32_t recv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) recv[i] = dpp_swap_pairs2(send[i]);
                const uint32_t sel_lo = par ? 0x01000504u : 0x05040100u;
                const uint32_t sel_hi = par ? 0x03020706u : 0x07060302u;
                // slot of the pair first key inside its slab: 4a -> 2a, 4a + 1 -> 8 + 2a (a = (jj >> 2) & 3)
                _Float16* vcol = Vt + (jj & ~15) + ((jj & 1) << 3) + (((jj >> 2) & 3) << 1);
#pragma unroll
                for (int mm = 0; mm < 4; ++mm) {
                    const int j = 2 * mm + par;
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int dim = b * Q_BLOCK + 4 * j + 2 * e;
                        *(uint32_t*) (vcol + (dim + 0) * VS) = __builtin_amdgcn_perm(recv[2 * mm + e], keep[2 * mm + e], sel_lo);
                        *(uint32_t*) (vcol + (dim + 1) * VS) = __builtin_amdgcn_perm(recv[2 * mm + e], keep[2 * mm + e], sel_hi);
                    }
                }
            }
        }
        __syncthreads();

        // ---- S^T = K Q^T: C register v of sub-tile kb = score of key kb * 16 + 2v + lh for this lane query
        v8f s[TK / 16];
#pragma unroll
        for (int kb = 0; kb < TK / 16; ++kb) s[kb] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int dk = 0; dk < D / 16; ++dk) {
#pragma unroll
            for (int kb = 0; kb < TK / 16; ++kb) {
                const v16h ak = *(const v16h_a16*) (Klds + (kb * 16 + lc) * K_STR + dk * 16);
                s[kb] = INKFA_WMMA(ak, qf[dk], s[kb]);
            }
        }

        // ---- scale, bias, mask.  Beyond `extent` the bias is zero; with a window every tile needs the mask.
        const bool partial = key0 + TK > n_kv;
        const bool masked = (key0 + TK - 1 > qmin_w) || partial || p.window > 0;
        const bool near = (key0 + TK - 1 > qmin_w - p.extent) || p.window > 0;
        float mx = -INFINITY;
#pragma unroll
        for (int kb = 0; kb < TK / 16; ++kb) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = key0 + kb * 16 + 2 * v + lh;
                float bt = 0.0f;
                bool inf = false;
                if (near || masked) {
                    const int dist = qpos - key;
                    if (dist < 0 || key >= n_kv || (p.window > 0 && dist >= p.window)) {
                        inf = true;
                    } else if (dist < p.extent) {
                        const float4* rb = (const float4*) (p.rbT + (int64_t) dist * D_REL);
                        float bsum = 0.0f;
#pragma unroll
                        for (int q4 = 0; q4 < D_REL / 4; ++q4) {
                            const float4 r4 = rb[q4];
                            bsum = fmaf(r4.x, rr[4 * q4 + 0], bsum);
                            bsum = fmaf(r4.y, rr[4 * q4 + 1], bsum);
                            bsum = fmaf(r4.z, rr[4 * q4 + 2], bsum);
                            bsum = fmaf(r4.w, rr[4 * q4 + 3], bsum);
                        }
                        bt = bsum;
                    }
                }
                const float sv = inf ? -INFINITY : s[kb][v] * p.scale + bt;
                s[kb][v] = sv;
                mx = fmaxf(mx, sv);
            }
        }

        // ---- online softmax: one query per lane, its 32 keys of the tile in registers; the partner lane (lane ^ 16) holds the other 32 keys
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float m_new = fmaxf(m, mx);
        const float alpha = __expf(m - m_new);
        m = m_new;
        float lsum = 0.0f;
        v16h pa[TK / 16];   // B operand of P^T per 16-key slab: elements 0-7 the even keys, 8-15 the odd keys
#pragma unroll
        for (int kb = 0; kb < TK / 16; ++kb) {
            uint32_t pk[4], qk[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p0 = __expf(s[kb][2 * i] - m_new), p1 = __expf(s[kb][2 * i + 1] - m_new);
                lsum += p0;
                lsum += p1;
                const half2_t h2 = {(_Float16) p0, (_Float16) p1};
                pk[i] = __builtin_bit_cast(uint32_t, h2);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) qk[i] = swap_rows16(pk[i]);
            v8u w;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                w[i] = lh ? qk[i] : pk[i];
                w[4 + i] = lh ? pk[i] : qk[i];
            }
            pa[kb] = __builtin_bit_cast(v16h, w);
        }
        l = l * alpha + lsum;
#pragma unroll
        for (int db = 0; db < D / 16; ++db)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[db][v] *= alpha;

        // ---- O^T += V^T P^T
#pragma unroll
        for (int kk = 0; kk < TK / 16; ++kk) {
#pragma unroll
            for (int db = 0; db < D / 16; ++db) {
                const v16h vb = *(const v16h_a16*) (Vt + (db * 16 + lc) * VS + kk * 16);
                o[db] = INKFA_WMMA(vb, pa[kk], o[db]);
            }
        }
    }

    // ---- finalise
    const float inv = 1.0f / (l + __shfl_xor(l, 16, 32));
    if (t_l < p.T) {
        float* dst = p.out + ((int64_t) t_l * p.H + h) * D;
#pragma unroll
        for (int db = 0; db < D / 16; ++db)
#pragma unroll
            for (int v = 0; v < 8; ++v) dst[db * 16 + 2 * v + lh] = o[db][v] * inv;
    }
}

template <int NG, int NQT, int TK>
constexpr size_t smem_bytes() {
    return (size_t) (TK * K_STR + D * (TK + 8)) * sizeof(_Float16);
}

}  // namespace inkfa
#endif  // STRATA_USE_HIP
