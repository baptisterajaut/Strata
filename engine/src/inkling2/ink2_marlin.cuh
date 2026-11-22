// ink2_marlin.cuh - --marlin 1: the NVFP4 experts (layers 0-19) stored in the Marlin layout and multiplied by the
// vendored Marlin MoE kernels (third_party/marlin: vLLM's csrc/moe/marlin_moe_wna16, Apache-2.0, see its README.md).
// Included by ink2_engine.cu.  One layout per pool slot, so with the flag every GPU reader of an NVFP4 expert goes
// through Marlin: prompt chunks, --spec verify blocks and the one-token decode.  fp16 activations in, fp32
// accumulation, fp16 products out (the default path keeps fp32 products).
// Scales: Marlin stores the e4m3 block scale * 2^7 as S0E5M3, exact for every normal e4m3 value; the checkpoint's
// scale bytes are all in [76, 126] (measured over layers 0-19), so nothing is lost.  The per-expert global scale
// stays an fp32 multiplier (* 2^7, GSCALE_MUL).
// CBA: resident checkpoint experts only (no --vram-reserve-mib, no GGUF): the offload paths copy experts from host
//      memory in the checkpoint layout (fetches, promotions, the PCIe share, prompt staging) and the CPU experts read
//      it; upgrade path: repack_part after each of those copies, CPU-computed rows left out of k_m_plan.
// CBA: the MXFP8 experts (layers 20-23, 4 of 48 layer passes) keep their layout and kernels: Marlin's MXFP8 kernels
//      exist for bf16 activations only (8-bit mantissa); upgrade path: instantiate them and add a bf16 lane.
#if !defined(STRATA_USE_HIP) && defined(STRATA_INK2_MARLIN)

struct MarlinState {
    float* c_tmp = nullptr;
    int* locks = nullptr;
    int *sorted = nullptr, *eids = nullptr, *npad = nullptr;   // prompt / verify: blocks of rows per expert
    int *dsorted = nullptr, *dnpad = nullptr;                  // decode: 8 blocks of one row (the experts are E.ids)
    half *dx = nullptr, *dgu = nullptr, *dact = nullptr, *dey = nullptr;   // decode fp16 scratch
    const float* gs13[NL] = {};
    const float* gs2[NL] = {};
    uint8_t* stage = nullptr;   // load: one layer in the checkpoint layout
};
MarlinState MS;

// a layer's experts, checkpoint layout on the host -> Marlin layout in its pool range
void marlin_load_layer(const uint8_t* lay, size_t bytes, uint8_t* dst) {
    if (!MS.stage) CK(cudaMalloc(&MS.stage, (size_t) NE * NV_BYTES));
    CK(cudaMemcpy(MS.stage, lay, bytes, cudaMemcpyHostToDevice));
    for (int e = 0; e < NE; ++e) {
        const size_t o = (size_t) e * NV_BYTES;
        ink_marlin::repack_part(MS.stage + o, dst + o, 2 * FF, H, cudaStreamPerThread);
        ink_marlin::repack_part(MS.stage + o + NV_A, dst + o + NV_A, H, FF, cudaStreamPerThread);
    }
    CK(cudaStreamSynchronize(cudaStreamPerThread));
}

int marlin_block(int T) { return T <= SMALL_T ? 8 : T <= 256 ? 16 : T < 2048 ? 32 : 64; }   // 3090 bench (mbench)

void marlin_alloc() {
    if (MS.stage) { CK(cudaFree(MS.stage)); MS.stage = nullptr; }
    for (int l = 0; l < MXFP8_FIRST; ++l) {
        vec a(NE), b(NE);
        for (int e = 0; e < NE; ++e) {
            a[e] = E.hf13[(size_t) l * NE + e] * ink_marlin::GSCALE_MUL;
            b[e] = E.hf2[(size_t) l * NE + e] * ink_marlin::GSCALE_MUL;
        }
        MS.gs13[l] = dvec(a);
        MS.gs2[l] = dvec(b);
    }
    const int T = std::max(PF.TP, 1), nr = T * TOPK;
    size_t ct = 0;
    for (int b : {8, 16, 32, 64}) ct = std::max(ct, ink_marlin::c_tmp_floats(b, nr + NE * b, H));
    MS.c_tmp = dfloats(ct);
    MS.locks = (int*) dalloc(ink_marlin::LOCKS * 4);
    CK(cudaMemset(MS.locks, 0, ink_marlin::LOCKS * 4));
    MS.sorted = (int*) dalloc(((size_t) nr + (size_t) NE * 64) * 4);
    MS.eids = (int*) dalloc(((size_t) nr / 8 + NE) * 4);
    MS.npad = (int*) dalloc(16);
    std::vector<int> ds(TOPK * 8, TOPK);
    for (int j = 0; j < TOPK; ++j) ds[j * 8] = j;
    const int np = TOPK * 8;
    MS.dsorted = (int*) upload(ds.data(), ds.size() * 4);
    MS.dnpad = (int*) upload(&np, 4);
    MS.dx = (half*) dalloc((size_t) H * 2);
    MS.dgu = (half*) dalloc((size_t) TOPK * 2 * FF * 2);
    MS.dact = (half*) dalloc((size_t) TOPK * FF * 2);
    MS.dey = (half*) dalloc((size_t) TOPK * H * 2);
}

__global__ void k_mh_from_f32(const float* __restrict__ x, half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
// SoftcappedSwiGLU, fp16 gu [nr][2FF] -> fp16 act [nr][FF]
__global__ void k_mh_swiglu(const half* __restrict__ gu, half* __restrict__ act, int nr) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nr * FF) return;
    const size_t k = i / FF, j = i % FF;
    const float2 p = __half22float2(*reinterpret_cast<const half2*>(gu + k * 2 * FF + 2 * j));
    act[i] = __float2half_rn(swiglu1(p.x, p.y));
}
// k_moe_tail on fp16 expert outputs
__global__ void k_mh_tail(const half* __restrict__ ey, const float* __restrict__ wts, float* __restrict__ hist,
                          const float* __restrict__ w, const float* __restrict__ r, float* __restrict__ out,
                          const int* d_pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float a = 0.0f;
    for (int k = 0; k < TOPK; ++k) a += __half2float(ey[(size_t) k * H + i]) * wts[k];
    out[i] = r[i] + short_conv1(a, hist, w, H, i, *d_pos);
}
// kb_combine on fp16 expert outputs
__global__ void k_mh_combine(const half* __restrict__ ey, const int* __restrict__ slot, const float* __restrict__ wts,
                             float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    float a = 0.0f;
    for (int k = 0; k < TOPK; ++k) a += __half2float(ey[(size_t) slot[t * TOPK + k] * H + i]) * wts[t * TOPK + k];
    out[(size_t) t * H + i] = a;
}
// Marlin's block list out of kb_moe_plan's expert-sorted rows (eoff / ecnt): per expert, blocks of `block` rows
// (sorted[b][i] = row, or nr past its last row), eids[b] = the expert, *npad = blocks * block.  One block, NE threads.
__global__ void k_m_plan(const int* __restrict__ eoff, const int* __restrict__ ecnt, int block, int nr, int* sorted,
                         int* eids, int* npad) {
    __shared__ int boff[NE + 1];
    const int e = threadIdx.x;
    if (e == 0) {
        int o = 0;
        for (int j = 0; j < NE; ++j) { boff[j] = o; o += (ecnt[j] + block - 1) / block; }
        boff[NE] = o;
        *npad = o * block;
    }
    __syncthreads();
    const int nb = boff[e + 1] - boff[e], c = ecnt[e], o = eoff[e];
    for (int b = 0; b < nb; ++b) {
        eids[boff[e] + b] = e;
        for (int i = 0; i < block; ++i) sorted[(size_t) (boff[e] + b) * block + i] = b * block + i < c ? o + b * block + i : nr;
    }
}

void marlin_gemm(const void* A, void* C, const uint8_t* const* tab, const float* gs, const int* sorted, const int* eids,
                 const int* npad, int block, int top_k, int m, int N, int K) {
    ink_marlin::Gemm g;
    g.A = A; g.C = C; g.c_tmp = MS.c_tmp; g.base = E.pool_nv; g.tab = tab; g.gscale = gs;
    g.sorted_ids = sorted; g.expert_ids = eids; g.num_padded = npad; g.locks = MS.locks;
    g.block = block; g.top_k = top_k; g.m = m; g.N = N; g.K = K;
    ink_marlin::gemm(g, cudaStreamPerThread);
}

// one token (moe(), after the router): E.ids / E.wts -> out
void marlin_moe(const Layer& Y, const float* x, const float* r, float* mhist, const float* mlp_conv, float* out) {
    const Experts& X = Y.ex;
    const int l = int(&Y - E.L);
    k_mh_from_f32<<<(H + 255) / 256, 256>>>(x, MS.dx, H);
    marlin_gemm(MS.dx, MS.dgu, X.tA, MS.gs13[l], MS.dsorted, E.ids, MS.dnpad, 8, TOPK, 1, 2 * FF, H);
    k_mh_swiglu<<<(TOPK * FF + 255) / 256, 256>>>(MS.dgu, MS.dact, TOPK);
    marlin_gemm(MS.dact, MS.dey, X.tB, MS.gs2[l], MS.dsorted, E.ids, MS.dnpad, 8, 1, TOPK, H, FF);
    k_mh_tail<<<(H + 255) / 256, 256>>>(MS.dey, E.wts, mhist, mlp_conv, r, out, E.d_pos);
}

// T tokens (moe_b(), after kb_moe_plan): the fp16 buffers reuse the shared region's phases, gu / ey [nr][2FF] halves
// at its start, the gathered input [nr][H] then act [nr][FF] halves in the act range
void marlin_moe_b(const Layer& Y, const float* x, float* out, int T) {
    const Experts& X = Y.ex;
    const int l = int(&Y - E.L), nr = T * TOPK, block = marlin_block(T);
    half *ah = reinterpret_cast<half*>(PF.act), *gh = reinterpret_cast<half*>(PF.gu);
    kb_gather_h<<<nr, 256>>>(x, PF.rows, ah);
    k_m_plan<<<1, NE>>>(PF.eoff, PF.ecnt, block, nr, MS.sorted, MS.eids, MS.npad);
    marlin_gemm(ah, gh, X.tA, MS.gs13[l], MS.sorted, MS.eids, MS.npad, block, 1, nr, 2 * FF, H);
    k_mh_swiglu<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(gh, ah, nr);
    marlin_gemm(ah, gh, X.tB, MS.gs2[l], MS.sorted, MS.eids, MS.npad, block, 1, nr, H, FF);
    k_mh_combine<<<dim3(H / 256, T), 256>>>(gh, PF.slot, PF.wts, out);
}
#else
void marlin_load_layer(const uint8_t*, size_t, uint8_t*) {}
void marlin_alloc() {}
void marlin_moe(const Layer&, const float*, const float*, float*, const float*, float*) {}
void marlin_moe_b(const Layer&, const float*, float*, int) {}
#endif
