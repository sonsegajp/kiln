// Anima LoRA training: forward with block checkpoints, backward through every block, LoRA gradients (see train.h).
#include "train.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

static const int D = 2048, NH = 16, DH = 128, FF = 4 * D;
static const float EPS = 1e-6f;

const char* const AnimaTrainer::kShort[NT] = {"sa_q", "sa_k", "sa_v", "sa_o", "ca_q", "ca_k", "ca_v", "ca_o", "mlp1", "mlp2"};
const char* const AnimaTrainer::kModule[NT] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                                               "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
                                               "mlp.layer1", "mlp.layer2"};

static inline unsigned nb(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// ---------------------------------------------------------------------------------------------- kernels
__device__ __forceinline__ float warp_sum_t(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// sum over a 256-thread block (every thread gets the total)
__device__ float block_sum_t(float v) {
    __shared__ float red[8];
    __shared__ float total;
    v = warp_sum_t(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += red[w];
        total = t;
    }
    __syncthreads();
    float r = total;
    __syncthreads();
    return r;
}

// backward of y = LN(x) (1 + scale) + shift (no affine): dx += rstd (g - mean(g) - xhat mean(g xhat)), g = dy (1 + scale)
__global__ void k_ln_mod_bwd(float* dx, const float* dy, const float* x, const float* scale, int dim, float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    const float* gr = dy + (size_t)blockIdx.x * dim;
    float* dr = dx + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    const float mean = block_sum_t(s) / dim;
    float v = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = xr[i] - mean; v += d * d; }
    const float rstd = rsqrtf(block_sum_t(v) / dim + eps);
    float sg = 0.f, sgx = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float g = gr[i] * (1.f + scale[i]), xh = (xr[i] - mean) * rstd;
        sg += g;
        sgx += g * xh;
    }
    const float mg = block_sum_t(sg) / dim, mgx = block_sum_t(sgx) / dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float g = gr[i] * (1.f + scale[i]), xh = (xr[i] - mean) * rstd;
        dr[i] += rstd * (g - mg - xh * mgx);
    }
}

// backward of y = x rsqrt(mean(x^2) + eps) w over rows of dim (one warp per row): dx = r (g - x r^2 mean(g x)), g = dy w
__global__ void k_rms_bwd(float* dx, const float* dy, const float* x, const float* w, int rows, int dim, float eps) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * dim;
    const float* yr = dy + (size_t)row * dim;
    float ss = 0.f, sgx = 0.f;
    for (int i = lane; i < dim; i += 32) { ss += xr[i] * xr[i]; sgx += yr[i] * w[i] * xr[i]; }
    ss = warp_sum_t(ss);
    sgx = warp_sum_t(sgx);
    const float r = rsqrtf(ss / dim + eps), k = r * r * r * sgx / dim;
    float* dr = dx + (size_t)row * dim;
    for (int i = lane; i < dim; i += 32) dr[i] = r * yr[i] * w[i] - xr[i] * k;
}

// dh *= gelu'(h0) (exact erf GELU)
__global__ void k_gelu_bwd(float* dh, const float* h0, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = h0[i];
    const float cdf = 0.5f * (1.f + erff(x * 0.70710678118654752f));
    const float pdf = 0.39894228040143268f * expf(-0.5f * x * x);
    dh[i] *= cdf + x * pdf;
}

__global__ void k_mul_cols(float* y, const float* x, const float* g, size_t n, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * g[i % dim];
}

__global__ void k_neg(float* y, const float* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = -x[i];
}

// softmax over rows of n scores with npad extra keys of score 0 (zero context padding), in place
__global__ void k_softmax_pad(float* s, int n, int npad) {
    float* r = s + (size_t)blockIdx.x * n;
    float m = npad > 0 ? 0.f : -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
    __shared__ float red[8];
    __shared__ float bc;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float t = red[0]; for (int w = 1; w < (int)(blockDim.x >> 5); w++) t = fmaxf(t, red[w]); bc = t; }
    __syncthreads();
    m = bc;
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float e = expf(r[i] - m); r[i] = e; z += e; }
    z = block_sum_t(z) + npad * expf(-m);
    const float inv = 1.f / z;
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] *= inv;
}

// dS = P (dP - sum_j P dP) per row (padding keys have dP = 0 and drop out of the sum)
__global__ void k_dsoftmax(float* dp, const float* p, int n) {
    const float* pr = p + (size_t)blockIdx.x * n;
    float* dr = dp + (size_t)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += pr[i] * dr[i];
    s = block_sum_t(s);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dr[i] = pr[i] * (dr[i] - s);
}

// x_t = (1 - sigma) latent + sigma noise; and the loss pieces: diff = pred - (noise - latent)
__global__ void k_mix(float* y, const float* a, const float* b, float s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (1.f - s) * a[i] + s * b[i];
}
__global__ void k_mse_grad(float* g, float* acc, const float* pred, const float* lat, const float* noise, size_t n, float k) {
    float l = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float d = pred[i] - (noise[i] - lat[i]);
        g[i] = k * d;
        l += d * d;
    }
    l = warp_sum_t(l);
    if ((threadIdx.x & 31) == 0) atomicAdd(acc, l);
}

// patchify for the trainer: latent [16, Hl, Wl] -> tokens [T, 68] (channel 16 = zero padding mask), as dit.cu
__global__ void k_patchify_t(float* tok, const float* lat, int Hl, int Wl) {
    int Wp = Wl / 2, T = (Hl / 2) * Wp;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 68) return;
    int f = i % 68, t = (int)(i / 68);
    int c = f / 4, m = (f / 2) % 2, nn = f % 2;
    int y = (t / Wp) * 2 + m, x = (t % Wp) * 2 + nn;
    tok[i] = c < 16 ? lat[((size_t)c * Hl + y) * Wl + x] : 0.f;
}
// unpatchify [T, 64] -> [16, Hl, Wl] and its transpose (gather the latent gradient back into token order)
__global__ void k_unpatchify_t(float* lat, const float* tok, int Hl, int Wl, int inverse) {
    int Wp = Wl / 2, T = (Hl / 2) * Wp;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 64) return;
    int f = i % 64, t = (int)(i / 64);
    int p1 = f / 32, p2 = (f / 16) % 2, c = f % 16;
    int y = (t / Wp) * 2 + p1, x = (t % Wp) * 2 + p2;
    size_t li = ((size_t)c * Hl + y) * Wl + x;
    if (inverse) const_cast<float*>(tok)[i] = lat[li];
    else lat[li] = tok[i];
}

// ---------------------------------------------------------------------------------------------- linear + LoRA
static const float* wf32(const Weight& w) {
    Weight t = w;
    if (t.on_host) t.p = (uint16_t*)stage_weight(t.p, t.host, (size_t)t.numel() * 2);
    return weight_f32(t);
}

namespace {
struct Ctx {
    AnimaTrainer& tr;
    int b;
    const float* mo;  // this block's modulation [9 D]
    const float* qn[4];  // fp32 norm weights: self q, self k, cross q, cross k [DH]
};

// y = x W^T + s (x A^T) B^T
void lin_fwd(float* y, const float* x, int T, const Weight& W, const AnimaTrainer::Lora& l, int r, float s) {
    linear(y, x, T, W);
    ProfScope ps("lora");
    const size_t m = G.arena.mark();
    float* xa = G.arena.f((size_t)T * r);
    gemm(false, true, T, r, l.in, 1.f, x, l.in, l.A, l.in, 0.f, xa, r);
    gemm(false, true, T, l.out, r, s, xa, r, l.B, r, 1.f, y, l.out);
    G.arena.release(m);
}

// LoRA gradients (accumulated) and, when dx is given, dx (+)= dy W + s (dy B) A
void lin_bwd(float* dx, bool acc, const float* dy, const float* x, int T, const Weight& W, AnimaTrainer::Lora& l, int r, float s) {
    ProfScope ps("bwd_linear");
    const size_t m = G.arena.mark();
    float* xa = G.arena.f((size_t)T * r);
    float* dyb = G.arena.f((size_t)T * r);
    gemm(false, true, T, r, l.in, 1.f, x, l.in, l.A, l.in, 0.f, xa, r);       // x A^T
    gemm(false, false, T, r, l.out, 1.f, dy, l.out, l.B, r, 0.f, dyb, r);     // dy B
    gemm(true, false, l.out, r, T, s, dy, l.out, xa, r, 1.f, l.dB, r);        // dB += s dy^T (x A^T)
    gemm(true, false, r, l.in, T, s, dyb, r, x, l.in, 1.f, l.dA, l.in);       // dA += s (dy B)^T x
    if (dx) {
        gemm(false, false, T, l.in, l.out, 1.f, dy, l.out, wf32(W), l.in, acc ? 1.f : 0.f, dx, l.in);
        gemm(false, false, T, l.in, r, s, dyb, r, l.A, l.in, 1.f, dx, l.in);
    }
    G.arena.release(m);
}

// exact attention backward, heads batched as far as the arena allows: q [Tq, D], k, v [Tk, D] (rows of NH x DH),
// dout [Tq, D] -> dq, dk, dv (written). npad zero keys with score 0 join the softmax.
void attention_bwd(float* dq, float* dk, float* dv, const float* q, const float* k, const float* v, const float* dout, int Tq, int Tk, int npad) {
    ProfScope ps("bwd_attention");
    const float scale = 1.f / sqrtf((float)DH), one = 1.f, zero = 0.f;
    const size_t per_head = (size_t)Tq * Tk;
    const size_t fr = G.arena.free_bytes(), budget = fr > ((size_t)64 << 20) ? fr - ((size_t)64 << 20) : 0;
    int hc = (int)std::max<size_t>(1, std::min<size_t>(NH, budget / (per_head * 8)));
    const size_t m = G.arena.mark();
    float* P = G.arena.f(hc * per_head);
    float* dP = G.arena.f(hc * per_head);
    for (int h0 = 0; h0 < NH; h0 += hc) {
        const int nh = std::min(hc, NH - h0);
        const size_t ho = (size_t)h0 * DH;
        // S = scale Q K^T, then P = softmax (column-major: S^T [Tk, Tq] = K Q^T)
        CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, Tk, Tq, DH, &scale, k + ho, D, DH, q + ho, D, DH, &zero, P, Tk, per_head, nh));
        k_softmax_pad<<<(unsigned)(Tq * nh), 256, 0, G.stream>>>(P, Tk, npad);
        // dP = dO V^T
        CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, Tk, Tq, DH, &one, v + ho, D, DH, dout + ho, D, DH, &zero, dP, Tk, per_head, nh));
        // dV = P^T dO   (column-major: dV^T [DH, Tk] = dO^T [DH, Tq] P^T... as dO_cm (DH x Tq) * P_cm^T (Tq x Tk))
        CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_N, CUBLAS_OP_T, DH, Tk, Tq, &one, dout + ho, D, DH, P, Tk, per_head, &zero, dv + ho, D, DH, nh));
        k_dsoftmax<<<(unsigned)(Tq * nh), 256, 0, G.stream>>>(dP, P, Tk);
        // dQ = scale dS K   (column-major: K_cm (DH x Tk) * dS_cm (Tk x Tq))
        CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, DH, Tq, Tk, &scale, k + ho, D, DH, dP, Tk, per_head, &zero, dq + ho, D, DH, nh));
        // dK = scale dS^T Q (column-major: Q_cm (DH x Tq) * dS_cm^T (Tq x Tk))
        CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_N, CUBLAS_OP_T, DH, Tk, Tq, &scale, q + ho, D, DH, dP, Tk, per_head, &zero, dk + ho, D, DH, nh));
    }
    G.arena.release(m);
}

// what a block's backward needs from its forward
struct Saved {
    float *n1, *q0, *k0, *q, *k, *v, *a1, *x1, *n2, *qc0, *qc, *kc0, *kc, *vc, *a2, *x2, *n3, *h0;
    void alloc(int T, int Lk) {
        for (float** p : {&n1, &q0, &k0, &q, &k, &v, &a1, &x1, &n2, &qc0, &qc, &a2, &x2, &n3}) *p = G.arena.f((size_t)T * D);
        for (float** p : {&kc0, &kc, &vc}) *p = G.arena.f((size_t)std::max(Lk, 1) * D);
        h0 = G.arena.f((size_t)T * FF);
    }
};

struct Step {
    const float *rcos, *rsin, *rsin_neg;
    const float* ctx;  // [Lk, 1024] real context rows
    int Lk, npad, T;
};

void block_fwd(Ctx& c, const Step& st, float* X, Saved& s) {
    auto& bl = c.tr.dit.blocks[c.b];
    auto& L = c.tr.lora[c.b];
    const int T = st.T, r = c.tr.rank;
    const float sc = c.tr.scale;
    const float* mo = c.mo;
    const size_t m = G.arena.mark();
    float* y = G.arena.f((size_t)T * D);
    float* h = G.arena.f((size_t)T * FF);
    // self-attention
    layernorm_mod(s.n1, X, mo + D, mo, T, D, EPS);
    lin_fwd(s.q0, s.n1, T, bl.sa_q, L[0], r, sc);
    lin_fwd(s.k0, s.n1, T, bl.sa_k, L[1], r, sc);
    lin_fwd(s.v, s.n1, T, bl.sa_v, L[2], r, sc);
    rmsnorm(s.q, s.q0, &bl.sa_qn, T * NH, DH, EPS);
    rmsnorm(s.k, s.k0, &bl.sa_kn, T * NH, DH, EPS);
    rope_half(s.q, T, NH, DH, st.rcos, st.rsin);
    rope_half(s.k, T, NH, DH, st.rcos, st.rsin);
    attention(s.a1, D, s.q, D, s.k, D, s.v, D, T, T, NH, DH, false);
    lin_fwd(y, s.a1, T, bl.sa_o, L[3], r, sc);
    add_gated(X, y, mo + 2 * D, T, D);
    CK(cudaMemcpyAsync(s.x1, X, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, G.stream));
    // cross-attention (K, V from the context, LoRA'd too)
    layernorm_mod(s.n2, X, mo + 4 * D, mo + 3 * D, T, D, EPS);
    lin_fwd(s.qc0, s.n2, T, bl.ca_q, L[4], r, sc);
    rmsnorm(s.qc, s.qc0, &bl.ca_qn, T * NH, DH, EPS);
    lin_fwd(s.kc0, st.ctx, st.Lk, bl.ca_k, L[5], r, sc);
    rmsnorm(s.kc, s.kc0, &bl.ca_kn, st.Lk * NH, DH, EPS);
    lin_fwd(s.vc, st.ctx, st.Lk, bl.ca_v, L[6], r, sc);
    attention(s.a2, D, s.qc, D, s.kc, D, s.vc, D, T, st.Lk, NH, DH, false, st.npad);
    lin_fwd(y, s.a2, T, bl.ca_o, L[7], r, sc);
    add_gated(X, y, mo + 5 * D, T, D);
    CK(cudaMemcpyAsync(s.x2, X, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, G.stream));
    // MLP
    layernorm_mod(s.n3, X, mo + 7 * D, mo + 6 * D, T, D, EPS);
    lin_fwd(s.h0, s.n3, T, bl.mlp1, L[8], r, sc);
    CK(cudaMemcpyAsync(h, s.h0, (size_t)T * FF * 4, cudaMemcpyDeviceToDevice, G.stream));
    gelu(h, (size_t)T * FF);
    lin_fwd(y, h, T, bl.mlp2, L[9], r, sc);
    add_gated(X, y, mo + 8 * D, T, D);
    G.arena.release(m);
}

// dX: gradient of the block's output in, of its input out; X0: the block's input
void block_bwd(Ctx& c, const Step& st, float* dX, const float* X0, const Saved& s) {
    auto& bl = c.tr.dit.blocks[c.b];
    auto& L = c.tr.lora[c.b];
    const int T = st.T, Lk = st.Lk, r = c.tr.rank;
    const float sc = c.tr.scale;
    const float* mo = c.mo;
    const size_t TD = (size_t)T * D, TF = (size_t)T * FF;
    const size_t m = G.arena.mark();
    float* g = G.arena.f(TD);   // gated output gradient
    float* dn = G.arena.f(TD);  // gradient into a norm's output
    float* dh = G.arena.f(TF);
    float* h = G.arena.f(TF);
    float* da = G.arena.f(TD);
    float* dq = G.arena.f(TD);
    float* dk = G.arena.f(std::max<size_t>(TD, (size_t)Lk * D));
    float* dv = G.arena.f(std::max<size_t>(TD, (size_t)Lk * D));
    float* dq0 = G.arena.f(TD);
    float* dk0 = G.arena.f(std::max<size_t>(TD, (size_t)Lk * D));

    // MLP: X3 = X2 + g3 (gelu(n3 W1) W2)
    k_mul_cols<<<nb(TD), 256, 0, G.stream>>>(g, dX, mo + 8 * D, TD, D);
    CK(cudaMemcpyAsync(h, s.h0, TF * 4, cudaMemcpyDeviceToDevice, G.stream));
    gelu(h, TF);
    lin_bwd(dh, false, g, h, T, bl.mlp2, L[9], r, sc);
    k_gelu_bwd<<<nb(TF), 256, 0, G.stream>>>(dh, s.h0, TF);
    lin_bwd(dn, false, dh, s.n3, T, bl.mlp1, L[8], r, sc);
    k_ln_mod_bwd<<<T, 256, 0, G.stream>>>(dX, dn, s.x2, mo + 7 * D, D, EPS);
    // cross-attention: X2 = X1 + g2 (attn(norm(n2 Wq), norm(ctx Wk), ctx Wv) Wo)
    k_mul_cols<<<nb(TD), 256, 0, G.stream>>>(g, dX, mo + 5 * D, TD, D);
    lin_bwd(da, false, g, s.a2, T, bl.ca_o, L[7], r, sc);
    attention_bwd(dq, dk, dv, s.qc, s.kc, s.vc, da, T, Lk, st.npad);
    k_rms_bwd<<<nb((size_t)T * NH, 8), 256, 0, G.stream>>>(dq0, dq, s.qc0, c.qn[2], T * NH, DH, EPS);
    k_rms_bwd<<<nb((size_t)Lk * NH, 8), 256, 0, G.stream>>>(dk0, dk, s.kc0, c.qn[3], Lk * NH, DH, EPS);
    lin_bwd(nullptr, false, dk0, st.ctx, Lk, bl.ca_k, L[5], r, sc);
    lin_bwd(nullptr, false, dv, st.ctx, Lk, bl.ca_v, L[6], r, sc);
    lin_bwd(dn, false, dq0, s.n2, T, bl.ca_q, L[4], r, sc);
    k_ln_mod_bwd<<<T, 256, 0, G.stream>>>(dX, dn, s.x1, mo + 4 * D, D, EPS);
    // self-attention: X1 = X0 + g1 (attn(rope(norm(n1 Wq)), rope(norm(n1 Wk)), n1 Wv) Wo)
    k_mul_cols<<<nb(TD), 256, 0, G.stream>>>(g, dX, mo + 2 * D, TD, D);
    lin_bwd(da, false, g, s.a1, T, bl.sa_o, L[3], r, sc);
    attention_bwd(dq, dk, dv, s.q, s.k, s.v, da, T, T, 0);
    rope_half(dq, T, NH, DH, st.rcos, st.rsin_neg);  // a rotation's transpose turns the other way
    rope_half(dk, T, NH, DH, st.rcos, st.rsin_neg);
    k_rms_bwd<<<nb((size_t)T * NH, 8), 256, 0, G.stream>>>(dq0, dq, s.q0, c.qn[0], T * NH, DH, EPS);
    k_rms_bwd<<<nb((size_t)T * NH, 8), 256, 0, G.stream>>>(dk0, dk, s.k0, c.qn[1], T * NH, DH, EPS);
    lin_bwd(dn, false, dq0, s.n1, T, bl.sa_q, L[0], r, sc);
    lin_bwd(dn, true, dk0, s.n1, T, bl.sa_k, L[1], r, sc);
    lin_bwd(dn, true, dv, s.n1, T, bl.sa_v, L[2], r, sc);
    k_ln_mod_bwd<<<T, 256, 0, G.stream>>>(dX, dn, X0, mo + D, D, EPS);
    G.arena.release(m);
}

// fp32 copies of a block's four q/k norm weights
void norm_weights(Dit::Block& bl, float* out) {
    const Weight* w[4] = {&bl.sa_qn, &bl.sa_kn, &bl.ca_qn, &bl.ca_kn};
    for (int i = 0; i < 4; i++) CK(cudaMemcpyAsync(out + i * DH, wf32(*w[i]), DH * 4, cudaMemcpyDeviceToDevice, G.stream));
}
}  // namespace

// ---------------------------------------------------------------------------------------------- trainer
Weight& AnimaTrainer::weight(int b, int t) {
    auto& bl = dit.blocks[b];
    Weight* w[NT] = {&bl.sa_q, &bl.sa_k, &bl.sa_v, &bl.sa_o, &bl.ca_q, &bl.ca_k, &bl.ca_v, &bl.ca_o, &bl.mlp1, &bl.mlp2};
    return *w[t];
}

void AnimaTrainer::init(int r, float a, unsigned long long seed) {
    release();
    rank = r;
    alpha = a > 0.f ? a : (float)r;
    scale = alpha / (float)r;
    lora.assign(dit.blocks.size(), {});
    n = 0;
    for (size_t b = 0; b < dit.blocks.size(); b++)
        for (int t = 0; t < NT; t++) {
            Weight& w = weight((int)b, t);
            lora[b][t].in = (int)w.cols;
            lora[b][t].out = (int)w.rows;
            n += (size_t)r * (w.cols + w.rows);
        }
    CK(cudaMalloc(&params, n * 4));
    CK(cudaMalloc(&grads, n * 4));
    std::vector<float> h(n, 0.f);
    std::mt19937_64 rng(seed);
    size_t o = 0;
    for (size_t b = 0; b < dit.blocks.size(); b++)
        for (int t = 0; t < NT; t++) {
            Lora& l = lora[b][t];
            // kaiming_uniform_(a = sqrt 5) on [r, in]: bound = sqrt(6 / ((1 + 5) in)) = 1 / sqrt(in)
            std::uniform_real_distribution<float> u(-1.f / sqrtf((float)l.in), 1.f / sqrtf((float)l.in));
            for (size_t i = 0; i < (size_t)r * l.in; i++) h[o + i] = u(rng);
            l.A = params + o;
            l.dA = grads + o;
            o += (size_t)r * l.in;
            l.B = params + o;  // zeros
            l.dB = grads + o;
            o += (size_t)l.out * r;
        }
    CK(cudaMemcpy(params, h.data(), n * 4, cudaMemcpyHostToDevice));
    zero_grad();
}

size_t AnimaTrainer::param_count(int r) {
    size_t k = 0;
    for (size_t b = 0; b < dit.blocks.size(); b++)
        for (int t = 0; t < NT; t++) k += (size_t)r * (weight((int)b, t).cols + weight((int)b, t).rows);
    return k;
}

void AnimaTrainer::release() {
    if (params) cudaFree(params);
    if (grads) cudaFree(grads);
    params = grads = nullptr;
    lora.clear();
    n = 0;
}

void AnimaTrainer::zero_grad() { if (grads) CK(cudaMemsetAsync(grads, 0, n * 4, G.stream)); }

float AnimaTrainer::step(const float* latent, const float* noise, int Hl, int Wl, float sigma, const Context& c, float grad_scale, int nblocks) {
    if (Hl % 2 || Wl % 2) throw std::runtime_error("train: latent size must be even");
    for (auto& [nm, w] : dit.named) if (w->lora) throw std::runtime_error("train: unload inference LoRAs first");
    const int NB = nblocks > 0 ? std::min<int>(nblocks, (int)dit.blocks.size()) : (int)dit.blocks.size();
    const int Hp = Hl / 2, Wp = Wl / 2, T = Hp * Wp;
    const size_t TD = (size_t)T * D, nl = (size_t)16 * Hl * Wl;
    const size_t m0 = G.arena.mark();
    float* mods = G.arena.f(Dit::mods_elems(dit.blocks.size()));
    float* rope = G.arena.f((size_t)3 * T * (DH / 2));
    dit.step_tables(sigma, Hp, Wp, mods, rope);
    float* rsin_neg = rope + (size_t)2 * T * (DH / 2);
    k_neg<<<nb((size_t)T * (DH / 2)), 256, 0, G.stream>>>(rsin_neg, rope + (size_t)T * (DH / 2), (size_t)T * (DH / 2));
    Step st{rope, rope + (size_t)T * (DH / 2), rsin_neg, c.p, c.real, c.len - c.real, T};
    float* qn = G.arena.f((size_t)NB * 4 * DH);
    for (int b = 0; b < NB; b++) norm_weights(dit.blocks[b], qn + (size_t)b * 4 * DH);

    // forward, keeping each block's input
    float* xt = G.arena.f(nl);
    k_mix<<<nb(nl), 256, 0, G.stream>>>(xt, latent, noise, sigma, nl);
    float* tok = G.arena.f((size_t)T * 68);
    k_patchify_t<<<nb((size_t)T * 68), 256, 0, G.stream>>>(tok, xt, Hl, Wl);
    float* Xs = G.arena.f((size_t)NB * TD);  // inputs of blocks 0..NB-1
    float* X = G.arena.f(TD);
    linear(X, tok, T, dit.x_embed);
    {
        const size_t m = G.arena.mark();
        Saved s;
        s.alloc(T, c.real);
        for (int b = 0; b < NB; b++) {
            CK(cudaMemcpyAsync(Xs + b * TD, X, TD * 4, cudaMemcpyDeviceToDevice, G.stream));
            Ctx cx{*this, b, mods + (size_t)b * 9 * D, {qn + b * 4 * DH, qn + b * 4 * DH + DH, qn + b * 4 * DH + 2 * DH, qn + b * 4 * DH + 3 * DH}};
            block_fwd(cx, st, X, s);
        }
        G.arena.release(m);
    }
    // final layer + loss
    const float* fmod = mods + dit.blocks.size() * 9 * D;
    float* N = G.arena.f(TD);
    layernorm_mod(N, X, fmod + D, fmod, T, D, EPS);
    float* out = G.arena.f((size_t)T * 64);
    linear(out, N, T, dit.final_lin);
    float* pred = G.arena.f(nl);
    k_unpatchify_t<<<nb((size_t)T * 64), 256, 0, G.stream>>>(pred, out, Hl, Wl, 0);
    float* gpred = G.arena.f(nl);
    float* acc = G.arena.f(1);
    CK(cudaMemsetAsync(acc, 0, 4, G.stream));
    k_mse_grad<<<120, 256, 0, G.stream>>>(gpred, acc, pred, latent, noise, nl, 2.f * grad_scale / (float)nl);
    k_unpatchify_t<<<nb((size_t)T * 64), 256, 0, G.stream>>>(gpred, out, Hl, Wl, 1);  // out <- d loss / d tokens
    float* dX = G.arena.f(TD);
    gemm(false, false, T, D, 64, 1.f, out, 64, wf32(dit.final_lin), D, 0.f, N, D);  // dN = dtok W
    fill(dX, 0.f, TD);
    k_ln_mod_bwd<<<T, 256, 0, G.stream>>>(dX, N, X, fmod + D, D, EPS);
    // backward through the blocks, recomputing each from its input
    {
        const size_t m = G.arena.mark();
        Saved s;
        s.alloc(T, c.real);
        for (int b = NB - 1; b >= 0; b--) {
            Ctx cx{*this, b, mods + (size_t)b * 9 * D, {qn + b * 4 * DH, qn + b * 4 * DH + DH, qn + b * 4 * DH + 2 * DH, qn + b * 4 * DH + 3 * DH}};
            CK(cudaMemcpyAsync(X, Xs + b * TD, TD * 4, cudaMemcpyDeviceToDevice, G.stream));
            block_fwd(cx, st, X, s);
            block_bwd(cx, st, dX, Xs + b * TD, s);
        }
        G.arena.release(m);
    }
    float loss = 0.f;
    CK(cudaMemcpy(&loss, acc, 4, cudaMemcpyDeviceToHost));
    G.arena.release(m0);
    return loss / (float)nl;
}

void AnimaTrainer::predict(float* v, const float* x, int Hl, int Wl, float sigma, const Context& c) {
    if (Hl % 2 || Wl % 2) throw std::runtime_error("predict: latent size must be even");
    const int Hp = Hl / 2, Wp = Wl / 2, T = Hp * Wp;
    const size_t TD = (size_t)T * D;
    const size_t m0 = G.arena.mark();
    float* mods = G.arena.f(Dit::mods_elems(dit.blocks.size()));
    float* rope = G.arena.f((size_t)2 * T * (DH / 2));
    dit.step_tables(sigma, Hp, Wp, mods, rope);
    Step st{rope, rope + (size_t)T * (DH / 2), rope + (size_t)T * (DH / 2), c.p, c.real, c.len - c.real, T};
    float* tok = G.arena.f((size_t)T * 68);
    k_patchify_t<<<nb((size_t)T * 68), 256, 0, G.stream>>>(tok, x, Hl, Wl);
    float* X = G.arena.f(TD);
    linear(X, tok, T, dit.x_embed);
    {
        const size_t m = G.arena.mark();
        Saved s;
        s.alloc(T, c.real);
        for (size_t b = 0; b < dit.blocks.size(); b++) {
            Ctx cx{*this, (int)b, mods + b * 9 * D, {nullptr, nullptr, nullptr, nullptr}};
            block_fwd(cx, st, X, s);
        }
        G.arena.release(m);
    }
    const float* fmod = mods + dit.blocks.size() * 9 * D;
    float* N = G.arena.f(TD);
    layernorm_mod(N, X, fmod + D, fmod, T, D, EPS);
    linear(tok, N, T, dit.final_lin);
    k_unpatchify_t<<<nb((size_t)T * 64), 256, 0, G.stream>>>(v, tok, Hl, Wl, 0);
    G.arena.release(m0);
}

void AnimaTrainer::test_block(int b, float* X, const float* dY, float* dX, int Hp, int Wp, float t, const Context& c) {
    const int T = Hp * Wp;
    const size_t m0 = G.arena.mark();
    float* mods = G.arena.f(Dit::mods_elems(dit.blocks.size()));
    float* rope = G.arena.f((size_t)3 * T * (DH / 2));
    dit.step_tables(t, Hp, Wp, mods, rope);
    float* rsin_neg = rope + (size_t)2 * T * (DH / 2);
    k_neg<<<nb((size_t)T * (DH / 2)), 256, 0, G.stream>>>(rsin_neg, rope + (size_t)T * (DH / 2), (size_t)T * (DH / 2));
    Step st{rope, rope + (size_t)T * (DH / 2), rsin_neg, c.p, c.real, c.len - c.real, T};
    float* qn = G.arena.f(4 * DH);
    norm_weights(dit.blocks[b], qn);
    Ctx cx{*this, b, mods + (size_t)b * 9 * D, {qn, qn + DH, qn + 2 * DH, qn + 3 * DH}};
    float* X0 = G.arena.f((size_t)T * D);
    CK(cudaMemcpyAsync(X0, X, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, G.stream));
    Saved s;
    s.alloc(T, c.real);
    block_fwd(cx, st, X, s);
    CK(cudaMemcpyAsync(dX, dY, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, G.stream));
    block_bwd(cx, st, dX, X0, s);
    gpu_sync();
    G.arena.release(m0);
}

// ---------------------------------------------------------------------------------------------- saving
static uint16_t to_f16(float f) {
    __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

void AnimaTrainer::save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& metadata) const {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), params, n * 4, cudaMemcpyDeviceToHost));
    struct Entry { std::string name; std::vector<int64_t> shape; std::vector<uint16_t> data; };
    std::vector<Entry> es;
    size_t o = 0;
    for (size_t b = 0; b < lora.size(); b++)
        for (int t = 0; t < NT; t++) {
            const Lora& l = lora[b][t];
            std::string mod = kModule[t];
            for (char& ch : mod) if (ch == '.') ch = '_';
            const std::string base = "lora_unet_blocks_" + std::to_string(b) + "_" + mod;
            Entry a{base + ".lora_down.weight", {rank, l.in}, {}}, bb{base + ".lora_up.weight", {l.out, rank}, {}}, al{base + ".alpha", {}, {to_f16(alpha)}};
            for (size_t i = 0; i < (size_t)rank * l.in; i++) a.data.push_back(to_f16(h[o + i]));
            o += (size_t)rank * l.in;
            for (size_t i = 0; i < (size_t)l.out * rank; i++) bb.data.push_back(to_f16(h[o + i]));
            o += (size_t)l.out * rank;
            es.push_back(std::move(a));
            es.push_back(std::move(bb));
            es.push_back(std::move(al));
        }
    auto esc = [](const std::string& s) { std::string r; for (char ch : s) { if (ch == '"' || ch == '\\') r += '\\'; r += ch; } return r; };
    std::ostringstream hj;
    hj << "{\"__metadata__\":{";
    for (size_t i = 0; i < metadata.size(); i++) hj << (i ? "," : "") << "\"" << esc(metadata[i].first) << "\":\"" << esc(metadata[i].second) << "\"";
    hj << "}";
    size_t off = 0;
    for (auto& e : es) {
        hj << ",\"" << e.name << "\":{\"dtype\":\"F16\",\"shape\":[";
        for (size_t i = 0; i < e.shape.size(); i++) hj << (i ? "," : "") << e.shape[i];
        hj << "],\"data_offsets\":[" << off << "," << off + e.data.size() * 2 << "]}";
        off += e.data.size() * 2;
    }
    hj << "}";
    std::string header = hj.str();
    while (header.size() % 8) header += ' ';
    std::ofstream f(path + ".part", std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const uint64_t hl = header.size();
    f.write((const char*)&hl, 8);
    f.write(header.data(), header.size());
    for (auto& e : es) f.write((const char*)e.data.data(), e.data.size() * 2);
    f.close();
    std::remove(path.c_str());
    if (std::rename((path + ".part").c_str(), path.c_str())) throw std::runtime_error("cannot write " + path);
}
