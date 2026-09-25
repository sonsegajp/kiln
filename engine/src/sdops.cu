#include "sdops.h"

#include <cmath>

static inline unsigned nb(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// ---------------------------------------------------------------------------
// GroupNorm
// A group holds (C/groups) * P values (up to ~160k in SDXL at 1024^2), so its statistics are reduced
// in two stages: blocks reduce a chunk each to (count, mean, M2), then per group the chunks are
// merged with Chan's formula (numerically stable, no E[x^2] - E[x]^2 cancellation).
// ---------------------------------------------------------------------------
constexpr int GN_CHUNK = 4096;

__global__ void k_gn_partial(float3* __restrict__ part, const float* __restrict__ x, size_t gsize, int chunks) {
    const int g = blockIdx.y, ch = blockIdx.x;
    const float* xg = x + (size_t)g * gsize;
    const size_t i0 = (size_t)ch * GN_CHUNK, i1 = min(gsize, i0 + GN_CHUNK);
    __shared__ float red[256];
    float s = 0.f;
    for (size_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) s += xg[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float n = (float)(i1 - i0), mean = red[0] / n;
    __syncthreads();
    float q = 0.f;
    for (size_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) { float d = xg[i] - mean; q += d * d; }
    red[threadIdx.x] = q;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[(size_t)g * chunks + ch] = make_float3(n, mean, red[0]);
}

__global__ void k_gn_final(float2* __restrict__ stat, const float3* __restrict__ part, int groups, int chunks, float eps) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    double n = 0, mean = 0, m2 = 0;
    for (int c = 0; c < chunks; c++) {
        float3 p = part[(size_t)g * chunks + c];
        double nb = p.x, d = p.y - mean, tot = n + nb;
        mean += d * nb / tot;
        m2 += p.z + d * d * n * nb / tot;
        n = tot;
    }
    stat[g] = make_float2((float)mean, (float)(1.0 / sqrt(m2 / n + eps)));
}

template <bool HALF>
__global__ void k_gn_apply(float* __restrict__ y, __half* __restrict__ y16, const float* __restrict__ x, const float2* __restrict__ stat,
                           const float* __restrict__ gamma, const float* __restrict__ beta, int C, int P, int cpg, bool silu) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * P) return;
    const int c = (int)(i / P);
    const float2 st = stat[c / cpg];
    float v = (x[i] - st.x) * st.y * gamma[c] + beta[c];
    if (silu) v = v / (1.f + __expf(-v));
    if (HALF) y16[i] = __float2half_rn(v);
    else y[i] = v;
}

void group_norm(float* y, __half* y16, const float* x, const float* gamma, const float* beta, int C, int P, int groups, float eps, bool silu) {
    ProfScope ps("groupnorm");
    const size_t gsize = (size_t)(C / groups) * P;
    const int chunks = (int)((gsize + GN_CHUNK - 1) / GN_CHUNK);
    size_t m = G.arena.mark();
    float3* part = (float3*)G.arena.f((size_t)groups * chunks * 3);
    float2* stat = (float2*)G.arena.f((size_t)groups * 2);
    k_gn_partial<<<dim3(chunks, groups), 256, 0, G.stream>>>(part, x, gsize, chunks);
    k_gn_final<<<nb(groups, 64), 64, 0, G.stream>>>(stat, part, groups, chunks, eps);
    const size_t n = (size_t)C * P;
    if (y16) k_gn_apply<true><<<nb(n), 256, 0, G.stream>>>(nullptr, y16, x, stat, gamma, beta, C, P, C / groups, silu);
    else k_gn_apply<false><<<nb(n), 256, 0, G.stream>>>(y, nullptr, x, stat, gamma, beta, C, P, C / groups, silu);
    G.arena.release(m);
}

void group_norm_stats(float2* stat, const float* x, int C, size_t P, int groups, float eps) {
    ProfScope ps("groupnorm");
    const size_t gsize = (size_t)(C / groups) * P;
    const int chunks = (int)((gsize + GN_CHUNK - 1) / GN_CHUNK);
    size_t m = G.arena.mark();
    float3* part = (float3*)G.arena.f((size_t)groups * chunks * 3);
    k_gn_partial<<<dim3(chunks, groups), 256, 0, G.stream>>>(part, x, gsize, chunks);
    k_gn_final<<<nb(groups, 64), 64, 0, G.stream>>>(stat, part, groups, chunks, eps);
    G.arena.release(m);
}

__global__ void k_gn_apply_band(__half* __restrict__ y16, const float* __restrict__ x, size_t cs, const float2* __restrict__ stat,
                                const float* __restrict__ gamma, const float* __restrict__ beta, int C, size_t n, int cpg, bool silu) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * n) return;
    const int c = (int)(i / n);
    const size_t j = i - (size_t)c * n;
    const float2 st = stat[c / cpg];
    float v = (x[(size_t)c * cs + j] - st.x) * st.y * gamma[c] + beta[c];
    if (silu) v = v / (1.f + __expf(-v));
    y16[i] = __float2half_rn(v);
}
void group_norm_apply16(__half* y16, const float* x, size_t cs, const float2* stat, const float* gamma, const float* beta, int C, size_t n,
                        int groups, bool silu) {
    ProfScope ps("groupnorm");
    k_gn_apply_band<<<nb((size_t)C * n), 256, 0, G.stream>>>(y16, x, cs, stat, gamma, beta, C, n, C / groups, silu);
}

__global__ void k_to16_band(__half* __restrict__ y16, const float* __restrict__ x, size_t cs, int C, size_t n, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * n) return;
    const int c = (int)(i / n);
    y16[i] = __float2half_rn(x[(size_t)c * cs + (i - (size_t)c * n)] * scale);
}
void to16_band(__half* y16, const float* x, size_t cs, int C, size_t n, float scale) {
    k_to16_band<<<nb((size_t)C * n), 256, 0, G.stream>>>(y16, x, cs, C, n, scale);
}

// ---------------------------------------------------------------------------
// LayerNorm (one block per row, two passes over the row in registers/L1)
// ---------------------------------------------------------------------------
__global__ void k_layer_norm(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                             int dim, float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* yr = y + (size_t)blockIdx.x * dim;
    __shared__ float red[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float mean = red[0] / dim;
    __syncthreads();
    float q = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = xr[i] - mean; q += d * d; }
    red[threadIdx.x] = q;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * r * w[i] + b[i];
}

void layer_norm(float* y, const float* x, const float* w, const float* b, int rows, int dim, float eps) {
    ProfScope ps("layernorm");
    k_layer_norm<<<rows, 256, 0, G.stream>>>(y, x, w, b, dim, eps);
}

// ---------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------
__global__ void k_geglu(float* __restrict__ y, const float* __restrict__ x, int T, int inner) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * inner) return;
    const size_t t = i / inner, j = i - t * inner;
    const float a = x[t * 2 * inner + j], g = x[t * 2 * inner + inner + j];
    y[i] = a * 0.5f * g * (1.f + erff(g * 0.70710678118654752f));
}
void geglu(float* y, const float* x, int T, int inner) {
    ProfScope ps("geglu");
    k_geglu<<<nb((size_t)T * inner), 256, 0, G.stream>>>(y, x, T, inner);
}

// ---------------------------------------------------------------------------
// small convolutions
// ---------------------------------------------------------------------------
// block = 256 pixels x 32 output channels
template <int CIN>
__global__ void k_conv_small_in(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ b,
                                int Cout, int H, int W) {
    constexpr int K = CIN * 9;
    __shared__ float sw[32][K + 1];
    __shared__ float sb[32];
    const int m0 = blockIdx.y * 32;
    for (int i = threadIdx.x; i < 32 * K; i += blockDim.x) {
        int mm = i / K, k = i - mm * K;
        sw[mm][k] = m0 + mm < Cout ? w[(size_t)(m0 + mm) * K + k] : 0.f;
    }
    if (threadIdx.x < 32) sb[threadIdx.x] = m0 + threadIdx.x < Cout ? b[m0 + threadIdx.x] : 0.f;
    __syncthreads();
    const int P = H * W, p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int y = p / W, x = p - y * W;
    float v[K];
#pragma unroll
    for (int c = 0; c < CIN; c++)
#pragma unroll
        for (int ky = 0; ky < 3; ky++)
#pragma unroll
            for (int kx = 0; kx < 3; kx++) {
                int yy = y + ky - 1, xx = x + kx - 1;
                v[c * 9 + ky * 3 + kx] = ((unsigned)yy < (unsigned)H && (unsigned)xx < (unsigned)W) ? in[((size_t)c * H + yy) * W + xx] : 0.f;
            }
    for (int mm = 0; mm < 32 && m0 + mm < Cout; mm++) {
        float acc = sb[mm];
#pragma unroll
        for (int k = 0; k < K; k++) acc += sw[mm][k] * v[k];
        out[(size_t)(m0 + mm) * P + p] = acc;
    }
}
void conv3x3_small_in(float* out, const float* in, const float* w, const float* b, int Cin, int Cout, int H, int W) {
    dim3 grid(nb((size_t)H * W), (Cout + 31) / 32);
    if (Cin == 3) k_conv_small_in<3><<<grid, 256, 0, G.stream>>>(out, in, w, b, Cout, H, W);
    else if (Cin == 4) k_conv_small_in<4><<<grid, 256, 0, G.stream>>>(out, in, w, b, Cout, H, W);
    else throw std::runtime_error("conv3x3_small_in: Cin must be 3 or 4");
}

// one thread per output pixel, weights [Cin*9][COUT] in shared memory
template <int COUT>
__global__ void k_conv_small_out(float* __restrict__ out, size_t out_cs, const __half* __restrict__ in, int Cin, int Hin, int W, int row_off,
                                 int rows, const float* __restrict__ w, const float* __restrict__ b) {
    extern __shared__ float sw[];
    const int K = Cin * 9;
    for (int i = threadIdx.x; i < K * COUT; i += blockDim.x) sw[(i % K) * COUT + i / K] = w[i];
    __syncthreads();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= rows * W) return;
    const int y = p / W + row_off, x = p % W;
    const size_t cs = (size_t)Hin * W;
    float acc[COUT];
#pragma unroll
    for (int o = 0; o < COUT; o++) acc[o] = b[o];
    for (int c = 0; c < Cin; c++) {
        const __half* src = in + (size_t)c * cs;
#pragma unroll
        for (int ky = 0; ky < 3; ky++) {
            const int yy = y + ky - 1;
            if ((unsigned)yy >= (unsigned)Hin) continue;
#pragma unroll
            for (int kx = 0; kx < 3; kx++) {
                const int xx = x + kx - 1;
                if ((unsigned)xx >= (unsigned)W) continue;
                const float v = __half2float(src[(size_t)yy * W + xx]);
                const float* ww = sw + (c * 9 + ky * 3 + kx) * COUT;
#pragma unroll
                for (int o = 0; o < COUT; o++) acc[o] += ww[o] * v;
            }
        }
    }
#pragma unroll
    for (int o = 0; o < COUT; o++) out[(size_t)o * out_cs + p] = acc[o];
}
void conv3x3_small_out(float* out, size_t out_cs, const __half* in, int Cin, int Hin, int W, int row_off, int rows, const float* w,
                       const float* b, int Cout) {
    const size_t smem = (size_t)Cin * 9 * Cout * 4;
    const unsigned grid = nb((size_t)rows * W, 128);
    if (Cout == 3) k_conv_small_out<3><<<grid, 128, smem, G.stream>>>(out, out_cs, in, Cin, Hin, W, row_off, rows, w, b);
    else if (Cout == 4) k_conv_small_out<4><<<grid, 128, smem, G.stream>>>(out, out_cs, in, Cin, Hin, W, row_off, rows, w, b);
    else throw std::runtime_error("conv3x3_small_out: Cout must be 3 or 4");
}

__global__ void k_conv1x1_small(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ b,
                                int Cin, int Cout, size_t P) {
    const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float v[8];
    for (int c = 0; c < Cin; c++) v[c] = in[(size_t)c * P + p];
    for (int o = 0; o < Cout; o++) {
        float acc = b[o];
        for (int c = 0; c < Cin; c++) acc += w[o * Cin + c] * v[c];
        out[(size_t)o * P + p] = acc;
    }
}
void conv1x1_small(float* out, const float* in, const float* w, const float* b, int Cin, int Cout, size_t P) {
    if (Cin > 8 || Cout > 8) throw std::runtime_error("conv1x1_small: at most 8 channels");
    k_conv1x1_small<<<nb(P), 256, 0, G.stream>>>(out, in, w, b, Cin, Cout, P);
}

__global__ void k_quick_gelu(float* x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v / (1.f + __expf(-1.702f * v)); }
}
void quick_gelu(float* x, size_t n) { k_quick_gelu<<<nb(n), 256, 0, G.stream>>>(x, n); }

__global__ void k_bias_rows(float* y, const float* b, size_t n, int dim) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % dim];
}
void bias_rows(float* y, const float* b, int rows, int dim) {
    k_bias_rows<<<nb((size_t)rows * dim), 256, 0, G.stream>>>(y, b, (size_t)rows * dim, dim);
}

__global__ void k_bias_channels(float* y, const float* b, size_t n, int P) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i / P];
}
void bias_channels(float* y, const float* b, int C, int P) {
    k_bias_channels<<<nb((size_t)C * P), 256, 0, G.stream>>>(y, b, (size_t)C * P, P);
}
void add_channels(float* y, const float* v, int C, int P) { bias_channels(y, v, C, P); }
