// fp16x2 convolutions for GPUs without tensor cores (Turing TU11x: HFMA2 runs at twice the fp32 rate),
// used by the VAEs and the SDXL UNet.
//
// Implicit GEMM: out[Cout, P] = W[Cout, K] * X[K, P], K = taps x Cin, with the im2col matrix X gathered
// on the fly into shared memory (never materialised). Every 16-deep k tile is 16 channels of one tap,
// so the gather is one bounds check per tap plus eight strided loads, no div/mod per element; tiles
// run channel-chunk major (all taps of 16 channels back to back) so the shifted re-reads hit L1.
// Operands are fp16: activations arrive already converted (rms16 / to16 below), weights are converted
// and reordered per call into a small arena slab ([tap][Cin][Cout]). Products accumulate in fp16
// pairs that are flushed into fp32 every 32 k (the hgemm.cuh scheme). The epilogue fuses scale, bias
// and the residual add (out may alias res: each element is read and written by the same thread).
// Nothing data-dependent is involved (no split-K, no atomics), so a pixel's result does not depend on
// the tile or band it lands in.
//
// Modes: M3  3x3 stride 1 pad 1                     (resblock convs)
//        M1  1x1                                    (shortcuts, attention qkv/proj, attention GEMMs)
//        MS2 3x3 stride 2, zero pad only right/bottom (encoder downsample)
//        MS2P 3x3 stride 2, pad 1                    (UNet downsample)
//        MUP nearest-2x upsample + 3x3 pad 1 (decoder upsample), computed as four 2x2 convs on
//            the low-res grid, one per output parity, whose kernels are sums of the 3x3 taps
//            (2.25x fewer MACs than convolving the upsampled image; the same function).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

#include "gpu.h"

namespace vc {

constexpr int BK = 16, FLUSH = 2;  // k tile; fp16 partials are flushed every FLUSH*BK = 32 k
enum Mode { M3 = 0, M1 = 1, MS2 = 2, MUP = 3, MS2P = 4 };  // MS2P: 3x3 stride 2, pad 1 on every side (UNet downsample)

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

struct Args {
    float* out;              // fp32 [Cout, out_cs]
    const float* res;        // fp32, same layout as out, or null; may alias out
    const __half* in;        // fp16 [Cin, in_cs], pixels H x W
    const __half* w;         // fp16 [taps][Cin][Cout] (MUP: [4 parities][4 taps][Cin][Cout])
    const uint16_t* bias;    // bf16 [Cout] or null
    const float* bias32;     // fp32 [Cout] or null (used when bias is null)
    int Cin, Cout, H, W;     // input
    int Wo, P;               // compute grid: P = Ho*Wo pixels (MUP: the low-res grid, = H x W)
    int row_off;             // input row of output row 0 (a band's halo rows above it); 0 for whole images
    int in_cs, out_cs;       // channel strides (elements)
    float scale;             // accumulators are multiplied by this before bias/residual
    int vec;                 // out/res rows allow float4 access (non-MUP only)
};

template <int MODE> __device__ __forceinline__ void tap_pos(int tap, int oy, int ox, int par, int& iy, int& ix) {
    if (MODE == M3) { iy = oy + tap / 3 - 1; ix = ox + tap % 3 - 1; }
    else if (MODE == MS2) { iy = 2 * oy + tap / 3; ix = 2 * ox + tap % 3; }
    else if (MODE == M1) { iy = oy; ix = ox; }
    else if (MODE == MS2P) { iy = 2 * oy + tap / 3 - 1; ix = 2 * ox + tap % 3 - 1; }
    else { iy = oy + (par >> 1) - 1 + (tap >> 1); ix = ox + (par & 1) - 1 + (tap & 1); }
}

// Tile BM x BNX x 16 (BM = 16*TM, BNX = 8*NTX), 16*NTX threads, each TM output channels x 8 consecutive pixels.
template <int TM, int MODE, int NTX = 16, int MINB = (TM <= 6 ? 2 : 1)>
__global__ void __launch_bounds__(16 * NTX, MINB) k_conv(const Args a) {
    constexpr int NTH = 16 * NTX, BNX = 8 * NTX;
    constexpr int BM = 16 * TM, NT = MODE == M1 ? 1 : MODE == MUP ? 4 : 9, CPR = BM / 8;
    constexpr int ACH = BK * CPR, AIT = (ACH + NTH - 1) / NTH;  // 8-half weight chunks per tile, per thread
    __shared__ __align__(16) __half2 As[2][BK][BM];   // weights as (w, w) pairs
    __shared__ __align__(16) __half Bs[2][BK][BNX];   // gathered activations
    const int tid = threadIdx.x, tx = tid % NTX, ty = tid / NTX;
    const int par = MODE == MUP ? (int)(blockIdx.x & 3) : 0;
    const int n0 = (MODE == MUP ? (int)(blockIdx.x >> 2) : (int)blockIdx.x) * BNX, m0 = blockIdx.y * BM;
    const int Cin = a.Cin, Cout = a.Cout, H = a.H, W = a.W;
    const __half* __restrict__ wb = a.w + (size_t)par * NT * Cin * Cout;
    const __half* __restrict__ in = a.in;
    const size_t ics = (size_t)a.in_cs;

    // A staging: 8-half chunks tid + r*NTH of the BK x BM tile (a warp reads consecutive output channels).
    // B staging: pixel b_n, channels b_k + 2q (a warp reads 32 consecutive pixels of one channel).
    const int b_n = tid % BNX, b_k = tid / BNX;
    int oy = 0, ox = 0;
    const bool pv = n0 + b_n < a.P;
    if (pv) { int p = n0 + b_n; oy = p / a.Wo; ox = p - oy * a.Wo; }

    const int nct = Cin / BK, tiles = NT * nct;
    uint4 ra[AIT];
    __half rb[8];
    const __half hz = __float2half(0.f);

    auto load = [&](int t) {
        const int cc = t / NT, tap = t - cc * NT, c0 = cc * BK;  // channel-chunk major: the taps of a chunk hit L1
#pragma unroll
        for (int r = 0; r < AIT; r++) {
            const int e = tid + r * NTH;
            if (e < ACH) ra[r] = __ldg((const uint4*)(wb + ((size_t)tap * Cin + c0 + e / CPR) * Cout + m0 + (e % CPR) * 8));
        }
        int iy, ix;
        tap_pos<MODE>(tap, oy, ox, par, iy, ix);
        iy += a.row_off;
        const bool v = pv && (unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W;
        const __half* src = in + (size_t)(c0 + b_k) * ics + (v ? (size_t)iy * W + ix : 0);
#pragma unroll
        for (int q = 0; q < 8; q++) rb[q] = v ? src[(size_t)(2 * q) * ics] : hz;
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int r = 0; r < AIT; r++) {
            const int e = tid + r * NTH;
            if (e < ACH) {
                const __half* h = (const __half*)&ra[r];
                __half2 d[8];
#pragma unroll
                for (int i = 0; i < 8; i++) d[i] = __half2half2(h[i]);
                *(uint4*)&As[buf][e / CPR][(e % CPR) * 8] = *(const uint4*)&d[0];
                *(uint4*)&As[buf][e / CPR][(e % CPR) * 8 + 4] = *(const uint4*)&d[4];
            }
        }
#pragma unroll
        for (int q = 0; q < 8; q++) Bs[buf][b_k + 2 * q][b_n] = rb[q];
    };

    float acc[TM][8];
    __half2 part[TM][4];
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) part[i][j] = __float2half2_rn(0.f);
    }

    load(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < tiles; t++) {
        const int buf = t & 1;
        if (t + 1 < tiles) load(t + 1);
#pragma unroll
        for (int k = 0; k < BK; k++) {
            __half2 a2[TM];
            if constexpr (TM % 4 == 0) {
#pragma unroll
                for (int v = 0; v < TM / 4; v++) {
                    uint4 av = *(const uint4*)&As[buf][k][ty * TM + v * 4];
                    a2[v * 4 + 0] = *(__half2*)&av.x; a2[v * 4 + 1] = *(__half2*)&av.y;
                    a2[v * 4 + 2] = *(__half2*)&av.z; a2[v * 4 + 3] = *(__half2*)&av.w;
                }
            } else {
#pragma unroll
                for (int v = 0; v < TM / 2; v++) {
                    uint2 av = *(const uint2*)&As[buf][k][ty * TM + v * 2];
                    a2[v * 2 + 0] = *(__half2*)&av.x; a2[v * 2 + 1] = *(__half2*)&av.y;
                }
            }
            uint4 bv = *(const uint4*)&Bs[buf][k][tx * 8];
            __half2 b2[4] = {*(__half2*)&bv.x, *(__half2*)&bv.y, *(__half2*)&bv.z, *(__half2*)&bv.w};
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) part[i][j] = __hfma2(a2[i], b2[j], part[i][j]);
        }
        if ((t + 1) % FLUSH == 0 || t + 1 == tiles) {
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    float2 f = __half22float2(part[i][j]);
                    acc[i][2 * j] += f.x;
                    acc[i][2 * j + 1] += f.y;
                    part[i][j] = __float2half2_rn(0.f);
                }
        }
        if (t + 1 < tiles) {
            store(buf ^ 1);
            __syncthreads();
        }
    }

    const int p0 = n0 + tx * 8;
    if (p0 >= a.P) return;
    const float sc = a.scale;
    const bool full = MODE != MUP && a.vec && p0 + 8 <= a.P;
#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int m = m0 + ty * TM + i;
        const float bb = a.bias ? bf(a.bias[m]) : a.bias32 ? a.bias32[m] : 0.f;
        float* orow = a.out + (size_t)m * a.out_cs;
        const float* rrow = a.res ? a.res + (size_t)m * a.out_cs : nullptr;
        if (full) {
            float4 lo = make_float4(acc[i][0] * sc + bb, acc[i][1] * sc + bb, acc[i][2] * sc + bb, acc[i][3] * sc + bb);
            float4 hi = make_float4(acc[i][4] * sc + bb, acc[i][5] * sc + bb, acc[i][6] * sc + bb, acc[i][7] * sc + bb);
            if (rrow) {
                float4 r0 = *(const float4*)(rrow + p0), r1 = *(const float4*)(rrow + p0 + 4);
                lo.x += r0.x; lo.y += r0.y; lo.z += r0.z; lo.w += r0.w;
                hi.x += r1.x; hi.y += r1.y; hi.z += r1.z; hi.w += r1.w;
            }
            *(float4*)(orow + p0) = lo;
            *(float4*)(orow + p0 + 4) = hi;
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int p = p0 + j;
                if (p >= a.P) break;
                size_t o = p;
                if (MODE == MUP) {
                    int yy = p / a.Wo, xx = p - yy * a.Wo;
                    o = (size_t)(2 * yy + (par >> 1)) * (2 * a.Wo) + 2 * xx + (par & 1);
                }
                float v = acc[i][j] * sc + bb;
                if (rrow) v += rrow[o];
                orow[o] = v;
            }
        }
    }
}

// bf16 [Cout][Cin][KK] -> fp16 [KK][Cin][Cout] (k = tap*Cin + c). UP: [4 parities][4 taps][Cin][Cout],
// each 2x2 tap the sum of the 3x3 taps that land on the same low-res input pixel:
//   output row parity 0: input row offset -1 <- ky 0; offset 0 <- ky 1, 2
//   output row parity 1: input row offset  0 <- ky 0, 1; offset +1 <- ky 2      (same for columns)
// One block = 32 output channels x 8 input channels, read coalesced, written coalesced.
template <bool UP, bool F16>
__global__ void k_prep(__half* __restrict__ wT, const uint16_t* __restrict__ w, int Cout, int Cin, int KK) {
    __shared__ float s[32][8 * 9 + 1];
    const int m0 = blockIdx.x * 32, c0 = blockIdx.y * 8, n = 8 * KK;
    for (int i = threadIdx.x; i < 32 * n; i += blockDim.x) {
        int mm = i / n, j = i - mm * n, m = m0 + mm, c = c0 + j / KK;
        const uint16_t u = (m < Cout && c < Cin) ? w[((size_t)m * Cin + c0) * KK + j] : 0;
        s[mm][j] = F16 ? __half2float(__ushort_as_half(u)) : bf(u);
    }
    __syncthreads();
    const int nt = UP ? 16 : KK;
    for (int i = threadIdx.x; i < 32 * 8 * nt; i += blockDim.x) {
        int mm = i % 32, r = i / 32, cc = r % 8, t = r / 8, m = m0 + mm, c = c0 + cc;
        if (m >= Cout || c >= Cin) continue;
        float v;
        if (!UP) {
            v = s[mm][cc * KK + t];
        } else {
            int par = t >> 2, tap = t & 3, py = par >> 1, px = par & 1, ry = tap >> 1, rx = tap & 1;
            int ky0 = (py == 0) ? (ry == 0 ? 0 : 1) : (ry == 0 ? 0 : 2), ky1 = (py == 0) ? (ry == 0 ? 0 : 2) : (ry == 0 ? 1 : 2);
            int kx0 = (px == 0) ? (rx == 0 ? 0 : 1) : (rx == 0 ? 0 : 2), kx1 = (px == 0) ? (rx == 0 ? 0 : 2) : (rx == 0 ? 1 : 2);
            v = 0.f;
            for (int ky = ky0; ky <= ky1; ky++)
                for (int kx = kx0; kx <= kx1; kx++) v += s[mm][cc * 9 + ky * 3 + kx];
        }
        wT[((size_t)t * Cin + c) * Cout + m] = __float2half_rn(v);
    }
}

// y = fp16(silu?(rms_norm_channels(x) * gamma)): x [C, P] fp32 -> y [C, P] fp16.
// Block = 32 pixels x 8 channel groups (reads stay coalesced along pixels). CPT > 0: C = 8*CPT and each
// thread keeps its CPT values in registers, so x is read once; CPT = 0: any C, x read twice.
template <int CPT>
__global__ void k_rms16(__half* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ g, int C, int P, int silu) {
    __shared__ float red[8][33];
    const int px = threadIdx.x, cy = threadIdx.y, p = blockIdx.x * 32 + px;
    float v[CPT > 0 ? CPT : 1];
    (void)v;
    float ss = 0.f;
    if (p < P) {
        if constexpr (CPT > 0) {
#pragma unroll
            for (int i = 0; i < CPT; i++) { v[i] = x[(size_t)(cy + 8 * i) * P + p]; ss += v[i] * v[i]; }
        } else {
            for (int c = cy; c < C; c += 8) { float t = x[(size_t)c * P + p]; ss += t * t; }
        }
    }
    red[cy][px] = ss;
    __syncthreads();
    if (p >= P) return;
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) s += red[i][px];
    const float r = sqrtf((float)C) / fmaxf(sqrtf(s), 1e-12f);
    auto out = [&](int c, float t) {
        t *= r * bf(g[c]);
        if (silu) t = t / (1.f + expf(-t));
        y[(size_t)c * P + p] = __float2half_rn(t);
    };
    if constexpr (CPT > 0) {
#pragma unroll
        for (int i = 0; i < CPT; i++) out(cy + 8 * i, v[i]);
    } else {
        for (int c = cy; c < C; c += 8) out(c, x[(size_t)c * P + p]);
    }
}

// y = fp16(x * s)
static __global__ void k_to16(__half* __restrict__ y, const float* __restrict__ x, size_t n, float s) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 4 <= n) {
        float4 v = *(const float4*)(x + i);
        *(__half2*)(y + i) = __floats2half2_rn(v.x * s, v.y * s);
        *(__half2*)(y + i + 2) = __floats2half2_rn(v.z * s, v.w * s);
    } else {
        for (; i < n; i++) y[i] = __float2half_rn(x[i] * s);
    }
}

// Decoder head: 3x3 conv Cin -> 3 (RGB), fp16 input, fp32 math. One thread per pixel
// (3 output channels would waste 29/32 of an implicit-GEMM tile).
static __global__ void k_head3(float* __restrict__ out, const __half* __restrict__ in, const uint16_t* __restrict__ w,
                        const uint16_t* __restrict__ b, int Cin, int H, int W) {
    extern __shared__ float sw[];  // [Cin*9][3]
    const int K = Cin * 9;
    for (int i = threadIdx.x; i < K * 3; i += blockDim.x) sw[(i % K) * 3 + i / K] = bf(w[i]);
    __syncthreads();
    const int P = H * W, p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int y = p / W, x = p - y * W;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    for (int c = 0; c < Cin; c++) {
        const __half* src = in + (size_t)c * P;
#pragma unroll
        for (int ky = 0; ky < 3; ky++) {
            int yy = y + ky - 1;
            if ((unsigned)yy >= (unsigned)H) continue;
#pragma unroll
            for (int kx = 0; kx < 3; kx++) {
                int xx = x + kx - 1;
                if ((unsigned)xx >= (unsigned)W) continue;
                float v = __half2float(src[(size_t)yy * W + xx]);
                const float* ww = sw + (c * 9 + ky * 3 + kx) * 3;
                a0 += ww[0] * v; a1 += ww[1] * v; a2 += ww[2] * v;
            }
        }
    }
    out[p] = a0 + bf(b[0]);
    out[(size_t)P + p] = a1 + bf(b[1]);
    out[2 * (size_t)P + p] = a2 + bf(b[2]);
}

// Encoder stem: 3x3 conv 3 -> Cout (pad 1), fp32. Block = 256 pixels x 32 output channels.
static __global__ void k_conv_in3(float* __restrict__ out, const float* __restrict__ in, const uint16_t* __restrict__ w,
                           const uint16_t* __restrict__ b, int Cout, int H, int W) {
    __shared__ float sw[32][28];
    __shared__ float sb[32];
    const int m0 = blockIdx.y * 32;
    for (int i = threadIdx.x; i < 32 * 27; i += blockDim.x) {
        int mm = i / 27, k = i - mm * 27;
        sw[mm][k] = m0 + mm < Cout ? bf(w[(size_t)(m0 + mm) * 27 + k]) : 0.f;
    }
    if (threadIdx.x < 32) sb[threadIdx.x] = m0 + threadIdx.x < Cout ? bf(b[m0 + threadIdx.x]) : 0.f;
    __syncthreads();
    const int P = H * W, p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int y = p / W, x = p - y * W;
    float v[27];
#pragma unroll
    for (int c = 0; c < 3; c++)
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
        for (int k = 0; k < 27; k++) acc += sw[mm][k] * v[k];
        out[(size_t)(m0 + mm) * P + p] = acc;
    }
}

// ---- attention helpers (one head, D = C channels, T tokens; K/V padded to Tp = roundup(T, 128)) ----
// y[c][j] = fp16(x[c][j]) for j < T, 0 for T <= j < Tp
static __global__ void k_pad16(__half* __restrict__ y, const float* __restrict__ x, int C, int T, int Tp) {
    size_t n = (size_t)C * Tp, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = (int)(i % Tp), c = (int)(i / Tp);
    y[i] = __float2half_rn(j < T ? x[(size_t)c * T + j] : 0.f);
}
// y[j][c] = fp16(x[c][j]) (transpose), rows T <= j < Tp zero. Block 32x8 over a 32x32 tile.
static __global__ void k_tpad16(__half* __restrict__ y, const float* __restrict__ x, int C, int T, int Tp) {
    __shared__ float tile[32][33];
    int j0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        int c = c0 + r, j = j0 + threadIdx.x;
        tile[r][threadIdx.x] = (c < C && j < T) ? x[(size_t)c * T + j] : 0.f;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        int j = j0 + r, c = c0 + threadIdx.x;
        if (j < Tp && c < C) y[(size_t)j * C + c] = __float2half_rn(tile[threadIdx.x][r]);
    }
}
// Column softmax: St [Tp][n] fp32 scores (key j, query i) -> Pt [Tp][n] fp16 probabilities over the
// keys j < T; rows T <= j < Tp are written as 0. Block = 32 queries x 16 key groups.
static __global__ void k_colsoftmax16(__half* __restrict__ Pt, const float* __restrict__ St, int T, int Tp, int n) {
    __shared__ float sm[16][33], ss[16][33];
    const int tx = threadIdx.x, jy = threadIdx.y, i = blockIdx.x * 32 + tx;
    float m = -INFINITY, s = 0.f;
    if (i < n)
        for (int j = jy; j < T; j += 16) {
            float v = St[(size_t)j * n + i];
            if (v > m) { s *= __expf(m - v); m = v; }
            s += __expf(v - m);
        }
    sm[jy][tx] = m;
    ss[jy][tx] = s;
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int r = 0; r < 16; r++) M = fmaxf(M, sm[r][tx]);
    float S = 0.f;
#pragma unroll
    for (int r = 0; r < 16; r++) S += sm[r][tx] == -INFINITY ? 0.f : ss[r][tx] * __expf(sm[r][tx] - M);
    const float inv = 1.f / S;
    if (i >= n) return;
    for (int j = jy; j < Tp; j += 16)
        Pt[(size_t)j * n + i] = __float2half_rn(j < T ? __expf(St[(size_t)j * n + i] - M) * inv : 0.f);
}

// ---- host side ----
inline unsigned nb(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// output-channel tile: 16*TM rows that divide Cout
inline int pick_tm(int cout) {
    if (cout % 128 == 0) return 8;
    if (cout % 96 == 0) return 6;
    if (cout % 64 == 0) return 4;
    if (cout % 32 == 0) return 2;
    return 0;
}
inline bool eligible(int cin, int cout) { return cin % BK == 0 && pick_tm(cout) != 0; }

template <int TM, int MODE, int NTX = 16, int MINB = (TM <= 6 ? 2 : 1)>
inline void launch_tm(const Args& a) {
    const int ntiles = (a.P + 8 * NTX - 1) / (8 * NTX);
    dim3 grid(MODE == MUP ? 4 * ntiles : ntiles, a.Cout / (16 * TM));
    k_conv<TM, MODE, NTX, MINB><<<grid, 16 * NTX, 0, G.stream>>>(a);
}
// Measured on the GTX 1660 Ti: Cout % 128 == 0 -> 128 couts x 64 pixels, 128 threads, 2 blocks/SM
// (~7% over 256 threads x 128 pixels); 96/192 couts -> 96 x 128, 256 threads, 2 blocks/SM.
template <int MODE>
inline void launch_mode(int tm, const Args& a) {
    switch (tm) {
        case 8: launch_tm<8, MODE, 8, 2>(a); break;
        case 6: launch_tm<6, MODE>(a); break;
        case 4: launch_tm<4, MODE>(a); break;
        default: launch_tm<2, MODE>(a); break;
    }
}

// w (bf16 or fp16) [Cout, Cin*KK] -> the kernel's fp16 layout [taps][Cin][Cout] (MUP: [4][4][Cin][Cout])
inline void prep_into(__half* wt, const Weight& w, int Cin, int mode) {
    const int Cout = (int)w.rows, KK = (int)(w.cols / Cin);
    dim3 pg((Cout + 31) / 32, (Cin + 7) / 8);
    if (mode == MUP) {
        if (w.f16) k_prep<true, true><<<pg, 256, 0, G.stream>>>(wt, w.p, Cout, Cin, KK);
        else k_prep<true, false><<<pg, 256, 0, G.stream>>>(wt, w.p, Cout, Cin, KK);
    } else {
        if (w.f16) k_prep<false, true><<<pg, 256, 0, G.stream>>>(wt, w.p, Cout, Cin, KK);
        else k_prep<false, false><<<pg, 256, 0, G.stream>>>(wt, w.p, Cout, Cin, KK);
    }
}
inline size_t prep_elems(const Weight& w, int Cin, int mode) { return (size_t)(mode == MUP ? 16 : w.cols / Cin) * Cin * w.rows; }

// Output grid: M3/M1 H x W, MS2 H/2 x W/2, MS2P ceil(H/2) x ceil(W/2), MUP 2H x 2W.
inline void run(float* out, const __half* in, int Cin, int H, int W, const __half* wt, int Cout, const uint16_t* bias,
                const float* bias32, const float* res, int mode, float scale, int tm) {
    Args a;
    a.out = out; a.res = res; a.in = in; a.w = wt; a.bias = bias; a.bias32 = bias32; a.row_off = 0;
    a.Cin = Cin; a.Cout = Cout; a.H = H; a.W = W;
    int Ho = H, Wo = W;
    if (mode == MS2) { Ho = H / 2; Wo = W / 2; }
    if (mode == MS2P) { Ho = (H + 1) / 2; Wo = (W + 1) / 2; }
    a.Wo = Wo; a.P = Ho * Wo;
    a.in_cs = H * W;
    a.out_cs = mode == MUP ? 4 * H * W : Ho * Wo;
    a.scale = scale;
    a.vec = (a.out_cs % 4 == 0) && ((uintptr_t)out % 16 == 0) && (!res || (uintptr_t)res % 16 == 0);
    switch (mode) {
        case M3: launch_mode<M3>(tm, a); break;
        case M1: launch_mode<M1>(tm, a); break;
        case MS2: launch_mode<MS2>(tm, a); break;
        case MS2P: launch_mode<MS2P>(tm, a); break;
        default: launch_mode<MUP>(tm, a); break;
    }
}

// One row band of a conv. `in` holds input rows [r_in, r_in + H) (the band plus its halo; the conv
// zero-pads outside them, which is exact at the image edges), output rows [r_out, r_out + rows) are
// written into a full [Cout, Ho_full, Wo] tensor with channel stride out_cs (res, if any, shares it).
// MUP: rows are low-resolution rows; the output tensor's row stride is 2*W.
inline void run_band(float* out, size_t out_cs, const float* res, const __half* in, int Cin, int H, int W, int row_off, int rows,
                     const __half* wt, int Cout, const float* bias32, int mode, float scale) {
    if (!eligible(Cin, Cout)) throw std::runtime_error("vc::run_band: unsupported shape");
    ProfScope ps("conv16");
    Args a;
    a.out = out; a.res = res; a.in = in; a.w = wt; a.bias = nullptr; a.bias32 = bias32; a.row_off = row_off;
    a.Cin = Cin; a.Cout = Cout; a.H = H; a.W = W;
    const int Wo = (mode == MS2 || mode == MS2P) ? (W + (mode == MS2P)) / 2 : W;
    a.Wo = Wo; a.P = rows * Wo;
    a.in_cs = H * W;
    a.out_cs = (int)out_cs;
    a.scale = scale;
    a.vec = mode != MUP && (out_cs % 4 == 0) && ((uintptr_t)out % 16 == 0) && (!res || (uintptr_t)res % 16 == 0);
    const int tm = pick_tm(Cout);
    switch (mode) {
        case M3: launch_mode<M3>(tm, a); break;
        case M1: launch_mode<M1>(tm, a); break;
        case MS2: launch_mode<MS2>(tm, a); break;
        case MS2P: launch_mode<MS2P>(tm, a); break;
        default: launch_mode<MUP>(tm, a); break;
    }
}

// A conv whose weights were arranged once at load (prep_into): out = scale * conv(in) + bias32 (+ res).
inline void conv_pre(float* out, const __half* in, int Cin, int H, int W, const __half* wt, int Cout, const float* bias32,
                     const float* res, int mode, float scale = 1.f) {
    if (!eligible(Cin, Cout)) throw std::runtime_error("vc::conv_pre: unsupported shape");
    ProfScope ps("conv16");
    run(out, in, Cin, H, W, wt, Cout, nullptr, bias32, res, mode, scale, pick_tm(Cout));
}

// out[Cout, ...] = scale * conv(in) + bias (+ res). in: fp16 [Cin, H, W]; w: bf16/fp16 [Cout, Cin*KK].
inline void conv(float* out, const __half* in, int Cin, int H, int W, const Weight& w, const Weight* b,
                 const float* res, int mode, float scale) {
    const int Cout = (int)w.rows, KK = (int)(w.cols / Cin), tm = pick_tm(Cout);
    if (!eligible(Cin, Cout)) throw std::runtime_error("vc::conv: unsupported shape");
    ProfScope ps("vae_conv16");
    const int nt = mode == MUP ? 16 : KK;
    size_t m = G.arena.mark();
    __half* wt = (__half*)G.arena.f(((size_t)nt * Cin * Cout + 1) / 2);
    prep_into(wt, w, Cin, mode);
    run(out, in, Cin, H, W, wt, Cout, b ? b->p : nullptr, nullptr, res, mode, scale, tm);
    G.arena.release(m);  // stream-ordered: later kernels reuse the slab only after this one ran
}

// Plain GEMM through the 1x1 path: out[m, p] (row stride out_cs) = scale * sum_k A[k, m] B[k, p] (+ res)
// A fp16 [K][M] (M contiguous), B fp16 [K][n] with row stride b_cs. K % 16 == 0, M a multiple of 32.
inline void gemm16(float* out, int out_cs, const float* res, const __half* A, int K, int M, const __half* B, int b_cs, int n, float scale) {
    ProfScope ps("vae_gemm16");
    const int tm = pick_tm(M);
    if (K % BK || !tm) throw std::runtime_error("vc::gemm16: unsupported shape");
    Args a;
    a.out = out; a.res = res; a.in = B; a.w = A; a.bias = nullptr; a.bias32 = nullptr; a.row_off = 0;
    a.Cin = K; a.Cout = M; a.H = 1; a.W = n;
    a.Wo = n; a.P = n; a.in_cs = b_cs; a.out_cs = out_cs;
    a.scale = scale;
    a.vec = (out_cs % 4 == 0) && ((uintptr_t)out % 16 == 0) && (!res || (uintptr_t)res % 16 == 0);
    launch_mode<M1>(tm, a);
}

inline void rms16(__half* y, const float* x, const Weight& gamma, int C, int P, bool silu) {
    ProfScope ps("vae_norm");
    dim3 grid((P + 31) / 32), blk(32, 8);
    const int sl = silu ? 1 : 0;
    switch (C) {
        case 96: k_rms16<12><<<grid, blk, 0, G.stream>>>(y, x, gamma.p, C, P, sl); break;
        case 192: k_rms16<24><<<grid, blk, 0, G.stream>>>(y, x, gamma.p, C, P, sl); break;
        case 384: k_rms16<48><<<grid, blk, 0, G.stream>>>(y, x, gamma.p, C, P, sl); break;
        default: k_rms16<0><<<grid, blk, 0, G.stream>>>(y, x, gamma.p, C, P, sl); break;
    }
}
inline void to16(__half* y, const float* x, size_t n, float s) {
    ProfScope ps("vae_misc");
    k_to16<<<nb((n + 3) / 4), 256, 0, G.stream>>>(y, x, n, s);
}
inline void head3(float* out, const __half* in, int Cin, int H, int W, const Weight& w, const Weight& b) {
    ProfScope ps("vae_conv_io");
    size_t P = (size_t)H * W;
    k_head3<<<nb(P), 256, (size_t)Cin * 27 * 4, G.stream>>>(out, in, w.p, b.p, Cin, H, W);
}
inline void conv_in3(float* out, const float* in, int H, int W, const Weight& w, const Weight& b) {
    ProfScope ps("vae_conv_io");
    int Cout = (int)w.rows;
    dim3 grid(nb((size_t)H * W), (Cout + 31) / 32);
    k_conv_in3<<<grid, 256, 0, G.stream>>>(out, in, w.p, b.p, Cout, H, W);
}

}  // namespace vc
