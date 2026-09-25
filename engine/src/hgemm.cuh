// fp16x2 GEMM for GPUs without tensor cores (Turing TU11x): C[M,N] = A[M,K] * W[N,K]^T (+ beta*C)
//   A: fp32 activations, W: bf16 (or fp16) weights. Both are rounded to fp16 while being staged in shared
//   memory, so no separate weight-conversion pass is needed. Products accumulate in fp16 pairs
//   (HFMA2, twice the fp32 rate on these chips) and are flushed into fp32 every 32 k, which keeps
//   the long reductions (K = 2048..8192) accurate.
// Requirements: N % 128 == 0, K % 16 == 0, K % 4 == 0 (float4 loads), 16-byte aligned rows.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace hg {

constexpr int BM = 128, BN = 128, BK = 16, FLUSH = 2;  // flush fp16 partials every FLUSH*BK k-steps

__device__ __forceinline__ __half bf2h(uint16_t b) { return __float2half_rn(__uint_as_float((uint32_t)b << 16)); }
// W element -> fp16: bf16 weights are rounded, fp16 weights (F16W) pass through
template <bool F16W> __device__ __forceinline__ __half w2h(uint16_t b) { return F16W ? __ushort_as_half(b) : bf2h(b); }

template <bool F16W>
__global__ void __launch_bounds__(256, 1)
hgemm_nt(float* __restrict__ C, const float* __restrict__ A, const uint16_t* __restrict__ W, int M, int N, int K, float beta) {
    __shared__ __align__(16) __half As[2][BK][BM];
    __shared__ __align__(16) __half Bs[2][BK][BN];

    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // global -> register staging: A 128x16 fp32 (2 float4 per thread), W 128x16 bf16 (1 uint4 per thread)
    const int a_row = tid / 4, a_col = (tid % 4) * 4;       // rows a_row and a_row+64
    const int b_row = tid / 2, b_col = (tid % 2) * 8;
    float4 ra[2];
    uint4 rb;

    auto load = [&](int k0) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int m = m0 + a_row + r * 64;
            ra[r] = m < M ? *(const float4*)(A + (size_t)m * K + k0 + a_col) : make_float4(0, 0, 0, 0);
        }
        rb = *(const uint4*)(W + (size_t)(n0 + b_row) * K + k0 + b_col);
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int m = a_row + r * 64;
            As[buf][a_col + 0][m] = __float2half_rn(ra[r].x);
            As[buf][a_col + 1][m] = __float2half_rn(ra[r].y);
            As[buf][a_col + 2][m] = __float2half_rn(ra[r].z);
            As[buf][a_col + 3][m] = __float2half_rn(ra[r].w);
        }
        const uint32_t w4[4] = {rb.x, rb.y, rb.z, rb.w};
#pragma unroll
        for (int q = 0; q < 4; q++) {
            Bs[buf][b_col + 2 * q][b_row] = w2h<F16W>((uint16_t)(w4[q] & 0xffff));
            Bs[buf][b_col + 2 * q + 1][b_row] = w2h<F16W>((uint16_t)(w4[q] >> 16));
        }
    };

    float acc[8][8];
    __half2 part[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) part[i][j] = __float2half2_rn(0.f);
    }

    const int tiles = K / BK;
    load(0);
    store(0);
    __syncthreads();

    for (int t = 0; t < tiles; t++) {
        const int buf = t & 1;
        if (t + 1 < tiles) load((t + 1) * BK);
#pragma unroll
        for (int k = 0; k < BK; k++) {
            uint4 av = *(const uint4*)&As[buf][k][ty * 8];
            uint4 bv = *(const uint4*)&Bs[buf][k][tx * 8];
            __half2 a2[4] = {*(__half2*)&av.x, *(__half2*)&av.y, *(__half2*)&av.z, *(__half2*)&av.w};
            __half2 b2[4] = {*(__half2*)&bv.x, *(__half2*)&bv.y, *(__half2*)&bv.z, *(__half2*)&bv.w};
#pragma unroll
            for (int i2 = 0; i2 < 4; i2++) {
                __half2 lo = __low2half2(a2[i2]), hi = __high2half2(a2[i2]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    part[2 * i2][j] = __hfma2(lo, b2[j], part[2 * i2][j]);
                    part[2 * i2 + 1][j] = __hfma2(hi, b2[j], part[2 * i2 + 1][j]);
                }
            }
        }
        if ((t + 1) % FLUSH == 0 || t + 1 == tiles) {
#pragma unroll
            for (int i = 0; i < 8; i++)
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

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int m = m0 + ty * 8 + i;
        if (m >= M) break;
        float* c = C + (size_t)m * N + n0 + tx * 8;
        float4 lo = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        float4 hi = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        if (beta != 0.f) {
            float4 o0 = *(float4*)c, o1 = *(float4*)(c + 4);
            lo.x += beta * o0.x; lo.y += beta * o0.y; lo.z += beta * o0.z; lo.w += beta * o0.w;
            hi.x += beta * o1.x; hi.y += beta * o1.y; hi.z += beta * o1.z; hi.w += beta * o1.w;
        }
        *(float4*)c = lo;
        *(float4*)(c + 4) = hi;
    }
}

// v2: A is staged in shared memory as duplicated half2 pairs (a, a), so the inner loop is pure
// LDS + HFMA2 with no register shuffles; thread->element mappings keep every shared-memory store
// conflict-free.
__global__ void __launch_bounds__(256, 1)
hgemm_nt2(float* __restrict__ C, const float* __restrict__ A, const uint16_t* __restrict__ W, int M, int N, int K, float beta) {
    __shared__ __align__(16) __half2 As[2][BK][BM];   // (a, a)
    __shared__ __align__(16) __half Bs[2][BK][BN];

    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // A: thread -> row tid%64 (+64), k-quad tid/64. W: thread -> row tid%128, k-octet tid/128.
    const int a_row = tid % 64, a_col = (tid / 64) * 4;
    const int b_row = tid % 128, b_col = (tid / 128) * 8;
    float4 ra[2];
    uint4 rb;

    auto load = [&](int k0) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int m = m0 + a_row + r * 64;
            ra[r] = m < M ? *(const float4*)(A + (size_t)m * K + k0 + a_col) : make_float4(0, 0, 0, 0);
        }
        rb = *(const uint4*)(W + (size_t)(n0 + b_row) * K + k0 + b_col);
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int m = a_row + r * 64;
            As[buf][a_col + 0][m] = __float2half2_rn(ra[r].x);
            As[buf][a_col + 1][m] = __float2half2_rn(ra[r].y);
            As[buf][a_col + 2][m] = __float2half2_rn(ra[r].z);
            As[buf][a_col + 3][m] = __float2half2_rn(ra[r].w);
        }
        const uint32_t w4[4] = {rb.x, rb.y, rb.z, rb.w};
#pragma unroll
        for (int q = 0; q < 4; q++) {
            Bs[buf][b_col + 2 * q][b_row] = bf2h((uint16_t)(w4[q] & 0xffff));
            Bs[buf][b_col + 2 * q + 1][b_row] = bf2h((uint16_t)(w4[q] >> 16));
        }
    };

    float acc[8][8];
    __half2 part[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) part[i][j] = __float2half2_rn(0.f);
    }

    const int tiles = K / BK;
    load(0);
    store(0);
    __syncthreads();

    for (int t = 0; t < tiles; t++) {
        const int buf = t & 1;
        if (t + 1 < tiles) load((t + 1) * BK);
#pragma unroll
        for (int k = 0; k < BK; k++) {
            uint4 a0 = *(const uint4*)&As[buf][k][ty * 8];
            uint4 a1 = *(const uint4*)&As[buf][k][ty * 8 + 4];
            uint4 bv = *(const uint4*)&Bs[buf][k][tx * 8];
            __half2 a2[8] = {*(__half2*)&a0.x, *(__half2*)&a0.y, *(__half2*)&a0.z, *(__half2*)&a0.w,
                             *(__half2*)&a1.x, *(__half2*)&a1.y, *(__half2*)&a1.z, *(__half2*)&a1.w};
            __half2 b2[4] = {*(__half2*)&bv.x, *(__half2*)&bv.y, *(__half2*)&bv.z, *(__half2*)&bv.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) part[i][j] = __hfma2(a2[i], b2[j], part[i][j]);
        }
        if ((t + 1) % FLUSH == 0 || t + 1 == tiles) {
#pragma unroll
            for (int i = 0; i < 8; i++)
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

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int m = m0 + ty * 8 + i;
        if (m >= M) break;
        float* c = C + (size_t)m * N + n0 + tx * 8;
        float4 lo = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        float4 hi = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        if (beta != 0.f) {
            float4 o0 = *(float4*)c, o1 = *(float4*)(c + 4);
            lo.x += beta * o0.x; lo.y += beta * o0.y; lo.z += beta * o0.z; lo.w += beta * o0.w;
            hi.x += beta * o1.x; hi.y += beta * o1.y; hi.z += beta * o1.z; hi.w += beta * o1.w;
        }
        *(float4*)c = lo;
        *(float4*)(c + 4) = hi;
    }
}

inline bool eligible(int M, int N, int K) { return N % BN == 0 && K % BK == 0 && M >= 16; }

inline void launch(float* C, const float* A, const uint16_t* W, int M, int N, int K, float beta, cudaStream_t s, int ver = 1, bool f16 = false) {
    dim3 grid(N / BN, (M + BM - 1) / BM);
    if (f16) hgemm_nt<true><<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
    else if (ver == 1) hgemm_nt<false><<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
    else hgemm_nt2<<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
}

}  // namespace hg
