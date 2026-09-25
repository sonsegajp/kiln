// W8A8 GEMM for Turing GPUs without tensor cores (TU11x), built on __dp4a (4 int8 MACs per INT32-pipe
// instruction, issued alongside the FP pipe).
//
//   C[M,N] = sa[m] * sw[n] * sum_k A[m,k] * W[n,k]   (+ beta * C)
//
//   A : int8 [M,K] row-major activations, sa[M] per-row (per-token) scales   -> quantize_rows()
//   W : int8 [N,K] row-major weights (PyTorch [out,in]), sw[N] per-output-channel scales -> quantize_weight()
//   C : fp32 [M,N]
// int32 accumulation is exact (|sum| <= K*127*127 < 2^31 for K <= 131k); the only error is the
// int8 rounding of A and W. Optional SmoothQuant: pass per-input-channel factors s[k] as
// col_inv = 1/s[k] to quantize_rows() and col_mul = s[k] to quantize_weight().
//
// Requirements (checked by eligible()): N % BN == 0, K % BK == 0 (so K % 16 == 0 and every row is 16-byte
// aligned), any M >= 1.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace i8 {

// ---------------------------------------------------------------------------------------------
// GEMM
// Block tile BM x BN, k-tile BK bytes. 256 threads, each owning a TM x TN register tile laid out as
// (TM/4) x (TN/4) groups of 4x4 spread 64 rows/cols apart, so that the per-k shared-memory reads are
// contiguous 16-byte chunks per thread (conflict-free LDS.128) and a warp's A reads are broadcasts.
// Shared tiles hold k-quads (one int32 = 4 consecutive k of one row) in [k/4][row] order, padded by 4
// words so the transposing STS.32 stores are conflict-free. Global->register prefetch of tile t+1 overlaps
// the dp4a work on tile t; one __syncthreads per tile (double-buffered shared memory).
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN, int MINB>
struct Tile {
    static constexpr int TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    static constexpr int KW = BK / 4;                   // k-quads per tile
    static constexpr int PAD = 4;
    static constexpr int SA = BM + PAD, SB = BN + PAD;  // shared row strides (words)
    static constexpr int CPR = BK / 16;                 // 16-byte chunks per tile row
    static constexpr int A_LD = BM * CPR / THREADS, B_LD = BN * CPR / THREADS;
    static constexpr int GA = TM / 4, GB = TN / 4;       // 4-row / 4-col groups per thread
    static constexpr int SPAN_A = BM / GA, SPAN_B = BN / GB;
    static_assert(BM * CPR % THREADS == 0 && BN * CPR % THREADS == 0, "staging must divide evenly");
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 16 == 0, "tile shape");
    static_assert(SPAN_A == TY * 4 && SPAN_B == TX * 4, "group layout");
};

template <int BM, int BN, int BK, int TM, int TN, int MINB>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), MINB)
gemm_nt(float* __restrict__ C, const int8_t* __restrict__ A, const float* __restrict__ sa, const int8_t* __restrict__ W,
        const float* __restrict__ sw, int M, int N, int K, float beta) {
    using T = Tile<BM, BN, BK, TM, TN, MINB>;
    __shared__ __align__(16) int As[2][T::KW][T::SA];
    __shared__ __align__(16) int Bs[2][T::KW][T::SB];

    const int tid = threadIdx.x, tx = tid % T::TX, ty = tid / T::TX;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    uint4 ra[T::A_LD], rb[T::B_LD];
    auto load = [&](int k0) {
#pragma unroll
        for (int r = 0; r < T::A_LD; r++) {
            int c = tid + r * T::THREADS, row = c / T::CPR, kq = c % T::CPR, m = m0 + row;
            ra[r] = m < M ? __ldg((const uint4*)(A + (size_t)m * K + k0 + kq * 16)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int r = 0; r < T::B_LD; r++) {
            int c = tid + r * T::THREADS, row = c / T::CPR, kq = c % T::CPR;
            rb[r] = __ldg((const uint4*)(W + (size_t)(n0 + row) * K + k0 + kq * 16));
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int r = 0; r < T::A_LD; r++) {
            int c = tid + r * T::THREADS, row = c / T::CPR, kq = c % T::CPR;
            As[buf][kq * 4 + 0][row] = (int)ra[r].x;
            As[buf][kq * 4 + 1][row] = (int)ra[r].y;
            As[buf][kq * 4 + 2][row] = (int)ra[r].z;
            As[buf][kq * 4 + 3][row] = (int)ra[r].w;
        }
#pragma unroll
        for (int r = 0; r < T::B_LD; r++) {
            int c = tid + r * T::THREADS, row = c / T::CPR, kq = c % T::CPR;
            Bs[buf][kq * 4 + 0][row] = (int)rb[r].x;
            Bs[buf][kq * 4 + 1][row] = (int)rb[r].y;
            Bs[buf][kq * 4 + 2][row] = (int)rb[r].z;
            Bs[buf][kq * 4 + 3][row] = (int)rb[r].w;
        }
    };

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0;

    const int tiles = K / BK;
    load(0);
    store(0);
    __syncthreads();

    for (int t = 0; t < tiles; t++) {
        const int buf = t & 1;
        if (t + 1 < tiles) load((t + 1) * BK);
#pragma unroll
        for (int kw = 0; kw < T::KW; kw++) {
            int a[TM], b[TN];
#pragma unroll
            for (int g = 0; g < T::GA; g++) *(int4*)&a[4 * g] = *(const int4*)&As[buf][kw][g * T::SPAN_A + ty * 4];
#pragma unroll
            for (int g = 0; g < T::GB; g++) *(int4*)&b[4 * g] = *(const int4*)&Bs[buf][kw][g * T::SPAN_B + tx * 4];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        if (t + 1 < tiles) {
            store(buf ^ 1);
            __syncthreads();
        }
    }

    // epilogue: dequantize, optional beta*C, float4 stores (16 threads cover 256 contiguous bytes of a row)
    float4 swv[T::GB];
#pragma unroll
    for (int g = 0; g < T::GB; g++) swv[g] = *(const float4*)(sw + n0 + g * T::SPAN_B + tx * 4);
#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int m = m0 + (i / 4) * T::SPAN_A + ty * 4 + (i % 4);
        if (m >= M) continue;
        const float s = sa[m];
#pragma unroll
        for (int g = 0; g < T::GB; g++) {
            float* c = C + (size_t)m * N + n0 + g * T::SPAN_B + tx * 4;
            float4 v = make_float4((float)acc[i][4 * g + 0] * s * swv[g].x, (float)acc[i][4 * g + 1] * s * swv[g].y,
                                   (float)acc[i][4 * g + 2] * s * swv[g].z, (float)acc[i][4 * g + 3] * s * swv[g].w);
            if (beta != 0.f) {
                float4 o = *(const float4*)c;
                v.x += beta * o.x; v.y += beta * o.y; v.z += beta * o.z; v.w += beta * o.w;
            }
            *(float4*)c = v;
        }
    }
}

// default configuration (tuned on GTX 1660 Ti Max-Q, see bench/i8_test.cu)
constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8, MINB = 2;

inline bool eligible(int M, int N, int K) { return M >= 1 && N % BN == 0 && K % BK == 0; }

inline void gemm(float* C, const int8_t* A, const float* sa, const int8_t* W, const float* sw, int M, int N, int K, float beta,
                 cudaStream_t s) {
    dim3 grid(N / BN, (M + BM - 1) / BM);
    gemm_nt<BM, BN, BK, TM, TN, MINB><<<grid, (BM / TM) * (BN / TN), 0, s>>>(C, A, sa, W, sw, M, N, K, beta);
}

// ---------------------------------------------------------------------------------------------
// quantizers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float blk_max256(float v) {
    __shared__ float sh[8];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    v = sh[0];
#pragma unroll
    for (int w = 1; w < 8; w++) v = fmaxf(v, sh[w]);
    return v;
}

__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
    int qa = __float2int_rn(a), qb = __float2int_rn(b), qc = __float2int_rn(c), qd = __float2int_rn(d);
    return (uint32_t)(qa & 0xff) | ((uint32_t)(qb & 0xff) << 8) | ((uint32_t)(qc & 0xff) << 16) | ((uint32_t)(qd & 0xff) << 24);
}

// Per-row symmetric int8: scale[m] = max_k |x[m,k]*col_inv[k]| / 127. One 256-thread block per row;
// the row is held in registers (K <= 256*4*KMAX4), so x is read once. col_inv may be null.
template <int KMAX4>
__global__ void __launch_bounds__(256) k_quant_rows(int8_t* __restrict__ q, float* __restrict__ scale, const float* __restrict__ x,
                                                    const float* __restrict__ col_inv, int K) {
    const float* xr = x + (size_t)blockIdx.x * K;
    float4 v[KMAX4];
    float mx = 0.f;
#pragma unroll
    for (int r = 0; r < KMAX4; r++) {
        int i = (threadIdx.x + r * 256) * 4;
        if (i < K) {
            v[r] = *(const float4*)(xr + i);
            if (col_inv) {
                float4 s = *(const float4*)(col_inv + i);
                v[r].x *= s.x; v[r].y *= s.y; v[r].z *= s.z; v[r].w *= s.w;
            }
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[r].x), fabsf(v[r].y)), fmaxf(fabsf(v[r].z), fabsf(v[r].w))));
        }
    }
    mx = blk_max256(mx);
    const float inv = mx > 0.f ? 127.f / mx : 0.f;
    if (threadIdx.x == 0) scale[blockIdx.x] = mx / 127.f;
    uint32_t* qr = (uint32_t*)(q + (size_t)blockIdx.x * K);
#pragma unroll
    for (int r = 0; r < KMAX4; r++) {
        int i = (threadIdx.x + r * 256) * 4;
        if (i < K) qr[i / 4] = pack4(v[r].x * inv, v[r].y * inv, v[r].z * inv, v[r].w * inv);
    }
}

// x fp32 [M,K] -> q int8 [M,K] + scale[M]. K % 4 == 0, K <= 8192.
inline void quantize_rows(int8_t* q, float* scale, const float* x, const float* col_inv, int M, int K, cudaStream_t s) {
    if (K <= 1024) k_quant_rows<1><<<M, 256, 0, s>>>(q, scale, x, col_inv, K);
    else if (K <= 2048) k_quant_rows<2><<<M, 256, 0, s>>>(q, scale, x, col_inv, K);
    else if (K <= 4096) k_quant_rows<4><<<M, 256, 0, s>>>(q, scale, x, col_inv, K);
    else k_quant_rows<8><<<M, 256, 0, s>>>(q, scale, x, col_inv, K);
}

// bf16 weight [N,K] (PyTorch [out,in]) -> int8 [N,K] + per-output-channel scale[N]; col_mul (SmoothQuant
// factors s[k]) may be null. One block per output row, two passes over the row (it stays in L1/L2).
__global__ void __launch_bounds__(256) k_quant_weight(int8_t* __restrict__ q, float* __restrict__ scale, const uint16_t* __restrict__ w,
                                                      const float* __restrict__ col_mul, int K) {
    const uint16_t* wr = w + (size_t)blockIdx.x * K;
    auto val = [&](int i) {
        float f = __uint_as_float((uint32_t)wr[i] << 16);
        return col_mul ? f * col_mul[i] : f;
    };
    float mx = 0.f;
    for (int i = threadIdx.x; i < K; i += 256) mx = fmaxf(mx, fabsf(val(i)));
    mx = blk_max256(mx);
    const float inv = mx > 0.f ? 127.f / mx : 0.f;
    if (threadIdx.x == 0) scale[blockIdx.x] = mx / 127.f;
    for (int i = threadIdx.x; i < K; i += 256) q[(size_t)blockIdx.x * K + i] = (int8_t)__float2int_rn(val(i) * inv);
}

inline void quantize_weight(int8_t* q, float* scale, const uint16_t* w_bf16, const float* col_mul, int N, int K, cudaStream_t s) {
    k_quant_weight<<<N, 256, 0, s>>>(q, scale, w_bf16, col_mul, K);
}

}  // namespace i8
