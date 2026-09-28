// Tensor-core GEMM for RTX GPUs: C[M,N] = A[M,K] * W[N,K]^T (+ beta*C), the same contract as hgemm.cuh.
//   A: fp32 activations, W: bf16 (or fp16 with F16W) weights in PyTorch layout [out, in].
//   Both are rounded to fp16 while being staged into shared memory (no separate conversion pass).
//
// Accumulation: on GeForce cards, mma with an fp16 accumulator runs at twice the rate of an fp32 one
// (e.g. RTX 4090: 165 vs 83 TFLOPS dense). PyTorch always accumulates in fp32, so it gets the slower rate.
// F16ACC keeps the fast rate and the accuracy: every 32-deep k tile accumulates in fp16 (two mma per
// output fragment) and is then flushed into fp32 totals, the same scheme hgemm.cuh uses on the CUDA cores.
// !F16ACC accumulates in fp32 throughout (for cards where both rates are equal).
//
// Tiling: 128x128 block tile, k tile 32, 256 threads = 2 x 4 warps, 64x32 warp tile = 4 x 4 m16n8 fragments.
// Shared rows are padded to 40 halves (80 bytes): the 8 row addresses of every ldmatrix phase land in
// distinct 16-byte bank groups. Global -> register prefetch of k tile t+1 overlaps the mma work on tile t;
// double-buffered shared memory, one __syncthreads per k tile.
// sm_80+: mma.m16n8k16. sm_75 (Turing RTX): two mma.m16n8k8 per k16 step.
// Requirements: N % 128 == 0, K % 32 == 0, 16-byte aligned rows.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tc {

constexpr int BM = 128, BN = 128, BK = 32, LDS = BK + 8;

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// d (fp16 pair accumulators, rows g and g+8) += A[16x16] * B[16x8]
__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#endif
}
// d (fp32: row g cols 2t,2t+1, row g+8 cols 2t,2t+1) += A[16x16] * B[16x8]
__device__ __forceinline__ void mma_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#endif
}

__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
// two packed bf16 -> two packed fp16
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
    return h2_bits(__floats2half2_rn(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)));
}
// flush an fp16 pair accumulator into fp32 totals and clear it
__device__ __forceinline__ void flush(float* acc, uint32_t& h) {
    float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
    acc[0] += f.x;
    acc[1] += f.y;
    h = 0;
}

template <bool F16W, bool F16ACC>
__global__ void __launch_bounds__(256, 1)
k_gemm(float* __restrict__ C, const float* __restrict__ A, const uint16_t* __restrict__ W, int M, int N, int K, float beta) {
    __shared__ __align__(16) __half As[2][BM][LDS];
    __shared__ __align__(16) __half Ws[2][BN][LDS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // global staging: A 128x32 fp32 = 4 float4 per thread, W 128x32 16-bit = 2 uint4 per thread
    const int a_row = tid >> 3, a_col = (tid & 7) * 4;
    const int w_row = tid >> 2, w_col = (tid & 3) * 8;
    float4 ra[4];
    uint4 rw[2];
    auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = m0 + a_row + 32 * i;
            ra[i] = r < M ? *reinterpret_cast<const float4*>(A + (size_t)r * K + k0 + a_col) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) rw[i] = *reinterpret_cast<const uint4*>(W + (size_t)(n0 + w_row + 64 * i) * K + k0 + w_col);
    };
    auto store = [&](int s) {
#pragma unroll
        for (int i = 0; i < 4; i++)
            *reinterpret_cast<uint2*>(&As[s][a_row + 32 * i][a_col]) =
                make_uint2(h2_bits(__floats2half2_rn(ra[i].x, ra[i].y)), h2_bits(__floats2half2_rn(ra[i].z, ra[i].w)));
#pragma unroll
        for (int i = 0; i < 2; i++) {
            uint4 v = rw[i];
            if (!F16W) { v.x = bf2_to_h2(v.x); v.y = bf2_to_h2(v.y); v.z = bf2_to_h2(v.z); v.w = bf2_to_h2(v.w); }
            *reinterpret_cast<uint4*>(&Ws[s][w_row + 64 * i][w_col]) = v;
        }
    };

    float acc[4][4][4];
    uint32_t hacc[4][4][2];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
            hacc[i][j][0] = hacc[i][j][1] = 0;
        }

    // ldmatrix lane addresses (see the fragment layouts of mma.m16n8k16)
    const int a_lrow = wm * 64 + (lane & 15), a_lcol = (lane >> 4) * 8;
    const int b_lrow = wn * 32 + (lane & 7) + ((lane >> 4) << 3), b_lcol = ((lane >> 3) & 1) * 8;

    const int KT = K / BK;
    load(0);
    store(0);
    __syncthreads();
    for (int kt = 0; kt < KT; kt++) {
        const int s = kt & 1;
        if (kt + 1 < KT) load((kt + 1) * BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) ldsm_x4(af[mt], smem_addr(&As[s][a_lrow + mt * 16][kk + a_lcol]));
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t r[4];
                ldsm_x4(r, smem_addr(&Ws[s][b_lrow + np * 16][kk + b_lcol]));
                bf[np * 2][0] = r[0]; bf[np * 2][1] = r[1];
                bf[np * 2 + 1][0] = r[2]; bf[np * 2 + 1][1] = r[3];
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    if (F16ACC) mma_f16(hacc[mt][nt], af[mt], bf[nt]);
                    else mma_f32(acc[mt][nt], af[mt], bf[nt]);
                }
        }
        if (F16ACC) {
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) { flush(&acc[mt][nt][0], hacc[mt][nt][0]); flush(&acc[mt][nt][2], hacc[mt][nt][1]); }
        }
        if (kt + 1 < KT) store(s ^ 1);
        __syncthreads();
    }

    // epilogue: fragment (mt, nt) holds rows g and g+8, columns 2t and 2t+1
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int col = n0 + wn * 32 + nt * 8 + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm * 64 + mt * 16 + g + 8 * h;
                if (row >= M) continue;
                float2* p = reinterpret_cast<float2*>(C + (size_t)row * N + col);
                float2 v = make_float2(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
                if (beta != 0.f) { float2 o = *p; v.x += beta * o.x; v.y += beta * o.y; }
                *p = v;
            }
        }
}

inline bool eligible(int M, int N, int K) { return M >= 1 && N % BN == 0 && K % BK == 0; }

inline void launch(float* C, const float* A, const uint16_t* W, int M, int N, int K, float beta, cudaStream_t s, bool f16w, bool f16acc = true) {
    dim3 grid(N / BN, (M + BM - 1) / BM);
    if (f16w) {
        if (f16acc) k_gemm<true, true><<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
        else k_gemm<true, false><<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
    } else {
        if (f16acc) k_gemm<false, true><<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
        else k_gemm<false, false><<<grid, 256, 0, s>>>(C, A, W, M, N, K, beta);
    }
}

}  // namespace tc
