// Tensor-core GEMM for the fp16-activation forward path (RTX GPUs):
//   acc = A[M,K] * W[N,K]^T  (+ A2[M,K2] * W2[N,K2]^T, the folded LoRA product),  all four fp16.
// The epilogue writes acc as fp32 (+ beta*C), as fp16, as fp16 GELU(acc), or adds gate[col]*acc into an fp32
// residual stream in place. Fusing those saves the fp32 round trips of the separate elementwise kernels.
//
// Same tiling and accumulation scheme as tcgemm.cuh (128x128 block tile, 8 warps of 64x32, k tile 32, fp16
// accumulation flushed into fp32 every k tile), so for identical fp16 inputs the result is bit-identical to
// tc::k_gemm. What differs is the staging: with fp16 inputs there is nothing to convert, so on sm_80+ tiles
// go global -> shared with cp.async through a STAGES-deep ring, and no registers are spent on staging.
// sm_75 (Turing RTX) has no cp.async: the same loop copies synchronously (STAGES = 2).
// Blocks are rasterized in groups of GROUP_M row tiles, so the blocks resident at once share weight tiles in L2.
// Requirements: N % 128 == 0, K % 32 == 0, K2 % 32 == 0, 16-byte aligned rows.
#pragma once
#include "tcgemm.cuh"

namespace tc16 {

constexpr int BM = 128, BN = 128, BK = 32, LDS = BK + 8, GROUP_M = 8;

enum Epi { EPI_F32 = 0, EPI_F16 = 1, EPI_GELU16 = 2, EPI_RESID = 3 };

struct Args {
    const __half* A = nullptr;
    const __half* W = nullptr;
    int K = 0;
    const __half* A2 = nullptr;  // second product (LoRA): [M, K2] and [N, K2]; K2 = 0: none
    const __half* W2 = nullptr;
    int K2 = 0;
    void* C = nullptr;           // fp32 (EPI_F32, EPI_RESID) or fp16 (EPI_F16, EPI_GELU16) rows of ldc elements
    int ldc = 0;
    const float* gate = nullptr; // EPI_RESID: C[r, c] += gate[c] * acc (null = 1)
    float beta = 0.f;            // EPI_F32: C = acc + beta * C
    int M = 0, N = 0;
};

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(tc::smem_addr(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
#else
    *reinterpret_cast<uint4*>(dst) = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }  // = k_gelu

template <int EPI, int STAGES>
__global__ void __launch_bounds__(256, 1) k_gemm16(const Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half (*As)[BM][LDS] = reinterpret_cast<__half (*)[BM][LDS]>(smem);
    __half (*Ws)[BN][LDS] = reinterpret_cast<__half (*)[BN][LDS]>(smem + (size_t)STAGES * BM * LDS * sizeof(__half));
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;

    // grouped rasterization: GROUP_M row tiles x every column tile, column-major inside the group
    const int tiles_m = (a.M + BM - 1) / BM, tiles_n = a.N / BN, per_group = GROUP_M * tiles_n;
    const int bid = blockIdx.x, first = (bid / per_group) * GROUP_M, gsz = min(tiles_m - first, GROUP_M), in_g = bid % per_group;
    const int m0 = (first + in_g % gsz) * BM, n0 = (in_g / gsz) * BN;

    const int KT1 = a.K / BK, KT = KT1 + a.K2 / BK;
    // one k tile: A 128x32 and W 128x32 fp16 = 512 16-byte chunks each, 2 + 2 per thread
    auto load = [&](int kt, int s) {
        const bool first_product = kt < KT1;
        const __half* pa = first_product ? a.A : a.A2;
        const __half* pw = first_product ? a.W : a.W2;
        const int ld = first_product ? a.K : a.K2, k0 = (first_product ? kt : kt - KT1) * BK;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = tid + i * 256, row = c >> 2, col = (c & 3) * 8;
            const int m = m0 + row;
            const bool ok = m < a.M;
            cp16(&As[s][row][col], pa + (size_t)(ok ? m : 0) * ld + k0 + col, ok);  // rows past M read as zeros
            cp16(&Ws[s][row][col], pw + (size_t)(n0 + row) * ld + k0 + col, true);
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

    const int a_lrow = wm * 64 + (lane & 15), a_lcol = (lane >> 4) * 8;
    const int b_lrow = wn * 32 + (lane & 7) + ((lane >> 4) << 3), b_lcol = ((lane >> 3) & 1) * 8;

    // Ring of STAGES tiles. Before the wait of iteration kt, STAGES-1+kt groups were committed (one per tile,
    // empty ones included), so waiting until at most STAGES-2 are pending means tile kt has landed. The
    // barrier then publishes it to every warp and also guarantees that everyone finished computing on the
    // slot of tile kt-1, which is the slot refilled with tile kt+STAGES-1 right after.
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load(s, s);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load(nk, nk % STAGES);
            cp_commit();
        }
        const int s = kt % STAGES;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) tc::ldsm_x4(af[mt], tc::smem_addr(&As[s][a_lrow + mt * 16][kk + a_lcol]));
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t r[4];
                tc::ldsm_x4(r, tc::smem_addr(&Ws[s][b_lrow + np * 16][kk + b_lcol]));
                bf[np * 2][0] = r[0]; bf[np * 2][1] = r[1];
                bf[np * 2 + 1][0] = r[2]; bf[np * 2 + 1][1] = r[3];
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) tc::mma_f16(hacc[mt][nt], af[mt], bf[nt]);
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) { tc::flush(&acc[mt][nt][0], hacc[mt][nt][0]); tc::flush(&acc[mt][nt][2], hacc[mt][nt][1]); }
    }
    cp_wait<0>();

    // epilogue: fragment (mt, nt) holds rows g and g+8, columns 2t and 2t+1
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        const int col = n0 + wn * 32 + nt * 8 + 2 * t;
        float g0 = 1.f, g1 = 1.f;
        if (EPI == EPI_RESID && a.gate) { g0 = a.gate[col]; g1 = a.gate[col + 1]; }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm * 64 + mt * 16 + g + 8 * h;
                if (row >= a.M) continue;
                const float v0 = acc[mt][nt][2 * h], v1 = acc[mt][nt][2 * h + 1];
                const size_t off = (size_t)row * a.ldc + col;
                if (EPI == EPI_F32) {
                    float2* p = reinterpret_cast<float2*>(static_cast<float*>(a.C) + off);
                    float2 v = make_float2(v0, v1);
                    if (a.beta != 0.f) { float2 o = *p; v.x += a.beta * o.x; v.y += a.beta * o.y; }
                    *p = v;
                } else if (EPI == EPI_RESID) {
                    float2* p = reinterpret_cast<float2*>(static_cast<float*>(a.C) + off);
                    float2 o = *p;
                    o.x += g0 * v0;
                    o.y += g1 * v1;
                    *p = o;
                } else {
                    const float w0 = EPI == EPI_GELU16 ? gelu(v0) : v0, w1 = EPI == EPI_GELU16 ? gelu(v1) : v1;
                    *reinterpret_cast<__half2*>(static_cast<__half*>(a.C) + off) = __floats2half2_rn(w0, w1);
                }
            }
    }
}

inline bool eligible(int M, int N, int K) { return M >= 1 && N % BN == 0 && K % BK == 0; }

template <int STAGES>
constexpr size_t smem_bytes() { return (size_t)STAGES * (BM + BN) * LDS * sizeof(__half); }

template <int EPI, int STAGES>
inline void launch_s(const Args& a, cudaStream_t s) {
    static bool attr = false;  // > 48 KB of dynamic shared memory must be opted into once per kernel
    if (!attr && smem_bytes<STAGES>() > 48 * 1024) {
        cudaFuncSetAttribute(k_gemm16<EPI, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes<STAGES>());
        attr = true;
    }
    const int tiles = (a.M + BM - 1) / BM * (a.N / BN);
    k_gemm16<EPI, STAGES><<<tiles, 256, smem_bytes<STAGES>(), s>>>(a);
}

// sm >= 80: 4-stage cp.async ring (80 KB of shared memory); sm_75: synchronous double buffer
template <int EPI>
inline void launch(const Args& a, cudaStream_t s, int sm) {
    if (sm >= 80) launch_s<EPI, 4>(a, s);
    else launch_s<EPI, 2>(a, s);
}

}  // namespace tc16
