// Fused multi-head attention for GPUs without tensor cores (Turing TU11x), head_dim 128.
// One block = 64 queries of one head; keys/values stream through shared memory 64 at a time with
// an online softmax, so the [Tq, Tk] score matrix never exists in memory.
// Both matmuls run as HFMA2 with operands packed in pairs along the contracted dimension
// ((q_d, q_d+1) x (k_d, k_d+1), (p_k, p_k+1) x (v_k, v_k+1)): every instruction is two useful
// multiply-adds with no register shuffles. Partial sums stay short (16-32 products per fp16 lane)
// before they are flushed into fp32.
#pragma once
#include <cuda_fp16.h>

namespace fa {

constexpr int D = 128, BR = 64, BC = 64, DP = D / 2, KP = BC / 2;

struct Smem {
    __half2 Q[DP][BR];  // (q_d, q_d+1) per query, pre-scaled by log2(e)/sqrt(D)
    __half2 K[DP][BC];  // (k_d, k_d+1) per key
    __half2 V[KP][D];   // (v_k, v_k+1) per dim
    __half2 P[KP][BR];  // (p_k, p_k+1) per query
};

__device__ __forceinline__ float hsum(__half2 h) { float2 f = __half22float2(h); return f.x + f.y; }

__global__ void __launch_bounds__(256, 1)
flash_fwd(float* __restrict__ out, int ldo, const float* __restrict__ q, int ldq, const float* __restrict__ k, int ldk,
          const float* __restrict__ v, int ldv, int Tq, int Tk, float qscale, int npad) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int h = blockIdx.y, q0 = blockIdx.x * BR;
    q += h * D; k += h * D; v += h * D; out += h * D;
    const float4 zero4 = make_float4(0, 0, 0, 0);

#pragma unroll
    for (int it = 0; it < 8; it++) {
        int idx = tid + it * 256, row = idx % BR, dq = idx / BR;
        float4 f = q0 + row < Tq ? *(const float4*)(q + (size_t)(q0 + row) * ldq + dq * 4) : zero4;
        sm.Q[dq * 2][row] = __floats2half2_rn(f.x * qscale, f.y * qscale);
        sm.Q[dq * 2 + 1][row] = __floats2half2_rn(f.z * qscale, f.w * qscale);
    }

    float o[4][8], m[4], l[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        m[r] = -INFINITY; l[r] = 0.f;
#pragma unroll
        for (int d = 0; d < 8; d++) o[r][d] = 0.f;
    }

    for (int k0 = 0; k0 < Tk; k0 += BC) {
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 8; it++) {
            int idx = tid + it * 256, key = idx % BC, dq = idx / BC;
            float4 f = k0 + key < Tk ? *(const float4*)(k + (size_t)(k0 + key) * ldk + dq * 4) : zero4;
            sm.K[dq * 2][key] = __floats2half2_rn(f.x, f.y);
            sm.K[dq * 2 + 1][key] = __floats2half2_rn(f.z, f.w);
        }
#pragma unroll
        for (int it = 0; it < 4; it++) {
            int idx = tid + it * 256, kp = idx / 32, dq = idx % 32;
            int ka = k0 + 2 * kp, kb = ka + 1;
            float4 a = ka < Tk ? *(const float4*)(v + (size_t)ka * ldv + dq * 4) : zero4;
            float4 b = kb < Tk ? *(const float4*)(v + (size_t)kb * ldv + dq * 4) : zero4;
            __half2 p4[4] = {__floats2half2_rn(a.x, b.x), __floats2half2_rn(a.y, b.y), __floats2half2_rn(a.z, b.z), __floats2half2_rn(a.w, b.w)};
            *(uint4*)&sm.V[kp][dq * 4] = *(uint4*)p4;
        }
        __syncthreads();

        // S = Q K^T for rows ty*4.., keys tx*4..
        float s[4][4];
        __half2 ps[4][4];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) { s[r][c] = 0.f; ps[r][c] = __float2half2_rn(0.f); }
#pragma unroll 16
        for (int p = 0; p < DP; p++) {
            uint4 qa = *(const uint4*)&sm.Q[p][ty * 4];
            uint4 kb = *(const uint4*)&sm.K[p][tx * 4];
            const __half2* q2 = (const __half2*)&qa;
            const __half2* k2 = (const __half2*)&kb;
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) ps[r][c] = __hfma2(q2[r], k2[c], ps[r][c]);
            if ((p & 15) == 15) {
#pragma unroll
                for (int r = 0; r < 4; r++)
#pragma unroll
                    for (int c = 0; c < 4; c++) { s[r][c] += hsum(ps[r][c]); ps[r][c] = __float2half2_rn(0.f); }
            }
        }

        // online softmax (log2 domain); a row's 64 scores live in the 16 lanes sharing ty
#pragma unroll
        for (int r = 0; r < 4; r++) {
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                if (k0 + tx * 4 + c >= Tk) s[r][c] = -INFINITY;
                mx = fmaxf(mx, s[r][c]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            float mn = fmaxf(m[r], mx);
            float corr = exp2f(m[r] - mn);
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < 4; c++) { s[r][c] = exp2f(s[r][c] - mn); sum += s[r][c]; }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            l[r] = l[r] * corr + sum;
            m[r] = mn;
#pragma unroll
            for (int d = 0; d < 8; d++) o[r][d] *= corr;
            sm.P[tx * 2][ty * 4 + r] = __floats2half2_rn(s[r][0], s[r][1]);
            sm.P[tx * 2 + 1][ty * 4 + r] = __floats2half2_rn(s[r][2], s[r][3]);
        }
        __syncthreads();

        // O += P V for rows ty*4.., dims tx*8..
        __half2 po[4][8];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int d = 0; d < 8; d++) po[r][d] = __float2half2_rn(0.f);
#pragma unroll 8
        for (int kp = 0; kp < KP; kp++) {
            uint4 pa = *(const uint4*)&sm.P[kp][ty * 4];
            uint4 va = *(const uint4*)&sm.V[kp][tx * 8];
            uint4 vb = *(const uint4*)&sm.V[kp][tx * 8 + 4];
            const __half2* p2 = (const __half2*)&pa;
            __half2 v2[8];
            *(uint4*)&v2[0] = va;
            *(uint4*)&v2[4] = vb;
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int d = 0; d < 8; d++) po[r][d] = __hfma2(p2[r], v2[d], po[r][d]);
        }
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int d = 0; d < 8; d++) o[r][d] += hsum(po[r][d]);
    }

    // npad all-zero keys: score 0, value 0 -> only the denominator grows
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = q0 + ty * 4 + r;
        if (row >= Tq) continue;
        float lr = l[r], scale = 1.f;
        if (npad) {
            float mn = fmaxf(m[r], 0.f);
            scale = exp2f(m[r] - mn);
            lr = lr * scale + (float)npad * exp2f(-mn);
        }
        float inv = scale / lr;
        float* dst = out + (size_t)row * ldo + tx * 8;
        *(float4*)dst = make_float4(o[r][0] * inv, o[r][1] * inv, o[r][2] * inv, o[r][3] * inv);
        *(float4*)(dst + 4) = make_float4(o[r][4] * inv, o[r][5] * inv, o[r][6] * inv, o[r][7] * inv);
    }
}

inline void launch(float* out, int ldo, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                   int Tq, int Tk, int H, int npad, cudaStream_t s) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(flash_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
        init = true;
    }
    const float qscale = 1.4426950408889634f / sqrtf((float)D);
    dim3 grid((Tq + BR - 1) / BR, H);
    flash_fwd<<<grid, 256, sizeof(Smem), s>>>(out, ldo, q, ldq, k, ldk, v, ldv, Tq, Tk, qscale, npad);
}

// Every row pointer must be 16-byte aligned (float4 loads/stores).
inline bool eligible(int Dh, int ldq, int ldk, int ldv, int ldo) {
    return Dh == D && ldq % 4 == 0 && ldk % 4 == 0 && ldv % 4 == 0 && ldo % 4 == 0;
}

}  // namespace fa
