// Fused multi-head attention on tensor cores (RTX GPUs), head_dim 128; the same contract as flash.cuh.
// FlashAttention-2 layout: one block = 64 queries of one head, 4 warps, each warp owns 16 query rows for
// the whole pass, so the softmax statistics never leave registers. Keys/values stream through shared
// memory 64 at a time with an online softmax (base 2: Q is pre-scaled by log2(e)/sqrt(D)).
//   S = Q K^T : Q fragments stay in registers; K rows are the "col" operand as stored ([key][dim]).
//   O += P V  : the S accumulator layout of two adjacent key tiles is exactly the A-fragment layout of
//               one k16 step, so P goes from the softmax straight into the next mma without shared memory;
//               V comes in through ldmatrix.trans.
// Precision: QK^T accumulates in fp32 (softmax is sensitive to absolute score error). PV accumulates in
// fp16 over 32 keys (P <= 1) and is flushed into fp32, at the fast fp16-accumulate rate of GeForce cards.
// The context's all-zero padding keys are handled analytically, as in flash.cuh.
#pragma once
#include "tcgemm.cuh"

namespace tfa {

constexpr int D = 128, BR = 64, BC = 64, LDS = D + 8;  // padded rows: conflict-free ldmatrix

__global__ void __launch_bounds__(128, 2)
k_flash(float* __restrict__ out, int ldo, const float* __restrict__ q, int ldq, const float* __restrict__ k, int ldk,
        const float* __restrict__ v, int ldv, int Tq, int Tk, float qscale, int npad) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half (*Ks)[LDS] = reinterpret_cast<__half (*)[LDS]>(smem_raw);  // [BC][LDS]; holds Q first
    __half (*Vs)[LDS] = Ks + BC;                                         // [BC][LDS]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y, q0 = blockIdx.x * BR;
    q += h * D; k += h * D; v += h * D; out += h * D;
    const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);

    // Q tile -> shared (fp16, pre-scaled) -> each warp's fragments in registers
#pragma unroll
    for (int it = 0; it < 16; it++) {
        int idx = tid + it * 128, row = idx >> 5, c4 = (idx & 31) * 4;
        float4 f = q0 + row < Tq ? *reinterpret_cast<const float4*>(q + (size_t)(q0 + row) * ldq + c4) : zero4;
        *reinterpret_cast<uint2*>(&Ks[row][c4]) = make_uint2(tc::h2_bits(__floats2half2_rn(f.x * qscale, f.y * qscale)),
                                                             tc::h2_bits(__floats2half2_rn(f.z * qscale, f.w * qscale)));
    }
    __syncthreads();
    uint32_t qf[8][4];
#pragma unroll
    for (int c = 0; c < 8; c++) tc::ldsm_x4(qf[c], tc::smem_addr(&Ks[warp * 16 + (lane & 15)][c * 16 + (lane >> 4) * 8]));

    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};  // rows g and g+8; l is this lane's partial sum

    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_col = ((lane >> 3) & 1) * 8;   // K as the "col" operand
    const int v_row = (lane & 7) + ((lane >> 3) & 1) * 8, v_col = (lane >> 4) * 8;      // V through .trans

    for (int k0 = 0; k0 < Tk; k0 += BC) {
        __syncthreads();  // everyone is done with the previous K/V (or with Q)
#pragma unroll 4
        for (int it = 0; it < 16; it++) {
            int idx = tid + it * 128, row = idx >> 5, c4 = (idx & 31) * 4;
            bool in = k0 + row < Tk;
            float4 a = in ? *reinterpret_cast<const float4*>(k + (size_t)(k0 + row) * ldk + c4) : zero4;
            float4 b = in ? *reinterpret_cast<const float4*>(v + (size_t)(k0 + row) * ldv + c4) : zero4;
            *reinterpret_cast<uint2*>(&Ks[row][c4]) = make_uint2(tc::h2_bits(__floats2half2_rn(a.x, a.y)), tc::h2_bits(__floats2half2_rn(a.z, a.w)));
            *reinterpret_cast<uint2*>(&Vs[row][c4]) = make_uint2(tc::h2_bits(__floats2half2_rn(b.x, b.y)), tc::h2_bits(__floats2half2_rn(b.z, b.w)));
        }
        __syncthreads();

        // S = Q K^T: 16 rows x 64 keys = 8 key tiles
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int c = 0; c < 8; c++)
#pragma unroll
            for (int np = 0; np < 4; np++) {
                uint32_t r[4];
                tc::ldsm_x4(r, tc::smem_addr(&Ks[np * 16 + b_row][c * 16 + b_col]));
                const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                tc::mma_f32(s[np * 2], qf[c], b0);
                tc::mma_f32(s[np * 2 + 1], qf[c], b1);
            }

        // online softmax; a row's 64 scores live in the 4 lanes of one quad
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int key = k0 + j * 8 + 2 * t;
            if (key >= Tk) { s[j][0] = s[j][2] = -INFINITY; }
            if (key + 1 >= Tk) { s[j][1] = s[j][3] = -INFINITY; }
            mx[0] = fmaxf(mx[0], fmaxf(s[j][0], s[j][1]));
            mx[1] = fmaxf(mx[1], fmaxf(s[j][2], s[j][3]));
        }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);
            corr[r] = exp2f(m[r] - mn);
            m[r] = mn;
        }
        float sum[2] = {0.f, 0.f};
        uint32_t pa[4][4];  // P as A fragments, one per 16-key step
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float p0 = exp2f(s[j][0] - m[0]), p1 = exp2f(s[j][1] - m[0]);
            float p2 = exp2f(s[j][2] - m[1]), p3 = exp2f(s[j][3] - m[1]);
            sum[0] += p0 + p1;
            sum[1] += p2 + p3;
            pa[j >> 1][(j & 1) * 2] = tc::h2_bits(__floats2half2_rn(p0, p1));
            pa[j >> 1][(j & 1) * 2 + 1] = tc::h2_bits(__floats2half2_rn(p2, p3));
        }
        l[0] = l[0] * corr[0] + sum[0];
        l[1] = l[1] * corr[1] + sum[1];
#pragma unroll
        for (int j = 0; j < 16; j++) { o[j][0] *= corr[0]; o[j][1] *= corr[0]; o[j][2] *= corr[1]; o[j][3] *= corr[1]; }

        // O += P V, 16 dims at a time: fp16 accumulation over 32 keys, then flushed into o
#pragma unroll
        for (int dp = 0; dp < 8; dp++)
#pragma unroll
            for (int half = 0; half < 2; half++) {
                uint32_t oh[2][2] = {{0, 0}, {0, 0}};
#pragma unroll
                for (int kc = half * 2; kc < half * 2 + 2; kc++) {
                    uint32_t r[4];
                    tc::ldsm_x4_t(r, tc::smem_addr(&Vs[kc * 16 + v_row][dp * 16 + v_col]));
                    const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
                    tc::mma_f16(oh[0], pa[kc], b0);
                    tc::mma_f16(oh[1], pa[kc], b1);
                }
                tc::flush(&o[dp * 2][0], oh[0][0]); tc::flush(&o[dp * 2][2], oh[0][1]);
                tc::flush(&o[dp * 2 + 1][0], oh[1][0]); tc::flush(&o[dp * 2 + 1][2], oh[1][1]);
            }
    }

    // row sums across the quad; npad all-zero keys: score 0, value 0 -> only the denominator grows
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
        float scale = 1.f;
        if (npad) {
            const float mn = fmaxf(m[r], 0.f);
            scale = exp2f(m[r] - mn);
            l[r] = l[r] * scale + (float)npad * exp2f(-mn);
        }
        const float inv = scale / l[r];
        const int row = q0 + warp * 16 + g + 8 * r;
        if (row >= Tq) continue;
        float* dst = out + (size_t)row * ldo + 2 * t;
#pragma unroll
        for (int j = 0; j < 16; j++)
            *reinterpret_cast<float2*>(dst + j * 8) = make_float2(o[j][2 * r] * inv, o[j][2 * r + 1] * inv);
    }
}

constexpr size_t SMEM = (size_t)2 * BC * LDS * sizeof(__half);

inline void launch(float* out, int ldo, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                   int Tq, int Tk, int H, int npad, cudaStream_t s) {
    const float qscale = 1.4426950408889634f / sqrtf((float)D);
    dim3 grid((Tq + BR - 1) / BR, H);
    k_flash<<<grid, 128, SMEM, s>>>(out, ldo, q, ldq, k, ldk, v, ldv, Tq, Tk, qscale, npad);
}

// Every row pointer must be 16-byte aligned (float4 loads); ldo even (float2 stores).
inline bool eligible(int Dh, int ldq, int ldk, int ldv, int ldo) {
    return Dh == D && ldq % 4 == 0 && ldk % 4 == 0 && ldv % 4 == 0 && ldo % 2 == 0;
}

}  // namespace tfa
