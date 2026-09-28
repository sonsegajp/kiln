// Does this GPU execute tensor-core mma.sync (fp16, m16n8k8)? Checks one 16x8x8 product against the CPU
// for both accumulator types, then times a long chain of mma to show the hardware rate.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

__global__ void k_one(const __half* A, const __half* B, float* C32, __half* C16) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    // A 16x8 row-major, B 8x8 stored [n][k] (the "col" operand)
    uint32_t a0 = *(const uint32_t*)&A[g * 8 + 2 * t], a1 = *(const uint32_t*)&A[(g + 8) * 8 + 2 * t];
    uint32_t b0 = *(const uint32_t*)&B[g * 8 + 2 * t];
    float c[4] = {0, 0, 0, 0};
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
    uint32_t h[2] = {0, 0};
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(h[0]), "+r"(h[1]) : "r"(a0), "r"(a1), "r"(b0));
    C32[g * 8 + 2 * t] = c[0]; C32[g * 8 + 2 * t + 1] = c[1];
    C32[(g + 8) * 8 + 2 * t] = c[2]; C32[(g + 8) * 8 + 2 * t + 1] = c[3];
    *(uint32_t*)&C16[g * 8 + 2 * t] = h[0];
    *(uint32_t*)&C16[(g + 8) * 8 + 2 * t] = h[1];
}

template <bool F16ACC>
__global__ void k_rate(float* out, int iters) {
    uint32_t a0 = 0x3c003c00u ^ threadIdx.x, a1 = a0 + 1, b0 = a0 + 2;
    float c[4] = {0, 0, 0, 0};
    uint32_t h[4] = {0, 0, 0, 0};
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int r = 0; r < 4; r++) {  // 4 independent chains per warp
            if (F16ACC) asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                                     : "+r"(h[r]), "+r"(h[(r + 1) & 3]) : "r"(a0), "r"(a1), "r"(b0));
            else asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                              : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
        }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1] + c[2] + c[3] + (float)(h[0] ^ h[1] ^ h[2] ^ h[3]);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s  sm_%d%d  %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    __half hA[128], hB[64];
    float fA[128], fB[64];
    for (int i = 0; i < 128; i++) { fA[i] = (float)((i * 7) % 11 - 5) * 0.25f; hA[i] = __float2half(fA[i]); }
    for (int i = 0; i < 64; i++) { fB[i] = (float)((i * 5) % 9 - 4) * 0.5f; hB[i] = __float2half(fB[i]); }
    __half *dA, *dB, *dC16; float* dC32;
    cudaMalloc(&dA, 256); cudaMalloc(&dB, 128); cudaMalloc(&dC32, 512); cudaMalloc(&dC16, 256);
    cudaMemcpy(dA, hA, 256, cudaMemcpyHostToDevice); cudaMemcpy(dB, hB, 128, cudaMemcpyHostToDevice);
    k_one<<<1, 32>>>(dA, dB, dC32, dC16);
    cudaError_t e = cudaDeviceSynchronize();
    printf("mma launch: %s\n", cudaGetErrorString(e));
    if (e != cudaSuccess) return 1;
    float c32[128]; __half c16[128];
    cudaMemcpy(c32, dC32, 512, cudaMemcpyDeviceToHost); cudaMemcpy(c16, dC16, 256, cudaMemcpyDeviceToHost);
    double err32 = 0, err16 = 0;
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) {
            double r = 0;
            for (int k = 0; k < 8; k++) r += (double)fA[m * 8 + k] * fB[n * 8 + k];
            err32 = fmax(err32, fabs(c32[m * 8 + n] - r));
            err16 = fmax(err16, fabs(__half2float(c16[m * 8 + n]) - r));
        }
    printf("max err f32-acc %.3g  f16-acc %.3g\n", err32, err16);
    float* dout; cudaMalloc(&dout, 1 << 22);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = p.multiProcessorCount * 8, iters = 4096;
    for (int f = 0; f < 2; f++) {
        for (int w = 0; w < 2; w++) {
            cudaEventRecord(e0);
            if (f) k_rate<true><<<blocks, 128>>>(dout, iters); else k_rate<false><<<blocks, 128>>>(dout, iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            double flop = 2.0 * 16 * 8 * 8 * 4.0 * iters * blocks * 4;  // per warp: 4 mma per iter, 4 warps per block
            if (w) printf("%s accumulate: %.2f TFLOPS\n", f ? "f16" : "f32", flop / ms / 1e9);
        }
    }
    return 0;
}
