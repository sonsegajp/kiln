// Raw ALU peak: fp32 FMA vs fp16x2 HFMA2 on this GPU.
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void fma32(float* out, int iters) {
    float a = threadIdx.x * 1e-7f, b = 0.999f, c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    for (int i = 0; i < iters; i++) {
        c0 = fmaf(a, b, c0); c1 = fmaf(a, b, c1); c2 = fmaf(a, b, c2); c3 = fmaf(a, b, c3);
        c4 = fmaf(a, b, c4); c5 = fmaf(a, b, c5); c6 = fmaf(a, b, c6); c7 = fmaf(a, b, c7);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}

__global__ void fma16(float* out, int iters) {
    __half2 a = __float2half2_rn(threadIdx.x * 1e-4f), b = __float2half2_rn(0.999f);
    __half2 c0 = __float2half2_rn(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (int i = 0; i < iters; i++) {
        c0 = __hfma2(a, b, c0); c1 = __hfma2(a, b, c1); c2 = __hfma2(a, b, c2); c3 = __hfma2(a, b, c3);
        c4 = __hfma2(a, b, c4); c5 = __hfma2(a, b, c5); c6 = __hfma2(a, b, c6); c7 = __hfma2(a, b, c7);
    }
    __half2 s = __hadd2(__hadd2(__hadd2(c0, c1), __hadd2(c2, c3)), __hadd2(__hadd2(c4, c5), __hadd2(c6, c7)));
    out[blockIdx.x * blockDim.x + threadIdx.x] = __low2float(s) + __high2float(s);
}

int main() {
    float* out;
    cudaMalloc(&out, 1 << 24);
    int blocks = 24 * 16, threads = 256, iters = 1 << 18;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int k = 0; k < 6; k++) {
        fma32<<<blocks, threads>>>(out, iters);
        cudaEventRecord(a); fma32<<<blocks, threads>>>(out, iters); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        double f32 = 2.0 * 8 * iters * (double)blocks * threads / ms / 1e9;
        fma16<<<blocks, threads>>>(out, iters);
        cudaEventRecord(a); fma16<<<blocks, threads>>>(out, iters); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        double f16 = 2.0 * 2 * 8 * iters * (double)blocks * threads / ms / 1e9;
        printf("fp32 FMA  %.2f TFLOPS\nfp16 HFMA2 %.2f TFLOPS\n", f32, f16);
    }
    return 0;
}
