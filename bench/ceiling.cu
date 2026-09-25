// Measures what this GPU can actually do on Anima's shapes at 512x768:
// 1536 image tokens, width 2048, MLP 8192, 16 heads x 128.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

static float time_ms(cudaEvent_t a, cudaEvent_t b) { float ms; cudaEventElapsedTime(&ms, a, b); return ms; }

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s  SMs=%d  sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    cublasHandle_t h; cublasCreate(&h);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    const int N = 1536, D = 2048, F = 8192;
    struct Shape { int m, n, k; const char* name; } shapes[] = {
        {N, 3 * D, D, "qkv   1536x6144x2048"},
        {N, F, D,     "mlp1  1536x8192x2048"},
        {N, D, F,     "mlp2  1536x2048x8192"},
        {N, D, D,     "proj  1536x2048x2048"},
    };
    size_t maxe = (size_t)F * D;
    half *A, *B, *C; float *Af, *Bf, *Cf;
    CK(cudaMalloc(&A, maxe * 2)); CK(cudaMalloc(&B, maxe * 2)); CK(cudaMalloc(&C, maxe * 2));
    CK(cudaMalloc(&Af, maxe * 4)); CK(cudaMalloc(&Bf, maxe * 4)); CK(cudaMalloc(&Cf, maxe * 4));
    cudaMemset(A, 0, maxe * 2); cudaMemset(B, 0, maxe * 2); cudaMemset(Af, 0, maxe * 4); cudaMemset(Bf, 0, maxe * 4);

    for (auto& s : shapes) {
        double flop = 2.0 * s.m * s.n * s.k;
        const float one = 1, zero = 0; const half hone = __float2half(1.f), hzero = __float2half(0.f);
        struct Mode { const char* n; int kind; } modes[] = {{"fp16 acc16", 0}, {"fp16 acc32", 1}, {"fp32      ", 2}};
        for (auto& m : modes) {
            auto run = [&]() {
                if (m.kind == 0) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &hone, B, CUDA_R_16F, s.k, A, CUDA_R_16F, s.k, &hzero, C, CUDA_R_16F, s.n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
                else if (m.kind == 1) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &one, B, CUDA_R_16F, s.k, A, CUDA_R_16F, s.k, &zero, C, CUDA_R_16F, s.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                else cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &one, Bf, s.k, Af, s.k, &zero, Cf, s.n);
            };
            for (int i = 0; i < 3; i++) run();
            cudaEventRecord(e0); for (int i = 0; i < 20; i++) run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            double ms = time_ms(e0, e1) / 20;
            printf("%s  %s  %7.2f ms  %5.2f TFLOPS\n", s.name, m.n, ms, flop / ms / 1e9);
        }
    }
    CK(cudaGetLastError());
    return 0;
}
