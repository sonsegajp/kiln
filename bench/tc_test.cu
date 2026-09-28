// Accuracy of tc::k_gemm (tensor cores) vs cuBLAS fp32 and vs hgemm on Anima's shapes. On a GTX 16xx the
// tensor-core instructions are emulated, so only the error columns mean anything there.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include "../engine/src/hgemm.cuh"
#include "../engine/src/tcgemm.cuh"

static uint16_t to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16; }
static float from_bf16(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 10;
    cublasHandle_t h; cublasCreate(&h);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    struct S { int m, n, k; float beta; const char* name; } shapes[] = {
        {1536, 2048, 2048, 0.f, "q/k/v/o   1536x2048x2048"}, {1536, 8192, 2048, 0.f, "mlp1      1536x8192x2048"},
        {1536, 2048, 8192, 1.f, "mlp2+res  1536x2048x8192"}, {1500, 2048, 2048, 1.f, "ragged+res 1500x2048x2048"},
        {77, 2048, 2048, 0.f, "tiny M    77x2048x2048"},
    };
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& s : shapes) {
        std::vector<float> A((size_t)s.m * s.k), Wf((size_t)s.n * s.k), C0((size_t)s.m * s.n);
        std::vector<uint16_t> Wb(Wf.size());
        for (auto& v : A) v = nd(rng) * (s.k == 8192 ? 0.3f : 1.f);
        for (auto& v : C0) v = nd(rng);
        for (size_t i = 0; i < Wf.size(); i++) { Wb[i] = to_bf16(nd(rng) * 0.02f); Wf[i] = from_bf16(Wb[i]); }
        float *dA, *dWf, *dC, *dR; uint16_t* dWb;
        cudaMalloc(&dA, A.size() * 4); cudaMalloc(&dWf, Wf.size() * 4); cudaMalloc(&dWb, Wb.size() * 2);
        cudaMalloc(&dC, C0.size() * 4); cudaMalloc(&dR, C0.size() * 4);
        cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dWf, Wf.data(), Wf.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dWb, Wb.data(), Wb.size() * 2, cudaMemcpyHostToDevice);
        const float one = 1;
        auto reset = [&](float* d) { cudaMemcpy(d, C0.data(), C0.size() * 4, cudaMemcpyHostToDevice); };
        reset(dR);
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &one, dWf, s.k, dA, s.k, &s.beta, dR, s.n);
        std::vector<float> R(C0.size()), Cc(C0.size());
        cudaMemcpy(R.data(), dR, R.size() * 4, cudaMemcpyDeviceToHost);
        auto err = [&]() {
            cudaMemcpy(Cc.data(), dC, Cc.size() * 4, cudaMemcpyDeviceToHost);
            double num = 0, den = 0;
            for (size_t i = 0; i < R.size(); i++) { double d = Cc[i] - R[i]; num += d * d; den += (double)R[i] * R[i]; }
            return sqrt(num / den);
        };
        auto timeit = [&](auto fn) {
            fn();
            cudaEventRecord(e0); for (int i = 0; i < reps; i++) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / reps;
        };
        double flop = 2.0 * s.m * s.n * s.k;
        printf("%s\n", s.name);
        if (hg::eligible(s.m, s.n, s.k)) {
            reset(dC); hg::launch(dC, dA, dWb, s.m, s.n, s.k, s.beta, 0, 1); double e = err();
            float t = timeit([&] { hg::launch(dC, dA, dWb, s.m, s.n, s.k, 0.f, 0, 1); });
            printf("   hgemm (HFMA2)       rel_l2 %.2e   %7.2f ms %6.2f TF\n", e, t, flop / t / 1e9);
        }
        for (int f16acc = 1; f16acc >= 0; f16acc--) {
            reset(dC); tc::launch(dC, dA, dWb, s.m, s.n, s.k, s.beta, 0, false, f16acc); double e = err();
            float t = timeit([&] { tc::launch(dC, dA, dWb, s.m, s.n, s.k, 0.f, 0, false, f16acc); });
            printf("   tc %s acc          rel_l2 %.2e   %7.2f ms %6.2f TF   %s\n", f16acc ? "f16" : "f32", e, t, flop / t / 1e9,
                   cudaGetErrorString(cudaGetLastError()));
        }
        cudaFree(dA); cudaFree(dWf); cudaFree(dWb); cudaFree(dC); cudaFree(dR);
    }
    return 0;
}
