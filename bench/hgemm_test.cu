// Accuracy + speed of hg::hgemm_nt vs cuBLAS fp32 on Anima's shapes.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include "../engine/src/hgemm.cuh"

static uint16_t to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16; }
static float from_bf16(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static int reps = 30;
int main(int argc, char** argv) {
    if (argc > 1) reps = atoi(argv[1]);
    cublasHandle_t h; cublasCreate(&h);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    struct S { int m, n, k; const char* name; } shapes[] = {
        {1536, 2048, 2048, "q/k/v/o  1536x2048x2048"}, {1536, 8192, 2048, "mlp1     1536x8192x2048"},
        {1536, 2048, 8192, "mlp2     1536x2048x8192"}, {3952, 8192, 2048, "mlp1 big 3952x8192x2048"},
    };
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& s : shapes) {
        std::vector<float> A((size_t)s.m * s.k), Wf((size_t)s.n * s.k);
        std::vector<uint16_t> Wb(Wf.size());
        for (auto& v : A) v = nd(rng) * (s.k == 8192 ? 0.3f : 1.f);  // post-GELU inputs are smaller
        for (size_t i = 0; i < Wf.size(); i++) { Wb[i] = to_bf16(nd(rng) * 0.02f); Wf[i] = from_bf16(Wb[i]); }
        float *dA, *dWf, *dC, *dR; uint16_t* dWb;
        cudaMalloc(&dA, A.size() * 4); cudaMalloc(&dWf, Wf.size() * 4); cudaMalloc(&dWb, Wb.size() * 2);
        cudaMalloc(&dC, (size_t)s.m * s.n * 4); cudaMalloc(&dR, (size_t)s.m * s.n * 4);
        cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dWf, Wf.data(), Wf.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dWb, Wb.data(), Wb.size() * 2, cudaMemcpyHostToDevice);
        const float one = 1, zero = 0;
        auto ref = [&] { cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, s.m, s.k, &one, dWf, s.k, dA, s.k, &zero, dR, s.n); };
        auto ours = [&] { hg::launch(dC, dA, dWb, s.m, s.n, s.k, 0.f, 0, 1); };
        auto ours2 = [&] { hg::launch(dC, dA, dWb, s.m, s.n, s.k, 0.f, 0, 2); };
        double flop = 2.0 * s.m * s.n * s.k;
        auto timeit = [&](auto fn) {
            for (int i = 0; i < 5; i++) fn();
            cudaEventRecord(e0); for (int i = 0; i < reps; i++) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / reps;
        };
        // warm the clocks up
        for (int i = 0; i < 100; i++) ref();
        float tr = timeit(ref), to = timeit(ours), t2 = timeit(ours2);
        std::vector<float> R((size_t)s.m * s.n), Cc(R.size());
        cudaMemcpy(R.data(), dR, R.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(Cc.data(), dC, Cc.size() * 4, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (size_t i = 0; i < R.size(); i++) { double d = Cc[i] - R[i]; num += d * d; den += (double)R[i] * R[i]; }
        printf("%s  cublas32 %5.2f TF | v1 %5.2f TF | v2 %6.2f ms %5.2f TF | v2 rel_l2 %.2e  %s\n", s.name, flop / tr / 1e9,
               flop / to / 1e9, t2, flop / t2 / 1e9, sqrt(num / den), cudaGetErrorString(cudaGetLastError()));
        cudaFree(dA); cudaFree(dWf); cudaFree(dWb); cudaFree(dC); cudaFree(dR);
    }
    return 0;
}
