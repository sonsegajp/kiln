// tc16::k_gemm16 (fp16 activations, tcgemm16.cuh) against tc::k_gemm (fp32 activations, tcgemm.cuh) on Anima's
// shapes. With A values that are exact in fp16 and fp16 weights, both see the same fp16 operands and use the same
// accumulation order, so every epilogue must match bit for bit (the second product against one GEMM over the
// concatenated K). Runs anywhere tc::k_gemm runs; on a GTX 16xx the mma instructions are emulated and only the
// synchronous staging path is exercised (cp.async needs sm_80), so the timings mean nothing there.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../engine/src/tcgemm16.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static uint16_t f2h(float f) { __half x = __float2half_rn(f); uint16_t h; memcpy(&h, &x, 2); return h; }

template <typename T>
static T* up(const std::vector<T>& h) {
    T* d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// GELU of the reference on the device (the same erff as the engine's k_gelu), rounded to fp16
__global__ void k_gelu16(uint16_t* y, const float* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half_as_ushort(__float2half_rn(tc16::gelu(x[i])));
}

static int fails = 0;
static void check(const char* what, size_t bad, size_t n, double maxd) {
    printf("  %-34s %s (%zu of %zu differ, max |d| %.3g)\n", what, bad ? "FAIL" : "ok", bad, n, maxd);
    fails += bad != 0;
}

template <int EPI>
static void run16(const tc16::Args& a, int stages) {
    if (stages == 3) tc16::launch_s<EPI, 3>(a, 0);
    else if (stages == 4) tc16::launch_s<EPI, 4>(a, 0);
    else tc16::launch_s<EPI, 2>(a, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

static void test(int M, int N, int K, int K2, int stages, std::mt19937& rng) {
    printf("M %d N %d K %d K2 %d, %d stages\n", M, N, K, K2, stages);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int KK = K + K2;
    // A, A2 as fp16 values; the reference kernel reads their exact fp32 equivalent over the concatenated K
    std::vector<uint16_t> A16((size_t)M * K), A2_16((size_t)M * std::max(K2, 1)), W16((size_t)N * K), W2_16((size_t)N * std::max(K2, 1));
    std::vector<float> Acat((size_t)M * KK);
    std::vector<uint16_t> Wcat((size_t)N * KK);
    for (int r = 0; r < M; r++)
        for (int k = 0; k < KK; k++) {
            uint16_t h = f2h(nd(rng));
            if (k < K) A16[(size_t)r * K + k] = h; else A2_16[(size_t)r * K2 + k - K] = h;
            Acat[(size_t)r * KK + k] = h2f(h);
        }
    for (int r = 0; r < N; r++)
        for (int k = 0; k < KK; k++) {
            uint16_t h = f2h(nd(rng) * 0.02f);
            if (k < K) W16[(size_t)r * K + k] = h; else W2_16[(size_t)r * K2 + k - K] = h;
            Wcat[(size_t)r * KK + k] = h;
        }
    std::vector<float> gate(N), C0((size_t)M * N);
    for (auto& g : gate) g = nd(rng);
    for (auto& c : C0) c = nd(rng);

    float* dAcat = up(Acat);
    uint16_t* dWcat = up(Wcat);
    float* dRef;
    CK(cudaMalloc(&dRef, (size_t)M * N * 4));
    tc::launch(dRef, dAcat, dWcat, M, N, KK, 0.f, 0, true);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> ref = down(dRef, (size_t)M * N);

    uint16_t *dA = up(A16), *dA2 = up(A2_16), *dW = up(W16), *dW2 = up(W2_16);
    float* dG = up(gate);
    float* dC;
    CK(cudaMalloc(&dC, (size_t)M * N * 4));
    tc16::Args a;
    a.A = (const __half*)dA; a.W = (const __half*)dW; a.K = K;
    if (K2) { a.A2 = (const __half*)dA2; a.W2 = (const __half*)dW2; a.K2 = K2; }
    a.C = dC; a.ldc = N; a.M = M; a.N = N;

    auto cmp_f32 = [&](const char* what, const std::vector<float>& want) {
        std::vector<float> got = down(dC, (size_t)M * N);
        size_t bad = 0;
        double maxd = 0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = std::fabs((double)got[i] - want[i]);
            maxd = std::max(maxd, d);
            bad += memcmp(&got[i], &want[i], 4) != 0;
        }
        check(what, bad, got.size(), maxd);
    };
    auto cmp_f16 = [&](const char* what, const std::vector<uint16_t>& want) {
        std::vector<uint16_t> got = down((const uint16_t*)dC, (size_t)M * N);
        size_t bad = 0;
        double maxd = 0;
        for (size_t i = 0; i < got.size(); i++) {
            maxd = std::max(maxd, (double)std::fabs(h2f(got[i]) - h2f(want[i])));
            bad += got[i] != want[i];
        }
        check(what, bad, got.size(), maxd);
    };

    // plain fp32 output: poison the buffer first, every element must be written
    CK(cudaMemset(dC, 0xFF, (size_t)M * N * 4));
    run16<tc16::EPI_F32>(a, stages);
    cmp_f32("fp32 out == tc::k_gemm", ref);

    // beta: C = acc + 0.5 C0
    CK(cudaMemcpy(dC, C0.data(), C0.size() * 4, cudaMemcpyHostToDevice));
    a.beta = 0.5f;
    run16<tc16::EPI_F32>(a, stages);
    a.beta = 0.f;
    {
        std::vector<float> want(ref.size());
        for (size_t i = 0; i < want.size(); i++) { float v = ref[i]; v += 0.5f * C0[i]; want[i] = v; }
        cmp_f32("fp32 out, beta 0.5", want);
    }

    // gated residual: C0 += gate[col] * acc (the device contracts it into one fma)
    CK(cudaMemcpy(dC, C0.data(), C0.size() * 4, cudaMemcpyHostToDevice));
    a.gate = dG;
    run16<tc16::EPI_RESID>(a, stages);
    a.gate = nullptr;
    {
        std::vector<float> want(ref.size());
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++) want[(size_t)r * N + c] = std::fmaf(gate[c], ref[(size_t)r * N + c], C0[(size_t)r * N + c]);
        cmp_f32("gated residual", want);
    }

    // fp16 and GELU fp16 outputs
    std::vector<uint16_t> w16(ref.size());
    for (size_t i = 0; i < ref.size(); i++) w16[i] = f2h(ref[i]);
    uint16_t* dG16;
    CK(cudaMalloc(&dG16, ref.size() * 2));
    k_gelu16<<<(unsigned)((ref.size() + 255) / 256), 256>>>(dG16, dRef, ref.size());
    std::vector<uint16_t> wg = down(dG16, ref.size());
    cudaFree(dG16);
    run16<tc16::EPI_F16>(a, stages);
    cmp_f16("fp16 out", w16);
    run16<tc16::EPI_GELU16>(a, stages);
    cmp_f16("GELU fp16 out", wg);

    // sanity against a double-precision reference for a few rows
    double e2 = 0, r2 = 0;
    for (int r = 0; r < M; r += std::max(1, M / 7))
        for (int c = 0; c < N; c++) {
            double s = 0;
            for (int k = 0; k < KK; k++) s += (double)Acat[(size_t)r * KK + k] * h2f(Wcat[(size_t)c * KK + k]);
            double d = ref[(size_t)r * N + c] - s;
            e2 += d * d;
            r2 += s * s;
        }
    printf("  rel_l2 vs double reference: %.2e\n", std::sqrt(e2 / r2));

    for (void* p : {(void*)dAcat, (void*)dWcat, (void*)dRef, (void*)dA, (void*)dA2, (void*)dW, (void*)dW2, (void*)dG, (void*)dC}) cudaFree(p);
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    const int sm = p.major * 10 + p.minor;
    printf("%s, sm_%d\n", p.name, sm);
    std::mt19937 rng(7);
    const int big = sm >= 80 ? 4 : 3;  // sm_75 allows 64 KB of shared memory per block: at most 3 stages
    test(1536, 2048, 2048, 0, 2, rng);
    test(1536, 2048, 2048, 0, big, rng);
    test(1000, 256, 8192, 64, big, rng);     // M tail, long K, LoRA-like second product
    test(77, 128, 64, 32, 2, rng);           // one partial row tile, two k tiles + one LoRA tile
    test(77, 128, 64, 32, big, rng);         // fewer k tiles than stages
    test(3408, 1024, 256, 96, big, rng);     // 768x1136 row count, several row groups
    if (sm >= 80) test(1536, 2048, 2048, 32, 4, rng);
    printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
    return fails ? 1 : 0;
}
