// int8 dp4a GEMM (engine/src/i8gemm.cuh) vs fp16 HFMA2 GEMM (engine/src/hgemm.cuh) on Anima's shapes.
//   i8_test peak          raw ALU rates: FFMA, HFMA2, IDP4A
//   i8_test gemm [rounds] correctness (exact int32 check on sampled outputs) + best-of-N timing per variant
//   i8_test sustain [sec] every block linear of one 1536-token denoising step (x28 blocks) back to back for [sec]
//                          seconds per path (fp16 / int8 / int8 with mlp2 in fp16), with SM clock + power from NVML
// Timing: every launch is timed on its own with an event pair; variants are interleaved round-robin so a
// co-running process (the Kiln server) slows all of them alike; min and median over rounds are reported.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cuda_fp16.h>
#include "../engine/src/hgemm.cuh"
#include "../engine/src/i8gemm.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// ------------------------------------------------------------------ peak
__global__ void p_ffma(float* out, int iters) {
    float a = threadIdx.x * 1e-7f, b = 0.999f, c[8] = {0};
    for (int i = 0; i < iters; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) c[j] = fmaf(a, b, c[j]);
    float s = 0; for (int j = 0; j < 8; j++) s += c[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void p_hfma2(float* out, int iters) {
    __half2 a = __float2half2_rn(threadIdx.x * 1e-4f), b = __float2half2_rn(0.999f), c[8];
    for (int j = 0; j < 8; j++) c[j] = __float2half2_rn(0.f);
    for (int i = 0; i < iters; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) c[j] = __hfma2(a, b, c[j]);
    __half2 s = c[0]; for (int j = 1; j < 8; j++) s = __hadd2(s, c[j]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = __low2float(s) + __high2float(s);
}
__global__ void p_dp4a(float* out, int iters) {
    int a = threadIdx.x * 0x01010101, b = 0x03fd02fe, c[8] = {0};
    for (int i = 0; i < iters; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) c[j] = __dp4a(a, b, c[j]);
    int s = 0; for (int j = 0; j < 8; j++) s += c[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}
// dp4a and HFMA2 interleaved in one stream: do the INT and FP pipes overlap?
__global__ void p_mix(float* out, int iters) {
    int a = threadIdx.x * 0x01010101, b = 0x03fd02fe, c[4] = {0};
    __half2 ha = __float2half2_rn(threadIdx.x * 1e-4f), hb = __float2half2_rn(0.999f), h[4];
    for (int j = 0; j < 4; j++) h[j] = __float2half2_rn(0.f);
    for (int i = 0; i < iters; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) { c[j] = __dp4a(a, b, c[j]); h[j] = __hfma2(ha, hb, h[j]); }
    int s = 0; for (int j = 0; j < 4; j++) s += c[j];
    __half2 hs = __hadd2(__hadd2(h[0], h[1]), __hadd2(h[2], h[3]));
    out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s + __low2float(hs);
}

static int peak() {
    float* out;
    CK(cudaMalloc(&out, 24 * 16 * 256 * 4 * 4));
    const int blocks = 24 * 16, threads = 256, iters = 1 << 16;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    auto run = [&](auto kern) {
        float best = 1e30f;
        for (int k = 0; k < 15; k++) {
            cudaEventRecord(a); kern<<<blocks, threads>>>(out, iters); cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms);
        }
        return best;
    };
    double insts = 8.0 * iters * blocks * threads;  // per kernel: 8 ops per iteration per thread
    for (int r = 0; r < 3; r++) {
        float tf = run(p_ffma), th = run(p_hfma2), td = run(p_dp4a), tm = run(p_mix);
        printf("FFMA  %6.2f Ginst/s = %5.2f TFLOPS\n", insts / tf / 1e6, 2 * insts / tf / 1e9);
        printf("HFMA2 %6.2f Ginst/s = %5.2f TFLOPS\n", insts / th / 1e6, 4 * insts / th / 1e9);
        printf("IDP4A %6.2f Ginst/s = %5.2f TOPS\n", insts / td / 1e6, 8 * insts / td / 1e9);
        printf("mix(dp4a+hfma2 1:1) %6.2f Ginst/s total = %5.2f T(OPS+FLOPS)\n\n", insts / tm / 1e6,
               (4 * 4.0 + 8 * 4.0) * iters * blocks * threads / tm / 1e9);
    }
    return 0;
}

// ------------------------------------------------------------------ gemm
static uint16_t to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

struct Variant {
    const char* name;
    void (*launch)(float*, const int8_t*, const float*, const int8_t*, const float*, int, int, int, float, cudaStream_t);
    int bn, bk;
};
template <int BM, int BN, int BK, int TM, int TN, int MINB>
static void launch_v(float* C, const int8_t* A, const float* sa, const int8_t* W, const float* sw, int M, int N, int K, float beta, cudaStream_t s) {
    dim3 grid(N / BN, (M + BM - 1) / BM);
    i8::gemm_nt<BM, BN, BK, TM, TN, MINB><<<grid, (BM / TM) * (BN / TN), 0, s>>>(C, A, sa, W, sw, M, N, K, beta);
}
static Variant variants[] = {
    {"128x128x32 8x8  2/SM", launch_v<128, 128, 32, 8, 8, 2>, 128, 32},
    {"128x128x32 8x8  1/SM", launch_v<128, 128, 32, 8, 8, 1>, 128, 32},
    {"128x128x64 8x8  2/SM", launch_v<128, 128, 64, 8, 8, 2>, 128, 64},
    {"128x256x32 8x16 1/SM", launch_v<128, 256, 32, 8, 16, 1>, 256, 32},
    {"256x128x32 16x8 1/SM", launch_v<256, 128, 32, 16, 8, 1>, 128, 32},
    {"64x128x64  4x8  2/SM", launch_v<64, 128, 64, 4, 8, 2>, 128, 64},
};
static const int NV = sizeof(variants) / sizeof(variants[0]);

static int gemm_bench(int rounds, const std::vector<int>& only) {
    struct S { int m, n, k; const char* name; } shapes[] = {
        {1536, 2048, 2048, "q/k/v/o  1536x2048x2048"}, {1536, 8192, 2048, "mlp1     1536x8192x2048"},
        {1536, 2048, 8192, "mlp2     1536x2048x8192"}, {3952, 8192, 2048, "mlp1 big 3952x8192x2048"},
    };
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::mt19937 rng(7);
    for (auto& s : shapes) {
        const size_t MK = (size_t)s.m * s.k, NK = (size_t)s.n * s.k, MN = (size_t)s.m * s.n;
        std::vector<int8_t> Aq(MK), Wq(NK);
        std::vector<float> sa(s.m), sw(s.n);
        std::uniform_int_distribution<int> di(-127, 127);
        for (auto& v : Aq) v = (int8_t)di(rng);
        for (auto& v : Wq) v = (int8_t)di(rng);
        std::uniform_real_distribution<float> du(0.5f, 2.f);
        for (auto& v : sa) v = du(rng) * 1e-3f;
        for (auto& v : sw) v = du(rng) * 1e-3f;
        std::vector<float> Af(MK);
        std::vector<uint16_t> Wb(NK);
        for (size_t i = 0; i < MK; i++) Af[i] = Aq[i] * 0.01f;
        for (size_t i = 0; i < NK; i++) Wb[i] = to_bf16(Wq[i] * 1e-3f);

        int8_t *dA, *dW; float *dsa, *dsw, *dC, *dAf, *dsc; uint16_t* dWb; int8_t* dAq2;
        CK(cudaMalloc(&dA, MK)); CK(cudaMalloc(&dW, NK)); CK(cudaMalloc(&dsa, s.m * 4)); CK(cudaMalloc(&dsw, s.n * 4));
        CK(cudaMalloc(&dC, MN * 4)); CK(cudaMalloc(&dAf, MK * 4)); CK(cudaMalloc(&dWb, NK * 2));
        CK(cudaMalloc(&dAq2, MK)); CK(cudaMalloc(&dsc, s.m * 4));
        CK(cudaMemcpy(dA, Aq.data(), MK, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, Wq.data(), NK, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsa, sa.data(), s.m * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsw, sw.data(), s.n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dAf, Af.data(), MK * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWb, Wb.data(), NK * 2, cudaMemcpyHostToDevice));

        // exactness: every variant against a CPU int32 reference on sampled outputs (incl. the M tail)
        std::vector<float> Ch(MN);
        std::vector<std::pair<int, int>> probe;
        for (int i = 0; i < 4000; i++) probe.push_back({(int)(rng() % s.m), (int)(rng() % s.n)});
        probe.push_back({s.m - 1, s.n - 1});
        for (int v = 0; v < NV; v++) {
            CK(cudaMemset(dC, 0, MN * 4));
            variants[v].launch(dC, dA, dsa, dW, dsw, s.m, s.n, s.k, 0.f, 0);
            CK(cudaGetLastError());
            CK(cudaMemcpy(Ch.data(), dC, MN * 4, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (auto [m, n] : probe) {
                long long acc = 0;
                for (int k = 0; k < s.k; k++) acc += (int)Aq[(size_t)m * s.k + k] * (int)Wq[(size_t)n * s.k + k];
                float ref = (float)acc * sa[m] * sw[n];
                if (std::fabs(ref - Ch[(size_t)m * s.n + n]) > 1e-6f * std::fabs(ref) + 1e-12f) bad++;
            }
            if (bad) printf("  !! %s: %d / %zu sampled outputs wrong\n", variants[v].name, bad, probe.size());
        }

        // timing
        const double ops = 2.0 * s.m * s.n * s.k;
        std::vector<std::vector<float>> t(NV + 3);
        auto time1 = [&](auto fn) {
            cudaEventRecord(e0); fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); return ms;
        };
        auto hg1 = [&] { hg::launch(dC, dAf, dWb, s.m, s.n, s.k, 0.f, 0, 1); };
        auto hg2 = [&] { hg::launch(dC, dAf, dWb, s.m, s.n, s.k, 0.f, 0, 2); };
        auto quant = [&] { i8::quantize_rows(dAq2, dsc, dAf, nullptr, s.m, s.k, 0); };
        for (int i = 0; i < 3; i++) { hg1(); for (int v = 0; v < NV; v++) variants[v].launch(dC, dA, dsa, dW, dsw, s.m, s.n, s.k, 0.f, 0); }
        CK(cudaDeviceSynchronize());
        for (int r = 0; r < rounds; r++) {
            t[NV].push_back(time1(hg1));
            t[NV + 1].push_back(time1(hg2));
            t[NV + 2].push_back(time1(quant));
            for (int v = 0; v < NV; v++) {
                if (!only.empty() && std::find(only.begin(), only.end(), v) == only.end()) continue;
                t[v].push_back(time1([&] { variants[v].launch(dC, dA, dsa, dW, dsw, s.m, s.n, s.k, 0.f, 0); }));
            }
        }
        CK(cudaGetLastError());
        auto stat = [&](std::vector<float>& x, float& mn, float& med) {
            std::sort(x.begin(), x.end()); mn = x[0]; med = x[x.size() / 2];
        };
        float hmin, hmed, h2min, h2med, qmin, qmed;
        stat(t[NV], hmin, hmed); stat(t[NV + 1], h2min, h2med); stat(t[NV + 2], qmin, qmed);
        printf("%s\n  hgemm v1            %7.3f ms %5.2f TF   (median %7.3f ms %5.2f TF)\n", s.name, hmin, ops / hmin / 1e9, hmed, ops / hmed / 1e9);
        printf("  hgemm v2            %7.3f ms %5.2f TF   (median %7.3f ms %5.2f TF)\n", h2min, ops / h2min / 1e9, h2med, ops / h2med / 1e9);
        printf("  act quant (fp32->i8)%7.3f ms            (median %7.3f ms)\n", qmin, qmed);
        for (int v = 0; v < NV; v++) {
            if (t[v].empty()) continue;
            float mn, med; stat(t[v], mn, med);
            printf("  i8 %s %7.3f ms %5.2f TOPS (median %7.3f ms %5.2f TOPS)  x%.2f vs hgemm v1 (min), x%.2f incl. quant\n", variants[v].name, mn,
                   ops / mn / 1e9, med, ops / med / 1e9, hmin / mn, hmin / (mn + qmin));
        }
        fflush(stdout);
        cudaFree(dA); cudaFree(dW); cudaFree(dsa); cudaFree(dsw); cudaFree(dC); cudaFree(dAf); cudaFree(dWb); cudaFree(dAq2); cudaFree(dsc);
    }
    return 0;
}


// ------------------------------------------------------------------ sustained step-shaped load
struct Nvml {
    struct Util { unsigned gpu, mem; };
    int (*init)() = nullptr;
    int (*handle)(unsigned, void**) = nullptr;
    int (*clock)(void*, int, unsigned*) = nullptr;
    int (*power)(void*, unsigned*) = nullptr;
    int (*util)(void*, Util*) = nullptr;
    void* dev = nullptr;
    bool ok = false;
    Nvml() {
        HMODULE h = LoadLibraryA("nvml.dll");
        if (!h) return;
        init = (int (*)())GetProcAddress(h, "nvmlInit_v2");
        handle = (int (*)(unsigned, void**))GetProcAddress(h, "nvmlDeviceGetHandleByIndex_v2");
        clock = (int (*)(void*, int, unsigned*))GetProcAddress(h, "nvmlDeviceGetClockInfo");
        power = (int (*)(void*, unsigned*))GetProcAddress(h, "nvmlDeviceGetPowerUsage");
        util = (int (*)(void*, Util*))GetProcAddress(h, "nvmlDeviceGetUtilizationRates");
        ok = init && handle && clock && power && util && init() == 0 && handle(0, &dev) == 0;
    }
    unsigned sm_mhz() { unsigned v = 0; if (ok) clock(dev, 1, &v); return v; }
    double watts() { unsigned mw = 0; if (ok) power(dev, &mw); return mw / 1000.0; }
    unsigned busy() { Util u{}; if (ok) util(dev, &u); return u.gpu; }
};

static int sustain(double secs) {
    Nvml nv;
    if (!nv.ok) printf("(NVML not available: no clock/power readings)\n");
    // Wait (max ~90 s) for the GPU to be idle BEFORE allocating anything, so a co-running Kiln server is neither
    // time-sliced into the readings nor squeezed for VRAM while we wait. Total device footprint ~160 MB.
    for (int w = 0; w < 90 && nv.ok; w++) {
        bool idle = true;
        for (int k = 0; k < 5; k++) { if (nv.busy() > 5) { idle = false; break; } Sleep(100); }
        if (idle) break;
        if (w == 89) printf("(GPU never went idle: readings are contended)\n");
        Sleep(500);
    }
    const int M = 1536, D = 2048, F = 8192;
    float *A, *C; int8_t *Aq, *Wq; uint16_t* Wb; float *sa, *sw;
    CK(cudaMalloc(&A, (size_t)M * F * 4)); CK(cudaMalloc(&C, (size_t)M * F * 4)); CK(cudaMalloc(&Aq, (size_t)M * F));
    CK(cudaMalloc(&Wq, (size_t)D * F)); CK(cudaMalloc(&Wb, (size_t)D * F * 2)); CK(cudaMalloc(&sa, F * 4)); CK(cudaMalloc(&sw, F * 4));
    int8_t *Wq_dd = Wq, *Wq_big = Wq; uint16_t *Wb_dd = Wb, *Wb_big = Wb;  // one buffer per type; the 2048x2048 layers use its head
    {   // realistic magnitudes (int timing is data-independent; fp16 timing too)
        std::vector<float> h((size_t)M * F); std::mt19937 r(5); std::normal_distribution<float> nd;
        for (auto& v : h) v = nd(r);
        CK(cudaMemcpy(A, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        std::vector<uint16_t> w((size_t)D * F); for (auto& v : w) v = to_bf16(nd(r) * 0.02f);
        CK(cudaMemcpy(Wb, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        i8::quantize_weight(Wq, sw, Wb, nullptr, F, D, 0);
        CK(cudaDeviceSynchronize());
    }
    // one block's linears at 1536 tokens: q,k,v,o, cross q, cross o (2048x2048), mlp1 (8192x2048), mlp2 (2048x8192)
    auto h16 = [&](int n, int k) { hg::launch(C, A, n == F || k == F ? Wb_big : Wb_dd, M, n, k, 0.f, 0, 1); };
    auto q8 = [&](int k) { i8::quantize_rows(Aq, sa, A, nullptr, M, k, 0); };
    auto g8 = [&](int n, int k) { i8::gemm(C, Aq, sa, n == F || k == F ? Wq_big : Wq_dd, sw, M, n, k, 0.f, 0); };
    auto step_fp16 = [&] { for (int b = 0; b < 28; b++) { for (int i = 0; i < 6; i++) h16(D, D); h16(F, D); h16(D, F); } };
    auto step_i8 = [&](bool mlp2_fp16) {
        for (int b = 0; b < 28; b++) {
            q8(D); g8(D, D); g8(D, D); g8(D, D);  // qkv share one activation quantization
            q8(D); g8(D, D);                      // o
            q8(D); g8(D, D);                      // cross q
            q8(D); g8(D, D);                      // cross o
            q8(D); g8(F, D);                      // mlp1
            if (mlp2_fp16) h16(D, F); else { q8(F); g8(D, F); }
        }
    };
    const double ops = 28.0 * 2 * M * ((6.0 * D * D) + 2.0 * D * F);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    struct Path { const char* name; std::function<void()> fn; };
    std::vector<Path> paths = {{"fp16 hgemm v1 (all)", step_fp16}, {"int8 dp4a (all, incl. act quant)", [&] { step_i8(false); }},
                               {"int8 + mlp2 in fp16", [&] { step_i8(true); }}};
    for (auto& p : paths) {
        {
            p.fn(); CK(cudaDeviceSynchronize());  // warm
            std::vector<float> ms;
            std::vector<unsigned> clk;
            std::vector<double> pw;
            auto t0 = GetTickCount64();
            while (GetTickCount64() - t0 < secs * 1000) {
                cudaEventRecord(e0); p.fn(); cudaEventRecord(e1);
                while (cudaEventQuery(e1) == cudaErrorNotReady) { Sleep(25); clk.push_back(nv.sm_mhz()); pw.push_back(nv.watts()); }
                float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
            }
            std::vector<float> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            // skip the first 20% of samples (clock settling)
            double c = 0, w = 0; size_t n0 = clk.size() / 5, n = 0;
            for (size_t i = n0; i < clk.size(); i++, n++) { c += clk[i]; w += pw[i]; }
            double mhz = n ? c / n : 0;
            printf("%-34s step linears: best %7.1f ms  median %7.1f ms (%zu runs) = %5.2f T/s | SM %4.0f MHz, %4.1f W | %5.1f%% of %s peak at that clock\n",
                   p.name, sorted[0], sorted[sorted.size() / 2], ms.size(), ops / sorted[sorted.size() / 2] / 1e9, mhz, n ? w / n : 0,
                   mhz > 0 ? 100.0 * ops / (sorted[sorted.size() / 2] * 1e-3) / (24 * 64 * mhz * 1e6 * (p.name[0] == 'f' ? 4 : 8)) : 0.0,
                   p.name[0] == 'f' ? "HFMA2" : "IDP4A");
            fflush(stdout);
        }
    }
    cudaFree(A); cudaFree(C); cudaFree(Aq); cudaFree(Wq); cudaFree(Wb); cudaFree(sa); cudaFree(sw);
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "gemm";
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("%s  SMs=%d  sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    if (mode == "peak") return peak();
    if (mode == "sustain") return sustain(argc > 2 ? atof(argv[2]) : 5.0);
    int rounds = argc > 2 ? atoi(argv[2]) : 20;
    std::vector<int> only;
    for (int i = 3; i < argc; i++) only.push_back(atoi(argv[i]));
    return gemm_bench(rounds, only);
}
