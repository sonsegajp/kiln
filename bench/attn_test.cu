// Replays one attention() call from dumped q/k/v (logs/diag/attn_*.f32) and checks for non-finite output.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../engine/src/gpu.h"

int main(int argc, char** argv) {
    const int T = 1024, C = 640, H = 10;
    const char* dir = argc > 1 ? argv[1] : "C:/Users/hyper/Kiln/logs/diag";
    gpu_init(0);
    gpu_alloc_scratch((size_t)8192 * 2048, (size_t)600 << 20);
    auto load = [&](const char* n) {
        std::vector<float> h((size_t)T * C);
        FILE* f = fopen((std::string(dir) + "/attn_" + n + ".f32").c_str(), "rb");
        fread(h.data(), 4, h.size(), f);
        fclose(f);
        float* d = G.arena.f(h.size());
        CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        return d;
    };
    float *q = load("q"), *k = load("k"), *v = load("v");
    float* o = G.arena.f((size_t)T * C);
    for (int rep = 0; rep < 3; rep++) {
        CK(cudaMemset(o, 0, (size_t)T * C * 4));
        attention(o, C, q, C, k, C, v, C, T, T, H, 64, false);
        std::vector<float> h((size_t)T * C);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), o, h.size() * 4, cudaMemcpyDeviceToHost));
        int per[10] = {};
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) if (!std::isfinite(h[(size_t)t * C + c])) per[c / 64]++;
        printf("rep %d bad per head:", rep);
        for (int i = 0; i < H; i++) printf(" %d", per[i]);
        printf("\n");
    }
    return 0;
}
