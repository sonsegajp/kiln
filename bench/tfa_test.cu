// Accuracy + speed of fa::flash_fwd (HFMA2) and tfa::k_flash (tensor cores) vs a plain fp32 reference.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../engine/src/flash.cuh"
#include "../engine/src/tcflash.cuh"

// fp32 reference: one thread per (query, head), two passes over the keys
__global__ void ref_attn(float* out, const float* q, const float* k, const float* v, int T, int Tk, int H, int ld, int npad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    int t = i / H, h = i % H;
    const float* qr = q + (size_t)t * ld + h * 128;
    float m = npad ? 0.f : -INFINITY;
    for (int j = 0; j < Tk; j++) {
        const float* kr = k + (size_t)j * ld + h * 128;
        float s = 0;
        for (int d = 0; d < 128; d++) s += qr[d] * kr[d];
        m = fmaxf(m, s / sqrtf(128.f));
    }
    float acc[128] = {0}, l = npad * expf(-m);
    for (int j = 0; j < Tk; j++) {
        const float* kr = k + (size_t)j * ld + h * 128;
        float s = 0;
        for (int d = 0; d < 128; d++) s += qr[d] * kr[d];
        float p = expf(s / sqrtf(128.f) - m);
        l += p;
        for (int d = 0; d < 128; d++) acc[d] += p * v[(size_t)j * ld + h * 128 + d];
    }
    for (int d = 0; d < 128; d++) out[(size_t)t * ld + h * 128 + d] = acc[d] / l;
}

static void run(int T, int Tk, int npad, const char* name, bool tc) {
    const int H = 16, ld = H * 128;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> Q((size_t)T * ld), K((size_t)Tk * ld), V((size_t)Tk * ld);
    // q/k are RMS-normalized in the model (per-head norm weights ~1-3): use N(0, 1.5)
    for (auto& x : Q) x = nd(rng) * 1.5f;
    for (auto& x : K) x = nd(rng) * 1.5f;
    for (auto& x : V) x = nd(rng);
    float *dq, *dk, *dv, *o1, *o2;
    cudaMalloc(&dq, Q.size() * 4); cudaMalloc(&dk, K.size() * 4); cudaMalloc(&dv, V.size() * 4);
    cudaMalloc(&o1, Q.size() * 4); cudaMalloc(&o2, Q.size() * 4);
    cudaMemcpy(dq, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, V.data(), V.size() * 4, cudaMemcpyHostToDevice);
    ref_attn<<<(T * H + 127) / 128, 128>>>(o1, dq, dk, dv, T, Tk, H, ld, npad);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto go = [&] { if (tc) tfa::launch(o2, ld, dq, ld, dk, ld, dv, ld, T, Tk, H, npad, 0); else fa::launch(o2, ld, dq, ld, dk, ld, dv, ld, T, Tk, H, npad, 0); };
    for (int i = 0; i < 3; i++) go();
    cudaEventRecord(e0);
    const int reps = 3;
    for (int i = 0; i < reps; i++) go();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
    std::vector<float> A(Q.size()), B(Q.size());
    cudaMemcpy(A.data(), o1, A.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(B.data(), o2, B.size() * 4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t i = 0; i < A.size(); i++) { double d = B[i] - A[i]; num += d * d; den += (double)A[i] * A[i]; }
    double flop = 4.0 * T * Tk * 128 * H;
    printf("%s %-28s %7.2f ms  %5.2f TF  rel_l2 %.2e  %s\n", tc ? "tc   " : "hfma2", name, ms, flop / ms / 1e9, sqrt(num / den), cudaGetErrorString(cudaGetLastError()));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(o1); cudaFree(o2);
}

int main() {
    for (int tc = 0; tc < 2; tc++) {
        run(1536, 1536, 0, "self  512x768 (1536 tok)", tc);
        run(3952, 3952, 0, "self  832x1216 (3952 tok)", tc);
        run(1536, 26, 486, "cross 26 real + 486 pad", tc);
        run(1000, 1000, 0, "self  ragged 1000 tok", tc);
        run(1000, 77, 0, "cross ragged 77 keys", tc);
    }
    return 0;
}
