// Accuracy of the W8A8 dp4a path (engine/src/i8gemm.cuh) on REAL Anima activations.
//
// Runs the DiT (engine kernels from kernels.cu; weights streamed per block stage, so only ~100 MB of weights are
// resident next to the running server) on the golden 512x768 job (ref/golden: prompt context, noise, sigmas, turbo
// LoRA), for several "streams" that differ only in how the 8 block linears are computed:
//   r = fp32 reference (bf16 weights, fp32 FMA)         h = engine fp16 HFMA2 GEMM (hgemm v1)
//   p = int8 W8A8 (per-token act scale, per-out-channel weight scale)
//   s = p + SmoothQuant (per-input-channel factors, alpha, from a calibration run on a different noise seed)
// A stream is 6 letters for the sites [qkv, o, cross_q, cross_o, mlp1, mlp2], e.g. "pppppp" or "sssssh".
// Stream 0 is the reference. At every linear it also runs h/p/s on the *reference* input and records the
// per-layer rel_l2 vs the fp32 output (no error propagation), plus activation outlier statistics.
// Every stream integrates its own 8-step Euler trajectory from the golden noise, so the final latent measures the
// accumulated, image-level error. Attention is the engine's fp16 flash kernel in every stream.
// Needs ~400 MB of VRAM for ~8 minutes (refuses to start with < 1.2 GB free): run it with the Kiln server stopped.
//
// usage: i8_real [--streams hhhhhh,pppppp,...] [--alpha 0.5] [--nolora] [--steps 8] [--calib-steps 8] [--out dir]
//        i8_real --decode [--out dir]   decodes every <dir>/latent_final_*.npy with the engine VAE -> <dir>/image_*.npy
//                                        (HWC, [0,1]); bench/i8_compare.py turns them into PNGs + PSNR
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../engine/src/gpu.h"
#include "../engine/src/models.h"
#include "../engine/src/safetensors.h"
#include "../engine/src/i8gemm.cuh"

static const int D = 2048, NH = 16, DH = 128, F = 8192, NB = 28;
enum { S_QKV, S_O, S_CAQ, S_CAO, S_MLP1, S_MLP2, NSITE };
static const char* SITE_NAME[NSITE] = {"qkv", "o", "cross_q", "cross_o", "mlp1", "mlp2"};
enum { L_Q, L_K, L_V, L_O, L_CAQ, L_CAO, L_MLP1, L_MLP2, NLAYER };
static const char* LAYER_NAME[NLAYER] = {"self q", "self k", "self v", "self o", "cross q", "cross o", "mlp1", "mlp2"};
enum { X_H, X_P, X_S, NSCHEME };
static const char* SCHEME_NAME[NSCHEME] = {"fp16", "i8", "i8+smooth"};

// ------------------------------------------------------------------------------------------------ npy
static bool load_npy(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[6];
    unsigned char ver[2];
    uint32_t hl = 0;
    if (fread(magic, 1, 6, f) != 6 || fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
    if (ver[0] == 1) { uint16_t h; fread(&h, 2, 1, f); hl = h; } else fread(&hl, 4, 1, f);
    std::string h(hl, '\0');
    fread(&h[0], 1, hl, f);
    bool i4 = h.find("<i4") != std::string::npos;
    shape.clear();
    size_t p = h.find("'shape': (") + 10;
    while (p < h.size() && h[p] != ')') {
        if (isdigit((unsigned char)h[p])) { shape.push_back(strtoll(&h[p], nullptr, 10)); while (isdigit((unsigned char)h[p])) p++; }
        else p++;
    }
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    out.resize(n);
    if (i4) { std::vector<int32_t> t(n); fread(t.data(), 4, n, f); for (size_t i = 0; i < n; i++) out[i] = (float)t[i]; }
    else fread(out.data(), 4, n, f);
    fclose(f);
    return true;
}
static void save_npy(const std::string& path, const float* d, const std::vector<int64_t>& shape) {
    std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); i++) { h += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? ", " : ""); n *= shape[i]; }
    h += "), }";
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    uint16_t hl = (uint16_t)h.size();
    fwrite(&hl, 2, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    fwrite(d, 4, n, f);
    fclose(f);
}

// ------------------------------------------------------------------------------------------------ kernels
__device__ __forceinline__ float bf(uint32_t b) { return __uint_as_float(b << 16); }

// fp32 reference: C[M,N] = A[M,K] * W[N,K]^T, W bf16, plain fp32 FMA. N % 128 == 0, K % 8 == 0.
__global__ void __launch_bounds__(256) k_sgemm128(float* __restrict__ C, const float* __restrict__ A, const uint16_t* __restrict__ W,
                                                  int M, int N, int K) {
    __shared__ __align__(16) float As[2][8][132];
    __shared__ __align__(16) float Bs[2][8][132];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, row = tid / 2, kq = tid % 2;
    const int m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
    float4 ra;
    uint2 rb;
    auto load = [&](int k0) {
        int m = m0 + row;
        ra = m < M ? *(const float4*)(A + (size_t)m * K + k0 + kq * 4) : make_float4(0, 0, 0, 0);
        rb = *(const uint2*)(W + (size_t)(n0 + row) * K + k0 + kq * 4);
    };
    auto store = [&](int b) {
        As[b][kq * 4 + 0][row] = ra.x; As[b][kq * 4 + 1][row] = ra.y; As[b][kq * 4 + 2][row] = ra.z; As[b][kq * 4 + 3][row] = ra.w;
        Bs[b][kq * 4 + 0][row] = bf(rb.x & 0xffff); Bs[b][kq * 4 + 1][row] = bf(rb.x >> 16);
        Bs[b][kq * 4 + 2][row] = bf(rb.y & 0xffff); Bs[b][kq * 4 + 3][row] = bf(rb.y >> 16);
    };
    float acc[8][8] = {};
    const int tiles = K / 8;
    load(0); store(0); __syncthreads();
    for (int t = 0; t < tiles; t++) {
        const int b = t & 1;
        if (t + 1 < tiles) load((t + 1) * 8);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float a[8], w[8];
            *(float4*)&a[0] = *(const float4*)&As[b][k][ty * 4];
            *(float4*)&a[4] = *(const float4*)&As[b][k][64 + ty * 4];
            *(float4*)&w[0] = *(const float4*)&Bs[b][k][tx * 4];
            *(float4*)&w[4] = *(const float4*)&Bs[b][k][64 + tx * 4];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], w[j], acc[i][j]);
        }
        if (t + 1 < tiles) { store(b ^ 1); __syncthreads(); }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int m = m0 + (i / 4) * 64 + ty * 4 + (i % 4);
        if (m >= M) continue;
#pragma unroll
        for (int g = 0; g < 2; g++)
            *(float4*)(C + (size_t)m * N + n0 + g * 64 + tx * 4) = make_float4(acc[i][4 * g], acc[i][4 * g + 1], acc[i][4 * g + 2], acc[i][4 * g + 3]);
    }
}

// generic fp32: C[M,N] = alpha * A[M,K](row stride lda) * W[N,K]^T + beta*C, any sizes (LoRA, x_embedder, final)
__global__ void k_sgemm_gen(float* C, const float* A, int lda, const uint16_t* W, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[32][33], Ws[32][33];
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5, m0 = blockIdx.y * 32, n0 = blockIdx.x * 32;
    float acc[4] = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int r = ty; r < 32; r += 8) {
            int m = m0 + r, n = n0 + r, k = k0 + tx;
            As[r][tx] = (m < M && k < K) ? A[(size_t)m * lda + k] : 0.f;
            Ws[r][tx] = (n < N && k < K) ? bf(W[(size_t)n * K + k]) : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; kk++) {
            float w = Ws[tx][kk];
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] = fmaf(As[ty + 8 * i][kk], w, acc[i]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        int m = m0 + ty + 8 * i, n = n0 + tx;
        if (m < M && n < N) { float* c = C + (size_t)m * N + n; *c = alpha * acc[i] + (beta != 0.f ? beta * *c : 0.f); }
    }
}

__global__ void k_err(const float* __restrict__ a, const float* __restrict__ r, size_t n, double* acc) {
    float d2 = 0.f, r2 = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float d = a[i] - r[i];
        d2 += d * d;
        r2 += r[i] * r[i];
    }
    for (int o = 16; o > 0; o >>= 1) { d2 += __shfl_xor_sync(0xffffffff, d2, o); r2 += __shfl_xor_sync(0xffffffff, r2, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(acc, (double)d2); atomicAdd(acc + 1, (double)r2); }
}

// per-column absmax, accumulated with atomicMax on the float bits (valid for non-negative floats)
__global__ void k_colmax(const float* x, int M, int K, unsigned* cmax) {
    int j = blockIdx.x * 256 + threadIdx.x;
    if (j >= K) return;
    int r0 = blockIdx.y * 64, r1 = min(M, r0 + 64);
    float m = 0.f;
    for (int r = r0; r < r1; r++) m = fmaxf(m, fabsf(x[(size_t)r * K + j]));
    atomicMax(cmax + j, __float_as_uint(m));
}
__global__ void k_colmax_bf16(const uint16_t* w, int N, int K, unsigned* cmax) {
    int j = blockIdx.x * 256 + threadIdx.x;
    if (j >= K) return;
    int r0 = blockIdx.y * 64, r1 = min(N, r0 + 64);
    float m = 0.f;
    for (int r = r0; r < r1; r++) m = fmaxf(m, fabsf(bf(w[(size_t)r * K + j])));
    atomicMax(cmax + j, __float_as_uint(m));
}
// per-row absmax and rms
__global__ void k_rowstats(const float* x, int K, float* rmax, float* rrms) {
    const float* xr = x + (size_t)blockIdx.x * K;
    float m = 0.f, s = 0.f;
    for (int i = threadIdx.x; i < K; i += 256) { float v = xr[i]; m = fmaxf(m, fabsf(v)); s += v * v; }
    __shared__ float sm[8], ss[8];
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o)); s += __shfl_xor_sync(0xffffffff, s, o); }
    if ((threadIdx.x & 31) == 0) { sm[threadIdx.x >> 5] = m; ss[threadIdx.x >> 5] = s; }
    __syncthreads();
    if (threadIdx.x == 0) {
        m = sm[0]; s = ss[0];
        for (int w = 1; w < 8; w++) { m = fmaxf(m, sm[w]); s += ss[w]; }
        rmax[blockIdx.x] = m;
        rrms[blockIdx.x] = sqrtf(s / K);
    }
}
// activation quantization error: xhat = q * srow[m] * (colmul ? colmul[k] : 1)
__global__ void k_qerr(const float* x, const int8_t* q, const float* srow, const float* colmul, int M, int K, double* acc) {
    float d2 = 0.f, r2 = 0.f;
    size_t n = (size_t)M * K;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int m = (int)(i / K), k = (int)(i % K);
        float xh = (float)q[i] * srow[m] * (colmul ? colmul[k] : 1.f);
        float d = xh - x[i];
        d2 += d * d;
        r2 += x[i] * x[i];
    }
    for (int o = 16; o > 0; o >>= 1) { d2 += __shfl_xor_sync(0xffffffff, d2, o); r2 += __shfl_xor_sync(0xffffffff, r2, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(acc, (double)d2); atomicAdd(acc + 1, (double)r2); }
}
// SmoothQuant factors s_k = xmax_k^a / wmax_k^(1-a)
__global__ void k_smooth(const float* xmax, const unsigned* wmax, int K, float alpha, float* mul, float* inv) {
    int k = blockIdx.x * 256 + threadIdx.x;
    if (k >= K) return;
    float xm = fmaxf(xmax[k], 1e-5f), wm = fmaxf(__uint_as_float(wmax[k]), 1e-5f);
    float s = powf(xm, alpha) / powf(wm, 1.f - alpha);
    s = fminf(fmaxf(s, 1e-4f), 1e4f);
    mul[k] = s;
    inv[k] = 1.f / s;
}
__global__ void k_patchify(float* tok, const float* lat, int Hl, int Wl) {
    int Wp = Wl / 2, T = (Hl / 2) * Wp;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 68) return;
    int f = i % 68, t = (int)(i / 68);
    int c = f / 4, m = (f / 2) % 2, nn = f % 2;
    int y = (t / Wp) * 2 + m, x = (t % Wp) * 2 + nn;
    tok[i] = c < 16 ? lat[((size_t)c * Hl + y) * Wl + x] : 0.f;
}
__global__ void k_unpatchify(float* lat, const float* tok, int Hl, int Wl) {
    int Wp = Wl / 2, T = (Hl / 2) * Wp;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 64) return;
    int f = i % 64, t = (int)(i / 64);
    int p1 = f / 32, p2 = (f / 16) % 2, c = f % 16;
    int y = (t / Wp) * 2 + p1, x = (t % Wp) * 2 + p2;
    lat[((size_t)c * Hl + y) * Wl + x] = tok[i];
}
__global__ void k_axpy2(float* x, const float* v, float a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a * v[i];
}

static inline unsigned nb(size_t n) { return (unsigned)((n + 255) / 256); }

// ------------------------------------------------------------------------------------------------ host helpers
template <class T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T) + 16)); return p; }
static double* g_acc = nullptr;
static double rel_err(const float* a, const float* r, size_t n) {
    CK(cudaMemsetAsync(g_acc, 0, 16, G.stream));
    k_err<<<480, 256, 0, G.stream>>>(a, r, n, g_acc);
    double h[2];
    CK(cudaMemcpy(h, g_acc, 16, cudaMemcpyDeviceToHost));
    return h[1] > 0 ? sqrt(h[0] / h[1]) : 0.0;
}
static double qerr(const float* x, const int8_t* q, const float* srow, const float* colmul, int M, int K) {
    CK(cudaMemsetAsync(g_acc, 0, 16, G.stream));
    k_qerr<<<480, 256, 0, G.stream>>>(x, q, srow, colmul, M, K, g_acc);
    double h[2];
    CK(cudaMemcpy(h, g_acc, 16, cudaMemcpyDeviceToHost));
    return h[1] > 0 ? sqrt(h[0] / h[1]) : 0.0;
}
static std::vector<float> dl(const float* d, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}
static double rel_host(const std::vector<float>& a, const std::vector<float>& r) {
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); i++) { double d = (double)a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; }
    return sqrt(d2 / r2);
}

static void ref_gemm(float* y, const float* x, int T, const Weight& w) {
    int N = (int)w.rows, K = (int)w.cols;
    if (N % 128 == 0 && K % 8 == 0) k_sgemm128<<<dim3(N / 128, (T + 127) / 128), 256, 0, G.stream>>>(y, x, w.p, T, N, K);
    else k_sgemm_gen<<<dim3((N + 31) / 32, (T + 31) / 32), 256, 0, G.stream>>>(y, x, K, w.p, T, N, K, 1.f, 0.f);
}

// ------------------------------------------------------------------------------------------------ model plumbing
static SafeTensors* g_st = nullptr;
static std::string g_prefix;
static std::map<std::string, std::pair<const StTensor*, const StTensor*>> g_lora;  // module -> (A, B)
static float g_lora_scale = 1.f;
static int T = 0;
static float* g_tmp = nullptr;  // LoRA rank buffer [T, 64]

struct Lin {
    Weight w, la, lb;
    float ls = 0.f;
    int8_t* qp = nullptr; float* sp = nullptr;  // plain int8
    int8_t* qs = nullptr; float* ss = nullptr;  // smoothed int8
    int N() const { return (int)w.rows; }
    int K() const { return (int)w.cols; }
};
static Lin load_lin(const std::string& mod) {
    Lin l;
    l.w = upload_weight(g_st->get(g_prefix + mod + ".weight"), Place::Device);
    auto it = g_lora.find(mod);
    if (it != g_lora.end()) {
        l.la = upload_weight(*it->second.first, Place::Device);
        l.lb = upload_weight(*it->second.second, Place::Device);
        l.ls = g_lora_scale;
    }
    return l;
}
static void free_lin(Lin& l) {
    free_weight(l.w);
    if (l.la) free_weight(l.la);
    if (l.lb) free_weight(l.lb);
    for (void* p : {(void*)l.qp, (void*)l.sp, (void*)l.qs, (void*)l.ss}) if (p) CK(cudaFree(p));
    l = Lin();
}
static void lora_side(float* y, const float* x, int M, const Lin& l) {
    if (!l.la) return;
    int r = (int)l.la.rows, K = l.K(), N = l.N();
    k_sgemm_gen<<<dim3((r + 31) / 32, (M + 31) / 32), 256, 0, G.stream>>>(g_tmp, x, K, l.la.p, M, r, K, 1.f, 0.f);
    k_sgemm_gen<<<dim3((N + 31) / 32, (M + 31) / 32), 256, 0, G.stream>>>(y, g_tmp, r, l.lb.p, M, N, r, l.ls, 1.f);
}
// modulation weights run through the engine's matvec, which applies an attached LoRA itself
struct ModW { Weight w; std::vector<LoraAdd> lora; };
static ModW load_mod(const std::string& mod) {
    ModW m;
    m.w = upload_weight(g_st->get(g_prefix + mod + ".weight"), Place::Device);
    auto it = g_lora.find(mod);
    if (it != g_lora.end()) {
        LoraAdd a;
        a.A = upload_weight(*it->second.first, Place::Device);
        a.B = upload_weight(*it->second.second, Place::Device);
        a.scale = g_lora_scale;
        m.lora.push_back(a);
    }
    return m;
}
static void free_mod(ModW& m) {
    free_weight(m.w);
    for (auto& a : m.lora) { free_weight(a.A); free_weight(a.B); }
    m = ModW();
}
static void matvec_mod(float* y, ModW& m, const float* x, const float* add) {
    m.w.lora = m.lora.empty() ? nullptr : &m.lora;
    matvec(y, m.w, x, add);
    m.w.lora = nullptr;
}

// Cosmos 3D RoPE for an image (copied from dit.cu)
static void cosmos_rope(std::vector<float>& c, std::vector<float>& s, int Hp, int Wp) {
    const int dim_h = DH / 6 * 2, dim_t = DH - 2 * dim_h;
    double ntk = pow(4.0, (double)dim_h / (dim_h - 2));
    float h_theta = (float)(10000.0 * ntk);
    std::vector<float> hf(dim_h / 2);
    for (int j = 0; j < dim_h / 2; j++) hf[j] = 1.0f / powf(h_theta, (float)(2 * j) / (float)dim_h);
    int TT = Hp * Wp, half = DH / 2;
    c.assign((size_t)TT * half, 1.f);
    s.assign((size_t)TT * half, 0.f);
    for (int t = 0; t < TT; t++) {
        int y = t / Wp, x = t % Wp;
        for (int j = 0; j < dim_h / 2; j++) {
            float ah = (float)y * hf[j], aw = (float)x * hf[j];
            c[(size_t)t * half + dim_t / 2 + j] = cosf(ah);
            s[(size_t)t * half + dim_t / 2 + j] = sinf(ah);
            c[(size_t)t * half + dim_t / 2 + dim_h / 2 + j] = cosf(aw);
            s[(size_t)t * half + dim_t / 2 + dim_h / 2 + j] = sinf(aw);
        }
    }
}

// ------------------------------------------------------------------------------------------------ experiment state
struct Stream {
    std::string mode;  // 6 letters
    float* X = nullptr;
    float* lat = nullptr;
    float* vel = nullptr;
};
static std::vector<Stream> S;
static float *bN, *bY, *bBig, *bAlt, *bTok;  // shared scratch
static int8_t *xq_p, *xq_s;
static float *xs_p, *xs_s;
static unsigned* g_cm;          // column max scratch (bits)
static float *g_rmax, *g_rrms;  // row stats scratch
static float alpha = 0.5f;
static bool analyze = false, calibrating = false, have_cal = false;
static int cur_step = 0, cur_block = 0;
// calibration: per (block, site) per-input-channel absmax over all calibration tokens and steps
static std::vector<float> cal[NB][NSITE];
struct SmoothV { float* mul = nullptr; float* inv = nullptr; };
static SmoothV sm[NSITE];  // current block

struct ErrRec { int step, block, layer, scheme; double rel; };
static std::vector<ErrRec> errs;
struct ActRec { int step, block, site; double ch_ratio, max_abs, rms, tok_ratio_med, tok_ratio_max, qerr_p, qerr_s; int n_out; };
static std::vector<ActRec> acts;

static bool site_needs(int site, char c) {
    if (analyze && have_cal) return true;  // reference stream probes every scheme
    if (analyze && c == 'p') return true;
    for (auto& s : S) if (s.mode[site] == c) return true;
    return false;
}

// Prepares int8 weights (+ smoothing vectors) for one site's linears.
static void prep_site(int site, std::vector<Lin*> lins) {
    const int K = lins[0]->K();
    bool need_p = site_needs(site, 'p') || analyze, need_s = have_cal && site_needs(site, 's');
    if (need_p)
        for (Lin* l : lins) {
            l->qp = dalloc<int8_t>((size_t)l->N() * K);
            l->sp = dalloc<float>(l->N());
            i8::quantize_weight(l->qp, l->sp, l->w.p, nullptr, l->N(), K, G.stream);
        }
    if (need_s) {
        CK(cudaMemsetAsync(g_cm, 0, K * 4, G.stream));
        for (Lin* l : lins) k_colmax_bf16<<<dim3((K + 255) / 256, (l->N() + 63) / 64), 256, 0, G.stream>>>(l->w.p, l->N(), K, g_cm);
        float* xm = dalloc<float>(K);
        CK(cudaMemcpy(xm, cal[cur_block][site].data(), K * 4, cudaMemcpyHostToDevice));
        sm[site].mul = dalloc<float>(K);
        sm[site].inv = dalloc<float>(K);
        k_smooth<<<nb(K), 256, 0, G.stream>>>(xm, g_cm, K, alpha, sm[site].mul, sm[site].inv);
        CK(cudaStreamSynchronize(G.stream));
        CK(cudaFree(xm));
        for (Lin* l : lins) {
            l->qs = dalloc<int8_t>((size_t)l->N() * K);
            l->ss = dalloc<float>(l->N());
            i8::quantize_weight(l->qs, l->ss, l->w.p, sm[site].mul, l->N(), K, G.stream);
        }
    }
}
static void unprep_site(int site) {
    if (sm[site].mul) { CK(cudaFree(sm[site].mul)); CK(cudaFree(sm[site].inv)); }
    sm[site] = SmoothV();
}

static void run_one(char c, float* y, const float* x, const Lin& l) {
    const int K = l.K(), N = l.N();
    switch (c) {
        case 'r': ref_gemm(y, x, T, l.w); break;
        case 'h': linear(y, x, T, l.w); break;
        case 'p': i8::gemm(y, xq_p, xs_p, l.qp, l.sp, T, N, K, 0.f, G.stream); break;
        case 's': i8::gemm(y, xq_s, xs_s, l.qs, l.ss, T, N, K, 0.f, G.stream); break;
        default: throw std::runtime_error(std::string("bad mode ") + c);
    }
    lora_side(y, x, T, l);
}

// Activation statistics of a site input (reference stream only).
static void act_stats(int site, const float* x, int K) {
    CK(cudaMemsetAsync(g_cm, 0, K * 4, G.stream));
    k_colmax<<<dim3((K + 255) / 256, (T + 63) / 64), 256, 0, G.stream>>>(x, T, K, g_cm);
    std::vector<float> cm(K);
    CK(cudaMemcpy(cm.data(), g_cm, K * 4, cudaMemcpyDeviceToHost));
    if (calibrating) {
        auto& c = cal[cur_block][site];
        if (c.empty()) c.assign(K, 0.f);
        for (int k = 0; k < K; k++) c[k] = std::max(c[k], cm[k]);
        return;
    }
    ActRec a{};
    a.step = cur_step; a.block = cur_block; a.site = site;
    std::vector<float> s = cm;
    std::nth_element(s.begin(), s.begin() + K / 2, s.end());
    float med = s[K / 2], mx = *std::max_element(cm.begin(), cm.end());
    a.ch_ratio = mx / std::max(med, 1e-12f);
    a.max_abs = mx;
    a.n_out = 0;
    for (float v : cm) if (v > 8.f * med) a.n_out++;
    k_rowstats<<<T, 256, 0, G.stream>>>(x, K, g_rmax, g_rrms);
    std::vector<float> rm = dl(g_rmax, T), rr = dl(g_rrms, T), ratio(T);
    double ms = 0;
    for (int t = 0; t < T; t++) { ratio[t] = rm[t] / std::max(rr[t], 1e-12f); ms += (double)rr[t] * rr[t]; }
    a.rms = sqrt(ms / T);
    std::sort(ratio.begin(), ratio.end());
    a.tok_ratio_med = ratio[T / 2];
    a.tok_ratio_max = ratio[T - 1];
    a.qerr_p = qerr(x, xq_p, xs_p, nullptr, T, K);
    a.qerr_s = have_cal ? qerr(x, xq_s, xs_s, sm[site].mul, T, K) : 0.0;
    acts.push_back(a);
}

// One site (1 or 3 linears sharing the input x) for a stream with mode letter c.
static void run_site(bool is_ref, char c, int site, const float* x, std::vector<Lin*> lins, std::vector<float*> outs, std::vector<int> layer_ids) {
    const int K = lins[0]->K();
    bool probe = is_ref && analyze;
    if (c == 'p' || probe) i8::quantize_rows(xq_p, xs_p, x, nullptr, T, K, G.stream);
    if (c == 's' || (probe && have_cal)) i8::quantize_rows(xq_s, xs_s, x, sm[site].inv, T, K, G.stream);
    for (size_t i = 0; i < lins.size(); i++) run_one(c, outs[i], x, *lins[i]);
    if (is_ref && (analyze || calibrating)) act_stats(site, x, K);
    if (!probe) return;
    for (size_t i = 0; i < lins.size(); i++) {
        const size_t n = (size_t)T * lins[i]->N();
        const char sch[NSCHEME] = {'h', 'p', 's'};
        for (int k = 0; k < NSCHEME; k++) {
            if (sch[k] == 's' && !have_cal) continue;
            run_one(sch[k], bAlt, x, *lins[i]);
            errs.push_back({cur_step, cur_block, layer_ids[i], k, rel_err(bAlt, outs[i], n)});
        }
    }
}

// ------------------------------------------------------------------------------------------------ forward
static void forward_all(float t, int Hl, int Wl, const float* rcos, const float* rsin, const float* ctx, int Lc, int npad) {
    const float eps = 1e-6f;
    const int Hp = Hl / 2, Wp = Wl / 2;
    (void)Hp; (void)Wp;
    // timestep embedding (engine order: sinusoid -> linear_1 -> silu -> linear_2 = adaLN-LoRA vector; temb = silu(rmsnorm(sinusoid)))
    std::vector<float> sh(D);
    for (int i = 0; i < D / 2; i++) {
        float e = expf((float)i * (-logf(10000.f)) / (float)(D / 2));
        sh[i] = cosf(t * e);
        sh[i + D / 2] = sinf(t * e);
    }
    float* sv = dalloc<float>(D);
    float* s1 = dalloc<float>(D);
    float* lora = dalloc<float>(3 * D);
    float* temb = dalloc<float>(D);
    float* h256 = dalloc<float>(256);
    float* mo = dalloc<float>(9 * D);
    CK(cudaMemcpy(sv, sh.data(), D * 4, cudaMemcpyHostToDevice));
    {
        ModW l1 = load_mod("t_embedder.1.linear_1"), l2 = load_mod("t_embedder.1.linear_2");
        Weight tn = upload_weight(g_st->get(g_prefix + "t_embedding_norm.weight"), Place::Device);
        matvec_mod(s1, l1, sv, nullptr);
        silu(s1, s1, D);
        matvec_mod(lora, l2, s1, nullptr);
        rmsnorm(temb, sv, &tn, 1, D, eps);
        silu(temb, temb, D);
        gpu_sync();
        free_mod(l1); free_mod(l2); free_weight(tn);
    }
    // x_embedder
    {
        Weight xe = upload_weight(g_st->get(g_prefix + "x_embedder.proj.1.weight"), Place::Device);
        for (auto& s : S) {
            k_patchify<<<nb((size_t)T * 68), 256, 0, G.stream>>>(bTok, s.lat, Hl, Wl);
            ref_gemm(s.X, bTok, T, xe);
        }
        gpu_sync();
        free_weight(xe);
    }
    float *Q = bBig, *Kb = bBig + (size_t)T * D, *V = bBig + 2 * (size_t)T * D, *A = bBig + 3 * (size_t)T * D;
    float* Kc = dalloc<float>((size_t)Lc * D);
    float* Vc = dalloc<float>((size_t)Lc * D);
    for (int b = 0; b < NB; b++) {
        cur_block = b;
        const std::string p = "blocks." + std::to_string(b) + ".";
        {   // modulation vectors for this block
            const char* nm[3] = {"adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp"};
            for (int j = 0; j < 3; j++) {
                ModW m1 = load_mod(p + nm[j] + ".1"), m2 = load_mod(p + nm[j] + ".2");
                matvec_mod(h256, m1, temb, nullptr);
                matvec_mod(mo + j * 3 * D, m2, h256, lora);
                gpu_sync();
                free_mod(m1); free_mod(m2);
            }
        }
        const float *msa = mo, *mca = mo + 3 * D, *mml = mo + 6 * D;
        {   // self-attention
            Lin q = load_lin(p + "self_attn.q_proj"), k = load_lin(p + "self_attn.k_proj"), v = load_lin(p + "self_attn.v_proj"),
                o = load_lin(p + "self_attn.output_proj");
            Weight qn = upload_weight(g_st->get(g_prefix + p + "self_attn.q_norm.weight"), Place::Device);
            Weight kn = upload_weight(g_st->get(g_prefix + p + "self_attn.k_norm.weight"), Place::Device);
            prep_site(S_QKV, {&q, &k, &v});
            prep_site(S_O, {&o});
            for (size_t si = 0; si < S.size(); si++) {
                auto& s = S[si];
                layernorm_mod(bN, s.X, msa + D, msa, T, D, eps);
                run_site(si == 0, s.mode[S_QKV], S_QKV, bN, {&q, &k, &v}, {Q, Kb, V}, {L_Q, L_K, L_V});
                rmsnorm(Q, Q, &qn, T * NH, DH, eps);
                rmsnorm(Kb, Kb, &kn, T * NH, DH, eps);
                rope_half(Q, T, NH, DH, rcos, rsin);
                rope_half(Kb, T, NH, DH, rcos, rsin);
                attention(A, D, Q, D, Kb, D, V, D, T, T, NH, DH, false);
                run_site(si == 0, s.mode[S_O], S_O, A, {&o}, {bY}, {L_O});
                add_gated(s.X, bY, msa + 2 * D, T, D);
            }
            gpu_sync();
            unprep_site(S_QKV); unprep_site(S_O);
            free_lin(q); free_lin(k); free_lin(v); free_lin(o); free_weight(qn); free_weight(kn);
        }
        {   // cross-attention (context K/V in fp32 for every stream: 26 rows, cached per image in the engine)
            Lin q = load_lin(p + "cross_attn.q_proj"), k = load_lin(p + "cross_attn.k_proj"), v = load_lin(p + "cross_attn.v_proj"),
                o = load_lin(p + "cross_attn.output_proj");
            Weight qn = upload_weight(g_st->get(g_prefix + p + "cross_attn.q_norm.weight"), Place::Device);
            Weight kn = upload_weight(g_st->get(g_prefix + p + "cross_attn.k_norm.weight"), Place::Device);
            ref_gemm(Kc, ctx, Lc, k.w); lora_side(Kc, ctx, Lc, k);
            rmsnorm(Kc, Kc, &kn, Lc * NH, DH, eps);
            ref_gemm(Vc, ctx, Lc, v.w); lora_side(Vc, ctx, Lc, v);
            prep_site(S_CAQ, {&q});
            prep_site(S_CAO, {&o});
            for (size_t si = 0; si < S.size(); si++) {
                auto& s = S[si];
                layernorm_mod(bN, s.X, mca + D, mca, T, D, eps);
                run_site(si == 0, s.mode[S_CAQ], S_CAQ, bN, {&q}, {Q}, {L_CAQ});
                rmsnorm(Q, Q, &qn, T * NH, DH, eps);
                attention(A, D, Q, D, Kc, D, Vc, D, T, Lc, NH, DH, false, npad);
                run_site(si == 0, s.mode[S_CAO], S_CAO, A, {&o}, {bY}, {L_CAO});
                add_gated(s.X, bY, mca + 2 * D, T, D);
            }
            gpu_sync();
            unprep_site(S_CAQ); unprep_site(S_CAO);
            free_lin(q); free_lin(k); free_lin(v); free_lin(o); free_weight(qn); free_weight(kn);
        }
        {   // MLP
            Lin m1 = load_lin(p + "mlp.layer1"), m2 = load_lin(p + "mlp.layer2");
            prep_site(S_MLP1, {&m1});
            prep_site(S_MLP2, {&m2});
            for (size_t si = 0; si < S.size(); si++) {
                auto& s = S[si];
                layernorm_mod(bN, s.X, mml + D, mml, T, D, eps);
                run_site(si == 0, s.mode[S_MLP1], S_MLP1, bN, {&m1}, {bBig}, {L_MLP1});
                gelu(bBig, (size_t)T * F);
                run_site(si == 0, s.mode[S_MLP2], S_MLP2, bBig, {&m2}, {bY}, {L_MLP2});
                add_gated(s.X, bY, mml + 2 * D, T, D);
            }
            gpu_sync();
            unprep_site(S_MLP1); unprep_site(S_MLP2);
            free_lin(m1); free_lin(m2);
        }
    }
    {   // final layer
        ModW f1 = load_mod("final_layer.adaln_modulation.1"), f2 = load_mod("final_layer.adaln_modulation.2");
        Weight fl = upload_weight(g_st->get(g_prefix + "final_layer.linear.weight"), Place::Device);
        matvec_mod(h256, f1, temb, nullptr);
        matvec_mod(mo, f2, h256, lora);
        for (auto& s : S) {
            layernorm_mod(bN, s.X, mo + D, mo, T, D, eps);
            ref_gemm(bTok, bN, T, fl);
            k_unpatchify<<<nb((size_t)T * 64), 256, 0, G.stream>>>(s.vel, bTok, Hl, Wl);
        }
        gpu_sync();
        free_mod(f1); free_mod(f2); free_weight(fl);
    }
    for (void* ptr : {(void*)sv, (void*)s1, (void*)lora, (void*)temb, (void*)h256, (void*)mo, (void*)Kc, (void*)Vc}) CK(cudaFree(ptr));
}

// ------------------------------------------------------------------------------------------------ self-test of the reference GEMM
static void selftest_ref() {
    const int M = 200, N = 256, K = 1000;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    std::vector<float> a((size_t)M * K);
    std::vector<uint16_t> w((size_t)N * K);
    for (auto& v : a) v = nd(rng);
    for (auto& v : w) { float f = nd(rng) * 0.05f; uint32_t u; memcpy(&u, &f, 4); v = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
    float *da = dalloc<float>(a.size()), *dc = dalloc<float>((size_t)M * N), *dc2 = dalloc<float>((size_t)M * N);
    uint16_t* dw = dalloc<uint16_t>(w.size());
    CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    k_sgemm128<<<dim3(N / 128, (M + 127) / 128), 256, 0, G.stream>>>(dc, da, dw, M, N, K);
    k_sgemm_gen<<<dim3(N / 32, (M + 31) / 32), 256, 0, G.stream>>>(dc2, da, K, dw, M, N, K, 1.f, 0.f);
    std::vector<float> c = dl(dc, (size_t)M * N), c2 = dl(dc2, (size_t)M * N);
    double e1 = 0, e2 = 0, rr = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) { uint32_t u = (uint32_t)w[(size_t)n * K + k] << 16; float f; memcpy(&f, &u, 4); s += (double)a[(size_t)m * K + k] * f; }
            e1 += (c[(size_t)m * N + n] - s) * (c[(size_t)m * N + n] - s);
            e2 += (c2[(size_t)m * N + n] - s) * (c2[(size_t)m * N + n] - s);
            rr += s * s;
        }
    printf("reference GEMM self-test vs fp64: tiled rel %.2e, generic rel %.2e\n", sqrt(e1 / rr), sqrt(e2 / rr));
    cudaFree(da); cudaFree(dc); cudaFree(dc2); cudaFree(dw);
}


// ------------------------------------------------------------------------------------------------ VAE decode of final latents
__constant__ float c_mean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                 0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
__constant__ float c_std[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
__global__ void k_to_vae(float* z, int P) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)16 * P) z[i] = z[i] * c_std[i / P] + c_mean[i / P];
}
static int decode_mode(const std::string& root, const std::string& dir) {
    gpu_init(0);
    size_t fb = gpu_free_bytes();
    if (fb < ((size_t)1536 << 20)) {  // the decoder needs a ~450 MB arena: never squeeze a running Kiln server
        printf("only %zu MB VRAM free: refusing to decode next to a running engine (needs ~1.5 GB free)\n", fb >> 20);
        return 1;
    }
    size_t arena = std::min<size_t>(fb > ((size_t)200 << 20) ? fb - ((size_t)200 << 20) : 0, (size_t)448 << 20);
    printf("free VRAM %zu MB, decode arena %zu MB (VAE weights in pinned host RAM)\n", fb >> 20, arena >> 20);
    gpu_alloc_scratch((size_t)4 << 20, arena);
    Vae vae;
    vae.load(root + "/models/qwen_image_vae.safetensors", Place::Host);
    const int Hl = 96, Wl = 64, P = Hl * Wl, H = Hl * 8, W = Wl * 8;
    float* z = G.arena.f((size_t)16 * P);
    float* img = G.arena.f((size_t)3 * H * W);
    for (auto& e : std::filesystem::directory_iterator(dir)) {
        std::string fn = e.path().filename().string();
        if (fn.rfind("latent_final_", 0) != 0 || e.path().extension() != ".npy") continue;
        std::vector<float> lat;
        std::vector<int64_t> shp;
        if (!load_npy(e.path().string(), lat, shp) || lat.size() != (size_t)16 * P) continue;
        CK(cudaMemcpy(z, lat.data(), lat.size() * 4, cudaMemcpyHostToDevice));
        k_to_vae<<<nb((size_t)16 * P), 256, 0, G.stream>>>(z, P);
        vae.decode(z, Hl, Wl, img);
        std::vector<float> rgb = dl(img, (size_t)3 * H * W), hwc((size_t)H * W * 3);
        for (int c = 0; c < 3; c++)
            for (size_t i = 0; i < (size_t)H * W; i++) hwc[i * 3 + c] = std::min(1.f, std::max(0.f, (rgb[(size_t)c * H * W + i] + 1.f) * 0.5f));
        std::string tag = fn.substr(13, fn.size() - 13 - 4);
        save_npy(dir + "/image_" + tag + ".npy", hwc.data(), {H, W, 3});
        printf("decoded %s\n", tag.c_str());
        fflush(stdout);
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
    std::string root = "..", out_dir = "i8_out", streams_arg = "hhhhhh,pppppp,ssssss,pppphh,sssssh";
    bool use_lora = true, decode = false;
    int steps = 8, calib_steps = 8;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--streams") streams_arg = next();
        else if (a == "--alpha") alpha = std::stof(next());
        else if (a == "--nolora") use_lora = false;
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--calib-steps") calib_steps = std::stoi(next());
        else if (a == "--out") out_dir = next();
        else if (a == "--root") root = next();
        else if (a == "--decode") decode = true;
    }
    if (decode) {
        try { return decode_mode(root, out_dir); }
        catch (std::exception& e) { printf("ERROR: %s\n", e.what()); return 1; }
    }
    try {
        CK(cudaSetDevice(0));
        CK(cudaSetDeviceFlags(cudaDeviceMapHost));
        {   // ~400 MB of scratch + one block stage of weights; do not run it next to a VRAM-tight Kiln server
            size_t f, t;
            CK(cudaMemGetInfo(&f, &t));
            if (f < ((size_t)1200 << 20)) { printf("only %zu MB VRAM free: needs ~400 MB plus headroom, refusing\n", f >> 20); return 1; }
        }
        CK(cudaStreamCreate(&G.stream));
        G.fp16 = true;
        G.arena.cap = 8 << 20;
        CK(cudaMalloc(&G.arena.base, G.arena.cap));
        selftest_ref();

        SafeTensors st(root + "/models/anima-base-v1.0.safetensors");
        g_st = &st;
        for (auto& [name, t] : st.all()) {
            const std::string tail = "x_embedder.proj.1.weight";
            if (name.size() >= tail.size() && name.compare(name.size() - tail.size(), tail.size(), tail) == 0) { g_prefix = name.substr(0, name.size() - tail.size()); break; }
        }
        std::unique_ptr<SafeTensors> lst;
        if (use_lora) {
            lst = std::make_unique<SafeTensors>(root + "/models/anima-turbo-lora-v0.2.safetensors");
            std::map<std::string, const StTensor*> A, B;
            for (auto& [name, t] : lst->all()) {
                std::string n = name;
                if (n.rfind("diffusion_model.", 0) == 0) n = n.substr(16);
                auto ends = [&](const char* s) { size_t l = strlen(s); return n.size() > l && n.compare(n.size() - l, l, s) == 0; };
                if (ends(".lora_A.weight")) A[n.substr(0, n.size() - 14)] = &t;
                else if (ends(".lora_B.weight")) B[n.substr(0, n.size() - 14)] = &t;
            }
            for (auto& [m, a] : A) if (B.count(m)) g_lora[m] = {a, B[m]};
            printf("turbo LoRA: %zu modules, scale %.2f\n", g_lora.size(), g_lora_scale);
        }

        const std::string gd = root + "/ref/golden/";
        std::vector<float> noise, sig, ctxh;
        std::vector<int64_t> shp;
        if (!load_npy(gd + "noise.npy", noise, shp) || !load_npy(gd + "sigmas.npy", sig, shp) ||
            !load_npy(gd + (use_lora ? "adapter_out.npy" : "adapter_out_nolora.npy"), ctxh, shp))
            throw std::runtime_error("golden files missing in " + gd);
        const int Hl = 96, Wl = 64, P = Hl * Wl, Lfull = (int)shp[0];
        T = (Hl / 2) * (Wl / 2);
        int Lc = 0;
        for (int r = 0; r < Lfull; r++)
            for (int c = 0; c < 1024; c++) if (ctxh[(size_t)r * 1024 + c] != 0.f) { Lc = r + 1; break; }
        const int npad = Lfull - Lc;
        printf("tokens %d, context %d real + %d zero rows, sigmas", T, Lc, npad);
        for (float s : sig) printf(" %.3f", s);
        printf("\n");
        float* ctx = dalloc<float>((size_t)Lc * 1024);
        CK(cudaMemcpy(ctx, ctxh.data(), (size_t)Lc * 1024 * 4, cudaMemcpyHostToDevice));
        std::vector<float> rc, rs;
        cosmos_rope(rc, rs, Hl / 2, Wl / 2);
        float* rope = dalloc<float>(rc.size() * 2);
        CK(cudaMemcpy(rope, rc.data(), rc.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(rope + rc.size(), rs.data(), rs.size() * 4, cudaMemcpyHostToDevice));

        // scratch
        bN = dalloc<float>((size_t)T * D); bY = dalloc<float>((size_t)T * D);
        bBig = dalloc<float>((size_t)T * F); bAlt = dalloc<float>((size_t)T * F); bTok = dalloc<float>((size_t)T * 68);
        xq_p = dalloc<int8_t>((size_t)T * F); xq_s = dalloc<int8_t>((size_t)T * F);
        xs_p = dalloc<float>(T); xs_s = dalloc<float>(T);
        g_cm = dalloc<unsigned>(F); g_rmax = dalloc<float>(T); g_rrms = dalloc<float>(T);
        g_tmp = dalloc<float>((size_t)T * 64);
        g_acc = dalloc<double>(2);

        auto make_streams = [&](const std::vector<std::string>& modes) {
            for (auto& s : S) { cudaFree(s.X); cudaFree(s.lat); cudaFree(s.vel); }
            S.clear();
            for (auto& m : modes) {
                if (m.size() != NSITE) throw std::runtime_error("stream spec must have 6 letters: " + m);
                Stream s;
                s.mode = m;
                s.X = dalloc<float>((size_t)T * D); s.lat = dalloc<float>((size_t)16 * P); s.vel = dalloc<float>((size_t)16 * P);
                S.push_back(s);
            }
        };
        auto t_start = std::chrono::steady_clock::now();
        auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };

        // ---------------- phase 1: calibration trajectory (reference only, different noise seed)
        if (calib_steps > 0) {
            make_streams({"rrrrrr"});
            std::mt19937 rng(12345);
            std::normal_distribution<float> nd;
            std::vector<float> cn((size_t)16 * P);
            for (auto& v : cn) v = nd(rng) * sig[0];
            CK(cudaMemcpy(S[0].lat, cn.data(), cn.size() * 4, cudaMemcpyHostToDevice));
            calibrating = true;
            for (int i = 0; i < calib_steps && i < (int)sig.size() - 1; i++) {
                cur_step = i;
                forward_all(sig[i], Hl, Wl, rope, rope + rc.size(), ctx, Lc, npad);
                k_axpy2<<<nb((size_t)16 * P), 256, 0, G.stream>>>(S[0].lat, S[0].vel, sig[i + 1] - sig[i], (size_t)16 * P);
                gpu_sync();
                printf("calibration step %d done (%.0f s)\n", i, secs());
                fflush(stdout);
            }
            calibrating = false;
            have_cal = true;
            // per-input-channel activation absmax, for SmoothQuant factors at engine load:
            // float32, block-major, sites in order qkv, o, cross_q, cross_o (2048 each), mlp1 (2048), mlp2 (8192)
            std::filesystem::create_directories(out_dir);
            if (FILE* f = fopen((out_dir + "/calib_absmax.f32").c_str(), "wb")) {
                for (int b = 0; b < NB; b++)
                    for (int st = 0; st < NSITE; st++) fwrite(cal[b][st].data(), 4, cal[b][st].size(), f);
                fclose(f);
            }
        }

        // ---------------- phase 2: evaluation trajectory on the golden noise
        std::vector<std::string> modes = {"rrrrrr"};
        for (size_t p0 = 0; p0 <= streams_arg.size();) {
            size_t p1 = streams_arg.find(',', p0);
            if (p1 == std::string::npos) p1 = streams_arg.size();
            std::string m = streams_arg.substr(p0, p1 - p0);
            if (!m.empty()) {
                if (!have_cal && m.find('s') != std::string::npos) { printf("skipping %s (no calibration)\n", m.c_str()); }
                else modes.push_back(m);
            }
            p0 = p1 + 1;
        }
        make_streams(modes);
        for (auto& s : S) CK(cudaMemcpy(s.lat, noise.data(), noise.size() * 4, cudaMemcpyHostToDevice));  // sigma0 = 1
        analyze = true;
        std::vector<std::vector<double>> vel_err(S.size()), lat_err(S.size());
        std::vector<double> ref_vs_golden_v, ref_vs_golden_lat;
        for (int i = 0; i < steps && i < (int)sig.size() - 1; i++) {
            cur_step = i;
            forward_all(sig[i], Hl, Wl, rope, rope + rc.size(), ctx, Lc, npad);
            std::vector<float> vref = dl(S[0].vel, (size_t)16 * P), g;
            if (load_npy(gd + "dit_out_step" + std::to_string(i) + ".npy", g, shp) && use_lora) ref_vs_golden_v.push_back(rel_host(vref, g));
            for (size_t k = 1; k < S.size(); k++) vel_err[k].push_back(rel_err(S[k].vel, S[0].vel, (size_t)16 * P));
            for (auto& s : S) k_axpy2<<<nb((size_t)16 * P), 256, 0, G.stream>>>(s.lat, s.vel, sig[i + 1] - sig[i], (size_t)16 * P);
            gpu_sync();
            std::vector<float> lref = dl(S[0].lat, (size_t)16 * P);
            if (load_npy(gd + "latent_step" + std::to_string(i) + ".npy", g, shp) && use_lora) ref_vs_golden_lat.push_back(rel_host(lref, g));
            for (size_t k = 1; k < S.size(); k++) lat_err[k].push_back(rel_err(S[k].lat, S[0].lat, (size_t)16 * P));
            printf("eval step %d done (%.0f s)\n", i, secs());
            fflush(stdout);
        }

        // ---------------- report
        std::filesystem::create_directories(out_dir);
        {
            FILE* f = fopen((out_dir + "/layer_err.csv").c_str(), "w");
            fprintf(f, "step,block,layer,scheme,rel_l2\n");
            for (auto& e : errs) fprintf(f, "%d,%d,%s,%s,%.6e\n", e.step, e.block, LAYER_NAME[e.layer], SCHEME_NAME[e.scheme], e.rel);
            fclose(f);
            f = fopen((out_dir + "/act_stats.csv").c_str(), "w");
            fprintf(f, "step,block,site,ch_max_over_median,n_ch_gt8x_median,max_abs,rms,tok_absmax_over_rms_median,tok_absmax_over_rms_max,act_qerr_i8,act_qerr_i8smooth\n");
            for (auto& a : acts)
                fprintf(f, "%d,%d,%s,%.3f,%d,%.4g,%.4g,%.3f,%.3f,%.4e,%.4e\n", a.step, a.block, SITE_NAME[a.site], a.ch_ratio, a.n_out, a.max_abs, a.rms,
                        a.tok_ratio_med, a.tok_ratio_max, a.qerr_p, a.qerr_s);
            fclose(f);
        }
        printf("\n== per-layer rel_l2 vs fp32 on the reference input (mean over %d steps x 28 blocks; worst block in brackets)\n", steps);
        printf("%-9s", "layer");
        for (int k = 0; k < NSCHEME; k++) printf(" %24s", SCHEME_NAME[k]);
        printf("\n");
        for (int l = 0; l < NLAYER; l++) {
            printf("%-9s", LAYER_NAME[l]);
            for (int k = 0; k < NSCHEME; k++) {
                double sum = 0, mx = 0;
                int n = 0, mb = -1;
                std::vector<double> perblock(NB, 0);
                std::vector<int> cnt(NB, 0);
                for (auto& e : errs) if (e.layer == l && e.scheme == k) { sum += e.rel; n++; perblock[e.block] += e.rel; cnt[e.block]++; }
                for (int b = 0; b < NB; b++) if (cnt[b] && perblock[b] / cnt[b] > mx) { mx = perblock[b] / cnt[b]; mb = b; }
                if (n) printf("   %9.2e [%8.2e b%2d]", sum / n, mx, mb);
                else printf(" %24s", "-");
            }
            printf("\n");
        }
        printf("\n== activation statistics per site (median over blocks and steps; max in brackets)\n");
        printf("%-8s %22s %16s %24s %24s %24s\n", "site", "ch absmax max/median", "#ch > 8x median", "token absmax/rms (max)", "act qerr i8", "act qerr i8+smooth");
        for (int s = 0; s < NSITE; s++) {
            std::vector<double> cr, no, tr, qp, qs;
            double crm = 0, trm = 0, qpm = 0, qsm = 0, nom = 0;
            for (auto& a : acts)
                if (a.site == s) {
                    cr.push_back(a.ch_ratio); no.push_back(a.n_out); tr.push_back(a.tok_ratio_max); qp.push_back(a.qerr_p); qs.push_back(a.qerr_s);
                    crm = std::max(crm, a.ch_ratio); nom = std::max(nom, (double)a.n_out); trm = std::max(trm, a.tok_ratio_max);
                    qpm = std::max(qpm, a.qerr_p); qsm = std::max(qsm, a.qerr_s);
                }
            auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            printf("%-8s %12.1f [%7.1f] %8.0f [%5.0f] %14.1f [%7.1f] %14.2e [%7.2e] %14.2e [%7.2e]\n", SITE_NAME[s], med(cr), crm, med(no), nom, med(tr), trm,
                   med(qp), qpm, med(qs), qsm);
        }
        printf("\n== end to end (each stream runs its own 8-step trajectory from the golden noise)\n");
        if (!ref_vs_golden_v.empty()) {
            printf("reference vs ComfyUI golden: velocity rel_l2 per step");
            for (double v : ref_vs_golden_v) printf(" %.2e", v);
            printf("\n                              latent rel_l2 per step  ");
            for (double v : ref_vs_golden_lat) printf(" %.2e", v);
            printf("\n");
        }
        for (size_t k = 1; k < S.size(); k++) {
            printf("stream %s  velocity rel_l2 vs ref per step:", S[k].mode.c_str());
            for (double v : vel_err[k]) printf(" %.2e", v);
            printf("\n        %s  latent rel_l2 vs ref per step:  ", std::string(S[k].mode.size(), ' ').c_str());
            for (double v : lat_err[k]) printf(" %.2e", v);
            printf("\n");
        }
        for (size_t k = 0; k < S.size(); k++) {
            std::vector<float> l = dl(S[k].lat, (size_t)16 * P);
            save_npy(out_dir + "/latent_final_" + S[k].mode + ".npy", l.data(), {16, Hl, Wl});
        }
        printf("\ntotal %.0f s; csv + final latents in %s/\n", secs(), out_dir.c_str());
    } catch (std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
