#include "gpu.h"
#include "hgemm.cuh"
#include "flash.cuh"
#include "tcflash.cuh"
#include "tcgemm.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

Gpu G;

// ---------------------------------------------------------------------------
// profiler
// ---------------------------------------------------------------------------
#include <map>
struct ProfEvent { const char* name; cudaEvent_t a, b; };
static bool prof_on = false;
static std::vector<ProfEvent> prof_events;
static size_t prof_used = 0;

ProfScope::ProfScope(const char* name) {
    if (!prof_on) return;
    if (prof_used == prof_events.size()) {
        ProfEvent e{name, nullptr, nullptr};
        cudaEventCreate(&e.a);
        cudaEventCreate(&e.b);
        prof_events.push_back(e);
    }
    slot = (int)prof_used++;
    prof_events[slot].name = name;
    cudaEventRecord(prof_events[slot].a, G.stream);
}
ProfScope::~ProfScope() {
    if (slot >= 0) cudaEventRecord(prof_events[slot].b, G.stream);
}
void prof_enable(bool on) { prof_on = on; prof_used = 0; }
std::string prof_report() {
    cudaStreamSynchronize(G.stream);
    std::map<std::string, std::pair<double, int>> acc;
    double total = 0;
    for (size_t i = 0; i < prof_used; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, prof_events[i].a, prof_events[i].b);
        acc[prof_events[i].name].first += ms;
        acc[prof_events[i].name].second++;
        total += ms;
    }
    std::string out;
    char b[160];
    for (auto& [n, v] : acc) {
        snprintf(b, sizeof b, "%-14s %8.1f ms %5.1f%% (%d calls)\n", n.c_str(), v.first, 100 * v.first / total, v.second);
        out += b;
    }
    snprintf(b, sizeof b, "%-14s %8.1f ms\n", "sum", total);
    return out + b;
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------
// Times the tensor-core GEMM against the HFMA2 one on a DiT-sized product (ms each, best of 3).
static bool time_gemms(double& t_tc, double& t_hg) {
    const int M = 1024, N = 2048, K = 2048;
    float *A = nullptr, *C = nullptr;
    uint16_t* W = nullptr;
    bool ok = cudaMalloc(&A, (size_t)M * K * 4) == cudaSuccess && cudaMalloc(&C, (size_t)M * N * 4) == cudaSuccess &&
              cudaMalloc(&W, (size_t)N * K * 2) == cudaSuccess;
    if (ok) {
        cudaMemset(A, 0, (size_t)M * K * 4);
        cudaMemset(W, 0, (size_t)N * K * 2);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        auto best = [&](auto fn) {
            fn();
            float b = 1e30f;
            for (int i = 0; i < 3; i++) {
                cudaEventRecord(e0);
                fn();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                b = std::min(b, ms);
            }
            return (double)b;
        };
        t_tc = best([&] { tc::launch(C, A, W, M, N, K, 0.f, 0, false); });
        t_hg = best([&] { hg::launch(C, A, W, M, N, K, 0.f, 0, 1, false); });
        ok = cudaGetLastError() == cudaSuccess;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    cudaGetLastError();
    cudaFree(A);
    cudaFree(C);
    cudaFree(W);
    return ok;
}

// Reads the GPU and picks the fp16 kernels: tensor cores on RTX / Ampere and newer, HFMA2 on the CUDA
// cores otherwise. Turing is ambiguous (RTX 20xx has tensor cores, GTX 16xx does not, both are sm_75),
// so there the two GEMMs race on a real-sized product and the faster one wins.
static void pick_kernels(int device) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, device));
    G.gpu_name = p.name;
    G.sm = p.major * 10 + p.minor;
    G.vram_total = p.totalGlobalMem;
    const char* env = getenv("KILN_TC");
    char why[160];
    if (env && *env) {
        G.tc = env[0] == '1';
        snprintf(why, sizeof why, "KILN_TC=%s", env);
    } else if (G.sm >= 80) {
        G.tc = true;
        snprintf(why, sizeof why, "sm_%d has tensor cores", G.sm);
    } else if (G.sm == 75) {
        double t_tc = 0, t_hg = 0;
        if (time_gemms(t_tc, t_hg)) {
            G.tc = t_tc < 0.9 * t_hg;
            snprintf(why, sizeof why, "tensor cores %.2f ms vs CUDA cores %.2f ms per test GEMM", t_tc, t_hg);
        } else {
            snprintf(why, sizeof why, "the GEMM race failed");
        }
    } else {
        snprintf(why, sizeof why, "sm_%d", G.sm);
    }
    G.tc_reason = why;
}

void gpu_init(int device) {
    CK(cudaSetDevice(device));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    // Blocking stream on purpose: the plain cudaMemcpy calls (legacy default stream) must be ordered
    // against the engine's kernels. A non-blocking stream let the first kernels of a job read inputs
    // before their upload landed, i.e. read the previous job's leftovers.
    CK(cudaStreamCreate(&G.stream));
    CB(cublasCreate(&G.blas));
    CB(cublasSetStream(G.blas, G.stream));
    CB(cublasSetMathMode(G.blas, CUBLAS_PEDANTIC_MATH));  // plain fp32 FMA, no TF32/fast paths
    pick_kernels(device);
}

size_t gpu_free_bytes() {
    size_t f, t;
    CK(cudaMemGetInfo(&f, &t));
    return f;
}

void gpu_sync() { CK(cudaStreamSynchronize(G.stream)); }

__global__ void k_checksum(const uint16_t* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        s += (unsigned long long)p[i] * (i % 65521 + 1);
    atomicAdd(out, s);
}

std::vector<unsigned long long> weight_checksums() {
    unsigned long long* d;
    size_t n = G.weights.size();
    CK(cudaMalloc(&d, n * 8));
    CK(cudaMemset(d, 0, n * 8));
    for (size_t i = 0; i < n; i++) k_checksum<<<64, 256, 0, G.stream>>>(G.weights[i].p, G.weights[i].n, d + i);
    std::vector<unsigned long long> h(n);
    CK(cudaMemcpyAsync(h.data(), d, n * 8, cudaMemcpyDeviceToHost, G.stream));
    gpu_sync();
    CK(cudaFree(d));
    return h;
}

void gpu_alloc_scratch(size_t wbuf_elems, size_t arena_bytes) {
    if (G.wbuf) CK(cudaFree(G.wbuf));
    if (G.arena.base) CK(cudaFree(G.arena.base));
    CK(cudaMalloc(&G.wbuf, wbuf_elems * 4));
    G.wbuf_elems = wbuf_elems;
    CK(cudaMalloc(&G.arena.base, arena_bytes));
    G.arena.cap = arena_bytes;
    G.arena.used = G.arena.peak = 0;
}

void gpu_reserve_stage(size_t elems) {
    if (G.stage) CK(cudaFree(G.stage));
    G.stage = nullptr;
    G.stage_elems = 0;
    if (!elems || getenv("KILN_NO_STAGE")) return;  // KILN_NO_STAGE: A/B test, kernels read RAM weights directly
    if (cudaMalloc(&G.stage, elems * 2) != cudaSuccess) { cudaGetLastError(); G.stage = nullptr; return; }
    G.stage_elems = elems;
}

const void* stage_weight(const void* dev, const void* host, size_t bytes) {
    if (!dev && host && (!G.stage || bytes > G.stage_elems * 2)) throw std::runtime_error("a file-mapped weight needs the VRAM stage");
    if (!host || !G.stage || bytes > G.stage_elems * 2) return dev;  // no stage: kernels read the mapped RAM directly
    ProfScope ps("weight_stream");
    CK(cudaMemcpyAsync(G.stage, host, bytes, cudaMemcpyHostToDevice, G.stream));
    return G.stage;
}

size_t gpu_resize_arena(size_t bytes) {
    if (G.arena.used != 0 || bytes == G.arena.cap) return G.arena.cap;
    size_t old = G.arena.cap;
    CK(cudaFree(G.arena.base));
    G.arena.base = nullptr;
    if (cudaMalloc(&G.arena.base, bytes) != cudaSuccess) {
        cudaGetLastError();
        CK(cudaMalloc(&G.arena.base, old));
        bytes = old;
    }
    G.arena.cap = bytes;
    G.arena.peak = 0;
    return bytes;
}

float* Arena::f(size_t n) {
    size_t bytes = (n * 4 + 255) & ~(size_t)255;
    if (used + bytes > cap)
        throw std::runtime_error("out of scratch VRAM: need " + std::to_string((used + bytes) >> 20) + " MB, arena is " + std::to_string(cap >> 20) + " MB");
    float* p = (float*)(base + used);
    used += bytes;
    peak = std::max(peak, used);
    return p;
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0;
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static void fill_bf16(std::vector<uint16_t>& out, const StTensor& t, bool temporal_last) {
    if (!temporal_last) {
        int64_t n = t.numel();
        out.resize(n);
        if (t.dtype == DType::BF16) memcpy(out.data(), t.data, n * 2);
        else for (int64_t i = 0; i < n; i++) out[i] = f32_to_bf16(st_elem_f32(t, i));
        return;
    }
    int64_t co = t.shape[0], ci = t.shape[1], kt = t.shape[2], kh = t.shape[3], kw = t.shape[4];
    out.resize(co * ci * kh * kw);
    for (int64_t o = 0; o < co * ci; o++)
        for (int64_t s = 0; s < kh * kw; s++) {
            int64_t src = (o * kt + (kt - 1)) * kh * kw + s;
            out[o * kh * kw + s] = t.dtype == DType::BF16 ? ((const uint16_t*)t.data)[src] : f32_to_bf16(st_elem_f32(t, src));
        }
}

static void weight_shape(Weight& w, const StTensor& t, bool temporal_last) {
    if (t.shape.empty()) { w.rows = 1; w.cols = 1; return; }
    w.rows = t.shape[0];
    int64_t rest = 1;
    for (size_t i = 1; i < t.shape.size(); i++) if (!(temporal_last && i == 2)) rest *= t.shape[i];
    w.cols = rest;
}

Weight upload_weight(const StTensor& t, Place place, bool temporal_last, bool f16) {
    Weight w;
    weight_shape(w, t, temporal_last);
    w.f16 = f16;
    size_t bytes = (size_t)w.numel() * 2;
    bool device = place == Place::Device || (place == Place::Auto && gpu_free_bytes() > bytes + G.reserve_bytes);
    // Sharing the card (--vram-budget): big matrices that don't fit in VRAM stream straight from the model
    // file's mapping, which Windows can page and share with its file cache, instead of a locked copy in RAM
    // (locked copies next to another app's model ran a 16 GB PC out of memory). linear() stages them.
    if (!device && G.leave_free && G.file_map_ok && !G.no_file_map && !f16 && !temporal_last && t.dtype == DType::BF16 &&
        w.rows > 1 && w.cols > 1 && bytes >= ((size_t)256 << 10) && bytes <= ((size_t)64 << 20)) {
        w.host = const_cast<void*>(static_cast<const void*>(t.data));
        w.on_host = true;
        w.file_mapped = true;
        return w;
    }
    if (device) {
        CK(cudaMalloc(&w.p, bytes));
    } else {
        void* h;
        CK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        void* d;
        CK(cudaHostGetDevicePointer(&d, h, 0));
        w.p = (uint16_t*)d;
        w.host = h;
        w.on_host = true;
    }
    std::vector<uint16_t> tmp;
    const void* src = t.data;
    if (f16) {
        if (temporal_last) throw std::runtime_error("upload_weight: fp16 storage of temporal kernels is not supported");
        if (t.dtype != DType::F16) {
            tmp.resize(w.numel());
            for (int64_t i = 0; i < w.numel(); i++) tmp[i] = __half_as_ushort(__float2half_rn(st_elem_f32(t, i)));
            src = tmp.data();
        }
    } else if (t.dtype != DType::BF16 || temporal_last) {
        fill_bf16(tmp, t, temporal_last);
        src = tmp.data();
    }
    CK(cudaMemcpy(w.p, src, bytes, cudaMemcpyHostToDevice));
    G.weights.push_back({w.p, (size_t)w.numel()});
    return w;
}

float* upload_f32(const StTensor& t) {
    std::vector<float> h(t.numel());
    for (int64_t i = 0; i < t.numel(); i++) h[i] = st_elem_f32(t, i);
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool promote_weight(Weight& w, size_t margin) {
    if (!w.p || !w.on_host) return false;
    size_t bytes = (size_t)w.numel() * 2;
    if (gpu_free_bytes() < bytes + margin) return false;
    uint16_t* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) { cudaGetLastError(); return false; }
    CK(cudaMemcpy(d, w.host, bytes, cudaMemcpyHostToDevice));
    for (auto& r : G.weights) if (r.p == w.p) r.p = d;
    CK(cudaFreeHost(w.host));
    w.p = d;
    w.host = nullptr;
    w.on_host = false;
    return true;
}

void free_weight(Weight& w) {
    if (!w.p) return;
    for (size_t i = 0; i < G.weights.size(); i++)
        if (G.weights[i].p == w.p) { G.weights.erase(G.weights.begin() + i); break; }
    if (w.on_host) CK(cudaFreeHost(w.host));
    else CK(cudaFree(w.p));
    w.p = nullptr;
    w.host = nullptr;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bfv(const uint16_t* p, size_t i) { return __uint_as_float(((uint32_t)p[i]) << 16); }

static inline unsigned nblk(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// Sum across a block of up to 1024 threads; all threads get the result.
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ float block_max(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = warp_max(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// ---------------------------------------------------------------------------
// dense
// ---------------------------------------------------------------------------
__global__ void k_bf16_to_f32(float* __restrict__ y, const uint16_t* __restrict__ x, size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        uint4 v = *(const uint4*)(x + i);
        float4 a = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u), __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
        float4 b = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u), __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
        *(float4*)(y + i) = a;
        *(float4*)(y + i + 4) = b;
    } else {
        for (; i < n; i++) y[i] = bfv(x, i);
    }
}

__global__ void k_f16_to_f32(float* __restrict__ y, const __half* __restrict__ x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

float* weight_f32(const Weight& w) {
    size_t n = w.numel();
    if (n > G.wbuf_elems) throw std::runtime_error("weight staging buffer too small");
    if (w.f16) k_f16_to_f32<<<nblk(n), 256, 0, G.stream>>>(G.wbuf, (const __half*)w.p, n);
    else k_bf16_to_f32<<<nblk((n + 7) / 8), 256, 0, G.stream>>>(G.wbuf, w.p, n);
    return G.wbuf;
}

void gemm(bool ta, bool tb, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    CB(cublasSgemm(G.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

static void lora_side(float* y, const float* x, int T, const Weight& W) {
    ProfScope ps("lora");
    for (auto& l : *W.lora) {
        int r = (int)l.A.rows;
        size_t m = G.arena.mark();
        float* t = G.arena.f((size_t)T * r);
        gemm(false, true, T, r, (int)W.cols, 1.f, x, (int)W.cols, weight_f32(l.A), (int)W.cols, 0.f, t, r);
        gemm(false, true, T, (int)W.rows, r, l.scale, t, r, weight_f32(l.B), r, 1.f, y, (int)W.rows);
        G.arena.release(m);
    }
}

void linear(float* y, const float* x, int T, const Weight& Win, const Weight* bias, float beta) {
    Weight W = Win;
    if (W.on_host) W.p = (uint16_t*)stage_weight(W.p, W.host, (size_t)W.numel() * 2);
    if (G.fp16 && G.tc && tc::eligible(T, (int)W.rows, (int)W.cols)) {
        ProfScope ps("gemm_tc");
        tc::launch(y, x, W.p, T, (int)W.rows, (int)W.cols, beta, G.stream, W.f16);
    } else if (G.fp16 && hg::eligible(T, (int)W.rows, (int)W.cols)) {
        ProfScope ps("gemm_fp16");
        hg::launch(y, x, W.p, T, (int)W.rows, (int)W.cols, beta, G.stream, 1, W.f16);
    } else {
        ProfScope ps("gemm_fp32");
        const float* wf = weight_f32(W);
        gemm(false, true, T, (int)W.rows, (int)W.cols, 1.f, x, (int)W.cols, wf, (int)W.cols, beta, y, (int)W.rows);
    }
    if (Win.lora) lora_side(y, x, T, Win);
    if (bias) add_bias(y, *bias, T, (int)W.rows);
}

__global__ void k_matvec(float* y, const uint16_t* W, const float* x, const float* add, int rows, int cols) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint16_t* w = W + (size_t)row * cols;
    float s = 0.f;
    for (int i = lane; i < cols; i += 32) s += bfv(w, i) * x[i];
    s = warp_sum(s);
    if (lane == 0) y[row] = s + (add ? add[row] : 0.f);
}

__global__ void k_scale(float* y, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

void matvec(float* y, const Weight& W, const float* x, const float* add) {
    ProfScope ps("matvec");
    int rows = (int)W.rows;
    const uint16_t* wp = W.on_host ? (const uint16_t*)stage_weight(W.p, W.host, (size_t)W.numel() * 2) : W.p;
    k_matvec<<<nblk(rows, 8), 256, 0, G.stream>>>(y, wp, x, add, rows, (int)W.cols);
    if (!W.lora) return;
    for (auto& l : *W.lora) {
        int r = (int)l.A.rows;
        size_t m = G.arena.mark();
        float* t = G.arena.f(r);
        k_matvec<<<nblk(r, 8), 256, 0, G.stream>>>(t, l.A.p, x, nullptr, r, (int)l.A.cols);
        k_scale<<<nblk(r), 256, 0, G.stream>>>(t, l.scale, r);
        k_matvec<<<nblk(rows, 8), 256, 0, G.stream>>>(y, l.B.p, t, y, rows, r);
        G.arena.release(m);
    }
}

// ---------------------------------------------------------------------------
// norms / elementwise
// ---------------------------------------------------------------------------
__global__ void k_rmsnorm_block(float* y, const float* x, const uint16_t* w, int dim, float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* yr = y + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    float r = rsqrtf(block_sum(s) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = xr[i] * r * (w ? bfv(w, i) : 1.f);
}

__global__ void k_rmsnorm_warp(float* y, const float* x, const uint16_t* w, int rows, int dim, float eps) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;
    float s = 0.f;
    for (int i = lane; i < dim; i += 32) s += xr[i] * xr[i];
    float r = rsqrtf(warp_sum(s) / dim + eps);
    for (int i = lane; i < dim; i += 32) yr[i] = xr[i] * r * (w ? bfv(w, i) : 1.f);
}

void rmsnorm(float* y, const float* x, const Weight* w, int rows, int dim, float eps) {
    ProfScope ps("norm");
    const uint16_t* wp = w ? w->p : nullptr;
    if (dim <= 256) k_rmsnorm_warp<<<nblk(rows, 8), 256, 0, G.stream>>>(y, x, wp, rows, dim, eps);
    else k_rmsnorm_block<<<rows, 256, 0, G.stream>>>(y, x, wp, dim, eps);
}

__global__ void k_layernorm_mod(float* y, const float* x, const float* scale, const float* shift, int dim, float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* yr = y + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    float mean = block_sum(s) / dim;
    float v = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = xr[i] - mean; v += d * d; }
    float r = rsqrtf(block_sum(v) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * r * (1.f + scale[i]) + shift[i];
}

void layernorm_mod(float* y, const float* x, const float* scale, const float* shift, int rows, int dim, float eps) {
    ProfScope ps("norm");
    k_layernorm_mod<<<rows, 256, 0, G.stream>>>(y, x, scale, shift, dim, eps);
}

__global__ void k_add_gated(float* x, const float* y, const float* gate, size_t n, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += (gate ? gate[i % dim] : 1.f) * y[i];
}
void add_gated(float* x, const float* y, const float* gate, int rows, int dim) {
    ProfScope ps("elementwise");
    size_t n = (size_t)rows * dim;
    k_add_gated<<<nblk(n), 256, 0, G.stream>>>(x, y, gate, n, dim);
}

__global__ void k_add_bias(float* y, const uint16_t* b, size_t n, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += bfv(b, i % dim);
}
void add_bias(float* y, const Weight& b, int rows, int dim) {
    size_t n = (size_t)rows * dim;
    k_add_bias<<<nblk(n), 256, 0, G.stream>>>(y, b.p, n, dim);
}

__global__ void k_scale_rows(float* y, const float* s, size_t n, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s[i / dim];
}
void scale_rows(float* y, const float* s, int rows, int dim) {
    size_t n = (size_t)rows * dim;
    k_scale_rows<<<nblk(n), 256, 0, G.stream>>>(y, s, n, dim);
}

__global__ void k_gelu(float* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }
}
void gelu(float* x, size_t n) {
    ProfScope ps("elementwise"); k_gelu<<<nblk(n), 256, 0, G.stream>>>(x, n); }

__global__ void k_silu(float* y, const float* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.f + expf(-v)); }
}
void silu(float* y, const float* x, size_t n) { k_silu<<<nblk(n), 256, 0, G.stream>>>(y, x, n); }

__global__ void k_silu_mul(float* g, const float* u, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = g[i]; g[i] = v / (1.f + expf(-v)) * u[i]; }
}
void silu_mul(float* g, const float* u, size_t n) { k_silu_mul<<<nblk(n), 256, 0, G.stream>>>(g, u, n); }

__global__ void k_axpy(float* y, const float* x, float a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
void axpy(float* y, const float* x, float a, size_t n) { k_axpy<<<nblk(n), 256, 0, G.stream>>>(y, x, a, n); }

__global__ void k_fill(float* y, float v, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = v;
}
void fill(float* y, float v, size_t n) { if (n) k_fill<<<nblk(n), 256, 0, G.stream>>>(y, v, n); }

__global__ void k_rope_half(float* x, int T, int H, int Dh, const float* cos_t, const float* sin_t) {
    int half = Dh / 2;
    size_t n = (size_t)T * H * half;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int i = idx % half;
    size_t th = idx / half;
    int t = (int)(th / H);
    float* p = x + th * Dh;
    float c = cos_t[(size_t)t * half + i], s = sin_t[(size_t)t * half + i];
    float a = p[i], b = p[i + half];
    p[i] = a * c - b * s;
    p[i + half] = b * c + a * s;
}
void rope_half(float* x, int T, int H, int Dh, const float* cos_t, const float* sin_t) {
    ProfScope ps("rope");
    size_t n = (size_t)T * H * (Dh / 2);
    k_rope_half<<<nblk(n), 256, 0, G.stream>>>(x, T, H, Dh, cos_t, sin_t);
}

__global__ void k_embed(float* y, const uint16_t* table, const int* ids, int n, int dim, bool f16) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * dim) return;
    int t = (int)(i / dim), d = (int)(i % dim);
    size_t k = (size_t)ids[t] * dim + d;
    y[i] = f16 ? __half2float(__ushort_as_half(table[k])) : bfv(table, k);
}
void embed_rows(float* y, const Weight& table, const int* ids, int n) {
    int dim = (int)table.cols;
    k_embed<<<nblk((size_t)n * dim), 256, 0, G.stream>>>(y, table.p, ids, n, dim, table.f16);
}

__global__ void k_repeat_kv(float* dst, const float* src, int T, int Hkv, int group, int Dh) {
    size_t n = (size_t)T * Hkv * group * Dh;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % Dh;
    size_t r = i / Dh;
    int h = (int)(r % (Hkv * group));
    int t = (int)(r / (Hkv * group));
    dst[i] = src[((size_t)t * Hkv + h / group) * Dh + d];
}
void repeat_kv(float* dst, const float* src, int T, int Hkv, int group, int Dh) {
    size_t n = (size_t)T * Hkv * group * Dh;
    k_repeat_kv<<<nblk(n), 256, 0, G.stream>>>(dst, src, T, Hkv, group, Dh);
}

// ---------------------------------------------------------------------------
// attention: scores = q k^T * scale -> softmax -> @ v, heads batched through cuBLAS
// ---------------------------------------------------------------------------
__global__ void k_softmax_rows(float* s, int cols, int rows_per_head, bool causal, int npad, int q0) {
    float* r = s + (size_t)blockIdx.x * cols;
    int qi = q0 + blockIdx.x % rows_per_head;
    int lim = causal ? qi + 1 : cols;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < lim; i += blockDim.x) m = fmaxf(m, r[i]);
    m = block_max(m);
    if (npad) m = fmaxf(m, 0.f);
    // (only with padding: m is then >= 0; without, m can be < -88 and expf(-m) = inf, 0 * inf = NaN)
    float sum = threadIdx.x == 0 && npad ? (float)npad * expf(-m) : 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = i < lim ? expf(r[i] - m) : 0.f;
        r[i] = e;
        sum += e;
    }
    float inv = 1.f / block_sum(sum);
    for (int i = threadIdx.x; i < lim; i += blockDim.x) r[i] *= inv;
}

void attention(float* out, int ldo, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
               int Tq, int Tk, int H, int Dh, bool causal, int npad) {
    if (G.fp16 && G.tc && !causal && tfa::eligible(Dh, ldq, ldk, ldv, ldo)) {
        ProfScope ps("attention");
        tfa::launch(out, ldo, q, ldq, k, ldk, v, ldv, Tq, Tk, H, npad, G.stream);
        return;
    }
    if (G.fp16 && !causal && fa::eligible(Dh, ldq, ldk, ldv, ldo)) {
        ProfScope ps("attention");
        fa::launch(out, ldo, q, ldq, k, ldk, v, ldv, Tq, Tk, H, npad, G.stream);
        return;
    }
    // Score buffer bounded by the arena: batch several heads when a whole head fits,
    // otherwise walk one head in blocks of query rows (the VAE's single 15k-token head).
    size_t per_head = (size_t)Tq * Tk;
    size_t budget = std::min<size_t>(G.arena.free_bytes() / 2, (size_t)512 << 20);
    int hc = 1, qc = Tq;
    if (per_head * 4 <= budget) hc = (int)std::min<size_t>(H, budget / (per_head * 4));
    else qc = (int)std::max<size_t>(1, budget / ((size_t)Tk * 4));
    size_t m = G.arena.mark();
    float* S = G.arena.f((size_t)hc * qc * Tk);
    const float scale = 1.f / sqrtf((float)Dh), one = 1.f, zero = 0.f;
    ProfScope ps("attention");
    for (int h0 = 0; h0 < H; h0 += hc) {
        int nh = std::min(hc, H - h0);
        for (int q0 = 0; q0 < Tq; q0 += qc) {
            int nq = std::min(qc, Tq - q0);
            size_t stride = (size_t)nq * Tk;
            // S_h[nq,Tk] = Q_h K_h^T  (column-major: S^T = K Q^T)
            CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, Tk, nq, Dh, &scale,
                                         k + (size_t)h0 * Dh, ldk, Dh, q + (size_t)q0 * ldq + (size_t)h0 * Dh, ldq, Dh, &zero, S, Tk, stride, nh));
            k_softmax_rows<<<(unsigned)(nq * nh), 256, 0, G.stream>>>(S, Tk, nq, causal, npad, q0);
            // O_h[nq,Dh] = P_h V_h  (column-major: O^T = V^T P^T)
            CB(cublasSgemmStridedBatched(G.blas, CUBLAS_OP_N, CUBLAS_OP_N, Dh, nq, Tk, &one,
                                         v + (size_t)h0 * Dh, ldv, Dh, S, Tk, stride, &zero, out + (size_t)q0 * ldo + (size_t)h0 * Dh, ldo, Dh, nh));
        }
    }
    G.arena.release(m);
}

// ---------------------------------------------------------------------------
// image ops
// ---------------------------------------------------------------------------
// Column matrix for a 3x3 conv (pad 1) over output pixels [p0, p0+Pc) of an H x W output.
// With up = 2 the input is H/2 x W/2 and is nearest-upsampled on the fly.
__global__ void k_im2col3(float* col, const float* in, int C, int H, int W, int p0, int Pc, int up) {
    size_t n = (size_t)C * 9 * Pc;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = (int)(i % Pc);
    size_t r = i / Pc;          // c*9 + ky*3 + kx
    int kx = r % 3, ky = (r / 3) % 3;
    int c = (int)(r / 9);
    int p = p0 + j;
    int y = p / W + ky - 1, x = p % W + kx - 1;
    int Hi = H / up, Wi = W / up;
    col[i] = (y >= 0 && y < H && x >= 0 && x < W) ? in[((size_t)c * Hi + y / up) * Wi + x / up] : 0.f;
}

static void conv3x3(float* out, const float* in, int Cin, int H, int W, const Weight& w, int up) {
    int Cout = (int)w.rows, P = H * W, K = Cin * 9;
    const float* wf = weight_f32(w);
    size_t m = G.arena.mark();
    size_t budget = std::min<size_t>(G.arena.free_bytes() - 1024, (size_t)96 << 20);
    int Pc = (int)std::min<size_t>(P, std::max<size_t>(W, budget / ((size_t)K * 4)));
    float* col = G.arena.f((size_t)K * Pc);
    for (int p0 = 0; p0 < P; p0 += Pc) {
        int n = std::min(Pc, P - p0);
        k_im2col3<<<nblk((size_t)K * n), 256, 0, G.stream>>>(col, in, Cin, H, W, p0, n, up);
        gemm(false, false, Cout, n, K, 1.f, wf, K, col, n, 0.f, out + p0, P);
    }
    G.arena.release(m);
}

__global__ void k_im2col3_s2(float* col, const float* in, int C, int H, int W, int p0, int Pc) {
    int Wo = W / 2;
    size_t n = (size_t)C * 9 * Pc;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = (int)(i % Pc);
    size_t r = i / Pc;
    int kx = r % 3, ky = (r / 3) % 3;
    int c = (int)(r / 9);
    int p = p0 + j;
    int y = (p / Wo) * 2 + ky, x = (p % Wo) * 2 + kx;
    col[i] = (y < H && x < W) ? in[((size_t)c * H + y) * W + x] : 0.f;
}

void conv2d_s2(float* out, const float* in, int Cin, int H, int W, const Weight& w, const Weight* b) {
    int Cout = (int)w.rows, Wo = W / 2, P = (H / 2) * Wo, K = Cin * 9;
    const float* wf = weight_f32(w);
    size_t m = G.arena.mark();
    size_t budget = std::min<size_t>(G.arena.free_bytes() - 1024, (size_t)96 << 20);
    int Pc = (int)std::min<size_t>(P, std::max<size_t>(Wo, budget / ((size_t)K * 4)));
    float* col = G.arena.f((size_t)K * Pc);
    for (int p0 = 0; p0 < P; p0 += Pc) {
        int n = std::min(Pc, P - p0);
        k_im2col3_s2<<<nblk((size_t)K * n), 256, 0, G.stream>>>(col, in, Cin, H, W, p0, n);
        gemm(false, false, Cout, n, K, 1.f, wf, K, col, n, 0.f, out + p0, P);
    }
    G.arena.release(m);
    if (b) add_bias_channels(out, *b, Cout, P);
}

// Lanczos-3 along one axis. in is [C, rows, n_in] (resize along the last axis) written transposed
// to out [C, n_out, rows], so two passes resize both axes and restore the layout.
__device__ __forceinline__ float lanczos3(float x) {
    x = fabsf(x);
    if (x < 1e-6f) return 1.f;
    if (x >= 3.f) return 0.f;
    const float pi = 3.14159265358979f;
    return 3.f * sinf(pi * x) * sinf(pi * x / 3.f) / (pi * pi * x * x);
}

__global__ void k_resize_axis(float* out, const float* in, int C, int rows, int n_in, int n_out) {
    size_t n = (size_t)C * n_out * rows;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i % rows;
    size_t t = i / rows;
    int o = t % n_out;
    int c = (int)(t / n_out);
    float scale = (float)n_in / n_out;
    float support = scale > 1.f ? scale : 1.f;             // widen the filter when shrinking
    float center = (o + 0.5f) * scale - 0.5f;
    int lo = (int)floorf(center - 3.f * support), hi = (int)ceilf(center + 3.f * support);
    const float* src = in + ((size_t)c * rows + r) * n_in;
    float acc = 0.f, wsum = 0.f;
    for (int k = lo; k <= hi; k++) {
        float wgt = lanczos3((k - center) / support);
        if (wgt == 0.f) continue;
        int kk = min(max(k, 0), n_in - 1);                  // clamp at the edges
        acc += wgt * src[kk];
        wsum += wgt;
    }
    out[i] = acc / wsum;
}

void resize_lanczos(float* out, const float* in, int C, int Hi, int Wi, int Ho, int Wo) {
    size_t m = G.arena.mark();
    float* tmp = G.arena.f((size_t)C * Wo * Hi);
    size_t n1 = (size_t)C * Wo * Hi, n2 = (size_t)C * Ho * Wo;
    k_resize_axis<<<nblk(n1), 256, 0, G.stream>>>(tmp, in, C, Hi, Wi, Wo);   // [C,Hi,Wi] -> [C,Wo,Hi]
    k_resize_axis<<<nblk(n2), 256, 0, G.stream>>>(out, tmp, C, Wo, Hi, Ho);  // [C,Wo,Hi] -> [C,Ho,Wo]
    G.arena.release(m);
}

void conv2d(float* out, const float* in, int Cin, int H, int W, const Weight& w, const Weight* b, int ksize) {
    int Cout = (int)w.rows, P = H * W;
    if (ksize == 1) gemm(false, false, Cout, P, Cin, 1.f, weight_f32(w), Cin, in, P, 0.f, out, P);
    else conv3x3(out, in, Cin, H, W, w, 1);
    if (b) add_bias_channels(out, *b, Cout, P);
}

void conv2d_up2(float* out, const float* in, int Cin, int H, int W, const Weight& w, const Weight* b) {
    conv3x3(out, in, Cin, 2 * H, 2 * W, w, 2);
    if (b) add_bias_channels(out, *b, (int)w.rows, 4 * H * W);
}

__global__ void k_rms_channels(float* y, const float* x, const uint16_t* g, int C, int P, bool apply_silu) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float s = 0.f;
    for (int c = 0; c < C; c++) { float v = x[(size_t)c * P + p]; s += v * v; }
    float r = sqrtf((float)C) / fmaxf(sqrtf(s), 1e-12f);
    for (int c = 0; c < C; c++) {
        float v = x[(size_t)c * P + p] * r * bfv(g, c);
        if (apply_silu) v = v / (1.f + expf(-v));
        y[(size_t)c * P + p] = v;
    }
}
void rms_channels(float* y, const float* x, const Weight& gamma, int C, int P, bool apply_silu) {
    k_rms_channels<<<nblk(P), 256, 0, G.stream>>>(y, x, gamma.p, C, P, apply_silu);
}

__global__ void k_upsample2x(float* y, const float* x, int C, int H, int W) {
    size_t n = (size_t)C * H * 2 * W * 2;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xo = i % (2 * W);
    size_t r = i / (2 * W);
    int yo = r % (2 * H);
    int c = (int)(r / (2 * H));
    y[i] = x[((size_t)c * H + yo / 2) * W + xo / 2];
}
void upsample2x(float* y, const float* x, int C, int H, int W) {
    size_t n = (size_t)C * H * W * 4;
    k_upsample2x<<<nblk(n), 256, 0, G.stream>>>(y, x, C, H, W);
}

__global__ void k_transpose(float* y, const float* x, int R, int C) {
    __shared__ float tile[32][33];
    int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += 8) {
        int r = by + j, c = bx + threadIdx.x;
        if (r < R && c < C) tile[j][threadIdx.x] = x[(size_t)r * C + c];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += 8) {
        int c = bx + j, r = by + threadIdx.x;
        if (r < R && c < C) y[(size_t)c * R + r] = tile[threadIdx.x][j];
    }
}
void transpose(float* y, const float* x, int R, int C) {
    dim3 grid((C + 31) / 32, (R + 31) / 32);
    k_transpose<<<grid, dim3(32, 8), 0, G.stream>>>(y, x, R, C);
}

__global__ void k_add(float* y, const float* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
void add_inplace(float* y, const float* x, size_t n) { k_add<<<nblk(n), 256, 0, G.stream>>>(y, x, n); }

__global__ void k_bias_ch(float* y, const uint16_t* b, int C, int P) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)C * P) y[i] += bfv(b, i / P);
}
void add_bias_channels(float* y, const Weight& b, int C, int P) {
    k_bias_ch<<<nblk((size_t)C * P), 256, 0, G.stream>>>(y, b.p, C, P);
}


__global__ void k_cfg(float* v, const float* vn, float cfg, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = vn[i] + cfg * (v[i] - vn[i]);
}
void cfg_combine(float* v, const float* vn, float cfg, size_t n) { k_cfg<<<nblk(n), 256, 0, G.stream>>>(v, vn, cfg, n); }

// ---------------------------------------------------------------------------
// general resampling (ComfyUI's ImageScale / LatentUpscale methods, PyTorch interpolate rules)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float cubic_w(float x) {  // a = -0.75, like torch bicubic
    const float a = -0.75f;
    x = fabsf(x);
    if (x <= 1.f) return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f) return ((a * x - 5.f * a) * x + 8.f * a) * x - 4.f * a;
    return 0.f;
}

// one axis; in [C, rows, n_in] -> out [C, n_out, rows] (transposed, see resize_lanczos)
// mode: 0 nearest-exact, 1 bilinear, 2 bicubic, 3 area
__global__ void k_resize_axis_mode(float* out, const float* in, int C, int rows, int n_in, int n_out, int mode) {
    size_t n = (size_t)C * n_out * rows;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i % rows;
    size_t t = i / rows;
    int o = t % n_out;
    int c = (int)(t / n_out);
    const float* src = in + ((size_t)c * rows + r) * n_in;
    float scale = (float)n_in / n_out, v = 0.f;
    if (mode == 0) {
        v = src[min((int)floorf((o + 0.5f) * scale), n_in - 1)];
    } else if (mode == 1) {
        float x = fmaxf((o + 0.5f) * scale - 0.5f, 0.f);
        int x0 = min((int)x, n_in - 1), x1 = min(x0 + 1, n_in - 1);
        float l = x - x0;
        v = src[x0] * (1.f - l) + src[x1] * l;
    } else if (mode == 2) {
        float x = (o + 0.5f) * scale - 0.5f;
        int x0 = (int)floorf(x);
        float tt = x - x0;
        for (int k = -1; k <= 2; k++) v += cubic_w(tt - k) * src[min(max(x0 + k, 0), n_in - 1)];
    } else {
        int s0 = (int)floorf((float)o * n_in / n_out), s1 = (int)ceilf((float)(o + 1) * n_in / n_out);
        for (int k = s0; k < s1; k++) v += src[k];
        v /= (float)(s1 - s0);
    }
    out[i] = v;
}

void resize_mode(float* out, const float* in, int C, int Hi, int Wi, int Ho, int Wo, const std::string& method) {
    if (method == "lanczos") { resize_lanczos(out, in, C, Hi, Wi, Ho, Wo); return; }
    int mode = method == "nearest-exact" || method == "nearest" ? 0 : method == "bicubic" ? 2 : method == "area" ? 3 : 1;
    size_t m = G.arena.mark();
    float* tmp = G.arena.f((size_t)C * Wo * Hi);
    size_t n1 = (size_t)C * Wo * Hi, n2 = (size_t)C * Ho * Wo;
    k_resize_axis_mode<<<nblk(n1), 256, 0, G.stream>>>(tmp, in, C, Hi, Wi, Wo, mode);
    k_resize_axis_mode<<<nblk(n2), 256, 0, G.stream>>>(out, tmp, C, Wo, Hi, Ho, mode);
    G.arena.release(m);
}
