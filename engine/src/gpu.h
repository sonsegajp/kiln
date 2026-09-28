// Core GPU plumbing: weights, scratch arena, and the op set every model is built from.
// Convention: activations are fp32, row-major. Weights are bf16 in PyTorch layout [out, in].
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "safetensors.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string((int)s_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

// bf16 tensor readable by kernels. It lives in VRAM, or in pinned host RAM that the GPU
// reads over PCIe when VRAM is short (the conversion kernel streams it either way).
struct LoraAdd;

struct Weight {
    uint16_t* p = nullptr;
    int64_t rows = 0, cols = 0;   // conv weights: rows = Cout, cols = Cin*kh*kw
    bool f16 = false;             // p holds fp16 instead of bf16 (fp16 checkpoints such as SDXL keep their precision)
    bool on_host = false;
    void* host = nullptr;  // host address of a mapped allocation (not always equal to p on WDDM)
    bool file_mapped = false;  // host points into the model file's mapping (pageable, no device pointer):
                               // only linear() may use it, through the stage
    // LoRAs run unmerged as a low-rank side path: y += scale * B(A x). Merging is not an option
    // for bf16 weights: typical deltas (~1e-5) are below bf16's step (~8e-5 at |w| ~ 0.02).
    std::vector<LoraAdd>* lora = nullptr;
    int64_t numel() const { return rows * cols; }
    explicit operator bool() const { return p != nullptr || host != nullptr; }
};

struct LoraAdd { Weight A, B; float scale; };  // A [r, in], B [out, r]

enum class Place { Auto, Device, Host };

// Uploads a tensor as bf16, or as fp16 with `f16`. `temporal_last` keeps only the last temporal tap of a
// [Cout, Cin, T, kh, kw] conv3d kernel: a causal conv on a single frame sees only that tap.
Weight upload_weight(const StTensor& t, Place place, bool temporal_last = false, bool f16 = false);
// A small tensor (bias, norm scale) as fp32 in VRAM, for kernels that read parameters in fp32.
float* upload_f32(const StTensor& t);
void free_weight(Weight& w);
// Moves a weight that had to live in pinned system RAM into VRAM if at least `margin` bytes would
// remain free afterwards. Returns true if it moved.
bool promote_weight(Weight& w, size_t margin);

struct Arena {
    char* base = nullptr;
    size_t cap = 0, used = 0, peak = 0;
    float* f(size_t n);
    int* i(size_t n) { return (int*)f(n); }
    size_t mark() const { return used; }
    void release(size_t m) { used = m; }
    size_t free_bytes() const { return cap - used; }
};

struct WeightRec { const uint16_t* p; size_t n; };

struct Gpu {
    std::vector<WeightRec> weights;  // every live upload, for integrity checks
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    Arena arena;
    float* wbuf = nullptr;        // fp32 staging for one weight matrix
    // Weights that did not fit in VRAM live in mapped system RAM. Kernels would re-read them over PCIe for
    // every tile, so each use copies the weight here first (stream-ordered: one weight at a time).
    uint16_t* stage = nullptr;
    size_t stage_elems = 0;
    size_t wbuf_elems = 0;
    size_t reserve_bytes = 0;     // VRAM that Auto placement must leave free
    size_t leave_free = 0;        // VRAM that belongs to another app (--vram-budget): Kiln never grows into it
    bool file_map_ok = false;     // set only while the DiT / text encoder load (their matrices go through linear/matvec)
    bool no_file_map = false;     // the loader marks weights read outside linear() (embedding tables)
    bool fp16 = true;             // fp16x2 GEMMs (fast path); false = exact fp32 cuBLAS everywhere
};
extern Gpu G;

// Per-category GPU timing (--profile). Scopes record event pairs; prof_report syncs and sums them.
struct ProfScope {
    int slot = -1;
    explicit ProfScope(const char* name);
    ~ProfScope();
};
void prof_enable(bool on);
std::string prof_report();

void gpu_init(int device);
void gpu_alloc_scratch(size_t wbuf_elems, size_t arena_bytes);
// Reallocates the arena (must be empty) at `bytes`; keeps the old size if that fails. Returns the new capacity.
size_t gpu_resize_arena(size_t bytes);
// the RAM-weight staging buffer (fp16/bf16 elements); 0 frees it
void gpu_reserve_stage(size_t elems);
// device pointer to use for a weight living at `host` (mapped RAM): copied into the stage when it fits
const void* stage_weight(const void* dev, const void* host, size_t bytes);
size_t gpu_free_bytes();
void gpu_sync();
std::vector<unsigned long long> weight_checksums();  // one per G.weights entry

// ---- dense ----
float* weight_f32(const Weight& w);  // converted copy in G.wbuf, valid until the next call
// Row-major C[M,N] = alpha*op(A)[M,K]*op(B)[K,N] + beta*C
void gemm(bool ta, bool tb, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
// y[T, W.rows] = x[T, W.cols] * W^T (+ bias) (+ beta*y)
void linear(float* y, const float* x, int T, const Weight& W, const Weight* bias = nullptr, float beta = 0.f);
// y = W x (+ add), single vector, W read straight from bf16
void matvec(float* y, const Weight& W, const float* x, const float* add = nullptr);

// ---- elementwise / norms ----
void rmsnorm(float* y, const float* x, const Weight* w, int rows, int dim, float eps);
void layernorm_mod(float* y, const float* x, const float* scale, const float* shift, int rows, int dim, float eps);
void add_gated(float* x, const float* y, const float* gate, int rows, int dim);  // x += gate*y (gate may be null)
void add_bias(float* y, const Weight& b, int rows, int dim);
void scale_rows(float* y, const float* s, int rows, int dim);                    // y[r,:] *= s[r]
void gelu(float* x, size_t n);
void silu(float* y, const float* x, size_t n);
void silu_mul(float* g, const float* u, size_t n);                              // g = silu(g)*u
void axpy(float* y, const float* x, float a, size_t n);                          // y += a*x
void fill(float* y, float v, size_t n);
void cfg_combine(float* v, const float* vn, float cfg, size_t n);         // v = vn + cfg*(v - vn)
void rope_half(float* x, int T, int H, int Dh, const float* cos_t, const float* sin_t);  // pairs (i, i+Dh/2), tables [T, Dh/2]
void embed_rows(float* y, const Weight& table, const int* ids, int n);
void repeat_kv(float* dst, const float* src, int T, int Hkv, int group, int Dh);

// Multi-head attention. q [Tq, *] with row stride ldq, head h at column h*Dh; same for k, v, out.
// npad extra keys that are exactly zero (K = 0, V = 0) are accounted for analytically: they add
// npad * exp(0 - max) to the softmax denominator and nothing to the output.
void attention(float* out, int ldo, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
               int Tq, int Tk, int H, int Dh, bool causal, int npad = 0);

// ---- image ops, channel-first [C, H*W] ----
void conv2d(float* out, const float* in, int Cin, int H, int W, const Weight& w, const Weight* b, int ksize);
// nearest 2x upsample of in [Cin, H, W] fused into a 3x3 conv -> out [Cout, 2H, 2W]
void conv2d_up2(float* out, const float* in, int Cin, int H, int W, const Weight& w, const Weight* b);
// 3x3 stride-2 conv with zero padding only on the right/bottom (PyTorch ZeroPad2d((0,1,0,1)) + Conv2d(stride 2)):
// in [Cin, H, W] (H, W even) -> out [Cout, H/2, W/2]
void conv2d_s2(float* out, const float* in, int Cin, int H, int W, const Weight& w, const Weight* b);
// Separable Lanczos-3 resize of [C, Hi, Wi] -> [C, Ho, Wo]; the kernel widens when shrinking (antialiased).
void resize_lanczos(float* out, const float* in, int C, int Hi, int Wi, int Ho, int Wo);
// ComfyUI resize methods: nearest-exact, bilinear, bicubic, area, lanczos (bislerp -> bilinear)
void resize_mode(float* out, const float* in, int C, int Hi, int Wi, int Ho, int Wo, const std::string& method);
void rms_channels(float* y, const float* x, const Weight& gamma, int C, int P, bool apply_silu);
void upsample2x(float* y, const float* x, int C, int H, int W);
void transpose(float* y, const float* x, int R, int C);
void add_inplace(float* y, const float* x, size_t n);
void add_bias_channels(float* y, const Weight& b, int C, int P);

