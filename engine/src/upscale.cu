// RRDBNet (ESRGAN / Real-ESRGAN family) upscaler, e.g. 4x-AnimeSharp.
//
//   feat  = conv_first(x)
//   trunk = nb x RRDB(feat), RRDB(x) = x + 0.2 * RDB3(RDB2(RDB1(x)))
//           RDB(x): x_k = lrelu(conv_k(cat(x, x_1..x_{k-1}))) for k = 1..4, then x + 0.2 * conv5(cat(x, x_1..x_4))
//   feat += conv_body(trunk)
//   n_up x [nearest 2x -> conv -> lrelu], lrelu(conv_hr), conv_last, clamp to [0,1]
//
// Layout: channel-first [C, H*W] fp32 like the rest of the engine. A dense block's concat is one
// [nf + 4gc, P] buffer: each conv writes its output straight into its channel slice and the next
// conv reads the prefix, so nothing is ever copied.
//
// Weights stay fp16 as shipped (one blob, K-major) instead of going through upload_weight's bf16:
// same memory, and bf16 rounding alone costs rel_l2 1.1e-3 on the output (15% of 8-bit values
// move by 1) where the fp16 weights with fp32 math match a PyTorch fp32 reference to 4e-7.
//
// Convs are direct implicit-GEMM 3x3 kernels (no im2col slab), in the precision the engine uses
// for everything else: fp16x2 when G.fp16 (default; ~1.4x faster, rel_l2 ~7e-4 vs fp32, 8-bit
// output within +-1 level) or fp32 (--fp32; matches the fp32 reference to ~4e-7). `reference`
// switches to im2col + cuBLAS SGEMM, the slow exactness baseline.
//
// Tiling: every op is local, but the receptive field is huge: 1 + 23*15 + 1 = 347 LR px through the
// trunk (+ ~1.25 through the upsampling tail), so a halo that makes tiles bit-exact would be larger
// than the images themselves. What bounds the error in practice is the 0.2 residual scaling:
// the influence of far pixels decays fast. Measured (fp32, 256 px image, 64 px tile cores vs one
// untiled pass), max abs error by halo: 8 px 3.5e-2, 16 px 5.4e-3, 24 px 1.2e-3, 32 px 1.7e-4,
// 48 px 6e-6 (fp32 noise). Default 32: 20x below one 8-bit step and below the fp16 path's own
// rounding. When the whole image fits the arena it runs untiled (exact).
// The upsampling tail is exact-banded within a tile: its receptive field is 1.25 LR px, so bands
// with 2 LR rows/cols of halo reproduce the full-tile result bit for bit, and it only ever
// computes the tile's core.
#include "upscale.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

enum { ACT_NONE = 0, ACT_BIAS = 1, ACT_BIAS_LRELU = 2 };
static const int TAIL_HALO = 2;  // LR px; the tail's receptive field is 1.25 LR px (see header)
// Conv kernels, tuned on a GTX 1660 Ti Max-Q (sm_75): K-chunks of 8 input channels; the fp16x2
// kernel flushes its half2 partial sums into fp32 every 4 input channels (36 products), which
// halves its error vs flushing every 8 at no measurable cost (every 2: -4% speed, little gain).
static constexpr int CONV_CK = 8, CONV_FL = 4;

static inline unsigned nb_(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }
__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ float lrelu(float v) { return v < 0.f ? 0.2f * v : v; }
__device__ __forceinline__ float lo_f(uint32_t u) { return __half2float(__ushort_as_half((uint16_t)(u & 0xffff))); }
__device__ __forceinline__ float hi_f(uint32_t u) { return __half2float(__ushort_as_half((uint16_t)(u >> 16))); }

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
static uint16_t f32_to_f16_bits(float f) { __half h = __float2half_rn(f); uint16_t b; memcpy(&b, &h, 2); return b; }
static float bf16_round(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

void Upscaler::load(const std::string& path, Place place) {
    SafeTensors st(path);
    // Two naming schemes: old ESRGAN ("model.0", "model.1.sub.N.RDB1.conv1.0", ...) and
    // BasicSR / Real-ESRGAN ("conv_first", "body.N.rdb1.conv1", "conv_body", "conv_up1", ...).
    struct Name { std::string w, b; };
    Name first, body, hr, last;
    std::vector<Name> ups;
    std::vector<std::vector<Name>> blocks;  // [block][rdb*5 + k]
    auto nm = [](const std::string& p) { return Name{p + ".weight", p + ".bias"}; };
    if (st.has("model.0.weight")) {
        first = nm("model.0");
        for (int i = 0;; i++) {
            std::string p = "model.1.sub." + std::to_string(i);
            if (!st.has(p + ".RDB1.conv1.0.weight")) { body = nm(p); break; }
            std::vector<Name> b;
            for (int r = 1; r <= 3; r++)
                for (int k = 1; k <= 5; k++) b.push_back(nm(p + ".RDB" + std::to_string(r) + ".conv" + std::to_string(k) + ".0"));
            blocks.push_back(b);
        }
        // after the trunk: model.N convs, N >= 2: upconvs..., hr, last (activations/upsamples have no weights)
        std::vector<int> idx;
        for (auto& [k, t] : st.all()) {
            if (k.rfind("model.", 0) != 0 || k.size() < 13 || k.compare(k.size() - 7, 7, ".weight") != 0) continue;
            std::string mid = k.substr(6, k.size() - 13);
            if (mid.find('.') != std::string::npos) continue;
            int n = std::stoi(mid);
            if (n >= 2) idx.push_back(n);
        }
        std::sort(idx.begin(), idx.end());
        if (idx.size() < 2) throw std::runtime_error("upscaler: no hr/last convs in " + path);
        for (size_t i = 0; i + 2 < idx.size(); i++) ups.push_back(nm("model." + std::to_string(idx[i])));
        hr = nm("model." + std::to_string(idx[idx.size() - 2]));
        last = nm("model." + std::to_string(idx.back()));
    } else if (st.has("conv_first.weight")) {
        first = nm("conv_first");
        for (int i = 0; st.has("body." + std::to_string(i) + ".rdb1.conv1.weight"); i++) {
            std::string p = "body." + std::to_string(i);
            std::vector<Name> b;
            for (int r = 1; r <= 3; r++)
                for (int k = 1; k <= 5; k++) b.push_back(nm(p + ".rdb" + std::to_string(r) + ".conv" + std::to_string(k)));
            blocks.push_back(b);
        }
        body = nm("conv_body");
        for (int i = 1; st.has("conv_up" + std::to_string(i) + ".weight"); i++) ups.push_back(nm("conv_up" + std::to_string(i)));
        hr = nm("conv_hr");
        last = nm("conv_last");
    } else {
        throw std::runtime_error("upscaler: not an RRDBNet (ESRGAN) checkpoint: " + path);
    }

    // shapes
    auto shape = [&](const Name& n) -> const std::vector<int64_t>& {
        const StTensor& t = st.get(n.w);
        if (t.shape.size() != 4 || t.shape[2] != 3 || t.shape[3] != 3) throw std::runtime_error("upscaler: " + n.w + " is not a 3x3 conv");
        return t.shape;
    };
    nf = (int)shape(first)[0];
    int in_nc = (int)shape(first)[1];
    if (in_nc != 3) throw std::runtime_error("upscaler: " + std::to_string(in_nc) + "-channel input (pixel-unshuffle x1/x2 variants) is not supported");
    if ((int)shape(last)[0] != 3) throw std::runtime_error("upscaler: output is not RGB");
    nb = (int)blocks.size();
    gc = nb ? (int)shape(blocks[0][0])[0] : 32;
    n_up = (int)ups.size();
    scale_ = 1 << n_up;
    for (auto& b : blocks)
        for (int j = 0; j < 15; j++) {
            auto& s = shape(b[j]);
            int k = j % 5;
            if (s[1] != nf + k * gc || s[0] != (k == 4 ? nf : gc)) throw std::runtime_error("upscaler: unexpected RDB shape at " + b[j].w);
        }

    // one fp16 blob for everything
    std::vector<const Name*> all = {&first, &body, &hr, &last};
    for (auto& u : ups) all.push_back(&u);
    for (auto& b : blocks) for (auto& n : b) all.push_back(&n);
    size_t total = 0;
    auto pad8 = [](size_t n) { return (n + 7) & ~(size_t)7; };  // every tensor starts 16-byte aligned
    for (auto* n : all) total += pad8(st.get(n->w).numel()) + pad8(st.get(n->b).numel());
    std::vector<uint16_t> host(total);
    std::map<const Name*, std::pair<size_t, size_t>> off;
    size_t o = 0;
    // Conv weights are stored K-major, [Cin*9][Cout]: a K-chunk of one conv is then a contiguous,
    // coalesced block for the fast kernel, and a transposed GEMM operand for the reference path.
    auto put = [&](const StTensor& t) {
        size_t start = o;
        int64_t n = t.numel(), cout = t.shape[0], k = n / cout;
        for (int64_t i = 0; i < n; i++) {
            float v = st_elem_f32(t, i);
            if (bf16_weights) v = bf16_round(v);
            uint16_t h = t.dtype == DType::F16 && !bf16_weights ? ((const uint16_t*)t.data)[i] : f32_to_f16_bits(v);
            host[start + (i % k) * cout + i / k] = h;  // (co, kk) -> kk * Cout + co
        }
        o += pad8(n);
        return start;
    };
    for (auto* n : all) { size_t w = put(st.get(n->w)); size_t b = put(st.get(n->b)); off[n] = {w, b}; }

    if (blob_) {  // reloading: drop the previous model
        if (blob_host_) CK(cudaFreeHost(blob_));
        else CK(cudaFree(blob_));
        blob_ = nullptr;
        blob_host_ = false;
    }
    size_t bytes = total * 2;
    blob_bytes_ = bytes;
    bool device = place == Place::Device || (place == Place::Auto && gpu_free_bytes() > bytes + G.reserve_bytes);
    if (device) {
        CK(cudaMalloc(&blob_, bytes));
    } else {
        void* h;
        CK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        void* d;
        CK(cudaHostGetDevicePointer(&d, h, 0));
        blob_ = (uint16_t*)d;
        blob_host_ = true;
    }
    CK(cudaMemcpy(blob_, host.data(), bytes, cudaMemcpyHostToDevice));

    auto mk = [&](const Name& n) {
        Conv c;
        auto& s = st.get(n.w).shape;
        c.cout = (int)s[0];
        c.cin = (int)s[1];
        c.w = blob_ + off[&n].first;
        c.b = blob_ + off[&n].second;
        return c;
    };
    first_ = mk(first); body_ = mk(body); hr_ = mk(hr); last_ = mk(last);
    ups_.clear();
    for (auto& u : ups) ups_.push_back(mk(u));
    blocks_.assign(nb, Rrdb{});
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < 15; j++) blocks_[i].r[j / 5].c[j % 5] = mk(blocks[i][j]);
}

Upscaler::~Upscaler() {
    if (!blob_) return;
    if (blob_host_) cudaFreeHost(blob_);
    else cudaFree(blob_);
}

// ---------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------
__global__ void k_h2f(float* y, const uint16_t* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = h2f(x[i]);
}

// y[c, p] (channel stride cs) = act(y + b[c])
__global__ void k_bias_act(float* y, size_t cs, const uint16_t* b, int C, int P, int lr) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * P) return;
    int c = (int)(i / P), p = (int)(i % P);
    float v = y[c * cs + p] + h2f(b[c]);
    y[c * cs + p] = lr ? lrelu(v) : v;
}

// x = x + alpha * (t + b[c]); then, if s: x = s + 0.2 * x   (dense-block / RRDB residuals)
__global__ void k_residual(float* x, const float* t, const uint16_t* b, const float* s, int C, int P, float alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * P) return;
    float v = x[i] + alpha * (t[i] + h2f(b[i / P]));
    x[i] = s ? s[i] + 0.2f * v : v;
}

// out[c, oy0 + i, ox0 + j] = clamp(o[c, ly0 + i, lx0 + j] + b[c], 0, 1)
__global__ void k_store(float* out, int OH, int OW, int oy0, int ox0, const float* o, int bh, int bw, int ly0, int lx0,
                        int nh, int nw, const uint16_t* b) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)3 * nh * nw) return;
    int j = (int)(i % nw);
    int r = (int)((i / nw) % nh);
    int c = (int)(i / ((size_t)nw * nh));
    float v = o[((size_t)c * bh + ly0 + r) * bw + lx0 + j] + h2f(b[c]);
    out[((size_t)c * OH + oy0 + r) * OW + ox0 + j] = fminf(fmaxf(v, 0.f), 1.f);
}

// ---------------------------------------------------------------------------
// conv 3x3, pad 1: reference path (im2col + fp32 cuBLAS)
// ---------------------------------------------------------------------------
// Input pixel (c, y, x) of an H x W output region lives at in[c*cs + y*rs + x]; up = 2 reads a
// half-size input nearest-upsampled on the fly.
__global__ void k_im2col(float* col, const float* in, size_t cs, int rs, int C, int H, int W, int p0, int n, int up) {
    size_t N = (size_t)C * 9 * n;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = (int)(i % n);
    size_t r = i / n;  // c*9 + ky*3 + kx
    int kx = (int)(r % 3), ky = (int)((r / 3) % 3), c = (int)(r / 9);
    int p = p0 + j;
    int y = p / W + ky - 1, x = p % W + kx - 1;
    col[i] = (y >= 0 && y < H && x >= 0 && x < W) ? in[c * cs + (size_t)(y / up) * rs + x / up] : 0.f;
}

// ---------------------------------------------------------------------------
// conv 3x3, pad 1: fast path (direct implicit GEMM, fp32 FMA, no im2col)
// ---------------------------------------------------------------------------
// A block computes MB = 8*MG output channels over a TH x TW pixel tile. Per K-chunk of CK input
// channels it stages the (TH+2) x (TW+2) input patch and the [CK*9][MB] weight slab in shared
// memory; each thread then accumulates 8 channels x 8 consecutive pixels of one row, loading a
// 10-wide patch row once per (ci, ky) and reusing it for all three kx taps (~20 FMA per LDS).
template <int MG, int TH, int TW, int CK>
__global__ void __launch_bounds__(256, 2)
k_conv3x3(float* __restrict__ out, size_t ocs, const float* __restrict__ in, size_t ics, int irs, int Cin, int Cout, int H, int W,
          int sh, const uint16_t* __restrict__ wt, const uint16_t* __restrict__ bias, int act) {
    constexpr int MB = MG * 8, PGX = TW / 8, PG = TH * PGX;
    constexpr int PH = TH + 2, PX = TW + 2, PW = (PX + 3) / 4 * 4;
    static_assert(MG * PG == 256, "256 threads");
    __shared__ __align__(16) float s_in[CK][PH][PW];
    __shared__ __align__(16) float s_w[CK * 9][MB];

    const int tid = threadIdx.x;
    const int mg = tid / PG, pg = tid % PG;  // one m-group per warp: weight reads are broadcasts
    const int py = pg % TH, px = (pg / TH) * 8;  // a quarter-warp spans 8 rows: conflict-free patch loads
    const int tiles_x = (W + TW - 1) / TW;
    const int ty0 = (blockIdx.x / tiles_x) * TH, tx0 = (blockIdx.x % tiles_x) * TW;
    const int mb0 = blockIdx.y * MB;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

    for (int c0 = 0; c0 < Cin; c0 += CK) {
        __syncthreads();
        for (int i = tid; i < CK * PH * PX; i += 256) {
            int c = i / (PH * PX), r = (i / PX) % PH, x = i % PX;
            int gy = ty0 + r - 1, gx = tx0 + x - 1, ci = c0 + c;
            float v = 0.f;
            if (ci < Cin && gy >= 0 && gy < H && gx >= 0 && gx < W) v = in[ci * ics + (size_t)(gy >> sh) * irs + (gx >> sh)];
            s_in[c][r][x] = v;
        }
        if ((Cout & 7) == 0) {  // weight rows as 8-half vectors
            for (int i = tid; i < CK * 9 * MB / 8; i += 256) {
                int m = (i % (MB / 8)) * 8, r = i / (MB / 8), co = mb0 + m;
                uint4 u = make_uint4(0, 0, 0, 0);
                if (c0 + r / 9 < Cin && co < Cout) u = *(const uint4*)(wt + ((size_t)c0 * 9 + r) * Cout + co);
                *(float4*)&s_w[r][m] = make_float4(lo_f(u.x), hi_f(u.x), lo_f(u.y), hi_f(u.y));
                *(float4*)&s_w[r][m + 4] = make_float4(lo_f(u.z), hi_f(u.z), lo_f(u.w), hi_f(u.w));
            }
        } else {
            for (int i = tid; i < CK * 9 * MB; i += 256) {
                int m = i % MB, r = i / MB;
                int co = mb0 + m;
                s_w[r][m] = (c0 + r / 9 < Cin && co < Cout) ? h2f(wt[((size_t)c0 * 9 + r) * Cout + co]) : 0.f;
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int c = 0; c < CK; c++) {
#pragma unroll
            for (int ky = 0; ky < 3; ky++) {
                const float* row = &s_in[c][py + ky][px];
                float4 a = *(const float4*)row, b = *(const float4*)(row + 4);
                float2 d = *(const float2*)(row + 8);
                const float v[10] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, d.x, d.y};
#pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    const float* wp = &s_w[c * 9 + ky * 3 + kx][mg * 8];
                    float4 w0 = *(const float4*)wp, w1 = *(const float4*)(wp + 4);
                    const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                    for (int i = 0; i < 8; i++)
#pragma unroll
                        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(w[i], v[j + kx], acc[i][j]);
                }
            }
        }
    }

    const int y = ty0 + py, x0 = tx0 + px;
    if (y >= H) return;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int co = mb0 + mg * 8 + i;
        if (co >= Cout) break;
        float bv = act != ACT_NONE ? h2f(bias[co]) : 0.f;
        float* o = out + co * ocs + (size_t)y * W + x0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (x0 + j >= W) break;
            float r = acc[i][j] + bv;
            o[j] = act == ACT_BIAS_LRELU ? lrelu(r) : r;
        }
    }
}

// fp16x2 variant (G.fp16), same tiling. HFMA2 works on pairs of output channels: the patch is
// staged as duplicated (v, v) half2 and the weights as plain fp16, where two adjacent channels
// form a half2. The inner step is 3 LDS for a 10-pixel row, 1 LDS.128 of weights per kx, then
// 32 HFMA2 per kx, with no shuffles: half the issue slots of the fp32 kernel. Partial sums stay in
// half2 for FL input channels (9*FL products) and are then flushed into fp32 accumulators, which
// bounds the fp16 rounding; activations are rounded to fp16 on the way into shared memory.
__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int MG, int TH, int TW, int CK, int FL>
__global__ void __launch_bounds__(256, 2)
k_conv3x3_h2(float* __restrict__ out, size_t ocs, const float* __restrict__ in, size_t ics, int irs, int Cin, int Cout, int H, int W,
             int sh, const uint16_t* __restrict__ wt, const uint16_t* __restrict__ bias, int act) {
    constexpr int MB = MG * 8, PGX = TW / 8, PG = TH * PGX;
    constexpr int PH = TH + 2, PX = TW + 2, PW = (PX + 3) / 4 * 4;
    static_assert(MG * PG == 256 && CK % FL == 0, "256 threads");
    __shared__ __align__(16) uint32_t s_in[CK][PH][PW];  // (v, v) half2
    __shared__ __align__(16) uint16_t s_w[CK * 9][MB];   // fp16

    const int tid = threadIdx.x;
    const int mg = tid / PG, pg = tid % PG;
    const int py = pg % TH, px = (pg / TH) * 8;  // a quarter-warp spans 8 rows: conflict-free patch loads
    const int tiles_x = (W + TW - 1) / TW;
    const int ty0 = (blockIdx.x / tiles_x) * TH, tx0 = (blockIdx.x % tiles_x) * TW;
    const int mb0 = blockIdx.y * MB;
    const bool vec = (Cout & 7) == 0;  // weight rows load as 8-half vectors

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

    for (int c0 = 0; c0 < Cin; c0 += CK) {
        __syncthreads();
        for (int i = tid; i < CK * PH * PX; i += 256) {
            int c = i / (PH * PX), r = (i / PX) % PH, x = i % PX;
            int gy = ty0 + r - 1, gx = tx0 + x - 1, ci = c0 + c;
            float v = 0.f;
            if (ci < Cin && gy >= 0 && gy < H && gx >= 0 && gx < W) v = in[ci * ics + (size_t)(gy >> sh) * irs + (gx >> sh)];
            uint32_t b = __half_as_ushort(__float2half_rn(v));
            s_in[c][r][x] = b | b << 16;
        }
        if (vec) {
            for (int i = tid; i < CK * 9 * MB / 8; i += 256) {
                int m = (i % (MB / 8)) * 8, r = i / (MB / 8), co = mb0 + m;
                uint4 u = make_uint4(0, 0, 0, 0);
                if (c0 + r / 9 < Cin && co < Cout) u = *(const uint4*)(wt + ((size_t)c0 * 9 + r) * Cout + co);
                *(uint4*)&s_w[r][m] = u;
            }
        } else {
            for (int i = tid; i < CK * 9 * MB; i += 256) {
                int m = i % MB, r = i / MB, co = mb0 + m;
                s_w[r][m] = (c0 + r / 9 < Cin && co < Cout) ? wt[((size_t)c0 * 9 + r) * Cout + co] : (uint16_t)0;
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int cf = 0; cf < CK; cf += FL) {
            uint32_t part[4][8];  // [channel pair][pixel], half2 bits
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) part[i][j] = 0u;
#pragma unroll 1
            for (int c = cf; c < cf + FL; c++) {
#pragma unroll
                for (int ky = 0; ky < 3; ky++) {
                    const uint32_t* row = &s_in[c][py + ky][px];
                    uint4 a = *(const uint4*)row, b = *(const uint4*)(row + 4);
                    uint2 d = *(const uint2*)(row + 8);
                    const uint32_t v[10] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, d.x, d.y};
#pragma unroll
                    for (int kx = 0; kx < 3; kx++) {
                        uint4 w4 = *(const uint4*)&s_w[c * 9 + ky * 3 + kx][mg * 8];
                        const uint32_t w[4] = {w4.x, w4.y, w4.z, w4.w};  // channels (0,1) (2,3) (4,5) (6,7)
#pragma unroll
                        for (int i = 0; i < 4; i++)
#pragma unroll
                            for (int j = 0; j < 8; j++) part[i][j] = hfma2u(w[i], v[j + kx], part[i][j]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[2 * i][j] += lo_f(part[i][j]);
                    acc[2 * i + 1][j] += hi_f(part[i][j]);
                }
        }
    }

    const int y = ty0 + py, x0 = tx0 + px;
    if (y >= H) return;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int co = mb0 + mg * 8 + i;
        if (co >= Cout) break;
        float bv = act != ACT_NONE ? h2f(bias[co]) : 0.f;
        float* o = out + co * ocs + (size_t)y * W + x0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (x0 + j >= W) break;
            float r = acc[i][j] + bv;
            o[j] = act == ACT_BIAS_LRELU ? lrelu(r) : r;
        }
    }
}

// Convs with a handful of output channels (conv_last, 64 -> 3): fp32, one pixel per thread per
// step, the whole weight tensor in shared memory, input through L1.
template <int CO>
__global__ void k_conv3x3_small(float* __restrict__ out, size_t ocs, const float* __restrict__ in, size_t ics, int irs, int Cin, int Cout,
                                int H, int W, int sh, const uint16_t* __restrict__ wt, const uint16_t* __restrict__ bias, int act) {
    extern __shared__ float s_wf[];  // [Cin*9][CO]
    for (int i = threadIdx.x; i < Cin * 9 * CO; i += blockDim.x) {
        int m = i % CO, r = i / CO;
        s_wf[i] = m < Cout ? h2f(wt[(size_t)r * Cout + m]) : 0.f;
    }
    __syncthreads();
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < H * W; p += gridDim.x * blockDim.x) {
        int y = p / W, x = p % W;
        float acc[CO];
#pragma unroll
        for (int m = 0; m < CO; m++) acc[m] = 0.f;
        for (int ci = 0; ci < Cin; ci++) {
            const float* ip = in + ci * ics;
#pragma unroll
            for (int ky = 0; ky < 3; ky++) {
                int yy = y + ky - 1;
                if (yy < 0 || yy >= H) continue;
#pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    int xx = x + kx - 1;
                    if (xx < 0 || xx >= W) continue;
                    float v = __ldg(ip + (size_t)(yy >> sh) * irs + (xx >> sh));
                    const float* w = s_wf + (ci * 9 + ky * 3 + kx) * CO;
#pragma unroll
                    for (int m = 0; m < CO; m++) acc[m] = fmaf(w[m], v, acc[m]);
                }
            }
        }
        for (int m = 0; m < Cout; m++) {
            float r = acc[m] + (act != ACT_NONE ? h2f(bias[m]) : 0.f);
            out[m * ocs + p] = act == ACT_BIAS_LRELU ? lrelu(r) : r;
        }
    }
}

// out [Cout, H*W] with channel stride ocs.
void Upscaler::conv(float* out, size_t ocs, const float* in, size_t ics, int irs, int H, int W, const Conv& c, int up, int act,
                    const char* tag) {
    int P = H * W, K = c.cin * 9;
    int sh = up == 2 ? 1 : 0;
    const uint16_t *w = wp(c.w), *b = wp(c.b);
    if (!reference) {
        ProfScope ps(tag);
        if (c.cout <= 4) {
            unsigned blocks = (unsigned)std::min<size_t>(nb_(P), 24 * 16);
            k_conv3x3_small<4><<<blocks, 256, c.cin * 9 * 4 * 4, G.stream>>>(out, ocs, in, ics, irs, c.cin, c.cout, H, W, sh, w, b, act);
        } else if (G.fp16) {
            if (c.cout <= 32) {
                dim3 grid(((W + 31) / 32) * ((H + 15) / 16), 1);
                k_conv3x3_h2<4, 16, 32, CONV_CK, CONV_FL><<<grid, 256, 0, G.stream>>>(out, ocs, in, ics, irs, c.cin, c.cout, H, W, sh, w, b, act);
            } else {
                dim3 grid(((W + 15) / 16) * ((H + 15) / 16), (c.cout + 63) / 64);
                k_conv3x3_h2<8, 16, 16, CONV_CK, CONV_FL><<<grid, 256, 0, G.stream>>>(out, ocs, in, ics, irs, c.cin, c.cout, H, W, sh, w, b, act);
            }
        } else if (c.cout <= 32) {
            dim3 grid(((W + 31) / 32) * ((H + 15) / 16), 1);
            k_conv3x3<4, 16, 32, CONV_CK><<<grid, 256, 0, G.stream>>>(out, ocs, in, ics, irs, c.cin, c.cout, H, W, sh, w, b, act);
        } else {
            dim3 grid(((W + 15) / 16) * ((H + 15) / 16), (c.cout + 63) / 64);
            k_conv3x3<8, 16, 16, CONV_CK><<<grid, 256, 0, G.stream>>>(out, ocs, in, ics, irs, c.cin, c.cout, H, W, sh, w, b, act);
        }
        return;
    }
    {
        ProfScope ps("w->f32");
        k_h2f<<<nb_((size_t)c.cout * K), 256, 0, G.stream>>>(wf_, w, (size_t)c.cout * K);
    }
    int Pc = (int)std::min<size_t>(P, col_cap_ / K);
    if (Pc < 1) throw std::runtime_error("upscaler: im2col slab too small");
    for (int p0 = 0; p0 < P; p0 += Pc) {
        int n = std::min(Pc, P - p0);
        {
            ProfScope ps("im2col");
            k_im2col<<<nb_((size_t)K * n), 256, 0, G.stream>>>(col_, in, ics, irs, c.cin, H, W, p0, n, up);
        }
        ProfScope ps(tag);
        // weights are stored [K][Cout]: op(A) = A^T
        gemm(true, false, c.cout, n, K, 1.f, wf_, c.cout, col_, n, 0.f, out + p0, (int)ocs);
    }
    if (act != ACT_NONE) {
        ProfScope ps("eltwise");
        k_bias_act<<<nb_((size_t)c.cout * P), 256, 0, G.stream>>>(out, ocs, b, c.cout, P, act == ACT_BIAS_LRELU);
    }
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------
// One tile: input extent [y0,y1) x [x0,x1) (core + halo, clipped to the image), writes the scaled
// core [cy0,cy1) x [cx0,cx1) into out. Everything in image LR coordinates.
void Upscaler::tile(float* out, const float* in, int H, int W, int y0, int y1, int x0, int x1, int cy0, int cy1, int cx0, int cx1) {
    const int th = y1 - y0, tw = x1 - x0;
    const size_t P = (size_t)th * tw;
    const int ctot = nf + 4 * gc;
    size_t m0 = G.arena.mark();
    float* feat = G.arena.f(nf * P);
    size_t m1 = G.arena.mark();
    float* save = G.arena.f(nf * P);
    float* buf = G.arena.f(ctot * P);
    float* tmp = G.arena.f(nf * P);

    conv(feat, P, in + (size_t)y0 * W + x0, (size_t)H * W, W, th, tw, first_, 1, ACT_BIAS, "conv first");
    CK(cudaMemcpyAsync(buf, feat, nf * P * 4, cudaMemcpyDeviceToDevice, G.stream));
    for (auto& blk : blocks_) {
        CK(cudaMemcpyAsync(save, buf, nf * P * 4, cudaMemcpyDeviceToDevice, G.stream));
        for (int r = 0; r < 3; r++) {
            const Rdb& d = blk.r[r];
            static const char* tags[4] = {"rdb conv1", "rdb conv2", "rdb conv3", "rdb conv4"};
            for (int k = 0; k < 4; k++) conv(buf + (nf + k * gc) * P, P, buf, P, tw, th, tw, d.c[k], 1, ACT_BIAS_LRELU, tags[k]);
            conv(tmp, P, buf, P, tw, th, tw, d.c[4], 1, ACT_NONE, "rdb conv5");
            ProfScope ps("eltwise");
            k_residual<<<nb_(nf * P), 256, 0, G.stream>>>(buf, tmp, wp(d.c[4].b), r == 2 ? save : nullptr, nf, (int)P, 0.2f);
        }
    }
    conv(tmp, P, buf, P, tw, th, tw, body_, 1, ACT_NONE, "conv body");
    {
        ProfScope ps("eltwise");
        k_residual<<<nb_(nf * P), 256, 0, G.stream>>>(feat, tmp, wp(body_.b), nullptr, nf, (int)P, 1.f);
    }
    G.arena.release(m1);

    // Upsampling tail over the core only, in row bands with TAIL_HALO context (exact). Two
    // ping-pong buffers of nf x (s*s) floats per band pixel: up_1 -> A, up_2 -> B, ..., hr, last.
    const int s = scale_;
    const int f0 = std::max(0, cx0 - x0 - TAIL_HALO), f1 = std::min(tw, cx1 - x0 + TAIL_HALO), bw = f1 - f0;
    const int r_lo = cy0 - y0, r_hi = cy1 - y0;  // core rows, tile-local
    size_t per_row = (size_t)2 * nf * s * s * bw * 4;
    size_t avail = G.arena.free_bytes() > 4096 ? G.arena.free_bytes() - 4096 : 0;
    int rows_fit = (int)(avail / per_row) - 2 * TAIL_HALO;
    if (rows_fit < 1) throw std::runtime_error("upscaler: not enough scratch VRAM for the upsampling tail");
    int band = std::min(rows_fit, r_hi - r_lo);
    for (int r0 = r_lo; r0 < r_hi; r0 += band) {
        int r1 = std::min(r_hi, r0 + band);
        int e0 = std::max(0, r0 - TAIL_HALO), e1 = std::min(th, r1 + TAIL_HALO), bh = e1 - e0;
        size_t mb = G.arena.mark();
        size_t cap = (size_t)nf * s * s * bh * bw;
        float* A = G.arena.f(cap);
        float* B = G.arena.f(cap);
        const float* src = feat + (size_t)e0 * tw + f0;
        size_t scs = P;
        int srs = tw, h = bh, w = bw;
        float* dst = A;
        for (auto& u : ups_) {
            h *= 2; w *= 2;
            conv(dst, (size_t)h * w, src, scs, srs, h, w, u, 2, ACT_BIAS_LRELU, "tail up");
            src = dst; scs = (size_t)h * w; srs = w;
            dst = dst == A ? B : A;
        }
        conv(dst, (size_t)h * w, src, scs, srs, h, w, hr_, 1, ACT_BIAS_LRELU, "tail hr");
        src = dst; dst = dst == A ? B : A;
        conv(dst, (size_t)h * w, src, (size_t)h * w, w, h, w, last_, 1, ACT_NONE, "tail last");
        {
            ProfScope ps("eltwise");
            int nh = (r1 - r0) * s, nw = (cx1 - cx0) * s;
            k_store<<<nb_((size_t)3 * nh * nw), 256, 0, G.stream>>>(out, H * s, W * s, (y0 + r0) * s, cx0 * s, dst, h, w,
                                                                   (r0 - e0) * s, (cx0 - x0 - f0) * s, nh, nw, wp(last_.b));
        }
        G.arena.release(mb);
    }
    G.arena.release(m0);
}

void Upscaler::run(float* out, const float* in, int H, int W) {
    if (!blob_) throw std::runtime_error("upscaler: no model loaded");
    size_t m = G.arena.mark();
    wbase_ = blob_;
    if (blob_host_ && G.arena.free_bytes() > blob_bytes_ + ((size_t)128 << 20)) {  // one PCIe copy per run
        uint16_t* d = (uint16_t*)G.arena.f((blob_bytes_ + 3) / 4);
        CK(cudaMemcpyAsync(d, blob_, blob_bytes_, cudaMemcpyDefault, G.stream));
        wbase_ = d;
    }
    if (reference) {  // reference path: fp32 weight staging + im2col slab (big enough for efficient GEMMs)
        const int kmax = (nf + 4 * gc) * 9;
        wf_ = G.arena.f((size_t)std::max(nf, gc) * kmax);
        size_t free0 = G.arena.free_bytes();
        col_cap_ = std::min<size_t>(free0 / 4 / 5, (size_t)12 << 20);  // floats; <= 48 MB
        col_cap_ = std::max<size_t>(col_cap_, (size_t)kmax * 64);
        col_ = G.arena.f(col_cap_);
    }

    // Tile plan: the trunk needs (4 nf + 4 gc) floats per tile pixel; the tail bands reuse that
    // space afterwards. Pick the grid with the least total (halo-inflated) work that fits.
    size_t per_px = (size_t)(4 * nf + 4 * gc) * 4;
    size_t avail = G.arena.free_bytes();
    size_t tail_min = (size_t)2 * nf * scale_ * scale_ * (1 + 2 * TAIL_HALO) * std::min(W, 4096) * 4;
    size_t pmax = avail > tail_min + ((size_t)1 << 20) ? (avail - tail_min - ((size_t)1 << 20)) / per_px : 0;
    auto extent_sum = [&](int n, int len, int& emax) {  // sum and max of clipped tile extents along one axis
        int c = (len + n - 1) / n;
        long long sum = 0;
        emax = 0;
        for (int i = 0; i < n; i++) {
            int a = std::max(0, i * c - (n > 1 ? halo : 0)), b = std::min(len, (i + 1) * c + (n > 1 ? halo : 0));
            if (i * c >= len) break;
            sum += b - a;
            emax = std::max(emax, b - a);
        }
        return sum;
    };
    int best_ny = 0, best_nx = 0;
    long long best = -1;
    for (int ny = 1; ny <= std::min(H, 64); ny++) {
        if (max_tile > 0 && (H + ny - 1) / ny > max_tile) continue;
        int eh;
        long long sy = extent_sum(ny, H, eh);
        for (int nx = 1; nx <= std::min(W, 64); nx++) {
            if (max_tile > 0 && (W + nx - 1) / nx > max_tile) continue;
            int ew;
            long long sx = extent_sum(nx, W, ew);
            if ((size_t)eh * ew > pmax) continue;
            long long cost = sy * sx;
            if (best < 0 || cost < best) { best = cost; best_ny = ny; best_nx = nx; }
        }
    }
    if (best < 0) throw std::runtime_error("upscaler: not enough scratch VRAM for even a small tile (" + std::to_string(avail >> 20) + " MB free)");

    int ch = (H + best_ny - 1) / best_ny, cw = (W + best_nx - 1) / best_nx;
    tiles_used = 0;
    for (int ty = 0; ty < best_ny; ty++) {
        int cy0 = ty * ch, cy1 = std::min(H, cy0 + ch);
        if (cy0 >= H) break;
        int y0 = best_ny > 1 ? std::max(0, cy0 - halo) : 0, y1 = best_ny > 1 ? std::min(H, cy1 + halo) : H;
        for (int tx = 0; tx < best_nx; tx++) {
            int cx0 = tx * cw, cx1 = std::min(W, cx0 + cw);
            if (cx0 >= W) break;
            int x0 = best_nx > 1 ? std::max(0, cx0 - halo) : 0, x1 = best_nx > 1 ? std::min(W, cx1 + halo) : W;
            tile(out, in, H, W, y0, y1, x0, x1, cy0, cy1, cx0, cx1);
            tiles_used++;
        }
    }
    G.arena.release(m);
}
