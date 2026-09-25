// YOLOv8 detector: see detect.h. The graph comes from the resolved "layers" list that
// tools/convert_yolo.py writes (Conv, C2f, SPPF, Upsample, Concat, Detect), flattened here into
// a list of primitive ops over channel-first fp32 tensors:
//   conv  = fp16 weights (default): fp16x2 implicit-GEMM kernel (ig::k_conv) with fused bias/SiLU/residual
//           fp32 weights (exact mode): im2col (skipped for 1x1/s1) + cuBLAS SGEMM + epilogue kernel
//   C2f   = cv1 writes both chunks straight into the concat buffer, each bottleneck appends its
//           slice, cv2 reads the whole buffer (no split/concat copies)
//   SPPF  = cv1 + three 5x5 max-pools chained through slices of one concat buffer
//   Detect= per-level conv chains writing into one [no, N] head tensor (N = all anchors),
//           then DFL + dist2bbox + sigmoid decode; confidence filter + NMS on the host.
#include "detect.h"

#include "json.h"
#include "safetensors.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>

namespace {

inline unsigned nb(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

struct ConvW {
    size_t w = 0;   // element offset into the weight store
    size_t b = 0;   // element offset into the bias store
    int cin = 0, cout = 0, k = 1, s = 1, p = 0;
    bool silu = false;
};

enum class OpK { Letterbox, Conv, MaxPool, Up2, Copy, Decode };
enum TKind { T_NORMAL, T_RAW, T_DEC };

struct TDesc { int C = 0; int div = 1; TKind kind = T_NORMAL; int first = 1 << 30, last = -1; };
struct View { int t = -1; int c0 = 0; int lvl = -1; };
struct Op { OpK k; View in, out, res; int conv = -1; int C = 0; int pk = 0; std::string prof; };

struct Levels { int n; int off[4], h[4], w[4]; float stride[4]; };

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------
// Ultralytics LetterBox: cv2.resize INTER_LINEAR (half-pixel centres, source coordinate clamped
// to [0, size-1], no antialias) of the image into [top, top+nh) x [left, left+nw), 114 gray around.
__global__ void k_letterbox(float* out, const float* in, int H, int W, int oh, int ow,
                            int top, int left, int nh, int nw, float sy, float sx) {
    size_t n = (size_t)3 * oh * ow;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = (int)(i % ow);
    size_t r = i / ow;
    int y = (int)(r % oh), c = (int)(r / oh);
    int yy = y - top, xx = x - left;
    float v = 114.f / 255.f;
    if (yy >= 0 && yy < nh && xx >= 0 && xx < nw) {
        float fy = (yy + 0.5f) * sy - 0.5f, fx = (xx + 0.5f) * sx - 0.5f;
        int y0 = (int)floorf(fy), x0 = (int)floorf(fx);
        float wy = fy - y0, wx = fx - x0;
        if (y0 < 0) { y0 = 0; wy = 0.f; }
        if (y0 >= H - 1) { y0 = H - 1; wy = 0.f; }
        if (x0 < 0) { x0 = 0; wx = 0.f; }
        if (x0 >= W - 1) { x0 = W - 1; wx = 0.f; }
        int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        const float* p = in + (size_t)c * H * W;
        float a = p[(size_t)y0 * W + x0], b = p[(size_t)y0 * W + x1];
        float d = p[(size_t)y1 * W + x0], e = p[(size_t)y1 * W + x1];
        v = (a + (b - a) * wx) * (1.f - wy) + (d + (e - d) * wx) * wy;
    }
    out[i] = v;
}

// Column matrix for output pixels [p0, p0+n) of a k x k conv (stride s, zero pad) -> col [C*k*k, n].
// grid.y = column-matrix row (c, ky, kx), grid.x over pixels; 32-bit index math.
__global__ void k_im2col(float* __restrict__ col, const float* __restrict__ in, int H, int W,
                         int k, int s, int pad, int Wo, int p0, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int r = blockIdx.y, kk = k * k;
    int c = r / kk, q = r - c * kk, ky = q / k, kx = q - ky * k;
    int p = p0 + j, oy = p / Wo, ox = p - oy * Wo;
    int y = oy * s - pad + ky, x = ox * s - pad + kx;
    col[(size_t)r * n + j] = ((unsigned)y < (unsigned)H && (unsigned)x < (unsigned)W) ? __ldg(in + ((size_t)c * H + y) * W + x) : 0.f;
}

// ---- fp16x2 implicit-GEMM convolution (GPUs without tensor cores: HFMA2 runs at 2x the fp32 rate) ----
// out[Cout, P] = W[Cout, K] * X[K, P], K = Cin*KS*KS, where X is the im2col of the input gathered on
// the fly into shared memory (never materialised). Weights are fp16, stored K-major (wT = W^T);
// activations are rounded to fp16 while staged. Products accumulate in fp16 pairs and are flushed
// into fp32 every FLUSH*BK k (the scheme hgemm.cuh uses), so long reductions (K up to 5184) stay
// accurate. The epilogue fuses bias, SiLU and the bottleneck residual.
// Tile BM x 128 x 16 (BM = 16*TM), 256 threads, TM x 8 outputs per thread.
namespace ig {
constexpr int BN = 128, BK = 16, FLUSH = 2;

template <int TM, int KS>
__global__ void __launch_bounds__(256, TM <= 6 ? 2 : 1)
k_conv(float* __restrict__ out, int ldo, const float* __restrict__ in, const __half* __restrict__ wT,
       const float* __restrict__ bias, const float* __restrict__ res, int H, int Wd, int Cout, int Wo, int P,
       int stride, int pad, int K, int silu) {
    constexpr int BM = 16 * TM, KK = KS * KS;
    __shared__ __align__(16) __half2 As[2][BK][BM];   // weights, stored as (w, w) pairs
    __shared__ __align__(16) __half Bs[2][BK][BN];    // gathered activations
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;

    // A staging (weights are stored K-major, wT[k][m]): element tid + q*256 of the BK x BM tile, so a
    // warp reads consecutive output channels of one k row (coalesced) and stores conflict-free.
    // B staging: this thread's pixel b_n, k rows b_k + 2q (q < 8); a warp reads 32 consecutive pixels.
    const int b_n = tid % BN, b_k = tid / BN;
    int iy0, ix0;
    {
        int p = n0 + b_n, oy = p / Wo, ox = p - oy * Wo;
        iy0 = p < P ? oy * stride - pad : -(1 << 20);   // out-of-range pixel: always reads zero
        ix0 = ox * stride - pad;
    }
    const size_t HW = (size_t)H * Wd;
    __half ra[TM];
    float rb[8];

    auto load = [&](int k0) {
#pragma unroll
        for (int q = 0; q < TM; q++) {
            int e = tid + q * 256, k = k0 + e / BM, m = m0 + e % BM;
            ra[q] = (k < K && m < Cout) ? wT[(size_t)k * Cout + m] : __float2half(0.f);
        }
#pragma unroll
        for (int q = 0; q < 8; q++) {
            int k = k0 + b_k + 2 * q;
            float v = 0.f;
            if (k < K) {
                int c = k / KK, r = k - c * KK, ky = r / KS, kx = r - ky * KS;
                int y = iy0 + ky, x = ix0 + kx;
                if ((unsigned)y < (unsigned)H && (unsigned)x < (unsigned)Wd) v = __ldg(in + c * HW + y * Wd + x);
            }
            rb[q] = v;
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int q = 0; q < TM; q++) {
            int e = tid + q * 256;
            As[buf][e / BM][e % BM] = __half2half2(ra[q]);
        }
#pragma unroll
        for (int q = 0; q < 8; q++) Bs[buf][b_k + 2 * q][b_n] = __float2half_rn(rb[q]);
    };

    float acc[TM][8];
    __half2 part[TM][4];
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) part[i][j] = __float2half2_rn(0.f);
    }

    const int tiles = (K + BK - 1) / BK;
    load(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < tiles; t++) {
        const int buf = t & 1;
        if (t + 1 < tiles) load((t + 1) * BK);
#pragma unroll
        for (int k = 0; k < BK; k++) {
            __half2 a2[TM];
            if constexpr (TM % 4 == 0) {
#pragma unroll
                for (int v = 0; v < TM / 4; v++) {
                    uint4 av = *(const uint4*)&As[buf][k][ty * TM + v * 4];
                    a2[v * 4 + 0] = *(__half2*)&av.x; a2[v * 4 + 1] = *(__half2*)&av.y;
                    a2[v * 4 + 2] = *(__half2*)&av.z; a2[v * 4 + 3] = *(__half2*)&av.w;
                }
            } else if constexpr (TM % 2 == 0) {
#pragma unroll
                for (int v = 0; v < TM / 2; v++) {
                    uint2 av = *(const uint2*)&As[buf][k][ty * TM + v * 2];
                    a2[v * 2 + 0] = *(__half2*)&av.x; a2[v * 2 + 1] = *(__half2*)&av.y;
                }
            } else {
#pragma unroll
                for (int v = 0; v < TM; v++) a2[v] = As[buf][k][ty * TM + v];
            }
            uint4 bv = *(const uint4*)&Bs[buf][k][tx * 8];
            __half2 b2[4] = {*(__half2*)&bv.x, *(__half2*)&bv.y, *(__half2*)&bv.z, *(__half2*)&bv.w};
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) part[i][j] = __hfma2(a2[i], b2[j], part[i][j]);
        }
        if ((t + 1) % FLUSH == 0 || t + 1 == tiles) {
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    float2 f = __half22float2(part[i][j]);
                    acc[i][2 * j] += f.x;
                    acc[i][2 * j + 1] += f.y;
                    part[i][j] = __float2half2_rn(0.f);
                }
        }
        if (t + 1 < tiles) {
            store(buf ^ 1);
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int m = m0 + ty * TM + i;
        if (m >= Cout) break;
        float b = bias[m];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int p = n0 + tx * 8 + j;
            if (p >= P) break;
            float v = acc[i][j] + b;
            if (silu) v = v / (1.f + expf(-v));
            if (res) v += res[(size_t)m * P + p];
            out[(size_t)m * ldo + p] = v;
        }
    }
}

template <int TM>
void launch_tm(int ks, float* out, int ldo, const float* in, const __half* w, const float* bias, const float* res,
               int H, int Wd, int Cout, int Wo, int P, int s, int pad, int K, int silu) {
    dim3 grid((P + BN - 1) / BN, (Cout + 16 * TM - 1) / (16 * TM));
    if (ks == 3) k_conv<TM, 3><<<grid, 256, 0, G.stream>>>(out, ldo, in, w, bias, res, H, Wd, Cout, Wo, P, s, pad, K, silu);
    else k_conv<TM, 1><<<grid, 256, 0, G.stream>>>(out, ldo, in, w, bias, res, H, Wd, Cout, Wo, P, s, pad, K, silu);
}

// Tile height BM = 16*TM that divides Cout, preferring 96 (TM=6: 128 regs -> 2 blocks/SM), then
// 128/64/48 (YOLOv8 widths are all multiples of 48 or 64); else the one wasting the fewest rows.
inline int pick_tm(int cout) {
    const int tms[4] = {6, 8, 4, 3};
    for (int t : tms) if (cout % (16 * t) == 0) return t;
    int best = 3, waste = 1 << 30;
    for (int t : tms) {
        int w = (cout + 16 * t - 1) / (16 * t) * 16 * t - cout;
        if (w < waste) { waste = w; best = t; }
    }
    return best;
}

void launch(int tm, int ks, float* out, int ldo, const float* in, const __half* w, const float* bias, const float* res,
            int H, int Wd, int Cout, int Wo, int P, int s, int pad, int K, int silu) {
    switch (tm) {
        case 8: launch_tm<8>(ks, out, ldo, in, w, bias, res, H, Wd, Cout, Wo, P, s, pad, K, silu); break;
        case 6: launch_tm<6>(ks, out, ldo, in, w, bias, res, H, Wd, Cout, Wo, P, s, pad, K, silu); break;
        case 4: launch_tm<4>(ks, out, ldo, in, w, bias, res, H, Wd, Cout, Wo, P, s, pad, K, silu); break;
        default: launch_tm<3>(ks, out, ldo, in, w, bias, res, H, Wd, Cout, Wo, P, s, pad, K, silu); break;
    }
}
}  // namespace ig

// y[c, p] (row stride ld) = act(y + b[c]) (+ res[c, p], row stride P)
__global__ void k_bias_act(float* y, int ld, const float* __restrict__ b, int C, int P, int silu, const float* __restrict__ res) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * P) return;
    int c = (int)(i / P), p = (int)(i % P);
    float* py = y + (size_t)c * ld + p;
    float v = *py + b[c];
    if (silu) v = v / (1.f + expf(-v));
    if (res) v += res[i];
    *py = v;
}

// k x k max pool, stride 1, pad k/2 (padding never wins, like PyTorch's -inf padding)
__global__ void k_maxpool(float* __restrict__ y, const float* __restrict__ x, int C, int H, int W, int k) {
    size_t n = (size_t)C * H * W;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xx = (int)(i % W);
    size_t r = i / W;
    int yy = (int)(r % H), c = (int)(r / H);
    int h = k / 2;
    const float* p = x + (size_t)c * H * W;
    float m = -INFINITY;
    for (int dy = -h; dy <= h; dy++) {
        int sy = yy + dy;
        if (sy < 0 || sy >= H) continue;
        for (int dx = -h; dx <= h; dx++) {
            int sx = xx + dx;
            if (sx >= 0 && sx < W) m = fmaxf(m, p[(size_t)sy * W + sx]);
        }
    }
    y[i] = m;
}

__global__ void k_up2(float* __restrict__ y, const float* __restrict__ x, int C, int H, int W) {
    size_t n = (size_t)C * H * W * 4;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xo = (int)(i % (2 * W));
    size_t r = i / (2 * W);
    int yo = (int)(r % (2 * H)), c = (int)(r / (2 * H));
    y[i] = x[((size_t)c * H + yo / 2) * W + xo / 2];
}

// raw [4*rm + nc, N] -> dec [4 + nc, N]: DFL expectation per side, dist2bbox (xywh) * stride, sigmoid
__global__ void k_decode(float* __restrict__ dec, const float* __restrict__ raw, int N, int nc, int rm, Levels L) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= N) return;
    int l = 0;
    while (l + 1 < L.n && a >= L.off[l + 1]) l++;
    int q = a - L.off[l];
    float ax = (float)(q % L.w[l]) + 0.5f, ay = (float)(q / L.w[l]) + 0.5f, st = L.stride[l];
    float d[4];
    for (int s = 0; s < 4; s++) {
        const float* r = raw + (size_t)s * rm * N + a;
        float m = -INFINITY;
        for (int b = 0; b < rm; b++) m = fmaxf(m, r[(size_t)b * N]);
        float se = 0.f, sw = 0.f;
        for (int b = 0; b < rm; b++) { float e = expf(r[(size_t)b * N] - m); se += e; sw += e * (float)b; }
        d[s] = sw / se;
    }
    float x1 = ax - d[0], y1 = ay - d[1], x2 = ax + d[2], y2 = ay + d[3];
    dec[a] = (x1 + x2) * 0.5f * st;
    dec[(size_t)N + a] = (y1 + y2) * 0.5f * st;
    dec[(size_t)2 * N + a] = (x2 - x1) * st;
    dec[(size_t)3 * N + a] = (y2 - y1) * st;
    for (int c = 0; c < nc; c++) {
        float v = raw[(size_t)(4 * rm + c) * N + a];
        dec[(size_t)(4 + c) * N + a] = 1.f / (1.f + expf(-v));
    }
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct ArenaGuard {
    size_t m;
    ArenaGuard() : m(G.arena.mark()) {}
    ~ArenaGuard() { G.arena.release(m); }
};

}  // namespace

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------
struct FaceDetector::Impl {
    std::vector<ConvW> convs;
    std::vector<TDesc> tensors;
    std::vector<Op> ops;
    int t_input = -1, t_raw = -1, t_dec = -1;
    std::vector<int> lvl_tensor;   // tensor feeding each Detect level
    std::vector<float> lvl_stride;
    int reg_max = 16, no = 0, nc = 0, max_stride = 32;

    void* wstore = nullptr;        // fp16 or fp32 conv weights (device-visible pointer)
    float* bstore = nullptr;       // fp32 biases (device-visible pointer)
    void* host_block = nullptr;    // host placement: one cudaHostAlloc'd mapped block holding both
    bool w_host = false, w_fp16 = true;
    size_t w_bytes = 0;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    // host-side staging while loading
    std::vector<__half> hw16;
    std::vector<float> hw32, hb;

    int add_tensor(int C, int div, TKind k = T_NORMAL) {
        TDesc d;
        d.C = C; d.div = div; d.kind = k;
        tensors.push_back(d);
        return (int)tensors.size() - 1;
    }

    int add_conv(const Json& r, const SafeTensors& st) {
        ConvW c;
        c.cin = (int)r["cin"].i64(); c.cout = (int)r["cout"].i64();
        c.k = (int)r["k"].i64(); c.s = (int)r["s"].i64(); c.p = (int)r["p"].i64();
        std::string act = r["act"].str();
        if (act != "silu" && act != "none") throw std::runtime_error("detector: unsupported activation " + act);
        c.silu = act == "silu";
        if ((c.k != 1 && c.k != 3) || c.s < 1 || c.p != c.k / 2) throw std::runtime_error("detector: only 1x1/3x3 'same'-padded convs are supported");
        std::string name = r["name"].str();
        const StTensor& w = st.get(name + ".weight");
        const StTensor& b = st.get(name + ".bias");
        if (w.shape.size() != 4 || w.shape[0] != c.cout || w.shape[1] != c.cin || w.shape[2] != c.k || w.shape[3] != c.k)
            throw std::runtime_error("detector: weight shape mismatch for " + name);
        if (b.numel() != c.cout) throw std::runtime_error("detector: bias shape mismatch for " + name);
        size_t n = (size_t)w.numel();
        auto pad64 = [](size_t x) { return (x + 63) & ~(size_t)63; };
        if (w_fp16) {  // K-major [K, Cout] for the implicit-GEMM kernel's coalesced tile loads
            c.w = hw16.size();
            hw16.resize(pad64(c.w + n), __float2half(0.f));
            size_t K = n / c.cout;
            for (size_t m = 0; m < (size_t)c.cout; m++)
                for (size_t k = 0; k < K; k++) hw16[c.w + k * c.cout + m] = __float2half_rn(st_elem_f32(w, (int64_t)(m * K + k)));
        } else {
            c.w = hw32.size();
            hw32.resize(pad64(c.w + n), 0.f);
            for (size_t i = 0; i < n; i++) hw32[c.w + i] = st_elem_f32(w, (int64_t)i);
        }
        c.b = hb.size();
        hb.resize(pad64(c.b + c.cout), 0.f);
        for (int i = 0; i < c.cout; i++) hb[c.b + i] = st_elem_f32(b, i);
        convs.push_back(c);
        return (int)convs.size() - 1;
    }

    int cur_layer = 0;
    void op_conv(int in, int c0in, View out, int conv, View res = View()) {
        Op o{};
        o.k = OpK::Conv; o.in = View{in, c0in, -1}; o.out = out; o.res = res; o.conv = conv;
        const ConvW& c = convs[conv];
        char b[64];
        snprintf(b, sizeof b, "L%02d %dx%ds%d %d>%d /%d", cur_layer, c.k, c.k, c.s, c.cin, c.cout, tensors[in].div);
        o.prof = b;
        ops.push_back(o);
    }
    void op_simple(OpK k, View in, View out, int C, int pk = 0) {
        Op o{};
        o.k = k; o.in = in; o.out = out; o.C = C; o.pk = pk;
        ops.push_back(o);
    }

    void build(const Json& J, const SafeTensors& st) {
        const Json& layers = J["layers"];
        if (layers.type != Json::Arr || !layers.size()) throw std::runtime_error("detector json: no layers");
        std::vector<int> lout(layers.size(), -1);
        t_input = add_tensor(3, 1);
        op_simple(OpK::Letterbox, View(), View{t_input, 0, -1}, 3);
        bool have_detect = false;
        for (size_t li = 0; li < layers.size(); li++) {
            const Json& L = layers[li];
            std::string type = L["type"].str();
            cur_layer = (int)li;
            auto src = [&](int64_t fi) {
                int64_t j = fi < 0 ? (int64_t)li + fi : fi;
                if (j < 0) return t_input;
                if (j >= (int64_t)li || lout[j] < 0) throw std::runtime_error("detector json: bad 'from' in layer " + std::to_string(li));
                return lout[j];
            };
            const Json& f = L["f"];
            int in = f.type == Json::Num ? src(f.i64()) : -1;
            auto need_in = [&]() { if (in < 0) throw std::runtime_error("detector json: layer " + std::to_string(li) + " needs a single input"); };
            if (type == "Conv") {
                need_in();
                int c = add_conv(L["conv"], st);
                if (tensors[in].C != convs[c].cin) throw std::runtime_error("detector: channel mismatch at layer " + std::to_string(li));
                int out = add_tensor(convs[c].cout, tensors[in].div * convs[c].s);
                op_conv(in, 0, View{out, 0, -1}, c);
                lout[li] = out;
            } else if (type == "C2f") {
                need_in();
                int div = tensors[in].div;
                int ch = (int)L["c"].i64();
                const Json& M = L["m"];
                int n = (int)M.size();
                int cv1 = add_conv(L["cv1"], st);
                if (convs[cv1].cin != tensors[in].C || convs[cv1].cout != 2 * ch || convs[cv1].s != 1)
                    throw std::runtime_error("detector: C2f cv1 mismatch at layer " + std::to_string(li));
                int cat = add_tensor((2 + n) * ch, div);
                op_conv(in, 0, View{cat, 0, -1}, cv1);
                for (int j = 0; j < n; j++) {
                    int b1 = add_conv(M[j]["cv1"], st), b2 = add_conv(M[j]["cv2"], st);
                    if (convs[b1].cin != ch || convs[b2].cout != ch || convs[b1].s != 1 || convs[b2].s != 1)
                        throw std::runtime_error("detector: bottleneck mismatch at layer " + std::to_string(li));
                    int tmp = add_tensor(convs[b1].cout, div);
                    op_conv(cat, (1 + j) * ch, View{tmp, 0, -1}, b1);
                    View res;
                    if (M[j]["add"].b) res = View{cat, (1 + j) * ch, -1};
                    op_conv(tmp, 0, View{cat, (2 + j) * ch, -1}, b2, res);
                }
                int cv2 = add_conv(L["cv2"], st);
                if (convs[cv2].cin != (2 + n) * ch || convs[cv2].s != 1) throw std::runtime_error("detector: C2f cv2 mismatch");
                int out = add_tensor(convs[cv2].cout, div);
                op_conv(cat, 0, View{out, 0, -1}, cv2);
                lout[li] = out;
            } else if (type == "SPPF") {
                need_in();
                int div = tensors[in].div;
                int cv1 = add_conv(L["cv1"], st);
                int c_ = convs[cv1].cout, k = (int)L["k"].i64();
                int cat = add_tensor(4 * c_, div);
                op_conv(in, 0, View{cat, 0, -1}, cv1);
                for (int j = 0; j < 3; j++) op_simple(OpK::MaxPool, View{cat, j * c_, -1}, View{cat, (j + 1) * c_, -1}, c_, k);
                int cv2 = add_conv(L["cv2"], st);
                if (convs[cv2].cin != 4 * c_) throw std::runtime_error("detector: SPPF cv2 mismatch");
                int out = add_tensor(convs[cv2].cout, div);
                op_conv(cat, 0, View{out, 0, -1}, cv2);
                lout[li] = out;
            } else if (type == "Upsample") {
                need_in();
                if (L["scale"].i64() != 2 || tensors[in].div % 2) throw std::runtime_error("detector: only 2x nearest upsampling is supported");
                int out = add_tensor(tensors[in].C, tensors[in].div / 2);
                op_simple(OpK::Up2, View{in, 0, -1}, View{out, 0, -1}, tensors[in].C);
                lout[li] = out;
            } else if (type == "Concat") {
                if (f.type != Json::Arr || L["dim"].i64() != 1) throw std::runtime_error("detector: Concat must be over channels");
                std::vector<int> srcs;
                int C = 0;
                for (auto& x : f.a) { srcs.push_back(src(x.i64())); C += tensors[srcs.back()].C; }
                int div = tensors[srcs[0]].div;
                for (int s : srcs) if (tensors[s].div != div) throw std::runtime_error("detector: Concat of different resolutions");
                int out = add_tensor(C, div);
                int c0 = 0;
                for (int s : srcs) { op_simple(OpK::Copy, View{s, 0, -1}, View{out, c0, -1}, tensors[s].C); c0 += tensors[s].C; }
                lout[li] = out;
            } else if (type == "Detect") {
                if (f.type != Json::Arr || f.size() > 4) throw std::runtime_error("detector: Detect needs 1-4 input levels");
                nc = (int)L["nc"].i64();
                reg_max = (int)L["reg_max"].i64();
                no = 4 * reg_max + nc;
                t_raw = add_tensor(no, 0, T_RAW);
                for (size_t l = 0; l < f.size(); l++) {
                    int T = src(f[l].i64());
                    float s = (float)L["stride"][l].num();
                    if ((float)tensors[T].div != s) throw std::runtime_error("detector: Detect stride does not match feature resolution");
                    lvl_tensor.push_back(T);
                    lvl_stride.push_back(s);
                    max_stride = std::max(max_stride, (int)s);
                    auto chain = [&](const Json& seq, int c0, int cout_last) {
                        int cur = T;
                        for (size_t j = 0; j < seq.size(); j++) {
                            int c = add_conv(seq[j], st);
                            if (convs[c].s != 1 || convs[c].cin != tensors[cur].C) throw std::runtime_error("detector: Detect conv mismatch");
                            if (j + 1 < seq.size()) {
                                int o = add_tensor(convs[c].cout, tensors[cur].div);
                                op_conv(cur, 0, View{o, 0, -1}, c);
                                cur = o;
                            } else {
                                if (convs[c].cout != cout_last) throw std::runtime_error("detector: Detect head width mismatch");
                                op_conv(cur, 0, View{t_raw, c0, (int)l}, c);
                            }
                        }
                    };
                    chain(L["cv2"][l], 0, 4 * reg_max);
                    chain(L["cv3"][l], 4 * reg_max, nc);
                }
                t_dec = add_tensor(4 + nc, 0, T_DEC);
                op_simple(OpK::Decode, View{t_raw, 0, -1}, View{t_dec, 0, -1}, no);
                lout[li] = t_dec;
                have_detect = true;
                if (li + 1 != layers.size()) throw std::runtime_error("detector: Detect must be the last layer");
            } else {
                throw std::runtime_error("detector: unsupported layer type " + type);
            }
        }
        if (!have_detect) throw std::runtime_error("detector: no Detect layer");
        // liveness over op indices; the decoded tensor stays live past the last op (host copy)
        for (int i = 0; i < (int)ops.size(); i++) {
            for (const View* v : {&ops[i].in, &ops[i].out, &ops[i].res}) {
                if (v->t < 0) continue;
                tensors[v->t].first = std::min(tensors[v->t].first, i);
                tensors[v->t].last = std::max(tensors[v->t].last, i);
            }
        }
        tensors[t_dec].last = (int)ops.size();
    }

    // ---- per-input-size memory plan ----
    struct Plan {
        int h = 0, w = 0, N = 0;
        Levels L{};
        std::vector<size_t> off, size;
        size_t act_bytes = 0, col_floats = 0, bytes = 0;   // col: im2col scratch (fp32 path only)
    };

    size_t tensor_floats(const TDesc& d, int h, int w, int N) const {
        if (d.kind == T_NORMAL) return (size_t)d.C * (h / d.div) * (w / d.div);
        return (size_t)d.C * N;
    }

    Plan plan(int h, int w, size_t col_budget, bool debug) const {
        Plan p;
        // debug keeps the letterboxed input alive to the end for the host copy
        auto first = [&](int t) { return tensors[t].first; };
        auto last = [&](int t) { return debug && t == t_input ? (int)ops.size() : tensors[t].last; };
        p.h = h; p.w = w;
        p.L.n = (int)lvl_tensor.size();
        for (int l = 0; l < p.L.n; l++) {
            int div = tensors[lvl_tensor[l]].div;
            p.L.off[l] = p.N;
            p.L.h[l] = h / div; p.L.w[l] = w / div;
            p.L.stride[l] = lvl_stride[l];
            p.N += p.L.h[l] * p.L.w[l];
        }
        size_t nt = tensors.size();
        p.off.assign(nt, 0);
        p.size.resize(nt);
        for (size_t t = 0; t < nt; t++) p.size[t] = (tensor_floats(tensors[t], h, w, p.N) * 4 + 255) & ~(size_t)255;
        // greedy by size: lowest offset that does not collide with a placed tensor live at the same time
        std::vector<int> order(nt);
        for (size_t t = 0; t < nt; t++) order[t] = (int)t;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p.size[a] > p.size[b]; });
        std::vector<int> placed;
        for (int t : order) {
            std::vector<std::pair<size_t, size_t>> busy;
            for (int u : placed)
                if (!(last(u) < first(t) || last(t) < first(u)))
                    busy.push_back({p.off[u], p.off[u] + p.size[u]});
            std::sort(busy.begin(), busy.end());
            size_t cand = 0;
            for (auto& [a, b] : busy) {
                if (cand + p.size[t] <= a) break;
                cand = std::max(cand, b);
            }
            p.off[t] = cand;
            placed.push_back(t);
            p.act_bytes = std::max(p.act_bytes, cand + p.size[t]);
        }
        // fp32 path only: im2col scratch = the largest column matrix, capped by the budget (convs
        // then run in pixel chunks). The fp16 path gathers its columns on the fly.
        size_t need = 0, row_min = 0;
        for (const Op& o : ops) {
            if (w_fp16 || o.k != OpK::Conv) continue;
            const ConvW& c = convs[o.conv];
            if (c.k == 1 && c.s == 1 && c.p == 0) continue;
            const TDesc& d = tensors[o.in.t];
            int Hi = h / d.div, Wi = w / d.div;
            int Ho = (Hi + 2 * c.p - c.k) / c.s + 1, Wo = (Wi + 2 * c.p - c.k) / c.s + 1;
            size_t K = (size_t)c.cin * c.k * c.k;
            need = std::max(need, K * Ho * Wo);
            row_min = std::max(row_min, K * Wo);
        }
        p.col_floats = w_fp16 ? 0 : std::max(row_min, std::min(need, col_budget / 4));
        p.bytes = p.act_bytes + ((p.col_floats * 4 + 255) & ~(size_t)255);
        return p;
    }

    void conv(const ConvW& c, const float* in, int H, int W, float* out, int ldo, const float* res,
              float* col, size_t col_floats, const char* prof_name) const {
        int Ho = (H + 2 * c.p - c.k) / c.s + 1, Wo = (W + 2 * c.p - c.k) / c.s + 1;
        int P = Ho * Wo, K = c.cin * c.k * c.k;
        if (w_fp16) {
            ProfScope ps(prof_name ? prof_name : c.k == 1 ? "det_conv1x1" : "det_conv3x3");
            ig::launch(ig::pick_tm(c.cout), c.k, out, ldo, in, (const __half*)wstore + c.w, bstore + c.b, res,
                       H, W, c.cout, Wo, P, c.s, c.p, K, c.silu);
            return;
        }
        const float* wf = (const float*)wstore + c.w;
        if (c.k == 1 && c.s == 1 && c.p == 0) {
            ProfScope ps("det_gemm1x1");
            gemm(false, false, c.cout, P, K, 1.f, wf, K, in, P, 0.f, out, ldo);
        } else {
            int Pc = (int)std::min<size_t>(P, col_floats / K);
            for (int p0 = 0; p0 < P; p0 += Pc) {
                int n = std::min(Pc, P - p0);
                {
                    ProfScope ps("det_im2col");
                    k_im2col<<<dim3(nb(n), K), 256, 0, G.stream>>>(col, in, H, W, c.k, c.s, c.p, Wo, p0, n);
                }
                ProfScope ps("det_gemm3x3");
                gemm(false, false, c.cout, n, K, 1.f, wf, K, col, n, 0.f, out + p0, ldo);
            }
        }
        ProfScope ps("det_epilogue");
        k_bias_act<<<nb((size_t)c.cout * P), 256, 0, G.stream>>>(out, ldo, bstore + c.b, c.cout, P, c.silu ? 1 : 0, res);
    }

    void free_all() {
        if (host_block) {
            cudaFreeHost(host_block);
        } else {
            if (wstore) cudaFree(wstore);
            if (bstore) cudaFree(bstore);
        }
        host_block = nullptr; wstore = nullptr; bstore = nullptr;
        if (ev0) { cudaEventDestroy(ev0); ev0 = nullptr; }
        if (ev1) { cudaEventDestroy(ev1); ev1 = nullptr; }
    }
    ~Impl() { free_all(); }
};

FaceDetector::FaceDetector() = default;
FaceDetector::~FaceDetector() = default;

bool FaceDetector::loaded() const { return impl && impl->wstore; }
size_t FaceDetector::weight_bytes() const { return impl ? impl->w_bytes : 0; }
bool FaceDetector::weights_on_host() const { return impl && impl->w_host; }
void FaceDetector::unload() { impl.reset(); }

void FaceDetector::load(const std::string& st_path, const std::string& json_path, Place place) {
    impl.reset();
    auto I = std::make_unique<Impl>();
    I->w_fp16 = !fp32_weights;
    Json J = Json::parse(read_file(json_path));
    if (J["format"].str() != "kiln-yolov8-det/1") throw std::runtime_error("detector: unexpected json format in " + json_path);
    SafeTensors st(st_path);
    I->build(J, st);
    nc = I->nc;
    imgsz = (int)J["imgsz"].i64(640);
    names.assign(nc, "");
    for (int c = 0; c < nc; c++) names[c] = J["names"][std::to_string(c)].str();

    // weights + biases: in VRAM, or (VRAM short / Place::Host) in one mapped pinned host block that
    // the kernels read over PCIe. The host block is filled with a plain memcpy (it is host memory).
    const void* src = I->w_fp16 ? (const void*)I->hw16.data() : (const void*)I->hw32.data();
    I->w_bytes = I->w_fp16 ? I->hw16.size() * 2 : I->hw32.size() * 4;
    const size_t b_bytes = I->hb.size() * 4, w_span = (I->w_bytes + 255) & ~(size_t)255;
    bool device = place == Place::Device ||
                  (place == Place::Auto && gpu_free_bytes() > I->w_bytes + b_bytes + G.reserve_bytes);
    if (device) {
        CK(cudaMalloc(&I->wstore, I->w_bytes));
        CK(cudaMalloc(&I->bstore, b_bytes));
        CK(cudaMemcpy(I->wstore, src, I->w_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(I->bstore, I->hb.data(), b_bytes, cudaMemcpyHostToDevice));
    } else {
        CK(cudaHostAlloc(&I->host_block, w_span + b_bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        memcpy(I->host_block, src, I->w_bytes);
        memcpy((char*)I->host_block + w_span, I->hb.data(), b_bytes);
        void* d = nullptr;
        CK(cudaHostGetDevicePointer(&d, I->host_block, 0));
        I->wstore = d;
        I->bstore = (float*)((char*)d + w_span);
        I->w_host = true;
    }
    std::vector<__half>().swap(I->hw16);
    std::vector<float>().swap(I->hw32);
    std::vector<float>().swap(I->hb);
    CK(cudaEventCreate(&I->ev0));
    CK(cudaEventCreate(&I->ev1));
    impl = std::move(I);
}

std::vector<Box> FaceDetector::detect(const float* rgb, int H, int W, float conf, float iou) {
    if (!loaded()) throw std::runtime_error("detector: not loaded");
    if (H <= 0 || W <= 0) throw std::runtime_error("detector: empty image");
    Impl& I = *impl;
    auto t0 = std::chrono::steady_clock::now();

    // Ultralytics LetterBox(auto=True, scaleup=True, center=True); Python round() is half-to-even = nearbyint
    double r = std::min((double)imgsz / H, (double)imgsz / W);
    int nw = (int)std::nearbyint(W * r), nh = (int)std::nearbyint(H * r);
    int dw = (imgsz - nw) % I.max_stride, dh = (imgsz - nh) % I.max_stride;
    int top = (int)std::nearbyint(dh / 2.0 - 0.1), bottom = (int)std::nearbyint(dh / 2.0 + 0.1);
    int left = (int)std::nearbyint(dw / 2.0 - 0.1), right = (int)std::nearbyint(dw / 2.0 + 0.1);
    int lh = nh + top + bottom, lw = nw + left + right;
    if (lh % I.max_stride || lw % I.max_stride) throw std::runtime_error("detector: letterbox is not stride-aligned");

    Impl::Plan p = I.plan(lh, lw, col_budget, keep_debug);
    in_h = lh; in_w = lw; num_anchors = p.N;
    last_arena_bytes = p.bytes;

    ArenaGuard guard;
    char* base = (char*)G.arena.f(p.bytes / 4 + 64);
    float* col = (float*)(base + p.act_bytes);
    auto tptr = [&](const View& v) -> float* {
        const TDesc& d = I.tensors[v.t];
        float* b = (float*)(base + p.off[v.t]);
        if (d.kind == T_NORMAL) return b + (size_t)v.c0 * (lh / d.div) * (lw / d.div);
        return b + (size_t)v.c0 * p.N + (v.lvl >= 0 ? p.L.off[v.lvl] : 0);
    };

    static const bool prof_layers = getenv("KILN_DET_PROF_LAYERS") != nullptr;  // per-conv --profile rows
    CK(cudaEventRecord(I.ev0, G.stream));
    for (const Op& o : I.ops) {
        switch (o.k) {
            case OpK::Letterbox: {
                ProfScope ps("det_misc");
                size_t n = (size_t)3 * lh * lw;
                k_letterbox<<<nb(n), 256, 0, G.stream>>>(tptr(o.out), rgb, H, W, lh, lw, top, left, nh, nw,
                                                          (float)((double)H / nh), (float)((double)W / nw));
                break;
            }
            case OpK::Conv: {
                const ConvW& c = I.convs[o.conv];
                const TDesc& di = I.tensors[o.in.t];
                const TDesc& dout = I.tensors[o.out.t];
                int Hi = lh / di.div, Wi = lw / di.div;
                int Ho = (Hi + 2 * c.p - c.k) / c.s + 1, Wo = (Wi + 2 * c.p - c.k) / c.s + 1;
                int ldo;
                if (dout.kind == T_NORMAL) {
                    if (Ho != lh / dout.div || Wo != lw / dout.div) throw std::runtime_error("detector: conv output size mismatch");
                    ldo = Ho * Wo;
                } else {
                    if (Ho != p.L.h[o.out.lvl] || Wo != p.L.w[o.out.lvl]) throw std::runtime_error("detector: head size mismatch");
                    ldo = p.N;
                }
                I.conv(c, tptr(o.in), Hi, Wi, tptr(o.out), ldo, o.res.t >= 0 ? tptr(o.res) : nullptr, col, p.col_floats,
                       prof_layers ? o.prof.c_str() : nullptr);
                break;
            }
            case OpK::MaxPool: {
                ProfScope ps("det_misc");
                const TDesc& d = I.tensors[o.in.t];
                int h = lh / d.div, w = lw / d.div;
                size_t n = (size_t)o.C * h * w;
                k_maxpool<<<nb(n), 256, 0, G.stream>>>(tptr(o.out), tptr(o.in), o.C, h, w, o.pk);
                break;
            }
            case OpK::Up2: {
                ProfScope ps("det_misc");
                const TDesc& d = I.tensors[o.in.t];
                int h = lh / d.div, w = lw / d.div;
                size_t n = (size_t)o.C * h * w * 4;
                k_up2<<<nb(n), 256, 0, G.stream>>>(tptr(o.out), tptr(o.in), o.C, h, w);
                break;
            }
            case OpK::Copy: {
                ProfScope ps("det_misc");
                const TDesc& d = I.tensors[o.in.t];
                size_t n = (size_t)o.C * (lh / d.div) * (lw / d.div);
                CK(cudaMemcpyAsync(tptr(o.out), tptr(o.in), n * 4, cudaMemcpyDeviceToDevice, G.stream));
                break;
            }
            case OpK::Decode: {
                ProfScope ps("det_misc");
                k_decode<<<nb(p.N, 128), 128, 0, G.stream>>>(tptr(o.out), tptr(o.in), p.N, I.nc, I.reg_max, p.L);
                break;
            }
        }
    }
    CK(cudaGetLastError());
    const int nco = 4 + I.nc, N = p.N;
    std::vector<float> dec((size_t)nco * N);
    CK(cudaMemcpyAsync(dec.data(), tptr(View{I.t_dec, 0, -1}), dec.size() * 4, cudaMemcpyDeviceToHost, G.stream));
    CK(cudaEventRecord(I.ev1, G.stream));
    if (keep_debug) {
        dbg_input.resize((size_t)3 * lh * lw);
        dbg_raw.resize((size_t)I.no * N);
        CK(cudaMemcpyAsync(dbg_input.data(), tptr(View{I.t_input, 0, -1}), dbg_input.size() * 4, cudaMemcpyDeviceToHost, G.stream));
        CK(cudaMemcpyAsync(dbg_raw.data(), tptr(View{I.t_raw, 0, -1}), dbg_raw.size() * 4, cudaMemcpyDeviceToHost, G.stream));
    }
    gpu_sync();
    CK(cudaEventElapsedTime(&last_gpu_ms, I.ev0, I.ev1));
    if (keep_debug) dbg_decoded = dec;

    // ---- Ultralytics non_max_suppression (multi_label=False, agnostic=False) + scale_boxes ----
    struct Cand { float x0, y0, x1, y1, s; int c; };
    std::vector<Cand> cand;
    for (int a = 0; a < N; a++) {
        int bc = 0;
        float bs = dec[(size_t)4 * N + a];
        for (int c = 1; c < I.nc; c++) { float s = dec[(size_t)(4 + c) * N + a]; if (s > bs) { bs = s; bc = c; } }
        if (!(bs > conf)) continue;
        float cx = dec[a], cy = dec[(size_t)N + a], bw = dec[(size_t)2 * N + a], bh = dec[(size_t)3 * N + a];
        cand.push_back({cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bs, bc});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    if (cand.size() > 30000) cand.resize(30000);
    std::vector<char> dead(cand.size(), 0);
    std::vector<Box> out;
    double gain = std::min((double)lh / H, (double)lw / W);
    double padx = std::nearbyint((lw - W * gain) / 2 - 0.1), pady = std::nearbyint((lh - H * gain) / 2 - 0.1);
    for (size_t i = 0; i < cand.size() && (int)out.size() < max_det; i++) {
        if (dead[i]) continue;
        const Cand& a = cand[i];
        float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
        for (size_t j = i + 1; j < cand.size(); j++) {
            if (dead[j] || cand[j].c != a.c) continue;
            const Cand& b = cand[j];
            float iw = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
            float ih = std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
            float inter = iw * ih;
            float u = area_a + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
            if (inter / u > iou) dead[j] = 1;
        }
        Box bx;
        bx.x0 = (float)std::clamp((a.x0 - padx) / gain, 0.0, (double)W);
        bx.y0 = (float)std::clamp((a.y0 - pady) / gain, 0.0, (double)H);
        bx.x1 = (float)std::clamp((a.x1 - padx) / gain, 0.0, (double)W);
        bx.y1 = (float)std::clamp((a.y1 - pady) / gain, 0.0, (double)H);
        bx.score = a.s;
        out.push_back(bx);
    }
    last_total_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}
