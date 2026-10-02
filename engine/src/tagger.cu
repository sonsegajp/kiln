#include "tagger.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "image_io.h"
#include "json.h"
#include "safetensors.h"
#include "sdops.h"

static inline unsigned nblk(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// [3, S, S] -> patches [g*g, 3*P*P], each row ordered (channel, ky, kx) like the conv weight it multiplies
__global__ void k_im2patch(float* out, const float* img, int S, int P, int g) {
    const size_t n = (size_t)g * g * 3 * P * P;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int K = 3 * P * P, t = (int)(i / K), r = (int)(i % K);
    const int c = r / (P * P), ky = (r / P) % P, kx = r % P, py = t / g, px = t % g;
    out[i] = img[(size_t)c * S * S + (size_t)(py * P + ky) * S + (px * P + kx)];
}

__global__ void k_gelu_tanh(float* x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = x[i]; x[i] = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))); }
}

// x [T, ld] rows npre.. : interleaved pairs of every head rotated by the token's angles (tables [T - npre, 64])
__global__ void k_rope_eva(float* x, int ld, int T, int npre, int H, const float* sn, const float* cs) {
    const int pairs = H * 32;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)(T - npre) * pairs) return;
    const int t = (int)(i / pairs), p = (int)(i % pairs), h = p / 32, j = p % 32;
    float* v = x + (size_t)(t + npre) * ld + h * 64 + 2 * j;
    const float s = sn[(size_t)t * 64 + 2 * j], c = cs[(size_t)t * 64 + 2 * j];
    const float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = b * c + a * s;
}

// y [D] = mean of x rows [r0, T)
__global__ void k_mean_rows(float* y, const float* x, int r0, int T, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float s = 0.f;
    for (int t = r0; t < T; t++) s += x[(size_t)t * D + d];
    y[d] = s / (float)(T - r0);
}

__global__ void k_add_rows(float* x, const float* pos, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += pos[i];
}

static Linear16 lin(const SafeTensors& st, const std::string& k, Place place) { return load_linear16(st, k, place); }

void Tagger::load(const std::string& dir, Place place) {
    free();
    std::ifstream cf(dir + "\\config.json");
    if (!cf) throw std::runtime_error("tagger: no config.json in " + dir);
    std::stringstream ss;
    ss << cf.rdbuf();
    const Json cfg = Json::parse(ss.str());
    SafeTensors st(dir + "\\model.safetensors");
    eva_ = st.has("cls_token");
    const StTensor& pw = st.get("patch_embed.proj.weight");  // [D, 3, P, P]
    dim_ = (int)pw.shape[0];
    patch_px_ = (int)pw.shape[2];
    heads_ = dim_ / 64;
    size_ = (int)cfg["pretrained_cfg"]["input_size"][2].i64(448);
    grid_ = size_ / patch_px_;
    {  // the patch conv as a linear over flattened patches
        StTensor flat = pw;
        flat.shape = {pw.shape[0], pw.shape[1] * pw.shape[2] * pw.shape[3]};
        patch_.w = upload_weight(flat, place, false, true);
        patch_.b = upload_f32(st.get("patch_embed.proj.bias"));
    }
    pos_ = upload_f32(st.get("pos_embed"));
    if (eva_) cls_ = upload_f32(st.get("cls_token"));
    for (int i = 0; st.has("blocks." + std::to_string(i) + ".norm1.weight"); i++) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        Block b;
        b.n1 = load_norm32(st, p + "norm1");
        b.n2 = load_norm32(st, p + "norm2");
        if (st.has(p + "attn.qkv.weight")) b.qkv = lin(st, p + "attn.qkv", place);
        else {
            b.q = lin(st, p + "attn.q_proj", place);
            b.k = lin(st, p + "attn.k_proj", place);
            b.v = lin(st, p + "attn.v_proj", place);
        }
        if (st.has(p + "attn.norm.weight")) throw std::runtime_error("tagger: attention inner norm is not supported");
        if (st.has(p + "ls1.gamma")) throw std::runtime_error("tagger: layer scale is not supported");
        b.proj = lin(st, p + "attn.proj", place);
        if (st.has(p + "mlp.fc1_g.weight")) {
            b.fc1_g = lin(st, p + "mlp.fc1_g", place);
            b.fc1_x = lin(st, p + "mlp.fc1_x", place);
            if (st.has(p + "mlp.norm.weight")) b.mlp_norm = load_norm32(st, p + "mlp.norm");
            hidden_ = (int)b.fc1_g.w.rows;
        } else {
            b.fc1 = lin(st, p + "mlp.fc1", place);
            hidden_ = (int)b.fc1.w.rows;
        }
        b.fc2 = lin(st, p + "mlp.fc2", place);
        blocks_.push_back(std::move(b));
    }
    if (st.has("norm.weight")) norm_ = load_norm32(st, "norm");
    if (st.has("fc_norm.weight")) fc_norm_ = load_norm32(st, "fc_norm");
    head_ = lin(st, "head", place);
    classes_ = (int)head_.w.rows;
    if (eva_) {  // timm RotaryEmbeddingCat(in_pixels=False, ref_feat_shape): y bands then x bands, pairs interleaved
        const Json& ref = cfg["model_args"]["ref_feat_shape"];
        const float ry = (float)ref[0].num(grid_), rx = ref.size() > 1 ? (float)ref[1].num(grid_) : ry;
        const int nb = 16;  // head dim 64 / 4
        std::vector<float> sn((size_t)grid_ * grid_ * 64), cs(sn.size());
        for (int y = 0; y < grid_; y++)
            for (int x = 0; x < grid_; x++) {
                const float ty = (float)y / grid_ * ry, tx = (float)x / grid_ * rx;
                float* s = &sn[((size_t)y * grid_ + x) * 64];
                float* c = &cs[((size_t)y * grid_ + x) * 64];
                for (int k = 0; k < 2 * nb; k++) {
                    const float band = 1.f / powf(10000.f, (float)(k % nb) / nb);
                    const float a = (k < nb ? ty : tx) * band;
                    s[2 * k] = s[2 * k + 1] = sinf(a);
                    c[2 * k] = c[2 * k + 1] = cosf(a);
                }
            }
        CK(cudaMalloc(&rsin_, sn.size() * 4));
        CK(cudaMalloc(&rcos_, cs.size() * 4));
        CK(cudaMemcpy(rsin_, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(rcos_, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
    }
    dir_ = dir;
}

void Tagger::free() {
    auto fl = [](Linear16& l) { if (l.w.p || l.w.host) free_linear16(l); };
    for (auto& b : blocks_) {
        free_norm32(b.n1);
        free_norm32(b.n2);
        if (b.mlp_norm.w) free_norm32(b.mlp_norm);
        for (Linear16* l : {&b.qkv, &b.q, &b.k, &b.v, &b.proj, &b.fc1, &b.fc1_g, &b.fc1_x, &b.fc2}) fl(*l);
    }
    blocks_.clear();
    fl(patch_);
    fl(head_);
    for (float** p : {&pos_, &cls_, &rsin_, &rcos_})
        if (*p) { cudaFree(*p); *p = nullptr; }
    if (norm_.w) free_norm32(norm_);
    if (fc_norm_.w) free_norm32(fc_norm_);
    classes_ = 0;
    dir_.clear();
}

void Tagger::run(const float* img, std::vector<float>& probs) {
    ProfScope ps("tagger");
    const int D = dim_, P = patch_px_, g = grid_, NP = g * g, npre = eva_ ? 1 : 0, T = NP + npre, F = hidden_;
    const size_t m = G.arena.mark();
    float* patches = G.arena.f((size_t)NP * 3 * P * P);
    k_im2patch<<<nblk((size_t)NP * 3 * P * P), 256, 0, G.stream>>>(patches, img, size_, P, g);
    float* x = G.arena.f((size_t)T * D);
    if (eva_) CK(cudaMemcpyAsync(x, cls_, (size_t)D * 4, cudaMemcpyDeviceToDevice, G.stream));
    linear(x + (size_t)npre * D, patches, NP, patch_.w);
    bias_rows(x + (size_t)npre * D, patch_.b, NP, D);
    k_add_rows<<<nblk((size_t)T * D), 256, 0, G.stream>>>(x, pos_, (size_t)T * D);
    float* a = G.arena.f((size_t)T * D);
    float* qkv = G.arena.f((size_t)T * 3 * D);
    float* o = G.arena.f((size_t)T * D);
    float* h = G.arena.f((size_t)T * F);
    float* u = eva_ ? G.arena.f((size_t)T * F) : nullptr;
    for (const Block& b : blocks_) {
        layer_norm(a, x, b.n1.w, b.n1.b, T, D, 1e-6f);
        const float *q, *k, *v;
        int ld;
        if (b.qkv.w.p || b.qkv.w.host) {
            linear(qkv, a, T, b.qkv.w);
            bias_rows(qkv, b.qkv.b, T, 3 * D);
            q = qkv; k = qkv + D; v = qkv + 2 * D; ld = 3 * D;
        } else {
            float* qq = qkv;
            float* kk = qkv + (size_t)T * D;
            float* vv = qkv + (size_t)2 * T * D;
            linear(qq, a, T, b.q.w); if (b.q.b) bias_rows(qq, b.q.b, T, D);
            linear(kk, a, T, b.k.w); if (b.k.b) bias_rows(kk, b.k.b, T, D);
            linear(vv, a, T, b.v.w); if (b.v.b) bias_rows(vv, b.v.b, T, D);
            q = qq; k = kk; v = vv; ld = D;
        }
        if (eva_) {
            const size_t n = (size_t)NP * heads_ * 32;
            k_rope_eva<<<nblk(n), 256, 0, G.stream>>>((float*)q, ld, T, npre, heads_, rsin_, rcos_);
            k_rope_eva<<<nblk(n), 256, 0, G.stream>>>((float*)k, ld, T, npre, heads_, rsin_, rcos_);
        }
        attention(o, D, q, ld, k, ld, v, ld, T, T, heads_, 64, false);
        linear(x, o, T, b.proj.w, nullptr, 1.f);  // x += proj(o)
        bias_rows(x, b.proj.b, T, D);
        layer_norm(a, x, b.n2.w, b.n2.b, T, D, 1e-6f);
        if (eva_) {  // SwiGLU: silu(fc1_g a) * fc1_x a, normed, then fc2
            linear(h, a, T, b.fc1_g.w);
            bias_rows(h, b.fc1_g.b, T, F);
            linear(u, a, T, b.fc1_x.w);
            bias_rows(u, b.fc1_x.b, T, F);
            silu_mul(h, u, (size_t)T * F);
            if (b.mlp_norm.w) layer_norm(h, h, b.mlp_norm.w, b.mlp_norm.b, T, F, 1e-6f);
        } else {
            linear(h, a, T, b.fc1.w);
            bias_rows(h, b.fc1.b, T, F);
            k_gelu_tanh<<<nblk((size_t)T * F), 256, 0, G.stream>>>(h, (size_t)T * F);
        }
        linear(x, h, T, b.fc2.w, nullptr, 1.f);
        bias_rows(x, b.fc2.b, T, D);
    }
    if (norm_.w) layer_norm(x, x, norm_.w, norm_.b, T, D, 1e-6f);
    float* pooled = G.arena.f(D);
    k_mean_rows<<<nblk(D), 256, 0, G.stream>>>(pooled, x, npre, T, D);
    if (fc_norm_.w) layer_norm(pooled, pooled, fc_norm_.w, fc_norm_.b, 1, D, 1e-6f);
    float* logits = G.arena.f(classes_);
    linear(logits, pooled, 1, head_.w);
    bias_rows(logits, head_.b, 1, classes_);
    probs.resize(classes_);
    CK(cudaMemcpy(probs.data(), logits, (size_t)classes_ * 4, cudaMemcpyDeviceToHost));
    G.arena.release(m);
    for (float& p : probs) p = 1.f / (1.f + expf(-p));
}

// PIL's resampling (ImagingResample): separable, the kernel widened by the scale when shrinking, 8-bit rounding
// after each pass. Bicubic with a = -0.5.
static float bicubic(float x) {
    const float a = -0.5f;
    x = fabsf(x);
    if (x < 1.f) return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f) return (((x - 5.f) * x + 8.f) * x - 4.f) * a;
    return 0.f;
}
static void resample_axis(const std::vector<uint8_t>& in, int inN, int other, bool horiz, int outN, std::vector<uint8_t>& out) {
    const double scale = (double)inN / outN, fs = std::max(scale, 1.0), support = 2.0 * fs;
    std::vector<int> lo(outN), cnt(outN);
    std::vector<std::vector<float>> wts(outN);
    for (int o = 0; o < outN; o++) {
        const double center = (o + 0.5) * scale;
        int xmin = std::max((int)(center - support + 0.5), 0), xmax = std::min((int)(center + support + 0.5), inN);
        std::vector<float> w(xmax - xmin);
        double tot = 0;
        for (int i = 0; i < xmax - xmin; i++) { w[i] = bicubic((float)((i + xmin - center + 0.5) / fs)); tot += w[i]; }
        for (float& v : w) v = tot != 0 ? (float)(v / tot) : 0.f;
        lo[o] = xmin;
        cnt[o] = xmax - xmin;
        wts[o] = std::move(w);
    }
    // in/out are [rows][cols][3] interleaved
    const int ow = horiz ? outN : other, oh = horiz ? other : outN, iw = horiz ? inN : other;
    out.assign((size_t)ow * oh * 3, 0);
    for (int r = 0; r < (horiz ? other : outN); r++)
        for (int c = 0; c < (horiz ? outN : other); c++)
            for (int ch = 0; ch < 3; ch++) {
                const int o = horiz ? c : r;
                float s = 0.f;
                for (int i = 0; i < cnt[o]; i++) {
                    const int src = lo[o] + i;
                    const uint8_t v = horiz ? in[((size_t)r * iw + src) * 3 + ch] : in[((size_t)src * iw + c) * 3 + ch];
                    s += v * wts[o][i];
                }
                out[((size_t)r * ow + c) * 3 + ch] = (uint8_t)std::min(255.f, std::max(0.f, floorf(s + 0.5f)));
            }
}

bool tagger_preprocess(const std::string& path, int size, std::vector<float>& out, std::string& err) {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    if (!load_image_rgb(path, w, h, rgb, err)) return false;  // alpha is composited onto white
    const int s = std::max(w, h), ox = (s - w) / 2, oy = (s - h) / 2;
    std::vector<uint8_t> sq((size_t)s * s * 3, 255);
    for (int y = 0; y < h; y++) memcpy(&sq[((size_t)(y + oy) * s + ox) * 3], &rgb[(size_t)y * w * 3], (size_t)w * 3);
    std::vector<uint8_t> a, b;
    const std::vector<uint8_t>* img = &sq;
    if (s != size) {
        resample_axis(sq, s, s, true, size, a);    // width first, as PIL does
        resample_axis(a, s, size, false, size, b);  // then height
        img = &b;
    }
    out.resize((size_t)3 * size * size);
    const size_t P = (size_t)size * size;
    for (size_t p = 0; p < P; p++)
        for (int c = 0; c < 3; c++) out[(size_t)(2 - c) * P + p] = (*img)[p * 3 + c] / 127.5f - 1.f;  // RGB -> BGR planes
    return true;
}
