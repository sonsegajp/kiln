#include "sdvae.h"

#include <algorithm>

#include "sdops.h"
#include "vaeconv.cuh"

namespace {

// Inputs that are raw residual-stream values (shortcut and resample convs; no norm before them) are
// scaled down before the fp16 conversion: this VAE's activations exceed fp16's range.
constexpr float RAW = 1.f / 64.f;
constexpr size_t BAND_BYTES = (size_t)160 << 20;  // arena left for band temporaries

// A full activation [C, H, W]: in the arena when it fits, else in mapped pinned system RAM (the kernels
// stream it over PCIe).
struct Tens {
    float* p = nullptr;
    void* host = nullptr;
    int C = 0, H = 0, W = 0;
    size_t P() const { return (size_t)H * W; }
};

Tens make(int C, int H, int W) {
    Tens t;
    t.C = C; t.H = H; t.W = W;
    const size_t bytes = (size_t)C * H * W * 4;
    if (G.arena.free_bytes() > bytes + BAND_BYTES) {
        t.p = G.arena.f(bytes / 4);
    } else {
        void* d;
        CK(cudaHostAlloc(&t.host, bytes, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&d, t.host, 0));
        t.p = (float*)d;
    }
    return t;
}
void drop(Tens& t) {
    if (t.host) {
        gpu_sync();
        CK(cudaFreeHost(t.host));
    }
    t = Tens();
}

// rows of output computed per band: the fp16 input band [Cin, rows + halo, W] fits BAND_BYTES
int band_rows(int Cin, int W, int in_per_out) {
    size_t per = (size_t)Cin * W * 2 * in_per_out;
    return (int)std::max<size_t>(8, BAND_BYTES / 2 / per);
}

// out = conv(in') + bias (+ res), where in' = silu(groupnorm(in)) with `stat` or the raw input scaled by RAW.
// All operands are full tensors; the input is converted band by band.
void conv_layer(Tens& out, const Tens& in, const Conv16& c, const float2* stat, const Norm32* n, const Tens* res) {
    const int H = in.H, W = in.W, mode = c.mode;
    const int grid_rows = (mode == vc::MS2 || mode == vc::MS2P) ? out.H : H;  // output rows (low-res rows for MUP)
    const int step = (mode == vc::MS2 || mode == vc::MS2P) ? 2 : 1;
    const int rows = band_rows(c.cin, W, step);
    for (int r0 = 0; r0 < grid_rows; r0 += rows) {
        const int r1 = std::min(grid_rows, r0 + rows);
        int i0, i1;  // input rows the band reads
        if (mode == vc::M1) { i0 = r0; i1 = r1; }
        else if (mode == vc::MS2) { i0 = 2 * r0; i1 = std::min(H, 2 * r1 + 1); }
        else if (mode == vc::MS2P) { i0 = std::max(0, 2 * r0 - 1); i1 = std::min(H, 2 * r1 + 1); }
        else { i0 = std::max(0, r0 - 1); i1 = std::min(H, r1 + 1); }
        const int row_off = (step == 2 ? 2 * r0 : r0) - i0;
        const size_t n_in = (size_t)(i1 - i0) * W;
        size_t m = G.arena.mark();
        __half* b16 = (__half*)G.arena.f(((size_t)c.cin * n_in + 1) / 2);
        if (stat) group_norm_apply16(b16, in.p + (size_t)i0 * W, in.P(), stat, n->w, n->b, c.cin, n_in, 32, true);
        else to16_band(b16, in.p + (size_t)i0 * W, in.P(), c.cin, n_in, RAW);
        const size_t o_off = mode == vc::MUP ? (size_t)(2 * r0) * (2 * W) : (size_t)r0 * out.W;
        vc::run_band(out.p + o_off, out.P(), res ? res->p + o_off : nullptr, b16, c.cin, i1 - i0, W, row_off, r1 - r0, c.w, c.cout, c.b, mode,
                     stat ? 1.f : 1.f / RAW);
        G.arena.release(m);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------
static Conv16 load_conv16(const SafeTensors& st, const std::string& k, int mode, Place place, int pad_cout = 0) {
    const StTensor& t0 = st.get(k + ".weight");
    std::vector<uint16_t> padded;
    StTensor t = t0;
    if (pad_cout > (int)t0.shape[0]) {  // zero output channels up to a tile-friendly count
        const size_t per = (size_t)t0.numel() / t0.shape[0];
        padded.assign((size_t)pad_cout * per, 0);
        for (size_t i = 0; i < (size_t)t0.numel(); i++) padded[i] = __half_as_ushort(__float2half_rn(st_elem_f32(t0, (int64_t)i)));
        t.dtype = DType::F16;
        t.shape[0] = pad_cout;
        t.data = (const uint8_t*)padded.data();
        t.bytes = padded.size() * 2;
    }
    Conv16 c;
    c.cout = (int)t.shape[0];
    c.cin = (int)t.shape[1];
    c.mode = mode;
    Weight raw = upload_weight(t, Place::Auto, false, true);
    const size_t bytes = vc::prep_elems(raw, c.cin, mode) * 2;
    if (place == Place::Device || (place == Place::Auto && gpu_free_bytes() > bytes + G.reserve_bytes)) {
        CK(cudaMalloc(&c.w, bytes));
    } else {
        void* d;
        CK(cudaHostAlloc(&c.host, bytes, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&d, c.host, 0));
        c.w = (__half*)d;
    }
    vc::prep_into(c.w, raw, c.cin, mode);
    CK(cudaStreamSynchronize(G.stream));
    free_weight(raw);
    std::vector<float> b(c.cout, 0.f);
    const StTensor& bt = st.get(k + ".bias");
    for (int64_t i = 0; i < bt.numel(); i++) b[i] = st_elem_f32(bt, i);
    CK(cudaMalloc(&c.b, b.size() * 4));
    CK(cudaMemcpy(c.b, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    return c;
}

void SdVae::load(const SafeTensors& st, const std::string& p, Place place) {
    auto res = [&](const std::string& k) {
        Res r;
        r.n1 = load_norm32(st, k + ".norm1");
        r.c1 = load_conv16(st, k + ".conv1", vc::M3, place);
        r.n2 = load_norm32(st, k + ".norm2");
        r.c2 = load_conv16(st, k + ".conv2", vc::M3, place);
        if (st.has(k + ".nin_shortcut.weight")) r.skip = load_conv16(st, k + ".nin_shortcut", vc::M1, place);
        r.cin = r.c1.cin;
        r.cout = r.c1.cout;
        return r;
    };
    auto attn = [&](const std::string& k) {
        Attn a;
        a.n = load_norm32(st, k + ".norm");
        a.q = load_conv16(st, k + ".q", vc::M1, place);
        a.k = load_conv16(st, k + ".k", vc::M1, place);
        a.v = load_conv16(st, k + ".v", vc::M1, place);
        a.proj = load_conv16(st, k + ".proj_out", vc::M1, place);
        a.C = a.q.cout;
        return a;
    };
    pq_w_ = upload_f32(st.get(p + "post_quant_conv.weight"));
    pq_b_ = upload_f32(st.get(p + "post_quant_conv.bias"));
    q_w_ = upload_f32(st.get(p + "quant_conv.weight"));
    q_b_ = upload_f32(st.get(p + "quant_conv.bias"));
    // decoder
    dec_in_w_ = upload_f32(st.get(p + "decoder.conv_in.weight"));
    dec_in_b_ = upload_f32(st.get(p + "decoder.conv_in.bias"));
    dec_mid1_ = res(p + "decoder.mid.block_1");
    dec_attn_ = attn(p + "decoder.mid.attn_1");
    dec_mid2_ = res(p + "decoder.mid.block_2");
    int nup = 0;
    while (st.has(p + "decoder.up." + std::to_string(nup) + ".block.0.norm1.weight")) nup++;
    for (int i = nup - 1; i >= 0; i--) {  // up.N-1 runs first (lowest resolution)
        const std::string u = p + "decoder.up." + std::to_string(i);
        Level L;
        for (int j = 0; st.has(u + ".block." + std::to_string(j) + ".norm1.weight"); j++) L.res.push_back(res(u + ".block." + std::to_string(j)));
        if (st.has(u + ".upsample.conv.weight")) L.resample = load_conv16(st, u + ".upsample.conv", vc::MUP, place);
        dec_up_.push_back(std::move(L));
    }
    dec_norm_out_ = load_norm32(st, p + "decoder.norm_out");
    dec_out_w_ = upload_f32(st.get(p + "decoder.conv_out.weight"));
    dec_out_b_ = upload_f32(st.get(p + "decoder.conv_out.bias"));
    // encoder
    enc_in_w_ = upload_f32(st.get(p + "encoder.conv_in.weight"));
    enc_in_b_ = upload_f32(st.get(p + "encoder.conv_in.bias"));
    for (int i = 0; st.has(p + "encoder.down." + std::to_string(i) + ".block.0.norm1.weight"); i++) {
        const std::string d = p + "encoder.down." + std::to_string(i);
        Level L;
        for (int j = 0; st.has(d + ".block." + std::to_string(j) + ".norm1.weight"); j++) L.res.push_back(res(d + ".block." + std::to_string(j)));
        if (st.has(d + ".downsample.conv.weight")) L.resample = load_conv16(st, d + ".downsample.conv", vc::MS2, place);
        enc_down_.push_back(std::move(L));
    }
    enc_mid1_ = res(p + "encoder.mid.block_1");
    enc_attn_ = attn(p + "encoder.mid.attn_1");
    enc_mid2_ = res(p + "encoder.mid.block_2");
    enc_norm_out_ = load_norm32(st, p + "encoder.norm_out");
    enc_out_ = load_conv16(st, p + "encoder.conv_out", vc::M3, place, 32);  // 8 channels, padded to 32
}

static void free_conv16(Conv16& c) {
    if (!c.w) return;
    if (c.host) CK(cudaFreeHost(c.host));
    else CK(cudaFree(c.w));
    if (c.b) CK(cudaFree(c.b));
    c = Conv16();
}

void SdVae::free() {
    auto fres = [](Res& r) { free_norm32(r.n1); free_norm32(r.n2); free_conv16(r.c1); free_conv16(r.c2); free_conv16(r.skip); };
    auto fattn = [](Attn& a) { free_norm32(a.n); free_conv16(a.q); free_conv16(a.k); free_conv16(a.v); free_conv16(a.proj); };
    for (auto* v : {&dec_up_, &enc_down_})
        for (auto& L : *v) {
            for (auto& r : L.res) fres(r);
            free_conv16(L.resample);
        }
    dec_up_.clear();
    enc_down_.clear();
    fres(dec_mid1_); fres(dec_mid2_); fres(enc_mid1_); fres(enc_mid2_);
    fattn(dec_attn_); fattn(enc_attn_);
    free_norm32(dec_norm_out_); free_norm32(enc_norm_out_);
    free_conv16(enc_out_);
    for (float** p : {&dec_in_w_, &dec_in_b_, &enc_in_w_, &enc_in_b_, &dec_out_w_, &dec_out_b_, &pq_w_, &pq_b_, &q_w_, &q_b_}) {
        if (*p) CK(cudaFree(*p));
        *p = nullptr;
    }
}

// ---------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------
namespace {

template <class R>
void resblock(Tens& x, const R& r) {
    const int H = x.H, W = x.W;
    float2* st = (float2*)G.arena.f(64);
    group_norm_stats(st, x.p, r.cin, x.P(), 32, 1e-6f);
    Tens h = make(r.cout, H, W);
    conv_layer(h, x, r.c1, st, &r.n1, nullptr);
    group_norm_stats(st, h.p, r.cout, h.P(), 32, 1e-6f);
    Tens out = make(r.cout, H, W);
    if (r.skip.w) conv_layer(out, x, r.skip, nullptr, nullptr, nullptr);
    else CK(cudaMemcpyAsync(out.p, x.p, x.P() * r.cin * 4, cudaMemcpyDefault, G.stream));
    conv_layer(out, h, r.c2, st, &r.n2, &out);
    drop(h);
    drop(x);
    x = out;  // tensors made in the arena stay until the whole decode/encode releases it
}

// single-head self-attention over the H*W pixels (the mid block, at latent resolution)
template <class A>
void attention_block(Tens& x, const A& a) {
    const int C = a.C, T = (int)x.P();
    Tens out = make(C, x.H, x.W);
    size_t m = G.arena.mark();
    float2* st = (float2*)G.arena.f(64);
    group_norm_stats(st, x.p, C, x.P(), 32, 1e-6f);
    __half* n16 = (__half*)G.arena.f(((size_t)C * T + 1) / 2);
    group_norm_apply16(n16, x.p, x.P(), st, a.n.w, a.n.b, C, x.P(), 32, false);
    float* q = G.arena.f((size_t)C * T);
    float* k = G.arena.f((size_t)C * T);
    float* v = G.arena.f((size_t)C * T);
    vc::conv_pre(q, n16, C, x.H, x.W, a.q.w, C, a.q.b, nullptr, vc::M1);
    vc::conv_pre(k, n16, C, x.H, x.W, a.k.w, C, a.k.b, nullptr, vc::M1);
    vc::conv_pre(v, n16, C, x.H, x.W, a.v.w, C, a.v.b, nullptr, vc::M1);
    float* qt = G.arena.f((size_t)C * T);
    float* kt = G.arena.f((size_t)C * T);
    float* vt = G.arena.f((size_t)C * T);
    transpose(qt, q, C, T);
    transpose(kt, k, C, T);
    transpose(vt, v, C, T);
    attention(q, C, qt, C, kt, C, vt, C, T, T, 1, C, false);  // q <- o [T, C]
    transpose(k, q, T, C);                                     // [C, T]
    vc::to16(n16, k, (size_t)C * T, 1.f);
    CK(cudaMemcpyAsync(out.p, x.p, x.P() * C * 4, cudaMemcpyDefault, G.stream));
    vc::conv_pre(out.p, n16, C, x.H, x.W, a.proj.w, C, a.proj.b, out.p, vc::M1);
    G.arena.release(m);
    drop(x);
    x = out;
}

}  // namespace

void SdVae::decode(const float* latent, int h, int w, float* rgb) {
    ProfScope ps("sdvae_decode");
    const size_t m0 = G.arena.mark();
    Tens z = make(4, h, w);
    conv1x1_small(z.p, latent, pq_w_, pq_b_, 4, 4, (size_t)h * w);
    Tens x = make(dec_mid1_.cin, h, w);
    conv3x3_small_in(x.p, z.p, dec_in_w_, dec_in_b_, 4, x.C, h, w);
    resblock(x, dec_mid1_);
    attention_block(x, dec_attn_);
    resblock(x, dec_mid2_);
    for (auto& L : dec_up_) {
        for (auto& r : L.res) resblock(x, r);
        if (L.resample.w) {
            Tens u = make(x.C, 2 * x.H, 2 * x.W);
            conv_layer(u, x, L.resample, nullptr, nullptr, nullptr);
            drop(x);
            x = u;
        }
    }
    // norm_out + silu + conv_out (C -> 3), band by band, straight into rgb
    float2* st = (float2*)G.arena.f(64);
    group_norm_stats(st, x.p, x.C, x.P(), 32, 1e-6f);
    const int H = x.H, W = x.W, rows = band_rows(x.C, W, 1);
    for (int r0 = 0; r0 < H; r0 += rows) {
        const int r1 = std::min(H, r0 + rows), i0 = std::max(0, r0 - 1), i1 = std::min(H, r1 + 1);
        size_t m = G.arena.mark();
        __half* b16 = (__half*)G.arena.f(((size_t)x.C * (i1 - i0) * W + 1) / 2);
        group_norm_apply16(b16, x.p + (size_t)i0 * W, x.P(), st, dec_norm_out_.w, dec_norm_out_.b, x.C, (size_t)(i1 - i0) * W, 32, true);
        conv3x3_small_out(rgb + (size_t)r0 * W, (size_t)H * W, b16, x.C, i1 - i0, W, r0 - i0, r1 - r0, dec_out_w_, dec_out_b_, 3);
        G.arena.release(m);
    }
    drop(x);
    drop(z);
    G.arena.release(m0);
}

void SdVae::encode(const float* rgb, int H, int W, float* latent) {
    ProfScope ps("sdvae_encode");
    const size_t m0 = G.arena.mark();
    Tens x = make(enc_down_.front().res.front().cin, H, W);
    conv3x3_small_in(x.p, rgb, enc_in_w_, enc_in_b_, 3, x.C, H, W);
    for (auto& L : enc_down_) {
        for (auto& r : L.res) resblock(x, r);
        if (L.resample.w) {
            Tens d = make(x.C, x.H / 2, x.W / 2);
            conv_layer(d, x, L.resample, nullptr, nullptr, nullptr);
            drop(x);
            x = d;
        }
    }
    resblock(x, enc_mid1_);
    attention_block(x, enc_attn_);
    resblock(x, enc_mid2_);
    float2* st = (float2*)G.arena.f(64);
    group_norm_stats(st, x.p, x.C, x.P(), 32, 1e-6f);
    Tens mom = make(enc_out_.cout, x.H, x.W);  // 8 real channels (mean, logvar) of the 32 computed
    conv_layer(mom, x, enc_out_, st, &enc_norm_out_, nullptr);
    Tens q = make(8, x.H, x.W);
    conv1x1_small(q.p, mom.p, q_w_, q_b_, 8, 8, x.P());
    CK(cudaMemcpyAsync(latent, q.p, x.P() * 4 * 4, cudaMemcpyDefault, G.stream));  // the mean: channels 0-3
    drop(q);
    drop(mom);
    drop(x);
    G.arena.release(m0);
}
