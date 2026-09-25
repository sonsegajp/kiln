#include "sdxl.h"

#include <algorithm>
#include <cmath>
#include <tuple>

#include "sdops.h"
#include "sdpipe.h"
#include "vaeconv.cuh"

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------
Norm32 load_norm32(const SafeTensors& st, const std::string& k) {
    return {upload_f32(st.get(k + ".weight")), upload_f32(st.get(k + ".bias"))};
}

Linear16 load_linear16(const SafeTensors& st, const std::string& k, Place place) {
    Linear16 l;
    l.w = upload_weight(st.get(k + ".weight"), place, false, true);
    if (st.has(k + ".bias")) l.b = upload_f32(st.get(k + ".bias"));
    return l;
}

static Conv16 load_conv(const SafeTensors& st, const std::string& k, int mode, Place place) {
    const StTensor& t = st.get(k + ".weight");
    Conv16 c;
    c.cout = (int)t.shape[0];
    c.cin = (int)t.shape[1];
    c.mode = mode;
    Weight raw = upload_weight(t, Place::Auto, false, true);
    const size_t bytes = vc::prep_elems(raw, c.cin, mode) * 2;
    const bool device = place == Place::Device || (place == Place::Auto && gpu_free_bytes() > bytes + G.reserve_bytes);
    if (device) {
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
    c.b = upload_f32(st.get(k + ".bias"));
    return c;
}

static SdResBlock load_res(const SafeTensors& st, const std::string& k, Place place) {
    SdResBlock r;
    r.n1 = load_norm32(st, k + ".in_layers.0");
    r.c1 = load_conv(st, k + ".in_layers.2", vc::M3, place);
    r.emb = load_linear16(st, k + ".emb_layers.1", place);
    r.n2 = load_norm32(st, k + ".out_layers.0");
    r.c2 = load_conv(st, k + ".out_layers.3", vc::M3, place);
    if (st.has(k + ".skip_connection.weight")) r.skip = load_conv(st, k + ".skip_connection", vc::M1, place);
    r.cin = r.c1.cin;
    r.cout = r.c1.cout;
    return r;
}

static SdTransformer load_tr(const SafeTensors& st, const std::string& k, Place place) {
    SdTransformer t;
    t.norm = load_norm32(st, k + ".norm");
    t.proj_in = load_linear16(st, k + ".proj_in", place);
    t.proj_out = load_linear16(st, k + ".proj_out", place);
    t.C = (int)t.proj_in.w.rows;
    for (int i = 0; st.has(k + ".transformer_blocks." + std::to_string(i) + ".norm1.weight"); i++) {
        const std::string b = k + ".transformer_blocks." + std::to_string(i);
        SdTBlock tb;
        tb.n1 = load_norm32(st, b + ".norm1");
        tb.n2 = load_norm32(st, b + ".norm2");
        tb.n3 = load_norm32(st, b + ".norm3");
        tb.q1 = load_linear16(st, b + ".attn1.to_q", place);
        tb.k1 = load_linear16(st, b + ".attn1.to_k", place);
        tb.v1 = load_linear16(st, b + ".attn1.to_v", place);
        tb.o1 = load_linear16(st, b + ".attn1.to_out.0", place);
        tb.q2 = load_linear16(st, b + ".attn2.to_q", place);
        tb.k2 = load_linear16(st, b + ".attn2.to_k", place);
        tb.v2 = load_linear16(st, b + ".attn2.to_v", place);
        tb.o2 = load_linear16(st, b + ".attn2.to_out.0", place);
        tb.ff1 = load_linear16(st, b + ".ff.net.0.proj", place);
        tb.ff2 = load_linear16(st, b + ".ff.net.2", place);
        t.blocks.push_back(std::move(tb));
    }
    return t;
}

void Unet::load(const SafeTensors& st, const std::string& p, Place place) {
    conv_in_w_ = upload_f32(st.get(p + "input_blocks.0.0.weight"));
    conv_in_b_ = upload_f32(st.get(p + "input_blocks.0.0.bias"));
    t1_ = load_linear16(st, p + "time_embed.0", place);
    t2_ = load_linear16(st, p + "time_embed.2", place);
    l1_ = load_linear16(st, p + "label_emb.0.0", place);
    l2_ = load_linear16(st, p + "label_emb.0.2", place);
    // a block is [resblock] [transformer] [down/upsample]; which of them it has shows in its keys
    auto stage = [&](const std::string& b) {
        Stage s;
        for (int j = 0; j < 3; j++) {
            const std::string k = b + "." + std::to_string(j);
            if (st.has(k + ".in_layers.0.weight")) { s.res = true; s.rb = load_res(st, k, place); }
            else if (st.has(k + ".norm.weight")) { s.st = true; s.tr = load_tr(st, k, place); }
            else if (st.has(k + ".op.weight")) s.down = load_conv(st, k + ".op", vc::MS2P, place);
            else if (st.has(k + ".conv.weight")) s.up = load_conv(st, k + ".conv", vc::MUP, place);
        }
        return s;
    };
    in_.clear();
    out_.clear();
    in_.push_back(Stage());  // input_blocks.0: conv_in
    auto block = [&](const char* kind, int i) { return p + kind + "." + std::to_string(i); };
    for (int i = 1; st.has(block("input_blocks", i) + ".0.in_layers.0.weight") || st.has(block("input_blocks", i) + ".0.op.weight"); i++)
        in_.push_back(stage(block("input_blocks", i)));
    mid_a_.res = true;
    mid_a_.rb = load_res(st, p + "middle_block.0", place);
    mid_a_.st = true;
    mid_a_.tr = load_tr(st, p + "middle_block.1", place);
    mid_b_.res = true;
    mid_b_.rb = load_res(st, p + "middle_block.2", place);
    for (int i = 0; st.has(block("output_blocks", i) + ".0.in_layers.0.weight"); i++) out_.push_back(stage(block("output_blocks", i)));
    out_norm_ = load_norm32(st, p + "out.0");
    out_w_ = upload_f32(st.get(p + "out.2.weight"));
    out_b_ = upload_f32(st.get(p + "out.2.bias"));
}

void free_linear16(Linear16& l) {
    free_weight(l.w);
    if (l.b) CK(cudaFree(l.b));
    l = Linear16();
}
void free_norm32(Norm32& n) {
    if (n.w) CK(cudaFree(n.w));
    if (n.b) CK(cudaFree(n.b));
    n = Norm32();
}
static void free_conv(Conv16& c) {
    if (!c.w) return;
    if (c.host) CK(cudaFreeHost(c.host));
    else CK(cudaFree(c.w));
    if (c.b) CK(cudaFree(c.b));
    c = Conv16();
}
static void free_res(SdResBlock& r) {
    free_norm32(r.n1); free_norm32(r.n2);
    free_conv(r.c1); free_conv(r.c2); free_conv(r.skip);
    free_linear16(r.emb);
}
static void free_tr(SdTransformer& t) {
    free_norm32(t.norm);
    free_linear16(t.proj_in); free_linear16(t.proj_out);
    for (auto& b : t.blocks) {
        free_norm32(b.n1); free_norm32(b.n2); free_norm32(b.n3);
        for (Linear16* l : {&b.q1, &b.k1, &b.v1, &b.o1, &b.q2, &b.k2, &b.v2, &b.o2, &b.ff1, &b.ff2}) free_linear16(*l);
    }
    t.blocks.clear();
}

void Unet::free() {
    kv_.clear();
    for (float* p : {conv_in_w_, conv_in_b_, out_w_, out_b_}) if (p) CK(cudaFree(p));
    conv_in_w_ = conv_in_b_ = out_w_ = out_b_ = nullptr;
    free_linear16(t1_); free_linear16(t2_); free_linear16(l1_); free_linear16(l2_);
    for (auto* v : {&in_, &out_})
        for (auto& s : *v) {
            if (s.res) free_res(s.rb);
            if (s.st) free_tr(s.tr);
            free_conv(s.down); free_conv(s.up);
        }
    in_.clear();
    out_.clear();
    free_res(mid_a_.rb); free_tr(mid_a_.tr); free_res(mid_b_.rb);
    free_norm32(out_norm_);
}

// conv weights that live in mapped RAM are copied to VRAM for the call (see stage_weight)
static const __half* staged(const Conv16& c) {
    const size_t bytes = (size_t)c.cout * c.cin * (c.mode == vc::MUP ? 16 : c.mode == vc::M1 ? 1 : 9) * 2;
    return (const __half*)stage_weight(c.w, c.host, bytes);
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------
// [cos(t f), sin(t f)], f_i = exp(-ln(10000) i / half), in fp32 like the reference
void sinusoidal_embedding(float* out, float t, int dim) {
    const int half = dim / 2;
    for (int i = 0; i < half; i++) {
        const float f = expf(-logf(10000.f) * (float)i / (float)half), a = t * f;
        out[i] = cosf(a);
        out[half + i] = sinf(a);
    }
}

float* Unet::res_block(const SdResBlock& rb, const float* x, int H, int W, const float* emb_silu) {
    const int P = H * W;
    float* out = G.arena.f((size_t)rb.cout * P);
    size_t m = G.arena.mark();
    __half* h16 = (__half*)G.arena.f(((size_t)std::max(rb.cin, rb.cout) * P + 1) / 2);
    group_norm(nullptr, h16, x, rb.n1.w, rb.n1.b, rb.cin, P, 32, 1e-5f, true);
    float* h = G.arena.f((size_t)rb.cout * P);
    vc::conv_pre(h, h16, rb.cin, H, W, staged(rb.c1), rb.cout, rb.c1.b, nullptr, vc::M3);
    sd_debug("  rb conv1", h, (size_t)rb.cout * P);
    float* e = G.arena.f(rb.cout);
    linear(e, emb_silu, 1, rb.emb.w);
    bias_rows(e, rb.emb.b, 1, rb.cout);
    add_channels(h, e, rb.cout, P);
    // the skip path goes into `out` first; the second conv then accumulates onto it
    if (rb.skip.w) {
        vc::to16(h16, x, (size_t)rb.cin * P, 1.f);
        vc::conv_pre(out, h16, rb.cin, H, W, staged(rb.skip), rb.cout, rb.skip.b, nullptr, vc::M1);
        sd_debug("  rb skip", out, (size_t)rb.cout * P);
    } else {
        CK(cudaMemcpyAsync(out, x, (size_t)rb.cout * P * 4, cudaMemcpyDeviceToDevice, G.stream));
    }
    group_norm(nullptr, h16, h, rb.n2.w, rb.n2.b, rb.cout, P, 32, 1e-5f, true);
    vc::conv_pre(out, h16, rb.cout, H, W, staged(rb.c2), rb.cout, rb.c2.b, out, vc::M3);
    sd_debug("  rb out", out, (size_t)rb.cout * P);
    G.arena.release(m);
    return out;
}

void Unet::transformer(const SdTransformer& tr, float* x, int H, int W, const SdContext& ctx, const KV* kv, int& bi) {
    const int C = tr.C, T = H * W, heads = C / 64;
    size_t m = G.arena.mark();
    float* h = G.arena.f((size_t)T * C);   // residual stream [T, C]
    float* a = G.arena.f((size_t)T * C);   // normed input / scratch [T, C]
    group_norm(a, nullptr, x, tr.norm.w, tr.norm.b, C, T, 32, 1e-6f, false);
    transpose(h, a, C, T);
    linear(a, h, T, tr.proj_in.w);
    bias_rows(a, tr.proj_in.b, T, C);
    std::swap(a, h);
    float* q = G.arena.f((size_t)T * C);
    float* k = G.arena.f((size_t)T * C);
    float* v = G.arena.f((size_t)T * C);
    float* o = G.arena.f((size_t)T * C);
    float* ff = G.arena.f((size_t)T * 8 * C);
    float* g = G.arena.f((size_t)T * 4 * C);
    float *kc = nullptr, *vc_ = nullptr;
    if (!kv) {
        kc = G.arena.f((size_t)ctx.len * C);
        vc_ = G.arena.f((size_t)ctx.len * C);
    }
    for (const SdTBlock& b : tr.blocks) {
        layer_norm(a, h, b.n1.w, b.n1.b, T, C, 1e-5f);
        sd_debug("  tb ln1", a, (size_t)T * C);
        linear(q, a, T, b.q1.w);
        linear(k, a, T, b.k1.w);
        linear(v, a, T, b.v1.w);
        sd_debug("  tb q", q, (size_t)T * C);
        sd_debug("  tb k", k, (size_t)T * C);
        sd_debug("  tb v", v, (size_t)T * C);
        attention(o, C, q, C, k, C, v, C, T, T, heads, 64, false);
        sd_debug("  tb self-attn", o, (size_t)T * C);
        if (getenv("KILN_SD_DUMP")) {  // bring-up: dump the first self-attention that produces non-finite values
            static bool dumped = false;
            std::vector<float> ho((size_t)T * C);
            CK(cudaStreamSynchronize(G.stream));
            CK(cudaMemcpy(ho.data(), o, ho.size() * 4, cudaMemcpyDeviceToHost));
            bool bad = false;
            for (float f : ho) bad |= !std::isfinite(f);
            if (bad && !dumped) {
                dumped = true;
                auto dump = [&](const char* n, const float* d) {
                    std::vector<float> h((size_t)T * C);
                    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
                    FILE* f = fopen((std::string(getenv("KILN_SD_DUMP")) + "/attn_" + n + ".f32").c_str(), "wb");
                    fwrite(h.data(), 4, h.size(), f);
                    fclose(f);
                };
                dump("q", q); dump("k", k); dump("v", v); dump("o", o);
                fprintf(stderr, "dumped attention T=%d C=%d heads=%d arena_free=%zu MB\n", T, C, heads, G.arena.free_bytes() >> 20);
            }
        }
        linear(h, o, T, b.o1.w, nullptr, 1.f);
        bias_rows(h, b.o1.b, T, C);

        layer_norm(a, h, b.n2.w, b.n2.b, T, C, 1e-5f);
        linear(q, a, T, b.q2.w);
        const float *ck = kc, *cv = vc_;
        if (kv) {
            ck = kv->k[bi];
            cv = kv->v[bi];
        } else {
            linear(kc, ctx.p, ctx.len, b.k2.w);
            linear(vc_, ctx.p, ctx.len, b.v2.w);
        }
        attention(o, C, q, C, ck, C, cv, C, T, ctx.len, heads, 64, false);
        sd_debug("  tb cross-attn", o, (size_t)T * C);
        linear(h, o, T, b.o2.w, nullptr, 1.f);
        bias_rows(h, b.o2.b, T, C);

        layer_norm(a, h, b.n3.w, b.n3.b, T, C, 1e-5f);
        linear(ff, a, T, b.ff1.w);
        bias_rows(ff, b.ff1.b, T, 8 * C);
        geglu(g, ff, T, 4 * C);
        sd_debug("  tb geglu", g, (size_t)T * 4 * C);
        linear(h, g, T, b.ff2.w, nullptr, 1.f);
        bias_rows(h, b.ff2.b, T, C);
        sd_debug("  tb out", h, (size_t)T * C);
        bi++;
    }
    linear(a, h, T, tr.proj_out.w);
    bias_rows(a, tr.proj_out.b, T, C);
    transpose(h, a, T, C);
    add_inplace(x, h, (size_t)C * T);
    G.arena.release(m);
}

void Unet::placement(size_t& vram, size_t& ram, int& ram_tensors) const {
    vram = ram = 0;
    ram_tensors = 0;
    auto lin = [&](const Linear16& l) {
        if (!l.w.p) return;
        size_t b = (size_t)l.w.numel() * 2;
        if (l.w.on_host) { ram += b; ram_tensors++; } else vram += b;
    };
    auto conv = [&](const Conv16& c) {
        if (!c.w) return;
        size_t b = (size_t)c.cout * c.cin * (c.mode == 3 ? 16 : c.mode == 1 ? 1 : 9) * 2;
        if (c.host) { ram += b; ram_tensors++; } else vram += b;
    };
    auto res = [&](const SdResBlock& r) { conv(r.c1); conv(r.c2); conv(r.skip); lin(r.emb); };
    auto tr = [&](const SdTransformer& t) {
        lin(t.proj_in); lin(t.proj_out);
        for (auto& b : t.blocks) for (const Linear16* l : {&b.q1, &b.k1, &b.v1, &b.o1, &b.q2, &b.k2, &b.v2, &b.o2, &b.ff1, &b.ff2}) lin(*l);
    };
    auto stage = [&](const Stage& s) {
        if (s.res) res(s.rb);
        if (s.st) tr(s.tr);
        conv(s.down); conv(s.up);
    };
    lin(t1_); lin(t2_); lin(l1_); lin(l2_);
    for (auto& s : in_) stage(s);
    stage(mid_a_);
    stage(mid_b_);
    for (auto& s : out_) stage(s);
}

const Unet::KV* Unet::find_kv(const float* ctx) const {
    for (auto& k : kv_) if (k.ctx == ctx) return &k;
    return nullptr;
}

size_t Unet::kv_bytes(int len) const {
    size_t n = 0;
    auto add = [&](const Stage& s) { if (s.st) n += s.tr.blocks.size() * 2 * (size_t)len * s.tr.C * 4; };
    for (auto& s : in_) add(s);
    add(mid_a_);
    for (auto& s : out_) add(s);
    return n;
}

void Unet::cache_context(const SdContext& ctx) {
    if (find_kv(ctx.p)) return;
    KV e;
    e.ctx = ctx.p;
    auto add = [&](const Stage& s) {
        if (!s.st) return;
        for (auto& b : s.tr.blocks) {
            float* k = G.arena.f((size_t)ctx.len * s.tr.C);
            float* v = G.arena.f((size_t)ctx.len * s.tr.C);
            linear(k, ctx.p, ctx.len, b.k2.w);
            linear(v, ctx.p, ctx.len, b.v2.w);
            e.k.push_back(k);
            e.v.push_back(v);
        }
    };
    for (auto& s : in_) add(s);
    add(mid_a_);
    for (auto& s : out_) add(s);
    kv_.push_back(std::move(e));
}

// Peak scratch of one forward at Hl x Wl (skip tensors, decoder concatenations, the level-1 transformer
// temporaries), with margin; attention scores are chunked into whatever is left.
size_t Unet::forward_bytes(int Hl, int Wl) { return (size_t)Hl * Wl * 36000 + ((size_t)96 << 20); }

void Unet::forward(float* out, const float* x, int Hl, int Wl, float t, const float* y, const SdContext& ctx) {
    ProfScope ps("unet");
    const size_t m0 = G.arena.mark();
    // time and size/pooled-text embeddings -> silu(emb), shared by every resblock
    float th[320];
    sinusoidal_embedding(th, t, 320);
    float* te = G.arena.f(320);
    CK(cudaMemcpyAsync(te, th, sizeof th, cudaMemcpyHostToDevice, G.stream));
    float* e1 = G.arena.f(1280);
    float* emb = G.arena.f(1280);
    linear(e1, te, 1, t1_.w);
    bias_rows(e1, t1_.b, 1, 1280);
    silu(e1, e1, 1280);
    linear(emb, e1, 1, t2_.w);
    bias_rows(emb, t2_.b, 1, 1280);
    linear(e1, y, 1, l1_.w);
    bias_rows(e1, l1_.b, 1, 1280);
    silu(e1, e1, 1280);
    linear(emb, e1, 1, l2_.w, nullptr, 1.f);
    bias_rows(emb, l2_.b, 1, 1280);
    silu(emb, emb, 1280);

    const KV* kv = find_kv(ctx.p);
    int bi = 0, H = Hl, W = Wl, C = 320;
    std::vector<std::tuple<float*, int, int, int>> hs;  // skip tensors: buffer, channels, H, W
    float* h = G.arena.f((size_t)C * H * W);
    conv3x3_small_in(h, x, conv_in_w_, conv_in_b_, 4, C, H, W);
    sd_debug("unet emb", emb, 1280);
    sd_debug("unet conv_in", h, (size_t)C * H * W);
    hs.emplace_back(h, C, H, W);

    auto downsample = [&](const Conv16& d) {
        const int Ho = (H + 1) / 2, Wo = (W + 1) / 2;
        float* o = G.arena.f((size_t)C * Ho * Wo);
        size_t m = G.arena.mark();
        __half* h16 = (__half*)G.arena.f(((size_t)C * H * W + 1) / 2);
        vc::to16(h16, h, (size_t)C * H * W, 1.f);
        vc::conv_pre(o, h16, C, H, W, staged(d), C, d.b, nullptr, vc::MS2P);
        G.arena.release(m);
        h = o;
        H = Ho;
        W = Wo;
    };
    auto upsample = [&](const Conv16& u) {
        float* o = G.arena.f((size_t)C * 4 * H * W);
        size_t m = G.arena.mark();
        __half* h16 = (__half*)G.arena.f(((size_t)C * H * W + 1) / 2);
        vc::to16(h16, h, (size_t)C * H * W, 1.f);
        vc::conv_pre(o, h16, C, H, W, staged(u), C, u.b, nullptr, vc::MUP);
        G.arena.release(m);
        h = o;
        H *= 2;
        W *= 2;
    };

    for (size_t i = 1; i < in_.size(); i++) {
        const Stage& s = in_[i];
        if (s.res) { h = res_block(s.rb, h, H, W, emb); C = s.rb.cout; }
        if (s.st) transformer(s.tr, h, H, W, ctx, kv, bi);
        if (s.down.w) downsample(s.down);
        sd_debug(("unet in " + std::to_string(i)).c_str(), h, (size_t)C * H * W);
        hs.emplace_back(h, C, H, W);
    }
    h = res_block(mid_a_.rb, h, H, W, emb);
    transformer(mid_a_.tr, h, H, W, ctx, kv, bi);
    h = res_block(mid_b_.rb, h, H, W, emb);
    sd_debug("unet mid", h, (size_t)C * H * W);
    int oi = 0;
    for (const Stage& s : out_) {
        auto [sk, sc, sh, sw] = hs.back();
        hs.pop_back();
        if (sh != H || sw != W) throw std::runtime_error("unet: skip connection size mismatch (latent size must be a multiple of 8)");
        const size_t P = (size_t)H * W;
        float* cat = G.arena.f((C + sc) * P);
        CK(cudaMemcpyAsync(cat, h, C * P * 4, cudaMemcpyDeviceToDevice, G.stream));
        CK(cudaMemcpyAsync(cat + C * P, sk, sc * P * 4, cudaMemcpyDeviceToDevice, G.stream));
        h = res_block(s.rb, cat, H, W, emb);
        C = s.rb.cout;
        if (s.st) transformer(s.tr, h, H, W, ctx, kv, bi);
        if (s.up.w) upsample(s.up);
        sd_debug(("unet out " + std::to_string(oi++)).c_str(), h, (size_t)C * H * W);
    }
    __half* h16 = (__half*)G.arena.f(((size_t)C * H * W + 1) / 2);
    group_norm(nullptr, h16, h, out_norm_.w, out_norm_.b, C, H * W, 32, 1e-5f, true);
    conv3x3_small_out(out, (size_t)H * W, h16, C, H, W, 0, H, out_w_, out_b_, 4);
    G.arena.release(m0);
}
