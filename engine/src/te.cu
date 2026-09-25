// Qwen3-0.6B text encoder (28 layers, GQA 16/8, head 128, q/k RMSNorm, RoPE theta 1e6).
#include "models.h"

#include <cmath>

void TextEncoder::load(const std::string& path, Place place) {
    st = std::make_unique<SafeTensors>(path);
    auto W = [&](const std::string& n) { return upload_weight(st->get(n), place); };
    embed = &st->get("model.embed_tokens.weight");
    for (int i = 0;; i++) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        if (!st->has(p + "input_layernorm.weight")) break;
        Layer l;
        l.in_norm = W(p + "input_layernorm.weight");
        l.post_norm = W(p + "post_attention_layernorm.weight");
        l.q = W(p + "self_attn.q_proj.weight");
        l.k = W(p + "self_attn.k_proj.weight");
        l.v = W(p + "self_attn.v_proj.weight");
        l.o = W(p + "self_attn.o_proj.weight");
        l.q_norm = W(p + "self_attn.q_norm.weight");
        l.k_norm = W(p + "self_attn.k_norm.weight");
        l.gate = W(p + "mlp.gate_proj.weight");
        l.up = W(p + "mlp.up_proj.weight");
        l.down = W(p + "mlp.down_proj.weight");
        layers.push_back(l);
    }
    norm = W("model.norm.weight");
}

// cos/sin tables [T, Dh/2] for rotate-half RoPE at positions 0..T-1, computed the way torch does (fp32).
void rope_table(std::vector<float>& c, std::vector<float>& s, int T, int Dh, float theta) {
    int half = Dh / 2;
    c.resize((size_t)T * half);
    s.resize((size_t)T * half);
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)Dh);
        for (int t = 0; t < T; t++) {
            float f = (float)t * inv;
            c[(size_t)t * half + i] = cosf(f);
            s[(size_t)t * half + i] = sinf(f);
        }
    }
}

void TextEncoder::encode(const std::vector<int>& ids, float* out) {
    const int L = (int)ids.size(), D = 1024, H = 16, Hkv = 8, Dh = 128;
    const int F = (int)layers[0].gate.rows;
    const float eps = 1e-6f;

    std::vector<float> e((size_t)L * D);
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) e[(size_t)t * D + d] = st_elem_f32(*embed, (int64_t)ids[t] * D + d);
    CK(cudaMemcpy(out, e.data(), e.size() * 4, cudaMemcpyHostToDevice));

    size_t m = G.arena.mark();
    float* h = G.arena.f((size_t)L * D);
    float* q = G.arena.f((size_t)L * H * Dh);
    float* k = G.arena.f((size_t)L * Hkv * Dh);
    float* v = G.arena.f((size_t)L * Hkv * Dh);
    float* ke = G.arena.f((size_t)L * H * Dh);
    float* ve = G.arena.f((size_t)L * H * Dh);
    float* att = G.arena.f((size_t)L * H * Dh);
    float* g = G.arena.f((size_t)L * F);
    float* u = G.arena.f((size_t)L * F);
    float* cs = G.arena.f((size_t)L * Dh);
    std::vector<float> ch, sh;
    rope_table(ch, sh, L, Dh, 1000000.f);
    CK(cudaMemcpy(cs, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cs + ch.size(), sh.data(), sh.size() * 4, cudaMemcpyHostToDevice));
    const float* cos_t = cs;
    const float* sin_t = cs + ch.size();

    // Qwen's residual stream reaches thousands; fp16 GEMM inputs would overflow. The TE is a
    // tiny share of the work, so it always runs in exact fp32.
    bool fp16 = G.fp16;
    G.fp16 = false;
    float* x = out;
    for (auto& l : layers) {
        rmsnorm(h, x, &l.in_norm, L, D, eps);
        linear(q, h, L, l.q);
        linear(k, h, L, l.k);
        linear(v, h, L, l.v);
        rmsnorm(q, q, &l.q_norm, L * H, Dh, eps);
        rmsnorm(k, k, &l.k_norm, L * Hkv, Dh, eps);
        rope_half(q, L, H, Dh, cos_t, sin_t);
        rope_half(k, L, Hkv, Dh, cos_t, sin_t);
        repeat_kv(ke, k, L, Hkv, H / Hkv, Dh);
        repeat_kv(ve, v, L, Hkv, H / Hkv, Dh);
        attention(att, H * Dh, q, H * Dh, ke, H * Dh, ve, H * Dh, L, L, H, Dh, true);
        linear(x, att, L, l.o, nullptr, 1.f);
        rmsnorm(h, x, &l.post_norm, L, D, eps);
        linear(g, h, L, l.gate);
        linear(u, h, L, l.up);
        silu_mul(g, u, (size_t)L * F);
        linear(x, g, L, l.down, nullptr, 1.f);
    }
    rmsnorm(x, x, &norm, L, D, eps);
    G.fp16 = fp16;
    G.arena.release(m);
}

void TextEncoder::free() {
    for (auto& l : layers)
        for (Weight* w : {&l.in_norm, &l.post_norm, &l.q, &l.k, &l.v, &l.o, &l.q_norm, &l.k_norm, &l.gate, &l.up, &l.down}) free_weight(*w);
    layers.clear();
    free_weight(norm);
    embed = nullptr;
    st.reset();
}
