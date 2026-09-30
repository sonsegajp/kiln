// Anima = Cosmos-Predict2 MiniTrainDIT (28 blocks, 2048 wide, adaLN-LoRA, 3D RoPE)
// plus the 6-layer LLM adapter that maps Qwen3 states + T5 ids into cross-attn context.
#include "models.h"

#include <cmath>

void rope_table(std::vector<float>& c, std::vector<float>& s, int T, int Dh, float theta);

static const int D = 2048, NH = 16, DH = 128, AD = 1024, AH = 16, ADH = 64;

void Dit::load(const std::string& path, Place place) {
    struct MapOk { MapOk() { G.file_map_ok = true; } ~MapOk() { G.file_map_ok = false; } } map_ok;  // st stays open: mapped weights point into it
    st = std::make_unique<SafeTensors>(path);
    for (auto& [name, t] : st->all()) {
        const std::string tail = "x_embedder.proj.1.weight";
        if (name.size() >= tail.size() && name.compare(name.size() - tail.size(), tail.size(), tail) == 0) {
            prefix = name.substr(0, name.size() - tail.size());
            break;
        }
    }
    auto W = [&](const std::string& n) {
        Weight w = upload_weight(st->get(prefix + n), place);
        return w;
    };
    // Register every weight under its prefix-free name so LoRAs can find it.
    auto R = [&](Weight& w, const std::string& n) { w = W(n); named.push_back({n, &w}); };

    blocks.resize(28);
    for (int i = 0; i < 28; i++) {
        auto& b = blocks[i];
        std::string p = "blocks." + std::to_string(i) + ".";
        R(b.sa_q, p + "self_attn.q_proj.weight");
        R(b.sa_k, p + "self_attn.k_proj.weight");
        R(b.sa_v, p + "self_attn.v_proj.weight");
        R(b.sa_o, p + "self_attn.output_proj.weight");
        R(b.sa_qn, p + "self_attn.q_norm.weight");
        R(b.sa_kn, p + "self_attn.k_norm.weight");
        R(b.ca_q, p + "cross_attn.q_proj.weight");
        R(b.ca_k, p + "cross_attn.k_proj.weight");
        R(b.ca_v, p + "cross_attn.v_proj.weight");
        R(b.ca_o, p + "cross_attn.output_proj.weight");
        R(b.ca_qn, p + "cross_attn.q_norm.weight");
        R(b.ca_kn, p + "cross_attn.k_norm.weight");
        R(b.mlp1, p + "mlp.layer1.weight");
        R(b.mlp2, p + "mlp.layer2.weight");
        R(b.mod_sa1, p + "adaln_modulation_self_attn.1.weight");
        R(b.mod_sa2, p + "adaln_modulation_self_attn.2.weight");
        R(b.mod_ca1, p + "adaln_modulation_cross_attn.1.weight");
        R(b.mod_ca2, p + "adaln_modulation_cross_attn.2.weight");
        R(b.mod_mlp1, p + "adaln_modulation_mlp.1.weight");
        R(b.mod_mlp2, p + "adaln_modulation_mlp.2.weight");
    }
    R(x_embed, "x_embedder.proj.1.weight");
    R(t_lin1, "t_embedder.1.linear_1.weight");
    R(t_lin2, "t_embedder.1.linear_2.weight");
    R(t_norm, "t_embedding_norm.weight");
    R(final_mod1, "final_layer.adaln_modulation.1.weight");
    R(final_mod2, "final_layer.adaln_modulation.2.weight");
    R(final_lin, "final_layer.linear.weight");

    ad_blocks.resize(6);
    for (int i = 0; i < 6; i++) {
        auto& b = ad_blocks[i];
        std::string p = "llm_adapter.blocks." + std::to_string(i) + ".";
        R(b.n_sa, p + "norm_self_attn.weight");
        R(b.sa_q, p + "self_attn.q_proj.weight");
        R(b.sa_k, p + "self_attn.k_proj.weight");
        R(b.sa_v, p + "self_attn.v_proj.weight");
        R(b.sa_o, p + "self_attn.o_proj.weight");
        R(b.sa_qn, p + "self_attn.q_norm.weight");
        R(b.sa_kn, p + "self_attn.k_norm.weight");
        R(b.n_ca, p + "norm_cross_attn.weight");
        R(b.ca_q, p + "cross_attn.q_proj.weight");
        R(b.ca_k, p + "cross_attn.k_proj.weight");
        R(b.ca_v, p + "cross_attn.v_proj.weight");
        R(b.ca_o, p + "cross_attn.o_proj.weight");
        R(b.ca_qn, p + "cross_attn.q_norm.weight");
        R(b.ca_kn, p + "cross_attn.k_norm.weight");
        R(b.n_mlp, p + "norm_mlp.weight");
        R(b.mlp0, p + "mlp.0.weight");
        R(b.mlp0_b, p + "mlp.0.bias");
        R(b.mlp2, p + "mlp.2.weight");
        R(b.mlp2_b, p + "mlp.2.bias");
    }
    G.no_file_map = true;  // embed_rows reads it directly
    R(ad_embed, "llm_adapter.embed.weight");
    G.no_file_map = false;
    R(ad_out, "llm_adapter.out_proj.weight");
    R(ad_out_b, "llm_adapter.out_proj.bias");
    R(ad_norm, "llm_adapter.norm.weight");
    half_weights();
}

// With tensor cores, the blocks' GEMM weights are stored as fp16: the tensor-core kernels read fp16 either way
// (tcgemm.cuh converts bf16 while staging), and the fp16-activation path streams them with cp.async as they are.
void Dit::half_weights() {
    if (!G.tc) return;
    for (auto& b : blocks)
        for (Weight* w : {&b.sa_q, &b.sa_k, &b.sa_v, &b.sa_o, &b.ca_q, &b.ca_k, &b.ca_v, &b.ca_o, &b.mlp1, &b.mlp2})
            if (linear16_eligible(*w)) weight_to_f16(*w);
}

// ---------------------------------------------------------------------------
// LLM adapter
// ---------------------------------------------------------------------------
void Dit::adapt(const float* qwen, int Lq, const std::vector<int>& t5_ids, const std::vector<float>& t5_w, float* ctx) {
    const int Lt = (int)t5_ids.size(), Lc = context_len(Lt);
    const float eps = 1e-6f;
    size_t m = G.arena.mark();
    int* ids = G.arena.i(Lt);
    CK(cudaMemcpy(ids, t5_ids.data(), Lt * 4, cudaMemcpyHostToDevice));
    float* x = G.arena.f((size_t)Lt * AD);
    float* n = G.arena.f((size_t)Lt * AD);
    float* q = G.arena.f((size_t)Lt * AD);
    float* k = G.arena.f((size_t)std::max(Lt, Lq) * AD);
    float* v = G.arena.f((size_t)std::max(Lt, Lq) * AD);
    float* a = G.arena.f((size_t)Lt * AD);
    float* hbuf = G.arena.f((size_t)Lt * 4 * AD);
    float* w = G.arena.f(Lt);
    CK(cudaMemcpy(w, t5_w.data(), Lt * 4, cudaMemcpyHostToDevice));

    std::vector<float> ct, sn, cc, sc;
    rope_table(ct, sn, Lt, ADH, 10000.f);
    rope_table(cc, sc, Lq, ADH, 10000.f);
    float* tab = G.arena.f(ct.size() * 2 + cc.size() * 2);
    CK(cudaMemcpy(tab, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tab + ct.size(), sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tab + 2 * ct.size(), cc.data(), cc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tab + 2 * ct.size() + cc.size(), sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
    const float *cos_t = tab, *sin_t = tab + ct.size(), *cos_c = tab + 2 * ct.size(), *sin_c = cos_c + cc.size();

    bool fp16 = G.fp16;  // runs once per prompt and feeds every step: keep it exact
    G.fp16 = false;
    embed_rows(x, ad_embed, ids, Lt);
    for (auto& b : ad_blocks) {
        rmsnorm(n, x, &b.n_sa, Lt, AD, eps);
        linear(q, n, Lt, b.sa_q);
        linear(k, n, Lt, b.sa_k);
        linear(v, n, Lt, b.sa_v);
        rmsnorm(q, q, &b.sa_qn, Lt * AH, ADH, eps);
        rmsnorm(k, k, &b.sa_kn, Lt * AH, ADH, eps);
        rope_half(q, Lt, AH, ADH, cos_t, sin_t);
        rope_half(k, Lt, AH, ADH, cos_t, sin_t);
        attention(a, AD, q, AD, k, AD, v, AD, Lt, Lt, AH, ADH, false);
        linear(x, a, Lt, b.sa_o, nullptr, 1.f);

        rmsnorm(n, x, &b.n_ca, Lt, AD, eps);
        linear(q, n, Lt, b.ca_q);
        linear(k, qwen, Lq, b.ca_k);
        linear(v, qwen, Lq, b.ca_v);
        rmsnorm(q, q, &b.ca_qn, Lt * AH, ADH, eps);
        rmsnorm(k, k, &b.ca_kn, Lq * AH, ADH, eps);
        rope_half(q, Lt, AH, ADH, cos_t, sin_t);
        rope_half(k, Lq, AH, ADH, cos_c, sin_c);
        attention(a, AD, q, AD, k, AD, v, AD, Lt, Lq, AH, ADH, false);
        linear(x, a, Lt, b.ca_o, nullptr, 1.f);

        rmsnorm(n, x, &b.n_mlp, Lt, AD, eps);
        linear(hbuf, n, Lt, b.mlp0, &b.mlp0_b);
        gelu(hbuf, (size_t)Lt * 4 * AD);
        linear(x, hbuf, Lt, b.mlp2, &b.mlp2_b, 1.f);
    }
    linear(n, x, Lt, ad_out, &ad_out_b);
    rmsnorm(ctx, n, &ad_norm, Lt, AD, eps);
    scale_rows(ctx, w, Lt, AD);
    fill(ctx + (size_t)Lt * AD, 0.f, (size_t)(Lc - Lt) * AD);
    G.fp16 = fp16;
    G.arena.release(m);
}

// ---------------------------------------------------------------------------
// DiT
// ---------------------------------------------------------------------------
__global__ void k_patchify(float* tok, const float* lat, int Hl, int Wl) {
    int Wp = Wl / 2, T = (Hl / 2) * Wp;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 68) return;
    int f = i % 68, t = (int)(i / 68);
    int c = f / 4, m = (f / 2) % 2, nn = f % 2;
    int y = (t / Wp) * 2 + m, x = (t % Wp) * 2 + nn;
    tok[i] = c < 16 ? lat[((size_t)c * Hl + y) * Wl + x] : 0.f;  // channel 16 = padding mask (zeros)
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

// Cosmos 3D RoPE for an image (T=1): per head 22 temporal + 21 height + 21 width frequencies,
// rotating pairs (i, i+64). Height/width use NTK-scaled theta (extrapolation ratio 4).
static void cosmos_rope(std::vector<float>& c, std::vector<float>& s, int Hp, int Wp) {
    const int dim_h = DH / 6 * 2, dim_t = DH - 2 * dim_h;  // 42, 44
    double ntk = pow(4.0, (double)dim_h / (dim_h - 2));
    float h_theta = (float)(10000.0 * ntk);
    std::vector<float> hf(dim_h / 2);
    for (int j = 0; j < dim_h / 2; j++) hf[j] = 1.0f / powf(h_theta, (float)(2 * j) / (float)dim_h);
    int T = Hp * Wp, half = DH / 2;
    c.assign((size_t)T * half, 1.f);
    s.assign((size_t)T * half, 0.f);
    for (int t = 0; t < T; t++) {
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

bool Dit::cache_context(const Context& c) {
    KVCache* free_slot = nullptr;
    for (auto& s : kv) if (!s.ctx) { free_slot = &s; break; }
    if (!free_slot) return false;
    KVCache& slot = *free_slot;
    const int L = c.real;  // zero rows give K = V = 0 and are handled inside attention()
    slot.k.assign(blocks.size(), nullptr);
    slot.v.assign(blocks.size(), nullptr);
    for (size_t i = 0; i < blocks.size(); i++) {
        slot.k[i] = G.arena.f((size_t)L * D);
        slot.v[i] = G.arena.f((size_t)L * D);
        linear(slot.k[i], c.p, L, blocks[i].ca_k);
        rmsnorm(slot.k[i], slot.k[i], &blocks[i].ca_kn, L * NH, DH, 1e-6f);
        linear(slot.v[i], c.p, L, blocks[i].ca_v);
    }
    slot.ctx = c.p;
    return true;
}

size_t Dit::forward_bytes(int Hl, int Wl, int B) {
    size_t T = (size_t)(Hl / 2) * (Wl / 2), R = B * T;
    // tok, X, N, Y, big (4 D wide) + small per-step tables; Kc/Vc for uncached contexts are tiny
    return (R * 68 + 3 * R * D + R * 4 * D) * 4 + T * 128 * 4 + ((size_t)16 << 20);
}

void Dit::forward(float* out, const float* latent, int Hl, int Wl, float t, const Context& c) {
    const Context* cs[1] = {&c};
    float* outs[1] = {out};
    forward_batch(outs, latent, Hl, Wl, t, cs, 1);
}

// NAG combine, one block per token row: a (Z+) <- alpha*clamp(g) + (1-alpha)*Z+, g = Z+*s - Z-*(s-1)
// (float rows, or fp16 rows on the fp16-activation path; the arithmetic is fp32 either way)
__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half_rn(v); }

template <typename T>
__global__ void k_nag(T* a, const T* an, int dim, float s, float tau, float alpha) {
    T* p = a + (size_t)blockIdx.x * dim;
    const T* n = an + (size_t)blockIdx.x * dim;
    __shared__ float red[2][32];
    float lp = 0.f, lg = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float pi = ld(p + i);
        float g = pi * s - ld(n + i) * (s - 1.f);
        lp += fabsf(pi);
        lg += fabsf(g);
    }
    for (int o = 16; o > 0; o >>= 1) { lp += __shfl_xor_sync(0xffffffff, lp, o); lg += __shfl_xor_sync(0xffffffff, lg, o); }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { red[0][wid] = lp; red[1][wid] = lg; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tp = 0.f, tg = 0.f;
        for (int w = 0; w < (int)(blockDim.x + 31) / 32; w++) { tp += red[0][w]; tg += red[1][w]; }
        red[0][0] = tp; red[1][0] = tg;
    }
    __syncthreads();
    float ratio = red[1][0] / fmaxf(red[0][0], 1e-12f);
    float k = ratio > tau ? tau / ratio : 1.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float pi = ld(p + i);
        float g = pi * s - ld(n + i) * (s - 1.f);
        st(p + i, alpha * g * k + (1.f - alpha) * pi);
    }
}

// sum |a - b| and sum |b| over n floats, into acc[0], acc[1]
__global__ void k_rel_l1(const float* a, const float* b, size_t n, float* acc) {
    float d = 0.f, r = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        d += fabsf(a[i] - b[i]);
        r += fabsf(b[i]);
    }
    for (int o = 16; o > 0; o >>= 1) { d += __shfl_xor_sync(0xffffffff, d, o); r += __shfl_xor_sync(0xffffffff, r, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(acc, d); atomicAdd(acc + 1, r); }
}

__global__ void k_sub(float* y, const float* a, const float* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - b[i];
}

__global__ void k_pad_circular(float* dst, const float* src, int Hs, int Ws, int Hd, int Wd) {
    size_t n = (size_t)16 * Hd * Wd;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = i % Wd, y = (i / Wd) % Hd, c = (int)(i / ((size_t)Wd * Hd));
    dst[i] = src[((size_t)c * Hs + y % Hs) * Ws + x % Ws];
}
__global__ void k_crop(float* dst, const float* src, int Hs, int Ws, int Hd, int Wd) {
    size_t n = (size_t)16 * Hd * Wd;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = i % Wd, y = (i / Wd) % Hd, c = (int)(i / ((size_t)Wd * Hd));
    dst[i] = src[((size_t)c * Hs + y) * Ws + x];
}

void Dit::forward_batch(float* const* outs, const float* latent, int Hl, int Wl, float t, const Context* const* cs, int B,
                        StepCache* sc) {
    if (Hl % 2 || Wl % 2) {  // pad to the 2x2 patch grid (circular, like ComfyUI), run, crop back
        int H2 = Hl + Hl % 2, W2 = Wl + Wl % 2;
        size_t n2 = (size_t)16 * H2 * W2, n1 = (size_t)16 * Hl * Wl;
        size_t m = G.arena.mark();
        float* pin = G.arena.f(n2);
        std::vector<float*> pout(B);
        for (int b = 0; b < B; b++) pout[b] = G.arena.f(n2);
        k_pad_circular<<<(unsigned)((n2 + 255) / 256), 256, 0, G.stream>>>(pin, latent, Hl, Wl, H2, W2);
        forward_batch(pout.data(), pin, H2, W2, t, cs, B, sc);
        for (int b = 0; b < B; b++) k_crop<<<(unsigned)((n1 + 255) / 256), 256, 0, G.stream>>>(outs[b], pout[b], H2, W2, Hl, Wl);
        G.arena.release(m);
        return;
    }
    const int Hp = Hl / 2, Wp = Wl / 2, T = Hp * Wp, R = B * T, F = 4 * D;
    const size_t TD = (size_t)T * D;
    const float eps = 1e-6f;
    size_t m = G.arena.mark();

    // timestep: sinusoid s -> adaLN-LoRA vector (6144) and normalized embedding
    std::vector<float> sh(D);
    for (int i = 0; i < D / 2; i++) {
        float e = expf((float)i * (-logf(10000.f)) / (float)(D / 2));
        sh[i] = cosf(t * e);
        sh[i + D / 2] = sinf(t * e);
    }
    float* s = G.arena.f(D);
    float* s1 = G.arena.f(D);
    float* lora = G.arena.f(3 * D);
    float* temb = G.arena.f(D);
    float* h256 = G.arena.f(256);
    float* mods = G.arena.f(blocks.size() * 9 * D + 2 * D);
    CK(cudaMemcpyAsync(s, sh.data(), D * 4, cudaMemcpyHostToDevice, G.stream));
    matvec(s1, t_lin1, s);
    silu(s1, s1, D);
    matvec(lora, t_lin2, s1);
    rmsnorm(temb, s, &t_norm, 1, D, eps);
    silu(temb, temb, D);  // every modulation MLP starts with SiLU
    for (size_t i = 0; i < blocks.size(); i++) {
        auto& b = blocks[i];
        float* mo = mods + i * 9 * D;
        matvec(h256, b.mod_sa1, temb); matvec(mo, b.mod_sa2, h256, lora);
        matvec(h256, b.mod_ca1, temb); matvec(mo + 3 * D, b.mod_ca2, h256, lora);
        matvec(h256, b.mod_mlp1, temb); matvec(mo + 6 * D, b.mod_mlp2, h256, lora);
    }
    float* fmod = mods + blocks.size() * 9 * D;
    matvec(h256, final_mod1, temb);
    matvec(fmod, final_mod2, h256, lora);

    std::vector<float> rc, rs;
    cosmos_rope(rc, rs, Hp, Wp);
    float* rope = G.arena.f(rc.size() * 2);
    CK(cudaMemcpyAsync(rope, rc.data(), rc.size() * 4, cudaMemcpyHostToDevice, G.stream));
    CK(cudaMemcpyAsync(rope + rc.size(), rs.data(), rs.size() * 4, cudaMemcpyHostToDevice, G.stream));
    const float *rcos = rope, *rsin = rope + rc.size();

    float* tok = G.arena.f((size_t)R * 68);
    float* X = G.arena.f((size_t)R * D);
    float* N = G.arena.f((size_t)R * D);
    float* Y = G.arena.f((size_t)R * D);
    float* big = G.arena.f((size_t)R * F);  // Q,K,V,A during attention; MLP hidden after
    float *Q = big, *K = big + (size_t)R * D, *V = big + 2 * (size_t)R * D, *A = big + 3 * (size_t)R * D;

    // cross-attn K/V: cached slot, or computed per block into scratch
    struct Src { const float *k = nullptr, *v = nullptr; const KVCache* cache = nullptr; float *kc = nullptr, *vc = nullptr; int L = 0, npad = 0; };
    auto make_src = [&](const Context& c) {
        Src sr;
        sr.L = c.real;
        sr.npad = c.len - c.real;
        for (auto& slot : kv) if (slot.ctx && slot.ctx == c.p) sr.cache = &slot;
        if (!sr.cache) { sr.kc = G.arena.f((size_t)sr.L * D); sr.vc = G.arena.f((size_t)sr.L * D); }
        return sr;
    };
    std::vector<Src> src(B);
    for (int b = 0; b < B; b++) src[b] = make_src(*cs[b]);
    // NAG guides batch item 0 (the prompt); in a batched CFG pass the negative item runs plain, like ComfyUI-NAG
    const bool use_nag = nag.neg != nullptr;
    Src nsrc;
    float* An = nullptr;
    if (use_nag) { nsrc = make_src(*nag.neg); An = G.arena.f(TD); }
    auto kv_for = [&](Src& sc, const Context& c, const Block& bl, size_t i, const float*& kk, const float*& vv) {
        if (sc.cache) { kk = sc.cache->k[i]; vv = sc.cache->v[i]; return; }
        linear(sc.kc, c.p, sc.L, bl.ca_k);
        rmsnorm(sc.kc, sc.kc, &bl.ca_kn, sc.L * NH, DH, 1e-6f);
        linear(sc.vc, c.p, sc.L, bl.ca_v);
        kk = sc.kc; vv = sc.vc;
    };

    // every batch element starts from the same latent
    k_patchify<<<(unsigned)(((size_t)T * 68 + 255) / 256), 256, 0, G.stream>>>(tok, latent, Hl, Wl);
    linear(X, tok, T, x_embed);
    for (int b = 1; b < B; b++) CK(cudaMemcpyAsync(X + b * TD, X, TD * 4, cudaMemcpyDeviceToDevice, G.stream));

    float* X0 = nullptr;  // input to block 0 (step cache)
    bool use_cache = sc && sc->threshold > 0.f;
    if (use_cache) {
        if (sc->R != R) { sc->valid = false; sc->R = R; }
        X0 = G.arena.f((size_t)R * D);
        CK(cudaMemcpyAsync(X0, X, (size_t)R * D * 4, cudaMemcpyDeviceToDevice, G.stream));
        sc->skipped = false;
    }
    const size_t RD = (size_t)R * D;
    // fp16-activation path (tcgemm16.cuh): GEMM inputs are fp16 (norm, attention and GELU outputs), the o-proj and
    // MLP-out GEMMs add into the residual stream X themselves, mlp1 applies GELU itself. X, Q, K, V stay fp32.
    const bool h16 = G.fp16 && G.tc16;
    __half* N16 = reinterpret_cast<__half*>(N);    // [R, D] in N's space (N is not used on this path)
    __half* A16 = reinterpret_cast<__half*>(A);    // [R, D] attention output in A's space
    __half* H16 = reinterpret_cast<__half*>(big);  // [R, 4D] MLP hidden (Q, K, V, A are dead by then)
    __half* An16 = reinterpret_cast<__half*>(An);
    for (size_t i = 0; i < blocks.size(); i++) {
        if (use_cache && i == 1) {
            // block 0 done: X0 <- this step's block-0 residual; compare with the last full forward's
            k_sub<<<(unsigned)((RD + 255) / 256), 256, 0, G.stream>>>(X0, X, X0, RD);
            if (sc->valid) {
                float* acc = G.arena.f(2);
                CK(cudaMemsetAsync(acc, 0, 8, G.stream));
                k_rel_l1<<<120, 256, 0, G.stream>>>(X0, sc->r0, RD, acc);
                float h[2];
                CK(cudaMemcpy(h, acc, 8, cudaMemcpyDeviceToHost));
                sc->last_change = h[1] > 0.f ? h[0] / h[1] : 1.f;
                if (sc->allow && sc->last_change < sc->threshold) {
                    add_inplace(X, sc->resid, RD);  // X = block-0 output + cached residual of blocks 1..27
                    sc->skipped = true;
                    break;
                }
            }
            CK(cudaMemcpyAsync(sc->r0, X0, RD * 4, cudaMemcpyDeviceToDevice, G.stream));
            CK(cudaMemcpyAsync(sc->resid, X, RD * 4, cudaMemcpyDeviceToDevice, G.stream));  // block-0 output, finished below
        }
        auto& bl = blocks[i];
        const float* mo = mods + i * 9 * D;
        if (h16) {
            // self-attention
            layernorm_mod16(N16, X, mo + D, mo, R, D, eps);
            linear16(Q, D, N16, R, bl.sa_q, Epi16::F32);
            linear16(K, D, N16, R, bl.sa_k, Epi16::F32);
            linear16(V, D, N16, R, bl.sa_v, Epi16::F32);
            rmsnorm(Q, Q, &bl.sa_qn, R * NH, DH, eps);
            rmsnorm(K, K, &bl.sa_kn, R * NH, DH, eps);
            for (int b = 0; b < B; b++) {
                rope_half(Q + b * TD, T, NH, DH, rcos, rsin);
                rope_half(K + b * TD, T, NH, DH, rcos, rsin);
                attention16(A16 + b * TD, D, Q + b * TD, D, K + b * TD, D, V + b * TD, D, T, T, NH, DH);
            }
            linear16(X, D, A16, R, bl.sa_o, Epi16::Resid, mo + 2 * D);
            // cross-attention
            const float* mc = mo + 3 * D;
            layernorm_mod16(N16, X, mc + D, mc, R, D, eps);
            linear16(Q, D, N16, R, bl.ca_q, Epi16::F32);
            rmsnorm(Q, Q, &bl.ca_qn, R * NH, DH, eps);
            for (int b = 0; b < B; b++) {
                const float *kk, *vv;
                kv_for(src[b], *cs[b], bl, i, kk, vv);
                attention16(A16 + b * TD, D, Q + b * TD, D, kk, D, vv, D, T, src[b].L, NH, DH, src[b].npad);
            }
            if (use_nag) {
                const float *kk, *vv;
                kv_for(nsrc, *nag.neg, bl, i, kk, vv);
                attention16(An16, D, Q, D, kk, D, vv, D, T, nsrc.L, NH, DH, nsrc.npad);
                k_nag<__half><<<(unsigned)T, 256, 0, G.stream>>>(A16, An16, D, nag.scale, nag.tau, nag.alpha);
            }
            linear16(X, D, A16, R, bl.ca_o, Epi16::Resid, mc + 2 * D);
            // MLP
            const float* mm = mo + 6 * D;
            layernorm_mod16(N16, X, mm + D, mm, R, D, eps);
            linear16(H16, F, N16, R, bl.mlp1, Epi16::Gelu16);
            linear16(X, D, H16, R, bl.mlp2, Epi16::Resid, mm + 2 * D);
            continue;
        }
        // self-attention
        layernorm_mod(N, X, mo + D, mo, R, D, eps);
        linear(Q, N, R, bl.sa_q);
        linear(K, N, R, bl.sa_k);
        linear(V, N, R, bl.sa_v);
        rmsnorm(Q, Q, &bl.sa_qn, R * NH, DH, eps);
        rmsnorm(K, K, &bl.sa_kn, R * NH, DH, eps);
        for (int b = 0; b < B; b++) {
            rope_half(Q + b * TD, T, NH, DH, rcos, rsin);
            rope_half(K + b * TD, T, NH, DH, rcos, rsin);
            attention(A + b * TD, D, Q + b * TD, D, K + b * TD, D, V + b * TD, D, T, T, NH, DH, false);
        }
        linear(Y, A, R, bl.sa_o);
        add_gated(X, Y, mo + 2 * D, R, D);
        // cross-attention, each batch element against its own context
        const float* mc = mo + 3 * D;
        layernorm_mod(N, X, mc + D, mc, R, D, eps);
        linear(Q, N, R, bl.ca_q);
        rmsnorm(Q, Q, &bl.ca_qn, R * NH, DH, eps);
        for (int b = 0; b < B; b++) {
            const float *kk, *vv;
            kv_for(src[b], *cs[b], bl, i, kk, vv);
            attention(A + b * TD, D, Q + b * TD, D, kk, D, vv, D, T, src[b].L, NH, DH, false, src[b].npad);
        }
        if (use_nag) {
            const float *kk, *vv;
            kv_for(nsrc, *nag.neg, bl, i, kk, vv);
            attention(An, D, Q, D, kk, D, vv, D, T, nsrc.L, NH, DH, false, nsrc.npad);
            k_nag<float><<<(unsigned)T, 256, 0, G.stream>>>(A, An, D, nag.scale, nag.tau, nag.alpha);
        }
        linear(Y, A, R, bl.ca_o);
        add_gated(X, Y, mc + 2 * D, R, D);
        // MLP
        const float* mm = mo + 6 * D;
        layernorm_mod(N, X, mm + D, mm, R, D, eps);
        linear(big, N, R, bl.mlp1);
        gelu(big, (size_t)R * F);
        linear(Y, big, R, bl.mlp2);
        add_gated(X, Y, mm + 2 * D, R, D);
    }
    if (use_cache && !sc->skipped) {
        // resid currently holds block 0's output: turn it into the residual of blocks 1..27
        k_sub<<<(unsigned)((RD + 255) / 256), 256, 0, G.stream>>>(sc->resid, X, sc->resid, RD);
        sc->valid = true;
    }
    layernorm_mod(N, X, fmod + D, fmod, R, D, eps);
    linear(tok, N, R, final_lin);  // [R, 64] fits in the 68-wide token buffer
    for (int b = 0; b < B; b++)
        k_unpatchify<<<(unsigned)(((size_t)T * 64 + 255) / 256), 256, 0, G.stream>>>(outs[b], tok + (size_t)b * T * 64, Hl, Wl);
    G.arena.release(m);
}
