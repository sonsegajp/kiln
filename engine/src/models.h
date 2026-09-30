#pragma once
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "gpu.h"
#include "safetensors.h"

// Qwen3-0.6B, final-norm hidden states. Embedding rows are gathered on the host.
struct TextEncoder {
    std::unique_ptr<SafeTensors> st;
    struct Layer { Weight in_norm, post_norm, q, k, v, o, q_norm, k_norm, gate, up, down; };
    std::vector<Layer> layers;
    Weight norm;
    const StTensor* embed = nullptr;

    void load(const std::string& path, Place place);
    void free();  // release every weight (another model family is taking the VRAM)
    // ids -> hidden [L, 1024] written to out (device, arena-owned by caller)
    void encode(const std::vector<int>& ids, float* out);
};

// Cross-attention context: rows [real, len) are all-zero padding (ComfyUI pads to 512).
struct Context { float* p = nullptr; int len = 0, real = 0; };

struct Dit {
    std::unique_ptr<SafeTensors> st;
    std::string prefix;  // "net." or "" depending on the checkpoint

    struct AdapterBlock { Weight n_sa, sa_q, sa_k, sa_v, sa_o, sa_qn, sa_kn, n_ca, ca_q, ca_k, ca_v, ca_o, ca_qn, ca_kn, n_mlp, mlp0, mlp0_b, mlp2, mlp2_b; };
    Weight ad_embed, ad_out, ad_out_b, ad_norm;
    std::vector<AdapterBlock> ad_blocks;

    struct Block {
        Weight sa_q, sa_k, sa_v, sa_o, sa_qn, sa_kn;
        Weight ca_q, ca_k, ca_v, ca_o, ca_qn, ca_kn;
        Weight mlp1, mlp2;
        Weight mod_sa1, mod_sa2, mod_ca1, mod_ca2, mod_mlp1, mod_mlp2;
    };
    std::vector<Block> blocks;
    Weight x_embed, t_lin1, t_lin2, t_norm, final_mod1, final_mod2, final_lin;

    std::vector<std::pair<std::string, Weight*>> named;  // for LoRA lookup, names without prefix

    void load(const std::string& path, Place place);
    // tensor-core GPUs: the blocks' GEMM weights in VRAM -> fp16 (after loading, and after weights move into VRAM)
    void half_weights();
    // Qwen hidden [Lq,1024] + t5 ids/weights -> cross-attn context [Lc,1024], Lc = max(512, Lt)
    int context_len(int t5_len) const { return t5_len < 512 ? 512 : t5_len; }
    void adapt(const float* qwen_hidden, int Lq, const std::vector<int>& t5_ids, const std::vector<float>& t5_w, float* ctx);

    // Per-generation cross-attention K/V cache (null = recompute each step).
    // Cross-attention K/V per context (prompt and negative), cached for a whole generation when VRAM allows.
    struct KVCache { const float* ctx = nullptr; std::vector<float*> k, v; };
    KVCache kv[3];  // prompt, negative, NAG negative
    bool cache_context(const Context& c);  // fills a free slot; false when all are taken
    void drop_context_cache() { for (auto& s : kv) s = KVCache(); }
    // latent [16, Hl, Wl] (model space) at timestep t -> velocity [16, Hl, Wl]
    void forward(float* out, const float* latent, int Hl, int Wl, float t, const Context& c);
    // First-block cache: across the steps of one sampling run, if block 0's residual barely moved since
    // the last full forward, reuse that forward's residual of blocks 1..27 instead of recomputing them.
    struct StepCache {
        float threshold = 0.f;            // relative L1 change of block 0's residual; 0 = off
        bool allow = false;               // set per step by the sampler (warmup/tail/consecutive guards)
        bool valid = false, skipped = false;
        int R = 0;                        // rows the buffers were sized for
        float *r0 = nullptr, *resid = nullptr;  // [R, D] each, owned by the sampler's arena scope
        float last_change = 0.f;
    };
    // Normalized Attention Guidance (Chen et al. 2025): negative prompts without CFG. In every
    // cross-attention, Z = a*clamp(Z+*s - Z-*(s-1)) + (1-a)*Z+, where the clamp limits each token's
    // L1 norm to tau times that of Z+. Set `nag.neg` to enable it for the next forward calls.
    struct Nag { const Context* neg = nullptr; float scale = 5.f, tau = 2.5f, alpha = 0.25f; };
    Nag nag;
    // Same latent under B contexts in one pass (CFG: prompt + negative). Every linear layer runs once on
    // B*T rows, so the weights are read once; attention runs per context. outs[b] gets context b's velocity.
    void forward_batch(float* const* outs, const float* latent, int Hl, int Wl, float t, const Context* const* cs, int B,
                       StepCache* sc = nullptr);
    // scratch a forward_batch of B needs at this size (for the caller's fits-in-VRAM check)
    static size_t forward_bytes(int Hl, int Wl, int B);
};

struct Vae {
    std::unique_ptr<SafeTensors> st;
    struct Res { Weight n1, c1, c1b, n2, c2, c2b, sc, scb; int cin = 0, cout = 0; };
    struct Attn { Weight n, qkv, qkvb, proj, projb; };
    struct Up { std::vector<Res> res; Weight rs, rsb; int cout = 0; };
    struct Down { std::vector<Res> res; Weight ds, dsb; int cout = 0; };
    // decoder
    Weight conv2, conv2b, conv1, conv1b, head_n, head_c, head_cb;
    Res mid0, mid1;
    Attn attn;
    std::vector<Up> ups;  // 4 stages; stages 0-2 end with a 2x upsample
    // encoder
    Weight e_conv1, e_conv1b, e_head_n, e_head_c, e_head_cb, q_conv, q_convb;
    Res e_mid0, e_mid1;
    Attn e_attn;
    std::vector<Down> downs;  // 4 stages; stages 0-2 end with a stride-2 downsample

    // FP16: convs as fp16x2 implicit GEMMs (activations stay fp32 in memory), ~3x faster.
    // FP32: im2col + cuBLAS SGEMM, the exact reference path. G.fp16 == false also selects FP32.
    enum class Precision { FP16, FP32 };
    Precision precision = Precision::FP16;
    int band_rows = 0;  // test hook: cap the band height (latent rows) of decode/encode; 0 = largest that fits
    std::vector<std::pair<std::string, float>>* probe = nullptr;  // debug: max |x| of named activations (syncs)

    void load(const std::string& path, Place place);
    void free();
    // latent [16, Hl, Wl] in VAE space -> rgb [3, 8Hl, 8Wl] in [-1, 1], written to out (device, caller-owned)
    void decode(const float* latent, int Hl, int Wl, float* out);
    // rgb [3, H, W] in [-1, 1] on device (H, W multiples of 8) -> latent mean [16, H/8, W/8] in VAE space
    void encode(const float* rgb, int H, int W, float* latent);
    bool use_fp16() const;

private:
    bool fp16_ok = false;
    void note(const std::string& name, const void* p, size_t n, bool half = false);
    void resblock(float*& x, float*& a, float*& b, const Res& r, int H, int W, const std::string& tag);
    void resblock16(float*& x, float*& t, __half* h, const Res& r, int H, int W, const std::string& tag);
    void attention_block(float* x, float* a, float* b, const Attn& at, int C, int H, int W);
    void attention_block16(float* x, float* a, __half* h, const Attn& at, int C, int H, int W);
    void upstack(const float* mid, int Hl, int Wl, float* out_rgb, int row0, int rows);
    void upstack16(const float* mid, int Hl, int Wl, float* out_rgb, int row0, int rows);
    void downstack(const float* rgb, int H, int W, float* mid, int lrow0, int lrows);
    void downstack16(const float* rgb, int H, int W, float* mid, int lrow0, int lrows);
    void up_caps(size_t& cap32, size_t& cap16) const;
    void down_caps(size_t& cap32, size_t& cap16) const;
};
