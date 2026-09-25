// SDXL UNet (SGM layout, e.g. Illustrious / NoobAI / Pony checkpoints). Weights stay fp16 (the
// checkpoint's own precision): matrices as fp16 Weights for the HFMA2 GEMM, convolutions arranged once
// at load into the fp16 implicit-GEMM layout of vaeconv.cuh, norms and biases as fp32.
#pragma once
#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "gpu.h"
#include "safetensors.h"

struct Norm32 { float* w = nullptr; float* b = nullptr; };
struct Linear16 { Weight w; float* b = nullptr; };
// conv weights in the implicit-GEMM layout ([taps][Cin][Cout] fp16), fp32 bias
struct Conv16 {
    __half* w = nullptr;
    float* b = nullptr;
    void* host = nullptr;  // host address when w lives in mapped system RAM (VRAM was short)
    int cin = 0, cout = 0, mode = 0;
};

struct SdResBlock {
    int cin = 0, cout = 0;
    Norm32 n1, n2;
    Conv16 c1, c2, skip;  // skip.w null: identity
    Linear16 emb;
};
struct SdTBlock {
    Norm32 n1, n2, n3;
    Linear16 q1, k1, v1, o1, q2, k2, v2, o2, ff1, ff2;
};
struct SdTransformer {
    int C = 0;
    Norm32 norm;
    Linear16 proj_in, proj_out;
    std::vector<SdTBlock> blocks;
};

// loaders shared by the SD-family models: norms/biases fp32, matrices fp16
Norm32 load_norm32(const SafeTensors& st, const std::string& key);          // key.weight, key.bias
Linear16 load_linear16(const SafeTensors& st, const std::string& key, Place place);  // key.weight [, key.bias]
void free_norm32(Norm32& n);
void free_linear16(Linear16& l);

// [cos(t f_i), sin(t f_i)], f_i = exp(-ln(10000) i / (dim/2)): timestep and SDXL size embeddings
void sinusoidal_embedding(float* out, float t, int dim);

// cross-attention context [len, 2048] on the GPU: CLIP-L (768) | CLIP-G (1280) per token
struct SdContext { const float* p = nullptr; int len = 0; };

class Unet {
public:
    void load(const SafeTensors& st, const std::string& prefix, Place place);
    void free();
    // out [4, Hl, Wl] = UNet(x [4, Hl, Wl] already scaled by c_in, timestep t, adm vector y [2816], context)
    void forward(float* out, const float* x, int Hl, int Wl, float t, const float* y, const SdContext& ctx);
    // cross-attention K/V of every transformer block for a context, kept (in the arena) until dropped
    void cache_context(const SdContext& ctx);
    void drop_context_cache() { kv_.clear(); }
    size_t kv_bytes(int len) const;           // what cache_context needs for a context of `len` tokens
    static size_t forward_bytes(int Hl, int Wl);
    bool loaded() const { return conv_in_w_ != nullptr; }
    // where the weights ended up: bytes in VRAM / in mapped system RAM, and how many tensors are in RAM
    void placement(size_t& vram, size_t& ram, int& ram_tensors) const;

private:
    struct Stage {
        bool res = false, st = false;
        SdResBlock rb;
        SdTransformer tr;
        Conv16 down, up;  // w null: none
    };
    float* conv_in_w_ = nullptr;  // [320][4*9] fp32
    float* conv_in_b_ = nullptr;
    Linear16 t1_, t2_, l1_, l2_;
    std::vector<Stage> in_, out_;  // in_[0] is conv_in (empty stage)
    Stage mid_a_, mid_b_;          // resblock + transformer, resblock
    Norm32 out_norm_;
    float* out_w_ = nullptr;       // [4][320*9] fp32
    float* out_b_ = nullptr;

    struct KV { const float* ctx; std::vector<float*> k, v; };  // per transformer block, in forward order
    std::vector<KV> kv_;

    const KV* find_kv(const float* ctx) const;
    float* res_block(const SdResBlock& rb, const float* x, int H, int W, const float* emb_silu);
    void transformer(const SdTransformer& tr, float* x, int H, int W, const SdContext& ctx, const KV* kv, int& block_index);
};
