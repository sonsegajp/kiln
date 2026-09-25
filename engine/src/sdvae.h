// The SD/SDXL KL autoencoder (4 latent channels), decode and encode.
// At 1024^2 its full-resolution activations are 0.5-1 GB each, more than the arena holds next to the
// SDXL UNet, and GroupNorm needs statistics over the whole image, so tiles can't be normalised on their
// own (the seams of "tiled VAE"). Every activation is a full tensor, in VRAM when it fits and in pinned
// system RAM otherwise; each layer streams row bands through the GPU, and every GroupNorm first gathers
// its statistics over all bands. The result is the same as one full-image pass.
#pragma once
#include <string>
#include <vector>

#include "gpu.h"
#include "safetensors.h"
#include "sdxl.h"

class SdVae {
public:
    void load(const SafeTensors& st, const std::string& prefix, Place place);  // prefix like "first_stage_model."
    void free();
    bool loaded() const { return !dec_up_.empty(); }
    // latent [4, h, w] in VAE space (already divided by the scale factor) -> rgb [3, 8h, 8w] in [-1, 1]
    void decode(const float* latent, int h, int w, float* rgb);
    // rgb [3, H, W] in [-1, 1] (H, W multiples of 8) -> latent [4, H/8, W/8] (the posterior mean)
    void encode(const float* rgb, int H, int W, float* latent);
    static constexpr float scale = 0.13025f;  // SDXL latent scale factor

private:
    struct Res { int cin = 0, cout = 0; Norm32 n1, n2; Conv16 c1, c2, skip; };
    struct Level { std::vector<Res> res; Conv16 resample; };  // up: upsample conv; down: downsample conv
    struct Attn { int C = 0; Norm32 n; Conv16 q, k, v, proj; };
    Conv16 dec_in_, enc_out_, post_quant_, quant_;
    float *dec_in_w_ = nullptr, *dec_in_b_ = nullptr;      // 4 -> 512, small-Cin conv
    float *enc_in_w_ = nullptr, *enc_in_b_ = nullptr;      // 3 -> 128
    float *dec_out_w_ = nullptr, *dec_out_b_ = nullptr;    // 128 -> 3
    float *pq_w_ = nullptr, *pq_b_ = nullptr;              // post_quant_conv 4 -> 4 (1x1)
    float *q_w_ = nullptr, *q_b_ = nullptr;                // quant_conv 8 -> 8 (1x1)
    Res dec_mid1_, dec_mid2_, enc_mid1_, enc_mid2_;
    Attn dec_attn_, enc_attn_;
    std::vector<Level> dec_up_, enc_down_;  // dec_up_ in execution order (lowest resolution first)
    Norm32 dec_norm_out_, enc_norm_out_;
};
