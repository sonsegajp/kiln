// SDXL as a whole: a single-file (SGM) checkpoint's text encoders, UNet and VAE, prompt conditioning,
// and the eps-prediction sampler (the SD family's discrete model sampling).
#pragma once
#include <atomic>
#include <string>
#include <vector>

#include "clip.h"
#include "pipeline.h"
#include "sdvae.h"
#include "sdxl.h"

// a tokenized prompt: chunks of 77 for each encoder (server/lib/clip_tokenize.js encodeSDXL)
struct SdTokens {
    std::vector<int> l_ids, g_ids;
    std::vector<float> l_w, g_w;
};

// conditioning on the GPU: context [len, 2048] and the size/pooled vector y [2816], one allocation
struct SdCond {
    SdContext ctx;
    float* y = nullptr;
    float* owned = nullptr;
};

class Sdxl {
public:
    std::string path;
    ClipText clip_l, clip_g;
    Unet unet;
    SdVae vae;

    void load(const std::string& path);
    void free();
    bool loaded() const { return unet.loaded(); }
    SdCond condition(const SdTokens& t, int width, int height);  // width/height: the image size (ADM)
    static void free_cond(SdCond& c);
};

// KILN_SD_DEBUG=1: log min / max / mean / non-finite count of a device tensor (syncs; bring-up aid)
void sd_debug(const char* name, const float* p, size_t n);

// latent [4, h, w] (model space, i.e. VAE latent * 0.13025) <-> VAE space, in place
void sd_latent_to_vae(float* z, size_t n);
void sd_latent_from_vae(float* z, size_t n);

// Sampling with ComfyUI's KSampler semantics for eps models: x [4, Hl, Wl] result in model space.
// init: model-space latent (img2img) or null; mask [Hl*Wl]: 1 = regenerate. The step callback gets x and
// the (CFG-combined) eps; x0 = x - eps * sigma.
void sample_sd(Unet& unet, float* x, int Hl, int Wl, const SdCond& pos, const SdCond* neg, const SampleParams& sp, const float* init,
               const float* mask, const StepFn& on_step, const std::atomic<bool>* cancel);
