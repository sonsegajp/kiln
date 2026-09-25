// Sampling (txt2img, img2img, masked inpainting) and the image-space helpers the post-processing
// passes share. Semantics follow ComfyUI's KSampler for Anima (flow matching, euler, "simple").
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "models.h"

// torch.manual_seed(seed); torch.randn(n) on CPU, float32 -- the exact noise ComfyUI uses
std::vector<float> torch_randn(uint64_t seed, size_t n);
// "simple" schedule over ModelSamplingDiscreteFlow(shift). denoise < 1 runs `steps` steps over the
// tail of a longer schedule, exactly like KSampler: int(steps / denoise) total, keep the last steps+1.
std::vector<float> flow_sigmas(int steps, float shift, float denoise = 1.f);
// Any ComfyUI scheduler (simple, sgm_uniform, karras, exponential, ddim_uniform, beta, normal,
// linear_quadratic, kl_optimal) over the same flow model sampling, with KSampler's denoise handling.
std::vector<float> schedule_sigmas(const std::string& scheduler, int steps, float shift, float denoise = 1.f);
// The same schedulers over the SD/SDXL discrete model sampling (eps prediction, 1000 scaled-linear steps).
std::vector<float> schedule_sigmas_sd(const std::string& scheduler, int steps, float denoise = 1.f);
float sd_timestep(float sigma);  // the model timestep the UNet sees for a sigma (nearest table index)
float sd_sigma_max();
bool known_scheduler(const std::string& name);
bool known_sampler(const std::string& name);

// Wan2.1 latent normalization, in place on device [16, P]
void latent_model_to_vae(float* z, int P);
void latent_vae_to_model(float* z, int P);

struct SampleParams {
    int steps = 8;
    float cfg = 1.f, shift = 3.f, denoise = 1.f;
    uint64_t seed = 0;
    float cfg_until = 1.f;  // run CFG for this fraction of the steps, then the prompt alone (1 = always)
    std::string sampler = "euler";  // euler | dpmpp_2m | res_multistep (ComfyUI's deterministic versions)
    float cache_threshold = 0.f;    // first-block step cache; 0 = off
    // NAG: negative prompt without CFG (used when there is no CFG negative pass)
    const Context* nag_neg = nullptr;
    float nag_scale = 5.f, nag_tau = 2.5f, nag_alpha = 0.25f;
    float nag_sigma_end = 0.f;          // NAG only while sigma >= this
    // graph-node extras (defaults = the simple-mode behaviour)
    std::string scheduler = "simple";
    std::vector<float> sigmas;          // explicit schedule (overrides scheduler/steps/denoise)
    const float* noise = nullptr;       // host noise [16*Hl*Wl] instead of randn(seed) (batch slices)
    bool add_noise = true;              // KSamplerAdvanced add_noise=disable -> zero noise
    float timestep_mult = 1.f;          // ModelSamplingSD3 feeds sigma*1000 to the model
    float cache_start = 0.15f, cache_end = 0.9f;  // fractions of the steps where the cache may skip
    int cache_max_hits = 2;             // consecutive skips allowed (-1 = unlimited)
};

// step (1-based), total, sigma of this step, current x and velocity (device), ms for the step
using StepFn = std::function<void(int, int, float, const float*, const float*, double)>;

// x: result latent [16, Hl, Wl] in model space (device, caller-owned).
// init: model-space latent to start from (img2img), or null for pure noise.
// mask: [Hl*Wl] on device, 1 = regenerate, 0 = keep init; requires init. Null = no mask.
extern int last_cache_skips;  // steps the first-block cache skipped in the last sample() run
void sample(Dit& dit, float* x, int Hl, int Wl, const Context& pos, const Context* neg, const SampleParams& sp,
            const float* init, const float* mask, const StepFn& on_step, const std::atomic<bool>* cancel);

// image helpers on device, [3, H, W]
void rgb_to_unit(float* dst, const float* src, size_t n);     // [-1,1] -> [0,1]
void rgb_from_unit(float* dst, const float* src, size_t n);   // [0,1] -> [-1,1]
// dst = mask*src + (1-mask)*dst over the rectangle at (x0, y0) of a [C, H, W] image; src/mask are [C|1, h, w]
void blend_rect(float* dst, int C, int H, int W, const float* src, const float* mask, int x0, int y0, int w, int h);
// copies the rectangle (x0, y0, w, h) of [C, H, W] into [C, h, w]
void crop_rect(float* dst, const float* src, int C, int H, int W, int x0, int y0, int w, int h);
