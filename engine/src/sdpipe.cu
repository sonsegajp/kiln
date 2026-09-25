#include "sdpipe.h"

#include <cmath>

static inline unsigned nblocks(size_t n) { return (unsigned)((n + 255) / 256); }

void sd_debug(const char* name, const float* p, size_t n) {
    static const bool on = getenv("KILN_SD_DEBUG") != nullptr;
    if (!on) return;
    std::vector<float> h(n);
    CK(cudaStreamSynchronize(G.stream));
    CK(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDefault));
    double mn = 1e30, mx = -1e30, sum = 0;
    size_t bad = 0, first = 0, last = 0;
    for (size_t i = 0; i < h.size(); i++) {
        const float v = h[i];
        if (!std::isfinite(v)) { if (!bad) first = i; last = i; bad++; continue; }
        mn = std::min(mn, (double)v);
        mx = std::max(mx, (double)v);
        sum += v;
    }
    char b[256];
    snprintf(b, sizeof b, "%-22s n=%zu min %.4g max %.4g mean %.4g nonfinite %zu [%zu..%zu]", name, n, mn, mx, sum / std::max<size_t>(1, n - bad), bad, first, last);
    fprintf(stderr, "%s\n", b);
}

void Sdxl::load(const std::string& p) {
    SafeTensors st(p);
    // VAE and UNet on the GPU (the UNet spills into mapped RAM if VRAM runs out); the text encoders run
    // once per prompt, so they stay in system RAM and are read over PCIe
    vae.load(st, "first_stage_model.", Place::Device);
    unet.load(st, "model.diffusion_model.", Place::Auto);
    clip_l.load_hf(st, "conditioner.embedders.0.transformer.text_model.", Place::Host);
    clip_g.load_openclip(st, "conditioner.embedders.1.model.", Place::Host);
    path = p;
}

void Sdxl::free() {
    unet.free();
    vae.free();
    clip_l.free();
    clip_g.free();
    path.clear();
}

__global__ void k_concat_cols(float* out, const float* a, int da, const float* b, int db, int T) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, D = (size_t)da + db;
    if (i >= (size_t)T * D) return;
    const size_t t = i / D, j = i - t * D;
    out[i] = j < (size_t)da ? a[t * da + j] : b[t * db + (j - da)];
}

SdCond Sdxl::condition(const SdTokens& t, int width, int height) {
    const int T = (int)t.l_ids.size(), DL = clip_l.dim(), DG = clip_g.dim();
    if (T == 0 || T % 77 || t.g_ids.size() != t.l_ids.size()) throw std::runtime_error("sdxl: token chunks must be 77 wide and match between CLIP-L and CLIP-G");
    SdCond c;
    CK(cudaMalloc(&c.owned, ((size_t)T * (DL + DG) + 2816) * 4));
    c.ctx.p = c.owned;
    c.ctx.len = T;
    c.y = c.owned + (size_t)T * (DL + DG);
    size_t m = G.arena.mark();
    float* hl = G.arena.f((size_t)T * DL);
    float* hg = G.arena.f((size_t)T * DG);
    float* pooled = G.arena.f(DG);
    clip_l.encode(t.l_ids, t.l_w, hl, nullptr);
    clip_g.encode(t.g_ids, t.g_w, hg, pooled);
    sd_debug("clip_l hidden", hl, (size_t)T * DL);
    sd_debug("clip_g hidden", hg, (size_t)T * DG);
    sd_debug("clip_g pooled", pooled, DG);
    k_concat_cols<<<nblocks((size_t)T * (DL + DG)), 256, 0, G.stream>>>(c.owned, hl, DL, hg, DG, T);
    // y = pooled CLIP-G | embed(height, width, crop_h, crop_w, target_height, target_width), 256 each
    std::vector<float> y(2816);
    CK(cudaMemcpy(y.data(), pooled, (size_t)DG * 4, cudaMemcpyDeviceToHost));
    const float sizes[6] = {(float)height, (float)width, 0.f, 0.f, (float)height, (float)width};
    for (int i = 0; i < 6; i++) sinusoidal_embedding(y.data() + DG + 256 * i, sizes[i], 256);
    CK(cudaMemcpy(c.y, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
    G.arena.release(m);
    return c;
}

void Sdxl::free_cond(SdCond& c) {
    if (c.owned) CK(cudaFree(c.owned));
    c = SdCond();
}

__global__ void k_scale_add(float* z, size_t n, float a, float b) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = z[i] * a + b;
}
void sd_latent_to_vae(float* z, size_t n) { k_scale_add<<<nblocks(n), 256, 0, G.stream>>>(z, n, 1.f / SdVae::scale, 0.f); }
void sd_latent_from_vae(float* z, size_t n) { k_scale_add<<<nblocks(n), 256, 0, G.stream>>>(z, n, SdVae::scale, 0.f); }
