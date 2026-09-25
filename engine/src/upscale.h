// RRDBNet (ESRGAN / Real-ESRGAN family) image upscaler, e.g. 4x-AnimeSharp. Old ESRGAN
// ("model.N...") and BasicSR ("conv_first", "body.N.rdb1...") key naming; nf / gc / blocks / scale
// are read off the tensors. Weights stay in their shipped precision (fp16; fp32 checkpoints are
// rounded to fp16), not bf16: see upscale.cu for why.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu.h"
#include "safetensors.h"

struct Upscaler {
    // Context (LR pixels) around each tile core. The net's receptive field is ~350 px, far larger
    // than any tile, so tiling is not bit-exact; 32 px keeps tiled output within 1.7e-4 of the
    // untiled result (measured, see upscale.cu). Untiled whenever the image fits the arena.
    // Scratch: ~1.5 KB per tile pixel (incl. halo) for the trunk, plus the tail's row bands.
    int halo = 32;
    int max_tile = 0;           // cap on the tile core side in LR px (0 = as large as the arena allows)
    bool bf16_weights = false;  // round weights through bf16 like upload_weight (accuracy experiment)
    // Conv precision follows the engine's G.fp16: fp16x2 direct convs (fast, default) or fp32 direct
    // convs (--fp32). reference = im2col + fp32 cuBLAS, the slow exactness baseline.
    bool reference = false;

    void load(const std::string& path, Place place = Place::Auto);
    int scale() const { return scale_; }
    // in: device RGB [3, H, W] in [0,1]; out: device RGB [3, s*H, s*W] in [0,1] (caller allocates out).
    // Scratch comes from G.arena (whatever is free); the image is processed in tiles when it doesn't fit.
    void run(float* out, const float* in, int H, int W);
    Upscaler() = default;
    Upscaler(const Upscaler&) = delete;  // owns the weight blob
    Upscaler& operator=(const Upscaler&) = delete;
    ~Upscaler();

    // Shape of the loaded net
    int nf = 0, gc = 0, nb = 0, n_up = 0, scale_ = 1;
    int tiles_used = 0;  // tiles in the last run()

    struct Conv { const uint16_t* w = nullptr; const uint16_t* b = nullptr; int cout = 0, cin = 0; };  // fp16 bits

private:
    struct Rdb { Conv c[5]; };
    struct Rrdb { Rdb r[3]; };
    Conv first_, body_, hr_, last_;
    std::vector<Conv> ups_;
    std::vector<Rrdb> blocks_;
    uint16_t* blob_ = nullptr;  // every weight + bias, fp16
    size_t blob_bytes_ = 0;
    bool blob_host_ = false;
    // Where convs read weights during run(): blob_, or an arena copy of it when blob_ is in host
    // memory (the conv kernels re-read weight slabs per block; over PCIe that costs ~1.7x).
    const uint16_t* wbase_ = nullptr;
    const uint16_t* wp(const uint16_t* p) const { return wbase_ + (p - blob_); }

    // per-run scratch (arena)
    float* wf_ = nullptr;   // fp32 copy of the current conv's weights (reference path)
    float* col_ = nullptr;  // im2col slab (reference path)
    size_t col_cap_ = 0;

    void conv(float* out, size_t ocs, const float* in, size_t ics, int irs, int H, int W, const Conv& c, int up, int act, const char* tag);
    void tile(float* out, const float* in, int H, int W, int y0, int y1, int x0, int x1, int cy0, int cy1, int cx0, int cx1);
};
