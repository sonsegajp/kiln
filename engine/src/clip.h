// CLIP text encoders for the SD family: CLIP-L (Hugging Face CLIPTextModel keys) and OpenCLIP bigG.
// Output is the penultimate hidden state (no final norm), token weights applied the usual way
// (z = z_empty + (z - z_empty) * w per token), plus OpenCLIP's projected pooled vector.
#pragma once
#include <string>
#include <vector>

#include "gpu.h"
#include "safetensors.h"
#include "sdxl.h"

class ClipText {
public:
    // CLIP-L: prefix like "conditioner.embedders.0.transformer.text_model."
    void load_hf(const SafeTensors& st, const std::string& prefix, Place place);
    // OpenCLIP: prefix like "conditioner.embedders.1.model."
    void load_openclip(const SafeTensors& st, const std::string& prefix, Place place);
    void free();
    int dim() const { return dim_; }
    bool has_projection() const { return proj_.p != nullptr; }
    // ids/weights: chunks of 77 tokens (BOS .. EOS, padded). hidden [chunks*77, dim] on the GPU;
    // pooled [dim] (projected, first chunk) when non-null and the model has a projection.
    void encode(const std::vector<int>& ids, const std::vector<float>& weights, float* hidden, float* pooled);

private:
    struct Layer {
        Norm32 ln1, ln2;
        Linear16 q, k, v, qkv, o, fc1, fc2;  // HF: q, k, v; OpenCLIP: fused qkv
    };
    std::vector<Layer> layers_;
    Weight tok_;              // [49408, dim] fp16
    float* pos_ = nullptr;    // [77, dim] fp32
    Norm32 ln_final_;
    Weight proj_;             // pooled = x_eos @ text_projection, stored transposed for linear()
    int dim_ = 0, heads_ = 0, pad_id_ = 0;
    bool quick_gelu_ = false;
};
