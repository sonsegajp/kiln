// Booru taggers: SmilingWolf's WD v3 models as timm checkpoints (model.safetensors + selected_tags.csv + config.json).
// ViT (wd-vit-tagger-v3, wd-vit-large-tagger-v3: no class token, tanh GELU, final norm, average pool) and EVA02
// (wd-eva02-large-tagger-v3: class token, 2D rotary positions on q/k, SwiGLU MLP with a norm, average pool + fc_norm).
// An image in, a sigmoid probability for every tag of selected_tags.csv out. Reference: ref/wd_tagger_ref.py.
#pragma once
#include <string>
#include <vector>

#include "gpu.h"
#include "sdxl.h"

class Tagger {
public:
    void load(const std::string& dir, Place place);
    void free();
    bool loaded() const { return classes_ > 0; }
    const std::string& dir() const { return dir_; }
    int size() const { return size_; }  // input side in pixels (448)
    int classes() const { return classes_; }
    // img [3, size, size] on the GPU (preprocessed: BGR, [-1, 1]) -> probs [classes] on the host
    void run(const float* img, std::vector<float>& probs);

private:
    struct Block {
        Norm32 n1, n2, mlp_norm;
        Linear16 qkv, q, k, v, proj, fc1, fc1_g, fc1_x, fc2;
    };
    std::vector<Block> blocks_;
    Linear16 patch_, head_;
    float* pos_ = nullptr;  // [tokens, D]
    float* cls_ = nullptr;  // [D] (EVA02)
    Norm32 norm_, fc_norm_;
    float *rsin_ = nullptr, *rcos_ = nullptr;  // EVA02 rotary tables [grid*grid, 64]
    int dim_ = 0, heads_ = 0, patch_px_ = 16, size_ = 448, grid_ = 28, classes_ = 0, hidden_ = 0;
    bool eva_ = false;
    std::string dir_;
};

// The models' preprocessing on the CPU: the image on a white square (alpha composited onto white), PIL-style bicubic
// resize to size x size, scaled to [-1, 1], channels in BGR order -> out [3, size, size].
bool tagger_preprocess(const std::string& path, int size, std::vector<float>& out, std::string& err);
