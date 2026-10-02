// LoRA training for Anima's DiT. Rank-r adapters on the 10 linears of every block (the modules sd-scripts' lora_anima
// trains: self/cross-attention q, k, v, output and both MLP layers), kept in fp32 on the GPU with their gradients.
// A step noises a latent at sigma (rectified flow), runs the DiT keeping only each block's input, computes the MSE
// against (noise - latent), and walks the blocks backwards, recomputing each block's internals from its saved input
// (gradient checkpointing) before taking it apart. The frozen weights get no gradients; activations and gradients
// stay fp32 (cuBLAS SGEMM for the backward products), so no loss scaling is needed.
#pragma once
#include <array>
#include <string>
#include <vector>

#include "models.h"

struct AnimaTrainer {
    static const int NT = 10;
    static const char* const kShort[NT];   // sa_q sa_k sa_v sa_o ca_q ca_k ca_v ca_o mlp1 mlp2
    static const char* const kModule[NT];  // self_attn.q_proj ... mlp.layer2 (checkpoint / LoRA-file names)

    struct Lora { int in = 0, out = 0; float *A = nullptr, *B = nullptr, *dA = nullptr, *dB = nullptr; };  // A [r, in], B [out, r]
    Dit& dit;
    int rank = 0;
    float alpha = 0.f, scale = 1.f;
    std::vector<std::array<Lora, NT>> lora;  // [block][target]
    float* params = nullptr;  // every A and B, contiguous (n floats)
    float* grads = nullptr;   // the matching gradients
    size_t n = 0;

    explicit AnimaTrainer(Dit& d) : dit(d) {}
    ~AnimaTrainer() { release(); }
    AnimaTrainer(const AnimaTrainer&) = delete;
    AnimaTrainer& operator=(const AnimaTrainer&) = delete;

    // allocate the adapters: A ~ kaiming-uniform(a = sqrt 5) like torch.nn.Linear, B = 0 (the LoRA starts as a no-op)
    void init(int rank, float alpha, unsigned long long seed);
    size_t param_count(int rank);  // floats init(rank) allocates for the adapters (and as many for their gradients)
    void release();

    // scratch (bytes, rounded up) for a step at T image tokens and Lk real context rows: every block's input, one
    // block's recompute and backward (~37 rows of D a token), the final layer, attention probabilities for 4 heads
    static size_t step_scratch(size_t T, size_t Lk) {
        return (65 * T * 2048 + 3 * Lk * 2048 + 8 * T * T + 600 * T) * 4 + ((size_t)64 << 20);
    }
    // ... and for a preview prediction (no saved inputs, no backward)
    static size_t predict_scratch(size_t T, size_t Lk) {
        return (26 * T * 2048 + 3 * Lk * 2048 + 2 * T * T + 300 * T) * 4 + ((size_t)64 << 20);
    }
    void zero_grad();
    Weight& weight(int block, int t);  // the frozen linear a target adapts

    // One example: x_t = (1 - sigma) latent + sigma noise, loss = mean((DiT(x_t, sigma) - (noise - latent))^2).
    // latent, noise [16, Hl, Wl] (model space, device); gradients accumulate (scaled by grad_scale). Returns the loss.
    // nblocks < 28 runs only the first blocks (test hook).
    float step(const float* latent, const float* noise, int Hl, int Wl, float sigma, const Context& c, float grad_scale = 1.f, int nblocks = 0);

    // the DiT with the adapters as they are: velocity v [16, Hl, Wl] at noisy latent x and sigma (previews)
    void predict(float* v, const float* x, int Hl, int Wl, float sigma, const Context& c);

    // test hooks: one block forward (X in place) and its backward (dX: gradient of the output in, of the input out)
    void test_block(int b, float* X, const float* dY, float* dX, int Hp, int Wp, float t, const Context& c);

    // the adapters as an sd-scripts / ComfyUI LoRA file (lora_unet_blocks_<i>_<module>.lora_down/.lora_up/.alpha, fp16)
    void save(const std::string& path, const std::vector<std::pair<std::string, std::string>>& metadata) const;
};
