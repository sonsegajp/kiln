// YOLOv8 detector (Ultralytics detection checkpoints converted by tools/convert_yolo.py), used to
// find faces for the face-detailing pass. Own kernels: letterbox, conv 1x1/3x3 at stride 1/2 with
// fused bias/SiLU/residual, SPPF max-pool, nearest upsample, DFL box decode; NMS on the host.
//
// Weights are fused conv+BN. Default: fp16 weights + an fp16x2 (HFMA2) implicit-GEMM conv that
// flushes partial sums into fp32 every 32 k (fast on tensor-core-less Turing; head output within
// ~1e-3 of fp32). fp32_weights = true: fp32 weights + im2col + cuBLAS SGEMM (matches the PyTorch
// reference to ~1e-6). Activations live in one block of G.arena for the duration of detect(),
// laid out by a liveness planner (~35-75 MB for a 640 input).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "gpu.h"

struct Box { float x0, y0, x1, y1, score; };   // pixel coords in the input image

struct FaceDetector {
    // ---- options, set before load() ----
    bool fp32_weights = false;       // true: exact fp32 path (2x the weight VRAM, ~2x slower)
    size_t col_budget = 32u << 20;   // fp32 path: im2col scratch cap in bytes (convs run in pixel chunks above it)
    int max_det = 300;               // Ultralytics default

    // ---- debug / stats (valid after detect()) ----
    bool keep_debug = false;         // copy the letterboxed input and head outputs to the host
    std::vector<float> dbg_input;    // [3, in_h, in_w] letterboxed network input
    std::vector<float> dbg_raw;      // [no, N] raw head output (4*reg_max box logits + nc class logits)
    std::vector<float> dbg_decoded;  // [4+nc, N] xywh (letterbox pixels) + class probabilities
    int in_h = 0, in_w = 0, num_anchors = 0;
    float last_gpu_ms = 0, last_total_ms = 0;
    size_t last_arena_bytes = 0;
    int nc = 0, imgsz = 640;
    std::vector<std::string> names;

    FaceDetector();
    ~FaceDetector();
    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;

    void load(const std::string& safetensors_path, const std::string& json_path, Place place = Place::Auto);
    void unload();
    bool loaded() const;
    size_t weight_bytes() const;
    bool weights_on_host() const;

    // rgb: device [3, H, W] in [0,1]. Letterbox exactly like Ultralytics predict (long side -> 640 keeping
    // aspect, pad to a multiple of 32 with 114/255 gray, centered), run, decode (DFL), confidence filter,
    // NMS, map back to input pixels. Sorted by score, highest first.
    std::vector<Box> detect(const float* rgb, int H, int W, float conf = 0.3f, float iou = 0.5f);

    struct Impl;
private:
    std::unique_ptr<Impl> impl;
};
