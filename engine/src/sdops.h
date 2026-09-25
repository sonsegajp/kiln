// Ops for the Stable Diffusion family (SDXL UNet, CLIP text encoders, KL VAE). Activations are fp32
// ([C, P] channel-major for convolutions, [T, C] row-major for transformers); parameters that are
// small (norm scales, biases) are fp32, matrices are Weights.
#pragma once
#include <cuda_fp16.h>

#include "gpu.h"

// GroupNorm over x [C, P] with `groups` groups, fp32 statistics, optional SiLU, y fp32 or fp16 (y16).
// Exactly one of y / y16 is non-null; y may alias x.
void group_norm(float* y, __half* y16, const float* x, const float* gamma, const float* beta, int C, int P, int groups, float eps, bool silu);
// The same in two steps, for tensors processed in row bands: statistics over the whole x [C, P]
// (stat[g] = mean, rstd; x may be mapped system RAM), then y16 [C, n] for n pixels of every channel
// starting at x (channel stride cs), e.g. one band of rows.
void group_norm_stats(float2* stat, const float* x, int C, size_t P, int groups, float eps);
void group_norm_apply16(__half* y16, const float* x, size_t cs, const float2* stat, const float* gamma, const float* beta, int C, size_t n,
                        int groups, bool silu);
// y16 [C, n] = fp16(x[c * cs + j] * scale), a band of a larger tensor
void to16_band(__half* y16, const float* x, size_t cs, int C, size_t n, float scale);
// LayerNorm over rows of x [rows, dim]: y = (x - mean) / sqrt(var + eps) * w + b. y may alias x.
void layer_norm(float* y, const float* x, const float* w, const float* b, int rows, int dim, float eps);
// GEGLU: x [T, 2*inner] -> y [T, inner] = x[:, :inner] * gelu(x[:, inner:]) (erf gelu)
void geglu(float* y, const float* x, int T, int inner);
// 3x3 pad-1 convolutions with too few channels for the implicit-GEMM tiles (fp32 math, fp32 weights
// [Cout][Cin*9]). small_in: Cin <= 4, fp32 input [Cin, H, W]. small_out: Cout 3 or 4, fp16 input band
// [Cin, Hin, W] (rows beyond it are zero), output rows [0, rows) at input row row_off + oy, written
// to out (channel stride out_cs).
void conv3x3_small_in(float* out, const float* in, const float* w, const float* b, int Cin, int Cout, int H, int W);
void conv3x3_small_out(float* out, size_t out_cs, const __half* in, int Cin, int Hin, int W, int row_off, int rows, const float* w,
                       const float* b, int Cout);
// 1x1 conv, Cin/Cout <= 8, fp32: out [Cout, P] = w [Cout][Cin] in + b
void conv1x1_small(float* out, const float* in, const float* w, const float* b, int Cin, int Cout, size_t P);
void quick_gelu(float* x, size_t n);                                       // x * sigmoid(1.702 x)
void bias_rows(float* y, const float* b, int rows, int dim);               // y[r, :] += b
void bias_channels(float* y, const float* b, int C, int P);                // y[c, :] += b[c]
void add_channels(float* y, const float* v, int C, int P);                 // y[c, :] += v[c] (e.g. time embedding)
