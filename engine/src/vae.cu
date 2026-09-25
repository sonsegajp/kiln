// Wan2.1 / Qwen-Image VAE decoder, single-frame path. For one frame every causal 3D conv
// reduces to a 2D conv with its last temporal tap, and the temporal upsamplers are skipped.
//
// Past the middle attention the decoder is purely local (3x3 convs, per-pixel norms,
// nearest upsampling), so it can run in horizontal bands with a halo and still produce
// exactly the same pixels as a full decode. Bands are only used when VRAM is short.
//
// Two precisions (Vae::precision; G.fp16 == false also selects FP32):
//   FP32  im2col + cuBLAS SGEMM, fp32 attention: the exact reference path (matches ComfyUI's fp32 VAE).
//   FP16  every conv of the resblocks / resamplers and both attention GEMMs run as fp16x2 implicit
//         GEMMs (vaeconv.cuh) with fp32 accumulation every 32 k. Activations are still STORED in fp32
//         between layers; only GEMM operands are rounded to fp16, right before use.
//         Overflow margin (fp16 max 65504), this checkpoint:
//         - conv inputs that come out of an RMS norm are bounded by sqrt(C) * max|gamma| <= 48.4, and
//           no 32-long window of |w| sums past 5.2, so an fp16 partial stays < 260 (even a whole
//           output row, sum|w| * bound, stays < 7000);
//         - the raw residual stream feeds the shortcut / upsample / downsample convs; its measured max
//           over golden latents, user renders and synthetic black/white/noise/checker/colour-bar
//           images is 43 (encoder) and 18 (decoder), i.e. partials < 250. RAW_SCALE (a power of two,
//           undone exactly in the epilogue) is the knob if another checkpoint needs headroom; it must
//           stay a constant, a data-dependent scale would break banded == unbanded;
//         - attention: |q| <= 9, |k| <= 6.2, |v| <= 2.1, scaled scores <= 44 (kept fp32 through the
//           softmax; probabilities are fp16).
//         Accuracy vs FP32: ~72 dB PSNR on a 512x768 decode, encoder latents rel. L2 ~3e-4.
#include "models.h"
#include "vaeconv.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>

static const int HALO = 16;  // latent rows; the upsample stack's receptive field needs ~12.3

// Scale applied to the raw residual stream before it is rounded to fp16 (undone in the epilogue).
// 1: the measured residual magnitudes need no headroom (see the top of the file).
static const float RAW_SCALE = 1.f;

void Vae::load(const std::string& path, Place place) {
    st = std::make_unique<SafeTensors>(path);
    auto W = [&](const std::string& n, bool t3 = false) { return upload_weight(st->get(n), place, t3); };
    auto res = [&](const std::string& p, int cin, int cout) {
        Res r;
        r.cin = cin; r.cout = cout;
        r.n1 = W(p + "residual.0.gamma");
        r.c1 = W(p + "residual.2.weight", true); r.c1b = W(p + "residual.2.bias");
        r.n2 = W(p + "residual.3.gamma");
        r.c2 = W(p + "residual.6.weight", true); r.c2b = W(p + "residual.6.bias");
        if (st->has(p + "shortcut.weight")) { r.sc = W(p + "shortcut.weight", true); r.scb = W(p + "shortcut.bias"); }
        return r;
    };
    conv2 = W("conv2.weight", true); conv2b = W("conv2.bias");
    conv1 = W("decoder.conv1.weight", true); conv1b = W("decoder.conv1.bias");
    int top = (int)conv1.rows;  // 384
    auto attn_w = [&](const std::string& p) {
        Attn a;
        a.n = W(p + "norm.gamma");
        a.qkv = W(p + "to_qkv.weight"); a.qkvb = W(p + "to_qkv.bias");
        a.proj = W(p + "proj.weight"); a.projb = W(p + "proj.bias");
        return a;
    };
    mid0 = res("decoder.middle.0.", top, top);
    attn = attn_w("decoder.middle.1.");
    mid1 = res("decoder.middle.2.", top, top);

    int idx = 0, cin = top;
    for (int s = 0; s < 4; s++) {
        Up u;
        for (int r = 0; r < 3; r++, idx++) {
            std::string p = "decoder.upsamples." + std::to_string(idx) + ".";
            int cout = (int)st->get(p + "residual.2.weight").shape[0];
            u.res.push_back(res(p, cin, cout));
            cin = cout;
        }
        u.cout = cin;
        std::string p = "decoder.upsamples." + std::to_string(idx) + ".";
        if (s < 3) {
            u.rs = W(p + "resample.1.weight"); u.rsb = W(p + "resample.1.bias");
            cin = (int)u.rs.rows;
            idx++;
        }
        ups.push_back(u);
    }
    head_n = W("decoder.head.0.gamma");
    head_c = W("decoder.head.2.weight", true); head_cb = W("decoder.head.2.bias");

    // encoder (single frame: causal 3D convs keep their last temporal tap; time_conv is unused)
    e_conv1 = W("encoder.conv1.weight", true); e_conv1b = W("encoder.conv1.bias");
    idx = 0;
    cin = (int)e_conv1.rows;  // 96
    for (int s = 0; s < 4; s++) {
        Down d;
        for (int r = 0; r < 2; r++, idx++) {
            std::string p = "encoder.downsamples." + std::to_string(idx) + ".";
            int cout = (int)st->get(p + "residual.2.weight").shape[0];
            d.res.push_back(res(p, cin, cout));
            cin = cout;
        }
        d.cout = cin;
        if (s < 3) {
            std::string p = "encoder.downsamples." + std::to_string(idx) + ".";
            d.ds = W(p + "resample.1.weight"); d.dsb = W(p + "resample.1.bias");
            idx++;
        }
        downs.push_back(d);
    }
    e_mid0 = res("encoder.middle.0.", cin, cin);
    e_attn = attn_w("encoder.middle.1.");
    e_mid1 = res("encoder.middle.2.", cin, cin);
    e_head_n = W("encoder.head.0.gamma");
    e_head_c = W("encoder.head.2.weight", true); e_head_cb = W("encoder.head.2.bias");
    q_conv = W("conv1.weight", true); q_convb = W("conv1.bias");

    // the fp16 kernels cover every conv of this architecture; check anyway (FP32 otherwise)
    fp16_ok = true;
    auto chk = [&](const Res& r) {
        fp16_ok &= vc::eligible(r.cin, r.cout) && vc::eligible(r.cout, r.cout);
        if (r.sc) fp16_ok &= vc::eligible(r.cin, r.cout);
    };
    for (auto* r : {&mid0, &mid1, &e_mid0, &e_mid1}) chk(*r);
    for (auto& u : ups) {
        for (auto& r : u.res) chk(r);
        if (u.rs) fp16_ok &= vc::eligible(u.cout, (int)u.rs.rows);
    }
    for (auto& d : downs) {
        for (auto& r : d.res) chk(r);
        if (d.ds) fp16_ok &= vc::eligible(d.cout, (int)d.ds.rows);
    }
    fp16_ok &= vc::eligible(e_mid1.cout, (int)e_head_c.rows) && head_c.rows == 3 && e_conv1.cols == 27;
}

bool Vae::use_fp16() const { return precision == Precision::FP16 && G.fp16 && fp16_ok; }

// ---------------------------------------------------------------------------
// debug probe: max |x| of named tensors (syncs; only when `probe` is set)
// ---------------------------------------------------------------------------
__global__ void k_vae_absmax(const float* x, size_t n, unsigned* out) {
    float m = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = fabsf(x[i]);
        m = v != v ? INFINITY : fmaxf(m, v);  // NaN counts as overflow
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}
__global__ void k_vae_absmax16(const __half* x, size_t n, unsigned* out) {
    float m = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = fabsf(__half2float(x[i]));
        m = v != v ? INFINITY : fmaxf(m, v);
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}

void Vae::note(const std::string& name, const void* p, size_t n, bool half) {
    if (!probe) return;
    size_t m = G.arena.mark();
    unsigned* d = (unsigned*)G.arena.f(1);
    CK(cudaMemsetAsync(d, 0, 4, G.stream));
    if (half) k_vae_absmax16<<<256, 256, 0, G.stream>>>((const __half*)p, n, d);
    else k_vae_absmax<<<256, 256, 0, G.stream>>>((const float*)p, n, d);
    unsigned h = 0;
    CK(cudaMemcpyAsync(&h, d, 4, cudaMemcpyDeviceToHost, G.stream));
    gpu_sync();
    G.arena.release(m);
    float v;
    memcpy(&v, &h, 4);
    probe->push_back({name, v});
}

// ---------------------------------------------------------------------------
// FP32 path (exact)
// ---------------------------------------------------------------------------
// x <- ResidualBlock(x); a, b are scratch of the same capacity.
void Vae::resblock(float*& x, float*& a, float*& b, const Res& r, int H, int W, const std::string& tag) {
    int P = H * W;
    note(tag + ".in", x, (size_t)r.cin * P);
    rms_channels(a, x, r.n1, r.cin, P, true);
    note(tag + ".h1", a, (size_t)r.cin * P);
    conv2d(b, a, r.cin, H, W, r.c1, &r.c1b, 3);
    note(tag + ".t", b, (size_t)r.cout * P);
    rms_channels(b, b, r.n2, r.cout, P, true);
    note(tag + ".h2", b, (size_t)r.cout * P);
    conv2d(a, b, r.cout, H, W, r.c2, &r.c2b, 3);
    if (r.sc) {
        conv2d(b, x, r.cin, H, W, r.sc, &r.scb, 1);
        add_inplace(a, b, (size_t)r.cout * P);
    } else {
        add_inplace(a, x, (size_t)r.cout * P);
    }
    std::swap(x, a);
}

// x += proj(attention(norm(x))): one head over every pixel. a, b are [C, H*W] scratch.
void Vae::attention_block(float* x, float* a, float* b, const Attn& at, int C, int H, int W) {
    ProfScope ps("vae_attn");
    int P = H * W;
    size_t m = G.arena.mark();
    float* qkv = G.arena.f((size_t)3 * C * P);
    float* t = G.arena.f((size_t)3 * C * P);
    rms_channels(a, x, at.n, C, P, false);
    conv2d(qkv, a, C, H, W, at.qkv, &at.qkvb, 1);
    for (int j = 0; j < 3; j++) transpose(t + (size_t)j * C * P, qkv + (size_t)j * C * P, C, P);
    attention(qkv, C, t, C, t + (size_t)C * P, C, t + 2 * (size_t)C * P, C, P, P, 1, C, false);
    transpose(a, qkv, P, C);
    conv2d(b, a, C, H, W, at.proj, &at.projb, 1);
    add_inplace(x, b, (size_t)C * P);
    G.arena.release(m);
}

// Runs the upsample stack on latent rows [row0, row0+rows) of mid [C, Hl, Wl] (with halo),
// writing the matching 8x rows into out_rgb [3, 8Hl, 8Wl].
void Vae::upstack(const float* mid, int Hl, int Wl, float* out_rgb, int row0, int rows) {
    int s0 = std::max(0, row0 - HALO), s1 = std::min(Hl, row0 + rows + HALO), hs = s1 - s0;
    int C = mid0.cout;
    size_t area = (size_t)hs * Wl;
    size_t cap = 0;
    {
        size_t scale = 1;
        cap = std::max(cap, (size_t)C * area);
        for (auto& u : ups) {
            for (auto& r : u.res) cap = std::max(cap, (size_t)std::max(r.cin, r.cout) * area * scale);
            if (u.rs) { scale *= 4; cap = std::max(cap, (size_t)u.rs.rows * area * scale); }
        }
    }
    size_t m = G.arena.mark();
    float* x = G.arena.f(cap);
    float* a = G.arena.f(cap);
    float* b = G.arena.f(cap);
    CK(cudaMemcpy2DAsync(x, area * 4, mid + (size_t)s0 * Wl, (size_t)Hl * Wl * 4, area * 4, C, cudaMemcpyDeviceToDevice, G.stream));

    int H = hs, W = Wl;
    for (size_t s = 0; s < ups.size(); s++) {
        auto& u = ups[s];
        for (size_t r = 0; r < u.res.size(); r++) resblock(x, a, b, u.res[r], H, W, "dec.up" + std::to_string(s) + ".r" + std::to_string(r));
        if (u.rs) {
            note("dec.up" + std::to_string(s) + ".rs.in", x, (size_t)u.cout * H * W);
            conv2d_up2(a, x, u.cout, H, W, u.rs, &u.rsb);
            std::swap(x, a);
            H *= 2; W *= 2;
        }
    }
    int cl = ups.back().cout;
    note("dec.head.in", x, (size_t)cl * H * W);
    rms_channels(a, x, head_n, cl, H * W, true);
    conv2d(b, a, cl, H, W, head_c, &head_cb, 3);

    // crop the band out of the halo'd result
    int off = (row0 - s0) * 8, n = rows * 8, OW = 8 * Wl;
    for (int c = 0; c < 3; c++)
        CK(cudaMemcpyAsync(out_rgb + ((size_t)c * 8 * Hl + (size_t)row0 * 8) * OW, b + ((size_t)c * H + off) * W,
                           (size_t)n * OW * 4, cudaMemcpyDeviceToDevice, G.stream));
    G.arena.release(m);
}

// ---------------------------------------------------------------------------
// FP16 path
// ---------------------------------------------------------------------------
// x <- ResidualBlock(x). t: fp32 scratch (capacity cout*P); h: fp16 scratch (capacity cout*P,
// plus cin*P more when the block has a shortcut). x may be swapped with t.
void Vae::resblock16(float*& x, float*& t, __half* h, const Res& r, int H, int W, const std::string& tag) {
    int P = H * W;
    note(tag + ".in", x, (size_t)r.cin * P);
    vc::rms16(h, x, r.n1, r.cin, P, true);
    note(tag + ".h1", h, (size_t)r.cin * P, true);
    vc::conv(t, h, r.cin, H, W, r.c1, &r.c1b, nullptr, vc::M3, 1.f);
    note(tag + ".t", t, (size_t)r.cout * P);
    vc::rms16(h, t, r.n2, r.cout, P, true);
    note(tag + ".h2", h, (size_t)r.cout * P, true);
    if (r.sc) {
        __half* xs = h + (size_t)r.cout * P;
        vc::to16(xs, x, (size_t)r.cin * P, RAW_SCALE);
        vc::conv(t, xs, r.cin, H, W, r.sc, &r.scb, nullptr, vc::M1, 1.f / RAW_SCALE);  // t is free again
        vc::conv(t, h, r.cout, H, W, r.c2, &r.c2b, t, vc::M3, 1.f);
        std::swap(x, t);
    } else {
        vc::conv(x, h, r.cout, H, W, r.c2, &r.c2b, x, vc::M3, 1.f);  // x += conv(h), in place
    }
}

// fp16 path of attention_block. The [T, T] scores are computed transposed (keys x queries) in query
// chunks, so both GEMMs read channel-first operands and write channel-first results:
//   St = K^T Q / sqrt(C) (fp32), Pt = softmax over keys (fp16), O = V Pt -> a [C, T].
// Keys are padded to a multiple of 128 (zero K columns / V rows, zero probabilities).
// a: fp32 [C, T] scratch; h: fp16 scratch of C*T elements.
void Vae::attention_block16(float* x, float* a, __half* h, const Attn& at, int C, int H, int W) {
    ProfScope ps("vae_attn");
    const int T = H * W, Tp = (T + 127) / 128 * 128;
    size_t m = G.arena.mark();
    float* qkv = G.arena.f((size_t)3 * C * T);
    __half* q16 = (__half*)G.arena.f(((size_t)C * T + 1) / 2);
    __half* k16 = (__half*)G.arena.f((size_t)C * Tp / 2);
    __half* v16 = (__half*)G.arena.f((size_t)C * Tp / 2);
    vc::rms16(h, x, at.n, C, T, false);
    vc::conv(qkv, h, C, H, W, at.qkv, &at.qkvb, nullptr, vc::M1, 1.f);
    const float *q = qkv, *k = qkv + (size_t)C * T, *v = qkv + 2 * (size_t)C * T;
    note("attn.q", q, (size_t)C * T);
    note("attn.k", k, (size_t)C * T);
    note("attn.v", v, (size_t)C * T);
    vc::to16(q16, q, (size_t)C * T, 1.f);
    vc::k_pad16<<<vc::nb((size_t)C * Tp), 256, 0, G.stream>>>(k16, k, C, T, Tp);
    vc::k_tpad16<<<dim3(Tp / 32, (C + 31) / 32), dim3(32, 8), 0, G.stream>>>(v16, v, C, T, Tp);

    // query chunk: the fp32 scores + fp16 probabilities of n queries, bounded by the arena
    size_t fr = G.arena.free_bytes(), slack = (size_t)16 << 20;
    size_t budget = std::min<size_t>(fr > slack ? fr - slack : 0, (size_t)256 << 20);
    int n = (int)std::min<size_t>(T, budget / ((size_t)Tp * 6));
    if (n < T) n = n / 128 * 128;
    if (n < 128) n = std::min(T, 128);
    float* St = G.arena.f((size_t)Tp * n);
    __half* Pt = (__half*)G.arena.f(((size_t)Tp * n + 1) / 2);
    const float scale = 1.f / sqrtf((float)C);
    for (int q0 = 0; q0 < T; q0 += n) {
        int nn = std::min(n, T - q0);
        vc::gemm16(St, nn, nullptr, k16, C, Tp, q16 + q0, T, nn, scale);
        if (q0 == 0) note("attn.scores", St, (size_t)T * nn);
        vc::k_colsoftmax16<<<(nn + 31) / 32, dim3(32, 16), 0, G.stream>>>(Pt, St, T, Tp, nn);
        vc::gemm16(a + q0, T, nullptr, v16, Tp, C, Pt, nn, nn, 1.f);
    }
    note("attn.out", a, (size_t)C * T);
    vc::to16(h, a, (size_t)C * T, 1.f);
    vc::conv(x, h, C, H, W, at.proj, &at.projb, x, vc::M1, 1.f);  // x += proj(o)
    G.arena.release(m);
}

// Per latent pixel of a band: fp32 elements of the largest stage tensor and fp16 elements of the
// largest conv operand (a shortcut block holds norm(t) and its raw input side by side).
void Vae::up_caps(size_t& cap32, size_t& cap16) const {
    size_t scale = 1;
    cap32 = cap16 = mid0.cout;
    for (auto& u : ups) {
        for (auto& r : u.res) {
            cap32 = std::max(cap32, (size_t)std::max(r.cin, r.cout) * scale);
            cap16 = std::max(cap16, (size_t)(r.sc ? r.cin + r.cout : std::max(r.cin, r.cout)) * scale);
        }
        if (u.rs) { scale *= 4; cap32 = std::max(cap32, (size_t)u.rs.rows * scale); }
    }
    cap16 = std::max(cap16, cap32);  // the head's norm output is fp16 too
}

void Vae::upstack16(const float* mid, int Hl, int Wl, float* out_rgb, int row0, int rows) {
    int s0 = std::max(0, row0 - HALO), s1 = std::min(Hl, row0 + rows + HALO), hs = s1 - s0;
    int C = mid0.cout;
    size_t area = (size_t)hs * Wl, c32, c16;
    up_caps(c32, c16);
    size_t m = G.arena.mark();
    float* x = G.arena.f(c32 * area);
    float* t = G.arena.f(c32 * area);
    __half* h = (__half*)G.arena.f((c16 * area + 1) / 2);
    CK(cudaMemcpy2DAsync(x, area * 4, mid + (size_t)s0 * Wl, (size_t)Hl * Wl * 4, area * 4, C, cudaMemcpyDeviceToDevice, G.stream));

    int H = hs, W = Wl;
    for (size_t s = 0; s < ups.size(); s++) {
        auto& u = ups[s];
        for (size_t r = 0; r < u.res.size(); r++) resblock16(x, t, h, u.res[r], H, W, "dec.up" + std::to_string(s) + ".r" + std::to_string(r));
        if (u.rs) {
            note("dec.up" + std::to_string(s) + ".rs.in", x, (size_t)u.cout * H * W);
            vc::to16(h, x, (size_t)u.cout * H * W, RAW_SCALE);
            vc::conv(t, h, u.cout, H, W, u.rs, &u.rsb, nullptr, vc::MUP, 1.f / RAW_SCALE);
            std::swap(x, t);
            H *= 2; W *= 2;
        }
    }
    int cl = ups.back().cout;
    note("dec.head.in", x, (size_t)cl * H * W);
    vc::rms16(h, x, head_n, cl, H * W, true);
    note("dec.head.h", h, (size_t)cl * H * W, true);
    vc::head3(t, h, cl, H, W, head_c, head_cb);

    int off = (row0 - s0) * 8, n = rows * 8, OW = 8 * Wl;
    for (int c = 0; c < 3; c++)
        CK(cudaMemcpyAsync(out_rgb + ((size_t)c * 8 * Hl + (size_t)row0 * 8) * OW, t + ((size_t)c * H + off) * W,
                           (size_t)n * OW * 4, cudaMemcpyDeviceToDevice, G.stream));
    G.arena.release(m);
}

void Vae::decode(const float* latent, int Hl, int Wl, float* out) {
    const bool f16 = use_fp16();
    int P = Hl * Wl, C = mid0.cout;
    size_t m = G.arena.mark();
    float* z = G.arena.f((size_t)16 * P);
    float* x = G.arena.f((size_t)C * P);
    float* a = G.arena.f((size_t)C * P);
    float* b = G.arena.f((size_t)C * P);

    conv2d(z, latent, 16, Hl, Wl, conv2, &conv2b, 1);
    conv2d(x, z, 16, Hl, Wl, conv1, &conv1b, 3);
    if (f16) {
        resblock16(x, a, (__half*)b, mid0, Hl, Wl, "dec.mid0");
        note("dec.attn.in", x, (size_t)C * P);
        attention_block16(x, a, (__half*)b, attn, C, Hl, Wl);
        resblock16(x, a, (__half*)b, mid1, Hl, Wl, "dec.mid1");
    } else {
        resblock(x, a, b, mid0, Hl, Wl, "dec.mid0");
        note("dec.attn.in", x, (size_t)C * P);
        attention_block(x, a, b, attn, C, Hl, Wl);
        resblock(x, a, b, mid1, Hl, Wl, "dec.mid1");
    }

    // Band height: the largest that fits.
    size_t per_row, reserve;
    if (f16) {
        size_t c32, c16;
        up_caps(c32, c16);
        per_row = (8 * c32 + 2 * c16) * Wl;
        reserve = (size_t)16 << 20;  // weight slabs + alignment
    } else {
        // 3 buffers of (last-stage channels * 64) floats per latent pixel, plus the im2col slab
        per_row = (size_t)3 * 4 * std::max<size_t>((size_t)ups.back().cout * 64, (size_t)C) * Wl;
        reserve = (size_t)112 << 20;
    }
    size_t avail = G.arena.free_bytes() > reserve ? G.arena.free_bytes() - reserve : 0;
    int fit = (int)(avail / per_row);
    int rows = fit >= Hl ? Hl : std::max(1, fit - 2 * HALO);
    if (fit < 2 * HALO + 1 && fit < Hl) throw std::runtime_error("not enough VRAM to decode even one band");
    if (band_rows > 0) rows = std::min(rows, band_rows);
    for (int r0 = 0; r0 < Hl; r0 += rows) {
        if (f16) upstack16(x, Hl, Wl, out, r0, std::min(rows, Hl - r0));
        else upstack(x, Hl, Wl, out, r0, std::min(rows, Hl - r0));
    }
    G.arena.release(m);
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------
static const int EHALO = 10;  // latent rows (80 px); the receptive field of the down stack is ~75 px

// Runs conv1 + the down stages on the input rows for latent rows [lrow0, lrow0+lrows) (plus halo),
// writing those latent rows of the pre-middle features into mid [C, Hl, Wl].
void Vae::downstack(const float* rgb, int H, int W, float* mid, int lrow0, int lrows) {
    int Hl = H / 8, Wl = W / 8;
    int s0 = std::max(0, lrow0 - EHALO), s1 = std::min(Hl, lrow0 + lrows + EHALO), hs = (s1 - s0) * 8;
    size_t area = (size_t)hs * W, cap = 0, scale = 1;
    for (auto& d : downs) {
        for (auto& r : d.res) cap = std::max(cap, (size_t)std::max(r.cin, r.cout) * area / scale);
        if (d.ds) scale *= 4;
    }
    cap = std::max(cap, (size_t)e_conv1.rows * area);
    size_t m = G.arena.mark();
    float* in = G.arena.f(3 * area);
    float* x = G.arena.f(cap);
    float* a = G.arena.f(cap);
    float* b = G.arena.f(cap);
    CK(cudaMemcpy2DAsync(in, area * 4, rgb + (size_t)s0 * 8 * W, (size_t)H * W * 4, area * 4, 3, cudaMemcpyDeviceToDevice, G.stream));
    int h = hs, w = W;
    conv2d(x, in, 3, h, w, e_conv1, &e_conv1b, 3);
    for (size_t s = 0; s < downs.size(); s++) {
        auto& d = downs[s];
        for (size_t r = 0; r < d.res.size(); r++) resblock(x, a, b, d.res[r], h, w, "enc.down" + std::to_string(s) + ".r" + std::to_string(r));
        if (d.ds) {
            note("enc.down" + std::to_string(s) + ".ds.in", x, (size_t)d.cout * h * w);
            conv2d_s2(a, x, d.cout, h, w, d.ds, &d.dsb);
            std::swap(x, a);
            h /= 2; w /= 2;
        }
    }
    int C = downs.back().cout, off = lrow0 - s0;
    CK(cudaMemcpy2DAsync(mid + (size_t)lrow0 * Wl, (size_t)Hl * Wl * 4, x + (size_t)off * Wl, (size_t)h * Wl * 4,
                         (size_t)lrows * Wl * 4, C, cudaMemcpyDeviceToDevice, G.stream));
    G.arena.release(m);
}

// Per 64 input pixels of a band (stage 3 runs at 1/64 of the area): fp32 elements of the largest
// stage tensor, fp16 elements of the largest conv operand.
void Vae::down_caps(size_t& cap32, size_t& cap16) const {
    cap32 = cap16 = (size_t)e_conv1.rows * 64;
    size_t scale = 1;
    for (auto& d : downs) {
        for (auto& r : d.res) {
            cap32 = std::max(cap32, (size_t)std::max(r.cin, r.cout) * 64 / scale);
            cap16 = std::max(cap16, (size_t)(r.sc ? r.cin + r.cout : std::max(r.cin, r.cout)) * 64 / scale);
        }
        if (d.ds) scale *= 4;
    }
    cap16 = std::max(cap16, cap32);
}

void Vae::downstack16(const float* rgb, int H, int W, float* mid, int lrow0, int lrows) {
    int Hl = H / 8, Wl = W / 8;
    int s0 = std::max(0, lrow0 - EHALO), s1 = std::min(Hl, lrow0 + lrows + EHALO), hs = (s1 - s0) * 8;
    size_t area = (size_t)hs * W, c32, c16;
    down_caps(c32, c16);
    size_t m = G.arena.mark();
    float* x = G.arena.f(c32 * area / 64);
    float* t = G.arena.f(c32 * area / 64);
    __half* hb = (__half*)G.arena.f((c16 * area / 64 + 1) / 2);
    {
        size_t mi = G.arena.mark();
        float* in = G.arena.f(3 * area);
        CK(cudaMemcpy2DAsync(in, area * 4, rgb + (size_t)s0 * 8 * W, (size_t)H * W * 4, area * 4, 3, cudaMemcpyDeviceToDevice, G.stream));
        vc::conv_in3(x, in, hs, W, e_conv1, e_conv1b);
        G.arena.release(mi);
    }
    int h = hs, w = W;
    for (size_t s = 0; s < downs.size(); s++) {
        auto& d = downs[s];
        for (size_t r = 0; r < d.res.size(); r++) resblock16(x, t, hb, d.res[r], h, w, "enc.down" + std::to_string(s) + ".r" + std::to_string(r));
        if (d.ds) {
            note("enc.down" + std::to_string(s) + ".ds.in", x, (size_t)d.cout * h * w);
            vc::to16(hb, x, (size_t)d.cout * h * w, RAW_SCALE);
            vc::conv(t, hb, d.cout, h, w, d.ds, &d.dsb, nullptr, vc::MS2, 1.f / RAW_SCALE);
            std::swap(x, t);
            h /= 2; w /= 2;
        }
    }
    int C = downs.back().cout, off = lrow0 - s0;
    CK(cudaMemcpy2DAsync(mid + (size_t)lrow0 * Wl, (size_t)Hl * Wl * 4, x + (size_t)off * Wl, (size_t)h * Wl * 4,
                         (size_t)lrows * Wl * 4, C, cudaMemcpyDeviceToDevice, G.stream));
    G.arena.release(m);
}

void Vae::encode(const float* rgb, int H, int W, float* latent) {
    const bool f16 = use_fp16();
    int Hl = H / 8, Wl = W / 8, P = Hl * Wl, C = downs.back().cout;
    size_t m = G.arena.mark();
    float* x = G.arena.f((size_t)C * P);
    float* a = G.arena.f((size_t)C * P);
    float* b = G.arena.f((size_t)C * P);

    size_t per_lrow, reserve;
    if (f16) {
        size_t c32, c16;
        down_caps(c32, c16);
        per_lrow = (size_t)W * 8 * (8 * c32 + 2 * c16) / 64;
        reserve = (size_t)16 << 20;
    } else {
        // 3 full-res buffers of the stage-0 width per input row, plus the input itself
        per_lrow = (size_t)4 * 8 * W * (3 * (size_t)e_conv1.rows + 3);
        reserve = (size_t)112 << 20;
    }
    size_t avail = G.arena.free_bytes() > reserve ? G.arena.free_bytes() - reserve : 0;
    int fit = (int)(avail / per_lrow);
    // an unbanded pass needs Hl rows (the FP32 path keeps its original, stricter test)
    int rows = fit >= (f16 ? Hl : Hl + 2 * EHALO) ? Hl : std::max(1, fit - 2 * EHALO);
    if (fit < 2 * EHALO + 1 && fit < Hl) throw std::runtime_error("not enough VRAM to encode even one band");
    if (band_rows > 0) rows = std::min(rows, band_rows);
    for (int r0 = 0; r0 < Hl; r0 += rows) {
        if (f16) downstack16(rgb, H, W, x, r0, std::min(rows, Hl - r0));
        else downstack(rgb, H, W, x, r0, std::min(rows, Hl - r0));
    }

    int zc = (int)e_head_c.rows;  // 32 = mean + log-variance
    if (f16) {
        __half* h = (__half*)b;
        resblock16(x, a, h, e_mid0, Hl, Wl, "enc.mid0");
        note("enc.attn.in", x, (size_t)C * P);
        attention_block16(x, a, h, e_attn, C, Hl, Wl);
        resblock16(x, a, h, e_mid1, Hl, Wl, "enc.mid1");
        note("enc.head.in", x, (size_t)C * P);
        vc::rms16(h, x, e_head_n, C, P, true);
        vc::conv(a, h, C, Hl, Wl, e_head_c, &e_head_cb, nullptr, vc::M3, 1.f);
        conv2d(b, a, zc, Hl, Wl, q_conv, &q_convb, 1);
        note("enc.z", b, (size_t)zc * P);
        CK(cudaMemcpyAsync(latent, b, (size_t)16 * P * 4, cudaMemcpyDeviceToDevice, G.stream));  // the mean
    } else {
        resblock(x, a, b, e_mid0, Hl, Wl, "enc.mid0");
        note("enc.attn.in", x, (size_t)C * P);
        attention_block(x, a, b, e_attn, C, Hl, Wl);
        resblock(x, a, b, e_mid1, Hl, Wl, "enc.mid1");
        note("enc.head.in", x, (size_t)C * P);
        rms_channels(a, x, e_head_n, C, P, true);
        conv2d(b, a, C, Hl, Wl, e_head_c, &e_head_cb, 3);
        conv2d(a, b, zc, Hl, Wl, q_conv, &q_convb, 1);
        note("enc.z", a, (size_t)zc * P);
        CK(cudaMemcpyAsync(latent, a, (size_t)16 * P * 4, cudaMemcpyDeviceToDevice, G.stream));  // the mean
    }
    G.arena.release(m);
}

void Vae::free() {
    auto res = [](Res& r) { for (Weight* w : {&r.n1, &r.c1, &r.c1b, &r.n2, &r.c2, &r.c2b, &r.sc, &r.scb}) free_weight(*w); };
    auto free_attn = [](Attn& a) { for (Weight* w : {&a.n, &a.qkv, &a.qkvb, &a.proj, &a.projb}) free_weight(*w); };
    for (Weight* w : {&conv2, &conv2b, &conv1, &conv1b, &head_n, &head_c, &head_cb, &e_conv1, &e_conv1b, &e_head_n, &e_head_c, &e_head_cb, &q_conv, &q_convb})
        free_weight(*w);
    res(mid0); res(mid1); res(e_mid0); res(e_mid1);
    free_attn(attn); free_attn(e_attn);
    for (auto& u : ups) { for (auto& r : u.res) res(r); free_weight(u.rs); free_weight(u.rsb); }
    for (auto& d : downs) { for (auto& r : d.res) res(r); free_weight(d.ds); free_weight(d.dsb); }
    ups.clear();
    downs.clear();
    st.reset();
}
