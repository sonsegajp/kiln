// kiln-engine: Anima inference server over stdin/stdout JSON lines.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "engine.h"
#include "graph.h"
#include "train.h"
#include "train_job.h"
#include "tag_job.h"

#include <fcntl.h>
#include <io.h>

// ---------------------------------------------------------------------------
// .npy (golden tensors from the reference harness)
// ---------------------------------------------------------------------------
static bool load_npy(const std::string& path, std::vector<float>& data, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    f.read(magic, 6);
    uint8_t ver[2];
    f.read((char*)ver, 2);
    uint32_t hl = 0;
    if (ver[0] == 1) { uint16_t h16; f.read((char*)&h16, 2); hl = h16; } else f.read((char*)&hl, 4);
    std::string h(hl, 0);
    f.read(&h[0], hl);
    std::string descr = h.substr(h.find("'descr'") + 10, 3);
    size_t a = h.find('(', h.find("'shape'")), b = h.find(')', a);
    std::string sh = h.substr(a + 1, b - a - 1);
    shape.clear();
    std::stringstream ss(sh);
    std::string tok;
    int64_t n = 1;
    while (std::getline(ss, tok, ',')) if (tok.find_first_not_of(' ') != std::string::npos) { shape.push_back(std::stoll(tok)); n *= shape.back(); }
    data.resize(n);
    if (descr == "<f4") f.read((char*)data.data(), n * 4);
    else if (descr == "<f8") { std::vector<double> d(n); f.read((char*)d.data(), n * 8); for (int64_t i = 0; i < n; i++) data[i] = (float)d[i]; }
    else if (descr == "<i4") { std::vector<int32_t> d(n); f.read((char*)d.data(), n * 4); for (int64_t i = 0; i < n; i++) data[i] = (float)d[i]; }
    else if (descr == "<i8") { std::vector<int64_t> d(n); f.read((char*)d.data(), n * 8); for (int64_t i = 0; i < n; i++) data[i] = (float)d[i]; }
    else throw std::runtime_error("npy dtype " + descr);
    return true;
}

static void compare(const char* what, const float* ours, const std::vector<float>& ref) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = (double)ours[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        mx = std::max(mx, std::fabs(d));
    }
    char b[256];
    snprintf(b, sizeof b, "%-16s rel_l2 %.3e  max_abs %.3e  (n=%zu)", what, sqrt(num / std::max(den, 1e-30)), mx, ref.size());
    log_msg(b);
}

static std::vector<float> dl(const float* d, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}

// Warm per-step timing at a given size, with a per-category breakdown of one step.
static int bench(Engine& E, int W, int H, const std::string& lora) {
    Cond c;
    c.qwen = {16, 28552, 11, 13529, 11, 1293, 6869, 11, 15289, 11, 220, 13629, 22362, 11, 1850, 4271, 11, 33532, 11, 40880, 49723, 6940, 11, 3330, 518, 25708};
    c.t5 = {209, 18722, 6, 6729, 6, 307, 1268, 6, 3993, 6, 20975, 3, 6, 200, 463, 6, 10962, 6, 15665, 19919, 7, 6, 479, 44, 17831, 1};
    c.t5w.assign(c.t5.size(), 1.f);
    if (!lora.empty()) log_msg("LoRA " + E.loras.apply(E.dit, {{lora, 1.f}}));
    int Hl = H / 8, Wl = W / 8, P = Hl * Wl;
    size_t m = G.arena.mark();
    Context ctx = E.condition(c);
    E.dit.cache_context(ctx);
    float* x = G.arena.f((size_t)16 * P);
    float* v = G.arena.f((size_t)16 * P);
    auto nz = torch_randn(1, (size_t)16 * P);
    CK(cudaMemcpy(x, nz.data(), nz.size() * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 2; i++) E.dit.forward(v, x, Hl, Wl, 0.9f, ctx);  // warm clocks + cuBLAS heuristics
    gpu_sync();
    auto t0 = Clock::now();
    const int n = 4;
    for (int i = 0; i < n; i++) E.dit.forward(v, x, Hl, Wl, 0.9f, ctx);
    gpu_sync();
    char b[160];
    snprintf(b, sizeof b, "%dx%d: %.0f ms/step (avg of %d, cross-KV cached)", W, H, ms_since(t0) / n, n);
    log_msg(b);
    if (G.tc) {  // the fp16-activation path against the fp32-input tensor-core path, same weights and input
        const bool keep = G.tc16;
        double ms[2];
        std::vector<float> out[2];
        for (int k = 0; k < 2; k++) {
            G.tc16 = k == 1;
            E.dit.forward(v, x, Hl, Wl, 0.9f, ctx);
            gpu_sync();
            auto tk = Clock::now();
            for (int i = 0; i < n; i++) E.dit.forward(v, x, Hl, Wl, 0.9f, ctx);
            gpu_sync();
            ms[k] = ms_since(tk) / n;
            out[k] = dl(v, (size_t)16 * P);
        }
        G.tc16 = keep;
        double d2 = 0, r2 = 0;
        for (size_t i = 0; i < out[0].size(); i++) { double d = out[1][i] - out[0][i]; d2 += d * d; r2 += (double)out[0][i] * out[0][i]; }
        snprintf(b, sizeof b, "fp32-input tensor-core path %.0f ms/step, fp16-activation path %.0f ms/step (rel_l2 %.2e); using %s", ms[0], ms[1],
                 std::sqrt(d2 / std::max(r2, 1e-30)), keep ? "fp16" : "fp32-input");
        log_msg(b);
    }
    {   // CFG: two passes vs one batched pass (same context twice is fine for timing)
        float* v2 = G.arena.f((size_t)16 * P);
        E.dit.cache_context(ctx);
        auto ta = Clock::now();
        for (int i = 0; i < 2; i++) { E.dit.forward(v, x, Hl, Wl, 0.9f, ctx); E.dit.forward(v2, x, Hl, Wl, 0.9f, ctx); }
        gpu_sync();
        double two = ms_since(ta) / 2;
        const Context* cs[2] = {&ctx, &ctx};
        float* outs[2] = {v, v2};
        E.dit.forward_batch(outs, x, Hl, Wl, 0.9f, cs, 2);
        gpu_sync();
        ta = Clock::now();
        for (int i = 0; i < 2; i++) E.dit.forward_batch(outs, x, Hl, Wl, 0.9f, cs, 2);
        gpu_sync();
        double one = ms_since(ta) / 2;
        std::vector<float> a = dl(v, (size_t)16 * P), b2 = dl(v2, (size_t)16 * P);
        double dmax = 0;
        for (size_t i = 0; i < a.size(); i++) dmax = std::max(dmax, (double)std::fabs(a[i] - b2[i]));
        snprintf(b, sizeof b, "CFG step: two passes %.0f ms, batched %.0f ms (batch halves differ by %.2e)", two, one, dmax);
        log_msg(b);
    }
    prof_enable(true);
    E.dit.forward(v, x, Hl, Wl, 0.9f, ctx);
    std::string rep = prof_report();
    prof_enable(false);
    std::stringstream ss(rep);
    std::string line;
    while (std::getline(ss, line)) log_msg(line);
    E.dit.drop_context_cache();
    G.arena.release(m);
    return 0;
}

// SDXL: where the weights landed, warm UNet step time and its per-category profile, VAE decode profile.
static int bench_sdxl(Engine& E, const std::string& ckpt, int W, int H) {
    auto tl = Clock::now();
    E.use_family("sdxl", ckpt);
    E.prepare_job();
    size_t vram = 0, ram = 0;
    int nram = 0;
    E.sdxl.unet.placement(vram, ram, nram);
    char b[256];
    snprintf(b, sizeof b, "load %.1f s; UNet weights: %zu MB VRAM, %zu MB system RAM (%d tensors); arena %zu MB; VRAM free %zu MB", ms_since(tl) / 1000,
             vram >> 20, ram >> 20, nram, G.arena.cap >> 20, gpu_free_bytes() >> 20);
    log_msg(b);
    SdTokens t;
    t.l_ids = {49406, 49407};
    t.l_ids.resize(77, 49407);
    t.g_ids = {49406, 49407};
    t.g_ids.resize(77, 0);
    t.l_w.assign(77, 1.f);
    t.g_w.assign(77, 1.f);
    auto te = Clock::now();
    SdCond c = E.sdxl.condition(t, W, H);
    gpu_sync();
    snprintf(b, sizeof b, "text encode (CLIP-L + CLIP-G, one chunk): %.0f ms", ms_since(te));
    log_msg(b);
    const int Hl = H / 8, Wl = W / 8, P = Hl * Wl;
    size_t m = G.arena.mark();
    float* x = G.arena.f((size_t)4 * P);
    float* out = G.arena.f((size_t)4 * P);
    auto nz = torch_randn(1, (size_t)4 * P);
    CK(cudaMemcpy(x, nz.data(), nz.size() * 4, cudaMemcpyHostToDevice));
    E.sdxl.unet.cache_context(c.ctx);
    E.sdxl.unet.forward(out, x, Hl, Wl, 500.f, c.y, c.ctx);  // warm clocks + cuBLAS heuristics
    gpu_sync();
    auto t0 = Clock::now();
    const int n = 2;
    for (int i = 0; i < n; i++) E.sdxl.unet.forward(out, x, Hl, Wl, 500.f, c.y, c.ctx);
    gpu_sync();
    snprintf(b, sizeof b, "%dx%d: UNet %.0f ms per pass (avg of %d; a CFG step is two)", W, H, ms_since(t0) / n, n);
    log_msg(b);
    prof_enable(true);
    E.sdxl.unet.forward(out, x, Hl, Wl, 500.f, c.y, c.ctx);
    std::string rep = prof_report();
    prof_enable(false);
    std::stringstream ss(rep);
    std::string line;
    log_msg("UNet profile:");
    while (std::getline(ss, line)) log_msg(line);
    E.sdxl.unet.drop_context_cache();
    float* img = G.arena.f((size_t)3 * H * W);
    sd_latent_to_vae(out, (size_t)4 * P);
    prof_enable(true);
    auto td = Clock::now();
    E.sdxl.vae.decode(out, Hl, Wl, img);
    gpu_sync();
    double dms = ms_since(td);
    rep = prof_report();
    prof_enable(false);
    snprintf(b, sizeof b, "VAE decode %.0f ms; profile:", dms);
    log_msg(b);
    std::stringstream ss2(rep);
    while (std::getline(ss2, line)) log_msg(line);
    G.arena.release(m);
    Sdxl::free_cond(c);
    return 0;
}

// LoRA training against ref/anima_train_ref.py's goldens (<dir>): one block's forward and backward (output, input
// gradient, every LoRA gradient), then a training step through patchify, two blocks, the final layer and the loss.
static std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<float> v((size_t)f.tellg() / 4);
    f.seekg(0);
    f.read((char*)v.data(), v.size() * 4);
    return v;
}
static double compare(const std::string& what, const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) { log_msg(what + ": size mismatch"); return 1e9; }
    double d2 = 0, r2 = 0, mx = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double d = (double)got[i] - want[i];
        d2 += d * d;
        r2 += (double)want[i] * want[i];
        mx = std::max(mx, std::fabs(d));
    }
    const double rel = std::sqrt(d2 / std::max(r2, 1e-30));
    char b[240];
    snprintf(b, sizeof b, "%-34s rel_l2 %.3e  max|d| %.3e", what.c_str(), rel, mx);
    log_msg(b);
    return rel;
}
static float* upload(const std::vector<float>& h) {
    float* d = G.arena.f(h.size());
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}
static int train_selftest(Engine& E, const std::string& dir) {
    const std::string d = dir + "\\";
    const int Hl = 16, Wl = 12, Hp = Hl / 2, Wp = Wl / 2, T = Hp * Wp, Lc = 20;
    const float t = 0.6f;
    AnimaTrainer tr(E.dit);
    auto set_lora = [&](const std::string& set, int b) {
        for (int k = 0; k < AnimaTrainer::NT; k++) {
            auto& l = tr.lora[b][k];
            const std::string p = d + set + "_lora" + std::to_string(b) + "_" + AnimaTrainer::kShort[k];
            auto a = read_f32(p + "_A.f32"), bb = read_f32(p + "_B.f32");
            CK(cudaMemcpy(l.A, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(l.B, bb.data(), bb.size() * 4, cudaMemcpyHostToDevice));
        }
    };
    auto check_grads = [&](const std::string& set, int b, const std::string& tag) {
        double worst = 0;
        for (int k = 0; k < AnimaTrainer::NT; k++) {
            auto& l = tr.lora[b][k];
            const std::string p = d + set + "_lora" + std::to_string(b) + "_" + AnimaTrainer::kShort[k];
            const std::string nm = tag + " blk" + std::to_string(b) + " " + AnimaTrainer::kShort[k];
            worst = std::max(worst, compare(nm + " dA", dl(l.dA, (size_t)tr.rank * l.in), read_f32(p + "_dA.f32")));
            worst = std::max(worst, compare(nm + " dB", dl(l.dB, (size_t)l.out * tr.rank), read_f32(p + "_dB.f32")));
        }
        return worst;
    };
    const size_t m = G.arena.mark();
    // the context: 20 real rows of the 512 the adapter pads to
    Context c;
    c.p = G.arena.f((size_t)512 * 1024);
    fill(c.p, 0.f, (size_t)512 * 1024);
    auto hc = read_f32(d + "blk_ctx.f32");
    CK(cudaMemcpy(c.p, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    c.len = 512;
    c.real = Lc;
    for (int exact = 1; exact >= 0; exact--) {
        const bool keep = G.fp16;
        G.fp16 = !exact;
        const std::string tag = exact ? "[fp32]" : "[fast]";
        // one block
        tr.init(4, 4.f, 1);
        set_lora("blk", 0);
        tr.zero_grad();
        const size_t mb = G.arena.mark();
        float* X = upload(read_f32(d + "blk_x.f32"));
        float* dY = upload(read_f32(d + "blk_g.f32"));
        float* dX = G.arena.f((size_t)T * 2048);
        tr.test_block(0, X, dY, dX, Hp, Wp, t, c);
        compare(tag + " block output", dl(X, (size_t)T * 2048), read_f32(d + "blk_y.f32"));
        compare(tag + " block input gradient", dl(dX, (size_t)T * 2048), read_f32(d + "blk_dx.f32"));
        log_msg(tag + " worst LoRA gradient rel_l2 " + std::to_string(check_grads("blk", 0, tag)));
        G.arena.release(mb);
        // a training step through two blocks
        tr.init(4, 4.f, 1);
        set_lora("chain", 0);
        set_lora("chain", 1);
        tr.zero_grad();
        float* lat = upload(read_f32(d + "chain_lat.f32"));
        float* noise = upload(read_f32(d + "chain_noise.f32"));
        auto t0 = Clock::now();
        const float loss = tr.step(lat, noise, Hl, Wl, t, c, 1.f, 2);
        gpu_sync();
        std::ifstream mf(d + "chain_pred.json");
        std::stringstream ss;
        ss << mf.rdbuf();
        const Json meta = Json::parse(ss.str());
        char b[160];
        snprintf(b, sizeof b, "%s chain loss %.6f (reference %.6f), step %d ms", tag.c_str(), loss, meta["loss"].num(0), (int)ms_since(t0));
        log_msg(b);
        double w = check_grads("chain", 0, tag);
        w = std::max(w, check_grads("chain", 1, tag));
        log_msg(tag + " chain worst LoRA gradient rel_l2 " + std::to_string(w));
        G.arena.release(mb);
        G.fp16 = keep;
    }
    G.arena.release(m);
    return 0;
}

// --train-bench WxH: a few rank-32 training steps on random data at that image size; time per step, where it goes,
// and the scratch it needs
static int train_bench(Engine& E, int W, int H, int steps) {
    const int Hl = H / 8, Wl = W / 8;
    AnimaTrainer tr(E.dit);
    tr.init(32, 32.f, 1);
    log_msg("LoRA rank 32 on " + std::to_string(tr.lora.size() * AnimaTrainer::NT) + " linears: " + std::to_string(tr.n / 1000000.0).substr(0, 5) + "M parameters");
    const size_t m = G.arena.mark();
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    std::vector<float> h((size_t)16 * Hl * Wl);
    auto rnd = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    float* lat = upload(rnd(h.size(), 1.f));
    float* noise = upload(rnd(h.size(), 1.f));
    Context c;
    c.p = G.arena.f((size_t)512 * 1024);
    fill(c.p, 0.f, (size_t)512 * 1024);
    auto hc = rnd((size_t)77 * 1024, 0.5f);
    CK(cudaMemcpy(c.p, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    c.len = 512;
    c.real = 77;
    G.arena.peak = G.arena.used;
    for (int i = 0; i < steps; i++) {
        if (i == steps - 1) prof_enable(true);
        auto t0 = Clock::now();
        const float loss = tr.step(lat, noise, Hl, Wl, 0.3f + 0.1f * i, c);
        gpu_sync();
        char b[200];
        snprintf(b, sizeof b, "%dx%d (%d tokens): step %d %.2f s, loss %.4f, scratch peak %zu MB of %zu", W, H, (Hl / 2) * (Wl / 2), i,
                 ms_since(t0) / 1000.0, loss, G.arena.peak >> 20, G.arena.cap >> 20);
        log_msg(b);
    }
    std::string rep = prof_report();
    prof_enable(false);
    std::stringstream ss(rep);
    std::string line;
    while (std::getline(ss, line)) log_msg(line);
    G.arena.release(m);
    return 0;
}

static int selftest(Engine& E, const std::string& dir, const std::string& lora) {
    std::vector<float> v, ref;
    std::vector<int64_t> sh;
    auto get = [&](const char* n, std::vector<float>& out) { return load_npy(dir + "\\" + n, out, sh); };
    Cond c;
    get("qwen_ids.npy", v); for (float f : v) c.qwen.push_back((int)f);
    get("t5_ids.npy", v); for (float f : v) c.t5.push_back((int)f);
    get("t5_weights.npy", c.t5w);
    std::vector<float> noise_ref;
    get("noise.npy", noise_ref);
    int Hl = (int)sh[sh.size() - 2], Wl = (int)sh.back(), P = Hl * Wl;

    uint64_t seed = 42;
    {
        std::ifstream mf(dir + "/meta.json");
        std::stringstream buf;
        buf << mf.rdbuf();
        if (!buf.str().empty()) seed = (uint64_t)Json::parse(buf.str())["seed"].num(42);
    }
    auto nz = torch_randn(seed, noise_ref.size());
    compare("noise", nz.data(), noise_ref);

    int Lc = 0;
    E.loras.apply(E.dit, {});
    size_t m = G.arena.mark();
    float* hid = G.arena.f(c.qwen.size() * 1024);
    E.te.encode(c.qwen, hid);
    if (get("te_hidden.npy", ref)) compare("te_hidden", dl(hid, ref.size()).data(), ref);
    Context ctx;
    ctx.len = Lc = E.dit.context_len((int)c.t5.size());
    ctx.real = (int)c.t5.size();
    ctx.p = G.arena.f((size_t)Lc * 1024);
    E.dit.adapt(hid, (int)c.qwen.size(), c.t5, c.t5w, ctx.p);
    if (get("adapter_out_nolora.npy", ref)) compare("adapter(nolora)", dl(ctx.p, ref.size()).data(), ref);

    std::vector<float> sig_ref, x0;
    get("sigmas.npy", sig_ref);
    float* x = G.arena.f((size_t)16 * P);
    float* out = G.arena.f((size_t)16 * P);
    x0 = noise_ref;
    for (auto& f : x0) f *= sig_ref[0];
    CK(cudaMemcpy(x, x0.data(), x0.size() * 4, cudaMemcpyHostToDevice));
    auto t0 = Clock::now();
    E.dit.forward(out, x, Hl, Wl, sig_ref[0], ctx);
    gpu_sync();
    log_msg("dit forward (cold) " + std::to_string((int)ms_since(t0)) + " ms");
    if (get("dit_out_step0_nolora.npy", ref)) compare("dit_step0(nolora)", dl(out, ref.size()).data(), ref);

    if (!lora.empty()) {
        log_msg("LoRA " + E.loras.apply(E.dit, {{lora, 1.f}}));
        E.te.encode(c.qwen, hid);
        E.dit.adapt(hid, (int)c.qwen.size(), c.t5, c.t5w, ctx.p);
        if (get("adapter_out.npy", ref)) compare("adapter(lora)", dl(ctx.p, ref.size()).data(), ref);
        t0 = Clock::now();
        E.dit.forward(out, x, Hl, Wl, sig_ref[0], ctx);
        gpu_sync();
        log_msg("dit forward (warm) " + std::to_string((int)ms_since(t0)) + " ms");
        if (get("dit_out_step0.npy", ref)) compare("dit_step0(lora)", dl(out, ref.size()).data(), ref);
        // full sampling run: shows how GEMM precision accumulates over the steps
        std::vector<float> fin;
        if (get("latent_final.npy", fin)) {
            CK(cudaMemcpy(x, x0.data(), x0.size() * 4, cudaMemcpyHostToDevice));
            t0 = Clock::now();
            for (size_t i = 0; i + 1 < sig_ref.size(); i++) {
                E.dit.forward(out, x, Hl, Wl, sig_ref[i], ctx);
                axpy(x, out, sig_ref[i + 1] - sig_ref[i], (size_t)16 * P);
            }
            gpu_sync();
            log_msg("sampling " + std::to_string(sig_ref.size() - 1) + " steps " + std::to_string((int)ms_since(t0)) + " ms");
            compare("latent_final", dl(x, fin.size()).data(), fin);
        }
    }
    std::vector<float> lat;
    if (get("latent_final.npy", lat)) {
        CK(cudaMemcpy(x, lat.data(), lat.size() * 4, cudaMemcpyHostToDevice));
        latent_model_to_vae(x, P);
        float* img = G.arena.f((size_t)3 * 64 * P);
        t0 = Clock::now();
        E.vae.decode(x, Hl, Wl, img);
        std::vector<float> rgb = dl(img, (size_t)3 * 64 * P);
        log_msg("vae decode " + std::to_string((int)ms_since(t0)) + " ms");
        if (get("image.npy", ref)) {  // [H, W, 3] in [0,1]
            std::vector<float> ours(ref.size());
            size_t HW = (size_t)64 * P;
            for (size_t p = 0; p < HW; p++)
                for (int k = 0; k < 3; k++) ours[p * 3 + k] = std::min(1.f, std::max(0.f, (rgb[k * HW + p] + 1.f) / 2.f));
            compare("vae_image", ours.data(), ref);

            // encoder round trip: encode the reference image, decode again, PSNR against it
            std::vector<float> chw(ref.size());
            for (size_t p = 0; p < HW; p++)
                for (int k = 0; k < 3; k++) chw[k * HW + p] = ref[p * 3 + k] * 2.f - 1.f;
            float* src = G.arena.f(chw.size());
            float* z = G.arena.f((size_t)16 * P);
            CK(cudaMemcpy(src, chw.data(), chw.size() * 4, cudaMemcpyHostToDevice));
            t0 = Clock::now();
            E.vae.encode(src, 8 * Hl, 8 * Wl, z);
            gpu_sync();
            log_msg("vae encode " + std::to_string((int)ms_since(t0)) + " ms");
            E.vae.decode(z, Hl, Wl, img);
            std::vector<float> back = dl(img, chw.size());
            double se = 0;
            for (size_t i = 0; i < chw.size(); i++) {
                double d = std::min(1.f, std::max(0.f, (back[i] + 1.f) / 2.f)) - (chw[i] + 1.f) / 2.f;
                se += d * d;
            }
            char b[96];
            snprintf(b, sizeof b, "vae roundtrip    PSNR %.2f dB", 10 * log10(1.0 / std::max(se / chw.size(), 1e-12)));
            log_msg(b);
        }
    }
    G.arena.release(m);
    log_msg("arena peak " + std::to_string(G.arena.peak >> 20) + " MB");
    return 0;
}

// ---------------------------------------------------------------------------
// main: stdin reader thread + one GPU worker
// ---------------------------------------------------------------------------
static Cond parse_cond(const Json& j);
static Job parse_job(const Json& r);

// Runs the jobs in a file back to back and reports (a) whether any weight changed while a job
// ran and (b) how far each job's latent is from the first one's. Identical jobs must match.
static int repeat_test(Engine& E, const std::string& file) {
    std::ifstream f(file);
    std::string line;
    std::vector<Job> jobs;
    while (std::getline(f, line)) if (line.find('{') != std::string::npos) jobs.push_back(parse_job(Json::parse(line)));
    std::vector<float> first;
    auto sums = weight_checksums();
    log_msg("weights tracked: " + std::to_string(sums.size()));
    for (size_t k = 0; k < jobs.size(); k++) {
        size_t nw_before = G.weights.size();
        E.generate(jobs[k]);
        auto now = weight_checksums();
        int changed = 0;
        size_t common = std::min(sums.size(), now.size());
        for (size_t i = 0; i < common && i < nw_before; i++)
            if (sums[i] != now[i]) {
                if (changed < 8) {
                    char b[160];
                    snprintf(b, sizeof b, "  weight #%zu (%zu elems, ptr %p) CHANGED", i, G.weights[i].n, (const void*)G.weights[i].p);
                    log_msg(b);
                }
                changed++;
            }
        std::string msg = "job " + jobs[k].id + ": " + std::to_string(changed) + " pre-existing weights changed, " +
                          std::to_string((long)now.size() - (long)nw_before) + " weights added";
        if (k == 0) first = E.last_latent;
        else if (first.size() == E.last_latent.size()) {
            double num = 0, den = 0;
            for (size_t i = 0; i < first.size(); i++) { double d = E.last_latent[i] - first[i]; num += d * d; den += (double)first[i] * first[i]; }
            char b[96];
            snprintf(b, sizeof b, "; latent vs job 1: rel_l2 %.3e", sqrt(num / std::max(den, 1e-30)));
            msg += b;
        }
        log_msg(msg);
        sums = now;
    }
    return 0;
}

static Cond parse_cond(const Json& j) {
    Cond c;
    for (auto& v : j["qwen_ids"].a) c.qwen.push_back((int)v.i64());
    for (auto& v : j["t5_ids"].a) c.t5.push_back((int)v.i64());
    for (auto& v : j["t5_weights"].a) c.t5w.push_back((float)v.num());
    if (c.qwen.empty()) c.qwen.push_back(151643);
    if (c.t5.empty()) c.t5.push_back(1);
    c.t5w.resize(c.t5.size(), 1.f);
    return c;
}

// {"l": {"ids": [[77 ints], ...], "weights": [[77 floats], ...]}, "g": {...}} (clip_tokenize.js encodeSDXL)
static SdTokens parse_sd_tokens(const Json& t) {
    SdTokens s;
    auto flat = [](const Json& chunks, std::vector<int>* ids, std::vector<float>* w) {
        for (auto& c : chunks.a)
            for (auto& v : c.a) {
                if (ids) ids->push_back((int)v.i64());
                else w->push_back((float)v.num(1.0));
            }
    };
    flat(t["l"]["ids"], &s.l_ids, nullptr);
    flat(t["l"]["weights"], nullptr, &s.l_w);
    flat(t["g"]["ids"], &s.g_ids, nullptr);
    flat(t["g"]["weights"], nullptr, &s.g_w);
    s.l_w.resize(s.l_ids.size(), 1.f);
    s.g_w.resize(s.g_ids.size(), 1.f);
    return s;
}

static Job parse_job(const Json& r) {
    Job j;
    j.family = r["family"].str("anima");
    j.checkpoint = r["checkpoint"].str();
    j.dit = r["dit"].str();
    if (j.family == "sdxl") {
        j.sd_pos = parse_sd_tokens(r["sd"]["pos"]);
        j.sd_neg = parse_sd_tokens(r["sd"]["neg"]);
    }
    j.scheduler = r["scheduler"].str(j.family == "sdxl" ? "karras" : "simple");
    j.id = r["id"].str();
    j.pos = parse_cond(r);
    j.neg = parse_cond(r.has("neg") ? r["neg"] : Json());
    const int align = j.family == "sdxl" ? 32 : 16;  // SDXL: the UNet halves the latent twice
    j.width = std::max(align, (int)r["width"].i64(512) / align * align);
    j.height = std::max(align, (int)r["height"].i64(768) / align * align);
    j.steps = std::max(1, (int)r["steps"].i64(8));
    j.cfg = (float)r["cfg"].num(1.0);
    j.cfg_until = std::min(1.f, std::max(0.f, (float)r["cfg_cutoff"].num(1.0)));
    j.sampler = r["sampler"].str("euler");
    j.cache = std::max(0.f, (float)r["step_cache"].num(0.0));
    if (r.has("nag")) {
        const Json& n = r["nag"];
        j.nag.on = n["enabled"].type == Json::Null ? true : n["enabled"].b;
        j.nag.scale = std::max(1.f, (float)n["scale"].num(5.0));
        j.nag.tau = std::max(1.f, (float)n["tau"].num(2.5));
        j.nag.alpha = std::min(1.f, std::max(0.f, (float)n["alpha"].num(0.25)));
    }
    j.shift = (float)r["shift"].num(3.0);
    j.seed = (uint64_t)r["seed"].num(0);
    j.preview = r["preview"].type == Json::Null ? true : r["preview"].b;
    j.out = r["out"].str();
    j.fp32 = r["precision"].str() == "fp32";
    j.skip_decode = r["skip_decode"].b;
    if (r.has("hires")) {
        const Json& h = r["hires"];
        j.hires.scale = (float)h["scale"].num(0);
        j.hires.denoise = (float)h["denoise"].num(0.4);
        j.hires.steps = std::max(1, (int)h["steps"].i64(4));
        j.hires.model = h["upscaler"].str("lanczos") == "model";  // the redraw hides resize differences; lanczos is far cheaper
    }
    if (r.has("face")) {
        const Json& f = r["face"];
        j.face.on = f["enabled"].type == Json::Null ? true : f["enabled"].b;
        j.face.denoise = (float)f["denoise"].num(0.4);
        j.face.steps = std::max(1, (int)f["steps"].i64(4));
        j.face.guide = (int)f["guide"].i64(384);
        j.face.max_size = (int)f["max_size"].i64(576);
        j.face.crop = (float)f["crop"].num(2.0);
        j.face.conf = (float)f["conf"].num(0.35);
        j.face.max_faces = (int)f["max_faces"].i64(4);
    }
    j.upscale = (int)r["upscale"]["factor"].i64(0);
    for (auto& l : r["loras"].a) j.loras.push_back({l["file"].str(), (float)l["strength"].num(1.0)});
    return j;
}


int main(int argc, char** argv) {
    _setmode(_fileno(stdout), _O_BINARY);
    std::string models = "models", golden, lora, initial_dit, train_golden, train_bench_size, train_file, tag_file;
    int device = 0;
    size_t reserve_mb = 1000, vram_budget_mb = 0;
    int bench_w = 0, bench_h = 0;
    std::string repeat_file, bench_sdxl_ckpt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--models") models = next();
        else if (a == "--dit") initial_dit = next();
        else if (a == "--device") device = std::stoi(next());
        else if (a == "--reserve-mb") reserve_mb = std::stoul(next());
        else if (a == "--vram-budget") vram_budget_mb = std::stoul(next());
        else if (a == "--selftest") golden = next();
        else if (a == "--train-selftest") train_golden = next();
        else if (a == "--train-bench") train_bench_size = next();
        else if (a == "--train") train_file = next();
        else if (a == "--tag") tag_file = next();
        else if (a == "--lora") lora = next();
        else if (a == "--repeat") repeat_file = next();
        else if (a == "--fp32") G.fp16 = false;
        else if (a == "--bench-sdxl") bench_sdxl_ckpt = next();
        else if (a == "--bench") { std::string wh = next(); bench_w = std::stoi(wh); bench_h = std::stoi(wh.substr(wh.find('x') + 1)); }
    }
    Engine E;
    try {
        gpu_init(device);
        {
            char b[320];
            snprintf(b, sizeof b, "GPU: %s (sm_%d, %.1f GB): %s (%s)", G.gpu_name.c_str(), G.sm, G.vram_total / 1073741824.0,
                     G.tc ? "tensor-core kernels ON" : "CUDA-core fp16x2 kernels", G.tc_reason.c_str());
            log_msg(b);
            if (G.tc) log_msg(std::string("fp16-activation path ") + (G.tc16 ? "ON" : "off") + " (" + G.tc16_reason + ")");
        }
        G.reserve_bytes = reserve_mb << 20;
        if (vram_budget_mb) {
            // share the card with another app (a game, a local LLM): Kiln uses at most this much VRAM; what doesn't
            // fit (the DiT, the text encoder) stays in system RAM and streams through a small stage
            size_t free0 = gpu_free_bytes(), budget = vram_budget_mb << 20;
            G.leave_free = free0 > budget ? free0 - budget : 0;
            G.reserve_bytes = G.leave_free + ARENA_TARGET + ((size_t)200 << 20);  // weights leave room for the arena
        }
        E.load(models, initial_dit);
    } catch (const std::exception& e) {
        emit(std::string("{\"ev\":\"fatal\",\"msg\":") + json_escape(e.what()) + "}");
        return 1;
    }
    if (!repeat_file.empty()) {
        try { return repeat_test(E, repeat_file); }
        catch (const std::exception& e) { log_msg(std::string("repeat failed: ") + e.what()); return 1; }
    }
    if (!bench_sdxl_ckpt.empty()) {
        try { return bench_sdxl(E, bench_sdxl_ckpt, bench_w ? bench_w : 1024, bench_h ? bench_h : 1024); }
        catch (const std::exception& e) { log_msg(std::string("bench failed: ") + e.what()); return 1; }
    }
    if (bench_w) {
        try { return bench(E, bench_w, bench_h, lora); }
        catch (const std::exception& e) { log_msg(std::string("bench failed: ") + e.what()); return 1; }
    }
    if (!train_file.empty()) {  // run a train job from a file (same JSON as the "train" command), events on stdout
        try {
            std::ifstream f(train_file);
            std::stringstream ss;
            ss << f.rdbuf();
            const Json job = Json::parse(ss.str());
            return trainjob::run(E, job, "cli");
        } catch (const std::exception& e) { log_msg(std::string("train failed: ") + e.what()); return 1; }
    }
    if (!tag_file.empty()) {  // run a tag job from a file (same JSON as the "tag" command), events on stdout
        try {
            std::ifstream f(tag_file);
            std::stringstream ss;
            ss << f.rdbuf();
            const Json job = Json::parse(ss.str());
            return tagjob::run(E, job, "cli");
        } catch (const std::exception& e) { log_msg(std::string("tag failed: ") + e.what()); return 1; }
    }
    if (!train_bench_size.empty()) {
        int bw = 512, bh = 768;
        sscanf(train_bench_size.c_str(), "%dx%d", &bw, &bh);
        try { return train_bench(E, bw, bh, 3); }
        catch (const std::exception& e) { log_msg(std::string("train bench failed: ") + e.what()); return 1; }
    }
    if (!train_golden.empty()) {
        try { return train_selftest(E, train_golden); }
        catch (const std::exception& e) { log_msg(std::string("train selftest failed: ") + e.what()); return 1; }
    }
    if (!golden.empty()) {
        try { return selftest(E, golden, lora); }
        catch (const std::exception& e) { log_msg(std::string("selftest failed: ") + e.what()); return 1; }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::string gpu_name = prop.name;
    emit("{\"ev\":\"ready\",\"gpu\":" + json_escape(gpu_name) + ",\"tc\":" + (G.tc ? "true" : "false") + ",\"tc16\":" + (G.tc16 ? "true" : "false") + "}");

    std::mutex mu;
    std::condition_variable cv;
    // one GPU, one worker: generate jobs and node graphs share the queue
    struct Work { std::string id; Job job; std::shared_ptr<Json> graph, train, tag; };
    std::deque<Work> queue;
    std::string running;
    bool quit = false;

    std::thread worker([&] {
        for (;;) {
            Work w;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !queue.empty(); });
                if (queue.empty()) return;
                w = std::move(queue.front());
                queue.pop_front();
                running = w.id;
                E.cancel = false;
            }
            if (w.graph) run_graph(E, *w.graph);
            else if (w.train) {
                try { trainjob::run(E, *w.train, w.id); }
                catch (const std::exception& e) {
                    if (E.cancel) emit("{\"id\":" + json_escape(w.id) + ",\"ev\":\"cancelled\"}");
                    else emit("{\"id\":" + json_escape(w.id) + ",\"ev\":\"error\",\"msg\":" + json_escape(e.what()) + "}");
                }
            } else if (w.tag) {
                try { tagjob::run(E, *w.tag, w.id); }
                catch (const std::exception& e) {
                    if (E.cancel) emit("{\"id\":" + json_escape(w.id) + ",\"ev\":\"cancelled\"}");
                    else emit("{\"id\":" + json_escape(w.id) + ",\"ev\":\"error\",\"msg\":" + json_escape(e.what()) + "}");
                }
            } else E.generate(w.job);
            std::lock_guard<std::mutex> lk(mu);
            running.clear();
        }
    });

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.size() >= 3 && (uint8_t)line[0] == 0xEF && (uint8_t)line[1] == 0xBB && (uint8_t)line[2] == 0xBF) line.erase(0, 3);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        Json r;
        try { r = Json::parse(line); }
        catch (const std::exception& e) { log_msg(std::string("bad request: ") + e.what()); continue; }
        std::string cmd = r["cmd"].str(), id = r["id"].str();
        if (cmd == "generate" || cmd == "graph" || cmd == "train" || cmd == "tag") {
            Work w;
            w.id = id;
            if (cmd == "graph") w.graph = std::make_shared<Json>(std::move(r));
            else if (cmd == "train") w.train = std::make_shared<Json>(std::move(r));
            else if (cmd == "tag") w.tag = std::make_shared<Json>(std::move(r));
            else w.job = parse_job(r);
            std::lock_guard<std::mutex> lk(mu);
            queue.push_back(std::move(w));
            cv.notify_one();
        } else if (cmd == "ext_result" || cmd == "ext_op") {
            E.ext_inbox.push(std::move(r));  // the running graph is waiting on a pack node
        } else if (cmd == "cancel") {
            std::lock_guard<std::mutex> lk(mu);
            if (running == id) {
                E.cancel = true;
                E.ext_inbox.cv.notify_all();
            }
            for (auto it = queue.begin(); it != queue.end(); ++it)
                if (it->id == id) { queue.erase(it); emit("{\"id\":" + json_escape(id) + ",\"ev\":\"cancelled\"}"); break; }
        } else if (cmd == "info") {
            emit("{\"id\":" + json_escape(id) + ",\"ev\":\"info\",\"gpu\":" + json_escape(gpu_name) + ",\"vram_free_mb\":" + std::to_string(gpu_free_bytes() >> 20) +
                 ",\"arena_mb\":" + std::to_string(G.arena.cap >> 20) + ",\"family\":" + json_escape(E.family) + ",\"tc\":" + (G.tc ? "true" : "false") + ",\"features\":{\"ext\":true,\"train\":true,\"tag\":true}}");
        } else {
            log_msg("unknown cmd " + cmd);
        }
    }
    {   // stdin closed: finish what is queued, then exit
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_one();
    worker.join();
    return 0;
}
