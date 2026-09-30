// The engine: models, LoRAs, prompt cache and the image pipeline shared by the stdin protocol
// (main.cpp) and the node-graph executor (graph.cpp).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "models.h"
#include "pipeline.h"
#include "sdpipe.h"
// optional modules, enabled by the build script when their sources exist
#ifdef KILN_FACE
#include "detect.h"
#endif
#ifdef KILN_UPSCALE
#include "upscale.h"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---------------------------------------------------------------------------
// output
// ---------------------------------------------------------------------------
inline void emit(const std::string& line) {
    static std::mutex out_mu;
    std::lock_guard<std::mutex> lk(out_mu);
    fwrite(line.data(), 1, line.size(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
}
inline void log_msg(const std::string& msg) { emit("{\"ev\":\"log\",\"msg\":" + json_escape(msg) + "}"); }

inline std::string b64(const uint8_t* p, size_t n) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = p[i] << 16 | (i + 1 < n ? p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        o += T[v >> 18]; o += T[(v >> 12) & 63];
        o += i + 1 < n ? T[(v >> 6) & 63] : '=';
        o += i + 2 < n ? T[v & 63] : '=';
    }
    return o;
}

// latent -> RGB approximation for live previews (ComfyUI latent_rgb_factors for Wan2.1)
inline const float PREVIEW_RGB[16][3] = {
    {-0.1299f, -0.1692f, 0.2932f}, {0.0671f, 0.0406f, 0.0442f}, {0.3568f, 0.2548f, 0.1747f}, {0.0372f, 0.2344f, 0.1420f},
    {0.0313f, 0.0189f, -0.0328f}, {0.0296f, -0.0956f, -0.0665f}, {-0.3477f, -0.4059f, -0.2925f}, {0.0166f, 0.1902f, 0.1975f},
    {-0.0412f, 0.0267f, -0.1364f}, {-0.1293f, 0.0740f, 0.1636f}, {0.0680f, 0.3019f, 0.1128f}, {0.0032f, 0.0581f, 0.0639f},
    {-0.1251f, 0.0927f, 0.1699f}, {0.0060f, -0.0633f, 0.0005f}, {0.3477f, 0.2275f, 0.2950f}, {0.1984f, 0.0913f, 0.1861f}};
inline const float PREVIEW_BIAS[3] = {-0.1835f, -0.0868f, -0.3360f};

// ---------------------------------------------------------------------------
// LoRA: kept unmerged (see LoraAdd in gpu.h); any number can be stacked and swapped instantly
// ---------------------------------------------------------------------------
struct LoraSpec { std::string file; float strength; };

class LoraManager {
public:
    std::string apply(Dit& dit, const std::vector<LoraSpec>& specs) {
        std::string key;
        for (auto& s : specs) key += s.file + "|" + std::to_string(s.strength) + ";";
        if (key == current_) return "";
        clear();
        if (by_dot_.empty())
            for (auto& [name, w] : dit.named) {
                if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) continue;
                std::string mod = name.substr(0, name.size() - 7);
                by_dot_[mod] = w;
                std::string u = mod;
                std::replace(u.begin(), u.end(), '.', '_');
                by_us_[u] = w;
            }
        std::string report;
        for (auto& s : specs)
            if (s.strength != 0.f) report += load(s) + " ";
        for (auto& [w, list] : lists_) w->lora = &list;
        current_ = key;
        return report;
    }

    const std::string& key() const { return current_; }
    // the DiT is being replaced: drop every pointer into it
    void reset() {
        clear();
        by_dot_.clear();
        by_us_.clear();
    }

    void clear() {
        for (auto& [w, list] : lists_) {
            w->lora = nullptr;
            for (auto& l : list) { free_weight(l.A); free_weight(l.B); }
        }
        lists_.clear();
        current_.clear();
    }

private:
    std::string current_;
    std::map<Weight*, std::vector<LoraAdd>> lists_;
    std::map<std::string, Weight*> by_dot_, by_us_;

    Weight* find(std::string mod) {
        for (const char* p : {"model.diffusion_model.", "diffusion_model.", "net.", "base_model.model.", "lora_unet_", "transformer."})
            if (mod.rfind(p, 0) == 0) { mod = mod.substr(strlen(p)); break; }
        auto it = by_dot_.find(mod);
        if (it != by_dot_.end()) return it->second;
        it = by_us_.find(mod);
        return it != by_us_.end() ? it->second : nullptr;
    }

    static bool all_zero(const StTensor& t) {
        for (int64_t i = 0; i < t.numel(); i++) if (st_elem_f32(t, i) != 0.f) return false;
        return true;
    }

    std::string load(const LoraSpec& spec) {
        SafeTensors st(spec.file);
        struct Pair { const StTensor *down = nullptr, *up = nullptr; float alpha = -1; };
        std::map<std::string, Pair> pairs;
        int ignored = 0;
        auto ends = [](const std::string& s, const std::string& t) { return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0; };
        for (auto& [name, t] : st.all()) {
            if (ends(name, ".lora_A.weight")) pairs[name.substr(0, name.size() - 14)].down = &t;
            else if (ends(name, ".lora_B.weight")) pairs[name.substr(0, name.size() - 14)].up = &t;
            else if (ends(name, ".lora_down.weight")) pairs[name.substr(0, name.size() - 17)].down = &t;
            else if (ends(name, ".lora_up.weight")) pairs[name.substr(0, name.size() - 15)].up = &t;
            else if (ends(name, ".alpha")) pairs[name.substr(0, name.size() - 6)].alpha = st_elem_f32(t, 0);
            else ignored++;
        }
        int applied = 0, missing = 0, zero = 0;
        for (auto& [mod, p] : pairs) {
            if (!p.down || !p.up) continue;
            Weight* w = find(mod);
            int64_t r = p.down->shape[0], in = p.down->numel() / r, out = p.up->shape[0];
            if (!w || w->rows != out || w->cols != in) { missing++; continue; }
            if (all_zero(*p.up)) { zero++; continue; }  // untrained (zero-init) pair: no effect
            LoraAdd l;
            l.A = upload_weight(*p.down, Place::Auto);
            l.B = upload_weight(*p.up, Place::Auto);
            l.B.cols = r;
            l.scale = spec.strength * (p.alpha > 0 ? p.alpha / (float)r : 1.f);
            lists_[w].push_back(l);
            applied++;
        }
        std::string name = spec.file.substr(spec.file.find_last_of("/\\") + 1);
        return name + ": " + std::to_string(applied) + " layers" + (zero ? ", " + std::to_string(zero) + " zero" : "") +
               (missing ? ", " + std::to_string(missing) + " unmatched" : "") +
               (ignored ? ", " + std::to_string(ignored) + " unsupported tensors ignored" : "");
    }
};

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------
struct Cond { std::vector<int> qwen, t5; std::vector<float> t5w; };

struct Job {
    std::string id;
    std::string family = "anima";  // anima | sdxl
    std::string checkpoint;        // sdxl: the single-file checkpoint
    std::string dit;               // anima: the DiT file to render with (empty = the default model)
    Cond pos, neg;                 // anima prompts
    SdTokens sd_pos, sd_neg;       // sdxl prompts
    std::string scheduler = "simple";
    int width = 512, height = 768, steps = 8;
    float cfg = 1.f, shift = 3.f;
    float cfg_until = 1.f;  // fraction of steps that use CFG (speed option for cfg > 1)
    std::string sampler = "euler";
    float cache = 0.f;      // first-block step cache threshold (0 = off)
    struct { bool on = false; float scale = 5.f, tau = 2.5f, alpha = 0.25f; } nag;  // negative prompt at CFG 1
    uint64_t seed = 0;
    std::vector<LoraSpec> loras;
    std::string out;
    bool preview = true;
    bool fp32 = false;  // exact path: every GEMM and attention in fp32 (slower)
    bool skip_decode = false;  // debug: stop after sampling
    struct { float scale = 0.f, denoise = 0.4f; int steps = 4; bool model = false; } hires;      // 2nd pass at higher res
    struct { bool on = false; float denoise = 0.4f, crop = 2.0f, conf = 0.35f; int steps = 4, guide = 384, max_size = 576, max_faces = 4; } face;
    int upscale = 0;  // final 2x / 4x with the upscale model
};

inline const size_t ARENA_TARGET = (size_t)800 << 20;
inline const size_t STAGE_ELEMS = (size_t)32 << 20;  // 64 MB: the largest SDXL conv (1280x2560x3x3) fits  // enough for hires/face/upscale at common sizes

// Replies for the running graph's pack-node calls (ext_result, ext_op), fed by the stdin reader.
struct Inbox {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Json> q;
    void push(Json j) {
        { std::lock_guard<std::mutex> lk(mu); q.push_back(std::move(j)); }
        cv.notify_all();
    }
};

struct Engine {
    TextEncoder te;
    Dit dit;
    Vae vae;
    LoraManager loras;
    std::atomic<bool> cancel{false};
    Inbox ext_inbox;
    std::string models;

    std::string dit_path, default_dit;

    // initial_dit: start on this Anima checkpoint instead of the stock base model (the server passes the
    // one its last job used, so a restart doesn't cost a model swap on the next render)
    void load(const std::string& dir, const std::string& initial_dit = "") {
        models = dir;
        default_dit = dit_path = dir + "\\anima-base-v1.0.safetensors";
        if (!initial_dit.empty() && GetFileAttributesA(initial_dit.c_str()) != INVALID_FILE_ATTRIBUTES) dit_path = initial_dit;
        auto t0 = Clock::now();
        auto path = [&](const char* f) { return dir + "\\" + f; };
        // small, conv-heavy models first: with a VRAM budget they keep the VRAM and the big DiT and text
        // encoder (plain matmuls, which stream well) go to system RAM
        emit("{\"ev\":\"loading\",\"what\":\"vae\"}");
        vae.load(path("qwen_image_vae.safetensors"), Place::Auto);
#ifdef KILN_FACE
        if (GetFileAttributesA((dir + "\\detect\\face_yolov8m.safetensors").c_str()) != INVALID_FILE_ATTRIBUTES) {
            emit("{\"ev\":\"loading\",\"what\":\"face\"}");
            face_model();
        }
#endif
#ifdef KILN_UPSCALE
        emit("{\"ev\":\"loading\",\"what\":\"upscale\"}");
        upscale_model();
#endif
        emit("{\"ev\":\"loading\",\"what\":\"dit\"}");
        dit.load(dit_path, Place::Auto);
        emit("{\"ev\":\"loading\",\"what\":\"te\"}");
        te.load(path("qwen_3_06b_base.safetensors"), Place::Auto);
        bool spilled = false;  // weights in system RAM stream through a VRAM stage (see stage_weight)
        for (auto& [n, w] : dit.named) spilled |= w->on_host;
        if (spilled) gpu_reserve_stage(STAGE_ELEMS);
        size_t free_b = gpu_free_bytes();
        size_t wbuf = WBUF_ELEMS;
        size_t margin = (size_t)160 << 20;
        size_t arena = free_b > wbuf * 4 + margin + G.leave_free ? free_b - wbuf * 4 - margin - G.leave_free : 0;
        if (G.leave_free) {
            arena = std::min(arena, ARENA_TARGET);
            G.reserve_bytes = G.leave_free + ((size_t)128 << 20);  // LoRAs may use what's left of the budget
        }
        gpu_alloc_scratch(wbuf, arena);
        int host_dit = 0;
        for (auto& [n, w] : dit.named) host_dit += w->on_host;
        log_msg("loaded in " + std::to_string((int)ms_since(t0)) + " ms; scratch arena " + std::to_string(arena >> 20) +
                " MB; DiT tensors in system RAM: " + std::to_string(host_dit) + "/" + std::to_string(dit.named.size()));
    }

    // prompt -> cross-attn context [Lc, 1024] allocated in the arena
    // Recent prompt encodings stay on the GPU (a few MB each): re-rolling seeds or tweaking
    // settings skips the text encoder + adapter entirely. Keyed by exact tokens, weights and LoRA set.
    struct CachedCtx { std::string key; Context ctx; };
    std::deque<CachedCtx> ctx_cache;
    static const size_t CTX_CACHE_MAX = 4;

    Context condition(const Cond& c) {
        std::string key = loras.key() + "#";
        auto put = [&](const void* p, size_t n) { key.append((const char*)p, n); };
        size_t nq = c.qwen.size(), nt = c.t5.size();
        put(&nq, sizeof nq); put(c.qwen.data(), nq * 4);
        put(&nt, sizeof nt); put(c.t5.data(), nt * 4); put(c.t5w.data(), nt * 4);
        for (auto it = ctx_cache.begin(); it != ctx_cache.end(); ++it)
            if (it->key == key) {
                CachedCtx hit = *it;
                ctx_cache.erase(it);
                ctx_cache.push_front(hit);
                return hit.ctx;
            }

        Context ctx;
        ctx.len = dit.context_len((int)nt);
        ctx.real = (int)nt;
        size_t bytes = (size_t)ctx.len * 1024 * 4;
        float* owned = nullptr;
        if (cudaMalloc(&owned, bytes) != cudaSuccess) { cudaGetLastError(); owned = nullptr; }
        ctx.p = owned ? owned : G.arena.f((size_t)ctx.len * 1024);  // no room to cache: job-lifetime scratch
        size_t m = G.arena.mark();
        float* hid = G.arena.f(nq * 1024);
        te.encode(c.qwen, hid);
        dit.adapt(hid, (int)nq, c.t5, c.t5w, ctx.p);
        G.arena.release(m);
        if (owned) {
            if (ctx_cache.size() >= CTX_CACHE_MAX) {
                gpu_sync();
                CK(cudaFree(ctx_cache.back().ctx.p));
                ctx_cache.pop_back();
            }
            ctx_cache.push_front({key, ctx});
        }
        return ctx;
    }

    std::vector<float> last_latent;  // model-space latent after sampling (debug)

#ifdef KILN_FACE
    std::unique_ptr<FaceDetector> detector;
    FaceDetector& face_model() {
        if (!detector) {
            detector = std::make_unique<FaceDetector>();
            detector->load(models + "\\detect\\face_yolov8m.safetensors", models + "\\detect\\face_yolov8m.json", Place::Auto);
        }
        return *detector;
    }
#endif
#ifdef KILN_UPSCALE
    std::unique_ptr<Upscaler> upscaler;
    std::string upscaler_path;
    Upscaler* upscale_model(const std::string& p) {
        if (upscaler && upscaler_path == p) return upscaler.get();
        upscaler.reset();
        upscaler = std::make_unique<Upscaler>();
        upscaler->load(p, Place::Auto);
        upscaler_path = p;
        return upscaler.get();
    }
    Upscaler* upscale_model() {
        if (!upscaler) {
            std::string p = models + "\\upscale\\4x-AnimeSharp.safetensors";
            if (GetFileAttributesA(p.c_str()) == INVALID_FILE_ATTRIBUTES) return nullptr;
            return upscale_model(p);
        }
        return upscaler.get();
    }
#endif

    std::string preview_json(const float* x, const float* v, float s, int Hl, int Wl, int C = 16) {
        static const float SD_RGB[4][3] = {{0.3651f, 0.4232f, 0.4341f}, {-0.2533f, -0.0042f, 0.1068f}, {0.1076f, 0.1111f, -0.0362f}, {-0.3165f, -0.2492f, -0.2188f}};
        static const float SD_BIAS[3] = {0.1084f, -0.0175f, -0.0011f};
        const float (*fac)[3] = C == 4 ? SD_RGB : PREVIEW_RGB;
        const float* bias = C == 4 ? SD_BIAS : PREVIEW_BIAS;
        int P = Hl * Wl;
        std::vector<float> hx((size_t)C * P), hv((size_t)C * P);
        CK(cudaMemcpy(hx.data(), x, hx.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv.data(), v, hv.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<uint8_t> prev((size_t)3 * P);
        for (int p = 0; p < P; p++) {
            float rgb[3] = {bias[0], bias[1], bias[2]};
            for (int c = 0; c < C; c++) {
                float x0 = hx[(size_t)c * P + p] - hv[(size_t)c * P + p] * s;
                for (int k = 0; k < 3; k++) rgb[k] += x0 * fac[c][k];
            }
            for (int k = 0; k < 3; k++) prev[(size_t)p * 3 + k] = (uint8_t)(std::min(1.f, std::max(0.f, (rgb[k] + 1.f) / 2.f)) * 255.f);
        }
        return ",\"preview\":{\"w\":" + std::to_string(Wl) + ",\"h\":" + std::to_string(Hl) + ",\"rgb\":\"" + b64(prev.data(), prev.size()) + "\"}";
    }

    StepFn stepper(const Job& j, const char* stage, int Hl, int Wl, double* total_ms, int C = 16) {
        return [this, &j, stage, Hl, Wl, total_ms, C](int step, int of, float s, const float* x, const float* v, double ms) {
            *total_ms += ms;
            std::string pv = j.preview ? preview_json(x, v, s, Hl, Wl, C) : "";
            emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"step\",\"stage\":\"" + stage + "\",\"step\":" + std::to_string(step) +
                 ",\"of\":" + std::to_string(of) + ",\"ms\":" + std::to_string((int)ms) + pv + "}");
        };
    }

    // [-1,1] image resize; with use_model the 4x upscale network does the enlarging first
    void resize_image(float* dst, const float* src, int H, int W, int H2, int W2, bool use_model) {
#ifdef KILN_UPSCALE
        Upscaler* up = use_model && (H2 > H || W2 > W) ? upscale_model() : nullptr;
        if (up) {
            int s = up->scale();
            size_t m = G.arena.mark();
            float* unit = G.arena.f((size_t)3 * H * W);
            float* big = G.arena.f((size_t)3 * H * W * s * s);
            rgb_to_unit(unit, src, (size_t)3 * H * W);
            up->run(big, unit, H, W);
            if (H * s == H2 && W * s == W2) rgb_from_unit(dst, big, (size_t)3 * H2 * W2);
            else {
                resize_lanczos(dst, big, 3, H * s, W * s, H2, W2);
                rgb_from_unit(dst, dst, (size_t)3 * H2 * W2);
            }
            G.arena.release(m);
            return;
        }
#endif
        resize_lanczos(dst, src, 3, H, W, H2, W2);
    }

    static int round16(float v) { return std::max(16, (int)std::lround(v / 16.f) * 16); }

#ifdef KILN_FACE
    // ADetailer / FaceDetailer: detect faces, redraw each one at a higher resolution with a masked
    // img2img pass, then blend it back with a feathered mask. Returns the number of faces redrawn.
    int face_detail(float* img, int H, int W, const Context& pos, const Context* neg, const Job& j, double* ms, const Context* nag_ctx) {
        std::vector<Box> boxes;
        {
            size_t m = G.arena.mark();
            float* unit = G.arena.f((size_t)3 * H * W);
            rgb_to_unit(unit, img, (size_t)3 * H * W);
            boxes = face_model().detect(unit, H, W, j.face.conf, 0.5f);
            G.arena.release(m);
        }
        int done = 0;
        for (auto& b : boxes) {
            if (done >= j.face.max_faces) break;
            float bw = b.x1 - b.x0, bh = b.y1 - b.y0;
            if (std::max(bw, bh) < 12.f) continue;
            float cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
            int x0 = std::max(0, (int)std::floor(cx - bw * j.face.crop / 2)), x1 = std::min(W, (int)std::ceil(cx + bw * j.face.crop / 2));
            int y0 = std::max(0, (int)std::floor(cy - bh * j.face.crop / 2)), y1 = std::min(H, (int)std::ceil(cy + bh * j.face.crop / 2));
            int w = x1 - x0, h = y1 - y0;
            if (w < 16 || h < 16) continue;
            // enlarge so the face's long side reaches `guide`, but keep the crop within max_size
            float sc = std::min(std::max((float)j.face.guide / std::max(bw, bh), 1.f), (float)j.face.max_size / std::max(w, h));
            int tw = round16(w * sc), th = round16(h * sc), tl = (th / 8) * (tw / 8);

            size_t m = G.arena.mark();
            float* crop = G.arena.f((size_t)3 * w * h);
            float* big = G.arena.f((size_t)3 * tw * th);
            float* z = G.arena.f((size_t)16 * tl);
            float* xz = G.arena.f((size_t)16 * tl);
            float* lmask = G.arena.f(tl);
            float* pmask = G.arena.f((size_t)w * h);
            crop_rect(crop, img, 3, H, W, x0, y0, w, h);
            resize_lanczos(big, crop, 3, h, w, th, tw);
            vae.encode(big, th, tw, z);
            latent_vae_to_model(z, tl);

            // mask: the face box grown 10% per side, feathered over 6% of the crop
            auto make_mask = [&](int mh, int mw, float scale_x, float scale_y) {
                std::vector<float> mk((size_t)mh * mw);
                float rx0 = (b.x0 - x0 - 0.1f * bw) * scale_x, rx1 = (b.x1 - x0 + 0.1f * bw) * scale_x;
                float ry0 = (b.y0 - y0 - 0.1f * bh) * scale_y, ry1 = (b.y1 - y0 + 0.1f * bh) * scale_y;
                float feather = std::max(1.f, 0.06f * std::max(mw, mh));
                for (int yy = 0; yy < mh; yy++)
                    for (int xx = 0; xx < mw; xx++) {
                        float px = xx + 0.5f, py = yy + 0.5f;
                        float d = std::min(std::min(px - rx0, rx1 - px), std::min(py - ry0, ry1 - py));  // >0 inside
                        mk[(size_t)yy * mw + xx] = std::min(1.f, std::max(0.f, d / feather + 0.5f));
                    }
                return mk;
            };
            auto lm = make_mask(th / 8, tw / 8, tw / 8.f / w, th / 8.f / h);
            auto pm = make_mask(h, w, 1.f, 1.f);
            CK(cudaMemcpy(lmask, lm.data(), lm.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(pmask, pm.data(), pm.size() * 4, cudaMemcpyHostToDevice));

            SampleParams sp{j.face.steps, j.cfg, j.shift, j.face.denoise, j.seed, j.cfg_until, j.sampler, j.cache};
            if (nag_ctx) { sp.nag_neg = nag_ctx; sp.nag_scale = j.nag.scale; sp.nag_tau = j.nag.tau; sp.nag_alpha = j.nag.alpha; }
            sample(dit, xz, th / 8, tw / 8, pos, neg, sp, z, lmask, stepper(j, "face", th / 8, tw / 8, ms), &cancel);
            latent_model_to_vae(xz, tl);
            vae.decode(xz, th / 8, tw / 8, big);
            resize_lanczos(crop, big, 3, th, tw, h, w);
            blend_rect(img, 3, H, W, crop, pmask, x0, y0, w, h);
            G.arena.release(m);
            done++;
        }
        return done;
    }
#endif

    void reload_dit(const std::string& path) {
        loras.reset();
        for (auto& c : ctx_cache) CK(cudaFree(c.ctx.p));  // adapter outputs depend on the DiT
        ctx_cache.clear();
        dit.drop_context_cache();
        for (auto& [n, w] : dit.named) free_weight(*w);
        dit = Dit();
        size_t keep = G.reserve_bytes;
        G.reserve_bytes = ((size_t)256 << 20) + G.leave_free;  // the arena already exists: only a small margin is needed
        dit_path.clear();  // if the load fails, the next job loads again
        dit.load(path, Place::Auto);
        G.reserve_bytes = keep;
        dit_path = path;
    }

    // Start of every job. If another app held VRAM when the engine started, the scratch arena came up
    // small and some weights landed in system RAM. Reclaim both as soon as the VRAM is free again: the
    // arena first (without it jobs fail), then the weights, largest first (they cost the most per step).
    void prepare_job() {
        if (G.arena.used == 0 && G.arena.cap < ARENA_TARGET) {
            size_t margin = (size_t)160 << 20, fr = gpu_free_bytes();
            size_t avail = G.arena.cap + (fr > G.leave_free ? fr - G.leave_free : 0);
            size_t target = std::min(ARENA_TARGET, avail > margin ? avail - margin : 0);
            if (target > G.arena.cap + ((size_t)32 << 20)) {
                size_t was = G.arena.cap;
                gpu_resize_arena(target);
                log_msg("scratch arena " + std::to_string(was >> 20) + " -> " + std::to_string(G.arena.cap >> 20) + " MB");
            }
        }
        std::vector<Weight*> host;
        for (auto& [n, w] : dit.named) if (w->on_host) host.push_back(w);
        if (!host.empty()) {
            std::sort(host.begin(), host.end(), [](Weight* a, Weight* b) { return a->numel() > b->numel(); });
            int moved = 0;
            for (Weight* w : host) moved += promote_weight(*w, ((size_t)256 << 20) + G.leave_free);
            if (moved) {
                dit.half_weights();  // tensor-core GPUs keep the block weights in VRAM as fp16
                log_msg("moved " + std::to_string(moved) + " of " + std::to_string(host.size()) + " DiT tensors from system RAM back to VRAM");
            }
        }
    }

    // ---- model families: 6 GB holds Anima or SDXL, not both, so a job for the other family swaps them
    Sdxl sdxl;
    std::string family = "anima";
    static constexpr size_t WBUF_ELEMS = (size_t)8192 * 2048;  // fp32 staging for one weight matrix

    void unload_anima() {
        loras.reset();
        for (auto& c : ctx_cache) CK(cudaFree(c.ctx.p));
        ctx_cache.clear();
        dit.drop_context_cache();
        for (auto& [n, w] : dit.named) free_weight(*w);
        dit = Dit();
        te.free();
        te = TextEncoder();
        vae.free();
        vae = Vae();
    }
    void use_family(const std::string& fam, const std::string& ckpt = "") {
        if (fam == family && (fam != "sdxl" || sdxl.path == ckpt)) return;
        auto t0 = Clock::now();
        if (family == "anima") unload_anima();
        else if (family == "sdxl") sdxl.free();
        gpu_sync();
        size_t keep = G.reserve_bytes;
        G.reserve_bytes = ((size_t)200 << 20) + G.leave_free;  // the arena already exists: leave only a small margin
        try {
            if (fam == "sdxl") {
                if (!G.stage) gpu_reserve_stage(STAGE_ELEMS);  // SDXL's UNet rarely fits entirely next to the arena
                sdxl.load(ckpt);
            } else {
                dit.load(dit_path, Place::Auto);
                vae.load(models + "\\qwen_image_vae.safetensors", Place::Auto);
                te.load(models + "\\qwen_3_06b_base.safetensors", Place::Auto);
                bool spilled = false;  // weights in system RAM stream through the stage
                for (auto& [n, w] : dit.named) spilled |= w->on_host;
                if (spilled && !G.stage) gpu_reserve_stage(STAGE_ELEMS);
            }
        } catch (...) {
            G.reserve_bytes = keep;
            family = "none";  // half loaded: the next job reloads from scratch
            throw;
        }
        G.reserve_bytes = keep;
        family = fam;
        log_msg((fam == "sdxl" ? "SDXL " + ckpt : std::string("Anima")) + " loaded in " + std::to_string((int)ms_since(t0)) + " ms");
    }


    // [-1,1] rgb [3, H, W] on the GPU -> RGB8 file, then the job's done event
    void finish_image(const Job& j, const float* img, int H, int W, Clock::time_point t0, std::ostringstream& tm) {
        std::vector<float> rgb((size_t)3 * H * W);
        CK(cudaMemcpy(rgb.data(), img, rgb.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<uint8_t> out8((size_t)W * H * 3);
        for (int c = 0; c < 3; c++)
            for (size_t p = 0; p < (size_t)W * H; p++) {
                float f = std::min(1.f, std::max(0.f, (rgb[(size_t)c * W * H + p] + 1.f) / 2.f));
                out8[p * 3 + c] = (uint8_t)(f * 255.f);  // truncation, as ComfyUI's SaveImage does
            }
        std::ofstream f(j.out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + j.out);
        f.write((const char*)out8.data(), out8.size());
        f.close();
        tm << ",\"arena_peak_mb\":" << (G.arena.peak >> 20);
        emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"done\",\"out\":" + json_escape(j.out) + ",\"w\":" + std::to_string(W) + ",\"h\":" + std::to_string(H) +
             ",\"total_ms\":" + std::to_string((int)ms_since(t0)) + ",\"timings\":{" + tm.str() + "}}");
    }

    void generate_sd(const Job& j) {
        auto t0 = Clock::now();
        size_t m0 = G.arena.mark();
        std::ostringstream tm;
        SdCond pos, neg;
        try {
            if (j.checkpoint.empty()) throw std::runtime_error("no SDXL checkpoint given");
            auto tl = Clock::now();
            use_family("sdxl", j.checkpoint);
            tm << "\"load_ms\":" << (int)ms_since(tl);
            prepare_job();
            if (!j.loras.empty()) log_msg("SDXL LoRAs are not supported yet: rendering without them");
            int H = j.height, W = j.width, Hl = H / 8, Wl = W / 8, P = Hl * Wl;
            auto te0 = Clock::now();
            pos = sdxl.condition(j.sd_pos, W, H);
            const bool use_neg = j.cfg > 1.f;
            if (use_neg) neg = sdxl.condition(j.sd_neg, W, H);
            gpu_sync();
            double enc_ms = ms_since(te0);
            emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"encoded\",\"ms\":" + std::to_string((int)enc_ms) + "}");
            tm << ",\"encode_ms\":" << (int)enc_ms;

            SampleParams sp;
            sp.steps = j.steps;
            sp.cfg = j.cfg;
            sp.seed = j.seed;
            sp.cfg_until = j.cfg_until;
            sp.sampler = j.sampler;
            sp.scheduler = j.scheduler;
            float* x = G.arena.f((size_t)4 * P);
            double sample_ms = 0;
            sample_sd(sdxl.unet, x, Hl, Wl, pos, use_neg ? &neg : nullptr, sp, nullptr, nullptr, stepper(j, "base", Hl, Wl, &sample_ms, 4), &cancel);
            tm << ",\"sample_ms\":" << (int)sample_ms;

            auto td = Clock::now();
            sd_latent_to_vae(x, (size_t)4 * P);
            float* img = G.arena.f((size_t)3 * H * W);
            sdxl.vae.decode(x, Hl, Wl, img);
            sd_debug("vae out", img, (size_t)3 * H * W);
            gpu_sync();
            double dec_ms = ms_since(td);
            emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"decoded\",\"ms\":" + std::to_string((int)dec_ms) + "}");
            tm << ",\"decode_ms\":" << (int)dec_ms;

            if (j.hires.scale > 1.01f) {  // hires fix through SDXL's own VAE and sampler
                auto th = Clock::now();
                int H2 = (int)std::lround(H * j.hires.scale) / 32 * 32, W2 = (int)std::lround(W * j.hires.scale) / 32 * 32, Hl2 = H2 / 8, Wl2 = W2 / 8;
                size_t n2 = (size_t)4 * Hl2 * Wl2;
                float* img2 = G.arena.f((size_t)3 * H2 * W2);
                resize_image(img2, img, H, W, H2, W2, j.hires.model);
                float* z = G.arena.f(n2);
                float* x2 = G.arena.f(n2);
                sdxl.vae.encode(img2, H2, W2, z);
                sd_latent_from_vae(z, n2);
                SampleParams sp2 = sp;
                sp2.steps = j.hires.steps;
                sp2.denoise = j.hires.denoise;
                double hs_ms = 0;
                sample_sd(sdxl.unet, x2, Hl2, Wl2, pos, use_neg ? &neg : nullptr, sp2, z, nullptr, stepper(j, "hires", Hl2, Wl2, &hs_ms, 4), &cancel);
                sd_latent_to_vae(x2, n2);
                sdxl.vae.decode(x2, Hl2, Wl2, img2);
                img = img2; H = H2; W = W2;
                gpu_sync();
                tm << ",\"hires_ms\":" << (int)ms_since(th);
            }
            if (j.face.on) log_msg("the face detailer does not run on SDXL yet");
            if (j.upscale > 1) {
                auto tu = Clock::now();
                int H2 = H * j.upscale, W2 = W * j.upscale;
                float* out = G.arena.f((size_t)3 * H2 * W2);
                resize_image(out, img, H, W, H2, W2, true);
                img = out; H = H2; W = W2;
                gpu_sync();
                tm << ",\"upscale_ms\":" << (int)ms_since(tu);
            }
            finish_image(j, img, H, W, t0, tm);
        } catch (const std::exception& e) {
            sdxl.unet.drop_context_cache();
            cudaStreamSynchronize(G.stream);
            cudaGetLastError();
            if (cancel) emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"cancelled\"}");
            else emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"error\",\"msg\":" + json_escape(e.what()) + "}");
        }
        Sdxl::free_cond(pos);
        Sdxl::free_cond(neg);
        G.arena.release(m0);
    }

    void generate(const Job& j) {
        if (j.family == "sdxl") {
            generate_sd(j);
            return;
        }
        auto t0 = Clock::now();
        struct Precision {  // restores the global GEMM mode however generate() exits
            bool saved = G.fp16;
            explicit Precision(bool fp32) { if (fp32) G.fp16 = false; }
            ~Precision() { G.fp16 = saved; }
        } precision(j.fp32);
        size_t m0 = G.arena.mark();
        if (getenv("KILN_POISON")) CK(cudaMemset(G.arena.base + m0, 0xFF, G.arena.cap - m0));  // NaN: exposes reads of unwritten scratch
        std::ostringstream tm;
        try {
            // the checkpoint picked in the UI; the text encoder and VAE are shared by every Anima model
            const std::string want = j.dit.empty() ? default_dit : j.dit;
            if (family != "anima") dit_path = want;  // the family swap loads it directly
            use_family("anima");
            int load_ms = -1;
            if (_stricmp(want.c_str(), dit_path.c_str()) != 0) {
                emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"loading\",\"what\":\"model\"}");
                auto tr = Clock::now();
                reload_dit(want);
                load_ms = (int)ms_since(tr);
                log_msg("model " + want + " loaded in " + std::to_string(load_ms) + " ms");
            }
            prepare_job();
            auto tl = Clock::now();
            std::string lr = loras.apply(dit, j.loras);
            if (!lr.empty()) log_msg("LoRA " + lr);
            tm << "\"lora_ms\":" << (int)ms_since(tl);
            if (load_ms >= 0) tm << ",\"load_ms\":" << load_ms;

            auto te0 = Clock::now();
            Context ctx = condition(j.pos), nctx;
            bool use_neg = j.cfg > 1.f;
            bool use_nag = !use_neg && j.nag.on;  // NAG is the negative prompt's route when CFG is off
            if (use_neg || use_nag) nctx = condition(j.neg);
            const Context* neg = use_neg ? &nctx : nullptr;
            auto with_nag = [&](SampleParams p) {
                if (use_nag) { p.nag_neg = &nctx; p.nag_scale = j.nag.scale; p.nag_tau = j.nag.tau; p.nag_alpha = j.nag.alpha; }
                return p;
            };
            gpu_sync();
            double enc_ms = ms_since(te0);
            emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"encoded\",\"ms\":" + std::to_string((int)enc_ms) + "}");
            tm << ",\"encode_ms\":" << (int)enc_ms;

            int H = j.height, W = j.width, Hl = H / 8, Wl = W / 8, P = Hl * Wl;
            float* x = G.arena.f((size_t)16 * P);
            double sample_ms = 0;
            sample(dit, x, Hl, Wl, ctx, neg, with_nag(SampleParams{j.steps, j.cfg, j.shift, 1.f, j.seed, j.cfg_until, j.sampler, j.cache}), nullptr, nullptr,
                   stepper(j, "base", Hl, Wl, &sample_ms), &cancel);
            tm << ",\"sample_ms\":" << (int)sample_ms;
            if (j.cache > 0.f) tm << ",\"cache_skips\":" << last_cache_skips;
            last_latent.resize((size_t)16 * P);
            CK(cudaMemcpy(last_latent.data(), x, last_latent.size() * 4, cudaMemcpyDeviceToHost));
            if (j.skip_decode) {
                emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"done\",\"skipped_decode\":true}");
                G.arena.release(m0);
                return;
            }

            auto td = Clock::now();
            latent_model_to_vae(x, P);
            float* img = G.arena.f((size_t)3 * H * W);
            vae.decode(x, Hl, Wl, img);
            gpu_sync();
            double dec_ms = ms_since(td);
            emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"decoded\",\"ms\":" + std::to_string((int)dec_ms) + "}");
            tm << ",\"decode_ms\":" << (int)dec_ms;

#ifdef KILN_FACE
            // faces first, at the base size: the hires pass then refines them along with the rest
            if (j.face.on) {
                auto tf = Clock::now();
                double fs_ms = 0;
                int n = face_detail(img, H, W, ctx, neg, j, &fs_ms, use_nag ? &nctx : nullptr);
                gpu_sync();
                tm << ",\"faces\":" << n << ",\"face_ms\":" << (int)ms_since(tf);
            }
#endif
            if (j.hires.scale > 1.01f) {  // hires fix: enlarge, then redraw at the new size with partial denoise
                auto th = Clock::now();
                int H2 = round16(H * j.hires.scale), W2 = round16(W * j.hires.scale), Hl2 = H2 / 8, Wl2 = W2 / 8, P2 = Hl2 * Wl2;
                float* img2 = G.arena.f((size_t)3 * H2 * W2);
                resize_image(img2, img, H, W, H2, W2, j.hires.model);
                float* z = G.arena.f((size_t)16 * P2);
                float* x2 = G.arena.f((size_t)16 * P2);
                vae.encode(img2, H2, W2, z);
                latent_vae_to_model(z, P2);
                double hs_ms = 0;
                sample(dit, x2, Hl2, Wl2, ctx, neg, with_nag(SampleParams{j.hires.steps, j.cfg, j.shift, j.hires.denoise, j.seed, j.cfg_until, j.sampler, j.cache}), z, nullptr,
                       stepper(j, "hires", Hl2, Wl2, &hs_ms), &cancel);
                latent_model_to_vae(x2, P2);
                vae.decode(x2, Hl2, Wl2, img2);
                img = img2; H = H2; W = W2;
                gpu_sync();
                tm << ",\"hires_ms\":" << (int)ms_since(th);
            }

            if (j.upscale > 1) {
                auto tu = Clock::now();
                int H2 = H * j.upscale, W2 = W * j.upscale;
                float* out = G.arena.f((size_t)3 * H2 * W2);
                resize_image(out, img, H, W, H2, W2, true);
                img = out; H = H2; W = W2;
                gpu_sync();
                tm << ",\"upscale_ms\":" << (int)ms_since(tu);
            }

            finish_image(j, img, H, W, t0, tm);
        } catch (const std::exception& e) {
            dit.drop_context_cache();
            cudaStreamSynchronize(G.stream);
            cudaGetLastError();
            if (cancel) emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"cancelled\"}");
            else emit("{\"id\":" + json_escape(j.id) + ",\"ev\":\"error\",\"msg\":" + json_escape(e.what()) + "}");
        }
        G.arena.release(m0);
    }
};

