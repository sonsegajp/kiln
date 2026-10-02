// The "train" command: an Anima LoRA from images with captions.
//   {"cmd":"train","id":..,"items":[{"image":path,"qwen_ids":[..],"t5_ids":[..],"t5_weights":[..]}, ...],
//    "out":".../name.safetensors","rank":32,"alpha":32,"epochs":10 | "steps":1000,"repeats":1,"optimizer":"prodigy"|"adamw","lr":1.0,
//    "resolution":512,"save_every_epochs":0,"seed":1,"grad_clip":1.0,"weight_decay":0.01,"warmup":0,"batch_size":1,
//    "timestep_sampling":"sigmoid"|"shift"|"uniform","sigmoid_scale":1.3,"discrete_flow_shift":3,"noise_offset":0,"flip":false,
//    "caption_dropout":0,"empty_caption":{tokens},"shuffle_caption":false,"keep_tokens":0,
//    (an item may carry "variants":[{tokens}, ...]: more orders of its caption, from tag shuffling)
//    "previews":[{qwen_ids,t5_ids,t5_weights}, ...],"preview_negative":{...},"preview_every":100,"preview_w":512,
//    "preview_h":768,"preview_steps":20,"preview_cfg":4,"preview_shift":3,"preview_seed":42}
// Events: train (stage loading / cache / ready), step (step, of, epoch, epochs, loss, avg, ms, lr, d), preview (step, index,
// file), saved (file), done (file) | error | cancelled. Previews are PNGs in <out without .safetensors>_previews/.
#pragma once
#include <filesystem>
#include <future>
#include <random>

#include "engine.h"
#include "image_io.h"
#include "optim.h"
#include "pipeline.h"
#include "train.h"

namespace trainjob {

struct Ctx { std::vector<float> rows; int len = 0, real = 0; };  // a caption's context on the host ([real, 1024])

struct Item {
    std::string image;
    std::vector<Cond> conds;  // the caption; with tag shuffling, several orders of it (one is picked per step)
    int bw = 0, bh = 0;
    std::vector<float> lat;       // [16, bh/8, bw/8] model space
    std::vector<float> lat_flip;  // the mirrored image's latent (flip augmentation)
    std::vector<Ctx> caps;        // conds through the text encoder and adapter
};

// sd-scripts-style bucket: sides in steps of 64 under the area res^2, the aspect closest to the image's
inline void bucket(int w, int h, int res, int& bw, int& bh) {
    const double area = (double)res * res, a = (double)w / h;
    double best = 1e9;
    bw = bh = res / 64 * 64;
    for (int W = 256; W <= 2048; W += 64) {
        const int H = (int)(area / W) / 64 * 64;
        if (H < 256 || H > 2048) continue;
        const double sc = std::fabs(std::log((double)W / H / a));
        if (sc < best - 1e-9) { best = sc; bw = W; bh = H; }
    }
}

inline void ev(const std::string& id, const std::string& body) { emit("{\"id\":" + json_escape(id) + "," + body + "}"); }

// ---- the on-disk cache
inline uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
// (its own name: fnv("|", h) would pick the pointer overload above and hash h bytes)
inline uint64_t fnv_str(const std::string& s, uint64_t h = 1469598103934665603ull) { return fnv(s.data(), s.size(), h); }
inline std::string hex(uint64_t h) { char b[17]; snprintf(b, sizeof b, "%016llx", (unsigned long long)h); return b; }
// a file's identity for the cache: path, size and modification time
inline std::string file_sig(const std::string& p) {
    std::error_code e1, e2;
    const auto sz = std::filesystem::file_size(p, e1);
    const auto mt = std::filesystem::last_write_time(p, e2);
    return p + "|" + std::to_string(e1 ? 0 : sz) + "|" + std::to_string(e2 ? 0 : (long long)mt.time_since_epoch().count());
}
inline bool read_file(const std::filesystem::path& f, std::vector<char>& out) {
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    if (!in) return false;
    out.resize((size_t)in.tellg());
    in.seekg(0);
    in.read(out.data(), (std::streamsize)out.size());
    return (bool)in;
}
// written under a temporary name, then renamed: a crash never leaves a short file under the real name
inline void write_file(const std::filesystem::path& f, const std::vector<std::pair<const void*, size_t>>& parts) {
    const std::string tmp = f.string() + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary);
        for (auto& [p, n] : parts) o.write((const char*)p, (std::streamsize)n);
        if (!o) return;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, f, ec);
    if (ec) std::filesystem::remove(tmp, ec);
}
// keep the cache under `cap` bytes, the least recently written files going first
inline void trim_cache(const std::filesystem::path& dir, uintmax_t cap) {
    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
    uintmax_t total = 0;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        total += e.file_size(ec);
        files.push_back({e.last_write_time(ec), e.path()});
    }
    if (total <= cap) return;
    std::sort(files.begin(), files.end());
    for (auto& [t, p] : files) {
        if (total <= cap * 3 / 4) break;
        const uintmax_t n = std::filesystem::file_size(p, ec);
        if (std::filesystem::remove(p, ec)) total -= n;
    }
}

inline Cond cond_of(const Json& j) {
    Cond c;
    for (auto& v : j["qwen_ids"].a) c.qwen.push_back((int)v.i64());
    for (auto& v : j["t5_ids"].a) c.t5.push_back((int)v.i64());
    for (auto& v : j["t5_weights"].a) c.t5w.push_back((float)v.num(1.0));
    c.t5w.resize(c.t5.size(), 1.f);
    return c;
}

// RGB8 -> PNG with stored (uncompressed) deflate blocks: small code, no zlib
inline void write_png(const std::string& path, const uint8_t* rgb, int w, int h) {
    static uint32_t crc_tab[256];
    static bool init = false;
    if (!init) {
        for (uint32_t n = 0; n < 256; n++) { uint32_t c = n; for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; crc_tab[n] = c; }
        init = true;
    }
    auto crc = [&](const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) { for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) & 0xFF] ^ (c >> 8); return c; };
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (w * 3 + 1));
    for (int y = 0; y < h; y++) { raw.push_back(0); raw.insert(raw.end(), rgb + (size_t)y * w * 3, rgb + (size_t)(y + 1) * w * 3); }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t o = 0; o < raw.size() || o == 0; o += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - o);
        z.push_back(o + n >= raw.size() ? 1 : 0);
        z.push_back(n & 0xFF); z.push_back(n >> 8); z.push_back(~n & 0xFF); z.push_back((~n >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + o, raw.begin() + o + n);
        if (o + n >= raw.size()) break;
    }
    const uint32_t ad = (b << 16) | a;
    for (int i = 3; i >= 0; i--) z.push_back((ad >> (8 * i)) & 0xFF);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    auto be32 = [&](uint32_t v) { uint8_t q[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v}; f.write((char*)q, 4); };
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        be32((uint32_t)data.size());
        std::vector<uint8_t> td(type, type + 4);
        td.insert(td.end(), data.begin(), data.end());
        f.write((const char*)td.data(), td.size());
        be32(crc(td.data(), td.size()) ^ 0xFFFFFFFFu);
    };
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    f.write((const char*)sig, 8);
    std::vector<uint8_t> ihdr = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
}

inline int run(Engine& E, const Json& r, const std::string& id) {
    auto t0 = Clock::now();
    const int rank = (int)r["rank"].i64(32), res = (int)r["resolution"].i64(512);
    const int save_every_epochs = (int)r["save_every_epochs"].i64(0), warmup = (int)r["warmup"].i64(0);
    const float alpha = (float)r["alpha"].num(rank), lr = (float)r["lr"].num(1.0), clip = (float)r["grad_clip"].num(1.0);
    // timesteps (sd-scripts' timestep_sampling): sigmoid(scale * randn), that pushed by the flow shift, or uniform
    const std::string ts_mode = r["timestep_sampling"].str("sigmoid");
    const float sig_scale = (float)r["sigmoid_scale"].num(1.3), flow_shift = (float)r["discrete_flow_shift"].num(3.0);
    const float noise_offset = (float)r["noise_offset"].num(0.0), caption_dropout = (float)r["caption_dropout"].num(0.0);
    const bool flip_aug = r["flip"].num(0) != 0;
    const int batch = std::max(1, std::min(64, (int)r["batch_size"].i64(1)));  // gradient accumulation: examples per optimizer step
    const unsigned long long seed = (unsigned long long)r["seed"].i64(1);
    std::string out = r["out"].str();
    const std::string opt_kind = r["optimizer"].str("prodigy");
    if (out.empty()) throw std::runtime_error("train: no output file");
    if (out.size() < 12 || out.compare(out.size() - 12, 12, ".safetensors") != 0) out += ".safetensors";
    const std::string stem = out.substr(0, out.size() - 12);
    std::vector<Item> items;
    for (auto& it : r["items"].a) {
        Item x;
        x.image = it["image"].str();
        x.conds.push_back(cond_of(it));
        for (auto& v : it["variants"].a) x.conds.push_back(cond_of(v));
        for (auto& cd : x.conds)
            if (cd.qwen.empty() || cd.t5.empty()) throw std::runtime_error("train: an item has no caption tokens: " + x.image);
        items.push_back(std::move(x));
    }
    if (items.empty()) throw std::runtime_error("train: no images");
    const int epochs = (int)r["epochs"].i64(0), repeats = std::max(1, (int)r["repeats"].i64(1));
    // previews
    std::vector<Cond> pv_conds;
    for (auto& p : r["previews"].a) pv_conds.push_back(cond_of(p));
    const bool pv_neg_on = r.has("preview_negative") && !r["preview_negative"]["t5_ids"].a.empty();
    const int pv_every = (int)r["preview_every"].i64(0), pv_steps = (int)r["preview_steps"].i64(20);
    const int pv_w = (int)r["preview_w"].i64(512) / 16 * 16, pv_h = (int)r["preview_h"].i64(768) / 16 * 16;
    const float pv_cfg = (float)r["preview_cfg"].num(4.0), pv_shift = (float)r["preview_shift"].num(3.0);
    const uint64_t pv_seed = (uint64_t)r["preview_seed"].i64(42);
    const std::string pv_dir = stem + "_previews";

    // ---- cache: latents through the VAE, captions through the text encoder and adapter
    ev(id, "\"ev\":\"train\",\"stage\":\"loading\"");
    // the Anima checkpoint to train on (renders with the same checkpoint match the training best)
    const std::string want = r["dit"].str().empty() ? E.default_dit : r["dit"].str();
    if (E.family != "anima") E.dit_path = want;  // the family swap loads it directly
    E.use_family("anima");
    if (_stricmp(want.c_str(), E.dit_path.c_str()) != 0) E.reload_dit(want);
    E.loras.reset();  // a render's LoRAs must not leak into the cached captions (or the frozen model)
    const size_t arena0 = G.arena.cap, mark0 = G.arena.mark();

    // The disk cache (Kiln/cache/train): latents keyed by the image file, the training resolution, flip and the VAE;
    // caption contexts by their tokens and the checkpoint (its adapter makes them). A rerun on the same dataset
    // only encodes what changed.
    namespace fs = std::filesystem;
    const fs::path cache_dir = fs::path(E.models).parent_path() / "cache" / "train";
    {
        std::error_code ec;
        fs::create_directories(cache_dir, ec);
        trim_cache(cache_dir, (uintmax_t)6 << 30);
    }
    const std::string vae_sig = file_sig(E.models + "\\qwen_image_vae.safetensors"), dit_sig = file_sig(E.dit_path);
    int ctx_hits = 0, ctx_miss = 0, lat_hits = 0;
    double ms_decode = 0, ms_vae = 0, ms_text = 0;
    auto host_ctx = [&](const Cond& cd) {
        uint64_t k = fnv_str(dit_sig + "|ctx1");
        k = fnv(cd.qwen.data(), cd.qwen.size() * 4, k);
        k = fnv(cd.t5.data(), cd.t5.size() * 4, fnv_str("|t5", k));
        k = fnv(cd.t5w.data(), cd.t5w.size() * 4, fnv_str("|w", k));
        const fs::path f = cache_dir / (hex(k) + ".ctx");
        Ctx h;
        std::vector<char> buf;
        if (read_file(f, buf) && buf.size() >= 12 && memcmp(buf.data(), "KCT1", 4) == 0) {
            memcpy(&h.len, buf.data() + 4, 4);
            memcpy(&h.real, buf.data() + 8, 4);
            if (buf.size() == 12 + (size_t)h.real * 1024 * 4) {
                h.rows.resize((size_t)h.real * 1024);
                memcpy(h.rows.data(), buf.data() + 12, h.rows.size() * 4);
                ctx_hits++;
                return h;
            }
        }
        auto t = Clock::now();
        const size_t m = G.arena.mark();
        Context c = E.condition(cd);
        h.len = c.len;
        h.real = c.real;
        h.rows.resize((size_t)c.real * 1024);
        CK(cudaMemcpy(h.rows.data(), c.p, h.rows.size() * 4, cudaMemcpyDeviceToHost));
        G.arena.release(m);  // a context parked in the scratch (no VRAM to cache it) lives on the host now
        write_file(f, {{"KCT1", 4}, {&h.len, 4}, {&h.real, 4}, {h.rows.data(), h.rows.size() * 4}});
        ms_text += ms_since(t);
        ctx_miss++;
        return h;
    };
    std::vector<Ctx> pv_ctx;
    Ctx pv_neg, empty_ctx;  // empty_ctx: the empty caption (caption dropout)
    {
        // latents: hits come straight from disk; misses decode on a worker thread one image ahead of the VAE
        struct Decoded { bool ok = false; int w = 0, h = 0; std::vector<uint8_t> rgb; std::string err; };
        std::vector<fs::path> lat_file(items.size());
        std::vector<char> hit(items.size(), 0);
        for (size_t i = 0; i < items.size(); i++) {
            Item& x = items[i];
            const uint64_t k = fnv_str(file_sig(x.image) + "|res" + std::to_string(res) + (flip_aug ? "|flip" : "") + "|" + vae_sig + "|lat1");
            lat_file[i] = cache_dir / (hex(k) + ".lat");
            std::vector<char> buf;
            if (read_file(lat_file[i], buf) && buf.size() >= 16 && memcmp(buf.data(), "KLT1", 4) == 0) {
                int bw, bh, fl;
                memcpy(&bw, buf.data() + 4, 4);
                memcpy(&bh, buf.data() + 8, 4);
                memcpy(&fl, buf.data() + 12, 4);
                const size_t n = (size_t)16 * (bh / 8) * (bw / 8);
                if (fl == (flip_aug ? 1 : 0) && buf.size() == 16 + n * 4 * (fl ? 2 : 1)) {
                    x.bw = bw;
                    x.bh = bh;
                    x.lat.resize(n);
                    memcpy(x.lat.data(), buf.data() + 16, n * 4);
                    if (fl) { x.lat_flip.resize(n); memcpy(x.lat_flip.data(), buf.data() + 16 + n * 4, n * 4); }
                    hit[i] = 1;
                    lat_hits++;
                }
            }
        }
        auto decode = [&](size_t i) {
            Decoded d;
            d.ok = load_image_rgb(items[i].image, d.w, d.h, d.rgb, d.err);
            return d;
        };
        auto next_miss = [&](size_t from) { while (from < items.size() && hit[from]) from++; return from; };
        std::vector<std::string> skipped;
        size_t pf = next_miss(0);
        std::future<Decoded> pending;
        if (pf < items.size()) pending = std::async(std::launch::async, decode, pf);
        int done = 0;
        for (size_t i = 0; i < items.size(); i++) {
            Item& x = items[i];
            if (E.cancel) throw std::runtime_error("cancelled");
            if (hit[i]) {
                for (auto& cd : x.conds) x.caps.push_back(host_ctx(cd));
                ev(id, "\"ev\":\"train\",\"stage\":\"cache\",\"done\":" + std::to_string(++done) + ",\"of\":" + std::to_string(items.size()) +
                           ",\"w\":" + std::to_string(x.bw) + ",\"h\":" + std::to_string(x.bh));
                continue;
            }
            auto td = Clock::now();
            Decoded d = pending.get();  // image i (the prefetch always runs on the next miss)
            const size_t nx = next_miss(i + 1);
            if (nx < items.size()) pending = std::async(std::launch::async, decode, nx);
            ms_decode += ms_since(td);
            if (!d.ok) {  // an image Windows can't decode is left out (reported below), not the end of the run
                skipped.push_back(fs::path(x.image).filename().string() + " (" + d.err + ")");
                ev(id, "\"ev\":\"train\",\"stage\":\"cache\",\"done\":" + std::to_string(++done) + ",\"of\":" + std::to_string(items.size()));
                continue;
            }
            const int w = d.w, h = d.h;
            const std::vector<uint8_t>& rgb = d.rgb;
            bucket(w, h, res, x.bw, x.bh);
            auto tv = Clock::now();
            const size_t m = G.arena.mark();
            std::vector<float> hf((size_t)3 * w * h);
            for (size_t p = 0; p < (size_t)w * h; p++)
                for (int c = 0; c < 3; c++) hf[(size_t)c * w * h + p] = rgb[p * 3 + c] / 127.5f - 1.f;
            float* img = G.arena.f(hf.size());
            CK(cudaMemcpy(img, hf.data(), hf.size() * 4, cudaMemcpyHostToDevice));
            const double sc = std::max((double)x.bw / w, (double)x.bh / h);
            const int rw = std::max(x.bw, (int)std::lround(w * sc)), rh = std::max(x.bh, (int)std::lround(h * sc));
            float* big = G.arena.f((size_t)3 * rw * rh);
            resize_lanczos(big, img, 3, h, w, rh, rw);
            float* crop = G.arena.f((size_t)3 * x.bw * x.bh);
            crop_rect(crop, big, 3, rh, rw, (rw - x.bw) / 2, (rh - x.bh) / 2, x.bw, x.bh);
            const int Hl = x.bh / 8, Wl = x.bw / 8;
            float* z = G.arena.f((size_t)16 * Hl * Wl);
            E.vae.encode(crop, x.bh, x.bw, z);
            latent_vae_to_model(z, Hl * Wl);
            x.lat.resize((size_t)16 * Hl * Wl);
            CK(cudaMemcpy(x.lat.data(), z, x.lat.size() * 4, cudaMemcpyDeviceToHost));
            if (flip_aug) {  // the mirrored image through the VAE too (a mirrored latent is not quite the same)
                std::vector<float> hc((size_t)3 * x.bw * x.bh), hm(hc.size());
                CK(cudaMemcpy(hc.data(), crop, hc.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t row = 0; row < (size_t)3 * x.bh; row++)
                    for (int col = 0; col < x.bw; col++) hm[row * x.bw + col] = hc[row * x.bw + (x.bw - 1 - col)];
                CK(cudaMemcpy(crop, hm.data(), hm.size() * 4, cudaMemcpyHostToDevice));
                E.vae.encode(crop, x.bh, x.bw, z);
                latent_vae_to_model(z, Hl * Wl);
                x.lat_flip.resize(x.lat.size());
                CK(cudaMemcpy(x.lat_flip.data(), z, x.lat_flip.size() * 4, cudaMemcpyDeviceToHost));
            }
            G.arena.release(m);
            ms_vae += ms_since(tv);
            {
                const int hd[4] = {0, x.bw, x.bh, flip_aug ? 1 : 0};
                std::vector<std::pair<const void*, size_t>> parts = {{"KLT1", 4}, {hd + 1, 12}, {x.lat.data(), x.lat.size() * 4}};
                if (flip_aug) parts.push_back({x.lat_flip.data(), x.lat_flip.size() * 4});
                write_file(lat_file[i], parts);
            }
            for (auto& cd : x.conds) x.caps.push_back(host_ctx(cd));
            ev(id, "\"ev\":\"train\",\"stage\":\"cache\",\"done\":" + std::to_string(++done) + ",\"of\":" + std::to_string(items.size()) +
                       ",\"w\":" + std::to_string(x.bw) + ",\"h\":" + std::to_string(x.bh));
        }
        for (auto& pc : pv_conds) pv_ctx.push_back(host_ctx(pc));
        if (pv_neg_on) pv_neg = host_ctx(cond_of(r["preview_negative"]));
        if (caption_dropout > 0.f) {
            Cond ec = cond_of(r["empty_caption"]);
            if (ec.qwen.empty() || ec.t5.empty()) throw std::runtime_error("train: caption dropout needs the empty caption's tokens");
            empty_ctx = host_ctx(ec);
        }
        if (!skipped.empty()) {
            std::string list;
            for (auto& s : skipped) list += (list.empty() ? "" : ", ") + s;
            log_msg("train: left out " + std::to_string(skipped.size()) + " image(s) that could not be read: " + list);
            ev(id, "\"ev\":\"train\",\"stage\":\"warn\",\"msg\":" + json_escape("Left out " + std::to_string(skipped.size()) + " image(s) that could not be read: " + list));
            items.erase(std::remove_if(items.begin(), items.end(), [](const Item& x) { return x.lat.empty(); }), items.end());
            if (items.empty()) throw std::runtime_error("none of the images could be read");
        }
        char tb[256];
        snprintf(tb, sizeof tb, "train: cached %zu images (%d latents from disk), %d captions (%d from disk) in %.1f s: decode %.1f s, VAE %.1f s, text %.1f s",
                 items.size(), lat_hits, ctx_hits + ctx_miss, ctx_hits, ms_since(t0) / 1000.0, ms_decode / 1000.0, ms_vae / 1000.0, ms_text / 1000.0);
        log_msg(tb);
    }
    // the schedule, from the images that made it
    const int nimg = (int)items.size();
    const int per_epoch = nimg * repeats;  // an epoch shows every image `repeats` times (sd-scripts' num_repeats)
    const int epoch_steps = (per_epoch + batch - 1) / batch;  // optimizer steps an epoch
    const int steps = epochs > 0 ? epochs * epoch_steps : (int)r["steps"].i64(1000);
    const int total_epochs = (steps + epoch_steps - 1) / epoch_steps;
    // ---- make room: the text encoder and every cached context go (the VAE stays for previews); the scratch grows
    gpu_sync();
    for (auto& c : E.ctx_cache) CK(cudaFree(c.ctx.p));
    E.ctx_cache.clear();
    E.dit.drop_context_cache();
    E.loras.reset();
    E.te.free();
    E.te = TextEncoder();
    G.arena.release(mark0);  // contexts parked in the scratch for lack of VRAM: their copies are on the host now
    struct Restore {  // however training ends: Anima reloads on the next job and the scratch shrinks back
        Engine& E;
        size_t cap, mark;
        ~Restore() {
            cudaStreamSynchronize(G.stream);
            cudaGetLastError();
            G.arena.release(mark);
            E.unload_anima();
            E.family = "none";
            if (G.arena.used == 0 && G.arena.cap != cap) gpu_resize_arena(cap);
        }
    } restore{E, arena0, mark0};

    size_t max_lat = (size_t)16 * (pv_h / 8) * (pv_w / 8);
    int max_len = 512, max_real = 0;
    size_t max_T = 0;
    auto widen = [&](const Ctx& p) { max_len = std::max(max_len, p.len); max_real = std::max(max_real, p.real); };
    for (auto& x : items) {
        max_lat = std::max(max_lat, x.lat.size());
        for (auto& cp : x.caps) widen(cp);
        max_T = std::max(max_T, (size_t)(x.bw / 16) * (x.bh / 16));
    }
    for (auto& p : pv_ctx) widen(p);
    widen(pv_neg);
    widen(empty_ctx);

    // ---- room on the GPU: the adapters and their gradients, then the scratch a step needs at the largest bucket
    // (and a preview). When the weights leave too little of it, part of the DiT moves to system RAM and streams.
    AnimaTrainer tr(E.dit);
    {
        const size_t pv_T = pv_ctx.empty() ? 0 : (size_t)(pv_w / 16) * (pv_h / 16);
        const size_t bufs = ((size_t)2 * max_lat + (size_t)2 * max_len * 1024 + (size_t)3 * max_lat + (size_t)3 * pv_w * pv_h) * 4;
        const size_t need = std::max(AnimaTrainer::step_scratch(max_T, max_real), AnimaTrainer::predict_scratch(pv_T, max_real)) + bufs;
        const size_t adapters = (size_t)2 * 4 * tr.param_count(rank);
        const size_t margin = ((size_t)160 << 20) + G.leave_free;  // cuBLAS and the driver: everything else uses the arena
        auto room = [&] { const size_t f = gpu_free_bytes(); return G.arena.cap + (f > margin ? f - margin : 0); };
        gpu_sync();
        if (room() < need + adapters) {
            int was = 0;
            for (auto& [nm, w] : E.dit.named) was += w->on_host;
            ev(id, "\"ev\":\"train\",\"stage\":\"loading\"");
            for (auto& [nm, w] : E.dit.named) free_weight(*w);
            E.dit = Dit();
            gpu_sync();
            // Auto placement keeps free what the arena lacks for the scratch, the adapters, the margin and the stage
            const size_t keep = G.reserve_bytes;
            G.reserve_bytes = (need > G.arena.cap ? need - G.arena.cap : 0) + adapters + margin + (G.stage ? 0 : STAGE_ELEMS * 2);
            try { E.dit.load(E.dit_path, Place::Auto); } catch (...) { G.reserve_bytes = keep; throw; }
            G.reserve_bytes = keep;
            int now = 0;
            for (auto& [nm, w] : E.dit.named) now += w->on_host;
            if (now && !G.stage) gpu_reserve_stage(STAGE_ELEMS);
            log_msg("train: " + std::to_string(need >> 20) + " MB of scratch at " + std::to_string(max_T) + " tokens; DiT tensors in system RAM " +
                    std::to_string(was) + " -> " + std::to_string(now) + " of " + std::to_string(E.dit.named.size()));
        }
        tr.init(rank, alpha, seed);
        const size_t fr = gpu_free_bytes();
        if (fr > margin + ((size_t)64 << 20)) {
            if (G.arena.used) log_msg("train: " + std::to_string(G.arena.used >> 20) + " MB of scratch still in use; it cannot grow");
            else gpu_resize_arena(G.arena.cap + fr - margin);
        }
        if (G.arena.cap + ((size_t)32 << 20) < need)
            throw std::runtime_error("not enough VRAM to train at " + std::to_string(res) + " px (" + std::to_string(need >> 20) + " MB of scratch needed, " +
                                     std::to_string(G.arena.cap >> 20) + " MB available); lower the resolution or close other GPU apps");
    }
    float* d_lat = G.arena.f(max_lat);
    float* d_noise = G.arena.f(max_lat);
    Context c;
    c.p = G.arena.f((size_t)max_len * 1024);
    auto put_ctx = [&](Context& dst, const std::vector<float>& rows, int len, int real) {
        fill(dst.p, 0.f, (size_t)len * 1024);
        CK(cudaMemcpyAsync(dst.p, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, G.stream));
        dst.len = len;
        dst.real = real;
    };

    LoraOptimizer opt;
    opt.kind = opt_kind == "adamw" ? "adamw" : "prodigy";
    opt.weight_decay = (float)r["weight_decay"].num(0.01);
    if (opt.kind == "adamw") opt.beta2 = 0.999f;
    {
        std::vector<float> hp(tr.n);
        CK(cudaMemcpy(hp.data(), tr.params, tr.n * 4, cudaMemcpyDeviceToHost));
        opt.init(hp.data(), tr.n);
    }
    float* hg = nullptr;
    CK(cudaMallocHost(&hg, tr.n * 4));
    struct FreeHost { float* p; ~FreeHost() { if (p) cudaFreeHost(p); } } free_hg{hg};

    char b[640];
    snprintf(b, sizeof b, "\"ev\":\"train\",\"stage\":\"ready\",\"params\":%zu,\"arena_mb\":%zu,\"images\":%d,\"steps\":%d,\"epochs\":%d,\"batch\":%d", tr.n,
             G.arena.cap >> 20, nimg, steps, total_epochs, batch);
    ev(id, b);

    // Previews go through txt2img's own path: the adapters as they are now are saved to a scratch LoRA file and loaded
    // the way a render loads LoRAs (next to the turbo LoRA in turbo mode), then the regular sampler runs with the mode's
    // settings and the same seed every time. A preview is what a render with that checkpoint gives.
    const std::string pv_turbo = r["preview_turbo_lora"].str();
    const std::string pv_sampler = r["preview_sampler"].str(pv_turbo.empty() ? "dpmpp_2m" : "euler");
    const float pv_cfg_until = (float)r["preview_cfg_until"].num(1.0);
    const bool pv_nag = r["preview_nag"].num(0) != 0;
    const std::string pv_lora = (std::filesystem::temp_directory_path() / ("kiln-train-preview-" + id + ".safetensors")).string();
    struct RemoveFile { std::string p; ~RemoveFile() { std::error_code ec; std::filesystem::remove(p, ec); } } rm_pv_lora{pv_lora};
    auto previews = [&](int step) {
        if (pv_ctx.empty()) return;
        std::filesystem::create_directories(pv_dir);
        tr.save(pv_lora, {});
        std::vector<LoraSpec> specs;
        if (!pv_turbo.empty()) specs.push_back({pv_turbo, 1.f});
        specs.push_back({pv_lora, 1.f});
        E.loras.reset();  // the same file name every time: force a fresh load
        E.loras.apply(E.dit, specs);
        struct Unload { Engine& E; ~Unload() { E.loras.reset(); } } unload{E};  // training needs the bare DiT back
        const int Hl = pv_h / 8, Wl = pv_w / 8;
        const size_t nl = (size_t)16 * Hl * Wl;
        for (size_t i = 0; i < pv_ctx.size(); i++) {
            if (E.cancel) return;
            const size_t m = G.arena.mark();
            float* x = G.arena.f(nl);
            Context cn;
            cn.p = G.arena.f((size_t)max_len * 1024);
            put_ctx(c, pv_ctx[i].rows, pv_ctx[i].len, pv_ctx[i].real);
            const bool cfg = pv_neg_on && pv_cfg > 1.f, nag = pv_neg_on && !cfg && pv_nag;  // NAG: the negative at CFG 1
            if (cfg || nag) put_ctx(cn, pv_neg.rows, pv_neg.len, pv_neg.real);
            SampleParams sp{pv_steps, pv_cfg, pv_shift, 1.f, pv_seed + i, pv_cfg_until, pv_sampler, 0.f};
            if (nag) sp.nag_neg = &cn;
            sample(E.dit, x, Hl, Wl, c, cfg ? &cn : nullptr, sp, nullptr, nullptr, [](int, int, float, const float*, const float*, double) {}, &E.cancel);
            latent_model_to_vae(x, Hl * Wl);
            float* img = G.arena.f((size_t)3 * pv_h * pv_w);
            E.vae.decode(x, Hl, Wl, img);
            std::vector<float> hf((size_t)3 * pv_h * pv_w);
            CK(cudaMemcpy(hf.data(), img, hf.size() * 4, cudaMemcpyDeviceToHost));
            G.arena.release(m);
            std::vector<uint8_t> rgb((size_t)pv_w * pv_h * 3);
            const size_t P = (size_t)pv_w * pv_h;
            for (size_t p = 0; p < P; p++)
                for (int ch = 0; ch < 3; ch++) rgb[p * 3 + ch] = (uint8_t)(std::min(1.f, std::max(0.f, (hf[ch * P + p] + 1.f) * 0.5f)) * 255.f);
            char nm[64];
            snprintf(nm, sizeof nm, "step%05d_%zu.png", step, i);
            const std::string file = (std::filesystem::path(pv_dir) / nm).string();
            write_png(file, rgb.data(), pv_w, pv_h);
            ev(id, "\"ev\":\"preview\",\"step\":" + std::to_string(step) + ",\"index\":" + std::to_string(i) + ",\"file\":" + json_escape(file) +
                       ",\"w\":" + std::to_string(pv_w) + ",\"h\":" + std::to_string(pv_h));
        }
    };

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd;
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::vector<size_t> order((size_t)per_epoch);  // each image `repeats` times, shuffled every epoch
    for (size_t i = 0; i < order.size(); i++) order[i] = i % items.size();
    auto sample_sigma = [&]() -> float {
        if (ts_mode == "uniform") return std::max(1e-3f, u01(rng));
        float t = 1.f / (1.f + expf(-sig_scale * nd(rng)));
        if (ts_mode == "shift") t = flow_shift * t / (1.f + (flow_shift - 1.f) * t);
        return t;
    };
    std::vector<float> hn(max_lat);
    double avg = 0;
    auto fmt = [](double v) { char t[32]; snprintf(t, sizeof t, "%g", v); return std::string(t); };
    std::vector<std::pair<std::string, std::string>> meta = {
        {"ss_network_module", "networks.lora_anima"}, {"ss_network_dim", std::to_string(rank)}, {"ss_network_alpha", std::to_string((int)alpha)},
        {"ss_base_model_version", "anima"}, {"ss_sd_model_name", std::filesystem::path(want).filename().string()}, {"ss_optimizer", opt.kind}, {"ss_learning_rate", std::to_string(lr)},
        {"ss_resolution", std::to_string(res)}, {"ss_num_train_images", std::to_string(per_epoch)}, {"ss_num_repeats", std::to_string(repeats)},
        {"ss_batch_size_per_device", std::to_string(batch)}, {"ss_timestep_sampling", ts_mode}, {"ss_sigmoid_scale", fmt(sig_scale)},
        {"ss_discrete_flow_shift", fmt(flow_shift)}, {"ss_noise_offset", fmt(noise_offset)}, {"ss_caption_dropout_rate", fmt(caption_dropout)},
        {"ss_flip_aug", flip_aug ? "True" : "False"}, {"ss_shuffle_caption", r["shuffle_caption"].num(0) ? "True" : "False"},
        {"ss_keep_tokens", std::to_string(r["keep_tokens"].i64(0))}, {"ss_training_comment", "trained with Kiln"}};
    auto save_as = [&](const std::string& path, int step) {
        auto mm = meta;
        mm.push_back({"ss_steps", std::to_string(step)});
        mm.push_back({"ss_epoch", std::to_string((step + epoch_steps - 1) / epoch_steps)});
        tr.save(path, mm);
        ev(id, "\"ev\":\"saved\",\"file\":" + json_escape(path) + ",\"step\":" + std::to_string(step) + ",\"epoch\":" + std::to_string((step + epoch_steps - 1) / epoch_steps));
    };
    if (pv_every > 0) previews(0);  // the base model, for comparison
    for (int s = 1; s <= steps; s++) {
        if (E.cancel) {
            if (s > 1) save_as(stem + "-step" + std::to_string(s - 1) + ".safetensors", s - 1);
            ev(id, "\"ev\":\"cancelled\"");
            return 0;
        }
        auto ts = Clock::now();
        const int epoch = (s - 1) / epoch_steps + 1, in_epoch = (s - 1) % epoch_steps;
        if (in_epoch == 0) std::shuffle(order.begin(), order.end(), rng);
        // a batch: its examples' gradients add up (each scaled by 1/n), then one optimizer step
        const int first = in_epoch * batch, nb = std::min(batch, per_epoch - first);
        tr.zero_grad();
        double loss_sum = 0;
        float sigma = 0.f;
        int lw = 0, lh = 0;
        for (int j = 0; j < nb; j++) {
            const Item& x = items[order[(size_t)first + j]];
            const bool flipped = !x.lat_flip.empty() && u01(rng) < 0.5f;
            const std::vector<float>& lat = flipped ? x.lat_flip : x.lat;
            const Ctx& cap = caption_dropout > 0.f && u01(rng) < caption_dropout ? empty_ctx
                             : x.caps[x.caps.size() > 1 ? (size_t)(rng() % x.caps.size()) : 0];
            sigma = sample_sigma();
            const int Hl = x.bh / 8, Wl = x.bw / 8;
            const size_t nl = lat.size(), hw = (size_t)Hl * Wl;
            for (size_t i = 0; i < nl; i++) hn[i] = nd(rng);
            if (noise_offset > 0.f)  // a per-channel constant on the noise (sd-scripts' noise_offset)
                for (int ch = 0; ch < 16; ch++) {
                    const float o = noise_offset * nd(rng);
                    for (size_t i = 0; i < hw; i++) hn[ch * hw + i] += o;
                }
            CK(cudaMemcpyAsync(d_lat, lat.data(), nl * 4, cudaMemcpyHostToDevice, G.stream));
            CK(cudaMemcpyAsync(d_noise, hn.data(), nl * 4, cudaMemcpyHostToDevice, G.stream));
            put_ctx(c, cap.rows, cap.len, cap.real);
            loss_sum += tr.step(d_lat, d_noise, Hl, Wl, sigma, c, 1.f / (float)nb);  // returns after a sync: hn is free again
            lw = x.bw;
            lh = x.bh;
        }
        const float loss = (float)(loss_sum / nb);
        CK(cudaMemcpy(hg, tr.grads, tr.n * 4, cudaMemcpyDeviceToHost));
        // cosine schedule with linear warmup
        float lr_t = lr;
        if (warmup > 0 && s <= warmup) lr_t = lr * (float)s / (float)warmup;
        else lr_t = lr * 0.5f * (1.f + cosf(3.14159265f * (float)(s - 1 - warmup) / (float)std::max(1, steps - warmup)));
        const double gnorm = opt.step(hg, lr_t, clip);
        CK(cudaMemcpy(tr.params, opt.p.data(), tr.n * 4, cudaMemcpyHostToDevice));
        avg = s == 1 ? loss : avg * 0.95 + loss * 0.05;
        snprintf(b, sizeof b,
                 "\"ev\":\"step\",\"stage\":\"train\",\"step\":%d,\"of\":%d,\"epoch\":%d,\"epochs\":%d,\"loss\":%.5f,\"avg\":%.5f,\"sigma\":%.3f,"
                 "\"lr\":%.3g,\"d\":%.3g,\"gnorm\":%.3g,\"ms\":%d,\"w\":%d,\"h\":%d",
                 s, steps, epoch, total_epochs, loss, avg, sigma, lr_t, opt.kind == "prodigy" ? opt.d : 0.0, gnorm, (int)ms_since(ts), lw, lh);
        ev(id, b);
        const bool epoch_end = s % epoch_steps == 0;
        if (save_every_epochs > 0 && epoch_end && epoch % save_every_epochs == 0 && s < steps)
            save_as(stem + "-e" + std::to_string(epoch) + ".safetensors", s);
        if (pv_every > 0 && s % pv_every == 0 && s < steps) previews(s);
    }
    save_as(out, steps);
    if (pv_every > 0) previews(steps);
    ev(id, "\"ev\":\"done\",\"out\":" + json_escape(out) + ",\"total_ms\":" + std::to_string((int)ms_since(t0)));
    return 0;
}

}  // namespace trainjob
