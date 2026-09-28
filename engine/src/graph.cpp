#include "graph.h"

#include <functional>
#include <set>

namespace {

// ---------------------------------------------------------------------------
// values flowing along links (host memory; nodes upload what they use)
// ---------------------------------------------------------------------------
struct Val {
    std::string type;  // MODEL CLIP VAE CONDITIONING LATENT IMAGE MASK UPSCALE_MODEL BBOX_DETECTOR ...
    // MODEL: the loaded DiT plus patches collected along the chain
    std::string unet;
    std::vector<LoraSpec> loras;
    float shift = 3.f, mult = 1.f;
    float cache = 0.f, cache_start = 0.15f, cache_end = 0.9f;
    int cache_max = 2;
    // CONDITIONING: tokens; encoded when a sampler runs, under that sampler's LoRA set
    Cond cond;
    // LATENT [B,16,h,w] (VAE space), IMAGE [B,3,H,W] in [0,1], MASK [B,H,W]
    std::vector<float> data;
    int B = 0, C = 0, H = 0, W = 0;
    std::shared_ptr<Val> noise_mask;  // LATENT: mask from SetLatentNoiseMask
    // UPSCALE_MODEL / BBOX_DETECTOR
    std::string path;
    // INT FLOAT BOOLEAN STRING (and combo values) produced by pack nodes
    Json prim;
};
using ValPtr = std::shared_ptr<Val>;

struct NodeError : std::runtime_error {
    std::string node;
    NodeError(const std::string& n, const std::string& m) : std::runtime_error(m), node(n) {}
};

bool exists(const std::string& p) { return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
std::string stem(std::string n) {
    size_t s = n.find_last_of("/\\");
    if (s != std::string::npos) n = n.substr(s + 1);
    size_t d = n.find_last_of('.');
    return d == std::string::npos ? n : n.substr(0, d);
}

// ---------------------------------------------------------------------------
// graph run
// ---------------------------------------------------------------------------
struct Run {
    Engine& E;
    std::string id, out_dir;
    const Json& graph;
    const Json& tokens;
    const Json& images;
    const Json& ext;  // pack nodes: class_type -> {outputs, output_node}
    bool preview = true;
    std::map<std::string, std::vector<ValPtr>> done;
    std::map<std::string, double> node_ms;
    std::set<std::string> visiting;
    std::string current;
    ValPtr done_mask;  // FaceDetailer's combined mask output

    Run(Engine& e, const Json& req)
        : E(e), id(req["id"].str()), out_dir(req["out_dir"].str()), graph(req["graph"]), tokens(req["tokens"]), images(req["images"]), ext(req["ext"]) {
        preview = req["preview"].type == Json::Null ? true : req["preview"].b;
    }

    [[noreturn]] void fail(const std::string& msg) { throw NodeError(current, msg); }

    void event(const std::string& body) { emit("{\"id\":" + json_escape(id) + "," + body + "}"); }

    // ---- inputs
    const Json& node(const std::string& nid) {
        const Json& n = graph[nid];
        if (n.type != Json::Obj) throw NodeError(nid, "missing node " + nid);
        return n;
    }
    ValPtr link(const std::string& nid, const char* name, const char* want) {
        const Json& in = node(nid)["inputs"][name];
        if (in.type != Json::Arr || in.size() != 2) fail(std::string("input '") + name + "' is not connected");
        std::string src = in[0].type == Json::Str ? in[0].s : std::to_string(in[0].i64());
        auto& outs = eval(src);
        size_t k = (size_t)in[1].i64();
        if (k >= outs.size()) fail(std::string("input '") + name + "': node " + src + " has no output " + std::to_string(k));
        ValPtr v = outs[k];
        if (want && v->type != want) fail(std::string("input '") + name + "' needs " + want + ", got " + v->type);
        return v;
    }
    static bool is_link(const Json& v) { return v.type == Json::Arr && v.size() == 2 && v[1].type == Json::Num; }
    // a widget value: the literal, or the primitive a pack node (e.g. a Seed node) produced for it
    Json value(const std::string& nid, const char* name) {
        const Json& v = node(nid)["inputs"][name];
        if (!is_link(v)) return v;
        ValPtr p = link(nid, name, nullptr);
        if (p->prim.type == Json::Null) fail(std::string("input '") + name + "' needs a value, got " + p->type);
        return p->prim;
    }
    double num(const std::string& nid, const char* name, double def) {
        Json v = value(nid, name);
        if (v.type == Json::Null) return def;
        if (v.type == Json::Bool) return v.b ? 1 : 0;
        if (v.type != Json::Num) fail(std::string("input '") + name + "' must be a number");
        return v.n;
    }
    std::string str(const std::string& nid, const char* name, const std::string& def) {
        Json v = value(nid, name);
        if (v.type == Json::Null) return def;
        if (v.type == Json::Bool) return v.b ? "true" : "false";
        return v.type == Json::Str ? v.s : def;
    }
    bool flag(const std::string& nid, const char* name, bool def) {
        Json v = value(nid, name);
        if (v.type == Json::Bool) return v.b;
        if (v.type == Json::Num) return v.n != 0;
        if (v.type == Json::Str) return v.s == "true" || v.s == "True" || v.s == "enable";
        return def;
    }
    uint64_t seed(const std::string& nid, const char* name) {
        Json v = value(nid, name);
        if (v.type != Json::Num) fail(std::string("input '") + name + "' must be a number");
        return (uint64_t)v.n;
    }

    // ---- model files
    std::string models() { return E.models; }
    std::string find_model(const std::string& name, std::initializer_list<const char*> dirs, std::initializer_list<const char*> exts) {
        std::string base = name;
        for (char& c : base) if (c == '/') c = '\\';
        for (const char* d : dirs) {
            std::string p = models() + (d[0] ? std::string("\\") + d : "") + "\\" + base;
            if (exists(p)) return p;
        }
        std::string st = stem(name);  // e.g. bbox/face_yolov8m.pt -> face_yolov8m.safetensors
        for (const char* d : dirs)
            for (const char* e : exts) {
                std::string p = models() + (d[0] ? std::string("\\") + d : "") + "\\" + st + e;
                if (exists(p)) return p;
            }
        fail("model file not found: " + name);
    }

    // ---- GPU helpers
    void ensure_model(const Val& m) {
        if (_stricmp(m.unet.c_str(), E.dit_path.c_str()) != 0) {
            event("\"ev\":\"log\",\"msg\":" + json_escape("loading diffusion model " + stem(m.unet)));
            E.reload_dit(m.unet);
        }
        std::string lr = E.loras.apply(E.dit, m.loras);
        if (!lr.empty()) log_msg("LoRA " + lr);
    }

    // IMAGE item b as a device [3,H,W] buffer in [-1,1] (arena)
    float* image_to_device(const Val& im, int b) {
        size_t n = (size_t)3 * im.H * im.W;
        float* d = G.arena.f(n);
        CK(cudaMemcpy(d, im.data.data() + b * n, n * 4, cudaMemcpyHostToDevice));
        rgb_from_unit(d, d, n);
        return d;
    }
    void image_from_device(Val& im, int b, const float* d_pm1) {
        size_t n = (size_t)3 * im.H * im.W;
        std::vector<float> h(n);
        CK(cudaMemcpy(h.data(), d_pm1, n * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) im.data[b * n + i] = std::min(1.f, std::max(0.f, (h[i] + 1.f) * 0.5f));
    }

    StepFn stepper(const std::string& nid, int Hl, int Wl) {
        return [this, nid, Hl, Wl](int step, int of, float s, const float* x, const float* v, double ms) {
            std::string pv = preview ? E.preview_json(x, v, s, Hl, Wl) : "";
            event("\"ev\":\"step\",\"node\":" + json_escape(nid) + ",\"step\":" + std::to_string(step) + ",\"of\":" + std::to_string(of) +
                  ",\"ms\":" + std::to_string((int)ms) + pv);
        };
    }

    // bilinear resize of a [Hs,Ws] mask to [Hd,Wd] (align_corners=False), on the host
    static std::vector<float> resize_mask(const float* m, int Hs, int Ws, int Hd, int Wd) {
        std::vector<float> o((size_t)Hd * Wd);
        for (int y = 0; y < Hd; y++) {
            float fy = std::max((y + 0.5f) * Hs / Hd - 0.5f, 0.f);
            int y0 = std::min((int)fy, Hs - 1), y1 = std::min(y0 + 1, Hs - 1);
            float ly = fy - y0;
            for (int x = 0; x < Wd; x++) {
                float fx = std::max((x + 0.5f) * Ws / Wd - 0.5f, 0.f);
                int x0 = std::min((int)fx, Ws - 1), x1 = std::min(x0 + 1, Ws - 1);
                float lx = fx - x0;
                o[(size_t)y * Wd + x] = (m[(size_t)y0 * Ws + x0] * (1 - lx) + m[(size_t)y0 * Ws + x1] * lx) * (1 - ly) +
                                        (m[(size_t)y1 * Ws + x0] * (1 - lx) + m[(size_t)y1 * Ws + x1] * lx) * ly;
            }
        }
        return o;
    }

    // ---- sampling shared by the KSampler family
    ValPtr run_sampler(const std::string& nid, const Val& model, const Val& pos, const Val* neg, const Val* nag_neg, const Val& latent,
                       SampleParams sp) {
        ensure_model(model);
        Context cpos = E.condition(pos.cond), cneg, cnag;
        bool use_neg = sp.cfg > 1.f && neg;
        if (use_neg) cneg = E.condition(neg->cond);
        if (nag_neg) { cnag = E.condition(nag_neg->cond); sp.nag_neg = &cnag; }  // with CFG, NAG guides the prompt pass
        sp.shift = model.shift;
        sp.timestep_mult = model.mult;
        sp.cache_threshold = model.cache;
        sp.cache_start = model.cache_start;
        sp.cache_end = model.cache_end;
        sp.cache_max_hits = model.cache_max;

        const int Hl = latent.H, Wl = latent.W, P = Hl * Wl, B = latent.B;
        const size_t n = (size_t)16 * P;
        auto out = std::make_shared<Val>(latent);
        out->noise_mask = nullptr;
        // one randn(seed) tensor for the whole batch, sliced per item (like ComfyUI)
        std::vector<float> noise = torch_randn(sp.seed, n * B);
        for (int b = 0; b < B; b++) {
            size_t m = G.arena.mark();
            float* init = G.arena.f(n);
            float* x = G.arena.f(n);
            CK(cudaMemcpy(init, latent.data.data() + b * n, n * 4, cudaMemcpyHostToDevice));
            latent_vae_to_model(init, P);
            float* mask = nullptr;
            if (latent.noise_mask) {
                const Val& nm = *latent.noise_mask;
                int mb = std::min(b, nm.B - 1);
                auto lm = resize_mask(nm.data.data() + (size_t)mb * nm.H * nm.W, nm.H, nm.W, Hl, Wl);
                mask = G.arena.f(P);
                CK(cudaMemcpy(mask, lm.data(), (size_t)P * 4, cudaMemcpyHostToDevice));
            }
            SampleParams spb = sp;
            spb.noise = noise.data() + b * n;
            sample(E.dit, x, Hl, Wl, cpos, use_neg ? &cneg : nullptr, spb, init, mask, stepper(nid, Hl, Wl), &E.cancel);
            latent_model_to_vae(x, P);
            CK(cudaMemcpy(out->data.data() + b * n, x, n * 4, cudaMemcpyDeviceToHost));
            G.arena.release(m);
        }
        return out;
    }

    // ---- operations shared by the built-in nodes and pack nodes' ctx.ops
    static ValPtr make(const char* type) {
        auto v = std::make_shared<Val>();
        v->type = type;
        return v;
    }
    ValPtr cond_from_tokens(const Json& t) {
        if (t.type != Json::Obj) fail("prompt was not tokenized by the server");
        auto v = make("CONDITIONING");
        for (auto& x : t["qwen_ids"].a) v->cond.qwen.push_back((int)x.i64());
        for (auto& x : t["t5_ids"].a) v->cond.t5.push_back((int)x.i64());
        for (auto& x : t["t5_weights"].a) v->cond.t5w.push_back((float)x.num());
        if (v->cond.qwen.empty()) v->cond.qwen.push_back(151643);
        if (v->cond.t5.empty()) v->cond.t5.push_back(1);
        v->cond.t5w.resize(v->cond.t5.size(), 1.f);
        return v;
    }
    ValPtr vae_encode(const Val& im) {
        int H = im.H / 8 * 8, W = im.W / 8 * 8;  // ComfyUI center-crops to a multiple of 8
        if (H < 8 || W < 8) fail("image too small to encode");
        int oy = (im.H % 8) / 2, ox = (im.W % 8) / 2;
        auto v = make("LATENT");
        v->B = im.B; v->C = 16; v->H = H / 8; v->W = W / 8;
        size_t n = (size_t)16 * v->H * v->W;
        v->data.resize(n * v->B);
        for (int b = 0; b < im.B; b++) {
            size_t m = G.arena.mark();
            float* full = image_to_device(im, b);
            float* crop = G.arena.f((size_t)3 * H * W);
            crop_rect(crop, full, 3, im.H, im.W, ox, oy, W, H);
            float* z = G.arena.f(n);
            E.vae.encode(crop, H, W, z);
            CK(cudaMemcpy(v->data.data() + b * n, z, n * 4, cudaMemcpyDeviceToHost));
            G.arena.release(m);
        }
        return v;
    }
    ValPtr vae_decode(const Val& lat) {
        auto v = make("IMAGE");
        v->B = lat.B; v->C = 3; v->H = lat.H * 8; v->W = lat.W * 8;
        size_t n = (size_t)16 * lat.H * lat.W, ni = (size_t)3 * v->H * v->W;
        v->data.resize(ni * v->B);
        for (int b = 0; b < lat.B; b++) {
            size_t m = G.arena.mark();
            float* out = G.arena.f(ni);
            float* z = G.arena.f(n);
            CK(cudaMemcpy(z, lat.data.data() + b * n, n * 4, cudaMemcpyHostToDevice));
            E.vae.decode(z, lat.H, lat.W, out);
            image_from_device(*v, b, out);
            G.arena.release(m);
        }
        return v;
    }
    // IMAGE, MASK or LATENT to Wo x Ho; `center` crops the source to the target aspect first
    ValPtr resize(const Val& src, int Wo, int Ho, const std::string& method, bool center) {
        bool latent = src.type == "LATENT";
        int C = latent ? 16 : src.type == "MASK" ? 1 : 3;
        auto v = std::make_shared<Val>(src);
        v->H = Ho; v->W = Wo;
        if (!latent) v->noise_mask = nullptr;
        size_t ns = (size_t)C * src.H * src.W, nd = (size_t)C * Ho * Wo;
        v->data.assign(nd * src.B, 0.f);
        for (int b = 0; b < src.B; b++) {
            size_t m = G.arena.mark();
            float* in = G.arena.f(ns);
            CK(cudaMemcpy(in, src.data.data() + b * ns, ns * 4, cudaMemcpyHostToDevice));
            int cx = 0, cy = 0, cw = src.W, ch = src.H;
            if (center) {
                double ra = (double)src.W / src.H, rb = (double)Wo / Ho;
                if (ra > rb) { cw = (int)std::lround(src.H * rb); cx = (src.W - cw) / 2; }
                else if (ra < rb) { ch = (int)std::lround(src.W / rb); cy = (src.H - ch) / 2; }
            }
            float* cropped = in;
            if (cw != src.W || ch != src.H) {
                cropped = G.arena.f((size_t)C * cw * ch);
                crop_rect(cropped, in, C, src.H, src.W, cx, cy, cw, ch);
            }
            float* out = G.arena.f(nd);
            resize_mode(out, cropped, C, ch, cw, Ho, Wo, method);
            CK(cudaMemcpy(v->data.data() + b * nd, out, nd * 4, cudaMemcpyDeviceToHost));
            G.arena.release(m);
        }
        if (!latent) for (auto& x : v->data) x = std::min(1.f, std::max(0.f, x));
        return v;
    }
    ValPtr upscale(const Val& um, const Val& im) {
#ifdef KILN_UPSCALE
        Upscaler* up = E.upscale_model(um.path);
        if (!up) fail("upscale model could not be loaded");
        int s = up->scale();
        auto v = make("IMAGE");
        v->B = im.B; v->C = 3; v->H = im.H * s; v->W = im.W * s;
        size_t ni = (size_t)3 * im.H * im.W, no = ni * s * s;
        v->data.resize(no * im.B);
        for (int b = 0; b < im.B; b++) {
            size_t m = G.arena.mark();
            float* in = G.arena.f(ni);
            float* out = G.arena.f(no);
            CK(cudaMemcpy(in, im.data.data() + b * ni, ni * 4, cudaMemcpyHostToDevice));
            up->run(out, in, im.H, im.W);
            CK(cudaMemcpy(v->data.data() + b * no, out, no * 4, cudaMemcpyDeviceToHost));
            G.arena.release(m);
        }
        for (auto& x : v->data) x = std::min(1.f, std::max(0.f, x));
        return v;
#else
        (void)um; (void)im;
        fail("this engine build has no upscaler");
#endif
    }

    // ---- pack nodes (docs/NODE_API.md): JavaScript nodes run by the server, called over stdin/stdout
    std::map<std::string, ValPtr> handles;  // MODEL, CLIP, VAE, CONDITIONING, ... handed to pack code
    int ext_seq = 0, file_seq = 0;

    std::string write_tensor(const std::string& type, const float* data, const std::vector<int64_t>& shape) {
        std::string path = out_dir + "\\" + id + "_x" + std::to_string(file_seq++) + ".f32";
        size_t n = 1;
        for (auto d : shape) n *= (size_t)d;
        std::ofstream f(path, std::ios::binary);
        f.write((const char*)data, n * 4);
        if (!f) fail("cannot write " + path);
        std::string s = "{\"type\":" + json_escape(type) + ",\"path\":" + json_escape(path) + ",\"shape\":[";
        for (size_t i = 0; i < shape.size(); i++) s += (i ? "," : "") + std::to_string(shape[i]);
        return s + "]}";
    }
    // IMAGE goes out interleaved [B,H,W,3] (the layout ported nodes expect), MASK [B,H,W], LATENT [B,16,h,w]
    std::string to_wire(const ValPtr& v) {
        if (v->prim.type != Json::Null) return json_dump(v->prim);
        if (v->type == "IMAGE") {
            size_t hw = (size_t)v->H * v->W;
            std::vector<float> hwc(v->data.size());
            for (int b = 0; b < v->B; b++)
                for (int c = 0; c < 3; c++)
                    for (size_t p = 0; p < hw; p++) hwc[((size_t)b * hw + p) * 3 + c] = v->data[((size_t)b * 3 + c) * hw + p];
            return write_tensor("IMAGE", hwc.data(), {v->B, v->H, v->W, 3});
        }
        if (v->type == "MASK") return write_tensor("MASK", v->data.data(), {v->B, v->H, v->W});
        if (v->type == "LATENT") return write_tensor("LATENT", v->data.data(), {v->B, 16, v->H, v->W});
        std::string h = "h" + std::to_string(handles.size() + 1);
        handles[h] = v;
        return "{\"type\":" + json_escape(v->type) + ",\"handle\":" + json_escape(h) + "}";
    }
    ValPtr from_wire(const Json& w, const std::string& type) {
        if (w.type == Json::Obj && w.has("handle") && w["handle"].str().rfind("js:", 0) != 0) {
            auto it = handles.find(w["handle"].str());
            if (it == handles.end()) fail("unknown handle " + w["handle"].str());
            return it->second;
        }
        auto v = std::make_shared<Val>();
        v->type = type;
        if (w.type != Json::Obj || !w.has("path")) {  // INT FLOAT STRING BOOLEAN, js: handles, any JSON a pack passes along
            if (w.type == Json::Obj && w.has("type")) v->type = w["type"].str(type);
            v->prim = w;
            return v;
        }
        std::vector<int64_t> shape;
        size_t n = 1;
        for (auto& d : w["shape"].a) { shape.push_back(d.i64()); n *= (size_t)d.i64(); }
        std::vector<float> buf(n);
        std::string path = w["path"].str();
        {
            std::ifstream f(path, std::ios::binary);
            if (!f || !f.read((char*)buf.data(), n * 4)) fail("cannot read " + path);
        }
        DeleteFileA(path.c_str());
        v->type = w["type"].str(type);
        if (v->type == "IMAGE") {
            if (shape.size() != 4 || shape[3] != 3) fail("an IMAGE must be [B,H,W,3]");
            v->B = (int)shape[0]; v->H = (int)shape[1]; v->W = (int)shape[2]; v->C = 3;
            size_t hw = (size_t)v->H * v->W;
            v->data.resize(n);
            for (int b = 0; b < v->B; b++)
                for (int c = 0; c < 3; c++)
                    for (size_t p = 0; p < hw; p++) v->data[((size_t)b * 3 + c) * hw + p] = buf[((size_t)b * hw + p) * 3 + c];
        } else if (v->type == "MASK") {
            if (shape.size() != 3) fail("a MASK must be [B,H,W]");
            v->B = (int)shape[0]; v->H = (int)shape[1]; v->W = (int)shape[2];
            v->data = std::move(buf);
        } else if (v->type == "LATENT") {
            if (shape.size() != 4 || shape[1] != 16) fail("a LATENT must be [B,16,h,w]");
            v->B = (int)shape[0]; v->C = 16; v->H = (int)shape[2]; v->W = (int)shape[3];
            v->data = std::move(buf);
        } else {
            fail("pack nodes can't return a tensor of type " + v->type);
        }
        return v;
    }

    std::vector<ValPtr> ext_node(const std::string& nid, const std::string& ct) {
        std::string ins;
        for (auto& [name, v] : node(nid)["inputs"].o) {
            std::string w = is_link(v) ? to_wire(link(nid, name.c_str(), nullptr)) : json_dump(v);
            ins += (ins.empty() ? "" : ",") + json_escape(name) + ":" + w;
        }
        const int call = ++ext_seq;
        event("\"ev\":\"ext_call\",\"call\":" + std::to_string(call) + ",\"node\":" + json_escape(nid) + ",\"class_type\":" + json_escape(ct) +
              ",\"inputs\":{" + ins + "}");
        for (;;) {
            Json m;
            {
                std::unique_lock<std::mutex> lk(E.ext_inbox.mu);
                E.ext_inbox.cv.wait(lk, [&] { return E.cancel || !E.ext_inbox.q.empty(); });
                if (E.cancel) throw std::runtime_error("cancelled");
                m = std::move(E.ext_inbox.q.front());
                E.ext_inbox.q.pop_front();
            }
            if (m["id"].str() != id || m["call"].i64() != call) continue;  // late reply for a finished call
            if (m["cmd"].str() == "ext_op") {
                ext_op(m);
                continue;
            }
            if (m.has("error")) fail(m["error"].str("pack node failed"));
            const Json& types = ext[ct]["outputs"];
            std::vector<ValPtr> outs;
            for (size_t i = 0; i < m["outputs"].size(); i++) outs.push_back(from_wire(m["outputs"][i], i < types.size() ? types[i].str("*") : "*"));
            return outs;
        }
    }

    // ctx.ops.* from pack code, run on this worker while its node waits
    void ext_op(const Json& m) {
        const std::string op = m["op"].str(), opid = json_dump(m["op_id"]);
        try {
            auto arg = [&](const char* k, const char* type) {
                if (m[k].type == Json::Null) fail(std::string("op ") + op + ": missing '" + k + "'");
                return from_wire(m[k], type);
            };
            std::string result;
            if (op == "vae_decode") result = to_wire(vae_decode(*arg("latent", "LATENT")));
            else if (op == "vae_encode") result = to_wire(vae_encode(*arg("image", "IMAGE")));
            else if (op == "resize") {
                int w = (int)m["width"].i64(), h = (int)m["height"].i64();
                if (w < 1 || h < 1) fail("op resize: width and height must be positive");
                result = to_wire(resize(*arg("image", "IMAGE"), w, h, m["method"].str("lanczos"), false));
            } else if (op == "upscale") result = to_wire(upscale(*arg("model", "UPSCALE_MODEL"), *arg("image", "IMAGE")));
            else if (op == "detect_faces") result = detect_faces(*arg("image", "IMAGE"), (float)m["threshold"].num(0.35));
            else if (op == "encode_text") result = to_wire(cond_from_tokens(m));  // qwen_ids, t5_ids, t5_weights
            else if (op == "sample") result = to_wire(ext_sample(m));
            else fail("unknown op '" + op + "'");
            event("\"ev\":\"ext_op_done\",\"op_id\":" + opid + ",\"result\":" + result);
        } catch (const std::exception& e) {
            if (E.cancel) throw;
            event("\"ev\":\"ext_op_done\",\"op_id\":" + opid + ",\"error\":" + json_escape(e.what()));
        }
    }
    std::string detect_faces(const Val& im, float thr) {
#ifdef KILN_FACE
        FaceDetector& fd = E.face_model();
        std::string s = "[";
        for (int b = 0; b < im.B; b++) {
            size_t m = G.arena.mark();
            float* unit = G.arena.f((size_t)3 * im.H * im.W);
            CK(cudaMemcpy(unit, im.data.data() + (size_t)b * 3 * im.H * im.W, (size_t)3 * im.H * im.W * 4, cudaMemcpyHostToDevice));
            std::vector<Box> boxes = fd.detect(unit, im.H, im.W, thr, 0.5f);
            G.arena.release(m);
            s += b ? ",[" : "[";
            for (size_t i = 0; i < boxes.size(); i++) {
                char buf[160];
                snprintf(buf, sizeof buf, "%s{\"x0\":%.1f,\"y0\":%.1f,\"x1\":%.1f,\"y1\":%.1f,\"score\":%.4f}", i ? "," : "", boxes[i].x0, boxes[i].y0,
                         boxes[i].x1, boxes[i].y1, boxes[i].score);
                s += buf;
            }
            s += "]";
        }
        return s + "]";
#else
        (void)im; (void)thr;
        fail("this engine build has no face detector");
#endif
    }
    ValPtr ext_sample(const Json& m) {
        auto model = from_wire(m["model"], "MODEL");
        auto pos = from_wire(m["positive"], "CONDITIONING");
        ValPtr neg = m["negative"].type == Json::Null ? nullptr : from_wire(m["negative"], "CONDITIONING");
        auto lat = from_wire(m["latent"], "LATENT");
        if (model->type != "MODEL" || pos->type != "CONDITIONING" || lat->type != "LATENT") fail("op sample: needs a MODEL, CONDITIONING and LATENT");
        if (m["mask"].type != Json::Null) {
            lat = std::make_shared<Val>(*lat);
            lat->noise_mask = from_wire(m["mask"], "MASK");
        }
        const Json& o = m;
        SampleParams sp;
        sp.seed = (uint64_t)o["seed"].num(0);
        sp.steps = (int)o["steps"].num(20);
        sp.cfg = (float)o["cfg"].num(4.5);
        sp.sampler = o["sampler"].str("euler");
        sp.scheduler = o["scheduler"].str("simple");
        sp.denoise = (float)o["denoise"].num(1.0);
        if (!known_sampler(sp.sampler)) fail("op sample: sampler '" + sp.sampler + "' is not available in Kiln");
        if (!known_scheduler(sp.scheduler)) fail("op sample: unknown scheduler '" + sp.scheduler + "'");
        return run_sampler(current, *model, *pos, neg.get(), nullptr, *lat, sp);
    }

    // ---- nodes
    std::vector<ValPtr> exec(const std::string& nid, const std::string& ct);

    std::vector<ValPtr>& eval(const std::string& nid) {
        auto it = done.find(nid);
        if (it != done.end()) return it->second;
        if (visiting.count(nid)) throw NodeError(nid, "cycle in the graph");
        visiting.insert(nid);
        std::string ct = node(nid)["class_type"].str();
        // inputs first (depth-first), so the running node's events come after its dependencies
        for (auto& [name, v] : node(nid)["inputs"].o)
            if (v.type == Json::Arr && v.size() == 2) eval(v[0].type == Json::Str ? v[0].s : std::to_string(v[0].i64()));
        if (E.cancel) throw std::runtime_error("cancelled");
        std::string saved = current;
        current = nid;
        event("\"ev\":\"node\",\"node\":" + json_escape(nid) + ",\"class_type\":" + json_escape(ct) + ",\"status\":\"start\"");
        auto t0 = Clock::now();
        std::vector<ValPtr> outs = exec(nid, ct);
        gpu_sync();
        double ms = ms_since(t0);
        node_ms[nid] = ms;
        event("\"ev\":\"node\",\"node\":" + json_escape(nid) + ",\"class_type\":" + json_escape(ct) + ",\"status\":\"done\",\"ms\":" + std::to_string((int)ms));
        current = saved;
        visiting.erase(nid);
        return done[nid] = std::move(outs);
    }

    int image_counter = 0;
    void write_images(const std::string& nid, const Val& im, const char* kind, const std::string& prefix) {
        size_t hw = (size_t)im.H * im.W;
        for (int b = 0; b < im.B; b++) {
            std::vector<uint8_t> px(hw * 3);
            const float* src = im.data.data() + (size_t)b * 3 * hw;
            for (int c = 0; c < 3; c++)
                for (size_t p = 0; p < hw; p++) px[p * 3 + c] = (uint8_t)(std::min(1.f, std::max(0.f, src[c * hw + p])) * 255.f);
            std::string path = out_dir + "\\" + id + "_" + nid + "_" + std::to_string(image_counter++) + ".rgb";
            std::ofstream f(path, std::ios::binary);
            if (!f) fail("cannot write " + path);
            f.write((const char*)px.data(), px.size());
            f.close();
            event("\"ev\":\"image\",\"node\":" + json_escape(nid) + ",\"index\":" + std::to_string(b) + ",\"kind\":\"" + kind + "\",\"prefix\":" +
                  json_escape(prefix) + ",\"out\":" + json_escape(path) + ",\"w\":" + std::to_string(im.W) + ",\"h\":" + std::to_string(im.H));
        }
    }

    ValPtr face_detailer(const std::string& nid);
};

// ---------------------------------------------------------------------------
// node implementations
// ---------------------------------------------------------------------------
std::vector<ValPtr> Run::exec(const std::string& nid, const std::string& ct) {
    auto mk = [](const char* t) { auto v = std::make_shared<Val>(); v->type = t; return v; };

    if (ct == "UNETLoader" || ct == "UnetLoaderGGUF") {
        auto v = mk("MODEL");
        std::string name = str(nid, "unet_name", "");
        std::string p = models() + "\\" + name;
        if (!exists(p)) {
            std::string alt = models() + "\\" + stem(name) + ".safetensors";
            p = exists(alt) ? alt : E.default_dit;  // GGUF names resolve to the safetensors Kiln runs
        }
        v->unet = p;
        return {v};
    }
    if (ct == "CLIPLoader") {
        find_model(str(nid, "clip_name", ""), {""}, {".safetensors"});
        return {mk("CLIP")};
    }
    if (ct == "VAELoader") {
        find_model(str(nid, "vae_name", ""), {""}, {".safetensors"});
        return {mk("VAE")};
    }
    if (ct == "LoraLoaderModelOnly" || ct == "LoraLoader") {
        auto v = std::make_shared<Val>(*link(nid, "model", "MODEL"));
        float s = (float)num(nid, "strength_model", 1.0);
        if (s != 0.f) v->loras.push_back({find_model(str(nid, "lora_name", ""), {"loras", ""}, {".safetensors"}), s});
        if (ct == "LoraLoader") return {v, link(nid, "clip", "CLIP")};
        return {v};
    }
    if (ct == "ModelSamplingAuraFlow" || ct == "ModelSamplingSD3") {
        auto v = std::make_shared<Val>(*link(nid, "model", "MODEL"));
        v->shift = (float)num(nid, "shift", ct == "ModelSamplingSD3" ? 3.0 : 1.73);
        v->mult = ct == "ModelSamplingSD3" ? 1000.f : 1.f;
        return {v};
    }
    if (ct == "ApplyFBCacheOnModel") {
        auto v = std::make_shared<Val>(*link(nid, "model", "MODEL"));
        v->cache = (float)num(nid, "residual_diff_threshold", 0.1);
        v->cache_start = (float)num(nid, "start", 0.15);
        v->cache_end = (float)num(nid, "end", 0.9);
        v->cache_max = (int)num(nid, "max_consecutive_cache_hits", 2);
        return {v};
    }
    if (ct == "UpscaleModelLoader") {
        auto v = mk("UPSCALE_MODEL");
        v->path = find_model(str(nid, "model_name", ""), {"upscale", ""}, {".safetensors"});
        return {v};
    }
    if (ct == "UltralyticsDetectorProvider") {
        auto v = mk("BBOX_DETECTOR");
        v->path = find_model(str(nid, "model_name", ""), {"detect"}, {".safetensors"});
        auto s = mk("SEGM_DETECTOR");
        s->path = v->path;
        return {v, s};
    }
    if (ct == "CLIPTextEncode") {
        link(nid, "clip", "CLIP");
        return {cond_from_tokens(tokens[nid])};
    }
    if (ct == "EmptySD3LatentImage" || ct == "EmptyLatentImage") {
        auto v = mk("LATENT");
        v->B = std::max(1, (int)num(nid, "batch_size", 1));
        v->C = 16;
        v->H = (int)num(nid, "height", 512) / 8;
        v->W = (int)num(nid, "width", 512) / 8;
        if (v->H < 1 || v->W < 1) fail("image size too small");
        v->data.assign((size_t)v->B * 16 * v->H * v->W, 0.f);
        return {v};
    }
    if (ct == "VAEEncode") {
        auto im = link(nid, "pixels", "IMAGE");
        link(nid, "vae", "VAE");
        return {vae_encode(*im)};
    }
    if (ct == "VAEDecode") {
        auto lat = link(nid, "samples", "LATENT");
        link(nid, "vae", "VAE");
        return {vae_decode(*lat)};
    }
    if (ct == "SetLatentNoiseMask") {
        auto v = std::make_shared<Val>(*link(nid, "samples", "LATENT"));
        v->noise_mask = link(nid, "mask", "MASK");
        return {v};
    }
    if (ct == "LatentUpscale" || ct == "LatentUpscaleBy" || ct == "ImageScale" || ct == "ImageScaleBy") {
        bool latent = ct.rfind("Latent", 0) == 0;
        auto src = latent ? link(nid, "samples", "LATENT") : link(nid, "image", "IMAGE");
        std::string method = str(nid, "upscale_method", "nearest-exact");
        int Ho, Wo;
        if (ct == "LatentUpscaleBy" || ct == "ImageScaleBy") {
            double s = num(nid, "scale_by", 1.0);
            Ho = (int)std::lround(src->H * s);
            Wo = (int)std::lround(src->W * s);
        } else {
            int w = (int)num(nid, "width", 512), h = (int)num(nid, "height", 512);
            if (latent) { w /= 8; h /= 8; }
            if (w == 0 && h == 0) return {src};
            Wo = w ? w : std::max(1, (int)std::lround((double)src->W * h / src->H));
            Ho = h ? h : std::max(1, (int)std::lround((double)src->H * w / src->W));
        }
        return {resize(*src, std::max(1, Wo), std::max(1, Ho), method, str(nid, "crop", "disabled") == "center")};
    }
    if (ct == "KSampler" || ct == "KSamplerAdvanced" || ct == "KSamplerWithNAG") {
        auto model = link(nid, "model", "MODEL");
        auto pos = link(nid, "positive", "CONDITIONING");
        auto neg = link(nid, "negative", "CONDITIONING");
        auto lat = link(nid, "latent_image", "LATENT");
        SampleParams sp;
        sp.steps = (int)num(nid, "steps", 20);
        sp.cfg = (float)num(nid, "cfg", 8.0);
        sp.sampler = str(nid, "sampler_name", "euler");
        sp.scheduler = str(nid, "scheduler", "simple");
        if (!known_sampler(sp.sampler)) fail("sampler '" + sp.sampler + "' is not available in Kiln (euler, euler_ancestral, dpmpp_2m, res_multistep, er_sde)");
        if (!known_scheduler(sp.scheduler)) fail("unknown scheduler '" + sp.scheduler + "'");
        ValPtr nag;
        if (ct == "KSamplerAdvanced") {
            sp.seed = seed(nid, "noise_seed");
            sp.add_noise = str(nid, "add_noise", "enable") == "enable";
            int start = (int)num(nid, "start_at_step", 0), end = (int)num(nid, "end_at_step", 10000);
            bool full = str(nid, "return_with_leftover_noise", "disable") != "enable";
            std::vector<float> s = schedule_sigmas(sp.scheduler, sp.steps, model->shift, 1.f);
            if (end < (int)s.size() - 1) {
                s.resize(end + 1);
                if (full) s.back() = 0.f;
            }
            if (start >= (int)s.size() - 1) return {lat};  // nothing left to do
            s.erase(s.begin(), s.begin() + start);
            sp.sigmas = s;
        } else {
            sp.seed = seed(nid, "seed");
            sp.denoise = (float)num(nid, "denoise", 1.0);
        }
        if (ct == "KSamplerWithNAG") {
            nag = link(nid, "nag_negative", "CONDITIONING");
            sp.nag_scale = (float)num(nid, "nag_scale", 5.0);
            sp.nag_tau = (float)num(nid, "nag_tau", 2.5);
            sp.nag_alpha = (float)num(nid, "nag_alpha", 0.25);
            sp.nag_sigma_end = (float)num(nid, "nag_sigma_end", 0.0);
        }
        return {run_sampler(nid, *model, *pos, neg.get(), nag.get(), *lat, sp)};
    }
    if (ct == "LoadImage") {
        const Json& im = images[nid];
        if (im.type != Json::Obj) fail("image was not uploaded / decoded by the server");
        int W = (int)im["w"].i64(), H = (int)im["h"].i64();
        std::ifstream f(im["path"].str(), std::ios::binary);
        std::vector<uint8_t> px((size_t)W * H * 4);
        if (!f.read((char*)px.data(), px.size())) fail("cannot read image " + im["path"].str());
        auto v = mk("IMAGE");
        v->B = 1; v->C = 3; v->H = H; v->W = W;
        size_t hw = (size_t)H * W;
        v->data.resize(3 * hw);
        bool alpha = false;
        for (size_t p = 0; p < hw; p++) {
            for (int c = 0; c < 3; c++) v->data[c * hw + p] = px[p * 4 + c] / 255.f;
            alpha |= px[p * 4 + 3] != 255;
        }
        auto mask = mk("MASK");
        if (alpha) {  // ComfyUI: mask = 1 - alpha; images without alpha give a 64x64 zero mask
            mask->B = 1; mask->H = H; mask->W = W;
            mask->data.resize(hw);
            for (size_t p = 0; p < hw; p++) mask->data[p] = 1.f - px[p * 4 + 3] / 255.f;
        } else {
            mask->B = 1; mask->H = 64; mask->W = 64;
            mask->data.assign(64 * 64, 0.f);
        }
        return {v, mask};
    }
    if (ct == "SaveImage" || ct == "PreviewImage") {
        auto im = link(nid, "images", "IMAGE");
        bool save = ct == "SaveImage";
        write_images(nid, *im, save ? "save" : "preview", save ? str(nid, "filename_prefix", "Kiln") : "");
        return {};
    }
    if (ct == "ImageUpscaleWithModel") {
        auto um = link(nid, "upscale_model", "UPSCALE_MODEL");
        auto im = link(nid, "image", "IMAGE");
        return {upscale(*um, *im)};
    }
    if (ext.has(ct)) return ext_node(nid, ct);
    if (ct == "FaceDetailer") {
        ValPtr img = face_detailer(nid);
        auto empty = std::make_shared<Val>();
        empty->type = "IMAGE"; empty->B = 1; empty->C = 3; empty->H = 64; empty->W = 64;
        empty->data.assign(3 * 64 * 64, 0.f);
        auto pipe = std::make_shared<Val>();
        pipe->type = "DETAILER_PIPE";
        return {img, empty, empty, done_mask, pipe, empty};
    }
    fail("unsupported node type " + ct);
}

// Impact Pack FaceDetailer semantics (bbox path): detect, crop bbox*crop_factor, enlarge so the bbox
// (or the crop) reaches guide_size within max_size, masked img2img with the noise mask, paste back
// through a feathered mask. `cycle` repeats the redraw.
ValPtr Run::face_detailer(const std::string& nid) {
#ifdef KILN_FACE
    auto src = link(nid, "image", "IMAGE");
    auto model = link(nid, "model", "MODEL");
    link(nid, "clip", "CLIP");
    link(nid, "vae", "VAE");
    auto pos = link(nid, "positive", "CONDITIONING");
    auto neg = link(nid, "negative", "CONDITIONING");
    auto det = link(nid, "bbox_detector", "BBOX_DETECTOR");
    float guide = (float)num(nid, "guide_size", 512), max_size = (float)num(nid, "max_size", 1024);
    bool for_bbox = flag(nid, "guide_size_for", true), force = flag(nid, "force_inpaint", true), use_mask = flag(nid, "noise_mask", true);
    float thr = (float)num(nid, "bbox_threshold", 0.5), crop_factor = (float)num(nid, "bbox_crop_factor", 3.0);
    int dil = (int)num(nid, "bbox_dilation", 10), feather = (int)num(nid, "feather", 5), drop = (int)num(nid, "drop_size", 10);
    int mask_feather = (int)num(nid, "noise_mask_feather", 20), cycles = std::max(1, (int)num(nid, "cycle", 1));
    SampleParams sp;
    sp.steps = (int)num(nid, "steps", 20);
    sp.cfg = (float)num(nid, "cfg", 8.0);
    sp.sampler = str(nid, "sampler_name", "euler");
    sp.scheduler = str(nid, "scheduler", "simple");
    sp.denoise = (float)num(nid, "denoise", 0.5);
    sp.seed = seed(nid, "seed");
    if (!known_sampler(sp.sampler)) fail("sampler '" + sp.sampler + "' is not available in Kiln");

    ensure_model(*model);
    Context cpos = E.condition(pos->cond), cneg;
    bool use_neg = sp.cfg > 1.f;
    if (use_neg) cneg = E.condition(neg->cond);
    sp.shift = model->shift;
    sp.timestep_mult = model->mult;
    FaceDetector& fd = E.face_model();
    (void)det;

    auto out = std::make_shared<Val>(*src);
    auto mask_out = std::make_shared<Val>();
    mask_out->type = "MASK"; mask_out->B = 1; mask_out->H = src->H; mask_out->W = src->W;
    mask_out->data.assign((size_t)src->H * src->W, 0.f);
    const int H = src->H, W = src->W;

    // soft rectangle: 1 inside [x0,x1)x[y0,y1), linear ramp over `f` pixels outside... inward like a blur
    auto rect_mask = [](int mw, int mh, float x0, float y0, float x1, float y1, float f) {
        std::vector<float> mk((size_t)mw * mh);
        for (int y = 0; y < mh; y++)
            for (int x = 0; x < mw; x++) {
                float px = x + 0.5f, py = y + 0.5f;
                float d = std::min(std::min(px - x0, x1 - px), std::min(py - y0, y1 - py));
                mk[(size_t)y * mw + x] = f <= 0.f ? (d > 0.f ? 1.f : 0.f) : std::min(1.f, std::max(0.f, d / f + 0.5f));
            }
        return mk;
    };

    for (int b = 0; b < src->B; b++) {
        size_t mb = G.arena.mark();
        float* img = image_to_device(*out, b);  // [-1,1]
        std::vector<Box> boxes;
        {
            size_t m2 = G.arena.mark();
            float* unit = G.arena.f((size_t)3 * H * W);
            rgb_to_unit(unit, img, (size_t)3 * H * W);
            boxes = fd.detect(unit, H, W, thr, 0.5f);
            G.arena.release(m2);
        }
        for (auto& bx : boxes) {
            float bw = bx.x1 - bx.x0, bh = bx.y1 - bx.y0;
            if (bw < drop || bh < drop) continue;
            float cx = (bx.x0 + bx.x1) / 2, cy = (bx.y0 + bx.y1) / 2;
            int x0 = std::max(0, (int)std::floor(cx - bw * crop_factor / 2)), x1 = std::min(W, (int)std::ceil(cx + bw * crop_factor / 2));
            int y0 = std::max(0, (int)std::floor(cy - bh * crop_factor / 2)), y1 = std::min(H, (int)std::ceil(cy + bh * crop_factor / 2));
            int w = x1 - x0, h = y1 - y0;
            if (w < 16 || h < 16) continue;
            float up = for_bbox ? guide / std::min(bw, bh) : guide / std::min(w, h);
            float nw = w * up, nh = h * up;
            if (nw > max_size || nh > max_size) up *= max_size / std::max(nw, nh);
            if (up <= 1.f) {
                if (!force) continue;
                up = 1.f;
            }
            int tw = std::max(16, (int)std::lround(w * up / 16.f) * 16), th = std::max(16, (int)std::lround(h * up / 16.f) * 16);
            int tl = (th / 8) * (tw / 8);

            // masks: the dilated bbox, feathered (noise mask in latent space, paste mask in pixels)
            float rx0 = bx.x0 - x0 - dil, rx1 = bx.x1 - x0 + dil, ry0 = bx.y0 - y0 - dil, ry1 = bx.y1 - y0 + dil;
            auto pm = rect_mask(w, h, rx0, ry0, rx1, ry1, (float)feather);
            float sx = (tw / 8.f) / w, sy = (th / 8.f) / h;
            auto lm = rect_mask(tw / 8, th / 8, rx0 * sx, ry0 * sy, rx1 * sx, ry1 * sy, mask_feather * sx);

            size_t m3 = G.arena.mark();
            float* crop = G.arena.f((size_t)3 * w * h);
            float* big = G.arena.f((size_t)3 * tw * th);
            float* z = G.arena.f((size_t)16 * tl);
            float* xz = G.arena.f((size_t)16 * tl);
            float* lmask = G.arena.f(tl);
            float* pmask = G.arena.f((size_t)w * h);
            CK(cudaMemcpy(lmask, lm.data(), lm.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(pmask, pm.data(), pm.size() * 4, cudaMemcpyHostToDevice));
            crop_rect(crop, img, 3, H, W, x0, y0, w, h);
            resize_lanczos(big, crop, 3, h, w, th, tw);
            for (int cyc = 0; cyc < cycles; cyc++) {  // each cycle redraws the previous cycle's result
                E.vae.encode(big, th, tw, z);
                latent_vae_to_model(z, tl);
                SampleParams spc = sp;
                spc.seed = sp.seed + cyc;
                sample(E.dit, xz, th / 8, tw / 8, cpos, use_neg ? &cneg : nullptr, spc, z, use_mask ? lmask : nullptr, stepper(nid, th / 8, tw / 8), &E.cancel);
                latent_model_to_vae(xz, tl);
                E.vae.decode(xz, th / 8, tw / 8, big);
            }
            resize_lanczos(crop, big, 3, th, tw, h, w);
            blend_rect(img, 3, H, W, crop, pmask, x0, y0, w, h);
            G.arena.release(m3);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    float& d = mask_out->data[(size_t)(y0 + y) * W + x0 + x];
                    d = std::max(d, pm[(size_t)y * w + x]);
                }
        }
        image_from_device(*out, b, img);
        G.arena.release(mb);
    }
    done_mask = mask_out;
    return out;
#else
    fail("this engine build has no face detector");
#endif
}

}  // namespace

void run_graph(Engine& E, const Json& req) {
    Run r(E, req);
    std::string id = r.id;
    auto t0 = Clock::now();
    size_t m0 = G.arena.mark();
    try {
        // fail fast on anything we can't run, before touching the GPU
        static const std::set<std::string> known = {
            "UNETLoader", "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "LoraLoader", "UpscaleModelLoader",
            "UltralyticsDetectorProvider", "ModelSamplingAuraFlow", "ModelSamplingSD3", "ApplyFBCacheOnModel", "CLIPTextEncode",
            "EmptySD3LatentImage", "EmptyLatentImage", "VAEEncode", "VAEDecode", "SetLatentNoiseMask", "LatentUpscale", "LatentUpscaleBy",
            "KSampler", "KSamplerAdvanced", "KSamplerWithNAG", "LoadImage", "SaveImage", "PreviewImage", "ImageScale", "ImageScaleBy",
            "ImageUpscaleWithModel", "FaceDetailer"};
        std::vector<std::string> outputs;
        for (auto& [nid, n] : r.graph.o) {
            std::string ct = n["class_type"].str();
            bool pack = r.ext.has(ct);
            if (!known.count(ct) && !pack) throw NodeError(nid, "unsupported node type " + ct);
            if (ct == "SaveImage" || ct == "PreviewImage" || (pack && r.ext[ct]["output_node"].b)) outputs.push_back(nid);
        }
        if (outputs.empty()) throw NodeError("", "the graph has no output node (SaveImage, PreviewImage or a pack output node)");
        E.cancel = false;
        {
            std::lock_guard<std::mutex> lk(E.ext_inbox.mu);
            E.ext_inbox.q.clear();  // replies meant for an earlier, cancelled graph
        }
        E.use_family("anima");  // the graph nodes run Anima
        E.prepare_job();
        for (auto& o : outputs) r.eval(o);
        std::ostringstream nm;
        bool first = true;
        for (auto& [nid, ms] : r.node_ms) { nm << (first ? "" : ",") << json_escape(nid) << ":" << (int)ms; first = false; }
        r.event("\"ev\":\"done\",\"total_ms\":" + std::to_string((int)ms_since(t0)) + ",\"node_ms\":{" + nm.str() + "}");
    } catch (const NodeError& e) {
        E.dit.drop_context_cache();
        cudaStreamSynchronize(G.stream);
        cudaGetLastError();
        if (E.cancel) r.event("\"ev\":\"cancelled\"");
        else r.event("\"ev\":\"error\",\"node\":" + (e.node.empty() ? std::string("null") : json_escape(e.node)) + ",\"msg\":" + json_escape(e.what()));
    } catch (const std::exception& e) {
        E.dit.drop_context_cache();
        cudaStreamSynchronize(G.stream);
        cudaGetLastError();
        if (E.cancel) r.event("\"ev\":\"cancelled\"");
        else r.event("\"ev\":\"error\",\"node\":" + (r.current.empty() ? std::string("null") : json_escape(r.current)) + ",\"msg\":" + json_escape(e.what()));
    }
    G.arena.release(m0);
}
