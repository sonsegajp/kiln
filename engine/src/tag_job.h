// The "tag" command: booru tags for a list of images with a WD tagger (tagger.h).
//   {"cmd":"tag","id":..,"model":"<folder with model.safetensors, config.json>","images":[path, ...],"floor":0.05}
// Events: tagged (index, probs [[tag index, p], ...] for p >= floor) or tag_failed (index, msg) per image, then done.
// The tagger loads for the job and is freed after it, so it never holds VRAM a render wants.
#pragma once
#include <future>

#include "engine.h"
#include "tagger.h"

namespace tagjob {

inline int run(Engine& E, const Json& r, const std::string& id) {
    auto t0 = Clock::now();
    const std::string dir = r["model"].str();
    const float floor = (float)r["floor"].num(0.05);
    auto ev = [&](const std::string& body) { emit("{\"id\":" + json_escape(id) + "," + body + "}"); };
    ev("\"ev\":\"loading\",\"what\":\"tagger\"");
    Tagger tg;
    struct Free { Tagger& t; ~Free() { t.free(); } } free_tg{tg};
    tg.load(dir, Place::Auto);
    log_msg("tagger " + dir + " loaded in " + std::to_string((int)ms_since(t0)) + " ms");
    const auto& list = r["images"].a;
    // the next image is read and resized on a worker thread while the GPU tags this one
    struct Pre { bool ok = false; std::vector<float> img; std::string err; };
    auto prep = [&](size_t i) { Pre p; p.ok = tagger_preprocess(list[i].str(), tg.size(), p.img, p.err); return p; };
    std::future<Pre> next;
    if (!list.empty()) next = std::async(std::launch::async, prep, (size_t)0);
    std::vector<float> probs;
    for (size_t i = 0; i < list.size(); i++) {
        Pre p = next.get();
        if (i + 1 < list.size()) next = std::async(std::launch::async, prep, i + 1);
        if (E.cancel) { ev("\"ev\":\"cancelled\""); return 0; }
        if (!p.ok) {
            ev("\"ev\":\"tag_failed\",\"index\":" + std::to_string(i) + ",\"msg\":" + json_escape(p.err));
            continue;
        }
        const size_t m = G.arena.mark();
        float* d = G.arena.f(p.img.size());
        CK(cudaMemcpy(d, p.img.data(), p.img.size() * 4, cudaMemcpyHostToDevice));
        tg.run(d, probs);
        G.arena.release(m);
        std::string s = "\"ev\":\"tagged\",\"index\":" + std::to_string(i) + ",\"probs\":[";
        bool first = true;
        char b[48];
        for (size_t k = 0; k < probs.size(); k++) {
            if (probs[k] < floor) continue;
            snprintf(b, sizeof b, "%s[%zu,%.4f]", first ? "" : ",", k, probs[k]);
            s += b;
            first = false;
        }
        ev(s + "]");
    }
    ev("\"ev\":\"done\",\"total_ms\":" + std::to_string((int)ms_since(t0)));
    return 0;
}

}  // namespace tagjob
