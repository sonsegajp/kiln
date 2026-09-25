// Standalone harness for engine/src/upscale.cu (RRDBNet / ESRGAN upscaler).
//
//   upscale_test.exe <model.safetensors> <in.ppm> <out.png> [options]
//     --arena MB      scratch arena (default 256); input/output images are carved from it too
//     --reserve MB    G.reserve_bytes for Auto weight placement (default 64)
//     --halo N        tile halo in LR px (default: Upscaler's)
//     --tile N        cap the tile core side (forces tiling)
//     --fp32          fp32 direct convs (G.fp16 = false, like the engine's --fp32); default is fp16x2
//     --ref           reference path (im2col + fp32 cuBLAS) instead of the direct convs
//     --bf16          round weights through bf16 (what upload_weight would do)
//     --vs-ref        also run the reference path and report the difference
//     --sweep T:h1,h2,...  rerun tiled with core <= T and each halo; compare with the first (untiled) run
//     --dump f.bin    raw float32 [3, sH, sW] of the output
//     --reps N        timed warm runs (default 1)
//     --profile       per-op GPU time breakdown of one warm run
#include "../engine/src/upscale.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static bool read_ppm(const char* path, int& w, int& h, std::vector<uint8_t>& rgb) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    int maxv;
    f >> magic;
    auto skip = [&] { while (f.peek() == '#' || isspace(f.peek())) { if (f.peek() == '#') { std::string l; std::getline(f, l); } else f.get(); } };
    skip(); f >> w; skip(); f >> h; skip(); f >> maxv;
    f.get();
    if (magic != "P6" || maxv != 255) return false;
    rgb.resize((size_t)w * h * 3);
    f.read((char*)rgb.data(), rgb.size());
    return (bool)f;
}

static uint32_t crc_tab[256];
static uint32_t crc(const uint8_t* p, size_t n, uint32_t c) {
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c;
}
// PNG with stored (uncompressed) deflate blocks: no zlib needed.
static void write_png(const char* path, int w, int h, const uint8_t* rgb) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_tab[i] = c;
    }
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (w * 3 + 1));
    for (int y = 0; y < h; y++) { raw.push_back(0); raw.insert(raw.end(), rgb + (size_t)y * w * 3, rgb + (size_t)(y + 1) * w * 3); }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t o = 0; o < raw.size() || o == 0;) {
        size_t n = std::min<size_t>(65535, raw.size() - o);
        z.push_back(o + n == raw.size() ? 1 : 0);
        z.push_back(n & 0xff); z.push_back(n >> 8); z.push_back(~n & 0xff); z.push_back((~n >> 8) & 0xff);
        z.insert(z.end(), raw.begin() + o, raw.begin() + o + n);
        o += n;
        if (n == 0) break;
    }
    uint32_t ad = (b << 16) | a;
    for (int i = 3; i >= 0; i--) z.push_back((ad >> (8 * i)) & 0xff);
    FILE* f = fopen(path, "wb");
    if (!f) { printf("cannot write %s\n", path); return; }
    auto be32 = [&](uint32_t v) { uint8_t q[4] = {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)}; fwrite(q, 1, 4, f); };
    auto chunk = [&](const char* type, const uint8_t* d, size_t n) {
        be32((uint32_t)n);
        fwrite(type, 1, 4, f);
        if (n) fwrite(d, 1, n, f);
        uint32_t c = crc((const uint8_t*)type, 4, 0xffffffffu);
        c = crc(d, n, c);
        be32(c ^ 0xffffffffu);
    };
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13] = {uint8_t(w >> 24), uint8_t(w >> 16), uint8_t(w >> 8), uint8_t(w), uint8_t(h >> 24), uint8_t(h >> 16), uint8_t(h >> 8), uint8_t(h), 8, 2, 0, 0, 0};
    chunk("IHDR", ihdr, 13);
    chunk("IDAT", z.data(), z.size());
    chunk("IEND", nullptr, 0);
    fclose(f);
}

struct Diff { double rel_l2, max_abs; size_t q8_diff; };
static Diff diff(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0, den = 0, mx = 0;
    size_t q = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        mx = std::max(mx, std::fabs(d));
        q += lrintf(a[i] * 255.f) != lrintf(ref[i] * 255.f);
    }
    return {std::sqrt(num / std::max(den, 1e-30)), mx, q};
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: upscale_test <model> <in.ppm> <out.png> [options]\n"); return 1; }
    const char *model = argv[1], *inp = argv[2], *outp = argv[3];
    size_t arena_mb = 256, reserve_mb = 64;
    int halo = -1, tile = 0, reps = 1, sweep_tile = 0;
    bool ref = false, bf16 = false, vs_ref = false, profile = false;
    std::string dump, sweep;
    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--arena") arena_mb = std::stoul(next());
        else if (a == "--reserve") reserve_mb = std::stoul(next());
        else if (a == "--halo") halo = std::stoi(next());
        else if (a == "--tile") tile = std::stoi(next());
        else if (a == "--ref") ref = true;
        else if (a == "--fp32") G.fp16 = false;
        else if (a == "--bf16") bf16 = true;
        else if (a == "--vs-ref") vs_ref = true;
        else if (a == "--dump") dump = next();
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--profile") profile = true;
        else if (a == "--sweep") { sweep = next(); sweep_tile = std::stoi(sweep.substr(0, sweep.find(':'))); sweep = sweep.substr(sweep.find(':') + 1); }
        else { printf("unknown option %s\n", a.c_str()); return 1; }
    }
    try {
        // same bring-up as engine/src/main.cpp: device, reserve, weights, then the scratch arena
        gpu_init(0);
        G.reserve_bytes = reserve_mb << 20;
        printf("VRAM free at start: %zu MB\n", gpu_free_bytes() >> 20);
        Upscaler up;
        up.bf16_weights = bf16;
        up.reference = ref;
        if (halo >= 0) up.halo = halo;
        up.max_tile = tile;
        auto t0 = Clock::now();
        up.load(model, Place::Auto);
        printf("loaded %s in %.0f ms: nf=%d gc=%d nb=%d scale=%d%s\n", model, ms_since(t0), up.nf, up.gc, up.nb, up.scale(), bf16 ? " (bf16-rounded weights)" : "");
        gpu_alloc_scratch(1 << 16, arena_mb << 20);
        printf("VRAM free after arena (%zu MB): %zu MB\n", arena_mb, gpu_free_bytes() >> 20);

        int W, H;
        std::vector<uint8_t> rgb;
        if (!read_ppm(inp, W, H, rgb)) throw std::runtime_error(std::string("cannot read PPM ") + inp);
        const int s = up.scale(), OH = H * s, OW = W * s;
        std::vector<float> hin((size_t)3 * H * W);
        for (int c = 0; c < 3; c++)
            for (size_t p = 0; p < (size_t)H * W; p++) hin[c * (size_t)H * W + p] = rgb[p * 3 + c] / 255.f;
        float* din = G.arena.f(hin.size());
        float* dout = G.arena.f((size_t)3 * OH * OW);
        CK(cudaMemcpy(din, hin.data(), hin.size() * 4, cudaMemcpyHostToDevice));
        printf("input %dx%d -> %dx%d, scratch left for run(): %zu MB\n", W, H, OW, OH, G.arena.free_bytes() >> 20);

        auto timed = [&](const char* label) {
            auto t = Clock::now();
            up.run(dout, din, H, W);
            gpu_sync();
            double ms = ms_since(t);
            printf("%-6s %8.1f ms  (%d tile%s, halo %d, arena peak %zu MB)\n", label, ms, up.tiles_used, up.tiles_used == 1 ? "" : "s", up.halo, G.arena.peak >> 20);
            return ms;
        };
        timed("cold");
        double best = 1e30;
        for (int r = 0; r < reps; r++) best = std::min(best, timed("warm"));
        if (profile) {
            prof_enable(true);
            up.run(dout, din, H, W);
            printf("%s", prof_report().c_str());
            prof_enable(false);
        }
        std::vector<float> hout((size_t)3 * OH * OW);
        CK(cudaMemcpy(hout.data(), dout, hout.size() * 4, cudaMemcpyDeviceToHost));

        if (vs_ref) {
            up.reference = true;
            up.run(dout, din, H, W);
            std::vector<float> r(hout.size());
            CK(cudaMemcpy(r.data(), dout, r.size() * 4, cudaMemcpyDeviceToHost));
            Diff d = diff(hout, r);
            printf("vs reference path: rel_l2 %.3e  max_abs %.3e  8-bit values differing %zu / %zu\n", d.rel_l2, d.max_abs, d.q8_diff, r.size());
            up.reference = ref;
        }
        if (!sweep.empty()) {
            std::stringstream ss(sweep);
            std::string tok;
            int t_saved = up.max_tile, h_saved = up.halo;
            while (std::getline(ss, tok, ',')) {
                up.max_tile = sweep_tile;
                up.halo = std::stoi(tok);
                auto t = Clock::now();
                up.run(dout, din, H, W);
                gpu_sync();
                double ms = ms_since(t);
                std::vector<float> r(hout.size());
                CK(cudaMemcpy(r.data(), dout, r.size() * 4, cudaMemcpyDeviceToHost));
                Diff d = diff(r, hout);
                printf("tiled core<=%d halo %3d (%2d tiles, %7.1f ms) vs untiled: rel_l2 %.3e  max_abs %.3e  8-bit values differing %zu / %zu\n",
                       sweep_tile, up.halo, up.tiles_used, ms, d.rel_l2, d.max_abs, d.q8_diff, r.size());
            }
            up.max_tile = t_saved; up.halo = h_saved;
        }

        if (!dump.empty()) {
            FILE* f = fopen(dump.c_str(), "wb");
            fwrite(hout.data(), 4, hout.size(), f);
            fclose(f);
        }
        std::vector<uint8_t> o8((size_t)OH * OW * 3);
        for (int c = 0; c < 3; c++)
            for (size_t p = 0; p < (size_t)OH * OW; p++) o8[p * 3 + c] = (uint8_t)lrintf(hout[c * (size_t)OH * OW + p] * 255.f);
        write_png(outp, OW, OH, o8.data());
        printf("wrote %s; best warm run %.1f ms\n", outp, best);
    } catch (const std::exception& e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
