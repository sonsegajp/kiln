// VAE harness: accuracy (vs the ComfyUI golden dumps and fp16 vs fp32), band exactness, activation
// ranges and timing. Loads only the VAE.
//   vae_test.exe [--models DIR] [--arena-mb 160] [--runs 5] [--prec fp16|fp32|both] [--device-weights] <command> ...
//   (keep the arena small while the Kiln server runs: this process shares the 6 GB card with it; weights
//   default to mapped host RAM, --device-weights puts them in VRAM (~250 MB) for timing)
//     golden DIR           decode DIR/vae_in.npy vs DIR/image.npy; encode DIR/image.npy and decode back
//     band DIR ROWS        banded (ROWS latent rows) vs unbanded decode + encode, both precisions
//     time W H             decode + encode timing at W x H (min of --runs), both precisions
//     prof W H             per-op GPU time (ProfScope categories) of one decode + encode, both precisions
//     probe DIR|PNG ...    max |activation| per tensor over all inputs (encode + decode of each), both precisions
//     png FILE.png ...     encode + decode renders: fp16 vs fp32 latents / images
//     save DIR OUT.png     decode DIR/vae_in.npy with fp16 and write a PNG (visual check)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "models.h"

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------
static std::vector<float> load_npy(const std::string& path, std::vector<int>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    uint16_t hl = 0;
    if (magic[6] == 1) f.read((char*)&hl, 2);
    else { uint32_t h4; f.read((char*)&h4, 4); hl = (uint16_t)h4; }
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": need C-order float32");
    size_t a = hdr.find('(', hdr.find("'shape'")), b = hdr.find(')', a);
    std::string sh = hdr.substr(a + 1, b - a - 1);
    shape.clear();
    size_t n = 1;
    for (size_t i = 0; i < sh.size();) {
        while (i < sh.size() && !isdigit((unsigned char)sh[i])) i++;
        if (i >= sh.size()) break;
        int v = 0;
        while (i < sh.size() && isdigit((unsigned char)sh[i])) v = v * 10 + (sh[i++] - '0');
        shape.push_back(v);
        n *= v;
    }
    std::vector<float> d(n);
    f.read((char*)d.data(), n * 4);
    return d;
}

static IWICImagingFactory* wic() {
    static IWICImagingFactory* f = nullptr;
    if (!f) {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f))))
            throw std::runtime_error("WIC unavailable");
    }
    return f;
}
static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(n - 1);
    return w;
}
#define HR(x) do { HRESULT h_ = (x); if (FAILED(h_)) { char b_[96]; snprintf(b_, sizeof b_, "WIC 0x%08lx at line %d", (unsigned long)h_, __LINE__); throw std::runtime_error(b_); } } while (0)

// PNG -> [3, H, W] in [-1, 1], cropped to multiples of 8
static std::vector<float> load_png(const std::string& path, int& W, int& H) {
    IWICBitmapDecoder* dec = nullptr;
    IWICBitmapFrameDecode* fr = nullptr;
    IWICBitmapSource* cv = nullptr;
    HR(wic()->CreateDecoderFromFilename(widen(path).c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec));
    HR(dec->GetFrame(0, &fr));
    HR(WICConvertBitmapSource(GUID_WICPixelFormat32bppBGRA, fr, &cv));
    UINT w, h;
    HR(cv->GetSize(&w, &h));
    std::vector<uint8_t> bgra((size_t)w * h * 4);
    HR(cv->CopyPixels(nullptr, w * 4, (UINT)bgra.size(), bgra.data()));
    cv->Release(); fr->Release(); dec->Release();
    W = (int)w / 8 * 8; H = (int)h / 8 * 8;
    std::vector<float> out((size_t)3 * W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) out[((size_t)c * H + y) * W + x] = bgra[((size_t)y * w + x) * 4 + (2 - c)] / 127.5f - 1.f;
    return out;
}

static void save_png(const std::string& path, const std::vector<float>& chw, int W, int H) {
    std::vector<uint8_t> bgr((size_t)W * H * 3);
    for (size_t i = 0; i < (size_t)W * H; i++)
        for (int c = 0; c < 3; c++) {
            float v = std::min(1.f, std::max(0.f, (chw[(size_t)c * W * H + i] + 1.f) / 2.f));
            bgr[i * 3 + (2 - c)] = (uint8_t)std::lround(v * 255.f);
        }
    IWICStream* s = nullptr;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* fr = nullptr;
    IPropertyBag2* props = nullptr;
    HR(wic()->CreateStream(&s));
    HR(s->InitializeFromFilename(widen(path).c_str(), GENERIC_WRITE));
    HR(wic()->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc));
    HR(enc->Initialize(s, WICBitmapEncoderNoCache));
    HR(enc->CreateNewFrame(&fr, &props));
    HR(fr->Initialize(props));
    HR(fr->SetSize(W, H));
    WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
    HR(fr->SetPixelFormat(&fmt));
    HR(fr->WritePixels(H, W * 3, (UINT)bgr.size(), bgr.data()));
    HR(fr->Commit());
    HR(enc->Commit());
    if (props) props->Release();
    fr->Release(); enc->Release(); s->Release();
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------
struct Cmp { double rel_l2, max_abs, psnr, frac_1lsb; };  // frac_1lsb: share of values off by > 1/255
// a, b: same layout; values mapped through f (e.g. [-1,1] -> [0,1] clamp) before comparing; psnr for peak 1
template <class F>
static Cmp compare(const std::vector<float>& a, const std::vector<float>& b, F f) {
    double se = 0, sb = 0, mx = 0;
    size_t n1 = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double x = f(a[i]), y = f(b[i]), d = x - y;
        se += d * d; sb += y * y; mx = std::max(mx, std::fabs(d));
        n1 += std::fabs(d) > 1.0 / 255;
    }
    double mse = se / a.size();
    return {std::sqrt(se / std::max(sb, 1e-30)), mx, mse > 0 ? 10 * std::log10(1.0 / mse) : INFINITY, (double)n1 / a.size()};
}
static float unit(float v) { return std::min(1.f, std::max(0.f, (v + 1.f) / 2.f)); }
static float ident(float v) { return v; }
static void print_cmp(const char* what, const Cmp& c) {
    printf("  %-38s rel_l2 %.3e  max_abs %.3e  PSNR %6.2f dB  >1/255: %.4f%%\n", what, c.rel_l2, c.max_abs, c.psnr, 100 * c.frac_1lsb);
}

// golden image [H, W, 3] in [0,1] -> [3, H, W] in [-1, 1]
static std::vector<float> hwc01_to_chw(const std::vector<float>& img, int H, int W) {
    std::vector<float> o((size_t)3 * H * W);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) o[((size_t)c * H + y) * W + x] = img[((size_t)y * W + x) * 3 + c] * 2.f - 1.f;
    return o;
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------
static Vae V;
using Prec = Vae::Precision;

static float* dev(const std::vector<float>& h) {
    float* d = G.arena.f(h.size());
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}
static std::vector<float> host(const float* d, size_t n) {
    gpu_sync();
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<float> decode(const std::vector<float>& z, int Hl, int Wl, Prec p, int band = 0) {
    size_t m = G.arena.mark();
    float* dz = dev(z);
    float* out = G.arena.f((size_t)3 * 64 * Hl * Wl);
    V.precision = p;
    V.band_rows = band;
    V.decode(dz, Hl, Wl, out);
    auto r = host(out, (size_t)3 * 64 * Hl * Wl);
    V.band_rows = 0;
    G.arena.release(m);
    return r;
}
static std::vector<float> encode(const std::vector<float>& rgb, int H, int W, Prec p, int band = 0) {
    size_t m = G.arena.mark();
    float* di = dev(rgb);
    float* z = G.arena.f((size_t)16 * (H / 8) * (W / 8));
    V.precision = p;
    V.band_rows = band;
    V.encode(di, H, W, z);
    auto r = host(z, (size_t)16 * (H / 8) * (W / 8));
    V.band_rows = 0;
    G.arena.release(m);
    return r;
}

static const char* pname(Prec p) { return p == Prec::FP16 ? "fp16" : "fp32"; }
static std::vector<Prec> g_precs = {Prec::FP32, Prec::FP16};  // --prec fp16|fp32|both
static bool has(Prec p) { return std::find(g_precs.begin(), g_precs.end(), p) != g_precs.end(); }

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------
static void cmd_golden(const std::string& dir) {
    std::vector<int> zs, is;
    auto z = load_npy(dir + "/vae_in.npy", zs);
    auto img = load_npy(dir + "/image.npy", is);
    int Hl = zs[1], Wl = zs[2], H = is[0], W = is[1];
    printf("%s: latent [16,%d,%d] -> image %dx%d\n", dir.c_str(), Hl, Wl, W, H);
    auto ref = hwc01_to_chw(img, H, W);
    const bool b32 = has(Prec::FP32), b16 = has(Prec::FP16);
    std::vector<float> d32, d16, e32, e16, r32, r16;
    if (b32) d32 = decode(z, Hl, Wl, Prec::FP32);
    if (b16) d16 = decode(z, Hl, Wl, Prec::FP16);
    printf(" decode (image space [0,1]):\n");
    if (b32) print_cmp("fp32 vs ComfyUI", compare(d32, ref, unit));
    if (b16) print_cmp("fp16 vs ComfyUI", compare(d16, ref, unit));
    if (b32 && b16) print_cmp("fp16 vs fp32", compare(d16, d32, unit));
    if (b32) e32 = encode(ref, H, W, Prec::FP32);
    if (b16) e16 = encode(ref, H, W, Prec::FP16);
    printf(" encode:\n");
    if (b32 && b16) print_cmp("latent fp16 vs fp32 (raw)", compare(e16, e32, ident));
    if (b32) r32 = decode(e32, Hl, Wl, Prec::FP32);
    if (b16) r16 = decode(e16, Hl, Wl, Prec::FP16);
    if (b32) print_cmp("roundtrip fp32 (enc+dec) vs image", compare(r32, ref, unit));
    if (b16) print_cmp("roundtrip fp16 (enc+dec) vs image", compare(r16, ref, unit));
    if (b32 && b16) print_cmp("roundtrip fp16 vs roundtrip fp32", compare(r16, r32, unit));
}

static void cmd_band(const std::string& dir, int rows) {
    std::vector<int> zs, is;
    auto z = load_npy(dir + "/vae_in.npy", zs);
    auto img = load_npy(dir + "/image.npy", is);
    int Hl = zs[1], Wl = zs[2], H = is[0], W = is[1];
    auto ref = hwc01_to_chw(img, H, W);
    for (Prec p : g_precs) {
        auto a = decode(z, Hl, Wl, p), b = decode(z, Hl, Wl, p, rows);
        double md = 0;
        for (size_t i = 0; i < a.size(); i++) md = std::max(md, (double)std::fabs(a[i] - b[i]));
        auto ea = encode(ref, H, W, p), eb = encode(ref, H, W, p, rows);
        double me = 0;
        for (size_t i = 0; i < ea.size(); i++) me = std::max(me, (double)std::fabs(ea[i] - eb[i]));
        printf(" %s  band %d rows: decode max|banded - unbanded| = %g, encode = %g\n", pname(p), rows, md, me);
    }
}

static void cmd_time(int W, int H, int runs) {
    int Hl = H / 8, Wl = W / 8;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> z((size_t)16 * Hl * Wl);
    for (auto& v : z) v = nd(rng) * 2.f;
    size_t m = G.arena.mark();
    float* dz = dev(z);
    float* img = G.arena.f((size_t)3 * H * W);
    float* z2 = G.arena.f(z.size());
    printf("%dx%d (latent %dx%d), min / median of %d runs, arena %zu MB:\n", W, H, Wl, Hl, runs, G.arena.cap >> 20);
    for (Prec p : g_precs) {
        V.precision = p;
        std::vector<double> td, te;
        for (int i = 0; i < runs + 1; i++) {
            gpu_sync();
            auto t0 = Clock::now();
            V.decode(dz, Hl, Wl, img);
            gpu_sync();
            auto t1 = Clock::now();
            V.encode(img, H, W, z2);
            gpu_sync();
            auto t2 = Clock::now();
            if (i == 0) continue;  // warm-up
            td.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            te.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(td.begin(), td.end());
        std::sort(te.begin(), te.end());
        printf("  %s  decode %7.1f / %7.1f ms   encode %7.1f / %7.1f ms   (arena peak %zu MB)\n", pname(p), td[0], td[td.size() / 2], te[0],
               te[te.size() / 2], G.arena.peak >> 20);
    }
    G.arena.release(m);
}

// per-op GPU time (ProfScope categories) of one decode and one encode at W x H
static void cmd_prof(int W, int H) {
    int Hl = H / 8, Wl = W / 8;
    std::vector<float> z((size_t)16 * Hl * Wl, 0.5f);
    size_t m = G.arena.mark();
    float* dz = dev(z);
    float* img = G.arena.f((size_t)3 * H * W);
    float* z2 = G.arena.f(z.size());
    for (Prec p : g_precs) {
        V.precision = p;
        V.decode(dz, Hl, Wl, img);  // warm-up
        prof_enable(true);
        V.decode(dz, Hl, Wl, img);
        printf("%s decode %dx%d:\n%s", pname(p), W, H, prof_report().c_str());
        prof_enable(true);
        V.encode(img, H, W, z2);
        printf("%s encode %dx%d:\n%s", pname(p), W, H, prof_report().c_str());
        prof_enable(false);
    }
    G.arena.release(m);
}

static void cmd_probe(const std::vector<std::string>& inputs) {
    std::map<std::string, float> mx[2];
    std::vector<std::string> order;
    std::vector<std::pair<std::string, float>> rec;
    V.probe = &rec;
    for (auto& in : inputs) {
        std::vector<float> rgb;
        int H, W;
        if (in.size() > 4 && in.substr(in.size() - 4) == ".png") {
            rgb = load_png(in, W, H);
        } else {
            std::vector<int> is;
            auto img = load_npy(in + "/image.npy", is);
            H = is[0]; W = is[1];
            rgb = hwc01_to_chw(img, H, W);
        }
        printf("  %s (%dx%d)\n", in.c_str(), W, H);
        for (int pi = 0; pi < 2; pi++) {
            Prec p = pi ? Prec::FP16 : Prec::FP32;
            if (!has(p)) continue;
            rec.clear();
            auto z = encode(rgb, H, W, p);
            decode(z, H / 8, W / 8, p);
            if (!in.empty() && in.substr(in.size() - 4) != ".png") {  // golden: also the sampled latent
                std::vector<int> zs;
                auto zg = load_npy(in + "/vae_in.npy", zs);
                decode(zg, zs[1], zs[2], p);
            }
            for (auto& [n, v] : rec) {
                if (!mx[pi].count(n)) {
                    mx[pi][n] = 0;
                    if (std::find(order.begin(), order.end(), n) == order.end()) order.push_back(n);
                }
                mx[pi][n] = std::max(mx[pi][n], v);
            }
        }
    }
    V.probe = nullptr;
    printf("%-26s %12s %12s\n", "tensor", "fp32 max|x|", "fp16 max|x|");
    for (auto& n : order)  // -1: tensor not recorded by that path
        printf("%-26s %12.2f %12.2f\n", n.c_str(), mx[0].count(n) ? mx[0][n] : -1.f, mx[1].count(n) ? mx[1][n] : -1.f);
}

static void cmd_png(const std::vector<std::string>& files) {
    for (auto& f : files) {
        int W, H;
        auto rgb = load_png(f, W, H);
        printf("%s (%dx%d)\n", f.c_str(), W, H);
        auto e32 = encode(rgb, H, W, Prec::FP32), e16 = encode(rgb, H, W, Prec::FP16);
        print_cmp("latent fp16 vs fp32 (raw)", compare(e16, e32, ident));
        auto d32 = decode(e32, H / 8, W / 8, Prec::FP32), d16 = decode(e32, H / 8, W / 8, Prec::FP16);
        print_cmp("decode(fp32 latent): fp16 vs fp32", compare(d16, d32, unit));
        auto r16 = decode(e16, H / 8, W / 8, Prec::FP16);
        print_cmp("roundtrip fp32 vs input", compare(d32, rgb, unit));
        print_cmp("roundtrip fp16 vs input", compare(r16, rgb, unit));
        print_cmp("roundtrip fp16 vs roundtrip fp32", compare(r16, d32, unit));
    }
}

int main(int argc, char** argv) {
    std::string models = "..\\models";
    size_t arena_mb = 160;
    int runs = 5;
    bool device_weights = false;  // default: weights in mapped host RAM (slower, ~no VRAM); timing wants --device-weights
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) models = argv[++i];
        else if (a == "--arena-mb" && i + 1 < argc) arena_mb = std::stoul(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
        else if (a == "--device-weights") device_weights = true;
        else if (a == "--prec" && i + 1 < argc) {
            std::string v = argv[++i];
            g_precs = v == "fp16" ? std::vector<Prec>{Prec::FP16} : v == "fp32" ? std::vector<Prec>{Prec::FP32} : std::vector<Prec>{Prec::FP32, Prec::FP16};
        }
        else args.push_back(a);
    }
    if (args.empty()) { fprintf(stderr, "usage: see the header of bench/vae_test.cu\n"); return 2; }
    try {
        gpu_init(0);
        G.reserve_bytes = 0;
        V.load(models + "\\qwen_image_vae.safetensors", device_weights ? Place::Device : Place::Host);
        gpu_alloc_scratch((size_t)2 << 20, arena_mb << 20);
        size_t wb = 0;
        for (auto& w : G.weights) wb += w.n * 2;
        printf("VAE weights %zu MB (%s), arena %zu MB, staging 8 MB\n", wb >> 20, device_weights ? "VRAM" : "mapped host RAM", arena_mb);
        std::string c = args[0];
        if (c == "golden") for (size_t i = 1; i < args.size(); i++) cmd_golden(args[i]);
        else if (c == "band") cmd_band(args.at(1), std::stoi(args.at(2)));
        else if (c == "time") cmd_time(std::stoi(args.at(1)), std::stoi(args.at(2)), runs);
        else if (c == "prof") cmd_prof(std::stoi(args.at(1)), std::stoi(args.at(2)));
        else if (c == "probe") cmd_probe(std::vector<std::string>(args.begin() + 1, args.end()));
        else if (c == "png") cmd_png(std::vector<std::string>(args.begin() + 1, args.end()));
        else if (c == "save") {
            std::vector<int> zs;
            auto z = load_npy(args.at(1) + "/vae_in.npy", zs);
            auto img = decode(z, zs[1], zs[2], Prec::FP16);
            save_png(args.at(2), img, zs[2] * 8, zs[1] * 8);
        } else { fprintf(stderr, "unknown command %s\n", c.c_str()); return 2; }
        printf("arena peak %zu MB\n", G.arena.peak >> 20);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
