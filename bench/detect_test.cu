// Face detector harness: load PNGs (WIC), run FaceDetector, print boxes + timing, save annotated PNGs.
//   detect_test.exe [--model <base>] [--fp32w] [--conf 0.3] [--iou 0.5] [--runs 10] [--arena-mb 192]
//                   [--out <dir>] [--dump <dir>] image.png [image2.png ...]
// --dump writes <name>_input.npy [3,h,w], _raw.npy [65,N], _dec.npy [5,N], _boxes.txt for tools/yolo_ref.py.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "detect.h"

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

// -> interleaved RGB8
static void load_png(const std::string& path, int& W, int& H, std::vector<uint8_t>& rgb) {
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
    W = (int)w; H = (int)h;
    rgb.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) { rgb[i * 3] = bgra[i * 4 + 2]; rgb[i * 3 + 1] = bgra[i * 4 + 1]; rgb[i * 3 + 2] = bgra[i * 4]; }
}

static void save_png(const std::string& path, int W, int H, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> bgr(rgb.size());
    for (size_t i = 0; i < (size_t)W * H; i++) { bgr[i * 3] = rgb[i * 3 + 2]; bgr[i * 3 + 1] = rgb[i * 3 + 1]; bgr[i * 3 + 2] = rgb[i * 3]; }
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
    if (fmt != GUID_WICPixelFormat24bppBGR) throw std::runtime_error("PNG encoder refused 24bpp BGR");
    HR(fr->WritePixels(H, W * 3, (UINT)bgr.size(), bgr.data()));
    HR(fr->Commit());
    HR(enc->Commit());
    if (props) props->Release();
    fr->Release(); enc->Release(); s->Release();
}

static void save_npy(const std::string& path, const std::vector<float>& v, const std::vector<int>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); i++) sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : i + 1 < shape.size() ? ", " : "");
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + hdr.size() + 1;
    hdr += std::string((64 - total % 64) % 64, ' ') + "\n";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    unsigned short hl = (unsigned short)hdr.size();
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
}

// ---- drawing: rectangles + a 3x5 digit font for the score ----
static const char* GLYPH[11] = {"111101101101111", "010110010010111", "111001111100111", "111001111001111", "101101111001001",
                                "111100111001111", "111100111101111", "111001001001001", "111101111101111", "111101111001111",
                                "000000000000010"};

static void put(std::vector<uint8_t>& img, int W, int H, int x, int y, const uint8_t c[3]) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    memcpy(&img[((size_t)y * W + x) * 3], c, 3);
}
static void fill_rect(std::vector<uint8_t>& img, int W, int H, int x0, int y0, int x1, int y1, const uint8_t c[3]) {
    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) put(img, W, H, x, y, c);
}
static void draw_box(std::vector<uint8_t>& img, int W, int H, const Box& b, int t, const uint8_t c[3]) {
    int x0 = (int)b.x0, y0 = (int)b.y0, x1 = (int)b.x1, y1 = (int)b.y1;
    fill_rect(img, W, H, x0 - t, y0 - t, x1 + t, y0, c);
    fill_rect(img, W, H, x0 - t, y1, x1 + t, y1 + t, c);
    fill_rect(img, W, H, x0 - t, y0, x0, y1, c);
    fill_rect(img, W, H, x1, y0, x1 + t, y1, c);
    char s[16];
    snprintf(s, sizeof s, "%.2f", b.score);
    int sc = std::max(2, t), tx = x0 - t, th = 7 * sc;
    int ty = y0 - t - th >= 0 ? y0 - t - th : y0;
    fill_rect(img, W, H, tx, ty, tx + (int)strlen(s) * 4 * sc + sc, ty + th, c);
    const uint8_t k[3] = {0, 0, 0};
    for (int i = 0; s[i]; i++) {
        const char* g = s[i] == '.' ? GLYPH[10] : GLYPH[s[i] - '0'];
        for (int gy = 0; gy < 5; gy++) for (int gx = 0; gx < 3; gx++)
            if (g[gy * 3 + gx] == '1') fill_rect(img, W, H, tx + sc + (i * 4 + gx) * sc, ty + sc + gy * sc, tx + sc + (i * 4 + gx + 1) * sc, ty + sc + (gy + 1) * sc, k);
    }
}

static std::string stem(const std::string& p) {
    size_t a = p.find_last_of("/\\");
    std::string s = a == std::string::npos ? p : p.substr(a + 1);
    size_t d = s.find_last_of('.');
    return d == std::string::npos ? s : s.substr(0, d);
}

static std::string exe_dir() {
    char b[MAX_PATH];
    GetModuleFileNameA(nullptr, b, MAX_PATH);
    std::string s = b;
    return s.substr(0, s.find_last_of("\\/") + 1);
}

int main(int argc, char** argv) {
    std::string model = exe_dir() + "..\\models\\detect\\face_yolov8m", out_dir = exe_dir() + "detect_out", dump;
    float conf = 0.3f, iou = 0.5f;
    int runs = 10;
    size_t arena_mb = 192;
    bool fp32w = false, profile = false, host_weights = false;
    std::vector<std::string> images;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model = next();
        else if (a == "--fp32w") fp32w = true;
        else if (a == "--conf") conf = std::stof(next());
        else if (a == "--iou") iou = std::stof(next());
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--arena-mb") arena_mb = std::stoul(next());
        else if (a == "--out") out_dir = next();
        else if (a == "--dump") dump = next();
        else if (a == "--profile") profile = true;
        else if (a == "--host-weights") host_weights = true;
        else images.push_back(a);
    }
    if (images.empty()) { fprintf(stderr, "usage: detect_test [options] image.png ...\n"); return 2; }
    try {
        gpu_init(0);
        G.reserve_bytes = (size_t)64 << 20;
        size_t f0 = gpu_free_bytes();
        FaceDetector det;
        det.fp32_weights = fp32w;
        det.keep_debug = !dump.empty();
        det.load(model + ".safetensors", model + ".json", host_weights ? Place::Host : Place::Auto);
        printf("model %s: %zu MB %s weights on %s; classes:", model.c_str(), det.weight_bytes() >> 20, fp32w ? "fp32" : "fp16",
               det.weights_on_host() ? "HOST (mapped)" : "device");
        for (auto& n : det.names) printf(" %s", n.c_str());
        printf("\n");
        gpu_alloc_scratch(1024, arena_mb << 20);
        printf("VRAM free before load %zu MB, after load+arena(%zu MB) %zu MB\n", f0 >> 20, arena_mb, gpu_free_bytes() >> 20);
        CreateDirectoryA(out_dir.c_str(), nullptr);
        if (!dump.empty()) CreateDirectoryA(dump.c_str(), nullptr);

        for (auto& path : images) {
            int W, H;
            std::vector<uint8_t> rgb;
            load_png(path, W, H, rgb);
            std::vector<float> chw((size_t)3 * H * W);
            for (int c = 0; c < 3; c++)
                for (size_t i = 0; i < (size_t)H * W; i++) chw[(size_t)c * H * W + i] = rgb[i * 3 + c] / 255.f;
            size_t m = G.arena.mark();
            float* d = G.arena.f(chw.size());
            CK(cudaMemcpy(d, chw.data(), chw.size() * 4, cudaMemcpyHostToDevice));

            std::vector<Box> boxes = det.detect(d, H, W, conf, iou);  // warm-up (cuBLAS heuristics, first launches)
            float gmin = 1e9f, gsum = 0, tmin = 1e9f, tsum = 0;
            for (int r = 0; r < runs; r++) {
                boxes = det.detect(d, H, W, conf, iou);
                gmin = std::min(gmin, det.last_gpu_ms); gsum += det.last_gpu_ms;
                tmin = std::min(tmin, det.last_total_ms); tsum += det.last_total_ms;
            }
            std::string prof;
            if (profile) {
                prof_enable(true);
                det.detect(d, H, W, conf, iou);
                prof = prof_report();
                prof_enable(false);
            }
            G.arena.release(m);

            printf("\n%s  %dx%d -> letterbox %dx%d, %d anchors, scratch %.1f MB\n", path.c_str(), W, H, det.in_w, det.in_h,
                   det.num_anchors, det.last_arena_bytes / 1048576.0);
            printf("  time over %d runs: GPU min %.2f / mean %.2f ms, total (incl. NMS) min %.2f / mean %.2f ms\n", runs, gmin,
                   gsum / std::max(runs, 1), tmin, tsum / std::max(runs, 1));
            if (!prof.empty()) printf("%s", prof.c_str());
            printf("  %zu detection(s) (conf > %.2f, iou %.2f):\n", boxes.size(), conf, iou);
            for (auto& b : boxes) printf("    [%7.1f %7.1f %7.1f %7.1f]  %.4f  (%.0fx%.0f)\n", b.x0, b.y0, b.x1, b.y1, b.score, b.x1 - b.x0, b.y1 - b.y0);

            int t = std::max(2, std::min(W, H) / 300);
            const uint8_t col[3] = {255, 40, 40};
            for (auto& b : boxes) draw_box(rgb, W, H, b, t, col);
            std::string op = out_dir + "\\" + stem(path) + "_det.png";
            save_png(op, W, H, rgb);
            printf("  annotated -> %s\n", op.c_str());

            if (!dump.empty()) {
                std::string b = dump + "\\" + stem(path);
                int N = det.num_anchors;
                save_npy(b + "_input.npy", det.dbg_input, {3, det.in_h, det.in_w});
                save_npy(b + "_raw.npy", det.dbg_raw, {(int)(det.dbg_raw.size() / N), N});
                save_npy(b + "_dec.npy", det.dbg_decoded, {(int)(det.dbg_decoded.size() / N), N});
                FILE* f = fopen((b + "_boxes.txt").c_str(), "w");
                for (auto& x : boxes) fprintf(f, "%.4f %.4f %.4f %.4f %.6f\n", x.x0, x.y0, x.x1, x.y1, x.score);
                fclose(f);
            }
        }
        printf("\narena peak %.1f MB\n", G.arena.peak / 1048576.0);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
