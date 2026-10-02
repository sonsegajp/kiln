#include "image_io.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

namespace {
template <typename T> struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() { return p; }
};
}  // namespace

bool load_image_rgb(const std::string& path, int& w, int& h, std::vector<uint8_t>& rgb, std::string& err) {
    // COM stays initialised until every COM object below is released (they are declared after this guard): releasing
    // them after the last CoUninitialize of a thread crashes
    struct ComInit {
        bool on;
        ComInit() : on(SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {}
        ~ComInit() { if (on) CoUninitialize(); }
    } com;
    auto fail = [&](const char* what, HRESULT hr) {
        char b[160];
        snprintf(b, sizeof b, "%s (0x%08lx)", what, (unsigned long)hr);
        err = b;
        return false;
    };
    {
        int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wp(n > 0 ? n : 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wp[0], n);
        Com<IWICImagingFactory> f;
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f));
        if (FAILED(hr)) return fail("no image decoder (WIC)", hr);
        Com<IWICBitmapDecoder> dec;
        hr = f->CreateDecoderFromFilename(wp.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec);
        if (FAILED(hr)) return fail("cannot decode the image", hr);
        Com<IWICBitmapFrameDecode> fr;
        hr = dec->GetFrame(0, &fr);
        if (FAILED(hr)) return fail("cannot read the first frame", hr);
        Com<IWICFormatConverter> cv;
        hr = f->CreateFormatConverter(&cv);
        if (FAILED(hr)) return fail("no pixel converter", hr);
        hr = cv->Initialize(fr.p, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) return fail("cannot convert the pixels", hr);
        UINT uw = 0, uh = 0;
        cv->GetSize(&uw, &uh);
        if (!uw || !uh || (size_t)uw * uh > (size_t)100 << 20) return fail("bad image size", E_FAIL);
        std::vector<uint8_t> rgba((size_t)uw * uh * 4);
        hr = cv->CopyPixels(nullptr, uw * 4, (UINT)rgba.size(), rgba.data());
        if (FAILED(hr)) return fail("cannot copy the pixels", hr);
        w = (int)uw;
        h = (int)uh;
        rgb.resize((size_t)w * h * 3);
        for (size_t i = 0; i < (size_t)w * h; i++) {
            const unsigned a = rgba[i * 4 + 3];
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = (uint8_t)((rgba[i * 4 + c] * a + 255u * (255u - a) + 127u) / 255u);
        }
    }
    return true;
}
