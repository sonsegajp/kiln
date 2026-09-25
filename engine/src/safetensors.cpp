#include "safetensors.h"
#include "json.h"

#include <cstring>
#include <stdexcept>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(n - 1);
    return w;
}

SafeTensors::SafeTensors(const std::string& path) : path_(path) {
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER size;
    GetFileSizeEx(f, &size);
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); throw std::runtime_error("cannot map " + path); }
    const uint8_t* base = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("cannot view " + path); }
    file_ = f; mapping_ = m; base_ = base;

    uint64_t hlen;
    memcpy(&hlen, base, 8);
    if (hlen + 8 > (uint64_t)size.QuadPart) throw std::runtime_error("bad safetensors header: " + path);
    Json h = Json::parse(std::string((const char*)base + 8, hlen));
    const uint8_t* data = base + 8 + hlen;
    uint64_t data_size = size.QuadPart - 8 - hlen;
    for (auto& [name, v] : h.o) {
        if (name == "__metadata__") continue;
        StTensor t;
        std::string dt = v["dtype"].str();
        t.dtype = dt == "BF16" ? DType::BF16 : dt == "F16" ? DType::F16 : dt == "F32" ? DType::F32 : dt == "I64" ? DType::I64 : DType::Other;
        for (auto& d : v["shape"].a) t.shape.push_back(d.i64());
        uint64_t b0 = v["data_offsets"][0].i64(), b1 = v["data_offsets"][1].i64();
        if (b1 > data_size) throw std::runtime_error("truncated safetensors (still downloading?): " + path);
        t.data = data + b0;
        t.bytes = b1 - b0;
        tensors_[name] = t;
    }
}

SafeTensors::~SafeTensors() {
    if (base_) UnmapViewOfFile(base_);
    if (mapping_) CloseHandle(mapping_);
    if (file_) CloseHandle(file_);
}

const StTensor& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name + " in " + path_);
    return it->second;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, man = h & 1023, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 1024)) { man <<= 1; exp--; }
            man &= 1023;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) f = (sign << 31) | (255u << 23) | (man << 13);
    else f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    float r; memcpy(&r, &f, 4); return r;
}

float st_elem_f32(const StTensor& t, int64_t i) {
    switch (t.dtype) {
        case DType::BF16: { uint32_t u = (uint32_t)((const uint16_t*)t.data)[i] << 16; float f; memcpy(&f, &u, 4); return f; }
        case DType::F16: return half_to_float(((const uint16_t*)t.data)[i]);
        case DType::F32: return ((const float*)t.data)[i];
        default: throw std::runtime_error("unsupported dtype");
    }
}
