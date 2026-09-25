// Memory-mapped .safetensors reader. Tensors are views into the mapping.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum class DType { BF16, F16, F32, I64, Other };

struct StTensor {
    DType dtype = DType::Other;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);
    ~SafeTensors();
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const StTensor& get(const std::string& name) const;
    const std::map<std::string, StTensor>& all() const { return tensors_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    void* file_ = nullptr;
    void* mapping_ = nullptr;
    const uint8_t* base_ = nullptr;
    std::map<std::string, StTensor> tensors_;
};

// bf16/f16/f32 element -> float, for host-side use
float st_elem_f32(const StTensor& t, int64_t i);
