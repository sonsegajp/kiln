// Image files through Windows Imaging Component (JPEG, PNG, BMP, GIF, TIFF, WebP with the system codec): no
// third-party decoders. Transparent pixels are composited onto white.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// path (UTF-8) -> RGB8 [h][w][3]; false with err set when it can't be read
bool load_image_rgb(const std::string& path, int& w, int& h, std::vector<uint8_t>& rgb, std::string& err);
