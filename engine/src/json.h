// Minimal JSON: enough for safetensors headers and the stdin protocol.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;

    bool has(const std::string& k) const { return type == Obj && o.count(k); }
    const Json& operator[](const std::string& k) const {
        static const Json null;
        if (type != Obj) return null;
        auto it = o.find(k);
        return it == o.end() ? null : it->second;
    }
    const Json& operator[](size_t i) const { return a.at(i); }
    size_t size() const { return type == Arr ? a.size() : type == Obj ? o.size() : 0; }
    double num(double def = 0) const { return type == Num ? n : type == Bool ? (b ? 1 : 0) : def; }
    int64_t i64(int64_t def = 0) const { return type == Num ? (int64_t)n : def; }
    std::string str(const std::string& def = "") const { return type == Str ? s : def; }

    static Json parse(const std::string& text) {
        size_t p = 0;
        Json v = parse_value(text, p);
        skip_ws(text, p);
        if (p != text.size()) throw std::runtime_error("json: trailing characters");
        return v;
    }

private:
    static void skip_ws(const std::string& t, size_t& p) {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' || t[p] == '\r')) p++;
    }
    static void put_utf8(std::string& out, uint32_t c) {
        if (c < 0x80) out += (char)c;
        else if (c < 0x800) { out += (char)(0xC0 | (c >> 6)); out += (char)(0x80 | (c & 63)); }
        else if (c < 0x10000) { out += (char)(0xE0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 63)); out += (char)(0x80 | (c & 63)); }
        else { out += (char)(0xF0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 63)); out += (char)(0x80 | ((c >> 6) & 63)); out += (char)(0x80 | (c & 63)); }
    }
    static std::string parse_string(const std::string& t, size_t& p) {
        if (t[p] != '"') throw std::runtime_error("json: expected string");
        p++;
        std::string out;
        while (p < t.size() && t[p] != '"') {
            char c = t[p++];
            if (c != '\\') { out += c; continue; }
            char e = t.at(p++);
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    uint32_t c1 = (uint32_t)std::stoul(t.substr(p, 4), nullptr, 16); p += 4;
                    if (c1 >= 0xD800 && c1 < 0xDC00 && t.compare(p, 2, "\\u") == 0) {
                        uint32_t c2 = (uint32_t)std::stoul(t.substr(p + 2, 4), nullptr, 16); p += 6;
                        c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                    }
                    put_utf8(out, c1);
                    break;
                }
                default: out += e;
            }
        }
        if (p >= t.size()) throw std::runtime_error("json: unterminated string");
        p++;
        return out;
    }
    static Json parse_value(const std::string& t, size_t& p) {
        skip_ws(t, p);
        if (p >= t.size()) throw std::runtime_error("json: unexpected end");
        Json v;
        char c = t[p];
        if (c == '{') {
            v.type = Obj; p++;
            skip_ws(t, p);
            if (t[p] == '}') { p++; return v; }
            for (;;) {
                skip_ws(t, p);
                std::string k = parse_string(t, p);
                skip_ws(t, p);
                if (t.at(p++) != ':') throw std::runtime_error("json: expected ':'");
                v.o[k] = parse_value(t, p);
                skip_ws(t, p);
                char d = t.at(p++);
                if (d == '}') break;
                if (d != ',') throw std::runtime_error("json: expected ','");
            }
        } else if (c == '[') {
            v.type = Arr; p++;
            skip_ws(t, p);
            if (t[p] == ']') { p++; return v; }
            for (;;) {
                v.a.push_back(parse_value(t, p));
                skip_ws(t, p);
                char d = t.at(p++);
                if (d == ']') break;
                if (d != ',') throw std::runtime_error("json: expected ','");
            }
        } else if (c == '"') {
            v.type = Str; v.s = parse_string(t, p);
        } else if (t.compare(p, 4, "true") == 0) { v.type = Bool; v.b = true; p += 4; }
        else if (t.compare(p, 5, "false") == 0) { v.type = Bool; p += 5; }
        else if (t.compare(p, 4, "null") == 0) { p += 4; }
        else {
            char* end;
            v.type = Num; v.n = strtod(t.c_str() + p, &end);
            if (end == t.c_str() + p) throw std::runtime_error("json: bad value");
            p = end - t.c_str();
        }
        return v;
    }
};

inline std::string json_escape(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += (char)c;
    }
    return out + "\"";
}

inline std::string json_dump(const Json& v) {
    switch (v.type) {
        case Json::Bool: return v.b ? "true" : "false";
        case Json::Num: { char b[32]; snprintf(b, sizeof b, "%.17g", v.n); return b; }
        case Json::Str: return json_escape(v.s);
        case Json::Arr: {
            std::string s = "[";
            for (size_t i = 0; i < v.a.size(); i++) s += (i ? "," : "") + json_dump(v.a[i]);
            return s + "]";
        }
        case Json::Obj: {
            std::string s = "{";
            for (auto& [k, x] : v.o) s += (s.size() > 1 ? "," : "") + json_escape(k) + ":" + json_dump(x);
            return s + "}";
        }
        default: return "null";
    }
}
