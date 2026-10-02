// LoRA optimizers on the CPU (the 46M-parameter adapters' state would cost ~0.7 GB of VRAM): the master parameters
// and the optimizer state live in system RAM; each step brings the gradients down and sends the parameters back.
// Prodigy follows prodigyopt 1.0 (lr is a multiplier on its own step-size estimate d).
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

struct LoraOptimizer {
    std::string kind = "prodigy";  // prodigy | adamw
    float beta1 = 0.9f, beta2 = 0.99f, eps = 1e-8f, weight_decay = 0.01f;
    // Prodigy
    float d0 = 1e-6f, d_coef = 1.f;
    bool bias_correction = true, safeguard_warmup = true;
    double d = 1e-6, d_max = 1e-6, d_numerator = 0.0;
    long long k = 0;  // steps taken

    std::vector<float> p, m, v, s, p0;

    void init(const float* params, size_t n) {
        p.assign(params, params + n);
        m.assign(n, 0.f);
        v.assign(n, 0.f);
        if (kind == "prodigy") { s.assign(n, 0.f); p0 = p; d = d_max = d0; d_numerator = 0.0; }
        k = 0;
    }

    template <typename F> static void par(size_t n, F f) {
        const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        std::vector<std::thread> ts;
        for (int t = 0; t < nt; t++) ts.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt, t); });
        for (auto& th : ts) th.join();
    }

    // g: the gradients (modified: clipped). Returns the gradient norm before clipping.
    double step(float* g, float lr, float max_norm) {
        const size_t n = p.size();
        std::vector<double> part(64, 0.0), part2(64, 0.0);
        par(n, [&](size_t a, size_t b, int t) { double s2 = 0; for (size_t i = a; i < b; i++) s2 += (double)g[i] * g[i]; part[t] = s2; });
        double norm = 0;
        for (double x : part) norm += x;
        norm = std::sqrt(norm);
        if (max_norm > 0 && norm > max_norm) {
            const float c = (float)(max_norm / (norm + 1e-6));
            par(n, [&](size_t a, size_t b, int) { for (size_t i = a; i < b; i++) g[i] *= c; });
        }
        if (kind == "adamw") {
            const double bc1 = 1.0 - std::pow((double)beta1, (double)(k + 1)), bc2 = 1.0 - std::pow((double)beta2, (double)(k + 1));
            const float step_size = (float)(lr / bc1), bc2s = (float)std::sqrt(bc2), wd = lr * weight_decay;
            par(n, [&](size_t a, size_t b, int) {
                for (size_t i = a; i < b; i++) {
                    m[i] = beta1 * m[i] + (1.f - beta1) * g[i];
                    v[i] = beta2 * v[i] + (1.f - beta2) * g[i] * g[i];
                    p[i] -= wd * p[i];
                    p[i] -= step_size * m[i] / (std::sqrt(v[i]) / bc2s + eps);
                }
            });
        } else {
            const double beta3 = std::sqrt((double)beta2);
            const double bc = bias_correction ? std::sqrt(1.0 - std::pow((double)beta2, (double)(k + 1))) / (1.0 - std::pow((double)beta1, (double)(k + 1))) : 1.0;
            const double dlr = d * lr * bc;
            const float fd = (float)d, a1 = (float)(d * (1.0 - beta1)), a2 = (float)(d * d * (1.0 - beta2)), b3 = (float)beta3;
            const float sa = (float)((d / d0) * (safeguard_warmup ? d : dlr));
            par(n, [&](size_t a, size_t b, int t) {
                double num = 0, den = 0;
                for (size_t i = a; i < b; i++) {
                    num += (double)g[i] * (p0[i] - p[i]);
                    m[i] = beta1 * m[i] + a1 * g[i];
                    v[i] = beta2 * v[i] + a2 * g[i] * g[i];
                    s[i] = b3 * s[i] + sa * g[i];
                    den += std::fabs((double)s[i]);
                }
                part[t] = num;
                part2[t] = den;
            });
            (void)fd;
            double num = 0, den = 0;
            for (int t = 0; t < 64; t++) { num += part[t]; den += part2[t]; }
            d_numerator = d_numerator * beta3 + (d / d0) * dlr * num;
            if (den > 0 && lr > 0) {
                const double d_hat = d_coef * d_numerator / den;
                if (d == d0) d = std::max(d, d_hat);
                d_max = std::max(d_max, d_hat);
                d = d_max;
            }
            const float fdlr = (float)dlr, deps = (float)(d * eps), wd = (float)(weight_decay * dlr);
            par(n, [&](size_t a, size_t b, int) {
                for (size_t i = a; i < b; i++) {
                    p[i] -= wd * p[i];
                    p[i] -= fdlr * m[i] / (std::sqrt(v[i]) + deps);
                }
            });
        }
        k++;
        return norm;
    }
};
