#include "pipeline.h"

#include "sdpipe.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static inline unsigned nb(size_t n) { return (unsigned)((n + 255) / 256); }

// ---------------------------------------------------------------------------
// torch-compatible CPU randn
// ---------------------------------------------------------------------------
namespace {
struct TorchMT {
    uint32_t s[624];
    int idx = 624;
    explicit TorchMT(uint64_t seed) {
        s[0] = (uint32_t)(seed & 0xffffffffu);
        for (int j = 1; j < 624; j++) s[j] = 1812433253u * (s[j - 1] ^ (s[j - 1] >> 30)) + j;
    }
    uint32_t next() {
        if (idx >= 624) {
            for (int k = 0; k < 624; k++) {
                uint32_t y = (s[k] & 0x80000000u) | (s[(k + 1) % 624] & 0x7fffffffu);
                s[k] = s[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
            }
            idx = 0;
        }
        uint32_t y = s[idx++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }
    float uniform() { return (float)(next() & ((1u << 24) - 1)) * (1.0f / (1u << 24)); }
};

void normal_fill_16(float* d) {
    for (int j = 0; j < 8; j++) {
        float u1 = 1.f - d[j], u2 = d[j + 8];
        float radius = sqrtf(-2.f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;
        d[j] = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}
}  // namespace

std::vector<float> torch_randn(uint64_t seed, size_t n) {
    TorchMT g(seed);
    std::vector<float> d(n);
    for (auto& v : d) v = g.uniform();
    for (size_t i = 0; i + 16 <= n; i += 16) normal_fill_16(&d[i]);
    if (n % 16) {
        float* t = &d[n - 16];
        for (int i = 0; i < 16; i++) t[i] = g.uniform();
        normal_fill_16(t);
    }
    return d;
}

std::vector<float> flow_sigmas(int steps, float shift, float denoise) {
    int total = denoise > 0.9999f ? steps : (int)((double)steps / (double)denoise);
    std::vector<float> table(1000);
    for (int i = 0; i < 1000; i++) {
        float t = (float)(i + 1) / 1000.f;
        table[i] = shift == 1.f ? t : shift * t / (1.f + (shift - 1.f) * t);
    }
    std::vector<float> s;
    double ss = 1000.0 / total;
    for (int x = 0; x < total; x++) s.push_back(table[999 - (int)(x * ss)]);
    s.push_back(0.f);
    return std::vector<float>(s.end() - (steps + 1), s.end());
}

// ---------------------------------------------------------------------------
// ComfyUI schedulers over ModelSamplingDiscreteFlow(shift, multiplier) -- the multiplier cancels out
// ---------------------------------------------------------------------------
static double snr_shift(double shift, double t) { return shift == 1.0 ? t : shift * t / (1.0 + (shift - 1.0) * t); }

static std::vector<float> flow_table(float shift) {
    std::vector<float> table(1000);
    for (int i = 0; i < 1000; i++) {
        float t = (float)(i + 1) / 1000.f;
        table[i] = shift == 1.f ? t : shift * t / (1.f + (shift - 1.f) * t);
    }
    return table;
}

// regularized incomplete beta I_x(a, b) (continued fraction, Numerical Recipes) and its inverse
static double betacf(double a, double b, double x) {
    const int MAXIT = 300;
    const double EPS = 1e-15, FPMIN = 1e-300;
    double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (std::fabs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; m++) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < EPS) break;
    }
    return h;
}
static double betainc(double a, double b, double x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
static double beta_ppf(double p, double a, double b) {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    double lo = 0, hi = 1;
    for (int i = 0; i < 200; i++) {
        double mid = (lo + hi) / 2;
        (betainc(a, b, mid) < p ? lo : hi) = mid;
    }
    return (lo + hi) / 2;
}

static const char* SCHEDULERS[] = {"simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"};
bool known_scheduler(const std::string& n) {
    for (auto s : SCHEDULERS) if (n == s) return true;
    return false;
}
bool known_sampler(const std::string& n) { return n == "euler" || n == "euler_ancestral" || n == "dpmpp_2m" || n == "res_multistep"; }

// The model's sigma space: the sigmas of its 1000 timesteps (ascending) and the timestep <-> sigma maps
// the schedulers use. Flow (ModelSamplingDiscreteFlow): timestep = sigma (the multiplier cancels),
// sigma(t) = shift(t). Discrete (ModelSamplingDiscrete, SD/SDXL): timestep = the nearest table index in
// log-sigma, sigma(t) = log-linear interpolation of the table.
struct SigmaSpace {
    std::vector<float> tab, log_tab;
    bool flow = true;
    float shift = 1.f;
    double timestep(double s) const {
        if (flow) return s;
        const float ls = logf((float)s);
        int best = 0;
        for (int i = 1; i < (int)log_tab.size(); i++) if (fabsf(ls - log_tab[i]) < fabsf(ls - log_tab[best])) best = i;
        return best;
    }
    double sigma(double t) const {
        if (flow) return snr_shift(shift, t);
        const float tc = std::min(std::max((float)t, 0.f), 999.f);
        const int lo = (int)floorf(tc), hi = (int)ceilf(tc);
        const float w = tc - floorf(tc);
        return expf((1.f - w) * log_tab[lo] + w * log_tab[hi]);
    }
};

static SigmaSpace flow_space(float shift) {
    SigmaSpace S;
    S.tab = flow_table(shift);
    S.shift = shift;
    return S;
}

// scaled_linear betas 0.00085 .. 0.012 over 1000 steps, in float64 like the reference, stored as float32
static const SigmaSpace& sd_space() {
    static SigmaSpace S = [] {
        SigmaSpace s;
        s.flow = false;
        double ac = 1.0, a = std::sqrt(0.00085), b = std::sqrt(0.012);
        for (int i = 0; i < 1000; i++) {
            const double beta = std::pow(a + (b - a) * i / 999.0, 2.0);
            ac *= 1.0 - beta;
            const double sig = std::sqrt((1.0 - ac) / ac);
            s.tab.push_back((float)sig);
            s.log_tab.push_back((float)std::log(sig));
        }
        return s;
    }();
    return S;
}

static std::vector<float> full_schedule(const std::string& name, int steps, const SigmaSpace& S) {
    const std::vector<float>& tab = S.tab;
    double smin = tab[0], smax = tab[999];
    std::vector<float> s;
    if (name == "simple") {
        double ss = 1000.0 / steps;
        for (int x = 0; x < steps; x++) s.push_back(tab[999 - (int)(x * ss)]);
        s.push_back(0.f);
    } else if (name == "normal" || name == "sgm_uniform") {
        // linspace in timestep space, then back to sigmas
        double start = S.timestep(smax), end = S.timestep(smin);
        bool append_zero = true;
        int n = steps;
        std::vector<double> ts;
        if (name == "sgm_uniform") {
            for (int i = 0; i <= n; i++) ts.push_back(start + (end - start) * i / n);
            ts.pop_back();
        } else {
            if (std::fabs(S.sigma(end)) < 1e-5) { n += 1; append_zero = false; }
            for (int i = 0; i < n; i++) ts.push_back(n == 1 ? start : start + (end - start) * i / (n - 1));
        }
        for (double t : ts) s.push_back((float)S.sigma(t));
        if (append_zero) s.push_back(0.f);
    } else if (name == "karras") {
        double rho = 7.0, a = std::pow(smax, 1 / rho), b = std::pow(smin, 1 / rho);
        for (int i = 0; i < steps; i++) {
            double r = steps == 1 ? 0.0 : (double)i / (steps - 1);
            s.push_back((float)std::pow(a + r * (b - a), rho));
        }
        s.push_back(0.f);
    } else if (name == "exponential") {
        double la = std::log(smax), lb = std::log(smin);
        for (int i = 0; i < steps; i++) s.push_back((float)std::exp(steps == 1 ? la : la + (lb - la) * i / (steps - 1)));
        s.push_back(0.f);
    } else if (name == "ddim_uniform") {
        std::vector<float> sigs;
        int x = 1, n = steps;
        if (std::fabs(tab[x]) < 1e-5) n += 1; else sigs.push_back(0.f);
        int ss = std::max(1000 / n, 1);
        while (x < 1000) { sigs.push_back(tab[x]); x += ss; }
        s.assign(sigs.rbegin(), sigs.rend());
    } else if (name == "beta") {
        int last = -1;
        for (int i = 0; i < steps; i++) {
            double p = 1.0 - (double)i / steps;
            int t = (int)std::nearbyint(beta_ppf(p, 0.6, 0.6) * 999.0);
            if (t != last) s.push_back(tab[t]);
            last = t;
        }
        s.push_back(0.f);
    } else if (name == "linear_quadratic") {
        if (steps == 1) s = {1.f, 0.f};
        else {
            double thr = 0.025;
            int lin = steps / 2, quad = steps - lin;
            std::vector<double> sch;
            for (int i = 0; i < lin; i++) sch.push_back(i * thr / lin);
            double diff = lin - thr * steps;
            double qc = diff / (lin * (double)quad * quad), lc = thr / lin - 2 * diff / ((double)quad * quad), cst = qc * lin * lin;
            for (int i = lin; i < steps; i++) sch.push_back(qc * i * i + lc * i + cst);
            sch.push_back(1.0);
            for (double v : sch) s.push_back((float)((1.0 - v) * smax));
        }
    } else if (name == "kl_optimal") {
        for (int i = 0; i < steps; i++) {
            double adj = steps == 1 ? 0.0 : (double)i / (steps - 1);
            s.push_back((float)std::tan(adj * std::atan(smin) + (1 - adj) * std::atan(smax)));
        }
        s.push_back(0.f);
    } else {
        throw std::runtime_error("unknown scheduler " + name);
    }
    return s;
}

static std::vector<float> with_denoise(const std::string& name, int steps, float denoise, const SigmaSpace& S) {
    if (denoise > 0.9999f) return full_schedule(name, steps, S);
    if (denoise <= 0.f) return {};
    int total = (int)((double)steps / (double)denoise);
    std::vector<float> s = full_schedule(name, total, S);
    if ((int)s.size() < steps + 1) return s;
    return std::vector<float>(s.end() - (steps + 1), s.end());
}

std::vector<float> schedule_sigmas(const std::string& name, int steps, float shift, float denoise) {
    return with_denoise(name, steps, denoise, flow_space(shift));
}

std::vector<float> schedule_sigmas_sd(const std::string& name, int steps, float denoise) { return with_denoise(name, steps, denoise, sd_space()); }
float sd_timestep(float sigma) { return (float)sd_space().timestep(sigma); }
float sd_sigma_max() { return sd_space().tab[999]; }

// ---------------------------------------------------------------------------
// latent normalization
// ---------------------------------------------------------------------------
__constant__ float c_mean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                 0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
__constant__ float c_std[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

__global__ void k_latent_norm(float* z, int P, bool to_vae) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)16 * P) return;
    int c = (int)(i / P);
    z[i] = to_vae ? z[i] * c_std[c] + c_mean[c] : (z[i] - c_mean[c]) / c_std[c];
}
void latent_model_to_vae(float* z, int P) { k_latent_norm<<<nb((size_t)16 * P), 256, 0, G.stream>>>(z, P, true); }
void latent_vae_to_model(float* z, int P) { k_latent_norm<<<nb((size_t)16 * P), 256, 0, G.stream>>>(z, P, false); }

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------
// x = s*noise + (1-s)*init   (flow-matching noise scaling; init may be null)
__global__ void k_start(float* x, const float* noise, const float* init, float s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = s * noise[i] + (init ? (1.f - s) * init[i] : 0.f);
}

// model input under a mask: regenerate where m = 1, the init re-noised to this sigma where m = 0
__global__ void k_inpaint_in(float* xin, const float* x, const float* noise, const float* init, const float* mask, float s, int P) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)16 * P) return;
    float m = mask[i % P];
    xin[i] = m * x[i] + (1.f - m) * (s * noise[i] + (1.f - s) * init[i]);
}

// denoised = xin - v*s (flow matching), pinned to init outside the mask
__global__ void k_denoised(float* den, const float* xin, const float* v, const float* init, const float* mask, float s, int P) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)16 * P) return;
    float d = xin[i] - v[i] * s;
    if (mask) { float m = mask[i % P]; d = m * d + (1.f - m) * init[i]; }
    den[i] = d;
}

// Euler in its original arithmetic, x + (x - den)/s * (sn - s): the same step as k_update's first-order
// form, but kept bit-identical so earlier renders reproduce exactly from their seed.
__global__ void k_euler_step(float* x, const float* den, float s, float sn, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + (x[i] - den[i]) / s * (sn - s);
}

// every supported sampler step is x <- a*x + b*denoised + c*previous_denoised
__global__ void k_update(float* x, const float* den, const float* old, float a, float b, float c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + b * den[i] + (old ? c * old[i] : 0.f);
}

// Coefficients of ComfyUI's k-diffusion samplers (deterministic variants), in their own
// parameterization: t = -log(sigma), h = t_next - t.
struct StepCoef { float a, b, c; };
static StepCoef step_coef(const std::string& sampler, const std::vector<float>& sig, int i, bool have_old) {
    double s = sig[i], sn = sig[i + 1];
    if (sampler == "euler" || !have_old || sn == 0.0)  // first-order step (Euler == DDIM here)
        return {(float)(sn / s), (float)(1.0 - sn / s), 0.f};
    double t = -std::log(s), tn = -std::log(sn), tp = -std::log((double)sig[i - 1]), h = tn - t;
    if (sampler == "dpmpp_2m") {
        double r = (t - tp) / h;
        double b = -std::expm1(-h);
        return {(float)(sn / s), (float)(b * (1.0 + 1.0 / (2.0 * r))), (float)(-b / (2.0 * r))};
    }
    // res_multistep (eta 0): x = e^-h x + h (b1 den + b2 old), c2 = (t_prev - t) / h
    double c2 = (tp - t) / h;
    double phi1 = std::expm1(-h) / (-h), phi2 = (phi1 - 1.0) / (-h);
    double b1 = phi1 - phi2 / c2, b2 = phi2 / c2;
    return {(float)std::exp(-h), (float)(h * b1), (float)(h * b2)};
}

int last_cache_skips = 0;

void sample(Dit& dit, float* x, int Hl, int Wl, const Context& pos, const Context* neg, const SampleParams& sp,
            const float* init, const float* mask, const StepFn& on_step, const std::atomic<bool>* cancel) {
    const int P = Hl * Wl;
    const size_t n = (size_t)16 * P;
    std::vector<float> sig = !sp.sigmas.empty() ? sp.sigmas
                             : sp.scheduler == "simple" ? flow_sigmas(sp.steps, sp.shift, sp.denoise)
                                                        : schedule_sigmas(sp.scheduler, sp.steps, sp.shift, sp.denoise);
    const int steps = (int)sig.size() - 1;
    if (steps < 1) {  // nothing to do (denoise 0): the result is the input
        if (init) CK(cudaMemcpy(x, init, n * 4, cudaMemcpyDeviceToDevice));
        return;
    }
    std::vector<float> nh = !sp.add_noise ? std::vector<float>(n, 0.f)
                            : sp.noise ? std::vector<float>(sp.noise, sp.noise + n) : torch_randn(sp.seed, n);

    size_t m0 = G.arena.mark();
    bool cached = false;
    try {
        float* noise = G.arena.f(n);
        CK(cudaMemcpy(noise, nh.data(), n * 4, cudaMemcpyHostToDevice));
        // cross-attn K/V once per image when the arena can spare it (28 blocks x 2 x L x 2048 floats)
        int B = neg ? 2 : 1;
        size_t kv_bytes = (size_t)28 * 2 * (pos.real + (neg ? neg->real : 0)) * 2048 * 4;
        size_t step_need = Dit::forward_bytes(Hl, Wl, B) + ((size_t)96 << 20);
        const Context* nagc = sp.nag_neg;  // with CFG too: NAG then guides only the prompt pass
        // NAG's extra per-forward scratch: attention output for the prompt rows + uncached K/V
        const size_t nag_need = nagc ? (size_t)(Hl / 2) * (Wl / 2) * 2048 * 4 + (size_t)2 * nagc->len * 2048 * 4 : 0;
        step_need += nag_need;
        if (nagc) kv_bytes += (size_t)28 * 2 * nagc->real * 2048 * 4;
        if (G.arena.free_bytes() > kv_bytes + step_need) {
            dit.cache_context(pos);
            if (neg) dit.cache_context(*neg);
            if (nagc) dit.cache_context(*nagc);
            cached = true;
        }
        dit.nag = Dit::Nag();
        if (nagc) { dit.nag.neg = nagc; dit.nag.scale = sp.nag_scale; dit.nag.tau = sp.nag_tau; dit.nag.alpha = sp.nag_alpha; }
        float* v = G.arena.f(n);
        float* vn = neg ? G.arena.f(n) : nullptr;
        // CFG in one batched pass when the doubled activations fit, otherwise two passes
        bool batch = neg && G.arena.free_bytes() > Dit::forward_bytes(Hl, Wl, 2) + nag_need + ((size_t)64 << 20);
        int cfg_steps = neg ? (int)std::ceil(sp.cfg_until * steps - 1e-4f) : 0;
        float* xin = mask ? G.arena.f(n) : x;
        float* den = G.arena.f(n);
        float* old = G.arena.f(n);
        bool have_old = false;
        if (!known_sampler(sp.sampler)) throw std::runtime_error("unknown sampler " + sp.sampler);
        float* anoise = sp.sampler == "euler_ancestral" ? G.arena.f(n) : nullptr;

        // first-block cache: never in the first 15% or last 10% of steps, at most 2 skips in a row
        Dit::StepCache sc;
        Dit::StepCache* scp = nullptr;
        int skips = 0, run = 0;
        if (sp.cache_threshold > 0.f) {
            int R = (batch ? 2 : 1) * (Hl / 2) * (Wl / 2);
            sc.threshold = sp.cache_threshold;
            sc.R = R;
            sc.r0 = G.arena.f((size_t)R * 2048);
            sc.resid = G.arena.f((size_t)R * 2048);
            scp = &sc;
        }
        int warm = std::max(2, (int)std::ceil(sp.cache_start * steps));
        int tail = std::max(1, (int)std::ceil((1.f - sp.cache_end) * steps));
        int max_hits = sp.cache_max_hits < 0 ? steps : sp.cache_max_hits;
        k_start<<<nb(n), 256, 0, G.stream>>>(x, noise, init, sig[0], n);

        for (int i = 0; i < steps; i++) {
            if (cancel && *cancel) throw std::runtime_error("cancelled");
            auto t0 = std::chrono::steady_clock::now();
            float s = sig[i];
            if (mask) k_inpaint_in<<<nb(n), 256, 0, G.stream>>>(xin, x, noise, init, mask, s, P);
            if (scp) scp->allow = i >= warm && i < steps - tail && run < max_hits;
            if (nagc) dit.nag.neg = s >= sp.nag_sigma_end ? nagc : nullptr;
            const float tt = s * sp.timestep_mult;  // what the model sees as its timestep
            bool cfg_now = neg && i < cfg_steps;
            if (cfg_now && batch) {
                const Context* cs[2] = {&pos, neg};
                float* outs[2] = {v, vn};
                dit.forward_batch(outs, xin, Hl, Wl, tt, cs, 2, scp);
            } else {
                // unbatched CFG would skip the prompt pass but not the negative one: no cache there
                Dit::StepCache* use = (!neg || batch) ? scp : nullptr;
                const Context* cs[1] = {&pos};
                float* outs[1] = {v};
                dit.forward_batch(outs, xin, Hl, Wl, tt, cs, 1, use);
                if (cfg_now) {  // the negative pass never gets NAG
                    const Context* keep = dit.nag.neg;
                    dit.nag.neg = nullptr;
                    dit.forward(vn, xin, Hl, Wl, tt, *neg);
                    dit.nag.neg = keep;
                }
            }
            if (cfg_now) cfg_combine(v, vn, sp.cfg, n);
            if (scp) {  // forward_batch invalidates the cache itself when the batch shape changes
                if (scp->skipped) { skips++; run++; } else run = 0;
                if (getenv("KILN_CACHE_LOG")) fprintf(stderr, "step %2d sigma %.3f change %.4f %s\n", i, s, scp->last_change, scp->skipped ? "SKIP" : "");
            }
            if (on_step) {
                gpu_sync();
                on_step(i + 1, steps, s, x, v, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            k_denoised<<<nb(n), 256, 0, G.stream>>>(den, xin, v, init, mask, s, P);
            if (sp.sampler == "euler") {
                k_euler_step<<<nb(n), 256, 0, G.stream>>>(x, den, s, sig[i + 1], n);
            } else if (sp.sampler == "euler_ancestral") {
                // ComfyUI's flow-model (RF) ancestral step, eta = 1. Kiln draws its own noise (CPU randn
                // per step), so it is not seed-identical to ComfyUI's GPU noise.
                double sn = sig[i + 1];
                if (sn == 0.0) {
                    k_update<<<nb(n), 256, 0, G.stream>>>(x, den, nullptr, 0.f, 1.f, 0.f, n);
                } else {
                    double sd = sn * (1.0 + (sn / s - 1.0)), aip1 = 1.0 - sn, ad = 1.0 - sd;
                    double renoise = std::sqrt(std::max(0.0, sn * sn - sd * sd * aip1 * aip1 / (ad * ad)));
                    double r = sd / s;
                    std::vector<float> nz = torch_randn(sp.seed + 1 + i, n);
                    CK(cudaMemcpy(anoise, nz.data(), n * 4, cudaMemcpyHostToDevice));
                    float a = (float)(aip1 / ad);
                    k_update<<<nb(n), 256, 0, G.stream>>>(x, den, nullptr, (float)(a * r), (float)(a * (1.0 - r)), 0.f, n);
                    k_update<<<nb(n), 256, 0, G.stream>>>(x, anoise, nullptr, 1.f, (float)renoise, 0.f, n);
                }
            } else {
                StepCoef co = step_coef(sp.sampler, sig, i, have_old);
                k_update<<<nb(n), 256, 0, G.stream>>>(x, den, have_old ? old : nullptr, co.a, co.b, co.c, n);
            }
            std::swap(den, old);
            have_old = true;
        }
        // stopped above sigma 0 (KSamplerAdvanced leftover noise): hand back x / (1 - sigma) like ComfyUI's
        // CONST inverse_noise_scaling, so a second pass with add_noise off rebuilds exactly this x
        if (sig.back() > 0.f && sig.back() < 1.f)
            k_update<<<nb(n), 256, 0, G.stream>>>(x, x, nullptr, 1.f / (1.f - sig.back()), 0.f, 0.f, n);
        gpu_sync();
        if (scp) last_cache_skips = skips;
    } catch (...) {
        dit.nag = Dit::Nag();
        if (cached) dit.drop_context_cache();
        G.arena.release(m0);
        throw;
    }
    dit.nag = Dit::Nag();
    if (cached) dit.drop_context_cache();
    G.arena.release(m0);
}

// ---------------------------------------------------------------------------
// eps sampling (SD / SDXL: ModelSamplingDiscrete, EPS)
// ---------------------------------------------------------------------------
// x = noise * a + init, a = sqrt(1 + s^2) when the schedule starts at the model's sigma_max, else s
__global__ void k_sd_start(float* x, const float* noise, const float* init, float a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = noise[i] * a + (init ? init[i] : 0.f);
}
// masked model input: regenerate where m = 1, the init re-noised to this sigma (init + noise * s) where m = 0
__global__ void k_sd_inpaint_in(float* xin, const float* x, const float* noise, const float* init, const float* mask, float s, int P, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * P) return;
    float m = mask[i % P];
    xin[i] = m * x[i] + (1.f - m) * (init[i] + noise[i] * s);
}
// denoised = xin - eps * s, pinned to init outside the mask
__global__ void k_sd_denoised(float* den, const float* xin, const float* e, const float* init, const float* mask, float s, int P, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * P) return;
    float d = xin[i] - e[i] * s;
    if (mask) { float m = mask[i % P]; d = m * d + (1.f - m) * init[i]; }
    den[i] = d;
}
__global__ void k_scale_to(float* y, const float* x, float a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * a;
}

void sample_sd(Unet& unet, float* x, int Hl, int Wl, const SdCond& pos, const SdCond* neg, const SampleParams& sp, const float* init,
               const float* mask, const StepFn& on_step, const std::atomic<bool>* cancel) {
    const int P = Hl * Wl, C = 4;
    const size_t n = (size_t)C * P;
    std::vector<float> sig = !sp.sigmas.empty() ? sp.sigmas : schedule_sigmas_sd(sp.scheduler, sp.steps, sp.denoise);
    const int steps = (int)sig.size() - 1;
    if (steps < 1) {
        if (init) CK(cudaMemcpy(x, init, n * 4, cudaMemcpyDeviceToDevice));
        else CK(cudaMemset(x, 0, n * 4));
        return;
    }
    if (!known_sampler(sp.sampler)) throw std::runtime_error("unknown sampler " + sp.sampler);
    std::vector<float> nh = !sp.add_noise ? std::vector<float>(n, 0.f) : sp.noise ? std::vector<float>(sp.noise, sp.noise + n) : torch_randn(sp.seed, n);

    size_t m0 = G.arena.mark();
    bool cached = false;
    try {
        float* noise = G.arena.f(n);
        CK(cudaMemcpy(noise, nh.data(), n * 4, cudaMemcpyHostToDevice));
        // every cross-attention's K/V once per run, when the arena can spare it next to a forward pass
        size_t kv = unet.kv_bytes(pos.ctx.len) + (neg ? unet.kv_bytes(neg->ctx.len) : 0);
        if (G.arena.free_bytes() > kv + Unet::forward_bytes(Hl, Wl) + ((size_t)64 << 20)) {
            unet.cache_context(pos.ctx);
            if (neg) unet.cache_context(neg->ctx);
            cached = true;
        }
        float* e = G.arena.f(n);
        float* en = neg ? G.arena.f(n) : nullptr;
        float* xin = mask ? G.arena.f(n) : x;
        float* xs = G.arena.f(n);
        float* den = G.arena.f(n);
        float* old = G.arena.f(n);
        float* anoise = sp.sampler == "euler_ancestral" ? G.arena.f(n) : nullptr;
        bool have_old = false;
        const int cfg_steps = neg ? (int)std::ceil(sp.cfg_until * steps - 1e-4f) : 0;
        const bool max_denoise = sig[0] >= sd_sigma_max() * (1.f - 1e-5f);
        k_sd_start<<<nb(n), 256, 0, G.stream>>>(x, noise, init, max_denoise ? sqrtf(1.f + sig[0] * sig[0]) : sig[0], n);

        for (int i = 0; i < steps; i++) {
            if (cancel && *cancel) throw std::runtime_error("cancelled");
            auto t0 = std::chrono::steady_clock::now();
            const float s = sig[i], sn = sig[i + 1];
            if (mask) k_sd_inpaint_in<<<nb(n), 256, 0, G.stream>>>(xin, x, noise, init, mask, s, P, C);
            k_scale_to<<<nb(n), 256, 0, G.stream>>>(xs, xin, 1.f / sqrtf(s * s + 1.f), n);
            const float t = sd_timestep(s);
            unet.forward(e, xs, Hl, Wl, t, pos.y, pos.ctx);
            sd_debug(("eps step " + std::to_string(i)).c_str(), e, n);
            if (neg && i < cfg_steps) {
                unet.forward(en, xs, Hl, Wl, t, neg->y, neg->ctx);
                cfg_combine(e, en, sp.cfg, n);
            }
            if (on_step) {
                gpu_sync();
                on_step(i + 1, steps, s, x, e, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            k_sd_denoised<<<nb(n), 256, 0, G.stream>>>(den, xin, e, init, mask, s, P, C);
            if (sp.sampler == "euler") {
                k_euler_step<<<nb(n), 256, 0, G.stream>>>(x, den, s, sn, n);
            } else if (sp.sampler == "euler_ancestral") {
                // k-diffusion's ancestral step, eta 1 (Kiln draws its own per-step noise, so it is not
                // seed-identical to other programs' GPU noise)
                const double up = sn == 0.f ? 0.0 : std::min((double)sn, std::sqrt((double)sn * sn * ((double)s * s - (double)sn * sn) / ((double)s * s)));
                const double down = std::sqrt(std::max(0.0, (double)sn * sn - up * up));
                k_update<<<nb(n), 256, 0, G.stream>>>(x, den, nullptr, (float)(down / s), (float)(1.0 - down / s), 0.f, n);
                if (up > 0.0) {
                    std::vector<float> nz = torch_randn(sp.seed + 1 + i, n);
                    CK(cudaMemcpy(anoise, nz.data(), n * 4, cudaMemcpyHostToDevice));
                    k_update<<<nb(n), 256, 0, G.stream>>>(x, anoise, nullptr, 1.f, (float)up, 0.f, n);
                }
            } else {
                StepCoef co = step_coef(sp.sampler, sig, i, have_old);
                k_update<<<nb(n), 256, 0, G.stream>>>(x, den, have_old ? old : nullptr, co.a, co.b, co.c, n);
            }
            std::swap(den, old);
            have_old = true;
        }
        gpu_sync();
        sd_debug("final latent", x, n);
    } catch (...) {
        if (cached) unet.drop_context_cache();
        G.arena.release(m0);
        throw;
    }
    if (cached) unet.drop_context_cache();
    G.arena.release(m0);
}

// ---------------------------------------------------------------------------
// image helpers
// ---------------------------------------------------------------------------
__global__ void k_affine(float* d, const float* s, float a, float b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i] * a + b;
}
void rgb_to_unit(float* dst, const float* src, size_t n) { k_affine<<<nb(n), 256, 0, G.stream>>>(dst, src, 0.5f, 0.5f, n); }
void rgb_from_unit(float* dst, const float* src, size_t n) { k_affine<<<nb(n), 256, 0, G.stream>>>(dst, src, 2.f, -1.f, n); }

__global__ void k_blend_rect(float* dst, int C, int H, int W, const float* src, const float* mask, int x0, int y0, int w, int h) {
    size_t n = (size_t)C * w * h;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xx = i % w;
    size_t r = i / w;
    int yy = r % h;
    int c = (int)(r / h);
    float m = mask[(size_t)yy * w + xx];
    float* d = dst + ((size_t)c * H + y0 + yy) * W + x0 + xx;
    *d = m * src[i] + (1.f - m) * *d;
}
void blend_rect(float* dst, int C, int H, int W, const float* src, const float* mask, int x0, int y0, int w, int h) {
    size_t n = (size_t)C * w * h;
    k_blend_rect<<<nb(n), 256, 0, G.stream>>>(dst, C, H, W, src, mask, x0, y0, w, h);
}

void crop_rect(float* dst, const float* src, int C, int H, int W, int x0, int y0, int w, int h) {
    for (int c = 0; c < C; c++)
        CK(cudaMemcpy2DAsync(dst + (size_t)c * w * h, (size_t)w * 4, src + ((size_t)c * H + y0) * W + x0, (size_t)W * 4,
                             (size_t)w * 4, h, cudaMemcpyDeviceToDevice, G.stream));
}
