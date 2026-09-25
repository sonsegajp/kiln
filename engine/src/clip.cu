#include "clip.h"

#include <cuda_fp16.h>

#include "sdops.h"

static inline unsigned nblocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// OpenCLIP applies text_projection as x @ P; linear() computes x W^T, so W = P^T
static Weight upload_transposed(const StTensor& t, Place place) {
    const int64_t R = t.shape[0], C = t.shape[1];
    std::vector<uint16_t> h((size_t)R * C);
    for (int64_t r = 0; r < R; r++)
        for (int64_t c = 0; c < C; c++) h[(size_t)c * R + r] = __half_as_ushort(__float2half_rn(st_elem_f32(t, r * C + c)));
    StTensor s;
    s.dtype = DType::F16;
    s.shape = {C, R};
    s.data = (const uint8_t*)h.data();
    s.bytes = h.size() * 2;
    return upload_weight(s, place, false, true);
}

void ClipText::load_hf(const SafeTensors& st, const std::string& p, Place place) {
    tok_ = upload_weight(st.get(p + "embeddings.token_embedding.weight"), place, false, true);
    pos_ = upload_f32(st.get(p + "embeddings.position_embedding.weight"));
    for (int i = 0; st.has(p + "encoder.layers." + std::to_string(i) + ".layer_norm1.weight"); i++) {
        const std::string l = p + "encoder.layers." + std::to_string(i);
        Layer L;
        L.ln1 = load_norm32(st, l + ".layer_norm1");
        L.ln2 = load_norm32(st, l + ".layer_norm2");
        L.q = load_linear16(st, l + ".self_attn.q_proj", place);
        L.k = load_linear16(st, l + ".self_attn.k_proj", place);
        L.v = load_linear16(st, l + ".self_attn.v_proj", place);
        L.o = load_linear16(st, l + ".self_attn.out_proj", place);
        L.fc1 = load_linear16(st, l + ".mlp.fc1", place);
        L.fc2 = load_linear16(st, l + ".mlp.fc2", place);
        layers_.push_back(std::move(L));
    }
    ln_final_ = load_norm32(st, p + "final_layer_norm");
    if (st.has(p + "text_projection.weight")) proj_ = upload_weight(st.get(p + "text_projection.weight"), place, false, true);
    dim_ = (int)tok_.cols;
    heads_ = dim_ / 64;
    pad_id_ = 49407;  // CLIP-L pads with EOS
    quick_gelu_ = true;
}

void ClipText::load_openclip(const SafeTensors& st, const std::string& p, Place place) {
    tok_ = upload_weight(st.get(p + "token_embedding.weight"), place, false, true);
    pos_ = upload_f32(st.get(p + "positional_embedding"));
    for (int i = 0; st.has(p + "transformer.resblocks." + std::to_string(i) + ".ln_1.weight"); i++) {
        const std::string l = p + "transformer.resblocks." + std::to_string(i);
        Layer L;
        L.ln1 = load_norm32(st, l + ".ln_1");
        L.ln2 = load_norm32(st, l + ".ln_2");
        L.qkv.w = upload_weight(st.get(l + ".attn.in_proj_weight"), place, false, true);
        L.qkv.b = upload_f32(st.get(l + ".attn.in_proj_bias"));
        L.o = load_linear16(st, l + ".attn.out_proj", place);
        L.fc1 = load_linear16(st, l + ".mlp.c_fc", place);
        L.fc2 = load_linear16(st, l + ".mlp.c_proj", place);
        layers_.push_back(std::move(L));
    }
    ln_final_ = load_norm32(st, p + "ln_final");
    proj_ = upload_transposed(st.get(p + "text_projection"), place);
    dim_ = (int)tok_.cols;
    heads_ = dim_ / 64;
    pad_id_ = 0;  // the bigG tokenizer pads with 0
    quick_gelu_ = false;
}

void ClipText::free() {
    for (auto& L : layers_) {
        free_norm32(L.ln1);
        free_norm32(L.ln2);
        for (Linear16* l : {&L.q, &L.k, &L.v, &L.qkv, &L.o, &L.fc1, &L.fc2}) free_linear16(*l);
    }
    layers_.clear();
    free_weight(tok_);
    free_weight(proj_);
    if (pos_) CK(cudaFree(pos_));
    pos_ = nullptr;
    free_norm32(ln_final_);
}

__global__ void k_add_pos(float* x, const float* pos, int chunks, int D) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)chunks * 77 * D) x[i] += pos[i % ((size_t)77 * D)];
}

// out = z_empty + (z - z_empty) * w for tokens with w != 1 (z_empty: the empty prompt, same position)
__global__ void k_token_weights(float* out, const float* z, const float* ze, const float* w, int T, int D) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    const int t = (int)(i / D), d = (int)(i - (size_t)t * D);
    const float wt = w[t];
    out[i] = wt == 1.f ? z[i] : ze[(size_t)(t % 77) * D + d] + (z[i] - ze[(size_t)(t % 77) * D + d]) * wt;
}

void ClipText::encode(const std::vector<int>& ids, const std::vector<float>& weights, float* hidden, float* pooled) {
    ProfScope ps("clip");
    const int chunks = (int)(ids.size() / 77), D = dim_;
    bool weighted = false;
    for (float w : weights) weighted |= w != 1.f;
    std::vector<int> all = ids;
    if (weighted) {  // the empty prompt, for the weight interpolation
        all.push_back(49406);
        all.push_back(49407);
        all.resize(all.size() + 75, pad_id_);
    }
    const int n = (int)(all.size() / 77), T = n * 77;
    int eos = 0;
    while (eos < 76 && ids[eos] != 49407) eos++;

    size_t m = G.arena.mark();
    int* d_ids = G.arena.i(T);
    CK(cudaMemcpyAsync(d_ids, all.data(), (size_t)T * 4, cudaMemcpyHostToDevice, G.stream));
    float* x = G.arena.f((size_t)T * D);
    float* a = G.arena.f((size_t)T * D);
    float* q = G.arena.f((size_t)T * 3 * D);
    float* o = G.arena.f((size_t)T * D);
    float* f = G.arena.f((size_t)T * 4 * D);
    float* z = weighted ? G.arena.f((size_t)T * D) : hidden;
    embed_rows(x, tok_, d_ids, T);
    k_add_pos<<<nblocks((size_t)T * D), 256, 0, G.stream>>>(x, pos_, n, D);

    const int L = (int)layers_.size();
    const bool want_pool = pooled && proj_.p;
    for (int li = 0; li < L; li++) {
        if (li == L - 1) {  // x is now hidden_states[-2]
            CK(cudaMemcpyAsync(z, x, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, G.stream));
            if (!want_pool) break;
        }
        const Layer& ly = layers_[li];
        layer_norm(a, x, ly.ln1.w, ly.ln1.b, T, D, 1e-5f);
        const float *qp, *kp, *vp;
        int ld;
        if (ly.qkv.w.p) {
            linear(q, a, T, ly.qkv.w);
            bias_rows(q, ly.qkv.b, T, 3 * D);
            qp = q; kp = q + D; vp = q + 2 * D; ld = 3 * D;
        } else {
            float* k = q + (size_t)T * D;
            float* v = q + (size_t)2 * T * D;
            linear(q, a, T, ly.q.w); bias_rows(q, ly.q.b, T, D);
            linear(k, a, T, ly.k.w); bias_rows(k, ly.k.b, T, D);
            linear(v, a, T, ly.v.w); bias_rows(v, ly.v.b, T, D);
            qp = q; kp = k; vp = v; ld = D;
        }
        for (int c = 0; c < n; c++) {  // each chunk is its own causal sequence
            const size_t r = (size_t)c * 77;
            attention(o + r * D, D, qp + r * ld, ld, kp + r * ld, ld, vp + r * ld, ld, 77, 77, heads_, 64, true);
        }
        linear(x, o, T, ly.o.w, nullptr, 1.f);
        bias_rows(x, ly.o.b, T, D);
        layer_norm(a, x, ly.ln2.w, ly.ln2.b, T, D, 1e-5f);
        linear(f, a, T, ly.fc1.w);
        bias_rows(f, ly.fc1.b, T, 4 * D);
        if (quick_gelu_) quick_gelu(f, (size_t)T * 4 * D);
        else gelu(f, (size_t)T * 4 * D);
        linear(x, f, T, ly.fc2.w, nullptr, 1.f);
        bias_rows(x, ly.fc2.b, T, D);
    }
    if (want_pool) {
        layer_norm(a, x + (size_t)eos * D, ln_final_.w, ln_final_.b, 1, D, 1e-5f);
        linear(pooled, a, 1, proj_);
    }
    if (weighted) {
        float* w = G.arena.f(weights.size());
        CK(cudaMemcpyAsync(w, weights.data(), weights.size() * 4, cudaMemcpyHostToDevice, G.stream));
        k_token_weights<<<nblocks((size_t)chunks * 77 * D), 256, 0, G.stream>>>(hidden, z, z + (size_t)chunks * 77 * D, w, chunks * 77, D);
    }
    G.arena.release(m);
}
