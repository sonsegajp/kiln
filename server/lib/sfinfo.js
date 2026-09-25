'use strict';
// Model files: safetensors header -> role (model / lora / ...) + family (anima / sdxl / ...) + base
// model name (CivitAI's names: "Anima", "Illustrious", "NoobAI", "Pony", "SDXL 1.0", ...), SHA256
// hashing with a persistent cache, and CivitAI sidecars (<name>.civitai.json + <name>.preview.*).

const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const crypto = require('crypto');

// ---- CivitAI base model names <-> families ----------------------------------------------
const FAMILY_BASES = {
  anima: ['Anima'],
  sdxl: ['SDXL 1.0', 'Pony', 'Illustrious', 'NoobAI'],
  sd15: ['SD 1.5'],
  flux: ['Flux.1 D', 'Flux.1 S'],
};
const KNOWN_BASES = ['Anima', 'Illustrious', 'NoobAI', 'Pony', 'SDXL 1.0', 'SDXL Lightning', 'SDXL Hyper', 'SDXL Turbo', 'SD 1.5', 'Flux.1 D', 'Flux.1 S', 'Qwen', 'Wan Video', 'Other'];
function familyOfBase(base) {
  const b = String(base || '').toLowerCase();
  if (!b) return null;
  if (b === 'anima') return 'anima';
  if (/^(sdxl|pony|illustrious|noobai)/.test(b)) return 'sdxl';
  if (/^sd 1\./.test(b)) return 'sd15';
  if (/^flux/.test(b)) return 'flux';
  return 'other';
}
const FAMILY_LABEL = { anima: 'Anima', sdxl: 'SDXL', sd15: 'SD 1.5', flux: 'Flux', other: 'Other', unknown: '?' };

// finer SDXL base from free text (metadata titles, training base names, file names)
function sdxlBaseFromText(t) {
  const s = String(t || '').toLowerCase();
  if (/noob/.test(s)) return 'NoobAI';
  if (/illustrious|(^|[^a-z])ilxl|(^|[^a-z])il_?v\d/.test(s)) return 'Illustrious';
  if (/pony/.test(s)) return 'Pony';
  return null;
}

// ---- header ---------------------------------------------------------------------------------
const headerCache = new Map();   // abs path -> { size, mtime, info }
function readHeader(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const b = Buffer.alloc(8);
    if (fs.readSync(fd, b, 0, 8, 0) !== 8) throw new Error('too short');
    const n = Number(b.readBigUInt64LE(0));
    if (!(n > 1) || n > 100 * 1024 * 1024) throw new Error('not a safetensors file');
    const h = Buffer.alloc(n);
    fs.readSync(fd, h, 0, n, 8);
    return JSON.parse(h.toString('utf8'));
  } finally { fs.closeSync(fd); }
}

function classify(keys, meta) {
  const has = (re) => keys.some(k => re.test(k));
  const metaText = Object.values(meta || {}).filter(v => typeof v === 'string' && v.length < 4000).join(' ');
  let role = 'unknown', family = 'unknown';
  const lora = has(/lora_(A|B|up|down)\b|\.lora_(A|B|up|down)\.|\.alpha$|hada_w1|lokr_w1|\.dora_scale$/);
  if (lora) {
    role = 'lora';
    if (has(/llm_adapter|adaln_modulation/)) family = 'anima';
    else if (has(/lora_te2_|text_encoder_2|input_blocks|output_blocks|add_embedding|label_emb/)) family = 'sdxl';
    else if (has(/double_blocks|single_blocks|single_transformer_blocks/)) family = 'flux';
    else if (has(/lora_unet_down_blocks|lora_te_text_model|unet\.down_blocks/)) family = 'sd15';
    const ssBase = String((meta && (meta.ss_base_model_version || '')) || '').toLowerCase();
    const arch = String((meta && meta['modelspec.architecture']) || '').toLowerCase();
    if (/sdxl/.test(ssBase) || /stable-diffusion-xl/.test(arch)) family = 'sdxl';
    else if (/sd_v1|sd_v1/.test(ssBase) || /stable-diffusion-v1/.test(arch)) family = 'sd15';
    else if (/flux/.test(ssBase + arch)) family = 'flux';
  } else if (has(/^(net\.)?llm_adapter\./)) { role = 'model'; family = 'anima'; }
  else if (keys.includes('model.diffusion_model.label_emb.0.0.weight') || has(/^conditioner\.embedders\.1\./)) { role = 'model'; family = 'sdxl'; }
  else if (has(/^model\.diffusion_model\.input_blocks\./)) { role = 'model'; family = 'sd15'; }
  else if (has(/^(model\.diffusion_model\.)?double_blocks\./)) { role = 'model'; family = 'flux'; }
  else if (has(/^(model\.)?embed_tokens|^text_model\.encoder|^encoder\.block\.|^model\.layers\.0\./)) role = 'text_encoder';
  else if (has(/(^|\.)decoder\.(conv_in|up)/) && !has(/diffusion_model/)) role = 'vae';
  else if (has(/RDB\d|^body\.\d|^model\.1\.sub\./)) role = 'upscaler';
  let base = null;
  if (family === 'anima') base = 'Anima';
  else if (family === 'sdxl') base = sdxlBaseFromText(metaText + ' ' + ((meta && meta.ss_sd_model_name) || '')) || null;
  else if (family === 'sd15') base = 'SD 1.5';
  else if (family === 'flux') base = 'Flux.1 D';
  return { role, family, base };
}

// { role, family, base, baseSource } for a model file (cached by size + mtime)
function inspect(file) {
  let st;
  try { st = fs.statSync(file); } catch (_) { return { role: 'missing', family: 'unknown', base: null }; }
  const c = headerCache.get(file);
  if (c && c.size === st.size && c.mtime === st.mtimeMs) return c.info;
  let info;
  try {
    const h = readHeader(file);
    const meta = h.__metadata__ && typeof h.__metadata__ === 'object' ? h.__metadata__ : {};
    const keys = Object.keys(h).filter(k => k !== '__metadata__');
    info = classify(keys, meta);
    info.baseSource = info.base ? (info.family === 'sdxl' ? 'metadata' : 'family') : null;
    info.title = typeof meta['modelspec.title'] === 'string' ? meta['modelspec.title'].slice(0, 120) : undefined;
  } catch (e) {
    info = { role: 'unknown', family: 'unknown', base: null, error: e.message };
  }
  headerCache.set(file, { size: st.size, mtime: st.mtimeMs, info });
  return info;
}

// ---- sidecars ----------------------------------------------------------------------------------
const stemPath = (file) => file.replace(/\.(safetensors|sft)$/i, '');
const sidecarPath = (file) => stemPath(file) + '.civitai.json';
function readSidecar(file) {
  try { return JSON.parse(fs.readFileSync(sidecarPath(file), 'utf8')); } catch (_) { return null; }
}
function previewPath(file) {
  const s = stemPath(file);
  for (const ext of ['.preview.jpeg', '.preview.jpg', '.preview.png', '.preview.webp']) if (fs.existsSync(s + ext)) return s + ext;
  return null;
}

// ---- SHA256 cache ------------------------------------------------------------------------------
class Hasher {
  constructor(cacheFile, log) {
    this.file = cacheFile;
    this.log = log || (() => { });
    this.cache = {};
    try { this.cache = JSON.parse(fs.readFileSync(cacheFile, 'utf8')) || {}; } catch (_) { }
    this.queue = [];
    this.active = null;       // { file, done, total }
    this.waiters = new Map(); // file -> [resolve]
    this.onProgress = () => { };
  }
  cached(file) {
    try {
      const st = fs.statSync(file);
      const c = this.cache[file];
      return c && c.size === st.size && c.mtime === Math.floor(st.mtimeMs) ? c.sha256 : null;
    } catch (_) { return null; }
  }
  remember(file, sha256) {
    try {
      const st = fs.statSync(file);
      this.cache[file] = { size: st.size, mtime: Math.floor(st.mtimeMs), sha256: sha256.toUpperCase() };
      this.save();
    } catch (_) { }
  }
  save() {
    clearTimeout(this.saveT);
    this.saveT = setTimeout(() => { try { fs.mkdirSync(path.dirname(this.file), { recursive: true }); fs.writeFileSync(this.file, JSON.stringify(this.cache, null, 1)); } catch (_) { } }, 300);
  }
  // queue a file; resolves to the upper-case SHA256 (or null on error)
  hash(file) {
    const c = this.cached(file);
    if (c) return Promise.resolve(c);
    return new Promise((resolve) => {
      if (!this.waiters.has(file)) { this.waiters.set(file, []); this.queue.push(file); }
      this.waiters.get(file).push(resolve);
      this.pump();
    });
  }
  status() { return { active: this.active, queued: this.queue.length }; }
  async pump() {
    if (this.active || !this.queue.length) return;
    const file = this.queue.shift();
    let total = 0;
    try { total = fs.statSync(file).size; } catch (_) { }
    this.active = { file, done: 0, total };
    let sha = null;
    const t0 = Date.now();
    try {
      const h = crypto.createHash('sha256');
      let last = 0;
      await new Promise((resolve, reject) => {
        const rs = fs.createReadStream(file, { highWaterMark: 4 << 20 });
        rs.on('data', (b) => {
          h.update(b);
          this.active.done += b.length;
          if (Date.now() - last > 500) { last = Date.now(); this.onProgress(this.status()); }
        });
        rs.on('end', resolve);
        rs.on('error', reject);
      });
      sha = h.digest('hex').toUpperCase();
      this.remember(file, sha);
      this.log(`hashed ${path.basename(file)} (${(total / 1e9).toFixed(2)} GB) in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    } catch (e) { this.log(`hashing ${file} failed: ${e.message}`); }
    const ws = this.waiters.get(file) || [];
    this.waiters.delete(file);
    this.active = null;
    this.onProgress(this.status());
    for (const w of ws) w(sha);
    setImmediate(() => this.pump());
  }
}

module.exports = { inspect, classify, readSidecar, sidecarPath, previewPath, stemPath, Hasher, familyOfBase, sdxlBaseFromText, FAMILY_BASES, FAMILY_LABEL, KNOWN_BASES };
