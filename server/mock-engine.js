#!/usr/bin/env node
'use strict';
// Kiln mock engine: speaks the exact kiln-engine stdin/stdout JSON-lines protocol.
// Fakes ~1 s per step (KILN_MOCK_STEP_MS), emits low-res previews that sharpen as the
// "denoise" progresses, and writes a seeded plasma/gradient image as raw RGB8.

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const STEP_MS = Number(process.env.KILN_MOCK_STEP_MS || 1000);
const LOAD_MS = Number(process.env.KILN_MOCK_LOAD_MS || 600);
// KILN_MOCK_NO_FACE=1: engine reports features.face=false; =silent: face requests are
// silently ignored (like an engine build without face support that doesn't say so).
const NO_FACE = process.env.KILN_MOCK_NO_FACE || '';
// SDXL (family "sdxl"): switching model family "swaps VRAM" like the real engine (20-35 s there)
const SWAP_MS = Number(process.env.KILN_MOCK_SWAP_MS || 1200);
const SDXL_SAMPLERS = ['euler', 'euler_ancestral', 'dpmpp_2m', 'res_multistep'];
const SDXL_SCHEDULERS = ['normal', 'karras', 'simple', 'sgm_uniform', 'exponential', 'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal'];
let family = 'anima';

function send(obj) { process.stdout.write(JSON.stringify(obj) + '\n'); }
function log(msg) { send({ ev: 'log', msg: String(msg) }); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ---- deterministic RNG ----------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function seedRng(seed) {
  const s = Math.abs(Math.floor(Number(seed) || 0));
  const lo = s % 4294967296, hi = Math.floor(s / 4294967296);
  return mulberry32((lo ^ Math.imul(hi, 0x9E3779B1) ^ 0x5bd1e995) >>> 0);
}

function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => { const k = (n + h * 12) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)];
}

function renderImage(seed, w, h) {
  const rnd = seedRng(seed);
  const baseHue = rnd();
  const pal = [0, 1, 2, 3].map(i => hsl((baseHue + i * (0.08 + rnd() * 0.18)) % 1, 0.55 + rnd() * 0.35, 0.25 + i * 0.14 + rnd() * 0.08));
  const waves = [];
  for (let i = 0; i < 5; i++) waves.push({ fx: (rnd() * 2 - 1) * 9, fy: (rnd() * 2 - 1) * 9, ph: rnd() * 6.283, a: 0.4 + rnd() });
  const blobs = [];
  for (let i = 0; i < 9; i++) blobs.push({ x: rnd(), y: rnd(), r: 0.04 + rnd() * 0.16, c: hsl((baseHue + 0.5 + rnd() * 0.3) % 1, 0.7, 0.6 + rnd() * 0.3), a: 0.25 + rnd() * 0.5 });
  const grain = seedRng(Number(seed) + 7);
  const out = Buffer.alloc(w * h * 3);
  const asp = w / h;
  for (let y = 0; y < h; y++) {
    const v = y / h;
    for (let x = 0; x < w; x++) {
      const u = x / w;
      let s = 0;
      for (const wv of waves) s += wv.a * Math.sin(wv.fx * u * asp + wv.fy * v + wv.ph);
      s = 0.5 + 0.5 * Math.tanh(s * 0.6 + (v - 0.5) * 1.5);
      const t = Math.min(2.999, s * 3);
      const i0 = Math.floor(t), f = t - i0;
      let r = pal[i0][0] * (1 - f) + pal[i0 + 1][0] * f;
      let g = pal[i0][1] * (1 - f) + pal[i0 + 1][1] * f;
      let b = pal[i0][2] * (1 - f) + pal[i0 + 1][2] * f;
      for (const bl of blobs) {
        const dx = (u - bl.x) * asp, dy = v - bl.y;
        const d2 = (dx * dx + dy * dy) / (bl.r * bl.r);
        if (d2 < 4) {
          const k = bl.a * Math.exp(-d2 * 1.6);
          r += (bl.c[0] - r) * k; g += (bl.c[1] - g) * k; b += (bl.c[2] - b) * k;
        }
      }
      const vig = 1 - 0.35 * ((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2;
      const n = (grain() - 0.5) * 0.04;
      const o = (y * w + x) * 3;
      out[o] = Math.max(0, Math.min(255, (r * vig + n) * 255));
      out[o + 1] = Math.max(0, Math.min(255, (g * vig + n) * 255));
      out[o + 2] = Math.max(0, Math.min(255, (b * vig + n) * 255));
    }
  }
  return out;
}

function downscale(rgb, w, h, tw, th) {
  const out = Buffer.alloc(tw * th * 3);
  for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
    const x0 = Math.floor(tx * w / tw), x1 = Math.floor((tx + 1) * w / tw);
    const y0 = Math.floor(ty * h / th), y1 = Math.floor((ty + 1) * h / th);
    let r = 0, g = 0, b = 0, n = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const p = (y * w + x) * 3; r += rgb[p]; g += rgb[p + 1]; b += rgb[p + 2]; n++;
    }
    const o = (ty * tw + tx) * 3;
    out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n;
  }
  return out;
}

// ---- job handling -----------------------------------------------------------
let current = null; // {id, cancelled}

class Cancelled extends Error { }

// Sleep ~ms in slices so cancel stays responsive.
async function work(job, ms) {
  const ts = Date.now();
  while (Date.now() - ts < ms) {
    if (job.cancelled) throw new Cancelled();
    await sleep(Math.min(50, ms - (Date.now() - ts)));
  }
  if (job.cancelled) throw new Cancelled();
  return Date.now() - ts;
}

// Blend `img` toward `noiseBuf` (same size) by 1-a.
function mix(img, noiseBuf, a) {
  const out = Buffer.alloc(img.length);
  for (let i = 0; i < out.length; i++) out[i] = img[i] * a + noiseBuf[i] * (1 - a);
  return out;
}
function noiseOf(seed, n) {
  const r = seedRng(seed);
  const b = Buffer.alloc(n);
  for (let i = 0; i < n; i++) b[i] = r() * 255;
  return b;
}
function crop(rgb, w, h, x0, y0, cw, ch) {
  const out = Buffer.alloc(cw * ch * 3);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const sx = Math.min(w - 1, Math.max(0, x0 + x)), sy = Math.min(h - 1, Math.max(0, y0 + y));
    const s = (sy * w + sx) * 3, d = (y * cw + x) * 3;
    out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2];
  }
  return out;
}
function upscaleNearest(rgb, w, h, k) {
  const W = w * k, H = h * k;
  const out = Buffer.alloc(W * H * 3);
  for (let y = 0; y < H; y++) {
    const sy = (y / k) | 0;
    let d = y * W * 3;
    for (let x = 0; x < W; x++, d += 3) {
      const s = (sy * w + ((x / k) | 0)) * 3;
      out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2];
    }
  }
  return out;
}
// "Redraw" a face: a soft bright disc with a ring, so mock outputs visibly show face passes.
function paintFace(rgb, w, h, cx, cy, r) {
  for (let y = Math.max(0, cy - r); y < Math.min(h, cy + r); y++) for (let x = Math.max(0, cx - r); x < Math.min(w, cx + r); x++) {
    const d = Math.hypot(x - cx, y - cy) / r;
    if (d > 1) continue;
    const o = (y * w + x) * 3;
    const ring = Math.abs(d - 0.92) < 0.06 ? 1 : 0;
    const k = ring ? 0.9 : 0.35 * (1 - d);
    const c = ring ? [255, 200, 120] : [255, 235, 215];
    for (let i = 0; i < 3; i++) rgb[o + i] = rgb[o + i] * (1 - k) + c[i] * k;
  }
}

async function generate(req) {
  const id = req.id;
  if (current) { send({ id, ev: 'error', msg: 'engine busy' }); return; }
  const job = { id, cancelled: false };
  current = job;
  try {
    const w = req.width | 0, h = req.height | 0;
    const fam = req.family === 'sdxl' ? 'sdxl' : 'anima';
    if (fam === 'sdxl') {
      if (w < 32 || h < 32 || w % 32 || h % 32) throw new Error(`SDXL size ${w}x${h} is not a multiple of 32`);
      if (!req.checkpoint || !fs.existsSync(req.checkpoint)) throw new Error('SDXL checkpoint not found: ' + req.checkpoint);
      const chunks = (x) => x && x.l && x.g && Array.isArray(x.l.ids) && Array.isArray(x.g.ids) && x.l.ids.every(c => c.length === 77) && x.l.ids.length === x.g.ids.length && Array.isArray(x.l.weights);
      if (!req.sd || !chunks(req.sd.pos) || !chunks(req.sd.neg)) throw new Error('SDXL job needs sd.pos / sd.neg {l, g} token chunks');
      if (!SDXL_SAMPLERS.includes(req.sampler)) throw new Error('SDXL sampler not supported: ' + req.sampler);
      if (!SDXL_SCHEDULERS.includes(req.scheduler)) throw new Error('SDXL scheduler not supported: ' + req.scheduler);
      if (req.loras && req.loras.length) log('SDXL: LoRAs are not supported yet; rendering without them');
      if (req.face) log('SDXL: face detailer is not supported yet; rendering without it');
    } else {
      if (w < 16 || h < 16 || w % 16 || h % 16) throw new Error(`bad size ${w}x${h}`);
      if (!Array.isArray(req.qwen_ids) || !Array.isArray(req.t5_ids)) throw new Error('missing token ids');
      if (req.t5_weights && req.t5_weights.length !== req.t5_ids.length) throw new Error('t5_weights length mismatch');
    }
    if (fam !== family) {
      for (const p of [0, 0.25, 0.5, 0.75, 1]) { send({ id, ev: 'loading', what: fam === 'sdxl' ? 'SDXL checkpoint' : 'Anima model', progress: p }); if (p < 1) await work(job, SWAP_MS / 4); }
      family = fam;
    }
    if (!req.out) throw new Error('missing out path');
    const steps = Math.max(1, req.steps | 0);
    const seed = Number(req.seed) || 0;
    const cfgMul = req.cfg > 1 ? 1.9 : 1;
    // NAG (negative prompt at CFG 1): a few % per sampling step
    const nag = req.nag && req.nag.enabled !== false ? req.nag : null;
    const nagMul = nag ? 1.05 : 1;
    // cfg_cutoff: the negative pass only runs for the first ceil(cutoff*steps) base steps
    const cutoff = Math.min(1, Math.max(0, Number(req.cfg_cutoff) || 1));
    const cfgSteps = Math.ceil(cutoff * steps);
    // step_cache: pretend some middle steps were served from the cache (never the first 2 or the last)
    const cache = Math.min(0.5, Math.max(0, Number(req.step_cache) || 0));
    const skipped = new Set();
    if (cache > 0 && steps >= 4) {
      const r = seedRng(seed + 99);
      const p = Math.min(0.9, cache * 4);
      for (let s = 3; s <= steps - 1; s++) if (r() < p) skipped.add(s);
    }
    const preview = req.preview !== false;
    const timings = { mock: true };
    if (req.cfg > 1 && req.cfg_cutoff != null) timings.cfg_cutoff = Number(req.cfg_cutoff);
    if (nag) timings.nag_scale = Number(nag.scale);
    const t0 = Date.now();
    await sleep(40 + (req.neg ? 30 : 0));
    timings.encode_ms = Date.now() - t0;
    send({ id, ev: 'encoded', ms: timings.encode_ms });

    // ---- base
    const hires = req.hires && Number(req.hires.scale) > 1 ? req.hires : null;
    const hw = hires ? Math.round(w * hires.scale / 16) * 16 : w;
    const hh = hires ? Math.round(h * hires.scale / 16) * 16 : h;
    let img = renderImage(seed, hw, hh);          // the picture we'll converge to
    const pw = Math.max(1, Math.round(w / 8)), ph = Math.max(1, Math.round(h / 8));
    const baseSmall = downscale(img, hw, hh, pw, ph);
    const baseNoise = noiseOf(seed + 1, pw * ph * 3);
    timings.steps_ms = [];
    for (let s = 1; s <= steps; s++) {
      const ms = await work(job, STEP_MS * (0.9 + 0.2 * Math.random()) * (s - 1 < cfgSteps ? cfgMul : 1) * nagMul * (skipped.has(s) ? 0.25 : 1));
      timings.steps_ms.push(ms);
      const ev = { id, ev: 'step', stage: 'base', step: s, of: steps, ms };
      if (preview) ev.preview = { w: pw, h: ph, rgb: mix(baseSmall, baseNoise, Math.pow(s / steps, 0.8)).toString('base64') };
      send(ev);
    }

    if (cache > 0) timings.cache_skips = skipped.size;

    // ---- hires fix (bigger latent; starts from the upscaled base, light renoise)
    if (hires) {
      const hs = Math.max(1, hires.steps | 0 || 4);
      const hpw = Math.max(1, Math.round(hw / 8)), hph = Math.max(1, Math.round(hh / 8));
      const hSmall = downscale(img, hw, hh, hpw, hph);
      const hNoise = noiseOf(seed + 2, hpw * hph * 3);
      const th = Date.now();
      await work(job, 120 * hires.scale); // upscale + re-encode
      const dn = Math.min(1, Math.max(0, Number(hires.denoise) || 0.4));
      for (let s = 1; s <= hs; s++) {
        const ms = await work(job, STEP_MS * hires.scale * hires.scale * (0.9 + 0.2 * Math.random()) * cfgMul);
        const ev = { id, ev: 'step', stage: 'hires', step: s, of: hs, ms };
        if (preview) ev.preview = { w: hpw, h: hph, rgb: mix(hSmall, hNoise, 1 - dn * 0.5 * (1 - s / hs)).toString('base64') };
        send(ev);
      }
      timings.hires_ms = Date.now() - th;
    }

    // ---- face detail (once per detected face)
    const wantFace = fam !== 'sdxl' && req.face && req.face.enabled && NO_FACE !== 'silent' && !NO_FACE;
    if (wantFace) {
      const f = req.face;
      const rnd = seedRng(seed + 3);
      const count = Math.min(Math.max(1, f.max_faces | 0 || 4), 1 + Math.floor(rnd() * 2)); // 1-2 "faces"
      const fs_ = Math.max(1, f.steps | 0 || 4);
      const guide = Math.max(64, f.guide | 0 || 384);
      const tf = Date.now();
      for (let k = 1; k <= count; k++) {
        const r = Math.round(Math.min(hw, hh) * (0.08 + rnd() * 0.06));
        const cx = Math.round(hw * (0.25 + rnd() * 0.5)), cy = Math.round(hh * (0.15 + rnd() * 0.3));
        const side = Math.round(r * (Number(f.crop) || 2));
        const fpw = Math.max(8, Math.round(guide / 8)), fph = fpw;
        const faceCrop = crop(img, hw, hh, cx - side, cy - side, side * 2, side * 2);
        const painted = Buffer.from(faceCrop);
        paintFace(painted, side * 2, side * 2, side, side, r);
        const small = downscale(painted, side * 2, side * 2, fpw, fph);
        const nz = noiseOf(seed + 10 + k, fpw * fph * 3);
        for (let s = 1; s <= fs_; s++) {
          const ms = await work(job, Math.max(80, STEP_MS * (guide * guide) / (w * h)) * (0.9 + 0.2 * Math.random()) * cfgMul);
          const ev = { id, ev: 'step', stage: 'face', step: s, of: fs_, ms };
          if (preview) ev.preview = { w: fpw, h: fph, rgb: mix(small, nz, 0.6 + 0.4 * s / fs_).toString('base64') };
          send(ev);
        }
        paintFace(img, hw, hh, cx, cy, r);
      }
      timings.faces = count;
      timings.face_ms = Date.now() - tf;
    }

    // ---- VAE decode
    const td = Date.now();
    await work(job, Math.min(800, 150 + hw * hh / 1500));
    timings.decode_ms = Date.now() - td;
    send({ id, ev: 'decoded', ms: timings.decode_ms });

    // ---- final upscale
    let W = hw, H = hh;
    const factor = req.upscale && (req.upscale.factor === 2 || req.upscale.factor === 4) ? req.upscale.factor : 1;
    if (factor > 1) {
      const tu = Date.now();
      await work(job, 250 * factor);
      img = upscaleNearest(img, hw, hh, factor);
      W = hw * factor; H = hh * factor;
      timings.upscale_ms = Date.now() - tu;
    }

    fs.mkdirSync(path.dirname(req.out), { recursive: true });
    fs.writeFileSync(req.out, img);
    send({ id, ev: 'done', out: req.out, w: W, h: H, total_ms: Date.now() - t0, timings });
  } catch (e) {
    if (e instanceof Cancelled) send({ id, ev: 'cancelled' });
    else send({ id, ev: 'error', msg: e.message });
  } finally {
    if (current === job) current = null;
  }
}

// ---- node graphs (cmd "graph") ------------------------------------------------------
let CATALOG = null;
function catalog() {
  if (!CATALOG) {
    try { CATALOG = JSON.parse(fs.readFileSync(path.join(__dirname, 'nodes.json'), 'utf8')); } catch (_) { CATALOG = {}; }
  }
  return CATALOG;
}
const isLinkV = (v) => Array.isArray(v) && v.length === 2 && Number.isInteger(v[1]);

// ---- pack nodes (docs/NODE_API.md): ext_call / ext_result, ext_op / ext_op_done --------
// Mock values: IMAGE {w,h,batch,seed?,data?} (data = Float32Array [B,H,W,3]), MASK {mask,w,h,batch,data?},
// LATENT {w,h,batch,data?} (w/h in pixels; data = [B,16,h/8,w/8]), handles = anything else.
const pendingCalls = new Map();   // call -> { resolve, job, node }
const handles = new Map();        // engine handle id -> mock value
let callSeq = 0, handleSeq = 0, fileSeq = 0;
const PRIM = new Set(['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO']);
function f32File(dir, id, data) {
  const p = path.join(dir, `${id}_m${++fileSeq}.f32`);
  fs.writeFileSync(p, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  return p;
}
function readF32(w, what) {
  if (!w || typeof w.path !== 'string' || !Array.isArray(w.shape)) throw new Error(`${what}: expected {type, path, shape}`);
  const n = w.shape.reduce((a, b) => a * b, 1);
  const buf = fs.readFileSync(w.path);
  if (buf.length !== n * 4) throw new Error(`${what}: ${path.basename(w.path)} has ${buf.length} bytes, shape ${JSON.stringify(w.shape)} needs ${n * 4}`);
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + n * 4));
}
function imageData(im, seed) {
  if (im.data) return im.data;
  const B = im.batch || 1, n = im.w * im.h * 3, out = new Float32Array(B * n);
  for (let b = 0; b < B; b++) {
    const rgb = renderImage((im.seed != null ? im.seed : seed) + b * 101, im.w, im.h);
    for (let i = 0; i < n; i++) out[b * n + i] = rgb[i] / 255;
  }
  return out;
}
function latentDims(l) { return [l.batch || 1, 16, Math.max(1, Math.round(l.h / 8)), Math.max(1, Math.round(l.w / 8))]; }
function noiseF32(seed, n) { const r = seedRng(seed); const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (r() - 0.5) * 2; return a; }
function toWire(type, v, dir, id, seed) {
  if (type === '*') type = v && typeof v === 'object' ? (v.__type || 'OPAQUE') : 'PRIM';
  if (PRIM.has(type) || type === 'PRIM') return v;
  if (type === 'IMAGE') return { type, path: f32File(dir, id, imageData(v, seed)), shape: [v.batch || 1, v.h, v.w, 3] };
  if (type === 'MASK') {
    const B = v.batch || 1, data = v.data || new Float32Array(B * v.h * v.w);
    return { type, path: f32File(dir, id, data), shape: [B, v.h, v.w] };
  }
  if (type === 'LATENT') {
    const shape = latentDims(v);
    return { type, path: f32File(dir, id, v.data || noiseF32(seed + 3, shape.reduce((a, b) => a * b, 1))), shape };
  }
  if (v && v.__wire) return v.__wire;   // an opaque handle a pack node returned (e.g. "js:3")
  const h = 'h' + (++handleSeq);
  handles.set(h, v);
  return { type, handle: h };
}
function fromWire(w, what) {
  if (w === null || typeof w !== 'object' || Array.isArray(w)) return w;
  if (typeof w.path === 'string') {
    const data = readF32(w, what);
    const sh = w.shape;
    if (w.type === 'IMAGE') { if (sh.length !== 4 || sh[3] !== 3) throw new Error(`${what}: IMAGE shape must be [B,H,W,3]`); return { __type: 'IMAGE', w: sh[2], h: sh[1], batch: sh[0], data }; }
    if (w.type === 'MASK') { if (sh.length !== 3) throw new Error(`${what}: MASK shape must be [B,H,W]`); return { __type: 'MASK', mask: true, w: sh[2], h: sh[1], batch: sh[0], data }; }
    if (w.type === 'LATENT') { if (sh.length !== 4) throw new Error(`${what}: LATENT shape must be [B,C,h,w]`); return { __type: 'LATENT', w: sh[3] * 8, h: sh[2] * 8, batch: sh[0], data }; }
    throw new Error(`${what}: unknown tensor type ${w.type}`);
  }
  if (w.handle != null) {
    if (handles.has(w.handle)) return handles.get(w.handle);
    return { __type: w.type, __wire: { type: w.type, handle: w.handle } };
  }
  return w;
}
// nearest / bilinear resample of [B,H,W,3]
function resampleImage(data, B, H, W, h2, w2, nearest) {
  const out = new Float32Array(B * h2 * w2 * 3);
  for (let b = 0; b < B; b++) for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    const o = ((b * h2 + y) * w2 + x) * 3;
    if (nearest) {
      const sy = Math.min(H - 1, Math.floor(y * H / h2)), sx = Math.min(W - 1, Math.floor(x * W / w2));
      const i = ((b * H + sy) * W + sx) * 3;
      out[o] = data[i]; out[o + 1] = data[i + 1]; out[o + 2] = data[i + 2];
    } else {
      const fy = Math.max(0, Math.min(H - 1, (y + 0.5) * H / h2 - 0.5)), fx = Math.max(0, Math.min(W - 1, (x + 0.5) * W / w2 - 0.5));
      const y0 = Math.floor(fy), x0 = Math.floor(fx), y1 = Math.min(H - 1, y0 + 1), x1 = Math.min(W - 1, x0 + 1), ty = fy - y0, tx = fx - x0;
      for (let c = 0; c < 3; c++) {
        const p = (yy, xx) => data[((b * H + yy) * W + xx) * 3 + c];
        out[o + c] = (p(y0, x0) * (1 - tx) + p(y0, x1) * tx) * (1 - ty) + (p(y1, x0) * (1 - tx) + p(y1, x1) * tx) * ty;
      }
    }
  }
  return out;
}
function resizeImageValue(im, w2, h2, nearest, seed) {
  if (!im.data) return { w: w2, h: h2, batch: im.batch || 1, seed: im.seed };
  return { w: w2, h: h2, batch: im.batch || 1, data: resampleImage(im.data, im.batch || 1, im.h, im.w, h2, w2, nearest) };
}

async function extCall(ctx, nid, ct, n) {
  const { req, job, out, outTypes, baseSeed } = ctx;
  const call = ++callSeq;
  const inputs = {};
  for (const [k, v] of Object.entries(n.inputs || {})) {
    if (isLinkV(v)) {
      const src = String(v[0]);
      const val = out[src] ? out[src][v[1]] : undefined;
      if (val === undefined) continue;
      inputs[k] = toWire(outTypes[src][v[1]], val, req.out_dir, req.id, baseSeed + Number(src));
    } else inputs[k] = v;
  }
  const res = await new Promise((resolve) => {
    pendingCalls.set(call, { resolve, job, node: nid });
    send({ id: req.id, ev: 'ext_call', call, node: nid, class_type: ct, inputs });
  });
  if (job.cancelled) throw new Cancelled();
  if (res.error != null) throw Object.assign(new Error(String(res.error)), { node: nid });
  const types = req.ext[ct].outputs || [];
  if (!Array.isArray(res.outputs) || res.outputs.length !== types.length) throw Object.assign(new Error(`ext_result has ${res.outputs && res.outputs.length} outputs, expected ${types.length}`), { node: nid });
  out[nid] = res.outputs.map((w, i) => fromWire(w, `${ct} output ${i}`));
  outTypes[nid] = res.outputs.map((w, i) => (types[i] !== '*' ? types[i] : w && typeof w === 'object' && !Array.isArray(w) ? w.type : 'PRIM'));
}

async function handleExtOp(req) {
  const { id, op_id, op } = req;
  const pc = pendingCalls.get(req.call);
  const t0 = Date.now();
  try {
    if (!current || current.id !== id || !pc) throw new Error(`no open ext_call ${req.call} for job ${id}`);
    const job = pc.job, dir = current.outDir;
    const img = (w, what) => { if (!w || w.type !== 'IMAGE') throw new Error(`${what} must be an IMAGE`); return { data: readF32(w, what), B: w.shape[0], H: w.shape[1], W: w.shape[2] }; };
    const hnd = (w, type, what) => { if (!w || w.type !== type || !handles.has(w.handle)) throw new Error(`${what} must be a ${type} handle from this graph`); return handles.get(w.handle); };
    let result;
    switch (op) {
      case 'resize': {
        const im = img(req.image, 'resize.image');
        const w2 = req.width | 0, h2 = req.height | 0;
        if (w2 < 1 || h2 < 1) throw new Error('resize: bad size');
        if (!['nearest-exact', 'bilinear', 'bicubic', 'area', 'lanczos'].includes(req.method)) throw new Error('resize: unknown method ' + req.method);
        await work(job, 30);
        result = { type: 'IMAGE', path: f32File(dir, id, resampleImage(im.data, im.B, im.H, im.W, h2, w2, req.method === 'nearest-exact')), shape: [im.B, h2, w2, 3] };
        break;
      }
      case 'upscale': {
        hnd(req.model, 'UPSCALE_MODEL', 'upscale.model');
        const im = img(req.image, 'upscale.image');
        await work(job, 200);
        result = { type: 'IMAGE', path: f32File(dir, id, resampleImage(im.data, im.B, im.H, im.W, im.H * 4, im.W * 4, true)), shape: [im.B, im.H * 4, im.W * 4, 3] };
        break;
      }
      case 'vae_decode': {
        hnd(req.vae, 'VAE', 'vae_decode.vae');
        if (!req.latent || req.latent.type !== 'LATENT') throw new Error('vae_decode.latent must be a LATENT');
        readF32(req.latent, 'vae_decode.latent');
        const [B, , h, w] = req.latent.shape;
        await work(job, 150);
        result = { type: 'IMAGE', path: f32File(dir, id, imageData({ w: w * 8, h: h * 8, batch: B }, 4242 + op_id)), shape: [B, h * 8, w * 8, 3] };
        break;
      }
      case 'vae_encode': {
        hnd(req.vae, 'VAE', 'vae_encode.vae');
        const im = img(req.image, 'vae_encode.image');
        const shape = [im.B, 16, Math.max(1, Math.floor(im.H / 8)), Math.max(1, Math.floor(im.W / 8))];
        await work(job, 100);
        result = { type: 'LATENT', path: f32File(dir, id, noiseF32(op_id, shape.reduce((a, b) => a * b, 1))), shape };
        break;
      }
      case 'detect_faces': {
        const im = img(req.image, 'detect_faces.image');
        if (req.detector) hnd(req.detector, 'BBOX_DETECTOR', 'detect_faces.detector');
        await work(job, 60);
        const th = Number(req.threshold);
        result = Array.from({ length: im.B }, () => (0.9 >= th ? [{ x0: Math.round(im.W * 0.35), y0: Math.round(im.H * 0.2), x1: Math.round(im.W * 0.65), y1: Math.round(im.H * 0.45), score: 0.9 }] : []));
        break;
      }
      case 'encode_text': {
        hnd(req.clip, 'CLIP', 'encode_text.clip');
        if (!Array.isArray(req.t5_ids) || !Array.isArray(req.qwen_ids) || typeof req.text !== 'string') throw new Error('encode_text needs text, qwen_ids and t5_ids');
        await work(job, 30);
        const h = 'h' + (++handleSeq);
        handles.set(h, { cond: req.t5_ids.length, text: req.text });
        result = { type: 'CONDITIONING', handle: h };
        break;
      }
      case 'sample': {
        hnd(req.model, 'MODEL', 'sample.model');
        hnd(req.positive, 'CONDITIONING', 'sample.positive');
        hnd(req.negative, 'CONDITIONING', 'sample.negative');
        if (!req.latent || req.latent.type !== 'LATENT') throw new Error('sample.latent must be a LATENT');
        readF32(req.latent, 'sample.latent');
        if (req.mask) readF32(req.mask, 'sample.mask');
        const steps = Math.max(1, req.steps | 0);
        for (let st = 1; st <= steps; st++) {
          const ms = await work(job, STEP_MS * (Number(req.cfg) > 1 ? 1.9 : 1) * (0.9 + 0.2 * Math.random()));
          send({ id, ev: 'step', node: pc.node, step: st, of: steps, ms });
        }
        const shape = req.latent.shape;
        result = { type: 'LATENT', path: f32File(dir, id, noiseF32((req.seed | 0) + 17, shape.reduce((a, b) => a * b, 1))), shape: shape.slice() };
        break;
      }
      default:
        throw new Error('unknown ext_op ' + op);
    }
    send({ id, ev: 'ext_op_done', op_id, result, ms: Date.now() - t0 });
  } catch (e) {
    send({ id, ev: 'ext_op_done', op_id, error: e instanceof Cancelled ? 'cancelled' : e.message });
  }
}

async function runGraph(req) {
  const id = req.id;
  if (current) { send({ id, ev: 'error', node: null, msg: 'engine busy' }); return; }
  const job = { id, cancelled: false };
  current = job;
  const t0 = Date.now();
  const nodeMs = {};
  let curNode = null;
  try {
    const g = req.graph || {};
    const cat = catalog();
    // fail fast on unsupported types
    const ext = req.ext || {};
    for (const [nid, n] of Object.entries(g)) {
      if (!cat[n.class_type] && !ext[n.class_type]) { send({ id, ev: 'error', node: nid, msg: `unsupported node type ${n.class_type}` }); return; }
    }
    job.outDir = req.out_dir;
    // topo sort (Kahn)
    const deps = {}, users = {};
    for (const [nid, n] of Object.entries(g)) {
      deps[nid] = new Set();
      for (const v of Object.values(n.inputs || {})) if (isLinkV(v) && g[String(v[0])]) deps[nid].add(String(v[0]));
    }
    for (const [nid, ds] of Object.entries(deps)) for (const d of ds) (users[d] = users[d] || []).push(nid);
    const indeg = Object.fromEntries(Object.entries(deps).map(([k, v]) => [k, v.size]));
    const order = [], ready = Object.keys(indeg).filter(k => !indeg[k]).sort((a, b) => Number(a) - Number(b));
    while (ready.length) {
      const k = ready.shift();
      order.push(k);
      for (const u of users[k] || []) if (--indeg[u] === 0) ready.push(u);
    }
    if (order.length !== Object.keys(g).length) { send({ id, ev: 'error', node: null, msg: 'graph has a cycle' }); return; }

    const out = {}; // nid -> [output values]
    const outTypes = {}; // nid -> [output types] (for encoding values sent to pack nodes)
    const seeds = [];
    for (const n of Object.values(g)) for (const k of ['seed', 'noise_seed']) if (typeof (n.inputs || {})[k] === 'number') seeds.push(n.inputs[k]);
    const baseSeed = seeds.length ? seeds[0] : 1;
    const inp = (nid, name) => {
      const v = (g[nid].inputs || {})[name];
      if (isLinkV(v)) { const o = out[String(v[0])]; return o ? o[v[1]] : undefined; }
      return v;
    };
    const need = (nid, name) => {
      const v = inp(nid, name);
      if (v === undefined) throw Object.assign(new Error(`missing input "${name}"`), { node: nid });
      return v;
    };
    let loaded = false;
    for (const nid of order) {
      if (job.cancelled) { send({ id, ev: 'cancelled' }); return; }
      const n = g[nid], ct = n.class_type;
      // widget inputs may be linked to a pack node's INT/FLOAT/STRING/BOOLEAN output
      const I = {};
      for (const [k, v] of Object.entries(n.inputs || {})) {
        if (isLinkV(v)) { const src = out[String(v[0])]; const val = src ? src[v[1]] : undefined; I[k] = val !== null && typeof val !== 'object' && val !== undefined ? val : v; }
        else I[k] = v;
      }
      curNode = nid;
      const ts = Date.now();
      send({ id, ev: 'node', node: nid, class_type: ct, status: 'start' });
      if (ext[ct]) {
        await extCall({ req, job, out, outTypes, baseSeed }, nid, ct, n);
        loaded = true;
        nodeMs[nid] = Date.now() - ts;
        send({ id, ev: 'node', node: nid, class_type: ct, status: 'done', ms: nodeMs[nid] });
        continue;
      }
      const sampler = async (latent, steps, cfg, extra = 1) => {
        const pw = Math.max(1, Math.round(latent.w / 8)), ph = Math.max(1, Math.round(latent.h / 8));
        const small = downscale(renderImage(baseSeed + Number(nid), latent.w, latent.h), latent.w, latent.h, pw, ph);
        const nz = noiseOf(baseSeed + 7 + Number(nid), pw * ph * 3);
        for (let s = 1; s <= steps; s++) {
          const ms = await work(job, STEP_MS * (0.9 + 0.2 * Math.random()) * (cfg > 1 ? 1.9 : 1) * extra);
          const ev = { id, ev: 'step', node: nid, step: s, of: steps, ms };
          if (req.preview !== false) ev.preview = { w: pw, h: ph, rgb: mix(small, nz, Math.pow(s / steps, 0.8)).toString('base64') };
          send(ev);
        }
      };
      switch (ct) {
        case 'UNETLoader': case 'UnetLoaderGGUF': case 'CLIPLoader': case 'VAELoader': case 'UpscaleModelLoader': case 'UltralyticsDetectorProvider':
          await work(job, loaded ? 20 : 150);
          out[nid] = ct === 'UltralyticsDetectorProvider' ? [{ bbox: I.model_name }, { segm: I.model_name }] : [{ name: I.unet_name || I.clip_name || I.vae_name || I.model_name }];
          break;
        case 'LoraLoaderModelOnly': case 'ModelSamplingAuraFlow': case 'ModelSamplingSD3': case 'ApplyFBCacheOnModel':
          need(nid, 'model'); await work(job, 20); out[nid] = [{ model: true }];
          break;
        case 'LoraLoader':
          need(nid, 'model'); need(nid, 'clip'); await work(job, 20); out[nid] = [{ model: true }, { clip: true }];
          break;
        case 'CLIPTextEncode':
          need(nid, 'clip');
          if (!req.tokens || !req.tokens[nid]) throw Object.assign(new Error('no tokens for this CLIPTextEncode'), { node: nid });
          await work(job, 30); out[nid] = [{ cond: req.tokens[nid].t5_ids.length }];
          break;
        case 'EmptySD3LatentImage': case 'EmptyLatentImage':
          out[nid] = [{ w: I.width | 0, h: I.height | 0, batch: Math.max(1, I.batch_size | 0) }];
          break;
        case 'LoadImage': {
          const im = req.images && req.images[nid];
          if (!im) throw Object.assign(new Error('LoadImage without decoded image'), { node: nid });
          if (!fs.existsSync(im.path) || fs.statSync(im.path).size !== im.w * im.h * 4) throw Object.assign(new Error('raw RGBA file missing or wrong size'), { node: nid });
          await work(job, 40);
          out[nid] = [{ w: im.w, h: im.h, batch: 1 }, { mask: true, w: im.w, h: im.h }];
          break;
        }
        case 'VAEEncode': { const px = need(nid, 'pixels'); need(nid, 'vae'); await work(job, 200); out[nid] = [{ w: px.w, h: px.h, batch: px.batch }]; break; }
        case 'VAEDecode': { const l = need(nid, 'samples'); need(nid, 'vae'); await work(job, 300); out[nid] = [{ w: l.w, h: l.h, batch: l.batch, seed: baseSeed + Number(nid) }]; break; }
        case 'SetLatentNoiseMask': { const l = need(nid, 'samples'); need(nid, 'mask'); out[nid] = [l]; break; }
        case 'LatentUpscaleBy': { const l = need(nid, 'samples'); const k = Number(I.scale_by) || 1; out[nid] = [{ w: Math.round(l.w * k / 8) * 8, h: Math.round(l.h * k / 8) * 8, batch: l.batch }]; await work(job, 30); break; }
        case 'LatentUpscale': {
          const l = need(nid, 'samples'); let w = I.width | 0, h = I.height | 0;
          if (!w && !h) { w = l.w; h = l.h; } else if (!w) w = Math.round(l.w * h / l.h); else if (!h) h = Math.round(l.h * w / l.w);
          out[nid] = [{ w: Math.round(w / 8) * 8, h: Math.round(h / 8) * 8, batch: l.batch }]; await work(job, 30); break;
        }
        case 'KSampler': case 'KSamplerWithNAG': {
          need(nid, 'model'); need(nid, 'positive'); need(nid, 'negative'); const l = need(nid, 'latent_image');
          if (ct === 'KSamplerWithNAG') need(nid, 'nag_negative');
          await sampler(l, Math.max(1, I.steps | 0), Number(I.cfg), ct === 'KSamplerWithNAG' ? 1.05 : 1);
          out[nid] = [l];
          break;
        }
        case 'KSamplerAdvanced': {
          need(nid, 'model'); const l = need(nid, 'latent_image');
          const steps = Math.max(1, Math.min(I.steps | 0, I.end_at_step | 0) - (I.start_at_step | 0));
          await sampler(l, steps, Number(I.cfg));
          out[nid] = [l];
          break;
        }
        case 'ImageScaleBy': { const im = need(nid, 'image'); const k = Number(I.scale_by) || 1; out[nid] = [resizeImageValue(im, Math.max(1, Math.round(im.w * k)), Math.max(1, Math.round(im.h * k)), I.upscale_method === 'nearest-exact')]; await work(job, 60); break; }
        case 'ImageScale': {
          const im = need(nid, 'image'); let w = I.width | 0, h = I.height | 0;
          if (!w && !h) { w = im.w; h = im.h; } else if (!w) w = Math.round(im.w * h / im.h); else if (!h) h = Math.round(im.h * w / im.w);
          out[nid] = [resizeImageValue(im, w, h, I.upscale_method === 'nearest-exact')]; await work(job, 60); break;
        }
        case 'ImageUpscaleWithModel': { need(nid, 'upscale_model'); const im = need(nid, 'image'); await work(job, 400); out[nid] = [resizeImageValue(im, im.w * 4, im.h * 4, true)]; break; }
        case 'FaceDetailer': {
          const im = need(nid, 'image'); for (const k of ['model', 'clip', 'vae', 'positive', 'negative', 'bbox_detector']) need(nid, k);
          const faces = 1 + (Math.abs(baseSeed + Number(nid)) % 2);
          for (let f = 0; f < faces; f++) await sampler({ w: I.guide_size | 0 || 384, h: I.guide_size | 0 || 384 }, Math.max(1, I.steps | 0), Number(I.cfg));
          out[nid] = [im, { w: 256, h: 256, batch: 1 }, { w: 256, h: 256, batch: 1 }, { mask: true }, { pipe: true }, { w: 1, h: 1, batch: 1 }];
          break;
        }
        case 'SaveImage': case 'PreviewImage': {
          const im = need(nid, 'images');
          fs.mkdirSync(req.out_dir, { recursive: true });
          for (let b = 0; b < (im.batch || 1); b++) {
            let rgb;
            if (im.data) {
              const n3 = im.w * im.h * 3;
              rgb = Buffer.alloc(n3);
              for (let i = 0; i < n3; i++) rgb[i] = Math.max(0, Math.min(255, Math.round(im.data[b * n3 + i] * 255)));
            } else rgb = renderImage(baseSeed + b * 101 + Number(nid) * 7, im.w, im.h);
            const file = path.join(req.out_dir, `${id}_${nid}_${b}.rgb`);
            fs.writeFileSync(file, rgb);
            send({ id, ev: 'image', node: nid, index: b, kind: ct === 'SaveImage' ? 'save' : 'preview', prefix: ct === 'SaveImage' ? String(I.filename_prefix || 'Kiln') : undefined, out: file, w: im.w, h: im.h });
          }
          out[nid] = [];
          break;
        }
        default:
          throw Object.assign(new Error(`mock engine has no implementation for ${ct}`), { node: nid });
      }
      outTypes[nid] = (cat[ct].output || []).slice();
      loaded = true;
      nodeMs[nid] = Date.now() - ts;
      send({ id, ev: 'node', node: nid, class_type: ct, status: 'done', ms: nodeMs[nid] });
    }
    send({ id, ev: 'done', total_ms: Date.now() - t0, node_ms: nodeMs });
  } catch (e) {
    if (e instanceof Cancelled) send({ id, ev: 'cancelled' });
    else send({ id, ev: 'error', node: e.node != null ? e.node : curNode, msg: e.message });
  } finally {
    if (current === job) current = null;
    for (const [c, p] of pendingCalls) if (p.job === job) pendingCalls.delete(c);
    handles.clear();
  }
}

async function main() {
  const t0 = Date.now();
  for (const what of ['text_encoder', 'dit', 'vae']) {
    for (const p of [0, 0.5, 1]) {
      send({ id: '', ev: 'loading', what, progress: p });
      await sleep(LOAD_MS / 9);
    }
  }
  log(`kiln mock engine up in ${Date.now() - t0} ms (step ${STEP_MS} ms)`);
  send({ id: '', ev: 'ready' });

  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let req;
    try { req = JSON.parse(line); } catch (e) { log('bad json: ' + line.slice(0, 200)); return; }
    switch (req.cmd) {
      case 'generate': generate(req); break;
      case 'graph': runGraph(req); break;
      case 'ext_result': {
        const p = pendingCalls.get(req.call);
        if (p && current && req.id === current.id) { pendingCalls.delete(req.call); p.resolve(req); }
        else log(`ext_result for unknown call ${req.call}`);
        break;
      }
      case 'ext_op': handleExtOp(req); break;
      case 'cancel':
        if (current && (!req.id || current.id === req.id)) current.cancelled = true;
        break;
      case 'info':
        send({
          id: req.id, ev: 'info', engine: 'kiln-mock', version: 'mock-1', device: 'CPU (mock engine)',
          vram_mb: 0, samplers: ['euler', 'dpmpp_2m', 'res_multistep'], step_ms: STEP_MS, busy: !!current,
          features: NO_FACE === 'silent' ? { hires: true, upscale: true, ext: true, sdxl: true } : { hires: true, face: !NO_FACE, upscale: true, ext: true, sdxl: true }, family,
        });
        break;
      default:
        send({ id: req.id, ev: 'error', msg: 'unknown cmd ' + req.cmd });
    }
  });
  rl.on('close', () => process.exit(0));
}

main();
