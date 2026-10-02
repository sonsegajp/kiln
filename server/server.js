#!/usr/bin/env node
'use strict';
// Kiln server: static web UI + JSON API + engine supervisor + job queue.
// Zero npm dependencies.  Start: node server/server.js

const http = require('http');
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const os = require('os');
const { URL } = require('url');

const { encodePrompt, getT5Mode, warmup } = require('./lib/tokenize');
const { encodePNG, encodePNGAsync, readPNGText, readPNGSize, thumbnailPNG, thumbnailFromRGB, downscaleRGB } = require('./lib/png');
const { Engine } = require('./lib/engine');
const G = require('./lib/graph');
const { decodePNGRGBA, isPNG } = require('./lib/pngdecode');
const { Packs } = require('./lib/packs');
const X = require('./lib/extrun');
const A = require('./lib/a1111');
const SF = require('./lib/sfinfo');
const { CivitAI } = require('./lib/civitai');
const { createLibrary } = require('./lib/library');
const { createDatasets } = require('./lib/datasets');
const { createTaggers } = require('./lib/taggers');
const { createGrab } = require('./lib/grab');
const { encodeSDXL } = require('./lib/clip_tokenize');
const SDXL_SAMPLERS = ['euler', 'euler_ancestral', 'dpmpp_2m', 'res_multistep'];
const SDXL_SCHEDULERS = ['normal', 'karras', 'simple', 'sgm_uniform', 'exponential', 'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal'];
// SDXL renders work but are still far too slow on low-end cards (minutes per image), so they are off
// until the engine's SDXL path is optimized. KILN_SDXL=1 turns them on for engine work.
const SDXL_ENABLED = process.env.KILN_SDXL === '1';
// can the running engine build run this model family? (an engine that doesn't report "sdxl" is assumed able)
const canRunFamily = (fam) => fam === 'anima' || (fam === 'sdxl' && SDXL_ENABLED && !(engineInfo && engineInfo.features && engineInfo.features.sdxl === false));

const VERSION = '1.3.0';
const ROOT = path.resolve(__dirname, '..');
const MODELS = path.resolve(process.env.KILN_MODELS || path.join(ROOT, 'models'));
const OUTPUTS = path.resolve(process.env.KILN_OUTPUTS || path.join(ROOT, 'outputs'));
const TMP = path.join(OUTPUTS, 'tmp');
const WEB = path.resolve(process.env.KILN_WEB || path.join(ROOT, 'web'));
const PORT = Number(process.env.KILN_PORT || 8090);
const HOST = process.env.KILN_HOST || '0.0.0.0';
const INPUTS = path.resolve(process.env.KILN_INPUTS || path.join(ROOT, 'inputs'));
const INPUTS_RAW = path.join(INPUTS, '.raw');
const WORKFLOWS = path.resolve(process.env.KILN_WORKFLOWS || path.join(ROOT, 'workflows'));
const PREVIEWS = path.join(TMP, 'previews');
const CATALOG_FILE = path.join(__dirname, 'nodes.json');
const ENGINE_EXE = path.resolve(process.env.KILN_ENGINE || path.join(ROOT, 'engine', 'build', 'kiln-engine.exe'));
const EXTENSIONS = path.resolve(process.env.KILN_EXTENSIONS || path.join(ROOT, 'extensions'));
// installing / enabling packs runs their code in this process: only from this PC unless KILN_EXT_LAN=1
const EXT_LAN = process.env.KILN_EXT_LAN === '1';
const CONFIG = path.resolve(process.env.KILN_CONFIG || path.join(ROOT, 'config'));

// ---------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------
const logRing = [];
function log(msg) {
  const line = `${new Date().toISOString().slice(11, 23)} ${msg}`;
  logRing.push(line);
  if (logRing.length > 400) logRing.shift();
  console.log(line);
  broadcast({ type: 'log', msg: line });
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------
const clients = new Set();
function broadcast(obj) {
  if (!clients.size) return;
  const data = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of clients) {
    try { res.write(data); } catch (_) { clients.delete(res); }
  }
}
// a named event (comment lines never reach the page): the page reconnects when these stop arriving
setInterval(() => { for (const res of clients) { try { res.write('event: ping\ndata: {}\n\n'); } catch (_) { } } }, 15000).unref();

// ---------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------
function modelKind(name) {
  if (/(^|[^a-z])\d+x[-_ ]|esrgan|upscal|animesharp/i.test(name)) return 'upscaler';
  if (/yolo|detect/i.test(name)) return 'detector';
  if (/lora/i.test(name)) return 'lora';
  if (/vae/i.test(name)) return 'vae';
  if (/qwen|t5|text.?enc|clip|umt5/i.test(name)) return 'text_encoder';
  return 'diffusion';
}

const MODEL_EXT = /\.(safetensors|sft)$/i;
// files under dir (recursive), as forward-slash paths relative to dir
function scanTree(dir, re, rel = '', depth = 0, out = []) {
  let ents = [];
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return out; }
  for (const e of ents) {
    if (e.name.startsWith('.')) continue;
    if (e.isDirectory() && depth < 4) scanTree(path.join(dir, e.name), re, rel + e.name + '/', depth + 1, out);
    else if (e.isFile() && re.test(e.name)) out.push(rel + e.name);
  }
  return out.sort((a, b) => a.localeCompare(b));
}

function listModels() {
  const checkpoints = [];
  const loras = [];
  const scan = (dir, rel, forceLora) => {
    let ents = [];
    try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return; }
    for (const e of ents) {
      if (!e.isFile() || !/\.safetensors$/i.test(e.name)) continue;
      let size = 0;
      try { size = fs.statSync(path.join(dir, e.name)).size; } catch (_) { }
      const kind = forceLora ? 'lora' : modelKind(e.name);
      if (kind === 'upscaler' || kind === 'detector') continue; // those live in upscale/ and detect/
      const item = { file: rel ? rel + '/' + e.name : e.name, name: e.name.replace(/\.safetensors$/i, ''), size, kind, path: path.join(dir, e.name) };
      if (kind === 'lora') loras.push(item); else checkpoints.push(item);
    }
  };
  // Only the top level and loras/ are scanned here; detect/ and upscale/ are never listed as
  // checkpoints or LoRAs. ComfyUI's standard subfolders are added by scanTree below.
  scan(MODELS, '', false);
  for (const f of scanTree(path.join(MODELS, 'loras'), MODEL_EXT)) {
    const full = path.join(MODELS, 'loras', ...f.split('/'));
    let size = 0;
    try { size = fs.statSync(full).size; } catch (_) { }
    loras.push({ file: 'loras/' + f, name: path.basename(f).replace(MODEL_EXT, ''), size, kind: 'lora', path: full });
  }
  for (const [sub, kind] of [['diffusion_models', 'diffusion'], ['unet', 'diffusion'], ['text_encoders', 'text_encoder'], ['clip', 'text_encoder'], ['vae', 'vae'], ['checkpoints', 'checkpoint']]) {
    for (const f of scanTree(path.join(MODELS, sub), MODEL_EXT)) {
      const full = path.join(MODELS, sub, ...f.split('/'));
      let size = 0;
      try { size = fs.statSync(full).size; } catch (_) { }
      checkpoints.push({ file: sub + '/' + f, name: path.basename(f).replace(MODEL_EXT, ''), size, kind, path: full, rel: f });
    }
  }
  // download folders the user moved outside models/ (Settings): listed by absolute path
  const extra = civitai.extraDirs();
  const abs = (d, f) => path.join(d, ...f.split('/'));
  if (extra.loras) for (const f of scanTree(extra.loras, MODEL_EXT)) loras.push({ file: abs(extra.loras, f).split(path.sep).join('/'), name: path.basename(f).replace(MODEL_EXT, ''), size: 0, kind: 'lora', path: abs(extra.loras, f) });
  if (extra.checkpoints) for (const f of scanTree(extra.checkpoints, MODEL_EXT)) checkpoints.push({ file: abs(extra.checkpoints, f).split(path.sep).join('/'), name: path.basename(f).replace(MODEL_EXT, ''), size: 0, kind: 'checkpoint', path: abs(extra.checkpoints, f), rel: f });
  loras.sort((a, b) => a.file.localeCompare(b.file));
  checkpoints.sort((a, b) => a.file.localeCompare(b.file));
  return { checkpoints, loras };
}
const publicModel = ({ file, name, size, kind }) => ({ file, name, size, kind });

// Post-processing availability: model files on disk + (if the engine reports it) engine support.
const FEATURE_FILES = {
  face: ['detect/face_yolov8m.safetensors', 'detect/face_yolov8m.json'],
  upscale_model: ['upscale/4x-AnimeSharp.safetensors'],
};
function features() {
  const missing = (list) => list.filter(f => !fs.existsSync(path.join(MODELS, ...f.split('/'))));
  const ef = engineInfo && engineInfo.features && typeof engineInfo.features === 'object' ? engineInfo.features : null;
  const engineLacks = (k) => !!ef && ef[k] === false;
  const faceMissing = missing(FEATURE_FILES.face);
  const upMissing = missing(FEATURE_FILES.upscale_model);
  const why = (miss, k) => miss.length ? 'missing models/' + miss.join(' + models/') : engineLacks(k) ? 'engine build lacks it' : '';
  return {
    hires: !engineLacks('hires'),
    face: !faceMissing.length && !engineLacks('face'),
    upscale_model: !upMissing.length && !engineLacks('upscale'),
    reasons: {
      hires: engineLacks('hires') ? 'engine build lacks it' : '',
      face: why(faceMissing, 'face'),
      upscale_model: why(upMissing, 'upscale'),
    },
  };
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------
const engine = new Engine({
  exe: ENGINE_EXE,
  args: ['--models', MODELS],
  mockScript: path.join(__dirname, 'mock-engine.js'),
});
let engineInfo = null;
const infoWaiters = new Map();

// ---- GPU sharing: another app on this PC (for example a local LLM server) can park the engine to use the
// VRAM. A parked engine is stopped; the next render unparks it, first asking the holder to let go of the
// GPU through its release URL so the engine never loads next to the other app's model.
let parked = null;  // { holder, release_url, since }
const engineStatus = () => Object.assign(engine.status(), parked ? { parked: parked.holder } : {},
  running && running.gpuWaiting ? { gpu_wait: running.gpuWaiting } : {}, gpuPeer ? { gpu_peer: gpuPeer.name } : {});

// ---- GPU peer: another app on this PC with its own model loaded on the same card (for example a local
// LLM). Both stay loaded; they take turns: before each render Kiln asks the peer for the GPU (the peer
// answers once its own work is done) and gives it back when the render ends. Rendering while the other
// model computes is what locks the card up.
let gpuPeer = null;  // { name, acquire_url, release_url }
const localUrl = (u) => typeof u === 'string' && /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?\//.test(u) ? u : null;
async function acquireGpu(j) {
  const p = gpuPeer;
  if (!p) return false;
  j.gpuWaiting = p.name;
  broadcast({ type: 'engine', engine: engineStatus() });
  try {
    const r = await fetch(p.acquire_url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ holder: 'Kiln', job: j.id }), signal: AbortSignal.timeout(20 * 60000) });
    if (r.status === 404 && gpuPeer === p) { gpuPeer = null; log(`${p.name} is gone; rendering without asking it`); }
    return r.ok;
  } catch (e) {
    log(`${p.name} did not grant the GPU (${e.message}); rendering anyway`);
    return false;
  } finally {
    j.gpuWaiting = null;
    broadcast({ type: 'engine', engine: engineStatus() });
  }
}
function releaseGpu(p) {
  if (p && p.release_url) fetch(p.release_url, { method: 'POST', signal: AbortSignal.timeout(10000) }).catch(() => { });
}
async function releaseHolder(p) {
  if (!p.release_url) return;
  try {
    const r = await fetch(p.release_url, { method: 'POST', signal: AbortSignal.timeout(90000) });
    log(`${p.holder} released the GPU (HTTP ${r.status})`);
  } catch (e) {
    log(`${p.holder} did not answer its release URL (${e.message}); starting the engine anyway`);
  }
}
function unpark(reason) {
  if (!parked) return;
  const p = parked;
  parked = null;
  log(`engine unparked (${reason}); it was parked for ${p.holder}`);
  broadcast({ type: 'engine', engine: engineStatus() });
  releaseHolder(p).then(() => { if (!engine.proc && !parked) { engine.args = engineArgs(); engine.start(); } });
}

engine.on('log', (m) => log(m));
engine.on('state', () => {
  broadcast({ type: 'engine', engine: engineStatus() });
  if (engine.state === 'ready') {
    // learn engine capabilities (e.g. features.face === false) before the first job
    if (!running) engineInfoRequest(3000).then((ev) => { if (ev) broadcast({ type: 'features', features: features() }); });
    pump();
  }
});
engine.on('exit', () => {
  engineFamily = 'anima';
  engineDit = lastDit;  // what the next start loads (engineArgs)
  if (running) {
    const j = running;
    running = null;
    finishJob(j, 'error', { error: 'engine exited while generating' });
  }
});
engine.on('event', onEngineEvent);
// Animadex character index (tags for ~36k characters the Anima model knows)
const animadex = require('./lib/animadex').createAnimadex({ modelsDir: MODELS, log });

// hot-swap: mock -> real engine as soon as the exe appears (only while idle)
setInterval(() => {
  if (engine.mock && !running && engine.state === 'ready' && fs.existsSync(ENGINE_EXE)) {
    log('kiln-engine.exe appeared; switching from mock to real engine');
    engine.restart();
  }
}, 5000).unref();

// ---------------------------------------------------------------------------
// jobs
// ---------------------------------------------------------------------------
let jobSeq = 0, groupSeq = 0;
let engineFamily = 'anima';   // model family loaded in the engine (it starts with Anima)
// Anima DiT loaded in the engine (it starts with the stock base model; null = unknown after a graph swapped it)
const DEFAULT_DIT = path.join(MODELS, 'anima-base-v1.0.safetensors');
let engineDit = DEFAULT_DIT;
// the DiT the last Anima job used: a restarted engine (crash, park) starts on it, saving a model swap
let lastDit = DEFAULT_DIT;
function vramBudget() {
  try { const c = JSON.parse(fs.readFileSync(path.join(CONFIG, 'engine.json'), 'utf8')); return Math.floor(Number(c.vram_budget_mb)) || 0; } catch (_) { return 0; }
}
const engineArgs = () => ['--models', MODELS, ...(sameFile(lastDit, DEFAULT_DIT) ? [] : ['--dit', lastDit]),
  ...(vramBudget() ? ['--vram-budget', String(vramBudget())] : [])];
const sameFile = (a, b) => !!a && !!b && path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase();
const jobs = new Map();
const queue = [];
let running = null;

function publicJob(j) {
  if (!j) return null;
  return {
    id: j.id, group: j.group, index: j.index, count: j.count, status: j.status,
    params: j.params, created: j.created, started: j.started, finished: j.finished,
    step: j.step, of: j.of, stage: j.stage, face: j.face, timeline: j.timeline,
    step_ms: j.stepMs, encode_ms: j.encodeMs, decode_ms: j.decodeMs, timings: j.timings,
    total_ms: j.totalMs, wall_ms: j.wallMs, image: j.image, error: j.error, loading: j.loading,
    position: j.status === 'queued' ? queue.indexOf(j) + 1 : 0,
    kind: j.kind || 'generate', swap: j.swap || undefined, train: j.kind === 'train' ? trainView(j) : undefined,
    tag: j.kind === 'tag' ? { dir: j.tag.opt.dir, failed: j.tag.failed, changed: j.tag.changed } : undefined, error_node: j.errorNode, node: j.node, node_status: j.nodeStatus, outputs: j.outputs, node_ms: j.nodeMs,
  };
}
function queueSnapshot() {
  return { running: publicJob(running), queued: queue.map(publicJob) };
}
function emitJob(j) { broadcast({ type: 'job', job: publicJob(j) }); }
function emitQueue() { broadcast({ type: 'queue', queue: queueSnapshot() }); }

function randomSeed() { return Math.floor(Math.random() * 9007199254740992); } // 0..2^53-1

function num(v, def, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n)) return def;
  return Math.min(max, Math.max(min, n));
}
function snap16(v, def) {
  const n = num(v, def, 64, 4096);
  return Math.min(2048, Math.max(256, Math.round(n / 16) * 16));
}

// "name" from <lora:name:w> -> a LoRA file (name, path without extension, or file)
function resolveLora(models, name) {
  const n = String(name).trim().toLowerCase().replace(/\\/g, '/').replace(MODEL_EXT, '');
  const strip = (f) => f.toLowerCase().replace(MODEL_EXT, '');
  return models.loras.find(l => l.name.toLowerCase() === n) || models.loras.find(l => strip(l.file) === n || strip(l.file).replace(/^loras\//, '') === n) || null;
}
function validateRequest(b) {
  const models = listModels();
  const warnings = [], tagDropped = [];
  // "Stable Diffusion checkpoint": only Anima runs today (the engine picks its default Anima model)
  const sdList = models.checkpoints.filter(c => c.kind === 'diffusion' || c.kind === 'checkpoint');
  let model = null;
  if (typeof b.model === 'string' && b.model.trim()) {
    const c = sdList.find(x => x.file === b.model || x.name === b.model);
    if (!c) throw new Error('unknown checkpoint: ' + b.model);
    const fam = library.describe(c.path).family;
    if (!canRunFamily(fam)) throw new Error(fam === 'sdxl' ? (SDXL_ENABLED ? `this kiln-engine build can't run SDXL checkpoints (${c.name})` : `SDXL support is coming soon (${c.name})`) : `${c.name} is not a model Kiln can run yet (${SF.FAMILY_LABEL[fam] || fam}).`);
    model = { file: c.file, family: fam, path: c.path };
  } else {
    const c = sdList.find(x => SF.inspect(x.path).family === 'anima');
    if (c) model = { file: c.file, family: 'anima', path: c.path };
  }
  const loraMap = new Map(models.loras.map(l => [l.file, l]));
  const loras = [];
  for (const l of Array.isArray(b.loras) ? b.loras : []) {
    if (!l || typeof l.file !== 'string') continue;
    const m = loraMap.get(l.file) || models.loras.find(x => x.name === l.file || path.basename(x.file) === l.file);
    if (!m) throw new Error('unknown LoRA: ' + l.file);
    const strength = num(l.strength, 1, -4, 4);
    if (strength === 0) continue;
    const i = loras.findIndex(x => x.file === m.file);
    if (i >= 0) loras[i].strength = strength; else loras.push({ file: m.file, strength, path: m.path });
  }
  // A1111 <lora:name:weight> in the prompt: added to the slots (a tag overrides a slot of the same LoRA)
  const slotLoras = loras.map(l => ({ ...l }));
  const withTags = (list, conv) => {
    for (const t of conv.tags) {
      const m = resolveLora(models, t.name);
      if (!m) { if (!tagDropped.some(d => d.startsWith(`<lora:${t.name}>`))) tagDropped.push(`<lora:${t.name}> (no LoRA with that name)`); continue; }
      const strength = Math.max(-4, Math.min(4, t.weight));
      const i = list.findIndex(x => x.file === m.file);
      if (i >= 0) { if (strength === 0) list.splice(i, 1); else list[i].strength = strength; }
      else if (strength !== 0) list.push({ file: m.file, strength, path: m.path });
    }
    return list;
  };
  const conv = A.toKilnPrompt(typeof b.prompt === 'string' ? b.prompt : '');
  withTags(loras, conv);
  // prompt variants (packs such as outfit lists): one set of images per prompt, in one group; each variant's
  // own <lora:> tags apply to it alone
  let variants = null;
  if (Array.isArray(b.variants) && b.variants.length) {
    if (b.variants.length > 32) throw new Error('at most 32 prompt variants');
    variants = b.variants.map((v) => {
      const text = typeof v === 'string' ? v : v && typeof v.prompt === 'string' ? v.prompt : null;
      if (text == null) throw new Error('prompt variants must be strings');
      const c = A.toKilnPrompt(text);
      return { prompt: text, promptText: c.text, loras: withTags(slotLoras.map(l => ({ ...l })), c) };
    });
  }
  const sdxl = !!model && model.family === 'sdxl';
  if (model && !sdxl) {
    for (const l of loras) {
      const d = library.describe(l.path);
      if (d.family !== 'unknown' && d.family !== model.family) warnings.push(`${path.basename(l.file).replace(MODEL_EXT, '')} is made for ${d.base || SF.FAMILY_LABEL[d.family]}, but the checkpoint is ${SF.FAMILY_LABEL[model.family]}`);
    }
  }
  const sampler = typeof b.sampler === 'string' && /^[a-z0-9_+-]{1,32}$/i.test(b.sampler) ? b.sampler : 'euler';
  let seed = Number(b.seed);
  if (!Number.isFinite(seed) || seed < 0) seed = -1;
  else seed = Math.min(9007199254740991, Math.floor(seed));
  // SDXL: the engine rounds sizes down to multiples of 32; do it here so the job reports the real size
  const snap32 = (v, def) => { const n = Number(v); return Math.min(2048, Math.max(256, Math.floor((Number.isFinite(n) ? n : def) / 32) * 32)); };
  const width = sdxl ? snap32(b.width, 832) : snap16(b.width, 512), height = sdxl ? snap32(b.height, 1216) : snap16(b.height, 768);
  const enh = validateEnhance(b, width, height);
  const ret = {
    prompt: typeof b.prompt === 'string' ? b.prompt : '',
    negative: typeof b.negative === 'string' ? b.negative : '',
    width,
    height,
    steps: Math.round(num(b.steps, 8, 1, 200)),
    cfg: num(b.cfg, 1, 0, 30),
    // CFG only for the first cfg_cutoff fraction of steps (1 = always = old behaviour)
    cfg_cutoff: Math.round(num(b.cfg_cutoff, 1, 0.1, 1) * 1000) / 1000,
    // first-block step cache threshold (0 = off): reuse blocks 1..27 when block 0 barely changed
    step_cache: Math.round(num(b.step_cache, 0, 0, 0.5) * 1000) / 1000,
    // NAG: negative prompt at CFG 1 (the engine uses it only when cfg <= 1)
    nag: b.nag && b.nag.enabled !== false ? {
      scale: Math.round(num(b.nag.scale, 5, 1, 20) * 100) / 100,
      tau: Math.round(num(b.nag.tau, 2.5, 1, 10) * 100) / 100,
      alpha: Math.round(num(b.nag.alpha, 0.25, 0, 1) * 100) / 100,
    } : null,
    seed,
    sampler,
    shift: num(b.shift, 3, 0, 100),
    loras,
    batch: Math.round(num(b.batch ?? b.count ?? b.batch_count, 1, 1, 64)),
    model: model ? model.file : undefined,
    // Anima: the DiT file the engine renders with (it swaps models when this changes)
    ditPath: model && !sdxl ? model.path : undefined,
    // what gets tokenized: <lora:> tags removed, A1111 [de-emphasis] -> (text:0.9091)
    promptText: conv.text,
    negativeText: A.deemphasize(A.extractLoraTags(typeof b.negative === 'string' ? b.negative : '').text),
    variants,
    // variants share one seed (only the prompt changes between them); 'different' gives each its own
    variant_seeds: b.variant_seeds === 'different' ? 'different' : 'same',
    warnings,
    ...enh,
    dropped: [...tagDropped, ...enh.dropped],
  };
  if (variants && variants.length * ret.batch > 256) throw new Error(`${variants.length * ret.batch} images in one go is too many (at most 256)`);
  if (sdxl) {
    const p = ret;
    p.family = 'sdxl';
    p.checkpointPath = model.path;
    p.scheduler = SDXL_SCHEDULERS.includes(b.scheduler) ? b.scheduler : 'normal';
    if (!SDXL_SAMPLERS.includes(p.sampler)) throw new Error(`sampler "${p.sampler}" isn't available for SDXL (${SDXL_SAMPLERS.join(', ')})`);
    // not on SDXL yet (the engine would log and ignore them): say so instead of dropping them silently
    const notes = [];
    if (p.loras.length) notes.push(`LoRAs are not supported on SDXL yet, rendered without: ${p.loras.map(l => path.basename(l.file).replace(MODEL_EXT, '')).join(', ')}`);
    if (p.face) notes.push('the face detailer is not supported on SDXL yet');
    if (p.nag && p.cfg <= 1 && p.negative.trim()) notes.push('NAG is Anima-only');
    if (p.step_cache > 0) notes.push('step cache is Anima-only');
    p.loras = []; p.face = null; p.nag = null; p.step_cache = 0; p.shift = undefined;   // shift is an Anima (flow) setting
    p.dropped = [...notes, ...p.dropped];
  }
  return ret;
}

const round3 = (v) => Math.round(v * 1000) / 1000;
const MAX_OUTPUT_PIXELS = 100e6;

// Post-processing options (hires fix -> face detail -> final upscale). Options whose
// models/engine support are missing are dropped (reported back in `dropped`).
function validateEnhance(b, width, height) {
  const feat = features();
  const dropped = [];
  let hires = null, face = null, upscale = null;

  const h = b.hires;
  if (h && typeof h === 'object' && h.enabled !== false) {
    const scale = num(h.scale, 1.5, 1, 4);
    if (scale > 1.001) {
      if (!feat.hires) dropped.push('hires fix (' + feat.reasons.hires + ')');
      else {
        let upscaler = h.upscaler === 'lanczos' ? 'lanczos' : 'model';
        if (upscaler === 'model' && !feat.upscale_model) { upscaler = 'lanczos'; dropped.push('hires model upscaler (using lanczos; ' + feat.reasons.upscale_model + ')'); }
        hires = { scale: round3(scale), denoise: round3(num(h.denoise, 0.4, 0.01, 1)), steps: Math.round(num(h.steps, 4, 1, 100)), upscaler };
      }
    }
  }

  const f = b.face;
  if (f && typeof f === 'object' && f.enabled) {
    if (!feat.face) dropped.push('face detail (' + feat.reasons.face + ')');
    else {
      face = {
        enabled: true,
        denoise: round3(num(f.denoise, 0.4, 0.01, 1)),
        steps: Math.round(num(f.steps, 4, 1, 100)),
        guide: Math.round(num(f.guide, 384, 64, 2048)),
        max_size: Math.round(num(f.max_size, 576, 64, 4096)),
        crop: round3(num(f.crop, 2.0, 1, 8)),
        conf: round3(num(f.conf, 0.35, 0.01, 0.99)),
        max_faces: Math.round(num(f.max_faces, 4, 1, 32)),
      };
    }
  }

  const factor = Number(b.upscale && typeof b.upscale === 'object' ? b.upscale.factor : b.upscale);
  if (factor === 2 || factor === 4) {
    if (!feat.upscale_model) dropped.push(`${factor}x upscale (${feat.reasons.upscale_model})`);
    else upscale = { factor };
  }

  // rough final size (engine does the exact rounding) for a sanity cap
  const hw = hires ? Math.round(width * hires.scale / 16) * 16 : width;
  const hh = hires ? Math.round(height * hires.scale / 16) * 16 : height;
  const k = upscale ? upscale.factor : 1;
  if (hw * k * hh * k > MAX_OUTPUT_PIXELS) {
    throw new Error(`output would be ~${hw * k}x${hh * k} (${Math.round(hw * k * hh * k / 1e6)} MP); limit is ${MAX_OUTPUT_PIXELS / 1e6} MP. Lower the hires scale or upscale factor.`);
  }
  return { hires, face, upscale, dropped, expected: { w: hw * k, h: hh * k } };
}

function createJobs(p) {
  const group = 'g' + (++groupSeq);
  const sdxl = p.family === 'sdxl';
  const negText = p.negativeText != null ? p.negativeText : p.negative;
  const neg = !sdxl && (p.cfg > 1 || (p.nag && negText.trim())) ? encodePrompt(negText) : null;
  const variants = p.variants || [{ prompt: p.prompt, promptText: p.promptText, loras: p.loras }];
  // with several variants and a random seed, one random base seed is shared so they stay comparable
  const base = variants.length > 1 && p.seed < 0 && p.variant_seeds === 'same' ? randomSeed() : p.seed;
  const created = Date.now();
  const out = [];
  const count = variants.length * p.batch;
  if (count > 256) throw new Error(`${count} images in one go is too many (at most 256)`);
  for (let v = 0; v < variants.length; v++) {
    const V = variants[v];
    const posText = V.promptText != null ? V.promptText : V.prompt;
    const qwen = sdxl ? null : encodePrompt(posText);
    const sd = sdxl ? { pos: encodeSDXL(posText), neg: encodeSDXL(negText) } : null;
    const loras = sdxl ? [] : V.loras;
    for (let i = 0; i < p.batch; i++) {
      const seed = base < 0 ? randomSeed() : Math.min(9007199254740991, base + i + (p.variant_seeds === 'different' ? v * p.batch : 0));
      const j = {
        id: 'j' + (++jobSeq), group, index: v * p.batch + i, count, status: 'queued', created,
        params: {
          prompt: V.prompt, negative: p.negative, width: p.width, height: p.height, steps: p.steps,
          cfg: p.cfg, cfg_cutoff: p.cfg_cutoff, step_cache: p.step_cache, nag: p.nag, seed, sampler: p.sampler, shift: p.shift,
          loras: loras.map(l => ({ file: l.file, strength: l.strength })),
          model: p.model,
          family: sdxl ? 'sdxl' : undefined,
          scheduler: sdxl ? p.scheduler : undefined,
        },
        loraPaths: loras.map(l => ({ file: l.path, strength: l.strength })),
        enc: qwen, neg, sd, family: sdxl ? 'sdxl' : 'anima', checkpoint: sdxl ? p.checkpointPath : null, dit: sdxl ? null : p.ditPath || null,
        step: 0, of: p.steps, stepMs: [], stage: 'base', timeline: [],
      };
      // post-processing options live in params (so they land in the PNG's kiln tEXt)
      if (p.hires) j.params.hires = { ...p.hires };
      if (p.face) j.params.face = { ...p.face };
      if (p.upscale) j.params.upscale = { ...p.upscale };
      jobs.set(j.id, j);
      queue.push(j);
      out.push(j);
    }
  }
  // keep memory bounded
  if (jobs.size > 500) {
    for (const [id, j] of jobs) {
      if (jobs.size <= 400) break;
      if (j.status !== 'queued' && j.status !== 'running') jobs.delete(id);
    }
  }
  return { group, jobs: out };
}

function pump() {
  if (parked) { if (queue.length) unpark('a render was queued'); return; }
  if (running || engine.state !== 'ready' || !queue.length) return;
  const j = queue.shift();
  running = j;
  if (gpuPeer && !j.gpuLease) {
    const p = gpuPeer;
    acquireGpu(j).then((granted) => {
      if (granted) j.gpuLease = p;
      if (running !== j) { if (j.gpuLease) { releaseGpu(j.gpuLease); j.gpuLease = null; } setImmediate(pump); return; }  // cancelled meanwhile
      j.gpuLease = j.gpuLease || true;
      runJob(j);
    });
    emitQueue();
    return;
  }
  runJob(j);
}

// ---------------------------------------------------------------------------
// LoRA training (Anima): a folder of images with .txt captions -> models/loras/<name>.safetensors, with
// checkpoints and preview PNGs beside it. The engine runs it as one long job; renders queue behind it.
// ---------------------------------------------------------------------------
const TRAIN_IMG_RE = /\.(png|jpe?g|webp|bmp|gif)$/i;
const LORA_NAME_RE = /^[\w\- .]{1,80}$/;
function scanDataset(dir) {
  const abs = path.resolve(String(dir || '').trim() || '.');
  let names;
  try { names = fs.readdirSync(abs); } catch (e) { throw new Error('cannot read the folder ' + abs); }
  const items = [];
  for (const f of names.sort()) {
    if (!TRAIN_IMG_RE.test(f)) continue;
    let caption = '';
    try { caption = fs.readFileSync(path.join(abs, f.replace(/\.[^.]+$/, '.txt')), 'utf8').trim(); } catch (_) { }
    items.push({ file: f, image: path.join(abs, f), caption });
  }
  return { dir: abs, items };
}
// captions are literal text: Kiln's prompt weighting would read "star (sky)" as emphasis
const literalText = (t) => String(t).replace(/([()[\]])/g, '\\$1');

function validateTrain(b) {
  const ds = scanDataset(b.dataset);
  if (!ds.items.length) throw new Error('no images (png, jpg, webp) in ' + ds.dir);
  const name = String(b.name || '').trim();
  if (!LORA_NAME_RE.test(name)) throw new Error('give the LoRA a name: letters, digits, spaces, - and _');
  const trigger = String(b.trigger || '').trim();
  const optimizer = b.optimizer === 'adamw' ? 'adamw' : 'prodigy';
  const epochs = Math.round(num(b.epochs, 10, 1, 1000));
  const repeats = Math.round(num(b.repeats, 1, 1, 1000));  // an epoch shows every image this many times
  const batch = Math.round(num(b.batch_size, 1, 1, 16));    // images per optimizer step (gradient accumulation)
  const prompts = String(b.preview_prompt || '').split(/\n+/).map(t => t.trim()).filter(Boolean).slice(0, 4);
  const on = (v) => v === true || v === 1 || v === 'true' || v === 'on';
  // previews render like txt2img's quick presets: Turbo (the turbo LoRA, 8 steps, CFG 1, Euler, NAG for the
  // negative) or Base (20 steps, CFG 4.5, DPM++ 2M, CFG for the first 70% of the steps)
  const turbo = on(b.preview_turbo);
  let turboLora = null;
  if (turbo) {
    const t = listModels().loras.find(l => /turbo/i.test(l.file));
    if (!t) throw new Error('turbo previews need the Anima turbo LoRA in models (none found)');
    turboLora = t.path;
  }
  // the Anima checkpoint to train on (empty: the engine's default, anima-base)
  let model = null;
  if (typeof b.model === 'string' && b.model.trim()) {
    const c = listModels().checkpoints.find(x => ['diffusion', 'checkpoint'].includes(x.kind) && (x.file === b.model || x.name === b.model));
    if (!c) throw new Error('unknown checkpoint: ' + b.model);
    if (library.describe(c.path).family !== 'anima') throw new Error(c.name + ' is not an Anima model; only Anima LoRAs can be trained');
    model = { file: c.file, path: c.path };
  }
  const pv = turbo ? { steps: 8, cfg: 1, sampler: 'euler', cfg_until: 1, nag: true } : { steps: 20, cfg: 4.5, sampler: 'dpmpp_2m', cfg_until: 0.7, nag: false };
  return {
    model, preview_turbo: turbo, preview_turbo_lora: turboLora, preview_sampler: pv.sampler, preview_cfg_until: pv.cfg_until, preview_nag: pv.nag,
    ds, name, trigger, optimizer, epochs, repeats, batch, steps: epochs * Math.ceil(repeats * ds.items.length / batch),
    preset: String(b.preset || '').slice(0, 40),
    shuffle: on(b.shuffle_caption), keep_tokens: Math.round(num(b.keep_tokens, 1, 0, 32)), caption_variants: Math.round(num(b.caption_variants, 4, 1, 32)),
    caption_dropout: num(b.caption_dropout, 0, 0, 0.9), flip: on(b.flip), noise_offset: num(b.noise_offset, 0, 0, 0.5),
    timestep_sampling: ['sigmoid', 'shift', 'uniform'].includes(b.timestep_sampling) ? b.timestep_sampling : 'sigmoid',
    sigmoid_scale: num(b.sigmoid_scale, 1.3, 0.1, 5), flow_shift: num(b.discrete_flow_shift, 3, 0.1, 10),
    warmup: Math.round(num(b.warmup, 0, 0, 100000)),
    rank: Math.round(num(b.rank, 32, 1, 256)), alpha: num(b.alpha, num(b.rank, 32, 1, 256), 0.01, 256),
    lr: num(b.lr, optimizer === 'prodigy' ? 1 : 1e-4, 1e-7, 10),
    resolution: Math.round(num(b.resolution, 512, 256, 1024) / 64) * 64,
    save_every_epochs: Math.round(num(b.save_every_epochs, 0, 0, 1000)),
    preview_every: Math.round(num(b.preview_every, 0, 0, 100000)),
    prompts, negative: typeof b.preview_negative === 'string' ? b.preview_negative : 'worst quality, low quality, blurry, jpeg artifacts',
    preview_w: snap16(b.preview_w, 512), preview_h: snap16(b.preview_h, 768),
    preview_steps: pv.steps, preview_cfg: pv.cfg,
    seed: Math.round(num(b.seed, 1, 0, 2 ** 31)),
    out: path.join(MODELS, 'loras', name + '.safetensors'),
  };
}

// Tag shuffling (sd-scripts' shuffle_caption + keep_tokens): the first keep_tokens tags (the trigger counts as one) stay
// put and the rest are shuffled. Captions go through the text encoder once, before training, so each image gets a few
// shuffled orders up front and the trainer picks one of them every time it shows the image.
function captionVariants(caption, p, seed) {
  // the trigger (one or more comma-separated tags) goes in front once: a caption that already has it loses its copy,
  // and every trigger tag stays put while the rest shuffle
  const split = (t) => String(t || '').split(',').map(x => x.trim()).filter(Boolean);
  const norm = (t) => t.toLowerCase().replace(/_/g, ' ').replace(/\\([()])/g, '$1');
  const trig = split(p.trigger), seen = new Set(trig.map(norm));
  const all = [...trig, ...split(caption).filter(t => !seen.has(norm(t)))];
  if (!p.shuffle) return [all.join(', ')];
  const nkeep = Math.max(p.keep_tokens, trig.length);
  const keep = all.slice(0, nkeep), rest = all.slice(nkeep);
  let st = seed >>> 0;
  const rand = () => {  // mulberry32: the same variants for the same seed
    st = (st + 0x6D2B79F5) >>> 0;
    let t = st;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Set();
  for (let k = 0; k < p.caption_variants * 4 && out.size < p.caption_variants; k++) {
    const r = rest.slice();
    for (let i = r.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [r[i], r[j]] = [r[j], r[i]]; }
    out.add([...keep, ...r].join(', '));
  }
  return [...out];
}

function createTrainJob(p) {
  const tok = (t) => { const e = encodePrompt(t); return { qwen_ids: e.qwen_ids, t5_ids: e.t5_ids, t5_weights: e.t5_weights }; };
  const items = p.ds.items.map((it, i) => {
    const v = captionVariants(it.caption, p, p.seed * 7919 + i).map(t => tok(literalText(t)));
    return { image: it.image, ...v[0], variants: v.slice(1) };
  });
  const created = Date.now();
  const j = {
    id: 'j' + (++jobSeq), group: 'g' + (++groupSeq), index: 0, count: 1, status: 'queued', created, kind: 'train', family: 'anima',
    params: {
      kind: 'train', name: p.name, dataset: p.ds.dir, images: p.ds.items.length, trigger: p.trigger, epochs: p.epochs, repeats: p.repeats, batch_size: p.batch, steps: p.steps, preset: p.preset, model: p.model ? p.model.file : '',
      shuffle_caption: p.shuffle, keep_tokens: p.keep_tokens, caption_dropout: p.caption_dropout, flip: p.flip, noise_offset: p.noise_offset,
      timestep_sampling: p.timestep_sampling, sigmoid_scale: p.sigmoid_scale, discrete_flow_shift: p.flow_shift, warmup: p.warmup,
      rank: p.rank, alpha: p.alpha, optimizer: p.optimizer, lr: p.lr, resolution: p.resolution, save_every_epochs: p.save_every_epochs,
      preview_every: p.preview_every, preview_prompts: p.prompts, preview_turbo: p.preview_turbo, out: path.relative(MODELS, p.out).split(path.sep).join('/'),
    },
    trainReq: {
      items, out: p.out, dit: p.model ? p.model.path : undefined, rank: p.rank, alpha: p.alpha, epochs: p.epochs, repeats: p.repeats, optimizer: p.optimizer, lr: p.lr, resolution: p.resolution,
      save_every_epochs: p.save_every_epochs, seed: p.seed, grad_clip: 1.0, weight_decay: 0.01, warmup: p.warmup, batch_size: p.batch,
      timestep_sampling: p.timestep_sampling, sigmoid_scale: p.sigmoid_scale, discrete_flow_shift: p.flow_shift, noise_offset: p.noise_offset,
      flip: p.flip, caption_dropout: p.caption_dropout, empty_caption: p.caption_dropout > 0 ? tok('') : undefined,
      shuffle_caption: p.shuffle, keep_tokens: p.keep_tokens,
      previews: p.preview_every > 0 ? p.prompts.map(tok) : [], preview_negative: tok(p.negative), preview_every: p.preview_every,
      preview_w: p.preview_w, preview_h: p.preview_h, preview_steps: p.preview_steps, preview_cfg: p.preview_cfg, preview_seed: 42,
      preview_sampler: p.preview_sampler, preview_cfg_until: p.preview_cfg_until, preview_nag: p.preview_nag,
      preview_turbo_lora: p.preview_turbo_lora || undefined,
    },
    train: { stage: 'queued', cache: null, losses: [], previews: [], saved: [], epoch: 0, epochs: p.epochs, msAvg: 0 },
    step: 0, of: p.steps, stepMs: [], stage: 'train', timeline: [],
  };
  jobs.set(j.id, j);
  queue.push(j);
  return j;
}

// files the trainer writes live under models/loras: served by path relative to it
function trainFileUrl(file) {
  const rel = path.relative(path.join(MODELS, 'loras'), file);
  return rel.startsWith('..') || path.isAbsolute(rel) ? null : '/api/train/file?f=' + encodeURIComponent(rel.split(path.sep).join('/'));
}
function trainView(j) {
  const t = j.train;
  // the loss curve, thinned to at most ~600 points
  const L = t.losses, k = Math.max(1, Math.ceil(L.length / 600));
  return {
    stage: t.stage, cache: t.cache, epoch: t.epoch, epochs: t.epochs, msAvg: Math.round(t.msAvg), lr: t.lr, d: t.d,
    params: t.params, arenaMb: t.arenaMb, last: L.length ? L[L.length - 1] : null,
    losses: k === 1 ? L : L.filter((_, i) => i % k === 0 || i === L.length - 1),
    previews: t.previews, saved: t.saved, warnings: t.warnings || [],
  };
}

function onTrainEvent(j, ev) {
  const t = j.train;
  const send = (extra) => broadcast({ type: 'train', id: j.id, step: j.step, of: j.of, train: trainView(j), ...extra });
  switch (ev.ev) {
    case 'loading':
      t.stage = 'loading';
      send();
      break;
    case 'train':
      if (ev.stage === 'warn') {  // e.g. images that could not be read and were left out
        t.warnings = [...(t.warnings || []), String(ev.msg || '')];
        log(`job ${j.id} ${ev.msg}`);
        send();
        break;
      }
      t.stage = ev.stage;
      if (ev.stage === 'cache') t.cache = { done: ev.done, of: ev.of };
      if (ev.stage === 'ready') { t.params = ev.params; t.arenaMb = ev.arena_mb; }
      send();
      break;
    case 'step':
      t.stage = 'train';
      j.step = ev.step; j.of = ev.of;
      t.epoch = ev.epoch; t.epochs = ev.epochs; t.lr = ev.lr; t.d = ev.d;
      t.msAvg = t.msAvg ? t.msAvg * 0.9 + ev.ms * 0.1 : ev.ms;
      t.losses.push([ev.step, +ev.loss.toFixed(5), +ev.avg.toFixed(5)]);
      send();
      break;
    case 'preview': {
      const url = trainFileUrl(ev.file);
      if (url) t.previews.push({ step: ev.step, index: ev.index, url, w: ev.w, h: ev.h });
      send();
      break;
    }
    case 'saved':
      t.saved.push({ file: path.basename(ev.file), step: ev.step, epoch: ev.epoch });
      send();
      break;
    case 'done':
    case 'cancelled':
    case 'error':
      if (running === j) running = null;
      engineFamily = 'none';  // the trainer unloads Anima; the next render reloads it
      t.stage = ev.ev;
      j.trainReq = null;
      if (ev.ev === 'done') finishJob(j, 'done', { totalMs: ev.total_ms });
      else if (ev.ev === 'cancelled') finishJob(j, 'cancelled');
      else finishJob(j, 'error', { error: ev.msg || 'training failed' });
      break;
  }
}

// ---------------------------------------------------------------------------
// Auto-tagging (Datasets tab): the engine runs a WD tagger over the images; each image's tags become its caption
// (appended to what is there, replacing it, or only where there is none).
// ---------------------------------------------------------------------------
function createTagJob(p) {
  const created = Date.now();
  const floor = Math.max(0.02, Math.min(p.general, p.character) - 0.001);
  const j = {
    id: 'j' + (++jobSeq), group: 'g' + (++groupSeq), index: 0, count: 1, status: 'queued', created, kind: 'tag', family: 'none',
    params: { kind: 'tag', dir: p.dir, model: p.model, images: p.files.length, mode: p.mode, general: p.general, character: p.character },
    tagReq: { model: taggers.dirOf(p.model), images: p.files.map(f => path.join(p.dir, f)), floor },
    tag: { files: p.files, opt: p, failed: [], changed: 0 },
    step: 0, of: p.files.length, stepMs: [], stage: 'tag', timeline: [],
  };
  jobs.set(j.id, j);
  queue.push(j);
  return j;
}

function onTagEvent(j, ev) {
  const t = j.tag;
  switch (ev.ev) {
    case 'loading':
      j.loading = { what: 'tagger' };
      emitJob(j);
      break;
    case 'tagged':
    case 'tag_failed': {
      const file = t.files[ev.index];
      let caption = null;
      if (ev.ev === 'tag_failed') t.failed.push(`${file}: ${ev.msg}`);
      else {
        try {
          const o = t.opt;
          const img = path.join(o.dir, file);
          const have = datasets.split(datasets.readCap(img));
          const got = taggers.caption(o.model, ev.probs, o).tags;
          let next = null;
          if (o.mode === 'missing') { if (!have.length) next = got; }
          else if (o.mode === 'replace') {
            const keep = have.slice(0, o.keep_first), seen = new Set(keep.map(datasets.key));
            next = [...keep, ...got.filter(x => !seen.has(datasets.key(x)))];
          } else {
            const seen = new Set(have.map(datasets.key));
            next = [...have, ...got.filter(x => !seen.has(datasets.key(x)))];
          }
          if (next) {
            caption = datasets.writeCaption(o.dir, file, next.join(', '));
            t.changed++;
          }
        } catch (e) { t.failed.push(`${file}: ${e.message}`); }
      }
      j.step = Math.max(j.step, ev.index + 1);
      broadcast({ type: 'autotag', id: j.id, dir: t.opt.dir, file, caption, done: j.step, of: j.of, failed: t.failed.length });
      break;
    }
    case 'done':
    case 'cancelled':
    case 'error':
      if (running === j) running = null;
      j.loading = null;
      if (ev.ev === 'done') finishJob(j, 'done', { totalMs: ev.total_ms });
      else if (ev.ev === 'cancelled') finishJob(j, 'cancelled');
      else finishJob(j, 'error', { error: ev.msg || 'tagging failed' });
      break;
  }
}

function runJob(j) {
  if (engine.state !== 'ready') { running = null; queue.unshift(j); if (j.gpuLease && j.gpuLease !== true) releaseGpu(j.gpuLease); j.gpuLease = null; return; }
  j.status = 'running';
  j.started = Date.now();
  const p = j.params;
  fs.mkdirSync(TMP, { recursive: true });
  if (j.kind === 'tag') {
    if (!engine.send({ id: j.id, cmd: 'tag', ...j.tagReq })) { running = null; j.status = 'queued'; queue.unshift(j); return; }
    log(`job ${j.id} start tagging ${p.images} images in ${p.dir} with ${p.model} (${p.mode}, general ${p.general}, character ${p.character})`);
    emitJob(j);
    emitQueue();
    return;
  }
  if (j.kind === 'train') {
    if (!engine.send({ id: j.id, cmd: 'train', ...j.trainReq })) { running = null; j.status = 'queued'; queue.unshift(j); return; }
    j.swap = engineFamily !== 'anima';
    j.train.stage = 'starting';
    log(`job ${j.id} start training "${p.name}": ${p.images} images × ${p.repeats} repeats, ${p.epochs} epochs, batch ${p.batch_size} (${p.steps} steps), rank ${p.rank}, ${p.optimizer} lr ${p.lr}, ${p.resolution}px`);
    emitJob(j);
    emitQueue();
    return;
  }
  if (j.kind === 'graph') {
    const eg = j.engineGraph || j.graph;
    if (!Object.keys(eg).length) {
      // every node was a constant pack node, already run when the job was queued
      emitJob(j);
      emitFolded(j);
      running = null;
      finishJob(j, 'done', { totalMs: Object.values((j.extFold && j.extFold.ms) || {}).reduce((a, b) => a + b, 0) });
      return;
    }
    const req = { id: j.id, cmd: 'graph', graph: eg, tokens: j.tokens, images: j.images, out_dir: TMP, preview: true };
    const ext = {};
    for (const n of Object.values(eg)) if (j.extDefs && j.extDefs[n.class_type]) ext[n.class_type] = { outputs: j.extDefs[n.class_type].entry.output.slice() };
    if (Object.keys(ext).length) {
      req.ext = ext;
      j.ext = newExtSession(j);
    }
    if (!engine.send(req)) { if (j.ext) { j.ext.close(); j.ext = null; } running = null; j.status = 'queued'; queue.unshift(j); return; }
    if (Object.values(eg).some(n => n.class_type === 'UNETLoader')) {
      if (engineFamily !== 'anima') j.swap = true;
      engineFamily = 'anima';
      engineDit = null;  // the graph picks its own DiT
    }
    log(`job ${j.id} start graph: ${Object.keys(eg).length} nodes (${p.classes.join(', ')})${req.ext ? ' packs: ' + Object.keys(req.ext).join(', ') : ''} (${queue.length} queued)`);
    emitJob(j);
    emitFolded(j);
    emitQueue();
    return;
  }
  j.tmpOut = path.join(TMP, j.id + '_' + process.pid + '.rgb');
  // the first job of the other model family makes the engine swap models in VRAM (20-35 s)
  const swap = j.family !== engineFamily || (j.family === 'anima' && !!j.dit && !sameFile(j.dit, engineDit));
  const req = j.family === 'sdxl' ? {
    id: j.id, cmd: 'generate', family: 'sdxl', checkpoint: j.checkpoint, sd: j.sd,
    width: p.width, height: p.height, steps: p.steps, cfg: p.cfg, seed: p.seed,
    sampler: p.sampler, scheduler: p.scheduler, out: j.tmpOut, preview: true,
  } : {
    id: j.id, cmd: 'generate',
    qwen_ids: j.enc.qwen_ids, t5_ids: j.enc.t5_ids, t5_weights: j.enc.t5_weights,
    width: p.width, height: p.height, steps: p.steps, cfg: p.cfg, seed: p.seed,
    sampler: p.sampler, shift: p.shift, loras: j.loraPaths, out: j.tmpOut, preview: true,
  };
  if (j.family === 'anima' && j.dit) req.dit = j.dit;
  if (j.neg) req.neg = { qwen_ids: j.neg.qwen_ids, t5_ids: j.neg.t5_ids, t5_weights: j.neg.t5_weights };
  // only meaningful with a negative pass; absent = 1 (always) for older engine builds
  if (p.cfg > 1 && p.cfg_cutoff < 1) req.cfg_cutoff = p.cfg_cutoff;
  if (p.step_cache > 0) req.step_cache = p.step_cache;
  if (p.nag && p.cfg <= 1 && j.neg) req.nag = { enabled: true, scale: p.nag.scale, tau: p.nag.tau, alpha: p.nag.alpha };
  if (p.hires) req.hires = p.hires;
  if (p.face) req.face = p.face;
  if (p.upscale) req.upscale = p.upscale;
  if (!engine.send(req)) {
    running = null;
    j.status = 'queued';
    queue.unshift(j);
    return;
  }
  j.swap = swap; engineFamily = j.family;
  if (j.family === 'anima' && j.dit) {
    engineDit = j.dit;
    if (!sameFile(lastDit, j.dit)) { lastDit = j.dit; engine.args = engineArgs(); }
  }
  const enh = [p.hires && `hires=${p.hires.scale}x/${p.hires.steps}st/${p.hires.upscaler}`, p.face && `face=${p.face.steps}st`, p.upscale && `upscale=${p.upscale.factor}x`].filter(Boolean).join(' ');
  log(`job ${j.id} start${j.family === 'sdxl' ? ' SDXL ' + path.basename(j.checkpoint) + (j.swap ? ' (model swap)' : '') + ' ' + p.scheduler : ''} ${p.width}x${p.height} steps=${p.steps} cfg=${p.cfg}${p.cfg > 1 && p.cfg_cutoff < 1 ? '@' + p.cfg_cutoff : ''} seed=${p.seed}${p.sampler !== 'euler' ? ' ' + p.sampler : ''}${p.step_cache > 0 ? ' cache=' + p.step_cache : ''}${req.nag ? ' nag=' + req.nag.scale : ''}${enh ? ' ' + enh : ''} (${queue.length} queued)`);
  emitJob(j);
  emitQueue();
}

function finishJob(j, status, extra = {}) {
  Object.assign(j, extra);
  if (j.gpuLease) { if (j.gpuLease !== true) releaseGpu(j.gpuLease); j.gpuLease = null; }
  j.status = status;
  j.finished = Date.now();
  if (j.started) j.wallMs = j.finished - j.started;
  if (j.tmpOut) fs.promises.unlink(j.tmpOut).catch(() => { });
  j.enc = null; j.neg = null; j.sd = null; // free token arrays
  if (j.kind === 'graph') {
    j.tokens = null; j.images = null; j.node = null;
    if (j.ext) { j.ext.close(); j.ext = null; }
    j.extDefs = null; j.extFold = null; j.engineGraph = null;
    X.cleanupTensors(TMP, j.id);
  }
  log(`job ${j.id} ${status}${j.error ? ': ' + j.error : ''}${j.totalMs ? ` (${(j.totalMs / 1000).toFixed(2)} s)` : ''}`);
  emitJob(j);
  emitQueue();
  setImmediate(pump);
}

function onEngineEvent(ev) {
  if (ev.ev === 'info') {
    engineInfo = ev;
    const w = infoWaiters.get(ev.id);
    if (w) { infoWaiters.delete(ev.id); w(ev); }
    return;
  }
  if (ev.ev === 'loading' && !ev.id) { broadcast({ type: 'engine', engine: engineStatus() }); return; }
  const j = ev.id ? jobs.get(ev.id) : null;
  if (!j) {
    if (ev.ev === 'error') log(`engine error${ev.id ? ' (' + ev.id + ')' : ''}: ${ev.msg}`);
    return;
  }
  if (j.kind === 'graph') return onGraphEvent(j, ev);
  if (j.kind === 'train') return onTrainEvent(j, ev);
  if (j.kind === 'tag') return onTagEvent(j, ev);
  switch (ev.ev) {
    case 'loading':
      j.loading = { what: ev.what, progress: ev.progress };
      broadcast({ type: 'loading', id: j.id, what: ev.what, progress: ev.progress });
      break;
    case 'encoded':
      j.encodeMs = ev.ms; j.loading = null;
      broadcast({ type: 'encoded', id: j.id, ms: ev.ms });
      break;
    case 'step': {
      // stages: base -> hires -> face (once per detected face); numbering restarts per stage/face
      const stage = typeof ev.stage === 'string' && ev.stage ? ev.stage : 'base';
      let g = j.timeline[j.timeline.length - 1];
      if (!g || g.stage !== stage || (ev.step <= 1 && g.ms.length)) {
        g = { stage, of: ev.of || 0, ms: [] };
        if (stage === 'face') g.face = j.timeline.filter(x => x.stage === 'face').length + 1;
        j.timeline.push(g);
      }
      if (ev.of) g.of = ev.of;
      if (typeof ev.ms === 'number') {
        g.ms.push(ev.ms);
        if (stage === 'base') j.stepMs.push(ev.ms);
      }
      j.stage = stage; j.face = g.face || 0;
      j.step = ev.step; j.of = ev.of || j.of; j.loading = null;
      broadcast({ type: 'step', id: j.id, stage, face: g.face || 0, step: ev.step, of: j.of, ms: ev.ms, preview: ev.preview || null });
      break;
    }
    case 'decoded':
      j.decodeMs = ev.ms;
      broadcast({ type: 'decoded', id: j.id, ms: ev.ms });
      break;
    case 'done':
      if (running === j) running = null;
      j.timings = cleanTimings(ev.timings);
      saveResult(j, ev).then(
        (image) => finishJob(j, 'done', { image, totalMs: ev.total_ms ?? (Date.now() - j.started) }),
        (e) => finishJob(j, 'error', { error: 'saving image failed: ' + e.message }));
      break;
    case 'cancelled':
      if (running === j) running = null;
      finishJob(j, 'cancelled');
      break;
    case 'error':
      if (running === j) running = null;
      finishJob(j, 'error', { error: ev.msg || 'engine error' });
      break;
    default:
      broadcast({ type: 'engine-event', id: j.id, ev });
  }
}

// ---------------------------------------------------------------------------
// saving images
// ---------------------------------------------------------------------------
function pad(n, w = 2) { return String(n).padStart(w, '0'); }
function asciiJSON(obj) {
  return JSON.stringify(obj).replace(/[\u007f-￿]/g, (c) => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
}

// keep only numbers / small numeric arrays / flat objects of numbers from the engine's timings
function cleanTimings(t) {
  if (!t || typeof t !== 'object') return null;
  const out = {};
  for (const [k, v] of Object.entries(t)) {
    if (!/^[a-z0-9_]{1,40}$/i.test(k)) continue;
    if (typeof v === 'number' && Number.isFinite(v)) out[k] = v;
    else if (typeof v === 'boolean') out[k] = v;
    else if (Array.isArray(v) && v.length <= 256 && v.every(x => typeof x === 'number' && Number.isFinite(x))) out[k] = v;
  }
  return out;
}

const VIEW_MAX = 2048;   // stage / lightbox display size for big (upscaled) images
const VIEW_OVER = 2600;  // only images whose long side exceeds this get a display version
const THUMB_MAX = 320;
const needsView = (w, h) => Math.max(w, h) > VIEW_OVER;
const viewUrl = (file, w, h) => needsView(w, h) ? `/api/thumb/${file}?max=${VIEW_MAX}` : '/outputs/' + file;

async function saveResult(j, ev) {
  // FINAL size comes from the engine (hires scale / upscale factor change it)
  const w = ev.w || j.params.width, h = ev.h || j.params.height;
  const src = ev.out || j.tmpOut;
  const rgb = await fsp.readFile(src);
  if (rgb.length < w * h * 3) throw new Error(`engine output is ${rgb.length} bytes, expected ${w}x${h}x3 = ${w * h * 3}`);
  const d = new Date();
  const day = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}-${pad(d.getMilliseconds(), 3)}`;
  const dir = path.join(OUTPUTS, day);
  await fsp.mkdir(dir, { recursive: true });
  let name = `kiln_${stamp}_${j.params.seed}.png`;
  for (let k = 2; fs.existsSync(path.join(dir, name)); k++) name = `kiln_${stamp}_${j.params.seed}_${k}.png`;
  const totalMs = ev.total_ms ?? (Date.now() - j.started);
  // width/height stay the REQUESTED base size (so "reuse settings" reproduces the run);
  // out_w/out_h is the final image size.
  const meta = {
    kiln: 1, ...j.params, out_w: w, out_h: h,
    engine: engine.mock ? 'mock' : 'kiln-engine', total_ms: totalMs,
    step_ms: j.stepMs, encode_ms: j.encodeMs, decode_ms: j.decodeMs,
    timings: j.timings || undefined,
    stages: j.timeline && j.timeline.some(g => g.stage !== 'base') ? j.timeline.filter(g => g.stage !== 'base') : undefined,
    created: d.toISOString(), t5_mode: getT5Mode(),
  };
  const big = w * h > 8e6;
  const png = await encodePNGAsync(rgb, w, h, { parameters: A.infotext(meta, { Version: 'Kiln ' + VERSION }), kiln: asciiJSON(meta), Software: 'Kiln ' + VERSION }, big ? 4 : 6);
  await fsp.writeFile(path.join(dir, name), png);
  if (ev.out && ev.out !== j.tmpOut && path.resolve(ev.out).startsWith(TMP + path.sep)) fsp.unlink(ev.out).catch(() => { });
  const file = `${day}/${name}`;
  // pre-build the thumbnail (and the display-size version for big images) from the raw RGB
  try {
    if (needsView(w, h)) {
      // one pass over the big buffer: full -> display size, then display -> thumb
      const s = VIEW_MAX / Math.max(w, h);
      const vw = Math.max(1, Math.round(w * s)), vh = Math.max(1, Math.round(h * s));
      const vrgb = downscaleRGB(rgb, w, h, vw, vh);
      cachePut(viewCache, file, encodePNG(vrgb, vw, vh, {}, 6), 8);
      cachePut(thumbCache, file, thumbnailFromRGB(vrgb, vw, vh, THUMB_MAX), 400);
    } else {
      cachePut(thumbCache, file, thumbnailFromRGB(rgb, w, h, THUMB_MAX), 400);
    }
  } catch (_) { /* built lazily instead */ }
  galleryDirty = true;
  metaCache.set(file, { mtime: Date.now(), params: meta, w, h });
  return { file, url: '/outputs/' + file, thumb: '/api/thumb/' + file, view: viewUrl(file, w, h), w, h, size: png.length };
}

// ---------------------------------------------------------------------------
// gallery
// ---------------------------------------------------------------------------
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const FILE_RE = /^(\d{4}-\d{2}-\d{2})\/([^/\\:*?"<>|]+\.png)$/i;
let galleryList = [];
let galleryAt = 0;
let galleryDirty = true;
const metaCache = new Map();

async function refreshGallery() {
  if (!galleryDirty && Date.now() - galleryAt < 5000) return galleryList;
  const list = [];
  let days = [];
  try { days = (await fsp.readdir(OUTPUTS, { withFileTypes: true })).filter(e => e.isDirectory() && DAY_RE.test(e.name)).map(e => e.name); } catch (_) { }
  for (const day of days) {
    let files = [];
    try { files = await fsp.readdir(path.join(OUTPUTS, day)); } catch (_) { continue; }
    for (const f of files) {
      if (!/\.png$/i.test(f)) continue;
      try {
        const st = await fsp.stat(path.join(OUTPUTS, day, f));
        list.push({ file: `${day}/${f}`, mtime: st.mtimeMs, size: st.size });
      } catch (_) { }
    }
  }
  list.sort((a, b) => b.mtime - a.mtime || (a.file < b.file ? 1 : -1));
  galleryList = list;
  galleryAt = Date.now();
  galleryDirty = false;
  return list;
}

async function readMeta(file, mtime) {
  const c = metaCache.get(file);
  if (c && (!mtime || c.mtime >= mtime - 1)) return c;
  let params = null, w = 0, h = 0;
  try {
    const fh = await fsp.open(path.join(OUTPUTS, file), 'r');
    try {
      const buf = Buffer.alloc(256 * 1024);
      const { bytesRead } = await fh.read(buf, 0, buf.length, 0);
      const b = buf.subarray(0, bytesRead);
      const sz = readPNGSize(b);
      if (sz) { w = sz.w; h = sz.h; }
      const t = readPNGText(b);
      if (t.kiln) { try { params = JSON.parse(t.kiln); } catch (_) { } }
      else if (t.prompt || t.parameters) params = { foreign: true, text: t.parameters || t.prompt };
    } finally { await fh.close(); }
  } catch (_) { }
  const m = { mtime: mtime || Date.now(), params, w, h };
  metaCache.set(file, m);
  return m;
}

function resolveOutput(file) {
  const m = FILE_RE.exec(file || '');
  if (!m) return null;
  const p = path.resolve(OUTPUTS, m[1], m[2]);
  if (!p.startsWith(OUTPUTS + path.sep)) return null;
  return p;
}

// LRU caches: small gallery thumbs, and a few display-size (2048px) versions of big images
const thumbCache = new Map();
const viewCache = new Map();
function cachePut(cache, key, val, limit) {
  cache.delete(key);
  cache.set(key, val);
  while (cache.size > limit) cache.delete(cache.keys().next().value);
}
// Returns a PNG Buffer, the string 'original' (image already small enough), or null.
async function getThumb(file, max) {
  const view = max > THUMB_MAX;
  const cache = view ? viewCache : thumbCache;
  const hit = cache.get(file);
  if (hit) { cachePut(cache, file, hit, view ? 8 : 400); return hit; }
  const p = resolveOutput(file);
  if (!p) return null;
  if (view) {
    const m = await readMeta(file);
    if (m.w && m.h && !needsView(m.w, m.h)) return 'original';
  }
  const buf = await fsp.readFile(p);
  let t;
  try { t = thumbnailPNG(buf, view ? VIEW_MAX : THUMB_MAX, view ? 6 : 9); } catch (_) { t = view ? 'original' : buf; }
  if (t !== 'original') cachePut(cache, file, t, view ? 8 : 400);
  return t;
}

// ---------------------------------------------------------------------------
// Nodes mode: catalog lists, uploaded input images, graph jobs, workflows
// ---------------------------------------------------------------------------
function listDir(dir, re) {
  try { return fs.readdirSync(dir, { withFileTypes: true }).filter(e => e.isFile() && re.test(e.name)).map(e => e.name).sort((a, b) => a.localeCompare(b)); }
  catch (_) { return []; }
}
function modelLists() {
  const md = listModels();
  // names like ComfyUI: relative to the model-type folder (flat models/ files keep their bare name)
  const byKind = (k) => [...new Set(md.checkpoints.filter(m => m.kind === k).map(m => m.rel || m.file))];
  return {
    diffusion_models: byKind('diffusion'),
    text_encoders: byKind('text_encoder'),
    vaes: byKind('vae'),
    checkpoints: byKind('checkpoint'),
    // LoRA names relative to models/loras/ (then models/), like ComfyUI
    loras: [...new Set(md.loras.map(l => l.file.replace(/^loras\//, '')))],
    upscale_models: [...new Set([...listDir(path.join(MODELS, 'upscale'), /\.(safetensors|pth|pt)$/i), ...scanTree(path.join(MODELS, 'upscale_models'), /\.(safetensors|pth|pt)$/i)])],
    bbox_models: [...new Set([...listDir(path.join(MODELS, 'detect'), /\.safetensors$/i).map(n => 'bbox/' + n.replace(/\.safetensors$/i, '.pt')),
      ...scanTree(path.join(MODELS, 'ultralytics', 'bbox'), /\.(safetensors|pt)$/i).map(n => 'bbox/' + n.replace(/\.safetensors$/i, '.pt'))])],
    samplers: G.SAMPLERS.slice(),
    schedulers: G.SCHEDULERS.slice(),
    images: listInputImages(),
  };
}
function catalogRaw() { return G.loadCatalog(CATALOG_FILE); }
// built-in nodes + pack nodes (each tagged kiln_pack: "<pack name>")
function catalogAll() { return Object.assign({}, catalogRaw(), packs.catalog); }
function catalogFilled() { return G.fillCatalog(catalogAll(), modelLists()); }
// socket types only the engine produces / consumes (a pack node with such an output is never folded)
let engineTypesOf = null, engineTypesSet = null;
function engineTypes() {
  const cat = catalogRaw();
  if (engineTypesOf !== cat) {
    const t = new Set();
    for (const e of Object.values(cat)) {
      for (const o of e.output || []) t.add(o);
      for (const sect of ['required', 'optional']) for (const spec of Object.values((e.input && e.input[sect]) || {})) if (typeof spec[0] === 'string' && !spec[0].startsWith('$')) t.add(spec[0]);
    }
    for (const p of ['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO', '*']) t.delete(p);
    engineTypesOf = cat; engineTypesSet = t;
  }
  return engineTypesSet;
}

// ---- model library + CivitAI (civitai.red / civitai.com) ---------------------------------
const civitai = new CivitAI({ configDir: CONFIG, modelsDir: MODELS, log, testBase: process.env.KILN_CIVITAI_BASE || null });
const hasher = new SF.Hasher(path.join(CONFIG, 'hashes.json'), log);
const library = createLibrary({
  MODELS, OUTPUTS, CONFIG, log, broadcast, sendJSON, readJSON, listModels, civitai, hasher,
  isLocal: (req) => EXT_LAN || isLoopback(req), manageDenied: (req) => manageDenied(req),
  canRun: (fam) => canRunFamily(fam),
});
civitai.on('download', (d) => broadcast({ type: 'download', download: d }));
const datasets = createDatasets({ ROOT, CONFIG });
const taggers = createTaggers({ MODELS, onProgress: (d) => broadcast({ type: 'tagger', download: d }) });
const grab = createGrab({ CONFIG, datasets, broadcast, log });
civitai.on('installed', (d) => { log(`installed ${d.rel} from CivitAI`); broadcast({ type: 'models' }); });
hasher.onProgress = (st) => broadcast({ type: 'hashing', hasher: st });

// ---- packs (Kiln/extensions/<pack>/, docs/NODE_API.md) ----------------------------
const packs = new Packs({ dir: EXTENSIONS, builtin: () => catalogRaw(), log });
function extChanged() { broadcast({ type: 'extensions', extensions: packs.summary() }); }
function logPackError(pack, node, cls, e) {
  const stack = String((e && e.stack) || e).split('\n').filter(l => !/node:internal|[\\/]server[\\/]lib[\\/]extrun\.js/.test(l)).slice(0, 4).join(' | ');
  log(`[${pack}] #${node} ${cls || ''} failed: ${stack}`);
}
function newExtSession(j) {
  return new X.ExtSession({
    jobId: j.id, defs: j.extDefs, dir: TMP, engineTypes: engineTypes(), cache: j.extFold ? j.extFold.cache : {},
    send: (obj) => engine.send(obj),
    tokenize: (text) => encodePrompt(text),
    onProgress: (node, done, total) => { j.step = done; j.of = total; j.node = node; broadcast({ type: 'step', id: j.id, node, step: done, of: total, ms: null }); },
    onLog: (node, pack, msg) => log(`[${pack}] #${node} ${msg}`),
    onUi: (node, ui) => broadcast({ type: 'gext', id: j.id, node, ui }),
    onError: (node, pack, e) => logPackError(pack, node, j.graph[node] && j.graph[node].class_type, e),
  });
}
// gnode events for pack nodes that were folded (run at queue time) and dropped from the engine graph
function emitFolded(j) {
  const f = j.extFold;
  if (!f) return;
  for (const id of f.folded) {
    const ct = j.graph[id] && j.graph[id].class_type;
    if (f.removed.includes(id)) {
      j.nodeStatus[id] = 'done';
      (j.nodeMs = j.nodeMs || {})[id] = f.ms[id];
      broadcast({ type: 'gnode', id: j.id, node: id, class_type: ct, status: 'start' });
      broadcast({ type: 'gnode', id: j.id, node: id, class_type: ct, status: 'done', ms: f.ms[id], folded: true });
    }
    if (f.ui[id]) broadcast({ type: 'gext', id: j.id, node: id, ui: f.ui[id] });
  }
}

// ---- uploaded images (inputs/) ------------------------------------------------
const INPUT_NAME_RE = /^[^/\\:*?"<>|\x00-\x1f]{1,180}\.png$/i;
function listInputImages() { return listDir(INPUTS, /\.png$/i).filter(n => INPUT_NAME_RE.test(n)); }
function inputPath(name) {
  if (typeof name !== 'string' || !INPUT_NAME_RE.test(name) || name.startsWith('.')) return null;
  const p = path.resolve(INPUTS, name);
  return p.startsWith(INPUTS + path.sep) ? p : null;
}
function sanitizeInputName(name) {
  let base = path.basename(String(name || 'image.png')).replace(/[/\\:*?"<>|\x00-\x1f]/g, '_').replace(/^\.+/, '').trim();
  base = base.replace(/\.[a-z0-9]{1,5}$/i, '') || 'image';
  return base.slice(0, 150) + '.png';
}
// decoded RGBA8 cache: inputs/.raw/<name>.rgba + <name>.json {w,h,mtime,size}
async function ensureRaw(name) {
  const src = inputPath(name);
  if (!src) throw new Error('bad image name');
  const st = await fsp.stat(src);
  const rawFile = path.join(INPUTS_RAW, name + '.rgba');
  const metaFile = path.join(INPUTS_RAW, name + '.json');
  try {
    const m = JSON.parse(await fsp.readFile(metaFile, 'utf8'));
    if (m.mtime === st.mtimeMs && m.size === st.size && fs.existsSync(rawFile)) return { path: rawFile, w: m.w, h: m.h };
  } catch (_) { /* (re)decode */ }
  const img = decodePNGRGBA(await fsp.readFile(src));
  await fsp.mkdir(INPUTS_RAW, { recursive: true });
  await fsp.writeFile(rawFile, img.rgba);
  await fsp.writeFile(metaFile, JSON.stringify({ w: img.w, h: img.h, mtime: st.mtimeMs, size: st.size, alpha: img.hasAlpha }));
  return { path: rawFile, w: img.w, h: img.h };
}
async function saveUpload(buf, wantedName) {
  if (!isPNG(buf)) throw Object.assign(new Error('only PNG uploads are accepted; convert other formats to PNG in the browser (canvas.toBlob)'), { code: 415 });
  let img;
  try { img = decodePNGRGBA(buf); } catch (e) { throw Object.assign(new Error('could not decode PNG: ' + e.message), { code: 400 }); }
  await fsp.mkdir(INPUTS, { recursive: true });
  let name = sanitizeInputName(wantedName);
  // same name: reuse if identical bytes, else "name (1).png" like ComfyUI
  for (let k = 1; ; k++) {
    const p = path.join(INPUTS, name);
    if (!fs.existsSync(p)) break;
    const old = await fsp.readFile(p);
    if (old.equals(buf)) break;
    name = sanitizeInputName(wantedName).replace(/\.png$/i, '') + ` (${k}).png`;
  }
  const p = path.join(INPUTS, name);
  if (!fs.existsSync(p)) await fsp.writeFile(p, buf);
  inputThumbCache.delete(name);
  const raw = await ensureRaw(name);
  return { name, w: img.w, h: img.h, alpha: img.hasAlpha, url: '/api/input/' + encodeURIComponent(name), thumb: '/api/input/thumb/' + encodeURIComponent(name), raw: path.basename(raw.path) };
}
// minimal multipart/form-data parser: returns [{name, filename, type, data}]
function parseMultipart(buf, contentType) {
  const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType || '');
  if (!m) throw new Error('multipart: no boundary');
  const boundary = Buffer.from('--' + (m[1] || m[2]).trim());
  const parts = [];
  let pos = buf.indexOf(boundary);
  while (pos >= 0) {
    let start = pos + boundary.length;
    if (buf[start] === 45 && buf[start + 1] === 45) break; // closing --
    if (buf[start] === 13 && buf[start + 1] === 10) start += 2;
    const next = buf.indexOf(boundary, start);
    if (next < 0) break;
    const part = buf.subarray(start, next - 2); // strip CRLF before boundary
    const hEnd = part.indexOf('\r\n\r\n');
    if (hEnd >= 0) {
      const head = part.subarray(0, hEnd).toString('utf8');
      const disp = /content-disposition:[^\r\n]*/i.exec(head);
      const nm = disp && /\bname="([^"]*)"/i.exec(disp[0]);
      const fn = disp && /\bfilename="([^"]*)"/i.exec(disp[0]);
      const ct = /content-type:\s*([^\r\n]+)/i.exec(head);
      parts.push({ name: nm ? nm[1] : '', filename: fn ? fn[1] : null, type: ct ? ct[1].trim() : '', data: part.subarray(hEnd + 4) });
    }
    pos = next;
  }
  return parts;
}
const inputThumbCache = new Map();
async function inputThumb(name) {
  const hit = inputThumbCache.get(name);
  if (hit) return hit;
  const raw = await ensureRaw(name);
  const rgba = await fsp.readFile(raw.path);
  // composite over a dark checker so transparent areas stay visible
  const rgb = Buffer.alloc(raw.w * raw.h * 3);
  for (let i = 0, y = 0; y < raw.h; y++) for (let x = 0; x < raw.w; x++, i++) {
    const a = rgba[i * 4 + 3] / 255, bg = ((x >> 3) + (y >> 3)) & 1 ? 58 : 38;
    for (let c = 0; c < 3; c++) rgb[i * 3 + c] = Math.round(rgba[i * 4 + c] * a + bg * (1 - a));
  }
  const t = thumbnailFromRGB(rgb, raw.w, raw.h, 256, 6);
  inputThumbCache.set(name, t);
  if (inputThumbCache.size > 200) inputThumbCache.delete(inputThumbCache.keys().next().value);
  return t;
}

// ---- graph jobs -----------------------------------------------------------------
async function createGraphJob(body) {
  const cat = catalogAll();
  const chk = G.checkGraph(body.graph, cat, { textLinks: true });
  if (!chk.ok) return { status: 400, body: { error: chk.error, node_errors: chk.errors, unknown: chk.unknown } };
  const graph = chk.graph;
  // pack nodes: snapshot their definitions for this job, then fold the constant ones
  const defs = {};
  for (const n of Object.values(graph)) {
    const d = packs.get(n.class_type);
    if (d) defs[n.class_type] = { def: d.def, entry: d.entry, pack: { name: d.pack.name, dir: d.pack.dir } };
  }
  let engineGraph = graph, extFold = null;
  if (Object.keys(defs).length) {
    const f = await X.fold(graph, {
      defs, engineTypes: engineTypes(),
      log: (id, pack, msg) => log(`[${pack}] #${id} ${msg}`),
      onError: (id, pack, e) => logPackError(pack, id, graph[id].class_type, e),
    });
    if (!f.ok) return { status: 400, body: { error: 'invalid graph: ' + Object.entries(f.errors).map(([id, m]) => `#${id} ${graph[id].class_type}: ${m}`).join(' | '), node_errors: f.errors } };
    extFold = f;
    engineGraph = f.engineGraph;
    if (Object.keys(engineGraph).length) {
      // folded values were substituted as literals: validate / coerce again
      const chk2 = G.checkGraph(engineGraph, cat);
      if (chk2.ok) engineGraph = chk2.graph;
      else if (!Object.keys(chk2.errors).length && !chk2.unknown.length && /no output node/.test(chk2.error)) engineGraph = {};
      else return { status: 400, body: { error: chk2.error, node_errors: chk2.errors, unknown: chk2.unknown } };
    }
    const inEngine = Object.keys(engineGraph).filter(id => defs[engineGraph[id].class_type]);
    const ef = engineInfo && engineInfo.features;
    if (inEngine.length && ef && !ef.ext) {
      const msg = 'this kiln-engine build cannot run pack nodes yet (no ext_call support)';
      return { status: 400, body: { error: msg, node_errors: Object.fromEntries(inEngine.map(id => [id, msg])) } };
    }
  }
  const tokens = {}, images = {}, errors = {};
  for (const [id, n] of Object.entries(engineGraph)) {
    if (n.class_type === 'CLIPTextEncode') {
      try { tokens[id] = encodePrompt(n.inputs.text); } catch (e) { errors[id] = 'tokenize failed: ' + e.message; }
    } else if (n.class_type === 'LoadImage') {
      const name = n.inputs.image;
      if (!inputPath(name) || !fs.existsSync(inputPath(name))) { errors[id] = `image "${name}" not found in inputs/ (upload it first)`; continue; }
      try { images[id] = await ensureRaw(name); } catch (e) { errors[id] = 'cannot read image: ' + e.message; }
    }
  }
  if (Object.keys(errors).length) return { status: 400, body: { error: 'invalid graph: ' + Object.entries(errors).map(([id, m]) => `#${id}: ${m}`).join(' | '), node_errors: errors } };
  const summary = G.summarize(Object.keys(engineGraph).length ? engineGraph : graph);
  const ui = body.ui && typeof body.ui === 'object' ? body.ui : null;
  const j = {
    id: 'j' + (++jobSeq), kind: 'graph', group: 'g' + (++groupSeq), index: 0, count: 1, status: 'queued', created: Date.now(),
    params: summary, graph, engineGraph, extDefs: defs, extFold, tokens, images, ui, nodeStatus: {}, outputs: [], saving: [], step: 0, of: 0, stepMs: [], timeline: [],
  };
  jobs.set(j.id, j);
  queue.push(j);
  emitJob(j);
  emitQueue();
  pump();
  return { status: 200, body: { id: j.id, group: j.group, nodes: Object.keys(graph), pruned: chk.prunedCount, outputs: chk.outputs, folded: extFold ? extFold.folded : [] } };
}

function onGraphEvent(j, ev) {
  switch (ev.ev) {
    case 'node': {
      const node = ev.node != null ? String(ev.node) : null;
      if (!node) break;
      j.nodeStatus[node] = ev.status === 'done' ? 'done' : 'running';
      if (ev.status === 'start') j.node = node;
      if (ev.status === 'done' && typeof ev.ms === 'number') { j.nodeMs = j.nodeMs || {}; j.nodeMs[node] = ev.ms; }
      broadcast({ type: 'gnode', id: j.id, node, class_type: ev.class_type, status: ev.status, ms: ev.ms });
      break;
    }
    case 'loading':
      broadcast({ type: 'loading', id: j.id, what: ev.what, progress: ev.progress, node: ev.node != null ? String(ev.node) : undefined });
      break;
    case 'step': {
      const node = ev.node != null ? String(ev.node) : j.node;
      j.step = ev.step; j.of = ev.of; j.node = node;
      broadcast({ type: 'step', id: j.id, node, step: ev.step, of: ev.of, ms: ev.ms, preview: ev.preview || null });
      break;
    }
    case 'image': {
      const p = saveGraphImage(j, ev).then(
        (img) => { if (img) { j.outputs.push(img); broadcast({ type: 'gimage', id: j.id, node: img.node, index: img.index, kind: img.kind, image: img }); } },
        (e) => log(`job ${j.id}: saving image from node ${ev.node} failed: ${e.message}`));
      j.saving.push(p);
      break;
    }
    case 'done':
      if (running === j) running = null;
      Promise.allSettled(j.saving).then(() => finishJob(j, 'done', { totalMs: ev.total_ms ?? (Date.now() - j.started), nodeMs: ev.node_ms || j.nodeMs, saving: [] }));
      break;
    case 'cancelled':
      if (running === j) running = null;
      Promise.allSettled(j.saving).then(() => finishJob(j, 'cancelled', { saving: [] }));
      break;
    case 'error':
      if (running === j) running = null;
      Promise.allSettled(j.saving).then(() => finishJob(j, 'error', { error: ev.msg || 'engine error', errorNode: ev.node != null ? String(ev.node) : null, saving: [] }));
      break;
    case 'ext_call':
      if (j.ext) j.ext.handleCall(ev);
      else engine.send({ cmd: 'ext_result', id: j.id, call: ev.call, error: 'this job has no pack nodes' });
      break;
    case 'ext_op_done':
      if (j.ext) j.ext.handleOpDone(ev);
      break;
    default:
      broadcast({ type: 'engine-event', id: j.id, ev });
  }
}

const sanitizePrefix = (p) => (String(p || 'Kiln').split(/[/\\]/).filter(Boolean).pop() || 'Kiln').replace(/[^\w.-]+/g, '_').slice(0, 60) || 'Kiln';

async function saveGraphImage(j, ev) {
  const w = ev.w | 0, h = ev.h | 0;
  if (!w || !h || !ev.out) throw new Error('image event without size/out');
  const src = path.resolve(String(ev.out));
  const rgb = await fsp.readFile(src);
  if (rgb.length < w * h * 3) throw new Error(`raw image is ${rgb.length} bytes, expected ${w * h * 3}`);
  if (src.startsWith(TMP + path.sep)) fsp.unlink(src).catch(() => { });
  const node = String(ev.node), index = ev.index | 0;
  const kind = ev.kind === 'preview' ? 'preview' : 'save';
  if (kind === 'preview') {
    await fsp.mkdir(PREVIEWS, { recursive: true });
    const name = `${j.id}_${node.replace(/\W/g, '_')}_${index}_${Date.now().toString(36)}.png`;
    const png = await encodePNGAsync(rgb, w, h, {}, 3);
    await fsp.writeFile(path.join(PREVIEWS, name), png);
    prunePreviews();
    const url = '/api/preview/' + name;
    return { node, index, kind, w, h, url, thumb: url, view: url, size: png.length };
  }
  const d = new Date();
  const day = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}-${pad(d.getMilliseconds(), 3)}`;
  const dir = path.join(OUTPUTS, day);
  await fsp.mkdir(dir, { recursive: true });
  const prefix = sanitizePrefix(ev.prefix || (j.graph[node] && j.graph[node].inputs.filename_prefix));
  let name = `${prefix}_${stamp}_${String(index).padStart(2, '0')}.png`;
  for (let k = 2; fs.existsSync(path.join(dir, name)); k++) name = `${prefix}_${stamp}_${String(index).padStart(2, '0')}_${k}.png`;
  const s = j.params;
  const meta = {
    kiln: 1, kind: 'graph', prompt: s.prompt, negative: s.negative, seed: s.seed, steps: s.steps, cfg: s.cfg, sampler: s.sampler,
    width: w, height: h, out_w: w, out_h: h, node, index, engine: engine.mock ? 'mock' : 'kiln-engine',
    created: d.toISOString(), graph: j.graph, ui: j.ui || undefined, t5_mode: getT5Mode(),
  };
  // "prompt" = ComfyUI API workflow, so ComfyUI can load our PNGs too; "parameters" = A1111 text
  const text = { kiln: asciiJSON(meta), prompt: asciiJSON(j.graph), Software: 'Kiln ' + VERSION };
  const info = graphInfotext(j.graph, s, w, h);
  if (info) text.parameters = info;
  const png = await encodePNGAsync(rgb, w, h, text, w * h > 8e6 ? 4 : 6);
  await fsp.writeFile(path.join(dir, name), png);
  const file = `${day}/${name}`;
  try { cachePut(thumbCache, file, thumbnailFromRGB(rgb, w, h, THUMB_MAX), 400); } catch (_) { }
  galleryDirty = true;
  metaCache.set(file, { mtime: Date.now(), params: meta, w, h });
  return { node, index, kind, file, url: '/outputs/' + file, thumb: '/api/thumb/' + file, view: viewUrl(file, w, h), w, h, size: png.length };
}
function graphInfotext(g, s, w, h) {
  try {
    const nodes = Object.values(g || {});
    const ks = nodes.find(n => /^KSampler/.test(n.class_type));
    const unet = nodes.find(n => n.class_type === 'UNETLoader' || n.class_type === 'UnetLoaderGGUF');
    const loras = nodes.filter(n => /^LoraLoader/.test(n.class_type) && typeof n.inputs.lora_name === 'string').map(n => ({ file: n.inputs.lora_name, strength: n.inputs.strength_model }));
    if (!ks && !nodes.some(n => n.class_type === 'ImageUpscaleWithModel')) return null;
    const I = ks ? ks.inputs : {};
    const extra = { Version: 'Kiln ' + VERSION };
    const up = nodes.find(n => n.class_type === 'UpscaleModelLoader');
    if (up) extra.Upscaler = String(up.inputs.model_name || '').replace(/\.[a-z]+$/i, '');
    return A.infotext({
      prompt: s.prompt || '', negative: s.negative || '', steps: typeof I.steps === 'number' ? I.steps : undefined, sampler: I.sampler_name, scheduler: I.scheduler,
      cfg: typeof I.cfg === 'number' ? I.cfg : undefined, seed: typeof I.seed === 'number' ? I.seed : undefined, width: w, height: h,
      model: unet ? String(unet.inputs.unet_name || '') : undefined, denoise: typeof I.denoise === 'number' ? I.denoise : undefined, loras,
    }, extra);
  } catch (_) { return null; }
}
function prunePreviews() {
  try {
    const files = fs.readdirSync(PREVIEWS).filter(f => f.endsWith('.png')).map(f => ({ f, t: fs.statSync(path.join(PREVIEWS, f)).mtimeMs })).sort((a, b) => b.t - a.t);
    for (const x of files.slice(200)) fs.unlinkSync(path.join(PREVIEWS, x.f));
  } catch (_) { }
}

// ---- saved workflows (workflows/*.json) -------------------------------------------
const WF_NAME_RE = /^[\w .()+-]{1,80}$/;
function workflowPath(name) {
  if (typeof name !== 'string' || !WF_NAME_RE.test(name) || name.trim() !== name || name.startsWith('.')) return null;
  const p = path.resolve(WORKFLOWS, name + '.json');
  return p.startsWith(WORKFLOWS + path.sep) ? p : null;
}
async function listWorkflows() {
  let ents = [];
  try { ents = await fsp.readdir(WORKFLOWS); } catch (_) { return []; }
  const out = [];
  for (const f of ents) {
    if (!f.endsWith('.json')) continue;
    const name = f.slice(0, -5);
    if (!workflowPath(name)) continue;
    try { const st = await fsp.stat(path.join(WORKFLOWS, f)); out.push({ name, mtime: st.mtimeMs, size: st.size }); } catch (_) { }
  }
  return out.sort((a, b) => b.mtime - a.mtime);
}

async function readRaw(req, limit) {
  return new Promise((resolve, reject) => {
    const parts = []; let n = 0;
    req.on('data', (c) => { n += c.length; if (n > limit) { reject(Object.assign(new Error('upload too large'), { code: 413 })); req.destroy(); } else parts.push(c); });
    req.on('end', () => resolve(Buffer.concat(parts)));
    req.on('error', reject);
  });
}

// ---- extensions panel API -----------------------------------------------------------
const isLoopback = (req) => !req.headers['x-kiln-remote'] && /^(127\.|::1$|::ffff:127\.)/.test(req.socket.remoteAddress || '');  // requests forwarded by a local reverse proxy come from 127.0.0.1 too
function manageDenied(req) {
  if (!EXT_LAN && !isLoopback(req)) return 'managing extensions is only allowed on the PC running Kiln (start the server with KILN_EXT_LAN=1 to allow other devices)';
  // browsers: same-origin pages only, so another website can't install packs through your browser
  const origin = req.headers.origin;
  if (origin !== undefined) {
    let host = null;
    try { host = new URL(origin).host; } catch (_) { }
    if (host !== req.headers.host) return 'cross-origin request refused';
  }
  if (req.method === 'POST' && !/^application\/json\b/i.test(req.headers['content-type'] || '')) return 'send JSON (Content-Type: application/json)';
  return null;
}
function extState(req) {
  return { dir: EXTENSIONS, manage: EXT_LAN || isLoopback(req), busy: packs.busy, summary: packs.summary(), packs: packs.list() };
}
async function handleExtRoutes(req, res, u, p, m) {
  if (p === '/api/extensions' && m === 'GET') { sendJSON(res, 200, extState(req)); return true; }
  if (!p.startsWith('/api/extensions/')) return false;
  const why = manageDenied(req);
  if (why) { sendJSON(res, 403, { error: why }); return true; }
  let body = {};
  if (m === 'POST') {
    try { body = await readJSON(req); } catch (e) { sendJSON(res, 400, { error: 'bad json: ' + e.message }); return true; }
  }
  const rest = p.slice('/api/extensions/'.length);
  try {
    let r = null;
    if (rest === 'reload' && m === 'POST') { packs.load(); r = { reloaded: true }; }
    else if (rest === 'install' && m === 'POST') r = { pack: await packs.install(body.source) };
    else if (rest === 'create' && m === 'POST') r = { pack: await packs.create(body) };
    else {
      const mm = rest.match(/^([^/]+)(?:\/(enable|disable|update))?$/);
      if (mm && mm[2] && m === 'POST') r = { pack: mm[2] === 'update' ? await packs.update(mm[1]) : await packs.setEnabled(mm[1], mm[2] === 'enable') };
      else if (mm && !mm[2] && m === 'DELETE') r = await packs.uninstall(mm[1]);
    }
    if (!r) { sendJSON(res, 404, { error: 'no such endpoint' }); return true; }
    extChanged();
    sendJSON(res, 200, Object.assign(extState(req), r));
  } catch (e) {
    sendJSON(res, Number.isInteger(e.code) && e.code >= 400 && e.code < 600 ? e.code : 500, { error: e.message });
  }
  return true;
}

// Routes for Nodes mode; returns true when handled.
async function handleGraphRoutes(req, res, u, p, m) {
  if (p === '/api/nodes' && m === 'GET') { sendJSON(res, 200, catalogFilled()); return true; }
  if (p === '/api/graph/examples' && m === 'GET') { sendJSON(res, 200, G.examples(modelLists())); return true; }
  if (p === '/api/graph' && m === 'POST') {
    let body;
    try { body = JSON.parse((await readBody(req, 16 << 20)) || '{}'); } catch (e) { sendJSON(res, 400, { error: 'bad json: ' + e.message }); return true; }
    const r = await createGraphJob(body);
    sendJSON(res, r.status, r.body);
    return true;
  }
  if (p === '/api/upload/image' && m === 'POST') {
    try {
      const buf = await readRaw(req, 64 << 20);
      const ct = req.headers['content-type'] || '';
      let data = buf, name = u.searchParams.get('name') || req.headers['x-filename'] || 'image.png';
      if (/multipart\/form-data/i.test(ct)) {
        const parts = parseMultipart(buf, ct);
        const f = parts.find(x => x.name === 'image' && x.filename != null) || parts.find(x => x.filename != null);
        if (!f) { sendJSON(res, 400, { error: 'multipart upload without a file part (field "image")' }); return true; }
        data = f.data; name = f.filename || name;
      }
      const r = await saveUpload(data, decodeURIComponent(String(name)));
      log(`uploaded input ${r.name} (${r.w}x${r.h})`);
      broadcast({ type: 'inputs', images: listInputImages() });
      sendJSON(res, 200, r);
    } catch (e) { sendJSON(res, e.code || 500, { error: e.message }); }
    return true;
  }
  if (p === '/api/inputs' && m === 'GET') { sendJSON(res, 200, { images: listInputImages() }); return true; }
  if (p.startsWith('/api/input/thumb/') && m === 'GET') {
    const name = p.slice('/api/input/thumb/'.length);
    if (!inputPath(name) || !fs.existsSync(inputPath(name))) { sendJSON(res, 404, { error: 'not found' }); return true; }
    try {
      const t = await inputThumb(name);
      res.writeHead(200, { 'Content-Type': 'image/png', 'Content-Length': t.length, 'Cache-Control': 'no-cache' });
      res.end(t);
    } catch (e) { sendJSON(res, 500, { error: e.message }); }
    return true;
  }
  if (p.startsWith('/api/input/') && m === 'GET') {
    const f = inputPath(p.slice('/api/input/'.length));
    if (!f) { sendJSON(res, 404, { error: 'not found' }); return true; }
    await serveFile(req, res, f, { 'Cache-Control': 'no-cache' });
    return true;
  }
  if (p.startsWith('/api/preview/') && m === 'GET') {
    const name = p.slice('/api/preview/'.length);
    if (!/^[\w.-]+\.png$/.test(name)) { sendJSON(res, 404, { error: 'not found' }); return true; }
    await serveFile(req, res, path.join(PREVIEWS, name), { 'Cache-Control': 'no-cache' });
    return true;
  }
  if (p === '/api/workflows' && m === 'GET') { sendJSON(res, 200, { workflows: await listWorkflows() }); return true; }
  if (p === '/api/workflows' && m === 'POST') {
    let b;
    try { b = JSON.parse((await readBody(req, 16 << 20)) || '{}'); } catch (e) { sendJSON(res, 400, { error: 'bad json' }); return true; }
    const f = workflowPath(b.name);
    if (!f) { sendJSON(res, 400, { error: 'bad workflow name (letters, digits, space . ( ) + - _ ; max 80)' }); return true; }
    if (!b.doc || typeof b.doc !== 'object') { sendJSON(res, 400, { error: 'missing doc' }); return true; }
    await fsp.mkdir(WORKFLOWS, { recursive: true });
    await fsp.writeFile(f, JSON.stringify(b.doc, null, 1));
    sendJSON(res, 200, { saved: b.name });
    return true;
  }
  if (p.startsWith('/api/workflows/') && (m === 'GET' || m === 'DELETE')) {
    const name = p.slice('/api/workflows/'.length);
    const f = workflowPath(name);
    if (!f) { sendJSON(res, 400, { error: 'bad workflow name' }); return true; }
    if (m === 'GET') {
      try { sendJSON(res, 200, { name, doc: JSON.parse(await fsp.readFile(f, 'utf8')) }); } catch (_) { sendJSON(res, 404, { error: 'not found' }); }
    } else {
      try { await fsp.unlink(f); sendJSON(res, 200, { deleted: name }); } catch (_) { sendJSON(res, 404, { error: 'not found' }); }
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json', '.txt': 'text/plain; charset=utf-8', '.jpg': 'image/jpeg', '.woff2': 'font/woff2',
};
function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}
function readBody(req, limit = 1 << 20) {
  return new Promise((resolve, reject) => {
    const parts = []; let n = 0;
    req.on('data', (c) => { n += c.length; if (n > limit) { reject(new Error('body too large')); req.destroy(); } else parts.push(c); });
    req.on('end', () => resolve(Buffer.concat(parts).toString('utf8')));
    req.on('error', reject);
  });
}
async function readJSON(req) {
  const s = await readBody(req);
  if (!s.trim()) return {};
  return JSON.parse(s);
}
async function serveFile(req, res, file, headers = {}) {
  let st;
  try { st = await fsp.stat(file); } catch (_) { return sendJSON(res, 404, { error: 'not found' }); }
  if (!st.isFile()) return sendJSON(res, 404, { error: 'not found' });
  const type = MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
  const etag = `"${st.size.toString(16)}-${Math.floor(st.mtimeMs).toString(16)}"`;
  if (req.headers['if-none-match'] === etag) { res.writeHead(304, { ETag: etag }); return res.end(); }
  res.writeHead(200, { 'Content-Type': type, 'Content-Length': st.size, ETag: etag, ...headers });
  if (req.method === 'HEAD') return res.end();
  fs.createReadStream(file).pipe(res);
}

function lanUrls() {
  const urls = [];
  for (const [name, addrs] of Object.entries(os.networkInterfaces())) {
    for (const a of addrs || []) {
      if (a.family === 'IPv4' && !a.internal && !/^169\.254\./.test(a.address)) urls.push({ iface: name, url: `http://${a.address}:${PORT}/` });
    }
  }
  // real adapters (Wi-Fi / Ethernet) first, virtual ones (WSL, Hyper-V, VPN, VM) last
  const virt = (n) => /vEthernet|WSL|Hyper-V|VirtualBox|VMware|Loopback|Tailscale|ZeroTier|Docker/i.test(n);
  urls.sort((a, b) => virt(a.iface) - virt(b.iface));
  return urls;
}

function serverInfo() {
  return {
    name: 'kiln', version: VERSION, mock: engine.mock, engine_exe: ENGINE_EXE, engine: engineStatus(),
    models_dir: MODELS, outputs_dir: OUTPUTS, t5_mode: getT5Mode(), lan: lanUrls().map(u => u.url),
    port: PORT, node: process.version, features: features(), extensions: packs.summary(),
  };
}

function engineInfoRequest(timeout = 2000) {
  return new Promise((resolve) => {
    const id = 'info' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const t = setTimeout(() => { infoWaiters.delete(id); resolve(null); }, timeout);
    infoWaiters.set(id, (ev) => { clearTimeout(t); resolve(ev); });
    if (engine.state === 'down' || !engine.send({ id, cmd: 'info' })) { clearTimeout(t); infoWaiters.delete(id); resolve(null); }
  });
}

function cancelJob(j) {
  if (!j) return false;
  if (j.status === 'queued') {
    const i = queue.indexOf(j);
    if (i >= 0) queue.splice(i, 1);
    if (running === j) running = null;  // it was waiting for the GPU peer
    finishJob(j, 'cancelled');
    return true;
  }
  if (j.status === 'running') {
    if (!j.cancelRequested) { j.cancelRequested = true; engine.send({ id: j.id, cmd: 'cancel' }); }
    if (j.ext) j.ext.cancel();
    // safety net: if the engine never acknowledges, free the queue after 20 s
    setTimeout(() => {
      if (running === j && j.status === 'running') {
        log(`engine did not acknowledge cancel of ${j.id}; restarting engine`);
        running = null;
        finishJob(j, 'cancelled');
        engine.restart();
      }
    }, 20000).unref();
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// routes
// ---------------------------------------------------------------------------
async function handle(req, res) {
  const u = new URL(req.url, 'http://x');
  const p = decodeURIComponent(u.pathname);
  const m = req.method;

  if (p.startsWith('/api/')) {
    if (await handleGraphRoutes(req, res, u, p, m)) return;
    if (await handleExtRoutes(req, res, u, p, m)) return;
    if (await animadex.handle(req, res, u, p, m, sendJSON)) return;
    if (await library.handle(req, res, u, p, m, features)) return;
    if (p === '/api/events' && m === 'GET') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
      res.write('retry: 2000\n\n');
      res.write(`data: ${JSON.stringify({ type: 'hello', server: serverInfo(), queue: queueSnapshot(), recent: [...jobs.values()].filter(j => j.status === 'done').slice(-1).map(publicJob) })}\n\n`);
      clients.add(res);
      req.on('close', () => clients.delete(res));
      return;
    }
    if (p === '/api/generate' && m === 'POST') {
      let body;
      try { body = await readJSON(req); } catch (e) { return sendJSON(res, 400, { error: 'bad json: ' + e.message }); }
      let params;
      try { params = validateRequest(body); } catch (e) { return sendJSON(res, 400, { error: e.message }); }
      let r;
      try { r = createJobs(params); } catch (e) { return sendJSON(res, 500, { error: 'tokenize failed: ' + e.message }); }
      for (const j of r.jobs) emitJob(j);
      emitQueue();
      pump();
      return sendJSON(res, 200, { id: r.jobs[0].id, ids: r.jobs.map(j => j.id), group: r.group, seeds: r.jobs.map(j => j.params.seed), width: params.width, height: params.height, dropped: params.dropped, warnings: params.warnings, loras: params.loras.map(l => ({ file: l.file, strength: l.strength })), expected: params.expected });
    }
    if (p.startsWith('/api/cancel') && m === 'POST') {
      const id = p.slice('/api/cancel'.length).replace(/^\//, '');
      const cancelled = [];
      const targets = id === 'all' || id === '' ? [...queue, running].filter(Boolean)
        : jobs.has(id) ? [jobs.get(id)]
          : [...queue, running].filter(j => j && j.group === id);
      // cancel queued first so the queue doesn't advance into them
      targets.sort((a, b) => (a.status === 'running') - (b.status === 'running'));
      for (const j of targets) if (cancelJob(j)) cancelled.push(j.id);
      if (!targets.length) return sendJSON(res, 404, { error: 'no such job', cancelled });
      return sendJSON(res, 200, { cancelled });
    }
    if (p === '/api/queue' && m === 'GET') return sendJSON(res, 200, queueSnapshot());
    if (p === '/api/train/dataset' && m === 'GET') {
      try {
        const ds = scanDataset(u.searchParams.get('dir'));
        return sendJSON(res, 200, { dir: ds.dir, images: ds.items.length, captioned: ds.items.filter(i => i.caption).length,
          items: ds.items.slice(0, 500).map(i => ({ file: i.file, caption: i.caption })) });
      } catch (e) { return sendJSON(res, 400, { error: e.message }); }
    }
    if (p === '/api/train' && m === 'POST') {
      let body;
      try { body = await readJSON(req); } catch (e) { return sendJSON(res, 400, { error: 'bad json: ' + e.message }); }
      if (engine.mock) return sendJSON(res, 400, { error: 'training needs the real engine' });
      if (!(engineInfo && engineInfo.features && engineInfo.features.train)) return sendJSON(res, 400, { error: 'this engine build cannot train (update the engine)' });
      let params;
      try { params = validateTrain(body); } catch (e) { return sendJSON(res, 400, { error: e.message }); }
      if ([...queue, running].some(q => q && q.kind === 'train')) return sendJSON(res, 400, { error: 'a training run is already queued or running' });
      let j;
      try { j = createTrainJob(params); } catch (e) { return sendJSON(res, 500, { error: 'tokenizing the captions failed: ' + e.message }); }
      emitJob(j);
      emitQueue();
      pump();
      return sendJSON(res, 200, { id: j.id, images: params.ds.items.length, steps: params.steps, epochs: params.epochs, out: j.params.out });
    }
    // ---- datasets (Datasets tab)
    if (p.startsWith('/api/datasets') || p.startsWith('/api/taggers') || p.startsWith('/api/booru')) {
      const origin = req.headers.origin;
      if (m !== 'GET' && m !== 'HEAD' && origin !== undefined) {  // another website must not edit datasets through the browser
        let host = null;
        try { host = new URL(origin).host; } catch (_) { }
        if (host !== req.headers.host) return sendJSON(res, 403, { error: 'cross-origin request refused' });
      }
      const q = (k) => u.searchParams.get(k);
      try {
        if (p === '/api/datasets' && m === 'GET') return sendJSON(res, 200, datasets.list());
        // ---- BooruGrab
        if (p === '/api/booru/settings' && m === 'GET') return sendJSON(res, 200, grab.settings(isLoopback(req)));
        if (p === '/api/booru/status' && m === 'GET') return sendJSON(res, 200, grab.status());
        if (p === '/api/booru/search' && m === 'GET') {
          try { return sendJSON(res, 200, await grab.search(q('site'), q('q'), Number(q('page')) || 1)); }
          catch (e) { return sendJSON(res, 502, { error: e.message }); }
        }
        if (p === '/api/booru/img' && m === 'GET') {
          let r;
          try { r = await grab.image(q('site'), q('url')); } catch (e) { return sendJSON(res, 502, { error: e.message }); }
          res.writeHead(200, { 'Content-Type': r.type, 'Content-Length': r.buffer.length, 'Cache-Control': 'private, max-age=3600' });
          return res.end(r.buffer);
        }
        if (p === '/api/datasets/items' && m === 'GET') return sendJSON(res, 200, datasets.items(q('dir')));
        if (p === '/api/datasets/image' && (m === 'GET' || m === 'HEAD')) {
          const f = datasets.fileIn(q('dir'), q('f'));
          if (!datasets.IMG_RE.test(f)) return sendJSON(res, 404, { error: 'not found' });
          return serveFile(req, res, f, { 'Cache-Control': 'no-cache' });
        }
        if (p === '/api/datasets/thumb' && (m === 'GET' || m === 'HEAD')) {
          const t = datasets.thumbPath(q('dir'), q('f'));
          if (!fs.existsSync(t)) return sendJSON(res, 404, { error: 'no thumbnail yet' });
          return serveFile(req, res, t, { 'Cache-Control': 'no-cache' });
        }
        if (p === '/api/datasets/thumb' && m === 'POST') {
          datasets.putThumb(q('dir'), q('f'), await readRaw(req, 400 * 1024));
          return sendJSON(res, 200, { ok: true });
        }
        if (m === 'POST') {
          const b = await readJSON(req);
          if (p === '/api/datasets/create') return sendJSON(res, 200, datasets.create(b.name));
          if (p === '/api/booru/settings') { grab.update(b); return sendJSON(res, 200, grab.settings(isLoopback(req))); }
          if (p === '/api/booru/import') {
            const why = manageDenied(req);  // reads a file on this PC
            if (why) return sendJSON(res, 403, { error: why });
            const r = grab.importBooruGrab();
            log(`imported BooruGrab settings: ${r.sites} sites`);
            return sendJSON(res, 200, { ...r, settings: grab.settings(isLoopback(req)) });
          }
          if (p === '/api/booru/resolve') {
            try { return sendJSON(res, 200, await grab.resolve(b.site, b.post || {})); } catch (e) { return sendJSON(res, 502, { error: e.message }); }
          }
          if (p === '/api/booru/download') {
            const n = grab.enqueue(b.dir, Array.isArray(b.posts) ? b.posts.slice(0, 2000) : []);
            return sendJSON(res, 200, { queued: n, status: grab.status() });
          }
          if (p === '/api/booru/cancel') { grab.cancel(); return sendJSON(res, 200, grab.status()); }
          if (p === '/api/datasets/open') {
            const why = manageDenied(req);  // a new folder on this PC: only from the PC itself
            if (why) return sendJSON(res, 403, { error: why });
            return sendJSON(res, 200, datasets.open(b.dir));
          }
          if (p === '/api/datasets/forget') { datasets.forget(b.dir); return sendJSON(res, 200, { ok: true }); }
          if (p === '/api/datasets/caption') return sendJSON(res, 200, { caption: datasets.writeCaption(b.dir, b.file, b.caption) });
          if (p === '/api/datasets/bulk') return sendJSON(res, 200, datasets.bulk(b.dir, b.files, b.op, b.tags, b.to));
          if (p === '/api/datasets/remove') return sendJSON(res, 200, datasets.remove(b.dir, b.files));
          if (p === '/api/taggers/download') { await taggers.download(String(b.id || '')); return sendJSON(res, 200, { ok: true }); }
          if (p === '/api/datasets/autotag') {
            if (!(engineInfo && engineInfo.features && engineInfo.features.tag)) return sendJSON(res, 400, { error: 'this engine build cannot tag images (update the engine)' });
            const dir = datasets.allowed(b.dir);
            const model = String(b.model || '');
            if (!taggers.installed(model)) return sendJSON(res, 400, { error: 'download the tagger first' });
            const all = datasets.items(dir).items.map(i => i.file);
            const files = Array.isArray(b.files) && b.files.length ? all.filter(f => b.files.includes(f)) : all;
            if (!files.length) return sendJSON(res, 400, { error: 'no images to tag' });
            const j = createTagJob({
              dir, files, model, mode: ['append', 'replace', 'missing'].includes(b.mode) ? b.mode : 'append',
              general: num(b.general, 0.35, 0.05, 0.99), character: num(b.character, 0.85, 0.05, 0.99),
              exclude: datasets.split(b.exclude), underscores: !!b.underscores, rating: !!b.rating, keep_first: Math.round(num(b.keep_first, 0, 0, 20)),
            });
            emitJob(j);
            emitQueue();
            pump();
            return sendJSON(res, 200, { id: j.id, images: files.length });
          }
        }
        if (p === '/api/taggers' && m === 'GET') return sendJSON(res, 200, { taggers: taggers.list() });
        return sendJSON(res, 404, { error: 'not found' });
      } catch (e) { return sendJSON(res, 400, { error: e.message }); }
    }
    if (p === '/api/train/file' && (m === 'GET' || m === 'HEAD')) {
      const base = path.join(MODELS, 'loras');
      const f = path.resolve(base, String(u.searchParams.get('f') || ''));
      if (!f.startsWith(base + path.sep) || !/\.png$/i.test(f)) return sendJSON(res, 404, { error: 'not found' });
      return serveFile(req, res, f, { 'Cache-Control': 'no-cache' });
    }
    if (p === '/api/engine' && m === 'GET') return sendJSON(res, 200, engineStatus());
    if (p === '/api/engine/park' && m === 'POST') {
      if (!isLoopback(req)) return sendJSON(res, 403, { error: 'only apps on this PC can park the engine' });
      let b = {};
      try { b = await readJSON(req); } catch (_) { }
      if (running || queue.length) return sendJSON(res, 409, { error: 'busy', running: !!running, queued: queue.length });
      const url = typeof b.release_url === 'string' && /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?\//.test(b.release_url) ? b.release_url : null;
      parked = { holder: String(b.holder || 'another app').slice(0, 40), release_url: url, since: Date.now() };
      if (engine.proc) await new Promise((r) => { const t = setTimeout(r, 10000); engine.once('exit', () => { clearTimeout(t); r(); }); engine.stop(); });
      else engine.stop();  // also cancels a pending restart
      log(`engine parked: VRAM freed for ${parked.holder}`);
      broadcast({ type: 'engine', engine: engineStatus() });
      return sendJSON(res, 200, engineStatus());
    }
    if (p === '/api/gpu/peer' && m === 'POST') {
      if (!isLoopback(req)) return sendJSON(res, 403, { error: 'only apps on this PC can share the GPU' });
      let b = {};
      try { b = await readJSON(req); } catch (_) { }
      const acq = localUrl(b.acquire_url), rel = localUrl(b.release_url);
      if (!acq || !rel) return sendJSON(res, 400, { error: 'acquire_url and release_url must be http://127.0.0.1 URLs' });
      const name = String(b.name || 'another app').slice(0, 40);
      if (!gpuPeer || gpuPeer.acquire_url !== acq) log(`sharing the GPU with ${name}: renders wait for its turn`);
      gpuPeer = { name, acquire_url: acq, release_url: rel };
      if (parked) unpark(`${name} shares the GPU now`);
      broadcast({ type: 'engine', engine: engineStatus() });
      return sendJSON(res, 200, engineStatus());
    }
    if (p === '/api/gpu/peer' && m === 'DELETE') {
      if (!isLoopback(req)) return sendJSON(res, 403, { error: 'only apps on this PC can do that' });
      if (gpuPeer) log(`stopped sharing the GPU with ${gpuPeer.name}`);
      gpuPeer = null;
      broadcast({ type: 'engine', engine: engineStatus() });
      return sendJSON(res, 200, engineStatus());
    }
    if (p === '/api/engine/unpark' && m === 'POST') {
      if (!isLoopback(req)) return sendJSON(res, 403, { error: 'only apps on this PC can unpark the engine' });
      unpark('requested');
      return sendJSON(res, 200, engineStatus());
    }
    if (p.startsWith('/api/job/') && m === 'GET') {
      const j = jobs.get(p.slice(9));
      return j ? sendJSON(res, 200, publicJob(j)) : sendJSON(res, 404, { error: 'no such job' });
    }
    if (p === '/api/gallery' && m === 'GET') {
      const list = await refreshGallery();
      const offset = Math.max(0, parseInt(u.searchParams.get('offset') || '0', 10) || 0);
      const limit = Math.min(200, Math.max(1, parseInt(u.searchParams.get('limit') || '40', 10) || 40));
      const slice = list.slice(offset, offset + limit);
      const items = await Promise.all(slice.map(async (it) => {
        const meta = await readMeta(it.file, it.mtime);
        return { file: it.file, url: '/outputs/' + it.file, thumb: '/api/thumb/' + it.file, view: viewUrl(it.file, meta.w, meta.h), mtime: it.mtime, size: it.size, w: meta.w, h: meta.h, params: meta.params };
      }));
      return sendJSON(res, 200, { total: list.length, offset, limit, items });
    }
    if (p.startsWith('/api/gallery/') && m === 'DELETE') {
      const file = p.slice('/api/gallery/'.length);
      const abs = resolveOutput(file);
      if (!abs) return sendJSON(res, 400, { error: 'bad file name' });
      try { await fsp.unlink(abs); } catch (e) { return sendJSON(res, 404, { error: 'not found' }); }
      metaCache.delete(file); thumbCache.delete(file); viewCache.delete(file); galleryDirty = true;
      log('deleted ' + file);
      broadcast({ type: 'deleted', file });
      return sendJSON(res, 200, { deleted: file });
    }
    if (p.startsWith('/api/thumb/') && (m === 'GET' || m === 'HEAD')) {
      const file = p.slice('/api/thumb/'.length);
      const max = Number(u.searchParams.get('max')) > THUMB_MAX ? VIEW_MAX : THUMB_MAX;
      let t;
      try { t = await getThumb(file, max); } catch (_) { t = null; }
      if (!t) return sendJSON(res, 404, { error: 'not found' });
      if (t === 'original') { res.writeHead(302, { Location: '/outputs/' + file.split('/').map(encodeURIComponent).join('/'), 'Cache-Control': 'no-cache' }); return res.end(); }
      res.writeHead(200, { 'Content-Type': 'image/png', 'Content-Length': t.length, 'Cache-Control': 'public, max-age=31536000, immutable' });
      return res.end(m === 'HEAD' ? undefined : t);
    }
    if (p === '/api/info' && m === 'GET') {
      const ev = await engineInfoRequest(running ? 800 : 2500);
      return sendJSON(res, 200, { server: serverInfo(), engine: ev || engineInfo, engine_stale: !ev, features: features() });
    }
    if (p === '/api/log' && m === 'GET') return sendJSON(res, 200, { lines: logRing });
    if (p === '/api/tokenize' && m === 'POST') {
      try {
        const b = await readJSON(req);
        const raw = String(b.prompt || b.text || '');
        const text = b.raw ? raw : A.toKilnPrompt(raw).text;   // what generate tokenizes (<lora:> tags removed, [x] -> (x:0.9091))
        if (b.family === 'sdxl') {
          // CLIP: 75 tokens per chunk (BOS + 75 + EOS), like A1111's "n/75" counter
          const e = encodeSDXL(text);
          const tokens = e.l.ids.reduce((n, ch) => { const k = ch.indexOf(49407, 1); return n + (k > 0 ? k - 1 : 75); }, 0);
          return sendJSON(res, 200, { sdxl: { tokens, chunks: e.l.ids.length } });
        }
        return sendJSON(res, 200, encodePrompt(text));
      }
      catch (e) { return sendJSON(res, 400, { error: e.message }); }
    }
    return sendJSON(res, 404, { error: 'no such endpoint' });
  }

  if (m !== 'GET' && m !== 'HEAD') return sendJSON(res, 405, { error: 'method not allowed' });

  if (p.startsWith('/outputs/')) {
    const file = p.slice('/outputs/'.length);
    const abs = resolveOutput(file);
    if (!abs) return sendJSON(res, 404, { error: 'not found' });
    const headers = { 'Cache-Control': 'public, max-age=31536000, immutable' };
    if (u.searchParams.has('dl')) headers['Content-Disposition'] = `attachment; filename="${path.basename(abs)}"`;
    return serveFile(req, res, abs, headers);
  }

  // pack UI modules / assets: /extensions/<pack folder>/<file>
  if (p.startsWith('/extensions/')) {
    const mm = p.match(/^\/extensions\/([^/]+)\/(.+)$/);
    const f = mm && packs.staticFile(mm[1], mm[2]);
    if (!f) return sendJSON(res, 404, { error: 'not found' });
    return serveFile(req, res, f, { 'Cache-Control': 'no-cache' });
  }

  // static web/
  let rel = p === '/' ? '/index.html' : p;
  const abs = path.resolve(WEB, '.' + rel);
  if (!abs.startsWith(WEB + path.sep)) return sendJSON(res, 403, { error: 'forbidden' });
  return serveFile(req, res, abs, { 'Cache-Control': 'no-cache' });
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
function main() {
  fs.mkdirSync(OUTPUTS, { recursive: true });
  // leftover raw buffers from a previous run (only our own tmp/*.rgb files)
  try { for (const f of fs.readdirSync(TMP)) if (/\.(rgb|f32)$/.test(f)) fs.unlinkSync(path.join(TMP, f)); } catch (_) { }
  const t0 = Date.now();
  warmup();
  log(`tokenizers loaded in ${Date.now() - t0} ms (t5 mode: ${getT5Mode()})`);
  packs.load();

  const server = http.createServer((req, res) => {
    handle(req, res).catch((e) => {
      log('request error: ' + (e && e.stack || e));
      if (!res.headersSent) sendJSON(res, 500, { error: String(e && e.message || e) });
      else try { res.end(); } catch (_) { }
    });
  });
  server.keepAliveTimeout = 65000;
  server.on('error', (e) => {
    console.error(`Kiln server failed to listen on ${HOST}:${PORT}: ${e.message}`);
    process.exit(1);
  });
  server.listen(PORT, HOST, () => {
    const haveExe = fs.existsSync(ENGINE_EXE);
    console.log('');
    console.log('  KILN ' + VERSION);
    console.log(`  Local:   http://localhost:${PORT}/`);
    for (const u of lanUrls()) console.log(`  LAN:     ${u.url}   (${u.iface})`);
    console.log(`  Engine:  ${haveExe ? ENGINE_EXE : 'MOCK (not found: ' + ENGINE_EXE + ')'}`);
    console.log(`  Outputs: ${OUTPUTS}`);
    const ps = packs.summary();
    console.log(`  Packs:   ${ps.packs} in ${EXTENSIONS} (${ps.nodes} nodes${ps.errors ? ', ' + ps.errors + ' with errors' : ''})`);
    console.log('');
    engine.args = engineArgs();
    engine.start();
    civitai.pump();
  });

  const shutdown = () => {
    log('shutting down');
    engine.stop();
    for (const res of clients) { try { res.end(); } catch (_) { } }
    server.close();
    setTimeout(() => process.exit(0), 300).unref();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
