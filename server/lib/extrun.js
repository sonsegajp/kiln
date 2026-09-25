'use strict';
// Pack-node runtime (docs/NODE_API.md): wire values <-> JS values (raw float32 tensor files), the
// ctx handed to run(), the per-job ext_call / ext_op session with the engine, and constant folding
// (pack nodes whose inputs are all constants run once on the server when the job is queued).

const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const { isLink, SAMPLERS, SCHEDULERS } = require('./graph');

const TENSOR_DIMS = { IMAGE: 4, MASK: 3, LATENT: 4 };
const PRIM_TYPES = new Set(['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO']);
const RESIZE_METHODS = ['nearest-exact', 'bilinear', 'bicubic', 'area', 'lanczos'];
const MAX_VALUES = 256 * 1024 * 1024;   // 1 GiB of float32 per tensor
const PROGRESS_MS = 50;                 // ctx.progress -> SSE at most every 50 ms per node
const FOLD_TIMEOUT_MS = 30000;          // a constant node must finish within 30 s at queue time

class Cancelled extends Error { constructor() { super('cancelled'); this.name = 'Cancelled'; } }

const isPrim = (v) => typeof v === 'string' || typeof v === 'boolean' || (typeof v === 'number' && Number.isFinite(v));
const isTensor = (v) => !!(v && typeof v === 'object' && TENSOR_DIMS[v.type] && v.data != null);
const isHandle = (v) => !!(v && typeof v === 'object' && typeof v.type === 'string' && v.handle != null && v.data == null);
const show = (v) => (v === undefined ? 'undefined' : typeof v === 'string' ? JSON.stringify(v.slice(0, 40)) : typeof v === 'object' && v ? (v.type || (Array.isArray(v) ? 'an array' : 'an object')) : String(v));

// ---------------------------------------------------------------------------
// tensors
// ---------------------------------------------------------------------------
function shapeOf(v) {
  if (v.type === 'IMAGE') return [v.batch, v.height, v.width, 3];
  if (v.type === 'MASK') return [v.batch, v.height, v.width];
  return [v.batch, v.channels, v.height, v.width];
}
function fromShape(type, shape, data) {
  if (type === 'IMAGE') return { type, batch: shape[0], height: shape[1], width: shape[2], data };
  if (type === 'MASK') return { type, batch: shape[0], height: shape[1], width: shape[2], data };
  return { type, batch: shape[0], channels: shape[1], height: shape[2], width: shape[3], data };
}
function checkShape(type, shape, what) {
  if (!Array.isArray(shape) || shape.length !== TENSOR_DIMS[type] || !shape.every(d => Number.isInteger(d) && d > 0)) throw new Error(`${what}: bad ${type} size ${JSON.stringify(shape)}`);
  if (type === 'IMAGE' && shape[3] !== 3) throw new Error(`${what}: IMAGE must have 3 channels, got ${shape[3]}`);
  const n = shape.reduce((a, b) => a * b, 1);
  if (n > MAX_VALUES) throw new Error(`${what}: ${type} ${shape.join('x')} is too large`);
  return n;
}
function newTensor(type, shape) {
  const full = type === 'IMAGE' ? [...shape, 3] : shape;
  const n = checkShape(type, full, `ctx.${type.toLowerCase()}()`);
  return fromShape(type, full, new Float32Array(n));
}
async function readTensor(w, dir, what) {
  const n = checkShape(w.type, w.shape, what);
  const abs = path.resolve(String(w.path));
  if (!abs.startsWith(dir + path.sep)) throw new Error(`${what}: tensor file outside the job folder: ${w.path}`);
  const buf = await fsp.readFile(abs);
  if (buf.length !== n * 4) throw new Error(`${what}: ${path.basename(abs)} has ${buf.length} bytes, ${w.type} ${w.shape.join('x')} needs ${n * 4}`);
  const data = buf.byteOffset % 4 === 0 ? new Float32Array(buf.buffer, buf.byteOffset, n) : new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + n * 4));
  return fromShape(w.type, w.shape.slice(), data);
}
async function writeTensor(type, v, file, what) {
  if (!isTensor(v) || v.type !== type) throw new Error(`${what}: expected ${type}, got ${show(v)}`);
  const shape = shapeOf(v);
  const n = checkShape(type, shape, what);
  let data = v.data;
  if (!(data instanceof Float32Array)) {
    if (Array.isArray(data) || ArrayBuffer.isView(data)) data = Float32Array.from(data);
    else throw new Error(`${what}: ${type}.data must be a Float32Array`);
  }
  if (data.length !== n) throw new Error(`${what}: data has ${data.length} values, ${type} ${shape.join('x')} needs ${n}`);
  await fsp.writeFile(file, Buffer.from(data.buffer, data.byteOffset, n * 4));
  return { type, path: file, shape };
}

// primitives, coerced to the declared type
function coercePrim(type, v, what) {
  if (type === 'INT' || type === 'FLOAT') {
    const n = typeof v === 'string' && v.trim() !== '' ? Number(v) : typeof v === 'boolean' ? Number(v) : v;
    if (typeof n !== 'number' || !Number.isFinite(n)) throw new Error(`${what}: expected ${type}, got ${show(v)}`);
    return type === 'INT' ? Math.round(n) : n;
  }
  if (type === 'BOOLEAN') {
    if (typeof v === 'boolean') return v;
    if (v === 'true' || v === 1) return true;
    if (v === 'false' || v === 0) return false;
    throw new Error(`${what}: expected BOOLEAN, got ${show(v)}`);
  }
  if (v == null || typeof v === 'object') throw new Error(`${what}: expected ${type}, got ${show(v)}`);
  return String(v);
}

// what run() returned -> { outputs: [...], ui }
function normalizeResult(res, outputs) {
  let ui = null;
  if (res && typeof res === 'object' && !Array.isArray(res) && !isTensor(res) && !isHandle(res) && ('outputs' in res || 'ui' in res)) {
    ui = res.ui || null;
    res = res.outputs === undefined ? [] : res.outputs;
  }
  if (res === undefined && outputs.length === 0) res = [];
  if (!Array.isArray(res)) {
    if (outputs.length === 1) res = [res];
    else throw new Error(`run() must return an array of ${outputs.length} value(s)`);
  }
  if (res.length !== outputs.length) throw new Error(`run() returned ${res.length} value(s); the node declares ${outputs.length} output(s)`);
  let u = null;
  if (ui && typeof ui === 'object' && ui.text != null) {
    u = { text: (Array.isArray(ui.text) ? ui.text : [ui.text]).map(t => (typeof t === 'string' ? t : JSON.stringify(t))).join('\n').slice(0, 4000) };
  }
  return { outputs: res, ui: u };
}

// ---------------------------------------------------------------------------
// ctx
// ---------------------------------------------------------------------------
const OP_NAMES = ['vaeDecode', 'vaeEncode', 'resize', 'upscale', 'detectFaces', 'encodeText', 'sample'];
function makeCtx(o) {
  return Object.freeze({
    node: o.node,
    class_type: o.classType,
    pack: o.pack,
    image: (b, h, w) => newTensor('IMAGE', [b, h, w]),
    mask: (b, h, w) => newTensor('MASK', [b, h, w]),
    latent: (b, c, h, w) => newTensor('LATENT', [b, c, h, w]),
    progress: (done, total) => { if (Number.isFinite(done) && Number.isFinite(total) && total > 0) o.progress(Math.max(0, Math.min(done, total)), total); },
    log: (msg) => o.log(typeof msg === 'string' ? msg : JSON.stringify(msg)),
    cancelled: () => !!o.cancelled(),
    ops: o.ops,
  });
}
const FOLD_OPS = Object.freeze(Object.fromEntries(OP_NAMES.map(k => [k, async () => {
  throw new Error(`ctx.ops.${k} is not available while Kiln runs a constant node at queue time (set fold: false on the node)`);
}])));

// ---------------------------------------------------------------------------
// constant folding
// ---------------------------------------------------------------------------
// A pack node is folded when every input is a literal or comes from another folded node, none of its
// outputs is an engine type (IMAGE, MODEL, ...) and it doesn't say fold: false. Folded nodes run here,
// at queue time; primitive results replace the links into the rest of the graph (so e.g. a text node
// can feed CLIPTextEncode, which is tokenized on the server). A folded node whose non-primitive output
// still feeds a node that runs in the engine stays in the engine graph and its ext_call is answered
// from `cache`.
async function fold(graph, o) {
  const { defs, engineTypes } = o;
  const memo = {};
  const can = (id, stack) => {
    if (id in memo) return memo[id];
    if (stack.has(id)) return false;
    stack.add(id);
    const n = graph[id], d = n && defs[n.class_type];
    let ok = !!d && d.def.fold !== false && !d.entry.output.some(t => engineTypes.has(t));
    if (ok) for (const v of Object.values(n.inputs)) if (isLink(v) && !can(String(v[0]), stack)) { ok = false; break; }
    stack.delete(id);
    memo[id] = ok;
    return ok;
  };
  const order = [], seen = new Set();
  const visit = (id) => {
    if (seen.has(id)) return;
    seen.add(id);
    for (const v of Object.values(graph[id].inputs)) if (isLink(v)) visit(String(v[0]));
    order.push(id);
  };
  for (const id of Object.keys(graph)) if (can(id, new Set())) visit(id);

  const values = {}, errors = {}, ms = {}, ui = {};
  for (const id of order) {
    const n = graph[id], d = defs[n.class_type];
    const inputs = {};
    let upstreamFailed = false;
    for (const [k, v] of Object.entries(n.inputs)) {
      if (isLink(v)) {
        const src = values[String(v[0])];
        if (!src) { upstreamFailed = true; break; }
        inputs[k] = src[v[1]];
      } else inputs[k] = v;
    }
    if (upstreamFailed) continue;
    const t0 = Date.now();
    try {
      const ctx = makeCtx({
        node: id, classType: n.class_type, pack: d.pack.name,
        progress: () => { }, log: (m) => o.log(id, d.pack.name, m), cancelled: () => false, ops: FOLD_OPS,
      });
      let timer = null;
      const ran = Promise.resolve().then(() => d.def.run(inputs, ctx));
      const limit = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`run() took longer than ${FOLD_TIMEOUT_MS / 1000} s at queue time (give the node fold: false if it is slow)`)), FOLD_TIMEOUT_MS); });
      let out;
      try { out = await Promise.race([ran, limit]); } finally { clearTimeout(timer); }
      const res = normalizeResult(out, d.entry.output);
      values[id] = res.outputs.map((v, i) => {
        const t = d.entry.output[i], what = `output ${i} (${d.entry.output_name[i]})`;
        if (PRIM_TYPES.has(t)) return coercePrim(t, v, what);
        if (v === undefined) throw new Error(`${what} is undefined`);
        return v;
      });
      if (res.ui) ui[id] = res.ui;
    } catch (e) {
      errors[id] = String((e && e.message) || e);
      if (o.onError) o.onError(id, d.pack.name, e);
    }
    ms[id] = Date.now() - t0;
  }
  if (Object.keys(errors).length) return { ok: false, errors };

  // substitute primitive results; keep folded nodes whose other values still feed the engine graph
  const eg = {};
  for (const [id, n] of Object.entries(graph)) eg[id] = { class_type: n.class_type, inputs: Object.assign({}, n.inputs) };
  const removed = new Set(order);
  for (let changed = true; changed;) {
    changed = false;
    for (const [id, n] of Object.entries(eg)) {
      if (removed.has(id)) continue;
      for (const [k, v] of Object.entries(n.inputs)) {
        if (!isLink(v) || !values[String(v[0])]) continue;
        const val = values[String(v[0])][v[1]];
        if (isPrim(val)) n.inputs[k] = val;
        else if (removed.has(String(v[0]))) { removed.delete(String(v[0])); changed = true; }
      }
    }
  }
  for (const id of removed) delete eg[id];
  const cache = {};
  for (const id of order) if (!removed.has(id)) cache[id] = values[id];
  return { ok: true, engineGraph: eg, folded: order, removed: [...removed], cache, ms, ui };
}

// ---------------------------------------------------------------------------
// ext session: one per running graph job that has pack nodes in its engine graph
// ---------------------------------------------------------------------------
class ExtSession {
  constructor(o) {
    this.jobId = o.jobId;
    this.defs = o.defs;
    this.dir = path.resolve(o.dir);
    this.send = o.send;
    this.engineTypes = o.engineTypes;
    this.tokenize = o.tokenize;
    this.onProgress = o.onProgress || (() => { });
    this.onLog = o.onLog || (() => { });
    this.onUi = o.onUi || (() => { });
    this.onError = o.onError || (() => { });
    this.cache = o.cache || {};
    this.store = new Map();       // 'js:<n>' -> values of pack-only types (never leave the server)
    this.calls = new Map();       // call -> record
    this.ops = new Map();         // op_id -> pending op
    this.seq = { op: 0, file: 0, js: 0 };
    this.lastProgress = {};
    this.cancelled = false;
    this.closed = false;
  }
  file(tag) { return path.join(this.dir, `${this.jobId}_${tag}_${++this.seq.file}.f32`); }

  async fromWire(v, what) {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) return v;
    if (TENSOR_DIMS[v.type] && typeof v.path === 'string') return readTensor(v, this.dir, what);
    if (typeof v.type === 'string' && v.handle != null) {
      const h = String(v.handle);
      if (h.startsWith('js:')) {
        if (!this.store.has(h)) throw new Error(`${what}: unknown value ${h}`);
        return this.store.get(h);
      }
      return Object.freeze({ type: v.type, handle: v.handle });
    }
    return v;
  }
  async toWire(type, v, what, tag) {
    if (type === '*') {
      if (isPrim(v)) return v;
      if (v === undefined || v === null) throw new Error(`${what} is ${v}`);
      if (isTensor(v)) type = v.type;
      else if (isHandle(v)) return { type: v.type, handle: v.handle };
      else type = typeof v.type === 'string' && /^[A-Z][A-Z0-9_]*$/.test(v.type) ? v.type : 'JS';
    }
    if (PRIM_TYPES.has(type)) return coercePrim(type, v, what);
    if (TENSOR_DIMS[type]) return writeTensor(type, v, this.file(tag), what);
    if (isHandle(v)) {
      if (v.type !== type) throw new Error(`${what}: expected ${type}, got a ${v.type}`);
      return { type: v.type, handle: v.handle };
    }
    if (this.engineTypes.has(type)) throw new Error(`${what}: ${type} values come from the engine; return the one you were given (or one from ctx.ops)`);
    if (v === undefined) throw new Error(`${what} is undefined`);
    const h = 'js:' + (++this.seq.js);
    this.store.set(h, v);
    return { type, handle: h };
  }

  async handleCall(ev) {
    const rec = { call: ev.call, node: String(ev.node), cls: ev.class_type, answered: false, ops: new Set() };
    this.calls.set(ev.call, rec);
    const d = this.defs[ev.class_type];
    try {
      if (this.cancelled || this.closed) throw new Cancelled();
      if (!d) throw new Error(`no loaded pack provides "${ev.class_type}"`);
      let outputs;
      if (this.cache[rec.node]) outputs = this.cache[rec.node];
      else {
        const inputs = {};
        for (const [k, v] of Object.entries(ev.inputs || {})) inputs[k] = await this.fromWire(v, `input "${k}"`);
        const ctx = makeCtx({
          node: rec.node, classType: ev.class_type, pack: d.pack.name,
          progress: (a, b) => this.progress(rec.node, a, b),
          log: (m) => this.onLog(rec.node, d.pack.name, m),
          cancelled: () => rec.answered || this.cancelled || this.closed,
          ops: this.opsFor(rec),
        });
        const res = normalizeResult(await d.def.run(inputs, ctx), d.entry.output);
        outputs = res.outputs;
        if (res.ui && !rec.answered) this.onUi(rec.node, res.ui);
      }
      if (rec.answered) return;
      const wire = [];
      for (let i = 0; i < d.entry.output.length; i++) wire.push(await this.toWire(d.entry.output[i], outputs[i], `output ${i} (${d.entry.output_name[i]})`, `x${ev.call}o${i}`));
      this.reply(rec, { outputs: wire });
    } catch (e) {
      const cancelled = e instanceof Cancelled || rec.answered || this.cancelled;
      if (!cancelled) this.onError(rec.node, d ? d.pack.name : '?', e);
      this.reply(rec, { error: cancelled ? 'cancelled' : String((e && e.message) || e) });
    }
  }
  reply(rec, body) {
    if (rec.answered) return;
    rec.answered = true;
    this.calls.delete(rec.call);
    for (const id of rec.ops) { const o = this.ops.get(id); if (o) { this.ops.delete(id); o.reject(new Cancelled()); } }
    rec.ops.clear();
    this.send(Object.assign({ cmd: 'ext_result', id: this.jobId, call: rec.call }, body));
  }
  progress(node, done, total) {
    const now = Date.now();
    if (done < total && now - (this.lastProgress[node] || 0) < PROGRESS_MS) return;
    this.lastProgress[node] = now;
    this.onProgress(node, done, total);
  }

  op(rec, name, args) {
    return new Promise((resolve, reject) => {
      if (rec.answered || this.cancelled || this.closed) { reject(new Cancelled()); return; }
      const op_id = ++this.seq.op;
      this.ops.set(op_id, { resolve, reject, rec, name });
      rec.ops.add(op_id);
      if (!this.send(Object.assign({ cmd: 'ext_op', id: this.jobId, call: rec.call, op_id, op: name }, args))) {
        this.ops.delete(op_id);
        rec.ops.delete(op_id);
        reject(new Error('the engine is not running'));
      }
    });
  }
  handleOpDone(ev) {
    const o = this.ops.get(ev.op_id);
    if (!o) return;
    this.ops.delete(ev.op_id);
    o.rec.ops.delete(ev.op_id);
    if (ev.error != null) o.reject(new Error(`${o.name}: ${ev.error}`));
    else o.resolve(ev.result);
  }
  opsFor(rec) {
    const S = this;
    const tensor = (type, v, what) => { if (!isTensor(v) || v.type !== type) throw new Error(`${what} must be an ${type} value, got ${show(v)}`); return v; };
    const handle = (type, v, what) => { if (!isHandle(v) || v.type !== type) throw new Error(`${what} must be a ${type} (the value from a ${type} input), got ${show(v)}`); return { type: v.type, handle: v.handle }; };
    const tw = (type, v, what) => S.toWire(type, tensor(type, v, what), what, `o${S.seq.op + 1}`);
    const result = async (p, type, what) => {
      const r = await p;
      const v = await S.fromWire(r, what);
      if (type && (TENSOR_DIMS[type] ? !isTensor(v) || v.type !== type : !isHandle(v) || v.type !== type)) throw new Error(`${what}: the engine returned ${show(v)}, expected ${type}`);
      return v;
    };
    const num = (v, what, def) => { const n = v === undefined ? def : Number(v); if (!Number.isFinite(n)) throw new Error(`${what} must be a number`); return n; };
    return Object.freeze({
      async vaeDecode(latent, vae) {
        const a = { latent: await tw('LATENT', latent, 'ops.vaeDecode(latent)'), vae: handle('VAE', vae, 'ops.vaeDecode(vae)') };
        return result(S.op(rec, 'vae_decode', a), 'IMAGE', 'ops.vaeDecode');
      },
      async vaeEncode(image, vae) {
        const a = { image: await tw('IMAGE', image, 'ops.vaeEncode(image)'), vae: handle('VAE', vae, 'ops.vaeEncode(vae)') };
        return result(S.op(rec, 'vae_encode', a), 'LATENT', 'ops.vaeEncode');
      },
      async resize(image, width, height, method = 'bilinear') {
        const w = Math.round(num(width, 'ops.resize(width)')), h = Math.round(num(height, 'ops.resize(height)'));
        if (w < 1 || h < 1 || w > 16384 || h > 16384) throw new Error(`ops.resize: bad size ${w}x${h}`);
        if (!RESIZE_METHODS.includes(method)) throw new Error(`ops.resize: method must be one of ${RESIZE_METHODS.join(', ')}`);
        const a = { image: await tw('IMAGE', image, 'ops.resize(image)'), width: w, height: h, method };
        return result(S.op(rec, 'resize', a), 'IMAGE', 'ops.resize');
      },
      async upscale(image, model) {
        const a = { image: await tw('IMAGE', image, 'ops.upscale(image)'), model: handle('UPSCALE_MODEL', model, 'ops.upscale(model)') };
        return result(S.op(rec, 'upscale', a), 'IMAGE', 'ops.upscale');
      },
      async detectFaces(image, threshold = 0.5, detector) {
        const a = { image: await tw('IMAGE', image, 'ops.detectFaces(image)'), threshold: num(threshold, 'ops.detectFaces(threshold)', 0.5) };
        if (detector !== undefined && detector !== null) a.detector = handle('BBOX_DETECTOR', detector, 'ops.detectFaces(detector)');
        const r = await S.op(rec, 'detect_faces', a);
        if (!Array.isArray(r)) throw new Error('ops.detectFaces: the engine returned ' + show(r));
        return r.map(list => (Array.isArray(list) ? list : []).map(b => ({ x0: +b.x0, y0: +b.y0, x1: +b.x1, y1: +b.y1, score: +b.score })));
      },
      async encodeText(clip, text) {
        if (typeof text !== 'string') throw new Error('ops.encodeText(text) must be a string');
        const c = handle('CLIP', clip, 'ops.encodeText(clip)');
        const tok = S.tokenize(text);
        const a = { clip: c, text, qwen_ids: tok.qwen_ids, t5_ids: tok.t5_ids, t5_weights: tok.t5_weights };
        return result(S.op(rec, 'encode_text', a), 'CONDITIONING', 'ops.encodeText');
      },
      async sample(model, positive, negative, latent, opts = {}) {
        const o = opts || {};
        const sampler = o.sampler === undefined ? 'euler' : String(o.sampler);
        const scheduler = o.scheduler === undefined ? 'simple' : String(o.scheduler);
        if (!SAMPLERS.includes(sampler)) throw new Error(`ops.sample: sampler must be one of ${SAMPLERS.join(', ')}`);
        if (!SCHEDULERS.includes(scheduler)) throw new Error(`ops.sample: scheduler must be one of ${SCHEDULERS.join(', ')}`);
        const a = {
          model: handle('MODEL', model, 'ops.sample(model)'),
          positive: handle('CONDITIONING', positive, 'ops.sample(positive)'),
          negative: handle('CONDITIONING', negative, 'ops.sample(negative)'),
          latent: await tw('LATENT', latent, 'ops.sample(latent)'),
          seed: Math.round(num(o.seed, 'ops.sample: seed', 0)),
          steps: Math.max(1, Math.round(num(o.steps, 'ops.sample: steps', 20))),
          cfg: num(o.cfg, 'ops.sample: cfg', 4.5),
          sampler, scheduler,
          denoise: Math.min(1, Math.max(0, num(o.denoise, 'ops.sample: denoise', 1))),
        };
        if (o.mask !== undefined && o.mask !== null) a.mask = await tw('MASK', o.mask, 'ops.sample: mask');
        return result(S.op(rec, 'sample', a), 'LATENT', 'ops.sample');
      },
    });
  }

  // job cancelled: answer every open call (the engine unwinds), reject pending ops
  cancel(reason = 'cancelled') {
    this.cancelled = true;
    for (const rec of [...this.calls.values()]) this.reply(rec, { error: reason });
  }
  close() {
    if (this.closed) return;
    this.cancelled = true;
    for (const rec of [...this.calls.values()]) { rec.answered = true; for (const id of rec.ops) { const o = this.ops.get(id); if (o) o.reject(new Cancelled()); } }
    this.calls.clear();
    this.ops.clear();
    this.store.clear();
    this.closed = true;
  }
}

// remove a job's tensor files (<job id>_*.f32) from the shared out_dir
async function cleanupTensors(dir, jobId) {
  let names = [];
  try { names = await fsp.readdir(dir); } catch (_) { return; }
  const pre = jobId + '_';
  await Promise.all(names.filter(n => n.startsWith(pre) && n.endsWith('.f32')).map(n => fsp.unlink(path.join(dir, n)).catch(() => { })));
}

module.exports = { ExtSession, fold, cleanupTensors, makeCtx, normalizeResult, coercePrim, readTensor, writeTensor, newTensor, Cancelled, RESIZE_METHODS, OP_NAMES };
