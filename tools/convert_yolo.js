'use strict';
// Ultralytics YOLOv8 detection .pt -> fused .safetensors + .json, the same output as convert_yolo.py
// but with no Python or torch: a zip reader, a small pickle VM for torch checkpoints, and the
// BatchNorm fusion done in fp32 (each op rounded to fp32, so the weights match torch bit for bit).
//
//   node tools/convert_yolo.js models/detect/face_yolov8m.pt  -> face_yolov8m.safetensors / .json

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

// ---- zip (torch checkpoints are stored, uncompressed zips) ---------------------------------
function openZip(buf) {
  let e = buf.length - 22;
  while (e >= 0 && buf.readUInt32LE(e) !== 0x06054b50) e--;
  if (e < 0) throw new Error('not a zip file (old-style torch checkpoints are not supported)');
  const n = buf.readUInt16LE(e + 10);
  let p = buf.readUInt32LE(e + 16);
  if (n === 0xffff || p === 0xffffffff) throw new Error('zip64 checkpoints are not supported');
  const files = new Map();
  for (let k = 0; k < n; k++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('bad zip central directory');
    const method = buf.readUInt16LE(p + 10), csize = buf.readUInt32LE(p + 20);
    const nl = buf.readUInt16LE(p + 28), xl = buf.readUInt16LE(p + 30), cl = buf.readUInt16LE(p + 32);
    const off = buf.readUInt32LE(p + 42);
    files.set(buf.toString('utf8', p + 46, p + 46 + nl), { method, csize, off });
    p += 46 + nl + xl + cl;
  }
  return {
    names: [...files.keys()],
    read(name) {
      const f = files.get(name);
      if (!f) throw new Error('missing zip entry ' + name);
      const d = f.off + 30 + buf.readUInt16LE(f.off + 26) + buf.readUInt16LE(f.off + 28);
      const raw = buf.subarray(d, d + f.csize);
      if (f.method === 0) return raw;
      if (f.method === 8) return zlib.inflateRawSync(raw);
      throw new Error('unsupported zip compression ' + f.method);
    },
  };
}

// ---- pickle ------------------------------------------------------------------------------
// Dicts become Maps (keys keep their type and order), tuples and lists become arrays, and any
// other class becomes a PyObj whose BUILD state lands in its own properties.
class PyObj {
  constructor(cls, args) {
    Object.defineProperty(this, '__cls', { value: cls });
    if (args && args.length) Object.defineProperty(this, '__args', { value: args });
  }
}
const STORAGE_DTYPE = {
  HalfStorage: 'f16', FloatStorage: 'f32', DoubleStorage: 'f64', BFloat16Storage: 'bf16',
  LongStorage: 'i64', IntStorage: 'i32', ShortStorage: 'i16', CharStorage: 'i8', ByteStorage: 'u8', BoolStorage: 'bool',
};

function unpickle(data, persistentLoad) {
  const stack = [], marks = [], memo = new Map();
  let p = 0;
  const popMark = () => stack.splice(marks.pop());
  const top = () => stack[stack.length - 1];
  const line = () => { const e = data.indexOf(10, p); const s = data.toString('latin1', p, e); p = e + 1; return s; };
  const u8 = () => data[p++];
  const u32 = () => { const v = data.readUInt32LE(p); p += 4; return v; };
  const str = (len, enc) => { const s = data.toString(enc, p, p + len); p += len; return s; };
  const bytes = (len) => { const b = data.subarray(p, p + len); p += len; return b; };
  const setItems = (d, kv) => { for (let i = 0; i < kv.length; i += 2) d instanceof Map ? d.set(kv[i], kv[i + 1]) : (d[kv[i]] = kv[i + 1]); };
  const call = (fn, args) => {
    const name = fn.mod + '.' + fn.name;
    switch (name) {
      case 'torch._utils._rebuild_tensor_v2': {
        const [storage, offset, size, stride] = args;
        return { tensor: true, storage, offset, shape: Array.from(size), stride: Array.from(stride) };
      }
      case 'torch._utils._rebuild_parameter': return args[0];
      case 'collections.OrderedDict': return new Map(args[0] || []);
      case '__builtin__.set': case 'builtins.set': case '__builtin__.frozenset': case 'builtins.frozenset': return new Set(args[0] || []);
      case 'torch.Size': return Array.from(args[0]);
      default: return new PyObj(fn, args);
    }
  };
  for (;;) {
    const op = data[p++];
    switch (op) {
      case 0x80: p++; break; // PROTO
      case 0x95: p += 8; break; // FRAME
      case 0x2e: return stack.pop(); // STOP
      case 0x28: marks.push(stack.length); break; // MARK
      case 0x29: stack.push([]); break; // EMPTY_TUPLE
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x7d: stack.push(new Map()); break; // EMPTY_DICT
      case 0x8f: stack.push(new Set()); break; // EMPTY_SET
      case 0x74: case 0x6c: stack.push(popMark()); break; // TUPLE, LIST
      case 0x85: stack.push([stack.pop()]); break; // TUPLE1
      case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
      case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
      case 0x64: { const d = new Map(); setItems(d, popMark()); stack.push(d); break; } // DICT
      case 0x91: stack.push(new Set(popMark())); break; // FROZENSET
      case 0x61: { const v = stack.pop(); top().push(v); break; } // APPEND
      case 0x65: { const vs = popMark(); top().push(...vs); break; } // APPENDS
      case 0x73: { const v = stack.pop(), k = stack.pop(); setItems(top(), [k, v]); break; } // SETITEM
      case 0x75: { const kv = popMark(); setItems(top(), kv); break; } // SETITEMS
      case 0x90: { const vs = popMark(); for (const v of vs) top().add(v); break; } // ADDITEMS
      case 0x58: { const n = u32(); stack.push(str(n, 'utf8')); break; } // BINUNICODE
      case 0x8c: { const n = u8(); stack.push(str(n, 'utf8')); break; } // SHORT_BINUNICODE
      case 0x54: { const n = u32(); stack.push(str(n, 'latin1')); break; } // BINSTRING
      case 0x55: { const n = u8(); stack.push(str(n, 'latin1')); break; } // SHORT_BINSTRING
      case 0x42: { const n = u32(); stack.push(bytes(n)); break; } // BINBYTES
      case 0x43: { const n = u8(); stack.push(bytes(n)); break; } // SHORT_BINBYTES
      case 0x4a: stack.push(data.readInt32LE(p)); p += 4; break; // BININT
      case 0x4b: stack.push(u8()); break; // BININT1
      case 0x4d: stack.push(data.readUInt16LE(p)); p += 2; break; // BININT2
      case 0x8a: { // LONG1
        const n = u8();
        let v = 0n;
        for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(data[p + i]);
        if (n && data[p + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
        p += n;
        stack.push(Number(v));
        break;
      }
      case 0x47: stack.push(data.readDoubleBE(p)); p += 8; break; // BINFLOAT
      case 0x4e: stack.push(null); break; // NONE
      case 0x88: stack.push(true); break; // NEWTRUE
      case 0x89: stack.push(false); break; // NEWFALSE
      case 0x71: memo.set(u8(), top()); break; // BINPUT
      case 0x72: memo.set(u32(), top()); break; // LONG_BINPUT
      case 0x94: memo.set(memo.size, top()); break; // MEMOIZE
      case 0x68: stack.push(memo.get(u8())); break; // BINGET
      case 0x6a: stack.push(memo.get(u32())); break; // LONG_BINGET
      case 0x63: { const mod = line(), name = line(); stack.push({ mod, name }); break; } // GLOBAL
      case 0x93: { const name = stack.pop(), mod = stack.pop(); stack.push({ mod, name }); break; } // STACK_GLOBAL
      case 0x52: { const args = stack.pop(), fn = stack.pop(); stack.push(call(fn, args)); break; } // REDUCE
      case 0x81: { const args = stack.pop(), cls = stack.pop(); stack.push(new PyObj(cls, args)); break; } // NEWOBJ
      case 0x62: { // BUILD
        let state = stack.pop();
        const obj = top();
        let slots = null;
        if (Array.isArray(state) && state.length === 2) [state, slots] = state;
        for (const s of [state, slots]) if (s instanceof Map) for (const [k, v] of s) obj[k] = v;
        break;
      }
      case 0x51: stack.push(persistentLoad(stack.pop())); break; // BINPERSID
      case 0x30: stack.pop(); break; // POP
      case 0x31: popMark(); break; // POP_MARK
      case 0x32: stack.push(top()); break; // DUP
      default: throw new Error(`unsupported pickle opcode 0x${op.toString(16)} at ${p - 1}`);
    }
  }
}

function loadTorch(file) {
  const zip = openZip(fs.readFileSync(file));
  const pkl = zip.names.find(n => /^[^/]+\/data\.pkl$/.test(n));
  if (!pkl) throw new Error('no data.pkl in ' + file);
  const prefix = pkl.slice(0, -'data.pkl'.length);
  const storages = new Map();
  return unpickle(zip.read(pkl), (pid) => {
    const [kind, type, key, , numel] = pid;
    if (kind !== 'storage') throw new Error('unknown persistent id ' + kind);
    if (!storages.has(key)) {
      const dtype = STORAGE_DTYPE[type.name];
      if (!dtype) throw new Error('unsupported storage type ' + type.name);
      storages.set(key, { dtype, numel, bytes: zip.read(prefix + 'data/' + key) });
    }
    return storages.get(key);
  });
}

// ---- tensors -----------------------------------------------------------------------------
function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
function elemReader(st) {
  const b = st.bytes;
  switch (st.dtype) {
    case 'f16': return (i) => halfToFloat(b.readUInt16LE(i * 2));
    case 'bf16': return (i) => { const t = Buffer.alloc(4); t.writeUInt16LE(b.readUInt16LE(i * 2), 2); return t.readFloatLE(0); };
    case 'f32': return (i) => b.readFloatLE(i * 4);
    case 'f64': return (i) => b.readDoubleLE(i * 8);
    case 'i64': return (i) => Number(b.readBigInt64LE(i * 8));
    case 'i32': return (i) => b.readInt32LE(i * 4);
    case 'i16': return (i) => b.readInt16LE(i * 2);
    case 'i8': return (i) => b.readInt8(i);
    default: return (i) => b[i];
  }
}
const numel = (t) => t.shape.reduce((a, b) => a * b, 1);
// tensor -> Float32Array in row-major order (like .float().contiguous())
function f32(t) {
  const n = numel(t), out = new Float32Array(n), get = elemReader(t.storage), nd = t.shape.length;
  const idx = new Array(nd).fill(0);
  let src = t.offset;
  for (let i = 0; i < n; i++) {
    out[i] = get(src);
    for (let d = nd - 1; d >= 0; d--) {
      idx[d]++;
      src += t.stride[d];
      if (idx[d] < t.shape[d]) break;
      src -= t.stride[d] * t.shape[d];
      idx[d] = 0;
    }
  }
  return out;
}

// ---- nn.Module access ----------------------------------------------------------------------
const kind = (m) => (m && m.__cls ? m.__cls.name : typeof m);
function ga(m, name, dflt) {
  if (m && name in m) return m[name];
  for (const k of ['_parameters', '_buffers', '_modules']) if (m && m[k] instanceof Map && m[k].has(name)) return m[k].get(name);
  if (arguments.length > 2) return dflt;
  throw new Error(`${kind(m)} has no attribute ${name}`);
}
const children = (m) => [...ga(m, '_modules').values()];
const pair = (v) => (Array.isArray(v) ? v : [v, v]);
function toJSON(v) {
  if (v instanceof Map) { const o = {}; for (const [k, x] of v) o[String(k)] = toJSON(x); return o; }
  if (Array.isArray(v)) return v.map(toJSON);
  if (v instanceof Set) return [...v].map(toJSON);
  if (v && v.tensor) return Array.from(f32(v));
  if (v instanceof PyObj) return toJSON(new Map(Object.entries(v)));
  return v;
}

function actName(a) {
  const n = kind(a);
  if (n === 'SiLU') return 'silu';
  if (n === 'Identity') return 'none';
  throw new Error('unsupported activation ' + n);
}

class Converter {
  constructor() { this.tensors = new Map(); }

  geom(c) {
    const k = pair(ga(c, 'kernel_size')), s = pair(ga(c, 'stride')), p = pair(ga(c, 'padding')), d = pair(ga(c, 'dilation'));
    const groups = ga(c, 'groups');
    if (k[0] !== k[1] || s[0] !== s[1] || p[0] !== p[1] || d[0] !== 1 || d[1] !== 1 || groups !== 1)
      throw new Error(`unsupported conv geometry k=${k} s=${s} p=${p} d=${d} g=${groups}`);
    return { cin: ga(c, 'in_channels'), cout: ga(c, 'out_channels'), k: k[0], s: s[0], p: p[0] };
  }

  // ultralytics Conv (conv+bn+act) or a bare nn.Conv2d -> fused record
  conv(p, m) {
    let c, bn, act, name;
    if (kind(m) === 'Conv2d') { c = m; bn = null; act = 'none'; name = p; }
    else { c = ga(m, 'conv'); bn = ga(m, 'bn', null); act = actName(ga(m, 'act')); name = p + '.conv'; }
    const wt = ga(c, 'weight'), bias = ga(c, 'bias', null);
    let w = f32(wt);
    const cout = wt.shape[0], per = w.length / cout;
    let b = bias ? f32(bias) : new Float32Array(cout);
    if (bn) {
      const g = f32(ga(bn, 'weight')), beta = f32(ga(bn, 'bias')), mean = f32(ga(bn, 'running_mean')), vr = f32(ga(bn, 'running_var'));
      const eps = Math.fround(ga(bn, 'eps'));
      const wf = new Float32Array(w.length), bf = new Float32Array(cout);
      for (let o = 0; o < cout; o++) {
        const inv = Math.fround(g[o] / Math.fround(Math.sqrt(Math.fround(vr[o] + eps))));
        for (let j = 0; j < per; j++) wf[o * per + j] = w[o * per + j] * inv;
        bf[o] = Math.fround(Math.fround(b[o] - mean[o]) * inv) + beta[o];
      }
      w = wf; b = bf;
    }
    this.tensors.set(name + '.weight', { shape: wt.shape, data: w });
    this.tensors.set(name + '.bias', { shape: [cout], data: b });
    return { ...this.geom(c), name, act };
  }

  layer(i, m) {
    const p = `model.${i}`, t = kind(m);
    const rec = { i: ga(m, 'i', i), f: toJSON(ga(m, 'f')), type: t };
    if (t === 'Conv') {
      rec.conv = this.conv(p, m);
    } else if (t === 'C2f') {
      rec.c = ga(m, 'c');
      rec.cv1 = this.conv(p + '.cv1', ga(m, 'cv1'));
      rec.cv2 = this.conv(p + '.cv2', ga(m, 'cv2'));
      rec.m = children(ga(m, 'm')).map((b, j) => ({
        cv1: this.conv(`${p}.m.${j}.cv1`, ga(b, 'cv1')), cv2: this.conv(`${p}.m.${j}.cv2`, ga(b, 'cv2')), add: !!ga(b, 'add'),
      }));
    } else if (t === 'SPPF') {
      const mp = ga(m, 'm'), k = ga(mp, 'kernel_size');
      if (!(ga(mp, 'stride') === 1 && ga(mp, 'padding') === Math.floor(k / 2))) throw new Error('unexpected SPPF pool geometry');
      Object.assign(rec, { cv1: this.conv(p + '.cv1', ga(m, 'cv1')), cv2: this.conv(p + '.cv2', ga(m, 'cv2')), k });
    } else if (t === 'Upsample') {
      if (ga(m, 'mode') !== 'nearest') throw new Error('only nearest upsampling is supported');
      Object.assign(rec, { scale: Math.trunc(ga(m, 'scale_factor')), mode: ga(m, 'mode') });
    } else if (t === 'Concat') {
      rec.dim = ga(m, 'd');
    } else if (t === 'Detect') {
      if (ga(m, 'end2end', false)) throw new Error('end2end (one2one) heads are not supported');
      const rm = ga(m, 'reg_max'), dflw = ga(ga(ga(m, 'dfl'), 'conv'), 'weight'), dw = f32(dflw);
      if (dw.length !== rm || dw.some((v, j) => v !== j)) throw new Error('DFL conv is not arange(reg_max)');
      this.tensors.set(p + '.dfl.conv.weight', { shape: dflw.shape, data: dw });
      Object.assign(rec, { nc: ga(m, 'nc'), reg_max: rm, no: ga(m, 'no'), stride: Array.from(f32(ga(m, 'stride'))) });
      const heads = (key) => children(ga(m, key)).map((s, l) => children(s).map((x, j) => this.conv(`${p}.${key}.${l}.${j}`, x)));
      rec.cv2 = heads('cv2');
      rec.cv3 = heads('cv3');
      for (const s of children(ga(m, 'cv3')))
        if (children(s).some(x => !['Conv', 'Conv2d'].includes(kind(x)))) throw new Error('non-legacy Detect cv3 (DWConv) is not supported');
    } else {
      throw new Error('unsupported layer type ' + t);
    }
    return rec;
  }
}

function saveSafetensors(file, tensors, metadata) {
  const names = [...tensors.keys()].sort();
  const header = { __metadata__: metadata };
  let off = 0;
  for (const n of names) {
    const t = tensors.get(n), len = t.data.length * 4;
    header[n] = { dtype: 'F32', shape: t.shape, data_offsets: [off, off + len] };
    off += len;
  }
  let hj = JSON.stringify(header);
  hj += ' '.repeat((8 - (Buffer.byteLength(hj) % 8)) % 8);
  const hb = Buffer.from(hj), lenb = Buffer.alloc(8);
  lenb.writeBigUInt64LE(BigInt(hb.length));
  const fd = fs.openSync(file, 'w');
  try {
    fs.writeSync(fd, lenb);
    fs.writeSync(fd, hb);
    for (const n of names) { const d = tensors.get(n).data; fs.writeSync(fd, Buffer.from(d.buffer, d.byteOffset, d.byteLength)); }
  } finally {
    fs.closeSync(fd);
  }
}

function convert(src) {
  const base = src.replace(/\.[^.\\/]+$/, '');
  const ck = loadTorch(src);
  const model = ck.get('model') != null ? ck.get('model') : ck.get('ema');
  if (model == null) throw new Error('checkpoint has neither model nor ema');
  const cv = new Converter();
  const layers = children(ga(model, 'model')).map((m, i) => cv.layer(i, m));
  const yaml = toJSON(ga(model, 'yaml'));
  let ta = ck.get('train_args') || new Map();
  if (!(ta instanceof Map)) ta = new Map(Object.entries(ta));
  const imgsz = ta.has('imgsz') ? ta.get('imgsz') : 640;
  const meta = {
    format: 'kiln-yolov8-det/1',
    source: path.basename(src),
    ultralytics_version: ck.get('version') ?? null,
    date: ck.get('date') ?? null,
    imgsz: Array.isArray(imgsz) ? imgsz[0] : Math.trunc(imgsz),
    yaml,
    names: toJSON(ga(model, 'names')),
    nc: ga(model, 'nc', null) ?? yaml.nc,
    stride: Array.from(f32(ga(model, 'stride'))),
    save: [...ga(model, 'save')].map(Number).sort((a, b) => a - b),
    layers,
  };
  saveSafetensors(base + '.safetensors', cv.tensors, { format: 'kiln-yolov8-det/1' });
  fs.writeFileSync(base + '.json', JSON.stringify(meta, null, 1));
  let nparam = 0;
  for (const t of cv.tensors.values()) nparam += t.data.length;
  console.log(`layers ${layers.length}  tensors ${cv.tensors.size}  params ${(nparam / 1e6).toFixed(2)}M  names ${JSON.stringify(meta.names)}`);
  console.log('wrote', base + '.safetensors', 'and', base + '.json');
}

module.exports = { convert };

if (require.main === module) {
  if (!process.argv[2]) { console.error('usage: node tools/convert_yolo.js <model.pt>'); process.exit(2); }
  try { convert(process.argv[2]); } catch (e) { console.error('convert_yolo: ' + e.message); process.exit(1); }
}
