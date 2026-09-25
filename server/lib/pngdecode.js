'use strict';
// Full PNG decoder -> RGBA8 (node zlib only).
// Handles colour types 0/2/3/4/6, bit depths 1/2/4/8/16, PLTE + tRNS, all 5 filters and
// Adam7 interlacing. 16-bit samples are reduced to 8 bits (high byte, like most viewers).

const zlib = require('zlib');

const SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
const ADAM7 = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]];

function isPNG(buf) { return buf && buf.length >= 8 && buf.subarray(0, 8).equals(SIG); }

function parse(buf) {
  if (!isPNG(buf)) throw new Error('not a PNG file');
  let p = 8, ihdr = null, plte = null, trns = null;
  const idat = [];
  while (p + 8 <= buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString('latin1', p + 4, p + 8);
    if (p + 12 + len > buf.length) throw new Error('png: truncated ' + type + ' chunk');
    const data = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') ihdr = data;
    else if (type === 'PLTE') plte = data;
    else if (type === 'tRNS') trns = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    p += 12 + len;
  }
  if (!ihdr) throw new Error('png: missing IHDR');
  if (!idat.length) throw new Error('png: missing IDAT');
  const w = ihdr.readUInt32BE(0), h = ihdr.readUInt32BE(4);
  const depth = ihdr[8], ct = ihdr[9], interlace = ihdr[12];
  if (!CHANNELS[ct]) throw new Error('png: bad colour type ' + ct);
  if (![1, 2, 4, 8, 16].includes(depth)) throw new Error('png: bad bit depth ' + depth);
  if (ct === 3 && !plte) throw new Error('png: palette image without PLTE');
  if (!w || !h || w > 32768 || h > 32768 || w * h > 200e6) throw new Error(`png: unsupported size ${w}x${h}`);
  return { w, h, depth, ct, interlace, plte, trns, data: zlib.inflateSync(Buffer.concat(idat)) };
}

// Undo the per-scanline filters of one (sub)image; returns [bytes, nextOffset].
function unfilter(raw, off, pw, ph, bitsPerPixel) {
  const stride = Math.ceil(pw * bitsPerPixel / 8);
  const bpp = Math.max(1, bitsPerPixel >> 3);
  const out = Buffer.alloc(stride * ph);
  for (let y = 0; y < ph; y++) {
    if (off >= raw.length) throw new Error('png: image data too short');
    const f = raw[off++];
    const row = y * stride, prev = row - stride;
    for (let x = 0; x < stride; x++) {
      const v = raw[off + x];
      const a = x >= bpp ? out[row + x - bpp] : 0;
      const b = y > 0 ? out[prev + x] : 0;
      const c = x >= bpp && y > 0 ? out[prev + x - bpp] : 0;
      let r;
      switch (f) {
        case 0: r = v; break;
        case 1: r = v + a; break;
        case 2: r = v + b; break;
        case 3: r = v + ((a + b) >> 1); break;
        case 4: {
          const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
          r = v + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
          break;
        }
        default: throw new Error('png: bad filter type ' + f);
      }
      out[row + x] = r & 255;
    }
    off += stride;
  }
  return [out, off];
}

function decodePNGRGBA(buf) {
  const img = parse(buf);
  const { w, h, depth, ct, interlace, plte, trns, data } = img;
  const ch = CHANNELS[ct];
  const bitsPP = ch * depth;
  const rgba = Buffer.alloc(w * h * 4);
  const maxv = (1 << depth) - 1;
  const scale8 = depth === 16 ? null : 255 / maxv;
  // transparency keys (compared at native depth)
  let keyGray = -1, keyR = -1, keyG = -1, keyB = -1;
  if (trns && ct === 0 && trns.length >= 2) keyGray = trns.readUInt16BE(0);
  if (trns && ct === 2 && trns.length >= 6) { keyR = trns.readUInt16BE(0); keyG = trns.readUInt16BE(2); keyB = trns.readUInt16BE(4); }

  const sampleAt = (bytes, rowOff, idx) => {
    // idx = sample index within the row
    if (depth === 8) return bytes[rowOff + idx];
    if (depth === 16) return (bytes[rowOff + idx * 2] << 8) | bytes[rowOff + idx * 2 + 1];
    const bit = idx * depth;
    const byte = bytes[rowOff + (bit >> 3)];
    const shift = 8 - depth - (bit & 7);
    return (byte >> shift) & maxv;
  };
  const to8 = (v) => depth === 16 ? v >> 8 : depth === 8 ? v : Math.round(v * scale8);

  const put = (bytes, pw, ph, x0, y0, dx, dy) => {
    const stride = Math.ceil(pw * bitsPP / 8);
    for (let y = 0; y < ph; y++) {
      const ro = y * stride;
      const oy = y0 + y * dy;
      for (let x = 0; x < pw; x++) {
        const o = ((oy * w) + (x0 + x * dx)) * 4;
        const s = x * ch;
        let r, g, b, a = 255;
        switch (ct) {
          case 0: {
            const v = sampleAt(bytes, ro, s);
            r = g = b = to8(v);
            if (v === keyGray) a = 0;
            break;
          }
          case 2: {
            const vr = sampleAt(bytes, ro, s), vg = sampleAt(bytes, ro, s + 1), vb = sampleAt(bytes, ro, s + 2);
            r = to8(vr); g = to8(vg); b = to8(vb);
            if (vr === keyR && vg === keyG && vb === keyB) a = 0;
            break;
          }
          case 3: {
            const i = sampleAt(bytes, ro, s);
            if (i * 3 + 2 < plte.length) { r = plte[i * 3]; g = plte[i * 3 + 1]; b = plte[i * 3 + 2]; } else { r = g = b = 0; }
            if (trns && i < trns.length) a = trns[i];
            break;
          }
          case 4: {
            const v = sampleAt(bytes, ro, s);
            r = g = b = to8(v); a = to8(sampleAt(bytes, ro, s + 1));
            break;
          }
          case 6:
            r = to8(sampleAt(bytes, ro, s)); g = to8(sampleAt(bytes, ro, s + 1));
            b = to8(sampleAt(bytes, ro, s + 2)); a = to8(sampleAt(bytes, ro, s + 3));
            break;
        }
        rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = a;
      }
    }
  };

  if (!interlace) {
    const [bytes] = unfilter(data, 0, w, h, bitsPP);
    put(bytes, w, h, 0, 0, 1, 1);
  } else {
    let off = 0;
    for (const [xs, ys, dx, dy] of ADAM7) {
      const pw = Math.ceil((w - xs) / dx), ph = Math.ceil((h - ys) / dy);
      if (pw <= 0 || ph <= 0) continue;
      let bytes;
      [bytes, off] = unfilter(data, off, pw, ph, bitsPP);
      put(bytes, pw, ph, xs, ys, dx, dy);
    }
  }
  return { w, h, rgba, hasAlpha: ct === 4 || ct === 6 || !!trns };
}

module.exports = { decodePNGRGBA, isPNG };
