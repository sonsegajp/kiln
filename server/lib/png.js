'use strict';
// Minimal PNG encoder/decoder (node zlib only).
//  encodePNG(rgb, w, h, text)  -> Buffer   (8-bit RGB, filter 0 per row, tEXt chunks)
//  readPNGText(buf)            -> {key: value}
//  decodePNG(buf)              -> {w, h, channels, data} (8-bit RGB/RGBA/Gray/GA, non-interlaced)
//  thumbnailPNG(buf, maxSide)  -> Buffer (box-filtered RGB PNG) or null

const zlib = require('zlib');

const SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf, crc = 0xffffffff) {
  for (let i = 0; i < buf.length; i++) crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  return crc;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE((crc32(td) ^ 0xffffffff) >>> 0, 0);
  return Buffer.concat([len, td, crc]);
}

// tEXt is Latin-1; callers pass ASCII (JSON with \u escapes) to stay lossless.
// (values with characters beyond Latin-1, e.g. an A1111 'parameters' text, are written as UTF-8 iTXt)
function textChunk(key, value) {
  if (/[^\u0000-\u00ff]/.test(value)) return chunk('iTXt', Buffer.concat([Buffer.from(key, 'latin1'), Buffer.from([0, 0, 0, 0, 0]), Buffer.from(value, 'utf8')]));
  return chunk('tEXt', Buffer.concat([Buffer.from(key, 'latin1'), Buffer.from([0]), Buffer.from(value, 'latin1')]));
}

function rawScanlines(rgb, w, h) {
  const stride = w * 3;
  if (rgb.length < stride * h) throw new Error(`png: rgb buffer too small (${rgb.length} < ${stride * h})`);
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0; // filter: None
    rgb.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride);
  }
  return raw;
}

function encodePNG(rgb, w, h, text = {}, level = 6) {
  const raw = rawScanlines(rgb, w, h);
  return assemblePNG(w, h, text, zlib.deflateSync(raw, { level }));
}

// Same output, but deflate runs on the libuv threadpool so big (upscaled) images
// don't stall the event loop / SSE stream.
function encodePNGAsync(rgb, w, h, text = {}, level = 6) {
  const raw = rawScanlines(rgb, w, h);
  return new Promise((resolve, reject) => {
    zlib.deflate(raw, { level }, (err, z) => {
      if (err) return reject(err);
      try { resolve(assemblePNG(w, h, text, z)); } catch (e) { reject(e); }
    });
  });
}

function assemblePNG(w, h, text, idat) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // color type: truecolor RGB
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter method
  ihdr[12] = 0; // no interlace
  const parts = [SIG, chunk('IHDR', ihdr)];
  for (const [k, v] of Object.entries(text)) parts.push(textChunk(k, v));
  // split IDAT into 1 MiB chunks (valid PNG, friendlier to streaming decoders)
  for (let p = 0; p < idat.length; p += 1 << 20) parts.push(chunk('IDAT', idat.subarray(p, Math.min(idat.length, p + (1 << 20)))));
  if (!idat.length) parts.push(chunk('IDAT', idat));
  parts.push(chunk('IEND', Buffer.alloc(0)));
  return Buffer.concat(parts);
}

function* chunks(buf) {
  if (buf.length < 8 || !buf.subarray(0, 8).equals(SIG)) return;
  let p = 8;
  while (p + 8 <= buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString('latin1', p + 4, p + 8);
    if (p + 12 + len > buf.length) { yield { type, data: null, truncated: true }; return; }
    yield { type, data: buf.subarray(p + 8, p + 8 + len) };
    if (type === 'IEND') return;
    p += 12 + len;
  }
}

function readPNGText(buf) {
  const out = {};
  for (const c of chunks(buf)) {
    if (!c.data) break;
    if (c.type === 'tEXt') {
      const z = c.data.indexOf(0);
      if (z > 0) out[c.data.toString('latin1', 0, z)] = c.data.toString('latin1', z + 1);
    } else if (c.type === 'iTXt') {
      const z = c.data.indexOf(0);
      if (z > 0) {
        const key = c.data.toString('latin1', 0, z);
        const comp = c.data[z + 1];
        let p = z + 3;
        const l1 = c.data.indexOf(0, p); p = l1 + 1;
        const l2 = c.data.indexOf(0, p); p = l2 + 1;
        try {
          const body = c.data.subarray(p);
          out[key] = (comp ? zlib.inflateSync(body) : body).toString('utf8');
        } catch (_) { /* ignore */ }
      }
    } else if (c.type === 'IDAT' || c.type === 'IEND') break;
  }
  return out;
}

function readPNGSize(buf) {
  for (const c of chunks(buf)) {
    if (c.type === 'IHDR' && c.data) return { w: c.data.readUInt32BE(0), h: c.data.readUInt32BE(4) };
    break;
  }
  return null;
}

function decodePNG(buf) {
  let ihdr = null;
  const idat = [];
  for (const c of chunks(buf)) {
    if (!c.data) throw new Error('png: truncated');
    if (c.type === 'IHDR') ihdr = c.data;
    else if (c.type === 'IDAT') idat.push(c.data);
  }
  if (!ihdr) throw new Error('png: no IHDR');
  const w = ihdr.readUInt32BE(0), h = ihdr.readUInt32BE(4);
  const depth = ihdr[8], ct = ihdr[9], interlace = ihdr[12];
  const chMap = { 0: 1, 2: 3, 4: 2, 6: 4 };
  const channels = chMap[ct];
  if (depth !== 8 || !channels || interlace) throw new Error('png: unsupported format');
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * channels;
  const out = Buffer.alloc(stride * h);
  const bpp = channels;
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    const prev = dst - stride;
    for (let x = 0; x < stride; x++) {
      const v = raw[src + x];
      const a = x >= bpp ? out[dst + x - bpp] : 0;
      const b = y > 0 ? out[prev + x] : 0;
      const c = (x >= bpp && y > 0) ? out[prev + x - bpp] : 0;
      let r;
      switch (f) {
        case 0: r = v; break;
        case 1: r = v + a; break;
        case 2: r = v + b; break;
        case 3: r = v + ((a + b) >> 1); break;
        case 4: {
          const p = a + b - c;
          const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          r = v + ((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c));
          break;
        }
        default: throw new Error('png: bad filter ' + f);
      }
      out[dst + x] = r & 0xff;
    }
  }
  return { w, h, channels, data: out };
}

function toRGB(img) {
  if (img.channels === 3) return img.data;
  const n = img.w * img.h;
  const rgb = Buffer.alloc(n * 3);
  for (let i = 0; i < n; i++) {
    const s = i * img.channels;
    if (img.channels >= 3) { rgb[i * 3] = img.data[s]; rgb[i * 3 + 1] = img.data[s + 1]; rgb[i * 3 + 2] = img.data[s + 2]; }
    else { rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = img.data[s]; }
  }
  return rgb;
}

// Area-average downscale of RGB8.
function downscaleRGB(rgb, w, h, tw, th) {
  const out = Buffer.alloc(tw * th * 3);
  for (let ty = 0; ty < th; ty++) {
    const y0 = Math.floor(ty * h / th), y1 = Math.max(y0 + 1, Math.floor((ty + 1) * h / th));
    for (let tx = 0; tx < tw; tx++) {
      const x0 = Math.floor(tx * w / tw), x1 = Math.max(x0 + 1, Math.floor((tx + 1) * w / tw));
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = y0; y < y1; y++) {
        let p = (y * w + x0) * 3;
        for (let x = x0; x < x1; x++, p += 3) { r += rgb[p]; g += rgb[p + 1]; b += rgb[p + 2]; n++; }
      }
      const o = (ty * tw + tx) * 3;
      out[o] = Math.round(r / n); out[o + 1] = Math.round(g / n); out[o + 2] = Math.round(b / n);
    }
  }
  return out;
}

// Downscaled PNG straight from an RGB8 buffer (used at save time, no decode needed).
function thumbnailFromRGB(rgb, w, h, maxSide = 320, level = 9) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  const tw = Math.max(1, Math.round(w * s)), th = Math.max(1, Math.round(h * s));
  const small = s < 1 ? downscaleRGB(rgb, w, h, tw, th) : rgb;
  return encodePNG(small, tw, th, {}, level);
}

function thumbnailPNG(buf, maxSide = 320, level = 9) {
  const img = decodePNG(buf);
  return thumbnailFromRGB(toRGB(img), img.w, img.h, maxSide, level);
}

module.exports = { encodePNG, encodePNGAsync, readPNGText, readPNGSize, decodePNG, thumbnailPNG, thumbnailFromRGB, downscaleRGB, crc32 };
