'use strict';
// Training datasets: folders of images with a .txt caption of the same name (comma-separated tags). Kiln's own
// datasets live in <root>/datasets/<name>; other folders are opened once and remembered (config/datasets.json).
// Only those folders are read or written: the server may be reachable from the LAN or a tunnel.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const IMG_RE = /\.(png|jpe?g|webp|bmp|gif|avif)$/i;
const NAME_RE = /^[\w\- .()]{1,80}$/;

function createDatasets({ ROOT, CONFIG }) {
  const BASE = path.join(ROOT, 'datasets');
  const THUMBS = path.join(ROOT, 'cache', 'thumbs');
  const REG = path.join(CONFIG, 'datasets.json');

  const readReg = () => { try { return JSON.parse(fs.readFileSync(REG, 'utf8')).folders || []; } catch (_) { return []; } };
  const writeReg = (folders) => { fs.mkdirSync(CONFIG, { recursive: true }); fs.writeFileSync(REG, JSON.stringify({ folders }, null, 2)); };
  const norm = (d) => path.resolve(String(d || '')).replace(/[\\/]+$/, '');
  const same = (a, b) => a.toLowerCase() === b.toLowerCase();

  // a folder the API may touch: inside datasets/, or opened before
  function allowed(dir) {
    const d = norm(dir);
    const rel = path.relative(BASE, d);
    if (rel && !rel.startsWith('..') && !path.isAbsolute(rel) && !rel.includes(path.sep)) return d;
    if (readReg().some(f => same(norm(f), d))) return d;
    throw new Error('not a dataset folder (open it in the Datasets tab first): ' + d);
  }
  // a file inside an allowed folder, by name only
  function fileIn(dir, file) {
    const d = allowed(dir), f = String(file || '');
    if (!f || f !== path.basename(f) || f.startsWith('.')) throw new Error('bad file name');
    return path.join(d, f);
  }
  const capPath = (img) => img.replace(/\.[^.\\/]+$/, '.txt');
  const readCap = (img) => { try { return fs.readFileSync(capPath(img), 'utf8').trim(); } catch (_) { return ''; } };

  function summary(d) {
    let images = 0, captioned = 0, first = null;
    try {
      for (const f of fs.readdirSync(d)) {
        if (!IMG_RE.test(f)) continue;
        images++;
        if (!first) first = f;
        if (fs.existsSync(capPath(path.join(d, f)))) captioned++;
      }
    } catch (_) { return null; }
    return { name: path.basename(d), dir: d, images, captioned, first };
  }

  function list() {
    fs.mkdirSync(BASE, { recursive: true });
    const out = [];
    for (const n of fs.readdirSync(BASE).sort((a, b) => a.localeCompare(b))) {
      const d = path.join(BASE, n);
      try { if (!fs.statSync(d).isDirectory() || n.startsWith('.')) continue; } catch (_) { continue; }
      const s = summary(d);
      if (s) out.push({ ...s, own: true });
    }
    for (const f of readReg()) {
      const s = summary(norm(f));
      out.push(s ? { ...s, own: false } : { name: path.basename(f), dir: norm(f), images: 0, captioned: 0, own: false, missing: true });
    }
    return { base: BASE, list: out };
  }

  function create(name) {
    const n = String(name || '').trim();
    if (!NAME_RE.test(n) || n.startsWith('.')) throw new Error('name it with letters, digits, spaces, - _ . ( )');
    const d = path.join(BASE, n);
    if (fs.existsSync(d)) throw new Error('a dataset called ' + n + ' already exists');
    fs.mkdirSync(d, { recursive: true });
    return summary(d);
  }

  function open(dir) {
    const d = norm(dir);
    let st;
    try { st = fs.statSync(d); } catch (_) { throw new Error('no such folder: ' + d); }
    if (!st.isDirectory()) throw new Error('not a folder: ' + d);
    const reg = readReg();
    const rel = path.relative(BASE, d);
    const inside = rel && !rel.startsWith('..') && !path.isAbsolute(rel) && !rel.includes(path.sep);
    if (!inside && !reg.some(f => same(norm(f), d))) writeReg([...reg, d]);
    return summary(d);
  }

  function forget(dir) { const d = norm(dir); writeReg(readReg().filter(f => !same(norm(f), d))); }

  // tags of a caption: comma separated, trimmed, empties dropped
  const split = (c) => String(c || '').split(',').map(t => t.trim()).filter(Boolean);
  const key = (t) => t.toLowerCase().replace(/_/g, ' ').replace(/\\([()])/g, '$1').replace(/\s+/g, ' ');

  function items(dir) {
    const d = allowed(dir);
    const out = [], freq = new Map();
    for (const f of fs.readdirSync(d).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))) {
      if (!IMG_RE.test(f)) continue;
      const p = path.join(d, f);
      let st;
      try { st = fs.statSync(p); } catch (_) { continue; }
      const caption = readCap(p);
      for (const t of split(caption)) {
        const k = key(t);
        const e = freq.get(k) || { tag: t, n: 0 };
        e.n++;
        freq.set(k, e);
      }
      out.push({ file: f, caption, size: st.size, mtime: Math.round(st.mtimeMs) });
    }
    const tags = [...freq.values()].sort((a, b) => b.n - a.n || a.tag.localeCompare(b.tag)).map(e => [e.tag, e.n]);
    return { dir: d, items: out, tags };
  }

  function writeCaption(dir, file, caption) {
    const img = fileIn(dir, file);
    if (!IMG_RE.test(img) || !fs.existsSync(img)) throw new Error('no such image: ' + file);
    const c = split(caption).join(', ');
    fs.writeFileSync(capPath(img), c ? c + '\n' : '');
    return c;
  }

  // bulk edits over some files (or all): prepend (trigger tags in front, once), append, remove, replace
  function bulk(dir, files, op, tags, to) {
    const d = allowed(dir);
    const all = items(d).items.map(i => i.file);
    const pick = Array.isArray(files) && files.length ? all.filter(f => files.includes(f)) : all;
    const want = split(Array.isArray(tags) ? tags.join(',') : tags);
    const wantKeys = new Set(want.map(key));
    const repl = split(to);
    const changed = [];
    for (const f of pick) {
      const img = path.join(d, f);
      const before = split(readCap(img));
      let after = before;
      if (op === 'prepend') after = [...want, ...before.filter(t => !wantKeys.has(key(t)))];
      else if (op === 'append') { const have = new Set(before.map(key)); after = [...before, ...want.filter(t => !have.has(key(t)))]; }
      else if (op === 'remove') after = before.filter(t => !wantKeys.has(key(t)));
      else if (op === 'replace') {
        const out = [], seen = new Set();
        for (const t of before) for (const u of (wantKeys.has(key(t)) ? repl : [t])) if (!seen.has(key(u))) { seen.add(key(u)); out.push(u); }
        after = out;
      } else throw new Error('unknown edit: ' + op);
      if (after.join(', ') !== before.join(', ')) {
        fs.writeFileSync(capPath(img), after.length ? after.join(', ') + '\n' : '');
        changed.push({ file: f, caption: after.join(', ') });
      }
    }
    return { changed };
  }

  // out of the dataset without deleting anything: into <dataset>/_removed (the trainer only reads the top level)
  function remove(dir, files) {
    const d = allowed(dir);
    const bin = path.join(d, '_removed');
    fs.mkdirSync(bin, { recursive: true });
    let n = 0;
    for (const f of files || []) {
      const img = fileIn(d, f);
      if (!IMG_RE.test(img) || !fs.existsSync(img)) continue;
      for (const p of [img, capPath(img)]) {
        if (!fs.existsSync(p)) continue;
        let dest = path.join(bin, path.basename(p));
        if (fs.existsSync(dest)) dest = path.join(bin, Date.now() + '_' + path.basename(p));
        fs.renameSync(p, dest);
      }
      n++;
    }
    return { removed: n };
  }

  // thumbnails: made by the browser (it decodes every format), cached here by file identity
  function thumbKey(img) {
    const st = fs.statSync(img);
    return crypto.createHash('sha1').update(img.toLowerCase() + '|' + st.size + '|' + Math.round(st.mtimeMs)).digest('hex').slice(0, 24);
  }
  function thumbPath(dir, file) {
    const img = fileIn(dir, file);
    return path.join(THUMBS, thumbKey(img) + '.jpg');
  }
  function putThumb(dir, file, buf) {
    if (!buf || buf.length > 400 * 1024 || buf[0] !== 0xFF || buf[1] !== 0xD8) throw new Error('expected a small JPEG');
    fs.mkdirSync(THUMBS, { recursive: true });
    fs.writeFileSync(thumbPath(dir, file), buf);
  }

  return { BASE, IMG_RE, allowed, fileIn, capPath, readCap, split, key, list, create, open, forget, items, writeCaption, bulk, remove, thumbPath, putThumb };
}

module.exports = { createDatasets };
