'use strict';
// Booru taggers for datasets: SmilingWolf's WD v3 models (timm checkpoints the engine runs itself), downloaded from
// Hugging Face into models/taggers/<repo>, plus turning the engine's tag probabilities into captions.
const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');

const KNOWN = [
  { id: 'wd-eva02-large-tagger-v3', name: 'WD EVA02-Large v3', note: 'most accurate', size: 1.26e9 },
  { id: 'wd-vit-large-tagger-v3', name: 'WD ViT-Large v3', note: 'accurate', size: 1.26e9 },
  { id: 'wd-vit-tagger-v3', name: 'WD ViT v3', note: 'fastest', size: 3.8e8 },
];
const FILES = ['config.json', 'selected_tags.csv', 'model.safetensors'];

function createTaggers({ MODELS, onProgress }) {
  const DIR = path.join(MODELS, 'taggers');
  const jobs = new Map();  // id -> { done, total, error }
  const tagCache = new Map();

  const installed = (id) => FILES.every(f => fs.existsSync(path.join(DIR, id, f)));
  function list() {
    return KNOWN.map(k => {
      const j = jobs.get(k.id);
      return { ...k, dir: path.join(DIR, k.id), installed: installed(k.id), downloading: !!(j && !j.error && !j.finished), progress: j ? { done: j.done, total: j.total, error: j.error } : null };
    });
  }

  async function download(id) {
    const k = KNOWN.find(x => x.id === id);
    if (!k) throw new Error('unknown tagger: ' + id);
    if (installed(id)) return;
    const cur = jobs.get(id);
    if (cur && !cur.error && !cur.finished) return;
    const j = { done: 0, total: k.size, error: null, finished: false };
    jobs.set(id, j);
    const dir = path.join(DIR, id);
    fs.mkdirSync(dir, { recursive: true });
    const tick = () => onProgress && onProgress({ id, done: j.done, total: j.total, error: j.error, finished: j.finished });
    (async () => {
      try {
        let base = 0;
        for (const f of FILES) {
          const dest = path.join(dir, f);
          if (fs.existsSync(dest)) continue;
          const r = await fetch(`https://huggingface.co/SmilingWolf/${id}/resolve/main/${f}`, { redirect: 'follow', headers: { 'User-Agent': 'Kiln' } });
          if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
          const len = Number(r.headers.get('content-length')) || 0;
          if (f === 'model.safetensors' && len) j.total = base + len;
          let last = 0;
          const body = Readable.fromWeb(r.body);
          body.on('data', (c) => { j.done += c.length; if (Date.now() - last > 400) { last = Date.now(); tick(); } });
          await pipeline(body, fs.createWriteStream(dest + '.part'));
          fs.renameSync(dest + '.part', dest);
          base = j.done;
        }
        j.finished = true;
      } catch (e) {
        j.error = e.message;
      }
      tick();
    })();
  }

  // selected_tags.csv: tag_id,name,category,count (category 9 rating, 0 general, 4 character)
  function tags(id) {
    if (tagCache.has(id)) return tagCache.get(id);
    const lines = fs.readFileSync(path.join(DIR, id, 'selected_tags.csv'), 'utf8').split(/\r?\n/).slice(1).filter(Boolean);
    const out = lines.map(l => {
      const a = l.indexOf(','), b = l.lastIndexOf(','), c = l.lastIndexOf(',', b - 1);
      return { name: l.slice(a + 1, c), cat: Number(l.slice(c + 1, b)) };
    });
    tagCache.set(id, out);
    return out;
  }

  // probabilities [[index, p], ...] -> tags: characters above their threshold, then general tags, most likely first.
  // Booru names with underscores become spaces (except emoticons like ^_^), as BooruGrab writes them.
  function caption(id, probs, opt) {
    const T = tags(id);
    const exclude = new Set((opt.exclude || []).map(t => t.toLowerCase().replace(/_/g, ' ').trim()));
    const fmt = (n) => opt.underscores ? n : (/[A-Za-z0-9]/.test(n) ? n.replace(/_/g, ' ') : n);
    const chars = [], general = [];
    let rating = null;
    for (const [i, p] of probs) {
      const t = T[i];
      if (!t) continue;
      if (t.cat === 9) { if (!rating || p > rating.p) rating = { name: t.name, p }; continue; }
      const name = fmt(t.name);
      if (exclude.has(name.toLowerCase().replace(/_/g, ' '))) continue;
      if (t.cat === 4 && p >= opt.character) chars.push([name, p]);
      else if (t.cat === 0 && p >= opt.general) general.push([name, p]);
    }
    chars.sort((a, b) => b[1] - a[1]);
    general.sort((a, b) => b[1] - a[1]);
    const list = [...chars, ...general].map(x => x[0]);
    if (opt.rating && rating) list.unshift(rating.name);
    return { tags: list, rating: rating ? rating.name : null };
  }

  return { DIR, KNOWN, list, download, installed, tags, caption, dirOf: (id) => path.join(DIR, id) };
}

module.exports = { createTaggers };
