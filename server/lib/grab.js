'use strict';
// BooruGrab inside Kiln: its settings (config/booru.json, importable from BooruGrab's settings.json), searches, image
// previews through the server (the hosts want a Referer), and a download queue into a dataset folder.
const fs = require('fs');
const path = require('path');
const os = require('os');
const B = require('./booru');

const KEY_MASK = '••••••••';

function createGrab({ CONFIG, datasets, broadcast, log }) {
  const FILE = path.join(CONFIG, 'booru.json');
  const DEFAULTS = {
    sites: B.DEFAULT_SITES, limit: 40, concurrency: 3, tagFormat: 'spaces', includeMeta: false, writeJson: false, skipExisting: true,
    blacklist: [], query: '', ratings: 'gs', sitesOff: [],
  };
  let st = load();

  function load() {
    let s = null;
    try { s = JSON.parse(fs.readFileSync(FILE, 'utf8')); } catch (_) { }
    s = Object.assign({}, DEFAULTS, s || {});
    if (!Array.isArray(s.sites) || !s.sites.length) s.sites = B.DEFAULT_SITES.map(x => ({ ...x }));
    for (const x of s.sites) if (/paheal\.net/.test(x.url) && x.engine !== 'shimmie') x.engine = 'shimmie';
    return s;
  }
  function save() { fs.mkdirSync(CONFIG, { recursive: true }); fs.writeFileSync(FILE, JSON.stringify(st, null, 2)); }
  const site = (id) => st.sites.find(s => s.id === id);

  // for the page: API keys only reach the PC Kiln runs on (other devices see a mask and keep the key unless they type one)
  function settings(local) {
    return { ...st, sites: st.sites.map(s => ({ ...s, key: s.key ? (local ? s.key : KEY_MASK) : '' })), engines: B.ENGINES };
  }
  function update(b) {
    const n = (v, d, lo, hi) => { const x = Number(v); return Number.isFinite(x) ? Math.min(hi, Math.max(lo, Math.round(x))) : d; };
    const next = { ...st };
    if (Array.isArray(b.sites)) {
      const used = new Set();
      next.sites = b.sites.filter(s => s && s.url).map(s => {
        const old = st.sites.find(o => o.id === s.id);
        let id = String(s.id || s.name || 'site').toLowerCase().replace(/[^a-z0-9]+/g, '') || 'site';
        while (used.has(id)) id += '2';
        used.add(id);
        const engine = B.ENGINES.some(e => e.id === s.engine) ? s.engine : 'danbooru';
        return {
          id, name: String(s.name || id).slice(0, 40), engine: /paheal\.net/.test(s.url) ? 'shimmie' : engine, url: String(s.url).trim().replace(/\/+$/, ''),
          user: String(s.user || '').trim(), key: s.key === KEY_MASK && old ? old.key : String(s.key || '').trim(), enabled: s.enabled !== false,
        };
      });
    }
    if (b.limit != null) next.limit = n(b.limit, 40, 5, 200);
    if (b.concurrency != null) next.concurrency = n(b.concurrency, 3, 1, 8);
    if (b.tagFormat != null) next.tagFormat = b.tagFormat === 'raw' ? 'raw' : 'spaces';
    for (const k of ['includeMeta', 'writeJson', 'skipExisting']) if (b[k] != null) next[k] = !!b[k];
    if (b.blacklist != null) next.blacklist = (Array.isArray(b.blacklist) ? b.blacklist : String(b.blacklist).split(/[\s,]+/)).map(t => String(t).trim()).filter(Boolean);
    if (typeof b.query === 'string') next.query = b.query.slice(0, 500);
    if (typeof b.ratings === 'string') next.ratings = b.ratings.replace(/[^gsqe]/g, '');
    if (Array.isArray(b.sitesOff)) next.sitesOff = b.sitesOff.map(String);
    st = next;
    save();
  }
  // BooruGrab's own settings.json (next to BooruGrab.exe): sites with their keys, blacklist and options
  function importBooruGrab(file) {
    const f = file || path.join(os.homedir(), 'BooruGrab', 'settings.json');
    let g;
    try { g = JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { throw new Error('cannot read ' + f); }
    update({
      sites: Array.isArray(g.sites) && g.sites.length ? g.sites : undefined, limit: g.limit, concurrency: g.concurrency, tagFormat: g.tagFormat,
      includeMeta: g.includeMeta, writeJson: g.writeJson, skipExisting: g.skipExisting, blacklist: g.blacklist, ratings: g.ratings, sitesOff: g.sitesOff,
    });
    return { sites: st.sites.length, from: f };
  }

  async function search(siteId, q, page) {
    const s = site(siteId);
    if (!s) throw new Error('unknown site ' + siteId);
    return B.search(s, String(q || '').trim(), Math.max(1, page | 0), st);
  }
  // a thumbnail or sample, fetched with the site's Referer (only from the site's own hosts)
  async function image(siteId, url) {
    const s = site(siteId);
    if (!s || !B.sameSite(s, url)) throw new Error('not an image of that site');
    const r = await B.getBytes(s, url, 60000);
    if (!/^image\//.test(r.type)) throw new Error('not an image');
    return r;
  }
  async function resolve(siteId, post) {
    const s = site(siteId);
    if (!s || !B.sameSite(s, post.page)) throw new Error('not a post of that site');
    await B.resolve(s, post);
    return post;
  }

  // ---- downloads: a queue, a few at a time
  const q = [];
  let active = 0, total = 0;
  const stats = { done: 0, skipped: 0, failed: 0, last_error: '' };
  const status = () => ({ ...stats, pending: q.length + active, queued: q.length, total });
  const emit = (extra) => broadcast({ type: 'booru', ...status(), ...extra });
  function enqueue(dir, posts) {
    const d = datasets.allowed(dir);
    if (!q.length && !active) { total = 0; stats.done = stats.skipped = stats.failed = 0; stats.last_error = ''; }
    let n = 0;
    for (const p of posts || []) {
      const s = site(p && p.site);
      if (!s) continue;
      const urls = [p.file, p.page, p.preview].filter(Boolean);
      if (!urls.every(u => B.sameSite(s, u))) continue;  // only the site's own hosts
      q.push({ dir: d, s, p: { ...p } });
      n++;
    }
    total += n;
    pump();
    emit();
    return n;
  }
  function cancel() { q.length = 0; emit(); }
  function pump() {
    while (active < Math.max(1, st.concurrency) && q.length) {
      const job = q.shift();
      active++;
      one(job).then((r) => {
        if (r.skipped) stats.skipped++;
        else stats.done++;
        emit({ dir: job.dir, file: r.skipped ? null : path.basename(r.file) });
      }, (e) => {
        stats.failed++;
        stats.last_error = `${job.s.name} #${job.p.id}: ${e.message}`;
        emit();
      }).finally(() => { active--; pump(); });
    }
  }
  async function one({ dir, s, p }) {
    const r = await B.download(s, p, dir, st);
    if (!r.skipped && p.preview) {  // the site's thumbnail is the dataset grid's thumbnail
      try {
        const t = await B.getBytes(s, p.preview, 30000);
        if (t.buffer[0] === 0xFF && t.buffer[1] === 0xD8) datasets.putThumb(dir, path.basename(r.file), t.buffer);
      } catch (_) { }
    }
    return r;
  }

  return { settings, update, importBooruGrab, search, image, resolve, enqueue, cancel, status, caption: (p) => B.caption(p, st) };
}

module.exports = { createGrab };
