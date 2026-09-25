'use strict';
// CivitAI (civitai.red / civitai.com, same API): settings (the API key never leaves the server),
// search / model / by-hash lookups with a small cache, description sanitizing, and a resumable
// download queue (disk-space check, Range resume of .part files, SHA256 verify, sidecar + preview).

const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const http = require('http');
const https = require('https');
const crypto = require('crypto');
const { EventEmitter } = require('events');
const { Transform, pipeline } = require('stream');

const DOMAINS = ['civitai.red', 'civitai.com'];
const MARGIN = 500 * 1024 * 1024;               // keep 500 MB free after a download
const TYPE_DIRS = { Checkpoint: 'checkpoints', LORA: 'loras', LoCon: 'loras', DoRA: 'loras', LyCORIS: 'loras', TextualInversion: 'embeddings', Upscaler: 'upscale' };
const PICKLE_EXT = /\.(ckpt|pt|pth|bin|pkl|pickle)$/i;

const fmtBytes = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : Math.max(0, Math.round(n / 1e3)) + ' KB');
const uerr = (msg, code = 400) => Object.assign(new Error(msg), { code, user: true });

// ---------------------------------------------------------------------------
// HTML sanitizer (allow-list): model descriptions come from CivitAI users
// ---------------------------------------------------------------------------
const ALLOWED = new Set(['p', 'br', 'strong', 'b', 'em', 'i', 'u', 's', 'strike', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'code', 'pre', 'a', 'hr', 'span', 'div', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'sup', 'sub']);
const DROP_WITH_CONTENT = new Set(['script', 'style', 'iframe', 'object', 'embed', 'noscript', 'template', 'svg', 'math', 'form', 'textarea', 'select', 'button', 'head', 'title', 'frame', 'frameset', 'applet']);
const escText = (s) => s.replace(/[<>]/g, (c) => (c === '<' ? '&lt;' : '&gt;'));
function sanitizeHtml(html) {
  html = String(html || '').slice(0, 200000);
  let out = '';
  const open = [];
  let skip = null, skipDepth = 0;
  const re = /<!--[\s\S]*?-->|<(\/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^\s"'<>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*)\s*(\/?)>/g;
  let last = 0, m;
  while ((m = re.exec(html))) {
    const text = html.slice(last, m.index);
    last = re.lastIndex;
    if (!skip) out += escText(text);
    if (m[0].startsWith('<!--')) continue;
    const closing = !!m[1], tag = m[2].toLowerCase();
    if (skip) {
      if (tag === skip) skipDepth += closing ? -1 : (m[4] ? 0 : 1);
      if (skipDepth <= 0) skip = null;
      continue;
    }
    if (DROP_WITH_CONTENT.has(tag)) { if (!closing && !m[4]) { skip = tag; skipDepth = 1; } continue; }
    if (!ALLOWED.has(tag)) continue;
    if (closing) {
      const i = open.lastIndexOf(/^h[1-3]$/.test(tag) ? 'h4' : tag);
      if (i >= 0) { while (open.length > i) out += `</${open.pop()}>`; }
      continue;
    }
    const t = /^h[1-3]$/.test(tag) ? 'h4' : tag;   // no giant headings in a card
    if (t === 'br' || t === 'hr') { out += `<${t}>`; continue; }
    let attrs = '';
    if (t === 'a') {
      const hm = /\shref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/i.exec(m[3] || '');
      const href = hm ? (hm[1] || hm[2] || hm[3] || '').replace(/&amp;/g, '&').trim() : '';
      if (/^https?:\/\//i.test(href)) attrs = ` href="${href.replace(/"/g, '%22').replace(/</g, '%3C')}" target="_blank" rel="noopener noreferrer nofollow"`;
    }
    out += `<${t}${attrs}>`;
    open.push(t);
  }
  if (!skip) out += escText(html.slice(last));
  while (open.length) out += `</${open.pop()}>`;
  return out;
}

// ---------------------------------------------------------------------------
// response normalizing
// ---------------------------------------------------------------------------
// image.civitai.com/<key>/<uuid>/<transform>/<name>: set the transform (width, still frame for videos)
function imgUrl(url, width, still) {
  if (typeof url !== 'string' || !/^https:\/\//.test(url)) return null;
  const parts = url.split('/');
  if (parts.length < 6) return url;
  const t = `${still ? 'anim=false,' : ''}width=${width}`;
  if (parts[parts.length - 2].includes('=')) parts[parts.length - 2] = t;
  else parts.splice(parts.length - 1, 0, t);
  return parts.join('/');
}
function normImage(i) {
  const video = i.type === 'video';
  return { url: i.url, thumb: imgUrl(i.url, 320, true), view: imgUrl(i.url, 900, false), type: video ? 'video' : 'image', nsfwLevel: i.nsfwLevel || 0, w: i.width, h: i.height };
}
function normFile(f) {
  const fmt = f.metadata && f.metadata.format;
  const name = String(f.name || '');
  return {
    id: f.id, name, sizeKB: f.sizeKB, type: f.type, format: fmt, fp: f.metadata && f.metadata.fp, size: f.metadata && f.metadata.size,
    primary: !!f.primary, sha256: f.hashes && f.hashes.SHA256 ? String(f.hashes.SHA256).toUpperCase() : null,
    safe: /\.safetensors$/i.test(name) && (!fmt || fmt === 'SafeTensor'),
    pickle: PICKLE_EXT.test(name) || fmt === 'PickleTensor',
    scan: f.pickleScanResult, virus: f.virusScanResult, downloadUrl: f.downloadUrl,
  };
}
function normVersion(v, full) {
  return {
    id: v.id, name: v.name, baseModel: v.baseModel || '', trainedWords: (v.trainedWords || []).slice(0, 40).map(String),
    publishedAt: v.publishedAt || v.createdAt, availability: v.availability, early: v.availability === 'EarlyAccess',
    stats: v.stats ? { downloads: v.stats.downloadCount, likes: v.stats.thumbsUpCount } : undefined,
    files: (v.files || []).map(normFile),
    images: (v.images || []).slice(0, full ? 24 : 4).map(normImage),
    description: full && v.description ? sanitizeHtml(v.description) : undefined,
  };
}
function normModel(m, domain, full) {
  const s = m.stats || {};
  return {
    id: m.id, name: m.name, type: m.type, nsfw: !!m.nsfw, nsfwLevel: m.nsfwLevel || 0, poi: !!m.poi,
    creator: m.creator ? { username: m.creator.username, image: m.creator.image ? imgUrl(m.creator.image, 64, true) : null } : null,
    stats: { downloads: s.downloadCount || 0, likes: s.thumbsUpCount || 0, dislikes: s.thumbsDownCount || 0, comments: s.commentCount || 0 },
    tags: (m.tags || []).slice(0, full ? 30 : 8).map(t => (typeof t === 'string' ? t : t && t.name)).filter(Boolean),
    url: `https://${domain}/models/${m.id}`,
    versions: (m.modelVersions || []).slice(0, full ? 50 : 8).map(v => normVersion(v, full)),
    description: full ? sanitizeHtml(m.description) : undefined,
  };
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
function request(u, headers, signal) {
  return new Promise((resolve, reject) => {
    const lib = u.protocol === 'https:' ? https : http;
    const req = lib.request(u, { method: 'GET', headers: Object.assign({ 'User-Agent': 'Kiln (local image generator)', Accept: '*/*' }, headers), signal, timeout: 60000 }, resolve);
    req.on('timeout', () => req.destroy(new Error('connection timed out')));
    req.on('error', reject);
    req.end();
  });
}
async function readSmall(res, limit = 65536) {
  const chunks = [];
  let n = 0;
  for await (const c of res) { n += c.length; if (n <= limit) chunks.push(c); }
  return Buffer.concat(chunks).toString('utf8');
}

class CivitAI extends EventEmitter {
  constructor({ configDir, modelsDir, log, testBase }) {
    super();
    this.configDir = configDir;
    this.modelsDir = modelsDir;
    this.log = log || (() => { });
    this.testBase = testBase || null;       // tests only: a local mock instead of https://<domain>
    this.file = path.join(configDir, 'civitai.json');
    this.cfg = { api_key: '', domain: 'civitai.red', nsfw: false, folders: {} };
    try { Object.assign(this.cfg, JSON.parse(fs.readFileSync(this.file, 'utf8'))); } catch (_) { }
    if (!DOMAINS.includes(this.cfg.domain)) this.cfg.domain = 'civitai.red';
    this.cache = new Map();
    this.idsFile = path.join(configDir, 'civitai_ids.json');
    this.ids = {};
    try { this.ids = JSON.parse(fs.readFileSync(this.idsFile, 'utf8')) || {}; } catch (_) { }
    this.dlFile = path.join(configDir, 'downloads.json');
    this.downloads = [];
    try { this.downloads = (JSON.parse(fs.readFileSync(this.dlFile, 'utf8')) || []).filter(d => d && d.id); } catch (_) { }
    for (const d of this.downloads) if (d.status === 'downloading' || d.status === 'verifying') d.status = 'queued';   // resume after a restart
    this.active = null;
  }

  // ---- settings ----
  save() { fs.mkdirSync(this.configDir, { recursive: true }); fs.writeFileSync(this.file, JSON.stringify(this.cfg, null, 2)); }
  base() { return this.testBase || `https://${this.cfg.domain}`; }
  apiHost() { return new URL(this.base()).host; }
  publicSettings() {
    const k = this.cfg.api_key || '';
    return {
      domain: this.cfg.domain, domains: DOMAINS, nsfw: !!this.cfg.nsfw, has_key: !!k, key_hint: k ? '••••' + k.slice(-4) : '',
      folders: Object.fromEntries(Object.keys(TYPE_DIRS_UNIQUE).map(t => [t, this.cfg.folders && this.cfg.folders[t] || ''])),
      resolved: Object.fromEntries(Object.keys(TYPE_DIRS_UNIQUE).map(t => [t, this.folder(t)])),
    };
  }
  setSettings(b) {
    if (b.clear_key) this.cfg.api_key = '';
    if (typeof b.api_key === 'string' && b.api_key.trim()) {
      const k = b.api_key.trim();
      if (!/^[A-Za-z0-9_\-.]{16,200}$/.test(k)) throw uerr('that does not look like a CivitAI API key (Account settings → API keys on the site)');
      this.cfg.api_key = k;
    }
    if (b.domain !== undefined) {
      if (!DOMAINS.includes(b.domain)) throw uerr('domain must be civitai.red or civitai.com');
      this.cfg.domain = b.domain;
      this.cache.clear();
    }
    if (b.nsfw !== undefined) this.cfg.nsfw = !!b.nsfw;
    if (b.folders && typeof b.folders === 'object') {
      this.cfg.folders = this.cfg.folders || {};
      for (const [t, v] of Object.entries(b.folders)) {
        if (!TYPE_DIRS_UNIQUE[t]) continue;
        const s = String(v || '').trim();
        if (!s) { delete this.cfg.folders[t]; continue; }
        const abs = path.isAbsolute(s) ? path.resolve(s) : path.resolve(this.modelsDir, s);
        if (!path.isAbsolute(s) && !abs.startsWith(this.modelsDir + path.sep)) throw uerr('relative folders must stay inside the models folder');
        if (/^\\\\/.test(abs)) throw uerr('network (UNC) paths are not supported');
        this.cfg.folders[t] = s;
      }
    }
    this.save();
    return this.publicSettings();
  }
  // absolute folder for a kind: checkpoints | loras | upscale | embeddings
  folder(kind) {
    const s = this.cfg.folders && this.cfg.folders[kind];
    if (!s) return path.join(this.modelsDir, kind);
    return path.isAbsolute(s) ? path.resolve(s) : path.resolve(this.modelsDir, s);
  }
  extraDirs() {
    const out = {};
    for (const k of Object.keys(TYPE_DIRS_UNIQUE)) {
      const d = this.folder(k);
      if (!d.startsWith(this.modelsDir + path.sep)) out[k] = d;
    }
    return out;
  }

  // ---- API ----
  async api(p, { auth = true, ttl = 300000 } = {}) {
    const url = this.base() + p;
    const c = this.cache.get(url);
    if (c && Date.now() - c.t < ttl) return c.data;
    const headers = { Accept: 'application/json' };
    if (auth && this.cfg.api_key) headers.Authorization = 'Bearer ' + this.cfg.api_key;
    let r;
    try { r = await fetch(url, { headers, signal: AbortSignal.timeout(25000), redirect: 'follow' }); }
    catch (e) { throw uerr(`can't reach ${this.cfg.domain}: ${e.cause && e.cause.code || e.message}`, 502); }
    const text = await r.text();
    let data = null;
    try { data = JSON.parse(text); } catch (_) { }
    if (!r.ok) throw this.httpError(r.status, data);
    if (!data) throw uerr(`${this.cfg.domain} returned something that isn't JSON`, 502);
    this.cache.set(url, { t: Date.now(), data });
    if (this.cache.size > 300) this.cache.delete(this.cache.keys().next().value);
    return data;
  }
  httpError(status, body) {
    const msg = body && (body.message || body.error) ? String(body.message || body.error).slice(0, 300) : '';
    const key = !!this.cfg.api_key;
    if (status === 401) return uerr(key ? `CivitAI rejected the API key (401)${msg ? ': ' + msg : ''}. Check it in Settings.` : `This file needs a CivitAI login (401)${msg ? ': ' + msg : ''}. Add your API key in Settings.`, 401);
    if (status === 403) return uerr(`CivitAI refused access (403)${msg ? ': ' + msg : ''}. Early-access or restricted models must be unlocked on the site with your account.`, 403);
    if (status === 404) return uerr('not found on CivitAI' + (msg ? ': ' + msg : ''), 404);
    if (status === 429) return uerr('CivitAI is rate-limiting requests; wait a minute and try again', 429);
    return uerr(`CivitAI error ${status}${msg ? ': ' + msg : ''}`, 502);
  }
  async search(q) {
    const sp = new URLSearchParams();
    sp.set('limit', String(Math.min(48, Math.max(1, Number(q.limit) || 24))));
    for (const t of [].concat(q.types || []).filter(Boolean)) sp.append('types', t);
    for (const b of [].concat(q.baseModels || []).filter(Boolean)) sp.append('baseModels', b);
    if (q.query) sp.set('query', String(q.query).slice(0, 200));
    if (q.tag) sp.set('tag', String(q.tag).slice(0, 80));
    if (q.username) sp.set('username', String(q.username).slice(0, 80));
    if (q.sort) sp.set('sort', String(q.sort));
    if (q.period) sp.set('period', String(q.period));
    sp.set('nsfw', q.nsfw ? 'true' : 'false');
    if (q.cursor) sp.set('cursor', String(q.cursor));
    const j = await this.api('/api/v1/models?' + sp.toString(), { ttl: 120000 });
    return { items: (j.items || []).map(m => normModel(m, this.cfg.domain, false)), nextCursor: j.metadata && j.metadata.nextCursor || null };
  }
  async model(id) {
    if (!/^\d{1,12}$/.test(String(id))) throw uerr('bad model id');
    return normModel(await this.api('/api/v1/models/' + id), this.cfg.domain, true);
  }
  async version(id) {
    if (!/^\d{1,12}$/.test(String(id))) throw uerr('bad version id');
    return this.api('/api/v1/model-versions/' + id);
  }
  // CivitAI info for a SHA256 (cached; misses are remembered for a day)
  async byHash(sha) {
    sha = String(sha || '').toUpperCase();
    if (!/^[0-9A-F]{64}$/.test(sha)) throw uerr('bad SHA256');
    const c = this.ids[sha];
    if (c && (c.found || Date.now() - c.at < 86400000)) return c.found ? c : null;
    let v = null;
    try { v = await this.api('/api/v1/model-versions/by-hash/' + sha, { ttl: 0 }); }
    catch (e) { if (e.code !== 404) throw e; }
    const rec = v ? {
      found: true, at: Date.now(), modelId: v.modelId, versionId: v.id, name: v.model && v.model.name, versionName: v.name,
      type: v.model && v.model.type, baseModel: v.baseModel, trainedWords: (v.trainedWords || []).slice(0, 40),
      image: v.images && v.images[0] ? imgUrl(v.images[0].url, 256, true) : null, nsfw: !!(v.model && v.model.nsfw),
    } : { found: false, at: Date.now() };
    this.ids[sha] = rec;
    try { fs.mkdirSync(this.configDir, { recursive: true }); fs.writeFileSync(this.idsFile, JSON.stringify(this.ids, null, 1)); } catch (_) { }
    return rec.found ? rec : null;
  }

  // ---------------------------------------------------------------------------
  // downloads
  // ---------------------------------------------------------------------------
  publicDownloads() {
    return this.downloads.map(d => ({
      id: d.id, versionId: d.versionId, modelId: d.modelId, name: d.name, versionName: d.versionName, type: d.type, baseModel: d.baseModel,
      file: path.basename(d.dest), dest: d.dest, rel: d.rel, size: d.size, received: d.received || 0, speed: d.speed || 0, eta: d.eta,
      status: d.status, error: d.error, created: d.created, finished: d.finished, trainedWords: d.trainedWords,
    }));
  }
  saveDownloads() {
    clearTimeout(this.dlSaveT);
    this.dlSaveT = setTimeout(() => {
      const keep = this.downloads.filter(d => !['done', 'cancelled'].includes(d.status) || Date.now() - (d.finished || 0) < 7 * 86400000).slice(-60);
      const plain = keep.map(({ abort, ...d }) => d);
      try { fs.mkdirSync(this.configDir, { recursive: true }); fs.writeFileSync(this.dlFile, JSON.stringify(plain, null, 1)); } catch (_) { }
    }, 400);
  }
  emitDl(d, force) {
    const now = Date.now();
    if (!force && now - (d.lastEmit || 0) < 250) return;
    d.lastEmit = now;
    this.emit('download', this.publicDownloads().find(x => x.id === d.id));
  }
  freeBytes(dir) {
    try { const s = fs.statfsSync(dir); return Number(s.bavail) * Number(s.bsize); } catch (_) { return Infinity; }
  }
  spaceCheck(dir, bytes) {
    const free = this.freeBytes(dir);
    if (free < bytes + MARGIN) {
      const drive = path.parse(dir).root || dir;
      throw uerr(`Not enough disk space on ${drive}: this needs ${fmtBytes(bytes)} plus 500 MB spare (${fmtBytes(bytes + MARGIN)}), but only ${fmtBytes(free)} is free. Free up space, or pick a download folder on another drive in Settings.`, 507);
    }
    return free;
  }
  // queue a version's file for download; throws a clear error for unusable files
  async enqueue(versionId, fileId) {
    const v = await this.version(versionId);
    const type = v.model && v.model.type;
    const kind = TYPE_DIRS[type];
    if (!kind) throw uerr(`Kiln can't use ${type || 'this kind of'} files`);
    const files = (v.files || []).map(normFile);
    const f = fileId ? files.find(x => String(x.id) === String(fileId)) : (files.find(x => x.primary && x.safe) || files.find(x => x.safe) || files.find(x => x.primary) || files[0]);
    if (!f) throw uerr('this version has no files');
    if (!f.safe) throw uerr(`${f.name} is a ${f.pickle ? 'pickle (.ckpt/.pt) file: those can run arbitrary code when loaded, so Kiln only downloads .safetensors' : 'non-safetensors file; Kiln only downloads .safetensors'}. Pick a version that offers a SafeTensor file.`);
    const dir = this.folder(kind);
    fs.mkdirSync(dir, { recursive: true });
    let name = String(f.name).replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/\s+/g, ' ').trim().slice(0, 150);
    if (!/\.safetensors$/i.test(name)) name += '.safetensors';
    let dest = path.join(dir, name);
    const size = Math.round((f.sizeKB || 0) * 1024);
    const dup = this.downloads.find(d => d.versionId === v.id && String(d.fileId) === String(f.id) && ['queued', 'downloading', 'verifying'].includes(d.status));
    if (dup) return dup;
    if (fs.existsSync(dest)) {
      const side = readJSON(dest.replace(/\.safetensors$/i, '.civitai.json'));
      if (side && side.versionId === v.id) throw uerr(`already downloaded: ${path.relative(this.modelsDir, dest)}`, 409);
      dest = path.join(dir, name.replace(/\.safetensors$/i, `-${v.id}.safetensors`));
      if (fs.existsSync(dest)) throw uerr(`already downloaded: ${path.relative(this.modelsDir, dest)}`, 409);
    }
    let have = 0;
    try { have = fs.statSync(dest + '.part').size; } catch (_) { }
    this.spaceCheck(dir, Math.max(0, size - have));
    const d = {
      id: 'd' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
      versionId: v.id, modelId: v.modelId, fileId: f.id, name: v.model && v.model.name || f.name, versionName: v.name, type, kind,
      baseModel: v.baseModel || '', trainedWords: (v.trainedWords || []).slice(0, 40), nsfw: !!(v.model && v.model.nsfw),
      sha256: f.sha256, size, url: f.downloadUrl || `${this.base()}/api/download/models/${v.id}`,
      preview: pickPreview(v.images || []), dest, rel: path.relative(this.modelsDir, dest).split(path.sep).join('/'),
      status: 'queued', received: have, created: Date.now(),
    };
    this.downloads.push(d);
    this.saveDownloads();
    this.emitDl(d, true);
    this.pump();
    return d;
  }
  cancel(id) {
    const d = this.downloads.find(x => x.id === id);
    if (!d) throw uerr('no such download', 404);
    if (['done', 'cancelled'].includes(d.status)) return d;
    d.status = 'cancelled';
    d.finished = Date.now();
    if (d.abort) d.abort.abort();
    else fsp.unlink(d.dest + '.part').catch(() => { });
    d.received = 0;
    this.saveDownloads();
    this.emitDl(d, true);
    return d;
  }
  retry(id) {
    const d = this.downloads.find(x => x.id === id);
    if (!d) throw uerr('no such download', 404);
    if (!['error', 'cancelled'].includes(d.status)) return d;
    let have = 0;
    try { have = fs.statSync(d.dest + '.part').size; } catch (_) { }
    this.spaceCheck(path.dirname(d.dest), Math.max(0, d.size - have));
    Object.assign(d, { status: 'queued', error: null, received: have, finished: null });
    this.saveDownloads();
    this.emitDl(d, true);
    this.pump();
    return d;
  }
  clearFinished() {
    this.downloads = this.downloads.filter(d => !['done', 'cancelled', 'error'].includes(d.status));
    this.saveDownloads();
  }
  pump() {
    if (this.active) return;
    const d = this.downloads.find(x => x.status === 'queued');
    if (!d) return;
    this.active = d;
    this.run(d).catch(() => { }).finally(() => { this.active = null; setImmediate(() => this.pump()); });
  }
  validUrl(u) {
    const url = new URL(u, this.base());
    const b = new URL(this.base());
    if (url.host !== b.host && !DOMAINS.some(dm => url.host === dm)) throw uerr('unexpected download host ' + url.host);
    url.protocol = b.protocol; url.host = b.host;   // always the configured site (and its key)
    return url;
  }
  // site=true: a CivitAI download URL (pinned to the configured site, gets the key); else any https URL (images)
  async open(u, headers, signal, site = true) {
    let url = site ? this.validUrl(u) : new URL(u);
    if (!site && url.protocol !== 'https:' && !(this.testBase && url.protocol === 'http:')) throw uerr('bad image URL');
    const apiHost = this.apiHost();
    for (let i = 0; i < 8; i++) {
      const h = Object.assign({}, headers);
      if (this.cfg.api_key && url.host === apiHost) h.Authorization = 'Bearer ' + this.cfg.api_key;   // never to the CDN
      const res = await request(url, h, signal);
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
        res.resume();
        const next = new URL(res.headers.location, url);
        if (next.protocol !== 'https:' && !(this.testBase && next.protocol === 'http:')) throw uerr('refusing a non-https redirect');
        url = next;
        continue;
      }
      return res;
    }
    throw uerr('too many redirects');
  }
  async run(d) {
    d.status = 'downloading';
    d.error = null;
    d.abort = new AbortController();
    this.emitDl(d, true);
    const part = d.dest + '.part';
    const t0 = Date.now();
    try {
      fs.mkdirSync(path.dirname(d.dest), { recursive: true });
      let have = 0;
      try { have = fs.statSync(part).size; } catch (_) { }
      if (d.size && have > d.size) { fs.unlinkSync(part); have = 0; }
      this.spaceCheck(path.dirname(d.dest), Math.max(0, d.size - have));
      let hash = crypto.createHash('sha256');
      if (have) {
        await new Promise((res, rej) => fs.createReadStream(part, { highWaterMark: 4 << 20 }).on('data', b => hash.update(b)).on('end', res).on('error', rej));
        this.log(`download ${d.name}: resuming at ${fmtBytes(have)}`);
      }
      const res = await this.open(d.url, have ? { Range: `bytes=${have}-` } : {}, d.abort.signal);
      if (res.statusCode >= 400) {
        const body = await readSmall(res);
        let j = null;
        try { j = JSON.parse(body); } catch (_) { }
        throw this.httpError(res.statusCode, j);
      }
      if (res.statusCode === 200 && have) { have = 0; hash = crypto.createHash('sha256'); }   // server ignored Range: start over
      else if (res.statusCode === 206) {
        const m = /bytes (\d+)-/.exec(res.headers['content-range'] || '');
        if (!m || Number(m[1]) !== have) { res.resume(); throw uerr('the server resumed at the wrong offset; retry to start over', 502); }
      } else if (res.statusCode !== 200) { res.resume(); throw uerr('unexpected response ' + res.statusCode, 502); }
      const ct = String(res.headers['content-type'] || '');
      if (/text\/html|application\/json/.test(ct)) { const body = await readSmall(res); throw uerr('CivitAI returned a page instead of the file: ' + body.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').slice(0, 200), 502); }
      const len = Number(res.headers['content-length'] || 0);
      const total = len ? have + len : d.size;
      if (total) d.size = total;
      if (have + len > 0) this.spaceCheck(path.dirname(d.dest), len);
      d.received = have;
      let lastT = Date.now(), lastB = have;
      const counter = new Transform({
        transform: (chunk, enc, cb) => {
          hash.update(chunk);
          d.received += chunk.length;
          const now = Date.now();
          if (now - lastT >= 500) {
            const sp = (d.received - lastB) / ((now - lastT) / 1000);
            d.speed = d.speed ? d.speed * 0.6 + sp * 0.4 : sp;
            d.eta = d.size && d.speed ? Math.round((d.size - d.received) / d.speed) : null;
            lastT = now; lastB = d.received;
            this.emitDl(d);
          }
          cb(null, chunk);
        },
      });
      await new Promise((resolve, reject) => pipeline(res, counter, fs.createWriteStream(part, { flags: have ? 'a' : 'w' }), (e) => (e ? reject(e) : resolve())));
      d.status = 'verifying';
      this.emitDl(d, true);
      const sha = hash.digest('hex').toUpperCase();
      if (d.sha256 && sha !== d.sha256) {
        await fsp.unlink(part).catch(() => { });
        d.received = 0;
        throw uerr(`checksum mismatch (got ${sha.slice(0, 10)}…, CivitAI lists ${d.sha256.slice(0, 10)}…): the file was corrupted in transit and has been deleted; retry`, 502);
      }
      if (fs.existsSync(d.dest)) throw uerr('a file with that name appeared meanwhile: ' + d.rel, 409);
      await fsp.rename(part, d.dest);
      d.sha256 = sha;
      await this.writeSidecar(d);
      d.status = 'done';
      d.finished = Date.now();
      d.speed = 0; d.eta = 0;
      this.log(`downloaded ${d.rel} (${fmtBytes(d.size)}) in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
      this.emitDl(d, true);
      this.emit('installed', d);
    } catch (e) {
      if (d.status === 'cancelled') {
        await fsp.unlink(part).catch(() => { });
        this.log(`download ${d.name} cancelled`);
      } else {
        d.status = 'error';
        d.error = e.user ? e.message : `download failed: ${e.message}`;
        d.finished = Date.now();
        this.log(`download ${d.name}: ${d.error}`);
      }
      this.emitDl(d, true);
    } finally {
      d.abort = null;
      this.saveDownloads();
    }
  }
  async writeSidecar(d) {
    const stem = d.dest.replace(/\.safetensors$/i, '');
    const side = {
      source: 'civitai', domain: this.cfg.domain, modelId: d.modelId, versionId: d.versionId, name: d.name, versionName: d.versionName,
      type: d.type, baseModel: d.baseModel, trainedWords: d.trainedWords, sha256: d.sha256, file: path.basename(d.dest), nsfw: d.nsfw,
      url: `https://${this.cfg.domain}/models/${d.modelId}?modelVersionId=${d.versionId}`, downloadedAt: new Date().toISOString(),
    };
    if (d.preview) {
      try {
        const res = await this.open(d.preview, {}, AbortSignal.timeout(20000), false);
        const ct = String(res.headers['content-type'] || '');
        const ext = /png/.test(ct) ? '.png' : /webp/.test(ct) ? '.webp' : /jpe?g/.test(ct) ? '.jpeg' : null;
        if (res.statusCode === 200 && ext) {
          const chunks = [];
          let n = 0;
          for await (const c of res) { n += c.length; if (n > 4 << 20) throw new Error('preview too large'); chunks.push(c); }
          await fsp.writeFile(stem + '.preview' + ext, Buffer.concat(chunks));
          side.preview = path.basename(stem + '.preview' + ext);
        } else res.resume();
      } catch (e) { this.log(`preview for ${d.name}: ${e.message}`); }
    }
    await fsp.writeFile(stem + '.civitai.json', JSON.stringify(side, null, 2));
  }
}
const TYPE_DIRS_UNIQUE = { checkpoints: 1, loras: 1, upscale: 1, embeddings: 1 };
function readJSON(f) { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (_) { return null; } }
// a small still preview, preferring safe-for-work images
function pickPreview(images) {
  const im = images.find(i => (i.nsfwLevel || 0) <= 2 && i.type !== 'video') || images.find(i => i.type !== 'video') || images[0];
  return im ? imgUrl(im.url, 256, true) : null;
}

module.exports = { CivitAI, sanitizeHtml, imgUrl, normModel, fmtBytes, TYPE_DIRS, DOMAINS, MARGIN };
