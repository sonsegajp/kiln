'use strict';
// Kiln's model library: enriched model lists (family / base model / sidecar trigger words + preview),
// a local index for "downloaded" / "update available" on CivitAI cards, background identification
// (SHA256 -> CivitAI by-hash), UI settings (defaults + presets), and the HTTP routes for all of it.

const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const { spawn } = require('child_process');
const SF = require('./sfinfo');
const { crc32 } = require('./png');

function createLibrary(ctx) {
  const { MODELS, OUTPUTS, CONFIG, log, broadcast, sendJSON, readJSON, isLocal, manageDenied, listModels, civitai, hasher } = ctx;
  const uiFile = path.join(CONFIG, 'ui.json');
  let ui = { defaults: null, presets: {} };
  try { Object.assign(ui, JSON.parse(fs.readFileSync(uiFile, 'utf8'))); } catch (_) { }
  const identifying = {};   // rel file -> 'hashing' | 'looking up' | 'done' | 'error: …'

  const rel = (abs) => {
    const r = path.relative(MODELS, abs);
    return r.startsWith('..') || path.isAbsolute(r) ? abs.split(path.sep).join('/') : r.split(path.sep).join('/');
  };

  // base model for a file: sidecar > identified hash > safetensors metadata > family default
  function describe(abs) {
    const info = SF.inspect(abs);
    const side = SF.readSidecar(abs);
    const sha = (side && side.sha256) || hasher.cached(abs);
    const ids = !side && sha && civitai.ids[sha] ? civitai.ids[sha] : null;
    let base = info.base, baseSource = info.baseSource;
    if (side && side.baseModel) { base = side.baseModel; baseSource = 'civitai'; }
    else if (ids && ids.found && ids.baseModel) { base = ids.baseModel; baseSource = 'hash'; }
    else if (!base && info.family === 'sdxl' && ids && !ids.found) { base = 'SDXL 1.0'; baseSource = 'family'; }
    let family = info.family;
    if (family === 'unknown' && base) family = SF.familyOfBase(base) || family;
    const cv = side ? { modelId: side.modelId, versionId: side.versionId, name: side.name, versionName: side.versionName, url: side.url }
      : ids && ids.found ? { modelId: ids.modelId, versionId: ids.versionId, name: ids.name, versionName: ids.versionName, url: `https://${civitai.cfg.domain}/models/${ids.modelId}?modelVersionId=${ids.versionId}` } : null;
    const words = (side && side.trainedWords) || (ids && ids.found && ids.trainedWords) || [];
    const pv = SF.previewPath(abs);
    return {
      role: info.role, family, base, baseSource, title: info.title, sha256: sha || null, civitai: cv, words: words.slice(0, 30),
      preview: pv ? '/api/models/preview?file=' + encodeURIComponent(rel(abs)) : (ids && ids.found && ids.image) || null,
      nsfw: !!((side && side.nsfw) || (ids && ids.nsfw)), identifying: identifying[rel(abs)] || null,
    };
  }

  // every model file Kiln knows (for previews and the local index)
  function allFiles() {
    const m = listModels();
    const out = [];
    for (const l of m.loras) out.push({ abs: l.path, kind: 'lora' });
    for (const c of m.checkpoints) out.push({ abs: c.path, kind: c.kind });
    for (const [k, d] of [['upscale', civitai.folder('upscale')], ['embeddings', civitai.folder('embeddings')]]) {
      let ents = [];
      try { ents = fs.readdirSync(d); } catch (_) { }
      for (const f of ents) if (/\.safetensors$/i.test(f)) out.push({ abs: path.join(d, f), kind: k });
    }
    return out;
  }

  function modelsPayload(features) {
    const m = listModels();
    const loras = m.loras.map(l => Object.assign({ file: l.file, name: l.name, size: l.size, kind: 'lora' }, describe(l.path)));
    const sd = [];
    for (const c of m.checkpoints) {
      if (!['diffusion', 'checkpoint'].includes(c.kind)) continue;
      const d = describe(c.path);
      if (d.role !== 'model') continue;
      sd.push(Object.assign({ file: c.file, name: c.name, size: c.size, kind: c.kind, runnable: ctx.canRun ? ctx.canRun(d.family) : d.family === 'anima' }, d));
    }
    return {
      models_dir: MODELS,
      checkpoints: m.checkpoints.map(({ file, name, size, kind }) => ({ file, name, size, kind })),
      loras, sd_models: sd,
      upscalers: fs.existsSync(civitai.folder('upscale')) ? fs.readdirSync(civitai.folder('upscale')).filter(f => /\.safetensors$/i.test(f)) : [],
      features, bases: SF.KNOWN_BASES, family_bases: SF.FAMILY_BASES,
    };
  }

  // versionId / sha256 / modelId -> local file (from sidecars and identified hashes)
  let idx = null, idxAt = 0;
  function localIndex() {
    if (idx && Date.now() - idxAt < 3000) return idx;
    const byVersion = new Map(), bySha = new Map(), byModel = new Map();
    for (const f of allFiles()) {
      const side = SF.readSidecar(f.abs);
      const sha = (side && side.sha256) || hasher.cached(f.abs);
      const ids = side ? { modelId: side.modelId, versionId: side.versionId } : sha && civitai.ids[sha] && civitai.ids[sha].found ? civitai.ids[sha] : null;
      const r = rel(f.abs);
      if (sha) bySha.set(String(sha).toUpperCase(), r);
      if (ids && ids.versionId) {
        byVersion.set(Number(ids.versionId), r);
        if (!byModel.has(Number(ids.modelId))) byModel.set(Number(ids.modelId), []);
        byModel.get(Number(ids.modelId)).push({ versionId: Number(ids.versionId), file: r });
      }
    }
    idx = { byVersion, bySha, byModel };
    idxAt = Date.now();
    return idx;
  }
  function annotate(item) {
    const I = localIndex();
    for (const v of item.versions) {
      v.local = I.byVersion.get(Number(v.id)) || v.files.map(f => f.sha256 && I.bySha.get(f.sha256)).find(Boolean) || null;
    }
    const mine = I.byModel.get(Number(item.id)) || [];
    const localVersions = new Set([...mine.map(x => x.versionId), ...item.versions.filter(v => v.local).map(v => v.id)]);
    const newest = item.versions[0];
    item.local = {
      downloaded: localVersions.size > 0,
      files: [...new Set([...mine.map(x => x.file), ...item.versions.map(v => v.local).filter(Boolean)])],
      update: localVersions.size > 0 && !!newest && !localVersions.has(newest.id),
    };
    for (const d of civitai.downloads) if (['queued', 'downloading', 'verifying'].includes(d.status) && d.modelId === item.id) item.local.downloading = d.id;
    return item;
  }

  async function identify(abs) {
    const r = rel(abs);
    if (identifying[r] && !/^(done|error)/.test(identifying[r])) return;
    const tell = (state, extra) => { identifying[r] = state; broadcast(Object.assign({ type: 'identify', file: r, state }, extra || {})); };
    try {
      tell('hashing');
      const sha = await hasher.hash(abs);
      if (!sha) throw new Error('could not read the file');
      tell('looking up');
      const rec = await civitai.byHash(sha);
      idx = null;
      tell('done', { found: !!rec, base: rec ? rec.baseModel : null, name: rec ? rec.name : null });
    } catch (e) {
      tell('error: ' + e.message);
    }
  }

  function saveUi() { fs.mkdirSync(CONFIG, { recursive: true }); fs.writeFileSync(uiFile, JSON.stringify(ui, null, 1)); }
  // same-origin JSON only (LAN devices allowed): UI settings, identification
  function jsonDenied(req) {
    const origin = req.headers.origin;
    if (origin !== undefined) {
      let host = null;
      try { host = new URL(origin).host; } catch (_) { }
      if (host !== req.headers.host) return 'cross-origin request refused';
    }
    if (req.method === 'POST' && !/^application\/json\b/i.test(req.headers['content-type'] || '')) return 'send JSON (Content-Type: application/json)';
    return null;
  }
  const findModel = (r) => allFiles().find(f => rel(f.abs) === r);

  // zip (stored, no compression: PNGs are already compressed)
  async function zipFiles(files) {
    const parts = [], central = [];
    let offset = 0;
    for (const f of files) {
      const data = await fsp.readFile(f.abs);
      const name = Buffer.from(f.name, 'utf8');
      const crc = (crc32(data) ^ 0xffffffff) >>> 0;
      const lh = Buffer.alloc(30);
      lh.writeUInt32LE(0x04034b50, 0); lh.writeUInt16LE(20, 4); lh.writeUInt16LE(0x0800, 6); lh.writeUInt16LE(0, 8);
      lh.writeUInt32LE(0, 10); lh.writeUInt32LE(crc, 14); lh.writeUInt32LE(data.length, 18); lh.writeUInt32LE(data.length, 22);
      lh.writeUInt16LE(name.length, 26); lh.writeUInt16LE(0, 28);
      parts.push(lh, name, data);
      const ch = Buffer.alloc(46);
      ch.writeUInt32LE(0x02014b50, 0); ch.writeUInt16LE(20, 4); ch.writeUInt16LE(20, 6); ch.writeUInt16LE(0x0800, 8); ch.writeUInt16LE(0, 10);
      ch.writeUInt32LE(0, 12); ch.writeUInt32LE(crc, 16); ch.writeUInt32LE(data.length, 20); ch.writeUInt32LE(data.length, 24);
      ch.writeUInt16LE(name.length, 28); ch.writeUInt32LE(offset, 42);
      central.push(ch, name);
      offset += 30 + name.length + data.length;
    }
    const cd = Buffer.concat(central);
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(files.length, 8); end.writeUInt16LE(files.length, 10);
    end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(offset, 16);
    return Buffer.concat([...parts, cd, end]);
  }

  async function handle(req, res, u, p, m, features) {
    const q = u.searchParams;
    if (p === '/api/models' && m === 'GET') { sendJSON(res, 200, modelsPayload(features())); return true; }
    if (p === '/api/models/preview' && m === 'GET') {
      const f = findModel(q.get('file') || '');
      const pv = f && SF.previewPath(f.abs);
      if (!pv) { sendJSON(res, 404, { error: 'no preview' }); return true; }
      const ct = /\.png$/i.test(pv) ? 'image/png' : /\.webp$/i.test(pv) ? 'image/webp' : 'image/jpeg';
      const buf = await fsp.readFile(pv);
      res.writeHead(200, { 'Content-Type': ct, 'Content-Length': buf.length, 'Cache-Control': 'no-cache' });
      res.end(buf);
      return true;
    }
    if (p === '/api/models/identify' && m === 'POST') {
      const why = jsonDenied(req);
      if (why) { sendJSON(res, 403, { error: why }); return true; }
      const b = await readJSON(req);
      const files = [];
      if (typeof b.file === 'string') { const f = findModel(b.file); if (!f) { sendJSON(res, 404, { error: 'unknown model file' }); return true; } files.push(f); }
      if (b.loras) for (const f of allFiles()) if (f.kind === 'lora' && !SF.readSidecar(f.abs) && !(hasher.cached(f.abs) && civitai.ids[hasher.cached(f.abs)])) files.push(f);
      for (const f of files) identify(f.abs);
      sendJSON(res, 200, { queued: files.map(f => rel(f.abs)) });
      return true;
    }
    if (p === '/api/settings' && m === 'GET') { sendJSON(res, 200, { ui, civitai: civitai.publicSettings(), manage: isLocal(req) }); return true; }
    if (p === '/api/settings' && m === 'POST') {
      const why = jsonDenied(req);
      if (why) { sendJSON(res, 403, { error: why }); return true; }
      const b = await readJSON(req);
      if (b.defaults !== undefined) ui.defaults = b.defaults && typeof b.defaults === 'object' ? b.defaults : null;
      if (b.presets && typeof b.presets === 'object') {
        for (const [k, v] of Object.entries(b.presets)) {
          if (typeof k !== 'string' || !k.trim() || k.length > 60) continue;
          if (v === null) delete ui.presets[k]; else if (typeof v === 'object') ui.presets[k] = v;
        }
      }
      if (JSON.stringify(ui).length > 2e6) { sendJSON(res, 413, { error: 'settings too large' }); return true; }
      saveUi();
      broadcast({ type: 'settings' });
      sendJSON(res, 200, { ui });
      return true;
    }
    if (p === '/api/zip' && m === 'GET') {
      const list = String(q.get('files') || '').split(',').filter(Boolean).slice(0, 200);
      const files = [];
      for (const f of list) {
        if (!/^\d{4}-\d{2}-\d{2}\/[^/\\:*?"<>|]+\.png$/i.test(f)) continue;
        const abs = path.join(OUTPUTS, ...f.split('/'));
        if (fs.existsSync(abs)) files.push({ abs, name: f.split('/').pop() });
      }
      if (!files.length) { sendJSON(res, 404, { error: 'no images' }); return true; }
      const zip = await zipFiles(files);
      res.writeHead(200, { 'Content-Type': 'application/zip', 'Content-Length': zip.length, 'Content-Disposition': `attachment; filename="kiln_${files.length}_images.zip"` });
      res.end(zip);
      return true;
    }
    if (p === '/api/open-folder' && m === 'POST') {
      // opens Explorer on the PC running Kiln (only for requests from that PC)
      const why = manageDenied(req);
      if (why) { sendJSON(res, 403, { error: why }); return true; }
      const b = await readJSON(req);
      let target = OUTPUTS;
      if (typeof b.file === 'string' && /^\d{4}-\d{2}-\d{2}\/[^/\\:*?"<>|]+\.png$/i.test(b.file)) target = path.join(OUTPUTS, ...b.file.split('/'));
      if (!fs.existsSync(target)) target = OUTPUTS;
      if (process.platform !== 'win32') { sendJSON(res, 501, { error: 'only on Windows' }); return true; }
      const args = target === OUTPUTS ? [OUTPUTS] : ['/select,', target];
      spawn('explorer.exe', args, { detached: true, stdio: 'ignore' }).unref();
      sendJSON(res, 200, { opened: target });
      return true;
    }
    if (!p.startsWith('/api/civitai/')) return false;
    const rest = p.slice('/api/civitai/'.length);
    try {
      if (rest === 'settings' && m === 'GET') { sendJSON(res, 200, Object.assign(civitai.publicSettings(), { manage: isLocal(req) })); return true; }
      if (rest === 'search' && m === 'GET') {
        const r = await civitai.search({
          query: q.get('query') || '', tag: q.get('tag') || '', username: q.get('username') || '',
          types: q.getAll('types'), baseModels: q.getAll('baseModels'), sort: q.get('sort') || 'Highest Rated',
          period: q.get('period') || 'AllTime', nsfw: q.get('nsfw') === 'true', cursor: q.get('cursor') || '', limit: q.get('limit') || 24,
        });
        r.items.forEach(annotate);
        sendJSON(res, 200, r);
        return true;
      }
      let mm = rest.match(/^model\/(\d{1,12})$/);
      if (mm && m === 'GET') { sendJSON(res, 200, annotate(await civitai.model(mm[1]))); return true; }
      if (rest === 'downloads' && m === 'GET') { sendJSON(res, 200, { downloads: civitai.publicDownloads(), hasher: hasher.status() }); return true; }
      // everything below writes to disk / uses the key: this PC only
      const why = manageDenied(req);
      if (why) { sendJSON(res, 403, { error: why }); return true; }
      const b = m === 'POST' ? await readJSON(req) : {};
      if (rest === 'settings' && m === 'POST') { sendJSON(res, 200, Object.assign(civitai.setSettings(b), { manage: true })); broadcast({ type: 'settings' }); return true; }
      if (rest === 'download' && m === 'POST') {
        const d = await civitai.enqueue(b.versionId, b.fileId);
        sendJSON(res, 200, { download: civitai.publicDownloads().find(x => x.id === d.id) });
        return true;
      }
      if (rest === 'downloads/clear' && m === 'POST') { civitai.clearFinished(); sendJSON(res, 200, { downloads: civitai.publicDownloads() }); return true; }
      mm = rest.match(/^downloads\/([\w]+)\/(cancel|retry)$/);
      if (mm && m === 'POST') {
        if (mm[2] === 'cancel') civitai.cancel(mm[1]); else civitai.retry(mm[1]);
        sendJSON(res, 200, { downloads: civitai.publicDownloads() });
        return true;
      }
      sendJSON(res, 404, { error: 'no such endpoint' });
    } catch (e) {
      sendJSON(res, Number.isInteger(e.code) && e.code >= 400 && e.code < 600 ? e.code : 500, { error: e.message });
    }
    return true;
  }

  return { handle, modelsPayload, describe, localIndex, annotate, identify, get ui() { return ui; } };
}

module.exports = { createLibrary };
