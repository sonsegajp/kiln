/* Kiln Extras (upscale an image), PNG Info (Kiln / A1111 / ComfyUI metadata) and Settings tabs. */
(() => {
  'use strict';
  const K = window.Kiln;
  const { $, el, api, toast, fmtSec, fmtBytes } = K;
  const GEN = () => window.KilnGen;

  function dropZone(zone, fn) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('over'));
    zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('over'); const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) fn(f, f.name); });
    zone.addEventListener('paste', (e) => { const f = [...(e.clipboardData && e.clipboardData.files || [])][0]; if (f) fn(f, f.name || 'pasted.png'); });
  }
  const pick = (btn, input, fn) => { btn.addEventListener('click', () => input.click()); input.addEventListener('change', () => { const f = input.files[0]; if (f) fn(f, f.name); input.value = ''; }); };

  // ================================================================= Extras
  const EX = { blob: null, name: '', job: null, scale: 2 };
  async function setExtrasImage(src, name) {
    try {
      EX.blob = src instanceof Blob ? src : await (await fetch(src)).blob();
      EX.name = name || 'image.png';
      const url = URL.createObjectURL(EX.blob);
      const img = $('exImg');
      img.onload = () => { $('exNote').textContent = `${EX.name} · ${img.naturalWidth}×${img.naturalHeight} → ${Math.round(img.naturalWidth * EX.scale)}×${Math.round(img.naturalHeight * EX.scale)}`; };
      img.src = url;
      img.hidden = false;
      $('exHint').hidden = true;
    } catch (e) { toast('Could not read that image: ' + e.message, 'err'); }
  }
  pick($('exPick'), $('exFile'), setExtrasImage);
  dropZone($('exDrop'), setExtrasImage);
  function renderUpscalers() {
    const sel = $('exUpscaler');
    const list = (K.S.models && K.S.models.upscalers) || [];
    const cur = sel.value;
    sel.textContent = '';
    for (const u of list) { const o = el('option', '', u.replace(/\.[a-z]+$/i, '')); o.value = u; sel.appendChild(o); }
    const o = el('option', '', 'Lanczos (no model)'); o.value = ''; sel.appendChild(o);
    sel.value = [...sel.options].some(x => x.value === cur) ? cur : (list[0] || '');
  }
  K.on('models', renderUpscalers);
  // "Resize" slider (its own state, not part of the txt2img settings)
  (() => {
    const box = document.querySelector('.a-slider[data-key="ex.scale"]');
    const head = el('div', 'a-sl-head'), lab = el('label', '', box.dataset.label), num = el('input'), range = el('input');
    num.type = 'number'; num.min = '1'; num.max = '8'; num.step = '0.25'; num.value = EX.scale; num.setAttribute('aria-label', 'Resize');
    range.type = 'range'; range.min = '1'; range.max = '8'; range.step = '0.25'; range.value = EX.scale; range.id = 'exScale'; lab.htmlFor = 'exScale';
    head.append(lab, num); box.append(head, range);
    const set = (v) => { v = Math.min(8, Math.max(1, Math.round(Number(v) * 4) / 4 || 2)); EX.scale = v; num.value = v; range.value = v; const img = $('exImg'); if (img.naturalWidth) $('exNote').textContent = `${EX.name} · ${img.naturalWidth}×${img.naturalHeight} → ${Math.round(img.naturalWidth * v)}×${Math.round(img.naturalHeight * v)}`; };
    range.addEventListener('input', () => set(range.value));
    num.addEventListener('change', () => set(num.value));
  })();
  async function toPngBlob(blob) {
    if (blob.type === 'image/png') return blob;
    const bmp = await createImageBitmap(blob);
    const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
    c.getContext('2d').drawImage(bmp, 0, 0);
    return new Promise(r => c.toBlob(r, 'image/png'));
  }
  async function extrasGo() {
    if (!EX.blob) { toast('Add an image first', 'err'); return; }
    const btn = $('exGo');
    btn.disabled = true;
    try {
      const png = await toPngBlob(EX.blob);
      const name = 'kiln_extras_' + EX.name.replace(/\.[a-z]+$/i, '').replace(/[^\w.-]+/g, '_').slice(0, 60) + '.png';
      const up = await fetch('/api/upload/image?name=' + encodeURIComponent(name), { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: png });
      const u = await up.json();
      if (!up.ok) throw new Error(u.error || 'upload failed');
      const model = $('exUpscaler').value;
      const g = { 1: { class_type: 'LoadImage', inputs: { image: u.name } } };
      let img = ['1', 0], factor = EX.scale;
      if (model) {
        g[2] = { class_type: 'UpscaleModelLoader', inputs: { model_name: model } };
        g[3] = { class_type: 'ImageUpscaleWithModel', inputs: { upscale_model: ['2', 0], image: img } };
        img = ['3', 0];
        factor = EX.scale / 4;   // the model is 4x
      }
      if (Math.abs(factor - 1) > 1e-3) { g[4] = { class_type: 'ImageScaleBy', inputs: { image: img, upscale_method: 'lanczos', scale_by: Math.round(factor * 10000) / 10000 } }; img = ['4', 0]; }
      g[5] = { class_type: 'SaveImage', inputs: { images: img, filename_prefix: 'Kiln_extras' } };
      const r = await api('/api/graph', { method: 'POST', body: { graph: g, ui: { kiln_simple: 'extras' } } });
      EX.job = r.id;
      EX.t0 = Date.now();
      $('exOut').textContent = '';
      $('exOut').appendChild(el('div', 'a-drop-hint', 'Upscaling…'));
      $('exInfo').textContent = '';
    } catch (e) { toast('Extras failed: ' + e.message, 'err', 6000); }
    finally { btn.disabled = false; }
  }
  $('exGo').addEventListener('click', extrasGo);
  K.on('generate', (d) => { if (d.tab === 'extras') extrasGo(); });
  $('exToI2i').addEventListener('click', () => {
    const a = $('exSave');
    if (!a.href || a.hidden) return;
    K.setTab('img2img').then(() => GEN().setI2iImage(a.href.replace(/\?dl=1$/, ''), a.download, { mode: 'img2img', fitSize: true }));
  });
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    if (m.type === 'gimage' && m.id === EX.job && m.image) {
      const out = $('exOut');
      out.textContent = '';
      const img = el('img'); img.src = m.image.view || m.image.url; img.alt = '';
      out.appendChild(img);
      const a = $('exSave'); a.href = m.image.url + '?dl=1'; a.download = m.image.file.split('/').pop(); a.hidden = false;
      $('exToI2i').hidden = false;
      $('exInfo').textContent = `${m.image.w}×${m.image.h} · ${m.image.file}`;
    } else if (m.type === 'job' && m.job.id === EX.job) {
      if (m.job.status === 'done') $('exInfo').textContent += `${$('exInfo').textContent ? ' · ' : ''}Time taken: ${fmtSec(m.job.total_ms)} s`;
      else if (m.job.status === 'error') { $('exOut').textContent = ''; $('exOut').appendChild(el('div', 'a-drop-hint', 'Failed: ' + m.job.error)); }
    }
  });

  // ================================================================= PNG Info
  const PI = { file: null, meta: null, a1: null, kiln: null, comfy: null };
  async function readChunks(buf) {
    const b = new Uint8Array(buf);
    const sig = [137, 80, 78, 71, 13, 10, 26, 10];
    if (!sig.every((v, i) => b[i] === v)) return null;
    const dv = new DataView(buf);
    const out = {};
    const latin1 = (a) => { let s = ''; for (let i = 0; i < a.length; i++) s += String.fromCharCode(a[i]); return s; };
    const inflate = async (a) => new Uint8Array(await new Response(new Blob([a]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
    for (let p = 8; p + 8 <= b.length;) {
      const len = dv.getUint32(p), type = latin1(b.subarray(p + 4, p + 8)), d = b.subarray(p + 8, p + 8 + len);
      if (type === 'tEXt') { const z = d.indexOf(0); if (z > 0) out[latin1(d.subarray(0, z))] = latin1(d.subarray(z + 1)); }
      else if (type === 'zTXt') { const z = d.indexOf(0); if (z > 0) { try { out[latin1(d.subarray(0, z))] = latin1(await inflate(d.subarray(z + 2))); } catch (_) { } } }
      else if (type === 'iTXt') {
        const z = d.indexOf(0);
        if (z > 0) {
          const comp = d[z + 1];
          let q = d.indexOf(0, z + 3); q = d.indexOf(0, q + 1);
          let t = d.subarray(q + 1);
          try { if (comp) t = await inflate(t); out[latin1(d.subarray(0, z))] = new TextDecoder().decode(t); } catch (_) { }
        }
      }
      if (type === 'IEND') break;
      p += 12 + len;
    }
    return out;
  }
  async function pngInfo(file, name) {
    PI.file = file;
    const url = URL.createObjectURL(file);
    $('piImg').src = url; $('piImg').hidden = false; $('piHint').hidden = true;
    const out = $('piOut');
    out.textContent = '';
    let t = null;
    try { t = await readChunks(await file.arrayBuffer()); } catch (_) { }
    PI.kiln = null; PI.a1 = null; PI.comfy = null;
    if (!t) { out.appendChild(el('div', 'a-hint', 'Not a PNG (only PNG files carry generation parameters).')); $('piBtns').hidden = false; $('piToTxt').hidden = true; return; }
    if (t.kiln) { try { PI.kiln = JSON.parse(t.kiln); } catch (_) { } }
    if (t.parameters) PI.a1 = t.parameters;
    if (t.workflow || t.prompt) PI.comfy = { workflow: t.workflow, prompt: t.prompt };
    const add = (title, node) => { const s = el('div'); s.append(el('h4', '', title), node); out.appendChild(s); };
    if (PI.a1) add('parameters', el('div', 'a-info', PI.a1));
    if (PI.kiln && !PI.a1) add('Kiln', el('div', 'a-info', GEN().infotext(PI.kiln, {})));
    if (PI.kiln) {
      const dl = el('dl', 'a-kv');
      const kv = (k, v) => { if (v === undefined || v === null || v === '') return; dl.append(el('dt', '', k), el('dd', '', typeof v === 'object' ? JSON.stringify(v) : String(v))); };
      kv('kind', PI.kiln.kind || 'txt2img'); kv('model', PI.kiln.model); kv('engine', PI.kiln.engine); kv('output', PI.kiln.out_w && `${PI.kiln.out_w}×${PI.kiln.out_h}`);
      kv('time', PI.kiln.total_ms && fmtSec(PI.kiln.total_ms) + ' s'); kv('created', PI.kiln.created);
      add('Kiln metadata', dl);
    }
    if (PI.comfy) {
      let n = 0;
      try { const w = JSON.parse(PI.comfy.workflow || PI.comfy.prompt); n = w.nodes ? w.nodes.length : Object.keys(w).length; } catch (_) { }
      add('ComfyUI workflow', el('div', 'a-hint', `${n} nodes${PI.comfy.workflow ? ' (editor layout included)' : ''}. Open it in the Nodes tab.`));
    }
    const other = Object.keys(t).filter(k => !['kiln', 'parameters', 'prompt', 'workflow'].includes(k));
    for (const k of other) add(k, el('div', 'a-info', String(t[k]).slice(0, 4000)));
    if (!Object.keys(t).length) out.appendChild(el('div', 'a-hint', 'No text metadata in this PNG.'));
    $('piBtns').hidden = false;
    $('piToTxt').hidden = !(PI.kiln || PI.a1);
    $('piToI2i').hidden = false;
    $('piToNodes').hidden = !PI.comfy;
    void name;
  }
  pick($('piPick'), $('piFile'), pngInfo);
  dropZone($('piDrop'), pngInfo);
  function piParams() {
    if (PI.kiln && (PI.kiln.kind !== 'graph' || PI.kiln.prompt)) return PI.kiln;
    if (PI.a1) return GEN().a1111ToParams(PI.a1).params;
    return null;
  }
  $('piToTxt').addEventListener('click', async () => {
    const p = piParams();
    if (!p) return;
    GEN().applyParams(p);
    await K.setTab('txt2img');
    toast('Parameters sent to txt2img', 'ok', 1500);
  });
  const toI2i = async (mode) => {
    const p = piParams();
    if (p) GEN().applyParams(p);
    await K.setTab('img2img');
    GEN().setI2iImage(PI.file, PI.file.name || 'image.png', { mode, fitSize: true });
  };
  $('piToI2i').addEventListener('click', () => toI2i('img2img'));
  $('piToInpaint').addEventListener('click', () => toI2i('inpaint'));
  $('piToExtras').addEventListener('click', async () => { await K.setTab('extras'); setExtrasImage(PI.file, PI.file.name || 'image.png'); });
  $('piToNodes').addEventListener('click', async () => { await K.setTab('nodes'); window.KilnNodes.importFile(PI.file); });

  // ================================================================= Settings
  const ST = { built: false };
  function field(grid, label, node, id) { const l = el('label', '', label); if (id) l.htmlFor = id; grid.append(l, node); }
  function renderSettings() {
    const s = K.S.settings;
    const body = $('settingsBody');
    if (!s) { body.textContent = 'Loading…'; return; }
    const cv = s.civitai || {};
    const manage = !!s.manage;
    body.textContent = '';
    // ---- CivitAI
    const b1 = el('section', 'a-block');
    b1.appendChild(el('h3', '', 'CivitAI'));
    if (!manage) b1.appendChild(el('div', 'a-note', 'These settings can only be changed on the PC running Kiln (the API key never leaves the server). Browsing works from here.'));
    const g1 = el('div', 'a-set-grid');
    const keyRow = el('div', 'a-inline');
    keyRow.appendChild(el('span', 'a-keymask', cv.has_key ? `${cv.key_hint} (set)` : 'not set'));
    const key = el('input'); key.type = 'password'; key.id = 'setKey'; key.placeholder = cv.has_key ? 'Paste a new key to replace it' : 'Paste your API key (civitai.com → Account settings → API keys)'; key.autocomplete = 'off'; key.disabled = !manage;
    const setK = el('button', 'a-btn sm primary', 'Set'); setK.type = 'button'; setK.disabled = !manage;
    const clrK = el('button', 'a-btn sm danger', 'Clear'); clrK.type = 'button'; clrK.disabled = !manage || !cv.has_key;
    keyRow.append(key, setK, clrK);
    field(g1, 'API key', keyRow, 'setKey');
    const dom = el('select'); dom.id = 'setDomain'; dom.disabled = !manage;
    for (const d of cv.domains || ['civitai.red', 'civitai.com']) { const o = el('option', '', d); o.value = d; dom.appendChild(o); }
    dom.value = cv.domain || 'civitai.red';
    field(g1, 'Site', dom, 'setDomain');
    const nsfwL = el('label', 'a-check'); const nsfw = el('input'); nsfw.type = 'checkbox'; nsfw.id = 'setNsfw'; nsfw.checked = !!cv.nsfw; nsfw.disabled = !manage;
    nsfwL.append(nsfw, el('span', '', 'Show NSFW models and previews by default'));
    field(g1, 'NSFW', nsfwL);
    const folders = {};
    for (const [k, label] of [['loras', 'LoRA download folder'], ['checkpoints', 'Checkpoint download folder'], ['upscale', 'Upscaler folder'], ['embeddings', 'Embedding folder']]) {
      const w = el('div', 'a-inline');
      const inp = el('input'); inp.type = 'text'; inp.id = 'setF_' + k; inp.value = (cv.folders && cv.folders[k]) || ''; inp.placeholder = `default: models\\${k}`; inp.disabled = !manage; inp.spellcheck = false;
      folders[k] = inp;
      w.append(inp, el('span', 'a-hint', (cv.resolved && cv.resolved[k]) || ''));
      field(g1, label, w, inp.id);
    }
    const saveCv = el('button', 'a-btn primary', 'Save CivitAI settings'); saveCv.type = 'button'; saveCv.disabled = !manage;
    b1.append(g1, el('div', 'a-hint', 'Folders: a path inside models (e.g. loras\\civitai) or an absolute folder on another drive (e.g. D:\\models\\checkpoints) for big checkpoints. Kiln also lists models it finds there.'), saveCv);
    const post = async (bodyObj, ok) => {
      try { const r = await api('/api/civitai/settings', { method: 'POST', body: bodyObj }); K.S.settings.civitai = r; toast(ok, 'ok', 1800); renderSettings(); K.loadModels(); }
      catch (e) { toast(e.message, 'err', 6000); }
    };
    setK.addEventListener('click', () => { if (!key.value.trim()) { key.focus(); return; } post({ api_key: key.value.trim() }, 'API key saved (stored on the server only)'); });
    clrK.addEventListener('click', () => { if (confirm('Remove the CivitAI API key from this Kiln?')) post({ clear_key: true }, 'API key removed'); });
    saveCv.addEventListener('click', () => post({ domain: dom.value, nsfw: nsfw.checked, folders: Object.fromEntries(Object.entries(folders).map(([k, i]) => [k, i.value.trim()])) }, 'CivitAI settings saved'));
    body.appendChild(b1);

    // ---- UI defaults + presets
    const b2 = el('section', 'a-block');
    b2.appendChild(el('h3', '', 'Defaults'));
    const g2 = el('div', 'a-set-grid');
    const def = s.ui && s.ui.defaults;
    const saveDef = el('button', 'a-btn', 'Use the current txt2img settings as defaults'); saveDef.type = 'button';
    const loadDef = el('button', 'a-btn', 'Reset txt2img to the defaults'); loadDef.type = 'button'; loadDef.disabled = !def;
    const clrDef = el('button', 'a-btn danger', 'Forget the defaults'); clrDef.type = 'button'; clrDef.disabled = !def;
    const w2 = el('div', 'a-inline'); w2.append(saveDef, loadDef, clrDef);
    field(g2, 'UI defaults', w2);
    b2.append(g2, el('div', 'a-hint', def ? `Saved: ${def.sampler || '?'} · ${def.steps} steps · CFG ${def.cfg} · ${def.width}×${def.height}${def.slots && def.slots.some(x => x.on) ? ' · LoRA slots' : ''}. New devices and "Reset" start from these (the prompt is not included).` : 'No defaults saved: new devices start from Kiln\'s built-in settings.'));
    const KEYS = ['negative', 'sampler', 'scheduler', 'steps', 'width', 'height', 'batchCount', 'batchSize', 'cfg', 'cfgCutoff', 'stepCache', 'shift', 'nag', 'hires', 'face', 'upscale', 'slots', 'denoise'];
    saveDef.addEventListener('click', async () => {
      const G = GEN().G;
      const d = JSON.parse(JSON.stringify(Object.fromEntries(KEYS.map(k => [k, G[k]]))));
      try { const r = await api('/api/settings', { method: 'POST', body: { defaults: d } }); K.S.settings.ui = r.ui; toast('Defaults saved', 'ok'); renderSettings(); } catch (e) { toast(e.message, 'err'); }
    });
    loadDef.addEventListener('click', () => { const d = K.S.settings.ui.defaults; if (!d) return; GEN().setSettings(d); toast('txt2img reset to the defaults', 'ok'); K.setTab('txt2img'); });
    clrDef.addEventListener('click', async () => { try { const r = await api('/api/settings', { method: 'POST', body: { defaults: null } }); K.S.settings.ui = r.ui; renderSettings(); } catch (e) { toast(e.message, 'err'); } });
    const ps = (s.ui && s.ui.presets) || {};
    const pl = el('div', 'a-inline');
    if (!Object.keys(ps).length) pl.appendChild(el('span', 'a-hint', 'None yet: use 💾 next to the Generate button.'));
    for (const n of Object.keys(ps).sort()) {
      const b = el('button', 'a-btn sm', n + ' ✕'); b.type = 'button'; b.title = 'Delete this preset';
      b.addEventListener('click', async () => { if (!confirm(`Delete preset "${n}"?`)) return; try { const r = await api('/api/settings', { method: 'POST', body: { presets: { [n]: null } } }); K.S.settings.ui = r.ui; renderSettings(); K.emit('settings', K.S.settings); } catch (e) { toast(e.message, 'err'); } });
      pl.appendChild(b);
    }
    const g3 = el('div', 'a-set-grid');
    field(g3, 'Presets', pl);
    b2.appendChild(g3);
    body.appendChild(b2);

    // ---- about
    const b3 = el('section', 'a-block');
    b3.appendChild(el('h3', '', 'About'));
    const srv = K.S.server || {};
    const dl = el('dl', 'a-kv');
    const kv = (k, v) => { if (v) dl.append(el('dt', '', k), el('dd', '', String(v))); };
    kv('Kiln', srv.version); kv('Engine', srv.mock ? 'mock (kiln-engine.exe not found)' : srv.engine_exe); kv('Models', srv.models_dir); kv('Outputs', srv.outputs_dir);
    kv('Extensions', srv.extensions ? `${srv.extensions.packs} packs, ${srv.extensions.nodes} nodes` : '');
    kv('Checkpoint', K.ckptInfo() ? `${K.ckptInfo().name} (${K.ckptInfo().base || K.FAMILY[K.ckptInfo().family]})` : '');
    b3.appendChild(dl);
    body.appendChild(b3);
  }
  K.on('settings', () => { if (K.S.tab === 'settings') renderSettings(); });
  K.on('tab', (t) => { if (t.tab === 'settings') { K.loadSettings(); renderSettings(); } });

  window.KilnTabs = { setExtrasImage, pngInfo, renderSettings };
})();
