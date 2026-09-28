/* Kiln web UI core: helpers, tabs (txt2img | img2img | Extras | PNG Info | Nodes | Settings), server
   events, engine / queue indicators, the "Stable Diffusion checkpoint" picker. Plain JS, no build.
   The tabs live in gen.js (txt2img + img2img), xnet.js (extra networks + CivitAI), tabs.js
   (Extras, PNG Info, Settings), nodes.js (Nodes) and extensions.js (packs panel). */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  function debounce(fn, ms) { let t = 0; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function fmtSec(ms) {
    if (ms == null || !isFinite(ms)) return '–';
    const s = ms / 1000;
    return s < 10 ? s.toFixed(2) : s < 100 ? s.toFixed(1) : Math.round(s).toString();
  }
  function fmtMs(ms) {
    if (ms == null || !isFinite(ms)) return '–';
    return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
  }
  const fmtBytes = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : Math.max(0, Math.round((n || 0) / 1e3)) + ' KB');
  const fmtCount = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k' : String(n || 0));
  const lsGet = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch (_) { return null; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* private mode */ } };
  async function api(path, opts = {}) {
    const r = await fetch(path, {
      method: opts.method || 'GET',
      headers: opts.body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
    let data = null;
    try { data = await r.json(); } catch (_) { /* empty */ }
    if (!r.ok) { const err = new Error((data && data.error) || `${r.status} ${r.statusText}`); err.data = data; err.status = r.status; throw err; }
    return data;
  }
  function toast(msg, kind = '', ms = 3200) {
    const t = el('div', 'toast ' + kind, msg);
    $('toasts').appendChild(t);
    setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 300); }, ms);
  }
  async function copyText(t, msg) {
    try { await navigator.clipboard.writeText(t); toast(msg || 'Copied', 'ok', 1500); }
    catch (_) { toast(t); }
  }
  const isMobile = () => window.matchMedia('(max-width: 900px)').matches;

  // ---------------------------------------------------------------- shared state + events
  const LS_TAB = 'kiln.tab.v1', LS_MODE_OLD = 'kiln.mode.v1', LS_TIMES = 'kiln.times.v1', LS_CKPT = 'kiln.ckpt.v1';
  const S = {
    server: null, engine: null, connected: false, running: null, queued: [], jobs: new Map(),
    models: null, ckpt: lsGet(LS_CKPT) || '', settings: null, tab: null,
    times: (lsGet(LS_TIMES) || []).filter(n => typeof n === 'number').slice(-10),
  };
  const bus = new EventTarget();
  const on = (name, fn) => bus.addEventListener(name, (e) => fn(e.detail));
  const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));

  // ---------------------------------------------------------------- tabs
  const TABS = ['txt2img', 'img2img', 'extras', 'pnginfo', 'nodes', 'settings'];
  async function setTab(tab) {
    if (tab === 'simple') tab = 'txt2img';
    if (!TABS.includes(tab)) tab = 'txt2img';
    const prev = S.tab;
    S.tab = tab;
    document.body.classList.toggle('mode-nodes', tab === 'nodes');
    for (const t of TABS) document.body.classList.toggle('tab-' + t, t === tab);
    for (const b of $('modeSwitch').querySelectorAll('button')) b.setAttribute('aria-selected', String(b.dataset.tab === tab));
    for (const p of document.querySelectorAll('[data-tab-panel]')) p.hidden = !p.dataset.tabPanel.split(' ').includes(tab);
    $('nodesView').hidden = tab !== 'nodes';
    lsSet(LS_TAB, tab);
    lsSet(LS_MODE_OLD, tab === 'nodes' ? 'nodes' : 'simple');
    if (tab === 'nodes') { if (window.KilnNodes) await window.KilnNodes.activate(); }
    else if (prev === 'nodes' && window.KilnNodes) window.KilnNodes.deactivate();
    emit('tab', { tab, prev });
  }
  const setMode = (m) => setTab(m === 'simple' ? 'txt2img' : m);
  $('modeSwitch').addEventListener('click', (e) => { const b = e.target.closest('button[data-tab]'); if (b) setTab(b.dataset.tab); });

  // ---------------------------------------------------------------- engine + queue indicators
  function renderEngine() {
    const pill = $('enginePill'), label = $('engineLabel');
    if (!S.connected) { pill.dataset.state = 'offline'; label.textContent = 'Server offline · retrying'; return; }
    const e = S.engine || {};
    let state = e.state, text;
    if (state === 'ready' && e.gpu_wait) { state = 'parked'; text = `Waiting for ${e.gpu_wait} to finish with the GPU`; }
    else if (state === 'ready') {
      if (S.running) { state = 'busy'; text = e.mock ? 'Rendering (mock)' : 'Rendering'; }
      else { state = e.mock ? 'mock' : 'ready'; text = e.mock ? 'Mock engine' : 'Engine ready'; }
    } else if (state === 'starting') {
      text = e.loading ? `Loading ${e.loading.what || ''} ${e.loading.progress != null ? Math.round(e.loading.progress * 100) + '%' : ''}` : 'Engine starting…';
    } else if (state === 'down') text = 'Engine down · restarting';
    else if (e.parked) { state = 'parked'; text = `Parked for ${e.parked} · starts on the next render`; }
    else text = 'Engine stopped';
    pill.dataset.state = state || 'offline';
    label.textContent = text.trim();
    pill.title = text.trim();
    const banner = $('mockBanner');
    banner.hidden = !e.mock;
    document.body.classList.toggle('has-banner', !!e.mock);
    if (e.mock && S.server && S.server.engine_exe) {
      $('mockBannerText').textContent = `kiln-engine.exe wasn't found at ${S.server.engine_exe}, so images are fake gradients. The server switches to the real engine automatically once it exists.`;
    }
  }
  function stageLabel(stage, face, step, of) {
    if (stage === 'hires') return `Hires ${step}/${of}`;
    if (stage === 'face') return `Face ${face || 1} · ${step}/${of}`;
    if (stage && stage !== 'base') return `${stage} ${step}/${of}`;
    return `Step ${step}/${of}`;
  }
  // job updates from the server; step events build a finer per-step timeline in the page, which a
  // server snapshot taken earlier must not shrink
  function mergeJob(nj) {
    const prev = S.jobs.get(nj.id);
    const count = (tl) => (tl || []).reduce((n, g) => n + ((g && g.ms) || []).length, 0);
    const keepTl = prev && prev.timeline && count(prev.timeline) > count(nj.timeline) ? prev.timeline : null;
    const keepSm = prev && prev.step_ms && (!nj.step_ms || prev.step_ms.length > nj.step_ms.length) ? prev.step_ms : null;
    const j = Object.assign(prev || {}, nj);
    if (keepTl) j.timeline = keepTl;
    if (keepSm) j.step_ms = keepSm;
    S.jobs.set(j.id, j);
    return j;
  }
  function applyQueue(q) {
    S.running = q.running;
    S.queued = q.queued || [];
    for (const j of [q.running, ...S.queued]) if (j) mergeJob(j);
    renderQueue();
    renderEngine();
    emit('queue', q);
  }
  async function cancel(id) {
    try { await api('/api/cancel/' + encodeURIComponent(id), { method: 'POST' }); }
    catch (e) { if (!/no such job/.test(e.message)) toast('Cancel failed: ' + e.message, 'err'); }
  }
  function renderQueue() {
    const busy = !!S.running || S.queued.length > 0;
    const pill = $('queuePill');
    pill.classList.toggle('active', busy);
    let label = 'Idle';
    if (S.running) {
      const r = S.running;
      label = r.kind === 'graph' ? (r.step ? `Graph · step ${r.step}/${r.of}` : r.swap ? 'Loading model' : 'Graph running') : r.step ? stageLabel(r.stage, r.face, r.step, r.of) : r.swap ? 'Loading model' : 'Starting';
      if (S.queued.length) label += ` · ${S.queued.length} queued`;
    } else if (S.queued.length) label = `${S.queued.length} queued`;
    $('queueLabel').textContent = label;
    const list = $('queueList');
    list.textContent = '';
    const all = [S.running, ...S.queued].filter(Boolean);
    if (!all.length) { list.appendChild(el('li', 'queue-empty', 'Nothing queued.')); return; }
    for (const j of all) {
      const li = el('li', j.status === 'running' ? 'running' : '');
      const stTxt = j.status === 'running' ? (j.step ? (j.stage === 'hires' ? 'H ' : j.stage === 'face' ? `F${j.face || 1} ` : '') + `${j.step}/${j.of}` : 'start') : `#${j.position || ''}`;
      const p = j.params || {};
      const txt = el('span', 'q-text', j.kind === 'graph' ? `Graph · ${p.nodes} nodes — ${p.prompt || '(no prompt)'}` : `${p.width}×${p.height} · ${p.steps} st · seed ${p.seed} — ${p.prompt || '(empty prompt)'}`);
      txt.title = p.prompt || '';
      const x = el('button', 'q-x', '✕');
      x.type = 'button';
      x.title = 'Cancel';
      x.addEventListener('click', (e) => { e.stopPropagation(); cancel(j.id); });
      li.append(el('span', 'q-state', stTxt), txt, x);
      list.appendChild(li);
    }
  }
  function renderAvg() {
    const t = S.times;
    $('avgSpeedVal').textContent = t.length ? fmtSec(t.reduce((a, b) => a + b, 0) / t.length) : '–';
  }
  function recordTime(ms) {
    if (!ms) return;
    S.times.push(ms);
    S.times = S.times.slice(-10);
    lsSet(LS_TIMES, S.times);
    renderAvg();
  }
  $('queuePill').addEventListener('click', (e) => {
    e.stopPropagation();
    const pop = $('queuePop');
    pop.hidden = !pop.hidden;
    $('queuePill').setAttribute('aria-expanded', String(!pop.hidden));
  });
  $('queuePop').addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', () => { $('queuePop').hidden = true; $('queuePill').setAttribute('aria-expanded', 'false'); });
  $('queueCancelAll').addEventListener('click', () => cancel('all'));

  // ---------------------------------------------------------------- models + checkpoint picker
  const FAMILY = { anima: 'Anima', sdxl: 'SDXL', sd15: 'SD 1.5', flux: 'Flux', other: 'Other', unknown: '?' };
  function ckptInfo() {
    const list = (S.models && S.models.sd_models) || [];
    return list.find(m => m.file === S.ckpt) || list.find(m => m.runnable) || list[0] || null;
  }
  function ckptLabel(m) {
    const base = m.base || FAMILY[m.family] || '?';
    return `${m.name}  [${base}]${m.runnable ? '' : m.family === 'sdxl' ? ' · coming soon' : ' · not supported'}`;
  }
  function renderCkpt() {
    const sel = $('ckptSel');
    const list = (S.models && S.models.sd_models) || [];
    sel.textContent = '';
    if (!list.length) { const o = el('option', '', 'No checkpoints found in models/'); o.value = ''; sel.appendChild(o); return; }
    const groups = {};
    for (const m of list) (groups[m.family] = groups[m.family] || []).push(m);
    for (const [fam, ms] of Object.entries(groups)) {
      const g = el('optgroup');
      g.label = FAMILY[fam] || fam;
      for (const m of ms) {
        const o = el('option', '', ckptLabel(m));
        o.value = m.file;
        o.title = `${m.file}${m.base ? ' · ' + m.base : ''}${m.baseSource ? ' (' + m.baseSource + ')' : ''}`;
        g.appendChild(o);
      }
      sel.appendChild(g);
    }
    const cur = ckptInfo();
    sel.value = cur ? cur.file : '';
    sel.classList.toggle('sdxl', !!cur && !cur.runnable);
    sel.title = cur ? `${cur.file} · ${cur.base || FAMILY[cur.family]}${cur.runnable ? '' : cur.family === 'sdxl' ? ' · SDXL support is coming soon' : ' · not supported yet'}` : '';
  }
  $('ckptSel').addEventListener('change', () => {
    S.ckpt = $('ckptSel').value;
    lsSet(LS_CKPT, S.ckpt);
    renderCkpt();
    emit('ckpt', ckptInfo());
  });
  let modelsLoading = null;
  async function loadModels() {
    if (modelsLoading) return modelsLoading;
    modelsLoading = (async () => {
      try {
        S.models = await api('/api/models');
        renderCkpt();
        emit('models', S.models);
      } catch (e) {
        toast('Could not load models: ' + e.message, 'err');
      } finally { modelsLoading = null; }
    })();
    return modelsLoading;
  }
  const reloadModelsSoon = debounce(loadModels, 400);
  $('ckptRefresh').addEventListener('click', async () => {
    const b = $('ckptRefresh');
    b.classList.add('spin');
    await loadModels();
    setTimeout(() => b.classList.remove('spin'), 300);
    emit('refresh');
  });
  async function loadSettings() {
    try { S.settings = await api('/api/settings'); emit('settings', S.settings); } catch (_) { }
  }

  // ---------------------------------------------------------------- server events
  let es = null, hiddenAt = 0, retry = null, lastMsg = Date.now(), lastResync = 0;
  function connect() {
    if (es) { try { es.close(); } catch (_) { } }
    clearTimeout(retry);
    es = new EventSource('/api/events');
    lastMsg = Date.now();
    es.addEventListener('ping', () => { lastMsg = Date.now(); });
    es.onopen = () => { S.connected = true; lastMsg = Date.now(); renderEngine(); };
    es.onerror = () => {
      S.connected = false;
      renderEngine();
      // the browser gives up for good after an HTTP error (a reverse proxy answers 502 while Kiln restarts)
      if (es.readyState === EventSource.CLOSED) retry = setTimeout(connect, 3000);
    };
    es.onmessage = (e) => {
      lastMsg = Date.now();
      let m;
      try { m = JSON.parse(e.data); } catch (_) { return; }
      if (m.type === 'job') mergeJob(m.job);
      try { window.dispatchEvent(new CustomEvent('kiln:sse', { detail: m })); } catch (_) { /* a tab's handler failed */ }
      switch (m.type) {
        case 'hello': {
          const first = !S.server;
          S.server = m.server;
          S.engine = m.server.engine;
          S.connected = true;
          applyQueue(m.queue);
          if (!first) { loadModels(); loadSettings(); }
          break;
        }
        case 'engine': S.engine = m.engine; renderEngine(); break;
        case 'queue': applyQueue(m.queue); break;
        case 'models': reloadModelsSoon(); break;
        case 'identify': if (m.state === 'done') reloadModelsSoon(); break;
        case 'settings': loadSettings(); break;
        default: break;
      }
    };
  }

  // Back from the background (phones suspend the page and drop the event stream): reconnect, catch up on
  // jobs that finished or moved on while away, and let the tabs reload what they show.
  async function resync() {
    if (Date.now() - lastResync < 2000) return;  // several signals fire for one wake-up
    lastResync = Date.now();
    connect();
    try { applyQueue(await api('/api/queue')); } catch (_) { }
    const open = [...S.jobs.values()].filter((j) => j.status === 'running' || j.status === 'queued');
    for (const j of open) {
      try {
        const fresh = await api('/api/job/' + encodeURIComponent(j.id));
        mergeJob(fresh);
        window.dispatchEvent(new CustomEvent('kiln:sse', { detail: { type: 'job', job: fresh } }));
      } catch (_) { /* gone from the server: the queue snapshot on reconnect settles it */ }
    }
    emit('resume');
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) hiddenAt = Date.now();
    else if (hiddenAt && Date.now() - hiddenAt > 2000) { hiddenAt = 0; resync(); }
  });
  window.addEventListener('pageshow', (e) => { if (e.persisted) resync(); });
  window.addEventListener('online', () => resync());
  window.addEventListener('kiln:resume', () => resync());   // a wrapper app (WebView), on coming back to the front
  document.addEventListener('resume', () => resync());      // page lifecycle: unfrozen
  // A frozen page runs no timers: a tick that arrives far too late means we were frozen. And a stream that
  // has gone quiet past two server pings is dead even if the browser hasn't noticed yet.
  let lastTick = Date.now();
  setInterval(() => {
    const now = Date.now();
    if (now - lastTick > 10000 || (S.connected && now - lastMsg > 40000)) resync();
    lastTick = now;
  }, 3000);

  // Ctrl+Enter: generate in the current tab (Nodes queues its graph)
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (S.tab === 'nodes') { if (window.KilnNodes) window.KilnNodes.queue(); }
      else emit('generate', { tab: S.tab });
    } else if (e.key === 'Escape') $('queuePop').hidden = true;
  });

  window.Kiln = {
    api, toast, lsGet, lsSet, fmtSec, fmtMs, fmtBytes, fmtCount, setMode, setTab, el, $, clamp, debounce, copyText, isMobile,
    S, on, emit, ckptInfo, loadModels, loadSettings, recordTime, cancel, stageLabel, FAMILY,
  };

  // ---------------------------------------------------------------- boot
  renderQueue();
  renderEngine();
  renderAvg();
  const hash = location.hash.slice(1);
  const startTab = TABS.includes(hash) || hash === 'simple' ? hash : (lsGet(LS_TAB) || (lsGet(LS_MODE_OLD) === 'nodes' ? 'nodes' : 'txt2img'));
  // the other scripts load after this one: start once they have registered their handlers
  // (the event stream opens only now: its first 'hello' must reach every tab's handler)
  window.addEventListener('DOMContentLoaded', () => {
    connect();
    setTab(startTab);
    loadModels();
    loadSettings();
  });
})();
