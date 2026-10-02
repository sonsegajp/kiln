/* Kiln Datasets tab, "Grab from boorus": BooruGrab inside Kiln. Searches every site that's on at once, filters by
   rating, previews a post with its tags by category, and downloads originals with .txt captions into the dataset the
   tab has open. Settings (sites with keys, tag format, blacklist, ...) live in config/booru.json on the server. */
(() => {
  'use strict';
  const K = window.Kiln;
  const { $, el, api, toast } = K;
  const enc = encodeURIComponent;
  const RATINGS = [['g', 'General'], ['s', 'Sensitive'], ['q', 'Questionable'], ['e', 'Explicit']];
  const CATS = ['character', 'copyright', 'artist', 'species', 'general', 'meta'];
  const ENGINE_HELP = 'Engines: Danbooru (danbooru.donmai.us and clones), Gelbooru 0.2 (gelbooru.com, safebooru.org, api.rule34.xxx, xbooru, tbib, ...), ' +
    'Moebooru (konachan.com, konachan.net, yande.re), e621 (e621.net, e926.net), Shimmie (rule34.paheal.net). ' +
    "Gelbooru works without a key (its web pages are read instead); Rule34.xxx needs your user ID and API key from your account's options page. " +
    'Danbooru lets guests search 2 tags at a time. Posts tagged as depicting minors are always skipped unless rated General.';
  const g = { set: null, posts: [], seen: new Set(), pages: {}, open: {}, gen: 0, sel: new Set(), anchor: null, focus: null, pvGen: 0, q: '' };
  const keyOf = (p) => p.site + ':' + p.id;
  const siteOf = (id) => (g.set && g.set.sites.find(s => s.id === id)) || null;
  const saveSoon = K.debounce(() => { if (g.set) api('/api/booru/settings', { method: 'POST', body: { query: $('gQuery').value, ratings: g.set.ratings, sitesOff: g.set.sitesOff } }).catch(() => { }); }, 600);
  window.KilnGrab = { reload: () => load() };

  // ---------------------------------------------------------------- settings -> toggles
  async function load() {
    try {
      g.set = await api('/api/booru/settings');
      if (!$('gQuery').value) $('gQuery').value = g.set.query || '';
      renderToggles();
    } catch (e) { toast('Booru settings: ' + e.message, 'err'); }
  }
  function renderToggles() {
    const box = $('gSites');
    box.textContent = '';
    const on = g.set.sites.filter(s => s.enabled);
    for (const s of on) {
      const l = el('label', 'a-check');
      const c = el('input');
      c.type = 'checkbox';
      c.checked = !g.set.sitesOff.includes(s.id);
      c.addEventListener('change', () => { g.set.sitesOff = g.set.sitesOff.filter(x => x !== s.id); if (!c.checked) g.set.sitesOff.push(s.id); saveSoon(); });
      l.append(c, document.createTextNode(' ' + s.name));
      box.appendChild(l);
    }
    if (!on.length) box.appendChild(el('span', 'a-hint', 'none enabled (Settings…)'));
    const rb = $('gRatings');
    rb.textContent = '';
    for (const [k, name] of RATINGS) {
      const l = el('label', 'a-check r-' + k);
      const c = el('input');
      c.type = 'checkbox';
      c.checked = g.set.ratings.includes(k);
      c.addEventListener('change', () => {
        g.set.ratings = RATINGS.map(r => r[0]).filter(r => (r === k ? c.checked : g.set.ratings.includes(r))).join('');
        saveSoon();
        renderGrid();
      });
      l.append(c, document.createTextNode(' ' + name));
      rb.appendChild(l);
    }
  }
  const activeSites = () => g.set ? g.set.sites.filter(s => s.enabled && !g.set.sitesOff.includes(s.id)) : [];
  const passes = (p) => p.rating === '?' ? /[qe]/.test(g.set.ratings) : g.set.ratings.includes(p.rating);

  // ---------------------------------------------------------------- search
  function notice(t) { const n = $('gNotices'); n.hidden = false; n.textContent = (n.textContent ? n.textContent + '\n' : '') + t; }
  async function search(fresh) {
    if (!g.set) await load();
    const sites = activeSites();
    if (!sites.length) { notice('Turn on at least one site.'); return; }
    if (fresh) {
      g.q = $('gQuery').value.trim();
      g.posts = [];
      g.seen.clear();
      g.sel.clear();
      g.focus = null;
      g.pages = {};
      g.open = {};
      for (const s of sites) { g.pages[s.id] = 0; g.open[s.id] = true; }
      $('gNotices').textContent = '';
      $('gNotices').hidden = true;
      saveSoon();
      preview(null);
      renderGrid();
    }
    const my = ++g.gen;
    $('gGo').disabled = $('gMore').disabled = true;
    $('gStatus').textContent = 'Searching…';
    await Promise.all(sites.filter(s => g.open[s.id]).map(async (s) => {
      const page = (g.pages[s.id] || 0) + 1;
      try {
        const r = await api(`/api/booru/search?site=${enc(s.id)}&q=${enc(g.q)}&page=${page}`);
        if (my !== g.gen) return;
        g.pages[s.id] = page;
        g.open[s.id] = r.full;
        add(r.posts);
      } catch (e) {
        if (my !== g.gen) return;
        g.open[s.id] = false;
        notice(`${s.name}: ${e.message}`);
      }
    }));
    if (my !== g.gen) return;
    $('gGo').disabled = false;
    $('gMore').disabled = !Object.values(g.open).some(Boolean);
    renderStatus();
  }
  function add(list) {
    const grid = $('gGrid');
    for (const p of list) {
      if (g.seen.has(keyOf(p))) continue;
      g.seen.add(keyOf(p));
      g.posts.push(p);
      if (passes(p)) grid.appendChild(card(p));
    }
    renderStatus();
  }
  $('gGo').addEventListener('click', () => search(true));
  $('gMore').addEventListener('click', () => search(false));
  $('gQuery').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); search(true); } });

  // ---------------------------------------------------------------- grid
  const cards = new Map();
  function card(p) {
    const s = siteOf(p.site);
    const c = el('div', 'ds-card r-' + (p.rating === '?' ? 'q' : p.rating));
    c.dataset.key = keyOf(p);
    c.classList.toggle('sel', g.sel.has(keyOf(p)));
    const im = el('img');
    im.loading = 'lazy';
    im.decoding = 'async';
    im.alt = `${s ? s.name : p.site} #${p.id}`;
    const u = p.preview || p.sample;
    if (u) im.src = `/api/booru/img?site=${enc(p.site)}&url=${enc(u)}`;
    im.onerror = () => im.classList.add('bad');
    c.append(im, el('span', 'ds-tick'), el('div', 'ds-cap', `${p.rating.toUpperCase()} · ${s ? s.name : p.site}` + (p.w ? ` · ${p.w}×${p.h}` : '')));
    c.title = Object.values(p.tags || {}).flat().join(' ').slice(0, 600);
    cards.set(keyOf(p), c);
    return c;
  }
  function renderGrid() {
    const grid = $('gGrid');
    grid.textContent = '';
    cards.clear();
    const frag = document.createDocumentFragment();
    for (const p of g.posts) if (passes(p)) frag.appendChild(card(p));
    grid.appendChild(frag);
    for (const k of [...g.sel]) if (!cards.has(k)) g.sel.delete(k);
    renderStatus();
  }
  function shown() { return g.posts.filter(passes); }
  function renderStatus() {
    const n = shown().length, hidden = g.posts.length - n;
    $('gStatus').textContent = `${n} result${n === 1 ? '' : 's'}` + (hidden > 0 ? ` (${hidden} hidden by rating)` : '') + ` · ${g.sel.size} selected`;
    $('gDlSel').disabled = !g.sel.size;
    const ds = target();
    $('gDlSel').textContent = ds ? `Download ${g.sel.size || ''} to ${ds.name}`.replace('  ', ' ') : 'Download selected';
  }
  function paint() { for (const [k, c] of cards) c.classList.toggle('sel', g.sel.has(k)); renderStatus(); }
  $('gGrid').addEventListener('click', (e) => {
    const c = e.target.closest('.ds-card');
    if (!c) return;
    const k = c.dataset.key;
    if (e.shiftKey && g.anchor) {
      const order = shown().map(keyOf), a = order.indexOf(g.anchor), b = order.indexOf(k);
      if (a >= 0 && b >= 0) for (let i = Math.min(a, b); i <= Math.max(a, b); i++) g.sel.add(order[i]);
    } else if (e.ctrlKey || e.metaKey || e.target.classList.contains('ds-tick')) {
      if (g.sel.has(k)) g.sel.delete(k); else g.sel.add(k);
      g.anchor = k;
    } else {
      g.sel.clear();
      g.sel.add(k);
      g.anchor = k;
    }
    paint();
    preview(g.posts.find(p => keyOf(p) === k) || null);
  });
  $('gGrid').addEventListener('dblclick', (e) => {
    const c = e.target.closest('.ds-card');
    const p = c && g.posts.find(x => keyOf(x) === c.dataset.key);
    if (p && p.page) window.open(p.page, '_blank', 'noopener');
  });
  function selectAll() { for (const p of shown()) g.sel.add(keyOf(p)); paint(); }
  $('gSelAll').addEventListener('click', selectAll);
  document.addEventListener('keydown', (e) => {
    if (K.S.tab !== 'datasets' || $('dsGrab').hidden || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    if ((e.ctrlKey || e.metaKey) && e.key === 'a') { e.preventDefault(); selectAll(); }
  });

  // ---------------------------------------------------------------- preview
  function captionOf(p) {  // as the server writes it (BooruGrab's format)
    const order = ['character', 'copyright', 'artist', 'species', 'general'];
    if (g.set && g.set.includeMeta) order.push('meta');
    const seen = new Set(), tags = [];
    for (const c of order) for (const t of (p.tags && p.tags[c]) || []) if (t && !seen.has(t)) { seen.add(t); tags.push(t); }
    if (g.set && g.set.tagFormat === 'raw') return tags.join(' ');
    return tags.map(t => /[A-Za-z0-9]/.test(t) ? t.replace(/_/g, ' ') : t).join(', ');
  }
  async function preview(p) {
    const my = ++g.pvGen;
    g.focus = p;
    for (const id of ['gPvDl', 'gPvCopy']) $(id).disabled = !p;
    $('gPvOpen').hidden = !p;
    const box = $('gPvTags');
    box.textContent = '';
    $('gPvImg').removeAttribute('src');
    if (!p) { $('gPvTitle').textContent = 'Click a thumbnail to preview it'; return; }
    const s = siteOf(p.site);
    $('gPvTitle').textContent = `${s ? s.name : p.site} #${p.id}` + (p.w ? ` · ${p.w}×${p.h} · ${p.ext}` : '');
    $('gPvOpen').href = p.page || '#';
    const flat = Object.keys(p.tags || {}).length === 1;
    for (const cat of CATS) {
      const list = (p.tags || {})[cat];
      if (!list || !list.length) continue;
      box.appendChild(el('div', 'g-cat', flat ? 'TAGS' : cat.toUpperCase()));
      for (const t of list) {
        const a = el('span', 'g-t c-' + cat, t.replace(/_/g, ' '));
        a.title = 'Search ' + t;
        a.addEventListener('click', () => { $('gQuery').value = t; search(true); });
        box.appendChild(a);
      }
    }
    try {
      if (p.html && !p.file) {
        const r = await api('/api/booru/resolve', { method: 'POST', body: { site: p.site, post: p } });
        if (my !== g.pvGen) return;
        Object.assign(p, r);
        $('gPvTitle').textContent = `${s ? s.name : p.site} #${p.id} · ${p.w}×${p.h} · ${p.ext}`;
      }
      const u = p.sample || p.file;
      if (u) $('gPvImg').src = `/api/booru/img?site=${enc(p.site)}&url=${enc(u)}`;
    } catch (e) { if (my === g.pvGen) $('gPvTitle').textContent += `  (preview failed: ${e.message})`; }
  }
  $('gPvCopy').addEventListener('click', () => { if (g.focus) K.copyText(captionOf(g.focus), 'Tags copied'); });
  $('gPvDl').addEventListener('click', () => { if (g.focus) download([g.focus]); });

  // ---------------------------------------------------------------- downloads (into the open dataset)
  function target() {
    const dir = window.KilnDatasets ? window.KilnDatasets.current() : '';
    if (!dir) return null;
    return { dir, name: dir.split(/[\\/]/).filter(Boolean).pop() };
  }
  async function download(posts) {
    const t = target();
    if (!t) { toast('Pick or create a dataset at the top first', 'err'); return; }
    try {
      const r = await api('/api/booru/download', { method: 'POST', body: { dir: t.dir, posts } });
      if (!r.queued) toast('Nothing to download', 'err');
      dlStatus(r.status);
    } catch (e) { toast('Download: ' + e.message, 'err', 6000); }
  }
  $('gDlSel').addEventListener('click', () => download(g.posts.filter(p => g.sel.has(keyOf(p)))));
  $('gCancel').addEventListener('click', () => api('/api/booru/cancel', { method: 'POST', body: {} }).then(dlStatus).catch(() => { }));
  function dlStatus(s) {
    if (!s) return;
    const finished = s.done + s.skipped + s.failed;
    $('gDlStatus').textContent = s.total ? `Downloaded ${s.done} · skipped ${s.skipped} · failed ${s.failed}` + (s.pending ? ` · ${s.pending} to go` : '') : '';
    $('gDlStatus').title = s.last_error ? 'Last failure: ' + s.last_error : '';
    $('gBar').hidden = !s.pending;
    if (s.pending) $('gBarFill').style.width = (100 * finished / Math.max(1, s.total)).toFixed(1) + '%';
    $('gCancel').hidden = !s.queued;
  }
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    if (m.type !== 'booru') return;
    dlStatus(m);
    if (m.file) window.dispatchEvent(new CustomEvent('kiln:dsfile', { detail: { dir: m.dir, file: m.file } }));
  });
  window.addEventListener('kiln:dataset', renderStatus);

  // ---------------------------------------------------------------- settings dialog
  function row(s) {
    const tr = el('tr');
    const cell = (node) => { const td = el('td'); td.appendChild(node); tr.appendChild(td); return node; };
    const on = cell(el('input'));
    on.type = 'checkbox';
    on.checked = s.enabled !== false;
    on.dataset.f = 'enabled';
    for (const [f, ph] of [['name', 'Name'], ['engine', ''], ['url', 'https://…'], ['user', ''], ['key', '']]) {
      let n;
      if (f === 'engine') {
        n = el('select');
        for (const e of g.set.engines) { const o = el('option', '', e.name); o.value = e.id; n.appendChild(o); }
        n.value = s.engine || 'danbooru';
      } else {
        n = el('input');
        n.type = f === 'key' ? 'password' : 'text';
        n.placeholder = ph;
        n.value = s[f] || '';
        n.spellcheck = false;
      }
      n.dataset.f = f;
      cell(n);
    }
    const del = cell(el('button', 'a-btn sm', '×'));
    del.type = 'button';
    del.title = 'Remove this site';
    del.addEventListener('click', () => tr.remove());
    tr.dataset.id = s.id || '';
    return tr;
  }
  function openSettings() {
    if (!g.set) return;
    $('gsFmt').value = g.set.tagFormat;
    $('gsLimit').value = g.set.limit;
    $('gsConc').value = g.set.concurrency;
    $('gsSkip').checked = g.set.skipExisting;
    $('gsMeta').checked = g.set.includeMeta;
    $('gsJson').checked = g.set.writeJson;
    $('gsBlack').value = g.set.blacklist.join(' ');
    const tb = $('gsSites');
    tb.textContent = '';
    for (const s of g.set.sites) tb.appendChild(row(s));
    $('gsEngines').textContent = ENGINE_HELP;
    $('gSettings').showModal();
  }
  $('gSetBtn').addEventListener('click', async () => { if (!g.set) await load(); openSettings(); });
  $('gsAdd').addEventListener('click', () => $('gsSites').appendChild(row({ name: '', engine: 'danbooru', url: '', enabled: true })));
  $('gsCancel').addEventListener('click', () => $('gSettings').close());
  $('gsSave').addEventListener('click', async () => {
    const sites = [...$('gsSites').querySelectorAll('tr')].map(tr => {
      const s = { id: tr.dataset.id };
      for (const n of tr.querySelectorAll('[data-f]')) s[n.dataset.f] = n.type === 'checkbox' ? n.checked : n.value;
      return s;
    }).filter(s => s.url.trim());
    try {
      g.set = await api('/api/booru/settings', { method: 'POST', body: {
        sites, tagFormat: $('gsFmt').value, limit: $('gsLimit').value, concurrency: $('gsConc').value, skipExisting: $('gsSkip').checked,
        includeMeta: $('gsMeta').checked, writeJson: $('gsJson').checked, blacklist: $('gsBlack').value,
      } });
      $('gSettings').close();
      renderToggles();
      toast('Booru settings saved', 'ok', 1800);
    } catch (e) { toast(e.message, 'err', 6000); }
  });
  $('gsImport').addEventListener('click', async () => {
    try {
      const r = await api('/api/booru/import', { method: 'POST', body: {} });
      g.set = r.settings;
      openSettings();
      renderToggles();
      toast(`Imported ${r.sites} sites from BooruGrab`, 'ok');
    } catch (e) { toast('Import: ' + e.message, 'err', 7000); }
  });

  window.addEventListener('kiln:dsview', (e) => { if (e.detail.view === 'grab') { if (!g.set) load(); api('/api/booru/status').then(dlStatus).catch(() => { }); renderStatus(); } });
})();
