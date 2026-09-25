/* Kiln Extra Networks (under the prompts): local LoRA / checkpoint cards and the CivitAI browser
   (the same browser also opens full screen). Downloads are managed by the server; this file shows
   them (progress, speed, ETA, cancel, retry). The CivitAI API key never reaches the browser. */
(() => {
  'use strict';
  const K = window.Kiln;
  const { $, el, api, toast, lsGet, lsSet, debounce, fmtBytes, fmtCount } = K;
  const GEN = () => window.KilnGen;

  const FAMILY_BASES = { anima: ['Anima'], sdxl: ['SDXL 1.0', 'Pony', 'Illustrious', 'NoobAI'], sd15: ['SD 1.5'], flux: ['Flux.1 D', 'Flux.1 S'] };
  function familyOfBase(b) {
    b = String(b || '').toLowerCase();
    if (!b) return null;
    if (b === 'anima') return 'anima';
    if (/^(sdxl|pony|illustrious|noobai)/.test(b)) return 'sdxl';
    if (/^sd 1\./.test(b)) return 'sd15';
    if (/^flux/.test(b)) return 'flux';
    return 'other';
  }
  const models = () => K.S.models || { loras: [], sd_models: [], bases: [] };
  const canManage = () => !!(K.S.settings && K.S.settings.manage);

  // ---------------------------------------------------------------- Extra Networks panel
  const X = { open: false, tab: lsGet('kiln.xnet.tab') || 'lora', q: '', sort: 'name' };
  function setOpen(o) {
    X.open = o;
    $('xnet').hidden = !o;
    $('toolNets').setAttribute('aria-pressed', String(o));
    if (o) render();
  }
  K.on('xnet', (d) => setOpen(!!d.open));
  $('xnetTabs').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-x]');
    if (!b) return;
    X.tab = b.dataset.x;
    lsSet('kiln.xnet.tab', X.tab);
    render();
  });
  $('xnetSearch').addEventListener('input', debounce(() => { X.q = $('xnetSearch').value.trim(); if (X.tab === 'civitai') { CV.q.query = X.q; CV.search(true); } else renderLocal(); }, 250));
  $('xnetSort').addEventListener('change', () => { X.sort = $('xnetSort').value; renderLocal(); });
  $('xnetRefresh').addEventListener('click', async () => { await K.loadModels(); if (X.tab === 'civitai') CV.search(true); });
  function render() {
    for (const b of $('xnetTabs').querySelectorAll('button')) b.setAttribute('aria-selected', String(b.dataset.x === X.tab));
    const cv = X.tab === 'civitai';
    $('xnetSort').hidden = cv;
    $('xnetSearch').placeholder = cv ? 'Search CivitAI' : X.tab === 'lora' ? 'Search LoRAs (name, trigger words, base)' : 'Search checkpoints';
    $('xnetCards').hidden = cv;
    $('cvHostInline').hidden = !cv;
    if (cv) { $('xnetSearch').value = CV.q.query; CV.mount($('cvHostInline'), 'inline'); CV.activate(); }
    else { $('xnetSearch').value = X.q; renderLocal(); }
  }
  function renderLocal() {
    const box = $('xnetCards');
    box.textContent = '';
    const q = X.q.toLowerCase();
    const ck = K.ckptInfo();
    const prompt = GEN() ? GEN().G.prompt : '';
    const inPrompt = new Set(GEN() ? GEN().extractLoraTags(prompt).tags.map(t => t.name.toLowerCase()) : []);
    const slots = new Set(GEN() ? GEN().G.slots.filter(s => s.on && s.file).map(s => s.file) : []);
    let list = X.tab === 'lora' ? models().loras.slice() : models().sd_models.slice();
    if (q) list = list.filter(m => [m.name, m.file, m.base, m.civitai && m.civitai.name, ...(m.words || [])].some(v => v && String(v).toLowerCase().includes(q)));
    if (X.sort === 'base') list.sort((a, b) => String(a.base || a.family).localeCompare(String(b.base || b.family)) || a.name.localeCompare(b.name));
    else if (X.sort === 'new') list.sort((a, b) => (b.size || 0) - (a.size || 0));
    else list.sort((a, b) => a.name.localeCompare(b.name));
    for (const m of list) {
      const card = el('div', 'cv-card a-card');
      card.tabIndex = 0;
      card.setAttribute('role', 'button');
      const on = X.tab === 'lora' ? inPrompt.has(m.name.toLowerCase()) || slots.has(m.file) : ck && ck.file === m.file;
      card.classList.toggle('on', !!on);
      const media = el('div', 'cv-media');
      if (m.preview) { const im = el('img'); im.src = m.preview; im.alt = ''; im.loading = 'lazy'; im.referrerPolicy = 'no-referrer'; media.appendChild(im); }
      else media.appendChild(el('div', 'a-card-ph', m.name.slice(0, 1).toUpperCase()));
      const base = m.base || K.FAMILY[m.family] || '?';
      const other = ck && m.family && m.family !== 'unknown' && ck.family && m.family !== ck.family;
      const tl = el('div', 'cv-corner tl');
      tl.appendChild(el('span', 'a-badge' + (X.tab === 'lora' ? (other ? ' bad' : ' ok') : m.runnable ? ' ok' : ' bad'), base));
      if (X.tab === 'ckpt' && !m.runnable) tl.appendChild(el('span', 'a-badge bad', 'not supported'));
      if (on) tl.appendChild(el('span', 'a-badge up', X.tab === 'lora' ? 'in use' : 'selected'));
      media.appendChild(tl);
      const scrim = el('div', 'cv-scrim');
      scrim.appendChild(el('div', 'cv-title', m.civitai && m.civitai.name ? m.civitai.name : m.name));
      scrim.appendChild(el('div', 'cv-by', m.civitai && m.civitai.name ? m.name : (X.tab === 'lora' ? (other ? 'other base model' : ck && ck.family === 'sdxl' ? 'LoRA · not on SDXL yet' : 'LoRA') : m.runnable ? 'runs in Kiln' : 'this engine build can\'t run it')));
      media.appendChild(scrim);
      card.appendChild(media);
      if (m.words && m.words.length) {
        const chips = el('div', 'cv-chips');
        for (const w of m.words.slice(0, 8)) chips.appendChild(el('span', 'cv-chip', w));
        card.appendChild(chips);
      }
      card.title = `${m.file}${m.civitai ? '\n' + m.civitai.name + (m.civitai.versionName ? ' · ' + m.civitai.versionName : '') : ''}${m.words && m.words.length ? '\nTrigger words: ' + m.words.join(', ') : ''}${other ? '\nFor ' + base + ', not the selected checkpoint' : ''}`;
      // not matched to CivitAI yet: offer "identify" (hashes the file, asks CivitAI by hash)
      const busyId = m.identifying && !/^(done|error)/.test(m.identifying);
      const idBtn = () => {
        if (m.civitai || busyId) return null;
        const b = el('button', '', 'identify'); b.type = 'button';
        b.title = 'Look this file up on CivitAI by its SHA256 (base model, name, trigger words). Big checkpoints take a while to hash.';
        b.addEventListener('click', async (e) => {
          e.stopPropagation();
          try { await api('/api/models/identify', { method: 'POST', body: { file: m.file } }); toast(`Identifying ${m.name}…`, '', 2500); m.identifying = 'hashing'; renderLocal(); }
          catch (err) { toast(err.message, 'err'); }
        });
        return b;
      };
      if (busyId) tl.appendChild(el('span', 'a-badge up', m.identifying + '…'));
      if (X.tab === 'lora') {
        const act = el('div', 'a-card-act');
        const ib = idBtn();
        if (ib) act.appendChild(ib);
        const slot = el('button', '', '+slot'); slot.type = 'button'; slot.title = 'Put this LoRA in a slot';
        slot.addEventListener('click', (e) => { e.stopPropagation(); GEN().addToSlot(m.file, 1); renderLocal(); });
        act.appendChild(slot);
        if (m.words && m.words.length) {
          const w = el('button', '', 'words'); w.type = 'button'; w.title = 'Add the trigger words to the prompt';
          w.addEventListener('click', (e) => { e.stopPropagation(); GEN().insertText(m.words.slice(0, 6)); });
          act.appendChild(w);
        }
        card.appendChild(act);
        card.addEventListener('click', () => { GEN().toggleLoraTag(m.name, 1); renderLocal(); });
        card.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); card.click(); } });
      } else {
        const ib = idBtn();
        if (ib) { const act = el('div', 'a-card-act'); act.appendChild(ib); card.appendChild(act); }
        card.addEventListener('click', () => { const sel = $('ckptSel'); sel.value = m.file; sel.dispatchEvent(new Event('change')); renderLocal(); });
        card.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); card.click(); } });
      }
      box.appendChild(card);
    }
    if (!list.length) box.appendChild(el('div', 'a-cards-empty', q ? 'Nothing matches.' : X.tab === 'lora' ? 'No LoRAs in models/loras yet. Find some in the CivitAI tab.' : 'No checkpoints found.'));
  }
  K.on('models', () => { if (X.open && X.tab !== 'civitai') renderLocal(); CV.onModels(); });
  K.on('prompt', () => { if (X.open && X.tab === 'lora') renderLocal(); });
  K.on('ckpt', () => { if (X.open && X.tab !== 'civitai') renderLocal(); CV.onCkpt(); });

  // ---------------------------------------------------------------- CivitAI browser
  const TYPES = [
    ['LORA', 'LoRA'], ['LORA,LoCon,DoRA', 'All LoRA types'], ['LoCon', 'LoCon / LyCORIS'], ['DoRA', 'DoRA'],
    ['Checkpoint', 'Checkpoint'], ['TextualInversion', 'Embedding'], ['Upscaler', 'Upscaler'],
  ];
  const SORTS = ['Highest Rated', 'Most Downloaded', 'Newest'];
  const PERIODS = [['AllTime', 'All time'], ['Year', 'Year'], ['Month', 'Month'], ['Week', 'Week'], ['Day', 'Day']];
  const CV = {
    q: Object.assign({ query: '', tag: '', username: '', types: 'LORA', base: 'auto', sort: 'Highest Rated', period: 'AllTime', nsfw: null }, lsGet('kiln.cv.q.v1') || {}, { query: '', tag: '', username: '' }),
    items: [], cursor: null, loading: false, done: false, err: null, token: 0, root: null, where: null, built: false, identified: new Set(),
  };
  let ui = {};
  function ckBases() {
    const ck = K.ckptInfo();
    if (!ck) return { label: 'any', list: [] };
    if (ck.base) return { label: ck.base, list: [ck.base], family: ck.family };
    return { label: `all ${K.FAMILY[ck.family] || ck.family}`, list: FAMILY_BASES[ck.family] || [], family: ck.family };
  }
  function baseList() {
    const ck = K.ckptInfo();
    if (CV.q.base === 'auto') return ckBases().list;
    if (CV.q.base === 'compatible') return ck ? FAMILY_BASES[ck.family] || [] : [];
    if (CV.q.base === 'any') return [];
    return [CV.q.base];
  }
  CV.build = function () {
    if (CV.built) return;
    CV.built = true;
    const root = el('div', 'cv');
    const bar = el('div', 'cv-bar');
    ui.query = el('input'); ui.query.type = 'search'; ui.query.placeholder = 'Search CivitAI (name, style, character…)'; ui.query.autocomplete = 'off'; ui.query.spellcheck = false;
    ui.query.className = 'cv-query';
    const sel = (opts, value) => { const s = el('select'); for (const [v, l] of opts) { const o = el('option', '', l); o.value = v; s.appendChild(o); } s.value = value; return s; };
    ui.type = sel(TYPES, CV.q.types);
    ui.base = el('select');
    ui.sort = sel(SORTS.map(s => [s, s]), CV.q.sort);
    ui.period = sel(PERIODS, CV.q.period);
    const nsfwL = el('label', 'a-check');
    ui.nsfw = el('input'); ui.nsfw.type = 'checkbox';
    nsfwL.append(ui.nsfw, el('span', '', 'NSFW'));
    ui.more = el('button', 'a-btn sm cv-more', 'Filters'); ui.more.type = 'button'; ui.more.setAttribute('aria-pressed', 'false');
    ui.full = el('button', 'a-btn sm cv-more', '⛶ Full screen'); ui.full.type = 'button';
    ui.dl = el('button', 'a-btn sm cv-more', 'Downloads'); ui.dl.type = 'button';
    bar.append(ui.query, ui.type, ui.base, ui.sort, ui.period, nsfwL, ui.more, ui.dl, ui.full);
    const adv = el('div', 'cv-adv');
    adv.hidden = true;
    ui.tag = el('input'); ui.tag.type = 'text'; ui.tag.placeholder = 'Tag (e.g. style, character, clothing)'; ui.tag.autocomplete = 'off';
    ui.user = el('input'); ui.user.type = 'text'; ui.user.placeholder = 'Creator username'; ui.user.autocomplete = 'off';
    ui.clear = el('button', 'a-btn sm', 'Clear filters'); ui.clear.type = 'button';
    adv.append(ui.tag, ui.user, ui.clear);
    ui.status = el('div', 'cv-status');
    ui.grid = el('div', 'cv-grid');
    ui.sentinel = el('div', 'cv-sentinel');
    ui.grid.appendChild(ui.sentinel);
    root.append(bar, adv, ui.status, ui.grid);
    CV.root = root;
    const again = () => { lsSet('kiln.cv.q.v1', Object.assign({}, CV.q, { query: '', tag: '', username: '' })); CV.search(true); };
    const typed = debounce(again, 400);
    ui.query.addEventListener('input', () => { CV.q.query = ui.query.value.trim(); if ($('xnetSearch') && X.tab === 'civitai') $('xnetSearch').value = ui.query.value; typed(); });
    ui.type.addEventListener('change', () => { CV.q.types = ui.type.value; again(); });
    ui.base.addEventListener('change', () => { CV.q.base = ui.base.value; again(); });
    ui.sort.addEventListener('change', () => { CV.q.sort = ui.sort.value; again(); });
    ui.period.addEventListener('change', () => { CV.q.period = ui.period.value; again(); });
    ui.nsfw.addEventListener('change', () => { CV.q.nsfw = ui.nsfw.checked; again(); });
    ui.tag.addEventListener('input', () => { CV.q.tag = ui.tag.value.trim(); typed(); });
    ui.user.addEventListener('input', () => { CV.q.username = ui.user.value.trim(); typed(); });
    ui.more.addEventListener('click', () => { adv.hidden = !adv.hidden; ui.more.setAttribute('aria-pressed', String(!adv.hidden)); });
    ui.clear.addEventListener('click', () => { CV.q.tag = ''; CV.q.username = ''; CV.q.query = ''; ui.tag.value = ''; ui.user.value = ''; ui.query.value = ''; if (X.tab === 'civitai') $('xnetSearch').value = ''; again(); });
    ui.full.addEventListener('click', () => (CV.where === 'full' ? closeFull() : openFull()));
    ui.dl.addEventListener('click', openDownloads);
    CV.io = new IntersectionObserver((ents) => { for (const e of ents) if (e.isIntersecting && !CV.loading && !CV.done && CV.items.length) CV.search(false); }, { root: ui.grid, rootMargin: '600px' });
    CV.io.observe(ui.sentinel);
  };
  CV.renderBase = function () {
    const ck = K.ckptInfo();
    const cur = CV.q.base;
    ui.base.textContent = '';
    const add = (v, l) => { const o = el('option', '', l); o.value = v; ui.base.appendChild(o); };
    add('auto', `Checkpoint: ${ckBases().label}`);
    if (ck && FAMILY_BASES[ck.family] && FAMILY_BASES[ck.family].length > 1) add('compatible', `All ${K.FAMILY[ck.family]} bases`);
    for (const b of models().bases || []) add(b, b);
    add('any', 'Any base');
    ui.base.value = [...ui.base.options].some(o => o.value === cur) ? cur : 'auto';
  };
  CV.mount = function (host, where) {
    CV.build();
    if (CV.root.parentNode !== host) host.appendChild(CV.root);
    CV.where = where;
    ui.query.hidden = where === 'inline';     // inline: the Extra Networks search bar is the query box
    ui.full.textContent = where === 'full' ? 'Exit full screen' : '⛶ Full screen';
    ui.dl.hidden = where === 'full';     // the full-screen header has its own Downloads button
  };
  CV.activate = function () {
    if (CV.q.nsfw === null) CV.q.nsfw = !!(K.S.settings && K.S.settings.civitai && K.S.settings.civitai.nsfw);
    ui.nsfw.checked = !!CV.q.nsfw;
    ui.query.value = CV.q.query;
    CV.renderBase();
    identifyOnce();
    if (!CV.items.length && !CV.loading) CV.search(true);
    refreshDownloads();
  };
  CV.onModels = function () { if (CV.built) { CV.renderBase(); for (const c of ui.grid.querySelectorAll('.cv-card')) c.dataset.stale = '1'; } };
  CV.onCkpt = function () { if (CV.built && (CV.where === 'full' || (X.open && X.tab === 'civitai'))) { CV.renderBase(); if (CV.q.base === 'auto' || CV.q.base === 'compatible') CV.search(true); } };
  // work out the selected checkpoint's base (hash lookup) and match local LoRAs to CivitAI, in the background
  function identifyOnce() {
    const ck = K.ckptInfo();
    if (ck && !ck.base && !ck.civitai && ck.family !== 'unknown' && !CV.identified.has(ck.file)) {
      CV.identified.add(ck.file);
      api('/api/models/identify', { method: 'POST', body: { file: ck.file } }).catch(() => { });
    }
    if (!CV.identified.has('*loras')) { CV.identified.add('*loras'); api('/api/models/identify', { method: 'POST', body: { loras: true } }).catch(() => { }); }
  }
  CV.search = async function (reset) {
    CV.build();
    const token = ++CV.token;
    if (reset) { CV.items = []; CV.cursor = null; CV.done = false; CV.err = null; for (const c of [...ui.grid.querySelectorAll('.cv-card, .cv-empty, .cv-err')]) c.remove(); }
    if (CV.done) return;
    CV.loading = true;
    renderStatus();
    const sp = new URLSearchParams();
    for (const t of CV.q.types.split(',')) sp.append('types', t);
    for (const b of baseList()) sp.append('baseModels', b);
    if (CV.q.query) sp.set('query', CV.q.query);
    if (CV.q.tag) sp.set('tag', CV.q.tag);
    if (CV.q.username) sp.set('username', CV.q.username);
    sp.set('sort', CV.q.sort); sp.set('period', CV.q.period); sp.set('nsfw', CV.q.nsfw ? 'true' : 'false');
    if (CV.cursor) sp.set('cursor', CV.cursor);
    try {
      const r = await api('/api/civitai/search?' + sp.toString());
      if (token !== CV.token) return;
      CV.cursor = r.nextCursor;
      CV.done = !r.nextCursor || !r.items.length;
      const frag = document.createDocumentFragment();
      for (const m of r.items) { if (CV.items.some(x => x.id === m.id)) continue; CV.items.push(m); frag.appendChild(card(m)); }
      ui.grid.insertBefore(frag, ui.sentinel);
      if (!CV.items.length) ui.grid.insertBefore(el('div', 'cv-empty', 'Nothing found. Try another base model, type or search.'), ui.sentinel);
    } catch (e) {
      if (token !== CV.token) return;
      CV.err = e.message;
      ui.grid.insertBefore(el('div', 'cv-err', e.message), ui.sentinel);
    } finally {
      if (token === CV.token) { CV.loading = false; renderStatus(); }
    }
  };
  function renderStatus() {
    if (!ui.status) return;
    const ck = K.ckptInfo();
    ui.status.textContent = '';
    const bases = baseList();
    ui.status.append(el('span', '', CV.loading ? 'Loading…' : `${CV.items.length}${CV.done ? '' : '+'} result${CV.items.length === 1 ? '' : 's'}`));
    ui.status.append(el('span', '', `base: ${bases.length ? bases.join(', ') : 'any'}`));
    if (ck && ck.identifying && !/^(done|error)/.test(ck.identifying)) ui.status.append(el('span', '', `identifying ${ck.name} (${ck.identifying})…`));
    if (!canManage()) ui.status.append(el('span', '', 'downloads start on the PC running Kiln only'));
    $('cvFullSub').textContent = ck ? `checkpoint: ${ck.name} (${ck.base || K.FAMILY[ck.family]})` : '';
  }
  function thumbFor(m) {
    const v = m.versions[0];
    const ims = v ? v.images : [];
    return ims.find(i => i.type === 'image') || ims[0] || null;
  }
  // image.civitai.com/<key>/<uuid>/<transform>/<name>: a CDN size that matches the card at this screen's DPR
  const CDN_W = [320, 450, 640, 900];
  function cdn(url, cssWidth, still) {
    if (typeof url !== 'string' || !/^https:\/\//.test(url)) return url;
    const want = Math.min(1100, cssWidth * Math.min(2, window.devicePixelRatio || 1));
    const w = CDN_W.find(x => x >= want) || 900;
    const parts = url.split('/');
    if (parts.length < 6) return url;
    const t = `${still ? 'anim=false,' : ''}width=${w}`;
    if (parts[parts.length - 2].includes('=')) parts[parts.length - 2] = t; else parts.splice(parts.length - 1, 0, t);
    return parts.join('/');
  }
  const cardWidth = () => Math.max(200, Math.min(420, (ui.grid && ui.grid.clientWidth ? ui.grid.clientWidth / Math.max(1, Math.floor(ui.grid.clientWidth / (K.isMobile() ? 160 : 240))) : 280)));
  function mediaBox(m, im, width) {
    const blur = blurred(m, im);
    const media = el('div', 'cv-media loading' + (blur ? ' cv-blur' : ''));
    if (im) {
      const img = el('img');
      img.alt = '';
      img.loading = 'lazy';
      img.decoding = 'async';
      img.referrerPolicy = 'no-referrer';
      img.onload = () => media.classList.remove('loading');
      img.onerror = () => media.classList.remove('loading');
      img.src = cdn(im.url, width, true);
      media.appendChild(img);
      if (im.type === 'video') {
        media.appendChild(el('span', 'cv-play', '▶'));
        let vid = null;
        media.addEventListener('mouseenter', () => {
          if (media.classList.contains('cv-blur')) return;
          if (!vid) {
            vid = el('video');
            vid.muted = true; vid.loop = true; vid.playsInline = true; vid.preload = 'auto';
            vid.poster = img.src;
            vid.src = cdn(im.url, width, false);
            media.insertBefore(vid, img.nextSibling);
          }
          vid.play().then(() => media.classList.add('playing')).catch(() => { });
        });
        media.addEventListener('mouseleave', () => { if (vid) { vid.pause(); media.classList.remove('playing'); } });
      }
    } else media.classList.remove('loading');
    if (blur) {
      const r = el('div', 'cv-reveal');
      r.append(el('b', '', 'NSFW'), el('span', '', 'click to reveal'));
      media.appendChild(r);
    }
    return media;
  }
  const blurred = (m, im) => !CV.q.nsfw && ((im && im.nsfwLevel >= 4) || (!im && m.nsfw));
  function compat(m) {
    const ck = K.ckptInfo();
    const v = m.versions[0];
    if (!ck || !v) return null;
    const f = familyOfBase(v.baseModel);
    if (!f || f === 'other') return 'other';
    return f === ck.family ? 'ok' : 'other';
  }
  function card(m) {
    const c = el('div', 'cv-card');
    c.tabIndex = 0;
    c.setAttribute('role', 'button');
    c.dataset.id = m.id;
    const v = m.versions[0] || {};
    const media = mediaBox(m, thumbFor(m), cardWidth());
    // corner pills: base model (+ fit) top-left, local state top-right; never over the title
    const tl = el('div', 'cv-corner tl'), tr = el('div', 'cv-corner tr');
    const cp = compat(m);
    tl.appendChild(el('span', 'a-badge' + (cp === 'ok' ? ' ok' : cp ? ' bad' : ''), v.baseModel || '?'));
    if (m.type !== 'LORA') tl.appendChild(el('span', 'a-badge', m.type === 'TextualInversion' ? 'Embedding' : m.type));
    if (m.local && m.local.downloaded) tr.appendChild(el('span', 'a-badge ok', 'Downloaded ✓'));
    if (m.local && m.local.update) tr.appendChild(el('span', 'a-badge up', 'Update'));
    if (m.local && m.local.downloading) tr.appendChild(el('span', 'a-badge up', 'Downloading'));
    if (v.early) tr.appendChild(el('span', 'a-badge bad', 'Early access'));
    media.append(tl, tr);
    const scrim = el('div', 'cv-scrim');
    const title = el('div', 'cv-title', m.name);
    title.title = m.name;
    scrim.appendChild(title);
    if (m.creator) scrim.appendChild(el('div', 'cv-by', m.creator.username));
    scrim.appendChild(el('div', 'cv-stats', `⬇ ${fmtCount(m.stats.downloads)}   👍 ${fmtCount(m.stats.likes)}${cp === 'other' ? '   · other base' : ''}`));
    media.appendChild(scrim);
    c.appendChild(media);
    if (v.trainedWords && v.trainedWords.length) {
      const chips = el('div', 'cv-chips');
      for (const w of v.trainedWords.slice(0, 8)) chips.appendChild(el('span', 'cv-chip', w));
      chips.title = v.trainedWords.join(', ');
      c.appendChild(chips);
    }
    const open = () => {
      if (media.classList.contains('cv-blur')) { media.classList.remove('cv-blur'); const r = media.querySelector('.cv-reveal'); if (r) r.remove(); return; }
      openDetail(m.id);
    };
    c.addEventListener('click', open);
    c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    return c;
  }

  // ---------------------------------------------------------------- full screen
  function openFull() {
    CV.mount($('cvHostFull'), 'full');
    $('cvFull').hidden = false;
    document.body.style.overflow = 'hidden';
    CV.activate();
    renderStatus();
    ui.query.focus();
  }
  function closeFull() {
    $('cvFull').hidden = true;
    document.body.style.overflow = '';
    if (X.open && X.tab === 'civitai') CV.mount($('cvHostInline'), 'inline');
    else CV.where = null;
  }
  $('cvFullClose').addEventListener('click', closeFull);
  $('toolCivitai').addEventListener('click', openFull);
  $('cvDownloadsBtn').addEventListener('click', openDownloads);

  // ---------------------------------------------------------------- detail view
  const D = { m: null, v: 0 };
  // second line of defence: the server already sanitized the description
  function safeHtml(html) {
    const t = document.createElement('template');
    t.innerHTML = html || '';
    const ok = new Set(['P', 'BR', 'STRONG', 'B', 'EM', 'I', 'U', 'S', 'STRIKE', 'UL', 'OL', 'LI', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 'CODE', 'PRE', 'A', 'HR', 'SPAN', 'DIV', 'TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'SUP', 'SUB']);
    const walk = (n) => {
      for (const c of [...n.childNodes]) {
        if (c.nodeType === 1) {
          if (!ok.has(c.tagName)) { c.replaceWith(document.createTextNode(c.textContent || '')); continue; }
          for (const a of [...c.attributes]) if (!(c.tagName === 'A' && ['href', 'target', 'rel'].includes(a.name))) c.removeAttribute(a.name);
          if (c.tagName === 'A') { if (!/^https?:\/\//i.test(c.getAttribute('href') || '')) c.removeAttribute('href'); c.target = '_blank'; c.rel = 'noopener noreferrer nofollow'; }
          walk(c);
        } else if (c.nodeType !== 3) c.remove();
      }
    };
    walk(t.content);
    return t.content;
  }
  async function openDetail(id) {
    const box = $('cvDetail');
    box.hidden = false;
    box.textContent = '';
    const panel = el('div', 'cv-panel');
    panel.appendChild(el('div', 'cv-phead', 'Loading…'));
    box.appendChild(panel);
    try {
      D.m = await api('/api/civitai/model/' + id);
      // open on the version that fits the checkpoint (the newest one may be for another base)
      const want = baseList(), ck = K.ckptInfo();
      let vi = D.m.versions.findIndex(v => want.includes(v.baseModel));
      if (vi < 0 && ck) vi = D.m.versions.findIndex(v => familyOfBase(v.baseModel) === ck.family);
      D.v = Math.max(0, vi);
      renderDetail();
    } catch (e) {
      panel.textContent = '';
      const h = el('div', 'cv-phead');
      h.append(el('b', '', 'Could not load the model'), el('span', 'a-spacer'), closeBtn());
      panel.append(h, el('div', 'cv-pbody', e.message));
    }
  }
  function closeBtn() { const x = el('button', 'a-x', '✕'); x.type = 'button'; x.setAttribute('aria-label', 'Close'); x.addEventListener('click', closeDetail); return x; }
  function closeDetail() { $('cvDetail').hidden = true; $('cvDetail').textContent = ''; D.m = null; }
  $('cvDetail').addEventListener('click', (e) => { if (e.target === $('cvDetail')) closeDetail(); });
  function renderDetail() {
    const m = D.m, v = m.versions[D.v] || m.versions[0];
    const box = $('cvDetail');
    box.textContent = '';
    const panel = el('div', 'cv-panel');
    const head = el('div', 'cv-phead');
    const link = el('a', 'a-btn sm', 'Open on CivitAI ↗'); link.href = m.url + (v ? `?modelVersionId=${v.id}` : ''); link.target = '_blank'; link.rel = 'noopener noreferrer';
    head.append(el('b', '', m.name), el('span', 'a-badge', m.type === 'TextualInversion' ? 'Embedding' : m.type), el('span', 'a-spacer'), link, closeBtn());
    const body = el('div', 'cv-pbody');
    // images
    const gal = el('div', 'cv-gallery');
    for (const im of (v ? v.images : [])) {
      const md = mediaBox(m, im, 240);
      md.addEventListener('click', () => { if (md.classList.contains('cv-blur')) { md.classList.remove('cv-blur'); const r = md.querySelector('.cv-reveal'); if (r) r.remove(); return; } bigView(im); });
      gal.appendChild(md);
    }
    if (!gal.childNodes.length) gal.appendChild(el('div', 'a-hint', 'No preview images.'));
    const side = el('div', 'cv-side');
    const facts = el('dl', 'a-kv');
    const fact = (k, val) => { if (val === undefined || val === null || val === '') return; facts.append(el('dt', '', k), el('dd', '', String(val))); };
    fact('Creator', m.creator && m.creator.username);
    fact('Base model', v && v.baseModel);
    const cp = v ? familyOfBase(v.baseModel) : null;
    const ck = K.ckptInfo();
    fact('Fits the checkpoint', ck && cp ? (cp === ck.family ? `yes (${ck.base || K.FAMILY[ck.family]})` : `no: the checkpoint is ${ck.base || K.FAMILY[ck.family]}`) : null);
    fact('Downloads', fmtCount(m.stats.downloads));
    fact('Likes', fmtCount(m.stats.likes));
    if (v && v.publishedAt) fact('Published', new Date(v.publishedAt).toLocaleDateString());
    if (m.local && m.local.files.length) fact('On disk', m.local.files.join(', '));
    side.appendChild(facts);
    if (m.local && m.local.update) side.appendChild(el('div', 'a-note', 'A newer version than the one on disk is available.'));
    // versions
    if (m.versions.length > 1) {
      const vs = el('div', 'cv-versions');
      m.versions.forEach((x, i) => {
        const b = el('button', '', x.name + (x.local ? ' ✓' : '')); b.type = 'button';
        b.setAttribute('aria-pressed', String(i === D.v));
        b.title = `${x.baseModel}${x.publishedAt ? ' · ' + new Date(x.publishedAt).toLocaleDateString() : ''}`;
        b.addEventListener('click', () => { D.v = i; renderDetail(); });
        vs.appendChild(b);
      });
      side.appendChild(vs);
    }
    // trigger words
    if (v && v.trainedWords.length) {
      side.appendChild(el('h4', '', 'Trigger words'));
      const w = el('div', 'cv-words-box');
      for (const t of v.trainedWords) { const b = el('button', 'cv-word', t); b.type = 'button'; b.title = 'Add to the prompt'; b.addEventListener('click', () => { GEN().insertText([t]); toast('Added to the prompt', 'ok', 1200); }); w.appendChild(b); }
      side.appendChild(w);
    }
    // files
    side.appendChild(el('h4', '', 'Files'));
    const files = el('div', 'cv-files');
    for (const f of (v ? v.files : [])) files.appendChild(fileRow(m, v, f));
    if (!files.childNodes.length) files.appendChild(el('div', 'a-hint', 'No files.'));
    side.appendChild(files);
    if (m.tags.length) {
      const tg = el('div', 'cv-tags');
      for (const t of m.tags) { const b = el('button', 'cv-tag', '#' + t); b.type = 'button'; b.title = 'Search this tag'; b.addEventListener('click', () => { CV.q.tag = t; ui.tag.value = t; closeDetail(); CV.search(true); }); tg.appendChild(b); }
      side.appendChild(tg);
    }
    const desc = el('div', 'cv-desc');
    desc.appendChild(safeHtml(v && v.description ? v.description + (m.description || '') : m.description));
    side.appendChild(desc);
    body.append(gal, side);
    panel.append(head, body);
    box.appendChild(panel);
  }
  function fileRow(m, v, f) {
    const row = el('div', 'cv-file');
    row.appendChild(el('span', 'cv-fname', f.name));
    row.appendChild(el('span', 'cv-fmeta', [fmtBytes((f.sizeKB || 0) * 1024), f.format, f.fp, f.primary ? 'primary' : ''].filter(Boolean).join(' · ')));
    const local = v.local;
    const isLora = ['LORA', 'LoCon', 'DoRA'].includes(m.type);
    if (local) {
      row.appendChild(el('span', 'a-badge ok', 'Downloaded ✓'));
      if (isLora) row.append(...afterDownload(local, v.trainedWords));
      else if (m.type === 'Checkpoint') { const b = el('button', 'a-btn sm', 'Use checkpoint'); b.type = 'button'; b.addEventListener('click', () => useCheckpoint(local)); row.appendChild(b); }
      return row;
    }
    const b = el('button', 'a-btn sm primary', 'Download'); b.type = 'button';
    if (!f.safe) { b.disabled = true; b.title = f.pickle ? 'Pickle (.ckpt/.pt) files can run code when loaded; Kiln only downloads .safetensors' : 'Kiln only downloads .safetensors files'; row.appendChild(el('span', 'a-badge bad', f.pickle ? 'pickle: not downloaded' : 'not safetensors')); }
    else if (!canManage()) { b.disabled = true; b.title = 'Downloads start on the PC running Kiln only (Settings)'; }
    else if (!['LORA', 'LoCon', 'DoRA', 'Checkpoint', 'TextualInversion', 'Upscaler'].includes(m.type)) { b.disabled = true; b.title = `Kiln can't use ${m.type} files`; }
    b.addEventListener('click', async () => {
      b.disabled = true; b.textContent = 'Queued…';
      try {
        const r = await api('/api/civitai/download', { method: 'POST', body: { versionId: v.id, fileId: f.id } });
        toast(`Downloading ${r.download.file} (${fmtBytes(r.download.size)})`, 'ok', 3000);
        refreshDownloads();
      } catch (e) {
        b.disabled = false; b.textContent = 'Download';
        toast(e.message, 'err', 9000);
      }
    });
    row.appendChild(b);
    return row;
  }
  function afterDownload(file, words) {
    const name = file.split('/').pop().replace(/\.safetensors$/i, '');
    const l = models().loras.find(x => x.file === file || x.name === name);
    const slot = el('button', 'a-btn sm', 'Add to slot'); slot.type = 'button';
    slot.addEventListener('click', () => { GEN().addToSlot(l ? l.file : file, 1); toast('Added to a LoRA slot', 'ok', 1400); });
    const ins = el('button', 'a-btn sm', 'Insert into prompt'); ins.type = 'button';
    ins.addEventListener('click', () => { GEN().toggleLoraTag(l ? l.name : name, 1); if (words && words.length) GEN().insertText(words.slice(0, 4)); toast('Inserted into the prompt', 'ok', 1400); });
    return [slot, ins];
  }
  function useCheckpoint(file) {
    const c = models().sd_models.find(x => x.file === file || x.file.endsWith('/' + file.split('/').pop()));
    if (!c) { toast('Refresh the model list first', 'err'); return; }
    const sel = $('ckptSel'); sel.value = c.file; sel.dispatchEvent(new Event('change'));
    toast(c.runnable ? `Checkpoint: ${c.name}` : `${c.name} selected, but this engine build can't run it`, c.runnable ? 'ok' : 'err', 3000);
  }
  function bigView(im) {
    const o = el('div', 'cv-lightbox');
    if (im.type === 'video') { const v = el('video'); v.src = cdn(im.url, Math.min(window.innerWidth, 900), false); v.controls = true; v.autoplay = true; v.loop = true; v.muted = true; o.appendChild(v); }
    else { const i = el('img'); i.src = cdn(im.url, Math.min(window.innerWidth, 1100), false); i.alt = ''; i.referrerPolicy = 'no-referrer'; o.appendChild(i); }
    o.addEventListener('click', () => o.remove());
    document.body.appendChild(o);
  }

  // ---------------------------------------------------------------- downloads manager
  const DL = { list: [] };
  async function refreshDownloads() {
    try { const r = await api('/api/civitai/downloads'); DL.list = r.downloads; renderDlCount(); if (!$('cvDl').hidden) renderDownloads(); } catch (_) { }
  }
  function renderDlCount() {
    const n = DL.list.filter(d => ['queued', 'downloading', 'verifying'].includes(d.status)).length;
    $('cvDlCount').textContent = n ? `(${n})` : '';
    if (ui.dl) ui.dl.textContent = n ? `Downloads (${n})` : 'Downloads';
  }
  function openDownloads() { $('cvDl').hidden = false; renderDownloads(); refreshDownloads(); }
  $('cvDl').addEventListener('click', (e) => { if (e.target === $('cvDl')) $('cvDl').hidden = true; });
  function renderDownloads() {
    const box = $('cvDl');
    box.textContent = '';
    const panel = el('div', 'cv-panel');
    const head = el('div', 'cv-phead');
    const clear = el('button', 'a-btn sm', 'Clear finished'); clear.type = 'button';
    clear.addEventListener('click', async () => { try { const r = await api('/api/civitai/downloads/clear', { method: 'POST', body: {} }); DL.list = r.downloads; renderDownloads(); renderDlCount(); } catch (e) { toast(e.message, 'err'); } });
    const x = el('button', 'a-x', '✕'); x.type = 'button'; x.setAttribute('aria-label', 'Close'); x.addEventListener('click', () => { box.hidden = true; });
    head.append(el('b', '', 'Downloads'), el('span', 'a-spacer'), clear, x);
    panel.appendChild(head);
    if (!DL.list.length) panel.appendChild(el('div', 'cv-dlrow', 'No downloads yet. Open a model and press Download.'));
    for (const d of DL.list.slice().reverse()) panel.appendChild(dlRow(d));
    box.appendChild(panel);
  }
  function dlRow(d) {
    const row = el('div', 'cv-dlrow');
    row.dataset.id = d.id;
    const top = el('div', 'cv-dltop');
    top.append(el('span', 'cv-dlname', `${d.name}${d.versionName ? ' · ' + d.versionName : ''}`), el('span', 'a-badge' + (d.status === 'done' ? ' ok' : d.status === 'error' ? ' bad' : ''), d.status));
    const act = (label, path, cls) => {
      const b = el('button', 'a-btn sm' + (cls ? ' ' + cls : ''), label); b.type = 'button';
      b.disabled = !canManage();
      b.addEventListener('click', async () => { try { const r = await api(`/api/civitai/downloads/${d.id}/${path}`, { method: 'POST', body: {} }); DL.list = r.downloads; renderDownloads(); renderDlCount(); } catch (e) { toast(e.message, 'err', 7000); } });
      top.appendChild(b);
    };
    if (['queued', 'downloading', 'verifying'].includes(d.status)) act('Cancel', 'cancel', 'danger');
    if (['error', 'cancelled'].includes(d.status)) act('Retry', 'retry');
    if (d.status === 'done' && ['LORA', 'LoCon', 'DoRA'].includes(d.type)) top.append(...afterDownload(d.rel, d.trainedWords));
    if (d.status === 'done' && d.type === 'Checkpoint') { const b = el('button', 'a-btn sm', 'Use checkpoint'); b.type = 'button'; b.addEventListener('click', () => useCheckpoint(d.rel)); top.appendChild(b); }
    row.appendChild(top);
    const pct = d.size ? Math.min(100, d.received / d.size * 100) : 0;
    const bar = el('div', 'cv-bar2'); const fill = el('div'); fill.style.width = (d.status === 'done' ? 100 : pct) + '%'; bar.appendChild(fill);
    if (d.status !== 'done') row.appendChild(bar);
    const meta = [d.file, d.size ? `${fmtBytes(d.received)} / ${fmtBytes(d.size)}` : fmtBytes(d.received)];
    if (d.status === 'downloading' && d.speed) meta.push(`${fmtBytes(d.speed)}/s`, d.eta != null ? `ETA ${fmtEta(d.eta)}` : '');
    if (d.status === 'done') meta.push('→ ' + d.rel);
    row.appendChild(el('div', 'cv-dlmeta', meta.filter(Boolean).join(' · ')));
    if (d.error) row.appendChild(el('div', 'cv-dlerr', d.error));
    return row;
  }
  const fmtEta = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`);
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    if (m.type === 'download' && m.download) {
      const i = DL.list.findIndex(x => x.id === m.download.id);
      if (i >= 0) DL.list[i] = m.download; else DL.list.push(m.download);
      renderDlCount();
      if (!$('cvDl').hidden) {
        const old = $('cvDl').querySelector(`.cv-dlrow[data-id="${m.download.id}"]`);
        if (old) old.replaceWith(dlRow(m.download)); else renderDownloads();
      }
      if (m.download.status === 'done') {
        toast(`Downloaded ${m.download.file}`, 'ok', 4000);
        if (D.m && D.m.id === m.download.modelId) openDetail(D.m.id);
      } else if (m.download.status === 'error') toast(`${m.download.name}: ${m.download.error}`, 'err', 9000);
    } else if (m.type === 'identify') {
      if (X.open && X.tab !== 'civitai' && /^(done|error)/.test(m.state)) K.loadModels();
      if (/^error/.test(m.state)) toast(`Identify ${m.file}: ${m.state.slice(7)}`, 'err', 6000);
      else if (m.state === 'done') toast(m.found ? `${m.file}: ${m.name} (${m.base})` : `${m.file}: not found on CivitAI`, m.found ? 'ok' : '', 4000);
      if (m.state === 'done' && CV.built) { renderStatus(); if (CV.q.base === 'auto' && K.ckptInfo() && m.file === K.ckptInfo().file) setTimeout(() => { CV.renderBase(); CV.search(true); }, 600); }
    } else if (m.type === 'hashing' && CV.built) {
      const a = m.hasher && m.hasher.active;
      ui.status.dataset.hash = a ? `hashing ${a.file.split(/[\\/]/).pop()} ${Math.round(a.done / Math.max(1, a.total) * 100)}%` : '';
      if (a) ui.status.lastChild && (ui.status.lastChild.textContent = ui.status.dataset.hash);
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (document.querySelector('.cv-lightbox')) document.querySelector('.cv-lightbox').remove();
    else if (!$('cvDl').hidden) $('cvDl').hidden = true;
    else if (!$('cvDetail').hidden) closeDetail();
    else if (!$('cvFull').hidden) closeFull();
  });

  window.KilnCivitai = { _card: (m) => card(m), openFull, closeFull, openDetail, openDownloads, search: (q) => { Object.assign(CV.q, q || {}); CV.search(true); }, get state() { return CV; } };
  // boot
  const G0 = GEN() && GEN().G;
  setOpen(!!(G0 && G0.xnetOpen));
  refreshDownloads();
})();
