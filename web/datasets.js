/* Kiln Datasets tab: folders of training images with .txt captions. A thumbnail grid with selection, a tag editor for
   one image, bulk edits (trigger tags in front, add / remove / replace), auto-tagging with a WD tagger on the engine,
   and the dataset's tag list. The "Grab from boorus" view (grab.js) downloads into the dataset shown here. */
(() => {
  'use strict';
  const K = window.Kiln;
  const { $, el, api, toast, lsGet, lsSet } = K;
  const LS = 'kiln.datasets.v1';
  const D = Object.assign({ dir: '', view: 'images', gen: 0.35, char: 0.85, mode: 'append', keep: 0, exclude: '', rating: false, unders: false, tagger: '' }, lsGet(LS) || {});
  const save = K.debounce(() => lsSet(LS, D), 250);
  const st = { list: [], items: [], byFile: new Map(), tags: [], sel: new Set(), anchor: null, focus: null, taggers: [], tagJob: null };
  const enc = encodeURIComponent;
  const q = (dir) => 'dir=' + enc(dir);
  const split = (c) => String(c || '').split(',').map(t => t.trim()).filter(Boolean);
  const key = (t) => t.toLowerCase().replace(/_/g, ' ').replace(/\\([()])/g, '$1').replace(/\s+/g, ' ');
  window.KilnDatasets = { current: () => D.dir, reload: () => loadItems(), select: (dir) => pick(dir) };

  // ---------------------------------------------------------------- dataset list
  async function loadList(keepDir) {
    try {
      const r = await api('/api/datasets');
      st.list = r.list;
      const sel = $('dsPick');
      sel.textContent = '';
      if (!r.list.length) { const o = el('option', '', 'No datasets yet: New… or open a folder'); o.value = ''; sel.appendChild(o); }
      for (const d of r.list) {
        const o = el('option', '', `${d.name}${d.own ? '' : '  (' + d.dir + ')'}${d.missing ? ' — missing' : ` · ${d.images}`}`);
        o.value = d.dir;
        sel.appendChild(o);
      }
      const want = keepDir || D.dir;
      if (r.list.some(d => d.dir === want)) sel.value = want;
      else if (r.list.length) sel.value = r.list[0].dir;
      if (sel.value !== D.dir || !st.items.length) pick(sel.value);
    } catch (e) { toast('Datasets: ' + e.message, 'err'); }
  }
  function pick(dir) {
    D.dir = dir || '';
    save();
    $('dsPick').value = D.dir;
    st.sel.clear();
    st.focus = null;
    st.anchor = null;
    $('dsTrain').disabled = !D.dir;
    loadItems();
    window.dispatchEvent(new CustomEvent('kiln:dataset', { detail: { dir: D.dir } }));
  }
  $('dsPick').addEventListener('change', () => pick($('dsPick').value));
  $('dsNew').addEventListener('click', async () => {
    const name = prompt('Name of the new dataset (a folder in Kiln/datasets):');
    if (!name) return;
    try { const d = await api('/api/datasets/create', { method: 'POST', body: { name } }); await loadList(d.dir); toast('Created ' + d.name, 'ok'); }
    catch (e) { toast(e.message, 'err', 6000); }
  });
  async function openFolder() {
    const dir = $('dsOpenPath').value.trim();
    if (!dir) return;
    try { const d = await api('/api/datasets/open', { method: 'POST', body: { dir } }); $('dsOpenPath').value = ''; await loadList(d.dir); }
    catch (e) { toast(e.message, 'err', 7000); }
  }
  $('dsOpen').addEventListener('click', openFolder);
  $('dsOpenPath').addEventListener('keydown', (e) => { if (e.key === 'Enter') openFolder(); });
  $('dsTrain').addEventListener('click', () => {
    if (!D.dir) return;
    K.setTab('train');
    if (window.KilnTrain) window.KilnTrain.useDataset(D.dir);
  });

  // ---------------------------------------------------------------- views
  function setView(v) {
    D.view = v === 'grab' && window.KilnGrab ? 'grab' : 'images';
    save();
    for (const b of document.querySelectorAll('.ds-views button')) b.setAttribute('aria-checked', String(b.dataset.v === D.view));
    $('dsImages').hidden = D.view !== 'images';
    $('dsGrab').hidden = D.view !== 'grab';
    if (D.view === 'images' && D.dir) loadItems();
    window.dispatchEvent(new CustomEvent('kiln:dsview', { detail: { view: D.view } }));
  }
  for (const b of document.querySelectorAll('.ds-views button')) b.addEventListener('click', () => setView(b.dataset.v));

  // ---------------------------------------------------------------- items
  async function loadItems() {
    if (!D.dir) { st.items = []; st.byFile.clear(); st.tags = []; render(); return; }
    try {
      const r = await api('/api/datasets/items?' + q(D.dir));
      st.items = r.items;
      st.byFile = new Map(r.items.map(i => [i.file, i]));
      st.tags = r.tags;
      for (const f of [...st.sel]) if (!st.byFile.has(f)) st.sel.delete(f);
      if (st.focus && !st.byFile.has(st.focus)) st.focus = null;
      render();
    } catch (e) {
      st.items = [];
      st.byFile.clear();
      render();
      $('dsEmpty').hidden = false;
      $('dsEmpty').textContent = e.message;
    }
  }
  function recount() {  // the tag list from the captions in memory
    const m = new Map();
    for (const it of st.items) for (const t of split(it.caption)) { const k = key(t); const e = m.get(k) || { tag: t, n: 0 }; e.n++; m.set(k, e); }
    st.tags = [...m.values()].sort((a, b) => b.n - a.n || a.tag.localeCompare(b.tag)).map(e => [e.tag, e.n]);
  }

  // filter: "a, b" = has both; "-c" = doesn't have c
  function visible() {
    const f = split($('dsFilter').value), show = $('dsShow').value;
    const need = f.filter(t => !t.startsWith('-')).map(key), not = f.filter(t => t.startsWith('-')).map(t => key(t.slice(1))).filter(Boolean);
    return st.items.filter(it => {
      if (show === 'untagged' && it.caption) return false;
      if (show === 'selected' && !st.sel.has(it.file)) return false;
      if (!need.length && !not.length) return true;
      const have = new Set(split(it.caption).map(key));
      return need.every(t => have.has(t)) && !not.some(t => have.has(t));
    });
  }

  // ---------------------------------------------------------------- grid
  const cards = new Map();  // file -> card element (rebuilt on render)
  function render() {
    const list = visible();
    const grid = $('dsGrid');
    grid.textContent = '';
    cards.clear();
    const frag = document.createDocumentFragment();
    for (const it of list) frag.appendChild(card(it));
    grid.appendChild(frag);
    $('dsEmpty').hidden = !!list.length || !D.dir;
    $('dsEmpty').textContent = !st.items.length ? 'No images here yet. Grab some from the boorus, or copy images (with .txt captions) into the folder.' : 'No image matches the filter.';
    const cap = st.items.filter(i => i.caption).length;
    $('dsSum').textContent = D.dir ? `${st.items.length} images · ${cap} captioned` : '';
    renderCount(list.length);
    renderEditor();
    renderTags();
    renderScope();
  }
  function renderCount(shown) {
    $('dsCount').textContent = `${shown == null ? visible().length : shown} shown · ${st.sel.size} selected`;
  }
  function card(it) {
    const c = el('div', 'ds-card');
    c.dataset.file = it.file;
    c.classList.toggle('sel', st.sel.has(it.file));
    c.classList.toggle('focus', st.focus === it.file);
    c.classList.toggle('untagged', !it.caption);
    const im = el('img');
    im.alt = it.file;
    im.loading = 'lazy';
    im.decoding = 'async';
    thumb(im, it);
    const tick = el('span', 'ds-tick');
    const cap = el('div', 'ds-cap', it.caption || 'no caption');
    c.title = it.file + (it.caption ? '\n' + it.caption : '');
    c.append(im, tick, cap);
    cards.set(it.file, c);
    return c;
  }
  function updateCard(file) {
    const c = cards.get(file), it = st.byFile.get(file);
    if (!c || !it) return;
    c.classList.toggle('untagged', !it.caption);
    c.querySelector('.ds-cap').textContent = it.caption || 'no caption';
    c.title = it.file + (it.caption ? '\n' + it.caption : '');
  }
  function paintSel() {
    for (const [f, c] of cards) { c.classList.toggle('sel', st.sel.has(f)); c.classList.toggle('focus', st.focus === f); }
    renderCount();
    renderEditor();
    renderScope();
  }
  $('dsGrid').addEventListener('click', (e) => {
    const c = e.target.closest('.ds-card');
    if (!c) return;
    const f = c.dataset.file;
    if (e.shiftKey && st.anchor) {  // a range in the shown order
      const order = visible().map(i => i.file), a = order.indexOf(st.anchor), b = order.indexOf(f);
      if (a >= 0 && b >= 0) for (let i = Math.min(a, b); i <= Math.max(a, b); i++) st.sel.add(order[i]);
    } else if (e.ctrlKey || e.metaKey || e.target.classList.contains('ds-tick')) {
      if (st.sel.has(f)) st.sel.delete(f); else st.sel.add(f);
      st.anchor = f;
    } else {
      st.sel.clear();
      st.sel.add(f);
      st.anchor = f;
    }
    st.focus = st.sel.has(f) ? f : (st.sel.size ? [...st.sel][st.sel.size - 1] : null);
    paintSel();
  });
  $('dsSelAll').addEventListener('click', () => { for (const it of visible()) st.sel.add(it.file); paintSel(); });
  $('dsSelNone').addEventListener('click', () => { st.sel.clear(); st.focus = null; paintSel(); });
  $('dsFilter').addEventListener('input', K.debounce(render, 200));
  $('dsShow').addEventListener('change', render);
  document.addEventListener('keydown', (e) => {
    if (K.S.tab !== 'datasets' || D.view !== 'images' || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    if ((e.ctrlKey || e.metaKey) && e.key === 'a') { e.preventDefault(); for (const it of visible()) st.sel.add(it.file); paintSel(); }
    if (e.key === 'Escape') { st.sel.clear(); st.focus = null; paintSel(); }
  });

  // ---------------------------------------------------------------- thumbnails: cached on the server, made here
  const pending = [];
  let busy = 0;
  function thumb(im, it) {
    const tq = `/api/datasets/thumb?${q(D.dir)}&f=${enc(it.file)}&v=${it.mtime}`;
    im.onerror = () => { im.onerror = null; pending.push({ im, it, dir: D.dir }); pump(); };
    im.src = tq;
  }
  function pump() {
    while (busy < 3 && pending.length) {
      const job = pending.shift();
      if (!job.im.isConnected) continue;
      busy++;
      make(job).finally(() => { busy--; pump(); });
    }
  }
  async function make({ im, it, dir }) {
    const src = new Image();
    src.decoding = 'async';
    src.src = `/api/datasets/image?${q(dir)}&f=${enc(it.file)}&v=${it.mtime}`;
    try { await src.decode(); } catch (_) { im.classList.add('bad'); im.alt = 'cannot show ' + it.file; return; }
    it.w = src.naturalWidth;
    it.h = src.naturalHeight;
    const s = Math.min(1, 320 / Math.max(src.naturalWidth, src.naturalHeight));
    const cv = document.createElement('canvas');
    cv.width = Math.max(1, Math.round(src.naturalWidth * s));
    cv.height = Math.max(1, Math.round(src.naturalHeight * s));
    const g = cv.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, cv.width, cv.height);
    g.imageSmoothingQuality = 'high';
    g.drawImage(src, 0, 0, cv.width, cv.height);
    const blob = await new Promise(r => cv.toBlob(r, 'image/jpeg', 0.85));
    if (!blob) return;
    im.src = URL.createObjectURL(blob);
    fetch(`/api/datasets/thumb?${q(dir)}&f=${enc(it.file)}`, { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob }).catch(() => { });
  }

  // ---------------------------------------------------------------- editor (one image) / selection summary
  function renderEditor() {
    const one = st.focus && st.sel.size <= 1 ? st.byFile.get(st.focus) : null;
    $('dsEditNone').hidden = !!one || st.sel.size > 1;
    $('dsEditOne').hidden = !one;
    $('dsEditMany').hidden = st.sel.size <= 1;
    if (st.sel.size > 1) {
      // the tags every selected image has, and how common the others are
      const n = st.sel.size, m = new Map();
      for (const f of st.sel) for (const t of split((st.byFile.get(f) || {}).caption)) { const k = key(t); const e = m.get(k) || { tag: t, n: 0 }; e.n++; m.set(k, e); }
      const all = [...m.values()].sort((a, b) => b.n - a.n);
      const box = $('dsEditMany');
      box.textContent = '';
      box.appendChild(el('div', 'ds-edit-name', `${n} images selected. Bulk edits and auto-tag apply to them. Click a tag to remove it from all of them.`));
      const chips = el('div', 'ds-chips');
      for (const e of all.slice(0, 120)) {
        const c = el('span', 'ds-chip' + (e.n === n ? ' all' : ''), e.tag);
        c.title = `${e.n} of ${n} — click to remove from the selected images`;
        c.appendChild(el('span', 'ds-chip-n', String(e.n)));
        c.addEventListener('click', () => bulk('remove', [e.tag]));
        chips.appendChild(c);
      }
      box.appendChild(chips);
    }
    if (!one) return;
    $('dsEditImg').src = `/api/datasets/image?${q(D.dir)}&f=${enc(one.file)}&v=${one.mtime}`;
    $('dsEditName').textContent = one.file + (one.w ? ` · ${one.w}×${one.h}` : '');
    renderChips(one);
  }
  function renderChips(it) {
    const box = $('dsChips');
    box.textContent = '';
    const tags = split(it.caption);
    tags.forEach((t, i) => {
      const c = el('span', 'ds-chip', t);
      c.draggable = true;
      c.dataset.i = i;
      const x = el('button', 'ds-chip-x', '×');
      x.type = 'button';
      x.title = 'Remove';
      x.addEventListener('click', () => setCaption(it, tags.filter((_, j) => j !== i)));
      c.appendChild(x);
      c.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', String(i)); });
      c.addEventListener('dragover', (e) => e.preventDefault());
      c.addEventListener('drop', (e) => {  // drag a tag onto another to move it there
        e.preventDefault();
        const from = Number(e.dataTransfer.getData('text/plain')), to = i;
        if (from === to || isNaN(from)) return;
        const t2 = tags.slice();
        const [m] = t2.splice(from, 1);
        t2.splice(to, 0, m);
        setCaption(it, t2);
      });
      box.appendChild(c);
    });
    if (!tags.length) box.appendChild(el('span', 'a-hint', 'No tags yet.'));
    $('dsRaw').value = it.caption;
  }
  async function setCaption(it, tags) {
    const caption = tags.join(', ');
    try {
      const r = await api('/api/datasets/caption', { method: 'POST', body: { dir: D.dir, file: it.file, caption } });
      it.caption = r.caption;
      updateCard(it.file);
      recount();
      renderTags();
      if (st.focus === it.file) renderChips(it);
    } catch (e) { toast('Could not save: ' + e.message, 'err'); }
  }
  $('dsAddTag').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ',') return;
    e.preventDefault();
    const it = st.byFile.get(st.focus);
    const add = split($('dsAddTag').value);
    if (!it || !add.length) return;
    const have = split(it.caption), seen = new Set(have.map(key));
    setCaption(it, [...have, ...add.filter(t => !seen.has(key(t)))]);
    $('dsAddTag').value = '';
  });
  $('dsRawBtn').addEventListener('click', () => { $('dsRaw').hidden = !$('dsRaw').hidden; if (!$('dsRaw').hidden) $('dsRaw').focus(); });
  $('dsRaw').addEventListener('change', () => { const it = st.byFile.get(st.focus); if (it) setCaption(it, split($('dsRaw').value)); });

  // ---------------------------------------------------------------- bulk edits
  function scopeFiles() { return st.sel.size ? [...st.sel] : null; }
  function renderScope() {
    const s = st.sel.size ? `— ${st.sel.size} selected` : `— all ${st.items.length} images`;
    $('dsScope').textContent = s;
    $('dsTagScope').textContent = s;
    $('dsRemove').disabled = !st.sel.size;
    $('dsTagGo').textContent = `Tag ${st.sel.size || st.items.length} image${(st.sel.size || st.items.length) === 1 ? '' : 's'}`;
  }
  async function bulk(op, tags, to) {
    if (!D.dir) return;
    try {
      const r = await api('/api/datasets/bulk', { method: 'POST', body: { dir: D.dir, files: scopeFiles(), op, tags, to } });
      for (const c of r.changed) { const it = st.byFile.get(c.file); if (it) { it.caption = c.caption; updateCard(c.file); } }
      recount();
      renderTags();
      renderEditor();
      toast(`${r.changed.length} caption${r.changed.length === 1 ? '' : 's'} changed`, 'ok', 2000);
    } catch (e) { toast(e.message, 'err', 6000); }
  }
  $('dsTrigGo').addEventListener('click', () => { const t = split($('dsTrig').value); if (t.length) bulk('prepend', t); });
  $('dsAddGo').addEventListener('click', () => { const t = split($('dsAddMany').value); if (t.length) bulk('append', t); });
  $('dsDelGo').addEventListener('click', () => { const t = split($('dsDelMany').value); if (t.length) bulk('remove', t); });
  $('dsRepGo').addEventListener('click', () => {
    const from = split($('dsRepFrom').value);
    if (!from.length) return;
    bulk('replace', from, $('dsRepTo').value);
  });
  $('dsRemove').addEventListener('click', async () => {
    const files = [...st.sel];
    if (!files.length || !confirm(`Move ${files.length} image${files.length === 1 ? '' : 's'} (and captions) into this dataset's _removed folder?`)) return;
    try {
      const r = await api('/api/datasets/remove', { method: 'POST', body: { dir: D.dir, files } });
      toast(`Removed ${r.removed} from the dataset (kept in _removed)`, 'ok');
      st.sel.clear();
      st.focus = null;
      loadItems();
    } catch (e) { toast(e.message, 'err'); }
  });

  // ---------------------------------------------------------------- tag list
  function renderTags() {
    const find = key($('dsTagFind').value || '');
    const box = $('dsTags');
    box.textContent = '';
    const dl = $('dsTagList');
    dl.textContent = '';
    for (const [t] of st.tags.slice(0, 800)) { const o = el('option'); o.value = t; dl.appendChild(o); }
    for (const [t, n] of st.tags) {
      if (find && !key(t).includes(find)) continue;
      const row = el('div', 'ds-tag');
      const name = el('button', 'ds-tag-name', t);
      name.type = 'button';
      name.title = 'Show the images with this tag';
      name.addEventListener('click', () => { $('dsFilter').value = t; render(); });
      const x = el('button', 'ds-tag-x', '×');
      x.type = 'button';
      x.title = 'Remove from every image (or the selected ones)';
      x.addEventListener('click', () => { if (confirm(`Remove "${t}" from ${st.sel.size ? st.sel.size + ' selected' : 'all'} images?`)) bulk('remove', [t]); });
      row.append(name, el('span', 'ds-tag-n', String(n)), x);
      box.appendChild(row);
    }
    if (!st.tags.length) box.appendChild(el('div', 'a-hint', 'No captions yet.'));
  }
  $('dsTagFind').addEventListener('input', K.debounce(renderTags, 150));

  // ---------------------------------------------------------------- auto-tag
  const tagFields = [['dsGen', 'gen', Number], ['dsChar', 'char', Number], ['dsMode', 'mode', String], ['dsKeep', 'keep', Number], ['dsExclude', 'exclude', String], ['dsRating', 'rating', Boolean], ['dsUnders', 'unders', Boolean]];
  for (const [id, k, conv] of tagFields) {
    const e = $(id);
    if (e.type === 'checkbox') e.checked = !!D[k]; else e.value = D[k];
    e.addEventListener('change', () => { D[k] = e.type === 'checkbox' ? e.checked : conv(e.value); save(); $('dsKeepRow').hidden = D.mode !== 'replace'; });
  }
  $('dsKeepRow').hidden = D.mode !== 'replace';
  async function loadTaggers() {
    try {
      const r = await api('/api/taggers');
      st.taggers = r.taggers;
      const sel = $('dsTagger');
      const cur = D.tagger || (r.taggers.find(t => t.installed) || r.taggers[0] || {}).id;
      sel.textContent = '';
      for (const t of r.taggers) {
        const o = el('option', '', `${t.name} — ${t.note}${t.installed ? '' : ` (download ${(t.size / 1e9).toFixed(2)} GB)`}`);
        o.value = t.id;
        sel.appendChild(o);
      }
      sel.value = cur;
      D.tagger = sel.value;
      renderTagger();
    } catch (e) { $('dsTaggerDl').textContent = e.message; }
  }
  function renderTagger() {
    const t = st.taggers.find(x => x.id === D.tagger);
    const box = $('dsTaggerDl');
    box.textContent = '';
    if (!t) return;
    if (t.installed) { $('dsTagGo').disabled = !!st.tagJob || !st.items.length; return; }
    $('dsTagGo').disabled = true;
    const p = t.progress;
    if (t.downloading && p) box.textContent = `Downloading ${Math.round(100 * p.done / Math.max(1, p.total))}% of ${(p.total / 1e9).toFixed(2)} GB…`;
    else {
      const b = el('button', 'a-btn sm', `Download ${t.name} (${(t.size / 1e9).toFixed(2)} GB)`);
      b.type = 'button';
      b.addEventListener('click', async () => {
        try { await api('/api/taggers/download', { method: 'POST', body: { id: t.id } }); t.downloading = true; t.progress = { done: 0, total: t.size }; renderTagger(); }
        catch (e) { toast(e.message, 'err'); }
      });
      box.appendChild(b);
      if (p && p.error) box.appendChild(el('div', 't-warn', 'Last try failed: ' + p.error));
    }
  }
  $('dsTagger').addEventListener('change', () => { D.tagger = $('dsTagger').value; save(); renderTagger(); });
  $('dsTagGo').addEventListener('click', async () => {
    if (!D.dir) return;
    const files = scopeFiles();
    try {
      const r = await api('/api/datasets/autotag', { method: 'POST', body: {
        dir: D.dir, files, model: D.tagger, general: D.gen, character: D.char, mode: D.mode, keep_first: D.keep,
        exclude: D.exclude, rating: D.rating, underscores: D.unders,
      } });
      st.tagJob = { id: r.id, done: 0, of: r.images, failed: 0 };
      $('dsTagGo').disabled = true;
      $('dsTagStatus').textContent = `Queued: ${r.images} images. It runs after anything already on the GPU (renders, training).`;
    } catch (e) { toast('Auto-tag: ' + e.message, 'err', 6000); }
  });

  // ---------------------------------------------------------------- live updates
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    if (m.type === 'tagger') {
      const t = st.taggers.find(x => x.id === m.download.id);
      if (t) {
        t.progress = m.download;
        t.downloading = !m.download.finished && !m.download.error;
        if (m.download.finished) loadTaggers();
        else renderTagger();
      }
    } else if (m.type === 'autotag') {
      if (st.tagJob && st.tagJob.id === m.id) {
        st.tagJob.done = m.done;
        st.tagJob.failed = m.failed;
        $('dsTagStatus').textContent = `Tagging ${m.done} / ${m.of}` + (m.failed ? ` · ${m.failed} failed` : '');
      }
      if (m.dir === D.dir && m.caption != null) {
        const it = st.byFile.get(m.file);
        if (it) {
          it.caption = m.caption;
          updateCard(m.file);
          if (st.focus === m.file) renderChips(it);
          recountSoon();
        }
      }
    } else if (m.type === 'job' && m.job && m.job.kind === 'tag' && st.tagJob && st.tagJob.id === m.job.id) {
      const j = m.job;
      if (j.status === 'running') $('dsTagStatus').textContent = j.loading ? 'Loading the tagger…' : `Tagging ${j.step} / ${j.of}`;
      if (['done', 'error', 'cancelled'].includes(j.status)) {
        const t = j.tag || {};
        $('dsTagStatus').textContent = j.status === 'done' ? `Done: ${t.changed || 0} captions written` + (t.failed && t.failed.length ? `; ${t.failed.length} failed (${t.failed.slice(0, 3).join('; ')})` : '')
          : j.status === 'cancelled' ? 'Stopped.' : 'Failed: ' + (j.error || '');
        st.tagJob = null;
        renderTagger();
        if (j.status === 'done' && (t.dir || '') === D.dir) loadItems();
      }
    }
  });
  const recountSoon = K.debounce(() => { recount(); renderTags(); }, 400);

  // a booru download landed in this dataset
  window.addEventListener('kiln:dsfile', () => { if (D.view === 'images') loadItemsSoon(); });
  const loadItemsSoon = K.debounce(() => loadItems(), 800);

  K.on('tab', ({ tab }) => { if (tab === 'datasets') { loadList(); loadTaggers(); } });
  setView(D.view);
  if (K.S.tab === 'datasets') { loadList(); loadTaggers(); }
})();
