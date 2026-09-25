/* Kiln Extensions panel: Kiln's own node packs (Kiln/extensions/<pack>/, docs/NODE_API.md).
   List packs (version, author, nodes, load errors), enable / disable, install from a git URL or a
   local folder, update (git pull), uninstall, reload, and scaffold a new pack. */
(() => {
  'use strict';
  const K = window.Kiln;
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const btn = (cls, text, onClick, title) => { const b = el('button', cls, text); b.type = 'button'; if (title) b.title = title; if (onClick) b.addEventListener('click', onClick); return b; };
  const S = { built: false, open: false, data: null, busy: '', createOpen: false };
  let root, list, sub, note, src, installBtn, createBox, busyLine;

  // ---------------------------------------------------------------- top-bar button
  function renderButton(sum) {
    const b = $('packBtn');
    if (!b || !sum) return;
    $('packCount').textContent = sum.nodes ? String(sum.packs) : '';
    b.classList.toggle('has-errors', !!sum.errors);
    b.title = `Extensions: ${sum.packs} pack(s), ${sum.nodes} node(s)${sum.errors ? `, ${sum.errors} with errors` : ''}`;
  }

  // ---------------------------------------------------------------- panel
  function build() {
    if (S.built) return;
    S.built = true;
    root = el('div', 'xp');
    root.id = 'packPanel';
    root.hidden = true;
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-label', 'Extensions');
    const panel = el('div', 'xp-panel');

    const head = el('div', 'xp-head');
    const titles = el('div', 'xp-titles');
    titles.append(el('b', 'xp-title', 'Extensions'), sub = el('span', 'xp-sub', ''));
    head.append(titles, el('span', 'xp-spacer'),
      btn('xp-btn', 'Reload', reload, 'Re-read every pack from disk (after editing a pack\'s files)'),
      btn('xp-btn', 'Create pack…', toggleCreate, 'Start a new pack from a template'),
      btn('xp-x', '✕', close, 'Close'));
    head.lastChild.setAttribute('aria-label', 'Close');

    note = el('div', 'xp-note');
    note.hidden = true;

    createBox = el('form', 'xp-create');
    createBox.hidden = true;
    const f = (name, label, ph) => { const w = el('label', 'xp-field'); const i = el('input'); i.name = name; i.placeholder = ph; i.autocomplete = 'off'; w.append(el('span', '', label), i); return w; };
    createBox.append(f('name', 'Name', 'my-nodes'), f('author', 'Author', 'you'), f('description', 'Description', 'What the nodes do'));
    const cb = el('button', 'xp-btn primary', 'Create');
    cb.type = 'submit';
    createBox.append(cb, el('p', 'xp-hint', 'Writes extensions/<name>/kiln.json and nodes.js with one example node. Edit nodes.js, then press Reload.'));
    createBox.addEventListener('submit', (e) => { e.preventDefault(); create(); });

    const bar = el('form', 'xp-install');
    src = el('input', 'xp-src');
    src.placeholder = 'Git URL (https://github.com/…) or a folder path on this PC';
    src.autocomplete = 'off';
    src.spellcheck = false;
    installBtn = el('button', 'xp-btn primary', 'Install');
    installBtn.type = 'submit';
    bar.append(src, installBtn);
    bar.addEventListener('submit', (e) => { e.preventDefault(); install(); });
    busyLine = el('div', 'xp-busy');
    busyLine.hidden = true;

    list = el('div', 'xp-list');
    const foot = el('div', 'xp-foot');
    foot.append(el('span', '', 'Packs are JavaScript that runs inside the Kiln server with full access to this PC. Only install packs you trust. Writing one: docs/NODE_API.md.'));

    panel.append(head, note, createBox, bar, busyLine, list, foot);
    root.appendChild(panel);
    root.addEventListener('click', (e) => { if (e.target === root) close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.open) close(); });
    document.body.appendChild(root);
  }

  async function open() {
    build();
    S.open = true;
    root.hidden = false;
    await refresh();
  }
  function close() {
    if (!S.built) return;
    S.open = false;
    root.hidden = true;
  }
  function toggleCreate() {
    S.createOpen = !S.createOpen;
    createBox.hidden = !S.createOpen;
    if (S.createOpen) createBox.querySelector('input').focus();
  }

  async function refresh() {
    try {
      S.data = await K.api('/api/extensions');
      render();
      renderButton(S.data.summary);
    } catch (e) {
      list.textContent = '';
      list.appendChild(el('div', 'xp-empty', 'Could not load the pack list: ' + e.message));
    }
  }

  function render() {
    const d = S.data;
    if (!d || !S.built) return;
    sub.textContent = `${d.summary.packs} pack${d.summary.packs === 1 ? '' : 's'} · ${d.summary.nodes} node${d.summary.nodes === 1 ? '' : 's'}${d.summary.errors ? ` · ${d.summary.errors} with errors` : ''}`;
    note.hidden = d.manage;
    note.textContent = d.manage ? '' : 'Installing, removing and enabling packs is only possible on the PC running Kiln (or start the server with KILN_EXT_LAN=1).';
    const locked = !d.manage || !!S.busy || !!d.busy;
    for (const b of root.querySelectorAll('.xp-head .xp-btn, .xp-install button, .xp-create button')) b.disabled = locked;
    src.disabled = locked;
    busyLine.hidden = !(S.busy || d.busy);
    busyLine.textContent = S.busy || (d.busy ? `Working: ${d.busy}…` : '');
    list.textContent = '';
    if (!d.packs.length) {
      const e = el('div', 'xp-empty');
      e.append(el('b', '', 'No packs installed.'), el('span', '', ` Install one above, create your own, or drop a folder into ${d.dir}.`));
      list.appendChild(e);
      return;
    }
    for (const p of d.packs) list.appendChild(row(p, locked));
  }

  function row(p, locked) {
    const r = el('div', 'xp-row');
    r.dataset.pack = p.dir;
    const nodeErrs = Object.entries(p.node_errors || {});
    r.classList.toggle('off', !p.enabled);
    r.classList.toggle('bad', !!p.error);
    const main = el('div', 'xp-main');
    const name = el('div', 'xp-name');
    name.append(el('b', '', p.name));
    if (p.version) name.append(el('span', 'xp-ver', 'v' + p.version));
    if (p.author) name.append(el('span', 'xp-by', 'by ' + p.author));
    if (!p.enabled) name.append(el('span', 'xp-tag', 'disabled'));
    else if (p.error) name.append(el('span', 'xp-tag bad', 'failed to load'));
    else name.append(el('span', 'xp-tag on', `${p.nodes.length} node${p.nodes.length === 1 ? '' : 's'}`));
    if (p.ui) name.append(el('span', 'xp-tag', 'ui'));
    if (p.git) name.append(el('span', 'xp-tag', 'git'));
    main.append(name);
    if (p.description) main.append(el('div', 'xp-desc', p.description));
    if (p.error) main.append(el('div', 'xp-err', p.error));
    for (const [cls, msg] of nodeErrs) main.append(el('div', 'xp-err', `${cls}: ${msg}`));
    if (p.nodes.length) {
      const chips = el('div', 'xp-nodes');
      for (const n of p.nodes) { const c = el('span', 'xp-chip', n.display_name); c.title = `${n.class} · ${n.category}`; chips.appendChild(c); }
      main.append(chips);
    }
    const meta = el('div', 'xp-path', p.path);
    if (p.url) { const a = el('a', '', p.url); a.href = p.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; meta.append(' · ', a); }
    main.append(meta);
    const acts = el('div', 'xp-acts');
    const act = (label, fn, title) => { const b = btn('xp-btn small', label, fn, title); b.disabled = locked; acts.appendChild(b); return b; };
    act(p.enabled ? 'Disable' : 'Enable', () => op(`${p.enabled ? 'Disabling' : 'Enabling'} ${p.name}…`, `/api/extensions/${encodeURIComponent(p.dir)}/${p.enabled ? 'disable' : 'enable'}`, 'POST'));
    if (p.git) act('Update', () => op(`Updating ${p.name} (git pull)…`, `/api/extensions/${encodeURIComponent(p.dir)}/update`, 'POST'), 'git pull --ff-only');
    const un = act('Uninstall', () => {
      if (!confirm(`Uninstall "${p.name}"? This deletes ${p.path}.`)) return;
      op(`Removing ${p.name}…`, `/api/extensions/${encodeURIComponent(p.dir)}`, 'DELETE');
    });
    un.classList.add('danger');
    r.append(main, acts);
    return r;
  }

  async function op(label, path, method, body) {
    S.busy = label;
    render();
    try {
      const r = await K.api(path, { method, body: method === 'POST' ? (body || {}) : undefined });
      S.data = r;
      if (r.pack && r.pack.error) K.toast(`${r.pack.name}: ${r.pack.error}`, 'err', 7000);
      return r;
    } catch (e) {
      K.toast(e.message, 'err', 7000);
      return null;
    } finally {
      S.busy = '';
      render();
      if (S.data) renderButton(S.data.summary);
    }
  }
  async function reload() {
    const r = await op('Reloading packs…', '/api/extensions/reload', 'POST');
    if (r) K.toast(`Reloaded: ${r.summary.packs} pack(s), ${r.summary.nodes} node(s)${r.summary.errors ? `, ${r.summary.errors} with errors` : ''}`, r.summary.errors ? 'err' : '');
  }
  async function install() {
    const source = src.value.trim();
    if (!source) { src.focus(); return; }
    const git = /^(https?:\/\/|git@|ssh:\/\/)/i.test(source);
    const r = await op(git ? `Cloning ${source}…` : `Copying ${source}…`, '/api/extensions/install', 'POST', { source });
    if (r && r.pack) {
      src.value = '';
      K.toast(`Installed ${r.pack.name}${r.pack.nodes.length ? `: ${r.pack.nodes.length} node(s)` : ''}`, r.pack.error ? 'err' : 'ok');
    }
  }
  async function create() {
    const v = (n) => createBox.querySelector(`input[name=${n}]`).value.trim();
    const body = { name: v('name'), author: v('author'), description: v('description') };
    if (!body.name) { createBox.querySelector('input[name=name]').focus(); return; }
    const r = await op(`Creating ${body.name}…`, '/api/extensions/create', 'POST', body);
    if (r && r.pack) {
      for (const i of createBox.querySelectorAll('input')) i.value = '';
      toggleCreate();
      K.toast(`Created ${r.pack.path.replace(/\\/g, '/')}/nodes.js with an example node. Edit it, then Reload.`, 'ok', 7000);
    }
  }

  // ---------------------------------------------------------------- live updates
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    if (m.type === 'hello' && m.server && m.server.extensions) renderButton(m.server.extensions);
    else if (m.type === 'extensions') {
      renderButton(m.extensions);
      if (S.open && !S.busy) refresh();
    }
  });
  const b = $('packBtn');
  if (b) b.addEventListener('click', () => (S.open ? close() : open()));
  window.KilnExtensions = { open, close, refresh };
})();
