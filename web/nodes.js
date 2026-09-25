/* Kiln Nodes mode: a ComfyUI-style graph editor. Vanilla JS, DOM node cards + SVG links.
   Graphs run as ComfyUI API-format workflows (POST /api/graph). */
(() => {
  'use strict';

  const K = window.Kiln; // helpers exported by app.js
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const SVGNS = 'http://www.w3.org/2000/svg';
  const LS_DOC = 'kiln.graph.v1';
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  // ---------------------------------------------------------------- catalog
  let CAT = null;                  // filled /api/nodes
  let EXAMPLES = [];               // /api/graph/examples
  const WIDGET_T = new Set(['INT', 'FLOAT', 'STRING', 'BOOLEAN']);
  const CONTROL = ['fixed', 'increment', 'decrement', 'randomize'];
  const PRIMITIVES = { PrimitiveInt: 1, PrimitiveFloat: 1, PrimitiveString: 1, PrimitiveStringMultiline: 1, PrimitiveBoolean: 1, PrimitiveNode: 1 };
  const VIRTUAL = { Reroute: 1, SetNode: 1, GetNode: 1, ...PRIMITIVES };
  const NOTE_TYPES = { Note: 1, MarkdownNote: 1, '__note': 1 };
  const TYPE_COLORS = {
    MODEL: '#b39ddb', CLIP: '#ffd500', VAE: '#ff6e6e', CONDITIONING: '#ffa931', LATENT: '#ff9cf9', IMAGE: '#64b5f6', MASK: '#81c784',
    UPSCALE_MODEL: '#4dd0c8', BBOX_DETECTOR: '#e6a85c', SEGM_DETECTOR: '#c98c58', SAM_MODEL: '#9fa8ff', DETAILER_PIPE: '#aab2bd', DETAILER_HOOK: '#aab2bd',
    INT: '#7fd49b', FLOAT: '#9be0b4', STRING: '#8ec9ff', BOOLEAN: '#f4a6c8', COMBO: '#c6c9d0', '*': '#c6c9d0',
  };
  const typeColor = (t) => TYPE_COLORS[t] || '#8b909a';
  const packOf = (type) => (CAT && CAT[type] && CAT[type].kiln_pack) || '';

  // ---------------------------------------------------------------- pack UI modules (ui.js, docs/NODE_API.md "UI")
  const UIX = { widgets: new Map(), nodeWidgets: new Map(), styles: [], gen: 0 };
  function customWidget(n, inp) {
    return UIX.nodeWidgets.get(n.type + '\u0000' + inp.name) || (inp.opts.widget ? UIX.widgets.get(inp.opts.widget) : null) || null;
  }
  function uiApi(p) {
    const checkImpl = (impl, what) => { if (!impl || typeof impl.create !== 'function') throw new Error(`${what}: impl.create(ctx) must be a function`); };
    return Object.freeze({
      version: 1,
      pack: p.name,
      registerWidget(name, impl) {
        if (typeof name !== 'string' || !name) throw new Error('registerWidget(name, impl): name must be a string');
        checkImpl(impl, 'registerWidget');
        const old = UIX.widgets.get(name);
        if (old && old.pack !== p.name) console.warn(`[kiln] widget "${name}" from pack ${p.name} replaces the one from ${old.pack}`);
        UIX.widgets.set(name, { pack: p.name, impl });
      },
      registerNodeWidget(nodeType, inputName, impl) {
        if (typeof nodeType !== 'string' || typeof inputName !== 'string') throw new Error('registerNodeWidget(nodeType, inputName, impl)');
        checkImpl(impl, 'registerNodeWidget');
        UIX.nodeWidgets.set(nodeType + '\u0000' + inputName, { pack: p.name, impl });
      },
      css(text) {
        const tag = document.createElement('style');
        tag.dataset.pack = p.dir;
        tag.textContent = String(text);
        document.head.appendChild(tag);
        UIX.styles.push(tag);
        return tag;
      },
      url: (rel) => `/extensions/${encodeURIComponent(p.dir)}/${String(rel).replace(/^\/+/, '')}`,
      toast: (msg, kind) => K.toast(String(msg), kind),
      api: (path, opts) => K.api(path, opts),
    });
  }
  // (re)load every enabled pack's ui.js; registrations from an earlier load are dropped first
  async function loadPackUIs() {
    let d;
    try { d = await K.api('/api/extensions'); } catch (_) { return; }
    const gen = ++UIX.gen;
    UIX.widgets.clear();
    UIX.nodeWidgets.clear();
    for (const t of UIX.styles.splice(0)) t.remove();
    for (const p of d.packs) {
      if (!p.loaded || !p.enabled || !p.ui) continue;
      try {
        const url = `/extensions/${encodeURIComponent(p.dir)}/${p.ui.split('/').map(encodeURIComponent).join('/')}?v=${d.summary.gen}`;
        const mod = await import(url);
        if (gen !== UIX.gen) return;
        const setup = typeof mod.default === 'function' ? mod.default : typeof mod.setup === 'function' ? mod.setup : null;
        if (!setup) throw new Error('ui.js must export a default function (or "setup")');
        await setup(uiApi(p));
      } catch (e) {
        console.warn(`[kiln] pack ${p.name}: ui.js failed:`, e);
        K.toast(`Pack ${p.name}: ui.js failed: ${e.message}`, 'err', 6000);
      }
    }
    if (gen === UIX.gen && st.inited) for (const n of st.nodes.values()) if (n.el && CAT && CAT[n.type] && catInputs(n.type).some(i => i.widget && customWidget(n, i))) renderNode(n);
  }

  function specKind(spec) {
    if (Array.isArray(spec[0])) return 'COMBO';
    return spec[0];
  }
  const isWidgetSpec = (spec) => Array.isArray(spec[0]) || WIDGET_T.has(spec[0]);
  function catInputs(type) {
    const e = CAT && CAT[type];
    if (!e) return [];
    const res = [];
    for (const sect of ['required', 'optional']) {
      const inp = e.input && e.input[sect];
      if (!inp) continue;
      for (const [name, spec] of Object.entries(inp)) res.push({ name, spec, optional: sect === 'optional', kind: specKind(spec), widget: isWidgetSpec(spec), opts: spec[1] || {} });
    }
    return res;
  }
  const catOutputs = (type) => {
    const e = CAT && CAT[type];
    return e ? e.output.map((t, i) => ({ type: t, name: (e.output_name && e.output_name[i]) || t })) : [];
  };
  const displayName = (type) => (CAT && CAT[type] && CAT[type].display_name) || type;

  // ---------------------------------------------------------------- state
  const st = {
    nodes: new Map(), links: new Map(), nextNode: 1, nextLink: 1,
    view: { x: 40, y: 40, z: 0.85 },
    sel: new Set(), selLink: null,
    docName: null,
    myJobs: new Set(), runningJob: null,
    clipboard: null, mouse: [100, 100],
    undo: [], redo: [], restoring: false,
    inited: false, active: false, banner: {},
    queueing: 0, early: [],   // SSE events of jobs whose POST /api/graph hasn't answered yet
  };
  let V, W, NODES, LINKG, TEMP; // DOM roots

  // ---------------------------------------------------------------- node model helpers
  function makeNode(type, pos, extra = {}) {
    const id = extra.id != null ? String(extra.id) : String(st.nextNode++);
    if (Number(id) >= st.nextNode) st.nextNode = Number(id) + 1;
    const n = {
      id, type, title: extra.title || null, pos: [pos[0], pos[1]], w: extra.w || defaultWidth(type), mode: extra.mode || 0,
      widgets: {}, control: {}, converted: new Set(extra.converted || []), unsupported: extra.unsupported || null, text: extra.text || '',
      images: [], status: '', error: '', progress: null, el: null, sp: { in: {}, out: [] },
    };
    if (CAT && CAT[type]) {
      for (const inp of catInputs(type)) {
        if (!inp.widget) continue;
        n.widgets[inp.name] = defaultValue(inp);
        if (inp.opts.control_after_generate) n.control[inp.name] = 'randomize';
      }
    }
    if (extra.widgets) Object.assign(n.widgets, extra.widgets);
    if (extra.control) Object.assign(n.control, extra.control);
    st.nodes.set(id, n);
    return n;
  }
  function defaultWidth(type) {
    if (type === '__note') return 260;
    if (/KSampler|FaceDetailer/.test(type)) return 300;
    if (type === 'CLIPTextEncode') return 320;
    return 270;
  }
  function defaultValue(inp) {
    const o = inp.opts;
    if (inp.kind === 'COMBO') return o.default !== undefined ? o.default : (inp.spec[0][0] !== undefined ? inp.spec[0][0] : '');
    if (o.default !== undefined) return o.default;
    if (inp.kind === 'INT' || inp.kind === 'FLOAT') return o.min !== undefined ? Math.max(0, o.min) : 0;
    if (inp.kind === 'BOOLEAN') return false;
    return '';
  }
  // sockets shown on a node: catalog sockets + converted widgets (or the unsupported node's own list)
  function socketInputs(n) {
    if (n.type === '__note') return [];
    if (n.unsupported || !CAT || !CAT[n.type]) return (n.unsupported && n.unsupported.inputs) || [];
    return catInputs(n.type).filter(i => !i.widget || n.converted.has(i.name)).map(i => ({ name: i.name, type: i.widget ? i.kind : i.kind, optional: i.optional, converted: i.widget }));
  }
  function socketOutputs(n) {
    if (n.type === '__note') return [];
    if (n.unsupported || !CAT || !CAT[n.type]) return (n.unsupported && n.unsupported.outputs) || [];
    return catOutputs(n.type);
  }
  const inputType = (n, name) => { const s = socketInputs(n).find(i => i.name === name); return s ? s.type : null; };
  const linkInto = (nodeId, name) => { for (const l of st.links.values()) if (l.to === nodeId && l.toInput === name) return l; return null; };
  const linksOf = (nodeId) => [...st.links.values()].filter(l => l.from === nodeId || l.to === nodeId);
  const compatible = (outT, inT) => outT === inT || outT === '*' || inT === '*' || (inT === 'COMBO' && outT === 'STRING');

  function connect(fromId, fromSlot, toId, toInput, opts = {}) {
    const a = st.nodes.get(fromId), b = st.nodes.get(toId);
    if (!a || !b || fromId === toId) return null;
    const outT = (socketOutputs(a)[fromSlot] || {}).type;
    const inT = inputType(b, toInput);
    if (!outT || !inT || (!opts.force && !compatible(outT, inT))) return null;
    if (wouldCycle(fromId, toId)) { K.toast('That link would create a loop', 'err'); return null; }
    const old = linkInto(toId, toInput);
    if (old) removeLink(old.id, true);
    const l = { id: opts.id != null ? opts.id : st.nextLink++, from: fromId, fromSlot, to: toId, toInput, type: outT };
    if (l.id >= st.nextLink) st.nextLink = l.id + 1;
    st.links.set(l.id, l);
    drawLink(l);
    return l;
  }
  function wouldCycle(fromId, toId) {
    // adding from->to creates a cycle if `to` reaches `from` downstream
    const seen = new Set(), stack = [toId];
    while (stack.length) {
      const x = stack.pop();
      if (x === fromId) return true;
      if (seen.has(x)) continue;
      seen.add(x);
      for (const l of st.links.values()) if (l.from === x) stack.push(l.to);
    }
    return false;
  }
  function removeLink(id, silent) {
    const l = st.links.get(id);
    if (!l) return;
    st.links.delete(id);
    if (l.path) { l.path.remove(); l.hit.remove(); }
    if (st.selLink === id) st.selLink = null;
    if (!silent) commit();
  }
  function removeNode(id) {
    const n = st.nodes.get(id);
    if (!n) return;
    for (const l of linksOf(id)) removeLink(l.id, true);
    if (n.el) n.el.remove();
    st.nodes.delete(id);
    st.sel.delete(id);
  }

  // ---------------------------------------------------------------- view transform
  function applyView() {
    const { x, y, z } = st.view;
    W.style.transform = `translate(${x}px, ${y}px) scale(${z})`;
    const g = 24 * z;
    V.style.backgroundSize = `${g}px ${g}px, ${g * 5}px ${g * 5}px`;
    V.style.backgroundPosition = `${x}px ${y}px, ${x}px ${y}px`;
  }
  function toWorld(cx, cy) {
    const r = V.getBoundingClientRect();
    return [(cx - r.left - st.view.x) / st.view.z, (cy - r.top - st.view.y) / st.view.z];
  }
  function zoomAt(cx, cy, factor) {
    const r = V.getBoundingClientRect();
    const z = clamp(st.view.z * factor, 0.15, 2.5);
    const k = z / st.view.z;
    st.view.x = cx - r.left - (cx - r.left - st.view.x) * k;
    st.view.y = cy - r.top - (cy - r.top - st.view.y) * k;
    st.view.z = z;
    applyView();
    saveSoon();
  }
  function fitView() {
    if (!st.nodes.size) { st.view = { x: 40, y: 40, z: 0.85 }; applyView(); return; }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of st.nodes.values()) {
      const h = n.el ? n.el.offsetHeight : 120;
      x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1]);
      x1 = Math.max(x1, n.pos[0] + n.w); y1 = Math.max(y1, n.pos[1] + h);
    }
    const r = V.getBoundingClientRect();
    const pad = 40;
    const z = clamp(Math.min((r.width - pad * 2) / (x1 - x0), (r.height - pad * 2) / (y1 - y0)), 0.2, 1.2);
    st.view = { z, x: (r.width - (x1 - x0) * z) / 2 - x0 * z, y: (r.height - (y1 - y0) * z) / 2 - y0 * z };
    applyView();
    saveSoon();
  }

  // ---------------------------------------------------------------- rendering: nodes
  function renderNode(n) {
    const old = n.el;
    const e = el('div', 'ng-node');
    e.dataset.id = n.id;
    n.el = e;
    e.style.left = n.pos[0] + 'px';
    e.style.top = n.pos[1] + 'px';
    e.style.width = n.w + 'px';
    const supported = n.type === '__note' || (CAT && CAT[n.type] && !n.unsupported);
    e.classList.toggle('unsupported', !supported);
    e.classList.toggle('note', n.type === '__note');
    e.classList.toggle('selected', st.sel.has(n.id));
    e.classList.toggle('bypass', n.mode === 4);
    e.classList.toggle('muted', n.mode === 2);
    if (n.status) e.classList.add(n.status);
    if (n.error) e.classList.add('error');
    const cat = CAT && CAT[n.type] ? CAT[n.type].category || '' : '';
    e.dataset.cat = cat.split('/')[0];

    // title
    const head = el('div', 'ng-head');
    const title = el('span', 'ng-title', n.type === '__note' ? (n.title || 'Note') : (n.title || displayName(n.type)));
    const pack = supported ? packOf(n.type) : '';
    title.title = n.type === '__note' ? '' : `${n.type}  #${n.id}${cat ? '  ·  ' + cat : ''}${pack ? '  ·  pack ' + pack : ''}`;
    head.appendChild(title);
    if (pack) { const pb = el('span', 'ng-pack', pack); pb.title = `from the "${pack}" pack`; head.appendChild(pb); }
    const badge = el('span', 'ng-badge', n.mode === 4 ? 'bypass' : n.mode === 2 ? 'muted' : (!supported ? 'unsupported' : '#' + n.id));
    head.appendChild(badge);
    e.appendChild(head);
    const prog = el('div', 'ng-prog');
    prog.appendChild(el('div'));
    e.appendChild(prog);

    const body = el('div', 'ng-body');
    if (n.type === '__note') {
      const ta = el('textarea', 'ng-note-text');
      ta.value = n.text || '';
      ta.rows = 5;
      ta.addEventListener('input', () => { n.text = ta.value; commitSoon(); });
      body.appendChild(ta);
    } else {
      // sockets
      const ins = socketInputs(n), outs = socketOutputs(n);
      const rows = Math.max(ins.length, outs.length);
      if (rows) {
        const io = el('div', 'ng-io');
        for (let i = 0; i < rows; i++) {
          const row = el('div', 'ng-row');
          const L = el('div', 'ng-in');
          const R = el('div', 'ng-out');
          if (ins[i]) {
            const s = sock(n, 'in', ins[i].name, ins[i].type);
            const lab = el('span', 'ng-lab', ins[i].name);
            if (ins[i].optional) lab.classList.add('opt');
            if (ins[i].converted) lab.classList.add('conv');
            L.append(s, lab);
          }
          if (outs[i]) {
            const lab = el('span', 'ng-lab', outs[i].name);
            R.append(lab, sock(n, 'out', i, outs[i].type));
          }
          row.append(L, R);
          io.appendChild(row);
        }
        body.appendChild(io);
      }
      if (supported) {
        const ws = el('div', 'ng-widgets');
        for (const inp of catInputs(n.type)) {
          if (!inp.widget || n.converted.has(inp.name)) continue;
          ws.appendChild(buildWidget(n, inp));
        }
        if (ws.childNodes.length) body.appendChild(ws);
      } else {
        const msg = el('div', 'ng-unsup', CAT ? `"${n.type}" is not available in Kiln. Replace or delete this node to run the graph.` : 'Loading node catalog…');
        body.appendChild(msg);
        if (n.unsupported && Array.isArray(n.unsupported.widgets_values) && n.unsupported.widgets_values.length) {
          const pre = el('div', 'ng-unsup-vals', n.unsupported.widgets_values.map(v => typeof v === 'object' ? JSON.stringify(v) : String(v)).join(' · ').slice(0, 300));
          body.appendChild(pre);
        }
      }
      if (n.images && n.images.length) body.appendChild(buildImages(n));
      if (n.uiText) body.appendChild(el('pre', 'ng-uitext', n.uiText));
    }
    if (n.error) body.appendChild(el('div', 'ng-err', n.error));
    e.appendChild(body);
    const rz = el('div', 'ng-resize');
    rz.title = 'Drag to resize';
    e.appendChild(rz);

    if (old && old.parentNode) old.replaceWith(e); else NODES.appendChild(e);
    resizeObs.observe(e);
    requestAnimationFrame(() => { measure(n); updateLinksFor(n.id); });
    return e;
  }
  function sock(n, dir, key, type) {
    const s = el('span', 'ng-sock');
    s.dataset.node = n.id;
    s.dataset.dir = dir;
    s.dataset.key = String(key);
    s.dataset.type = type;
    s.style.setProperty('--c', typeColor(type));
    s.title = `${type}`;
    return s;
  }
  // socket centres relative to the node's top-left, in world units
  function measure(n) {
    if (!n.el || !n.el.isConnected) return;
    const nr = n.el.getBoundingClientRect();
    if (!nr.width) return;
    const z = st.view.z;
    n.sp = { in: {}, out: [] };
    for (const s of n.el.querySelectorAll('.ng-sock')) {
      const r = s.getBoundingClientRect();
      const p = [(r.left + r.width / 2 - nr.left) / z, (r.top + r.height / 2 - nr.top) / z];
      if (s.dataset.dir === 'in') n.sp.in[s.dataset.key] = p; else n.sp.out[Number(s.dataset.key)] = p;
    }
    n.h = nr.height / z;
  }
  const resizeObs = new ResizeObserver((ents) => {
    for (const en of ents) {
      const n = st.nodes.get(en.target.dataset.id);
      if (n && n.el === en.target) { measure(n); updateLinksFor(n.id); }
    }
  });
  function sockPos(n, dir, key) {
    const off = dir === 'in' ? n.sp.in[key] : n.sp.out[key];
    if (off) return [n.pos[0] + off[0], n.pos[1] + off[1]];
    return dir === 'in' ? [n.pos[0], n.pos[1] + 14] : [n.pos[0] + n.w, n.pos[1] + 14];
  }

  // ---------------------------------------------------------------- widgets
  function buildWidget(n, inp) {
    const row = el('div', 'ng-w');
    row.dataset.name = inp.name;
    const lab = el('span', 'ng-wl', inp.name);
    lab.title = inp.opts.tooltip || inp.name;
    const val = n.widgets[inp.name];
    const set = (v) => { n.widgets[inp.name] = v; commitSoon(); };
    const cw = customWidget(n, inp);
    if (cw) {
      try {
        const w = cw.impl.create({
          value: val, set,
          input: { name: inp.name, type: inp.kind, options: Object.assign({}, inp.opts), values: inp.kind === 'COMBO' ? inp.spec[0].slice() : undefined },
          node: { id: n.id, type: n.type, title: n.title || displayName(n.type) },
        });
        if (!(w instanceof Node)) throw new Error('create() must return a DOM element');
        const box = el('div', 'ng-cw');
        box.appendChild(w);
        row.classList.add('custom');
        if (cw.impl.fullWidth) row.classList.add('multi'); else row.appendChild(lab);
        row.appendChild(box);
        return row;
      } catch (e) {
        console.warn(`[kiln] pack ${cw.pack}: widget for ${n.type}.${inp.name} failed, using the default:`, e);
      }
    }
    if (inp.kind === 'INT' || inp.kind === 'FLOAT') {
      const o = inp.opts;
      const input = el('input', 'ng-num');
      input.type = 'number';
      input.step = o.step != null ? o.step : (inp.kind === 'INT' ? 1 : 0.01);
      if (o.min != null) input.min = o.min;
      if (o.max != null && o.max < 1e15) input.max = o.max;
      input.value = val;
      const fix = (v) => {
        v = Number(v);
        if (!Number.isFinite(v)) v = defaultValue(inp);
        if (o.min != null) v = Math.max(o.min, v);
        if (o.max != null) v = Math.min(o.max, v);
        if (inp.kind === 'INT') v = Math.round(v);
        else v = Math.round(v * 1e6) / 1e6;
        return v;
      };
      input.addEventListener('change', () => { const v = fix(input.value); input.value = v; set(v); });
      // drag the label horizontally to scrub the value
      lab.classList.add('scrub');
      lab.addEventListener('pointerdown', (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        lab.setPointerCapture(ev.pointerId);
        const x0 = ev.clientX, v0 = Number(n.widgets[inp.name]) || 0;
        const step = Number(input.step) || 1;
        const move = (m) => { const v = fix(v0 + Math.round((m.clientX - x0) / 6) * step); input.value = v; n.widgets[inp.name] = v; };
        const up = () => { lab.removeEventListener('pointermove', move); lab.removeEventListener('pointerup', up); lab.removeEventListener('pointercancel', up); commitSoon(); };
        lab.addEventListener('pointermove', move);
        lab.addEventListener('pointerup', up);
        lab.addEventListener('pointercancel', up);
      });
      row.append(lab, input);
      if (o.control_after_generate) {
        const sel = el('select', 'ng-ctl');
        for (const c of CONTROL) { const op = el('option', '', c); op.value = c; sel.appendChild(op); }
        sel.value = n.control[inp.name] || 'randomize';
        sel.title = 'control after generate';
        sel.addEventListener('change', () => { n.control[inp.name] = sel.value; commitSoon(); });
        row.classList.add('has-ctl');
        row.appendChild(sel);
      }
    } else if (inp.kind === 'STRING') {
      if (inp.opts.multiline) {
        row.classList.add('multi');
        const ta = el('textarea', 'ng-text');
        ta.rows = n.type === 'CLIPTextEncode' ? 4 : 2;
        ta.value = val == null ? '' : String(val);
        ta.placeholder = inp.name;
        ta.addEventListener('input', () => set(ta.value));
        row.append(ta);
      } else {
        const input = el('input', 'ng-str');
        input.type = 'text';
        input.value = val == null ? '' : String(val);
        input.addEventListener('input', () => set(input.value));
        row.append(lab, input);
      }
    } else if (inp.kind === 'BOOLEAN') {
      const b = el('button', 'ng-bool');
      b.type = 'button';
      const render = () => { const on = !!n.widgets[inp.name]; b.classList.toggle('on', on); b.textContent = on ? (inp.opts.label_on || 'true') : (inp.opts.label_off || 'false'); };
      render();
      b.addEventListener('click', () => { set(!n.widgets[inp.name]); render(); });
      row.append(lab, b);
    } else if (inp.kind === 'COMBO') {
      const list = inp.spec[0] || [];
      const sel = el('select', 'ng-combo');
      for (const v of list) { const op = el('option', '', String(v)); op.value = String(v); sel.appendChild(op); }
      const missing = val != null && val !== '' && !list.map(String).includes(String(val));
      if (missing || (val === '' && !list.length)) {
        const op = el('option', '', `${val || '(none)'}  (missing)`);
        op.value = String(val);
        sel.prepend(op);
        row.classList.add('missing');
      }
      sel.value = String(val);
      sel.addEventListener('change', () => { set(sel.value); if (row.classList.contains('missing')) renderNode(n); refreshBanner(); });
      row.append(lab, sel);
      if (missing) {
        const sug = suggest(String(val), list, inp.opts.kiln_list);
        const hint = el('div', 'ng-miss');
        hint.append(el('span', '', inp.opts.kiln_list ? 'not found locally' : 'not supported'));
        if (sug) {
          const btn = el('button', 'ng-use', `use ${sug}`);
          btn.type = 'button';
          btn.addEventListener('click', () => { set(sug); renderNode(n); refreshBanner(); });
          hint.appendChild(btn);
        }
        row.appendChild(hint);
      }
      if (inp.opts.image_upload) {
        const up = el('button', 'ng-upload', 'Upload…');
        up.type = 'button';
        up.addEventListener('click', () => pickUpload(n, inp.name));
        row.appendChild(up);
        if (val && !missing) {
          const img = el('img', 'ng-thumb');
          img.src = '/api/input/thumb/' + encodeURIComponent(val) + '?t=' + (n.thumbT || 0);
          img.alt = val;
          img.addEventListener('click', () => openViewer({ url: '/api/input/' + encodeURIComponent(val), w: 0, h: 0, title: val }));
          row.appendChild(img);
        }
      }
    } else {
      row.append(lab, el('span', 'muted', String(val)));
    }
    return row;
  }

  function suggest(value, list, isFiles) {
    if (!list.length) return null;
    const norm = (s) => String(s).toLowerCase().split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx)$/, '').replace(/[^a-z0-9]+/g, ' ').trim();
    const nv = norm(value);
    let best = null, bestScore = -1;
    for (const c of list) {
      const nc = norm(c);
      let score;
      if (nc === nv) score = 10;
      else {
        const a = new Set(nv.split(' ')), b = new Set(nc.split(' '));
        const inter = [...a].filter(x => b.has(x)).length;
        const jac = inter / (new Set([...a, ...b]).size || 1);
        score = jac + (nc.startsWith(nv.split(' ')[0] || '~') ? 0.2 : 0) + lcs(nv, nc) / Math.max(nv.length, nc.length, 1) * 0.5;
      }
      if (score > bestScore) { bestScore = score; best = c; }
    }
    if (bestScore >= 0.3) return best;
    return isFiles && list.length === 1 ? list[0] : (isFiles ? null : list[0]);
  }
  function lcs(a, b) {
    if (a.length > 80 || b.length > 80) return 0;
    const dp = new Array(b.length + 1).fill(0);
    for (let i = 1; i <= a.length; i++) {
      let prev = 0;
      for (let j = 1; j <= b.length; j++) {
        const tmp = dp[j];
        dp[j] = a[i - 1] === b[j - 1] ? prev + 1 : Math.max(dp[j], dp[j - 1]);
        prev = tmp;
      }
    }
    return dp[b.length];
  }

  function buildImages(n) {
    const box = el('div', 'ng-images');
    for (const im of n.images.slice(0, 8)) {
      const img = el('img');
      img.src = im.thumb || im.url;
      img.alt = '';
      img.title = `${im.w}×${im.h}${im.kind === 'preview' ? ' (preview)' : ''}`;
      img.addEventListener('click', () => openViewer(im));
      box.appendChild(img);
    }
    if (n.images.length > 1) box.classList.add('multi');
    return box;
  }

  // ---------------------------------------------------------------- rendering: links
  function linkPathD(a, b) {
    const dx = Math.max(40, Math.abs(b[0] - a[0]) * 0.5);
    return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
  }
  function drawLink(l) {
    if (!l.path) {
      l.hit = document.createElementNS(SVGNS, 'path');
      l.hit.setAttribute('class', 'ng-link-hit');
      l.hit.dataset.link = l.id;
      l.path = document.createElementNS(SVGNS, 'path');
      l.path.setAttribute('class', 'ng-link');
      l.path.style.setProperty('--c', typeColor(l.type));
      LINKG.append(l.hit, l.path);
    }
    const a = st.nodes.get(l.from), b = st.nodes.get(l.to);
    if (!a || !b) return;
    const d = linkPathD(sockPos(a, 'out', l.fromSlot), sockPos(b, 'in', l.toInput));
    l.path.setAttribute('d', d);
    l.hit.setAttribute('d', d);
    l.path.classList.toggle('sel', st.selLink === l.id);
    l.path.classList.toggle('dim', a.mode === 2 || b.mode === 2);
  }
  function updateLinksFor(nodeId) { for (const l of st.links.values()) if (l.from === nodeId || l.to === nodeId) drawLink(l); }
  function redrawAllLinks() { for (const l of st.links.values()) drawLink(l); }

  // ---------------------------------------------------------------- selection
  function select(ids, add) {
    if (!add) st.sel.clear();
    for (const id of ids) st.sel.add(id);
    st.selLink = null;
    for (const n of st.nodes.values()) if (n.el) n.el.classList.toggle('selected', st.sel.has(n.id));
    for (const l of st.links.values()) if (l.path) l.path.classList.remove('sel');
  }
  function selectLink(id) {
    select([], false);
    st.selLink = id;
    const l = st.links.get(id);
    if (l && l.path) l.path.classList.add('sel');
  }

  // ---------------------------------------------------------------- pointer interaction
  const ptrs = new Map();
  let drag = null;       // current gesture
  let longPress = null;

  function onPointerDown(e) {
    if (e.button === 2) return; // context menu handles
    closeMenus();
    const t = e.target;
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    st.mouse = toWorld(e.clientX, e.clientY);
    if (ptrs.size >= 2 && e.pointerType === 'touch') {
      // second finger: pinch-zoom takes over whatever the first finger started
      if (drag && drag.kind === 'link') { TEMP.style.display = 'none'; }
      try { V.setPointerCapture(e.pointerId); } catch (_) { }
      startPinch();
      return;
    }
    if (t.closest('input, select, textarea, button, .ng-thumb, .ng-images img, .ng-note-text, .ng-cw')) return; // widgets work normally
    const sockEl = t.closest('.ng-sock');
    const nodeEl = t.closest('.ng-node');
    const hit = t.closest('.ng-link-hit');
    V.setPointerCapture(e.pointerId);
    if (spaceDown || e.button === 1) { drag = { kind: 'pan', x: e.clientX, y: e.clientY, vx: st.view.x, vy: st.view.y, moved: false }; return; }
    if (sockEl) { startLinkDrag(sockEl, e); return; }
    if (t.classList.contains('ng-resize') && nodeEl) {
      const n = st.nodes.get(nodeEl.dataset.id);
      drag = { kind: 'resize', n, x: e.clientX, w0: n.w };
      return;
    }
    if (nodeEl) {
      const n = st.nodes.get(nodeEl.dataset.id);
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        if (st.sel.has(n.id)) { st.sel.delete(n.id); nodeEl.classList.remove('selected'); } else select([n.id], true);
      } else if (!st.sel.has(n.id)) select([n.id], false);
      NODES.appendChild(nodeEl); // bring to front
      drag = { kind: 'node', x: e.clientX, y: e.clientY, start: [...st.sel].map(id => [id, st.nodes.get(id).pos.slice()]), moved: false };
      return;
    }
    if (hit) { selectLink(Number(hit.dataset.link)); drag = null; return; }
    // background
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      const r = V.getBoundingClientRect();
      drag = { kind: 'box', x0: e.clientX - r.left, y0: e.clientY - r.top, add: true };
      return;
    }
    drag = { kind: 'pan', x: e.clientX, y: e.clientY, vx: st.view.x, vy: st.view.y, moved: false };
    if (e.pointerType === 'touch') {
      longPress = setTimeout(() => {
        if (drag && drag.kind === 'pan' && !drag.moved) { drag = null; openSearch(e.clientX, e.clientY); }
      }, 550);
    }
  }
  function startPinch() {
    clearTimeout(longPress);
    const [a, b] = [...ptrs.values()];
    drag = { kind: 'pinch', d0: Math.hypot(a.x - b.x, a.y - b.y), z0: st.view.z, vx: st.view.x, vy: st.view.y, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  }
  function onPointerMove(e) {
    if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    st.mouse = toWorld(e.clientX, e.clientY);
    if (!drag) return;
    const z = st.view.z;
    switch (drag.kind) {
      case 'pan': {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; clearTimeout(longPress); }
        st.view.x = drag.vx + dx; st.view.y = drag.vy + dy;
        applyView();
        break;
      }
      case 'pinch': {
        if (ptrs.size < 2) break;
        const [a, b] = [...ptrs.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        const r = V.getBoundingClientRect();
        const nz = clamp(drag.z0 * d / (drag.d0 || 1), 0.15, 2.5);
        const mx = (a.x + b.x) / 2 - r.left, my = (a.y + b.y) / 2 - r.top;
        const wx = (drag.mx - r.left - drag.vx) / drag.z0, wy = (drag.my - r.top - drag.vy) / drag.z0;
        st.view = { z: nz, x: mx - wx * nz, y: my - wy * nz };
        applyView();
        break;
      }
      case 'node': {
        const dx = (e.clientX - drag.x) / z, dy = (e.clientY - drag.y) / z;
        if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
        for (const [id, p0] of drag.start) {
          const n = st.nodes.get(id);
          n.pos = [Math.round(p0[0] + dx), Math.round(p0[1] + dy)];
          n.el.style.left = n.pos[0] + 'px';
          n.el.style.top = n.pos[1] + 'px';
          updateLinksFor(id);
        }
        break;
      }
      case 'resize': {
        drag.n.w = Math.max(160, Math.round(drag.w0 + (e.clientX - drag.x) / z));
        drag.n.el.style.width = drag.n.w + 'px';
        break;
      }
      case 'box': {
        const r = V.getBoundingClientRect();
        const x1 = e.clientX - r.left, y1 = e.clientY - r.top;
        const bx = $('ngBox');
        bx.hidden = false;
        Object.assign(bx.style, { left: Math.min(drag.x0, x1) + 'px', top: Math.min(drag.y0, y1) + 'px', width: Math.abs(x1 - drag.x0) + 'px', height: Math.abs(y1 - drag.y0) + 'px' });
        drag.x1 = x1; drag.y1 = y1;
        break;
      }
      case 'link': {
        const p = toWorld(e.clientX, e.clientY);
        TEMP.setAttribute('d', drag.dir === 'out' ? linkPathD(drag.anchor, p) : linkPathD(p, drag.anchor));
        // highlight a compatible socket under the pointer
        const over = document.elementFromPoint(e.clientX, e.clientY);
        const s = over && over.closest('.ng-sock');
        if (drag.hover && drag.hover !== s) drag.hover.classList.remove('hot');
        drag.hover = null;
        if (s && s.dataset.dir !== drag.dir && s.dataset.node !== drag.node) {
          const ok = drag.dir === 'out' ? compatible(drag.type, s.dataset.type) : compatible(s.dataset.type, drag.type);
          if (ok) { s.classList.add('hot'); drag.hover = s; }
        }
        break;
      }
    }
  }
  function onPointerUp(e) {
    ptrs.delete(e.pointerId);
    clearTimeout(longPress);
    if (!drag) return;
    const d = drag;
    if (d.kind === 'pinch') { drag = ptrs.size === 1 ? null : drag; saveSoon(); return; }
    drag = null;
    switch (d.kind) {
      case 'pan':
        if (!d.moved) select([], false);
        saveSoon();
        break;
      case 'node':
        if (d.moved) commit();
        break;
      case 'resize':
        commit();
        break;
      case 'box': {
        $('ngBox').hidden = true;
        if (d.x1 == null) break;
        const a = toWorld(Math.min(d.x0, d.x1) + V.getBoundingClientRect().left, Math.min(d.y0, d.y1) + V.getBoundingClientRect().top);
        const b = toWorld(Math.max(d.x0, d.x1) + V.getBoundingClientRect().left, Math.max(d.y0, d.y1) + V.getBoundingClientRect().top);
        const ids = [...st.nodes.values()].filter(n => n.pos[0] < b[0] && n.pos[0] + n.w > a[0] && n.pos[1] < b[1] && n.pos[1] + (n.h || 100) > a[1]).map(n => n.id);
        select(ids, true);
        break;
      }
      case 'link': endLinkDrag(e, d); break;
    }
  }

  // ---- link dragging
  function startLinkDrag(s, e) {
    const n = st.nodes.get(s.dataset.node);
    if (s.dataset.dir === 'in') {
      const existing = linkInto(n.id, s.dataset.key);
      if (existing) {
        // pick the link up from its source (ComfyUI behaviour)
        const src = st.nodes.get(existing.from);
        removeLink(existing.id, true);
        drag = { kind: 'link', dir: 'out', node: src.id, key: existing.fromSlot, type: existing.type, anchor: sockPos(src, 'out', existing.fromSlot), detached: true };
      } else {
        drag = { kind: 'link', dir: 'in', node: n.id, key: s.dataset.key, type: s.dataset.type, anchor: sockPos(n, 'in', s.dataset.key) };
      }
    } else {
      drag = { kind: 'link', dir: 'out', node: n.id, key: Number(s.dataset.key), type: s.dataset.type, anchor: sockPos(n, 'out', Number(s.dataset.key)) };
    }
    TEMP.style.setProperty('--c', typeColor(drag.type));
    TEMP.setAttribute('d', '');
    TEMP.style.display = '';
  }
  function endLinkDrag(e, d) {
    TEMP.style.display = 'none';
    if (d.hover) d.hover.classList.remove('hot');
    const over = document.elementFromPoint(e.clientX, e.clientY);
    const s = over && over.closest('.ng-sock');
    const nodeEl = over && over.closest('.ng-node');
    if (s && s.dataset.dir !== d.dir && s.dataset.node !== d.node) {
      if (d.dir === 'out') connect(d.node, d.key, s.dataset.node, s.dataset.key);
      else connect(s.dataset.node, Number(s.dataset.key), d.node, d.key);
      commit();
      return;
    }
    if (nodeEl && nodeEl.dataset.id !== d.node) {
      // dropped on a node body: first compatible free (or any compatible) socket
      const n = st.nodes.get(nodeEl.dataset.id);
      if (d.dir === 'out') {
        const ins = socketInputs(n).filter(i => compatible(d.type, i.type));
        const tgt = ins.find(i => !linkInto(n.id, i.name)) || ins[0];
        if (tgt) { connect(d.node, d.key, n.id, tgt.name); commit(); return; }
      } else {
        const oi = socketOutputs(n).findIndex(o => compatible(o.type, d.type));
        if (oi >= 0) { connect(n.id, oi, d.node, d.key); commit(); return; }
      }
      if (d.detached) commit();
      return;
    }
    if (!nodeEl) {
      // dropped into empty space: search nodes compatible with this type, then auto-connect
      openSearch(e.clientX, e.clientY, { dir: d.dir, node: d.node, key: d.key, type: d.type });
      if (d.detached) commit();
      return;
    }
    if (d.detached) commit();
  }

  function onWheel(e) {
    const ta = e.target.closest && e.target.closest('textarea, select');
    if (ta && ta.tagName === 'TEXTAREA' && ta.scrollHeight > ta.clientHeight && !e.ctrlKey) return;
    e.preventDefault();
    const factor = Math.exp(-(e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY) * (e.ctrlKey ? 0.01 : 0.0015));
    zoomAt(e.clientX, e.clientY, factor);
  }

  // ---------------------------------------------------------------- menus + search
  let menuStack = [];
  function closeMenus() {
    for (const m of menuStack) m.remove();
    menuStack = [];
    const s = $('ngSearch');
    if (s && !s.hidden && !s.contains(document.activeElement)) { /* keep search if focused */ }
  }
  function openMenu(items, cx, cy, level = 0) {
    while (menuStack.length > level) menuStack.pop().remove();
    const m = el('div', 'ng-menu');
    for (const it of items) {
      if (it.sep) { m.appendChild(el('div', 'ng-menu-sep')); continue; }
      if (it.header) { m.appendChild(el('div', 'ng-menu-head', it.header)); continue; }
      const row = el('div', 'ng-menu-item' + (it.disabled ? ' disabled' : '') + (it.sub ? ' has-sub' : ''));
      row.appendChild(el('span', '', it.label));
      if (it.pack) row.appendChild(el('span', 'ng-pack', it.pack));
      if (it.hint) row.appendChild(el('span', 'ng-menu-hint', it.hint));
      const openSub = () => {
        const r = row.getBoundingClientRect();
        openMenu(typeof it.sub === 'function' ? it.sub() : it.sub, r.right - 2, r.top, level + 1);
      };
      if (it.sub) {
        row.addEventListener('mouseenter', () => { if (!matchMedia('(hover: none)').matches) openSub(); });
        row.addEventListener('click', (ev) => { ev.stopPropagation(); openSub(); });
      } else if (!it.disabled) {
        row.addEventListener('mouseenter', () => { while (menuStack.length > level + 1) menuStack.pop().remove(); });
        row.addEventListener('click', (ev) => { ev.stopPropagation(); closeMenus(); it.action && it.action(); });
      }
      m.appendChild(row);
    }
    document.body.appendChild(m);
    const r = m.getBoundingClientRect();
    let x = cx, y = cy;
    if (x + r.width > innerWidth - 4) x = level ? Math.max(4, cx - r.width - (menuStack[level - 1] ? menuStack[level - 1].getBoundingClientRect().width - 4 : 0)) : Math.max(4, innerWidth - r.width - 4);
    if (y + r.height > innerHeight - 4) y = Math.max(4, innerHeight - r.height - 4);
    m.style.left = x + 'px';
    m.style.top = y + 'px';
    m.addEventListener('pointerdown', (ev) => ev.stopPropagation());
    menuStack.push(m);
    return m;
  }
  function categoryTree(filterFn) {
    const root = {};
    for (const [type, e] of Object.entries(CAT || {})) {
      if (filterFn && !filterFn(type)) continue;
      const parts = (e.category || 'other').split('/');
      let node = root;
      for (const p of parts) node = (node[p] = node[p] || {});
      (node.__items = node.__items || []).push(type);
    }
    return root;
  }
  function treeItems(tree, at) {
    const items = [];
    for (const k of Object.keys(tree).filter(k => k !== '__items').sort()) items.push({ label: k, sub: () => treeItems(tree[k], at) });
    for (const t of (tree.__items || []).sort((a, b) => displayName(a).localeCompare(displayName(b)))) items.push({ label: displayName(t), pack: packOf(t), hint: t !== displayName(t) && !packOf(t) ? t : '', action: () => addNodeAt(t, at) });
    return items;
  }
  function canvasMenu(cx, cy) {
    const at = toWorld(cx, cy);
    openMenu([
      { label: 'Add Node', sub: () => treeItems(categoryTree(), at) },
      { label: 'Search nodes…', hint: 'dbl-click', action: () => openSearch(cx, cy) },
      { label: 'Add Note', action: () => { const n = makeNode('__note', at); renderNode(n); select([n.id]); commit(); } },
      { label: 'Templates', sub: () => EXAMPLES.map(x => ({ label: x.title, hint: x.description, action: () => loadExample(x) })) },
      { sep: true },
      { label: 'Paste', hint: 'Ctrl+V', disabled: !st.clipboard, action: () => paste(at) },
      { label: 'Select all', hint: 'Ctrl+A', action: () => select([...st.nodes.keys()]) },
      { label: 'Fit view', hint: 'F', action: fitView },
      { label: 'Arrange (auto layout)', action: () => { autoLayout(); redrawAll(); commit(); fitView(); } },
    ], cx, cy);
  }
  function nodeMenu(n, cx, cy) {
    if (!st.sel.has(n.id)) select([n.id]);
    const widgets = CAT && CAT[n.type] ? catInputs(n.type).filter(i => i.widget) : [];
    const items = [
      { header: n.type === '__note' ? 'Note' : `${displayName(n.type)}  #${n.id}` },
      { label: 'Title…', action: () => { const t = prompt('Node title', n.title || displayName(n.type)); if (t != null) { n.title = t.trim() || null; renderNode(n); commit(); } } },
    ];
    if (n.type !== '__note') {
      items.push(
        { label: n.mode === 4 ? 'Un-bypass' : 'Bypass', hint: 'Ctrl+B', action: () => setMode(4) },
        { label: n.mode === 2 ? 'Unmute' : 'Mute', hint: 'Ctrl+M', action: () => setMode(2) },
      );
      const conv = widgets.filter(i => !n.converted.has(i.name));
      const back = widgets.filter(i => n.converted.has(i.name));
      if (conv.length) items.push({ label: 'Convert widget to input', sub: () => conv.map(i => ({ label: i.name, action: () => { n.converted.add(i.name); renderNode(n); commit(); } })) });
      if (back.length) items.push({ label: 'Convert input to widget', sub: () => back.map(i => ({ label: i.name, action: () => { const l = linkInto(n.id, i.name); if (l) removeLink(l.id, true); n.converted.delete(i.name); renderNode(n); commit(); } })) });
      if (n.type === 'LoadImage') items.push({ label: 'Upload image…', action: () => pickUpload(n, 'image') });
    }
    items.push(
      { sep: true },
      { label: 'Copy', hint: 'Ctrl+C', action: copySel },
      { label: 'Duplicate', hint: 'Ctrl+D', action: duplicateSel },
      { label: 'Delete', hint: 'Del', action: deleteSel },
    );
    openMenu(items, cx, cy);
  }
  function onContextMenu(e) {
    e.preventDefault();
    const t = e.target;
    if (t.closest('input, textarea, select')) return;
    const nodeEl = t.closest('.ng-node');
    const hit = t.closest('.ng-link-hit');
    if (nodeEl) return nodeMenu(st.nodes.get(nodeEl.dataset.id), e.clientX, e.clientY);
    if (hit) {
      const id = Number(hit.dataset.link);
      selectLink(id);
      return openMenu([{ label: 'Delete link', action: () => removeLink(id) }], e.clientX, e.clientY);
    }
    canvasMenu(e.clientX, e.clientY);
  }

  // fuzzy node search
  function fuzzyScore(q, s) {
    // spaces / _ / - are ignored so "empty sd3" matches "EmptySD3LatentImage"
    s = s.toLowerCase().replace(/[\s_\-()]+/g, '');
    q = q.replace(/[\s_\-()]+/g, '');
    if (!q) return 1;
    const idx = s.indexOf(q);
    if (idx >= 0) return 100 - idx - s.length * 0.01 + (idx === 0 ? 20 : 0);
    let si = 0, score = 0, streak = 0;
    for (const ch of q) {
      const f = s.indexOf(ch, si);
      if (f < 0) return -1;
      streak = f === si ? streak + 1 : 0;
      score += 1 + streak;
      si = f + 1;
    }
    return score - s.length * 0.01;
  }
  let searchCtx = null;
  function openSearch(cx, cy, link) {
    closeMenus();
    const box = $('ngSearch');
    searchCtx = { at: toWorld(cx, cy), link: link || null, sel: 0, items: [] };
    box.hidden = false;
    const vr = V.getBoundingClientRect();
    const w = Math.min(340, vr.width - 16);
    box.style.width = w + 'px';
    box.style.left = clamp(cx - vr.left, 8, vr.width - w - 8) + 'px';
    box.style.top = clamp(cy - vr.top, 8, Math.max(8, vr.height - 360)) + 'px';
    const input = $('ngSearchInput');
    input.value = '';
    $('ngSearchHint').textContent = link ? `${link.dir === 'out' ? 'inputs accepting' : 'outputs of type'} ${link.type}` : 'type to search · Enter to add';
    renderSearch();
    setTimeout(() => input.focus(), 0);
  }
  function closeSearch() { $('ngSearch').hidden = true; searchCtx = null; }
  function searchCandidates() {
    const L = searchCtx.link;
    const all = Object.keys(CAT || {});
    const out = [];
    for (const t of all) {
      if (L) {
        if (L.dir === 'out' && !catInputs(t).some(i => !i.widget && compatible(L.type, i.kind))) continue;
        if (L.dir === 'in' && !catOutputs(t).some(o => compatible(o.type, L.type))) continue;
      }
      out.push(t);
    }
    return out;
  }
  function renderSearch() {
    const q = $('ngSearchInput').value.trim().toLowerCase();
    const list = $('ngSearchList');
    list.textContent = '';
    let items = searchCandidates().map(t => ({ t, s: Math.max(fuzzyScore(q, displayName(t)), fuzzyScore(q, t) - 5, fuzzyScore(q, (CAT[t].category || '') + ' ' + displayName(t)) - 30, packOf(t) ? fuzzyScore(q, packOf(t) + ' ' + displayName(t)) - 30 : -1) }))
      .filter(x => x.s >= 0);
    if (!searchCtx.link && (!q || fuzzyScore(q, 'note comment') >= 0)) items.push({ t: '__note', s: q ? fuzzyScore(q, 'note comment') - 10 : -1 });
    if (q) items.sort((a, b) => b.s - a.s);
    else items.sort((a, b) => ((CAT[a.t] || {}).category || 'zz').localeCompare((CAT[b.t] || {}).category || 'zz') || displayName(a.t).localeCompare(displayName(b.t)));
    items = items.slice(0, 80);
    searchCtx.items = items.map(x => x.t);
    searchCtx.sel = clamp(searchCtx.sel, 0, Math.max(0, items.length - 1));
    let lastCat = null;
    items.forEach((x, i) => {
      const cat = x.t === '__note' ? 'utils' : (CAT[x.t].category || 'other');
      if (!q && cat !== lastCat) { list.appendChild(el('div', 'ng-search-cat', cat)); lastCat = cat; }
      const row = el('div', 'ng-search-item' + (i === searchCtx.sel ? ' sel' : ''));
      const label = el('span', 'ng-search-name', x.t === '__note' ? 'Note' : displayName(x.t));
      if (x.t !== '__note' && packOf(x.t)) label.appendChild(el('span', 'ng-pack', packOf(x.t)));
      row.append(label, el('span', 'ng-search-type', x.t === '__note' ? 'comment' : (q ? cat : x.t)));
      row.addEventListener('pointerdown', (ev) => { ev.preventDefault(); pickSearch(i); });
      list.appendChild(row);
    });
    if (!items.length) list.appendChild(el('div', 'ng-search-empty', 'No matching nodes'));
    const selEl = list.querySelector('.sel');
    if (selEl) selEl.scrollIntoView({ block: 'nearest' });
  }
  function pickSearch(i) {
    const t = searchCtx.items[i];
    if (!t) return;
    const ctx = searchCtx;
    closeSearch();
    if (t === '__note') { const n = makeNode('__note', ctx.at); renderNode(n); select([n.id]); commit(); return; }
    const n = addNodeAt(t, ctx.at, true);
    const L = ctx.link;
    if (L) {
      requestAnimationFrame(() => {
        if (L.dir === 'out') {
          const inp = socketInputs(n).find(s => compatible(L.type, s.type));
          if (inp) connect(L.node, L.key, n.id, inp.name);
        } else {
          const oi = socketOutputs(n).findIndex(o => compatible(o.type, L.type));
          if (oi >= 0) {
            // place the new node to the left of the input it feeds
            n.pos[0] -= n.w + 20;
            n.el.style.left = n.pos[0] + 'px';
            connect(n.id, oi, L.node, L.key);
          }
        }
        commit();
      });
    } else commit();
  }
  function addNodeAt(type, at, noCommit) {
    const n = makeNode(type, [Math.round(at[0]), Math.round(at[1])]);
    renderNode(n);
    select([n.id]);
    if (!noCommit) commit();
    hideEmpty();
    return n;
  }

  // ---------------------------------------------------------------- keyboard
  let spaceDown = false;
  function onKey(e) {
    if (!st.active) return;
    const typing = e.target.closest && e.target.closest('input, textarea, select, [contenteditable]');
    if (e.key === 'Escape') { closeMenus(); closeSearch(); closeViewer(); return; }
    if (typing) return;
    const ctrl = e.ctrlKey || e.metaKey;
    if (e.code === 'Space' && !spaceDown) { spaceDown = true; V.classList.add('grab'); e.preventDefault(); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (st.selLink != null) { removeLink(st.selLink); e.preventDefault(); return; }
      if (st.sel.size) { deleteSel(); e.preventDefault(); }
      return;
    }
    if (!ctrl && (e.key === 'f' || e.key === '.')) { fitView(); return; }
    if (!ctrl) return;
    const k = e.key.toLowerCase();
    if (k === 'a') { e.preventDefault(); select([...st.nodes.keys()]); }
    else if (k === 'c') { copySel(); }
    else if (k === 'v') { e.preventDefault(); paste(st.mouse); }
    else if (k === 'd') { e.preventDefault(); duplicateSel(); }
    else if (k === 'b') { e.preventDefault(); setMode(4); }
    else if (k === 'm') { e.preventDefault(); setMode(2); }
    else if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
    else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); redo(); }
    else if (k === 's') { e.preventDefault(); saveWorkflow(false); }
  }
  function onKeyUp(e) { if (e.code === 'Space') { spaceDown = false; V.classList.remove('grab'); } }

  function setMode(mode) {
    const nodes = [...st.sel].map(id => st.nodes.get(id)).filter(n => n && n.type !== '__note');
    if (!nodes.length) return;
    const allOn = nodes.every(n => n.mode === mode);
    for (const n of nodes) { n.mode = allOn ? 0 : mode; renderNode(n); updateLinksFor(n.id); }
    commit();
    refreshBanner();
  }
  function deleteSel() {
    if (!st.sel.size) return;
    for (const id of [...st.sel]) removeNode(id);
    commit();
    refreshBanner();
  }
  function copySel() {
    if (!st.sel.size) return;
    const ids = new Set(st.sel);
    st.clipboard = {
      nodes: [...ids].map(id => serializeNode(st.nodes.get(id))),
      links: [...st.links.values()].filter(l => ids.has(l.from) && ids.has(l.to)).map(l => [l.from, l.fromSlot, l.to, l.toInput]),
    };
    K.lsSet('kiln.graph.clip', st.clipboard);
  }
  function paste(at) {
    const clip = st.clipboard || K.lsGet('kiln.graph.clip');
    if (!clip || !clip.nodes || !clip.nodes.length) return;
    const minX = Math.min(...clip.nodes.map(n => n.pos[0])), minY = Math.min(...clip.nodes.map(n => n.pos[1]));
    const map = {};
    for (const sn of clip.nodes) {
      const n = deserializeNode(Object.assign({}, sn, { id: undefined, pos: [Math.round(at[0] + sn.pos[0] - minX), Math.round(at[1] + sn.pos[1] - minY)] }));
      map[sn.id] = n.id;
      renderNode(n);
    }
    requestAnimationFrame(() => {
      for (const [f, s, t, i] of clip.links) if (map[f] && map[t]) connect(map[f], s, map[t], i);
      commit();
    });
    select(Object.values(map));
    hideEmpty();
  }
  function duplicateSel() {
    if (!st.sel.size) return;
    copySel();
    const ns = [...st.sel].map(id => st.nodes.get(id));
    paste([Math.min(...ns.map(n => n.pos[0])) + 30, Math.min(...ns.map(n => n.pos[1])) + 30]);
  }

  // ---------------------------------------------------------------- serialization / undo / autosave
  function serializeNode(n) {
    const o = { id: n.id, type: n.type, pos: n.pos.slice(), w: n.w, mode: n.mode || 0 };
    if (n.title) o.title = n.title;
    if (n.type === '__note') { o.text = n.text; return o; }
    o.widgets = Object.assign({}, n.widgets);
    if (Object.keys(n.control).length) o.control = Object.assign({}, n.control);
    if (n.converted.size) o.converted = [...n.converted];
    if (n.unsupported) o.unsupported = n.unsupported;
    return o;
  }
  function deserializeNode(o) {
    const n = makeNode(o.type, o.pos || [0, 0], { id: o.id, title: o.title, w: o.w, mode: o.mode, converted: o.converted, unsupported: (!CAT || !CAT[o.type]) && o.type !== '__note' ? (o.unsupported || { inputs: [], outputs: [] }) : null, text: o.text, control: o.control });
    if (o.widgets) Object.assign(n.widgets, o.widgets);
    return n;
  }
  function serialize() {
    return {
      kiln_graph: 1,
      name: st.docName || undefined,
      nodes: [...st.nodes.values()].map(serializeNode),
      links: [...st.links.values()].map(l => [l.id, l.from, l.fromSlot, l.to, l.toInput, l.type]),
      view: Object.assign({}, st.view),
      next: { node: st.nextNode, link: st.nextLink },
    };
  }
  function clearGraph() {
    for (const n of st.nodes.values()) if (n.el) n.el.remove();
    for (const l of st.links.values()) if (l.path) { l.path.remove(); l.hit.remove(); }
    st.nodes.clear(); st.links.clear(); st.sel.clear(); st.selLink = null;
    st.nextNode = 1; st.nextLink = 1;
  }
  function loadDoc(doc, opts = {}) {
    st.restoring = true;
    clearGraph();
    for (const o of doc.nodes || []) deserializeNode(o);
    if (doc.next) { st.nextNode = Math.max(st.nextNode, doc.next.node || 1); st.nextLink = Math.max(st.nextLink, doc.next.link || 1); }
    if (doc.view && !opts.fit) st.view = Object.assign({ x: 40, y: 40, z: 0.85 }, doc.view);
    if (opts.name !== undefined) st.docName = opts.name; else if (doc.name) st.docName = doc.name;
    for (const n of st.nodes.values()) renderNode(n);
    applyView();
    requestAnimationFrame(() => {
      for (const n of st.nodes.values()) measure(n);
      for (const [id, from, slot, to, input] of doc.links || []) connect(String(from), slot, String(to), input, { id, force: true });
      st.restoring = false;
      if (opts.fit) fitView();
      if (!opts.noHistory) commit(true);
      refreshBanner();
      toggleEmpty();
    });
  }
  let commitT = 0, saveT = 0;
  function commit(fresh) {
    if (st.restoring) return;
    const snap = JSON.stringify(serialize());
    if (fresh) { st.undo = [snap]; st.redo = []; }
    else if (st.undo[st.undo.length - 1] !== snap) { st.undo.push(snap); if (st.undo.length > 60) st.undo.shift(); st.redo = []; }
    saveSoon();
    refreshBanner();
    toggleEmpty();
  }
  function commitSoon() { clearTimeout(commitT); commitT = setTimeout(() => commit(), 400); }
  function saveSoon() { clearTimeout(saveT); saveT = setTimeout(() => K.lsSet(LS_DOC, serialize()), 300); }
  function undo() {
    if (st.undo.length < 2) return;
    st.redo.push(st.undo.pop());
    loadDoc(JSON.parse(st.undo[st.undo.length - 1]), { noHistory: true });
    saveSoon();
  }
  function redo() {
    const s = st.redo.pop();
    if (!s) return;
    st.undo.push(s);
    loadDoc(JSON.parse(s), { noHistory: true });
    saveSoon();
  }
  function redrawAll() {
    for (const n of st.nodes.values()) { n.el.style.left = n.pos[0] + 'px'; n.el.style.top = n.pos[1] + 'px'; }
    redrawAllLinks();
  }

  // ---------------------------------------------------------------- graph -> ComfyUI API format
  function toApi() {
    const api = {};
    const resolve = (fromId, slot, type, depth = 0) => {
      const n = st.nodes.get(fromId);
      if (!n || depth > 50) return null;
      if (n.mode === 2) return null;                 // muted: output is absent
      if (n.mode === 4) {                            // bypass: pass through the first input of the same type
        const outT = (socketOutputs(n)[slot] || {}).type || type;
        for (const s of socketInputs(n)) {
          if (s.type !== outT) continue;
          const l = linkInto(n.id, s.name);
          if (l) return resolve(l.from, l.fromSlot, outT, depth + 1);
        }
        return null;
      }
      return [n.id, slot];
    };
    for (const n of st.nodes.values()) {
      if (n.type === '__note' || n.mode === 2 || n.mode === 4) continue;
      const inputs = {};
      if (CAT && CAT[n.type] && !n.unsupported) {
        for (const inp of catInputs(n.type)) {
          if (inp.widget && (!n.converted.has(inp.name) || !linkInto(n.id, inp.name))) {
            let v = n.widgets[inp.name];
            if (inp.kind === 'INT' || inp.kind === 'FLOAT') v = Number(v);
            inputs[inp.name] = v;
          }
        }
      }
      for (const l of st.links.values()) {
        if (l.to !== n.id) continue;
        const src = resolve(l.from, l.fromSlot, l.type);
        if (src) inputs[l.toInput] = src;
        else if (n.converted.has(l.toInput)) inputs[l.toInput] = n.widgets[l.toInput]; // fall back to the widget value
      }
      api[n.id] = { class_type: n.type, inputs };
      if (n.title) api[n.id]._meta = { title: n.title };
    }
    return api;
  }
  // nodes the output nodes need (for the "unsupported blocks queue" rule)
  function reachable() {
    // output nodes: catalog output_node, plus unsupported "sink" nodes (nothing consumes them — e.g. WAS "Image Save")
    const consumed = new Set([...st.links.values()].map(l => l.from));
    const outs = [...st.nodes.values()].filter(n => n.mode === 0 && n.type !== '__note' && ((CAT && CAT[n.type] && !n.unsupported && CAT[n.type].output_node) || ((n.unsupported || !CAT || !CAT[n.type]) && !consumed.has(n.id))));
    const keep = new Set(), stack = outs.map(n => n.id);
    while (stack.length) {
      const id = stack.pop();
      if (keep.has(id)) continue;
      keep.add(id);
      for (const l of st.links.values()) if (l.to === id) stack.push(l.from);
    }
    return keep;
  }

  // ---------------------------------------------------------------- banner (unsupported / missing models / errors)
  function refreshBanner() {
    const b = $('ngBanner');
    if (!b || !CAT) return;
    const need = reachable();
    const unsup = [...st.nodes.values()].filter(n => n.type !== '__note' && (!CAT[n.type] || n.unsupported));
    const blocking = unsup.filter(n => need.has(n.id) && n.mode === 0);
    const missing = [];
    for (const n of st.nodes.values()) {
      if (!CAT[n.type] || n.unsupported || n.mode === 2) continue;
      for (const inp of catInputs(n.type)) {
        if (inp.kind !== 'COMBO' || n.converted.has(inp.name)) continue;
        const v = n.widgets[inp.name];
        if (!inp.spec[0].map(String).includes(String(v))) missing.push({ n, inp, v });
      }
    }
    const errs = [...st.nodes.values()].filter(n => n.error);
    b.textContent = '';
    const parts = [];
    if (unsup.length) {
      const types = [...new Set(unsup.map(n => n.type))];
      const p = el('div', 'ng-bn-row bad');
      p.append(el('b', '', `Unsupported nodes (${unsup.length}): `), el('span', '', types.join(', ')));
      p.append(el('span', 'ng-bn-hint', blocking.length ? ' — Kiln can’t run these. Delete or replace them (right-click the canvas → Add Node), or install a pack that provides them (Extensions), to queue.' : ' — not needed by any output, so the graph can still run.'));
      parts.push(p);
    }
    if (missing.length) {
      const p = el('div', 'ng-bn-row warn');
      p.append(el('b', '', `Missing values (${missing.length}): `), el('span', '', [...new Set(missing.map(m => `${m.v}`))].slice(0, 6).join(', ')));
      const fixable = missing.map(m => ({ m, s: suggest(String(m.v), m.inp.spec[0], m.inp.opts.kiln_list) })).filter(x => x.s);
      if (fixable.length) {
        const btn = el('button', 'ng-bn-btn', `Use suggestions (${fixable.length})`);
        btn.addEventListener('click', () => {
          for (const { m, s } of fixable) m.n.widgets[m.inp.name] = s;
          for (const n of new Set(fixable.map(x => x.m.n))) renderNode(n);
          commit();
        });
        p.appendChild(btn);
      }
      parts.push(p);
    }
    if (errs.length) {
      const p = el('div', 'ng-bn-row bad');
      p.append(el('b', '', `Errors (${errs.length}): `), el('span', '', errs.map(n => `#${n.id} ${displayName(n.type)}: ${n.error}`).join(' · ').slice(0, 400)));
      const btn = el('button', 'ng-bn-btn', 'Clear');
      btn.addEventListener('click', () => { for (const n of errs) { n.error = ''; renderNode(n); } refreshBanner(); });
      p.appendChild(btn);
      parts.push(p);
    }
    b.append(...parts);
    b.hidden = !parts.length;
    const q = $('ngQueue');
    q.disabled = blocking.length > 0 || !st.nodes.size;
    q.title = blocking.length ? `Can't queue: unsupported node(s) ${[...new Set(blocking.map(n => n.type))].join(', ')}` : 'Queue (Ctrl+Enter)';
  }

  // ---------------------------------------------------------------- run
  async function queue() {
    if (!st.nodes.size) return;
    if ($('ngQueue').disabled) { K.toast($('ngQueue').title, 'err', 4000); return; }
    const count = clamp(Number($('ngCount').value) || 1, 1, 64);
    for (let i = 0; i < count; i++) {
      const graph = toApi();
      if (!Object.keys(graph).length) { K.toast('Nothing to run (all nodes muted/bypassed?)', 'err'); return; }
      for (const n of st.nodes.values()) if (n.error) { n.error = ''; renderNode(n); }
      st.queueing++;
      try {
        const r = await K.api('/api/graph', { method: 'POST', body: { graph, ui: serialize() } });
        st.myJobs.add(r.id);
        setStatus(`Queued ${r.id}${r.pruned ? ` · ${r.pruned} unused node(s) skipped` : ''}`);
        // events of this job that arrived before the response (e.g. pack nodes folded as it started)
        const early = st.early.filter(m => sseJob(m) === r.id);
        st.early = st.early.filter(m => sseJob(m) !== r.id);
        for (const m of early) onSSE(m);
      } catch (e) {
        const d = e.data || {};
        if (d.node_errors) for (const [id, msg] of Object.entries(d.node_errors)) { const n = st.nodes.get(id); if (n) { n.error = msg; renderNode(n); } }
        if (d.unknown) for (const u of d.unknown) { const n = st.nodes.get(u.id); if (n) { n.error = 'unsupported node type'; renderNode(n); } }
        refreshBanner();
        K.toast('Queue failed: ' + e.message, 'err', 6000);
        return;
      } finally {
        if (!--st.queueing) st.early = [];
      }
      applyControlAfterGenerate();
    }
  }
  function applyControlAfterGenerate() {
    let changed = false;
    for (const n of st.nodes.values()) {
      if (!CAT || !CAT[n.type] || n.mode === 2) continue;
      for (const inp of catInputs(n.type)) {
        if (!inp.opts.control_after_generate || n.converted.has(inp.name)) continue;
        const mode = n.control[inp.name] || 'randomize';
        const v = Number(n.widgets[inp.name]) || 0;
        let nv = v;
        if (mode === 'randomize') nv = Math.floor(Math.random() * 1125899906842624);
        else if (mode === 'increment') nv = v + 1;
        else if (mode === 'decrement') nv = Math.max(0, v - 1);
        if (nv !== v) {
          n.widgets[inp.name] = nv;
          const input = n.el && n.el.querySelector(`.ng-w[data-name="${inp.name}"] .ng-num`);
          if (input) input.value = nv;
          changed = true;
        }
      }
    }
    if (changed) commit();
  }
  function setStatus(t) { const s = $('ngStatus'); if (s) s.textContent = t || ''; }

  const sseJob = (m) => m.id || (m.job && m.job.id) || null;
  function onSSE(m) {
    const jid = sseJob(m);
    if (jid && st.queueing && !st.myJobs.has(jid)) { st.early.push(m); if (st.early.length > 1000) st.early.shift(); return; }
    const mine = m.id && st.myJobs.has(m.id);
    switch (m.type) {
      case 'job': {
        const j = m.job;
        if (!st.myJobs.has(j.id)) return;
        if (j.status === 'running') {
          st.runningJob = j.id;
          for (const n of st.nodes.values()) {
            if (n.status || n.error || n.progress) { n.status = ''; n.error = ''; n.progress = null; if (n.el) { n.el.classList.remove('running', 'done', 'error'); n.el.querySelector('.ng-prog > div').style.width = '0'; } }
            n.jobImagesFrom = null;
            if (n.uiText) { n.uiText = ''; if (n.el) renderNode(n); }
          }
          setStatus('Running…');
          refreshBanner();
        } else if (['done', 'error', 'cancelled'].includes(j.status)) {
          if (st.runningJob === j.id) st.runningJob = null;
          for (const n of st.nodes.values()) if (n.el) { n.el.classList.remove('running'); n.el.querySelector('.ng-prog').classList.remove('on'); }
          if (j.status === 'done') setStatus(`Done in ${K.fmtSec(j.total_ms)} s`);
          else if (j.status === 'cancelled') setStatus('Cancelled');
          else {
            setStatus('Error');
            const n = j.error_node && st.nodes.get(j.error_node);
            if (n) { n.error = j.error; renderNode(n); }
            K.toast('Graph failed' + (n ? ` at #${n.id} ${displayName(n.type)}` : '') + ': ' + j.error, 'err', 7000);
            refreshBanner();
          }
        } else if (j.status === 'queued') setStatus(`Queued (#${j.position || 1})`);
        break;
      }
      case 'gnode': {
        if (!mine) return;
        const n = st.nodes.get(m.node);
        if (!n || !n.el) return;
        if (m.status === 'start') {
          n.status = 'running';
          n.el.classList.add('running');
          n.el.classList.remove('done');
          setStatus(`Running ${displayName(n.type)} #${n.id}`);
        } else {
          n.status = 'done';
          n.el.classList.remove('running');
          n.el.classList.add('done');
          n.el.querySelector('.ng-prog').classList.remove('on');
          const b = n.el.querySelector('.ng-badge');
          if (b && m.ms != null) b.textContent = `${(m.ms / 1000).toFixed(m.ms < 10000 ? 2 : 1)} s`;
        }
        break;
      }
      case 'step': {
        if (!mine || m.node == null) return;
        const n = st.nodes.get(m.node);
        if (!n || !n.el) return;
        const p = n.el.querySelector('.ng-prog');
        p.classList.add('on');
        p.firstChild.style.width = `${(m.step / m.of) * 100}%`;
        setStatus(`${displayName(n.type)} #${n.id} · step ${m.step}/${m.of}${m.ms != null ? ' · ' + K.fmtMs(m.ms) : ''}`);
        break;
      }
      case 'gimage': {
        if (!mine) return;
        const n = st.nodes.get(m.node);
        if (!n) return;
        if (n.jobImagesFrom !== m.id) { n.images = []; n.jobImagesFrom = m.id; }
        n.images.push(m.image);
        renderNode(n);
        break;
      }
      case 'gext': {
        if (!mine) return;
        const n = st.nodes.get(m.node);
        if (!n || !m.ui) return;
        n.uiText = m.ui.text || '';
        renderNode(n);
        break;
      }
      case 'extensions':
        if (st.inited) loadCatalog(true);
        break;
      case 'inputs': {
        if (!CAT || !CAT.LoadImage) return;
        CAT.LoadImage.input.required.image[0] = m.images;
        for (const n of st.nodes.values()) if (n.type === 'LoadImage') renderNode(n);
        refreshBanner();
        break;
      }
      case 'hello':
        if (st.inited) loadCatalog(true);
        break;
    }
  }

  // ---------------------------------------------------------------- uploads
  function pickUpload(n, name) {
    const f = el('input');
    f.type = 'file';
    f.accept = 'image/*';
    f.addEventListener('change', async () => { if (f.files[0]) await uploadImage(f.files[0], n, name); });
    f.click();
  }
  async function toPngBlob(file) {
    if (file.type === 'image/png') return file;
    const bmp = await createImageBitmap(file);
    const c = document.createElement('canvas');
    c.width = bmp.width; c.height = bmp.height;
    c.getContext('2d').drawImage(bmp, 0, 0);
    return new Promise((res, rej) => c.toBlob(b => b ? res(b) : rej(new Error('PNG conversion failed')), 'image/png'));
  }
  async function uploadImage(file, n, name) {
    try {
      const blob = await toPngBlob(file);
      const fname = (file.name || 'image').replace(/\.[a-z0-9]+$/i, '') + '.png';
      const r = await fetch('/api/upload/image?name=' + encodeURIComponent(fname), { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: blob });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || r.statusText);
      if (CAT && CAT.LoadImage) {
        const list = CAT.LoadImage.input.required.image[0];
        if (!list.includes(d.name)) { list.push(d.name); list.sort((a, b) => a.localeCompare(b)); }
      }
      if (n) { n.widgets[name] = d.name; n.thumbT = Date.now(); renderNode(n); commit(); }
      K.toast(`Uploaded ${d.name} (${d.w}×${d.h})`, 'ok', 2000);
      return d;
    } catch (e) {
      K.toast('Upload failed: ' + e.message, 'err', 5000);
      return null;
    }
  }

  // ---------------------------------------------------------------- viewer
  function openViewer(im) {
    const v = $('ngViewer');
    $('ngViewerImg').src = im.view || im.url;
    $('ngViewerCap').textContent = `${im.title || (im.file || '').split('/').pop() || (im.kind === 'preview' ? 'preview' : '')}${im.w ? ` · ${im.w}×${im.h}` : ''}${im.kind === 'preview' ? ' · preview (not saved)' : ''}`;
    const a = $('ngViewerDl');
    a.href = im.file ? im.url + '?dl=1' : im.url;
    a.setAttribute('download', (im.file || im.url).split('/').pop());
    v.hidden = false;
  }
  function closeViewer() { const v = $('ngViewer'); if (v) v.hidden = true; }

  // ---------------------------------------------------------------- import: ComfyUI UI format, API format, PNGs
  async function readPngText(buf) {
    const dv = new DataView(buf);
    const u8 = new Uint8Array(buf);
    const sig = [137, 80, 78, 71, 13, 10, 26, 10];
    if (!sig.every((b, i) => u8[i] === b)) throw new Error('not a PNG');
    const out = {};
    const latin1 = (a) => { let s = ''; for (let i = 0; i < a.length; i += 8192) s += String.fromCharCode.apply(null, a.subarray(i, i + 8192)); return s; };
    const inflate = async (a) => new Uint8Array(await new Response(new Blob([a]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
    let p = 8;
    while (p + 8 <= u8.length) {
      const len = dv.getUint32(p);
      const type = latin1(u8.subarray(p + 4, p + 8));
      const data = u8.subarray(p + 8, p + 8 + len);
      if (type === 'tEXt') {
        const z = data.indexOf(0);
        const key = latin1(data.subarray(0, z));
        // ComfyUI writes UTF-8 in tEXt when it can; try UTF-8 first
        try { out[key] = new TextDecoder('utf-8', { fatal: true }).decode(data.subarray(z + 1)); } catch (_) { out[key] = latin1(data.subarray(z + 1)); }
      } else if (type === 'iTXt') {
        const z = data.indexOf(0);
        const key = latin1(data.subarray(0, z));
        const comp = data[z + 1];
        let q = z + 3;
        q = data.indexOf(0, q) + 1; // language
        q = data.indexOf(0, q) + 1; // translated keyword
        const body = data.subarray(q);
        out[key] = new TextDecoder().decode(comp ? await inflate(body) : body);
      } else if (type === 'zTXt') {
        const z = data.indexOf(0);
        out[latin1(data.subarray(0, z))] = latin1(await inflate(data.subarray(z + 2)));
      } else if (type === 'IDAT' || type === 'IEND') break;
      p += 12 + len;
    }
    return out;
  }
  async function importFile(file) {
    try {
      if (/\.png$/i.test(file.name) || file.type === 'image/png') {
        const t = await readPngText(await file.arrayBuffer());
        if (t.kiln) {
          const meta = JSON.parse(t.kiln);
          if (meta.kind === 'graph') return openFromMeta(meta, file.name);
          return importSimple(meta, file.name);
        }
        if (t.workflow) return importAny(JSON.parse(t.workflow), file.name);
        if (t.prompt) return importAny(JSON.parse(t.prompt), file.name);
        K.toast('No workflow found in this PNG', 'err');
        return;
      }
      const text = await file.text();
      importAny(JSON.parse(text), file.name);
    } catch (e) {
      K.toast('Import failed: ' + e.message, 'err', 6000);
    }
  }
  function importAny(obj, name) {
    if (!obj || typeof obj !== 'object') throw new Error('not a workflow');
    if (obj.kiln_graph) { loadDoc(obj, { name: null }); report(name, 'Kiln'); return; }
    if (obj.workflow && Array.isArray(obj.workflow.nodes)) obj = obj.workflow;
    if (Array.isArray(obj.nodes) && obj.links) { loadDoc(importComfyUI(obj), { fit: true, name: null }); report(name, 'ComfyUI workflow'); return; }
    if (obj.prompt && typeof obj.prompt === 'object' && !obj.class_type) obj = obj.prompt;
    const vals = Object.values(obj);
    if (vals.length && vals.every(v => v && typeof v === 'object' && typeof v.class_type === 'string')) { loadDoc(importApi(obj), { fit: true, name: null }); report(name, 'ComfyUI API'); return; }
    throw new Error('unrecognised JSON (expected a ComfyUI workflow, API prompt, or Kiln graph)');
  }
  function report(name, fmt) {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const unsup = [...st.nodes.values()].filter(n => n.unsupported).length;
      K.toast(`Imported ${name ? '"' + name + '" ' : ''}(${fmt}): ${st.nodes.size} nodes${unsup ? `, ${unsup} unsupported` : ''}`, unsup ? 'err' : 'ok', 3500);
    }));
  }

  // map ComfyUI widgets_values onto catalog widget names (required then optional, declaration order;
  // seeds are followed by their control_after_generate value; LoadImage has an extra "upload" value)
  function mapWidgetValues(type, values) {
    const widgets = {}, control = {};
    if (!values) return { widgets, control };
    const specs = catInputs(type).filter(i => i.widget);
    if (!Array.isArray(values)) {
      for (const s of specs) if (values[s.name] !== undefined) widgets[s.name] = values[s.name];
      return { widgets, control };
    }
    let i = 0;
    for (const s of specs) {
      if (i >= values.length) break;
      widgets[s.name] = values[i++];
      if (s.opts.control_after_generate && CONTROL.includes(values[i])) control[s.name] = values[i++];
      if (s.opts.image_upload && (values[i] === 'image' || values[i] === 'upload' || values[i] === null)) i++;
    }
    return { widgets, control };
  }
  function importComfyUI(wf) {
    const src = wf.nodes || [];
    const byId = new Map(src.map(n => [String(n.id), n]));
    const links = new Map();
    for (const L of wf.links || []) {
      const l = Array.isArray(L) ? { id: L[0], from: L[1], fromSlot: L[2], to: L[3], toSlot: L[4], type: L[5] } : { id: L.id, from: L.origin_id, fromSlot: L.origin_slot, to: L.target_id, toSlot: L.target_slot, type: L.type };
      links.set(String(l.id), l);
    }
    const posOf = (n) => Array.isArray(n.pos) ? [n.pos[0], n.pos[1]] : n.pos ? [n.pos[0] ?? n.pos['0'] ?? 0, n.pos[1] ?? n.pos['1'] ?? 0] : [0, 0];
    const sizeW = (n) => { const s = n.size; const w = Array.isArray(s) ? s[0] : s ? (s[0] ?? s['0']) : null; return w ? Math.max(200, Math.round(w)) : null; };
    const setters = new Map();
    for (const n of src) if (n.type === 'SetNode' && n.widgets_values) setters.set(String(n.widgets_values[0]), n);
    // follow virtual nodes back to a real source (or a literal value)
    const resolve = (fromId, slot, depth = 0) => {
      const n = byId.get(String(fromId));
      if (!n || depth > 40) return null;
      const inLink = (k) => { const inp = (n.inputs || [])[k]; return inp && inp.link != null ? links.get(String(inp.link)) : null; };
      if (n.type === 'Reroute') { const l = inLink(0); return l ? resolve(l.from, l.fromSlot, depth + 1) : null; }
      if (n.type === 'GetNode') {
        const s = n.widgets_values && setters.get(String(n.widgets_values[0]));
        if (!s) return null;
        const inp = (s.inputs || [])[0];
        const l = inp && inp.link != null ? links.get(String(inp.link)) : null;
        return l ? resolve(l.from, l.fromSlot, depth + 1) : null;
      }
      if (n.type === 'SetNode') { const l = inLink(0); return l ? resolve(l.from, l.fromSlot, depth + 1) : null; }
      if (PRIMITIVES[n.type]) {
        const wv = n.widgets_values || [];
        return { value: wv[0], control: CONTROL.includes(wv[1]) ? wv[1] : undefined };
      }
      return { id: String(n.id), slot };
    };
    const doc = { kiln_graph: 1, nodes: [], links: [], next: { node: 1, link: 1 } };
    let maxId = 0, linkId = 1;
    for (const n of src) {
      const id = String(n.id);
      maxId = Math.max(maxId, Number(n.id) || 0);
      if (VIRTUAL[n.type]) continue;
      const pos = posOf(n);
      if (NOTE_TYPES[n.type]) {
        doc.nodes.push({ id, type: '__note', pos, w: sizeW(n) || 260, title: n.title && n.title !== n.type ? n.title : (n.type === 'MarkdownNote' ? 'Markdown note' : 'Note'), text: String((n.widgets_values || [])[0] || '') });
        continue;
      }
      const o = { id, type: n.type, pos, w: sizeW(n) || defaultWidth(n.type), mode: n.mode === 2 || n.mode === 4 ? n.mode : 0 };
      if (n.title && n.title !== displayName(n.type) && n.title !== n.type) o.title = n.title;
      const known = CAT && CAT[n.type];
      if (known) {
        const m = mapWidgetValues(n.type, n.widgets_values);
        o.widgets = m.widgets;
        o.control = m.control;
      } else {
        o.unsupported = {
          inputs: (n.inputs || []).map(i => ({ name: (i.widget && i.widget.name) || i.name, type: String(i.type || '*') })),
          outputs: (n.outputs || []).map(x => ({ name: x.name || String(x.type), type: String(x.type || '*') })),
          widgets_values: n.widgets_values,
        };
      }
      o.converted = [];
      (n.inputs || []).forEach((inp, k) => {
        if (inp.link == null) return;
        const l = links.get(String(inp.link));
        if (!l) return;
        const name = (inp.widget && inp.widget.name) || inp.name;
        const s = resolve(l.from, l.fromSlot);
        if (!s) return;
        const isWidget = known && catInputs(n.type).some(i => i.widget && i.name === name);
        if (s.value !== undefined) {
          if (known) { o.widgets[name] = s.value; if (s.control) o.control[name] = s.control; }
          return;
        }
        if (isWidget) o.converted.push(name);
        doc.links.push([linkId++, s.id, s.slot, id, name, String(l.type || '*')]);
        void k;
      });
      doc.nodes.push(o);
    }
    // drop links whose source was removed (e.g. a Reroute with nothing behind it)
    const ids = new Set(doc.nodes.map(n => n.id));
    doc.links = doc.links.filter(l => ids.has(String(l[1])) && ids.has(String(l[3])));
    doc.next = { node: maxId + 1, link: linkId };
    return doc;
  }
  function importApi(api, pos) {
    const doc = { kiln_graph: 1, nodes: [], links: [], next: { node: 1, link: 1 } };
    let linkId = 1, maxId = 0;
    const entries = Object.entries(api);
    const prim = {};
    for (const [id, n] of entries) if (PRIMITIVES[n.class_type]) prim[id] = n.inputs ? n.inputs.value : undefined;
    // outputs referenced per unknown node (to give unsupported nodes some sockets)
    const refOut = {};
    for (const [, n] of entries) for (const v of Object.values(n.inputs || {})) if (Array.isArray(v) && v.length === 2) refOut[String(v[0])] = Math.max(refOut[String(v[0])] || 0, v[1] + 1);
    for (const [id, n] of entries) {
      maxId = Math.max(maxId, Number(id) || 0);
      if (PRIMITIVES[n.class_type]) continue;
      const known = CAT && CAT[n.class_type];
      const o = { id: String(id), type: n.class_type, pos: (pos && pos[id]) || null, w: defaultWidth(n.class_type), widgets: {}, converted: [] };
      if (n._meta && n._meta.title && n._meta.title !== displayName(n.class_type)) o.title = n._meta.title;
      const specs = known ? catInputs(n.class_type) : [];
      for (const [k, v] of Object.entries(n.inputs || {})) {
        if (Array.isArray(v) && v.length === 2 && typeof v[1] === 'number') {
          const srcId = String(v[0]);
          if (srcId in prim) { o.widgets[k] = prim[srcId]; continue; }
          const spec = specs.find(s => s.name === k);
          if (spec && spec.widget) o.converted.push(k);
          doc.links.push([linkId++, srcId, v[1], String(id), k, '*']);
        } else o.widgets[k] = v;
      }
      if (known) {
        // like ComfyUI: seeds default to "randomize" (applied after queueing, so the imported seed is used first)
        for (const s of specs) if (s.widget && s.opts.control_after_generate) (o.control = o.control || {})[s.name] = 'randomize';
      } else {
        o.unsupported = {
          inputs: Object.entries(n.inputs || {}).filter(([, v]) => Array.isArray(v)).map(([k]) => ({ name: k, type: '*' })),
          outputs: Array.from({ length: refOut[String(id)] || 0 }, (_, i) => ({ name: 'out ' + i, type: '*' })),
          widgets_values: Object.entries(n.inputs || {}).filter(([, v]) => !Array.isArray(v)).map(([k, v]) => `${k}=${v}`),
          api: true,
        };
      }
      doc.nodes.push(o);
    }
    // link types from the source node's outputs
    const typeOf = (id, slot) => { const n = doc.nodes.find(x => x.id === id); return n && CAT && CAT[n.type] ? CAT[n.type].output[slot] || '*' : '*'; };
    doc.links = doc.links.filter(l => doc.nodes.some(n => n.id === l[1])).map(l => (l[5] = typeOf(l[1], l[2]), l));
    if (doc.nodes.some(n => !n.pos)) layoutDoc(doc);
    doc.next = { node: maxId + 1, link: linkId };
    return doc;
  }
  // columns by longest path from sources
  function layoutDoc(doc) {
    const deps = {};
    for (const n of doc.nodes) deps[n.id] = [];
    for (const l of doc.links) if (deps[l[3]]) deps[l[3]].push(String(l[1]));
    const depth = {};
    const d = (id, guard = 0) => {
      if (depth[id] != null) return depth[id];
      if (guard > 200) return 0;
      depth[id] = 0;
      const v = deps[id] && deps[id].length ? 1 + Math.max(...deps[id].map(x => d(x, guard + 1))) : 0;
      depth[id] = v;
      return v;
    };
    const cols = {};
    for (const n of doc.nodes) (cols[d(n.id)] = cols[d(n.id)] || []).push(n);
    for (const [c, list] of Object.entries(cols)) {
      let y = 40;
      for (const n of list) {
        n.pos = [40 + Number(c) * 330, y];
        const widgets = CAT && CAT[n.type] ? catInputs(n.type).filter(i => i.widget).length : 3;
        y += 70 + widgets * 30 + (n.type === 'CLIPTextEncode' ? 60 : 0);
      }
    }
  }
  function autoLayout() {
    const doc = serialize();
    layoutDoc(doc);
    for (const o of doc.nodes) { const n = st.nodes.get(o.id); if (n) n.pos = o.pos; }
  }
  function openFromMeta(meta, name) {
    if (meta.ui && meta.ui.kiln_graph) { loadDoc(meta.ui, { fit: true, name: null }); report(name, 'Kiln image'); return; }
    if (meta.graph) { loadDoc(importApi(meta.graph), { fit: true, name: null }); report(name, 'Kiln image'); return; }
    K.toast('No graph stored in this image', 'err');
  }
  // a Simple-mode image -> the equivalent node graph
  function importSimple(p, name) {
    if (!p || p.prompt == null) { K.toast('No Kiln settings in this image', 'err'); return; }
    const L = (k) => (CAT && CAT[k.split('.')[0]] ? CAT[k.split('.')[0]].input.required[k.split('.')[1]][0] : []);
    const g = {
      1: { class_type: 'UNETLoader', inputs: { unet_name: L('UNETLoader.unet_name')[0] || 'anima-base-v1.0.safetensors', weight_dtype: 'default' } },
      3: { class_type: 'CLIPLoader', inputs: { clip_name: L('CLIPLoader.clip_name')[0] || 'qwen_3_06b_base.safetensors', type: 'stable_diffusion', device: 'default' } },
      4: { class_type: 'CLIPTextEncode', inputs: { text: p.prompt || '', clip: ['3', 0] } },
      5: { class_type: 'CLIPTextEncode', inputs: { text: p.negative || '', clip: ['3', 0] } },
      6: { class_type: 'EmptySD3LatentImage', inputs: { width: p.width || 512, height: p.height || 768, batch_size: 1 } },
      8: { class_type: 'VAELoader', inputs: { vae_name: L('VAELoader.vae_name')[0] || 'qwen_image_vae.safetensors' } },
    };
    let model = ['1', 0], id = 20;
    for (const l of p.loras || []) {
      g[id] = { class_type: 'LoraLoaderModelOnly', inputs: { model, lora_name: String(l.file).replace(/^loras\//, ''), strength_model: l.strength } };
      model = [String(id), 0]; id++;
    }
    if (p.shift != null && Number(p.shift) !== 3) { g[id] = { class_type: 'ModelSamplingSD3', inputs: { model, shift: Number(p.shift) } }; model = [String(id), 0]; id++; }
    const common = { model, seed: p.seed || 0, steps: p.steps || 8, cfg: p.cfg ?? 1, sampler_name: p.sampler || 'euler', scheduler: 'simple', positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0], denoise: 1 };
    if (p.nag && (p.cfg ?? 1) <= 1) g[7] = { class_type: 'KSamplerWithNAG', inputs: Object.assign(common, { nag_scale: p.nag.scale, nag_tau: p.nag.tau, nag_alpha: p.nag.alpha, nag_sigma_end: 0, nag_negative: ['5', 0] }) };
    else g[7] = { class_type: 'KSampler', inputs: common };
    g[9] = { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } };
    g[10] = { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Kiln' } };
    loadDoc(importApi(g), { fit: true, name: null });
    const extra = [p.hires && 'hires fix', p.face && 'face detail', p.upscale && 'upscale', p.step_cache && 'step cache', p.cfg_cutoff < 1 && 'CFG cutoff'].filter(Boolean);
    K.toast(`Rebuilt the Simple-mode settings as a graph${extra.length ? ` (not carried over: ${extra.join(', ')})` : ''}`, 'ok', 5000);
    void name;
  }
  function loadExample(x) {
    if (st.nodes.size && !confirm(`Replace the current graph with "${x.title}"?`)) return;
    const doc = importApi(x.graph, x.pos);
    doc.name = undefined;
    st.docName = null;
    loadDoc(doc, { fit: true, name: null });
  }

  // ---------------------------------------------------------------- server workflows + export
  async function saveWorkflow(asNew) {
    let name = st.docName;
    if (asNew || !name) {
      name = prompt('Save workflow as', name || 'my workflow');
      if (!name) return;
      name = name.trim();
    }
    try {
      await K.api('/api/workflows', { method: 'POST', body: { name, doc: serialize() } });
      st.docName = name;
      saveSoon();
      K.toast(`Saved "${name}"`, 'ok', 1800);
      setStatus(`Workflow: ${name}`);
    } catch (e) { K.toast('Save failed: ' + e.message, 'err'); }
  }
  async function workflowsMenu(anchor) {
    let list = [];
    try { list = (await K.api('/api/workflows')).workflows; } catch (_) { }
    const r = anchor.getBoundingClientRect();
    const items = [
      { label: st.docName ? `Save "${st.docName}"` : 'Save…', hint: 'Ctrl+S', action: () => saveWorkflow(false) },
      { label: 'Save as…', action: () => saveWorkflow(true) },
      { sep: true },
    ];
    if (!list.length) items.push({ label: 'No saved workflows yet', disabled: true });
    for (const w of list) {
      items.push({
        label: w.name, hint: new Date(w.mtime).toLocaleDateString(), sub: [
          { label: 'Open', action: async () => { const d = await K.api('/api/workflows/' + encodeURIComponent(w.name)); loadDoc(d.doc, { name: w.name }); setStatus(`Workflow: ${w.name}`); } },
          { label: 'Delete', action: async () => { if (confirm(`Delete workflow "${w.name}"?`)) { await K.api('/api/workflows/' + encodeURIComponent(w.name), { method: 'DELETE' }); if (st.docName === w.name) st.docName = null; K.toast('Deleted', 'ok', 1400); } } },
        ],
      });
    }
    openMenu(items, r.left, r.bottom + 4);
  }
  function download(name, obj) {
    const a = el('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' }));
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  function exportMenu(anchor) {
    const r = anchor.getBoundingClientRect();
    const base = (st.docName || 'kiln-graph').replace(/[^\w.-]+/g, '_');
    openMenu([
      { label: 'Save (Kiln .json)', action: () => download(base + '.json', serialize()) },
      { label: 'Export API json (ComfyUI)', action: () => download(base + '_api.json', toApi()) },
    ], r.left, r.bottom + 4);
  }

  // ---------------------------------------------------------------- empty hint
  function toggleEmpty() { const e = $('ngEmpty'); if (e) e.hidden = st.nodes.size > 0; }
  function hideEmpty() { const e = $('ngEmpty'); if (e) e.hidden = true; }

  // ---------------------------------------------------------------- init
  async function loadCatalog(refresh) {
    try {
      CAT = await K.api('/api/nodes');
      try { EXAMPLES = await K.api('/api/graph/examples'); } catch (_) { EXAMPLES = []; }
    } catch (e) {
      K.toast('Could not load the node catalog: ' + e.message, 'err', 6000);
      return false;
    }
    await loadPackUIs();
    if (refresh) {
      let adopted = 0;
      for (const n of st.nodes.values()) {
        if (n.unsupported && CAT[n.type]) { adoptUnsupported(n); adopted++; }
        renderNode(n);
      }
      requestAnimationFrame(() => { for (const n of st.nodes.values()) measure(n); redrawAllLinks(); });
      refreshBanner();
      if (adopted) { K.toast(`${adopted} node(s) are now provided by a pack`, 'ok'); commit(); }
    }
    return true;
  }
  // A red "unsupported" card whose class is now in the catalog (its pack was installed or enabled):
  // map the values it was imported with onto the real widgets and keep the links that still fit.
  function adoptUnsupported(n) {
    const u = n.unsupported;
    n.unsupported = null;
    if (u && Array.isArray(u.widgets_values) && !u.api) {
      const m = mapWidgetValues(n.type, u.widgets_values);
      Object.assign(n.widgets, m.widgets);
      Object.assign(n.control, m.control);
    }
    for (const inp of catInputs(n.type)) {
      if (!inp.widget) continue;
      if (n.widgets[inp.name] === undefined) n.widgets[inp.name] = defaultValue(inp);
      if (inp.opts.control_after_generate && !n.control[inp.name]) n.control[inp.name] = 'randomize';
    }
    const widgetNames = new Set(catInputs(n.type).filter(i => i.widget).map(i => i.name));
    for (const l of st.links.values()) if (l.to === n.id && widgetNames.has(l.toInput)) n.converted.add(l.toInput);
    for (const l of [...st.links.values()]) {
      if (l.to === n.id && !socketInputs(n).some(x => x.name === l.toInput)) removeLink(l.id, true);
      else if (l.from === n.id) {
        const o = socketOutputs(n)[l.fromSlot];
        if (!o) removeLink(l.id, true);
        else { l.type = o.type; if (l.path) l.path.style.setProperty('--c', typeColor(l.type)); }
      }
    }
  }
  function buildTemplatesList() {
    const box = $('ngEmptyTemplates');
    box.textContent = '';
    for (const x of EXAMPLES) {
      const b = el('button', 'ng-tpl');
      b.type = 'button';
      b.append(el('b', '', x.title), el('small', '', x.description || ''));
      b.addEventListener('click', () => loadExample(x));
      box.appendChild(b);
    }
  }
  async function init() {
    if (st.inited) return;
    st.inited = true;
    if (!st.bound) { st.bound = true; bindUI(); }
    await loadAll();
  }
  function bindUI() {
    V = $('ngView'); W = $('ngWorld'); NODES = $('ngNodes'); LINKG = $('ngLinkG'); TEMP = $('ngTemp');
    TEMP.style.display = 'none';
    V.addEventListener('pointerdown', onPointerDown);
    V.addEventListener('pointermove', onPointerMove);
    V.addEventListener('pointerup', onPointerUp);
    V.addEventListener('pointercancel', onPointerUp);
    V.addEventListener('wheel', onWheel, { passive: false });
    V.addEventListener('contextmenu', onContextMenu);
    V.addEventListener('dblclick', (e) => {
      // pointer capture retargets clicks to the view, so hit-test the real element under the pointer
      const t = document.elementFromPoint(e.clientX, e.clientY);
      if (t && V.contains(t) && !t.closest('.ng-node, input, textarea, select, button, .ng-link-hit')) openSearch(e.clientX, e.clientY);
    });
    document.addEventListener('keydown', onKey);
    document.addEventListener('keyup', onKeyUp);
    document.addEventListener('pointerdown', (e) => {
      if (!e.target.closest('.ng-menu')) closeMenus();
      if (!e.target.closest('#ngSearch') && !$('ngSearch').hidden) closeSearch();
    });
    const si = $('ngSearchInput');
    si.addEventListener('input', () => { searchCtx.sel = 0; renderSearch(); });
    si.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); searchCtx.sel++; renderSearch(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); searchCtx.sel = Math.max(0, searchCtx.sel - 1); renderSearch(); }
      else if (e.key === 'Enter') { e.preventDefault(); pickSearch(searchCtx.sel); }
      else if (e.key === 'Escape') closeSearch();
    });
    // toolbar
    $('ngQueue').addEventListener('click', queue);
    $('ngCancel').addEventListener('click', async () => { for (const id of st.myJobs) { try { await K.api('/api/cancel/' + id, { method: 'POST' }); } catch (_) { } } });
    $('ngTemplates').addEventListener('click', (e) => { const r = e.currentTarget.getBoundingClientRect(); openMenu(EXAMPLES.map(x => ({ label: x.title, hint: x.description, action: () => loadExample(x) })), r.left, r.bottom + 4); });
    $('ngWorkflows').addEventListener('click', (e) => workflowsMenu(e.currentTarget));
    $('ngExport').addEventListener('click', (e) => exportMenu(e.currentTarget));
    $('ngOpen').addEventListener('click', () => $('ngFile').click());
    $('ngFile').addEventListener('change', () => { const f = $('ngFile').files[0]; if (f) importFile(f); $('ngFile').value = ''; });
    $('ngAdd').addEventListener('click', () => { const r = V.getBoundingClientRect(); openSearch(r.left + r.width / 2 - 170, r.top + 60); });
    $('ngFab').addEventListener('click', () => { const r = V.getBoundingClientRect(); openSearch(r.left + 12, r.top + 60); });
    $('ngFit').addEventListener('click', fitView);
    $('ngUndo').addEventListener('click', undo);
    $('ngClear').addEventListener('click', () => { if (!st.nodes.size || confirm('Clear the whole graph?')) { clearGraph(); st.docName = null; commit(); toggleEmpty(); refreshBanner(); } });
    $('ngViewerClose').addEventListener('click', closeViewer);
    $('ngViewer').addEventListener('click', (e) => { if (e.target.id === 'ngViewer') closeViewer(); });
    // drag & drop import (json / png) — or an image onto a LoadImage node
    V.addEventListener('dragover', (e) => { e.preventDefault(); V.classList.add('drop'); });
    V.addEventListener('dragleave', () => V.classList.remove('drop'));
    V.addEventListener('drop', async (e) => {
      e.preventDefault();
      V.classList.remove('drop');
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (!f) return;
      const nodeEl = e.target.closest && e.target.closest('.ng-node');
      const n = nodeEl && st.nodes.get(nodeEl.dataset.id);
      if (n && n.type === 'LoadImage' && /^image\//.test(f.type)) { await uploadImage(f, n, 'image'); return; }
      importFile(f);
    });
    window.addEventListener('kiln:sse', (e) => onSSE(e.detail));
  }
  async function loadAll() {
    applyView();
    const ok = await loadCatalog(false);
    if (!ok) {
      const e = $('ngEmpty');
      e.hidden = false;
      e.querySelector('p').textContent = 'Nodes mode needs a newer Kiln server (GET /api/nodes failed). Restart the server (node server/server.js) and reload this page.';
      $('ngQueue').disabled = true;
      st.inited = false; // retry on the next activation
      return;
    }
    buildTemplatesList();
    const saved = K.lsGet(LS_DOC);
    if (saved && saved.kiln_graph && saved.nodes && saved.nodes.length) loadDoc(saved, {});
    else if (ok && EXAMPLES.length) { loadDoc(importApi(EXAMPLES[0].graph, EXAMPLES[0].pos), { fit: true, name: null }); }
    else { commit(true); toggleEmpty(); }
    if (st.docName) setStatus(`Workflow: ${st.docName}`);
  }

  window.KilnNodes = {
    async activate() {
      st.active = true;
      await init();
      requestAnimationFrame(() => { for (const n of st.nodes.values()) measure(n); redrawAllLinks(); });
    },
    deactivate() { st.active = false; closeMenus(); closeSearch(); },
    queue,
    openFromMeta: (meta, name) => openFromMeta(meta, name),
    importObject: (obj, name) => importAny(obj, name),
    importFile,
    toApi, serialize, loadDoc,
    upload: (file, nodeId, name) => uploadImage(file, st.nodes.get(String(nodeId)), name || 'image'),
    reloadPackUIs: () => loadPackUIs(),
    _state: st,
    _uix: UIX,
  };
  if (document.body.classList.contains('mode-nodes')) window.KilnNodes.activate();
})();
