'use strict';
// Kiln packs (docs/NODE_API.md): scan Kiln/extensions/<pack>/, validate kiln.json and the node
// schemas, load nodes.js (errors are reported per pack, never thrown), and install / uninstall /
// enable / disable / scaffold packs. Pack code runs inside the server process, so a pack is trusted
// code, exactly like a ComfyUI custom node.

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const TYPE_RE = /^[A-Z][A-Z0-9_]{0,63}$/;          // socket / value types: IMAGE, MY_TYPE, ...
const LIST_RE = /^\$[a-z_]{1,40}$/;                 // "$upscale_models" etc., filled by the server
const CLASS_RE = /^[A-Za-z_][\w .+\-|()]{0,99}$/;   // node class names
const INPUT_RE = /^[A-Za-z_][\w .-]{0,63}$/;
const DIR_RE = /^[A-Za-z0-9][\w.-]{0,63}$/;         // pack folder names
const NAME_RE = /^[\w][\w .-]{0,63}$/;              // kiln.json "name"
const DISABLED = '.disabled';
const MAX_ERR = 600;

const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));
const exists = (p) => { try { fs.accessSync(p); return true; } catch (_) { return false; } };
function readJSONFile(f) { return JSON.parse(fs.readFileSync(f, 'utf8').replace(/^\uFEFF/, '')); }

// a path from kiln.json, resolved inside the pack folder (null if it escapes it)
function inside(dir, rel) {
  if (typeof rel !== 'string' || !rel || path.isAbsolute(rel)) return null;
  const p = path.resolve(dir, rel);
  return p.startsWith(dir + path.sep) ? p : null;
}

// "SyntaxError: Unexpected token (nodes.js:12:5)"
function loadError(e, dir) {
  const msg = `${e && e.name ? e.name + ': ' : ''}${e && e.message ? e.message : String(e)}`;
  const stack = String((e && e.stack) || '');
  const esc = dir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const m = stack.match(new RegExp(esc + '[\\\\/]([^:\\n\\r]+?):(\\d+)(?::(\\d+))?'));
  const loc = m ? ` (${m[1].replace(/\\/g, '/')}:${m[2]}${m[3] ? ':' + m[3] : ''})` : '';
  return (msg + loc).slice(0, MAX_ERR);
}

// ---------------------------------------------------------------------------
// node schema validation -> catalog entry (the /api/nodes shape)
// ---------------------------------------------------------------------------
function checkSpec(spec) {
  if (!Array.isArray(spec) || spec.length < 1 || spec.length > 2) return 'must be [type] or [type, options]';
  const t = spec[0];
  if (Array.isArray(t)) {
    if (!t.every(v => typeof v === 'string' || (typeof v === 'number' && Number.isFinite(v)))) return 'combo values must be strings or numbers';
  } else if (typeof t !== 'string' || !(t === '*' || TYPE_RE.test(t) || LIST_RE.test(t))) {
    return `bad type ${JSON.stringify(t)} (use an UPPER_CASE type, "*", a combo list or a "$list")`;
  }
  const o = spec[1];
  if (o === undefined) return null;
  if (o === null || typeof o !== 'object' || Array.isArray(o)) return 'options must be an object';
  if (t === 'INT' || t === 'FLOAT') {
    for (const k of ['default', 'min', 'max', 'step']) if (o[k] !== undefined && (typeof o[k] !== 'number' || !Number.isFinite(o[k]))) return `options.${k} must be a number`;
  }
  if (o.widget !== undefined && (typeof o.widget !== 'string' || !o.widget)) return 'options.widget must be a widget name (see the UI section)';
  try { JSON.stringify(o); } catch (_) { return 'options must be plain JSON'; }
  return null;
}

function normalizeNode(cls, def, packName) {
  if (!CLASS_RE.test(cls)) throw new Error('bad class name (letters, digits, space . + - | ( ) _, max 100)');
  if (!def || typeof def !== 'object') throw new Error('the definition must be an object');
  if (typeof def.run !== 'function') throw new Error('run(inputs, ctx) is missing');
  const inputs = def.inputs == null ? {} : def.inputs;
  if (typeof inputs !== 'object' || Array.isArray(inputs)) throw new Error('inputs must be { required: {...}, optional: {...} }');
  for (const k of Object.keys(inputs)) if (k !== 'required' && k !== 'optional') throw new Error(`inputs.${k}: only "required" and "optional" are supported`);
  const input = {};
  const seen = new Set();
  for (const sect of ['required', 'optional']) {
    const s = inputs[sect];
    if (s == null) continue;
    if (typeof s !== 'object' || Array.isArray(s)) throw new Error(`inputs.${sect} must be an object`);
    input[sect] = {};
    for (const [name, spec] of Object.entries(s)) {
      if (!INPUT_RE.test(name)) throw new Error(`input "${name}": bad name`);
      if (seen.has(name)) throw new Error(`input "${name}" is declared twice`);
      seen.add(name);
      const err = checkSpec(spec);
      if (err) throw new Error(`input "${name}": ${err}`);
      input[sect][name] = spec.length > 1 ? [clone(spec[0]), clone(spec[1])] : [clone(spec[0])];
    }
  }
  const outputs = def.outputs == null ? [] : def.outputs;
  if (!Array.isArray(outputs) || !outputs.every(t => typeof t === 'string' && (t === '*' || TYPE_RE.test(t)))) throw new Error('outputs must be an array of UPPER_CASE types (or "*")');
  let names = def.output_names;
  if (names == null) names = outputs.slice();
  if (!Array.isArray(names) || names.length !== outputs.length || !names.every(n => typeof n === 'string')) throw new Error('output_names must be strings, one per output');
  if (def.fold !== undefined && typeof def.fold !== 'boolean') throw new Error('fold must be true or false');
  const str = (v) => (typeof v === 'string' ? v.trim() : '');
  return {
    display_name: str(def.display_name) || cls,
    category: str(def.category).replace(/^\/+|\/+$/g, '') || 'packs/' + packName,
    description: str(def.description),
    input,
    output: outputs.slice(),
    output_name: names.slice(),
    output_node: !!def.output_node,
    kiln_pack: packName,
  };
}

// ---------------------------------------------------------------------------
// helpers: git, recursive delete
// ---------------------------------------------------------------------------
function git(args, cwd, timeoutMs = 300000) {
  return new Promise((resolve, reject) => {
    let p;
    // no credential helpers / prompts: a private repo fails fast instead of opening a login window
    const env = Object.assign({}, process.env, { GIT_TERMINAL_PROMPT: '0', GCM_INTERACTIVE: 'never' });
    try {
      p = spawn('git', ['-c', 'credential.helper=', '-c', 'core.askPass=', ...args], { cwd, windowsHide: true, env });
    } catch (e) { reject(e); return; }
    let out = '';
    const add = (d) => { out += d; if (out.length > 20000) out = out.slice(-20000); };
    p.stdout.on('data', add);
    p.stderr.on('data', add);
    const t = setTimeout(() => { try { p.kill(); } catch (_) { } reject(new Error('git timed out')); }, timeoutMs);
    p.on('error', (e) => { clearTimeout(t); reject(e.code === 'ENOENT' ? new Error('git is not installed (not on PATH)') : e); });
    p.on('close', (code) => {
      clearTimeout(t);
      if (code === 0) resolve(out);
      else reject(new Error(out.trim().split(/\r?\n/).filter(Boolean).slice(-3).join(' · ') || `git exited with code ${code}`));
    });
  });
}
function chmodTree(p) {
  try {
    const st = fs.lstatSync(p);
    if (st.isSymbolicLink()) return;
    fs.chmodSync(p, 0o666 | (st.isDirectory() ? 0o111 : 0));
    if (st.isDirectory()) for (const e of fs.readdirSync(p)) chmodTree(path.join(p, e));
  } catch (_) { }
}
function rmrf(p) {
  let st;
  try { st = fs.lstatSync(p); } catch (_) { return; }
  if (st.isSymbolicLink()) { fs.unlinkSync(p); return; }  // a linked pack: remove the link only
  try { fs.rmSync(p, { recursive: true, force: true, maxRetries: 3 }); }
  catch (_) { chmodTree(p); fs.rmSync(p, { recursive: true, force: true, maxRetries: 3 }); } // git objects are read-only
}
const slug = (s) => String(s).trim().replace(/[^\w.-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 64);
const pascal = (s) => String(s).split(/[^A-Za-z0-9]+/).filter(Boolean).map(w => w[0].toUpperCase() + w.slice(1)).join('') || 'My';

// ---------------------------------------------------------------------------
// the registry
// ---------------------------------------------------------------------------
class Packs {
  // builtin(): the built-in catalog (class names packs may not reuse); log(msg)
  constructor({ dir, builtin, log }) {
    this.dir = path.resolve(dir);
    this.builtin = builtin;
    this.log = log || (() => { });
    this.packs = [];
    this.nodes = new Map();   // class -> { pack, def, entry }
    this.catalog = {};        // class -> catalog entry
    this.gen = 0;
    this.busy = null;
  }

  load() {
    const t0 = Date.now();
    this.packs = [];
    this.nodes = new Map();
    this.catalog = {};
    try { fs.mkdirSync(this.dir, { recursive: true }); } catch (_) { }
    let ents = [];
    try { ents = fs.readdirSync(this.dir, { withFileTypes: true }); } catch (_) { }
    const dirs = ents.filter(e => (e.isDirectory() || e.isSymbolicLink()) && !e.name.startsWith('.')).map(e => e.name).sort((a, b) => a.localeCompare(b));
    let builtin = {};
    try { builtin = this.builtin() || {}; } catch (_) { }
    for (const d of dirs) this.packs.push(this.loadPack(d, builtin));
    this.gen++;
    const s = this.summary();
    this.log(`extensions: ${s.packs} pack(s), ${s.nodes} node(s)${s.errors ? `, ${s.errors} with errors` : ''} (${Date.now() - t0} ms)`);
    for (const p of this.packs) {
      if (p.error) this.log(`  pack ${p.dir}: ${p.error}`);
      for (const [c, m] of Object.entries(p.node_errors)) this.log(`  pack ${p.dir}: node ${c}: ${m}`);
    }
    return s;
  }

  loadPack(dirName, builtin) {
    const full = path.join(this.dir, dirName);
    const p = {
      dir: dirName, path: full, name: dirName, version: '', description: '', author: '', url: '',
      enabled: !exists(path.join(full, DISABLED)), loaded: false, error: null, node_errors: {}, nodes: [],
      ui: null, git: exists(path.join(full, '.git')), load_ms: 0,
    };
    const t0 = Date.now();
    this.forget(full);
    if (!DIR_RE.test(dirName)) { p.error = 'folder name must be letters, digits, . _ - (max 64)'; return p; }
    let meta;
    try { meta = readJSONFile(path.join(full, 'kiln.json')); } catch (e) {
      p.error = e.code === 'ENOENT' ? 'kiln.json is missing (not a Kiln pack)' : 'kiln.json: ' + e.message;
      return p;
    }
    if (!meta || typeof meta !== 'object' || Array.isArray(meta)) { p.error = 'kiln.json must be an object'; return p; }
    if (typeof meta.name !== 'string' || !NAME_RE.test(meta.name.trim())) { p.error = 'kiln.json: "name" is missing or invalid (letters, digits, space . _ -, max 64)'; return p; }
    p.name = meta.name.trim();
    for (const k of ['version', 'description', 'author', 'url']) if (typeof meta[k] === 'string') p[k] = meta[k].slice(0, 500);
    const other = this.packs.find(x => x.name === p.name && x.loaded);
    if (other) { p.error = `another pack (folder "${other.dir}") is already called "${p.name}"`; return p; }
    if (!p.enabled) return p;
    const nodesFile = inside(full, meta.nodes == null ? 'nodes.js' : meta.nodes);
    if (!nodesFile) { p.error = 'kiln.json: "nodes" must be a file inside the pack'; return p; }
    if (!exists(nodesFile)) { p.error = `${path.relative(full, nodesFile).replace(/\\/g, '/')} is missing`; return p; }
    let mod;
    try { mod = require(nodesFile); } catch (e) { p.error = loadError(e, full); this.forget(full); return p; }
    const defs = mod && (mod.nodes || (mod.default && mod.default.nodes));
    if (!defs || typeof defs !== 'object' || Array.isArray(defs)) { p.error = 'nodes.js must export { nodes: { ClassName: {...} } }'; return p; }
    for (const [cls, def] of Object.entries(defs)) {
      try {
        if (builtin[cls]) throw new Error('a built-in Kiln node already has this name');
        if (this.nodes.has(cls)) throw new Error(`already defined by pack "${this.nodes.get(cls).pack.name}"`);
        const entry = normalizeNode(cls, def, p.name);
        this.nodes.set(cls, { pack: p, def, entry });
        this.catalog[cls] = entry;
        p.nodes.push({ class: cls, display_name: entry.display_name, category: entry.category });
      } catch (e) { p.node_errors[cls] = String(e.message || e).slice(0, MAX_ERR); }
    }
    if (meta.ui != null) {
      const ui = inside(full, meta.ui);
      if (!ui || !/\.m?js$/i.test(ui)) p.node_errors['(ui)'] = 'kiln.json: "ui" must be a .js file inside the pack';
      else if (!exists(ui)) p.node_errors['(ui)'] = `${meta.ui} is missing`;
      else p.ui = path.relative(full, ui).replace(/\\/g, '/');
    }
    p.loaded = true;
    p.load_ms = Date.now() - t0;
    return p;
  }

  // drop a pack's modules from the require cache so the next load re-reads them
  forget(full) {
    for (const k of Object.keys(require.cache)) if (k.startsWith(full + path.sep)) delete require.cache[k];
  }

  get(cls) { return this.nodes.get(cls) || null; }
  pack(dir) { return this.packs.find(p => p.dir === dir) || null; }
  summary() {
    return {
      packs: this.packs.length, enabled: this.packs.filter(p => p.enabled).length, nodes: this.nodes.size,
      errors: this.packs.filter(p => p.error || Object.keys(p.node_errors).length).length, gen: this.gen,
    };
  }
  list() {
    return this.packs.map(p => ({
      dir: p.dir, name: p.name, version: p.version, description: p.description, author: p.author, url: p.url,
      enabled: p.enabled, loaded: p.loaded, error: p.error, node_errors: p.node_errors, nodes: p.nodes, ui: p.ui,
      git: p.git, path: p.path, load_ms: p.load_ms,
    }));
  }
  // absolute path of a static file inside an enabled pack (for ui.js and its assets), or null
  staticFile(dir, rel) {
    const p = this.pack(dir);
    if (!p || !p.loaded || !p.enabled) return null;
    const f = inside(p.path, rel);
    if (!f || rel.split('/').some(s => s.startsWith('.')) || /(^|\/)(node_modules)(\/|$)/.test(rel)) return null;
    return f;
  }

  // ---- management (serialised: one operation at a time) ----
  async exclusive(label, fn) {
    if (this.busy) throw Object.assign(new Error(`another extension operation is running (${this.busy})`), { code: 409 });
    this.busy = label;
    try { return await fn(); } finally { this.busy = null; }
  }
  packDir(dir) {
    if (typeof dir !== 'string' || !DIR_RE.test(dir)) throw Object.assign(new Error('bad pack folder name'), { code: 400 });
    const full = path.join(this.dir, dir);
    if (!exists(full)) throw Object.assign(new Error(`no pack folder "${dir}"`), { code: 404 });
    return full;
  }

  install(source) {
    return this.exclusive('install', async () => {
      const src = String(source || '').trim();
      if (!src) throw Object.assign(new Error('give a git URL or a local folder path'), { code: 400 });
      if (/^(https?:\/\/|git@|ssh:\/\/)/i.test(src)) return this.installGit(src);
      return this.installLocal(src);
    });
  }
  async installGit(url) {
    if (!/^(https?:\/\/[^\s'"]+|git@[\w.-]+:[^\s'"]+|ssh:\/\/[^\s'"]+)$/i.test(url)) throw Object.assign(new Error('unsupported git URL (use https://…, ssh://… or git@host:path)'), { code: 400 });
    const base = url.replace(/[?#].*$/, '').replace(/\/+$/, '').split(/[/:]/).pop().replace(/\.git$/i, '');
    const dir = slug(base);
    if (!DIR_RE.test(dir)) throw Object.assign(new Error(`can't make a folder name from "${base}"`), { code: 400 });
    const target = path.join(this.dir, dir);
    if (exists(target)) throw Object.assign(new Error(`extensions/${dir} already exists (uninstall it first)`), { code: 409 });
    fs.mkdirSync(this.dir, { recursive: true });
    this.log(`extensions: cloning ${url} -> extensions/${dir}`);
    try { await git(['clone', '--depth', '1', '--', url, target], this.dir); }
    catch (e) { rmrf(target); throw Object.assign(new Error('git clone failed: ' + e.message), { code: 400 }); }
    if (!exists(path.join(target, 'kiln.json'))) {
      rmrf(target);
      throw Object.assign(new Error('not a Kiln pack: the repository has no kiln.json at its root'), { code: 400 });
    }
    this.load();
    return this.pack(dir);
  }
  async installLocal(src) {
    if (!path.isAbsolute(src)) throw Object.assign(new Error('local installs need an absolute folder path (e.g. C:\\packs\\my-pack)'), { code: 400 });
    const from = path.resolve(src);
    let st;
    try { st = fs.statSync(from); } catch (_) { throw Object.assign(new Error('folder not found: ' + from), { code: 404 }); }
    if (!st.isDirectory()) throw Object.assign(new Error('not a folder: ' + from), { code: 400 });
    if (from === this.dir || from.startsWith(this.dir + path.sep)) throw Object.assign(new Error('that folder is already inside extensions/'), { code: 400 });
    let meta;
    try { meta = readJSONFile(path.join(from, 'kiln.json')); } catch (e) { throw Object.assign(new Error('not a Kiln pack: ' + (e.code === 'ENOENT' ? 'no kiln.json in that folder' : 'kiln.json: ' + e.message)), { code: 400 }); }
    const dir = slug(path.basename(from)) || slug(meta && meta.name);
    if (!DIR_RE.test(dir)) throw Object.assign(new Error(`can't make a folder name from "${path.basename(from)}"`), { code: 400 });
    const target = path.join(this.dir, dir);
    if (exists(target)) throw Object.assign(new Error(`extensions/${dir} already exists (uninstall it first)`), { code: 409 });
    fs.mkdirSync(this.dir, { recursive: true });
    try {
      fs.cpSync(from, target, { recursive: true, filter: (s) => !path.relative(from, s).split(path.sep).includes('.git') });
    } catch (e) { rmrf(target); throw e; }
    this.log(`extensions: copied ${from} -> extensions/${dir}`);
    this.load();
    return this.pack(dir);
  }
  uninstall(dir) {
    return this.exclusive('uninstall', async () => {
      const full = this.packDir(dir);
      this.forget(full);
      rmrf(full);
      this.log(`extensions: removed extensions/${dir}`);
      this.load();
      return { removed: dir };
    });
  }
  setEnabled(dir, on) {
    return this.exclusive('setEnabled', async () => {
      const full = this.packDir(dir);
      const marker = path.join(full, DISABLED);
      if (on) { try { fs.unlinkSync(marker); } catch (_) { } }
      else fs.writeFileSync(marker, 'This pack is disabled in Kiln. Delete this file (or press Enable) to load it again.\n');
      this.log(`extensions: ${on ? 'enabled' : 'disabled'} ${dir}`);
      this.load();
      return this.pack(dir);
    });
  }
  update(dir) {
    return this.exclusive('update', async () => {
      const full = this.packDir(dir);
      if (!exists(path.join(full, '.git'))) throw Object.assign(new Error('not a git checkout (installed from a folder or created here)'), { code: 400 });
      const out = await git(['pull', '--ff-only'], full).catch((e) => { throw Object.assign(new Error('git pull failed: ' + e.message), { code: 400 }); });
      this.log(`extensions: updated ${dir}: ${out.trim().split(/\r?\n/).pop() || 'ok'}`);
      this.load();
      return Object.assign({}, this.pack(dir), { git_output: out.trim().slice(-400) });
    });
  }
  create(opts) {
    return this.exclusive('create', async () => {
      const name = String((opts && opts.name) || '').trim();
      if (!NAME_RE.test(name)) throw Object.assign(new Error('pack name: letters, digits, space . _ - (max 64)'), { code: 400 });
      const dir = slug(name);
      if (!DIR_RE.test(dir)) throw Object.assign(new Error('pick a name that starts with a letter or digit'), { code: 400 });
      const target = path.join(this.dir, dir);
      if (exists(target)) throw Object.assign(new Error(`extensions/${dir} already exists`), { code: 409 });
      const author = String((opts && opts.author) || '').trim().slice(0, 100);
      const description = String((opts && opts.description) || '').trim().slice(0, 300) || 'My Kiln nodes.';
      const cls = pascal(name) + 'InvertImage';
      fs.mkdirSync(target, { recursive: true });
      const meta = { name, version: '0.1.0', description, author, url: '', nodes: 'nodes.js' };
      fs.writeFileSync(path.join(target, 'kiln.json'), JSON.stringify(meta, null, 2) + '\n');
      fs.writeFileSync(path.join(target, 'nodes.js'), scaffoldNodes(name, cls));
      this.log(`extensions: created extensions/${dir}`);
      this.load();
      return Object.assign({}, this.pack(dir), { files: ['kiln.json', 'nodes.js'] });
    });
  }
}

function scaffoldNodes(name, cls) {
  const q = JSON.stringify;
  return `// ${name}: a Kiln pack. API: Kiln/docs/NODE_API.md
// Edit this file, then press Reload in the Extensions panel. Errors show up there, per pack.
'use strict';

module.exports = {
  nodes: {
    ${q(cls)}: {
      display_name: 'Invert Image',
      category: ${q(name)},
      description: 'Example node: inverts an image. Replace it with your own.',
      inputs: {
        required: {
          image: ['IMAGE'],
          strength: ['FLOAT', { default: 1.0, min: 0, max: 1, step: 0.05 }],
        },
      },
      outputs: ['IMAGE'],
      output_names: ['IMAGE'],
      // image.data is a Float32Array, interleaved [batch, height, width, 3], values 0..1
      async run({ image, strength }, ctx) {
        const out = ctx.image(image.batch, image.height, image.width);
        const src = image.data, dst = out.data;
        const rows = image.batch * image.height, row = image.width * 3;
        for (let y = 0; y < rows; y++) {
          for (let i = y * row, end = i + row; i < end; i++) dst[i] = src[i] + (1 - 2 * src[i]) * strength;
          if ((y & 63) === 0) {
            if (ctx.cancelled()) throw new Error('cancelled');
            ctx.progress(y, rows);
          }
        }
        return [out];
      },
    },
  },
};
`;
}

module.exports = { Packs, normalizeNode, checkSpec, TYPE_RE };
