/* Kiln txt2img + img2img (A1111-style): prompts, parameters, accordions, LoRA slots, presets,
   generate / interrupt / skip, img2img + inpaint (built as native graph jobs), the output panel
   (live preview, batch thumbnails, A1111 generation info, send-to buttons) and the history gallery. */
(() => {
  'use strict';
  const K = window.Kiln;
  const { $, el, clamp, api, toast, lsGet, lsSet, fmtSec, fmtMs, debounce } = K;

  // ---------------------------------------------------------------- A1111 prompt helpers (mirror of server/lib/a1111.js)
  const LORA_TAG = /<(lora|lyco):([^:<>]+)(?::([^:<>]*))?(?::[^<>]*)?>/gi;
  function extractLoraTags(prompt) {
    const tags = [];
    const text = String(prompt || '').replace(LORA_TAG, (m, kind, name, w) => {
      const n = w === undefined || !w.trim() ? 1 : Number(w);
      tags.push({ name: name.trim(), weight: Number.isFinite(n) ? n : 1, tag: m });
      return '';
    });
    if (!tags.length) return { text, tags };
    return { text: text.split('\n').map(l => l.replace(/(\s*,\s*){2,}/g, ', ').replace(/^\s*,\s*|\s*,\s*$/g, '').replace(/[ \t]{2,}/g, ' ').trim()).join('\n').trim(), tags };
  }
  const OPEN = '\u0001', CLOSE = '\u0002';
  function deemphasize(src) {
    src = String(src || '');
    if (!src.includes('[')) return src;
    let frame = [];
    const stack = [];
    for (let i = 0; i < src.length; i++) {
      const c = src[i];
      if (c === '\\' && i + 1 < src.length) { frame.push(c + src[i + 1]); i++; continue; }
      if (c === '[') { stack.push(frame); frame = []; continue; }
      if (c === ']' && stack.length) {
        const inner = frame.join('');
        frame = stack.pop();
        const level = inner.replace(/\\./g, '').replace(new RegExp(OPEN + '[^' + OPEN + CLOSE + ']*' + CLOSE, 'g'), '').replace(/\([^()]*\)/g, '');
        frame.push(/[:|]/.test(level) ? '[' + inner + ']' : OPEN + inner + CLOSE);
        continue;
      }
      frame.push(c);
    }
    while (stack.length) { const inner = frame.join(''); frame = stack.pop(); frame.push('[' + inner); }
    const expand = (s, w) => {
      let out = '';
      for (let i = 0; i < s.length; i++) {
        if (s[i] !== OPEN) { out += s[i]; continue; }
        let depth = 1, j = i + 1;
        for (; j < s.length && depth; j++) { if (s[j] === OPEN) depth++; else if (s[j] === CLOSE) depth--; }
        const nw = w / 1.1;
        out += `(${expand(s.slice(i + 1, j - 1), nw)}:${Math.round(nw * 10000) / 10000})`;
        i = j - 1;
      }
      return out;
    };
    return expand(frame.join(''), 1);
  }
  const SAMPLER_LABEL = { euler: 'Euler', euler_ancestral: 'Euler a', dpmpp_2m: 'DPM++ 2M', res_multistep: 'Res Multistep' };
  const SCHED_LABEL = { simple: 'Simple', sgm_uniform: 'SGM Uniform', karras: 'Karras', exponential: 'Exponential', ddim_uniform: 'DDIM Uniform', beta: 'Beta', normal: 'Normal', linear_quadratic: 'Linear Quadratic', kl_optimal: 'KL Optimal' };
  const stem = (f) => String(f || '').split('/').pop().replace(/\.(safetensors|sft|ckpt|pt)$/i, '');
  const q = (v) => (/[,:\n"]/.test(String(v)) ? JSON.stringify(String(v)) : String(v));
  function infotext(p, extra = {}) {
    const f = [];
    const add = (k, v) => { if (v !== undefined && v !== null && v !== '') f.push(`${k}: ${q(v)}`); };
    add('Steps', p.steps); add('Sampler', SAMPLER_LABEL[p.sampler] || p.sampler); add('Schedule type', SCHED_LABEL[p.scheduler || 'simple'] || p.scheduler);
    add('CFG scale', p.cfg); add('Seed', p.seed);
    if (p.width && p.height) add('Size', `${p.width}x${p.height}`);
    if (p.model) add('Model', stem(p.model));
    if (p.denoise != null && p.denoise < 1) add('Denoising strength', p.denoise);
    if (p.hires) {
      if (!(p.denoise != null && p.denoise < 1)) add('Denoising strength', p.hires.denoise);
      add('Hires upscale', p.hires.scale); add('Hires steps', p.hires.steps); add('Hires upscaler', p.hires.upscaler === 'model' ? '4x-AnimeSharp' : 'Lanczos');
    }
    if (p.face) {
      add('ADetailer model', 'face_yolov8m.pt'); add('ADetailer confidence', p.face.conf); add('ADetailer denoising strength', p.face.denoise);
      add('ADetailer steps', p.face.steps); add('ADetailer inpaint width', p.face.guide); add('ADetailer inpaint height', p.face.guide);
    }
    if (p.upscale) add('Postprocess upscale by', p.upscale.factor);
    if (Array.isArray(p.loras) && p.loras.length) add('Lora', p.loras.map(l => `${stem(l.file)}:${l.strength}`).join(', '));
    if (p.shift != null) add('Shift', p.shift);
    if (p.cfg > 1 && p.cfg_cutoff != null && p.cfg_cutoff < 1) add('CFG cutoff', p.cfg_cutoff);
    if (p.nag && !(p.cfg > 1)) { add('NAG scale', p.nag.scale); add('NAG tau', p.nag.tau); add('NAG alpha', p.nag.alpha); }
    if (p.step_cache > 0) add('Step cache', p.step_cache);
    for (const [k, v] of Object.entries(extra)) add(k, v);
    let s = String(p.prompt || '');
    if (p.negative) s += `\nNegative prompt: ${p.negative}`;
    return s + '\n' + f.join(', ');
  }
  // A1111 "parameters" text -> { prompt, negative, params: {Key: value} }
  function parseInfotext(text) {
    const lines = String(text || '').replace(/\r/g, '').trim().split('\n');
    let params = {};
    const last = lines[lines.length - 1] || '';
    if (lines.length && /(^|,\s*)Steps: \d+/.test(last)) {
      lines.pop();
      const re = /\s*(\w[\w \-/.+()]*):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)/g;
      let m;
      while ((m = re.exec(last)) && m[0]) {
        let v = m[2].trim();
        if (v.startsWith('"')) { try { v = JSON.parse(v); } catch (_) { } }
        params[m[1].trim()] = v;
      }
    }
    const ni = lines.findIndex(l => l.startsWith('Negative prompt:'));
    const prompt = (ni >= 0 ? lines.slice(0, ni) : lines).join('\n');
    const negative = ni >= 0 ? [lines[ni].slice('Negative prompt:'.length).trim(), ...lines.slice(ni + 1)].join('\n') : '';
    return { prompt, negative, params };
  }

  // ---------------------------------------------------------------- state
  const LS = 'kiln.a1.v1', LS_SDXL = 'kiln.a1.sdxl.v1', LS_FAM = 'kiln.a1.fam.v1', LS_STEPMS = 'kiln.stepms.v1';
  const SLOTS = 6;
  // each model family keeps its own settings (sampler, size, CFG, negative, LoRA slots...);
  // these follow you across families
  const SHARED = ['prompt', 'seed', 'batchCount', 'batchSize', 'denoise', 'resize', 'i2iMode', 'xnetOpen', 'accOpen'];
  let FAM = lsGet(LS_FAM) === 'sdxl' ? 'sdxl' : 'anima';
  const lsKey = (f) => (f === 'sdxl' ? LS_SDXL : LS);
  const DEF = {
    prompt: '', negative: 'worst quality, low quality, blurry, jpeg artifacts, watermark',
    sampler: 'euler', scheduler: 'simple', steps: 8, width: 512, height: 768, batchCount: 1, batchSize: 1, cfg: 1, seed: -1,
    cfgCutoff: 1, stepCache: 0, shift: 3, nag: { on: true, scale: 5, tau: 2.5, alpha: 0.25 },
    hires: { on: false, upscaler: 'lanczos', steps: 4, denoise: 0.4, scale: 1.5, tw: 0, th: 0 },
    face: { on: false, conf: 0.35, max_faces: 4, denoise: 0.4, steps: 4, guide: 384, max_size: 576, crop: 2 },
    upscale: 0, slots: [], denoise: 0.6, resize: 'just', i2iMode: 'img2img', xnetOpen: false,
    accOpen: { hires: false, face: false, lora: true, kiln: false },
  };
  // SDXL / Illustrious / NoobAI / Pony defaults
  const DEF_SDXL = {
    negative: 'worst quality, low quality, lowres, bad anatomy, watermark, signature',
    sampler: 'euler_ancestral', scheduler: 'normal', steps: 28, width: 832, height: 1216, cfg: 5.5, cfgCutoff: 1, stepCache: 0,
  };
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const defFor = (f) => (f === 'sdxl' ? Object.assign(clone(DEF), clone(DEF_SDXL)) : clone(DEF));
  function merged(src, f = FAM) {
    const D = defFor(f);
    const G = Object.assign(clone(D), src && typeof src === 'object' ? src : {});
    delete G.family;
    for (const k of ['nag', 'hires', 'face', 'accOpen']) G[k] = Object.assign(clone(D[k]), src && src[k] && typeof src[k] === 'object' ? src[k] : {});
    if (!Array.isArray(G.slots)) G.slots = [];
    G.slots = G.slots.filter(s => s && typeof s === 'object').map(s => ({ on: !!s.on, file: typeof s.file === 'string' ? s.file : '', strength: Number.isFinite(Number(s.strength)) ? Number(s.strength) : 1 }));
    while (G.slots.length < SLOTS) G.slots.push({ on: false, file: '', strength: 1 });
    return G;
  }
  // one-time migration from the previous Simple mode's settings
  function migrate() {
    const o = lsGet('kiln.settings.v1');
    if (!o || typeof o !== 'object') return null;
    const G = {};
    for (const k of ['prompt', 'negative', 'width', 'height', 'steps', 'cfg', 'sampler', 'shift', 'cfgCutoff', 'stepCache']) if (o[k] !== undefined) G[k] = o[k];
    if (o.seedLocked && o.seed !== '') G.seed = Number(o.seed);
    if (o.nag) G.nag = o.nag;
    if (o.enh) { if (o.enh.hires) G.hires = o.enh.hires; if (o.enh.face) G.face = o.enh.face; G.upscale = Number(o.enh.upscale) || 0; }
    if (o.loras) G.slots = Object.entries(o.loras).filter(([, v]) => v && v.on).map(([file, v]) => ({ on: true, file, strength: Number(v.strength) || 1 }));
    return G;
  }
  let G = FAM === 'sdxl' ? merged(lsGet(LS_SDXL), 'sdxl') : merged(lsGet(LS) || migrate(), 'anima');
  const save = debounce(() => lsSet(lsKey(FAM), G), 250);
  const isSDXL = () => FAM === 'sdxl';
  const famName = (f) => (f === 'sdxl' ? 'SDXL' : 'Anima');
  // the checkpoint's family decides which settings are live; switching saves one set and loads the other
  function syncFamily() {
    const c = K.ckptInfo();
    if (!c) return false;
    const f = c.family === 'sdxl' ? 'sdxl' : 'anima';
    if (f === FAM) return false;
    lsSet(lsKey(FAM), G);
    const shared = {};
    for (const k of SHARED) if (G[k] !== undefined) shared[k] = clone(G[k]);
    const had = lsGet(lsKey(f));
    FAM = f;
    lsSet(LS_FAM, f);
    G = merged(Object.assign({}, had || {}, shared), f);
    save();
    renderAll();
    toast(had ? `${famName(f)} settings restored (each model family keeps its own)` : `${famName(f)} defaults: ${G.width}×${G.height} · ${G.steps} steps · CFG ${G.cfg} · ${SAMPLER_LABEL[G.sampler] || G.sampler} · ${SCHED_LABEL[G.scheduler] || G.scheduler}`, 'ok', 3200);
    return true;
  }
  const getKey = (key) => key.split('.').reduce((o, k) => (o == null ? undefined : o[k]), G);
  const setKey = (key, v) => { const ks = key.split('.'); let o = G; for (const k of ks.slice(0, -1)) o = o[k]; o[ks[ks.length - 1]] = v; };
  const tab = () => (K.S.tab === 'img2img' ? 'img2img' : 'txt2img');

  // ---------------------------------------------------------------- sliders
  const sliders = new Map();
  function buildSlider(box) {
    const key = box.dataset.key;
    const min = Number(box.dataset.min), max = Number(box.dataset.max), step0 = Number(box.dataset.step);
    const grid = () => ((key === 'width' || key === 'height') && FAM === 'sdxl' ? 32 : step0);
    const id = 'sl_' + key.replace(/\W/g, '_');
    const head = el('div', 'a-sl-head');
    const lab = el('label', '', box.dataset.label);
    lab.htmlFor = id;
    const num = el('input');
    num.type = 'number'; num.min = String(min); num.step = String(step0); num.inputMode = 'decimal';
    if (!['steps', 'batchCount', 'width', 'height', 'hires.tw', 'hires.th'].includes(key)) num.max = String(max);
    num.setAttribute('aria-label', box.dataset.label);
    const range = el('input');
    range.type = 'range'; range.id = id; range.min = String(min); range.max = String(max); range.step = String(step0);
    head.append(lab, num);
    box.append(head, range);
    const fix = (v) => {
      const step = grid();
      v = Number(v);
      if (!Number.isFinite(v)) v = getKey(key);
      v = Math.max(min, v);
      if (num.max) v = Math.min(max, v);
      const dec = (String(step).split('.')[1] || '').length;
      v = dec ? Math.round(v / step) * step : Math.round(v / step) * step;
      return Number(v.toFixed(Math.max(dec, 0)));
    };
    const commit = (v) => { setKey(key, v); save(); onParam(key); };
    range.addEventListener('input', () => { const v = fix(range.value); num.value = v; commit(v); });
    num.addEventListener('change', () => { let v = fix(num.value); if (key === 'width' || key === 'height') v = clamp(Math.round(v / grid()) * grid(), 256, 4096); num.value = v; range.value = clamp(v, min, max); commit(v); });
    sliders.set(key, { update: () => { const v = getKey(key); range.step = num.step = String(grid()); num.value = v; range.value = clamp(Number(v), min, max); }, box });
  }
  for (const b of document.querySelectorAll('.a-slider[data-key]')) buildSlider(b);
  const renderSliders = () => { for (const s of sliders.values()) s.update(); };

  // ---------------------------------------------------------------- selects, radios, checkboxes
  const ENGINE_SAMPLERS = ['euler', 'dpmpp_2m', 'res_multistep'];
  const SDXL_SAMPLERS = ['euler', 'euler_ancestral', 'dpmpp_2m', 'res_multistep'];
  const SDXL_SCHEDS = ['normal', 'karras', 'simple', 'sgm_uniform', 'exponential', 'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal'];
  function fillSel(sel, list, labels, cur) {
    sel.textContent = '';
    for (const v of list) { const o = el('option', '', labels[v] || v); o.value = v; sel.appendChild(o); }
    sel.value = cur;
  }
  function renderSamplers() {
    if (isSDXL()) {
      // SDXL: the engine's SDXL samplers and every schedule type
      if (!SDXL_SAMPLERS.includes(G.sampler)) G.sampler = 'euler_ancestral';
      if (!SDXL_SCHEDS.includes(G.scheduler)) G.scheduler = 'normal';
      fillSel($('sampler'), SDXL_SAMPLERS, SAMPLER_LABEL, G.sampler);
      fillSel($('scheduler'), SDXL_SCHEDS, SCHED_LABEL, G.scheduler);
      $('scheduler').title = '';
      return;
    }
    const sel = $('sampler');
    const list = (K.S.engineSamplers && K.S.engineSamplers.length ? K.S.engineSamplers : ENGINE_SAMPLERS).slice();
    if (tab() === 'img2img' && !list.includes('euler_ancestral')) list.push('euler_ancestral');
    if (!list.includes(G.sampler)) list.push(G.sampler);
    sel.textContent = '';
    for (const v of list) { const o = el('option', '', SAMPLER_LABEL[v] || v); o.value = v; sel.appendChild(o); }
    sel.value = G.sampler;
    const sc = $('scheduler');
    sc.textContent = '';
    const scheds = tab() === 'img2img' ? Object.keys(SCHED_LABEL) : ['simple'];
    for (const v of scheds) { const o = el('option', '', SCHED_LABEL[v]); o.value = v; sc.appendChild(o); }
    sc.value = scheds.includes(G.scheduler) ? G.scheduler : 'simple';
    sc.title = tab() === 'txt2img' ? 'txt2img uses the Simple schedule with Shift (Kiln options); img2img offers all schedules' : '';
  }
  $('sampler').addEventListener('change', () => { G.sampler = $('sampler').value; save(); onParam('sampler'); });
  $('scheduler').addEventListener('change', () => { G.scheduler = $('scheduler').value; save(); });
  function radio(box, get, set) {
    const render = () => { for (const b of box.querySelectorAll('button')) b.setAttribute('aria-checked', String(b.dataset.v === String(get()))); };
    box.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b || b.disabled) return; set(b.dataset.v); render(); save(); });
    return render;
  }
  const renderCache = radio($('cacheSeg'), () => G.stepCache, (v) => { G.stepCache = Number(v); onParam('stepCache'); });
  const renderUp = radio($('upSeg'), () => G.upscale, (v) => { G.upscale = Number(v); onParam('upscale'); });
  const renderResize = radio($('resizeMode'), () => G.resize, (v) => { G.resize = v; drawI2i(); });
  const check = (id, key) => $(id).addEventListener('change', () => { setKey(key, $(id).checked); save(); onParam(key); });
  check('hiresOn', 'hires.on'); check('faceOn', 'face.on'); check('nagOn', 'nag.on');
  $('hiresUpscaler').addEventListener('change', () => { G.hires.upscaler = $('hiresUpscaler').value; save(); onParam('hires.upscaler'); });
  // clicking the checkbox in an accordion header must not toggle the accordion
  for (const c of document.querySelectorAll('.a-acc-check')) c.addEventListener('click', (e) => e.stopPropagation());
  for (const [id, k] of [['accHires', 'hires'], ['accFace', 'face'], ['accLora', 'lora'], ['accKiln', 'kiln']]) {
    $(id).addEventListener('toggle', () => { G.accOpen[k] = $(id).open; save(); });
  }
  $('swapWH').addEventListener('click', () => { [G.width, G.height] = [G.height, G.width]; save(); renderSliders(); onParam('width'); });
  $('seed').addEventListener('change', () => {
    const v = $('seed').value.trim();
    G.seed = v === '' || v === '-1' || !/^\d+$/.test(v) ? -1 : Math.min(9007199254740991, Number(v));
    $('seed').value = String(G.seed);
    save();
  });
  $('seedRandom').addEventListener('click', () => { G.seed = -1; $('seed').value = '-1'; save(); });
  $('seedReuse').addEventListener('click', () => {
    const it = OUT.items[OUT.sel] || OUT.items[0];
    const seed = it && it.params && it.params.seed;
    if (seed == null) { toast('No image yet to take the seed from'); return; }
    G.seed = seed; $('seed').value = String(seed); save();
    toast(`Seed ${seed}`, 'ok', 1400);
  });

  // ---------------------------------------------------------------- prompts + token counters
  const promptEl = $('prompt'), negEl = $('negative');
  function autoGrow(ta, min) { ta.style.height = 'auto'; ta.style.height = `${Math.max(min, Math.min(window.innerHeight * 0.45, ta.scrollHeight + 2))}px`; }
  const tokCount = async (text, outId) => {
    const out = $(outId);
    if (!text.trim()) { out.textContent = ''; return; }
    try {
      const r = await api('/api/tokenize', { method: 'POST', body: isSDXL() ? { prompt: text, family: 'sdxl' } : { prompt: text } });
      if (r.sdxl) {
        // A1111's counter: CLIP works in 75-token chunks
        out.textContent = `${r.sdxl.tokens}/${Math.max(1, r.sdxl.chunks) * 75}`;
        out.title = `CLIP tokens (SDXL): ${r.sdxl.tokens} in ${r.sdxl.chunks} chunk${r.sdxl.chunks === 1 ? '' : 's'} of 75`;
        return;
      }
      if (!r.qwen_ids) { out.textContent = ''; return; }
      out.textContent = `${r.qwen_ids.length} / ${r.t5_ids.length}`;
      out.title = `Qwen ${r.qwen_ids.length} tokens · T5 ${r.t5_ids.length} tokens${r.t5_weights.some(w => w !== 1) ? ' · weighted' : ''}`;
    } catch (_) { out.textContent = ''; }
  };
  const tokPos = debounce(() => tokCount(G.prompt, 'tokCount'), 350);
  const tokNeg = debounce(() => tokCount(G.negative, 'tokNeg'), 350);
  const promptChanged = debounce(() => K.emit('prompt'), 300);
  promptEl.addEventListener('input', () => { G.prompt = promptEl.value; autoGrow(promptEl, 76); save(); tokPos(); renderSlotWarn(); promptChanged(); });
  negEl.addEventListener('input', () => { G.negative = negEl.value; autoGrow(negEl, 58); save(); tokNeg(); });
  $('toolClear').addEventListener('click', () => { if (!G.prompt || confirm('Clear the prompt?')) { G.prompt = ''; promptEl.value = ''; save(); tokPos(); } });
  $('toolReuse').addEventListener('click', () => {
    const it = OUT.items[OUT.sel] || S.gallery[0];
    if (!it || !it.params) { toast('No image with parameters yet'); return; }
    applyParams(it.params);
    toast('Parameters loaded', 'ok', 1500);
  });
  // insert / remove <lora:name:w> (A1111 extra networks toggle)
  function toggleLoraTag(name, weight = 1) {
    const re = new RegExp(`\\s*,?\\s*<(lora|lyco):${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(:[^>]*)?>`, 'i');
    let t = promptEl.value;
    if (re.test(t)) { t = t.replace(re, ''); toast(`Removed <lora:${name}>`, '', 1400); }
    else { t = (t.trim() ? t.replace(/\s*$/, '') + ', ' : '') + `<lora:${name}:${weight}>`; }
    promptEl.value = t; G.prompt = t; autoGrow(promptEl, 76); save(); tokPos();
    K.emit('prompt');
  }
  function insertText(words) {
    let t = promptEl.value;
    const add = words.filter(w => !t.includes(w));
    if (!add.length) return;
    t = (t.trim() ? t.replace(/\s*$/, '') + ', ' : '') + add.join(', ');
    promptEl.value = t; G.prompt = t; autoGrow(promptEl, 76); save(); tokPos();
  }

  // ---------------------------------------------------------------- LoRA slots
  const loras = () => (K.S.models && K.S.models.loras) || [];
  const loraBy = (file) => loras().find(l => l.file === file);
  const famLabel = (f) => K.FAMILY[f] || f;
  function mismatch(l) {
    const c = K.ckptInfo();
    if (!l || !c || !l.family || l.family === 'unknown' || !c.family || c.family === 'unknown') return '';
    return l.family !== c.family ? `made for ${l.base || famLabel(l.family)}, the checkpoint is ${c.base || famLabel(c.family)}` : '';
  }
  function renderSlots() {
    const box = $('loraSlots');
    box.textContent = '';
    G.slots.forEach((s, i) => box.appendChild(slotRow(s, i)));
    renderSlotSum();
  }
  function renderSlotSum() {
    const on = G.slots.filter(s => s.on && s.file);
    const tags = extractLoraTags(G.prompt).tags.length;
    $('loraSum').textContent = on.length || tags ? `${on.map(s => `${stem(s.file)}:${s.strength}`).join(', ')}${tags ? ` + ${tags} in prompt` : ''}` : 'none';
    $('accLora').classList.toggle('on', on.length > 0);
  }
  function renderSlotWarn() {
    const rows = $('loraSlots').children;
    G.slots.forEach((s, i) => { const w = rows[i] && rows[i].querySelector('.a-slot-warn'); if (w) w.textContent = s.file ? mismatch(loraBy(s.file)) : ''; });
    renderSlotSum();
    renderWarn();
  }
  function slotRow(s, i) {
    const row = el('div', 'a-slot' + (s.on ? '' : ' off'));
    const cb = el('input'); cb.type = 'checkbox'; cb.checked = s.on; cb.title = 'Enable'; cb.setAttribute('aria-label', `Enable LoRA slot ${i + 1}`);
    const pick = el('div', 'a-slot-pick');
    const inp = el('input'); inp.type = 'text'; inp.placeholder = 'None'; inp.autocomplete = 'off'; inp.spellcheck = false;
    inp.setAttribute('aria-label', `LoRA slot ${i + 1}`);
    const cur = () => { const l = loraBy(s.file); return l ? l.name : s.file ? stem(s.file) + ' (missing)' : ''; };
    inp.value = cur();
    pick.appendChild(inp);
    const range = el('input'); range.type = 'range'; range.min = '-2'; range.max = '2'; range.step = '0.05'; range.value = s.strength;
    const num = el('input'); num.type = 'number'; num.min = '-2'; num.max = '2'; num.step = '0.05'; num.value = s.strength; num.inputMode = 'decimal';
    num.setAttribute('aria-label', `Strength ${i + 1}`);
    const x = el('button', 'a-slot-x', '✕'); x.type = 'button'; x.title = i < SLOTS ? 'Clear slot' : 'Remove slot';
    const warn = el('div', 'a-slot-warn', s.file ? mismatch(loraBy(s.file)) : '');
    row.append(cb, pick, range, num, x, warn);
    const setStr = (v) => { v = clamp(Math.round(Number(v) * 100) / 100, -2, 2); if (!Number.isFinite(v)) return; s.strength = v; range.value = v; num.value = v; save(); renderSlotSum(); };
    cb.addEventListener('change', () => { s.on = cb.checked; row.classList.toggle('off', !s.on); save(); renderSlotSum(); renderWarn(); });
    range.addEventListener('input', () => setStr(range.value));
    num.addEventListener('change', () => setStr(num.value));
    x.addEventListener('click', () => { if (i >= SLOTS) G.slots.splice(i, 1); else Object.assign(s, { on: false, file: '', strength: 1 }); save(); renderSlots(); renderWarn(); });
    // searchable dropdown
    let dd = null, sel = 0, items = [];
    const close = () => { if (dd) { dd.remove(); dd = null; } inp.value = cur(); };
    const choose = (l) => {
      s.file = l ? l.file : ''; s.on = !!l;
      cb.checked = s.on; row.classList.toggle('off', !s.on);
      inp.value = cur(); warn.textContent = l ? mismatch(l) : '';
      if (dd) { dd.remove(); dd = null; }
      save(); renderSlotSum(); renderWarn();
    };
    const render = () => {
      const qq = inp.value.trim().toLowerCase();
      items = [null, ...loras().filter(l => !qq || l.name.toLowerCase().includes(qq) || l.file.toLowerCase().includes(qq) || (l.words || []).some(w => w.toLowerCase().includes(qq)))];
      sel = clamp(sel, 0, items.length - 1);
      if (!dd) { dd = el('div', 'a-dd'); pick.appendChild(dd); }
      dd.textContent = '';
      items.slice(0, 200).forEach((l, k) => {
        const it = el('div', 'a-dd-item' + (k === sel ? ' sel' : '') + (l && mismatch(l) ? ' bad' : ''));
        it.appendChild(el('span', '', l ? l.name : 'None'));
        if (l) it.appendChild(el('small', '', l.base || famLabel(l.family)));
        it.addEventListener('pointerdown', (e) => { e.preventDefault(); choose(l); });
        dd.appendChild(it);
      });
      if (items.length === 1 && qq) dd.appendChild(el('div', 'a-dd-empty', 'No LoRA matches. Try the CivitAI tab in Extra Networks.'));
    };
    inp.addEventListener('focus', () => { inp.select(); sel = 0; render(); });
    inp.addEventListener('input', () => { sel = 1; render(); });
    inp.addEventListener('blur', () => setTimeout(close, 120));
    inp.addEventListener('keydown', (e) => {
      if (!dd) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(items.length - 1, sel + 1); render(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); render(); }
      else if (e.key === 'Enter') {
        e.preventDefault();
        // typed a search then Enter: the first match (even if focus handling reset the highlight)
        const typed = inp.value.trim() && inp.value !== cur();
        choose(sel === 0 && typed && items[1] ? items[1] : items[sel]);
        inp.blur();
      }
      else if (e.key === 'Escape') { inp.blur(); }
    });
    return row;
  }
  $('slotAdd').addEventListener('click', () => { G.slots.push({ on: false, file: '', strength: 1 }); save(); renderSlots(); });
  function addToSlot(file, strength = 1) {
    let s = G.slots.find(x => x.file === file) || G.slots.find(x => !x.file);
    if (!s) { s = { on: false, file: '', strength: 1 }; G.slots.push(s); }
    Object.assign(s, { on: true, file, strength });
    save(); renderSlots(); renderWarn();
    $('accLora').open = true;
  }

  // ---------------------------------------------------------------- derived hints + availability
  const feat = () => (K.S.models && K.S.models.features) || null;
  function hiresScale() {
    const h = G.hires;
    if (h.tw > 0 || h.th > 0) return Math.max(h.tw > 0 ? h.tw / G.width : 0, h.th > 0 ? h.th / G.height : 0, 1);
    return h.scale;
  }
  function onParam(key) {
    if (key === 'width' || key === 'height') { fitFrame(); drawI2i(); }
    renderAccSums();
    renderQuick();
    renderNeg();
  }
  function renderNeg() {
    if (isSDXL()) {
      const inert = !(G.cfg > 1);
      $('negField').classList.toggle('inert', inert);
      negEl.title = inert ? 'CFG is 1: the negative prompt has no effect on SDXL (NAG is Anima-only). Raise CFG.' : '';
      $('nagHint').textContent = 'Anima-only: SDXL uses the negative prompt through CFG.';
      return;
    }
    const inert = !(G.cfg > 1) && !G.nag.on;
    $('negField').classList.toggle('inert', inert);
    negEl.title = inert ? 'CFG is 1 and NAG is off: the negative prompt has no effect. Turn on NAG (Kiln options) or raise CFG.' : '';
    $('nagHint').textContent = G.cfg > 1 ? 'CFG > 1 uses the classic negative pass; NAG only applies at CFG 1.' : G.nag.on ? 'Negative prompt applied via NAG (turbo-friendly).' : 'Off: at CFG 1 the negative prompt is ignored.';
  }
  function renderAccSums() {
    const f = feat(), h = G.hires;
    const hs = hiresScale();
    const hw = Math.round(G.width * hs / 16) * 16, hh = Math.round(G.height * hs / 16) * 16;
    $('hiresSum').textContent = h.on ? `${hs.toFixed(2)}× → ${hw}×${hh} · ${h.steps} st · ${h.denoise}` : '';
    $('accHires').classList.toggle('on', !!h.on);
    const noModel = f && !f.upscale_model;
    $('hiresUpscaler').querySelector('[value=model]').disabled = !!noModel;
    $('hiresNote').textContent = (h.tw > 0 || h.th > 0 ? `Resize to ${h.tw || '–'}×${h.th || '–'} = ${hs.toFixed(2)}× → ${hw}×${hh} (Kiln keeps the aspect ratio). ` : '') + (noModel ? `Model upscaler unavailable (${f.reasons.upscale_model || 'missing model'}).` : '');
    $('faceSum').textContent = G.face.on ? `${G.face.steps} st · ${G.face.denoise}` : '';
    $('accFace').classList.toggle('on', !!G.face.on);
    const sd = isSDXL();
    const noFace = sd || (f && !f.face);
    $('faceOn').disabled = !!noFace;
    $('accFace').classList.toggle('disabled', !!noFace);
    $('faceNote').textContent = sd ? 'Not on SDXL yet: the Face Detailer runs with Anima checkpoints only for now.' : noFace ? `Unavailable: ${f.reasons.face || 'needs models/detect/face_yolov8m'}` : 'Kiln\'s face pass (like ADetailer): detects faces, redraws each at the inpaint size, pastes it back.';
    if (sd) $('faceSum').textContent = G.face.on ? 'off on SDXL' : '';
    // Kiln options that only exist for Anima
    $('nagOn').disabled = sd;
    for (const k of ['nag.scale', 'nag.tau', 'nag.alpha', 'shift']) { const sl = sliders.get(k); if (sl) { sl.box.classList.toggle('a-na', sd); for (const i of sl.box.querySelectorAll('input')) i.disabled = sd; } }
    for (const b of $('cacheSeg').querySelectorAll('button')) b.disabled = sd;
    // LoRA slots: not on SDXL yet
    let ln = $('loraNote');
    if (!ln) { ln = el('div', 'a-hint a-fam-note'); ln.id = 'loraNote'; $('loraSlots').before(ln); }
    ln.hidden = !sd;
    ln.textContent = sd ? 'LoRAs are not supported on SDXL yet. Slots and <lora:> tags are skipped for SDXL checkpoints (you get a notice when that happens).' : '';
    for (const b of $('upSeg').querySelectorAll('button')) b.disabled = b.dataset.v !== '0' && !!(f && !f.upscale_model);
    const k = [];
    if (!sd && G.nag.on && !(G.cfg > 1)) k.push('NAG'); if (!sd && G.stepCache > 0) k.push('cache ' + G.stepCache); if (G.cfg > 1 && G.cfgCutoff < 1) k.push('cutoff ' + G.cfgCutoff);
    if (G.upscale) k.push(G.upscale + '× up'); if (!sd && G.shift !== 3) k.push('shift ' + G.shift);
    $('kilnSum').textContent = k.join(' · ');
    $('kilnHint').textContent = sd ? 'SDXL: NAG, Shift and Step cache are Anima-only. CFG cutoff and the final upscale work.' : G.stepCache > 0 && G.steps < 20 ? 'Step cache helps most at 20+ steps.' : '';
  }
  // quick presets (turbo LoRA on/off)
  const turboLora = () => loras().find(l => /turbo/i.test(l.file));
  const turboOn = () => { const t = turboLora(); return !!t && G.slots.some(s => s.on && s.file === t.file && s.strength !== 0); };
  const MODE_PRESETS = { base: { steps: 20, cfg: 4.5, sampler: 'dpmpp_2m', cfgCutoff: 0.7, turbo: false }, turbo: { steps: 8, cfg: 1, sampler: 'euler', turbo: true } };
  const SDXL_BASE = { steps: 28, cfg: 5.5, sampler: 'euler_ancestral', scheduler: 'normal', cfgCutoff: 1 };
  function renderQuick() {
    if (isSDXL()) {
      const p = SDXL_BASE;
      $('presetBase').setAttribute('aria-pressed', String(G.steps === p.steps && G.cfg === p.cfg && G.sampler === p.sampler && G.scheduler === p.scheduler));
      $('presetBase').title = 'SDXL: 28 steps · CFG 5.5 · Euler a · Normal';
      $('presetTurbo').setAttribute('aria-pressed', 'false');
      $('presetTurbo').disabled = true;
      $('presetTurbo').title = 'Turbo is the Anima turbo LoRA (Anima checkpoints only)';
      return;
    }
    $('presetBase').title = 'Base: 20 steps · CFG 4.5 · DPM++ 2M · cutoff 0.7 · turbo off';
    $('presetTurbo').title = 'Turbo: turbo LoRA 1.0 · 8 steps · CFG 1 · Euler';
    const t = turboLora();
    const m = (p) => G.steps === p.steps && G.cfg === p.cfg && G.sampler === p.sampler && (!t || turboOn() === p.turbo) && (p.cfgCutoff == null || G.cfgCutoff === p.cfgCutoff);
    $('presetBase').setAttribute('aria-pressed', String(m(MODE_PRESETS.base)));
    $('presetTurbo').setAttribute('aria-pressed', String(m(MODE_PRESETS.turbo)));
    $('presetTurbo').disabled = !t;
  }
  function quick(key) {
    if (isSDXL()) {
      if (key !== 'base') return;
      Object.assign(G, clone(SDXL_BASE));
      save(); renderAll();
      toast('SDXL: 28 steps · CFG 5.5 · Euler a · Normal', 'ok', 2200);
      return;
    }
    const p = MODE_PRESETS[key];
    Object.assign(G, { steps: p.steps, cfg: p.cfg, sampler: p.sampler });
    if (p.cfgCutoff != null) G.cfgCutoff = p.cfgCutoff;
    const t = turboLora();
    if (t) {
      if (p.turbo) addToSlot(t.file, 1);
      else for (const s of G.slots) if (s.file === t.file) s.on = false;
    }
    save(); renderAll();
    toast(key === 'base' ? 'Base: 20 steps · CFG 4.5 · DPM++ 2M · cutoff 0.7 · turbo off' : 'Turbo: turbo LoRA 1.0 · 8 steps · CFG 1 · Euler', 'ok', 2200);
  }
  $('presetBase').addEventListener('click', () => quick('base'));
  $('presetTurbo').addEventListener('click', () => quick('turbo'));

  // named presets (stored on the server, shared by every device)
  const PRESET_KEYS = ['sampler', 'scheduler', 'steps', 'width', 'height', 'batchCount', 'batchSize', 'cfg', 'cfgCutoff', 'stepCache', 'shift', 'nag', 'hires', 'face', 'upscale', 'slots', 'denoise'];
  const snapshot = (keys = PRESET_KEYS) => Object.assign(clone(Object.fromEntries(keys.map(k => [k, G[k]]))), { family: FAM });
  function renderPresets() {
    const sel = $('presetSel');
    const ps = (K.S.settings && K.S.settings.ui && K.S.settings.ui.presets) || {};
    const cur = sel.value;
    sel.textContent = '';
    const o0 = el('option', '', Object.keys(ps).length ? 'Presets…' : 'No presets yet'); o0.value = ''; sel.appendChild(o0);
    for (const n of Object.keys(ps).sort()) { const o = el('option', '', n + (ps[n] && ps[n].family === 'sdxl' ? ' (SDXL)' : '')); o.value = n; sel.appendChild(o); }
    sel.value = ps[cur] ? cur : '';
  }
  $('presetSel').addEventListener('change', () => {
    const ps = (K.S.settings && K.S.settings.ui && K.S.settings.ui.presets) || {};
    const p = ps[$('presetSel').value];
    if (!p) return;
    const pf = p.family === 'sdxl' ? 'sdxl' : 'anima';
    G = merged(Object.assign({}, G, clone(p)));
    save(); renderAll();
    toast(pf !== FAM ? `Preset "${$('presetSel').value}" was saved for ${famName(pf)}; the checkpoint is ${famName(FAM)}` : `Preset "${$('presetSel').value}" applied`, pf !== FAM ? 'err' : 'ok', pf !== FAM ? 4000 : 1500);
  });
  $('presetSave').addEventListener('click', async () => {
    const name = (prompt('Save the current settings (not the prompt) as preset:', $('presetSel').value || '') || '').trim();
    if (!name) return;
    try {
      const r = await api('/api/settings', { method: 'POST', body: { presets: { [name]: snapshot() } } });
      K.S.settings = Object.assign(K.S.settings || {}, { ui: r.ui });
      renderPresets(); $('presetSel').value = name;
      toast(`Preset "${name}" saved`, 'ok', 1500);
    } catch (e) { toast('Could not save the preset: ' + e.message, 'err'); }
  });
  $('presetDel').addEventListener('click', async () => {
    const name = $('presetSel').value;
    if (!name || !confirm(`Delete preset "${name}"?`)) return;
    try {
      const r = await api('/api/settings', { method: 'POST', body: { presets: { [name]: null } } });
      K.S.settings = Object.assign(K.S.settings || {}, { ui: r.ui });
      renderPresets();
    } catch (e) { toast(e.message, 'err'); }
  });

  // ---------------------------------------------------------------- warnings + generate availability
  function renderWarn() {
    const c = K.ckptInfo();
    const w = [];
    let block = '';
    if (c && !c.runnable) block = c.family === 'sdxl' ? `SDXL support is coming soon (${c.name}). Pick an Anima checkpoint to generate.` : `${c.name} isn't a model Kiln can run yet.`;
    else if (isSDXL() && tab() === 'img2img') block = `img2img and inpaint run on Anima checkpoints for now. ${c ? c.name + ' is SDXL: ' : ''}use txt2img, or pick an Anima checkpoint.`;
    if (tab() === 'img2img' && !I2I.img) w.push('Add an image to img2img first.');
    if (isSDXL() && tab() === 'txt2img') {
      const nl = G.slots.filter(s => s.on && s.file && s.strength !== 0).length + extractLoraTags(G.prompt).tags.length;
      if (nl) w.push(`LoRAs are not supported on SDXL yet: ${nl} LoRA${nl > 1 ? 's' : ''} will be skipped.`);
    }
    const bad = G.slots.filter(s => s.on && s.file).map(s => loraBy(s.file)).filter(l => l && mismatch(l));
    if (bad.length) w.push(`${bad.map(l => l.name).join(', ')}: ${bad.length > 1 ? 'these LoRAs are' : 'this LoRA is'} for a different base model than the checkpoint.`);
    const box = $('genWarn');
    box.textContent = [block, ...w].filter(Boolean).join(' ');
    box.hidden = !box.textContent;
    const gen = $('generateBtn');
    gen.disabled = !!block;
    gen.title = block || 'Generate (Ctrl+Enter)';
  }
  K.on('ckpt', () => { if (!syncFamily()) { renderWarn(); renderSlotWarn(); renderAccSums(); } });

  // ---------------------------------------------------------------- render all
  function renderAll() {
    promptEl.value = G.prompt; negEl.value = G.negative;
    autoGrow(promptEl, 76); autoGrow(negEl, 58);
    renderSliders(); renderSamplers();
    $('seed').value = String(G.seed);
    $('hiresOn').checked = !!G.hires.on; $('faceOn').checked = !!G.face.on; $('nagOn').checked = !!G.nag.on;
    $('hiresUpscaler').value = G.hires.upscaler === 'model' ? 'model' : 'lanczos';
    renderCache(); renderUp(); renderResize();
    for (const [id, k] of [['accHires', 'hires'], ['accFace', 'face'], ['accLora', 'lora'], ['accKiln', 'kiln']]) $(id).open = !!G.accOpen[k];
    renderSlots(); renderAccSums(); renderQuick(); renderNeg(); renderWarn();
    for (const i2i of document.querySelectorAll('#i2iTabs button')) i2i.setAttribute('aria-selected', String(i2i.dataset.i === G.i2iMode));
    tokPos(); tokNeg();
  }
  function applyTab() {
    const img = tab() === 'img2img';
    $('i2iBox').hidden = !img;
    for (const e of document.querySelectorAll('[data-only="txt2img"]')) e.hidden = img;
    renderSamplers(); renderWarn();
    if (img) drawI2i();
    requestAnimationFrame(fitFrame);
  }
  K.on('tab', (t) => { if (t.tab === 'txt2img' || t.tab === 'img2img') applyTab(); });
  K.on('models', () => { if (!syncFamily()) { renderSlots(); renderAccSums(); renderQuick(); renderWarn(); } });
  K.on('settings', () => { renderPresets(); maybeDefaults(); });
  function maybeDefaults() {
    // a device without local settings starts from the server's UI defaults (Settings tab)
    const d = K.S.settings && K.S.settings.ui && K.S.settings.ui.defaults;
    if (d && FAM === 'anima' && !lsGet(LS) && !lsGet('kiln.settings.v1')) { G = merged(Object.assign(clone(DEF), clone(d))); save(); renderAll(); }
  }

  // ---------------------------------------------------------------- generate (txt2img / img2img)
  const BUSY = { groups: new Set(), graphJobs: new Set() };
  async function generate() {
    if ($('generateBtn').disabled) { toast($('generateBtn').title, 'err', 4000); return; }
    const gen = $('generateBtn');
    gen.classList.remove('flash'); void gen.offsetWidth; gen.classList.add('flash');
    if (tab() === 'img2img') return generateI2i();
    const c = K.ckptInfo();
    const body = {
      prompt: G.prompt, negative: G.negative, width: G.width, height: G.height, steps: G.steps, cfg: G.cfg,
      cfg_cutoff: G.cfg > 1 ? G.cfgCutoff : 1, step_cache: G.stepCache,
      nag: G.nag.on ? { enabled: true, scale: G.nag.scale, tau: G.nag.tau, alpha: G.nag.alpha } : { enabled: false },
      seed: G.seed >= 0 ? G.seed : -1, sampler: G.sampler, scheduler: G.scheduler, shift: G.shift, batch: G.batchCount * G.batchSize,
      loras: G.slots.filter(s => s.on && s.file && s.strength !== 0).map(s => ({ file: s.file, strength: s.strength })),
      model: c ? c.file : undefined,
    };
    const sd = isSDXL();
    // Anima-only options stay out of SDXL jobs (LoRAs are sent: the server reports them as skipped)
    if (sd) { body.nag = { enabled: false }; body.step_cache = 0; delete body.shift; }
    const f = feat();
    if (G.hires.on && hiresScale() > 1.001) body.hires = { scale: Math.round(hiresScale() * 1000) / 1000, denoise: G.hires.denoise, steps: G.hires.steps, upscaler: G.hires.upscaler === 'model' && (!f || f.upscale_model) ? 'model' : 'lanczos' };
    if (!sd && G.face.on && (!f || f.face)) body.face = { enabled: true, denoise: G.face.denoise, steps: G.face.steps, guide: G.face.guide, max_size: G.face.max_size, crop: G.face.crop, conf: G.face.conf, max_faces: G.face.max_faces };
    if (G.upscale) body.upscale = { factor: G.upscale };
    try {
      const r = await api('/api/generate', { method: 'POST', body });
      BUSY.groups.add(r.group);
      outBegin({ group: r.group, ids: r.ids, kind: 'txt2img' });
      // the job may already be running (its SSE event came before this response): start the stage now,
      // so a model swap shows its loading state instead of waiting for the first step
      const run = K.S.running && K.S.jobs.get(K.S.running.id);
      if (run && run.status === 'running' && mine(run)) stageBegin(run);
      if (r.width !== G.width || r.height !== G.height) toast(`Size snapped to ${r.width}×${r.height}`);
      if (Array.isArray(r.dropped) && r.dropped.length) toast('Skipped: ' + r.dropped.join('; '), 'err', 6000);
      if (Array.isArray(r.warnings) && r.warnings.length) toast(r.warnings.join(' · '), 'err', 6000);
      if (K.isMobile()) $('stageCol').scrollIntoView({ behavior: 'smooth', block: 'start' });
      renderBusy();
    } catch (e) {
      toast('Generate failed: ' + e.message, 'err', 6000);
    }
  }
  K.on('generate', (d) => { if (d.tab === 'txt2img' || d.tab === 'img2img') generate(); });
  $('generateBtn').addEventListener('click', generate);
  $('interruptBtn').addEventListener('click', async () => {
    for (const g of BUSY.groups) await K.cancel(g);
    for (const id of BUSY.graphJobs) await K.cancel(id);
  });
  $('skipBtn').addEventListener('click', () => {
    const r = K.S.running;
    if (r && (BUSY.groups.has(r.group) || BUSY.graphJobs.has(r.id))) K.cancel(r.id);
  });
  function renderBusy() {
    const live = [K.S.running, ...K.S.queued].filter(Boolean);
    for (const g of [...BUSY.groups]) if (!live.some(j => j.group === g)) BUSY.groups.delete(g);
    for (const id of [...BUSY.graphJobs]) if (!live.some(j => j.id === id)) BUSY.graphJobs.delete(id);
    const busy = BUSY.groups.size + BUSY.graphJobs.size > 0;
    $('genBusy').hidden = !busy;
    $('generateLabel').textContent = busy ? 'Generate (queue)' : 'Generate';
  }
  K.on('queue', renderBusy);

  // ---------------------------------------------------------------- img2img / inpaint input
  const I2I = { img: null, name: '', srcUrl: '', mask: null, erase: false };
  const cv = $('i2iCanvas'), mk = $('i2iMask');
  async function setI2iImage(src, name, opts = {}) {
    try {
      const blob = src instanceof Blob ? src : await (await fetch(src)).blob();
      const bmp = await createImageBitmap(blob);
      I2I.img = bmp; I2I.name = name || 'image';
      I2I.maskDirty = false;
      if (opts.mode) { G.i2iMode = opts.mode; save(); }
      $('i2iInfo').textContent = `${name || 'image'} · ${bmp.width}×${bmp.height}`;
      if (opts.fitSize) { G.width = clamp(Math.round(bmp.width / 16) * 16, 256, 2048); G.height = clamp(Math.round(bmp.height / 16) * 16, 256, 2048); save(); renderSliders(); }
      mk.getContext('2d').clearRect(0, 0, mk.width, mk.height);
      drawI2i(true);
      renderWarn();
      for (const b of document.querySelectorAll('#i2iTabs button')) b.setAttribute('aria-selected', String(b.dataset.i === G.i2iMode));
    } catch (e) { toast('Could not read that image: ' + e.message, 'err'); }
  }
  // the source resized to the target size with the chosen resize mode (what gets uploaded)
  function drawI2i(resetMask) {
    const has = !!I2I.img;
    $('i2iHint').hidden = has;
    cv.hidden = !has;
    const inpaint = G.i2iMode === 'inpaint';
    mk.hidden = !has || !inpaint;
    $('maskTools').hidden = !has || !inpaint;
    $('i2iClear').hidden = !has;
    if (!has) return;
    const W = G.width, H = G.height, img = I2I.img;
    const old = mk.width > 1 && !resetMask ? (() => { const c = document.createElement('canvas'); c.width = mk.width; c.height = mk.height; c.getContext('2d').drawImage(mk, 0, 0); return c; })() : null;
    cv.width = W; cv.height = H; mk.width = W; mk.height = H;
    const x = cv.getContext('2d');
    x.clearRect(0, 0, W, H);
    const s = G.resize === 'crop' ? Math.max(W / img.width, H / img.height) : Math.min(W / img.width, H / img.height);
    if (G.resize === 'just') x.drawImage(img, 0, 0, W, H);
    else {
      if (G.resize === 'fill') { x.filter = 'blur(24px)'; x.drawImage(img, 0, 0, W, H); x.filter = 'none'; }
      const dw = img.width * s, dh = img.height * s;
      x.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
    }
    if (old) mk.getContext('2d').drawImage(old, 0, 0, W, H);
  }
  // mask brush
  let painting = null;
  const maskPos = (e) => { const r = mk.getBoundingClientRect(); return [(e.clientX - r.left) * mk.width / r.width, (e.clientY - r.top) * mk.height / r.height]; };
  mk.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    mk.setPointerCapture(e.pointerId);
    painting = maskPos(e);
    stroke(painting, painting);
  });
  mk.addEventListener('pointermove', (e) => { if (!painting) return; const p = maskPos(e); stroke(painting, p); painting = p; });
  const endPaint = () => { painting = null; };
  mk.addEventListener('pointerup', endPaint); mk.addEventListener('pointercancel', endPaint);
  function stroke(a, b) {
    const x = mk.getContext('2d');
    const r = mk.width / Math.max(1, mk.getBoundingClientRect().width);
    x.globalCompositeOperation = I2I.erase ? 'destination-out' : 'source-over';
    x.strokeStyle = '#fff'; x.lineCap = 'round'; x.lineJoin = 'round';
    x.lineWidth = Number($('brushSize').value) * r;
    x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0] + 0.01, b[1]); x.stroke();
    I2I.maskDirty = true;
  }
  $('brushSize').addEventListener('input', () => { $('brushSizeV').textContent = $('brushSize').value; });
  $('maskErase').addEventListener('click', () => { I2I.erase = !I2I.erase; $('maskErase').setAttribute('aria-pressed', String(I2I.erase)); });
  $('maskClear').addEventListener('click', () => { mk.getContext('2d').clearRect(0, 0, mk.width, mk.height); I2I.maskDirty = false; });
  $('i2iClear').addEventListener('click', () => { I2I.img = null; $('i2iInfo').textContent = ''; drawI2i(true); renderWarn(); });
  $('i2iTabs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    G.i2iMode = b.dataset.i; save();
    for (const x of $('i2iTabs').querySelectorAll('button')) x.setAttribute('aria-selected', String(x === b));
    drawI2i();
  });
  const pickFile = (input, fn) => { input.addEventListener('change', () => { const f = input.files[0]; if (f) fn(f, f.name); input.value = ''; }); };
  $('i2iPick').addEventListener('click', () => $('i2iFile').click());
  pickFile($('i2iFile'), (f, n) => setI2iImage(f, n));
  function dropZone(zone, fn) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('over'));
    zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('over'); const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f && /^image\//.test(f.type)) fn(f, f.name); });
    zone.addEventListener('paste', (e) => { const f = [...(e.clipboardData && e.clipboardData.files || [])].find(x => /^image\//.test(x.type)); if (f) fn(f, 'pasted.png'); });
  }
  dropZone($('i2iDrop'), (f, n) => setI2iImage(f, n));
  // PNG encoder (RGBA, straight alpha) so masked pixels keep their colour (canvas.toBlob premultiplies)
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(bytes) { let c = 0xffffffff; for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
  function pngChunk(type, data) {
    const out = new Uint8Array(12 + data.length);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  }
  async function encodePNGRGBA(rgba, w, h) {
    const stride = w * 4 + 1, raw = new Uint8Array(stride * h);
    for (let y = 0; y < h; y++) raw.set(rgba.subarray(y * w * 4, (y + 1) * w * 4), y * stride + 1);
    const z = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
    const ihdr = new Uint8Array(13);
    const dv = new DataView(ihdr.buffer);
    dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 6;
    return new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), pngChunk('IHDR', ihdr), pngChunk('IDAT', z), pngChunk('IEND', new Uint8Array(0))], { type: 'image/png' });
  }
  function fnv(bytes) { let h = 0x811c9dc5; for (let i = 0; i < bytes.length; i += 7) { h ^= bytes[i]; h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); }
  async function i2iUpload() {
    const W = cv.width, H = cv.height;
    const px = cv.getContext('2d').getImageData(0, 0, W, H).data;
    let blob;
    if (G.i2iMode === 'inpaint') {
      // mask (optionally blurred) -> alpha = 1 - mask, the LoadImage MASK convention
      const blur = clamp(Number($('maskBlur').value) || 0, 0, 64);
      const tmp = document.createElement('canvas'); tmp.width = W; tmp.height = H;
      const tx = tmp.getContext('2d');
      if (blur) tx.filter = `blur(${blur}px)`;
      tx.drawImage(mk, 0, 0);
      const m = tx.getImageData(0, 0, W, H).data;
      let any = false;
      const rgba = new Uint8Array(W * H * 4);
      for (let i = 0; i < W * H; i++) {
        rgba[i * 4] = px[i * 4]; rgba[i * 4 + 1] = px[i * 4 + 1]; rgba[i * 4 + 2] = px[i * 4 + 2];
        const a = m[i * 4 + 3];
        if (a > 8) any = true;
        rgba[i * 4 + 3] = 255 - a;
      }
      if (!any) throw new Error('paint the area to redraw first (Inpaint tab)');
      blob = await encodePNGRGBA(rgba, W, H);
    } else {
      blob = await new Promise(r => cv.toBlob(r, 'image/png'));
    }
    const buf = new Uint8Array(await blob.arrayBuffer());
    const name = `kiln_${G.i2iMode}_${W}x${H}_${fnv(buf)}.png`;
    const r = await fetch('/api/upload/image?name=' + encodeURIComponent(name), { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: blob });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'upload failed');
    return j.name;
  }
  let graphBase = null;
  async function loaders() {
    if (!graphBase) {
      const ex = await api('/api/graph/examples');
      const g = (ex.find(e => e.name === 'anima-turbo-txt2img') || ex[0]).graph;
      graphBase = { clip: g['3'].inputs, vae: g['8'].inputs };
    }
    return graphBase;
  }
  async function generateI2i() {
    if (!I2I.img) { toast('Add an image first (drop, paste or upload)', 'err'); return; }
    const c = K.ckptInfo();
    let image;
    try { image = await i2iUpload(); } catch (e) { toast(e.message, 'err', 5000); return; }
    const L = await loaders();
    const conv = extractLoraTags(G.prompt);
    const unet = c ? c.file.replace(/^(diffusion_models|unet|checkpoints)\//, '') : 'anima-base-v1.0.safetensors';
    const all = G.slots.filter(s => s.on && s.file && s.strength !== 0).map(s => ({ file: s.file, strength: s.strength }));
    for (const t of conv.tags) {
      const l = loras().find(x => x.name.toLowerCase() === t.name.toLowerCase() || stem(x.file).toLowerCase() === t.name.toLowerCase());
      if (!l) { toast(`<lora:${t.name}>: no LoRA with that name`, 'err', 5000); continue; }
      const i = all.findIndex(x => x.file === l.file);
      if (i >= 0) all[i].strength = t.weight; else all.push({ file: l.file, strength: t.weight });
    }
    const g = {
      1: { class_type: 'UNETLoader', inputs: { unet_name: unet, weight_dtype: 'default' } },
      3: { class_type: 'CLIPLoader', inputs: Object.assign({}, L.clip) },
      4: { class_type: 'CLIPTextEncode', inputs: { text: deemphasize(conv.text), clip: ['3', 0] } },
      5: { class_type: 'CLIPTextEncode', inputs: { text: deemphasize(extractLoraTags(G.negative).text), clip: ['3', 0] } },
      8: { class_type: 'VAELoader', inputs: Object.assign({}, L.vae) },
      11: { class_type: 'LoadImage', inputs: { image } },
      6: { class_type: 'VAEEncode', inputs: { pixels: ['11', 0], vae: ['8', 0] } },
      9: { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
      10: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: G.i2iMode === 'inpaint' ? 'Kiln_inpaint' : 'Kiln_img2img' } },
    };
    let model = ['1', 0], nid = 20;
    for (const l of all) { g[nid] = { class_type: 'LoraLoaderModelOnly', inputs: { model, lora_name: l.file.replace(/^loras\//, ''), strength_model: l.strength } }; model = [String(nid), 0]; nid++; }
    g[nid] = { class_type: 'ModelSamplingAuraFlow', inputs: { model, shift: G.shift } }; model = [String(nid), 0]; nid++;
    if (G.stepCache > 0) { g[nid] = { class_type: 'ApplyFBCacheOnModel', inputs: { model, object_to_patch: 'diffusion_model', residual_diff_threshold: G.stepCache, start: 0.15, end: 0.9, max_consecutive_cache_hits: 2 } }; model = [String(nid), 0]; nid++; }
    let latent = ['6', 0];
    if (G.i2iMode === 'inpaint') { g[12] = { class_type: 'SetLatentNoiseMask', inputs: { samples: ['6', 0], mask: ['11', 1] } }; latent = ['12', 0]; }
    const nag = G.nag.on && !(G.cfg > 1) && G.negative.trim();
    const n = G.batchCount * G.batchSize;
    const base = G.seed >= 0 ? G.seed : Math.floor(Math.random() * 2 ** 48);
    const ids = [];
    const infos = {};
    for (let k = 0; k < n; k++) {
      const seed = base + k;
      g[7] = nag
        ? { class_type: 'KSamplerWithNAG', inputs: { model, seed, steps: G.steps, cfg: G.cfg, nag_scale: G.nag.scale, nag_tau: G.nag.tau, nag_alpha: G.nag.alpha, nag_sigma_end: 0, sampler_name: G.sampler, scheduler: G.scheduler, positive: ['4', 0], negative: ['5', 0], nag_negative: ['5', 0], latent_image: latent, denoise: G.denoise } }
        : { class_type: 'KSampler', inputs: { model, seed, steps: G.steps, cfg: G.cfg, sampler_name: G.sampler, scheduler: G.scheduler, positive: ['4', 0], negative: ['5', 0], latent_image: latent, denoise: G.denoise } };
      try {
        const r = await api('/api/graph', { method: 'POST', body: { graph: g, ui: { kiln_simple: G.i2iMode } } });
        ids.push(r.id);
        BUSY.graphJobs.add(r.id);
        infos[r.id] = { prompt: G.prompt, negative: G.negative, steps: G.steps, sampler: G.sampler, scheduler: G.scheduler, cfg: G.cfg, seed, width: G.width, height: G.height, model: c && c.file, denoise: G.denoise, loras: all, shift: G.shift, nag: nag ? G.nag : null, step_cache: G.stepCache, kind: 'graph', mode: G.i2iMode };
      } catch (e) {
        toast('img2img failed: ' + e.message, 'err', 6000);
        break;
      }
    }
    if (ids.length) {
      outBegin({ ids, kind: G.i2iMode, infos });
      if (G.seed < 0) { /* random seed stays -1, like A1111 */ }
      renderBusy();
      if (K.isMobile()) $('stageCol').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // ---------------------------------------------------------------- output panel
  const OUT = { group: null, ids: new Set(), kind: 'txt2img', items: [], sel: 0, infos: {} };
  const frame = $('frame'), canvas = $('previewCanvas'), finalImg = $('finalImg');
  const ctx = canvas.getContext('2d');
  let frameAR = { w: 512, h: 768 };
  const S = { stageJob: null, stageFirstPreview: false, stageTicker: 0, gallery: [], galTotal: 0, galLoading: false, galDone: false, lb: -1 };

  function outBegin(o) {
    OUT.group = o.group || null;
    OUT.ids = new Set(o.ids || []);
    OUT.kind = o.kind;
    // the previous results stay (and stay usable) until this run's first image arrives
    OUT.fresh = true;
    OUT.infos = Object.assign({}, OUT.infos, o.infos || {});
    OUT.t0 = Date.now();
    if (o.kind !== 'txt2img') { frameAR = { w: G.width, h: G.height }; frame.dataset.state = 'running'; fitFrame(); setBadge('Queued…'); $('progressBar').style.width = '0'; }
  }
  const mine = (j) => j && ((OUT.group && j.group === OUT.group) || OUT.ids.has(j.id));
  function fitFrame() {
    const stage = $('stage');
    let w = frameAR.w, h = frameAR.h;
    if (frame.dataset.state === 'empty') { w = G.width; h = G.height; }
    const aw = stage.clientWidth - 16, ah = stage.clientHeight - 16;
    if (aw <= 0 || ah <= 0) return;
    const s = Math.min(aw / w, ah / h);
    frame.style.width = `${Math.floor(w * s)}px`;
    frame.style.height = `${Math.floor(h * s)}px`;
  }
  function setBadge(text, err) {
    const b = $('stageBadge');
    if (!text) { b.hidden = true; return; }
    b.hidden = false; b.textContent = text; b.classList.toggle('err', !!err);
  }
  function putRGB(cvs, c2d, pv) {
    const bin = atob(pv.rgb);
    const n = pv.w * pv.h;
    if (bin.length < n * 3) return false;
    if (cvs.width !== pv.w || cvs.height !== pv.h) { cvs.width = pv.w; cvs.height = pv.h; }
    const img = c2d.createImageData(pv.w, pv.h);
    const d = img.data;
    for (let i = 0, p = 0; i < n; i++, p += 3) { d[i * 4] = bin.charCodeAt(p); d[i * 4 + 1] = bin.charCodeAt(p + 1); d[i * 4 + 2] = bin.charCodeAt(p + 2); d[i * 4 + 3] = 255; }
    c2d.putImageData(img, 0, 0);
    return true;
  }
  function drawPreview(pv, frac) {
    try {
      if (!putRGB(canvas, ctx, pv)) return;
      const blur = Math.max(0, 0.45 - frac) * 5;
      canvas.style.filter = blur > 0.05 ? `blur(${blur.toFixed(2)}px)` : 'none';
      if (!S.stageFirstPreview) { S.stageFirstPreview = true; canvas.classList.add('show'); finalImg.classList.remove('show'); finalImg.style.opacity = ''; }
    } catch (_) { /* malformed preview */ }
  }
  const faceInset = $('faceInset'), faceCanvas = $('faceCanvas'), faceCtx = faceCanvas.getContext('2d');
  function drawFacePreview(pv, face, step, of) {
    try {
      if (!putRGB(faceCanvas, faceCtx, pv)) return;
      faceInset.style.setProperty('--ar', `${pv.w} / ${pv.h}`);
      $('faceInsetLabel').textContent = `Face ${face || 1} · ${step}/${of}`;
      faceInset.hidden = false;
    } catch (_) { }
  }
  // ---- per-stage timeline (base -> face per detected face -> hires), mirrored from step events
  function pushTimeline(j, m) {
    j.timeline = j.timeline || [];
    const stage = m.stage || 'base';
    let g = j.timeline[j.timeline.length - 1];
    if (!g || g.stage !== stage || (m.step <= 1 && g.ms.length)) {
      g = { stage, of: m.of || 0, ms: [] };
      if (stage === 'face') g.face = m.face || j.timeline.filter(x => x.stage === 'face').length + 1;
      j.timeline.push(g);
    }
    if (m.of) g.of = m.of;
    if (typeof m.ms === 'number' && g.ms.length < m.step) g.ms.push(m.ms);
    if (stage === 'base') j.step_ms = g.ms;
  }
  function planTimeline(j) {
    const p = j.params || {}, tl = j.timeline || [];
    const out = [];
    let base = tl.find(g => g.stage === 'base');
    if (!base) base = { stage: 'base', of: p.steps || j.of || 0, ms: (j.step_ms || []).slice() };
    out.push(base);
    // faces are redrawn at the base size, before the hires pass
    const hires = tl.find(g => g.stage === 'hires');
    const faces = tl.filter(g => g.stage === 'face');
    if (faces.length) out.push(...faces);
    else if (p.face && j.status === 'running' && !hires) out.push({ stage: 'face', of: p.face.steps || 0, ms: [], face: 1, maybe: true });
    if (p.hires) out.push(hires || { stage: 'hires', of: p.hires.steps || 0, ms: [] });
    for (const g of tl) if (!['base', 'hires', 'face'].includes(g.stage)) out.push(g);
    return out;
  }
  function paintTimeline(j) {
    const v = $('stepsViz');
    const groups = planTimeline(j);
    let max = 1;
    for (const g of groups) for (const ms of g.ms) if (ms > max) max = ms;
    const running = j.status === 'running';
    let active = false, total = 0, done = 0;
    const frag = document.createDocumentFragment();
    groups.forEach((g, gi) => {
      const n = clamp(Math.max(g.of || 0, g.ms.length), 0, 120);
      total += n; done += Math.min(n, g.ms.length);
      for (let i = 0; i < n; i++) {
        const ms = g.ms[i];
        const seg = el('div', `seg st-${g.stage}${i === 0 && gi > 0 ? ' grp' : ''}${g.maybe ? ' maybe' : ''}`);
        if (ms != null) { seg.classList.add('done'); seg.style.height = `${30 + 70 * (ms / max)}%`; }
        else if (running && !active) { seg.classList.add('active'); active = true; }
        seg.title = K.stageLabel(g.stage, g.face, i + 1, n) + (ms != null ? `: ${fmtMs(ms)}` : '');
        frag.appendChild(seg);
      }
    });
    v.textContent = '';
    v.appendChild(frag);
    renderStageBars(j, groups);
    return total ? done / total : 0;
  }
  // ---- model family swap (first SDXL job, or back to Anima): 20-35 s in the engine
  const loadDone = (j) => j.load_ms != null || j.encode_ms != null || (j.timeline || []).some(g => g.ms.length) || j.status !== 'running';
  function markLoaded(j) { if (j && j.swap && j.load_ms == null && j.loadT0) j.load_ms = Date.now() - j.loadT0; }
  // ---- ETA: remaining steps x measured s/it (before the first step: the last run's speed, scaled by size)
  function fmtClock(ms) {
    const sec = Math.max(0, Math.round(ms / 1000));
    return sec < 60 ? `${sec} s` : `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
  }
  const avgOf = (a) => (a && a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  function etaMs(j) {
    const p = j.params || {};
    const groups = planTimeline(j);
    const base = groups.find(g => g.stage === 'base');
    let per = avgOf(base && base.ms), approx = false;
    if (per == null) {
      const m = (lsGet(LS_STEPMS) || {})[p.family === 'sdxl' ? 'sdxl' : 'anima'];
      if (!m || !m.ms || !m.px || !p.width) return null;
      per = m.ms * (p.width * p.height) / m.px;
      approx = true;
    }
    let rem = 0;
    for (const g of groups) {
      const n = Math.max(g.of || 0, g.ms.length), left = n - g.ms.length;
      if (left <= 0) continue;
      let gp = avgOf(g.ms);
      if (gp == null) gp = g.stage === 'hires' && p.hires ? per * p.hires.scale * p.hires.scale : g.stage === 'face' ? per * 0.5 : per;
      rem += left * gp;
    }
    return { ms: rem, approx };
  }
  function updateEta(j) {
    const e = j && j.status === 'running' && !(j.swap && !loadDone(j)) ? etaMs(j) : null;
    S.etaEnd = e ? Date.now() + e.ms : 0;
    S.etaApprox = !!(e && e.approx);
    S.etaSlow = !!(j && j.params && j.params.family === 'sdxl');
    renderEta(j);
  }
  function renderEta(j) {
    const box = $('etaLine');
    if (!box) return;
    const live = !!(j && j.status === 'running' && S.stageTicker);
    const loading = live && j.swap && !loadDone(j);
    // SDXL takes minutes: always say something (the first step gives the speed)
    const text = !live ? '' : loading ? 'loading model…' : S.etaEnd ? `${S.etaApprox ? '~' : ''}${fmtClock(S.etaEnd - Date.now())} left` : S.etaSlow ? 'ETA after step 1' : '';
    box.hidden = !text;
    box.classList.toggle('slow', !!S.etaSlow);
    box.textContent = text;
  }
  function rememberSpeed(j) {
    const p = j.params || {}, sm = j.step_ms || [];
    if (sm.length < 2 || !p.width || j.kind === 'graph') return;
    const all = lsGet(LS_STEPMS) || {};
    all[p.family === 'sdxl' ? 'sdxl' : 'anima'] = { ms: Math.round(avgOf(sm)), px: p.width * p.height };
    lsSet(LS_STEPMS, all);
  }
  // one bar per stage: label, steps, time; upscaling (no steps) runs after the decode
  function renderStageBars(j, groups) {
    const box = $('stageBars');
    box.textContent = '';
    const p = j.params || {}, t = j.timings || {};
    const running = j.status === 'running', finished = j.status === 'done';
    const sum = (a) => a.reduce((x, y) => x + y, 0);
    const bars = [];
    if (j.swap) {
      const ld = loadDone(j);
      const what = p.family === 'sdxl' ? 'SDXL' : 'Anima';
      bars.push({ stage: 'load', label: `Load ${what}`, meta: j.load_ms != null ? fmtSec(j.load_ms) + ' s' : ld ? 'done' : j.loadProg != null ? `swap ${Math.round(j.loadProg * 100)}%` : 'swapping model…', frac: ld ? 1 : (j.loadProg || 0), state: ld ? 'done' : 'run', indet: !ld && !j.loadProg });
    }
    const eta = running && S.etaEnd ? ` · ${S.etaApprox ? '~' : ''}${fmtClock(S.etaEnd - Date.now())} left` : '';
    for (const g of groups) {
      const n = Math.max(g.of || 0, g.ms.length);
      const label = g.stage === 'base' ? (j.kind === 'graph' ? 'Sampling' : 'Base') : g.stage === 'hires' ? `Hires ${p.hires ? p.hires.scale + '×' : ''}` : g.stage === 'face' ? `Face ${g.face || 1}` : g.stage;
      const time = g.ms.length ? fmtSec(sum(g.ms)) + ' s' : '';
      const state = g.ms.length >= n && n ? 'done' : g.ms.length || (running && !bars.some(b => b.state !== 'done')) ? 'run' : 'wait';
      bars.push({ stage: g.stage, label, meta: g.maybe ? `${n} st each` : `${g.ms.length}/${n}${time ? ' · ' + time : ''}${state === 'run' && !g.maybe ? eta : ''}`, frac: n ? g.ms.length / n : 0, state: g.maybe ? 'wait' : state });
    }
    if (p.face && finished && !groups.some(g => g.stage === 'face')) bars.push({ stage: 'face', label: 'Face', meta: t.faces === 0 ? 'no faces found' : 'skipped', frac: 1, state: 'done' });
    if (p.upscale) {
      const ms = t.upscale_ms;
      const st = ms != null || finished ? 'done' : j.upscaling ? 'run' : 'wait';
      bars.push({ stage: 'upscale', label: `Upscale ${p.upscale.factor}×`, meta: ms != null ? fmtSec(ms) + ' s' : st === 'run' ? 'upscaling…' : '4x model', frac: st === 'done' ? 1 : 0, state: st, indet: st === 'run' });
    }
    for (const b of bars) {
      const d = el('div', `a-stg st-${b.stage} ${b.state}`);
      const head = el('div', 'a-stg-head');
      head.append(el('b', '', b.label), el('span', '', b.meta));
      const bar = el('div', 'a-stg-bar' + (b.indet ? ' indet' : ''));
      const fill = el('div'); fill.style.width = `${Math.round(b.frac * 100)}%`;
      bar.appendChild(fill);
      d.append(head, bar);
      box.appendChild(d);
    }
  }
  function setBreakdown(j) {
    const dl = $('breakdown');
    const t = j.timings || {}, p = j.params || {}, tl = j.timeline || [];
    const running = j.status === 'running', finished = j.status === 'done';
    const sum = (a) => a.reduce((x, y) => x + y, 0);
    const sm = j.step_ms || [];
    const items = [];
    if (j.swap) items.push(['Model load', j.load_ms != null ? fmtMs(j.load_ms) : (running ? '…' : '–')]);
    items.push(
      ['Encode', j.encode_ms != null ? fmtMs(j.encode_ms) : (running && j.kind !== 'graph' ? '…' : '–')],
      ['Per step', sm.length ? `${(sum(sm) / sm.length / 1000).toFixed(2)} s/it` : '–'],
    );
    if (p.face) {
      const fg = tl.filter(x => x.stage === 'face');
      const n = t.faces != null ? t.faces : fg.length;
      const ms = t.face_ms != null ? t.face_ms : (fg.length ? sum(fg.flatMap(g => g.ms)) : null);
      items.push(['Faces', finished && !n ? 'none found' : ms != null ? `${n} · ${fmtMs(ms)}` : (running ? '…' : '–')]);
    }
    if (p.hires) { const g = tl.find(x => x.stage === 'hires'); const ms = t.hires_ms != null ? t.hires_ms : (g && g.ms.length ? sum(g.ms) : null); items.push([`Hires ${p.hires.scale}×`, ms != null ? fmtMs(ms) : (running ? '…' : '–')]); }
    if (p.upscale) items.push([`Upscale ${p.upscale.factor}×`, t.upscale_ms != null ? fmtMs(t.upscale_ms) : (running ? '…' : '–')]);
    if (t.cache_skips != null) items.push(['Cache', `${t.cache_skips} skipped`]);
    items.push(['Decode', j.decode_ms != null ? fmtMs(j.decode_ms) : '–']);
    dl.textContent = '';
    for (const [k, v] of items) { const d = el('div'); d.append(el('dt', '', k), el('dd', '', v)); dl.appendChild(d); }
  }
  function startTicker(j) {
    stopTicker();
    $('statBig').dataset.live = '1';
    let spent = j.encode_ms || 0;
    for (const g of j.timeline || []) spent += g.ms.reduce((a, b) => a + b, 0);
    const base = Date.now() - spent;
    S.stageTicker = setInterval(() => { $('bigNum').textContent = ((Date.now() - base) / 1000).toFixed(1); renderEta(K.S.jobs.get(S.stageJob)); }, 100);
  }
  function stopTicker() { clearInterval(S.stageTicker); S.stageTicker = 0; S.etaEnd = 0; $('statBig').dataset.live = '0'; renderEta(null); }
  function stageBegin(j) {
    if (S.stageJob === j.id) return;
    S.stageJob = j.id;
    S.stageFirstPreview = false;
    const p = j.params || {};
    frameAR = { w: p.width || G.width, h: p.height || G.height };
    frame.dataset.state = 'running';
    fitFrame();
    faceInset.hidden = true;
    if (finalImg.classList.contains('show')) finalImg.style.opacity = '0.28';
    $('stageOpen').hidden = true;
    $('progressBar').style.width = '0';
    const batch = j.count > 1 ? ` · ${j.index + 1}/${j.count}` : '';
    if (j.swap) {
      j.loadT0 = j.loadT0 || Date.now();
      setBadge(`Loading ${p.family === 'sdxl' ? 'the SDXL checkpoint' : 'the Anima model'}… (model swap, about 20–35 s)${batch}`);
    } else setBadge(j.kind === 'graph' ? 'Running…' : `Encoding prompt…${batch}`);
    startTicker(j);
    updateEta(j);
    $('progressBar').style.width = `${paintTimeline(j) * 100}%`;
    setBreakdown(j);
  }
  function showImage(it) {
    frameAR = { w: it.w, h: it.h };
    frame.dataset.state = 'done';
    fitFrame();
    const img = new Image();
    img.onload = () => { finalImg.src = img.src; finalImg.style.opacity = ''; finalImg.classList.add('show'); setTimeout(() => { if (frame.dataset.state === 'done') canvas.classList.remove('show'); }, 500); };
    img.src = it.view || it.url;
    $('stageOpen').hidden = false;
    $('stageOpen').onclick = () => { const i = S.gallery.findIndex(g => g.file === it.file); if (i >= 0) openLightbox(i); else window.open(it.url, '_blank', 'noopener'); };
    $('infoText').textContent = it.info || '';
    const p = it.params || {};
    $('outTiming').textContent = `Time taken: ${fmtSec(p.total_ms)} s${p.step_ms && p.step_ms.length ? ` · ${(p.step_ms.reduce((a, b) => a + b, 0) / p.step_ms.length / 1000).toFixed(2)} s/it` : ''} · ${it.w}×${it.h}`;
  }
  function renderThumbs() {
    const box = $('outThumbs');
    box.textContent = '';
    if (OUT.items.length > 1) {
      OUT.items.forEach((it, i) => {
        const b = el('button', i === OUT.sel ? 'sel' : '');
        b.type = 'button';
        b.title = `seed ${it.params && it.params.seed}`;
        const im = el('img'); im.src = it.thumb || it.url; im.alt = ''; im.loading = 'lazy';
        b.appendChild(im);
        b.addEventListener('click', () => { OUT.sel = i; renderThumbs(); showImage(it); });
        box.appendChild(b);
      });
    }
    const has = OUT.items.length > 0;
    for (const id of ['outSave', 'outZip', 'outI2i', 'outInpaint', 'outExtras']) $(id).disabled = !has;
    $('outZip').disabled = OUT.items.length < 1;
  }
  function addOut(it) {
    if (OUT.fresh) { OUT.fresh = false; OUT.items = []; OUT.sel = 0; }
    OUT.items.push(it);
    if (OUT.items.length === 1 || OUT.sel === OUT.items.length - 2) OUT.sel = OUT.items.length - 1;
    renderThumbs();
    showImage(OUT.items[OUT.sel]);
  }
  function onStep(m) {
    const j = K.S.jobs.get(m.id);
    if (!mine(j || { id: m.id })) return;
    if (j) { markLoaded(j); pushTimeline(j, m); }
    if (j && S.stageJob !== j.id) stageBegin(j);
    if (S.stageJob !== m.id) return;
    const stage = m.stage || 'base';
    const frac = m.of ? m.step / m.of : 0;
    if (j) { j.step = m.step; j.of = m.of; setBreakdown(j); }
    if (m.preview) {
      if (stage === 'face') drawFacePreview(m.preview, m.face, m.step, m.of);
      else { faceInset.hidden = true; drawPreview(m.preview, stage === 'base' ? frac : 1); }
    }
    if (j) updateEta(j);
    $('progressBar').style.width = `${(j ? paintTimeline(j) : frac) * 100}%`;
    const batch = j && j.count > 1 ? ` · img ${j.index + 1}/${j.count}` : '';
    setBadge(`${K.stageLabel(stage, m.face, m.step, m.of)}${m.ms != null ? ' · ' + fmtMs(m.ms) : ''}${S.etaEnd ? ' · ' + (S.etaApprox ? '~' : '') + fmtClock(S.etaEnd - Date.now()) + ' left' : ''}${batch}`);
    document.title = `${K.stageLabel(stage, m.face, m.step, m.of)} · Kiln`;
  }
  function onJob(j) {
    if (!mine(j)) {
      if (j.kind === 'graph' && j.status === 'error' && K.S.tab !== 'nodes' && OUT.ids.has(j.id)) toast('Failed: ' + j.error, 'err', 6000);
      return;
    }
    if (j.status === 'running') stageBegin(j);
    else if (j.status === 'done') {
      stopTicker();
      rememberSpeed(j);
      if (S.stageJob === j.id || !S.stageJob) {
        S.stageJob = j.id;
        $('bigNum').textContent = fmtSec(j.total_ms);
        $('progressBar').style.width = '100%';
        paintTimeline(j);
        setBreakdown(j);
      }
      K.recordTime(j.total_ms);
      document.title = 'Kiln';
      if (j.kind !== 'graph' && j.image) {
        const params = Object.assign({}, j.params, { out_w: j.image.w, out_h: j.image.h, total_ms: j.total_ms, step_ms: j.step_ms, timings: j.timings });
        const it = { file: j.image.file, url: j.image.url, thumb: j.image.thumb, view: j.image.view, w: j.image.w, h: j.image.h, params, info: infotext(j.params, { Version: 'Kiln' }) };
        addOut(it);
        addToGallery(Object.assign({ mtime: Date.now() }, it));
      }
      if (j.kind === 'graph') { $('outTiming').textContent = `Time taken: ${fmtSec(j.total_ms)} s`; }
      setBadge('');
    } else if (j.status === 'error') {
      stopTicker();
      if (S.stageJob === j.id) paintTimeline(j);
      toast('Generation failed: ' + (j.error || 'unknown error'), 'err', 7000);
      setBadge('Error: ' + (j.error || 'unknown'), true);
      frame.dataset.state = finalImg.src ? 'done' : 'empty';
      document.title = 'Kiln';
    } else if (j.status === 'cancelled') {
      stopTicker();
      if (S.stageJob === j.id) paintTimeline(j);
      if (S.stageJob === j.id) { setBadge('Interrupted'); frame.dataset.state = finalImg.src ? 'done' : 'empty'; canvas.classList.remove('show'); }
      document.title = 'Kiln';
    }
  }
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    switch (m.type) {
      case 'job': onJob(K.S.jobs.get(m.job.id) || m.job); break;
      case 'step': onStep(m); break;
      case 'loading': {
        const j = K.S.jobs.get(m.id);
        if (j) {
          j.loadT0 = j.loadT0 || Date.now();
          if (m.progress != null) j.loadProg = m.progress;
          if (m.progress >= 1) markLoaded(j);
          if (mine(j) && S.stageJob !== j.id && j.status === 'running') stageBegin(j);
        }
        if (S.stageJob === m.id) {
          setBadge(`Loading ${m.what || 'model'}${m.progress != null ? ' ' + Math.round(m.progress * 100) + '%' : '…'}${j && j.swap ? ' (model swap)' : ''}`);
          if (j) { paintTimeline(j); setBreakdown(j); }
        }
        break;
      }
      case 'encoded': {
        const j = K.S.jobs.get(m.id);
        if (j) { markLoaded(j); j.encode_ms = m.ms; if (S.stageJob === m.id) { setBreakdown(j); paintTimeline(j); updateEta(j); } }
        if (S.stageJob === m.id) setBadge(`Prompt encoded · ${fmtMs(m.ms)}`);
        break;
      }
      case 'decoded': {
        const j = K.S.jobs.get(m.id);
        if (j) { j.decode_ms = m.ms; if (j.params && j.params.upscale) j.upscaling = true; }
        if (S.stageJob === m.id) {
          faceInset.hidden = true;
          if (j) { setBreakdown(j); paintTimeline(j); }
          setBadge(j && j.upscaling ? `Decoded · ${fmtMs(m.ms)} · upscaling ${j.params.upscale.factor}×…` : `Decoded · ${fmtMs(m.ms)}`);
        }
        break;
      }
      case 'gimage':
        if (m.kind === 'save' && m.image && m.image.file) {
          const gj = K.S.jobs.get(m.id);
          const info = OUT.infos[m.id];
          const it = { file: m.image.file, url: m.image.url, thumb: m.image.thumb, view: m.image.view, w: m.image.w, h: m.image.h, params: Object.assign({ kind: 'graph', graph_job: m.id }, gj ? gj.params : {}, info || {}) };
          if (OUT.ids.has(m.id)) { it.info = info ? infotext(info, { Version: 'Kiln' }) : ''; addOut(it); }
          addToGallery(Object.assign({ mtime: Date.now() }, it));
        }
        break;
      case 'deleted': removeFromGallery(m.file); break;
      default: break;
    }
  });
  new ResizeObserver(() => fitFrame()).observe($('stage'));
  // output buttons
  const selItem = () => OUT.items[OUT.sel];
  $('outSave').addEventListener('click', () => { const it = selItem(); if (!it) return; const a = el('a'); a.href = it.url + '?dl=1'; a.download = it.file.split('/').pop(); document.body.appendChild(a); a.click(); a.remove(); });
  $('outZip').addEventListener('click', () => { if (!OUT.items.length) return; location.href = '/api/zip?files=' + OUT.items.map(i => encodeURIComponent(i.file)).join(','); });
  $('outFolder').addEventListener('click', async () => {
    const it = selItem();
    try { await api('/api/open-folder', { method: 'POST', body: { file: it ? it.file : '' } }); }
    catch (e) { toast(e.status === 403 ? 'The folder opens on the PC running Kiln only' : e.message, 'err', 4000); }
  });
  function sendTo(target) {
    const it = selItem();
    if (!it) return;
    if (target === 'extras') { window.KilnTabs && window.KilnTabs.setExtrasImage(it.url, it.file.split('/').pop()); K.setTab('extras'); return; }
    if (it.params) applyParams(it.params, { keepSize: false });
    G.i2iMode = target === 'inpaint' ? 'inpaint' : 'img2img';
    save();
    K.setTab('img2img').then(() => setI2iImage(it.url, it.file.split('/').pop(), { mode: G.i2iMode, fitSize: true }));
  }
  $('outI2i').addEventListener('click', () => sendTo('img2img'));
  $('outInpaint').addEventListener('click', () => sendTo('inpaint'));
  $('outExtras').addEventListener('click', () => sendTo('extras'));

  // ---------------------------------------------------------------- params from an image (Kiln metadata or A1111 text)
  function applyParams(p, opts = {}) {
    if (!p || typeof p !== 'object') return;
    if (p.model && K.S.models) {
      const c = K.S.models.sd_models.find(m => m.file === p.model || m.name === stem(p.model));
      if (c && c.file !== K.S.ckpt) { K.S.ckpt = c.file; lsSet('kiln.ckpt.v1', c.file); $('ckptSel').value = c.file; K.emit('ckpt', c); }
      else if (!c) toast('Checkpoint not found: ' + stem(p.model), 'err', 4000);
    }
    const gr = isSDXL() ? 32 : 16;
    if (typeof p.prompt === 'string') G.prompt = p.prompt;
    if (typeof p.negative === 'string') G.negative = p.negative;
    if (p.width && p.height && opts.keepSize !== true) { G.width = clamp(Math.round(p.width / gr) * gr, 256, 2048); G.height = clamp(Math.round(p.height / gr) * gr, 256, 2048); }
    if (p.steps) G.steps = clamp(Math.round(p.steps), 1, 150);
    if (p.cfg != null && isFinite(p.cfg)) G.cfg = Number(p.cfg);
    if (p.cfg_cutoff != null) G.cfgCutoff = clamp(Number(p.cfg_cutoff) || 1, 0.1, 1);
    if (p.sampler) G.sampler = p.sampler;
    if (p.scheduler) G.scheduler = p.scheduler;
    if (p.shift != null) G.shift = Number(p.shift);
    if (p.seed != null) G.seed = Number(p.seed);
    if (p.denoise != null) G.denoise = Number(p.denoise);
    if ('step_cache' in p) G.stepCache = Number(p.step_cache) || 0;
    if ('nag' in p) G.nag = p.nag ? Object.assign({}, G.nag, p.nag, { on: true }) : Object.assign({}, G.nag, { on: false });
    if (!p.graph_job) {
      G.hires = p.hires ? Object.assign({}, G.hires, p.hires, { on: true, tw: 0, th: 0 }) : Object.assign({}, G.hires, { on: false });
      G.face = p.face ? Object.assign({}, G.face, p.face, { on: true }) : Object.assign({}, G.face, { on: false });
      G.upscale = p.upscale && [2, 4].includes(Number(p.upscale.factor)) ? Number(p.upscale.factor) : 0;
    }
    if (Array.isArray(p.loras)) {
      // slot LoRAs only (prompt tags are still in the prompt)
      const tags = new Set(extractLoraTags(G.prompt).tags.map(t => t.name.toLowerCase()));
      const list = p.loras.filter(l => !tags.has(stem(l.file).toLowerCase()));
      const missing = [];
      G.slots = G.slots.map((s, i) => (i < SLOTS ? { on: false, file: '', strength: 1 } : null)).filter(Boolean);
      list.forEach((l, i) => {
        const m = loraBy(l.file) || loras().find(x => stem(x.file).toLowerCase() === stem(l.file).toLowerCase());
        if (!m) { missing.push(stem(l.file)); return; }
        if (!G.slots[i]) G.slots.push({ on: false, file: '', strength: 1 });
        G.slots[i] = { on: true, file: m.file, strength: Number(l.strength) };
      });
      if (missing.length) toast('Missing LoRA: ' + missing.join(', '), 'err', 5000);
    }
    save();
    renderAll();
  }
  // A1111 "parameters" -> Kiln params
  function a1111ToParams(text) {
    const x = parseInfotext(text), P = x.params;
    const num = (k) => (P[k] !== undefined && P[k] !== '' && isFinite(Number(P[k])) ? Number(P[k]) : undefined);
    const p = { prompt: x.prompt, negative: x.negative, steps: num('Steps'), cfg: num('CFG scale'), seed: num('Seed'), shift: num('Shift'), denoise: num('Denoising strength') };
    const sm = String(P.Sampler || '').toLowerCase();
    const SK = { 'euler': 'euler', 'euler a': 'euler_ancestral', 'dpm++ 2m': 'dpmpp_2m', 'dpm++ 2m karras': 'dpmpp_2m', 'res multistep': 'res_multistep' };
    if (SK[sm]) p.sampler = SK[sm];
    if (/karras/.test(sm)) p.scheduler = 'karras';
    if (P['Schedule type']) { const s = String(P['Schedule type']).toLowerCase().replace(/ /g, '_'); if (SCHED_LABEL[s]) p.scheduler = s; }
    const size = /^(\d+)x(\d+)$/.exec(P.Size || '');
    if (size) { p.width = Number(size[1]); p.height = Number(size[2]); }
    if (P['Hires upscale']) p.hires = { scale: num('Hires upscale'), steps: num('Hires steps') || 4, denoise: num('Denoising strength') || 0.4, upscaler: /anime|esrgan|model|4x/i.test(P['Hires upscaler'] || '') ? 'model' : 'lanczos' };
    if (P['ADetailer model']) p.face = { conf: num('ADetailer confidence') || 0.35, denoise: num('ADetailer denoising strength') || 0.4, steps: num('ADetailer steps') || 4, guide: num('ADetailer inpaint width') || 384 };
    if (P['NAG scale']) p.nag = { scale: num('NAG scale'), tau: num('NAG tau') || 2.5, alpha: num('NAG alpha') || 0.25 };
    if (P['CFG cutoff']) p.cfg_cutoff = num('CFG cutoff');
    if (P['Step cache']) p.step_cache = num('Step cache');
    if (P.Lora) p.loras = String(P.Lora).split(/,\s*/).map(s => { const i = s.lastIndexOf(':'); return { file: i > 0 ? s.slice(0, i) : s, strength: i > 0 ? Number(s.slice(i + 1)) : 1 }; });
    if (P.Model) p.model = P.Model;
    return { params: p, raw: P, prompt: x.prompt, negative: x.negative };
  }

  // ---------------------------------------------------------------- history gallery + lightbox (all outputs)
  const galleryEl = $('gallery'), sentinel = $('gallerySentinel');
  function makeTile(item, isNew) {
    const b = el('button', 'tile' + (isNew ? ' new' : ''));
    b.type = 'button';
    b.dataset.file = item.file;
    if (item.w && item.h) b.style.setProperty('--ar', `${item.w} / ${item.h}`);
    const img = el('img');
    img.alt = (item.params && item.params.prompt) ? String(item.params.prompt).slice(0, 120) : item.file;
    img.loading = 'lazy'; img.decoding = 'async';
    img.onload = () => img.classList.add('loaded');
    img.onerror = () => { img.src = item.url; };
    img.src = item.thumb;
    b.appendChild(img);
    b.addEventListener('click', () => { const i = S.gallery.findIndex(g => g.file === item.file); if (i >= 0) openLightbox(i); });
    return b;
  }
  function renderGalCount() {
    $('galCount').textContent = S.galTotal ? `(${S.galTotal})` : '';
    const empty = galleryEl.querySelector('.gallery-empty');
    if (!S.gallery.length && S.galDone) { if (!empty) galleryEl.insertBefore(el('div', 'gallery-empty', 'No images yet.'), sentinel); }
    else if (empty) empty.remove();
  }
  async function loadGallery(reset) {
    if (S.galLoading || (!reset && S.galDone)) return;
    S.galLoading = true;
    try {
      const offset = reset ? 0 : S.gallery.length;
      const r = await api(`/api/gallery?offset=${offset}&limit=48`);
      if (reset) { S.gallery = []; for (const t of [...galleryEl.querySelectorAll('.tile')]) t.remove(); }
      const have = new Set(S.gallery.map(g => g.file));
      const frag = document.createDocumentFragment();
      for (const it of r.items) { if (have.has(it.file)) continue; S.gallery.push(it); frag.appendChild(makeTile(it, false)); }
      galleryEl.insertBefore(frag, sentinel);
      S.galTotal = r.total;
      S.galDone = S.gallery.length >= r.total || r.items.length === 0;
    } catch (e) { toast('Gallery: ' + e.message, 'err'); }
    finally { S.galLoading = false; renderGalCount(); }
  }
  function addToGallery(item) {
    if (S.gallery.some(g => g.file === item.file)) return;
    S.gallery.unshift(item);
    S.galTotal++;
    galleryEl.insertBefore(makeTile(item, true), galleryEl.querySelector('.tile') || sentinel);
    renderGalCount();
    if (S.lb >= 0) { S.lb++; renderLbIndex(); }
  }
  function removeFromGallery(file) {
    const i = S.gallery.findIndex(g => g.file === file);
    if (i < 0) return;
    S.gallery.splice(i, 1);
    S.galTotal = Math.max(0, S.galTotal - 1);
    const t = galleryEl.querySelector(`.tile[data-file="${CSS.escape(file)}"]`);
    if (t) t.remove();
    renderGalCount();
    if (S.lb >= 0) {
      if (S.lb === i) { if (!S.gallery.length) closeLightbox(); else { S.lb = Math.min(i, S.gallery.length - 1); renderLightbox(); } }
      else if (S.lb > i) { S.lb--; renderLbIndex(); }
    }
  }
  new IntersectionObserver((ents) => { for (const e of ents) if (e.isIntersecting) loadGallery(false); }, { rootMargin: '300px' }).observe(sentinel);
  $('history').addEventListener('toggle', () => { if ($('history').open && !S.gallery.length) loadGallery(true); });

  const lb = $('lightbox');
  function openLightbox(i) { S.lb = i; renderLightbox(); lb.hidden = false; document.body.style.overflow = 'hidden'; }
  function closeLightbox() { lb.hidden = true; S.lb = -1; document.body.style.overflow = ''; $('lbImg').removeAttribute('src'); }
  function renderLbIndex() {
    $('lbIndex').textContent = S.lb >= 0 ? `${S.lb + 1} / ${S.galTotal || S.gallery.length}` : '';
    $('lbPrev').disabled = S.lb <= 0;
    $('lbNext').disabled = S.lb >= S.gallery.length - 1 && S.galDone;
  }
  async function lbStep(d) {
    const i = S.lb + d;
    if (i < 0) return;
    if (i >= S.gallery.length) { if (S.galDone) return; await loadGallery(false); if (i >= S.gallery.length) return; }
    S.lb = i; renderLightbox();
  }
  function renderLightbox() {
    const it = S.gallery[S.lb];
    if (!it) return closeLightbox();
    $('lbImg').src = it.view || it.url;
    $('lbImg').alt = (it.params && it.params.prompt) || it.file;
    renderLbIndex();
    const p = it.params || {};
    const body = $('lbBody');
    body.textContent = '';
    if (p.foreign) body.appendChild(el('div', 'a-info', p.text || ''));
    else if (!it.params) body.appendChild(el('div', 'muted', 'No Kiln metadata in this file.'));
    else body.appendChild(el('div', 'a-info', p.kind === 'graph' && !p.steps ? `Node graph · ${p.nodes || '?'} nodes\n${p.prompt || ''}` : infotext(p, {})));
    body.appendChild(el('div', 'muted small mono', `${it.file}${it.mtime ? ' · ' + new Date(it.mtime).toLocaleString() : ''}`));
    $('lbDownload').href = it.url + '?dl=1';
    $('lbDownload').setAttribute('download', it.file.split('/').pop());
    $('lbDelete').classList.remove('armed'); $('lbDelete').textContent = 'Delete';
    $('lbReuse').disabled = !it.params || !!p.foreign;
    $('lbReuse').textContent = p.kind === 'graph' && !p.mode ? 'Open in Nodes' : 'Reuse settings';
    $('lbSeed').disabled = !it.params || p.seed == null;
  }
  $('lbClose').addEventListener('click', closeLightbox);
  $('lbPrev').addEventListener('click', () => lbStep(-1));
  $('lbNext').addEventListener('click', () => lbStep(1));
  lb.addEventListener('click', (e) => { if (e.target === lb || e.target === $('lbMedia')) closeLightbox(); });
  document.addEventListener('keydown', (e) => {
    if (lb.hidden) return;
    if (e.key === 'Escape') closeLightbox(); else if (e.key === 'ArrowLeft') lbStep(-1); else if (e.key === 'ArrowRight') lbStep(1);
  });
  $('lbReuse').addEventListener('click', async () => {
    const it = S.gallery[S.lb];
    if (!it || !it.params) return;
    if (it.params.kind === 'graph' && !it.params.mode) {
      let meta = it.params;
      if (!meta.graph) { try { const r = await api('/api/gallery?offset=0&limit=200'); const f = r.items.find(x => x.file === it.file); if (f && f.params) meta = f.params; } catch (_) { } }
      closeLightbox();
      await K.setTab('nodes');
      window.KilnNodes.openFromMeta(meta, it.file.split('/').pop());
      return;
    }
    applyParams(it.params);
    closeLightbox();
    await K.setTab(it.params.mode ? 'img2img' : 'txt2img');
    toast('Settings loaded', 'ok', 1500);
  });
  $('lbSeed').addEventListener('click', () => {
    const it = S.gallery[S.lb];
    if (!it || !it.params || it.params.seed == null) return;
    G.seed = it.params.seed; $('seed').value = String(G.seed); save();
    toast(`Seed ${G.seed}`, 'ok', 1500);
  });
  $('lbDelete').addEventListener('click', async () => {
    const del = $('lbDelete'), it = S.gallery[S.lb];
    if (!it) return;
    if (!del.classList.contains('armed')) {
      del.classList.add('armed'); del.textContent = 'Tap again to delete';
      clearTimeout(del._t); del._t = setTimeout(() => { del.classList.remove('armed'); del.textContent = 'Delete'; }, 3000);
      return;
    }
    clearTimeout(del._t);
    try { await api('/api/gallery/' + it.file.split('/').map(encodeURIComponent).join('/'), { method: 'DELETE' }); removeFromGallery(it.file); toast('Deleted', 'ok', 1400); }
    catch (e) { toast('Delete failed: ' + e.message, 'err'); del.classList.remove('armed'); del.textContent = 'Delete'; }
  });
  let touchX = null, touchY = null;
  $('lbMedia').addEventListener('touchstart', (e) => { const t = e.changedTouches[0]; touchX = t.clientX; touchY = t.clientY; }, { passive: true });
  $('lbMedia').addEventListener('touchend', (e) => {
    if (touchX == null) return;
    const t = e.changedTouches[0], dx = t.clientX - touchX, dy = t.clientY - touchY;
    touchX = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.3) lbStep(dx < 0 ? 1 : -1);
  }, { passive: true });

  // engine samplers (for the sampler list)
  async function loadInfo() {
    try {
      const r = await api('/api/info');
      if (r.engine && Array.isArray(r.engine.samplers) && r.engine.samplers.length) { K.S.engineSamplers = r.engine.samplers.slice(); renderSamplers(); }
    } catch (_) { }
  }
  window.addEventListener('kiln:sse', (e) => { if (e.detail.type === 'engine' && e.detail.engine.state === 'ready') loadInfo(); });

  $('toolNets').addEventListener('click', () => { G.xnetOpen = !G.xnetOpen; save(); K.emit('xnet', { open: G.xnetOpen }); });

  window.KilnGen = {
    get G() { return G; }, get family() { return FAM; }, applyParams, a1111ToParams, parseInfotext, infotext, toggleLoraTag, insertText, addToSlot, setI2iImage,
    generate, extractLoraTags, deemphasize, mismatch, renderWarn, loadGallery, stem,
    setSettings(obj) { G = merged(Object.assign({}, G, clone(obj || {}))); save(); renderAll(); },
  };

  // boot
  renderAll();
  applyTab();
  loadInfo();
  loadGallery(true);
  requestAnimationFrame(fitFrame);
})();
