/* Kiln Train tab: an Anima LoRA from a folder of images with .txt captions. Posts to /api/train, follows the run over
   the event stream (caching, steps with the loss, previews, checkpoints) and keeps the last run on screen. */
(() => {
  'use strict';
  const K = window.Kiln;
  const { $, el, api, toast, fmtSec, lsGet, lsSet } = K;
  const LS = 'kiln.train.v1';
  const DEF = {
    dir: '', name: '', trigger: '', preset: 'character', repeats: 10, epochs: 10, rank: '32', res: '640', opt: 'prodigy', lr: 1, saveEvery: 2,
    pvEvery: 100, pvSize: '512x768', pvPrompt: '', pvNeg: 'worst quality, low quality, blurry, jpeg artifacts', lastJob: '',
    shuffle: true, keep: 1, drop: 0, flip: false, batch: 1, warmup: 0, ts: 'sigmoid', sig: 1.3, shift: 3, noiseOff: 0, pvTurbo: true, model: '',
  };
  const T = Object.assign({}, DEF, lsGet(LS) || {});
  if (lsGet(LS) && !('preset' in lsGet(LS))) T.preset = 'custom';  // settings saved before presets existed
  const save = K.debounce(() => lsSet(LS, T), 250);
  const st = { job: null, ds: null };
  // seconds per image on this machine's measurements, by training resolution (a rough guide for the plan line)
  const SEC = { 512: 4.3, 640: 6.2, 768: 8.8 };

  // Presets: starting points, not rules. Repeats follow the image count so an epoch stays near `perEpoch` images, and
  // the epochs keep the total near perEpoch x epochs even for big datasets (where repeats bottom out at 1).
  const SHARED = { opt: 'prodigy', lr: 1, batch: 1, warmup: 0, ts: 'sigmoid', sig: 1.3, shift: 3, noiseOff: 0, shuffle: true, keep: 1 };
  const PRESETS = {
    // characters: the published Anima recipe (AdamW 1e-4, rank 8, ~1000-1500 steps for 20-30 images). Prodigy's own
    // estimate stays small on varied character sets and the LoRA comes out weak.
    character: {
      ...SHARED, opt: 'adamw', lr: 1e-4, warmupFrac: 0.05, perEpoch: 100, epochs: 12, rank: '8', res: '640', saveEvery: 2, pvEvery: 100, drop: 0, flip: false,
      hint: 'Character: AdamW at 1e-4, rank 8, about 1200 steps (the published Anima character recipe). Put a unique trigger word first; it stays in front while the other tags shuffle. Flip is off because hair, outfits and accessories are often one-sided. 15–40 varied images of the character work best.',
    },
    character_quick: {
      ...SHARED, opt: 'adamw', lr: 1e-4, warmupFrac: 0.05, perEpoch: 50, epochs: 12, rank: '8', res: '512', saveEvery: 4, pvEvery: 100, drop: 0, flip: false,
      hint: 'A fast check at 512 px, about 600 steps (AdamW 1e-4, rank 8): do the dataset and captions work before a full run?',
    },
    style: {
      ...SHARED, perEpoch: 100, epochs: 12, rank: '16', res: '768', saveEvery: 2, pvEvery: 150, drop: 5, flip: true,
      hint: 'Style: varied subjects (20+ images) so it learns the look, not one character. 768 px keeps linework and texture; flip doubles the data; 5% caption dropout lets the style show without its tags. Add a preview prompt with a subject that is not in your images.',
    },
    style_quick: {
      ...SHARED, perEpoch: 50, epochs: 10, rank: '16', res: '512', saveEvery: 5, pvEvery: 100, drop: 5, flip: true,
      hint: 'A fast style check at 512 px, about 500 steps. Fine linework is softer at 512; use Style for the real run.',
    },
  };
  const PRESET_KEYS = ['repeats', 'epochs', 'rank', 'res', 'opt', 'lr', 'saveEvery', 'pvEvery', 'shuffle', 'keep', 'drop', 'flip', 'batch', 'warmup', 'ts', 'sig', 'shift', 'noiseOff'];

  // ---- form <-> state
  const fields = [
    ['tDir', 'dir', String], ['tName', 'name', String], ['tTrigger', 'trigger', String], ['tRepeats', 'repeats', Number], ['tEpochs', 'epochs', Number],
    ['tRank', 'rank', String], ['tRes', 'res', String], ['tOpt', 'opt', String], ['tLr', 'lr', Number], ['tSaveEvery', 'saveEvery', Number],
    ['tPvEvery', 'pvEvery', Number], ['tPvSize', 'pvSize', String], ['tPvPrompt', 'pvPrompt', String], ['tPvNeg', 'pvNeg', String],
    ['tShuffle', 'shuffle', Boolean], ['tKeep', 'keep', Number], ['tDrop', 'drop', Number], ['tFlip', 'flip', Boolean], ['tBatch', 'batch', Number],
    ['tWarmup', 'warmup', Number], ['tTs', 'ts', String], ['tSig', 'sig', Number], ['tShift', 'shift', Number], ['tNoiseOff', 'noiseOff', Number],
    ['tPvTurbo', 'pvTurbo', Boolean],
  ];
  const show = () => {
    for (const [id, key] of fields) { const e = $(id); if (e.type === 'checkbox') e.checked = !!T[key]; else e.value = T[key]; }
    $('tPreset').value = T.preset;
    $('tPresetHint').textContent = (PRESETS[T.preset] || {}).hint || 'Your own settings.';
    $('tShiftF').hidden = T.ts !== 'shift';
    $('tSigF').hidden = T.ts === 'uniform';
  };
  for (const [id, key, conv] of fields) {
    const e = $(id);
    e.addEventListener(e.tagName === 'TEXTAREA' || e.type === 'text' ? 'input' : 'change', () => {
      T[key] = e.type === 'checkbox' ? e.checked : conv(e.value);
      if (PRESET_KEYS.includes(key) && T.preset !== 'custom') T.preset = 'custom';
      show();
      save();
      plan();
    });
  }
  $('tOpt').addEventListener('change', () => {  // Prodigy's lr is a multiplier on its own estimate; AdamW needs a real one
    T.lr = T.opt === 'prodigy' ? 1 : 1e-4;
    $('tLr').value = T.lr;
    save();
  });
  function applyPreset(name) {
    const p = PRESETS[name];
    T.preset = p ? name : 'custom';
    if (p) {
      for (const k of PRESET_KEYS) if (p[k] !== undefined) T[k] = p[k];
      const n = st.ds && st.ds.images ? st.ds.images : 10;
      T.repeats = Math.max(1, Math.round(p.perEpoch / n));
      T.epochs = Math.max(2, Math.round(p.perEpoch * p.epochs / (n * T.repeats)));
      const steps = T.epochs * Math.ceil(n * T.repeats / Math.max(1, T.batch || 1));
      T.warmup = p.warmupFrac ? Math.round(steps * p.warmupFrac) : 0;
    }
    show();
    save();
    plan();
  }
  $('tPreset').addEventListener('change', () => applyPreset($('tPreset').value));
  show();

  // the Anima checkpoints; until one is picked here, the one txt2img uses
  function fillModels() {
    const list = ((K.S.models && K.S.models.sd_models) || []).filter(m => m.family === 'anima');
    const sel = $('tModel');
    sel.textContent = '';
    for (const m of list) { const o = el('option', '', m.name); o.value = m.file; sel.appendChild(o); }
    if (!list.some(m => m.file === T.model)) {
      const c = K.ckptInfo();
      T.model = c && c.family === 'anima' ? c.file : (list[0] ? list[0].file : '');
    }
    sel.value = T.model;
  }
  $('tModel').addEventListener('change', () => { T.model = $('tModel').value; save(); });
  K.on('models', fillModels);
  fillModels();

  // ---- dataset
  const literal = (t) => t.replace(/([()[\]])/g, '\\$1');
  async function scan() {
    if (!T.dir.trim()) { $('tDs').textContent = 'Type the folder of your images first.'; return; }
    try {
      const r = await api('/api/train/dataset?dir=' + encodeURIComponent(T.dir.trim()));
      st.ds = r;
      $('tDs').textContent = `${r.images} image${r.images === 1 ? '' : 's'}, ${r.captioned} with captions · ${r.dir}`;
      const sel = $('tPvFrom');
      while (sel.options.length > 1) sel.remove(1);
      for (const it of r.items) {
        const o = el('option');
        o.value = it.file;
        o.textContent = it.file.replace(/\.[^.]+$/, '') + (it.caption ? '' : ' (no caption)');
        sel.appendChild(o);
      }
      if (!T.name) { T.name = r.dir.split(/[\\/]/).filter(Boolean).pop() || ''; $('tName').value = T.name; save(); }
      if (PRESETS[T.preset]) applyPreset(T.preset);  // repeats and epochs follow the image count
      plan();
    } catch (e) {
      st.ds = null;
      $('tDs').textContent = e.message;
      plan();
    }
  }
  $('tScan').addEventListener('click', scan);
  // the Datasets tab's "Train on this"
  window.KilnTrain = { useDataset(dir) { T.dir = dir; $('tDir').value = dir; save(); scan(); } };
  $('tDir').addEventListener('keydown', (e) => { if (e.key === 'Enter') scan(); });
  $('tPvFrom').addEventListener('change', () => {
    const it = st.ds && st.ds.items.find(i => i.file === $('tPvFrom').value);
    if (!it) return;
    // the trigger in front, once (as the trainer captions the image)
    const split = (t) => String(t || '').split(',').map(x => x.trim()).filter(Boolean);
    const norm = (t) => t.toLowerCase().replace(/_/g, ' ');
    const trig = split(T.trigger), seen = new Set(trig.map(norm));
    // tags that are right in a caption but would ask the preview for them: signatures, watermarks, usernames ...
    const NOT_IN_PROMPT = /^(signature|watermark|artist name|dated|web address|copyright name|character name|.* username|.* logo|patreon .*|commentary.*|.* commentary|translated|translation request|text|english text|speech bubble)$/;
    const cap = [...trig, ...split(it.caption).filter(t => !seen.has(norm(t)) && !NOT_IN_PROMPT.test(norm(t)))].join(', ');
    T.pvPrompt = literal(cap);
    $('tPvPrompt').value = T.pvPrompt;
    save();
  });

  function plan() {
    if (!st.ds || !st.ds.images) { $('tPlan').textContent = ''; return; }
    const per = st.ds.images * Math.max(1, T.repeats || 1), batch = Math.max(1, T.batch || 1);
    const epochSteps = Math.ceil(per / batch), steps = epochSteps * Math.max(1, T.epochs || 1);
    const sec = steps * batch * (SEC[T.res] || 5);
    $('tPlan').textContent = `${st.ds.images} images × ${T.repeats} repeats = ${per} images an epoch` +
      (batch > 1 ? ` (${epochSteps} steps of ${batch})` : '') + `; × ${T.epochs} epochs = ${steps} steps, about ${fmtDur(sec)} on this PC` +
      (T.pvEvery > 0 ? ` plus about 30 s per preview (every ${T.pvEvery} steps).` : '.') + ` Saves to models/loras/${T.name || '…'}.safetensors.`;
  }
  function fmtDur(s) {
    s = Math.max(0, Math.round(s));
    if (s < 90) return s + ' s';
    if (s < 5400) return Math.round(s / 60) + ' min';
    return (s / 3600).toFixed(1) + ' h';
  }

  // ---- start / stop
  async function start() {
    if (st.job && ['queued', 'running'].includes(st.job.status)) { toast('A training run is already going'); return; }
    if (!st.ds) await scan();
    if (!st.ds || !st.ds.images) { toast('Pick a folder with images first', 'err'); return; }
    const [pw, ph] = T.pvSize.split('x').map(Number);
    try {
      const r = await api('/api/train', { method: 'POST', body: {
        dataset: T.dir.trim(), name: T.name.trim(), trigger: T.trigger.trim(), repeats: T.repeats, epochs: T.epochs, rank: Number(T.rank), alpha: Number(T.rank),
        optimizer: T.opt, lr: T.lr, resolution: Number(T.res), save_every_epochs: T.saveEvery, preview_every: T.pvEvery,
        preview_prompt: T.pvPrompt, preview_negative: T.pvNeg, preview_w: pw, preview_h: ph, preview_turbo: T.pvTurbo, preset: T.preset, model: T.model,
        batch_size: T.batch, warmup: T.warmup, shuffle_caption: T.shuffle, keep_tokens: T.keep, caption_dropout: (T.drop || 0) / 100,
        flip: T.flip, noise_offset: T.noiseOff, timestep_sampling: T.ts, sigmoid_scale: T.sig, discrete_flow_shift: T.shift,
      } });
      T.lastJob = r.id;
      save();
      st.job = { id: r.id, status: 'queued', params: { name: T.name, steps: r.steps, out: r.out }, of: r.steps, step: 0, train: { stage: 'queued', losses: [], previews: [], saved: [] } };
      render();
    } catch (e) { toast('Could not start: ' + e.message, 'err', 6000); }
  }
  $('tGo').addEventListener('click', start);
  $('tCancel').addEventListener('click', () => { if (st.job) K.cancel(st.job.id); });

  // ---- view
  function render() {
    const j = st.job;
    const busy = !!j && ['queued', 'running'].includes(j.status);
    $('tGo').disabled = busy;
    $('tCancel').hidden = !busy;
    if (!j) return;
    const t = j.train || {};
    const p = j.params || {};
    let text = '', frac = null, stage = '';
    if (j.status === 'queued') text = 'Waiting for the renders ahead of it…';
    else if (j.status === 'done') text = `<b>Done.</b> ${esc(p.out || '')} · ${fmtSec(j.total_ms || 0)} s`;
    else if (j.status === 'cancelled') text = `<b>Stopped</b> at step ${j.step || 0}` + (t.saved && t.saved.length ? `; saved ${esc(t.saved[t.saved.length - 1].file)}` : '');
    else if (j.status === 'error') text = '<b>Failed:</b> ' + esc(j.error || 'unknown error');
    else if (t.stage === 'cache' && t.cache) { text = `Caching images and captions ${t.cache.done}/${t.cache.of}…`; frac = t.cache.done / t.cache.of; stage = 'caching'; }
    else if (t.stage === 'train' || t.stage === 'ready') {
      const left = Math.max(0, (j.of - j.step)) * (t.msAvg || 0) / 1000;
      const last = t.last;
      text = `Epoch <b>${t.epoch || 1}/${t.epochs || '?'}</b> · step <b>${j.step}/${j.of}</b>` +
        (t.msAvg ? ` · ${(t.msAvg / 1000).toFixed(1)} s/step · about ${fmtDur(left)} left` : '') +
        (last ? ` · loss ${last[2].toFixed(4)}` : '') + (t.d ? ` · Prodigy d ${Number(t.d).toExponential(1)}` : '');
      frac = j.of ? j.step / j.of : 0;
      stage = 'training';
    } else text = 'Loading Anima…';
    $('tStatus').innerHTML = `<b>${esc(p.name || '')}</b> — ` + text + (t.warnings || []).map(w => `<div class="t-warn">${esc(w)}</div>`).join('');
    $('tProg').hidden = frac == null;
    if (frac != null) { $('tBar').style.width = (100 * Math.min(1, frac)).toFixed(1) + '%'; $('tStage').textContent = stage; }
    drawLoss(t.losses || []);
    // previews, newest step first
    const pv = t.previews || [];
    $('tPvBlock').hidden = !pv.length;
    const box = $('tPreviews');
    box.textContent = '';
    const steps = [...new Set(pv.map(x => x.step))].sort((a, b) => b - a);
    for (const s of steps) {
      const row = el('div');
      row.appendChild(el('div', 't-pv-step', s === 0 ? 'Before training (step 0)' : `Step ${s}`));
      const imgs = el('div', 't-pv-row');
      for (const x of pv.filter(y => y.step === s).sort((a, b) => a.index - b.index)) {
        const a = el('a');
        a.href = x.url; a.target = '_blank'; a.rel = 'noopener';
        const im = el('img');
        im.src = x.url; im.loading = 'lazy'; im.alt = `step ${s}`;
        a.appendChild(im);
        imgs.appendChild(a);
      }
      row.appendChild(imgs);
      box.appendChild(row);
    }
    const sv = t.saved || [];
    $('tSavedBlock').hidden = !sv.length;
    $('tSaved').textContent = '';
    for (const x of sv) {
      const d = el('div');
      d.textContent = x.file + ' ';
      d.appendChild(el('span', '', `epoch ${x.epoch}, step ${x.step}`));
      $('tSaved').appendChild(d);
    }
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function drawLoss(L) {
    const cv = $('tLoss');
    cv.hidden = L.length < 2;
    if (L.length < 2) return;
    const w = cv.clientWidth || 640, h = cv.clientHeight || 160, dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const g = cv.getContext('2d');
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const x0 = L[0][0], x1 = L[L.length - 1][0];
    const vals = L.map(r => r[1]).sort((a, b) => a - b);
    const top = vals[Math.floor(vals.length * 0.98)] * 1.1 || 1;
    const X = (s) => 8 + (w - 16) * (s - x0) / Math.max(1, x1 - x0), Y = (v) => h - 18 - (h - 30) * Math.min(1, v / top);
    g.fillStyle = 'rgba(156,163,175,.35)';
    for (const r of L) g.fillRect(X(r[0]) - 1, Y(r[1]) - 1, 2, 2);
    g.strokeStyle = '#f97316'; g.lineWidth = 2;
    g.beginPath();
    L.forEach((r, i) => (i ? g.lineTo(X(r[0]), Y(r[2])) : g.moveTo(X(r[0]), Y(r[2]))));
    g.stroke();
    g.fillStyle = '#9ca3af'; g.font = '11px system-ui';
    g.fillText(`loss (average ${L[L.length - 1][2].toFixed(4)})`, 10, 14);
    g.fillText(`step ${x1}`, w - 60, h - 4);
  }

  // ---- live updates
  window.addEventListener('kiln:sse', (e) => {
    const m = e.detail;
    if (m.type === 'train') {
      if (!st.job || st.job.id !== m.id) st.job = { id: m.id, status: 'running', params: {}, train: {} };
      st.job.status = 'running';
      st.job.step = m.step; st.job.of = m.of; st.job.train = m.train;
      render();
    } else if (m.type === 'job' && m.job && m.job.kind === 'train') {
      if (!st.job || st.job.id === m.job.id || ['queued', 'running'].includes(m.job.status)) {
        st.job = m.job;
        if (m.job.id !== T.lastJob) { T.lastJob = m.job.id; save(); }
        render();
        if (m.job.status === 'done') toast(`LoRA saved: ${m.job.params.out}`, 'ok', 6000);
      }
    }
  });
  window.addEventListener('resize', K.debounce(() => { if (st.job && st.job.train) drawLoss(st.job.train.losses || []); }, 200));

  // the last run (or the one going now) after a reload
  (async () => {
    if (T.dir) scan();
    try {
      const q = await api('/api/queue');
      const live = [q.running, ...(q.queued || [])].find(j => j && j.kind === 'train');
      if (live) { st.job = live; render(); return; }
    } catch (_) { }
    if (T.lastJob) {
      try { st.job = await api('/api/job/' + encodeURIComponent(T.lastJob)); render(); } catch (_) { }
    }
  })();
})();
