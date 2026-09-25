'use strict';
// A1111-compatible prompt handling and generation info ("infotext"):
//  - <lora:name:weight> (and <lyco:...>) tags are taken out of the prompt and returned separately
//  - [text] de-emphasis (weight / 1.1 per bracket level) becomes ComfyUI-style (text:0.9091), which
//    Kiln's tokenizer understands; A1111 prompt editing [a:b:0.5] and alternation [a|b] are left as text
//  - infotext(): the "parameters" text A1111 writes into PNGs (and shows under the gallery)

const LORA_TAG = /<(lora|lyco):([^:<>]+)(?::([^:<>]*))?(?::[^<>]*)?>/gi;

function extractLoraTags(prompt) {
  const tags = [];
  const text = String(prompt || '').replace(LORA_TAG, (m, kind, name, w) => {
    const n = w === undefined || !w.trim() ? 1 : Number(w);
    tags.push({ name: name.trim(), weight: Number.isFinite(n) ? n : 1, tag: m });
    return '';
  });
  if (!tags.length) return { text, tags };
  // tidy the separators the tags leave behind: "a, <lora:x>, b" -> "a, b"
  const tidy = text.split('\n').map(line => line.replace(/(\s*,\s*){2,}/g, ', ').replace(/^\s*,\s*|\s*,\s*$/g, '').replace(/[ \t]{2,}/g, ' ').trim()).join('\n').trim();
  return { text: tidy, tags };
}

const round4 = (v) => Math.round(v * 10000) / 10000;
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
      // ':' or '|' at this level (outside parentheses / nested groups) = prompt editing or alternation
      const level = inner.replace(/\\./g, '').replace(new RegExp(OPEN + '[^' + OPEN + CLOSE + ']*' + CLOSE, 'g'), '').replace(/\([^()]*\)/g, '');
      frame.push(/[:|]/.test(level) ? '[' + inner + ']' : OPEN + inner + CLOSE);
      continue;
    }
    frame.push(c);
  }
  while (stack.length) { const inner = frame.join(''); frame = stack.pop(); frame.push('[' + inner); }
  return expand(frame.join(''), 1);
}
// markers -> (text:w); nested levels multiply (ComfyUI's explicit weights replace, so write the product)
function expand(s, w) {
  let out = '';
  for (let i = 0; i < s.length; i++) {
    if (s[i] !== OPEN) { out += s[i]; continue; }
    let depth = 1, j = i + 1;
    for (; j < s.length && depth; j++) { if (s[j] === OPEN) depth++; else if (s[j] === CLOSE) depth--; }
    const inner = s.slice(i + 1, j - 1);
    const nw = w / 1.1;
    out += `(${expand(inner, nw)}:${round4(nw)})`;
    i = j - 1;
  }
  return out;
}

// the text Kiln tokenizes for an A1111-style prompt
function toKilnPrompt(prompt) {
  const x = extractLoraTags(prompt);
  return { text: deemphasize(x.text), tags: x.tags };
}

// ---------------------------------------------------------------------------
// infotext
// ---------------------------------------------------------------------------
const SAMPLER_LABELS = { euler: 'Euler', euler_ancestral: 'Euler a', dpmpp_2m: 'DPM++ 2M', res_multistep: 'Res Multistep' };
const SAMPLER_KEYS = Object.fromEntries(Object.entries(SAMPLER_LABELS).map(([k, v]) => [v.toLowerCase(), k]));
const SCHED_LABELS = { simple: 'Simple', sgm_uniform: 'SGM Uniform', karras: 'Karras', exponential: 'Exponential', ddim_uniform: 'DDIM Uniform', beta: 'Beta', normal: 'Normal', linear_quadratic: 'Linear Quadratic', kl_optimal: 'KL Optimal' };
const cap = (s) => String(s || '').replace(/(^|_)([a-z])/g, (m, a, b) => (a ? ' ' : '') + b.toUpperCase());
const quote = (v) => (/[,:\n"]/.test(String(v)) ? JSON.stringify(String(v)) : String(v));
const stem = (f) => String(f || '').split('/').pop().replace(/\.(safetensors|sft|ckpt|pt)$/i, '');

function infotext(p, extra = {}) {
  const f = [];
  const add = (k, v) => { if (v !== undefined && v !== null && v !== '') f.push(`${k}: ${quote(v)}`); };
  add('Steps', p.steps);
  add('Sampler', SAMPLER_LABELS[p.sampler] || p.sampler);
  add('Schedule type', SCHED_LABELS[p.scheduler || 'simple'] || cap(p.scheduler));
  add('CFG scale', p.cfg);
  add('Seed', p.seed);
  const w = p.width, h = p.height;
  if (w && h) add('Size', `${w}x${h}`);
  if (p.model) add('Model', stem(p.model));
  if (p.denoise != null && p.denoise < 1) add('Denoising strength', p.denoise);
  if (p.hires) {
    add('Denoising strength', p.denoise != null && p.denoise < 1 ? undefined : p.hires.denoise);
    add('Hires upscale', p.hires.scale);
    add('Hires steps', p.hires.steps);
    add('Hires upscaler', p.hires.upscaler === 'model' ? '4x-AnimeSharp' : 'Lanczos');
  }
  if (p.face) {
    add('ADetailer model', 'face_yolov8m.pt');
    add('ADetailer confidence', p.face.conf);
    add('ADetailer denoising strength', p.face.denoise);
    add('ADetailer steps', p.face.steps);
    add('ADetailer inpaint width', p.face.guide);
    add('ADetailer inpaint height', p.face.guide);
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

module.exports = { extractLoraTags, deemphasize, toKilnPrompt, infotext, SAMPLER_LABELS, SAMPLER_KEYS };
