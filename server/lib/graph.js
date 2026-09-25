'use strict';
// Kiln node graphs: catalog (server/nodes.json, ComfyUI object_info shape), validation,
// pruning and built-in example workflows. Graphs are ComfyUI API-format workflows.

const fs = require('fs');

const WIDGET_TYPES = new Set(['INT', 'FLOAT', 'STRING', 'BOOLEAN']);
const SAMPLERS = ['euler', 'euler_ancestral', 'dpmpp_2m', 'res_multistep'];
const SCHEDULERS = ['simple', 'sgm_uniform', 'karras', 'exponential', 'ddim_uniform', 'beta', 'normal', 'linear_quadratic', 'kl_optimal'];

let rawCatalog = null, rawMtime = 0;
function loadCatalog(file) {
  const st = fs.statSync(file);
  if (!rawCatalog || st.mtimeMs !== rawMtime) {
    const j = JSON.parse(fs.readFileSync(file, 'utf8'));
    for (const k of Object.keys(j)) if (k.startsWith('_')) delete j[k];
    rawCatalog = j; rawMtime = st.mtimeMs;
  }
  return rawCatalog;
}

// Replace "$name" combo placeholders with concrete lists. Filled combos get
// opts.kiln_list = "$name" so the editor knows they are file lists.
function fillCatalog(cat, lists) {
  const out = {};
  for (const [cls, e] of Object.entries(cat)) {
    const c = JSON.parse(JSON.stringify(e));
    for (const sect of ['required', 'optional']) {
      const inputs = c.input && c.input[sect];
      if (!inputs) continue;
      for (const [name, spec] of Object.entries(inputs)) {
        if (typeof spec[0] === 'string' && spec[0].startsWith('$')) {
          const key = spec[0].slice(1);
          const opts = Object.assign({}, spec[1] || {}, { kiln_list: spec[0] });
          inputs[name] = [(lists[key] || []).slice(), opts];
        }
      }
    }
    c.name = cls;
    out[cls] = c;
  }
  return out;
}

function inputEntries(entry) {
  const res = [];
  for (const sect of ['required', 'optional']) {
    const inputs = entry.input && entry.input[sect];
    if (!inputs) continue;
    for (const [name, spec] of Object.entries(inputs)) res.push({ name, spec, optional: sect === 'optional' });
  }
  return res;
}
const isWidgetSpec = (spec) => Array.isArray(spec[0]) || WIDGET_TYPES.has(spec[0]) || (typeof spec[0] === 'string' && spec[0].startsWith('$'));
const isLink = (v) => Array.isArray(v) && v.length === 2 && (typeof v[0] === 'string' || typeof v[0] === 'number') && Number.isInteger(v[1]);

// Validate + prune an API-format graph. Returns { ok, graph (pruned), errors {id: msg}, unknown [{id,class_type}], outputs [ids] }.
// opts.textLinks: CLIPTextEncode "text" may still be a link (to a pack node the server folds to a constant).
function checkGraph(graph, cat, opts = {}) {
  const errors = {};
  const unknown = [];
  if (!graph || typeof graph !== 'object' || Array.isArray(graph)) return { ok: false, error: 'graph must be an object keyed by node id', errors, unknown };
  const ids = Object.keys(graph);
  if (!ids.length) return { ok: false, error: 'graph is empty', errors, unknown };
  for (const id of ids) {
    const n = graph[id];
    if (!n || typeof n !== 'object' || typeof n.class_type !== 'string') { errors[id] = 'node has no class_type'; continue; }
    if (!cat[n.class_type]) unknown.push({ id, class_type: n.class_type });
    if (n.inputs != null && (typeof n.inputs !== 'object' || Array.isArray(n.inputs))) errors[id] = 'inputs must be an object';
  }
  // output nodes
  const outputs = ids.filter(id => graph[id] && cat[graph[id].class_type] && cat[graph[id].class_type].output_node);
  // reachability (following links backwards from output nodes)
  const keep = new Set();
  const stack = outputs.slice();
  while (stack.length) {
    const id = stack.pop();
    if (keep.has(id)) continue;
    keep.add(id);
    const n = graph[id];
    for (const v of Object.values((n && n.inputs) || {})) {
      if (isLink(v)) {
        const src = String(v[0]);
        if (graph[src]) stack.push(src);
      }
    }
  }
  // unknown nodes only matter if they're needed; but an unknown node with no known consumer might itself be an output node
  const unknownNeeded = unknown.filter(u => keep.has(u.id) || !ids.some(other => Object.values((graph[other] && graph[other].inputs) || {}).some(v => isLink(v) && String(v[0]) === u.id)));
  if (unknownNeeded.length) return { ok: false, error: 'unsupported node types: ' + [...new Set(unknownNeeded.map(u => u.class_type))].join(', '), errors, unknown: unknownNeeded };
  if (!outputs.length) return { ok: false, error: 'graph has no output node (SaveImage / PreviewImage)', errors, unknown };

  const pruned = {};
  for (const id of ids) if (keep.has(id)) pruned[id] = { class_type: graph[id].class_type, inputs: Object.assign({}, graph[id].inputs || {}) };
  // per-node checks on what will run
  for (const id of Object.keys(pruned)) {
    const n = pruned[id], e = cat[n.class_type];
    const msgs = [];
    for (const inp of inputEntries(e)) {
      const v = n.inputs[inp.name];
      const widget = isWidgetSpec(inp.spec);
      if (v === undefined || v === null) {
        if (!inp.optional) {
          if (widget && inp.spec[1] && inp.spec[1].default !== undefined) n.inputs[inp.name] = inp.spec[1].default;
          else msgs.push(`missing input "${inp.name}"`);
        }
        continue;
      }
      if (isLink(v)) {
        const src = pruned[String(v[0])];
        if (!src) { msgs.push(`input "${inp.name}" links to missing node ${v[0]}`); continue; }
        const se = cat[src.class_type];
        const outType = se && se.output[v[1]];
        if (!outType) { msgs.push(`input "${inp.name}" links to output ${v[1]} of ${src.class_type}, which has ${se ? se.output.length : 0} outputs`); continue; }
        const want = Array.isArray(inp.spec[0]) || String(inp.spec[0]).startsWith('$') ? 'COMBO' : inp.spec[0];
        if (want !== '*' && outType !== '*' && outType !== want) msgs.push(`input "${inp.name}" expects ${want} but gets ${outType} from ${src.class_type}`);
        n.inputs[inp.name] = [String(v[0]), v[1]];
        continue;
      }
      if (!widget) { msgs.push(`input "${inp.name}" (${inp.spec[0]}) must be connected`); continue; }
      const t = Array.isArray(inp.spec[0]) || String(inp.spec[0]).startsWith('$') ? 'COMBO' : inp.spec[0];
      if ((t === 'INT' || t === 'FLOAT')) {
        const num = typeof v === 'string' && v.trim() !== '' ? Number(v) : v;
        if (typeof num !== 'number' || !Number.isFinite(num)) msgs.push(`input "${inp.name}" must be a number`);
        else n.inputs[inp.name] = t === 'INT' ? Math.round(num) : num;
      } else if (t === 'BOOLEAN') {
        if (typeof v !== 'boolean') n.inputs[inp.name] = v === 'true' || v === 1 || v === true;
      } else if (t === 'STRING' || t === 'COMBO') {
        if (typeof v !== 'string') n.inputs[inp.name] = String(v);
      }
    }
    if (n.class_type === 'CLIPTextEncode' && typeof n.inputs.text !== 'string' && !(opts.textLinks && isLink(n.inputs.text))) msgs.push('"text" must be a literal string, or come from a pack node whose inputs are all constants');
    if (msgs.length) errors[id] = msgs.join('; ');
  }
  if (Object.keys(errors).length) return { ok: false, error: 'invalid graph: ' + Object.entries(errors).map(([id, m]) => `#${id} ${pruned[id] ? pruned[id].class_type : ''}: ${m}`).join(' | '), errors, unknown };
  return { ok: true, graph: pruned, errors, unknown, outputs, prunedCount: ids.length - Object.keys(pruned).length };
}

// A short human summary of a graph (for the queue list / gallery params).
function summarize(graph) {
  const nodes = Object.entries(graph);
  const samplers = nodes.filter(([, n]) => /^KSampler/.test(n.class_type));
  const s = samplers.length ? samplers[0][1].inputs : {};
  let prompt = '', negative = '';
  const textOf = (link) => {
    if (!Array.isArray(link)) return '';
    const n = graph[String(link[0])];
    return n && n.class_type === 'CLIPTextEncode' && typeof n.inputs.text === 'string' ? n.inputs.text : '';
  };
  if (samplers.length) { prompt = textOf(s.positive); negative = textOf(s.negative); }
  if (!prompt) { const t = nodes.find(([, n]) => n.class_type === 'CLIPTextEncode'); if (t) prompt = t[1].inputs.text || ''; }
  const lat = nodes.find(([, n]) => /EmptyS?D?3?LatentImage/.test(n.class_type));
  return {
    kind: 'graph', nodes: nodes.length, prompt, negative,
    seed: typeof s.seed === 'number' ? s.seed : (typeof s.noise_seed === 'number' ? s.noise_seed : undefined),
    steps: typeof s.steps === 'number' ? s.steps : undefined, cfg: typeof s.cfg === 'number' ? s.cfg : undefined,
    sampler: typeof s.sampler_name === 'string' ? s.sampler_name : undefined,
    width: lat ? lat[1].inputs.width : undefined, height: lat ? lat[1].inputs.height : undefined,
    classes: [...new Set(nodes.map(([, n]) => n.class_type))],
  };
}

// ---------------------------------------------------------------------------
// Built-in example workflows (API format + editor positions)
// ---------------------------------------------------------------------------
function examples(lists) {
  const pick = (list, re, fallback) => (list || []).find(x => re.test(x)) || (list || [])[0] || fallback;
  const dit = pick(lists.diffusion_models, /anima.*base/i, 'anima-base-v1.0.safetensors');
  const turbo = pick(lists.loras, /turbo/i, 'anima-turbo-lora-v0.2.safetensors');
  const te = pick(lists.text_encoders, /qwen/i, 'qwen_3_06b_base.safetensors');
  const vae = pick(lists.vaes, /vae/i, 'qwen_image_vae.safetensors');
  const ups = pick(lists.upscale_models, /animesharp/i, '4x-AnimeSharp.safetensors');
  const bbox = pick(lists.bbox_models, /face/i, 'bbox/face_yolov8m.pt');
  const img = (lists.images || [])[0] || 'example.png';
  const POS = '1girl, solo, long hair, looking at viewer, smile, upper body, masterpiece, best quality';
  const NEG = 'worst quality, low quality, blurry, jpeg artifacts, watermark';

  const base = (o = {}) => ({
    1: { class_type: 'UNETLoader', inputs: { unet_name: dit, weight_dtype: 'default' } },
    3: { class_type: 'CLIPLoader', inputs: { clip_name: te, type: 'stable_diffusion', device: 'default' } },
    4: { class_type: 'CLIPTextEncode', inputs: { text: o.pos || POS, clip: ['3', 0] }, _meta: { title: 'Positive' } },
    5: { class_type: 'CLIPTextEncode', inputs: { text: o.neg || NEG, clip: ['3', 0] }, _meta: { title: 'Negative' } },
    8: { class_type: 'VAELoader', inputs: { vae_name: vae } },
  });
  const turboNodes = () => ({ 2: { class_type: 'LoraLoaderModelOnly', inputs: { model: ['1', 0], lora_name: turbo, strength_model: 1.0 } } });
  const ks = (model, latent, o = {}) => ({
    class_type: 'KSampler', inputs: {
      model, seed: o.seed ?? 0, steps: o.steps ?? 8, cfg: o.cfg ?? 1.0, sampler_name: o.sampler || 'euler', scheduler: o.scheduler || 'simple',
      positive: ['4', 0], negative: ['5', 0], latent_image: latent, denoise: o.denoise ?? 1.0,
    },
  });
  const P = {
    1: [40, 40], 2: [360, 40], 3: [40, 250], 4: [360, 230], 5: [360, 470], 6: [360, 700], 8: [700, 620],
    7: [700, 40], 9: [1040, 40], 10: [1340, 40],
  };

  const ex = [];
  // 1. turbo txt2img
  ex.push({
    name: 'anima-turbo-txt2img', title: 'Anima turbo txt2img', description: 'Turbo LoRA 1.0 · 8 steps · CFG 1 · euler/simple · 512×768',
    graph: Object.assign(base(), turboNodes(), {
      6: { class_type: 'EmptySD3LatentImage', inputs: { width: 512, height: 768, batch_size: 1 } },
      7: ks(['2', 0], ['6', 0]),
      9: { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
      10: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Kiln' } },
    }),
    pos: P,
  });
  // 2. turbo + FaceDetailer + 4x-AnimeSharp (then 0.5x)
  ex.push({
    name: 'anima-turbo-facedetailer-upscale', title: 'Turbo + FaceDetailer + upscale', description: 'FaceDetailer (guide 384 / max 576 / 4 steps / denoise 0.4 / crop 2.0) → 4x-AnimeSharp → 0.5× → Save',
    graph: Object.assign(base(), turboNodes(), {
      6: { class_type: 'EmptySD3LatentImage', inputs: { width: 512, height: 768, batch_size: 1 } },
      7: ks(['2', 0], ['6', 0]),
      9: { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
      11: { class_type: 'UltralyticsDetectorProvider', inputs: { model_name: bbox } },
      12: {
        class_type: 'FaceDetailer', inputs: {
          image: ['9', 0], model: ['2', 0], clip: ['3', 0], vae: ['8', 0], guide_size: 384, guide_size_for: true, max_size: 576,
          seed: 0, steps: 4, cfg: 1.0, sampler_name: 'euler', scheduler: 'simple', positive: ['4', 0], negative: ['5', 0], denoise: 0.4,
          feather: 5, noise_mask: true, force_inpaint: true, bbox_threshold: 0.5, bbox_dilation: 10, bbox_crop_factor: 2.0,
          sam_detection_hint: 'center-1', sam_dilation: 0, sam_threshold: 0.93, sam_bbox_expansion: 0, sam_mask_hint_threshold: 0.7,
          sam_mask_hint_use_negative: 'False', drop_size: 10, bbox_detector: ['11', 0], wildcard: '', cycle: 1,
        },
      },
      13: { class_type: 'UpscaleModelLoader', inputs: { model_name: ups } },
      14: { class_type: 'ImageUpscaleWithModel', inputs: { upscale_model: ['13', 0], image: ['12', 0] } },
      15: { class_type: 'ImageScaleBy', inputs: { image: ['14', 0], upscale_method: 'lanczos', scale_by: 0.5 } },
      16: { class_type: 'PreviewImage', inputs: { images: ['9', 0] }, _meta: { title: 'Before detailer' } },
      10: { class_type: 'SaveImage', inputs: { images: ['15', 0], filename_prefix: 'Kiln_detailed' } },
    }),
    pos: Object.assign({}, P, { 9: [1040, 40], 16: [1040, 200], 11: [700, 780], 12: [1340, 40], 13: [1680, 420], 14: [1680, 40], 15: [1680, 200], 10: [1980, 40] }),
  });
  // 3. base Anima, DPM++ 2M 20 steps CFG 4.5
  ex.push({
    name: 'anima-base-dpmpp2m', title: 'Base Anima (DPM++ 2M, CFG 4.5)', description: 'No LoRA · dpmpp_2m · 20 steps · CFG 4.5',
    graph: Object.assign(base(), {
      6: { class_type: 'EmptySD3LatentImage', inputs: { width: 512, height: 768, batch_size: 1 } },
      7: ks(['1', 0], ['6', 0], { steps: 20, cfg: 4.5, sampler: 'dpmpp_2m' }),
      9: { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
      10: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Kiln_base' } },
    }),
    pos: P,
  });
  // 3b. base Anima with KSamplerWithNAG
  ex.push({
    name: 'anima-base-nag', title: 'Base Anima + NAG', description: 'KSamplerWithNAG · dpmpp_2m · 20 steps · CFG 4.5 · NAG 5 / 2.5 / 0.25',
    graph: Object.assign(base(), {
      6: { class_type: 'EmptySD3LatentImage', inputs: { width: 512, height: 768, batch_size: 1 } },
      7: {
        class_type: 'KSamplerWithNAG', inputs: {
          model: ['1', 0], seed: 0, steps: 20, cfg: 4.5, nag_scale: 5.0, nag_tau: 2.5, nag_alpha: 0.25, nag_sigma_end: 0.0,
          sampler_name: 'dpmpp_2m', scheduler: 'simple', positive: ['4', 0], negative: ['5', 0], nag_negative: ['5', 0], latent_image: ['6', 0], denoise: 1.0,
        },
      },
      9: { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
      10: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Kiln_nag' } },
    }),
    pos: P,
  });
  // 4. img2img
  ex.push({
    name: 'anima-img2img', title: 'img2img (denoise 0.5)', description: 'LoadImage → VAEEncode → turbo KSampler denoise 0.5',
    graph: Object.assign(base(), turboNodes(), {
      11: { class_type: 'LoadImage', inputs: { image: img } },
      6: { class_type: 'VAEEncode', inputs: { pixels: ['11', 0], vae: ['8', 0] } },
      7: ks(['2', 0], ['6', 0], { denoise: 0.5 }),
      9: { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
      10: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Kiln_img2img' } },
    }),
    pos: Object.assign({}, P, { 11: [40, 700], 6: [360, 700] }),
  });
  // 5. latent hires fix
  ex.push({
    name: 'anima-hires-latent', title: 'Hires fix (latent 1.5×)', description: 'KSampler → Upscale Latent By 1.5 → KSampler denoise 0.45',
    graph: Object.assign(base(), turboNodes(), {
      6: { class_type: 'EmptySD3LatentImage', inputs: { width: 512, height: 768, batch_size: 1 } },
      7: ks(['2', 0], ['6', 0]),
      12: { class_type: 'LatentUpscaleBy', inputs: { samples: ['7', 0], upscale_method: 'nearest-exact', scale_by: 1.5 } },
      13: ks(['2', 0], ['12', 0], { denoise: 0.45 }),
      9: { class_type: 'VAEDecode', inputs: { samples: ['13', 0], vae: ['8', 0] } },
      10: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Kiln_hires' } },
    }),
    pos: Object.assign({}, P, { 12: [1040, 40], 13: [1340, 40], 9: [1680, 40], 10: [1960, 40], 8: [1340, 620] }),
  });
  // string ids everywhere
  for (const e of ex) {
    const g = {};
    for (const [id, n] of Object.entries(e.graph)) {
      const inputs = {};
      for (const [k, v] of Object.entries(n.inputs)) inputs[k] = isLink(v) ? [String(v[0]), v[1]] : v;
      g[String(id)] = Object.assign({}, n, { inputs });
    }
    e.graph = g;
  }
  return ex;
}

module.exports = { loadCatalog, fillCatalog, checkGraph, summarize, examples, isLink, SAMPLERS, SCHEDULERS };
