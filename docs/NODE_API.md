# Kiln node API

Kiln is its own program: extensions are Kiln packs, written against this API. A ComfyUI node is
brought over by porting it (its `INPUT_TYPES` maps 1:1 onto `inputs` below; its Python body becomes
`run`). Built-in nodes stay native C++ in the engine; pack nodes are JavaScript running in Kiln's
server, and call back into the engine for GPU work.

## Pack layout

```
Kiln/extensions/<pack>/
  kiln.json    {"name", "version", "description", "author", "url", "nodes": "nodes.js", "ui": "ui.js"}
  nodes.js     CommonJS module (below)
  ui.js        optional: custom widgets for Kiln's editor
```

```js
module.exports = {
  nodes: {
    "ImageBrightness": {
      display_name: "Image Brightness",
      category: "image/adjust",
      description: "Scales pixel values.",
      inputs: {
        required: { image: ["IMAGE"], amount: ["FLOAT", { default: 1.0, min: 0, max: 4, step: 0.05 }] },
        optional: {},
      },
      outputs: ["IMAGE"],
      output_names: ["IMAGE"],
      output_node: false,
      async run({ image, amount }, ctx) {
        const out = ctx.image(image.batch, image.height, image.width);
        for (let i = 0; i < out.data.length; i++) out.data[i] = Math.min(1, image.data[i] * amount);
        return [out];
      },
    },
  },
};
```

Input specs use the same shapes as Kiln's catalog (`["INT", {default,min,max,step}]`, `[["a","b"], {default}]`,
`["STRING", {multiline}]`, `["IMAGE"]`, ...). A combo of the user's files is written as a list name, filled in by
the server: `["$upscale_models"]`, `$loras`, `$diffusion_models`, `$text_encoders`, `$vaes`, `$checkpoints`,
`$bbox_models`, `$images`, `$samplers`, `$schedulers`. Keep ComfyUI's class name and input names when porting a
node, so ComfyUI workflows that use it load and run unchanged.

## Values

| Type | In JS |
|---|---|
| INT, FLOAT, BOOLEAN, STRING, combo | JS primitives |
| IMAGE | `{type:"IMAGE", batch, height, width, data: Float32Array}` — interleaved `[B,H,W,3]`, values 0..1 |
| MASK | `{type:"MASK", batch, height, width, data: Float32Array}` — `[B,H,W]`, 0..1 |
| LATENT | `{type:"LATENT", batch, channels, height, width, data: Float32Array}` — `[B,C,h,w]`, raw VAE space |
| MODEL, CLIP, VAE, CONDITIONING, UPSCALE_MODEL, BBOX_DETECTOR | opaque handles `{type, handle}`: pass them on, or hand them to `ctx.ops` |

`ctx`: `image(b,h,w)`, `mask(b,h,w)`, `latent(b,c,h,w)` (zero-filled), `progress(done, total)` (drives the node's
progress bar), `log(msg)` (server log, prefixed `[pack] #node`), `cancelled()`, `node` (id), `class_type`, `pack`, and
`ops` — engine operations that run on the GPU and resolve to values:

| op | args | result |
|---|---|---|
| `ops.vaeDecode(latent, vae)` | | IMAGE |
| `ops.vaeEncode(image, vae)` | | LATENT |
| `ops.resize(image, width, height, method)` | method: nearest-exact, bilinear (default), bicubic, area, lanczos | IMAGE |
| `ops.upscale(image, upscaleModel)` | | IMAGE |
| `ops.detectFaces(image, threshold, detector?)` | threshold default 0.5; optional BBOX_DETECTOR handle | `[{x0,y0,x1,y1,score}]` per batch item (pixels) |
| `ops.encodeText(clip, text)` | text is tokenized by the server (weights like `(word:1.2)` work) | CONDITIONING |
| `ops.sample(model, positive, negative, latent, {seed, steps, cfg, sampler, scheduler, denoise, mask})` | defaults: seed 0, steps 20, cfg 4.5, euler, simple, denoise 1; mask = MASK | LATENT |

`run` may be `async`. It returns an array with one value per output (a single-output node may return the value
itself, unless that value is an array), or `{ outputs: [...], ui: { text: "..." } }` to also show text under the node in the editor. Throwing
fails the job at that node with the error message; the stack goes to the server log. Inputs are read-only: build
results with `ctx.image()` etc. Options on a node definition: `output_node: true` (a graph root, like SaveImage) and
`fold: false` (never run at queue time, see below). An input's options may name a custom editor widget with
`widget: "<name>"` (see UI).

Types: besides the built-in ones, a pack can use its own UPPER_CASE types between its nodes (e.g. `MY_BOXES`); those
values are plain JS objects that never leave the server. `"*"` accepts / returns any type.

### Constant folding

A pack node whose inputs are all constants (widget values, or outputs of other such nodes) and whose outputs are
not engine types (IMAGE, MASK, LATENT, MODEL, CLIP, VAE, CONDITIONING, ...) runs on the server when the graph is
queued. Its primitive results replace the links into the rest of the graph, so a Seed can feed several samplers and
a text node can feed CLIPTextEncode (whose text is tokenized by the server). Folded nodes never reach the engine.
Errors at this stage reject the queue request with the node marked. Set `fold: false` for nodes that must run at
execution time. `ctx.ops` is not available while folding, and a folded `run()` must finish within 30 s (calls
during the job have no time limit; they end when the job is cancelled).

### Pack management

`GET /api/extensions` lists packs (`dir, name, version, description, author, url, enabled, loaded, error,
node_errors, nodes, ui, git, path`). `POST /api/extensions/reload | install {source} | create {name, author,
description}`, `POST /api/extensions/<dir>/enable | disable | update` (git pull), `DELETE /api/extensions/<dir>`.
A disabled pack has a `.disabled` file in its folder. Install takes a git URL (`git clone --depth 1`, the repo
needs `kiln.json` at its root) or an absolute local folder (copied). These change what code the server runs, so
they are only accepted from the PC running Kiln (set `KILN_EXT_LAN=1` to allow other devices), as JSON, and not
from other origins. Pack code runs inside the server with full access to the PC: install only packs you trust.
Reload re-runs `nodes.js`, so keep module-level side effects (timers, servers) out of it.

## Engine protocol (server ⇄ engine)

A `graph` request carries the pack nodes it uses: `"ext": {"<class_type>": {"outputs": ["IMAGE", ...]}}`.
The engine's graph executor evaluates the graph as usual; at a pack node it emits

```
{"id", "ev":"ext_call", "call": <n>, "node": "<id>", "class_type": "...", "inputs": {name: value}}
```

and waits. Values on the wire: primitives as JSON; IMAGE/MASK/LATENT as
`{"type", "path", "shape"}` (raw float32 file in the request's `out_dir`, layouts as above); handles as
`{"type", "handle"}`. The server runs `run()` and answers on stdin:

```
{"cmd":"ext_result", "id", "call": <n>, "outputs": [value, ...]}      or   {..., "error": "msg"}
```

While a call is open the server may send `{"cmd":"ext_op", "id", "call", "op_id", "op", ...args}`; the engine
runs it on its worker and emits `{"id", "ev":"ext_op_done", "op_id", "result"}` (or `"error"`). Cancelling the
job answers every open call with an error.

Details the engine must match (the mock engine, `server/mock-engine.js`, implements all of it):

- The engine reports `"features": {"ext": true}` in its `info` reply. Without it the server refuses graphs that need
  pack nodes at execution time (constant pack nodes still work, they never reach the engine).
- `ext_call`: `call` is unique within the job; `inputs` has every connected / literal input of the node. Emit the
  usual `node` start/done events around it. Tensor files are little-endian float32, written into `out_dir` and named
  `<job id>_….f32`; the server deletes all of them when the job ends (the engine may delete its own earlier).
- `ext_result` outputs follow `ext[class_type].outputs`. A `"*"` output takes its type from the value: a
  `{"type","path","shape"}` tensor, a `{"type","handle"}` handle, or a JSON primitive.
- Handles: the server passes engine handles back unchanged. A handle whose id starts with `js:` is a pack-only value
  held by the server; the engine stores it like any handle and hands it to downstream pack nodes as-is.
- Primitive outputs (INT, FLOAT, STRING, BOOLEAN) may feed linked widget inputs of built-in nodes (e.g. an INT into
  ImageScale.width), so resolve those links to the values.
- `error` in `ext_result` fails the job with `{"ev":"error", "node": "<pack node id>", "msg": <that message>}`.
- On `cancel` the server also answers every open `ext_call` with `"error":"cancelled"`; finish with `cancelled`.
- `ext_op` messages (all tensors / handles in the wire format above; results likewise):

| op | args | result |
|---|---|---|
| `vae_decode` | `latent` (LATENT), `vae` (VAE handle) | IMAGE |
| `vae_encode` | `image` (IMAGE), `vae` | LATENT |
| `resize` | `image`, `width`, `height`, `method` (nearest-exact, bilinear, bicubic, area, lanczos) | IMAGE |
| `upscale` | `image`, `model` (UPSCALE_MODEL handle) | IMAGE |
| `detect_faces` | `image`, `threshold`, optional `detector` (BBOX_DETECTOR handle) | `[[{x0,y0,x1,y1,score}], ...]` per batch item |
| `encode_text` | `clip` (CLIP handle), `text`, `qwen_ids`, `t5_ids`, `t5_weights` (tokenized by the server) | CONDITIONING handle |
| `sample` | `model`, `positive`, `negative` (handles), `latent`, `seed`, `steps`, `cfg`, `sampler`, `scheduler`, `denoise`, optional `mask` (MASK) | LATENT |

  While `sample` runs, emit `{"id","ev":"step","node":"<the calling pack node>","step","of","ms"}` (previews optional).

## UI (editor widgets)

A pack's optional `ui.js` (named by `"ui"` in kiln.json) is loaded by the node editor as an ES module. It default-
exports a setup function (or exports `setup`) that receives the `kiln` object:

```js
export default function setup(kiln) {
  kiln.css(`.my-swatch { width: 100%; height: 26px; }`);          // a <style> tag, removed on reload
  kiln.registerWidget('my-pack.color', {                           // used by inputs with { widget: 'my-pack.color' }
    create({ value, set, input, node }) {
      const el = document.createElement('input');
      el.type = 'color';
      el.className = 'my-swatch';
      el.value = value;
      el.addEventListener('input', () => set(el.value));
      return el;                                                   // any DOM element
    },
  });
}
```

| `kiln.` | |
|---|---|
| `registerWidget(name, impl)` | a widget any input can pick with the input option `widget: "<name>"`. Names are global: prefix them with your pack name. |
| `registerNodeWidget(nodeType, inputName, impl)` | replace the widget of one input of one node type (wins over `widget:`). |
| `css(text)` | add a stylesheet; `url(path)` → URL of a file in your pack folder (images, fonts, more modules). |
| `toast(msg, kind)`, `api(path, opts)` | a notification (`kind`: `''`, `'ok'`, `'err'`); a JSON call to Kiln's HTTP API. |
| `version`, `pack` | API version (1), your pack's name. |

`impl.create(ctx)` builds the widget and returns an element; `ctx` = `{ value, set(v), input: {name, type, options,
values}, node: {id, type, title} }` (`type` is INT, FLOAT, STRING, BOOLEAN or COMBO; `values` is the combo list). Call
`set(v)` whenever the value changes: it is stored in the graph (undo, autosave, workflows) and sent to `run()` as a
plain value, so the node still works without the UI. With `impl.fullWidth: true` the widget spans the whole row
(no label). The widget handles its own pointer / keyboard events (the canvas doesn't pan or drag from inside it).
If `create` throws, the editor falls back to its default widget; ui.js errors show as a toast and never break the
editor. Reloading packs re-imports ui.js and re-renders the nodes that use its widgets.

Reference packs: `extensions/kiln-utils` (Seed, Concatenate, Text Join, Int/Float Math, Switch), `extensions/kiln-image`
(Image Adjust with a slider widget from its ui.js; ports of ComfyUI's ImageFlip, ImageRotate, ImageCrop) and
`extensions/kiln-upscale-to-size` (ctx.ops.upscale + ctx.ops.resize).
