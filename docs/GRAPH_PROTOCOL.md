# Kiln node graphs — server ⇄ engine contract

Graphs are **ComfyUI API-format workflows**: an object keyed by node id,
`{"<id>": {"class_type": "KSampler", "inputs": {"seed": 5, "model": ["4", 0], ...}}}`.
An input is either a literal (number / string / bool) or a link `[source_node_id, output_index]`.
The node catalog (class_types, inputs, outputs, defaults) is `server/nodes.json`, in ComfyUI
`object_info` shape; the server serves it (with `$name` combo lists filled in) as `GET /api/nodes`.

## Request (server → engine, one JSON line)
```json
{"id": "g12", "cmd": "graph",
 "graph":  { ...API-format workflow, only the nodes that must run... },
 "tokens": {"<CLIPTextEncode id>": {"qwen_ids": [...], "t5_ids": [...], "t5_weights": [...]}},
 "images": {"<LoadImage id>": {"path": "C:/.../upload.rgba", "w": 832, "h": 1216}},
 "out_dir": "C:/Users/hyper/Kiln/outputs/tmp",
 "preview": true}
```
- The server tokenizes every `CLIPTextEncode` `text` (ComfyUI weight syntax, same `encodePrompt`) and
  puts the result in `tokens` keyed by node id. `text` must be a literal by the time it reaches the
  engine (the server resolves UI-only primitive nodes first).
- `LoadImage`: the server decodes the file and writes raw **RGBA8** (w*h*4 bytes); alpha becomes the
  MASK output exactly like ComfyUI (mask = 1 - alpha; no alpha -> all-zero mask).
- File-name inputs (`unet_name`, `clip_name`, `vae_name`, `lora_name`, `model_name`) are passed as the
  user picked them. The engine resolves them itself: diffusion models / text encoders / VAEs in
  `models/`, LoRAs in `models/loras/` then `models/`, upscalers in `models/upscale/`, detectors in
  `models/detect/` (a ComfyUI name like `bbox/face_yolov8m.pt` maps to `face_yolov8m.safetensors`).
- Node ids that nothing reaches from an output node (SaveImage / PreviewImage) may be omitted; the
  engine only runs what the output nodes need.

## Events (engine → server)
```
{"id","ev":"node","node":"<id>","class_type":"KSampler","status":"start"}
{"id","ev":"step","node":"<id>","step":3,"of":20,"ms":1200,"preview":{"w","h","rgb"}}    (samplers, FaceDetailer)
{"id","ev":"node","node":"<id>","class_type":"KSampler","status":"done","ms":24000}
{"id","ev":"image","node":"<id>","index":0,"kind":"save"|"preview","prefix":"Kiln","out":"<raw RGB8 path>","w":512,"h":768}
{"id","ev":"done","total_ms":31000,"node_ms":{"<id>":ms,...}}
{"id","ev":"error","node":"<id>"|null,"msg":"..."}
{"id","ev":"cancelled"}
```
- `image` events: raw **RGB8** files the server must turn into PNGs (`save` → outputs gallery with
  the prefix, like ComfyUI's SaveImage; `preview` → temporary, shown in the node/graph view only).
  One event per batch item (`index`).
- `cancel` works like for generate jobs (`{"id","cmd":"cancel"}`).
- Unsupported class_types: the engine fails fast with `{"ev":"error","node":"<id>","msg":"unsupported node type X"}`
  before running anything. The server should pre-check against `nodes.json` and show them in the UI.

## Semantics (what matches ComfyUI)
- LATENT is in VAE space like ComfyUI (VAEEncode output; KSampler applies Anima's latent
  normalization in and out). Empty latents are zeros. Batches: `batch_size` > 1 samples each item
  with its slice of one `torch.randn(seed)` tensor, like ComfyUI.
- KSampler: flow-matching Euler / DPM++ 2M / res_multistep (+ `euler_ancestral` with Kiln's own noise,
  not seed-identical to ComfyUI), ComfyUI schedulers, `denoise` < 1 = ComfyUI's truncated schedule,
  noise masks as in ComfyUI's inpaint sampler. cfg <= 1 skips the negative pass.
- MODEL values carry patches along the chain: LoRAs (LoraLoader*), shift (ModelSampling*), step cache
  (ApplyFBCacheOnModel). Default shift for Anima is 3.0.
