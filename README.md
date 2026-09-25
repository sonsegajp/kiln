<p align="center"><img src="docs/logo.svg" alt="Kiln" width="300"></p>

<p align="center"><b>A fast local image generator for low-end GPUs.</b><br>
Its own C++/CUDA inference engine and web UI, built from scratch for cards like the GTX 16xx.</p>

---

Kiln runs diffusion models on its own engine instead of PyTorch. The kernels are written for GPUs
without tensor cores: fp16x2 (HFMA2) GEMMs and implicit-GEMM convolutions, a fused flash attention,
and weights kept in their checkpoint precision. On a laptop GTX 1660 Ti Max-Q (6 GB, power-capped),
it renders the same seeds several times faster than PyTorch-based UIs.

| Anima, 512x768, turbo LoRA, 8 steps | Time |
|---|---|
| ComfyUI (same seed, same card) | ~99 s |
| Kiln | ~12 s |
| Kiln + face detailer + hires fix | ~40 s |

## Features

- **Simple mode**: an A1111-style layout (txt2img, img2img with inpainting, Extras, PNG Info), with live
  step previews and per-stage progress for sampling, face refining, hires fix and upscaling.
- **Nodes mode**: a node editor that runs graphs natively on the engine and imports ComfyUI workflows
  (API or UI JSON, or a PNG) built from the nodes Kiln implements.
- **Packs**: Kiln's own extension API (`docs/NODE_API.md`). JavaScript nodes run in the server and call
  the engine for GPU work. Install packs from git or a folder, or scaffold a new one in the Extensions
  panel. Three example packs are bundled.
- **CivitAI browser**: search and download checkpoints and LoRAs (API key supported). Downloads are
  resumable and hash-verified.
- **LoRA slots** plus `<lora:name:weight>` prompt syntax, and A1111/ComfyUI prompt weighting.
- **Negative prompts at CFG 1** through NAG (Normalized Attention Guidance).
- **Built-in post-processing**: a YOLOv8 face detailer, hires fix and a 4x RRDB upscaler (4x-AnimeSharp).
- **Samplers and schedulers**: euler, euler ancestral, DPM++ 2M and res multistep, with every
  ComfyUI scheduler. The first-block step cache is optional.
- **Low-VRAM friendly**: weights that don't fit in VRAM stream from system RAM, VAE decodes are
  banded, and the scratch memory regrows automatically.

**Models**: Anima (Cosmos-Predict2 2B with the Qwen3-0.6B text encoder and the Qwen-Image VAE).
SDXL (Illustrious, NoobAI, Pony) already renders in the engine but is turned off in the UI until it's
fast enough on low-end cards.

## Requirements

- Windows 10 (version 1803 or newer) or Windows 11.
- An NVIDIA GPU, Turing (GTX 16xx / RTX 20xx) or newer, with driver **580 or newer** (Kiln uses CUDA 13).
  6 GB of VRAM is recommended; smaller cards work, with weights streamed from system RAM.
- About 7 GB of free disk space for the engine and models, plus room for your renders.

Nothing else needs to be installed beforehand: `setup.bat` fetches Node.js if you don't have it.

## Install

1. **Get the code.** Either clone it:

   ```
   git clone https://github.com/drewb583/kiln.git
   ```

   or, while the repository is private, use the GitHub CLI (`gh auth login` once, then
   `gh repo clone drewb583/kiln`). You can also download the ZIP from the repository page and unpack it.
   Any folder works; paths with spaces are fine.

2. **Run `setup.bat`** (double-click it, or run it from a terminal). It downloads, into the Kiln folder:

   | What | From | Size |
   |---|---|---|
   | Node.js 24 (portable), only if Node.js 20+ isn't installed | nodejs.org | 35 MB |
   | The engine, `kiln-engine.exe` | this repository's latest release | 7 MB |
   | NVIDIA cuBLAS runtime DLLs | NVIDIA's CUDA redistributables | 420 MB download |
   | Anima base model, Qwen3 0.6B text encoder, Qwen-Image VAE | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | 5.6 GB |
   | Anima turbo LoRA (8-step renders) | [circlestone-labs/Anima-Official-LoRAs](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs) | 150 MB |
   | 4x-AnimeSharp upscaler | [Kim2091/AnimeSharp](https://huggingface.co/Kim2091/AnimeSharp) | 33 MB |
   | YOLOv8 face detector (converted for the engine on your PC) | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) | 52 MB |

   Every download is pinned to a SHA256 and resumes if it's interrupted, so if anything fails, just run
   `setup.bat` again. Finished steps are skipped. Options:

   | Option | Does |
   |---|---|
   | `setup.bat --no-extras` | skips the upscaler and the face detector |
   | `setup.bat --verify` | re-checks the hashes of model files that are already there |
   | `setup.bat --build` | builds the engine from source instead of downloading it (see below) |

   While the repository is private, the engine download needs the GitHub CLI signed in to an account
   with access (`gh auth login`). Without it, setup builds the engine from source instead.

3. **Run `start.bat`** and open **http://localhost:8090/**. The console window is the server; close it
   to stop Kiln. The first render after starting takes longer while the models load.

## Using Kiln

### Your first image

1. In **txt2img**, click the **Turbo** preset next to the prompt box (turbo LoRA at 1.0, 8 steps, CFG 1,
   Euler). **Base** is the slower, full-quality alternative: 20 steps, CFG 4.5, DPM++ 2M.
2. Type a prompt. Anima works well with Danbooru-style tags, natural language, or both. A negative prompt
   still applies at CFG 1 through NAG.
3. Pick a size (512x768 is a good start on 6 GB) and press **Generate**. The bars show each stage's
   progress, with a live preview while it samples.

Renders are saved as PNGs in `outputs\<date>\`, with the generation settings embedded. Drop one on
**PNG Info** to read them back, or send them on to img2img, Extras or Nodes.

### Better results

- **Face Detailer** finds faces with the YOLOv8 detector and re-renders each one at higher resolution.
  It runs before hires fix, so the hires pass refines the new faces along with the rest of the image.
- **Hires. fix** renders small and then refines at a larger size. The **4x-AnimeSharp (model)** upscaler
  gives sharper lines than Lanczos.
- **Extras** upscales any image with 4x-AnimeSharp.
- **img2img** reworks an existing image. Its **Inpaint** tab repaints only the area you mask.

### LoRAs

Put LoRA `.safetensors` files in `models\loras\`, or download them with the CivitAI browser. Enable
them in the LoRA slots under the prompt, or write `<lora:name:0.8>` in the prompt. Anima LoRAs only
work with Anima checkpoints.

### CivitAI

Open **Settings → CivitAI** and paste an API key (civitai.com → Account settings → API keys). Some
models can only be downloaded with a key. The key is stored on your PC in `config\civitai.json`. It is
never sent to the browser, and it can only be changed from the PC running Kiln.

Browse with the 🌐 button or the **CivitAI** tab next to Lora and Checkpoints. Results are filtered to
the base model of the checkpoint you have selected (Anima, Illustrious, and so on). Downloads land in
`models\loras\` or `models\checkpoints\` with their preview images, and are hash-checked.

### Nodes and packs

**Nodes** is a node editor for building your own pipelines. Drop in a ComfyUI workflow (API or UI JSON,
or a PNG that carries one) to import it. The nodes Kiln implements run natively on the engine.

The **Extensions** button manages packs: installing them from a git URL or a folder, enabling or
disabling them, and scaffolding new ones. A pack is JavaScript that runs inside the Kiln server, so only
install packs you trust. To write one, see [`docs/NODE_API.md`](docs/NODE_API.md).

### Phones and tablets

The server prints LAN addresses when it starts (for example `http://192.168.1.20:8090/`). Open one on
any device on the same network to generate from there. Settings and extensions can only be changed on
the PC itself. If Windows asks whether Node.js may use the network, allow private networks.

## Folders

| Folder | Holds |
|---|---|
| `models\` | the Anima base model, text encoder, VAE and turbo LoRA |
| `models\loras\` | LoRAs |
| `models\checkpoints\` | other checkpoints (SDXL checkpoints are listed but can't run yet) |
| `models\upscale\` | upscaler models |
| `models\detect\` | the face detector |
| `outputs\` | your renders, one folder per day |
| `inputs\` | images uploaded for img2img, inpainting and Extras |
| `workflows\` | node workflows you've saved |
| `config\` | settings and your CivitAI key |
| `extensions\` | installed packs |

None of these are tracked by git except the bundled example packs, so updating never touches your
models, renders or key.

## Settings

The server reads these environment variables. Set them in a terminal before running `start.bat`, e.g.
`set KILN_PORT=8091`.

| Variable | Default | What |
|---|---|---|
| `KILN_PORT` | `8090` | web server port |
| `KILN_HOST` | `0.0.0.0` | listen address; `127.0.0.1` keeps Kiln off your network |
| `KILN_MODELS` | `models` | models folder (`setup.bat` downloads there too) |
| `KILN_OUTPUTS` | `outputs` | where renders are saved |
| `KILN_INPUTS` | `inputs` | uploaded images |
| `KILN_WORKFLOWS` | `workflows` | saved node workflows |
| `KILN_CONFIG` | `config` | settings and the CivitAI key |
| `KILN_EXTENSIONS` | `extensions` | packs folder |
| `KILN_EXT_LAN` | off | `1` allows managing packs from other devices |
| `KILN_ENGINE` | `engine\build\kiln-engine.exe` | engine executable |
| `KILN_SDXL` | off | `1` enables the unfinished SDXL path (slow) |

## Updating

```
git pull
setup.bat
```

`setup.bat` fetches anything new that the update needs. To get a newer engine as well, delete
`engine\build\kiln-engine.exe` before running it, or use `setup.bat --build`.

## Troubleshooting

- **The engine exits right away or reports a CUDA error.** Update the NVIDIA driver to 580 or newer.
  `setup.bat` prints your GPU and driver at the top.
- **Out of memory.** Close other programs that use the GPU (games, browsers with hardware
  acceleration, other AI tools), render smaller, or turn off hires fix. Weights that don't fit in VRAM
  stream from system RAM automatically, which is slower.
- **"Port 8090 is in use."** Another program, or another Kiln, is using the port. Close it or set
  `KILN_PORT`.
- **A download failed or was interrupted.** Run `setup.bat` again; it resumes.
- **Something looks corrupted.** Run `setup.bat --verify`. Files that fail the check are downloaded again.
- **The face detailer or the model upscaler is unavailable.** Its model is missing; run `setup.bat`
  (without `--no-extras`). The note under the option says what's missing.

## Building the engine from source

`setup.bat --build` does everything. It unpacks NVIDIA's CUDA 13.4 toolkit redistributables (nvcc,
cudart, CCCL, cuBLAS) into `third_party\cuda`, then runs `engine\build.bat`. That needs the Visual
Studio 2022 C++ build tools; if they're missing, setup offers to install them with winget, or you can
install "Build Tools for Visual Studio 2022" with the "Desktop development with C++" workload yourself.

Once the toolkit is in place, `engine\build.bat` rebuilds on its own. It compiles for sm_75 with PTX, so
the same binary runs on Turing and every newer NVIDIA GPU.

## Layout

| Path | What |
|---|---|
| `engine/src` | the C++/CUDA engine: models, kernels, samplers, graph executor (JSON over stdin/stdout) |
| `server` | zero-dependency Node.js server: queue, tokenizers, PNG metadata, CivitAI, packs |
| `web` | the UI (Simple and Nodes modes) |
| `extensions` | bundled example packs |
| `tools` | `setup.js` (used by `setup.bat`) and the face detector converter |
| `docs` | engine graph protocol and the pack API |
| `bench` | kernel benchmarks and tests |

## Credits

Kiln doesn't include any model weights. `setup.bat` downloads them from their authors' repositories,
and they remain under their own licenses: Anima and its LoRAs by circlestone-labs, 4x-AnimeSharp by
Kim2091, and the face detector by Bingsu (trained with Ultralytics YOLOv8).

The tokenizer vocabularies in `server/tokenizers` come from Qwen2.5 (Apache 2.0), T5 (Apache 2.0) and
OpenAI CLIP (MIT).
