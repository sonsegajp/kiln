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
| Kiln + hires fix + face detailer | ~40 s |

## Features

- **Simple mode**: an A1111-style layout (txt2img, img2img with inpainting, Extras, PNG Info), with live
  step previews and per-stage progress for sampling, hires fix, face refining and upscaling.
- **Nodes mode**: a node editor that runs graphs natively on the engine and imports ComfyUI workflows
  (API or UI JSON, or a PNG) built from the nodes Kiln implements.
- **Packs**: Kiln's own extension API (`docs/NODE_API.md`). JavaScript nodes run in the server and call
  the engine for GPU work. Install packs from git or a folder, or scaffold a new one in the Extensions
  panel. Three example packs are bundled.
- **CivitAI browser**: search and download checkpoints and LoRAs (API key supported). Downloads are
  resumable and hash-verified.
- **LoRA slots** plus `<lora:name:weight>` prompt syntax, and A1111/ComfyUI prompt weighting.
- **Negative prompts at CFG 1** through NAG (Normalized Attention Guidance).
- **Built-in post-processing**: hires fix, a YOLOv8 face detailer and a 4x RRDB upscaler (4x-AnimeSharp).
- **Samplers and schedulers**: euler, euler ancestral, DPM++ 2M and res multistep, with every
  ComfyUI scheduler. The first-block step cache is optional.
- **Low-VRAM friendly**: weights that don't fit in VRAM stream from system RAM, VAE decodes are
  banded, and the scratch memory regrows automatically.

**Models**: Anima (Cosmos-Predict2 2B with the Qwen3-0.6B text encoder and the Qwen-Image VAE).
SDXL (Illustrious, NoobAI, Pony) already renders in the engine but is turned off in the UI until it's
fast enough on low-end cards.

## Requirements

- Windows 10 (1803 or newer) or Windows 11.
- An NVIDIA GPU, Turing (GTX 16xx / RTX 20xx) or newer, with driver 580 or newer (CUDA 13).
  6 GB of VRAM recommended; smaller cards work with weights streamed from system RAM.
- About 7 GB of free disk space.

## Setup

1. Run **`setup.bat`**. It downloads everything Kiln needs into its own folder:
   - a portable Node.js, only if Node.js 20 or newer isn't installed
   - the engine: the prebuilt `kiln-engine.exe` from the latest release, plus NVIDIA's cuBLAS runtime
   - the models, into `models\`: the Anima base model, the Qwen3 0.6B text encoder and the Qwen-Image VAE
     from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima), and the turbo LoRA
     (8-step renders) from [circlestone-labs/Anima-Official-LoRAs](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs)
   - extras: the [4x-AnimeSharp](https://huggingface.co/Kim2091/AnimeSharp) upscaler and a YOLOv8 face detector
     from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer), converted for the engine. Skip them with
     `setup.bat --no-extras`.

   Every download is pinned to a SHA256 and resumes if it's interrupted. Re-running `setup.bat` skips
   whatever is already there; `setup.bat --verify` also re-checks the model files.
2. Run **`start.bat`** and open http://localhost:8090/. The server supervises the engine and prints
   LAN URLs for phones and tablets.

**Building the engine yourself**: `setup.bat --build` unpacks NVIDIA's CUDA 13.4 toolkit redistributables
into `third_party\cuda` and runs `engine\build.bat`. It needs the Visual Studio 2022 C++ build tools, and
offers to install them with winget if they're missing.

## Layout

| Path | What |
|---|---|
| `engine/src` | the C++/CUDA engine: models, kernels, samplers, graph executor (JSON over stdin/stdout) |
| `server` | zero-dependency Node.js server: queue, tokenizers, PNG metadata, CivitAI, packs |
| `web` | the UI (Simple and Nodes modes) |
| `extensions` | bundled example packs |
| `docs` | engine graph protocol and the pack API |
| `tools` | setup and model conversion scripts |
| `bench` | kernel benchmarks and tests |

The tokenizer vocabularies in `server/tokenizers` come from Qwen2.5 (Apache 2.0), T5 (Apache 2.0) and
OpenAI CLIP (MIT).
