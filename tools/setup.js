'use strict';
// Kiln setup: gets the engine, the CUDA runtime and the models. Run it through setup.bat, which
// first makes sure Node.js is available. Safe to re-run: finished steps are skipped, interrupted
// downloads resume, and every download is checked against a pinned SHA256.
//
//   setup.bat                   a menu: everything, prerequisites only, pick parts, or list them
//   setup.bat all               everything (what the menu's default does)
//   setup.bat prereqs           only what Kiln needs to run: engine + NVIDIA cuBLAS runtime
//   setup.bat models            only the models
//   setup.bat anima vae ...     just these parts (see --list for the names)
//   setup.bat --list            each part: installed or not, its size, download link and where it goes
//                               (to download by hand and drop in place)
//   setup.bat --from <folder>   reuse model files you already have (a ComfyUI models folder, say):
//                               files with the right name and size are linked or copied, not downloaded
//     --build      build the engine from source (fetches the CUDA toolkit; needs VS 2022 C++ tools)
//     --no-extras  skip the upscaler and the face detector
//     --verify     re-hash model files that are already present
// Anything already on this PC is skipped: Node.js 20+ (setup.bat), an installed CUDA 13 toolkit's
// cuBLAS, model files in models\ or in a --from folder or a usual ComfyUI location.

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const MODELS = path.resolve(process.env.KILN_MODELS || path.join(ROOT, 'models'));
const ENGINE_DIR = path.join(ROOT, 'engine', 'build');
const ENGINE_EXE = path.join(ENGINE_DIR, 'kiln-engine.exe');
const CUDA_DIR = path.join(ROOT, 'third_party', 'cuda');
const DL_DIR = path.join(ROOT, 'third_party', 'downloads');
// the release zip already holds the engine; this is the fallback when it's missing (no login needed)
const ENGINE_URL = process.env.KILN_ENGINE_URL || 'https://github.com/sonsegajp/kiln/releases/latest/download/kiln-engine.exe';

const argv = process.argv.slice(2);
const args = new Set(argv.filter((a) => a.startsWith('--')));
const FORCE_BUILD = args.has('--build');
const EXTRAS = !args.has('--no-extras');
const VERIFY = args.has('--verify');
const FROM = [];
for (let i = 0; i < argv.length; i++) if (argv[i] === '--from' && argv[i + 1]) FROM.push(path.resolve(argv[++i]));
const WORDS = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--from').map((a) => a.toLowerCase());

// ---- pinned downloads ----------------------------------------------------------------------
const HF = (repo, rev, file) => `https://huggingface.co/${repo}/resolve/${rev}/${file}`;
const ANIMA = 'f973fc41ec7545364ac9776c2440285f43ff2a30';
const ANIMA_LORAS = '218b5466a07e8a79328dd8b73ff810706d73cb86';

const CORE_MODELS = [
  { file: 'anima-base-v1.0.safetensors', what: 'Anima base model', size: 4182218328,
    sha256: 'bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e',
    url: HF('circlestone-labs/Anima', ANIMA, 'split_files/diffusion_models/anima-base-v1.0.safetensors') },
  { file: 'qwen_3_06b_base.safetensors', what: 'Qwen3 0.6B text encoder', size: 1192135096,
    sha256: 'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba',
    url: HF('circlestone-labs/Anima', ANIMA, 'split_files/text_encoders/qwen_3_06b_base.safetensors') },
  { file: 'qwen_image_vae.safetensors', what: 'Qwen-Image VAE', size: 253806246,
    sha256: 'a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f',
    url: HF('circlestone-labs/Anima', ANIMA, 'split_files/vae/qwen_image_vae.safetensors') },
  { file: 'anima-turbo-lora-v0.2.safetensors', what: 'Anima turbo LoRA (8-step renders)', size: 148902616,
    sha256: '1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1',
    url: HF('circlestone-labs/Anima-Official-LoRAs', ANIMA_LORAS, 'anima-turbo-lora-v0.2.safetensors') },
];
const EXTRA_MODELS = [
  { file: 'upscale/4x-AnimeSharp.safetensors', what: '4x-AnimeSharp upscaler', size: 33467822,
    sha256: '7fc60054d291579acac4fb537efd91c80611cbd281ef8b90f434048cc1313b39',
    url: HF('Kim2091/AnimeSharp', '7696d95ced82b0c1f2a41f6ac73336133f0a90e1', '4x-AnimeSharp.safetensors') },
  { file: 'detect/face_yolov8m.pt', what: 'YOLOv8 face detector', size: 52026019,
    sha256: '717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f',
    url: HF('Bingsu/adetailer', '53cc19de382014514d9d4038601d261a7faa9b7b', 'face_yolov8m.pt') },
];

// CUDA 13.4 redistributables (NVIDIA's own archives; no installer needed)
const NV = 'https://developer.download.nvidia.com/compute/cuda/redist/';
const CUBLAS = { file: 'libcublas-windows-x86_64-13.8.0.4-archive.zip', size: 422415931,
  sha256: '0974318e9861a61cb9091cf7de8d4c2880e9787fe311cb4bcbd08865663c5f43', url: NV + 'libcublas/windows-x86_64/libcublas-windows-x86_64-13.8.0.4-archive.zip' };
const CUDA_TOOLKIT = [
  { file: 'cuda_nvcc-windows-x86_64-13.4.92-archive.zip', size: 32991192,
    sha256: '23950047aa5405993129a982a99924b4e6d5eb9bc312300030f3afd9b3b49ea4', url: NV + 'cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.4.92-archive.zip' },
  { file: 'cuda_cudart-windows-x86_64-13.4.92-archive.zip', size: 2737397,
    sha256: '621a70c4778287b1abf8c27c61b841d797fda2aeff01033374dd19624bcfa249', url: NV + 'cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.4.92-archive.zip' },
  { file: 'cuda_crt-windows-x86_64-13.4.92-archive.zip', size: 166601,
    sha256: 'a38e3aa75e259128a1c7fb46e706182e4babca407cd3f3c447d21ab7fb14c7fd', url: NV + 'cuda_crt/windows-x86_64/cuda_crt-windows-x86_64-13.4.92-archive.zip' },
  { file: 'libnvvm-windows-x86_64-13.4.92-archive.zip', size: 60031433,
    sha256: '64fa39ad6332d6c40a02115d64827272f6befbc86e50568254f44d37d924e10b', url: NV + 'libnvvm/windows-x86_64/libnvvm-windows-x86_64-13.4.92-archive.zip' },
  { file: 'cccl-windows-x86_64-13.3.4.3.1-archive.zip', size: 4034377,
    sha256: '17d0ca1592c0dfc974a5fc39f470a6b7bae40da239e3e19bd0752017735acbca', url: NV + 'cccl/windows-x86_64/cccl-windows-x86_64-13.3.4.3.1-archive.zip' },
  CUBLAS,
];
const CUBLAS_DLLS = ['cublas64_13.dll', 'cublasLt64_13.dll'];

// ---- helpers -------------------------------------------------------------------------------
const SYS32 = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32');
const CURL = fs.existsSync(path.join(SYS32, 'curl.exe')) ? path.join(SYS32, 'curl.exe') : 'curl';
const TAR = fs.existsSync(path.join(SYS32, 'tar.exe')) ? path.join(SYS32, 'tar.exe') : 'tar';
const CMD = process.env.ComSpec || path.join(SYS32, 'cmd.exe');

const gb = (n) => (n / 1e9).toFixed(n < 1e9 ? 2 : 1) + ' GB';
const step = (s) => console.log('\n== ' + s);
const info = (s) => console.log('   ' + s);
class SetupError extends Error {}
const fail = (s) => { throw new SetupError(s); };
const run = (exe, argv, opts = {}) => spawnSync(exe, argv, { stdio: 'inherit', windowsHide: true, ...opts });
const quiet = (exe, argv) => spawnSync(exe, argv, { encoding: 'utf8', windowsHide: true });

async function ask(q) {
  if (!process.stdin.isTTY) return '';
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try { return await new Promise(res => rl.question('   ' + q + ' ', res)); } finally { rl.close(); }
}

function sha256(file, label) {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash('sha256'), total = fs.statSync(file).size;
    let done = 0, last = 0;
    fs.createReadStream(file, { highWaterMark: 8 << 20 })
      .on('data', (b) => {
        h.update(b);
        done += b.length;
        if (total > 256e6 && Date.now() - last > 500) { last = Date.now(); process.stdout.write(`\r   verifying ${label}: ${Math.floor(done * 100 / total)}%  `); }
      })
      .on('error', reject)
      .on('end', () => { if (total > 256e6) process.stdout.write('\r' + ' '.repeat(60) + '\r'); resolve(h.digest('hex')); });
  });
}

// Download to <dest>.part (resuming a previous attempt), check size + SHA256, then rename.
async function fetchFile(item, dest) {
  const part = dest + '.part';
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(part) && fs.statSync(part).size > item.size) fs.unlinkSync(part);
  const have = fs.existsSync(part) ? fs.statSync(part).size : 0;
  if (have < item.size) {
    info(`${have ? 'resuming' : 'downloading'} ${item.what || item.file} (${gb(item.size)})`);
    for (let attempt = 1; ; attempt++) {
      const r = run(CURL, ['-fL', '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30', '-C', '-', '-o', part, item.url]);
      if (r.status === 0) break;
      // 33: the server refused the resume range; start over once
      if (r.status === 33 && attempt === 1) { fs.unlinkSync(part); continue; }
      fail(`download failed (curl exit ${r.status}) for ${item.url}\n   Re-run setup to resume.`);
    }
  }
  const size = fs.statSync(part).size;
  if (size !== item.size) { fs.unlinkSync(part); fail(`${item.file}: got ${size} bytes, expected ${item.size}. Re-run setup.`); }
  const got = await sha256(part, item.file);
  if (got !== item.sha256) { fs.unlinkSync(part); fail(`${item.file}: SHA256 mismatch (got ${got}). The file was removed; re-run setup.`); }
  fs.renameSync(part, dest);
  info('ok  ' + path.relative(ROOT, dest));
}

// Folders that may already hold the model files: --from folders, then the usual ComfyUI places.
function searchRoots() {
  const home = process.env.USERPROFILE || '';
  const guesses = [path.join(home, 'Documents', 'ComfyUI', 'models'), path.join(home, 'ComfyUI', 'models'),
    'C:\\ComfyUI\\models', 'C:\\ComfyUI_windows_portable\\ComfyUI\\models', 'D:\\ComfyUI\\models', 'D:\\ComfyUI_windows_portable\\ComfyUI\\models'];
  return [...FROM, ...guesses].filter((d, i, a) => a.indexOf(d) === i && d !== MODELS && fs.existsSync(d));
}
function findExisting(item) {
  const want = path.basename(item.file).toLowerCase();
  const walk = (dir, depth) => {
    let ents;
    try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return null; }
    for (const e of ents) {
      const p = path.join(dir, e.name);
      if (e.isFile() && e.name.toLowerCase() === want) { try { if (fs.statSync(p).size === item.size) return p; } catch { } }
    }
    if (depth <= 0) return null;
    for (const e of ents) if (e.isDirectory() && !e.name.startsWith('.')) { const f = walk(path.join(dir, e.name), depth - 1); if (f) return f; }
    return null;
  };
  for (const root of searchRoots()) { const f = walk(root, 3); if (f) return f; }
  return null;
}

async function ensureModel(item) {
  const dest = path.join(MODELS, ...item.file.split('/'));
  if (fs.existsSync(dest) && fs.statSync(dest).size === item.size) {
    if (!VERIFY) { info('have ' + item.file); return; }
    if (await sha256(dest, item.file) === item.sha256) { info('ok   ' + item.file); return; }
    info(item.file + ' failed verification; downloading it again');
    fs.unlinkSync(dest);
  }
  const found = findExisting(item);
  if (found && (await sha256(found, item.file)) === item.sha256) {
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    try { fs.linkSync(found, dest); info(`linked ${item.file} from ${found} (no download, no extra disk space)`); }
    catch { info(`copying ${item.file} from ${found}`); fs.copyFileSync(found, dest); info('ok  ' + item.file); }
    return;
  }
  if (found) info(`${found} has the right name and size but different contents; downloading instead`);
  await fetchFile(item, dest);
}

// Unzip only the members matching `patterns` (bsdtar wildcards) into `dir`.
function unzip(zip, dir, patterns = []) {
  fs.mkdirSync(dir, { recursive: true });
  const r = run(TAR, ['-xf', zip, '-C', dir, ...patterns]);
  if (r.status !== 0) fail('could not extract ' + path.basename(zip));
}
function findFile(dir, name) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { const f = findFile(p, name); if (f) return f; } else if (e.name.toLowerCase() === name.toLowerCase()) return p;
  }
  return null;
}

// ---- checks --------------------------------------------------------------------------------
function checkGpu() {
  step('GPU');
  const r = quiet('nvidia-smi', ['--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader,nounits']);
  if (r.status !== 0 || !r.stdout.trim()) {
    info('WARNING: nvidia-smi was not found. Kiln needs an NVIDIA GPU (GTX 16xx / RTX 20xx or newer) and its driver.');
    return;
  }
  const [name, mem, driver, cc] = r.stdout.trim().split('\n')[0].split(',').map(s => s.trim());
  info(`${name}, ${(Number(mem) / 1024).toFixed(1)} GB VRAM, driver ${driver}${cc ? ', compute ' + cc : ''}`);
  if (parseInt(driver, 10) < 580) info('WARNING: Kiln uses CUDA 13, which needs NVIDIA driver 580 or newer. Update your driver first.');
  if (cc && Number(cc) < 7.5) info('WARNING: this GPU is older than Turing (compute 7.5); the engine will not run on it.');
  if (Number(mem) < 5500) info('Under 6 GB of VRAM: Kiln will stream weights from system RAM, which is slower but works.');
}

function checkDisk(need) {
  try {
    const s = fs.statfsSync(ROOT), free = s.bavail * s.bsize;
    if (free < need) fail(`not enough disk space: ${gb(need)} needed, ${gb(free)} free on the Kiln drive.`);
  } catch (e) {
    if (e instanceof SetupError) throw e;
  }
}

function findVcvars() {
  const pf86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
  const vswhere = path.join(pf86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe');
  if (!fs.existsSync(vswhere)) return null;
  const r = quiet(vswhere, ['-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath']);
  const dir = r.status === 0 && r.stdout.trim().split(/\r?\n/)[0];
  const bat = dir && path.join(dir, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
  return bat && fs.existsSync(bat) ? bat : null;
}

// ---- engine --------------------------------------------------------------------------------
function looksLikeExe(f) {
  try {
    const fd = fs.openSync(f, 'r'), b = Buffer.alloc(2);
    fs.readSync(fd, b, 0, 2, 0);
    fs.closeSync(fd);
    return b.toString() === 'MZ' && fs.statSync(f).size > 1e6;
  } catch { return false; }
}

async function downloadPrebuilt() {
  const tmp = ENGINE_EXE + '.part';
  fs.mkdirSync(ENGINE_DIR, { recursive: true });
  info('downloading the prebuilt engine (latest release)');
  const sums = quiet(CURL, ['-fsSL', '--retry', '3', ENGINE_URL + '.sha256']);
  const want = sums.status === 0 && /^[0-9a-f]{64}/i.exec(sums.stdout || '');
  const r = run(CURL, ['-fL', '--retry', '3', '-o', tmp, ENGINE_URL]);
  let ok = r.status === 0 && looksLikeExe(tmp);
  if (ok && want && (await sha256(tmp, 'kiln-engine.exe')) !== want[0].toLowerCase()) {
    info('the downloaded engine failed its SHA256 check');
    ok = false;
  }
  if (ok) { fs.renameSync(tmp, ENGINE_EXE); return true; }
  try { fs.unlinkSync(tmp); } catch {}
  return false;
}

async function ensureCublasDlls() {
  if (CUBLAS_DLLS.every(d => fs.existsSync(path.join(ENGINE_DIR, d)))) { info('have the cuBLAS runtime'); return; }
  // a CUDA 13 toolkit already on this PC (Kiln's own, or NVIDIA's installer: CUDA_PATH / CUDA_PATH_V13_x)
  const roots = [CUDA_DIR, ...Object.entries(process.env).filter(([k]) => /^CUDA_PATH(_V13_\d+)?$/i.test(k)).map(([, v]) => v)];
  for (const root of roots) {
    for (const bin of [path.join(root, 'bin', 'x64'), path.join(root, 'bin')]) {
      if (CUBLAS_DLLS.every(d => fs.existsSync(path.join(bin, d)))) {
        fs.mkdirSync(ENGINE_DIR, { recursive: true });
        for (const d of CUBLAS_DLLS) fs.copyFileSync(path.join(bin, d), path.join(ENGINE_DIR, d));
        info('copied the cuBLAS runtime from ' + bin + ' (already on this PC)');
        return;
      }
    }
  }
  checkDisk(CUBLAS.size * 3);
  const zip = path.join(DL_DIR, CUBLAS.file);
  if (!fs.existsSync(zip)) await fetchFile({ ...CUBLAS, what: 'NVIDIA cuBLAS runtime' }, zip);
  const tmp = path.join(DL_DIR, 'cublas-x');
  unzip(zip, tmp, CUBLAS_DLLS.map(d => '*/bin/x64/' + d));
  for (const d of CUBLAS_DLLS) {
    const f = findFile(tmp, d);
    if (!f) fail(d + ' is missing from ' + CUBLAS.file);
    fs.copyFileSync(f, path.join(ENGINE_DIR, d));
  }
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.unlinkSync(zip);
  info('ok  cuBLAS runtime');
}

async function ensureCudaToolkit() {
  const ready = ['bin/nvcc.exe', 'include/cccl/cub', 'include/cublas_v2.h', 'lib/x64/cublas.lib', 'nvvm/bin']
    .every(p => fs.existsSync(path.join(CUDA_DIR, ...p.split('/'))));
  if (ready) { info('have the CUDA toolkit in third_party\\cuda'); return; }
  checkDisk(CUDA_TOOLKIT.reduce((a, c) => a + c.size, 0) * 3);
  for (const c of CUDA_TOOLKIT) {
    const zip = path.join(DL_DIR, c.file);
    if (!fs.existsSync(zip)) await fetchFile({ ...c, what: c.file.replace(/-windows.*/, '') }, zip);
    const tmp = path.join(DL_DIR, 'x');
    fs.rmSync(tmp, { recursive: true, force: true });
    unzip(zip, tmp);
    for (const top of fs.readdirSync(tmp)) fs.cpSync(path.join(tmp, top), CUDA_DIR, { recursive: true, force: true });
    fs.rmSync(tmp, { recursive: true, force: true });
    fs.unlinkSync(zip);
  }
  info('ok  CUDA toolkit in third_party\\cuda');
}

async function buildEngine() {
  let vcvars = findVcvars();
  if (!vcvars) {
    info('Building the engine needs the Visual Studio 2022 C++ build tools, which were not found.');
    const winget = quiet('winget', ['--version']).status === 0;
    if (winget && /^y/i.test(await ask('Install "Visual Studio 2022 Build Tools" (C++, about 3 GB) with winget now? [y/N]'))) {
      const r = run('winget', ['install', '--id', 'Microsoft.VisualStudio.2022.BuildTools', '-e', '--accept-package-agreements', '--accept-source-agreements',
        '--override', '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended']);
      if (r.status !== 0) fail('the Build Tools install did not finish (winget exit ' + r.status + ').');
      vcvars = findVcvars();
    }
    if (!vcvars) fail('install "Build Tools for Visual Studio 2022" with the "Desktop development with C++" workload\n   (https://visualstudio.microsoft.com/downloads/), then run setup.bat --build again.');
  }
  await ensureCudaToolkit();
  info('building the engine (a few minutes)');
  const r = run(CMD, ['/d', '/c', path.join(ROOT, 'engine', 'build.bat')], { cwd: path.join(ROOT, 'engine') });
  if (r.status !== 0 || !fs.existsSync(ENGINE_EXE)) fail('the engine build failed; see the compiler output above.');
}

async function ensureEngine() {
  step('Engine');
  if (!FORCE_BUILD && looksLikeExe(ENGINE_EXE)) { info('have engine\\build\\kiln-engine.exe'); return; }
  if (FORCE_BUILD) {
    await buildEngine();
  } else if (looksLikeExe(ENGINE_EXE)) {
    info('have engine\\build\\kiln-engine.exe');
  } else if (!(await downloadPrebuilt())) {
    info('No prebuilt engine is available; building it from source instead.');
    await buildEngine();
  }
}

// ---- parts ---------------------------------------------------------------------------------
// Everything setup can get, by name. prereq: Kiln can't run without it; model: a model file.
const MODEL = (m, extra) => ({ what: m.what, size: m.size, group: extra ? 'extra' : 'model', item: m,
  have: () => { const f = path.join(MODELS, ...m.file.split('/')); return fs.existsSync(f) && fs.statSync(f).size === m.size; },
  run: async () => { step(m.what); await ensureModel(m); if (m.file === 'detect/face_yolov8m.pt') faceConvert(); } });
const PARTS = {
  engine: { what: 'Kiln engine (prebuilt; --build to compile it)', size: 7e6, group: 'prereq', have: () => looksLikeExe(ENGINE_EXE), run: ensureEngine },
  cublas: { what: 'NVIDIA cuBLAS runtime', size: CUBLAS.size, group: 'prereq', have: () => CUBLAS_DLLS.every(d => fs.existsSync(path.join(ENGINE_DIR, d))),
    run: async () => { step('NVIDIA cuBLAS runtime'); await ensureCublasDlls(); } },
  anima: MODEL(CORE_MODELS[0]),
  te: MODEL(CORE_MODELS[1]),
  vae: MODEL(CORE_MODELS[2]),
  turbo: MODEL(CORE_MODELS[3]),
  upscaler: MODEL(EXTRA_MODELS[0], true),
  face: MODEL(EXTRA_MODELS[1], true),
};
const ALIASES = { model: 'anima', dit: 'anima', diffusion: 'anima', 'text-encoder': 'te', encoder: 'te', qwen: 'te', lora: 'turbo', upscale: 'upscaler', detector: 'face', cuda: 'cublas' };

function faceConvert() {
  const pt = path.join(MODELS, 'detect', 'face_yolov8m.pt');
  if (fs.existsSync(pt) && !['.safetensors', '.json'].every(x => fs.existsSync(pt.replace(/\.pt$/, x)))) {
    info('converting the face detector for the engine');
    require('./convert_yolo.js').convert(pt);
  }
}

function pickFromWords(words) {
  const out = new Set();
  for (const w0 of words) {
    const w = ALIASES[w0] || w0;
    if (w === 'all' || w === 'everything') for (const [k, p] of Object.entries(PARTS)) { if (p.group !== 'extra' || EXTRAS) out.add(k); }
    else if (w === 'prereqs' || w === 'prerequisites') { out.add('engine'); out.add('cublas'); }
    else if (w === 'models') for (const [k, p] of Object.entries(PARTS)) { if (p.group === 'model' || (p.group === 'extra' && EXTRAS)) out.add(k); }
    else if (PARTS[w]) out.add(w);
    else fail(`unknown part "${w0}". Parts: ${Object.keys(PARTS).join(', ')}; or all, prereqs, models.`);
  }
  return Object.keys(PARTS).filter((k) => out.has(k));
}

function listParts() {
  console.log('\nKiln parts (setup.bat <name> gets one; drop a downloaded file in the folder shown to skip the download):\n');
  for (const [k, p] of Object.entries(PARTS)) {
    const where = p.item ? path.relative(ROOT, path.join(MODELS, ...p.item.file.split('/'))) : k === 'engine' ? 'engine\\build\\kiln-engine.exe' : 'engine\\build\\cublas64_13.dll + cublasLt64_13.dll';
    console.log(`  ${k.padEnd(9)} ${p.have() ? '[installed]' : '[missing]  '} ${p.what} (${gb(p.size)})`);
    console.log(`            -> ${where}`);
    if (p.item) console.log(`            ${p.item.url}`);
    else if (k === 'cublas') console.log(`            ${CUBLAS.url}  (zip; the two DLLs are in bin\\x64)`);
  }
  console.log('\n  Shortcuts: all, prereqs (engine + cublas), models. Node.js is handled by setup.bat itself.');
}

async function menu() {
  const missing = Object.entries(PARTS).filter(([, p]) => !p.have());
  console.log('\nWhat should setup get?');
  console.log('  1  Everything that\'s missing (recommended)' + (missing.length ? `: ${missing.map(([k]) => k).join(', ')}` : ': nothing, all installed'));
  console.log('  2  Only the prerequisites (engine + NVIDIA cuBLAS), I\'ll bring the models myself');
  console.log('  3  Pick parts');
  console.log('  4  List the parts with their download links');
  const a = (await ask('Choice [1]:')).trim() || '1';
  if (a === '2') return pickFromWords(['prereqs']);
  if (a === '3') {
    const keys = Object.keys(PARTS);
    keys.forEach((k, i) => console.log(`     ${String(i + 1).padStart(2)}  ${k.padEnd(9)} ${PARTS[k].have() ? '[installed]' : '[missing]  '} ${PARTS[k].what} (${gb(PARTS[k].size)})`));
    const pick = (await ask('Numbers or names, separated by spaces:')).trim().split(/[\s,]+/).filter(Boolean);
    return pickFromWords(pick.map((x) => (/^\d+$/.test(x) && keys[Number(x) - 1]) || x));
  }
  if (a === '4') { listParts(); return []; }
  return pickFromWords(['all']);
}

async function main() {
  console.log('Kiln setup');
  if (args.has('--list')) { listParts(); return; }
  const chosen = WORDS.length ? pickFromWords(WORDS) : process.stdin.isTTY ? await menu() : pickFromWords(['all']);
  if (!chosen.length) return;
  checkGpu();
  if (FROM.length) info('reusing model files from: ' + FROM.join(', '));
  const need = chosen.filter((k) => PARTS[k].group !== 'prereq' && !PARTS[k].have()).reduce((a, k) => a + PARTS[k].size, 0);
  if (need) checkDisk(need + 256e6);
  for (const k of chosen) await PARTS[k].run();
  for (const d of ['loras', 'checkpoints', 'upscale', 'detect']) fs.mkdirSync(path.join(MODELS, d), { recursive: true });
  fs.mkdirSync(path.join(ROOT, 'outputs'), { recursive: true });
  try { fs.rmdirSync(DL_DIR); } catch {}
  const still = Object.entries(PARTS).filter(([, p]) => p.group !== 'extra' && !p.have()).map(([k]) => k);
  if (still.length) console.log(`\nDone. Still missing before Kiln can render: ${still.join(', ')} (setup.bat --list shows where they go).`);
  else console.log('\nAll set. Run start.bat and open http://localhost:8090/');
}

main().catch((e) => {
  console.error('\nSetup stopped: ' + (e instanceof SetupError ? e.message : e.stack || e));
  process.exit(1);
});
