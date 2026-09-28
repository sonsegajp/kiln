'use strict';
// Builds dist\Kiln-<date>.zip: a copy of Kiln to hand to other people. It holds the source (server, web
// UI, engine source, tools, bundled extension packs, docs), the prebuilt engine and its CUDA runtime DLL,
// and setup.bat, which downloads the rest (portable Node if needed, NVIDIA's cuBLAS, the models).
// Never included: config\ (API keys and local settings), models, outputs, logs.
//   node tools\make-dist.js
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const git = (...a) => execFileSync('git', a, { cwd: ROOT, encoding: 'utf8' }).split('\n').map((s) => s.trim()).filter(Boolean);

// every file git would ship (tracked, plus new ones not ignored), minus developer-only folders
const SKIP = [/^\.git/, /^local\//, /^bench\//, /^dist\//, /^server\/tests\//, /^tools\/(yolo_ref\.py|convert_yolo\.py|make-dist\.js)$/];
const files = [...new Set([...git('ls-files'), ...git('ls-files', '--others', '--exclude-standard')])]
  .filter((f) => fs.existsSync(path.join(ROOT, f)) && !SKIP.some((re) => re.test(f)));
// the prebuilt engine: nobody should need Visual Studio to run Kiln
const ENGINE = ['engine/build/kiln-engine.exe', 'engine/build/cudart64_13.dll'];
for (const f of ENGINE) if (!fs.existsSync(path.join(ROOT, f))) throw new Error('missing ' + f + ' (build the engine first)');
files.push(...ENGINE);

// nothing personal goes out: the CivitAI key and any config file must not be in the list or the files
const secrets = [];
try { const c = JSON.parse(fs.readFileSync(path.join(ROOT, 'config', 'civitai.json'), 'utf8')); for (const v of Object.values(c)) if (typeof v === 'string' && v.length >= 20) secrets.push(v); } catch (_) { }
for (const f of files) {
  if (/^config\//.test(f)) throw new Error('config file in the list: ' + f);
  if (secrets.length && !/\.(exe|dll)$/i.test(f)) {
    const s = fs.readFileSync(path.join(ROOT, f), 'utf8');
    for (const k of secrets) if (s.includes(k)) throw new Error('the CivitAI key is in ' + f);
  }
}

const stamp = new Date().toISOString().slice(0, 10);
const name = `Kiln-${stamp}`;
const stage = path.join(DIST, name);
fs.rmSync(stage, { recursive: true, force: true });
for (const f of files) {
  const dst = path.join(stage, 'Kiln', ...f.split('/'));
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(path.join(ROOT, f), dst);
}
fs.writeFileSync(path.join(stage, 'Kiln', 'START HERE.txt'), [
  'Kiln - an image generator for the Anima model that runs on your own PC.',
  '',
  'Needs: Windows 10 (1803) or 11, an NVIDIA graphics card (GTX 16xx / RTX 20xx or newer) with a',
  'current driver, and about 8 GB of free disk space.',
  '',
  '1. Double-click setup.bat. It downloads everything else Kiln needs into this folder:',
  '   the NVIDIA cuBLAS runtime, the Anima model, text encoder and VAE, the turbo LoRA, an upscaler',
  '   and a face detector (about 6 GB), plus a portable Node.js if yours is missing or old.',
  '   It can be stopped and re-run: finished downloads are kept.',
  '2. Double-click start.bat, then open http://localhost:8090/ in your browser.',
  '   Phones on the same Wi-Fi can use the LAN address start.bat prints.',
  '',
  'More in README.md.',
  '',
].join('\r\n'));

const zip = path.join(DIST, name + '.zip');
fs.rmSync(zip, { force: true });
execFileSync(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe'), ['-a', '-c', '-f', zip, '-C', stage, 'Kiln'], { stdio: 'inherit' });
fs.rmSync(stage, { recursive: true, force: true });
console.log(`${zip}  (${(fs.statSync(zip).size / 1e6).toFixed(1)} MB, ${files.length + 1} files)`);
