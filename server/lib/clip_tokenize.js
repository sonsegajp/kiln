'use strict';
// Kiln CLIP tokenizer for SDXL (CLIP-L ViT-L/14 + OpenCLIP bigG share one BPE vocab).
// Pure-JS reimplementation of what ComfyUI does for SDXL prompts:
//   - comfy/sd1_clip.py SDTokenizer.tokenize_with_weights (weights, embedding: split,
//     77-token chunking, word-boundary rule, padding) + comfy/sdxl_clip.py SDXLTokenizer
//     (L pads with EOS 49407, G pads with 0)
//   - HF transformers CLIPTokenizer underneath it (vocab/merges from
//     openai/clip-vit-large-patch14, byte-level BPE with "</w>" end-of-word markers)
//
// HF text cleanup differs by transformers version, so there are two modes:
//   'fast' (default) transformers >= 5 CLIPTokenizer (backed by the `tokenizers` library):
//                    NFC, \s+ -> ' ', lowercase, CLIP regex, ByteLevel, BPE. What a fresh
//                    ComfyUI install (transformers>=4.50.3, no upper bound) runs today.
//   'slow'           transformers 4.x slow CLIPTokenizer without ftfy (ComfyUI does not
//                    depend on ftfy): BERT BasicTokenizer cleanup (drops control/format
//                    chars, spaces out CJK ideographs, NFC, whitespace split, lowercase).
//   The modes agree on ordinary prompts. They differ only on CJK ideographs (slow makes
//   each one its own word), control/format characters such as U+200B, U+FEFF, U+0085 or
//   \v (slow deletes them, joining the neighbours), and word-final capital sigma.
//   Select with env KILN_CLIP_MODE=slow or setClipMode('slow').
//   Not implemented: the ftfy branch of the slow tokenizer (ftfy.fix_text: HTML unescape,
//   curly-quote uncurling, fullwidth folding, mojibake repair). Neither mode unescapes
//   HTML entities, exactly like ComfyUI without ftfy: "&amp;" stays "&", "amp", ";".
//
// Prompt syntax (ComfyUI semantics, parser shared with tokenize.js):
//   (text) = x1.1 per nesting level, (text:1.3) sets the weight (inner parens multiply
//   on top of it), \( \) are literal parens. [text] has NO meaning in ComfyUI: brackets
//   are ordinary characters. "embedding:name" is unsupported: Kiln has no textual
//   inversion, so it is tokenized as plain text (ComfyUI with no embedding directory).
// Zero dependencies. Vocab/merges load lazily on first use: ~5 ms from bpe_ranks.bin,
// ~75 ms when that cache has to be (re)built from vocab.json + merges.txt.
// Verified against HF transformers 5.4.0 ('fast') and 4.47.1 ('slow') driving ComfyUI's
// own SDTokenizer code: see server/tests/clip_tokenize_test.js.

const fs = require('fs');
const path = require('path');
const { tokenWeights, escapeImportant, unescapeImportant } = require('./tokenize');

const CLIPDIR = path.join(__dirname, '..', 'tokenizers', 'clip');

const BOS = 49406;          // <|startoftext|>
const EOS = 49407;          // <|endoftext|>
const PAD_L = EOS;          // SDTokenizer(pad_with_end=True)
const PAD_G = 0;            // SDXLClipGTokenizer(pad_with_end=False) -> pad_token 0
const CHUNK = 77;           // max_length
const MAX_WORD = 8;         // SDTokenizer.max_word_length
const SPECIALS = [['<|startoftext|>', BOS], ['<|endoftext|>', EOS]];

let MODE = process.env.KILN_CLIP_MODE === 'slow' ? 'slow' : 'fast';
function setClipMode(m) { MODE = m === 'slow' ? 'slow' : 'fast'; }

// ---------------------------------------------------------------------------
// Vocab + merges
// ---------------------------------------------------------------------------
let CLIP = null;

// GPT-2 byte -> printable unicode table (same as tokenize.js / HF bytes_to_unicode).
function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const map = new Array(256);
  for (let i = 0; i < bs.length; i++) map[bs[i]] = String.fromCodePoint(cs[i]);
  return map;
}

// Parse vocab.json + merges.txt into one Int32Array:
//   [byteId x256, byteIdEnd x256 (byte symbol + "</w>"), (idA, idB, idMerged) x nMerges]
function buildTables() {
  const vocab = JSON.parse(fs.readFileSync(path.join(CLIPDIR, 'vocab.json'), 'utf8'));
  const id = (k) => {
    const v = vocab[k];
    if (typeof v !== 'number' || !Object.prototype.hasOwnProperty.call(vocab, k))
      throw new Error('clip: symbol not in vocab: ' + JSON.stringify(k));
    return v;
  };
  if (id('<|startoftext|>') !== BOS || id('<|endoftext|>') !== EOS)
    throw new Error('clip: unexpected special token ids');
  // HF: merges.read().strip().split("\n")[1 : 49152-256-2+1] -> 48894 merges.
  const lines = fs.readFileSync(path.join(CLIPDIR, 'merges.txt'), 'utf8').trim().split('\n');
  const nMerges = Math.min(lines.length - 1, 49152 - 256 - 2);
  const t = new Int32Array(512 + 3 * nMerges);
  const byteMap = bytesToUnicode();
  for (let b = 0; b < 256; b++) { t[b] = id(byteMap[b]); t[256 + b] = id(byteMap[b] + '</w>'); }
  for (let r = 0; r < nMerges; r++) {
    const line = lines[r + 1].trimEnd();
    const sp = line.indexOf(' ');
    const a = line.slice(0, sp), b = line.slice(sp + 1);
    t[512 + 3 * r] = id(a); t[513 + 3 * r] = id(b); t[514 + 3 * r] = id(a + b);
  }
  return t;
}

// Binary cache of buildTables() next to the sources (parsing the JSON costs ~70 ms, a
// cached load ~5 ms). Keyed on the sources' size + mtime; rebuilt (best effort) when stale.
const CACHE_FILE = path.join(CLIPDIR, 'bpe_ranks.bin');
const CACHE_MAGIC = 0x504c434b; // 'KCLP'
const CACHE_VER = 1;
const HDR = 48;

function sourceStamp() {
  const v = fs.statSync(path.join(CLIPDIR, 'vocab.json'));
  const m = fs.statSync(path.join(CLIPDIR, 'merges.txt'));
  return [v.size, v.mtimeMs, m.size, m.mtimeMs];
}

function readCache(stamp) {
  let buf;
  try { buf = fs.readFileSync(CACHE_FILE); } catch (_) { return null; }
  if (buf.length < HDR || buf.readUInt32LE(0) !== CACHE_MAGIC || buf.readUInt32LE(4) !== CACHE_VER) return null;
  if (buf.readDoubleLE(8) !== stamp[0] || buf.readDoubleLE(16) !== stamp[1] ||
      buf.readDoubleLE(24) !== stamp[2] || buf.readDoubleLE(32) !== stamp[3]) return null;
  const n = buf.readUInt32LE(40);
  if (buf.length !== HDR + 4 * n) return null;
  const t = new Int32Array(n); // host byte order == LE on every platform Kiln runs on
  new Uint8Array(t.buffer).set(buf.subarray(HDR, HDR + 4 * n));
  return t;
}

function writeCache(stamp, t) {
  const buf = Buffer.alloc(HDR + 4 * t.length);
  buf.writeUInt32LE(CACHE_MAGIC, 0); buf.writeUInt32LE(CACHE_VER, 4);
  for (let i = 0; i < 4; i++) buf.writeDoubleLE(stamp[i], 8 + 8 * i);
  buf.writeUInt32LE(t.length, 40);
  Buffer.from(t.buffer, t.byteOffset, t.byteLength).copy(buf, HDR);
  const tmp = CACHE_FILE + '.' + process.pid + '.tmp';
  fs.writeFileSync(tmp, buf);
  fs.renameSync(tmp, CACHE_FILE);
}

// Merge ranks live in an open-addressing hash table keyed by the (idA, idB) pair:
// cheaper to build and probe than a Map with 32-bit keys (not V8 small ints).
const HBITS = 17, HSIZE = 1 << HBITS, HMASK = HSIZE - 1;
function hslot(a, b) { return (Math.imul(a, 0x9E3779B1) ^ Math.imul(b + 0x7f4a7c15, 0x85EBCA77)) >>> (32 - HBITS); }

function loadClip() {
  if (CLIP) return CLIP;
  const stamp = sourceStamp();
  let t = readCache(stamp);
  if (!t) {
    t = buildTables();
    try { writeCache(stamp, t); } catch (_) { /* read-only install: parse every start */ }
  }
  const nMerges = (t.length - 512) / 3;
  const hA = new Int32Array(HSIZE).fill(-1), hB = new Int32Array(HSIZE), hR = new Int32Array(HSIZE);
  const mergedId = new Int32Array(nMerges);
  for (let r = 0; r < nMerges; r++) {
    const a = t[512 + 3 * r], b = t[513 + 3 * r];
    mergedId[r] = t[514 + 3 * r];
    let i = hslot(a, b);
    while (hA[i] !== -1 && !(hA[i] === a && hB[i] === b)) i = (i + 1) & HMASK;
    if (hA[i] === -1) { hA[i] = a; hB[i] = b; hR[i] = r; } // first (lowest) rank wins
  }
  CLIP = {
    hA, hB, hR, mergedId,
    byteId: t.subarray(0, 256), byteIdEnd: t.subarray(256, 512),
    cache: new Map(), decoder: null,
  };
  return CLIP;
}

function mergeRank(C, a, b) {
  let i = hslot(a, b);
  for (;;) {
    const x = C.hA[i];
    if (x === -1) return -1;
    if (x === a && C.hB[i] === b) return C.hR[i];
    i = (i + 1) & HMASK;
  }
}

// ---------------------------------------------------------------------------
// BPE on one pre-token (a regex match), memoized. Same result as HF slow bpe() (merge
// the lowest-ranked pair everywhere, left to right) and tokenizers' BPE model.
// ---------------------------------------------------------------------------
function bpeWord(word, C) {
  const hit = C.cache.get(word);
  if (hit !== undefined) return hit;
  const bytes = Buffer.from(word, 'utf8');
  const n = bytes.length;
  let sym = new Array(n);
  for (let i = 0; i < n - 1; i++) sym[i] = C.byteId[bytes[i]];
  sym[n - 1] = C.byteIdEnd[bytes[n - 1]];
  while (sym.length > 1) {
    let bestRank = Infinity, bestA = -1, bestB = -1;
    for (let i = 0; i < sym.length - 1; i++) {
      const r = mergeRank(C, sym[i], sym[i + 1]);
      if (r >= 0 && r < bestRank) { bestRank = r; bestA = sym[i]; bestB = sym[i + 1]; }
    }
    if (bestRank === Infinity) break;
    const m = C.mergedId[bestRank];
    const out = [];
    for (let i = 0; i < sym.length;) {
      if (i < sym.length - 1 && sym[i] === bestA && sym[i + 1] === bestB) { out.push(m); i += 2; }
      else { out.push(sym[i]); i++; }
    }
    sym = out;
  }
  if (C.cache.size > 50000) C.cache.clear();
  C.cache.set(word, sym);
  return sym;
}

// CLIP pre-tokenizer regex. \s is the Unicode White_Space property (Oniguruma in HF
// tokenizers; JS \s would also treat U+FEFF as space).
const CLIP_PAT_SRC =
  "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d" +
  "|\\p{L}+|\\p{N}|[^\\p{White_Space}\\p{L}\\p{N}]+";
const PAT_FAST = new RegExp(CLIP_PAT_SRC, 'gu');
const PAT_SLOW = new RegExp(CLIP_PAT_SRC, 'giu'); // HF slow compiles it with re.IGNORECASE

function emitWords(s, pat, C, ids) {
  pat.lastIndex = 0;
  let m;
  while ((m = pat.exec(s)) !== null) {
    const w = m[0];
    if (w === '<|startoftext|>') { ids.push(BOS); continue; }
    if (w === '<|endoftext|>') { ids.push(EOS); continue; }
    const t = bpeWord(w, C);
    for (let i = 0; i < t.length; i++) ids.push(t[i]);
  }
}

// Split out typed special tokens (leftmost match, like HF added-token extraction).
function splitSpecials(s, onText, onSpecial) {
  let pos = 0;
  for (;;) {
    let best = -1, bestTok = null;
    for (const tok of SPECIALS) {
      const k = s.indexOf(tok[0], pos);
      if (k >= 0 && (best < 0 || k < best)) { best = k; bestTok = tok; }
    }
    if (best < 0) break;
    if (best > pos) onText(s.slice(pos, best));
    onSpecial(bestTok[1]);
    pos = best + bestTok[0].length;
  }
  if (pos < s.length) onText(s.slice(pos));
}

// 'fast': normalizers NFC -> Replace(\s+, ' ') -> Lowercase (per char, like Rust
// char::to_lowercase: no final-sigma rule); normalized special tokens are matched on the
// normalized text; then Split(CLIP regex) + ByteLevel (its GPT-2 regex is a no-op on CLIP
// pieces) + BPE.
const WS_RUN = /\p{White_Space}+/gu;
function lowerPerChar(s) {
  let out = '';
  for (const ch of s) out += ch.toLowerCase();
  return out;
}
function encodeFast(text, C, ids) {
  let s = text.normalize('NFC').replace(WS_RUN, ' ');
  // Plain toLowerCase() is identical except for word-final capital sigma.
  s = s.includes('\u03a3') ? lowerPerChar(s) : s.toLowerCase();
  splitSpecials(s, (t) => emitWords(t, PAT_FAST, C, ids), (id) => ids.push(id));
}

// 'slow': HF PreTrainedTokenizer.tokenize splits typed special tokens on the raw text,
// then CLIPTokenizer._tokenize runs BasicTokenizer(strip_accents=False,
// do_split_on_punc=False) and the CLIP regex (IGNORECASE) on the joined words.
const PY_SPACE_RUN = /[\p{White_Space}\x1c-\x1f]+/u; // Python str.split()
const IS_C = /^\p{C}$/u;
const IS_ZS = /^\p{Zs}$/u;
function isCjk(cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
    (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
    (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
    (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}
function basicTokenize(text) {
  let s = '';
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') { s += ' '; continue; }
    if (cp === 0 || cp === 0xFFFD || IS_C.test(ch)) continue;   // _clean_text
    if (IS_ZS.test(ch)) { s += ' '; continue; }
    s += isCjk(cp) ? ' ' + ch + ' ' : ch;                       // _tokenize_chinese_chars
  }
  const words = s.normalize('NFC').split(PY_SPACE_RUN).filter(Boolean);
  return words.map((w) => w.toLowerCase()).join(' ');           // str.lower() per word
}
function encodeSlow(text, C, ids) {
  splitSpecials(text, (t) => emitWords(basicTokenize(t), PAT_SLOW, C, ids), (id) => ids.push(id));
}

// HF CLIPTokenizer(text)["input_ids"] without BOS/EOS.
function clipEncodeNoSpecial(text) {
  const C = loadClip();
  const ids = [];
  if (MODE === 'slow') encodeSlow(String(text), C, ids); else encodeFast(String(text), C, ids);
  return ids;
}

// HF CLIPTokenizer(text)["input_ids"]: BOS + tokens + EOS.
function clipEncode(text) {
  return [BOS].concat(clipEncodeNoSpecial(text), [EOS]);
}

// Token strings for debugging, e.g. ['1', 'girl</w>'].
function clipTokens(text) {
  const C = loadClip();
  if (!C.decoder) {
    const vocab = JSON.parse(fs.readFileSync(path.join(CLIPDIR, 'vocab.json'), 'utf8'));
    C.decoder = [];
    for (const k in vocab) C.decoder[vocab[k]] = k;
  }
  return clipEncodeNoSpecial(text).map((id) => C.decoder[id]);
}

// ---------------------------------------------------------------------------
// ComfyUI SDTokenizer.tokenize_with_weights
// ---------------------------------------------------------------------------
// re.split(r'(?<=\s)embedding:', seg), re-prefixing each later piece. Python \s ==
// str.isspace(): Unicode White_Space plus U+001C..U+001F.
const EMB = 'embedding:';
const EMB_SPLIT = /(?<=[\p{White_Space}\x1c-\x1f])embedding:/u;
function embeddingPieces(seg) {
  const parts = seg.split(EMB_SPLIT);
  const out = [parts[0]];
  for (let i = 1; i < parts.length; i++) out.push(EMB + parts[i]);
  return out.filter((x) => x !== '');
}

// Returns chunks as [{ids: [...77], weights: [...77]}] with the pad slots marked null
// (filled per encoder by padChunks). Weights are raw ComfyUI floats (may be NaN/Inf).
function chunkPrompt(text) {
  text = String(text == null ? '' : text);
  // One token group per weighted segment / embedding piece ("word" in ComfyUI terms).
  const groups = [];
  for (const [seg, w] of tokenWeights(escapeImportant(text), 1.0)) {
    for (const piece of embeddingPieces(unescapeImportant(seg))) {
      groups.push({ ids: clipEncodeNoSpecial(piece), w });
    }
  }
  const chunks = [];
  let ids = [BOS], ws = [1.0];
  const closeChunk = (pad) => {
    ids.push(EOS); ws.push(1.0);
    for (let k = 0; k < pad; k++) { ids.push(null); ws.push(1.0); }
    chunks.push({ ids, weights: ws });
    ids = [BOS]; ws = [1.0];
  };
  const room = CHUNK - 1; // has_end_token
  for (const g of groups) {
    const isLarge = g.ids.length >= MAX_WORD;
    let t = g.ids;
    while (t.length > 0) {
      if (t.length + ids.length > room) {
        const remaining = room - ids.length;
        if (isLarge) {
          // Long word: fill this chunk to the brim and continue in the next one.
          for (let k = 0; k < remaining; k++) { ids.push(t[k]); ws.push(g.w); }
          t = t.slice(remaining);
          closeChunk(0);
        } else {
          // Short word: close this chunk (EOS + padding), move the whole word on.
          closeChunk(remaining);
        }
      } else {
        for (let k = 0; k < t.length; k++) { ids.push(t[k]); ws.push(g.w); }
        t = [];
      }
    }
  }
  closeChunk(Math.max(0, CHUNK - ids.length - 1));
  return chunks;
}

function padChunks(chunks, pad, sanitize) {
  const ids = [], weights = [];
  for (const c of chunks) {
    ids.push(c.ids.map((x) => (x === null ? pad : x)));
    weights.push(sanitize ? c.weights.map(cleanWeight) : c.weights.slice());
  }
  return { ids, weights };
}

// ComfyUI passes float("nan") / float("inf") weights straight into the encoder (the
// image turns to NaN). Kiln clamps them like tokenize.js does for T5: NaN -> 1.0,
// +-Inf -> +-1e4, so the result stays JSON-safe.
function cleanWeight(w) {
  if (Number.isFinite(w)) return w;
  return Number.isNaN(w) ? 1.0 : Math.sign(w) * 1e4;
}

// SDXLTokenizer.tokenize_with_weights -> { l: {ids, weights}, g: {ids, weights} }.
// Every row has 77 entries; both encoders get the same number of chunks.
function encodeSDXL(text) {
  const chunks = chunkPrompt(text);
  return { l: padChunks(chunks, PAD_L, true), g: padChunks(chunks, PAD_G, true) };
}

// Same, with the unsanitized ComfyUI weights (tests / diagnostics).
function encodeSDXLRaw(text) {
  const chunks = chunkPrompt(text);
  return { l: padChunks(chunks, PAD_L, false), g: padChunks(chunks, PAD_G, false) };
}

module.exports = {
  encodeSDXL, encodeSDXLRaw, clipEncode, clipEncodeNoSpecial, clipTokens,
  setClipMode, getClipMode: () => MODE,
  BOS, EOS, PAD_L, PAD_G, CHUNK,
  warmup() { loadClip(); },
};
