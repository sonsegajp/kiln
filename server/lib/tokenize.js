'use strict';
// Kiln tokenizers: exact pure-JS reimplementations of
//   - HF Qwen2Tokenizer (byte-level BPE, NFC, Qwen2 pre-tokenizer regex, added tokens)
//   - HF T5TokenizerFast from tokenizer.json (Precompiled charsmap normalizer,
//     Strip(right), Replace(' {2,}' -> '▁'), Metaspace(first), Unigram Viterbi)
//   - ComfyUI prompt weighting (comfy/sd1_clip.py: escape_important, token_weights,
//     embedding: splitting) as used by comfy/text_encoders/anima.py
// Zero dependencies. Everything loads lazily on first use.

const fs = require('fs');
const path = require('path');

const TOKDIR = path.join(__dirname, '..', 'tokenizers');

// ---------------------------------------------------------------------------
// Shared: added-token splitting (leftmost-longest, like HF AddedVocabulary)
// ---------------------------------------------------------------------------
// Returns [{text, start, special:true|false, id}] pieces covering the input.
function splitAddedTokens(text, added /* array of {content,id} sorted by len desc */) {
  const out = [];
  let last = 0;
  let i = 0;
  const n = text.length;
  while (i < n) {
    let hit = null;
    if (text.charCodeAt(i) === 60 /* '<' */ || added.anyNonAngle) {
      for (const tok of added) {
        if (text.startsWith(tok.content, i)) { hit = tok; break; } // sorted longest first
      }
    }
    if (hit) {
      if (i > last) out.push({ text: text.slice(last, i), start: last, added: false });
      out.push({ text: hit.content, start: i, added: true, id: hit.id });
      i += hit.content.length;
      last = i;
    } else {
      i++;
    }
  }
  if (last < n) out.push({ text: text.slice(last), start: last, added: false });
  return out;
}

function prepAdded(list) {
  const a = list.slice().sort((x, y) => y.content.length - x.content.length);
  a.anyNonAngle = a.some(t => !t.content.startsWith('<'));
  return a;
}

// ---------------------------------------------------------------------------
// Qwen2 byte-level BPE
// ---------------------------------------------------------------------------
let QWEN = null;

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

// Qwen2 pre-tokenizer regex. \s/\S are expanded to the Unicode White_Space property
// (what Oniguruma uses in HF tokenizers; JS \s differs on U+0085 and U+FEFF).
const QWEN_PAT = new RegExp(
  "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])" +
  "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
  "|\\p{N}" +
  "| ?[^\\p{White_Space}\\p{L}\\p{N}]+[\\r\\n]*" +
  "|\\p{White_Space}*[\\r\\n]+" +
  "|\\p{White_Space}+(?!\\P{White_Space})" +
  "|\\p{White_Space}+",
  'gu');

function loadQwen() {
  if (QWEN) return QWEN;
  const dir = path.join(TOKDIR, 'qwen25_tokenizer');
  const vocab = JSON.parse(fs.readFileSync(path.join(dir, 'vocab.json'), 'utf8'));
  const encoder = new Map(Object.entries(vocab));
  const lines = fs.readFileSync(path.join(dir, 'merges.txt'), 'utf8').split('\n');
  const ranks = new Map();
  let r = 0;
  for (const line of lines) {
    if (!line || line.startsWith('#version')) continue;
    const sp = line.indexOf(' ');
    if (sp <= 0) continue;
    ranks.set(line, r++); // key "a b"
  }
  const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer_config.json'), 'utf8'));
  const added = [];
  for (const [id, t] of Object.entries(cfg.added_tokens_decoder || {})) {
    added.push({ content: t.content, id: Number(id) });
    encoder.set(t.content, Number(id));
  }
  QWEN = { encoder, ranks, added: prepAdded(added), byteMap: bytesToUnicode(), cache: new Map() };
  return QWEN;
}

function bpe(word, Q) {
  const cached = Q.cache.get(word);
  if (cached) return cached;
  let parts = Array.from(word);
  while (parts.length > 1) {
    let best = -1, bestRank = Infinity;
    for (let i = 0; i < parts.length - 1; i++) {
      const rk = Q.ranks.get(parts[i] + ' ' + parts[i + 1]);
      if (rk !== undefined && rk < bestRank) { bestRank = rk; best = i; }
    }
    if (best < 0) break;
    const a = parts[best], b = parts[best + 1];
    const merged = [];
    for (let i = 0; i < parts.length;) {
      if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i += 2; }
      else { merged.push(parts[i]); i++; }
    }
    parts = merged;
  }
  const ids = parts.map(p => {
    const id = Q.encoder.get(p);
    if (id === undefined) throw new Error('qwen bpe: symbol not in vocab: ' + JSON.stringify(p));
    return id;
  });
  if (Q.cache.size > 20000) Q.cache.clear();
  Q.cache.set(word, ids);
  return ids;
}

function qwenEncode(text) {
  const Q = loadQwen();
  const ids = [];
  for (const piece of splitAddedTokens(text, Q.added)) {
    if (piece.added) { ids.push(piece.id); continue; }
    const norm = piece.text.normalize('NFC');
    const emit = (str) => {
      const bytes = Buffer.from(str, 'utf8');
      let w = '';
      for (const b of bytes) w += Q.byteMap[b];
      for (const id of bpe(w, Q)) ids.push(id);
    };
    QWEN_PAT.lastIndex = 0;
    let m, pos = 0;
    while ((m = QWEN_PAT.exec(norm)) !== null) {
      if (m[0].length === 0) { QWEN_PAT.lastIndex++; continue; }
      if (m.index > pos) emit(norm.slice(pos, m.index)); // Split(Isolated) keeps gaps
      emit(m[0]);
      pos = m.index + m[0].length;
    }
    if (pos < norm.length) emit(norm.slice(pos));
  }
  return ids;
}

// ---------------------------------------------------------------------------
// T5 (SentencePiece Unigram via HF tokenizer.json)
// ---------------------------------------------------------------------------
let T5 = null;

// Precompiled charsmap (spm nmt normalization), darts-clone double array trie.
function parseCharsmap(b64) {
  const buf = Buffer.from(b64, 'base64');
  const trieSize = buf.readUInt32LE(0);
  const units = new Uint32Array(trieSize / 4);
  for (let i = 0; i < units.length; i++) units[i] = buf.readUInt32LE(4 + i * 4);
  const normalized = buf.subarray(4 + trieSize);
  return { units, normalized };
}

function dartsCommonPrefix(units, bytes) {
  // Returns list of values (offsets into normalized blob), shortest match first.
  const res = [];
  let nodePos = 0;
  let unit = units[nodePos];
  nodePos ^= dOffset(unit);
  for (let i = 0; i < bytes.length; i++) {
    const c = bytes[i];
    if (c === 0) break;
    nodePos ^= c;
    unit = units[nodePos];
    if (unit === undefined || dLabel(unit) !== c) return res;
    nodePos ^= dOffset(unit);
    if (dHasLeaf(unit)) res.push(dValue(units[nodePos]));
  }
  return res;
}
function dHasLeaf(u) { return ((u >>> 8) & 1) === 1; }
function dValue(u) { return u & 0x7fffffff; }
function dLabel(u) { return (u & (0x80000000 | 0xff)) >>> 0; }
function dOffset(u) { return ((u >>> 10) << ((u & (1 << 9)) >>> 6)) >>> 0; }

function charsmapTransform(cm, chunk) {
  const r = dartsCommonPrefix(cm.units, Buffer.from(chunk, 'utf8'));
  if (!r.length) return null;
  const start = r[0];
  let end = start;
  while (end < cm.normalized.length && cm.normalized[end] !== 0) end++;
  return cm.normalized.subarray(start, end).toString('utf8');
}

let graphemeSeg = null;
function precompiledNormalize(cm, s) {
  if (!graphemeSeg) graphemeSeg = new Intl.Segmenter('en', { granularity: 'grapheme' });
  let out = '';
  for (const { segment: g } of graphemeSeg.segment(s)) {
    if (Buffer.byteLength(g, 'utf8') < 6) {
      const n = charsmapTransform(cm, g);
      if (n !== null) { out += n; continue; }
    }
    for (const c of g) {
      const n = charsmapTransform(cm, c);
      out += n !== null ? n : c;
    }
  }
  return out;
}

function loadT5() {
  if (T5) return T5;
  const dir = path.join(TOKDIR, 't5_tokenizer');
  const tj = JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf8'));
  // Sanity-check that the declared pipeline is the one we implement.
  const norms = tj.normalizer.type === 'Sequence' ? tj.normalizer.normalizers : [tj.normalizer];
  const kinds = norms.map(n => n.type).join(',');
  if (kinds !== 'Precompiled,Strip,Replace') throw new Error('t5: unexpected normalizer ' + kinds);
  if (tj.pre_tokenizer.type !== 'Metaspace') throw new Error('t5: unexpected pre_tokenizer');
  if (tj.model.type !== 'Unigram') throw new Error('t5: unexpected model');
  const cm = parseCharsmap(norms[0].precompiled_charsmap);
  const strip = norms[1];
  const rep = norms[2];
  const repRe = new RegExp(rep.pattern.Regex !== undefined ? rep.pattern.Regex
    : rep.pattern.String.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gu');
  const ms = tj.pre_tokenizer;
  const vocab = tj.model.vocab;
  const pieceToId = new Map();
  const scores = new Float64Array(vocab.length);
  let minScore = Infinity;
  const root = new Map(); // char trie: Map<char, node>, node.id
  for (let id = 0; id < vocab.length; id++) {
    const [piece, score] = vocab[id];
    pieceToId.set(piece, id);
    scores[id] = score;
    if (score < minScore) minScore = score;
    let node = root;
    for (const ch of piece) {
      let nx = node.get(ch);
      if (!nx) { nx = new Map(); node.set(ch, nx); }
      node = nx;
    }
    node.id = id;
  }
  const added = prepAdded(tj.added_tokens.map(t => ({ content: t.content, id: t.id })));
  let eosId = 1;
  try {
    const sp = tj.post_processor.special_tokens['</s>'];
    if (sp) eosId = sp.ids[0];
  } catch (_) { /* default */ }
  T5 = {
    cm, strip, repRe, repContent: rep.content,
    replacement: ms.replacement, prepend: ms.prepend_scheme || (ms.add_prefix_space ? 'always' : 'never'),
    split: ms.split !== false,
    pieceToId, scores, minScore, root, unkId: tj.model.unk_id, added, eosId,
  };
  return T5;
}

const WS_END = /\p{White_Space}+$/u;
const WS_START = /^\p{White_Space}+/u;

function unigramEncode(T, s) {
  // Viterbi over code points (equivalent to HF encode_optimized over UTF-8 bytes).
  const chars = Array.from(s);
  const n = chars.length;
  const bestScore = new Float64Array(n + 1);
  const from = new Int32Array(n + 1).fill(-1);
  const bestId = new Int32Array(n + 1);
  const unkScore = T.minScore - 10.0;
  for (let st = 0; st < n; st++) {
    if (st > 0 && from[st] < 0) continue; // unreachable (cannot happen: unk fallback)
    const base = bestScore[st];
    let hasSingle = false;
    let node = T.root;
    for (let k = st; k < n; k++) {
      node = node.get(chars[k]);
      if (!node) break;
      if (node.id !== undefined) {
        const end = k + 1;
        const cand = T.scores[node.id] + base;
        if (from[end] < 0 || cand > bestScore[end]) {
          bestScore[end] = cand; from[end] = st; bestId[end] = node.id;
        }
        if (end === st + 1) hasSingle = true;
      }
    }
    if (!hasSingle) {
      const end = st + 1;
      const cand = unkScore + base;
      if (from[end] < 0 || cand > bestScore[end]) {
        bestScore[end] = cand; from[end] = st; bestId[end] = T.unkId;
      }
    }
  }
  const ids = [];
  let e = n;
  let inUnk = false;
  while (e > 0) {
    const st = from[e];
    const id = bestId[e];
    if (id === T.unkId) {
      if (!inUnk) { ids.push(T.unkId); inUnk = true; } // fuse_unk
    } else {
      ids.push(id); inUnk = false;
    }
    e = st;
  }
  ids.reverse();
  // A fused unk string is looked up in the vocab; it can't be a real piece
  // (otherwise Viterbi would have matched it), so unk id stands.
  return ids;
}

// T5 pipeline mode:
//   'json'          (default) exactly what tokenizer.json declares = HF `tokenizers`,
//                   transformers 4.x T5TokenizerFast (what Anima was trained with).
//   'transformers5' transformers >=5 rebuilds the T5 pipeline itself: Precompiled only,
//                   WhitespaceSplit + Metaspace(prepend always). Differs from 'json' only
//                   after special tokens (</s> etc. typed in the prompt) and on exotic
//                   whitespace like U+0085. Select with env KILN_T5_MODE=transformers5.
let T5_MODE = process.env.KILN_T5_MODE === 'transformers5' ? 'transformers5' : 'json';
function setT5Mode(m) { T5_MODE = m === 'transformers5' ? 'transformers5' : 'json'; }

function metaspaceWords(s, rep, prepend) {
  s = s.split(' ').join(rep);
  if (prepend && !s.startsWith(rep)) s = rep + s;
  const words = [];
  let cur = '';
  for (const ch of s) { // SplitDelimiterBehavior::MergedWithNext
    if (ch === rep && cur.length) { words.push(cur); cur = ''; }
    cur += ch;
  }
  if (cur.length) words.push(cur);
  return words;
}

function t5EncodeNoEos(text) {
  const T = loadT5();
  const ids = [];
  for (const piece of splitAddedTokens(text, T.added)) {
    if (piece.added) { ids.push(piece.id); continue; }
    if (T5_MODE === 'transformers5') {
      const s = precompiledNormalize(T.cm, piece.text);
      for (const ws of s.split(/\p{White_Space}+/u)) {
        if (!ws.length) continue;
        for (const w of metaspaceWords(ws, T.replacement, true))
          for (const id of unigramEncode(T, w)) ids.push(id);
      }
      continue;
    }
    let s = precompiledNormalize(T.cm, piece.text);
    if (T.strip.strip_right) s = s.replace(WS_END, '');
    if (T.strip.strip_left) s = s.replace(WS_START, '');
    s = s.replace(T.repRe, T.repContent);
    if (!s.length) continue;
    s = s.split(' ').join(T.replacement);
    const rep = T.replacement;
    if (!s.startsWith(rep)) {
      if (T.prepend === 'always' || (T.prepend === 'first' && piece.start === 0)) s = rep + s;
    }
    const words = [];
    if (T.split) {
      // SplitDelimiterBehavior::MergedWithNext on the replacement char
      let cur = '';
      for (const ch of s) {
        if (ch === rep && cur.length) { words.push(cur); cur = ''; }
        cur += ch;
      }
      if (cur.length) words.push(cur);
    } else words.push(s);
    for (const w of words) for (const id of unigramEncode(T, w)) ids.push(id);
  }
  return ids;
}

// Mirrors T5TokenizerFast(text)["input_ids"] (post-processor appends </s>).
function t5Encode(text) {
  const ids = t5EncodeNoEos(text);
  ids.push(loadT5().eosId);
  return ids;
}

// ---------------------------------------------------------------------------
// ComfyUI prompt weighting (comfy/sd1_clip.py)
// ---------------------------------------------------------------------------
const ESC_R = '\x00\x01', ESC_L = '\x00\x02';
function escapeImportant(t) { return t.split('\\)').join(ESC_R).split('\\(').join(ESC_L); }
function unescapeImportant(t) { return t.split(ESC_R).join(')').split(ESC_L).join('('); }

function parseParentheses(string) {
  const result = [];
  let cur = '';
  let nest = 0;
  for (const ch of string) {
    if (ch === '(') {
      if (nest === 0) {
        if (cur) { result.push(cur); cur = '('; } else cur = '(';
      } else cur += ch;
      nest += 1;
    } else if (ch === ')') {
      nest -= 1;
      if (nest === 0) { result.push(cur + ')'); cur = ''; } else cur += ch;
    } else cur += ch;
  }
  if (cur) result.push(cur);
  return result;
}

// Python float() emulation. Returns null when Python would raise ValueError.
const PY_WS = /^[\p{White_Space}\x1c-\x1f]+|[\p{White_Space}\x1c-\x1f]+$/gu;
function pyFloat(s) {
  let t = s.replace(PY_WS, '');
  // Unicode decimal digits -> ASCII (Python accepts any Nd digit).
  t = t.replace(/\p{Nd}/gu, (d) => {
    let cp = d.codePointAt(0);
    if (cp >= 48 && cp <= 57) return d;
    let start = cp;
    while (start > 0 && /\p{Nd}/u.test(String.fromCodePoint(start - 1))) start--;
    return String((cp - start) % 10);
  });
  const m = /^([+-]?)(?:(inf(?:inity)?)|(nan)|((?:\d(?:_?\d)*)?(?:\.(?:\d(?:_?\d)*)?)?(?:[eE][+-]?\d(?:_?\d)*)?))$/i.exec(t);
  if (!m) return null;
  const sign = m[1] === '-' ? -1 : 1;
  if (m[2]) return sign * Infinity;
  if (m[3]) return NaN;
  const num = m[4].replace(/_/g, '');
  // must contain at least one digit in the mantissa
  const mant = num.split(/[eE]/)[0];
  if (!/\d/.test(mant)) return null;
  // Python forbids "_" adjacent to "." or "e": the regex above already enforces digit_(digit)
  return sign * Number(num);
}

function tokenWeights(string, currentWeight) {
  const a = parseParentheses(string);
  let out = [];
  for (let x of a) {
    let weight = currentWeight;
    if (x.length >= 2 && x[x.length - 1] === ')' && x[0] === '(') {
      x = x.slice(1, -1);
      const xx = x.lastIndexOf(':');
      weight *= 1.1;
      if (xx > 0) {
        const f = pyFloat(x.slice(xx + 1));
        if (f !== null) { weight = f; x = x.slice(0, xx); }
      }
      out = out.concat(tokenWeights(x, weight));
    } else {
      out.push([x, currentWeight]);
    }
  }
  return out;
}

// ComfyUI splits each segment on "embedding:" preceded by whitespace and, since Kiln
// has no textual-inversion embeddings, the embedding name is dropped and any leftover
// text after it is tokenized (identical to ComfyUI with a missing embedding file).
const EMB = 'embedding:';
function splitEmbeddings(seg) {
  // Python re \s == str.isspace(): Unicode White_Space plus U+001C..U+001F
  const parts = seg.split(/(?<=[\p{White_Space}\x1c-\x1f])embedding:/u);
  const words = [parts[0]];
  for (let i = 1; i < parts.length; i++) words.push(EMB + parts[i]);
  const out = [];
  for (let word of words) {
    if (word === '') continue;
    if (word.startsWith(EMB)) {
      let name = word.slice(EMB.length).replace(/^\n+|\n+$/g, '');
      const sp = name.split(/[\p{White_Space}\x1c-\x1f]+/u).filter(Boolean);
      if (!sp.length) continue; // ComfyUI would crash here; treat as nothing
      let ename = sp[0];
      let leftover = sp.slice(1).join(' ');
      const mm = /[<[]/.exec(ename);
      if (mm) {
        leftover = ename.slice(mm.index) + (leftover ? ' ' + leftover : '');
        ename = ename.slice(0, mm.index);
      }
      const stripped = ename.replace(/^,+|,+$/g, '');
      if (stripped.length < ename.length) leftover = ename.slice(stripped.length) + ' ' + leftover;
      if (leftover !== '') word = leftover; else continue;
    }
    out.push(word);
  }
  return out;
}

function tokenizeWeighted(text, encodeFn, stripEnd) {
  const segs = tokenWeights(escapeImportant(text), 1.0);
  const ids = [];
  const weights = [];
  for (const [seg, w] of segs) {
    for (const word of splitEmbeddings(unescapeImportant(seg))) {
      let t = encodeFn(word);
      if (stripEnd) t = t.slice(0, -1);
      for (const id of t) { ids.push(id); weights.push(w); }
    }
  }
  return { ids, weights };
}

const QWEN_PAD = 151643;

function encodePrompt(text) {
  text = String(text == null ? '' : text);
  const q = tokenizeWeighted(text, qwenEncode, false);
  const qwen_ids = q.ids.length ? q.ids : [QWEN_PAD];
  const t = tokenizeWeighted(text, t5Encode, true);
  const t5_ids = t.ids.concat([loadT5().eosId]);
  const t5_weights = t.weights.map(w => (Number.isFinite(w) ? w : (Number.isNaN(w) ? 1.0 : Math.sign(w) * 1e4))).concat([1.0]);
  return { qwen_ids, t5_ids, t5_weights };
}

module.exports = {
  encodePrompt, qwenEncode, t5Encode, t5EncodeNoEos, setT5Mode,
  getT5Mode: () => T5_MODE,
  tokenWeights, parseParentheses, escapeImportant, unescapeImportant, pyFloat,
  _raw_t5_weights: (text) => tokenizeWeighted(text, t5Encode, true),
  warmup() { loadQwen(); loadT5(); },
};
