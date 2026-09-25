'use strict';
// Tests for server/lib/clip_tokenize.js (SDXL CLIP-L / CLIP-G prompt tokenization).
//   node server/tests/clip_tokenize_test.js
//   node server/tests/clip_tokenize_test.js --dump-corpus <out.json> [nBpe nSdxl seed]
//        writes the fuzz inputs, to regenerate the golden hashes with an HF oracle
//
// Sections:
//   1. published CLIP encodings
//   2. vocab cross-check: token strings exist in vocab.json and spell the input back
//   3. ComfyUI weight syntax (expected values derived by hand from comfy/sd1_clip.py)
//   4. ComfyUI chunking / padding rules (hand-derived from SDTokenizer.tokenize_with_weights)
//   5. HF oracle goldens: explicit ids for tricky strings, 'fast' and 'slow' modes
//   6. HF oracle fuzz: per-case hashes for a seeded corpus, both modes
// Oracle = ComfyUI's SDTokenizer code (exec'd verbatim) on HF transformers 5.4.0
// CLIPTokenizer ('fast') and transformers 4.47.1 slow CLIPTokenizer without ftfy ('slow').

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const clip = require('../lib/clip_tokenize');

const BOS = 49406, EOS = 49407;

// ---------------------------------------------------------------------------
// Seeded fuzz corpus (shared with the oracle via --dump-corpus)
// ---------------------------------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const WORDS = ['a', 'the', 'photo', 'of', 'cat', 'dog', 'girl', 'hair', 'eyes', 'looking', 'viewer',
  'quality', 'masterpiece', 'best', 'hello', 'world', 'running', 'beautiful', 'extremely',
  'detailed', 'Background', 'SUNLIGHT', 'McDonald', 'iPhone', 'x', 'I', 'embedding', 'foo'];
const TAGS = ['1girl', '2boys', 'solo', 'long_hair', 'short hair', 'blue eyes', 'looking at viewer',
  'absurdres', 'highres', 'score_9', 'rating:safe', 'artist name', 'signature', 'watermark',
  'hatsune miku', 'tachi-e', 'o_o', ':d', ';)', '^_^', '3d', '2.5d', '16:9', 'x-ray', 'k-pop',
  'pok\u00e9mon', 'na\u00efve', 'caf\u00e9', 'very aesthetic', 'newest', 'from side', 'upper body', 'depth of field',
  'cherry blossoms', 'school uniform', 'pleated skirt', 'thighhighs', 'zettai ryouiki', 'v-shaped eyebrows'];
const PUNCT = [',', '.', '!', '?', '...', '--', '\u2014', '\u2013', '\u201c', '\u201d', '\u2018', '\u2019',
  '\u00ab', '\u00bb', '(', ')', '[', ']', '{', '}', '<', '>', '|', '\\', '/', ':', ';', '@', '#', '$', '%',
  '^', '&', '&amp;', '&lt;', '*', '+', '=', '~', '`', '_', '__', '???', '!!!', "'", "''", '"'];
const APOS = ["don't", "it's", "we're", "I've", "I'm", "you'll", "he'd", "DON'T", "IT'S", "'s", "'t",
  "rock'n'roll", "o'clock", "'re'", "''s", "x's", "'ll'd", '\u2019s'];
const UNI = ['\u521d\u97f3\u30df\u30af', '\u6771\u65b9Project', '\u6f22\u5b57', '\u3072\u3089\u304c\u306a',
  '\u30ab\u30bf\u30ab\u30ca', '\ud55c\uad6d\uc5b4', '\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac',
  '\u039f\u0394\u039f\u03a3', '\u03a3\u0391\u03a3 \u03a3', '\u0130stanbul', 'stra\u00dfe', '\ufb01sh',
  '\uff26\uff55\uff4c\uff4c', '\uff08\u5168\u89d2\uff09', '\ud83d\ude00', '\ud83d\udc69\u200d\ud83d\udcbb',
  '\u2764\ufe0f', '\ud83c\udff3\ufe0f\u200d\ud83c\udf08', 'e\u0301', '\u0301', 'A\u030a', '\u01c5',
  '\u017f', "'\u017f", '\u212a', 'a\u00adb', 'a\u200bb', 'a\u200db', '\ufeffbom', 'l1\u2028l2',
  'x\u3000y', 'nb\u00a0sp', 'n\u0085l', 'v\x0bt', 'f\x0cf', 'fs\x1cx', 'us\x1fx', 'del\x7fx',
  'nul\x00x', 'rep\ufffdx', 'pua\ue000x', '\ud840\udc00', '\uf900', '\ud835\udd6c', '\u2460\u2167',
  '\u00bd\u00b2', '\u0663\u0664', '\u0967\u0968', 'ol\u00e1', '\u00c9COLE', 'Stra\u00dfe',
  '\u0418\u0432\u0430\u043d', '\u05e9\u05dc\u05d5\u05dd', '\u0645\u0631\u062d\u0628\u0627',
  '\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35', '\u1100\u1161', '\u00c5ngstr\u00f6m'];
const SPECIAL = ['<|endoftext|>', '<|startoftext|>', '<|EndOfText|>', 'a<|endoftext|>b', '<|endoftext'];
const SEPS = [' ', ' ', ' ', ', ', ',', '', '_', '  ', '\t', '\n', '\r\n', ' \n ', '-'];
const WEIGHTS = ['1.2', '1.1', '0.8', '1.5', '2', '0', '-1', '1.', '.5', ' 1.3 ', '1e1', 'abc', '',
  'nan', 'inf', '-inf', '1_0', '0.9 ', '1.25', '+1.4', '\u0661.\u0665', '1:2', ')'];

function makeCorpus(nBpe, nSdxl, seed) {
  const R = mulberry32(seed);
  const int = (n) => Math.floor(R() * n);
  const pick = (a) => a[int(a.length)];
  const letters = (n) => { let s = ''; for (let i = 0; i < n; i++) s += String.fromCharCode(97 + int(26)); return s; };
  const randWord = () => {
    const k = R();
    if (k < 0.28) return pick(WORDS);
    if (k < 0.48) return pick(TAGS);
    if (k < 0.60) return pick(PUNCT);
    if (k < 0.68) return pick(APOS);
    if (k < 0.82) return pick(UNI);
    if (k < 0.90) return letters(1 + int(24));
    if (k < 0.96) return String(int(1000000)) + (R() < 0.3 ? '.' + int(100) : '');
    return pick(SPECIAL);
  };
  const bpe = [];
  for (let i = 0; i < nBpe; i++) {
    const n = 1 + int(10);
    let s = R() < 0.1 ? pick(SEPS) : '';
    for (let j = 0; j < n; j++) s += randWord() + (j < n - 1 ? pick(SEPS) : '');
    if (R() < 0.1) s += pick(SEPS);
    const c = R();
    bpe.push(c < 0.08 ? s.toUpperCase() : s);
  }
  const weighted = (inner) => {
    const k = R();
    if (k < 0.35) return '(' + inner + ')';
    if (k < 0.47) return '((' + inner + '))';
    if (k < 0.75) return '(' + inner + ':' + pick(WEIGHTS) + ')';
    if (k < 0.80) return '[' + inner + ']';
    if (k < 0.85) return '\\(' + inner + '\\)';
    if (k < 0.93) return '(' + inner + ' (' + pick(TAGS) + ':' + pick(WEIGHTS) + ') ' + pick(WORDS) + ')';
    return pick(['(', ')', '((', '))', ')(', '\\)', '(\\(']) + inner;
  };
  const sdxl = [];
  for (let i = 0; i < nSdxl; i++) {
    const n = 1 + int(R() < 0.35 ? 110 : 25);
    const parts = [];
    for (let j = 0; j < n; j++) {
      const k = R();
      let t;
      if (k < 0.55) t = pick(TAGS);
      else if (k < 0.75) { t = pick(WORDS); const m = int(4); for (let q = 0; q < m; q++) t += ' ' + pick(WORDS); }
      else if (k < 0.83) t = randWord();
      else if (k < 0.88) t = letters(20 + int(160));                 // one word, many tokens
      else if (k < 0.92) t = pick(['!', '?', '.', 'ha']).repeat(4 + int(60));
      else if (k < 0.96) t = 'embedding:' + pick(WORDS) + (R() < 0.5 ? ' ' + pick(TAGS) : '');
      else t = pick(UNI);
      if (R() < 0.38) t = weighted(t);
      if (R() < 0.08) t = weighted(t);
      parts.push(t);
    }
    let s = '';
    for (let j = 0; j < parts.length; j++) s += (j ? pick([', ', ',', ' ', ' , ', ',\n', ', ']) : '') + parts[j];
    sdxl.push(s);
  }
  return { bpe, sdxl };
}

const FUZZ = { nBpe: 800, nSdxl: 300, seed: 20260925 };

if (process.argv[2] === '--dump-corpus') {
  const nB = Number(process.argv[4] || FUZZ.nBpe), nS = Number(process.argv[5] || FUZZ.nSdxl);
  const seed = Number(process.argv[6] || FUZZ.seed);
  fs.writeFileSync(process.argv[3], JSON.stringify(makeCorpus(nB, nS, seed)));
  console.log('wrote', process.argv[3], nB, nS, seed);
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail) {
  if (cond) { pass++; return true; }
  fail++;
  failures.push(name + (detail ? '\n      ' + detail : ''));
  return false;
}
const J = (x) => JSON.stringify(x);
function eq(name, got, exp) { return ok(name, J(got) === J(exp), 'got ' + J(got) + '\n      exp ' + J(exp)); }

// Canonical form used for hashing (non-finite weights as strings, like the oracle dump).
function canonWeights(ws) {
  return ws.map((row) => row.map((w) => (Number.isFinite(w) ? w : Number.isNaN(w) ? 'nan' : w > 0 ? 'inf' : '-inf')));
}
function canonSDXL(r) {
  return J({ l: { ids: r.l.ids, weights: canonWeights(r.l.weights) }, g: { ids: r.g.ids, weights: canonWeights(r.g.weights) } });
}
const h6 = (s) => crypto.createHash('sha256').update(s).digest('hex').slice(0, 6);

// Content of a chunk up to and including its EOS, plus what fills the rest.
function contentOf(row) { const k = row.indexOf(EOS, 1); return row.slice(0, k + 1); }

// ---------------------------------------------------------------------------
// 1. Published encodings
// ---------------------------------------------------------------------------
clip.setClipMode('fast');
// openai/CLIP README + HF CLIPTokenizer docs example: "a photo of a cat".
eq('published: a photo of a cat', clip.clipEncode('a photo of a cat'), [49406, 320, 1125, 539, 320, 2368, 49407]);
// HF CLIPModel docs example batch ["a photo of a cat", "a photo of a dog"].
eq('published: a photo of a dog', clip.clipEncode('a photo of a dog'), [49406, 320, 1125, 539, 320, 1929, 49407]);
eq('published: empty string', clip.clipEncode(''), [49406, 49407]);
eq('published: case folded', clip.clipEncode('A PHOTO OF A CAT'), clip.clipEncode('a photo of a cat'));

// ---------------------------------------------------------------------------
// 2. Vocab cross-check of token strings
// ---------------------------------------------------------------------------
const VOCAB = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'tokenizers', 'clip', 'vocab.json'), 'utf8'));
const byteDec = (() => {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice(); let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const m = new Map(); for (let i = 0; i < 256; i++) m.set(String.fromCodePoint(cs[i]), bs[i]);
  return m;
})();
// Token strings -> list of spelled words (a word ends at a "</w>" token or a special token).
function spellWords(tokens) {
  const words = [];
  let bytes = [];
  for (const t of tokens) {
    if (t === '<|startoftext|>' || t === '<|endoftext|>') { words.push(t); continue; }
    for (const ch of t.replace(/<\/w>$/, '')) bytes.push(byteDec.get(ch));
    if (t.endsWith('</w>')) { words.push(Buffer.from(bytes).toString('utf8')); bytes = []; }
  }
  if (bytes.length) words.push(Buffer.from(bytes).toString('utf8') + '<no </w>>');
  return words;
}
// The published CLIP pre-tokenizer (HF CLIPTokenizer pattern) on the 'fast' normalized
// text: NFC, whitespace runs -> ' ', per-character lowercase.
function clipWordsRef(s) {
  let t = s.normalize('NFC').replace(/\p{White_Space}+/gu, ' ');
  t = Array.from(t, (c) => c.toLowerCase()).join('');
  const pat = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\p{White_Space}\p{L}\p{N}]+/gu;
  return t.match(pat) || [];
}
const XCHECK = [
  ['1girl, solo, long_hair, (masterpiece:1.2), best quality, artist name',
    ['1</w>', 'girl</w>', ',</w>', 'solo</w>', ',</w>', 'long</w>', '_</w>', 'hair</w>', ',</w>', '(</w>',
      'masterpiece</w>', ':</w>', '1</w>', '.</w>', '2</w>', '),</w>', 'best</w>', 'quality</w>', ',</w>',
      'artist</w>', 'name</w>']],
  ["Don't stop, it's 3:45pm!!", ['don</w>', "'t</w>", 'stop</w>', ',</w>', 'it</w>', "'s</w>", '3</w>', ':</w>',
    '4</w>', '5</w>', 'pm</w>', '!!</w>']],
  ['multiple    spaces\tand\nnewlines', ['multiple</w>', 'spaces</w>', 'and</w>', 'new', 'lines</w>']],
  ['12345', ['1</w>', '2</w>', '3</w>', '4</w>', '5</w>']],
  ['caf\u00e9 na\u00efve \u00c9COLE', null],
  ['\u521d\u97f3\u30df\u30af \ud83d\ude00 \u039f\u0394\u039f\u03a3', null],
  ['score_9, score_8_up, rating:explicit, 16:9, x-ray, ^_^', null],
  ['\u201csmart quotes\u201d \u2014 and \u2019apostrophes\u2019', null],
  ['supercalifragilisticexpialidocious', null],
  ['<|endoftext|> inside <|startoftext|>', null],
];
for (const [s, expTokens] of XCHECK) {
  const toks = clip.clipTokens(s);
  const ids = clip.clipEncodeNoSpecial(s);
  ok('xcheck in-vocab: ' + J(s), toks.every((t, i) => t !== undefined && VOCAB[t] === ids[i]), J(toks));
  // Every word's last token (and only it) carries </w>, and the bytes spell the word.
  eq('xcheck spells back word by word: ' + J(s), spellWords(toks), clipWordsRef(s));
  if (expTokens) eq('xcheck tokens: ' + J(s), toks, expTokens);
}

// ---------------------------------------------------------------------------
// 3. Weight syntax (comfy/sd1_clip.py token_weights / parse_parentheses)
// ---------------------------------------------------------------------------
// Returns [[tokenString, weight], ...] for the content tokens of a short prompt.
const DEC = []; for (const k in VOCAB) DEC[VOCAB[k]] = k;
function tw(prompt) {
  const r = clip.encodeSDXLRaw(prompt);
  const ids = r.l.ids[0], ws = r.l.weights[0];
  const out = [];
  for (let i = 1; ids[i] !== EOS; i++) out.push([DEC[ids[i]].replace('</w>', ''), ws[i]]);
  return out;
}
eq('weight: plain', tw('cat'), [['cat', 1]]);
eq('weight: (x) = 1.1', tw('(cat)'), [['cat', 1.1]]);
eq('weight: ((x)) = 1.1*1.1', tw('((cat))'), [['cat', 1.1 * 1.1]]);
eq('weight: (((x))) = 1.1^3 by repeated *=', tw('(((cat)))'), [['cat', 1.1 * 1.1 * 1.1]]);
eq('weight: (x:1.3)', tw('(cat:1.3)'), [['cat', 1.3]]);
eq('weight: ((x:1.3)) explicit replaces outer', tw('((cat:1.3))'), [['cat', 1.3]]);
eq('weight: ((x):1.3) inner multiplies explicit', tw('((cat):1.3)'), [['cat', 1.3 * 1.1]]);
eq('weight: nested spans', tw('(a (dog) b)'), [['a', 1.1], ['dog', 1.1 * 1.1], ['b', 1.1]]);
eq('weight: (x:0.5) de-emphasis', tw('(cat:0.5)'), [['cat', 0.5]]);
eq('weight: (x: 1.3 ) python float() strips', tw('(cat: 1.3 )'), [['cat', 1.3]]);
eq('weight: (x:abc) keeps text, 1.1', tw('(cat:abc)'), [['cat', 1.1], [':', 1.1], ['abc', 1.1]]);
eq('weight: (x:1.2:1.3) last colon wins', tw('(cat:1.2:1.3)'), [['cat', 1.3], [':', 1.3], ['1', 1.3], ['.', 1.3], ['2', 1.3]]);
eq('weight: (:1.5) colon at 0 is text', tw('(:1.5)'), [[':', 1.1], ['1', 1.1], ['.', 1.1], ['5', 1.1]]);
eq('weight: (x:-1) negative', tw('(cat:-1)'), [['cat', -1]]);
eq('weight: \\(x\\) is literal', tw('\\(cat\\)'), [['(', 1], ['cat', 1], [')', 1]]);
eq('weight: (\\(x\\)) literal inside group', tw('(\\(cat\\))'), [['(', 1.1], ['cat', 1.1], [')', 1.1]]);
eq('weight: [x] has no meaning in ComfyUI', tw('[cat]'), [['[', 1], ['cat', 1], [']', 1]]);
eq('weight: unbalanced ((x) -> "(x" at 1.1', tw('((cat)'), [['(', 1.1], ['cat', 1.1]]);
eq('weight: stray ) disables later groups', tw('a) (b) c'), [['a', 1], [')', 1], ['(', 1], ['b', 1], [')', 1], ['c', 1]]);
eq('weight: segment text keeps its commas', tw('(a, b:1.2), c'), [['a', 1.2], [',', 1.2], ['b', 1.2], [',', 1], ['c', 1]]);
eq('weight: embedding: is plain text', tw('x embedding:foo'), [['x', 1], ['embed', 1], ['ding', 1], [':', 1], ['foo', 1]]);
{
  const raw = clip.encodeSDXLRaw('(cat:nan) (dog:inf)');
  ok('weight: raw keeps NaN', Number.isNaN(raw.l.weights[0][1]));
  eq('weight: raw keeps Inf', raw.l.weights[0][2], Infinity);
  const clean = clip.encodeSDXL('(cat:nan) (dog:inf) (x:-inf)');
  eq('weight: encodeSDXL clamps non-finite', clean.l.weights[0].slice(1, 4), [1, 1e4, -1e4]);
}

// ---------------------------------------------------------------------------
// 4. Chunking and padding (SDTokenizer.tokenize_with_weights)
// ---------------------------------------------------------------------------
const CAT = 2368, DOG = 1929;
const cats = (n) => Array(n).fill('cat').join(' ');
const dogsW = (n, w) => '(' + Array(n).fill('dog').join(' ') + ':' + w + ')';
function shape(r) { return r.l.ids.map((row) => row.length); }
function allRows77(r) { return ['l', 'g'].every((k) => r[k].ids.every((x) => x.length === 77) && r[k].weights.every((x) => x.length === 77)); }
{
  const r = clip.encodeSDXL('');
  eq('chunk: empty prompt L', r.l.ids, [[BOS, EOS].concat(Array(75).fill(EOS))]);
  eq('chunk: empty prompt G (pad 0)', r.g.ids, [[BOS, EOS].concat(Array(75).fill(0))]);
  eq('chunk: empty prompt weights', r.l.weights, [Array(77).fill(1)]);
  eq('chunk: whitespace-only = empty', clip.encodeSDXL('  \n\t '), r);
  eq('chunk: "()" = empty', clip.encodeSDXL('()'), r);
}
{
  const r = clip.encodeSDXL('a photo of a cat');
  eq('chunk: L pads with EOS', r.l.ids[0], [49406, 320, 1125, 539, 320, 2368, 49407].concat(Array(70).fill(EOS)));
  eq('chunk: G pads with 0', r.g.ids[0], [49406, 320, 1125, 539, 320, 2368, 49407].concat(Array(70).fill(0)));
  eq('chunk: G weights == L weights', r.g.weights, r.l.weights);
}
{
  const r = clip.encodeSDXL(cats(75));
  eq('chunk: 75 tokens fit one chunk exactly', r.l.ids, [[BOS].concat(Array(75).fill(CAT), [EOS])]);
  eq('chunk: 75 tokens, G has no pad', r.g.ids, r.l.ids);
}
{
  // One plain run = one group of 76 (>= 8, "large"): split at the boundary.
  const r = clip.encodeSDXL(cats(76));
  eq('chunk: 76-token group splits 75 + 1', r.l.ids,
    [[BOS].concat(Array(75).fill(CAT), [EOS]), [BOS, CAT, EOS].concat(Array(74).fill(EOS))]);
  eq('chunk: 76-token group, G second chunk pads 0', r.g.ids[1], [BOS, CAT, EOS].concat(Array(74).fill(0)));
}
{
  // 74 plain + a 3-token weighted group: 3 doesn't fit in the 1 slot left -> whole group moves.
  const r = clip.encodeSDXL(cats(74) + ' ' + dogsW(3, 1.5));
  eq('chunk: short group moves to next chunk (L)', r.l.ids,
    [[BOS].concat(Array(74).fill(CAT), [EOS, EOS]), [BOS, DOG, DOG, DOG, EOS].concat(Array(72).fill(EOS))]);
  eq('chunk: short group moves to next chunk (G pad 0 at slot 76)', r.g.ids[0], [BOS].concat(Array(74).fill(CAT), [EOS, 0]));
  eq('chunk: moved group keeps its weight', r.l.weights[1].slice(0, 5), [1, 1.5, 1.5, 1.5, 1]);
}
{
  // 70 plain + 7-token group (< 8): moves. 70 plain + 8-token group (>= 8): split 5 + 3.
  const r7 = clip.encodeSDXL(cats(70) + ' ' + dogsW(7, 1.2));
  eq('chunk: 7-token group (< max_word_length 8) moves whole', r7.l.ids.map(contentOf),
    [[BOS].concat(Array(70).fill(CAT), [EOS]), [BOS].concat(Array(7).fill(DOG), [EOS])]);
  eq('chunk: 7-token group padding in chunk 1', r7.l.ids[0].slice(71), Array(6).fill(EOS));
  const r8 = clip.encodeSDXL(cats(70) + ' ' + dogsW(8, 1.2));
  eq('chunk: 8-token group (>= 8) splits across chunks', r8.l.ids.map(contentOf),
    [[BOS].concat(Array(70).fill(CAT), Array(5).fill(DOG), [EOS]), [BOS].concat(Array(3).fill(DOG), [EOS])]);
  eq('chunk: split group weights', r8.l.weights[0].slice(70, 77), [1, 1.2, 1.2, 1.2, 1.2, 1.2, 1]);
}
{
  const r = clip.encodeSDXL(cats(70) + ' ' + dogsW(5, 1.2));
  eq('chunk: group that exactly fills 75 stays', r.l.ids, [[BOS].concat(Array(70).fill(CAT), Array(5).fill(DOG), [EOS])]);
}
{
  // "embedding:" splits the run into a separate group even though it's plain text:
  // 70 cats | "embedding:dog dog dog dog" = embed ding : dog x4 (7 tokens, small) -> moves.
  const r = clip.encodeSDXL(cats(70) + ' embedding:dog dog dog dog');
  eq('chunk: embedding: starts a new group', r.l.ids.map(contentOf),
    [[BOS].concat(Array(70).fill(CAT), [EOS]), [BOS, 19703, 796, 281].concat(Array(4).fill(DOG), [EOS])]);
  // Without the exact prefix it's one 78-token run -> large -> cut at 75.
  const r2 = clip.encodeSDXL(cats(70) + ' embeddingx:dog dog dog dog');
  eq('chunk: no split without the exact "embedding:" prefix', r2.l.ids.map(contentOf).map((x) => x.length), [77, 5]);
}
{
  const r = clip.encodeSDXL(cats(300));
  eq('chunk: 300 tokens -> 4 chunks', shape(r), [77, 77, 77, 77]);
  ok('chunk: every row is 77 wide (L and G)', allRows77(r));
  eq('chunk: 300 = 4 x 75, last chunk full', contentOf(r.l.ids[3]), [BOS].concat(Array(75).fill(CAT), [EOS]));
  eq('chunk: L/G same chunk count', r.g.ids.length, r.l.ids.length);
  eq('chunk: BOS/EOS weights are 1', [r.l.weights[1][0], r.l.weights[1][76]], [1, 1]);
}
{
  // A 200-token weighted group is cut 75 / 75 / 50 and keeps its weight in every chunk.
  const r = clip.encodeSDXL('(' + cats(200) + ':1.1)');
  eq('chunk: long weighted group spans chunks', r.l.ids.map(contentOf).map((x) => x.length), [77, 77, 52]);
  ok('chunk: long weighted group weights', allRows77(r) &&
    r.l.weights[0].slice(1, 76).every((w) => w === 1.1) && r.l.weights[2].slice(1, 51).every((w) => w === 1.1) &&
    r.l.weights[2].slice(51).every((w) => w === 1));
  // A single whitespace "word" of 100+ tokens is cut mid-word too (only groups < 8 move).
  const long = clip.encodeSDXL(cats(40) + ' ' + '!?'.repeat(60));
  ok('chunk: huge single word is cut mid-word', long.l.ids.length >= 2 && allRows77(long) &&
    contentOf(long.l.ids[0]).length === 77);
}

// ---------------------------------------------------------------------------
// 5. HF oracle goldens (explicit ids, without BOS/EOS)
// ---------------------------------------------------------------------------
/*GOLDEN-BEGIN*/ // generated from the HF oracle dumps; do not edit by hand
const GOLDEN = {
  fuzz: {"nBpe":800,"nSdxl":300,"seed":20260925},
  tricky: [
    "a photo of a cat",
    "1girl, solo, long_hair, (masterpiece:1.2), best quality, artist name",
    "Don't stop, it's 3:45pm!!",
    "I'M GOING, YOU'LL SEE, WE'D",
    "rock'n'roll o'clock ''s '' x's 'll'd",
    "multiple    spaces\tand\nnewlines\r\nend",
    "  leading and trailing  ",
    "12345 3.14159 1,000,000 0x1F 2024-09-25",
    "score_9, score_8_up, rating:explicit, 16:9, x-ray, ^_^, :3, ;), o_o, >_<",
    "!!!!!!!!!!!!!!!!!!!!!!!! ??? ... -- ~~ ** ## @@",
    "caf\u00e9 na\u00efve \u00c9COLE Stra\u00dfe \u00c5ngstr\u00f6m",
    "e\u0301 A\u030a nfd accents",
    "\u201csmart quotes\u201d \u2014 and \u2019apostrophes\u2019 \u2018x\u2019",
    "&amp; &lt;b&gt; &quot;html&quot; &#39;",
    "\u521d\u97f3\u30df\u30af \u6771\u65b9Project \u6f22\u5b57 \u3072\u3089\u304c\u306a",
    "\ud55c\uad6d\uc5b4 \u0418\u0432\u0430\u043d \u05e9\u05dc\u05d5\u05dd \u0645\u0631\u062d\u0628\u0627",
    "\u039f\u0394\u039f\u03a3 \u03a3\u0391\u03a3 \u0130stanbul \u017f \u212a \u01c5",
    "\ud83d\ude00 \ud83d\udc69\u200d\ud83d\udcbb \u2764\ufe0f \ud83c\udff3\ufe0f\u200d\ud83c\udf08",
    "\uff26\uff55\uff4c\uff4c\uff57\uff49\uff44\uff54\uff48 \uff08\u5168\u89d2\uff09 \ufb01sh",
    "zero\u200bwidth soft\u00adhyphen bom\ufeffx nbsp\u00a0x ideo\u3000space",
    "ctrl\u000bvt\fff\u001cfs\u007fdel\u0085nel\u2028ls",
    "nul\u0000x rep\ufffdx pua\ue000x",
    "<|endoftext|> <|startoftext|> <|EndOfText|> a<|endoftext|>b <|endoftext",
    "supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis",
    "embedding:easynegative, (embedding:foo:1.2)",
    "\\(escaped\\) [brackets] {braces} <angle>",
    "",
    " ",
    "\u2460\u2167 \u00bd\u00b2 \u0663\u0664 \u0967\u0968 \ud835\udd6c",
    "hatsune miku, tachi-e, zettai ryouiki, pok\u00e9mon, v-shaped eyebrows",
  ],
  fast: { // HF transformers 5.4.0 CLIPTokenizer
    tricky: [
      [49406,320,1125,539,320,2368,49407],
      [49406,272,1611,267,5797,267,1538,318,2225,267,263,12066,281,272,269,273,2361,949,3027,267,2456,1981,49407],
      [49406,847,713,1691,267,585,568,274,281,275,276,990,748,49407],
      [49406,328,880,1245,267,592,1342,862,267,649,1896,49407],
      [49406,2172,262,333,262,3341,334,262,6716,8445,338,8445,343,568,1342,1896,49407],
      [49406,6470,9006,537,1218,3418,806,49407],
      [49406,3833,537,37427,49407],
      [49406,272,273,274,275,276,274,269,272,275,272,276,280,272,267,271,271,271,267,271,271,271,271,343,272,325,273,271,273,275,268,271,280,268,273,276,49407],
      [49406,4431,318,280,267,4431,318,279,318,705,267,10567,281,33228,267,272,277,281,280,267,343,268,3077,267,14423,61,267,281,274,267,26,2361,334,318,334,267,44144,283,49407],
      [49406,30146,21622,3824,678,2432,16671,4441,15483,30314,49407],
      [49406,15304,1097,35689,563,3459,8166,1894,127,253,324,23176,10352,807,7255,332,49407],
      [49406,4166,127,354,77,11525,31738,49407],
      [49406,728,506,4115,5808,728,507,2005,537,728,503,4089,3121,745,542,728,503,728,502,343,728,503,49407],
      [49406,261,6259,282,261,3333,282,321,261,4725,282,261,47587,282,18231,261,47587,282,5,258,274,280,282,49407],
      [49406,42748,251,165,253,111,2429,253,3909,363,48338,29032,117,1965,162,120,95,35751,501,4813,110,3909,231,4813,234,4813,359,49407],
      [49406,15197,250,31871,255,31625,368,29503,110,31090,377,147,102,147,250,147,243,147,507,12986,10948,25722,16378,22625,49407],
      [49406,138,123,138,112,138,123,139,481,139,225,138,109,139,481,328,16384,11231,129,379,330,131,484,49407],
      [49406,7334,26304,18135,1752,32610,49407],
      [49406,171,121,228,171,121,243,171,121,234,171,121,234,171,121,245,171,121,231,171,121,226,171,121,242,171,121,486,171,120,486,34314,101,164,100,496,171,120,487,171,105,223,849,49407],
      [49406,5848,9844,23571,3773,22618,1441,745,576,19469,171,119,379,343,6459,4393,343,1909,334,2138,49407],
      [49406,35602,331,10291,1304,472,3854,477,2003,2693,1268,49407],
      [49406,1156,331,444,343,6487,39802,343,755,320,36288,478,343,49407],
      [49406,49407,49406,49407,320,49407,321,27,347,40786,4160,49407],
      [49406,1642,2857,13093,2076,5868,26850,835,639,38466,28714,749,20253,9800,535,532,1065,901,1556,13697,9916,78,39031,13903,49407],
      [49406,19703,796,281,18308,8869,267,263,19703,796,281,23435,281,272,269,273,264,49407],
      [49406,59,263,18707,59,264,314,36183,316,346,19249,348,283,6946,285,49407],
      [49406,49407],
      [49406,49407],
      [49406,158,239,510,158,227,371,33613,41175,149,352,149,353,7410,356,7410,357,6874,243,361,49407],
      [49406,3447,41096,37703,267,648,4039,268,324,267,34098,13040,1511,520,72,2608,267,22066,267,341,268,10127,19923,49407],
    ],
    bpe: [
      '103db81ff4a941986f44adff481be89aa7fd10f6495ef0b908c080dfd1e1be9d34f8d25df3c9f1d396aa2b60270066ea',
      'c8e32b1098430645b64c543366c1a76c0d5768dda197177dfae96eef48c1b24586be3cbcf5174810d0d5351f243af917',
      'ca9246513ce6486c07f3c200fa8f8e39aa59200a03e1c4a99b79ec7deff8b575e3a95798c05b5ec61e331295467abe8b',
      '236f9939928e0a2c62382300fc06b72ec44cb0d69197c9f2b16d13f8f4131ab3e2748160c9a2629aea95d216ae4ed11b',
      'dc1ff71ab3f6d2638dda11aa30009a0fd41dba329b77041bdb4838a21fcafd346994310148a9c0e8407e44051f3468ac',
      '02c6afe2d681c7d12d7d587f136082a79bc099e48c09961acb2782a9c1fe8d8c3ebc3809d5272b17bc4fe43dcb21b8f4',
      'b5178978c833168c0439145376a3eab55fec727ecf5d4545cd966577c253bc85b99cae8a2d6024ea0a475fec68dc0a7b',
      '50e7dc5c8ef368d45a428d7377260d75dc3529e8cb22737e095ba7ca846780844f9e457b49d33171f357487420a5b7ee',
      'c00a58cdccae23176e4a0ce486aad1817d8a81e7fd0640c4829ab6b0fd4b886e5d17a285c9e40ff8eae448d71a1de4c9',
      '105323e13672276ef7da51cd7535ef3b6e95b3c608f5a81414a5ce57d7a75c47a7c1fe68707e82610bd2136082380cdc',
      'd64d6993de485c30c63a14cc26966a8c71fcb57c7328aa641682e68ea888e5e66ce1b58a8f718a08e2777584a4943101',
      '65b27a908f777b8a2e8096f00b688aa4f6ae54e262ce8ced2be4260d307006b7107901b6a568169b47c725edb0028c58',
      '9bb9a0c67a5438667b5dae84dffe4eeb6f440f34753de1b9124022cb0dfa8c71fce6fc2eb329cc2957341d63287b561a',
      '9e8811a81779c0102161e3893eb4418e7f4606287b2a4fa96d99136abf9671d7c4e1cb2366154951063183fff85b5b54',
      'dec0ba016e9922f9cd8df374b68bb2337111d2faa37845b5b07cf212ea4263f6ba8714f747939b4c521e88b768369cac',
      'db5ecab7a4e3ca34296737e6437ea6edd2c2a3334b727c90691a9cc5c58f2259edb43931355312c5577613d8ba052cca',
      'b94d4b6b9dde08de79acf09ad9ae49b1e11d02f36f4815369d1e3ad65874c1630ff6744b1a966bf42f425cff739a4747',
      '0da691f290379d068748f9ee420f21b2e8794192e4f1cc7e76ee72093d76e95ab8ac1be5efcd7376214374cba841f0d8',
      '3e6748c5dd35417314e4f3c199189166b15378bbf7b1fc48bd7ec4dd3c8c1c1ce6fa30b173af8e12ab8304c6551fcaaf',
      'b440fb62d90bd6b450c30bde34757f9a6542c8b9045c9e989f163a70337bd0eaab4b9def57ac84acbff54354ab5a5ac1',
      '25731494777907c1ee1f18e5d3d7b8e56fba10b0af8281100f3300d2e29af0d658a33139f94481e1844f9410657c8ba0',
      '6c327226a9704812963d170ee1a19e2edd03c782a778adc3d42da0a28f1883d63df1043ff4f9f4d83b12b5c180093356',
      '705c16044c9e0d6f68d6328a21166180cb6bdfbe475931a3b1e17965fe2c13d8ba60d0695d83093269418c4b1950da61',
      '1261371d0272d29ac0073b67912d3fe98402ba55ddb4c5d2843db39debcad90d2bd362ac58739ba49d242ce898e6fc33',
      '40b06dcef8f20ccaa633753061bbad3544d00368c3b5b9938fc78070b7fa442874b71b5aaf2b1181045651ef351f7a92',
      '697a4093baf4942cec1eaa07511a2a18d198da027d5515aaeb9c885417c875df6d81fd65ab315701e3f5298a4febbae8',
      '30e7502ad87fd48e9a864c4826a55c663190c0c4205b56e84d9afec4542c63bda182b04ad8732de0c27403674d5d457b',
      'c51948da33ed79ce5653d94e15a9f38fda50464082157d2e3892954017797a022f82c19f445f3bcb114bd2e8d4760ae9',
      '51469ce26101d2b126b18ffb1a622eb911fe474f169cb88a1bbacc878e5cd9aacd41a661a4f6ae9b38813218735ad5a3',
      '637cd40a9cdbd55804aa34f9729e822685fd7e579c1caa6e87ce1c968ab3f0fda14e5c416671f6d8d4d0a7e5529b2b82',
      '746832fb812a3f38ce0bf80b3d52f9ec9eddba9e6aaed15bb9dd4cda77bf6549439e2c9e42842bcdfc8918d198a59f0e',
      '45f055bf54f8ddb36d419d5bfafbb843020c39ab430e553af4949e616f2ae75e0639f98360f33c5cb22f965d4ea2ed8f',
      'da33edeebb9e9ecefeff29c98c2282c37b2ec983ac3630fb59a0ff3e97e0b2984cf253b5137e191c59de4c5433ea53fc',
      'f989903ebcf89002fb4320b6e543aafb9a3e6621b8e273ee9df2dce155957fb322f66e32e669b59344294875bfc71e09',
      '81ddead7a61e83242596a0d3839fbd91853de76c65682ee7759acc6912c673814cec2bacac45e8e85ed9adf0d48dd7fc',
      '5130338f20340edc2fe0b2751b6a3da6c0157ef1cbc690f29d2d10c8fb5e637e71fe339d0b4e804e82f08654a18314a2',
      'f2757fdd315ee5a2ca79effd10bf5f655b527aea3e8e13e8ad7f87d376a54e79c6b36d67dba35346235a8e8e69cfcfe3',
      '582aca6c1121bfd124d90d2b14433dbf9f3f558b0ee315dcbfda197859167e022edc830a859f546bb7762d4b06629307',
      'f5aa3d13116a0664c4c1df4bb21f0b8c71fc859c8ecd6526bba8e22cefc0a2963cac43b42d9ddae82d13e06f79485d9c',
      '14a2fdfb794bf5a8143463554779496f746868d45aef3d304a88a2b6c3b72564c7cd611c20c9ed0bb0abc98f8cc86f51',
      '18a5a183d0dd54d1c2a2b4c8a13198382e2db2751c37ddea1af5196e06703d4bba81f88cde24af6444a6c99ca5b4443b',
      'bdc48a199c85892a13eddba8ec223c3304784e426664dce1cee0736fc559c8680bf81f8134facdbcf4547c9c5ffe3f15',
      '4ca2aff31b08412eed747ad6b79388f699a5275a3137c30362f8b5918d72b6e9159e087e727ecfe7148ad90d2b0c3a6d',
      '9a148a76fe84bf73e868c578b75a620910748d9b260dcf31cf9a823644c25e0e37dd1ac16e37bd48f42050c6c8ecee59',
      '738861092a2e953d7eb8f80f50f4a90ccf51244eac34bbf91ff5b7210b15180688f5ffb97bbd364bea9d09961a019bd1',
      'cfd2359e0ad5da33ed5e4346e5a325fbbc395c7faea44e2d496bc4376fa56ec5bed064fd6a3aa636aac28f968ad1a50a',
      '80c813479a63e8630461614d43e42757086bab945cd39afa191d434df05dec426377d1163e2fc2b78f97f5a814e920f9',
      'e92ad42f21c78462d7b8c0d5448edc93fcc13e4dc71f55c68607a6ded7346238bab2ae324eebe2e36a9cc2022a93a7a1',
      '39ab6b9604dbf5a81497c5594c8b79ee899d2a9b362aa0462ed4bee863c542cc31fb1a01cba5bd2db93ec68745120ac7',
      '644df258b36b17a90fa56e4f7b12b482d09f4f7685b9df0e1c07a89b28d2d942636572ea4aa093d3e3d54c13e686310d',
    ].join(''),
    sdxl: [
      '701f4323e4321856842b75ead81f1fd902c839c57566fa8a280900786ef581dd00aaa6a9365489da5e13c9c94b776d05',
      'a5bc55a56a74241f502d37032762789c6879a874905322190934d4154e25ffd55c86dfbf1cbd8a729ba3d11346dd59f6',
      '4ef1085452d6b70753327f9ed30c7b8c7f41de6b35ec21ee1f6b58e8964590f50ee958322accaa38e98461e722f77e63',
      'dea165e1a636067c1cefdf7dc090ad1ce2a708d8e24be69df2c7528715285863527dc9459f506f29798b86f090a5bd42',
      '9d6131898d14726e5f8967963639f561dcbb11444c57a6894ecc47ec4f9f5583f699c7f4f5fe8c8c896a2189e21e81e7',
      'b8800f28c5a667e169d5e97dc830cdb24e7d4380bc5255000c0fe6a1b6189bc9dcf3cbd4c8bc64d48d9fb8a936c7b92b',
      'cae0e57274439ead38a89b462de2d4164970e1d41f40e06924ce6f095ca5dbb84923f6ca504936db229ad59d73772252',
      'f9975a5bb76cf5028513751b94e8e6a0c40a04aeab83958305d75a688c0d883299517eebf1fc630dae24495e85efef2c',
      '33fd865156345e5bdb5c46625357dd2f519a2470ab21254e4d8e63d6f787db1641ba15d5c5eb8fe293d26246e122f921',
      '6124c486153eceb8e2dc4582ba6fbbf84f0091b84ee0dd0c141ccd35a9fb93f3c1643d6209e91c94934fbbfe8b31bd1b',
      '367e0b1686424642d0f36cf56f5fc4e8c5fff3641152638c921c51642559ad2ab4d64a8d55bccacfe4881b9292b7bf2a',
      'be20d095a3b91ff3348bd280fe95e6c2ef2836588f956275ed51c9b4b0fe80036e2c63a22d96593da7fc4dc30e64af67',
      '5a82941466a2e9a1dc234c60d2d22349c47b18fdb56f36d5537e06240a7fcfe2687b88e08ff0572101b8d513f5657caa',
      'a1f8f1c7501e8fdf003f8f1546393d3c98a102fc66340f52867f0fec942a348bbe0050fad02d464b5da1eea81ebb78af',
      '14925310877a618d2a9b4271e0ff46d3d2d97eac7762f7a3092463bdf5e7258b44ef024a98694c20a1d6b6cf7c6b26e5',
      'b50b7b8cd36505a3370bbb5f628045de8ca5973ca3f9de31febb7f7ff2a905939fcf27d43a4d86a05364c48486592668',
      '5c67c39e38d7ac0d115bd3344e8aeac13280836344d78e93f3b078898a6eed7cb76a6c8b0b8d3d532a069e4061ea4839',
      '3d3510c6de9e6b7f89302f5c4f4650c1d6dcd300f38a3318cb4c9809f467fb89e8c14e09c41d10a5e36629db743ca12d',
      'cac6c7ff178ec66154a628ab3fad521a7c2a8be516f7c451df587cffbd1c7b242fa9ad2d',
    ].join(''),
  },
  slow: { // HF transformers 4.47.1 CLIPTokenizer
    tricky: [
      [49406,320,1125,539,320,2368,49407],
      [49406,272,1611,267,5797,267,1538,318,2225,267,263,12066,281,272,269,273,2361,949,3027,267,2456,1981,49407],
      [49406,847,713,1691,267,585,568,274,281,275,276,990,748,49407],
      [49406,328,880,1245,267,592,1342,862,267,649,1896,49407],
      [49406,2172,262,333,262,3341,334,262,6716,8445,338,8445,343,568,1342,1896,49407],
      [49406,6470,9006,537,1218,3418,806,49407],
      [49406,3833,537,37427,49407],
      [49406,272,273,274,275,276,274,269,272,275,272,276,280,272,267,271,271,271,267,271,271,271,271,343,272,325,273,271,273,275,268,271,280,268,273,276,49407],
      [49406,4431,318,280,267,4431,318,279,318,705,267,10567,281,33228,267,272,277,281,280,267,343,268,3077,267,14423,61,267,281,274,267,26,2361,334,318,334,267,44144,283,49407],
      [49406,30146,21622,3824,678,2432,16671,4441,15483,30314,49407],
      [49406,15304,1097,35689,563,3459,8166,1894,127,253,324,23176,10352,807,7255,332,49407],
      [49406,4166,127,354,77,11525,31738,49407],
      [49406,728,506,4115,5808,728,507,2005,537,728,503,4089,3121,745,542,728,503,728,502,343,728,503,49407],
      [49406,261,6259,282,261,3333,282,321,261,4725,282,261,47587,282,18231,261,47587,282,5,258,274,280,282,49407],
      [49406,42748,507,165,253,367,2429,253,3909,363,27667,365,29032,373,1965,162,120,351,35751,501,4813,110,3909,231,4813,234,4813,359,49407],
      [49406,15197,250,31871,255,31625,368,29503,110,31090,377,147,102,147,250,147,243,147,507,12986,10948,25722,16378,22625,49407],
      [49406,138,123,138,112,138,123,139,480,139,225,138,109,139,480,328,16384,11231,129,379,330,131,484,49407],
      [49406,7334,18800,18135,1752,26844,13042,49407],
      [49406,171,121,228,171,121,243,171,121,234,171,121,234,171,121,245,171,121,231,171,121,226,171,121,242,171,121,486,171,120,486,34314,357,164,100,496,171,120,487,171,105,223,849,49407],
      [49406,14867,23571,1571,7804,745,576,3012,343,6459,4393,343,1909,334,2138,49407],
      [49406,35602,10862,83,1021,23699,1233,2693,1268,49407],
      [49406,29358,343,4229,343,755,9203,49407],
      [49406,49407,49406,49407,320,49407,321,27,347,40786,4160,49407],
      [49406,1642,2857,13093,2076,5868,26850,835,639,38466,28714,749,20253,9800,535,532,1065,901,1556,13697,9916,78,39031,13903,49407],
      [49406,19703,796,281,18308,8869,267,263,19703,796,281,23435,281,272,269,273,264,49407],
      [49406,59,263,18707,59,264,314,36183,316,346,19249,348,283,6946,285,49407],
      [49406,49407],
      [49406,49407],
      [49406,158,239,510,158,227,371,33613,41175,149,352,149,353,7410,356,7410,357,6874,243,361,49407],
      [49406,3447,41096,37703,267,648,4039,268,324,267,34098,13040,1511,520,72,2608,267,22066,267,341,268,10127,19923,49407],
    ],
    bpe: [
      '103db81ff4a9a1bfd844adff481be8b36c0d10f6495ef0b908c080dfd1e1be9d34f8d25df3c9f1d396aa2b60270066ea',
      'c8e32b7c54e50645b64c543366c1a76c0d5768dda197177d671e9c00477bb24586be3cbcd2df0969f531351f243af917',
      'ca9246513ce6a8b04af3c200fa8f8e39aa59200a03e1c4a99b79ec7deff8b575e3a95798417bd3c61e33129546b051e6',
      'e0a51539928e5ee4f4087996fc06b72ec44cb0d69197c9f2b16d13f8f4131ab3e2dc7e00c9a2629aea95d216ae4ed11b',
      'dc1ff71ab3f6d2638dda11aa30009a0fd41dba329b77041bdb4838a21fcafd346994310148a9c0e8407e44051f3468ac',
      'cba8573fa409c7d12d7d587f136082a79bc099e48c09961acb2782a9c1fe8d8c3ebc3809b4e34e17bc4fe43dcb21b8f4',
      '39a903782a18168c0439145376a3eab55fec727ecf5d4545060c6677c253bc85b99cae8a2d6024ea0a475fec68dc0a7b',
      'b2d03b5c8ef368d45a428d7377260d75dc3529e8cb22737e095ba7ca846780844f9e457bc6310871f3574874207d1282',
      'c00a58cdccae23176e4a0ce486aad10df3a081e7fd0640c4e48071b0fd4b886e5d17a285c9e40ff8eae48a2d761de4c9',
      '49ac6ce13672276ef7cc7fe87535ef3b6e95b3c608f5a81414a5ce57d7a7358bcfc1fe68a366b9e9c89e136082380cdc',
      'd4a33993de485c30c63a14cc26966a8c71fcb57c7328aa641682e6491547e5e66ce1b58a8f718a4507997584a4943101',
      '65b27ae14ce69271608096f00b688aa4f6ae34f3c1ce8ced38da560d307006b7107901b6859bff9b47c725edb0028c58',
      '9bb9a0641a0938667b5dae84dffe4eeb6f440f3475bf2ad8124022cb0dfa8c71fcab3ffab329cc362d291d63287b561a',
      '1640a6a81779c01021a47de83eb441440a3906287b2a4fa96d99136abf9671d7c4e1cb23661549c4ef0b83fff85b5b54',
      'dec0ba016e9922f9cd8df37462dc24337111d2faa37845b5b07cf23b0e6763f6ba8714f747939b4c521ef99b7d369cac',
      'db5ecab7a4e3ca34296737e6437ea6c2c713f6a03e727c90691a9cc5c58f2259edb43931355312c5577613d8ba052cca',
      '55ca2d6b9dde08de79acf09ad9ae49b1e11d02f36f481536eeb2b7d65874c1630ff6744b1a966be5e8805cff739a4747',
      '0da691f6c5859d068748f9eea932e5b2e879209b96f1cc7e50c389093d76e95ab83ab689fecc6c76214374cba841f0d8',
      '3e6748c5dd35417314e4f3c199189166b15378bbf71d96cebd7ec44574a11c1ce6fa30b173af8e12ab8304c6551fcaaf',
      'b440fb62d90bd6b4507ca71134757f9a6542c8b9045c9e989f163a70337bd0eaabdf3cb257ac84acbff54354ab5a5ac1',
      '24e97e94777907c1ee1f18e5d3d7b8efece910b0af8281100f3300d2e29a0b06f6a33139be5c1be1844f9410657c8ba0',
      '6c327226a970312d103d170ee1a19e2edd03c782a778adc3d42da09dbcad83d63dfff5a1f4f9f4d83b12b5c180093356',
      '705c16044c9e0d6f68d6328a4ca1c780cb6bdfbe475931a3b1e17965fe2c13d8ba60d0695d8309b55f74f9df9a50da61',
      '1261371d0272d29ac0f4ccb9912d3faa5c11ba55ddb4c5d28438dc641d28d90d2bd362ac58739ba49d242ce898e6fc33',
      '40b06dcef8f20ccaa633753061bbad3544d00368c3b5b9938fc78070b7fa442874b71b5aaf2b1181045651ef351f7a92',
      '307a3b93baf4942cec1eaa07511a2a18d198da027d5515aaeb9c885417c875df6d81fd658c72c901e3f5059f6eebbae8',
      '30e7502ad87f450e4ddd746f26a55c663190c0c420aa7a144d9afe3fc005f3dc7b82b04ad8732de0c27403674d64b732',
      'bfb086da33ed18e7ee53d94e15a9f38fda50464082157d2e3892954017797a022f82c19fa10d78cb114bd2e8d4760ae9',
      'd8dd261ba54cd9162ab18ffb264532b911fe8cc06e9cb88a1bbacc878e5cd9aacd41a661a4f6ae9b3881321873d39aca',
      '483e24c4126ad55804aa34f9729e822685fd7e579c1caa6e87ce1c968ab3f0fda1dc7f496671f6d8d4d0a7e5529b2b82',
      '74683257de7c3f38ce0bf80b3d52f9ec9eddba9e6aaed15be3ea3ada77bf6549439e2c9e1fac73cdfc8918d198a59f0e',
      '668574bf54f8ddb36d419d5bfafbb843020c39ab430e553af4949e2410c6e75e066f1d7b60f33c5937bd965d4ea2ed8f',
      'da33edeebb9e7e7026ff29c918982fc37b2ec983ac3630fbd5d74a3e97e0836b99f253b5137e195006594c5433ea53fc',
      '7c13533ebcf89002fb4320b6d11d35fb9a3e6621b84e34a89df2dce15595a65e88f66e32b6008e9344294875bfb46584',
      '81ddead7a61ef235f949960859689791853de76c65682ee7759acc6912c673814cec2bac61bc9fe85ed9adf0d48dd7fc',
      '5130338f20340edc2fe0b2751b6a3da6c0157ef1cb83fc93515017c8fb5e637e71fe339d0b4e804e82f08654a18314a2',
      'f2757fdd315ee5a2ca79effd10bf5f655b527aea3e8e13e8ad7f87d376a533cab8b36d67960c3846235a8e8e69cfcfe3',
      '5d8a366c1121bfd124d90d2bd2f520bf9f3f558b0ee315dcdd4a067859167e022e84381b859f546bb7762d4b06629307',
      'adea1f13116a0664c40d693cb21f0b8c71fc63212952607bbba8e22cefc0a2963c413f7a2d9ddae82d13e06f796b3b6b',
      'e1e8d2fb794bf5a8149767c04779496f746868d45aef3d304a88a2b6c3b72564c7cd611c3642f40bb0abc98f8cf7d677',
      '1b500583d0ddbf85b977adf4a13198382e2d17b06537ddea1af5196e06703d4bba81f88cde24af6444a661bacbb4443b',
      'bdc48a199c859348e1eddba8ec223c3304784e426664dce1cee0736fc559c8680bf81f8134facdfed1fc295530fe3f15',
      '4ca2aff31b08412eed747ad6b79388f699a5e04a2f37c303fe473a3474f4b6e9159e087e727ecfe7148ad90d2bc45034',
      '9a148a76fe84bf73e868c578b75a6288ea963339c90dcf31cf9a821747585e0e376c18ab46d17148f42050c6c8ecee59',
      '738861eb9579953d7eb8f80f87f0990ccf51244eac6cbfc11ff5b7210b15180688f5ffb97bbd364bea9d09961a019bd1',
      'cfd2359e0ad5da33ed5e4346e5a325fbbc395c7faea44e2d496bc4376fa56ec5bed064fd6a3aa636aac28f968ad1a50a',
      '80c813479a63e8630461614d0043a457086bb3a4bba54264191d434df05d9b3ccdceb2dd3e2fc2b78f97f5a814e920f9',
      'e92ad42f21c78462d7e1d8a2448edc93fcc13e4dc7ebb0ab8607a6915afc6238ba57fd805044f4c23766c2022ab0c0e2',
      '39ab6b9604dbf5a81497c5594c8b79ee899db3976f2aa0462ed4bee863c542cc31fb1a01cba5bd2db93ec68745120ac7',
      '6fd1e791412f64195f182594306db94718244f7685b9df0e1c07a89b28d2d942636572ea4aa093d3e3d54c13e693e754',
    ].join(''),
    sdxl: [
      'e3d47a0b9c62185684d4203fd81f1fd902c839c5757b3634280900786ef581dd00aaa6a9365489da5e13c9c94be5f197',
      'cd14bab6a535241f502d37032762789c6879a8749053221939332285407dffd55c19a57e1cbd8a729ba3d11346dd59f6',
      '79c5c55452d6eed201327f9ed30c7b8c7f41de6b354b59171f6b58e8964590f50e1a1f85c39db4d298cfb54eee0f5e3f',
      'dea165e1a63609cbf416923ee6f44b4af78a08d8e22a2d77f2c7528715285863527dc9459f506f29798b86f090a5bd42',
      '9d6131898d1462260d8967963639f5ec75930595579352914ecc47566953c0de0a99c7f44f7d388c896a2189e21e81e7',
      '623fee28c5a6c3ed1be61db97dc5b3b24e7d4380bc5255000c0fe64d9ec79bc9dc7ec8b9c8bc641fba05633265c7b92b',
      'c05dbd96ede99ead3893f8532de2d47beda7f161139a733b5529b6095ca5dbb84923f6ca504936db229ad59d73772252',
      'f9975a5bb76cf5028513751b096a6da0c40a04aeabda9f6d05d75a688c0d883299517eebf1fc630dae2476e2faefef2c',
      '33fd865156345e5bdb5c46629cb98bdd72628dd00921254e4d8e6314f9c66cc4beba15d5a0a077e293d26246e122f921',
      '6124c41b046306c3f1dc4582ba6fbbf84f002b7dcb94d9fd141ccd35a9fb93f3c1643d6209e91c561f00bbfe8b31bd1b',
      '6825c41686424642d06a47706f5fc4e8c5fff3641152638c921c5133e177ad2ab4f0f6d655bccacfe4881b9292a899dd',
      '00074f1329521ff3348bd280fe95e6c2ef28c0fc97956275ed51c9b4b0fe25d6bd2c63a22d96593da7fc4dc30e221901',
      'bbaed0f553ede9a1dc0b501fd2d2237cd10618fdb56f36d5537e06240a7fcfe2687b88e08ff057253256d513f5657caa',
      'a1f8f1c7501e8fdf003f8f1503d9d93c98a1a0840a340f52867f0fec942a348bbe0050fad02d464b5da1eea81ec2fadb',
      '1492535f497cd8ab3d199d05e0ff46a1a8ba7eac7762f7a3092463bdf5e733c04d47e6b0c0f418c601efb6cf7cbc70b1',
      'b50b7b8cd36505a3370bbb5f628045de8ca5cc663d3c4a6256d1bf22e4f205939fcf27d43a4d86a053645835ee8f2115',
      '5c67c39e38d7ac0d115bd3344e8aeac13280836344d78e93f3b078898a6e71eeb06a6c8b3d63ae532a06514baf8fcffd',
      'eb20b2c6de9ee6fb1c302f5c4f4650c1d6dc625f8eb0770dcb4c9809f467fb89e8c14e096de639a5e36629db743ca12d',
      'cac6c7ff178e8dc664a628ab3fad521a7c2a8be516f7c451df587c2c68767b242fa9ad2d',
    ].join(''),
  },
};
/*GOLDEN-END*/

if (GOLDEN) {
  for (const mode of ['fast', 'slow']) {
    clip.setClipMode(mode);
    GOLDEN.tricky.forEach((s, i) => {
      eq('oracle ' + mode + ' ids: ' + J(s), clip.clipEncode(s), GOLDEN[mode].tricky[i]);
    });
  }
  // ---------------------------------------------------------------------------
  // 6. HF oracle fuzz corpus (hashes)
  // ---------------------------------------------------------------------------
  const corpus = makeCorpus(GOLDEN.fuzz.nBpe, GOLDEN.fuzz.nSdxl, GOLDEN.fuzz.seed);
  for (const mode of ['fast', 'slow']) {
    clip.setClipMode(mode);
    const exp = GOLDEN[mode];
    let bad = 0;
    corpus.bpe.forEach((s, i) => {
      if (h6(J(clip.clipEncode(s))) !== exp.bpe.slice(6 * i, 6 * i + 6)) {
        if (bad++ < 5) ok('fuzz ' + mode + ' bpe #' + i + ' ' + J(s), false);
      }
    });
    ok('fuzz ' + mode + ': ' + corpus.bpe.length + ' random strings match HF CLIPTokenizer', bad === 0, bad + ' mismatches');
    bad = 0;
    corpus.sdxl.forEach((s, i) => {
      if (h6(canonSDXL(clip.encodeSDXLRaw(s))) !== exp.sdxl.slice(6 * i, 6 * i + 6)) {
        if (bad++ < 5) ok('fuzz ' + mode + ' sdxl #' + i + ' ' + J(s.slice(0, 120)), false);
      }
    });
    ok('fuzz ' + mode + ': ' + corpus.sdxl.length + ' random weighted prompts match ComfyUI SDXLTokenizer', bad === 0, bad + ' mismatches');
  }
  clip.setClipMode('fast');
} else {
  ok('oracle goldens present', false, 'GOLDEN block is empty');
}

// ---------------------------------------------------------------------------
// Timing (informational)
// ---------------------------------------------------------------------------
clip.setClipMode('fast');
const TYPICAL = 'masterpiece, best quality, amazing quality, very aesthetic, absurdres, 1girl, solo, ' +
  '(long hair:1.2), silver hair, blue eyes, looking at viewer, smile, (school uniform), pleated skirt, ' +
  'outdoors, cherry blossoms, (sunlight:0.8), upper body';
const nTok = clip.clipEncodeNoSpecial(TYPICAL).length;
function timeIt(fn, n) {
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < n; i++) fn(i);
  return Number(process.hrtime.bigint() - t0) / 1e6 / n;
}
const warm = timeIt(() => clip.encodeSDXL(TYPICAL), 2000);
// Memo misses: every word is new (random letters), about the same token count.
const RT = mulberry32(99);
const freshPrompts = Array.from({ length: 500 }, () => Array.from({ length: 15 }, () =>
  Array.from({ length: 4 + Math.floor(RT() * 5) }, () => String.fromCharCode(97 + Math.floor(RT() * 26))).join('')).join(', '));
const freshTok = freshPrompts.reduce((a, p) => a + clip.clipEncodeNoSpecial(p).length, 0) / freshPrompts.length;
const fresh = timeIt((i) => clip.encodeSDXL(freshPrompts[i] + ' qq' + i), freshPrompts.length);

console.log('');
for (const f of failures) console.log('FAIL ' + f);
console.log(`clip_tokenize: ${pass} passed, ${fail} failed`);
console.log(`timing: typical prompt (${nTok} tokens) encodeSDXL ${warm.toFixed(4)} ms with word memo; ` +
  `all-new words (~${Math.round(freshTok)} tokens) ${fresh.toFixed(4)} ms`);
process.exit(fail ? 1 : 0);
