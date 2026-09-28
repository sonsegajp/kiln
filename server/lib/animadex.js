'use strict';
// Animadex: the character index built by animadex.net (~36,500 characters the Anima model knows, each
// with its trigger - "ganyu (genshin impact), genshin impact" - and the Danbooru tags for its look and
// usual outfit). Kept in models/animadex/characters.json, downloaded on first use; searched here so
// the browser never has to load the 32 MB list.
//   GET  /api/animadex/status            -> { ready, count, downloading, error }
//   POST /api/animadex/download          fetch the index
//   GET  /api/animadex/search?q=&limit=  -> { items: [{ slug, name, series, trigger, look, outfit, thumb, count, loras }] }

const fs = require('fs');
const path = require('path');
const https = require('https');

const SOURCE = 'https://raw.githubusercontent.com/wrt122311/ComfyUI-Animadex-Node/master/animadex_top_characters.json';

// clothing and accessories: these tags are the character's usual outfit, the rest is how they look
const OUTFIT = /\b(dress|shirt|skirt|jacket|coat|gloves?|sleeves?|uniform|hat|cap|beret|boots?|shoes|heels|sandals|sneakers|loafers|thighhighs|kneehighs|pantyhose|stockings|socks|leggings|leotard|bodysuit|bodystocking|kimono|yukata|hakama|armor|armour|cape|cloak|scarf|ribbon|bow|bowtie|necktie|tie|choker|collar|bell|headband|hairband|hair ornament|hair ribbon|hair bow|hairclip|hairpin|earrings?|jewelry|necklace|pendant|belt|apron|hood|hoodie|sweater|cardigan|vest|shorts|pants|trousers|jeans|swimsuit|bikini|bra|panties|lingerie|mask|crown|tiara|helmet|veil|sash|obi|detached|serafuku|sailor|blazer|gauntlets|bracelet|wristband|wrist cuffs|garter|headwear|headgear|eyewear|goggles|epaulettes|cuffs|brooch|corset|tabard|robe|cloak|tunic|overalls|suspenders|maid|nurse|school uniform|clothing|clothes|outfit|bare shoulders|cleavage cutout|clothing cutout|off-shoulder|sleeveless|strapless|turtleneck|capelet|gem|jewel|button|zipper|frills|lace|fur trim|wings? ornament)\b/i;
const COUNT = /^(\d\+?)?(girls?|boys?|others?)$|^solo$|^multiple (girls|boys)$/i;

function createAnimadex({ modelsDir, log }) {
  const DIR = path.join(modelsDir, 'animadex');
  const FILE = path.join(DIR, 'characters.json');
  let list = null, loading = null, downloading = null, error = null;

  function compact(c) {
    const look = [], outfit = [];
    for (const t of Array.isArray(c.tags) ? c.tags : []) {
      if (typeof t !== 'string' || !t.trim()) continue;
      (OUTFIT.test(t) && !COUNT.test(t) ? outfit : look).push(t.trim());
    }
    return {
      slug: String(c.slug || ''), name: String(c.name || c.slug || '?'), series: String(c.copyright_name || c.copyright || ''),
      trigger: String(c.trigger || '').trim(), look, outfit, thumb: typeof c.thumb_url === 'string' ? c.thumb_url : '',
      count: Number(c.count) || 0, loras: Array.isArray(c.loras) ? c.loras.length : 0,
      key: `${c.name || ''} ${c.trigger || ''} ${c.copyright_name || ''} ${c.slug || ''}`.toLowerCase(),
    };
  }

  function load() {
    if (list) return Promise.resolve(list);
    if (loading) return loading;
    loading = fs.promises.readFile(FILE, 'utf8').then((s) => {
      const raw = JSON.parse(s.replace(/^﻿/, ''));
      const arr = Array.isArray(raw) ? raw : Array.isArray(raw.characters) ? raw.characters : [];
      list = arr.filter((c) => c && !c.is_hidden && c.trigger).map(compact).sort((a, b) => b.count - a.count);
      log(`animadex: ${list.length} characters loaded`);
      return list;
    }).finally(() => { loading = null; });
    return loading;
  }

  function download() {
    if (downloading) return downloading;
    error = null;
    downloading = new Promise((resolve, reject) => {
      fs.mkdirSync(DIR, { recursive: true });
      const tmp = FILE + '.part';
      const get = (url, hops = 0) => https.get(url, { headers: { 'User-Agent': 'Kiln' } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && hops < 5) { res.resume(); return get(res.headers.location, hops + 1); }
        if (res.statusCode !== 200) { res.resume(); return reject(new Error(`download failed: HTTP ${res.statusCode}`)); }
        const out = fs.createWriteStream(tmp);
        res.pipe(out);
        out.on('finish', () => out.close(() => {
          try { JSON.parse(fs.readFileSync(tmp, 'utf8')); } catch (e) { return reject(new Error('the downloaded list is not valid JSON')); }
          fs.renameSync(tmp, FILE);
          list = null;
          resolve();
        }));
        out.on('error', reject);
      }).on('error', reject);
      get(SOURCE);
    }).then(() => load()).then((l) => { log(`animadex: downloaded ${l.length} characters`); return l; })
      .catch((e) => { error = e.message; log('animadex: ' + e.message); throw e; })
      .finally(() => { downloading = null; });
    return downloading;
  }

  function search(q, limit) {
    const words = String(q || '').toLowerCase().replace(/[_,]+/g, ' ').split(/\s+/).filter(Boolean);
    const out = [];
    for (const c of list) {
      if (words.length && !words.every((w) => c.key.includes(w))) continue;
      out.push(c);
      if (!words.length && out.length >= limit) break;
    }
    if (words.length) {
      const q0 = words.join(' ');
      // whole-word name matches first ("miku" -> Hatsune Miku before Mikuma), then name prefixes, then the rest
      const esc = q0.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const word = new RegExp('(^|[^a-z0-9])' + esc + '($|[^a-z0-9])');
      const rank = (c) => { const n = c.name.toLowerCase(); return word.test(n) ? 0 : n.startsWith(q0) ? 1 : n.includes(q0) ? 2 : 3; };
      out.sort((a, b) => rank(a) - rank(b) || b.count - a.count);
    }
    return out.slice(0, limit).map(({ key, ...rest }) => rest);
  }

  async function handle(req, res, u, p, m, sendJSON) {
    if (!p.startsWith('/api/animadex/')) return false;
    const rest = p.slice('/api/animadex/'.length);
    try {
      if (rest === 'status' && m === 'GET') {
        const ready = !!list || fs.existsSync(FILE);
        if (ready && !list) await load();
        sendJSON(res, 200, { ready, count: list ? list.length : 0, downloading: !!downloading, error });
        return true;
      }
      if (rest === 'download' && m === 'POST') {
        await download();
        sendJSON(res, 200, { ready: true, count: list.length });
        return true;
      }
      if (rest === 'search' && m === 'GET') {
        if (!list) {
          if (!fs.existsSync(FILE)) { sendJSON(res, 409, { error: 'the character list is not downloaded yet' }); return true; }
          await load();
        }
        const limit = Math.max(1, Math.min(100, parseInt(u.searchParams.get('limit') || '40', 10) || 40));
        sendJSON(res, 200, { items: search(u.searchParams.get('q'), limit) });
        return true;
      }
      sendJSON(res, 404, { error: 'no such endpoint' });
    } catch (e) {
      sendJSON(res, 500, { error: e.message });
    }
    return true;
  }

  return { handle };
}

module.exports = { createAnimadex };
