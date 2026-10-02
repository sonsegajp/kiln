'use strict';
// Booru search and download, ported from BooruGrab: Danbooru, Gelbooru 0.2 (reading the web pages when the API wants a
// key, as Grabber does), Moebooru, e621 and Shimmie (paheal) sites. Posts carry their tags by category; a download
// saves the original file and a .txt caption (tags, underscores to spaces) named <Site>_<id>.
const fs = require('fs');
const path = require('path');

const UA = 'BooruGrab/1.0 (personal image downloader; Kiln)';
const BROWSER_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) BooruGrab/1.0';
const IMAGE_EXT = new Set(['jpg', 'jpeg', 'png', 'gif', 'webp']);
// Posts tagged as depicting minors are never shown or downloaded unless rated General. Built in, not a setting.
const MINOR_TAGS = new Set(['loli', 'shota', 'lolicon', 'shotacon', 'oppai_loli', 'toddler', 'child', 'children', 'female_child', 'male_child',
  'young', 'cub', 'baby', 'kindergarten_uniform', 'elementary_school_student']);
const ENGINES = [
  { id: 'danbooru', name: 'Danbooru' }, { id: 'gelbooru', name: 'Gelbooru 0.2' }, { id: 'moebooru', name: 'Moebooru' },
  { id: 'e621', name: 'e621' }, { id: 'shimmie', name: 'Shimmie (paheal)' },
];
const DEFAULT_SITES = [
  { id: 'danbooru', name: 'Danbooru', engine: 'danbooru', url: 'https://danbooru.donmai.us', user: '', key: '', enabled: true },
  { id: 'safebooru', name: 'Safebooru', engine: 'gelbooru', url: 'https://safebooru.org', user: '', key: '', enabled: true },
  { id: 'konachan', name: 'Konachan', engine: 'moebooru', url: 'https://konachan.com', user: '', key: '', enabled: true },
  { id: 'yandere', name: 'Yande.re', engine: 'moebooru', url: 'https://yande.re', user: '', key: '', enabled: true },
  { id: 'gelbooru', name: 'Gelbooru', engine: 'gelbooru', url: 'https://gelbooru.com', user: '', key: '', enabled: false },
  { id: 'rule34', name: 'Rule34', engine: 'gelbooru', url: 'https://api.rule34.xxx', user: '', key: '', enabled: false },
  { id: 'e621', name: 'e621', engine: 'e621', url: 'https://e621.net', user: '', key: '', enabled: false },
];
const HTML_PAGE = 42;  // posts per page on Gelbooru's web pages

const qs = (kv) => Object.entries(kv).filter(([, v]) => v != null && String(v) !== '').map(([k, v]) => k + '=' + encodeURIComponent(String(v))).join('&');
const split = (s) => String(s || '').split(/[ \t\n]+/).filter(Boolean);
const extOf = (u) => { const m = /\.([A-Za-z0-9]+)$/.exec(String(u || '').split('?')[0]); return m ? m[1].toLowerCase() : ''; };
const abs = (b, u) => !u ? null : u.startsWith('//') ? 'https:' + u : u.startsWith('/') ? b + u : u;
const link = (b, h) => h.startsWith('http') ? h : h.startsWith('//') ? 'https:' + h : h.startsWith('/') ? b + h : b + '/' + h;
const imageBase = (s) => s.url.replace('://api.', '://');
const apiBase = (s) => s.url.replace(/^(https?:\/\/)(?:www\.)?(rule34\.xxx)$/, '$1api.$2');  // rule34.xxx answers its API only on api.
const htmlDecode = (t) => String(t || '').replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos|#39);/gi, (m, e) => {
  const x = e.toLowerCase();
  if (x[0] === '#') return String.fromCodePoint(x[1] === 'x' ? parseInt(x.slice(2), 16) : parseInt(x.slice(1), 10));
  return { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" }[x] || m;
});
const str = (o, k) => o && typeof o === 'object' && o[k] != null ? String(o[k]) : null;
const int = (o, k) => { const n = parseInt(str(o, k), 10); return Number.isFinite(n) ? n : 0; };
const arr = (o) => Array.isArray(o) ? o : [];
const strList = (o) => arr(o).map(String).filter(Boolean);

// g general, s sensitive, q questionable, e explicit (Moebooru / e621 "s" means safe)
function rating(r, sIsSafe) {
  switch (String(r || '').toLowerCase()) {
    case 'g': case 'general': case 'safe': return 'g';
    case 's': return sIsSafe ? 'g' : 's';
    case 'sensitive': return 's';
    case 'q': case 'questionable': return 'q';
    case 'e': case 'explicit': return 'e';
  }
  return '?';
}

function searchUrl(s, q, page, limit) {
  switch (s.engine) {
    case 'danbooru': return s.url + '/posts.json?' + qs({ tags: q, page, limit, login: s.user, api_key: s.key });
    case 'gelbooru': return apiBase(s) + '/index.php?' + qs({ page: 'dapi', s: 'post', q: 'index', json: 1, tags: q, pid: page - 1, limit, api_key: s.key, user_id: s.user });
    case 'moebooru': return s.url + '/post.json?' + qs({ tags: q, page, limit, login: s.user, api_key: s.key });
    case 'e621': return s.url + '/posts.json?' + qs({ tags: q, page, limit, login: s.user, api_key: s.key });
    case 'shimmie': return s.url + '/api/danbooru/find_posts?' + qs({ tags: q, limit, page });
  }
  throw new Error('unknown engine ' + s.engine);
}

function parse(s, d) {
  const list = [], b = imageBase(s);
  switch (s.engine) {
    case 'danbooru':
      for (const p of arr(d)) {
        const file = str(p, 'file_url');
        list.push({
          id: int(p, 'id'), file, sample: str(p, 'large_file_url') || file, preview: str(p, 'preview_file_url'),
          w: int(p, 'image_width'), h: int(p, 'image_height'), ext: (str(p, 'file_ext') || extOf(file)).toLowerCase(),
          md5: str(p, 'md5'), rating: rating(str(p, 'rating'), false), source: str(p, 'source') || '', page: s.url + '/posts/' + int(p, 'id'),
          tags: {
            character: split(str(p, 'tag_string_character')), copyright: split(str(p, 'tag_string_copyright')),
            artist: split(str(p, 'tag_string_artist')), general: split(str(p, 'tag_string_general')), meta: split(str(p, 'tag_string_meta')),
          },
        });
      }
      break;
    case 'gelbooru': {
      const posts = d && !Array.isArray(d) && typeof d === 'object' ? arr(d.post) : arr(d);
      for (const p of posts) {
        const dir = str(p, 'directory'), img = str(p, 'image');
        const stem = img == null ? '' : img.replace(/\.[^.]+$/, '');
        const file = abs(b, str(p, 'file_url')) || (dir != null && img != null ? `${b}/images/${dir}/${img}` : null);
        let sample = abs(b, str(p, 'sample_url'));
        if (!sample) sample = str(p, 'sample') === '1' || str(p, 'sample') === 'True' || str(p, 'sample') === 'true' ? `${b}/samples/${dir}/sample_${stem}.jpg` : file;
        list.push({
          id: int(p, 'id'), file, sample,
          preview: abs(b, str(p, 'preview_url')) || (dir != null && img != null ? `${b}/thumbnails/${dir}/thumbnail_${stem}.jpg` : null),
          w: int(p, 'width'), h: int(p, 'height'), ext: extOf(img || file), md5: str(p, 'hash') || str(p, 'md5'),
          rating: rating(str(p, 'rating'), false), source: str(p, 'source') || '',
          page: `${b}/index.php?page=post&s=view&id=${int(p, 'id')}`,
          tags: { general: split(htmlDecode(str(p, 'tags'))) },
        });
      }
      break;
    }
    case 'moebooru':
      for (const p of arr(d)) {
        const file = abs(s.url, str(p, 'file_url'));
        list.push({
          id: int(p, 'id'), file, sample: abs(s.url, str(p, 'sample_url') || str(p, 'jpeg_url')) || file,
          preview: abs(s.url, str(p, 'preview_url')), w: int(p, 'width'), h: int(p, 'height'),
          ext: (str(p, 'file_ext') || extOf(file)).toLowerCase(), md5: str(p, 'md5'),
          rating: rating(str(p, 'rating'), true), source: str(p, 'source') || '', page: s.url + '/post/show/' + int(p, 'id'),
          tags: { general: split(str(p, 'tags')) },
        });
      }
      break;
    case 'e621':
      for (const p of arr(d && d.posts)) {
        const f = p.file || {}, t = p.tags || {};
        list.push({
          id: int(p, 'id'), file: str(f, 'url'), sample: str(p.sample, 'url') || str(f, 'url'),
          preview: str(p.preview, 'url'), w: int(f, 'width'), h: int(f, 'height'),
          ext: (str(f, 'ext') || '').toLowerCase(), md5: str(f, 'md5'), rating: rating(str(p, 'rating'), true),
          source: strList(p.sources)[0] || '', page: s.url + '/posts/' + int(p, 'id'),
          tags: {
            character: strList(t.character), copyright: strList(t.copyright), artist: strList(t.artist), species: strList(t.species),
            general: strList(t.general), meta: [...strList(t.meta), ...strList(t.lore)],
          },
        });
      }
      break;
  }
  for (const p of list) p.site = s.id;
  return list;
}

function parseShimmie(s, xml) {
  const list = [];
  for (const m of xml.matchAll(/<(?:post|tag)\s([^>]*?)\/?>/g)) {
    const a = {};
    for (const am of m[1].matchAll(/(\w+)=(?:'([^']*)'|"([^"]*)")/g)) a[am[1]] = htmlDecode(am[2] !== undefined ? am[2] : am[3]);
    if (a.id == null || a.file_url == null) continue;
    const id = parseInt(a.id, 10) || 0;
    list.push({
      site: s.id, id, file: abs(s.url, a.file_url), sample: abs(s.url, a.file_url), preview: abs(s.url, a.preview_url),
      w: parseInt(a.width, 10) || 0, h: parseInt(a.height, 10) || 0, ext: extOf(a.file_name) || extOf(a.file_url), md5: a.md5 || null,
      rating: rating(a.rating, true), source: a.source || '', page: s.url + '/post/view/' + id,
      tags: { general: split(a.tags) },
    });
  }
  return list;
}

const allTags = (p) => Object.values(p.tags || {}).flat();
function keep(p, blacklist) {
  if (!p.html && (!p.file || !IMAGE_EXT.has(p.ext))) return false;
  const tags = allTags(p).map(t => t.toLowerCase());
  if (p.rating !== 'g' && tags.some(t => MINOR_TAGS.has(t))) return false;
  const bl = (blacklist || []).map(t => String(t).trim().toLowerCase().replace(/ /g, '_')).filter(Boolean);
  return !tags.some(t => bl.includes(t));
}

class ApiRefused extends Error { }

// one API request at a time per site, at least 400 ms apart
const lanes = new Map();
function inLane(id, fn) {
  const prev = lanes.get(id) || Promise.resolve();
  const run = prev.then(fn, fn);
  lanes.set(id, run.then(() => new Promise(r => setTimeout(r, 400)), () => new Promise(r => setTimeout(r, 400))));
  return run;
}
// Gelbooru-type sites whose API refused (key required): their web pages are read instead
const htmlMode = new Set();

async function get(url, headers, timeout = 60000) {
  return fetch(url, { headers: { 'User-Agent': UA, ...headers }, redirect: 'follow', signal: AbortSignal.timeout(timeout) });
}

async function searchApi(s, q, page, st) {
  const res = await get(searchUrl(s, q, page, st.limit), { Accept: 'application/json' });
  const text = await res.text();
  if (s.engine === 'shimmie' && res.ok && text.trimStart().startsWith('<')) {
    const raw = parseShimmie(s, text), kept = raw.filter(p => keep(p, st.blacklist));
    return { posts: kept, full: raw.length >= st.limit, hidden: raw.length - kept.length };
  }
  let data = null;
  try { data = text.trim().length === 0 ? [] : JSON.parse(text); } catch (_) { }
  if (!res.ok) {
    let msg = str(data, 'message') || str(data, 'reason') || str(data, 'error') || text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    if (msg.length > 160) msg = msg.slice(0, 160);
    const code = res.status;
    const t = `HTTP ${code}` + (msg ? ': ' + msg : '') + (code === 401 || code === 403 ? ' (this site needs your user ID and API key: Settings)' : '');
    if (code === 401 || code === 403 || code === 404) throw new ApiRefused(t);
    throw new Error(t);
  }
  if (data === null) throw new ApiRefused('the site did not return JSON (it may need an API key)');
  if (typeof data === 'string') throw new ApiRefused(data);  // rule34.xxx: "Missing authentication. ..."
  const raw = parse(s, data), kept = raw.filter(p => keep(p, st.blacklist));
  return { posts: kept, full: raw.length >= st.limit, hidden: raw.length - kept.length };
}

async function getPage(url) {
  const res = await get(url, { 'User-Agent': BROWSER_UA, Cookie: 'fringeBenefits=yup' });  // gelbooru.com: don't hide "fringe" posts
  const html = await res.text();
  if (/<title>[^<]*captcha/i.test(html) || res.status === 403)
    throw new Error("this site shows a CAPTCHA to programs, so it needs your user ID and API key (from your account's options page): Settings");
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return html;
}

async function searchHtml(s, q, page, st) {
  const b = imageBase(s);
  const html = await getPage(`${b}/index.php?page=post&s=list&tags=${encodeURIComponent(q.length ? q : 'all')}&pid=${(page - 1) * HTML_PAGE}`);
  const raw = [];
  for (const m of html.matchAll(/<a\b[^>]*\bid="p(\d+)"[^>]*\bhref="([^"]+)"[^>]*>\s*<img\b([^>]*)>/gi)) {
    const attrs = m[3];
    const src = htmlDecode((/\bsrc="([^"]+)"/.exec(attrs) || [])[1] || '');
    const title = htmlDecode((/\btitle="([^"]*)"/.exec(attrs) || [])[1] || '');
    const tags = [];
    let r = '?';
    for (const t of split(title)) {
      if (t.startsWith('rating:')) r = rating(t.slice(7), false);
      else if (!t.startsWith('score:')) tags.push(t);
    }
    raw.push({ site: s.id, id: parseInt(m[1], 10), html: true, preview: abs(b, src), rating: r, page: link(b, htmlDecode(m[2])), tags: { general: tags } });
  }
  const kept = raw.filter(p => keep(p, st.blacklist));
  return { posts: kept, full: raw.length >= HTML_PAGE, hidden: raw.length - kept.length };
}

// one page of results from one site: { posts, full (another page may follow), hidden (filtered out) }
function search(s, q, page, st) {
  return inLane(s.id, async () => {
    if (s.engine === 'gelbooru' && htmlMode.has(s.id)) return searchHtml(s, q, page, st);
    try { return await searchApi(s, q, page, st); }
    catch (e) {
      if (!(e instanceof ApiRefused) || s.engine !== 'gelbooru' || s.key) throw e;
      try { const r = await searchHtml(s, q, page, st); htmlMode.add(s.id); return r; }
      catch (h) { throw new Error(h.message + ' (API: ' + e.message + ')'); }
    }
  });
}

// a post read from the web pages: its original file, sample and size come from the post's own page
async function resolve(s, p) {
  if (!p.html || p.file) return;
  const html = await getPage(p.page);
  let m = /<a\b[^>]*href="([^"]+)"[^>]*>\s*Original image\s*</i.exec(html);
  if (!m) m = /property="og:image"\s+content="([^"]+)"/.exec(html);
  if (!m) throw new Error("could not find the image on the post's page");
  const b = imageBase(s);
  p.file = abs(b, htmlDecode(m[1]));
  p.ext = extOf(p.file);
  const sm = /id="image"[^>]*\bsrc="([^"]+)"/.exec(html);
  p.sample = sm ? abs(b, htmlDecode(sm[1])) : p.file;
  p.w = parseInt((/data-width="(\d+)"/.exec(html) || [])[1], 10) || p.w || 0;
  p.h = parseInt((/data-height="(\d+)"/.exec(html) || [])[1], 10) || p.h || 0;
}

const isImage = (p) => IMAGE_EXT.has(p.ext || '');

async function getBytes(s, url, timeout = 120000) {
  const res = await get(url, { Referer: imageBase(s) + '/' }, timeout);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return { buffer: Buffer.from(await res.arrayBuffer()), type: res.headers.get('content-type') || '' };
}

function caption(p, st) {
  const order = ['character', 'copyright', 'artist', 'species', 'general'];
  if (st.includeMeta) order.push('meta');
  const seen = new Set(), tags = [];
  for (const cat of order) for (const t of (p.tags && p.tags[cat]) || []) if (t && !seen.has(t)) { seen.add(t); tags.push(t); }
  if (st.tagFormat === 'raw') return tags.join(' ');
  return tags.map(t => /[A-Za-z0-9]/.test(t) ? t.replace(/_/g, ' ') : t).join(', ');  // ^_^ and >_< keep their underscores
}

const label = (s) => (String(s.name || '').replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '')) || s.id;

// the original file and its caption into dir as <Site>_<id>.<ext> / .txt; false when skipped (already there)
async function download(s, p, dir, st) {
  const stem = path.join(dir, `${label(s)}_${p.id}`);
  let file = `${stem}.${p.ext || 'jpg'}`;
  if (st.skipExisting && !p.html && fs.existsSync(file)) return { skipped: true, file };
  if (p.html) {
    if (st.skipExisting && [...IMAGE_EXT].some(e => fs.existsSync(`${stem}.${e}`))) return { skipped: true, file };
    await resolve(s, p);
    if (!isImage(p)) throw new Error(`not an image (${p.ext})`);
    file = `${stem}.${p.ext}`;
    if (st.skipExisting && fs.existsSync(file)) return { skipped: true, file };
  }
  const { buffer } = await getBytes(s, p.file);
  fs.writeFileSync(file + '.part', buffer);
  if (fs.existsSync(file)) fs.unlinkSync(file);
  fs.renameSync(file + '.part', file);
  fs.writeFileSync(stem + '.txt', caption(p, st) + '\n');
  if (st.writeJson) {
    fs.writeFileSync(stem + '.json', JSON.stringify({
      site: s.name, id: p.id, page: p.page, file: p.file, source: p.source, rating: p.rating, width: p.w, height: p.h, md5: p.md5, tags: p.tags,
    }, null, 2));
  }
  return { skipped: false, file };
}

// is this URL on the site (or its image hosts, e.g. cdn.donmai.us, img3.gelbooru.com, r34i.paheal-cdn.net)?
function sameSite(s, url) {
  let u, h;
  try { u = new URL(url); h = new URL(s.url); } catch (_) { return false; }
  if (!/^https?:$/.test(u.protocol)) return false;
  const base = (host) => host.split('.').slice(-2).join('.');
  const name = base(h.hostname).split('.')[0];
  return base(u.hostname) === base(h.hostname) || base(u.hostname).startsWith(name);
}

module.exports = { ENGINES, DEFAULT_SITES, search, resolve, download, getBytes, caption, isImage, sameSite, label };
