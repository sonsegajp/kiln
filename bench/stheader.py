import json, struct, sys, urllib.request, collections

def header_from_url(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    return json.loads(urllib.request.urlopen(req).read())

def header_from_file(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

src = sys.argv[1]
h = header_from_url(src) if src.startswith("http") else header_from_file(src)
meta = h.pop("__metadata__", None)
if meta: print("META", json.dumps(meta)[:400])
dtypes = collections.Counter(v["dtype"] for v in h.values())
print("tensors", len(h), dict(dtypes))
mode = sys.argv[2] if len(sys.argv) > 2 else "all"
import re
seen = set()
for k in sorted(h):
    if mode == "collapse":
        kk = re.sub(r"\.(\d+)\.", lambda m: ".N." if int(m.group(1)) > 0 else ".0.", k)
        if kk in seen: continue
        seen.add(kk)
        if ".N." in kk: continue
    print(k, h[k]["dtype"], h[k]["shape"])
