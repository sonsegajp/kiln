# Independent fp32 CPU reference for engine/src/upscale.cu: a plain-PyTorch RRDBNet (old ESRGAN
# key naming, as shipped in 4x-AnimeSharp), compared against the harness's --dump output.
#   python upscale_ref.py <model.safetensors> <in.ppm> <cuda_dump.bin> [more dumps...]
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

torch.set_grad_enabled(False)
model, img_path, dumps = sys.argv[1], sys.argv[2], sys.argv[3:]
sd = {k: v.float() for k, v in load_file(model).items()}

def conv(x, name):
    return F.conv2d(x, sd[name + ".weight"], sd[name + ".bias"], padding=1)

def lrelu(x):
    return F.leaky_relu(x, 0.2)

def rdb(x, p):
    feats = [x]
    for k in range(1, 5):
        feats.append(lrelu(conv(torch.cat(feats, 1), f"{p}.conv{k}.0")))
    return x + 0.2 * conv(torch.cat(feats, 1), f"{p}.conv5.0")

def rrdb(x, p):
    y = rdb(rdb(rdb(x, p + ".RDB1"), p + ".RDB2"), p + ".RDB3")
    return x + 0.2 * y

nb = 0
while f"model.1.sub.{nb}.RDB1.conv1.0.weight" in sd:
    nb += 1
tail = sorted(int(k.split(".")[1]) for k in sd if k.count(".") == 2 and k.endswith(".weight") and int(k.split(".")[1]) >= 2)
ups, hr, last = tail[:-2], tail[-2], tail[-1]

def net(x):
    feat = conv(x, "model.0")
    t = feat
    for i in range(nb):
        t = rrdb(t, f"model.1.sub.{i}")
    feat = feat + conv(t, f"model.1.sub.{nb}")
    for u in ups:
        feat = lrelu(conv(F.interpolate(feat, scale_factor=2, mode="nearest"), f"model.{u}"))
    return conv(lrelu(conv(feat, f"model.{hr}")), f"model.{last}")

img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
x = torch.from_numpy(img).permute(2, 0, 1)[None].contiguous()
t0 = time.time()
ref = net(x).clamp(0, 1)[0].numpy().astype(np.float64)
print(f"reference: nb={nb} ups={ups} hr={hr} last={last}, {x.shape[3]}x{x.shape[2]} -> {ref.shape[2]}x{ref.shape[1]} in {time.time() - t0:.1f} s")

for d in dumps:
    out = np.fromfile(d, dtype=np.float32).reshape(ref.shape).astype(np.float64)
    diff = out - ref
    rel = np.linalg.norm(diff) / np.linalg.norm(ref)
    q = np.count_nonzero(np.rint(out * 255) != np.rint(ref * 255))
    print(f"{d}: rel_l2 {rel:.3e}  max_abs {np.abs(diff).max():.3e}  mean_abs {np.abs(diff).mean():.3e}  "
          f"8-bit values differing {q}/{ref.size} (max {np.abs(np.rint(out * 255) - np.rint(ref * 255)).max():.0f} levels)")
