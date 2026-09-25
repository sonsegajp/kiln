# Image-level comparison of the i8_real streams: PSNR of each decoded final image against the fp32-linear
# reference stream (image_rrrrrr.npy) and against the ComfyUI golden image, plus PNGs and an x8 diff sheet.
# usage: python i8_compare.py [dir=i8_out]
import glob, os, sys
import numpy as np
from PIL import Image

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "i8_out")
golden_path = os.path.join(os.path.dirname(__file__), "..", "ref", "golden", "image.npy")
imgs = {os.path.basename(p)[6:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(d, "image_*.npy")))}
if "rrrrrr" not in imgs:
    sys.exit("no reference image_rrrrrr.npy in " + d)
ref = imgs["rrrrrr"]
gold = np.load(golden_path) if os.path.exists(golden_path) else None


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def q8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


print(f"{'stream':8} {'PSNR vs ref':>12} {'mean|d| (8-bit)':>16} {'p99|d| (8-bit)':>15} {'PSNR vs golden':>15}")
order = sorted(imgs, key=lambda k: (k != "rrrrrr", k))
for k in order:
    a = imgs[k]
    dd = np.abs(q8(a).astype(int) - q8(ref).astype(int))
    pg = psnr(a, gold) if gold is not None and gold.shape == a.shape else float("nan")
    print(f"{k:8} {psnr(a, ref):12.2f} {dd.mean():16.3f} {np.percentile(dd, 99):15.1f} {pg:15.2f}")
    Image.fromarray(q8(a)).save(os.path.join(d, f"image_{k}.png"))

# contact sheet: images on top, x8 |diff| vs reference below
tiles = [q8(imgs[k]) for k in order]
diffs = [q8(np.clip(np.abs(imgs[k] - ref) * 8, 0, 1)) for k in order]
sheet = np.concatenate([np.concatenate(tiles, 1), np.concatenate(diffs, 1)], 0)
Image.fromarray(sheet).save(os.path.join(d, "sheet.png"))
print("columns:", ", ".join(order), "-> sheet.png (bottom row: |diff vs rrrrrr| x8)")
