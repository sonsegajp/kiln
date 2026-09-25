# CPU-only image-level view of the i8_real streams (no VAE, no GPU): each final latent (model space, sigma = 0,
# so it is x0) is mapped to RGB with the engine's live-preview matrix (ComfyUI latent_rgb_factors for Wan2.1,
# copied from engine/src/main.cpp) and compared with the fp32-linear reference stream.
# usage: python i8_preview.py [dir=i8_out]  -> prints metrics, writes <dir>/preview_sheet.png
import glob, os, sys
import numpy as np
from PIL import Image

RGB = np.array([
    [-0.1299, -0.1692, 0.2932], [0.0671, 0.0406, 0.0442], [0.3568, 0.2548, 0.1747], [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328], [0.0296, -0.0956, -0.0665], [-0.3477, -0.4059, -0.2925], [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364], [-0.1293, 0.0740, 0.1636], [0.0680, 0.3019, 0.1128], [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699], [0.0060, -0.0633, 0.0005], [0.3477, 0.2275, 0.2950], [0.1984, 0.0913, 0.1861]], np.float32)
BIAS = np.array([-0.1835, -0.0868, -0.3360], np.float32)

here = os.path.dirname(os.path.abspath(__file__))
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "i8_out")
lat = {os.path.basename(p)[13:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(d, "latent_final_*.npy")))}
gp = os.path.join(here, "..", "ref", "golden", "latent_final.npy")
if os.path.exists(gp):
    lat["golden"] = np.load(gp)
ref = lat["rrrrrr"]


def preview(z):  # [16,H,W] -> [H,W,3] in [0,1]
    return np.clip((np.einsum("chw,ck->hwk", z, RGB) + BIAS + 1) / 2, 0, 1)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def corr(a, b):  # mean per-channel spatial correlation of the latents (1 = same picture)
    return float(np.mean([np.corrcoef(a[c].ravel(), b[c].ravel())[0, 1] for c in range(a.shape[0])]))


pref = preview(ref)
order = ["golden", "rrrrrr"] + sorted(k for k in lat if k not in ("golden", "rrrrrr"))
order = [k for k in order if k in lat]
print(f"{'stream':8} {'latent rel_l2 vs ref':>21} {'latent corr':>12} {'preview PSNR vs ref':>20}")
for k in order:
    print(f"{k:8} {rel(lat[k], ref):21.3f} {corr(lat[k], ref):12.4f} {psnr(preview(lat[k]), pref):20.2f}")

up = 4
tiles = [np.kron(preview(lat[k]), np.ones((up, up, 1))) for k in order]
diffs = [np.kron(np.clip(np.abs(preview(lat[k]) - pref) * 4, 0, 1), np.ones((up, up, 1))) for k in order]
sheet = np.concatenate([np.concatenate(tiles, 1), np.concatenate(diffs, 1)], 0)
Image.fromarray((sheet * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "preview_sheet.png"))
print("columns:", ", ".join(order), "-> preview_sheet.png (bottom row: |preview diff vs rrrrrr| x4)")
