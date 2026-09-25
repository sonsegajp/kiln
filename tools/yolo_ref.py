"""Plain-PyTorch YOLOv8 detection forward (no ultralytics), built from the checkpoint's yaml
(scales, depth/width multiples, backbone/head spec) + the fused weights from convert_yolo.py.
Reference for engine/src/detect.cu.

  python yolo_ref.py <model base> <image.png> [--dump DIR] [--conf 0.3] [--iou 0.5]

<model base> is e.g. models/detect/face_yolov8m (no extension). With --dump pointing at the
directory bench/detect_test.exe --dump wrote, it compares against the CUDA module:
  letterbox  : this file's numpy letterbox vs the CUDA letterbox kernel
  raw head   : [65, N] head output of this forward run on the CUDA's letterboxed input
  decoded    : [5, N] xywh + score after DFL/dist2bbox/sigmoid
  boxes      : full pipeline here (own letterbox -> forward -> decode -> NMS -> scale) vs CUDA boxes
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


def make_div(x, d=8):
    return math.ceil(x / d) * d


class YoloRef:
    def __init__(self, base):
        meta = json.load(open(base + '.json'))
        self.T = load_file(base + '.safetensors')
        self.used = set()
        y = meta['yaml']
        self.nc = y['nc']
        self.names = meta['names']
        self.imgsz = meta.get('imgsz', 640)
        scales = y.get('scales')
        if scales:
            depth, width, maxc = scales[y.get('scale') or next(iter(scales))]
        else:
            depth, width, maxc = y.get('depth_multiple', 1.0), y.get('width_multiple', 1.0), float('inf')
        # mirror of ultralytics parse_model for the module set YOLOv8 detect uses
        self.layers = []
        ch = [y.get('ch', 3)]
        for i, (f, n, m, args) in enumerate(y['backbone'] + y['head']):
            n = max(round(n * depth), 1) if n > 1 else n
            args = [self.nc if a == 'nc' else (None if a == 'None' else a) for a in args]
            spec = {}
            if m in ('Conv', 'C2f', 'SPPF'):
                c1, c2 = ch[f], args[0]
                if c2 != self.nc:
                    c2 = make_div(min(c2, maxc) * width, 8)
                spec = dict(c1=c1, c2=c2)
                if m == 'Conv':
                    spec.update(k=args[1] if len(args) > 1 else 1, s=args[2] if len(args) > 2 else 1)
                elif m == 'C2f':
                    spec.update(n=n, shortcut=bool(args[1]) if len(args) > 1 else False)
                else:
                    spec.update(k=args[1] if len(args) > 1 else 5)
            elif m == 'nn.Upsample':
                c2 = ch[f]
                spec = dict(scale=args[1], mode=args[2])
            elif m == 'Concat':
                c2 = sum(ch[x] for x in f)
            elif m == 'Detect':
                c2 = None
                spec = dict(ch=[ch[x] for x in f], reg_max=16)
            else:
                raise SystemExit('unsupported module ' + m)
            self.layers.append((i, f, m, spec))
            if i == 0:
                ch = []
            ch.append(c2)
        self.reg_max = 16
        self.stride = meta['stride']

    def conv(self, x, name, c1, c2, k, s, act=True):
        w, b = self.T[name + '.weight'], self.T[name + '.bias']
        assert tuple(w.shape) == (c2, c1, k, k), (name, tuple(w.shape), (c2, c1, k, k))
        self.used.update((name + '.weight', name + '.bias'))
        y = F.conv2d(x, w, b, s, k // 2)
        return F.silu(y) if act else y

    @torch.no_grad()
    def forward(self, x):
        ys = []
        for i, f, m, sp in self.layers:
            p = f'model.{i}'
            if f != -1:
                x = ys[f] if isinstance(f, int) else [x if j == -1 else ys[j] for j in f]
            if m == 'Conv':
                x = self.conv(x, p + '.conv', sp['c1'], sp['c2'], sp['k'], sp['s'])
            elif m == 'C2f':
                c = int(sp['c2'] * 0.5)
                y = list(self.conv(x, p + '.cv1.conv', sp['c1'], 2 * c, 1, 1).chunk(2, 1))
                for j in range(sp['n']):
                    h = self.conv(self.conv(y[-1], f'{p}.m.{j}.cv1.conv', c, c, 3, 1), f'{p}.m.{j}.cv2.conv', c, c, 3, 1)
                    y.append(y[-1] + h if sp['shortcut'] else h)
                x = self.conv(torch.cat(y, 1), p + '.cv2.conv', (2 + sp['n']) * c, sp['c2'], 1, 1)
            elif m == 'SPPF':
                c_ = sp['c1'] // 2
                y = [self.conv(x, p + '.cv1.conv', sp['c1'], c_, 1, 1)]
                for _ in range(3):
                    y.append(F.max_pool2d(y[-1], sp['k'], 1, sp['k'] // 2))
                x = self.conv(torch.cat(y, 1), p + '.cv2.conv', 4 * c_, sp['c2'], 1, 1)
            elif m == 'nn.Upsample':
                x = F.interpolate(x, scale_factor=sp['scale'], mode=sp['mode'])
            elif m == 'Concat':
                x = torch.cat(x, 1)
            elif m == 'Detect':
                rm, ch0 = sp['reg_max'], sp['ch'][0]
                c2, c3 = max(16, ch0 // 4, rm * 4), max(ch0, min(self.nc, 100))
                outs = []
                for l, xl in enumerate(x):
                    a = self.conv(xl, f'{p}.cv2.{l}.0.conv', sp['ch'][l], c2, 3, 1)
                    a = self.conv(a, f'{p}.cv2.{l}.1.conv', c2, c2, 3, 1)
                    a = self.conv(a, f'{p}.cv2.{l}.2', c2, 4 * rm, 1, 1, act=False)
                    b = self.conv(xl, f'{p}.cv3.{l}.0.conv', sp['ch'][l], c3, 3, 1)
                    b = self.conv(b, f'{p}.cv3.{l}.1.conv', c3, c3, 3, 1)
                    b = self.conv(b, f'{p}.cv3.{l}.2', c3, self.nc, 1, 1, act=False)
                    outs.append(torch.cat([a, b], 1))
                shapes = [tuple(o.shape[2:]) for o in outs]
                x = torch.cat([o.flatten(2) for o in outs], 2)  # [1, no, N]
            ys.append(x)
        return x[0], shapes

    def decode(self, raw, shapes):
        """raw [no, N] -> [4+nc, N] (xywh in letterbox pixels, sigmoid scores), like Detect._inference."""
        anchors, st = [], []
        for (h, w), s in zip(shapes, self.stride):
            sy, sx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing='ij')
            anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
            st.append(torch.full((h * w,), float(s)))
        anchors, st = torch.cat(anchors).T, torch.cat(st)
        rm = self.reg_max
        box, cls = raw.split((4 * rm, self.nc), 0)
        dist = (box.view(4, rm, -1).softmax(1) * torch.arange(rm, dtype=raw.dtype).view(1, rm, 1)).sum(1)
        lt, rb = dist[:2], dist[2:]
        x1y1, x2y2 = anchors - lt, anchors + rb
        dbox = torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), 0) * st
        return torch.cat((dbox, cls.sigmoid()), 0)


def letterbox(img, new=640, stride=32):
    """float [H, W, 3] in [0,1] -> [3, h, w]: Ultralytics LetterBox(auto=True) with cv2 INTER_LINEAR semantics."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new - nw) % stride, (new - nh) % stride
    dw, dh = dw / 2, dh / 2
    x = img.astype(np.float64)
    if (w, h) != (nw, nh):
        def taps(n_out, n_in):
            f = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
            i0 = np.floor(f).astype(np.int64)
            t = f - i0
            t[i0 < 0] = 0
            i0[i0 < 0] = 0
            hi = i0 >= n_in - 1
            t[hi] = 0
            i0[hi] = n_in - 1
            return i0, np.minimum(i0 + 1, n_in - 1), t
        y0, y1, ty = taps(nh, h)
        x0, x1, tx = taps(nw, w)
        rows = x[y0] * (1 - ty)[:, None, None] + x[y1] * ty[:, None, None]
        x = rows[:, x0] * (1 - tx)[None, :, None] + rows[:, x1] * tx[None, :, None]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = np.full((nh + top + bottom, nw + left + right, 3), 114 / 255, np.float64)
    out[top:top + nh, left:left + nw] = x
    return out.transpose(2, 0, 1).astype(np.float32)


def nms_and_scale(dec, nc, conf, iou, lh, lw, H, W, max_det=300):
    """Ultralytics non_max_suppression (agnostic=False, multi_label=False) + scale_boxes + clip."""
    d = dec.T  # [N, 4+nc]
    scores, cls = d[:, 4:4 + nc].max(1)
    keep = scores > conf
    xywh, scores, cls = d[keep, :4], scores[keep], cls[keep]
    xyxy = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    order = torch.argsort(scores, descending=True, stable=True)
    xyxy, scores, cls = xyxy[order], scores[order], cls[order]
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    dead = torch.zeros(len(scores), dtype=torch.bool)
    out = []
    for i in range(len(scores)):
        if dead[i]:
            continue
        out.append(i)
        if len(out) == max_det:
            break
        lt = torch.maximum(xyxy[i, :2], xyxy[i + 1:, :2])
        rb = torch.minimum(xyxy[i, 2:], xyxy[i + 1:, 2:])
        inter = (rb - lt).clamp(min=0).prod(1)
        ious = inter / (area[i] + area[i + 1:] - inter)
        dead[i + 1:] |= (ious > iou) & (cls[i + 1:] == cls[i])
    gain = min(lh / H, lw / W)
    padx, pady = round((lw - W * gain) / 2 - 0.1), round((lh - H * gain) / 2 - 0.1)
    boxes = []
    for i in out:
        x0, y0, x1, y1 = xyxy[i].tolist()
        boxes.append([min(max((x0 - padx) / gain, 0), W), min(max((y0 - pady) / gain, 0), H),
                      min(max((x1 - padx) / gain, 0), W), min(max((y1 - pady) / gain, 0), H), scores[i].item()])
    return boxes


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('image')
    ap.add_argument('--dump')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--iou', type=float, default=0.5)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    net = YoloRef(a.model)
    img = np.asarray(Image.open(a.image).convert('RGB'), np.float32) / 255
    H, W = img.shape[:2]
    inp = letterbox(img, net.imgsz)
    lh, lw = inp.shape[1:]
    raw, shapes = net.forward(torch.from_numpy(inp)[None])
    unused = set(k for k in net.T if not k.endswith('dfl.conv.weight')) - net.used
    assert not unused, f'weights not used by the yaml-built graph: {sorted(unused)[:5]}'
    dec = net.decode(raw, shapes)
    boxes = nms_and_scale(dec, net.nc, a.conf, a.iou, lh, lw, H, W)
    name = os.path.splitext(os.path.basename(a.image))[0]
    print(f'{name}: {W}x{H} -> letterbox {lw}x{lh}, {raw.shape[1]} anchors; reference boxes:')
    for b in boxes:
        print('   [%7.1f %7.1f %7.1f %7.1f]  %.4f' % tuple(b))
    if not a.dump:
        return
    d = os.path.join(a.dump, name)
    c_in = np.load(d + '_input.npy')
    c_raw = np.load(d + '_raw.npy')
    c_dec = np.load(d + '_dec.npy')
    c_boxes = np.loadtxt(d + '_boxes.txt', ndmin=2) if os.path.getsize(d + '_boxes.txt') else np.zeros((0, 5))
    assert c_in.shape == inp.shape, (c_in.shape, inp.shape)
    print(f'  letterbox  CUDA vs numpy : rel_l2 {rel_l2(c_in, inp):.2e}  max|d| {np.abs(c_in - inp).max():.2e}')
    # isolate the network: run the reference on the CUDA letterboxed input
    raw2, _ = net.forward(torch.from_numpy(c_in)[None])
    dec2 = net.decode(raw2, shapes)
    r2 = raw2.numpy()
    rm4 = 4 * net.reg_max
    print(f'  raw head [{r2.shape[0]},{r2.shape[1]}]  : rel_l2 {rel_l2(c_raw, r2):.2e}  (box logits {rel_l2(c_raw[:rm4], r2[:rm4]):.2e}, '
          f'class logits {rel_l2(c_raw[rm4:], r2[rm4:]):.2e})  max|d| {np.abs(c_raw - r2).max():.2e}')
    d2 = dec2.numpy()
    print(f'  decoded [{d2.shape[0]},{d2.shape[1]}]    : rel_l2 {rel_l2(c_dec, d2):.2e}  box max|d| {np.abs(c_dec[:4] - d2[:4]).max():.3f} px  '
          f'score max|d| {np.abs(c_dec[4:] - d2[4:]).max():.2e}')
    print(f'  final boxes: CUDA {len(c_boxes)}, reference {len(boxes)}')
    if len(c_boxes) == len(boxes) and len(boxes):
        rb = np.array(boxes)
        print(f'    max |coord diff| {np.abs(c_boxes[:, :4] - rb[:, :4]).max():.3f} px, max |score diff| {np.abs(c_boxes[:, 4] - rb[:, 4]).max():.2e}')
    for b in c_boxes:
        print('   CUDA [%7.1f %7.1f %7.1f %7.1f]  %.4f' % tuple(b))


if __name__ == '__main__':
    main()
