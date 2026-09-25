"""Ultralytics YOLOv8 detection .pt -> fused .safetensors + .json, without ultralytics installed.

  python convert_yolo.py models/detect/face_yolov8m.pt   -> face_yolov8m.safetensors / .json next to it

The checkpoint is a pickle of the whole nn.Module tree. Every ultralytics.* class is replaced
by a stub nn.Module subclass while unpickling; torch.nn children (Conv2d, BatchNorm2d, ...)
load for real, so the tree keeps its attributes (f, i, c, add, stride, ...) and parameters.

Each ultralytics Conv (Conv2d without bias + BatchNorm2d + act) is fused into a single conv with
bias, stored as fp32 under Ultralytics' own fused naming ("model.<i>...conv.weight/.bias").
The .json holds the training yaml, class names, strides and a resolved layer list the engine
builds from (exact channel counts, kernel/stride/pad, activation, tensor names).
"""
import json
import os
import pickle
import sys
import types

import torch
from safetensors.torch import save_file


class _Stub(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()


_stubs = {}


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split('.')[0] == 'ultralytics':
            key = mod + '.' + name
            if key not in _stubs:
                _stubs[key] = type(name, (_Stub,), {'__module__': 'ultralytics_stub', '_ul_path': key})
            return _stubs[key]
        return super().find_class(mod, name)


_pm = types.ModuleType('ul_pickle')
_pm.Unpickler = _Unpickler
_pm.load = pickle.load


def kind(m):
    return type(m).__name__


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False, pickle_module=_pm)
    model = ck.get('model') if ck.get('model') is not None else ck.get('ema')
    if model is None:
        raise SystemExit('checkpoint has neither model nor ema')
    return ck, model


def act_name(a):
    n = kind(a)
    if n == 'SiLU':
        return 'silu'
    if n == 'Identity':
        return 'none'
    raise SystemExit('unsupported activation ' + n)


class Converter:
    def __init__(self):
        self.tensors = {}
        self.max_fuse_err = 0.0

    def _geom(self, c):
        k, s, p, d = c.kernel_size, c.stride, c.padding, c.dilation
        if k[0] != k[1] or s[0] != s[1] or p[0] != p[1] or d != (1, 1) or c.groups != 1:
            raise SystemExit(f'unsupported conv geometry k={k} s={s} p={p} d={d} g={c.groups}')
        return dict(cin=c.in_channels, cout=c.out_channels, k=k[0], s=s[0], p=p[0])

    def conv(self, path, m):
        """ultralytics Conv (conv+bn+act) or a bare nn.Conv2d -> fused record."""
        if kind(m) == 'Conv2d':
            c, bn, act = m, None, 'none'
            name = path
        else:
            c, bn, act = m.conv, getattr(m, 'bn', None), act_name(m.act)
            name = path + '.conv'
        w = c.weight.detach().float()
        b = c.bias.detach().float() if c.bias is not None else torch.zeros(w.shape[0])
        if bn is not None:
            inv = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
            wf = w * inv.view(-1, 1, 1, 1)
            bf = (b - bn.running_mean.detach().float()) * inv + bn.bias.detach().float()
            # one-shot check: fused conv == conv -> bn (eval) on a random input
            with torch.no_grad():
                x = torch.randn(1, w.shape[1], 9, 9)
                g = self._geom(c)
                ref = torch.nn.functional.batch_norm(
                    torch.nn.functional.conv2d(x, w, b if c.bias is not None else None, g['s'], g['p']),
                    bn.running_mean.float(), bn.running_var.float(), bn.weight.float(), bn.bias.float(), False, 0.0, bn.eps)
                got = torch.nn.functional.conv2d(x, wf, bf, g['s'], g['p'])
                self.max_fuse_err = max(self.max_fuse_err, ((got - ref).norm() / ref.norm()).item())
            w, b = wf, bf
        self.tensors[name + '.weight'] = w.contiguous()
        self.tensors[name + '.bias'] = b.contiguous()
        r = self._geom(c)
        r.update(name=name, act=act)
        return r

    def layer(self, i, m):
        p = f'model.{i}'
        rec = dict(i=int(getattr(m, 'i', i)), f=m.f, type=kind(m))
        t = kind(m)
        if t == 'Conv':
            rec['conv'] = self.conv(p, m)
        elif t == 'C2f':
            rec['c'] = int(m.c)
            rec['cv1'] = self.conv(p + '.cv1', m.cv1)
            rec['cv2'] = self.conv(p + '.cv2', m.cv2)
            rec['m'] = [dict(cv1=self.conv(f'{p}.m.{j}.cv1', b.cv1), cv2=self.conv(f'{p}.m.{j}.cv2', b.cv2), add=bool(b.add))
                        for j, b in enumerate(m.m)]
        elif t == 'SPPF':
            mp = m.m
            if not (mp.stride == 1 and mp.padding == mp.kernel_size // 2):
                raise SystemExit('unexpected SPPF pool geometry')
            rec.update(cv1=self.conv(p + '.cv1', m.cv1), cv2=self.conv(p + '.cv2', m.cv2), k=int(mp.kernel_size))
        elif t == 'Upsample':
            if m.mode != 'nearest':
                raise SystemExit('only nearest upsampling is supported')
            rec.update(scale=int(m.scale_factor), mode=m.mode)
        elif t == 'Concat':
            rec['dim'] = int(m.d)
        elif t == 'Detect':
            if getattr(m, 'end2end', False):
                raise SystemExit('end2end (one2one) heads are not supported')
            rm = int(m.reg_max)
            dw = m.dfl.conv.weight.detach().float().flatten()
            if not torch.equal(dw, torch.arange(rm, dtype=torch.float)):
                raise SystemExit('DFL conv is not arange(reg_max)')
            self.tensors[p + '.dfl.conv.weight'] = m.dfl.conv.weight.detach().float().contiguous()
            rec.update(nc=int(m.nc), reg_max=rm, no=int(m.no), stride=[float(s) for s in m.stride])
            rec['cv2'] = [[self.conv(f'{p}.cv2.{l}.{j}', s[j]) for j in range(len(s))] for l, s in enumerate(m.cv2)]
            rec['cv3'] = [[self.conv(f'{p}.cv3.{l}.{j}', s[j]) for j in range(len(s))] for l, s in enumerate(m.cv3)]
            for l, s in enumerate(m.cv3):
                if any(kind(x) not in ('Conv', 'Conv2d') for x in s):
                    raise SystemExit('non-legacy Detect cv3 (DWConv) is not supported')
        else:
            raise SystemExit('unsupported layer type ' + t)
        return rec


def main():
    src = sys.argv[1]
    base = os.path.splitext(src)[0]
    ck, model = load_ckpt(src)
    print('checkpoint keys:', list(ck.keys()), '| ultralytics', ck.get('version'), '| date', ck.get('date'))
    print('stub classes:', sorted(_stubs))
    cv = Converter()
    layers = [cv.layer(i, m) for i, m in enumerate(model.model)]
    names = {str(k): v for k, v in model.names.items()}
    ta = ck.get('train_args') or {}
    ta = ta if isinstance(ta, dict) else getattr(ta, '__dict__', {})
    meta = dict(
        format='kiln-yolov8-det/1',
        source=os.path.basename(src),
        ultralytics_version=ck.get('version'),
        date=ck.get('date'),
        imgsz=int(ta.get('imgsz', 640)) if not isinstance(ta.get('imgsz', 640), (list, tuple)) else int(ta['imgsz'][0]),
        yaml=model.yaml,
        names=names,
        nc=int(model.nc) if hasattr(model, 'nc') else int(model.yaml['nc']),
        stride=[float(s) for s in model.stride],
        save=sorted(int(s) for s in model.save),
        layers=layers,
    )
    save_file(cv.tensors, base + '.safetensors', metadata={'format': 'kiln-yolov8-det/1'})
    with open(base + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    nparam = sum(t.numel() for t in cv.tensors.values())
    print(f'layers {len(layers)}  tensors {len(cv.tensors)}  params {nparam/1e6:.2f}M  names {names}  stride {meta["stride"]}')
    print(f'max BN-fusion rel err (fp32 check) {cv.max_fuse_err:.2e}')
    print('wrote', base + '.safetensors', 'and', base + '.json')


if __name__ == '__main__':
    main()
