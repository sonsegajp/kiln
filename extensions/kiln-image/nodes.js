// kiln-image: pixel operations on IMAGE values. Reference pack for docs/NODE_API.md.
//
// IMAGE = { type: 'IMAGE', batch, height, width, data: Float32Array }, interleaved [B,H,W,3], 0..1.
// Pixel (b, y, x), channel c lives at data[((b * height + y) * width + x) * 3 + c].
// Nodes never modify their inputs: they allocate the result with ctx.image(b, h, w).
'use strict';

const MAX_RESOLUTION = 16384;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// copy pixels with a coordinate map: out(y, x) = in(sy, sx) for every batch item
function remap(image, ctx, outH, outW, map) {
  const out = ctx.image(image.batch, outH, outW);
  const src = image.data, dst = out.data, W = image.width, H = image.height;
  const at = [0, 0];
  for (let b = 0; b < image.batch; b++) {
    for (let y = 0; y < outH; y++) {
      let o = ((b * outH + y) * outW) * 3;
      for (let x = 0; x < outW; x++, o += 3) {
        map(y, x, at);
        const i = ((b * H + at[0]) * W + at[1]) * 3;
        dst[o] = src[i]; dst[o + 1] = src[i + 1]; dst[o + 2] = src[i + 2];
      }
    }
  }
  return out;
}

module.exports = {
  nodes: {
    ImageAdjust: {
      display_name: 'Image Adjust (Brightness / Contrast / Saturation)',
      category: 'image/adjust',
      description: 'PIL-style enhance factors, 1 = unchanged: brightness scales, contrast pushes away from the mean grey, saturation away from each pixel\'s grey.',
      inputs: {
        required: {
          image: ['IMAGE'],
          // "widget" picks a custom editor widget registered by this pack's ui.js (a slider);
          // the value is still a plain FLOAT, and the default number box is used if ui.js is missing
          brightness: ['FLOAT', { default: 1, min: 0, max: 3, step: 0.01, widget: 'kiln-image.slider' }],
          contrast: ['FLOAT', { default: 1, min: 0, max: 3, step: 0.01, widget: 'kiln-image.slider' }],
          saturation: ['FLOAT', { default: 1, min: 0, max: 3, step: 0.01, widget: 'kiln-image.slider' }],
        },
      },
      outputs: ['IMAGE'],
      async run({ image, brightness, contrast, saturation }, ctx) {
        const out = ctx.image(image.batch, image.height, image.width);
        const src = image.data, dst = out.data;
        const px = image.height * image.width;
        for (let b = 0; b < image.batch; b++) {
          const start = b * px * 3, end = start + px * 3;
          // mean luma of the brightened image (PIL's contrast reference)
          let sum = 0;
          for (let i = start; i < end; i += 3) sum += (0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2]) * brightness;
          const mean = sum / px;
          for (let i = start; i < end; i += 3) {
            let r = src[i] * brightness, g = src[i + 1] * brightness, bl = src[i + 2] * brightness;
            r = mean + (r - mean) * contrast; g = mean + (g - mean) * contrast; bl = mean + (bl - mean) * contrast;
            const l = 0.299 * r + 0.587 * g + 0.114 * bl;
            dst[i] = clamp01(l + (r - l) * saturation);
            dst[i + 1] = clamp01(l + (g - l) * saturation);
            dst[i + 2] = clamp01(l + (bl - l) * saturation);
          }
          if (ctx.cancelled()) throw new Error('cancelled');
          ctx.progress(b + 1, image.batch);
        }
        return [out];
      },
    },

    // ---- ports of ComfyUI core nodes (comfy_extras/nodes_images.py): same names and inputs ----
    ImageFlip: {
      display_name: 'Image Flip',
      category: 'image/transform',
      inputs: {
        required: {
          image: ['IMAGE'],
          flip_method: [['x-axis: vertically', 'y-axis: horizontally']],
        },
      },
      outputs: ['IMAGE'],
      run({ image, flip_method }, ctx) {
        const H = image.height, W = image.width;
        const vertical = flip_method.startsWith('x-axis');
        return [remap(image, ctx, H, W, vertical
          ? (y, x, at) => { at[0] = H - 1 - y; at[1] = x; }
          : (y, x, at) => { at[0] = y; at[1] = W - 1 - x; })];
      },
    },

    ImageRotate: {
      display_name: 'Image Rotate',
      category: 'image/transform',
      description: 'Rotates clockwise.',
      inputs: {
        required: {
          image: ['IMAGE'],
          rotation: [['none', '90 degrees', '180 degrees', '270 degrees']],
        },
      },
      outputs: ['IMAGE'],
      run({ image, rotation }, ctx) {
        const H = image.height, W = image.width;
        if (rotation.startsWith('90')) return [remap(image, ctx, W, H, (y, x, at) => { at[0] = H - 1 - x; at[1] = y; })];
        if (rotation.startsWith('180')) return [remap(image, ctx, H, W, (y, x, at) => { at[0] = H - 1 - y; at[1] = W - 1 - x; })];
        if (rotation.startsWith('270')) return [remap(image, ctx, W, H, (y, x, at) => { at[0] = x; at[1] = W - 1 - y; })];
        return [image];   // "none": pass the input on unchanged
      },
    },

    ImageCrop: {
      display_name: 'Image Crop',
      category: 'image/transform',
      description: 'Crops width x height at (x, y); the box is clipped to the image, like ComfyUI.',
      inputs: {
        required: {
          image: ['IMAGE'],
          width: ['INT', { default: 512, min: 1, max: MAX_RESOLUTION, step: 1 }],
          height: ['INT', { default: 512, min: 1, max: MAX_RESOLUTION, step: 1 }],
          x: ['INT', { default: 0, min: 0, max: MAX_RESOLUTION, step: 1 }],
          y: ['INT', { default: 0, min: 0, max: MAX_RESOLUTION, step: 1 }],
        },
      },
      outputs: ['IMAGE'],
      run({ image, width, height, x, y }, ctx) {
        const x0 = Math.min(x, image.width - 1), y0 = Math.min(y, image.height - 1);
        const w = Math.min(x0 + width, image.width) - x0, h = Math.min(y0 + height, image.height) - y0;
        return [remap(image, ctx, h, w, (yy, xx, at) => { at[0] = y0 + yy; at[1] = x0 + xx; })];
      },
    },
  },
};
