// kiln-upscale-to-size: a pack node that drives the GPU through ctx.ops (docs/NODE_API.md).
//
// The UPSCALE_MODEL input is an opaque engine handle ({ type, handle }): the node never sees the
// weights, it just hands the handle back to the engine with ctx.ops.upscale(). Every ctx.ops call
// is a round-trip to kiln-engine (ext_op), so the heavy lifting stays on the GPU.
'use strict';

const METHODS = ['lanczos', 'bicubic', 'bilinear', 'area', 'nearest-exact'];

module.exports = {
  nodes: {
    UpscaleImageToSize: {
      display_name: 'Upscale Image To Size (model)',
      category: 'image/upscaling',
      description: 'Upscales with the model only when the target is bigger than the image, then resizes to the target. "fit inside" keeps the aspect ratio (the result fits in width x height); "stretch" gives exactly width x height.',
      inputs: {
        required: {
          image: ['IMAGE'],
          upscale_model: ['UPSCALE_MODEL'],
          width: ['INT', { default: 2048, min: 16, max: 16384, step: 8 }],
          height: ['INT', { default: 2048, min: 16, max: 16384, step: 8 }],
          method: [METHODS, { default: 'lanczos' }],
          keep_proportion: ['BOOLEAN', { default: true, label_on: 'fit inside', label_off: 'stretch' }],
        },
      },
      outputs: ['IMAGE', 'INT', 'INT'],
      output_names: ['IMAGE', 'width', 'height'],
      async run({ image, upscale_model, width, height, method, keep_proportion }, ctx) {
        let w = width, h = height;
        if (keep_proportion) {
          const k = Math.min(width / image.width, height / image.height);
          w = Math.max(1, Math.round(image.width * k));
          h = Math.max(1, Math.round(image.height * k));
        }
        let img = image;
        ctx.progress(0, 2);
        if (w > image.width || h > image.height) {
          img = await ctx.ops.upscale(image, upscale_model);
          ctx.log(`model: ${image.width}x${image.height} -> ${img.width}x${img.height}`);
        }
        ctx.progress(1, 2);
        if (img.width !== w || img.height !== h) img = await ctx.ops.resize(img, w, h, method);
        ctx.progress(2, 2);
        return [img, img.width, img.height];
      },
    },
  },
};
