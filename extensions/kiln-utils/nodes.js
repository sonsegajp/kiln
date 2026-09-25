// kiln-utils: small, commonly needed utility nodes. Reference pack for docs/NODE_API.md.
//
// Every node here only takes constants (widgets, or other constant nodes), so Kiln runs them once on
// the server when the graph is queued ("folding") and passes their results on as plain values. That
// is why a Text Join can feed CLIPTextEncode and a Seed can feed several samplers.
'use strict';

const MAX_SEED = 18446744073709552000;           // same range as KSampler's seed widget
const MAX_INT = Number.MAX_SAFE_INTEGER;

// ---- math -------------------------------------------------------------------------------------
const INT_OPS = {
  add: [(a, b) => a + b, '+'],
  subtract: [(a, b) => a - b, '-'],
  multiply: [(a, b) => a * b, '×'],
  divide: [(a, b) => a / b, '/'],               // INT output rounds toward -infinity (like Python //)
  modulo: [(a, b) => ((a % b) + b) % b, 'mod'], // sign of b, like Python
  power: [(a, b) => a ** b, '^'],
  min: [(a, b) => Math.min(a, b), 'min'],
  max: [(a, b) => Math.max(a, b), 'max'],
};
const FLOAT_OPS = Object.assign({}, INT_OPS, {
  atan2: [(a, b) => Math.atan2(a, b), 'atan2'],
  log: [(a, b) => Math.log(a) / Math.log(b), 'log base'],   // log of a in base b
});

function math(ops, a, op, b, intFloor) {
  const f = ops[op];
  if (!f) throw new Error(`unknown operation "${op}"`);
  if ((op === 'divide' || op === 'modulo') && b === 0) throw new Error('division by zero');
  const r = f[0](a, b);
  if (!Number.isFinite(r)) throw new Error(`${a} ${f[1]} ${b} is not a finite number`);
  const asInt = intFloor && op === 'divide' ? Math.floor(r) : Math.round(r);
  // { outputs, ui }: ui.text is shown under the node in the editor
  return { outputs: [asInt, r], ui: { text: `${a} ${f[1]} ${b} = ${Number.isInteger(r) ? r : +r.toPrecision(10)}` } };
}

module.exports = {
  nodes: {
    // ---- seed ---------------------------------------------------------------------------------
    Seed: {
      display_name: 'Seed',
      category: 'utils',
      description: 'One seed for several samplers. "control after generate" randomizes / increments it after each queue.',
      inputs: {
        required: {
          seed: ['INT', { default: 0, min: 0, max: MAX_SEED, control_after_generate: true }],
        },
      },
      outputs: ['INT'],
      output_names: ['seed'],
      run({ seed }) {
        return [seed];
      },
    },

    // ---- text ---------------------------------------------------------------------------------
    // Port of ComfyUI's core StringConcatenate (same class name and inputs, so ComfyUI workflows
    // that use it load and run unchanged).
    StringConcatenate: {
      display_name: 'Concatenate',
      category: 'utils/string',
      description: 'string_a + delimiter + string_b.',
      inputs: {
        required: {
          string_a: ['STRING', { multiline: true }],
          string_b: ['STRING', { multiline: true }],
          delimiter: ['STRING', { multiline: false, default: '' }],
        },
      },
      outputs: ['STRING'],
      run({ string_a, string_b, delimiter }) {
        return [string_a + delimiter + string_b];
      },
    },

    TextJoin: {
      display_name: 'Text Join',
      category: 'utils/string',
      description: 'Joins up to four texts with a separator ("\\n" = new line). Empty parts are skipped, so optional prompt pieces leave no stray commas.',
      inputs: {
        required: {
          separator: ['STRING', { default: ', ' }],
          skip_empty: ['BOOLEAN', { default: true, label_on: 'skip empty', label_off: 'keep empty' }],
        },
        optional: {
          text_1: ['STRING', { multiline: true, default: '' }],
          text_2: ['STRING', { multiline: true, default: '' }],
          text_3: ['STRING', { multiline: true, default: '' }],
          text_4: ['STRING', { multiline: true, default: '' }],
        },
      },
      outputs: ['STRING'],
      output_names: ['text'],
      run(i) {
        const parts = [i.text_1, i.text_2, i.text_3, i.text_4].map(t => (t == null ? '' : String(t).trim()));
        const sep = String(i.separator).replace(/\\n/g, '\n').replace(/\\t/g, '\t');
        return [(i.skip_empty ? parts.filter(Boolean) : parts).join(sep)];
      },
    },

    // ---- math ---------------------------------------------------------------------------------
    IntMath: {
      display_name: 'Int Math',
      category: 'utils/math',
      description: 'a (operation) b on integers. INT is the integer result (divide floors), FLOAT the exact one.',
      inputs: {
        required: {
          a: ['INT', { default: 0, min: -MAX_INT, max: MAX_INT }],
          operation: [Object.keys(INT_OPS), { default: 'add' }],
          b: ['INT', { default: 1, min: -MAX_INT, max: MAX_INT }],
        },
      },
      outputs: ['INT', 'FLOAT'],
      run({ a, operation, b }) {
        return math(INT_OPS, a, operation, b, true);
      },
    },

    FloatMath: {
      display_name: 'Float Math',
      category: 'utils/math',
      description: 'a (operation) b on floats. INT is the result rounded to the nearest integer.',
      inputs: {
        required: {
          a: ['FLOAT', { default: 0, min: -1e12, max: 1e12, step: 0.01 }],
          operation: [Object.keys(FLOAT_OPS), { default: 'multiply' }],
          b: ['FLOAT', { default: 1, min: -1e12, max: 1e12, step: 0.01 }],
        },
      },
      outputs: ['FLOAT', 'INT'],
      run({ a, operation, b }) {
        const r = math(FLOAT_OPS, a, operation, b, false);
        return { outputs: [r.outputs[1], r.outputs[0]], ui: r.ui };
      },
    },

    // ---- logic --------------------------------------------------------------------------------
    // "*" sockets accept any type. With constant inputs it folds on the server; with an image,
    // model, ... connected it runs during the job and passes the chosen value through.
    Switch: {
      display_name: 'Switch',
      category: 'utils/logic',
      description: 'Passes on_true or on_false on, depending on "select". Any type: images, text, models, numbers...',
      inputs: {
        required: {
          select: ['BOOLEAN', { default: true, label_on: 'on_true', label_off: 'on_false' }],
        },
        optional: {
          on_true: ['*'],
          on_false: ['*'],
        },
      },
      outputs: ['*'],
      output_names: ['value'],
      run({ select, on_true, on_false }) {
        const v = select ? on_true : on_false;
        if (v === undefined) throw new Error(`the selected input (${select ? 'on_true' : 'on_false'}) is not connected`);
        return [v];
      },
    },
  },
};
