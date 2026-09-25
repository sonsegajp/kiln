// kiln-image editor UI: a "kiln-image.slider" widget (range slider + number box).
// Loaded by Kiln's node editor as an ES module; see the "UI" section of docs/NODE_API.md.
// Any FLOAT / INT input can use it with ["FLOAT", { ..., widget: "kiln-image.slider" }].

export default function setup(kiln) {
  kiln.css(`
    .kis { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; }
    .kis input[type=range] { flex: 1; min-width: 50px; accent-color: var(--accent, #ff7a3d); }
    .kis input[type=number] { width: 58px; }
  `);

  kiln.registerWidget('kiln-image.slider', {
    create({ value, set, input }) {
      const o = input.options;
      const min = o.min ?? 0, max = o.max ?? 1, step = o.step ?? (input.type === 'INT' ? 1 : 0.01);
      const box = document.createElement('div');
      box.className = 'kis';
      const range = document.createElement('input');
      range.type = 'range';
      const num = document.createElement('input');
      num.type = 'number';
      num.className = 'ng-num';
      for (const el of [range, num]) { el.min = min; el.max = max; el.step = step; el.value = value; }
      const fix = (v) => {
        v = Math.min(max, Math.max(min, Number(v)));
        if (!Number.isFinite(v)) v = o.default ?? min;
        return input.type === 'INT' ? Math.round(v) : Math.round(v / step) * step;
      };
      const apply = (v, commit) => {
        v = +fix(v).toFixed(6);
        range.value = v;
        num.value = v;
        if (commit) set(v);
      };
      range.addEventListener('input', () => apply(range.value, false));
      range.addEventListener('change', () => apply(range.value, true));
      num.addEventListener('change', () => apply(num.value, true));
      // double-click the slider to reset to the default
      range.addEventListener('dblclick', () => apply(o.default ?? min, true));
      box.append(range, num);
      return box;
    },
  });
}
