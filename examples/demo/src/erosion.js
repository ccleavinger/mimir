/* Only compact previews cross the network; the full simulation lives on the chosen backend. */
(() => {
  const $ = id => document.getElementById(id);
  let simulation = null, selected = null, original = null, current = null;
  let running = false, busy = false, generation = 0, timer = null;
  function pause() { running = false; clearTimeout(timer); if (simulation && !busy) note('Paused. The terrain remains on the backend.'); update(); }
  function update() {
    $('erosion-run').textContent = running ? 'Pause' : 'Run erosion';
    $('erosion-run').disabled = !simulation || (busy && !running);
    $('erosion-create').disabled = busy || running;
    $('erosion-step').disabled = !simulation || busy || running;
    $('erosion-compare').disabled = !simulation || busy || running;
    $('erosion-settings').disabled = busy || running;
  }
  function note(text) { $('erosion-note').textContent = text; }
  async function request(operation, args) {
    const token = generation, destination = target.value;
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 35000);
    const start = performance.now();
    try {
      const response = await fetch('/sim/' + destination + '/' + operation, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(args), signal: controller.signal
      });
      const text = await response.text(), result = JSON.parse(text);
      if (token !== generation) return null;
      if (!response.ok) throw new Error(result.error || 'Backend request failed.');
      result.request_ms = performance.now() - start;
      result.bytes = new TextEncoder().encode(text).length;
      return result;
    } finally { clearTimeout(timeout); }
  }
  function decode(result) {
    const n = result.preview_size, height = [], water = [];
    for (let i = 0; i < n * n; i++) {
      height.push(parseInt(result.height.slice(i * 2, i * 2 + 2), 16) / 255);
      water.push(parseInt(result.water.slice(i * 2, i * 2 + 2), 16));
    }
    return {n, height, water};
  }
  function draw(id, data) {
    const canvas = $(id), ctx = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = '#eee6d5'; ctx.fillRect(0, 0, width, height);
    if (!data) return;
    const {n, height: h, water} = data;
    const iso = $('erosion-view').value === 'iso';
    const project = (x, y, z) => iso ? [width / 2 + (x - y) * width / (n * 2.5), 130 + (x + y) * 1.6 - z * 170] : [x * width / n, y * height / n];
    function color(x, y) {
      const i = y * n + x, z = h[i];
      const shade = Math.max(.48, Math.min(1.25, .92 + ((h[y * n + Math.max(0, x - 1)] - h[y * n + Math.min(n - 1, x + 1)]) * 9) + ((h[Math.max(0, y - 1) * n + x] - h[Math.min(n - 1, y + 1) * n + x]) * 5)));
      const t = Math.min(1, Math.max(0, (z - .2) / .55));
      let r = (56 + t * 171) * shade, g = (91 + t * 122) * shade, b = (60 + t * 122) * shade;
      const loss = original ? original.height[i] - z : 0;
      const exposed = Math.min(.8, Math.abs(loss) * 15);
      const earth = loss > 0 ? [177, 103, 56] : [204, 176, 117];
      r = r * (1 - exposed) + earth[0] * exposed; g = g * (1 - exposed) + earth[1] * exposed; b = b * (1 - exposed) + earth[2] * exposed;
      const wet = Math.max(0, (water[i] - 35) / 220) * .85;
      r = r * (1 - wet) + 40 * wet; g = g * (1 - wet) + 124 * wet; b = b * (1 - wet) + 180 * wet;
      return `rgb(${r},${g},${b})`;
    }
    if (iso) {
      ctx.fillStyle = '#968363';
      for (const edge of [0, 1]) {
        ctx.beginPath();
        for (let i = 0; i < n; i++) { const x = edge ? n - 1 : i, y = edge ? i : n - 1; const p = project(x, y, h[y * n + x]); if (!i) ctx.moveTo(...p); else ctx.lineTo(...p); }
        ctx.lineTo(...project(n - 1, n - 1, -.1)); ctx.lineTo(...project(edge ? n - 1 : 0, edge ? 0 : n - 1, -.1)); ctx.closePath(); ctx.fill();
      }
      for (let diagonal = 0; diagonal < 2 * n - 3; diagonal++) for (let y = 0; y < n - 1; y++) {
        const x = diagonal - y; if (x < 0 || x >= n - 1) continue;
        const i = y * n + x; ctx.beginPath(); ctx.moveTo(...project(x, y, h[i]));
        ctx.lineTo(...project(x + 1, y, h[i + 1])); ctx.lineTo(...project(x + 1, y + 1, h[i + n + 1])); ctx.lineTo(...project(x, y + 1, h[i + n]));
        ctx.closePath(); ctx.fillStyle = color(x, y); ctx.fill();
      }
    } else {
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { ctx.fillStyle = color(x, y); ctx.fillRect(x * width / n, y * height / n, width / n + 1, height / n + 1); }
    }
  }
  function redraw() { draw('erosion-original', original); draw('erosion-current', current); }
  function display(result) {
    current = decode(result); if (!original) original = current;
    $('erosion-stats').textContent = `${result.size} × ${result.size} simulation · ${result.steps.toLocaleString()} steps · GPU ${result.gpu_ms.toFixed(1)} ms · readback ${result.readback_ms.toFixed(1)} ms · request ${result.request_ms.toFixed(0)} ms · ${(result.bytes / 1024).toFixed(1)} KiB preview`;
    $('erosion-current-label').textContent = 'After ' + result.steps.toLocaleString() + ' steps · blue shows water'; redraw();
  }
  const rain = () => Number($('erosion-rain').value) / 1000;
  $('erosion-create').onclick = async () => {
    if (busy) return; pause(); generation++; busy = true; simulation = null; update(); note('Building terrain on the selected backend…');
    $('erosion-comparison').replaceChildren();
    try {
      const result = await request('erosion-create', {size: Number($('erosion-size').value), seed: Number($('erosion-seed').value), rain: rain()});
      if (!result) return;
      simulation = result.simulation; selected = target.value; original = null; display(result); note('Terrain ready. Run rainfall or compare identical steps on GPU and CPU.');
    } catch (e) { note('Could not create terrain: ' + e.message); }
    finally { busy = false; update(); }
  };
  async function step() {
    if (busy || !simulation) return;
    if (selected !== target.value) { window.erosionDemo.targetChanged(); return; }
    busy = true; update();
    try {
      const result = await request('erosion-step', {simulation, steps: 64, rain: rain()});
      if (result) { display(result); note(running ? 'Rainfall is reshaping the terrain. Preview updates are paced for a slow connection.' : 'Paused. The terrain remains on the backend.'); }
    } catch (e) { pause(); note('Simulation paused: ' + e.message + ' The last preview is kept.'); }
    finally { busy = false; update(); if (running) timer = setTimeout(step, 800); }
  }
  $('erosion-step').onclick = step;
  $('erosion-run').onclick = () => { if (running) pause(); else { running = true; update(); step(); } };
  $('erosion-view').onchange = redraw;
  $('erosion-compare').onclick = async () => {
    if (busy || !simulation) return; pause(); busy = true; update(); note('Comparing warmed GPU execution with a single CPU thread on the selected backend…');
    try {
      const r = await request('erosion-compare', {simulation, steps: 64, rain: rain()}); if (!r) return;
      const root = $('erosion-comparison'); root.replaceChildren();
      const heading = document.createElement('h3'); heading.textContent = r.passed ? 'GPU and CPU agree within tolerance' : 'Numerical difference exceeds tolerance'; root.append(heading);
      const table = document.createElement('table');
      for (const [label, value] of [
        ['Same starting grid', `${r.size} × ${r.size} · ${r.steps} steps`],
        ['GPU synchronous compute', r.gpu_ms.toFixed(2) + ' ms'], ['Naive CPU, one thread', r.cpu_ms.toFixed(2) + ' ms'],
        ['CPU time / GPU time', r.speedup.toFixed(2) + '×' + (r.speedup < 1 ? ' (CPU was faster)' : ' (GPU was faster)')],
        ['GPU readback, separate', r.readback_ms.toFixed(2) + ' ms'], ['Whole comparison request', r.request_ms.toFixed(0) + ' ms'],
        ['Max height error', r.terrain_max_error.toExponential(3)], ['Max error, all state', r.max_error.toExponential(3)],
        ['RMS error, all state', r.rmse.toExponential(3)], ['Absolute tolerance', r.tolerance.toExponential(1)]
      ]) { const row = table.insertRow(); const title = document.createElement('th'); title.scope = 'row'; title.textContent = label; row.append(title); row.insertCell().textContent = value; }
      root.append(table); note('Comparison complete. Both ran from the same snapshot; the live terrain was not advanced.');
    } catch (e) { note('Comparison failed: ' + e.message); }
    finally { busy = false; update(); }
  };
  window.erosionDemo = {
    leave: pause,
    targetChanged() { generation++; pause(); simulation = null; selected = null; $('erosion-comparison').replaceChildren(); note('Backend changed. Create terrain on the new backend; the last preview is kept.'); update(); }
  };
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  window.addEventListener('pagehide', pause); redraw(); update();
})();
