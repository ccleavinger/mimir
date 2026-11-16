/* Webcam media and frame capture stay in the browser; HTMX sends forms and swaps results. */
(() => {
  const $ = id => document.getElementById(id);
  const video = $('webcam');
  const MAX = 60;
  let classes = [{name: 'Class 1', samples: []}, {name: 'Class 2', samples: []}];
  let stream = null, starting = false, mediaVersion = 0, busy = false, capturing = false;
  let model = null, modelTarget = null, version = 0, predictionBusy = false, timer = null;
  const frame = document.createElement('canvas'); frame.width = frame.height = 32;
  const context = frame.getContext('2d', {willReadFrequently: true});
  function note(message) { $('teach-note').textContent = message; }
  function update() {
    $('teach-train').disabled = busy || capturing || !classes.every(c => c.samples.length >= 10 && c.name.trim());
    $('teach-collection').disabled = busy;
    $('teach-reset').disabled = busy || capturing;
    $('teach-add').disabled = busy || capturing || classes.length >= 4;
    $('camera-start').disabled = !!stream || starting;
    $('camera-stop').disabled = !stream && !starting;
    $('teach-live').disabled = busy || !stream || !model;
    for (const b of document.querySelectorAll('[data-capture]')) b.disabled = busy || capturing || !stream || (classes[Number(b.dataset.capture)]?.samples.length ?? MAX) >= MAX;
    for (const b of document.querySelectorAll('[data-edit]')) b.disabled = busy || capturing;
  }
  function stopLive() {
    clearTimeout(timer); timer = null; $('teach-live').checked = false;
  }
  function invalidate(message) {
    version++; stopLive(); model = null; modelTarget = null;
    $('teach-training').replaceChildren(); $('teach-prediction').replaceChildren();
    if (message) note(message); update();
  }
  function renderClasses() {
    const root = $('teach-classes'); root.replaceChildren();
    classes.forEach((c, index) => {
      const card = document.createElement('div'); card.className = 'teach-class';
      const label = document.createElement('label'); label.textContent = 'Class ' + (index + 1);
      const name = document.createElement('input'); name.type = 'text'; name.maxLength = 40; name.value = c.name;
      name.oninput = () => { c.name = name.value; invalidate('Class names changed. Train again when ready.'); };
      name.disabled = capturing;
      label.append(name); card.append(label);
      const count = document.createElement('span'); count.className = 'teach-count'; count.textContent = `${c.samples.length} / ${MAX} examples`;
      card.append(count);
      const controls = document.createElement('div'); controls.className = 'controls';
      function button(text, fn, capture = false) {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = text;
        if (capture) b.dataset.capture = index; else b.dataset.edit = '';
        b.onclick = fn; controls.append(b);
      }
      button('Capture one', () => capture(index, 1), true);
      button('Capture 10', () => capture(index, 10), true);
      button('Undo', () => { c.samples.pop(); invalidate('Examples changed. Train again when ready.'); renderClasses(); });
      button('Clear', () => { c.samples = []; invalidate('Examples cleared.'); renderClasses(); });
      if (classes.length > 2) button('Remove class', () => { classes.splice(index, 1); invalidate('Classes changed.'); renderClasses(); });
      card.append(controls);
      const thumbnails = document.createElement('div'); thumbnails.className = 'thumbnails';
      for (const sample of c.samples.slice(-8)) { const img = document.createElement('img'); img.src = sample.thumbnail; img.alt = 'Captured example for ' + c.name; thumbnails.append(img); }
      card.append(thumbnails); root.append(card);
    });
    update();
  }
  function cameraFrame() {
    if (!stream || video.readyState < 2 || !video.videoWidth) throw new Error('Wait for the camera preview before capturing.');
    const side = Math.min(video.videoWidth, video.videoHeight);
    // Match the square, mirrored preview exactly in training and prediction.
    context.save(); context.translate(32, 0); context.scale(-1, 1);
    context.drawImage(video, (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side, 0, 0, 32, 32); context.restore();
    const rgba = context.getImageData(0, 0, 32, 32).data, pixels = [];
    for (let i = 0; i < rgba.length; i += 4) pixels.push(rgba[i], rgba[i + 1], rgba[i + 2]);
    return {pixels, thumbnail: frame.toDataURL('image/png')};
  }
  async function capture(index, amount) {
    if (busy || capturing || !stream) return;
    const token = mediaVersion; capturing = true; invalidate(); renderClasses();
    try {
      for (let i = 0; i < amount && classes[index].samples.length < MAX; i++) {
        if (!stream || token !== mediaVersion) break;
        classes[index].samples.push(cameraFrame()); renderClasses();
        note(`Collecting ${classes[index].name}: ${classes[index].samples.length} examples. Move the object gently.`);
        if (amount > 1) await new Promise(resolve => setTimeout(resolve, 180));
      }
      note('Examples ready. Collect other classes or train on the selected GPU.');
    } catch (e) { note(e.message); }
    finally { capturing = false; renderClasses(); }
  }
  function stopCamera() {
    mediaVersion++; stopLive();
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; starting = false; video.srcObject = null; $('camera-placeholder').hidden = false;
    $('camera-note').textContent = 'Camera stopped. Captured examples stay in this page until reset or reload.';
    update();
  }
  $('camera-start').onclick = async () => {
    if (starting || stream) return;
    if (!navigator.mediaDevices?.getUserMedia) { $('camera-note').textContent = 'Camera access requires localhost or HTTPS. Open the laptop UI at http://localhost:8080, or use an HTTPS connection.'; return; }
    const token = ++mediaVersion; starting = true; update();
    try {
      const media = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}, facingMode: 'user'}, audio: false});
      if (token !== mediaVersion) { media.getTracks().forEach(t => t.stop()); return; }
      stream = media; video.srcObject = media; await video.play();
      if (token !== mediaVersion) return;
      media.getVideoTracks()[0].addEventListener('ended', () => { if (stream === media) stopCamera(); });
      $('camera-placeholder').hidden = true;
      $('camera-note').textContent = 'Keep the object inside the square. Capture different angles, distances and positions.';
    } catch (e) {
      if (token === mediaVersion) { stopCamera(); $('camera-note').textContent = 'Could not start the camera. Allow camera permission and check that another app is not using it. ' + e.message; }
    } finally { if (token === mediaVersion) { starting = false; update(); } }
  };
  $('camera-stop').onclick = stopCamera;
  $('teach-add').onclick = () => { if (classes.length < 4 && !busy && !capturing) { classes.push({name: 'Class ' + (classes.length + 1), samples: []}); invalidate(); renderClasses(); } };
  $('teach-reset').onclick = () => { if (busy || capturing) return; classes = [{name: 'Class 1', samples: []}, {name: 'Class 2', samples: []}]; invalidate('Lesson reset. Collect examples for each class.'); renderClasses(); };
  $('teach-train').onclick = async () => {
    if (busy || capturing) return;
    const names = classes.map(c => c.name.trim().toLowerCase());
    if (new Set(names).size !== names.length) { note('Give each class a unique name.'); return; }
    invalidate(); busy = true; update();
    const selected = target.value, token = version;
    const wasDisabled = target.disabled; target.disabled = true;
    $('teach-note').setAttribute('aria-busy', 'true'); note('Training on ' + target.selectedOptions[0].textContent + '…');
    try {
      await htmx.ajax('POST', '/run/' + selected + '/teach-train', {
        target: '#teach-training', values: {classes: JSON.stringify(classes.map(c => ({name: c.name.trim(), samples: c.samples.map(s => s.pixels)})))}
      });
      if (token !== version) return;
      const result = $('teach-training').querySelector('[data-teach-model]');
      if (!result) { note('Training failed. Check the message below and try again.'); return; }
      model = Number(result.dataset.teachModel); modelTarget = selected;
      note('Training complete. Turn on live predictions and try examples the camera has not seen.');
    } catch (e) { note('Could not reach the selected backend. Check the connection and train again.'); }
    finally { busy = false; target.disabled = wasDisabled; $('teach-note').removeAttribute('aria-busy'); update(); }
  };
  async function predict() {
    if (!$('teach-live').checked || !stream || !model || busy || predictionBusy) return;
    if (target.value !== modelTarget) { invalidate('Backend changed. Train this lesson on the new backend.'); return; }
    const token = version;
    predictionBusy = true;
    try {
      const sample = cameraFrame();
      await htmx.ajax('POST', '/run/' + modelTarget + '/teach-predict', {target: '#teach-prediction', values: {model, pixels: JSON.stringify(sample.pixels)}});
      if (token !== version) { $('teach-prediction').replaceChildren(); return; }
      if (!$('teach-prediction').querySelector('[data-teach-prediction]')) { stopLive(); note('Live prediction stopped. Check the error, or train again if the backend restarted.'); }
    } catch (e) { if (token === version) { stopLive(); note('Prediction stopped: ' + e.message); } }
    finally {
      predictionBusy = false;
      if (token === version && $('teach-live').checked) timer = setTimeout(predict, 350);
    }
  }
  $('teach-live').onchange = () => { if ($('teach-live').checked) predict(); else stopLive(); };
  window.teachDemo = {leave: stopCamera, targetChanged: () => invalidate('Backend changed. Captured examples are kept; train them on the new device.')};
  window.addEventListener('pagehide', stopCamera);
  renderClasses();
})();
