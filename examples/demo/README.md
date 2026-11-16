# Mimir GPU demo

The HTMX interface uses parchment colors, serif titles and five demo tabs.
Fractal forge and photo blur produce actual images on the selected GPU.
MNIST drawing runs dense-network inference on the selected GPU. The webcam lesson trains a small image classifier on the selected GPU.

```sh
# Laptop: website on :8080, Vulkan API on :8081
MIMIR_DEMO_TARGETS='Laptop=http://127.0.0.1:8081,Jetson=http://127.0.0.1:9081' \
  cargo run -p mimir-demo --release

# Jetson: CUDA API only on :8081
cargo run -p mimir-demo --release --no-default-features --features cuda

# Laptop: forward Jetson's API
ssh -N -L 9081:127.0.0.1:8081 user@jetson
```

Use `MIMIR_DEMO_UI_ADDR` and `MIMIR_DEMO_API_ADDR` to change listen addresses.
Defaults bind to `127.0.0.1`. These variables and `MIMIR_DEMO_TARGETS` support
compile-time defaults and runtime overrides. Targets are comma-separated
`Name=http://host:port` entries. Both machines should run the same API version.
The UI server forwards requests; the browser need only reach the UI port.

Build features: `ui`, `backend`, `vulkan`, `cuda`. Defaults are
`ui,backend,vulkan`; each GPU feature enables `backend`. For UI alone use
`--no-default-features --features ui` with remote targets configured.
HTMX and optional title fonts load from CDNs. Title fonts have local serif fallbacks.

## Image demos

Fractal forge computes 691,200 Mandelbrot pixels, each with up to 512 iterations.
Select magnification and render to explore the boundary.

Photo blur accepts images the browser can decode, resizes to fit within 768×768,
and composites transparency onto white. The GPU runs horizontal and vertical
box-blur passes with a radius of 1–32 pixels. The original and blurred image are
displayed side by side. Animated formats use the frame decoded by the browser.

API endpoints:

- `GET /api/status`: backend, devices and available demos.
- `POST /api/fractal`: JSON `{ "zoom": 1 }`, magnification 1–10000.
- `POST /api/blur`: JSON `{ "width": W, "height": H, "radius": 8, "pixels": [...] }`.
  Pixels are flat RGBA byte values; dimensions must be 1–768 and radius 1–32.

Image responses contain an `image` BMP data URL, dimensions, `elapsed_ms` and
`detail`. Timings include allocation, first-use compilation, GPU work and
readback, and exclude encoding and network transfer. They are demo timings.
The UI renders image responses as HTML fragments rather than printing JSON.
Requests use Content-Length framing with a 16 MiB body limit. Validation/GPU
errors return JSON with status 503. This is a trusted-network demo without auth.

## MNIST inference

Place the trained model from `mnist` at
`examples/demo/assets/mnist-model.json` on each backend machine before starting.
The backend loads it once, checks connected layer shapes (784 inputs, 10 outputs), parameter
counts and finite values, and uploads weights before opening any listening ports.
A missing or invalid model terminates startup with its full path and reason.
The asset path is relative to the example source directory at build time,
independent of the launch working directory. UI-only builds do not need a model.

Draw one digit and press **Predict digit**. The browser fits the drawing within
20×20 pixels, centers its ink mass in a 28×28 frame, and converts it to light ink
on black. The result shows the predicted digit, softmax scores for all ten
classes, and a preview of exactly what the model received. Dense layers and
ReLU run on the selected GPU; the ten output logits are normalized on the host.

`POST /api/mnist` accepts JSON `{ "pixels": [...] }` with exactly 784 grayscale
byte values in row-major order (0 background, 255 ink). Values are normalized to
0–1 for inference. The response contains `prediction`, `probabilities`,
`elapsed_ms`, and an `image` data URL for the 28×28 input. Empty drawings and
malformed inputs return an error. Timings include input upload, first-use
compilation, dense inference and output readback.

The current trainer uses two hidden layers (784 → 512 → 128 → 10), corrected
ReLU gradients, Adam, and translation augmentation. Its version 2 model format
is supported alongside existing version 1 models. To train and replace the demo
asset only after passing 95% on the held-out MNIST test set:

```sh
MNIST_OUTPUT=examples/demo/assets/mnist-model.json cargo run -p mnist --release
```

See `examples/mnist/README.md` for training settings. Restart backend services
after replacing the asset; models are loaded once at startup. Copy the qualified
asset to the same demo assets directory on the Jetson as well. MNIST test accuracy
does not guarantee the same accuracy for freehand drawings.

## Webcam lesson

Open **Teach images**, start the camera, and name two to four classes. Capture
10–60 frames per class using single capture or ten-frame bursts, moving the
object between views. The cards show recent thumbnails; Undo, Clear and Remove
let you correct the collection. Include a background/empty-hand class if useful.
Press **Train on GPU**, then enable **Live predictions** to see class scores.
Add more varied examples and retrain when the model confuses classes.

The preview is a mirrored center square. Training and prediction use exactly
that crop, resized to 32×32 RGB. A 3072 → 64 → class-count dense network trains
from scratch with Adam for 150 full-batch epochs through Mimir. Training is
synchronous with a busy indicator. Every fifth frame is withheld for a quick
validation score; adjacent frames are correlated, so this is not a measure of
recognition in new environments. Use visibly distinct objects or poses with
similar backgrounds and lighting to those captured. No pretrained image model
or extra asset download is needed.

The browser handles camera permission and capture. Use localhost (including
`127.0.0.1`) or HTTPS for browser camera access. Opening the website over a plain
HTTP LAN IP generally disables getUserMedia. The existing SSH forwarding setup
keeps the UI on laptop localhost and can send training to Jetson.

Only captured samples and live inference frames are sent to the selected
backend. Live requests run one at a time, with a 350ms pause between responses.
Turning off the camera, leaving the tab or closing the page stops capture.
Examples remain in browser memory until reset/reload. Editing examples or
changing backend invalidates the selected model and requires retraining.
Models stay only in backend memory: each backend retains its latest eight,
existing lookups retain their model during inference, and restart clears them.
Only one training run per backend runs at a time; another receives a busy error.

- `POST /api/teach-train`: JSON `{ "classes": [{ "name": "Cup", "samples": [[...3072 RGB bytes...], ...] }, ...] }`.
  Requires 2–4 unique named classes and 10–60 samples each. Returns `model` (a
  backend-local ID), `names`, `samples`, `epochs`, `training_accuracy`,
  `validation_accuracy` and `elapsed_ms`.
- `POST /api/teach-predict`: JSON `{ "model": ID, "pixels": [...3072 RGB bytes...] }`.
  Returns `names`, `scores`, `winner` (class index), and `elapsed_ms`.

`/run/{target}/teach-train` and `/run/{target}/teach-predict` accept corresponding
browser form values and return HTMX fragments. Live scores are labeled with
class names rather than printing raw JSON. Invalid/expired models require a
new training run; the UI displays the error and stops live requests.

## Terrain erosion and CPU comparison

**Terrain erosion** creates a deterministic landscape on the selected backend.
Choose a 128², 256², or 512² (default) grid and a seed, then create terrain. Run
rainfall, pause, or advance 64 steps. Rainfall strength can change during the
simulation. Raised-terrain and overhead views are rendered locally from the same
preview. Brown/orange highlights exposed soil, sand highlights deposits, and
blue shows water. Original and current terrain remain visible side by side.

This is a simplified hydraulic erosion model: one pass computes four downhill
water outflows per cell, constrained by available water. A second transports
water and sediment, erodes/deposits according to flow and local slope, and
evaporates water. Boundaries are closed. Terrain plus sediment is conserved
apart from floating-point error. The model is intended for a visual compute
demo, not geophysical prediction or an interacting particle solver.

The backend retains state, scratch space and flux buffers on the GPU. Each
preview averages the full grid down to 64² cells and returns one-byte heights
and water amounts as hexadecimal strings: about 16 KiB plus metadata, independent
of simulation grid size. The browser projects this into terrain views. A single
request advances 64 steps, then pauses 800 ms before the next request; requests
never overlap. Leaving the tab, hiding the page, changing backend or encountering
a connection error pauses the loop. The last preview remains visible on errors.
The simulation advances only on requests and does not run in the background.
Up to four simulations per backend remain in memory; restart or eviction requires
creating terrain again. Changing backend requires a new terrain on that target.

**Compare CPU / GPU** pauses the animation and copies the current state into
independent GPU and single-thread CPU buffers. Both execute the same 64 steps
with identical rainfall; the live terrain is not changed. The CPU uses scalar
f32 loops with matching boundaries and equations, without a numerical library.
The comparison checks all full-resolution height, water and sediment values
before preview quantization, reporting maximum absolute error and RMS error.
The displayed pass tolerance is 0.002 in normalized simulation units.

GPU pipelines are warmed on scratch data. GPU timing covers synchronous launches
and submission overhead; CPU timing covers the scalar simulation loop. Both
exclude allocation and input cloning. GPU readback is shown separately. The
whole request timing includes GPU/CPU work, readback, serialization and network
latency, and is not presented as pure Wi-Fi latency. Both computations run on
the selected backend machine. The UI reports the actual ratio even if CPU wins;
these single measurements are illustrative, not a hardware benchmark suite.
Use release builds for a meaningful CPU comparison.

API endpoints (JSON):

- `POST /api/erosion-create`: `{ "size": 256, "seed": 42, "rain": 0.002 }`.
- `POST /api/erosion-step`: `{ "simulation": ID, "steps": 64, "rain": 0.002 }`.
- `POST /api/erosion-compare`: same arguments as step, with no live-state change.

Size is 128/256/512, seed is u32, steps are 1–128 and rain is 0–0.01. Create/step
return simulation ID, total steps, size, seed, 64² preview dimensions, height and
water hex strings, `gpu_ms` and `readback_ms`. Comparison returns `cpu_ms`,
`gpu_ms`, `speedup` (CPU time / GPU time), `readback_ms`, maximum terrain/all-state
errors, RMS error, tolerance, and pass/fail. The UI's `/sim/{target}/{operation}`
JSON gateway forwards to the API and adds `proxy_ms`. Browser timings measure
the whole request; preview payload size is measured from the response bytes.
