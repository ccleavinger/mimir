# Mimir GPU demo

The HTMX interface uses parchment colors, serif titles and four demo tabs.
Fractal forge and photo blur produce actual images on the selected GPU.
MNIST drawing runs dense-network inference on the selected GPU. Image training remains a placeholder.

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

Place the version 1 model from `mnist-cnn` at
`examples/demo/assets/mnist-model.json` on each backend machine before starting.
The backend loads it once, checks the 784 → 256 → 10 architecture, parameter
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
