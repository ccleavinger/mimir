# Mimir GPU demo

The HTMX interface uses parchment colors, serif titles and four demo tabs.
Fractal forge and photo blur produce actual images on the selected GPU.
MNIST prediction and image training remain placeholders.

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
