# Software Triangle Rasterizer (CPU only)

An interactive software rasterizer. Every pixel is produced by a pure-CPU
kernel in `src/core/`. The viewport is a plain 2D `<canvas>` bitmap of that
kernel's output — there is **no WebGL**, no GPU shader, and no third-party
graphics engine.

```
npm start          # http://localhost:5173/   (interactive viewport)
npm test           # 20 numeric kernel tests
npm run render     # offline PNG frames + step-by-step report in ./out
```

`npm start` runs a tiny static server (`tools/serve.mjs`, no dependencies).
Open the printed URL in a browser. ES modules need HTTP, so opening
`web/index.html` via `file://` will not work.

## Pipeline

```
model/world ──viewProj──▶ homogeneous clip space ──6-plane clip──▶
perspective divide ──viewport + pixel centres──▶ fixed-point edge functions
──top-left coverage──▶ depth test ──perspective-correct uv/colour──▶ framebuffer
```

- **Handedness / depth**: right-handed world, camera looks down **−Z**.
  Clip space is OpenGL-style `−w ≤ x,y,z ≤ w`, near plane is `z = −w`,
  NDC z ∈ `[−1, 1]` (`−1` at near, `+1` at far). Depth test is "smaller NDC z
  wins".
- **Screen mapping**: NDC → viewport, origin top-left, y down, pixel centres
  at `(x+0.5, y+0.5)`.
- **Coverage**: 8-bit subpixel fixed point (`SUBPIXEL = 256`) integer edge
  functions plus the top-left fill rule, so shared-edge coverage is exact and
  cracks cannot appear from floating-point round-off.
- **Clipping**: done in 4D **before** the perspective divide (`src/core/clip.js`).
  uv and colour are interpolated with the **same** parameter `t` as the
  position, so attributes are perspective-correct by construction. A pipeline
  that divides by `w` first and clips in 2D is provided as `pipeline: 'naive2d'`
  for comparison — it drops any triangle that crosses `w = 0`.
- **Degenerate / unsafe input**: zero-area triangles are counted and skipped;
  any non-finite value (from `w ≤ 0`, a zero denominator, etc.) is rejected
  before it can reach the framebuffer.

## Watertight welding vs. intentional seams

Each vertex carries a stable string key. When clipping creates an intersection
vertex, its key is `plane + (sorted endpoint keys)`. Consequences:

- Two triangles that **share** an original edge (welded asset) get a
  **bit-identical** intersection vertex regardless of traversal direction, so
  their clipped screen edges coincide exactly and the top-left rule leaves
  zero holes — confirmed by `welded pair has zero interior holes` and
  `welded triangles agree on 1/w and uv along their shared edge`.
- Two triangles with **independent** keys (seamed asset, the "different
  precision" version) get distinct intersections, so the hairline seam stays
  open and the background shows through — confirmed by the seam tests.

The seam is authored smaller than the other panel features. Push the camera in
(`贴近缝隙` button, or scroll) until the projected seam exceeds 1 px; the info
line reports the projected width in pixels and whether it is wider than one
sample spacing. At that point the welded and seamed assets produce visibly
different coverage masks.

## Numeric pixel audit

Click any pixel in the viewport. The panel shows the exact chain the kernel
used and **recomputes** it independently from the stored primitive:

- integer edge functions `w0,w1,w2` and fixed-point area
- screen barycentrics `l0,l1,l2`
- `1/w` interpolation denominator
- perspective-correct barycentrics
- interpolated uv and colour attributes
- bilinear texel sample, light factor
- recomputed colour vs. stored framebuffer colour (max channel error)

The same audit runs headless in `npm run render` (section 4 of the report) and
in the test `pixel audit reproduces stored colours from barycentrics + inverse w`.

## Interaction drives the kernel

- **Drag a triangle**: the pick reads the rasterizer's own coverage mask
  (`maskAsset`) to decide what was hit; the drag updates a world-space offset
  and re-runs the rasterizer. Nothing is moved by a separate scene graph.
- **Orbit**: drag empty space; **zoom**: wheel.
- **Near / FOV / seam sliders**: each change rebuilds the clip+view matrices
  and re-rasterizes.
- **扫描近裁面**: sweeps the near plane through the scene, ping-ponging once.

## Verification

- `npm test` — 20 assertions covering conventions, clipping, attribute
  interpolation, welding identity, seam growth, perspective correctness,
  degenerate handling, and no-NaN-leak.
- `npm run render` — writes `out/*.png` and `out/report.txt` with the geometry
  and pixel tables.
- `tools/verify-ui.mjs` — headless Chromium drives the real page and checks
  frame production, near-plane clipping, pixel audit, watertightness vs. gap,
  close-up seam width, drag, orbit, and console cleanliness. Playwright is a
  dev-only tool:

  ```
  PLAYWRIGHT_PATH=/path/to/playwright node tools/verify-ui.mjs
  ```

## Files

| path | role |
|------|------|
| `src/core/math.js` | vec/mat, look-at, perspective, inverse |
| `src/core/clip.js` | 6-plane homogeneous clipping + welding cache |
| `src/core/raster.js` | viewport mapping, edge functions, depth, attributes, audit |
| `src/core/scene.js` | assets, camera, degenerate/sliver probes |
| `src/core/texture.js` | procedural checker atlas, bilinear sampling |
| `web/` | viewport (HTML/CSS + `app.js`) |
| `tools/serve.mjs` | static server |
| `tools/render-offline.mjs` | offline PNG + report |
| `tools/verify-ui.mjs` | headless UI verification |
| `test/core.test.js` | kernel tests |
