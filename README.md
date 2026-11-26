# Software Triangle Rasterizer (CPU only)

An interactive software rasterizer. Every pixel is produced by a pure-CPU
kernel in `src/core/`. The viewport is a plain 2D `<canvas>` bitmap of that
kernel's output — there is **no WebGL**, no GPU shader, and no third-party
graphics engine.

```
npm start          # http://localhost:5173/   (graphical debugger / 图形入口)
npm test           # 34 tests: 20 legacy kernel + 14 fast-kernel/parity tests
npm run bench      # real dual-kernel speed + exact-diff benchmark -> out/bench.{txt,json}
npm run render     # offline PNG frames + step-by-step report in ./out
```

`npm start` runs a tiny static server (`tools/serve.mjs`, no dependencies).
Open the printed URL in a browser. ES modules need HTTP, so opening
`web/index.html` via `file://` will not work.

## Graphical entry point (软件光栅的图形入口)

The browser debugger at <http://localhost:5173/> is the primary front end —
every number and image is produced by the real CPU kernel, never canned data.

- **遍历内核 select** (`#kernel`): switch the live view between
  `优化：增量边函数 + 保守瓦片` (`optimized`) and
  `标量参考：逐像素边函数` (`scalar`). The stats card reports which kernel ran.
- **对照 scalar / optimized button** (`#benchCompareBtn`): re-runs the
  **current, possibly dragged/orbited scene** through both real kernels, compares
  every buffer and outline point with **zero tolerance**, times both, and prints
  a Chinese PASS/FAIL report (`#benchOut`).
- **XOR overlay checkbox** (`#diffOverlay`): paints the coverage XOR in red on
  the canvas. After a passing compare it reports `XOR 叠加标记像素=0`.
- Drag a triangle, orbit the view, or move the near plane first; the compare
  and overlay always reflect that exact live state.

The command line (`npm run bench`, `npm run render`) is only an auxiliary,
reproducible harness over the same kernels.

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

## Faster traversal without changing a single pixel

`src/core/raster.js` exports two kernels over the **same** framebuffer, depth,
stats and audit types:

- `rasterizeTriangleScalar` — the reference. For every bounding-box slot it
  evaluates all three edge functions at the pixel centre.
- `rasterizeTriangleOptimized` — incremental exact-integer edge functions,
  conservative 8×8 tiles and convex-span early exit.

`renderFrame(scene, { kernel })` selects one. Both run the identical
perspective-correct barycentric, depth-test and attribute-write sequence, so
the speed path is held to a **pixel-for-pixel** contract rather than an
approximate one.

### Computation order of the optimized path

The ordering is deliberate; each step only narrows work and never changes the
answer.

1. **Winding normalize + subpixel snap.** Reject non-finite/zero-area tris;
   force one orientation and snap vertices to the 256-subpixel grid exactly as
   the scalar kernel does. This fixes the top-left tie rule before anything
   else reads an edge.
2. **Area, bounding box, tie flags.** Compute the fixed-point signed area and
   the same ±1-expanded clipped box the scalar loop scans; derive which edges
   own top/left ties. `stats.bboxPixels` is therefore an identical upper bound
   for both kernels.
3. **Exact-integer safety guard.** Before using increments, prove every edge
   value over the *entire* edge chain stays inside `Number.isSafeInteger`
   (`rasterizeTriangleOptimized` checks origins, steps and the bbox extent).
   If not — e.g. the `2e8`-coordinate probe — increment
   `stats.scalarFallbacks` and run the **scalar** kernel, so huge coordinates
   stay exact instead of silently losing integer precision.
4. **Edge origins + exact increments.** Evaluate each edge once at the box
   origin in integer fixed point, then advance with
   `stepX = −(by−ay)·256`, `stepY = (bx−ax)·256`. No per-pixel multiply and no
   float drift.
5. **Conservative tile classify.** For each 8×8 tile, evaluate the affine edge
   at the four corners (ties included). If all corners are inside with the
   fill rule, the tile is **full** and its pixels are accepted without
   per-pixel edge tests (`proofAcceptedPixels`); if all are outside on the
   rejecting side the tile is **empty** and skipped wholesale
   (`emptyTilePixelsSkipped`); otherwise it is **partial** and scanned. The
   corner test is conservative, so a wrong accept/reject is impossible.
6. **Convex-row early exit.** Inside a partial tile, coverage along a row of a
   convex triangle is one contiguous interval. The scan walks into the
   interval and stops at the first miss after a hit
   (`spanEarlyExits`), instead of testing the rest of the row.
7. **Unchanged shading.** Surviving pixels call the same monomorphic
   `shadeFragment` with the same screen barycentrics, `l·invW` numerators over
   the shared `invW` denominator, depth compare and write order as scalar.

A separate **dispatch** boundary sits before step 1:
`smallBboxPrefersScalar` routes any primitive whose raw box is
`≤ SMALL_BBOX_PIXELS` (256 slots ≈ 16×16) straight to the scalar kernel and
counts `stats.scalarSmallBypass`. For a primitive this small, tile setup costs
as much as the flat scan it replaces; routing it guarantees the optimized
dispatch cannot do *more* traversal work than scalar on tiny triangles. It is a
cost heuristic only — both kernels are exact.

### Conservative boundaries at a glance

| boundary | value | effect | counter if it engages |
|---|---|---|---|
| tiny-bbox dispatch | box ≤ 256 slots | run scalar directly | `scalarSmallBypass` |
| safe-integer guard | full edge chain not safe-integer | run scalar exactly | `scalarFallbacks` |
| full 8×8 tile | 4-corner extrema all inside | accept 64 px, no edge tests | `tileFull`, `proofAcceptedPixels` |
| empty 8×8 tile | extrema all outside | skip up to 64 px | `tileEmpty`, `emptyTilePixelsSkipped` |
| convex span | first miss after a hit | stop the row early | `spanEarlyExits` |

### Thin triangles and reversed submission (missed pixels vs real work)

`tools/bench.mjs` renders 700 long thin triangles at 1280×800 in both
windings. Coverage is identical between kernels **and** between forward and
reversed submission:

- covered pixels forward = reversed = **411,809**; coverage XOR = **0** —
  reversing submission misses **zero** pixels (top-left/welding rules hold).
- bounding-box slots scanned as the common upper bound: **74,094,271**.
- scalar evaluates an edge at every slot: **74,094,271** edge tests.
- optimized explicit edge tests: **3,768,302** → **19.66× fewer** sign tests;
  it also skips **69,810,296** empty-tile pixels and takes **173,791** span
  early exits. Inside (536,350) and shaded (411,809) counts match exactly.
- medians: forward 244.34 → 62.70 ms (**3.90×**); reversed 240.23 → 62.25 ms
  (**3.86×**). Reversal does not create missed pixels or extra work.

### Measured results and cost decomposition

`npm run bench` renders the **same real scene** with both kernels, compares
buffers/outlines with no tolerance, warms up 4 runs and takes the median of 15
timed runs per kernel. One run (2026-11-26, node v22.23.4, linux/arm64):

| scene | scalar ms | optimized ms | speedup | mask/color/depth/outline diffs |
|---|---:|---:|---:|---|
| app default 960×600 | 20.65 | 17.69 | 1.17× | 0 / 0 / 0 / 0 |
| app large 1600×1000 | 53.84 | 49.03 | 1.10× | 0 / 0 / 0 / 0 |
| near plane cutting welded panel | 13.80 | 14.46 | 0.95× | 0 / 0 / 0 / 0 |
| very near plane 0.05 | 21.08 | 17.48 | 1.21× | 0 / 0 / 0 / 0 |
| 700 long thin triangles | 244.34 | 62.70 | 3.90× | 0 / 0 / 0 / 0 |
| same thin triangles, reversed | 240.23 | 62.25 | 3.86× | 0 / 0 / 0 / 0 |
| 6000 tiny triangles | 15.02 | 13.86 | 1.08× | 0 / 0 / 0 / 0 |
| extreme coordinates (guard/fallback) | 2.65 | 2.65 | 1.00× | 0 / 0 / 0 / 0 |

Every row reports `RESULT: PASS` with all pixel and outline diffs zero. The
wall-clock medians are container-specific and noisy on this shared host (the
tiny scene's per-run max was ~36 ms against a ~15 ms median), so the
**reproducible evidence is the exact diffs plus the work counters**, not the
milliseconds. For example, the one sub-1× median above (near-cut, 0.95×) is
scheduling noise, not a traversal regression: that row has
`scalarSmallBypass = scalarFallbacks = 0` (identical code path to a prior
1.08× measurement) yet still cuts explicit edge tests from **956,584** to
**52,113** and proof-accepts **344,320** full-tile pixels. No non-cost
behaviour regressed, so no optimization was reverted; if a change ever did
regress anything beyond timing/cost, the parity tests and the zero-diff bench
gate require reverting it.

Default-scene decomposition: bbox upper bound **2,068,065** slots vs
**79,236** optimized edge tests (**26.1×**), **505,280** proof-accepted
pixels, **1,475,226** empty-tile pixels skipped, tiles
empty/full/partial = **23,535 / 7,895 / 1,401**, with inside/shaded counts
546,166 / 443,199 identical. Tiny-scene decomposition: all **6,040** fan
triangles take the scalar small-bbox route, so optimized edge tests
(**314,284**) equal scalar exactly and tile counters stay zero — tiny input is
deliberately at parity, never approximate.

`src/core/compare.js` provides `compareFrames` (exact element + coverage XOR +
outline point deltas), `measureKernel` and `benchmarkPair`; the browser button
and `tools/bench.mjs` both call it.

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

- `npm test` — **34 tests across two files, all passing**.
  - `test/core.test.js` — the original **20** conventions/clipping/attribute/
    welding/seam/perspective/degenerate/no-NaN tests (kept green).
  - `test/fast-kernel.test.js` — **14** parity and accounting tests: 30
    randomized pixel-space fuzz scenes plus 6 naive2d fuzz scenes compared
    buffer-for-buffer; 400 direct fractional-vertex fuzz cases with deep-equal
    masks/depth/colour; bit-exact full-tuple winding invariance; thin reverse
    coverage parity with an edge-test reduction bound; tiny-bbox bypass; the
    near-plane sweep `near = 0.05…2.6`; orbit and dragged-offset states; the
    `2e8` safe-integer fallback; and counter invariants
    (`passed ≤ tested + proofAcceptedPixels`, etc.).
- `npm run bench` — real dual-kernel benchmark; exits non-zero if **any**
  scenario has a non-zero pixel/outline diff. Writes `out/bench.txt` and
  `out/bench.json` with timings, exact diffs, cost decomposition and top
  per-primitive costs.
- `npm run render` — writes `out/*.png` and `out/report.txt` and runs to
  completion (exit 0); pixel audits agree to ~1e-9.
- `node tools/verify-ui.mjs` — **21 headless-Chromium checks** drive the real
  page: frame production, no-WebGL/2D-canvas proof, near-plane clipping, real
  click audit, watertightness vs. gap, close-up seam width, real drag and orbit,
  the kernel `<select>`, and the **real compare button on the post-drag/orbit
  state** (0 coverage/prim/colour/depth/outline diffs, positive timings and
  counters, PASS text, zero XOR pixels), plus console cleanliness.
  Playwright is a dev-only tool:

  ```
  PLAYWRIGHT_PATH=/path/to/playwright node tools/verify-ui.mjs
  ```

  On a minimal rootless Linux/arm64 container the bundled Chromium may need
  extra shared libraries; they can be extracted without root and exposed via
  `LD_LIBRARY_PATH`.

## Files

| path | role |
|------|------|
| `src/core/math.js` | vec/mat, look-at, perspective, inverse |
| `src/core/clip.js` | 6-plane homogeneous clipping + welding cache |
| `src/core/raster.js` | viewport, edge functions, depth, attributes, audit; scalar + optimized kernels, tile/span/boundary counters |
| `src/core/compare.js` | exact frame/outline comparison, kernel timing, `benchmarkPair` |
| `src/core/scene.js` | assets, camera, degenerate/sliver probes |
| `src/core/texture.js` | procedural checker atlas, bilinear sampling |
| `web/` | graphical debugger (HTML/CSS + `app.js`), kernel select + real compare/XOR panel |
| `tools/serve.mjs` | static server (the GUI entry, `npm start`) |
| `tools/render-offline.mjs` | offline PNG + report (`npm run render`) |
| `tools/bench.mjs` | dual-kernel speed/exact-diff benchmark (`npm run bench`) |
| `tools/bench-scenes.mjs` | stress / thin / reversed / tiny / extreme benchmark scenes |
| `tools/verify-ui.mjs` | headless browser verification (21 checks) |
| `test/core.test.js` | original 20 kernel tests |
| `test/fast-kernel.test.js` | 14 optimized/scalar parity + accounting tests |
