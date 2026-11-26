import { test } from 'node:test';
import assert from 'node:assert/strict';

import { identity4 } from '../src/core/math.js';
import { buildScene, DEFAULT_STATE } from '../src/core/scene.js';
import { makeCheckerTexture } from '../src/core/texture.js';
import {
  renderFrame,
  newStats,
  rasterizeTriangleScalar,
  rasterizeTriangleOptimized,
} from '../src/core/raster.js';
import { compareFrames } from '../src/core/compare.js';
import { thinScene, tinyScene, mulberry32 } from '../tools/bench-scenes.mjs';

// Every test here is a DIFFERENTIAL test: the optimized kernel must reproduce
// the scalar reference buffer-for-buffer and outline-for-outline. It also
// locks the honest work counters that back the speedup claims.

function stateWith(patch = {}) {
  const s = JSON.parse(JSON.stringify(DEFAULT_STATE));
  s.visible = { welded: true, seamed: true, span: true, sliver: true, degenerate: true };
  return Object.assign(s, patch);
}

function assertFramesIdentical(scene, options = {}, label = '') {
  const a = renderFrame(scene, { ...options, kernel: 'scalar' });
  const b = renderFrame(scene, { ...options, kernel: 'optimized' });
  const d = compareFrames(a, b);
  assert.equal(d.maskAssetDiff, 0, `${label}: asset mask diff ${d.maskAssetDiff}`);
  assert.equal(d.maskPrimDiff, 0, `${label}: prim mask diff ${d.maskPrimDiff}`);
  assert.equal(d.colorDiffPixels, 0, `${label}: color diff ${d.colorDiffPixels} (max ${d.maxColorDelta})`);
  assert.equal(d.depthDiff, 0, `${label}: depth diff ${d.depthDiff} (max ${d.maxDepthDelta})`);
  assert.equal(d.outlineCountDiff, 0, `${label}: outline count diff`);
  assert.equal(d.outlinePointDiff, 0, `${label}: outline point count diff`);
  assert.equal(d.outlineMaxDelta, 0, `${label}: outline coordinate delta ${d.outlineMaxDelta}`);
  return { scalar: a, fast: b, diff: d };
}

// ---------------------------------------------------------------------------
// Randomized pixel-space triangles: tiny, huge, thin, reversed, offscreen
// ---------------------------------------------------------------------------

const ID = identity4();
const checker = makeCheckerTexture(32, 4);

function pixelTri(index, ptsPx, width, height) {
  const world = ptsPx.map(([x, y]) => [(2 * x) / width - 1, 1 - (2 * y) / height, 0, 1]);
  return {
    assetId: 'fuzz',
    index,
    keys: [`f${index}a`, `f${index}b`, `f${index}c`],
    world,
    uv: [
      [0.1, 0.1],
      [0.9, 0.2],
      [0.4, 0.9],
    ],
    color: [
      [0.9, 0.3, 0.35],
      [0.3, 0.85, 0.5],
      [0.35, 0.5, 0.95],
    ],
    normal: [0, 0, 1],
    texture: checker,
  };
}

function fuzzScene(seed, { width = 200, height = 150, count = 60 } = {}) {
  const rand = mulberry32(seed);
  const tris = [];
  for (let i = 0; i < count; i++) {
    const cx = rand() * width;
    const cy = rand() * height;
    const kind = rand();
    let pts;
    if (kind < 0.3) {
      // Tiny / sub-pixel.
      const r = 0.2 + rand() * 4;
      pts = Array.from({ length: 3 }, (_, k) => {
        const a = (k * Math.PI * 2) / 3 + rand();
        return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
      });
    } else if (kind < 0.6) {
      // Long thin sliver at a random angle.
      const ang = rand() * Math.PI;
      const len = 40 + rand() * 260;
      const hw = 0.4 + rand() * 2.5;
      const dx = Math.cos(ang);
      const dy = Math.sin(ang);
      pts = [
        [cx + (dx * len) / 2, cy + (dy * len) / 2],
        [cx - (dx * len) / 2 - dy * hw, cy - (dy * len) / 2 + dx * hw],
        [cx - (dx * len) / 2 + dy * hw, cy - (dy * len) / 2 - dx * hw],
      ];
    } else {
      // General triangle, often partly offscreen.
      const r = 5 + rand() * width * 0.7;
      pts = Array.from({ length: 3 }, () => [cx + (rand() - 0.5) * 2 * r, cy + (rand() - 0.5) * 2 * r]);
    }
    if (rand() < 0.5) pts = [pts[2], pts[1], pts[0]];
    tris.push(pixelTri(i, pts, width, height));
  }
  return {
    state: { viewport: { width, height }, visible: { fuzz: true } },
    camera: { eye: [0, 0, 3], target: [0, 0, 0] },
    view: ID,
    proj: ID,
    viewProj: ID,
    invViewProj: ID,
    assets: [{ id: 'fuzz', label: 'fuzz', kind: 'fuzz', triangles: tris, texture: checker }],
  };
}

test('fuzz: optimized and scalar kernels agree exactly on 30 randomized scenes', () => {
  for (let seed = 1; seed <= 30; seed++) {
    const scene = fuzzScene(seed * 7919 + 13);
    assertFramesIdentical(scene, {}, `seed ${seed}`);
  }
});

test('fuzz naive2d pipeline also matches between kernels', () => {
  for (let seed = 100; seed < 106; seed++) {
    const scene = fuzzScene(seed * 104729, { width: 180, height: 120 });
    assertFramesIdentical(scene, { pipeline: 'naive2d' }, `naive seed ${seed}`);
  }
});

// ---------------------------------------------------------------------------
// Adversarial workloads: thin slivers, reversed submission, tiny triangles
// ---------------------------------------------------------------------------

test('thin triangles: reversed submission loses/changes zero coverage pixels', () => {
  const forward = renderFrame(thinScene({ seed: 1337 }), { kernel: 'scalar' });
  const revScalar = renderFrame(thinScene({ reverse: true, seed: 1337 }), { kernel: 'scalar' });
  const revFast = renderFrame(thinScene({ reverse: true, seed: 1337 }), { kernel: 'optimized' });

  // Winding normalization must be coverage-invariant: the same pixels are
  // covered (no missed or extra pixels) when submission order is reversed.
  // Colour is not compared across submissions because clipped polygons are
  // re-fanned from a different anchor and this synthetic scene ties every
  // depth at 0; the kernel-level winding invariance is checked separately.
  const revDiff = compareFrames(forward, revScalar);
  assert.equal(revDiff.maskAssetDiff, 0, `forward/reverse coverage differs: ${revDiff.maskAssetDiff}`);
  assert.equal(revDiff.coveredA, revDiff.coveredB);
  assert.ok(revDiff.coveredA > 300000, `thin scene should cover substantial pixels, got ${revDiff.coveredA}`);

  // The exact correctness gate: on the reversed submission the optimized
  // kernel still reproduces the scalar reference bit-for-bit.
  const d = compareFrames(revScalar, revFast);
  assert.ok(d.identical, `reversed thin optimized != scalar: mask ${d.maskAssetDiff} color ${d.colorDiffPixels}`);
});

test('kernel is bit-exact winding-invariant for a direct perspective primitive', () => {
  // Same triangle as [A,B,C] vs the full tuple reversal [C,B,A]; the kernel
  // normalizes winding and must associate every attribute with its vertex.
  const width = 80;
  const height = 60;
  const mk = (x, y, i) => ({
    key: `k${i}`,
    sx: x,
    sy: y,
    invW: [1.7, 0.6, 1.15][i],
    ndcZ: [0.2, 0.55, 0.4][i],
    uv: [
      [0.1, 0.1],
      [0.9, 0.2],
      [0.4, 0.9],
    ][i],
    color: [
      [0.9, 0.3, 0.35],
      [0.3, 0.85, 0.5],
      [0.35, 0.5, 0.95],
    ][i],
    world: [x, y, 0],
  });
  const p = [
    [10, 8],
    [70, 18],
    [40, 52],
  ];
  const fwd = [mk(p[0][0], p[0][1], 0), mk(p[1][0], p[1][1], 1), mk(p[2][0], p[2][1], 2)];
  const rev = [mk(p[2][0], p[2][1], 2), mk(p[1][0], p[1][1], 1), mk(p[0][0], p[0][1], 0)];
  const a = runScalarOnce(width, height, fwd);
  const b = runScalarOnce(width, height, rev);
  assertBuffersEqual(a, b, 'forward vs reversed winding');
});

function runScalarOnce(width, height, pts) {
  const buffers = directBuffers(width, height);
  const stats = newStats();
  rasterizeTriangleScalar(pts, {
    width,
    height,
    buffers,
    prims: [],
    stats,
    assetId: 'd',
    assetSlot: 0,
    triangleIndex: 0,
    texture: checker,
    normal: [0, 0, 1],
    planesHit: [],
    perspectiveCorrect: true,
  });
  return buffers;
}

test('thin triangles: optimized kernel performs far fewer edge tests with identical coverage', () => {
  const s = renderFrame(thinScene({ count: 700 }), { kernel: 'scalar' });
  const f = renderFrame(thinScene({ count: 700 }), { kernel: 'optimized' });
  assert.equal(f.stats.bboxPixels, s.stats.bboxPixels, 'same bounding-box upper bound');
  assert.ok(
    f.stats.fragmentsTested < s.stats.fragmentsTested * 0.25,
    `edge tests ${f.stats.fragmentsTested} vs scalar ${s.stats.fragmentsTested}`
  );
  assert.ok(f.stats.tilePartial > 0, 'partial tiles were classified and scanned');
  // Coverage counters agree exactly.
  assert.equal(f.stats.fragmentsPassed, s.stats.fragmentsPassed);
  assert.equal(f.stats.fragmentsShaded, s.stats.fragmentsShaded);
});

test('tiny triangles: exact equivalence and no work regression in counters', () => {
  const scene = tinyScene({});
  const { scalar, fast } = assertFramesIdentical(scene);
  assert.ok(fast.stats.scalarSmallBypass > 0, 'small-bbox conservative bypass engaged');
  assert.ok(
    fast.stats.fragmentsTested <= scalar.stats.fragmentsTested,
    `${fast.stats.fragmentsTested} > ${scalar.stats.fragmentsTested}`
  );
});

// ---------------------------------------------------------------------------
// Real app scenes: near-plane sweeps, orbit and drag offsets
// ---------------------------------------------------------------------------

test('buildScene across near-plane sweep matches exactly between kernels', () => {
  for (const near of [0.05, 0.1, 0.15, 0.4, 1.0, 2.0, 2.15, 2.25, 2.4, 2.6]) {
    assertFramesIdentical(buildScene(stateWith({ near })), {}, `near=${near}`);
  }
});

test('buildScene across orbit viewpoints matches exactly', () => {
  const orbits = [
    { yaw: 0, pitch: 0.09, distance: 2.55, target: [0, 0, 0] },
    { yaw: 0.62, pitch: 0.05, distance: 1.1, target: [0, 0, -0.6] },
    { yaw: -1.2, pitch: 0.9, distance: 1.8, target: [0.2, 0.1, 0] },
    { yaw: 2.6, pitch: -0.6, distance: 0.7, target: [0.56, 0.04, 0.02] },
  ];
  for (const orbit of orbits) {
    assertFramesIdentical(buildScene(stateWith({ orbit })), {}, `orbit ${orbit.yaw}`);
  }
});

test('dragged asset offsets (including extreme drags) match exactly', () => {
  const offsets = [
    { welded: [0.4, -0.2, 0.1], seamed: [0, 0, 0], span: [0, 0, 0], sliver: [0, 0, 0] },
    { welded: [-2.5, 1.5, 0], seamed: [1.2, -0.8, 0.3], span: [0, 0.4, -0.2], sliver: [0, 0, 0] },
    { welded: [8, 6, 0], seamed: [-8, -6, 0], span: [0, 0, 0], sliver: [0, 0, 0] },
  ];
  for (const offset of offsets) {
    assertFramesIdentical(buildScene(stateWith({ offsets: offset })), {}, `offset ${offset.welded}`);
  }
});

test('naive2d pipeline on the full default scene matches between kernels', () => {
  assertFramesIdentical(buildScene(stateWith()), { pipeline: 'naive2d' }, 'default naive2d');
});

// ---------------------------------------------------------------------------
// Honest accounting invariants
// ---------------------------------------------------------------------------

test('work counters obey traversal upper bound and stay internally consistent', () => {
  const scenes = [
    buildScene(stateWith()),
    thinScene({ count: 300 }),
    tinyScene({ count: 2000 }),
    fuzzScene(4242, { count: 120 }),
  ];
  for (const scene of scenes) {
    const s = renderFrame(scene, { kernel: 'scalar' });
    const f = renderFrame(scene, { kernel: 'optimized' });
    assert.equal(f.stats.bboxPixels, s.stats.bboxPixels);
    assert.ok(f.stats.fragmentsTested <= s.stats.fragmentsTested, 'optimized never tests more pixels');
    assert.ok(f.stats.fragmentsPassed === s.stats.fragmentsPassed, 'same inside coverage count');
    assert.ok(f.stats.fragmentsShaded === s.stats.fragmentsShaded, 'same shaded count');
    // Explicit edge tests plus proof-accepted full-tile pixels bound the
    // number of inside fragments the optimized kernel can report.
    assert.ok(
      f.stats.fragmentsPassed <= f.stats.fragmentsTested + f.stats.proofAcceptedPixels,
      'passed bounded by tested + proof-accepted'
    );
    assert.ok(f.stats.fragmentsPassed >= f.stats.fragmentsShaded, 'passed >= shaded');
    const classified = f.stats.tileEmpty + f.stats.tileFull + f.stats.tilePartial;
    const scalarRouted = f.stats.scalarSmallBypass + f.stats.scalarFallbacks;
    assert.ok(
      classified > 0 || scalarRouted >= f.stats.fanTriangles,
      'tile classifier ran, or every fan triangle was conservatively routed to scalar'
    );
    assert.ok(f.stats.proofAcceptedPixels >= f.stats.tileFull, 'each full tile proves at least one pixel');
  }
});

// ---------------------------------------------------------------------------
// Direct primitive-level tests: the safe-integer fallback for coordinates the
// clipped render path never produces (astronomical screen positions).
// ---------------------------------------------------------------------------

function directBuffers(width, height) {
  const n = width * height;
  return {
    color: new Float32Array(n * 3),
    depth: new Float32Array(n).fill(Infinity),
    maskAsset: new Int32Array(n).fill(-1),
    maskPrim: new Int32Array(n).fill(-1),
  };
}

function directVertex(x, y, i, persp) {
  return {
    key: `d${i}`,
    sx: x,
    sy: y,
    invW: persp ? [1.7, 0.6, 1.15][i] : 1,
    ndcZ: [0.2, 0.55, 0.4][i],
    uv: [
      [0.1, 0.1],
      [0.9, 0.2],
      [0.4, 0.9],
    ][i],
    color: [
      [0.9, 0.3, 0.35],
      [0.3, 0.85, 0.5],
      [0.35, 0.5, 0.95],
    ][i],
    world: [x, y, 0],
  };
}

function runDirect(kernel, ptsPx, { width = 64, height = 48, persp = true } = {}) {
  const buffers = directBuffers(width, height);
  const stats = newStats();
  const prims = [];
  const pts = ptsPx.map(([x, y], i) => directVertex(x, y, i, persp));
  const ctx = {
    width,
    height,
    buffers,
    prims,
    stats,
    assetId: 'direct',
    assetSlot: 7,
    triangleIndex: 0,
    texture: checker,
    normal: [0, 0, 1],
    planesHit: [],
    perspectiveCorrect: persp,
  };
  const fn = kernel === 'scalar' ? rasterizeTriangleScalar : rasterizeTriangleOptimized;
  fn(pts, ctx);
  return { buffers, stats, prims };
}

function assertBuffersEqual(a, b, label) {
  assert.deepEqual(a.maskAsset, b.maskAsset, `${label}: maskAsset`);
  assert.deepEqual(a.maskPrim, b.maskPrim, `${label}: maskPrim`);
  assert.deepEqual(a.depth, b.depth, `${label}: depth`);
  assert.deepEqual(a.color, b.color, `${label}: color bits`);
}

test('direct fuzz: arbitrary fractional screen triangles match buffer-for-buffer', () => {
  const rand = mulberry32(987654);
  for (let i = 0; i < 400; i++) {
    const w = 24 + Math.floor(rand() * 80);
    const h = 24 + Math.floor(rand() * 60);
    const scale = 1.6; // allow offscreen coordinates
    const pts = Array.from({ length: 3 }, () => [
      (rand() - 0.3) * w * scale,
      (rand() - 0.3) * h * scale,
    ]);
    if (rand() < 0.5) pts.reverse();
    const persp = rand() < 0.5;
    const s = runDirect('scalar', pts, { width: w, height: h, persp });
    const f = runDirect('optimized', pts, { width: w, height: h, persp });
    assertBuffersEqual(s.buffers, f.buffers, `direct fuzz ${i}`);
    assert.equal(f.stats.fragmentsShaded, s.stats.fragmentsShaded, `direct fuzz ${i} shaded count`);
  }
});

test('huge-coordinate primitive trips the exact-integer guard and matches scalar', () => {
  const huge = 2e8;
  const pts = [
    [-huge, -huge],
    [huge * 1.7, -huge * 0.4],
    [huge * 0.2, huge * 1.3],
  ];
  const s = runDirect('scalar', pts, { width: 96, height: 72 });
  const f = runDirect('optimized', pts, { width: 96, height: 72 });
  assert.equal(f.stats.scalarFallbacks, 1, 'optimized kernel counted a safe-integer fallback');
  assertBuffersEqual(s.buffers, f.buffers, 'huge-coordinate fallback');
  assert.ok(f.stats.fragmentsShaded > 1000, 'fallback still rasterized the visible region');
});

test('moderate large primitive uses the tile path without fallback or bypass', () => {
  const pts = [
    [6, 8],
    [118, 20],
    [60, 92],
  ];
  const f = runDirect('optimized', pts, { width: 128, height: 96 });
  assert.equal(f.stats.scalarFallbacks, 0);
  assert.equal(f.stats.scalarSmallBypass, 0, `bbox is large enough for tiles`);
  assert.ok(f.stats.tilePartial + f.stats.tileFull > 0, 'tile path engaged');
  const s = runDirect('scalar', pts, { width: 128, height: 96 });
  assertBuffersEqual(s.buffers, f.buffers, 'moderate primitive');
});
