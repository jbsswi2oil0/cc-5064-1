// Differential comparison + honest timing shared by the CLI benchmark and the
// in-browser comparison panel. Every comparison runs BOTH real software
// kernels on the SAME scene; nothing here uses canned coverage/outline data.

import { renderFrame } from './raster.js';

/**
 * Exact comparison of two frames produced by renderFrame(). Coverage masks,
 * primitive ids, depth and colour are compared element by element. Returns a
 * JSON-safe summary plus a Uint8Array XOR coverage mask for visualization.
 */
export function compareFrames(a, b) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`frame size mismatch ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  }
  const n = a.width * a.height;
  const coverageXor = new Uint8Array(n);
  let maskAssetDiff = 0;
  let maskPrimDiff = 0;
  let colorDiffPixels = 0;
  let depthDiff = 0;
  let maxColorDelta = 0;
  let maxDepthDelta = 0;
  let coveredA = 0;
  let coveredB = 0;

  for (let i = 0; i < n; i++) {
    if (a.maskAsset[i] >= 0) coveredA++;
    if (b.maskAsset[i] >= 0) coveredB++;
    if (a.maskAsset[i] !== b.maskAsset[i]) {
      maskAssetDiff++;
      coverageXor[i] = 1;
    }
    if (a.maskPrim[i] !== b.maskPrim[i]) maskPrimDiff++;
    if (a.depth[i] !== b.depth[i]) {
      // Infinity === Infinity; only count real numeric divergence.
      const da = a.depth[i];
      const db = b.depth[i];
      if (Number.isFinite(da) && Number.isFinite(db)) {
        depthDiff++;
        maxDepthDelta = Math.max(maxDepthDelta, Math.abs(da - db));
      } else if (da !== db) {
        depthDiff++;
      }
    }
    let pixelDiff = false;
    for (let k = 0; k < 3; k++) {
      const delta = Math.abs(a.color[i * 3 + k] - b.color[i * 3 + k]);
      if (delta > 0) pixelDiff = true;
      if (delta > maxColorDelta) maxColorDelta = delta;
    }
    if (pixelDiff) colorDiffPixels++;
  }

  // Outlines (post-clip contours) must match coordinate-for-coordinate because
  // both kernels share the exact same clip + viewport stage.
  let outlinePointCountA = 0;
  let outlinePointCountB = 0;
  let outlineMaxDelta = 0;
  const outlinesA = a.outlines || [];
  const outlinesB = b.outlines || [];
  for (let o = 0; o < Math.max(outlinesA.length, outlinesB.length); o++) {
    const pa = outlinesA[o]?.points || [];
    const pb = outlinesB[o]?.points || [];
    outlinePointCountA += pa.length;
    outlinePointCountB += pb.length;
    for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
      outlineMaxDelta = Math.max(
        outlineMaxDelta,
        Math.abs(pa[i].x - pb[i].x),
        Math.abs(pa[i].y - pb[i].y)
      );
    }
  }
  const outlineCountDiff = outlinesA.length !== outlinesB.length ? Math.abs(outlinesA.length - outlinesB.length) : 0;
  const outlinePointDiff = Math.abs(outlinePointCountA - outlinePointCountB);

  const identical =
    maskAssetDiff === 0 &&
    maskPrimDiff === 0 &&
    colorDiffPixels === 0 &&
    depthDiff === 0 &&
    outlineCountDiff === 0 &&
    outlinePointDiff === 0 &&
    outlineMaxDelta === 0;

  return {
    width: a.width,
    height: a.height,
    pixels: n,
    coveredA,
    coveredB,
    maskAssetDiff,
    maskPrimDiff,
    coverageMissedOrExtra: maskAssetDiff,
    colorDiffPixels,
    depthDiff,
    maxColorDelta,
    maxDepthDelta,
    outlineCountA: outlinesA.length,
    outlineCountB: outlinesB.length,
    outlinePointCountA,
    outlinePointCountB,
    outlineCountDiff,
    outlinePointDiff,
    outlineMaxDelta,
    identical,
    coverageXor,
  };
}

function median(xs) {
  const s = [...xs].sort((x, y) => x - y);
  return s.length % 2 ? s[(s.length - 1) >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

function compactStats(s) {
  return {
    kernel: s.kernel,
    triangles: s.triangles,
    fanTriangles: s.fanTriangles,
    clippedTriangles: s.clippedTriangles,
    generatedVertices: s.generatedVertices,
    degenerateSkipped: s.degenerateSkipped,
    subpixelSkipped: s.subpixelSkipped,
    nonFiniteSkipped: s.nonFiniteSkipped,
    bboxPixels: s.bboxPixels,
    fragmentsTested: s.fragmentsTested,
    fragmentsPassed: s.fragmentsPassed,
    fragmentsShaded: s.fragmentsShaded,
    proofAcceptedPixels: s.proofAcceptedPixels,
    emptyTilePixelsSkipped: s.emptyTilePixelsSkipped,
    tileEmpty: s.tileEmpty,
    tileFull: s.tileFull,
    tilePartial: s.tilePartial,
    spanEarlyExits: s.spanEarlyExits,
    scalarFallbacks: s.scalarFallbacks,
    scalarSmallBypass: s.scalarSmallBypass,
    msRasterLast: Number(s.msRaster.toFixed(3)),
    msTotalLast: Number(s.msTotal.toFixed(3)),
  };
}

/**
 * Time one kernel with warmup and interleaved-friendly repeated runs.
 */
export function measureKernel(scene, options = {}) {
  const kernel = options.kernel ?? 'optimized';
  const pipeline = options.pipeline ?? 'correct';
  const warmup = options.warmup ?? 5;
  const iterations = options.iterations ?? 21;
  const renderOptions = { kernel, pipeline };
  if (options.width) renderOptions.width = options.width;
  if (options.height) renderOptions.height = options.height;

  for (let i = 0; i < warmup; i++) renderFrame(scene, renderOptions);
  const times = [];
  let last = null;
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    last = renderFrame(scene, renderOptions);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return {
    kernel,
    iterations,
    medianMs: Number(median(times).toFixed(4)),
    minMs: Number(times[0].toFixed(4)),
    maxMs: Number(times[times.length - 1].toFixed(4)),
    meanMs: Number((times.reduce((a, b) => a + b, 0) / times.length).toFixed(4)),
    stats: compactStats(last.stats),
  };
}

/**
 * Full before/after comparison for one scene: exact frame diff plus timed
 * scalar reference vs optimized kernel and per-primitive cost leaders.
 */
export function benchmarkPair(scene, options = {}) {
  const pipeline = options.pipeline ?? 'correct';
  const scalarFrame = renderFrame(scene, { kernel: 'scalar', pipeline });
  const fastFrame = renderFrame(scene, { kernel: 'optimized', pipeline });
  const diff = compareFrames(scalarFrame, fastFrame);

  const scalar = measureKernel(scene, { ...options, kernel: 'scalar', pipeline });
  const optimized = measureKernel(scene, { ...options, kernel: 'optimized', pipeline });

  const speedup = scalar.medianMs > 0 ? scalar.medianMs / optimized.medianMs : 0;

  const primCosts = fastFrame.prims
    .map((p) => ({
      asset: p.asset,
      triangle: p.triangleIndex,
      planesHit: p.planesHit?.length ? p.planesHit.join('+') : '-',
      kernel: p.kernel,
      ...p.cost,
    }))
    .sort((a, b) => b.bboxPixels - a.bboxPixels)
    .slice(0, options.topCosts ?? 10);

  return {
    pipeline,
    scalar,
    optimized,
    speedup: Number(speedup.toFixed(3)),
    diff: { ...diff, coverageXor: undefined },
    coverageXor: diff.coverageXor,
    primCosts,
  };
}
