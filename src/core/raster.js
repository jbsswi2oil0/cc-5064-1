// CPU triangle rasterizer.
//
// Pipeline per triangle (correct mode):
//   world -> homogeneous clip -> 6-plane clip -> fan -> perspective divide ->
//   viewport (pixel centers) -> integer subpixel edge functions ->
//   top-left coverage -> depth test -> perspective-correct attributes.
//
// Rasterization uses fixed-point integer edge functions (SUBPIXEL = 256), so
// coverage decisions are exact. Two clipped triangles that share a clip-space
// edge produce bit-identical screen vertices, hence bit-identical integer
// edges. The top-left rule then assigns every boundary pixel to exactly one
// of them: that is what makes a welded asset watertight. Vertices with
// independent keys (the seamed asset) produce different intersections and
// therefore a real, visible gap.
//
// A deliberately broken "naive2d" pipeline is provided for comparison: it
// divides by w first, drops any triangle that straddles w <= 0, clips what
// remains in 2D screen space, and interpolates attributes screen-linearly.
// That is the classic source of near-plane cracks and texture swim.

import { clipTriangle, createClipContext, MIN_W } from './clip.js';
import { sampleBilinear } from './texture.js';
import { shadeFactor } from './scene.js';
import { transform4 } from './math.js';

export const SUBPIXEL = 256;
export const DEPTH_CLEAR = Infinity;
export const BACKGROUND = [0.055, 0.065, 0.09];

function isFiniteVertex(v) {
  return v.every((n) => Number.isFinite(n));
}

function normalizeWinding(pts) {
  const area =
    (pts[1].sx - pts[0].sx) * (pts[2].sy - pts[0].sy) -
    (pts[1].sy - pts[0].sy) * (pts[2].sx - pts[0].sx);
  if (!Number.isFinite(area) || area === 0) return { area: 0, pts };
  if (area < 0) return { area: -area, pts: [pts[0], pts[2], pts[1]] };
  return { area, pts };
}

function topLeft(ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  return dy > 0 || (dy === 0 && dx < 0);
}

function unprojectClip(invViewProj, clip) {
  const x = invViewProj[0] * clip[0] + invViewProj[4] * clip[1] + invViewProj[8] * clip[2] + invViewProj[12] * clip[3];
  const y = invViewProj[1] * clip[0] + invViewProj[5] * clip[1] + invViewProj[9] * clip[2] + invViewProj[13] * clip[3];
  const z = invViewProj[2] * clip[0] + invViewProj[6] * clip[1] + invViewProj[10] * clip[2] + invViewProj[14] * clip[3];
  const w = invViewProj[3] * clip[0] + invViewProj[7] * clip[1] + invViewProj[11] * clip[2] + invViewProj[15] * clip[3];
  if (!Number.isFinite(w) || Math.abs(w) < 1e-300) return [NaN, NaN, NaN];
  return [x / w, y / w, z / w];
}

function makeBuffers(width, height) {
  const n = width * height;
  const color = new Float32Array(n * 3);
  const depth = new Float32Array(n).fill(DEPTH_CLEAR);
  const maskAsset = new Int32Array(n).fill(-1);
  const maskPrim = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    color[i * 3] = BACKGROUND[0];
    color[i * 3 + 1] = BACKGROUND[1];
    color[i * 3 + 2] = BACKGROUND[2];
  }
  return { color, depth, maskAsset, maskPrim };
}

export function newStats() {
  return {
    pipeline: 'correct',
    kernel: 'optimized',
    assets: 0,
    triangles: 0,
    clippedTriangles: 0,
    generatedVertices: 0,
    fanTriangles: 0,
    degenerateSkipped: 0,
    subpixelSkipped: 0,
    nonFiniteSkipped: 0,
    naiveDropped: 0,
    // Real fragment accounting. fragmentsTested counts pixels that actually
    // executed the three edge sign tests; bboxPixels is the traversal upper
    // bound (every bounding-box slot the scalar kernel visits).
    fragmentsTested: 0,
    fragmentsPassed: 0,
    fragmentsShaded: 0,
    bboxPixels: 0,
    proofAcceptedPixels: 0,
    emptyTilePixelsSkipped: 0,
    tileEmpty: 0,
    tileFull: 0,
    tilePartial: 0,
    spanEarlyExits: 0,
    scalarFallbacks: 0,
    scalarSmallBypass: 0,
    msTotal: 0,
    msRaster: 0,
    planeHits: {},
  };
}

function nowMs() {
  return typeof globalThis.performance?.now === 'function' ? globalThis.performance.now() : Date.now();
}

/**
 * Render a scene produced by buildScene().
 * options: {
 *   width, height,
 *   only: string[] | null,
 *   pipeline: 'correct'|'naive2d',
 *   kernel: 'optimized'|'scalar'  (optimized = incremental fixed-point edges
 *                                  + conservative tiles, scalar = reference)
 * }
 */
export function renderFrame(scene, options = {}) {
  const tStart = nowMs();
  const width = options.width ?? scene.state.viewport.width;
  const height = options.height ?? scene.state.viewport.height;
  const pipeline = options.pipeline ?? 'correct';
  const kernel = options.kernel ?? 'optimized';
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error(`invalid render size ${width}x${height}`);
  }
  if (pipeline !== 'correct' && pipeline !== 'naive2d') {
    throw new Error(`unknown pipeline "${pipeline}"`);
  }
  if (kernel !== 'optimized' && kernel !== 'scalar') {
    throw new Error(`unknown raster kernel "${kernel}"`);
  }

  const buffers = makeBuffers(width, height);
  const clipCtx = createClipContext();
  const prims = [];
  const outlines = [];
  const stats = newStats();
  stats.pipeline = pipeline;
  stats.kernel = kernel;
  const rasterFn = kernel === 'scalar' ? rasterizeTriangleScalar : rasterizeTriangleOptimized;

  const vp = scene.viewProj;
  const invVp = scene.invViewProj;
  const assetFilter = options.only ? new Set(options.only) : null;

  for (let assetSlot = 0; assetSlot < scene.assets.length; assetSlot++) {
    const asset = scene.assets[assetSlot];
    if (!scene.state.visible[asset.id]) continue;
    if (assetFilter && !assetFilter.has(asset.id)) continue;
    stats.assets++;

    for (const triangle of asset.triangles) {
      stats.triangles++;
      const clipVerts = [];
      let bad = false;
      for (let i = 0; i < 3; i++) {
        const c = transform4(vp, [triangle.world[i][0], triangle.world[i][1], triangle.world[i][2], 1]);
        if (!isFiniteVertex(c)) {
          bad = true;
          break;
        }
        clipVerts.push({
          key: triangle.keys[i],
          clip: c,
          uv: triangle.uv[i],
          color: triangle.color[i],
          generatedBy: null,
        });
      }
      if (bad) {
        stats.nonFiniteSkipped++;
        continue;
      }

      let screenVerts;
      let planesHit = [];

      if (pipeline === 'correct') {
        const clipped = clipTriangle(clipVerts, clipCtx);
        planesHit = clipped.planesHit;
        if (planesHit.length) stats.clippedTriangles++;
        for (const p of planesHit) stats.planeHits[p] = (stats.planeHits[p] || 0) + 1;
        if (clipped.polygon.length < 3) continue;

        screenVerts = [];
        for (const v of clipped.polygon) {
          const w = v.clip[3];
          if (!(w > MIN_W) || !Number.isFinite(w)) {
            bad = true;
            break;
          }
          const invW = 1 / w;
          const ndcX = v.clip[0] * invW;
          const ndcY = v.clip[1] * invW;
          const ndcZ = v.clip[2] * invW;
          const sx = (ndcX * 0.5 + 0.5) * width;
          const sy = (1 - (ndcY * 0.5 + 0.5)) * height;
          if (![sx, sy, invW, ndcZ].every(Number.isFinite)) {
            bad = true;
            break;
          }
          screenVerts.push({
            key: v.key,
            sx,
            sy,
            invW,
            ndcZ,
            uv: v.uv,
            color: v.color,
            generatedBy: v.generatedBy,
            world: unprojectClip(invVp, v.clip),
          });
        }
        if (bad) {
          stats.nonFiniteSkipped++;
          continue;
        }
        stats.generatedVertices = clipCtx.generated.length;
      } else {
        // Naive: perspective divide first, then treat the triangle as 2D.
        if (clipVerts.some((v) => !(v.clip[3] > MIN_W))) {
          stats.naiveDropped++;
          continue;
        }
        const raw = clipVerts.map((v) => {
          const invW = 1 / v.clip[3];
          return {
            key: v.key,
            sx: (v.clip[0] * invW * 0.5 + 0.5) * width,
            sy: (1 - (v.clip[1] * invW * 0.5 + 0.5)) * height,
            z: v.clip[2] * invW,
            uv: v.uv,
            color: v.color,
            generatedBy: null,
          };
        });
        if (raw.some((v) => Math.abs(v.sx) > 1e7 || Math.abs(v.sy) > 1e7)) {
          stats.naiveDropped++;
          continue;
        }
        const clipped2d = clipPolygon2D(raw, width, height);
        if (clipped2d.length < 3) continue;
        screenVerts = clipped2d.map((v) => ({
          key: v.key,
          sx: v.sx,
          sy: v.sy,
          invW: 1,
          ndcZ: v.z,
          uv: v.uv,
          color: v.color,
          generatedBy: v.generatedBy,
          world: [NaN, NaN, NaN],
        }));
        planesHit = ['screenRect'];
      }

      outlines.push({
        asset: asset.id,
        assetSlot,
        triangleIndex: triangle.index,
        planesHit,
        generated: screenVerts.some((v) => v.generatedBy),
        points: screenVerts.map((v) => ({
          x: v.sx / width,
          y: v.sy / height,
          key: v.key,
          generatedBy: v.generatedBy,
        })),
      });

      for (let f = 1; f + 1 < screenVerts.length; f++) {
        stats.fanTriangles++;
        const triPts = [screenVerts[0], screenVerts[f], screenVerts[f + 1]];
        // Conservative cost boundary applied BEFORE kernel setup: a small
        // bounding box goes straight to the flat scalar reference, avoiding
        // both tile classification and a duplicated setup pass.
        let fn = rasterFn;
        if (kernel === 'optimized' && smallBboxPrefersScalar(triPts, width, height)) {
          stats.scalarSmallBypass++;
          fn = rasterizeTriangleScalar;
        }
        const tR0 = nowMs();
        fn(triPts, {
          width,
          height,
          buffers,
          prims,
          stats,
          kernel,
          assetId: asset.id,
          assetSlot,
          triangleIndex: triangle.index,
          texture: triangle.texture,
          normal: triangle.normal,
          planesHit,
          perspectiveCorrect: pipeline === 'correct',
        });
        stats.msRaster += nowMs() - tR0;
      }
    }
  }

  stats.generatedVertices = clipCtx.generated.length;
  stats.msTotal = nowMs() - tStart;

  return {
    width,
    height,
    pipeline,
    color: buffers.color,
    depth: buffers.depth,
    maskAsset: buffers.maskAsset,
    maskPrim: buffers.maskPrim,
    prims,
    outlines,
    stats,
    clipVertices: clipCtx.generated,
    scene,
  };
}

function clipPolygon2D(poly, width, height) {
  const planes = [
    { side: (v) => v.sx },                   // x >= 0
    { side: (v) => width - v.sx },           // x <= width
    { side: (v) => v.sy },                   // y >= 0
    { side: (v) => height - v.sy },          // y <= height
  ];
  let out = poly;
  for (const plane of planes) {
    const next = [];
    for (let i = 0; i < out.length; i++) {
      const a = out[i];
      const b = out[(i + 1) % out.length];
      const da = plane.side(a);
      const db = plane.side(b);
      if (!Number.isFinite(da) || !Number.isFinite(db)) return [];
      const aIn = da >= 0;
      const bIn = db >= 0;
      if (aIn) next.push(a);
      if (aIn !== bIn) {
        const t = da / (da - db);
        if (!Number.isFinite(t)) return [];
        next.push({
          key: `n(${a.key}|${b.key})`,
          sx: a.sx + t * (b.sx - a.sx),
          sy: a.sy + t * (b.sy - a.sy),
          z: a.z + t * (b.z - a.z),
          uv: [a.uv[0] + t * (b.uv[0] - a.uv[0]), a.uv[1] + t * (b.uv[1] - a.uv[1])],
          color: [
            a.color[0] + t * (b.color[0] - a.color[0]),
            a.color[1] + t * (b.color[1] - a.color[1]),
            a.color[2] + t * (b.color[2] - a.color[2]),
          ],
          generatedBy: 'screenRect',
        });
      }
    }
    out = next;
    if (!out.length) return out;
  }
  return out;
}

/**
 * Scalar reference kernel. Every bounding-box pixel evaluates all three edge
 * functions from the stored fixed-point vertices; barycentrics and attributes
 * use the direct expressions below. This is the correctness baseline and the
 * fallback for primitives whose fixed-point arithmetic would leave the exact
 * safe-integer range. It must stay bit-stable: the optimized kernel is
 * differentially tested against this function.
 */
export function rasterizeTriangleScalar(pts, ctx) {
  const { width, height, buffers, prims, stats } = ctx;
  const { color, depth, maskAsset, maskPrim } = buffers;
  const cost = {
    kernel: 'scalar',
    bboxPixels: 0,
    edgeTests: 0,
    inside: 0,
    depthPassed: 0,
    shaded: 0,
    tileEmpty: 0,
    tileFull: 0,
    tilePartial: 0,
    earlyExits: 0,
    fallback: false,
  };
  const { area, pts: ordered } = normalizeWinding(pts);
  if (area === 0) {
    stats.degenerateSkipped++;
    return false;
  }

  const vx = new Float64Array(3);
  const vy = new Float64Array(3);
  for (let i = 0; i < 3; i++) {
    vx[i] = Math.round(ordered[i].sx * SUBPIXEL);
    vy[i] = Math.round(ordered[i].sy * SUBPIXEL);
    if (!Number.isFinite(vx[i]) || !Number.isFinite(vy[i])) {
      stats.nonFiniteSkipped++;
      return false;
    }
  }
  const areaFixed = (vx[1] - vx[0]) * (vy[2] - vy[0]) - (vy[1] - vy[0]) * (vx[2] - vx[0]);
  if (!Number.isFinite(areaFixed) || areaFixed === 0) {
    stats.subpixelSkipped++;
    return false;
  }

  const minXf = Math.min(vx[0], vx[1], vx[2]);
  const maxXf = Math.max(vx[0], vx[1], vx[2]);
  const minYf = Math.min(vy[0], vy[1], vy[2]);
  const maxYf = Math.max(vy[0], vy[1], vy[2]);
  const minX = Math.max(0, Math.floor(minXf / SUBPIXEL) - 1);
  const maxX = Math.min(width - 1, Math.ceil(maxXf / SUBPIXEL) + 1);
  const minY = Math.max(0, Math.floor(minYf / SUBPIXEL) - 1);
  const maxY = Math.min(height - 1, Math.ceil(maxYf / SUBPIXEL) + 1);
  if (minX > maxX || minY > maxY) return false;

  cost.bboxPixels = (maxX - minX + 1) * (maxY - minY + 1);
  stats.bboxPixels += cost.bboxPixels;

  const primIndex = prims.length;
  prims.push({
    asset: ctx.assetId,
    assetSlot: ctx.assetSlot,
    triangleIndex: ctx.triangleIndex,
    planesHit: ctx.planesHit,
    perspectiveCorrect: ctx.perspectiveCorrect,
    kernel: 'scalar',
    cost,
    vertices: ordered.map((v) => ({
      key: v.key,
      sx: v.sx,
      sy: v.sy,
      invW: v.invW,
      ndcZ: v.ndcZ,
      uv: [...v.uv],
      color: [...v.color],
      generatedBy: v.generatedBy,
      world: [...v.world],
    })),
    areaFixed,
    screenArea: area,
    texture: ctx.texture,
    normal: ctx.normal,
  });

  const invArea = 1 / areaFixed;
  if (!Number.isFinite(invArea)) {
    stats.subpixelSkipped++;
    prims.pop();
    return false;
  }
  const light = shadeFactor(ctx.normal);
  const persp = ctx.perspectiveCorrect;

  const e = (ax, ay, bx, by, px, py) => (bx - ax) * (py - ay) - (by - ay) * (px - ax);
  const tl0 = topLeft(vx[1], vy[1], vx[2], vy[2]);
  const tl1 = topLeft(vx[2], vy[2], vx[0], vy[0]);
  const tl2 = topLeft(vx[0], vy[0], vx[1], vy[1]);
  const a0 = ordered[0];
  const a1 = ordered[1];
  const a2 = ordered[2];

  let wrote = false;
  for (let py = minY; py <= maxY; py++) {
    const fy = Math.round((py + 0.5) * SUBPIXEL);
    for (let px = minX; px <= maxX; px++) {
      const fx = Math.round((px + 0.5) * SUBPIXEL);
      stats.fragmentsTested++;
      cost.edgeTests++;

      const w0 = e(vx[1], vy[1], vx[2], vy[2], fx, fy);
      const w1 = e(vx[2], vy[2], vx[0], vy[0], fx, fy);
      const w2 = e(vx[0], vy[0], vx[1], vy[1], fx, fy);

      if (!(w0 > 0 || (w0 === 0 && tl0))) continue;
      if (!(w1 > 0 || (w1 === 0 && tl1))) continue;
      if (!(w2 > 0 || (w2 === 0 && tl2))) continue;
      const l0 = w0 * invArea;
      const l1 = w1 * invArea;
      const l2 = w2 * invArea;
      if (!Number.isFinite(l0) || !Number.isFinite(l1) || !Number.isFinite(l2)) continue;

      let denom;
      let uvx;
      let uvy;
      let cr;
      let cg;
      let cb;
      let zPersp;
      if (persp) {
        const invW = l0 * a0.invW + l1 * a1.invW + l2 * a2.invW;
        if (!(invW > 0) || !Number.isFinite(invW)) continue;
        denom = invW;
        uvx = (l0 * a0.uv[0] * a0.invW + l1 * a1.uv[0] * a1.invW + l2 * a2.uv[0] * a2.invW) / denom;
        uvy = (l0 * a0.uv[1] * a0.invW + l1 * a1.uv[1] * a1.invW + l2 * a2.uv[1] * a2.invW) / denom;
        cr = (l0 * a0.color[0] * a0.invW + l1 * a1.color[0] * a1.invW + l2 * a2.color[0] * a2.invW) / denom;
        cg = (l0 * a0.color[1] * a0.invW + l1 * a1.color[1] * a1.invW + l2 * a2.color[1] * a2.invW) / denom;
        cb = (l0 * a0.color[2] * a0.invW + l1 * a1.color[2] * a1.invW + l2 * a2.color[2] * a2.invW) / denom;
        zPersp = (l0 * a0.ndcZ * a0.invW + l1 * a1.ndcZ * a1.invW + l2 * a2.ndcZ * a2.invW) / denom;
      } else {
        denom = 1;
        uvx = l0 * a0.uv[0] + l1 * a1.uv[0] + l2 * a2.uv[0];
        uvy = l0 * a0.uv[1] + l1 * a1.uv[1] + l2 * a2.uv[1];
        cr = l0 * a0.color[0] + l1 * a1.color[0] + l2 * a2.color[0];
        cg = l0 * a0.color[1] + l1 * a1.color[1] + l2 * a2.color[1];
        cb = l0 * a0.color[2] + l1 * a1.color[2] + l2 * a2.color[2];
        zPersp = l0 * a0.ndcZ + l1 * a1.ndcZ + l2 * a2.ndcZ;
      }
      if (!Number.isFinite(zPersp) || !Number.isFinite(uvx) || !Number.isFinite(uvy)) continue;

      stats.fragmentsPassed++;
      cost.inside++;
      const idx = py * width + px;
      if (!(zPersp < depth[idx])) continue;
      cost.depthPassed++;

      depth[idx] = zPersp;
      maskAsset[idx] = ctx.assetSlot;
      maskPrim[idx] = primIndex;

      const tex = sampleBilinear(ctx.texture, uvx, uvy);
      const o = idx * 3;
      color[o] = Math.min(1, Math.max(0, tex[0] * cr * light));
      color[o + 1] = Math.min(1, Math.max(0, tex[1] * cg * light));
      color[o + 2] = Math.min(1, Math.max(0, tex[2] * cb * light));
      stats.fragmentsShaded++;
      cost.shaded++;
      wrote = true;
    }
  }
  return wrote;
}

// Tile size for the optimized kernel. 8x8 keeps corner classification cheap
// while the per-row edge increments stay in registers/local variables.
export const TILE_SIZE = 8;

// Dispatch cost boundary for tiny primitives. When the raw bounding box is at
// or below this many slots (about a 16x16 region, a handful of 8x8 tiles),
// tile setup costs as much as the flat scan it replaces, so renderFrame runs
// the scalar reference directly and the optimized kernel cannot do MORE
// traversal work than scalar on tiny triangles. Performance heuristic only:
// either kernel is exact, and the differential tests cover both.
export const SMALL_BBOX_PIXELS = 256;

// Cheap raw-coordinate estimate of the scanned bounding-box slot count. The
// +/-1 margin mirrors the fixed-point kernel's expanded box; it is only a
// performance heuristic, so borderline misclassification stays correct either
// way (both kernels are differentially identical). Fully off-screen fans are
// routed to scalar, whose setup rejects them cheaply.
export function smallBboxPrefersScalar(pts, width, height) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    if (p.sx < minX) minX = p.sx;
    if (p.sx > maxX) maxX = p.sx;
    if (p.sy < minY) minY = p.sy;
    if (p.sy > maxY) maxY = p.sy;
  }
  const x0 = Math.max(0, Math.floor(minX) - 1);
  const x1 = Math.min(width - 1, Math.ceil(maxX) + 1);
  const y0 = Math.max(0, Math.floor(minY) - 1);
  const y1 = Math.min(height - 1, Math.ceil(maxY) + 1);
  if (x0 > x1 || y0 > y1) return true;
  return (x1 - x0 + 1) * (y1 - y0 + 1) <= SMALL_BBOX_PIXELS;
}

/**
 * Shared covered-fragment work for the optimized kernel. This is the exact
 * scalar attribute/depth/write chain, kept in one monomorphic helper so both
 * tile paths run stable JIT-compiled code.
 * Returns 0 (rejected by finite guards), 1 (inside coverage, depth failed),
 * 2 (depth passed and a texel was written).
 */
function shadeFragment(c, w0, w1, w2, px, py) {
  const { stats, cost } = c;
  const invArea = c.invArea;
  const l0 = w0 * invArea;
  const l1 = w1 * invArea;
  const l2 = w2 * invArea;
  if (!Number.isFinite(l0) || !Number.isFinite(l1) || !Number.isFinite(l2)) return 0;

  let denom;
  let uvx;
  let uvy;
  let cr;
  let cg;
  let cb;
  let zPersp;
  if (c.persp) {
    const a0 = c.a0;
    const a1 = c.a1;
    const a2 = c.a2;
    const invW = l0 * a0.invW + l1 * a1.invW + l2 * a2.invW;
    if (!(invW > 0) || !Number.isFinite(invW)) return 0;
    denom = invW;
    uvx = (l0 * a0.uv[0] * a0.invW + l1 * a1.uv[0] * a1.invW + l2 * a2.uv[0] * a2.invW) / denom;
    uvy = (l0 * a0.uv[1] * a0.invW + l1 * a1.uv[1] * a1.invW + l2 * a2.uv[1] * a2.invW) / denom;
    cr = (l0 * a0.color[0] * a0.invW + l1 * a1.color[0] * a1.invW + l2 * a2.color[0] * a2.invW) / denom;
    cg = (l0 * a0.color[1] * a0.invW + l1 * a1.color[1] * a1.invW + l2 * a2.color[1] * a2.invW) / denom;
    cb = (l0 * a0.color[2] * a0.invW + l1 * a1.color[2] * a1.invW + l2 * a2.color[2] * a2.invW) / denom;
    zPersp = (l0 * a0.ndcZ * a0.invW + l1 * a1.ndcZ * a1.invW + l2 * a2.ndcZ * a2.invW) / denom;
  } else {
    const a0 = c.a0;
    const a1 = c.a1;
    const a2 = c.a2;
    denom = 1;
    uvx = l0 * a0.uv[0] + l1 * a1.uv[0] + l2 * a2.uv[0];
    uvy = l0 * a0.uv[1] + l1 * a1.uv[1] + l2 * a2.uv[1];
    cr = l0 * a0.color[0] + l1 * a1.color[0] + l2 * a2.color[0];
    cg = l0 * a0.color[1] + l1 * a1.color[1] + l2 * a2.color[1];
    cb = l0 * a0.color[2] + l1 * a1.color[2] + l2 * a2.color[2];
    zPersp = l0 * a0.ndcZ + l1 * a1.ndcZ + l2 * a2.ndcZ;
  }
  if (!Number.isFinite(zPersp) || !Number.isFinite(uvx) || !Number.isFinite(uvy)) return 0;

  stats.fragmentsPassed++;
  cost.inside++;
  const idx = py * c.width + px;
  if (!(zPersp < c.depth[idx])) return 1;
  cost.depthPassed++;

  c.depth[idx] = zPersp;
  c.maskAsset[idx] = c.assetSlot;
  c.maskPrim[idx] = c.primIndex;

  const tex = sampleBilinear(c.texture, uvx, uvy);
  const o = idx * 3;
  c.color[o] = Math.min(1, Math.max(0, tex[0] * cr * c.light));
  c.color[o + 1] = Math.min(1, Math.max(0, tex[1] * cg * c.light));
  c.color[o + 2] = Math.min(1, Math.max(0, tex[2] * cb * c.light));
  stats.fragmentsShaded++;
  cost.shaded++;
  return 2;
}

/**
 * Optimized kernel. Same fixed-point coverage and same attribute math as the
 * scalar reference; only traversal/work changes:
 *
 * Computation order (important for exact equivalence):
 *   1. identical winding normalization and Math.round(v*256) vertex snap;
 *   2. identical areaFixed / bbox / top-left tie flags;
 *   3. exact-integer guard: only run the fast traversal when every edge
 *      intermediate (coordinate offsets, products, corner values) and every
 *      x/y increment is a safe integer. Otherwise this primitive is handed to
 *      rasterizeTriangleScalar, so near-plane and extreme-coordinate inputs
 *      cannot silently lose precision;
 *   4. edge values at tile/row origins use the SAME direct edge expression as
 *      the scalar kernel; interior values are advanced by exact constants per
 *      pixel-right / row-down step. Inside the guard these integer adds
 *      reproduce the direct products bit-for-bit;
 *   5. each 8x8 tile is classified from the exact edge extrema at its four
 *      extreme sample points:
 *        - empty: some edge fails at every sample (tie-aware max test),
 *        - full : every edge passes at every sample (tie-aware min test),
 *        - partial: anything else -> exact per-pixel top-left tests.
 *      This is a CONSERVATIVE boundary: only provably empty/full tiles take
 *      the bulk path; all boundary/ambiguous tiles get the full scalar test;
 *   6. partial rows scan left to right and stop at the end of the convex
 *      coverage span (the pass set on an integer row is one interval);
 *   7. covered pixels run shadeFragment, which keeps the unchanged
 *      l = w*invArea barycentrics and the unchanged perspective
 *      multiply/divide order, depth test and writes.
 */
export function rasterizeTriangleOptimized(pts, ctx) {
  const { width, height, buffers, prims, stats } = ctx;

  const { area, pts: ordered } = normalizeWinding(pts);
  if (area === 0) {
    stats.degenerateSkipped++;
    return false;
  }

  const vx = new Float64Array(3);
  const vy = new Float64Array(3);
  for (let i = 0; i < 3; i++) {
    vx[i] = Math.round(ordered[i].sx * SUBPIXEL);
    vy[i] = Math.round(ordered[i].sy * SUBPIXEL);
    if (!Number.isFinite(vx[i]) || !Number.isFinite(vy[i])) {
      stats.nonFiniteSkipped++;
      return false;
    }
  }
  const areaFixed = (vx[1] - vx[0]) * (vy[2] - vy[0]) - (vy[1] - vy[0]) * (vx[2] - vx[0]);
  if (!Number.isFinite(areaFixed) || areaFixed === 0) {
    stats.subpixelSkipped++;
    return false;
  }

  const minXf = Math.min(vx[0], vx[1], vx[2]);
  const maxXf = Math.max(vx[0], vx[1], vx[2]);
  const minYf = Math.min(vy[0], vy[1], vy[2]);
  const maxYf = Math.max(vy[0], vy[1], vy[2]);
  const minX = Math.max(0, Math.floor(minXf / SUBPIXEL) - 1);
  const maxX = Math.min(width - 1, Math.ceil(maxXf / SUBPIXEL) + 1);
  const minY = Math.max(0, Math.floor(minYf / SUBPIXEL) - 1);
  const maxY = Math.min(height - 1, Math.ceil(maxYf / SUBPIXEL) + 1);
  if (minX > maxX || minY > maxY) return false;

  // Edge descriptors match the scalar w0,w1,w2 ordering exactly.
  const ex = [
    [vx[1], vy[1], vx[2], vy[2]],
    [vx[2], vy[2], vx[0], vy[0]],
    [vx[0], vy[0], vx[1], vy[1]],
  ];
  const edgeE = (k, px, py) => {
    const d = ex[k];
    return (d[2] - d[0]) * (py - d[1]) - (d[3] - d[1]) * (px - d[0]);
  };
  const tl = [
    topLeft(vx[1], vy[1], vx[2], vy[2]),
    topLeft(vx[2], vy[2], vx[0], vy[0]),
    topLeft(vx[0], vy[0], vx[1], vy[1]),
  ];

  // Exact-integer guard. Require the whole direct edge computation at every
  // bounding corner plus every per-pixel/per-row increment to be an exact safe
  // integer; otherwise repeated addition could diverge from the scalar's
  // direct products, so run the actual scalar reference for this primitive.
  const fxLo = Math.round((minX + 0.5) * SUBPIXEL);
  const fxHi = Math.round((maxX + 0.5) * SUBPIXEL);
  const fyLo = Math.round((minY + 0.5) * SUBPIXEL);
  const fyHi = Math.round((maxY + 0.5) * SUBPIXEL);
  let exactIntegers = Number.isSafeInteger(areaFixed);
  if (exactIntegers) {
    for (let i = 0; i < 3; i++) {
      if (!Number.isSafeInteger(vx[i]) || !Number.isSafeInteger(vy[i])) {
        exactIntegers = false;
        break;
      }
    }
  }
  for (let k = 0; k < 3 && exactIntegers; k++) {
    const d = ex[k];
    const dx = d[2] - d[0];
    const dy = d[3] - d[1];
    const sx = -dy * SUBPIXEL;
    const sy = dx * SUBPIXEL;
    if (
      !Number.isSafeInteger(dx) ||
      !Number.isSafeInteger(dy) ||
      !Number.isSafeInteger(sx) ||
      !Number.isSafeInteger(sy)
    ) {
      exactIntegers = false;
      break;
    }
    for (const fx of [fxLo, fxHi]) {
      for (const fy of [fyLo, fyHi]) {
        const ox = fx - d[0];
        const oy = fy - d[1];
        const p = dx * oy;
        const q = dy * ox;
        const w = p - q;
        if (
          !Number.isSafeInteger(ox) ||
          !Number.isSafeInteger(oy) ||
          !Number.isSafeInteger(p) ||
          !Number.isSafeInteger(q) ||
          !Number.isSafeInteger(w)
        ) {
          exactIntegers = false;
        }
      }
    }
  }
  if (!exactIntegers) {
    stats.scalarFallbacks++;
    return rasterizeTriangleScalar(pts, ctx);
  }

  const stepX = ex.map((d) => -(d[3] - d[1]) * SUBPIXEL);
  const stepY = ex.map((d) => (d[2] - d[0]) * SUBPIXEL);

  const cost = {
    kernel: 'optimized',
    bboxPixels: (maxX - minX + 1) * (maxY - minY + 1),
    edgeTests: 0,
    inside: 0,
    depthPassed: 0,
    shaded: 0,
    tileEmpty: 0,
    tileFull: 0,
    tilePartial: 0,
    earlyExits: 0,
    fallback: false,
  };
  stats.bboxPixels += cost.bboxPixels;

  const primIndex = prims.length;
  prims.push({
    asset: ctx.assetId,
    assetSlot: ctx.assetSlot,
    triangleIndex: ctx.triangleIndex,
    planesHit: ctx.planesHit,
    perspectiveCorrect: ctx.perspectiveCorrect,
    kernel: 'optimized',
    cost,
    vertices: ordered.map((v) => ({
      key: v.key,
      sx: v.sx,
      sy: v.sy,
      invW: v.invW,
      ndcZ: v.ndcZ,
      uv: [...v.uv],
      color: [...v.color],
      generatedBy: v.generatedBy,
      world: [...v.world],
    })),
    areaFixed,
    screenArea: area,
    texture: ctx.texture,
    normal: ctx.normal,
  });

  const invArea = 1 / areaFixed;
  if (!Number.isFinite(invArea)) {
    stats.subpixelSkipped++;
    prims.pop();
    return false;
  }

  const c = {
    width,
    color: buffers.color,
    depth: buffers.depth,
    maskAsset: buffers.maskAsset,
    maskPrim: buffers.maskPrim,
    primIndex,
    assetSlot: ctx.assetSlot,
    stats,
    cost,
    invArea,
    persp: ctx.perspectiveCorrect,
    a0: ordered[0],
    a1: ordered[1],
    a2: ordered[2],
    light: shadeFactor(ctx.normal),
    texture: ctx.texture,
  };

  let wrote = false;
  const TILE = TILE_SIZE;
  const tileX0 = Math.floor(minX / TILE);
  const tileX1 = Math.floor(maxX / TILE);
  const tileY0 = Math.floor(minY / TILE);
  const tileY1 = Math.floor(maxY / TILE);

  for (let ty = tileY0; ty <= tileY1; ty++) {
    const py0 = Math.max(minY, ty * TILE);
    const py1 = Math.min(maxY, ty * TILE + TILE - 1);

    for (let tx = tileX0; tx <= tileX1; tx++) {
      const px0 = Math.max(minX, tx * TILE);
      const px1 = Math.min(maxX, tx * TILE + TILE - 1);

      const fx0 = Math.round((px0 + 0.5) * SUBPIXEL);
      const fx1 = Math.round((px1 + 0.5) * SUBPIXEL);
      const fy0 = Math.round((py0 + 0.5) * SUBPIXEL);
      const fy1 = Math.round((py1 + 0.5) * SUBPIXEL);

      // Conservative tile classification from exact corner extrema.
      let tileEmpty = false;
      let tileFull = true;
      for (let k = 0; k < 3; k++) {
        const c00 = edgeE(k, fx0, fy0);
        const c10 = edgeE(k, fx1, fy0);
        const c01 = edgeE(k, fx0, fy1);
        const c11 = edgeE(k, fx1, fy1);
        const wMax = c00 > c10 ? (c00 > c01 ? (c00 > c11 ? c00 : c11) : c01 > c11 ? c01 : c11) : c10 > c01 ? (c10 > c11 ? c10 : c11) : c01 > c11 ? c01 : c11;
        const wMin = c00 < c10 ? (c00 < c01 ? (c00 < c11 ? c00 : c11) : c01 < c11 ? c01 : c11) : c10 < c01 ? (c10 < c11 ? c10 : c11) : c01 < c11 ? c01 : c11;
        if (tl[k] ? wMax < 0 : wMax <= 0) {
          tileEmpty = true;
          break;
        }
        if (!(tl[k] ? wMin >= 0 : wMin > 0)) tileFull = false;
      }

      const tilePixels = (px1 - px0 + 1) * (py1 - py0 + 1);
      if (tileEmpty) {
        stats.tileEmpty++;
        cost.tileEmpty++;
        stats.emptyTilePixelsSkipped += tilePixels;
        continue;
      }
      if (tileFull) {
        stats.tileFull++;
        cost.tileFull++;
        stats.proofAcceptedPixels += tilePixels;
      } else {
        stats.tilePartial++;
        cost.tilePartial++;
      }

      // Exact edge values at the first sample of the top row; row starts are
      // advanced by stepY, x values by stepX inside the row.
      let rw0 = edgeE(0, fx0, fy0);
      let rw1 = edgeE(1, fx0, fy0);
      let rw2 = edgeE(2, fx0, fy0);
      const sx0 = stepX[0];
      const sx1 = stepX[1];
      const sx2 = stepX[2];
      const sy0 = stepY[0];
      const sy1 = stepY[1];
      const sy2 = stepY[2];

      if (tileFull) {
        for (let py = py0; py <= py1; py++) {
          let w0 = rw0;
          let w1 = rw1;
          let w2 = rw2;
          for (let px = px0; px <= px1; px++) {
            if (shadeFragment(c, w0, w1, w2, px, py) === 2) wrote = true;
            w0 += sx0;
            w1 += sx1;
            w2 += sx2;
          }
          rw0 += sy0;
          rw1 += sy1;
          rw2 += sy2;
        }
      } else {
        for (let py = py0; py <= py1; py++) {
          let w0 = rw0;
          let w1 = rw1;
          let w2 = rw2;
          let entered = false;
          for (let px = px0; px <= px1; px++) {
            stats.fragmentsTested++;
            cost.edgeTests++;
            const pass =
              (w0 > 0 || (w0 === 0 && tl[0])) &&
              (w1 > 0 || (w1 === 0 && tl[1])) &&
              (w2 > 0 || (w2 === 0 && tl[2]));
            if (!pass) {
              if (entered) {
                stats.spanEarlyExits++;
                cost.earlyExits++;
                break;
              }
            } else {
              entered = true;
              if (shadeFragment(c, w0, w1, w2, px, py) === 2) wrote = true;
            }
            w0 += sx0;
            w1 += sx1;
            w2 += sx2;
          }
          rw0 += sy0;
          rw1 += sy1;
          rw2 += sy2;
        }
      }
    }
  }
  return wrote;
}

/**
 * Independent recomputation of a single pixel from the stored primitive.
 * Every intermediate value is returned so the UI can display a numeric chain.
 */
export function auditPixel(frame, px, py) {
  if (!Number.isInteger(px) || !Number.isInteger(py) || px < 0 || py < 0 || px >= frame.width || py >= frame.height) {
    return { ok: false, reason: 'pixel out of range' };
  }
  const idx = py * frame.width + px;
  const primIndex = frame.maskPrim[idx];
  if (primIndex < 0) {
    return {
      ok: false,
      reason: 'background (no coverage)',
      pixel: { x: px, y: py },
      color: [frame.color[idx * 3], frame.color[idx * 3 + 1], frame.color[idx * 3 + 2]],
    };
  }
  const prim = frame.prims[primIndex];
  const [a0, a1, a2] = prim.vertices;
  const vx = prim.vertices.map((v) => Math.round(v.sx * SUBPIXEL));
  const vy = prim.vertices.map((v) => Math.round(v.sy * SUBPIXEL));
  const fx = Math.round((px + 0.5) * SUBPIXEL);
  const fy = Math.round((py + 0.5) * SUBPIXEL);
  const e = (ax, ay, bx, by, qx, qy) => (bx - ax) * (qy - ay) - (by - ay) * (qx - ax);
  const w0 = e(vx[1], vy[1], vx[2], vy[2], fx, fy);
  const w1 = e(vx[2], vy[2], vx[0], vy[0], fx, fy);
  const w2 = e(vx[0], vy[0], vx[1], vy[1], fx, fy);
  const area = prim.areaFixed;
  const l0 = w0 / area;
  const l1 = w1 / area;
  const l2 = w2 / area;

  let invW = 1;
  let uv;
  let colorAttr;
  let ndcZ;
  let perspBarycentric;
  if (prim.perspectiveCorrect) {
    invW = l0 * a0.invW + l1 * a1.invW + l2 * a2.invW;
    perspBarycentric = [
      (l0 * a0.invW) / invW,
      (l1 * a1.invW) / invW,
      (l2 * a2.invW) / invW,
    ];
    uv = [
      (l0 * a0.uv[0] * a0.invW + l1 * a1.uv[0] * a1.invW + l2 * a2.uv[0] * a2.invW) / invW,
      (l0 * a0.uv[1] * a0.invW + l1 * a1.uv[1] * a1.invW + l2 * a2.uv[1] * a2.invW) / invW,
    ];
    colorAttr = [
      (l0 * a0.color[0] * a0.invW + l1 * a1.color[0] * a1.invW + l2 * a2.color[0] * a2.invW) / invW,
      (l0 * a0.color[1] * a0.invW + l1 * a1.color[1] * a1.invW + l2 * a2.color[1] * a2.invW) / invW,
      (l0 * a0.color[2] * a0.invW + l1 * a1.color[2] * a1.invW + l2 * a2.color[2] * a2.invW) / invW,
    ];
    ndcZ = (l0 * a0.ndcZ * a0.invW + l1 * a1.ndcZ * a1.invW + l2 * a2.ndcZ * a2.invW) / invW;
  } else {
    uv = [l0 * a0.uv[0] + l1 * a1.uv[0] + l2 * a2.uv[0], l0 * a0.uv[1] + l1 * a1.uv[1] + l2 * a2.uv[1]];
    colorAttr = [
      l0 * a0.color[0] + l1 * a1.color[0] + l2 * a2.color[0],
      l0 * a0.color[1] + l1 * a1.color[1] + l2 * a2.color[1],
      l0 * a0.color[2] + l1 * a1.color[2] + l2 * a2.color[2],
    ];
    ndcZ = l0 * a0.ndcZ + l1 * a1.ndcZ + l2 * a2.ndcZ;
    perspBarycentric = [l0, l1, l2];
  }
  const tex = sampleBilinear(prim.texture, uv[0], uv[1]);
  const light = shadeFactor(prim.normal);
  const final = [
    Math.min(1, Math.max(0, tex[0] * colorAttr[0] * light)),
    Math.min(1, Math.max(0, tex[1] * colorAttr[1] * light)),
    Math.min(1, Math.max(0, tex[2] * colorAttr[2] * light)),
  ];
  const stored = [frame.color[idx * 3], frame.color[idx * 3 + 1], frame.color[idx * 3 + 2]];
  const maxErr = Math.max(...final.map((c, i) => Math.abs(c - stored[i])));
  return {
    ok: true,
    pixel: { x: px, y: py },
    samplePoint: { x: px + 0.5, y: py + 0.5 },
    subpixel: { x: fx, y: fy, scale: SUBPIXEL },
    primitive: {
      asset: prim.asset,
      triangleIndex: prim.triangleIndex,
      planesHit: prim.planesHit,
      perspectiveCorrect: prim.perspectiveCorrect,
      vertices: prim.vertices,
      areaFixed: area,
      screenArea: prim.screenArea,
    },
    edges: { w0, w1, w2 },
    barycentric: { l0, l1, l2 },
    perspBarycentric,
    invW,
    depthNdc: ndcZ,
    uv,
    colorAttr,
    texel: tex,
    light,
    final,
    stored,
    maxErr,
    agrees: maxErr < 1e-5,
  };
}

export function assetCoverage(frame, assetId) {
  const slot = frame.scene.assets.findIndex((a) => a.id === assetId);
  const mask = new Uint8Array(frame.width * frame.height);
  if (slot < 0) return mask;
  for (let i = 0; i < mask.length; i++) mask[i] = frame.maskAsset[i] === slot ? 1 : 0;
  return mask;
}

export function coverageCount(frame, assetId) {
  const slot = frame.scene.assets.findIndex((a) => a.id === assetId);
  if (slot < 0) return 0;
  let count = 0;
  for (let i = 0; i < frame.maskAsset.length; i++) if (frame.maskAsset[i] === slot) count++;
  return count;
}
