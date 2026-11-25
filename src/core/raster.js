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

function newStats() {
  return {
    pipeline: 'correct',
    assets: 0,
    triangles: 0,
    clippedTriangles: 0,
    generatedVertices: 0,
    fanTriangles: 0,
    degenerateSkipped: 0,
    subpixelSkipped: 0,
    nonFiniteSkipped: 0,
    naiveDropped: 0,
    fragmentsTested: 0,
    fragmentsPassed: 0,
    planeHits: {},
  };
}

/**
 * Render a scene produced by buildScene().
 * options: { width, height, only: string[] | null, pipeline: 'correct'|'naive2d' }
 */
export function renderFrame(scene, options = {}) {
  const width = options.width ?? scene.state.viewport.width;
  const height = options.height ?? scene.state.viewport.height;
  const pipeline = options.pipeline ?? 'correct';
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error(`invalid render size ${width}x${height}`);
  }
  if (pipeline !== 'correct' && pipeline !== 'naive2d') {
    throw new Error(`unknown pipeline "${pipeline}"`);
  }

  const buffers = makeBuffers(width, height);
  const clipCtx = createClipContext();
  const prims = [];
  const outlines = [];
  const stats = newStats();
  stats.pipeline = pipeline;

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
        rasterizeTriangle([screenVerts[0], screenVerts[f], screenVerts[f + 1]], {
          width,
          height,
          buffers,
          prims,
          stats,
          assetId: asset.id,
          assetSlot,
          triangleIndex: triangle.index,
          texture: triangle.texture,
          normal: triangle.normal,
          planesHit,
          perspectiveCorrect: pipeline === 'correct',
        });
      }
    }
  }

  stats.generatedVertices = clipCtx.generated.length;

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

function rasterizeTriangle(pts, ctx) {
  const { width, height, buffers, prims, stats } = ctx;
  const { color, depth, maskAsset, maskPrim } = buffers;
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

  const primIndex = prims.length;
  prims.push({
    asset: ctx.assetId,
    assetSlot: ctx.assetSlot,
    triangleIndex: ctx.triangleIndex,
    planesHit: ctx.planesHit,
    perspectiveCorrect: ctx.perspectiveCorrect,
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
      const idx = py * width + px;
      if (!(zPersp < depth[idx])) continue;

      depth[idx] = zPersp;
      maskAsset[idx] = ctx.assetSlot;
      maskPrim[idx] = primIndex;

      const tex = sampleBilinear(ctx.texture, uvx, uvy);
      const o = idx * 3;
      color[o] = Math.min(1, Math.max(0, tex[0] * cr * light));
      color[o + 1] = Math.min(1, Math.max(0, tex[1] * cg * light));
      color[o + 2] = Math.min(1, Math.max(0, tex[2] * cb * light));
      wrote = true;
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
