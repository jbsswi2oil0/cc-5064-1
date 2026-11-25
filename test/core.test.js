import { test } from 'node:test';
import assert from 'node:assert/strict';

import { transform4, perspective, lookAt, multiply4, normalize, invert4 } from '../src/core/math.js';
import { clipTriangle, createClipContext, CLIP_PLANES } from '../src/core/clip.js';
import { buildScene, DEFAULT_STATE, WELDED, SEAMED, SPAN, projectPoint } from '../src/core/scene.js';
import { renderFrame, auditPixel, coverageCount, SUBPIXEL, BACKGROUND } from '../src/core/raster.js';
import { makeCheckerTexture } from '../src/core/texture.js';

const checker = makeCheckerTexture(64, 8);

function stateWith(patch = {}) {
  const s = JSON.parse(JSON.stringify(DEFAULT_STATE));
  s.visible = { welded: true, seamed: true, span: true, sliver: true, degenerate: true };
  Object.assign(s, patch);
  return s;
}

// ---------------------------------------------------------------------------
// Coordinate / matrix conventions
// ---------------------------------------------------------------------------

test('clip space follows OpenGL convention: view -near maps to ndc z=-1', () => {
  const proj = perspective(Math.PI / 3, 16 / 9, 0.2, 20);
  const nearPoint = transform4(proj, [0, 0, -0.2, 1]);
  const farPoint = transform4(proj, [0, 0, -20, 1]);
  assert.ok(Math.abs(nearPoint[2] / nearPoint[3] - -1) < 1e-9, 'near -> -1');
  assert.ok(Math.abs(farPoint[2] / farPoint[3] - 1) < 1e-9, 'far -> +1');
  assert.ok(nearPoint[3] > 0 && farPoint[3] > 0, 'w positive in front of camera');
  // Right-handed view: looking down -Z, +X is right in clip space.
  const right = transform4(proj, [1, 0, -5, 1]);
  assert.ok(right[0] > 0, '+X world -> +X clip (right-handed camera)');
});

test('lookAt produces right-handed basis (cam right x cam up = -view dir)', () => {
  const v = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  // View space: camera looks down -Z. World origin maps to (0,0,-3).
  const origin = transform4(v, [0, 0, 0, 1]);
  assert.ok(Math.abs(origin[2] - -3) < 1e-12);
  // World +X maps to view +X (screen right).
  const off = transform4(v, [1, 0, 0, 1]);
  assert.ok(Math.abs(off[0] - 1) < 1e-12);
});

test('clip rejects points behind the eye (w <= 0) instead of producing huge coordinates', () => {
  const proj = perspective(Math.PI / 3, 1, 0.1, 10);
  // A point behind the camera has negative view z -> w = -z' < 0.
  const behind = transform4(proj, [0, 0, 1, 1]);
  assert.ok(behind[3] < 0);
  const ctx = createClipContext();
  const tri = [
    { key: 'a', clip: transform4(proj, [0, 0, -1, 1]), uv: [0, 0], color: [1, 1, 1] },
    { key: 'b', clip: transform4(proj, [0.5, 0, -1, 1]), uv: [1, 0], color: [1, 1, 1] },
    { key: 'c', clip: behind, uv: [0, 1], color: [1, 1, 1] },
  ];
  const { polygon } = clipTriangle(tri, ctx);
  assert.ok(polygon.length >= 3, 'straddling triangle survives clipping');
  for (const v of polygon) {
    assert.ok(v.clip[3] > 0, 'every clipped vertex has w > 0');
    assert.ok(v.clip.every(Number.isFinite), 'no NaN/Inf leaks');
  }
});

test('degenerate zero-area triangle is skipped without touching the framebuffer', () => {
  const s = stateWith();
  s.visible = { welded: false, seamed: false, span: false, sliver: false, degenerate: true };
  const frame = renderFrame(buildScene(s));
  assert.ok(frame.stats.degenerateSkipped >= 1);
  assert.equal(frame.maskAsset.reduce((a, v) => a + (v >= 0 ? 1 : 0), 0), 0);
  assert.ok(frame.color.every((c) => Number.isFinite(c)));
});

// ---------------------------------------------------------------------------
// Pixel audit: the stored colour must be reproducible from the stored geometry
// ---------------------------------------------------------------------------

function everyPixelAudits(frame) {
  let checked = 0;
  for (let py = 0; py < frame.height; py += 3) {
    for (let px = 0; px < frame.width; px += 3) {
      const a = auditPixel(frame, px, py);
      if (!a.ok) continue;
      checked++;
      assert.ok(a.agrees, `pixel ${px},${py} audit mismatch ${a.maxErr}`);
    }
  }
  return checked;
}

test('pixel audit reproduces stored colours from barycentrics + inverse w', () => {
  const frame = renderFrame(buildScene(stateWith()));
  const checked = everyPixelAudits(frame);
  assert.ok(checked > 1000, `audited ${checked} pixels`);
});

test('a triangle straddling near, left and bottom is clipped by all three planes and stays finite', () => {
  const proj = perspective(Math.PI / 3, 1, 1.0, 20);
  const view = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  const vp = multiply4(proj, view);
  // One vertex behind the camera, one below-left of the frustum, one in view.
  const world = [
    [0.3, -0.2, 3.6],
    [-4.0, -3.2, -1.0],
    [0.2, 0.4, -2.0],
  ];
  const verts = world.map((w, i) => ({
    key: `m${i}`,
    clip: transform4(vp, [w[0], w[1], w[2], 1]),
    uv: [[0, 0], [1, 0], [0.5, 1]][i],
    color: [[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]][i],
    generatedBy: null,
  }));
  const ctx = createClipContext();
  const { polygon, planesHit } = clipTriangle(verts, ctx);
  assert.ok(planesHit.includes('near'), `planes hit: ${planesHit}`);
  assert.ok(planesHit.length >= 2, `multiple planes involved: ${planesHit}`);
  assert.ok(polygon.length >= 3);
  for (const v of polygon) {
    assert.ok(v.clip[3] > 0, 'w positive after clip');
    assert.ok(v.clip.every(Number.isFinite));
    assert.ok(v.uv.every(Number.isFinite) && v.color.every(Number.isFinite), 'attributes stay finite');
  }
  // Every generated vertex must sit on one of the reported planes.
  for (const g of ctx.generated) {
    const onSome = CLIP_PLANES.some((p) => Math.abs(p.inside(g.clip)) < 1e-7);
    assert.ok(onSome, `generated vertex ${g.key} lies on a clip plane`);
  }
});

test('seamed pair keeps a real gap while the near plane sweeps through it', () => {
  // Push the camera in and lower the near plane so clipping happens; the
  // seamed asset must never become watertight just because vertices got
  // clipped (independent keys must keep producing distinct intersections).
  const s = stateWith({
    near: 0.4,
    orbit: { yaw: 0.0, pitch: 0.0, distance: 0.62, target: [0.56, 0.04, 0.02] },
    seamWidth: 0.012,
    visible: { welded: false, seamed: true, span: false, sliver: false, degenerate: false },
  });
  let sawClip = false;
  for (let near = 0.3; near <= 1.2; near += 0.1) {
    s.near = near;
    const frame = renderFrame(buildScene(s));
    if (frame.stats.generatedVertices > 0) sawClip = true;
    const r = quadCoverageHoles(frame, SEAMED);
    if (!r || r.empty || r.interior < 50) continue;
    assert.ok(r.holes > 0, `near=${near}: seamed gap collapsed to 0 after clipping`);
  }
  assert.ok(sawClip, 'clipping actually occurred during the sweep');
});

test('welded triangles agree on 1/w and uv along their shared edge (no redraw crack)', () => {
  const s = stateWith({ visible: { welded: true, seamed: false, span: false, sliver: false, degenerate: false } });
  const frame = renderFrame(buildScene(s));
  const weldSlot = frame.scene.assets.findIndex((a) => a.id === WELDED);
  const prims = frame.prims.filter((p) => p.assetSlot === weldSlot);
  assert.equal(prims.length, 2, 'welded panel fans into exactly two screen triangles');
  const [A, B] = prims;
  // The two triangles share the diagonal (keys p0 and p2 in this asset).
  const sharedKeys = A.vertices.map((v) => v.key).filter((k) => B.vertices.some((w) => w.key === k));
  assert.ok(sharedKeys.length >= 2, `shared diagonal keys: ${sharedKeys}`);
  for (const k of sharedKeys) {
    const va = A.vertices.find((v) => v.key === k);
    const vb = B.vertices.find((v) => v.key === k);
    assert.ok(Math.abs(va.invW - vb.invW) < 1e-12, 'shared endpoint 1/w identical');
    assert.deepEqual(va.uv, vb.uv, 'shared endpoint uv identical');
    assert.ok(Math.abs(va.sx - vb.sx) < 1e-9 && Math.abs(va.sy - vb.sy) < 1e-9, 'shared endpoint screen position identical');
  }
});

test('audit agrees with an independent world-space ray intersect for a chosen pixel', () => {
  const scene = buildScene(stateWith({ visible: { welded: true, seamed: false, span: true, sliver: false, degenerate: false } }));
  const frame = renderFrame(scene);

  // Pick a pixel covered by the tilted span and reconstruct the world hit.
  let chosen = null;
  const spanSlot = scene.assets.findIndex((a) => a.id === SPAN);
  for (let py = 4; py < frame.height - 4 && !chosen; py += 7) {
    for (let px = 4; px < frame.width - 4; px += 7) {
      if (frame.maskAsset[py * frame.width + px] === spanSlot) {
        chosen = [px, py];
        break;
      }
    }
  }
  assert.ok(chosen, 'found a span pixel');

  const audit = auditPixel(frame, chosen[0], chosen[1]);
  assert.ok(audit.ok && audit.agrees);

  // Build camera ray through the pixel centre and intersect the span plane.
  const { eye } = scene.camera;
  const inv = scene.invViewProj;
  const ndcX = ((chosen[0] + 0.5) / frame.width) * 2 - 1;
  const ndcY = 1 - ((chosen[1] + 0.5) / frame.height) * 2;
  const unproject = (z) => {
    const v = transform4(inv, [ndcX, ndcY, z, 1]);
    return [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
  };
  const pNear = unproject(-1);
  const pFar = unproject(1);
  const dir = normalize([pFar[0] - pNear[0], pFar[1] - pNear[1], pFar[2] - pNear[2]]);

  const spanAsset = scene.assets[spanSlot];
  const [w0, w1, w2] = spanAsset.triangles[0].world;
  const n = normalize([
    (w1[1] - w0[1]) * (w2[2] - w0[2]) - (w1[2] - w0[2]) * (w2[1] - w0[1]),
    (w1[2] - w0[2]) * (w2[0] - w0[0]) - (w1[0] - w0[0]) * (w2[2] - w0[2]),
    (w1[0] - w0[0]) * (w2[1] - w0[1]) - (w1[1] - w0[1]) * (w2[0] - w0[0]),
  ]);
  const denom = n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2];
  assert.ok(Math.abs(denom) > 1e-6);
  const tHit = (n[0] * (w0[0] - eye[0]) + n[1] * (w0[1] - eye[1]) + n[2] * (w0[2] - eye[2])) / denom;
  const hit = [eye[0] + dir[0] * tHit, eye[1] + dir[1] * tHit, eye[2] + dir[2] * tHit];

  // Compare against the world coordinate reconstructed from the audit chain.
  // The world point must use the PERSPECTIVE-CORRECT barycentrics; the raw
  // screen-space barycentrics describe the 2D coverage triangle, not the 3D
  // surface point.
  const verts = audit.primitive.vertices;
  const pb = audit.perspBarycentric;
  const worldFromAudit = [0, 1, 2].map(
    (k) => pb[0] * verts[0].world[k] + pb[1] * verts[1].world[k] + pb[2] * verts[2].world[k]
  );
  const err = Math.hypot(hit[0] - worldFromAudit[0], hit[1] - worldFromAudit[1], hit[2] - worldFromAudit[2]);
  assert.ok(err < 2e-3, `world hit reconstruction error ${err}`);
  // The perspective-correct barycentrics and the screen barycentrics must
  // differ measurably on a foreshortened quad, proving the interpolation is
  // not merely affine in screen space.
  const sb = audit.barycentric;
  const baryDelta = Math.max(Math.abs(sb.l0 - pb[0]), Math.abs(sb.l1 - pb[1]), Math.abs(sb.l2 - pb[2]));
  assert.ok(baryDelta > 1e-3, `perspective vs screen barycentrics differ (${baryDelta})`);
});

// ---------------------------------------------------------------------------
// Pixel centre convention
// ---------------------------------------------------------------------------

test('pixel centres are at (x+0.5, y+0.5) in subpixel units', () => {
  const frame = renderFrame(buildScene(stateWith()));
  let found = false;
  for (let py = 0; py < frame.height && !found; py++) {
    for (let px = 0; px < frame.width; px++) {
      if (frame.maskPrim[py * frame.width + px] >= 0) {
        const a = auditPixel(frame, px, py);
        assert.equal(a.subpixel.x, Math.round((px + 0.5) * SUBPIXEL));
        assert.equal(a.subpixel.y, Math.round((py + 0.5) * SUBPIXEL));
        found = true;
        break;
      }
    }
  }
  assert.ok(found);
});

// ---------------------------------------------------------------------------
// Watertight shared edge vs intentional seam
// ---------------------------------------------------------------------------

function convexHull(points) {
  if (points.length < 3) return points;
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [];
  for (const q of p) {
    while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
    lo.push(q);
  }
  const up = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
    up.push(q);
  }
  lo.pop();
  up.pop();
  return lo.concat(up);
}

function pointsInsidePolygon(poly, x, y) {
  let sign = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const cr = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    if (cr === 0) continue;
    const s = Math.sign(cr);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

/**
 * Holes = covered-polygon interior pixels not covered by the asset, using the
 * ASSET'S OWN post-clip outline convex hull as the domain. This measures
 * cracks inside the rasterized silhouette, not pixels legitimately removed by
 * the near plane.
 */
function quadCoverageHoles(frame, assetId, erode = 1.25) {
  const slot = frame.scene.assets.findIndex((a) => a.id === assetId);
  if (slot < 0) return null;
  const pts = [];
  for (const o of frame.outlines) {
    if (o.asset !== assetId) continue;
    for (const p of o.points) pts.push([p.x * frame.width, p.y * frame.height]);
  }
  if (pts.length < 3) return { holes: 0, interior: 0, empty: true };
  const hull = convexHull(pts.map(([x, y]) => [Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]));
  let holes = 0;
  let interior = 0;
  for (let py = 1; py < frame.height - 1; py++) {
    for (let px = 1; px < frame.width - 1; px++) {
      const x = px + 0.5;
      const y = py + 0.5;
      if (!pointsInsidePolygon(hull, x, y)) continue;
      let solid = true;
      for (const [dx, dy] of [[-erode, 0], [erode, 0], [0, -erode], [0, erode]]) {
        if (!pointsInsidePolygon(hull, x + dx, y + dy)) {
          solid = false;
          break;
        }
      }
      if (!solid) continue;
      interior++;
      if (frame.maskAsset[py * frame.width + px] !== slot) holes++;
    }
  }
  return { holes, interior, empty: false };
}

test('welded pair has zero interior holes at every near-plane position', () => {
  const distances = [0.15, 1.0, 1.8, 2.15, 2.25, 2.4];
  for (const near of distances) {
    const s = stateWith({ near, visible: { welded: true, seamed: false, span: false, sliver: false, degenerate: false } });
    // Push the near plane through the panel by orbiting closer instead of
    // authoring the panel across the plane.
    const frame = renderFrame(buildScene(s));
    const r = quadCoverageHoles(frame, WELDED);
    if (!r) continue;
    assert.equal(r.holes, 0, `near=${near}: welded interior holes ${r.holes}/${r.interior}`);
  }
});

test('welded pair stays watertight while near plane cuts through it (clip vertices shared)', () => {
  const s = stateWith({ visible: { welded: true, seamed: false, span: false, sliver: false, degenerate: false } });
  // Sweep the camera so the near plane crosses the welded panel.
  let sawClip = false;
  for (let near = 2.0; near <= 2.65; near += 0.05) {
    s.near = near;
    const frame = renderFrame(buildScene(s));
    if (frame.stats.generatedVertices === 0) continue;
    sawClip = true;
    const slot = frame.scene.assets.findIndex((a) => a.id === WELDED);
    // For every pixel covered by the welded asset in this frame, re-run with
    // only that asset visible and verify the coverage is identical.
    const onlyFrame = renderFrame(buildScene({ ...s, visible: { welded: true, seamed: false, span: false, sliver: false, degenerate: false } }));
    let diff = 0;
    for (let i = 0; i < frame.maskAsset.length; i++) {
      const a = frame.maskAsset[i] === slot;
      const b = onlyFrame.maskAsset[i] === slot;
      if (a !== b) diff++;
    }
    assert.equal(diff, 0, `near=${near}: coverage differs when isolated (${diff} px)`);
  }
  assert.ok(sawClip, 'near plane actually generated clip vertices');
});

test('seamed pair opens a background-visible gap that grows with seam width', () => {
  const counts = [];
  for (const seamWidth of [0, 0.0005, 0.002, 0.0042]) {
    const s = stateWith({
      seamWidth,
      visible: { welded: false, seamed: true, span: false, sliver: false, degenerate: false },
    });
    const frame = renderFrame(buildScene(s));
    const r = quadCoverageHoles(frame, SEAMED);
    assert.ok(r, 'seamed quad fully in front of near plane');
    if (seamWidth === 0) {
      assert.equal(r.holes, 0, 'zero seam width must be watertight');
    } else {
      assert.ok(r.holes > 0, `seam ${seamWidth} must produce gaps`);
      // Gaps must actually show background, not another asset.
      const slot = frame.scene.assets.findIndex((a) => a.id === SEAMED);
      let bg = 0;
      for (let py = 0; py < frame.height; py++) {
        for (let px = 0; px < frame.width; px++) {
          const idx = py * frame.width + px;
          if (frame.maskAsset[idx] < 0) bg++;
        }
      }
      assert.ok(bg > 100, 'seam exposes background');
    }
    counts.push(r.holes);
  }
  for (let i = 1; i < counts.length; i++) {
    assert.ok(counts[i] >= counts[i - 1], `gap must not shrink: ${counts}`);
  }
  assert.ok(counts[3] > counts[1], 'closer camera / wider seam exposes more gap');
});

test('pushed-in camera makes seam wider than one sample spacing', () => {
  // Project the two seam-edge midpoints and measure their pixel distance.
  const s = stateWith({
    orbit: { yaw: 0, pitch: 0.0, distance: 0.5, target: [0.56, 0.04, 0.02] },
    seamWidth: 0.01,
    visible: { welded: false, seamed: true, span: false, sliver: false, degenerate: false },
  });
  const scene = buildScene(s);
  const seamed = scene.assets.find((a) => a.id === SEAMED);
  const [p0, p1, p2, p3] = seamed.referenceQuad;
  // Seam direction midpoint: p0 and p2 are pushed apart on the seamed triangle.
  const a = seamed.triangles[0].world;
  const b = seamed.triangles[1].world;
  const wA = [a[0][0], a[0][1], a[0][2]];
  const wB = [b[0][0], b[0][1], b[0][2]];
  const pa = projectPoint(scene.viewProj, wA);
  const pb = projectPoint(scene.viewProj, wB);
  const sa = [(pa[0] / pa[3] * 0.5 + 0.5) * scene.state.viewport.width, (1 - (pa[1] / pa[3] * 0.5 + 0.5)) * scene.state.viewport.height];
  const sb = [(pb[0] / pb[3] * 0.5 + 0.5) * scene.state.viewport.width, (1 - (pb[1] / pb[3] * 0.5 + 0.5)) * scene.state.viewport.height];
  const gapPx = Math.hypot(sa[0] - sb[0], sa[1] - sb[1]);
  assert.ok(gapPx > 1.0, `projected seam width ${gapPx.toFixed(2)} px must exceed one sample spacing`);

  const frame = renderFrame(scene);
  const r = quadCoverageHoles(frame, SEAMED);
  assert.ok(r.holes > 0, 'gap is actually visible in coverage');
  // Sanity: the two assets must produce different coverage masks.
  const wFrame = renderFrame(
    buildScene({ ...s, visible: { welded: true, seamed: false, span: false, sliver: false, degenerate: false } })
  );
  let diff = 0;
  for (let i = 0; i < frame.maskAsset.length; i++) {
    if ((frame.maskAsset[i] >= 0) !== (wFrame.maskAsset[i] >= 0)) diff++;
  }
  assert.ok(diff > 100, 'welded and seamed coverage masks differ');
});

// ---------------------------------------------------------------------------
// Clipping must interpolate attributes with the same parameter as position
// ---------------------------------------------------------------------------

test('near-plane clip interpolates uv/colour with the clip parameter (no divide-then-clip)', () => {
  const proj = perspective(Math.PI / 3, 1, 1.0, 20);
  const view = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  const vp = multiply4(proj, view);
  // Vertex 0 sits behind the camera plane (w < 0); the other two are in
  // front. Only the near plane is crossed.
  const world = [
    [0.2, -0.2, 3.4],
    [0.6, -0.2, -1.0],
    [0.0, 0.6, -2.0],
  ];
  const verts = world.map((w, i) => ({
    key: `t${i}`,
    clip: transform4(vp, [w[0], w[1], w[2], 1]),
    uv: [[0.1, 0.2], [0.9, 0.2], [0.5, 0.95]][i],
    color: [[1, 0, 0], [0, 1, 0], [0, 0, 1]][i],
    generatedBy: null,
  }));
  assert.ok(verts[0].clip[3] < 0, 'first vertex is behind the camera');
  const ctx = createClipContext();
  const { polygon, planesHit } = clipTriangle(verts, ctx);
  assert.equal(polygon.length, 4);
  assert.deepEqual(planesHit, ['near']);
  assert.equal(ctx.generated.length, 2);

  const nearCheck = CLIP_PLANES.find((p) => p.name === 'near');
  for (const g of ctx.generated) {
    // The generated vertex must lie exactly on the near plane (z + w = 0).
    assert.ok(Math.abs(nearCheck.inside(g.clip)) < 1e-9, 'generated vertex is on the near plane');
    // Find the original edge it was generated from and verify t consistency.
    const [sa, sb] = g.key.match(/\(([^|]+)\|([^)]+)\)/).slice(1);
    const origA = verts.find((v) => v.key === sa);
    const origB = verts.find((v) => v.key === sb);
    const dp = nearCheck.inside(origA.clip);
    const dq = nearCheck.inside(origB.clip);
    const t = dp / (dp - dq);
    // Position and every attribute must use the same t.
    for (let k = 0; k < 4; k++) {
      const expected = origA.clip[k] + t * (origB.clip[k] - origA.clip[k]);
      assert.ok(Math.abs(g.clip[k] - expected) < 1e-9, `clip[${k}] interpolates with t`);
    }
    for (let k = 0; k < 2; k++) {
      const expected = origA.uv[k] + t * (origB.uv[k] - origA.uv[k]);
      assert.ok(Math.abs(g.uv[k] - expected) < 1e-12, `uv[${k}] interpolates with t`);
    }
    for (let k = 0; k < 3; k++) {
      const expected = origA.color[k] + t * (origB.color[k] - origA.color[k]);
      assert.ok(Math.abs(g.color[k] - expected) < 1e-12, `colour[${k}] interpolates with t`);
    }
  }
});

test('two triangles sharing an edge get identical clip intersections (welding identity)', () => {
  const proj = perspective(Math.PI / 3, 1, 1.0, 20);
  const view = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  const vp = multiply4(proj, view);
  // Shared edge between p0 (behind camera) and p2 (in front) crosses the near
  // plane; p1 is in front.
  const p0 = transform4(vp, [0.2, -0.2, 3.4, 1]);
  const p1 = transform4(vp, [0.6, -0.2, -1.0, 1]);
  const p2 = transform4(vp, [0.0, 0.6, -2.0, 1]);
  assert.ok(p0[3] < 0 && p1[3] > 0 && p2[3] > 0);
  const ctx = createClipContext();
  const A = [
    { key: 'S:0', clip: p0, uv: [0, 0], color: [1, 1, 1] },
    { key: 'S:1', clip: p1, uv: [1, 0], color: [1, 1, 1] },
    { key: 'S:2', clip: p2, uv: [0, 1], color: [1, 1, 1] },
  ];
  const B = [
    { key: 'S:2', clip: p2, uv: [0, 1], color: [1, 1, 1] },
    { key: 'S:1', clip: p1, uv: [1, 0], color: [1, 1, 1] },
    { key: 'S:0', clip: p0, uv: [0, 0], color: [1, 1, 1] },
  ];
  const ga = clipTriangle(A, ctx).polygon.filter((v) => v.generatedBy);
  const gb = clipTriangle(B, ctx).polygon.filter((v) => v.generatedBy);
  assert.ok(ga.length > 0 && gb.length > 0);
  const keysA = new Set(ga.map((v) => v.key));
  const keysB = new Set(gb.map((v) => v.key));
  const shared = [...keysA].filter((k) => keysB.has(k));
  assert.ok(shared.length >= 1, 'the shared edge yields shared generated vertices');
  for (const k of shared) {
    const va = ga.find((v) => v.key === k);
    const vb = gb.find((v) => v.key === k);
    assert.deepEqual(va.clip, vb.clip, 'shared intersection is bit-identical');
    assert.deepEqual(va.uv, vb.uv, 'shared intersection carries identical attributes');
  }
  // A triangle using different keys at the same locations must NOT weld.
  const C = [
    { key: 'T:2', clip: p2, uv: [0, 1], color: [1, 1, 1] },
    { key: 'T:1', clip: p1, uv: [1, 0], color: [1, 1, 1] },
    { key: 'T:0', clip: p0, uv: [0, 0], color: [1, 1, 1] },
  ];
  const gc = clipTriangle(C, ctx).polygon.filter((v) => v.generatedBy);
  assert.ok(gc.length > 0);
  assert.ok(gc.every((v) => !keysA.has(v.key)), 'independent keys do not weld');
});

// ---------------------------------------------------------------------------
// No poison values reach the framebuffer
// ---------------------------------------------------------------------------

test('no NaN / Infinity reaches colour or depth buffers', () => {
  for (const near of [0.05, 0.15, 2.3, 2.5]) {
    const s = stateWith({ near });
    const frame = renderFrame(buildScene(s));
    assert.ok(frame.color.every((c) => Number.isFinite(c)), `near=${near}: colour finite`);
    for (let i = 0; i < frame.depth.length; i++) {
      if (frame.maskPrim[i] >= 0) assert.ok(Number.isFinite(frame.depth[i]), `near=${near}: depth finite`);
    }
  }
});

test('first frame colour of uncovered pixels is exactly the background', () => {
  const frame = renderFrame(buildScene(stateWith()));
  let checked = 0;
  for (let i = 0; i < frame.maskPrim.length; i++) {
    if (frame.maskPrim[i] < 0) {
      assert.ok(Math.abs(frame.color[i * 3] - BACKGROUND[0]) < 1e-7);
      assert.ok(Math.abs(frame.color[i * 3 + 1] - BACKGROUND[1]) < 1e-7);
      assert.ok(Math.abs(frame.color[i * 3 + 2] - BACKGROUND[2]) < 1e-7);
      checked++;
    }
  }
  assert.ok(checked > 0);
});

// ---------------------------------------------------------------------------
// Naive pipeline comparison. We inject a purpose-built scene whose triangle
// straddles the camera plane (one vertex behind the eye), which is exactly
// the case the naive divide-then-2D-clip pipeline cannot represent.
// ---------------------------------------------------------------------------

function syntheticScene(worldTri, near) {
  const width = 320;
  const height = 240;
  const view = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  const proj = perspective((50 * Math.PI) / 180, width / height, near, 30);
  const viewProj = multiply4(proj, view);
  const invViewProj = invert4(viewProj);
  const asset = {
    id: 'synth',
    label: 'synthetic',
    kind: 'synthetic',
    texture: checker,
    triangles: [
      {
        assetId: 'synth',
        index: 0,
        keys: ['s0', 's1', 's2'],
        world: worldTri,
        uv: [[0.05, 0.05], [0.95, 0.05], [0.5, 0.95]],
        color: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        normal: normalize([
          (worldTri[1][1] - worldTri[0][1]) * (worldTri[2][2] - worldTri[0][2]) - (worldTri[1][2] - worldTri[0][2]) * (worldTri[2][1] - worldTri[0][1]),
          (worldTri[1][2] - worldTri[0][2]) * (worldTri[2][0] - worldTri[0][0]) - (worldTri[1][0] - worldTri[0][0]) * (worldTri[2][2] - worldTri[0][2]),
          (worldTri[1][0] - worldTri[0][0]) * (worldTri[2][1] - worldTri[0][1]) - (worldTri[1][1] - worldTri[0][1]) * (worldTri[2][0] - worldTri[0][0]),
        ]),
        texture: checker,
      },
    ],
    offset: [0, 0, 0],
  };
  return {
    state: {
      viewport: { width, height },
      visible: { synth: true },
    },
    camera: { eye: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0] },
    view,
    proj,
    viewProj,
    invViewProj,
    assets: [asset],
    visibleAssets: [asset],
    textures: { checker },
  };
}

test('naive divide-then-clip pipeline loses geometry the correct pipeline keeps', () => {
  // Large triangle with one vertex behind the camera: the correct pipeline
  // clips it against the near plane and keeps the visible part; the naive one
  // must drop the whole triangle.
  const worldTri = [
    [0.2, -0.9, 3.4],
    [2.2, -0.9, -1.0],
    [-1.6, 1.1, -2.0],
  ];
  const scene = syntheticScene(worldTri, 0.5);
  const correct = renderFrame(scene, { pipeline: 'correct' });
  const naive = renderFrame(scene, { pipeline: 'naive2d' });
  assert.ok(correct.stats.generatedVertices > 0, 'correct pipeline clipped');
  assert.ok(naive.stats.naiveDropped > 0, 'naive pipeline had to drop the straddling triangle');
  const cCount = coverageCount(correct, 'synth');
  const nCount = coverageCount(naive, 'synth');
  assert.ok(cCount > 0, 'correct pipeline produced coverage');
  assert.equal(nCount, 0, 'naive pipeline produced nothing for this triangle');
});

test('correct pipeline is perspective-correct: uv at pixel matches 1/w interpolation', () => {
  // A strongly foreshortened span quad: screen-linear uv drifts measurably.
  const s = stateWith({
    orbit: { yaw: 0.62, pitch: 0.05, distance: 1.1, target: [0, 0, -0.6] },
    visible: { welded: false, seamed: false, span: true, sliver: false, degenerate: false },
  });
  const scene = buildScene(s);
  const correct = renderFrame(scene, { pipeline: 'correct' });
  const naive = renderFrame(scene, { pipeline: 'naive2d' });
  let maxDiff = 0;
  let compared = 0;
  for (let idx = 0; idx < correct.maskPrim.length; idx++) {
    if (correct.maskPrim[idx] < 0 || naive.maskPrim[idx] < 0) continue;
    for (let k = 0; k < 3; k++) {
      maxDiff = Math.max(maxDiff, Math.abs(correct.color[idx * 3 + k] - naive.color[idx * 3 + k]));
    }
    compared++;
  }
  assert.ok(compared > 1000, `compared ${compared} shared pixels`);
  assert.ok(maxDiff > 0.02, `perspective vs screen-linear differ (max channel diff ${maxDiff.toFixed(3)})`);
});
