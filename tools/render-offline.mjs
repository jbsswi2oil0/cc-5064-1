// Offline renderer + numeric geometry/pixel report. This runs the SAME CPU
// kernel the browser viewport uses, writes PNG frames, and prints a step-by-step
// comparison of clip-space geometry, coverage masks and audited pixels.
//
//   node tools/render-offline.mjs [outDir]
//
import { writeFile, mkdir } from 'node:fs/promises';
import { deflateSync } from 'node:zlib';
import { join } from 'node:path';

import { buildScene, DEFAULT_STATE, WELDED, SEAMED, SPAN, projectPoint } from '../src/core/scene.js';
import { renderFrame, auditPixel, coverageCount } from '../src/core/raster.js';
import { perspective, lookAt, multiply4, invert4 } from '../src/core/math.js';
import { makeCheckerTexture } from '../src/core/texture.js';

const outDir = process.argv[2] || new URL('../out', import.meta.url).pathname;

// ---------------------------------------------------------------------------
// PNG encoder (truecolour + alpha, no dependencies)
// ---------------------------------------------------------------------------

function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}

function encodePng(width, height, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type RGBA
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0; // filter none
    rgba.copy(raw, y * (width * 4 + 1) + 1, y * width * 4, (y + 1) * width * 4);
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

function frameToPng(frame, gamma = 1 / 2.2, showMask = false) {
  const { width, height, color, maskAsset } = frame;
  const rgba = Buffer.alloc(width * height * 4);
  const palette = [
    [79, 195, 255],
    [255, 159, 67],
    [157, 123, 255],
    [255, 92, 138],
    [170, 170, 170],
  ];
  for (let i = 0; i < width * height; i++) {
    let r = color[i * 3];
    let g = color[i * 3 + 1];
    let b = color[i * 3 + 2];
    if (showMask) {
      const a = maskAsset[i];
      if (a < 0) {
        r = 0.03; g = 0.035; b = 0.05;
      } else {
        const p = palette[a % palette.length];
        r = r * 0.2 + (p[0] / 255) * 0.8;
        g = g * 0.2 + (p[1] / 255) * 0.8;
        b = b * 0.2 + (p[2] / 255) * 0.8;
      }
    }
    rgba[i * 4] = Math.round(255 * Math.pow(clamp01(r), gamma));
    rgba[i * 4 + 1] = Math.round(255 * Math.pow(clamp01(g), gamma));
    rgba[i * 4 + 2] = Math.round(255 * Math.pow(clamp01(b), gamma));
    rgba[i * 4 + 3] = 255;
  }
  return encodePng(width, height, rgba);
}

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

// ---------------------------------------------------------------------------
// Geometry / pixel report
// ---------------------------------------------------------------------------

function stateWith(patch = {}) {
  const s = JSON.parse(JSON.stringify(DEFAULT_STATE));
  s.visible = { welded: true, seamed: true, span: true, sliver: true, degenerate: false };
  Object.assign(s, patch);
  return s;
}

function fmt(n, d = 4) {
  return Number.isFinite(n) ? n.toFixed(d) : String(n);
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const lines = [];
  const log = (s = '') => {
    lines.push(s);
    console.log(s);
  };

  log('============================================================');
  log(' SOFTWARE TRIANGLE RASTERIZER - offline geometry/pixel report');
  log('============================================================');
  log('coordinate system: right-handed; camera looks down -Z');
  log('clip space:        -w <= x,y,z <= w ; ndc z in [-1,1] (near = -1)');
  log('screen:            origin top-left, y down, pixel centre (x+0.5,y+0.5)');
  log('coverage:          8-bit fixed-point edge functions, top-left rule');
  log('weld rule:         generated vertex key = plane + sorted endpoint keys');
  log('');

  // --- 1. default frame ----------------------------------------------------
  {
    const scene = buildScene(stateWith());
    const frame = renderFrame(scene);
    log('[1] DEFAULT FRAME');
    log(`    triangles=${frame.stats.triangles} fan=${frame.stats.fanTriangles} clipped=${frame.stats.clippedTriangles} generated=${frame.stats.generatedVertices}`);
    log(`    plane hits: ${JSON.stringify(frame.stats.planeHits)}`);
    log(`    degenerate skipped=${frame.stats.degenerateSkipped}`);
    log(`    fragments tested=${frame.stats.fragmentsTested} inside=${frame.stats.fragmentsPassed}`);
    log(`    welded covered=${coverageCount(frame, WELDED)} px   seamed covered=${coverageCount(frame, SEAMED)} px`);
    await writeFile(join(outDir, 'default.png'), frameToPng(frame));
    await writeFile(join(outDir, 'default-mask.png'), frameToPng(frame, 1 / 2.2, true));
    log('    wrote default.png, default-mask.png');
    log('');
  }

  // --- 2. near-plane sweep -------------------------------------------------
  {
    log('[2] NEAR-PLANE SWEEP THROUGH THE WELDED PANEL');
    log('    near    gen  planes            weldedPx  seamedPx   weldedHoles seamedHoles');
    for (const near of [0.15, 1.2, 2.0, 2.15, 2.25, 2.4, 2.55]) {
      const scene = buildScene(
        stateWith({ near, visible: { welded: true, seamed: true, span: false, sliver: false, degenerate: false } })
      );
      const frame = renderFrame(scene);
      const wH = holes(frame, WELDED);
      const sH = holes(frame, SEAMED);
      log(
        `    ${fmt(near, 2)}   ${String(frame.stats.generatedVertices).padStart(2)}   ${JSON.stringify(frame.stats.planeHits).padEnd(18)} ${String(
          coverageCount(frame, WELDED)
        ).padStart(8)}  ${String(coverageCount(frame, SEAMED)).padStart(8)}   ${String(wH).padStart(11)} ${String(sH).padStart(10)}`
      );
      if (near === 2.25) {
        await writeFile(join(outDir, 'clipped-near225.png'), frameToPng(frame));
        await writeFile(join(outDir, 'clipped-near225-mask.png'), frameToPng(frame, 1 / 2.2, true));
      }
    }
    log('    -> welded interior holes stay 0 while the near plane cuts it.');
    log('    -> seamed holes persist (intentional gap exposes background).');
    log('');
  }

  // --- 3. close-up seam vs sample spacing ---------------------------------
  {
    log('[3] CLOSE-UP: SEAM WIDER THAN ONE SAMPLE SPACING');
    for (const distance of [2.55, 1.2, 0.75, 0.5]) {
      for (const seamWidth of [0.0, 0.002, 0.0042, 0.01]) {
        const scene = buildScene(
          stateWith({
            orbit: { yaw: 0, pitch: 0.0, distance, target: [0.56, 0.04, 0.02] },
            seamWidth,
            visible: { welded: false, seamed: true, span: false, sliver: false, degenerate: false },
          })
        );
        const frame = renderFrame(scene);
        const seamed = scene.assets.find((a) => a.id === SEAMED);
        const gapPx = projectedSeamPx(scene, seamed);
        log(`    dist=${fmt(distance, 2)} seam=${fmt(seamWidth, 4)} -> projected gap ${fmt(gapPx, 2)} px, holes ${holes(frame, SEAMED)}`);
        if (distance === 0.5 && Math.abs(seamWidth - 0.01) < 1e-9) {
          await writeFile(join(outDir, 'closeup-seam.png'), frameToPng(frame));
        }
      }
    }
    log('');
  }

  // --- 4. step-by-step pixel audit ----------------------------------------
  {
    log('[4] STEP-BY-STEP PIXEL AUDIT (default frame)');
    const scene = buildScene(stateWith());
    const frame = renderFrame(scene);
    const picks = [];
    for (const assetId of [WELDED, SEAMED, SPAN]) {
      const slot = scene.assets.findIndex((a) => a.id === assetId);
      outer: for (let py = 8; py < frame.height - 8; py += 6) {
        for (let px = 8; px < frame.width - 8; px += 6) {
          if (frame.maskAsset[py * frame.width + px] === slot) {
            picks.push({ assetId, x: px, y: py });
            break outer;
          }
        }
      }
    }
    for (const pick of picks) {
      const a = auditPixel(frame, pick.x, pick.y);
      log(`    asset=${pick.assetId} pixel(${pick.x},${pick.y}) sample(${pick.x + 0.5},${pick.y + 0.5})`);
      log(`      triangle=${a.primitive.triangleIndex} planes=${JSON.stringify(a.primitive.planesHit)}`);
      log(`      edge w0,w1,w2 = ${a.edges.w0}, ${a.edges.w1}, ${a.edges.w2}   area=${a.primitive.areaFixed}`);
      log(`      screen bary = ${fmt(a.barycentric.l0, 5)}, ${fmt(a.barycentric.l1, 5)}, ${fmt(a.barycentric.l2, 5)}`);
      log(`      invW = ${fmt(a.invW, 6)}   persp bary = ${a.perspBarycentric.map((v) => fmt(v, 5)).join(', ')}`);
      log(`      uv = ${a.uv.map((v) => fmt(v, 5)).join(', ')}   texel = ${a.texel.map((v) => fmt(v, 4)).join(', ')}`);
      log(`      final = ${a.final.map((v) => fmt(v, 5)).join(', ')}  stored = ${a.stored.map((v) => fmt(v, 5)).join(', ')}`);
      log(`      maxErr = ${a.maxErr.toExponential(2)}  agrees=${a.agrees}`);
    }
    log('');
  }

  // --- 5. naive pipeline comparison ---------------------------------------
  {
    log('[5] NAIVE "divide-then-2D-clip" PIPELINE vs CORRECT');
    log('    synthetic triangle with one vertex behind the camera plane (w < 0):');
    const worldTri = [
      [0.2, -0.9, 3.4],
      [2.2, -0.9, -1.0],
      [-1.6, 1.1, -2.0],
    ];
    const scene = syntheticScene(worldTri, 0.5);
    const correct = renderFrame(scene, { pipeline: 'correct' });
    const naive = renderFrame(scene, { pipeline: 'naive2d' });
    log(`    correct: generated=${correct.stats.generatedVertices} covered=${coverageCount(correct, 'synth')} px`);
    log(`    naive:   dropped=${naive.stats.naiveDropped} covered=${coverageCount(naive, 'synth')} px`);
    log('    -> the naive pipeline discards the whole triangle; the correct one clips and keeps the visible part.');
    await writeFile(join(outDir, 'naive-straddle.png'), frameToPng(naive));
    await writeFile(join(outDir, 'correct-straddle.png'), frameToPng(correct));
    log('');
    log('    real scene: near plane cuts through the welded panel');
    for (const near of [2.25, 2.4]) {
      const s2 = buildScene(
        stateWith({ near, visible: { welded: true, seamed: false, span: false, sliver: false, degenerate: false } })
      );
      const c2 = renderFrame(s2, { pipeline: 'correct' });
      const n2 = renderFrame(s2, { pipeline: 'naive2d' });
      log(
        `    near=${fmt(near, 2)} correct: gen=${c2.stats.generatedVertices} weldedPx=${coverageCount(c2, WELDED)} | naive: dropped=${n2.stats.naiveDropped} weldedPx=${coverageCount(
          n2,
          WELDED
        )} (naive clips in 2D only, so it keeps the stretched triangle)`
      );
      await writeFile(join(outDir, `naive-near${String(near).replace('.', '')}.png`), frameToPng(n2));
    }
    log('');
    log('============================================================');
    log(' Images written to: ' + outDir);
    log('============================================================');
    await writeFile(join(outDir, 'report.txt'), lines.join('\n') + '\n');
  }
}

const checker = makeCheckerTexture(64, 8);

function syntheticScene(worldTri, near) {
  const width = 320;
  const height = 240;
  const view = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
  const proj = perspective((50 * Math.PI) / 180, width / height, near, 30);
  const viewProj = multiply4(proj, view);
  const invViewProj = invert4(viewProj);
  const wx = worldTri[1][0] - worldTri[0][0];
  const wy = worldTri[1][1] - worldTri[0][1];
  const wz = worldTri[1][2] - worldTri[0][2];
  const ux = worldTri[2][0] - worldTri[0][0];
  const uy = worldTri[2][1] - worldTri[0][1];
  const uz = worldTri[2][2] - worldTri[0][2];
  const n = [wy * uz - wz * uy, wz * ux - wx * uz, wx * uy - wy * ux];
  const len = Math.hypot(...n) || 1;
  const asset = {
    id: 'synth',
    label: 'synthetic straddle',
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
        normal: [n[0] / len, n[1] / len, n[2] / len],
        texture: checker,
      },
    ],
    offset: [0, 0, 0],
  };
  return {
    state: { viewport: { width, height }, visible: { synth: true } },
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

function projectedSeamPx(scene, seamed) {
  const a = seamed.triangles[0].world[0];
  const b = seamed.triangles[1].world[0];
  const W = scene.state.viewport.width;
  const H = scene.state.viewport.height;
  const pa = projectPoint(scene.viewProj, a);
  const pb = projectPoint(scene.viewProj, b);
  if (pa[3] <= 0 || pb[3] <= 0) return NaN;
  const sa = [(pa[0] / pa[3] * 0.5 + 0.5) * W, (1 - (pa[1] / pa[3] * 0.5 + 0.5)) * H];
  const sb = [(pb[0] / pb[3] * 0.5 + 0.5) * W, (1 - (pb[1] / pb[3] * 0.5 + 0.5)) * H];
  return Math.hypot(sa[0] - sb[0], sa[1] - sb[1]);
}

function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross2 = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [];
  for (const q of p) {
    while (lo.length >= 2 && cross2(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
    lo.push(q);
  }
  const up = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (up.length >= 2 && cross2(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
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

function holes(frame, assetId, erode = 1.25) {
  const slot = frame.scene.assets.findIndex((a) => a.id === assetId);
  if (slot < 0) return 0;
  const pts = [];
  for (const o of frame.outlines) {
    if (o.asset !== assetId) continue;
    for (const p of o.points) pts.push([p.x * frame.width, p.y * frame.height]);
  }
  if (pts.length < 3) return 0;
  const hull = convexHull(pts);
  let count = 0;
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
      if (frame.maskAsset[py * frame.width + px] !== slot) count++;
    }
  }
  return count;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
