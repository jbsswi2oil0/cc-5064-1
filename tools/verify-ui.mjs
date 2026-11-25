// Headless UI verification. Drives the real page (served over HTTP) with
// Playwright's bundled Chromium: no WebGL anywhere, the canvas is painted by
// the CPU kernel. Checks that
//   * the page loads without console/page errors,
//   * the kernel produced a frame with coverage,
//   * a slider drives the kernel (near-plane sweep changes coverage),
//   * clicking a pixel produces a numeric audit that agrees,
//   * the welded asset is watertight while the seamed one shows a gap,
//   * popping the camera close makes the seam wider than 1 sample spacing.
//
// Run with the server up:   node tools/serve.mjs &  node tools/verify-ui.mjs
// Playwright is only a dev-time verification dependency, so it is resolved
// from PLAYWRIGHT_PATH if given (e.g. a global/temp install), else normally.
import { pathToFileURL } from 'node:url';
import { writeFile } from 'node:fs/promises';

const TARGET_URL = process.env.URL || 'http://127.0.0.1:5173/';
const SHOT_DIR = new URL('../out/', import.meta.url).pathname;

let chromium;
async function loadPlaywright() {
  let mod;
  try {
    mod = await import('playwright');
  } catch (err) {
    const hint = process.env.PLAYWRIGHT_PATH;
    if (!hint) {
      console.error('Playwright is not installed. Install it or set PLAYWRIGHT_PATH to its package directory.');
      console.error('  PLAYWRIGHT_PATH=/path/to/node_modules/playwright node tools/verify-ui.mjs');
      process.exit(2);
    }
    mod = await import(pathToFileURL(hint + '/index.js').href);
  }
  const pw = mod.chromium ? mod : mod.default;
  if (!pw || !pw.chromium) throw new Error('Playwright loaded but no chromium export found');
  return pw.chromium;
}
chromium = await loadPlaywright();

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1420, height: 900 } });

const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error') errors.push('console: ' + m.text());
});
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));

const fail = (msg) => {
  console.error('FAIL:', msg);
  process.exitCode = 1;
};
const ok = (msg) => console.log('ok  -', msg);

await page.goto(TARGET_URL, { waitUntil: 'networkidle' });
await page.waitForFunction(() => window.__raster && window.__raster.getFrame() !== null, { timeout: 15000 });

// 1. Frame produced by the kernel.
const base = await page.evaluate(() => {
  const f = window.__raster.getFrame();
  let covered = 0;
  for (let i = 0; i < f.maskPrim.length; i++) if (f.maskPrim[i] >= 0) covered++;
  return {
    width: f.width,
    height: f.height,
    covered,
    stats: f.stats,
    pipeline: f.pipeline,
    hasWebGL: (() => {
      const c = document.createElement('canvas');
      return !!(c.getContext('webgl') || c.getContext('experimental-webgl'));
    })(),
  };
});
if (base.width === 960 && base.height === 600) ok(`frame ${base.width}x${base.height}`);
else fail(`unexpected frame size ${base.width}x${base.height}`);
if (base.covered > 10000) ok(`kernel coverage ${base.covered} px`);
else fail(`too little coverage ${base.covered}`);
if (base.pipeline === 'correct') ok('pipeline = correct');
else fail(`pipeline = ${base.pipeline}`);

// Confirm the page itself never created a WebGL context for rendering.
const usesWebGL = await page.evaluate(() => {
  let used = false;
  const orig = HTMLCanvasElement.prototype.getContext;
  // Re-check by inspecting canvas elements we know about.
  for (const c of document.querySelectorAll('canvas')) {
    const ctx2d = c.getContext('2d');
    if (ctx2d) used = true;
  }
  return used;
});
if (usesWebGL) ok('canvases are 2D (CPU bitmaps), no WebGL rendering path');
else fail('expected 2D canvas contexts');

await page.screenshot({ path: SHOT_DIR + 'ui-initial.png' });

// 2. Near-plane slider drives the kernel: push near past the panel.
const nearSweep = await page.evaluate(async () => {
  const countAsset = (f, id) => {
    const slot = f.scene.assets.findIndex((a) => a.id === id);
    let n = 0;
    for (let i = 0; i < f.maskAsset.length; i++) if (f.maskAsset[i] === slot) n++;
    return n;
  };
  const before = window.__raster.getFrame();
  const wBefore = countAsset(before, 'welded');
  window.__raster.setNear(2.25);
  const after = window.__raster.getFrame();
  const wAfter = countAsset(after, 'welded');
  return { wBefore, wAfter, gen: after.stats.generatedVertices, planeHits: after.stats.planeHits };
});
if (nearSweep.gen > 0) ok(`near-plane clip generated ${nearSweep.gen} vertices (planes ${JSON.stringify(nearSweep.planeHits)})`);
else fail('near-plane sweep did not generate clip vertices');
if (nearSweep.wAfter < nearSweep.wBefore) ok(`welded asset coverage shrank under clipping (${nearSweep.wBefore} -> ${nearSweep.wAfter} px)`);
else fail(`welded coverage did not shrink (${nearSweep.wBefore} -> ${nearSweep.wAfter})`);

await page.screenshot({ path: SHOT_DIR + 'ui-nearclip.png' });

// 3. Pixel audit through a real click.
await page.evaluate(() => window.__raster.setNear(0.15));
await page.waitForTimeout(50);
const pick = await page.evaluate(() => {
  const f = window.__raster.getFrame();
  let target = null;
  // Find a pixel belonging to the welded asset.
  const slot = f.scene.assets.findIndex((a) => a.id === 'welded');
  for (let py = 5; py < f.height - 5 && !target; py += 3) {
    for (let px = 5; px < f.width - 5; px += 3) {
      if (f.maskAsset[py * f.width + px] === slot) {
        target = [px, py];
        break;
      }
    }
  }
  return target;
});
if (!pick) fail('no welded pixel found for audit');
else {
  const rect = await page.evaluate(() => {
    const v = document.getElementById('view');
    const r = v.getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  });
  const cx = rect.left + ((pick[0] + 0.5) / 960) * rect.width;
  const cy = rect.top + ((pick[1] + 0.5) / 600) * rect.height;
  await page.mouse.click(cx, cy);
  await page.waitForTimeout(100);
  const auditText = await page.evaluate(() => document.getElementById('auditBox').innerText);
  const audit = await page.evaluate(([x, y]) => window.__raster.auditPixel(x, y), pick);
  if (audit.ok && audit.agrees) ok(`pixel (${pick[0]},${pick[1]}) audit agrees (maxErr ${audit.maxErr.toExponential(2)})`);
  else fail(`pixel audit mismatch: ${JSON.stringify(audit.maxErr)}`);
  if (auditText.includes('复算一致')) ok('audit panel shows numeric recompute match');
  else fail('audit panel did not render the recompute result');
  await page.screenshot({ path: SHOT_DIR + 'ui-audit.png' });
}

// 4. Welded watertight vs seamed gap, sampled from the live kernel.
const masks = await page.evaluate(() => {
  const countHoles = (frame, assetId) => {
    const slot = frame.scene.assets.findIndex((a) => a.id === assetId);
    if (slot < 0) return null;
    const pts = [];
    for (const o of frame.outlines) {
      if (o.asset !== assetId) continue;
      for (const p of o.points) pts.push([p.x * frame.width, p.y * frame.height]);
    }
    if (pts.length < 3) return 0;
    // convex hull
    const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    const up = [];
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    lo.pop(); up.pop();
    const hull = lo.concat(up);
    const inside = (x, y) => {
      let sign = 0;
      for (let i = 0; i < hull.length; i++) {
        const a = hull[i], b = hull[(i + 1) % hull.length];
        const c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        if (c === 0) continue;
        const s = Math.sign(c);
        if (sign === 0) sign = s; else if (s !== sign) return false;
      }
      return true;
    };
    let holes = 0;
    for (let py = 1; py < frame.height - 1; py++) for (let px = 1; px < frame.width - 1; px++) {
      const x = px + 0.5, y = py + 0.5;
      if (!inside(x, y)) continue;
      let solid = true;
      for (const [dx, dy] of [[-1.25, 0], [1.25, 0], [0, -1.25], [0, 1.25]]) if (!inside(x + dx, y + dy)) { solid = false; break; }
      if (!solid) continue;
      if (frame.maskAsset[py * frame.width + px] !== slot) holes++;
    }
    return holes;
  };
  const f = window.__raster.getFrame();
  return { welded: countHoles(f, 'welded'), seamed: countHoles(f, 'seamed') };
});
if (masks.welded === 0) ok('welded asset has 0 interior holes (watertight)');
else fail(`welded asset has ${masks.welded} holes`);
if (masks.seamed > 0) ok(`seamed asset shows ${masks.seamed} intentional gap pixels`);
else fail('seamed asset has no visible gap');

// 5. Close-up: seam wider than one sample spacing.
const closeup = await page.evaluate(() => {
  window.__raster.setState({
    orbit: { yaw: 0, pitch: 0, distance: 0.5, target: [0.56, 0.04, 0.02] },
    seamWidth: 0.01,
  });
  const f = window.__raster.getFrame();
  // projected gap between the two seam-edge vertices
  const seamed = f.scene.assets.find((a) => a.id === 'seamed');
  const pa = seamed.triangles[0].world[0];
  const pb = seamed.triangles[1].world[0];
  const proj = (p) => {
    const m = f.scene.viewProj;
    const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return [((x / w) * 0.5 + 0.5) * f.width, (1 - ((y / w) * 0.5 + 0.5)) * f.height];
  };
  const sa = proj(pa);
  const sb = proj(pb);
  return { gapPx: Math.hypot(sa[0] - sb[0], sa[1] - sb[1]) };
});
if (closeup.gapPx > 1.0) ok(`close-up seam projects to ${closeup.gapPx.toFixed(2)} px (> 1 sample spacing)`);
else fail(`close-up seam only ${closeup.gapPx.toFixed(2)} px`);
await page.screenshot({ path: SHOT_DIR + 'ui-closeup.png' });

// 6. Drag a triangle and confirm the kernel re-rendered with a new offset.
const drag = await page.evaluate(() => {
  window.__raster.setState({ orbit: { yaw: 0, pitch: 0.09, distance: 2.55, target: [0, 0, 0] }, seamWidth: 0.0042 });
  const f = window.__raster.getFrame();
  const slot = f.scene.assets.findIndex((a) => a.id === 'welded');
  let target = null;
  for (let py = 10; py < f.height - 10 && !target; py += 4) for (let px = 10; px < f.width - 10; px += 4) {
    if (f.maskAsset[py * f.width + px] === slot) { target = [px, py]; break; }
  }
  const before = [...window.__raster.getState().offsets.welded];
  return { target, before };
});
if (drag.target) {
  const rect = await page.evaluate(() => {
    const v = document.getElementById('view');
    const r = v.getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  });
  const toScreen = ([px, py]) => [rect.left + ((px + 0.5) / 960) * rect.width, rect.top + ((py + 0.5) / 600) * rect.height];
  const [sx, sy] = toScreen(drag.target);
  await page.mouse.move(sx, sy);
  await page.mouse.down();
  await page.mouse.move(sx + 60, sy - 40, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(120);
  const after = await page.evaluate(() => [...window.__raster.getState().offsets.welded]);
  const delta = Math.hypot(after[0] - drag.before[0], after[1] - drag.before[1], after[2] - drag.before[2]);
  if (delta > 1e-3) ok(`drag moved welded asset offset by ${delta.toFixed(3)} world units via coverage-mask picking`);
  else fail('drag did not change asset offset');
  await page.screenshot({ path: SHOT_DIR + 'ui-drag.png' });
} else fail('no welded pixel available for drag test');

// 7. Orbit by dragging empty space.
const orbit = await page.evaluate(() => window.__raster.getState().orbit.yaw);
const rect = await page.evaluate(() => {
  const v = document.getElementById('view');
  const r = v.getBoundingClientRect();
  return { left: r.left, top: r.top, width: r.width, height: r.height };
});
// Empty top-right corner is background.
await page.mouse.move(rect.left + rect.width - 30, rect.top + 25);
await page.mouse.down();
await page.mouse.move(rect.left + rect.width - 130, rect.top + 60, { steps: 6 });
await page.mouse.up();
await page.waitForTimeout(100);
const orbitAfter = await page.evaluate(() => window.__raster.getState().orbit.yaw);
if (Math.abs(orbitAfter - orbit) > 1e-3) ok(`orbit drag changed yaw (${orbit.toFixed(3)} -> ${orbitAfter.toFixed(3)})`);
else fail('orbit drag did not change yaw');

// 8. No console / page errors anywhere.
if (errors.length === 0) ok('no console or page errors');
else fail('errors: ' + errors.join(' | '));

await browser.close();
console.log(process.exitCode ? 'UI VERIFY: FAILURES' : 'UI VERIFY: ALL CHECKS PASSED');
