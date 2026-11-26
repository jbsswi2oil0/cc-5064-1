// Interactive viewport. Every visible pixel is produced by the CPU rasterizer
// in ../src/core; the canvases are plain 2D bitmaps of that output. There is
// no WebGL, no GPU, no pre-made engine.
//
// Interaction feedback loops THROUGH the rasterizer:
//   * dragging a triangle picks a pixel, then asks the frame's coverage mask
//     which asset/primitive owns it (so only pixels the rasterizer actually
//     covered are draggable),
//   * the near-plane sweep and seam-width sliders re-run the kernel each tick,
//   * clicking a pixel runs auditPixel() which recomputes barycentrics, 1/w,
//     uv, texel and final colour from the stored primitive and compares to the
//     stored framebuffer value.

import { buildScene, DEFAULT_STATE, cloneState, WELDED, SEAMED, SPAN, SLIVER, DEGENERATE, unprojectRay } from '../src/core/scene.js';
import { renderFrame, auditPixel } from '../src/core/raster.js';
import { add, sub, scale, normalize, dot } from '../src/core/math.js';

const state = cloneState(DEFAULT_STATE);
state.viewport = { width: 960, height: 600 };
state.visible.degenerate = false;

const view = document.getElementById('view');
const overlay = document.getElementById('overlay');
const vctx = view.getContext('2d');
const octx = overlay.getContext('2d');
let imageData = vctx.createImageData(view.width, view.height);
let frame = null;
let selected = null; // { x, y }
let dragging = null; // { assetId, grabWorld, plane }
let orbiting = null;
let lastCompare = null; // real benchmarkPair() result for the current state
let diffLayer = null; // offscreen canvas holding the coverage XOR overlay

const ui = {
  pipeline: document.getElementById('pipeline'),
  kernel: document.getElementById('kernel'),
  showOutlines: document.getElementById('showOutlines'),
  showMask: document.getElementById('showMask'),
  showSeamOnly: document.getElementById('showSeamOnly'),
  showWeldedOnly: document.getElementById('showWeldedOnly'),
  showSliver: document.getElementById('showSliver'),
  showDegenerate: document.getElementById('showDegenerate'),
  near: document.getElementById('near'),
  nearOut: document.getElementById('nearOut'),
  fov: document.getElementById('fov'),
  fovOut: document.getElementById('fovOut'),
  seam: document.getElementById('seam'),
  seamOut: document.getElementById('seamOut'),
  resetBtn: document.getElementById('resetBtn'),
  closeupBtn: document.getElementById('closeupBtn'),
  sweepBtn: document.getElementById('sweepBtn'),
  sweepStopBtn: document.getElementById('sweepStopBtn'),
  nearInfo: document.getElementById('nearInfo'),
  stats: document.getElementById('stats'),
  seamStatus: document.getElementById('seamStatus'),
  pixelPick: document.getElementById('pixelPick'),
  auditBox: document.getElementById('auditBox'),
  benchCompareBtn: document.getElementById('benchCompareBtn'),
  benchOut: document.getElementById('benchOut'),
  diffOverlay: document.getElementById('diffOverlay'),
};

const ASSET_COLORS = {
  [WELDED]: '#4fc3ff',
  [SEAMED]: '#ff9f43',
  [SPAN]: '#9d7bff',
  [SLIVER]: '#ff5c8a',
  [DEGENERATE]: '#aaaaaa',
};

function syncUiFromState() {
  ui.near.value = state.near;
  ui.nearOut.textContent = state.near.toFixed(2);
  ui.fov.value = state.fovDeg;
  ui.fovOut.textContent = state.fovDeg + '°';
  ui.seam.value = state.seamWidth;
  ui.seamOut.textContent = state.seamWidth.toFixed(4);
  ui.showSliver.checked = state.visible.sliver;
  ui.showDegenerate.checked = state.visible.degenerate;
}

function applyVisibilityToggles() {
  const onlySeam = ui.showSeamOnly.checked;
  const onlyWeld = ui.showWeldedOnly.checked;
  state.visible.welded = !onlySeam && !onlyWeld ? true : onlyWeld;
  state.visible.seamed = !onlySeam && !onlyWeld ? true : onlySeam;
  // When a "only" filter is active, hide the big span so the pair fills view.
  state.visible.span = !(onlySeam || onlyWeld);
  state.visible.sliver = ui.showSliver.checked && !(onlySeam || onlyWeld);
  state.visible.degenerate = ui.showDegenerate.checked && !(onlySeam || onlyWeld);
}

function pipelineMode() {
  return ui.pipeline.value;
}

function kernelMode() {
  return ui.kernel.value;
}

function render() {
  applyVisibilityToggles();
  // Guard slider states the kernel cannot honour, and show the kernel's own
  // refusal in the viewport rather than silently producing garbage.
  if (!(state.near > 0) || !(state.far > state.near)) {
    showError(new Error(`near 必须 > 0 且 far(${state.far}) > near(${state.near})`));
    return;
  }
  try {
    frame = renderFrame(buildScene(state), { pipeline: pipelineMode(), kernel: kernelMode() });
  } catch (err) {
    showError(err);
    return;
  }
  paintFrame();
  paintOverlay();
  updateStats();
  updateNearInfo();
  if (selected) runAudit(selected.x, selected.y);
}

function showError(err) {
  frame = null;
  vctx.fillStyle = '#2a0f12';
  vctx.fillRect(0, 0, view.width, view.height);
  vctx.fillStyle = '#ff9a9a';
  vctx.font = '14px monospace';
  vctx.fillText('内核拒绝该状态: ' + err.message, 20, 30);
  ui.stats.textContent = 'render error: ' + err.message;
}

function paintFrame() {
  const { width, height, color, maskAsset } = frame;
  if (imageData.width !== width || imageData.height !== height) {
    view.width = width;
    view.height = height;
    overlay.width = width;
    overlay.height = height;
    imageData = vctx.createImageData(width, height);
  }
  const data = imageData.data;
  const showMask = ui.showMask.checked;
  for (let i = 0; i < width * height; i++) {
    let r = color[i * 3];
    let g = color[i * 3 + 1];
    let b = color[i * 3 + 2];
    if (showMask) {
      const a = maskAsset[i];
      if (a < 0) {
        r = 0.03; g = 0.035; b = 0.05;
      } else {
        const hex = ASSET_COLORS[frame.scene.assets[a].id] || '#888';
        const cr = parseInt(hex.slice(1, 3), 16) / 255;
        const cg = parseInt(hex.slice(3, 5), 16) / 255;
        const cb = parseInt(hex.slice(5, 7), 16) / 255;
        r = r * 0.25 + cr * 0.75;
        g = g * 0.25 + cg * 0.75;
        b = b * 0.25 + cb * 0.75;
      }
    }
    // Convert linear-ish 0..1 to sRGB bytes with gamma 1/2.2 for display.
    data[i * 4] = Math.round(255 * Math.pow(clamp01(r), 1 / 2.2));
    data[i * 4 + 1] = Math.round(255 * Math.pow(clamp01(g), 1 / 2.2));
    data[i * 4 + 2] = Math.round(255 * Math.pow(clamp01(b), 1 / 2.2));
    data[i * 4 + 3] = 255;
  }
  vctx.putImageData(imageData, 0, 0);
}

function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function paintOverlay() {
  const { width, height } = frame;
  octx.clearRect(0, 0, width, height);
  if (!ui.showOutlines.checked) return;

  // Coverage boundary emphasis: draw each triangle outline.
  for (const o of frame.outlines) {
    const color = ASSET_COLORS[o.asset] || '#fff';
    octx.strokeStyle = color;
    octx.lineWidth = o.generated ? 2.5 : 1.25;
    octx.globalAlpha = o.generated ? 0.95 : 0.6;
    octx.beginPath();
    o.points.forEach((p, i) => {
      const x = p.x * width;
      const y = p.y * height;
      if (i === 0) octx.moveTo(x, y);
      else octx.lineTo(x, y);
    });
    octx.closePath();
    octx.stroke();

    // Mark clip-generated vertices.
    for (const p of o.points) {
      if (!p.generatedBy) continue;
      octx.fillStyle = '#ffffff';
      octx.beginPath();
      octx.arc(p.x * width, p.y * height, 3, 0, Math.PI * 2);
      octx.fill();
      octx.strokeStyle = color;
      octx.lineWidth = 1.5;
      octx.stroke();
    }
  }
  octx.globalAlpha = 1;

  // Near-plane intersection band (NDC z = -1) is implicit; show a hint label.
  octx.fillStyle = 'rgba(230,240,255,0.85)';
  octx.font = '11px ui-monospace, monospace';
  octx.fillText(`pipeline=${frame.pipeline}  gen=${frame.stats.generatedVertices}  clipped=${frame.stats.clippedTriangles}`, 8, height - 8);

  if (selected && frame.maskPrim[selected.y * width + selected.x] >= 0) {
    octx.strokeStyle = '#fff';
    octx.lineWidth = 1;
    octx.strokeRect(selected.x + 0.5, selected.y + 0.5, 1, 1);
    octx.strokeStyle = 'rgba(255,255,255,0.45)';
    octx.strokeRect(selected.x - 3.5, selected.y - 3.5, 8, 8);
  }

  if (ui.diffOverlay.checked && diffLayer) {
    octx.drawImage(diffLayer, 0, 0);
  }
}

function updateStats() {
  const s = frame.stats;
  const fmtN = (n) => Number(n).toLocaleString('en-US');
  const lines = [
    `pipeline      ${s.pipeline}`,
    `kernel        ${s.kernel}`,
    `assets        ${s.assets}`,
    `triangles     ${s.triangles}  (fan ${s.fanTriangles})`,
    `clipped tris  ${s.clippedTriangles}`,
    `gen vertices  ${s.generatedVertices}`,
    `plane hits    ${JSON.stringify(s.planeHits)}`,
    `degenerate    ${s.degenerateSkipped} (zero area)`,
    `subpixel-dgn  ${s.subpixelSkipped}`,
    `non-finite    ${s.nonFiniteSkipped}`,
    `naive dropped ${s.naiveDropped}`,
    `bbox slots    ${fmtN(s.bboxPixels)} (traversal upper bound)`,
    `edge tests    ${fmtN(s.fragmentsTested)}`,
    `inside/shaded ${fmtN(s.fragmentsPassed)} / ${fmtN(s.fragmentsShaded)}`,
  ];
  if (s.kernel === 'optimized') {
    lines.push(
      `tiles E/F/P   ${fmtN(s.tileEmpty)} / ${fmtN(s.tileFull)} / ${fmtN(s.tilePartial)}`,
      `proof pixels  ${fmtN(s.proofAcceptedPixels)}`,
      `empty skipped ${fmtN(s.emptyTilePixelsSkipped)}`,
      `span early    ${fmtN(s.spanEarlyExits)}`,
      `small bypass  ${fmtN(s.scalarSmallBypass)}   safe fallback ${fmtN(s.scalarFallbacks)}`
    );
  }
  lines.push(`raster ms     ${s.msRaster.toFixed(2)}   total ${s.msTotal.toFixed(2)}`);
  ui.stats.textContent = lines.join('\n');

  const seamSlot = frame.scene.assets.findIndex((a) => a.id === SEAMED);
  const weldSlot = frame.scene.assets.findIndex((a) => a.id === WELDED);
  const status = [];
  if (seamSlot >= 0 && state.visible.seamed) {
    const holes = countSeamHoles();
    status.push(`留缝资产内部缺口像素: <b style="color:#ff9f43">${holes}</b>`);
  }
  if (weldSlot >= 0 && state.visible.welded) {
    const holes = countSeamHoles(WELDED);
    const okStyle = holes === 0 ? 'color:#62d98a' : 'color:#ff6b6b';
    status.push(`焊接资产内部孔洞像素: <b style="${okStyle}">${holes}</b>`);
  }
  if (s.nonFiniteSkipped > 0) {
    status.push(`<span style="color:#ff9a9a">内核跳过 ${s.nonFiniteSkipped} 个退化/非有限几何</span>`);
  }
  ui.seamStatus.innerHTML = status.join('<br/>');
}

function updateNearInfo() {
  const seamPx = seamWidthPx();
  const ever = seamPx >= 1 ? '<b style="color:#ff9f43">已超过 1 采样间距</b>' : '仍窄于 1 采样间距';
  ui.nearInfo.innerHTML =
    `近裁面 ${state.near.toFixed(2)} · 相机距目标 ${state.orbit.distance.toFixed(2)}<br/>` +
    `投影缝宽 ≈ ${seamPx.toFixed(2)} px（${ever}）<br/>` +
    `拖动“扫描近裁面”观察焊接/留缝差异`;
}

// Projected seam width in pixels, measured between the two seam-edge vertices
// that the seamed/low-precision triangle and its neighbour do NOT share.
function seamWidthPx() {
  if (!frame) return 0;
  const seamed = frame.scene.assets.find((a) => a.id === SEAMED);
  if (!seamed) return 0;
  const [a, , c] = seamed.triangles[0].world.flat ? seamed.triangles[0].world : [];
  const A = seamed.triangles[0].world[0];
  const B = seamed.triangles[1].world[0];
  const { viewProj } = frame.scene;
  const project = (p) => {
    const x = viewProj[0] * p[0] + viewProj[4] * p[1] + viewProj[8] * p[2] + viewProj[12];
    const y = viewProj[1] * p[0] + viewProj[5] * p[1] + viewProj[9] * p[2] + viewProj[13];
    const w = viewProj[3] * p[0] + viewProj[7] * p[1] + viewProj[11] * p[2] + viewProj[15];
    if (!(w > 0)) return null;
    return [((x / w) * 0.5 + 0.5) * frame.width, (1 - ((y / w) * 0.5 + 0.5)) * frame.height];
  };
  const sa = project(A);
  const sb = project(B);
  if (!sa || !sb) return 0;
  return Math.hypot(sa[0] - sb[0], sa[1] - sb[1]);
}

// Detect interior cracks for an asset using the same convex-hull technique as
// the tests (outline domain, erode by ~1.25 px).
function countSeamHoles(assetId = SEAMED) {
  if (!frame) return 0;
  const slot = frame.scene.assets.findIndex((a) => a.id === assetId);
  if (slot < 0) return 0;
  const pts = [];
  for (const o of frame.outlines) {
    if (o.asset !== assetId) continue;
    for (const p of o.points) pts.push([p.x * frame.width, p.y * frame.height]);
  }
  if (pts.length < 3) return 0;
  const hull = convexHull(pts);
  let holes = 0;
  for (let py = 1; py < frame.height - 1; py++) {
    for (let px = 1; px < frame.width - 1; px++) {
      const x = px + 0.5;
      const y = py + 0.5;
      if (!pointsInsidePolygon(hull, x, y)) continue;
      let solid = true;
      for (const [dx, dy] of [[-1.25, 0], [1.25, 0], [0, -1.25], [0, 1.25]]) {
        if (!pointsInsidePolygon(hull, x + dx, y + dy)) {
          solid = false;
          break;
        }
      }
      if (!solid) continue;
      if (frame.maskAsset[py * frame.width + px] !== slot) holes++;
    }
  }
  return holes;
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

// ---------------------------------------------------------------------------
// Pixel audit panel
// ---------------------------------------------------------------------------

function runAudit(px, py) {
  if (!frame) return;
  const a = auditPixel(frame, px, py);
  ui.pixelPick.textContent = `像素 (${px}, ${py}) · 采样点 (${px + 0.5}, ${py + 0.5})`;
  if (!a.ok) {
    ui.auditBox.innerHTML = `<div class="k">${escapeHtml(a.reason)}</div><div class="k">背景色 = (${a.color.map((c) => c.toFixed(4)).join(', ')})</div>`;
    return;
  }
  const p = a.primitive;
  const fmt = (x, n = 6) => (Number.isFinite(x) ? x.toFixed(n) : String(x));
  const rows = [
    ['资产 / 三角形', `${p.asset} / tri#${p.triangleIndex}`],
    ['裁剪平面', p.planesHit.length ? p.planesHit.join(', ') : '无（未被裁剪）'],
    ['屏幕顶点 A', fmt(p.vertices[0].sx, 3) + ', ' + fmt(p.vertices[0].sy, 3)],
    ['屏幕顶点 B', fmt(p.vertices[1].sx, 3) + ', ' + fmt(p.vertices[1].sy, 3)],
    ['屏幕顶点 C', fmt(p.vertices[2].sx, 3) + ', ' + fmt(p.vertices[2].sy, 3)],
    ['定点边函数 w0,w1,w2', `${a.edges.w0}, ${a.edges.w1}, ${a.edges.w2}`],
    ['定点面积 (×256²)', `${p.areaFixed}`],
    ['屏幕重心 l0,l1,l2', `${fmt(a.barycentric.l0, 5)}, ${fmt(a.barycentric.l1, 5)}, ${fmt(a.barycentric.l2, 5)}`],
    ['1/w (透视分母)', fmt(a.invW)],
    ['透视重心 (×1/w 归一)', a.perspBarycentric.map((v) => fmt(v, 5)).join(', ')],
    ['NDC 深度 z', fmt(a.depthNdc)],
    ['插值 uv', `${fmt(a.uv[0], 5)}, ${fmt(a.uv[1], 5)}`],
    ['插值颜色 attr', a.colorAttr.map((v) => fmt(v, 4)).join(', ')],
    ['纹理采样 (双线性)', a.texel.map((v) => fmt(v, 4)).join(', ')],
    ['光照系数', fmt(a.light, 4)],
    ['复算最终色', a.final.map((v) => fmt(v, 5)).join(', ')],
    ['帧缓冲存储色', a.stored.map((v) => fmt(v, 5)).join(', ')],
    ['最大通道误差', `<span class="${a.agrees ? 'ok' : 'bad'}">${a.maxErr.toExponential(3)} ${a.agrees ? '✓ 复算一致' : '✗ 不一致'}</span>`],
  ];
  ui.auditBox.innerHTML = `<table>${rows.map(([k, v]) => `<tr><td class="l">${k}</td><td class="r">${v}</td></tr>`).join('')}</table>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------------------
// Interaction: orbiting, asset dragging via coverage mask, pixel picking
// ---------------------------------------------------------------------------

function canvasPixel(ev) {
  const rect = view.getBoundingClientRect();
  const x = Math.floor(((ev.clientX - rect.left) / rect.width) * frame.width);
  const y = Math.floor(((ev.clientY - rect.top) / rect.height) * frame.height);
  return { x: Math.max(0, Math.min(frame.width - 1, x)), y: Math.max(0, Math.min(frame.height - 1, y)) };
}

function screenRay(px, py) {
  const ndcX = ((px + 0.5) / frame.width) * 2 - 1;
  const ndcY = 1 - ((py + 0.5) / frame.height) * 2;
  const { near, far } = unprojectRay(frame.scene.invViewProj, ndcX, ndcY);
  return { origin: frame.scene.camera.eye, dir: normalize(sub(far, near)) };
}

function rayPlane(ray, planePoint, planeNormal) {
  const denom = dot(planeNormal, ray.dir);
  if (Math.abs(denom) < 1e-8) return null;
  const t = dot(planeNormal, sub(planePoint, ray.origin)) / denom;
  if (t <= 0) return null;
  return add(ray.origin, scale(ray.dir, t));
}

const DRAGGABLE = new Set([WELDED, SEAMED, SPAN, SLIVER]);

view.addEventListener('pointerdown', (ev) => {
  if (!frame) return;
  view.setPointerCapture(ev.pointerId);
  const { x, y } = canvasPixel(ev);

  // Pick through the rasterizer's own coverage mask: only covered pixels are
  // draggable, and the hit asset comes from maskAsset rather than a raycast.
  const idx = y * frame.width + x;
  const slot = frame.maskAsset[idx];
  if (slot >= 0) {
    const asset = frame.scene.assets[slot];
    if (DRAGGABLE.has(asset.id)) {
      const ray = screenRay(x, y);
      // Drag in a plane parallel to the camera image plane through the hit
      // point, so the triangle follows the cursor closely.
      const camNormal = normalize(sub(frame.scene.camera.eye, frame.scene.camera.target));
      const hit = rayPlane(ray, asset.referenceQuad[0], camNormal);
      const anchor = hit || asset.referenceQuad[0];
      // Keep the grab vector constant during the drag.
      const grab = sub(anchor, add(asset.referenceQuad[0], state.offsets[asset.id]));
      dragging = {
        assetId: asset.id,
        anchor: [...anchor],
        planeNormal: camNormal,
        grab,
        origin: [...asset.referenceQuad[0]],
      };
      selected = { x, y };
      runAudit(x, y);
      paintOverlay();
      return;
    }
  }
  // Empty / non-draggable: orbit.
  orbiting = { x: ev.clientX, y: ev.clientY, yaw: state.orbit.yaw, pitch: state.orbit.pitch };
});

view.addEventListener('pointermove', (ev) => {
  if (!frame) return;
  if (dragging) {
    const { x, y } = canvasPixel(ev);
    const ray = screenRay(x, y);
    const hit = rayPlane(ray, dragging.anchor, dragging.planeNormal);
    if (!hit) return;
    // New anchor minus the constant grab vector gives the asset offset.
    const targetOrigin = sub(hit, dragging.grab);
    state.offsets[dragging.assetId] = sub(targetOrigin, dragging.origin);
    scheduleRender();
    return;
  }
  if (orbiting) {
    const dx = ev.clientX - orbiting.x;
    const dy = ev.clientY - orbiting.y;
    state.orbit.yaw = orbiting.yaw + dx * 0.006;
    state.orbit.pitch = Math.max(-1.35, Math.min(1.35, orbiting.pitch + dy * 0.006));
    scheduleRender();
  }
});

function endDrag() {
  dragging = null;
  orbiting = null;
}

view.addEventListener('pointerup', (ev) => {
  if (!frame) return;
  const { x, y } = canvasPixel(ev);
  const wasDragging = !!dragging;
  endDrag();
  if (!wasDragging) {
    selected = { x, y };
    runAudit(x, y);
    paintOverlay();
  }
});

view.addEventListener('pointercancel', endDrag);

view.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const factor = Math.exp(ev.deltaY * 0.0012);
  state.orbit.distance = Math.max(0.3, Math.min(8, state.orbit.distance * factor));
  scheduleRender();
}, { passive: false });

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

let pending = false;
function scheduleRender() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    render();
  });
}

ui.near.addEventListener('input', () => {
  state.near = parseFloat(ui.near.value);
  ui.nearOut.textContent = state.near.toFixed(2);
  scheduleRender();
});
ui.fov.addEventListener('input', () => {
  state.fovDeg = parseFloat(ui.fov.value);
  ui.fovOut.textContent = state.fovDeg + '°';
  scheduleRender();
});
ui.seam.addEventListener('input', () => {
  state.seamWidth = parseFloat(ui.seam.value);
  ui.seamOut.textContent = state.seamWidth.toFixed(4);
  scheduleRender();
});
ui.pipeline.addEventListener('change', scheduleRender);
ui.kernel.addEventListener('change', scheduleRender);
ui.diffOverlay.addEventListener('change', () => paintOverlay());
for (const el of [ui.showOutlines, ui.showMask, ui.showSeamOnly, ui.showWeldedOnly, ui.showSliver, ui.showDegenerate]) {
  el.addEventListener('change', () => {
    if (el === ui.showSeamOnly && el.checked) ui.showWeldedOnly.checked = false;
    if (el === ui.showWeldedOnly && el.checked) ui.showSeamOnly.checked = false;
    scheduleRender();
  });
}

ui.resetBtn.addEventListener('click', () => {
  const fresh = cloneState(DEFAULT_STATE);
  Object.assign(state, fresh);
  state.viewport = { width: 960, height: 600 };
  state.visible.degenerate = false;
  ui.showSliver.checked = true;
  ui.showDegenerate.checked = false;
  ui.showSeamOnly.checked = false;
  ui.showWeldedOnly.checked = false;
  syncUiFromState();
  selected = null;
  ui.auditBox.innerHTML = '';
  ui.pixelPick.textContent = '点击画布任意像素';
  scheduleRender();
});

ui.closeupBtn.addEventListener('click', () => {
  state.orbit = { yaw: 0.0, pitch: 0.0, distance: 0.62, target: [0.56, 0.04, 0.02] };
  state.seamWidth = Math.max(state.seamWidth, 0.01);
  state.near = 0.05;
  syncUiFromState();
  scheduleRender();
});

let sweep = null;
ui.sweepBtn.addEventListener('click', () => {
  if (sweep) return;
  const startNear = 0.15;
  const endNear = Math.max(0.15, state.orbit.distance + 0.35);
  let t0 = performance.now();
  const duration = 6000;
  sweep = requestAnimationFrame(function tick(now) {
    const k = Math.min(1, (now - t0) / duration);
    // Ping-pong so the near plane sweeps in and out.
    const phase = k < 0.5 ? k * 2 : (1 - k) * 2;
    state.near = startNear + (endNear - startNear) * phase;
    ui.near.value = state.near;
    ui.nearOut.textContent = state.near.toFixed(2);
    render();
    if (k < 1) {
      sweep = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(sweep);
      sweep = null;
    }
  });
});
ui.sweepStopBtn.addEventListener('click', () => {
  if (sweep) cancelAnimationFrame(sweep);
  sweep = null;
});

// ---------------------------------------------------------------------------
// Real scalar-vs-optimized comparison for the CURRENT state (driven by the
// button and by automated UI verification). No canned coverage/outlines: both
// kernels actually render buildScene(state) inside benchmarkPair().
// ---------------------------------------------------------------------------

function compareSummary(result) {
  const d = result.diff;
  return {
    speedup: result.speedup,
    identical: d.identical,
    scalarMedianMs: result.scalar.medianMs,
    optimizedMedianMs: result.optimized.medianMs,
    scalarMinMs: result.scalar.minMs,
    optimizedMinMs: result.optimized.minMs,
    width: d.width,
    height: d.height,
    coveredA: d.coveredA,
    coveredB: d.coveredB,
    maskAssetDiff: d.maskAssetDiff,
    maskPrimDiff: d.maskPrimDiff,
    colorDiffPixels: d.colorDiffPixels,
    depthDiff: d.depthDiff,
    maxColorDelta: d.maxColorDelta,
    outlineCountA: d.outlineCountA,
    outlineCountB: d.outlineCountB,
    outlinePointCountA: d.outlinePointCountA,
    outlinePointCountB: d.outlinePointCountB,
    outlineMaxDelta: d.outlineMaxDelta,
    scalarStats: result.scalar.stats,
    optimizedStats: result.optimized.stats,
  };
}

function buildDiffLayer(result) {
  const { width, height } = result.diff;
  if (!diffLayer) diffLayer = document.createElement('canvas');
  diffLayer.width = width;
  diffLayer.height = height;
  const c = diffLayer.getContext('2d');
  const img = c.createImageData(width, height);
  const xor = result.coverageXor;
  let marked = 0;
  for (let i = 0; i < width * height; i++) {
    if (xor[i]) {
      img.data[i * 4] = 255;
      img.data[i * 4 + 1] = 45;
      img.data[i * 4 + 2] = 90;
      img.data[i * 4 + 3] = 220;
      marked++;
    }
  }
  c.putImageData(img, 0, 0);
  return marked;
}

function renderBenchOut(result, marked) {
  const d = result.diff;
  const b = result.optimized.stats;
  const a = result.scalar.stats;
  const verdict = d.identical ? 'PASS · 逐像素/深度/轮廓完全一致' : 'FAIL · 存在差异，见上';
  const lines = [
    verdict,
    `覆盖像素 scalar=${d.coveredA} optimized=${d.coveredB}`,
    `maskAsset 差异=${d.maskAssetDiff}  maskPrim 差异=${d.maskPrimDiff}`,
    `颜色差异像素=${d.colorDiffPixels} (max Δ=${d.maxColorDelta})  深度差异=${d.depthDiff}`,
    `轮廓 条数 ${d.outlineCountA}/${d.outlineCountB}  点数 ${d.outlinePointCountA}/${d.outlinePointCountB}  最大坐标Δ=${d.outlineMaxDelta}`,
    `计时中位数 scalar=${result.scalar.medianMs}ms  optimized=${result.optimized.medianMs}ms  提速=${result.speedup}x`,
    `bbox槽位=${a.bboxPixels}  scalar边测试=${a.fragmentsTested}  opt边测试=${b.fragmentsTested}`,
    `瓦片 空/满/部分=${b.tileEmpty}/${b.tileFull}/${b.tilePartial}  整瓦证明像素=${b.proofAcceptedPixels}`,
    `空瓦跳过=${b.emptyTilePixelsSkipped}  提前结束=${b.spanEarlyExits}  小盒旁路=${b.scalarSmallBypass}  安全回退=${b.scalarFallbacks}`,
    `XOR 叠加标记像素=${marked}（勾选框可在画布查看；0 表示无差异）`,
  ];
  ui.benchOut.dataset.state = d.identical ? 'pass' : 'fail';
  ui.benchOut.textContent = lines.join('\n');
}

async function runCompare() {
  ui.benchCompareBtn.disabled = true;
  ui.benchOut.dataset.state = 'running';
  ui.benchOut.textContent = '运行中：同一当前场景分别真实运行 scalar 与 optimized 内核（含预热与计时）…';
  // Let the browser paint the "running" message before the synchronous bench.
  await new Promise((r) => setTimeout(r, 40));
  try {
    const { benchmarkPair } = await import('../src/core/compare.js');
    const scene = buildScene(state);
    const result = benchmarkPair(scene, { warmup: 3, iterations: 9, topCosts: 8 });
    lastCompare = result;
    const marked = buildDiffLayer(result);
    ui.diffOverlay.checked = true;
    renderBenchOut(result, marked);
    if (frame) paintOverlay();
    return compareSummary(result);
  } catch (err) {
    ui.benchOut.dataset.state = 'error';
    ui.benchOut.textContent = '对照运行失败: ' + err.message;
    throw err;
  } finally {
    ui.benchCompareBtn.disabled = false;
  }
}

ui.benchCompareBtn.addEventListener('click', () => {
  runCompare().catch(() => {});
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

syncUiFromState();
render();

// Expose for automated verification (tools/verify-ui.mjs) and manual probing.
window.__raster = {
  getFrame: () => frame,
  getState: () => state,
  render,
  auditPixel: (x, y) => auditPixel(frame, x, y),
  setKernel(k) {
    ui.kernel.value = k;
    scheduleRender();
  },
  // Runs the REAL button path and resolves to a JSON-safe comparison summary.
  compare: () => runCompare(),
  clickCompareButton: () => {
    ui.benchCompareBtn.click();
  },
  getLastCompare: () => (lastCompare ? compareSummary(lastCompare) : null),
  setState(patch) {
    Object.assign(state, patch);
    syncUiFromState();
    render();
  },
  setNear(v) {
    state.near = v;
    ui.near.value = v;
    ui.nearOut.textContent = v.toFixed(2);
    render();
  },
  setSeam(v) {
    state.seamWidth = v;
    ui.seam.value = v;
    ui.seamOut.textContent = v.toFixed(4);
    render();
  },
  pickPixel(x, y) {
    selected = { x, y };
    runAudit(x, y);
    paintOverlay();
  },
};
