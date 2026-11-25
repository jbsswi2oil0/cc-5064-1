// Scene construction. Right-handed world, camera looks down -Z (OpenGL style).
// All geometry is authored in world space; assets can be translated by the UI.

import {
  lookAt,
  perspective,
  multiply4,
  invert4,
  add,
  sub,
  scale,
  normalize,
  cross,
  dot,
  clamp,
} from './math.js';
import { makeCheckerTexture, makeFloorTexture } from './texture.js';

export const WELDED = 'welded';
export const SEAMED = 'seamed';
export const SPAN = 'span';
export const SLIVER = 'sliver';
export const DEGENERATE = 'degenerate';

const checker = makeCheckerTexture(64, 8);
const floor = makeFloorTexture(64, 8);

export const LIGHT_DIR = normalize([-0.35, 0.55, 0.75]);

function rotateEuler(p, yaw, pitch, roll) {
  // R = Ry(yaw) * Rx(pitch) * Rz(roll), applied to local point.
  let [x, y, z] = p;
  // Rz
  const cz = Math.cos(roll), sz = Math.sin(roll);
  [x, y] = [x * cz - y * sz, x * sz + y * cz];
  // Rx
  const cx = Math.cos(pitch), sx = Math.sin(pitch);
  [y, z] = [y * cx - z * sx, y * sx + z * cx];
  // Ry
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  [x, z] = [x * cy + z * sy, -x * sy + z * cy];
  return [x, y, z];
}

function defaultColors() {
  return [
    [1.0, 0.82, 0.78],
    [0.78, 0.92, 1.0],
    [1.0, 0.96, 0.80],
    [0.82, 1.0, 0.86],
  ];
}

function tri(assetId, index, keys, world, uv, color, texture) {
  const n = normalize(cross(sub(world[1], world[0]), sub(world[2], world[0])));
  return {
    assetId,
    index,
    keys,
    world,
    uv,
    color,
    normal: n,
    texture,
  };
}

/**
 * Build a quad split into two triangles. This is the "two triangles sharing
 * an edge" test asset, authored as if coming from two different-precision
 * models:
 *
 *   weld = true  -> the shared diagonal is declared to be the SAME vertices
 *                   (identical keys) on both triangles. Welding identity.
 *   weld = false -> triangle B's diagonal is the low-precision model's version
 *                   of that edge: it is displaced in-plane by seamWidth, so
 *                   the two triangles are topologically adjacent but
 *                   geometrically split by a hairline seam.
 *
 * The displacement is deliberately smaller than the other panel features so
 * it only becomes visible (wider than one sample spacing) when the camera is
 * pushed in.
 */
export function buildPanelAsset({
  id,
  label,
  center,
  offset,
  halfW,
  halfH,
  yaw = 0,
  pitch = 0,
  roll = 0,
  seamWidth = 0,
  weld = true,
  texture = checker,
  colors = defaultColors(),
  uvScale = 1,
}) {
  const local = [
    [-halfW, -halfH, 0],
    [halfW, -halfH, 0],
    [halfW, halfH, 0],
    [-halfW, halfH, 0],
  ];
  const uvs = [
    [0, 0],
    [uvScale, 0],
    [uvScale, uvScale],
    [0, uvScale],
  ];

  const worldOf = (p) => {
    const r = rotateEuler(p, yaw, pitch, roll);
    const c = add(center, offset);
    return add(c, r);
  };

  const p = local.map(worldOf);
  const keys = [0, 1, 2, 3].map((i) => `${id}:p${i}`);

  const triangles = [
    tri(id, 0, [keys[0], keys[1], keys[2]], [p[0], p[1], p[2]], [uvs[0], uvs[1], uvs[2]], [colors[0], colors[1], colors[2]], texture),
  ];

  if (weld || seamWidth === 0) {
    triangles.push(
      tri(id, 1, [keys[0], keys[2], keys[3]], [p[0], p[2], p[3]], [uvs[0], uvs[2], uvs[3]], [colors[0], colors[2], colors[3]], texture)
    );
  } else {
    // Offset the shared diagonal of triangle B along the in-plane normal.
    const d = [local[2][0] - local[0][0], local[2][1] - local[0][1]];
    const len = Math.hypot(d[0], d[1]) || 1;
    const nx = -d[1] / len;
    const ny = d[0] / len;
    const nLocal = [nx * seamWidth, ny * seamWidth, 0];
    const q0 = worldOf(add(local[0], nLocal));
    const q2 = worldOf(add(local[2], nLocal));
    const sKeys = [`${id}:s0`, `${id}:s2`, keys[3]];
    triangles.push(
      tri(
        id,
        1,
        sKeys,
        [q0, q2, p[3]],
        [uvs[0], uvs[2], uvs[3]],
        [colors[0], colors[2], colors[3]],
        texture
      )
    );
  }

  return {
    id,
    label,
    kind: weld ? 'welded' : 'seamed',
    triangles,
    texture,
    referenceQuad: p,
    seamWidth: weld ? 0 : seamWidth,
    offset,
  };
}

function buildSliverAsset(offset = [0, 0, 0]) {
  const center = add([0, -0.56, 0.42], offset);
  const hw = 0.26;
  const hh = 0.0032;
  const w = [
    add(center, [-hw, -hh, 0]),
    add(center, [hw, -hh, 0.02]),
    add(center, [-hw * 0.15, hh, 0.01]),
  ];
  const uv = [
    [0.05, 0.45],
    [0.95, 0.5],
    [0.5, 0.55],
  ];
  const color = [
    [1.0, 0.25, 0.35],
    [0.25, 0.6, 1.0],
    [1.0, 0.9, 0.2],
  ];
  return {
    id: SLIVER,
    label: 'Thin sliver (极细三角形)',
    kind: 'sliver',
    triangles: [tri(SLIVER, 0, [`${SLIVER}:a`, `${SLIVER}:b`, `${SLIVER}:c`], w, uv, color, checker)],
    texture: checker,
    offset,
  };
}

function buildDegenerateAsset(offset = [0, 0, 0]) {
  const c = add([0.0, 0.72, 0.3], offset);
  const w = [
    add(c, [-0.25, 0, 0]),
    add(c, [0.0, 0, 0]),
    add(c, [0.25, 0, 0]),
  ];
  return {
    id: DEGENERATE,
    label: 'Degenerate probe (零面积)',
    kind: 'degenerate',
    triangles: [
      tri(
        DEGENERATE,
        0,
        [`${DEGENERATE}:a`, `${DEGENERATE}:b`, `${DEGENERATE}:c`],
        w,
        [[0, 0], [1, 0], [0, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
        checker
      ),
    ],
    texture: checker,
    offset,
  };
}

export const DEFAULT_STATE = {
  viewport: { width: 960, height: 600 },
  orbit: { yaw: 0.0, pitch: 0.09, distance: 2.55, target: [0, 0, 0] },
  fovDeg: 44,
  near: 0.15,
  far: 12,
  seamWidth: 0.0042,
  offsets: {
    welded: [0, 0, 0],
    seamed: [0, 0, 0],
    span: [0, 0, 0],
    sliver: [0, 0, 0],
  },
  visible: { welded: true, seamed: true, span: true, sliver: true, degenerate: false },
};

export function cloneState(s) {
  return {
    viewport: { ...s.viewport },
    orbit: { ...s.orbit, target: [...s.orbit.target] },
    fovDeg: s.fovDeg,
    near: s.near,
    far: s.far,
    seamWidth: s.seamWidth,
    offsets: Object.fromEntries(Object.entries(s.offsets).map(([k, v]) => [k, [...v]])),
    visible: { ...s.visible },
  };
}

export function cameraFromState(state) {
  const { yaw, pitch, distance, target } = state.orbit;
  const cp = Math.cos(pitch);
  const eye = add(target, [
    distance * cp * Math.sin(yaw),
    distance * Math.sin(pitch),
    distance * cp * Math.cos(yaw),
  ]);
  const up = [0, 1, 0];
  return { eye, target: [...target], up, fovDeg: state.fovDeg, near: state.near, far: state.far };
}

export function buildScene(state) {
  const s = cloneState(state);
  const { width, height } = s.viewport;
  const cam = cameraFromState(s);

  if (!(s.near > 0)) throw new Error(`near plane must be > 0 (got ${s.near})`);
  if (!(s.far > s.near)) throw new Error(`far plane must exceed near (near=${s.near}, far=${s.far})`);
  if (!(s.fovDeg > 1 && s.fovDeg < 179)) throw new Error(`invalid fov ${s.fovDeg}`);

  const view = lookAt(cam.eye, cam.target, cam.up);
  const proj = perspective((s.fovDeg * Math.PI) / 180, width / height, s.near, s.far);
  const viewProj = multiply4(proj, view);
  const invViewProj = invert4(viewProj);
  if (!invViewProj) throw new Error('view-projection matrix is singular');

  const assets = [];
  assets.push(
    buildPanelAsset({
      id: WELDED,
      label: 'Welded pair (焊接身份 · 高/低精度对齐)',
      center: [-0.56, 0.04, 0.18],
      offset: s.offsets.welded,
      halfW: 0.42,
      halfH: 0.33,
      yaw: 0.42,
      pitch: -0.13,
      roll: 0.05,
      weld: true,
    })
  );
  assets.push(
    buildPanelAsset({
      id: SEAMED,
      label: 'Seamed pair (不同精度 · 刻意留缝)',
      center: [0.56, 0.04, 0.02],
      offset: s.offsets.seamed,
      halfW: 0.42,
      halfH: 0.33,
      yaw: -0.42,
      pitch: 0.13,
      roll: -0.05,
      seamWidth: s.seamWidth,
      weld: false,
    })
  );
  assets.push(
    buildPanelAsset({
      id: SPAN,
      label: 'Tilted checker span (跨平面)',
      center: [0, 0.16, -1.35],
      offset: s.offsets.span,
      halfW: 3.1,
      halfH: 1.72,
      yaw: 0.62,
      pitch: 0.16,
      roll: 0.0,
      texture: floor,
      uvScale: 3,
      weld: true,
      colors: [
        [0.9, 0.9, 0.92],
        [0.9, 0.9, 0.92],
        [0.95, 0.93, 0.88],
        [0.95, 0.93, 0.88],
      ],
    })
  );
  assets.push(buildSliverAsset(s.offsets.sliver));
  assets.push(buildDegenerateAsset([0, 0, 0]));

  const visibleAssets = assets.filter((a) => s.visible[a.id]);

  return {
    state: s,
    camera: cam,
    view,
    proj,
    viewProj,
    invViewProj,
    assets,
    visibleAssets,
    textures: { checker, floor },
  };
}

export function projectPoint(viewProj, p) {
  const x = viewProj[0] * p[0] + viewProj[4] * p[1] + viewProj[8] * p[2] + viewProj[12];
  const y = viewProj[1] * p[0] + viewProj[5] * p[1] + viewProj[9] * p[2] + viewProj[13];
  const z = viewProj[2] * p[0] + viewProj[6] * p[1] + viewProj[10] * p[2] + viewProj[14];
  const w = viewProj[3] * p[0] + viewProj[7] * p[1] + viewProj[11] * p[2] + viewProj[15];
  return [x, y, z, w];
}

export function unprojectRay(invViewProj, ndcX, ndcY) {
  const apply = (z) => {
    const x = invViewProj[0] * ndcX + invViewProj[4] * ndcY + invViewProj[8] * z + invViewProj[12];
    const y = invViewProj[1] * ndcX + invViewProj[5] * ndcY + invViewProj[9] * z + invViewProj[13];
    const zz = invViewProj[2] * ndcX + invViewProj[6] * ndcY + invViewProj[10] * z + invViewProj[14];
    const w = invViewProj[3] * ndcX + invViewProj[7] * ndcY + invViewProj[11] * z + invViewProj[15];
    return [x / w, y / w, zz / w];
  };
  return { near: apply(-1), far: apply(1) };
}

export function shadeFactor(normal) {
  return 0.45 + 0.55 * Math.max(0, dot(normal, LIGHT_DIR));
}
