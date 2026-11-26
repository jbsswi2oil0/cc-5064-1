// Deterministic stress scenes for the triangle-processing benchmark and the
// differential kernel tests. Triangles are specified in PIXEL coordinates and
// converted to clip space; the scenes use an identity view-projection, so
// world == clip and w == 1, which makes thin/tiny coverage exact to reason
// about. Perspective/near-plane stress comes from buildScene() separately.

import { identity4 } from '../src/core/math.js';
import { makeCheckerTexture } from '../src/core/texture.js';

const checker = makeCheckerTexture(64, 8);
const ID = identity4();

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pxToWorld(x, y, width, height, z = 0) {
  return [(2 * x) / width - 1, 1 - (2 * y) / height, z, 1];
}

function makeTri(index, ptsPx, width, height, extra = {}) {
  const world = ptsPx.map(([x, y]) => pxToWorld(x, y, width, height, extra.z ?? 0));
  let uv = [
    [0.05, 0.05],
    [0.95, 0.1],
    [0.5, 0.95],
  ];
  let color = [
    [1, 0.35 + (index % 5) * 0.12, 0.4],
    [0.35, 0.6, 1],
    [1, 0.9, 0.35],
  ];
  // When positions are submitted reversed, the attributes must travel with
  // their vertices (full tuple reversal), not stay bound to slot indices.
  if (extra.reverseAttrs) {
    uv = [uv[2], uv[1], uv[0]];
    color = [color[2], color[1], color[0]];
  }
  return {
    assetId: 'stress',
    index,
    keys: [`s${index}a`, `s${index}b`, `s${index}c`],
    world,
    uv,
    color,
    normal: [0, 0, 1],
    texture: checker,
  };
}

export function makeStressScene(width, height, triangles, id = 'stress') {
  return {
    state: { viewport: { width, height }, visible: { [id]: true } },
    camera: { eye: [0, 0, 3], target: [0, 0, 0] },
    view: ID,
    proj: ID,
    viewProj: ID,
    invViewProj: ID,
    assets: [
      {
        id,
        label: id,
        kind: 'stress',
        triangles,
        texture: checker,
        offset: [0, 0, 0],
        referenceQuad: [],
      },
    ],
  };
}

/**
 * Many long, very thin isosceles triangles at varied angles. Their bounding
 * boxes are large while their covered area is tiny: the worst case for a
 * naive bounding-box scan. reverse submits the same triangles with reversed
 * vertex order to exercise winding normalization.
 */
export function thinScene({ width = 1280, height = 800, count = 700, reverse = false, seed = 1337 } = {}) {
  const rand = mulberry32(seed);
  const tris = [];
  for (let i = 0; i < count; i++) {
    const cx = rand() * width;
    const cy = rand() * height;
    const angle = rand() * Math.PI;
    const length = 220 + rand() * 760;
    const halfW = 0.6 + rand() * 2.2; // ~1..4 px base width
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    const px = -dy;
    const py = dx;
    const tip = [cx + (dx * length) / 2, cy + (dy * length) / 2];
    const baseL = [cx - (dx * length) / 2 + px * halfW, cy - (dy * length) / 2 + py * halfW];
    const baseR = [cx - (dx * length) / 2 - px * halfW, cy - (dy * length) / 2 - py * halfW];
    let pts = [tip, baseL, baseR];
    if (reverse) pts = [pts[2], pts[1], pts[0]];
    tris.push(makeTri(i, pts, width, height, { reverseAttrs: reverse }));
  }
  return makeStressScene(width, height, tris, reverse ? 'thin-rev' : 'thin');
}

/**
 * Thousands of sub-pixel to few-pixel triangles. Exercises fixed-point
 * collapse (subpixelSkipped), setup cost and the exact-integer guard.
 */
export function tinyScene({ width = 960, height = 600, count = 6000, reverse = false, seed = 4242 } = {}) {
  const rand = mulberry32(seed);
  const tris = [];
  for (let i = 0; i < count; i++) {
    const cx = rand() * width;
    const cy = rand() * height;
    const r = 0.25 + rand() * 3.2;
    const a0 = rand() * Math.PI * 2;
    const pts = [0, 1, 2].map((k) => {
      const a = a0 + (k * Math.PI * 2) / 3 + (rand() - 0.5) * 0.6;
      const rr = r * (0.5 + rand());
      return [cx + Math.cos(a) * rr, cy + Math.sin(a) * rr];
    });
    let ordered = pts;
    if (reverse) ordered = [pts[2], pts[1], pts[0]];
    tris.push(makeTri(i, ordered, width, height, { reverseAttrs: reverse }));
  }
  return makeStressScene(width, height, tris, reverse ? 'tiny-rev' : 'tiny');
}

/**
 * A handful of triangles with extreme clip coordinates. Their screen vertices
 * are astronomically far away, forcing the safe-integer guard to fall back to
 * the scalar reference; used to prove near-clip/extreme input equivalence and
 * to price the fallback path honestly.
 */
export function extremeScene({ width = 320, height = 240 } = {}) {
  const huge = 2e8;
  const pts = [
    [-huge, -huge],
    [huge * 1.7, -huge * 0.4],
    [huge * 0.2, huge * 1.3],
  ];
  const tris = [makeTri(0, pts, width, height)];
  // A second moderate triangle so the scene also paints something measurable.
  tris.push(
    makeTri(
      1,
      [
        [40, 40],
        [280, 60],
        [160, 210],
      ],
      width,
      height
    )
  );
  return makeStressScene(width, height, tris, 'extreme');
}
