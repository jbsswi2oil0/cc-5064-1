// Homogeneous view-frustum clipping against the six clip-space half-spaces.
//
// Clip space is OpenGL-style, right-handed:
//   -w <= x <= w,  -w <= y <= w,  -w <= z <= w,  w > 0
// so the near plane is z >= -w and NDC z == -1 at the near plane.
//
// Clipping is done BEFORE the perspective divide, on 4D homogeneous
// coordinates. Attributes (uv, color) are interpolated with the same
// barycentric parameter t as the clip-space position, so after the divide
// they are perspective-correct by construction. Doing the divide first and
// then clipping in 2D loses the classification of points with w <= 0 and is
// the classic source of cracks around the near plane.
//
// Welding: vertices carry a stable string key. Generated intersection
// vertices are cached per render pass under a key derived from the plane and
// the *sorted* endpoint keys. Two triangles that share an original edge get
// bit-identical intersection vertices, no matter in which direction each
// triangle traverses that edge. Triangles that merely sit close together but
// use distinct vertex keys get distinct intersections, so a deliberate seam
// stays open.

export const CLIP_PLANES = [
  { name: 'near', inside: (c) => c[2] + c[3] },   // z + w >= 0
  { name: 'far', inside: (c) => c[3] - c[2] },    // w - z >= 0
  { name: 'left', inside: (c) => c[0] + c[3] },   // x + w >= 0
  { name: 'right', inside: (c) => c[3] - c[0] },  // w - x >= 0
  { name: 'bottom', inside: (c) => c[1] + c[3] }, // y + w >= 0
  { name: 'top', inside: (c) => c[3] - c[1] },    // w - y >= 0
];

export const MIN_W = 1e-7;

export function makeClipVertex(key, clip, uv, color) {
  return { key, clip, uv, color, generatedBy: null };
}

function canonical(a, b) {
  return a.key < b.key ? [a, b] : [b, a];
}

function intersect(a, b, plane, planeIndex, ctx) {
  const [p, q] = canonical(a, b);
  const key = `g${planeIndex}(${p.key}|${q.key})`;
  const cached = ctx.edgeCache.get(key);
  if (cached) return cached;

  const dp = plane.inside(p.clip);
  const dq = plane.inside(q.clip);
  const denom = dp - dq;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-300) return null;
  const t = dp / denom;
  if (!Number.isFinite(t)) return null;

  const clip = [
    p.clip[0] + t * (q.clip[0] - p.clip[0]),
    p.clip[1] + t * (q.clip[1] - p.clip[1]),
    p.clip[2] + t * (q.clip[2] - p.clip[2]),
    p.clip[3] + t * (q.clip[3] - p.clip[3]),
  ];
  const uv = [
    p.uv[0] + t * (q.uv[0] - p.uv[0]),
    p.uv[1] + t * (q.uv[1] - p.uv[1]),
  ];
  const color = [
    p.color[0] + t * (q.color[0] - p.color[0]),
    p.color[1] + t * (q.color[1] - p.color[1]),
    p.color[2] + t * (q.color[2] - p.color[2]),
  ];
  for (const c of [...clip, ...uv, ...color]) {
    if (!Number.isFinite(c)) return null;
  }

  const v = makeClipVertex(key, clip, uv, color);
  v.generatedBy = plane.name;
  ctx.edgeCache.set(key, v);
  ctx.generated.push(v);
  return v;
}

/**
 * Clip a triangle (array of 3 clip vertices) against all six planes.
 * Returns { polygon, planesHit } where polygon is an array of clip vertices
 * (0, 3..9 entries) and planesHit is the sorted list of planes that actually
 * removed or added something.
 */
export function clipTriangle(vertices, ctx) {
  let polygon = vertices;
  const planesHit = new Set();

  for (let planeIndex = 0; planeIndex < CLIP_PLANES.length; planeIndex++) {
    if (polygon.length === 0) break;
    const plane = CLIP_PLANES[planeIndex];
    const out = [];
    let touched = false;

    const push = (v) => {
      if (!v) return;
      if (out.length && out[out.length - 1].key === v.key) return;
      out.push(v);
    };

    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      const da = plane.inside(a.clip);
      const db = plane.inside(b.clip);
      if (!Number.isFinite(da) || !Number.isFinite(db)) return { polygon: [], planesHit: [...planesHit] };
      const aIn = da >= 0;
      const bIn = db >= 0;

      if (aIn) push(a);
      if (aIn !== bIn) {
        touched = true;
        const v = intersect(a, b, plane, planeIndex, ctx);
        if (!v) return { polygon: [], planesHit: [...planesHit] };
        push(v);
      }
    }
    if (out.length > 1 && out[0].key === out[out.length - 1].key) out.pop();
    if (touched) planesHit.add(plane.name);
    polygon = out;
  }

  return { polygon, planesHit: [...planesHit] };
}

export function createClipContext() {
  return { edgeCache: new Map(), generated: [] };
}
