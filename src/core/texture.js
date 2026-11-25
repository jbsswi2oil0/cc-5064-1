// Procedural texture atlas. A real texel array sampled by the rasterizer;
// bilinear filtering with wrap addressing. Values are 0..1 RGB, sRGB-ish
// (we do not linearize, so the numeric audit can reproduce samples exactly).

export function makeCheckerTexture(size = 64, cell = 8) {
  const data = new Float32Array(size * size * 3);
  const baseA = [0.14, 0.20, 0.31];
  const baseB = [0.80, 0.72, 0.52];
  const accent = [0.86, 0.34, 0.24];
  const grid = [0.05, 0.06, 0.09];

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cx = Math.floor(x / cell);
      const cy = Math.floor(y / cell);
      const checker = (cx + cy) & 1;
      let c = checker ? baseB : baseA;
      c = c.slice();
      // Diagonal accent band inside every tile, offset per tile.
      const lx = x % cell;
      const ly = y % cell;
      const diag = (lx + ly + (checker ? 3 : 0)) % cell;
      if (diag === 0 || diag === 1) {
        c = accent;
      }
      // Dark grid lines at tile borders.
      if (lx === 0 || ly === 0) {
        c = grid;
      }
      const i = (y * size + x) * 3;
      data[i] = c[0];
      data[i + 1] = c[1];
      data[i + 2] = c[2];
    }
  }
  return { size, data };
}

export function makeFloorTexture(size = 64, cell = 8) {
  const data = new Float32Array(size * size * 3);
  const a = [0.16, 0.22, 0.27];
  const b = [0.55, 0.57, 0.52];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const checker = ((Math.floor(x / cell) + Math.floor(y / cell)) & 1) !== 0;
      const c = checker ? b : a;
      const lx = x % cell;
      const ly = y % cell;
      const edge = lx === 0 || ly === 0;
      const i = (y * size + x) * 3;
      data[i] = edge ? 0.07 : c[0];
      data[i + 1] = edge ? 0.08 : c[1];
      data[i + 2] = edge ? 0.10 : c[2];
    }
  }
  return { size, data };
}

export function makeBackdropTexture(size = 64, cell = 16) {
  const data = new Float32Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const checker = ((Math.floor(x / cell) + Math.floor(y / cell)) & 1) !== 0;
      const base = checker ? [0.10, 0.13, 0.20] : [0.07, 0.09, 0.15];
      const edge = x % cell === 0 || y % cell === 0;
      const i = (y * size + x) * 3;
      data[i] = edge ? base[0] + 0.05 : base[0];
      data[i + 1] = edge ? base[1] + 0.06 : base[1];
      data[i + 2] = edge ? base[2] + 0.09 : base[2];
    }
  }
  return { size, data };
}

export function wrap01(t) {
  return t - Math.floor(t);
}

/** Bilinear sample with wrap. Returns [r, g, b] in 0..1. */
export function sampleBilinear(tex, u, v) {
  const size = tex.size;
  const data = tex.data;
  const fx = wrap01(u) * size - 0.5;
  const fy = wrap01(v) * size - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const xa = ((x0 % size) + size) % size;
  const xb = (xa + 1) % size;
  const ya = ((y0 % size) + size) % size;
  const yb = (ya + 1) % size;

  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const p00 = data[(ya * size + xa) * 3 + c];
    const p10 = data[(ya * size + xb) * 3 + c];
    const p01 = data[(yb * size + xa) * 3 + c];
    const p11 = data[(yb * size + xb) * 3 + c];
    const top = p00 + (p10 - p00) * tx;
    const bot = p01 + (p11 - p01) * tx;
    out[c] = top + (bot - top) * ty;
  }
  return out;
}
