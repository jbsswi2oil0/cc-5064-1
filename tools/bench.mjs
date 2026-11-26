// Honest CPU benchmark + cost decomposition for the two software kernels.
//
//   npm run bench            (or: node tools/bench.mjs)
//
// Every scenario renders the REAL scene through BOTH kernels, compares the
// resulting frame buffers element-by-element (coverage / prim ids / colour /
// depth / post-clip outlines), times each kernel with warm-up + median, and
// reports the real fragment counters. It writes out/bench.txt and
// out/bench.json. It exits non-zero if any pixel/outline difference is found,
// so a green run is actual execution evidence rather than a claim.

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { buildScene, DEFAULT_STATE } from '../src/core/scene.js';
import { benchmarkPair } from '../src/core/compare.js';
import { thinScene, tinyScene, extremeScene } from './bench-scenes.mjs';

const outDir = process.env.BENCH_OUT || new URL('../out', import.meta.url).pathname;
const ITER = Number(process.env.BENCH_ITER ?? 15);
const WARM = Number(process.env.BENCH_WARM ?? 4);

function stateWith(patch = {}) {
  const s = JSON.parse(JSON.stringify(DEFAULT_STATE));
  s.visible = { welded: true, seamed: true, span: true, sliver: true, degenerate: false };
  return Object.assign(s, patch);
}

const scenarios = [
  { id: 'app-default-960x600', label: 'app default scene', scene: buildScene(stateWith()) },
  {
    id: 'app-large-1600x1000',
    label: 'app scene at 1600x1000',
    scene: buildScene(stateWith({ viewport: { width: 1600, height: 1000 } })),
  },
  {
    id: 'app-near-cut',
    label: 'near plane cutting welded panel',
    scene: buildScene(
      stateWith({
        near: 2.25,
        orbit: { yaw: 0.12, pitch: 0.04, distance: 1.0, target: [0.35, 0.02, 0] },
      })
    ),
  },
  {
    id: 'app-near-005',
    label: 'very near plane 0.05',
    scene: buildScene(stateWith({ near: 0.05 })),
  },
  { id: 'thin-1280x800', label: '700 long thin triangles', scene: thinScene({}) },
  {
    id: 'thin-reversed-1280x800',
    label: 'same thin triangles, reversed submission',
    scene: thinScene({ reverse: true }),
  },
  { id: 'tiny-960x600', label: '6000 tiny triangles', scene: tinyScene({}) },
  { id: 'extreme-320x240', label: 'extreme coordinates (guard/fallback)', scene: extremeScene() },
];

function n(x) {
  return Number(x).toLocaleString('en-US');
}
function fx(x, d = 3) {
  return Number(x).toFixed(d);
}

function row(s) {
  const d = s.result.diff;
  const bad = d.maskAssetDiff + d.maskPrimDiff + d.colorDiffPixels + d.depthDiff + d.outlineMaxDelta;
  return {
    id: s.id,
    label: s.label,
    width: s.width,
    height: s.height,
    scalarMs: s.result.scalar.medianMs,
    optimizedMs: s.result.optimized.medianMs,
    speedup: s.result.speedup,
    minMs: [s.result.scalar.minMs, s.result.optimized.minMs],
    maxMs: [s.result.scalar.maxMs, s.result.optimized.maxMs],
    identical: d.identical,
    maskAssetDiff: d.maskAssetDiff,
    maskPrimDiff: d.maskPrimDiff,
    colorDiffPixels: d.colorDiffPixels,
    depthDiff: d.depthDiff,
    maxColorDelta: d.maxColorDelta,
    outlineMaxDelta: d.outlineMaxDelta,
    covered: d.coveredA,
    scalar: s.result.scalar.stats,
    optimized: s.result.optimized.stats,
    workRatio:
      s.result.optimized.stats.fragmentsTested > 0
        ? Number((s.result.scalar.stats.fragmentsTested / s.result.optimized.stats.fragmentsTested).toFixed(2))
        : null,
  };
}

function printTable(rows, log) {
  log('');
  log('SCENE                          scalar ms   opt ms   speedup   maskDiff colorDiff depthDiff outlineΔ');
  log('-----------------------------  ---------  -------  --------  -------- --------- --------- --------');
  for (const r of rows) {
    log(
      [
        r.label.padEnd(29),
        fx(r.scalarMs, 2).padStart(9),
        fx(r.optimizedMs, 2).padStart(8),
        (fx(r.speedup, 2) + 'x').padStart(8),
        n(r.maskAssetDiff).padStart(8),
        n(r.colorDiffPixels).padStart(9),
        n(r.depthDiff).padStart(9),
        fx(r.outlineMaxDelta, 4).padStart(8),
      ].join('  ')
    );
  }
}

function printCostBreakdown(r, log) {
  const a = r.scalar;
  const b = r.optimized;
  log('');
  log(`COST DECOMPOSITION — ${r.label} (${r.width}x${r.height})`);
  log(`  timing median (min..max): scalar ${r.scalarMs} ms (${r.minMs[0]}..${r.maxMs[0]})`);
  log(`                            opt    ${r.optimizedMs} ms (${r.minMs[1]}..${r.maxMs[1]})   speedup ${r.speedup}x`);
  log(`  triangles/fan/clipped/gen: ${b.triangles}/${b.fanTriangles}/${b.clippedTriangles}/${b.generatedVertices}`);
  log(`  bbox traversal slots      : ${n(a.bboxPixels)}   (identical upper bound both kernels)`);
  log(`  scalar edge tests         : ${n(a.fragmentsTested)}  (= every bbox slot)`);
  log(`  opt explicit edge tests   : ${n(b.fragmentsTested)}  -> ${r.workRatio}x fewer sign tests`);
  log(`  opt proof-accepted pixels : ${n(b.proofAcceptedPixels)} (full 8x8 tiles, no per-pixel edge test)`);
  log(`  empty-tile pixels skipped : ${n(b.emptyTilePixelsSkipped)}`);
  log(`  tiles empty/full/partial  : ${n(b.tileEmpty)} / ${n(b.tileFull)} / ${n(b.tilePartial)}`);
  log(`  span early terminations   : ${n(b.spanEarlyExits)}`);
  log(`  small-bbox scalar bypass  : ${n(b.scalarSmallBypass)}`);
  log(`  safe-integer fallbacks    : ${n(b.scalarFallbacks)}`);
  log(`  inside / shaded (scalar)  : ${n(a.fragmentsPassed)} / ${n(a.fragmentsShaded)}`);
  log(`  inside / shaded (opt)     : ${n(b.fragmentsPassed)} / ${n(b.fragmentsShaded)}`);
}

function printTopPrims(result, log) {
  log('');
  log('TOP PRIMITIVE COSTS (optimized, by bounding-box pixels)');
  log('  asset        tri  planes       kernel     bbox      edgeTest inside  shaded  E/F/P tiles');
  for (const p of result.primCosts.slice(0, 8)) {
    log(
      [
        String(p.asset).padEnd(12),
        String(p.triangle).padStart(4),
        String(p.planesHit).padEnd(12),
        String(p.kernel).padEnd(10),
        n(p.bboxPixels).padStart(8),
        n(p.edgeTests).padStart(9),
        n(p.inside).padStart(7),
        n(p.shaded).padStart(7),
        `${p.tileEmpty}/${p.tileFull}/${p.tilePartial}`.padStart(13),
      ].join(' ')
    );
  }
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const lines = [];
  const log = (s = '') => {
    lines.push(s);
    console.log(s);
  };

  log('==============================================================');
  log(' SOFTWARE RASTERIZER — KERNEL SPEED / CORRECTNESS BENCHMARK');
  log('==============================================================');
  log(`date: ${new Date().toISOString()}  node: ${process.version}  platform: ${process.platform}/${process.arch}`);
  log(`method: warmup=${WARM}, median of ${ITER} timed runs per kernel; both kernels render the same real scene`);
  log('correctness: exact Uint8/element buffer + outline comparison (no tolerance, no canned data)');

  const results = [];
  const rows = [];
  let anyDiff = false;

  for (const sc of scenarios) {
    process.stdout.write(`bench ${sc.id} ... `);
    const result = benchmarkPair(sc.scene, { warmup: WARM, iterations: ITER, topCosts: 10 });
    const width = sc.scene.state.viewport.width;
    const height = sc.scene.state.viewport.height;
    const entry = { ...sc, result, width, height };
    results.push(entry);
    const r = row(entry);
    rows.push(r);
    if (!result.diff.identical) anyDiff = true;
    console.log(`${result.speedup}x  diffs mask=${result.diff.maskAssetDiff} color=${result.diff.colorDiffPixels}`);
  }

  printTable(rows, log);

  // Detailed decomposition for the two headline workloads.
  for (const id of ['thin-1280x800', 'app-default-960x600', 'tiny-960x600']) {
    const entry = results.find((x) => x.id === id);
    const r = rows.find((x) => x.id === id);
    printCostBreakdown(r, log);
    printTopPrims(entry.result, log);
  }

  // Explicit forward-vs-reverse coverage parity for the thin workload.
  const thinF = rows.find((r) => r.id === 'thin-1280x800');
  const thinR = rows.find((r) => r.id === 'thin-reversed-1280x800');
  log('');
  log('REVERSED SUBMISSION PARITY (thin triangles)');
  log(`  covered forward=${n(thinF.covered)}  reversed=${n(thinR.covered)}`);
  log(`  median forward=${thinF.scalarMs}/${thinF.optimizedMs} ms (scalar/opt)`);
  log(`  median reversed=${thinR.scalarMs}/${thinR.optimizedMs} ms (scalar/opt)`);

  log('');
  if (anyDiff) {
    log('RESULT: FAIL — at least one scenario has non-zero pixel/outline differences.');
  } else {
    log('RESULT: PASS — scalar and optimized kernels are pixel/outline identical on every scenario.');
  }
  log('Timings are container-specific medians; the counters and diffs are the reproducible evidence.');

  const report = lines.join('\n') + '\n';
  await writeFile(join(outDir, 'bench.txt'), report);

  const json = {
    generatedAt: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform}/${process.arch}`,
    method: { warmup: WARM, iterations: ITER, statistic: 'median' },
    pass: !anyDiff,
    scenarios: rows,
    reversedParity: {
      coveredForward: thinF.covered,
      coveredReversed: thinR.covered,
      equal: thinF.covered === thinR.covered,
    },
  };
  await writeFile(join(outDir, 'bench.json'), JSON.stringify(json, null, 2) + '\n');
  console.log(`\nwrote ${join(outDir, 'bench.txt')} and bench.json`);

  if (anyDiff) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
