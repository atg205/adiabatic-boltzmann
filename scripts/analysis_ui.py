"""
Generate a self-contained HTML analysis report from benchmark results.

Usage:
    python generate_report.py                          # reads results/, writes report.html
    python generate_report.py --results path/to/results --output report.html
"""

import json
import argparse
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(root: Path) -> list[dict]:
    records = []
    for file in root.rglob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
        except Exception:
            print(f"  [warn] skipping: {file}")
            continue

        config = data["config"]
        history = data["history"]

        records.append(
            {
                "file": str(file),
                "size": int(config["size"]),
                "h": float(config["h"]),
                "rbm": config["rbm"],
                "n_hidden": int(config["n_hidden"] or config["size"]),
                "sampler": config["sampler"],
                "sampling_method": config["sampling_method"],
                "lr": float(config["learning_rate"]),
                "reg": float(config["regularization"]),
                "n_samples": int(config["n_samples"]),
                "seed": int(config["seed"]),
                "final_energy": float(data["final_energy"]),
                "exact_energy": float(data["exact_energy"]),
                "abs_error": float(abs(data["final_energy"] - data["exact_energy"])),
                "rel_error": float(
                    abs(data["final_energy"] - data["exact_energy"])
                    / abs(data["exact_energy"])
                    * 100
                ),
                "energy_curve": history.get("energy", []),
                "error_curve": history.get("error", []),
                "energy_error_curve": history.get("energy_error", []),
                "grad_norm_curve": history.get("grad_norm", []),
                "cond_curve": history.get("s_condition_number", []),
                "weight_norm_curve": history.get("weight_norm", []),
            }
        )

    return sorted(records, key=lambda r: (r["size"], r["h"], r["rel_error"]))


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VMC Benchmark Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg:        #0a0c10;
    --bg2:       #0f1218;
    --bg3:       #161b24;
    --border:    #1e2633;
    --border2:   #2a3444;
    --accent:    #3b82f6;
    --accent2:   #60a5fa;
    --green:     #10b981;
    --yellow:    #f59e0b;
    --red:       #ef4444;
    --orange:    #f97316;
    --text:      #e2e8f0;
    --text2:     #94a3b8;
    --text3:     #64748b;
    --mono:      'JetBrains Mono', monospace;
    --display:   'Syne', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    display: flex;
    align-items: baseline;
    gap: 24px;
    background: var(--bg2);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  header h1 {
    font-family: var(--display);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text);
  }
  header h1 span { color: var(--accent); }
  .run-count {
    font-size: 11px;
    color: var(--text3);
    font-weight: 500;
  }

  /* ── Layout ── */
  .layout {
    display: grid;
    grid-template-columns: 260px 1fr;
    min-height: calc(100vh - 73px);
  }

  /* ── Sidebar ── */
  aside {
    border-right: 1px solid var(--border);
    padding: 20px 0;
    background: var(--bg2);
    overflow-y: auto;
    position: sticky;
    top: 73px;
    height: calc(100vh - 73px);
  }

  .sidebar-section {
    padding: 0 16px 8px;
  }
  .sidebar-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--text3);
    text-transform: uppercase;
    padding: 16px 16px 8px;
    display: block;
  }

  .nh-btn {
    display: block;
    width: 100%;
    text-align: left;
    padding: 8px 16px;
    background: none;
    border: none;
    color: var(--text2);
    cursor: pointer;
    font-family: var(--mono);
    font-size: 12px;
    border-left: 2px solid transparent;
    transition: all 0.15s;
  }
  .nh-btn:hover { background: var(--bg3); color: var(--text); }
  .nh-btn.active {
    color: var(--accent2);
    border-left-color: var(--accent);
    background: rgba(59,130,246,0.06);
  }
  .nh-btn .badge {
    float: right;
    font-size: 10px;
    color: var(--text3);
    background: var(--bg3);
    padding: 1px 6px;
    border-radius: 3px;
  }
  .nh-btn .err-badge {
    float: right;
    font-size: 10px;
    margin-right: 6px;
    padding: 1px 6px;
    border-radius: 3px;
  }

  /* ── Main content ── */
  main {
    padding: 32px 40px;
    overflow-y: auto;
    background: var(--bg);
  }

  .panel { display: none; }
  .panel.active { display: block; }

  /* ── Overview panel ── */
  .heatmap-grid {
    display: grid;
    gap: 2px;
    margin-bottom: 32px;
  }
  .heatmap-cell {
    padding: 10px 8px;
    text-align: center;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: transform 0.1s, box-shadow 0.1s;
    position: relative;
  }
  .heatmap-cell:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    z-index: 2;
  }
  .heatmap-cell .cell-label { font-size: 10px; color: rgba(255,255,255,0.6); }
  .heatmap-cell .cell-val   { font-size: 13px; font-weight: 700; }

  /* ── Section headers ── */
  .section-title {
    font-family: var(--display);
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: -0.3px;
  }
  .section-sub {
    color: var(--text3);
    font-size: 11px;
    margin-bottom: 24px;
  }

  /* ── Stats row ── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 28px;
  }
  .stat-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
  }
  .stat-label { font-size: 10px; color: var(--text3); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
  .stat-value { font-size: 20px; font-weight: 700; font-family: var(--display); }
  .stat-sub   { font-size: 10px; color: var(--text3); margin-top: 2px; }
  .good  { color: var(--green); }
  .warn  { color: var(--yellow); }
  .bad   { color: var(--red); }

  /* ── Charts grid ── */
  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  .charts-grid.wide { grid-template-columns: 1fr; }
  .charts-grid.three { grid-template-columns: 1fr 1fr 1fr; }

  .chart-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
  }
  .chart-card.span2 { grid-column: span 2; }
  .chart-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 12px;
  }
  .chart-wrap { position: relative; height: 220px; }
  .chart-wrap.tall { height: 300px; }

  /* ── Run table ── */
  .run-table-wrap {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 24px;
  }
  .run-table-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .run-table-header span {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text3);
  }
  table { width: 100%; border-collapse: collapse; }
  th {
    background: var(--bg3);
    padding: 8px 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text3);
    text-align: left;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  th:hover { color: var(--text); }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    color: var(--text2);
    white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg3); }
  tr.selected td { background: rgba(59,130,246,0.08); }
  .pill {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
  }
  .pill-green  { background: rgba(16,185,129,0.15); color: var(--green); }
  .pill-yellow { background: rgba(245,158,11,0.15); color: var(--yellow); }
  .pill-red    { background: rgba(239,68,68,0.15);  color: var(--red); }

  /* ── Diagnostics panel ── */
  .diag-selector {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .diag-selector label { font-size: 11px; color: var(--text3); }
  .diag-selector select {
    background: var(--bg2);
    border: 1px solid var(--border2);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 4px;
    min-width: 260px;
  }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0;
  }

  /* ── Hyperparam sensitivity ── */
  .sensitivity-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <h1>VMC <span>Benchmark</span> Analysis</h1>
  <span class="run-count" id="run-count"></span>
</header>

<div class="layout">
  <aside>
    <span class="sidebar-label">Overview</span>
    <div class="sidebar-section">
      <button class="nh-btn active" onclick="showPanel('overview')" id="btn-overview">
        All runs
      </button>
    </div>

    <span class="sidebar-label">By (N, h)</span>
    <div id="nh-buttons"></div>
  </aside>

  <main>
    <!-- OVERVIEW PANEL -->
    <div class="panel active" id="panel-overview">
      <div class="section-title">Overview</div>
      <div class="section-sub">All runs across all (N, h) combinations</div>

      <div class="stats-row" id="overview-stats"></div>

      <div class="section-title" style="font-size:14px; margin-bottom:16px;">Error Heatmap — best relative error (%) per (N, h)</div>
      <div id="heatmap-container" style="margin-bottom:32px; overflow-x:auto;"></div>

      <div class="charts-grid">
        <div class="chart-card">
          <div class="chart-title">Relative Error Distribution</div>
          <div class="chart-wrap"><canvas id="overview-error-hist"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Error vs N (best per N)</div>
          <div class="chart-wrap"><canvas id="overview-error-n"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Error vs n_samples</div>
          <div class="chart-wrap"><canvas id="overview-error-samples"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Error vs Regularization</div>
          <div class="chart-wrap"><canvas id="overview-error-reg"></canvas></div>
        </div>
      </div>
    </div>

    <!-- PER (N,h) PANELS — injected by JS -->
    <div id="nh-panels"></div>
  </main>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
const ALL_RUNS = __RUNS_JSON__;

// ── Chart defaults ────────────────────────────────────────────────────────
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#1e2633';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;
const PALETTE = ['#3b82f6','#10b981','#f59e0b','#ef4444','#a855f7','#06b6d4','#f97316','#ec4899'];

// ── Helpers ───────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const fmtE  = v => typeof v === 'number' ? v.toExponential(2) : '—';
const fmtF  = (v, d=4) => typeof v === 'number' ? v.toFixed(d) : '—';
const fmtP  = v => typeof v === 'number' ? v.toFixed(3)+'%' : '—';
const mean  = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : NaN;
const minOf = arr => arr.length ? Math.min(...arr) : NaN;

function errClass(pct) {
  if (pct < 1)  return 'good';
  if (pct < 5)  return 'warn';
  return 'bad';
}
function pillClass(pct) {
  if (pct < 1)  return 'pill-green';
  if (pct < 5)  return 'pill-yellow';
  return 'pill-red';
}
function heatColor(val, min, max) {
  const t = Math.max(0, Math.min(1, (val - min) / (max - min + 1e-9)));
  const r = Math.round(16 + t * (239 - 16));
  const g = Math.round(185 - t * (185 - 68));
  const b = Math.round(129 - t * (129 - 68));
  return `rgb(${r},${g},${b})`;
}

function makeChart(id, type, data, options={}) {
  const existing = Chart.getChart(id);
  if (existing) existing.destroy();
  return new Chart($(id), {
    type, data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 10, padding: 12 } } },
      ...options
    }
  });
}

function lineDataset(label, curve, color, dash=[]) {
  return {
    label,
    data: curve.map((v,i)=>({x:i,y:v})),
    borderColor: color,
    backgroundColor: color+'22',
    borderWidth: 1.5,
    borderDash: dash,
    pointRadius: 0,
    tension: 0.3,
  };
}

// ── Group by (N, h) ───────────────────────────────────────────────────────
const nhMap = {};
ALL_RUNS.forEach(r => {
  const k = `N=${r.size}, h=${r.h}`;
  if (!nhMap[k]) nhMap[k] = [];
  nhMap[k].push(r);
});
const nhKeys = Object.keys(nhMap).sort((a,b)=>{
  const pa = a.match(/N=(\d+), h=([\d.]+)/), pb = b.match(/N=(\d+), h=([\d.]+)/);
  return (+pa[1] - +pb[1]) || (+pa[2] - +pb[2]);
});

$('run-count').textContent = `${ALL_RUNS.length} runs  ·  ${nhKeys.length} (N,h) pairs`;

// ── Build sidebar ─────────────────────────────────────────────────────────
const btnContainer = $('nh-buttons');
nhKeys.forEach(key => {
  const runs = nhMap[key];
  const best = Math.min(...runs.map(r=>r.rel_error));
  const btn = document.createElement('button');
  btn.className = 'nh-btn';
  btn.id = 'btn-' + key;
  btn.onclick = () => showPanel('nh-' + key);
  const cls = pillClass(best);
  btn.innerHTML = `${key}<span class="err-badge ${cls}">${best.toFixed(1)}%</span><span class="badge">${runs.length}</span>`;
  btnContainer.appendChild(btn);
});

// ── Panel switcher ────────────────────────────────────────────────────────
function showPanel(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nh-btn').forEach(b => b.classList.remove('active'));
  const panel = $('panel-' + id);
  if (panel) panel.classList.add('active');
  const btn = $('btn-' + id);
  if (btn) btn.classList.add('active');
}

// ── Overview panel ────────────────────────────────────────────────────────
function buildOverview() {
  const errors = ALL_RUNS.map(r=>r.rel_error);
  const best = Math.min(...errors);
  const med  = errors.slice().sort((a,b)=>a-b)[Math.floor(errors.length/2)];
  const worst = Math.max(...errors);

  $('overview-stats').innerHTML = `
    <div class="stat-card">
      <div class="stat-label">Total Runs</div>
      <div class="stat-value">${ALL_RUNS.length}</div>
      <div class="stat-sub">${nhKeys.length} (N,h) pairs</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Best Rel. Error</div>
      <div class="stat-value ${errClass(best)}">${fmtP(best)}</div>
      <div class="stat-sub">across all runs</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Median Rel. Error</div>
      <div class="stat-value ${errClass(med)}">${fmtP(med)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Worst Rel. Error</div>
      <div class="stat-value ${errClass(worst)}">${fmtP(worst)}</div>
    </div>
  `;

  // Heatmap
  const sizes = [...new Set(ALL_RUNS.map(r=>r.size))].sort((a,b)=>a-b);
  const hs    = [...new Set(ALL_RUNS.map(r=>r.h))].sort((a,b)=>a-b);
  const cellData = [];
  sizes.forEach(sz => hs.forEach(h => {
    const runs = ALL_RUNS.filter(r=>r.size===sz && r.h===h);
    if (runs.length) cellData.push({ sz, h, best: Math.min(...runs.map(r=>r.rel_error)), n: runs.length });
  }));
  const allBest = cellData.map(c=>c.best);
  const minV = Math.min(...allBest), maxV = Math.max(...allBest);

  const heatDiv = $('heatmap-container');
  const colW = Math.max(80, Math.floor(600 / hs.length));

  let html = `<div style="display:grid; grid-template-columns: 60px ${hs.map(()=>colW+'px').join(' ')}; gap:3px; min-width:${60+hs.length*(colW+3)}px;">`;
  html += `<div></div>`;
  hs.forEach(h => { html += `<div style="text-align:center;font-size:10px;color:var(--text3);padding-bottom:4px;">h=${h}</div>`; });
  sizes.forEach(sz => {
    html += `<div style="font-size:10px;color:var(--text3);display:flex;align-items:center;">N=${sz}</div>`;
    hs.forEach(h => {
      const cell = cellData.find(c=>c.sz===sz && c.h===h);
      if (!cell) { html += `<div style="height:56px;border-radius:4px;background:var(--bg3);"></div>`; return; }
      const bg = heatColor(cell.best, minV, maxV);
      const nhKey = `N=${sz}, h=${h}`;
      html += `<div class="heatmap-cell" style="background:${bg};" onclick="showPanel('nh-${nhKey}')">
        <div class="cell-label">N=${sz} h=${h}</div>
        <div class="cell-val">${cell.best.toFixed(2)}%</div>
        <div class="cell-label">${cell.n} runs</div>
      </div>`;
    });
  });
  html += '</div>';
  heatDiv.innerHTML = html;

  // Error histogram
  const buckets = 20;
  const bmin = 0, bmax = Math.min(100, Math.ceil(worst));
  const bsize = (bmax - bmin) / buckets;
  const hist = Array(buckets).fill(0);
  errors.forEach(e => { const i = Math.min(buckets-1, Math.floor((e-bmin)/bsize)); hist[i]++; });
  makeChart('overview-error-hist', 'bar', {
    labels: hist.map((_,i)=>(bmin+i*bsize).toFixed(1)+'%'),
    datasets: [{ label: 'runs', data: hist, backgroundColor: '#3b82f666', borderColor: '#3b82f6', borderWidth: 1 }]
  }, { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 45 } } } });

  // Error vs N
  const byN = {};
  ALL_RUNS.forEach(r => { if (!byN[r.size]) byN[r.size] = []; byN[r.size].push(r.rel_error); });
  makeChart('overview-error-n', 'bar', {
    labels: Object.keys(byN).map(n=>'N='+n),
    datasets: [
      { label: 'best',   data: Object.values(byN).map(a=>Math.min(...a)),             backgroundColor: '#10b98166', borderColor: '#10b981', borderWidth: 1 },
      { label: 'median', data: Object.values(byN).map(a=>a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)]), backgroundColor: '#3b82f666', borderColor: '#3b82f6', borderWidth: 1 },
    ]
  }, { scales: { y: { title: { display: true, text: 'Rel. error (%)' } } } });

  // Error vs n_samples
  const bySamp = {};
  ALL_RUNS.forEach(r => { if (!bySamp[r.n_samples]) bySamp[r.n_samples] = []; bySamp[r.n_samples].push(r.rel_error); });
  const sampKeys = Object.keys(bySamp).map(Number).sort((a,b)=>a-b);
  makeChart('overview-error-samples', 'line', {
    labels: sampKeys,
    datasets: [
      { label: 'mean', data: sampKeys.map(k=>mean(bySamp[k])), borderColor: '#3b82f6', backgroundColor: '#3b82f622', borderWidth: 2, pointRadius: 4 },
      { label: 'best', data: sampKeys.map(k=>Math.min(...bySamp[k])), borderColor: '#10b981', backgroundColor: '#10b98122', borderWidth: 2, pointRadius: 4 },
    ]
  }, { scales: { x: { title: { display: true, text: 'n_samples' } }, y: { title: { display: true, text: 'Rel. error (%)' } } } });

  // Error vs reg
  const byReg = {};
  ALL_RUNS.forEach(r => { if (!byReg[r.reg]) byReg[r.reg] = []; byReg[r.reg].push(r.rel_error); });
  const regKeys = Object.keys(byReg).map(Number).sort((a,b)=>a-b);
  makeChart('overview-error-reg', 'line', {
    labels: regKeys.map(v=>v.toExponential(0)),
    datasets: [
      { label: 'mean', data: regKeys.map(k=>mean(byReg[k])), borderColor: '#f59e0b', backgroundColor: '#f59e0b22', borderWidth: 2, pointRadius: 4 },
      { label: 'best', data: regKeys.map(k=>Math.min(...byReg[k])), borderColor: '#10b981', backgroundColor: '#10b98122', borderWidth: 2, pointRadius: 4 },
    ]
  }, { scales: { x: { title: { display: true, text: 'regularization' } }, y: { title: { display: true, text: 'Rel. error (%)' } } } });
}

// ── Per-(N,h) panel builder ───────────────────────────────────────────────
function buildNHPanel(key) {
  const runs = nhMap[key].slice().sort((a,b)=>a.rel_error-b.rel_error);
  const safeKey = key;
  const exact = runs[0].exact_energy;
  const errors = runs.map(r=>r.rel_error);
  const bestRun = runs[0];

  const container = document.createElement('div');
  container.className = 'panel';
  container.id = 'panel-nh-' + key;

  const bestE = Math.min(...errors), medE = errors[Math.floor(errors.length/2)], worstE = Math.max(...errors);

  container.innerHTML = `
    <div class="section-title">${key}</div>
    <div class="section-sub">Exact ground energy: ${fmtF(exact, 6)} &nbsp;·&nbsp; ${runs.length} runs</div>

    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-label">Exact Energy</div>
        <div class="stat-value" style="font-size:16px;">${fmtF(exact,4)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Best Rel. Error</div>
        <div class="stat-value ${errClass(bestE)}">${fmtP(bestE)}</div>
        <div class="stat-sub">abs: ${fmtF(bestRun.abs_error,6)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Median Error</div>
        <div class="stat-value ${errClass(medE)}">${fmtP(medE)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Worst Error</div>
        <div class="stat-value ${errClass(worstE)}">${fmtP(worstE)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Best Final Energy</div>
        <div class="stat-value" style="font-size:16px;">${fmtF(bestRun.final_energy,4)}</div>
      </div>
    </div>

    <!-- Convergence: top 5 runs -->
    <div class="charts-grid wide" style="margin-bottom:0">
      <div class="chart-card">
        <div class="chart-title">Energy Convergence — Top 5 runs</div>
        <div class="chart-wrap tall"><canvas id="conv-${key}"></canvas></div>
      </div>
    </div>
    <div class="charts-grid" style="margin-top:16px;">
      <div class="chart-card">
        <div class="chart-title">Error vs n_samples</div>
        <div class="chart-wrap"><canvas id="samp-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Error vs Regularization</div>
        <div class="chart-wrap"><canvas id="reg-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Error vs Learning Rate</div>
        <div class="chart-wrap"><canvas id="lr-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Error vs Hidden Units</div>
        <div class="chart-wrap"><canvas id="nh-${key}"></canvas></div>
      </div>
    </div>

    <hr class="divider">

    <!-- All runs table -->
    <div class="run-table-wrap">
      <div class="run-table-header">
        <span>All Runs</span>
        <span style="color:var(--text3)">${runs.length} experiments</span>
      </div>
      <div style="overflow-x:auto;">
        <table id="table-${key}">
          <thead><tr>
            <th>rank</th>
            <th>rel err%</th>
            <th>abs err</th>
            <th>final E</th>
            <th>lr</th>
            <th>reg</th>
            <th>n_samples</th>
            <th>n_hidden</th>
            <th>seed</th>
            <th>grad‖·‖ final</th>
            <th>κ(S) final</th>
            <th>‖w‖ final</th>
          </tr></thead>
          <tbody id="tbody-${key}"></tbody>
        </table>
      </div>
    </div>

    <hr class="divider">

    <!-- Diagnostics for selected run -->
    <div class="section-title" style="font-size:14px;margin-bottom:4px;">Run Diagnostics</div>
    <div class="section-sub">Select a row in the table above, or use the picker</div>
    <div class="diag-selector">
      <label>Run:</label>
      <select id="diag-sel-${key}" onchange="renderDiag('${key}', this.value)"></select>
    </div>
    <div class="charts-grid">
      <div class="chart-card span2">
        <div class="chart-title">Energy Convergence</div>
        <div class="chart-wrap tall"><canvas id="diag-energy-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Gradient Norm ‖x‖</div>
        <div class="chart-wrap"><canvas id="diag-grad-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">S Condition Number κ(S)</div>
        <div class="chart-wrap"><canvas id="diag-cond-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Weight Norm ‖w‖</div>
        <div class="chart-wrap"><canvas id="diag-weight-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Energy Statistical Error σ/√n</div>
        <div class="chart-wrap"><canvas id="diag-eerr-${key}"></canvas></div>
      </div>
    </div>
  `;

  $('nh-panels').appendChild(container);

  // Charts — deferred so DOM is ready
  setTimeout(() => {
    // Convergence top 5
    const top5 = runs.slice(0, 5);
    makeChart(`conv-${key}`, 'line', {
      datasets: [
        ...top5.map((r,i) => lineDataset(
          `#${i+1} err=${fmtP(r.rel_error)} lr=${r.lr} reg=${fmtE(r.reg)} ns=${r.n_samples}`,
          r.energy_curve, PALETTE[i]
        )),
        { label: `Exact: ${fmtF(exact,4)}`, data: [{x:0,y:exact},{x:Math.max(...top5.map(r=>r.energy_curve.length||0))-1,y:exact}],
          borderColor:'#ef444488', borderDash:[6,3], borderWidth:1.5, pointRadius:0 }
      ]
    }, { scales: { x: { type:'linear', title:{display:true,text:'iteration'} }, y: { title:{display:true,text:'energy'} } } });

    // Sensitivity charts helper
    function sensitivityChart(canvasId, field, label) {
      const grouped = {};
      runs.forEach(r => { const k=r[field]; if(!grouped[k]) grouped[k]=[]; grouped[k].push(r.rel_error); });
      const keys = Object.keys(grouped).map(Number).sort((a,b)=>a-b);
      makeChart(canvasId, 'line', {
        labels: keys.map(k=>field==='reg'?k.toExponential(1):String(k)),
        datasets: [
          { label:'mean %', data: keys.map(k=>mean(grouped[k])),             borderColor:'#3b82f6', backgroundColor:'#3b82f622', borderWidth:2, pointRadius:4 },
          { label:'best %', data: keys.map(k=>Math.min(...grouped[k])),       borderColor:'#10b981', backgroundColor:'#10b98122', borderWidth:2, pointRadius:4 },
        ]
      }, { scales: { x:{title:{display:true,text:label}}, y:{title:{display:true,text:'Rel. error (%)'}} } });
    }
    sensitivityChart(`samp-${key}`, 'n_samples', 'n_samples');
    sensitivityChart(`reg-${key}`,  'reg',        'regularization');
    sensitivityChart(`lr-${key}`,   'lr',         'learning rate');
    sensitivityChart(`nh-${key}`,   'n_hidden',   'n_hidden');

    // Table
    const tbody = $(`tbody-${key}`);
    const sel   = $(`diag-sel-${key}`);
    runs.forEach((r,i) => {
      const gn = r.grad_norm_curve, cc = r.cond_curve, wn = r.weight_norm_curve;
      const lastGrad = gn && gn.length ? gn[gn.length-1] : null;
      const lastCond = cc && cc.length ? cc[cc.length-1] : null;
      const lastW    = wn && wn.length ? wn[wn.length-1] : null;
      const tr = document.createElement('tr');
      tr.dataset.idx = i;
      tr.onclick = () => {
        document.querySelectorAll(`#table-${key} tr.selected`).forEach(t=>t.classList.remove('selected'));
        tr.classList.add('selected');
        sel.value = i;
        renderDiag(key, i);
      };
      tr.innerHTML = `
        <td>${i+1}</td>
        <td><span class="pill ${pillClass(r.rel_error)}">${fmtP(r.rel_error)}</span></td>
        <td>${fmtF(r.abs_error,6)}</td>
        <td>${fmtF(r.final_energy,4)}</td>
        <td>${r.lr}</td>
        <td>${fmtE(r.reg)}</td>
        <td>${r.n_samples}</td>
        <td>${r.n_hidden}</td>
        <td>${r.seed}</td>
        <td>${lastGrad !== null ? fmtF(lastGrad,4) : '—'}</td>
        <td>${lastCond !== null ? lastCond.toExponential(2) : '—'}</td>
        <td>${lastW !== null ? fmtF(lastW,4) : '—'}</td>
      `;
      tbody.appendChild(tr);

      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `#${i+1}  err=${fmtP(r.rel_error)}  lr=${r.lr}  reg=${fmtE(r.reg)}  ns=${r.n_samples}  seed=${r.seed}`;
      sel.appendChild(opt);
    });

    // Render diagnostics for best run by default
    renderDiag(key, 0);
    tbody.firstElementChild?.classList.add('selected');
  }, 0);
}

function renderDiag(key, idx) {
  const runs = nhMap[key].slice().sort((a,b)=>a.rel_error-b.rel_error);
  const r = runs[+idx];
  const exact = r.exact_energy;
  const iters = r.energy_curve.length || 100;

  function diagLine(canvasId, curve, label, color, logy=false) {
    const existing = Chart.getChart(canvasId);
    if (existing) existing.destroy();
    if (!$(canvasId)) return;
    new Chart($(canvasId), {
      type: 'line',
      data: { datasets: [lineDataset(label, curve, color)] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 8 } } },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'iteration' } },
          y: { type: logy && curve.some(v=>v>0) ? 'logarithmic' : 'linear',
               title: { display: true, text: label } }
        }
      }
    });
  }

  // Energy with exact line
  const enExisting = Chart.getChart(`diag-energy-${key}`);
  if (enExisting) enExisting.destroy();
  if ($(`diag-energy-${key}`)) {
    new Chart($(`diag-energy-${key}`), {
      type: 'line',
      data: { datasets: [
        lineDataset('VMC Energy', r.energy_curve, '#3b82f6'),
        { label: `Exact: ${fmtF(exact,4)}`,
          data: [{x:0,y:exact},{x:iters-1,y:exact}],
          borderColor: '#ef4444', borderDash:[6,3], borderWidth: 1.5, pointRadius: 0 }
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 8 } } },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'iteration' } },
          y: { title: { display: true, text: 'energy' } }
        }
      }
    });
  }

  diagLine(`diag-grad-${key}`,   r.grad_norm_curve    || [], '‖x‖',   '#f59e0b');
  diagLine(`diag-cond-${key}`,   r.cond_curve         || [], 'κ(S)',   '#ef4444', true);
  diagLine(`diag-weight-${key}`, r.weight_norm_curve  || [], '‖w‖',   '#a855f7');
  diagLine(`diag-eerr-${key}`,   r.energy_error_curve || [], 'σ/√n',  '#06b6d4');
}

// ── Build everything ──────────────────────────────────────────────────────
buildOverview();
nhKeys.forEach(key => buildNHPanel(key));
</script>
</body>
</html>
"""


def generate_report(results_dir: Path, output_path: Path):
    runs = load_results(results_dir)
    if not runs:
        print("No results found.")
        return

    print(f"Loaded {len(runs)} runs.")

    runs_json = json.dumps(runs, indent=None, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__RUNS_JSON__", runs_json)

    output_path.write_text(html, encoding="utf-8")
    print(f"Report written → {output_path}  ({output_path.stat().st_size // 1024} KB)")
    print(f"Open with:  open {output_path}  (or drag into browser)")


def main():
    p = argparse.ArgumentParser(description="Generate HTML benchmark report")
    p.add_argument("--results", default="results/", help="Results directory")
    p.add_argument("--output", default="report.html", help="Output HTML file")
    args = p.parse_args()
    generate_report(Path(args.results), Path(args.output))


if __name__ == "__main__":
    main()
