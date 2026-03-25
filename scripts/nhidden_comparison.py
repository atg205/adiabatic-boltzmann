"""
Generate a self-contained HTML report comparing the effect of n_hidden on VMC accuracy.

Usage:
    python nhidden_comparison.py                          # reads results/, writes nhidden_report.html
    python nhidden_comparison.py --results path/to/results --output nhidden_report.html
"""

import json
import argparse
from pathlib import Path


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
        model = config.get("model", "1d")
        size = int(config["size"])
        n_visible = size if model == "1d" else size * size
        n_hidden = int(config["n_hidden"] or size)
        alpha = round(n_hidden / n_visible, 4)

        records.append(
            {
                "file": str(file),
                "model": model,
                "size": size,
                "n_visible": n_visible,
                "h": float(config["h"]),
                "rbm": config["rbm"],
                "n_hidden": n_hidden,
                "alpha": alpha,
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
                "energy_curve": [float(x) for x in history.get("energy", [])],
                "grad_norm_curve": [float(x) for x in history.get("grad_norm", [])],
            }
        )

    return sorted(records, key=lambda r: (r["model"], r["size"], r["alpha"], r["rel_error"]))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>n_hidden Scaling Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');
  :root {
    --bg:#0a0c10; --bg2:#0f1218; --bg3:#161b24;
    --border:#1e2633; --border2:#2a3444;
    --accent:#3b82f6; --accent2:#60a5fa;
    --green:#10b981; --yellow:#f59e0b; --red:#ef4444; --purple:#a855f7;
    --text:#e2e8f0; --text2:#94a3b8; --text3:#64748b;
    --mono:'JetBrains Mono',monospace; --display:'Syne',sans-serif;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.6;min-height:100vh}

  header{border-bottom:1px solid var(--border);padding:16px 32px;display:flex;align-items:center;gap:16px;background:var(--bg2);position:sticky;top:0;z-index:200;flex-wrap:wrap}
  header h1{font-family:var(--display);font-size:19px;font-weight:800;letter-spacing:-.5px;white-space:nowrap}
  header h1 span{color:var(--accent)}
  .run-count{font-size:11px;color:var(--text3);white-space:nowrap}

  .filter-bar{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;margin-left:auto}
  .fg{display:flex;flex-direction:column;gap:2px}
  .fg label{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--text3)}
  .fg select{background:var(--bg3);border:1px solid var(--border2);color:var(--text);font-family:var(--mono);font-size:11px;padding:4px 8px;border-radius:4px;cursor:pointer;min-width:90px}
  .fg select:focus{outline:1px solid var(--accent)}
  .filter-badge{display:none;background:rgba(59,130,246,.15);color:var(--accent2);font-size:10px;padding:3px 8px;border-radius:3px}
  #btn-reset{display:none;background:none;border:1px solid var(--border2);color:var(--text3);font-family:var(--mono);font-size:10px;padding:4px 10px;border-radius:4px;cursor:pointer}
  #btn-reset:hover{border-color:var(--accent);color:var(--accent2)}

  .layout{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 57px)}
  aside{border-right:1px solid var(--border);padding:14px 0;background:var(--bg2);overflow-y:auto;position:sticky;top:57px;height:calc(100vh - 57px)}
  .sidebar-label{font-size:9px;font-weight:700;letter-spacing:2px;color:var(--text3);text-transform:uppercase;padding:12px 14px 5px;display:block}
  .nav-btn{display:block;width:100%;text-align:left;padding:6px 14px;background:none;border:none;color:var(--text2);cursor:pointer;font-family:var(--mono);font-size:11px;border-left:2px solid transparent;transition:all .15s}
  .nav-btn:hover{background:var(--bg3);color:var(--text)}
  .nav-btn.active{color:var(--accent2);border-left-color:var(--accent);background:rgba(59,130,246,.06)}
  .nav-btn .badge{float:right;font-size:9px;color:var(--text3);background:var(--bg3);padding:1px 5px;border-radius:3px}
  .nav-btn.dimmed{opacity:.3}

  main{padding:26px 34px;overflow-y:auto;background:var(--bg)}
  .panel{display:none}.panel.active{display:block}

  .section-title{font-family:var(--display);font-size:16px;font-weight:800;margin-bottom:4px;letter-spacing:-.3px}
  .section-sub{color:var(--text3);font-size:11px;margin-bottom:18px}
  .filter-notice{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);border-radius:4px;padding:6px 12px;font-size:11px;color:var(--accent2);margin-bottom:14px;display:none}
  .filter-notice.on{display:block}

  .stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:9px;margin-bottom:20px}
  .stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:5px;padding:11px 13px}
  .stat-label{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px}
  .stat-value{font-size:18px;font-weight:700;font-family:var(--display)}
  .stat-sub{font-size:9px;color:var(--text3);margin-top:2px}
  .good{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}

  .charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px}
  .charts-grid.wide{grid-template-columns:1fr}
  .charts-grid.quad{grid-template-columns:1fr 1fr 1fr 1fr}
  .chart-card{background:var(--bg2);border:1px solid var(--border);border-radius:5px;padding:13px}
  .chart-card.span2{grid-column:span 2}
  .chart-title{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3);margin-bottom:9px}
  .chart-wrap{position:relative;height:200px}
  .chart-wrap.tall{height:270px}
  .chart-wrap.short{height:160px}

  .alpha-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px}

  .run-table-wrap{background:var(--bg2);border:1px solid var(--border);border-radius:5px;overflow:hidden;margin-bottom:18px}
  .run-table-header{padding:9px 13px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
  .run-table-header span{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3)}
  table{width:100%;border-collapse:collapse}
  th{background:var(--bg3);padding:6px 9px;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3);text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
  td{padding:5px 9px;border-bottom:1px solid var(--border);font-size:11px;color:var(--text2);white-space:nowrap}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:var(--bg3)}
  .pill{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600}
  .pill-green{background:rgba(16,185,129,.15);color:var(--green)}
  .pill-yellow{background:rgba(245,158,11,.15);color:var(--yellow)}
  .pill-red{background:rgba(239,68,68,.15);color:var(--red)}
  .pill-blue{background:rgba(59,130,246,.15);color:var(--accent2)}
  .pill-purple{background:rgba(168,85,247,.15);color:var(--purple)}
  .divider{border:none;border-top:1px solid var(--border);margin:22px 0}
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:var(--bg2)}
  ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
</head>
<body>
<header>
  <h1>n_hidden <span>Scaling</span></h1>
  <span class="run-count" id="run-count"></span>
  <div class="filter-bar">
    <div class="fg"><label>Model</label>
      <select id="f-model" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Sampler</label>
      <select id="f-sampler" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Method</label>
      <select id="f-method" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>h</label>
      <select id="f-h" onchange="applyFilters()"><option value="">All</option></select></div>
    <div class="fg"><label>Seed</label>
      <select id="f-seed" onchange="applyFilters()"><option value="">All</option></select></div>
    <span class="filter-badge" id="filter-badge"></span>
    <button id="btn-reset" onclick="resetFilters()">✕ Reset</button>
  </div>
</header>

<div class="layout">
  <aside>
    <span class="sidebar-label">Overview</span>
    <button class="nav-btn active" onclick="showPanel('overview')" id="btn-overview">All sizes</button>
    <span class="sidebar-label">By size</span>
    <div id="size-buttons"></div>
  </aside>
  <main>
    <div class="panel active" id="panel-overview">
      <div class="section-title">n_hidden Scaling Overview</div>
      <div class="section-sub" id="ov-sub"></div>
      <div class="filter-notice" id="ov-notice"></div>
      <div class="stats-row" id="ov-stats"></div>
      <div class="charts-grid wide" style="margin-bottom:18px">
        <div class="chart-card">
          <div class="chart-title">Error vs α (n_hidden / n_visible) — best per (size, sampler, α)</div>
          <div class="chart-wrap tall"><canvas id="ov-alpha-err"></canvas></div>
        </div>
      </div>
      <div class="charts-grid">
        <div class="chart-card">
          <div class="chart-title">Error vs α — mean across seeds &amp; lr</div>
          <div class="chart-wrap"><canvas id="ov-alpha-mean"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Relative gain: error(α=0.25) / error(α=1.0) per size</div>
          <div class="chart-wrap"><canvas id="ov-gain"></canvas></div>
        </div>
      </div>
    </div>
    <div id="size-panels"></div>
  </main>
</div>

<script>
const ALL_RUNS = __RUNS_JSON__;

Chart.defaults.color='#64748b'; Chart.defaults.borderColor='#1e2633';
Chart.defaults.font.family="'JetBrains Mono',monospace"; Chart.defaults.font.size=10;
// Alpha-indexed palette: darker = fewer hidden units, brighter = more
const ALPHA_PAL=['#3b5bdb','#1971c2','#0ea5e9','#06b6d4'];
const SIZE_PAL=['#3b82f6','#10b981','#f59e0b','#ef4444','#a855f7','#06b6d4'];
const SM_PAL=['#3b82f6','#10b981','#f59e0b','#ef4444','#a855f7','#06b6d4','#f97316'];

const $=id=>document.getElementById(id);
const fmtP=v=>typeof v==='number'?v.toFixed(3)+'%':'—';
const fmtF=(v,d=4)=>typeof v==='number'?v.toFixed(d):'—';
const fmtE=v=>typeof v==='number'?v.toExponential(2):'—';
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;
const errCls=p=>p<1?'good':p<5?'warn':'bad';
const pillCls=p=>p<1?'pill-green':p<5?'pill-yellow':'pill-red';
const smKey=r=>`${r.sampler}/${r.sampling_method}`;
const sizeKey=r=>`${r.model}_sz${r.size}`;
const sizeLabel=r=>r.model==='2d'?`2D L=${r.size} (N=${r.n_visible})`:  `1D N=${r.size}`;

function mkChart(id,type,data,opts={}){
  const ex=Chart.getChart(id);if(ex)ex.destroy();
  const el=$(id);if(!el)return;
  return new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{boxWidth:9,padding:8}}},...opts}});
}
function lineds(label,pts,color,dash=[]){
  return{label,data:pts,borderColor:color,backgroundColor:color+'22',
    borderWidth:1.8,borderDash:dash,pointRadius:3,tension:.3};
}

// ── unique values ─────────────────────────────────────────────────────────
const allAlphas=[...new Set(ALL_RUNS.map(r=>r.alpha))].sort((a,b)=>a-b);
const allSizes=[...new Set(ALL_RUNS.map(sizeKey))].sort();
const allSizeLabels={};
ALL_RUNS.forEach(r=>{allSizeLabels[sizeKey(r)]=sizeLabel(r)});

// ── filter state ─────────────────────────────────────────────────────────
let ACTIVE=ALL_RUNS.slice();

function applyFilters(){
  const fs={
    model:$('f-model').value,
    sampler:$('f-sampler').value,
    sampling_method:$('f-method').value,
    h:$('f-h').value,
    seed:$('f-seed').value,
  };
  ACTIVE=ALL_RUNS.filter(r=>
    (!fs.model||r.model===fs.model)&&
    (!fs.sampler||r.sampler===fs.sampler)&&
    (!fs.sampling_method||r.sampling_method===fs.sampling_method)&&
    (!fs.h||String(r.h)===fs.h)&&
    (!fs.seed||String(r.seed)===fs.seed)
  );
  const nActive=Object.values(fs).filter(Boolean).length;
  const badge=$('filter-badge'),reset=$('btn-reset');
  if(nActive){
    badge.textContent=`${nActive} filter${nActive>1?'s':''} · ${ACTIVE.length} runs`;
    badge.style.display='inline-block'; reset.style.display='inline-block';
  } else {
    badge.style.display='none'; reset.style.display='none';
  }
  document.querySelectorAll('.nav-btn[data-k]').forEach(btn=>{
    btn.classList.toggle('dimmed',!ACTIVE.some(r=>sizeKey(r)===btn.dataset.k));
  });
  rebuildOverview();
  allSizes.forEach(k=>rebuildSizeCharts(k));
}

function resetFilters(){
  ['f-model','f-sampler','f-method','f-h','f-seed'].forEach(id=>$(id).value='');
  applyFilters();
}

function popSel(id,vals){
  const el=$(id);
  [...new Set(vals)].sort().forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)});
}
popSel('f-model',   ALL_RUNS.map(r=>r.model));
popSel('f-sampler', ALL_RUNS.map(r=>r.sampler));
popSel('f-method',  ALL_RUNS.map(r=>r.sampling_method));
popSel('f-h',       ALL_RUNS.map(r=>String(r.h)));
popSel('f-seed',    ALL_RUNS.map(r=>String(r.seed)));

$('run-count').textContent=`${ALL_RUNS.length} runs · ${allAlphas.length} α values · ${allSizes.length} sizes`;

// sidebar
allSizes.forEach(key=>{
  const runs=ALL_RUNS.filter(r=>sizeKey(r)===key);
  const alphas=[...new Set(runs.map(r=>r.alpha))].sort((a,b)=>a-b);
  const btn=document.createElement('button');
  btn.className='nav-btn'; btn.id='btn-'+key; btn.dataset.k=key;
  btn.onclick=()=>showPanel('sz-'+key);
  btn.innerHTML=`${allSizeLabels[key]}<span class="badge">${runs.length}</span>`;
  $('size-buttons').appendChild(btn);
});

function showPanel(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  const p=$('panel-'+id);if(p)p.classList.add('active');
  const b=$('btn-'+id);if(b)b.classList.add('active');
}

// ── overview ─────────────────────────────────────────────────────────────
function rebuildOverview(){
  const runs=ACTIVE;
  if(!runs.length){$('ov-stats').innerHTML=`<div class="stat-card"><div class="stat-value bad">No runs</div></div>`;return;}

  const errs=runs.map(r=>r.rel_error);
  const best=Math.min(...errs),worst=Math.max(...errs);
  const alphas=[...new Set(runs.map(r=>r.alpha))].sort((a,b)=>a-b);
  const sizes=[...new Set(runs.map(sizeKey))];

  const notice=$('ov-notice');
  if(ACTIVE.length<ALL_RUNS.length){notice.textContent=`Filtered: ${ACTIVE.length}/${ALL_RUNS.length} runs`;notice.classList.add('on')}
  else notice.classList.remove('on');

  $('ov-sub').textContent=`${runs.length} runs · α ∈ {${alphas.join(', ')}} · ${sizes.length} sizes`;
  $('ov-stats').innerHTML=`
    <div class="stat-card"><div class="stat-label">Runs</div><div class="stat-value">${runs.length}</div></div>
    <div class="stat-card"><div class="stat-label">α values</div><div class="stat-value">${alphas.length}</div></div>
    <div class="stat-card"><div class="stat-label">Best Error</div><div class="stat-value ${errCls(best)}">${fmtP(best)}</div><div class="stat-sub">α=${runs.find(r=>r.rel_error===best)?.alpha}</div></div>
    <div class="stat-card"><div class="stat-label">Worst Error</div><div class="stat-value ${errCls(worst)}">${fmtP(worst)}</div><div class="stat-sub">α=${runs.find(r=>r.rel_error===worst)?.alpha}</div></div>
    <div class="stat-card"><div class="stat-label">Sizes</div><div class="stat-value" style="font-size:13px">${sizes.map(k=>allSizeLabels[k]).join(', ')}</div></div>
  `;

  // Error vs alpha — best per (size×sampler, alpha)
  const smKeys=[...new Set(runs.map(r=>`${sizeKey(r)} / ${smKey(r)}`))].sort();
  const datasetsB=smKeys.map((sk,i)=>{
    const sub=runs.filter(r=>`${sizeKey(r)} / ${smKey(r)}`===sk);
    return lineds(sk, alphas.map(a=>{
      const s=sub.filter(r=>r.alpha===a);
      return s.length?{x:a,y:Math.min(...s.map(r=>r.rel_error))}:null;
    }).filter(Boolean), SIZE_PAL[i%SIZE_PAL.length]);
  });
  mkChart('ov-alpha-err','line',{datasets:datasetsB},{
    scales:{x:{type:'linear',title:{display:true,text:'α = n_hidden / n_visible'}},
            y:{title:{display:true,text:'Best rel. error (%)'}}}});

  // Error vs alpha — mean
  const datasetsMean=smKeys.map((sk,i)=>{
    const sub=runs.filter(r=>`${sizeKey(r)} / ${smKey(r)}`===sk);
    return lineds(sk, alphas.map(a=>{
      const s=sub.filter(r=>r.alpha===a);
      return s.length?{x:a,y:mean(s.map(r=>r.rel_error))}:null;
    }).filter(Boolean), SIZE_PAL[i%SIZE_PAL.length]);
  });
  mkChart('ov-alpha-mean','line',{datasets:datasetsMean},{
    scales:{x:{type:'linear',title:{display:true,text:'α = n_hidden / n_visible'}},
            y:{title:{display:true,text:'Mean rel. error (%)'}}}});

  // Relative gain: error(min_alpha) / error(max_alpha) — how much does increasing alpha help?
  const gainLabels=[];const gainVals=[];
  smKeys.forEach(sk=>{
    const sub=runs.filter(r=>`${sizeKey(r)} / ${smKey(r)}`===sk);
    const loAlpha=Math.min(...alphas),hiAlpha=Math.max(...alphas);
    const lo=sub.filter(r=>r.alpha===loAlpha);
    const hi=sub.filter(r=>r.alpha===hiAlpha);
    if(!lo.length||!hi.length)return;
    const gain=mean(lo.map(r=>r.rel_error))/mean(hi.map(r=>r.rel_error));
    gainLabels.push(sk); gainVals.push(+gain.toFixed(3));
  });
  mkChart('ov-gain','bar',{
    labels:gainLabels,
    datasets:[{label:'error(α_min)/error(α_max)',data:gainVals,
      backgroundColor:gainVals.map((v,i)=>SIZE_PAL[i%SIZE_PAL.length]+'aa'),
      borderColor:gainVals.map((v,i)=>SIZE_PAL[i%SIZE_PAL.length]),borderWidth:1}]
  },{scales:{y:{title:{display:true,text:'gain ratio (>1 means more hidden helps)'}}}});
}

// ── per-size panel ────────────────────────────────────────────────────────
function buildSizePanel(key){
  const allForKey=ALL_RUNS.filter(r=>sizeKey(r)===key);
  const alphas=[...new Set(allForKey.map(r=>r.alpha))].sort((a,b)=>a-b);
  const smKeys=[...new Set(allForKey.map(smKey))].sort();

  const el=document.createElement('div');
  el.className='panel'; el.id='panel-sz-'+key;

  // Build alpha convergence cards HTML
  const convCards=alphas.map((a,i)=>`
    <div class="chart-card">
      <div class="chart-title">α=${a} (n_hid=${allForKey.find(r=>r.alpha===a)?.n_hidden})</div>
      <div class="chart-wrap short"><canvas id="conv-${key}-a${i}"></canvas></div>
    </div>`).join('');

  el.innerHTML=`
    <div class="section-title">${allSizeLabels[key]}</div>
    <div class="section-sub">n_visible=${allForKey[0]?.n_visible} &nbsp;·&nbsp; α ∈ {${alphas.join(', ')}} &nbsp;·&nbsp; <span id="szrc-${key}">${allForKey.length}</span> runs</div>
    <div class="filter-notice" id="szn-${key}"></div>
    <div class="stats-row" id="szs-${key}"></div>

    <div class="charts-grid wide">
      <div class="chart-card">
        <div class="chart-title">Error vs α — by sampler/method (best and mean)</div>
        <div class="chart-wrap tall"><canvas id="sz-alpha-${key}"></canvas></div>
      </div>
    </div>

    <div class="section-title" style="font-size:12px;margin:18px 0 9px">Convergence by α</div>
    <div class="alpha-grid">${convCards}</div>

    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">Best convergence per α (best sampler)</div>
        <div class="chart-wrap tall"><canvas id="sz-bestconv-${key}"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Gradient norm ‖x‖ at convergence vs α</div>
        <div class="chart-wrap tall"><canvas id="sz-grad-${key}"></canvas></div>
      </div>
    </div>

    <hr class="divider">
    <div class="run-table-wrap">
      <div class="run-table-header"><span>All Runs</span><span id="sztc-${key}"></span></div>
      <div style="overflow-x:auto">
        <table id="tbl-${key}"><thead><tr>
          <th>α</th><th>n_hidden</th><th>rel err%</th><th>abs err</th><th>final E</th>
          <th>sampler</th><th>method</th><th>lr</th><th>reg</th><th>seed</th>
        </tr></thead><tbody id="tb-${key}"></tbody></table>
      </div>
    </div>
  `;
  $('size-panels').appendChild(el);
  setTimeout(()=>rebuildSizeCharts(key),0);
}

function rebuildSizeCharts(key){
  const allForKey=ALL_RUNS.filter(r=>sizeKey(r)===key);
  const runs=ACTIVE.filter(r=>sizeKey(r)===key).sort((a,b)=>a.alpha-b.alpha||a.rel_error-b.rel_error);
  const exact=allForKey[0]?.exact_energy;
  const alphas=[...new Set(allForKey.map(r=>r.alpha))].sort((a,b)=>a-b);
  const smKeys=[...new Set(runs.map(smKey))].sort();

  const notice=$(`szn-${key}`);
  if(runs.length<allForKey.length){notice.textContent=`Showing ${runs.length}/${allForKey.length} runs`;notice.classList.add('on')}
  else notice.classList.remove('on');
  $(`szrc-${key}`).textContent=runs.length;
  $(`sztc-${key}`).textContent=`${runs.length} experiments`;

  if(!runs.length){$(`szs-${key}`).innerHTML=`<div class="stat-card"><div class="stat-value bad">No runs match</div></div>`;return;}

  // Stats: best error per alpha
  const bestPerAlpha=alphas.map(a=>{
    const s=runs.filter(r=>r.alpha===a);
    return s.length?Math.min(...s.map(r=>r.rel_error)):null;
  });
  const bestAlphaIdx=bestPerAlpha.reduce((bi,v,i)=>v!==null&&(bestPerAlpha[bi]===null||v<bestPerAlpha[bi])?i:bi,0);
  $(`szs-${key}`).innerHTML=`
    <div class="stat-card"><div class="stat-label">Exact E</div><div class="stat-value" style="font-size:14px">${fmtF(exact,4)}</div></div>
    <div class="stat-card"><div class="stat-label">Best α</div><div class="stat-value" style="font-size:16px">${alphas[bestAlphaIdx]}</div><div class="stat-sub">err=${fmtP(bestPerAlpha[bestAlphaIdx])}</div></div>
    ${alphas.map((a,i)=>`<div class="stat-card"><div class="stat-label">α=${a}</div><div class="stat-value ${errCls(bestPerAlpha[i]||99)}">${fmtP(bestPerAlpha[i])}</div><div class="stat-sub">n_hid=${runs.find(r=>r.alpha===a)?.n_hidden||'—'}</div></div>`).join('')}
  `;

  // Error vs alpha (bar: best + mean per sampler)
  const datasets=[];
  smKeys.forEach((sk,i)=>{
    const sub=runs.filter(r=>smKey(r)===sk);
    datasets.push({label:`${sk} best`,data:alphas.map(a=>{const s=sub.filter(r=>r.alpha===a);return s.length?Math.min(...s.map(r=>r.rel_error)):null}),
      borderColor:SM_PAL[i%SM_PAL.length],backgroundColor:SM_PAL[i%SM_PAL.length]+'aa',borderWidth:1,type:'bar'});
    datasets.push({label:`${sk} mean`,data:alphas.map(a=>{const s=sub.filter(r=>r.alpha===a);return s.length?mean(s.map(r=>r.rel_error)):null}),
      borderColor:SM_PAL[i%SM_PAL.length],backgroundColor:'transparent',borderWidth:2,type:'line',
      pointRadius:5,spanGaps:true});
  });
  mkChart(`sz-alpha-${key}`,'bar',{labels:alphas.map(a=>`α=${a}`),datasets},{
    scales:{y:{title:{display:true,text:'Rel. error (%)'}}}});

  // Best convergence per alpha (best run per alpha)
  const bestRuns=alphas.map(a=>runs.filter(r=>r.alpha===a).sort((x,y)=>x.rel_error-y.rel_error)[0]).filter(Boolean);
  const maxL=Math.max(1,...bestRuns.map(r=>r.energy_curve.length));
  mkChart(`sz-bestconv-${key}`,'line',{datasets:[
    ...bestRuns.map((r,i)=>({
      label:`α=${r.alpha} (${smKey(r)}, err=${fmtP(r.rel_error)})`,
      data:r.energy_curve.map((v,x)=>({x,y:v})),
      borderColor:ALPHA_PAL[i%ALPHA_PAL.length],backgroundColor:ALPHA_PAL[i%ALPHA_PAL.length]+'22',
      borderWidth:1.8,pointRadius:0,tension:.3
    })),
    {label:`Exact: ${fmtF(exact,4)}`,data:[{x:0,y:exact},{x:maxL-1,y:exact}],
      borderColor:'#ef444488',borderDash:[6,3],borderWidth:1.5,pointRadius:0}
  ]},{scales:{x:{type:'linear',title:{display:true,text:'iteration'}},y:{title:{display:true,text:'energy'}}}});

  // Per-alpha convergence charts (one per alpha, all samplers)
  alphas.forEach((a,i)=>{
    const id=`conv-${key}-a${i}`;
    const sub=runs.filter(r=>r.alpha===a).sort((x,y)=>x.rel_error-y.rel_error);
    const mxL=Math.max(1,...sub.map(r=>r.energy_curve.length));
    mkChart(id,'line',{datasets:[
      ...sub.slice(0,4).map((r,j)=>({
        label:`${smKey(r)} err=${fmtP(r.rel_error)}`,
        data:r.energy_curve.map((v,x)=>({x,y:v})),
        borderColor:SM_PAL[j%SM_PAL.length],backgroundColor:'transparent',
        borderWidth:1.5,pointRadius:0,tension:.3
      })),
      {label:`Exact`,data:[{x:0,y:exact},{x:mxL-1,y:exact}],
        borderColor:'#ef444488',borderDash:[4,2],borderWidth:1,pointRadius:0}
    ]},{scales:{x:{type:'linear'},y:{title:{display:true,text:'energy'}}},
      plugins:{legend:{labels:{boxWidth:7,font:{size:9}}}}});
  });

  // Gradient norm at last iteration vs alpha
  mkChart(`sz-grad-${key}`,'line',{datasets:smKeys.map((sk,i)=>{
    const sub=runs.filter(r=>smKey(r)===sk);
    return lineds(sk,alphas.map(a=>{
      const s=sub.filter(r=>r.alpha===a);
      if(!s.length)return null;
      const lastGrads=s.map(r=>{const g=r.grad_norm_curve;return g&&g.length?g[g.length-1]:null}).filter(v=>v!==null);
      return lastGrads.length?{x:a,y:mean(lastGrads)}:null;
    }).filter(Boolean),SM_PAL[i%SM_PAL.length]);
  })},{scales:{x:{type:'linear',title:{display:true,text:'α'}},y:{title:{display:true,text:'Mean ‖grad‖ at last iter'}}}});

  // Table
  const tbody=$(`tb-${key}`);
  tbody.innerHTML='';
  runs.forEach(r=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td><span class="pill pill-purple">α=${r.alpha}</span></td>
      <td>${r.n_hidden}</td>
      <td><span class="pill ${pillCls(r.rel_error)}">${fmtP(r.rel_error)}</span></td>
      <td>${fmtF(r.abs_error,5)}</td><td>${fmtF(r.final_energy,4)}</td>
      <td><span class="pill pill-blue">${r.sampler}</span></td>
      <td>${r.sampling_method}</td>
      <td>${r.lr}</td><td>${fmtE(r.reg)}</td><td>${r.seed}</td>
    `;
    tbody.appendChild(tr);
  });
}

rebuildOverview();
allSizes.forEach(k=>buildSizePanel(k));
</script>
</body>
</html>
"""


def generate_report(results_dir: Path, output_path: Path):
    runs = load_results(results_dir)
    if not runs:
        print("No results found.")
        return

    # Warn if no alpha variation found
    alphas = {r["alpha"] for r in runs}
    if len(alphas) <= 1:
        print(f"  [warn] Only {len(alphas)} distinct alpha value(s) found: {alphas}")
        print("         Run the n_hidden sweep first for meaningful comparison.")

    print(f"Loaded {len(runs)} runs across {len(alphas)} alpha values: {sorted(alphas)}")
    runs_json = json.dumps(runs, indent=None, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__RUNS_JSON__", runs_json)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written → {output_path}  ({output_path.stat().st_size // 1024} KB)")
    print(f"Open with:  open {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate n_hidden scaling HTML report")
    p.add_argument("--results", default="results/", help="Results directory")
    p.add_argument("--output", default="nhidden_report.html", help="Output HTML file")
    args = p.parse_args()
    generate_report(Path(args.results), Path(args.output))


if __name__ == "__main__":
    main()
