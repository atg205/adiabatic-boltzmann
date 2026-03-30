#!/usr/bin/env python3
"""
Comprehensive hyperparameter analysis of VMC+RBM results.
Run from repo root:  python scripts/analyze_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "plots" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load all result files ──────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    records = []
    for path in RESULTS_DIR.rglob("*.json"):
        if "plots" in path.parts:
            continue
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue
        cfg = d.get("config", {})
        hist = d.get("history", {})
        energy_curve = hist.get("energy", [])
        error_curve  = hist.get("error", [])

        exact = d.get("exact_energy")
        final = d.get("final_energy")
        rel_err = abs(d.get("error", 0)) / abs(exact) * 100 if exact else None

        record = {
            # identity
            "path":             str(path),
            "model":            cfg.get("model"),
            "size":             cfg.get("size"),
            "h":                cfg.get("h"),
            "rbm":              cfg.get("rbm"),
            "n_hidden":         cfg.get("n_hidden"),
            "sampler":          cfg.get("sampler"),
            "method":           cfg.get("sampling_method"),
            "iterations":       cfg.get("iterations"),
            "lr":               cfg.get("learning_rate"),
            "reg":              cfg.get("regularization"),
            "n_samples":        cfg.get("n_samples"),
            "seed":             cfg.get("seed"),
            # SBM-specific
            "sb_mode":          cfg.get("sb_mode"),
            "sb_heated":        cfg.get("sb_heated"),
            "sb_max_steps":     cfg.get("sb_max_steps"),
            # CEM
            "cem":              bool(cfg.get("cem", False)),
            # outcomes
            "final_energy":     final,
            "exact_energy":     exact,
            "abs_error":        d.get("error"),
            "rel_error_pct":    rel_err,
            "n_iter_actual":    len(energy_curve),
            # convergence
            "energy_curve":     energy_curve,
            "error_curve":      error_curve,
        }
        records.append(record)

    df = pd.DataFrame(records)
    # drop runs with missing outcomes
    df = df.dropna(subset=["rel_error_pct", "final_energy"])
    return df


# ── plotting helpers ───────────────────────────────────────────────────────────

METHOD_COLORS = {
    "pegasus":            "#1f77b4",
    "zephyr":             "#ff7f0e",
    "velox":              "#2ca02c",
    "simulated_annealing":"#d62728",
    "metropolis":         "#9467bd",
    "sbm":                "#8c564b",
}

def method_color(m: str) -> str:
    return METHOD_COLORS.get(m, "#888888")


def savefig(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── Figure 1: method leaderboard (box plot of rel error) ──────────────────────

def fig_method_leaderboard(df: pd.DataFrame) -> None:
    methods = (df.groupby("method")["rel_error_pct"]
                 .median()
                 .sort_values()
                 .index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sampling method comparison — relative energy error (%)",
                 fontsize=13, fontweight="bold")

    # left: box plot
    ax = axes[0]
    data = [df[df.method == m]["rel_error_pct"].dropna().values for m in methods]
    bp = ax.boxplot(data, patch_artist=True, vert=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(method_color(m))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("rel. error (%)")
    ax.set_title("Distribution of rel. error per method")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # right: mean ± std bar chart
    ax = axes[1]
    stats = (df.groupby("method")["rel_error_pct"]
               .agg(["mean", "std", "count"])
               .reindex(methods))
    xs = range(len(methods))
    bars = ax.bar(xs, stats["mean"],
                  yerr=stats["std"], capsize=4,
                  color=[method_color(m) for m in methods], alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("mean rel. error (%)")
    ax.set_title("Mean ± std rel. error per method")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["std"] * 0.05 + 0.05,
                f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    savefig(fig, "1_method_leaderboard.png")


# ── Figure 2: LR × regularization interaction ─────────────────────────────────

def fig_lr_reg(df: pd.DataFrame) -> None:
    methods = sorted(df.method.unique())
    lrs  = sorted(df.lr.unique())
    regs = sorted(df.reg.unique())

    fig, axes = plt.subplots(len(methods), 2, figsize=(12, 3 * len(methods)),
                             sharey="row")
    fig.suptitle("Learning rate × regularization effect on rel. error",
                 fontsize=13, fontweight="bold")

    for row_i, method in enumerate(methods):
        sub = df[df.method == method]
        for col_i, xlabel in enumerate(["lr", "reg"]):
            ax = axes[row_i, col_i]
            xvals = lrs if col_i == 0 else regs
            medians = [sub[sub[xlabel] == x]["rel_error_pct"].median() for x in xvals]
            ax.bar([str(x) for x in xvals], medians,
                   color=method_color(method), alpha=0.8)
            ax.set_title(f"{method} — by {xlabel}", fontsize=9)
            ax.set_ylabel("median rel. error (%)" if col_i == 0 else "")
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    savefig(fig, "2_lr_reg_effect.png")


# ── Figure 3: error vs h (transverse field) per method ────────────────────────

def fig_vs_h(df: pd.DataFrame) -> None:
    methods = sorted(df.method.unique())
    hs = sorted(df.h.unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Rel. error vs transverse field h (all sizes averaged)",
                 fontsize=13, fontweight="bold")

    for method in methods:
        sub = df[df.method == method]
        means = [sub[sub.h == h]["rel_error_pct"].mean() for h in hs]
        stds  = [sub[sub.h == h]["rel_error_pct"].std()  for h in hs]
        ax.errorbar(hs, means, yerr=stds,
                    label=method, color=method_color(method),
                    marker="o", linewidth=2, capsize=4)

    ax.set_xlabel("h (transverse field strength)", fontsize=11)
    ax.set_ylabel("mean rel. error (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hs)
    plt.tight_layout()
    savefig(fig, "3_error_vs_h.png")


# ── Figure 4: error vs system size ────────────────────────────────────────────

def fig_vs_size(df: pd.DataFrame) -> None:
    methods = sorted(df.method.unique())

    # separate 1d and 2d
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rel. error vs system size", fontsize=13, fontweight="bold")

    for col_i, model in enumerate(["1d", "2d"]):
        ax = axes[col_i]
        sub_model = df[df.model == model]
        sizes = sorted(sub_model["size"].unique())
        for method in methods:
            sub = sub_model[sub_model.method == method]
            if sub.empty:
                continue
            means = [sub[sub["size"] == s]["rel_error_pct"].mean() for s in sizes]
            ax.plot(sizes, means, marker="o", label=method,
                    color=method_color(method), linewidth=2)
        ax.set_title(f"{model.upper()} model", fontsize=11, fontweight="bold")
        ax.set_xlabel("system size (N)", fontsize=10)
        ax.set_ylabel("mean rel. error (%)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, "4_error_vs_size.png")


# ── Figure 5: convergence curves per method ───────────────────────────────────

def fig_convergence(df: pd.DataFrame) -> None:
    methods = sorted(df.method.unique())
    # pick a fixed, common case: 1d, size=16, h=1.0, lr=0.1, reg=0.001
    # fall back gracefully
    candidates = df[(df.model == "1d") & (df.h == 1.0) & (df.lr == 0.1)]
    if candidates.empty:
        candidates = df[df.lr == 0.1]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Convergence curves (1D, h=1.0, lr=0.1, one seed per method)",
                 fontsize=13, fontweight="bold")

    plotted = set()
    for _, row in candidates.sort_values("size").iterrows():
        m = row["method"]
        if m in plotted:
            continue
        curve = row["energy_curve"]
        if not curve:
            continue
        exact = row["exact_energy"]
        rel_curve = [abs(e - exact) / abs(exact) * 100 for e in curve]
        ax.plot(rel_curve, color=method_color(m), linewidth=2,
                label=f"{m} (N={row['size']}, seed={row['seed']})")
        plotted.add(m)

    ax.set_xlabel("iteration", fontsize=11)
    ax.set_ylabel("rel. error (%)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    savefig(fig, "5_convergence_curves.png")


# ── Figure 6: SBM-specific analysis ───────────────────────────────────────────

def fig_sbm(df: pd.DataFrame) -> None:
    sbm = df[df.method == "sbm"].copy()
    if sbm.empty:
        print("  no SBM runs found, skipping SBM figure")
        return

    sbm["sb_mode"]     = sbm["sb_mode"].fillna("unknown")
    sbm["sb_heated"]   = sbm["sb_heated"].astype(str)
    sbm["sb_max_steps"] = sbm["sb_max_steps"].fillna(0).astype(int)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("SBM hyperparameter analysis", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. box: mode
    ax = fig.add_subplot(gs[0, 0])
    modes = sorted(sbm["sb_mode"].unique())
    ax.boxplot([sbm[sbm.sb_mode == m]["rel_error_pct"].dropna().values for m in modes],
               patch_artist=True, medianprops=dict(color="black"))
    ax.set_xticks(range(1, len(modes)+1)); ax.set_xticklabels(modes)
    ax.set_title("Mode (discrete vs ballistic)"); ax.set_ylabel("rel. error (%)")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. box: heated
    ax = fig.add_subplot(gs[0, 1])
    heateds = sorted(sbm["sb_heated"].unique())
    ax.boxplot([sbm[sbm.sb_heated == h]["rel_error_pct"].dropna().values for h in heateds],
               patch_artist=True, medianprops=dict(color="black"))
    ax.set_xticks(range(1, len(heateds)+1)); ax.set_xticklabels(heateds)
    ax.set_title("Heated"); ax.set_ylabel("rel. error (%)")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. error vs max_steps
    ax = fig.add_subplot(gs[0, 2])
    steps_vals = sorted(sbm[sbm.sb_max_steps > 0]["sb_max_steps"].unique())
    if steps_vals:
        for mode in modes:
            sub = sbm[sbm.sb_mode == mode]
            means = [sub[sub.sb_max_steps == s]["rel_error_pct"].mean() for s in steps_vals]
            ax.plot(steps_vals, means, marker="o", label=mode, linewidth=2)
        ax.set_xscale("log"); ax.set_xlabel("max_steps"); ax.set_ylabel("mean rel. error (%)")
        ax.set_title("Error vs max_steps"); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    # 4. mode × heated heatmap (if enough data)
    ax = fig.add_subplot(gs[1, 0])
    pivot = sbm.groupby(["sb_mode", "sb_heated"])["rel_error_pct"].mean().unstack(fill_value=np.nan)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    ax.set_title("mode × heated — mean rel. error (%)")
    plt.colorbar(im, ax=ax)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    # 5. CEM effect within SBM
    ax = fig.add_subplot(gs[1, 1])
    cem_groups = sbm.groupby("cem")["rel_error_pct"].agg(["mean", "std", "count"])
    labels = ["no CEM" if not c else "CEM" for c in cem_groups.index]
    ax.bar(labels, cem_groups["mean"], yerr=cem_groups["std"],
           color=["#1f77b4", "#ff7f0e"][:len(labels)], alpha=0.8, capsize=5)
    ax.set_ylabel("mean rel. error (%)"); ax.set_title("CEM effect (SBM)")
    ax.grid(True, alpha=0.3, axis="y")

    # 6. convergence: sbm vs best classical
    ax = fig.add_subplot(gs[1, 2])
    for method in ["sbm", "simulated_annealing", "metropolis"]:
        sub = df[(df.method == method) & (df.lr == 0.1)]
        if sub.empty:
            continue
        # average convergence curve (pad to max length)
        curves = [r for r in sub["energy_curve"].values if r]
        exacts = sub[sub["energy_curve"].apply(bool)]["exact_energy"].values
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        exacts_b = exacts[:len(curves)]
        rel_curves = np.abs(padded - exacts_b[:, None]) / np.abs(exacts_b[:, None]) * 100
        mean_curve = rel_curves.mean(axis=0)
        ax.plot(mean_curve, color=method_color(method), linewidth=2, label=method)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("mean rel. error (%)")
    ax.set_title("SBM vs classical convergence"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    savefig(fig, "6_sbm_analysis.png")


# ── Figure 7: CEM effect across all methods ───────────────────────────────────

def fig_cem(df: pd.DataFrame) -> None:
    cem_df = df[df.cem == True]
    nocem_df = df[df.cem == False]
    if cem_df.empty:
        print("  no CEM runs, skipping CEM figure")
        return

    methods_with_cem = sorted(cem_df.method.unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("CEM effect — rel. error with vs without CEM",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(methods_with_cem))
    w = 0.35
    for i, (label, subset) in enumerate([("no CEM", nocem_df), ("CEM", cem_df)]):
        means = [subset[subset.method == m]["rel_error_pct"].mean()
                 for m in methods_with_cem]
        stds  = [subset[subset.method == m]["rel_error_pct"].std()
                 for m in methods_with_cem]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=4,
               label=label, alpha=0.8)

    ax.set_xticks(x + w / 2); ax.set_xticklabels(methods_with_cem, rotation=15, ha="right")
    ax.set_ylabel("mean rel. error (%)"); ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig(fig, "7_cem_effect.png")


# ── Figure 8: summary table ────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("COMPREHENSIVE SUMMARY — sorted by median rel. error (%)")
    print("=" * 100)

    grp = (df.groupby(["method", "lr", "reg", "cem"])
             .agg(
                 median_err=("rel_error_pct", "median"),
                 mean_err  =("rel_error_pct", "mean"),
                 std_err   =("rel_error_pct", "std"),
                 min_err   =("rel_error_pct", "min"),
                 n         =("rel_error_pct", "count"),
             )
             .reset_index()
             .sort_values("median_err"))

    print(f"{'method':<22} {'lr':>6} {'reg':>8} {'cem':>5} "
          f"{'median%':>9} {'mean%':>9} {'std':>7} {'min%':>7} {'n':>4}")
    print("-" * 100)
    for _, row in grp.iterrows():
        print(f"{row['method']:<22} {row['lr']:>6.3f} {row['reg']:>8.5f} "
              f"{str(row['cem']):>5} "
              f"{row['median_err']:>9.3f} {row['mean_err']:>9.3f} "
              f"{row['std_err']:>7.3f} {row['min_err']:>7.3f} {int(row['n']):>4}")
    print("=" * 100)

    # SBM sub-table
    sbm = df[df.method == "sbm"]
    if not sbm.empty:
        print("\nSBM sub-table — (mode, heated, max_steps):")
        print("-" * 70)
        sbm_grp = (sbm.groupby(["sb_mode", "sb_heated", "sb_max_steps"])
                      .agg(median_err=("rel_error_pct","median"),
                           mean_err  =("rel_error_pct","mean"),
                           n         =("rel_error_pct","count"))
                      .reset_index()
                      .sort_values("median_err"))
        print(sbm_grp.to_string(index=False))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading results...")
    df = load_results()
    print(f"  {len(df)} runs loaded, {df.method.nunique()} methods, "
          f"{df.model.nunique()} model types")

    print("\nGenerating figures...")
    fig_method_leaderboard(df)
    fig_lr_reg(df)
    fig_vs_h(df)
    fig_vs_size(df)
    fig_convergence(df)
    fig_sbm(df)
    fig_cem(df)
    print_summary_table(df)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
