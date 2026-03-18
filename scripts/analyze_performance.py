"""
Benchmark Results Analysis
==========================
Organized by (N, h) combination — each system size and field strength
gets its own section showing best/worst runs and hyperparameter sensitivity.

Usage:
    python analyze_performance.py
    python analyze_performance.py --results results/ --top 3 --no-plots
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


ROOT = Path("results")


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------


def load_results(root: Path) -> pd.DataFrame:
    records = []

    for file in root.rglob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print(f"  [warn] skipping malformed file: {file}")
            continue

        config = data["config"]
        history = data["history"]

        records.append(
            {
                # identity
                "file": str(file),
                "size": config["size"],
                "h": config["h"],
                "rbm": config["rbm"],
                "n_hidden": config["n_hidden"],
                "sampler": config["sampler"],
                "sampling_method": config["sampling_method"],
                # hyperparams
                "lr": config["learning_rate"],
                "reg": config["regularization"],
                "n_samples": config["n_samples"],
                "seed": config["seed"],
                # results
                "final_energy": data["final_energy"],
                "exact_energy": data["exact_energy"],
                "abs_error": data.get(
                    "error", abs(data["final_energy"] - data["exact_energy"])
                ),
                "rel_error": abs(data["final_energy"] - data["exact_energy"])
                / abs(data["exact_energy"])
                * 100,
                # curves
                "energy_curve": history.get("energy", []),
                "error_curve": history.get("error", []),
                "energy_error_curve": history.get("energy_error", []),
                "grad_norm_curve": history.get("grad_norm", []),
                "cond_curve": history.get("s_condition_number", []),
                "weight_norm_curve": history.get("weight_norm", []),
            }
        )

    if not records:
        raise FileNotFoundError(f"No result JSON files found under {root}")

    df = pd.DataFrame(records)
    df["size"] = df["size"].astype(int)
    df["n_samples"] = df["n_samples"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Per-(N, h) summary table
# ---------------------------------------------------------------------------


def print_nh_summary(df: pd.DataFrame, top_n: int = 3):
    """
    For every (N, h) pair print:
      - exact ground energy
      - number of runs
      - best / median / worst relative error across all runs
      - best hyperparameter configuration
    """
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY BY (N, h)")
    print("=" * 80)

    for (size, h), group in df.groupby(["size", "h"], sort=True):
        exact = group["exact_energy"].iloc[0]
        n_runs = len(group)
        best_err = group["rel_error"].min()
        med_err = group["rel_error"].median()
        worst = group["rel_error"].max()

        print(f"\n{'─' * 70}")
        print(
            f"  N = {size:>3d}   h = {h:.1f}   |   exact E = {exact:.6f}   |   runs = {n_runs}"
        )
        print(f"{'─' * 70}")
        print(
            f"  Relative error  →  best: {best_err:.3f}%   median: {med_err:.3f}%   worst: {worst:.3f}%"
        )

        # Best runs
        print(f"\n  Top {top_n} configurations:")
        print(
            f"  {'rank':<5} {'rel_err%':<10} {'abs_err':<12} {'final_E':<14} "
            f"{'lr':<8} {'reg':<8} {'n_samp':<8} {'n_hid':<6} {'seed':<5}"
        )
        print(
            f"  {'─' * 5} {'─' * 9} {'─' * 11} {'─' * 13} "
            f"{'─' * 7} {'─' * 7} {'─' * 7} {'─' * 5} {'─' * 4}"
        )

        top_rows = group.nsmallest(top_n, "rel_error")
        for rank, (_, row) in enumerate(top_rows.iterrows(), 1):
            print(
                f"  {rank:<5} {row.rel_error:<10.3f} {row.abs_error:<12.6f} "
                f"{row.final_energy:<14.6f} "
                f"{row.lr:<8} {row.reg:<8} {row.n_samples:<8} "
                f"{row.n_hidden:<6} {row.seed:<5}"
            )

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# 3. Hyperparameter sensitivity per (N, h)
# ---------------------------------------------------------------------------


def print_hyperparam_sensitivity(df: pd.DataFrame):
    """
    For each (N, h), show mean relative error grouped by each hyperparameter.
    Tells you which hyperparameter settings work best for each system.
    """
    print("\n" + "=" * 80)
    print("  HYPERPARAMETER SENSITIVITY BY (N, h)")
    print("=" * 80)

    hyperparams = {
        "n_samples": "Samples",
        "reg": "Regularization",
        "lr": "Learning Rate",
        "n_hidden": "Hidden Units",
    }

    for (size, h), group in df.groupby(["size", "h"], sort=True):
        print(f"\n  N={size}, h={h}  (exact E = {group['exact_energy'].iloc[0]:.4f})")

        for col, label in hyperparams.items():
            if group[col].nunique() < 2:
                continue
            pivot = (
                group.groupby(col)["rel_error"]
                .agg(["mean", "min", "count"])
                .rename(columns={"mean": "mean_%", "min": "best_%", "count": "runs"})
            )
            pivot.index.name = label
            print(f"\n    By {label}:")
            print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# 4. Convergence plots organized by (N, h)
# ---------------------------------------------------------------------------


def plot_convergence_by_nh(df: pd.DataFrame, top_n: int = 3):
    """
    One subplot per (N, h) combination showing the top_n convergence curves
    and a dashed line at exact energy.
    """
    nh_pairs = sorted(df.groupby(["size", "h"]).groups.keys())
    n_pairs = len(nh_pairs)

    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (size, h) in zip(axes, nh_pairs):
        group = df[(df["size"] == size) & (df["h"] == h)]
        exact = group["exact_energy"].iloc[0]
        top = group.nsmallest(top_n, "rel_error")

        for _, row in top.iterrows():
            curve = row["energy_curve"]
            if not curve:
                continue
            label = f"lr={row.lr} reg={row.reg} ns={row.n_samples}"
            ax.plot(curve, linewidth=1.2, label=label)

        ax.axhline(
            exact,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Exact: {exact:.3f}",
        )
        ax.set_title(f"N={size}, h={h}", fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)

    # Hide unused axes
    for ax in axes[n_pairs:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Convergence Curves — Top {top_n} Runs per (N, h)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Heatmap: best relative error over (N, h) grid
# ---------------------------------------------------------------------------


def plot_error_heatmap(df: pd.DataFrame):
    """
    2D heatmap: rows = N, cols = h, color = best relative error (%).
    Immediately shows which (N, h) regimes are hardest.
    """
    pivot = df.groupby(["size", "h"])["rel_error"].min().unstack("h")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best error
    im0 = axes[0].imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels([f"h={h}" for h in pivot.columns])
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels([f"N={n}" for n in pivot.index])
    axes[0].set_title("Best Relative Error (%) per (N, h)", fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label="Rel. error (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.2f}%", ha="center", va="center", fontsize=8)

    # Number of runs
    counts = df.groupby(["size", "h"])["rel_error"].count().unstack("h")
    im1 = axes[1].imshow(counts.values, aspect="auto", cmap="Blues")
    axes[1].set_xticks(range(len(counts.columns)))
    axes[1].set_xticklabels([f"h={h}" for h in counts.columns])
    axes[1].set_yticks(range(len(counts.index)))
    axes[1].set_yticklabels([f"N={n}" for n in counts.index])
    axes[1].set_title("Number of Runs per (N, h)", fontweight="bold")
    plt.colorbar(im1, ax=axes[1], label="# runs")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            axes[1].text(
                j,
                i,
                str(int(counts.values[i, j])),
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Diagnostics for a single run
# ---------------------------------------------------------------------------


def plot_diagnostics(row: pd.Series):
    """Full diagnostic panel for one run."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    title = (
        f"N={row['size']}, h={row['h']} | "
        f"lr={row['lr']}, reg={row['reg']}, ns={row['n_samples']} | "
        f"rel_err={row['rel_error']:.3f}%"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")

    def _plot(pos, curve_key, ylabel, title_, logy=False):
        if not row[curve_key]:
            return
        ax = fig.add_subplot(pos)
        ax.plot(row[curve_key], linewidth=1.2)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title_)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # Energy convergence with exact line
    if row["energy_curve"]:
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(row["energy_curve"], label="VMC energy", linewidth=1.5)
        ax.axhline(
            row["exact_energy"],
            color="red",
            linestyle="--",
            label=f"Exact: {row['exact_energy']:.4f}",
        )
        ax.set_title("Energy Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.legend()
        ax.grid(alpha=0.3)

    _plot(gs[0, 2], "energy_error_curve", "Std/√N", "Statistical Error")
    _plot(gs[1, 0], "grad_norm_curve", "‖x‖", "Gradient Norm")
    _plot(gs[1, 1], "cond_curve", "κ(S)", "S Condition Number", logy=True)
    _plot(gs[1, 2], "weight_norm_curve", "‖w‖", "Weight Norm")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Analyze benchmark results by (N, h)")
    p.add_argument("--results", default="results/", help="Results directory")
    p.add_argument("--top", type=int, default=3, help="Top N runs to show per (N,h)")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    p.add_argument(
        "--diag-best", action="store_true", help="Show diagnostics for best run"
    )
    p.add_argument(
        "--diag-worst", action="store_true", help="Show diagnostics for worst run"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading results from {args.results} ...")
    df = load_results(Path(args.results))
    print(
        f"Loaded {len(df)} runs across "
        f"{df.groupby(['size', 'h']).ngroups} (N, h) combinations."
    )

    # Always print tables
    print_nh_summary(df, top_n=args.top)
    print_hyperparam_sensitivity(df)

    if not args.no_plots:
        # plot_error_heatmap(df)
        plot_convergence_by_nh(df, top_n=args.top)

        if args.diag_best:
            plot_diagnostics(df.loc[df["rel_error"].idxmin()])

        if args.diag_worst:
            plot_diagnostics(df.loc[df["rel_error"].idxmax()])


if __name__ == "__main__":
    main()
