"""
plot_electrostatics.py

Generate presentation-quality plots from the electrostatics sweep:
  1. Per-variant MAE box/strip plot across (seed, fold) pairs
  2. ΔMAE vs Charge baseline with paired Wilcoxon p-value annotations
  3. Predicted vs True scatter for the best variant (best seed)
  4. Permutation importance bar chart

Outputs PNGs to <results-dir>/plots/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANT_ORDER = ["Charge", "InducedDip", "PermDip", "BothDip", "CoulombEdge", "CoulombEdgeBoth"]
VARIANT_LABELS = {
    "Charge":          "Charge\n(paper)",
    "InducedDip":      "+Induced\nDipole",
    "PermDip":         "+Perm\nDipole",
    "BothDip":         "+Both\nDipoles",
    "CoulombEdge":     "+Coulomb\nEdge",
    "CoulombEdgeBoth": "+Coulomb\n+BothDip",
}
PALETTE = {
    "Charge":          "#7f7f7f",
    "InducedDip":      "#1f77b4",
    "PermDip":         "#2ca02c",
    "BothDip":         "#9467bd",
    "CoulombEdge":     "#ff7f0e",
    "CoulombEdgeBoth": "#d62728",
}


def plot_box(perfold: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    variants = [v for v in VARIANT_ORDER if v in perfold["variant"].unique()]
    data = [perfold.loc[perfold["variant"] == v, "MAE"].values for v in variants]
    pos = np.arange(len(variants))
    box = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                     showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                    markeredgecolor="black", markersize=6),
                     medianprops=dict(color="black", linewidth=1.5))
    for patch, v in zip(box["boxes"], variants):
        patch.set_facecolor(PALETTE.get(v, "#888"))
        patch.set_alpha(0.55)
    # Strip plot of individual fold MAEs
    for i, (v, vals) in enumerate(zip(variants, data)):
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=10, alpha=0.35, color=PALETTE.get(v, "#222"), edgecolors="none")
    ax.set_xticks(pos)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=10)
    ax.set_ylabel("Per-fold MAE  (pKa units)", fontsize=11)
    ax.set_title(f"FFX 138-PDB rotopt set, radius 9 Å\n"
                 f"{summary['n_seeds'].iloc[0]} seeds × 10 folds = "
                 f"{summary['n_folds'].iloc[0]} per variant", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")
    plt.close(fig)


def plot_delta(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    s = summary[summary["variant"] != "Charge"].copy()
    s = s.sort_values("mean_delta_MAE")
    pos = np.arange(len(s))
    colors = [PALETTE.get(v, "#888") for v in s["variant"]]
    ax.barh(pos, s["mean_delta_MAE"], color=colors, alpha=0.85, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (_, row) in enumerate(s.iterrows()):
        p = row["wilcoxon_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        x = row["mean_delta_MAE"]
        ax.text(x + (0.0008 if x >= 0 else -0.0008), i,
                f"Δ={x:+.4f}  p={p:.3g} {sig}",
                va="center", ha="left" if x >= 0 else "right", fontsize=9)
    ax.set_yticks(pos)
    ax.set_yticklabels([VARIANT_LABELS.get(v, v).replace("\n", " ") for v in s["variant"]])
    ax.set_xlabel("ΔMAE  vs  Charge baseline   (negative = better)", fontsize=11)
    ax.set_title("Paired Wilcoxon test across (seed, fold) pairs", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")
    plt.close(fig)


def plot_scatter(results_dir: Path, variant: str, seed: int, ds_idx: int,
                 out: Path) -> None:
    csv = results_dir / f"Training_{variant}_seed{seed}" / "predictions" / f"dataset_{ds_idx}_all_folds.csv"
    if not csv.exists():
        print(f"  scatter: missing {csv}")
        return
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = float(min(df["True_pKa"].min(), df["Predicted_pKa"].min())) - 0.5
    hi = float(max(df["True_pKa"].max(), df["Predicted_pKa"].max())) + 0.5
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.scatter(df["True_pKa"], df["Predicted_pKa"], s=18, alpha=0.55,
               color=PALETTE.get(variant, "#1f77b4"), edgecolors="none")
    mae = (df["Predicted_pKa"] - df["True_pKa"]).abs().mean()
    rmse = float(np.sqrt(((df["Predicted_pKa"] - df["True_pKa"]) ** 2).mean()))
    r = float(np.corrcoef(df["True_pKa"], df["Predicted_pKa"])[0, 1])
    ax.set_xlabel("Experimental pKa", fontsize=11)
    ax.set_ylabel("Predicted pKa", fontsize=11)
    ax.set_title(f"{variant}  seed={seed}\n"
                 f"MAE={mae:.3f}   RMSE={rmse:.3f}   R={r:.3f}   N={len(df)}",
                 fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")
    plt.close(fig)


def plot_perm_importance(perm_csv: Path, out: Path, variant: str) -> None:
    if not perm_csv.exists():
        print(f"  perm-importance: missing {perm_csv}")
        return
    df = pd.read_csv(perm_csv).sort_values("delta_MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.45 * len(df) + 1)))
    pos = np.arange(len(df))
    colors = ["#d62728" if d > 0.01 else "#7f7f7f" for d in df["delta_MAE"]]
    ax.barh(pos, df["delta_MAE"], color=colors, alpha=0.85, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        x = row["delta_MAE"]
        ax.text(x + 0.001, i, f"{x:+.3f}", va="center", fontsize=9)
    ax.set_yticks(pos)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("ΔMAE on shuffle  (larger = more important)", fontsize=11)
    ax.set_title(f"Permutation importance — {variant}", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("Graph_pKa/Net_FFX138_Electro"))
    ap.add_argument("--ds-idx", type=int, default=2)
    args = ap.parse_args()

    perfold = pd.read_csv(args.results_dir / "sweep_electrostatics_138_perfold.csv")
    summary = pd.read_csv(args.results_dir / "sweep_electrostatics_138_summary.csv")
    plots = args.results_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    plot_box(perfold, summary, plots / "01_mae_box.png")
    plot_delta(summary, plots / "02_delta_vs_charge.png")

    # Best variant by mean MAE
    best_variant = summary.iloc[0]["variant"]
    best_pf = perfold[perfold["variant"] == best_variant]
    if not best_pf.empty:
        # Pick the seed with lowest mean MAE for that variant
        best_seed = int(best_pf.groupby("seed")["MAE"].mean().idxmin())
        plot_scatter(args.results_dir, best_variant, best_seed, args.ds_idx,
                     plots / f"03_scatter_{best_variant}_seed{best_seed}.png")

    # Also the Charge baseline scatter for comparison
    cg_pf = perfold[perfold["variant"] == "Charge"]
    if not cg_pf.empty:
        cg_seed = int(cg_pf.groupby("seed")["MAE"].mean().idxmin())
        plot_scatter(args.results_dir, "Charge", cg_seed, args.ds_idx,
                     plots / f"04_scatter_Charge_seed{cg_seed}.png")

    perm_csv = args.results_dir / f"perm_importance_{best_variant}.csv"
    plot_perm_importance(perm_csv, plots / f"05_perm_importance_{best_variant}.png",
                         best_variant)
    # Also plot any other perm-importance CSVs found in results-dir
    for extra in args.results_dir.glob("perm_importance_*.csv"):
        if extra == perm_csv:
            continue
        name = extra.stem.replace("perm_importance_", "")
        plot_perm_importance(extra, plots / f"06_perm_importance_{name}.png", name)


if __name__ == "__main__":
    main()
