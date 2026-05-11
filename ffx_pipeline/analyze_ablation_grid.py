#!/usr/bin/env python3
"""
analyze_ablation_grid.py — aggregate the 40-job sweep produced by
run_ablation_sweep.py.

Layout consumed:
    Graph_pKa/Results/Comparison_2026/{mode}_{cond}/predictions/
        dataset_{0..4}_all_folds.csv

mode      ∈ {rotopt, titrate}
cond      ∈ {full, noDip, noPerm, noElec}
radius    7..11 Å  ↔ dataset_idx 0..4

Outputs (under --out-dir, default Comparison_2026/_analysis):
    summary_by_radius.csv          overall MAE/RMSE/N per (mode, cond, radius)
    per_residue_by_radius.csv      MAE per (mode, cond, radius, residue)
    paired_tests_by_radius.csv     Wilcoxon |err| + bootstrap ΔMAE for key pairs
    best_per_mode.csv              best (cond, radius) per mode
    mae_vs_radius_{mode}.png       4-line plot per mode
    delta_mae_vs_radius.png        ablation contributions per mode
    per_residue_heatmap_r{R}.png   residue × cond MAE heatmap (best radius)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT_DEFAULT = Path("Graph_pKa/Results/Comparison_2026")
RADII        = [7, 8, 9, 10, 11]
DATASET_IDX  = {r: i for i, r in enumerate(RADII)}
MODES        = ["rotopt", "titrate"]
CONDS        = ["full", "noDip", "noPerm", "noElec"]
RESIDUES     = ["Aspartate", "Glutamate", "Histidine", "Lysine", "Cysteine", "Tyrosine"]
KEY_COLS     = ["PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"]

# Paired comparisons: (label, A, B)  ⇒  ΔMAE = MAE(A) − MAE(B); test |err_A| vs |err_B|
PAIRS_PER_MODE = [
    ("noDip-vs-full",   "noDip",  "full"),   # cost of removing induced dipoles
    ("noPerm-vs-full",  "noPerm", "full"),   # cost of removing permanent multipoles
    ("noElec-vs-full",  "noElec", "full"),   # cost of removing all electrostatics
    ("noPerm-vs-noDip", "noPerm", "noDip"),  # which contributes more?
]


# ─────────────────────────── helpers ───────────────────────────
def _key(df: pd.DataFrame) -> pd.Series:
    return (df["PDB_ID"].astype(str) + "|" +
            df["Chain_ID"].astype(str) + "|" +
            df["Residue_Number"].astype(int).astype(str) + "|" +
            df["Residue_Name"].astype(str))


def _agg_titrate(df: pd.DataFrame) -> pd.DataFrame:
    """Average Predicted_pKa over pH replicates per residue key."""
    return df.groupby(KEY_COLS, as_index=False).agg(
        True_pKa=("True_pKa", "first"),
        Predicted_pKa=("Predicted_pKa", "mean"),
    )


def _load_pred(root: Path, mode: str, cond: str, radius: int,
               log: logging.Logger) -> pd.DataFrame | None:
    p = root / f"{mode}_{cond}" / "predictions" / f"dataset_{DATASET_IDX[radius]}_all_folds.csv"
    if not p.exists():
        log.warning(f"missing: {p}")
        return None
    df = pd.read_csv(p)
    miss = [c for c in KEY_COLS + ["True_pKa", "Predicted_pKa"] if c not in df.columns]
    if miss:
        log.warning(f"{p}: missing cols {miss}")
        return None
    df = _agg_titrate(df) if mode == "titrate" else df.groupby(
        KEY_COLS, as_index=False).agg(
            True_pKa=("True_pKa", "first"),
            Predicted_pKa=("Predicted_pKa", "mean"))
    df["abs_err"] = (df["True_pKa"] - df["Predicted_pKa"]).abs()
    df["key"] = _key(df)
    return df


def _bootstrap_dmae(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ─────────────────────────── main ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot",  type=int, default=5000)
    args = ap.parse_args()

    out = args.out_dir or (args.root / "_analysis")
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    log = logging.getLogger("agg")

    # data[mode][cond][radius] -> DataFrame
    data: dict = {m: {c: {} for c in CONDS} for m in MODES}
    for m in MODES:
        for c in CONDS:
            for r in RADII:
                df = _load_pred(args.root, m, c, r, log)
                if df is not None:
                    data[m][c][r] = df

    # 1) summary_by_radius.csv
    rows = []
    for m in MODES:
        for c in CONDS:
            for r in RADII:
                df = data[m][c].get(r)
                if df is None:
                    continue
                rmse = float(np.sqrt(((df["True_pKa"] - df["Predicted_pKa"]) ** 2).mean()))
                rows.append({"mode": m, "condition": c, "radius": r,
                             "N": len(df),
                             "MAE": float(df["abs_err"].mean()),
                             "RMSE": rmse})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_by_radius.csv", index=False)
    log.info(f"wrote {out/'summary_by_radius.csv'}")

    # 2) per_residue_by_radius.csv
    rows = []
    for m in MODES:
        for c in CONDS:
            for r in RADII:
                df = data[m][c].get(r)
                if df is None:
                    continue
                for res in RESIDUES:
                    sub = df[df["Residue_Name"] == res]
                    if len(sub):
                        rows.append({"mode": m, "condition": c, "radius": r,
                                     "residue": res, "N": len(sub),
                                     "MAE": float(sub["abs_err"].mean())})
    per_res = pd.DataFrame(rows)
    per_res.to_csv(out / "per_residue_by_radius.csv", index=False)
    log.info(f"wrote {out/'per_residue_by_radius.csv'}")

    # 3) paired tests within each mode (intersection across 4 conditions)
    rows = []
    for m in MODES:
        for r in RADII:
            present = [c for c in CONDS if r in data[m][c]]
            if len(present) < 4:
                log.warning(f"{m} r={r}: only {present} present, skipping pairs")
                continue
            sets = [set(data[m][c][r]["key"]) for c in CONDS]
            common = sorted(set.intersection(*sets))
            if len(common) < 10:
                log.warning(f"{m} r={r}: intersection too small ({len(common)})")
                continue
            err = {}
            for c in CONDS:
                d = data[m][c][r].set_index("key").loc[common]
                err[c] = d["abs_err"].values
            for label, A, B in PAIRS_PER_MODE:
                a, b = err[A], err[B]
                try:
                    w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    W, p = float(w.statistic), float(w.pvalue)
                except ValueError:
                    W, p = float("nan"), float("nan")
                dmae, lo, hi = _bootstrap_dmae(a, b, args.n_boot,
                                               seed=hash((m, r, label)) & 0xFFFF)
                rows.append({"mode": m, "radius": r, "pair": label,
                             "A": A, "B": B, "N": len(common),
                             "MAE_A": float(a.mean()), "MAE_B": float(b.mean()),
                             "delta_MAE": dmae, "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_W": W, "p_value": p})
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired_tests_by_radius.csv", index=False)
    log.info(f"wrote {out/'paired_tests_by_radius.csv'}")

    # 4) best per mode
    best_rows = []
    for m in MODES:
        sub = summary[summary["mode"] == m]
        if len(sub) == 0:
            continue
        b = sub.loc[sub["MAE"].idxmin()]
        best_rows.append(dict(b))
    best = pd.DataFrame(best_rows)
    best.to_csv(out / "best_per_mode.csv", index=False)

    # ── plots ────────────────────────────────────────────────
    cond_color = {"full": "tab:blue", "noDip": "tab:orange",
                  "noPerm": "tab:green", "noElec": "tab:red"}
    for m in MODES:
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in CONDS:
            sub = summary[(summary["mode"] == m) & (summary["condition"] == c)]
            sub = sub.sort_values("radius")
            if len(sub):
                ax.plot(sub["radius"], sub["MAE"], marker="o",
                        color=cond_color[c], label=c)
        ax.set_xlabel("Cutoff radius (Å)")
        ax.set_ylabel("MAE (pKa units)")
        ax.set_title(f"{m}: MAE vs radius (default HP, 10-fold CV)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"mae_vs_radius_{m}.png", dpi=160)
        plt.close(fig)

    # ΔMAE per mode (one panel each)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    pair_color = {"noDip-vs-full": "tab:orange",
                  "noPerm-vs-full": "tab:green",
                  "noElec-vs-full": "tab:red",
                  "noPerm-vs-noDip": "tab:purple"}
    for ax, m in zip(axes, MODES):
        for label, _, _ in PAIRS_PER_MODE:
            sub = paired[(paired["mode"] == m) & (paired["pair"] == label)]
            sub = sub.sort_values("radius")
            if len(sub) == 0:
                continue
            x = sub["radius"].values
            y = sub["delta_MAE"].values
            lo = sub["ci_lo"].values
            hi = sub["ci_hi"].values
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o",
                        capsize=3, label=label, color=pair_color[label])
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("Cutoff radius (Å)")
        ax.set_title(f"{m}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("ΔMAE = MAE(A) − MAE(B)")
    fig.suptitle("Ablation contributions (Wilcoxon-tested, bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(out / "delta_mae_vs_radius.png", dpi=160)
    plt.close(fig)

    # per-residue heatmap at each mode's best radius
    for m in MODES:
        sub_m = summary[summary["mode"] == m]
        if len(sub_m) == 0:
            continue
        best_radius = int(sub_m.loc[sub_m["MAE"].idxmin(), "radius"])
        sub = per_res[(per_res["mode"] == m) & (per_res["radius"] == best_radius)]
        if len(sub) == 0:
            continue
        piv = sub.pivot(index="residue", columns="condition", values="MAE")
        piv = piv.reindex(index=RESIDUES, columns=CONDS)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > np.nanmean(piv.values) else "black",
                            fontsize=8)
        fig.colorbar(im, ax=ax, label="MAE")
        ax.set_title(f"{m}: per-residue MAE @ r={best_radius} Å")
        fig.tight_layout()
        fig.savefig(out / f"per_residue_heatmap_{m}_r{best_radius}.png", dpi=160)
        plt.close(fig)

    # ── stdout headline ─────────────────────────────────────
    print("\n=== OVERALL MAE (rows=radius, cols=condition) ===")
    for m in MODES:
        print(f"\n[{m}]")
        sub = summary[summary["mode"] == m]
        if len(sub):
            piv = sub.pivot(index="radius", columns="condition", values="MAE")
            piv = piv.reindex(columns=CONDS)
            print(piv.round(4).to_string())

    print("\n=== BEST PER MODE ===")
    print(best.to_string(index=False))

    print(f"\nAll outputs → {out}")


if __name__ == "__main__":
    main()
