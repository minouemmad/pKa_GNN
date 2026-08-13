#!/usr/bin/env python
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

# ───────────────────────────── config ──────────────────────────────
RADII = [7, 8, 9, 10, 11]
DATASET_IDX = {7: 0, 8: 1, 9: 2, 10: 3, 11: 4}  # 07_train naming
PRIMARY_RADIUS = 9
RESIDUES = ["ASP", "GLU", "HIS", "LYS", "CYS", "TYR"]

CONDITIONS = {
    # tag -> sub-dir under sweep root
    "rot_withDip": "Sweep2_rot_withDip",
    "rot_noDip":   "Sweep2_rot_noDip",
    "tit_withDip": "Sweep2_tit_withDip",
    "tit_noDip":   "Sweep2_tit_noDip",
}

PAIRS = [
    # (label, A, B)  → test |err_A| vs |err_B|; ΔMAE = MAE(A) − MAE(B)
    ("rot:dip-vs-noDip", "rot_withDip", "rot_noDip"),
    ("tit:dip-vs-noDip", "tit_withDip", "tit_noDip"),
    ("tit-vs-rot:withDip", "tit_withDip", "rot_withDip"),
    ("tit-vs-rot:noDip",   "tit_noDip",   "rot_noDip"),
]

KEY_COLS = ["PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"]

def _key(df: pd.DataFrame) -> pd.Series:
    return (
        df["PDB_ID"].astype(str) + "|" +
        df["Chain_ID"].astype(str) + "|" +
        df["Residue_Number"].astype(int).astype(str) + "|" +
        df["Residue_Name"].astype(str)
    )

def _agg_titrate(df: pd.DataFrame) -> pd.DataFrame:
    """Titrate mode has multiple pH copies per residue; average predictions
    over copies to get one residue-level prediction matching rotopt."""
    grp = df.groupby(KEY_COLS, as_index=False).agg(
        True_pKa=("True_pKa", "first"),
        Predicted_pKa=("Predicted_pKa", "mean"),
    )
    return grp

def _load_predictions(sweep_root: Path, subdir: str, radius: int,
                      is_titrate: bool, log: logging.Logger) -> pd.DataFrame | None:
    p = sweep_root / subdir / "predictions" / f"dataset_{DATASET_IDX[radius]}_all_folds.csv"
    if not p.exists():
        log.warning(f"missing predictions: {p}")
        return None
    df = pd.read_csv(p)
    miss = [c for c in KEY_COLS + ["True_pKa", "Predicted_pKa"] if c not in df.columns]
    if miss:
        log.warning(f"{p} missing cols {miss}; skipping")
        return None
    if is_titrate:
        df = _agg_titrate(df)
    else:
        # Average over duplicate keys just in case
        df = df.groupby(KEY_COLS, as_index=False).agg(
            True_pKa=("True_pKa", "first"),
            Predicted_pKa=("Predicted_pKa", "mean"),
        )
    df["abs_err"] = (df["True_pKa"] - df["Predicted_pKa"]).abs()
    df["key"] = _key(df)
    return df

def _bootstrap_delta_mae(a_err: np.ndarray, b_err: np.ndarray,
                         n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean(a_err − b_err) → (ΔMAE point, lo, hi 95%)."""
    rng = np.random.default_rng(seed)
    n = len(a_err)
    d = a_err - b_err
    point = float(d.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", default="Graph_pKa/Results",
                    help="Directory containing the four condition subdirs.")
    ap.add_argument("--out-dir", default="Graph_pKa/Results/Analysis_Sweep2")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    log = logging.getLogger("analyze")

    sweep_root = Path(args.sweep_root)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load everything ───────────────────────────────────────────
    # data[tag][radius] = DataFrame with key, True_pKa, Predicted_pKa, abs_err
    data: dict[str, dict[int, pd.DataFrame]] = {tag: {} for tag in CONDITIONS}
    for tag, subdir in CONDITIONS.items():
        is_tit = tag.startswith("tit")
        for r in RADII:
            df = _load_predictions(sweep_root, subdir, r, is_tit, log)
            if df is not None:
                data[tag][r] = df
                log.info(f"{tag} r={r}: {len(df)} residues  MAE={df['abs_err'].mean():.4f}")
            else:
                log.warning(f"{tag} r={r}: missing")

    # ── summary by radius (overall) ───────────────────────────────
    summary_rows = []
    for tag in CONDITIONS:
        for r in RADII:
            if r not in data[tag]:
                continue
            df = data[tag][r]
            err = df["abs_err"].values
            mae  = float(np.mean(err))
            rmse = float(np.sqrt(np.mean((df["True_pKa"] - df["Predicted_pKa"]) ** 2)))
            summary_rows.append({"condition": tag, "radius": r,
                                  "N": len(df), "MAE": mae, "RMSE": rmse})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_by_radius.csv", index=False)
    log.info(f"wrote {out_dir/'summary_by_radius.csv'}")

    # ── paired tests on intersection per radius ───────────────────
    paired_rows = []
    for r in RADII:
        if not all(r in data[tag] for tag in CONDITIONS):
            log.warning(f"r={r}: not all conditions present, skipping paired tests")
            continue
        # Intersection of keys across all 4 conditions:
        key_sets = [set(data[tag][r]["key"]) for tag in CONDITIONS]
        common = sorted(set.intersection(*key_sets))
        if len(common) < 10:
            log.warning(f"r={r}: intersection too small ({len(common)}), skipping")
            continue
        log.info(f"r={r}: paired-test N={len(common)}")
        # Build aligned error arrays
        err = {}
        for tag in CONDITIONS:
            df = data[tag][r].set_index("key").loc[common]
            err[tag] = df["abs_err"].values
        for label, A, B in PAIRS:
            a, b = err[A], err[B]
            try:
                w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                W, pval = float(w.statistic), float(w.pvalue)
            except ValueError:
                W, pval = float("nan"), float("nan")
            dmae, lo, hi = _bootstrap_delta_mae(a, b, n_boot=args.n_boot, seed=r)
            paired_rows.append({
                "radius": r, "pair": label, "A": A, "B": B,
                "N": len(common),
                "MAE_A": float(a.mean()), "MAE_B": float(b.mean()),
                "delta_MAE": dmae, "ci_lo": lo, "ci_hi": hi,
                "wilcoxon_W": W, "p_value": pval,
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_tests_by_radius.csv", index=False)
    log.info(f"wrote {out_dir/'paired_tests_by_radius.csv'}")

    # ── per-residue-type breakdown ────────────────────────────────
    per_res_rows = []
    for tag in CONDITIONS:
        for r in RADII:
            if r not in data[tag]:
                continue
            df = data[tag][r]
            for res in RESIDUES:
                sub = df[df["Residue_Name"] == res]
                if len(sub) == 0:
                    continue
                per_res_rows.append({
                    "condition": tag, "radius": r, "residue": res,
                    "N": len(sub),
                    "MAE": float(sub["abs_err"].mean()),
                })
    per_res = pd.DataFrame(per_res_rows)
    per_res.to_csv(out_dir / "per_residue_by_radius.csv", index=False)
    log.info(f"wrote {out_dir/'per_residue_by_radius.csv'}")

    # ── plots ─────────────────────────────────────────────────────
    # 1. MAE vs radius (4 lines)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"rot_withDip": "tab:blue", "rot_noDip": "tab:cyan",
              "tit_withDip": "tab:red",  "tit_noDip": "tab:orange"}
    styles = {"rot_withDip": "-", "rot_noDip": "--",
              "tit_withDip": "-", "tit_noDip": "--"}
    for tag in CONDITIONS:
        sub = summary[summary["condition"] == tag].sort_values("radius")
        if len(sub):
            ax.plot(sub["radius"], sub["MAE"], marker="o",
                    color=colors[tag], linestyle=styles[tag], label=tag)
    ax.set_xlabel("Cutoff radius (Å)")
    ax.set_ylabel("MAE (pKa units)")
    ax.set_title("Overall MAE vs radius")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_vs_radius.png", dpi=160)
    plt.close(fig)

    # 2. ΔMAE vs radius with 95% CI
    fig, ax = plt.subplots(figsize=(7, 5))
    pair_colors = {"rot:dip-vs-noDip": "tab:blue",
                   "tit:dip-vs-noDip": "tab:red",
                   "tit-vs-rot:withDip": "tab:green",
                   "tit-vs-rot:noDip":   "tab:olive"}
    for label, _, _ in PAIRS:
        sub = paired[paired["pair"] == label].sort_values("radius")
        if len(sub) == 0:
            continue
        x = sub["radius"].values
        y = sub["delta_MAE"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o",
                    capsize=3, label=label, color=pair_colors.get(label))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Cutoff radius (Å)")
    ax.set_ylabel("ΔMAE = MAE(A) − MAE(B)")
    ax.set_title("Paired ΔMAE with bootstrap 95% CI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_mae_vs_radius.png", dpi=160)
    plt.close(fig)

    # 3. per-residue heatmap @ primary radius
    sub = per_res[per_res["radius"] == PRIMARY_RADIUS]
    if len(sub):
        piv = sub.pivot(index="residue", columns="condition", values="MAE")
        piv = piv.reindex(index=RESIDUES, columns=list(CONDITIONS.keys()))
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > np.nanmean(piv.values) else "black",
                            fontsize=8)
        fig.colorbar(im, ax=ax, label="MAE")
        ax.set_title(f"Per-residue MAE @ r={PRIMARY_RADIUS} Å")
        fig.tight_layout()
        fig.savefig(out_dir / "per_residue_heatmap.png", dpi=160)
        plt.close(fig)

    # ── stdout headline ──────────────────────────────────────────
    print("\n=== SUMMARY (overall MAE) ===")
    piv = summary.pivot(index="radius", columns="condition", values="MAE")
    piv = piv.reindex(columns=list(CONDITIONS.keys()))
    print(piv.round(4).to_string())

    print("\n=== PAIRED TESTS @ primary radius r={} ===".format(PRIMARY_RADIUS))
    sub = paired[paired["radius"] == PRIMARY_RADIUS]
    if len(sub):
        cols = ["pair", "N", "MAE_A", "MAE_B", "delta_MAE", "ci_lo", "ci_hi", "p_value"]
        print(sub[cols].round(4).to_string(index=False))

    print(f"\nAll outputs → {out_dir}")

if __name__ == "__main__":
    main()
