
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

VARIANTS = ["Charge", "Invariant", "PhysEdge", "InvariantPhysEdge"]

def load_perfold(results_root: Path, variant: str, seed: int, ds_idx: int = 2) -> pd.DataFrame | None:
    p = results_root / f"Training_{variant}_seed{seed}" / "predictions" / f"dataset_{ds_idx}_all_folds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["abs_err"] = (df["Predicted_pKa"] - df["True_pKa"]).abs()
    df["sq_err"]  = (df["Predicted_pKa"] - df["True_pKa"]) ** 2
    out = df.groupby("fold").agg(
        MAE=("abs_err", "mean"),
        RMSE=("sq_err", lambda x: float(np.sqrt(x.mean()))),
        N=("abs_err", "size"),
    ).reset_index()
    out.insert(0, "seed", seed)
    out.insert(0, "variant", variant)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("Graph_pKa/Net_FFX138_PhysEdge"))
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 1, 7, 123, 2026, 17, 99, 314])
    ap.add_argument("--ds-idx", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    for v in VARIANTS:
        for s in args.seeds:
            df = load_perfold(args.results_dir, v, s, args.ds_idx)
            if df is None:
                print(f"  MISSING: {v}_seed{s}")
                continue
            rows.append(df)
    if not rows:
        print("No results found.")
        return

    perfold = pd.concat(rows, ignore_index=True)
    perfold_path = out_dir / "sweep_physedge_138_perfold.csv"
    perfold.to_csv(perfold_path, index=False)
    print(f"Wrote {perfold_path}  ({len(perfold)} rows)")

    summary = perfold.groupby("variant").agg(
        n_seeds=("seed", "nunique"),
        n_folds=("fold", "size"),
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"),
    ).reset_index()

    base = perfold[perfold["variant"] == "Charge"][["seed", "fold", "MAE"]].rename(columns={"MAE": "MAE_base"})
    deltas: list[dict] = []
    for v in VARIANTS:
        if v == "Charge":
            deltas.append(dict(variant=v, n_pairs=0, mean_delta_MAE=0.0,
                               wilcoxon_p=np.nan, ttest_p=np.nan, frac_better=np.nan))
            continue
        sub = perfold[perfold["variant"] == v][["seed", "fold", "MAE"]]
        merged = sub.merge(base, on=["seed", "fold"], how="inner")
        if len(merged) < 2:
            deltas.append(dict(variant=v, n_pairs=len(merged), mean_delta_MAE=np.nan,
                               wilcoxon_p=np.nan, ttest_p=np.nan, frac_better=np.nan))
            continue
        delta = merged["MAE"].values - merged["MAE_base"].values
        try:
            _, w_p = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            w_p = np.nan
        try:
            _, t_p = ttest_rel(merged["MAE"].values, merged["MAE_base"].values)
        except ValueError:
            t_p = np.nan
        deltas.append(dict(
            variant=v,
            n_pairs=len(merged),
            mean_delta_MAE=float(delta.mean()),
            wilcoxon_p=float(w_p) if not np.isnan(w_p) else np.nan,
            ttest_p=float(t_p) if not np.isnan(t_p) else np.nan,
            frac_better=float((delta < 0).mean()),
        ))
    delta_df = pd.DataFrame(deltas)
    summary = summary.merge(delta_df, on="variant", how="left")
    summary = summary.sort_values("mean_MAE").reset_index(drop=True)

    summary_path = out_dir / "sweep_physedge_138_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
