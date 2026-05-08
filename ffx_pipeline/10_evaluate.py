#!/usr/bin/env python3
"""
10_evaluate.py  (FFX pipeline)

Paper-style evaluation of the GAT pK_a model, mirroring:

    Graph-based deep learning models for predicting pKa values of protein
    ionizable residues via physically inspired ...   (J. Chem. Inf. Model. 2026)

Implements:
  • Per-residue **ensemble** prediction = mean over the 10 fold checkpoints
    (and, for titrate mode, over the per-pH replicates).
  • Overall MAE, RMSE, Pearson R with **bootstrap 95% CIs**.
  • Quartile **rolling MAE/RMSE** binned by |ΔpKa_exp| where
        ΔpKa_exp = pKa_exp − reference_pKa(residue)
    Default reference values from the paper (Asp 3.7, Glu 4.2, His 6.5,
    Lys 10.4, Cys 8.6, Tyr 10.0).  By default the bin edges are the
    paper's Figure-10 edges [0.0, 0.2, 0.5, 1.0, +∞] but they can be
    overridden with --bin-edges.
  • Per-residue-type breakdown (MAE / RMSE / R / N).
  • Optional merging of baseline predictions from CSVs supplied via
    --baseline NAME=path/to/preds.csv ...  (e.g. PROPKA3, DeepKa,
    pKAI+).  Each baseline CSV must have columns:
        PDB_ID, Chain_ID, Residue_Number, Residue_Name, Predicted_pKa
    Rows missing in the baseline are dropped from the *intersection*
    evaluation (paper Figure 10 reports on the 796 residues present in
    all methods).  Per-method metrics over their own coverage are also
    reported.

Run:
    # Single results dir (uses all dataset_*_all_folds.csv it finds):
    python ffx_pipeline/10_evaluate.py \\
        --results-dir Graph_pKa/Results/Train_FFX_rotopt_best

    # Specific predictions file:
    python ffx_pipeline/10_evaluate.py \\
        --predictions Graph_pKa/Results/Train_FFX_rotopt_best/predictions/dataset_2_all_folds.csv

    # With baselines:
    python ffx_pipeline/10_evaluate.py \\
        --results-dir Graph_pKa/Results/Train_FFX_rotopt_best \\
        --baseline PROPKA3=baselines/propka3_predictions.csv \\
        --baseline DeepKa=baselines/deepka_predictions.csv \\
        --baseline pKAI+=baselines/pkai_predictions.csv \\
        --intersect-only

Outputs (under <results-dir>/evaluation/ or alongside --predictions):
    summary_overall.csv         ← method, MAE, RMSE, R, N, with 95% CIs
    summary_by_residue.csv      ← per-residue-type metrics for every method
    summary_quartiles.csv       ← Q1..Q4 MAE/RMSE per method (rolling)
    ensemble_predictions.csv    ← merged per-residue table used for evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── Defaults from the paper ──────────────────────────────────────────────────
REFERENCE_PKA: dict[str, float] = {
    "ASP": 3.7,  "GLU": 4.2,  "HIS": 6.5,  "LYS": 10.4,
    "CYS": 8.6,  "TYR": 10.0,
}
DEFAULT_BIN_EDGES = [0.0, 0.2, 0.5, 1.0, float("inf")]
DEFAULT_BIN_LABELS = ["Q1", "Q2", "Q3", "Q4"]

# Map the long Residue_Name written by 06_create_datasets.py to 3-letter codes.
LONG_TO_CODE: dict[str, str] = {
    "Aspartate":   "ASP", "Aspartic Acid": "ASP",
    "Glutamate":   "GLU", "Glutamic Acid": "GLU",
    "Histidine":   "HIS",
    "Lysine":      "LYS",
    "Cysteine":    "CYS",
    "Tyrosine":    "TYR",
}


def to_residue_code(name: str) -> str:
    """Return 3-letter uppercase code for a residue name (long or short)."""
    if pd.isna(name):
        return ""
    n = str(name).strip()
    if n.upper() in REFERENCE_PKA:
        return n.upper()
    return LONG_TO_CODE.get(n, n[:3].upper())


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def _mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def _rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def _pearson(y, yhat):
    if len(y) < 2:
        return float("nan")
    r, _ = pearsonr(np.asarray(y), np.asarray(yhat))
    return float(r)


def bootstrap_ci(y, yhat, statistic, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42):
    """Return (point, low, high) for `statistic(y, yhat)` via percentile
    bootstrap.  NaN-safe."""
    y    = np.asarray(y,    dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n = len(y)
    if n == 0:
        return (float("nan"),) * 3
    point = statistic(y, yhat)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(y[idx], yhat[idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def metrics_with_ci(y, yhat, n_boot: int = 1000, seed: int = 42) -> dict:
    mae,  mae_lo,  mae_hi  = bootstrap_ci(y, yhat, _mae,     n_boot, seed=seed)
    rmse, rmse_lo, rmse_hi = bootstrap_ci(y, yhat, _rmse,    n_boot, seed=seed)
    r,    r_lo,    r_hi    = bootstrap_ci(y, yhat, _pearson, n_boot, seed=seed)
    return {
        "N":        len(y),
        "MAE":      mae,  "MAE_lo":  mae_lo,  "MAE_hi":  mae_hi,
        "RMSE":     rmse, "RMSE_lo": rmse_lo, "RMSE_hi": rmse_hi,
        "Pearson":  r,    "Pearson_lo": r_lo, "Pearson_hi": r_hi,
    }


# ════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ════════════════════════════════════════════════════════════════════════════

KEY_COLS = ["PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"]


def load_predictions_csv(path: Path) -> pd.DataFrame:
    """Load an `_all_folds.csv` (or any CSV with the standard columns)."""
    df = pd.read_csv(path)
    needed = set(KEY_COLS + ["True_pKa", "Predicted_pKa"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["PDB_ID"]    = df["PDB_ID"].astype(str)
    df["Chain_ID"]  = df["Chain_ID"].astype(str)
    df["res_code"]  = df["Residue_Name"].map(to_residue_code)
    return df


def aggregate_ensemble(df: pd.DataFrame, label: str = "Ensemble") -> pd.DataFrame:
    """Mean per-residue prediction across folds (and pH replicates).

    Returns one row per (PDB, Chain, ResNum, ResName) with columns
    True_pKa and `<label>` for the predicted value.
    """
    g = df.groupby(KEY_COLS, as_index=False).agg(
        True_pKa=("True_pKa", "mean"),
        Predicted_pKa=("Predicted_pKa", "mean"),
    )
    g = g.rename(columns={"Predicted_pKa": label})
    g["res_code"] = g["Residue_Name"].map(to_residue_code)
    return g


def discover_predictions(results_dir: Path) -> list[Path]:
    pred_dir = results_dir / "predictions"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"No predictions/ dir under {results_dir}")
    files = sorted(pred_dir.glob("dataset_*_all_folds.csv"))
    if not files:
        raise FileNotFoundError(f"No dataset_*_all_folds.csv in {pred_dir}")
    return files


def load_baseline(path: Path, label: str) -> pd.DataFrame:
    """Load a baseline predictions CSV.  Expected columns:
        PDB_ID, Chain_ID, Residue_Number, Residue_Name, Predicted_pKa
    Returns a DataFrame with KEY_COLS + `<label>`.
    """
    df = pd.read_csv(path)
    rename_map = {
        "pdb_id": "PDB_ID", "PDB": "PDB_ID",
        "chain_id": "Chain_ID", "Chain": "Chain_ID",
        "residue_number": "Residue_Number", "ResID": "Residue_Number",
        "residue_name": "Residue_Name", "ResName": "Residue_Name",
        "pKa": "Predicted_pKa", "predicted_pka": "Predicted_pKa",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    needed = set(KEY_COLS + ["Predicted_pKa"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} ({label}): missing columns {missing}")
    df["PDB_ID"]   = df["PDB_ID"].astype(str)
    df["Chain_ID"] = df["Chain_ID"].astype(str)
    df["res_code"] = df["Residue_Name"].map(to_residue_code)
    out = df[KEY_COLS + ["Predicted_pKa"]].rename(columns={"Predicted_pKa": label})
    return out


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════

def add_delta_and_quartiles(df: pd.DataFrame,
                            bin_edges: list[float],
                            bin_labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["ref_pKa"]      = df["res_code"].map(REFERENCE_PKA)
    df["delta_pKa"]    = (df["True_pKa"] - df["ref_pKa"]).abs()
    df["quartile"]     = pd.cut(
        df["delta_pKa"], bins=bin_edges, labels=bin_labels,
        include_lowest=True, right=True,
    )
    return df


def evaluate_method(df: pd.DataFrame, method: str,
                    n_boot: int, seed: int) -> dict:
    sub = df.dropna(subset=["True_pKa", method])
    return {"method": method, **metrics_with_ci(
        sub["True_pKa"].values, sub[method].values,
        n_boot=n_boot, seed=seed,
    )}


def evaluate_by_residue(df: pd.DataFrame, methods: list[str],
                        n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for method in methods:
        for code in sorted(df["res_code"].dropna().unique()):
            sub = df[df["res_code"] == code].dropna(subset=["True_pKa", method])
            if len(sub) == 0:
                continue
            row = {"method": method, "residue": code,
                   **metrics_with_ci(sub["True_pKa"].values,
                                     sub[method].values,
                                     n_boot=n_boot, seed=seed)}
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_by_quartile(df: pd.DataFrame, methods: list[str],
                         bin_labels: list[str],
                         n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for method in methods:
        for q in bin_labels:
            sub = df[df["quartile"] == q].dropna(subset=["True_pKa", method])
            if len(sub) == 0:
                continue
            row = {"method": method, "quartile": q,
                   **metrics_with_ci(sub["True_pKa"].values,
                                     sub[method].values,
                                     n_boot=n_boot, seed=seed)}
            rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_baseline(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--baseline expects NAME=PATH, got: {arg!r}")
    name, path = arg.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-style ensemble evaluation of GAT pKa predictions")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Training results dir; will use predictions/dataset_*_all_folds.csv")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Direct path to a single predictions CSV")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save evaluation CSVs (default: <results-dir>/evaluation)")
    parser.add_argument("--label", type=str, default="GAT_ensemble",
                        help="Label for the GNN ensemble column (default GAT_ensemble)")
    parser.add_argument("--baseline", action="append", default=[],
                        type=parse_baseline, metavar="NAME=PATH",
                        help="Baseline predictions to merge in.  May be repeated.")
    parser.add_argument("--intersect-only", action="store_true",
                        help="Restrict every metric to residues with predictions "
                             "from EVERY method (paper's Figure-10 setup).")
    parser.add_argument("--bin-edges", type=float, nargs="+",
                        default=DEFAULT_BIN_EDGES,
                        help="|ΔpKa_exp| bin edges (default: 0 0.2 0.5 1.0 inf)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for 95%% CI (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.results_dir is None and args.predictions is None:
        parser.error("Provide --results-dir or --predictions")

    # ── 1. Load and aggregate GNN predictions ─────────────────────────────
    if args.predictions is not None:
        pred_files = [args.predictions]
        out_dir = args.out_dir or args.predictions.parent
    else:
        pred_files = discover_predictions(args.results_dir)
        out_dir = args.out_dir or (args.results_dir / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir         : {out_dir}")

    # Aggregate every dataset_*_all_folds.csv into one ensemble per residue.
    # If multiple radii are present, they each get their own column so the
    # user can decide which radius to compare; we also write a global
    # "<label>" column = mean over radii.
    gnn_frames: list[pd.DataFrame] = []
    radius_labels: list[str] = []
    for f in pred_files:
        ds_idx = "".join(c for c in f.stem.split("_")[1] if c.isdigit())
        col = f"{args.label}_r{ds_idx}" if len(pred_files) > 1 else args.label
        radius_labels.append(col)
        df = load_predictions_csv(f)
        gnn_frames.append(aggregate_ensemble(df, label=col))
        print(f"  Loaded {f.name}: {len(df):>6} rows -> {len(gnn_frames[-1])} unique residues ({col})")

    merged = gnn_frames[0]
    for g in gnn_frames[1:]:
        merged = merged.merge(g.drop(columns=["res_code"]),
                              on=KEY_COLS + ["True_pKa"], how="outer")
    if len(pred_files) > 1:
        merged[args.label] = merged[radius_labels].mean(axis=1)

    methods = [args.label]

    # ── 2. Merge baseline predictions ─────────────────────────────────────
    for name, path in args.baseline:
        b = load_baseline(path, label=name).drop(columns=["res_code"])
        before = len(merged)
        merged = merged.merge(b, on=KEY_COLS, how="left")
        n_present = merged[name].notna().sum()
        print(f"  Baseline {name:<10} from {path}: {n_present}/{before} residues matched")
        methods.append(name)

    if args.intersect_only and len(methods) > 1:
        before = len(merged)
        merged = merged.dropna(subset=methods)
        print(f"  --intersect-only: {len(merged)}/{before} residues kept "
              f"(all methods present)")

    # ── 3. Quartile assignment ────────────────────────────────────────────
    bin_labels = [f"Q{i+1}" for i in range(len(args.bin_edges) - 1)]
    merged = add_delta_and_quartiles(merged, args.bin_edges, bin_labels)

    # ── 4. Compute and save ───────────────────────────────────────────────
    overall = pd.DataFrame([
        evaluate_method(merged, m, args.n_boot, args.seed) for m in methods
    ])
    by_res  = evaluate_by_residue(merged, methods, args.n_boot, args.seed)
    by_quar = evaluate_by_quartile(merged, methods, bin_labels,
                                    args.n_boot, args.seed)

    overall.to_csv(out_dir / "summary_overall.csv",   index=False)
    by_res .to_csv(out_dir / "summary_by_residue.csv", index=False)
    by_quar.to_csv(out_dir / "summary_quartiles.csv",  index=False)
    merged.to_csv(out_dir / "ensemble_predictions.csv", index=False)

    # ── 5. Print a compact summary ────────────────────────────────────────
    pd.set_option("display.float_format", lambda v: f"{v:6.3f}")
    print("\n=== Overall ===")
    cols = ["method", "N", "MAE", "MAE_lo", "MAE_hi",
            "RMSE", "RMSE_lo", "RMSE_hi", "Pearson", "Pearson_lo", "Pearson_hi"]
    print(overall[cols].to_string(index=False))

    print("\n=== Quartiles (rolling MAE/RMSE) ===")
    print(by_quar[["method", "quartile", "N",
                   "MAE", "MAE_lo", "MAE_hi",
                   "RMSE", "RMSE_lo", "RMSE_hi"]].to_string(index=False))

    print("\n=== By residue type ===")
    print(by_res[["method", "residue", "N",
                  "MAE", "MAE_lo", "MAE_hi",
                  "RMSE", "RMSE_lo", "RMSE_hi"]].to_string(index=False))

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
