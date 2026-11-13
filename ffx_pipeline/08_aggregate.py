"""
08_aggregate.py

Score every experiment on the same residues and build ensembles.

  python 08_aggregate.py R/gnn_base R/hgb ... --ensemble best=gnn_a,gnn_b,hgb

Each run directory holds predictions.csv or fold_*.csv (07b / 07c).  Runs with
several seeds in one directory are averaged per residue.  Reports RMSE / MAE
overall, without the SNase cluster, and per residue type, next to the
references: residue-type mean (fit on training folds), model-compound pKa
(null) and PROPKA3.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"]
MODEL_PKA = {"Aspartate": 3.94, "Glutamate": 4.4, "Histidine": 6.6, "Lysine": 10.4,
             "Cysteine": 8.55, "Tyrosine": 10.07}


def load(d: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(d / "fold_*.csv"))) or [str(d / "predictions.csv")]
    df = pd.concat([pd.read_csv(f, dtype={"Chain_ID": str}) for f in files], ignore_index=True)
    return df.groupby(KEY, as_index=False).agg(
        True_pKa=("True_pKa", "first"), Predicted_pKa=("Predicted_pKa", "mean"),
        Baseline_pKa=("Baseline_pKa", "first"), PROPKA_pKa=("PROPKA_pKa", "first"),
        Cluster=("Cluster", "first"), fold=("fold", "first"))


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def row(name: str, df: pd.DataFrame, col: str, snase: str) -> dict:
    e = df[col] - df["True_pKa"]
    ok = e.notna()
    out = {"model": name, "n": int(ok.sum()), "RMSE": rmse(e[ok]), "MAE": float(e[ok].abs().mean()),
           "RMSE_noSNase": rmse(e[ok & (df.Cluster != snase)])}
    for r, s in (("Aspartate", "ASP"), ("Glutamate", "GLU"), ("Histidine", "HIS"),
                 ("Lysine", "LYS"), ("Cysteine", "CYS"), ("Tyrosine", "TYR")):
        out[s] = rmse(e[ok & (df.Residue_Name == r)])
    # per-fold spread: is a difference between models larger than fold noise?
    out["fold_sd"] = float(np.std([rmse(e[ok & (df.fold == f)]) for f in sorted(df.fold.unique())]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ensemble", nargs="*", default=[], help="name=run1,run2,...  (simple average)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs = {d.name: load(d) for d in args.runs if glob.glob(str(d / "*.csv"))}
    first = next(iter(runs.values()))
    snase = first.Cluster.value_counts().index[0]
    first = first.assign(Null_pKa=first.Residue_Name.map(MODEL_PKA))
    rows = [row("residue-type mean", first, "Baseline_pKa", snase),
            row("model-compound pKa", first, "Null_pKa", snase),
            row("PROPKA3", first, "PROPKA_pKa", snase)]
    for name, df in runs.items():
        if len(df) != len(first):
            print(f"WARNING {name}: {len(df)} residues vs {len(first)} (incomplete folds?)")
        rows.append(row(name, df, "Predicted_pKa", snase))
    for spec in args.ensemble:
        name, members = spec.split("=")
        members = [m for m in members.split(",") if m in runs]
        if len(members) < 2:
            continue
        m = runs[members[0]][KEY + ["True_pKa", "Cluster", "fold", "Predicted_pKa"]].rename(
            columns={"Predicted_pKa": "p0"})
        for i, mem in enumerate(members[1:], 1):
            m = m.merge(runs[mem][KEY + ["Predicted_pKa"]].rename(columns={"Predicted_pKa": f"p{i}"}), on=KEY)
        m["Predicted_pKa"] = m[[f"p{i}" for i in range(len(members))]].mean(1)
        rows.append(row(f"ENS {name} ({len(members)})", m, "Predicted_pKa", snase))
        if args.out:
            m.to_csv(args.out.with_name(f"ensemble_{name}.csv"), index=False)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        tab.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
