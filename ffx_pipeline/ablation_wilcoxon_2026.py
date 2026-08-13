from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(r"C:\Users\maemm\OneDrive\Desktop\FFX\pKa_GNN")
COMP = ROOT / "Graph_pKa" / "Results" / "Comparison_2026"
OUT  = COMP / "_analysis"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"]

RUNS = {
    "rotopt": {
        "full":   "_rotopt_full_r10_tuned",
        "noDip":  "_rotopt_full_r10_tuned_ablate_induced_dipoles",
        "noPerm": "_rotopt_full_r10_tuned_ablate_perm_multipoles",
        "noElec": "_rotopt_full_r10_tuned_ablate_noElec",
    },
    "titrate": {
        "full":   "_titrate_full_r10_tuned",
        "noDip":  "_titrate_full_r10_tuned_ablate_induced_dipoles",
        "noPerm": "_titrate_full_r10_tuned_ablate_perm_multipoles",
        "noElec": "_titrate_full_r10_tuned_ablate_noElec",
    },
}

def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "predictions" / "dataset_3_all_folds.csv")
    df["abs_err"] = (df["Predicted_pKa"] - df["True_pKa"]).abs()
    # average over pH replicates (titrate) -> one row per residue
    g = df.groupby(KEYS, as_index=False)["abs_err"].mean()
    return g

rows = []
for mode, runs in RUNS.items():
    base = load(COMP / runs["full"])
    for cond in ("noDip", "noPerm", "noElec"):
        abl = load(COMP / runs[cond])
        merged = base.merge(abl, on=KEYS, suffixes=("_full", "_abl"))
        d = merged["abs_err_abl"] - merged["abs_err_full"]  # positive => ablation worse
        n = len(d)
        mean_d = d.mean()
        try:
            stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        # bootstrap 95% CI on mean delta
        rng = np.random.default_rng(42)
        boots = np.array([rng.choice(d.values, size=n, replace=True).mean()
                          for _ in range(5000)])
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        mae_full = merged["abs_err_full"].mean()
        mae_abl  = merged["abs_err_abl"].mean()
        rows.append({
            "mode": mode, "contrast": f"{cond} - full",
            "n_residues": n,
            "MAE_full": round(mae_full, 4),
            "MAE_abl":  round(mae_abl, 4),
            "delta_MAE": round(mean_d, 4),
            "CI95_lo": round(ci_lo, 4),
            "CI95_hi": round(ci_hi, 4),
            "wilcoxon_W": round(float(stat), 1) if not np.isnan(stat) else "NA",
            "p_value": round(float(p), 4) if not np.isnan(p) else "NA",
        })

out_df = pd.DataFrame(rows)
out_csv = OUT / "wilcoxon_ablation_2026.csv"
out_df.to_csv(out_csv, index=False)
print(out_df.to_string(index=False))
print(f"\nSaved -> {out_csv}")
