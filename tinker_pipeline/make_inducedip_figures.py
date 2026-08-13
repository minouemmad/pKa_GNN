from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "tinker_pipeline" / "Graph_pKa"
DS_PKL    = SWEEP_DIR / "Features_FFX138" / "Datasets_InducedDip_138" / "data_list_2.pkl"
ELECTRO   = SWEEP_DIR / "Net_FFX138_Electro"
BASELINE_CSV = ROOT / "ffx_pipeline" / "Graph_pKa" / "Results" \
                    / "Training_rotopt_naive_full138_allR" / "predictions" \
                    / "dataset_2_all_folds.csv"  # r=9, 343 rows, has CYS+TYR
OUT       = ROOT / "Graph_pKa" / "Presentation_FFX"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT = "InducedDip"     # winning variant (best mean MAE in sweep)
SEEDS   = [1, 7, 17, 42, 99, 123, 314, 2026]
HYBRID_RES = ("Cysteine", "Tyrosine")   # pulled from baseline (no variant data)

RES_SHORT = {
    "Aspartate": "ASP", "Glutamate": "GLU", "Histidine": "HIS",
    "Lysine":    "LYS", "Tyrosine":  "TYR", "Cysteine":  "CYS",
}
RES_COLORS = {
    "ASP": "#d62728", "GLU": "#ff7f0e", "HIS": "#2ca02c",
    "LYS": "#9467bd", "TYR": "#1f77b4", "CYS": "#8c564b",
}

# ─── reconstruct deterministic fold mapping ──────────────────────────────────
print("[1] reconstructing KFold(10, shuffle=True, random_state=42) …")
with open(DS_PKL, "rb") as f:
    data_list = pickle.load(f)
print(f"   dataset has {len(data_list)} graphs")

# meta per dataset index
meta = pd.DataFrame({
    "ds_idx":         range(len(data_list)),
    "PDB_ID":         [d.PDB_ID         for d in data_list],
    "Chain_ID":       [d.Chain_ID       for d in data_list],
    "Residue_Number": [d.Residue_Number for d in data_list],
    "Residue_Name":   [d.Residue_Name   for d in data_list],
    "True_pKa":       [float(d.y.item()) for d in data_list],
})

# Reproduce the exact KFold split used in training: rows in all_folds.csv are
# emitted in fold order 1..10, and within each fold in the val_idx order
# returned by sklearn's KFold(shuffle=True, random_state=42).
kf = KFold(n_splits=10, shuffle=True, random_state=42)
ordered_idx: list[int] = []
ordered_fold: list[int] = []
for fold, (_, val_idx) in enumerate(kf.split(np.arange(len(data_list))), start=1):
    ordered_idx.extend(val_idx.tolist())
    ordered_fold.extend([fold] * len(val_idx))
print(f"   reconstructed order length = {len(ordered_idx)}")

# ─── load InducedDip predictions, average across seeds, attach residue ───────
print(f"[2] loading {VARIANT} predictions across {len(SEEDS)} seeds …")
seed_dfs = []
for s in SEEDS:
    f = ELECTRO / f"Training_{VARIANT}_seed{s}" / "predictions" / "dataset_2_all_folds.csv"
    if not f.exists():
        print(f"   missing: {f}")
        continue
    d = pd.read_csv(f)
    d["seed"] = s
    seed_dfs.append(d)

all_seeds = pd.concat(seed_dfs, ignore_index=True)
n_per_seed = len(seed_dfs[0])
assert n_per_seed == len(ordered_idx), f"{n_per_seed} vs {len(ordered_idx)}"

# Sanity check: True_pKa in the CSV at row k should equal True_pKa for
# data_list[ordered_idx[k]] (within float tolerance).
true_check = np.array([data_list[i].y.item() for i in ordered_idx])
csv_true   = seed_dfs[0]["True_pKa"].values
mismatch   = np.abs(true_check - csv_true) > 1e-3
if mismatch.any():
    print(f"   WARNING: {mismatch.sum()}/{len(mismatch)} True_pKa mismatches "
          f"— fold mapping may be wrong")
else:
    print("   ✓ KFold mapping verified against True_pKa")

# attach Residue_Name + meta to every (row × seed) entry
row2idx = {k: ordered_idx[k] for k in range(n_per_seed)}
all_seeds["row"] = all_seeds.groupby("seed").cumcount()
all_seeds["ds_idx"] = all_seeds["row"].map(row2idx)
all_seeds = all_seeds.merge(
    meta[["ds_idx", "PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"]],
    on="ds_idx", how="left",
)

# average prediction per residue across the 8 seeds (each residue appears
# exactly once per seed, in exactly one fold)
mean_pred = (
    all_seeds.groupby("ds_idx")
             .agg(True_pKa=("True_pKa","first"),
                  Predicted_pKa=("Predicted_pKa","mean"),
                  fold=("fold","first"),
                  Residue_Name=("Residue_Name","first"))
             .reset_index()
)
print(f"   per-residue mean across seeds: {len(mean_pred)} rows")
mean_pred["source"] = VARIANT

# ─── pull CYS + TYR from BASELINE (Charge / naive rotopt allR) at r=9 ────────
print(f"[2b] adding {HYBRID_RES} from baseline {BASELINE_CSV.name} …")
base = pd.read_csv(BASELINE_CSV)
base_extra = base[base["Residue_Name"].isin(HYBRID_RES)][
    ["True_pKa", "Predicted_pKa", "fold", "Residue_Name"]
].copy()
base_extra["ds_idx"] = -1   # marker — not from variant pickle
base_extra["source"] = "Charge_baseline"
print(f"   baseline rows for {HYBRID_RES}: {len(base_extra)} "
      f"({base_extra['Residue_Name'].value_counts().to_dict()})")
mean_pred = pd.concat([mean_pred, base_extra], ignore_index=True)
mean_pred.to_csv(OUT / "InducedDip_per_residue_predictions.csv", index=False)

# ─── per-residue table ────────────────────────────────────────────────────────
def per_residue_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = df["Predicted_pKa"] - df["True_pKa"]
    return (
        df.groupby("Residue_Name")["err"]
          .agg(MAE=lambda x: x.abs().mean(),
               RMSE=lambda x: np.sqrt((x**2).mean()),
               bias=lambda x: x.mean(),
               n="size")
          .sort_values("MAE")
    )

per_res = per_residue_table(mean_pred)
per_res.to_csv(OUT / "InducedDip_per_residue_table.csv")
print("\n[3] per-residue (mean across 8 seeds, 10-fold InducedDip):")
print(per_res)
overall_mae  = (mean_pred["Predicted_pKa"] - mean_pred["True_pKa"]).abs().mean()
overall_rmse = np.sqrt(((mean_pred["Predicted_pKa"] - mean_pred["True_pKa"])**2).mean())
print(f"\noverall — n={len(mean_pred)}  MAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}")

# ─── figure: 3-panel per-residue (matches old 03_rotopt_per_residue layout) ──
print("\n[4] writing 03b_rotopt_per_residue_InducedDip.png …")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) per-residue MAE bar
ax = axes[0]
xs     = [RES_SHORT.get(r, r) for r in per_res.index]
colors = [RES_COLORS.get(s, "#888") for s in xs]
hatches = ["///" if r in HYBRID_RES else "" for r in per_res.index]
bars = ax.bar(xs, per_res["MAE"], color=colors, edgecolor="black", hatch=hatches)
for bar, n in zip(bars, per_res["n"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.03,
            f"n={n}", ha="center", fontsize=9)
ax.axhline(overall_mae, color="black", linestyle="--",
           label=f"overall MAE = {overall_mae:.3f}")
# legend entry for hatched bars
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="white", edgecolor="black",
          label=f"overall MAE = {overall_mae:.3f}", linestyle="--"),
    Patch(facecolor="lightgray", edgecolor="black", hatch="///",
          label="CYS/TYR (baseline only)"),
], fontsize=8, loc="upper left")
ax.set_ylabel("MAE (pKa units)")
ax.set_title("Per-residue MAE — InducedDip @ r = 9 Å (mean of 8 seeds)\n"
             "CYS/TYR shown from Charge baseline (excluded from variant scope)",
             fontsize=10)
ax.grid(axis="y", alpha=0.3)

# (b) signed-error violin
ax = axes[1]
data, labs = [], []
for res in per_res.index:
    sub = mean_pred[mean_pred["Residue_Name"] == res]
    data.append((sub["Predicted_pKa"] - sub["True_pKa"]).values)
    labs.append(RES_SHORT.get(res, res))
parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
for pc, lab in zip(parts["bodies"], labs):
    pc.set_facecolor(RES_COLORS.get(lab, "#888"))
    pc.set_alpha(0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(1, len(labs)+1)); ax.set_xticklabels(labs)
ax.set_ylabel("Predicted − Experimental pKa")
ax.set_title("Signed-error distribution (bias + spread)")
ax.grid(axis="y", alpha=0.3)

# (c) predicted-vs-experimental scatter
ax = axes[2]
for res in per_res.index:
    sub = mean_pred[mean_pred["Residue_Name"] == res]
    short = RES_SHORT.get(res, res)
    ax.scatter(sub["True_pKa"], sub["Predicted_pKa"],
               s=24, alpha=0.7, color=RES_COLORS.get(short, "#888"),
               label=f"{short} (n={len(sub)})")
lo = min(mean_pred["True_pKa"].min(), mean_pred["Predicted_pKa"].min()) - 0.5
hi = max(mean_pred["True_pKa"].max(), mean_pred["Predicted_pKa"].max()) + 0.5
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Experimental pKa"); ax.set_ylabel("Predicted pKa")
ax.set_title("Predicted vs experimental")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")

plt.suptitle(
    f"FFX rotopt + InducedDip features (best feature-engineering variant; "
    f"ΔMAE vs Charge baseline = −0.004, p = 0.38, NOT significant)",
    fontsize=10, y=1.02, color="#444"
)
plt.tight_layout()
plt.savefig(OUT / "03b_rotopt_per_residue_InducedDip.png",
            dpi=170, bbox_inches="tight")
plt.close()

# ─── note: radius sweep was r=9 only for variants ────────────────────────────
print("[5] writing radius-sweep note …")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")
ax.text(0.02, 0.95,
        "Radius sweep with feature variants — not run.",
        fontsize=14, weight="bold", va="top", color="#444")
ax.text(0.02, 0.78,
        "The feature-engineering sweep (10 variants × 8 seeds × 10 folds)\n"
        "was deliberately r = 9 Å only to keep total training cost tractable.\n"
        "Re-running the radius sweep with InducedDip across r ∈ {7, 8, 10, 11}\n"
        "would mean ~80 additional fold-trainings (≈ 1 sweep worth of compute).",
        fontsize=10, va="top")
ax.text(0.02, 0.30,
        "Honest caveat: even at r = 9 Å, InducedDip vs Charge baseline\n"
        "Wilcoxon ΔMAE = −0.0041, p = 0.38 — NOT significant.\n"
        "The radius curve for the baseline (figure 02) is therefore likely\n"
        "representative of any variant within ~0.01 pKa units.",
        fontsize=9.5, va="top", color="#a33")
plt.savefig(OUT / "02b_radius_note.png", dpi=170, bbox_inches="tight")
plt.close()

print("\nDONE.")
for p in sorted(OUT.glob("0[23]b_*")):
    print("  ", p.name)
