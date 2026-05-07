"""Rebuild the FFX-only figure pack for the GAT pKa project.

Sources used (all FFX rotopt or FFX titrate-rotopt — no Tinker):

  • ffx_pipeline/Graph_pKa/Results/Training_rotopt_naive_full138_allR
        full 138-PDB rotopt run, predictions for radii 7..11 Å
  • ffx_pipeline/Graph_pKa/Results/Training_titrate_naive_group-residue
        full ~198-residue titration run with proper residue-grouped CV
  • ffx_pipeline/Graph_pKa/Results/Training_titrate_film_group-residue
        same data, FiLM pH conditioning
  • tinker_pipeline/Graph_pKa/Net_FFX138_{Electro,PhysEdge}/*.csv
        electrostatics + physedge sweeps — these are FFX-rotopt-prepped graphs,
        the directory name is misleading.  All node/edge feats come from FFX.

All outputs land in `Graph_pKa/Presentation_FFX/`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
FFX_RES   = ROOT / "ffx_pipeline" / "Graph_pKa" / "Results"
SWEEP_DIR = ROOT / "tinker_pipeline" / "Graph_pKa"
OUT       = ROOT / "Graph_pKa" / "Presentation_FFX"
OUT.mkdir(parents=True, exist_ok=True)

RADII = [7, 8, 9, 10, 11]

# Residue-name mapping for short tick labels on figures
RES_SHORT = {
    "Aspartate":  "ASP",
    "Glutamate":  "GLU",
    "Histidine":  "HIS",
    "Lysine":     "LYS",
    "Tyrosine":   "TYR",
    "Cysteine":   "CYS",
}
RES_COLORS = {
    "ASP": "#d62728",
    "GLU": "#ff7f0e",
    "HIS": "#2ca02c",
    "LYS": "#9467bd",
    "TYR": "#1f77b4",
    "CYS": "#8c564b",
}

# ────────────────────────────── helpers ──────────────────────────────────────
def fold_metrics(df: pd.DataFrame) -> dict:
    err = (df["Predicted_pKa"] - df["True_pKa"])
    g = (
        df.assign(_e=err)
          .groupby("fold")["_e"]
          .agg(MAE=lambda x: x.abs().mean(),
               RMSE=lambda x: np.sqrt((x**2).mean()))
    )
    return {
        "MAE":  g["MAE"].mean(),
        "MAE_sd": g["MAE"].std(ddof=1),
        "RMSE": g["RMSE"].mean(),
        "RMSE_sd": g["RMSE"].std(ddof=1),
        "n_folds": int(g.shape[0]),
        "n_rows":  int(df.shape[0]),
    }

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


# ════════════════════════════════════════════════════════════════════════════
# 1.  GAT architecture diagram
# ════════════════════════════════════════════════════════════════════════════
print("[1] GAT architecture …")
fig, ax = plt.subplots(figsize=(13, 5.0))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, w, h, label, color, fontsize=10):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.08,rounding_size=0.15",
                       linewidth=1.4, facecolor=color, edgecolor="black")
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.2, color="black")
    ax.add_patch(a)

box(0.1, 1.4, 2.6, 2.2,
    "Per-atom node features\n(local graph, r = 9 Å)\n\n"
    "• 9-d atom-label OHE\n"
    "• 4-d residue-type OHE\n"
    "• atomic charge (q)\n"
    "• SASA\n"
    "• H-bond donor / acceptor\n"
    "• radius shell counts\n"
    "• local-frame coords (x,y,z)\n"
    "• AMOEBA dipoles\n"
    "  (induced & permanent)\n"
    "• dipole invariants*",
    "#dbe9f4", fontsize=8)
arrow(2.8, 2.5, 3.4, 2.5)

box(3.5, 1.7, 3.0, 1.6,
    "GATv2Conv\nhidden = 48\nheads = 4 (concat → 192)\n"
    "edge_dim = optional\nadd_self_loops = False\n"
    "→ ReLU → Dropout 0.5",
    "#fce8b2", fontsize=9)
arrow(6.6, 2.5, 7.4, 2.5)

box(7.5, 2.0, 2.0, 1.0,
    "global_mean_pool\n(per residue graph)",
    "#f9d7c1", fontsize=9)
arrow(9.6, 2.5, 10.4, 2.5)

box(10.5, 2.0, 2.2, 1.0,
    "Linear(192 → 1)\npredicted pKa",
    "#cde6c4", fontsize=10)

box(3.5, 0.1, 3.0, 1.1,
    "edge_attr (optional)\nq·μ_j projections\nμ_i·μ_j tensor\n(induced + perm)",
    "#e0e0e0", fontsize=8)
arrow(5.0, 1.2, 5.0, 1.7)

ax.text(6.5, 4.6, "GATv2 model for residue pKa on FFX rotopt graphs",
        ha="center", fontsize=14, weight="bold")
ax.text(6.5, 4.15,
        "1 message-passing layer · mean-pool · linear head · MSE loss · Adam(lr=1e-2) · 10-fold CV",
        ha="center", fontsize=10, color="#444")
ax.text(0.1, 0.45,
        "* invariant scalars: ‖μ‖, μ·z̑, μ·E_neigh — for both induced and permanent dipoles",
        fontsize=8, color="#444")

plt.savefig(OUT / "01_gat_architecture.png", dpi=170, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 2.  rotopt full138 — radius sweep (FFX only)
# ════════════════════════════════════════════════════════════════════════════
print("[2] rotopt radius sweep …")
rotopt_dir = FFX_RES / "Training_rotopt_naive_full138_allR" / "predictions"
rows = []
for i, r in enumerate(RADII):
    f = rotopt_dir / f"dataset_{i}_all_folds.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    m = fold_metrics(df); m["radius"] = r
    rows.append(m)
rad_df = pd.DataFrame(rows)
rad_df.to_csv(OUT / "rotopt_radius_sweep.csv", index=False)
print(rad_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.errorbar(rad_df["radius"], rad_df["MAE"], yerr=rad_df["MAE_sd"],
            marker="o", color="#dd8452", linewidth=2, capsize=4, label="MAE")
ax.errorbar(rad_df["radius"], rad_df["RMSE"], yerr=rad_df["RMSE_sd"],
            marker="s", color="#4c72b0", linewidth=2, capsize=4, label="RMSE")
for x, y in zip(rad_df["radius"], rad_df["MAE"]):
    ax.text(x, y - 0.06, f"{y:.3f}", ha="center", fontsize=8, color="#dd8452")
for x, y in zip(rad_df["radius"], rad_df["RMSE"]):
    ax.text(x, y + 0.04, f"{y:.3f}", ha="center", fontsize=8, color="#4c72b0")
ax.set_xticks(RADII)
ax.set_xlabel("Graph cutoff radius (Å)")
ax.set_ylabel("Error (pKa units)")
ax.set_title(f"FFX rotopt — radius sweep (138 PDBs, n = {rad_df['n_rows'].iloc[0]} residues, 10-fold)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_rotopt_radius.png", dpi=170, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 3.  rotopt — per-residue analysis at r=9 Å
# ════════════════════════════════════════════════════════════════════════════
print("[3] rotopt per-residue …")
rotopt_r9 = pd.read_csv(rotopt_dir / "dataset_2_all_folds.csv")
per_res = per_residue_table(rotopt_r9)
per_res.to_csv(OUT / "rotopt_per_residue.csv")
print(per_res)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) per-residue MAE bar
ax = axes[0]
xs = [RES_SHORT.get(r, r) for r in per_res.index]
colors = [RES_COLORS.get(s, "#888") for s in xs]
bars = ax.bar(xs, per_res["MAE"], color=colors, edgecolor="black")
for bar, n in zip(bars, per_res["n"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.03,
            f"n={n}", ha="center", fontsize=9)
overall = (rotopt_r9["Predicted_pKa"] - rotopt_r9["True_pKa"]).abs().mean()
ax.axhline(overall, color="black", linestyle="--",
           label=f"overall MAE = {overall:.3f}")
ax.set_ylabel("MAE (pKa units)")
ax.set_title("Per-residue MAE — FFX rotopt @ r = 9 Å")
ax.grid(axis="y", alpha=0.3)
ax.legend()

# (b) signed-error violin
ax = axes[1]
data = []; labs = []
for res in per_res.index:
    sub = rotopt_r9[rotopt_r9["Residue_Name"] == res]
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
    sub = rotopt_r9[rotopt_r9["Residue_Name"] == res]
    short = RES_SHORT.get(res, res)
    ax.scatter(sub["True_pKa"], sub["Predicted_pKa"],
               s=24, alpha=0.7, color=RES_COLORS.get(short, "#888"),
               label=f"{short} (n={len(sub)})")
lo = min(rotopt_r9["True_pKa"].min(), rotopt_r9["Predicted_pKa"].min()) - 0.5
hi = max(rotopt_r9["True_pKa"].max(), rotopt_r9["Predicted_pKa"].max()) + 0.5
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Experimental pKa"); ax.set_ylabel("Predicted pKa")
ax.set_title("Predicted vs experimental")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(OUT / "03_rotopt_per_residue.png", dpi=170, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 4.  rotopt — error vs experimental pKa (calibration)
# ════════════════════════════════════════════════════════════════════════════
print("[4] rotopt error calibration …")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
err = rotopt_r9["Predicted_pKa"] - rotopt_r9["True_pKa"]
for res in per_res.index:
    sub = rotopt_r9[rotopt_r9["Residue_Name"] == res]
    short = RES_SHORT.get(res, res)
    ax.scatter(sub["True_pKa"], sub["Predicted_pKa"] - sub["True_pKa"],
               s=24, alpha=0.7, color=RES_COLORS.get(short, "#888"), label=short)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.axhline(+1, color="red", linestyle=":", linewidth=0.8)
ax.axhline(-1, color="red", linestyle=":", linewidth=0.8)
ax.set_xlabel("Experimental pKa")
ax.set_ylabel("Residual (predicted − experimental)")
ax.set_title("Where does the model fail? — residual vs experimental pKa")
ax.legend(fontsize=8, ncol=2, loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "04_rotopt_residuals.png", dpi=170, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 5.  Feature-engineering sweep (FFX-rotopt prepped graphs)
# ════════════════════════════════════════════════════════════════════════════
print("[5] feature-engineering sweep …")
electro_pf = pd.read_csv(SWEEP_DIR / "Net_FFX138_Electro" /
                         "sweep_electrostatics_138_perfold.csv")
phys_pf    = pd.read_csv(SWEEP_DIR / "Net_FFX138_PhysEdge" /
                         "sweep_physedge_138_perfold.csv")
electro_sm = pd.read_csv(SWEEP_DIR / "Net_FFX138_Electro" /
                         "sweep_electrostatics_138_summary.csv")
phys_sm    = pd.read_csv(SWEEP_DIR / "Net_FFX138_PhysEdge" /
                         "sweep_physedge_138_summary.csv")
phys_pf_no_charge = phys_pf[phys_pf["variant"] != "Charge"]
phys_sm_no_charge = phys_sm[phys_sm["variant"] != "Charge"]
combo_pf = pd.concat([electro_pf, phys_pf_no_charge], ignore_index=True)
combo_sm = pd.concat([electro_sm, phys_sm_no_charge], ignore_index=True)
combo_sm = combo_sm.sort_values("mean_MAE").reset_index(drop=True)
combo_sm.to_csv(OUT / "feature_engineering_summary.csv", index=False)
print(combo_sm[["variant","mean_MAE","mean_delta_MAE","wilcoxon_p"]].to_string(index=False))

variant_order = combo_sm["variant"].tolist()
charge_baseline = float(combo_sm.loc[combo_sm["variant"] == "Charge", "mean_MAE"].iloc[0])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                         gridspec_kw={"width_ratios": [1.2, 1.0]})

# (a) per-fold MAE strip + box
ax = axes[0]
data = [combo_pf[combo_pf["variant"] == v]["MAE"].values for v in variant_order]
bp = ax.boxplot(data, tick_labels=variant_order,
                patch_artist=True, widths=0.55, showfliers=False)
palette10 = plt.cm.tab10(np.linspace(0, 1, len(variant_order)))
for patch, c in zip(bp["boxes"], palette10):
    patch.set_facecolor(c); patch.set_alpha(0.6)
# Overlay individual fold-MAEs as jittered points
rng = np.random.default_rng(0)
for i, vals in enumerate(data, start=1):
    jit = rng.normal(0, 0.08, size=len(vals))
    ax.scatter(np.full_like(vals, i, dtype=float) + jit, vals,
               s=8, color="black", alpha=0.25)
ax.axhline(charge_baseline, color="red", linestyle="--",
           label=f"Charge baseline = {charge_baseline:.3f}")
ax.set_ylabel("MAE per fold (pKa units)")
ax.set_title("Feature variants on FFX rotopt — 10 variants × 8 seeds × 10 folds = 80 fold-MAEs each")
ax.tick_params(axis="x", rotation=35)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# (b) ΔMAE bar with Wilcoxon p
ax = axes[1]
delta = combo_sm.set_index("variant").loc[variant_order]
delta_no_charge = delta.drop(index="Charge")
ypos = np.arange(len(delta_no_charge))
bars = ax.barh(ypos, delta_no_charge["mean_delta_MAE"],
               color=["#2ca02c" if v < 0 else "#d62728"
                      for v in delta_no_charge["mean_delta_MAE"]],
               edgecolor="black")
ax.set_yticks(ypos); ax.set_yticklabels(delta_no_charge.index)
for bar, p in zip(bars, delta_no_charge["wilcoxon_p"]):
    if pd.isna(p):
        s = ""
    elif p < 0.05:
        s = f"p={p:.3f}*"
    else:
        s = f"p={p:.2f}"
    x = bar.get_width()
    off = 0.0004 if x >= 0 else -0.0004
    ha = "left" if x >= 0 else "right"
    ax.text(x + off, bar.get_y() + bar.get_height()/2, s,
            va="center", ha=ha, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("ΔMAE vs Charge baseline   ←   better       worse   →")
ax.set_title("Paired Wilcoxon vs baseline (80 fold-pairs each)")
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUT / "05_feature_engineering.png", dpi=170, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 6.  Permutation importance (FFX-rotopt graphs)
# ════════════════════════════════════════════════════════════════════════════
print("[6] permutation importance …")
imp_paths = [
    ("Charge baseline",  SWEEP_DIR / "Net_FFX138_PhysEdge" / "perm_importance_Charge.csv"),
    ("Invariant",        SWEEP_DIR / "Net_FFX138_PhysEdge" / "perm_importance_Invariant.csv"),
    ("InducedDip",       SWEEP_DIR / "Net_FFX138_Electro"  / "perm_importance_InducedDip.csv"),
    ("BothDip",          SWEEP_DIR / "Net_FFX138_Electro"  / "perm_importance_BothDip.csv"),
]
imp_data = {}
for name, p in imp_paths:
    if not p.exists():
        continue
    d = pd.read_csv(p)
    feat_col = "feature" if "feature" in d.columns else d.columns[0]
    val_col  = "delta_MAE" if "delta_MAE" in d.columns else d.columns[1]
    imp_data[name] = d.set_index(feat_col)[val_col]

imp_df = pd.DataFrame(imp_data).fillna(0)
# Order rows by max importance ascending so biggest bars sit on top
order = imp_df.max(axis=1).sort_values(ascending=True).index
imp_df = imp_df.loc[order]
imp_df.to_csv(OUT / "perm_importance.csv")

fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42*len(imp_df))))
y = np.arange(len(imp_df))
bar_h = 0.78 / len(imp_df.columns)
palette = plt.cm.tab10(np.arange(len(imp_df.columns)))
for i, col in enumerate(imp_df.columns):
    ax.barh(y + i*bar_h - 0.39 + bar_h/2, imp_df[col],
            height=bar_h, color=palette[i], edgecolor="black",
            linewidth=0.5, label=col)
ax.set_yticks(y); ax.set_yticklabels(imp_df.index)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("ΔMAE on shuffle  (larger = model relies on this feature group more)")
ax.set_title("Permutation importance — FFX rotopt features (138 PDBs)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "06_perm_importance.png", dpi=170, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 7.  TITRATION — full ~198-residue completed set, residue-grouped CV
# ════════════════════════════════════════════════════════════════════════════
print("[7] titration full set …")
tit_runs = {
    "rotopt-only (138 PDBs)":
        FFX_RES / "Training_rotopt_naive_full138" / "predictions" / "dataset_0_all_folds.csv",
    "titrate (naive, group-residue CV)":
        FFX_RES / "Training_titrate_naive_group-residue" / "predictions" / "dataset_0_all_folds.csv",
    "titrate (FiLM,  group-residue CV)":
        FFX_RES / "Training_titrate_film_group-residue"  / "predictions" / "dataset_0_all_folds.csv",
}
tit_summary = []
tit_dfs = {}
for name, path in tit_runs.items():
    if not path.exists():
        continue
    df = pd.read_csv(path)
    m  = fold_metrics(df)
    tit_summary.append({"run": name, **m, "n_residues": df[["PDB_ID","Chain_ID","Residue_Number"]].drop_duplicates().shape[0]})
    tit_dfs[name] = df
tit_summary = pd.DataFrame(tit_summary)
tit_summary.to_csv(OUT / "titration_summary.csv", index=False)
print(tit_summary.to_string(index=False))

# (a) overall MAE/RMSE bar
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
ax = axes[0]
labels = tit_summary["run"].tolist()
xs = np.arange(len(labels))
w = 0.38
ax.bar(xs - w/2, tit_summary["MAE"], width=w,
       yerr=tit_summary["MAE_sd"], capsize=4,
       color="#dd8452", edgecolor="black", label="MAE")
ax.bar(xs + w/2, tit_summary["RMSE"], width=w,
       yerr=tit_summary["RMSE_sd"], capsize=4,
       color="#4c72b0", edgecolor="black", label="RMSE")
for i, (m, r) in enumerate(zip(tit_summary["MAE"], tit_summary["RMSE"])):
    ax.text(i - w/2, m + 0.04, f"{m:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, r + 0.04, f"{r:.3f}", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=10, ha="right")
ax.set_ylabel("Error (pKa units)")
ax.set_title("Titration rotop vs plain rotopt (10-fold, residue-grouped CV)")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)

# (b) per-residue MAE for the best titrate variant vs rotopt baseline
ax = axes[1]
key_titrate = "titrate (FiLM,  group-residue CV)"
key_rot     = "rotopt-only (138 PDBs)"
if key_titrate in tit_dfs and key_rot in tit_dfs:
    pr_t = per_residue_table(tit_dfs[key_titrate])
    pr_r = per_residue_table(tit_dfs[key_rot])
    common = sorted(set(pr_t.index) & set(pr_r.index),
                    key=lambda r: pr_r["MAE"].get(r, 99))
    xs = np.arange(len(common))
    w = 0.4
    ax.bar(xs - w/2, [pr_r["MAE"].get(r, 0) for r in common],
           width=w, color="#888888", edgecolor="black", label="rotopt-only")
    ax.bar(xs + w/2, [pr_t["MAE"].get(r, 0) for r in common],
           width=w, color="#55a868", edgecolor="black", label="titrate-FiLM")
    ax.set_xticks(xs)
    ax.set_xticklabels([RES_SHORT.get(r, r) for r in common])
    ax.set_ylabel("MAE (pKa units)")
    ax.set_title("Per-residue MAE — rotopt vs titrate-FiLM")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, r in enumerate(common):
        ax.text(i - w/2, pr_r["MAE"].get(r, 0) + 0.03,
                f"n={int(pr_r['n'].get(r, 0))}", ha="center", fontsize=7)
        ax.text(i + w/2, pr_t["MAE"].get(r, 0) + 0.03,
                f"n={int(pr_t['n'].get(r, 0))}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig(OUT / "07_titration_overall.png", dpi=170, bbox_inches="tight")
plt.close()

# (c) Titration scatter: predicted vs experimental, replicate clouds per residue
print("[7b] titration scatter …")
key = "titrate (FiLM,  group-residue CV)"
df = tit_dfs.get(key)
if df is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    # left: scatter coloured by residue
    ax = axes[0]
    pr = per_residue_table(df)
    for res in pr.index:
        sub = df[df["Residue_Name"] == res]
        short = RES_SHORT.get(res, res)
        ax.scatter(sub["True_pKa"], sub["Predicted_pKa"],
                   s=18, alpha=0.55, color=RES_COLORS.get(short, "#888"),
                   label=f"{short} (n={len(sub)})")
    lo = min(df["True_pKa"].min(), df["Predicted_pKa"].min()) - 0.5
    hi = max(df["True_pKa"].max(), df["Predicted_pKa"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental pKa"); ax.set_ylabel("Predicted pKa")
    ax.set_title("titrate-FiLM — predicted vs experimental (all pH replicates)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # right: per-residue replicate spread = how "noisy" predictions are across pH
    ax = axes[1]
    rep = (df.groupby(["PDB_ID", "Chain_ID", "Residue_Number", "Residue_Name"])
             ["Predicted_pKa"].std().reset_index(name="pred_std"))
    short_col = rep["Residue_Name"].map(RES_SHORT).fillna(rep["Residue_Name"])
    by_res = []
    labs  = []
    for res in pr.index:
        s = short_col == RES_SHORT.get(res, res)
        vals = rep.loc[s, "pred_std"].dropna().values
        if len(vals) == 0:
            continue
        by_res.append(vals)
        labs.append(RES_SHORT.get(res, res))
    parts = ax.boxplot(by_res, tick_labels=labs, patch_artist=True,
                       showfliers=True, widths=0.6)
    for patch, lab in zip(parts["boxes"], labs):
        patch.set_facecolor(RES_COLORS.get(lab, "#888"))
        patch.set_alpha(0.7)
    ax.set_ylabel("σ(predicted pKa) across pH replicates")
    ax.set_title("Per-residue prediction spread across the 4 pH replicates")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "08_titration_scatter.png", dpi=170, bbox_inches="tight")
    plt.close()

print("\nDONE — all figures in:", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name)
