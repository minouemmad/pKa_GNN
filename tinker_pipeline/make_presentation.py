from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT      = Path(__file__).resolve().parents[1]
TINKER    = ROOT / "tinker_pipeline" / "Graph_pKa"
FFX       = ROOT / "ffx_pipeline"    / "Graph_pKa"
OUT       = TINKER / "Presentation"
OUT.mkdir(parents=True, exist_ok=True)

RADII = [7, 8, 9, 10, 11]

# ────────────────────────────── helpers ──────────────────────────────────────
def metrics(df: pd.DataFrame) -> dict:
    """Compute MAE / RMSE averaged across folds."""
    g = df.groupby("fold").apply(
        lambda x: pd.Series({
            "MAE":  (x["Predicted_pKa"] - x["True_pKa"]).abs().mean(),
            "RMSE": np.sqrt(((x["Predicted_pKa"] - x["True_pKa"])**2).mean()),
        })
    )
    return {
        "MAE":  g["MAE"].mean(),
        "MAE_std":  g["MAE"].std(ddof=1),
        "RMSE": g["RMSE"].mean(),
        "RMSE_std": g["RMSE"].std(ddof=1),
        "n_folds": int(g.shape[0]),
        "n_rows":  int(df.shape[0]),
    }

def load_run_per_radius(run_dir: Path) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(RADII):
        f = run_dir / "predictions" / f"dataset_{i}_all_folds.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        m  = metrics(df); m["radius"] = r; m["dataset"] = i
        rows.append(m)
    return pd.DataFrame(rows)

def load_run_dataset0(run_dir: Path) -> dict | None:
    f = run_dir / "predictions" / "dataset_0_all_folds.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return {**metrics(df), "df": df}

# ────────────────────────────── 1. ROTOPT vs TINKER ──────────────────────────
print("[1/6] rotopt vs tinker per-radius …", flush=True)
rotopt_df = load_run_per_radius(FFX    / "Results" / "Training_rotopt_naive_full138_allR")
tinker_df = load_run_per_radius(TINKER / "Results" / "Training_Tinker_Paper")
ffxmin_df = load_run_per_radius(TINKER / "Results" / "Training_FFX_Paper")

cmp_rows = []
for label, df in [("Tinker minimize (paper feats, n=290)",        tinker_df),
                  ("FFX rotopt   (paper feats, n=292)",          ffxmin_df),
                  ("FFX rotopt   (titrate-aware feats, n=343)",  rotopt_df)]:
    for _, r in df.iterrows():
        cmp_rows.append({"method": label,
                         "radius": int(r["radius"]),
                         "MAE":   r["MAE"],
                         "RMSE":  r["RMSE"],
                         "n_rows": int(r["n_rows"])})
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(OUT / "rotopt_vs_tinker_metrics.csv", index=False)
print(cmp.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
colors = {"Tinker minimize (paper feats, n=290)":       "#888888",
          "FFX rotopt   (paper feats, n=292)":         "#4c72b0",
          "FFX rotopt   (titrate-aware feats, n=343)": "#dd8452"}
markers = {"Tinker minimize (paper feats, n=290)":       "o",
           "FFX rotopt   (paper feats, n=292)":         "s",
           "FFX rotopt   (titrate-aware feats, n=343)": "D"}
for ax, metric in zip(axes, ["MAE", "RMSE"]):
    for label in ["Tinker minimize (paper feats, n=290)",
                  "FFX rotopt   (paper feats, n=292)",
                  "FFX rotopt   (titrate-aware feats, n=343)"]:
        sub = cmp[cmp["method"] == label].sort_values("radius")
        if sub.empty:
            continue
        ax.plot(sub["radius"], sub[metric],
                marker=markers[label], color=colors[label],
                linewidth=2, markersize=8, label=label)
    ax.set_xlabel("Graph cutoff radius (Å)")
    ax.set_ylabel(f"{metric} (pKa units)")
    ax.set_title(f"{metric} vs cutoff radius")
    ax.grid(alpha=0.3)
    ax.set_xticks(RADII)
axes[0].legend(frameon=True, fontsize=8, loc="best")
plt.suptitle("Static-structure prep: FFX rotopt vs. Tinker minimize\n(rows 1–2 share the paper feature builder — the clean prep comparison)",
             y=1.04, fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig_rotopt_vs_tinker.png", dpi=160, bbox_inches="tight")
plt.close()

# ────────────────────────────── 2. PER-RESIDUE MAE (rotopt) ──────────────────
print("\n[2/6] per-residue breakdown (rotopt full138 @ r=9) …", flush=True)
rdir = FFX / "Results" / "Training_rotopt_naive_full138_allR" / "predictions" / "dataset_2_all_folds.csv"
rotopt_r9 = pd.read_csv(rdir)
# per-residue MAE
per_res = (rotopt_r9
           .assign(abs_err=(rotopt_r9["Predicted_pKa"] - rotopt_r9["True_pKa"]).abs())
           .groupby("Residue_Name")
           .agg(MAE=("abs_err", "mean"),
                RMSE=("abs_err", lambda x: np.sqrt((x**2).mean())),
                n=("abs_err", "size"))
           .sort_values("MAE"))
per_res.to_csv(OUT / "rotopt_per_residue.csv")
print(per_res)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

# Bar
order = per_res.index.tolist()
ax = axes[0]
bars = ax.bar(order, per_res["MAE"], color="#4c72b0", edgecolor="black")
for bar, n in zip(bars, per_res["n"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"n={n}", ha="center", fontsize=8)
ax.axhline(per_res["MAE"].mean(), color="red", linestyle="--",
           label=f"avg MAE = {per_res['MAE'].mean():.3f}")
ax.set_ylabel("MAE (pKa units)")
ax.set_title("Per-residue MAE — FFX rotopt @ r=9 Å")
ax.tick_params(axis="x", rotation=35)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Scatter coloured by residue
ax = axes[1]
focus = ["Aspartate", "Glutamate", "Histidine", "Lysine"]
palette = {"Aspartate": "#d62728", "Glutamate": "#ff7f0e",
           "Histidine": "#2ca02c", "Lysine": "#9467bd"}
for res in focus:
    sub = rotopt_r9[rotopt_r9["Residue_Name"] == res]
    if sub.empty:
        continue
    ax.scatter(sub["True_pKa"], sub["Predicted_pKa"],
               s=22, alpha=0.7,
               color=palette[res], label=f"{res} (n={len(sub)})")
other = rotopt_r9[~rotopt_r9["Residue_Name"].isin(focus)]
if not other.empty:
    ax.scatter(other["True_pKa"], other["Predicted_pKa"],
               s=14, alpha=0.35, color="grey", label=f"other (n={len(other)})")
lims = [rotopt_r9["True_pKa"].min()-0.5, rotopt_r9["True_pKa"].max()+0.5]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Experimental pKa")
ax.set_ylabel("Predicted pKa")
ax.set_title("Predicted vs experimental — FFX rotopt @ r=9 Å")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig_per_residue.png", dpi=160, bbox_inches="tight")
plt.close()

# ────────────────────────────── 3. FEATURE ENGINEERING SUMMARY ───────────────
print("\n[3/6] feature-engineering combined sweep …", flush=True)
electro_pf = pd.read_csv(TINKER / "Net_FFX138_Electro" / "sweep_electrostatics_138_perfold.csv")
phys_pf    = pd.read_csv(TINKER / "Net_FFX138_PhysEdge" / "sweep_physedge_138_perfold.csv")
electro_sm = pd.read_csv(TINKER / "Net_FFX138_Electro" / "sweep_electrostatics_138_summary.csv")
phys_sm    = pd.read_csv(TINKER / "Net_FFX138_PhysEdge" / "sweep_physedge_138_summary.csv")

# Combine — drop duplicate Charge baseline from one of them
phys_pf_no_charge = phys_pf[phys_pf["variant"] != "Charge"]
phys_sm_no_charge = phys_sm[phys_sm["variant"] != "Charge"]
combo_pf = pd.concat([electro_pf, phys_pf_no_charge], ignore_index=True)
combo_sm = pd.concat([electro_sm, phys_sm_no_charge], ignore_index=True)
combo_sm = combo_sm.sort_values("mean_MAE")
combo_sm.to_csv(OUT / "feature_engineering_summary.csv", index=False)
print(combo_sm[["variant","mean_MAE","std_MAE","mean_delta_MAE","wilcoxon_p"]].to_string(index=False))

variant_order = combo_sm["variant"].tolist()
charge_baseline = float(combo_sm[combo_sm["variant"] == "Charge"]["mean_MAE"].iloc[0])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# Box plot of per-fold MAEs by variant
ax = axes[0]
data = [combo_pf[combo_pf["variant"] == v]["MAE"].values for v in variant_order]
bp = ax.boxplot(data, labels=variant_order, patch_artist=True, widths=0.6)
palette10 = plt.cm.tab10(np.linspace(0, 1, len(variant_order)))
for patch, c in zip(bp["boxes"], palette10):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(charge_baseline, color="red", linestyle="--",
           label=f"Charge baseline = {charge_baseline:.3f}")
ax.set_ylabel("MAE (pKa units), per fold")
ax.set_title("Feature-engineering sweep — 10 variants × 8 seeds × 10 folds (138 PDBs, r=9 Å)")
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

# ΔMAE bar with Wilcoxon p annotation
ax = axes[1]
delta = combo_sm.set_index("variant").loc[variant_order]
delta_no_charge = delta.drop(index="Charge", errors="ignore")
bars = ax.barh(delta_no_charge.index, delta_no_charge["mean_delta_MAE"],
               color=["#2ca02c" if v < 0 else "#d62728"
                      for v in delta_no_charge["mean_delta_MAE"]])
for bar, p in zip(bars, delta_no_charge["wilcoxon_p"]):
    if pd.isna(p):
        s = ""
    elif p < 0.05:
        s = f"p={p:.3f}*"
    else:
        s = f"p={p:.2f}"
    x = bar.get_width()
    ax.text(x + (0.0003 if x >= 0 else -0.0003),
            bar.get_y() + bar.get_height()/2,
            s, va="center",
            ha="left" if x >= 0 else "right", fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("ΔMAE vs Charge baseline (negative = better)")
ax.set_title("Paired Wilcoxon vs baseline (320 fold-pairs each)")
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUT / "fig_feature_engineering.png", dpi=160, bbox_inches="tight")
plt.close()

# Permutation importance combined plot — pick Invariant + Charge from PhysEdge dir
imp_paths = [
    ("Charge baseline", TINKER / "Net_FFX138_PhysEdge" / "perm_importance_Charge.csv"),
    ("Invariant",       TINKER / "Net_FFX138_PhysEdge" / "perm_importance_Invariant.csv"),
    ("InducedDip",      TINKER / "Net_FFX138_Electro"  / "perm_importance_InducedDip.csv"),
    ("BothDip",         TINKER / "Net_FFX138_Electro"  / "perm_importance_BothDip.csv"),
]
imp_data = {}
for name, p in imp_paths:
    if p.exists():
        d = pd.read_csv(p)
        col = "delta_MAE" if "delta_MAE" in d.columns else d.columns[1]
        feat_col = "feature" if "feature" in d.columns else d.columns[0]
        imp_data[name] = d.set_index(feat_col)[col]

if imp_data:
    imp_df = pd.DataFrame(imp_data).fillna(0)
    # Order rows by max importance across runs
    imp_df = imp_df.loc[imp_df.max(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35*len(imp_df))))
    y = np.arange(len(imp_df))
    width = 0.8 / len(imp_df.columns)
    for i, col in enumerate(imp_df.columns):
        ax.barh(y + i*width - 0.4 + width/2, imp_df[col], height=width, label=col)
    ax.set_yticks(y); ax.set_yticklabels(imp_df.index)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔMAE on shuffle (larger = feature more used)")
    ax.set_title("Permutation importance — shuffle a feature group, measure MAE rise")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_perm_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

# ────────────────────────────── 4. TITRATION ROTOP SNEAK PEEK ────────────────
print("\n[4/6] titration rotop sneak peek (subset49) …", flush=True)
seeds = [42, 7, 123]
arches = ["naive", "film", "gated"]
rows = []
for seed in seeds:
    for tag in [("rotopt", "naive")] + [("titrate", a) for a in arches]:
        mode, arch = tag
        d = FFX / "Results" / f"Training_subset49_{mode}_{arch}_seed{seed}"
        f = d / "predictions" / "dataset_0_all_folds.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        m = metrics(df)
        rows.append({"mode": mode, "arch": arch, "seed": seed,
                     "MAE": m["MAE"], "RMSE": m["RMSE"], "n_rows": m["n_rows"]})
tit_df = pd.DataFrame(rows)
tit_df.to_csv(OUT / "titration_subset49.csv", index=False)
print(tit_df.to_string(index=False))

if not tit_df.empty:
    # Aggregate across seeds
    tit_agg = tit_df.groupby(["mode", "arch"]).agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        n_seeds=("seed", "nunique")).reset_index()
    tit_agg["label"] = tit_agg["mode"] + " / " + tit_agg["arch"]
    tit_agg = tit_agg.sort_values("MAE_mean")
    tit_agg.to_csv(OUT / "titration_subset49_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        x = np.arange(len(tit_agg))
        ax.bar(x, tit_agg[f"{metric}_mean"], yerr=tit_agg[f"{metric}_std"],
               color=["#888"] + ["#4c72b0", "#dd8452", "#55a868"][:len(tit_agg)-1],
               capsize=4, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(tit_agg["label"], rotation=15)
        ax.set_ylabel(f"{metric}")
        ax.set_title(f"Subset49: rotopt vs titration variants — {metric}")
        ax.grid(axis="y", alpha=0.3)
        for i, (m_val, s_val) in enumerate(zip(tit_agg[f"{metric}_mean"],
                                                tit_agg[f"{metric}_std"])):
            ax.text(i, m_val + 0.02, f"{m_val:.2f}\n±{s_val:.2f}",
                    ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_titration_rotop.png", dpi=160, bbox_inches="tight")
    plt.close()

# ────────────────────────────── 5. GAT ARCHITECTURE DIAGRAM ──────────────────
print("\n[5/6] GAT architecture diagram …", flush=True)
fig, ax = plt.subplots(figsize=(13, 5.0))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, w, h, label, color, fontsize=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                       linewidth=1.4, facecolor=color, edgecolor="black")
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.2, color="black")
    ax.add_patch(a)

# Stage 1: input features
box(0.1, 1.5, 2.4, 2.0,
    "Input features\n(per atom)\n\n• 9-dim atom-label OHE\n• 4-dim residue OHE\n"
    "• atomic charge\n• SASA\n• H-bond donor/acceptor\n• radius counts\n• "
    "local-frame coords (x,y,z)\n• AMOEBA dipole invariants*\n\n→ ≈ 24–30 dims",
    "#dbe9f4", fontsize=8)
arrow(2.6, 2.5, 3.4, 2.5)

# Stage 2: GATv2Conv
box(3.5, 1.7, 3.0, 1.6,
    "GATv2Conv\nhidden = 48\nheads = 4 (concat → 192)\nedge_dim = optional\nadd_self_loops = False\n→ ReLU → Dropout(0.5)",
    "#fce8b2", fontsize=9)
arrow(6.6, 2.5, 7.4, 2.5)

# Stage 3: pooling
box(7.5, 2.0, 2.0, 1.0,
    "global_mean_pool\n(per graph)",
    "#f9d7c1", fontsize=9)
arrow(9.6, 2.5, 10.4, 2.5)

# Stage 4: head
box(10.5, 2.0, 2.2, 1.0,
    "Linear(192 → 1)\npredicted pKa",
    "#cde6c4", fontsize=10)

# Edge attribute box (optional)
box(3.5, 0.1, 3.0, 1.1,
    "edge_attr (optional)\nCoulomb 1/r²\nμ·E (q-dipole)\ndipole-dipole 1/r³",
    "#e0e0e0", fontsize=8)
arrow(5.0, 1.2, 5.0, 1.7)

# Title
ax.text(6.5, 4.6, "GATv2 model for residue pKa on FFX-rotopt static graphs",
        ha="center", fontsize=14, weight="bold")
ax.text(6.5, 4.15,
        "1 message-passing layer • mean-pool • linear head • MSE loss • Adam(lr=1e-2) • 10-fold CV",
        ha="center", fontsize=10, color="#444")
ax.text(0.1, 0.5, "* invariant scalars: ‖μ‖, μ·z̑, μ·E_neigh — for both induced and permanent dipoles",
        fontsize=8, color="#444")

plt.savefig(OUT / "fig_gat_architecture.png", dpi=170, bbox_inches="tight")
plt.close()

print("\n[6/6] All figures written to:", OUT)
print("Files:")
for p in sorted(OUT.glob("*")):
    print(" ", p.name)
