"""Standalone Wilcoxon explainer figure for the FFX138 feature-engineering sweep.

Two panels (no overlap, generous margins):
  (a) Forest plot — mean ΔMAE per variant vs Charge baseline, with 95 % CI
      from the per-fold paired deltas, p-value annotated on the right side.
  (b) Paired-difference distribution for the best variant (InducedDip):
      MAE_variant − MAE_charge for each of the 80 (seed, fold) pairs, with
      median, mean, zero-line, and a count of "wins" (Δ < 0).

Inputs (already on disk):
  tinker_pipeline/Graph_pKa/Net_FFX138_Electro/sweep_electrostatics_138_summary.csv
  tinker_pipeline/Graph_pKa/Net_FFX138_Electro/sweep_electrostatics_138_perfold.csv
  tinker_pipeline/Graph_pKa/Net_FFX138_PhysEdge/sweep_physedge_138_summary.csv
  tinker_pipeline/Graph_pKa/Net_FFX138_PhysEdge/sweep_physedge_138_perfold.csv

Output:
  Graph_pKa/Presentation_FFX/04_wilcoxon_variants.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ELECTRO  = ROOT / "tinker_pipeline" / "Graph_pKa" / "Net_FFX138_Electro"
PHYSEDGE = ROOT / "tinker_pipeline" / "Graph_pKa" / "Net_FFX138_PhysEdge"
OUT      = ROOT / "Graph_pKa" / "Presentation_FFX"
OUT.mkdir(parents=True, exist_ok=True)


# ─── load and merge both sweep summaries / per-fold tables ──────────────────
sum_e = pd.read_csv(ELECTRO  / "sweep_electrostatics_138_summary.csv")
sum_p = pd.read_csv(PHYSEDGE / "sweep_physedge_138_summary.csv")
pf_e  = pd.read_csv(ELECTRO  / "sweep_electrostatics_138_perfold.csv")
pf_p  = pd.read_csv(PHYSEDGE / "sweep_physedge_138_perfold.csv")

summary = pd.concat([sum_e, sum_p[~sum_p["variant"].isin(sum_e["variant"])]],
                    ignore_index=True)
perfold = pd.concat([pf_e, pf_p[~pf_p["variant"].isin(pf_e["variant"])]],
                    ignore_index=True)

baseline = "Charge"
variants = [v for v in summary["variant"] if v != baseline]

# Per-fold paired deltas: ΔMAE = MAE_variant − MAE_baseline, on (seed, fold) keys
base_pf = perfold[perfold["variant"] == baseline][["seed", "fold", "MAE"]] \
                 .rename(columns={"MAE": "MAE_base"})

rows = []
delta_dict: dict[str, np.ndarray] = {}
for v in variants:
    var_pf = perfold[perfold["variant"] == v][["seed", "fold", "MAE"]] \
                    .rename(columns={"MAE": "MAE_var"})
    pair = var_pf.merge(base_pf, on=["seed", "fold"], how="inner")
    delta = (pair["MAE_var"] - pair["MAE_base"]).values
    delta_dict[v] = delta
    n = len(delta)
    mean_d = float(np.mean(delta))
    se     = float(np.std(delta, ddof=1) / np.sqrt(n))
    ci95   = 1.96 * se
    _, w_p = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
    rows.append({
        "variant":   v,
        "n_pairs":   n,
        "mean_d":    mean_d,
        "ci95":      ci95,
        "wilcoxon_p": float(w_p),
        "frac_better": float(np.mean(delta < 0)),
    })

forest = pd.DataFrame(rows).sort_values("mean_d").reset_index(drop=True)
print(forest.to_string(index=False))


# ─── figure ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(15.5, 6.8))
# Wide left panel for forest, narrower right panel for distribution.
# Big right margin keeps p-value text from running into the axes border.
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0],
                      left=0.09, right=0.97, top=0.86, bottom=0.22, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# (a) forest plot — ΔMAE ± 95 % CI, p-value to the right of the axes
ys = np.arange(len(forest))[::-1]    # best (most negative) at top
colors = ["#2ca02c" if (d < 0 and p < 0.05) else
          "#1f77b4" if d < 0 else
          "#d62728" if (d > 0 and p < 0.05) else
          "#888888"
          for d, p in zip(forest["mean_d"], forest["wilcoxon_p"])]

ax1.errorbar(forest["mean_d"], ys,
             xerr=forest["ci95"],
             fmt="o", color="black", ecolor="black",
             elinewidth=1.4, capsize=4, markersize=7,
             markerfacecolor="white", markeredgewidth=1.6, zorder=3)
# overlay coloured marker face
for x, y, c in zip(forest["mean_d"], ys, colors):
    ax1.plot(x, y, "o", markersize=7, color=c, zorder=4,
             markeredgecolor="black", markeredgewidth=1.0)

ax1.axvline(0, color="black", linewidth=1.0, linestyle="-")
ax1.axvspan(-0.001, 0.001, color="gray", alpha=0.06, zorder=1)  # "noise band"
ax1.set_yticks(ys)
ax1.set_yticklabels(forest["variant"])
ax1.set_xlabel("Mean ΔMAE   ( variant − Charge baseline )   [pKa units]")
ax1.set_title("Per-fold paired Wilcoxon vs Charge baseline\n"
              "(n = 80 pairs = 8 seeds × 10 folds, two-sided)",
              fontsize=11)
ax1.grid(axis="x", alpha=0.3)

# annotate p-value + frac_better on the right edge of the data area
xmin = (forest["mean_d"] - forest["ci95"]).min()
xmax = (forest["mean_d"] + forest["ci95"]).max()
pad  = 0.18 * (xmax - xmin)
ax1.set_xlim(xmin - 0.10 * (xmax - xmin), xmax + pad)
xtext = xmax + 0.02 * (xmax - xmin)
for y, row in zip(ys, forest.itertuples()):
    sig = "***" if row.wilcoxon_p < 0.001 else \
          "**"  if row.wilcoxon_p < 0.01  else \
          "*"   if row.wilcoxon_p < 0.05  else "ns"
    txt = f"p = {row.wilcoxon_p:.3f}  {sig}   wins {row.frac_better*100:.0f}%"
    ax1.text(xtext, y, txt, va="center", ha="left",
             fontsize=9, family="monospace",
             color="#2ca02c" if row.wilcoxon_p < 0.05 and row.mean_d < 0 else
                    "#d62728" if row.wilcoxon_p < 0.05 and row.mean_d > 0 else
                    "#444")

# legend explaining colors — placed BELOW the panel so it never overlaps rows
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(facecolor="#2ca02c", edgecolor="black",
          label="significantly better (p<0.05)"),
    Patch(facecolor="#1f77b4", edgecolor="black",
          label="numerically better, ns"),
    Patch(facecolor="#888888", edgecolor="black",
          label="indistinguishable"),
    Patch(facecolor="#d62728", edgecolor="black",
          label="significantly worse (p<0.05)"),
], fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
   ncol=4, framealpha=0.95, frameon=True)


# (b) paired-difference distribution for InducedDip (best variant)
best = "InducedDip"
delta = delta_dict[best]
mean_d = float(np.mean(delta))
med_d  = float(np.median(delta))
n_better = int(np.sum(delta < 0))
_, p_best = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")

# histogram + kernel-style density via numpy
ax2.hist(delta, bins=18, color="#1f77b4", alpha=0.65, edgecolor="black")
ax2.axvline(0, color="black", linewidth=1.0, label="no change")
ax2.axvline(mean_d, color="#d62728", linewidth=1.8,
            linestyle="--", label=f"mean = {mean_d:+.4f}")
ax2.axvline(med_d,  color="#2ca02c", linewidth=1.8,
            linestyle=":",  label=f"median = {med_d:+.4f}")

ax2.set_xlabel("ΔMAE per fold   ( InducedDip − Charge )   [pKa units]")
ax2.set_ylabel("Number of (seed, fold) pairs")
ax2.set_title(f"Paired-difference distribution — best variant: {best}\n"
              f"n = {len(delta)} pairs · {n_better}/{len(delta)} better "
              f"({n_better/len(delta)*100:.0f}%) · Wilcoxon p = {p_best:.3f}",
              fontsize=10.5)
ax2.legend(fontsize=9, loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.3)

# margin so legend / title never touch axes
ax2.margins(x=0.05)

fig.suptitle(
    "Did any engineered electrostatic feature significantly beat the Charge baseline? — No.",
    fontsize=12.5, weight="bold", y=0.965, color="#222"
)

out_path = OUT / "04_wilcoxon_variants.png"
fig.savefig(out_path, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"\n→ {out_path}")
forest.to_csv(OUT / "wilcoxon_variants_table.csv", index=False)
