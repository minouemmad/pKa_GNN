"""
09_feature_analysis.py

Feature-importance analysis for the FFX pipeline GNN.

Two complementary analyses:

  1.  PCA on the stacked node-feature matrix
      ──────────────────────────────────────
      Concatenates `data.x` across every graph in the chosen dataset, fits
      sklearn PCA, and reports
          • explained variance ratio per component
          • cumulative variance vs. # components
          • the top-k input features that load on each principal component
            (so "PC1 is mostly Dipole_X + recalculated_x + Perm_Charge", etc.)
      This answers "what are the principal components consisting of?" without
      having to combinatorially try feature subsets.

  2.  Group-wise permutation importance (model-based)
      ────────────────────────────────────────────────
      Loads a trained checkpoint, then for each FEATURE GROUP shuffles the
      values of those columns across the dataset and re-evaluates MAE.  The
      MAE-increase is the importance of that group.  Groups are
      domain-meaningful bundles ("multipoles", "induced dipoles", "neighbour
      counts at radius r", "pH", "protonation state", etc.) so the result is
      directly actionable for ablation experiments.

Usage:
    # PCA only — no model needed
    python 09_feature_analysis.py --mode titrate --dataset 0 \\
        --out Graph_pKa/Results/Feature_Analysis_titrate

    # Both PCA + permutation importance (requires trained checkpoint)
    python 09_feature_analysis.py --mode titrate --dataset 0 \\
        --checkpoint Graph_pKa/Results/Training_titrate_film/models/dataset_0/fold_1.pth \\
        --arch film --hidden 48 --heads 6 --dropout 0.3 \\
        --out Graph_pKa/Results/Feature_Analysis_titrate

Outputs written to <out>/dataset_{idx}/:
    pca_explained_variance.csv      one row per principal component
    pca_top_loadings.csv            top-k features per PC (long format)
    permutation_importance.csv      MAE delta per feature group (if --checkpoint set)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader

# ── Configuration ─────────────────────────────────────────────────────────────
RADII = [7, 8, 9, 10, 11]

# Feature-group definitions (regex applied to feature-name list).  Each group
# bundles columns that share a physical/chemical meaning so permutation
# importance answers "how much does the model rely on multipoles?" rather
# than per-column noise.
FEATURE_GROUPS: list[tuple[str, str]] = [
    ("residue_one_hot",    r"^Residue Name_"),
    ("local_coords",       r"^recalculated_[xyz]$"),
    ("induced_dipoles",    r"^Dipole_[XYZ]$"),
    ("perm_multipoles",    r"^Perm_"),
    ("hbonds",             r"^Number of H-Bonds"),
    ("sasa",               r"^SASA_Value$"),
    ("protonation",        r"^is_protonated$"),
    ("pH",                 r"^pH$"),
    ("neighbour_counts",   r"^Radius_\d+A_"),
    # The 10-class atom-label one-hot is appended at the end of `x`; we
    # detect it via column index in build_feature_names().
    ("atom_label_one_hot", r"^atom_label_oh_\d+$"),
]
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Feature-name reconstruction
# ════════════════════════════════════════════════════════════════════════════

def build_feature_names(feat_dir: Path, radius: int, n_cols: int) -> list[str]:
    """Reconstruct the column ordering used when 06_create_datasets.py built `x`.

    06 reads each per-residue feature CSV, drops 'Expt. pKa' and 'atom_label',
    keeps only numeric columns, fills NaN with 0, then appends a 10-dim
    one-hot encoding of the original `atom_label` integer column.

    We sample one feature CSV from the requested radius to read the column
    order; the names of the appended atom-label OHE entries are synthesised
    as 'atom_label_oh_0' … 'atom_label_oh_9'.
    """
    radius_dir = feat_dir / "Node_Feature_Vectors" / str(radius)
    candidates = sorted(radius_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No feature CSVs in {radius_dir}")

    df       = pd.read_csv(candidates[0])
    drop_set = {"Expt. pKa", "atom_label"}
    numeric  = df.drop(columns=[c for c in drop_set if c in df.columns]).select_dtypes(include=[np.number])
    base_names = list(numeric.columns)
    oh_names   = [f"atom_label_oh_{i}" for i in range(10)]
    names      = base_names + oh_names

    if len(names) != n_cols:
        log.warning(
            f"Feature-name reconstruction got {len(names)} names but x has "
            f"{n_cols} cols — appending generic names for the remainder.  "
            f"Sample CSV: {candidates[0].name}"
        )
        # pad / truncate to match
        if len(names) < n_cols:
            names += [f"feat_{i}" for i in range(len(names), n_cols)]
        else:
            names = names[:n_cols]
    return names


def assign_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """Map group-name → list of column indices in `x`."""
    groups: dict[str, list[int]] = {name: [] for name, _ in FEATURE_GROUPS}
    for idx, fname in enumerate(feature_names):
        for gname, pattern in FEATURE_GROUPS:
            if re.match(pattern, fname):
                groups[gname].append(idx)
                break
    return {g: cols for g, cols in groups.items() if cols}


# ════════════════════════════════════════════════════════════════════════════
# PCA
# ════════════════════════════════════════════════════════════════════════════

def stack_node_features(data_list) -> np.ndarray:
    """Concatenate every graph's `x` along the node axis → (N_total, F)."""
    return np.concatenate([d.x.numpy() for d in data_list], axis=0)


def run_pca(
    data_list,
    feature_names: list[str],
    out_dir: Path,
    n_components: int | None = None,
    top_k_loadings: int = 8,
) -> None:
    X = stack_node_features(data_list)
    log.info(f"  PCA input: {X.shape[0]} atoms × {X.shape[1]} features")

    # Standardise (mean-centre + unit variance) so loadings are comparable
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    n_components = n_components or min(X.shape[1], 50)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(Xs)

    var_df = pd.DataFrame({
        "component":              np.arange(1, n_components + 1),
        "explained_variance":     pca.explained_variance_,
        "explained_variance_pct": 100.0 * pca.explained_variance_ratio_,
        "cumulative_pct":         100.0 * np.cumsum(pca.explained_variance_ratio_),
    })
    var_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)
    log.info(f"  PCA explained variance → {out_dir / 'pca_explained_variance.csv'}")

    # Per-PC loadings: which input features dominate each component
    loadings = pca.components_                 # (k, F)
    rows: list[dict] = []
    for k in range(n_components):
        order = np.argsort(np.abs(loadings[k]))[::-1][:top_k_loadings]
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "component":  k + 1,
                "rank":       rank,
                "feature":    feature_names[idx],
                "loading":    float(loadings[k, idx]),
                "abs_loading": float(abs(loadings[k, idx])),
                "pc_var_pct": 100.0 * pca.explained_variance_ratio_[k],
            })
    load_df = pd.DataFrame(rows)
    load_df.to_csv(out_dir / "pca_top_loadings.csv", index=False)
    log.info(f"  Top loadings        → {out_dir / 'pca_top_loadings.csv'}")

    # Console summary for the first few PCs
    log.info("  --- PC summary ---")
    for k in range(min(5, n_components)):
        top = load_df[load_df["component"] == k + 1].head(5)
        feats = ", ".join(f"{r['feature']}({r['loading']:+.2f})" for _, r in top.iterrows())
        log.info(f"  PC{k+1} ({pca.explained_variance_ratio_[k]*100:5.1f}%) : {feats}")


# ════════════════════════════════════════════════════════════════════════════
# Permutation importance
# ════════════════════════════════════════════════════════════════════════════

def _import_train_module() -> object:
    """Import 07_train.py as a module (filename starts with a digit)."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ffx_train", here / "07_train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 07_train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ffx_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate_mae(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_abse, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch).view(-1)
            total_abse += (out - batch.y.view(-1)).abs().sum().item()
            n += batch.y.size(0)
    return total_abse / max(n, 1)


def permutation_importance(
    data_list,
    feature_names: list[str],
    checkpoint: Path,
    arch: str,
    hidden: int,
    heads: int,
    dropout: float,
    phs: list[float],
    out_dir: Path,
    n_repeats: int = 5,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    train_mod = _import_train_module()

    input_dim = data_list[0].x.shape[1]
    edge_dim  = data_list[0].edge_attr.shape[1] if data_list[0].edge_attr is not None else None

    # Replicate the args namespace expected by build_model
    args = argparse.Namespace(
        hidden=hidden, heads=heads, dropout=dropout, edge_dim=edge_dim, phs=phs,
    )
    model = train_mod.build_model(arch, input_dim, args).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    log.info(f"  Loaded checkpoint: {checkpoint}")

    loader   = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    baseline = evaluate_mae(model, loader, device)
    log.info(f"  Baseline MAE       : {baseline:.4f}")

    groups = assign_groups(feature_names)
    rng    = np.random.default_rng(seed)
    rows: list[dict] = []

    # Pre-stack node features for shuffling (so we keep marginal distribution)
    full_X = np.concatenate([d.x.numpy() for d in data_list], axis=0)
    sizes  = np.array([d.x.shape[0] for d in data_list])
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    for gname, cols in groups.items():
        deltas = []
        for rep in range(n_repeats):
            # Shuffle the chosen columns globally (across atoms)
            shuffled  = full_X.copy()
            perm      = rng.permutation(shuffled.shape[0])
            for c in cols:
                shuffled[:, c] = full_X[perm, c]

            # Apply the shuffle back into a fresh copy of data_list
            shuffled_list = []
            for i, d in enumerate(data_list):
                d2 = d.clone()
                d2.x = torch.tensor(shuffled[offsets[i]:offsets[i+1]], dtype=d.x.dtype)
                shuffled_list.append(d2)
            shuf_loader = DataLoader(shuffled_list, batch_size=batch_size, shuffle=False)
            mae = evaluate_mae(model, shuf_loader, device)
            deltas.append(mae - baseline)

        mean_delta = float(np.mean(deltas))
        std_delta  = float(np.std(deltas))
        rows.append({
            "group":            gname,
            "n_features":       len(cols),
            "baseline_MAE":     baseline,
            "shuffled_MAE":     baseline + mean_delta,
            "delta_MAE_mean":   mean_delta,
            "delta_MAE_std":    std_delta,
        })
        log.info(f"  {gname:22s}  ΔMAE = {mean_delta:+.4f} ± {std_delta:.4f}  "
                 f"({len(cols)} features)")

    pd.DataFrame(rows).sort_values("delta_MAE_mean", ascending=False).to_csv(
        out_dir / "permutation_importance.csv", index=False
    )
    log.info(f"  Permutation importance → {out_dir / 'permutation_importance.csv'}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + permutation importance for FFX-GNN features")
    parser.add_argument("--mode", choices=["rotopt", "titrate"], default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Override dataset directory (default: Graph_pKa/Features_{mode}/Datasets)")
    parser.add_argument("--feat-dir", type=Path, default=None,
                        help="Feature directory used to look up CSV column order "
                             "(default: Graph_pKa/Features_{mode})")
    parser.add_argument("--dataset", type=str, default="0",
                        help="pkl index to analyse: 0-4 or 'all'")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output root (default: Graph_pKa/Results/Feature_Analysis_{mode})")
    parser.add_argument("--n-components", type=int, default=None,
                        help="Number of principal components (default: min(F, 50))")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-k feature loadings to report per PC")
    # Permutation-importance options
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="If set, also run permutation importance with this trained model")
    parser.add_argument("--arch", choices=["naive", "concat", "film", "multi_branch", "gated"],
                        default="naive")
    parser.add_argument("--hidden",  type=int, default=48)
    parser.add_argument("--heads",   type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--phs",     type=float, nargs="+", default=[3.94, 4.4, 6.45, 8.55])
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--no-edge-attr", action="store_true",
                        help="Strip edge_attr before evaluating the checkpoint "
                             "(must match the flag the checkpoint was trained with)")
    args = parser.parse_args()

    # Path resolution
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    elif args.mode is not None:
        dataset_dir = Path(f"Graph_pKa/Features_{args.mode}/Datasets")
    else:
        dataset_dir = Path("Graph_pKa/Features/Datasets")

    if args.feat_dir is not None:
        feat_dir = args.feat_dir
    elif args.mode is not None:
        feat_dir = Path(f"Graph_pKa/Features_{args.mode}")
    else:
        feat_dir = Path("Graph_pKa/Features")

    out_root = args.out
    if out_root is None:
        tag = f"_{args.mode}" if args.mode else ""
        out_root = Path(f"Graph_pKa/Results/Feature_Analysis{tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(RADII))) if args.dataset == "all" else [int(args.dataset)]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    for idx in indices:
        pkl = dataset_dir / f"data_list_{idx}.pkl"
        if not pkl.exists():
            log.warning(f"Missing {pkl} — skipping")
            continue

        with open(pkl, "rb") as fh:
            data_list = pickle.load(fh)
        if not data_list:
            log.warning(f"{pkl} is empty — skipping")
            continue

        radius = RADII[idx]
        out_dir = out_root / f"dataset_{idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_cols = data_list[0].x.shape[1]
        feature_names = build_feature_names(feat_dir, radius, n_cols)
        log.info(f"\nDataset {idx} (radius {radius} Å, {len(data_list)} graphs, "
                 f"{n_cols} features)")

        run_pca(data_list, feature_names, out_dir,
                n_components=args.n_components, top_k_loadings=args.top_k)

        if args.checkpoint is not None:
            if args.no_edge_attr:
                for d in data_list:
                    d.edge_attr = None
            permutation_importance(
                data_list, feature_names, args.checkpoint,
                arch=args.arch, hidden=args.hidden, heads=args.heads,
                dropout=args.dropout, phs=args.phs, out_dir=out_dir,
                n_repeats=args.n_repeats, batch_size=args.batch,
                seed=args.seed, device=device,
            )

    log.info(f"\nDone.  Results → {out_root}")


if __name__ == "__main__":
    main()
