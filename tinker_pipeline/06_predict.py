#!/usr/bin/env python3
"""
12_predict_paper_ffx.py

Prediction and evaluation script for the paper-exact FFX + PKAD-R pipeline.

This is a direct analogue of Graph_pKa/Predict.py adapted for:
  * Chain-aware filenames  ({PDB}_{chain}_{resseq}.{ResName})
  * Paper-exact feature layout  (4-class OHE, 9-class atom labels, 26 features)
  * Models trained by 11_train_paper.py  (or 10_grid_search_paper_ffx.py)

Two operating modes:

  eval mode  (default)
      Loads the pre-built pickled datasets from Features_Paper/Datasets/,
      runs every saved fold model, averages predictions across all folds,
      and reports MAE / RMSE overall and per residue type.
      Use this mode to evaluate models trained by 11_train_paper.py.

  predict mode  (--predict-mode)
      Builds fresh PyG Data objects directly from the Features_Paper CSV
      files.  Useful for proteins not in the training set (new predictions)
      or to cross-check the pickled datasets.

Outputs to:
    Graph_pKa/Results/Predictions_Paper_FFX/
        predictions_dataset_{idx}_per_fold.csv    <- raw fold-by-fold points
        predictions_dataset_{idx}_averaged.csv    <- fold-averaged predictions
        summary_metrics.csv                       <- per-dataset MAE / RMSE

Usage (from pKa_GNN/ as CWD):
    python tinker_pipeline/06_predict.py
    python tinker_pipeline/06_predict.py --dataset 0       # only radius 7 A
    python tinker_pipeline/06_predict.py --model-dir Graph_pKa/Results/Training_Paper/models
    python tinker_pipeline/06_predict.py --predict-mode \\
        --adj-dir  Graph_pKa/Features_Paper/Adjacency_Matrices/With_Self_Loop \\
        --node-dir Graph_pKa/Features_Paper/Node_Feature_Vectors
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ── Default paths (relative to CWD = pKa_GNN root) ───────────────────────────
DATASET_DIR = Path("Graph_pKa/Features_Paper/Datasets")
MODEL_DIR   = Path("Graph_pKa/Results/Training_Paper/models")
RESULTS_DIR = Path("Graph_pKa/Results/Predictions_Paper_FFX")
ADJ_DIR     = Path("Graph_pKa/Features_Paper/Adjacency_Matrices/With_Self_Loop")
NODE_DIR    = Path("Graph_pKa/Features_Paper/Node_Feature_Vectors")

RADII                  = [7, 8, 9, 10, 11]
NUM_ATOM_LABEL_CLASSES = 9   # paper uses 9 classes (no sidechain-S)
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# Model -- must match GATModelPaper in 08_train_paper.py exactly
# ════════════════════════════════════════════════════════════════════════════

class GATModelPaper(torch.nn.Module):
    """Single GATv2Conv layer -> mean pool -> Linear (paper architecture)."""

    def __init__(self, input_dim: int, hidden_channels: int,
                 heads: int, dropout: float):
        super().__init__()
        self.conv1   = GATv2Conv(input_dim, hidden_channels, heads=heads,
                                 concat=True, add_self_loops=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.pool    = global_mean_pool
        self.out     = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x, data.batch)
        return self.out(x)


def _load_state_dict(path: Path, map_location: str = "cpu") -> dict:
    """Load model state dict with compatibility for older PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch < 2.0 does not have the weights_only kwarg
        return torch.load(path, map_location=map_location)


def _infer_arch(state_dict: dict) -> tuple[int, int]:
    """Infer (hidden_channels, heads) from a saved state dict.

    GATv2Conv with concat=True stores:
      conv1.att  : shape [1, heads, hidden]
      out.weight : shape [1, hidden * heads]
    """
    att_shape = state_dict["conv1.att"].shape   # (1, heads, hidden)
    heads  = int(att_shape[1])
    hidden = int(att_shape[2])
    return hidden, heads


# ════════════════════════════════════════════════════════════════════════════
# Build datasets from CSV (predict mode)
# ════════════════════════════════════════════════════════════════════════════

def _parse_stem(stem: str):
    """Parse {PDB}_{chain}_{resseq}.{ResName} -> (pdb_id, chain_id, resseq, residue_name)."""
    parts        = stem.rsplit(".", 1)         # ["1ABC_A_42", "Aspartate"]
    residue_name = parts[1]
    left_parts   = parts[0].rsplit("_", 1)    # ["1ABC_A", "42"]
    resseq       = int(left_parts[1])
    id_parts     = left_parts[0].split("_")   # ["1ABC", "A"]
    pdb_id       = id_parts[0]
    chain_id     = id_parts[1] if len(id_parts) > 1 else ""
    return pdb_id, chain_id, resseq, residue_name


def build_data_list_from_csv(adj_dir: Path, node_dir: Path, radius: int) -> list:
    """Build a list of PyG Data objects from Features_Paper CSVs.

    Mirrors the dataset-building logic in Graph_pKa/Predict.py but uses
    the paper-exact feature layout (4-class residue OHE, 9-class atom labels).
    """
    node_subdir = node_dir / str(radius)
    if not node_subdir.exists():
        raise FileNotFoundError(f"Node feature directory not found: {node_subdir}")

    data_list: list[Data] = []

    for adj_file in sorted(adj_dir.glob("*_adjacency.csv")):
        stem    = adj_file.name.replace("_adjacency.csv", "")
        nf_file = node_subdir / f"{stem}.csv"
        if not nf_file.exists():
            print(f"  WARNING: feature CSV not found for {stem} – skipping.")
            continue

        # Adjacency -> edge_index
        adj_matrix = pd.read_csv(adj_file, header=0, index_col=0).values
        edge_index  = torch.tensor(adj_matrix, dtype=torch.int).nonzero(
            as_tuple=False
        ).t().contiguous()

        # Node features
        nf = pd.read_csv(nf_file)
        if "atom_label" not in nf.columns:
            print(f"  WARNING: 'atom_label' missing in {nf_file.name} – skipping.")
            continue

        # pKa label (optional -- new proteins may not have one)
        pka_col = next(
            (c for c in ["Expt.pKa", "Expt. pKa", "Expt_pKa"] if c in nf.columns),
            None,
        )
        y = (
            torch.tensor([nf[pka_col].values[0]], dtype=torch.float)
            if pka_col
            else None
        )

        # Atom-label one-hot (9 classes -- paper-exact)
        atom_labels = (
            torch.tensor(nf["atom_label"].values, dtype=torch.long)
            .clamp(0, NUM_ATOM_LABEL_CLASSES - 1)
        )
        atom_ohe = F.one_hot(atom_labels, num_classes=NUM_ATOM_LABEL_CLASSES).float()

        # Residue type label (argmax of Residue Name_* columns)
        ohe_cols      = nf.filter(like="Residue Name_").values
        residue_label = int(np.argmax(ohe_cols, axis=1)[0])

        # Numeric feature tensor (drop pKa + atom_label raw column, then concat OHE)
        drop_cols = [c for c in [pka_col, "atom_label"] if c and c in nf.columns]
        x_numeric = torch.tensor(
            nf.drop(columns=drop_cols).fillna(0.0).values, dtype=torch.float
        )
        x = torch.cat([x_numeric, atom_ohe], dim=1)

        pdb_id, chain_id, resseq, residue_name = _parse_stem(stem)

        data              = Data(x=x, edge_index=edge_index, y=y)
        data.residue_label  = residue_label
        data.PDB_ID         = pdb_id
        data.Chain_ID       = chain_id
        data.Residue_Number = resseq
        data.Residue_Name   = residue_name

        data_list.append(data)

    return data_list


# ════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ════════════════════════════════════════════════════════════════════════════

def run_inference(
    model: GATModelPaper,
    data_list: list,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run model over the full data_list in eval mode.

    Returns a DataFrame with one row per graph containing metadata and
    Predicted_pKa (plus True_pKa when available).
    """
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.eval()
    rows: list[dict] = []
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            out = model(batch).view(-1)
            y   = batch.y.view(-1) if batch.y is not None else None
            n_b = out.size(0)

            for j in range(n_b):
                idx = n_done + j
                if idx < len(data_list):
                    g   = data_list[idx]
                    row = {
                        "PDB_ID":         getattr(g, "PDB_ID",          ""),
                        "Chain_ID":       getattr(g, "Chain_ID",         ""),
                        "Residue_Number": getattr(g, "Residue_Number",   ""),
                        "Residue":        getattr(g, "Residue_Name",     ""),
                        "Predicted_pKa":  out[j].item(),
                    }
                    if y is not None:
                        row["True_pKa"] = y[j].item()
                    rows.append(row)

            n_done += n_b

    return pd.DataFrame(rows)


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if "True_pKa" not in df.columns or df.empty:
        return {}
    true = df["True_pKa"].values
    pred = df["Predicted_pKa"].values
    return {
        "MAE":  float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict / evaluate using paper-exact FFX pipeline models. "
            "Analogous to Graph_pKa/Predict.py for the FFX + PKAD-R data."
        )
    )
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset index 0–4 or 'all'  (default: all)")
    parser.add_argument("--model-dir",   type=Path, default=MODEL_DIR,
                        help="Root of saved fold models (dataset_*/fold_*.pth)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size",  type=int,  default=32)

    # eval mode (default) -- load pre-built pickled datasets
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR,
                        help="(eval mode) Pickled dataset directory")

    # predict mode -- build datasets from raw CSVs
    parser.add_argument("--predict-mode", action="store_true",
                        help="Build datasets from CSVs instead of pickled files")
    parser.add_argument("--adj-dir", type=Path, default=ADJ_DIR,
                        help="(predict mode) Adjacency CSV directory")
    parser.add_argument("--node-dir", type=Path, default=NODE_DIR,
                        help="(predict mode) Node-feature vectors root (contains 7/ 8/ …)")

    # Model architecture -- must match the training run
    parser.add_argument("--hidden",  type=int,   default=48,
                        help="Hidden channels per head (paper default: 48)")
    parser.add_argument("--heads",   type=int,   default=4,
                        help="Attention heads (paper default: 4)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate (paper default: 0.5; used only for model init)")

    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    indices = (
        list(range(len(RADII)))
        if args.dataset == "all"
        else [int(args.dataset)]
    )

    summary_rows: list[dict] = []

    for dataset_idx in indices:
        radius = RADII[dataset_idx]
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_idx}  (neighbourhood radius {radius} Å)")
        print(f"{'='*60}")

        # ── Load or build data_list ────────────────────────────────────────
        if args.predict_mode:
            print("  Building dataset from CSV files (predict mode)…")
            data_list = build_data_list_from_csv(
                args.adj_dir, args.node_dir, radius
            )
        else:
            pkl_path = args.dataset_dir / f"data_list_{dataset_idx}.pkl"
            if not pkl_path.exists():
                print(f"  WARNING: {pkl_path} not found – skipping.")
                continue
            with open(pkl_path, "rb") as fh:
                data_list = pickle.load(fh)

        if not data_list:
            print("  Empty dataset – skipping.")
            continue

        input_dim = data_list[0].x.shape[1]
        print(f"  {len(data_list)} graphs, input_dim={input_dim}")

        # ── Discover trained fold models ───────────────────────────────────
        ds_model_dir = args.model_dir / f"dataset_{dataset_idx}"
        if not ds_model_dir.exists():
            print(f"  WARNING: model dir not found: {ds_model_dir}")
            print("  Run 08_train_paper.py first to generate trained models.")
            continue

        model_paths = sorted(ds_model_dir.glob("fold_*.pth"))
        if not model_paths:
            print(f"  WARNING: no fold_*.pth files in {ds_model_dir}")
            print("  Run 08_train_paper.py first to generate trained models.")
            continue

        print(f"  Found {len(model_paths)} fold model(s).")

        # ── Collect per-fold predictions ───────────────────────────────────
        fold_rows: list[dict] = []
        # Map graph index -> list of Predicted_pKa values across folds
        preds_by_graph: defaultdict[int, list[float]] = defaultdict(list)

        for k, model_path in enumerate(model_paths, start=1):
            print(f"  Fold {k}  ({model_path.name})")
            state_dict = _load_state_dict(model_path)
            hidden, heads = _infer_arch(state_dict)
            model = GATModelPaper(input_dim, hidden, heads, args.dropout)
            model.load_state_dict(state_dict)

            fold_df = run_inference(model, data_list, batch_size=args.batch_size)

            for i, row in fold_df.iterrows():
                fold_row = dict(row)
                fold_row["fold"] = k
                fold_row["graph_idx"] = i
                fold_rows.append(fold_row)
                preds_by_graph[int(i)].append(row["Predicted_pKa"])

        # Save per-fold predictions
        per_fold_csv = args.results_dir / f"predictions_dataset_{dataset_idx}_per_fold.csv"
        pd.DataFrame(fold_rows).to_csv(per_fold_csv, index=False)
        print(f"  Per-fold predictions -> {per_fold_csv}")

        # ── Average predictions across folds ──────────────────────────────
        avg_rows: list[dict] = []
        for i, preds in sorted(preds_by_graph.items()):
            g = data_list[i]
            row = {
                "PDB_ID":         getattr(g, "PDB_ID",          ""),
                "Chain_ID":       getattr(g, "Chain_ID",         ""),
                "Residue_Number": getattr(g, "Residue_Number",   ""),
                "Residue":        getattr(g, "Residue_Name",     ""),
                "Predicted_pKa":  float(np.mean(preds)),
            }
            if g.y is not None:
                row["True_pKa"] = g.y.item()
            avg_rows.append(row)

        avg_df = pd.DataFrame(avg_rows)
        avg_csv = args.results_dir / f"predictions_dataset_{dataset_idx}_averaged.csv"
        avg_df.to_csv(avg_csv, index=False)
        print(f"  Averaged predictions  -> {avg_csv}")

        # ── Metrics ────────────────────────────────────────────────────────
        if "True_pKa" in avg_df.columns:
            overall = _compute_metrics(avg_df)
            print(
                f"\n  Overall (avg over {len(model_paths)} folds)  "
                f"MAE={overall['MAE']:.4f}  RMSE={overall['RMSE']:.4f}"
            )

            print("\n  Per-residue type:")
            for res, sub in avg_df.groupby("Residue"):
                m = _compute_metrics(sub)
                print(
                    f"    {res:12s}  n={len(sub):4d}  "
                    f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}"
                )

            summary_rows.append({
                "dataset_idx": dataset_idx,
                "radius_A":    radius,
                "n_graphs":    len(data_list),
                "input_dim":   input_dim,
                "n_folds":     len(model_paths),
                "MAE":         overall["MAE"],
                "RMSE":        overall["RMSE"],
            })
        else:
            print("  No True_pKa available -- skipping metric computation.")

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary_rows:
        summary_csv = args.results_dir / "summary_metrics.csv"
        summary_df  = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary -> {summary_csv}")
        print(
            summary_df[["dataset_idx", "radius_A", "n_graphs", "n_folds", "MAE", "RMSE"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
