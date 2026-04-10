"""
08_train_paper.py

Paper-exact training script — replicates the GAT_1 model and training
procedure reported in Song et al., J. Chem. Inf. Model. 2026
(doi:10.1021/acs.jcim.5c01681) as closely as possible.

Architecture (GAT_1 — best model in paper):
    GATv2Conv(input_dim → hidden*heads, heads=4, concat=True, no self-loops)
    ReLU → Dropout → global_mean_pool → Linear(hidden*heads → 1)

Training (paper's reported settings):
    loss      : MSELoss      (paper grid-searched L1 / Huber / MSE)
    optimizer : Adam, lr=0.01
    batch     : 32
    dropout   : 0.5
    heads     : 4
    hidden    : 48
    patience  : 20   (early stop after >60 epochs with no val MAE improvement)
    max epochs: 500
    k-folds   : 10
    seed      : 42

Key differences vs 07_train.py:
  • MSELoss instead of SmoothL1Loss
  • lr=0.01, heads=4, dropout=0.5, batch=32, patience=20 (paper optimum)
  • No edge_dim (paper does not use edge features)
  • Reads from Graph_pKa/Features_Paper/Datasets/
  • Results written to Graph_pKa/Results/Training_Paper/

Run:
    python 08_train_paper.py
    python 08_train_paper.py --dataset 0          # only radius 7
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data  # noqa: F401
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("Graph_pKa/Features_Paper/Datasets")
RESULTS_DIR = Path("Graph_pKa/Results/Training_Paper")
RADII       = [7, 8, 9, 10, 11]
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(1)


# ════════════════════════════════════════════════════════════════════════════
# Model — mirrors GATConv class in Graph_pKa/Net/GNN_Grid_Search/GAT.py
# ════════════════════════════════════════════════════════════════════════════

class GATModelPaper(torch.nn.Module):
    """Single GATv2Conv layer → mean pool → Linear.

    Matches the architecture reported as GAT_1 (best performing) in the paper.
    No edge features, no multiple GNN layers, no batch normalisation.
    """
    def __init__(self, input_dim: int, hidden_channels: int, heads: int, dropout: float):
        super().__init__()
        print(f"GATModelPaper: input_dim={input_dim}, hidden={hidden_channels}, heads={heads}")
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


# ════════════════════════════════════════════════════════════════════════════
# Training helpers
# ════════════════════════════════════════════════════════════════════════════

def train_one_fold(
    model:      GATModelPaper,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    loss_fn,
    model_path:   Path,
    epochs:       int,
    patience:     int,
    batch_size:   int,
    min_epoch:    int = 60,
) -> list[dict]:
    """Train with early stopping; returns best-epoch validation predictions."""
    best_mae   = float("inf")
    best_preds: list[dict] = []
    no_improve = 0

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, n_train = 0.0, 0
        for batch in train_loader:
            n_batch = batch.y.size(0)
            optimizer.zero_grad()
            out  = model(batch).squeeze()
            loss = loss_fn(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * n_batch
            n_train    += n_batch

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        total_abse = 0.0
        total_sque = 0.0
        n_val      = 0
        epoch_preds: list[dict] = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                out  = model(batch).view(-1)
                y    = batch.y.view(-1)

                total_abse += F.l1_loss(out, y, reduction="sum").item()
                total_sque += F.mse_loss(out, y, reduction="sum").item()
                n_val      += y.size(0)

                for j in range(y.size(0)):
                    # Map back to original graph index for metadata
                    graph_pos = i * batch_size + j
                    epoch_preds.append({
                        "True_pKa":      y[j].item(),
                        "Predicted_pKa": out[j].item(),
                    })

        val_mae = total_abse / n_val

        if epoch % 10 == 0:
            print(f"    epoch {epoch:4d}  train_loss={total_loss/n_train:.4f}  val_MAE={val_mae:.4f}")

        if val_mae < best_mae:
            best_mae   = val_mae
            best_preds = epoch_preds.copy()
            torch.save(model.state_dict(), model_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch > min_epoch:
                print(f"    Early stop at epoch {epoch}  (best val_MAE={best_mae:.4f})")
                break

    return best_preds


# ════════════════════════════════════════════════════════════════════════════
# Per-dataset training
# ════════════════════════════════════════════════════════════════════════════

def train_dataset(
    dataset_idx: int,
    data_list:   list,
    args:        argparse.Namespace,
    results_dir: Path,
) -> dict:
    """10-fold CV on one radius dataset. Returns summary dict."""
    radius = RADII[dataset_idx]
    print(f"\n{'='*60}")
    print(f"Dataset {dataset_idx} — radius {radius} Å  ({len(data_list)} graphs)")
    print(f"  hidden={args.hidden}, heads={args.heads}, lr={args.lr}, "
          f"dropout={args.dropout}, batch={args.batch}, patience={args.patience}")
    print(f"{'='*60}")

    input_dim  = data_list[0].x.shape[1]
    model_dir  = results_dir / "models" / f"dataset_{dataset_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir   = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    kf             = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds:  list[dict] = []
    fold_maes:  list[float] = []
    fold_rmses: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list), start=1):
        print(f"\n  Fold {fold}/{args.folds}  (train={len(train_idx)}, val={len(val_idx)})")
        train_data   = [data_list[i] for i in train_idx]
        val_data     = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=args.batch, shuffle=False)

        model     = GATModelPaper(input_dim, args.hidden, args.heads, args.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn   = torch.nn.MSELoss()  # paper-exact loss function
        model_path = model_dir / f"fold_{fold}.pth"

        fold_preds = train_one_fold(
            model, train_loader, val_loader,
            optimizer, loss_fn, model_path,
            epochs=args.epochs, patience=args.patience,
            batch_size=args.batch,
        )

        for p in fold_preds:
            p["fold"] = fold
        all_preds.extend(fold_preds)

        true_vals = [p["True_pKa"]      for p in fold_preds]
        pred_vals = [p["Predicted_pKa"] for p in fold_preds]
        if true_vals:
            mae  = mean_absolute_error(true_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
            fold_maes.append(mae)
            fold_rmses.append(rmse)
            print(f"  Fold {fold} best  MAE={mae:.4f}  RMSE={rmse:.4f}")

    pred_csv = pred_dir / f"dataset_{dataset_idx}_all_folds.csv"
    pd.DataFrame(all_preds).to_csv(pred_csv, index=False)
    print(f"\n  Predictions → {pred_csv}")

    mean_mae  = float(np.mean(fold_maes))  if fold_maes  else float("nan")
    mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else float("nan")
    print(f"  CV mean  MAE={mean_mae:.4f}  RMSE={mean_rmse:.4f}")

    return {
        "dataset_idx": dataset_idx,
        "radius_A":    radius,
        "n_graphs":    len(data_list),
        "input_dim":   input_dim,
        "mean_MAE":    mean_mae,
        "mean_RMSE":   mean_rmse,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-exact GAT pKa training (08_train_paper.py)")

    # Paper-exact defaults (GAT_1 best model from grid search)
    parser.add_argument("--hidden",   type=int,   default=48,
                        help="Hidden channels per attention head (paper: 48)")
    parser.add_argument("--heads",    type=int,   default=4,
                        help="Number of attention heads (paper: 4)")
    parser.add_argument("--lr",       type=float, default=0.01,
                        help="Adam learning rate (paper: 0.01)")
    parser.add_argument("--dropout",  type=float, default=0.5,
                        help="Dropout rate (paper: 0.5)")
    parser.add_argument("--batch",    type=int,   default=32,
                        help="Batch size (paper: 32)")
    parser.add_argument("--patience", type=int,   default=20,
                        help="Early-stopping patience, applied after epoch 60 (paper: 20)")
    parser.add_argument("--epochs",   type=int,   default=500,
                        help="Maximum epochs per fold (paper: 500)")
    parser.add_argument("--folds",    type=int,   default=10,
                        help="K-fold CV splits (paper: 10)")
    parser.add_argument("--dataset",  type=str,   default="all",
                        help="PKL index to train on: 0-4 or 'all'")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR,
                        help="Directory containing data_list_*.pkl")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Where to write models + predictions")
    args = parser.parse_args()

    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(RADII))) if args.dataset == "all" else [int(args.dataset)]
    summary_rows: list[dict] = []

    for idx in indices:
        pkl_path = args.dataset_dir / f"data_list_{idx}.pkl"
        if not pkl_path.exists():
            print(f"WARNING: {pkl_path} not found – skipping.")
            continue
        with open(pkl_path, "rb") as fh:
            data_list = pickle.load(fh)
        if not data_list:
            print(f"WARNING: data_list_{idx}.pkl is empty – skipping.")
            continue

        row = train_dataset(idx, data_list, args, args.results_dir)
        summary_rows.append(row)

    if summary_rows:
        summary_path = args.results_dir / "predictions" / "summary_metrics.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary metrics → {summary_path}")
        print(pd.DataFrame(summary_rows)[["dataset_idx","radius_A","n_graphs","input_dim","mean_MAE","mean_RMSE"]].to_string(index=False))


if __name__ == "__main__":
    main()
