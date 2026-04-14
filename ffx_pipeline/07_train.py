"""
07_train.py

Single training run of the GATv2 pKa model on the datasets built by
06_create_datasets.py.  Uses 10-fold cross-validation, saves the best
checkpoint per fold, and writes per-residue predictions + summary metrics.

Outputs (inside Graph_pKa/Results/Training/):
    models/
        dataset_{r}/
            fold_{k}.pth           ← best checkpoint for each fold
    predictions/
        dataset_{r}_all_folds.csv  ← per-residue predictions (all folds)
        summary_metrics.csv        ← per-dataset MAE / RMSE

Run:
    python 07_train.py
    python 07_train.py --epochs 300 --hidden 64 --heads 8 --lr 0.001

Hyperparameters (defaults tuned for ~255-graph datasets):
    --hidden     48      hidden channels per head
    --heads       6      attention heads
    --lr        0.005    learning rate
    --dropout   0.3      dropout rate
    --batch      16      batch size
    --patience   30      early-stopping patience (min epoch 60)
    --epochs    500      maximum epochs per fold
    --folds      10      k-fold CV splits
    --dataset    all     which pkl index to train on (0-4 or 'all')
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
DATASET_DIR = Path("Graph_pKa/Features/Datasets")
RESULTS_DIR = Path("Graph_pKa/Results/Training")
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
# Model
# ════════════════════════════════════════════════════════════════════════════

class GATModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int, heads: int,
                 dropout: float, edge_dim: int | None = None):
        super().__init__()
        self.conv1    = GATv2Conv(input_dim, hidden_channels, heads=heads,
                                  concat=True, add_self_loops=False,
                                  edge_dim=edge_dim)
        self.dropout  = torch.nn.Dropout(dropout)
        self.pool     = global_mean_pool
        self.out      = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x, data.batch)
        return self.out(x)


# ════════════════════════════════════════════════════════════════════════════
# Training helpers
# ════════════════════════════════════════════════════════════════════════════

def train_one_fold(
    model: GATModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    model_path: Path,
    epochs: int,
    patience: int,
    min_epoch: int = 60,
) -> list[dict]:
    """Train with early stopping; return best-epoch val predictions."""
    best_mae      = float("inf")
    best_preds: list[dict] = []
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss, n_train = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(batch).squeeze()
            loss = loss_fn(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.y.size(0)
            n_train    += batch.y.size(0)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        total_abse, n_val = 0.0, 0
        epoch_preds: list[dict] = []

        with torch.no_grad():
            for batch in val_loader:
                out = model(batch).view(-1)
                total_abse += F.l1_loss(out, batch.y.view(-1), reduction="sum").item()
                n_val      += batch.y.size(0)

                for i in range(batch.y.size(0)):
                    epoch_preds.append({
                        "PDB_ID":          batch.PDB_ID[i],
                        "Chain_ID":        batch.Chain_ID[i],
                        "Residue_Number":  int(batch.Residue_Number[i]),
                        "Residue_Name":    batch.Residue_Name[i],
                        "True_pKa":        batch.y[i].item(),
                        "Predicted_pKa":   out[i].item(),
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
            if no_improve >= patience and epoch >= min_epoch:
                print(f"    early stop at epoch {epoch}  (best val_MAE={best_mae:.4f})")
                break

    return best_preds


# ════════════════════════════════════════════════════════════════════════════
# Per-dataset training
# ════════════════════════════════════════════════════════════════════════════

def train_dataset(
    dataset_idx: int,
    data_list: list,
    args: argparse.Namespace,
    results_dir: Path,
) -> dict:
    """10-fold CV on one radius dataset. Returns summary metrics dict."""
    radius = RADII[dataset_idx]
    print(f"\n{'='*60}")
    print(f"Dataset {dataset_idx} — radius {radius} Å  ({len(data_list)} graphs)")
    print(f"{'='*60}")

    input_dim   = data_list[0].x.shape[1]
    edge_dim    = data_list[0].edge_attr.shape[1] if data_list[0].edge_attr is not None else None
    print(f"  input_dim={input_dim}, edge_dim={edge_dim}, hidden={args.hidden}, heads={args.heads}, "
          f"lr={args.lr}, dropout={args.dropout}, batch={args.batch}")

    model_dir = results_dir / "models" / f"dataset_{dataset_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir  = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    kf             = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds:  list[dict] = []
    fold_maes:  list[float] = []
    fold_rmses: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list), start=1):
        print(f"\n  Fold {fold}/{args.folds}  (train={len(train_idx)}, val={len(val_idx)})")
        train_data  = [data_list[i] for i in train_idx]
        val_data    = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=args.batch, shuffle=False)

        model     = GATModel(input_dim, args.hidden, args.heads, args.dropout,
                              edge_dim=edge_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn   = torch.nn.SmoothL1Loss(beta=0.5)
        model_path = model_dir / f"fold_{fold}.pth"

        fold_preds = train_one_fold(
            model, train_loader, val_loader,
            optimizer, loss_fn, model_path,
            epochs=args.epochs, patience=args.patience,
        )

        for p in fold_preds:
            p["fold"] = fold
        all_preds.extend(fold_preds)

        true_vals  = [p["True_pKa"]      for p in fold_preds]
        pred_vals  = [p["Predicted_pKa"] for p in fold_preds]
        if true_vals:
            mae  = mean_absolute_error(true_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
            fold_maes.append(mae)
            fold_rmses.append(rmse)
            print(f"  Fold {fold} best  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # Save all-fold predictions
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
    parser = argparse.ArgumentParser(description="Train GATv2 pKa model (07_train.py)")
    parser.add_argument("--hidden",   type=int,   default=48)
    parser.add_argument("--heads",    type=int,   default=6)
    parser.add_argument("--lr",       type=float, default=0.005)
    parser.add_argument("--dropout",  type=float, default=0.3)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--patience", type=int,   default=30)
    parser.add_argument("--epochs",   type=int,   default=500)
    parser.add_argument("--folds",    type=int,   default=10)
    parser.add_argument("--dataset",  type=str,   default="all",
                        help="pkl index to train on: 0-4 or 'all'")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--no-edge-attr", action="store_true",
                        help="Ignore edge_attr (dx/dy/dz/distance) even if present in pkl")
    parser.add_argument("--exclude-cys-tyr", action="store_true",
                        help="Exclude CYS and TYR residues from training (match paper's 4-residue scope)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Override output directory (default: auto-derived from flags)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Determine output directory — avoid overwriting previous runs
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        suffix = ""
        if args.exclude_cys_tyr:
            suffix += "_no_cys_tyr"
        if args.no_edge_attr:
            suffix += "_no_edge"
        results_dir = RESULTS_DIR.parent / (RESULTS_DIR.name + suffix) if suffix else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {results_dir}")

    # Decide which datasets to run
    if args.dataset == "all":
        indices = list(range(len(RADII)))
    else:
        indices = [int(args.dataset)]

    summary_rows: list[dict] = []

    for idx in indices:
        pkl_path = DATASET_DIR / f"data_list_{idx}.pkl"
        if not pkl_path.exists():
            print(f"WARNING: {pkl_path} not found – skipping.")
            continue
        with open(pkl_path, "rb") as fh:
            data_list = pickle.load(fh)
        if not data_list:
            print(f"WARNING: data_list_{idx}.pkl is empty – skipping.")
            continue

        if args.no_edge_attr:
            for d in data_list:
                d.edge_attr = None
            print(f"  [--no-edge-attr] edge features stripped from dataset {idx}")

        if args.exclude_cys_tyr:
            before = len(data_list)
            data_list = [d for d in data_list
                         if d.Residue_Name not in ("Cysteine", "Tyrosine")]
            print(f"  [--exclude-cys-tyr] {before - len(data_list)} graphs removed "
                  f"(CYS/TYR); {len(data_list)} remaining")

        row = train_dataset(idx, data_list, args, results_dir)
        summary_rows.append(row)

    # Summary CSV
    if summary_rows:
        summary_path = results_dir / "predictions" / "summary_metrics.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary metrics → {summary_path}")
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
