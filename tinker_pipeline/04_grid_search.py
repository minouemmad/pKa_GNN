#!/usr/bin/env python3
"""
09_grid_search_paper_ffx.py

Full hyperparameter grid search for GAT on the FFX + PKAD-R dataset.

This is a direct analogue of Graph_pKa/Net/GNN_Grid_Search/GAT.py but
reads from the FFX-extracted, paper-exact feature datasets produced by:

    08_prepare_features_paper.py  ->  Graph_pKa/Features_Paper/
    08_create_datasets_paper.py   ->  Graph_pKa/Features_Paper/Datasets/

Hyperparameter grid (identical to the original paper grid search):
    heads          : 4, 6, 8
    hidden_channels: 16, 32, 48, 64
    batch_size     : 16, 24, 32, 40
    k_folds        : 10
    patience       : 20  (applied after epoch 60)
    learning_rate  : 0.001, 0.006, 0.01, 0.06, 0.1
    dropout        : 0.2, 0.3, 0.4, 0.5
    loss           : SmoothL1Loss(β=0.5), L1Loss, MSELoss

Model architecture (identical to paper):
    GATv2Conv(input_dim -> hidden*heads, concat=True, no self-loops)
    ReLU -> Dropout -> global_mean_pool -> Linear(hidden*heads -> 1)

Outputs to:
    Graph_pKa/Results/Grid_Search_Paper_FFX/
        grid_search_results.csv      ← all (dataset, combo) rows
        best_result.csv              ← single best row by MAE
        all_best_predictions/        ← per-fold held-out predictions

Run (from pKa_GNN/ as CWD):
    python tinker_pipeline/04_grid_search.py
    python tinker_pipeline/04_grid_search.py --dataset 0           # only radius 7 Å
    python tinker_pipeline/04_grid_search.py --single-core         # disable parallelism
    python tinker_pipeline/04_grid_search.py \\
        --dataset-dir Graph_pKa/Features_Paper/Datasets \\
        --results-dir Graph_pKa/Results/Grid_Search_Paper_FFX
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ── Default paths (relative to CWD = pKa_GNN root) ───────────────────────────
DATASET_DIR = Path("Graph_pKa/Features_Paper/Datasets")
RESULTS_DIR = Path("Graph_pKa/Results/Grid_Search_Paper_FFX")
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


set_seed(42)


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_training_data(dataset_dir: Path, max_index: int = 5):
    """Load graph datasets from pickled files in *dataset_dir*."""
    data_sets: list[list] = []
    for i in range(max_index):
        pkl_path = dataset_dir / f"data_list_{i}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as fh:
                dl = pickle.load(fh)
            if dl:
                data_sets.append(dl)
                print(f"Loaded dataset {i}: {pkl_path} ({len(dl)} graphs)")
        elif i > 0:
            break

    assert len(data_sets) > 0, f"No datasets found under {dataset_dir}"
    input_dim = data_sets[0][0].x.shape[1]
    return data_sets, input_dim


# ════════════════════════════════════════════════════════════════════════════
# Model -- identical to Graph_pKa/Net/GNN_Grid_Search/GAT.py  ::  class GATConv
# ════════════════════════════════════════════════════════════════════════════

class GATConv(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int,
                 dropout: float, heads: int):
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


# ════════════════════════════════════════════════════════════════════════════
# Prediction CSV helper
# ════════════════════════════════════════════════════════════════════════════

def save_predictions_to_csv(
    all_best_predictions: list[dict],
    dataset_idx: int,
    loss_function,
    hidden_channels: int,
    batch_size: int,
    patience: int,
    k_folds: int,
    lr: float,
    dropout: float,
    heads: int,
    results_dir: Path,
) -> None:
    tag = (
        f"{loss_function.__class__.__name__}"
        f"_h{hidden_channels}_b{batch_size}_lr{lr}_d{dropout}_hd{heads}"
    )
    out_dir = (
        results_dir
        / "all_best_predictions"
        / f"dataset_{dataset_idx}"
        / tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"predictions_dataset_{dataset_idx}_{tag}.csv"
    pd.DataFrame(all_best_predictions).to_csv(out_dir / fname, index=False)


# ════════════════════════════════════════════════════════════════════════════
# Core training worker  (one hyperparameter combination, called in parallel)
# ════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    loss_function,
    hidden_channels: int,
    batch_size: int,
    patience: int,
    k_folds: int,
    lr: float,
    dropout: float,
    heads: int,
    dataset_idx: int,
    data_list: list,
    input_dim: int,
    results_dir: Path,
) -> tuple:
    """
    10-fold CV for one hyperparameter combination, on one dataset.

    Returns tuple matching columns:
      (dataset_idx, Loss_fn, Hidden, Batch, Patience, K-Folds,
       LR, Dropout, Heads, MAE, RMSE)
    """
    device = torch.device("cpu")   # joblib workers share CPU

    save_dir = (
        results_dir
        / "saved_models"
        / f"dataset_{dataset_idx}"
        / (
            f"loss_{loss_function.__class__.__name__}"
            f"_h{hidden_channels}_b{batch_size}"
            f"_lr{lr}_d{dropout}_hd{heads}"
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    mae_total = 0.0
    mse_total = 0.0
    n_total   = 0
    all_best_predictions: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        model     = GATConv(input_dim, hidden_channels, dropout, heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_path = save_dir / f"best_model_fold_{fold + 1}.pth"

        train_data   = [data_list[i] for i in train_idx]
        val_data     = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

        best_mae   = float("inf")
        best_abse  = 0.0
        best_sque  = 0.0
        best_preds: list[dict] = []
        no_improve = 0

        for epoch in range(500):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            total_loss, n_train = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                n_b   = batch.y.size(0)
                optimizer.zero_grad()
                out  = model(batch).squeeze()
                loss = loss_function(out, batch.y.to(device).squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * n_b
                n_train    += n_b

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            total_abse, total_sque, n_val = 0.0, 0.0, 0
            epoch_preds: list[dict] = []

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    out   = model(batch).view(-1)
                    y     = batch.y.to(device).view(-1)

                    total_abse += F.l1_loss(out, y, reduction="sum").item()
                    total_sque += F.mse_loss(out, y, reduction="sum").item()
                    n_val      += y.size(0)

                    for j in range(y.size(0)):
                        graph_pos = i * batch_size + j
                        if graph_pos < len(val_idx):
                            g = data_list[val_idx[graph_pos]]
                            epoch_preds.append({
                                "graph_id":      int(val_idx[graph_pos]),
                                "PDB_ID":        getattr(g, "PDB_ID",        ""),
                                "Chain_ID":      getattr(g, "Chain_ID",       ""),
                                "Residue_Number": getattr(g, "Residue_Number", ""),
                                "Residue":       getattr(g, "Residue_Name",   ""),
                                "true_pKa":      y[j].item(),
                                "prediction":    out[j].item(),
                            })

            val_mae = total_abse / n_val

            if val_mae < best_mae:
                best_mae   = val_mae
                best_abse  = total_abse
                best_sque  = total_sque
                best_preds = epoch_preds.copy()
                torch.save(model.state_dict(), model_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and epoch > 60:
                    print(
                        f"  [{dataset_idx}] Early stop epoch {epoch}  "
                        f"(best val_MAE={best_mae:.4f})"
                    )
                    break

            if epoch % 10 == 0:
                avg_train = total_loss / n_train
                print(
                    f"  [{dataset_idx}] epoch {epoch:4d}  "
                    f"train_loss={avg_train:.4f}  val_MAE={val_mae:.4f}"
                )

        mae_total += best_abse
        mse_total += best_sque
        n_total   += n_val
        all_best_predictions.extend(best_preds)

    avg_mae  = mae_total / n_total
    avg_rmse = float(np.sqrt(mse_total / n_total))

    save_predictions_to_csv(
        all_best_predictions, dataset_idx, loss_function,
        hidden_channels, batch_size, patience, k_folds,
        lr, dropout, heads, results_dir,
    )

    return (
        dataset_idx,
        loss_function.__class__.__name__,
        hidden_channels, batch_size, patience, k_folds,
        lr, dropout, heads,
        avg_mae, avg_rmse,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search GAT hyperparameters for FFX + PKAD-R pipeline"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help="Directory with data_list_*.pkl  (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Output directory               (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        help="PKL index to search on: 0–4 or 'all'  (default: all)",
    )
    parser.add_argument(
        "--single-core", action="store_true",
        help="Disable joblib parallelism (useful for debugging)",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    available_cores = multiprocessing.cpu_count()
    num_cores = 1 if args.single_core else min(60, available_cores - 1)
    print(f"Using {num_cores} CPU core(s) for parallel training.")

    data_sets, input_dim = load_training_data(args.dataset_dir)
    print(f"input_dim = {input_dim}")

    # ── Hyperparameter grid -- identical to Graph_pKa/Net/GNN_Grid_Search/GAT.py
    heads_range    = [4, 6, 8]
    hidden_range   = list(range(16, 65, 16))            # 16, 32, 48, 64
    batch_range    = list(range(16, 41, 8))             # 16, 24, 32, 40
    k_folds_range  = [10]
    patience_range = [20]
    lr_range       = [0.001, 0.006, 0.01, 0.06, 0.1]
    dropout_range  = np.round(np.arange(0.2, 0.51, 0.1), 2).tolist()  # 0.2 … 0.5
    loss_fns       = [
        torch.nn.SmoothL1Loss(beta=0.5),
        torch.nn.L1Loss(),
        torch.nn.MSELoss(),
    ]
    # ─────────────────────────────────────────────────────────────────────────

    if args.dataset == "all":
        indices = range(len(data_sets))
    else:
        indices = [int(args.dataset)]

    all_results: list[tuple] = []

    for dataset_idx in indices:
        data_list = data_sets[dataset_idx]
        print(f"\n{'='*70}")
        print(
            f"Grid Search  Dataset {dataset_idx}"
            f"  ({len(data_list)} graphs, input_dim={input_dim})"
        )
        print(f"{'='*70}")

        param_combinations = list(product(
            loss_fns, hidden_range, batch_range,
            patience_range, k_folds_range,
            lr_range, dropout_range, heads_range,
        ))
        print(f"Total combinations: {len(param_combinations)}")

        worker_kwargs = dict(
            dataset_idx=dataset_idx,
            data_list=data_list,
            input_dim=input_dim,
            results_dir=args.results_dir,
        )

        if args.single_core:
            dataset_results = [
                train_and_evaluate(*params, **worker_kwargs)
                for params in param_combinations
            ]
        else:
            dataset_results = Parallel(n_jobs=num_cores)(
                delayed(train_and_evaluate)(*params, **worker_kwargs)
                for params in param_combinations
            )

        all_results.extend(dataset_results)

    # ── Summarise results ─────────────────────────────────────────────────────
    cols = [
        "Dataset", "Loss_fn", "Hidden", "Batch", "Patience",
        "K-Folds", "LR", "Dropout", "Heads", "MAE", "RMSE",
    ]
    df_all = pd.DataFrame(all_results, columns=cols)
    df_all.to_csv(args.results_dir / "grid_search_results.csv", index=False)

    best_idx = df_all["MAE"].idxmin()
    best = df_all.loc[best_idx]
    pd.DataFrame([best]).to_csv(args.results_dir / "best_result.csv", index=False)

    print("\n\nBest combination:")
    print(best.to_string())
    print(f"\nDataset={int(best['Dataset'])}, Loss={best['Loss_fn']}, "
          f"Hidden={int(best['Hidden'])}, Batch={int(best['Batch'])}, "
          f"LR={best['LR']}, Dropout={best['Dropout']}, Heads={int(best['Heads'])}")
    print(f"Best MAE: {best['MAE']:.4f}  RMSE: {best['RMSE']:.4f}")
    print(f"\nResults saved to {args.results_dir}")

    print(
        "\nTo train with the best hyperparameters, run:\n"
        f"  python 08_train_paper.py "
        f"--hidden {int(best['Hidden'])} "
        f"--heads {int(best['Heads'])} "
        f"--lr {best['LR']} "
        f"--dropout {best['Dropout']} "
        f"--batch {int(best['Batch'])}"
    )


if __name__ == "__main__":
    main()
