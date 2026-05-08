#!/usr/bin/env python3
"""
08b_grid_search_film.py  (FFX pipeline)

Hyperparameter grid search for the FiLM-conditioned GAT on the FFX titration
dataset.  Mirrors `08_grid_search.py` but:

  • Uses the FiLM architecture from `07_train.py::GATModel(arch='film')`:
        GATv2Conv → ReLU → (γ⊙h + β  with γ,β = MLP(pH))
                  → Dropout → global_mean_pool → Linear → 1
  • Uses GroupKFold by residue so that all per-pH replicates of the same
    (PDB, chain, residue) stay together in train/val splits — matching the
    "titrate (FiLM, group-residue CV)" setup used in
    `Graph_pKa/Presentation_FFX/titration_summary.csv`.
  • Defaults to `Graph_pKa/Features_titrate/Datasets/` →
    `Graph_pKa/Results/Grid_Search_FFX_titrate_film/`.

Hyperparameter grid (identical to 08_grid_search.py):
    heads          : 4, 6, 8
    hidden_channels: 16, 32, 48, 64
    batch_size     : 16, 24, 32, 40
    learning_rate  : 0.001, 0.006, 0.01, 0.06, 0.1
    dropout        : 0.2, 0.3, 0.4, 0.5
    loss           : SmoothL1Loss(β=0.5), L1Loss, MSELoss
    k_folds        : 10
    patience       : 20  (after epoch 60)

Run:
    python ffx_pipeline/08b_grid_search_film.py
    python ffx_pipeline/08b_grid_search_film.py --dataset 0
    python ffx_pipeline/08b_grid_search_film.py --single-core
    python ffx_pipeline/08b_grid_search_film.py \\
        --dataset-dir Graph_pKa/Features_titrate/Datasets \\
        --results-dir Graph_pKa/Results/Grid_Search_FFX_titrate_film
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ── Default paths (relative to CWD = pKa_GNN root) ───────────────────────────
DATASET_DIR = Path("Graph_pKa/Features_titrate/Datasets")
RESULTS_DIR = Path("Graph_pKa/Results/Grid_Search_FFX_titrate_film")
# ─────────────────────────────────────────────────────────────────────────────

# pH conditioning normalisation — identical to 07_train.py
_PH_MEAN = 6.0
_PH_STD  = 3.0


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
    # Sanity check: titration datasets must carry per-graph pH
    sample = data_sets[0][0]
    if not hasattr(sample, "pH"):
        raise RuntimeError(
            f"Dataset at {dataset_dir} has no .pH attribute — expected titration "
            "datasets built with `06_create_datasets.py --mode titrate`."
        )
    return data_sets, input_dim


# ════════════════════════════════════════════════════════════════════════════
# pH normalisation
# ════════════════════════════════════════════════════════════════════════════

def _normalize_ph(ph: torch.Tensor) -> torch.Tensor:
    """Normalise pH ∈ [3, 9] to roughly [-1, 1].  NaN → 0."""
    out = (ph - _PH_MEAN) / _PH_STD
    return torch.nan_to_num(out, nan=0.0)


# ════════════════════════════════════════════════════════════════════════════
# Model — FiLM-conditioned GAT (mirrors GATModel(arch='film') in 07_train.py)
# ════════════════════════════════════════════════════════════════════════════

class FiLMGATConv(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int,
                 dropout: float, heads: int):
        super().__init__()
        emb_dim = hidden_channels * heads
        self.conv1   = GATv2Conv(input_dim, hidden_channels, heads=heads,
                                 concat=True, add_self_loops=False)
        self.film    = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2 * emb_dim),
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.pool    = global_mean_pool
        self.head    = torch.nn.Linear(emb_dim, 1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)

        # Per-graph γ, β from pH; broadcast to nodes via batch index
        ph = _normalize_ph(data.pH.view(-1, 1)).to(x.device)
        gb = self.film(ph)                                  # (B, 2*emb)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + gamma                                 # init ≈ identity
        x = gamma[data.batch] * x + beta[data.batch]

        x = self.dropout(x)
        x = self.pool(x, data.batch)
        return self.head(x)


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
    out_dir = results_dir / "all_best_predictions" / f"dataset_{dataset_idx}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred in all_best_predictions:
        fold   = pred["fold"]
        preds  = pred["predictions"]
        labels = pred["labels"]
        ids    = pred.get("ids", [None] * len(preds))
        phs    = pred.get("pHs", [None] * len(preds))
        for pid, p, l, ph in zip(ids, preds, labels, phs):
            rows.append({"fold": fold, "id": pid, "pH": ph,
                         "prediction": p, "label": l})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# Training loop (one fold)
# ════════════════════════════════════════════════════════════════════════════

def train_one_fold(
    train_data, val_data,
    input_dim: int,
    hidden_channels: int,
    heads: int,
    dropout: float,
    lr: float,
    batch_size: int,
    patience: int,
    loss_fn,
    max_epochs: int = 500,
    device: str = "cpu",
) -> tuple[float, list[float], list[float], list[dict]]:
    """Train for one fold; return (best_val_mae, train_losses, val_maes, predictions)."""
    model = FiLMGATConv(input_dim, hidden_channels, dropout, heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    best_val_mae    = float("inf")
    best_predictions: list[dict] = []
    no_improve      = 0
    train_losses_: list[float] = []
    val_maes_: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch).squeeze()
            loss = loss_fn(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses_.append(total_loss / max(len(train_loader), 1))

        model.eval()
        preds_all, labels_all, ids_all, phs_all = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out   = model(batch).squeeze()
                preds_all.extend(out.cpu().tolist())
                labels_all.extend(batch.y.squeeze().cpu().tolist())
                ids_all.extend(getattr(batch, "PDB_ID", [None] * batch.num_graphs))
                phs_all.extend(batch.pH.view(-1).cpu().tolist())

        val_mae = float(np.mean(np.abs(np.array(preds_all) - np.array(labels_all))))
        val_maes_.append(val_mae)

        if epoch >= 60 and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_predictions = [{
                "predictions": preds_all,
                "labels":      labels_all,
                "ids":         ids_all,
                "pHs":         phs_all,
            }]
            no_improve = 0
        elif epoch >= 60:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_mae, train_losses_, val_maes_, best_predictions


# ════════════════════════════════════════════════════════════════════════════
# Grid search for one hyperparameter combo × one dataset (residue-grouped CV)
# ════════════════════════════════════════════════════════════════════════════

def run_one_combo(
    dataset_idx: int,
    dataset: list,
    input_dim: int,
    hidden_channels: int,
    heads: int,
    dropout: float,
    lr: float,
    batch_size: int,
    k_folds: int,
    patience: int,
    loss_fn,
    results_dir: Path,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    # Group by (PDB, chain, residue_number) so per-pH replicates stay together
    groups = [
        f"{getattr(d, 'PDB_ID', '')}_{getattr(d, 'Chain_ID', '')}_"
        f"{getattr(d, 'Residue_Number', '')}"
        for d in dataset
    ]
    unique_groups = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    order = {g: i for i, g in enumerate(unique_groups)}
    shuffled_groups = [order[g] for g in groups]
    n_splits = min(k_folds, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    fold_maes: list[float] = []
    all_best_predictions: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(dataset, groups=shuffled_groups)
    ):
        train_data = [dataset[i] for i in train_idx]
        val_data   = [dataset[i] for i in val_idx]

        best_mae, _, _, fold_preds = train_one_fold(
            train_data, val_data, input_dim,
            hidden_channels, heads, dropout, lr, batch_size, patience,
            loss_fn, device=device,
        )
        fold_maes.append(best_mae)
        for p in fold_preds:
            p["fold"] = fold
            all_best_predictions.append(p)

    mean_mae = float(np.mean(fold_maes))
    save_predictions_to_csv(
        all_best_predictions, dataset_idx, loss_fn,
        hidden_channels, batch_size, patience, k_folds, lr, dropout, heads,
        results_dir,
    )
    return {
        "dataset_idx":      dataset_idx,
        "hidden_channels":  hidden_channels,
        "heads":            heads,
        "dropout":          dropout,
        "lr":               lr,
        "batch_size":       batch_size,
        "k_folds":          n_splits,
        "patience":         patience,
        "loss":             loss_fn.__class__.__name__,
        "mean_val_mae":     mean_mae,
        "fold_maes":        fold_maes,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="FFX FiLM titration grid search")
    parser.add_argument("--dataset",     type=int, default=None, help="Run only this dataset index (0-4)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--single-core", action="store_true", help="Disable joblib parallelism")
    parser.add_argument("--seed",        type=int, default=42, help="Seed for group shuffling")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir : {args.dataset_dir}")
    print(f"Results dir : {args.results_dir}")

    data_sets, input_dim = load_training_data(args.dataset_dir)
    print(f"input_dim   : {input_dim}  ({len(data_sets)} radius variants)")

    # ── Hyperparameter grid (matches 08_grid_search.py) ─────────────────────
    HEADS           = [4, 6, 8]
    HIDDEN_CHANNELS = [16, 32, 48, 64]
    BATCH_SIZES     = [16, 24, 32, 40]
    LRS             = [0.001, 0.006, 0.01, 0.06, 0.1]
    DROPOUTS        = [0.2, 0.3, 0.4, 0.5]
    LOSS_FNS        = [
        torch.nn.SmoothL1Loss(beta=0.5),
        torch.nn.L1Loss(),
        torch.nn.MSELoss(),
    ]
    K_FOLDS  = 10
    PATIENCE = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")

    combos = list(product(HEADS, HIDDEN_CHANNELS, BATCH_SIZES, LRS, DROPOUTS, LOSS_FNS))
    print(f"Total combos: {len(combos)} × {len(data_sets)} datasets = "
          f"{len(combos)*len(data_sets)}")

    dataset_range = [args.dataset] if args.dataset is not None else range(len(data_sets))

    all_results: list[dict] = []
    for ds_idx in dataset_range:
        dataset = data_sets[ds_idx]
        print(f"\n── Dataset {ds_idx} ({len(dataset)} graphs) ──")

        if args.single_core:
            for hd, hc, bs, lr, dp, lf in combos:
                r = run_one_combo(ds_idx, dataset, input_dim, hc, hd, dp, lr, bs,
                                  K_FOLDS, PATIENCE, lf, args.results_dir,
                                  seed=args.seed, device=device)
                all_results.append(r)
                print(f"  {r['loss']} h={hc} lr={lr} hd={hd} d={dp} b={bs}  "
                      f"MAE={r['mean_val_mae']:.4f}")
        else:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_one_combo)(
                    ds_idx, dataset, input_dim, hc, hd, dp, lr, bs,
                    K_FOLDS, PATIENCE, lf, args.results_dir,
                    args.seed, device,
                )
                for hd, hc, bs, lr, dp, lf in combos
            )
            all_results.extend(results)

    # ── Save results ───────────────────────────────────────────────────────
    flat = []
    for r in all_results:
        flat.append({k: v for k, v in r.items() if k != "fold_maes"})
    df = pd.DataFrame(flat)
    df.to_csv(args.results_dir / "grid_search_results.csv", index=False)

    best = df.loc[df["mean_val_mae"].idxmin()]
    best.to_frame().T.to_csv(args.results_dir / "best_result.csv", index=False)

    print(f"\nBest MAE : {best['mean_val_mae']:.4f}")
    print(f"  loss   : {best['loss']}")
    print(f"  hidden : {best['hidden_channels']}  heads={best['heads']}")
    print(f"  lr     : {best['lr']}  dropout={best['dropout']}  batch={best['batch_size']}")
    print(f"\nResults  → {args.results_dir / 'grid_search_results.csv'}")
    print(f"Best row → {args.results_dir / 'best_result.csv'}")


if __name__ == "__main__":
    main()
