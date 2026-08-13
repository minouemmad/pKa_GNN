
from __future__ import annotations

import argparse
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data  # noqa: F401
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = Path("Graph_pKa/Features/Datasets")
DEFAULT_RESULTS_DIR = Path("Graph_pKa/Results/Training")
RADII               = [7, 8, 9, 10, 11]

# Default pH bucket centres for the multi_branch architecture.
# Must match TITRATION_PHS in 03_run_ffx_minimize.py.
DEFAULT_PHS = [3.94, 4.4, 6.45, 8.55]

# pH conditioning normalisation — maps pH ∈ [3, 9] to roughly [-1, 1].
_PH_MEAN = 6.0
_PH_STD  = 3.0

# Feature-group regex map for --ablate (must match FEATURE_GROUPS in 09).
# Columns matching the chosen group(s) are zeroed before training so the
# model cannot use them; input_dim is preserved so checkpoints stay
# loadable by 09_feature_analysis.py.
ABLATION_GROUPS: dict[str, str] = {
    "residue_one_hot":    r"^Residue Name_",
    "local_coords":       r"^recalculated_[xyz]$",
    "induced_dipoles":    r"^Dipole_[XYZ]$",
    "perm_multipoles":    r"^Perm_",
    "hbonds":             r"^Number of H-Bonds",
    "sasa":               r"^SASA_Value$",
    "protonation":        r"^is_protonated$",
    "pH_col":             r"^pH$",
    "neighbour_counts":   r"^Radius_\d+A_",
    "atom_label_one_hot": r"^atom_label_oh_\d+$",
}
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_feature_names(feat_dir: Path, radius: int, n_cols: int) -> list[str]:
    """Reproduce the column ordering used by 06_create_datasets.py.

    Reads one feature CSV from {feat_dir}/Node_Feature_Vectors/{radius}/, drops
    'Expt. pKa' + 'atom_label', keeps numeric columns, and appends the 10-dim
    one-hot atom-label encoding that 06 appends.
    """
    radius_dir = feat_dir / "Node_Feature_Vectors" / str(radius)
    candidates = sorted(radius_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No feature CSVs in {radius_dir}")
    df = pd.read_csv(candidates[0])
    drop_set = {"Expt. pKa", "atom_label"}
    numeric  = df.drop(columns=[c for c in drop_set if c in df.columns])\
                 .select_dtypes(include=[np.number])
    names = list(numeric.columns) + [f"atom_label_oh_{i}" for i in range(10)]
    if len(names) != n_cols:
        if len(names) < n_cols:
            names += [f"feat_{i}" for i in range(len(names), n_cols)]
        else:
            names = names[:n_cols]
    return names

def _ablate_columns(data_list, groups: list[str], feat_dir: Path, radius: int) -> list[int]:
    """Zero `x` columns belonging to the requested feature groups.  Returns the
    indices that were zeroed (for logging)."""
    if not groups or not data_list:
        return []
    n_cols = data_list[0].x.shape[1]
    names  = _reconstruct_feature_names(feat_dir, radius, n_cols)
    zero_cols: list[int] = []
    for g in groups:
        if g not in ABLATION_GROUPS:
            raise ValueError(f"Unknown ablation group: {g}.  "
                             f"Choose from {sorted(ABLATION_GROUPS)}")
        pattern = ABLATION_GROUPS[g]
        zero_cols.extend(i for i, n in enumerate(names) if re.match(pattern, n))
    zero_cols = sorted(set(zero_cols))
    if not zero_cols:
        return []
    cols_t = torch.tensor(zero_cols, dtype=torch.long)
    for d in data_list:
        d.x[:, cols_t] = 0.0
    return zero_cols

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
# Models — multiple pH-conditioning architectures
# ════════════════════════════════════════════════════════════════════════════
#
# All architectures share the same GATv2Conv backbone.  They differ only in
# how the per-graph scalar pH (data.pH) is folded into the prediction.
#
#   naive        : pH is left as one extra column in the node features
#                   (the way 05_prepare_features.py exports it).  No special
#                   conditioning — each (residue × pH) is an independent sample.
#
#   concat       : graph-pooled embedding is concatenated with the (normalised)
#                   pH scalar before the linear head.  Cheap and well-known.
#
#   film         : Feature-wise Linear Modulation (Perez et al., AAAI-18).
#                   A small MLP turns pH into per-channel (γ, β) and we replace
#                   the post-conv hidden state h with γ ⊙ h + β.  Empirically
#                   the strongest scalar-conditioning method in the literature.
#
#   multi_branch : One GATv2 branch per pH bucket; each graph activates the
#                   branch matching its pH (other branches contribute zero).
#                   Pooled embeddings are concatenated and decoded by a shared
#                   linear head.  Mirrors the user's "4 NNs concatenated" idea
#                   but trained jointly so the head can compare buckets.
#
#   gated        : pH → sigmoid scalar gate.  Final prediction is
#                   `µ(graph) + gate(pH) * Δ(graph)` where µ and Δ are two
#                   linear heads on top of the pooled embedding.  This is the
#                   "pH as a weight" idea: pH multiplicatively modulates a
#                   learned correction term, leaving the pH-independent part
#                   intact.
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_ph(ph: torch.Tensor) -> torch.Tensor:
    """Normalise a (B,) pH tensor to roughly [-1, 1].  NaN → 0 (rotopt)."""
    out = (ph - _PH_MEAN) / _PH_STD
    return torch.nan_to_num(out, nan=0.0)

def _ph_bucket(ph: torch.Tensor, ph_centres: torch.Tensor) -> torch.Tensor:
    """Assign each graph's pH to the closest bucket index in *ph_centres*.
    NaN pHs (rotopt) → bucket 0 (arbitrary; multi_branch is titrate-only anyway).
    """
    ph = torch.nan_to_num(ph, nan=ph_centres[0].item())
    diffs = (ph[:, None] - ph_centres[None, :]).abs()
    return diffs.argmin(dim=1)

class GATModel(torch.nn.Module):
    """Single-branch GATv2 with optional pH conditioning.

    arch ∈ {'naive', 'concat', 'film', 'gated'} — multi_branch lives in its
    own class below.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        heads: int,
        dropout: float,
        edge_dim: int | None = None,
        arch: str = "naive",
    ):
        super().__init__()
        assert arch in ("naive", "concat", "film", "gated")
        self.arch  = arch
        self.conv1 = GATv2Conv(input_dim, hidden_channels, heads=heads,
                                concat=True, add_self_loops=False,
                                edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.pool    = global_mean_pool
        emb_dim      = hidden_channels * heads

        if arch == "film":
            # Two MLPs producing per-channel γ and β from pH (a single scalar)
            self.film = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2 * emb_dim),
            )

        if arch == "concat":
            self.head = torch.nn.Linear(emb_dim + 1, 1)
        elif arch == "gated":
            self.head_mu    = torch.nn.Linear(emb_dim, 1)
            self.head_delta = torch.nn.Linear(emb_dim, 1)
            # learned scalar transform of normalised pH → sigmoid gate
            self.gate_mlp   = torch.nn.Sequential(
                torch.nn.Linear(1, 8), torch.nn.ReLU(),
                torch.nn.Linear(8, 1),
            )
        else:  # 'naive' or 'film'
            self.head = torch.nn.Linear(emb_dim, 1)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        if self.arch == "film":
            # Per-graph conditioning broadcast to nodes
            ph = _normalize_ph(data.pH.view(-1, 1)).to(x.device)
            gb = self.film(ph)                                  # (B, 2*emb)
            gamma, beta = gb.chunk(2, dim=-1)
            # initialise so it starts as identity (γ≈1, β≈0) at random init.
            gamma = 1.0 + gamma
            # broadcast (B, emb) → (N, emb) using batch index
            gamma_n = gamma[data.batch]
            beta_n  = beta[data.batch]
            x = gamma_n * x + beta_n

        x = self.dropout(x)
        x = self.pool(x, data.batch)

        if self.arch == "concat":
            ph = _normalize_ph(data.pH.view(-1, 1)).to(x.device)
            x = torch.cat([x, ph], dim=-1)
            return self.head(x)

        if self.arch == "gated":
            ph   = _normalize_ph(data.pH.view(-1, 1)).to(x.device)
            gate = torch.sigmoid(self.gate_mlp(ph))
            mu    = self.head_mu(x)
            delta = self.head_delta(x)
            return mu + gate * delta

        return self.head(x)        # naive / film

class MultiBranchGATModel(torch.nn.Module):
    """Independent GATv2 branch per pH bucket; concatenated pooled embeddings.

    For each graph only the matching branch's pooled embedding is non-zero;
    the others are masked.  This realises the user's "one NN per pH, then
    concat" idea but trained jointly so the final linear head can learn to
    weight buckets.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        heads: int,
        dropout: float,
        edge_dim: int | None = None,
        ph_centres: list[float] | None = None,
    ):
        super().__init__()
        ph_centres = ph_centres or DEFAULT_PHS
        self.register_buffer("ph_centres", torch.tensor(ph_centres, dtype=torch.float))
        self.n_buckets = len(ph_centres)

        self.branches = torch.nn.ModuleList([
            GATv2Conv(input_dim, hidden_channels, heads=heads,
                      concat=True, add_self_loops=False, edge_dim=edge_dim)
            for _ in range(self.n_buckets)
        ])
        emb_dim = hidden_channels * heads
        self.dropout = torch.nn.Dropout(dropout)
        self.pool    = global_mean_pool
        self.head    = torch.nn.Linear(emb_dim * self.n_buckets, 1)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        bucket = _ph_bucket(data.pH.to(self.ph_centres.device), self.ph_centres)  # (B,)
        outs = []
        for k, conv in enumerate(self.branches):
            h = conv(data.x, data.edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = self.dropout(h)
            pooled = self.pool(h, data.batch)               # (B, emb)
            mask   = (bucket == k).float().unsqueeze(-1)    # (B, 1)
            outs.append(pooled * mask)
        cat = torch.cat(outs, dim=-1)                        # (B, n_buckets * emb)
        return self.head(cat)

def build_model(arch: str, input_dim: int, args) -> torch.nn.Module:
    edge_dim = args.edge_dim if hasattr(args, "edge_dim") else None
    if arch == "multi_branch":
        return MultiBranchGATModel(
            input_dim, args.hidden, args.heads, args.dropout,
            edge_dim=edge_dim, ph_centres=args.phs,
        )
    return GATModel(input_dim, args.hidden, args.heads, args.dropout,
                    edge_dim=edge_dim, arch=arch)
# ─────────────────────────────────────────────────────────────────────────────

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
    args.edge_dim = edge_dim
    print(f"  arch={args.arch}, input_dim={input_dim}, edge_dim={edge_dim}, hidden={args.hidden}, heads={args.heads}, "
          f"lr={args.lr}, dropout={args.dropout}, batch={args.batch}")

    model_dir = results_dir / "models" / f"dataset_{dataset_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir  = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    kf             = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_preds:  list[dict] = []
    fold_maes:  list[float] = []
    fold_rmses: list[float] = []

    # Optional group-aware splitting (keeps all (PDB, residue, *pH) replicates in same fold)
    group_mode = getattr(args, "group_split", "none")
    if group_mode and group_mode != "none":
        if group_mode == "pdb":
            groups = [str(d.PDB_ID) for d in data_list]
        elif group_mode == "residue":
            groups = [f"{d.PDB_ID}_{getattr(d, 'Chain_ID', '')}_{d.Residue_Number}"
                      for d in data_list]
        else:
            raise ValueError(f"Unknown --group-split: {group_mode}")
        # GroupKFold is deterministic; shuffle group order with --seed so seeds vary the splits.
        unique_groups = sorted(set(groups))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(unique_groups)
        # Re-index so GroupKFold picks folds in the shuffled order
        order = {g: i for i, g in enumerate(unique_groups)}
        shuffled_groups = [order[g] for g in groups]
        n_groups = len(unique_groups)
        n_splits = min(args.folds, n_groups)
        if n_splits < args.folds:
            print(f"  [--group-split {group_mode}] only {n_groups} groups; "
                  f"reducing folds {args.folds} -> {n_splits}")
        gkf = GroupKFold(n_splits=n_splits)
        split_iter = gkf.split(data_list, groups=shuffled_groups)
        n_folds_eff = n_splits
        print(f"  [--group-split {group_mode}] {n_groups} unique groups across "
              f"{len(data_list)} graphs (seed={args.seed})")
    else:
        split_iter = kf.split(data_list)
        n_folds_eff = args.folds

    for fold, (train_idx, val_idx) in enumerate(split_iter, start=1):
        print(f"\n  Fold {fold}/{n_folds_eff}  (train={len(train_idx)}, val={len(val_idx)})")
        train_data  = [data_list[i] for i in train_idx]
        val_data    = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=args.batch, shuffle=False)

        model     = build_model(args.arch, input_dim, args)
        # Persist edge_dim for build_model on the args namespace
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.loss == "L1":
            loss_fn = torch.nn.L1Loss()
        elif args.loss == "MSE":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
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
    parser.add_argument("--exclude-cys", action="store_true",
                        help="Exclude only CYS residues from training")
    parser.add_argument("--pdb-subset", type=str, default=None,
                        help="Restrict training to a subset of PDB IDs.  Either a comma-separated "
                             "list (e.g. '1A91,1BNZ,...') or a path to a text file with one PDB ID "
                             "per line.  Used for fair head-to-head comparisons (e.g. rotopt vs titrate).")
    parser.add_argument("--group-split", choices=["none", "pdb", "residue"], default="none",
                        help="Use GroupKFold with groups by PDB or by (PDB, chain, residue_number). "
                             "Required for titrate mode to keep all 4 pH replicates of a residue "
                             "in the same fold and avoid leakage.")
    parser.add_argument("--ablate", nargs="+", default=[],
                        choices=sorted(ABLATION_GROUPS.keys()),
                        help="Zero out one or more feature groups before training "
                             "(input_dim is preserved so the same checkpoint format "
                             "is loadable by 09_feature_analysis.py).")
    parser.add_argument("--feat-dir", type=Path, default=None,
                        help="Feature directory (Node_Feature_Vectors/<radius>/) used to "
                             "reconstruct column names for --ablate.  Defaults to "
                             "Graph_pKa/Features_{mode}/ when --mode is set.")
    parser.add_argument("--mode",  choices=["rotopt", "titrate"], default=None,
                        help="Selects default dataset dir Graph_pKa/Features_{mode}/Datasets and "
                             "results subdir tag.  If unset uses --dataset-dir directly.")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Override dataset directory (default depends on --mode)")
    parser.add_argument("--arch", choices=["naive", "concat", "film",
                                            "multi_branch", "gated"],
                        default="naive",
                        help="pH-conditioning architecture (titrate mode).  In rotopt mode all "
                             "non-naive archs effectively reduce to vanilla GAT (pH=0).")
    parser.add_argument("--phs", type=float, nargs="+", default=DEFAULT_PHS,
                        help="pH bucket centres for multi_branch (default %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Override output directory (default: auto-derived from flags)")
    parser.add_argument("--loss", choices=["SmoothL1", "L1", "MSE"], default="SmoothL1",
                        help="Loss function (default SmoothL1Loss(beta=0.5))")
    args = parser.parse_args()

    set_seed(args.seed)

    # Dataset directory resolution
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    elif args.mode is not None:
        dataset_dir = Path(f"Graph_pKa/Features_{args.mode}/Datasets")
    else:
        dataset_dir = DEFAULT_DATASET_DIR

    # Determine output directory — avoid overwriting previous runs
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        suffix = ""
        if args.mode is not None:
            suffix += f"_{args.mode}"
        suffix += f"_{args.arch}"
        if args.exclude_cys_tyr:
            suffix += "_no_cys_tyr"
        if args.exclude_cys:
            suffix += "_no_cys"
        if args.ablate:
            suffix += "_ablate-" + "-".join(sorted(args.ablate))
        if args.group_split and args.group_split != "none":
            suffix += f"_group-{args.group_split}"
        if args.seed != 42:
            suffix += f"_seed{args.seed}"
        if args.no_edge_attr:
            suffix += "_no_edge"
        results_dir = DEFAULT_RESULTS_DIR.parent / (DEFAULT_RESULTS_DIR.name + suffix)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset → {dataset_dir}")
    print(f"Results → {results_dir}")

    # Decide which datasets to run
    if args.dataset == "all":
        indices = list(range(len(RADII)))
    else:
        indices = [int(args.dataset)]

    summary_rows: list[dict] = []

    for idx in indices:
        pkl_path = dataset_dir / f"data_list_{idx}.pkl"
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

        if args.exclude_cys:
            before = len(data_list)
            data_list = [d for d in data_list if d.Residue_Name != "Cysteine"]
            print(f"  [--exclude-cys] {before - len(data_list)} graphs removed "
                  f"(CYS); {len(data_list)} remaining")

        if args.pdb_subset:
            sub_arg = args.pdb_subset
            sub_path = Path(sub_arg)
            if sub_path.exists():
                pdb_set = {ln.strip().upper() for ln in sub_path.read_text().splitlines() if ln.strip()}
            else:
                pdb_set = {p.strip().upper() for p in sub_arg.split(",") if p.strip()}
            before = len(data_list)
            data_list = [d for d in data_list if str(d.PDB_ID).upper() in pdb_set]
            print(f"  [--pdb-subset] kept {len(data_list)}/{before} graphs across "
                  f"{len(pdb_set)} PDB IDs")

        if args.ablate:
            feat_dir = args.feat_dir
            if feat_dir is None and args.mode is not None:
                feat_dir = Path(f"Graph_pKa/Features_{args.mode}")
            if feat_dir is None:
                raise ValueError("--ablate requires --feat-dir or --mode to locate "
                                 "feature CSVs for column-name reconstruction.")
            zeroed = _ablate_columns(data_list, args.ablate, feat_dir, RADII[idx])
            print(f"  [--ablate {' '.join(args.ablate)}] zeroed {len(zeroed)} columns: {zeroed}")

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
