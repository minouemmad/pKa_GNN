from __future__ import annotations

import argparse
import importlib.util
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

THIS_DIR = Path(__file__).resolve().parent
TRAIN_PY = THIS_DIR / "05_train.py"

# Dynamically load 05_train.py module to reuse GATModelPaper
spec = importlib.util.spec_from_file_location("paper_train", TRAIN_PY)
paper_train = importlib.util.module_from_spec(spec)
sys.modules["paper_train"] = paper_train
spec.loader.exec_module(paper_train)  # type: ignore[arg-type]
GATModelPaper = paper_train.GATModelPaper

# Feature column names matching the saved CSV order in 02_prepare_features.py.
# After 9-class atom_label OHE is appended, the full layout (per dataset) is:
#   block 0: residue OHE (4)         "Residue Name_{Aspartate,Glutamate,Histidine,Lysine}"
#   block 1: recalculated_x/y/z (3)
#   block 2: Dipole_X/Y/Z (3)        — present only if not dropped
#   block 3: atomic_charge (1)       — present only if not dropped
#   block 4: PermDipole_X/Y/Z (3)    — present only if not dropped
#   block 5: Number of H-Bonds donor/acceptor (2)
#   block 6: SASA_Value (1)
#   block 7: Radius_9A_{N,CA_C,O,S}_Count (4)
#   block 8: atom_label OHE (9)      — appended at end

CANDIDATE_FEATURES = {
    # Single-feature names (dim=1)
    "atomic_charge":              ["atomic_charge"],
    "SASA":                       ["SASA_Value"],
    "Hbond_donor":                ["Number of H-Bonds as donor"],
    "Hbond_acceptor":             ["Number of H-Bonds as acceptor"],
    # Group names (multi-dim, permuted as a block)
    "Residue_OHE":                ["Residue Name_Aspartate", "Residue Name_Glutamate",
                                   "Residue Name_Histidine", "Residue Name_Lysine"],
    "LocalCoords":                ["recalculated_x", "recalculated_y", "recalculated_z"],
    "InducedDipole":              ["Dipole_X", "Dipole_Y", "Dipole_Z"],
    "PermDipole":                 ["PermDipole_X", "PermDipole_Y", "PermDipole_Z"],
    "InducedDipole_invariants":   ["Dipole_norm", "Dipole_align_z", "Dipole_field_proj"],
    "PermDipole_invariants":      ["PermDipole_norm", "PermDipole_align_z", "PermDipole_field_proj"],
    "RadiusCounts":               ["Radius_9A_N_Count", "Radius_9A_CA_C_Count",
                                   "Radius_9A_O_Count", "Radius_9A_S_Count"],
    # Atom-label OHE block sits at end of feature vector (last 9 dims)
    "AtomLabel_OHE":              ["__ATOM_LABEL_OHE__"],
}

def get_feature_layout(dataset_dir: Path, ds_idx: int) -> tuple[list, list[str]]:
    """Return (data_list, expected_column_order) for a dataset.

    The column ordering must mirror what 03_create_datasets.py produces:
    numeric columns from the saved CSV (in CSV order) MINUS dropped columns,
    followed by 9-dim atom_label OHE.
    """
    pkl = dataset_dir / f"data_list_{ds_idx}.pkl"
    with open(pkl, "rb") as fh:
        data_list = pickle.load(fh)
    return data_list

def find_feature_indices(input_dim: int, feat_dir: Path, dataset_dir: Path,
                         ds_idx: int, drop_cols: list[str]) -> dict[str, list[int]]:
    """Determine {group_name: [column_indices]} into data.x.

    The dataset's column order matches 03_create_datasets.py:
        kept_numeric_cols = [c for c in CSV_numeric_cols if c not in drop_cols]
        feat = kept_numeric_cols  (dim = len(kept))
             + atom_label_one_hot (dim = 9, appended at end)
    """
    node_dir = feat_dir / "Node_Feature_Vectors" / "9"
    sample_csv = next(node_dir.glob("*.csv"))
    nf = pd.read_csv(sample_csv, header=0)
    pka_col = next((c for c in ("Expt.pKa", "Expt. pKa", "Expt_pKa") if c in nf.columns), None)
    fixed_drop = [c for c in (pka_col, "atom_label") if c in nf.columns]
    # Helper columns that 03_create_datasets.py always strips from data.x
    HELPER_COLS = (
        "x_lab", "y_lab", "z_lab",
        "Dipole_lab_X", "Dipole_lab_Y", "Dipole_lab_Z",
        "PermDipole_lab_X", "PermDipole_lab_Y", "PermDipole_lab_Z",
    )
    fixed_drop += [c for c in HELPER_COLS if c in nf.columns and c not in fixed_drop]
    numeric_cols = nf.drop(columns=fixed_drop).select_dtypes(include=[np.number]).columns.tolist()

    drop_set = set(drop_cols)
    kept_cols = [c for c in numeric_cols if c not in drop_set]

    data_list = get_feature_layout(dataset_dir, ds_idx)
    actual_dim = data_list[0].x.shape[1]
    expected_dim = len(kept_cols) + 9
    if expected_dim != actual_dim:
        raise RuntimeError(
            f"Column count mismatch: expected_dim={expected_dim} "
            f"(kept={len(kept_cols)} + 9 atom OHE) but actual_dim={actual_dim}. "
            f"Check --drop-cols matches the dataset.")

    atom_ohe_start = len(kept_cols)
    name_to_idx = {c: i for i, c in enumerate(kept_cols)}
    out: dict[str, list[int]] = {}
    for grp, cols in CANDIDATE_FEATURES.items():
        if cols == ["__ATOM_LABEL_OHE__"]:
            out[grp] = list(range(atom_ohe_start, actual_dim))
            continue
        if all(c in name_to_idx for c in cols):
            out[grp] = [name_to_idx[c] for c in cols]
    return out

def evaluate(model: torch.nn.Module, loader: DataLoader, perm_idxs: list[int] | None,
             rng: np.random.Generator) -> float:
    """Compute MAE on loader; if perm_idxs is given, permute those columns of x
    across all nodes in each batch before forward pass."""
    model.eval()
    total_abse = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.x.clone()
            if perm_idxs is not None:
                # Permute across all nodes in this batch (preserves the marginal
                # distribution but breaks the correlation with the structure).
                idx = rng.permutation(x.shape[0])
                x[:, perm_idxs] = x[idx][:, perm_idxs]
            batch.x = x
            out = model(batch).view(-1)
            y   = batch.y.view(-1)
            total_abse += F.l1_loss(out, y, reduction="sum").item()
            n += y.size(0)
    return total_abse / max(n, 1)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir",    type=Path, default=Path("Graph_pKa/Features_FFX138"))
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--ds-idx",      type=int,  default=2)
    ap.add_argument("--models-dir",  type=Path, required=True,
                    help="Directory containing fold_1.pth ... fold_K.pth")
    ap.add_argument("--folds",       type=int,  default=10)
    ap.add_argument("--batch",       type=int,  default=32)
    ap.add_argument("--hidden",      type=int,  default=48)
    ap.add_argument("--heads",       type=int,  default=4)
    ap.add_argument("--dropout",     type=float, default=0.5)
    ap.add_argument("--readout",     type=str,  default="mean")
    ap.add_argument("--num-layers",  type=int,  default=1)
    ap.add_argument("--edge-dim",    type=int,  default=None)
    ap.add_argument("--out",         type=Path, required=True)
    ap.add_argument("--drop-cols",   type=str,  default="",
                    help="Comma-separated CSV columns that were dropped at dataset build time")
    ap.add_argument("--seed",        type=int,  default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load dataset (contains all 312 graphs)
    data_list = get_feature_layout(args.dataset_dir, args.ds_idx)
    input_dim = data_list[0].x.shape[1]
    print(f"input_dim = {input_dim}, n_graphs = {len(data_list)}")

    # Determine feature group → x column indices
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    grp_idx = find_feature_indices(input_dim, args.feat_dir, args.dataset_dir, args.ds_idx, drop_cols)
    print("Feature groups found:")
    for k, v in grp_idx.items():
        print(f"  {k:20s}  cols={v}")

    # Build the same KFold splits used in 05_train.py (random_state=42 hardcoded)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_splits = list(kf.split(data_list))

    # Compute baseline MAE on each fold's val set, plus permuted MAE per group
    rows: list[dict] = []
    base_maes: list[float] = []
    perm_maes: dict[str, list[float]] = {g: [] for g in grp_idx}

    for fold_idx, (_, val_idx) in enumerate(fold_splits, start=1):
        val_data = [data_list[i] for i in val_idx]
        val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False)

        model = GATModelPaper(
            input_dim, args.hidden, args.heads, args.dropout,
            readout=args.readout, num_layers=args.num_layers, edge_dim=args.edge_dim,
        )
        model_path = args.models_dir / f"fold_{fold_idx}.pth"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        base = evaluate(model, val_loader, None, rng)
        base_maes.append(base)

        for grp, idxs in grp_idx.items():
            # Reuse same RNG seed per fold for reproducibility
            fold_rng = np.random.default_rng(args.seed * 1000 + fold_idx)
            mae_perm = evaluate(model, val_loader, idxs, fold_rng)
            perm_maes[grp].append(mae_perm)

        print(f"  fold {fold_idx}: base MAE={base:.4f}")

    base_mean = float(np.mean(base_maes))
    print(f"\nBaseline mean MAE across folds: {base_mean:.4f}")
    print()
    out_rows: list[dict] = []
    for grp in grp_idx:
        m = float(np.mean(perm_maes[grp]))
        s = float(np.std(perm_maes[grp]))
        out_rows.append(dict(
            feature=grp,
            n_cols=len(grp_idx[grp]),
            baseline_MAE=base_mean,
            perm_MAE=m,
            delta_MAE=m - base_mean,
            std_perm_MAE=s,
        ))
    df = pd.DataFrame(out_rows).sort_values("delta_MAE", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
