
from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# ── Configuration ───────────────────────────────────────────────────────────────
DEFAULT_FEAT_DIR = Path("Graph_pKa/Features")  # legacy fallback when --mode not given
RADII            = [7, 8, 9, 10, 11]

# Stem pattern with optional per-pH suffix produced by 05_prepare_features.py
# Example stems:
#   '135L_A_35.Aspartate'              (rotopt mode, no pH)
#   '135L_A_35.Aspartate__pH3.94'      (titrate mode, pH appended)
_PH_SUFFIX_RE = re.compile(r"__pH(?P<ph>-?\d+(?:\.\d+)?)$")

# Must match 04_prepare_features.py
NUM_ATOM_LABEL_CLASSES = 10   # labels 0-9; 9 = sidechain S (CYS)
EDGE_FEAT_DIR_NAME     = "Edge_Features"   # sibling of Adjacency_Matrices/
EDGE_FEAT_COLS         = ["dx", "dy", "dz", "distance"]  # 4 edge features
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def parse_stem(stem: str) -> tuple[str, str, int, str, float | None]:
    """Parse a feature-CSV stem into (pdb_id, chain, resseq, res_name, pH).

    Stem formats produced by 05_prepare_features.py::

        rotopt : '{PDB}_{chain}_{resseq}.{ResName}'
        titrate: '{PDB}_{chain}_{resseq}.{ResName}__pH{pH}'

    pH is None for rotopt-mode stems.
    """
    # Strip optional pH suffix
    m  = _PH_SUFFIX_RE.search(stem)
    ph: float | None = None
    if m:
        ph    = float(m.group("ph"))
        stem  = stem[: m.start()]

    dot_idx = stem.rfind(".")
    if dot_idx == -1:
        raise ValueError(f"Cannot parse stem (no '.'): {stem!r}")
    res_name  = stem[dot_idx + 1:]          # e.g. 'Aspartate'
    left      = stem[:dot_idx]              # e.g. '2FWF_A_123'

    under_idx = left.rfind("_")
    if under_idx == -1:
        raise ValueError(f"Cannot parse stem (no '_' before resseq): {stem!r}")
    resseq    = int(left[under_idx + 1:])   # e.g. 123
    id_chain  = left[:under_idx]            # e.g. '2FWF_A'

    under2    = id_chain.rfind("_")
    if under2 == -1:
        raise ValueError(f"Cannot parse stem (no chain '_'): {stem!r}")
    pdb_id    = id_chain[:under2]           # e.g. '2FWF'
    chain     = id_chain[under2 + 1:]       # e.g. 'A'

    return pdb_id, chain, resseq, res_name, ph

def build_data_list(adj_dir: Path, node_dir: Path, radius: int) -> list[Data]:
    """Build a list of PyG Data objects for one radius directory."""
    data_list: list[Data] = []
    skipped_missing = 0
    skipped_bad     = 0
    node_feat_dir   = node_dir / str(radius)
    bond_dir        = adj_dir.parent.parent / EDGE_FEAT_DIR_NAME

    adj_files = sorted(adj_dir.glob("*_adjacency.csv"))
    if not adj_files:
        log.warning(f"  No adjacency CSVs found in {adj_dir}")
        return data_list

    for adj_path in adj_files:
        # Derive the matching node-feature filename
        stem            = adj_path.stem.replace("_adjacency", "")   # strip '_adjacency'
        feat_path       = node_feat_dir / f"{stem}.csv"

        if not feat_path.exists():
            skipped_missing += 1
            continue

        try:
            pdb_id, chain, resseq, res_name, ph = parse_stem(stem)
        except ValueError as exc:
            log.warning(f"  Skipping {stem}: {exc}")
            skipped_bad += 1
            continue

        # ── Load adjacency matrix ─────────────────────────────────────────
        adj_mat    = pd.read_csv(adj_path, header=0, index_col=0).values
        adj_tensor = torch.tensor(adj_mat, dtype=torch.int)
        edge_index = torch.stack(adj_tensor.nonzero(as_tuple=True), dim=0)

        # ── Load edge features (dx, dy, dz, distance in local frame) ─────
        bond_path = bond_dir / f"{stem}_bonds.csv"
        edge_attr: torch.Tensor | None = None
        if bond_path.exists():
            bond_df = pd.read_csv(bond_path)
            # Sort bonds to match edge_index ordering (row-major, same as nonzero)
            bond_df = bond_df.sort_values(["atom_i", "atom_j"]).reset_index(drop=True)
            edge_attr = torch.tensor(
                bond_df[EDGE_FEAT_COLS].values, dtype=torch.float
            )

        # ── Load node features ────────────────────────────────────────────
        nf = pd.read_csv(feat_path, header=0)

        if "atom_label" not in nf.columns:
            log.warning(f"  {stem}: missing 'atom_label' column – skipping")
            skipped_bad += 1
            continue

        # pKa label
        if "Expt. pKa" not in nf.columns or nf["Expt. pKa"].isna().all():
            log.warning(f"  {stem}: missing 'Expt. pKa' – skipping")
            skipped_bad += 1
            continue
        pka_tensor = torch.tensor([float(nf["Expt. pKa"].iloc[0])], dtype=torch.float)

        # residue_label: index of this residue type from the one-hot columns
        ohe_cols = nf.filter(like="Residue Name_").values
        if ohe_cols.shape[1] == 0:
            residue_label = -1
        else:
            residue_label = int(np.argmax(ohe_cols, axis=1)[0])

        # Build feature matrix: drop pKa & raw atom_label, append one-hot atom_label
        cols_to_drop  = [c for c in ["Expt. pKa", "atom_label"] if c in nf.columns]
        atom_labels   = torch.tensor(nf["atom_label"].values, dtype=torch.long)
        atom_label_oh = F.one_hot(atom_labels, num_classes=NUM_ATOM_LABEL_CLASSES).float()

        numeric_feats = nf.drop(columns=cols_to_drop)
        # Drop any non-numeric columns that may have crept in (e.g. index col)
        numeric_feats = numeric_feats.select_dtypes(include=[np.number])
        # Fill NaN (e.g. Perm_* columns absent when no .uperm file exists) with 0
        numeric_feats = numeric_feats.fillna(0.0)
        feat_tensor   = torch.tensor(numeric_feats.values, dtype=torch.float)
        feat_tensor   = torch.cat([feat_tensor, atom_label_oh], dim=1)

        data = Data(
            x              = feat_tensor,
            edge_index     = edge_index,
            edge_attr      = edge_attr,
            y              = pka_tensor,
            residue_label  = residue_label,
        )
        data.PDB_ID         = pdb_id
        data.Chain_ID       = chain
        data.Residue_Number = resseq
        data.Residue_Name   = res_name
        # pH is a per-graph scalar (NaN for rotopt mode).  Stored as a tensor so
        # PyG batches it correctly via DataLoader.
        data.pH = torch.tensor([float(ph) if ph is not None else float("nan")],
                               dtype=torch.float)

        data_list.append(data)

    log.info(f"  Radius {radius:2d} Å : {len(data_list)} graphs built  "
             f"(skipped – missing feat: {skipped_missing}, bad format: {skipped_bad})")
    return data_list

def main(feat_dir: Path, out_dir: Path) -> None:
    adj_dir  = feat_dir / "Adjacency_Matrices" / "With_Self_Loop"
    node_dir = feat_dir / "Node_Feature_Vectors"

    if not adj_dir.is_dir():
        log.error(f"Adjacency directory not found: {adj_dir}")
        return
    if not node_dir.is_dir():
        log.error(f"Node-feature directory not found: {node_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, radius in enumerate(RADII):
        log.info(f"Building dataset for radius {radius} Å …")
        data_list = build_data_list(adj_dir, node_dir, radius)

        if not data_list:
            log.warning(f"  No graphs produced for radius {radius} – pkl not written.")
            continue

        pkl_path = out_dir / f"data_list_{idx}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(data_list, fh)
        log.info(f"  Saved {len(data_list)} graphs → {pkl_path}")

    log.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyG datasets from 05_prepare_features output")
    parser.add_argument("--mode", choices=["rotopt", "titrate"], default=None,
                        help="If set, derives --feat-dir as Graph_pKa/Features_{mode} and "
                             "--out-dir as Graph_pKa/Features_{mode}/Datasets.")
    parser.add_argument("--feat-dir", type=Path, default=None,
                        help="Override feature directory (default depends on --mode)")
    parser.add_argument("--out-dir",  type=Path, default=None,
                        help="Output directory for .pkl files (default: <feat-dir>/Datasets)")
    args = parser.parse_args()

    if args.feat_dir is not None:
        feat_dir = args.feat_dir
    elif args.mode is not None:
        feat_dir = Path(f"Graph_pKa/Features_{args.mode}")
    else:
        feat_dir = DEFAULT_FEAT_DIR
    out_dir = args.out_dir if args.out_dir is not None else feat_dir / "Datasets"

    log.info(f"Feature dir : {feat_dir}")
    log.info(f"Output dir  : {out_dir}")
    main(feat_dir, out_dir)
