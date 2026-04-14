"""
09_create_datasets_paper.py

Paper-exact dataset builder — mirrors the original create_data.py from
Graph_pKa/Net/create_data.py as closely as possible, adapted for FFX output.

Key settings matching the paper:
  • num_classes = 9  (atom_label 0-8; no sidechain-S class 9)
  • No edge_attr (paper GAT_1 does not use edge features)
  • Reads from Graph_pKa/Features_Paper/ (produced by 08_prepare_features_paper.py)
  • input_dim = 26 after one-hot encoding atom_label

Outputs (inside Graph_pKa/Features_Paper/Datasets/):
    data_list_0.pkl   <- radius 7 A
    data_list_1.pkl   <- radius 8 A
    data_list_2.pkl   <- radius 9 A
    data_list_3.pkl   <- radius 10 A
    data_list_4.pkl   <- radius 11 A

Optional: pass --feat-dir to point at PKAD_Data for direct paper comparison:
    python 09_create_datasets_paper.py \\
        --feat-dir Graph_pKa/PKAD_Data \\
        --adj-subdir Adj_Matrix/With_Self_Loop \\
        --node-subdir 4_Residues_W_Local_Frame \\
        --out-dir Graph_pKa/PKAD_Data/Subsets_Paper

Run (from pKa_GNN/ as CWD):
    python tinker_pipeline/03_create_datasets.py
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# ── Default paths (paper-exact FFX features) ──────────────────────────────────
FEAT_DIR   = Path("Graph_pKa/Features_Paper")
ADJ_DIR    = FEAT_DIR / "Adjacency_Matrices" / "With_Self_Loop"
NODE_DIR   = FEAT_DIR / "Node_Feature_Vectors"
OUT_DIR    = FEAT_DIR / "Datasets"
RADII      = [7, 8, 9, 10, 11]

# Paper: 9-class atom label OHE (labels 0–8, matching original create_data.py)
NUM_ATOM_LABEL_CLASSES = 9

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_stem(stem: str) -> tuple[str, str, int, str]:
    """'{PDB}_{chain}_{resseq}.{ResName}' → (pdb_id, chain, resseq, res_name)."""
    dot_idx = stem.rfind(".")
    if dot_idx == -1:
        raise ValueError(f"Cannot parse stem (no '.'): {stem!r}")
    res_name  = stem[dot_idx + 1:]
    left      = stem[:dot_idx]

    under_idx = left.rfind("_")
    if under_idx == -1:
        raise ValueError(f"Cannot parse stem (no '_' before resseq): {stem!r}")
    resseq    = int(left[under_idx + 1:])
    id_chain  = left[:under_idx]

    under2    = id_chain.rfind("_")
    if under2 == -1:
        raise ValueError(f"Cannot parse stem (single field): {stem!r}")
    pdb_id    = id_chain[:under2]
    chain     = id_chain[under2 + 1:]
    return pdb_id, chain, resseq, res_name


def build_data_list(adj_dir: Path, node_dir: Path, radius: int) -> list[Data]:
    """Build PyG Data objects for one radius. No edge_attr (paper-exact)."""
    data_list:       list[Data] = []
    skipped_missing = 0
    skipped_bad     = 0
    node_feat_dir   = node_dir / str(radius)

    adj_files = sorted(adj_dir.glob("*_adjacency.csv"))
    if not adj_files:
        # Try without "_adjacency" suffix variant (PKAD_Data naming)
        adj_files = sorted(adj_dir.glob("*.csv"))
    if not adj_files:
        log.warning(f"  No adjacency CSVs found in {adj_dir}")
        return data_list

    for adj_path in adj_files:
        stem = adj_path.stem
        if stem.endswith("_adjacency"):
            stem = stem[:-len("_adjacency")]

        feat_path = node_feat_dir / f"{stem}.csv"
        if not feat_path.exists():
            skipped_missing += 1
            continue

        try:
            pdb_id, chain, resseq, res_name = parse_stem(stem)
        except ValueError as exc:
            log.warning(f"  Skipping {stem}: {exc}")
            skipped_bad += 1
            continue

        # Adjacency → edge_index
        adj_mat    = pd.read_csv(adj_path, header=0, index_col=0).values
        adj_tensor = torch.tensor(adj_mat, dtype=torch.int)
        edge_index = torch.stack(adj_tensor.nonzero(as_tuple=True), dim=0)

        # Node features
        nf = pd.read_csv(feat_path, header=0)

        if "atom_label" not in nf.columns:
            log.warning(f"  {stem}: missing 'atom_label' – skipping")
            skipped_bad += 1
            continue

        # Accept either pKa column name variant produced by different pipeline versions
        pka_col = None
        for candidate in ("Expt.pKa", "Expt. pKa", "Expt_pKa"):
            if candidate in nf.columns:
                pka_col = candidate
                break
        if pka_col is None or nf[pka_col].isna().all():
            log.warning(f"  {stem}: missing pKa column – skipping")
            skipped_bad += 1
            continue
        pka_tensor = torch.tensor([float(nf[pka_col].iloc[0])], dtype=torch.float)

        # residue_label: index of the residue type from its one-hot column
        ohe_cols = nf.filter(like="Residue Name_").values
        residue_label = int(np.argmax(ohe_cols, axis=1)[0]) if ohe_cols.shape[1] > 0 else -1

        # Build feature matrix exactly as create_data.py:
        #   features = [all cols except pKa and atom_label] ++ [atom_label 9-class OHE]
        atom_labels   = torch.tensor(nf["atom_label"].values, dtype=torch.long)
        # Clamp to valid range for num_classes=9 (labels 0–8)
        atom_labels   = atom_labels.clamp(0, NUM_ATOM_LABEL_CLASSES - 1)
        atom_label_oh = F.one_hot(atom_labels, num_classes=NUM_ATOM_LABEL_CLASSES).float()

        cols_to_drop  = [c for c in [pka_col, "atom_label"] if c in nf.columns]
        numeric_feats = nf.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
        feat_tensor   = torch.tensor(numeric_feats.values, dtype=torch.float)
        feat_tensor   = torch.cat([feat_tensor, atom_label_oh], dim=1)

        data = Data(
            x             = feat_tensor,
            edge_index    = edge_index,
            y             = pka_tensor,
            residue_label = residue_label,
        )
        data.PDB_ID         = pdb_id
        data.Chain_ID       = chain
        data.Residue_Number = resseq
        data.Residue_Name   = res_name

        data_list.append(data)

    log.info(f"  Radius {radius:2d} Å : {len(data_list)} graphs  "
             f"(skipped – missing: {skipped_missing}, bad: {skipped_bad})")
    return data_list


def main(feat_dir: Path, adj_subdir: str, node_subdir: str, out_dir: Path) -> None:
    adj_dir  = feat_dir / adj_subdir
    node_dir = feat_dir / node_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not adj_dir.exists():
        log.error(f"Adjacency directory not found: {adj_dir}")
        return

    for idx, radius in enumerate(RADII):
        log.info(f"\nBuilding dataset {idx} (radius {radius} Å) …")
        dl = build_data_list(adj_dir, node_dir, radius)
        if not dl:
            log.warning(f"  Dataset {idx} is empty – skipping.")
            continue

        input_dim = dl[0].x.shape[1]
        log.info(f"  input_dim = {input_dim}  (expected 26 for paper-exact)")

        pkl_path = out_dir / f"data_list_{idx}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(dl, fh)
        log.info(f"  Saved {len(dl)} graphs to {pkl_path}")

    log.info("\nDataset creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-exact dataset builder (08_create_datasets_paper.py)")
    parser.add_argument("--feat-dir",    type=Path, default=FEAT_DIR,
                        help="Root directory containing adjacency and node feature sub-dirs")
    parser.add_argument("--adj-subdir",  type=str,  default="Adjacency_Matrices/With_Self_Loop",
                        help="Sub-path under feat-dir for adjacency CSVs")
    parser.add_argument("--node-subdir", type=str,  default="Node_Feature_Vectors",
                        help="Sub-path under feat-dir for node feature CSVs")
    parser.add_argument("--out-dir",     type=Path, default=OUT_DIR,
                        help="Where to write data_list_*.pkl")
    args = parser.parse_args()

    main(args.feat_dir, args.adj_subdir, args.node_subdir, args.out_dir)
