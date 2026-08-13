#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Dataset index -> radius (must match 06_create_datasets.py)
INDEX_TO_RADIUS = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11}

# Re-declare the regex map locally so this script doesn't have to import a
# module whose name starts with a digit.  Must stay in sync with
# ABLATION_GROUPS in 07_train.py.
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

def reconstruct_feature_names(feat_dir: Path, radius: int, n_cols: int) -> list[str]:
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

def ablate_data_list(data_list, names: list[str], groups: list[str]) -> list[int]:
    zero_cols: list[int] = []
    for g in groups:
        if g not in ABLATION_GROUPS:
            raise ValueError(f"Unknown ablation group {g!r}.  "
                             f"Choose from {sorted(ABLATION_GROUPS)}")
        pat = re.compile(ABLATION_GROUPS[g])
        zero_cols.extend(i for i, n in enumerate(names) if pat.match(n))
    zero_cols = sorted(set(zero_cols))
    if not zero_cols:
        return []
    cols_t = torch.tensor(zero_cols, dtype=torch.long)
    for d in data_list:
        d.x[:, cols_t] = 0.0
    return zero_cols

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir",  type=Path, required=True,
                   help="Source datasets dir containing data_list_{0..4}.pkl")
    p.add_argument("--dst-dir",  type=Path, required=True,
                   help="Destination dir for the ablated data_list_*.pkl")
    p.add_argument("--feat-dir", type=Path, required=True,
                   help="Features root (has Node_Feature_Vectors/{radius}/)")
    p.add_argument("--ablate",   nargs="+", required=True,
                   help=f"Feature groups to zero.  Choose from "
                        f"{sorted(ABLATION_GROUPS)}")
    p.add_argument("--indices",  type=int, nargs="*", default=None,
                   help="Specific dataset indices (default: all 0..4 found)")
    args = p.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source  : {args.src_dir}")
    print(f"Dest    : {args.dst_dir}")
    print(f"Ablate  : {args.ablate}")

    indices = args.indices if args.indices else list(INDEX_TO_RADIUS)
    for idx in indices:
        src = args.src_dir / f"data_list_{idx}.pkl"
        if not src.exists():
            print(f"  [skip] {src.name} not found")
            continue
        radius = INDEX_TO_RADIUS[idx]
        with open(src, "rb") as fh:
            data_list = pickle.load(fh)
        if not data_list:
            print(f"  [skip] {src.name} empty")
            continue
        n_cols = data_list[0].x.shape[1]
        names  = reconstruct_feature_names(args.feat_dir, radius, n_cols)
        zeroed = ablate_data_list(data_list, names, args.ablate)
        zeroed_names = [names[i] for i in zeroed]
        dst = args.dst_dir / f"data_list_{idx}.pkl"
        with open(dst, "wb") as fh:
            pickle.dump(data_list, fh)
        print(f"  [ok]  idx={idx} (radius {radius} A): "
              f"{len(data_list)} graphs, zeroed {len(zeroed)} cols "
              f"-> {zeroed_names}")
        print(f"         wrote {dst}")

    print("Done.")

if __name__ == "__main__":
    main()
