#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")  # silence pKAI's autograd warning

from pKAI.pKAI import pKAI  # noqa: E402

LONG_NAME = {"ASP": "Aspartate", "GLU": "Glutamate", "HIS": "Histidine",
             "LYS": "Lysine",    "CYS": "Cysteine",  "TYR": "Tyrosine"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="pKAI",
                    help="pKAI or pKAI+ (note: pass the literal '+' character)")
    ap.add_argument("--pdb-dir", default="data/raw_pdbs")
    ap.add_argument("--gat-csv",
                    default="Graph_pKa/Results/Comparison_2026/_best_titrate_full_r10_tuned/predictions/dataset_3_all_folds.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(
        f"Graph_pKa/Results/Comparison_2026/_baselines/{args.model_name.replace('+','plus')}_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_dir = Path(args.pdb_dir)
    need = sorted(pd.read_csv(args.gat_csv)["PDB_ID"].astype(str).str.upper().unique())
    print(f"Running {args.model_name} on {len(need)} PDBs from {pdb_dir}")

    all_rows = []
    fails = []
    for i, pdb_id in enumerate(need, 1):
        cands = [pdb_dir / f"{pdb_id}.pdb", pdb_dir / f"{pdb_id.lower()}.pdb"]
        path = next((p for p in cands if p.exists()), None)
        if path is None:
            print(f"  [{i:3d}/{len(need)}] {pdb_id}: PDB not found")
            fails.append(pdb_id)
            continue
        try:
            r = pKAI(str(path), model_name=args.model_name, device="cpu")
        except Exception as e:
            print(f"  [{i:3d}/{len(need)}] {pdb_id}: ERROR {e}")
            fails.append(pdb_id)
            continue
        kept = 0
        for chain, resnum, res3, pka in r:
            res3 = res3.strip().upper()
            if res3 not in LONG_NAME:
                continue
            all_rows.append(dict(
                PDB_ID=pdb_id, Chain_ID=chain,
                Residue_Number=int(resnum),
                Residue_Name=LONG_NAME[res3],
                Predicted_pKa=float(pka),
            ))
            kept += 1
        print(f"  [{i:3d}/{len(need)}] {pdb_id}: {kept} titratable residues")

    df = pd.DataFrame(all_rows)[["PDB_ID", "Chain_ID", "Residue_Number",
                                  "Residue_Name", "Predicted_pKa"]]
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(df)} rows, from {df['PDB_ID'].nunique()} PDBs)")
    if fails:
        print(f"Failures: {fails}")

if __name__ == "__main__":
    main()
