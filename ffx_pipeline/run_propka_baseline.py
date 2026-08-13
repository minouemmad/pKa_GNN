#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from propka.run import single

LONG_NAME = {"ASP": "Aspartate", "GLU": "Glutamate", "HIS": "Histidine",
             "LYS": "Lysine",    "CYS": "Cysteine",  "TYR": "Tyrosine"}

def predict_one(pdb_path: Path) -> list[dict]:
    """Run propka on a PDB; return list of (chain, resnum, resname3, pKa)."""
    mol = single(str(pdb_path), optargs=("--quiet",), write_pka=False)
    rows = []
    for grp in mol.conformations[mol.conformation_names[0]].groups:
        # Titratable, has a model_pka, residue type is one we care about
        if not grp.titratable:
            continue
        res3 = grp.atom.res_name.strip().upper()
        if res3 not in LONG_NAME:
            continue
        rows.append(dict(
            Chain_ID=grp.atom.chain_id,
            Residue_Number=int(grp.atom.res_num),
            Residue_Name=LONG_NAME[res3],
            Predicted_pKa=float(grp.pka_value),
        ))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", default="data/raw_pdbs")
    ap.add_argument("--gat-csv",
                    default="Graph_pKa/Results/Comparison_2026/_best_titrate_full_r10_tuned/predictions/dataset_3_all_folds.csv")
    ap.add_argument("--out",
                    default="Graph_pKa/Results/Comparison_2026/_baselines/propka3_predictions.csv")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    need = sorted(pd.read_csv(args.gat_csv)["PDB_ID"].astype(str).str.upper().unique())
    print(f"Running PROPKA3 on {len(need)} PDBs from {pdb_dir}")

    all_rows = []
    fails = []
    for i, pdb_id in enumerate(need, 1):
        candidates = [pdb_dir / f"{pdb_id}.pdb",
                      pdb_dir / f"{pdb_id.lower()}.pdb"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            print(f"  [{i:3d}/{len(need)}] {pdb_id}: PDB not found")
            fails.append((pdb_id, "no PDB"))
            continue
        try:
            rows = predict_one(path)
        except Exception as e:
            print(f"  [{i:3d}/{len(need)}] {pdb_id}: PROPKA ERROR {e}")
            fails.append((pdb_id, str(e)))
            continue
        for r in rows:
            r["PDB_ID"] = pdb_id
        all_rows.extend(rows)
        print(f"  [{i:3d}/{len(need)}] {pdb_id}: {len(rows)} titratable residues")

    df = pd.DataFrame(all_rows)[["PDB_ID", "Chain_ID", "Residue_Number",
                                  "Residue_Name", "Predicted_pKa"]]
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(df)} rows, from {df['PDB_ID'].nunique()} PDBs)")
    if fails:
        print(f"Failures: {len(fails)}")
        for pid, msg in fails:
            print(f"  {pid}: {msg}")

if __name__ == "__main__":
    main()
