from __future__ import annotations
import argparse, os
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="tinker_pipeline/data/fixed_pdbs")
    ap.add_argument("--dst", default="tinker_pipeline/data/fixed_pdbs_rotopt_schema")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for d in sorted(src.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        fin_pdb = d / f"{pid}_final.pdb"
        fin_uind = d / f"{pid}_final.uind"
        if not fin_pdb.exists() or not fin_uind.exists():
            n_skip += 1
            continue
        out_dir = dst / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdb = out_dir / f"{pid}_rot.pdb"
        out_uind = out_dir / f"{pid}_input.uind"
        # remove existing then hardlink
        for tgt, src_file in [(out_pdb, fin_pdb), (out_uind, fin_uind)]:
            if tgt.exists() or tgt.is_symlink():
                tgt.unlink()
            try:
                os.link(src_file, tgt)
            except OSError:
                # cross-device or other failure → fallback to copy
                import shutil
                shutil.copy2(src_file, tgt)
        n_ok += 1
    print(f"Staged {n_ok} PDBs to {dst}  (skipped {n_skip})")

if __name__ == "__main__":
    main()
