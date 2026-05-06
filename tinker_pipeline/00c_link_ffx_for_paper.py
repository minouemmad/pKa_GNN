"""
00c_link_ffx_for_paper.py
=========================

Stage FFX rotopt outputs into the same `_final.pdb` / `_final.uind`
naming used by `tinker_pipeline/02_prepare_features.py`, so the same
paper-exact 26-feature extractor can be run on FFX data.

Source per PDB (FFX rotopt):
    ffx_pipeline/data/fixed_pdbs/<PDB>/<PDB>_rot.pdb
    ffx_pipeline/data/fixed_pdbs/<PDB>/<PDB>_input.uind

Destination per PDB (paper-style, hard copies):
    <out-root>/<PDB>/<PDB>_final.pdb
    <out-root>/<PDB>/<PDB>_final.uind
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffx-root", required=True,
                    help="ffx_pipeline/data/fixed_pdbs")
    ap.add_argument("--out-root", required=True,
                    help="destination (e.g. tinker_pipeline/data/fixed_pdbs_ffx)")
    ap.add_argument("--pdb-list", required=True,
                    help="file with PDB IDs, one per line")
    args = ap.parse_args()

    src_root = Path(args.ffx_root)
    out_root = Path(args.out_root)

    with open(args.pdb_list) as fh:
        pdb_ids = [ln.strip() for ln in fh if ln.strip()]

    succ, fail = 0, []
    for pid in pdb_ids:
        src_pdb   = src_root / pid / f"{pid}_rot.pdb"
        src_uind  = src_root / pid / f"{pid}_input.uind"
        src_uperm = src_root / pid / f"{pid}_input.uperm"
        if not src_pdb.exists() or not src_uind.exists():
            fail.append((pid, "missing src"))
            continue
        dst_dir = out_root / pid
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pdb,  dst_dir / f"{pid}_final.pdb")
        shutil.copy2(src_uind, dst_dir / f"{pid}_final.uind")
        if src_uperm.exists():
            shutil.copy2(src_uperm, dst_dir / f"{pid}_final.uperm")
        succ += 1

    log.info(f"Linked {succ}/{len(pdb_ids)} (failures={len(fail)})")
    for pid, msg in fail:
        log.warning(f"  FAIL {pid}: {msg}")


if __name__ == "__main__":
    main()
