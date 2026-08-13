from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Tokens that should not appear in protein heavy-atom records
NON_PROTEIN_TOKENS = {"Cl-", "Cl", "Na+", "Na", "K+", "K", "Mg+2", "Mg", "Ca+2", "Ca"}

def find_largest_xyz(pdb_dir: Path, pdb_id: str) -> Optional[Path]:
    candidates = []
    for p in pdb_dir.glob(f"{pdb_id}.xyz*"):
        m = re.match(rf"^{re.escape(pdb_id)}\.xyz(?:_(\d+))?$", p.name)
        if not m:
            continue
        idx = int(m.group(1)) if m.group(1) else 1
        candidates.append((idx, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]

def parse_tinker_xyz(xyz_path: Path) -> list[dict]:
    """Parse Tinker XYZ.  Returns list of {serial, name, x, y, z}.

    Stops at the first O–H–H–O–H–H pattern (water).  Also stops at any
    non-protein ion token (e.g. 'Cl-').
    """
    atoms: list[dict] = []
    raw_lines: list[list[str]] = []

    with open(xyz_path) as fh:
        for line in fh:
            parts = line.split()
            # First line is "<n_atoms>"; second is unit cell (no integer first part with len>=6)
            if not parts:
                continue
            if not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < 6:
                # likely the cell line "    70.0 70.0 ..."  → skip
                continue
            raw_lines.append(parts)

    n = len(raw_lines)
    stop = n
    for i, parts in enumerate(raw_lines):
        name = parts[1]
        if name in NON_PROTEIN_TOKENS:
            stop = i
            break
        # Detect O-H-H-O-H-H water pattern
        if name == "O" and i + 5 < n:
            names_ahead = [raw_lines[i + k][1] for k in range(1, 6)]
            if names_ahead == ["H", "H", "O", "H", "H"]:
                stop = i
                break

    for i in range(stop):
        parts = raw_lines[i]
        try:
            atoms.append({
                "serial": int(parts[0]),
                "name":   parts[1],
                "x":      float(parts[2]),
                "y":      float(parts[3]),
                "z":      float(parts[4]),
            })
        except (ValueError, IndexError):
            continue

    return atoms

def parse_input_pdb_lines(pdb_path: Path) -> list[tuple[int, str]]:
    """Return list of (line_index, atom_name) for ATOM records (heavy + H)."""
    out: list[tuple[int, str]] = []
    with open(pdb_path) as fh:
        for idx, line in enumerate(fh):
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                out.append((idx, name))
    return out

def is_hydrogen(name: str) -> bool:
    """Return True for hydrogen atoms by name convention."""
    if not name:
        return False
    s = name.strip()
    if s.startswith("H"):
        return True
    # numeric-prefixed Hs e.g. 1H, 2HA
    if len(s) >= 2 and s[0].isdigit() and s[1] == "H":
        return True
    return False

def write_final_pdb(
    input_pdb: Path,
    tinker_atoms: list[dict],
    out_pdb: Path,
) -> tuple[bool, str]:
    """Replace coordinates of heavy atoms in input_pdb with tinker_atoms.

    Tinker file may include H atoms; we strip those and align by sequence
    position to the input PDB ATOM records (which are all heavy in
    PDBFixer-prepared input).
    """
    # Filter Tinker heavy atoms (drop H)
    tinker_heavy = [a for a in tinker_atoms if not is_hydrogen(a["name"])]

    with open(input_pdb) as fh:
        pdb_lines = fh.readlines()

    pdb_atom_idx = []
    pdb_atom_names = []
    for idx, line in enumerate(pdb_lines):
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            if is_hydrogen(name):
                continue  # tolerate H if present
            pdb_atom_idx.append(idx)
            pdb_atom_names.append(name)

    if len(pdb_atom_idx) != len(tinker_heavy):
        return False, (
            f"heavy-atom count mismatch: "
            f"input PDB={len(pdb_atom_idx)}, tinker_heavy={len(tinker_heavy)}"
        )

    # Rewrite lines with new coordinates
    new_lines = list(pdb_lines)
    for line_idx, t_atom in zip(pdb_atom_idx, tinker_heavy):
        line = pdb_lines[line_idx]
        # PDB columns: 31-38 x, 39-46 y, 47-54 z (1-indexed) → 30:38, 38:46, 46:54 in 0-indexed
        new_coords = f"{t_atom['x']:8.3f}{t_atom['y']:8.3f}{t_atom['z']:8.3f}"
        new_line = line[:30] + new_coords + line[54:]
        new_lines[line_idx] = new_line

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as fh:
        fh.writelines(new_lines)

    return True, f"{len(tinker_heavy)} heavy atoms"

def process_pdb(
    pdb_id: str,
    tinker_root: Path,
    input_root: Path,
    fallback_input_root: Optional[Path],
) -> tuple[bool, str]:
    tinker_dir = tinker_root / pdb_id
    if not tinker_dir.is_dir():
        return False, "no tinker dir"

    xyz = find_largest_xyz(tinker_dir, pdb_id)
    if xyz is None:
        return False, "no .xyz file"

    uind = tinker_dir / f"{pdb_id}.uind"
    if not uind.exists():
        return False, "no .uind file"

    input_pdb = input_root / pdb_id / f"{pdb_id}_input.pdb"
    if not input_pdb.exists() and fallback_input_root is not None:
        input_pdb = fallback_input_root / pdb_id / f"{pdb_id}_input.pdb"
    if not input_pdb.exists():
        return False, f"no input PDB ({input_pdb})"

    out_dir = input_root / pdb_id
    out_pdb = out_dir / f"{pdb_id}_final.pdb"
    out_uind = out_dir / f"{pdb_id}_final.uind"

    tinker_atoms = parse_tinker_xyz(xyz)
    if not tinker_atoms:
        return False, f"empty parse of {xyz.name}"

    ok, msg = write_final_pdb(input_pdb, tinker_atoms, out_pdb)
    if not ok:
        return False, f"{msg} ({xyz.name})"

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(uind, out_uind)
    return True, f"{xyz.name} -> {msg}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinker-dir", required=True,
                    help="Graph_pKa/Data/7_Energy_Minimization_Systems")
    ap.add_argument("--input-root", required=True,
                    help="tinker_pipeline/data/fixed_pdbs (also output dir)")
    ap.add_argument("--fallback-input-root", default=None,
                    help="ffx_pipeline/data/fixed_pdbs")
    ap.add_argument("--pdb-list", default=None,
                    help="optional file with PDB IDs (one per line); if omitted, "
                         "process all subdirs of --tinker-dir")
    args = ap.parse_args()

    tinker_root = Path(args.tinker_dir)
    input_root  = Path(args.input_root)
    fallback    = Path(args.fallback_input_root) if args.fallback_input_root else None

    if args.pdb_list:
        with open(args.pdb_list) as fh:
            pdb_ids = [ln.strip() for ln in fh if ln.strip()]
    else:
        pdb_ids = sorted(p.name for p in tinker_root.iterdir() if p.is_dir())

    log.info(f"Processing {len(pdb_ids)} PDBs")
    succ, fail = [], []
    for pid in pdb_ids:
        ok, msg = process_pdb(pid, tinker_root, input_root, fallback)
        if ok:
            log.info(f"  OK   {pid}: {msg}")
            succ.append(pid)
        else:
            log.warning(f"  FAIL {pid}: {msg}")
            fail.append((pid, msg))

    log.info(f"Summary: {len(succ)} success, {len(fail)} fail")
    if fail:
        fail_log = input_root / "_xyz_to_pdb_failures.txt"
        with open(fail_log, "w") as fh:
            for pid, msg in fail:
                fh.write(f"{pid}\t{msg}\n")
        log.info(f"Failures: {fail_log}")

if __name__ == "__main__":
    main()
