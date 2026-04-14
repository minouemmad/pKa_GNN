"""
02_fix_structures.py
"""

import os
import csv
import traceback
from pathlib import Path

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
except ImportError:
    raise SystemExit(
        "pdbfixer / openmm not found.\n"
        "Install with:  conda install -c conda-forge pdbfixer openmm"
    )

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR   = "data/raw_pdbs"
FIXED_DIR = "data/fixed_pdbs"
LOG_PATH  = "data/fix_log.csv"

ADD_H_PH  = 7.0

# Structures too broken to use — skip entirely
EXCLUDE = set()  # 3WU2, 7M2Z, 1XSN re-enabled: all are in PKAD-R
# ─────────────────────────────────────────────────────────────────────────────

LOG_FIELDS = [
    "pdb_id", "status", "multi_model_stripped",
    "missing_residues_count", "missing_atoms_count",
    "nonstandard_replaced", "large_missing_residues", "notes"
]


# ── NMR multi-model stripper ──────────────────────────────────────────────────

def has_model_records(pdb_path: str) -> bool:
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                return True
    return False


def strip_to_model1(pdb_path: str, out_path: str) -> int:
    lines_out = []
    in_first_model = False
    first_model_done = False
    model_count = 0
    atom_count = 0

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()

            if rec == "MODEL":
                model_count += 1
                if model_count == 1:
                    in_first_model = True
                    continue
                else:
                    break

            if rec == "ENDMDL":
                if in_first_model:
                    in_first_model = False
                    first_model_done = True
                continue

            if model_count == 0:
                lines_out.append(line)
                if rec in ("ATOM", "HETATM"):
                    atom_count += 1
                continue

            if in_first_model:
                lines_out.append(line)
                if rec in ("ATOM", "HETATM"):
                    atom_count += 1

            if first_model_done and rec in ("CONECT", "END", "TER"):
                lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)

    return atom_count


def strip_hydrogens(pdb_path: str, out_path: str) -> int:
    """Remove all H/D atoms so FFX assigns its own via AMOEBA biotypes.

    Returns the number of heavy atoms written. Skipping hydrogens prevents
    FFX PolymerUtils from failing to assign atom types to backbone H atoms
    that PDBFixer places with non-AMOEBA-compatible names (e.g. 'H' on MET-1
    instead of 'H1'/'H2'/'H3').
    """
    heavy_count = 0
    lines_out = []

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                element   = line[76:78].strip() if len(line) > 76 else ""
                # Drop if name starts with H/D or element column says H/D
                if atom_name.startswith(("H", "D")) or element in ("H", "D"):
                    continue
                heavy_count += 1
            lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)

    return heavy_count


def fix_one(pdb_id: str, raw_path: str, fixed_path: str) -> dict:
    log = {k: "" for k in LOG_FIELDS}
    log["pdb_id"] = pdb_id
    stripped_tmp  = None
    noh_tmp       = None

    try:
        input_path = raw_path

        # ── Strip to model 1 for NMR ensembles ───────────────────────────────
        if has_model_records(raw_path):
            stripped_tmp = raw_path.replace(".pdb", "_model1_tmp.pdb")
            n_atoms = strip_to_model1(raw_path, stripped_tmp)
            input_path = stripped_tmp
            log["multi_model_stripped"] = f"yes ({n_atoms} ATOM/HETATM lines)"

            if n_atoms == 0:
                log["status"] = "FAILED"
                log["notes"] = "MODEL strip produced 0 atoms — malformed PDB"
                return log

        # ── Strip all hydrogens before PDBFixer ───────────────────────────────
        # FFX/AMOEBA rebuilds H positions itself; PDBFixer-placed H atoms often
        # carry PDB-convention names (e.g. "H" on N-terminal residues) that do
        # not match AMOEBA biotypes, causing "could not be assigned atom type".
        noh_tmp = input_path.replace(".pdb", "_noh_tmp.pdb")
        n_heavy = strip_hydrogens(input_path, noh_tmp)
        if n_heavy == 0:
            log["status"] = "FAILED"
            log["notes"] = "H-strip produced 0 heavy atoms — malformed PDB"
            return log
        input_path = noh_tmp

        # ── PDBFixer ──────────────────────────────────────────────────────────
        fixer = PDBFixer(filename=input_path)

        if fixer.topology is None:
            log["status"] = "FAILED"
            log["notes"] = "PDBFixer returned None topology"
            return log

        fixer.removeHeterogens(keepWater=False)

        fixer.findNonstandardResidues()
        ns = fixer.nonstandardResidues
        if ns:
            log["nonstandard_replaced"] = "; ".join(
                f"{r.name}{r.id}" for r, _ in ns
            )
        fixer.replaceNonstandardResidues()

        fixer.findMissingResidues()
        n_missing_res = sum(len(v) for v in fixer.missingResidues.values())
        log["missing_residues_count"] = n_missing_res

        large_gaps = [
            f"chain {k[0]} pos {k[1]} ({len(v)} residues)"
            for k, v in fixer.missingResidues.items()
            if len(v) > 5
        ]
        log["large_missing_residues"] = "; ".join(large_gaps)

        fixer.findMissingAtoms()
        n_missing_atoms = (
            sum(len(v) for v in fixer.missingAtoms.values()) +
            len(fixer.missingTerminals)
        )
        log["missing_atoms_count"] = n_missing_atoms
        fixer.addMissingAtoms()

        # Do NOT call addMissingHydrogens — FFX handles this via AMOEBA biotypes
        # fixer.addMissingHydrogens(ADD_H_PH)

        # Write PDBFixer output to a temp file, then strip any H atoms it added
        # via missingTerminals (e.g. N-terminal H named 'H' instead of H1/H2/H3)
        # and duplicate geminal H atoms placed at 0 A distance.  FFX rebuilds
        # all H positions itself via AMOEBA biotypes.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix="_pdbfixer.pdb",
                                        delete=False) as tmp:
            tmp_fixed_path = tmp.name
            PDBFile.writeFile(fixer.topology, fixer.positions, tmp)

        n_heavy_final = strip_hydrogens(tmp_fixed_path, fixed_path)
        os.remove(tmp_fixed_path)

        if n_heavy_final == 0:
            log["status"] = "FAILED"
            log["notes"] = "Post-PDBFixer H-strip produced 0 heavy atoms"
            return log

        log["status"] = "ok"

    except Exception:
        log["status"] = "FAILED"
        log["notes"] = traceback.format_exc().splitlines()[-1]

    finally:
        for tmp in (stripped_tmp, noh_tmp):
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    return log


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix PDB structures for FFX/AMOEBA.")
    parser.add_argument(
        "--file", "-f",
        metavar="PDB_PATH",
        help="Fix a single PDB file instead of the entire raw directory.",
    )
    args = parser.parse_args()

    os.makedirs(FIXED_DIR, exist_ok=True)

    if args.file:
        single = Path(args.file)
        if not single.exists():
            raise SystemExit(f"File not found: {single}")
        if single.suffix.lower() != ".pdb":
            raise SystemExit(f"Expected a .pdb file, got: {single}")
        raw_pdbs = [single]
        print(f"Single-file mode: {single}")
    else:
        raw_pdbs = sorted(Path(RAW_DIR).glob("*.pdb"))
        if not raw_pdbs:
            raise SystemExit(f"No PDB files found in {RAW_DIR}.")
        raw_pdbs = [p for p in raw_pdbs if p.stem.upper() not in EXCLUDE]
        print(f"Fixing {len(raw_pdbs)} structures  (hard-excluded: {sorted(EXCLUDE)})")

    with open(LOG_PATH, "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for raw_path in raw_pdbs:
            pdb_id = raw_path.stem.upper()
            # Strip any _fixed suffix if re-passing an already-named file
            if pdb_id.endswith("_FIXED"):
                pdb_id = pdb_id[:-6]
            fixed_path = os.path.join(FIXED_DIR, f"{pdb_id}_fixed.pdb")

            organized_path = os.path.join(FIXED_DIR, pdb_id, f"{pdb_id}_input.pdb")
            if not args.file and (os.path.exists(fixed_path) or os.path.exists(organized_path)):
                print(f"  [skip] {pdb_id} already fixed")
                continue

            log = fix_one(pdb_id, str(raw_path), fixed_path)
            writer.writerow(log)
            logf.flush()

            tag = " [NMR→model1]" if log["multi_model_stripped"] else ""
            sym = "✓" if log["status"] == "ok" else "✗"
            print(
                f"  [{sym}] {pdb_id:6s}{tag}"
                f"  miss_res={str(log['missing_residues_count']):>3}"
                f"  miss_atoms={str(log['missing_atoms_count']):>3}"
                f"  nonstd={log['nonstandard_replaced'] or '-'}"
                f"  {log['notes'] or ''}"
            )

    print(f"\nLog → {LOG_PATH}")
    print("Next step: generate_jobs.py")


if __name__ == "__main__":
    main()
