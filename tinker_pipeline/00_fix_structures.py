
import argparse
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
PIPELINE_ROOT = Path(__file__).resolve().parent   # tinker_pipeline/
REPO_ROOT     = PIPELINE_ROOT.parent              # pKa_GNN/
RAW_DIR   = str(REPO_ROOT     / "data/raw_pdbs")
FIXED_DIR = str(PIPELINE_ROOT / "data/fixed_pdbs")
LOG_PATH  = str(PIPELINE_ROOT / "data/fix_log.csv")

# Structures too broken to use — add PDB IDs here to skip them
# e.g. EXCLUDE = {"3WU2", "7M2Z", "1XSN"}
EXCLUDE = set()  # type: ignore[var-annotated]  # set[str], Python < 3.9 compat
# ─────────────────────────────────────────────────────────────────────────────

# ── Residue pre-processing tables ────────────────────────────────────────────
# Residue names to rename to their nearest standard equivalent before PDBFixer.
# PDBFixer's replaceNonstandardResidues() misses these.
_RESIDUE_REMAP = {
    "CSR": "CYS",   # cysteinesulfenic acid → cysteine (same backbone)
    "M3L": "LYS",   # N6,N6,N6-trimethyllysine → lysine
    "HSK": "SER",   # homoserine adduct → serine (closest backbone match)
    "MSE": "MET",   # selenomethionine (backup; PDBFixer usually handles this)
    "HSD": "HIS",   # CHARMM protonation variant
    "HSE": "HIS",
    "HSP": "HIS",
}

# Residue names to remove entirely — PDBFixer has no template and we can't
# meaningfully remap them.  NH2 is a C-terminal amide cap (not an amino acid).
# The DNA/RNA bases come from protein-nucleic acid co-crystal structures.
_RESIDUE_STRIP = {
    "NH2",                          # C-terminal amide cap
    "DA", "DC", "DG", "DT",         # DNA
    "DA3", "DC3", "DG3", "DT3",     # DNA 3′ terminal
    "DA5", "DC5", "DG5", "DT5",     # DNA 5′ terminal
    "2DT",                          # modified thymidine
    "A", "C", "G", "U",             # RNA
    "ADE", "CYT", "GUA", "THY",     # full-name variants
}
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

def preprocess_residues(pdb_path, out_path):
    """Remap fixable non-standard residues and strip unfixable ones.

    Also:
    - Strips _RESIDUE_STRIP names from SEQRES header lines so PDBFixer's
      findMissingResidues() never sees them (avoids KeyError on NH2 etc.).
    - Identifies chains where ALL coordinate records would be stripped and
      removes those chains entirely (avoids PDBFixer NoneType crash on empty
      chains, e.g. DNA-only chains in nucleoprotein structures like 1XSN).

    Returns (remapped_list, stripped_list) for logging.
    Must be called AFTER strip_hydrogens so atom counts are meaningful.
    """
    # ── Pass 1: identify chains that will be entirely stripped ───────────────
    # Also collect chains mentioned only in SEQRES (no ATOM/HETATM at all) —
    # PDBFixer creates empty chain objects for these which triggers the
    # '_current_chain' AttributeError in findMissingResidues().
    chain_keep  = set()   # chains with at least one non-stripped ATOM/HETATM
    chain_strip = set()   # chains where every ATOM/HETATM is stripped
    chain_seqres = set()  # chains mentioned in SEQRES records
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "SEQRES" and len(line) > 11:
                chain_seqres.add(line[11])
            if rec in ("ATOM", "HETATM") and len(line) > 21:
                resname = line[17:20].strip()
                chain   = line[21]
                if resname in _RESIDUE_STRIP:
                    chain_strip.add(chain)
                else:
                    chain_keep.add(chain)
    # A chain is truly empty if it has zero kept atoms, OR exists only in
    # SEQRES with no coordinate records at all (PDBFixer creates empty chain
    # objects for SEQRES-only chains, triggering _current_chain AttributeError).
    empty_chains = (chain_strip - chain_keep) | (chain_seqres - chain_keep - chain_strip)

    # ── Pass 2: write filtered file ──────────────────────────────────────────
    remapped  = []
    stripped  = []
    lines_out = []

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()

            # ── SEQRES: filter out stripped residue names token-by-token
            if rec == "SEQRES" and len(line) > 11:
                seqres_chain = line[11]
                if seqres_chain in empty_chains:
                    continue   # drop entire SEQRES line for empty chain
                prefix   = line[:19]     # everything up to first residue position
                tokens   = line[19:].split()
                filtered = [t for t in tokens if t not in _RESIDUE_STRIP]
                if not filtered:
                    continue   # whole line was stripped residues
                # Reconstruct: each residue name is 4 chars wide (name + space)
                lines_out.append(prefix + "".join("%-4s" % t for t in filtered).rstrip() + "\n")
                continue

            # ── Coordinate and annotation records for empty chains
            if rec in ("ATOM", "HETATM", "TER", "ANISOU") and len(line) > 21:
                chain = line[21]
                if chain in empty_chains:
                    continue

            if rec in ("ATOM", "HETATM"):
                resname = line[17:20].strip()
                if resname in _RESIDUE_STRIP:
                    stripped.append(resname)
                    continue
                if resname in _RESIDUE_REMAP:
                    new_name = _RESIDUE_REMAP[resname]
                    remapped.append("%s->%s" % (resname, new_name))
                    line = line[:17] + "%-3s" % new_name + line[20:]

            lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)

    return sorted(set(remapped)), sorted(set(stripped))

def fix_one(pdb_id, raw_path, fixed_path):
    log = {k: "" for k in LOG_FIELDS}
    log["pdb_id"] = pdb_id
    stripped_tmp = None
    noh_tmp      = None
    prepro_tmp   = None

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
        # carry PDB-convention names that don't match AMOEBA biotypes.
        noh_tmp = input_path.replace(".pdb", "_noh_tmp.pdb")
        n_heavy = strip_hydrogens(input_path, noh_tmp)
        if n_heavy == 0:
            log["status"] = "FAILED"
            log["notes"] = "H-strip produced 0 heavy atoms — malformed PDB"
            return log
        input_path = noh_tmp

        # ── Remap / strip non-standard residues before PDBFixer ───────────────
        prepro_tmp = input_path.replace(".pdb", "_prepro_tmp.pdb")
        remapped, stripped = preprocess_residues(input_path, prepro_tmp)
        input_path = prepro_tmp
        if remapped:
            log["notes"] = (log["notes"] + " remapped:" + ",".join(remapped)).strip()
        if stripped:
            log["notes"] = (log["notes"] + " stripped:" + ",".join(sorted(set(stripped)))).strip()

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

        # Clear periodic box so no CRYST1 record is written
        fixer.topology.setPeriodicBoxVectors(None)

        with open(fixed_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        log["status"] = "ok"

    except Exception:
        log["status"] = "FAILED"
        log["notes"] = traceback.format_exc().splitlines()[-1]

    finally:
        for tmp in (stripped_tmp, noh_tmp, prepro_tmp):
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    return log

def main():
    parser = argparse.ArgumentParser(
        description="Fix raw PDB files for Tinker/AMOEBA minimization."
    )
    parser.add_argument(
        "--only", nargs="+", metavar="PDB_ID",
        help="Process only these PDB IDs (uppercase, e.g. --only 1AZP 1BNZ 1XSN)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the fixed output already exists."
    )
    args = parser.parse_args()
    only_ids = {p.upper() for p in args.only} if args.only else None

    os.makedirs(FIXED_DIR, exist_ok=True)
    raw_pdbs = sorted(
        p for p in Path(RAW_DIR).glob("*.pdb")
        if "_tmp" not in p.name.lower()          # exclude leftover temp files
    )

    if not raw_pdbs:
        raise SystemExit(f"No PDB files found in {RAW_DIR}.")

    raw_pdbs = [p for p in raw_pdbs if p.stem.upper() not in EXCLUDE]
    if only_ids:
        raw_pdbs = [p for p in raw_pdbs if p.stem.upper() in only_ids]
        missing = only_ids - {p.stem.upper() for p in raw_pdbs}
        if missing:
            print(f"WARNING: --only specified PDB IDs not found in {RAW_DIR}: {sorted(missing)}")
    print(f"Fixing {len(raw_pdbs)} structures  (hard-excluded: {sorted(EXCLUDE)})")

    with open(LOG_PATH, "a" if only_ids else "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if not only_ids:
            writer.writeheader()

        for raw_path in raw_pdbs:
            pdb_id = raw_path.stem.upper()
            pdb_out_dir = os.path.join(FIXED_DIR, pdb_id)
            fixed_path = os.path.join(pdb_out_dir, f"{pdb_id}_input.pdb")

            if os.path.exists(fixed_path) and not args.force:
                print(f"  [skip] {pdb_id} already fixed  (use --force to redo)")
                continue

            os.makedirs(pdb_out_dir, exist_ok=True)

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
    print("Next step: 01_run_tinker_minimize.py")

if __name__ == "__main__":
    main()

