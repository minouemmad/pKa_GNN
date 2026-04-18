"""
tinker_prep_all.py

Helper called by tinker_prep.job — runs ALL preprocessing steps of the
paper's Tinker pipeline (Tinker_EM.py steps 1-11) for every protein in
data/fixed_pdbs/, but SKIPS the final minimize step (handled per-protein
by tinker_minimize_one.py).

Run from pKa_GNN/ as CWD:
    python tinker_pipeline/tinker_prep_all.py

Steps executed (with corrected paths for CWD = pKa_GNN/):
    NOTE: PDBFixer and water removal (paper steps 1-2) are handled upstream
    by tinker_pipeline/00_fix_structures.py.  This script picks up from
    the already-fixed PDBs in tinker_pipeline/data/fixed_pdbs/.

    0.  copy fixed PDBs                  → Graph_pKa/Data/2_Cleaned_PDB/
    1.  convert_pdb_to_xyz_files         → Graph_pKa/Data/2_Cleaned_PDB/*.xyz
    2.  move_center_of_mass_and_relocate → Graph_pKa/Data/4_Center_Moved_XYZ/
    3.  write_xyz_coordinate_ranges      → Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv
    4.  soak_proteins_with_waterbox      → Graph_pKa/Data/5_Dissolved_Proteins/
    5.  analyze_and_collect_charge       → Graph_pKa/Data/4_Center_Moved_XYZ/charge_info.csv
    6.  generate_system_solvent_info     → Graph_pKa/Data/5_Dissolved_Proteins/System_Solve_Info.csv
    7.  process_solvent_info_and_ohh_ohh
    8.  offset_charges_in_systems        → Graph_pKa/Data/6_Neutralized_System/
    9.  filter_copy_and_iterative_neutralization

When done, Graph_pKa/Data/6_Neutralized_System/ will contain
{PDB_ID}.xyz for each protein, ready for the per-protein minimize jobs.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ── Tinker binaries (pdbxyz, xyzedit, analyze, minimize) ─────────────────────
_TINKER_BIN = "/Dedicated/schnieders/programs/tinker-tools/bin"
os.environ["PATH"] = _TINKER_BIN + ":" + os.environ.get("PATH", "")

# ── Import Tinker_EM functions ────────────────────────────────────────────────
# Tinker_EM.py lives in Graph_pKa/ and its default paths assume CWD = Graph_pKa/.
# We import it and call each function with explicit pKa_GNN-relative paths so
# the script can be run from pKa_GNN/ as CWD.
# Anchor to pKa_GNN/ regardless of invocation CWD
PKA_GNN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKA_GNN_DIR / "Graph_pKa"))
from Tinker_EM import (  # type: ignore  # noqa: E402
    convert_pdb_to_xyz_files,
    move_center_of_mass_and_relocate,
    write_xyz_coordinate_ranges,
    soak_proteins_with_waterbox,
    analyze_and_collect_charge,
    generate_system_solvent_info,
    process_solvent_info_and_ohh_ohh,
    offset_charges_in_systems,
    filter_copy_and_iterative_neutralization,
)

# ── Paths (all anchored to pKa_GNN/ via __file__) ───────────────────────────
INPUT_PDB_DIR    = str(PKA_GNN_DIR / "tinker_pipeline/data/fixed_pdbs")
CLEANED_DIR      = str(PKA_GNN_DIR / "Graph_pKa/Data/2_Cleaned_PDB")
XYZ_DIR          = str(PKA_GNN_DIR / "Graph_pKa/Data/3_PDB_XYZ")
CENTER_XYZ_DIR   = str(PKA_GNN_DIR / "Graph_pKa/Data/4_Center_Moved_XYZ")
DISSOLVED_DIR    = str(PKA_GNN_DIR / "Graph_pKa/Data/5_Dissolved_Proteins")
NEUTRAL_DIR      = str(PKA_GNN_DIR / "Graph_pKa/Data/6_Neutralized_System")
PARAM_FILE       = str(PKA_GNN_DIR / "Graph_pKa/Tinker_params/amoebabio18.prm")
WATERBOX_DIR     = str(PKA_GNN_DIR / "Graph_pKa/Tinker_params/waterbox")
COORDS_CSV       = str(PKA_GNN_DIR / "Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv")
SOLVENT_INFO_CSV = str(PKA_GNN_DIR / "Graph_pKa/Data/5_Dissolved_Proteins/System_Solve_Info.csv")


def copy_input_pdbs() -> int:
    """Copy {PDB_ID}_input.pdb (already fixed by 00_fix_structures.py)
    from tinker_pipeline/data/fixed_pdbs/ → Graph_pKa/Data/2_Cleaned_PDB/{PDB_ID}.pdb.
    Skips files that already exist."""
    os.makedirs(CLEANED_DIR, exist_ok=True)
    copied = 0
    for pdb_path in sorted(Path(INPUT_PDB_DIR).glob("*/*_input.pdb")):
        pdb_id = pdb_path.parent.name
        dest   = Path(CLEANED_DIR) / f"{pdb_id}.pdb"
        if not dest.exists():
            shutil.copy(str(pdb_path), str(dest))
            copied += 1
    print(f"Copied {copied} PDB files to {CLEANED_DIR}")
    return copied


def main() -> None:
    print("=" * 60)
    print("Tinker preprocessing pipeline — all proteins")
    print(f"CWD: {os.getcwd()}")
    print("=" * 60)

    print("\n0. Copying fixed PDBs from 00_fix_structures output to 2_Cleaned_PDB/...")
    copy_input_pdbs()

    print("\n1. Converting PDB → XYZ with pdbxyz.x...")
    convert_pdb_to_xyz_files(
        pdb_dir=CLEANED_DIR,
        param_file=PARAM_FILE,
    )
    n_xyz = len(list(Path(CLEANED_DIR).glob("*.xyz")))
    print(f"   → {n_xyz} XYZ files in 2_Cleaned_PDB/")

    print("\n2. Centering structures with xyzedit.x...")
    move_center_of_mass_and_relocate(
        cleaned_pdb_dir=CLEANED_DIR,
        xyz_dir=XYZ_DIR,
        center_moved_xyz_dir=CENTER_XYZ_DIR,
        param_file=PARAM_FILE,   # absolute path; works regardless of subprocess CWD
    )
    n_centered = len(list(Path(CENTER_XYZ_DIR).glob("*.xyz")))
    print(f"   → {n_centered} XYZ files in 4_Center_Moved_XYZ/")

    print("\n3. Computing waterbox sizes...")
    write_xyz_coordinate_ranges(
        xyz_dir=CENTER_XYZ_DIR,
        output_csv=COORDS_CSV,
    )
    import csv as _csv
    try:
        with open(COORDS_CSV) as _f:
            n_csv = sum(1 for _ in _csv.DictReader(_f))
        print(f"   → {n_csv} rows in TinkerXYZ_coords.csv")
    except Exception:
        print("   → TinkerXYZ_coords.csv not found or empty")

    print("\n4. Soaking structures with waterbox...")
    soak_proteins_with_waterbox(
        tinker_coords_csv=COORDS_CSV,
        center_moved_xyz_dir=CENTER_XYZ_DIR,
        soaked_proteins_dir=DISSOLVED_DIR,
        param_file=PARAM_FILE,
        waterbox_dir=WATERBOX_DIR,
    )
    n_soaked = len(list(Path(DISSOLVED_DIR).glob("*.xyz")))
    print(f"   → {n_soaked} XYZ files in 5_Dissolved_Proteins/")

    print("\n5. Analyzing charges with analyze.x...")
    analyze_and_collect_charge(
        xyz_dir=CENTER_XYZ_DIR,
        param_file=PARAM_FILE,
    )

    print("\n6. Generating system solvent info...")
    generate_system_solvent_info(
        soaked_proteins_dir=DISSOLVED_DIR,
        solvent_info_csv=SOLVENT_INFO_CSV,
        source_xyz_dir=CENTER_XYZ_DIR,
    )

    MERGED_CSV = f"{NEUTRAL_DIR}/System_Solve_Info_with_Charge.csv"

    print("\n7. Processing solvent info and OHH-OHH patterns...")
    process_solvent_info_and_ohh_ohh(
        xyz_dir=CENTER_XYZ_DIR,
        solvent_info_csv=SOLVENT_INFO_CSV,
        merged_output_csv=MERGED_CSV,
    )

    print("\n8. Offsetting charges (neutralizing systems)...")
    offset_charges_in_systems(
        merged_system_solve_file=MERGED_CSV,
        soaked_proteins_dir=DISSOLVED_DIR,
        neutralized_system_dir=NEUTRAL_DIR,
        log_file_path=f"{NEUTRAL_DIR}/Failed/Neutralization_log.txt",
        param_file=PARAM_FILE,
    )
    n_neutral = len(list(Path(NEUTRAL_DIR).glob("*.xyz")))
    print(f"   → {n_neutral} XYZ files in 6_Neutralized_System/")

    print("\n9. Iterative neutralization for remaining failures...")
    filter_copy_and_iterative_neutralization(
        xyz_dir=NEUTRAL_DIR,
        output_csv=f"{NEUTRAL_DIR}/Failed/Failed_Neutralizations.csv",
        source_dir=DISSOLVED_DIR,
        work_dir=f"{NEUTRAL_DIR}/Failed",
        redo_dir=f"{NEUTRAL_DIR}/Redo",
        merged_csv_file=MERGED_CSV,
        log_file_path=f"{NEUTRAL_DIR}/Failed/Iterative_Neutralization_log.txt",
        param_file=PARAM_FILE,
    )

    n_ready = len(list(Path(NEUTRAL_DIR).glob("*.xyz")))
    print(f"\nDone.  {n_ready} proteins ready for minimization in {NEUTRAL_DIR}/")
    print("Submit per-protein minimize jobs next.")


if __name__ == "__main__":
    main()
