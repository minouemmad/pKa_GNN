"""
tinker_prep_all.py

Helper called by tinker_prep.job — runs ALL preprocessing steps of the
paper's Tinker pipeline (Tinker_EM.py steps 1-11) for every protein in
data/fixed_pdbs/, but SKIPS the final minimize step (handled per-protein
by tinker_minimize_one.py).

Run from pKa_GNN/ as CWD:
    python tinker_pipeline/tinker_prep_all.py

Steps executed (with corrected paths for CWD = pKa_GNN/):
    1.  fix_pdb_files_with_pdbfixer      → Graph_pKa/Data/1_PDB_After_Fixer/
    2.  remove_water_with_mdanalysis     → Graph_pKa/Data/2_Cleaned_PDB/
    3.  convert_pdb_to_xyz_files         → Graph_pKa/Data/2_Cleaned_PDB/*.xyz
    4.  move_center_of_mass_and_relocate → Graph_pKa/Data/4_Center_Moved_XYZ/
    5.  write_xyz_coordinate_ranges      → Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv
    6.  soak_proteins_with_waterbox      → Graph_pKa/Data/5_Dissolved_Proteins/
    7.  analyze_and_collect_charge       → Graph_pKa/Data/4_Center_Moved_XYZ/charge_info.csv
    8.  generate_system_solvent_info     → Graph_pKa/Data/5_Dissolved_Proteins/System_Solve_Info.csv
    9.  process_solvent_info_and_ohh_ohh
    10. offset_charges_in_systems        → Graph_pKa/Data/6_Neutralized_System/
    11. filter_copy_and_iterative_neutralization

When done, Graph_pKa/Data/6_Neutralized_System/ will contain
{PDB_ID}.xyz for each protein, ready for the per-protein minimize jobs.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ── Import Tinker_EM functions ────────────────────────────────────────────────
# Tinker_EM.py lives in Graph_pKa/ and its default paths assume CWD = Graph_pKa/.
# We import it and call each function with explicit pKa_GNN-relative paths so
# the script can be run from pKa_GNN/ as CWD.
sys.path.insert(0, str(Path("Graph_pKa").resolve()))
from Tinker_EM import (  # type: ignore  # noqa: E402
    fix_pdb_files_with_pdbfixer,
    remove_water_with_mdanalysis,
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

# ── Paths (all relative to CWD = pKa_GNN/) ───────────────────────────────────
INPUT_PDB_GLOB   = "data/fixed_pdbs"
RAW_PDB_DIR      = "Graph_pKa/Data/0_Raw_PDB"
FIXED_DIR        = "Graph_pKa/Data/1_PDB_After_Fixer"
CLEANED_DIR      = "Graph_pKa/Data/2_Cleaned_PDB"
XYZ_DIR          = "Graph_pKa/Data/3_PDB_XYZ"
CENTER_XYZ_DIR   = "Graph_pKa/Data/4_Center_Moved_XYZ"
DISSOLVED_DIR    = "Graph_pKa/Data/5_Dissolved_Proteins"
NEUTRAL_DIR      = "Graph_pKa/Data/6_Neutralized_System"
PARAM_FILE       = "Graph_pKa/Tinker_params/amoebabio18.prm"
COORDS_CSV       = f"{CENTER_XYZ_DIR}/TinkerXYZ_coords.csv"
SOLVENT_INFO_CSV = f"{DISSOLVED_DIR}/System_Solve_Info.csv"


def copy_input_pdbs() -> int:
    """Copy {PDB_ID}_input.pdb → 0_Raw_PDB/{PDB_ID}.pdb for every protein."""
    os.makedirs(RAW_PDB_DIR, exist_ok=True)
    copied = 0
    for pdb_path in sorted(Path(INPUT_PDB_GLOB).glob("*/*_input.pdb")):
        pdb_id = pdb_path.parent.name
        dest   = Path(RAW_PDB_DIR) / f"{pdb_id}.pdb"
        if not dest.exists():
            shutil.copy(str(pdb_path), str(dest))
            copied += 1
    print(f"Copied {copied} PDB files to {RAW_PDB_DIR}")
    return copied


def main() -> None:
    print("=" * 60)
    print("Tinker preprocessing pipeline — all proteins")
    print(f"CWD: {os.getcwd()}")
    print("=" * 60)

    print("\n0. Copying input PDBs to 0_Raw_PDB/...")
    copy_input_pdbs()

    print("\n1. Fixing PDB files with PDBFixer...")
    fix_pdb_files_with_pdbfixer(
        pdb_dir=RAW_PDB_DIR,
        output_dir=FIXED_DIR,
    )

    print("\n2. Removing water with MDAnalysis...")
    remove_water_with_mdanalysis(
        pdb_dir=FIXED_DIR,
        output_dir=CLEANED_DIR,
        log_file_path=f"{CLEANED_DIR}/mdanalysis_output.log",
    )

    print("\n3. Converting PDB → XYZ with pdbxyz.x...")
    convert_pdb_to_xyz_files(
        pdb_dir=CLEANED_DIR,
        param_file=PARAM_FILE,
    )

    print("\n4. Centering structures with xyzedit.x...")
    move_center_of_mass_and_relocate(
        cleaned_pdb_dir=CLEANED_DIR,
        xyz_dir=XYZ_DIR,
        center_moved_xyz_dir=CENTER_XYZ_DIR,
        # param_file here is used inside the subprocess CWD = XYZ_DIR,
        # so ../../Tinker_params/... resolves to Graph_pKa/Tinker_params/...
        param_file="../../Tinker_params/amoebabio18.prm",
    )

    print("\n5. Computing waterbox sizes...")
    write_xyz_coordinate_ranges(
        xyz_dir=CENTER_XYZ_DIR,
        output_csv=COORDS_CSV,
    )

    print("\n6. Soaking structures with waterbox...")
    soak_proteins_with_waterbox(
        tinker_coords_csv=COORDS_CSV,
        center_moved_xyz_dir=CENTER_XYZ_DIR,
        soaked_proteins_dir=DISSOLVED_DIR,
        param_file=PARAM_FILE,
    )

    print("\n7. Analyzing charges with analyze.x...")
    analyze_and_collect_charge(
        xyz_dir=CENTER_XYZ_DIR,
        param_file=PARAM_FILE,
    )

    print("\n8. Generating system solvent info...")
    generate_system_solvent_info(
        soaked_proteins_dir=DISSOLVED_DIR,
        solvent_info_csv=SOLVENT_INFO_CSV,
        source_xyz_dir=CENTER_XYZ_DIR,
    )

    MERGED_CSV = f"{NEUTRAL_DIR}/System_Solve_Info_with_Charge.csv"

    print("\n9. Processing solvent info and OHH-OHH patterns...")
    process_solvent_info_and_ohh_ohh(
        xyz_dir=CENTER_XYZ_DIR,
        solvent_info_csv=SOLVENT_INFO_CSV,
        merged_output_csv=MERGED_CSV,
    )

    print("\n10. Offsetting charges (neutralizing systems)...")
    offset_charges_in_systems(
        merged_system_solve_file=MERGED_CSV,
        soaked_proteins_dir=DISSOLVED_DIR,
        neutralized_system_dir=NEUTRAL_DIR,
        log_file_path=f"{NEUTRAL_DIR}/Failed/Neutralization_log.txt",
    )

    print("\n11. Iterative neutralization for remaining failures...")
    filter_copy_and_iterative_neutralization(
        xyz_dir=NEUTRAL_DIR,
        output_csv=f"{NEUTRAL_DIR}/Failed/Failed_Neutralizations.csv",
        source_dir=DISSOLVED_DIR,
        work_dir=f"{NEUTRAL_DIR}/Failed",
        redo_dir=f"{NEUTRAL_DIR}/Redo",
        merged_csv_file=MERGED_CSV,
        log_file_path=f"{NEUTRAL_DIR}/Failed/Iterative_Neutralization_log.txt",
    )

    n_ready = len(list(Path(NEUTRAL_DIR).glob("*.xyz")))
    print(f"\nDone.  {n_ready} proteins ready for minimization in {NEUTRAL_DIR}/")
    print("Submit per-protein minimize jobs next.")


if __name__ == "__main__":
    main()
