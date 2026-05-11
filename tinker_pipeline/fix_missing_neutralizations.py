#!/usr/bin/env python3
"""
fix_missing_neutralizations.py

Re-processes the 35 proteins that were dropped from the neutralization
pipeline because they were absent from charge_info.csv.

Steps:
  1. Verify the 35 XYZ files exist in 4_Center_Moved_XYZ
  2. Append charge data for those 35 to charge_info.csv
  3. Regenerate System_Solve_Info_with_Charge.csv (full merge)
  4. Neutralize only those 35 proteins

Run from the Graph_pKa/ directory:
    python fix_missing_neutralizations.py
"""

import os
import sys
from pathlib import Path

# ── Make Tinker_EM importable ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from Tinker_EM import (
    analyze_and_collect_charge,
    process_solvent_info_and_ohh_ohh,
    offset_charges_in_systems,
)

MISSING_IDS = {
    "1AO6", "1B2V", "1BI6", "1CDC", "1DSB", "1EH6", "1I0E", "1I7K",
    "1JBB", "1KXI", "1NLX", "1ST9", "1SU9", "1YPH", "1YPI", "1YPT",
    "2CI2", "2FWF", "2H1A", "2HNP", "2L6X", "2OXP", "2PPT", "2RDF",
    "3C71", "3CYF", "3DMU", "3EZG", "3WU2", "4GE0", "4GE3", "4HHB",
    "4MA9", "4QYT", "7M2Z",
}

CENTER_MOVED_XYZ_DIR = "../Graph_pKa/Data/4_Center_Moved_XYZ"
CHARGE_INFO_CSV      = "../Graph_pKa/Data/4_Center_Moved_XYZ/charge_info.csv"
SOLVENT_INFO_CSV     = "../Graph_pKa/Data/5_Dissolved_Proteins/System_Solve_Info.csv"
MERGED_CSV           = "../Graph_pKa/Data/6_Neutralized_System/System_Solve_Info_with_Charge.csv"
PARAM_FILE           = "../Graph_pKa/Tinker_params/amoebabio18.prm"
DISSOLVED_DIR        = "../Graph_pKa/Data/5_Dissolved_Proteins"
NEUTRALIZED_DIR      = "../Graph_pKa/Data/6_Neutralized_System"
NEUTRAL_LOG          = "../Graph_pKa/Data/6_Neutralized_System/Failed/Neutralization_log_fix.txt"

# ── Step 1: Verify XYZ files exist ────────────────────────────────────────────
print("=" * 60)
print("Step 1: Checking XYZ files in 4_Center_Moved_XYZ")
missing_xyz = []
for pid in sorted(MISSING_IDS):
    xyz_path = os.path.join(CENTER_MOVED_XYZ_DIR, f"{pid}.xyz")
    if not os.path.exists(xyz_path):
        missing_xyz.append(pid)
        print(f"  MISSING XYZ: {pid}")
    else:
        print(f"  OK: {pid}.xyz")

if missing_xyz:
    print(f"\nWARNING: {len(missing_xyz)} XYZ files not found in 4_Center_Moved_XYZ.")
    print("These proteins cannot be processed:", missing_xyz)
    processable = MISSING_IDS - set(missing_xyz)
else:
    processable = MISSING_IDS
    print(f"\nAll {len(MISSING_IDS)} XYZ files found.")

if not processable:
    print("Nothing to process. Exiting.")
    sys.exit(1)

# ── Step 2: Append charge data ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: Running analyze on {len(processable)} proteins (appending to charge_info.csv)")
analyze_and_collect_charge(
    xyz_dir=CENTER_MOVED_XYZ_DIR,
    param_file=PARAM_FILE,
    only_ids=processable,
)

# Verify rows were added
import csv
added = []
with open(CHARGE_INFO_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["PDB_ID"].upper() in processable:
            added.append(row["PDB_ID"])
print(f"  Charge data now present for {len(added)}/{len(processable)} proteins: {sorted(added)}")

still_missing = processable - set(p.upper() for p in added)
if still_missing:
    print(f"  WARNING: analyze failed for: {sorted(still_missing)}")
    print("  Check ../Graph_pKa/Data/4_Center_Moved_XYZ/analyze_errors.log")

# ── Step 3: Regenerate merged CSV ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Regenerating System_Solve_Info_with_Charge.csv")
process_solvent_info_and_ohh_ohh(
    xyz_dir=CENTER_MOVED_XYZ_DIR,
    solvent_info_csv=SOLVENT_INFO_CSV,
    merged_output_csv=MERGED_CSV,
    merge_mode="extracted_pdb_id",
)

import pandas as pd
df = pd.read_csv(MERGED_CSV)
print(f"  Merged CSV now has {len(df)} rows (was 147, expected ~182)")
recovered = [f"{pid}.xyz" for pid in sorted(processable) if any(pid in str(fn) for fn in df["Filename"].values)]
print(f"  Recovered proteins in merged CSV: {len(recovered)}")

# ── Step 4: Neutralize the recovered proteins ─────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 4: Neutralizing {len(processable)} proteins")
results = offset_charges_in_systems(
    merged_system_solve_file=MERGED_CSV,
    soaked_proteins_dir=DISSOLVED_DIR,
    neutralized_system_dir=NEUTRALIZED_DIR,
    log_file_path=NEUTRAL_LOG,
    param_file=PARAM_FILE,
    only_ids=processable,
)

successes = [r for r in results if "Successfully" in r[1]]
failures  = [r for r in results if "Successfully" not in r[1]]
print(f"  Succeeded: {len(successes)}")
print(f"  Failed:    {len(failures)}")
if failures:
    print("  Failures:")
    for fname, reason in failures:
        print(f"    {fname}: {reason[:120]}")

# ── Final tally ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
neutralized = [f for f in os.listdir(NEUTRALIZED_DIR) if f.endswith(".xyz")]
print(f"Final count in 6_Neutralized_System: {len(neutralized)} .xyz files")
print("Done.")

