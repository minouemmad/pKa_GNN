
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TINKER_BIN = "/Dedicated/schnieders/programs/tinker-tools/bin"

# ── Intel OpenMP runtime (required by Tinker binaries on Argon) ──────────────
_INTEL_OMP_LIB = (
    "/opt/ssoft/apps/2021.1/linux-centos7-x86_64/gcc-4.8.5"
    "/intel-oneapi-compilers-2021.2.0-sapdob5"
    "/compiler/2021.2.0/linux/compiler/lib/intel64_lin"
)
os.environ["LD_LIBRARY_PATH"] = (
    _INTEL_OMP_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
# Prevent Tinker AMOEBA segfaults: OMP thread stacks overflow with default 8 MB limit.
# Set a large per-thread stack size; fall back to single-threaded if still crashing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_STACKSIZE", "512m")

# ── Paths (all relative to CWD = pKa_GNN/) ───────────────────────────────────
NEUTRAL_DIR  = "Graph_pKa/Data/6_Neutralized_System"
MIN_DIR      = "Graph_pKa/Data/7_Energy_Minimization_Systems"
COORDS_CSV   = "Graph_pKa/Data/4_Center_Moved_XYZ/TinkerXYZ_coords.csv"
PARAM_FILE   = "Graph_pKa/Tinker_params/amoebabio18.prm"   # absolute path passed to keyfile
# ─────────────────────────────────────────────────────────────────────────────

# RMS gradient convergence criterion — matches the paper's minimize.x call
RMS_GRADIENT = "1"

def read_waterbox(pdb_id: str, coords_csv: str) -> int:
    """Return the waterbox size for *pdb_id* from TinkerXYZ_coords.csv."""
    with open(coords_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_id = row["PDB_ID"].strip().lstrip("'")
            if row_id == pdb_id:
                return int(float(row["Waterbox"]))
    raise KeyError(f"PDB_ID '{pdb_id}' not found in {coords_csv}")

def write_key_file(key_path: str, param_file_abs: str, waterbox: int) -> None:
    """Write a Tinker .key file that exactly matches the paper's format."""
    content = (
        f"parameters              {param_file_abs}\n"
        "verbose\n\n"
        f"a-axis                      {waterbox}\n"
        f"b-axis                      {waterbox}\n"
        f"c-axis                      {waterbox}\n"
        "neighbor-list\n"
        "vdw-cutoff                     12.0\n"
        "ewald\n"
        "ewald-cutoff                    7.0\n"
        "integrator                    RESPA\n\n"
        "polar-eps                   0.00001\n"
        "polar-predict\n"
        "save-induced\n"
    )
    with open(key_path, "w") as fh:
        fh.write(content)

def run_minimize(xyz_path: str, key_path: str, work_dir: str) -> int:
    """Call minimize and return the process return code."""
    result = subprocess.run(
        [os.path.join(_TINKER_BIN, "minimize"), xyz_path, "-k", key_path],
        input=RMS_GRADIENT + "\n",
        capture_output=False,      # let stdout/stderr flow to job log
        text=True,
        cwd=work_dir,
    )
    return result.returncode

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Tinker minimize.x for one protein (paper pipeline)."
    )
    parser.add_argument("--pdb-id",     required=True, help="PDB ID to minimise")
    parser.add_argument("--neutral-dir", default=NEUTRAL_DIR)
    parser.add_argument("--min-dir",     default=MIN_DIR)
    parser.add_argument("--coords-csv",  default=COORDS_CSV)
    parser.add_argument("--param-file",  default=PARAM_FILE)
    args = parser.parse_args()

    pdb_id  = args.pdb_id
    src_xyz = Path(args.neutral_dir) / f"{pdb_id}.xyz"

    if not src_xyz.exists():
        print(f"ERROR: neutralized XYZ not found: {src_xyz}", file=sys.stderr)
        print("       Has tinker_prep.job finished successfully?", file=sys.stderr)
        sys.exit(1)

    if not Path(args.coords_csv).exists():
        print(f"ERROR: coords CSV not found: {args.coords_csv}", file=sys.stderr)
        sys.exit(1)

    # Read waterbox size determined during prep
    waterbox = read_waterbox(pdb_id, args.coords_csv)
    print(f"[{pdb_id}] waterbox = {waterbox} Å")

    # Create per-protein output directory
    work_dir = Path(args.min_dir) / pdb_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy neutralized XYZ into the working directory
    dest_xyz = work_dir / f"{pdb_id}.xyz"
    shutil.copy(str(src_xyz), str(dest_xyz))

    # Resolve all paths to absolute so Tinker can find them regardless of cwd
    work_dir_abs = work_dir.resolve()
    dest_xyz_abs = dest_xyz.resolve()
    param_abs    = str(Path(args.param_file).resolve())
    key_path     = str(work_dir_abs / f"{pdb_id}.key")
    write_key_file(key_path, param_abs, waterbox)

    print(f"[{pdb_id}] Running minimize.x  (RMS gradient {RMS_GRADIENT} kcal/mol/Å)...")
    rc = run_minimize(str(dest_xyz_abs), key_path, str(work_dir_abs))

    if rc != 0:
        print(f"[{pdb_id}] ERROR: minimize.x exited with code {rc}", file=sys.stderr)
        sys.exit(rc)

    minimized = work_dir_abs / f"{pdb_id}.xyz_2"
    if minimized.exists():
        print(f"[{pdb_id}] Minimization complete → {minimized}")
    else:
        print(f"[{pdb_id}] WARNING: expected output {minimized} not found — check Tinker logs")

if __name__ == "__main__":
    main()
