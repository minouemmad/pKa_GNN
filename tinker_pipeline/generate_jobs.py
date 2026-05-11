#!/usr/bin/env python3
"""Generate SGE job scripts for every unique protein in the PKAD CSV.

Each generated job script runs:
  Step 1 – Coarse minimization        (FFX Minimize, RMS=0.8, GK)
  Step 2 – Start parallel scheduler
  Step 3 – ManyBody rotamer optimisation
  Step 4 – Tight final minimization   (FFX Minimize, RMS=0.1, GPU, --saveInduced)
  Step 5 – Print permanent multipoles (PrintMultipoles.groovy)

Usage
-----
python generate_jobs.py \
    --csv        1-PKAD-R-2025-09-03.csv \
    --pdb_dir    /Dedicated/schnieders/maemmad/pKa_GNN/data/fixed_pdbs \
    --ffx        /Dedicated/schnieders/maemmad/forcefieldx/bin/ffxc \
    --out_dir    /Dedicated/schnieders/maemmad/pKa_GNN/data/sge_jobs \
    --queue      "MS,UI-GPU" \
    --ncpus      20

The script infers the .properties file path as:
    {pdb_dir}/{ID}.properties
and expects the fixed PDB at:
    {pdb_dir}/{ID}_fixed.pdb
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

DEFAULT_CSV    = Path(__file__).parent / "1-PKAD-R-2025-09-03.csv"
DEFAULT_PDB_DIR = "/Dedicated/schnieders/maemmad/pKa_GNN/data/fixed_pdbs"
DEFAULT_FFX    = "/Dedicated/schnieders/maemmad/forcefieldx/bin/ffxc"
DEFAULT_OUT    = "/Dedicated/schnieders/maemmad/pKa_GNN/data/sge_jobs"
DEFAULT_QUEUE  = "MS,UI-GPU"
DEFAULT_NCPUS  = 20


# ---------------------------------------------------------------------------
# Job template
# ---------------------------------------------------------------------------

JOB_TEMPLATE = """\
#!/bin/bash
#$ -V
#$ -cwd
#$ -N min_{ID}
#$ -j y
#$ -q {queue}
#$ -pe smp {ncpus}
#$ -o $JOB_NAME.$JOB_ID.log
#$ -l h_rt=10000:00:00
#$ -S /bin/bash
#$ -l ngpus=1

# ── Environment ───────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "PDB: {pdb}"

# ── Step 1: Coarse minimization with GK implicit solvent ──────────────────────
echo "=== Step 1: Coarse Minimize (RMS=0.8) ==="
{ffx} Minimize \\
    -e 0.8 \\
    {pdb} \\
    -Dkey={props}

if [ ! -f "{pdb}_2" ]; then
    echo "ERROR: Step 1 output not found: {pdb}_2"
    exit 1
fi

echo "=== Renaming {pdb}_2 → {min_pdb} ==="
mv "{pdb}_2" "{min_pdb}"

# ── Step 2: Start Parallel Java scheduler (background) ────────────────────────
echo "=== Step 2: Starting Scheduler ==="
{ffx} Scheduler -p 5 -m 22G > scheduler_{ID}.log &
SCHEDULER_PID=$!
sleep 30s

# ── Step 3: Many-body rotamer optimization ────────────────────────────────────
echo "=== Step 3: ManyBody Rotamer Optimization ==="
{ffx} ManyBody \\
    -Dpj.nn=4 \\
    -Dpj.nt=5 \\
    -DnumCudaDevices=1 \\
    {min_pdb} \\
    -Dkey={props}

kill $SCHEDULER_PID 2>/dev/null || true

if [ ! -f "{min_pdb}_2" ]; then
    echo "WARNING: Step 3 output not found; using Step 1 output for final minimize."
    pdb_s4_in="{min_pdb}"
else
    echo "=== Renaming {min_pdb}_2 → {min2_pdb} ==="
    mv "{min_pdb}_2" "{min2_pdb}"
    pdb_s4_in="{min2_pdb}"
fi

# ── Step 4: Tight final minimization on GPU + write induced dipoles ────────────
echo "=== Step 4: Final Minimize (RMS=0.1, GPU, saveInduced) ==="
{ffx} Minimize \\
    -e 0.1 \\
    "${{pdb_s4_in}}" \\
    -Dkey={props} \\
    --saveInduced

# Determine final PDB path (FFX appends _2)
final_pdb="${{pdb_s4_in}}_2"
if [ ! -f "$final_pdb" ]; then
    echo "ERROR: Step 4 output not found: $final_pdb"
    exit 1
fi

# ── Step 5: Print permanent AMOEBA multipoles ─────────────────────────────────
echo "=== Step 5: PrintMultipoles ==="
{ffx} PrintMultipoles \\
    "$final_pdb" \\
    -Dkey={props}

echo "=== Done: $(date) ==="
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unique_pdb_ids(csv_path: Path) -> list[str]:
    """Return sorted unique uppercase PDB IDs from the PKAD CSV."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept 'pdb' or 'pdb_id'
    col = next((c for c in df.columns if c in ("pdb", "pdb_id")), None)
    if col is None:
        sys.exit(f"ERROR: Cannot find PDB column in {csv_path}")
    return sorted(df[col].str.upper().dropna().unique().tolist())


def write_job(
    pid: str,
    pdb_dir: str,
    ffx: str,
    out_dir: Path,
    queue: str,
    ncpus: int,
) -> Path:
    pdb      = f"{pdb_dir}/{pid}_fixed.pdb"
    props    = f"{pdb_dir}/{pid}.properties"
    min_pdb  = f"{pdb_dir}/{pid}_fixed_min.pdb"
    min2_pdb = f"{pdb_dir}/{pid}_fixed_min_2.pdb"

    body = JOB_TEMPLATE.format(
        ID=pid,
        pdb=pdb,
        props=props,
        min_pdb=min_pdb,
        min2_pdb=min2_pdb,
        ffx=ffx,
        queue=queue,
        ncpus=ncpus,
    )

    job_path = out_dir / f"{pid}_minimize.job"
    job_path.write_text(body)
    return job_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SGE job scripts for all PKAD proteins (Steps 1–5 including PrintMultipoles)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",     default=str(DEFAULT_CSV),     help="PKAD CSV file")
    parser.add_argument("--pdb_dir", default=DEFAULT_PDB_DIR,      help="Directory containing {ID}_fixed.pdb files on the cluster")
    parser.add_argument("--ffx",     default=DEFAULT_FFX,          help="Path to the ffxc executable on the cluster")
    parser.add_argument("--out_dir", default=DEFAULT_OUT,          help="Directory to write generated .job files into")
    parser.add_argument("--queue",   default=DEFAULT_QUEUE,        help="SGE queue string")
    parser.add_argument("--ncpus",   type=int, default=DEFAULT_NCPUS, help="Number of CPUs per job (-pe smp)")
    parser.add_argument("--ids",     nargs="*", metavar="PDB_ID",  help="Limit to these PDB IDs (default: all in CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids  = unique_pdb_ids(csv_path)
    if args.ids:
        ids = [i.upper() for i in args.ids]
    else:
        ids = all_ids

    print(f"Generating {len(ids)} job scripts → {out_dir}")
    for pid in ids:
        path = write_job(
            pid=pid,
            pdb_dir=args.pdb_dir,
            ffx=args.ffx,
            out_dir=out_dir,
            queue=args.queue,
            ncpus=args.ncpus,
        )
        print(f"  Wrote {path.name}")

    # Also print a submission snippet
    submit_script = out_dir / "submit_all.sh"
    lines = ["#!/bin/bash", f"# Submit all {len(ids)} pKa jobs", ""]
    for pid in ids:
        lines.append(f"qsub {pid}_minimize.job")
    submit_script.write_text("\n".join(lines) + "\n")
    submit_script.chmod(0o755)
    print(f"\nSubmission script → {submit_script}")
    print(f"Run on cluster:  cd {out_dir} && bash submit_all.sh")


if __name__ == "__main__":
    main()
