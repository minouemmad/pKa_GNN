
from __future__ import annotations

import csv
import os
import stat
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Anchor everything to pKa_GNN/ regardless of the CWD used to invoke this script.
PIPELINE_DIR = Path(__file__).resolve().parent          # pKa_GNN/tinker_pipeline/
PKA_GNN_DIR  = PIPELINE_DIR.parent                     # pKa_GNN/

PDB_DIR      = str(PIPELINE_DIR / "data/fixed_pdbs")
JOBS_DIR     = str(PKA_GNN_DIR / "data/sge_jobs_tinker")
LOG_PATH     = str(PKA_GNN_DIR / "data/tinker_minimize_log.csv")
MIN_DIR      = str(PKA_GNN_DIR / "Graph_pKa/Data/7_Energy_Minimization_Systems")
NEUTRAL_DIR  = str(PKA_GNN_DIR / "Graph_pKa/Data/6_Neutralized_System")

# ── SGE resource settings ─────────────────────────────────────────────────────
WALLTIME     = "10000:00:00"
MEM          = "22G"
N_CPUS       = 20

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN = "--dry"   in sys.argv
FORCE   = "--force" in sys.argv

# --only PDB_ID [PDB_ID ...]: submit only these proteins (checks NEUTRAL_DIR)
_only_indices = [i for i, a in enumerate(sys.argv) if a == "--only"]
ONLY_IDS: set | None = None
if _only_indices:
    _only_vals = []
    for _oi in _only_indices:
        _j = _oi + 1
        while _j < len(sys.argv) and not sys.argv[_j].startswith("--"):
            _only_vals.append(sys.argv[_j].upper())
            _j += 1
    ONLY_IDS = set(_only_vals) if _only_vals else None

LOG_FIELDS = ["pdb_id", "job_type", "job_script", "job_id", "status", "notes"]

# ── Script generators ─────────────────────────────────────────────────────────

def make_prep_script(pka_gnn_abs: str) -> str:
    """Generate the one-time preprocessing job bash script."""
    return f"""\
#!/bin/bash
#$ -V                        # Inherit current environment
#$ -cwd                      # Start job in submission directory
#$ -N tinker_prep            # Job name
#$ -j y                      # Merge stderr into stdout
#$ -q MS,UI                  # Queue (no GPU needed)
#$ -pe smp {N_CPUS}          # Threads
#$ -o $JOB_NAME.$JOB_ID.log  # Output log
#$ -l h_rt={WALLTIME}        # Wall time
#$ -S /bin/bash

echo "=== tinker_prep started: $(date) ==="
cd "{pka_gnn_abs}"

python tinker_pipeline/tinker_prep_all.py

echo "=== tinker_prep done: $(date) ==="
"""

def make_minimize_script(pdb_id: str, pka_gnn_abs: str, hold_prep: bool = True) -> str:
    """Generate a per-protein minimize job bash script."""
    job_name = f"min_{pdb_id}_tinker"
    hold_line = "#$ -hold_jid tinker_prep     # Wait for preprocessing to finish\n" if hold_prep else ""
    return f"""\
#!/bin/bash
#$ -V                        # Inherit current environment
#$ -cwd                      # Start job in submission directory
#$ -N {job_name}             # Job name
#$ -j y                      # Merge stderr into stdout
#$ -q MS,UI                  # Queue (no GPU needed for minimize.x)
#$ -pe smp 10          # Threads
#$ -o $JOB_NAME.$JOB_ID.log  # Output log
#$ -l h_rt={WALLTIME}        # Wall time
#$ -S /bin/bash
{hold_line}
echo "=== {pdb_id} tinker_min started: $(date) ==="
cd "{pka_gnn_abs}"

python tinker_pipeline/tinker_minimize_one.py --pdb-id {pdb_id}

echo "=== {pdb_id} tinker_min done: $(date) ==="
"""

# ── Utilities ─────────────────────────────────────────────────────────────────

def _write_script(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)

def _submit(script_path: str) -> tuple:
    try:
        result = subprocess.run(
            ["qsub", script_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            tokens = result.stdout.split()
            job_id = next((t for t in tokens if t.isdigit()), "unknown")
            return int(job_id) if job_id.isdigit() else job_id, ""
        return -1, result.stderr.strip()
    except FileNotFoundError:
        return -1, "qsub not found — are you on a submit node?"
    except subprocess.TimeoutExpired:
        return -1, "qsub timed out"

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)

    pka_gnn_abs = str(PKA_GNN_DIR)

    if ONLY_IDS:
        # Build list from NEUTRAL_DIR: only IDs that have a .xyz there
        neutral_path = Path(NEUTRAL_DIR)
        pdbs = []
        missing = []
        for pid in sorted(ONLY_IDS):
            xyz = neutral_path / f"{pid}.xyz"
            if xyz.exists():
                pdbs.append(pid)
            else:
                missing.append(pid)
        if missing:
            print(f"  [skip] not in {NEUTRAL_DIR}: {', '.join(missing)}")
        pdb_ids = pdbs
    else:
        pdb_ids = [p.parent.name for p in sorted(Path(PDB_DIR).glob("*/*_input.pdb"))]

    mode = "DRY RUN" if DRY_RUN else "SUBMITTING to SGE"
    force_tag = "  [--force]" if FORCE else ""
    only_tag  = f"  [--only {len(pdb_ids)} proteins]" if ONLY_IDS else ""
    print(f"Tinker pipeline — {len(pdb_ids)} proteins  [{mode}{force_tag}{only_tag}]")

    log_exists = Path(LOG_PATH).exists() and Path(LOG_PATH).stat().st_size > 0

    with open(LOG_PATH, "a", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if not log_exists:
            writer.writeheader()

        # ── 1. Prep job (one-time, all proteins) ─────────────────────────────
        prep_script = os.path.join(JOBS_DIR, "tinker_prep.job")
        _write_script(prep_script, make_prep_script(pka_gnn_abs))

        neutral_dir_exists = Path(NEUTRAL_DIR).is_dir() and any(
            Path(NEUTRAL_DIR).glob("*.xyz")
        )

        if neutral_dir_exists and not FORCE:
            print(f"  [done] prep   — neutralized XYZ already present, skipping submit")
        elif DRY_RUN:
            print(f"  [dry]  prep   → {prep_script}")
            writer.writerow({"pdb_id": "ALL", "job_type": "prep",
                             "job_script": prep_script, "job_id": "dry-run",
                             "status": "generated", "notes": ""})
            logf.flush()
        else:
            job_id, err = _submit(prep_script)
            status = "submitted" if job_id != -1 else "submit_failed"
            print(f"  [{'✓' if job_id != -1 else '✗'}] prep   job_id={job_id}"
                  + (f"  {err[:80]}" if err else ""))
            writer.writerow({"pdb_id": "ALL", "job_type": "prep",
                             "job_script": prep_script, "job_id": job_id,
                             "status": status, "notes": err[:120] if err else ""})
            logf.flush()

        # ── 2. Per-protein minimize jobs ──────────────────────────────────────
        for pdb_id in pdb_ids:
            uind_out  = Path(MIN_DIR) / pdb_id / f"{pdb_id}.uind"
            t_script  = os.path.join(JOBS_DIR, f"min_{pdb_id}_tinker.job")

            _write_script(t_script, make_minimize_script(pdb_id, pka_gnn_abs, hold_prep=not bool(ONLY_IDS)))

            if uind_out.exists() and not FORCE:
                print(f"  [done] {pdb_id:6s}  .uind already exists — script updated, skipping")
                continue

            if DRY_RUN:
                print(f"  [dry]  {pdb_id:6s} → {t_script}")
                writer.writerow({"pdb_id": pdb_id, "job_type": "minimize",
                                 "job_script": t_script, "job_id": "dry-run",
                                 "status": "generated", "notes": ""})
                logf.flush()
            else:
                job_id, err = _submit(t_script)
                status = "submitted" if job_id != -1 else "submit_failed"
                note   = err[:120] if err else ""
                print(f"  [{'✓' if job_id != -1 else '✗'}] {pdb_id:6s}  job_id={job_id}"
                      + (f"  {note}" if note else ""))
                writer.writerow({"pdb_id": pdb_id, "job_type": "minimize",
                                 "job_script": t_script, "job_id": job_id,
                                 "status": status, "notes": note})
                logf.flush()

    print(f"\nJob scripts → {JOBS_DIR}/")
    print(f"Log         → {LOG_PATH}")
    print(f"\nWorkflow order:")
    print(f"  1. tinker_prep.job  — preprocesses ALL proteins (pdbxyz, solvate, neutralise)")
    print(f"  2. {{PDB_ID}}_tinker_min.job  — minimize.x for each protein (auto-holds on prep)")

if __name__ == "__main__":
    main()
