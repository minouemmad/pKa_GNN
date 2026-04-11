"""
03_run_ffx_minimize.py
"""

import os
import sys
import csv
import stat
import subprocess
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PDB_DIR        = "data/fixed_pdbs"
JOBS_DIR       = "data/sge_jobs"
MIN_DIR        = "data/minimized_pdbs"
LOG_PATH       = "data/minimize_log.csv"

FFX_CMD        = "/Dedicated/schnieders/maemmad/forcefieldx/bin/ffxc"
FFX_PROPERTIES = os.path.abspath("ffx.properties")
SOLUTE_PRM     = os.path.abspath("naphosphate_solute.09Feb21.prm")

# ── SGE resource settings ─────────────────────────────────────────────────────
CORES_PER_NODE = 128
N_GPUS         = 1
PJ_NN          = 4
PJ_NT          = 5
MEM_PER_JOB    = "22G"
WALLTIME       = "10000:00:00"

# ── Minimization parameters ───────────────────────────────────────────────────
RMS_COARSE     = 0.8
RMS_FINAL      = 0.1

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN   = "--dry"    in sys.argv
FORCE     = "--force"  in sys.argv   # re-generate + re-submit even if script exists

LOG_FIELDS = ["pdb_id", "job_script", "job_id", "status", "notes"]


# ── Properties file writer ────────────────────────────────────────────────────

def write_properties_for(pdb_id: str, dest_dir: str) -> str:
    content = f"""\
forcefield amoeba-bio-2018-cphmd
patch {SOLUTE_PRM}
gkterm true
cavmodel gauss-disp
gaussvol-radii-scale 1.0
neck-correction false
tanh-correction false
element-hct-scale false
descreen-offset 0.0
surface-tension 0.103
descreen-hydrogen false
descreen-vdw true
hct-scale 0.7200
gkc 2.455
gk-radius solute
"""
    prop_path = os.path.join(dest_dir, f"{pdb_id}.properties")
    with open(prop_path, "w") as f:
        f.write(content)
    return prop_path


def make_job_script(pdb_id: str, pdb: str, prop_path: str) -> str:
    pdb_abs  = os.path.abspath(pdb)
    work_dir = os.path.dirname(pdb_abs)
    base     = Path(pdb_abs).stem
    job_name = f"min_{pdb_id}"

    step1_output_raw = os.path.join(work_dir, f"{base}.pdb_2")
    step1_renamed    = os.path.join(work_dir, f"{base}_min.pdb")
    step3_output_raw = os.path.join(work_dir, f"{base}_min.pdb_2")
    step3_renamed    = os.path.join(work_dir, f"{base}_min_2.pdb")
    step4_uind       = os.path.join(work_dir, f"{base}_min_2.uind")
    step5_uperm      = os.path.join(work_dir, f"{base}_min_2.uperm")

    script = f"""\
#!/bin/bash
#$ -V
#$ -cwd
#$ -N {job_name}
#$ -j y
#$ -q MS,UI-GPU
#$ -pe smp 20
#$ -o $JOB_NAME.$JOB_ID.log
#$ -l h_rt={WALLTIME}
#$ -S /bin/bash
#$ -l ngpus={N_GPUS}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "PDB: {pdb_abs}"

echo "=== Step 1: Coarse Minimize (RMS={RMS_COARSE}) ==="
{FFX_CMD} Minimize \\
    -e {RMS_COARSE} \\
    {pdb_abs} \\
    -Dkey={prop_path}

if [ ! -f "{step1_output_raw}" ]; then
    echo "ERROR: Step 1 output not found: {step1_output_raw}"
    exit 1
fi

echo "=== Renaming {step1_output_raw} to {step1_renamed} ==="
mv "{step1_output_raw}" "{step1_renamed}"

echo "=== Step 2: Starting Scheduler ==="
{FFX_CMD} Scheduler -p {PJ_NT} -m {MEM_PER_JOB} > scheduler_{pdb_id}.log &
SCHEDULER_PID=$!
sleep 30s

echo "=== Step 3: ManyBody Rotamer Optimization ==="
{FFX_CMD} ManyBody \\
    -Dpj.nn={PJ_NN} \\
    -Dpj.nt={PJ_NT} \\
    -DnumCudaDevices={N_GPUS} \\
    {step1_renamed} \\
    -Dkey={prop_path}

kill $SCHEDULER_PID 2>/dev/null || true

if [ ! -f "{step3_output_raw}" ]; then
    echo "ERROR: Step 3 output not found: {step3_output_raw}"
    echo "Falling back to Step 1 output for final minimize."
    pdb_s4_in="{step1_renamed}"
else
    echo "=== Renaming {step3_output_raw} to {step3_renamed} ==="
    mv "{step3_output_raw}" "{step3_renamed}"
    pdb_s4_in="{step3_renamed}"
fi

echo "=== Step 4: Final Minimize (RMS={RMS_FINAL}, GPU, saveInduced) ==="
{FFX_CMD} Minimize \\
    -e {RMS_FINAL} \\
    "${{pdb_s4_in}}" \\
    -Dkey={prop_path} \\
    --saveInduced

if [ ! -f "{step4_uind}" ]; then
    echo "WARNING: Step 4 .uind not found: {step4_uind}"
fi

echo "=== Step 5: Save Permanent Multipoles ==="
{FFX_CMD} SavePermanentMoments \\
    "${{pdb_s4_in}}" \\
    -Dkey={prop_path}

if [ ! -f "{step5_uperm}" ]; then
    echo "WARNING: Step 5 .uperm not found: {step5_uperm}"
fi

echo "=== Done: $(date) ==="
"""
    return script


def submit_job(script_path: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["qsub", script_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            tokens = result.stdout.split()
            job_id = next((t for t in tokens if t.isdigit()), "unknown")
            return int(job_id) if job_id.isdigit() else job_id, ""
        else:
            return -1, result.stderr.strip()
    except FileNotFoundError:
        return -1, "qsub not found — are you on a submit node?"
    except subprocess.TimeoutExpired:
        return -1, "qsub timed out"


def main():
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(MIN_DIR,  exist_ok=True)

    pdbs = sorted(Path(PDB_DIR).glob("*_fixed.pdb"))

    mode = "DRY RUN" if DRY_RUN else "SUBMITTING to SGE"
    print(f"Processing {len(pdbs)} fixed PDBs  [{mode}]")
    print(f"  --force flag: {'on (re-submit existing jobs)' if FORCE else 'off (skip existing job scripts)'}")

    with open(LOG_PATH, "a", newline="") as logf:   # append so prior runs are preserved
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if os.path.getsize(LOG_PATH) == 0:
            writer.writeheader()

        for fixed_path in pdbs:
            pdb_id      = fixed_path.stem.replace("_fixed", "")
            script_path = os.path.join(JOBS_DIR, f"{pdb_id}_minimize.job")

            # ── Skip if final outputs already exist (uind or pdb_2) ───────────
            uind_path = fixed_path.parent / f"{fixed_path.stem}_min_2.uind"
            pdb2_path = fixed_path.parent / f"{fixed_path.stem}_min_2.pdb_2"
            if (uind_path.exists() or pdb2_path.exists()) and not FORCE:
                print(f"  [done] {pdb_id:6s}  final output already exists")
                continue

            # ── Skip if the job script already exists (unless --force) ────────
            if os.path.exists(script_path) and not FORCE:
                print(f"  [skip] {pdb_id:6s}  job script already exists")
                continue

            prop_path   = write_properties_for(pdb_id, str(fixed_path.parent))
            script_text = make_job_script(pdb_id, str(fixed_path), prop_path)

            with open(script_path, "w") as f:
                f.write(script_text)
            os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

            if DRY_RUN:
                log = {
                    "pdb_id": pdb_id, "job_script": script_path,
                    "job_id": "dry-run", "status": "generated", "notes": ""
                }
                print(f"  [dry] {pdb_id:6s}  script → {script_path}")
            else:
                job_id, err = submit_job(script_path)
                if job_id != -1:
                    log = {
                        "pdb_id": pdb_id, "job_script": script_path,
                        "job_id": job_id, "status": "submitted", "notes": ""
                    }
                    print(f"  [✓] {pdb_id:6s}  job_id={job_id}")
                else:
                    log = {
                        "pdb_id": pdb_id, "job_script": script_path,
                        "job_id": -1, "status": "submit_failed", "notes": err
                    }
                    print(f"  [✗] {pdb_id:6s}  {err[:100]}")

            writer.writerow(log)
            logf.flush()

    print(f"\nJob scripts → {JOBS_DIR}/")
    print(f"Log         → {LOG_PATH}")


if __name__ == "__main__":
    main()
