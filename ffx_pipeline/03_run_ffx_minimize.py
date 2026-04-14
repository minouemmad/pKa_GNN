"""
03_run_ffx_minimize.py

Generate and optionally submit SGE job scripts for:
  1. Rotamer job  ({pdb_id}_rotamer.job)
     Coarse minimize + ManyBody rotamer optimisation.
     Output: {pdb_stem}.pdb_3

  2. Per-pH titration jobs  ({pdb_id}_titrate_pH{ph}.job)  — one per TITRATION_PHS
     Copies rotopt output, runs titration ManyBody, then final minimize with
     --saveInduced --savePermanentMoments.
     Output: {pdb_dir}/{pdb_id}_pH{ph}.pdb_2.uind / .uperm

Flags:
  --dry    Generate scripts but do not submit to SGE.
  --force  Regenerate + resubmit even if outputs already exist.
           Combined with --dry: regenerates all scripts without submitting.
"""

import os
import sys
import csv
import stat
import subprocess
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# PIPELINE_ROOT = ffx_pipeline/   (all pipeline data lives here)
# REPO_ROOT     = pKa_GNN/        (shared files: .prm, etc.)
PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT     = PIPELINE_ROOT.parent
PDB_DIR  = str(PIPELINE_ROOT / "data/fixed_pdbs")
JOBS_DIR = str(PIPELINE_ROOT / "data/sge_jobs")
LOG_PATH = str(PIPELINE_ROOT / "data/minimize_log.csv")

FFX_CMD         = "/Dedicated/schnieders/maemmad/forcefieldx/bin/ffxc"
FFX_CMD_TITRATE = "/Dedicated/schnieders/rgogal/software/forcefieldx/bin/ffxc"
SOLUTE_PRM      = str(REPO_ROOT / "naphosphate_solute.09Feb21.prm")

# ── SGE resource settings ─────────────────────────────────────────────────────
N_GPUS      = 1
MEM_PER_JOB = "22G"
WALLTIME    = "10000:00:00"

# ── pH values to run titration at ─────────────────────────────────────────────
TITRATION_PHS = [3.94, 4.4, 6.45, 8.55]

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN = "--dry"   in sys.argv
FORCE   = "--force" in sys.argv   # regenerate + resubmit even if done

LOG_FIELDS = ["pdb_id", "ph", "job_type", "job_script", "job_id", "status", "notes"]


# ── Properties file writers ───────────────────────────────────────────────────

_FFX_PROPS_BASE = """\
forcefield amoeba-bio-2018-cphmd
patch {solute_prm}
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
# scf-algorithm SOR
"""

_TITRATE_PROPS_EXTRA = """\
solute-dielectric 2.0
# scf-algorithm SOR -this should only be uncommented if the max scf failure error comes up.
"""


def write_properties_for(pdb_stem: str, dest_dir: str) -> tuple:
    """Write {pdb_stem}.properties and titrate.properties in dest_dir."""
    base_content    = _FFX_PROPS_BASE.format(solute_prm=SOLUTE_PRM)
    titrate_content = base_content + _TITRATE_PROPS_EXTRA
    ffx_path     = os.path.join(dest_dir, f"{pdb_stem}.properties")
    titrate_path = os.path.join(dest_dir, "titrate.properties")
    with open(ffx_path, "w") as f:
        f.write(base_content)
    with open(titrate_path, "w") as f:
        f.write(titrate_content)
    return ffx_path, titrate_path


def make_rotamer_job_script(pdb_id: str, pdb_abs: str, ffx_prop: str) -> str:
    job_name = f"{pdb_id}_rotamer"
    return f"""\
#!/bin/bash
#$ -V                        # Inherit current environment
#$ -cwd                      # Start job in submission directory
#$ -N {job_name}             # Job Name
#$ -j y                      # Combine stderr and stdout
#$ -q MS,UI-GPU              # Queue
#$ -pe smp 20                # Request 20 tasks/node
#$ -o $JOB_NAME.$JOB_ID.log  # Name of output file
#$ -l h_rt={WALLTIME}        # Run Time
#$ -S /bin/bash              # Shell to use
#$ -l ngpus={N_GPUS}

echo "=== Rotamer job started: $(date) ==="
echo "PDB: {pdb_abs}"

# Step 1: Coarse minimize
{FFX_CMD} Minimize -e 0.8 {pdb_abs} -Dplatform=OMM -Dkey={ffx_prop}

# Step 2: Rotamer optimization
{FFX_CMD} Scheduler -p 1 -m {MEM_PER_JOB} > scheduler_rotamer.log & sleep 30s
{FFX_CMD} ManyBody -Dpj.nn=1 -Dpj.nt=20 -DnumCudaDevices=1 -Dplatform=OMM {pdb_abs}_2 -Dkey={ffx_prop}

echo "=== Rotamer job done: $(date) ==="
"""


def make_titration_job_script(
    pdb_id: str, pdb_abs: str, pdb_dir: str,
    ffx_prop: str, titrate_prop: str, ph: float,
) -> str:
    ph_str      = str(ph)
    ph_safe     = ph_str.replace(".", "p")          # e.g. "3p94" for job name
    job_name    = f"{pdb_id}_pH{ph_safe}"
    rotamer_job = f"{pdb_id}_rotamer"
    rotopt      = f"{pdb_abs}_3"                    # output of rotamer ManyBody
    ph_input    = f"{pdb_dir}/{pdb_id}_pH{ph_str}.pdb"
    titrate_out = f"{ph_input}_2"                   # ManyBody --tR output

    return f"""\
#!/bin/bash
#$ -V                        # Inherit current environment
#$ -cwd                      # Start job in submission directory
#$ -N {job_name}             # Job Name
#$ -j y                      # Combine stderr and stdout
#$ -q MS,UI-GPU              # Queue
#$ -pe smp 20                # Request 20 tasks/node
#$ -o $JOB_NAME.$JOB_ID.log  # Name of output file
#$ -l h_rt={WALLTIME}        # Run Time
#$ -S /bin/bash              # Shell to use
#$ -l ngpus={N_GPUS}
#$ -hold_jid {rotamer_job}   # Wait for rotamer job if still running

echo "=== Titration pH {ph} started: $(date) ==="

# Copy rotamer output so each pH run has its own working copy
cp "{rotopt}" "{ph_input}"

# Titration rotamer optimization at pH {ph}
{FFX_CMD_TITRATE} ManyBody --tR --pH {ph} --oT -T --kpH 3.0 "{ph_input}" -Dkey={titrate_prop}

# Final minimize — saves induced dipoles (.uind) and permanent multipoles (.uperm)
{FFX_CMD} Minimize -e 0.1 "{titrate_out}" -Dplatform=OMM -Dkey={ffx_prop} --saveInduced --savePermanentMoments

echo "=== Titration pH {ph} done: $(date) ==="
"""


def submit_job(script_path: str) -> tuple:
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


def _write_script(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def main():
    os.makedirs(JOBS_DIR, exist_ok=True)

    log_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    pdbs  = sorted(Path(PDB_DIR).glob("*/*_input.pdb"))
    mode  = "DRY RUN" if DRY_RUN else "SUBMITTING to SGE"
    force = " [--force]" if FORCE else ""
    print(f"Processing {len(pdbs)} input PDBs  [{mode}{force}]")
    print(f"Titration pHs: {TITRATION_PHS}")

    with open(LOG_PATH, "a", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if not log_exists:
            writer.writeheader()

        for pdb_path in pdbs:
            pdb_id   = pdb_path.parent.name
            pdb_abs  = str(pdb_path.resolve())
            pdb_dir  = str(pdb_path.parent.resolve())
            pdb_stem = pdb_path.stem

            ffx_prop, titrate_prop = write_properties_for(pdb_stem, pdb_dir)

            # ── Rotamer job ──────────────────────────────────────────────────
            rotamer_script = os.path.join(JOBS_DIR, f"{pdb_id}_rotamer.job")
            _write_script(rotamer_script, make_rotamer_job_script(pdb_id, pdb_abs, ffx_prop))

            rotopt_done = (pdb_path.parent / f"{pdb_stem}.pdb_3").exists()

            if rotopt_done and not FORCE:
                print(f"  [done] {pdb_id}  rotamer output exists — script updated, skipping submit")
            elif DRY_RUN:
                print(f"  [dry]  {pdb_id}  rotamer → {rotamer_script}")
                writer.writerow({"pdb_id": pdb_id, "ph": "", "job_type": "rotamer",
                                 "job_script": rotamer_script, "job_id": "dry-run",
                                 "status": "generated", "notes": ""})
                logf.flush()
            else:
                job_id, err = submit_job(rotamer_script)
                status = "submitted" if job_id != -1 else "submit_failed"
                note   = err[:120] if err else ""
                print(f"  [{'✓' if job_id != -1 else '✗'}] {pdb_id}  rotamer  job_id={job_id}"
                      + (f"  {note}" if note else ""))
                writer.writerow({"pdb_id": pdb_id, "ph": "", "job_type": "rotamer",
                                 "job_script": rotamer_script, "job_id": job_id,
                                 "status": status, "notes": note})
                logf.flush()

            # ── Titration jobs — one per pH ──────────────────────────────────
            for ph in TITRATION_PHS:
                ph_str   = str(ph)
                t_script = os.path.join(JOBS_DIR, f"{pdb_id}_titrate_pH{ph_str}.job")
                uind_path = pdb_path.parent / f"{pdb_id}_pH{ph_str}.pdb_2.uind"

                _write_script(t_script, make_titration_job_script(
                    pdb_id, pdb_abs, pdb_dir, ffx_prop, titrate_prop, ph))

                titrate_done = uind_path.exists()

                if titrate_done and not FORCE:
                    print(f"  [done] {pdb_id}  pH {ph}  .uind exists — script updated, skipping submit")
                elif DRY_RUN:
                    print(f"  [dry]  {pdb_id}  pH {ph} → {t_script}")
                    writer.writerow({"pdb_id": pdb_id, "ph": ph_str, "job_type": "titration",
                                     "job_script": t_script, "job_id": "dry-run",
                                     "status": "generated", "notes": ""})
                    logf.flush()
                else:
                    job_id, err = submit_job(t_script)
                    status = "submitted" if job_id != -1 else "submit_failed"
                    note   = err[:120] if err else ""
                    print(f"  [{'✓' if job_id != -1 else '✗'}] {pdb_id}  pH {ph}  job_id={job_id}"
                          + (f"  {note}" if note else ""))
                    writer.writerow({"pdb_id": pdb_id, "ph": ph_str, "job_type": "titration",
                                     "job_script": t_script, "job_id": job_id,
                                     "status": status, "notes": note})
                    logf.flush()

    print(f"\nJob scripts → {JOBS_DIR}/")
    print(f"Log         → {LOG_PATH}")


if __name__ == "__main__":
    main()

