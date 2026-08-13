
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
DRY_RUN   = "--dry"     in sys.argv
FORCE     = "--force"   in sys.argv   # regenerate + resubmit even if done
ONLY_ROT  = "--rotamer" in sys.argv
ONLY_TIT  = "--titrate" in sys.argv or "--ph" in sys.argv

# --only PDB_ID [PDB_ID ...]: process only these proteins
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

# --ph can appear multiple times: collect all specified pHs
_ph_indices = [i for i, a in enumerate(sys.argv) if a == "--ph"]
PH_FILTER = []
for _i in _ph_indices:
    try:
        PH_FILTER.append(float(sys.argv[_i + 1]))
    except (IndexError, ValueError):
        print("ERROR: --ph requires a float value (e.g., --ph 3.94)")
        sys.exit(1)

# Derived: which job types to actually run
RUN_ROTAMER = ONLY_ROT or (not ONLY_ROT and not ONLY_TIT)
RUN_TITRATE = ONLY_TIT or (not ONLY_ROT and not ONLY_TIT)

# Active pH list (full list unless --ph filtered it)
ACTIVE_PHS = [ph for ph in TITRATION_PHS if ph in PH_FILTER] if PH_FILTER else TITRATION_PHS

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

def make_rotamer_job_script(pdb_id: str, pdb_abs: str, ffx_prop: str, restart_path: str) -> str:
    job_name = f"rot_{pdb_id}"
    return f"""\
#!/bin/bash
#$ -V                        # Inherit current environment
#$ -cwd                      # Start job in submission directory
#$ -N {job_name}             # Job Name
#$ -j y                      # Combine stderr and stdout
#$ -q MS,UI-GPU,all.q        # Queue
#$ -pe smp 20                # Request 20 tasks/node
#$ -o $JOB_NAME.$JOB_ID.log  # Name of output file
#$ -l h_rt={WALLTIME}        # Run Time
#$ -S /bin/bash              # Shell to use
#$ -l ngpus={N_GPUS}

echo "=== Rotamer job started: $(date) ==="
echo "PDB: {pdb_abs}"

# Step 1: Coarse minimize — runs at most once; skip if output already exists
# (protects against all.q preemption/restart rerunning an expensive step)
if [ ! -f "{pdb_abs}_2" ]; then
    {FFX_CMD} Minimize -e 0.8 {pdb_abs} -Dkey={ffx_prop}
else
    echo "  Minimize output already exists, skipping."
fi

# Step 2: Rotamer optimization — restart via --eR if a restart file is present
{FFX_CMD} Scheduler -p 20 -m {MEM_PER_JOB} > scheduler_rotamer.log & sleep 30s
if [ -f "{restart_path}" ]; then
    echo "  Restart file found: {restart_path} — adding --eR"
    {FFX_CMD} ManyBody -Dpj.nn=1 -Dpj.nt=20 -DnumCudaDevices=1 --eR "{restart_path}" {pdb_abs}_2 -Dkey={ffx_prop}
else
    {FFX_CMD} ManyBody -Dpj.nn=1 -Dpj.nt=20 -DnumCudaDevices=1 {pdb_abs}_2 -Dkey={ffx_prop}
fi

echo "=== Rotamer job done: $(date) ==="
"""

def make_titration_job_script(
    pdb_id: str, pdb_abs: str, pdb_dir: str,
    ffx_prop: str, titrate_prop: str, ph: float,
) -> str:
    ph_str          = str(ph)
    ph_safe         = ph_str.replace(".", "p")          # e.g. "3p94" for job name
    job_name        = f"titr_{pdb_id}_pH{ph_safe}"
    rotamer_job     = f"rot_{pdb_id}"
    rotopt          = f"{pdb_abs}_3"                    # output of rotamer ManyBody
    ph_input        = f"{pdb_dir}/{pdb_id}_pH{ph_str}.pdb"
    titrate_out     = f"{ph_input}_2"                   # ManyBody --tR output
    titrate_restart = f"{pdb_dir}/{pdb_id}_pH{ph_str}.restart"

    return f"""\
#!/bin/bash
#$ -V                        # Inherit current environment
#$ -cwd                      # Start job in submission directory
#$ -N {job_name}             # Job Name
#$ -j y                      # Combine stderr and stdout
#$ -q MS,UI-GPU,all.q        # Queue
#$ -pe smp 20                # Request 20 tasks/node
#$ -o $JOB_NAME.$JOB_ID.log  # Name of output file
#$ -l h_rt={WALLTIME}        # Run Time
#$ -S /bin/bash              # Shell to use
#$ -l ngpus={N_GPUS}
#$ -hold_jid {rotamer_job}   # Wait for rotamer job if still running

echo "=== Titration pH {ph} started: $(date) ==="

# Verify rotamer output exists before proceeding
if [ ! -f "{rotopt}" ]; then
  echo "ERROR: rotamer output not found: {rotopt}" >&2
  exit 1
fi

# Copy rotamer output — runs at most once; skip if already copied
# (protects against all.q preemption/restart rerunning this step)
if [ ! -f "{ph_input}" ]; then
    cp "{rotopt}" "{ph_input}"
else
    echo "  Copy already exists, skipping."
fi

# Titration rotamer optimization at pH {ph}
# Skipped if output already exists (i.e. job restarted mid-Minimize)
# Uses --eR if a restart file is present from a prior interrupted run
if [ ! -f "{titrate_out}" ]; then
    {FFX_CMD} Scheduler -p 20 -m {MEM_PER_JOB} > scheduler_rotamer.log & sleep 30s
    if [ -f "{titrate_restart}" ]; then
        echo "  Restart file found: {titrate_restart} — adding --eR"
        {FFX_CMD_TITRATE} ManyBody --tR --pH {ph} --oT -T --kPH 3.0 --eR "{titrate_restart}" "{ph_input}" -Dkey={titrate_prop}
    else
        {FFX_CMD_TITRATE} ManyBody --tR --pH {ph} --oT -T --kPH 3.0 "{ph_input}" -Dkey={titrate_prop}
    fi
else
    echo "  ManyBody output already exists, skipping."
fi

# Final minimize — saves induced dipoles (.uind) and permanent multipoles (.uperm)
{FFX_CMD} Minimize -e 0.1 "{titrate_out}" -Dkey={ffx_prop} --saveInduced --savePermanentMoments

echo "=== Titration pH {ph} done: $(date) ==="
"""

def get_queued_job_names() -> set:
    """Return the set of job names currently in the SGE queue (any state: running, pending, etc.)."""
    try:
        result = subprocess.run(["qstat"], capture_output=True, text=True, timeout=30)
        names = set()
        for line in result.stdout.splitlines()[2:]:   # first two lines are header
            parts = line.split()
            if len(parts) >= 3:
                names.add(parts[2])
        return names
    except FileNotFoundError:
        return set()
    except subprocess.TimeoutExpired:
        print("WARNING: qstat timed out — assuming no jobs in queue")
        return set()

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
    if ONLY_IDS:
        pdbs = [p for p in pdbs if p.parent.name.upper() in ONLY_IDS]
    mode  = "DRY RUN" if DRY_RUN else "SUBMITTING to SGE"
    force = " [--force]" if FORCE else ""
    only_tag = f" [--only {len(ONLY_IDS)} PDBs]" if ONLY_IDS else ""
    jobs_desc = ("rotamer only" if ONLY_ROT and not ONLY_TIT
                 else "titration only" if ONLY_TIT and not ONLY_ROT
                 else "rotamer + titration")
    print(f"Processing {len(pdbs)} input PDBs  [{mode}{force}{only_tag}]  jobs: {jobs_desc}")
    print(f"Titration pHs: {ACTIVE_PHS}")

    # Snapshot the SGE queue once — always, so in-progress jobs are never re-submitted.
    queued_names: set = get_queued_job_names()
    if queued_names:
        print(f"  [queue] {len(queued_names)} jobs found in SGE queue — running jobs will be skipped.")

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
            restart_path = str(pdb_path.parent / f"{pdb_stem}.restart")

            # ── Rotamer job ──────────────────────────────────────────────────
            if RUN_ROTAMER:
                job_name      = f"rot_{pdb_id}"
                rotamer_script = os.path.join(JOBS_DIR, f"rot_{pdb_id}.job")
                rotopt_done   = (pdb_path.parent / f"{pdb_stem}.pdb_3").exists()

                if job_name in queued_names:
                    # Job is active in the queue — leave script and queue entry untouched.
                    print(f"  [running] {pdb_id}  rotamer is in SGE queue — skipping")

                elif rotopt_done and not FORCE:
                    # Output exists and --force not set — refresh script but don't re-submit.
                    _write_script(rotamer_script, make_rotamer_job_script(pdb_id, pdb_abs, ffx_prop, restart_path))
                    print(f"  [done] {pdb_id}  rotamer output exists — script updated, skipping submit")

                elif DRY_RUN:
                    _write_script(rotamer_script, make_rotamer_job_script(pdb_id, pdb_abs, ffx_prop, restart_path))
                    print(f"  [dry]  {pdb_id}  rotamer → {rotamer_script}")
                    writer.writerow({"pdb_id": pdb_id, "ph": "", "job_type": "rotamer",
                                     "job_script": rotamer_script, "job_id": "dry-run",
                                     "status": "generated", "notes": ""})
                    logf.flush()

                else:
                    # Normal submit -or- --force for a job not currently in the queue.
                    _write_script(rotamer_script, make_rotamer_job_script(pdb_id, pdb_abs, ffx_prop, restart_path))
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
            if not RUN_TITRATE:
                continue
            for ph in ACTIVE_PHS:
                ph_str   = str(ph)
                ph_safe  = ph_str.replace(".", "p")        # matches job name in script
                job_name = f"titr_{pdb_id}_pH{ph_safe}"
                t_script = os.path.join(JOBS_DIR, f"titr_{pdb_id}_pH{ph_str}.job")
                uind_path = pdb_path.parent / f"{pdb_id}_pH{ph_str}.uind"
                titrate_done = uind_path.exists()

                if job_name in queued_names:
                    # Job is active in the queue — leave script and queue entry untouched.
                    print(f"  [running] {pdb_id}  pH {ph}  is in SGE queue — skipping")

                elif titrate_done:
                    # Output (.uind) exists in the PDB directory — job is finished, always skip.
                    _write_script(t_script, make_titration_job_script(
                        pdb_id, pdb_abs, pdb_dir, ffx_prop, titrate_prop, ph))
                    print(f"  [done] {pdb_id}  pH {ph}  .uind exists — script updated, skipping submit")

                elif DRY_RUN:
                    _write_script(t_script, make_titration_job_script(
                        pdb_id, pdb_abs, pdb_dir, ffx_prop, titrate_prop, ph))
                    print(f"  [dry]  {pdb_id}  pH {ph} → {t_script}")
                    writer.writerow({"pdb_id": pdb_id, "ph": ph_str, "job_type": "titration",
                                     "job_script": t_script, "job_id": "dry-run",
                                     "status": "generated", "notes": ""})
                    logf.flush()

                else:
                    # Normal submit -or- --force for a job not currently in the queue.
                    _write_script(t_script, make_titration_job_script(
                        pdb_id, pdb_abs, pdb_dir, ffx_prop, titrate_prop, ph))
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

