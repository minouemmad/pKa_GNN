#!/bin/bash
# fix_scf_failures.sh
# Run from the pKa_GNN directory (where min_*.log files live).
# Finds minimization logs with SCF failures, qdels any running job,
# deletes the old log, rewrites the job file with staged polarization,
# patches .properties, and resubmits.
#
# Usage:
#   ./fix_scf_failures.sh            # patch + submit
#   ./fix_scf_failures.sh --dry-run  # patch only, print qsub commands

BASE_DIR="$(pwd)"
SGE_DIR="${BASE_DIR}/data/sge_jobs"
PDB_DIR="${BASE_DIR}/data/fixed_pdbs"
FFX=/Dedicated/schnieders/maemmad/forcefieldx/bin/ffxc
DRY_RUN=${1:-}

echo "Scanning for SCF-failed min logs in: ${BASE_DIR}"
echo ""

found=0
resubmit_list=()

for logfile in min_*.*.log; do
    [ -f "${logfile}" ] || continue

    # Check for either failure pattern
    if ! grep -qE "Maximum SCF iterations reached|SCF convergence failure" "${logfile}"; then
        continue
    fi

    # Extract PDB name and job ID: min_2CI2.5801238.log → 2CI2, 5801238
    pdb=$(echo "${logfile}" | sed 's/^min_//; s/\.[0-9]*\.log$//')
    job_id=$(echo "${logfile}" | grep -oE '\.[0-9]+\.log$' | grep -oE '[0-9]+')

    echo "=== SCF failure: ${logfile}  →  PDB: ${pdb}  JOB_ID: ${job_id} ==="

    # ── Cancel the running job if it is still in the queue ────────────────────
    if [ -n "${job_id}" ]; then
        if qstat -j "${job_id}" &>/dev/null; then
            echo "  Cancelling: qdel ${job_id}"
            qdel "${job_id}" 2>/dev/null && echo "  Cancelled job ${job_id}" || echo "  WARNING: qdel ${job_id} failed (may have already finished)"
        else
            echo "  Job ${job_id} not in queue (already finished or never queued)"
        fi
    fi

    # ── Delete the failed log file ─────────────────────────────────────────────
    echo "  Removing:  ${logfile}"
    rm -f "${logfile}"

    job_file="${SGE_DIR}/${pdb}_minimize.job"
    props_file="${PDB_DIR}/${pdb}.properties"

    mkdir -p "${SGE_DIR}"

    # ── Write staged job file ─────────────────────────────────────────────────
    # Quoted heredoc: nothing expands here. sed patches in PDB/path values.
    cat << 'JOBEOF' | sed \
        -e "s|PDBNAME|${pdb}|g" \
        -e "s|FFXPATH|${FFX}|g" \
        -e "s|BASEPATH|${PDB_DIR}|g" \
        > "${job_file}"
#!/bin/bash
#$ -V
#$ -cwd
#$ -N min_PDBNAME
#$ -j y
#$ -q MS,UI-GPU
#$ -pe smp 20
#$ -o $JOB_NAME.$JOB_ID.log
#$ -l h_rt=10000:00:00
#$ -S /bin/bash
#$ -l ngpus=1

echo "Job started: $(date)"
echo "Host: $(hostname)"

BASE=BASEPATH
FFX=FFXPATH
PROPS="-Dkey=data/fixed_pdbs/PDBNAME.properties"
PDB_IN="${BASE}/PDBNAME_fixed.pdb"

# ── Step 1a: No polarization — relax idealized H atoms, zero SCF cost ─────────
echo "=== Step 1a: Minimize polarization=NONE (RMS=2.0) ==="
${FFX} Minimize -e 2.0 "${PDB_IN}" ${PROPS} -Dpolarization=NONE
if [ ! -f "${PDB_IN}_2" ]; then
    echo "ERROR: Step 1a output not found: ${PDB_IN}_2"; exit 1
fi
mv "${PDB_IN}_2" "${BASE}/PDBNAME_s1a.pdb"

# ── Step 1b: Direct polarization — still no SCF iterations ────────────────────
echo "=== Step 1b: Minimize polarization=DIRECT (RMS=1.0) ==="
${FFX} Minimize -e 1.0 "${BASE}/PDBNAME_s1a.pdb" ${PROPS} -Dpolarization=DIRECT
if [ ! -f "${BASE}/PDBNAME_s1a.pdb_2" ]; then
    echo "ERROR: Step 1b output not found: ${BASE}/PDBNAME_s1a.pdb_2"; exit 1
fi
mv "${BASE}/PDBNAME_s1a.pdb_2" "${BASE}/PDBNAME_s1b.pdb"

# ── Step 1c: Mutual polarization — geometry now conditioned for SCF ────────────
echo "=== Step 1c: Minimize polarization=MUTUAL (RMS=0.8) ==="
${FFX} Minimize -e 0.8 "${BASE}/PDBNAME_s1b.pdb" ${PROPS} -Dpolarization=MUTUAL
if [ ! -f "${BASE}/PDBNAME_s1b.pdb_2" ]; then
    echo "ERROR: Step 1c output not found: ${BASE}/PDBNAME_s1b.pdb_2"; exit 1
fi
mv "${BASE}/PDBNAME_s1b.pdb_2" "${BASE}/PDBNAME_fixed_min.pdb"

# ── Step 2: Start Parallel Java scheduler (background) ────────────────────────
echo "=== Step 2: Starting Scheduler ==="
${FFX} Scheduler -p 5 -m 22G > scheduler_PDBNAME.log &
SCHEDULER_PID=$!
sleep 30s

# ── Step 3: Many-body rotamer optimization ────────────────────────────────────
echo "=== Step 3: ManyBody Rotamer Optimization ==="
${FFX} ManyBody \
    -Dpj.nn=4 \
    -Dpj.nt=5 \
    -DnumCudaDevices=1 \
    "${BASE}/PDBNAME_fixed_min.pdb" \
    ${PROPS}

kill $SCHEDULER_PID 2>/dev/null || true

if [ ! -f "${BASE}/PDBNAME_fixed_min.pdb_2" ]; then
    echo "WARNING: Step 3 output not found, falling back to Step 1c output."
    pdb_s4_in="${BASE}/PDBNAME_fixed_min.pdb"
else
    mv "${BASE}/PDBNAME_fixed_min.pdb_2" "${BASE}/PDBNAME_fixed_min_2.pdb"
    pdb_s4_in="${BASE}/PDBNAME_fixed_min_2.pdb"
fi

# ── Step 4: Tight final minimization + write induced dipoles ──────────────────
echo "=== Step 4: Final Minimize (RMS=0.1, saveInduced) ==="
${FFX} Minimize -e 0.1 "${pdb_s4_in}" ${PROPS} --saveInduced
if [ ! -f "${pdb_s4_in}_2" ]; then
    echo "ERROR: Step 4 output not found: ${pdb_s4_in}_2"; exit 1
fi
mv "${pdb_s4_in}_2" "${BASE}/PDBNAME_final.pdb"

echo "=== Done: $(date) ==="
echo "Final structure: ${BASE}/PDBNAME_final.pdb"
JOBEOF

    chmod +x "${job_file}"
    echo "  Written:  ${job_file}"

    # ── Patch properties file ─────────────────────────────────────────────────
    if [ -f "${props_file}" ]; then
        if grep -q "^scf-predictor" "${props_file}"; then
            sed -i 's/^scf-predictor.*/scf-predictor    aspc/' "${props_file}"
            echo "  Updated:  scf-predictor → aspc in ${props_file}"
        else
            echo "scf-predictor    aspc" >> "${props_file}"
            echo "  Added:    scf-predictor aspc to ${props_file}"
        fi

        if grep -q "^polar-eps" "${props_file}"; then
            sed -i 's/^polar-eps.*/polar-eps        1.0e-5/' "${props_file}"
            echo "  Updated:  polar-eps → 1.0e-5 in ${props_file}"
        else
            echo "polar-eps        1.0e-5" >> "${props_file}"
            echo "  Added:    polar-eps 1.0e-5 to ${props_file}"
        fi
    else
        echo "  WARNING:  Properties not found: ${props_file}"
    fi

    echo ""
    resubmit_list+=("${job_file}")
    found=$((found + 1))
done

if [ "${found}" -eq 0 ]; then
    echo "No SCF-failed log files found."
else
    echo "Patched ${found} job(s)."
    echo ""
    for jf in "${resubmit_list[@]}"; do
        if [ "${DRY_RUN}" = "--dry-run" ]; then
            echo "  [dry-run] qsub ${jf}"
        else
            echo "  Submitting: qsub ${jf}"
            qsub "${jf}"
        fi
    done
fi
