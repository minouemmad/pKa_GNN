"""
03c_genz_pka_jobs.py

Physics pKa estimates from FFX GenZpKa (Boltzmann-weighted protonation
populations over the ManyBody rotamer x protonation-state ensemble), replacing
the old "full-protein rotamer job + 4 discrete-pH titration jobs" scheme.

Why: the discrete 4-pH titration states carried ~no pKa information (GLU
almost never protonated at pH 3.94, LYS/TYR/CYS above the 8.55 grid) and the
whole-protein rotamer optimisation is what made big proteins take days.

Per protein (one SGE job, restartable — every step has a done-marker):
  1. Coarse minimise the SSBOND-aware v2 input            ({ID}_input.pdb_2)
  2. GenZpKa at REF_PH with --pF  → full energy expansion + energy restart
     restricted (--lR) to titratable residues near labelled sites + their
     packing shell, chain-aware (FFX's own --oT/--iC selection uses a
     substring match on residue numbers and ignores chains, so we build it).
  3. GenZpKa at every other grid pH with --eR <restart> --rCS: pair / 3-body
     energies are pH independent, only self energies (which carry the pH
     bias) are recomputed — far cheaper than a full run per pH.
  4. Final Minimize of the REF_PH structure with --saveInduced
     --savePermanentMoments (GNN node features).  Guarded so restarts don't
     spawn extra .pdb_N files.

Outputs: data/fixed_pdbs_v2/{ID}/genz/pH{X}/  (logs + populations) — parsed by
04b_parse_genz.py into a per-residue physics pKa table.

Usage:
  python 03c_genz_pka_jobs.py --dry                # write job scripts only
  python 03c_genz_pka_jobs.py --only 135L          # submit one protein
  python 03c_genz_pka_jobs.py                      # submit all not yet done
"""
from __future__ import annotations

import argparse
import os
import stat
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT     = PIPELINE_ROOT.parent

PDB_DIR    = PIPELINE_ROOT / "data/fixed_pdbs_v2"
JOBS_DIR   = PIPELINE_ROOT / "data/sge_jobs_genz"
MANIFEST   = REPO_ROOT / "data/manifest.csv"
SOLUTE_PRM = REPO_ROOT / "naphosphate_solute.09Feb21.prm"

FFX_MIN  = "/Dedicated/schnieders/maemmad/forcefieldx/bin/ffxc"          # has --saveInduced/--savePermanentMoments
FFX_GENZ = "/Dedicated/schnieders/programs/ffx-builds/forcefieldx-2026-10-05/bin/ffxc"  # lab build v26.0.4 (GenZpKa)

REF_PH  = 7.0
PH_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]

# Residue selection (Å, side-chain heavy atoms)
TITR_RADIUS = 12.0   # titratable residues this close to a labelled site titrate with it
PACK_RADIUS = 6.0    # any residue this close to a titrating residue gets rotamers
FFX_TITRATABLE = {"ASP", "GLU", "HIS", "LYS", "CYS"}   # FFX ManyBody does not titrate TYR
BACKBONE = {"N", "CA", "C", "O", "OXT"}

ENV_ACTIVATE = "/Dedicated/schnieders/maemmad/envs/activate_pka_gnn.sh"
LARGE_ATOMS = 12000  # heavy-atom count above which the job goes to --large-queue with more memory

_PROPS = """\
forcefield amoeba-bio-2018-cphmd
patch {prm}
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


# ── Structure parsing / residue selection ────────────────────────────────────

def read_residues(pdb: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """Residues in file order with side-chain coordinates, plus SSBOND Cys keys."""
    res: dict[tuple, dict] = {}
    ss: set[tuple[str, str]] = set()
    for line in open(pdb):
        if line.startswith("SSBOND"):
            ss.add((line[15], line[17:21].strip()))
            ss.add((line[29], line[31:35].strip()))
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        if name.startswith("H"):
            continue
        key = (line[21], line[22:26].strip(), line[26])
        r = res.setdefault(key, {"chain": line[21], "num": line[22:26].strip(), "icode": line[26].strip(),
                                 "name": line[17:20].strip(), "sc": [], "all": []})
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        r["all"].append(xyz)
        if name not in BACKBONE:
            r["sc"].append(xyz)
    out = list(res.values())
    for r in out:
        r["sc"] = np.array(r["sc"] or r["all"])   # Gly: fall back to backbone
        r["all"] = np.array(r["all"])
    return out, ss


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())


def select_residues(residues: list[dict], labelled: set[tuple[str, str]],
                    ss: set[tuple[str, str]], TITR_RADIUS: float = TITR_RADIUS,
                    PACK_RADIUS: float = PACK_RADIUS) -> tuple[list[dict], list[dict]]:
    """Return (titrating residues, all residues to optimise) in file order."""
    lab = [r for r in residues if (r["chain"], r["num"]) in labelled]
    titr = []
    for r in residues:
        if r["name"] not in FFX_TITRATABLE:
            continue
        if r["name"] == "CYS" and (r["chain"], r["num"]) in ss:
            continue   # disulfide — not titratable
        if (r["chain"], r["num"]) in labelled or any(_min_dist(r["sc"], l["sc"]) <= TITR_RADIUS for l in lab):
            titr.append(r)
    titr_ids = {id(r) for r in titr}
    opt = [r for r in residues
           if id(r) in titr_ids or (r["chain"], r["num"]) in labelled
           or any(_min_dist(r["all"], t["sc"]) <= PACK_RADIUS for t in titr)]
    return titr, opt


def lr_string(res: list[dict]) -> str:
    return ",".join(f"{(r['chain'].strip() or 'A')}{r['num']}" for r in res)


# ── Job script ───────────────────────────────────────────────────────────────

def job_script(pdb_id: str, d: Path, lr: str, large: bool, args) -> str:
    q    = args.large_queue if large else args.queue
    mem  = "60G" if large else "22G"
    # Argon cgroups give a job (slots/node cores) of the node's RAM: 20 slots on
    # a 93G/80-core node is ~23G, so the 60G heap was OOM-killed (1QK1, 8I41).
    # 40 slots on a 256G/128-core node is ~78G.
    nt   = max(args.threads, 40) if large else args.threads
    # MS jobs get killed, so large systems go to UI-HM (high-memory nodes; 40
    # slots there is well above the 60G heap).  Only all.q needs the explicit
    # node-size request.
    big  = "#$ -l mem_256G=true\n" if large and q == "all.q" else ""
    grid = [REF_PH] + [p for p in PH_GRID if p != REF_PH]
    three = "-T" if args.three_body else ""
    phs  = " ".join(f"{p:g}" for p in grid)
    script = f"""#!/bin/bash
#$ -N genz_{pdb_id}
#$ -q {q}
#$ -pe smp {nt}
{big}#$ -l h_rt={args.walltime}
#$ -r y
#$ -j y
#$ -o {d}/logs/genz.$JOB_ID.log
#$ -S /bin/bash
set -u
set -o pipefail   # FFX exit status survives the log filter below
# ffxc defaults to a 4G heap unless FFX_MEMORY is set (the old "-m 22G" only
# applied to the background Scheduler) — the likely cause of the OOM failures.
export FFX_MEMORY={mem}
cd {d}
echo "=== genz {pdb_id} start $(date) on $(hostname) ==="

# 1. coarse minimise (once)
if [ ! -f {pdb_id}_input.pdb_2 ]; then
  {FFX_MIN} Minimize -e 0.8 -Dpj.nt={nt} {pdb_id}_input.pdb -Dkey=ffx.properties || exit 1
fi

# 2. ONE many-body energy expansion at the reference pH (self + pair energies of
#    every rotamer / protonation state in the region).  Resumable: the restart
#    file is reused and each attempt keeps its own log (04c merges them).
mkdir -p mb && cd mb
[ -f {pdb_id}.pdb ] || cp ../{pdb_id}_input.pdb_2 {pdb_id}.pdb
[ -f ffx.properties ] || cp ../titrate.properties ffx.properties
if [ ! -f DONE ]; then
  RST=""; [ -s {pdb_id}.restart ] && RST="--eR {pdb_id}.restart"
  n=$(ls mb.log.* 2>/dev/null | wc -l)
  s=$(date +%s)
  {FFX_GENZ} ManyBody -Dpj.nt={nt} --tR --pH {REF_PH:g} --kPH 0.0 {three} \
      --lR {lr} $RST {pdb_id}.pdb -Dkey=ffx.properties > mb.log.$n 2>&1 \
      && grep -qE "Time for 2-body energies|Loaded 2-body energies" mb.log.$n && touch DONE
  echo "ManyBody wall $(( $(date +%s) - s )) s"
fi

# 3. Titration curves for every pH by replica-exchange Monte Carlo on that
#    expansion (seconds; see 04c_mc_titration.py)
if [ -f DONE ] && [ ! -f mc_pka.csv ]; then
  set +u; source {ENV_ACTIVATE}; set -u
  OMP_NUM_THREADS=1 python {PIPELINE_ROOT}/04c_mc_titration.py --restart {pdb_id}.restart \
      --log mb.log.* --lr {lr} --ph-ref {REF_PH:g} --pdb-id {pdb_id} --out mc_pka.csv
fi
cd ..

# 4. final minimise of the many-body optimum with saved multipoles (once)
if [ ! -f final/DONE ] && [ -f mb/DONE ]; then
  mkdir -p final && cd final
  if ls ../mb/{pdb_id}.pdb_* >/dev/null 2>&1; then cp $(ls -t ../mb/{pdb_id}.pdb_* | head -1) {pdb_id}_ref.pdb
  else cp ../{pdb_id}_input.pdb_2 {pdb_id}_ref.pdb; fi
  cp ../ffx.properties {pdb_id}_ref.properties
  {FFX_MIN} Minimize -e 0.1 -Dpj.nt={nt} {pdb_id}_ref.pdb -Dkey={pdb_id}_ref.properties \
      --saveInduced --savePermanentMoments && touch DONE
  cd ..
fi
echo "=== genz {pdb_id} end $(date) ==="
"""
    if args.premin:
        # PDBFixer-built loops (and some deposited models, e.g. 4HHB) contain
        # sub-1 A contacts; the mutual-polarization SCF then diverges on the very
        # first energy and Minimize loops forever on "Maximum SCF iterations
        # reached".  Relieve the clashes with polarization off (no SCF), then
        # run the normal mutual-polarization minimisation from that structure.
        a = f"  {FFX_MIN} Minimize -e 0.8 -Dpj.nt={nt} {pdb_id}_input.pdb -Dkey=ffx.properties || exit 1\n"
        assert a in script
        b = (f"  cp {pdb_id}_input.pdb {pdb_id}_pre.pdb\n"
             f"  {FFX_MIN} Minimize -e 1.0 -Dpj.nt={nt} -Dpolarization=none {pdb_id}_pre.pdb -Dkey=ffx.properties || exit 1\n"
             f"  cp {pdb_id}_pre.pdb_2 {pdb_id}_relax.pdb\n"
             f"  {FFX_MIN} Minimize -e 0.8 -Dpj.nt={nt} {pdb_id}_relax.pdb -Dkey=ffx.properties || exit 1\n"
             f"  cp {pdb_id}_relax.pdb_2 {pdb_id}_input.pdb_2\n")
        script = script.replace(a, b)
    if args.minimize_only:
        # corrected-structure rebuild: coarse minimise → final minimise with saved
        # induced dipoles / permanent multipoles; no titration physics
        i, j = script.index("# 2. ONE many-body"), script.index("# 4. final minimise")
        script = (script[:i] + script[j:]).replace("[ -f mb/DONE ]", "true") \
            .replace("if ls ../mb/", "if false && ls ../mb/").replace(f"-N genz_{pdb_id}", f"-N min_{pdb_id}")
    return script


def queued_names() -> set[str]:
    try:
        out = subprocess.run(["qstat", "-r"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return set()
    return {l.split(":", 1)[1].strip() for l in out.splitlines() if "Full jobname:" in l}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--queue", default="all.q")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--walltime", default="72:00:00")
    ap.add_argument("--ensemble", type=float, default=4.0,
                    help="GenZpKa --rEE window (kcal/mol); 0 keeps only the GMEC (fractions become 0/1)")
    ap.add_argument("--minimize-only", action="store_true",
                    help="skip ManyBody/MC: only minimise and save multipoles into final/")
    ap.add_argument("--large-queue", default="UI-HM",
                    help="queue for systems above LARGE_ATOMS heavy atoms (not MS: jobs there are killed)")
    ap.add_argument("--premin", action="store_true",
                    help="relieve clashes with a polarization-off minimisation before the "
                         "mutual-polarization one (for inputs whose SCF diverges at step 0)")
    ap.add_argument("--titr-radius", type=float, default=TITR_RADIUS)
    ap.add_argument("--pack-radius", type=float, default=4.0,
                    help="packing shell around titrating residues (Å); 0 = titratable residues only")
    ap.add_argument("--three-body", dest="three_body", action="store_true",
                    help="Include 3-body energies (-T). Off by default: for 4PTI the region gave\n                    "                    "335k triples (~22 h at 20 threads) worth ~1e-4 kcal/mol each")
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    man["pdb_id"] = man["pdb_id"].str.upper()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    active = set() if args.dry else queued_names()
    rows = []
    for d in sorted(p for p in PDB_DIR.iterdir() if p.is_dir()):
        pdb_id = d.name.upper()
        if args.only and pdb_id not in {x.upper() for x in args.only}:
            continue
        inp = d / f"{pdb_id}_input.pdb"
        if not inp.exists():
            continue
        g = man[man.pdb_id == pdb_id]
        labelled = {(str(c).strip() or "A", str(int(n))) for c, n in zip(g.chain, g.res_id)}
        residues, ss = read_residues(inp)
        titr, opt = ([], []) if args.minimize_only else \
            select_residues(residues, labelled, ss, args.titr_radius, args.pack_radius)
        n_heavy = sum(len(r["all"]) for r in residues)
        missing = labelled - {(r["chain"], r["num"]) for r in residues}
        (d / "ffx.properties").write_text(_PROPS.format(prm=SOLUTE_PRM))
        (d / "titrate.properties").write_text(_PROPS.format(prm=SOLUTE_PRM) + "solute-dielectric 2.0\n")
        (d / "logs").mkdir(exist_ok=True)
        (d / "genz_selection.csv").write_text(
            "chain,num,name,titrating\n" + "".join(
                f"{r['chain']},{r['num']},{r['name']},{int(r in titr)}\n" for r in opt))
        script = JOBS_DIR / f"genz_{pdb_id}.job"
        script.write_text(job_script(pdb_id, d, lr_string(opt), n_heavy > LARGE_ATOMS, args))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        done = (d / "final/DONE").exists()
        status = "done" if done else ("queued" if f"{'min' if args.minimize_only else 'genz'}_{pdb_id}" in active else "dry" if args.dry else "submit")
        if status == "submit":
            r = subprocess.run(["qsub", str(script)], capture_output=True, text=True, cwd=d)
            status = r.stdout.strip() or r.stderr.strip()
        rows.append(dict(pdb=pdb_id, heavy_atoms=n_heavy, labelled=len(labelled), titrating=len(titr),
                         optimised=len(opt), of_residues=len(residues),
                         labelled_missing=";".join(f"{c}{n}" for c, n in sorted(missing)), status=status))
        print(f"  {pdb_id}: {len(titr)} titrating / {len(opt)} optimised of {len(residues)} residues"
              f"{'  [LARGE]' if n_heavy > LARGE_ATOMS else ''}  {status}"
              + (f"  (labelled not found: {rows[-1]['labelled_missing']})" if missing else ""))
    pd.DataFrame(rows).to_csv(JOBS_DIR / "genz_jobs_summary.csv", index=False)
    print(f"\nScripts → {JOBS_DIR}   summary → {JOBS_DIR / 'genz_jobs_summary.csv'}")


if __name__ == "__main__":
    main()
