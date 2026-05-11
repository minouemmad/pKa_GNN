#!/usr/bin/env python3
"""
run_ablation_sweep.py — Drive 40 training runs that compare the contribution
of induced dipoles (.uind) and permanent multipoles (.uperm) for both rotopt
and titrate (FiLM) conditioning, across all five radii.

Conditions per mode:
    full    — all features (induced dipoles + permanent multipoles)
    noDip   — Dipole_X/Y/Z zeroed
    noPerm  — Perm_*           zeroed
    noElec  — both Dipole_*  + Perm_* zeroed   (geometry/topology baseline)

Source pkl directories (built earlier with ablate_datasets.py):
    Graph_pKa/Features_{mode}{tag}/Datasets/data_list_{0..4}.pkl
where tag ∈ {"", "_noDip", "_noPerm", "_noElec"} and full == "".

For each (mode, condition, dataset_idx) the script invokes 07_train.py with
default hyper-parameters (matching the Sweep2 protocol) and writes results to
    Graph_pKa/Results/Comparison_2026/{mode}_{condition}/

The script is **idempotent**: a run is skipped if its
    predictions/dataset_{idx}_all_folds.csv
already exists.  This lets you Ctrl-C and resume safely.

Usage
-----
    python ffx_pipeline/run_ablation_sweep.py                # run everything
    python ffx_pipeline/run_ablation_sweep.py --mode rotopt  # one mode only
    python ffx_pipeline/run_ablation_sweep.py --dry-run      # just print plan
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
RESULTS_ROOT  = ROOT / "Graph_pKa" / "Results" / "Comparison_2026"
TRAIN_SCRIPT  = ROOT / "ffx_pipeline" / "07_train.py"
RADII         = [7, 8, 9, 10, 11]
DATASET_INDICES = list(range(len(RADII)))

# (mode, arch, group_split)
MODES = {
    "rotopt":  {"arch": "naive", "group_split": "none"},
    "titrate": {"arch": "film",  "group_split": "residue"},
}

# condition -> dataset-dir suffix for Features_{mode}{suffix}/Datasets
CONDITIONS = {
    "full":   "",
    "noDip":  "_noDip",
    "noPerm": "_noPerm",
    "noElec": "_noElec",
}

# Default HP (mirrors 07_train defaults, which in turn match Sweep2 protocol)
DEFAULT_HP = dict(hidden=48, heads=6, lr=0.005, dropout=0.3,
                  batch=16, patience=30, epochs=500, folds=10, seed=42)


def _outdir(mode: str, cond: str) -> Path:
    return RESULTS_ROOT / f"{mode}_{cond}"


def _dataset_dir(mode: str, cond: str) -> Path:
    return ROOT / "Graph_pKa" / f"Features_{mode}{CONDITIONS[cond]}" / "Datasets"


def _is_done(out_dir: Path, idx: int) -> bool:
    return (out_dir / "predictions" / f"dataset_{idx}_all_folds.csv").exists()


def build_jobs(modes: list[str], conditions: list[str], indices: list[int]):
    jobs = []
    for mode in modes:
        for cond in conditions:
            for idx in indices:
                jobs.append((mode, cond, idx))
    return jobs


def run_one(mode: str, cond: str, idx: int, log_dir: Path) -> tuple[bool, float]:
    """Run a single 07_train.py invocation; return (ok, elapsed_sec)."""
    out_dir = _outdir(mode, cond)
    ds_dir  = _dataset_dir(mode, cond)
    if _is_done(out_dir, idx):
        print(f"  [skip] {mode}/{cond} idx={idx} — predictions already present")
        return True, 0.0
    if not (ds_dir / f"data_list_{idx}.pkl").exists():
        print(f"  [MISS] {ds_dir}/data_list_{idx}.pkl not found — skip")
        return False, 0.0

    cfg = MODES[mode]
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--mode",         mode,
        "--dataset-dir",  str(ds_dir),
        "--results-dir",  str(out_dir),
        "--dataset",      str(idx),
        "--arch",         cfg["arch"],
        "--group-split",  cfg["group_split"],
        "--hidden",       str(DEFAULT_HP["hidden"]),
        "--heads",        str(DEFAULT_HP["heads"]),
        "--lr",           str(DEFAULT_HP["lr"]),
        "--dropout",      str(DEFAULT_HP["dropout"]),
        "--batch",        str(DEFAULT_HP["batch"]),
        "--patience",     str(DEFAULT_HP["patience"]),
        "--epochs",       str(DEFAULT_HP["epochs"]),
        "--folds",        str(DEFAULT_HP["folds"]),
        "--seed",         str(DEFAULT_HP["seed"]),
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{mode}_{cond}_idx{idx}.log"
    print(f"  [run]  {mode}/{cond} idx={idx} (radius {RADII[idx]} A) -> {out_dir}")
    print(f"         log: {log_path}")
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    ok = (proc.returncode == 0)
    status = "OK" if ok else f"FAIL(rc={proc.returncode})"
    print(f"         {status} in {elapsed/60:.1f} min")
    return ok, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode",      choices=list(MODES) + ["all"], default="all")
    ap.add_argument("--condition", choices=list(CONDITIONS) + ["all"], default="all")
    ap.add_argument("--dataset",   type=int, nargs="*", default=None,
                    help="Specific dataset indices (default: 0..4)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print plan without launching jobs")
    args = ap.parse_args()

    modes      = list(MODES) if args.mode == "all" else [args.mode]
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    indices    = args.dataset if args.dataset else DATASET_INDICES

    jobs = build_jobs(modes, conditions, indices)
    print(f"Plan: {len(jobs)} jobs")
    pending = [(m, c, i) for m, c, i in jobs if not _is_done(_outdir(m, c), i)]
    skipped = len(jobs) - len(pending)
    print(f"  already done: {skipped}")
    print(f"  pending:      {len(pending)}")
    for m, c, i in pending:
        print(f"    - {m}/{c} dataset_{i} (radius {RADII[i]} A)")
    if args.dry_run:
        return

    log_dir = RESULTS_ROOT / "_logs"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    master_log = RESULTS_ROOT / "_run_summary.csv"
    new_master = not master_log.exists()
    with master_log.open("a", encoding="utf-8") as fh:
        if new_master:
            fh.write("mode,condition,dataset_idx,radius_A,status,elapsed_sec\n")
        n_ok, n_fail, t_total = 0, 0, 0.0
        for k, (m, c, i) in enumerate(jobs, 1):
            print(f"\n[{k}/{len(jobs)}] {m}/{c} idx={i}")
            ok, elapsed = run_one(m, c, i, log_dir)
            t_total += elapsed
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            fh.write(f"{m},{c},{i},{RADII[i]},"
                     f"{'OK' if ok else 'FAIL'},{elapsed:.1f}\n")
            fh.flush()
        print(f"\nDone. ok={n_ok}  fail={n_fail}  total_time={t_total/3600:.2f} h")


if __name__ == "__main__":
    main()
