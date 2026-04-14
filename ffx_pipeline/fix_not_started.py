"""
fix_not_started.py

For every per-protein directory that has not started FFX minimisation
(contains only <ID>.properties and/or <ID>_input.pdb, no _coarse/_rotamer/_final):

  1. Re-run PDBFixer (via fix_one from 02_fix_structures.py) on the raw PDB
     and write the cleaned structure to  data/fixed_pdbs/<ID>/<ID>_input.pdb

  2. Update  data/sge_jobs/<ID>_minimize.job  so every path uses the
     per-protein subdirectory layout and the new file-naming convention:
       _fixed.pdb      → <ID>/<ID>_input.pdb
       _fixed_min.pdb  → <ID>/<ID>_coarse.pdb
       _fixed_min_2.pdb→ <ID>/<ID>_rotamer.pdb
       _s1a / _s1b     → <ID>/<ID>_s1a / _s1b  (same names, subdir changes)
       _final.pdb      → <ID>/<ID>_final.pdb
       PROPS key path  → data/fixed_pdbs/<ID>/<ID>.properties

Usage:
    python fix_not_started.py [--dry-run] [--skip-fix]

  --dry-run   Print what would be done without writing anything.
  --skip-fix  Skip the PDBFixer step (only update job files).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
FIXED_DIR = Path("data/fixed_pdbs")
RAW_DIR   = Path("data/raw_pdbs")
JOBS_DIR  = Path("data/sge_jobs")
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_not_started(subdir: Path) -> bool:
    """True if the per-protein subdir has no coarse/rotamer/final outputs."""
    names = {f.name for f in subdir.iterdir() if f.is_file()}
    progress_markers = ("_coarse.pdb", "_rotamer.pdb", "_final.uind", "_final.pdb")
    return not any(any(n.endswith(m) for m in progress_markers) for n in names)


def collect_not_started() -> list[str]:
    ids: list[str] = []
    for d in sorted(FIXED_DIR.iterdir()):
        if d.is_dir() and is_not_started(d):
            ids.append(d.name.upper())
    return ids


def run_pdbfixer(pdb_id: str, raw_path: Path, out_path: Path, dry_run: bool) -> bool:
    """Call fix_one from 02_fix_structures.py.  Returns True on success."""
    if dry_run:
        print(f"  [DRY-RUN] Would run PDBFixer: {raw_path} → {out_path}")
        return True

    # ── lazy-load fix_one to defer the pdbfixer import ────────────────────
    script = Path(__file__).parent / "02_fix_structures.py"
    spec = importlib.util.spec_from_file_location("_fix_structures", script)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:
        print(f"  ERROR loading 02_fix_structures.py: {exc}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = mod.fix_one(pdb_id, str(raw_path), str(out_path))
    status = log.get("status", "?")
    notes  = log.get("notes") or ""
    sym    = "✓" if status == "ok" else "✗"
    print(f"  [{sym}] PDBFixer status={status}  {notes}")
    return status == "ok"


def update_job_file(job_path: Path, pdb_id: str, dry_run: bool) -> None:
    """Rewrite the job file so all paths use the per-protein subdirectory layout."""
    if not job_path.exists():
        print(f"  WARNING: job file not found: {job_path}")
        return

    content = job_path.read_text()
    original = content

    # ── Step 1: Redirect every ${BASE}/<ID>_ reference into <ID>/<ID>_ ──────
    # This covers _fixed.pdb, _s1a, _s1b, _fixed_min, _fixed_min_2, _final,
    # AND any _<name>.pdb_2 suffixes that FFX appends, plus error-check lines.
    old_prefix = f"${{BASE}}/{pdb_id}_"
    new_prefix = f"${{BASE}}/{pdb_id}/{pdb_id}_"
    content = content.replace(old_prefix, new_prefix)

    # ── Step 2: Rename file stems (order matters — most-specific first) ────
    # _fixed_min_2.pdb → _rotamer.pdb
    content = content.replace(
        f"{pdb_id}_fixed_min_2.pdb",
        f"{pdb_id}_rotamer.pdb",
    )
    # _fixed_min.pdb → _coarse.pdb
    content = content.replace(
        f"{pdb_id}_fixed_min.pdb",
        f"{pdb_id}_coarse.pdb",
    )
    # _fixed.pdb → _input.pdb
    content = content.replace(
        f"{pdb_id}_fixed.pdb",
        f"{pdb_id}_input.pdb",
    )

    # ── Step 3: Fix PROPS key path ───────────────────────────────────────────
    content = content.replace(
        f"-Dkey=data/fixed_pdbs/{pdb_id}.properties",
        f"-Dkey=data/fixed_pdbs/{pdb_id}/{pdb_id}.properties",
    )

    if content == original:
        print(f"  [no change] {job_path.name}")
        return

    if dry_run:
        print(f"  [DRY-RUN] Would update: {job_path.name}")
        # Show first differing line for verification
        for i, (old, new) in enumerate(
            zip(original.splitlines(), content.splitlines()), 1
        ):
            if old != new:
                print(f"    line {i}: {old!r}")
                print(f"          → {new!r}")
        return

    job_path.write_text(content)
    print(f"  Updated:   {job_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Preview changes without writing files.")
    parser.add_argument("--skip-fix",  action="store_true",
                        help="Only update job files; skip PDBFixer step.")
    args = parser.parse_args()

    # ── Chdir to workspace root so relative paths work ───────────────────────
    os.chdir(Path(__file__).parent)

    not_started = collect_not_started()
    if not not_started:
        print("No not-started directories found.")
        return

    print(f"Found {len(not_started)} not-started director{'y' if len(not_started)==1 else 'ies'}:")
    print(" ", " ".join(not_started), "\n")

    for pdb_id in not_started:
        print(f"--- {pdb_id} ---")

        raw_path  = RAW_DIR   / f"{pdb_id}.pdb"
        subdir    = FIXED_DIR / pdb_id
        input_pdb = subdir    / f"{pdb_id}_input.pdb"
        job_path  = JOBS_DIR  / f"{pdb_id}_minimize.job"

        # ── 1. PDBFixer ───────────────────────────────────────────────────────
        if args.skip_fix:
            print("  [skip-fix] Skipping PDBFixer step.")
        elif not raw_path.exists():
            print(f"  WARNING: raw PDB not found ({raw_path}), skipping fix.")
        else:
            run_pdbfixer(pdb_id, raw_path, input_pdb, args.dry_run)

        # ── 2. Job file ───────────────────────────────────────────────────────
        update_job_file(job_path, pdb_id, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
