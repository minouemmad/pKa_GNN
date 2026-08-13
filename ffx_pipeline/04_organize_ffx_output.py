
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
PDB_DIR    = Path("data/fixed_pdbs")
REPORT_CSV = Path("data/organize_report.csv")

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def collect_protein_ids(root: Path) -> list[str]:
    """Collect every PDB ID present in the flat root directory.

    Identified by any file matching ``*_fixed.pdb`` (the canonical fixed
    structure produced by 02_fix_structures.py) *or* ``*.properties``
    (created by 03_run_ffx_minimize.py) that matches the 4-character PDB ID
    pattern.  Sub-directories are ignored so we can re-run safely.
    """
    ids: set[str] = set()

    import re
    _pdb_re = re.compile(r'^([A-Za-z0-9]{4,5})_fixed', re.IGNORECASE)

    for f in root.iterdir():
        if f.is_dir():
            continue
        stem = f.stem  # no extension

        # {PDB}_fixed  →  {PDB}  (e.g. 1ABC_fixed.pdb)
        if stem.endswith("_fixed"):
            ids.add(stem[:-6])
            continue

        # {PDB}_fixed_*  →  {PDB}  (e.g. 1ABC_fixed_min.pdb, 1ABC_fixed_min_2.pdb_2)
        m = _pdb_re.match(stem)
        if m:
            ids.add(m.group(1).upper())
            continue

        # Bare properties like 1ABC.properties
        if f.suffix == ".properties" and "_" not in stem:
            ids.add(stem)

    return sorted(ids)

# Mapping from flat filename → clean filename inside the per-protein subdir.
# The function returns (source_path, dest_path) or None if the source doesn't exist.

def _plan_for_protein(root: Path, pdb: str) -> dict[str, tuple[Path, Path]]:
    """Build a {label: (src, dst)} map for all files belonging to *pdb*.

    Labels (for reporting):
        properties, properties_min, properties_min2,
        input, coarse, rotamer,
        final_pdb, final_uind, final_restart,
        fallback_pdb, fallback_uind,     ← step-4 ran on step-1 output
        misc_*                            ← anything else matching {PDB}_*
    """
    d  = root          # source directory (flat)
    sd = root / pdb    # destination subdirectory

    plan: dict[str, tuple[Path, Path]] = {}

    def _add(label: str, src_name: str, dst_name: str) -> None:
        src = d / src_name
        if src.exists():
            plan[label] = (src, sd / dst_name)

    # Properties
    _add("properties",      f"{pdb}.properties",           f"{pdb}.properties")
    _add("properties_min",  f"{pdb}_min.properties",       f"{pdb}_min.properties")
    _add("properties_min2", f"{pdb}_min_2.properties",     f"{pdb}_min_2.properties")

    # Fixed (input) structure
    _add("input",           f"{pdb}_fixed.pdb",            f"{pdb}_input.pdb")

    _add("coarse",          f"{pdb}_fixed_min.pdb",        f"{pdb}_coarse.pdb")
    _add("rotamer",         f"{pdb}_fixed_min_2.pdb",      f"{pdb}_rotamer.pdb")
    _add("final_pdb",       f"{pdb}_fixed_min_2.pdb_2",    f"{pdb}_final.pdb")
    _add("final_uind",      f"{pdb}_fixed_min_2.uind",     f"{pdb}_final.uind")
    _add("final_restart",   f"{pdb}_fixed_min.restart",    f"{pdb}_final.restart")

    # These appear when ManyBody fails; step 4 consumes *_fixed_min.pdb and
    # produces *_fixed_min.pdb_2 / *_fixed_min.uind instead of the _2 variants.
    if "final_pdb" not in plan:
        _add("fallback_pdb",  f"{pdb}_fixed_min.pdb_2",   f"{pdb}_final.pdb")
    if "final_uind" not in plan:
        _add("fallback_uind", f"{pdb}_fixed_min.uind",    f"{pdb}_final.uind")

    # Catch-all for anything else starting with {pdb}_ (e.g. scheduler logs)
    already_accounted = {src for (src, _) in plan.values()}
    for f in sorted(d.glob(f"{pdb}_*")):
        if f not in already_accounted and f.is_file():
            plan[f"misc_{f.name}"] = (f, sd / f.name)

    return plan

def _status(plan: dict[str, tuple[Path, Path]]) -> str:
    """Derive completeness category from what files were found."""
    if "final_uind" in plan or "fallback_uind" in plan:
        return "complete"
    if "coarse" in plan or "rotamer" in plan or "final_pdb" in plan or "fallback_pdb" in plan:
        return "partial"
    return "not_started"

def organize(root: Path, dry_run: bool) -> None:
    if not root.is_dir():
        log.error(f"PDB directory not found: {root}")
        sys.exit(1)

    pdb_ids = collect_protein_ids(root)
    if not pdb_ids:
        log.error("No proteins detected in directory. Nothing to do.")
        sys.exit(0)

    log.info(f"Detected {len(pdb_ids)} protein IDs in {root}")
    if dry_run:
        log.info("DRY-RUN mode – no files will be moved.")

    report_rows: list[dict] = []
    counts = {"complete": 0, "partial": 0, "not_started": 0}
    total_moved = 0

    for pdb in pdb_ids:
        plan   = _plan_for_protein(root, pdb)
        status = _status(plan)
        counts[status] += 1
        files_moved = 0

        if not plan:
            log.debug(f"  {pdb}: no files found, skipping")
            continue

        subdir = root / pdb

        if not dry_run:
            subdir.mkdir(exist_ok=True)

        for label, (src, dst) in plan.items():
            if dry_run:
                log.info(f"  [dry] {src.name:50s}  →  {pdb}/{dst.name}")
            else:
                if dst.exists():
                    # Already organised (re-run safety)
                    log.debug(f"  {pdb}: {dst.name} already at destination, skipping")
                else:
                    shutil.move(str(src), str(dst))
                    log.debug(f"  {pdb}: {src.name} → {dst.name}")
                files_moved += 1

        total_moved += files_moved
        tag = "[dry]" if dry_run else f"+{files_moved} files"
        log.info(
            f"  {pdb:8s}  [{status:11s}]  {tag}"
        )

        report_rows.append(
            {
                "pdb_id":      pdb,
                "status":      status,
                "files_moved": files_moved if not dry_run else "?",
                "subdir":      str(subdir),
                "has_final_pdb":  int("final_pdb"  in plan or "fallback_pdb"  in plan),
                "has_final_uind": int("final_uind" in plan or "fallback_uind" in plan),
                "fallback_path":  int("fallback_pdb" in plan),
            }
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=== Summary ===")
    log.info(f"  complete    : {counts['complete']:4d}  (step-4 finished, ready for 04_prepare_features.py)")
    log.info(f"  partial     : {counts['partial']:4d}  (minimization not fully complete)")
    log.info(f"  not_started : {counts['not_started']:4d}  (only input/properties present)")
    if not dry_run:
        log.info(f"  total files moved: {total_moved}")

    # ── Write CSV report ──────────────────────────────────────────────────────
    if not dry_run and report_rows:
        REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        log.info(f"  Report written to {REPORT_CSV}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganise data/fixed_pdbs/ into per-protein subdirectories."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be moved without making any changes."
    )
    parser.add_argument(
        "--dir", default=str(PDB_DIR),
        help=f"Path to the flat fixed_pdbs directory (default: {PDB_DIR})"
    )
    args = parser.parse_args()

    organize(Path(args.dir), dry_run=args.dry_run)

if __name__ == "__main__":
    main()
