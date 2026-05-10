"""
04_prepare_features.py

Generates GNN-ready node feature vectors and adjacency matrices from
FFX-minimized PDB structures and AMOEBA induced-dipole (.uind) files.

This replaces the Tinker-based pipeline (Tinker_EM.py + Tinker_Output_Processing.py)
and works directly with FFX PDB output from 03_run_ffx_minimize.py.

Inputs (per protein, inside data/fixed_pdbs/{PDB}/):
    {PDB}_final.pdb   – final FFX geometry (step 4 minimize output).
    {PDB}_final.uind  – AMOEBA induced dipoles from --saveInduced (step 4).
    Run 05_organize_ffx_output.py first to create this layout.

Reference data:
    data/manifest.csv        – PKAD-R residue-level pKa labels built by 01_parse_and_download.py.

Outputs (inside Graph_pKa/Features/):
    Node_Feature_Vectors/{radius}/{PDB}_{chain}_{resid}.{ResName}.csv
    Adjacency_Matrices/With_Self_Loop/{PDB}_{chain}_{resid}.{ResName}_adjacency.csv

Pipeline steps (per protein):
    1.  Parse PDB atom records → DataFrame
    2.  Parse .uind induced-dipole file → merge with atoms by serial number
    3.  Classify each atom as backbone (BB) or sidechain (SC) → atom_label (0-9)
    4.  Compute local backbone frame per residue (CA/C/O) → recalculated x,y,z
    5.  Compute intra-residue adjacency matrix (distance cut-off)
    6.  Count backbone-type neighbor atoms within radii 7-11 Å (MDAnalysis)
    7.  Compute H-bonds (Baker-Hubbard) and per-atom SASA (Shrake-Rupley) via mdtraj
    8.  After all proteins are processed: global MinMax normalisation of counts
    9.  One-hot encode residue type (ASP/GLU/HIS/LYS/CYS/TYR)
   10.  Write per-residue node-feature CSVs and adjacency-matrix CSVs

Atom-label scheme (10 classes, 0-9):
    0 = backbone N       5 = sidechain C
    1 = backbone CA      6 = sidechain N
    2 = backbone C       7 = sidechain H
    3 = backbone O       8 = sidechain O
    4 = backbone H       9 = sidechain S

NOTE: The existing create_data.py / Predict.py assume num_classes=9 (labels 0-8).
      With CYS (sidechain S → label 9), update those scripts to num_classes=10.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    import MDAnalysis as mda  # noqa: F401
    from MDAnalysis.analysis import distances as mda_distances
except ImportError:
    raise SystemExit(
        "MDAnalysis not found.\n"
        "Install with:  conda install -c conda-forge mdanalysis"
    )

try:
    import mdtraj as md
except ImportError:
    raise SystemExit(
        "mdtraj not found.\n"
        "Install with:  conda install -c conda-forge mdtraj"
    )

# ── Configuration ─────────────────────────────────────────────────────────────
PDB_DIR     = "data/fixed_pdbs"
RAW_PDB_DIR = "data/raw_pdbs"
MANIFEST    = "data/manifest.csv"
# Mode-aware feature directory; final value set in main() once --mode is parsed.
FEAT_DIR_DEFAULT = "Graph_pKa/Features"   # legacy fallback

RADII           = [7, 8, 9, 10, 11]
TARGET_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"}

# Default pH list for titration mode (mirrors 03_run_ffx_minimize.py / TITRATION_PHS)
DEFAULT_PHS = [3.94, 4.4, 6.45, 8.55]

# Three-letter to one-letter amino acid code (for sequence alignment).
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common alt names mapped to their canonical 1-letter
    "HID": "H", "HIE": "H", "HIP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K",
    "CYM": "C", "CYX": "C", "TYD": "Y",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _read_chain_full_sequence(pdb_path: "Path") -> dict[str, list[tuple[int, str]]]:
    """Return ``{chain -> [(resseq, resname_three_upper), ...]}`` listing every
    standard residue in the order it appears in the file (one entry per
    residue, anchored on CA atoms; first MODEL only; primary altloc only).
    """
    out: dict[str, list[tuple[int, str]]] = {}
    seen: set[tuple[str, int, str]] = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break  # only first model for NMR ensembles
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                altloc    = line[16:17]
                resname   = line[17:20].strip().upper()
                chain     = line[21:22]
                resseq    = int(line[22:26])
                icode     = line[26:27].strip()
            except (ValueError, IndexError):
                continue
            if atom_name != "CA":
                continue
            if altloc not in (" ", "", "A"):
                continue
            if resname not in _AA3_TO_1:
                continue
            # Use chain id as-is (preserve blank chain "" → " ")
            chain = chain if chain.strip() else " "
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            out.setdefault(chain, []).append((resseq, resname))
    return out


def _read_chain_ionizable_sequence(pdb_path: "Path") -> dict[str, list[tuple[int, str]]]:
    """Subset of ``_read_chain_full_sequence`` keeping only TARGET_RESIDUES.
    Kept for backward compatibility (used by external diagnostics).
    """
    full = _read_chain_full_sequence(pdb_path)
    return {ch: [(s, n) for s, n in seq if n in TARGET_RESIDUES]
            for ch, seq in full.items()}


def _needleman_wunsch_residue_map(
    raw_seq: list[tuple[int, str]],
    fix_seq: list[tuple[int, str]],
) -> tuple[dict[int, int], float]:
    """Global Needleman-Wunsch alignment between two residue sequences.
    Each sequence is a list of ``(resseq, resname_three)``.

    Returns ``(raw_resseq_to_fix_resseq, identity_fraction)`` where the
    mapping contains only positions where the residue types match (a residue
    type mismatch inside the alignment is conservatively dropped from the
    map).  ``identity_fraction`` is matches / max(len_raw, len_fix).

    Match=+2, mismatch=-1, gap=-2 (favours short gaps over long mismatches —
    PDBFixer typically inserts contiguous missing-loop residues).
    """
    n, m = len(raw_seq), len(fix_seq)
    if n == 0 or m == 0:
        return {}, 0.0

    MATCH, MISMATCH, GAP = 2, -1, -2
    # Score matrix
    H = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        H[i][0] = i * GAP
    for j in range(1, m + 1):
        H[0][j] = j * GAP
    raw_one = [_AA3_TO_1.get(rn, "X") for _, rn in raw_seq]
    fix_one = [_AA3_TO_1.get(rn, "X") for _, rn in fix_seq]
    for i in range(1, n + 1):
        ri = raw_one[i - 1]
        for j in range(1, m + 1):
            s = MATCH if ri == fix_one[j - 1] else MISMATCH
            H[i][j] = max(H[i - 1][j - 1] + s,
                          H[i - 1][j] + GAP,
                          H[i][j - 1] + GAP)

    # Traceback
    mapping: dict[int, int] = {}
    matches = 0
    i, j = n, m
    while i > 0 and j > 0:
        s = MATCH if raw_one[i - 1] == fix_one[j - 1] else MISMATCH
        if H[i][j] == H[i - 1][j - 1] + s:
            if raw_one[i - 1] == fix_one[j - 1]:
                mapping[raw_seq[i - 1][0]] = fix_seq[j - 1][0]
                matches += 1
            i -= 1
            j -= 1
        elif H[i][j] == H[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1
    identity = matches / max(n, m)
    return mapping, identity


def _match_chains(
    raw_chains: dict[str, list[tuple[int, str]]],
    fix_chains: dict[str, list[tuple[int, str]]],
    min_identity: float = 0.7,
) -> dict[str, tuple[str, dict[int, int]]]:
    """Find the best fixed-chain partner for every raw chain, allowing
    PDBFixer to have renamed the chain IDs (e.g. ``A,B → C,D`` or
    ``H,L → A,B``).  Greedy assignment by NW identity, requires
    ``identity >= min_identity`` (default 0.7).

    Returns ``{raw_chain_id -> (fix_chain_id, raw_resseq->fix_resseq mapping)}``.
    """
    # Score every (raw, fix) pair once.
    scores: list[tuple[float, str, str, dict[int, int]]] = []
    for r_id, r_seq in raw_chains.items():
        for f_id, f_seq in fix_chains.items():
            mp, ident = _needleman_wunsch_residue_map(r_seq, f_seq)
            if ident >= min_identity:
                scores.append((ident, r_id, f_id, mp))
    scores.sort(reverse=True)  # highest identity first

    used_raw: set[str] = set()
    used_fix: set[str] = set()
    out: dict[str, tuple[str, dict[int, int]]] = {}
    for ident, r_id, f_id, mp in scores:
        if r_id in used_raw or f_id in used_fix:
            continue
        used_raw.add(r_id)
        used_fix.add(f_id)
        out[r_id] = (f_id, mp)
    return out


def _build_pka_lookup_aligned(
    manifest: "pd.DataFrame",
    fixed_pdb_dir: "Path",
    raw_pdb_dir: "Path",
    log,
) -> dict[tuple[str, str, int, str], float]:
    """Build ``{(pdb, fixed_chain, fixed_resseq, resname) -> pka}`` by aligning
    raw↔fixed PDBs with Needleman-Wunsch on their full CA-residue sequences.

    PDBFixer (with ``keepIds=False``) may renumber and even rename chains, and
    can insert missing residues between existing ones.  Direct ordinal
    indexing breaks under any of those.  Instead we run NW per (raw_chain,
    fixed_chain) pair, pick the best partner per raw chain (greedy on
    identity, ≥0.7), and translate manifest ``res_id`` via the alignment.

    Manifest entries whose chain or residue can't be mapped are kept as a
    direct (raw_chain, raw_resseq) lookup; this gives the best possible
    chance of recovering the residue if the fixed PDB happens to share the
    same numbering (the common case when no missing residues were inserted).
    """
    fixed_pdb_dir = Path(fixed_pdb_dir)
    raw_pdb_dir   = Path(raw_pdb_dir)

    man = manifest.copy()
    man["pdb_id"]   = man["pdb_id"].astype(str).str.upper()
    man["chain"]    = man["chain"].astype(str).str.strip()
    man["res_id"]   = man["res_id"].astype(int)
    man["res_name"] = man["res_name"].astype(str).str.upper()
    man = man[man["res_name"].isin(TARGET_RESIDUES)]

    pka_lookup: dict[tuple[str, str, int, str], float] = {}
    n_direct = 0           # mapped to identical (chain, resseq)
    n_translated = 0       # chain/resseq changed via alignment
    n_fallback = 0         # no alignment available; manifest key used as-is
    n_unresolved = 0       # alignment present but residue not in mapping
    pdb_chain_renames = []
    pdb_chain_skipped = []

    for pdb_id, gm in man.groupby("pdb_id"):
        # Locate raw and fixed PDB files for this protein.
        raw_path = None
        for cand in (raw_pdb_dir / f"{pdb_id}.pdb",
                     raw_pdb_dir / f"{pdb_id.lower()}.pdb",
                     raw_pdb_dir / f"{pdb_id.upper()}.pdb"):
            if cand.exists():
                raw_path = cand
                break

        fix_dir = fixed_pdb_dir / pdb_id
        fixed_path = None
        if fix_dir.is_dir():
            cands  = [fix_dir / f"{pdb_id}_rot.pdb"]
            cands += sorted(fix_dir.glob(f"{pdb_id}_pH*.pdb_3"))
            cands += sorted(fix_dir.glob(f"{pdb_id}_pH*.pdb_2"))
            cands += sorted(fix_dir.glob(f"{pdb_id}_input.pdb"))
            for cand in cands:
                if cand.exists():
                    fixed_path = cand
                    break

        if raw_path is None or fixed_path is None:
            # No alignment possible — fall back to direct manifest numbering.
            for _, row in gm.iterrows():
                key = (pdb_id, row.chain, int(row.res_id), row.res_name)
                pka_lookup[key] = float(row.pka)
                n_fallback += 1
            continue

        raw_chains = _read_chain_full_sequence(raw_path)
        fix_chains = _read_chain_full_sequence(fixed_path)
        chain_map  = _match_chains(raw_chains, fix_chains)

        for chain, gm_chain in gm.groupby("chain"):
            partner = chain_map.get(chain)
            if partner is None:
                # Last-ditch fallback: same chain id present in fixed?
                if chain in fix_chains:
                    fix_chain = chain
                    raw_to_fix = {s: s for s, _ in raw_chains.get(chain, [])}
                else:
                    pdb_chain_skipped.append(f"{pdb_id}/{chain}")
                    for _, row in gm_chain.iterrows():
                        # Keep direct manifest key in case downstream lookup
                        # happens to match (it usually won't, but no worse
                        # than dropping).
                        key = (pdb_id, row.chain, int(row.res_id), row.res_name)
                        pka_lookup[key] = float(row.pka)
                        n_fallback += 1
                    continue
            else:
                fix_chain, raw_to_fix = partner
                if fix_chain != chain:
                    pdb_chain_renames.append(f"{pdb_id}: {chain}→{fix_chain}")

            for _, row in gm_chain.iterrows():
                rseq = int(row.res_id)
                fseq = raw_to_fix.get(rseq)
                if fseq is None:
                    # Residue not in alignment (e.g. raw has it but it was
                    # dropped by PDBFixer).  Still record the manifest key
                    # under the new chain id so the original numbering may
                    # match if no renumbering happened.
                    key = (pdb_id, fix_chain, rseq, row.res_name)
                    pka_lookup[key] = float(row.pka)
                    n_unresolved += 1
                    continue
                key = (pdb_id, fix_chain, int(fseq), row.res_name)
                pka_lookup[key] = float(row.pka)
                if fix_chain == chain and int(fseq) == rseq:
                    n_direct += 1
                else:
                    n_translated += 1

    if pdb_chain_renames:
        log.info(f"Chain renames detected: {', '.join(sorted(set(pdb_chain_renames))[:20])}"
                 + (" ..." if len(set(pdb_chain_renames)) > 20 else ""))
    if pdb_chain_skipped:
        log.warning(f"Chains with no alignment partner (kept manifest keys as-is): "
                    f"{', '.join(sorted(set(pdb_chain_skipped))[:20])}"
                    + (" ..." if len(set(pdb_chain_skipped)) > 20 else ""))
    log.info(
        f"pka_lookup built: {len(pka_lookup)} entries "
        f"(direct={n_direct}, renumbered={n_translated}, "
        f"alignment-fallback={n_fallback}, unresolved-in-alignment={n_unresolved})."
    )
    return pka_lookup

# Atoms whose presence signals the protonated form of a titratable sidechain.
# Used to compute the per-residue `is_protonated` scalar feature.  Atom names follow
# AMOEBA / standard PDB conventions; the function checks ANY of these names.
_PROTONATION_DETECTOR = {
    # ASP protonated (ASH) carries an extra H on OD1 or OD2
    "ASP": ("HD1", "HD2"),
    # GLU protonated (GLH) carries an extra H on OE1 or OE2
    "GLU": ("HE1", "HE2"),
    # HIS doubly protonated (HIP) has both HD1 (on ND1) and HE2 (on NE2)
    "HIS": ("HD1", "HE2"),   # treated specially below (both required)
    # LYS NH3+ has three HZ; deprotonated LYN keeps two
    "LYS": ("HZ3",),
    # CYS thiol has HG; CYM does not
    "CYS": ("HG", "HG1"),
    # TYR phenol has HH; deprotonated TYD does not
    "TYR": ("HH",),
}

# Ordered list for 6-class residue one-hot encoding (matches paper exactly).
# Atoms from non-target residues (environment) get all-zero encoding.
TARGET_RESIDUE_OHE = ["ASP", "GLU", "HIS", "LYS", "CYS", "TYR"]

# Maximum covalent-bond length used to build intra-residue adjacency matrices.
# 1.9 Å covers C-S (~1.82 Å) in CYS and all other standard amino-acid bonds.
BOND_CUTOFF = 1.9

AMINO_ACID_3_TO_FULL: dict[str, str] = {
    "ALA": "Alanine",
    "ARG": "Arginine",
    "ASN": "Asparagine",
    "ASP": "Aspartate",
    "CYS": "Cysteine",
    "GLN": "Glutamine",
    "GLU": "Glutamate",
    "GLY": "Glycine",
    "HIS": "Histidine",
    "ILE": "Isoleucine",
    "LEU": "Leucine",
    "LYS": "Lysine",
    "MET": "Methionine",
    "PHE": "Phenylalanine",
    "PRO": "Proline",
    "SER": "Serine",
    "THR": "Threonine",
    "TRP": "Tryptophan",
    "TYR": "Tyrosine",
    "VAL": "Valine",
}

# Atoms that belong to the peptide backbone (by name)
BACKBONE_HEAVY = {"N", "CA", "C", "O"}
BACKBONE_H     = {"H", "HN", "H1", "H2", "H3", "HA", "HA2", "HA3"}

# atom_label assignment: (first_char_of_name_or_full_CA, BB_or_SC) → int
ATOM_LABEL_MAP: dict[tuple[str, str], int] = {
    ("N",  "BB"): 0,
    ("CA", "BB"): 1,
    ("C",  "BB"): 2,
    ("O",  "BB"): 3,
    ("H",  "BB"): 4,
    ("C",  "SC"): 5,
    ("N",  "SC"): 6,
    ("H",  "SC"): 7,
    ("O",  "SC"): 8,
    ("S",  "SC"): 9,
}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@contextlib.contextmanager
def _ensure_pdb_extension(path: str):
    """Yield a path that ends with '.pdb'.

    FFX titration outputs use suffixes like '.pdb_3' which MDAnalysis and
    mdtraj refuse to read because their format detection is extension-based.
    If *path* already ends with '.pdb' we yield it unchanged; otherwise we
    copy the file to a temp '.pdb' file and yield that, deleting it on exit.
    """
    p = Path(path)
    if p.suffix.lower() == ".pdb":
        yield str(p)
        return
    fd, tmp = tempfile.mkstemp(suffix=".pdb", prefix=f"{p.stem}_")
    os.close(fd)
    try:
        shutil.copyfile(p, tmp)
        yield tmp
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_pdb(path: str) -> pd.DataFrame:
    """Read ATOM/HETATM records from a PDB file into a tidy DataFrame.

    Returns columns: serial, name, resname, chain, resseq, x, y, z
    """
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            rows.append(
                dict(
                    serial  = int(line[6:11]),
                    name    = line[12:16].strip(),
                    resname = line[17:20].strip(),
                    chain   = line[21].strip() or "A",
                    resseq  = int(line[22:26]),
                    x       = float(line[30:38]),
                    y       = float(line[38:46]),
                    z       = float(line[46:54]),
                )
            )
    return pd.DataFrame(rows)


def parse_uind(path: str) -> dict[int, tuple[float, float, float]]:
    """Parse an FFX .uind induced-dipole file.

    The file format is:
        <header line>  (first token = atom count)
        <serial>  <atom_name>  <ux>  <uy>  <uz>  ...
    Returns {serial: (ux, uy, uz)}.
    """
    dipoles: dict[int, tuple[float, float, float]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < 5:
                continue
            try:
                serial = int(parts[0])
                ux, uy, uz = float(parts[2]), float(parts[3]), float(parts[4])
                dipoles[serial] = (ux, uy, uz)
            except (ValueError, IndexError):
                continue
    return dipoles


def parse_uperm(
    path: str,
) -> dict[int, tuple[float, float, float, float, float, float, float, float, float, float]]:
    """Parse an FFX .uperm permanent-multipole file.

    File format (written by SavePermanentMoments.groovy)::

        <nAtoms>  <assemblyName>
        <serial> <name>  <charge>  <dipX> <dipY> <dipZ>  <qXX> <qXY> <qYY> <qXZ> <qYZ> <qZZ>

    Returns {serial: (charge, dipX, dipY, dipZ, qXX, qXY, qYY, qXZ, qYZ, qZZ)}.
    """
    perms: dict[
        int,
        tuple[float, float, float, float, float, float, float, float, float, float],
    ] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < 12:   # serial + name + 10 multipole components
                continue
            try:
                serial = int(parts[0])
                components = tuple(float(parts[i]) for i in range(2, 12))
                perms[serial] = components  # type: ignore[assignment]
            except (ValueError, IndexError):
                continue
    return perms


# ════════════════════════════════════════════════════════════════════════════
# Atom classification and labelling
# ════════════════════════════════════════════════════════════════════════════

def classify_backbone_sidechain(atom_name: str) -> str:
    """Return 'BB' for backbone atoms, 'SC' for sidechain atoms."""
    if atom_name in BACKBONE_HEAVY or atom_name in BACKBONE_H:
        return "BB"
    return "SC"


def assign_atom_label(atom_name: str, bb_sc: str) -> int:
    """Return the 0-9 integer atom label.

    Uses the first character of the atom name, except 'CA' which is kept as-is
    to distinguish alpha-carbon (BB) from any other carbon starting with 'C'.
    Returns -1 for unmapped atoms.
    """
    key_char = "CA" if atom_name == "CA" else atom_name[0]
    return ATOM_LABEL_MAP.get((key_char, bb_sc), -1)


# ════════════════════════════════════════════════════════════════════════════
# Local backbone frame
# ════════════════════════════════════════════════════════════════════════════

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_local_frame(
    ca: np.ndarray, c: np.ndarray, o: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the residue-local orthonormal frame used for coordinate transforms.

    Axes:
        x  ← CA→C direction
        z  ← normal to the CA-C-O plane
        y  ← z × x  (right-handed)

    Returns (R [3×3 rotation matrix], origin [CA coordinates]).
    """
    x_axis = _normalize(c - ca)
    z_axis = _normalize(np.cross(c - ca, o - c))
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, ca


def apply_local_frame(
    xyz: np.ndarray, R: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Transform global coordinates to the local backbone frame."""
    return R.T @ (xyz - origin)


def compute_local_frame_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Add recalculated_x/y/z columns by transforming each atom into its
    residue's local backbone frame.

    If Dipole_X/Y/Z columns are present, also rotates them into the local
    frame (pure rotation, no translation) and overwrites those columns.
    This matches the paper's feature definition where dipole components are
    expressed along the local X/Y/Z axes.

    Groups atoms by (chain, resseq).  If a residue is missing CA, C, or O,
    the transformed coordinates/dipoles are left as NaN.
    """
    df = df.reset_index(drop=True)
    rx = np.full(len(df), np.nan)
    ry = np.full(len(df), np.nan)
    rz = np.full(len(df), np.nan)

    has_dipoles = all(c in df.columns for c in ("Dipole_X", "Dipole_Y", "Dipole_Z"))
    if has_dipoles:
        ldx = np.full(len(df), np.nan)
        ldy = np.full(len(df), np.nan)
        ldz = np.full(len(df), np.nan)

    for (chain, resseq), res_idx in df.groupby(["chain", "resseq"]).groups.items():
        res_rows = df.loc[res_idx]

        def _get_atom_xyz(name: str) -> np.ndarray | None:
            """Return xyz for the first occurrence of atom 'name' in this residue."""
            match = res_rows.loc[res_rows["name"] == name, ["x", "y", "z"]]
            if match.empty:
                return None
            return match.iloc[0].values.astype(float)

        ca_xyz = _get_atom_xyz("CA")
        c_xyz  = _get_atom_xyz("C")
        o_xyz  = _get_atom_xyz("O")

        if ca_xyz is None or c_xyz is None or o_xyz is None:
            continue

        R, origin = build_local_frame(ca_xyz, c_xyz, o_xyz)

        for pos in res_idx:
            row = df.loc[pos]
            lc = apply_local_frame(np.array([row.x, row.y, row.z]), R, origin)
            rx[pos] = lc[0]
            ry[pos] = lc[1]
            rz[pos] = lc[2]

            if has_dipoles:
                dv = np.array([row.Dipole_X, row.Dipole_Y, row.Dipole_Z])
                if not np.any(np.isnan(dv)):
                    # Pure rotation (no translation) — dipoles are vectors, not points
                    d_local = R.T @ dv
                    ldx[pos] = d_local[0]
                    ldy[pos] = d_local[1]
                    ldz[pos] = d_local[2]

    df["recalculated_x"] = rx
    df["recalculated_y"] = ry
    df["recalculated_z"] = rz

    if has_dipoles:
        # Overwrite global-frame dipoles with local-frame dipoles in place
        df["Dipole_X"] = ldx
        df["Dipole_Y"] = ldy
        df["Dipole_Z"] = ldz
        log.debug("  Dipole vectors rotated into local backbone frame.")

    return df


# ════════════════════════════════════════════════════════════════════════════
# Neighbourhood atom counts  (MDAnalysis)
# ════════════════════════════════════════════════════════════════════════════

def compute_neighbor_counts(pdb_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Count backbone-type heavy-atom neighbours within each radius for every atom.

    Atom types counted:
        N_Count    – backbone nitrogen (name == 'N')
        CA_C_Count – alpha carbon or carbonyl carbon (name in {'CA', 'C'})
        O_Count    – backbone oxygen (name == 'O')
        S_Count    – any sulphur (name == 'S' or starts with 'S')

    Own-residue atoms are excluded from all counts.

    Returns a DataFrame with columns: serial, Radius_{r}A_{type}_Count for r in RADII.
    """
    u = mda.Universe(pdb_path)
    heavy = u.select_atoms("not (name H* or name [0-9]H*)")

    coords  = heavy.positions                      # (N_heavy, 3)
    serials = heavy.indices + 1                    # MDAnalysis uses 0-based index
    resids  = heavy.resids
    names   = heavy.names

    # Pairwise distance matrix  (N_heavy × N_heavy)
    dist_mat = np.zeros((len(heavy), len(heavy)), dtype=np.float64)
    mda_distances.distance_array(coords, coords, result=dist_mat)

    records: list[dict] = []
    for i, atom in enumerate(heavy):
        serial = int(atom.index) + 1
        row: dict = {"serial": serial}
        other_res_mask = resids != resids[i]   # exclude own residue

        for r in RADII:
            in_sphere = (dist_mat[i] <= r) & other_res_mask
            nbr_names = names[in_sphere]
            row[f"Radius_{r}A_N_Count"]    = int(np.sum(nbr_names == "N"))
            row[f"Radius_{r}A_CA_C_Count"] = int(np.sum((nbr_names == "CA") | (nbr_names == "C")))
            row[f"Radius_{r}A_O_Count"]    = int(np.sum(nbr_names == "O"))
            row[f"Radius_{r}A_S_Count"]    = int(np.sum(
                np.char.startswith(nbr_names.astype("U10"), "S")
            ))
        records.append(row)

    nbr_df = pd.DataFrame(records)

    # Hydrogen atoms get zero counts (they were excluded from heavy-atom processing)
    h_serials = np.array([a.index + 1 for a in u.atoms if a not in heavy])
    if len(h_serials):
        h_rows = pd.DataFrame({"serial": h_serials})
        for r in RADII:
            for t in ["N_Count", "CA_C_Count", "O_Count", "S_Count"]:
                h_rows[f"Radius_{r}A_{t}"] = 0
        nbr_df = pd.concat([nbr_df, h_rows], ignore_index=True)

    return nbr_df


# ════════════════════════════════════════════════════════════════════════════
# H-bonds and SASA  (mdtraj)
# ════════════════════════════════════════════════════════════════════════════

def compute_hbonds_sasa(pdb_path: str, n_atoms: int) -> pd.DataFrame:
    """Compute Baker-Hubbard H-bond counts and Shrake-Rupley SASA per atom.

    Returns a DataFrame with columns:
        serial, Number of H-Bonds as donor, Number of H-Bonds as acceptor, SASA_Value
    """
    try:
        traj = md.load(pdb_path)

        hbonds          = md.baker_hubbard(traj, periodic=False)
        donor_counts    = defaultdict(int)
        acceptor_counts = defaultdict(int)
        for donor_idx, _h_idx, acceptor_idx in hbonds:
            donor_counts[donor_idx]    += 1
            acceptor_counts[acceptor_idx] += 1

        sasa = md.shrake_rupley(traj, mode="atom")[0]   # (n_atoms,)

        rows = [
            {
                "serial":                        atom.index + 1,
                "Number of H-Bonds as donor":    donor_counts[atom.index],
                "Number of H-Bonds as acceptor": acceptor_counts[atom.index],
                "SASA_Value":                    float(sasa[atom.index]),
            }
            for atom in traj.topology.atoms
        ]
        return pd.DataFrame(rows)

    except Exception as exc:
        log.warning(f"    H-bond/SASA failed ({exc}); returning zeros for all atoms.")
        return pd.DataFrame(
            {
                "serial":                        range(1, n_atoms + 1),
                "Number of H-Bonds as donor":    0,
                "Number of H-Bonds as acceptor": 0,
                "SASA_Value":                    0.0,
            }
        )


# ════════════════════════════════════════════════════════════════════════════
# Adjacency matrix
# ════════════════════════════════════════════════════════════════════════════

def build_adjacency_matrix(res_df: pd.DataFrame) -> pd.DataFrame:
    """Build a binary intra-residue adjacency matrix with self-loops.

    Edges are added between any two atoms within BOND_CUTOFF Å.  Self-loops
    are always included (diagonal = 1).

    Rows and columns are labelled by atom name.
    """
    coords = res_df[["x", "y", "z"]].values.astype(float)
    n      = len(res_df)
    adj    = np.eye(n, dtype=int)

    for i in range(n):
        diffs = coords[i] - coords
        dists = np.linalg.norm(diffs, axis=1)
        adj[i] = (dists <= BOND_CUTOFF).astype(int)

    names = res_df["name"].tolist()
    return pd.DataFrame(adj, index=names, columns=names)


def build_edge_features(res_df: pd.DataFrame) -> pd.DataFrame:
    """Build a table of directed edge features for every bond in the residue.

    For each directed edge i→j (including self-loops i→i) stores the
    displacement vector (dx, dy, dz) and scalar distance expressed in the
    residue's local backbone frame.  For self-loops all four values are 0.

    Coordinates used are the already-transformed `recalculated_x/y/z` columns
    (local-frame coords computed in compute_local_frame_coords), so no
    additional rotation is needed here.

    Returns a DataFrame with columns:
        atom_i, atom_j, dx, dy, dz, distance
    where atom_i / atom_j are 0-based positional indices into the residue
    atom list (matching the row/column order of the adjacency matrix).
    """
    # Prefer local-frame coords if available, fall back to global
    if "recalculated_x" in res_df.columns and not res_df["recalculated_x"].isna().all():
        coords = res_df[["recalculated_x", "recalculated_y", "recalculated_z"]].values.astype(float)
    else:
        coords = res_df[["x", "y", "z"]].values.astype(float)

    n = len(res_df)
    rows: list[dict] = []

    for i in range(n):
        for j in range(n):
            diff = coords[j] - coords[i]
            dist = float(np.linalg.norm(diff))
            # Include edge only if within bond cutoff OR it is a self-loop
            if i == j or dist <= BOND_CUTOFF:
                rows.append({
                    "atom_i":   i,
                    "atom_j":   j,
                    "dx":       float(diff[0]),
                    "dy":       float(diff[1]),
                    "dz":       float(diff[2]),
                    "distance": dist,
                })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Protonation-state detection
# ════════════════════════════════════════════════════════════════════════════

def detect_protonation(
    df: pd.DataFrame,
) -> dict[tuple[str, int], int]:
    """Return {(chain, resseq): is_protonated} for every titratable residue.

    Detection is by atom-name presence (AMOEBA-CpHMD writes canonical residue
    names; protonation state is encoded only by which polar hydrogens exist).
    HIS is special: requires *both* HD1 and HE2 to be considered fully
    protonated (HIP); HID/HIE both map to 0.
    Non-titratable residues are absent from the returned dict.
    """
    out: dict[tuple[str, int], int] = {}
    for (chain, resseq, resname), grp in df.groupby(["chain", "resseq", "resname"]):
        if resname not in _PROTONATION_DETECTOR:
            continue
        atoms_present = set(grp["name"].tolist())
        detector = _PROTONATION_DETECTOR[resname]
        if resname == "HIS":
            is_prot = int(("HD1" in atoms_present) and ("HE2" in atoms_present))
        else:
            is_prot = int(any(a in atoms_present for a in detector))
        out[(chain, int(resseq))] = is_prot
    return out


# ════════════════════════════════════════════════════════════════════════════
# Locate completed FFX jobs (mode-aware)
# ════════════════════════════════════════════════════════════════════════════

def _find_pdb_3(folder: Path, base: str) -> Path | None:
    """Find the most-advanced titration PDB for a given base stem.

    Looks for `{base}.pdb_3` first (preferred — full pipeline run including
    final Minimize after titration ManyBody), then falls back to `.pdb_4`
    (rare; observed when extra rounds were needed) or `.pdb_2` (titration
    ManyBody only, no final Minimize).
    """
    for ext in ("pdb_3", "pdb_4", "pdb_2"):
        p = folder / f"{base}.{ext}"
        if p.exists():
            return p
    return None


def find_completed_jobs(
    pdb_dir: str,
    mode: str,
    phs: list[float],
) -> list[tuple[str, str, str, str | None, float | None]]:
    """Return [(pdb_id, pdb_path, uind, uperm_or_None, pH_or_None), ...].

    mode == 'rotopt':
        Looks for {pdb}_rot.pdb + {pdb}_input.uind + {pdb}_input.uperm.
        pH is None.  One record per PDB.

    mode == 'titrate':
        For each pH p in *phs* looks for {pdb}_pH{p}.pdb_3 (or .pdb_4 / .pdb_2 fallback)
        + {pdb}_pH{p}.uind + {pdb}_pH{p}.uperm.
        Up to len(phs) records per PDB; missing pHs are silently skipped.

    Records missing the required uind file are dropped with a debug log.
    """
    d = Path(pdb_dir)
    jobs: list[tuple[str, str, str, str | None, float | None]] = []

    for folder in sorted(d.iterdir()):
        if not folder.is_dir():
            continue
        pdb_id = folder.name

        if mode == "rotopt":
            pdb_path  = folder / f"{pdb_id}_rot.pdb"
            uind      = folder / f"{pdb_id}_input.uind"
            uperm     = folder / f"{pdb_id}_input.uperm"
            if not pdb_path.exists() or not uind.exists():
                log.debug(f"  {pdb_id}: rotopt files missing — skipping")
                continue
            jobs.append((
                pdb_id, str(pdb_path), str(uind),
                str(uperm) if uperm.exists() else None,
                None,
            ))

        elif mode == "titrate":
            for ph in phs:
                ph_str = str(ph)
                base   = f"{pdb_id}_pH{ph_str}"
                pdb_path = _find_pdb_3(folder, base)
                uind     = folder / f"{base}.uind"
                uperm    = folder / f"{base}.uperm"
                if pdb_path is None or not uind.exists():
                    log.debug(f"  {pdb_id} pH {ph}: missing pdb_3 or .uind — skipping")
                    continue
                jobs.append((
                    pdb_id, str(pdb_path), str(uind),
                    str(uperm) if uperm.exists() else None,
                    float(ph),
                ))

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="FFX → GNN feature builder (mode-aware)")
    parser.add_argument("--mode", choices=["rotopt", "titrate"], default="titrate",
                        help="rotopt = single rotamer-optimised structure per PDB; "
                             "titrate = per-pH titration outputs (pdb_3 + .uind + .uperm)")
    parser.add_argument("--phs", type=float, nargs="+", default=DEFAULT_PHS,
                        help="pH values to look for in titrate mode (ignored in rotopt)")
    parser.add_argument("--pdb-dir", default=PDB_DIR,
                        help=f"Per-protein FFX output directory (default: {PDB_DIR})")
    parser.add_argument("--raw-pdb-dir", default=RAW_PDB_DIR,
                        help=f"Raw PDB directory used to align manifest residue "
                             f"numbering to the (possibly renumbered) fixed PDBs "
                             f"(default: {RAW_PDB_DIR})")
    parser.add_argument("--manifest", default=MANIFEST,
                        help=f"PKAD-R manifest CSV (default: {MANIFEST})")
    parser.add_argument("--feat-dir", default=None,
                        help="Override feature output directory "
                             "(default: Graph_pKa/Features_{mode})")
    args = parser.parse_args()

    mode  = args.mode
    phs   = list(args.phs) if mode == "titrate" else []
    feat_dir = args.feat_dir or f"Graph_pKa/Features_{mode}"
    node_dir = os.path.join(feat_dir, "Node_Feature_Vectors")
    adj_dir  = os.path.join(feat_dir, "Adjacency_Matrices/With_Self_Loop")
    bond_dir = os.path.join(feat_dir, "Edge_Features")

    log.info(f"Mode       : {mode}")
    if mode == "titrate":
        log.info(f"pH values  : {phs}")
    log.info(f"Feature dir: {feat_dir}")

    os.makedirs(adj_dir,  exist_ok=True)
    os.makedirs(bond_dir, exist_ok=True)
    for r in RADII:
        os.makedirs(os.path.join(node_dir, str(r)), exist_ok=True)

    # ── Load pKa labels ──────────────────────────────────────────────────────
    manifest = pd.read_csv(args.manifest)
    # Build {(PDB, chain, res_id, res_name) → pka}
    pka_lookup: dict[tuple[str, str, int, str], float] = {
        (
            str(row.pdb_id).upper(),
            str(row.chain).strip(),
            int(row.res_id),
            str(row.res_name).upper(),
        ): float(row.pka)
        for _, row in manifest.iterrows()
    }
    log.info(f"Loaded {len(pka_lookup)} pKa entries from {args.manifest}")

    # ── Discover completed FFX jobs ──────────────────────────────────────────
    jobs = find_completed_jobs(args.pdb_dir, mode, phs)
    if not jobs:
        log.error(f"No completed {mode} jobs found in {args.pdb_dir}.")
        return
    log.info(f"Found {len(jobs)} completed job record(s)")

    # ── Per-protein processing  (collect raw data for global normalisation) ──
    all_frames: list[pd.DataFrame] = []

    for pdb_id, pdb_path, uind_path, uperm_path, ph in jobs:
        ph_tag = f" pH {ph}" if ph is not None else ""
        log.info(f"Processing {pdb_id}{ph_tag}  ({Path(pdb_path).name})")

        # 1. Parse PDB
        df = parse_pdb(pdb_path)
        if df.empty:
            log.warning(f"  {pdb_id}: empty PDB – skipping")
            continue

        # 2. Add induced dipoles (by serial number)
        dipoles = parse_uind(uind_path)
        df["Dipole_X"] = df["serial"].map(lambda s: dipoles.get(s, (np.nan,)*3)[0])
        df["Dipole_Y"] = df["serial"].map(lambda s: dipoles.get(s, (np.nan,)*3)[1])
        df["Dipole_Z"] = df["serial"].map(lambda s: dipoles.get(s, (np.nan,)*3)[2])
        n_matched = df["serial"].isin(dipoles).sum()
        log.info(f"  Matched {n_matched}/{len(df)} atoms to dipole entries")

        # 2b. Add permanent multipoles (by serial number) — optional
        PERM_COLS = [
            "Perm_Charge",
            "Perm_DipX", "Perm_DipY", "Perm_DipZ",
            "Perm_QuadXX", "Perm_QuadXY", "Perm_QuadYY",
            "Perm_QuadXZ", "Perm_QuadYZ", "Perm_QuadZZ",
        ]
        if uperm_path is not None:
            perms = parse_uperm(uperm_path)
            for idx, col in enumerate(PERM_COLS):
                nan10 = (np.nan,) * 10
                df[col] = df["serial"].map(lambda s, i=idx: perms.get(s, nan10)[i])
            n_perm = df["serial"].isin(perms).sum()
            log.info(f"  Matched {n_perm}/{len(df)} atoms to permanent multipole entries")
        else:
            log.info(f"  No .uperm file for {pdb_id} — permanent multipole features will be NaN")
            for col in PERM_COLS:
                df[col] = np.nan

        # 3. Backbone / sidechain classification and atom_label
        df["bb_sc"]      = df["name"].apply(classify_backbone_sidechain)
        df["atom_label"] = df.apply(
            lambda r: assign_atom_label(r["name"], r["bb_sc"]), axis=1
        )

        # 4. Local frame coordinates
        df = compute_local_frame_coords(df)

        # 5. Neighbourhood atom counts
        log.info(f"  Computing neighbour counts…")
        try:
            with _ensure_pdb_extension(pdb_path) as pdb_for_tools:
                nbr_df = compute_neighbor_counts(pdb_for_tools, df)
            df = df.merge(nbr_df, on="serial", how="left")
        except Exception as exc:
            log.warning(f"  Neighbour counts failed ({exc}); inserting zeros.")
            for r in RADII:
                for t in ["N_Count", "CA_C_Count", "O_Count", "S_Count"]:
                    df[f"Radius_{r}A_{t}"] = 0

        # 6. H-bonds and SASA
        log.info(f"  Computing H-bonds and SASA…")
        with _ensure_pdb_extension(pdb_path) as pdb_for_tools:
            hb_df = compute_hbonds_sasa(pdb_for_tools, len(df))
        df = df.merge(hb_df, on="serial", how="left")

        # 6b. Per-residue protonation state (broadcast to every atom in residue)
        prot_map = detect_protonation(df)
        df["is_protonated"] = df.apply(
            lambda r: prot_map.get((r["chain"], int(r["resseq"])), 0),
            axis=1,
        ).astype(float)

        # 6c. pH column — broadcast scalar to every atom (NaN in rotopt mode)
        df["pH"] = float(ph) if ph is not None else np.nan

        # 7. Tag protein
        df["pdb_id"] = pdb_id.upper()
        df["_ph"]    = ph         # internal grouping key (None for rotopt)

        all_frames.append(df)
        log.info(f"  {pdb_id}: {len(df)} atoms processed")

    if not all_frames:
        log.error("No data processed.  Exiting.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    log.info(f"Total atoms across all proteins: {len(full_df)}")

    # ── Global MinMax normalisation of neighbour counts ──────────────────────
    radius_feat_cols = [
        f"Radius_{r}A_{t}"
        for r in RADII
        for t in ["N_Count", "CA_C_Count", "O_Count", "S_Count"]
    ]
    avail_cols = [c for c in radius_feat_cols if c in full_df.columns]
    if avail_cols:
        scaler = MinMaxScaler()
        full_df[avail_cols] = scaler.fit_transform(full_df[avail_cols].fillna(0))
        log.info("Applied global MinMax normalisation to neighbour-count features.")

    # ── One-hot encode residue name (6-class: target residues only) ─────────
    # Only the 6 titratable residue types get a 1; all other residues (environment
    # atoms included in the sphere) get all-zeros — consistent with the paper.
    residue_ohe_cols: list[str] = []
    for resname_3 in TARGET_RESIDUE_OHE:
        full_name = AMINO_ACID_3_TO_FULL[resname_3]
        col = f"Residue Name_{full_name}"
        full_df[col] = (full_df["resname"] == resname_3).astype(float)
        residue_ohe_cols.append(col)

    # ── Generate per-residue outputs ─────────────────────────────────────────
    PERM_COLS = [
        "Perm_Charge",
        "Perm_DipX", "Perm_DipY", "Perm_DipZ",
        "Perm_QuadXX", "Perm_QuadXY", "Perm_QuadYY",
        "Perm_QuadXZ", "Perm_QuadYZ", "Perm_QuadZZ",
    ]
    base_feature_cols = residue_ohe_cols + [
        "atom_label",
        "recalculated_x",
        "recalculated_y",
        "recalculated_z",
        "Dipole_X",
        "Dipole_Y",
        "Dipole_Z",
    ] + PERM_COLS + [
        "Number of H-Bonds as donor",
        "Number of H-Bonds as acceptor",
        "SASA_Value",
        "is_protonated",
        "pH",
    ]

    n_adj_saved  = 0
    n_feat_saved = 0
    n_skipped    = 0

    # Group by (pdb, ph, chain, resseq, resname) so titrate mode produces one set
    # of files per (residue, pH).  In rotopt mode `_ph` is None and behaves like
    # a single-bucket group key.
    for (pdb_id, ph_key, chain, resseq, resname), res_df in full_df.groupby(
        ["pdb_id", "_ph", "chain", "resseq", "resname"], sort=False, dropna=False
    ):
        if resname not in TARGET_RESIDUES:
            continue

        key = (pdb_id.upper(), str(chain), int(resseq), resname.upper())
        pka = pka_lookup.get(key)
        if pka is None:
            n_skipped += 1
            continue

        full_name = AMINO_ACID_3_TO_FULL.get(resname, resname)
        if ph_key is None or (isinstance(ph_key, float) and np.isnan(ph_key)):
            stem = f"{pdb_id}_{chain}_{resseq}.{full_name}"
        else:
            # Per-pH stem suffix; double-underscore separator is parsed by
            # 06_create_datasets.py to recover the pH value.
            stem = f"{pdb_id}_{chain}_{resseq}.{full_name}__pH{ph_key}"

        # Adjacency matrix (uses original x/y/z, not local-frame coords)
        adj_df   = build_adjacency_matrix(res_df)
        adj_path = os.path.join(adj_dir, f"{stem}_adjacency.csv")
        adj_df.to_csv(adj_path)
        n_adj_saved += 1

        # Edge features (local-frame displacement vectors + distance)
        bond_df   = build_edge_features(res_df)
        bond_path = os.path.join(bond_dir, f"{stem}_bonds.csv")
        bond_df.to_csv(bond_path, index=False)

        # Node feature vector  — one file per radius
        for radius in RADII:
            radius_cols = [
                f"Radius_{radius}A_N_Count",
                f"Radius_{radius}A_CA_C_Count",
                f"Radius_{radius}A_O_Count",
                f"Radius_{radius}A_S_Count",
            ]
            wanted   = base_feature_cols + radius_cols
            avail    = [c for c in wanted if c in res_df.columns]
            feat_df  = res_df[avail].copy()
            feat_df["Expt. pKa"] = pka

            out_path = os.path.join(node_dir, str(radius), f"{stem}.csv")
            feat_df.to_csv(out_path, index=False)
            n_feat_saved += 1

    log.info(f"Done.")
    log.info(f"  Adjacency matrices saved : {n_adj_saved}  → {adj_dir}")
    log.info(f"  Edge feature files saved : {n_adj_saved}  → {bond_dir}")
    log.info(f"  Node feature files saved : {n_feat_saved} → {node_dir}")
    log.info(f"  Residues skipped (no pKa in manifest): {n_skipped}")


if __name__ == "__main__":
    main()
