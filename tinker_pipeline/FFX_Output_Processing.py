#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths (relative to this file's location,  i.e.  Graph_pKa/)
BASE = Path(__file__).parent.resolve()

DATA_FIXED_PDB    = BASE / "Data" / "0_Fixed_PDB"
DATA_FINAL_PDB    = BASE / "Data" / "1_FFX_Final_PDB"
DATA_UIND         = BASE / "Data" / "2_Uind_Files"
DATA_MPOLES       = BASE / "Data" / "3_Mpoles_Files"
FEATURES_DIR      = BASE / "Features" / "Per_Protein"
NODE_FEAT_ROOT    = BASE / "Features" / "Node_Feature_Vectors"
ADJ_MATRIX_DIR    = BASE / "Features" / "Adjacency_Matrices" / "With_Self_Loop"
PARAM_FILE        = BASE / "Tinker_params" / "amoebabio18.prm"
PKAD_CSV          = BASE / "1-PKAD-R-2025-09-03.csv"

# Titratable residue 3-letter codes (upper-case)
TITRATABLE_RES = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR", "ARG"}

# Neighbourhood radii to build features for (Angstroms)
RADII = [7, 8, 9, 10, 11]

# One-hot columns for amino acid residue name
RESIDUE_ORDER = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
]
RESIDUE_TO_INDEX = {r: i for i, r in enumerate(RESIDUE_ORDER)}

AMINO_ACID_3_TO_FULL = {
    "ALA": "Alanine", "ARG": "Arginine", "ASN": "Asparagine", "ASP": "Aspartate",
    "CYS": "Cysteine", "GLU": "Glutamate", "GLN": "Glutamine", "GLY": "Glycine",
    "HIS": "Histidine", "ILE": "Isoleucine", "LEU": "Leucine", "LYS": "Lysine",
    "MET": "Methionine", "PHE": "Phenylalanine", "PRO": "Proline", "SER": "Serine",
    "THR": "Threonine", "TRP": "Tryptophan", "TYR": "Tyrosine", "VAL": "Valine",
}

# Step 0 – Build biotype lookup table from amoebabio18.prm

def load_biotype_table(param_file: Path = PARAM_FILE) -> dict[tuple[str, str], int]:
    """Parse biotype lines in the AMOEBA .prm file.

    Returns
    -------
    dict mapping (residue_full_name_lower, atom_name_upper) → AMOEBA atom-type int.

    Example:
        "biotype  7  N  "Alanine"  7"  →  ("alanine", "N") → 7
    """
    table: dict[tuple[str, str], int] = {}
    pattern = re.compile(
        r'^biotype\s+\d+\s+(\S+)\s+"([^"]+)"\s+(\d+)', re.IGNORECASE
    )
    with open(param_file, "r") as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                atom_name = m.group(1).upper()
                residue   = m.group(2).lower()
                atype     = int(m.group(3))
                # Multiple entries for same (residue, atom_name) are possible
                # (e.g., terminal variants). Keep the first encountered (standard).
                key = (residue, atom_name)
                if key not in table:
                    table[key] = atype
    logger.info(f"Loaded {len(table)} biotype entries from {param_file.name}")
    return table

BIOTYPE_TABLE: dict[tuple[str, str], int] = {}   # populated lazily

def get_biotype_table() -> dict[tuple[str, str], int]:
    global BIOTYPE_TABLE
    if not BIOTYPE_TABLE:
        BIOTYPE_TABLE = load_biotype_table()
    return BIOTYPE_TABLE

def assign_amoeba_type(residue_3: str, atom_name: str) -> int | None:
    """Return AMOEBA atom type for (residue 3-letter, atom_name), or None."""
    table = get_biotype_table()
    full = AMINO_ACID_3_TO_FULL.get(residue_3.upper(), residue_3).lower()
    key = (full, atom_name.upper())
    return table.get(key)

# Step 1 – Collect FFX outputs from the cluster into local Data/ layout

def collect_ffx_outputs(
    cluster_output_dir: str,
    pdb_id_list: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Copy and rename FFX Step-4 outputs into the local Data/ directories.

    Parameters
    ----------
    cluster_output_dir : str
        Local path to the directory containing the SGE job output files
        (e.g., a rsync'd copy of the cluster fixed_pdbs folder).
    pdb_id_list : list[str] | None
        If given, only process these PDB IDs.  Otherwise, all IDs discovered.
    overwrite : bool
        Whether to overwrite existing files in Data/.

    The function searches for, per protein ID:
        {ID}_fixed_min_2.pdb_2   (preferred Step-4 output)
        {ID}_fixed_min.pdb_2     (fallback)
        {ID}_fixed_min_2.uind    (preferred)
        {ID}_fixed_min.uind      (fallback)
    """
    DATA_FINAL_PDB.mkdir(parents=True, exist_ok=True)
    DATA_UIND.mkdir(parents=True, exist_ok=True)
    DATA_MPOLES.mkdir(parents=True, exist_ok=True)

    cluster_dir = Path(cluster_output_dir)
    errors: list[str] = []
    collected: list[str] = []

    # Discover IDs from files in the cluster directory if not provided
    if pdb_id_list is None:
        found_ids: set[str] = set()
        for f in cluster_dir.iterdir():
            m = re.match(r"^([A-Za-z0-9]{4,6})_fixed", f.name)
            if m:
                found_ids.add(m.group(1).upper())
        pdb_id_list = sorted(found_ids)
        logger.info(f"Auto-discovered {len(pdb_id_list)} protein IDs in {cluster_dir}")

    for pid in pdb_id_list:
        pid_up = pid.upper()
        pid_lo = pid.lower()

        pdb_candidates = [
            cluster_dir / f"{pid_up}_fixed_min_2.pdb_2",
            cluster_dir / f"{pid_lo}_fixed_min_2.pdb_2",
            cluster_dir / f"{pid_up}_fixed_min.pdb_2",
            cluster_dir / f"{pid_lo}_fixed_min.pdb_2",
            # also accept already-renamed _final.pdb from a previous run
            cluster_dir / f"{pid_up}_final.pdb",
        ]
        pdb_src = next((p for p in pdb_candidates if p.exists()), None)

        uind_candidates = [
            cluster_dir / f"{pid_up}_fixed_min_2.uind",
            cluster_dir / f"{pid_lo}_fixed_min_2.uind",
            cluster_dir / f"{pid_up}_fixed_min.uind",
            cluster_dir / f"{pid_lo}_fixed_min.uind",
        ]
        uind_src = next((u for u in uind_candidates if u.exists()), None)

        if pdb_src is None:
            errors.append(f"{pid}: no Step-4 PDB output found in {cluster_dir}")
            continue
        if uind_src is None:
            errors.append(f"{pid}: no .uind file found in {cluster_dir}")
            # still collect pdb even without uind

        pdb_dst  = DATA_FINAL_PDB / f"{pid_up}_final.pdb"
        uind_dst = DATA_UIND / f"{pid_up}.uind"

        if not pdb_dst.exists() or overwrite:
            shutil.copy2(pdb_src, pdb_dst)
            logger.info(f"  Copied  {pdb_src.name}  →  {pdb_dst.name}")
        if uind_src and (not uind_dst.exists() or overwrite):
            shutil.copy2(uind_src, uind_dst)
            logger.info(f"  Copied  {uind_src.name}  →  {uind_dst.name}")

        mpoles_candidates = [
            cluster_dir / f"{pid_up}.mpoles",
            cluster_dir / f"{pid_lo}.mpoles",
        ]
        mpoles_src = next((m for m in mpoles_candidates if m.exists()), None)
        mpoles_dst = DATA_MPOLES / f"{pid_up}.mpoles"
        if mpoles_src and (not mpoles_dst.exists() or overwrite):
            shutil.copy2(mpoles_src, mpoles_dst)
            logger.info(f"  Copied  {mpoles_src.name}  →  {mpoles_dst.name}")

        collected.append(pid_up)

    for e in errors:
        logger.warning(e)

    logger.info(f"Collected {len(collected)}/{len(pdb_id_list)} proteins")
    return {"collected": collected, "errors": errors}

# Step 2 – Parse FFX final PDB

def parse_ffx_pdb(pdb_path: Path) -> pd.DataFrame:
    """Parse ATOM/HETATM records from an FFX-minimised PDB.

    Returns a DataFrame with columns:
        atom_number, atom_name, residue_name, chain_id,
        residue_seq, x, y, z
    Only standard amino acid residues are returned (water / ions discarded).
    """
    rows = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            res_name = line[17:20].strip()
            if res_name not in AMINO_ACID_3_TO_FULL:
                continue  # skip HOH, ions, ligands
            try:
                rows.append({
                    "atom_number": int(line[6:11]),
                    "atom_name":   line[12:16].strip(),
                    "residue_name": res_name,
                    "chain_id":    line[21].strip(),
                    "residue_seq": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)

# Step 3 – Assign AMOEBA atom types

def assign_amoeba_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'atom_type' column (AMOEBA int) and 'description' column."""
    df = df.copy()
    df["atom_type"]   = df.apply(
        lambda r: assign_amoeba_type(r["residue_name"], r["atom_name"]), axis=1
    )
    df["description"] = df.apply(
        lambda r: f"{AMINO_ACID_3_TO_FULL.get(r['residue_name'], r['residue_name'])} {r['atom_name']}",
        axis=1,
    )
    return df

# Step 4 – Read FFX .uind induced-dipole file

def read_uind(uind_path: Path) -> dict[int, tuple[float, float, float]]:
    """Parse an FFX .uind file.

    FFX format (from AlgorithmsCommand.saveInducedDipoles):
        {natoms}  {name}
        {index}  {atom_name}  {ux}  {uy}  {uz}

    Returns dict  atom_index → (ux, uy, uz).
    """
    result: dict[int, tuple[float, float, float]] = {}
    with open(uind_path) as fh:
        lines = fh.readlines()
    for line in lines[1:]:   # skip header
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            ux, uy, uz = float(parts[2]), float(parts[3]), float(parts[4])
            result[idx] = (ux, uy, uz)
        except (ValueError, IndexError):
            continue
    return result

def attach_uind(df: pd.DataFrame, uind_path: Path | None) -> pd.DataFrame:
    """Attach induced dipole columns to the atom DataFrame."""
    df = df.copy()
    if uind_path is None or not uind_path.exists():
        df["ux"] = np.nan
        df["uy"] = np.nan
        df["uz"] = np.nan
        return df
    dipoles = read_uind(uind_path)
    df["ux"] = df["atom_number"].map(lambda i: dipoles.get(i, (np.nan,))[0])
    df["uy"] = df["atom_number"].map(lambda i: dipoles.get(i, (np.nan, np.nan))[1])
    df["uz"] = df["atom_number"].map(lambda i: dipoles.get(i, (np.nan, np.nan, np.nan))[2])
    return df

# Step 5 – Read .mpoles permanent-multipole file (from PrintMultipoles.groovy)
# File format written by PrintMultipoles.groovy:
#   {natoms}  {name}
#   {index}  {atom_name}  {q}  {px} {py} {pz}  {Qxx} {Qxy} {Qxz} {Qyy} {Qyz} {Qzz}
#
# Units: charge (e), dipole (e·Å), quadrupole (e·Å²) — AMOEBA local frame

MPOLE_COLS = ["q", "px", "py", "pz", "Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "Qzz"]

def read_mpoles(mpoles_path: Path) -> dict[int, dict[str, float]]:
    """Parse a .mpoles file produced by PrintMultipoles.groovy.

    Returns dict  atom_index → {q, px, py, pz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz}.
    """
    result: dict[int, dict[str, float]] = {}
    with open(mpoles_path) as fh:
        lines = fh.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 12:
            continue
        try:
            idx  = int(parts[0])
            vals = list(map(float, parts[2:12]))
            result[idx] = dict(zip(MPOLE_COLS, vals))
        except (ValueError, IndexError):
            continue
    return result

def attach_mpoles(df: pd.DataFrame, mpoles_path: Path | None) -> pd.DataFrame:
    """Attach permanent multipole columns to the atom DataFrame (optional)."""
    df = df.copy()
    if mpoles_path is None or not mpoles_path.exists():
        for col in MPOLE_COLS:
            df[col] = np.nan
        return df
    mpoles = read_mpoles(mpoles_path)
    for col in MPOLE_COLS:
        df[col] = df["atom_number"].map(lambda i, c=col: mpoles.get(i, {}).get(c, np.nan))
    return df

# Step 6 – Local frame transform (CA → C → O defines residue frame)

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def _build_local_frame(ca: np.ndarray, c: np.ndarray, o: np.ndarray):
    """Return rotation matrix R and origin (ca).

    Convention (same as Tinker_Output_Processing.py):
      x_axis = CA → C direction
      z_axis = perpendicular to CA-C-O plane
      y_axis = z × x
    """
    x = _normalize(c - ca)
    z = _normalize(np.cross(c - ca, o - c))
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])   # 3×3, columns are axes
    return R, ca

def compute_local_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add local-frame coordinate columns (lx, ly, lz) to the atom DataFrame.

    Groups by (chain_id, residue_seq) and finds the backbone N-CA-C-O pattern
    to define each residue's local frame.  Atoms without a valid frame get NaN.
    """
    df = df.copy()
    lx_col = np.full(len(df), np.nan)
    ly_col = np.full(len(df), np.nan)
    lz_col = np.full(len(df), np.nan)

    # Build per-residue backbone atom coordinates lookup
    # We walk through atoms sorted by atom_number to find N-CA-C-O runs
    R_current: np.ndarray | None = None
    origin_current: np.ndarray | None = None

    # Group by (chain_id, residue_seq) for robustness
    for (chain, seq), grp in df.groupby(["chain_id", "residue_seq"], sort=False):
        anames = grp.set_index("atom_name")
        if all(n in anames.index for n in ("CA", "C", "O")):
            ca = anames.loc["CA", ["x", "y", "z"]].values.astype(float)
            c  = anames.loc["C",  ["x", "y", "z"]].values.astype(float)
            o  = anames.loc["O",  ["x", "y", "z"]].values.astype(float)
            # Handle duplicate atom names (e.g. alternate conformations) – take first
            if ca.ndim == 2: ca = ca[0]
            if c.ndim  == 2: c  = c[0]
            if o.ndim  == 2: o  = o[0]
            try:
                R_current, origin_current = _build_local_frame(ca, c, o)
            except Exception:
                R_current = None

        if R_current is not None and origin_current is not None:
            for idx in grp.index:
                xyz = np.array([df.at[idx, "x"], df.at[idx, "y"], df.at[idx, "z"]])
                local = R_current.T @ (xyz - origin_current)
                lx_col[idx] = local[0]
                ly_col[idx] = local[1]
                lz_col[idx] = local[2]

    df["lx"] = lx_col
    df["ly"] = ly_col
    df["lz"] = lz_col
    return df

# Step 7 – Build per-titratable-residue node feature vectors

# Atom-label map: determines the 'atom_label' column for one-hot later in GNN
ATOM_LABEL_MAP = {
    "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
    "H": 5, "HA": 5, "HB": 5, "HN": 5,          # all H → 5
    "S": 6, "P": 7,
}
DEFAULT_ATOM_LABEL = 8   # catch-all for heavy atoms not listed above

def _atom_label(atom_name: str) -> int:
    stripped = atom_name.strip()
    if stripped in ATOM_LABEL_MAP:
        return ATOM_LABEL_MAP[stripped]
    # Hydrogen catch-all
    if stripped.startswith("H") or (len(stripped) > 1 and stripped[1] == "H"):
        return 5
    return DEFAULT_ATOM_LABEL

def _residue_one_hot(res3: str) -> list[float]:
    vec = [0.0] * len(RESIDUE_ORDER)
    idx = RESIDUE_TO_INDEX.get(res3.upper())
    if idx is not None:
        vec[idx] = 1.0
    return vec

def build_node_features(
    df: pd.DataFrame,
    pdb_id: str,
    expt_pka: pd.DataFrame | None,
    radii: list[int] = RADII,
    use_mpoles: bool = True,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Build per-titratable-residue node feature DataFrames for each radius.

    Returns
    -------
    dict radius → {"{pdb_id}_{chain}_{seq}.{res3}" : feature_DataFrame}
    """
    results: dict[int, dict[str, pd.DataFrame]] = {r: {} for r in radii}

    # Build a lookup of (chain, seq) → pKa from the experimental CSV
    pka_lookup: dict[tuple[str, int], float] = {}
    if expt_pka is not None:
        for _, row in expt_pka.iterrows():
            key = (str(row.get("chain_id", "A")).strip().upper(), int(row["res_num"]))
            pka_lookup[key] = float(row["pKa"])

    titratable = df[df["residue_name"].isin(TITRATABLE_RES)].copy()
    unique_res = titratable.groupby(["chain_id", "residue_seq", "residue_name"], sort=False).first().reset_index()

    for _, tit_row in unique_res.iterrows():
        chain = tit_row["chain_id"]
        seq   = int(tit_row["residue_seq"])
        res3  = tit_row["residue_name"]

        # Centre-of-mass of the titratable residue for distance calculation
        res_atoms = df[(df["chain_id"] == chain) & (df["residue_seq"] == seq)]
        if res_atoms.empty:
            continue
        com = res_atoms[["x", "y", "z"]].mean().values

        pka_val = pka_lookup.get((chain.upper(), seq))

        for radius in radii:
            # Select all atoms within `radius` Å of the residue COM
            dists = np.linalg.norm(df[["x", "y", "z"]].values - com, axis=1)
            nbr = df[dists <= radius].copy()
            if nbr.empty:
                continue

            rows = []
            for _, atom in nbr.iterrows():
                one_hot = _residue_one_hot(atom["residue_name"])
                feat: dict[str, float] = {}

                # Experimental pKa (target) – first atom of each residue carries it
                if pka_val is not None:
                    feat["Expt. pKa"] = pka_val

                feat["atom_label"] = _atom_label(atom["atom_name"])

                # Local-frame coordinates
                feat["lx"] = atom["lx"]
                feat["ly"] = atom["ly"]
                feat["lz"] = atom["lz"]

                # Induced dipole
                feat["ux"] = atom["ux"]
                feat["uy"] = atom["uy"]
                feat["uz"] = atom["uz"]

                # Permanent multipoles (present only if .mpoles was loaded)
                if use_mpoles:
                    for col in MPOLE_COLS:
                        feat[col] = atom[col]

                # Residue one-hot
                for i, rname in enumerate(RESIDUE_ORDER):
                    feat[f"Residue Name_{rname}"] = one_hot[i]

                rows.append(feat)

            if not rows:
                continue

            key = f"{pdb_id}_{chain}_{seq}.{res3}"
            results[radius][key] = pd.DataFrame(rows)

    return results

# Step 8 – Build adjacency matrices (within-radius inter-residue adjacency)

def build_adjacency_matrix(
    df: pd.DataFrame,
    pdb_id: str,
    titratable_chain: str,
    titratable_seq: int,
    titratable_res: str,
    radius: float,
    com: np.ndarray,
) -> pd.DataFrame:
    """Binary adjacency matrix (with self-loops) between residues in the neighbourhood.

    Rows/columns are residue labels; an edge exists if any two atoms of different
    residues are within `radius` Å, or it is the self-residue (self-loop).
    """
    dists = np.linalg.norm(df[["x", "y", "z"]].values - com, axis=1)
    nbr = df[dists <= radius].copy()

    # Unique residue labels in the neighbourhood
    nbr["res_label"] = nbr["chain_id"] + "_" + nbr["residue_seq"].astype(str) + "_" + nbr["residue_name"]
    res_labels = nbr["res_label"].unique().tolist()
    n = len(res_labels)
    label_idx = {l: i for i, l in enumerate(res_labels)}

    adj = np.zeros((n, n), dtype=int)
    # Self-loops
    np.fill_diagonal(adj, 1)

    # Build per-residue atom coordinate arrays
    res_coords: dict[str, np.ndarray] = {}
    for label, grp in nbr.groupby("res_label"):
        res_coords[label] = grp[["x", "y", "z"]].values

    for i, li in enumerate(res_labels):
        for j, lj in enumerate(res_labels):
            if i >= j:
                continue
            ci, cj = res_coords[li], res_coords[lj]
            # Minimum inter-residue distance
            dij = np.min(np.linalg.norm(ci[:, None, :] - cj[None, :, :], axis=-1))
            if dij <= radius:
                adj[i, j] = 1
                adj[j, i] = 1

    return pd.DataFrame(adj, index=res_labels, columns=res_labels)

# Main pipeline – process one protein

def process_protein(
    pdb_id: str,
    expt_pka_df: pd.DataFrame | None = None,
    use_mpoles: bool = True,
) -> bool:
    """Run the full local pipeline for one protein ID.

    Reads from Data/1_FFX_Final_PDB/, Data/2_Uind_Files/, Data/3_Mpoles_Files/.
    Writes feature CSVs to Features/Per_Protein/{pdb_id}/ and
    Features/Node_Feature_Vectors/{radius}/.
    """
    pid = pdb_id.upper()
    pdb_path   = DATA_FINAL_PDB / f"{pid}_final.pdb"
    uind_path  = DATA_UIND      / f"{pid}.uind"
    mpoles_path= DATA_MPOLES    / f"{pid}.mpoles"

    if not pdb_path.exists():
        logger.warning(f"[{pid}] PDB not found: {pdb_path}")
        return False

    logger.info(f"[{pid}] Parsing PDB ...")
    df = parse_ffx_pdb(pdb_path)
    if df.empty:
        logger.warning(f"[{pid}] Empty DataFrame after PDB parse")
        return False

    logger.info(f"[{pid}] Assigning AMOEBA types ...")
    df = assign_amoeba_types(df)

    logger.info(f"[{pid}] Attaching induced dipoles ...")
    df = attach_uind(df, uind_path if uind_path.exists() else None)

    has_mpoles = mpoles_path.exists()
    if use_mpoles:
        if has_mpoles:
            logger.info(f"[{pid}] Attaching permanent multipoles ...")
        else:
            logger.info(f"[{pid}] No .mpoles file found – permanent multipole columns will be NaN")
    df = attach_mpoles(df, mpoles_path if has_mpoles else None)

    logger.info(f"[{pid}] Computing local frames ...")
    df = compute_local_frame(df)

    # Save full per-protein CSV
    out_dir = FEATURES_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{pid}_atoms.csv", index=False)

    # Filter pKa rows for this protein
    pid_pka: pd.DataFrame | None = None
    if expt_pka_df is not None:
        pid_col = expt_pka_df.get("pdb_id", expt_pka_df.get("PDB_ID", None))
        if pid_col is not None:
            pid_pka = expt_pka_df[pid_col.str.upper() == pid]

    logger.info(f"[{pid}] Building node features ...")
    node_feats = build_node_features(
        df, pid, pid_pka,
        radii=RADII,
        use_mpoles=use_mpoles,
    )

    # Also build adjacency matrices (use the largest radius for the nbr selection)
    titratable = df[df["residue_name"].isin(TITRATABLE_RES)].copy()
    unique_res = titratable.groupby(
        ["chain_id", "residue_seq", "residue_name"], sort=False
    ).first().reset_index()

    for _, tit_row in unique_res.iterrows():
        chain = tit_row["chain_id"]
        seq   = int(tit_row["residue_seq"])
        res3  = tit_row["residue_name"]
        res_atoms = df[(df["chain_id"] == chain) & (df["residue_seq"] == seq)]
        if res_atoms.empty:
            continue
        com = res_atoms[["x", "y", "z"]].mean().values
        key = f"{pid}_{chain}_{seq}.{res3}"

        for radius in RADII:
            adj = build_adjacency_matrix(df, pid, chain, seq, res3, radius, com)
            adj_dir = ADJ_MATRIX_DIR
            adj_dir.mkdir(parents=True, exist_ok=True)
            adj.to_csv(adj_dir / f"{key}_r{radius}_adjacency.csv")

            feat_df = node_feats[radius].get(key)
            if feat_df is not None:
                nf_dir = NODE_FEAT_ROOT / str(radius)
                nf_dir.mkdir(parents=True, exist_ok=True)
                feat_df.to_csv(nf_dir / f"{key}.csv", index=False)

    logger.info(f"[{pid}] Done.")
    return True

# Main pipeline – process all proteins

def load_experimental_pka(csv_path: Path = PKAD_CSV) -> pd.DataFrame | None:
    """Load the PKAD experimental pKa CSV.

    Expected columns: PDB_ID (or pdb_id), chain_id, res_num, pKa (or similar).
    Returns None if the file does not exist.
    """
    if not csv_path.exists():
        logger.warning(f"Experimental pKa CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    # Normalise column names to lower-case
    df.columns = [c.strip().lower() for c in df.columns]
    # Rename PKAD-R CSV columns to the standard names expected downstream
    _rename = {
        "pdb":          "pdb_id",
        "chain":        "chain_id",
        "resid in pdb": "res_num",
        "resname":      "residue_name",
        "expt. pka":    "pKa",
    }
    df = df.rename(columns={k: v for k, v in _rename.items() if k in df.columns})
    logger.info(f"Loaded {len(df)} experimental pKa entries from {csv_path.name}")
    return df

def run_pipeline(
    pdb_ids: list[str] | None = None,
    use_mpoles: bool = True,
) -> None:
    """Process all proteins found in Data/1_FFX_Final_PDB/.

    Parameters
    ----------
    pdb_ids : list[str] | None
        Subset of IDs to process.  None = process all found.
    use_mpoles : bool
        Whether to include permanent multipole features.
    """
    expt_pka = load_experimental_pka()

    if pdb_ids is None:
        pdb_ids = [
            re.sub(r"_final\.pdb$", "", f.name, flags=re.IGNORECASE).upper()
            for f in DATA_FINAL_PDB.iterdir()
            if f.name.lower().endswith("_final.pdb")
        ]
        logger.info(f"Found {len(pdb_ids)} proteins in {DATA_FINAL_PDB}")

    ok = 0
    for pid in sorted(pdb_ids):
        success = process_protein(pid, expt_pka, use_mpoles=use_mpoles)
        if success:
            ok += 1

    logger.info(f"Pipeline complete: {ok}/{len(pdb_ids)} successful")

# Pickle dataset builder (feeds Net/)

def write_datasets(
    radii: list[int] = RADII,
    output_dir: Path = BASE / "Features" / "Node_Feature_Vectors" / "Subsets",
) -> None:
    """Bundle per-residue CSVs + adjacency matrices into PyG-ready pickles.

    Mirrors the logic in Net/create_data.py but adapted for the new file
    naming convention (adds radius suffix to adjacency filenames).

    Requires torch and torch_geometric to be installed.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
    except ImportError:
        logger.error("torch / torch_geometric not installed – skipping pickle write")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for radius in radii:
        nf_dir   = NODE_FEAT_ROOT / str(radius)
        adj_dir  = ADJ_MATRIX_DIR
        data_list = []

        for nf_csv in sorted(nf_dir.glob("*.csv")):
            key = nf_csv.stem                          # e.g. 1ABC_A_42.ASP
            adj_csv = adj_dir / f"{key}_r{radius}_adjacency.csv"

            if not adj_csv.exists():
                logger.debug(f"Skipping {key}: adjacency matrix not found")
                continue

            adj_mat = pd.read_csv(adj_csv, header=0, index_col=0).values
            adj_tensor = torch.tensor(adj_mat, dtype=torch.int)

            nf = pd.read_csv(nf_csv, header=0)
            if "atom_label" not in nf.columns:
                continue

            pka_tensor = None
            if "Expt. pKa" in nf.columns:
                pka_tensor = torch.tensor([nf["Expt. pKa"].values[0]], dtype=torch.float)

            atom_label_enc = F.one_hot(
                torch.tensor(nf["atom_label"].values.astype(int), dtype=torch.long),
                num_classes=9,
            ).float()

            one_hot_cols = nf.filter(like="Residue Name_").values
            residue_label = int(np.argmax(one_hot_cols, axis=1)[0])

            drop_cols = [c for c in ["Expt. pKa", "atom_label"] if c in nf.columns]
            feat_tensor = torch.tensor(nf.drop(columns=drop_cols).values, dtype=torch.float)
            feat_tensor = torch.cat([feat_tensor, atom_label_enc], dim=1)

            edge_index = adj_tensor.nonzero(as_tuple=True)
            edge_index = torch.stack(edge_index, dim=0)

            # Parse key: {pdb_id}_{chain}_{seq}.{res_name}
            parts = key.rsplit(".", 1)
            res_name = parts[1] if len(parts) == 2 else ""
            info_parts = parts[0].rsplit("_", 2)
            pdb_id      = info_parts[0] if len(info_parts) >= 1 else key
            chain_id    = info_parts[1] if len(info_parts) >= 2 else ""
            res_num     = int(info_parts[2]) if len(info_parts) >= 3 else 0

            data = Data(
                x=feat_tensor,
                edge_index=edge_index,
                y=pka_tensor,
                residue_label=residue_label,
            )
            data.PDB_ID         = pdb_id
            data.Chain_ID       = chain_id
            data.Residue_Number = res_num
            data.Residue_Name   = res_name

            data_list.append(data)

        pkl_path = output_dir / f"data_list_r{radius}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(data_list, fh)
        logger.info(f"Saved {len(data_list)} graphs → {pkl_path.name}")

# CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FFX-based pKa GNN feature pipeline"
    )
    parser.add_argument(
        "--collect",
        metavar="CLUSTER_DIR",
        help="Collect FFX outputs from a local copy of the cluster output directory",
    )
    parser.add_argument(
        "--process",
        nargs="*",
        metavar="PDB_ID",
        help="Run the local processing pipeline.  No args = all proteins in Data/1_FFX_Final_PDB/",
    )
    parser.add_argument(
        "--write-datasets",
        action="store_true",
        help="Bundle node features + adjacency matrices into PKL datasets",
    )
    parser.add_argument(
        "--no-mpoles",
        action="store_true",
        help="Skip permanent multipole features even if .mpoles files exist",
    )
    args = parser.parse_args()

    if args.collect:
        collect_ffx_outputs(args.collect)

    if args.process is not None:
        ids = args.process if args.process else None
        run_pipeline(pdb_ids=ids, use_mpoles=not args.no_mpoles)

    if args.write_datasets:
        write_datasets()

    if not any([args.collect, args.process is not None, args.write_datasets]):
        parser.print_help()
