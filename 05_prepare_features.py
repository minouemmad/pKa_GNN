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

import logging
import os
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
PDB_DIR  = "data/fixed_pdbs"
MANIFEST = "data/manifest.csv"
FEAT_DIR  = "Graph_pKa/Features"
NODE_DIR  = os.path.join(FEAT_DIR, "Node_Feature_Vectors")
ADJ_DIR   = os.path.join(FEAT_DIR, "Adjacency_Matrices/With_Self_Loop")
BOND_DIR  = os.path.join(FEAT_DIR, "Edge_Features")

RADII           = [7, 8, 9, 10, 11]
TARGET_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"}

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
# Locate completed FFX jobs
# ════════════════════════════════════════════════════════════════════════════

def find_completed_jobs(pdb_dir: str) -> list[tuple[str, str, str]]:
    """Return a list of (pdb_id, pdb_path, uind_path) for every protein that
    has a *_final.uind file in its per-protein subdirectory, indicating step 4
    of 03_run_ffx_minimize.py succeeded and 05_organize_ffx_output.py has run.

    Expected layout (after running 05_organize_ffx_output.py):
        {pdb_dir}/{PDB}/{PDB}_final.pdb   – final minimized geometry
        {pdb_dir}/{PDB}/{PDB}_final.uind  – AMOEBA induced dipoles
    """
    d = Path(pdb_dir)
    jobs: list[tuple[str, str, str]] = []

    for uind in sorted(d.glob("*/*_final.uind")):
        pdb_id   = uind.parent.name          # subdirectory name == PDB ID
        pdb_path = uind.parent / f"{pdb_id}_final.pdb"

        if not pdb_path.exists():
            log.warning(f"  {pdb_id}: _final.uind found but _final.pdb missing – skipping")
            continue

        jobs.append((pdb_id, str(pdb_path), str(uind)))

    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(ADJ_DIR,  exist_ok=True)
    os.makedirs(BOND_DIR, exist_ok=True)
    for r in RADII:
        os.makedirs(os.path.join(NODE_DIR, str(r)), exist_ok=True)

    # ── Load pKa labels ──────────────────────────────────────────────────────
    manifest = pd.read_csv(MANIFEST)
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
    log.info(f"Loaded {len(pka_lookup)} pKa entries from {MANIFEST}")

    # ── Discover completed FFX jobs ──────────────────────────────────────────
    jobs = find_completed_jobs(PDB_DIR)
    if not jobs:
        log.error(f"No completed minimisations found in {PDB_DIR}.")
        return
    log.info(f"Found {len(jobs)} completed job(s): {[j[0] for j in jobs]}")

    # ── Per-protein processing  (collect raw data for global normalisation) ──
    all_frames: list[pd.DataFrame] = []

    for pdb_id, pdb_path, uind_path in jobs:
        log.info(f"Processing {pdb_id}  ({Path(pdb_path).name})")

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
            nbr_df = compute_neighbor_counts(pdb_path, df)
            df = df.merge(nbr_df, on="serial", how="left")
        except Exception as exc:
            log.warning(f"  Neighbour counts failed ({exc}); inserting zeros.")
            for r in RADII:
                for t in ["N_Count", "CA_C_Count", "O_Count", "S_Count"]:
                    df[f"Radius_{r}A_{t}"] = 0

        # 6. H-bonds and SASA
        log.info(f"  Computing H-bonds and SASA…")
        hb_df = compute_hbonds_sasa(pdb_path, len(df))
        df = df.merge(hb_df, on="serial", how="left")

        # 7. Tag protein
        df["pdb_id"] = pdb_id.upper()

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
    base_feature_cols = residue_ohe_cols + [
        "atom_label",
        "recalculated_x",
        "recalculated_y",
        "recalculated_z",
        "Dipole_X",
        "Dipole_Y",
        "Dipole_Z",
        "Number of H-Bonds as donor",
        "Number of H-Bonds as acceptor",
        "SASA_Value",
    ]

    n_adj_saved  = 0
    n_feat_saved = 0
    n_skipped    = 0

    for (pdb_id, chain, resseq, resname), res_df in full_df.groupby(
        ["pdb_id", "chain", "resseq", "resname"], sort=False
    ):
        if resname not in TARGET_RESIDUES:
            continue

        key = (pdb_id.upper(), str(chain), int(resseq), resname.upper())
        pka = pka_lookup.get(key)
        if pka is None:
            n_skipped += 1
            continue

        full_name = AMINO_ACID_3_TO_FULL.get(resname, resname)
        stem      = f"{pdb_id}_{chain}_{resseq}.{full_name}"

        # Adjacency matrix (uses original x/y/z, not local-frame coords)
        adj_df   = build_adjacency_matrix(res_df)
        adj_path = os.path.join(ADJ_DIR, f"{stem}_adjacency.csv")
        adj_df.to_csv(adj_path)
        n_adj_saved += 1

        # Edge features (local-frame displacement vectors + distance)
        bond_df   = build_edge_features(res_df)
        bond_path = os.path.join(BOND_DIR, f"{stem}_bonds.csv")
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

            out_path = os.path.join(NODE_DIR, str(radius), f"{stem}.csv")
            feat_df.to_csv(out_path, index=False)
            n_feat_saved += 1

    log.info(f"Done.")
    log.info(f"  Adjacency matrices saved : {n_adj_saved}  → {ADJ_DIR}")
    log.info(f"  Edge feature files saved : {n_adj_saved}  → {BOND_DIR}")
    log.info(f"  Node feature files saved : {n_feat_saved} → {NODE_DIR}")
    log.info(f"  Residues skipped (no pKa in manifest): {n_skipped}")


if __name__ == "__main__":
    main()
