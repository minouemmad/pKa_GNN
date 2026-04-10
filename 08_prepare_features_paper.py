"""
08_prepare_features_paper.py

Paper-exact feature extraction — replicates the original Graph_pKa GitHub
pipeline (Song et al., J. Chem. Inf. Model. 2026) as closely as possible
using FFX/GK output instead of Tinker molecular dynamics.

Key differences vs 04_prepare_features.py:
  • 4-class residue OHE  (ASP / GLU / HIS / LYS only) — matches paper
  • 9-class atom-type OHE (labels 0-8, no sidechain-S class 9) — matches paper
  • No edge-feature output (paper does not include edge attributes)
  • input_dim = 4 + 3 (coords) + 3 (dipoles) + 4 (counts) + 2 (Hbond) + 1 (SASA)
              + 9 (atom OHE) = 26  — matches paper's reported feature count

Outputs (inside Graph_pKa/Features_Paper/):
    Node_Feature_Vectors/{radius}/{PDB}_{chain}_{resid}.{ResName}.csv
    Adjacency_Matrices/With_Self_Loop/{PDB}_{chain}_{resid}.{ResName}_adjacency.csv

Run:
    python 08_prepare_features_paper.py
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances as mda_distances
except ImportError:
    raise SystemExit("MDAnalysis not found. Install with: conda install -c conda-forge mdanalysis")

try:
    import mdtraj as md
except ImportError:
    raise SystemExit("mdtraj not found. Install with: conda install -c conda-forge mdtraj")


# ── Configuration ─────────────────────────────────────────────────────────────
PDB_DIR  = "data/fixed_pdbs"
MANIFEST = "data/manifest.csv"
# Separate output root so paper-exact outputs don't clobber existing Feature files
FEAT_DIR  = "Graph_pKa/Features_Paper"
NODE_DIR  = os.path.join(FEAT_DIR, "Node_Feature_Vectors")
ADJ_DIR   = os.path.join(FEAT_DIR, "Adjacency_Matrices/With_Self_Loop")

RADII           = [7, 8, 9, 10, 11]

# Paper: 4 target residues only (no CYS / TYR — excluded due to data scarcity)
TARGET_RESIDUES = {"ASP", "GLU", "HIS", "LYS"}
TARGET_RESIDUE_OHE = ["ASP", "GLU", "HIS", "LYS"]   # column order matches original PKAD_Data

# Paper: 9-class atom labels (no class 9 = sidechain S, which only appears in CYS)
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
    # ("S",  "SC"): 9  ← excluded; CYS not a target residue in paper
}

BOND_CUTOFF = 1.9   # Å — covers all covalent bonds including C-S

AMINO_ACID_3_TO_FULL: dict[str, str] = {
    "ALA": "Alanine",   "ARG": "Arginine", "ASN": "Asparagine", "ASP": "Aspartate",
    "CYS": "Cysteine",  "GLN": "Glutamine","GLU": "Glutamate",  "GLY": "Glycine",
    "HIS": "Histidine", "ILE": "Isoleucine","LEU": "Leucine",   "LYS": "Lysine",
    "MET": "Methionine","PHE": "Phenylalanine","PRO": "Proline", "SER": "Serine",
    "THR": "Threonine", "TRP": "Tryptophan","TYR": "Tyrosine",  "VAL": "Valine",
}

BACKBONE_HEAVY = {"N", "CA", "C", "O"}
BACKBONE_H     = {"H", "HN", "H1", "H2", "H3", "HA", "HA2", "HA3"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_pdb(path: str) -> pd.DataFrame:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            rows.append(dict(
                serial  = int(line[6:11]),
                name    = line[12:16].strip(),
                resname = line[17:20].strip(),
                chain   = line[21].strip() or "A",
                resseq  = int(line[22:26]),
                x       = float(line[30:38]),
                y       = float(line[38:46]),
                z       = float(line[46:54]),
            ))
    return pd.DataFrame(rows)


def parse_uind(path: str) -> dict[int, tuple[float, float, float]]:
    """Parse FFX .uind induced-dipole file → {serial: (ux, uy, uz)}."""
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
# Atom classification
# ════════════════════════════════════════════════════════════════════════════

def classify_backbone_sidechain(atom_name: str) -> str:
    return "BB" if (atom_name in BACKBONE_HEAVY or atom_name in BACKBONE_H) else "SC"


def assign_atom_label(atom_name: str, bb_sc: str) -> int:
    key_char = "CA" if atom_name == "CA" else atom_name[0]
    return ATOM_LABEL_MAP.get((key_char, bb_sc), -1)


# ════════════════════════════════════════════════════════════════════════════
# Local backbone frame (matches original Tinker_Output_Processing.py exactly)
# ════════════════════════════════════════════════════════════════════════════

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_local_frame(ca: np.ndarray, c: np.ndarray, o: np.ndarray):
    """
    x-axis  along CA→C
    z-axis  normal to CA-C-O plane
    y-axis  z × x (right-handed)
    Returns (R [3×3], origin [CA coords]).
    """
    x_axis = _normalize(c - ca)
    z_axis = _normalize(np.cross(c - ca, o - c))
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, ca


def compute_local_frame_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Add recalculated_x/y/z; rotate Dipole_X/Y/Z into local frame."""
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

        def _get_xyz(name):
            m = res_rows.loc[res_rows["name"] == name, ["x", "y", "z"]]
            return m.iloc[0].values.astype(float) if not m.empty else None

        ca, c, o = _get_xyz("CA"), _get_xyz("C"), _get_xyz("O")
        if ca is None or c is None or o is None:
            continue

        R, origin = build_local_frame(ca, c, o)

        for pos in res_idx:
            row = df.loc[pos]
            lc = R.T @ (np.array([row.x, row.y, row.z]) - origin)
            rx[pos], ry[pos], rz[pos] = lc

            if has_dipoles:
                dv = np.array([row.Dipole_X, row.Dipole_Y, row.Dipole_Z])
                if not np.any(np.isnan(dv)):
                    dl = R.T @ dv
                    ldx[pos], ldy[pos], ldz[pos] = dl

    df["recalculated_x"] = rx
    df["recalculated_y"] = ry
    df["recalculated_z"] = rz

    if has_dipoles:
        df["Dipole_X"] = ldx
        df["Dipole_Y"] = ldy
        df["Dipole_Z"] = ldz

    return df


# ════════════════════════════════════════════════════════════════════════════
# Neighbour counts (MDAnalysis)
# ════════════════════════════════════════════════════════════════════════════

def compute_neighbor_counts(pdb_path: str, df: pd.DataFrame) -> pd.DataFrame:
    u      = mda.Universe(pdb_path)
    heavy  = u.select_atoms("not (name H* or name [0-9]H*)")
    coords = heavy.positions
    resids = heavy.resids
    names  = heavy.names

    dist_mat = np.zeros((len(heavy), len(heavy)), dtype=np.float64)
    mda_distances.distance_array(coords, coords, result=dist_mat)

    records: list[dict] = []
    for i, atom in enumerate(heavy):
        serial = int(atom.index) + 1
        row: dict = {"serial": serial}
        other_res = resids != resids[i]

        for r in RADII:
            in_sphere  = (dist_mat[i] <= r) & other_res
            nbr_names  = names[in_sphere]
            row[f"Radius_{r}A_N_Count"]    = int(np.sum(nbr_names == "N"))
            row[f"Radius_{r}A_CA_C_Count"] = int(np.sum((nbr_names == "CA") | (nbr_names == "C")))
            row[f"Radius_{r}A_O_Count"]    = int(np.sum(nbr_names == "O"))
            row[f"Radius_{r}A_S_Count"]    = int(np.sum(
                np.char.startswith(nbr_names.astype("U10"), "S")
            ))
        records.append(row)

    nbr_df = pd.DataFrame(records)

    # Hydrogens get zero counts
    h_serials = np.array([a.index + 1 for a in u.atoms if a not in heavy])
    if len(h_serials):
        h_rows = pd.DataFrame({"serial": h_serials})
        for r in RADII:
            for t in ["N_Count", "CA_C_Count", "O_Count", "S_Count"]:
                h_rows[f"Radius_{r}A_{t}"] = 0
        nbr_df = pd.concat([nbr_df, h_rows], ignore_index=True)

    return nbr_df


# ════════════════════════════════════════════════════════════════════════════
# H-bonds and SASA (mdtraj)
# ════════════════════════════════════════════════════════════════════════════

def compute_hbonds_sasa(pdb_path: str, n_atoms: int) -> pd.DataFrame:
    try:
        traj     = md.load(pdb_path)
        hbonds   = md.baker_hubbard(traj, periodic=False)
        donor_c  = defaultdict(int)
        accept_c = defaultdict(int)
        for di, _hi, ai in hbonds:
            donor_c[di]  += 1
            accept_c[ai] += 1

        sasa = md.shrake_rupley(traj, mode="atom")[0]

        return pd.DataFrame([
            {
                "serial":                        atom.index + 1,
                "Number of H-Bonds as donor":    donor_c[atom.index],
                "Number of H-Bonds as acceptor": accept_c[atom.index],
                "SASA_Value":                    float(sasa[atom.index]),
            }
            for atom in traj.topology.atoms
        ])
    except Exception as exc:
        log.warning(f"    H-bond/SASA failed ({exc}); using zeros.")
        return pd.DataFrame({
            "serial":                        range(1, n_atoms + 1),
            "Number of H-Bonds as donor":    0,
            "Number of H-Bonds as acceptor": 0,
            "SASA_Value":                    0.0,
        })


# ════════════════════════════════════════════════════════════════════════════
# Adjacency matrix (distance-based, self-loops included)
# ════════════════════════════════════════════════════════════════════════════

def build_adjacency_matrix(res_df: pd.DataFrame) -> pd.DataFrame:
    coords = res_df[["x", "y", "z"]].values.astype(float)
    n = len(res_df)
    adj = np.eye(n, dtype=int)
    for i in range(n):
        dists = np.linalg.norm(coords[i] - coords, axis=1)
        adj[i] = (dists <= BOND_CUTOFF).astype(int)
    names = res_df["name"].tolist()
    return pd.DataFrame(adj, index=names, columns=names)


# ════════════════════════════════════════════════════════════════════════════
# Find completed FFX jobs
# ════════════════════════════════════════════════════════════════════════════

def find_completed_jobs(pdb_dir: str) -> list[tuple[str, str, str]]:
    d = Path(pdb_dir)
    jobs: list[tuple[str, str, str]] = []
    for uind in sorted(d.glob("*/*_final.uind")):
        pdb_id   = uind.parent.name
        pdb_path = uind.parent / f"{pdb_id}_final.pdb"
        if not pdb_path.exists():
            log.warning(f"  {pdb_id}: .uind found but .pdb missing – skipping")
            continue
        jobs.append((pdb_id, str(pdb_path), str(uind)))
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(ADJ_DIR, exist_ok=True)
    for r in RADII:
        os.makedirs(os.path.join(NODE_DIR, str(r)), exist_ok=True)

    # Load pKa labels
    manifest    = pd.read_csv(MANIFEST)
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

    jobs = find_completed_jobs(PDB_DIR)
    if not jobs:
        log.error(f"No completed minimisations found in {PDB_DIR}.")
        return
    log.info(f"Found {len(jobs)} completed job(s)")

    all_frames: list[pd.DataFrame] = []

    for pdb_id, pdb_path, uind_path in jobs:
        log.info(f"Processing {pdb_id}")

        df = parse_pdb(pdb_path)
        if df.empty:
            log.warning(f"  {pdb_id}: empty PDB – skipping")
            continue

        # Induced dipoles
        dipoles = parse_uind(uind_path)
        df["Dipole_X"] = df["serial"].map(lambda s: dipoles.get(s, (np.nan,)*3)[0])
        df["Dipole_Y"] = df["serial"].map(lambda s: dipoles.get(s, (np.nan,)*3)[1])
        df["Dipole_Z"] = df["serial"].map(lambda s: dipoles.get(s, (np.nan,)*3)[2])
        n_matched = df["serial"].isin(dipoles).sum()
        log.info(f"  Matched {n_matched}/{len(df)} atoms to dipole entries")

        # Atom classification
        df["bb_sc"]      = df["name"].apply(classify_backbone_sidechain)
        df["atom_label"] = df.apply(lambda r: assign_atom_label(r["name"], r["bb_sc"]), axis=1)

        # Local frame
        df = compute_local_frame_coords(df)

        # Neighbour counts
        log.info(f"  Computing neighbour counts…")
        try:
            nbr_df = compute_neighbor_counts(pdb_path, df)
            df = df.merge(nbr_df, on="serial", how="left")
        except Exception as exc:
            log.warning(f"  Neighbour counts failed ({exc}); using zeros.")
            for r in RADII:
                for t in ["N_Count", "CA_C_Count", "O_Count", "S_Count"]:
                    df[f"Radius_{r}A_{t}"] = 0

        # H-bonds + SASA
        log.info(f"  Computing H-bonds and SASA…")
        hb_df = compute_hbonds_sasa(pdb_path, len(df))
        df = df.merge(hb_df, on="serial", how="left")

        df["pdb_id"] = pdb_id.upper()
        all_frames.append(df)
        log.info(f"  {pdb_id}: {len(df)} atoms")

    if not all_frames:
        log.error("No data processed. Exiting.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    log.info(f"Total atoms: {len(full_df)}")

    # Global MinMax normalisation of neighbour counts (exactly as in original pipeline)
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

    # 4-class residue OHE (paper: ASP/GLU/HIS/LYS only)
    residue_ohe_cols: list[str] = []
    for resname_3 in TARGET_RESIDUE_OHE:
        full_name = AMINO_ACID_3_TO_FULL[resname_3]
        col       = f"Residue Name_{full_name}"
        full_df[col] = (full_df["resname"] == resname_3).astype(float)
        residue_ohe_cols.append(col)

    # Feature columns in paper-exact order
    # (matches original PKAD_Data column ordering, with dipoles instead of atomic_charge)
    base_feature_cols = (
        residue_ohe_cols                          # 4 cols
        + ["atom_label"]                          # 1  (→ 9-class OHE downstream)
        + ["recalculated_x", "recalculated_y", "recalculated_z"]   # 3
        + ["Dipole_X", "Dipole_Y", "Dipole_Z"]   # 3  (local-frame induced dipoles)
        + ["Number of H-Bonds as donor", "Number of H-Bonds as acceptor"]  # 2
        + ["SASA_Value"]                          # 1
        # Total non-OHE = 4+1+3+3+2+1 = 14; after 9-class OHE of atom_label → 26
    )

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

        # Adjacency matrix
        adj_df   = build_adjacency_matrix(res_df)
        adj_path = os.path.join(ADJ_DIR, f"{stem}_adjacency.csv")
        adj_df.to_csv(adj_path)
        n_adj_saved += 1

        # Node features (one file per radius)
        for radius in RADII:
            radius_cols = [
                f"Radius_{radius}A_N_Count",
                f"Radius_{radius}A_CA_C_Count",
                f"Radius_{radius}A_O_Count",
                f"Radius_{radius}A_S_Count",
            ]
            wanted  = base_feature_cols + radius_cols
            avail   = [c for c in wanted if c in res_df.columns]
            feat_df = res_df[avail].copy()
            feat_df["Expt.pKa"] = pka   # use original naming without space

            out_path = os.path.join(NODE_DIR, str(radius), f"{stem}.csv")
            feat_df.to_csv(out_path, index=False)
            n_feat_saved += 1

    log.info(f"Done.")
    log.info(f"  Adjacency matrices : {n_adj_saved}  → {ADJ_DIR}")
    log.info(f"  Node feature files : {n_feat_saved} → {NODE_DIR}")
    log.info(f"  Skipped (no pKa)   : {n_skipped}")
    log.info(f"  Effective input_dim (after 9-class OHE): "
             f"{len(residue_ohe_cols) + 3 + 3 + 2 + 1 + 4 + 9} = 26")


if __name__ == "__main__":
    main()
