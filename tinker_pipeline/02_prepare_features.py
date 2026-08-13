
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
PDB_DIR  = os.environ.get("PAPER_PDB_DIR",  "data/fixed_pdbs")
MANIFEST = os.environ.get("PAPER_MANIFEST", "data/manifest.csv")
# Separate output root so paper-exact outputs don't clobber existing Feature files
FEAT_DIR  = os.environ.get("PAPER_FEAT_DIR", "Graph_pKa/Features_Paper")
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

def parse_uperm(path: str) -> dict[int, tuple[float, float, float, float]]:
    """Parse FFX .uperm permanent-multipole file.

    Format per atom row: <serial> <name> <q> <dx> <dy> <dz> <Qxx> <Qyy> <Qzz> <Qxy> <Qxz> <Qyz>
    Returns {serial: (q, dx, dy, dz)} (monopole + permanent dipole).
    """
    out: dict[int, tuple[float, float, float, float]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6 or not parts[0].lstrip("-").isdigit():
                continue
            try:
                serial = int(parts[0])
                q = float(parts[2])
                dx, dy, dz = float(parts[3]), float(parts[4]), float(parts[5])
                out[serial] = (q, dx, dy, dz)
            except (ValueError, IndexError):
                continue
    return out

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
    """Add recalculated_x/y/z; rotate Dipole_X/Y/Z into local frame.

    Also preserves lab-frame copies of the dipoles (needed for rotation-
    invariant edge features such as the AMOEBA charge-dipole and dipole-
    dipole interaction terms).  Lab-frame copies are written to columns
    `Dipole_lab_X/Y/Z` and `PermDipole_lab_X/Y/Z`.
    """
    df = df.reset_index(drop=True)
    rx = np.full(len(df), np.nan)
    ry = np.full(len(df), np.nan)
    rz = np.full(len(df), np.nan)

    has_dipoles = all(c in df.columns for c in ("Dipole_X", "Dipole_Y", "Dipole_Z"))
    if has_dipoles:
        # Preserve lab-frame copies BEFORE rotation
        df["Dipole_lab_X"] = df["Dipole_X"].astype(float)
        df["Dipole_lab_Y"] = df["Dipole_Y"].astype(float)
        df["Dipole_lab_Z"] = df["Dipole_Z"].astype(float)
        ldx = np.full(len(df), np.nan)
        ldy = np.full(len(df), np.nan)
        ldz = np.full(len(df), np.nan)

    has_perm = all(c in df.columns for c in ("PermDipole_X", "PermDipole_Y", "PermDipole_Z"))
    if has_perm:
        df["PermDipole_lab_X"] = df["PermDipole_X"].astype(float)
        df["PermDipole_lab_Y"] = df["PermDipole_Y"].astype(float)
        df["PermDipole_lab_Z"] = df["PermDipole_Z"].astype(float)
        pdx = np.full(len(df), np.nan)
        pdy = np.full(len(df), np.nan)
        pdz = np.full(len(df), np.nan)

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

            if has_perm:
                pv = np.array([row.PermDipole_X, row.PermDipole_Y, row.PermDipole_Z])
                if not np.any(np.isnan(pv)):
                    pl = R.T @ pv
                    pdx[pos], pdy[pos], pdz[pos] = pl

    df["recalculated_x"] = rx
    df["recalculated_y"] = ry
    df["recalculated_z"] = rz

    if has_dipoles:
        df["Dipole_X"] = ldx
        df["Dipole_Y"] = ldy
        df["Dipole_Z"] = ldz

    if has_perm:
        df["PermDipole_X"] = pdx
        df["PermDipole_Y"] = pdy
        df["PermDipole_Z"] = pdz

    return df

# ════════════════════════════════════════════════════════════════════════════
# Rotation-invariant dipole scalars
# ════════════════════════════════════════════════════════════════════════════

def compute_dipole_invariants(df: pd.DataFrame, cutoff: float = 9.0) -> pd.DataFrame:
    """Add rotation-invariant scalar features derived from the (lab-frame)
    induced and permanent dipoles.  These replace the raw dipole components
    as node features, since a non-equivariant GAT cannot interpret raw
    components in a per-residue local frame.

    For each atom i we compute (per dipole μ ∈ {induced, permanent}):

        ‖μ_i‖                                — magnitude
        μ̂_i · ẑ_local                       — alignment with backbone normal
        μ_i · E_i                            — Coulomb-field projection,
                                                the polarisation-energy
                                                contribution

    where E_i = Σ_{j heavy, j≠i, |r_ij|<cutoff} q_j (r_j-r_i) / |r_ij|^3
    is the Coulomb field at i from neighbouring atomic charges.
    """
    from scipy.spatial import cKDTree

    df = df.reset_index(drop=True)
    n  = len(df)

    have_lab_ind  = all(c in df.columns for c in ("Dipole_lab_X","Dipole_lab_Y","Dipole_lab_Z"))
    have_lab_perm = all(c in df.columns for c in ("PermDipole_lab_X","PermDipole_lab_Y","PermDipole_lab_Z"))
    have_charge   = "atomic_charge" in df.columns

    coords = df[["x","y","z"]].to_numpy(dtype=float)

    # --- z-axis alignment uses local-frame dipoles (Dipole_Z is z-component
    #     in the per-residue local frame after compute_local_frame_coords).
    eps = 1e-12
    if have_lab_ind:
        mu_lab = df[["Dipole_lab_X","Dipole_lab_Y","Dipole_lab_Z"]].to_numpy(dtype=float)
        df["Dipole_norm"] = np.linalg.norm(mu_lab, axis=1)
        if all(c in df.columns for c in ("Dipole_X","Dipole_Y","Dipole_Z")):
            mu_loc = df[["Dipole_X","Dipole_Y","Dipole_Z"]].to_numpy(dtype=float)
            mag = np.linalg.norm(mu_loc, axis=1)
            df["Dipole_align_z"] = np.where(mag > eps, mu_loc[:,2] / np.maximum(mag, eps), 0.0)
        else:
            df["Dipole_align_z"] = 0.0
    if have_lab_perm:
        pmu_lab = df[["PermDipole_lab_X","PermDipole_lab_Y","PermDipole_lab_Z"]].to_numpy(dtype=float)
        df["PermDipole_norm"] = np.linalg.norm(pmu_lab, axis=1)
        if all(c in df.columns for c in ("PermDipole_X","PermDipole_Y","PermDipole_Z")):
            pmu_loc = df[["PermDipole_X","PermDipole_Y","PermDipole_Z"]].to_numpy(dtype=float)
            mag = np.linalg.norm(pmu_loc, axis=1)
            df["PermDipole_align_z"] = np.where(mag > eps, pmu_loc[:,2] / np.maximum(mag, eps), 0.0)
        else:
            df["PermDipole_align_z"] = 0.0

    # --- Coulomb field projection (uses lab-frame dipoles & charges)
    if have_charge and (have_lab_ind or have_lab_perm) and n > 1:
        q = df["atomic_charge"].to_numpy(dtype=float)
        tree  = cKDTree(coords)
        pairs = tree.query_pairs(r=cutoff, output_type="ndarray")  # (M,2) i<j
        E = np.zeros_like(coords)
        if pairs.size:
            i_idx = pairs[:,0]; j_idx = pairs[:,1]
            r_ij = coords[j_idx] - coords[i_idx]                     # j relative to i
            d2   = np.einsum("ij,ij->i", r_ij, r_ij)
            d3   = np.maximum(d2, eps) ** 1.5
            # contribution of j to E_i: q_j (r_j - r_i)/|r_ij|^3
            contrib = (r_ij.T / d3).T
            np.add.at(E, i_idx,  contrib * q[j_idx, None])
            # contribution of i to E_j: q_i (r_i - r_j)/|r_ij|^3 = -contrib * q_i
            np.add.at(E, j_idx, -contrib * q[i_idx, None])
        if have_lab_ind:
            df["Dipole_field_proj"] = np.einsum("ij,ij->i", mu_lab, E)
        if have_lab_perm:
            df["PermDipole_field_proj"] = np.einsum("ij,ij->i", pmu_lab, E)
    else:
        if have_lab_ind:
            df["Dipole_field_proj"] = 0.0
        if have_lab_perm:
            df["PermDipole_field_proj"] = 0.0

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

def find_completed_jobs(pdb_dir: str) -> list[tuple[str, str, str, str | None]]:
    d = Path(pdb_dir)
    jobs: list[tuple[str, str, str, str | None]] = []
    for uind in sorted(d.glob("*/*_final.uind")):
        pdb_id   = uind.parent.name
        pdb_path = uind.parent / f"{pdb_id}_final.pdb"
        if not pdb_path.exists():
            log.warning(f"  {pdb_id}: .uind found but .pdb missing – skipping")
            continue
        uperm = uind.parent / f"{pdb_id}_final.uperm"
        uperm_path = str(uperm) if uperm.exists() else None
        jobs.append((pdb_id, str(pdb_path), str(uind), uperm_path))
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

    for pdb_id, pdb_path, uind_path, uperm_path in jobs:
        log.info(f"Processing {pdb_id}")

        df = parse_pdb(pdb_path)
        if df.empty:
            log.warning(f"  {pdb_id}: empty PDB – skipping")
            continue

        # Induced dipoles  (Tinker .uind files for these jobs contain only
        # frame headers — no per-atom induced dipoles.  Zero-fill in that
        # case so downstream feature columns stay numeric.)
        dipoles = parse_uind(uind_path)
        if not dipoles:
            log.warning(f"  {pdb_id}: no dipoles parsed from {os.path.basename(uind_path)} — zero-filling")
            df["Dipole_X"] = 0.0
            df["Dipole_Y"] = 0.0
            df["Dipole_Z"] = 0.0
        else:
            df["Dipole_X"] = df["serial"].map(lambda s: dipoles.get(s, (0.0,)*3)[0])
            df["Dipole_Y"] = df["serial"].map(lambda s: dipoles.get(s, (0.0,)*3)[1])
            df["Dipole_Z"] = df["serial"].map(lambda s: dipoles.get(s, (0.0,)*3)[2])
            n_matched = df["serial"].isin(dipoles).sum()
            log.info(f"  Matched {n_matched}/{len(df)} atoms to dipole entries")

        # Permanent multipole: monopole charge + permanent dipole (lab frame)
        if uperm_path:
            uperm = parse_uperm(uperm_path)
        else:
            uperm = {}
        if uperm:
            df["atomic_charge"] = df["serial"].map(lambda s: uperm.get(s, (0.0,)*4)[0])
            df["PermDipole_X"]  = df["serial"].map(lambda s: uperm.get(s, (0.0,)*4)[1])
            df["PermDipole_Y"]  = df["serial"].map(lambda s: uperm.get(s, (0.0,)*4)[2])
            df["PermDipole_Z"]  = df["serial"].map(lambda s: uperm.get(s, (0.0,)*4)[3])
            n_uperm = df["serial"].isin(uperm).sum()
            log.info(f"  Matched {n_uperm}/{len(df)} atoms to .uperm entries")
        else:
            log.warning(f"  {pdb_id}: no .uperm — zero-filling atomic_charge")
            df["atomic_charge"] = 0.0
            df["PermDipole_X"]  = 0.0
            df["PermDipole_Y"]  = 0.0
            df["PermDipole_Z"]  = 0.0

        # Atom classification
        df["bb_sc"]      = df["name"].apply(classify_backbone_sidechain)
        df["atom_label"] = df.apply(lambda r: assign_atom_label(r["name"], r["bb_sc"]), axis=1)

        # Local frame
        df = compute_local_frame_coords(df)

        # Rotation-invariant dipole scalars (must follow local-frame step
        # because z-alignment uses local-frame dipole z-component, but
        # field projection uses lab-frame dipoles — both are present at
        # this point thanks to compute_local_frame_coords preserving copies)
        df = compute_dipole_invariants(df)

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
    # Notes on additions:
    #   • {Induced,Perm}Dipole_{norm,align_z,field_proj} — rotation-invariant
    #     scalar replacements for the raw 3-vector dipole components.
    #   • {x,y,z}_lab and {Induced,Perm}Dipole_lab_{X,Y,Z} are HELPER columns
    #     consumed by 03_create_datasets.py to build physical edge features
    #     (charge-dipole / dipole-dipole AMOEBA terms).  03 strips them from
    #     `data.x` so they are never used as raw node features.
    base_feature_cols = (
        residue_ohe_cols                          # 4 cols
        + ["atom_label"]                          # 1  (→ 9-class OHE downstream)
        + ["recalculated_x", "recalculated_y", "recalculated_z"]   # 3
        + ["Dipole_X", "Dipole_Y", "Dipole_Z"]   # 3  (local-frame induced dipoles)
        + ["atomic_charge"]                       # 1  (AMOEBA permanent monopole)
        + ["PermDipole_X", "PermDipole_Y", "PermDipole_Z"]   # 3 (local-frame permanent dipoles)
        + ["Dipole_norm", "Dipole_align_z", "Dipole_field_proj"]              # 3 (invariants)
        + ["PermDipole_norm", "PermDipole_align_z", "PermDipole_field_proj"]  # 3 (invariants)
        + ["Number of H-Bonds as donor", "Number of H-Bonds as acceptor"]  # 2
        + ["SASA_Value"]                          # 1
        # Helper columns for edge features (always dropped from data.x by 03)
        + ["x_lab", "y_lab", "z_lab"]                                                  # 3
        + ["Dipole_lab_X", "Dipole_lab_Y", "Dipole_lab_Z"]                             # 3
        + ["PermDipole_lab_X", "PermDipole_lab_Y", "PermDipole_lab_Z"]                 # 3
    )

    # Materialise lab-frame coord helper columns from the parsed PDB coords
    full_df["x_lab"] = full_df["x"].astype(float)
    full_df["y_lab"] = full_df["y"].astype(float)
    full_df["z_lab"] = full_df["z"].astype(float)

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
