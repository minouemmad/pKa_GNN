"""
05b_env_graphs.py

Environment graphs for the step-6 model.  The original Graph_pKa graph holds
only the titratable residue's own ~10-25 atoms, so the environment — which is
what shifts a pKa — enters only as a handful of counts.  Here each labelled
residue becomes a graph of ALL atoms within R_ENV Å of its titratable group,
carrying AMOEBA permanent + induced multipole information.

Per graph (saved as plain numpy, so building needs no torch):
  x        (N, F)   rotation-invariant node scalars (see NODE_FEATURES)
  pos      (N, 3)   coordinates relative to the titratable-group centre,
                    in the residue backbone frame (CA→C, CA-C-O plane)
  vec      (N, 2, 3) permanent dipole, induced dipole (same frame)
  edge_index (2, E) radius graph (R_EDGE Å), both directions
  edge_dist  (E,)
  target_mask (N,)  1 for atoms of the titratable residue (readout pool)
  graph_feat (G,)   site-level physics: electrostatic potential at the centre
                    from environment charges / permanent dipoles / induced
                    dipoles (target residue excluded), counts, SASA proxy
  y, pka_phys, res_name, pdb_id, chain, res_id

Inputs are the same FFX outputs 05_prepare_features.py reads (--mode genz,
rotopt or titrate), plus optional data/genz_pka.csv for pka_phys.

Run:  python 05b_env_graphs.py --mode genz
      python 05b_env_graphs.py --mode rotopt --pdb-dir data/fixed_pdbs   # old data
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prep05", HERE / "05_prepare_features.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)   # reuse parsers, frame, job discovery, label alignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("05b")

R_ENV  = 10.0   # Å, environment sphere around the titratable-group centre
R_EDGE = 4.5    # Å, radius-graph cutoff
COULOMB = 332.0637  # kcal·Å/(mol·e²)

TITR_GROUP = {   # atoms defining the titratable group centre
    "ASP": ("CG", "OD1", "OD2"), "GLU": ("CD", "OE1", "OE2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"), "LYS": ("NZ",),
    "CYS": ("SG",), "TYR": ("CZ", "OH"),
}
ELEMENTS = ["C", "N", "O", "S", "H"]
AA20 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
NODE_FEATURES = (
    [f"el_{e}" for e in ELEMENTS] + ["backbone", "in_target", "in_group", "res_protonated"]
    + [f"aa_{a}" for a in AA20]
    + ["q_perm", "mu_perm_norm", "mu_ind_norm", "mu_perm_radial", "mu_ind_radial",
       "inv_dist_center", "phi_contrib_q"]
)
GRAPH_FEATURES = ["phi_q", "phi_mu_perm", "phi_mu_ind", "phi_total",
                  "n_heavy_6A", "n_heavy_10A", "n_charged_pos_10A", "n_charged_neg_10A",
                  # burial / desolvation beyond the 10 Å graph (what PROPKA's desolvation term
                  # sees): heavy-atom counts to 20 Å, Coulomb-field Born integrals Σ1/r⁴, Σ1/r⁶,
                  # long-range charge potential from 10-20 Å, one-sided exposure (0 buried … 1)
                  "n_heavy_12A", "n_heavy_15A", "n_heavy_20A", "born_r4", "born_r6",
                  "phi_q_10_20A", "exposure_15A"]


def _element(name: str) -> str:
    n = name.lstrip("0123456789")
    return n[0] if n and n[0] in "CNOSH" else "C"


def _residue_protonation(df: pd.DataFrame) -> pd.Series:
    prot = prep.detect_protonation(df)
    return df.apply(lambda r: prot.get((r["chain"], int(r["resseq"])), np.nan), axis=1)


def build_site_graph(df: pd.DataFrame, chain: str, resseq: int, resname: str) -> dict | None:
    res = df[(df.chain == chain) & (df.resseq == resseq)]
    grp = res[res.name.isin(TITR_GROUP[resname])]
    bb = {n: res.loc[res.name == n, ["x", "y", "z"]].values for n in ("CA", "C", "O")}
    if grp.empty or any(len(v) == 0 for v in bb.values()):
        return None
    center = grp[["x", "y", "z"]].values.mean(0)
    R, _ = prep.build_local_frame(bb["CA"][0], bb["C"][0], bb["O"][0])

    xyz = df[["x", "y", "z"]].values
    d_c = np.linalg.norm(xyz - center, axis=1)
    sel = np.where(d_c <= R_ENV)[0]
    sub = df.iloc[sel]
    rel = (xyz[sel] - center) @ R            # == R.T @ v for each row
    dist = d_c[sel]
    unit = rel / np.maximum(dist, 1e-6)[:, None]

    mu_p = sub[["Perm_DipX", "Perm_DipY", "Perm_DipZ"]].values.astype(float) @ R
    mu_i = sub[["Dipole_X", "Dipole_Y", "Dipole_Z"]].values.astype(float) @ R
    q = sub["Perm_Charge"].values.astype(float)
    mu_p, mu_i, q = np.nan_to_num(mu_p), np.nan_to_num(mu_i), np.nan_to_num(q)

    in_target = ((sub.chain == chain) & (sub.resseq == resseq)).values.astype(float)
    in_group = in_target * sub.name.isin(TITR_GROUP[resname]).values
    el = np.array([[_element(n) == e for e in ELEMENTS] for n in sub.name], dtype=float)
    aa = np.array([[r == a for a in AA20] for r in sub.resname], dtype=float)
    backbone = sub.name.isin(prep.BACKBONE_HEAVY | prep.BACKBONE_H).values.astype(float)
    rprot = np.nan_to_num(sub["res_protonated"].values.astype(float), nan=0.0)

    # Electrostatic potential at the centre from everything outside the target residue.
    env = in_target == 0
    r = np.maximum(dist, 0.5)
    phi_q_atom = COULOMB * q / r
    # dipole potential: μ·(r_center − r_atom)/|·|³ = −μ·unit/r²
    phi_mp_atom = -COULOMB * (mu_p * unit).sum(1) / r ** 2
    phi_mi_atom = -COULOMB * (mu_i * unit).sum(1) / r ** 2
    phi_q, phi_mp, phi_mi = (float(a[env].sum()) for a in (phi_q_atom, phi_mp_atom, phi_mi_atom))

    heavy = el[:, 4] == 0
    chg_res = sub.assign(prot=rprot)[env & heavy].drop_duplicates(["chain", "resseq"])
    n_pos = int(((chg_res.resname.isin(["LYS", "ARG"])) |
                 ((chg_res.resname == "HIS") & (chg_res.prot == 1))).sum())
    n_neg = int(((chg_res.resname.isin(["ASP", "GLU"])) & (chg_res.prot == 0)).sum())

    x = np.column_stack([
        el, backbone, in_target, in_group, rprot, aa,
        q, np.linalg.norm(mu_p, axis=1), np.linalg.norm(mu_i, axis=1),
        (mu_p * unit).sum(1), (mu_i * unit).sum(1),
        1.0 / r, phi_q_atom * env / 100.0,
    ]).astype(np.float32)

    # radius graph
    D = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
    src, dst = np.where((D <= R_EDGE) & (D > 0))
    # long-range burial descriptors over the whole structure (target residue excluded)
    is_h = df.name.map(lambda n: _element(n) == "H").values
    tgt_all = ((df.chain == chain) & (df.resseq == resseq)).values
    hv = ~is_h & ~tgt_all
    dh = np.maximum(d_c[hv], 2.0)
    far = ~tgt_all & (d_c > R_ENV) & (d_c <= 20.0)
    q_all = np.nan_to_num(df["Perm_Charge"].values.astype(float))
    m15 = hv & (d_c <= 15.0)
    u15 = (xyz[m15] - center) / np.maximum(d_c[m15], 1e-6)[:, None]
    extra = [int((dh <= 12).sum()), int((dh <= 15).sum()), int((dh <= 20).sum()),
             float((dh[dh <= 20] ** -4).sum()), float((dh[dh <= 20] ** -6).sum()),
             float(COULOMB * (q_all[far] / d_c[far]).sum()),
             float(np.linalg.norm(u15.sum(0)) / max(1, m15.sum()))]

    return dict(
        x=x, pos=rel.astype(np.float32), vec=np.stack([mu_p, mu_i], 1).astype(np.float32),
        edge_index=np.vstack([src, dst]).astype(np.int64), edge_dist=D[src, dst].astype(np.float32),
        target_mask=in_target.astype(bool),
        graph_feat=np.array([phi_q, phi_mp, phi_mi, phi_q + phi_mp + phi_mi,
                             int((heavy & (dist <= 6)).sum()), int(heavy.sum()), n_pos, n_neg]
                            + extra, dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["genz", "rotopt", "titrate"], default="genz")
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--raw-pdb-dir", default=prep.RAW_PDB_DIR)
    ap.add_argument("--manifest", default=prep.MANIFEST)
    ap.add_argument("--genz-csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--all-sites", action="store_true",
                    help="Also write graphs for unlabelled titratable sites (y = NaN); used for\n"
                         "pretraining on physics / PROPKA labels")
    args = ap.parse_args()
    pdb_dir = args.pdb_dir or (str(HERE / "data/fixed_pdbs_v2") if args.mode == "genz" else prep.PDB_DIR)
    out = Path(args.out or f"Graph_pKa/Features_env_{args.mode}/env_graphs.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    genz_csv = args.genz_csv or (str(HERE / "data/genz_pka.csv") if args.mode == "genz" else None)
    phys = {}
    if genz_csv and Path(genz_csv).exists():
        g = pd.read_csv(genz_csv, dtype={"chain": str})
        phys = {(p, c, int(n), r): v for p, c, n, r, v in
                zip(g.pdb_id, g.chain, g.res_id, g.res_name, g.pKa_phys)}

    lookup = prep._build_pka_lookup_aligned(pd.read_csv(args.manifest), Path(pdb_dir),
                                            Path(args.raw_pdb_dir), log)
    jobs = prep.find_completed_jobs(pdb_dir, args.mode, prep.DEFAULT_PHS if args.mode == "titrate" else [])
    if args.only:
        jobs = [j for j in jobs if j[0].upper() in {o.upper() for o in args.only}]
    graphs = []
    for pdb_id, pdb_path, uind_path, uperm_path, ph in jobs:
        df = prep.parse_pdb(pdb_path)
        dip = prep.parse_uind(uind_path)
        for k, c in enumerate(("Dipole_X", "Dipole_Y", "Dipole_Z")):
            df[c] = df.serial.map(lambda s, k=k: dip.get(s, (np.nan,) * 3)[k])
        perms = prep.parse_uperm(uperm_path) if uperm_path else {}
        for k, c in enumerate(prep.PERM_COLS):
            df[c] = df.serial.map(lambda s, k=k: perms.get(s, (np.nan,) * 10)[k])
        if not perms:
            log.warning(f"  {pdb_id}: no .uperm — permanent multipole features are zero")
        df["res_protonated"] = _residue_protonation(df)
        n0 = len(graphs)
        sites = df[df.resname.isin(prep.TARGET_RESIDUES)][["chain", "resseq", "resname"]].drop_duplicates()
        for chain, resseq, resname in sites.itertuples(index=False):
            y = lookup.get((pdb_id.upper(), str(chain), int(resseq), resname))
            if y is None:
                if not args.all_sites:
                    continue
                y = float("nan")
            gph = build_site_graph(df, chain, int(resseq), resname)
            if gph is None:
                log.warning(f"  {pdb_id} {chain}{resseq} {resname}: missing backbone/group atoms")
                continue
            gph.update(y=float(y), pdb_id=pdb_id.upper(), chain=str(chain), res_id=int(resseq),
                       res_name=resname, pH=float(ph) if ph is not None else float("nan"),
                       pka_phys=float(phys.get((pdb_id.upper(), str(chain), int(resseq), resname), np.nan)))
            graphs.append(gph)
        log.info(f"  {pdb_id}{'' if ph is None else f' pH {ph}'}: {len(graphs) - n0} site graphs")

    with open(out, "wb") as fh:
        pickle.dump({"graphs": graphs, "node_features": NODE_FEATURES,
                     "graph_features": GRAPH_FEATURES, "r_env": R_ENV, "r_edge": R_EDGE}, fh)
    if graphs:
        n = np.array([g["x"].shape[0] for g in graphs]); e = np.array([g["edge_index"].shape[1] for g in graphs])
        log.info(f"{len(graphs)} graphs → {out}  (atoms/graph median {int(np.median(n))}, "
                 f"edges/graph median {int(np.median(e))})")


if __name__ == "__main__":
    main()
