"""
07c_train_trees.py

Tree / linear models on per-site descriptors — KaML ("are trees all you need?")
found gradient-boosted trees on hand-built features beat its GAT on PKAD-3, and
KaML-ESM / pKALM get RMSE ~0.7-0.8 from a small head on ESM embeddings alone.
Run on exactly the same seq30 folds as 07b so results are paired, and so
08_aggregate.py can ensemble them with the GNN.

Descriptors (all derived from the 05b environment graphs, i.e. AMOEBA based):
  site physics   potential at the titratable group from environment charges /
                 permanent / induced dipoles, total and by 0-4/4-6/6-8/8-10 Å shell
  packing        heavy-atom counts per shell, exposure asymmetry
  polar contacts N/O atoms within 3.5 Å of the group, nearest ±charged side chain,
                 counts of ±charged side-chain atoms within 4/6/8 Å
  self           induced-dipole magnitude and net charge on the residue
  + residue one-hot, PROPKA shift (--propka-feat), ESM-2 PCA (--esm-pca N)

Models: hgb (HistGradientBoosting), et (ExtraTrees), ridge (ridge on standardised
features); fixed hyper-parameters, no tuning on test folds.

  python 07c_train_trees.py --graphs F/env_graphs_all.pkl --extras F/site_extras.pkl \
      --model hgb --esm-pca 16 --propka-feat --results-dir R/hgb_esm_propka
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cv_common import make_folds  # noqa: E402

RES_TYPES = ["ASP", "GLU", "HIS", "LYS", "CYS", "TYR"]
FULL = {"ASP": "Aspartate", "GLU": "Glutamate", "HIS": "Histidine", "LYS": "Lysine",
        "CYS": "Cysteine", "TYR": "Tyrosine"}
MODEL_PKA = {"ASP": 3.94, "GLU": 4.4, "HIS": 6.6, "LYS": 10.4, "CYS": 8.55, "TYR": 10.07}
SHELLS = [(0, 4), (4, 6), (6, 8), (8, 10.01)]
# column indices in the 05b node-feature matrix
C, N, O, S, H, BB, TGT, GRP, PROT = range(9)
AA0 = 9
AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
QP, MUP, MUI, MUPR, MUIR, INVD, PHIQ = range(29, 36)
COULOMB = 332.0637


def descriptors(g: dict) -> tuple[np.ndarray, list[str]]:
    x, pos = g["x"], g["pos"]
    d = np.linalg.norm(pos, axis=1)
    env = x[:, TGT] == 0
    heavy = x[:, H] == 0
    aa = lambda n: x[:, AA0 + AA.index(n)] == 1                 # noqa: E731
    side = x[:, BB] == 0
    pos_atoms = env & side & (x[:, N] == 1) & (aa("LYS") | aa("ARG"))
    neg_atoms = env & side & (x[:, O] == 1) & (aa("ASP") | aa("GLU"))
    r = np.maximum(d, 0.5)
    phi_mp = -COULOMB * x[:, MUPR] / r ** 2
    phi_mi = -COULOMB * x[:, MUIR] / r ** 2
    f = list(g["graph_feat"])
    names = [f"gf{k}" for k in range(len(f))]                   # names: blob["graph_features"]
    for lo, hi in SHELLS:
        m = env & (d >= lo) & (d < hi)
        f += [(m & heavy).sum(), (m & (x[:, N] == 1)).sum(), (m & (x[:, O] == 1)).sum(),
              x[m, PHIQ].sum() * 100.0, phi_mp[m].sum(), phi_mi[m].sum(), x[m, QP].sum()]
        names += [f"{n}_{lo}-{int(hi)}" for n in ("heavy", "N", "O", "phi_q", "phi_mup", "phi_mui", "q")]
    grp = pos[x[:, GRP] == 1]
    polar = env & heavy & ((x[:, N] == 1) | (x[:, O] == 1))
    dg = lambda m: (np.linalg.norm(pos[m][:, None] - grp[None], axis=-1).min(1)   # noqa: E731
                    if m.any() and len(grp) else np.array([]))
    dpol, dpos, dneg = dg(polar), dg(pos_atoms), dg(neg_atoms)
    f += [(dpol < 3.5).sum(), (dpol < 3.0).sum(),
          dpos.min() if len(dpos) else 12.0, dneg.min() if len(dneg) else 12.0]
    names += ["polar_3.5", "polar_3.0", "dmin_pos", "dmin_neg"]
    for c in (4, 6, 8):
        f += [(dpos < c).sum(), (dneg < c).sum()]
        names += [f"npos_{c}", f"nneg_{c}"]
    eh = env & heavy
    unit = pos[eh] / np.maximum(d[eh], 1e-6)[:, None]
    f += [np.linalg.norm(unit.sum(0)) / max(1, eh.sum()),          # 0 buried … 1 exposed on one side
          x[~env, MUI].sum(), x[~env, MUI].max(), x[~env, QP].sum(), x[~env, PROT].max()]
    names += ["exposure_asym", "self_mu_ind_sum", "self_mu_ind_max", "self_q", "self_protonated"]
    return np.asarray(f, dtype=float), names


def make_model(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=400,
                                             max_depth=4, min_samples_leaf=8, l2_regularization=1.0,
                                             random_state=seed)
    if kind == "et":
        return ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.4,
                                   random_state=seed, n_jobs=-1)
    if kind == "ridge":
        return Ridge(alpha=30.0)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, required=True)
    ap.add_argument("--extras", type=Path, default=None)
    ap.add_argument("--model", choices=["hgb", "et", "ridge"], default="hgb")
    ap.add_argument("--offset", choices=["model", "propka", "hybrid"], default="model",
                    help="hybrid: PROPKA for ASP/GLU/HIS/LYS, model-compound pKa for CYS/TYR")
    ap.add_argument("--propka-feat", action="store_true")
    ap.add_argument("--esm-pca", type=int, default=0, help="ESM-2 PCA components (0 = no ESM)")
    ap.add_argument("--no-struct", action="store_true", help="drop structure descriptors (ESM-only model)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--manifest", type=Path, default=HERE.parent / "data/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("train07", HERE / "07_train.py")
    t07 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t07)

    graphs = [g for g in pickle.load(open(args.graphs, "rb"))["graphs"] if g["y"] == g["y"]]
    extras = pickle.load(open(args.extras, "rb")) if args.extras else {}
    key = lambda g: (g["pdb_id"], g["chain"], g["res_id"], g["res_name"])     # noqa: E731
    D = np.stack([descriptors(g)[0] for g in graphs])
    names = descriptors(graphs[0])[1]
    oh = np.array([[g["res_name"] == r for r in RES_TYPES] for g in graphs], float)
    mp = np.array([MODEL_PKA[g["res_name"]] for g in graphs])
    pk = np.array([extras.get(key(g), {}).get("propka", np.nan) for g in graphs], float)
    y = np.array([g["y"] for g in graphs])
    off = mp if args.offset == "model" else np.where(np.isnan(pk), mp, pk)
    if args.offset == "hybrid":
        off = np.where([g["res_name"] in ("CYS", "TYR") for g in graphs], mp, off)
    esm = None
    if args.esm_pca:
        esm = np.stack([np.zeros(1280, np.float32) if extras.get(key(g), {}).get("esm") is None
                        else np.asarray(extras[key(g)]["esm"], np.float32) for g in graphs])
    blocks = [oh] if args.no_struct else [D, oh]
    if args.propka_feat:
        blocks.append(np.column_stack([np.where(np.isnan(pk), 0.0, pk - mp), np.isnan(pk)]))
    X0 = np.column_stack(blocks)

    cl = t07._seq30_clusters(args.manifest)
    groups = [cl.get(g["pdb_id"], g["pdb_id"]) for g in graphs]
    recs = []
    for fold, (itr, iva, ite) in enumerate(make_folds(groups, args.folds, args.split_seed, args.val_frac), 1):
        tr = np.concatenate([itr, iva])            # no early stopping → use the inner-val rows too
        Xtr, Xte = X0[tr], X0[ite]
        if esm is not None:
            pca = PCA(n_components=args.esm_pca, random_state=0).fit(esm[tr])
            Xtr = np.column_stack([Xtr, pca.transform(esm[tr])])
            Xte = np.column_stack([Xte, pca.transform(esm[ite])])
        if args.model == "ridge":
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = make_model(args.model, args.seed).fit(Xtr, (y - off)[tr])
        pred = m.predict(Xte) + off[ite]
        tm = pd.Series(y[tr]).groupby([graphs[i]["res_name"] for i in tr]).mean()
        for j, i in enumerate(ite):
            g = graphs[i]
            recs.append(dict(PDB_ID=g["pdb_id"], Chain_ID=g["chain"], Residue_Number=g["res_id"],
                             Residue_Name=FULL[g["res_name"]], pH=np.nan, True_pKa=y[i],
                             Predicted_pKa=float(pred[j]), Physics_pKa=g.get("pka_phys", np.nan),
                             fold=fold, seed=args.seed, Baseline_pKa=float(tm.get(g["res_name"], y[tr].mean())),
                             PROPKA_pKa=pk[i], Cluster=groups[i]))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs)
    df.to_csv(args.results_dir / "predictions.csv", index=False)
    m = t07._metrics(df)
    print(f"{args.results_dir.name}: n={m['n_residues']} RMSE {m['residue_RMSE']:.3f} MAE {m['residue_MAE']:.3f} "
          f"baseline {m['baseline_residue_RMSE']:.3f} skill {m['skill_vs_baseline']:.3f}  ({X0.shape[1]} feats)")


if __name__ == "__main__":
    main()
