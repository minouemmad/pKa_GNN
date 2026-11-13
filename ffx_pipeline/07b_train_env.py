"""
07b_train_env.py

Equivariant message passing (PaiNN-style scalar + vector channels) over the
AMOEBA environment graphs from 05b_env_graphs.py, plus the ingredients current
protein-pKa predictors use (KaML, pKALM, PKAI, DeepKa):

  * Δ-learning target: pKa − offset, offset ∈ {model-compound pKa, PROPKA pKa,
    FFX many-body physics pKa}.                                 (--offset)
  * ESM-2 embedding of the titrating residue, projected with heavy dropout.
                                                                (--esm)
  * PROPKA shift as a site feature.                             (--propka-feat)
  * Pretraining on PROPKA shifts of every titratable site (labelled or not) in
    proteins outside the test / validation clusters, then fine-tuning on
    experiment.                                                 (--pretrain)
  * Seed ensembles and GNN + tree ensembles are formed by 08_aggregate.py.

Node inputs are rotation-invariant scalars plus two vector channels (permanent
and induced dipoles); messages use only distances and unit vectors, so the
network is rotation-equivariant by construction.

Evaluation: seq30 sequence-cluster folds from cv_common.make_folds (identical
for every model), inner validation split for early stopping, pooled per-residue
metrics vs the residue-type-mean baseline.

Run one fold (SGE array task) or all folds:
  python 07b_train_env.py --graphs F/env_graphs_all.pkl --extras F/site_extras.pkl \
      --offset model --esm --propka-feat --pretrain --fold $SGE_TASK_ID --results-dir R/exp
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cv_common import make_folds  # noqa: E402

_spec = importlib.util.spec_from_file_location("train07", HERE / "07_train.py")
t07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t07)

RES_TYPES = ["ASP", "GLU", "HIS", "LYS", "CYS", "TYR"]
FULL = {"ASP": "Aspartate", "GLU": "Glutamate", "HIS": "Histidine", "LYS": "Lysine",
        "CYS": "Cysteine", "TYR": "Tyrosine"}
ESM_DIM = 1280


# ── Data ─────────────────────────────────────────────────────────────────────

def _key(g: dict) -> tuple:
    return (g["pdb_id"], g["chain"], g["res_id"], g["res_name"])


def to_pyg(g: dict, offset_mode: str, gf_mean, gf_std, extras: dict,
           propka_feat: bool, target: str = "expt") -> Data | None:
    """target='expt': y = experimental pKa − offset.
    target='propka': y = PROPKA shift from the model pKa (pretraining; the PROPKA
    feature is masked so the task is not trivial)."""
    model_pka = t07._MODEL_PKA[FULL[g["res_name"]]]
    ex = extras.get(_key(g), {})
    propka = ex.get("propka", float("nan"))
    phys = g.get("pka_phys", float("nan"))
    nan = lambda v: v != v                                     # noqa: E731
    if target == "propka":
        if nan(propka):
            return None
        off, y = model_pka, propka
    else:
        y = g["y"]
        off = {"none": 0.0, "model": model_pka,
               "propka": model_pka if nan(propka) else propka,
               "hybrid": model_pka if nan(propka) or g["res_name"] in ("CYS", "TYR") else propka,
               "genz": model_pka if nan(phys) else phys}[offset_mode]
    res_oh = np.array([g["res_name"] == r for r in RES_TYPES], dtype=np.float32)
    show_pk = propka_feat and target == "expt" and not nan(propka)
    site = [0.0 if nan(phys) else (phys - model_pka) / 3.0, float(nan(phys)),
            (propka - model_pka) / 3.0 if show_pk else 0.0, float(not show_pk)]
    gfeat = np.concatenate([(g["graph_feat"] - gf_mean) / gf_std, res_oh, site]).astype(np.float32)
    e = ex.get("esm")
    esm = np.zeros(ESM_DIM, np.float32) if e is None else np.asarray(e, np.float32)
    d = Data(
        x=torch.from_numpy(g["x"]), pos=torch.from_numpy(g["pos"]),
        vec=torch.from_numpy(g["vec"]),                       # (N, 2, 3)
        edge_index=torch.from_numpy(g["edge_index"]),
        target_mask=torch.from_numpy(g["target_mask"].astype(bool)),
        gfeat=torch.from_numpy(gfeat).unsqueeze(0),
        esm=torch.from_numpy(esm).unsqueeze(0),
        y=torch.tensor([y - off], dtype=torch.float),
        offset=torch.tensor([off], dtype=torch.float),
        pka_phys=torch.tensor([phys], dtype=torch.float),
        pH=torch.tensor([g.get("pH", float("nan"))], dtype=torch.float),
    )
    d.PDB_ID, d.Chain_ID, d.Residue_Number = g["pdb_id"], g["chain"], g["res_id"]
    d.Residue_Name = FULL[g["res_name"]]
    return d


# ── Model ────────────────────────────────────────────────────────────────────

class RBF(nn.Module):
    def __init__(self, n: int, cutoff: float):
        super().__init__()
        self.register_buffer("mu", torch.linspace(0.0, cutoff, n))
        self.gamma = (n / cutoff) ** 2
        self.cutoff = cutoff

    def forward(self, d):
        env = 0.5 * (torch.cos(np.pi * d / self.cutoff) + 1.0) * (d < self.cutoff)
        return torch.exp(-self.gamma * (d[:, None] - self.mu) ** 2) * env[:, None]


class PaiNNLayer(nn.Module):
    """One PaiNN message + update block (Schütt et al. 2021)."""

    def __init__(self, h: int, n_rbf: int):
        super().__init__()
        self.h = h
        self.phi = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3 * h))
        self.w = nn.Linear(n_rbf, 3 * h)
        self.U = nn.Linear(h, h, bias=False)
        self.V = nn.Linear(h, h, bias=False)
        self.upd = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 3 * h))
        self.ln = nn.LayerNorm(h)

    def forward(self, s, v, edge_index, rbf, unit):
        src, dst = edge_index
        a, b, c = (self.phi(s)[src] * self.w(rbf)).split(self.h, dim=-1)
        ds = torch.zeros_like(s).index_add_(0, dst, a)
        msg_v = b[:, None, :] * v[src] + c[:, None, :] * unit[:, :, None]
        dv = torch.zeros_like(v).index_add_(0, dst, msg_v)
        s, v = s + ds, v + dv
        Uv, Vv = self.U(v), self.V(v)                        # mix channels (equivariant)
        vn = torch.sqrt((Vv ** 2).sum(1) + 1e-8)
        aa, ab, ac = self.upd(torch.cat([s, vn], -1)).split(self.h, dim=-1)
        s = self.ln(s + ab * (Uv * Vv).sum(1) + ac)
        v = v + aa[:, None, :] * Uv
        return s, v


class EnvPaiNN(nn.Module):
    def __init__(self, n_in: int, n_g: int, h: int = 64, layers: int = 3, n_rbf: int = 16,
                 cutoff: float = 4.5, dropout: float = 0.1, esm: bool = False, esm_proj: int = 32,
                 esm_dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, h), nn.SiLU(), nn.Linear(h, h))
        self.vec_in = nn.Linear(2, h, bias=False)            # 2 dipole channels → h
        self.rbf = RBF(n_rbf, cutoff)
        self.layers = nn.ModuleList(PaiNNLayer(h, n_rbf) for _ in range(layers))
        self.esm = (nn.Sequential(nn.LayerNorm(ESM_DIM), nn.Dropout(esm_dropout),
                                  nn.Linear(ESM_DIM, esm_proj), nn.SiLU()) if esm else None)
        n_head = h + n_g + (esm_proj if esm else 0)
        self.head = nn.Sequential(nn.Linear(n_head, h), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(h, 1))

    def forward(self, data):
        s = self.embed(data.x)
        v = self.vec_in(data.vec.transpose(1, 2))            # (N, 3, h)
        src, dst = data.edge_index
        rij = data.pos[src] - data.pos[dst]
        d = rij.norm(dim=-1).clamp_min(1e-6)
        rbf, unit = self.rbf(d), rij / d[:, None]
        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, rbf, unit)
        m = data.target_mask.float()[:, None]
        num = torch.zeros(data.num_graphs, s.shape[1], device=s.device).index_add_(0, data.batch, s * m)
        den = torch.zeros(data.num_graphs, 1, device=s.device).index_add_(0, data.batch, m)
        feats = [num / den.clamp_min(1.0), data.gfeat]
        if self.esm is not None:
            feats.append(self.esm(data.esm))
        return self.head(torch.cat(feats, -1)).view(-1)


# ── Training / CV ────────────────────────────────────────────────────────────

def pretrain(model, data, epochs: int, lr: float, wd: float, batch: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    loader = DataLoader(data, batch_size=batch, shuffle=True)
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for b in loader:
            opt.zero_grad()
            loss = loss_fn(model(b), b.y.view(-1))
            loss.backward()
            opt.step()
            tot += loss.item() * b.y.size(0)
            n += b.y.size(0)
        if ep % 5 == 0 or ep == epochs:
            print(f"    pretrain epoch {ep:3d}  loss={tot / n:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, required=True)
    ap.add_argument("--extras", type=Path, default=None, help="05c_site_extras.py pickle")
    ap.add_argument("--offset", choices=["none", "model", "propka", "hybrid", "genz"], default="model",
                    help="hybrid: PROPKA for ASP/GLU/HIS/LYS, model-compound pKa for CYS/TYR")
    ap.add_argument("--esm", action="store_true")
    ap.add_argument("--propka-feat", action="store_true")
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--fold", type=int, default=0, help="Run only this fold (1-based); 0 = all")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--group-split", choices=["seq30", "pdb", "residue"], default="seq30")
    ap.add_argument("--manifest", type=Path, default=HERE.parent / "data/manifest.csv")
    ap.add_argument("--res-types", nargs="+", default=RES_TYPES)
    ap.add_argument("--seed", type=int, default=42, help="model init / batching seed")
    ap.add_argument("--split-seed", type=int, default=42, help="CV split seed (keep fixed)")
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    blob = pickle.load(open(args.graphs, "rb"))
    allg = [g for g in blob["graphs"] if g["res_name"] in args.res_types]
    extras = pickle.load(open(args.extras, "rb")) if args.extras else {}
    lab = [g for g in allg if g["y"] == g["y"]]
    gf = np.stack([g["graph_feat"] for g in allg])
    gf_mean, gf_std = gf.mean(0), gf.std(0) + 1e-6       # unsupervised scaling only
    data = [to_pyg(g, args.offset, gf_mean, gf_std, extras, args.propka_feat) for g in lab]
    n_in, n_g = data[0].x.shape[1], data[0].gfeat.shape[1]
    print(f"{len(data)} labelled graphs ({len(allg)} sites total), node feats {n_in}, "
          f"site feats {n_g}, args={vars(args)}", flush=True)
    (args.results_dir / "models").mkdir(parents=True, exist_ok=True)

    cl = t07._seq30_clusters(args.manifest)
    groups = t07._make_groups(data, args.group_split, args.manifest)
    folds = make_folds(groups, args.folds, args.split_seed, args.val_frac)
    propka_of = {k: v.get("propka", float("nan")) for k, v in extras.items()}

    for fold, (itr, iva, ite) in enumerate(folds, 1):
        if args.fold and fold != args.fold:
            continue
        torch.manual_seed(args.seed * 100 + fold)
        np.random.seed(args.seed * 100 + fold)
        model = EnvPaiNN(n_in, n_g, args.hidden, args.layers, dropout=args.dropout, esm=args.esm)
        if args.pretrain:
            held = {groups[i] for i in list(iva) + list(ite)}
            pre = [to_pyg(g, "model", gf_mean, gf_std, extras, False, target="propka")
                   for g in allg if cl.get(g["pdb_id"], g["pdb_id"]) not in held]
            pre = [d for d in pre if d is not None]
            print(f"  fold {fold}: pretraining on {len(pre)} PROPKA-labelled sites", flush=True)
            pretrain(model, pre, args.pretrain_epochs, args.lr, args.wd, 32)
        tl = DataLoader([data[i] for i in itr], batch_size=args.batch, shuffle=True)
        vl = DataLoader([data[i] for i in iva], batch_size=64)
        sl = DataLoader([data[i] for i in ite], batch_size=64)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr * (0.5 if args.pretrain else 1.0),
                                weight_decay=args.wd)
        path = args.results_dir / "models" / f"fold_{fold}_seed{args.seed}.pth"
        preds, best_ep = t07.train_one_fold(model, tl, vl, sl, opt, nn.SmoothL1Loss(beta=0.5), path,
                                            epochs=args.epochs, patience=args.patience, min_epoch=30)
        tr_abs = [(d.Residue_Name, float(d.y + d.offset)) for d in (data[i] for i in itr)]
        tm = pd.DataFrame(tr_abs, columns=["r", "y"]).groupby("r").y.mean()
        for p in preds:
            k = (p["PDB_ID"], p["Chain_ID"], p["Residue_Number"],
                 {v: a for a, v in FULL.items()}[p["Residue_Name"]])
            p.update(fold=fold, best_epoch=best_ep, seed=args.seed,
                     Baseline_pKa=float(tm.get(p["Residue_Name"], np.mean([y for _, y in tr_abs]))),
                     PROPKA_pKa=propka_of.get(k, float("nan")),
                     Cluster=cl.get(p["PDB_ID"], p["PDB_ID"]))
        df = pd.DataFrame(preds)
        df.to_csv(args.results_dir / f"fold_{fold}.csv", index=False)
        fm = t07._metrics(df)
        print(f"  fold {fold}: test residue RMSE {fm['residue_RMSE']:.3f} "
              f"(baseline {fm['baseline_residue_RMSE']:.3f}) best_epoch {best_ep}", flush=True)
    print("FOLDS_DONE")


if __name__ == "__main__":
    main()
