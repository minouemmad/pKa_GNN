"""
05c_site_extras.py

Per-site inputs that recent protein-pKa predictors rely on, for every
titratable residue of every FFX structure:

  * PROPKA3 pKa   — cheap empirical physics; used as a feature, as a Δ-learning
                    offset, and as the label for pretraining on unlabelled sites
                    (PKAI pretrains on PypKa, KaML on its own tree model).
  * ESM-2 embedding of the residue (650M, last layer) — pKALM / KaML-ESM show
                    sequence context alone reaches RMSE ~0.7-0.8.

Output: pickle {(PDB, chain, resseq, RES): {"propka": float|nan, "esm": float16[1280]}}

Run (in an SGE job):  python 05c_site_extras.py --mode rotopt
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import pickle
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prep05", HERE / "05_prepare_features.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)
log = logging.getLogger("05c")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def _write_heavy_pdb(df, path: Path) -> None:
    """Heavy atoms only, canonical residue names (PROPKA ignores H and does not
    know FFX titration names)."""
    with open(path, "w") as fh:
        k = 0
        for r in df.itertuples(index=False):
            n = r.name.lstrip("0123456789")
            if n.startswith("H") or n.startswith("D"):
                continue
            k += 1
            el = n[0]
            nm = r.name if len(r.name) == 4 else f" {r.name:<3s}"
            fh.write(f"ATOM  {k:5d} {nm} {r.resname:>3s} {r.chain:1s}{r.resseq:4d}    "
                     f"{r.x:8.3f}{r.y:8.3f}{r.z:8.3f}  1.00  0.00          {el:>2s}\n")
        fh.write("END\n")


def run_propka(df, pdb_id: str) -> dict:
    import propka.run
    out = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{pdb_id}.pdb"
        _write_heavy_pdb(df, p)
        try:
            mol = propka.run.single(str(p), optargs=["--quiet"], write_pka=False)
        except Exception as exc:                     # noqa: BLE001
            log.warning(f"  {pdb_id}: PROPKA failed: {exc}")
            return out
        conf = mol.conformations[next(iter(mol.conformations))] if "AVR" not in mol.conformations \
            else mol.conformations["AVR"]
        for g in conf.groups:
            rt = getattr(g, "residue_type", "")
            if rt in prep.TARGET_RESIDUES and g.atom is not None:
                out[(g.atom.chain_id.strip() or "A", int(g.atom.res_num), rt)] = float(g.pka_value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["genz", "rotopt"], default="rotopt")
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    pdb_dir = args.pdb_dir or (str(HERE / "data/fixed_pdbs_v2") if args.mode == "genz" else prep.PDB_DIR)
    out = Path(args.out or HERE / f"Graph_pKa/Features_env_{args.mode}/site_extras.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.no_esm:
        # Hugging Face copy of ESM-2 (safetensors); Meta's dl.fbaipublicfiles.com
        # certificate is expired, so fair-esm's own downloader fails.
        import torch
        from transformers import AutoTokenizer, EsmModel
        tok = AutoTokenizer.from_pretrained(f"facebook/{args.esm_model}")
        model = EsmModel.from_pretrained(f"facebook/{args.esm_model}", add_pooling_layer=False).eval()

    jobs = prep.find_completed_jobs(pdb_dir, args.mode, [])
    if args.only:
        jobs = [j for j in jobs if j[0].upper() in {o.upper() for o in args.only}]
    extras: dict = {}
    for pdb_id, pdb_path, *_ in jobs:
        df = prep.parse_pdb(pdb_path)
        pk = run_propka(df, pdb_id.upper())
        emb = {}
        if model is not None:
            for chain, cdf in df.groupby("chain", sort=False):
                res = cdf.drop_duplicates(["resseq", "icode"])[["resseq", "resname"]]
                seq = "".join(prep._AA3_TO_1.get(r, "X") for r in res.resname)
                if len(seq) < 2 or len(seq) > 2500:
                    continue
                with torch.no_grad():
                    rep = model(**tok(seq, return_tensors="pt")).last_hidden_state[0, 1:-1]
                assert rep.shape[0] == len(seq)
                for k, (rs, rn) in enumerate(zip(res.resseq, res.resname)):
                    if rn in prep.TARGET_RESIDUES:
                        emb[(str(chain), int(rs), rn)] = rep[k].numpy().astype(np.float16)
        sites = df[df.resname.isin(prep.TARGET_RESIDUES)][["chain", "resseq", "resname"]].drop_duplicates()
        for chain, rs, rn in sites.itertuples(index=False):
            k = (str(chain), int(rs), rn)
            extras[(pdb_id.upper(),) + k] = {"propka": pk.get(k, float("nan")), "esm": emb.get(k)}
        log.info(f"  {pdb_id}: {len(sites)} sites, propka {sum(k in pk for k in map(tuple, sites.values.tolist()))}")
    with open(out, "wb") as fh:
        pickle.dump(extras, fh)
    log.info(f"{len(extras)} sites → {out}")


if __name__ == "__main__":
    main()
