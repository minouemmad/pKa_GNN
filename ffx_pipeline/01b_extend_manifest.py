#!/usr/bin/env python3
"""
01b_extend_manifest.py

Add PKAD-R rows outside the "Main" classification to data/manifest.csv, but only
where the label and the structure describe the same molecule in the same state.

01_parse_and_download.py keeps "Main" rows only (653 sites, 159 proteins).  The
other 246 rows fall into three classes; what is added and what is not:

  Alt. pKa (mutant)   ADDED.  The listed PDB entry is the mutant's own structure
                      (checked below: the residue at the mutated position must be
                      the mutant amino acid).  These are new sites.
  Alt. pKa (state)    ADDED only if the row's PDB entry is a structure of that
                      state and the site is not already a Main site (oxy-Hb 1HHO,
                      MbCO 1MBC, apo-Mb 1BVC, oxidised thioredoxin 1ERU).  Rows
                      whose label belongs to a state with no structure (7XJD N
                      intermediate) or that reuse the Main structure for another
                      state (apo-calbindin on holo 4ICB, apo-Mb on 1YMB) are NOT
                      added: the structure would contradict the label.
  Alt. pKa            NOT added.  108 of 117 are second measurements of sites
                      that already have a Main value (adding them double-weights
                      those sites with conflicting labels); the other 9 are the
                      1U2U leucine zipper, which PKAD-R itself says must not be
                      used together with 1FMH.

Still excluded everywhere (as in 01): N-/C-termini, residues absent from the
structure, and pKa values reported as bounds or approximations (">11", "~11").
Treating a bound as a number would bias the labels.

Repeated measurements of the same added site are averaged.

The added rows carry their PKAD-R 30 % / 90 % sequence-identity cluster ids, so
the cluster-grouped CV keeps every variant of a protein in the same fold.

Run:  python 01b_extend_manifest.py [--dry]
"""
import argparse
import re
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CSV_PATH = REPO / "1-PKAD-R-2025-09-03.csv"
MANIFEST = REPO / "data/manifest.csv"
RAW_DIR = REPO / "data/raw_pdbs"
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

TARGET = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"}
# state rows whose PDB entry is NOT a structure of the labelled state
STATE_EXCLUDE = {"7XJD", "4ICB", "1YMB"}
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
         "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
         "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def num(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def candidates() -> pd.DataFrame:
    d = pd.read_csv(CSV_PATH)
    d["pk"] = d["Expt. pKa"].map(num)
    d["W"] = d["Warning"].fillna("")
    d["PDB"] = d["PDB"].str.upper().str.strip()
    cls = d["pKa Classification"]
    ok = d["ResName"].isin(TARGET) & d["pk"].notna() & ~d["W"].str.contains("NOT exist|C/N-term")
    key = lambda f: list(zip(f.PDB, f.Chain, f["ResID in PDB"].astype(str), f.ResName))
    main_keys = set(key(d[cls == "Main"]))
    mut = d[ok & (cls == "Alt. pKa (mutant)")].copy()
    mut["source"] = "mutant"
    st = d[ok & (cls == "Alt. pKa (state)") & ~d.PDB.isin(STATE_EXCLUDE)].copy()
    st["source"] = "state"
    add = pd.concat([mut, st])
    add = add[[k not in main_keys for k in key(add)]]
    return add


def download(pdb_id: str) -> bool:
    out = RAW_DIR / f"{pdb_id}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return True
    try:
        urllib.request.urlretrieve(RCSB_URL.format(pdb_id=pdb_id), out)
        time.sleep(0.3)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed {pdb_id}: {exc}")
        out.unlink(missing_ok=True)
        return False


def pdb_residues(pdb_id: str) -> dict:
    """{(chain, resseq): resname} from ATOM/HETATM records of the first model."""
    res = {}
    for line in open(RAW_DIR / f"{pdb_id}.pdb"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")):
            res.setdefault((line[21], line[22:26].strip()), line[17:20].strip())
    return res


def mutation_ok(row, res) -> tuple:
    """The PDB must carry the mutant residue.  'R158C (R136C in PDB)' → use the
    in-PDB numbering; 'E75P/P76H' → every listed substitution must match."""
    txt = str(row["Mut. Pos."])
    m = re.search(r"\(([^)]*) in PDB\)", txt)
    txt = m.group(1) if m else txt
    muts = re.findall(r"([A-Z])(\d+)([A-Z])", txt)
    if not muts:
        return False, "unparsed mutation"
    for _, pos, new in muts:
        got = res.get((row["Chain"], pos))
        if got != THREE.get(new):
            return False, f"{pos} is {got}, expected {THREE.get(new)}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="report only; change nothing")
    args = ap.parse_args()

    add = candidates()
    print(f"candidate rows: {len(add)}  ({add.source.value_counts().to_dict()}), "
          f"{add.PDB.nunique()} PDB entries")

    keep, dropped = [], []
    for pdb_id, g in add.groupby("PDB"):
        if not args.dry and not download(pdb_id):
            dropped += [(pdb_id, "download failed", len(g))]
            continue
        if not (RAW_DIR / f"{pdb_id}.pdb").exists():
            keep.append(g)   # dry run without the file: cannot check further
            continue
        res = pdb_residues(pdb_id)
        for idx, row in g.iterrows():
            site = res.get((row["Chain"], str(row["ResID in PDB"])))
            if site != row["ResName"]:
                dropped.append((pdb_id, f"site {row['ResName']}{row['ResID in PDB']} is {site} in PDB", 1))
                continue
            if row["source"] == "mutant":
                good, why = mutation_ok(row, res)
                if not good:
                    dropped.append((pdb_id, f"structure is not the mutant ({why})", 1))
                    continue
            keep.append(add.loc[[idx]])
    add = pd.concat(keep) if keep else add.iloc[:0]
    for d in dropped:
        print("  dropped:", d)

    out = pd.DataFrame({
        "pkad_index": add["Index"], "protein_name": add["Protein Name"], "pdb_id": add["PDB"],
        "chain": add["Chain"], "res_id": add["ResID in PDB"], "res_name": add["ResName"],
        "pka": add["pk"], "seq_id_30": add["Sequence Identity > 30%"],
        "seq_id_90": add["Sequence Identity > 90%"], "uncertainty": add["Expt. Uncertainty"],
        "method": add["Expt. Method"], "source": add["source"]})
    site = ["pdb_id", "chain", "res_id", "res_name"]
    n_rep = int(out.duplicated(site).sum())
    spread = out.groupby(site)["pka"].agg(lambda s: s.max() - s.min())
    out = out.groupby(site, as_index=False, sort=False).agg(
        {c: ("mean" if c == "pka" else "first") for c in out.columns if c not in site})[list(out.columns)]
    out["pka"] = out["pka"].round(3)
    print(f"averaged {n_rep} repeated measurements (largest spread {spread.max():.2f} pKa units)")

    man = pd.read_csv(MANIFEST)
    if "source" not in man.columns:
        man["source"] = "main"
    man = man[man["source"] == "main"]
    new = pd.concat([man, out], ignore_index=True)
    assert not new.duplicated(site).any()
    print(f"manifest: {len(man)} main + {len(out)} added = {len(new)} sites, "
          f"{man.pdb_id.nunique()} → {new.pdb_id.nunique()} proteins")
    print(out.groupby(["source", "protein_name"]).size().to_string())
    print("added by residue type:", out.res_name.value_counts().to_dict())
    if args.dry:
        return
    backup = MANIFEST.with_name("manifest_main.csv")
    if not backup.exists():
        shutil.copy(MANIFEST, backup)
    new.to_csv(MANIFEST, index=False)
    print(f"wrote {MANIFEST}  (Main-only copy: {backup})")


if __name__ == "__main__":
    main()
