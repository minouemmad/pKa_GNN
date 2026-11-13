"""
04b_parse_genz.py

Parse GenZpKa outputs written by 03c_genz_pka_jobs.py and turn per-pH
protonation populations into a per-residue physics pKa.

For every titrating residue r and grid pH:
    f_r(pH) = (protonated Boltzmann weight) / (total Boltzmann weight)
Fit the Hill / Henderson–Hasselbalch form
    f(pH) = 1 / (1 + 10**(n * (pH - pKa)))
and also report the linear-interpolated 50 % crossing.  Residues that never
cross 0.5 on the grid are censored (pKa reported at the grid edge, flagged).

Output: data/genz_pka.csv  — one row per titrating residue, joined with the
PKAD-R experimental pKa where available.  Columns pKa_phys / hill / censored
feed 05_prepare_features.py (Δ-learning target = expt − pKa_phys).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent
PDB_DIR  = PIPELINE_ROOT / "data/fixed_pdbs_v2"
MANIFEST = PIPELINE_ROOT.parent / "data/manifest.csv"
OUT_CSV  = PIPELINE_ROOT / "data/genz_pka.csv"

_RES_RE = re.compile(r"Residue\s+([A-Z]{3})(-?\d+)\s+Protonated Boltzmann:\s*([-+0-9.eE]+|NaN|Infinity)")
_TOT_RE = re.compile(r"Total Boltzmann:\s*([-+0-9.eE]+|NaN|Infinity)")
_CANON = {"ASH": "ASP", "GLH": "GLU", "HID": "HIS", "HIE": "HIS", "LYD": "LYS", "CYD": "CYS"}


def parse_log(log: Path) -> tuple[list[tuple[str, str, float]], float]:
    txt = log.read_text(errors="replace")
    res = [(m.group(1), m.group(2), float(m.group(3))) for m in _RES_RE.finditer(txt)]
    tot = _TOT_RE.findall(txt)
    return res, (float(tot[-1]) if tot else float("nan"))


def assign_chains(entries: list[tuple[str, str, float]], sel: pd.DataFrame) -> list[tuple[str, str, str, float]]:
    """GenZpKa prints name+number without chain; walk the (file-ordered)
    selection list and match numbers in order to recover chain IDs."""
    queue = list(zip(sel.chain.astype(str), sel.num.astype(str), sel.name))
    out, k = [], 0
    for name, num, w in entries:
        j = next((i for i in range(k, len(queue)) if queue[i][1] == num), None)
        if j is None:
            j = next((i for i in range(len(queue)) if queue[i][1] == num), None)
            if j is None:
                continue
        else:
            k = j + 1
        out.append((queue[j][0], num, _CANON.get(name, name), w))
    return out


def fit_hill(ph: np.ndarray, f: np.ndarray) -> tuple[float, float, bool, float]:
    """Return (pKa, hill, censored, pKa_interp)."""
    order = np.argsort(ph)
    ph, f = ph[order], np.clip(f[order], 1e-6, 1 - 1e-6)
    # linear-interpolated 0.5 crossing on the (monotone-in-practice) curve
    above = f >= 0.5
    if above.all():
        interp, censored = ph[-1], True
    elif (~above).all():
        interp, censored = ph[0], True
    else:
        i = int(np.argmax(~above))          # first point below 0.5
        x0, x1, y0, y1 = ph[i - 1], ph[i], f[i - 1], f[i]
        interp, censored = float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)), False
    # Hill fit in logit space: log10(f/(1-f)) = n*(pKa - pH)
    y = np.log10(f / (1 - f))
    w = f * (1 - f)                         # down-weight saturated points
    if censored or w.sum() < 1e-8:
        return float(interp), float("nan"), censored, float(interp)
    A = np.vstack([-ph, np.ones_like(ph)]).T * w[:, None]
    n, c = np.linalg.lstsq(A, y * w, rcond=None)[0]
    pka = c / n if n > 1e-3 else interp
    return float(pka), float(n), censored, float(interp)


def main() -> None:
    rows = []
    for d in sorted(p for p in PDB_DIR.iterdir() if p.is_dir()):
        sel_path = d / "genz_selection.csv"
        if not sel_path.exists():
            continue
        sel = pd.read_csv(sel_path, dtype={"chain": str, "num": str})
        # blank chain IDs → "A" (same convention as 05_prepare_features.parse_pdb)
        sel["chain"] = sel["chain"].fillna("").str.strip().replace("", "A")
        per_ph: dict[float, dict] = {}
        for log in sorted(d.glob("genz/pH*/genz.log")):
            if not (log.parent / "DONE").exists():
                continue
            ph = float(log.parent.name[2:])
            entries, total = parse_log(log)
            if not np.isfinite(total) or total <= 0:
                print(f"  {d.name} pH {ph}: bad total Boltzmann ({total})", file=sys.stderr)
                continue
            per_ph[ph] = {(c, n, r): w / total for c, n, r, w in assign_chains(entries, sel)}
        if len(per_ph) < 3:
            continue
        titr = sel[sel.titrating == 1]
        for _, t in titr.iterrows():
            canon = _CANON.get(t["name"], t["name"])
            key = (t.chain, t.num, canon)
            pts = [(ph, v[key]) for ph, v in per_ph.items() if key in v]
            if len(pts) < 3:
                continue
            ph_arr, f_arr = map(np.array, zip(*sorted(pts)))
            pka, hill, cens, interp = fit_hill(ph_arr, f_arr)
            row = dict(pdb_id=d.name, chain=t.chain, res_id=int(t.num), res_name=canon,
                       pKa_phys=pka, pKa_phys_interp=interp, hill=hill, censored=cens,
                       n_ph=len(pts))
            row.update({f"f_pH{p:g}": f for p, f in zip(ph_arr, f_arr)})
            rows.append(row)
        print(f"  {d.name}: {len(per_ph)} pH points, {len(titr)} titrating residues")

    if not rows:
        print("No completed GenZpKa runs found.")
        return
    df = pd.DataFrame(rows)
    man = pd.read_csv(MANIFEST)
    man["pdb_id"] = man["pdb_id"].str.upper()
    man["chain"] = man["chain"].astype(str).str.strip()
    df = df.merge(man[["pdb_id", "chain", "res_id", "res_name", "pka"]].rename(columns={"pka": "pKa_expt"}),
                  on=["pdb_id", "chain", "res_id", "res_name"], how="left")
    df.to_csv(OUT_CSV, index=False)
    lab = df.dropna(subset=["pKa_expt"])
    print(f"\n{len(df)} titrating residues, {len(lab)} with experimental pKa → {OUT_CSV}")
    if len(lab):
        e = lab.pKa_phys - lab.pKa_expt
        mu = lab.groupby("res_name").pKa_expt.transform("mean")
        print(f"  physics RMSE {np.sqrt((e**2).mean()):.2f}  MAE {e.abs().mean():.2f}  "
              f"| residue-type-mean RMSE {np.sqrt(((mu - lab.pKa_expt)**2).mean()):.2f}  "
              f"| censored {lab.censored.mean():.0%}")
        for r, g in lab.groupby("res_name"):
            eg = g.pKa_phys - g.pKa_expt
            print(f"    {r}: n={len(g):3d}  RMSE {np.sqrt((eg**2).mean()):.2f}  "
                  f"r={np.corrcoef(g.pKa_phys, g.pKa_expt)[0,1] if len(g) > 2 else float('nan'):.2f}")


if __name__ == "__main__":
    main()
