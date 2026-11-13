"""
04c_mc_titration.py

Titration curves from ONE FFX many-body energy expansion, by pH replica-exchange
Monte Carlo over the rotamer × protonation-state model (the MCCE idea, with
AMOEBA / GK energies instead of Poisson-Boltzmann).

Why: GenZpKa enumerates every conformation inside an energy window at one pH
(~1 h per pH for 4PTI with a 4 kcal/mol window, and it does not recompute the
pH term when an energy restart is reused).  But the expansion
        E(x) = Σ_i self(i, r_i) + Σ_{i<j} pair(i, r_i, j, r_j)
is pH independent except for FFX's acidostat term on deprotonated states,
        bias(pH) = ln10·RT·(pKa_model − pH) − F_mod          (TitrationUtils)
so the energies stored at pH_ref give every other pH exactly:
        self_pH(i, r) = self_ref(i, r) + ln10·RT·(pH_ref − pH)·[r deprotonated]
Sampling that model with Gibbs moves (one residue's full rotamer/protonation
distribution at a time) plus pH replica exchange takes seconds, has no energy
window, and gives all pH values at once.

Inputs:  --restart  FFX energy restart (Self / Pair lines; Triple lines ignored)
         --log      the FFX log of the run that computed the self energies
                    (gives the residue + protonation-state name of every rotamer)
Output:  CSV with the protonated fraction per titrating residue per pH and a
         fitted pKa / Hill coefficient (same fit as 04b_parse_genz.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LN10_RT = np.log(10.0) * 1.98720425864083e-3 * 298.15        # kcal/mol per pH unit
BETA = 1.0 / (1.98720425864083e-3 * 298.15)
PROTONATED = {"ASH": "ASP", "GLH": "GLU", "HIS": "HIS", "LYS": "LYS", "CYS": "CYS"}
DEPROTONATED = {"ASP": "ASP", "GLU": "GLU", "HID": "HIS", "HIE": "HIS", "LYD": "LYS", "CYD": "CYS"}
_SELF_LOG = re.compile(r"^\s*Self\s+(\S?)(-?\d+)-([A-Z]{3})\s+(\d+)\s*:")
_SELF = re.compile(r"^Self (\d+) (\d+):\s*(\S+)")
_PAIR = re.compile(r"^Pair (\d+) (\d+), (\d+) (\d+):\s*(\S+)")


def read_names(logs: list[Path], lr: str | None = None) -> list[list[tuple[str, int, str]]]:
    """names[i][ri] = (chain, resnum, state name) in restart index order.

    Self-energy lines are merged over all logs (a run resumed after preemption
    only prints the energies it still had to compute).  Residue order is the
    --lR order when given (FFX indexes residues in that order), otherwise the
    order of first appearance."""
    res: dict[tuple, dict] = {}
    for log in logs:
        with open(log, errors="replace") as fh:
            for line in fh:
                m = _SELF_LOG.match(line)
                if m:
                    ch, num, nm, ri = m.group(1) or "A", int(m.group(2)), m.group(3), int(m.group(4))
                    res.setdefault((ch, num), {})[ri] = (ch, num, nm)
    if not res:
        raise SystemExit("no ' Self  <res> <rot> :' lines found in the log(s)")
    order = list(res)
    if lr:
        want = [(t[0] if t[0].isalpha() else "A", int(t.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")))
                for t in lr.split(",")]
        order = [k for k in want if k in res]
    return [[res[k].get(ri) for ri in range(max(res[k]) + 1)] for k in order]


def read_restart(path: Path):
    selfs, pairs = {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _SELF.match(line)
            if m:
                selfs[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
                continue
            m = _PAIR.match(line)
            if m:
                pairs.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                              float(m.group(5))))
    return selfs, pairs


def build(names, selfs, pairs, big: float = 1e4):
    """Flatten to global state indices.  Missing self energies (pruned before
    being computed) and NaN / missing-rotamer entries are forbidden states."""
    offs, n_state = [], 0
    for res in names:
        offs.append(n_state)
        n_state += len(res)
    E1 = np.full(n_state, big)
    dep = np.zeros(n_state, bool)
    prot = np.zeros(n_state, bool)
    for i, res in enumerate(names):
        for ri, nm in enumerate(res):
            e = selfs.get((i, ri))
            if nm is None or e is None or not np.isfinite(e):
                continue
            E1[offs[i] + ri] = e
            dep[offs[i] + ri] = nm[2] in DEPROTONATED
            prot[offs[i] + ri] = nm[2] in PROTONATED
    E2 = np.zeros((n_state, n_state), np.float32)
    for i, ri, j, rj, e in pairs:
        if i >= len(offs) or j >= len(offs):
            continue
        a, b = offs[i] + ri, offs[j] + rj
        e = big if not np.isfinite(e) else e
        E2[a, b] = E2[b, a] = e
    return np.array(offs + [n_state]), E1, E2, dep, prot


def run_mc(offs, E1, E2, dep, ph_ref: float, ph_grid: np.ndarray, sweeps: int, burn: int,
           seed: int = 0):
    rng = np.random.default_rng(seed)
    n_res, K = len(offs) - 1, len(ph_grid)
    shift = LN10_RT * (ph_ref - ph_grid)                       # (K,) added to deprotonated states
    # start every replica in its lowest self-energy state
    cur = np.stack([np.array([offs[i] + np.argmin(E1[offs[i]:offs[i + 1]] + s * dep[offs[i]:offs[i + 1]])
                              for i in range(n_res)]) for s in shift])          # (K, n_res)
    occ = np.zeros((K, len(E1)))
    acc = np.zeros(K - 1); tried = 0
    kk = np.arange(K)
    for sweep in range(sweeps + burn):
        for i in rng.permutation(n_res):
            sl = slice(offs[i], offs[i + 1])
            others = np.delete(cur, i, axis=1)                                   # (K, n_res-1)
            h = E1[sl][None, :] + shift[:, None] * dep[sl][None, :] \
                + E2[sl][:, others].sum(-1).T                                    # (K, n_i)
            p = np.exp(-BETA * (h - h.min(1, keepdims=True)))
            c = p.cumsum(1)
            pick = (c < rng.random(K)[:, None] * c[:, -1:]).sum(1)
            cur[:, i] = offs[i] + np.minimum(pick, p.shape[1] - 1)
        # pH replica exchange between neighbours (alternating pairs)
        nd = dep[cur].sum(1)
        for a in range(sweep % 2, K - 1, 2):
            d = np.log(10.0) * (ph_grid[a + 1] - ph_grid[a]) * (nd[a + 1] - nd[a])
            if d <= 0 or rng.random() < np.exp(-d):
                cur[[a, a + 1]] = cur[[a + 1, a]]
                nd[[a, a + 1]] = nd[[a + 1, a]]
                if sweep >= burn:
                    acc[a] += 1
        if sweep >= burn:
            tried += 0.5
            occ[kk[:, None], cur] += 1
    return occ / sweeps, acc / max(tried, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", type=Path, required=True)
    ap.add_argument("--log", type=Path, nargs="+", required=True)
    ap.add_argument("--lr", default=None, help="the --lR list given to FFX (fixes residue order)")
    ap.add_argument("--ph-ref", type=float, default=7.0, help="pH of the FFX run that wrote the restart")
    ap.add_argument("--ph-min", type=float, default=-2.0)
    ap.add_argument("--ph-max", type=float, default=16.0)
    ap.add_argument("--ph-step", type=float, default=0.5)
    ap.add_argument("--sweeps", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("p04b", HERE / "04b_parse_genz.py")
    p04b = importlib.util.module_from_spec(spec); spec.loader.exec_module(p04b)

    names = read_names(args.log, args.lr)
    selfs, pairs = read_restart(args.restart)
    offs, E1, E2, dep, prot = build(names, selfs, pairs)
    ph = np.arange(args.ph_min, args.ph_max + 1e-9, args.ph_step)
    occ, acc = run_mc(offs, E1, E2, dep, args.ph_ref, ph, args.sweeps, args.burn, args.seed)
    print(f"{len(names)} residues, {len(E1)} states, {len(pairs)} pair energies, {len(ph)} pH replicas; "
          f"replica-exchange acceptance min {acc.min():.2f} mean {acc.mean():.2f}")
    rows = []
    for i, res in enumerate(names):
        sl = slice(offs[i], offs[i + 1])
        if not (dep[sl].any() and prot[sl].any()):
            continue
        ch, num = next(n for n in res if n)[:2]
        canon = next(DEPROTONATED[n[2]] for n in res if n and n[2] in DEPROTONATED)
        f = occ[:, sl][:, prot[sl]].sum(1)
        pka, hill, censored, interp = p04b.fit_hill(ph, f)
        row = dict(pdb_id=args.pdb_id or args.restart.stem.upper(), chain=ch, res_id=num, res_name=canon,
                   pKa_phys=pka, hill=hill, censored=censored, pKa_interp=interp)
        if canon == "HIS":                      # tautomer split at the highest pH replica
            hid = occ[-1, sl][[n is not None and n[2] == "HID" for n in res]].sum()
            row["frac_HID"] = float(hid)
        row.update({f"f_pH{p:g}": float(v) for p, v in zip(ph, f)})
        rows.append(row)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df[["chain", "res_id", "res_name", "pKa_phys", "hill", "censored", "pKa_interp"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
