"""Shared, deterministic CV splitting so every model (GNN, trees, ensembles) is
scored on exactly the same sequence-cluster folds, including when folds are run
as separate SGE array tasks."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def make_folds(groups: list[str], n_folds: int = 10, seed: int = 42, val_frac: float = 0.15):
    """Return [(train_idx, val_idx, test_idx), ...].  Whole groups (seq30 clusters)
    stay together; the inner-validation groups are drawn from the training groups
    with an RNG that depends only on (seed, fold)."""
    ug = sorted(set(groups))
    np.random.default_rng(seed).shuffle(ug)
    lut = {g: i for i, g in enumerate(ug)}
    gidx = np.array([lut[g] for g in groups])
    out = []
    n = min(n_folds, len(ug))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n).split(gidx, groups=gidx), 1):
        trg = np.unique(gidx[tr])
        rng = np.random.default_rng(seed * 1000 + fold)
        vg = set(rng.choice(trg, size=max(1, int(round(val_frac * len(trg)))), replace=False).tolist())
        out.append((np.array([i for i in tr if gidx[i] not in vg]),
                    np.array([i for i in tr if gidx[i] in vg]), np.array(te)))
    return out
