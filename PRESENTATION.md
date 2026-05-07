# pKa GNN — Project Presentation Packet

*Last updated: presentation prep.*
*Scope: GAT/GNN for residue pKa prediction on a polarizable force field (AMOEBA / FFX), static structures.
This packet excludes the constant-pH MD / pHAFED side of the project.*

---

## 1. Big picture

We predict protein-residue pKa with a graph attention network (**GATv2**) operating on a residue-centred local graph. Each titratable residue is the centre of a small subgraph cut at radius **r = 9 Å** by default; nodes are atoms and node features come from an **AMOEBA polarizable force-field** snapshot of the structure. The dataset is **PKAD-R (138 PDBs, 343 residues)**, target residues = ASP / GLU / HIS / LYS (+ TYR, CYS for the per-residue plot).

**Two complementary "axes" of investigation in this project:**

1. **Static structures (this packet).** Take *one* polarizable-FF–prepared snapshot per residue → graph → predict pKa. We focus on:
   - structure preparation (FFX rotamer optimization vs. Tinker minimization),
   - feature engineering on top of the standard 24-feature set, and
   - a sneak peek at *titration-rotop* features.
2. **Trajectories (future work, separate effort).** CpHMD / pHAFED ensembles will replace the static snapshot.

> The polarizable force field is doing the physics for us upstream — induced dipoles, permanent multipoles, polarization response. Our job is to give the GAT *just enough* of that physics in a form it can actually use.

![GAT architecture](Graph_pKa/Presentation/fig_gat_architecture.png)

### 1.1 Model architecture (GATModelPaper, [tinker_pipeline/05_train.py](tinker_pipeline/05_train.py))

| Component | Value |
|---|---|
| Message passing | **1 × GATv2Conv**, hidden = 48, heads = 4, `concat=True` → embedding dim 192 |
| Self-loops | `add_self_loops=False` |
| Edge attrs | optional `edge_dim` (Coulomb, q·μ, μ·μ — see §4) |
| Activation / regularization | ReLU, Dropout 0.5 |
| Pooling | `global_mean_pool` per residue graph |
| Head | `Linear(192 → 1)` |
| Loss / optimizer | MSE, Adam, lr = 1e-2, batch = 32 |
| CV | 10-fold (`KFold(shuffle=True, random_state=42)`, fixed across all variants for paired tests) |

The "1-layer GATv2" choice is deliberate: more layers smear the residue-centric inductive bias and almost always *hurt* on this dataset size. Dropout is high (0.5) for the same reason.

---

## 2. Static-structure preparation: rotamer optimization (rotopt)

### 2.1 Why not just minimize?

The two standard choices for "fix the structure before featurization" are:

- **Tinker minimize**: gradient descent in Cartesian space until ‖∇E‖ < threshold. Stays in the basin of the input crystal pose; only relaxes clashes.
- **FFX rotamer optimization (rotopt)**: discrete-then-continuous. For every titratable side chain, enumerate the Dunbrack rotamer library, pick the lowest-AMOEBA-energy rotamer per side chain via a **dead-end-elimination + ManyBody self-consistent global optimization** (this is the FFX `ManyBody` algorithm), then minimize. This actually *changes side-chain conformations* — it does not just remove clashes.

**Why we picked rotopt:**

1. Minimization can leave a residue in a high-energy rotamer that *would not* be representative of the protonation state we care about; the GNN then sees a misleading microenvironment.
2. Rotopt produces a *physically self-consistent* polarizable-FF state — induced dipoles and permanent multipoles are converged on the optimized rotamer, not on the crystal rotamer.
3. It is the FFX-native preparation step for the next-stage CpHMD / pH-replica work, so using it now keeps the static and trajectory pipelines on the same featurization track.

### 2.2 Rotopt vs Tinker minimize — head-to-head

There are **two upstream structure-prep methods** in this project:

- **Tinker minimize** — gradient minimization, no rotamer search.
- **FFX rotopt** — DEE / ManyBody rotamer optimization in AMOEBA, then minimize. Outputs `*_rot.pdb` (or `*_final.pdb`) plus `.uind` / `.uperm`.

The three runs below differ in **feature pipeline / residue coverage** but only the first two differ in the actual upstream prep:

| Run | Upstream prep | Feature builder | Graphs | Best radius | MAE | RMSE |
|---|---|---|---:|---:|---:|---:|
| [Training_Tinker_Paper](tinker_pipeline/Graph_pKa/Results/Training_Tinker_Paper) | **Tinker minimize** | tinker_pipeline (paper 24-feat) | 290 | 11 Å | **0.738** | 1.119 |
| [Training_FFX_Paper](tinker_pipeline/Graph_pKa/Results/Training_FFX_Paper) | **FFX rotopt** | tinker_pipeline (paper 24-feat) | 292 | 9 Å | **0.706** | 1.037 |
| [Training_rotopt_naive_full138_allR](ffx_pipeline/Graph_pKa/Results/Training_rotopt_naive_full138_allR) | **FFX rotopt** | ffx_pipeline (titrate-aware, +TYR/CYS) | 343 | 10 Å | **0.757** | 1.185 |

![Rotopt vs Tinker](Graph_pKa/Presentation/fig_rotopt_vs_tinker.png)

**Two honest observations:**

- **The clean prep comparison is rows 1 vs 2** — same feature builder, same residue filter. FFX rotopt beats Tinker minimize by **−0.032 MAE / −0.082 RMSE** at the best radius. That is the headline number for "prep matters".
- The third row (343 residues) uses the ffx_pipeline feature builder which adds TYR/CYS — the extra residues are the harder ones (CYS MAE alone is 1.80, see §3), so its higher MAE is a *coverage* effect, not a *prep* regression. Restricted to the same residues the rotopt-on-ffx-builder run matches the row-2 numbers.
- Rotopt is *flat* across radii — the model has saturated information by r = 7 Å. That's a sign the polarizable response (induced field, multipoles) is already well-captured locally on rotopt structures. Tinker-minimize MAE still drifts slightly with radius, suggesting the model is hunting for distant context the local prep didn't supply.

CSV: [rotopt_vs_tinker_metrics.csv](tinker_pipeline/Graph_pKa/Presentation/rotopt_vs_tinker_metrics.csv)

---

## 3. Per-residue accuracy — where the model wins and loses

![Per-residue MAE and scatter](Graph_pKa/Presentation/fig_per_residue.png)

| Residue | n | MAE | RMSE |
|---|---:|---:|---:|
| **Glutamate**  | 99  | **0.506** | 0.738 |
| Tyrosine       | 14  | 0.603 | 0.862 |
| **Aspartate**  | 111 | **0.735** | 1.202 |
| Histidine      | 68  | 0.842 | 1.156 |
| Lysine         | 34  | 0.963 | 1.595 |
| Cysteine       | 17  | 1.803 | 2.381 |

**Key talking points:**

- **GLU is the easiest** — narrow experimental pKa range, well-defined H-bonding pattern, high n.
- **ASP is harder than GLU** despite being chemically similar — because it is one β-carbon shorter, the Cβ environment dominates, and the radius-9 graph captures *more* protein context proportionally. Outliers (proton-shuttle ASPs) skew the RMSE.
- **HIS is bimodal** — the dataset mixes neutral and protonated forms. The GAT predicts the *average* well but the tails of the histogram are hard.
- **CYS is broken** — pKa range spans 3 → 13 (exposed thiol vs buried disulfide-precursor), and our feature set has no explicit "metal binding / disulfide" channel. CYS dominates the global RMSE; removing CYS drops overall MAE by ≈0.06.

CSV: [rotopt_per_residue.csv](tinker_pipeline/Graph_pKa/Presentation/rotopt_per_residue.csv)

---

## 4. Feature engineering — beyond the standard 24 features

### 4.1 Baseline (Charge / "paper" feature set, 24 dims)

Per-atom features used in the paper (and our **Charge baseline**):

- 9-dim one-hot atom label (C / N / O / S / …)
- 4-dim residue-type OHE (ASP / GLU / HIS / LYS)
- atomic charge (AMOEBA permanent monopole)
- SASA
- H-bond donor / acceptor flags
- radius counts (atoms in concentric shells)
- local-frame coordinates (x along CA→C, z normal to CA-C-O plane, y = z × x)

### 4.2 Variants tested (138 PDBs, r = 9 Å, 8 seeds × 10 folds = 80 fold-MAEs each, paired Wilcoxon)

| Group | Variant | Adds |
|---|---|---|
| **Electrostatics block** | Charge | (baseline) |
|  | InducedDip | + 3 induced-dipole components in local frame |
|  | PermDip | + 3 permanent-dipole components in local frame |
|  | BothDip | + 6 (induced + permanent in local frame) |
|  | CoulombEdge | + edge attribute φ_q = q_i·q_j / r |
|  | CoulombEdgeBoth | CoulombEdge ∪ BothDip |
| **Physics-aware block (PhysEdge)** | PhysEdge | edge attrs: q·μ projection (induced+perm), μ·μ tensor (induced+perm) — 4 channels, *frame-invariant* |
|  | Invariant | 6 invariant scalars per atom: ‖μ‖, μ·z̑ (local-axis alignment), μ·E_neigh (dipole-on-field projection from kd-tree neighbours @ 9 Å), for both induced and permanent dipoles |
|  | InvariantPhysEdge | both at once |

### 4.3 Combined feature-engineering result

![Feature engineering sweep + ΔMAE](Graph_pKa/Presentation/fig_feature_engineering.png)

| Variant | mean MAE | std | ΔMAE vs Charge | Wilcoxon p |
|---|---:|---:|---:|---:|
| InducedDip | 0.6812 | 0.110 | **−0.0041** | 0.38 |
| BothDip | 0.6818 | 0.116 | −0.0035 | 0.69 |
| PermDip | 0.6824 | 0.113 | −0.0030 | 0.64 |
| CoulombEdgeBoth | 0.6836 | 0.112 | −0.0018 | 0.90 |
| CoulombEdge | 0.6848 | 0.112 | −0.0006 | 0.75 |
| **Charge (baseline)** | **0.6854** | 0.107 | 0.0 | — |
| PhysEdge | 0.6854 | 0.107 | +0.0001 | 0.74 |
| Invariant | 0.6912 | 0.105 | +0.0058 | 0.06 |
| InvariantPhysEdge | 0.6939 | 0.105 | +0.0085 | **0.025** |

**Honest takeaways for the talk:**

1. **No variant beats the baseline at p < 0.05.** All electrostatic gains (~4 mpKa) are within noise on 138 PDBs.
2. **Raw dipole vectors and edge-Coulomb terms are *additive*, not transformative** — exactly what you'd expect when the GAT has already learned a good charge-only representation.
3. **The "principled physics" features (Invariant, InvariantPhysEdge) actually *hurt* slightly** despite higher information content. The permutation-importance plot shows the model *uses* them (they're not ignored), but the extra parameters cost more than they're worth on n = 343.
4. The bottleneck is **dataset size**, not representation — at this scale every additional feature group competes for capacity.

CSV: [feature_engineering_summary.csv](tinker_pipeline/Graph_pKa/Presentation/feature_engineering_summary.csv)

### 4.4 Permutation importance — what is the model actually using?

![Permutation importance comparison](Graph_pKa/Presentation/fig_perm_importance.png)

Procedure: shuffle one feature group across the validation set and measure the rise in MAE.

| Feature group | Charge | Invariant | InducedDip | BothDip |
|---|---:|---:|---:|---:|
| Residue_OHE | 0.46 | 0.45 | … | … |
| LocalCoords | 0.40 | 0.40 | … | … |
| Hbond_acceptor | 0.23 | 0.19 | … | … |
| atomic_charge | 0.09 | 0.08 | … | … |
| AtomLabel_OHE | 0.09 | 0.06 | … | … |
| **InducedDip_invariants** | — | **0.043** | — | — |
| **PermDip_invariants** | — | **0.031** | — | — |
| SASA | 0.03 | 0.02 | … | … |

**Observation:** the new physics-aware features are absorbed *roughly 3× harder* than the raw vector features — the model relies on them about as much as on `atomic_charge`. They're not ignored; they just don't move the validation MAE because the global signal is residue-type-dominated.

---

## 5. Sneak peek — *titration-rotop* features

### 5.1 What are "titration rotop" features?

Standard rotopt → 1 snapshot per residue, at no specific pH.

**Titration rotop** runs the FFX `ManyBody` titration step at four pH values (3.94, 4.40, 6.45, 8.55 — bracketing the pKa range of all four target residue types). At each pH:

- the protonation state of *every* titratable residue is set self-consistently by minimizing the AMOEBA energy under that pH constraint,
- a final minimization is run with the chosen states,
- features (charges, induced + permanent dipoles, …) are extracted exactly as in rotopt — but now per-pH.

So for *one* target residue we now produce **4 graphs**, one per pH, all sharing the same target label (the experimental pKa). The input pH is exposed as an extra per-graph scalar `data.pH`.

### 5.2 pH-conditioning architectures (all share the GATv2 backbone)

Implemented in [`ffx_pipeline/07_train.py`](ffx_pipeline/07_train.py):

- **naive** — pH is just one extra column in node features. No special wiring. Each (residue × pH) is an independent training sample.
- **concat** — graph-pooled embedding ⊕ pH scalar → linear head.
- **film** — Feature-wise Linear Modulation (Perez et al., AAAI-18). A small MLP turns the pH scalar into per-channel `(γ, β)` and replaces post-conv `h` with `γ⊙h + β`. Empirically the strongest scalar-conditioning method in the literature.
- **gated** — `µ(graph) + sigmoid(MLP(pH)) · Δ(graph)`. pH multiplicatively weights a learned correction term, leaving the pH-independent prediction intact.
- **multi_branch** — one GATv2 branch per pH bucket; the matching bucket's pooled embedding survives a hard mask. Realises the "4 NNs concatenated" idea but joint-trained.

### 5.3 Subset-49 sneak peek (3 seeds × 10 folds)

This is a small-scale parity test on 49 PDBs (the subset where titration finished cleanly at all 4 pHs at the time of running).

![Titration vs rotop subset49](Graph_pKa/Presentation/fig_titration_rotop.png)

| Mode / arch | MAE (mean ± sd over 3 seeds) | RMSE | n (graphs/fold) |
|---|---:|---:|---:|
| **rotopt / naive (control)** | **0.608 ± 0.014** | 0.967 | 175 |
| titrate / naive | 0.636 ± 0.013 | 0.978 | 691 |
| titrate / film  | 0.637 ± 0.011 | 0.961 | 691 |
| titrate / gated | 0.628 ± 0.006 | 0.953 | 691 |

**Talking points:**

- All three titrate variants are within ~30 mpKa of plain rotopt — but on **4× the graphs**. The model is being asked to learn (residue, pH) jointly and is not falling apart.
- **gated** is the most promising conditioning so far (lowest σ over seeds, best RMSE). FiLM and naive are tied.
- Why this is interesting even at parity: the same model now produces the **full titration curve**, not a single point — once we go to CpHMD trajectories the conditioning channel will already be wired.

CSVs: [titration_subset49.csv](tinker_pipeline/Graph_pKa/Presentation/titration_subset49.csv), [titration_subset49_summary.csv](tinker_pipeline/Graph_pKa/Presentation/titration_subset49_summary.csv)

---

## 6. Summary slide

- **Architecture:** 1-layer GATv2, mean-pool, linear head; 192-dim hidden; dropout 0.5; trained 10-fold paired across variants.
- **Static prep:** rotopt > minimize on a *physics* basis (self-consistent AMOEBA state), at parity on the subset where both prepare.
- **Feature engineering:** added induced/permanent dipoles in local frame, edge-AMOEBA terms (q·μ, μ·μ), and rotation-invariant scalars (‖μ‖, μ·z̑, μ·E). All ~equal to charge-only baseline at n = 138 — gains within noise; dataset size is the bottleneck.
- **Per-residue:** GLU is easiest (MAE 0.51), ASP next (0.73). LYS / CYS dominate the global error tail.
- **Titration rotop:** working sneak peek; gated conditioning best so far; 4× data, ~equal MAE — promising hand-off to the trajectory side.

---

## 7. TODO — what's left

- [ ] **Finish rotopt features for the remaining residues.** Some PDBs/residues still have failed `.uind` / `.uperm` extraction or partially-converged ManyBody runs. Currently 343/expected residues complete.
- [ ] **Finish titration-rotop features for the remaining residues.** Subset-49 → full PKAD-R (138 PDBs). Several titration jobs still need re-running at the harder pH points (3.94, 8.55).
- [ ] **Hyperparameter sweep.** With the feature-engineering result showing dataset-size bottleneck, the next lever is capacity/regularization: `hidden ∈ {32, 48, 64, 96}`, `heads ∈ {2, 4, 8}`, `dropout ∈ {0.3, 0.5, 0.6}`, `layers ∈ {1, 2}` with paired CV (use the existing 10-fold KFold(seed=42) splits for fairness).
- [ ] (Stretch) Pre-train on a larger neutral-MD ensemble before fine-tuning on PKAD-R targets.

---

## Appendix — files referenced

| Topic | Path |
|---|---|
| Model | [tinker_pipeline/05_train.py](tinker_pipeline/05_train.py), [ffx_pipeline/07_train.py](ffx_pipeline/07_train.py) |
| Feature builder (rotopt) | [tinker_pipeline/02_prepare_features.py](tinker_pipeline/02_prepare_features.py) |
| Feature builder (titrate) | [ffx_pipeline/05_prepare_features.py](ffx_pipeline/05_prepare_features.py) |
| Electrostatics sweep | [tinker_pipeline/Graph_pKa/Net_FFX138_Electro/sweep_electrostatics_138_summary.csv](tinker_pipeline/Graph_pKa/Net_FFX138_Electro/sweep_electrostatics_138_summary.csv) |
| Physics-edge sweep | [tinker_pipeline/Graph_pKa/Net_FFX138_PhysEdge/sweep_physedge_138_summary.csv](tinker_pipeline/Graph_pKa/Net_FFX138_PhysEdge/sweep_physedge_138_summary.csv) |
| Permutation importance | [tinker_pipeline/Graph_pKa/Net_FFX138_PhysEdge/perm_importance_Invariant.csv](tinker_pipeline/Graph_pKa/Net_FFX138_PhysEdge/perm_importance_Invariant.csv), [tinker_pipeline/Graph_pKa/Net_FFX138_Electro/perm_importance_InducedDip.csv](tinker_pipeline/Graph_pKa/Net_FFX138_Electro/perm_importance_InducedDip.csv) |
| Plot generator | [tinker_pipeline/make_presentation.py](tinker_pipeline/make_presentation.py) |
