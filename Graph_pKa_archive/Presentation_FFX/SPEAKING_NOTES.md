# Speaking Notes — pKa GNN Presentation (~10 min)

> Bold = read off, the things that *must* be said.
> Italic = backup answers if the professor probes.

---

## Slide 1 — Title: Graph-Based Deep Learning for Protein pKa Prediction with AMOEBA

**"This project predicts the pKa of ionizable residues in proteins using a Graph Attention Network on top of the AMOEBA polarizable force field."**

- One sentence framing: a **machine-learning model that takes a protein structure as input and outputs a number — the pKa of one residue** — with the twist that the input features come from a *polarizable* physics simulation, not a fixed-charge one.
- *Backup:* AMOEBA = Atomic Multipole Optimized Energetics for Biomolecular Applications. It models charges as multipoles (monopole + dipole + quadrupole) plus induced dipoles that respond to the electric field. Standard force fields (AMBER, CHARMM) only have fixed point charges.

---

## Slide 2 — Motivation: Why pKa Prediction Matters

**"pKa is the pH at which a residue is half-protonated, half-deprotonated. It controls enzyme catalysis, binding, and folding. About 30% of residues are ionizable, and inside a protein their pKa can shift by up to 9 units from the solution value — that's a billion-fold change in protonation equilibrium."**

- **Why this is hard for traditional methods:** the protein's interior is a low-dielectric, anisotropic environment. A buried lysine surrounded by hydrophobic residues "wants" to give up its proton sooner — its pKa drops. Fixed-charge force fields **cannot capture this** because the charges don't respond to the local field. AMOEBA's induced dipoles do respond, so a model trained on AMOEBA features inherits that physics for free.
- **Our approach in one sentence: prepare each structure with FFX rotamer optimization (not just energy minimization), extract polarizable-FF features, and feed a small graph attention network. We extend the published baseline by including CYS and TYR, by adding pH-conditioned titration features, and by using FiLM conditioning.**
- *Backup — Song et al. 2026 reference:* The published baseline is the JCIM 2026 paper (Song et al.). They used a similar GAT but with a fixed-charge force field, only ASP/GLU/HIS/LYS, and no pH conditioning. Everything we beat or match is benchmarked against them.

---

## Slide 3 — GATv2 Architecture

**This is the most important slide for a CS audience. Spend ~75 seconds here.**

### What is a graph neural network, in plain language?

**"A neural net is a stack of matrix multiplications with non-linearities in between — it learns by adjusting those matrices so the predicted pKa matches the experimental pKa. A *graph* neural net is the same idea, but the input is a graph: nodes have feature vectors, edges connect related nodes, and at each layer every node updates its feature by mixing in features from its neighbours."**

### How we build the graph

- **One graph per titratable residue.** Center the residue, draw a sphere of radius **r = 9 Å**, and every atom inside the sphere becomes a node.
- **Edges are *through-space*, not just covalent bonds** — any two atoms within cutoff distance get an edge. This matters because pKa is dominated by *electrostatics*, which acts through space, not through chemical bonds.
- **Node features (24-dim baseline):** one-hot atom type, one-hot residue type, atomic charge from AMOEBA, SASA (solvent-accessible surface area — how exposed the atom is to water), H-bond donor/acceptor flags, radius shell counts, and local-frame coordinates (x along Cα→C, z perpendicular to the Cα-C-O plane). Using the *local frame* makes the features rotation-invariant — rotating the protein doesn't change the prediction.

### What is GATv2 specifically?

- **GAT = Graph Attention Network.** Instead of averaging neighbour features uniformly, the network *learns* an attention weight for each neighbour — "this oxygen matters more than that carbon for this prediction". GATv2 is a 2021 fix of the original GAT that makes the attention truly dynamic (it can attend differently to different query nodes, which the original couldn't).
- **One layer only**, hidden = 48 channels, **4 attention heads** running in parallel, concatenated → 192-dim residue embedding.
- **Pooling:** every atom in the residue's sphere has a 192-dim vector; we take the mean (`global_mean_pool`) → one 192-dim vector per residue.
- **Head:** a single linear layer maps 192 → 1 → predicted pKa.
- **Why one layer?** With ~340 training graphs, deeper GNNs over-smooth — every atom's vector starts looking the same after a few hops. One layer keeps the residue-centric inductive bias. **Dropout = 0.5** for the same reason: prevent overfitting.
- *Backup — training loop:* 10-fold cross-validation, fixed seed=42 so every feature variant is trained on the *same* splits → enables **paired statistical tests**. Adam optimizer, lr = 1e-2, batch 32, MSE loss. (Titration runs use SmoothL1 because the pH-conditioned data has more outliers.)

### Upstream: FFX rotopt, not plain EM — *cover this carefully*

**"Most pKa prediction papers run *energy minimization* on the crystal structure: gradient descent until forces are small. That just relieves clashes — the side chain stays in whatever rotamer the crystallographer fit."**

- **What we do instead — FFX rotamer optimization with the ManyBody algorithm:**
  1. For every titratable side chain, enumerate all rotamers from the **Dunbrack rotamer library** (a discrete catalog of common side-chain conformations).
  2. Compute the AMOEBA energy of every (residue, rotamer) and every pair (residue_i, rotamer_a, residue_j, rotamer_b) — these are "self" and "pair" energies.
  3. Apply **Dead-End Elimination (DEE)** — a pruning rule: if rotamer A on residue *i* is *always* worse than rotamer B no matter what every other residue does, A can never be in the global minimum and is pruned.
  4. After DEE, the remaining combinatorial space is solved by FFX's **ManyBody** algorithm — a self-consistent global optimization that finds the lowest-energy assignment of rotamers across all titratable residues *simultaneously*.
  5. Then a final Cartesian minimization on the chosen rotamers.
- **Why this is better than EM:** EM is a local move; rotopt is a global move. EM can leave a side chain pointing the wrong way if the crystal pose was wrong, and the GNN then sees a misleading microenvironment. Rotopt produces a **physically self-consistent polarizable state** — the induced dipoles and multipoles are converged on the *correct* rotamer.
- *Backup if asked "how big is ManyBody?":* For a typical 100-residue protein with ~10 titratable side chains and ~20 rotamers each, the naive search space is 20^10 ≈ 10^13. DEE typically prunes >99% of that, and ManyBody solves the rest with branch-and-bound + self-consistency.

---

## Slide 4 — Radius Sweep

**"We swept the graph cutoff from 7 to 11 Å. The MAE is essentially flat — 0.757 to 0.763."**

- **Important framing:** this sweep is on the **charge-only baseline (no dipoles yet)** — same feature set as the published Song paper, just on FFX-rotopt structures. Concretely the per-atom features are: 4 residue-OHE + 9 atom-label-OHE + 3 local-frame coords + **1 AMOEBA monopole charge (`atomic_charge`)** + 2 H-bond counts + 1 SASA. The dipole features come later (slide 7).
- **Why does this matter?** It tells us the model has *saturated* the useful local information at r ≈ 7 Å. Adding more atoms out to r = 11 Å doesn't help — and the reason is partly **structural / chemical** and partly an **artifact of the upstream prep**:
  - Each atom already carries its **AMOEBA permanent monopole charge** — a smarter point charge than a fixed-charge force field would assign, because it was derived from a polarizable-multipole fit. So even at small radii every atom we include encodes more about its environment than a Tinker/AMBER point charge would.
  - **FFX rotopt has already done the long-range work upstream** — the ManyBody global rotamer optimization saw the *whole protein*, so by the time we cut a 9 Å sphere the side chain is already in the conformation the rest of the protein wants. We don't need a big radius for the GNN to "see" distant context — the rotopt step has baked that context into the local geometry.
  - Beyond ~9 Å the new atoms are mostly bulk solvent and surface backbone — high feature variance, low pKa-relevant signal → noise.
- Compare to a fixed-charge model: there you'd expect MAE to keep dropping with radius because neither the charges *nor* the structure summarize the environment, so you have to physically include it.
- **Practical default = 9 Å:** small enough to be cheap, big enough that no information is lost.
- **9 Å also coincides with the typical "first hydration shell + first protein neighbour shell"** — physically motivated cutoff.
- *Backup:* I expect the radius sweep to look even *flatter* once dipole features are added, because dipoles strengthen the same "local atoms summarize distant context" effect. That's a planned ablation.

---

## Slide 5 — Per-Residue Performance

**"Overall MAE = 0.76 pKa units across 343 residues. That's competitive with the literature."**

- Walk through the residues in order:
  - **GLU 0.51 — easiest.** Narrow experimental pKa range (mostly 4.0–4.5), well-defined H-bond pattern, n = 99 (lots of training data).
  - **TYR 0.60 — surprisingly good.** Phenol pKa is dominated by aromatic ring environment, which the local graph captures.
  - **ASP 0.74 — harder than GLU** even though chemically similar. The reason: ASP has one fewer CH₂ in its side chain, so the carboxylate is *closer to the backbone*. A 9 Å sphere around the carboxylate captures proportionally more backbone context, which adds noise. ASP also has more proton-shuttle outliers (Asp-His pairs).
  - **HIS 0.84 — bimodal.** HIS has two tautomers (HID with proton on Nδ, HIE with proton on Nε) and the dataset mixes them. The model predicts the average; the tails of the distribution are hard.
  - **LYS 0.96 — small n=34, buried-LYS outliers.** A buried lysine can have pKa ~6 (down from solution 10.5). Few training examples of this regime.
  - **CYS 1.80 — broken.** CYS pKa spans 3 to 13 depending on whether the thiol is exposed, in a metal site, or near a disulfide. **Our feature set has no disulfide / metal-binding channel,** so the model has no way to distinguish these cases. CYS dominates global RMSE; removing it drops MAE by ~0.06.
- **Honest framing for the talk:** most published methods (PROPKA, the Song paper) **drop CYS and TYR entirely** because they're rare and noisy. **I kept them** to be honest about coverage. If we restrict to the same four residues as the paper, my 4-residue MAE is **~0.68**.

---

## Slide 6 — Residual Calibration

**"Residual = predicted − experimental. A perfect model would give a flat band at zero across the full pKa range."**

- We see a **slight negative slope at the extremes** — predictions are *too high* for low experimental pKa and *too low* for high experimental pKa. This is **regression to the mean**, a classic small-dataset symptom: when the model is uncertain it falls back toward the average of the training distribution.
- **LYS residuals above +1** = buried LYS underpredicted (model says 9, truth is 5–6). The model has too few examples of buried-LYS environments to recognize them.
- **CYS residuals below −1** = thiol context the model can't read.
- **ASP has a wide spread at low pKa values** because of proton-shuttling outliers.
- *Key takeaway:* **the cure is more data at the extremes, not a fancier model.** This is what motivates the next-step CpHMD trajectory work.
- *Backup — Brody et al. 2021 reference:* on the slide because GATv2 (which we use) was introduced in that paper.

---

## Slide 7 — Feature Engineering

**This slide is where you defend "why Wilcoxon, not PCA". Cover it explicitly.**

### The setup

**Exact baseline node features — 24 dimensions (the "Charge" variant):**

| Columns | # | What it is |
|---|---|---|
| `Residue Name_{ASP,GLU,HIS,LYS}` | 4 | one-hot residue type |
| `atom_label` → 9-class OHE | 9 | atom element (C/CA/N/O/CB/sidechain-N/sidechain-O/sidechain-S/other) |
| `recalculated_x/y/z` | 3 | Cartesian coords in the residue-local frame |
| `atomic_charge` | 1 | AMOEBA permanent monopole (partial charge from multipole fit) |
| `Number of H-Bonds as donor/acceptor` | 2 | H-bond count from mdtraj Baker-Hubbard |
| `SASA_Value` | 1 | solvent-accessible surface area from mdtraj |
| `Radius_{r}A_{N,CA_C,O,S}_Count` | **4** | neighbour-count shell features: number of backbone N, Cα/C, O, and S atoms from *other* residues within radius r Å (MinMax-normalized). One set of 4 per graph (the graph's own cutoff radius). |
| **Total** | **24** | *(matches Song et al.'s reported count; script `log.info` confirms 4+9+3+1+2+1+4 = 24)* |

*All dipole columns (`Dipole_X/Y/Z`, `PermDipole_X/Y/Z`, and all invariant scalars) are **dropped** from `data.x` for the Charge baseline via `DROP_FEATURE_COLS` env var in `03_create_datasets.py`.*

**The 8 variants tested (confirmed from `sweep_electrostatics_138.ps1` and `sweep_physedge_138.ps1`):**

| Variant | Node dim | What's added vs Charge | Edge features |
|---|---|---|---|
| **Charge** | 24 | — (baseline) | none |
| **InducedDip** | 27 | + `Dipole_X/Y/Z` (local-frame induced dipole 3-vector) | none |
| **PermDip** | 27 | + `PermDipole_X/Y/Z` (local-frame permanent dipole 3-vector) | none |
| **BothDip** | 30 | + both induced and permanent dipole 3-vectors | none |
| **CoulombEdge** | 24 | (Charge nodes) | + edge attr: `[‖r‖, q_i·q_j/‖r‖]` (2-dim Coulomb term) |
| **CoulombEdgeBoth** | 30 | + BothDip nodes | + same Coulomb edge |
| **Invariant** | 30 | + 6 rotation-invariant scalars: `Dipole_norm`, `Dipole_align_z`, `Dipole_field_proj`, and the same 3 for permanent dipole | none |
| **PhysEdge** | 24 | (Charge nodes) | + 6-dim AMOEBA edge: Coulomb + charge-dipole `φ_qd` (ind+perm) + dipole-dipole `φ_dd` (ind+perm) |
| **InvariantPhysEdge** | 30 | + Invariant nodes | + same 6-dim AMOEBA edge |

**What each extra feature *means*:**
- `Dipole_X/Y/Z` — the induced dipole vector for the atom rotated into the residue's local frame (x = Cα→C, z = normal to Cα-C-O plane). The magnitude and direction tell the GAT how the atom's electron cloud is distorted by its neighbours.
- `Dipole_norm` — ‖μ_induced‖, rotation-invariant magnitude. How strongly is this atom polarized?
- `Dipole_align_z` — μ_induced · ẑ_local, dot product with the backbone-perpendicular axis. Is the dipole pointing into/out of the backbone plane?
- `Dipole_field_proj` — μ_induced · E_neighbours, the induced dipole projected onto the Coulomb field from nearby atoms (computed over a 9 Å kd-tree). Is the dipole aligned with the field that caused it?
- `φ_qd` — charge-dipole interaction energy ∝ q_i (μ_j · r̂_ij)/r² on each edge. An AMOEBA electrostatic interaction term, rotation-invariant because r̂_ij is the bond direction.
- `φ_dd` — dipole-dipole interaction ∝ (μ_i · μ_j − 3(μ_i · r̂)(μ_j · r̂))/r³. The classical dipole coupling tensor along each edge.

- I tested **8 variants** (9 rows in the results table because Charge itself is listed as the baseline row).
- Each variant: **8 random seeds × 10 folds = 80 fold-level MAE values.** Same 10 splits used for every variant (paired design).

### Why **Wilcoxon signed-rank**, not PCA, not a t-test?

**"Each fold gives me a paired observation: the same data split, same hyperparameters, only the feature set differs. I want to ask: is variant X reliably better than baseline across folds?"**

- **PCA would answer a different question** — it's a *dimensionality reduction* of the feature space (does the input have redundant directions?). It cannot tell me whether a model with feature set A predicts better than one with feature set B. Wrong tool.
- **Why not a paired t-test?** t-test assumes the per-fold differences are normally distributed. Fold-MAE differences are not — they're skewed and have heavy tails (one bad PDB can dominate one fold). **Wilcoxon signed-rank is the non-parametric paired test:** it ranks the absolute differences and only uses the signs. It's robust to outliers and makes no normality assumption.
- *Backup — exact protocol:* For each fold and each variant we compute MAE; the difference (variant − baseline) is the paired observation; I run `scipy.stats.wilcoxon` on the 80 paired diffs to get a two-sided p-value.

### Results — be honest

- **InducedDip ΔMAE = −0.004, p = 0.38.** Best variant. Not statistically significant.
- **InvariantPhysEdge ΔMAE = +0.009, p = 0.025** — significantly *worse*. **More information ≠ better generalization** when the dataset is 343 residues and the model has 192-dim hidden state. Extra channels compete for capacity and overfit.
- **Headline:** the GAT has already learned a useful charge-only representation; adding raw dipole vectors is a small additive gain, but adding the principled invariants over-parameterizes.
- **Conclusion: the bottleneck is dataset size, not feature expressiveness.** This justifies the next-step trajectory work (more snapshots per residue) and the planned hyperparameter sweep with regularization.

### Storyline checkpoint — *say this out loud so the audience tracks the narrative*

**"Slides 4–6 were the charge-only baseline that the Song paper uses — 24 features per atom, n=343 including CYS and TYR, MAE 0.760. On this slide I tested feature additions on top of that baseline. InducedDip was the best variant, so from slide 8 onward the *working feature set is baseline + InducedDip = 27 features per atom* (the +3 are the local-frame x/y/z components of the AMOEBA induced dipole — it's a 3-vector, not a scalar). Slides 8 and 9 are computed with that 27-feature model."**

- **Why 27, not 25?** A dipole is a 3-D vector — μ_x, μ_y, μ_z in the residue's local frame. Each component is its own input channel. So InducedDip = +3 columns → 24 + 3 = **27**. (The Invariant variant packs the dipole into rotation-invariant *scalars* — that's where you'd get +1, +2, or +3 scalar features, but it's a *different* design and it's the one that didn't work.)

---

## Slide 8 — Permutation Importance

**"Permutation importance asks: if I randomly shuffle one feature group across the validation set, how much does the MAE go up? A big rise = the model relies on that feature. Zero rise = the model ignores it."**

- This is a *post-hoc* interpretability tool — it doesn't change the model, just probes it.
- **Decode the bar colors / groups (this is what's on the plot):**
  - **Residue_OHE (~0.46):** by far the biggest. The four-class one-hot for ASP/GLU/HIS/LYS dominates because each residue type has its own pKa baseline (ASP ~3.8, GLU ~4.4, HIS ~6.5, LYS ~10.5). Knowing the residue type is most of the signal.
  - **LocalCoords (~0.40):** atom positions in the local frame — captures geometry of the active site.
  - **Hbond_acceptor (~0.23):** very informative — H-bonding partners shift pKa.
  - **atomic_charge (~0.09):** the AMOEBA monopole. *Less* important than you might think because it's already implied by atom type + residue type.
  - **AtomLabel_OHE (~0.09):** atom element OHE (C/N/O/S).
  - **InducedDip_invariants (~0.043), PermDip_invariants (~0.031):** the *new* physics features. Used about as much as `atomic_charge` — definitely not ignored, but each contributes ~5% as much as residue type.
  - **SASA (~0.03):** small but nonzero. Surprising it's not larger; probably because SASA correlates with atom-level features the model already has.
- **Resolves the apparent contradiction with slide 7:** "the Invariant model *uses* the new features (importance ~0.045) yet has *worse* validation MAE — how?" Answer: the model exploits them on the training set but they don't generalize. **Permutation importance on val ≠ generalization gain.** The extra parameters used to embed those features hurt more on small data than the features help.

---

## Slide 9 — InducedDip Per-Residue

**"Even though the global MAE didn't move much, the InducedDip variant *redistributes* the error. Look where:"**

- **All numbers below are on the same n = 343 dataset as slide 5 — CYS and TYR included — so the comparison to the 0.760 baseline is apples-to-apples.**
- GLU 0.500 (was 0.506) — flat, already at ceiling.
- TYR 0.603 (was 0.603) — flat.
- ASP 0.648 (was **0.735**) — biggest gain, ~12% relative.
- HIS 0.731 (was **0.842**) — second biggest gain.
- LYS 1.026 (was 0.963) — slightly worse, within noise (small n=34).
- CYS 1.803 (was 1.803) — **unchanged**. Induced dipoles cannot fix CYS because the missing signal is the disulfide / metal-binding feature, not polarization.
- **Weighted overall MAE = (0.500·99 + 0.603·14 + 0.648·111 + 0.731·68 + 1.026·34 + 1.803·17) / 343 ≈ 0.715** — vs **0.760** charge baseline (slide 5). All gain comes from ASP + HIS.
- **Why did ASP and HIS gain?** Both rely on H-bond *partners* whose dipole orientations matter. ASP's carboxylate is short-range polarizable, HIS's imidazole is highly polarizable. Induced dipoles encode exactly that.
- *Caveat:* the **global** Wilcoxon p-value is still 0.38 — be honest that this is a per-residue *trend*, not a confirmed win on the global metric. The redistribution is real even though the headline number is within noise.

---

## Slide 10 — Titration + FiLM

**This is the most novel slide. Spend ~90 seconds.**

### What is "titration rotopt" and how is it different from regular rotopt?

- **Regular rotopt:** runs the ManyBody rotamer search assuming each titratable residue is in its **default protonation state** (ASP/GLU charged, HIS neutral HIE, LYS charged) — this is just the standard "fix the structure" step at one implicit pH.
- **Titration rotopt:** runs ManyBody **at four explicit pH values: 3.94, 4.40, 6.45, 8.55** (chosen to bracket the pKa range of all four target residue types). At each pH, FFX **expands the rotamer library to include both protonated and deprotonated rotamers of every titratable side chain**, and ManyBody picks the global minimum that *includes* the protonation state choice as part of the optimization. So at pH 3.94 most ASPs come out protonated; at pH 8.55 most are deprotonated.
- This produces **4 graphs per residue** instead of 1. Same target pKa label (the experimental number) but different microenvironments.

### Why is this better than running the model 4 times?

**Because the *residue itself* sits in different conformations at different pH, and the model now learns the pH→structure→pKa mapping jointly.** Running 4 independent models would make every variant a separate problem; instead we condition one model.

### Conditioning architectures — what is FiLM and why?

- **Naive:** just append pH as one more node feature. Cheap, works okay.
- **Concat:** pool the graph, then concatenate the pH scalar before the linear head.
- **FiLM (Feature-wise Linear Modulation, Perez et al. AAAI 2018):** a small MLP turns the pH scalar into **two vectors of length 192 — γ (gamma) and β (beta) — one pair per channel of the GAT embedding**. We then replace each atom's vector `h` with `γ ⊙ h + β` (elementwise multiply, then add). **In effect γ rescales each channel and β shifts it, *as a function of pH*.**
- **Why FiLM is more powerful than naive:**
  - Naive treats pH as just another input dimension — the model has to learn from scratch how to use it.
  - FiLM explicitly factors the network into "what does the structure say" (the GAT) and "how should I modulate that based on pH" (the FiLM MLP). This is the standard trick in vision-language models — FiLM was originally invented for "answer this visual question about this image where the question is text".
  - Mathematically FiLM is more expressive: a learned affine transform per channel per pH > a single concat.
- *Backup — implementation detail:* in `07_train.py` we use `gamma = 1 + MLP_gamma(pH)` so that at initialization the model is identity (gamma ≈ 1, beta ≈ 0) and only learns to modulate when needed.

### GroupKFold by residue — why?

- Naive 10-fold CV would split *graphs*, so the same residue at pH 3.94 could be in train and at pH 4.40 could be in val → **leakage**.
- **GroupKFold ensures all 4 pH-graphs of one residue are in the same fold.** This is the only valid way to estimate generalization to new residues.

### Result

- Rotopt baseline 0.760 → titrate-naive **0.603** → **titrate-FiLM 0.596**.
- LYS jumped from 0.963 → **0.49** — biggest absolute gain.
- CYS 1.80 → 1.34 — still bad but materially better.
- **Caveat I will say out loud:** the titration runs are on **n = 503 rows / 198 residues**, while the rotopt baseline is on **n = 343 / 343 residues**. **The titration set is currently the *easier subset* of the dataset** — those PDBs are the ones that **converged first** through the FFX titration pipeline, and the harder PDBs (more titratable residues, more conformational coupling, harder protonation networks) are still running. **So I should not yet claim 0.596 as a clean win until the full 343 finish.** It's a strong sneak peek, not a confirmed result.

---

## Slide 11 — Next Steps

### 1. Finish the rotopt FFX titrate jobs (~140 residues remaining)

- 198/343 done. The stalled jobs are at the extreme pH points (3.94, 8.55) on PDBs with many coupled titratable residues — these are exactly the cases where ManyBody has to choose protonation state for many residues at once and the combinatorial space is hardest.
- **Once full coverage finishes, rerun titrate-FiLM on the *complete* set with GroupKFold and recompute** — this is the test that decides whether FiLM is real or an easy-subset artifact.

### 2. Hyperparameter sweep — *with weight decay*

**"The feature-engineering result told us we are bottlenecked by dataset size, not feature expressiveness. The next lever is *regularization and capacity*."**

- **What is weight decay?** A penalty added to the loss that's proportional to the *squared L2 norm of the model weights*: `loss = MSE + λ · ||θ||²`. Equivalent to L2 regularization. **It pushes the network toward simpler functions** by shrinking weights — exactly what you want when overfitting is the failure mode.
- **Why I need it:** the InvariantPhysEdge result on slide 7 (significantly *worse* with more features) is the classic overfitting signature. Weight decay lets us keep the richer features but penalize the model for actually using them unless they help.
- **Sweep grid:**
  - hidden ∈ {32, 48, 64, 96}
  - heads ∈ {2, 4, 8}
  - dropout ∈ {0.3, 0.5, 0.6}
  - **weight_decay ∈ {0, 1e-4, 1e-3, 1e-2}** (probably switch optimizer to AdamW for decoupled decay)
  - layers ∈ {1, 2}
- **Same 10-fold seed=42 splits** so we can paired-Wilcoxon every cell of the grid against the current baseline.
- *Stretch goal:* pre-train on a larger neutral-pH MD ensemble before fine-tuning on PKAD-R targets — transfer learning to compensate for n = 343.

---

## Anticipated Q&A — quick-answer cheat sheet

| Question | Answer |
|---|---|
| **Why a GAT and not a GCN?** | GCN averages neighbours uniformly; in a residue active site, an H-bond donor matters more than a carbon four bonds away. Attention learns that weighting. |
| **Why not a transformer over atoms?** | Tried implicitly via 4 attention heads. A full transformer has no graph inductive bias and would need much more data. |
| **Why MSE not MAE as loss?** | MSE for the rotopt baseline (smooth, differentiable, standard). For titration we switched to SmoothL1 because outliers (LYS, CYS) blow up MSE gradients. |
| **What's `concat=True` in GATv2?** | Concatenate the 4 attention heads instead of averaging → 4 × 48 = 192 dim output. Concatenation preserves multi-view info; averaging mixes it. |
| **Why `add_self_loops=False`?** | Each atom would otherwise attend to itself with weight inflated by GATv2's softmax — distorts attention. We let mean-pool reintroduce the self-feature. |
| **How is AMOEBA's induced dipole computed?** | Self-consistently — at each step, every atom gets a dipole = polarizability × local field; the field depends on every other atom's dipole, so iterate to convergence (~10 SCF cycles). |
| **What does "DEE proves rotamer A is dominated" mean exactly?** | If `min_over_others(E_self(A) + Σ E_pair(A, x)) > max_over_others(E_self(B) + Σ E_pair(B, x))` then A can never be in the global minimum no matter what other residues do. Hard pruning. |
| **Why not use AlphaFold structures?** | Crystal pKa data is paired with crystal structures. AF would inject prediction error into the input. Future work. |
| **Could you predict ΔpKa instead of pKa?** | Yes — would re-anchor each prediction at the residue-type mean and reduce regression-to-mean. Easy ablation. |
| **What's the train/inference time?** | ~2 minutes for full 10-fold CV on a single GPU; inference is essentially free (one residue ≈ 1 ms). FFX rotopt is the cost — ~5–60 minutes per PDB. |
