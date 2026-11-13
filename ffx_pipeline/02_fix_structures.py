"""
02_fix_structures.py

Prepare raw PDB files for FFX/AMOEBA minimization:
  1. Strip NMR ensembles to model 1
  2. Strip hydrogens
  3. Run PDBFixer: remove heterogens/water, replace nonstandard residues,
     add missing internal residues/atoms (unresolved terminal tails are NOT built)
  4. Write output without CRYST1 record
  5. Add SSBOND records for SG–SG pairs ≤ 2.5 Å (FFX ignores CONECT)

Usage:
  python 02_fix_structures.py                    # manifest proteins → data/fixed_pdbs_v2
  python 02_fix_structures.py --only 135L 4PTI   # subset
  python 02_fix_structures.py --patch-ssbond --out-dir data/fixed_pdbs  # retrofit old inputs
"""

import os
import csv
import traceback
from pathlib import Path

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
except ImportError:
    raise SystemExit(
        "pdbfixer / openmm not found.\n"
        "Install with:  conda install -c conda-forge pdbfixer openmm"
    )

# ── Configuration ─────────────────────────────────────────────────────────────
# PIPELINE_ROOT = ffx_pipeline/   (outputs live here)
# REPO_ROOT     = pKa_GNN/        (shared raw inputs live here)
PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT     = PIPELINE_ROOT.parent
RAW_DIR   = str(REPO_ROOT     / "data/raw_pdbs")
MANIFEST  = str(REPO_ROOT     / "data/manifest.csv")
# v2 = SSBOND records + trimmed termini.  The original data/fixed_pdbs is left
# untouched so existing FFX results stay reproducible.
FIXED_DIR = str(PIPELINE_ROOT / "data/fixed_pdbs_v2")
LOG_PATH  = str(PIPELINE_ROOT / "data/fix_log_v2.csv")

SSBOND_CUTOFF  = 2.5   # Å; S–S bonds are ~2.05 Å, non-bonded SG pairs are > 3.4 Å
LARGE_GAP_WARN = 10    # log internal gaps longer than this (they are still built)

# Structures too broken to use — add PDB IDs here to skip them
EXCLUDE = set()  # type: ignore[var-annotated]  # set[str], Python < 3.9 compat
# ─────────────────────────────────────────────────────────────────────────────

# ── Residue pre-processing tables ────────────────────────────────────────────
# Residue names to rename to their nearest standard equivalent before PDBFixer.
# PDBFixer's replaceNonstandardResidues() misses these.
_RESIDUE_REMAP = {
    "CSR": "CYS",   # S-arsonocysteine → cysteine (SG is real sulfur; arsonate group dropped)
    "M3L": "LYS",   # N6,N6,N6-trimethyllysine → lysine
    "HSK": "HIS",   # hydroxy-histidine (3WU2 D336: full imidazole ring + OM) → histidine
    "MSE": "MET",   # selenomethionine (backup; PDBFixer usually handles this)
    "HSD": "HIS",   # CHARMM protonation variant
    "HSE": "HIS",
    "HSP": "HIS",
}

# Atom-level remapping applied when a residue is renamed via _RESIDUE_REMAP.
# Key:   (original_resname, original_atom_name)
# Value: new atom name string, or None to DROP the atom entirely.
#
# Drop atoms that have no equivalent in the target standard residue — keeping
# them causes FFX "could not be assigned an atom type" crashes.
_ATOM_REMAP = {
    # CSR (S-arsonocysteine) → CYS
    # SG already exists as the real cysteine sulfur in the raw PDB.
    # The arsenic and its three arsonate oxygens have no CYS equivalents.
    ("CSR", "AS"):  None,   # arsenic — drop
    ("CSR", "O1"):  None,   # arsonate oxygen 1 — drop
    ("CSR", "O2"):  None,   # arsonate oxygen 2 — drop
    ("CSR", "O3"):  None,   # arsonate oxygen 3 — drop
    # MSE (selenomethionine) → MET
    ("MSE", "SE"):  "SD",   # selenium → methionine sulfur SD
    # M3L (trimethyllysine) → LYS: trimethyl carbons on NZ have no LYS equivalent
    ("M3L", "C3"):  None,
    ("M3L", "C3A"): None,
    ("M3L", "C3B"): None,
    ("M3L", "CM1"): None,   # PDB-standard names of the same three methyls (1PRW)
    ("M3L", "CM2"): None,
    ("M3L", "CM3"): None,
    ("HSK", "OM"):  None,   # ring hydroxyl — no HIS equivalent
}

# Residue names to remove entirely — PDBFixer has no template and we can't
# meaningfully remap them.  NH2 is a C-terminal amide cap (not an amino acid).
# The DNA/RNA bases come from protein-nucleic acid co-crystal structures.
_RESIDUE_STRIP = {
    "NH2",                          # C-terminal amide cap
    "DA", "DC", "DG", "DT",         # DNA
    "DA3", "DC3", "DG3", "DT3",     # DNA 3′ terminal
    "DA5", "DC5", "DG5", "DT5",     # DNA 5′ terminal
    "2DT",                          # modified thymidine
    "A", "C", "G", "U",             # RNA
    "ADE", "CYT", "GUA", "THY",     # full-name variants
}
# ─────────────────────────────────────────────────────────────────────────────

LOG_FIELDS = [
    "pdb_id", "status", "multi_model_stripped",
    "missing_residues_count", "missing_atoms_count",
    "nonstandard_replaced", "large_missing_residues", "notes"
]


# ── NMR multi-model stripper ──────────────────────────────────────────────────

def has_model_records(pdb_path: str) -> bool:
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                return True
    return False


def strip_to_model1(pdb_path: str, out_path: str) -> int:
    lines_out = []
    in_first_model = False
    first_model_done = False
    model_count = 0
    atom_count = 0

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()

            if rec == "MODEL":
                model_count += 1
                if model_count == 1:
                    in_first_model = True
                    continue
                else:
                    break

            if rec == "ENDMDL":
                if in_first_model:
                    in_first_model = False
                    first_model_done = True
                continue

            if model_count == 0:
                lines_out.append(line)
                if rec in ("ATOM", "HETATM"):
                    atom_count += 1
                continue

            if in_first_model:
                lines_out.append(line)
                if rec in ("ATOM", "HETATM"):
                    atom_count += 1

            if first_model_done and rec in ("CONECT", "END", "TER"):
                lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)

    return atom_count


def strip_hydrogens(pdb_path: str, out_path: str) -> int:
    """Remove all H/D atoms so FFX assigns its own via AMOEBA biotypes.

    Returns the number of heavy atoms written. Skipping hydrogens prevents
    FFX PolymerUtils from failing to assign atom types to backbone H atoms
    that PDBFixer places with non-AMOEBA-compatible names (e.g. 'H' on MET-1
    instead of 'H1'/'H2'/'H3').
    """
    heavy_count = 0
    lines_out = []

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                element   = line[76:78].strip() if len(line) > 76 else ""
                # Drop if name starts with H/D or element column says H/D
                if atom_name.startswith(("H", "D")) or element in ("H", "D"):
                    continue
                heavy_count += 1
            lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)

    return heavy_count


def preprocess_residues(pdb_path, out_path):
    """Remap fixable non-standard residues and strip unfixable ones.

    Also:
    - Applies atom-level renaming/_ATOM_REMAP when a residue is remapped, so
      that modified side-chain atoms get correct names and orphaned atoms with
      no AMOEBA equivalent are dropped before PDBFixer sees them (prevents
      "could not be assigned an atom type" crashes in FFX).
    - Strips _RESIDUE_STRIP names from SEQRES header lines so PDBFixer's
      findMissingResidues() never sees them (avoids KeyError on NH2 etc.).
    - Identifies chains where ALL coordinate records would be stripped and
      removes those chains entirely (avoids PDBFixer NoneType crash on empty
      chains, e.g. DNA-only chains in nucleoprotein structures).

    Returns (remapped_list, stripped_list) for logging.
    Must be called AFTER strip_hydrogens so atom counts are meaningful.
    """
    # ── Pass 1: identify chains that will be entirely stripped ───────────────
    chain_keep  = set()   # chains with at least one non-stripped ATOM/HETATM
    chain_strip = set()   # chains where every ATOM/HETATM is stripped
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM") and len(line) > 21:
                resname = line[17:20].strip()
                chain   = line[21]
                if resname in _RESIDUE_STRIP:
                    chain_strip.add(chain)
                else:
                    chain_keep.add(chain)
    # A chain is truly empty only if it has zero kept atoms
    empty_chains = chain_strip - chain_keep

    # Chains whose IDs differ only by case (3WU2: PSII dimer, chains A-Z and a-z).
    # FFX matches SSBOND chain IDs case-insensitively, so the "o" disulfide is
    # also applied to chain "O" (SG bonded twice -> "No AngleType 12 12 12").
    # The lowercase copy is the second, equivalent half of the assembly; drop it.
    all_chains = chain_strip | chain_keep
    case_twins = {c for c in all_chains if c.islower() and c.upper() in all_chains}
    empty_chains |= case_twins

    # ── Pass 2: write filtered file ──────────────────────────────────────────
    remapped  = []
    stripped  = []
    lines_out = []

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()

            # ── SEQRES: filter out stripped residue names token-by-token ─────
            if rec == "SEQRES" and len(line) > 11:
                seqres_chain = line[11]
                if seqres_chain in empty_chains:
                    continue
                prefix   = line[:19]
                tokens   = line[19:].split()
                filtered = [t for t in tokens if t not in _RESIDUE_STRIP]
                if not filtered:
                    continue
                lines_out.append(prefix + "".join("%-4s" % t for t in filtered).rstrip() + "\n")
                continue

            # ── Drop all records belonging to entirely-stripped chains ────────
            if rec in ("ATOM", "HETATM", "TER", "ANISOU") and len(line) > 21:
                chain = line[21]
                if chain in empty_chains:
                    continue

            # ── Drop TER records of stripped residues ─────────────────────────
            # A DNA chain that shares its chain ID with waters is not "empty",
            # so its TER would otherwise survive and precede the first ATOM
            # (PDBFixer: 'NoneType' object has no attribute '_current_chain').
            if rec == "TER" and len(line) > 20 and line[17:20].strip() in _RESIDUE_STRIP:
                continue

            if rec in ("ATOM", "HETATM"):
                resname   = line[17:20].strip()
                atom_name = line[12:16].strip()

                # ── Entirely-stripped residue types ───────────────────────────
                if resname in _RESIDUE_STRIP:
                    stripped.append(resname)
                    continue

                # ── Remapped residue types ────────────────────────────────────
                if resname in _RESIDUE_REMAP:
                    new_resname = _RESIDUE_REMAP[resname]

                    # Atom-level handling: drop or rename non-standard atoms
                    atom_key = (resname, atom_name)
                    if atom_key in _ATOM_REMAP:
                        new_atom = _ATOM_REMAP[atom_key]
                        if new_atom is None:
                            # No equivalent in target residue — drop to prevent
                            # FFX "could not be assigned atom type" crash
                            continue
                        # Rename: atom name occupies cols 12-15 (4 chars)
                        line = line[:12] + "%-4s" % new_atom + line[16:]

                    remapped.append("%s->%s" % (resname, new_resname))
                    line = line[:17] + "%-3s" % new_resname + line[20:]

            lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)

    if case_twins:
        stripped.append("case-twin chains " + "".join(sorted(case_twins)))
    return sorted(set(remapped)), sorted(set(stripped))


def trim_terminal_missing_residues(fixer) -> list:
    """Remove N-/C-terminal entries from fixer.missingResidues (in place).

    PDBFixer keys missingResidues by (chain_index, insertion_index); an index of
    0 is before the first resolved residue and len(chain residues) is after the
    last one.  Returns a list of 'chain:position(n)' strings for logging.
    """
    chains = list(fixer.topology.chains())
    trimmed = []
    for key in list(fixer.missingResidues.keys()):
        chain_idx, pos = key
        n_res = len(list(chains[chain_idx].residues()))
        if pos == 0 or pos == n_res:
            trimmed.append(f"{chains[chain_idx].id}:{'N' if pos == 0 else 'C'}"
                           f"({len(fixer.missingResidues[key])})")
            del fixer.missingResidues[key]
    return trimmed


def _ssbond_line(serial: int, a: tuple, b: tuple) -> str:
    """Format an SSBOND record in the exact columns FFX's PDBFilter reads
    (chain at col 16/30, resSeq 18-21/32-35, iCode 22/36; 1-based)."""
    (ch1, num1, ic1), (ch2, num2, ic2) = a, b
    return (f"SSBOND {serial:3d} CYS {ch1:1s} {num1:>4s}{ic1:1s}   "
            f"CYS {ch2:1s} {num2:>4s}{ic2:1s}\n")


_LABELLED_CYS = None


def _labelled_cys(pdb_id: str) -> set:
    """Residue numbers (as strings) of Cys with an experimental pKa in the manifest."""
    global _LABELLED_CYS
    if _LABELLED_CYS is None:
        _LABELLED_CYS = {}
        if os.path.exists(MANIFEST):
            with open(MANIFEST, newline="") as f:
                for r in csv.DictReader(f):
                    if r["res_name"].strip().upper() == "CYS":
                        _LABELLED_CYS.setdefault(r["pdb_id"].strip().upper(), set()).add(
                            str(r["res_id"]).strip())
    return _LABELLED_CYS.get(pdb_id.upper(), set())


def add_ssbond_records(pdb_path: str, cutoff: float = SSBOND_CUTOFF) -> int:
    """Detect Cys SG–SG pairs within *cutoff* Å and prepend SSBOND records.

    Any existing SSBOND lines are replaced.  Each SG participates in at most one
    bond (closest partner first).  Returns the number of disulfides written.
    """
    with open(pdb_path) as f:
        lines = f.readlines()
    sg = []  # [(key, (x,y,z))]
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "SG" \
                and line[17:20] in ("CYS", "CYX", "CYM"):
            key = (line[21], line[22:26].strip(), line[26])
            sg.append((key, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    pairs = []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            d = sum((p - q) ** 2 for p, q in zip(sg[i][1], sg[j][1])) ** 0.5
            if d <= cutoff:
                pairs.append((d, i, j))
    pairs.sort()
    titrated = _labelled_cys(Path(pdb_path).parent.name)
    used, bonds = set(), []
    for d, i, j in pairs:
        if i in used or j in used:
            continue
        used.update((i, j))
        # A Cys with a measured pKa was titrated as a free thiol (reduced
        # thioredoxin/DsbA/glutaredoxin active sites; the crystal is usually the
        # oxidised form).  Bonding it makes the site non-titratable in FFX and
        # PROPKA returns its 99.99 sentinel, so leave that pair unbonded - in
        # every chain copy, since labels are given for one chain only.
        if sg[i][0][1] in titrated or sg[j][0][1] in titrated:
            continue
        bonds.append((sg[i][0], sg[j][0]))
    body = [l for l in lines if not l.startswith("SSBOND")]
    first_coord = next((k for k, l in enumerate(body)
                        if l.startswith(("ATOM", "HETATM", "MODEL"))), 0)
    ss_lines = [_ssbond_line(n + 1, a, b) for n, (a, b) in enumerate(bonds)]
    with open(pdb_path, "w") as f:
        f.writelines(body[:first_coord] + ss_lines + body[first_coord:])
    return len(bonds)


RELAX = False          # set by --relax
RELAX_K = 10000.0      # kJ/mol/nm^2 on every heavy atom that was in the deposited model


def relax_geometry(fixer, known: set):
    """Restrained Amber14 clean-up of the PDBFixer model (heavy-atom positions returned).

    Why: FFX builds hydrogens from heavy-atom geometry.  Where a side chain is
    nearly straight (deposited that way, e.g. 1YPI GLU B144, or built by
    PDBFixer) or atoms overlap (rebuilt loops, 4HHB), FFX puts HB2/HB3 on the
    same point and the first energy is NaN, or the AMOEBA SCF never converges.

    Atoms in *known* (present in the deposited model) are held by a harmonic
    restraint (0.3 A costs ~4.5 kJ/mol), atoms PDBFixer built are free.
    Charges are zeroed: this is a bonded/steric clean-up only.  With vacuum
    electrostatics left on, charged side chains collapsed onto each other
    (Lys NZ / Glu OE shifts of 1-3 A in 2OXP), which is exactly the geometry a
    pKa model must not have altered.
    Hydrogens are added only for the minimisation and dropped again so FFX
    still assigns its own.  Returns (positions, max displacement of known atoms in A).
    """
    import openmm
    from openmm import unit
    from openmm.app import Modeller, ForceField, CutoffNonPeriodic, Simulation

    ff = ForceField("amber14-all.xml")
    mod = Modeller(fixer.topology, fixer.positions)
    mod.addHydrogens(ff, pH=7.0)
    system = ff.createSystem(mod.topology, nonbondedMethod=CutoffNonPeriodic,
                             nonbondedCutoff=1.2 * unit.nanometer, constraints=None)
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            for i in range(f.getNumParticles()):
                q, sig, eps = f.getParticleParameters(i)
                f.setParticleParameters(i, 0.0, sig, eps)
            for i in range(f.getNumExceptions()):
                a, b, qq, sig, eps = f.getExceptionParameters(i)
                f.setExceptionParameters(i, a, b, 0.0, sig, eps)
    rest = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", RELAX_K)
    for q in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(q)
    pos = mod.positions.value_in_unit(unit.nanometer)
    heavy = []
    for a in mod.topology.atoms():
        if a.element is not None and a.element.symbol == "H":
            continue
        heavy.append(a.index)
        key = (a.residue.chain.index, a.residue.id, a.residue.insertionCode, a.name)
        if key in known:
            rest.addParticle(a.index, list(pos[a.index]))
    system.addForce(rest)
    sim = Simulation(mod.topology, system,
                     openmm.LangevinIntegrator(300, 1.0, 0.001),
                     openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=3000)
    new = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    ref_atoms = list(fixer.topology.atoms())
    mod_atoms = list(mod.topology.atoms())
    if len(heavy) != len(ref_atoms) or any(
            mod_atoms[i].name != r.name for i, r in zip(heavy, ref_atoms)):
        raise RuntimeError("relax: heavy-atom order changed")
    import numpy as np
    old = np.array(pos)
    moved = [np.linalg.norm(new[i] - old[i]) for i in heavy
             if (mod_atoms[i].residue.chain.index, mod_atoms[i].residue.id,
                 mod_atoms[i].residue.insertionCode, mod_atoms[i].name) in known]
    out = unit.Quantity([openmm.Vec3(*new[i]) for i in heavy], unit.nanometer)
    return out, 10.0 * float(max(moved)) if moved else 0.0


def fix_one(pdb_id, raw_path, fixed_path):
    log = {k: "" for k in LOG_FIELDS}
    log["pdb_id"] = pdb_id
    stripped_tmp = None
    noh_tmp      = None
    prepro_tmp   = None

    try:
        input_path = raw_path

        # ── Strip to model 1 for NMR ensembles ───────────────────────────────
        if has_model_records(raw_path):
            stripped_tmp = raw_path.replace(".pdb", "_model1_tmp.pdb")
            n_atoms = strip_to_model1(raw_path, stripped_tmp)
            input_path = stripped_tmp
            log["multi_model_stripped"] = f"yes ({n_atoms} ATOM/HETATM lines)"

            if n_atoms == 0:
                log["status"] = "FAILED"
                log["notes"] = "MODEL strip produced 0 atoms — malformed PDB"
                return log

        # ── Strip all hydrogens before PDBFixer ───────────────────────────────
        noh_tmp = input_path.replace(".pdb", "_noh_tmp.pdb")
        n_heavy = strip_hydrogens(input_path, noh_tmp)
        if n_heavy == 0:
            log["status"] = "FAILED"
            log["notes"] = "H-strip produced 0 heavy atoms — malformed PDB"
            return log
        input_path = noh_tmp

        # ── Remap / strip non-standard residues before PDBFixer ───────────────
        prepro_tmp = input_path.replace(".pdb", "_prepro_tmp.pdb")
        remapped, stripped = preprocess_residues(input_path, prepro_tmp)
        input_path = prepro_tmp
        if remapped:
            log["notes"] = (log["notes"] + " remapped:" + ",".join(remapped)).strip()
        if stripped:
            log["notes"] = (log["notes"] + " stripped:" + ",".join(sorted(set(stripped)))).strip()

        # ── PDBFixer ──────────────────────────────────────────────────────────
        fixer = PDBFixer(filename=input_path)

        if fixer.topology is None:
            log["status"] = "FAILED"
            log["notes"] = "PDBFixer returned None topology"
            return log

        fixer.removeHeterogens(keepWater=False)

        fixer.findNonstandardResidues()
        ns = fixer.nonstandardResidues
        if ns:
            log["nonstandard_replaced"] = "; ".join(
                f"{r.name}{r.id}" for r, _ in ns
            )
        fixer.replaceNonstandardResidues()

        fixer.findMissingResidues()
        n_missing_res = sum(len(v) for v in fixer.missingResidues.values())
        log["missing_residues_count"] = n_missing_res

        # Drop unresolved N/C-terminal tails: PDBFixer builds them as extended
        # strands that clash, slow ManyBody, and trigger SCF failures.  Internal
        # gaps are anchored at both ends and are kept (leaving a chain break
        # would let FFX bond across the gap).
        trimmed = trim_terminal_missing_residues(fixer)
        if trimmed:
            log["notes"] = (log["notes"] + " trimmed_termini:" + ",".join(trimmed)).strip()

        large_gaps = [
            f"chain {k[0]} pos {k[1]} ({len(v)} residues)"
            for k, v in fixer.missingResidues.items()
            if len(v) > LARGE_GAP_WARN
        ]
        log["large_missing_residues"] = "; ".join(large_gaps)

        fixer.findMissingAtoms()
        n_missing_atoms = (
            sum(len(v) for v in fixer.missingAtoms.values()) +
            len(fixer.missingTerminals)
        )
        log["missing_atoms_count"] = n_missing_atoms
        known = {(a.residue.chain.index, a.residue.id, a.residue.insertionCode, a.name)
                 for a in fixer.topology.atoms()}
        fixer.addMissingAtoms()

        if RELAX:
            try:
                fixer.positions, dmax = relax_geometry(fixer, known)
                log["notes"] = (log["notes"] + f" relaxed(max_shift={dmax:.2f}A)").strip()
            except Exception:
                log["notes"] = (log["notes"] + " relax_failed:"
                                + traceback.format_exc().splitlines()[-1][:80]).strip()

        # Clear periodic box so no CRYST1 record is written
        fixer.topology.setPeriodicBoxVectors(None)

        with open(fixed_path, "w") as f:
            # keepIds=True preserves original residue numbering from the raw PDB.
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

        # FFX builds disulfides ONLY from SSBOND records (CONECT is ignored), so
        # without this every disulfide is broken and both Cys get an HG.
        n_ss = add_ssbond_records(fixed_path)
        log["notes"] = (log["notes"] + f" ssbonds:{n_ss}").strip()

        log["status"] = "ok"

    except Exception:
        log["status"] = "FAILED"
        log["notes"] = traceback.format_exc().splitlines()[-1]

    finally:
        for tmp in (stripped_tmp, noh_tmp, prepro_tmp):
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    return log


def main():
    import argparse
    ap = argparse.ArgumentParser(description="PDBFixer prep for FFX (SSBOND-aware)")
    ap.add_argument("--out-dir", default=FIXED_DIR)
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--only", nargs="+", default=None, help="Process only these PDB IDs")
    ap.add_argument("--all-raw", action="store_true",
                    help="Process every raw PDB, not just those in the manifest")
    ap.add_argument("--relax", action="store_true",
                    help="restrained Amber14 geometry clean-up before writing (use for inputs "
                         "where FFX hydrogen building or the AMOEBA SCF fails at step 0)")
    ap.add_argument("--force", action="store_true", help="overwrite existing *_input.pdb")
    ap.add_argument("--patch-ssbond", action="store_true",
                    help="Only add SSBOND records to existing *_input.pdb files in --out-dir")
    args = ap.parse_args()
    fixed_dir = args.out_dir
    global RELAX
    RELAX = args.relax

    if args.patch_ssbond:
        for p in sorted(Path(fixed_dir).glob("*/*_input.pdb")):
            if args.only and p.parent.name.upper() not in {x.upper() for x in args.only}:
                continue
            print(f"  {p.parent.name}: {add_ssbond_records(str(p))} SSBOND records")
        return

    os.makedirs(fixed_dir, exist_ok=True)
    raw_pdbs = sorted(
        p for p in Path(RAW_DIR).glob("*.pdb")
        if "_tmp" not in p.name.lower()
    )

    if not raw_pdbs:
        raise SystemExit(f"No PDB files found in {RAW_DIR}.")

    raw_pdbs = [p for p in raw_pdbs if p.stem.upper() not in EXCLUDE]
    if not args.all_raw and os.path.exists(MANIFEST):
        import pandas as pd
        wanted = set(pd.read_csv(MANIFEST)["pdb_id"].astype(str).str.upper())
        raw_pdbs = [p for p in raw_pdbs if p.stem.upper() in wanted]
    if args.only:
        only = {x.upper() for x in args.only}
        raw_pdbs = [p for p in raw_pdbs if p.stem.upper() in only]
    print(f"Fixing {len(raw_pdbs)} structures → {fixed_dir}  (hard-excluded: {sorted(EXCLUDE)})")

    log_exists = os.path.exists(args.log) and os.path.getsize(args.log) > 0
    with open(args.log, "a", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
        if not log_exists:
            writer.writeheader()

        for raw_path in raw_pdbs:
            pdb_id = raw_path.stem.upper()
            pdb_out_dir = os.path.join(fixed_dir, pdb_id)
            fixed_path = os.path.join(pdb_out_dir, f"{pdb_id}_input.pdb")

            if os.path.exists(fixed_path) and not args.force:
                print(f"  [skip] {pdb_id} already fixed")
                continue

            os.makedirs(pdb_out_dir, exist_ok=True)

            log = fix_one(pdb_id, str(raw_path), fixed_path)
            writer.writerow(log)
            logf.flush()

            tag = " [NMR→model1]" if log["multi_model_stripped"] else ""
            sym = "✓" if log["status"] == "ok" else "✗"
            print(
                f"  [{sym}] {pdb_id:6s}{tag}"
                f"  miss_res={str(log['missing_residues_count']):>3}"
                f"  miss_atoms={str(log['missing_atoms_count']):>3}"
                f"  nonstd={log['nonstandard_replaced'] or '-'}"
                f"  {log['notes'] or ''}"
            )

    print(f"\nLog → {args.log}")
    print("Next step: 03_run_ffx_minimize.py")


if __name__ == "__main__":
    main()
