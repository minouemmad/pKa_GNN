
import os
import re
import time
import urllib.request
import urllib.error
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
CSV_PATH        = "1-PKAD-R-2025-09-03.csv"        # path to PKAD-R file
OUT_DIR         = "data"
PDB_DIR         = os.path.join(OUT_DIR, "raw_pdbs")
MANIFEST_PATH   = os.path.join(OUT_DIR, "manifest.csv")

# Ionizable residues the paper models (CYS is new)
TARGET_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"}

# Set True to keep only the "Main" classification (recommended for first run)
MAIN_ONLY = True

# RCSB PDB download URL template
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

SLEEP_BETWEEN_DOWNLOADS = 0.3
# ─────────────────────────────────────────────────────────────────────────────

def parse_pka(raw: str):
    
    raw = str(raw).strip()
    if re.match(r'^[<>~]', raw):
        return None
    try:
        return float(raw)
    except ValueError:
        return None

def load_and_filter(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")

    # 1. Residue type filter
    df = df[df["ResName"].isin(TARGET_RESIDUES)].copy()
    print(f"  After residue filter ({', '.join(sorted(TARGET_RESIDUES))}): {len(df)} rows")

    # 2. Classification filter
    if MAIN_ONLY:
        df = df[df["pKa Classification"] == "Main"].copy()
        print(f"  After Main-only filter: {len(df)} rows")

    # 3. Drop rows with structural warnings (e.g. "ResID NOT exist")
    warn_mask = df["Warning"].fillna("").str.contains("NOT exist", case=False)
    n_warn = warn_mask.sum()
    df = df[~warn_mask].copy()
    print(f"  Dropped {n_warn} rows with 'NOT exist' warnings: {len(df)} rows remain")

    # 4. Parse pKa — drop inequalities / approximates
    df["pKa_float"] = df["Expt. pKa"].apply(parse_pka)
    n_bad = df["pKa_float"].isna().sum()
    df = df.dropna(subset=["pKa_float"]).copy()
    print(f"  Dropped {n_bad} rows with non-numeric pKa: {len(df)} rows remain")

    # 5. Drop C/N-terminal entries (paper focuses on sidechain ionisable groups)
    cterm_mask = df["Warning"].fillna("").str.contains("C/N-term", case=False)
    n_cterm = cterm_mask.sum()
    df = df[~cterm_mask].copy()
    print(f"  Dropped {n_cterm} C/N-terminal entries: {len(df)} rows remain")

    return df

def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the columns we need downstream."""
    manifest = df[[
        "Index", "Protein Name", "PDB", "Chain",
        "ResID in PDB", "ResName", "pKa_float",
        "Sequence Identity > 30%", "Sequence Identity > 90%",
        "Expt. Uncertainty", "Expt. Method"
    ]].copy()

    manifest.columns = [
        "pkad_index", "protein_name", "pdb_id", "chain",
        "res_id", "res_name", "pka",
        "seq_id_30", "seq_id_90",
        "uncertainty", "method"
    ]

    manifest["pdb_id"] = manifest["pdb_id"].str.upper().str.strip()
    manifest["res_name"] = manifest["res_name"].str.upper().str.strip()
    manifest["chain"] = manifest["chain"].str.strip()

    return manifest.reset_index(drop=True)

def download_pdbs(pdb_ids: list[str], pdb_dir: str):
    os.makedirs(pdb_dir, exist_ok=True)
    failed = []

    for pdb_id in sorted(set(pdb_ids)):
        dest = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if os.path.exists(dest):
            print(f"  [skip] {pdb_id}.pdb already downloaded")
            continue

        url = RCSB_URL.format(pdb_id=pdb_id)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [ok]   {pdb_id}.pdb")
        except urllib.error.HTTPError as e:
            print(f"  [FAIL] {pdb_id}: HTTP {e.code}")
            failed.append(pdb_id)
        except Exception as e:
            print(f"  [FAIL] {pdb_id}: {e}")
            failed.append(pdb_id)

        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    if failed:
        print(f"\n  WARNING: {len(failed)} PDB(s) could not be downloaded:")
        for f in failed:
            print(f"    {f}")
    return failed

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse and filter
    df_filtered = load_and_filter(CSV_PATH)
    manifest    = build_manifest(df_filtered)

    # Save manifest
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest saved → {MANIFEST_PATH}  ({len(manifest)} entries)")

    # Print residue breakdown
    print("\nResidues in manifest:")
    print(manifest["res_name"].value_counts().to_string())

    # Download PDBs
    unique_pdbs = manifest["pdb_id"].unique().tolist()
    print(f"\nDownloading {len(unique_pdbs)} unique PDB structures...")
    failed = download_pdbs(unique_pdbs, PDB_DIR)

    # Remove manifest rows for failed downloads
    if failed:
        before = len(manifest)
        manifest = manifest[~manifest["pdb_id"].isin(failed)].copy()
        manifest.to_csv(MANIFEST_PATH, index=False)
        print(f"  Removed {before - len(manifest)} manifest rows for failed downloads.")

    print("\nDone.")

if __name__ == "__main__":
    main()

