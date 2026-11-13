
import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data

RADII = [7, 8, 9, 10, 11]

# FFX mode

def _parse_key_ffx(key: str):
    """Parse '{PDB_ID}_{chain}_{seq}.{res_name}' key.

    Returns (pdb_id, chain_id, res_number, res_name).
    """
    parts = key.rsplit(".", 1)
    res_name = parts[1] if len(parts) == 2 else ""
    info_parts = parts[0].rsplit("_", 2)
    pdb_id   = info_parts[0] if len(info_parts) >= 1 else key
    chain_id = info_parts[1] if len(info_parts) >= 2 else ""
    res_num  = int(info_parts[2]) if len(info_parts) >= 3 else 0
    return pdb_id, chain_id, res_num, res_name

def main_ffx(base_dir: str):
    """Build PKL datasets from FFX pipeline outputs."""
    base = Path(base_dir)
    adj_dir    = base / "Features" / "Adjacency_Matrices" / "With_Self_Loop"
    nf_root    = base / "Features" / "Node_Feature_Vectors"
    output_dir = base / "Features" / "Node_Feature_Vectors" / "Subsets"
    output_dir.mkdir(parents=True, exist_ok=True)

    for radius in RADII:
        nf_dir = nf_root / str(radius)
        if not nf_dir.exists():
            print(f"Node feature dir not found, skipping: {nf_dir}")
            continue

        data_list = []
        print(f"\nProcessing radius = {radius} Å  ({nf_dir})")

        for nf_csv in sorted(nf_dir.glob("*.csv")):
            key = nf_csv.stem
            adj_csv = adj_dir / f"{key}_r{radius}_adjacency.csv"

            if not adj_csv.exists():
                print(f"  Skipping {key}: adjacency matrix not found")
                continue

            adj_mat    = pd.read_csv(adj_csv, header=0, index_col=0).values
            adj_tensor = torch.tensor(adj_mat, dtype=torch.int)

            nf = pd.read_csv(nf_csv, header=0)
            if "atom_label" not in nf.columns:
                print(f"  Skipping {key}: missing 'atom_label' column")
                continue

            # pKa label is optional (unlabelled proteins allowed)
            pka_col = "Expt. pKa"
            pka_tensor = None
            if pka_col in nf.columns and len(nf[pka_col].dropna()) > 0:
                pka_tensor = torch.tensor([nf[pka_col].values[0]], dtype=torch.float)

            atom_label_enc = F.one_hot(
                torch.tensor(nf["atom_label"].values.astype(int), dtype=torch.long),
                num_classes=9,
            ).float()

            one_hot_cols  = nf.filter(like="Residue Name_").values
            residue_label = int(np.argmax(one_hot_cols, axis=1)[0])

            drop_cols = [c for c in [pka_col, "atom_label"] if c in nf.columns]
            feat_tensor = torch.tensor(
                nf.drop(columns=drop_cols).values.astype(float), dtype=torch.float
            )
            feat_tensor = torch.cat([feat_tensor, atom_label_enc], dim=1)

            edge_index = adj_tensor.nonzero(as_tuple=True)
            edge_index = torch.stack(edge_index, dim=0)

            pdb_id, chain_id, res_num, res_name = _parse_key_ffx(key)

            data = Data(
                x=feat_tensor,
                edge_index=edge_index,
                y=pka_tensor,
                residue_label=residue_label,
            )
            data.PDB_ID         = pdb_id
            data.Chain_ID       = chain_id
            data.Residue_Number = res_num
            data.Residue_Name   = res_name

            data_list.append(data)

        pkl_path = output_dir / f"data_list_r{radius}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(data_list, fh)
        print(f"Saved {len(data_list)} graphs → {pkl_path.name}")

# Legacy PKAD / Tinker mode (unchanged logic, kept for reference)

def main_pkad(base_dir: str):
    adjacency_matrix_dir = os.path.join(base_dir, 'Adj_Matrix/With_Self_Loop')
    node_feature_dirs = [
        os.path.join(base_dir, f'4_Residues_W_Local_Frame/{r}')
        for r in RADII
    ]
    output_dir = os.path.join(base_dir, "4_Residues_W_Local_Frame/Subsets")
    os.makedirs(output_dir, exist_ok=True)

    for idx, node_features_dir in enumerate(node_feature_dirs):
        print(f"\nProcessing node features from: {node_features_dir}")
        data_list = []

        for filename in os.listdir(adjacency_matrix_dir):
            if not filename.endswith('.csv'):
                continue
            adjacency_matrix = pd.read_csv(
                os.path.join(adjacency_matrix_dir, filename), header=0, index_col=0).values
            adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.int)

            features_filename   = filename.replace('_adjacency.csv', '.csv')
            node_features_path  = os.path.join(node_features_dir, features_filename)

            if not os.path.exists(node_features_path):
                print(f"Warning: {features_filename} not found. Skipping...")
                continue

            base_name = features_filename.replace('.csv', '')
            pdb_id, residue_info = base_name.split('_')
            residue_number, residue_name = residue_info.split('.')

            node_features = pd.read_csv(node_features_path, header=0)
            if 'Expt.pKa' not in node_features or 'atom_label' not in node_features:
                continue
            if len(node_features['Expt.pKa']) == 0:
                continue

            pKa_labels_tensor = torch.tensor([node_features['Expt.pKa'].values[0]], dtype=torch.float)
            atom_label_encoded = F.one_hot(
                torch.tensor(node_features['atom_label'].values, dtype=torch.long), num_classes=9
            ).float()
            one_hot_cols  = node_features.filter(like='Residue Name_').values
            residue_label = np.argmax(one_hot_cols, axis=1)[0]

            features_tensor = torch.tensor(
                node_features.drop(columns=['Expt.pKa', 'atom_label']).values, dtype=torch.float
            )
            features_tensor = torch.cat([features_tensor, atom_label_encoded], dim=1)

            edge_index = adjacency_tensor.nonzero(as_tuple=True)
            edge_index = torch.stack(edge_index, dim=0)

            data = Data(
                x=features_tensor,
                edge_index=edge_index,
                y=pKa_labels_tensor,
                residue_label=residue_label,
            )
            data.PDB_ID         = pdb_id
            data.Residue_Number = residue_number
            data.Residue_Name   = residue_name
            data_list.append(data)

        pkl_path = os.path.join(output_dir, f'data_list_{idx}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Saved dataset {idx} with {len(data_list)} graphs to {pkl_path}")

# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pKa dataset PKL files")
    parser.add_argument(
        "--base_dir", type=str, default="..",
        help="Root Graph_pKa directory (default: parent of Net/)",
    )
    parser.add_argument(
        "--mode", choices=["ffx", "pkad"], default="ffx",
        help="'ffx' = new FFX pipeline layout (default); 'pkad' = legacy Tinker/PKAD_Data layout",
    )
    args = parser.parse_args()

    if args.mode == "ffx":
        main_ffx(args.base_dir)
    else:
        main_pkad(os.path.join(args.base_dir, "PKAD_Data"))
