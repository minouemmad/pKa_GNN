import os
import sys
import re
import glob
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Global base directory
BASE_DIR = Path("../Graph_pKa")

def main():
    global BASE_DIR

    # FFX pipeline layout:
    #   Adjacency:   Features/Adjacency_Matrices/With_Self_Loop/{key}_r{radius}_adjacency.csv
    #   Node feats:  Features/Node_Feature_Vectors/{radius}/{key}.csv
    adj_dir  = BASE_DIR / 'Features' / 'Adjacency_Matrices' / 'With_Self_Loop'
    nf_root  = BASE_DIR / 'Features' / 'Node_Feature_Vectors'
    radii    = [7, 8, 9, 10, 11]

    output_dir = BASE_DIR / "Features" / "Node_Feature_Vectors" / "Prediction_Datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, radius in enumerate(radii):
        node_features_dir = nf_root / str(radius)
        if not node_features_dir.exists():
            print(f"Node feature dir not found, skipping: {node_features_dir}")
            continue

        print(f"\nProcessing radius = {radius} Å  ({node_features_dir})")
        data_list = []

        for nf_csv in sorted(node_features_dir.glob("*.csv")):
            key = nf_csv.stem   # e.g. 135L_A_42.ASP
            adj_csv = adj_dir / f"{key}_r{radius}_adjacency.csv"

            if not adj_csv.exists():
                print(f"  Warning: adjacency not found for {key} at r={radius}. Skipping...")
                continue

            adjacency_matrix = pd.read_csv(adj_csv, header=0, index_col=0).values
            adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.int)

            node_features = pd.read_csv(nf_csv, header=0)
            if 'atom_label' not in node_features.columns:
                print(f"  Missing 'atom_label' in {nf_csv.name}. Skipping...")
                continue

            # Parse key: {pdb_id}_{chain}_{seq}.{res_name}
            parts = key.rsplit('.', 1)
            residue_name = parts[1] if len(parts) == 2 else ""
            info_parts   = parts[0].rsplit('_', 2)
            pdb_id        = info_parts[0] if len(info_parts) >= 1 else key
            chain_id      = info_parts[1] if len(info_parts) >= 2 else ""
            residue_number = int(info_parts[2]) if len(info_parts) >= 3 else 0

            pKa_labels_tensor = None
            if 'Expt. pKa' in node_features.columns and len(node_features['Expt. pKa'].dropna()) > 0:
                pKa_labels_tensor = torch.tensor([node_features['Expt. pKa'].values[0]], dtype=torch.float)

            atom_label_encoded = F.one_hot(
                torch.tensor(node_features['atom_label'].values, dtype=torch.long), num_classes=9
            ).float()
            one_hot_cols  = node_features.filter(like='Residue Name_').values
            residue_label = np.argmax(one_hot_cols, axis=1)[0]

            cols_to_drop = [c for c in ['Expt. pKa', 'atom_label'] if c in node_features.columns]
            features_tensor = torch.tensor(
                node_features.drop(columns=cols_to_drop).values, dtype=torch.float
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
            data.Chain_ID       = chain_id
            data.Residue_Number = residue_number
            data.Residue_Name   = residue_name

            data_list.append(data)

        print(f"  Generated {len(data_list)} graphs for radius {radius} Å")
        pkl_path = output_dir / f'data_list_{idx}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"  Saved → {pkl_path.name}")

# Define the model class for predictions
class GATConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout, heads):
        super(GATConv, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.pooling_function = global_mean_pool
        self.out_layer = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, data):
        edge_index = data.edge_index
        batch = data.batch
        x = data.x

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pooling_function(x, batch)
        x = self.out_layer(x)
        return x

def inspect_checkpoint(model_path):
    """Inspect the structure of the checkpoint file."""
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    print(f"Keys in the checkpoint: {list(checkpoint.keys())}")
    return checkpoint

def predict_and_save(model, data_list, output_csv):
    """Predict and save results with optional labels."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions = []

    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)

            for i in range(len(out)):
                pred_dict = {
                    "PDB_ID": batch.PDB_ID[i],
                    "Chain_ID": batch.Chain_ID[i],
                    "Residue_Number": int(batch.Residue_Number[i]),
                    "Residue": batch.Residue_Name[i],
                    "Predicted_pKa": out[i].item()
                }
                # Add True_pKa only if it exists (optional labels)
                if batch.y is not None:
                    pred_dict["True_pKa"] = batch.y[i].item()
                predictions.append(pred_dict)

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    return predictions

def predict_unlabeled_data(model, data_list, output_csv):
    """Make predictions on feature vectors without experimental pKa labels."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions = []

    loader = DataLoader(data_list, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)

            for i in range(len(out)):
                predictions.append({
                    "PDB_ID": batch.PDB_ID[i],
                    "Chain_ID": batch.Chain_ID[i],
                    "Residue_Number": int(batch.Residue_Number[i]),
                    "Residue": batch.Residue_Name[i],
                    "Predicted_pKa": out[i].item()
                })

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    return predictions

def average_predictions_across_folds(predictions_dir, num_folds=10):
    """Average predictions from all folds for each dataset.
    
    Args:
        predictions_dir: Directory containing prediction CSV files
        num_folds: Number of folds (default 10)
    """
    import glob
    
    # Get all unique dataset numbers
    prediction_files = glob.glob(os.path.join(predictions_dir, "predictions_dataset_*.csv"))
    
    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}")
        return
    
    # Extract dataset numbers
    dataset_nums = set()
    for file in prediction_files:
        basename = os.path.basename(file)
        # Extract number from "predictions_dataset_X_best_model_fold_Y.csv"
        match = re.search(r'dataset_(\d+)_', basename)
        if match:
            dataset_nums.add(int(match.group(1)))
    
    # For each dataset, average predictions across all folds
    for dataset_num in sorted(dataset_nums):
        print(f"\nAveraging predictions for dataset {dataset_num}...")
        
        fold_dfs = []
        for fold in range(1, num_folds + 1):  # Folds are numbered 1-10, not 0-9
            fold_file = os.path.join(predictions_dir, f"predictions_dataset_{dataset_num}_best_model_fold_{fold}.csv")
            if os.path.exists(fold_file):
                df = pd.read_csv(fold_file)
                fold_dfs.append(df)
            else:
                print(f"  Warning: {fold_file} not found")
        
        if not fold_dfs:
            print(f"  No fold predictions found for dataset {dataset_num}")
            continue
        
        # Combine all fold predictions and group by residue identifiers
        combined_df = pd.concat(fold_dfs, ignore_index=True)
        
        # Group by PDB_ID, Chain_ID, and Residue_Number to average predictions
        grouped = combined_df.groupby(['PDB_ID', 'Chain_ID', 'Residue_Number', 'Residue']).agg({
            'Predicted_pKa': 'mean'
        }).reset_index()
        
        # If True_pKa exists, average that too
        if 'True_pKa' in combined_df.columns:
            true_pka_grouped = combined_df.groupby(['PDB_ID', 'Chain_ID', 'Residue_Number', 'Residue']).agg({
                'True_pKa': 'first'  # All folds have the same true value
            }).reset_index()
            grouped['True_pKa'] = true_pka_grouped['True_pKa']
        
        # Save averaged predictions
        output_subdir = os.path.join(predictions_dir, "Dataset_Averaged_Predictions")
        os.makedirs(output_subdir, exist_ok=True)
        output_file = os.path.join(output_subdir, f"predictions_dataset_{dataset_num}_averaged.csv")
        grouped.to_csv(output_file, index=False)
        print(f"  ✓ Saved averaged predictions to {output_file}")
        print(f"    - {len(grouped)} unique residues")
        print(f"    - Mean Predicted_pKa: {grouped['Predicted_pKa'].mean():.2f}")
        if 'True_pKa' in grouped.columns:
            print(f"    - Mean True_pKa: {grouped['True_pKa'].mean():.2f}")

def predict_with_all_models(model_dir, data_sets):
    """Predict and save results for all models with optional metrics calculation."""
    global BASE_DIR
    device = torch.device("cpu")
    output_dir = os.path.join(BASE_DIR, "Results/Predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pth files in the directory
    model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))

    metrics = []  # To store MAE and RMSE for each model and dataset

    for dataset_idx, data_list in enumerate(data_sets):
        print(f"Processing Dataset {dataset_idx + 1}...")

        for model_path in model_paths:
            print(f"Loading model: {model_path}")

            # Inspect the checkpoint structure
            checkpoint = inspect_checkpoint(model_path)

            # Load model
            model = GATConv(input_dim=data_list[0].x.shape[1], hidden_channels=48, dropout=0.3, heads=6)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()

            # Predict and save
            model_name = os.path.basename(model_path).replace(".pth", "")
            output_csv = os.path.join(output_dir, f"predictions_dataset_{dataset_idx + 1}_{model_name}.csv")
            predictions = predict_and_save(model, data_list, output_csv)

            # Calculate MAE and RMSE only if True_pKa exists in predictions
            if predictions and "True_pKa" in predictions[0]:
                true_labels = [pred["True_pKa"] for pred in predictions]
                predicted_labels = [pred["Predicted_pKa"] for pred in predictions]
                mae = mean_absolute_error(true_labels, predicted_labels)
                rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))

                metrics.append({"Dataset": dataset_idx + 1, "Model": model_name, "MAE": mae, "RMSE": rmse})
                print(f"Dataset {dataset_idx + 1}, Model: {model_name}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            else:
                print(f"Dataset {dataset_idx + 1}, Model: {model_name} - No true labels (unlabeled data)")

    # Save metrics to a CSV file if metrics exist
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_csv = os.path.join(output_dir, "model_metrics_all_datasets.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Metrics saved to {metrics_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pKa dataset PKL files and make predictions")
    parser.add_argument("--base_dir", type=str, default="../Graph_pKa",
                        help="Base directory containing the data folders")
    parser.add_argument("--model_dir", type=str, 
                        default="Net/GAT_Model",
                        help="Directory containing trained model files")
    args = parser.parse_args()
    
    # Initialize global base directory
    BASE_DIR = Path(args.base_dir).resolve()
    
    # Generate datasets
    print("=" * 60)
    print("STEP 1: Generate PyTorch Geometric Datasets")
    print("=" * 60)
    main()
    
    # Make predictions with trained models
    print("\n" + "=" * 60)
    print("STEP 2: Make Predictions with Trained Models")
    print("=" * 60)
    from Net.GNN_Grid_Search.GAT import load_training_data
    
    model_dir = os.path.join(BASE_DIR, args.model_dir)
    dataset_dir = os.path.join(BASE_DIR, "Features/Node_Feature_Vectors/Prediction_Datasets")
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    data_sets, input_dim, Residue_Type_labels = load_training_data(dataset_dir)
    predict_with_all_models(model_dir, data_sets)
    
    # Average predictions across all folds
    predictions_dir = os.path.join(BASE_DIR, "Results/Predictions")
    print("\n" + "="*80)
    print("STEP 3: Average Predictions Across All Folds")
    print("="*80)
    average_predictions_across_folds(predictions_dir, num_folds=10)
