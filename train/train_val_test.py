import sys
import os
import argparse # Added for command-line arguments
# Add project root to Python path to allow sibling module imports (dataset, model)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.profiler # For PyTorch Profiler
from torch.profiler import record_function # Import for explicit recording blocks
import csv
from datetime import datetime, timedelta # Added import
import torch.nn.functional as F
import pandas as pd # Added for MultiIndex DataFrame
import numpy as np # Added for np.isnan
# import torch.distributions # Not explicitly used in the notebook's training script part
from sklearn.metrics import matthews_corrcoef, f1_score, ndcg_score
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler # Added for scaling y_true for NDCG
from torch.utils.data import DataLoader # Added for DataLoader optimization

from dataset.graph_dataset_gen import MyDataset # Corrected import
from model.Multi_GDNN import MGDPR

# Configure the device for running the model on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train MGDPR model with configurable DataLoader and Profiler parameters.")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2 if os.cpu_count() else 4, help='Number of worker processes for data loading.')
parser.add_argument('--pin_memory', type=lambda x: (str(x).lower() == 'true'), default=True, help='Pin memory for faster CPU to GPU data transfer (True/False).')
parser.add_argument('--use_amp', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable Automatic Mixed Precision (AMP) training (True/False).')

# Profiler arguments
parser.add_argument('--profile', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable PyTorch profiler (True/False).')
parser.add_argument('--profile-steps', type=int, default=5, help='Number of active steps/batches for the profiler to record.')
parser.add_argument('--profile-wait', type=int, default=1, help='Number of wait steps before profiler starts recording.')
parser.add_argument('--profile-warmup', type=int, default=1, help='Number of warmup steps before active recording.')
parser.add_argument('--profile-epoch', type=int, default=0, help='Target epoch index (0-based) to run profiling.')
parser.add_argument('--profile-log-dir', type=str, default="./mgdpr_profiler_logs", help='Directory to save profiler trace files.')

args = parser.parse_args()

# --- Configuration Section ---
# These paths and parameters are from the demo.ipynb and may need adjustment for local execution.
# Please ensure the data paths point to your local dataset.

# Date ranges for training, validation, and testing
# TODO: IMPORTANT - Update these date ranges to match your dataset's actual time coverage!
# Example: If your data is from 2018 to 2023
train_sedate = ['2018-01-01', '2021-12-31'] # Replace with your actual training start and end dates
val_sedate = ['2022-01-01', '2022-12-31']   # Replace with your actual validation start and end dates
test_sedate = ['2023-01-01', '2023-12-31']  # Replace with your actual test start and end dates

# Market and dataset types
market_names = ['NASDAQ', 'NYSE', 'SSE'] # Example markets from notebook
current_market = "Shortlist" # Using a specific name for your shortlist data

dataset_types = ['Train', 'Validation', 'Test']

# Paths to company list CSVs (e.g., NASDAQ.csv, NYSE.csv)
# TODO: Update these paths to your local data storage
com_list_csv_paths = [
    # Example: 'data/company_lists/NASDAQ.csv',
    #          'data/company_lists/NYSE.csv',
    #          'data/company_lists/NYSE_missing.csv'
    # Using placeholder for NASDAQ list as per notebook example for now
    './NASDAQ_example.csv' # Create a dummy NASDAQ.csv or update this path
]

# Root directory for individual stock data CSVs (e.g., market_ticker_30Y.csv)
# MyDataset expects files like: os.path.join(root_data_dir, f'{market}_{ticker}_30Y.csv')
# TODO: Update this path to your local raw stock data CSV file
root_data_dir = "/workspaces/ai_testground/shortlist_stocks-1.csv" # Path to the single stock data CSV file

# Destination directory for generated graph .pt files
# MyDataset will create subfolders like: os.path.join(graph_dest_dir, f'{market}_{type}_{start}_{end}_{window}')
# TODO: Update this path to where you want processed graph data saved
graph_dest_dir = "./graph_data_processed" # Example: "data/processed_graph_data"

# --- DataLoader Configuration (from parsed arguments) ---
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
PIN_MEMORY = args.pin_memory
print(f"DataLoader Configuration: BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, PIN_MEMORY={PIN_MEMORY}")


# --- Load Company Lists ---
# This section is based on the notebook's logic for loading company lists.
# Ensure the CSV files at com_list_csv_paths exist and are formatted correctly (one ticker per line).

# For the example, we'll create a dummy NASDAQ_example.csv if it doesn't exist
# and populate a default NASDAQ company list.
nasdaq_com_list_example = ["AAPL", "MSFT", "GOOGL"] # Default if no company list AND no data CSV found initially

# Attempt to load company list from specified CSV path
# If the primary company list CSV does not exist, or if it's the default dummy path AND a custom data CSV is provided,
# try to derive tickers from the main data CSV.
primary_com_list_path = com_list_csv_paths[0] if com_list_csv_paths else None
target_com_list = []
attempt_derive_from_data_csv = True

# Condition to prioritize deriving from data_csv:
# 1. No primary_com_list_path specified.
# 2. primary_com_list_path does not exist.
# 3. primary_com_list_path IS the default './NASDAQ_example.csv' AND root_data_dir is NOT the default './dummy_stock_data.csv' (user provided own data).

if not primary_com_list_path:
    print("No company list CSV specified.")
    attempt_derive_from_data_csv = True
elif not os.path.exists(primary_com_list_path):
    print(f"Warning: Company list CSV '{primary_com_list_path}' not found.")
    attempt_derive_from_data_csv = True
elif primary_com_list_path == './NASDAQ_example.csv' and root_data_dir != './dummy_stock_data.csv':
    print(f"Default company list './NASDAQ_example.csv' specified, but a custom data CSV '{root_data_dir}' is used. Prioritizing deriving tickers from data CSV.")
    attempt_derive_from_data_csv = True
else:
    # Load from the specified (non-default or default-but-no-custom-data) company list path
    print(f"Loading company list from: {primary_com_list_path}")
    try:
        with open(primary_com_list_path, 'r') as f:
            file_reader = csv.reader(f)
            for line in file_reader:
                if line: # Ensure line is not empty
                    target_com_list.append(line[0].strip())
        if not target_com_list:
            print(f"Warning: Company list file {primary_com_list_path} is empty. Will attempt to derive from data CSV.")
            attempt_derive_from_data_csv = True # If file exists but is empty
    except Exception as e:
        print(f"Error reading company list {primary_com_list_path}: {e}. Will attempt to derive from data CSV.")
        attempt_derive_from_data_csv = True

# If conditions met (or loading failed/empty), try to derive it from the main data CSV (root_data_dir)
if attempt_derive_from_data_csv and not target_com_list: # Only if target_com_list isn't already populated
    print(f"Attempting to derive company list from data CSV: {root_data_dir}")
    if os.path.exists(root_data_dir):
        try:
            # Read only the 'Ticker' column to get unique tickers
            # This assumes the CSV has a header and 'Ticker' (or similar) is a column name.
            # The MyDataset class handles more complex header/no-header logic,
            # but for deriving comlist, a simpler read is attempted here.
            
            # First, try to infer headers and find 'Ticker' or 'ticker'
            df_for_tickers = pd.read_csv(root_data_dir, usecols=lambda col_name: str(col_name).strip().lower() in ['ticker', 'symbol'], low_memory=False)
            
            if 'Ticker' in df_for_tickers.columns:
                target_com_list = df_for_tickers['Ticker'].dropna().unique().tolist()
            elif 'ticker' in df_for_tickers.columns: # Fallback for lowercase
                 target_com_list = df_for_tickers['ticker'].dropna().unique().tolist()
            elif not df_for_tickers.empty and df_for_tickers.columns[0].lower() not in ['ticker', 'symbol']:
                # If no 'Ticker'/'ticker' header, and first column doesn't seem to be it,
                # assume no header and first column is tickers (as per user's CSV example)
                df_for_tickers_no_header = pd.read_csv(root_data_dir, header=None, usecols=[0], low_memory=False)
                target_com_list = df_for_tickers_no_header[0].dropna().unique().tolist()

            if target_com_list:
                print(f"Derived company list from {root_data_dir}: {target_com_list[:10]}... (Total: {len(target_com_list)})")
            else:
                print(f"Could not derive company list from {root_data_dir}. Columns found: {pd.read_csv(root_data_dir, nrows=0).columns.tolist()}")
        except Exception as e:
            print(f"Error deriving company list from {root_data_dir}: {e}")

# If still no company list, fall back to the dummy example (or error out)
if not target_com_list:
    print(f"Warning: No company list loaded or derived. Using default dummy list: {nasdaq_com_list_example}")
    target_com_list = nasdaq_com_list_example
    # Optionally, create the dummy NASDAQ_example.csv if it's going to be used by MyDataset's logic later
    # For now, MyDataset will use the target_com_list directly.
    # However, the dummy data generation for stock CSVs might still create NASDAQ_example.csv if it's in com_list_csv_paths
    # This part is a bit tangled. The primary goal is that `target_com_list` is correctly populated.

if not target_com_list: # Final check
    print("CRITICAL ERROR: Target company list is empty. Please provide a company list CSV or ensure tickers can be derived from your main data CSV.")
    sys.exit(1)
# --- Create Dummy CSV Stock Data if root_data_dir (file) is not set up ---
# This creates a single CSV file with columns: Ticker,Date,Open,High,Low,Close,Volume
# TODO: Replace this with your actual data loading mechanism or ensure data is present.
if not os.path.exists(root_data_dir):
    print(f"Dummy stock data CSV file {root_data_dir} not found. Creating it...")
    header = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
    
    dummy_rows = []
    for stock_ticker in target_com_list: # Using the com_list from NASDAQ_example.csv or hardcoded
        # Add a few dummy data rows
        dummy_rows.append([stock_ticker, "2013-01-01", 100.0, 102.0, 99.0, 101.0, 10000.0])
        dummy_rows.append([stock_ticker, "2013-01-02", 101.0, 103.0, 100.0, 102.0, 12000.0])
        
        # Generate more data to cover date ranges up to end of 2017
        start_date_gen = datetime.strptime("2013-01-02", "%Y-%m-%d")
        for i in range(2, 365 * 5 + 1): # Approx 5 years of daily data
            day_dt = start_date_gen + timedelta(days=i)
            day_str = day_dt.strftime("%Y-%m-%d") # Standard YYYY-MM-DD format
            dummy_rows.append([
                stock_ticker, day_str,
                round(100.0 + i * 0.1, 2),
                round(102.0 + i * 0.1, 2),
                round(99.0 + i * 0.1, 2),
                round(101.0 + i * 0.1, 2),
                10000.0 + i * 10
            ])
            if day_str >= test_sedate[1]: # Stop if we've passed the test end date
                break

    if dummy_rows:
        try:
            with open(root_data_dir, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(dummy_rows)
            print(f"Created dummy stock data CSV file: {root_data_dir}")
        except Exception as e:
            print(f"Could not create dummy stock data CSV file: {e}")
    else:
        print("No stock data generated for dummy CSV file.")

# --- Dataset Instantiation ---
# --- Dataset Instantiation ---
# Using MyDataset as defined in the notebook
# Parameters: root, desti, market, comlist, start, end, window, dataset_type
window_size = 19 # As per notebook

print("Initializing training dataset...")
train_dataset = MyDataset(root_data_dir, graph_dest_dir, current_market, target_com_list,
                          train_sedate[0], train_sedate[1], window_size, dataset_types[0])
print("Initializing validation dataset...")
validation_dataset = MyDataset(root_data_dir, graph_dest_dir, current_market, target_com_list,
                               val_sedate[0], val_sedate[1], window_size, dataset_types[1])
print("Initializing test dataset...")
test_dataset = MyDataset(root_data_dir, graph_dest_dir, current_market, target_com_list,
                         test_sedate[0], test_sedate[1], window_size, dataset_types[2])

# --- DataLoader Instantiation ---
# Wrap datasets with DataLoader for batching, shuffling, and parallel loading.
print(f"Creating DataLoaders with BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, PIN_MEMORY={PIN_MEMORY}")
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,  # Shuffle training data
                          num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY,
                          drop_last=True) # drop_last can be useful if batch processing logic expects full batches

val_loader = DataLoader(validation_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False, # No need to shuffle validation data
                        num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, # No need to shuffle test data
                         num_workers=NUM_WORKERS,
                         pin_memory=PIN_MEMORY)


# --- Model Definition ---
num_companies = len(target_com_list)
if num_companies == 0:
    print("Error: Number of companies is 0. Cannot define model.")
    sys.exit(1)

# Model hyperparameters from the notebook (can be tuned)
# The actual feature length from X is window_size.
# The '+1' for label was handled in MyDataset; X_processed has window_size time steps.
model_feature_len = window_size # This is the actual length of the feature vector per node/relation from X
# d_layers, num_relation, m_gamma, diffusion_steps = 6, 5, 2.5e-4, 7 # Old line
d_layers, num_relation, regularization_gamma, diffusion_steps = 5, 5, 2.5e-4, 7 # Renamed m_gamma, Set d_layers to 5
retention_decay_zeta = 1.27 # Changed to paper's specified value

# Note: `gamma` in notebook was 2.5e-4, renamed to regularization_gamma.
# The MGDPR model's `time_dim` parameter (for D_gamma, ParallelRetention) should be model_feature_len.

# diffusion_config: input to first MultiReDiffusion is model_feature_len
# Original diffusion_config seemed to list desired output dimensions of fc_layers within MultiReDiffusion,
# plus an initial dimension.
# dc_original = [model_feature_len, 3 * model_feature_len, 4 * model_feature_len, 5 * model_feature_len, 5 * model_feature_len, 6 * model_feature_len, 5 * model_feature_len]
# The MGDPR model expects diffusion_config to be [in_fc_MD0, out_fc_MD0, in_fc_MD1, out_fc_MD1, ...]

# retention_config defines in_dim, inter_dim, out_dim for ParallelRetention layers.
# The `in_dim` for ParallelRetention is derived from the output of MultiReDiffusion and how it's reshaped.
# The MGDPR model's ParallelRetention expects input `x` (which is `u_intermediate` from diffusion)
# to be reshaped to `(self.time_dim, self.in_dim)`. `self.time_dim` is `model_feature_len`.
# `self.in_dim` is `retention_config[3*i]`.
# So, `(num_relation * num_companies * features_from_diffusion_output) / model_feature_len == retention_config[3*i]`.
# `features_from_diffusion_output` is `diffusion_config[1]` (e.g., `3 * model_feature_len`).
# So, `(num_relation * num_companies * 3 * model_feature_len) / model_feature_len == num_relation * 3 * num_companies`. This matches.
retention_config = [num_relation*3*num_companies, num_relation*5*num_companies, num_relation*4*num_companies,
                    num_relation*4*num_companies, num_relation*5*num_companies, num_relation*5*num_companies,
                    num_relation*5*num_companies, num_relation*5*num_companies, num_relation*5*num_companies,
                    num_relation*5*num_companies, num_relation*5*num_companies, num_relation*5*num_companies,
                    num_relation*6*num_companies, num_relation*5*num_companies, num_relation*5*num_companies,
                    num_relation*5*num_companies, num_relation*5*num_companies, num_relation*5*num_companies] # This config seems to be for the output dim of ParallelRetention, not its in_dim.
                                                                                                               # The in_dim for ParallelRetention's QKV layers is retention_config[3*i].
                                                                                                               # The MGDPR notebook's retention_config is [in0, inter0, out0, in1, inter1, out1, ...]
                                                                                                               # So, retention_config[0] is in_dim for layer 0.
                                                                                                               # This needs to match (num_relation * num_companies * diffusion_output_dim_0) / model_feature_len
                                                                                                               # Let's assume the notebook's config values are for the *internal* dimensions of ParallelRetention's QKV, not the reshaped input.
                                                                                                               # The `in_dim` for ParallelRetention's Linear layers (Q,K,V) is `retention_config[3*i]`.
                                                                                                               # This `in_dim` must match `(num_relation * num_nodes * features_from_diffusion) / self.time_dim` (model_feature_len).
                                                                                                               # The `features_from_diffusion` is `diffusion_config[2*l+1]`.
                                                                                                               # So, `retention_config[3*i]` should be `(num_relation * num_companies * diffusion_config[1]) / model_feature_len` for the first layer.
                                                                                                               # `(5 * num_companies * 3 * model_feature_len) / model_feature_len = 15 * num_companies`.
                                                                                                               # The notebook has `num_relation*3*num_companies` for `retention_config[0]`. This is `5*3*num_companies = 15*num_companies`. This matches.

# Dynamically construct ret_linear_1_config and ret_linear_2_config
# to ensure dimensions match between layers.

mfl = model_feature_len
nr = num_relation
nc = num_companies # num_nodes in MGDPR model context

# Target output dimensions for ret_linear_1 layers (inspired by notebook structure)
# These are the 'out_features' for each ret_linear_1 layer.
# Example: [95, 95, 95, 95, 95, 95] for d_layers=6
target_out_dims_rl1 = [mfl * nr * 1] * d_layers

# Target output dimensions for ret_linear_2 layers (inspired by notebook structure)
# The last layer's output must match the MLP input dimension.
mlp_input_dim = mfl * nr * 6 # Example: 19 * 5 * 6 = 570
if d_layers == 1:
  target_out_dims_rl2 = [mlp_input_dim]
elif d_layers > 1:
  target_out_dims_rl2 = [mfl * nr * 5] + [mfl * nr * 6] * (d_layers - 2) + [mlp_input_dim]
else: # d_layers == 0, should not happen
  target_out_dims_rl2 = []


# Calculate eta_dims (feature dimension of eta from ParallelRetention for each company/node)
# eta = output_x.view(num_node_original_from_x, -1)
# output_x is (self.time_dim, self.out_dim) from ParallelRetention
# self.time_dim is model_feature_len (mfl)
# self.out_dim is retention_config[3*l+2]
# So eta per node has (mfl * retention_config[3*l+2]) / num_companies features.
eta_feature_dims_per_node = []
for i in range(d_layers):
  if (3 * i + 2) < len(retention_config):
      pr_out_dim = retention_config[3 * i + 2] # This is out_dim for ParallelRetention's QKV block
      # eta_dim_per_node = (mfl * pr_out_dim) / nc # This was the previous logic
      # The output of ParallelRetention.forward is output_x.view(num_node_original_from_x, -1)
      # output_x is (self.time_dim, self.out_dim) = (mfl, pr_out_dim)
      # So, view(nc, (mfl * pr_out_dim) / nc). The feature dim is (mfl * pr_out_dim) / nc
      # This must be an integer.
      calculated_eta_feat_dim = (mfl * pr_out_dim) / nc
      if calculated_eta_feat_dim != int(calculated_eta_feat_dim):
          print(f"ERROR: Calculated eta feature dimension for layer {i} is not an integer: {calculated_eta_feat_dim}. This indicates a config problem.")
          # This is a critical error. The dimensions must result in an integer number of features.
          # Defaulting to 0 or pr_out_dim will likely lead to further runtime errors.
          # This suggests retention_config or mfl/nc needs adjustment so (mfl * pr_out_dim) is divisible by nc.
          # For now, to proceed and see if this is the only issue, we'll cast to int, but this needs review.
          # A robust solution would be to ensure divisibility or re-evaluate the config.
          eta_feature_dims_per_node.append(int(calculated_eta_feat_dim))
      else:
          eta_feature_dims_per_node.append(int(calculated_eta_feat_dim))
  else:
      print(f"Error: retention_config is too short for layer {i} to determine eta_feature_dims.")
      eta_feature_dims_per_node.append(0) # Fallback, will likely cause issues

ret_linear_1_config_list = []
ret_linear_2_config_list = []

# Input to the first ret_linear_1 layer
current_rl1_input_dim = nr * mfl # From x_transformed_for_skip

for l in range(d_layers):
  # ret_linear_1[l]
  rl1_out_dim = target_out_dims_rl1[l]
  ret_linear_1_config_list.extend([current_rl1_input_dim, rl1_out_dim])

  # ret_linear_2[l]
  # Input is concat(eta[l], output_of_ret_linear_1[l])
  rl2_input_dim = eta_feature_dims_per_node[l] + rl1_out_dim
  rl2_out_dim = target_out_dims_rl2[l]
  ret_linear_2_config_list.extend([rl2_input_dim, rl2_out_dim])

  # Output of ret_linear_2[l] becomes input for next ret_linear_1[l+1]
  current_rl1_input_dim = rl2_out_dim

ret_linear_1_config = ret_linear_1_config_list
ret_linear_2_config = ret_linear_2_config_list

if not ret_linear_2_config: # Handle d_layers = 0 case, though unlikely
  final_mlp_input_dim = nr * mfl # If no MGDPR layers, MLP takes initial transformed x
else:
  final_mlp_input_dim = ret_linear_2_config[-1] # Output of the last ret_linear_2 layer

mlp_config = [final_mlp_input_dim, 128, 1] # Changed to 1 for continuous score output for ranking

# MGDPR's `time_dim` parameter is used for D_gamma and ParallelRetention's self.time_dim.
# This should be `model_feature_len` (i.e., `window_size`).

# Construct the actual diffusion_config needed by MGDPR model structure
# The MGDPR model's MultiReDiffusion layers expect config[2*l] as in_features and config[2*l+1] as out_features for their internal fc_layers.

# Desired output dimensions for the fc_layers within each MultiReDiffusion block `l`.
# Taken from the original interpretation of diffusion_config: dc_original[l+1]
dc_original_outputs = [
    3 * model_feature_len,  # For MD block 0
    4 * model_feature_len,  # For MD block 1
    5 * model_feature_len,  # For MD block 2
    5 * model_feature_len,  # For MD block 3
    6 * model_feature_len,  # For MD block 4
    5 * model_feature_len   # For MD block 5
]
if len(dc_original_outputs) < d_layers:
    raise ValueError(f"dc_original_outputs length {len(dc_original_outputs)} is less than d_layers {d_layers}")

actual_diffusion_config = []
# current_input_features_for_md_fc was the input to MultiReDiffusion's fc_layers.
# For the first MD block (l_idx=0), its fc_layers process h[rel] which has model_feature_len features.
current_fc_input_dim = model_feature_len

for l_idx in range(d_layers):
    fc_output_dim = dc_original_outputs[l_idx] # Desired output dimension for fc_layers in MD block l_idx
    
    actual_diffusion_config.append(current_fc_input_dim) # In-features for fc_layers in MD block l_idx
    actual_diffusion_config.append(fc_output_dim)      # Out-features for fc_layers in MD block l_idx
    
    # The output of the fc_layers in MD block l_idx becomes the feature dimension
    # for the input h[rel] to the fc_layers in the next MD block (l_idx+1).
    current_fc_input_dim = fc_output_dim


model = MGDPR(actual_diffusion_config, retention_config, ret_linear_1_config, ret_linear_2_config, mlp_config,
              d_layers, num_companies, model_feature_len, num_relation, retention_decay_zeta, diffusion_steps, regularization_gamma_param=regularization_gamma) # Pass retention_decay_zeta and optional regularization_gamma
model = model.to(device)

# --- Optimizer and Objective Function ---
optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4) # Reverted to paper's learning rate
# criterion = F.cross_entropy # Using criterion directly # Replaced by ListNetLoss

# --- Custom ListFold Exponential Loss Function ---
class ListFoldExponentialLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-9):
        super(ListFoldExponentialLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred_scores_batch, y_true_scores_batch):
        """
        Calculates ListFold loss with exponential transformation.
        Args:
            y_pred_scores_batch (torch.Tensor): Predicted scores from the model.
                                          Shape: (Batch_Size, Num_Nodes)
            y_true_scores_batch (torch.Tensor): True target scores.
                                          Shape: (Batch_Size, Num_Nodes)
                                          May contain NaNs for items that are not valid targets.
        Returns:
            torch.Tensor: The ListFold exponential loss value.
        """
        batch_total_loss = 0.0
        actual_samples_in_batch = 0

        for k_batch in range(y_pred_scores_batch.size(0)):
            y_pred_sample = y_pred_scores_batch[k_batch]
            y_true_sample = y_true_scores_batch[k_batch]

            # Filter out NaNs based on y_true_sample
            valid_mask = ~torch.isnan(y_true_sample)
            s_true_valid = y_true_sample[valid_mask]
            s_pred_valid = y_pred_sample[valid_mask]

            N_valid = s_true_valid.size(0)

            if N_valid < 2:  # Need at least one pair
                continue

            num_pairs = N_valid // 2
            if num_pairs == 0:
                continue
            
            actual_samples_in_batch += 1

            # Get true permutation indices (for s_true_valid, descending)
            perm_indices = torch.argsort(s_true_valid, descending=True)
            
            # Predicted scores sorted according to true relevance
            s_pred_sorted_by_true = s_pred_valid[perm_indices]

            sample_loss_accumulator = 0.0
            
            for i_pair_idx in range(num_pairs):
                score_pred_true_top = s_pred_sorted_by_true[i_pair_idx]
                score_pred_true_bottom = s_pred_sorted_by_true[N_valid - 1 - i_pair_idx]
                
                numerator_term = score_pred_true_top - score_pred_true_bottom

                current_remaining_scores_for_denom = s_pred_sorted_by_true[i_pair_idx : N_valid - i_pair_idx]
                len_remaining = current_remaining_scores_for_denom.size(0)

                if len_remaining < 2:
                    break

                exp_scores = torch.exp(current_remaining_scores_for_denom)
                exp_neg_scores = torch.exp(-current_remaining_scores_for_denom)
                
                sum_exp_scores = torch.sum(exp_scores)
                sum_exp_neg_scores = torch.sum(exp_neg_scores)
                
                denominator_sum_exp_terms = sum_exp_scores * sum_exp_neg_scores - len_remaining
                
                denominator_term = torch.log(denominator_sum_exp_terms + self.epsilon)

                sample_loss_accumulator -= (numerator_term - denominator_term)
            
            batch_total_loss += sample_loss_accumulator

        if actual_samples_in_batch > 0:
            return batch_total_loss / actual_samples_in_batch
        else:
            return torch.tensor(0.0, device=y_pred_scores_batch.device, requires_grad=True)

criterion = ListFoldExponentialLoss()
# --- Training, Validation, and Testing ---
# --- Profiler Configuration (from parsed arguments) ---
PROFILE_ENABLED = args.profile
PROFILE_EPOCH_TARGET = args.profile_epoch
PROFILE_ACTIVE_STEPS = args.profile_steps # This is the 'active' duration in the schedule
PROFILE_WAIT_STEPS = args.profile_wait
PROFILE_WARMUP_STEPS = args.profile_warmup
PROFILER_LOG_DIR = args.profile_log_dir
PROFILE_REPEAT_CYCLES = 1 # Number of times to repeat the wait, warmup, active cycle; typically 1 for a short profiling run.

# Create profiler log directory if it doesn't exist and profiling is enabled
if PROFILE_ENABLED and not os.path.exists(PROFILER_LOG_DIR):
    os.makedirs(PROFILER_LOG_DIR, exist_ok=True)
    print(f"Profiler log directory will be used/created: {PROFILER_LOG_DIR}")

# Helper function for a single training step to avoid code duplication
# profiler_context argument removed as prof.step() is handled by schedule, and record_function is global
def train_batch(batch_sample, model, criterion, optimizer, device, scaler, use_amp, is_profiling=False):
    """
    Processes a single batch of training data.
    """
    X = batch_sample['X'].to(device)
    X = batch_sample['X'].to(device)
    A = batch_sample['A'].to(device)
    # C_labels are now the true z-scored vol-adjusted returns (continuous)
    # Shape: (Batch_Size, Num_Nodes)
    C_target_scores = batch_sample['Y'].float().to(device)

    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass
    with torch.amp.autocast('cuda', enabled=use_amp):
        if is_profiling:
            with record_function("model_forward"):
                # Model output is now (Batch_Size, Num_Nodes, 1)
                out_predicted_scores_raw = model(X, A)
        else:
            out_predicted_scores_raw = model(X, A)
        
        # Squeeze the last dimension to get (Batch_Size, Num_Nodes)
        out_predicted_scores = out_predicted_scores_raw.squeeze(-1)
        
        # Calculate loss using ListNetLoss
        # Create a mask for valid target scores (not NaN)
        # This mask helps in calculating metrics correctly later, and can be used by loss if needed.
        valid_target_mask = ~torch.isnan(C_target_scores)

        # For ListNetLoss, we pass the raw scores. NaNs in C_target_scores are handled internally by the loss.
        # However, if all target scores in a batch item are NaN, p_true might become uniform,
        # and log_p_pred for those might be compared against it.
        # We should only compute loss for batch items that have at least one valid target.
        # For simplicity now, assume ListNetLoss handles internal NaNs in y_true_scores gracefully.
        # A more robust approach might be to filter batches or mask loss contributions.
        
        # Filter out samples where all target scores are NaN to prevent issues with softmax or loss calculation
        # if a sample has no valid information to learn from.
        # A sample is valid if it has at least one non-NaN target score.
        sample_has_valid_target = torch.any(valid_target_mask, dim=1)

        if not torch.any(sample_has_valid_target):
            # If no samples in the batch have any valid targets, skip loss calculation for this batch
            # print(f"DEBUG Batch {i+1}: Skipping loss calculation as no samples have valid targets.")
            loss = torch.tensor(0.0, device=device, requires_grad=True) # Or handle as appropriate
        else:
            # Filter inputs to the loss function to only include samples with at least one valid target
            valid_pred_scores = out_predicted_scores[sample_has_valid_target]
            valid_true_scores = C_target_scores[sample_has_valid_target]
            
            listfold_loss = criterion(valid_pred_scores, valid_true_scores)
            
            # Add theta regularization loss
            theta_reg_loss = model.get_theta_regularization_loss()
            
            # The regularization_gamma is already defined in the script (e.g., 2.5e-4)
            # It's accessible via model.regularization_gamma if it was stored there,
            # or directly if it's in the global scope of train_val_test.py.
            # Assuming `regularization_gamma` is accessible here.
            # If not, it should be passed to train_batch or accessed via model.
            # The MGDPR model stores it as self.regularization_gamma.
            
            if model.regularization_gamma is not None:
                loss = listfold_loss + model.regularization_gamma * theta_reg_loss
            else:
                loss = listfold_loss # No regularization if gamma is None
        
    # <<< End of torch.amp.autocast block >>>

    # Backward pass and optimization
    if use_amp_flag: # Corrected variable name from use_amp to use_amp_flag
        if is_profiling:
            with record_function("model_backward_amp"):
                scaler.scale(loss).backward()
                # Gradient clipping after scaling, before optimizer step
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else: # Not profiling, but using AMP
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    else: # Not using AMP
        if is_profiling:
            with record_function("model_backward_no_amp"):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else: # Not profiling and not using AMP
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # profiler.step() is now handled by the schedule, so no manual call here.

    return loss, out_predicted_scores, C_target_scores, sample_has_valid_target

epochs = 10 # Reduced for quick testing, notebook uses 10000
model.reset_parameters()

# --- AMP Scaler ---
use_amp_flag = args.use_amp and torch.cuda.is_available() and device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp_flag)
print(f"Automatic Mixed Precision (AMP) {'enabled' if use_amp_flag else 'disabled'}.")

print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_loss_sum = 0
    epoch_correct = 0
    epoch_total_samples = 0

    # Check if train_loader is empty.
    if len(train_loader) == 0: # Changed from train_dataset to train_loader
        print(f"Epoch {epoch+1}/{epochs}: Training loader is empty (no batches). Skipping training for this epoch.")
        # Check if the underlying dataset was also empty
        if len(train_dataset) == 0:
            print(f"Epoch {epoch+1}/{epochs}: Underlying training dataset is also empty.")
        break
    
    # Determine if profiling should be active for the current epoch
    is_profiling_active_epoch = PROFILE_ENABLED and epoch == PROFILE_EPOCH_TARGET

    if is_profiling_active_epoch:
        print(f"--- Profiling enabled for Epoch {epoch+1} ---")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() and device.type == 'cuda': # Check if CUDA is actually used
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        profiler_schedule = torch.profiler.schedule(
            wait=PROFILE_WAIT_STEPS,
            warmup=PROFILE_WARMUP_STEPS,
            active=PROFILE_ACTIVE_STEPS,
            repeat=PROFILE_REPEAT_CYCLES
        )
        print(f"Profiler schedule: wait={PROFILE_WAIT_STEPS}, warmup={PROFILE_WARMUP_STEPS}, active={PROFILE_ACTIVE_STEPS}, repeat={PROFILE_REPEAT_CYCLES}")
        
        # Define the trace handler
        trace_handler = torch.profiler.tensorboard_trace_handler(
            PROFILER_LOG_DIR,
            worker_name=f"worker_epoch{epoch}" # Optional: add worker name for distributed training, or just keep it simple
        ) if PROFILE_ENABLED else None # Only create handler if profiling

        # Profiler context manager
        with torch.profiler.profile(
            activities=activities,
            schedule=profiler_schedule,
            profile_memory=True,    # Enable memory profiling
            record_shapes=True,     # Enable shape recording for better analysis
            with_stack=True,        # Enable stack tracing for more detailed source attribution
            on_trace_ready=trace_handler # Use the configured trace handler
        ) as prof:
            # The loop needs to run for enough batches for the schedule to complete one cycle.
            num_batches_for_profiling_cycle = PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS + (PROFILE_ACTIVE_STEPS * PROFILE_REPEAT_CYCLES)
            print(f"Running profiling for up to {num_batches_for_profiling_cycle} batches to cover schedule...")
            
            for i, sample in enumerate(train_loader):
                if i >= num_batches_for_profiling_cycle:
                    print(f"Reached {i} batches, stopping profiling loop for this epoch.")
                    break # Stop after enough batches for the schedule
                
                # train_batch now returns: loss, out_predicted_scores, C_target_scores, sample_has_valid_target
                current_loss, _, _, _ = train_batch(
                    sample, model, criterion, optimizer, device, scaler, use_amp_flag, is_profiling=True
                )
                
                epoch_loss_sum += current_loss.item() # current_loss is already a scalar
                
                # Accuracy calculation is no longer relevant for ranking.
                # We will implement ranking metrics (e.g., NDCG, Spearman correlation) later.
                # For now, just track the loss.
                
                batch_num_display = i + 1
                # prof.step() is called by the profiler due to the schedule
                # Determine if currently in active recording phase for logging
                current_profiling_batch_num = -1
                if i >= PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS:
                    current_profiling_batch_num = i - (PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS) + 1
                
                log_message_profiling_phase = ""
                if current_profiling_batch_num > 0 and current_profiling_batch_num <= PROFILE_ACTIVE_STEPS:
                    log_message_profiling_phase = f"(Profiling active batch {current_profiling_batch_num} of {PROFILE_ACTIVE_STEPS})"
                elif i < PROFILE_WAIT_STEPS:
                    log_message_profiling_phase = f"(Profiler waiting, batch {i+1} of {PROFILE_WAIT_STEPS})"
                elif i < PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS:
                    log_message_profiling_phase = f"(Profiler warming up, batch {i+1-PROFILE_WAIT_STEPS} of {PROFILE_WARMUP_STEPS})"

                # The variable 'loss' was changed to 'current_loss' in this scope
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_num_display}/{len(train_loader)} {log_message_profiling_phase}: Loss={current_loss.item():.4f}")
            
            # --- Detailed Profiler Output (Inside 'with prof:' block, after the loop) ---
            print("\n--- Checking Raw Profiler Events (inside profiler context, after loop) ---")
            raw_events = prof.events()
            print(f"Number of raw events collected: {len(raw_events)}")
            if len(raw_events) > 0:
                print("First 5 raw events (if any):")
                for event_idx, event in enumerate(raw_events):
                    if event_idx >= 5: break
                    event_name = getattr(event, 'name', 'N/A')
                    cpu_time = getattr(event, 'cpu_time_total', 'N/A')
                    cuda_time = getattr(event, 'cuda_time_total', 'N/A')
                    thread_id = getattr(event, 'thread', 'N/A')
                    print(f"  Event {event_idx}: Name: {event_name}, CPU time: {cpu_time} us, CUDA time: {cuda_time} us, Thread: {thread_id}")
            else:
                print("No raw events were collected by the profiler or available.")

            print("\n--- PyTorch Profiler Key Averages (inside profiler context, after loop) ---")
            key_avg_obj = prof.key_averages()
            print(key_avg_obj)

            if not key_avg_obj:
                print("Profiler key_averages() returned an empty or None object.")
            else:
                print("\n--- PyTorch Profiler CPU Summary (Self Time) ---")
            try:
                print(key_avg_obj.table(sort_by="self_cpu_time_total", row_limit=15, header="Self CPU Time Total"))
            except Exception as e:
                print(f"Error generating CPU self time table: {e}")
            
            print("\n--- PyTorch Profiler CPU Summary (Total Time) ---")
            try:
                print(key_avg_obj.table(sort_by="cpu_time_total", row_limit=15, header="Total CPU Time"))
            except Exception as e:
                print(f"Error generating CPU total time table: {e}")
            
            if torch.profiler.ProfilerActivity.CUDA in activities:
                print("\n--- PyTorch Profiler CUDA Summary (Self Time) ---")
                try:
                    print(key_avg_obj.table(sort_by="self_cuda_time_total", row_limit=15, header="Self CUDA Time Total"))
                except Exception as e:
                    print(f"Error generating CUDA self time table: {e}")

                print("\n--- PyTorch Profiler CUDA Summary (Total Time) ---")
                try:
                    print(key_avg_obj.table(sort_by="cuda_time_total", row_limit=15, header="Total CUDA Time"))
                except Exception as e:
                    print(f"Error generating CUDA total time table: {e}")
            print("--- End of Profiler Summary (inside profiler context) ---\n")

        # Continue with the rest of the batches for this epoch (if any) without profiling
        start_batch_idx_after_profiling = num_batches_for_profiling_cycle
        
        if start_batch_idx_after_profiling < len(train_loader): 
            print(f"--- Continuing Epoch {epoch+1} after profiling ({start_batch_idx_after_profiling}/{len(train_loader)}) ---") 
            for i, sample in enumerate(train_loader): 
                if i < start_batch_idx_after_profiling:
                    continue # Skip already processed (profiled) batches

                # train_batch now returns: loss, out_predicted_scores, C_target_scores, sample_has_valid_target
                # Unpack correctly, using current_loss for the loss value
                current_loss, _out_predicted_scores, _C_target_scores, _sample_has_valid_target = train_batch(
                    sample, model, criterion, optimizer, device, scaler, use_amp_flag, is_profiling=False
                )
                
                epoch_loss_sum += current_loss.item()
                # Accuracy calculation removed
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader): # Print for last batch too
                     print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}: Loss={current_loss.item():.4f}")
    
    else: # Regular training for epochs not targeted for profiling (or if PROFILE_ENABLED is False)
        for i, sample in enumerate(train_loader):
            # train_batch now returns: loss, out_predicted_scores, C_target_scores, sample_has_valid_target
            # Unpack correctly, using current_loss for the loss value
            current_loss, _out_predicted_scores, _C_target_scores, _sample_has_valid_target = train_batch(
                sample, model, criterion, optimizer, device, scaler, use_amp_flag, is_profiling=False
            )
            
            epoch_loss_sum += current_loss.item()
            # Accuracy calculation removed
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader): # Print for last batch too
                 print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}: Loss={current_loss.item():.4f}")

    # Epoch summary (common for all cases)
    # if epoch_total_samples > 0: # epoch_total_samples is no longer tracked this way
    num_batches_in_loader = len(train_loader)
    if num_batches_in_loader > 0 : # Check if any batches were processed
        avg_epoch_loss = epoch_loss_sum / num_batches_in_loader
        # avg_epoch_acc = epoch_correct / epoch_total_samples # Accuracy removed
        print(f"Epoch {epoch+1}/{epochs} Summary: Avg Loss={avg_epoch_loss:.4f}") # Accuracy removed
    elif len(train_loader) > 0: # This case might be redundant if num_batches_in_loader covers it
        print(f"Epoch {epoch+1}/{epochs} Summary: No samples processed in this epoch (loader size: {len(train_loader)}, dataset size: {len(train_dataset)}).")
    # If len(train_loader) == 0, the 'break' earlier handles it.

    # Validation step (optional, can be done less frequently)
    # TODO: Update validation to use ranking metrics instead of classification metrics.
    
    # Define ndcg_k_values here so it's accessible in both validation and test loops
    ndcg_k_values = [2, 10] # Define k values for NDCG

    if (epoch + 1) % 5 == 0 and len(val_loader) > 0:
        model.eval()
        val_loss_sum = 0
        # val_correct = 0 # Classification metric
        # val_total_samples = 0 # Classification metric
        # val_f1_scores = [] # Classification metric
        # val_mcc_scores = [] # Classification metric
        
        # Placeholder for ranking metrics
        all_val_pred_scores_list = []
        all_val_true_scores_list = []

        with torch.no_grad():
            for i_val, val_sample in enumerate(val_loader):
                X_val = val_sample['X'].to(device)
                A_val = val_sample['A'].to(device)
                C_val_target_scores = val_sample['Y'].float().to(device) # True scores
                
                with torch.amp.autocast('cuda', enabled=use_amp_flag):
                    out_val_predicted_scores_raw = model(X_val, A_val) # (Batch, Nodes, 1)
                
                out_val_predicted_scores = out_val_predicted_scores_raw.squeeze(-1) # (Batch, Nodes)

                # Filter for loss calculation (samples with at least one valid target)
                val_sample_has_valid_target = torch.any(~torch.isnan(C_val_target_scores), dim=1)
                if torch.any(val_sample_has_valid_target):
                    valid_val_pred_scores = out_val_predicted_scores[val_sample_has_valid_target]
                    valid_val_true_scores = C_val_target_scores[val_sample_has_valid_target]
                    val_loss = criterion(valid_val_pred_scores, valid_val_true_scores)
                    val_loss_sum += val_loss.item()
                
                # Store all scores for offline metric calculation (e.g., NDCG, Spearman)
                all_val_pred_scores_list.append(out_val_predicted_scores.cpu())
                all_val_true_scores_list.append(C_val_target_scores.cpu())

        num_val_batches = len(val_loader)
        avg_val_loss = val_loss_sum / num_val_batches if num_val_batches > 0 else 0
        
        # Calculate and print ranking metrics
        ndcg_k_values = [2, 10] # Define k values for NDCG
        epoch_ndcg_scores = {k: [] for k in ndcg_k_values}
        epoch_spearman_coeffs = []

        if all_val_pred_scores_list and all_val_true_scores_list:
            # Concatenate all batch results
            # Each element in the list is a tensor of shape (batch_size_for_that_batch, num_nodes)
            # We want to process each sample (row) individually.
            
            # First, flatten the lists of tensors into lists of individual sample tensors
            flat_val_pred_scores = [item for batch_tensor in all_val_pred_scores_list for item in batch_tensor]
            flat_val_true_scores = [item for batch_tensor in all_val_true_scores_list for item in batch_tensor]

            for pred_scores_sample, true_scores_sample in zip(flat_val_pred_scores, flat_val_true_scores):
                # pred_scores_sample and true_scores_sample are 1D tensors for a single graph/day
                
                # Filter out NaNs: only consider pairs where true_score is not NaN
                valid_mask_sample = ~torch.isnan(true_scores_sample)
                
                if valid_mask_sample.sum() < 1: # Need at least 1 valid item for NDCG, 2 for Spearman
                    continue

                y_true_sample_np = true_scores_sample[valid_mask_sample].numpy()
                y_pred_sample_np = pred_scores_sample[valid_mask_sample].numpy()

                if len(y_true_sample_np) < 1: # Re-check after filtering, should be redundant if first check is <1
                    continue

                # NDCG
                # Scale y_true_sample_np to be non-negative for NDCG
                # Scale y_true_sample_np to be strictly positive for NDCG
                y_true_scaled_for_ndcg = np.array([]) # Initialize as empty

                if len(y_true_sample_np) > 0:
                    unique_vals = np.unique(y_true_sample_np)
                    if len(unique_vals) > 1:
                        # Scale to [epsilon, 1.0] to ensure positivity
                        scaler_ndcg = MinMaxScaler(feature_range=(1e-9, 1.0))
                        y_true_scaled_for_ndcg = scaler_ndcg.fit_transform(y_true_sample_np.reshape(-1, 1)).flatten()
                    elif len(unique_vals) == 1: # All values are the same
                        y_true_scaled_for_ndcg = np.full_like(y_true_sample_np, 0.5) # Assign a constant positive relevance
                # If y_true_sample_np was empty, y_true_scaled_for_ndcg remains empty.

                # ndcg_score expects 2D arrays: (n_samples, n_labels). Here, n_samples=1 for each graph.
                # y_true_scaled_for_ndcg contains non-negative relevance scores.
                # y_pred_sample_np contains predicted scores.
                for k_val in ndcg_k_values:
                    if len(y_true_scaled_for_ndcg) >= 1: # Can compute NDCG if there's at least one item
                        # Ensure k is not greater than the number of items
                        current_k = min(k_val, len(y_true_scaled_for_ndcg))
                        if current_k > 0:
                            ndcg_val = ndcg_score([y_true_scaled_for_ndcg], [y_pred_sample_np], k=current_k)
                            epoch_ndcg_scores[k_val].append(ndcg_val)
                
                # Spearman Correlation
                if len(y_true_sample_np) >= 2: # Spearman needs at least 2 data points
                    spearman_val, _ = spearmanr(y_true_sample_np, y_pred_sample_np)
                    if not np.isnan(spearman_val): # spearmanr can return NaN if input is constant
                        epoch_spearman_coeffs.append(spearman_val)

        avg_ndcg_scores_str = ", ".join([f"NDCG@{k}={np.mean(scores):.4f}" if scores else f"NDCG@{k}=N/A" for k, scores in epoch_ndcg_scores.items()])
        avg_spearman_coeff_str = f"Spearman={np.mean(epoch_spearman_coeffs):.4f}" if epoch_spearman_coeffs else "Spearman=N/A"
        
        print(f"Epoch {epoch+1} Validation: Avg Loss={avg_val_loss:.4f}, {avg_ndcg_scores_str}, {avg_spearman_coeff_str}")


# --- Final Testing ---
print("\nStarting final testing...")
model.eval()
test_loss_sum = 0
# test_correct = 0 # Classification metric
# test_total_samples = 0 # Classification metric
# test_f1_scores = [] # Classification metric
# test_mcc_scores = [] # Classification metric

# Placeholder for ranking metrics
all_test_pred_scores_list = []
all_test_true_scores_list = []


if len(test_loader) == 0:
    print("Test loader is empty. Skipping testing.")
    if len(test_dataset) == 0:
        print("Underlying test dataset is also empty.")
else:
    with torch.no_grad():
        for i_test, test_sample in enumerate(test_loader):
            X_test = test_sample['X'].to(device)
            A_test = test_sample['A'].to(device)
            C_test_target_scores = test_sample['Y'].float().to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp_flag):
                out_test_predicted_scores_raw = model(X_test, A_test) # (Batch, Nodes, 1)
            
            out_test_predicted_scores = out_test_predicted_scores_raw.squeeze(-1) # (Batch, Nodes)

            # Filter for loss calculation
            test_sample_has_valid_target = torch.any(~torch.isnan(C_test_target_scores), dim=1)
            if torch.any(test_sample_has_valid_target):
                valid_test_pred_scores = out_test_predicted_scores[test_sample_has_valid_target]
                valid_test_true_scores = C_test_target_scores[test_sample_has_valid_target]
                test_loss = criterion(valid_test_pred_scores, valid_test_true_scores)
                test_loss_sum += test_loss.item()

            all_test_pred_scores_list.append(out_test_predicted_scores.cpu())
            all_test_true_scores_list.append(C_test_target_scores.cpu())

    num_test_batches = len(test_loader)
    avg_test_loss = test_loss_sum / num_test_batches if num_test_batches > 0 else 0
    
    # Calculate and print ranking metrics for Test set
    test_ndcg_scores = {k: [] for k in ndcg_k_values}
    test_spearman_coeffs = []

    if all_test_pred_scores_list and all_test_true_scores_list:
        flat_test_pred_scores = [item for batch_tensor in all_test_pred_scores_list for item in batch_tensor]
        flat_test_true_scores = [item for batch_tensor in all_test_true_scores_list for item in batch_tensor]

        for pred_scores_sample, true_scores_sample in zip(flat_test_pred_scores, flat_test_true_scores):
            valid_mask_sample = ~torch.isnan(true_scores_sample)
            if valid_mask_sample.sum() < 1:
                continue
            y_true_sample_np = true_scores_sample[valid_mask_sample].numpy()
            y_pred_sample_np = pred_scores_sample[valid_mask_sample].numpy()

            if len(y_true_sample_np) < 1:
                continue

            # Scale y_true_sample_np to be non-negative for NDCG for the test set
            # Scale y_true_sample_np to be strictly positive for NDCG for the test set
            y_true_scaled_for_ndcg_test = np.array([]) # Initialize as empty

            if len(y_true_sample_np) > 0:
                unique_vals_test = np.unique(y_true_sample_np)
                if len(unique_vals_test) > 1:
                    scaler_ndcg_test = MinMaxScaler(feature_range=(1e-9, 1.0))
                    y_true_scaled_for_ndcg_test = scaler_ndcg_test.fit_transform(y_true_sample_np.reshape(-1, 1)).flatten()
                elif len(unique_vals_test) == 1:
                    y_true_scaled_for_ndcg_test = np.full_like(y_true_sample_np, 0.5)
            # If y_true_sample_np was empty, y_true_scaled_for_ndcg_test remains empty.
            
            for k_val in ndcg_k_values:
                if len(y_true_scaled_for_ndcg_test) >= 1:
                    current_k = min(k_val, len(y_true_scaled_for_ndcg_test))
                    if current_k > 0:
                        ndcg_val = ndcg_score([y_true_scaled_for_ndcg_test], [y_pred_sample_np], k=current_k)
                        test_ndcg_scores[k_val].append(ndcg_val)
            
            if len(y_true_sample_np) >= 2:
                spearman_val, _ = spearmanr(y_true_sample_np, y_pred_sample_np)
                if not np.isnan(spearman_val):
                    test_spearman_coeffs.append(spearman_val)

    avg_test_ndcg_str = ", ".join([f"NDCG@{k}={np.mean(scores):.4f}" if scores else f"NDCG@{k}=N/A" for k, scores in test_ndcg_scores.items()])
    avg_test_spearman_str = f"Spearman={np.mean(test_spearman_coeffs):.4f}" if test_spearman_coeffs else "Spearman=N/A"

    print(f"Test Results: Avg Loss={avg_test_loss:.4f}, {avg_test_ndcg_str}, {avg_test_spearman_str}")

# --- Save Model ---
# The notebook had a prompt, using a fixed name here for simplicity.
# Consider making the save path configurable.
save_model_path = "mgdpr_trained_model.pth"
try:
    # user_input = input(f"Save model to {save_model_path}? (yes/no): ")
    # if user_input.lower() == 'yes':
    torch.save(model.state_dict(), save_model_path)
    print(f"Model state_dict saved to {save_model_path}")
except Exception as e:
    print(f"Error saving model: {e}")

print("Script finished.")
