import sys
import os
# Add project root to Python path to allow sibling module imports (dataset, model)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import csv
from datetime import datetime, timedelta # Added import
import torch.nn.functional as F
import pandas as pd # Added for MultiIndex DataFrame
# import torch.distributions # Not explicitly used in the notebook's training script part
from sklearn.metrics import matthews_corrcoef, f1_score

from dataset.graph_dataset_gen import MyDataset # Corrected import
from model.Multi_GDNN import MGDPR

# Configure the device for running the model on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# --- Model Definition ---
num_companies = len(target_com_list)
if num_companies == 0:
    print("Error: Number of companies is 0. Cannot define model.")
    sys.exit(1)

# Model hyperparameters from the notebook (can be tuned)
# The actual feature length from X is window_size.
# The '+1' for label was handled in MyDataset; X_processed has window_size time steps.
model_feature_len = window_size # This is the actual length of the feature vector per node/relation from X
d_layers, num_relation, m_gamma, diffusion_steps = 6, 5, 2.5e-4, 7

# Note: `gamma` in notebook was 2.5e-4, renamed to m_gamma.
# The MGDPR model's `time_dim` parameter (for D_gamma, ParallelRetention) should be model_feature_len.

# diffusion_config: input to first MultiReDiffusion is model_feature_len
diffusion_config = [model_feature_len, 3 * model_feature_len, 4 * model_feature_len, 5 * model_feature_len, 5 * model_feature_len, 6 * model_feature_len, 5 * model_feature_len]

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

mlp_config = [final_mlp_input_dim, 128, 2] # 2 for binary classification

# MGDPR's `time_dim` parameter is used for D_gamma and ParallelRetention's self.time_dim.
# This should be `model_feature_len` (i.e., `window_size`).
model = MGDPR(diffusion_config, retention_config, ret_linear_1_config, ret_linear_2_config, mlp_config,
              d_layers, num_companies, model_feature_len, num_relation, m_gamma, diffusion_steps)
model = model.to(device)

# --- Optimizer and Objective Function ---
def theta_regularizer(theta_param): # Renamed to avoid conflict
    row_sums = torch.sum(theta_param.to(device), dim=-1)
    ones = torch.ones_like(row_sums)
    return torch.sum(torch.abs(row_sums - ones))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = F.cross_entropy # Using criterion directly

# --- Training, Validation, and Testing ---
epochs = 10 # Reduced for quick testing, notebook uses 10000
model.reset_parameters()

print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total_samples = 0

    # Check if train_dataset is empty
    if len(train_dataset) == 0:
        print(f"Epoch {epoch+1}/{epochs}: Training dataset is empty. Skipping training for this epoch.")
        # Optionally, break or sys.exit if training cannot proceed
        break 
        
    for i, sample in enumerate(train_dataset):
        X = sample['X'].to(device)
        A = sample['A'].to(device)
        C = sample['Y'].long().to(device)

        optimizer.zero_grad()
        out_logits = model(X, A)
        loss = criterion(out_logits, C)
        # loss += theta_regularizer(model.theta) # Optional regularization

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        preds = out_logits.argmax(dim=1)
        epoch_correct += (preds == C).sum().item()
        epoch_total_samples += C.size(0)
        
        if (i + 1) % 10 == 0: # Print progress every 10 batches
             print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_dataset)}: Loss={loss.item():.4f}")


    avg_epoch_loss = epoch_loss / len(train_dataset) if len(train_dataset) > 0 else 0
    avg_epoch_acc = epoch_correct / epoch_total_samples if epoch_total_samples > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} Summary: Avg Loss={avg_epoch_loss:.4f}, Avg Acc={avg_epoch_acc:.4f}")

    # Validation step (optional, can be done less frequently)
    if (epoch + 1) % 5 == 0 and len(validation_dataset) > 0: # Validate every 5 epochs
        model.eval()
        val_correct = 0
        val_total_samples = 0
        val_f1_scores = []
        val_mcc_scores = []
        with torch.no_grad():
            for val_sample in validation_dataset:
                X_val = val_sample['X'].to(device)
                A_val = val_sample['A'].to(device)
                C_val = val_sample['Y'].long().to(device)
                
                out_val_logits = model(X_val, A_val)
                val_preds = out_val_logits.argmax(dim=1)
                
                val_correct += (val_preds == C_val).sum().item()
                val_total_samples += C_val.size(0)
                val_f1_scores.append(f1_score(C_val.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0))
                val_mcc_scores.append(matthews_corrcoef(C_val.cpu().numpy(), val_preds.cpu().numpy()))
        
        avg_val_acc = val_correct / val_total_samples if val_total_samples > 0 else 0
        avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores) if val_f1_scores else 0
        avg_val_mcc = sum(val_mcc_scores) / len(val_mcc_scores) if val_mcc_scores else 0
        print(f"Epoch {epoch+1} Validation: Acc={avg_val_acc:.4f}, F1={avg_val_f1:.4f}, MCC={avg_val_mcc:.4f}")


# --- Final Testing ---
print("\nStarting final testing...")
model.eval()
test_correct = 0
test_total_samples = 0
test_f1_scores = []
test_mcc_scores = []

if len(test_dataset) == 0:
    print("Test dataset is empty. Skipping testing.")
else:
    with torch.no_grad():
        for test_sample in test_dataset:
            X_test = test_sample['X'].to(device)
            A_test = test_sample['A'].to(device)
            C_test = test_sample['Y'].long().to(device)
            
            out_test_logits = model(X_test, A_test)
            test_preds = out_test_logits.argmax(dim=1)
            
            test_correct += (test_preds == C_test).sum().item()
            test_total_samples += C_test.size(0)
            test_f1_scores.append(f1_score(C_test.cpu().numpy(), test_preds.cpu().numpy(), zero_division=0))
            test_mcc_scores.append(matthews_corrcoef(C_test.cpu().numpy(), test_preds.cpu().numpy()))

    avg_test_acc = test_correct / test_total_samples if test_total_samples > 0 else 0
    avg_test_f1 = sum(test_f1_scores) / len(test_f1_scores) if test_f1_scores else 0
    avg_test_mcc = sum(test_mcc_scores) / len(test_mcc_scores) if test_mcc_scores else 0
    print(f"Test Results: Accuracy={avg_test_acc:.4f}, F1 Score={avg_test_f1:.4f}, MCC={avg_test_mcc:.4f}")

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
