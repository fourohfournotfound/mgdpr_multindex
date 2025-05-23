import torch
# import csv # No longer directly used for reading individual stock files
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
import math
from torch.utils.data import Dataset
import multiprocessing # Added for multiprocessing

# Imports that were in the same cell block in the notebook, included for completeness,
# though not all are directly used by MyDataset.
from scipy.linalg import expm
import random
import sklearn.preprocessing as skp
from functools import lru_cache

def _static_prune(A: torch.Tensor, keep_rate: float = 0.25) -> torch.Tensor:
    """Keep the top-k edges per row of adjacency matrix.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency tensor of shape (..., N, N).
    keep_rate : float
        Fraction of edges to keep per row.

    Returns
    -------
    torch.Tensor
        Pruned adjacency tensor with the same shape as ``A``.
    """
    if A.numel() == 0:
        return A
    k = max(1, int(A.size(-1) * keep_rate))
    topk_vals, _ = torch.topk(A, k, dim=-1)
    thresh = topk_vals[..., -1:].expand_as(A)
    return A * (A >= thresh)

# Top-level worker function for multiprocessing
def _mp_worker_graph_generation_task(
    i_graph_idx: int,
    graph_file_path: str,
    current_window_dates_str_slice: List[str],
    # stock_data_df: pd.DataFrame, # Passed via my_dataset_instance
    comlist: List[str],
    market: str,
    window: int,
    # feature_columns: List[str], # Accessed via my_dataset_instance.node_feature_matrix
    my_dataset_instance: 'MyDataset', # Pass the instance to call its methods
    debug: bool = False,
) -> Tuple[int, str, str]:  # Returns (index, status, path_or_error)
    """
    Worker function to generate a single graph file.
    """
    if os.path.exists(graph_file_path):
        return i_graph_idx, "skipped_exists", graph_file_path

    try:
        # X_features will have shape (num_features, num_companies, window + 1)
        X_features_full_window = my_dataset_instance.node_feature_matrix(
            current_window_dates_str_slice, 
            comlist, # Use the comlist specific to this graph generation task (passed from _create_graphs)
            market
        )
        
        if X_features_full_window.shape[2] != window + 1:
            return i_graph_idx, "error_feature_matrix_shape", f"Node feature matrix for graph {i_graph_idx} has incorrect time steps: {X_features_full_window.shape[2]}, expected {window + 1}"

        # --- START: New Target Variable Calculation (Volatility-Adjusted, Z-Scored Returns) ---
        num_companies_in_graph = X_features_full_window.shape[1]
        # Initialize with NaNs, as some stocks might not have valid targets
        raw_vol_adj_returns_list = [float('nan')] * num_companies_in_graph
        
        target_date_str = current_window_dates_str_slice[-1] # This is t+1, the day for which we predict the return
        target_date_dt = pd.to_datetime(target_date_str)

        for company_idx in range(num_companies_in_graph):
            company_ticker = comlist[company_idx] # comlist is passed to the worker
            
            try:
                # Access pre-calculated DailyReturn and Volatility_20d from the main DataFrame
                # These are for the target_date_dt (t+1)
                # DailyReturn at t+1 is (Close_t+1 - Close_t) / Close_t
                # Volatility_20d for target_date_dt (t+1) is calculated using DailyReturns up to the previous day (t).
                next_day_return = my_dataset_instance.stock_data_df.loc[(company_ticker, target_date_dt), 'DailyReturn']
                volatility = my_dataset_instance.stock_data_df.loc[(company_ticker, target_date_dt), 'Volatility_20d']

                if pd.notna(next_day_return) and pd.notna(volatility):
                    if abs(volatility) > 1e-8: # Avoid division by zero or very small volatility
                        raw_vol_adj_returns_list[company_idx] = next_day_return / volatility
                    # else: remains NaN if volatility is too low (or zero)
                # else: remains NaN if next_day_return or volatility is NaN
            except KeyError:
                # Data might not be present for this specific ticker/date in the main df, leave as NaN
                # This handles cases where a stock might not have data for target_date_dt
                pass # raw_vol_adj_returns_list[company_idx] remains NaN

        raw_vol_adj_returns_tensor = torch.tensor(raw_vol_adj_returns_list, dtype=torch.float32)

        # Cross-sectional Z-scoring for the current day's (target_date_str) vol-adjusted returns
        valid_returns_mask = ~torch.isnan(raw_vol_adj_returns_tensor)
        C_labels = torch.full((num_companies_in_graph,), float('nan'), dtype=torch.float32) # Initialize final labels with NaNs

        if valid_returns_mask.any():
            valid_vol_adj_returns = raw_vol_adj_returns_tensor[valid_returns_mask]
            
            if len(valid_vol_adj_returns) > 1: # Need at least 2 points for a meaningful std dev
                mean_val = torch.mean(valid_vol_adj_returns)
                std_val = torch.std(valid_vol_adj_returns)
                
                if std_val > 1e-8: # Avoid division by zero if std is very small
                    C_labels[valid_returns_mask] = (valid_vol_adj_returns - mean_val) / std_val
                else:
                    # If std is zero (all valid returns are the same), their z-score is 0
                    C_labels[valid_returns_mask] = 0.0
            elif len(valid_vol_adj_returns) == 1:
                # If only one valid stock, its z-score is conventionally 0
                # (it's its own mean, std is undefined but for ranking purposes, it's neutral)
                 C_labels[valid_returns_mask] = 0.0
        # else: if no valid_vol_adj_returns found for this day, C_labels remains all NaNs.
        # This graph might be unusable or needs special handling in the loss/training.
        # --- END: New Target Variable Calculation ---
        
        X_processed = X_features_full_window[:, :, :-1].clone()

        for k_feature_idx in range(X_processed.shape[0]):
            # Clamp values to avoid log1p producing -inf or NaN when data
            # contains values <= -1. 1e-6 is used as a tiny epsilon so the
            # minimum passed to log1p is (-1 + epsilon).
            clamped_vals = torch.clamp(X_processed[k_feature_idx], min=-1 + 1e-6)
            X_processed[k_feature_idx] = torch.log1p(clamped_vals)
            # Z-score normalization removed to match demo notebook more closely for now
            # mean = X_processed[k_feature_idx].mean(dim=1, keepdim=True)
            # std = X_processed[k_feature_idx].std(dim=1, keepdim=True)
            # Add a small epsilon to std to prevent division by zero if a feature is constant
            # X_processed[k_feature_idx] = (X_processed[k_feature_idx] - mean) / (std + 1e-9)

        # Apply global scaling if scaler is provided
        if my_dataset_instance.scaler is not None:
            # X_processed has shape (num_total_features, num_companies, window_size)
            # We scale the first 'feature_dim' (e.g., 5 for O,H,L,C,V)
            num_main_features_to_scale = my_dataset_instance.feature_dim
            if X_processed.shape[0] >= num_main_features_to_scale:
                features_slice_to_scale = X_processed[:num_main_features_to_scale, :, :].clone() # (feature_dim, num_comp, window)
                original_slice_shape = features_slice_to_scale.shape
                
                # Transpose to (num_comp, window, feature_dim) then reshape to (num_comp*window, feature_dim)
                # This matches the shape scaler was fit on (samples, features)
                features_2d = features_slice_to_scale.permute(1, 2, 0).reshape(-1, num_main_features_to_scale)
                
                features_2d_np = features_2d.cpu().numpy()
                scaled_features_2d_np = my_dataset_instance.scaler.transform(features_2d_np)
                scaled_features_2d_torch = torch.tensor(scaled_features_2d_np, dtype=torch.float32, device=X_processed.device)
                
                # Reshape back to (num_comp, window, feature_dim) then permute to (feature_dim, num_comp, window)
                X_processed[:num_main_features_to_scale, :, :] = scaled_features_2d_torch.reshape(original_slice_shape[1], original_slice_shape[2], num_main_features_to_scale).permute(2, 0, 1)

        A_adj = torch.zeros((X_processed.shape[0], X_processed.shape[1], X_processed.shape[1]))
        for l_feature_idx in range(A_adj.shape[0]):
            A_adj[l_feature_idx] = my_dataset_instance.adjacency_matrix(X_processed[l_feature_idx])

        # Static sparsification: keep only a fraction of edges per row
        A_adj = _static_prune(A_adj, keep_rate=0.25)
        
        # Temporary print statements for debugging A_adj statistics
        if debug and i_graph_idx < 3:  # Log for the first few graphs if debugging enabled
            print(
                f"DEBUG Adjacency Matrix Stats for graph {i_graph_idx} (File: {os.path.basename(graph_file_path)}):"
            )
            for rel_idx in range(A_adj.shape[0]):
                adj_slice = A_adj[rel_idx]
                print(f"  Relation {rel_idx}:")
                print(
                    f"    Min: {adj_slice.min().item():.4e}, Max: {adj_slice.max().item():.4e}, Mean: {adj_slice.mean().item():.4e}"
                )
                print(
                    f"    Has NaN: {torch.isnan(adj_slice).any().item()}, Has Inf: {torch.isinf(adj_slice).any().item()}"
                )
            # Also log label distribution for this graph
            unique_labels, counts = torch.unique(C_labels, return_counts=True)
            label_dist_str = ", ".join(
                [f"Label {l.item()}: {c.item()}" for l, c in zip(unique_labels, counts)]
            )
            print(f"  Label Distribution for graph {i_graph_idx}: {label_dist_str}")

        torch.save({'X': X_processed, 'A': A_adj, 'Y': C_labels}, graph_file_path)
        return i_graph_idx, "success", graph_file_path
    except Exception as e:
        # Log error or handle as appropriate
        # print(f"Error generating graph {i_graph_idx} ({graph_file_path}): {e}")
        return i_graph_idx, "error_exception", f"Exception for graph {i_graph_idx}: {str(e)}"


class MyDataset(Dataset):
    """
    Custom PyTorch Dataset for MGDPR model.
    Handles loading stock data from a single CSV, processing it into a MultiIndex DataFrame,
    and generating graph structures (.pt files) for specified time windows.
    Each graph file contains node features (X), adjacency matrices (A) per relation (feature),
    and node labels (Y) for stock trend classification.

    The graph generation process involves:
    1. Loading a comprehensive stock data CSV.
    2. Filtering data for a given list of companies (comlist) and date range.
    3. For each valid time step, creating a lookback window of `window` days.
    4. Extracting node features (Open, High, Low, Close, Volume) for each stock in the window.
    5. Calculating an adjacency matrix for each feature type (relation) based on signal energy
       and information entropy, as per the MGDPR paper (Section 4.1).
    6. Determining node labels based on the price movement of the next day (relative to the window end).
    7. Saving each graph (X, A, Y) as a .pt file.
    
    The dataset uses multiprocessing for efficient graph generation if files are missing or incomplete.
    """
    def __init__(
        self,
        root_csv_path: str,
        desti: str,
        market: str,
        comlist: List[str],
        start: str,
        end: str,
        window: int,
        dataset_type: str,
        scaler=None,
        selected_features: List[int] | None = None,
        debug: bool = False,
        apply_scaler_on_load: bool = False,
        skip_graph_generation: bool = False,
    ):
        """
        Initializes the dataset, loads stock data, and triggers graph generation if needed.

        Args:
            root_csv_path (str): Path to the single CSV file containing all stock data.
                                 Expected columns: Ticker, Date, Open, High, Low, Close, Volume.
            desti (str): Destination directory where processed graph .pt files will be saved.
                         A subdirectory will be created here named:
                         f'{market}_{dataset_type}_{start_date}_{end_date}_{window_size}'.
            market (str): Name of the market (e.g., "NASDAQ", "Shortlist"). Used for naming the graph directory.
            comlist (List[str]): List of stock tickers to include in the dataset.
            start (str): Start date string for the dataset period (YYYY-MM-DD).
            end (str): End date string for the dataset period (YYYY-MM-DD).
            window (int): Lookback window size (tau in paper) for constructing graph features.
            dataset_type (str): Type of dataset (e.g., "Train", "Validation", "Test"). Used for naming.
            scaler (sklearn.preprocessing.StandardScaler, optional): Fitted scaler for features. Defaults to None.
            selected_features (List[int] | None, optional): Indices of features to keep
                for each graph sample. If ``None``, all features are returned during
                loading. Feature selection typically comes from an offline routine
                such as GRACES.
            debug (bool, optional): If ``True``, prints additional debug information
                during initialization and graph generation. Defaults to ``False``.
            apply_scaler_on_load (bool, optional): If ``True`` and ``scaler`` is
                provided, scaling is applied in ``__getitem__`` when graphs are
                loaded rather than during generation. This allows reusing existing
                unscaled graphs. Defaults to ``False``.
            skip_graph_generation (bool, optional): If ``True``, graph creation
                is skipped even if files appear missing. ``__len__`` will reflect
                only existing graphs. Useful when reusing graphs generated by
                another dataset instance. Defaults to ``False``.
        """
        self.debug = debug
        if self.debug:
            print("--- DEBUG: MyDataset __init__ ENTERED ---")
        super().__init__()

        self.comlist = comlist # This is the overall list of companies for the dataset instance
        self.market = market
        self.root_csv_path = root_csv_path
        self.desti = desti
        self.start_str = start
        self.end_str = end
        self.window = window
        self.dataset_type = dataset_type
        self.scaler = scaler
        self.selected_features = selected_features
        self.apply_scaler_on_load = apply_scaler_on_load
        self.skip_graph_generation = skip_graph_generation
        # feature_dim and feature_columns will be determined after loading the CSV
        self.feature_dim = 0
        self.feature_columns: List[str] = []
        
        # Load the CSV and prepare the MultiIndex DataFrame
        try:
            with open(self.root_csv_path, 'r') as f:
                first_line = f.readline().strip()
            
            looks_like_data = False
            if first_line:
                first_item = first_line.split(',')[0].strip('"').strip("'")
                if first_item in self.comlist: # Use self.comlist for initial check
                    looks_like_data = True
            
            expected_column_names = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # user_example_cols defines a potential set of columns if the CSV is headerless.
            # When using a headerless file we only guarantee these names for the
            # first few columns, but we still load all remaining columns so that
            # additional features are preserved.
            user_example_cols = [
                'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'AdjClose', 'OriginalDate1', 'OriginalDate2'
            ]

            if looks_like_data:
                # If the CSV appears to be headerless we still want to preserve
                # any additional columns beyond the expected ones.  Read the
                # entire file first, then assign names to the known columns.
                temp_df = pd.read_csv(self.root_csv_path, header=None, low_memory=False)
                col_count = temp_df.shape[1]
                default_names = user_example_cols[:col_count]
                if col_count > len(user_example_cols):
                    default_names += [f'ExtraCol{i}' for i in range(col_count - len(user_example_cols))]
                temp_df.columns = default_names
                raw_df = temp_df
            else:
                raw_df = pd.read_csv(self.root_csv_path, low_memory=False)
                rename_map = {}
                for col in raw_df.columns:
                    col_s = str(col).strip().lower()
                    if col_s in ['ticker', 'symbol']: rename_map[col] = 'Ticker'
                    elif col_s == 'date': rename_map[col] = 'Date'
                    elif col_s == 'open': rename_map[col] = 'Open'
                    elif col_s == 'high': rename_map[col] = 'High'
                    elif col_s == 'low': rename_map[col] = 'Low'
                    elif col_s == 'close' or col_s == 'closeadj': rename_map[col] = 'Close'
                    elif col_s == 'volume': rename_map[col] = 'Volume'
                raw_df = raw_df.rename(columns=rename_map)
    
            # This check is now more critical. If `usecols` was used, raw_df will only have those columns.
            # If it was a headed CSV, this check is still important after renaming.
            if not all(col in raw_df.columns for col in expected_column_names):
                # This block might need adjustment if usecols makes it impossible to reach here for headerless.
                # For headerless CSVs with `usecols=expected_column_names`, raw_df.columns should exactly be expected_column_names.
                # So, this `if` condition would primarily be for headed CSVs that after renaming are missing columns.
                if looks_like_data and len(raw_df.columns) >= len(expected_column_names):
                    # This part might be less relevant now for headerless if usecols is effective.
                    # If usecols correctly restricts columns, raw_df.columns will be expected_column_names.
                    col_map_by_pos = {raw_df.columns[i]: expected_column_names[i] for i in range(len(expected_column_names))}
                    raw_df = raw_df.rename(columns=col_map_by_pos) # This rename might be redundant if usecols + names worked as expected
                # For headed CSVs, if renaming didn't yield all expected columns:
                elif not looks_like_data:
                     raise ValueError(f"CSV (with header) must contain or be mappable to columns: {expected_column_names}. Found after renaming: {raw_df.columns.tolist()}")
                # For headerless, if usecols somehow didn't result in the expected columns (should not happen if CSV has enough columns)
                elif looks_like_data:
                     raise ValueError(f"Headerless CSV, despite using usecols={expected_column_names}, did not result in the expected columns. Columns found: {raw_df.columns.tolist()}")

            
            raw_df = raw_df.loc[:, ~raw_df.columns.duplicated(keep='first')]
            # Preserve all columns to allow additional features beyond the expected set.
            self.stock_data_df = raw_df.copy()


            try:
                self.stock_data_df['Date'] = pd.to_datetime(self.stock_data_df['Date'], format='%m-%d-%y')
            except ValueError:
                try:
                    self.stock_data_df['Date'] = pd.to_datetime(self.stock_data_df['Date'])
                except Exception as e_date:
                    raise ValueError(f"Could not parse Date column. Ensure format is MM-DD-YY or YYYY-MM-DD. Error: {e_date}")

            self.stock_data_df = self.stock_data_df.dropna(subset=['Ticker', 'Date'])
            self.stock_data_df = self.stock_data_df[self.stock_data_df['Ticker'].isin(self.comlist)] 

            numeric_cols_to_convert = [c for c in self.stock_data_df.columns if c not in ['Ticker', 'Date']]
            mandatory_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in self.stock_data_df.columns]
            if self.stock_data_df.empty:
                raise ValueError(
                    f"No valid data remaining for comlist: {self.comlist} from {self.root_csv_path} (DataFrame became empty before numeric conversion loop)."
                )

            for col_to_convert in numeric_cols_to_convert:
                if col_to_convert in self.stock_data_df:
                    try:
                        column_data_slice = self.stock_data_df[col_to_convert]
                        if isinstance(column_data_slice, pd.Series):
                            self.stock_data_df[col_to_convert] = pd.to_numeric(column_data_slice, errors='coerce')
                        elif isinstance(column_data_slice, pd.DataFrame):
                            if not column_data_slice.empty:
                                self.stock_data_df[col_to_convert] = pd.to_numeric(
                                    column_data_slice.iloc[:, 0], errors='coerce'
                                )
                    except Exception as e_slice_convert:
                        print(
                            f"ERROR MyDataset: Exception during slice or pd.to_numeric for column '{col_to_convert}': {e_slice_convert}"
                        )

            if mandatory_cols:
                self.stock_data_df = self.stock_data_df.dropna(subset=mandatory_cols)

            # Fill remaining NaNs in other feature columns with zeros
            non_index_cols = [c for c in self.stock_data_df.columns if c not in ['Ticker', 'Date']]
            self.stock_data_df[non_index_cols] = self.stock_data_df[non_index_cols].fillna(0.0)

            if self.stock_data_df.empty:
                raise ValueError(f"No valid data remaining after numeric conversion and NaN drop for comlist: {self.comlist} from {self.root_csv_path}")

            self.stock_data_df = self.stock_data_df.set_index(['Ticker', 'Date'])
            self.stock_data_df = self.stock_data_df.sort_index()

            # --- START: New code for target variable calculation prerequisites ---
            if not self.stock_data_df.empty:
                # Calculate Daily Returns
                # Group by Ticker, then calculate percentage change for 'Close' prices
                # Using apply with lambda for explicit group-wise operation.
                self.stock_data_df['DailyReturn'] = self.stock_data_df.groupby(level='Ticker', group_keys=False)['Close'].apply(lambda x: x.pct_change())

                # Calculate 20-day rolling volatility of daily returns
                # min_periods=20 ensures that we only get a value if there are 20 data points,
                # effectively making volatility NaN for initial periods with insufficient data.
                # Shift(1) ensures that volatility for day t+1 is based on returns up to day t.
                vol_series = self.stock_data_df.groupby(level='Ticker', group_keys=False)['DailyReturn'].apply(lambda x: x.rolling(window=20, min_periods=20).std().shift(1))
                self.stock_data_df['Volatility_20d'] = vol_series

                # Note: 'DailyReturn' will have NaN for the first entry of each stock.
                # 'Volatility_20d' will have NaNs for the first 20 data entries of each stock (19 from rolling, 1 from shift).
                # These NaNs are expected and will be handled during target calculation in the worker.
            else:
                # If stock_data_df is empty, create empty columns to prevent KeyErrors later
                # if other parts of the code expect these columns to exist.
                self.stock_data_df['DailyReturn'] = pd.Series(dtype=float)
                self.stock_data_df['Volatility_20d'] = pd.Series(dtype=float)
            # Determine available feature columns (exclude label-related columns)
            self.feature_columns = [
                c for c in self.stock_data_df.columns if c not in ['Ticker', 'Date', 'DailyReturn', 'Volatility_20d']
            ]
            self.feature_dim = len(self.feature_columns)
            # --- END: New code for target variable calculation prerequisites ---

        except FileNotFoundError:
            print(f"ERROR: Stock data CSV file not found at {self.root_csv_path}")
            raise
        except Exception as e:
            print(f"ERROR: Could not load or process stock data from {self.root_csv_path}: {e}")
            raise

        self.dates, self.next_day = self.find_dates(self.start_str, self.end_str, self.comlist)

        # Store dates_for_graph_gen for use in __getitem__
        self.dates_for_graph_gen_stored = []
        if self.dates: # Ensure self.dates is not empty
            self.dates_for_graph_gen_stored = list(self.dates) # Make a copy
            if self.next_day and self.next_day not in self.dates_for_graph_gen_stored and len(self.dates_for_graph_gen_stored) >= self.window:
                self.dates_for_graph_gen_stored.append(self.next_day)
            self.dates_for_graph_gen_stored.sort() # Ensure sorted order
        
        graph_dir_name = f'{self.market}_{self.dataset_type}_{self.start_str}_{self.end_str}_{self.window}'
        self.graph_directory_path = os.path.abspath(os.path.join(self.desti, graph_dir_name))

        theoretical_max_graphs = 0
        if self.dates: 
            temp_actual_dates_for_calc = list(self.dates) 
            if self.next_day and \
               self.next_day not in temp_actual_dates_for_calc and \
               len(temp_actual_dates_for_calc) >= self.window:
                temp_actual_dates_for_calc.append(self.next_day)
            
            if len(temp_actual_dates_for_calc) >= self.window + 1:
                theoretical_max_graphs = len(temp_actual_dates_for_calc) - self.window
        
        all_theoretical_files_exist = False 
        if theoretical_max_graphs == 0:
            all_theoretical_files_exist = True
        elif os.path.exists(self.graph_directory_path):
            all_theoretical_files_exist = True 
            for i in range(theoretical_max_graphs):
                if not os.path.exists(os.path.join(self.graph_directory_path, f'graph_{i}.pt')):
                    all_theoretical_files_exist = False
                    break
        
        needs_generation = False
        if theoretical_max_graphs > 0 and not all_theoretical_files_exist:
            needs_generation = True
            print(f"Graph files at {self.graph_directory_path} seem incomplete or missing for {theoretical_max_graphs} theoretical graphs. Regenerating...")
        elif theoretical_max_graphs > 0 and not os.path.exists(self.graph_directory_path): 
            needs_generation = True
            print(f"Graph directory {self.graph_directory_path} missing. Generating {theoretical_max_graphs} theoretical graphs...")
        
        if needs_generation:
            if self.skip_graph_generation:
                print("Graph generation skipped by configuration.")
            else:
                self._create_graphs(self.dates, self.desti, self.comlist, self.market, self.window, self.next_day)
        elif theoretical_max_graphs > 0 : 
             print(f"Graph files at {self.graph_directory_path} appear to be complete for {theoretical_max_graphs} theoretical graphs. Skipping generation.")
        else: 
            print(f"No graphs are theoretically possible (theoretical_max_graphs=0) for {self.graph_directory_path}. Skipping generation.")

        self._actual_len = 0
        if os.path.exists(self.graph_directory_path):
            for i in range(theoretical_max_graphs):
                if os.path.exists(os.path.join(self.graph_directory_path, f'graph_{i}.pt')):
                    self._actual_len += 1
                else:
                    break
        
        print(f"MyDataset initialized for {self.graph_directory_path}. Actual length determined: {self._actual_len}.")
        if self._actual_len == 0 and theoretical_max_graphs > 0:
            print(f"Warning: Actual length is 0, but {theoretical_max_graphs} graphs were theoretically possible. ")
        elif self._actual_len < theoretical_max_graphs:
             print(f"Warning: Actual length {self._actual_len} is less than theoretical max {theoretical_max_graphs} for {self.graph_directory_path}. ")

        if self._actual_len > 0:
            print(f"{self._actual_len} graph files are expected to be available for lazy loading.")
        elif theoretical_max_graphs > 0:
             print(f"Warning: No graph files seem to be available (_actual_len is 0), though {theoretical_max_graphs} were theoretically possible.")


    def __len__(self):
        return self._actual_len 

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._actual_len:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._actual_len}")
        
        data_path = os.path.join(self.graph_directory_path, f'graph_{idx}.pt')
        
        try:
            graph_data = torch.load(data_path)
            
            # Add target_date_str and original_ticker_order to the sample
            target_date_str_for_sample = "UNKNOWN_DATE" # Default
            if hasattr(self, 'dates_for_graph_gen_stored') and self.dates_for_graph_gen_stored:
                # The target date for graph 'idx' is the (idx + window_size)-th date in dates_for_graph_gen_stored
                # This corresponds to the last date in the (window_size + 1) slice used to create graph 'idx'
                # dates_for_graph_gen_stored[i_graph_idx : i_graph_idx + window_size + 1]
                # The last element of this slice is dates_for_graph_gen_stored[i_graph_idx + window_size]
                target_date_lookup_idx = idx + self.window
                if 0 <= target_date_lookup_idx < len(self.dates_for_graph_gen_stored):
                    target_date_str_for_sample = self.dates_for_graph_gen_stored[target_date_lookup_idx]
                else:
                    # This case should ideally not happen if _actual_len is correct
                    # and dates_for_graph_gen_stored covers all possible target dates.
                    # print(f"Warning: MyDataset.__getitem__({idx}): target_date_lookup_idx {target_date_lookup_idx} out of bounds for dates_for_graph_gen_stored (len {len(self.dates_for_graph_gen_stored)}).")
                    pass # Keep UNKNOWN_DATE or handle error
            else:
                # print(f"Warning: MyDataset.__getitem__({idx}): self.dates_for_graph_gen_stored not available or empty.")
                pass

            graph_data['target_date_str'] = target_date_str_for_sample
            graph_data['original_ticker_order'] = self.comlist  # self.comlist is the list of tickers for this dataset split

            if self.apply_scaler_on_load and self.scaler is not None and 'X' in graph_data:
                num_main_features_to_scale = self.feature_dim
                if graph_data['X'].shape[0] >= num_main_features_to_scale:
                    features_slice_to_scale = graph_data['X'][:num_main_features_to_scale, :, :].clone()
                    orig_shape = features_slice_to_scale.shape
                    features_2d = features_slice_to_scale.permute(1, 2, 0).reshape(-1, num_main_features_to_scale)
                    scaled_np = self.scaler.transform(features_2d.cpu().numpy())
                    scaled_torch = torch.tensor(scaled_np, dtype=torch.float32, device=graph_data['X'].device)
                    graph_data['X'][:num_main_features_to_scale, :, :] = scaled_torch.reshape(orig_shape[1], orig_shape[2], num_main_features_to_scale).permute(2, 0, 1)

            if self.selected_features is not None:
                feat_idx = torch.tensor(self.selected_features, dtype=torch.long)
                if 'X' in graph_data:
                    graph_data['X'] = graph_data['X'][feat_idx]
                if 'A' in graph_data:
                    graph_data['A'] = graph_data['A'][feat_idx]

            return graph_data
        except FileNotFoundError:
            print(f"ERROR MyDataset.__getitem__({idx}): File not found at {data_path}.")
            raise 
        except Exception as e:
            print(f"ERROR MyDataset.__getitem__({idx}): Failed to load graph from {data_path}: {e}")
            raise


    def check_years(self, date_str: str, start_str: str, end_str: str) -> bool:
        date_format = "%Y-%m-%d"
        date = datetime.strptime(date_str, date_format)
        start = datetime.strptime(start_str, date_format)
        end = datetime.strptime(end_str, date_format)
        return start <= date <= end

    def find_dates(self, start_str_param: str, end_str_param: str, comlist_arg: List[str]) -> Tuple[List[str], str]: # Renamed comlist to comlist_arg
        """
        Finds common trading dates for the given companies (comlist_arg) within self.stock_data_df.
        Uses parameters for start/end dates directly.
        Leverages pandas vectorized operations for improved performance.
        """
        start_dt = pd.to_datetime(start_str_param)
        end_dt = pd.to_datetime(end_str_param)
        future_limit_dt = end_dt + pd.Timedelta(days=90) 

        available_tickers_in_df = self.stock_data_df.index.get_level_values('Ticker').unique()
        valid_comlist_arg = [ticker for ticker in comlist_arg if ticker in available_tickers_in_df]
        
        if len(valid_comlist_arg) < len(comlist_arg):
            missing_tickers = set(comlist_arg) - set(valid_comlist_arg)
            if missing_tickers: 
                 print(f"Warning: Requested tickers not found in the pre-loaded DataFrame's data: {missing_tickers}. Proceeding with available tickers: {valid_comlist_arg}")
        
        if not valid_comlist_arg:
            print(f"Warning: None of the requested tickers {comlist_arg} (or valid subset) found in the DataFrame. Cannot find common dates.")
            return [], None
        
        comlist_to_use_for_find_dates = valid_comlist_arg
        num_expected_tickers = len(comlist_to_use_for_find_dates)
        
        idx_df = self.stock_data_df.index.to_frame(index=False)
        idx_df_filtered_for_current_comlist = idx_df[idx_df['Ticker'].isin(comlist_to_use_for_find_dates)]
        
        relevant_ticker_dates_df = idx_df_filtered_for_current_comlist[
            (idx_df_filtered_for_current_comlist['Date'] >= start_dt) &
            (idx_df_filtered_for_current_comlist['Date'] <= future_limit_dt)
        ]

        if relevant_ticker_dates_df.empty:
            print(f"Warning: No data found for the valid tickers {comlist_to_use_for_find_dates} within the date range {start_str_param} to {future_limit_dt.strftime('%Y-%m-%d')}.")
            return [], None

        date_counts = relevant_ticker_dates_df.groupby('Date')['Ticker'].nunique()
        common_dates_ts_series = date_counts[date_counts == num_expected_tickers].index

        if common_dates_ts_series.empty:
            print(f"Warning: No dates found where all {num_expected_tickers} companies in {comlist_to_use_for_find_dates} have data, for the period {start_str_param} to {future_limit_dt.strftime('%Y-%m-%d')}.")
            return [], None 
            
        actual_common_dates_str = []
        after_end_common_dates_str = []

        for date_ts in common_dates_ts_series:
            date_str = date_ts.strftime("%Y-%m-%d") 
            if date_ts <= end_dt: 
                actual_common_dates_str.append(date_str)
            elif date_ts > end_dt: 
                after_end_common_dates_str.append(date_str)
        
        next_common_day_str = min(after_end_common_dates_str) if after_end_common_dates_str else None

        if not actual_common_dates_str:
             print(f"Warning: No common trading dates found for companies {comlist_to_use_for_find_dates} strictly between {start_str_param} and {end_str_param}. (next_common_day_str exists: {next_common_day_str is not None})")
        
        return sorted(actual_common_dates_str), next_common_day_str

    def signal_energy(self, X_company_series: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(X_company_series), dim=1)

    def information_entropy(self, X_company_series: torch.Tensor) -> torch.Tensor:
        """Compute information entropy of each company's series.

        Parameters
        ----------
        X_company_series : torch.Tensor
            Tensor of shape (num_companies, window_size)

        Returns
        -------
        torch.Tensor
            Entropy values for each company (shape: ``(num_companies,)``).
        """
        num_companies = X_company_series.shape[0]
        device = X_company_series.device
        entropies = torch.zeros(num_companies, dtype=torch.float32, device=device)

        for i in range(num_companies):
            row = X_company_series[i]
            if row.numel() == 0:
                continue
            unique_elements, counts = torch.unique(row, return_counts=True)
            probabilities = counts.float() / counts.sum().float()
            if probabilities.numel() > 0:
                entropies[i] = -torch.sum(probabilities * torch.log(probabilities))

        return entropies

    def adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates the adjacency matrix based on information entropy and signal energy,
        as per Section 4.1 of the MGDPR paper.
        (a_{t,r})_{i,j} = (E(x_{t,r,i}) / E(x_{t,r,j})) * exp(H(x_{t,r,i}) - H(x_{t,r,j}))
        
        Note: The original implementation in the demo notebook included a log transformation
        after this calculation (A[A<1]=1 then log(A)). This has been removed to align
        strictly with the paper's formula.

        Args:
            X (torch.Tensor): Input node features for a single relation (feature type)
                              over the lookback window.
                              Shape: (num_companies, window_size_features).
        
        Returns:
            torch.Tensor: Calculated adjacency matrix for the given relation.
                          Shape: (num_companies, num_companies).
        """
        if X.ndim != 2:
            raise ValueError(f"Input X to adjacency_matrix must be 2D (num_companies, num_features_per_company). Got shape {X.shape}")
        num_companies = X.shape[0]
        if num_companies == 0:
            return torch.empty((0, 0), dtype=torch.float32)
        X_float = X.float() 
        energy_t = self.signal_energy(X_float) 
        entropy_t = self.information_entropy(X_float)
        energy_i = energy_t.unsqueeze(1)
        energy_j = energy_t.unsqueeze(0)
        entropy_i = entropy_t.unsqueeze(1)
        entropy_j = entropy_t.unsqueeze(0)
        zero_energy_j_mask = (energy_j == 0)
        A = torch.zeros((num_companies, num_companies), dtype=torch.float32, device=X.device)
        energy_ratio = torch.where(zero_energy_j_mask, torch.zeros_like(energy_i), energy_i / (energy_j + 1e-9)) 
        exp_entropy_diff = torch.exp(entropy_i - entropy_j)
        A_calculated_values = energy_ratio * exp_entropy_diff
        A = torch.where(zero_energy_j_mask.expand_as(A), torch.zeros_like(A), A_calculated_values)
        
        return A

    def node_feature_matrix(self, window_dates_str: List[str], comlist_arg: List[str], market: str) -> torch.Tensor:  # Renamed comlist to comlist_arg
        feature_columns = self.feature_columns
        num_features = len(feature_columns)
        num_companies = len(comlist_arg)
        num_days_in_window = len(window_dates_str)
        zero_X_tensor = torch.zeros((num_features, num_companies, num_days_in_window), dtype=torch.float32)

        if not comlist_arg or not window_dates_str:
            return zero_X_tensor
        try:
            window_dates_ts = [pd.to_datetime(date_str) for date_str in window_dates_str]
            multi_idx = pd.MultiIndex.from_product([comlist_arg, window_dates_ts], names=['Ticker', 'Date'])
            # Ensure self.stock_data_df is used here, it holds all the data for the instance
            reindexed_df = self.stock_data_df.loc[:, feature_columns].reindex(multi_idx)
            filled_df = reindexed_df.fillna(0.0)
            np_values = filled_df.values
            reshaped_array = np_values.reshape((num_companies, num_days_in_window, num_features))
            transposed_array = reshaped_array.transpose(2, 0, 1)
            X = torch.tensor(transposed_array, dtype=torch.float32)
            return X
        except KeyError as e:
            print(f"Error node_feature_matrix: KeyError - {e}. One or more feature columns not found. Returning zero tensor.")
            return zero_X_tensor
        except ValueError as e:
            print(f"Error node_feature_matrix: ValueError - {e}. Could not reshape array. Returning zero tensor.")
            return zero_X_tensor
        except Exception as e:
            print(f"Error node_feature_matrix: An unexpected error occurred - {e}. Returning zero tensor.")
            return zero_X_tensor

    def _create_graphs(self, dates: List[str], desti: str, comlist_for_graphs: List[str], market_name: str, window_size: int, next_day_str: str):
        # comlist_for_graphs is self.comlist passed from __init__
        # market_name is self.market, window_size is self.window, next_day_str is self.next_day
        if not dates:
            print("Warning: No dates provided to _create_graphs. Skipping graph generation.")
            return

        dates_for_graph_gen = list(dates) # Make a copy
        if next_day_str and next_day_str not in dates_for_graph_gen and len(dates_for_graph_gen) >= window_size:
            dates_for_graph_gen.append(next_day_str)

        if len(dates_for_graph_gen) <= window_size:
            print(f"Warning: Not enough dates ({len(dates_for_graph_gen)}) to form a window of size {window_size}+1. Skipping graph generation.")
            return
        
        os.makedirs(self.graph_directory_path, exist_ok=True)
        num_possible_graphs = len(dates_for_graph_gen) - window_size
        
        if num_possible_graphs <= 0:
            print(f"Not enough data points in dates_for_graph_gen ({len(dates_for_graph_gen)}) to create any graphs with window size {window_size}.")
            return

        tasks = []
        for i_graph_idx in range(num_possible_graphs):
            graph_file_path = os.path.join(self.graph_directory_path, f'graph_{i_graph_idx}.pt')
            current_window_dates_str_slice = dates_for_graph_gen[i_graph_idx : i_graph_idx + window_size + 1]
            
            if os.path.exists(graph_file_path):
                # print(f"Graph {i_graph_idx} already exists. Skipping generation.") # Optional: too verbose for MP
                continue
            
            if len(current_window_dates_str_slice) != window_size + 1:
                print(f"Warning: Skipped graph {i_graph_idx}. Not enough dates for full window + label. Got {len(current_window_dates_str_slice)}, needed {window_size + 1}.")
                continue
            
            # Arguments for the worker function:
            # (i_graph_idx, graph_file_path, current_window_dates_str_slice, comlist, market, window, my_dataset_instance)
            # comlist_for_graphs is self.comlist, used by node_feature_matrix inside worker
            task_args = (
                i_graph_idx,
                graph_file_path,
                current_window_dates_str_slice,
                comlist_for_graphs,  # This is self.comlist
                market_name,
                window_size,
                self,  # Pass the MyDataset instance itself
                self.debug,
            )
            tasks.append(task_args)

        if not tasks:
            print("No new graphs to generate.")
            return

        num_processes = os.cpu_count()
        print(f"Starting graph generation with {num_processes} processes for {len(tasks)} tasks...")

        try:
            if multiprocessing.get_start_method(allow_none=True) != 'fork':
                if 'fork' in multiprocessing.get_all_start_methods():
                    multiprocessing.set_start_method('fork', force=True) 
                    print("Set multiprocessing start method to 'fork'.")
        except (RuntimeError, AttributeError) as e_mp_setup: 
            print(f"Could not set multiprocessing start method to 'fork' (Error: {e_mp_setup}). Using default: {multiprocessing.get_start_method()}.")


        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for result in tqdm(pool.starmap(_mp_worker_graph_generation_task, tasks), total=len(tasks), desc="Generating Graphs (MP)"):
                results.append(result)
        
        successful_count = 0
        skipped_count = 0
        error_count = 0
        for r_idx, r_status, r_msg in results:
            if r_status == "success":
                successful_count += 1
            elif r_status == "skipped_exists":
                skipped_count +=1
            else: 
                error_count += 1
                print(f"Error in worker for graph {r_idx}: {r_status} - {r_msg}")
        
        print(f"Multiprocessing graph generation complete. Successful: {successful_count}, Skipped (exists): {skipped_count}, Errors: {error_count}")

# Placeholder for MultiIndexDataset if it's defined elsewhere or needed later.
# class MultiIndexDataset(Dataset):
#     def __init__(self, ...):
#         pass
#     def __len__(self):
#         pass
#     def __getitem__(self, idx):
#         pass
