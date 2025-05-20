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

# Imports that were in the same cell block in the notebook, included for completeness,
# though not all are directly used by MyDataset.
from scipy.linalg import expm
import random
import sklearn.preprocessing as skp
from functools import lru_cache


class MyDataset(Dataset):
    def __init__(self, root_csv_path: str, desti: str, market: str, comlist: List[str], start: str, end: str, window: int, dataset_type: str):
        print("--- DEBUG: MyDataset __init__ ENTERED ---") # Very simple debug print
        super().__init__()

        self.comlist = comlist
        self.market = market
        self.root_csv_path = root_csv_path
        self.desti = desti
        self.start_str = start # Store original string for directory naming
        self.end_str = end     # Store original string for directory naming
        self.window = window
        self.dataset_type = dataset_type # Initialize dataset_type earlier
        
        # Load the CSV and prepare the MultiIndex DataFrame
        try:
            # Attempt to read the first line to check for headers vs data
            with open(self.root_csv_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Heuristic: if the first item in the CSV looks like a ticker from comlist, assume no header
            # This is a simplified check. A more robust check might involve trying to parse dates, numbers etc.
            looks_like_data = False
            if first_line:
                first_item = first_line.split(',')[0].strip('"').strip("'")
                if first_item in comlist:
                    looks_like_data = True
            
            expected_column_names = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # If the user's CSV has more columns like the example, we'll need to adjust.
            # The example had 10 columns: Ticker,Date,Open,High,Low,Close,Volume,Adj Close,OriginalDate,AnotherDate
            # We will select the ones we need.
            user_example_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose', 'OriginalDate1', 'OriginalDate2']


            if looks_like_data:
                raw_df = pd.read_csv(self.root_csv_path, header=None, names=user_example_cols, low_memory=False)
            else:
                raw_df = pd.read_csv(self.root_csv_path, low_memory=False)
                # Attempt to standardize column names if headers are present
                rename_map = {}
                for col in raw_df.columns:
                    col_s = str(col).strip().lower()
                    if col_s in ['ticker', 'symbol']: rename_map[col] = 'Ticker'
                    elif col_s == 'date': rename_map[col] = 'Date'
                    elif col_s == 'open': rename_map[col] = 'Open'
                    elif col_s == 'high': rename_map[col] = 'High'
                    elif col_s == 'low': rename_map[col] = 'Low'
                    elif col_s == 'close' or col_s == 'closeadj': rename_map[col] = 'Close' # Added 'closeadj'
                    elif col_s == 'volume': rename_map[col] = 'Volume'
                raw_df = raw_df.rename(columns=rename_map)
    
                # Select only the columns we absolutely need
            if not all(col in raw_df.columns for col in expected_column_names):
                 # If expected names are not present after auto-detection/renaming, try to map by position if no header was detected
                if looks_like_data and len(raw_df.columns) >= len(expected_column_names):
                    # This assumes the first 7 columns are Ticker, Date, Open, High, Low, Close, Volume in order
                    col_map_by_pos = {raw_df.columns[i]: expected_column_names[i] for i in range(len(expected_column_names))}
                    raw_df = raw_df.rename(columns=col_map_by_pos)
                else:
                    raise ValueError(f"CSV must contain or be mappable to columns: {expected_column_names}. Found: {raw_df.columns.tolist()}")
            
            # De-duplicate columns in raw_df, keeping the first occurrence.
            # This is crucial if the renaming process (e.g. mapping both 'close'
            # and 'closeadj' to 'Close') created duplicate column names that
            # are part of expected_column_names. This ensures self.stock_data_df
            # will have the correct number of columns.
            raw_df = raw_df.loc[:, ~raw_df.columns.duplicated(keep='first')]
            self.stock_data_df = raw_df[expected_column_names].copy()

            # Convert 'Date' column, trying specific format first, then general parsing
            try:
                self.stock_data_df['Date'] = pd.to_datetime(self.stock_data_df['Date'], format='%m-%d-%y')
            except ValueError:
                try:
                    self.stock_data_df['Date'] = pd.to_datetime(self.stock_data_df['Date'])
                except Exception as e_date:
                    raise ValueError(f"Could not parse Date column. Ensure format is MM-DD-YY or YYYY-MM-DD. Error: {e_date}")

            # Convert OHLCV columns to numeric, coercing errors
            # Numeric conversion for 'Open', 'High', 'Low', 'Close', 'Volume' columns
            # is handled by the more robust loop below (starting around original line 127).
            # This earlier loop is removed to prevent TypeError if duplicate column names exist.
            
            # Drop rows where essential numeric data is missing after coercion (this was an earlier drop, let's refine)
            # First, ensure Ticker and Date are not NaN before attempting to_numeric on OHLCV
            self.stock_data_df = self.stock_data_df.dropna(subset=['Ticker', 'Date'])
            self.stock_data_df = self.stock_data_df[self.stock_data_df['Ticker'].isin(comlist)] # Filter by comlist early

            print(f"DEBUG MyDataset: Columns in self.stock_data_df after initial load & selection, before to_numeric: {self.stock_data_df.columns.tolist()}")
            print(f"DEBUG MyDataset: DataFrame shape before to_numeric: {self.stock_data_df.shape}")
            if not self.stock_data_df.empty:
                print(f"DEBUG MyDataset: DataFrame dtypes before to_numeric: \n{self.stock_data_df.dtypes}")
                # Check for duplicate columns, which can cause issues when selecting a single column
                if self.stock_data_df.columns.has_duplicates:
                    print(f"WARNING MyDataset: DataFrame has duplicate columns: {self.stock_data_df.columns[self.stock_data_df.columns.duplicated()].tolist()}")
            else:
                # This means after filtering by comlist, or initial dropna, the df is empty.
                # The error for this case will be raised later if it remains empty before numeric conversion.
                print(f"DEBUG MyDataset: DataFrame is empty after comlist filtering and Ticker/Date dropna, before numeric conversion loop.")


            # Ensure all expected columns for numeric conversion are present
            numeric_cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if self.stock_data_df.empty: # Check if empty after filtering by comlist and Ticker/Date dropna
                raise ValueError(f"No valid data remaining for comlist: {comlist} from {self.root_csv_path} (DataFrame became empty before numeric conversion loop).")

            for _col_check in numeric_cols_to_convert:
                if _col_check not in self.stock_data_df.columns:
                    # This error should ideally be caught earlier if column mapping failed
                    raise ValueError(f"CRITICAL PRE-CHECK MyDataset: Column '{_col_check}' not found in DataFrame before numeric conversion. Columns are: {self.stock_data_df.columns.tolist()}")

            # Convert OHLCV columns to numeric, coercing errors
            for col_to_convert in numeric_cols_to_convert:
                print(f"DEBUG MyDataset: Attempting pd.to_numeric for column: '{col_to_convert}'")
                if col_to_convert in self.stock_data_df:
                    # Crucially, check if selecting the column results in a Series or DataFrame (due to duplicate column names)
                    try:
                        column_data_slice = self.stock_data_df[col_to_convert]
                        print(f"DEBUG MyDataset: Type of self.stock_data_df['{col_to_convert}']: {type(column_data_slice)}")

                        if isinstance(column_data_slice, pd.Series):
                            print(f"DEBUG MyDataset: Head of Series self.stock_data_df['{col_to_convert}']: \n{column_data_slice.head()}")
                            self.stock_data_df[col_to_convert] = pd.to_numeric(column_data_slice, errors='coerce')
                        elif isinstance(column_data_slice, pd.DataFrame):
                            print(f"ERROR MyDataset: self.stock_data_df['{col_to_convert}'] is a DataFrame, likely due to duplicate column names. Columns: {column_data_slice.columns.tolist()}")
                            print(f"DEBUG MyDataset: Head of DataFrame self.stock_data_df['{col_to_convert}']: \n{column_data_slice.head()}")
                            # Attempt to convert the first column if it's a DataFrame due to duplicates
                            if not column_data_slice.empty:
                                print(f"DEBUG MyDataset: Attempting to_numeric on the first column of the DataFrame: {column_data_slice.iloc[:, 0].name}")
                                self.stock_data_df[col_to_convert] = pd.to_numeric(column_data_slice.iloc[:, 0], errors='coerce')
                            else:
                                print(f"WARNING MyDataset: DataFrame slice for '{col_to_convert}' is empty.")
                        else:
                            print(f"ERROR MyDataset: self.stock_data_df['{col_to_convert}'] is neither Series nor DataFrame. Type: {type(column_data_slice)}. Cannot convert to numeric.")
                            # Optionally raise an error here or leave as is, to be caught by dropna or later issues
                    except Exception as e_slice_convert:
                        print(f"ERROR MyDataset: Exception during slice or pd.to_numeric for column '{col_to_convert}': {e_slice_convert}")
                        # This might leave the column as is, or partially converted if error was in assignment
                else:
                    # This case should have been caught by the pre-check loop above.
                    print(f"WARNING MyDataset: Column '{col_to_convert}' was expected but not found in loop for pd.to_numeric. This is unexpected.")
            
            # Drop rows where any of the essential numeric columns became NaN after conversion
            print(f"DEBUG MyDataset: Columns before final dropna: {self.stock_data_df.columns.tolist()}")
            self.stock_data_df = self.stock_data_df.dropna(subset=numeric_cols_to_convert)

            if self.stock_data_df.empty:
                raise ValueError(f"No valid data remaining after numeric conversion and NaN drop for comlist: {comlist} from {self.root_csv_path}")

            self.stock_data_df = self.stock_data_df.set_index(['Ticker', 'Date'])
            self.stock_data_df = self.stock_data_df.sort_index()

        except FileNotFoundError:
            print(f"ERROR: Stock data CSV file not found at {self.root_csv_path}")
            raise
        except Exception as e:
            print(f"ERROR: Could not load or process stock data from {self.root_csv_path}: {e}")
            raise

        self.dates, self.next_day = self.find_dates(self.start_str, self.end_str, self.comlist)
        
        # Determine graph directory path
        graph_dir_name = f'{self.market}_{self.dataset_type}_{self.start_str}_{self.end_str}_{self.window}'
        # Ensure the graph directory path is absolute to avoid issues with changing CWD
        self.graph_directory_path = os.path.abspath(os.path.join(self.desti, graph_dir_name))

        # Calculate the theoretical maximum number of graphs based on dates
        theoretical_max_graphs = 0
        if self.dates: # Check if self.dates is not empty
            temp_actual_dates_for_calc = list(self.dates) # Make a copy
            # Condition to append next_day for calculation:
            # 1. self.next_day exists
            # 2. self.next_day is not already in the current list of dates
            # 3. The current list of dates has enough entries for at least one window
            if self.next_day and \
               self.next_day not in temp_actual_dates_for_calc and \
               len(temp_actual_dates_for_calc) >= self.window:
                temp_actual_dates_for_calc.append(self.next_day)
            
            # A graph sample requires 'window' days for features and 1 day for the label.
            # So, we need at least 'window + 1' days in temp_actual_dates_for_calc.
            if len(temp_actual_dates_for_calc) >= self.window + 1:
                # The number of possible start points for a (window + 1) slice is:
                # len(temp_actual_dates_for_calc) - (self.window + 1) + 1
                # = len(temp_actual_dates_for_calc) - self.window
                theoretical_max_graphs = len(temp_actual_dates_for_calc) - self.window
        
        # Check if all theoretically possible graph files exist
        all_theoretical_files_exist = False # Default to false
        if theoretical_max_graphs == 0:
            # If no graphs are theoretically possible, then "all" (zero) of them exist by definition,
            # or rather, no generation is needed.
            all_theoretical_files_exist = True
        elif os.path.exists(self.graph_directory_path):
            all_theoretical_files_exist = True # Assume true initially, verify below
            for i in range(theoretical_max_graphs):
                if not os.path.exists(os.path.join(self.graph_directory_path, f'graph_{i}.pt')):
                    all_theoretical_files_exist = False
                    break
        # If theoretical_max_graphs > 0 but directory doesn't exist, all_theoretical_files_exist remains False.

        # Decide whether to generate graphs
        needs_generation = False
        if theoretical_max_graphs > 0 and not all_theoretical_files_exist:
            needs_generation = True
            print(f"Graph files at {self.graph_directory_path} seem incomplete or missing for {theoretical_max_graphs} theoretical graphs. Regenerating...")
        elif theoretical_max_graphs > 0 and not os.path.exists(self.graph_directory_path): # Explicitly check if dir missing
            needs_generation = True
            print(f"Graph directory {self.graph_directory_path} missing. Generating {theoretical_max_graphs} theoretical graphs...")
        
        if needs_generation:
            self._create_graphs(self.dates, self.desti, self.comlist, self.market, self.window, self.next_day)
        elif theoretical_max_graphs > 0 : # Files exist or dir exists and files are complete
             print(f"Graph files at {self.graph_directory_path} appear to be complete for {theoretical_max_graphs} theoretical graphs. Skipping generation.")
        else: # theoretical_max_graphs is 0
            print(f"No graphs are theoretically possible (theoretical_max_graphs=0) for {self.graph_directory_path}. Skipping generation.")


        # After generation (or skipping it), determine the actual number of available graph files
        # This will be the true length of the dataset by finding contiguous files from index 0.
        self._actual_len = 0
        if os.path.exists(self.graph_directory_path):
            # Probe for graph_0.pt, graph_1.pt, ... up to theoretical_max_graphs
            # theoretical_max_graphs serves as an upper bound for probing to avoid infinite loops or excessive checks
            # if many more non-contiguous files exist.
            # If theoretical_max_graphs is 0, this loop won't run, _actual_len remains 0.
            for i in range(theoretical_max_graphs):
                if os.path.exists(os.path.join(self.graph_directory_path, f'graph_{i}.pt')):
                    self._actual_len += 1
                else:
                    # Stop at the first missing file in the sequence from 0.
                    break
        
        print(f"MyDataset initialized for {self.graph_directory_path}. Actual length determined: {self._actual_len}.")
        if self._actual_len == 0 and theoretical_max_graphs > 0:
            print(f"Warning: Actual length is 0, but {theoretical_max_graphs} graphs were theoretically possible. "
                  f"This might indicate issues in graph generation (e.g., all windows failed data checks) "
                  f"or data availability for {self.graph_directory_path}.")
        elif self._actual_len < theoretical_max_graphs:
             print(f"Warning: Actual length {self._actual_len} is less than theoretical max {theoretical_max_graphs} for {self.graph_directory_path}. "
                   f"Some graph windows might have been skipped during generation due to data issues.")

    def __len__(self):
        return self._actual_len

    def __getitem__(self, idx: int):
        # graph_directory_path is now an instance variable
        # Bound check to signal end of dataset properly
        if idx < 0 or idx >= self._actual_len:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._actual_len}")
        
        data_path = os.path.join(self.graph_directory_path, f'graph_{idx}.pt')

        file_exists = os.path.exists(data_path)
        # print(f"DEBUG MyDataset.__getitem__({idx}): Checking path '{data_path}'. Exists: {file_exists}") # Reduced verbosity

        if file_exists:
            try:
                return torch.load(data_path)
            except Exception as e:
                print(f"ERROR MyDataset.__getitem__({idx}): Failed to load '{data_path}' even though it exists. Error: {e}")
                raise FileNotFoundError(f"Error loading graph data for index {idx} at {data_path} (file existed but load failed). Original error: {e}")
        else:
            # This case should ideally be prevented by the IndexError above if idx is truly out of bounds.
            # If it's reached, it means _actual_len might be miscalculated or files are missing within the supposed range.
            print(f"DEBUG MyDataset.__getitem__({idx}): File not found at '{data_path}', but index was within _actual_len ({self._actual_len}). This indicates missing graph files that were expected to exist.")
            raise FileNotFoundError(f"No graph data found for index {idx} at {data_path} (expected based on _actual_len). Please ensure you've generated the required data correctly.")

    def check_years(self, date_str: str, start_str: str, end_str: str) -> bool:
        date_format = "%Y-%m-%d"
        date = datetime.strptime(date_str, date_format)
        start = datetime.strptime(start_str, date_format)
        end = datetime.strptime(end_str, date_format)
        return start <= date <= end

    def find_dates(self, start_str_param: str, end_str_param: str, comlist: List[str]) -> Tuple[List[str], str]:
        """
        Finds common trading dates for the given companies within the MultiIndex DataFrame.
        Uses parameters for start/end dates directly.
        """
        start_dt = pd.to_datetime(start_str_param)
        end_dt = pd.to_datetime(end_str_param)
        # Define a reasonable future limit for finding next_day
        future_limit_dt = end_dt + pd.Timedelta(days=90) # Look 90 days beyond end_str for next_day

        date_sets = []
        after_end_date_sets = []

        # Ensure all requested companies are in the DataFrame
        available_tickers = self.stock_data_df.index.get_level_values('Ticker').unique()
        valid_comlist = [ticker for ticker in comlist if ticker in available_tickers]
        
        if len(valid_comlist) < len(comlist):
            missing_tickers = set(comlist) - set(valid_comlist)
            print(f"Warning: Tickers not found in DataFrame: {missing_tickers}. Proceeding with available tickers: {valid_comlist}")
        
        if not valid_comlist:
            print(f"Warning: None of the requested tickers {comlist} found in the DataFrame. Cannot find common dates.")
            return [], None
        
        comlist_to_use = valid_comlist

        for ticker in comlist_to_use:
            try:
                # Dates for the current ticker from the DataFrame's index
                # Ensure the index level name 'Date' matches your DataFrame
                ticker_all_dates_pd = self.stock_data_df.loc[ticker].index.get_level_values('Date')
                
                # Filter dates within the [start_dt, end_dt] range
                dates_in_range = {
                    date.strftime("%Y-%m-%d") for date in ticker_all_dates_pd
                    if start_dt <= date <= end_dt
                }
                if dates_in_range:
                    date_sets.append(dates_in_range)

                # Filter dates after end_dt up to future_limit_dt for next_day calculation
                dates_after_end = {
                    date.strftime("%Y-%m-%d") for date in ticker_all_dates_pd
                    if end_dt < date <= future_limit_dt
                }
                if dates_after_end:
                    after_end_date_sets.append(dates_after_end)

            except KeyError:
                # This can happen if a ticker from comlist is not in self.stock_data_df after all
                print(f"Warning: Ticker {ticker} not found in DataFrame during date finding. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing dates for ticker {ticker}: {e}")
                # Potentially remove this ticker from comlist_to_use or handle more gracefully
                return [], None # Critical error, cannot proceed
        
        if not date_sets or len(date_sets) != len(comlist_to_use): # Ensure all valid companies contributed dates
            print(f"Warning: Could not find historical data for all companies in {comlist_to_use} between {start_str_param} and {end_str_param}.")
            # If even one company has no data in range, intersection will be empty.
            return [], None

        # Find common dates by intersection
        common_dates_str = list(set.intersection(*date_sets))
        
        next_common_day_str = None
        if after_end_date_sets and len(after_end_date_sets) == len(comlist_to_use): # Ensure all contributed for next day
            common_after_end_dates_str = list(set.intersection(*after_end_date_sets))
            if common_after_end_dates_str:
                next_common_day_str = min(common_after_end_dates_str)
        
        if not common_dates_str:
            print(f"Warning: No common trading dates found for companies {comlist_to_use} between {start_str_param} and {end_str_param}.")
            return [], None
            
        return sorted(common_dates_str), next_common_day_str

    def signal_energy(self, x_tuple: Tuple[float, ...]) -> float: # Corrected type hint
        x = np.array(x_tuple)
        return np.sum(np.square(x))

    def information_entropy(self, x_tuple: Tuple[float, ...]) -> float: # Corrected type hint
        x = np.array(x_tuple)
        if x.size == 0: return 0.0 # Handle empty array case
        unique, counts = np.unique(x, return_counts=True)
        probabilities = counts / np.sum(counts)
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        if probabilities.size == 0: return 0.0 # Handle case where all probabilities are zero (e.g. single unique value)
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy

    def adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        A = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32) # Ensure dtype
        X_np = X.numpy() # Convert once
        energy = np.array([self.signal_energy(tuple(x_row)) for x_row in X_np])
        entropy = np.array([self.information_entropy(tuple(x_row)) for x_row in X_np])
        
        for i in range(X_np.shape[0]):
            for j in range(X_np.shape[0]):
                if energy[j] == 0: # Avoid division by zero
                    A[i, j] = 0 # Or some other appropriate value like float('inf') or nan
                else:
                    value = (energy[i] / energy[j]) * (math.exp(entropy[i] - entropy[j]))
                    A[i, j] = torch.tensor(value, dtype=torch.float32)
        
        # Corrected logic for A[A<1]=1 then log(A)
        A[A < 1] = 1
        return torch.log(A)

    def node_feature_matrix(self, window_dates_str: List[str], comlist: List[str], market: str) -> torch.Tensor:
        """
        Creates the node feature matrix X from self.stock_data_df.
        X shape: (num_features, num_companies, num_days_in_window)
        Assumes 5 features: Open, High, Low, Close, Volume.
        """
        num_features = 5
        num_companies = len(comlist)
        num_days_in_window = len(window_dates_str)

        X = torch.zeros((num_features, num_companies, num_days_in_window))
        
        # Convert string dates in window_dates_str to pd.Timestamp for DataFrame indexing
        window_dates_ts = [pd.to_datetime(date_str) for date_str in window_dates_str]

        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        for company_idx, ticker in enumerate(comlist):
            for day_idx, current_ts_date in enumerate(window_dates_ts):
                try:
                    # Access data for the specific ticker and date
                    # Ensure current_ts_date is just the date part if DataFrame index has time
                    day_data = self.stock_data_df.loc[(ticker, current_ts_date.normalize())]
                    
                    # Extract the 5 features
                    # Ensure these columns exist and are numeric from __init__
                    features = day_data[feature_columns].values
                    X[:, company_idx, day_idx] = torch.tensor(features, dtype=torch.float32)

                except KeyError:
                    # Data for this ticker/date not found, X remains zeros for this entry
                    # print(f"Warning: Data not found for {ticker} on {current_ts_date.strftime('%Y-%m-%d')}. Using zeros.")
                    pass # Keep as zeros, already initialized
                except Exception as e:
                    print(f"Error extracting features for {ticker} on {current_ts_date.strftime('%Y-%m-%d')}: {e}. Using zeros.")
                    # X[:, company_idx, day_idx] remains zeros
                    pass
        return X

    def _create_graphs(self, dates: List[str], desti: str, comlist: List[str], market: str, window: int, next_day: str):
        if not dates:
            print("Warning: No dates provided to _create_graphs. Skipping graph generation.")
            return

        # Ensure next_day is appended only if it's valid and not already in dates
        # This logic might need refinement if `dates` can be empty or too short
        if next_day and next_day not in dates and len(dates) >= window :
            # Create a temporary list for graph generation if next_day is used
            dates_for_graph_gen = dates + [next_day]
        else:
            dates_for_graph_gen = dates # Use original dates if next_day is not valid or not needed for windowing

        if len(dates_for_graph_gen) <= window:
             print(f"Warning: Not enough dates ({len(dates_for_graph_gen)}) to form a window of size {window}+1. Skipping graph generation.")
             return
        
        # graph_directory_path is an instance variable, created in __init__
        os.makedirs(self.graph_directory_path, exist_ok=True)

        # The loop should go up to len(dates_for_graph_gen) - window,
        # as each iteration takes 'window + 1' days from dates_for_graph_gen
        num_possible_graphs = len(dates_for_graph_gen) - window + 1
        if num_possible_graphs <= 0:
            print(f"Not enough data points in dates_for_graph_gen ({len(dates_for_graph_gen)}) to create any graphs with window size {window}.")
            return

        for i_graph_idx in tqdm(range(num_possible_graphs), desc="Generating Graphs"):
            filename = os.path.join(self.graph_directory_path, f'graph_{i_graph_idx}.pt')

            if os.path.exists(filename):
                continue
            
            # Current window of dates: `window` days for features, `+1` day for label
            current_window_dates_str = dates_for_graph_gen[i_graph_idx : i_graph_idx + window + 1]
            
            if len(current_window_dates_str) != window + 1:
                print(f"Warning: Skipped graph {i_graph_idx}. Not enough dates for full window + label. Got {len(current_window_dates_str)}, needed {window + 1}.")
                continue

            # X_features will have shape (num_features, num_companies, window + 1)
            # The market parameter is passed but not used for data fetching here.
            X_features_full_window = self.node_feature_matrix(current_window_dates_str, comlist, market)
            
            if X_features_full_window.shape[2] != window + 1:
                 print(f"Warning: Node feature matrix for graph {i_graph_idx} has incorrect time steps: {X_features_full_window.shape[2]}, expected {window + 1}. Skipping.")
                 continue

            C_labels = torch.zeros(X_features_full_window.shape[1]) # Labels based on number of companies

            # Label generation: uses the last day (index -1) and second to last day (index -2) of the full window
            # The 'Close' price is assumed to be the 4th feature (index 3)
            # X_features_full_window[feature_idx, company_idx, time_idx]
            close_price_feature_idx = 3 # Assuming 'Close' is the 4th feature (0-indexed)
            for company_idx in range(C_labels.shape[0]):
                # Ensure we have data for both days for this company
                # Accessing -1 (last day in window) and -2 (second to last day in window)
                last_day_close = X_features_full_window[close_price_feature_idx, company_idx, -1]
                prev_day_close = X_features_full_window[close_price_feature_idx, company_idx, -2]
                
                # Check for non-zero to avoid issues with missing data filled as zero
                if last_day_close != 0 and prev_day_close != 0:
                    if last_day_close > prev_day_close:
                        C_labels[company_idx] = 1
            
            # X_processed for model input: uses data from the first `window` days
            # Shape: (num_features, num_companies, window)
            X_processed = X_features_full_window[:, :, :-1].clone()

            # Normalization on raw data (log1p) - applied to the feature part of the window
            for k_feature_idx in range(X_processed.shape[0]): # Renamed inner loop variable
                X_processed[k_feature_idx] = torch.Tensor(np.log1p(X_processed[k_feature_idx].numpy()))

            # Obtain adjacency tensor
            A_adj = torch.zeros((X_processed.shape[0], X_processed.shape[1], X_processed.shape[1])) # (num_features, num_companies, num_companies)
            for l_feature_idx in range(A_adj.shape[0]): # Renamed loop variable
                # Adjacency matrix per feature slice across companies
                A_adj[l_feature_idx] = self.adjacency_matrix(X_processed[l_feature_idx]) 
            
            torch.save({'X': X_processed, 'A': A_adj, 'Y': C_labels}, filename)

# Placeholder for MultiIndexDataset if it's defined elsewhere or needed later.
# class MultiIndexDataset(Dataset):
#     def __init__(self, ...):
#         pass
#     def __len__(self):
#         pass
#     def __getitem__(self, idx):
#         pass
