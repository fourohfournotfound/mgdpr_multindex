import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# It's generally better to pass necessary data directly (like stock_data_df, comlist)
# rather than the whole dataset instance if type hinting or avoiding circular deps is a concern.
from typing import Tuple # Added for type hinting

# from ..dataset.graph_dataset_gen import MyDataset # Example, if MyDataset type hint was strictly needed


def _calculate_daily_strategy_returns(daily_group: pd.DataFrame, num_tickers: int) -> Tuple[float, float, float]:
    """
    Calculates the strategy return for a single day's group of predictions.
    Strategy: Long top N stocks, short bottom N stocks based on prediction.

    Args:
        daily_group (pd.DataFrame): DataFrame for a single day, containing 'Prediction'
                                    and 'DailyReturn' columns for various tickers.
        num_tickers (int): Number of tickers for top-N long and bottom-N short.

    Returns:
        Tuple[float, float, float]: (avg_long_leg_return, avg_short_leg_pnl, combined_strategy_return)
                                    Returns (0.0, 0.0, 0.0) if valid positions cannot be formed.
    """
    if len(daily_group) < num_tickers * 2 and num_tickers > 0: # Need enough distinct stocks for long and short legs
        # If num_tickers is 1, need at least 2 stocks. If num_tickers is 2, need at least 4, etc.
        # However, if there are fewer than num_tickers*2 but at least num_tickers, we might still form one leg.
        # For simplicity, if we can't form both full legs, return 0 for now.
        # A more nuanced approach might allow partial legs or adjust N.
        # Let's refine: if we can't get N distinct for long AND N distinct for short, it's problematic.
        # The number of unique tickers available:
        unique_tickers_count = daily_group['Ticker'].nunique()
        if unique_tickers_count < num_tickers * 2 and num_tickers > 0 : # Stricter check for distinct tickers
             return 0.0, 0.0, 0.0
        if num_tickers == 0: # No trades
            return 0.0, 0.0, 0.0


    daily_group_sorted = daily_group.sort_values(by='Prediction', ascending=False)
    
    # Ensure we don't try to pick more tickers than available, even if unique_tickers_count was sufficient
    actual_n_pick = min(num_tickers, len(daily_group_sorted))

    top_stocks = daily_group_sorted.head(actual_n_pick)
    bottom_stocks = daily_group_sorted.tail(actual_n_pick)

    # Check for overlap if actual_n_pick * 2 > len(daily_group_sorted)
    # This can happen if num_tickers is large relative to available stocks on a given day.
    # Example: N=3, but only 4 stocks. Top 3 and Bottom 3 will overlap.
    # We need to ensure the chosen long and short portfolios are disjoint.
    # A simple way: if indices of top_stocks and bottom_stocks intersect, there's an issue.
    # For now, assume that if unique_tickers_count >= num_tickers * 2, head(N) and tail(N) will be distinct.
    # If not, the strategy is ill-defined for that N on that day.
    # A robust check:
    # Convert indices to sets to check for disjointness
    top_indices_set = set(top_stocks.index)
    bottom_indices_set = set(bottom_stocks.index)

    if not top_indices_set.isdisjoint(bottom_indices_set) and actual_n_pick > 0:
        # This means some stocks are in both top-N and bottom-N.
        # This can happen if N is large relative to the number of available stocks,
        # or if many stocks have identical predictions near the median.
        # For this strategy, we require distinct long and short portfolios.
        # If overlap, we can't form the N-N strategy cleanly.
        # Fallback: if N=1 and this happens, it means only 1 stock, handled by earlier len check.
        # If N > 1, this indicates an issue. For now, return 0.
        return 0.0, 0.0, 0.0


    long_leg_returns = top_stocks['DailyReturn'].fillna(0.0) # Fill NaNs with 0 for averaging
    short_stocks_actual_returns = bottom_stocks['DailyReturn'].fillna(0.0)

    avg_long_leg_return = long_leg_returns.mean() if not long_leg_returns.empty else 0.0
    avg_short_leg_pnl = -short_stocks_actual_returns.mean() if not short_stocks_actual_returns.empty else 0.0
    
    combined_strategy_return = avg_long_leg_return + avg_short_leg_pnl # Portfolio is long N, short N
    
    return avg_long_leg_return, avg_short_leg_pnl, combined_strategy_return


def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculates the annualized Sortino ratio for a series of daily returns.

    Args:
        daily_returns (pd.Series): A pandas Series of daily returns.
        risk_free_rate (float, optional): The daily risk-free rate of return. Defaults to 0.0.
        annualization_factor (int, optional): Number of trading periods in a year. Defaults to 252 for daily data.

    Returns:
        float: The annualized Sortino ratio.
    """
    if not isinstance(daily_returns, pd.Series):
        daily_returns = pd.Series(daily_returns)
    
    if daily_returns.empty:
        return np.nan
        
    # Calculate daily excess returns over the daily risk-free rate
    excess_returns = daily_returns - risk_free_rate
    
    # Calculate annualized mean excess return
    mean_daily_excess_return = excess_returns.mean()
    annualized_mean_excess_return = mean_daily_excess_return * annualization_factor
    
    # Calculate annualized downside deviation
    # Consider only returns below the target (risk_free_rate)
    downside_returns = excess_returns[excess_returns < 0] # Negative excess returns are downside
    
    if downside_returns.empty:
        # No downside returns observed
        if annualized_mean_excess_return > 0:
            return np.inf  # Positive returns with no observed downside risk
        elif annualized_mean_excess_return == 0:
            return 0.0 # Zero returns with no observed downside risk
        else: # annualized_mean_excess_return < 0
            return -np.inf # Negative returns but no downside volatility (implies all returns are negative but equal, or all >=0)
                           # This case is tricky; -inf suggests infinitely bad risk-adjusted return if any negative return.
                           # Or np.nan if we consider it undefined. For now, -inf.

    # Sum of squared downside returns, then mean, then sqrt, then annualize
    downside_deviation_daily = np.sqrt((downside_returns**2).mean())
    annualized_downside_deviation = downside_deviation_daily * np.sqrt(annualization_factor)
    
    if annualized_downside_deviation == 0:
        # This case should be covered by downside_returns.empty, but as a safeguard
        if annualized_mean_excess_return > 0:
            return np.inf
        elif annualized_mean_excess_return == 0:
            return 0.0
        else:
            return -np.inf # Or np.nan
            
    return annualized_mean_excess_return / annualized_downside_deviation


def plot_cumulative_returns(
    long_cumulative_returns: pd.Series,
    short_cumulative_pnl: pd.Series,
    strategy_cumulative_returns: pd.Series,
    benchmark_cumulative_returns: pd.Series,
    output_path: str
) -> None:
    """
    Plots and saves cumulative returns for the long leg, short leg, strategy, and benchmark.

    Args:
        long_cumulative_returns (pd.Series): Cumulative returns of the long leg.
        short_cumulative_pnl (pd.Series): Cumulative P&L of the short leg.
        strategy_cumulative_returns (pd.Series): Cumulative returns of the combined strategy.
        benchmark_cumulative_returns (pd.Series): Cumulative returns of the benchmark.
        output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(14, 8)) # Increased height slightly for more legend space
    long_cumulative_returns.plot(label='Long Leg Cumulative Returns', lw=1.5, linestyle=':')
    short_cumulative_pnl.plot(label='Short Leg Cumulative P&L', lw=1.5, linestyle=':')
    strategy_cumulative_returns.plot(label='Combined Strategy Cumulative Returns', lw=2, color='blue')
    benchmark_cumulative_returns.plot(label='Benchmark Cumulative Returns', lw=2, linestyle='--', color='grey')
    plt.title('Backtest: Cumulative Performance', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns (Growth of $1)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Backtest plot saved to {output_path}")


def run_backtest(
    all_predictions_list: list, 
    all_target_dates_list: list, 
    all_tickers_list: list, 
    test_dataset_instance, # Instance of MyDataset for the test set
    output_plot_path: str = "backtest_plot.png",
    risk_free_rate: float = 0.0,
    num_tickers_to_trade: int = 1
) -> dict:
    """
    Runs the backtesting process based on model predictions.

    Args:
        all_predictions_list (list): Flat list of model prediction scores.
        all_target_dates_list (list): Flat list of target dates (YYYY-MM-DD strings)
                                      corresponding to predictions.
        all_tickers_list (list): Flat list of tickers corresponding to predictions.
        test_dataset_instance: The instance of MyDataset for the test set.
                               Provides access to `stock_data_df` (with 'DailyReturn',
                               indexed by ['Ticker', 'Date']) and `comlist`.
        output_plot_path (str, optional): Path to save the cumulative returns plot.
                                          Defaults to "backtest_plot.png".
        risk_free_rate (float, optional): Risk-free rate for Sortino ratio calculation.
                                          Defaults to 0.0.
        num_tickers_to_trade (int, optional): Number of top/bottom tickers to trade. Defaults to 1.


    Returns:
        dict: A dictionary containing Sortino ratios for 'long_only', 'combined_strategy', and 'benchmark'.
              Example: {'long_only': 0.5, 'combined_strategy': 0.3, 'benchmark': 0.2}
              Returns NaNs if backtest cannot be run.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    print("Initializing backtest...")

    if not all_predictions_list or not all_target_dates_list or not all_tickers_list:
        print(f"Warning: Empty predictions, dates, or tickers list for N={num_tickers_to_trade}. Skipping backtest.")
        return {'long_only': np.nan, 'combined_strategy': np.nan, 'benchmark': np.nan}
        
    # 1. Construct DataFrame of predictions with Date and Ticker
    predictions_df = pd.DataFrame({
        'Date': pd.to_datetime(all_target_dates_list),
        'Ticker': all_tickers_list,
        'Prediction': all_predictions_list
    })
    
    if predictions_df.empty:
        print(f"Warning: Predictions DataFrame is empty after construction for N={num_tickers_to_trade}. Skipping backtest.")
        return {'long_only': np.nan, 'combined_strategy': np.nan, 'benchmark': np.nan}
    print(f"Constructed predictions_df with {len(predictions_df)} entries for N={num_tickers_to_trade}.")

    # 2. Align predictions with actual daily returns for P&L
    stock_data_df = test_dataset_instance.stock_data_df
    if not (isinstance(stock_data_df.index, pd.MultiIndex) and
            'Ticker' in stock_data_df.index.names and
            'Date' in stock_data_df.index.names):
        raise ValueError("test_dataset_instance.stock_data_df must have a MultiIndex with 'Ticker' and 'Date'.")

    stock_data_for_backtest = stock_data_df[['DailyReturn']].copy()
    
    # Ensure Date level of MultiIndex is datetime64
    date_level_values = stock_data_for_backtest.index.get_level_values('Date')
    if not pd.api.types.is_datetime64_any_dtype(date_level_values):
        stock_data_for_backtest = stock_data_for_backtest.reset_index()
        stock_data_for_backtest['Date'] = pd.to_datetime(stock_data_for_backtest['Date'])
        stock_data_for_backtest = stock_data_for_backtest.set_index(['Ticker', 'Date'])
    
    # Merge predictions with actual daily returns
    daily_predictions_with_returns = pd.merge(
        predictions_df,
        stock_data_for_backtest.reset_index(), # Reset index for merging on columns
        on=['Date', 'Ticker'],
        how='left' 
    )
    
    original_len = len(daily_predictions_with_returns)
    daily_predictions_with_returns.dropna(subset=['DailyReturn'], inplace=True)
    dropped_rows = original_len - len(daily_predictions_with_returns)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing 'DailyReturn' after merging.")

    if daily_predictions_with_returns.empty:
        print(f"Warning: No valid data after merging predictions with returns for N={num_tickers_to_trade}. Skipping backtest.")
        return {'long_only': np.nan, 'combined_strategy': np.nan, 'benchmark': np.nan}
    print(f"daily_predictions_with_returns ready with {len(daily_predictions_with_returns)} entries for N={num_tickers_to_trade}.")

    # 3. Calculate daily strategy returns (long, short P&L, combined)
    daily_predictions_with_returns = daily_predictions_with_returns.sort_values(by='Date')
    
    # Apply will return a Series of tuples. Pass num_tickers_to_trade.
    daily_returns_tuples_series = daily_predictions_with_returns.groupby('Date', group_keys=False).apply(
        lambda x: _calculate_daily_strategy_returns(x, num_tickers_to_trade)
    )

    # Unpack the tuples into separate Series
    if not daily_returns_tuples_series.empty:
        long_daily_returns_series = daily_returns_tuples_series.apply(lambda x: x[0])
        short_daily_pnl_series = daily_returns_tuples_series.apply(lambda x: x[1])
        strategy_daily_returns_series = daily_returns_tuples_series.apply(lambda x: x[2])
    else:
        # Handle case where no groups were formed (e.g., daily_predictions_with_returns was empty)
        empty_series = pd.Series(dtype=float)
        long_daily_returns_series = empty_series.copy()
        short_daily_pnl_series = empty_series.copy()
        strategy_daily_returns_series = empty_series.copy()
        
    # 4. Calculate daily benchmark returns
    unique_dates_in_strategy = strategy_daily_returns_series.index.unique()
    overall_comlist = test_dataset_instance.comlist
    
    benchmark_daily_returns_list = []
    for date_val in unique_dates_in_strategy:
        tickers_for_benchmark_on_date = [
            ticker for ticker in overall_comlist 
            if (ticker, date_val) in stock_data_for_backtest.index and 
               not pd.isna(stock_data_for_backtest.loc[(ticker, date_val), 'DailyReturn'])
        ]
        
        if not tickers_for_benchmark_on_date:
            daily_benchmark_return = 0.0
        else:
            returns_for_benchmark = stock_data_for_backtest.loc[
                (tickers_for_benchmark_on_date, date_val), 
                'DailyReturn'
            ]
            daily_benchmark_return = returns_for_benchmark.mean() if not returns_for_benchmark.empty else 0.0
        benchmark_daily_returns_list.append(daily_benchmark_return)

    benchmark_daily_returns_series = pd.Series(benchmark_daily_returns_list, index=unique_dates_in_strategy)

    benchmark_daily_returns_series = benchmark_daily_returns_series.fillna(0.0)
    long_daily_returns_series = long_daily_returns_series.fillna(0.0)
    short_daily_pnl_series = short_daily_pnl_series.fillna(0.0)
    # strategy_daily_returns_series is already filled if it came from daily_returns_tuples_series,
    # or it's an empty series that doesn't need fillna for cumprod (will result in empty).
    # However, to be safe, if it could be all NaNs from tuples:
    strategy_daily_returns_series = strategy_daily_returns_series.fillna(0.0)


    # 5. Calculate cumulative returns (growth of $1)
    long_cumulative_returns = (1 + long_daily_returns_series).cumprod()
    # For P&L series (like short_daily_pnl_series), if they represent actual P&L values and not % returns,
    # cumulative sum is more appropriate. If they are % returns, then (1+r).cumprod() is correct.
    # Assuming short_daily_pnl_series are daily % P&L for the short leg.
    short_cumulative_pnl = (1 + short_daily_pnl_series).cumprod()
    strategy_cumulative_returns = (1 + strategy_daily_returns_series).cumprod()
    benchmark_cumulative_returns = (1 + benchmark_daily_returns_series).cumprod()

    # 6. Calculate Sortino ratios
    sortino_long_only = calculate_sortino_ratio(long_daily_returns_series, risk_free_rate)
    sortino_combined = calculate_sortino_ratio(strategy_daily_returns_series, risk_free_rate)
    sortino_benchmark = calculate_sortino_ratio(benchmark_daily_returns_series, risk_free_rate)
    
    print(f"N={num_tickers_to_trade} - Sortino Ratio (Long Only): {sortino_long_only:.4f}")
    print(f"N={num_tickers_to_trade} - Sortino Ratio (Combined Strategy): {sortino_combined:.4f}")
    print(f"N={num_tickers_to_trade} - Sortino Ratio (Benchmark): {sortino_benchmark:.4f}")

    # 7. Plot cumulative returns
    plot_cumulative_returns(
        long_cumulative_returns=long_cumulative_returns,
        short_cumulative_pnl=short_cumulative_pnl,
        strategy_cumulative_returns=strategy_cumulative_returns,
        benchmark_cumulative_returns=benchmark_cumulative_returns,
        output_path=output_plot_path
    )

    warnings.filterwarnings("default", category=RuntimeWarning, message="Mean of empty slice")
    return {'long_only': sortino_long_only, 'combined_strategy': sortino_combined, 'benchmark': sortino_benchmark}