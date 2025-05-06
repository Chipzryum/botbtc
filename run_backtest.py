import pandas as pd
import numpy as np
import json
from ScalpingStrategy1 import compute_indicators, generate_signals # Import from your strategy file
import os
import traceback
import gzip
import datetime as dt

# ... (calculate_metrics function remains the same) ...
def calculate_metrics(trades_df, initial_capital, equity_curve):
    """Calculates performance metrics from a list of trades and equity curve."""
    if trades_df.empty:
        return {
            "Total Trades": 0,
            "Net Profit": 0,
            "Profit Factor": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate (%)": 0,
            "Average Trade PnL": 0,
            "Max Drawdown (%)": 0,
            "Final Capital": initial_capital,
            "Cumulative Return (%)": 0,
        }

    trades_df['PnL'] = trades_df['Exit Price'] - trades_df['Entry Price']
    # Adjust PnL for short trades
    short_mask = trades_df['Direction'] == 'Short'
    trades_df.loc[short_mask, 'PnL'] = trades_df['Entry Price'] - trades_df['Exit Price']

    total_trades = len(trades_df)
    net_profit = trades_df['PnL'].sum()
    gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    winning_trades = len(trades_df[trades_df['PnL'] > 0])
    losing_trades = len(trades_df[trades_df['PnL'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_trade_pnl = net_profit / total_trades if total_trades > 0 else 0

    # Max Drawdown calculation
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0

    final_capital = initial_capital + net_profit
    cumulative_return = (net_profit / initial_capital) * 100

    metrics = {
        "Initial Capital": initial_capital,
        "Final Capital": round(final_capital, 2),
        "Net Profit": round(net_profit, 2),
        "Cumulative Return (%)": round(cumulative_return, 2),
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Average Trade PnL": round(avg_trade_pnl, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != np.inf else "inf",
        "Max Drawdown (%)": round(max_drawdown, 2),
    }
    return metrics


# ... (execute_backtest function remains largely the same, but returns df_price_signals for potential later use if needed) ...
def execute_backtest(df, initial_capital=10000, exit_on_opposite_signal=True):
    """Runs the backtest simulation."""
    print("Computing indicators...")
    df_indicators = compute_indicators(df.copy()) # Use a copy
    if df_indicators is None:
        print("Error: Not enough data to compute indicators.")
        return None, None, None

    print("Generating signals...")
    df_indicators['signal'] = generate_signals(df_indicators)

    trades = []
    position = None  # Can be 'Long', 'Short', or None
    entry_price = 0
    entry_time = None
    equity = initial_capital
    equity_curve = pd.Series(index=df_indicators.index, dtype=float)
    start_index = df_indicators.index.get_loc(df_indicators.first_valid_index()) # Start after NaNs from indicators

    print("Starting simulation loop...")
    for i in range(start_index, len(df_indicators)):
        current_time = df_indicators.index[i]
        current_price = df_indicators['close'].iloc[i]
        signal = df_indicators['signal'].iloc[i]

        equity_curve.iloc[i] = equity # Record equity before potential trade changes

        # --- Exit Logic ---
        if position == 'Long':
            exit_signal = signal == 'SELL'
            # Add other exit conditions here (e.g., stop loss, take profit)
            if exit_signal and exit_on_opposite_signal:
                pnl = current_price - entry_price
                equity += pnl
                trades.append({
                    "Entry Time": entry_time.isoformat(),
                    "Exit Time": current_time.isoformat(),
                    "Direction": position,
                    "Entry Price": entry_price,
                    "Exit Price": current_price,
                    "PnL": pnl
                })
                # print(f"{current_time}: Exit Long @ {current_price:.2f}, PnL: {pnl:.2f}") # Reduce console noise
                position = None
                entry_price = 0
                entry_time = None

        elif position == 'Short':
            exit_signal = signal == 'BUY'
            # Add other exit conditions here
            if exit_signal and exit_on_opposite_signal:
                pnl = entry_price - current_price # PnL for short
                equity += pnl
                trades.append({
                    "Entry Time": entry_time.isoformat(),
                    "Exit Time": current_time.isoformat(),
                    "Direction": position,
                    "Entry Price": entry_price,
                    "Exit Price": current_price,
                    "PnL": pnl
                })
                # print(f"{current_time}: Exit Short @ {current_price:.2f}, PnL: {pnl:.2f}") # Reduce console noise
                position = None
                entry_price = 0
                entry_time = None

        # --- Entry Logic ---
        if position is None: # Only enter if flat
            if signal == 'BUY':
                position = 'Long'
                entry_price = current_price
                entry_time = current_time
                # print(f"{current_time}: Enter Long @ {entry_price:.2f}") # Reduce console noise
            elif signal == 'SELL':
                position = 'Short'
                entry_price = current_price
                entry_time = current_time
                # print(f"{current_time}: Enter Short @ {entry_price:.2f}") # Reduce console noise

        # Update equity curve for the last point if no trade happened
        if i == len(df_indicators) - 1:
             equity_curve.iloc[i] = equity

    print("Simulation finished.")
    trades_df = pd.DataFrame(trades)
    # Ensure equity curve is filled forward for periods without trades
    equity_curve.ffill(inplace=True)
    equity_curve.fillna(initial_capital, inplace=True) # Fill initial NaNs

    # We still need df_price_signals if we want to add price chart later with downsampling
    df_results = df.loc[df_indicators.index].copy() # Align original data with indicator data
    df_results['signal'] = df_indicators['signal']
    # df_results['predicted_price'] = np.nan # Remove if not used

    return trades_df, equity_curve, df_results # Return df_results


# --- Main Execution ---
if __name__ == "__main__":
    # Use the actual filename provided in the error context
    CSV_FILE = 'btc_minute_data.csv'
    JSON_OUTPUT_FILE = 'Website/backtest_results.json' # Output for the website
    INITIAL_CAPITAL = 10000
    
    # Configuration for data reduction
    MAX_EQUITY_POINTS = 1000  # Maximum number of equity curve points to save
    MAX_TRADES = 500  # Maximum number of trades to show in detail
    DOWNSAMPLE_METHOD = 'time'  # 'time' or 'points'
    COMPRESS_OUTPUT = True  # Whether to compress the output JSON

    try:
        print(f"Loading data from {CSV_FILE}...")
        # Adjust column names as needed based on your CSV
        df_historical = pd.read_csv(CSV_FILE, index_col='timestamp', parse_dates=True)

        # Rename columns consistently
        rename_map = {
            'Timestamp': 'timestamp', # If index is not set correctly initially
            'Date': 'timestamp',      # If date column is named 'Date'
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            # Add other potential variations if needed
        }
        # Only rename columns that exist in the DataFrame
        df_historical.rename(columns={k: v for k, v in rename_map.items() if k in df_historical.columns}, inplace=True)

        # Ensure index is datetime
        if not isinstance(df_historical.index, pd.DatetimeIndex):
             if 'timestamp' in df_historical.columns:
                 df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
                 df_historical.set_index('timestamp', inplace=True)
             else:
                 raise ValueError("CSV must have a 'timestamp' column or a DatetimeIndex.")


        # Ensure required columns exist AFTER renaming
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_historical.columns]
        if missing_cols:
             raise ValueError(f"CSV is missing required columns after renaming: {missing_cols}. Available columns: {df_historical.columns.tolist()}")

        print(f"Data loaded successfully. Shape: {df_historical.shape}")
        print(f"Time range: {df_historical.index.min()} to {df_historical.index.max()}")


        trades_df, equity_curve, df_price_signals = execute_backtest(
            df_historical,
            initial_capital=INITIAL_CAPITAL
        )

        if trades_df is not None:
            print("Calculating metrics...")
            metrics = calculate_metrics(trades_df, INITIAL_CAPITAL, equity_curve)

            print("\n--- Backtest Results ---")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            
            # ---- Data Reduction Techniques ----
            print(f"\nPreparing data for visualization (reducing size)...")
            
            # 1. Downsample equity curve to reduce points
            if len(equity_curve) > MAX_EQUITY_POINTS:
                print(f"Downsampling equity curve from {len(equity_curve)} to ~{MAX_EQUITY_POINTS} points...")
                if DOWNSAMPLE_METHOD == 'time':
                    # Calculate appropriate frequency to get ~MAX_EQUITY_POINTS
                    total_seconds = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
                    freq_seconds = int(total_seconds / MAX_EQUITY_POINTS)
                    freq = f"{freq_seconds}S"
                    downsampled_equity = equity_curve.resample(freq).last().dropna()
                else:  # 'points' method
                    # Use systematic sampling to get MAX_EQUITY_POINTS
                    step = max(1, len(equity_curve) // MAX_EQUITY_POINTS)
                    downsampled_equity = equity_curve.iloc[::step]
                
                print(f"Downsampled to {len(downsampled_equity)} points")
            else:
                downsampled_equity = equity_curve
            
            # 2. Limit number of detailed trades if too many
            if len(trades_df) > MAX_TRADES:
                print(f"Limiting detailed trades from {len(trades_df)} to {MAX_TRADES}")
                # Keep most recent trades and add summary for others
                recent_trades = trades_df.iloc[-MAX_TRADES:]
                
                # Create summary stats for earlier trades
                earlier_trades = trades_df.iloc[:-MAX_TRADES]
                trades_summary = {
                    "count": len(earlier_trades),
                    "total_pnl": earlier_trades['PnL'].sum(),
                    "winning_trades": len(earlier_trades[earlier_trades['PnL'] > 0]),
                    "losing_trades": len(earlier_trades[earlier_trades['PnL'] < 0]),
                    "time_range": f"{pd.to_datetime(earlier_trades['Entry Time'].min())} - {pd.to_datetime(earlier_trades['Exit Time'].max())}"
                }
                
                # Use recent trades for details
                trades_for_json = recent_trades.to_dict(orient='records')
                # Add a note about limited trades
                metrics["Note"] = f"Showing {MAX_TRADES} of {len(trades_df)} trades. Earlier trades summarized in 'trades_summary'."
            else:
                trades_for_json = trades_df.to_dict(orient='records')
                trades_summary = None
            
            # Prepare data for JSON output with reduced size
            output_data = {
                "metrics": metrics,
                "trades": trades_for_json,
                "equity_curve": [
                    {"timestamp": idx.isoformat(), "equity": float(val)}
                    for idx, val in downsampled_equity.items() if pd.notna(val)
                ]
            }
            
            if trades_summary:
                output_data["trades_summary"] = trades_summary
            
            # Generate a timestamp for the results
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure the Website directory exists
            os.makedirs(os.path.dirname(JSON_OUTPUT_FILE), exist_ok=True)
            
            # Save the results
            file_path = JSON_OUTPUT_FILE
            if COMPRESS_OUTPUT:
                file_path = file_path.replace('.json', '.json.gz')
                print(f"Saving compressed results to {file_path}...")
                with gzip.open(file_path, 'wt') as f:
                    json.dump(output_data, f)
            else:
                print(f"Saving results to {JSON_OUTPUT_FILE}...")
                with open(JSON_OUTPUT_FILE, 'w') as f:
                    json.dump(output_data, f, indent=2)  # Reduced indent for smaller file
            
            # Create a downsampled price chart for visualization if needed
            if df_price_signals is not None and len(df_price_signals) > 0:
                chart_file = 'Website/price_chart_data.json'
                print(f"Creating downsampled price chart data at {chart_file}...")
                
                # Determine appropriate sampling frequency based on data size
                total_points = len(df_price_signals)
                target_points = 2000  # Target number of points for chart
                
                if total_points > target_points:
                    sampling_ratio = max(1, total_points // target_points)
                    # Sample every Nth point for price chart
                    sampled_df = df_price_signals.iloc[::sampling_ratio].copy()
                else:
                    sampled_df = df_price_signals
                
                # Create simplified chart data with only necessary columns
                chart_data = []
                for idx, row in sampled_df.iterrows():
                    point = {
                        "timestamp": idx.isoformat(),
                        "price": row['close'],
                        "signal": row['signal'] if 'signal' in sampled_df.columns and pd.notna(row['signal']) else None
                    }
                    chart_data.append(point)
                
                # Save downsampled chart data
                with open(chart_file, 'w') as f:
                    json.dump({"chart_data": chart_data}, f)
                
                print(f"Chart data saved with {len(chart_data)} points")
            
            print("Results saved successfully.")

        else:
            print("Backtest execution failed.")

    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Please place your historical data CSV in the root directory.")
    except ValueError as ve:
        print(f"Data Error: {ve}")
    except MemoryError:
        print("Memory error: The data is too large to process. Try using a smaller dataset or increasing available memory.")
    except Exception as e:
        print(f"An unexpected error occurred during backtest execution: {e}")
        traceback.print_exc()
