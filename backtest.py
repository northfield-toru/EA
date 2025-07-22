import csv
import math
from datetime import datetime, timedelta

import numpy as np
from tensorflow.keras.models import load_model

# Load configuration
import json
with open('config.json', 'r') as f:
    config = json.load(f)

tp_pips = config.get("tp_pips")
sl_pips = config.get("sl_pips")
confidence_threshold = config.get("confidence_threshold")
tick_data_path = config.get("tick_data_path")
model_path = config.get("model_path")
output_log_path = config.get("output_log_path", "trades_log.csv")

# Define pip value for USD/JPY (0.01 = 1 pip if quotes have two decimal places)
pip_value = config.get("pip_value", 0.01)

# Load the pre-trained classification model
model = load_model(model_path)

# Prepare to read tick data
ticks = []  # will hold tuples of (timestamp, bid, ask)
with open(tick_data_path, 'r') as f:
    header = f.readline().strip().split()  # read header
    # Determine if file is tab-separated (assuming header has "DATE", "TIME", etc.)
    delimiter = '\t' if '\t' in f.readline() else ',' 
    f.seek(0)
    reader = csv.DictReader(f, delimiter=delimiter)
    for row in reader:
        date_str = row['DATE'].strip()
        time_str = row['TIME'].strip()
        # Combine date and time into a datetime object
        # Assuming format like YYYY-MM-DD for date and HH:MM:SS (possibly with milliseconds) for time
        timestamp = datetime.fromisoformat(date_str + ' ' + time_str)
        bid = float(row['BID'])
        ask = float(row['ASK'])
        ticks.append((timestamp, bid, ask))

# Initialize lists for open trades and completed trades
open_trades = []
closed_trades = []  # will store dicts with trade results

# Helper function to prepare model input (this will depend on model's expected input shape)
def prepare_features(index):
    """
    Prepare input features for the model based on historical data up to given index.
    This is a placeholder; actual implementation depends on the model's training.
    """
    # Example: using the last N mid-price changes or other technical indicators
    # For simplicity, we'll just use the current tick's bid/ask as features (not realistic for real model)
    _, bid, ask = ticks[index]
    # Features could be: [bid, ask] or [mid, spread], etc.
    mid = (bid + ask) / 2.0
    spread = ask - bid
    return np.array([[mid, spread]], dtype=float)

# Assume class indices: 0->BUY, 1->SELL, 2->NO_TRADE (adjust if model uses different ordering)
BUY_INDEX, SELL_INDEX, NO_TRADE_INDEX = 0, 1, 2

# Iterate through each tick in the data
for i, (time, bid, ask) in enumerate(ticks):
    # First, check if any open trades hit TP or SL at this tick
    current_mid = (bid + ask) / 2.0
    for trade in open_trades[:]:  # iterate over a copy of the list
        if i <= trade['entry_index']:
            # Skip checking exit on the same tick as entry to avoid immediate exit
            continue
        if trade['direction'] == 'BUY':
            # For a BUY trade, check if mid crosses TP or SL
            if current_mid >= trade['target_mid']:
                # Take-Profit hit
                exit_price = bid  # closing a long at Bid
                profit_pips = (exit_price - trade['entry_price']) / pip_value
                closed_trades.append({
                    'entry_time': trade['entry_time'],
                    'direction': 'BUY',
                    'exit_time': time,
                    'exit_reason': 'TP',
                    'pips': round(profit_pips, 2),
                    'duration': time - trade['entry_time']
                })
                open_trades.remove(trade)
            elif current_mid <= trade['stop_mid']:
                # Stop-Loss hit
                exit_price = bid  # closing a long at Bid
                profit_pips = (exit_price - trade['entry_price']) / pip_value
                closed_trades.append({
                    'entry_time': trade['entry_time'],
                    'direction': 'BUY',
                    'exit_time': time,
                    'exit_reason': 'SL',
                    'pips': round(profit_pips, 2),
                    'duration': time - trade['entry_time']
                })
                open_trades.remove(trade)
        elif trade['direction'] == 'SELL':
            # For a SELL trade, check if mid crosses TP or SL
            if current_mid <= trade['target_mid']:
                # Take-Profit hit for short
                exit_price = ask  # closing a short at Ask
                profit_pips = (trade['entry_price'] - exit_price) / pip_value
                closed_trades.append({
                    'entry_time': trade['entry_time'],
                    'direction': 'SELL',
                    'exit_time': time,
                    'exit_reason': 'TP',
                    'pips': round(profit_pips, 2),
                    'duration': time - trade['entry_time']
                })
                open_trades.remove(trade)
            elif current_mid >= trade['stop_mid']:
                # Stop-Loss hit for short
                exit_price = ask  # closing a short at Ask
                profit_pips = (trade['entry_price'] - exit_price) / pip_value
                closed_trades.append({
                    'entry_time': trade['entry_time'],
                    'direction': 'SELL',
                    'exit_time': time,
                    'exit_reason': 'SL',
                    'pips': round(profit_pips, 2),
                    'duration': time - trade['entry_time']
                })
                open_trades.remove(trade)
    # Next, get model prediction for this tick to potentially open new trade
    features = prepare_features(i)
    prediction = model.predict(features)
    # Assume prediction is a probability distribution [p_buy, p_sell, p_no_trade]
    pred_prob = prediction[0]  # first (only) sample in batch
    class_idx = int(np.argmax(pred_prob))
    confidence = float(pred_prob[class_idx])
    if confidence < confidence_threshold:
        continue  # skip if not confident enough
    # Determine trade direction from class index
    if class_idx == BUY_INDEX:
        # Open a BUY trade
        entry_price = ask  # buy at ask
        entry_mid = (bid + ask) / 2.0
        target_mid = entry_mid + tp_pips * pip_value
        stop_mid = entry_mid - sl_pips * pip_value
        open_trades.append({
            'direction': 'BUY',
            'entry_time': time,
            'entry_price': entry_price,
            'target_mid': target_mid,
            'stop_mid': stop_mid,
            'entry_index': i
        })
    elif class_idx == SELL_INDEX:
        # Open a SELL trade
        entry_price = bid  # sell at bid
        entry_mid = (bid + ask) / 2.0
        target_mid = entry_mid - tp_pips * pip_value
        stop_mid = entry_mid + sl_pips * pip_value
        open_trades.append({
            'direction': 'SELL',
            'entry_time': time,
            'entry_price': entry_price,
            'target_mid': target_mid,
            'stop_mid': stop_mid,
            'entry_index': i
        })
    else:
        # class_idx == NO_TRADE_INDEX or unrecognized – do nothing
        continue

# After iterating through all ticks, if any trades remain open (e.g., TP/SL not hit by end of data),
# close them at the final tick price to finalize the backtest.
if open_trades:
    final_time, final_bid, final_ask = ticks[-1]
    final_mid = (final_bid + final_ask) / 2.0
    for trade in open_trades:
        if trade['direction'] == 'BUY':
            # Close remaining BUY at final bid
            exit_price = final_bid
            profit_pips = (exit_price - trade['entry_price']) / pip_value
            exit_reason = 'SL'  # Could not hit TP by end of data, treat as stop-out
        else:  # SELL
            exit_price = final_ask
            profit_pips = (trade['entry_price'] - exit_price) / pip_value
            exit_reason = 'SL'
        closed_trades.append({
            'entry_time': trade['entry_time'],
            'direction': trade['direction'],
            'exit_time': final_time,
            'exit_reason': exit_reason,
            'pips': round(profit_pips, 2),
            'duration': final_time - trade['entry_time']
        })
    open_trades.clear()

# Write trade logs to CSV
with open(output_log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["EntryTime", "Direction", "ExitTime", "ExitReason", "Pips", "Duration"])
    for trade in closed_trades:
        # Format duration in seconds (or as timedelta string)
        duration_seconds = trade['duration'].total_seconds()
        writer.writerow([
            trade['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
            trade['direction'],
            trade['exit_time'].strftime("%Y-%m-%d %H:%M:%S"),
            trade['exit_reason'],
            f"{trade['pips']:.2f}",
            f"{duration_seconds:.2f}s"
        ])

# Calculate performance metrics
num_trades = len(closed_trades)
if num_trades == 0:
    print("No trades were executed.")
    exit()

wins = sum(1 for t in closed_trades if t['pips'] > 0)
losses = num_trades - wins
win_rate = (wins / num_trades) * 100.0

total_profit = sum(t['pips'] for t in closed_trades)
average_profit = total_profit / num_trades

# Calculate max drawdown
cumulative = 0.0
max_equity = 0.0
max_drawdown = 0.0
for t in closed_trades:
    cumulative += t['pips']
    if cumulative > max_equity:
        max_equity = cumulative
    drawdown = max_equity - cumulative
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# Calculate profit factor (gross profit / gross loss)
gross_profit = sum(t['pips'] for t in closed_trades if t['pips'] > 0)
gross_loss = sum(t['pips'] for t in closed_trades if t['pips'] < 0)
if gross_loss == 0:
    profit_factor = float('inf')
else:
    profit_factor = gross_profit / abs(gross_loss)

# Print summary
print("Backtest Summary:")
print(f"Number of trades: {num_trades}")
print(f"Win rate: {win_rate:.2f}%")
print(f"Average profit per trade: {average_profit:.2f} pips")
print(f"Total profit: {total_profit:.2f} pips")
print(f"Max drawdown: {max_drawdown:.2f} pips")
if math.isinf(profit_factor):
    print("Profit factor: infinity (no losing trades)")
else:
    print(f"Profit factor: {profit_factor:.2f}")
