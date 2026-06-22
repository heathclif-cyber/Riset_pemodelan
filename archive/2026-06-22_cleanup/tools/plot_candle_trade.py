import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT = Path(__file__).parent.parent

def plot_trade(trade_row, run_id):
    coin = trade_row["Coin"]
    opened_str = trade_row["Opened"]
    closed_str = trade_row["Closed"]
    direction = trade_row["Direction"]
    entry = float(trade_row["Entry"])
    exit_p = float(trade_row["Exit"])
    tp = float(trade_row["TP"])
    sl = float(trade_row["SL"])
    pnl_pct = float(trade_row["PnL (%)"])
    exit_reason = trade_row["Exit Reason"]
    
    # Load clean data
    clean_path = ROOT / "data" / "holdout-test" / "processed" / f"{coin}_clean.parquet"
    if not clean_path.exists():
        print(f"Error: clean parquet not found at {clean_path}")
        return None
        
    df = pd.read_parquet(clean_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Find timestamps
    opened_dt = pd.to_datetime(opened_str, utc=True)
    closed_dt = pd.to_datetime(closed_str, utc=True)
    
    # Slice data with padding (e.g. 10 hours before entry, 10 hours after exit)
    start_pad = opened_dt - pd.Timedelta(hours=10)
    end_pad = closed_dt + pd.Timedelta(hours=10)
    
    trade_df = df[(df.index >= start_pad) & (df.index <= end_pad)].copy()
    if trade_df.empty:
        print(f"Error: sliced data is empty for {coin}")
        return None
        
    # Plot candlestick chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors
    up_color = "#2ecc71"
    down_color = "#e74c3c"
    
    # Convert index to numeric for plotting widths
    trade_df["time_num"] = range(len(trade_df))
    
    for idx, row in trade_df.iterrows():
        t = row["time_num"]
        o = row["1h_open"] if "1h_open" in row else row["open"]
        h = row["1h_high"] if "1h_high" in row else row["high"]
        l = row["1h_low"] if "1h_low" in row else row["low"]
        c_val = row["1h_close"] if "1h_close" in row else row["close"]
        
        # Draw wicks
        ax.plot([t, t], [l, h], color="#2c3e50", linewidth=1.2, zorder=1)
        
        # Draw body
        if c_val >= o:
            body = patches.Rectangle((t - 0.3, o), 0.6, max(c_val - o, 0.00001), facecolor=up_color, edgecolor="#27ae60", linewidth=0.8, zorder=2)
        else:
            body = patches.Rectangle((t - 0.3, c_val), 0.6, max(o - c_val, 0.00001), facecolor=down_color, edgecolor="#c0392b", linewidth=0.8, zorder=2)
        ax.add_patch(body)
        
    # Align labels
    step = max(1, len(trade_df) // 12)
    ticks = list(range(0, len(trade_df), step))
    if len(trade_df) - 1 not in ticks:
        ticks.append(len(trade_df) - 1)
    tick_labels = [trade_df.index[t].strftime("%m-%d %H:%M") for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=9)
    
    # Find entry and exit indices
    entry_idx = np.argmin(np.abs((trade_df.index - opened_dt).total_seconds()))
    exit_idx = np.argmin(np.abs((trade_df.index - closed_dt).total_seconds()))
    
    # Shade trade duration
    ax.axvspan(entry_idx, exit_idx, color="#3498db", alpha=0.08, label="Trade Duration")
    
    # Plot TP, SL, and Entry lines
    ax.axhline(tp, color="#27ae60", linestyle="--", linewidth=1.2, label=f"TP Target ({tp:.4f})")
    ax.axhline(sl, color="#c0392b", linestyle="--", linewidth=1.2, label=f"SL Target ({sl:.4f})")
    ax.axhline(entry, color="#d35400", linestyle=":", linewidth=1.2, label=f"Entry Level ({entry:.4f})")
    
    # Annotate Entry
    high_col = "1h_high" if "1h_high" in trade_df.columns else "high"
    low_col = "1h_low" if "1h_low" in trade_df.columns else "low"
    y_offset = (trade_df[high_col].max() - trade_df[low_col].min()) * 0.03
    if direction == "LONG":
        ax.annotate("LONG ENTRY", xy=(entry_idx, entry), xytext=(entry_idx - 4, entry - y_offset),
                    arrowprops=dict(facecolor="#2ecc71", edgecolor="#27ae60", shrink=0.05, width=2, headwidth=8),
                    fontsize=10, fontweight="bold", color="#27ae60", ha="center")
    else:
        ax.annotate("SHORT ENTRY", xy=(entry_idx, entry), xytext=(entry_idx - 4, entry + y_offset),
                    arrowprops=dict(facecolor="#e74c3c", edgecolor="#c0392b", shrink=0.05, width=2, headwidth=8),
                    fontsize=10, fontweight="bold", color="#c0392b", ha="center")
                    
    # Annotate Exit
    ax.annotate(f"EXIT ({exit_reason})\nPrice: {exit_p:.4f}\nPnL: {pnl_pct:+.2f}%", xy=(exit_idx, exit_p), 
                xytext=(exit_idx + 4, exit_p + y_offset if exit_p < entry else exit_p - y_offset),
                arrowprops=dict(facecolor="#3498db", edgecolor="#2980b9", shrink=0.05, width=2, headwidth=8),
                fontsize=10, fontweight="bold", color="#2980b9", ha="center")
                
    # Style chart
    ax.set_title(f"Trade Candlestick Detail: {coin} {direction} ({opened_str} to {closed_str})\n"
                 f"Net PnL (5x Lev): {pnl_pct*5:+.2f}% | Exit Reason: {exit_reason} | Run: {run_id}", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Price (USDT)", fontsize=11)
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    
    # Save chart
    img_dir = ROOT / "reports" / "experiments"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"{run_id}_candle_{coin}_{opened_dt.strftime('%m%d_%H%M')}.png"
    img_path = img_dir / img_name
    plt.savefig(img_path, dpi=150)
    plt.close()
    
    print(f"[OK] Candlestick chart saved to: {img_path}")
    return img_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--coin", default=None)
    args = parser.parse_args()
    
    run_dir = ROOT / "models" / "runs" / args.run_id
    csv_path = run_dir / "holdout_trade_history.csv"
    if not csv_path.exists():
        csv_path = ROOT / "reports" / "experiments" / f"{args.run_id}_holdout_trade_history.csv"
        
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: Trade history is empty.")
        sys.exit(1)
        
    # Convert dates
    df["Opened_dt"] = pd.to_datetime(df["Opened"], utc=True)
    
    # Filter for wins
    wins_df = df[df["PnL (%)"] > 0]
    if wins_df.empty:
        print("No winning trades found to plot.")
        sys.exit(0)
        
    # Find best LONG and best SHORT trades
    best_long = wins_df[wins_df["Direction"] == "LONG"].sort_values("PnL (%)", ascending=False)
    best_short = wins_df[wins_df["Direction"] == "SHORT"].sort_values("PnL (%)", ascending=False)
    
    plotted = []
    
    if args.coin:
        coin_trades = df[df["Coin"] == args.coin].sort_values("PnL (%)", ascending=False)
        if not coin_trades.empty:
            trade_row = coin_trades.iloc[0]
            print(f"Plotting best trade for {args.coin}...")
            path = plot_trade(trade_row, args.run_id)
            if path: plotted.append(path)
    else:
        # Plot best LONG
        if not best_long.empty:
            trade_row = best_long.iloc[0]
            print(f"Plotting best LONG trade: {trade_row['Coin']} PnL: {trade_row['PnL (%)']}%...")
            path = plot_trade(trade_row, args.run_id)
            if path: plotted.append(path)
            
        # Plot best SHORT
        if not best_short.empty:
            trade_row = best_short.iloc[0]
            print(f"Plotting best SHORT trade: {trade_row['Coin']} PnL: {trade_row['PnL (%)']}%...")
            path = plot_trade(trade_row, args.run_id)
            if path: plotted.append(path)
            
    if plotted:
        print(f"\nSuccessfully generated {len(plotted)} candle chart(s).")
    else:
        print("No charts generated.")

if __name__ == "__main__":
    main()
