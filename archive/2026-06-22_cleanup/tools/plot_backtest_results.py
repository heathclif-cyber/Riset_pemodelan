import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent

def generate_plots(run_id: str):
    run_dir = ROOT / "models" / "runs" / run_id
    csv_path = run_dir / "holdout_trade_history.csv"
    
    if not csv_path.exists():
        # Fallback to reports
        csv_path = ROOT / "reports" / "experiments" / f"{run_id}_holdout_trade_history.csv"
        
    if not csv_path.exists():
        print(f"Error: File trade history tidak ditemukan untuk run-id: {run_id}")
        print(f"Pastikan Anda telah menjalankan backtest terlebih dahulu.")
        return

    print(f"Membaca data trade dari: {csv_path.name}")
    df = pd.read_parquet(csv_path) if csv_path.suffix == ".parquet" else pd.read_csv(csv_path)
    
    if df.empty:
        print("Data trade kosong. Tidak ada grafik yang di-generate.")
        return
        
    # Convert dates and sort
    df["Opened_dt"] = pd.to_datetime(df["Opened"])
    df = df.sort_values("Opened_dt")
    
    # 1. Calculate cumulative metrics
    df["cum_pnl_usd"] = df["PnL ($)"].cumsum()
    
    # Peak and Drawdown
    df["peak"] = df["cum_pnl_usd"].cummax()
    # Untuk portfolio drawdown, kita gunakan basis modal awal $500 (misal 5x leverage dari $100 exposure per trade, total portfolio drawdown)
    # Atau kita hitung drawdown dari peak balance USD (mengasumsikan modal awal $1000)
    initial_capital = 1000.0
    df["portfolio_value"] = initial_capital + df["cum_pnl_usd"]
    df["portfolio_peak"] = df["portfolio_value"].cummax()
    df["drawdown_pct"] = (df["portfolio_value"] - df["portfolio_peak"]) / df["portfolio_peak"] * 100
    
    # 2. Monthly Grouping
    df["Month"] = df["Opened_dt"].dt.strftime("%Y-%m")
    monthly_pnl = df.groupby("Month")["PnL ($)"].sum()
    
    # 3. Coin Grouping
    coin_pnl = df.groupby("Coin")["PnL ($)"].sum().sort_values()

    # Setup style
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Holdout Backtest Performance Scorecard - Run ID: {run_id}", fontsize=18, fontweight="bold", y=0.96)

    # Subplot 1: Equity Curve
    axes[0, 0].plot(df["Opened_dt"], df["cum_pnl_usd"], label="Cumulative Net PnL ($)", color="#2ecc71", linewidth=2.5)
    axes[0, 0].fill_between(df["Opened_dt"], df["cum_pnl_usd"], 0, alpha=0.1, color="#2ecc71")
    axes[0, 0].set_title("Portfolio Cumulative Net PnL ($)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("PnL ($)", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    for label in axes[0, 0].get_xticklabels():
        label.set_rotation(30)

    # Subplot 2: Drawdown Chart
    axes[0, 1].plot(df["Opened_dt"], df["drawdown_pct"], color="#e74c3c", linewidth=1.5, label="Drawdown %")
    axes[0, 1].fill_between(df["Opened_dt"], df["drawdown_pct"], 0, color="#e74c3c", alpha=0.2)
    axes[0, 1].set_title("Portfolio Drawdown Curve (%)", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Drawdown (%)", fontsize=11)
    axes[0, 1].set_ylim(top=0.5) # keep margin at top
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    for label in axes[0, 1].get_xticklabels():
        label.set_rotation(30)

    # Subplot 3: Monthly PnL Bar Chart
    colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in monthly_pnl.values]
    monthly_pnl.plot(kind="bar", ax=axes[1, 0], color=colors, edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Monthly Net PnL ($) Breakdown", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Month", fontsize=11)
    axes[1, 0].set_ylabel("PnL ($)", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1, 0].axhline(0, color="black", linewidth=1, linestyle="-")
    for label in axes[1, 0].get_xticklabels():
        label.set_rotation(0)

    # Subplot 4: PnL by Coin
    colors_coin = ["#3498db" if val >= 0 else "#e74c3c" for val in coin_pnl.values]
    coin_pnl.plot(kind="barh", ax=axes[1, 1], color=colors_coin, edgecolor="black", alpha=0.8)
    axes[1, 1].set_title("Net PnL ($) by Cryptocurency Token", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("PnL ($)", fontsize=11)
    axes[1, 1].set_ylabel("Token", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5, axis="x")
    axes[1, 1].axvline(0, color="black", linewidth=1, linestyle="-")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    # Save chart
    img_run_path = run_dir / "holdout_charts.png"
    img_report_path = ROOT / "reports" / "experiments" / f"{run_id}_holdout_charts.png"
    
    plt.savefig(img_run_path, dpi=150)
    plt.savefig(img_report_path, dpi=150)
    plt.close()
    
    print(f"\n[OK] Grafik sukses dibuat:")
    print(f"  - {img_run_path}")
    print(f"  - {img_report_path}")

    # 4. Append image link to markdown report
    md_path = ROOT / "reports" / "experiments" / f"{run_id}_holdout_report.md"
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check if already has charts image
        if "holdout_charts.png" not in content:
            # Insert after the title header
            title_end = content.find("\n", content.find("#"))
            image_markdown = f"\n\n![Performance Scorecard Summary Charts]({img_report_path.as_uri()})\n"
            new_content = content[:title_end] + image_markdown + content[title_end:]
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(new_content)
                
            # Also update run_dir copy
            run_md_path = run_dir / "holdout_report.md"
            if run_md_path.exists():
                with open(run_md_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            print(f"[OK] Link grafik telah disematkan ke dalam {md_path.name}")

    # Print scorecard statistics
    win_trades = df[df["PnL ($)"] > 0]
    loss_trades = df[df["PnL ($)"] <= 0]
    total_trades = len(df)
    winrate = len(win_trades) / total_trades if total_trades > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"📊 SUMMARY SCORECARD - RUN ID: {run_id}")
    print("="*50)
    print(f"Total Trades         : {total_trades}")
    print(f"Win Rate             : {winrate:.2%}")
    print(f"Total Net profit ($) : ${df['PnL ($)'].sum():+,.2f}")
    print(f"Max Drawdown (%)     : {df['drawdown_pct'].min():.2f}%")
    print(f"Average Win Trade ($): ${win_trades['PnL ($)'].mean():+,.2f}")
    print(f"Average Loss Trade ($): ${loss_trades['PnL ($)'].mean():+,.2f}")
    print(f"Profit Factor        : {win_trades['PnL ($)'].sum() / abs(loss_trades['PnL ($)'].sum()):.2f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run ID of the backtest to plot")
    args = parser.parse_args()
    generate_plots(args.run_id)
