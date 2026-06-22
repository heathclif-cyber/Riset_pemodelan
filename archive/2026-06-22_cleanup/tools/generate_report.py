"""
tools/generate_report.py — Retrospective Markdown Report Generator
Generates beautiful, comprehensive holdout backtest reports from JSON/CSV results of past runs.

Jalankan:
  python tools/generate_report.py --run-id final_trend_masterpiece --name cascade_v3.1
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "experiments"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate beautiful holdout backtest reports retrospectively")
    parser.add_argument("--run-id", required=True, help="The folder name under models/runs/")
    parser.add_argument("--name", default=None, help="Display name of the model (e.g. cascade_v3.1)")
    return parser.parse_args()

def safe_float(val, default=0.0):
    try:
        if pd.isna(val) or val is None:
            return default
        return float(val)
    except:
        return default

def safe_int(val, default=0):
    try:
        if pd.isna(val) or val is None:
            return default
        return int(val)
    except:
        return default

def main():
    args = parse_args()
    run_id = args.run_id
    model_name = args.name or run_id
    
    run_dir = MODEL_DIR / "runs" / run_id
    results_json_path = run_dir / "holdout_backtest_results.json"
    
    if not results_json_path.exists():
        print(f"Error: {results_json_path} tidak ditemukan!")
        return
        
    print(f"Loading results from {results_json_path}...")
    with open(results_json_path) as f:
        aggregate = json.load(f)
        
    # Look for trade history
    trades = []
    csv_paths = [
        run_dir / "holdout_trade_history.csv",
        REPORT_DIR / f"{run_id}_holdout_trade_history.csv"
    ]
    
    trade_history_found = False
    for csv_path in csv_paths:
        if csv_path.exists():
            print(f"Loading trade history from {csv_path}...")
            try:
                df_trades = pd.read_csv(csv_path)
                trades = df_trades.to_dict('records')
                trade_history_found = True
                break
            except Exception as e:
                print(f"Error loading CSV {csv_path}: {e}")
                
    # Fallback to parsing from results JSON trades key if available
    if not trade_history_found:
        print("CSV trade history not found, parsing from JSON...")
        for symbol, r in aggregate.get("per_symbol", {}).items():
            sym_trades = r.get("trades") or r.get("trade_log") or r.get("enriched_trades") or []
            for t in sym_trades:
                # Standardize keys
                direction = t.get("direction") or t.get("Direction") or "LONG"
                net_pnl = t.get("net_pnl") or t.get("PnL ($)") or t.get("pnl") or 0.0
                pnl_pct = t.get("PnL (%)") or t.get("pnl_pct") or 0.0
                if not pnl_pct and net_pnl:
                    pnl_pct = (net_pnl / 100.0) * 100  # assuming $100 modal
                outcome = t.get("outcome") or t.get("Outcome") or "TIMEOUT"
                exit_reason = t.get("Exit Reason") or t.get("exit_reason") or outcome
                opened = t.get("Opened") or t.get("Opened_dt") or ""
                
                trades.append({
                    "Coin": symbol,
                    "Direction": direction,
                    "PnL ($)": float(net_pnl),
                    "PnL (%)": float(pnl_pct),
                    "Exit Reason": exit_reason,
                    "Opened": opened,
                    "Outcome": outcome
                })
        if trades:
            print(f"Successfully loaded {len(trades)} trades from JSON.")
            
    # Extract overall metrics
    coins = aggregate.get("coins") or list(aggregate.get("per_symbol", {}).keys())
    n_coins = len(coins)
    
    mean_winrate = safe_float(aggregate.get("mean_winrate"), 0.0)
    mean_tpm = safe_float(aggregate.get("mean_trade_per_month"), 0.0)
    mean_dd5x = safe_float(aggregate.get("mean_drawdown_lev5x"), 0.0)
    mean_sharpe = safe_float(aggregate.get("mean_sharpe"), 0.0)
    mean_sortino = safe_float(aggregate.get("mean_sortino"), 0.0)
    mean_calmar = safe_float(aggregate.get("mean_calmar"), 0.0)
    mean_pf = safe_float(aggregate.get("mean_profit_factor"), 0.0)
    max_consec_loss = safe_int(aggregate.get("max_consecutive_loss"), 0)
    holdout_period = aggregate.get("holdout_period") or "N/A"
    
    # Advanced metrics from trades list
    total_trades = len(trades)
    long_count = 0
    short_count = 0
    long_wins = 0
    short_wins = 0
    total_pnl = 0.0
    worst_trade_pnl = safe_float(aggregate.get("worst_single_trade_pnl"), 0.0)
    p95_trade_loss = safe_float(aggregate.get("p95_single_trade_loss"), 0.0)
    
    avg_win_usd = 0.0
    avg_loss_usd = 0.0
    avg_win_pct = 0.0
    avg_loss_pct = 0.0
    
    wins_usd = []
    losses_usd = []
    wins_pct = []
    losses_pct = []
    
    exit_reasons = {}
    monthly_perf = {}
    
    for t in trades:
        direction = str(t.get("Direction", "")).upper()
        pnl_usd = safe_float(t.get("PnL ($)", 0.0))
        pnl_pct = safe_float(t.get("PnL (%)", 0.0))
        opened = str(t.get("Opened", ""))
        
        # Win / Loss counts
        is_win = pnl_usd > 0
        total_pnl += pnl_usd
        
        if direction == "LONG":
            long_count += 1
            if is_win:
                long_wins += 1
        elif direction == "SHORT":
            short_count += 1
            if is_win:
                short_wins += 1
                
        if is_win:
            wins_usd.append(pnl_usd)
            wins_pct.append(pnl_pct)
        else:
            losses_usd.append(pnl_usd)
            losses_pct.append(pnl_pct)
            
        # Exit reason mapping
        exit_r = t.get("Exit Reason") or t.get("exit_reason") or t.get("Outcome") or "unknown"
        exit_r = str(exit_r).lower()
        if "tp" in exit_r or "win" in exit_r:
            exit_r = "tp_hit"
        elif "sl" in exit_r or "loss" in exit_r:
            exit_r = "sl_hit"
        elif "guardian" in exit_r:
            if "momentum" in exit_r:
                exit_r = "guardian_momentum_exit"
            else:
                exit_r = "guardian_exit"
        elif "trailing" in exit_r:
            exit_r = "trailing_stop"
        elif "time" in exit_r or "timeout" in exit_r:
            exit_r = "time_exit"
            
        if exit_r not in exit_reasons:
            exit_reasons[exit_r] = {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        exit_reasons[exit_r]["count"] += 1
        exit_reasons[exit_r]["pnl"] += pnl_usd
        if is_win:
            exit_reasons[exit_r]["wins"] += 1
        else:
            exit_reasons[exit_r]["losses"] += 1
            
        # Monthly performance mapping (Opened: "YYYY-MM-DD HH:MM")
        if opened and len(opened) >= 7:
            month_str = opened[:7]  # "YYYY-MM"
            if month_str not in monthly_perf:
                monthly_perf[month_str] = {"trades": 0, "wins": 0, "pnl": 0.0}
            monthly_perf[month_str]["trades"] += 1
            monthly_perf[month_str]["pnl"] += pnl_usd
            if is_win:
                monthly_perf[month_str]["wins"] += 1

    # Computations
    long_winrate = (long_wins / long_count) if long_count > 0 else 0.0
    short_winrate = (short_wins / short_count) if short_count > 0 else 0.0
    
    if wins_usd:
        avg_win_usd = float(np.mean(wins_usd))
        avg_win_pct = float(np.mean(wins_pct))
    if losses_usd:
        avg_loss_usd = float(np.mean(losses_usd))
        avg_loss_pct = float(np.mean(losses_pct))
        
    if not worst_trade_pnl and losses_pct:
        worst_trade_pnl = float(np.min(losses_pct))
    if not p95_trade_loss and trades:
        p95_trade_loss = float(np.percentile([t["PnL (%)"] for t in trades], 5))
        
    portfolio_roi = (total_pnl / (100.0 * n_coins)) * 100 if n_coins > 0 else 0.0
    trades_per_day = (mean_tpm * 12) / 365.25 # Portfolio trades per day (average based on monthly)
    
    # Active Features List — Try reading models/feature_cols_v2.json or cv_results
    feature_cols = aggregate.get("feature_cols")
    if not feature_cols:
        # Fallback to loading from models/feature_cols_v2.json
        feat_path = MODEL_DIR / "feature_cols_v2.json"
        if feat_path.exists():
            try:
                with open(feat_path) as f:
                    feature_cols = json.load(f)
            except:
                pass
    if not feature_cols:
        feature_cols = ["open", "high", "low", "close", "volume"] # default placeholder
        
    # Generate Markdown content
    md = []
    md.append(f"# 📊 Holdout Backtest Report: `{model_name}`")
    md.append("")
    md.append(f"**Tanggal Pembuatan**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    md.append(f"**Model Run ID**: `{run_id}`")
    md.append(f"**Periode Pengujian (Temporal OOS)**: `{holdout_period}`")
    md.append("")
    
    md.append("> [!NOTE]")
    md.append("> **Ringkasan Portofolio Eksekutif**:")
    md.append(f"> *   **Total Net Profit**: **${total_pnl:+,.2f} USD** (ROI Portofolio: **{portfolio_roi:+.2f}%**)")
    md.append(f"> *   **Rata-rata Win Rate**: **{mean_winrate:.2%}** | Total Trades: **{total_trades:,}**")
    md.append(f"> *   **Rata-rata Max Drawdown (5x)**: **{mean_dd5x:.2%}**")
    md.append(f"> *   **Risk-Adjusted**: Sharpe: **{mean_sharpe:.2f}** | Sortino: **{mean_sortino:.2f}** | Calmar: **{mean_calmar:.2f}** | Profit Factor: **{mean_pf:.2f}**")
    md.append("")
    
    # Core Metrics Table
    md.append("## 📈 Performa Scorecard Portofolio")
    md.append("")
    md.append("| Metrik Utama | Nilai Portofolio | Catatan |")
    md.append("|:---|:---:|:---|")
    md.append(f"| **Total Net Profit ($)** | `${total_pnl:+,.2f}` | Akumulasi keuntungan bersih 5x leverage |")
    md.append(f"| **Portfolio ROI (%)** | `{portfolio_roi:+.2f}%` | ROI berdasarkan kapital portofolio $100/koin |")
    md.append(f"| **Overall Win Rate** | `{mean_winrate:.2%}` | Rasio kemenangan rata-rata seluruh aset |")
    md.append(f"| **Total Trades** | `{total_trades:,}` | Jumlah total posisi yang dieksekusi |")
    md.append(f"| **Rata-rata Trade / Bulan** | `{mean_tpm:.1f}` | Rata-rata frekuensi trade bulanan portofolio |")
    md.append(f"| **Rata-rata Trade / Hari** | `{trades_per_day:.2f}` | Rata-rata frekuensi trade harian portofolio |")
    md.append(f"| **Max Drawdown (5x)** | `{mean_dd5x:.2%}` | Rata-rata penurunan terdalam portofolio |")
    md.append(f"| **Sharpe Ratio** | `{mean_sharpe:.2f}` | Efisiensi profit terhadap volatilitas portofolio |")
    md.append(f"| **Sortino Ratio** | `{mean_sortino:.2f}` | Efisiensi profit terhadap downside deviation |")
    md.append(f"| **Calmar Ratio** | `{mean_calmar:.2f}` | Rasio return tahunan terhadap drawdown |")
    md.append(f"| **Profit Factor** | `{mean_pf:.2f}` | Rasio gross profit dibagi gross loss |")
    md.append(f"| **Max Consecutive Loss** | `{max_consec_loss}` trades | Streak kekalahan beruntun terpanjang |")
    md.append(f"| **Worst Single Trade PnL** | `{worst_trade_pnl:+.2f}%` | Kerugian terdalam dalam satu trade tunggal |")
    md.append(f"| **95% Trades Loss Under** | `{abs(p95_trade_loss):.2f}%` | Nilai risiko (VaR P95) kerugian maksimal |")
    md.append("")
    
    # Direction Analysis
    md.append("## ↕️ Analisis Arah Signal (LONG vs SHORT)")
    md.append("")
    md.append("| Arah Posisi | Jumlah Trade | Distribusi | Menang | Kalah | Win Rate | PnL Bersih ($) |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    long_dist = long_count / total_trades if total_trades > 0 else 0
    short_dist = short_count / total_trades if total_trades > 0 else 0
    md.append(f"| **LONG** | {long_count:,} | {long_dist:.1%} | {long_wins:,} | {long_count - long_wins:,} | {long_winrate:.2%} | {sum(t['PnL ($)'] for t in trades if t['Direction'] == 'LONG'):+,.2f} |")
    md.append(f"| **SHORT** | {short_count:,} | {short_dist:.1%} | {short_wins:,} | {short_count - short_wins:,} | {short_winrate:.2%} | {sum(t['PnL ($)'] for t in trades if t['Direction'] == 'SHORT'):+,.2f} |")
    md.append("")
    
    # Trade PnL stats
    md.append("### Rincian Rata-rata Profitabilitas per Trade")
    md.append("")
    md.append("| Tipe Trade | PnL Rata-rata ($) | PnL Rata-rata (%) |")
    md.append("|:---|:---:|:---:|")
    md.append(f"| **Trade Menang (Wins)** | `${avg_win_usd:+,.4f}` | `{avg_win_pct:+.2f}%` |")
    md.append(f"| **Trade Kalah (Losses)** | `${avg_loss_usd:+,.4f}` | `{avg_loss_pct:+.2f}%` |")
    md.append("")
    
    # Monthly breakdown (if available)
    if monthly_perf:
        md.append("## 📅 Scorecard Bulanan Portofolio")
        md.append("")
        md.append("| Bulan | Total Trades | Wins | Losses | Win Rate | Net PnL ($) |")
        md.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for month_str in sorted(monthly_perf.keys()):
            p = monthly_perf[month_str]
            losses_count = p["trades"] - p["wins"]
            wr = p["wins"] / p["trades"] if p["trades"] > 0 else 0.0
            md.append(f"| {month_str} | {p['trades']} | {p['wins']} | {losses_count} | {wr:.2%} | ${p['pnl']:+,.2f} |")
        md.append("")
        
    # Exit Reasons breakdown
    if exit_reasons:
        md.append("## 🚪 Distribusi Alasan Keluar Posisi (Exit Reasons)")
        md.append("")
        md.append("| Alasan Exit | Jumlah | Persentase | Wins | Losses | Win Rate | PnL Bersih ($) |")
        md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for exit_r, stats in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
            pct = stats["count"] / total_trades if total_trades > 0 else 0
            wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0.0
            md.append(f"| `{exit_r}` | {stats['count']:,} | {pct:.1%} | {stats['wins']:,} | {stats['losses']:,} | {wr:.2%} | ${stats['pnl']:+,.2f} |")
        md.append("")
        
    # Per Symbol Scorecard
    md.append("## 🪙 Scorecard Per Koin (Detailed Assets)")
    md.append("")
    md.append("| Token | Win Rate | Trades | LONG WR | SHORT WR | Net PnL ($) | Max DD | Sharpe | Sortino | Calmar | PF |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    per_sym = aggregate.get("per_symbol", {})
    for sym in sorted(per_sym.keys()):
        s = per_sym[sym]
        s_wr = safe_float(s.get("winrate"), 0.0)
        s_tr = safe_int(s.get("total_trades"), 0)
        s_pnl = safe_float(s.get("pnl_lev5x"), 0.0)
        s_dd = safe_float(s.get("max_drawdown_lev5x"), 0.0)
        s_sh = safe_float(s.get("sharpe_ratio"), 0.0)
        s_so = safe_float(s.get("sortino_ratio"), 0.0)
        s_ca = safe_float(s.get("calmar_ratio"), 0.0)
        s_pf = safe_float(s.get("profit_factor"), 0.0)
        
        win_class = s.get("win_by_class", {})
        l_wr = safe_float(win_class.get("LONG"), 0.0)
        s_wr_class = safe_float(win_class.get("SHORT"), 0.0)
        
        md.append(f"| **{sym.replace('USDT','')}** | {s_wr:.2%} | {s_tr:,} | {l_wr:.1%} | {s_wr_class:.1%} | `${s_pnl:+,.2f}` | {s_dd:.2%} | {s_sh:.2f} | {s_so:.2f} | {s_ca:.2f} | {s_pf:.2f} |")
    md.append("")
    
    # Active Features Used
    md.append("## ⛓️ Daftar Fitur Aktif dalam Model")
    md.append("")
    md.append(f"Total terdapat **{len(feature_cols)} fitur aktif** yang digunakan oleh LightGBM entry, LSTM Soft Confirmation, dan Exit Guardian v3:")
    md.append("")
    md.append("<details>")
    md.append("<summary>▶ Klik untuk melihat daftar lengkap fitur aktif</summary>")
    md.append("")
    for i, col in enumerate(feature_cols, 1):
        md.append(f"{i}. `{col}`")
    md.append("")
    md.append("</details>")
    md.append("")
    
    # Save files
    md_content = "\n".join(md)
    
    # 1. Save to reports/experiments/
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_file_name = f"{model_name}_holdout_report.md"
    report_path = REPORT_DIR / report_file_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report saved to {report_path}")
    
    # 2. Save a copy to models/runs/{run_id}/
    run_report_path = run_dir / "holdout_report.md"
    with open(run_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report copy saved to {run_report_path}")
    
    print("\nReport generation completed successfully!")

if __name__ == "__main__":
    main()
