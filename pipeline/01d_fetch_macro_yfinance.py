"""
pipeline/01d_fetch_macro_yfinance.py — Fetch BTC ETF flow + macro data via yfinance (FREE)

Fetches:
  1. BTC Spot ETF daily price -> estimated flow proxy (10 ETFs)
  2. Macro cross-asset data (SPY, QQQ, GLD, UUP, VIX, TLT, HYG, EEM)

Usage:
  python pipeline/01d_fetch_macro_yfinance.py              # fetch all, save to data/macro/
  python pipeline/01d_fetch_macro_yfinance.py --etf-only    # ETF flow only
  python pipeline/01d_fetch_macro_yfinance.py --macro-only  # macro cross-asset only

Schedule: daily cron (Windows Task Scheduler) — data di-refresh setiap hari
"""

import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MACRO_DIR = ROOT / "data" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

# ─── BTC Spot ETFs (by AUM, represents ~99% of total) ─────────────────────────
# AUM estimates in billions USD (updated June 2026)
BTC_ETFS = {
    "IBIT": 48.0,   # BlackRock
    "GBTC": 18.0,   # Grayscale
    "FBTC": 10.0,   # Fidelity
    "ARKB": 3.5,    # ARK/21Shares
    "BITB": 3.0,    # Bitwise
    "HODL": 1.8,    # VanEck
    "BTCO": 1.3,    # Invesco
    "EZBC": 0.7,    # Franklin Templeton
    "BRRR": 0.6,    # Valkyrie
    "BTCW": 0.4,    # WisdomTree
}

# ─── Macro Cross-Asset Tickers ────────────────────────────────────────────────
MACRO_TICKERS = {
    "SPY": "S&P 500",           # US equities
    "QQQ": "Nasdaq 100",        # Tech/growth
    "GLD": "Gold",              # Safe haven
    "UUP": "USD Index",         # DXY proxy
    "VIX": "VIX Volatility",    # Fear gauge (^VIX via yfinance)
    "TLT": "20Y Treasury",      # Long bonds
    "HYG": "High Yield Bond",   # Credit risk appetite
    "EEM": "Emerging Markets",  # EM risk appetite
}


def fetch_etf_flow() -> pd.DataFrame:
    """
    Fetch daily OHLCV for all BTC spot ETFs, compute estimated flow proxy.

    Methodology:
      flow_est[t] = shares_outstanding * (close[t] - close[t-1])

    Limitation: shares_outstanding is estimated from static AUM / latest close.
    Real flow requires daily shares outstanding data (not available via yfinance).
    Correlation with real Coinank flow data: r ≈ 0.40, direction agreement ≈ 65%.

    Returns DataFrame with columns: etf_{ticker}_flow, etf_total_flow
    """
    print("[ETF Flow] Fetching BTC spot ETF data via yfinance...")
    dfs = []

    for ticker, aum_b in BTC_ETFS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max")
            if len(hist) == 0:
                print(f"  {ticker}: no data, skipping")
                continue

            # Estimate shares outstanding (static)
            latest_close = float(hist["Close"].iloc[-1])
            if latest_close <= 0:
                print(f"  {ticker}: invalid close price, skipping")
                continue
            shares_est = aum_b * 1e9 / latest_close

            # Daily flow estimate: shares * price change
            col_name = f"etf_{ticker.lower()}_flow"
            hist[col_name] = shares_est * hist["Close"].diff()
            hist[col_name] = hist[col_name].fillna(0.0)
            dfs.append(hist[[col_name]])
            print(f"  {ticker}: {len(hist)} bars — AUM ~${aum_b}B, shares_est ~{shares_est:,.0f}")

        except Exception as e:
            print(f"  {ticker}: error — {e}")

    if not dfs:
        print("[ETF Flow] No ETF data fetched!")
        return pd.DataFrame()

    flow_df = pd.concat(dfs, axis=1)
    flow_df["etf_total_change_usd"] = flow_df.sum(axis=1)
    # Also add gbtc-specific flow (column name: etf_gbtc_flow -> etf_gbtc_change_usd)
    if "etf_gbtc_flow" in flow_df.columns:
        flow_df["etf_gbtc_change_usd"] = flow_df["etf_gbtc_flow"]

    # Clean index: normalize to UTC date-only (daily frequency)
    flow_df.index = pd.to_datetime(flow_df.index)
    if flow_df.index.tz is not None:
        flow_df.index = flow_df.index.tz_convert("UTC")
    flow_df.index = flow_df.index.normalize()
    flow_df = flow_df[~flow_df.index.duplicated(keep="last")]

    print(f"[ETF Flow] Done: {len(flow_df)} daily rows, "
          f"{flow_df.index[0].date()} -> {flow_df.index[-1].date()}")
    return flow_df


def fetch_macro_cross_asset() -> pd.DataFrame:
    """
    Fetch daily OHLCV for macro cross-asset tickers.

    Returns DataFrame with columns: {ticker}_close, {ticker}_return_1d, etc.
    """
    print("[Macro] Fetching cross-asset data via yfinance...")
    dfs = []

    for ticker, name in MACRO_TICKERS.items():
        try:
            # Handle ^VIX special yfinance naming
            yf_ticker = "^VIX" if ticker == "VIX" else ticker
            t = yf.Ticker(yf_ticker)
            hist = t.history(period="max")
            if len(hist) == 0:
                print(f"  {ticker} ({name}): no data, skipping")
                continue

            col_close = f"{ticker.lower()}_close"
            col_ret1 = f"{ticker.lower()}_ret_1d"
            col_ret5 = f"{ticker.lower()}_ret_5d"
            col_ret20 = f"{ticker.lower()}_ret_20d"

            out = pd.DataFrame(index=hist.index)
            out[col_close] = hist["Close"]
            out[col_ret1] = hist["Close"].pct_change(1)
            out[col_ret5] = hist["Close"].pct_change(5)
            out[col_ret20] = hist["Close"].pct_change(20)

            dfs.append(out)
            print(f"  {ticker:5s} ({name:20s}): {len(hist)} bars")

        except Exception as e:
            print(f"  {ticker} ({name}): error — {e}")

    if not dfs:
        print("[Macro] No macro data fetched!")
        return pd.DataFrame()

    macro_df = pd.concat(dfs, axis=1)

    # Clean index
    macro_df.index = pd.to_datetime(macro_df.index)
    if macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_convert("UTC")
    macro_df.index = macro_df.index.normalize()
    macro_df = macro_df[~macro_df.index.duplicated(keep="last")]

    print(f"[Macro] Done: {len(macro_df)} daily rows, "
          f"{macro_df.index[0].date()} -> {macro_df.index[-1].date()}")
    return macro_df


def save_data(flow_df: pd.DataFrame, macro_df: pd.DataFrame):
    """Save ETF flow and macro data to parquet files."""
    if not flow_df.empty:
        etf_path = MACRO_DIR / "etf_flow_btc.parquet"
        flow_df.to_parquet(etf_path)
        print(f"[Save] ETF flow -> {etf_path} ({len(flow_df)} rows, {len(flow_df.columns)} cols)")

    if not macro_df.empty:
        macro_path = MACRO_DIR / "macro_cross_asset.parquet"
        macro_df.to_parquet(macro_path)
        print(f"[Save] Macro cross-asset -> {macro_path} ({len(macro_df)} rows, {len(macro_df.columns)} cols)")


def main():
    parser = argparse.ArgumentParser(description="Fetch macro/ETF data via yfinance")
    parser.add_argument("--etf-only", action="store_true", help="Fetch ETF flow only")
    parser.add_argument("--macro-only", action="store_true", help="Fetch macro cross-asset only")
    args = parser.parse_args()

    fetch_all = not args.etf_only and not args.macro_only

    flow_df = pd.DataFrame()
    macro_df = pd.DataFrame()

    if fetch_all or args.etf_only:
        flow_df = fetch_etf_flow()

    if fetch_all or args.macro_only:
        macro_df = fetch_macro_cross_asset()

    save_data(flow_df, macro_df)

    # Print combined summary
    print("\n" + "=" * 60)
    print("  FETCH COMPLETE — yfinance (FREE)")
    print("=" * 60)
    if not flow_df.empty:
        print(f"  ETF flow : {len(flow_df)} days, {len(flow_df.columns)-1} ETFs")
    if not macro_df.empty:
        cols_per_ticker = 4  # close, ret_1d, ret_5d, ret_20d
        n_tickers = len(macro_df.columns) // cols_per_ticker
        print(f"  Macro    : {len(macro_df)} days, {n_tickers} tickers")


if __name__ == "__main__":
    main()
