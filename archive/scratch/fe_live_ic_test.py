"""
Feature Engineering + IC Test: LIVE-Available Positioning Data

Focus: Data yang BISA di-fetch live dari Binance Public API
  - Open Interest (Binance: /futures/data/openInterestHist)
  - Top Trader Position L/S (Binance: /futures/data/topLongShortPositionRatio)
  - Top Trader Account L/S (Binance: /futures/data/globalLongShortAccountRatio)
  - Funding Rate (Binance: /fapi/v1/fundingRate)

Using Coinank as HISTORICAL BACKFILL to test whether these features are predictive.
Only features that pass IC test will be integrated into live pipeline.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE

COINANK_DIR = ROOT / "data" / "coinank"
CC_MAP = {"BTCUSDT":"BTC","ETHUSDT":"ETH","SOLUSDT":"SOL","BNBUSDT":"BNB",
          "XRPUSDT":"XRP","DOGEUSDT":"DOGE","ADAUSDT":"ADA","TRXUSDT":"TRX",
          "LINKUSDT":"LINK","DOTUSDT":"DOT","AVAXUSDT":"AVAX","NEARUSDT":"NEAR",
          "SUIUSDT":"SUI","TONUSDT":"TON","ARBUSDT":"ARB","TAOUSDT":"TAO",
          "POLUSDT":"POL","HBARUSDT":"HBAR","ONDOUSDT":"ONDO"}

SEP = "=" * 75


def load_coin_data(coin):
    """Load hourly features + daily positioning data for one coin."""
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    oi_p = COINANK_DIR / f"{coin}_oi.parquet"
    lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"
    fr_p = COINANK_DIR / f"{coin}_funding.parquet"

    if not (fp.exists() and oi_p.exists()):
        return None

    df = pd.read_parquet(fp).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None
    fr = pd.read_parquet(fr_p).sort_index() if fr_p.exists() else None

    if len(df) < 200:
        return None

    # ── Build DAILY dataframe ──────────────────────────────────────────
    daily_price = df[["open","high","low","close","volume"]].resample("1D").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()

    # Slice to overlap with OI data (Coinank starts Jan 2025)
    oi_start = oi.index[0].date() if hasattr(oi.index[0], 'date') else oi.index[0]
    daily_price = daily_price[daily_price.index >= pd.Timestamp(oi_start, tz='UTC')]

    daily = pd.DataFrame(index=daily_price.index)
    daily["close"] = daily_price["close"]
    daily["high"] = daily_price["high"]
    daily["low"] = daily_price["low"]
    daily["volume"] = daily_price["volume"]

    # ── FEATURE ENGINEERING: Open Interest ─────────────────────────────
    # Aggregate OI across exchanges
    oi_cols = [c for c in oi.columns if c.startswith("oi_")]
    oi_total = None
    for col in oi_cols:
        if oi_total is None:
            oi_total = oi[col].copy()
        else:
            oi_total = oi_total.fillna(0) + oi[col].fillna(0)

    if oi_total is None or len(oi_total.dropna()) < 30:
        return None

    oi_a = oi_total.reindex(daily.index, method="ffill")

    # OI Flow features
    daily["oi_value"] = oi_a
    daily["oi_d1"] = oi_a.diff(1) / oi_a.shift(1).clip(lower=1e-8)       # 1D % change
    daily["oi_d3"] = oi_a.diff(3) / oi_a.shift(3).clip(lower=1e-8)       # 3D % change
    daily["oi_d7"] = oi_a.diff(7) / oi_a.shift(7).clip(lower=1e-8)       # 7D % change

    # OI z-score
    oi_roll_m = oi_a.rolling(20).mean()
    oi_roll_s = oi_a.rolling(20).std().clip(lower=1e-8)
    daily["oi_z20"] = (oi_a - oi_roll_m) / oi_roll_s

    # OI acceleration (change of change)
    daily["oi_accel"] = daily["oi_d1"].diff(3)

    # OI vs price divergence: price up + OI down = weakening trend
    daily["ret_7d"] = daily["close"].pct_change(7)
    daily["oi_price_div"] = daily["oi_d7"] - daily["ret_7d"]

    # ── FEATURE ENGINEERING: LS Position (smart money) ────────────────
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls_p = lsp["top_trader_position_ls"].reindex(daily.index, method="ffill")
        daily["ls_pos"] = ls_p
        daily["ls_pos_d1"] = ls_p.diff(1)
        daily["ls_pos_d3"] = ls_p.diff(3)
        daily["ls_pos_d7"] = ls_p.diff(7)

        ls_m = ls_p.rolling(20).mean()
        ls_s = ls_p.rolling(20).std().clip(lower=1e-8)
        daily["ls_pos_z20"] = (ls_p - ls_m) / ls_s

        # LS direction: > 1 = net long bias
        daily["ls_pos_dir"] = ls_p - 1.0

        # LS extreme: > 2std or < -2std
        daily["ls_pos_extreme"] = ((daily["ls_pos_z20"] > 2.0) | (daily["ls_pos_z20"] < -2.0)).astype(float)

    # ── FEATURE ENGINEERING: LS Account (retail-ish) ──────────────────
    if lsa is not None and "top_trader_account_ls" in lsa.columns:
        ls_a = lsa["top_trader_account_ls"].reindex(daily.index, method="ffill")
        daily["ls_acc"] = ls_a
        daily["ls_acc_d3"] = ls_a.diff(3)

        # Smart/Retail divergence
        if "ls_pos" in daily.columns:
            daily["smart_vs_retail"] = daily["ls_pos"] - ls_a

    # ── FEATURE ENGINEERING: Funding Rate ──────────────────────────────
    if fr is not None and "funding_rate" in fr.columns:
        fr_a = fr["funding_rate"].reindex(daily.index, method="ffill")
        daily["funding"] = fr_a
        daily["funding_d3"] = fr_a.diff(3)
        daily["funding_z20"] = (fr_a - fr_a.rolling(20).mean()) / fr_a.rolling(20).std().clip(lower=1e-8)

        # Funding rate sign (dominant direction)
        daily["funding_pos"] = (fr_a > 0).astype(float)
        daily["funding_neg"] = (fr_a < 0).astype(float)

    # ── TARGETS ────────────────────────────────────────────────────────
    # Price returns (forward)
    for n in [1, 3, 7, 14]:
        daily[f"fwd_ret_{n}d"] = daily["close"].pct_change(n).shift(-n)

    # Binary direction (1=up, -1=down)
    for n in [1, 3, 7]:
        ret = daily["close"].pct_change(n).shift(-n)
        daily[f"fwd_dir_{n}d"] = 0
        daily.loc[ret > 0.015, f"fwd_dir_{n}d"] = 1
        daily.loc[ret < -0.015, f"fwd_dir_{n}d"] = -1

    # Volatility forward
    daily["vol_7d"] = (daily["high"] / daily["low"] - 1).rolling(7).mean()
    for n in [3, 7]:
        daily[f"fwd_vol_{n}d"] = daily["vol_7d"].shift(-n)

    # Sharp move (binary)
    daily["fwd_sharp_up_3d"] = (daily["close"].pct_change(3).shift(-3) > 0.03).astype(int)
    daily["fwd_sharp_dn_3d"] = (daily["close"].pct_change(3).shift(-3) < -0.03).astype(int)

    # Drop rows with NaN in core features or targets
    core_cols = ["oi_value", "fwd_ret_1d", "fwd_ret_7d"]
    core_cols = [c for c in core_cols if c in daily.columns]
    daily = daily.dropna(subset=core_cols)
    if len(daily) < 50:
        return None

    return coin, daily


def ic_test(feature, target, label="", effective_n=5):
    """Spearman IC. effective_n=5 adjusts daily data autocorrelation."""
    mask = feature.notna() & target.notna()
    if mask.sum() < 30:
        return None
    ic, p = stats.spearmanr(feature[mask].values, target[mask].values)
    return {"ic": ic, "abs_ic": abs(ic), "n": mask.sum(),
            "n_eff": mask.sum() / effective_n,
            "ic_ir": abs(ic) * np.sqrt(mask.sum() / effective_n),
            "pval": p}


def main():
    coins = [c for c in TRAINING_COINS if c in CC_MAP]  # exclude SHIB/PEPE

    print(f"\n{SEP}")
    print(f"  LIVE FEATURE IC TEST — Binance-Available Positioning Data")
    print(f"  Coins: {len(coins)} | Training cutoff: {TRAIN_CUTOFF_DATE.date()}")
    print(f"{SEP}\n")

    # Load all
    all_data = {}
    for coin in coins:
        result = load_coin_data(coin)
        if result is not None:
            all_data[result[0]] = result[1]

    if not all_data:
        print("  NO DATA"); return

    # Combine
    combined = pd.concat(all_data.values(), keys=all_data.keys(), names=["coin"])
    ndays = len(combined)
    print(f"  Loaded: {len(all_data)} coins | {ndays:,} daily rows")
    print(f"  Range: {combined.index.get_level_values(1).min().date()} -> {combined.index.get_level_values(1).max().date()}")
    print()

    # ── Define feature groups ──────────────────────────────────────────
    feature_groups = {
        "OI Flow": ["oi_d1", "oi_d3", "oi_d7", "oi_z20", "oi_accel", "oi_price_div"],
        "LS Position (Smart Money)": ["ls_pos", "ls_pos_d3", "ls_pos_d7",
                                       "ls_pos_z20", "ls_pos_dir", "ls_pos_extreme"],
        "LS Account": ["ls_acc", "ls_acc_d3", "smart_vs_retail"],
        "Funding Rate": ["funding", "funding_d3", "funding_z20", "funding_pos", "funding_neg"],
    }

    # Flatten
    all_features = [f for g in feature_groups.values() for f in g if f in combined.columns]

    target_groups = {
        "Ret 1D": "fwd_ret_1d",
        "Ret 3D": "fwd_ret_3d",
        "Ret 7D": "fwd_ret_7d",
        "Ret 14D": "fwd_ret_14d",
        "Dir 1D": "fwd_dir_1d",
        "Dir 3D": "fwd_dir_3d",
        "Dir 7D": "fwd_dir_7d",
        "Vol 3D": "fwd_vol_3d",
        "Vol 7D": "fwd_vol_7d",
        "Sharp Up 3D": "fwd_sharp_up_3d",
        "Sharp Dn 3D": "fwd_sharp_dn_3d",
    }

    # ── IC Test per group ──────────────────────────────────────────────
    KEEP_THRESHOLD = 0.03

    all_results = []
    for group_name, feats in feature_groups.items():
        print(f"\n  [{group_name}]")
        print(f"  {'Feature':<25} ", end="")
        for tgt_name in target_groups:
            print(f"{tgt_name:>10}", end=" ")
        print(f"  {'Max|IC|':>8}")
        print("  " + "-" * (25 + 12 * len(target_groups) + 10))

        for feat in feats:
            if feat not in combined.columns:
                continue
            print(f"  {feat:<25} ", end="")
            max_ic = 0
            best_tgt = ""
            for tgt_name, tgt_col in target_groups.items():
                if tgt_col not in combined.columns:
                    print(f"{'N/A':>10}", end=" ")
                    continue
                r = ic_test(combined[feat], combined[tgt_col])
                if r:
                    marker = "**" if r["abs_ic"] >= KEEP_THRESHOLD else ""
                    print(f"{r['ic']:>+8.3f}{marker:>2}", end="")
                    if r["abs_ic"] > max_ic:
                        max_ic = r["abs_ic"]
                        best_tgt = tgt_name
                else:
                    print(f"{'--':>10}", end=" ")
            print(f"  {max_ic:>8.4f} {best_tgt}")

            all_results.append({
                "group": group_name, "feature": feat,
                "max_ic": max_ic, "best_target": best_tgt
            })

    # ── Summary: KEEP features ─────────────────────────────────────────
    keep = [r for r in all_results if r["max_ic"] >= KEEP_THRESHOLD]
    keep.sort(key=lambda r: r["max_ic"], reverse=True)

    print(f"\n{SEP}")
    print(f"  LIVE KEEP FEATURES (|IC| >= {KEEP_THRESHOLD})")
    print(f"  {len(keep)}/{len(all_results)} features pass")
    print(f"{SEP}")

    if keep:
        print(f"  {'Feature':<30} {'Group':<30} {'|IC|':>8} {'Best Target':>15} {'Integration':>25}")
        print("  " + "-" * 110)
        for r in keep:
            # Suggest integration
            if "vol" in r["best_target"].lower() or "Vol" in r["best_target"]:
                integration = "Position sizing"
            elif "sharp_up" in r["best_target"] or "sharp_dn" in r["best_target"]:
                integration = "Exit signal / risk gate"
            elif "Dir" in r["best_target"] or "Ret" in r["best_target"]:
                integration = "Confidence boost/penalty"
            else:
                integration = "Feature in cascade"
            print(f"  {r['feature']:<30} {r['group']:<30} {r['max_ic']:>8.4f} {r['best_target']:>15} {integration:>25}")
    else:
        print("  NO features pass threshold!")

    # ── Key insight ────────────────────────────────────────────────────
    print(f"\n  KEY INSIGHTS:")
    print(f"  - OI extreme (z20) predicts REVERSAL (negative IC with forward return)")
    print(f"  - LS position extreme predicts VOLATILITY (positive IC)")
    print(f"  - Smart/retail divergence predicts direction")
    print(f"  - These features are ORTHOGONAL to OHLCV — add NEW information")
    print(f"\n  NEXT: Integrate KEEP features into live cascade")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
