"""
scratch/ic_test_positioning_vs_win.py
IC test: apakah Binance Vision positioning features prediktif terhadap ic32 WIN/LOSS?

Challenge dari reviewer: IC test harus domain-matched.
Target: binary WIN/LOSS dari ic32 trade outcomes (bukan generic H1 return).
Source: data/meta_labels/ic32_oof_trades.parquet (dihasilkan oleh 08_generate_ic32_oof_trades.py)

Feature engineering (sesuai Challenge 3):
  - oi_usd_delta_pct: sudah percent change, tambah z-score rolling
  - taker_ls_vol_ratio: ratio sudah aman, tambah z-score rolling 20-bar
  - toptrader_ls_ratio: sama, z-score rolling
  - cross feature: oi_usd_delta_pct * taker_ls_vol_ratio (aggressive positioning proxy)

IC metric: Rank IC (Spearman correlation) karena target binary (0/1).
Threshold untuk lolos: IC > 0.03 dengan t-stat > 2.0 (lowered dari 0.05 karena binary target lebih noisy).

Usage:
  python scratch/ic_test_positioning_vs_win.py
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOF_PATH  = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"
POS_DIR   = ROOT / "data" / "positioning_hist"
ROLL_WIN  = 20  # z-score rolling window

# IC threshold untuk pass gate
IC_THRESH = 0.03
T_THRESH  = 2.0


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=5).mean()
    std = s.rolling(window, min_periods=5).std().replace(0, np.nan)
    return (s - mu) / std


def load_positioning(coin: str) -> pd.DataFrame | None:
    path = POS_DIR / f"{coin}_metrics.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    # Engineering (Challenge 3)
    # 1. OI delta already percent-change — add z-score
    if "oi_usd_delta_pct" in df.columns:
        df["oi_delta_z"] = rolling_zscore(df["oi_usd_delta_pct"], ROLL_WIN)

    # 2. Taker L/S ratio — z-score (level non-stationary)
    if "taker_ls_vol_ratio" in df.columns:
        df["taker_ls_z"] = rolling_zscore(df["taker_ls_vol_ratio"], ROLL_WIN)

    # 3. Top-trader L/S ratio — z-score
    if "toptrader_ls_ratio" in df.columns:
        df["toptrader_ls_z"] = rolling_zscore(df["toptrader_ls_ratio"], ROLL_WIN)

    # 4. Cross feature: OI delta × taker direction
    if "oi_usd_delta_pct" in df.columns and "taker_ls_vol_ratio" in df.columns:
        # taker > 1 = longs dominant, < 1 = shorts dominant
        taker_side = df["taker_ls_vol_ratio"] - 1.0  # centered at 0
        df["oi_taker_cross"] = df["oi_usd_delta_pct"] * taker_side

    # 5. Deltas (dari Binance Vision download)
    for col in ["taker_ls_delta", "toptrader_ls_delta"]:
        if col in df.columns:
            df[f"{col}_z"] = rolling_zscore(df[col], ROLL_WIN)

    return df


def rank_ic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank IC and t-stat. Returns (ic, t_stat)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return 0.0, 0.0
    ic, _ = stats.spearmanr(x, y)
    t = ic * np.sqrt((n - 2) / (1 - ic ** 2 + 1e-9))
    return float(ic), float(t)


def main():
    if not OOF_PATH.exists():
        print(f"[ERROR] OOF trades not found: {OOF_PATH}")
        print("  Run: python pipeline/08_generate_ic32_oof_trades.py first")
        return

    trades = pd.read_parquet(OOF_PATH)
    print(f"\n{'='*65}")
    print(f"  IC Test: Positioning Features vs ic32 WIN/LOSS")
    print(f"{'='*65}")
    print(f"  OOF trades : {len(trades):,}")
    print(f"  Overall WR : {trades['win'].mean()*100:.1f}%")
    print(f"  LONG trades: {(trades['direction']==1).sum():,}")
    print(f"  SHORT trades:{(trades['direction']==-1).sum():,}")
    print(f"  Z-score window: {ROLL_WIN} bars\n")

    # Normalize trade timestamp to date (for join with daily positioning data)
    trades["date"] = pd.to_datetime(trades["timestamp"], utc=True).dt.normalize()

    # Collect IC results across all coins
    feature_records: dict[str, list[float]] = {}
    n_coins_tested = 0

    for coin in trades["coin"].unique():
        pos_df = load_positioning(coin)
        if pos_df is None:
            continue

        coin_trades = trades[trades["coin"] == coin].copy()

        # T-1 lag: use positioning from previous day (known at entry time)
        pos_shifted = pos_df.shift(1)  # Daily: shift 1 day

        # Join by date
        coin_trades = coin_trades.join(pos_shifted, on="date", how="left")

        if len(coin_trades) < 30:
            continue

        n_coins_tested += 1
        y = coin_trades["win"].values.astype(float)

        pos_feat_cols = [c for c in pos_shifted.columns
                         if c not in ("symbol",) and c in coin_trades.columns]
        for feat in pos_feat_cols:
            x = coin_trades[feat].values.astype(float)
            ic, t = rank_ic(x, y)
            if feat not in feature_records:
                feature_records[feat] = []
            feature_records[feat].append(ic)

    if not feature_records:
        print("[WARN] No data to test. Check positioning files.")
        return

    # Aggregate: mean IC and t-stat across coins (Fama-MacBeth style)
    print(f"  Coins tested: {n_coins_tested}")
    print(f"\n  {'Feature':<30} {'Mean IC':>10} {'Std IC':>9} {'t-stat':>9} {'Pass':>6}")
    print(f"  {'-'*68}")

    results = []
    for feat, ics in sorted(feature_records.items()):
        arr = np.array(ics)
        mean_ic = float(np.nanmean(arr))
        std_ic  = float(np.nanstd(arr))
        n_c     = len(arr)
        # t-stat: mean_ic / (std_ic / sqrt(n_coins))
        t_stat  = mean_ic / (std_ic / np.sqrt(n_c) + 1e-9) if std_ic > 0 else 0.0
        pass_gate = abs(mean_ic) >= IC_THRESH and abs(t_stat) >= T_THRESH
        flag = "YES" if pass_gate else "---"
        results.append({
            "feature": feat, "mean_ic": round(mean_ic, 5),
            "std_ic": round(std_ic, 5), "t_stat": round(t_stat, 3),
            "n_coins": n_c, "pass": pass_gate,
        })
        print(f"  {feat:<30} {mean_ic:>+10.5f} {std_ic:>9.5f} {t_stat:>+9.3f} {flag:>6}")

    passed = [r for r in results if r["pass"]]
    print(f"\n  Features passing gate (|IC|>={IC_THRESH}, |t|>={T_THRESH}): "
          f"{len(passed)}/{len(results)}")
    if passed:
        print(f"\n  PASSING FEATURES:")
        for r in sorted(passed, key=lambda x: abs(x["mean_ic"]), reverse=True):
            print(f"    {r['feature']:<30} IC={r['mean_ic']:+.5f}  t={r['t_stat']:+.3f}")
    else:
        print("\n  [RESULT] No positioning features pass IC gate for ic32 WIN/LOSS.")
        print("  This means positioning data (at daily granularity) does not")
        print("  predict ic32 trade outcomes better than random.")

    # Also test with direction-adjusted WIN (does positioning align with direction?)
    print(f"\n{'='*65}")
    print("  Direction-adjusted test: WIN conditional on direction-match")
    print("  (Does taker_ls predict LONG-WIN more than SHORT-WIN?)")
    print(f"{'='*65}")

    for direction, label in [(1, "LONG"), (-1, "SHORT")]:
        dir_trades = trades[trades["direction"] == direction].copy()
        dir_trades["date"] = pd.to_datetime(dir_trades["timestamp"], utc=True).dt.normalize()
        print(f"\n  {label} trades ({len(dir_trades):,}):")

        dir_ics: dict[str, list[float]] = {}
        for coin in dir_trades["coin"].unique():
            pos_df = load_positioning(coin)
            if pos_df is None:
                continue
            ct = dir_trades[dir_trades["coin"] == coin].copy()
            pos_shifted = pos_df.shift(1)
            ct = ct.join(pos_shifted, on="date", how="left")
            if len(ct) < 20:
                continue
            y = ct["win"].values.astype(float)
            for feat in ["taker_ls_z", "oi_delta_z", "toptrader_ls_z", "oi_taker_cross"]:
                if feat in ct.columns:
                    x = ct[feat].values.astype(float)
                    ic, t = rank_ic(x, y)
                    if feat not in dir_ics:
                        dir_ics[feat] = []
                    dir_ics[feat].append(ic)

        if dir_ics:
            print(f"  {'Feature':<30} {'Mean IC':>10} {'t-stat':>9}")
            for feat, ics in sorted(dir_ics.items()):
                arr = np.array(ics)
                m = float(np.nanmean(arr))
                s = float(np.nanstd(arr))
                n_c = len(arr)
                t = m / (s / np.sqrt(n_c) + 1e-9) if s > 0 else 0.0
                print(f"  {feat:<30} {m:>+10.5f} {t:>+9.3f}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
