"""
IC Test V2: Coinank positioning features vs MULTIPLE targets
Tests: price return, regime change, OI divergence, trend strength
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE

COINANK_DIR = ROOT / "data" / "coinank"


def load_data(coin):
    """Load daily positioning + hourly features aligned."""
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    oi_p = COINANK_DIR / f"{coin}_oi.parquet"
    lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"

    if not (fp.exists() and oi_p.exists()):
        return None

    df = pd.read_parquet(fp).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if len(df) < 500:
        return None

    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None

    return coin, df, oi, lsp, lsa


def build_daily_features(coin, df, oi, lsp, lsa):
    """Build DAILY dataframe with positioning features AND forward targets."""
    # Resample price to daily
    daily_price = df[["open", "high", "low", "close", "volume"]].resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    # OI features
    oi_total = None
    for col in oi.columns:
        if col.startswith("oi_") and col != "oi_total":
            if oi_total is None:
                oi_total = oi[col]
            else:
                oi_total = oi_total.fillna(0) + oi[col].fillna(0)

    if oi_total is None:
        return None

    daily = pd.DataFrame(index=daily_price.index)
    daily["close"] = daily_price["close"]
    daily["volume"] = daily_price["volume"]
    daily["high"] = daily_price["high"]
    daily["low"] = daily_price["low"]

    # Price features
    daily["ret_1d"] = daily["close"].pct_change(1)
    daily["ret_vol_1d"] = daily["volume"].pct_change(1)
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"]

    # OI features
    oi_aligned = oi_total.reindex(daily.index, method="ffill")
    daily["oi_total"] = oi_aligned
    daily["oi_delta_1d"] = oi_aligned.diff(1)
    daily["oi_delta_3d"] = oi_aligned.diff(3)
    daily["oi_delta_7d"] = oi_aligned.diff(7)

    # OI z-score
    oi_roll_mean = oi_aligned.rolling(20).mean()
    oi_roll_std = oi_aligned.rolling(20).std().clip(lower=1e-8)
    daily["oi_zscore_20d"] = (oi_aligned - oi_roll_mean) / oi_roll_std

    # OI vs price divergence
    daily["oi_price_div_7d"] = daily["oi_delta_7d"] / oi_aligned.shift(7).clip(lower=1e-8) - daily["close"].pct_change(7)

    # LS Position features
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        lsp_aligned = lsp["top_trader_position_ls"].reindex(daily.index, method="ffill")
        daily["ls_position"] = lsp_aligned
        daily["ls_position_delta_3d"] = lsp_aligned.diff(3)
        daily["ls_position_delta_7d"] = lsp_aligned.diff(7)
        lsp_roll_mean = lsp_aligned.rolling(20).mean()
        lsp_roll_std = lsp_aligned.rolling(20).std().clip(lower=1e-8)
        daily["ls_position_zscore_20d"] = (lsp_aligned - lsp_roll_mean) / lsp_roll_std
        daily["ls_position_direction"] = lsp_aligned - 1.0

    # LS Account features
    if lsa is not None and "top_trader_account_ls" in lsa.columns:
        lsa_aligned = lsa["top_trader_account_ls"].reindex(daily.index, method="ffill")
        daily["ls_account"] = lsa_aligned
        daily["ls_account_delta_3d"] = lsa_aligned.diff(3)
        daily["smart_retail_div"] = daily.get("ls_position", 0) - lsa_aligned

    # TARGETS
    # Target 1: Future price return (1d, 3d, 7d)
    for n in [1, 3, 7]:
        daily[f"target_ret_{n}d"] = daily["close"].pct_change(n).shift(-n)

    # Target 2: Binary direction (1=up, -1=down, 0=flat)
    for n in [1, 3, 7]:
        ret = daily["close"].pct_change(n).shift(-n)
        daily[f"target_dir_{n}d"] = 0
        daily.loc[ret > 0.01, f"target_dir_{n}d"] = 1
        daily.loc[ret < -0.01, f"target_dir_{n}d"] = -1

    # Target 3: Sharp move (binary: >2% in 3 days)
    ret_3d = daily["close"].pct_change(3).shift(-3)
    daily["target_sharp_up"] = (ret_3d > 0.02).astype(int)
    daily["target_sharp_down"] = (ret_3d < -0.02).astype(int)

    # Target 4: Volatility expansion
    for n in [3, 7]:
        daily[f"target_vol_{n}d"] = daily["range_pct"].rolling(n).mean().shift(-n)

    # Drop NaN rows
    daily = daily.dropna()

    return daily if len(daily) >= 100 else None


def run_ic(feature, target, effective_n=4):
    """Spearman IC with temporal adjustment."""
    mask = feature.notna() & target.notna()
    if mask.sum() < 50:
        return None
    x = feature[mask].values
    y = target[mask].values
    ic, pval = stats.spearmanr(x, y)
    n_eff = len(x) / effective_n
    return {"ic": ic, "pval": pval, "n": len(x), "n_eff": n_eff}


def main():
    coins = TRAINING_COINS
    print(f"\n{'='*75}")
    print(f"  IC TEST V2: Positioning Features vs Multiple Targets")
    print(f"  DAILY frequency | Training data only (before {TRAIN_CUTOFF_DATE.date()})")
    print(f"  Effective N adjustment: /4 for daily autocorrelation")
    print(f"{'='*75}\n")

    all_daily = {}
    skipped = 0

    for coin in coins:
        data = load_data(coin)
        if data is None:
            skipped += 1
            continue
        coin_name, df, oi, lsp, lsa = data
        daily = build_daily_features(coin_name, df, oi, lsp, lsa)
        if daily is not None:
            all_daily[coin_name] = daily
        else:
            skipped += 1

    if not all_daily:
        print("  NO DATA")
        return

    # Merge all coins
    combined = pd.concat(all_daily.values(), keys=all_daily.keys(), names=["coin"])
    print(f"  Coins: {len(all_daily)} | Total daily rows: {len(combined):,} | Skipped: {skipped}")
    print(f"  Date range: {combined.index.get_level_values(1).min().date()} -> {combined.index.get_level_values(1).max().date()}")
    print()

    # Define feature and target groups
    feature_groups = {
        "OI Flow": ["oi_delta_1d", "oi_delta_3d", "oi_delta_7d", "oi_zscore_20d", "oi_price_div_7d"],
        "LS Position": ["ls_position", "ls_position_delta_3d", "ls_position_delta_7d", "ls_position_zscore_20d", "ls_position_direction"],
        "LS Account": ["ls_account", "ls_account_delta_3d", "smart_retail_div"],
    }

    target_groups = {
        "Price Return 1D": ["target_ret_1d", "target_dir_1d"],
        "Price Return 3D": ["target_ret_3d", "target_dir_3d"],
        "Price Return 7D": ["target_ret_7d", "target_dir_7d"],
        "Sharp Move Up 3D": ["target_sharp_up"],
        "Sharp Move Down 3D": ["target_sharp_down"],
        "Vol Expansion 3D/7D": ["target_vol_3d", "target_vol_7d"],
    }

    # Run IC for each feature-target pair
    print("  TOP IC RESULTS (|IC| >= 0.03):")
    print(f"  {'Feature':<30} {'Target':<25} {'IC':>8} {'|IC|':>7} {'N':>6} {'IC_IR':>7}")
    print("  " + "-" * 90)

    top_results = []
    for feat_group, feats in feature_groups.items():
        for feat in feats:
            if feat not in combined.columns:
                continue
            for tgt_group, targets in target_groups.items():
                for tgt in targets:
                    if tgt not in combined.columns:
                        continue
                    r = run_ic(combined[feat], combined[tgt])
                    if r is None:
                        continue
                    abs_ic = abs(r["ic"])
                    if abs_ic >= 0.03:
                        ic_ir = abs_ic * np.sqrt(r["n_eff"])
                        top_results.append({
                            "feat": feat, "tgt": tgt, "tgt_group": tgt_group,
                            "ic": r["ic"], "abs_ic": abs_ic, "ic_ir": ic_ir, "n": r["n"]
                        })
                        print(f"  {feat:<30} {tgt:<25} {r['ic']:>+8.4f} {abs_ic:>7.4f} {r['n']:>6} {ic_ir:>7.2f}")

    if not top_results:
        print("    (none found — all IC < 0.03)")
    else:
        top_results.sort(key=lambda r: r["abs_ic"], reverse=True)
        print(f"\n  TOP 10 OVERALL:")
        for i, r in enumerate(top_results[:10]):
            direction = "POS" if r["ic"] > 0 else "NEG"
            print(f"  {i+1}. {r['feat']:<30} x {r['tgt']:<25} IC={r['ic']:+.4f} |IC|={r['abs_ic']:.4f} IC_IR={r['ic_ir']:.2f} [{direction}]")

    # Summary per feature group
    print(f"\n  {'='*60}")
    print(f"  SUMMARY BY FEATURE GROUP (mean |IC| across all targets)")
    print(f"  {'='*60}")
    for feat_group, feats in feature_groups.items():
        group_ics = []
        for feat in feats:
            if feat not in combined.columns:
                continue
            for tgt_group, targets in target_groups.items():
                for tgt in targets:
                    if tgt not in combined.columns:
                        continue
                    r = run_ic(combined[feat], combined[tgt])
                    if r:
                        group_ics.append(abs(r["ic"]))
        if group_ics:
            print(f"  {feat_group:<20}: mean |IC|={np.mean(group_ics):.4f}  max={np.max(group_ics):.4f}  n_tests={len(group_ics)}")

    # Compare: correlation with future price return (the REAL target)
    print(f"\n  {'='*60}")
    print(f"  KEY TEST: Can positioning predict simple price direction?")
    print(f"  {'='*60}")
    for n in [1, 3, 7]:
        target = f"target_ret_{n}d"
        if target not in combined.columns:
            continue
        print(f"\n  --- {target} (future {n}d return) ---")
        for feat_group, feats in feature_groups.items():
            for feat in feats:
                if feat not in combined.columns:
                    continue
                r = run_ic(combined[feat], combined[target])
                if r:
                    sig = " **" if abs(r["ic"]) >= 0.03 else ""
                    print(f"  {feat:<30} IC={r['ic']:+.4f}  |IC|={abs(r['ic']):.4f}  n={r['n']}{sig}")

    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    main()
