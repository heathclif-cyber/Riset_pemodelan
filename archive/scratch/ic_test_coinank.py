"""
IC Test: Coinank positioning features vs momentum labels
Tests whether OI delta, LS ratio, etc. have temporal predictability for momentum.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE

COINANK_DIR = ROOT / "data" / "coinank"

def load_features(coin):
    """Load coinank positioning features and align with momentum labels."""
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    lp = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
    oi_p = COINANK_DIR / f"{coin}_oi.parquet"
    lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"

    if not all([fp.exists(), lp.exists(), oi_p.exists()]):
        return None

    # Load hourly features + labels
    df = pd.read_parquet(fp).sort_index()
    lbl = pd.read_parquet(lp).sort_index()
    df = df.join(lbl["momentum_v2_label"], how="inner")
    df = df.dropna(subset=["momentum_v2_label"])
    # TRAIN ONLY
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if len(df) < 200:
        return None

    # Load daily positioning data
    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None

    # Resample hourly data to daily for join
    df["date"] = pd.to_datetime(df.index.date, utc=True)

    # OI features
    oi_cols = [c for c in oi.columns if c.startswith("oi_") and c != "oi_total"]
    total_oi = None
    for col in oi_cols:
        if total_oi is None:
            total_oi = oi[col].copy()
        else:
            total_oi = total_oi.fillna(0) + oi[col].fillna(0)

    if total_oi is None or len(total_oi.dropna()) < 50:
        return None

    daily = pd.DataFrame({"oi_total": total_oi}, index=oi.index)

    # LS features
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        daily["ls_position"] = lsp["top_trader_position_ls"]
    if lsa is not None and "top_trader_account_ls" in lsa.columns:
        daily["ls_account"] = lsa["top_trader_account_ls"]

    # Engineered features
    daily["oi_delta_1d"] = daily["oi_total"].diff(1)
    daily["oi_delta_3d"] = daily["oi_total"].diff(3)
    daily["oi_delta_7d"] = daily["oi_total"].diff(7)
    daily["oi_momentum"] = daily["oi_total"].pct_change(7)
    daily["oi_zscore_20d"] = (daily["oi_total"] - daily["oi_total"].rolling(20).mean()) / daily["oi_total"].rolling(20).std().clip(lower=1e-8)

    if "ls_position" in daily.columns:
        daily["ls_position_delta_3d"] = daily["ls_position"].diff(3)
        daily["ls_position_zscore_20d"] = (daily["ls_position"] - daily["ls_position"].rolling(20).mean()) / daily["ls_position"].rolling(20).std().clip(lower=1e-8)
        daily["ls_position_direction"] = daily["ls_position"] - 1.0  # > 0 = net long bias

    if "ls_account" in daily.columns:
        daily["ls_account_delta_3d"] = daily["ls_account"].diff(3)
        daily["smart_retail_divergence"] = daily["ls_position"] - daily["ls_account"]  # position vs account

    # Join to hourly bars (each daily value -> all hours that day)
    daily_idx = pd.to_datetime(daily.index.date, utc=True)
    daily = daily[~daily.index.duplicated()]
    daily["_date"] = pd.to_datetime(daily.index.date, utc=True)

    join_cols = [c for c in daily.columns if c != "_date"]
    df = df.join(daily.set_index("_date")[join_cols], on="date", how="left")

    # Forward-fill missing daily data
    for c in join_cols:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)

    return coin, df, join_cols


def ic_test(feature_series, label_series, effective_n_factor=4):
    """Spearman IC between feature and label. effective_n_factor penalizes daily autocorrelation."""
    mask = feature_series.notna() & label_series.notna()
    x = feature_series[mask].values
    y = label_series[mask].values
    if len(x) < 50:
        return None

    ic, pval = stats.spearmanr(x, y)
    n_eff = len(x) / effective_n_factor
    ic_ir = abs(ic) * np.sqrt(n_eff) if n_eff > 0 else 0

    # Direction consistency
    halves = np.array_split(x, 2)
    signs = []
    for half_x in halves:
        if len(half_x) > 20:
            s, _ = stats.spearmanr(half_x[mask.values[:len(half_x)]][:len(half_x)] if len(half_x) <= len(y) else half_x,
                                   y[:len(half_x)] if len(half_x) <= len(y) else y)
            signs.append(1 if s > 0 else -1)

    consistency = sum(1 for s in signs if s == (1 if ic > 0 else -1)) / len(signs) * 100 if signs else 50

    return {"ic": ic, "pval": pval, "n_raw": len(x), "n_eff": n_eff, "ic_ir": ic_ir, "consistency": consistency}


def main():
    coins = TRAINING_COINS
    print(f"\n{'='*70}")
    print(f"  IC TEST — Coinank Positioning Features vs Momentum Labels")
    print(f"  Training period only (before {TRAIN_CUTOFF_DATE.date()})")
    print(f"  Effective N = N / 4 (daily autocorrelation adjustment)")
    print(f"{'='*70}\n")

    all_results = {}
    feature_names = set()

    for coin in coins:
        result = load_features(coin)
        if result is None:
            continue

        coin_name, df, feat_cols = result
        feature_names.update(feat_cols)
        all_results[coin_name] = df

    if not all_results:
        print("  NO DATA — check coinank parquet files")
        return

    print(f"  Coins with data: {len(all_results)}/{len(coins)}")
    print(f"  Positioning features: {len(feature_names)}")
    print()

    # Run IC tests per feature, aggregated across all coins
    feat_list = sorted(feature_names)
    print(f"  {'Feature':<35} {'Mean IC':>8} {'Std IC':>8} {'IC_IR':>7} {'Sign%':>7} {'Verdict':>10}")
    print("  " + "-" * 85)

    results = []
    for feat in feat_list:
        ic_vals = []
        n_totals = []
        for coin, df in all_results.items():
            if feat not in df.columns:
                continue
            # Map label to -1, 0, 1 (for continuous correlation)
            y = df["momentum_v2_label"].map({0: -1, 1: 0, 2: 1})
            r = ic_test(df[feat], y)
            if r:
                ic_vals.append(r["ic"])
                n_totals.append(r["n_raw"])

        if len(ic_vals) < 3:
            continue

        mean_ic = np.mean(ic_vals)
        std_ic = np.std(ic_vals)
        abs_ic = abs(mean_ic)
        mean_ic_ir = abs_ic * np.sqrt(np.mean(n_totals) / 4)
        sign_pct = sum(1 for v in ic_vals if v * mean_ic > 0) / len(ic_vals) * 100

        if abs_ic >= 0.03 and sign_pct >= 70:
            verdict = "KEEP ++"
        elif abs_ic >= 0.02 and sign_pct >= 60:
            verdict = "KEEP +"
        elif abs_ic >= 0.01:
            verdict = "WEAK"
        else:
            verdict = "DROP"

        results.append({
            "feature": feat,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": mean_ic_ir,
            "sign_pct": sign_pct,
            "verdict": verdict,
        })

        print(f"  {feat:<35} {mean_ic:>+8.4f} {std_ic:>8.4f} {mean_ic_ir:>7.2f} {sign_pct:>6.0f}% {verdict:>10}")

    # Summary
    keep = [r for r in results if "KEEP" in r["verdict"]]
    print(f"\n  {'='*50}")
    print(f"  KEEP features: {len(keep)}/{len(results)}")
    if keep:
        print(f"\n  TOP KEEP (sorted by |IC|):")
        keep_sorted = sorted(keep, key=lambda r: abs(r["mean_ic"]), reverse=True)
        for r in keep_sorted:
            print(f"  {r['feature']:<35} IC={r['mean_ic']:+.4f}  |IC|={abs(r['mean_ic']):.4f}  sign={r['sign_pct']:.0f}%")
    else:
        print(f"\n  WARNING: No features pass KEEP threshold!")

    # Compare with OHLCV features from previous IC test
    print(f"\n  Reference (OHLCV IC test, same methodology):")
    print(f"    Best OHLCV features: IC 0.05-0.12 (cvd_momentum_adv, ofi_z_score, etc.)")
    print(f"    OHLCV KEEP cutoff:   IC >= 0.02")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
