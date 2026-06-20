"""
scratch/ic_test_coinank_tb.py — IC test coinank features vs Triple Barrier labels.

Methodology:
- Target: TB label (ordinal: SHORT=-1, FLAT=0, LONG=1)
- Data: training coins, Jan 2025 – Oct 2025 (coinank overlap dengan training period)
- Frequency: H1 (coinank H4 di-forward-fill ke H1)
- IC: Spearman, N_eff = N/24 (koreksi autocorrelation H1)
- Threshold: |IC| >= 0.02, |t-stat| >= 2.0

Features yang ditest:
  Raw    : oi_binance, ls_position, ls_account
  Derived: delta 4H/1D, pct_change, z-score 20D, OI-price divergence,
           smart_money_div (ls_position - ls_account)
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE
from core.features import triple_barrier_labeling

COINANK_DIR = ROOT / "data" / "coinank"
TP_ATR = 2.0
SL_ATR = 1.5
MAX_HOLD = 36
# Only period dengan coinank coverage (Jan 2025 – TRAIN_CUTOFF)
COINANK_START = pd.Timestamp("2025-01-01", tz="UTC")

N_EFF_DIVISOR = 24   # H1 autocorrelation correction: 1 effective obs per day
MIN_IC        = 0.02
MIN_TSTAT     = 2.0


def build_coinank_features(df_h1, oi, lsp, lsa):
    """
    Merge coinank data ke H1 DataFrame dan generate derived features.
    Semua coinank data di-forward-fill ke H1 index.
    """
    idx = df_h1.index

    feats = pd.DataFrame(index=idx)

    # ── OI ──────────────────────────────────────────────────────────────────
    if oi is not None and "oi_binance" in oi.columns:
        oi_s = oi["oi_binance"].reindex(idx, method="ffill")
        feats["oi_raw"]            = oi_s
        feats["oi_delta_4h"]       = oi_s.diff(4)
        feats["oi_delta_1d"]       = oi_s.diff(24)
        feats["oi_pct_4h"]         = oi_s.pct_change(4)
        feats["oi_pct_1d"]         = oi_s.pct_change(24)
        oi_mean = oi_s.rolling(24*20).mean()
        oi_std  = oi_s.rolling(24*20).std().clip(lower=1e-8)
        feats["oi_zscore_20d"]     = (oi_s - oi_mean) / oi_std
        # OI vs price divergence (OI naik tapi price turun = bearish divergence)
        price = df_h1["close"]
        feats["oi_price_div_1d"]   = feats["oi_pct_1d"] - price.pct_change(24)
        feats["oi_price_div_3d"]   = oi_s.pct_change(72) - price.pct_change(72)

    # ── L/S Position (top trader) ────────────────────────────────────────────
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls_p = lsp["top_trader_position_ls"].reindex(idx, method="ffill")
        feats["ls_pos_raw"]        = ls_p
        feats["ls_pos_delta_4h"]   = ls_p.diff(4)
        feats["ls_pos_delta_1d"]   = ls_p.diff(24)
        ls_p_mean = ls_p.rolling(24*20).mean()
        ls_p_std  = ls_p.rolling(24*20).std().clip(lower=1e-8)
        feats["ls_pos_zscore_20d"] = (ls_p - ls_p_mean) / ls_p_std
        feats["ls_pos_direction"]  = ls_p - 1.0   # >0 = long bias, <0 = short bias

    # ── L/S Account (retail) ─────────────────────────────────────────────────
    if lsa is not None and "top_trader_account_ls" in lsa.columns:
        ls_a = lsa["top_trader_account_ls"].reindex(idx, method="ffill")
        feats["ls_acc_raw"]        = ls_a
        feats["ls_acc_delta_4h"]   = ls_a.diff(4)
        feats["ls_acc_delta_1d"]   = ls_a.diff(24)
        ls_a_mean = ls_a.rolling(24*20).mean()
        ls_a_std  = ls_a.rolling(24*20).std().clip(lower=1e-8)
        feats["ls_acc_zscore_20d"] = (ls_a - ls_a_mean) / ls_a_std

        # Smart money divergence: top trader position vs retail account
        if "ls_pos_raw" in feats:
            feats["smart_retail_div"]  = feats["ls_pos_raw"] - ls_a
            feats["smart_retail_div_delta_1d"] = feats["smart_retail_div"].diff(24)

    return feats


def run_ic(x, y):
    """Spearman IC dengan N_eff correction."""
    mask = x.notna() & y.notna()
    if mask.sum() < 100:
        return None
    xv, yv = x[mask].values, y[mask].values
    ic, _ = stats.spearmanr(xv, yv)
    n_eff  = mask.sum() / N_EFF_DIVISOR
    t_stat = ic * np.sqrt(n_eff - 2) / np.sqrt(1 - ic**2 + 1e-9)
    return {"ic": ic, "t_stat": t_stat, "n": int(mask.sum()), "n_eff": int(n_eff)}


def main():
    print(f"\n{'='*70}")
    print(f"  IC TEST: Coinank Features vs Triple Barrier Labels")
    print(f"  Period : {COINANK_START.date()} – {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Target : TB label (SHORT=-1, FLAT=0, LONG=1)")
    print(f"  N_eff  : N / {N_EFF_DIVISOR} (H1 autocorr correction)")
    print(f"  Thresh : |IC|>={MIN_IC}, |t-stat|>={MIN_TSTAT}")
    print(f"{'='*70}\n")

    frames = []   # list of (feats_df + label) per coin
    coins_ok = 0

    for sym in TRAINING_COINS:
        fp   = LABEL_DIR / f"{sym}_features_v3.parquet"
        oi_p = COINANK_DIR / f"{sym}_oi.parquet"
        lsp_p= COINANK_DIR / f"{sym}_ls_position.parquet"
        lsa_p= COINANK_DIR / f"{sym}_ls_account.parquet"

        if not fp.exists() or not oi_p.exists():
            continue

        df = pd.read_parquet(fp).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[(df.index >= COINANK_START) & (df.index < TRAIN_CUTOFF_DATE)]
        if len(df) < 500:
            continue
        if any(c not in df.columns for c in ["close", "high", "low", "atr_14_h1"]):
            continue

        # TB labels
        tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                     df["atr_14_h1"], TP_ATR, SL_ATR, MAX_HOLD)
        label_map = {"SHORT": -1, "FLAT": 0, "LONG": 1}
        y = tb.map(label_map).dropna()
        df = df.loc[y.index]

        oi  = pd.read_parquet(oi_p).sort_index()
        lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
        lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None

        feats = build_coinank_features(df, oi, lsp, lsa)
        feats = feats.loc[y.index]
        feats["__label__"] = y.values
        frames.append(feats)
        coins_ok += 1

    if not frames:
        print("  NO DATA — cek path coinank")
        return

    combined = pd.concat(frames, ignore_index=True)
    y_all = combined["__label__"]
    feat_cols = [c for c in combined.columns if c != "__label__"]

    print(f"  Coins OK  : {coins_ok}")
    print(f"  Total bars: {len(y_all):,}")
    dist = y_all.value_counts()
    print(f"  Label dist: SHORT={dist.get(-1,0):,} FLAT={dist.get(0,0):,} LONG={dist.get(1,0):,}\n")

    # Run IC per feature
    results = []
    for feat in feat_cols:
        r = run_ic(combined[feat], y_all)
        if r is None:
            continue
        verdict = "KEEP" if abs(r["ic"]) >= MIN_IC and abs(r["t_stat"]) >= MIN_TSTAT else "WEAK"
        results.append({
            "feature": feat,
            "ic": r["ic"], "abs_ic": abs(r["ic"]),
            "t_stat": r["t_stat"], "n": r["n"], "n_eff": r["n_eff"],
            "verdict": verdict,
        })

    results.sort(key=lambda x: x["abs_ic"], reverse=True)

    # Print results
    print(f"  {'Feature':<35} {'IC':>8} {'|IC|':>7} {'t-stat':>8} {'N_eff':>7} {'Verdict'}")
    print("  " + "-" * 75)
    for r in results:
        flag = " <-- KEEP" if r["verdict"] == "KEEP" else ""
        print(f"  {r['feature']:<35} {r['ic']:>+8.4f} {r['abs_ic']:>7.4f} "
              f"{r['t_stat']:>+8.2f} {r['n_eff']:>7,}{flag}")

    keep = [r for r in results if r["verdict"] == "KEEP"]
    print(f"\n  KEEP: {len(keep)} features | WEAK: {len(results)-len(keep)} features")
    print(f"\n  KEEP list: {[r['feature'] for r in keep]}")

    # ── Stage 2: IC Stability across time windows ────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STABILITY TEST — IC per temporal window (Jan-Apr / May-Aug / Sep-Oct 2025)")
    print(f"  STABLE = sign konsisten >= 3/4 window, IC_IR >= 0.5")
    print(f"{'='*70}")

    # Tambahkan timestamp per row: gabungkan index dari frames
    ts_list = []
    for sym in TRAINING_COINS:
        fp   = LABEL_DIR / f"{sym}_features_v3.parquet"
        oi_p = COINANK_DIR / f"{sym}_oi.parquet"
        if not fp.exists() or not oi_p.exists():
            continue
        df = pd.read_parquet(fp).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[(df.index >= COINANK_START) & (df.index < TRAIN_CUTOFF_DATE)]
        if len(df) < 500:
            continue
        if any(c not in df.columns for c in ["close", "high", "low", "atr_14_h1"]):
            continue
        tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                     df["atr_14_h1"], TP_ATR, SL_ATR, MAX_HOLD)
        y = tb.map({"SHORT": -1, "FLAT": 0, "LONG": 1}).dropna()
        ts_list.extend(y.index.tolist())

    combined["__ts__"] = ts_list[:len(combined)]  # align

    windows = [
        ("Jan-Apr 2025", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-05-01", tz="UTC")),
        ("May-Aug 2025", pd.Timestamp("2025-05-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC")),
        ("Sep-Oct 2025", pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC")),
    ]

    keep_feats = [r["feature"] for r in keep]
    print(f"\n  {'Feature':<35} " + " ".join(f"{'w'+str(i+1):>10}" for i in range(len(windows)))
          + f"  {'IC_IR':>7}  {'Verdict'}")
    print("  " + "-" * 80)

    stable_feats = []
    for feat in keep_feats:
        win_ics = []
        for w_label, w_start, w_end in windows:
            mask = (combined["__ts__"] >= w_start) & (combined["__ts__"] < w_end)
            sub = combined[mask]
            if len(sub) < 100:
                win_ics.append(np.nan)
                continue
            r = run_ic(sub[feat], sub["__label__"])
            win_ics.append(r["ic"] if r else np.nan)

        ic_arr    = np.array([x for x in win_ics if not np.isnan(x)])
        ic_ir     = float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-9)) if len(ic_arr) >= 2 else 0.0
        signs     = [np.sign(x) for x in ic_arr if not np.isnan(x)]
        dominant  = max(set(signs), key=signs.count) if signs else 0
        sign_cons = sum(1 for s in signs if s == dominant)
        stable    = sign_cons >= 2 and abs(ic_ir) >= 0.5

        verdict   = "STABLE" if stable else "UNSTABLE"
        if stable:
            stable_feats.append(feat)

        w_strs = [f"{v:>+10.4f}" if not np.isnan(v) else f"{'NaN':>10}" for v in win_ics]
        print(f"  {feat:<35} {''.join(w_strs)}  {ic_ir:>+7.2f}  {verdict}")

    print(f"\n  STABLE features ({len(stable_feats)}): {stable_feats}")
    print(f"\n  FINAL COINANK KEEP+STABLE features untuk training:")
    for f in stable_feats:
        r = next(x for x in results if x["feature"] == f)
        print(f"    {f:<35} IC={r['ic']:+.4f}  t={r['t_stat']:+.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
