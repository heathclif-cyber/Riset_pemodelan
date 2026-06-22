"""
pipeline/06_ic_test_tb_v2.py — IC Feature Selection expanded set untuk TB v2

Kandidat:
  1. Semua FEATURE_COLS_V3 (~109 fitur dari parquet)
  2. Candle structure baru: candle_body_ratio, upper_wick_ratio, lower_wick_ratio,
     candle_range_atr_ratio
  3. Funding dynamics: funding_rate_change_8h, funding_rate_zscore_20d
  4. Coinank derived: ls_pos_zscore_20d, smart_retail_div_delta_1d,
     oi_pct_1d, oi_price_div_1d

Tidak ada training — hanya IC test + decay + rekomendasi fitur.

Output: models/runs/tb_ic_v2/
  - tb_ic_v2_ic_results.json
  - tb_ic_v2_ic_decay.json
  - tb_ic_v2_recommended.json

Jalankan:
  python pipeline/06_ic_test_tb_v2.py
  python pipeline/06_ic_test_tb_v2.py --all
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR,
    FEATURE_COLS_V3, TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR,
)

import argparse

logger = setup_logger("06_ic_test_tb_v2")

RUN_NAME     = "tb_ic_v2"
COINANK_DIR  = ROOT / "data" / "coinank"
AUTOCORR_FACTOR = 24

MIN_STANDALONE_IC = 0.02
MIN_TSTAT         = 2.0
MIN_MARGINAL_IC   = 0.01

IC_STABILITY_IR_THRESH    = 0.5
IC_STABILITY_SIGN_THRESH  = 5  # min berapa window sign-nya konsisten dari 6

DECAY_WINDOWS = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-10-31"),
]

TB_LABEL_MAP  = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# TB v3 features yang sudah deployed (untuk menandai status tiap feature)
TB_V3_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

# Fitur baru yang belum ada di FEATURE_COLS_V3 (akan di-derive di sini)
NEW_FEATURES = [
    # Candle structure
    "candle_body_ratio",      # |close-open| / (high-low) — conviction candle
    "upper_wick_ratio",       # rejection atas
    "lower_wick_ratio",       # demand bawah
    "candle_range_atr_ratio", # (high-low)/ATR — normalized range expansion

    # Funding dynamics
    "funding_rate_change_8h",    # funding.diff(8) — momentum funding
    "funding_rate_zscore_20d",   # z-score funding vs 20d

    # Coinank — dihitung on-the-fly dari data coinank
    "ls_pos_zscore_20d",         # top_trader_position_ls zscore 20d
    "smart_retail_div_delta_1d", # (ls_pos - ls_acc).diff(24)
    "oi_pct_1d",                 # oi_binance.pct_change(24)
    "oi_price_div_1d",           # oi_pct_1d - close.pct_change(24)
]


# ═══════════════════════════════════════════════════════════════════════════════
# IC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def standalone_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 100:
        return 0.0
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def tstat_ic(ic: float, n: int) -> float:
    n_eff = max(n // AUTOCORR_FACTOR, 10)
    denom = np.sqrt(max(1.0 - ic ** 2, 1e-10))
    return ic * np.sqrt(n_eff) / denom


def _rank_norm(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    if mask.sum() < 10:
        return np.zeros_like(x, dtype=np.float64)
    out = np.zeros(len(x), dtype=np.float64)
    r = stats.rankdata(x[mask]).astype(np.float64)
    r -= r.mean()
    std = r.std()
    out[mask] = r / std if std > 1e-10 else 0.0
    return out


def _project_out(vec: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    norm_sq = np.dot(pivot, pivot)
    if norm_sq < 1e-10:
        return vec.copy()
    return vec - (np.dot(vec, pivot) / norm_sq) * pivot


def gram_schmidt_marginal(X_raw: np.ndarray, y_raw: np.ndarray,
                          feature_names: list, max_samples: int = 50000) -> dict:
    n, p = X_raw.shape
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        X_raw = X_raw[idx]
        y_raw = y_raw[idx]
        n = max_samples
        logger.info(f"  Gram-Schmidt: downsampled to {max_samples:,}")

    X = X_raw.copy().astype(np.float64)
    for j in range(p):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col

    X_r = np.column_stack([_rank_norm(X[:, j]) for j in range(p)])
    y_r = _rank_norm(y_raw.astype(np.float64))

    remaining = list(range(p))
    marginal = {}

    for _ in range(p):
        if not remaining:
            break
        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]
            nx = np.sqrt(np.dot(xj, xj))
            ny = np.sqrt(np.dot(y_r, y_r))
            corrs[k] = np.dot(xj, y_r) / (nx * ny) if nx > 1e-10 and ny > 1e-10 else 0.0

        best_k = int(np.argmax(np.abs(corrs)))
        best_j = remaining[best_k]
        marginal[feature_names[best_j]] = float(corrs[best_k])

        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j:
                X_r[:, j] = _project_out(X_r[:, j], pivot)
        y_r = _project_out(y_r, pivot)
        remaining.remove(best_j)

    return marginal


def make_verdict(sa: float, ts: float, mg: float) -> str:
    sa_pass = abs(sa) >= MIN_STANDALONE_IC and abs(ts) >= MIN_TSTAT
    mg_pass = abs(mg) >= MIN_MARGINAL_IC
    if sa_pass and mg_pass:
        return "KEEP"
    elif sa_pass and not mg_pass:
        return "REDUNDANT"
    elif not sa_pass and mg_pass:
        return "WEAK"
    else:
        return "DROP"


def run_ic_decay(df: pd.DataFrame, features: list, target_col: str) -> dict:
    results = {}
    for feat in features:
        if feat not in df.columns:
            continue
        window_ics = []
        for w_name, w_start, w_end in DECAY_WINDOWS:
            mask = (df.index >= w_start) & (df.index <= w_end)
            w_df = df.loc[mask]
            if len(w_df) < 300:
                window_ics.append({"window": w_name, "ic": None, "n": len(w_df)})
                continue
            ic = standalone_ic(w_df[feat].values, w_df[target_col].values)
            window_ics.append({"window": w_name, "ic": round(ic, 4), "n": len(w_df)})

        valid = [w["ic"] for w in window_ics if w["ic"] is not None]
        if len(valid) >= 3:
            ic_mean = np.mean(valid)
            ic_std  = np.std(valid)
            ic_ir   = abs(ic_mean) / ic_std if ic_std > 1e-10 else 0.0
            sign_cons = sum(1 for v in valid if (v >= 0) == (ic_mean >= 0))
            n_valid   = len(valid)
            is_stable = (ic_ir >= IC_STABILITY_IR_THRESH and
                         sign_cons >= min(IC_STABILITY_SIGN_THRESH, n_valid))
        else:
            ic_mean, ic_std, ic_ir = (valid[0] if valid else 0.0), 0.0, 0.0
            sign_cons, n_valid, is_stable = len(valid), len(valid), False

        results[feat] = {
            "ic_mean":         round(float(ic_mean), 4),
            "ic_std":          round(float(ic_std), 4),
            "ic_ir":           round(float(ic_ir), 2),
            "sign_consistency": sign_cons,
            "n_windows_valid": n_valid,
            "is_stable":       is_stable,
            "windows":         window_ics,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — fitur baru yang belum di parquet
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute new candidate features in-place."""
    hi, lo, cl, op = df["high"], df["low"], df["close"], df["open"]
    rng = (hi - lo).clip(lower=1e-10)

    df["candle_body_ratio"]      = (cl - op).abs() / rng
    df["upper_wick_ratio"]       = (hi - np.maximum(cl, op)) / rng
    df["lower_wick_ratio"]       = (np.minimum(cl, op) - lo) / rng
    df["candle_range_atr_ratio"] = rng / df["atr_14_h1"].clip(lower=1e-10)

    if "funding_rate" in df.columns:
        fr = df["funding_rate"]
        df["funding_rate_change_8h"] = fr.diff(8)
        roll_mean = fr.rolling(24 * 20, min_periods=100).mean()
        roll_std  = fr.rolling(24 * 20, min_periods=100).std().clip(lower=1e-10)
        df["funding_rate_zscore_20d"] = (fr - roll_mean) / roll_std

    return df


def add_coinank(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Load coinank data dan compute derived features. NaN jika tidak ada."""
    idx = df.index
    oi_p  = COINANK_DIR / f"{sym}_oi.parquet"
    lsp_p = COINANK_DIR / f"{sym}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{sym}_ls_account.parquet"

    if oi_p.exists():
        oi = pd.read_parquet(oi_p).sort_index()
        col = next((c for c in ["oi_binance", "oi"] if c in oi.columns), None)
        if col:
            oi_s = oi[col].reindex(idx, method="ffill")
            df["oi_pct_1d"]       = oi_s.pct_change(24)
            df["oi_price_div_1d"] = oi_s.pct_change(24) - df["close"].pct_change(24)

    ls_pos = None
    if lsp_p.exists():
        lsp = pd.read_parquet(lsp_p).sort_index()
        col = next((c for c in ["top_trader_position_ls", "ls_position"] if c in lsp.columns), None)
        if col:
            ls_pos = lsp[col].reindex(idx, method="ffill")
            lp_mean = ls_pos.rolling(24 * 20, min_periods=100).mean()
            lp_std  = ls_pos.rolling(24 * 20, min_periods=100).std().clip(lower=1e-8)
            df["ls_pos_zscore_20d"] = (ls_pos - lp_mean) / lp_std

    if lsa_p.exists() and ls_pos is not None:
        lsa = pd.read_parquet(lsa_p).sort_index()
        col = next((c for c in ["top_trader_account_ls", "ls_account"] if c in lsa.columns), None)
        if col:
            ls_acc = lsa[col].reindex(idx, method="ffill")
            df["smart_retail_div_delta_1d"] = (ls_pos - ls_acc).diff(24)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(coins: list) -> pd.DataFrame:
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{sym}] Parquet tidak ditemukan, skip")
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        req = ["close", "high", "low", "open", "atr_14_h1"]
        if any(c not in df.columns for c in req):
            logger.warning(f"[{sym}] Missing required cols, skip")
            continue

        # TB labels
        tb = triple_barrier_labeling(
            df["close"], df["high"], df["low"], df["atr_14_h1"],
            TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS
        )
        df["tb_label"] = tb.map(TB_LABEL_MAP)
        df = df.dropna(subset=["tb_label"])
        if len(df) < 100:
            continue

        # Engineer new features
        df = engineer_new_features(df)
        df = add_coinank(df, sym)
        df["coin"] = sym
        frames.append(df)

        n    = len(df)
        dist = df["tb_label"].value_counts()
        coinank_n = df.get("ls_pos_zscore_20d", pd.Series(dtype=float)).notna().sum()
        logger.info(
            f"  [{sym}] {n:,} bars | "
            f"LONG={dist.get(2,0)/n*100:.1f}% SHORT={dist.get(0,0)/n*100:.1f}% "
            f"FLAT={dist.get(1,0)/n*100:.1f}% | coinank={coinank_n:,} bars"
        )

    if not frames:
        raise RuntimeError("Tidak ada data!")
    return pd.concat(frames).sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true", help="Semua koin")
    parser.add_argument("--coins", nargs="+", default=None)
    args = parser.parse_args()

    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS)

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  IC FEATURE SELECTION — {RUN_NAME}")
    print(f"  Candidate: FEATURE_COLS_V3 + {len(NEW_FEATURES)} new features")
    print(f"  TB: TP={TP_SL_FALLBACK_TP}xATR  SL={TP_SL_FALLBACK_SL}xATR  MaxHold={MAX_HOLDING_BARS}")
    print(f"  Coins: {len(coins)}")
    print(f"{sep}\n")

    # ── Load + Label ────────────────────────────────────────────────────────────
    print("STAGE 1: Loading data + TB labeling + new feature engineering...")
    print("-" * 50)
    df = load_data(coins)

    df["tb_label_ord"] = df["tb_label"].map({0: -1, 1: 0, 2: 1})
    target = df["tb_label_ord"].values
    n_total = len(df)
    n_eff   = n_total // AUTOCORR_FACTOR

    dist = df["tb_label"].value_counts().to_dict()
    label_names = {0: "SHORT", 1: "FLAT", 2: "LONG"}
    print(f"\n  Rows: {n_total:,} | Effective N: {n_eff:,}")
    print(f"  Labels: {', '.join(f'{label_names[k]}={v:,}({v/n_total*100:.1f}%)' for k, v in sorted(dist.items()))}")

    # Coinank coverage
    for f in ["ls_pos_zscore_20d", "oi_pct_1d", "smart_retail_div_delta_1d"]:
        if f in df.columns:
            nn = df[f].notna().sum()
            print(f"  {f}: {nn:,} non-NaN ({nn/n_total*100:.1f}%)")

    # ── Build candidate list ────────────────────────────────────────────────────
    meta_cols = {"label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
                 "hmm_regime", "hmm_regime_enc", "tb_label", "tb_label_ord"}

    existing_candidates = [c for c in FEATURE_COLS_V3
                           if c in df.columns and c not in meta_cols]
    new_avail = [c for c in NEW_FEATURES if c in df.columns]
    all_candidates = existing_candidates + [c for c in new_avail
                                            if c not in existing_candidates]

    print(f"\n  Existing features (FEATURE_COLS_V3 available): {len(existing_candidates)}")
    print(f"  New features available: {len(new_avail)} / {len(NEW_FEATURES)}")
    missing_new = [c for c in NEW_FEATURES if c not in df.columns]
    if missing_new:
        print(f"  Missing new: {missing_new}")
    print(f"  Total candidates: {len(all_candidates)}")
    print(f"  Thresholds: |SA|>={MIN_STANDALONE_IC}, |t|>={MIN_TSTAT}, |MG|>={MIN_MARGINAL_IC}")

    # ── Stage 2a: Standalone IC ─────────────────────────────────────────────────
    print(f"\nSTAGE 2a: Standalone IC ({len(all_candidates)} features)...")
    print("-" * 50)

    standalone = {}
    tstats = {}
    for feat in all_candidates:
        ic = standalone_ic(df[feat].values, target)
        standalone[feat] = ic
        tstats[feat] = tstat_ic(ic, n_total)

    # Urutkan berdasarkan |IC| untuk display
    sorted_sa = sorted(all_candidates, key=lambda f: abs(standalone[f]), reverse=True)
    print(f"\n  {'Feature':<35} {'IC':>8} {'t-stat':>8} {'Status'}")
    print(f"  {'-'*35} {'--------':>8} {'--------':>8} {'------'}")
    for feat in sorted_sa[:30]:
        ic = standalone[feat]
        ts = tstats[feat]
        tag = ""
        if feat in TB_V3_FEATURES:
            tag = " [v3]"
        elif feat in NEW_FEATURES:
            tag = " [NEW]"
        sa_ok = abs(ic) >= MIN_STANDALONE_IC and abs(ts) >= MIN_TSTAT
        status = "PASS" if sa_ok else "fail"
        print(f"  {feat:<35} {ic:>+8.4f} {ts:>+8.2f} {status}{tag}")

    # ── Stage 2b: Marginal IC (Gram-Schmidt) ────────────────────────────────────
    print(f"\nSTAGE 2b: Marginal IC via Gram-Schmidt ({len(all_candidates)} features)...")
    print("-" * 50)

    X_mat = df[all_candidates].values
    marginal = gram_schmidt_marginal(X_mat, target, all_candidates)

    # ── Verdict ─────────────────────────────────────────────────────────────────
    ic_results = []
    for feat in all_candidates:
        sa = standalone[feat]
        ts = tstats[feat]
        mg = marginal.get(feat, 0.0)
        v  = make_verdict(sa, ts, mg)
        ic_results.append({
            "feature":       feat,
            "standalone_ic": round(sa, 4),
            "tstat":         round(ts, 2),
            "marginal_ic":   round(mg, 4),
            "verdict":       v,
            "is_new":        feat in NEW_FEATURES,
            "in_v3":         feat in TB_V3_FEATURES,
        })

    verdict_order = {"KEEP": 0, "REDUNDANT": 1, "WEAK": 2, "DROP": 3}
    ic_results.sort(key=lambda x: (verdict_order[x["verdict"]], -abs(x["standalone_ic"])))

    verdicts = [r["verdict"] for r in ic_results]
    summary = {v: verdicts.count(v) for v in ["KEEP", "REDUNDANT", "WEAK", "DROP"]}
    keep_features = [r["feature"] for r in ic_results if r["verdict"] == "KEEP"]
    new_keep = [r for r in ic_results if r["verdict"] == "KEEP" and r["is_new"]]
    existing_keep = [r for r in ic_results if r["verdict"] == "KEEP" and not r["is_new"]]

    print(f"\n  Summary: KEEP={summary['KEEP']} | REDUNDANT={summary['REDUNDANT']} | "
          f"WEAK={summary['WEAK']} | DROP={summary['DROP']}")

    print(f"\n  KEEP — existing (not in v3 deployed): "
          f"{sum(1 for r in existing_keep if not r['in_v3'])}")
    for r in existing_keep:
        if not r["in_v3"]:
            print(f"    {r['standalone_ic']:+.4f} (mg={r['marginal_ic']:+.4f})  {r['feature']}")

    if new_keep:
        print(f"\n  KEEP — NEW features: {len(new_keep)}")
        for r in new_keep:
            print(f"    {r['standalone_ic']:+.4f} (mg={r['marginal_ic']:+.4f})  {r['feature']}")

    print(f"\n  KEEP — already in v3 deployed: {sum(1 for r in existing_keep if r['in_v3'])}")

    # ── Stage 2c: IC Decay Stability ────────────────────────────────────────────
    print(f"\nSTAGE 2c: IC Decay Stability ({len(keep_features)} KEEP features)...")
    print("-" * 50)
    decay_results = run_ic_decay(df, keep_features, "tb_label_ord")

    print(f"\n  {'Feature':<35} {'IC_IR':>7} {'Sign':>5} {'Stable':>7} {'Source'}")
    print(f"  {'-'*35} {'-------':>7} {'-----':>5} {'-------':>7} {'------'}")
    for feat in keep_features:
        d = decay_results.get(feat, {})
        ir  = d.get("ic_ir", 0.0)
        sc  = d.get("sign_consistency", 0)
        nv  = d.get("n_windows_valid", 0)
        stb = "STABLE" if d.get("is_stable") else "unstbl"
        src = "v3" if feat in TB_V3_FEATURES else ("NEW" if feat in NEW_FEATURES else "exist")
        print(f"  {feat:<35} {ir:>7.2f} {sc:>2}/{nv:<2} {stb:>7} [{src}]")

    # Final: KEEP + STABLE
    final_features = [f for f in keep_features
                      if decay_results.get(f, {}).get("is_stable", False)]

    unstable_keep = [f for f in keep_features
                     if not decay_results.get(f, {}).get("is_stable", False)]

    print(f"\n  FINAL (KEEP + STABLE): {len(final_features)}")
    print(f"  Dropped (unstable):    {len(unstable_keep)}")
    if unstable_keep:
        for f in unstable_keep:
            d = decay_results.get(f, {})
            print(f"    [unstable] {f}: IC_IR={d.get('ic_ir',0):.2f} sign={d.get('sign_consistency',0)}/{d.get('n_windows_valid',0)}")

    # Diff vs v3 deployed
    added   = [f for f in final_features if f not in TB_V3_FEATURES]
    removed = [f for f in TB_V3_FEATURES if f not in final_features]

    print(f"\n  vs TB v3 deployed (18 feat):")
    print(f"  + Added ({len(added)}): {added}")
    print(f"  - Removed ({len(removed)}): {removed}")

    # ── Save ────────────────────────────────────────────────────────────────────
    print(f"\nSaving to {run_dir}...")

    ic_out = {
        "run_name":    RUN_NAME,
        "n_rows":      n_total,
        "n_eff":       n_eff,
        "thresholds":  {"min_standalone_ic": MIN_STANDALONE_IC,
                        "min_tstat": MIN_TSTAT,
                        "min_marginal_ic": MIN_MARGINAL_IC},
        "summary":     summary,
        "keep_features": keep_features,
        "results":     ic_results,
    }
    with open(run_dir / f"{RUN_NAME}_ic_results.json", "w") as f:
        json.dump(ic_out, f, indent=2, default=str)

    with open(run_dir / f"{RUN_NAME}_ic_decay.json", "w") as f:
        json.dump(decay_results, f, indent=2, default=str)

    recommended = {
        "run_name":       RUN_NAME,
        "final_features": final_features,
        "n_final":        len(final_features),
        "vs_v3": {
            "added":   added,
            "removed": removed,
            "same":    [f for f in final_features if f in TB_V3_FEATURES],
        },
        "new_features_result": {
            r["feature"]: {"verdict": r["verdict"],
                           "standalone_ic": r["standalone_ic"],
                           "tstat": r["tstat"]}
            for r in ic_results if r["is_new"]
        },
    }
    with open(run_dir / f"{RUN_NAME}_recommended.json", "w") as f:
        json.dump(recommended, f, indent=2)

    print(f"  -> {run_dir / f'{RUN_NAME}_ic_results.json'}")
    print(f"  -> {run_dir / f'{RUN_NAME}_ic_decay.json'}")
    print(f"  -> {run_dir / f'{RUN_NAME}_recommended.json'}")

    print(f"\n{sep}")
    print(f"  IC FEATURE SELECTION COMPLETE — {RUN_NAME}")
    print(f"  Candidates tested: {len(all_candidates)}")
    print(f"  KEEP: {summary['KEEP']} | FINAL (KEEP+STABLE): {len(final_features)}")
    print(f"  New features KEEP: {len(new_keep)}")
    print(f"  Added vs v3: {added}")
    print(f"  Removed vs v3: {removed}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
