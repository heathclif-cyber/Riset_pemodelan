"""
pipeline/06d_train_guardian_genuine_v2_hmm_v2.py -- Guardian Genuine OOF (HMM Config B, labeling v2)

KONTROL GENUINE (wajib):
  1. Entry signal dari OOF LGBM (has_oof=True) -- bukan in-sample model
  2. Trades dihasilkan simulate_trades_swing pada OOF signals (guardian_enabled=False)
  3. Feature parquet < TRAIN_CUTOFF_DATE only
  4. Guardian CV: scaler StandardScaler per fold (fit train, transform val only)
  5. Purged CV GUARDIAN_N_FOLDS=8, purge=GUARDIAN_PURGE_GAP_BARS=36
  6. HMM threshold Config B frozen (bukan dari holdout)
  7. Holdout TIDAK dievaluasi
  8. Promote ke guardian_best.pkl HANYA dengan --promote + audit PASS

Guardian Retrain: Labeling Lebih Ketat

Masalah di 06c (v1):
  GUARDIAN_EXIT terlalu agresif — 7,819 trade keluar di avg +$0.441 padahal
  tanpa Guardian rata-rata WIN = +$1.278. Guardian exit terlalu dini.

  Root cause: rule r3 dan profit_lock terlalu sensitif:
    - profit_lock: mfe>0.02, current>0, current<mfe*0.40  → terlalu mudah trigger
    - r3:          mfe>0.015, current<mfe*0.25             → exit saat masih sedikit profit
    - r5 PARTIAL:  mfe>0.015, current<mfe*0.55             → terlalu lebar

Perubahan labeling v2:
  1. profit_lock: mfe>0.03 (naik), current>0.003 (harus cukup profit), ratio<0.30 (turun)
  2. r3:          mfe>0.025 (naik), current>0.001 (wajib masih profit), ratio<0.20 (turun)
  3. r5 PARTIAL:  mfe>0.025 (naik), ratio<0.40 (turun dari 0.55)
  Lainnya tetap sama.

Prinsip: Guardian baru lebih "sabar" — hanya exit ketika sinyal reversal sangat jelas.

Entry gate, static/dynamic features, CV fold: identik dengan 06c.
Output: models/runs/tb_guardian_genuine_v2_hmm_v2/
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE,
    GUARDIAN_LGBM_PARAMS, GUARDIAN_EARLY_STOPPING,
    GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS, GUARDIAN_MIN_HOLD_BARS,
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, LABEL_DIR,
)
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from pipeline.shared import build_purged_folds

logger = setup_logger("06d_train_guardian_v2_hmm_v2")

LGBM_RUN_NAME   = "tb_lgbm_genuine_v2"
BASELINE_RUN    = "tb_guardian_genuine_v2_hmm_v2"
RUN_NAME        = "tb_guardian_genuine_v2_hmm_v2"
OUT_DIR         = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_LGBM_DOW = {"dow_cos", "dow_sin"}
MIN_SAMPLES       = 100_000
MAX_EXIT2_NEG_PCT = 10.0
MIN_CV_F1_MACRO   = 0.80

HMM_THR_CFG = {
    0:  (0.55, 0.55),
    1:  (0.55, 0.55),
    2:  (0.50, 0.50),
    3:  (0.45, 0.50),
    -1: (0.45, 0.45),
}

STATIC_FEATS = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4", "ofi_h4_delta",
    "wyckoff_phase", "Sell_Liq", "atr_percentile_h1", "stochrsi_k",
    "dist_liq_50x_short", "funding_rate", "ema_7_h1", "dow_cos",
    "cvd_div_h4", "dist_swing_low", "VAH", "cvd_momentum_adv",
    "dist_from_8h_high", "ema_200_h1",
]
DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]


def assert_oof_source_genuine(lgbm_run_dir: Path) -> dict:
    """Audit: OOF dari LGBM genuine, fitur entry sudah include dow."""
    oof_path = lgbm_run_dir / "oof_predictions.parquet"
    feat_path = lgbm_run_dir / "features.json"
    thr_path  = lgbm_run_dir / "best_thresholds.json"
    for p in (oof_path, feat_path, thr_path):
        if not p.exists():
            raise FileNotFoundError(f"Artefak LGBM genuine tidak lengkap: {p}")

    with open(feat_path, encoding="utf-8") as f:
        lgbm_feats = set(json.load(f))
    missing_dow = REQUIRED_LGBM_DOW - lgbm_feats
    if missing_dow:
        raise RuntimeError(
            f"LGBM OOF source belum +dow: missing {missing_dow}. "
            "Jalankan 04e_train_lgbm_genuine_v2_dow.py --promote dulu."
        )

    oof = pd.read_parquet(oof_path)
    if "has_oof" not in oof.columns:
        raise RuntimeError("oof_predictions.parquet tidak punya kolom has_oof")
    n_oof = int((oof["has_oof"] == True).sum())
    if n_oof < 500_000:
        raise RuntimeError(f"OOF coverage terlalu rendah: {n_oof:,} bars")

    with open(thr_path, encoding="utf-8") as f:
        thr = json.load(f)
    if thr.get("holdout_used"):
        raise RuntimeError("LEAKAGE: best_thresholds.json holdout_used=True")

    return {
        "lgbm_run": lgbm_run_dir.name,
        "n_lgbm_features": len(lgbm_feats),
        "dow_features_present": sorted(REQUIRED_LGBM_DOW),
        "oof_bars": n_oof,
        "threshold_method": thr.get("sweep_method", "unknown"),
        "holdout_used_for_threshold": False,
    }


def assert_genuine_sample_bounds(samples_df: pd.DataFrame) -> dict:
    if len(samples_df) < MIN_SAMPLES:
        raise RuntimeError(f"Terlalu sedikit Guardian samples: {len(samples_df):,} < {MIN_SAMPLES:,}")
    return {"n_samples": len(samples_df), "min_samples_ok": True}


def check_promotion_gate(mean_f1: float, exit2_neg_pct: float, n_samples: int) -> dict:
    f1_ok   = bool(mean_f1 >= MIN_CV_F1_MACRO)
    neg_ok  = bool(exit2_neg_pct <= MAX_EXIT2_NEG_PCT)
    n_ok    = bool(n_samples >= MIN_SAMPLES)
    passed  = bool(f1_ok and neg_ok and n_ok)
    return {
        "f1_ok": f1_ok, "exit2_neg_ok": neg_ok, "n_samples_ok": n_ok,
        "passed": passed,
        "mean_f1_macro": round(mean_f1, 4),
        "exit2_neg_pnl_pct": round(exit2_neg_pct, 1),
        "n_samples": n_samples,
        "thresholds": {
            "min_cv_f1_macro": MIN_CV_F1_MACRO,
            "max_exit2_neg_pct": MAX_EXIT2_NEG_PCT,
            "min_samples": MIN_SAMPLES,
        },
    }


def maybe_promote_guardian(final_model, final_scaler, feat_cols: list, gate: dict) -> bool:
    if not gate["passed"]:
        logger.warning("Guardian promotion gate FAIL -- tidak promote ke models/")
        return False

    joblib.dump(final_model,  MODEL_DIR / "guardian_best.pkl")
    joblib.dump(final_scaler, MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2)
    logger.info("Guardian promotion gate PASS -- promoted to models/guardian_best.pkl")
    return True


def _apply_hmm_threshold(p0, p2, hmm_enc):
    n = len(p0)
    default_tl, default_ts = HMM_THR_CFG[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1:
            continue
        tl_arr[hmm_enc == state] = tl
        ts_arr[hmm_enc == state] = ts
    long_mask  = p2 >= tl_arr
    short_mask = (p0 >= ts_arr) & ~long_mask
    y = np.ones(n, dtype=np.int32)
    y[long_mask]  = 2
    y[short_mask] = 0
    return y


def generate_oof_samples_for_coin(sym: str, oof_pred_df: pd.DataFrame) -> list:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if df.empty:
        return []

    sym_oof       = oof_pred_df[oof_pred_df["coin"] == sym]
    sym_oof       = sym_oof[sym_oof["has_oof"] == True]
    sym_oof_proba = sym_oof[["p0", "p1", "p2"]].reindex(df.index)
    has_oof       = sym_oof_proba["p0"].notna()
    df_oof        = df[has_oof].copy()
    sym_oof_proba = sym_oof_proba[has_oof]
    n             = len(df_oof)
    if n < 30:
        return []

    p0 = sym_oof_proba["p0"].values.astype(np.float32)
    p2 = sym_oof_proba["p2"].values.astype(np.float32)
    hmm_enc = (
        df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
        if "hmm_regime_enc" in df_oof.columns
        else np.full(n, -1, np.int8)
    )

    y_pred = _apply_hmm_threshold(p0, p2, hmm_enc)
    if (y_pred != 1).sum() == 0:
        return []

    close_arr = df_oof["close"].values
    high_arr  = df_oof["high"].values
    low_arr   = df_oof["low"].values
    atr_arr   = df_oof["atr_14_h1"].values
    h4_sh = df_oof["h4_swing_high"].values if "h4_swing_high" in df_oof.columns \
            else np.full(n, np.nan)
    h4_sl = df_oof["h4_swing_low"].values if "h4_swing_low" in df_oof.columns \
            else np.full(n, np.nan)

    result = simulate_trades_swing(
        y_pred=y_pred,
        close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )
    trades = result.get("trades", [])
    if not trades:
        return []

    g_static_cols = [c for c in STATIC_FEATS if c in df_oof.columns]
    X_static = np.zeros((n, len(g_static_cols)), dtype=np.float64)
    for idx, col in enumerate(g_static_cols):
        X_static[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)

    samples = []
    rule_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, "profit_lock": 0}

    for t in trades:
        bar_in      = t["bar_in"]
        bar_out     = t["bar_out"]
        direction   = 2 if t["direction"] == "LONG" else 0
        entry_price = t["entry"]
        atr_entry   = atr_arr[bar_in] if bar_in < n else 0.01

        if bar_out <= bar_in + 1:
            continue

        for j in range(bar_in + 1, min(bar_out, n)):
            cp = close_arr[j]
            if np.isnan(cp):
                continue

            bars_held = j - bar_in
            current_pnl = (cp - entry_price) / entry_price if direction == 2 \
                          else (entry_price - cp) / entry_price

            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]):
                    continue
                mfe_sofar = max(mfe_sofar,
                    (close_arr[k] - entry_price) / entry_price if direction == 2
                    else (entry_price - close_arr[k]) / entry_price)

            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close_arr[k]):
                    continue
                best_future_pnl = max(best_future_pnl,
                    (close_arr[k] - entry_price) / entry_price if direction == 2
                    else (entry_price - close_arr[k]) / entry_price)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl \
                           if best_future_pnl > 0.001 else 0.0

            # ── Labeling v2: lebih ketat pada EXIT sebelum TP ─────────────
            label = None

            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0   # wajib HOLD di awal

            # profit_lock v2: MFE ≥ 3% (naik dari 2%), still in profit ≥ 0.3%,
            # sudah jatuh ke < 30% of peak (lebih ketat dari 40%)
            elif mfe_sofar > 0.030 and current_pnl > 0.003 and current_pnl < mfe_sofar * 0.30:
                label = 2; rule_counts["profit_lock"] += 1

            # r3 v2: MFE ≥ 2.5% (naik dari 1.5%), wajib masih profit ≥ 0.1%,
            # sudah jatuh ke < 20% of peak (lebih ketat dari 25%)
            elif mfe_sofar > 0.025 and current_pnl > 0.001 and current_pnl < mfe_sofar * 0.20:
                label = 2; rule_counts[3] += 1

            # r4: near-peak exit (future tidak lebih baik) — tetap sama
            elif current_pnl >= best_future_pnl * 0.95 and current_pnl > 0.005:
                label = 2; rule_counts[4] += 1

            # r5 PARTIAL v2: MFE ≥ 2.5% (naik dari 1.5%), ratio < 40% (turun dari 55%)
            elif mfe_sofar > 0.025 and current_pnl < mfe_sofar * 0.40:
                label = 1; rule_counts[5] += 1

            # r6 PARTIAL: upside sangat kecil — tetap sama
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1; rule_counts[6] += 1

            # r7 HOLD: future lebih baik — tetap sama
            elif best_future_pnl > current_pnl * 1.05:
                label = 0; rule_counts[7] += 1

            else:
                continue

            atr_pct         = atr_entry / entry_price if entry_price > 0 else 0.01
            bars_held_norm  = bars_held / MAX_HOLDING_BARS
            current_pnl_atr = current_pnl / atr_pct if atr_pct > 0 else 0.0
            dd_from_peak    = (mfe_sofar - current_pnl) / mfe_sofar \
                              if mfe_sofar > 0.001 else 0.0
            entry_ratio     = entry_price / cp if cp > 0 else 1.0

            sample = {
                **{c: float(X_static[j, idx]) for idx, c in enumerate(g_static_cols)},
                "bars_held_norm":         bars_held_norm,
                "current_pnl_pct":        current_pnl,
                "current_pnl_atr":        current_pnl_atr,
                "max_favorable_pnl_pct":  mfe_sofar,
                "drawdown_from_peak_pct": dd_from_peak,
                "direction":              1.0 if direction == 2 else 0.0,
                "entry_price_ratio":      entry_ratio,
                "label":                  label,
            }
            samples.append(sample)

    logger.info(
        f"[{sym}] {len(trades)} trades -> {len(samples)} samples | "
        f"pl={rule_counts['profit_lock']} r3={rule_counts[3]} r4={rule_counts[4]} "
        f"r5={rule_counts[5]} r6={rule_counts[6]} hold={rule_counts[7]}"
    )
    return samples


def train_guardian(samples_df: pd.DataFrame):
    g_static  = [c for c in samples_df.columns if c not in set(DYNAMIC_FEATS) and c != "label"]
    feat_cols = g_static + DYNAMIC_FEATS
    X_all     = samples_df[feat_cols].values.astype(np.float64)
    y_all     = samples_df["label"].values.astype(np.int64)

    n0, n1, n2 = (y_all == 0).sum(), (y_all == 1).sum(), (y_all == 2).sum()
    pct_neg = (samples_df.loc[samples_df["label"] == 2, "current_pnl_pct"] < 0).mean() * 100
    logger.info(f"Total: {len(samples_df):,} | HOLD={n0:,} PARTIAL={n1:,} EXIT={n2:,}")
    logger.info(f"EXIT-2 saat PnL<0: {pct_neg:.1f}%  (v1 was 26.9%, target <15%)")
    if pct_neg > 10:
        logger.warning(f"EXIT-2 at negative PnL: {pct_neg:.1f}%")

    folds      = build_purged_folds(samples_df.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)
    cv_results = []
    best_ll    = float("inf")
    best_iters = None

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if len(val_idx) < 10:
            continue
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_all[train_idx])
        X_te = scaler_fold.transform(X_all[val_idx])
        y_tr, y_te = y_all[train_idx], y_all[val_idx]

        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING, verbose=False),
                              lgb.log_evaluation(period=0)])

        y_prob = model.predict_proba(X_te)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        ll = -np.mean(np.log(y_prob[np.arange(len(y_te)), y_te] + 1e-10))
        cv_results.append({"fold": fold_idx+1, "n_train": len(train_idx),
                           "n_val": len(val_idx), "logloss": round(ll, 4),
                           "f1_macro": round(f1, 4), "best_iter": model.best_iteration_})
        logger.info(f"  Fold {fold_idx+1}: logloss={ll:.4f} f1={f1:.4f} iter={model.best_iteration_}")
        if ll < best_ll:
            best_ll = ll; best_iters = model.best_iteration_

    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X_all)
    fp = {**GUARDIAN_LGBM_PARAMS}
    if best_iters: fp["n_estimators"] = best_iters
    final_model = lgb.LGBMClassifier(**fp)
    final_model.fit(X_all_s, y_all)
    logger.info(f"Final Guardian: {fp['n_estimators']} iter")
    return final_model, final_scaler, feat_cols, cv_results, best_ll, int(n0), int(n1), int(n2), pct_neg


def main():
    parser = argparse.ArgumentParser(description="Guardian genuine OOF retrain")
    parser.add_argument(
        "--promote", action="store_true",
        help="Promote ke models/guardian_best.pkl HANYA jika audit + quality gate PASS",
    )
    args = parser.parse_args()

    lgbm_run_dir = MODEL_DIR / "runs" / LGBM_RUN_NAME
    oof_audit    = assert_oof_source_genuine(lgbm_run_dir)
    oof_path     = lgbm_run_dir / "oof_predictions.parquet"
    oof_pred_df  = pd.read_parquet(oof_path)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GUARDIAN GENUINE OOF RETRAIN -- {RUN_NAME}")
    print(f"  LGBM OOF source : {LGBM_RUN_NAME} ({oof_audit['n_lgbm_features']} feat, dow OK)")
    print(f"  OOF bars        : {oof_audit['oof_bars']:,}")
    print(f"  Holdout         : TIDAK DIJALANKAN (sealed)")
    print(f"  Promote flag    : {args.promote}")
    print(f"  Run: {RUN_NAME}")
    print(f"")
    print(f"  Perubahan labeling vs v1 (06c):")
    print(f"    profit_lock: mfe>0.03 (was 0.02), pnl>0.003, ratio<0.30 (was 0.40)")
    print(f"    r3:          mfe>0.025 (was 0.015), pnl>0.001 (BARU), ratio<0.20 (was 0.25)")
    print(f"    r5 PARTIAL:  mfe>0.025 (was 0.015), ratio<0.40 (was 0.55)")
    print(f"  Entry gate: HMM Config B (identik dengan v1)")
    print(f"{sep}\n")

    print("[1/3] Generating samples (HMM-gated OOF trades, labeling v2)...")
    print("-" * 55)
    all_samples = []
    for sym in ALL_COINS:
        smp = generate_oof_samples_for_coin(sym, oof_pred_df)
        all_samples.extend(smp)

    if len(all_samples) < 100:
        raise RuntimeError(f"Terlalu sedikit samples ({len(all_samples)})")

    samples_df = pd.DataFrame(all_samples).reset_index(drop=True)
    sample_audit = assert_genuine_sample_bounds(samples_df)
    n0 = int((samples_df["label"] == 0).sum())
    n1 = int((samples_df["label"] == 1).sum())
    n2 = int((samples_df["label"] == 2).sum())
    pct_neg = (samples_df.loc[samples_df["label"] == 2, "current_pnl_pct"] < 0).mean() * 100

    print(f"\n  Total samples : {len(samples_df):,}  (v1 was 246,079)")
    print(f"  HOLD (0)      : {n0:,} ({n0/len(samples_df)*100:.1f}%)")
    print(f"  PARTIAL (1)   : {n1:,} ({n1/len(samples_df)*100:.1f}%)")
    print(f"  EXIT (2)      : {n2:,} ({n2/len(samples_df)*100:.1f}%)")
    print(f"  EXIT-2 PnL<0  : {pct_neg:.1f}%  (v1 was 26.9%, target <15%)")

    print(f"\n[2/3] Training Guardian v2...")
    print("-" * 55)
    final_model, final_scaler, feat_cols, cv_results, best_ll, n0, n1, n2, pct_neg = \
        train_guardian(samples_df)

    mean_ll = float(np.mean([r["logloss"] for r in cv_results]))
    mean_f1 = float(np.mean([r["f1_macro"] for r in cv_results]))
    print(f"\n  CV Mean logloss : {mean_ll:.4f}  (v1: 0.2184)")
    print(f"  CV Mean F1 macro: {mean_f1:.4f}  (v1: 0.8546)")

    print(f"\n[3/3] Saving to {OUT_DIR}...")
    joblib.dump(final_model,  OUT_DIR / "guardian.pkl")
    joblib.dump(final_scaler, OUT_DIR / "guardian_scaler.pkl")
    with open(OUT_DIR / "guardian_features.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    cv_out = {
        "run_name": RUN_NAME, "created": datetime.now().isoformat(),
        "lgbm_source": LGBM_RUN_NAME,
        "version": "v2 — tighter labeling (profit_lock, r3, r5)",
        "labeling_changes": {
            "profit_lock": "mfe>0.03 pnl>0.003 ratio<0.30  (was: mfe>0.02 pnl>0 ratio<0.40)",
            "r3":          "mfe>0.025 pnl>0.001 ratio<0.20 (was: mfe>0.015 NO_pnl_check ratio<0.25)",
            "r5_partial":  "mfe>0.025 ratio<0.40            (was: mfe>0.015 ratio<0.55)",
        },
        "hmm_thresholds": {str(k): list(v) for k, v in HMM_THR_CFG.items()},
        "n_samples": len(samples_df), "n_features": len(feat_cols),
        "features": feat_cols, "n_folds": GUARDIAN_N_FOLDS,
        "purge_bars": GUARDIAN_PURGE_GAP_BARS,
        "mean_logloss": round(mean_ll, 4), "best_logloss": round(best_ll, 4),
        "mean_f1_macro": round(mean_f1, 4), "folds": cv_results,
        "label_dist": {"HOLD": n0, "PARTIAL": n1, "EXIT": n2},
        "exit2_neg_pnl_pct": round(pct_neg, 1),
        "static_feats": STATIC_FEATS, "dynamic_feats": DYNAMIC_FEATS,
    }
    with open(OUT_DIR / "guardian_cv_results.json", "w") as f:
        json.dump(cv_out, f, indent=2)

    gate = check_promotion_gate(mean_f1, pct_neg, len(samples_df))
    genuine_audit = {
        "methodology": "guardian_genuine_oof_v1",
        "holdout_evaluated": False,
        "train_cutoff_enforced": True,
        "entry_signal": {
            "source": "OOF_LGBM_predictions",
            "lgbm_run": LGBM_RUN_NAME,
            "has_oof_filter": True,
            "guardian_enabled_during_trade_gen": False,
            "hmm_threshold": "Config B frozen",
            "holdout_used_for_entry": False,
        },
        "oof_source_audit": oof_audit,
        "sample_audit": sample_audit,
        "cv": {
            "n_folds": GUARDIAN_N_FOLDS,
            "purge_bars": GUARDIAN_PURGE_GAP_BARS,
            "scaler_per_fold": True,
            "labeling_version": "v2_tight",
        },
        "promotion_gate": gate,
        "promoted": False,
    }

    promoted = False
    if args.promote:
        promoted = maybe_promote_guardian(final_model, final_scaler, feat_cols, gate)
        genuine_audit["promoted"] = bool(promoted)
    else:
        print("  Promote: SKIP (--promote tidak diberikan)")

    with open(OUT_DIR / "genuine_audit.json", "w") as f:
        json.dump(genuine_audit, f, indent=2)

    print(f"\n  Promotion gate:")
    print(f"    F1>={MIN_CV_F1_MACRO}     : {gate['mean_f1_macro']} ({'PASS' if gate['f1_ok'] else 'FAIL'})")
    print(f"    EXIT2 neg<={MAX_EXIT2_NEG_PCT}% : {gate['exit2_neg_pnl_pct']}% ({'PASS' if gate['exit2_neg_ok'] else 'FAIL'})")
    print(f"    n_samples>={MIN_SAMPLES}: {gate['n_samples']:,} ({'PASS' if gate['n_samples_ok'] else 'FAIL'})")
    print(f"  Audit: {OUT_DIR / 'genuine_audit.json'}")

    print(f"\n{sep}")
    print(f"  DONE -- {RUN_NAME}")
    print(f"  Promoted: {promoted}")
    print(f"  CV Mean logloss : {mean_ll:.4f}  (v1: 0.2184)")
    print(f"  CV Mean F1 macro: {mean_f1:.4f}  (v1: 0.8546)")
    print(f"  Samples         : {len(samples_df):,}  (v1: 246,079)")
    print(f"  EXIT-2 PnL<0    : {pct_neg:.1f}%  (v1: 26.9%)")
    print(f"")
    print(f"  LANGKAH BERIKUTNYA:")
    print(f"  python pipeline/05f_eval_pipeline_with_guardian.py  (update GUARDIAN_RUN)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
