"""
pipeline/06c_train_guardian_genuine_v2_hmm.py -- Guardian Training dengan HMM-gated Entries

Sama dengan 06b kecuali:
  - Entry signal pakai HMM Config B per-state threshold (bukan flat 0.45/0.45)
  - Signal generation vectorized numpy (konsisten dengan 05e)
  - Output: tb_guardian_genuine_v2_hmm

HMM Config B:
  S0 TRENDING_DN  : thr_long=0.55, thr_short=0.55
  S1 RANGING_LOW  : thr_long=0.55, thr_short=0.55
  S2 RANGING_HIGH : thr_long=0.50, thr_short=0.50
  S3 TRENDING_UP  : thr_long=0.45, thr_short=0.50  (dir-aware)
  unknown         : thr_long=0.45, thr_short=0.45

Kontrol genuine (sama dengan 06b):
  [Fix Aturan 2] Entry trades dari OOF signals genuine_v2 (bukan in-sample)
  [Fix Aturan 3] Scaler di-fit per fold dalam CV loop

Input:
  - models/runs/tb_lgbm_genuine_v2/oof_predictions.parquet
  - models/runs/tb_lgbm_genuine_v2/hmm_threshold_best.json  (Config B params)

Output: models/runs/tb_guardian_genuine_v2_hmm/
"""
import json, sys, warnings
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

logger = setup_logger("06c_train_guardian_genuine_v2_hmm")

LGBM_RUN_NAME = "tb_lgbm_genuine_v2"
RUN_NAME      = "tb_guardian_genuine_v2_hmm"
OUT_DIR       = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HMM Config B per-state thresholds ────────────────────────────────────────
# Dibaca dari hmm_threshold_best.json Phase 3 Config B
HMM_THR_CFG = {
    0:  (0.55, 0.55),   # S0 TRENDING_DN
    1:  (0.55, 0.55),   # S1 RANGING_LOW
    2:  (0.50, 0.50),   # S2 RANGING_HIGH
    3:  (0.45, 0.50),   # S3 TRENDING_UP -- dir-aware: long agresif, short konservatif
    -1: (0.45, 0.45),   # unknown / no HMM
}

# Guardian features — identik dengan genuine_v1 dan v2 (sengaja tidak berubah)
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


def _apply_hmm_threshold(p0, p2, hmm_enc, thr_cfg):
    """Vectorized: apply per-state threshold → y_pred array."""
    n = len(p0)
    default_tl, default_ts = thr_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in thr_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    long_mask  = p2 >= tl_arr
    short_mask = (p0 >= ts_arr) & ~long_mask
    y_pred = np.ones(n, dtype=np.int32)
    y_pred[long_mask]  = 2
    y_pred[short_mask] = 0
    return y_pred


def generate_oof_samples_for_coin(sym: str, oof_pred_df: pd.DataFrame) -> list:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{sym}] Data not found: {path.name}")
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if df.empty:
        return []

    sym_oof = oof_pred_df[oof_pred_df["coin"] == sym].copy()
    sym_oof = sym_oof[sym_oof["has_oof"] == True]
    if len(sym_oof) < 30:
        logger.warning(f"[{sym}] Hanya {len(sym_oof)} OOF bar -- skip")
        return []

    sym_oof_proba = sym_oof[["p0", "p1", "p2"]].reindex(df.index)
    has_oof       = sym_oof_proba["p0"].notna()
    df_oof        = df[has_oof].copy()
    sym_oof_proba = sym_oof_proba[has_oof]
    n             = len(df_oof)

    if n < 30:
        logger.warning(f"[{sym}] Setelah join: hanya {n} bar -- skip")
        return []

    p0 = sym_oof_proba["p0"].values.astype(np.float32)
    p2 = sym_oof_proba["p2"].values.astype(np.float32)

    hmm_enc = (
        df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
        if "hmm_regime_enc" in df_oof.columns
        else np.full(n, -1, np.int8)
    )

    # Apply HMM Config B per-state threshold
    y_pred = _apply_hmm_threshold(p0, p2, hmm_enc, HMM_THR_CFG)

    n_long  = int((y_pred == 2).sum())
    n_short = int((y_pred == 0).sum())
    if n_long + n_short == 0:
        logger.warning(f"[{sym}] Tidak ada sinyal setelah HMM gate -- skip")
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
        logger.warning(f"[{sym}] OOF: tidak ada trade setelah simulasi")
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

            if direction == 2:
                current_pnl = (cp - entry_price) / entry_price
            else:
                current_pnl = (entry_price - cp) / entry_price

            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]):
                    continue
                if direction == 2:
                    mfe_sofar = max(mfe_sofar, (close_arr[k] - entry_price) / entry_price)
                else:
                    mfe_sofar = max(mfe_sofar, (entry_price - close_arr[k]) / entry_price)

            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close_arr[k]):
                    continue
                if direction == 2:
                    best_future_pnl = max(best_future_pnl, (close_arr[k] - entry_price) / entry_price)
                else:
                    best_future_pnl = max(best_future_pnl, (entry_price - close_arr[k]) / entry_price)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl \
                           if best_future_pnl > 0.001 else 0.0

            label = None
            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif mfe_sofar > 0.02 and current_pnl > 0 and current_pnl < mfe_sofar * 0.40:
                label = 2; rule_counts["profit_lock"] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2; rule_counts[3] += 1
            elif current_pnl >= best_future_pnl * 0.95 and current_pnl > 0.005:
                label = 2; rule_counts[4] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1; rule_counts[5] += 1
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1; rule_counts[6] += 1
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
        f"[{sym}] {len(trades)} trades (HMM-gated, L={n_long} S={n_short}) "
        f"-> {len(samples)} Guardian samples | "
        f"pl={rule_counts['profit_lock']} r3={rule_counts[3]} r4={rule_counts[4]} "
        f"r5={rule_counts[5]} r6={rule_counts[6]} hold={rule_counts[7]}"
    )
    return samples


def train_guardian_with_oof(samples_df: pd.DataFrame):
    g_static  = [c for c in samples_df.columns if c not in set(DYNAMIC_FEATS) and c != "label"]
    feat_cols = g_static + DYNAMIC_FEATS
    X_all     = samples_df[feat_cols].values.astype(np.float64)
    y_all     = samples_df["label"].values.astype(np.int64)

    n0 = int((y_all == 0).sum())
    n1 = int((y_all == 1).sum())
    n2 = int((y_all == 2).sum())
    logger.info(f"Total samples: {len(samples_df):,} | HOLD={n0:,} PARTIAL={n1:,} EXIT={n2:,}")

    pnl_col      = samples_df["current_pnl_pct"].values
    exit2_mask   = (y_all == 2)
    pct_neg_exit = (pnl_col[exit2_mask] < 0).mean() * 100 if exit2_mask.sum() > 0 else 0
    logger.info(f"EXIT-2 at negative PnL: {pct_neg_exit:.1f}% (target: <5%)")
    if pct_neg_exit > 10:
        logger.warning("EXIT-2 masih banyak saat PnL negatif -- cek labeling!")

    folds      = build_purged_folds(samples_df.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)
    cv_results = []
    best_ll    = float("inf")
    best_iters = None

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if len(val_idx) < 10:
            continue

        X_tr_raw = X_all[train_idx]
        X_te_raw = X_all[val_idx]
        y_tr     = y_all[train_idx]
        y_te     = y_all[val_idx]

        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_tr_raw)
        X_te = scaler_fold.transform(X_te_raw)

        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(GUARDIAN_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_prob = model.predict_proba(X_te)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        ll = -np.mean(np.log(y_prob[np.arange(len(y_te)), y_te] + 1e-10))

        cv_results.append({
            "fold":      fold_idx + 1,
            "n_train":   len(train_idx),
            "n_val":     len(val_idx),
            "logloss":   round(ll, 4),
            "f1_macro":  round(f1, 4),
            "best_iter": model.best_iteration_,
        })
        logger.info(f"  Fold {fold_idx+1}: logloss={ll:.4f} f1={f1:.4f} iter={model.best_iteration_}")
        if ll < best_ll:
            best_ll    = ll
            best_iters = model.best_iteration_

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_params = {**GUARDIAN_LGBM_PARAMS}
    if best_iters and best_iters > 0:
        final_params["n_estimators"] = best_iters
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_all_scaled, y_all)
    logger.info(f"Final Guardian: {final_params['n_estimators']} iter")

    return final_model, final_scaler, feat_cols, cv_results, best_ll


def main():
    lgbm_run_dir = MODEL_DIR / "runs" / LGBM_RUN_NAME
    oof_path = lgbm_run_dir / "oof_predictions.parquet"
    hmm_path = lgbm_run_dir / "hmm_threshold_best.json"

    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF predictions tidak ditemukan: {oof_path}\n"
            "Jalankan dahulu: python pipeline/04c_train_lgbm_genuine_v2.py"
        )
    if not hmm_path.exists():
        raise FileNotFoundError(
            f"HMM threshold config tidak ditemukan: {hmm_path}\n"
            "Jalankan dahulu: python pipeline/05e_hmm_threshold_sweep.py"
        )

    oof_pred_df = pd.read_parquet(oof_path)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GUARDIAN GENUINE OOF v2 + HMM Config B -- {RUN_NAME}")
    print(f"  Entry source  : OOF dari {LGBM_RUN_NAME} + HMM per-state threshold")
    print(f"  HMM Config B  :")
    for s, (tl, ts) in HMM_THR_CFG.items():
        sname = {0:"TRENDING_DN",1:"RANGING_LOW",2:"RANGING_HIGH",3:"TRENDING_UP",-1:"unknown"}.get(s,"?")
        print(f"    S{s} {sname:<14}: L={tl:.2f} S={ts:.2f}")
    print(f"  Static feats  : {len(STATIC_FEATS)} (sama dengan genuine_v1/v2)")
    print(f"  Dynamic feats : {len(DYNAMIC_FEATS)}")
    print(f"  CV folds      : {GUARDIAN_N_FOLDS}, purge={GUARDIAN_PURGE_GAP_BARS}")
    print(f"  Output        : {OUT_DIR}")
    print(f"{sep}\n")

    print("[1/3] Generating in-trade samples dari HMM-gated OOF trades...")
    print("-" * 55)
    all_samples = []
    for sym in ALL_COINS:
        smp = generate_oof_samples_for_coin(sym, oof_pred_df)
        all_samples.extend(smp)

    if len(all_samples) < 100:
        raise RuntimeError(f"Terlalu sedikit samples ({len(all_samples)}). Cek OOF predictions.")

    samples_df = pd.DataFrame(all_samples).reset_index(drop=True)

    n0 = int((samples_df["label"] == 0).sum())
    n1 = int((samples_df["label"] == 1).sum())
    n2 = int((samples_df["label"] == 2).sum())
    print(f"\n  Total samples : {len(samples_df):,}")
    print(f"  HOLD (0)      : {n0:,} ({n0/len(samples_df)*100:.1f}%)")
    print(f"  PARTIAL (1)   : {n1:,} ({n1/len(samples_df)*100:.1f}%)")
    print(f"  EXIT (2)      : {n2:,} ({n2/len(samples_df)*100:.1f}%)")

    pct_neg_exit_preview = (
        samples_df.loc[samples_df["label"] == 2, "current_pnl_pct"] < 0
    ).mean() * 100
    print(f"  EXIT-2 saat PnL < 0: {pct_neg_exit_preview:.1f}%  (target <5%)")

    print(f"\n[2/3] Training Guardian (scaler per fold CV)...")
    print("-" * 55)
    final_model, final_scaler, feat_cols, cv_results, best_ll = \
        train_guardian_with_oof(samples_df)

    mean_ll = float(np.mean([r["logloss"] for r in cv_results]))
    mean_f1 = float(np.mean([r["f1_macro"] for r in cv_results]))
    print(f"\n  CV Mean logloss : {mean_ll:.4f}  (genuine_v2 flat: 0.1898)")
    print(f"  CV Mean F1 macro: {mean_f1:.4f}  (genuine_v2 flat: 0.8547)")
    print(f"  Best logloss    : {best_ll:.4f}")

    print(f"\n[3/3] Saving to {OUT_DIR}...")
    joblib.dump(final_model,  OUT_DIR / "guardian.pkl")
    joblib.dump(final_scaler, OUT_DIR / "guardian_scaler.pkl")
    with open(OUT_DIR / "guardian_features.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    cv_out = {
        "run_name":       RUN_NAME,
        "created":        datetime.now().isoformat(),
        "lgbm_source":    LGBM_RUN_NAME,
        "hmm_config":     "B",
        "hmm_thresholds": {str(k): list(v) for k, v in HMM_THR_CFG.items()},
        "n_samples":      len(samples_df),
        "n_features":     len(feat_cols),
        "features":       feat_cols,
        "n_folds":        GUARDIAN_N_FOLDS,
        "purge_bars":     GUARDIAN_PURGE_GAP_BARS,
        "mean_logloss":   round(mean_ll, 4),
        "best_logloss":   round(best_ll, 4),
        "mean_f1_macro":  round(mean_f1, 4),
        "folds":          cv_results,
        "label_dist":     {"HOLD": n0, "PARTIAL": n1, "EXIT": n2},
        "methodology":    "OOF trades genuine_v2 (34-feat LGBM + HMM Config B gate), scaler per fold",
        "static_feats":   STATIC_FEATS,
        "dynamic_feats":  DYNAMIC_FEATS,
    }
    with open(OUT_DIR / "guardian_cv_results.json", "w") as f:
        json.dump(cv_out, f, indent=2)

    print(f"\n{sep}")
    print(f"  DONE -- {RUN_NAME}")
    print(f"  CV Mean logloss : {mean_ll:.4f}")
    print(f"  CV Mean F1 macro: {mean_f1:.4f}")
    print(f"  Samples         : {len(samples_df):,}")
    print(f"")
    print(f"  KONTROL GENUINE TERPENUHI:")
    print(f"  [OK] Entry dari OOF tb_lgbm_genuine_v2 + HMM Config B (bukan in-sample)")
    print(f"  [OK] Scaler di-fit per fold (bukan full dataset)")
    print(f"  [OK] Holdout Apr-Jun 2026 tidak disentuh")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
