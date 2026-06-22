"""
pipeline/06_train_guardian_ic32_continuation_v1.py
Guardian Continuation untuk ic32_regime_v1 — OOF entry + momentum HOLD labeling.

Entry  : OOF LGBM ic32 (thr 0.69/0.59) — genuine, bukan in-sample
Basis  : tb_guardian_continuation_v1 (momentum delta + flow override)
Output : models/runs/ic32_guardian_continuation_v1/

Prerequisite:
  python pipeline/04_train_lgbm_ic32_genuine_oof.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sc_stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    ALL_COINS,
    LABEL_MAP,
    GUARDIAN_LGBM_PARAMS,
    GUARDIAN_EARLY_STOPPING,
    GUARDIAN_N_FOLDS,
    GUARDIAN_PURGE_GAP_BARS,
    GUARDIAN_MIN_HOLD_BARS,
    TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS,
    MODAL_PER_TRADE,
    LEVERAGE_SIM,
    FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP,
    TP_SL_FALLBACK_SL,
    LGBM_THRESHOLD_LONG,
    LGBM_THRESHOLD_SHORT,
    MODEL_DIR,
    LABEL_DIR,
)
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from pipeline.shared import build_purged_folds

logger = setup_logger("06_ic32_guardian_cont_v1")

LGBM_RUN = "ic32_regime_v1"
RUN_NAME = "ic32_guardian_continuation_v1"
OUT_DIR = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

THR_LONG = LGBM_THRESHOLD_LONG
THR_SHORT = LGBM_THRESHOLD_SHORT
IC_MIN = 0.01
FLOW_MOM_WINDOW = 3

STATIC_BASE = [
    "cvd_slope_h4", "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1", "dow_cos",
    "cvd_div_h4", "dist_swing_low", "VAH", "cvd_momentum_adv",
    "dist_from_8h_high", "ema_200_h1",
]
STATIC_MOM = [
    "cvd_slope_h4_delta", "ofi_h4_accel", "rsi_h4_slope", "dist_liq_50x_long",
]
STATIC_CONT = [
    "h4_trend", "trend_accel_4h", "price_accel_1h", "log_ret_5",
    "whale_retail_divergence", "absorption_z", "hmm_regime_enc", "ofi_z_score",
]
STATIC_CANDIDATES = STATIC_BASE + STATIC_MOM + STATIC_CONT

DYNAMIC_BASE = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]
DYNAMIC_EXTRA = [
    "cvd_slope_h4_delta_entry", "ofi_h4_delta_entry", "flow_momentum_3bar",
]
DYNAMIC_FEATS = DYNAMIC_BASE + DYNAMIC_EXTRA


def compute_derived_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cvd_slope_h4" in df.columns:
        df["cvd_slope_h4_delta"] = df["cvd_slope_h4"].diff(1)
    else:
        df["cvd_slope_h4_delta"] = 0.0
    if "ofi_h4_delta" in df.columns:
        df["ofi_h4_accel"] = df["ofi_h4_delta"].diff(2)
    else:
        df["ofi_h4_accel"] = 0.0
    if "rsi_h4" in df.columns:
        df["rsi_h4_slope"] = df["rsi_h4"].diff(2)
    else:
        df["rsi_h4_slope"] = 0.0
    if "dist_liq_50x_long" not in df.columns:
        df["dist_liq_50x_long"] = 0.0
    if "ofi_z_score" in df.columns:
        df["flow_momentum_3bar"] = df["ofi_z_score"].rolling(FLOW_MOM_WINDOW, min_periods=1).mean()
    else:
        df["flow_momentum_3bar"] = 0.0
    return df


def flow_still_aligned(direction, bar, bar_in, df, n, best_future_pnl, current_pnl) -> bool:
    if best_future_pnl <= current_pnl * 1.05:
        return False
    cvd = df["cvd_slope_h4"].values if "cvd_slope_h4" in df.columns else np.zeros(n)
    ofi = df["ofi_h4_delta"].values if "ofi_h4_delta" in df.columns else np.zeros(n)
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else np.zeros(n)
    pa = df["price_accel_1h"].values if "price_accel_1h" in df.columns else np.zeros(n)

    def _v(arr, idx):
        v = arr[idx] if idx < len(arr) else 0.0
        return 0.0 if np.isnan(v) else float(v)

    cvd_delta = _v(cvd, bar) - _v(cvd, bar_in)
    ofi_delta = _v(ofi, bar) - _v(ofi, bar_in)
    pa_val = _v(pa, bar)
    h4t_val = _v(h4t, bar)

    if direction == 2:
        flow_score = cvd_delta + ofi_delta + pa_val * 10.0
        trend_ok = h4t_val >= 0.0
    else:
        flow_score = -(cvd_delta + ofi_delta + pa_val * 10.0)
        trend_ok = h4t_val <= 0.0
    return flow_score > 0.3 and trend_ok


def generate_oof_samples_for_coin(sym: str, oof_pred_df: pd.DataFrame) -> list:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    df = compute_derived_feats(df)

    rp = LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        except Exception:
            df["hmm_regime_enc"] = 1
    elif "hmm_regime_enc" not in df.columns:
        df["hmm_regime_enc"] = 1

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()

    sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
    sym_oof = sym_oof[sym_oof["has_oof"] == True]
    if len(sym_oof) < 30:
        return []

    sym_oof_proba = sym_oof[["p0", "p1", "p2"]].reindex(df.index)
    has_oof = sym_oof_proba["p0"].notna()
    df_oof = df[has_oof].copy()
    sym_oof_proba = sym_oof_proba[has_oof]
    n = len(df_oof)
    if n < 30:
        return []

    p0 = sym_oof_proba["p0"].values
    p2 = sym_oof_proba["p2"].values
    y_pred = np.full(n, 1, np.int32)
    y_pred[p2 >= THR_LONG] = 2
    y_pred[(p0 >= THR_SHORT) & (y_pred != 2)] = 0

    close_arr = df_oof["close"].values
    high_arr = df_oof["high"].values
    low_arr = df_oof["low"].values
    atr_arr = df_oof["atr_14_h1"].values
    h4_sh = df_oof["h4_swing_high"].values if "h4_swing_high" in df_oof.columns else np.full(n, np.nan)
    h4_sl = df_oof["h4_swing_low"].values if "h4_swing_low" in df_oof.columns else np.full(n, np.nan)

    result = simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
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

    g_static_cols = [c for c in STATIC_CANDIDATES if c in df_oof.columns]
    X_static = np.zeros((n, len(g_static_cols)), dtype=np.float64)
    for idx, col in enumerate(g_static_cols):
        X_static[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)
    col_idx = {c: i for i, c in enumerate(g_static_cols)}
    flow_arr = df_oof["flow_momentum_3bar"].values

    samples = []
    rule_counts = {"profit_lock": 0, "hold_override": 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    for t in trades:
        bar_in = t["bar_in"]
        bar_out = t["bar_out"]
        direction = 2 if t["direction"] == "LONG" else 0
        entry_price = t["entry"]
        atr_entry = atr_arr[bar_in] if bar_in < n else 0.01
        if bar_out <= bar_in + 1:
            continue

        for j in range(bar_in + 1, min(bar_out, n)):
            cp = close_arr[j]
            if np.isnan(cp):
                continue
            current_pnl = (cp - entry_price) / entry_price if direction == 2 else (entry_price - cp) / entry_price
            bars_held = j - bar_in

            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]):
                    continue
                pnl_k = (close_arr[k] - entry_price) / entry_price if direction == 2 else (entry_price - close_arr[k]) / entry_price
                mfe_sofar = max(mfe_sofar, pnl_k)

            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close_arr[k]):
                    continue
                pnl_k = (close_arr[k] - entry_price) / entry_price if direction == 2 else (entry_price - close_arr[k]) / entry_price
                best_future_pnl = max(best_future_pnl, pnl_k)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl if best_future_pnl > 0.001 else 0.0

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

            if label in (1, 2) and flow_still_aligned(direction, j, bar_in, df_oof, n, best_future_pnl, current_pnl):
                label = 0
                rule_counts["hold_override"] += 1

            atr_pct = atr_entry / entry_price if entry_price > 0 else 0.01
            cvd_ent = X_static[bar_in, col_idx["cvd_slope_h4"]] if "cvd_slope_h4" in col_idx else 0.0
            cvd_cur = X_static[j, col_idx["cvd_slope_h4"]] if "cvd_slope_h4" in col_idx else 0.0
            ofi_ent = X_static[bar_in, col_idx["ofi_h4_delta"]] if "ofi_h4_delta" in col_idx else 0.0
            ofi_cur = X_static[j, col_idx["ofi_h4_delta"]] if "ofi_h4_delta" in col_idx else 0.0

            sample = {
                **{c: float(X_static[j, idx]) for idx, c in enumerate(g_static_cols)},
                "timestamp": df_oof.index[j],
                "bars_held_norm": bars_held / MAX_HOLDING_BARS,
                "current_pnl_pct": current_pnl,
                "current_pnl_atr": current_pnl / atr_pct if atr_pct > 0 else 0.0,
                "max_favorable_pnl_pct": mfe_sofar,
                "drawdown_from_peak_pct": (mfe_sofar - current_pnl) / mfe_sofar if mfe_sofar > 0.001 else 0.0,
                "direction": 1.0 if direction == 2 else 0.0,
                "entry_price_ratio": entry_price / cp if cp > 0 else 1.0,
                "cvd_slope_h4_delta_entry": float(cvd_cur - cvd_ent),
                "ofi_h4_delta_entry": float(ofi_cur - ofi_ent),
                "flow_momentum_3bar": float(flow_arr[j]) if j < len(flow_arr) else 0.0,
                "label": label,
            }
            samples.append(sample)

    logger.info(
        f"[{sym}] {len(trades)} OOF trades -> {len(samples)} samples | "
        f"override={rule_counts['hold_override']} hold={rule_counts[7]}"
    )
    return samples


def select_features_ic(samples_df: pd.DataFrame) -> list[str]:
    y = samples_df["label"].values.astype(np.float64)
    static_avail = [c for c in STATIC_CANDIDATES if c in samples_df.columns]
    selected = []
    for feat in static_avail:
        x = samples_df[feat].values.astype(np.float64)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 100:
            continue
        corr, _ = sc_stats.spearmanr(x[mask], y[mask])
        if not np.isnan(corr) and abs(float(corr)) >= IC_MIN:
            selected.append(feat)
    for must in STATIC_MOM + ["h4_trend", "cvd_slope_h4", "ofi_h4_delta"]:
        if must in static_avail and must not in selected:
            selected.append(must)
    return sorted(set(selected)) + DYNAMIC_FEATS


def train_guardian(samples_df: pd.DataFrame, feat_cols: list[str]):
    X_all = samples_df[feat_cols].values.astype(np.float64)
    y_all = samples_df["label"].values.astype(np.int64)

    samples_df = samples_df.copy()
    samples_df["timestamp"] = pd.to_datetime(samples_df["timestamp"])
    samples_df = samples_df.set_index("timestamp").sort_index()

    folds = build_purged_folds(samples_df.index, GUARDIAN_N_FOLDS, GUARDIAN_PURGE_GAP_BARS)
    cv_results = []
    best_ll = float("inf")
    best_iters = None

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if len(val_idx) < 10:
            continue
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_all[train_idx])
        X_va = scaler_fold.transform(X_all[val_idx])
        y_tr, y_va = y_all[train_idx], y_all[val_idx]

        model = lgb.LGBMClassifier(**GUARDIAN_LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(GUARDIAN_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
        )
        y_prob = model.predict_proba(X_va)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
        ll = -np.mean(np.log(y_prob[np.arange(len(y_va)), y_va] + 1e-10))
        cv_results.append({"fold": fold_idx + 1, "logloss": round(ll, 4), "f1_macro": round(f1, 4)})
        logger.info(f"  Fold {fold_idx+1}: logloss={ll:.4f} f1={f1:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_iters = model.best_iteration_

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_all)
    params = {**GUARDIAN_LGBM_PARAMS}
    if best_iters and best_iters > 0:
        params["n_estimators"] = best_iters
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_scaled, y_all)
    return final_model, final_scaler, cv_results, best_ll


def main():
    oof_path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing {oof_path} — run 04_train_lgbm_ic32_genuine_oof.py first")

    oof_df = pd.read_parquet(oof_path)
    print(f"\n{'='*65}")
    print(f"  ic32 Guardian Continuation v1")
    print(f"  Entry OOF: {LGBM_RUN} thr={THR_LONG}/{THR_SHORT}")
    print(f"  Out: {OUT_DIR}")
    print(f"{'='*65}\n")

    all_samples = []
    for sym in ALL_COINS:
        all_samples.extend(generate_oof_samples_for_coin(sym, oof_df))
    if len(all_samples) < 500:
        raise RuntimeError(f"Too few samples: {len(all_samples)}")

    samples_df = pd.DataFrame(all_samples)
    label_dist = samples_df["label"].value_counts().to_dict()
    print(f"Samples: {len(samples_df):,} | dist={label_dist}")

    feat_cols = select_features_ic(samples_df)
    print(f"Features: {len(feat_cols)} ({len(feat_cols)-len(DYNAMIC_FEATS)} static + {len(DYNAMIC_FEATS)} dynamic)")

    model, scaler, cv_results, best_ll = train_guardian(samples_df, feat_cols)
    if cv_results:
        print(f"CV logloss mean={np.mean([r['logloss'] for r in cv_results]):.4f} best={best_ll:.4f}")

    joblib.dump(model, OUT_DIR / "guardian.pkl")
    joblib.dump(scaler, OUT_DIR / "guardian_scaler.pkl")
    with open(OUT_DIR / "guardian_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2)

    exit2 = samples_df[samples_df["label"] == 2]["current_pnl_pct"]
    meta = {
        "run_name": RUN_NAME,
        "trained_at": datetime.now().isoformat(),
        "entry_model": LGBM_RUN,
        "entry_oof": True,
        "thr_long": THR_LONG,
        "thr_short": THR_SHORT,
        "n_samples": len(samples_df),
        "label_dist": {str(k): int(v) for k, v in label_dist.items()},
        "feat_cols": feat_cols,
        "cv_results": cv_results,
        "best_logloss": best_ll,
        "exit2_pnl_mean_pct": round(float(exit2.mean() * 100), 3) if len(exit2) else 0,
    }
    with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()