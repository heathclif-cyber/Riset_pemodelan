"""
pipeline/12_train_logreg_meta.py — Logistic Regression Meta-Combiner (Simons-style)

Input:  Output dari model lain (bukan raw features)
Output: P(good_trade) — seberapa yakin kombinasi sinyal ini valid?

5 parameter. Anti-overfit. Transparan.
Bobot = "model mana yang paling informatif"

Usage: python pipeline/12_train_logreg_meta.py
"""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS, LABEL_MAP, MODAL_PER_TRADE,
    CONFIDENCE_THRESHOLD_ENTRY,
)
from pipeline.shared import build_purged_folds

COINANK_DIR = ROOT / "data" / "coinank"
MACRO_DIR = ROOT / "data" / "macro"

# ─── Features for LogReg (MODEL OUTPUTS, not raw features) ──────────────
# These are SIMONS-STYLE signals: outputs of existing models combined
FEATURE_COLS = [
    # Model 1: HMM regime (0=TREND_DN, 1=RANGE_LO, 2=RANGE_HI, 3=TREND_UP)
    "hmm_regime_enc",
    # Model 2: LGBM confidence at entry
    "lgbm_confidence",
    # Model 3: LSTM Survival support (from existing model)
    "lstm_support",
    # Model 4: Trend context — are we with or against H4 trend?
    "is_with_trend",
    # Model 5: Positioning context (OI extreme = caution)
    "oi_zscore",
    # Model 6: Macro context (Fear & Greed)
    "fear_greed_zscore",
    # Model 7: Is this a TRENDING regime? (0,3 = trending)
    "is_trending",
]

def load_positioning():
    """Load OI data for positioning context."""
    pos = {}
    for coin in TRAINING_COINS:
        oi_p = COINANK_DIR / f"{coin}_oi.parquet"
        if not oi_p.exists(): continue
        oi = pd.read_parquet(oi_p).sort_index()
        oi_cols = [c for c in oi.columns if c.startswith("oi_") or "oi" in c.lower()]
        if not oi_cols: continue
        oi_t = oi[oi_cols[0]]
        om = oi_t.rolling(20).mean(); os = oi_t.rolling(20).std().clip(lower=1e-8)
        daily = pd.DataFrame({
            "oi_zscore": (oi_t - om) / os,
        }, index=oi.index)
        pos[coin] = daily
    return pos

def load_fear_greed():
    """Load Fear & Greed macro data."""
    fg_p = MACRO_DIR / "fear_greed.parquet"
    if not fg_p.exists(): return None
    fg = pd.read_parquet(fg_p)
    if "fear_greed_value" not in fg.columns: return None
    return fg[["fear_greed_value"]]

def generate_oof_trades(coins):
    """
    Walk-forward OOF trade generation.
    Same methodology as Phase 2 meta-labeling — genuine OOF.
    For each fold:
      - Retrain LGBM on training folds
      - Predict on test fold → simulate simple trades
      - Extract features at entry + label (is_good_trade)
    """
    pos_data = load_positioning()
    fg_data = load_fear_greed()
    lgbm_feats = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
    lstm_feats = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
    from core.models import load_lstm
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    all_samples = []
    total_trades = 0

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
        if len(df) < 500: continue

        ts_index = pd.DatetimeIndex(df.index)
        folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)
        feat_cols = [c for c in lgbm_feats if c in df.columns]

        for fi, (tr_idx, te_idx) in enumerate(folds):
            if len(te_idx) < 100: continue
            df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]

            X_tr = df_tr[feat_cols].ffill().fillna(0)
            y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
            if len(np.unique(y_tr)) < 3: continue

            # Retrain LGBM per fold (GENUINE OOF)
            fold_model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=300,
                learning_rate=0.05, max_depth=6, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                verbose=-1, n_jobs=-1, random_state=42)
            fold_model.fit(X_tr, y_tr)

            n_te = len(df_te)
            X_te = np.zeros((n_te, len(feat_cols)))
            for i, col in enumerate(feat_cols):
                if col in df_te.columns: X_te[:, i] = df_te[col].ffill().fillna(0).values

            proba = fold_model.predict_proba(X_te)

            # LSTM support
            X_lstm = np.zeros((n_te, len(lstm_feats)))
            for i, col in enumerate(lstm_feats):
                if col in df_te.columns: X_lstm[:, i] = df_te[col].ffill().fillna(0).values

            lstm_support = np.full(n_te, 0.35)  # default neutral
            try:
                import torch
                X_lstm_scaled = lstm_scaler.transform(X_lstm).reshape(n_te, 16, -1)
                with torch.no_grad():
                    lstm_proba = lstm_model(torch.from_numpy(X_lstm_scaled.astype(np.float32)))
                    lstm_proba = torch.softmax(lstm_proba, dim=1).cpu().numpy()
                lstm_support = np.maximum(lstm_proba[:, 2], lstm_proba[:, 0])  # max directional prob
            except: pass

            # Get OI zscore for this coin
            oi_z = np.zeros(n_te)
            if coin in pos_data:
                daily = pos_data[coin]
                for j in range(n_te):
                    entry_date = pd.Timestamp(df_te.index[j].date(), tz="UTC")
                    avail = daily[daily.index <= entry_date]
                    if len(avail) > 0 and "oi_zscore" in avail.columns:
                        v = avail["oi_zscore"].iloc[-1]
                        oi_z[j] = float(v) if pd.notna(v) else 0.0

            # Get Fear & Greed at entry
            fg_z = np.zeros(n_te)
            if fg_data is not None:
                for j in range(n_te):
                    entry_date = pd.Timestamp(df_te.index[j].date(), tz="UTC")
                    avail = fg_data[fg_data.index <= entry_date]
                    if len(avail) > 0:
                        fg_z[j] = (float(avail["fear_greed_value"].iloc[-1]) - 50.0) / 25.0

            # Simulate simple trades + extract features
            close = df_te["close"].values

            for i in range(n_te):
                y_pred = np.argmax(proba[i]); conf = proba[i, y_pred]
                if y_pred == 1: continue  # FLAT
                if (y_pred == 2 and conf < 0.69) or (y_pred == 0 and conf < 0.59): continue

                direction = 1 if y_pred == 2 else -1
                entry_price = close[i]
                tp = entry_price * (1 + 0.02 * direction)
                sl = entry_price * (1 - 0.015 * direction)

                exit_bar = i + 1
                while exit_bar < min(i + 48, n_te):
                    if direction == 1:
                        if df_te["high"].iloc[exit_bar] >= tp: break
                        if df_te["low"].iloc[exit_bar] <= sl: break
                    else:
                        if df_te["low"].iloc[exit_bar] <= tp: break
                        if df_te["high"].iloc[exit_bar] >= sl: break
                    exit_bar += 1
                if exit_bar >= n_te or exit_bar >= i + 48: exit_bar = min(i + 47, n_te - 1)

                exit_price = close[exit_bar]
                pnl = (exit_price - entry_price) * direction * MODAL_PER_TRADE / entry_price
                is_good = 1 if pnl > 0 else 0

                # Extract features at entry
                hmm_regime = int(df_te["hmm_regime_enc"].iloc[i])
                h4_t = float(df_te["h4_trend"].iloc[i]) if "h4_trend" in df_te.columns else 0
                lgbm_conf = conf
                lstm_sup = float(lstm_support[i])
                is_with = (y_pred == 2 and h4_t > 0) or (y_pred == 0 and h4_t < 0)
                is_trending = 1 if hmm_regime in (0, 3) else 0

                all_samples.append([
                    hmm_regime,           # HMM state
                    lgbm_conf,            # LGBM confidence
                    lstm_sup,             # LSTM support
                    float(is_with),       # With trend?
                    float(oi_z[i]),       # OI zscore
                    float(fg_z[i]),       # Fear & Greed zscore
                    float(is_trending),   # Is trending?
                    is_good,              # TARGET
                ])
                total_trades += 1

    return np.array(all_samples), total_trades


def main():
    coins = TRAINING_COINS[:5]  # Same 5 coins as Phase 2

    print(f"\n{'='*60}")
    print(f"  LOGISTIC REGRESSION META-COMBINER")
    print(f"  Input: model outputs, not raw features")
    print(f"  {len(FEATURE_COLS)} features -> 1 output: P(good_trade)")
    print(f"  Simons: 5 params, anti-overfit, fully interpretable")
    print(f"{'='*60}\n")

    print("Generating OOF trades (walk-forward purged CV)...")
    samples, total_trades = generate_oof_trades(coins)

    X = samples[:, :-1]
    y = samples[:, -1]
    good_pct = y.mean() * 100

    print(f"  OOF trades: {total_trades:,}")
    print(f"  Good trades: {y.sum():.0f} ({good_pct:.1f}%)")
    print(f"  Features: {X.shape[1]} | Random baseline: {max(good_pct, 100-good_pct)/100:.3f} AUC")

    # Train LogReg with 5-fold CV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simple LogReg with L2 regularization
    logreg = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)

    # CV
    cv_aucs = cross_val_score(logreg, X_scaled, y, cv=5, scoring="roc_auc")
    print(f"\n  CV AUC (5-fold): {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

    # Fit on all data
    logreg.fit(X_scaled, y)

    # Report coefficients
    print(f"\n  COEFFICIENTS (model contribusi):")
    print(f"  {'Feature':<25} {'Weight':>8} {'Interpretation':>40}")
    print(f"  {'-'*75}")
    for name, coef in zip(FEATURE_COLS, logreg.coef_[0]):
        direction = "BOOST" if coef > 0 else "PENALIZE"
        impact = abs(coef)
        bar = "#" * int(impact * 20)
        print(f"  {name:<25} {coef:>+8.4f}  {bar} {direction}")

    bias = logreg.intercept_[0]
    print(f"  {'(bias/intercept)':<25} {bias:>+8.4f}  baseline good trade rate: {good_pct:.1f}%")

    # Feature importance
    importances = np.abs(logreg.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  FEATURE IMPORTANCE (by |weight|):")
    for idx in sorted_idx:
        name = FEATURE_COLS[idx]
        coef = logreg.coef_[0][idx]
        print(f"  {idx+1}. {name:<25} |weight|={abs(coef):.4f}  {'[STRONG]' if abs(coef)>0.3 else '[MODERATE]' if abs(coef)>0.1 else '[WEAK]'}")

    # Threshold analysis
    proba = logreg.predict_proba(X_scaled)[:, 1]
    print(f"\n  THRESHOLD ANALYSIS:")
    for thr in [0.55, 0.60, 0.65, 0.70]:
        mask = proba >= thr
        if mask.sum() > 0:
            wr = y[mask].mean() * 100
            pct = mask.mean() * 100
            print(f"  thr={thr:.2f}: WR={wr:.1f}% ({mask.sum():.0f}/{len(y)} trades, {pct:.1f}% selected)")

    # Compare: if we block trades below 0.45
    blocked_mask = proba < 0.45
    if blocked_mask.sum() > 0:
        blocked_wr = y[blocked_mask].mean() * 100
        kept_mask = proba >= 0.45
        kept_wr = y[kept_mask].mean() * 100
        print(f"\n  BLOCK TRADES BELOW 0.45:")
        print(f"  Blocked: {blocked_mask.sum()} trades, WR={blocked_wr:.1f}% (should be LOW)")
        print(f"  Kept:    {kept_mask.sum()} trades, WR={kept_wr:.1f}% (should be HIGHER)")
        print(f"  Improvement: {kept_wr - good_pct:+.1f}pp WR")

    # Save model
    run_dir = MODEL_DIR / "runs" / "logreg_meta_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, run_dir / "logreg_meta.pkl")
    joblib.dump(scaler, run_dir / "logreg_scaler.pkl")
    json.dump({
        "features": FEATURE_COLS,
        "n_samples": int(len(y)),
        "good_rate": float(good_pct),
        "cv_auc_mean": float(np.mean(cv_aucs)),
        "cv_auc_std": float(np.std(cv_aucs)),
        "coefficients": {name: float(c) for name, c in zip(FEATURE_COLS, logreg.coef_[0])},
        "intercept": float(bias),
    }, open(run_dir / "logreg_meta_info.json", "w"), indent=2)

    print(f"\n  Model saved: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
