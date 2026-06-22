"""
pipeline/04_train_lgbm_tb_binary.py — Triple Barrier LGBM Binary (widyawardhana_v2)

Perbaikan dari v1:
- Binary SHORT vs LONG (drop FLAT samples)
- Threshold disesuaikan untuk TB confidence scale
- Fast training tanpa Gram-Schmidt bottleneck

Jalankan:
  python pipeline/04_train_lgbm_tb_binary.py --all
"""

import argparse, json, sys, warnings, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import joblib, lightgbm as lgb
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger
from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS, TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

logger = setup_logger("04_train_lgbm_tb_bin")

RUN_NAME = "tb_lgbm_widyawardhana_v2"
AUTOCORR_FACTOR = 24
META_COLS = {"label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
             "hmm_regime", "hmm_regime_enc", "tb_label", "tb_label_ord"}
TEMPORAL_LAGS = [0, 1, 2, 4, 8, 12, 16, 24, 32]

# ─── TB-specific LGBM params ──────────────────────────────────────────────────
TB_LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
TB_EARLY_STOPPING = 50


def load_and_label(coins, tp_atr, sl_atr, max_hold):
    """Load data, generate TB labels, drop FLAT → binary SHORT vs LONG."""
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue

        required = ["close", "high", "low", "atr_14_h1"]
        if any(c not in df.columns for c in required):
            continue

        tb = triple_barrier_labeling(df["close"], df["high"], df["low"],
                                     df["atr_14_h1"], tp_atr, sl_atr, max_hold)
        df["tb_label"] = tb.map({"SHORT": 0, "FLAT": 1, "LONG": 2})

        # DROP FLAT → binary: 0=SHORT, 1=LONG
        df = df[df["tb_label"] != 1].copy()
        if len(df) < 100:
            continue

        # Remap: SHORT stays 0, LONG becomes 1
        df["tb_binary"] = (df["tb_label"] == 2).astype(np.int32)

        df["coin"] = sym
        frames.append(df)

        n = len(df)
        n_long = (df["tb_binary"] == 1).sum()
        n_short = (df["tb_binary"] == 0).sum()
        logger.info(f"  [{sym}] {n:,} bars | LONG={n_long/n*100:.1f}% SHORT={n_short/n*100:.1f}%")

    if not frames:
        raise RuntimeError("No training data!")
    return pd.concat(frames).sort_index()


def standalone_ic(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 100:
        return 0.0
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def tstat_ic(ic: float, n: int) -> float:
    """t-stat with autocorrelation correction (H1 data)."""
    n_eff = max(n // AUTOCORR_FACTOR, 10)
    denom = np.sqrt(max(1.0 - ic ** 2, 1e-10))
    return ic * np.sqrt(n_eff) / denom


def main():
    parser = argparse.ArgumentParser(description="TB LGBM Binary widyawardhana_v2")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--tp", type=float, default=TP_SL_FALLBACK_TP)
    parser.add_argument("--sl", type=float, default=TP_SL_FALLBACK_SL)
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS)
    args = parser.parse_args()

    coins = args.coins or (ALL_COINS if args.all else TRAINING_COINS)
    tp_atr, sl_atr, max_hold = args.tp, args.sl, args.max_hold

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TB LGBM BINARY — {RUN_NAME}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  MaxHold={max_hold}")
    print(f"  Binary: SHORT=0 vs LONG=1 (FLAT dropped)")
    print(f"{sep}\n")

    # ── Load + Label ──────────────────────────────────────────────────────────
    print("STAGE 1: Loading + Binary TB Labeling...")
    df = load_and_label(coins, tp_atr, sl_atr, max_hold)
    y_binary = df["tb_binary"].values.astype(np.int32)
    print(f"  Total: {len(df):,} bars | LONG={(y_binary==1).sum():,} SHORT={(y_binary==0).sum():,}")

    # ── Feature selection: MULTISTAGE ─────────────────────────────────────────
    from config import FEATURE_COLS_V3
    avail = [c for c in FEATURE_COLS_V3 if c in df.columns and c not in META_COLS]
    y_target = y_binary.astype(np.float64)
    n_total = len(df)

    # ── Stage 2a: Standalone IC + Marginal IC (Gram-Schmidt) ─────────────────
    print(f"\nSTAGE 2a: Standalone IC + Marginal IC (binary target)...")
    print(f"  Rows: {n_total:,} | Features: {len(avail)}")
    print(f"  Thresholds: |SA|>=0.02, |t|>=2.0, |MG|>=0.01")

    standalone = {}
    tstats = {}
    for feat in avail:
        ic = standalone_ic(df[feat].values, y_target)
        standalone[feat] = ic
        tstats[feat] = tstat_ic(ic, n_total)

    # Gram-Schmidt marginal (downsampled to 50K for speed)
    from scipy import stats as sc_stats
    def _rn(x):
        r = sc_stats.rankdata(x).astype(np.float64); r -= r.mean(); s = r.std()
        return r / s if s > 1e-10 else np.zeros_like(r)
    def _po(vec, pivot):
        nq = np.dot(pivot, pivot)
        return vec - (np.dot(vec, pivot)/nq)*pivot if nq > 1e-10 else vec.copy()

    X_mat = df[avail].values.astype(np.float64)
    y_arr = y_target.copy()
    max_gs = 50000
    if n_total > max_gs:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, max_gs, replace=False)
        X_gs = X_mat[idx]; y_gs = y_arr[idx]
        print(f"  Gram-Schmidt: downsampled to {max_gs:,} rows")
    else:
        X_gs = X_mat; y_gs = y_arr

    for j in range(X_gs.shape[1]):
        col = X_gs[:, j]; nm = np.isnan(col)
        if nm.any(): col[nm] = np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0.0
    X_r = np.column_stack([_rn(X_gs[:, j]) for j in range(X_gs.shape[1])])
    y_r = _rn(y_gs)

    remaining = list(range(len(avail))); marginal = {}
    for _ in range(len(avail)):
        if not remaining: break
        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]; nx = np.sqrt(np.dot(xj, xj)); ny = np.sqrt(np.dot(y_r, y_r))
            corrs[k] = np.dot(xj, y_r)/(nx*ny) if nx > 1e-10 and ny > 1e-10 else 0.0
        best_j = remaining[int(np.argmax(np.abs(corrs)))]
        marginal[avail[best_j]] = float(corrs[np.argmax(np.abs(corrs))])
        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j: X_r[:, j] = _po(X_r[:, j], pivot)
        y_r = _po(y_r, pivot); remaining.remove(best_j)

    # Verdict
    ic_results = []
    for feat in avail:
        sa = standalone[feat]; ts = tstats[feat]; mg = marginal.get(feat, 0.0)
        sa_ok = abs(sa) >= 0.02 and abs(ts) >= 2.0; mg_ok = abs(mg) >= 0.01
        if sa_ok and mg_ok: v = "KEEP"
        elif sa_ok and not mg_ok: v = "REDUNDANT"
        elif not sa_ok and mg_ok: v = "WEAK"
        else: v = "DROP"
        ic_results.append({"feature": feat, "standalone_ic": round(sa,4), "tstat": round(ts,2),
                          "marginal_ic": round(mg,4), "verdict": v})
    ic_results.sort(key=lambda x: {"KEEP":0,"REDUNDANT":1,"WEAK":2,"DROP":3}[x["verdict"]])
    verdicts = [r["verdict"] for r in ic_results]
    summary = {v: verdicts.count(v) for v in ["KEEP","REDUNDANT","WEAK","DROP"]}
    keep_features = [r["feature"] for r in ic_results if r["verdict"] == "KEEP"]

    print(f"  Summary: KEEP={summary['KEEP']} REDUNDANT={summary['REDUNDANT']} WEAK={summary['WEAK']} DROP={summary['DROP']}")
    print(f"  KEEP features ({summary['KEEP']}):")
    for r in ic_results:
        if r["verdict"] == "KEEP":
            print(f"    {r['standalone_ic']:+.4f}  {r['feature']}")

    # ── Stage 2b: IC Decay Stability ──────────────────────────────────────────
    print(f"\nSTAGE 2b: IC Decay Stability (6 temporal windows)...")
    decay_windows = [
        ("2020","2020-01-01","2020-12-31"), ("2021","2021-01-01","2021-12-31"),
        ("2022","2022-01-01","2022-12-31"), ("2023","2023-01-01","2023-12-31"),
        ("2024","2024-01-01","2024-12-31"), ("2025","2025-01-01","2025-10-31"),
    ]
    decay_results = {}
    for feat in avail:
        w_ics = []
        for wn, ws, we in decay_windows:
            mask = (df.index >= ws) & (df.index <= we)
            w_df = df.loc[mask]
            if len(w_df) < 500:
                w_ics.append(float("nan"))
            else:
                w_ics.append(standalone_ic(w_df[feat].values, w_df["tb_binary"].values.astype(np.float64)))
        valid = [v for v in w_ics if not np.isnan(v)]
        if len(valid) >= 4:
            mu, sd = np.mean(valid), np.std(valid)
            ic_ir = abs(mu)/sd if sd > 1e-10 else 0.0
            sc = sum(1 for v in valid if np.sign(v) == np.sign(mu))
            stable = ic_ir >= 0.5 and sc >= len(valid)-1
        else:
            mu, ic_ir, sc, stable = 0.0, 0.0, 0, False
        decay_results[feat] = {"ic_mean": round(float(mu),4), "ic_ir": round(float(ic_ir),2),
                               "sign_cons": sc, "n_windows": len(valid), "is_stable": stable}
    n_stable = sum(1 for v in decay_results.values() if v["is_stable"])
    print(f"  Stable: {n_stable}/{len(decay_results)}")
    unstable_from_keep = [f for f in keep_features if f in decay_results and not decay_results[f]["is_stable"]]
    if unstable_from_keep:
        print(f"  [DROP-unstable] {', '.join(unstable_from_keep)}")

    # ── Stage 2c: Temporal IC ─────────────────────────────────────────────────
    print(f"\nSTAGE 2c: Temporal IC (half-life)...")
    temporal_lags = [0, 1, 2, 4, 8, 12, 16, 24, 32]
    temporal_results = {}
    for feat in avail:
        x = df[feat]; y = df["tb_binary"]
        lag_ics = {}
        for k in temporal_lags:
            xl = x.shift(k); m = ~(xl.isna() | y.isna())
            if m.sum() < 100: lag_ics[str(k)] = float("nan")
            else:
                c, _ = sc_stats.spearmanr(xl[m], y[m])
                lag_ics[str(k)] = round(float(c) if not np.isnan(c) else float("nan"), 4)
        ic0 = abs(lag_ics.get("0", 0.0) or 0.0)
        hl = None
        if ic0 >= 0.01:
            for k in temporal_lags:
                if abs(lag_ics.get(str(k), 0.0) or 0.0) <= ic0 * 0.5: hl = k; break
            if hl is None: hl = temporal_lags[-1]
        cat = "STRONG" if (hl and hl >= 4) else ("MODERATE" if (hl and hl >= 2) else "SHORT")
        temporal_results[feat] = {"ic_0": lag_ics.get("0",float("nan")), "half_life": hl, "category": cat, "lag_ics": lag_ics}
    scount = sum(1 for v in temporal_results.values() if v["category"]=="STRONG")
    mcount = sum(1 for v in temporal_results.values() if v["category"]=="MODERATE")
    print(f"  STRONG(hl>=4): {scount} MODERATE(hl=2-4): {mcount} SHORT(hl<2): {len(avail)-scount-mcount}")
    top_strong = sorted([(f,d["half_life"]) for f,d in temporal_results.items() if d["category"]=="STRONG"],
                       key=lambda x: x[1], reverse=True)[:10]
    if top_strong:
        print(f"  Top STRONG: {', '.join(f'{f}(hl={hl})' for f,hl in top_strong[:5])}")

    # ── Final selection: KEEP + STABLE ───────────────────────────────────────
    selected = [f for f in keep_features if f in decay_results and decay_results[f]["is_stable"]]
    if not selected:
        print("  WARNING: No KEEP+STABLE features! Using KEEP only.")
        selected = keep_features
    print(f"\n  FINAL features: {len(selected)} (KEEP + STABLE, dropped {len(keep_features)-len(selected)} unstable)")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nSTAGE 3: Binary LGBM Training ({len(selected)} features)...")

    X_train = df[selected].ffill().fillna(0)
    y_train = df["tb_binary"].values.astype(np.int32)
    assert len(X_train) == len(y_train), f"Length mismatch: {len(X_train)} vs {len(y_train)}"

    folds = build_purged_folds(X_train.index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    all_metrics = []
    best_f1, best_model, best_fold = -1.0, None, -1

    for fold, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**TB_LGBM_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(TB_EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(period=-1)])

        y_pred = model.predict(X_val)
        f1 = float(f1_score(y_val, y_pred, average="binary", zero_division=0))
        acc = float(accuracy_score(y_val, y_pred))
        cm = confusion_matrix(y_val, y_pred)

        y_tr_pred = model.predict(X_tr)
        tr_f1 = float(f1_score(y_tr, y_tr_pred, average="binary", zero_division=0))

        metrics = {
            "fold": fold, "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "train_f1": round(tr_f1, 4),
            "val_f1": round(f1, 4),
            "val_acc": round(acc, 4),
            "confusion_matrix": cm.tolist(),
        }
        all_metrics.append(metrics)
        if f1 > best_f1:
            best_f1, best_model, best_fold = f1, model, fold

        gap = tr_f1 - f1
        logger.info(f"  Fold {fold}: Train F1={tr_f1:.4f} | Val F1={f1:.4f} | Gap={gap:+.4f} | "
                     f"Acc={acc:.4f} | Iter={model.best_iteration_}")

    # Full retrain
    avg_iter = int(np.mean([m["best_iteration"] for m in all_metrics]))
    logger.info(f"CV complete. Avg best_iteration: {avg_iter} | Best Fold: {best_fold} (F1={best_f1:.4f})")

    final_params = TB_LGBM_PARAMS.copy()
    final_params["n_estimators"] = max(avg_iter, 50)  # minimum 50 trees
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)
    logger.info(f"Final model trained with n_estimators={final_params['n_estimators']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSTAGE 4: Saving...")
    model_path = run_dir / "lgbm.pkl"
    joblib.dump(final_model, model_path)

    feat_path = run_dir / f"{RUN_NAME}_features.json"
    with open(feat_path, "w") as f:
        json.dump(selected, f, indent=2)

    f1s = [m["val_f1"] for m in all_metrics]
    accs = [m["val_acc"] for m in all_metrics]
    cv_summary = {
        "run_name": RUN_NAME, "binary": True,
        "tp_atr_mult": tp_atr, "sl_atr_mult": sl_atr, "max_hold": max_hold,
        "n_features": len(selected), "n_folds": N_FOLDS,
        "mean_val_f1": round(float(np.mean(f1s)), 4),
        "std_val_f1": round(float(np.std(f1s)), 4),
        "mean_val_acc": round(float(np.mean(accs)), 4),
        "best_fold": best_fold, "best_f1": round(best_f1, 4),
        "avg_best_iteration": avg_iter,
        "final_n_estimators": final_params["n_estimators"],
        "folds": all_metrics,
    }
    with open(run_dir / f"{RUN_NAME}_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    meta = {
        "run_name": RUN_NAME, "created": datetime.now().isoformat(),
        "binary": True, "tp_atr_mult": tp_atr, "sl_atr_mult": sl_atr, "max_hold": max_hold,
        "n_samples": len(df), "n_coins": len(coins), "n_features": len(selected),
        "label_dist": {"SHORT": int((y_binary==0).sum()), "LONG": int((y_binary==1).sum())},
        "cv_mean_f1": round(float(np.mean(f1s)), 4),
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance
    imp = list(zip(selected, final_model.feature_importances_))
    imp.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 features:")
    for i, (f, v) in enumerate(imp[:15]):
        print(f"  {i+1:>2}. {f:<35} {v:>8.1f}")

    print(f"\n{sep}")
    print(f"  TB LGBM BINARY COMPLETE — {RUN_NAME}")
    print(f"  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Random baseline: 0.50 (binary)")
    print(f"  Model: {model_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
