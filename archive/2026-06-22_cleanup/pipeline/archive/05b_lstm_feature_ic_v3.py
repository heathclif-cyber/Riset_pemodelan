"""
pipeline/05b_lstm_feature_ic_v3.py -- Feature IC untuk LSTM pump/dump momentum

Target label : momentum_v4 (continuation on pump/dump bars)
Sample filter: is_pump_dump_bar == 1 (vol_spike>=2 OR range_expansion>=1.5)
Period       : training only (< TRAIN_CUTOFF_DATE) -- holdout tidak disentuh

Stage 1: Spearman IC per coin pada CV val fold (OOF-style, purge=36)
Stage 2: Greedy marginal IC (Gram-Schmidt) pada pooled gate bars
Stage 3: Temporal IC stability (5 windows, consistency >= 60%)

Output: models/runs/tb_lstm_genuine_v2/lstm_v4_feature_selection.json
"""
import json, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR, N_FOLDS, TB_PURGE_GAP_BARS
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("05b_lstm_feature_ic_v3")

RUN_DIR = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Pump/dump sample gate
VOL_SPIKE_THR = 2.0
RANGE_EXP_THR = 1.5

# IC gates
MIN_ABS_IC      = 0.03
MIN_TSTAT       = 2.0
MIN_MARGINAL_IC = 0.02
MIN_CONSISTENCY = 0.60
N_TEMPORAL_WIN  = 5
MAX_FEATURES    = 14

ORDINAL = {0: -1, 1: 0, 2: 1}  # BEARISH / NEUTRAL / BULLISH

CANDIDATE_FEATS = [
    # Flow accumulation
    "cvd", "volume_delta", "buy_volume", "sell_volume",
    "ofi_raw", "ofi_z_score", "ofi_acceleration",
    "cvd_slope_h4", "cvd_momentum_adv", "cvd_div_h4",
    # Distribution
    "absorption_z", "effort_vs_result", "vol_efficiency",
    "dynamic_position_pressure",
    # Pump/dump detectors
    "vol_spike_zscore", "range_expansion_h4", "ultra_high_vol",
    "vol_accel_3h", "trend_accel_4h",
    # Velocity (not level)
    "log_ret_1", "log_ret_5", "rsi_slope_h4",
    "swing_momentum", "bars_since_BOS",
    # Smart vs retail
    "whale_retail_divergence",
]

# Computed at runtime if missing
COMPUTED_FEATS = {
    "rsi_divergence": lambda d: (
        d["rsi_6"].diff(4) - np.log(d["close"] / d["close"].shift(4).replace(0, np.nan))
    ).fillna(0) if "rsi_6" in d.columns else pd.Series(0, index=d.index),
}


def pump_dump_mask(df: pd.DataFrame) -> pd.Series:
    vs = df["vol_spike_zscore"] if "vol_spike_zscore" in df.columns else pd.Series(0, index=df.index)
    re = df["range_expansion_h4"] if "range_expansion_h4" in df.columns else pd.Series(0, index=df.index)
    return (vs >= VOL_SPIKE_THR) | (re >= RANGE_EXP_THR)


def load_coin_frame(coin: str) -> pd.DataFrame | None:
    feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
    lbl_path  = LABEL_DIR / f"{coin}_momentum_v4_labels.parquet"
    if not feat_path.exists() or not lbl_path.exists():
        return None

    df = pd.read_parquet(feat_path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    lbl = pd.read_parquet(lbl_path).sort_index()
    df = df.join(lbl[["momentum_v4_label", "is_pump_dump_bar"]], how="inner")
    df = df.dropna(subset=["momentum_v4_label"])

    for name, fn in COMPUTED_FEATS.items():
        if name not in df.columns:
            try:
                df[name] = fn(df)
            except Exception:
                df[name] = 0.0

    avail = [c for c in CANDIDATE_FEATS if c in df.columns]
    if "rsi_divergence" in COMPUTED_FEATS and "rsi_divergence" in df.columns:
        if "rsi_divergence" not in avail:
            avail.append("rsi_divergence")

    mask = df["is_pump_dump_bar"] == 1
    sub = df.loc[mask, avail + ["momentum_v4_label"]].copy()
    if len(sub) < 200:
        return None

    sub["y_ord"] = sub["momentum_v4_label"].map(ORDINAL)
    sub["coin"] = coin
    return sub


def ic_stats(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 100:
        return 0.0, 0.0, 1.0, int(valid.sum())
    rho, pval = spearmanr(x[valid], y[valid])
    n = valid.sum()
    tstat = rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-9))
    return float(rho), float(tstat), float(pval), n


def load_pooled(coins: list) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    feat_cols = None
    for coin in coins:
        sub = load_coin_frame(coin)
        if sub is None:
            logger.warning(f"  [{coin}] skip -- insufficient pump/dump samples")
            continue
        if feat_cols is None:
            feat_cols = [c for c in sub.columns if c not in ("momentum_v4_label", "y_ord", "coin")]
        frames.append(sub)

    if not frames:
        raise RuntimeError("No data for IC test. Run 05a_momentum_labels_v4.py --all first.")

    pooled = pd.concat(frames, axis=0).sort_index()
    return pooled, feat_cols


def stage1_cv_fold_ic(pooled: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """IC on CV val folds only -- each bar scored OOS within its fold."""
    ts_index = pd.to_datetime(pooled.index, utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=TB_PURGE_GAP_BARS)
    rows = []

    for fold_i, (_, val_idx) in enumerate(folds, start=1):
        val_sub = pooled.iloc[val_idx]
        for coin, coin_sub in val_sub.groupby("coin"):
            if len(coin_sub) < 80:
                continue
            y = coin_sub["y_ord"].values.astype(np.float64)
            for feat in feat_cols:
                ic, tstat, pval, n = ic_stats(coin_sub[feat].values.astype(np.float64), y)
                rows.append({
                    "fold": fold_i, "coin": coin, "feature": feat,
                    "ic": ic, "tstat": tstat, "pval": pval, "n": n,
                })

    if not rows:
        raise RuntimeError("CV-fold IC produced no rows.")
    return pd.DataFrame(rows)


def aggregate_ic(df_ic: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_ic.groupby("feature")
        .agg(
            mean_ic=("ic", "mean"),
            std_ic=("ic", "std"),
            median_ic=("ic", "median"),
            n_coins=("ic", "count"),
            hit_rate=("ic", lambda x: (np.sign(x) == np.sign(x.mean())).mean()),
            sig_pct=("pval", lambda x: (x < 0.05).mean()),
        )
        .assign(t_stat=lambda d: d["mean_ic"] / (d["std_ic"] / d["n_coins"].pow(0.5) + 1e-9))
        .sort_values("mean_ic", key=abs, ascending=False)
    )
    agg["pass_s1"] = (agg["mean_ic"].abs() >= MIN_ABS_IC) & (agg["t_stat"].abs() >= MIN_TSTAT)
    return agg


def greedy_marginal_ic(pooled: pd.DataFrame, candidates: list) -> list[dict]:
    y = pooled["y_ord"].values.astype(np.float64)
    ts = pooled.index

    ranked = sorted(candidates, key=lambda f: abs(
        spearmanr(pooled[f].fillna(0).values, y)[0]
    ), reverse=True)

    selected = []
    residual = y.copy().astype(np.float64)
    residual -= residual.mean()
    residual /= max(residual.std(), 1e-9)

    for feat in ranked:
        x = pooled[feat].fillna(0).values.astype(np.float64)
        x = (x - x.mean()) / max(x.std(), 1e-9)

        if selected:
            for sf in selected:
                sx = pooled[sf].fillna(0).values.astype(np.float64)
                sx = (sx - sx.mean()) / max(sx.std(), 1e-9)
                beta = np.dot(sx, x) / max(np.dot(sx, sx), 1e-9)
                x = x - beta * sx

        norm = np.linalg.norm(x)
        if norm < 1e-9:
            continue
        x /= norm

        mic, tstat, _, _ = ic_stats(x, residual)
        if abs(mic) >= MIN_MARGINAL_IC:
            selected.append(feat)
            proj = np.dot(x, residual) / max(np.dot(x, x), 1e-9) * x
            residual = residual - proj
            residual /= max(residual.std(), 1e-9)
        if len(selected) >= MAX_FEATURES:
            break

    out = []
    for feat in candidates:
        out.append({
            "feature": feat,
            "selected": feat in selected,
            "rank": ranked.index(feat) + 1 if feat in ranked else 999,
        })
    return out, selected


def temporal_stability(pooled: pd.DataFrame, features: list) -> dict:
    y = pooled["y_ord"].values.astype(np.float64)
    n = len(pooled)
    bounds = np.linspace(0, n, N_TEMPORAL_WIN + 1).astype(int)
    stable = {}

    for feat in features:
        col = pooled[feat].fillna(0).values.astype(np.float64)
        signs = []
        for w in range(N_TEMPORAL_WIN):
            s, e = bounds[w], bounds[w + 1]
            ic, _, _, _ = ic_stats(col[s:e], y[s:e])
            signs.append(np.sign(ic))
        same = sum(1 for s in signs if s == np.sign(np.mean(signs))) / N_TEMPORAL_WIN
        stable[feat] = round(same, 3)

    return stable


def main():
    print(f"\n{'='*68}")
    print(f"  LSTM Feature IC -- momentum_v4 (pump/dump subset)")
    print(f"  Filter: vol_spike>={VOL_SPIKE_THR} OR range_expansion>={RANGE_EXP_THR}")
    print(f"  Period: training < {TRAIN_CUTOFF_DATE.date()} | holdout NOT used")
    print(f"  Candidates: {len(CANDIDATE_FEATS)}")
    print(f"{'='*68}\n")

    pooled, feat_cols = load_pooled(ALL_COINS)
    n_samples = len(pooled)
    label_dist = pooled["momentum_v4_label"].value_counts(normalize=True).to_dict()
    logger.info(f"Pooled pump/dump samples: {n_samples:,}")
    logger.info(f"Label dist: {label_dist}")
    logger.info(f"CV folds: {N_FOLDS}, purge={TB_PURGE_GAP_BARS}")

    df_ic = stage1_cv_fold_ic(pooled, feat_cols)
    agg = aggregate_ic(df_ic)
    s1_pass = agg[agg["pass_s1"]].index.tolist()
    logger.info(f"Stage 1 PASS: {len(s1_pass)} / {len(agg)} features")

    print(f"\n{'Feature':<28} {'mean_IC':>8} {'t_stat':>7} {'hit%':>6} {'S1':>4}")
    print("-" * 58)
    for feat, row in agg.iterrows():
        flag = "PASS" if row["pass_s1"] else "    "
        print(f"{feat:<28} {row['mean_ic']:>+8.4f} {row['t_stat']:>7.2f} "
              f"{row['hit_rate']*100:>5.0f}% {flag:>4}")

    marginal_detail, selected = greedy_marginal_ic(pooled, s1_pass)
    stability = temporal_stability(pooled, selected)
    selected_stable = [f for f in selected if stability.get(f, 0) >= MIN_CONSISTENCY]

    if not selected_stable:
        selected_stable = selected[:max(8, len(selected))]

    print(f"\n-- Stage 2: Greedy Marginal IC (selected {len(selected)}) --")
    for f in selected:
        print(f"  {f}")

    print(f"\n-- Stage 3: Temporal stability (>= {MIN_CONSISTENCY:.0%}) --")
    for f in selected:
        ok = "OK" if stability.get(f, 0) >= MIN_CONSISTENCY else "WEAK"
        print(f"  {f:<28} consistency={stability.get(f, 0):.2f}  {ok}")

    print(f"\n-- FINAL KEEP ({len(selected_stable)} features) --")
    for f in selected_stable:
        row = agg.loc[f]
        print(f"  {f:<28} IC={row['mean_ic']:+.4f}  t={row['t_stat']:.2f}")

    out = {
        "run_name": "tb_lstm_genuine_v2",
        "label": "momentum_v4_continuation",
        "sample_filter": {
            "is_pump_dump_bar": 1,
            "vol_spike_zscore_gte": VOL_SPIKE_THR,
            "range_expansion_h4_gte": RANGE_EXP_THR,
        },
        "n_pump_dump_samples": n_samples,
        "label_distribution": {str(k): round(v, 4) for k, v in label_dist.items()},
        "stage1_gate": {"min_abs_ic": MIN_ABS_IC, "min_tstat": MIN_TSTAT},
        "cv_folds": N_FOLDS,
        "purge_gap": TB_PURGE_GAP_BARS,
        "ic_method": "cv_val_fold_per_coin",
        "stage2_gate": {"min_marginal_ic": MIN_MARGINAL_IC, "max_features": MAX_FEATURES},
        "stage3_gate": {"min_consistency": MIN_CONSISTENCY, "n_windows": N_TEMPORAL_WIN},
        "all_candidates": CANDIDATE_FEATS,
        "stage1_results": agg.reset_index().to_dict(orient="records"),
        "selected_features": selected_stable,
        "selected_all_marginal": selected,
        "temporal_stability": stability,
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "methodology": "IC on CV val folds (OOF-style), pump/dump gate bars, training period only",
    }

    out_path = RUN_DIR / "lstm_v4_feature_selection.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    keep_path = RUN_DIR / "lstm_v4_selected_features.json"
    with open(keep_path, "w") as f:
        json.dump(selected_stable, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {keep_path}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()