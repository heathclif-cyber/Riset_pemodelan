"""
pipeline/05p_lstm_lgbm_conf_feat_v1.py
LSTM Momentum + LGBM score history (variant B).

Perbedaan vs tb_lstm_genuine_v2:
  - Input: 8 market feats + 3 LGBM OOF probs per timestep (p_short, p_flat, p_long)
  - LGBM scores = historis seq_len bar (bukan constant di bar entry saja)
  - Sumber LGBM: tb_lgbm_genuine_v2/oof_predictions.parquet (has_oof=True ONLY)
  - Tidak ffill skor LGBM yang hilang — sequence di-skip jika ada gap OOF di window

Genuine protocol:
  - Purged CV OOF LSTM, RobustScaler per fold (Aturan 3)
  - LGBM input = OOF prediksi (bukan model final in-sample)
  - TRAIN_CUTOFF_DATE, gate bars only, holdout tidak disentuh
  - Eval complement + simpan oof_lstm_predictions.parquet untuk 05j

Usage:
  python pipeline/05p_lstm_lgbm_conf_feat_v1.py --all
"""
import argparse
import gc
import importlib.util
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, TB_PURGE_GAP_BARS, LSTM_SEQ_LEN, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_V2_HIDDEN, LSTM_V2_LAYERS,
    LSTM_V2_DROPOUT,
)
from core.models import save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05p_lstm_lgbm_conf_feat_v1")
DEVICE = get_lstm_device()

RUN_NAME = "tb_lstm_lgbm_seq_v1"
LGBM_RUN = "tb_lgbm_genuine_v2"
BASE_LSTM_RUN = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"

LGBM_SEQ_COLS = ["lgbm_p_short", "lgbm_p_flat", "lgbm_p_long"]
MOMENTUM_LABEL_MAP = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

_spec = importlib.util.spec_from_file_location(
    "lstm_base", ROOT / "pipeline" / "05_train_lstm_genuine_v2.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def load_feature_list() -> list[str]:
    return _base.load_feature_list(BASE_LSTM_RUN)


def load_lgbm_oof() -> pd.DataFrame:
    path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run pipeline/04_train_lgbm_genuine_v1.py first."
        )
    df = pd.read_parquet(path)
    df = df.loc[df["has_oof"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_data_lgbm_seq(
    coins: list[str],
    market_cols: list[str],
    lgbm_oof: pd.DataFrame,
):
    """
    Build (n, seq_len, n_market+3) tensors.
    LGBM channel = OOF p0/p1/p2 at each timestep in the window (variant B).
    """
    X_seqs, y_seqs, ts_seqs, meta_rows = [], [], [], []
    skipped = []
    n_skip_no_oof = 0

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v4_labels.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists() or not lp.exists():
            skipped.append(coin)
            continue

        df = pd.read_parquet(fp).sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        lbl.index = pd.to_datetime(lbl.index, utc=True)
        df = df.join(lbl[["momentum_v4_label", "is_pump_dump_bar"]], how="inner")
        df = df.dropna(subset=["momentum_v4_label"])

        if "hmm_regime_enc" not in df.columns:
            if rp.exists():
                reg = pd.read_parquet(rp).sort_index()
                reg.index = pd.to_datetime(reg.index, utc=True)
                df = df.join(reg[["hmm_regime_enc"]], how="left")
            else:
                df["hmm_regime_enc"] = -1
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(-1).astype(np.int8)

        sym_oof = lgbm_oof[lgbm_oof["coin"] == coin][["p0", "p1", "p2"]]
        lgbm_aligned = sym_oof.reindex(df.index)
        df["lgbm_p_short"] = lgbm_aligned["p0"].values.astype(np.float32)
        df["lgbm_p_flat"] = lgbm_aligned["p1"].values.astype(np.float32)
        df["lgbm_p_long"] = lgbm_aligned["p2"].values.astype(np.float32)

        avail = [c for c in market_cols if c in df.columns]
        if len(avail) < len(market_cols):
            missing = set(market_cols) - set(avail)
            logger.warning(f"  [{coin}] missing market feats: {missing}")

        if len(df) < LSTM_SEQ_LEN + 10:
            skipped.append(coin)
            continue

        feat_vals = {}
        for c in avail:
            vals = df[c].ffill().fillna(0).values.astype(np.float32)
            if c in _base._PERCOIN_ZSCORE_FEATS:
                vals = _base._percoin_z(vals.astype(np.float64)).astype(np.float32)
            feat_vals[c] = vals

        X_market = np.column_stack([feat_vals[c] for c in avail]).astype(np.float32)
        X_lgbm = df[LGBM_SEQ_COLS].values.astype(np.float32)
        y_c = df["momentum_v4_label"].values.astype(np.int64)
        gate = df["is_pump_dump_bar"].values.astype(bool)
        ts_c = df.index.values
        vol_spike = (
            df["vol_spike_zscore"].values.astype(np.float32)
            if "vol_spike_zscore" in df.columns
            else np.zeros(len(df), np.float32)
        )
        hmm_enc = df["hmm_regime_enc"].values.astype(np.int8)

        n_gate = n_kept = 0
        for i in range(LSTM_SEQ_LEN - 1, len(df)):
            if not gate[i]:
                continue
            n_gate += 1
            sl = slice(i - LSTM_SEQ_LEN + 1, i + 1)
            lgbm_win = X_lgbm[sl]
            if not np.isfinite(lgbm_win).all():
                n_skip_no_oof += 1
                continue
            market_win = X_market[sl]
            seq = np.concatenate([market_win, lgbm_win], axis=1)
            n_kept += 1
            X_seqs.append(seq)
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            meta_rows.append({
                "coin": coin,
                "vol_spike": float(vol_spike[i]),
                "hmm_enc": int(hmm_enc[i]),
                "lgbm_p_short": float(lgbm_win[-1, 0]),
                "lgbm_p_flat": float(lgbm_win[-1, 1]),
                "lgbm_p_long": float(lgbm_win[-1, 2]),
                "is_gate": 1,
            })

        sub = y_c[gate]
        logger.info(
            f"  [{coin}] gate={n_gate:,} kept={n_kept:,} "
            f"(skip_no_oof={n_gate - n_kept:,}) | "
            f"BULL={(sub == 2).mean()*100:.0f}% "
            f"NEU={(sub == 1).mean()*100:.0f}% "
            f"BEAR={(sub == 0).mean()*100:.0f}%"
        )

    if skipped:
        logger.warning(f"Skipped coins: {skipped}")
    if not X_seqs:
        raise ValueError(
            "No sequences with full LGBM OOF window. "
            "Check oof_predictions.parquet coverage."
        )

    logger.info(f"Total skipped (incomplete LGBM OOF window): {n_skip_no_oof:,}")

    X = np.stack(X_seqs)
    y = np.array(y_seqs, dtype=np.int64)
    ts = np.array(ts_seqs)
    meta_df = pd.DataFrame(meta_rows)
    meta_df["ts"] = ts
    order = np.argsort(ts)
    feat_cols_used = avail + LGBM_SEQ_COLS
    return (
        X[order], y[order], ts[order],
        meta_df.iloc[order].reset_index(drop=True),
        feat_cols_used,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]
    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    market_cols = load_feature_list()
    lgbm_oof = load_lgbm_oof()
    logger.info(f"LGBM OOF loaded: {len(lgbm_oof):,} bars (has_oof=True)")

    print(f"\n{'='*66}")
    print(f"  LSTM + LGBM Seq History -- {RUN_NAME}")
    print(f"  Partner : {LGBM_RUN} (OOF p0/p1/p2 per timestep)")
    print(f"  Variant : B — {len(market_cols)} market + 3 LGBM = {len(market_cols)+3} feat/step")
    print(f"  Seq len : {LSTM_SEQ_LEN} | Purge: {TB_PURGE_GAP_BARS} | Folds: {N_FOLDS}")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*66}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, meta_df, feat_cols_used = load_data_lgbm_seq(coins, market_cols, lgbm_oof)
    logger.info(f"Gate sequences: {X.shape[0]:,} | seq={LSTM_SEQ_LEN} | feat={X.shape[2]}")

    for lbl_int, lbl_str in MOMENTUM_LABEL_MAP.items():
        cnt = (y == lbl_int).sum()
        logger.info(f"  {lbl_str}: {cnt:,} ({cnt / len(y) * 100:.1f}%)")

    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(feat_cols_used, f, indent=2)

    ts_index = pd.to_datetime(ts, utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=TB_PURGE_GAP_BARS)

    all_metrics = []
    oof_proba_all = np.full((len(y), 3), np.nan, dtype=np.float64)
    oof_has = np.zeros(len(y), dtype=bool)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m, oof_proba = _base.train_one_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1
        )
        all_metrics.append(m)
        oof_proba_all[te_idx] = oof_proba
        oof_has[te_idx] = True

    val_f1s = [m["val_f1"] for m in all_metrics]
    mean_f1 = float(np.mean(val_f1s))
    std_f1 = float(np.std(val_f1s))

    oof_df = pd.DataFrame({
        "coin": meta_df["coin"].values,
        "p0": oof_proba_all[:, 0],
        "p1": oof_proba_all[:, 1],
        "p2": oof_proba_all[:, 2],
        "has_oof": oof_has,
        "momentum_v4_label": y.astype(np.int8),
        "is_gate": np.ones(len(y), dtype=np.int8),
        "vol_spike": meta_df["vol_spike"].values,
        "hmm_enc": meta_df["hmm_enc"].values.astype(np.int8),
        "lgbm_p_short": meta_df["lgbm_p_short"].values,
        "lgbm_p_flat": meta_df["lgbm_p_flat"].values,
        "lgbm_p_long": meta_df["lgbm_p_long"].values,
    }, index=pd.to_datetime(ts, utc=True))
    oof_df.to_parquet(run_dir / "oof_lstm_predictions.parquet")
    logger.info(f"Saved oof_lstm_predictions.parquet ({oof_has.sum():,} bars)")

    sweep_results, best_thr, _ = _base.sweep_complement_thr(
        oof_proba_all[oof_has],
        meta_df.iloc[np.where(oof_has)[0]].reset_index(drop=True),
        y[oof_has],
    )

    print(f"\n-- Complement Gate Sweep (OOF, LGBM flat + vol_spike>={_base.VOL_SPIKE_THR}) --")
    print(f"  {'thr':>5} {'n':>7} {'prec_dir':>9} {'prec_all':>9} {'cov%':>6}")
    for r in sweep_results:
        print(
            f"  {r['lstm_thr']:>5.2f} {r.get('n_complement', 0):>7,} "
            f"{r.get('precision_directional', 0):>9.3f} "
            f"{r.get('precision_all', 0):>9.3f} "
            f"{r.get('coverage_pct', 0):>6.1f}"
        )
    print(f"  BEST lstm_thr = {best_thr}")

    with open(run_dir / "best_lstm_complement.json", "w") as f:
        json.dump({
            "lstm_comp_thr": best_thr,
            "vol_spike_thr": _base.VOL_SPIKE_THR,
            "hmm_thr_cfg": {str(k): v for k, v in _base.HMM_THR_CFG.items()},
            "sweep_method": "OOF_complement_simulation",
            "sweep_all": sweep_results,
            "created": datetime.now().isoformat(),
        }, f, indent=2)

    avg_epochs = int(np.median([m.get("best_epoch", 30) for m in all_metrics]))
    final_epochs = max(20, min(avg_epochs + 5, LSTM_EPOCHS))

    logger.info(f"\nRetraining final model ({final_epochs} epochs)...")
    final_model, final_scaler = _base.retrain_final(X, y, final_epochs)

    save_lstm(final_model, run_dir / "lstm_momentum.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_scaler.pkl")

    best_comp = next((r for r in sweep_results if r["lstm_thr"] == best_thr), {})
    baseline_f1 = 0.3987

    meta = {
        "run_name": RUN_NAME,
        "model_type": "lstm_momentum_lgbm_seq",
        "variant": "B_lgbm_history_per_timestep",
        "lgbm_partner": LGBM_RUN,
        "lgbm_input_source": "oof_predictions.parquet has_oof=True",
        "lgbm_input_cols": LGBM_SEQ_COLS,
        "lgbm_seq_policy": "full_window_oof_required_no_ffill",
        "role": "momentum on gate bars with LGBM score context",
        "label_type": "momentum_v4_continuation_option_A",
        "sample_filter": "is_pump_dump_bar == 1",
        "n_market_features": len(market_cols),
        "n_features": len(feat_cols_used),
        "features": feat_cols_used,
        "seq_len": LSTM_SEQ_LEN,
        "purge_gap": TB_PURGE_GAP_BARS,
        "n_folds": N_FOLDS,
        "hidden": LSTM_V2_HIDDEN,
        "layers": LSTM_V2_LAYERS,
        "dropout": LSTM_V2_DROPOUT,
        "n_samples": int(X.shape[0]),
        "n_coins": len(coins),
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "baseline_f1_tb_lstm_genuine_v2": baseline_f1,
        "f1_delta_vs_baseline": round(mean_f1 - baseline_f1, 4),
        "folds": all_metrics,
        "complement_gate": {
            "best_lstm_thr": best_thr,
            "vol_spike_thr": _base.VOL_SPIKE_THR,
            "oof_sweep": sweep_results,
            "best_precision_directional": best_comp.get("precision_directional"),
            "best_n_complement": best_comp.get("n_complement"),
        },
        "inference_note": (
            "Live: append LGBM p0/p1/p2 history per bar to market seq "
            "(same order as features.json). LGBM runs before LSTM."
        ),
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "methodology": (
            "purged CV OOF LSTM, scaler per fold, LGBM OOF per timestep "
            "(no in-sample LGBM, no ffill), gate bars only, holdout not used"
        ),
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  {RUN_NAME} COMPLETE")
    print(f"  CV Mean F1     : {mean_f1:.4f} +/- {std_f1:.4f}  (baseline {baseline_f1})")
    print(f"  F1 delta       : {mean_f1 - baseline_f1:+.4f}")
    print(f"  Complement thr : {best_thr} (OOF sweep)")
    print(f"  Next step      : 05j eval with lstm_run={RUN_NAME}")
    print(f"  OOF saved      : {run_dir}/oof_lstm_predictions.parquet")
    print(f"  Model          : {run_dir}/lstm_momentum.pt")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()