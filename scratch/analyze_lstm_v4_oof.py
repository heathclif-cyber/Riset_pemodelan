"""Analisis OOF LSTM momentum v4 opsi A -- training period only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR

RUN = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"
LGBM_RUN = MODEL_DIR / "runs" / "tb_lgbm_genuine_v2"

HMM_THR_CFG = {
    0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50),
    3: (0.45, 0.50), -1: (0.45, 0.45),
}
VOL_SPIKE_THR = 2.0
LSTM_THRS = [0.40, 0.45, 0.50, 0.55, 0.60]
MAP = {0: "BEAR", 1: "NEU", 2: "BULL"}


def lgbm_flat(p0, p2, hmm):
    tl = np.full(len(p0), HMM_THR_CFG[-1][0])
    ts = np.full(len(p0), HMM_THR_CFG[-1][1])
    for st, (l, s) in HMM_THR_CFG.items():
        if st == -1:
            continue
        m = hmm == st
        tl[m], ts[m] = l, s
    long_sig = p2 >= tl
    short_sig = (p0 >= ts) & ~long_sig
    return ~(long_sig | short_sig)


def main():
    oof = pd.read_parquet(RUN / "oof_lstm_predictions.parquet")
    oof = oof.loc[oof["has_oof"]].copy()
    print("=" * 72)
    print("  LSTM momentum v4 (opsi A) -- OOF ANALYSIS")
    print("=" * 72)
    print(f"\n  OOF bars: {len(oof):,}")

    y = oof["momentum_v4_label"].values.astype(int)
    pred = np.argmax(oof[["p0", "p1", "p2"]].values, axis=1)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    f1_cls = f1_score(y, pred, average=None, zero_division=0, labels=[0, 1, 2])
    print(f"\n  OOF F1 macro: {f1:.4f}")
    print(f"  Per-class F1: BEAR={f1_cls[0]:.3f}  NEU={f1_cls[1]:.3f}  BULL={f1_cls[2]:.3f}")
    print(f"  Random baseline (3-class): 0.333")

    vc = pd.Series(y).value_counts(normalize=True)
    print(f"\n  Label dist: BEAR={vc.get(0,0)*100:.1f}%  NEU={vc.get(1,0)*100:.1f}%  BULL={vc.get(2,0)*100:.1f}%")

    print("\n  Classification report:")
    print(classification_report(y, pred, target_names=["BEAR", "NEU", "BULL"], zero_division=0))

    # BULL precision/recall -- kunci untuk complement
    bull_mask = y == 2
    pred_bull = pred == 2
    bull_prec = (y[pred_bull] == 2).mean() if pred_bull.any() else 0
    bull_rec = (pred[bull_mask] == 2).mean() if bull_mask.any() else 0
    print(f"  BULL precision: {bull_prec:.3f}  recall: {bull_rec:.3f}")

    # Complement gate
    lgbm = pd.read_parquet(LGBM_RUN / "oof_predictions.parquet")
    lgbm = lgbm.loc[lgbm["has_oof"]].copy()
    lgbm.index = pd.to_datetime(lgbm.index, utc=True)
    lgbm_df = lgbm.reset_index().rename(columns={"index": "ts"})
    if "ts" not in lgbm_df.columns:
        lgbm_df = lgbm_df.rename(columns={lgbm_df.columns[0]: "ts"})

    oof2 = oof.reset_index().rename(columns={"index": "ts"})
    if "ts" not in oof2.columns:
        oof2 = oof2.rename(columns={oof2.columns[0]: "ts"})

    m = oof2.merge(
        lgbm_df[["coin", "ts", "p0", "p2"]].rename(columns={"p0": "p0_lgbm", "p2": "p2_lgbm"}),
        on=["coin", "ts"],
        how="inner",
    )
    print(f"\n  Joined LGBM+LSTM OOF: {len(m):,} bars")

    flat = lgbm_flat(m["p0_lgbm"].values, m["p2_lgbm"].values, m["hmm_enc"].values.astype(np.int8))
    vol_hi = m["vol_spike"].values >= VOL_SPIKE_THR
    lstm_dom = np.maximum(m["p0"].values, m["p2"].values)
    lstm_dir = np.where(m["p2"].values > m["p0"].values, 2, np.where(m["p0"].values > m["p2"].values, 0, 1))

    print(f"\n  COMPLEMENT GATE SWEEP (LGBM flat + vol_spike>={VOL_SPIKE_THR})")
    print(f"  {'thr':>5} {'n':>8} {'prec_dir':>9} {'bull%':>6} {'bear%':>6}")
    best = None
    for thr in LSTM_THRS:
        conf = (lstm_dir != 1) & (lstm_dom >= thr)
        comp = flat & vol_hi & conf
        n = comp.sum()
        if n == 0:
            continue
        labels = m["momentum_v4_label"].values[comp]
        dirs = lstm_dir[comp]
        correct = ((dirs == 2) & (labels == 2)) | ((dirs == 0) & (labels == 0))
        dir_lbl = labels != 1
        prec = correct[dir_lbl].mean() if dir_lbl.any() else 0
        print(
            f"  {thr:>5.2f} {n:>8,} {prec:>9.3f} "
            f"{(dirs==2).mean()*100:>5.1f}% {(dirs==0).mean()*100:>5.1f}%"
        )
        if best is None or prec > best[1]:
            best = (thr, prec, n)

    if best:
        print(f"\n  Best complement thr: {best[0]} (prec_dir={best[1]:.3f}, n={best[2]:,})")

    # Confidence calibration
    print(f"\n  LSTM confidence when true BULL (n={bull_mask.sum():,}):")
    bull_probs = m.loc[m["momentum_v4_label"] == 2, "p2"]
    for q in [0.5, 0.75, 0.9]:
        print(f"    p50/p75/p90 p2: {bull_probs.quantile(0.5):.3f} / {bull_probs.quantile(0.75):.3f} / {bull_probs.quantile(0.9):.3f}")


if __name__ == "__main__":
    main()