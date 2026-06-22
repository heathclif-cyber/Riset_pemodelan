"""Analisis kelas BEAR -- LSTM momentum v4 OOF."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR

RUN = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"
LGBM = MODEL_DIR / "runs" / "tb_lgbm_genuine_v2"
HMM_THR = {0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50), 3: (0.45, 0.50), -1: (0.45, 0.45)}
VOL = 2.0
THRS = [0.40, 0.45, 0.50, 0.55, 0.60]


def main():
    oof = pd.read_parquet(RUN / "oof_lstm_predictions.parquet")
    oof = oof.loc[oof["has_oof"]].copy()
    y = oof["momentum_v4_label"].values.astype(int)
    proba = oof[["p0", "p1", "p2"]].values
    pred = proba.argmax(1)
    lstm_dir = np.where(
        proba[:, 2] > proba[:, 0], 2,
        np.where(proba[:, 0] > proba[:, 2], 0, 1),
    )

    bear_mask = y == 0
    pred_bear = pred == 0
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)

    print("=" * 60)
    print("  BEAR CLASS -- OOF DETAIL (momentum v4 opsi A)")
    print("=" * 60)
    print(f"  Support BEAR: {bear_mask.sum():,} ({bear_mask.mean()*100:.1f}% gate bars)")
    print(f"  F1={f[0]:.3f}  precision={p[0]:.3f}  recall={r[0]:.3f}")
    print(f"  BULL: F1={f[2]:.3f} prec={p[2]:.3f} rec={r[2]:.3f}")
    print(f"  NEU : F1={f[1]:.3f} prec={p[1]:.3f} rec={r[1]:.3f}")

    true_bear_p0 = oof.loc[oof["momentum_v4_label"] == 0, "p0"]
    print(f"\n  p0 saat true BEAR: p50={true_bear_p0.quantile(0.5):.3f} "
          f"p75={true_bear_p0.quantile(0.75):.3f} p90={true_bear_p0.quantile(0.9):.3f}")

    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    print("\n  Confusion (row=true, col=pred):")
    print("            BEAR      NEU      BULL")
    for i, name in enumerate(["BEAR", "NEU ", "BULL"]):
        print(f"  {name}  {cm[i,0]:>7,} {cm[i,1]:>7,} {cm[i,2]:>7,}")

    false_bull = ((y == 0) & (pred == 2)).sum()
    false_neu = ((y == 0) & (pred == 1)).sum()
    n_bear = bear_mask.sum()
    print(f"\n  True BEAR salah label:")
    print(f"    -> BULL: {false_bull:,} ({false_bull/n_bear*100:.1f}%)")
    print(f"    -> NEU : {false_neu:,} ({false_neu/n_bear*100:.1f}%)")

    # merge LGBM for complement
    lgbm = pd.read_parquet(LGBM / "oof_predictions.parquet")
    lgbm = lgbm.loc[lgbm["has_oof"]].copy()
    lgbm.index = pd.to_datetime(lgbm.index, utc=True)
    lg = lgbm.reset_index()
    lg = lg.rename(columns={lg.columns[0]: "ts"})
    o2 = oof.reset_index().rename(columns={oof.index.name or "index": "ts"})
    if "ts" not in o2.columns:
        o2 = o2.rename(columns={o2.columns[0]: "ts"})
    m = o2.merge(
        lg[["coin", "ts", "p0", "p2"]].rename(columns={"p0": "p0_lgbm", "p2": "p2_lgbm"}),
        on=["coin", "ts"], how="inner",
    )

    p0g = m["p0_lgbm"].values
    p2g = m["p2_lgbm"].values
    hmm = m["hmm_enc"].values.astype(int)
    tl = np.full(len(p0g), 0.45)
    ts_ = np.full(len(p0g), 0.45)
    for st, (l, s) in HMM_THR.items():
        if st == -1:
            continue
        mask = hmm == st
        tl[mask], ts_[mask] = l, s
    long_sig = p2g >= tl
    short_sig = (p0g >= ts_) & ~long_sig
    flat = ~(long_sig | short_sig)
    vol = m["vol_spike"].values >= VOL

    proba_m = m[["p0", "p1", "p2"]].values
    dom_m = np.maximum(proba_m[:, 0], proba_m[:, 2])
    dir_m = np.where(
        proba_m[:, 2] > proba_m[:, 0], 2,
        np.where(proba_m[:, 0] > proba_m[:, 2], 0, 1),
    )
    labels_m = m["momentum_v4_label"].values

    print(f"\n  COMPLEMENT mixed (LGBM flat + vol_spike>={VOL})")
    print(f"  {'thr':>5} {'n':>7} {'prec_dir':>9} {'trueBEAR':>9} {'predBEAR':>9}")
    for thr in THRS:
        conf = (dir_m != 1) & (dom_m >= thr)
        comp = flat & vol & conf
        n = comp.sum()
        if n == 0:
            continue
        d = dir_m[comp]
        lb = labels_m[comp]
        correct = ((d == 0) & (lb == 0)) | ((d == 2) & (lb == 2))
        dir_lbl = lb != 1
        prec = correct[dir_lbl].mean() if dir_lbl.any() else 0
        print(f"  {thr:>5.2f} {n:>7,} {prec:>9.3f} {(lb==0).mean()*100:>8.1f}% {(d==0).mean()*100:>8.1f}%")

    print("\n  BEAR-only complement (LSTM pred BEAR + conf >= thr)")
    pool = flat & vol
    base_bear_rate = (labels_m[pool] == 0).mean()
    print(f"  Baseline BEAR% di pool (flat+vol): {base_bear_rate*100:.1f}%")
    for thr in THRS:
        comp = pool & (dir_m == 0) & (proba_m[:, 0] >= thr)
        n = comp.sum()
        if n == 0:
            continue
        prec = (labels_m[comp] == 0).mean()
        print(f"  thr={thr:.2f}  n={n:>5,}  BEAR precision={prec:.3f}  lift vs base={prec/base_bear_rate:.2f}x")


if __name__ == "__main__":
    main()