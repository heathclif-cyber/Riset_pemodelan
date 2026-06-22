"""Probe complement asimetris pada OOF existing -- tanpa retrain."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR

RUN = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"
LGBM = MODEL_DIR / "runs" / "tb_lgbm_genuine_v2"

BULL_THR, BEAR_THR = 0.40, 0.55
VOL = 2.0
HMM_THR = {0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50), 3: (0.45, 0.50), -1: (0.45, 0.45)}


def lgbm_flat(p0, p2, hmm):
    tl = np.full(len(p0), 0.45)
    ts = np.full(len(p0), 0.45)
    for st, (l, s) in HMM_THR.items():
        if st == -1:
            continue
        m = hmm == st
        tl[m], ts[m] = l, s
    long_sig = p2 >= tl
    short_sig = (p0 >= ts) & ~long_sig
    return ~(long_sig | short_sig)


def eval_asym(m, bull_thr, bear_thr):
    flat = lgbm_flat(m["p0_lgbm"].values, m["p2_lgbm"].values, m["hmm_enc"].values.astype(int))
    vol = m["vol_spike"].values >= VOL
    pool = flat & vol
    p0, p2 = m["p0"].values, m["p2"].values
    labels = m["momentum_v4_label"].values

    bull_fire = pool & (p2 > p0) & (p2 >= bull_thr)
    bear_fire = pool & (p0 > p2) & (p0 >= bear_thr)
    comp = bull_fire | bear_fire

    def side_stats(mask, cls):
        n = mask.sum()
        if n == 0:
            return {"n": 0, "precision": 0.0}
        return {"n": int(n), "precision": round(float((labels[mask] == cls).mean()), 4)}

    bull_s = side_stats(bull_fire, 2)
    bear_s = side_stats(bear_fire, 0)
    dirs = np.where(bull_fire, 2, np.where(bear_fire, 0, -1))
    active = comp
    correct = ((dirs == 2) & (labels == 2)) | ((dirs == 0) & (labels == 0))
    dir_lbl = labels[active] != 1
    prec_mix = correct[active][dir_lbl].mean() if dir_lbl.any() else 0

    return {
        "bull_thr": bull_thr, "bear_thr": bear_thr,
        "n_total": int(comp.sum()), "n_bull": bull_s["n"], "n_bear": bear_s["n"],
        "bull_precision": bull_s["precision"], "bear_precision": bear_s["precision"],
        "mixed_precision_dir": round(float(prec_mix), 4),
        "pool_n": int(pool.sum()),
    }


def main():
    oof = pd.read_parquet(RUN / "oof_lstm_predictions.parquet").loc[lambda d: d["has_oof"]]
    lgbm = pd.read_parquet(LGBM / "oof_predictions.parquet").loc[lambda d: d["has_oof"]]
    lgbm.index = pd.to_datetime(lgbm.index, utc=True)
    lg = lgbm.reset_index().rename(columns={lgbm.index.name or "index": "ts"})
    o2 = oof.reset_index().rename(columns={oof.index.name or "index": "ts"})
    m = o2.merge(
        lg[["coin", "ts", "p0", "p2"]].rename(columns={"p0": "p0_lgbm", "p2": "p2_lgbm"}),
        on=["coin", "ts"], how="inner",
    )

    print("=" * 64)
    print("  PROBE A: asymmetric complement on EXISTING OOF")
    print("=" * 64)

    sym = eval_asym(m, BULL_THR, BEAR_THR)
    uni = eval_asym(m, 0.40, 0.40)

    print(f"\n  Symmetric thr=0.40:")
    print(f"    n={sym['n_total']:,}  mixed_prec={uni['mixed_precision_dir']:.3f}")
    print(f"    bull_prec={uni['bull_precision']:.3f}  bear_prec={uni['bear_precision']:.3f}")

    print(f"\n  Asymmetric BULL={BULL_THR} BEAR={BEAR_THR}:")
    print(f"    n={sym['n_total']:,} (bull={sym['n_bull']:,} bear={sym['n_bear']:,})")
    print(f"    bull_prec={sym['bull_precision']:.3f}  bear_prec={sym['bear_precision']:.3f}")
    print(f"    mixed_prec_dir={sym['mixed_precision_dir']:.3f}")

    print("\n  Grid (bull_thr x bear_thr) -- mixed_precision_dir:")
    print(f"  {'bull':>6} ", end="")
    for bt in [0.45, 0.50, 0.55, 0.60]:
        print(f"bear={bt:.2f} ", end="")
    print()
    for bull in [0.35, 0.40, 0.45]:
        print(f"  {bull:.2f}   ", end="")
        for bear in [0.45, 0.50, 0.55, 0.60]:
            r = eval_asym(m, bull, bear)
            print(f" {r['mixed_precision_dir']:.3f}  ", end="")
        print()


if __name__ == "__main__":
    main()