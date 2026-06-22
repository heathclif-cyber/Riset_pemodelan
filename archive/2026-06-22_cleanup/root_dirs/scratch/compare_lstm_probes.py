"""Fair complement comparison across baseline + probe A/B."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR

spec = importlib.util.spec_from_file_location(
    "lstm_train", ROOT / "pipeline" / "05_train_lstm_genuine_v2.py"
)
_lstm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_lstm)

spec2 = importlib.util.spec_from_file_location(
    "probe_b", ROOT / "pipeline" / "05d_lstm_probe_asym_b.py"
)
_probe_b = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(_probe_b)

RUNS = [
    ("baseline 8-fold", "tb_lstm_genuine_v2", 0.40, 0.50),
    ("baseline sym 0.40", "tb_lstm_genuine_v2", 0.40, 0.40),
    ("probe-A", "tb_lstm_probe_asym_a", 0.40, 0.55),
    ("probe-B", "tb_lstm_probe_asym_b", 0.40, 0.50),
]


def eval_run(name, run, bull_thr, bear_thr):
    oof = pd.read_parquet(MODEL_DIR / "runs" / run / "oof_lstm_predictions.parquet")
    oof = oof.loc[oof["has_oof"]]
    proba = oof[["p0", "p1", "p2"]].values
    y = oof["momentum_v4_label"].values.astype(int)

    meta = oof.reset_index().rename(columns={"index": "ts"})
    if "ts" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "ts"})
    meta = meta[["ts", "coin", "vol_spike", "hmm_enc"]].copy()
    meta["ts"] = pd.to_datetime(meta["ts"], utc=True)

    pred = proba.argmax(1)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    f1_bear = f1_score(y, pred, average=None, zero_division=0, labels=[0, 1, 2])[0]

    frame = _lstm.build_complement_frame(proba, meta, y)
    asym = _probe_b.complement_asymmetric(frame, bull_thr, bear_thr, 2.0)
    return f1, f1_bear, asym


def main():
    print("Run                  F1mac  F1bear | n_comp  n_bull  n_bear | bull_p bear_p mixed_p | pool")
    print("-" * 95)
    for name, run, bt, br in RUNS:
        f1, fb, a = eval_run(name, run, bt, br)
        print(
            f"{name:18} {f1:.4f}  {fb:.3f}  | "
            f"{a['n_complement']:6,} {a['n_bull']:6,} {a['n_bear']:6,} | "
            f"{a['bull_precision']:.3f}  {a['bear_precision']:.3f}  {a['mixed_precision_dir']:.3f}  | "
            f"{a['n_pool']:,}"
        )


if __name__ == "__main__":
    main()