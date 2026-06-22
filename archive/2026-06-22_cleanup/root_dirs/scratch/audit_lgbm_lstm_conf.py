"""Quick audit: LGBM conf vs LSTM OOF overlap and correlation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LSTM_OOF = ROOT / "models/runs/tb_lstm_genuine_v2/oof_lstm_predictions.parquet"
LGBM_OOF = ROOT / "models/runs/tb_lgbm_genuine_v2/oof_predictions.parquet"
FEAT_JSON = ROOT / "models/runs/tb_lstm_genuine_v2/lstm_v4_selected_features.json"

VOL_SPIKE_THR = 2.0


def main():
    lstm = pd.read_parquet(LSTM_OOF)
    lgbm = pd.read_parquet(LGBM_OOF)

    print("=== FILE STRUCTURE ===")
    print(f"LSTM OOF: {lstm.shape} cols={list(lstm.columns)} index={lstm.index.names}")
    print(f"LGBM OOF: {lgbm.shape} cols={list(lgbm.columns)} index={lgbm.index.names}")

    with open(FEAT_JSON) as f:
        lstm_feats = json.load(f)
    print(f"\nLSTM training features ({len(lstm_feats)}): {lstm_feats}")
    lgbm_feat_in_lstm = [c for c in lstm_feats if c in lgbm.columns]
    print(f"Overlap with LGBM OOF cols: {lgbm_feat_in_lstm}")

    # Normalize keys for merge
    lstm_m = lstm.reset_index()
    lgbm_m = lgbm.reset_index()

    # Detect timestamp column
    for df_name, df in [("lstm", lstm_m), ("lgbm", lgbm_m)]:
        ts_cols = [c for c in df.columns if c in ("ts", "timestamp", "index") or "time" in c.lower()]
        print(f"{df_name} ts-like cols: {ts_cols[:5]}")

    # Common merge keys
    if "coin" not in lstm_m.columns:
        print("LSTM missing coin column after reset_index")
        print(lstm_m.head(2))
        return

    # LGBM may use index as ts
    if "ts" not in lgbm_m.columns:
        if lgbm.index.name:
            lgbm_m = lgbm_m.rename(columns={lgbm.index.name: "ts"})
        elif "index" in lgbm_m.columns:
            lgbm_m = lgbm_m.rename(columns={"index": "ts"})

    if "ts" not in lstm_m.columns:
        if lstm.index.name:
            lstm_m = lstm_m.rename(columns={lstm.index.name: "ts"})
        elif "index" in lstm_m.columns:
            lstm_m = lstm_m.rename(columns={"index": "ts"})

    merge_cols = ["coin", "ts"]
    lgbm_keep = ["coin", "ts", "p0", "p1", "p2"]
    lgbm_m = lgbm_m[lgbm_keep].rename(columns={"p0": "p0_lgbm", "p1": "p1_lgbm", "p2": "p2_lgbm"})
    lstm_keep = ["coin", "ts", "p0", "p1", "p2", "vol_spike", "hmm_enc"]
    lstm_m = lstm_m[[c for c in lstm_keep if c in lstm_m.columns]].rename(
        columns={"p0": "p0_lstm", "p1": "p1_lstm", "p2": "p2_lstm"}
    )
    merged = lstm_m.merge(lgbm_m, on=merge_cols, how="inner")
    print(f"\n=== MERGE on {merge_cols} ===")
    print(f"Merged rows: {len(merged):,} / LSTM {len(lstm_m):,} / LGBM {len(lgbm_m):,}")

    merged["lgbm_conf"] = merged[["p0_lgbm", "p2_lgbm"]].max(axis=1)
    merged["lgbm_dir"] = np.where(merged["p0_lgbm"] >= merged["p2_lgbm"], 0, 2)
    merged["lstm_bull"] = merged["p2_lstm"]
    merged["lstm_bear"] = merged["p0_lstm"]
    merged["lstm_conf"] = merged[["p0_lstm", "p2_lstm"]].max(axis=1)
    HMM_THR = {0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50), 3: (0.45, 0.50), -1: (0.45, 0.45)}
    tl = np.full(len(merged), 0.45)
    ts_thr = np.full(len(merged), 0.45)
    hmm = merged["hmm_enc"].values.astype(int) if "hmm_enc" in merged.columns else np.full(len(merged), -1)
    for state, (tl_s, ts_s) in HMM_THR.items():
        m = hmm == state
        tl[m], ts_thr[m] = tl_s, ts_s
    long_sig = merged["p2_lgbm"].values >= tl
    short_sig = (merged["p0_lgbm"].values >= ts_thr) & ~long_sig
    merged["lgbm_flat"] = ~(long_sig | short_sig)

    vol_col = "vol_spike" if "vol_spike" in merged.columns else "vol_spike_zscore"
    if vol_col in merged.columns:
        gate = merged[vol_col] >= VOL_SPIKE_THR
        complement = gate & merged["lgbm_flat"]
        print(f"\n=== CORRELATION (vol_spike >= {VOL_SPIKE_THR}) ===")
        for name, sub in [
            ("all merged", merged),
            ("gate bars", merged[gate]),
            ("complement (flat+vol)", merged[complement]),
        ]:
            print(f"\n[{name}] n={len(sub):,}")
            print(f"  corr(lgbm_conf, lstm_conf): {sub['lgbm_conf'].corr(sub['lstm_conf']):.4f}")
            print(f"  corr(lgbm_conf, lstm_bull): {sub['lgbm_conf'].corr(sub['lstm_bull']):.4f}")
            print(f"  corr(p0_lgbm, p0_lstm): {sub['p0_lgbm'].corr(sub['p0_lstm']):.4f}")
            print(f"  corr(p2_lgbm, p2_lstm): {sub['p2_lgbm'].corr(sub['p2_lstm']):.4f}")
        sub = merged[complement]
        short_lstm_bull = sub[(sub["lgbm_dir"] == 0) & (sub["lstm_bull"] >= 0.38)]
        print(f"\nLGBM SHORT + LSTM bull>=0.38 on complement bars: {len(short_lstm_bull):,}")
        lgbm_short_gate = merged[gate & (merged["p0_lgbm"] >= ts_thr) & ~long_sig]
        print(f"LGBM SHORT signal on gate bars: {len(lgbm_short_gate):,}")
        conflict = lgbm_short_gate[lgbm_short_gate["lstm_bull"] >= 0.38]
        print(f"  of which LSTM bull>=0.38 (fusion penalty zone): {len(conflict):,}")

    print("\n=== LGBM conf IN LSTM training? ===")
    print("NO - lstm_v4_selected_features.json has only market OHLCV/CVD/OFI feats")
    print("LGBM p0/p2 used only in complement sweep (post-hoc), not in load_data() sequences")


if __name__ == "__main__":
    main()