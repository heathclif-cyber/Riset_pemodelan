# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.live_db_bridge import load_signals, LOCAL_DB

sig = load_signals(LOCAL_DB)
sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True)
sig = sig.sort_values("signal_time", ascending=False)
cut = pd.Timestamp("2026-06-18 15:39:00", tz="UTC")
post = sig[sig["signal_time"] >= cut]
print(f"Total={len(sig)} pre_fix={len(sig[sig['signal_time'] < cut])} post_fix={len(post)}")
print("Latest 15:")
for _, r in sig.head(15).iterrows():
    fs = json.loads(r["feature_snapshot"] or "{}")
    ts = str(r["signal_time"])[:19]
    print(f"  {r['coin_symbol']:14s} {ts} LSR={fs.get('long_short_ratio')} HMM={fs.get('hmm_regime_enc')}")
if len(post):
    lsrs = [json.loads(r["feature_snapshot"] or "{}").get("long_short_ratio") for _, r in post.iterrows()]
    print(f"Post-fix LSR zero: {sum(1 for x in lsrs if x == 0 or x == 0.0)}/{len(lsrs)}")