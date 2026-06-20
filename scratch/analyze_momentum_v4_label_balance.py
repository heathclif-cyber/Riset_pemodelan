"""Analisis distribusi label momentum_v4 -- kenapa BEAR/BULL seimbang?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LABEL_DIR, TRAINING_COINS, TRAIN_CUTOFF_DATE

FWD_N = 8
VOL_SPIKE_THR = 2.0
RANGE_EXP_THR = 1.5
MARKET_STRESS_THR = -0.01
MARKET_RET_BARS = 4


def pump_dump_gate(df):
    vs = df.get("vol_spike_zscore", pd.Series(0, index=df.index))
    re = df.get("range_expansion_h4", pd.Series(0, index=df.index))
    return (vs >= VOL_SPIKE_THR) | (re >= RANGE_EXP_THR)


def build_market_stress(coins):
    frames = []
    for coin in coins:
        p = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        ret4 = np.log(df["close"] / df["close"].shift(MARKET_RET_BARS).replace(0, np.nan))
        frames.append(ret4.rename(coin))
    panel = pd.concat(frames, axis=1)
    return (panel.median(axis=1) < MARKET_STRESS_THR).rename("market_stress")


def decompose_labels(df, stress: pd.Series, bull_thr=0.010, bear_thr=-0.010):
    n = len(df)
    gate = pump_dump_gate(df).values
    close = df["close"].values.astype(np.float64)
    cvd = df.get("cvd_slope_h4", pd.Series(0, index=df.index)).values
    ofi = df.get("ofi_h4_delta", pd.Series(0, index=df.index)).values
    flow = cvd + ofi
    stress_arr = df.index.to_series().map(stress).fillna(False).values.astype(bool)

    rows = []
    for t in range(n - FWD_N):
        if not gate[t]:
            continue
        c0, cN = close[t], close[t + FWD_N]
        if c0 <= 0 or np.isnan(c0) or np.isnan(cN):
            continue
        fwd_ret = float(np.log(cN / c0))
        flow_fwd = float(np.nanmean(np.diff(flow[t:t + FWD_N + 1])))

        label = 1
        bear_reason = None
        if fwd_ret >= bull_thr and flow_fwd > 0:
            label = 2
        elif fwd_ret <= bear_thr:
            label = 0
            bear_reason = "fwd_ret"
        elif stress_arr[t] and fwd_ret < 0:
            label = 0
            bear_reason = "market_stress"
        elif fwd_ret < 0 and flow_fwd < 0:
            label = 0
            bear_reason = "flow_neg"

        rows.append({
            "fwd_ret": fwd_ret,
            "flow_fwd": flow_fwd,
            "label": label,
            "bear_reason": bear_reason,
            "market_stress": bool(stress_arr[t]),
            "vol_spike": float(df["vol_spike_zscore"].iloc[t]) if "vol_spike_zscore" in df else 0,
        })
    return pd.DataFrame(rows)


def simulate_thresholds(df, stress, thresholds):
    """Sweep bull/bear threshold pairs on gate bars."""
    gate_df = decompose_labels(df, stress, bull_thr=0.005, bear_thr=-0.005)
    if gate_df.empty:
        return None

    close = df["close"].values.astype(np.float64)
    gate = pump_dump_gate(df).values
    cvd = df.get("cvd_slope_h4", pd.Series(0, index=df.index)).values
    ofi = df.get("ofi_h4_delta", pd.Series(0, index=df.index)).values
    flow = cvd + ofi
    stress_arr = df.index.to_series().map(stress).fillna(False).values.astype(bool)

    results = []
    for bull_thr, bear_thr in thresholds:
        counts = {0: 0, 1: 0, 2: 0}
        idx = 0
        for t in range(len(df) - FWD_N):
            if not gate[t]:
                continue
            c0, cN = close[t], close[t + FWD_N]
            if c0 <= 0 or np.isnan(c0) or np.isnan(cN):
                continue
            fwd_ret = float(np.log(cN / c0))
            flow_fwd = float(np.nanmean(np.diff(flow[t:t + FWD_N + 1])))
            if fwd_ret >= bull_thr and flow_fwd > 0:
                counts[2] += 1
            elif fwd_ret <= bear_thr or (stress_arr[t] and fwd_ret < 0) or (fwd_ret < 0 and flow_fwd < 0):
                counts[0] += 1
            else:
                counts[1] += 1
        total = sum(counts.values())
        results.append({
            "bull_thr": bull_thr,
            "bear_thr": bear_thr,
            "n": total,
            "BEAR%": counts[0] / total * 100,
            "NEU%": counts[1] / total * 100,
            "BULL%": counts[2] / total * 100,
        })
    return pd.DataFrame(results)


def main():
    print("=" * 72)
    print("  momentum_v4 LABEL BALANCE ANALYSIS (training only)")
    print("=" * 72)

    stress = build_market_stress(TRAINING_COINS)
    all_parts = []

    for coin in TRAINING_COINS:
        p = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        sub = decompose_labels(df, stress)
        if not sub.empty:
            sub["coin"] = coin
            all_parts.append(sub)

    pool = pd.concat(all_parts, ignore_index=True)
    n = len(pool)
    vc = pool["label"].value_counts(normalize=True)

    print(f"\n  Gate bars pooled: {n:,}")
    print(f"  BEAR={vc.get(0,0)*100:.1f}%  NEU={vc.get(1,0)*100:.1f}%  BULL={vc.get(2,0)*100:.1f}%")

    bear = pool[pool["label"] == 0]
    print(f"\n  BEAR breakdown ({len(bear):,} bars):")
    for reason, pct in bear["bear_reason"].value_counts(normalize=True).items():
        print(f"    {reason}: {pct*100:.1f}%")

    print(f"\n  Forward return on gate bars (8h):")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"    p{int(q*100):>2}: {pool['fwd_ret'].quantile(q)*100:+.2f}%")
    print(f"    mean: {pool['fwd_ret'].mean()*100:+.2f}%")

    # BULL bars: how strong is momentum?
    bull = pool[pool["label"] == 2]
    bear_only = pool[pool["label"] == 0]
    print(f"\n  BULL fwd_ret: mean={bull['fwd_ret'].mean()*100:+.2f}%  "
          f"median={bull['fwd_ret'].median()*100:+.2f}%  "
          f"p25={bull['fwd_ret'].quantile(0.25)*100:+.2f}%")
    print(f"  BEAR fwd_ret: mean={bear_only['fwd_ret'].mean()*100:+.2f}%  "
          f"median={bear_only['fwd_ret'].median()*100:+.2f}%")

    # Threshold sensitivity (BTC as proxy, then all coins aggregate)
    print(f"\n  THRESHOLD SWEEP (all coins aggregate):")
    print(f"  {'bull_thr':>8} {'bear_thr':>8} {'BEAR%':>6} {'NEU%':>6} {'BULL%':>6}")
    thresholds = [
        (0.005, -0.005), (0.008, -0.008), (0.010, -0.010),
        (0.015, -0.015), (0.020, -0.020),
        (0.010, -0.005), (0.015, -0.008),
    ]
    agg = {t: {0: 0, 1: 0, 2: 0} for t in thresholds}
    for coin in TRAINING_COINS:
        p = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        gate = pump_dump_gate(df).values
        close = df["close"].values.astype(np.float64)
        cvd = df.get("cvd_slope_h4", pd.Series(0, index=df.index)).values
        ofi = df.get("ofi_h4_delta", pd.Series(0, index=df.index)).values
        flow = cvd + ofi
        stress_arr = df.index.to_series().map(stress).fillna(False).values.astype(bool)
        for bull_thr, bear_thr in thresholds:
            for t in range(len(df) - FWD_N):
                if not gate[t]:
                    continue
                c0, cN = close[t], close[t + FWD_N]
                if c0 <= 0 or np.isnan(c0) or np.isnan(cN):
                    continue
                fwd_ret = float(np.log(cN / c0))
                flow_fwd = float(np.nanmean(np.diff(flow[t:t + FWD_N + 1])))
                key = (bull_thr, bear_thr)
                if fwd_ret >= bull_thr and flow_fwd > 0:
                    agg[key][2] += 1
                elif fwd_ret <= bear_thr or (stress_arr[t] and fwd_ret < 0) or (fwd_ret < 0 and flow_fwd < 0):
                    agg[key][0] += 1
                else:
                    agg[key][1] += 1

    for (bull_thr, bear_thr), counts in agg.items():
        total = sum(counts.values())
        print(
            f"  {bull_thr*100:>7.1f}% {bear_thr*100:>7.1f}% "
            f"{counts[0]/total*100:>5.1f}% {counts[1]/total*100:>5.1f}% {counts[2]/total*100:>5.1f}%"
        )

    # Pump days: what label do gate bars get?
    print(f"\n  PUMP DAYS (top daily ret per coin, gate bar label dist):")
    pump_rows = []
    for coin in TRAINING_COINS[:10]:
        p = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v4_labels.parquet"
        if not p.exists() or not lp.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        df = df.join(lbl, how="inner")
        daily = df.assign(day=df.index.floor("D")).groupby("day").apply(
            lambda g: (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100,
            include_groups=False,
        ).sort_values(ascending=False)
        if daily.empty:
            continue
        best_day = daily.index[0]
        sub = df[df.index.floor("D") == best_day]
        gate = sub[sub["is_pump_dump_bar"] == 1]
        if len(gate) == 0:
            continue
        vc2 = gate["momentum_v4_label"].value_counts(normalize=True)
        pump_rows.append({
            "coin": coin,
            "day": str(best_day.date()),
            "daily_ret": daily.iloc[0],
            "gate_n": len(gate),
            "BULL%": vc2.get(2, 0) * 100,
            "BEAR%": vc2.get(0, 0) * 100,
        })

    for r in sorted(pump_rows, key=lambda x: -x["daily_ret"])[:8]:
        print(
            f"    {r['coin']:<16} {r['day']} ret={r['daily_ret']:+.0f}%  "
            f"gate={r['gate_n']:>3}  BULL={r['BULL%']:.0f}% BEAR={r['BEAR%']:.0f}%"
        )

    print(f"\n  INTERPRETASI:")
    print(f"  - BEAR tinggi karena 3 jalur: fwd<=-0.5%, market_stress+red, flow_neg")
    print(f"  - BULL butuh fwd>=+0.5% DAN flow>0 (lebih ketat dari BEAR)")
    print(f"  - Threshold 0.5% di 8h = mudah tercapai di bar volatil -> banyak directional")
    print(f"  - Untuk 'tangkap momentum kuat', pertimbangkan bull_thr 1.0-1.5%")


if __name__ == "__main__":
    main()