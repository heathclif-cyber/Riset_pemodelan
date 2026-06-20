"""Audit momentum_v4 labels on TRAINING period only -- holdout tidak disentuh.

Validasi label/fitur hanya boleh pakai data < TRAIN_CUTOFF_DATE (OOF period).
Holdout dibuka sekali di akhir setelah config freeze.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LABEL_DIR, TRAINING_COINS, TRAIN_CUTOFF_DATE

BASE = LABEL_DIR
MAP = {0: "BEAR", 1: "NEU", 2: "BULL"}


def load_coin(coin: str) -> pd.DataFrame | None:
    feat = BASE / f"{coin}_features_v3.parquet"
    lbl = BASE / f"{coin}_momentum_v4_labels.parquet"
    if not feat.exists() or not lbl.exists():
        return None
    df = pd.read_parquet(feat).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    lbl_df = pd.read_parquet(lbl).sort_index()
    return df.join(lbl_df, how="inner")


def fmt_dist(labels: pd.Series) -> str:
    vc = labels.value_counts(normalize=True)
    return (
        f"BEAR={vc.get(0, 0)*100:.0f}% "
        f"NEU={vc.get(1, 0)*100:.0f}% "
        f"BULL={vc.get(2, 0)*100:.0f}%"
    )


def audit_day(df: pd.DataFrame, day: str, title: str):
    m = (df.index >= day) & (df.index < pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1))
    sub = df.loc[m]
    if sub.empty:
        print(f"  {day}: no data")
        return

    ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100
    gate = sub["is_pump_dump_bar"] == 1
    gate_pct = gate.mean() * 100

    all_dist = fmt_dist(sub["momentum_v4_label"])
    gate_dist = fmt_dist(sub.loc[gate, "momentum_v4_label"]) if gate.any() else "no gate bars"

    print(f"\n  {title} | {day}  daily_ret={ret:+.1f}%  gate_bars={gate_pct:.0f}%")
    print(f"    ALL bars:  {all_dist}")
    print(f"    ON gate:   {gate_dist}")

    if gate.any() and "vol_spike_zscore" in sub.columns:
        top = sub.loc[gate].nlargest(2, "vol_spike_zscore")[
            ["close", "vol_spike_zscore", "momentum_v4_label"]
        ]
        for ts, row in top.iterrows():
            print(
                f"      peak {ts}  vol_spike={row['vol_spike_zscore']:.2f}  "
                f"label={MAP[int(row['momentum_v4_label'])]}"
            )


def overall_summary():
    print("=" * 72)
    print(f"  momentum_v4 -- TRAINING PERIOD DISTRIBUTION (< {TRAIN_CUTOFF_DATE.date()})")
    print("  Holdout NOT used -- validation on OOF period only")
    print("=" * 72)

    all_rows = []
    gate_rows = []
    for coin in TRAINING_COINS:
        df = load_coin(coin)
        if df is None:
            continue
        all_rows.append(df["momentum_v4_label"])
        gate_rows.append(df.loc[df["is_pump_dump_bar"] == 1, "momentum_v4_label"])

    all_lbl = pd.concat(all_rows)
    gate_lbl = pd.concat(gate_rows)
    n_all = len(all_lbl)
    n_gate = len(gate_lbl)

    print(f"\n  Total bars: {n_all:,}  |  pump/dump gate: {n_gate:,} ({n_gate/n_all*100:.1f}%)")
    print(f"  ALL bars:  {fmt_dist(all_lbl)}")
    print(f"  ON gate:   {fmt_dist(gate_lbl)}")

    non_gate = all_lbl.iloc[:0]
    for coin in TRAINING_COINS:
        df = load_coin(coin)
        if df is None:
            continue
        ng = df.loc[df["is_pump_dump_bar"] == 0, "momentum_v4_label"]
        non_gate = pd.concat([non_gate, ng])

    neu_non_gate = (non_gate == 1).mean() * 100 if len(non_gate) else 0
    print(f"\n  NEUTRAL on non-gate bars: {neu_non_gate:.1f}% (expect 100%)")


def find_pump_days(top_n: int = 5) -> list[dict]:
    events = []
    for coin in TRAINING_COINS:
        df = load_coin(coin)
        if df is None:
            continue
        sub = df.assign(day=df.index.floor("D"))
        daily = sub.groupby("day").apply(
            lambda g: (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100,
            include_groups=False,
        ).sort_values(ascending=False)
        for day, ret in daily.head(top_n).items():
            events.append({"coin": coin, "day": str(day.date()), "daily_ret": ret, "type": "pump"})
    return events


def find_dump_days(top_n: int = 8) -> list[str]:
    rows = []
    for coin in TRAINING_COINS:
        df = load_coin(coin)
        if df is None:
            continue
        sub = df.assign(day=df.index.floor("D"))
        for day, g in sub.groupby("day"):
            if len(g) < 2:
                continue
            ret = (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100
            rows.append({"day": day, "coin": coin, "ret": ret})

    all_d = pd.DataFrame(rows)
    med = all_d.groupby("day")["ret"].median().sort_values()
    return [str(d.date()) for d in med.head(top_n).index]


def sanity_check_pump(events: list[dict]):
    print("\n" + "=" * 72)
    print("  SANITY (training): pump days -- on_gate BULL% vs baseline")
    print("=" * 72)

    baseline_bull = []
    for coin in TRAINING_COINS:
        df = load_coin(coin)
        if df is None:
            continue
        gate = df["is_pump_dump_bar"] == 1
        if gate.any():
            baseline_bull.append((df.loc[gate, "momentum_v4_label"] == 2).mean())

    base = np.mean(baseline_bull) if baseline_bull else 0.17
    print(f"\n  Baseline BULL% on gate (training): {base*100:.1f}%")

    ok, fail = 0, 0
    for ev in sorted(events, key=lambda x: -x["daily_ret"])[:15]:
        df = load_coin(ev["coin"])
        if df is None:
            continue
        day = ev["day"]
        m = (df.index >= day) & (df.index < pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1))
        sub = df.loc[m]
        gate = sub["is_pump_dump_bar"] == 1
        if not gate.any():
            continue
        bull_pct = (sub.loc[gate, "momentum_v4_label"] == 2).mean()
        flag = "OK" if bull_pct >= base else "LOW"
        if bull_pct >= base:
            ok += 1
        else:
            fail += 1
        print(
            f"  {ev['coin']:<16} {day}  ret={ev['daily_ret']:+.1f}%  "
            f"gate_BULL={bull_pct*100:.0f}%  [{flag}]"
        )
    print(f"\n  Pump days with BULL >= baseline: {ok}/{ok+fail}")


def sanity_check_dump(dump_days: list[str]):
    print("\n" + "=" * 72)
    print("  SANITY (training): market dump days -- on_gate BEAR% vs baseline")
    print("=" * 72)

    baseline_bear = []
    for coin in TRAINING_COINS:
        df = load_coin(coin)
        if df is None:
            continue
        gate = df["is_pump_dump_bar"] == 1
        if gate.any():
            baseline_bear.append((df.loc[gate, "momentum_v4_label"] == 0).mean())

    base = np.mean(baseline_bear) if baseline_bear else 0.45
    print(f"\n  Baseline BEAR% on gate (training): {base*100:.1f}%")

    for day in dump_days[:6]:
        bear_pcts = []
        for coin in TRAINING_COINS:
            df = load_coin(coin)
            if df is None:
                continue
            m = (df.index >= day) & (df.index < pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1))
            sub = df.loc[m]
            gate = sub["is_pump_dump_bar"] == 1
            if gate.any():
                bear_pcts.append((sub.loc[gate, "momentum_v4_label"] == 0).mean())

        med_bear = np.median(bear_pcts) if bear_pcts else 0
        flag = "OK" if med_bear >= base else "LOW"
        print(f"  {day}  median_gate_BEAR={med_bear*100:.0f}% across {len(bear_pcts)} coins  [{flag}]")

        btc = load_coin("BTCUSDT")
        if btc is not None:
            audit_day(btc, day, "BTCUSDT dump detail")


def main():
    overall_summary()

    pump_events = find_pump_days(top_n=3)
    dump_days = find_dump_days(top_n=8)

    sanity_check_pump(pump_events)
    sanity_check_dump(dump_days)

    print("\n" + "=" * 72)
    print("  SAMPLE: top 5 pump events in training period")
    print("=" * 72)
    for ev in sorted(pump_events, key=lambda x: -x["daily_ret"])[:5]:
        df = load_coin(ev["coin"])
        if df is not None:
            audit_day(df, ev["day"], f"{ev['coin']} pump +{ev['daily_ret']:.1f}%")


if __name__ == "__main__":
    main()