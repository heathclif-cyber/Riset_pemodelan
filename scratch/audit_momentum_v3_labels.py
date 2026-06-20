"""Audit momentum_v3 labels vs known pump/dump events (holdout May-Jun 2026)."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT))

from config import HOLDOUT_DIR, TRAINING_COINS

HOLD = HOLDOUT_DIR / "labeled"
MAP = {0: "BEAR", 1: "NEU", 2: "BULL"}


def per_coin_z(s, w=500):
    s = pd.Series(s)
    w = min(w, max(len(s) // 3, 50))
    m = s.rolling(w, min_periods=30).mean()
    sd = s.rolling(w, min_periods=30).std().clip(lower=1e-8)
    return ((s - m) / sd).clip(-4, 4).fillna(0).values


def pump_mask(df):
    vs = df["vol_spike_zscore"] if "vol_spike_zscore" in df.columns else 0
    re = df["range_expansion_h4"] if "range_expansion_h4" in df.columns else 0
    return (vs >= 2) | (re >= 1.5)


def label_broken(df, N=8, flow_thr=0.3, div_thr=0.5):
    n = len(df)
    labels = np.ones(n, dtype=int)
    cvd = df.get("cvd_slope_h4", pd.Series(0, index=df.index)).values
    ofi = df.get("ofi_h4_delta", pd.Series(0, index=df.index)).values
    vd = per_coin_z(df["volume_delta"].values.astype(float)) if "volume_delta" in df.columns else np.zeros(n)
    lr = df.get("log_ret_1", pd.Series(0, index=df.index)).values
    for t in range(n - N - 1):
        end = t + N + 1
        fwd = slice(t + 1, end)
        fr = np.clip(
            float(np.nanmean(np.diff(cvd[t:end])) + np.nanmean(np.diff(ofi[t:end])) + np.nanmean(vd[fwd])),
            -4, 4,
        )
        pr = np.clip(float(np.nanmean(lr[fwd])) * 100, -4, 4)
        d = pr - fr
        if fr > flow_thr and d < -div_thr:
            labels[t] = 2
        elif fr > flow_thr and d > div_thr:
            labels[t] = 0
        elif fr < -flow_thr and d > div_thr:
            labels[t] = 0
    return labels


def label_fixed(df, N=8, flow_thr=0.5, div_thr=0.5):
    n = len(df)
    labels = np.ones(n, dtype=int)
    cvd = df.get("cvd_slope_h4", pd.Series(0, index=df.index)).values
    ofi = df.get("ofi_h4_delta", pd.Series(0, index=df.index)).values
    vd = per_coin_z(df["volume_delta"].values.astype(float)) if "volume_delta" in df.columns else np.zeros(n)
    lr = df.get("log_ret_1", pd.Series(0, index=df.index)).values
    flow_raw, price_raw, idxs = [], [], []
    for t in range(n - N - 1):
        end = t + N + 1
        fwd = slice(t + 1, end)
        flow_raw.append(
            float(np.nanmean(np.diff(cvd[t:end])) + np.nanmean(np.diff(ofi[t:end])) + np.nanmean(vd[fwd]))
        )
        price_raw.append(float(np.nanmean(lr[fwd])))
        idxs.append(t)
    fr = np.array(flow_raw)
    pr = np.array(price_raw)
    fz = (fr - fr.mean()) / (fr.std() + 1e-9)
    pz = (pr - pr.mean()) / (pr.std() + 1e-9)
    for i, t in enumerate(idxs):
        d = pz[i] - fz[i]
        if fz[i] > flow_thr and d < -div_thr:
            labels[t] = 2
        elif fz[i] > flow_thr and d > div_thr:
            labels[t] = 0
        elif fz[i] < -flow_thr and d > div_thr:
            labels[t] = 0
    return labels


def audit_coin(coin, days: list):
    path = HOLD / f"{coin}_features_v3.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path).sort_index()
    lb = label_broken(df)
    lf = label_fixed(df)
    df = df.copy()
    df["lbl_broken"] = lb
    df["lbl_fixed"] = lf
    df["is_pump_bar"] = pump_mask(df)

    print(f"\n{'='*72}")
    print(f"  {coin} -- label audit on known event days")
    print(f"{'='*72}")
    for day in days:
        m = (df.index >= day) & (df.index < pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1))
        sub = df.loc[m]
        if sub.empty:
            print(f"  {day}: no data")
            continue
        ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100
        pump_pct = sub["is_pump_bar"].mean() * 100
        b_dist = sub["lbl_broken"].value_counts(normalize=True).to_dict()
        f_dist = sub["lbl_fixed"].value_counts(normalize=True).to_dict()

        def fmt(d):
            return f"BEAR={d.get(0,0)*100:.0f}% NEU={d.get(1,0)*100:.0f}% BULL={d.get(2,0)*100:.0f}%"

        print(f"\n  {day}  daily_ret={ret:+.1f}%  pump_bars={pump_pct:.0f}%")
        print(f"    BROKEN: {fmt(b_dist)}")
        print(f"    FIXED:  {fmt(f_dist)}")

        # show peak pump hours
        if "vol_spike_zscore" in sub.columns:
            top = sub.nlargest(3, "vol_spike_zscore")[["close", "vol_spike_zscore", "lbl_broken", "lbl_fixed"]]
            print(f"    Top vol_spike hours:")
            for ts, row in top.iterrows():
                print(f"      {ts}  vol_spike={row['vol_spike_zscore']:.2f}  "
                      f"broken={MAP[int(row['lbl_broken'])]}  fixed={MAP[int(row['lbl_fixed'])]}")


def find_event_days():
    print("\n" + "=" * 72)
    print("  AUTO-DETECT: top pump days May-Jun 2026 + market dump days")
    print("=" * 72)

    pump_days = {}
    for coin in ["TAOUSDT", "TONUSDT", "NEARUSDT"]:
        df = pd.read_parquet(HOLD / f"{coin}_features_v3.parquet").sort_index()
        m = (df.index >= "2026-05-01") & (df.index < "2026-07-01")
        sub = df.loc[m].assign(day=df.loc[m].index.floor("D"))
        daily = sub.groupby("day").apply(
            lambda g: (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100, include_groups=False
        ).sort_values(ascending=False)
        print(f"\n  {coin} best pump days:")
        for d, r in daily.head(5).items():
            print(f"    {d.date()}  ret={r:+.1f}%")
        pump_days[coin] = [str(d.date()) for d in daily.head(3).index]

    # market-wide dump
    rows = []
    for coin in TRAINING_COINS:
        p = HOLD / f"{coin}_features_v3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        m = (df.index >= "2026-04-01") & (df.index < "2026-07-01")
        sub = df.loc[m].assign(day=df.loc[m].index.floor("D"))
        for day, g in sub.groupby("day"):
            ret = (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100 if len(g) > 1 else 0
            rows.append({"day": day, "coin": coin, "ret": ret})
    all_d = pd.DataFrame(rows)
    med = all_d.groupby("day")["ret"].median().sort_values()
    print("\n  Market-wide dump days (worst median return, 21 coins):")
    dump_days = []
    btc = pd.read_parquet(HOLD / "BTCUSDT_features_v3.parquet").sort_index()
    for day, v in med.head(8).items():
        bm = (btc.index >= day) & (btc.index < day + pd.Timedelta(days=1))
        btc_ret = (btc.loc[bm, "close"].iloc[-1] / btc.loc[bm, "close"].iloc[0] - 1) * 100 if bm.sum() > 1 else 0
        n_red = (all_d[all_d["day"] == day]["ret"] < 0).sum()
        print(f"    {day.date()}  median={v:+.2f}%  BTC={btc_ret:+.2f}%  coins_red={n_red}/21")
        dump_days.append(str(day.date()))

    return pump_days, dump_days


if __name__ == "__main__":
    pump_days, dump_days = find_event_days()

    for coin, days in pump_days.items():
        audit_coin(coin, days)

    print(f"\n\n{'='*72}")
    print("  MARKET DUMP DAYS -- BTC + alts label check")
    print("=" * 72)
    for day in dump_days[:5]:
        audit_coin("BTCUSDT", [day])
        audit_coin("NEARUSDT", [day])