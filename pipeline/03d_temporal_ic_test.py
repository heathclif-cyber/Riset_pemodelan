"""
pipeline/03d_temporal_ic_test.py — Temporal IC Decay Test (Simon LSTM Gate)
Menjawab: apakah sinyal fitur bertahan lintas waktu?

IC(feat_{t-k}, label_t) untuk k = 0..24
Fitur dengan decay lambat = cocok masuk LSTM sequence.
Fitur yang sinyalnya hanya di t=0 = tidak perlu LSTM.

Metrik:
  half_life   = lag dimana |IC| turun ke 50% dari |IC_0|
  cum_ic_lag  = sum |IC_k| untuk k=1..seq_len (total temporal signal)
  sign_flip   = apakah ada sign flip signifikan di lag tertentu (mean-reversion)

Verdict:
  STRONG   (half_life >= 4)  → sertakan di LSTM, temporal pattern kuat
  MODERATE (half_life 2-4)   → opsional, ada sinyal tapi cepat decay
  WEAK     (half_life < 2)   → skip dari LSTM, sinyal hanya di snapshot

Usage:
    python pipeline/03d_temporal_ic_test.py
    python pipeline/03d_temporal_ic_test.py --feature-cols models/feature_cols_ic32.json
    python pipeline/03d_temporal_ic_test.py --seq-len 32 --lags 0 1 2 4 8 12 16 24 32
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_CUTOFF_DATE, LABEL_DIR

LABEL_ORDINAL   = {"SHORT": -1, "FLAT": 0, "LONG": 1}
AUTOCORR_FACTOR = 24
DEFAULT_LAGS    = [0, 1, 2, 4, 8, 12, 16, 24, 32]

# Threshold verdict
STRONG_HALFLIFE   = 4    # half_life >= 4 bar → STRONG
MODERATE_HALFLIFE = 2    # half_life >= 2 bar → MODERATE
MIN_IC0           = 0.02 # IC_0 harus lolos threshold IC test dulu


def load_data(feature_cols: list, cutoff=None) -> pd.DataFrame:
    if cutoff is None:
        cutoff = TRAIN_CUTOFF_DATE
    pattern = str(Path(LABEL_DIR) / "*_features_v3.parquet")
    files   = sorted(glob.glob(pattern))

    dfs = []
    for fpath in files:
        df = pd.read_parquet(fpath)
        df = df[df.index < cutoff]
        if df.empty:
            continue
        avail = [c for c in feature_cols if c in df.columns]
        df = df[avail + ["label"]].copy()
        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[combined["label"].isin(LABEL_ORDINAL)].copy()
    combined["y"] = combined["label"].map(LABEL_ORDINAL)
    print(f"  Loaded {len(combined):,} rows dari {len(dfs)} koin")
    return combined


def lag_ic(x: pd.Series, y: pd.Series, k: int) -> float:
    """IC(feature shifted k bars, label)."""
    x_lag = x.shift(k)
    mask  = ~(x_lag.isna() | y.isna())
    if mask.sum() < 100:
        return float("nan")
    corr, _ = stats.spearmanr(x_lag[mask], y[mask])
    return float(corr) if not np.isnan(corr) else float("nan")


def compute_halflife(ic_by_lag: dict, ic0: float) -> float:
    """
    Lag pertama dimana |IC_k| <= 0.5 * |IC_0|.
    Interpolasi linear antara dua lag terdekat.
    Return inf jika tidak pernah turun setengah.
    """
    target = 0.5 * abs(ic0)
    lags   = sorted(ic_by_lag.keys())
    prev_k, prev_ic = 0, abs(ic0)

    for k in lags:
        if k == 0:
            continue
        ic_k = ic_by_lag[k]
        if ic_k is None or np.isnan(ic_k):
            continue
        abs_ic = abs(ic_k)
        if abs_ic <= target:
            # interpolasi linear
            if prev_ic == abs_ic:
                return float(k)
            frac = (prev_ic - target) / (prev_ic - abs_ic)
            return round(prev_k + frac * (k - prev_k), 2)
        prev_k, prev_ic = k, abs_ic

    return float("inf")  # tidak pernah turun setengah dalam range lags


def compute_temporal_ic(
    df: pd.DataFrame,
    feature_cols: list,
    lags: list,
) -> list:
    y = df["y"]
    results = []

    for feat in feature_cols:
        if feat not in df.columns:
            continue
        x = df[feat]

        # IC per lag
        ic_by_lag = {}
        for k in lags:
            ic_by_lag[k] = lag_ic(x, y, k)

        ic0 = ic_by_lag.get(0, float("nan"))
        if ic0 is None or np.isnan(ic0):
            results.append({
                "feature":    feat,
                "ic_0":       None,
                "half_life":  None,
                "cum_ic_lag": None,
                "sign_flip_lag": None,
                "verdict":    "NO_IC0",
                "ic_by_lag":  {str(k): round(v, 4) if v is not None and not np.isnan(v) else None
                               for k, v in ic_by_lag.items()},
            })
            continue

        # Half-life
        hl = compute_halflife(ic_by_lag, ic0)

        # Cumulative |IC| untuk lag > 0 (total temporal signal)
        cum = sum(abs(v) for k, v in ic_by_lag.items()
                  if k > 0 and v is not None and not np.isnan(v))

        # Sign flip: lag pertama dimana sign berbeda dengan IC_0 signifikan (|IC| > 0.01)
        sign0 = np.sign(ic0)
        sign_flip_lag = None
        for k in sorted(lags):
            if k == 0:
                continue
            v = ic_by_lag.get(k)
            if v is not None and not np.isnan(v) and abs(v) > 0.01:
                if np.sign(v) != sign0:
                    sign_flip_lag = k
                    break

        # Verdict
        if abs(ic0) < MIN_IC0:
            verdict = "WEAK"    # tidak lolos IC test, tidak cocok untuk LSTM
        elif hl >= STRONG_HALFLIFE:
            verdict = "STRONG"
        elif hl >= MODERATE_HALFLIFE:
            verdict = "MODERATE"
        else:
            verdict = "WEAK"

        results.append({
            "feature":      feat,
            "ic_0":         round(ic0, 4),
            "half_life":    round(hl, 2) if hl != float("inf") else 99.0,
            "cum_ic_lag":   round(cum, 4),
            "sign_flip_lag": sign_flip_lag,
            "verdict":      verdict,
            "ic_by_lag":    {str(k): round(v, 4) if v is not None and not np.isnan(v) else None
                             for k, v in ic_by_lag.items()},
        })

    # sort: STRONG dulu, lalu by half_life desc
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "NO_IC0": 3}
    results.sort(key=lambda x: (
        order.get(x["verdict"], 9),
        -(x["half_life"] or 0),
    ))
    return results


def print_table(results: list, lags: list):
    lag_hdrs = " ".join(f"lag{k:02d}" for k in lags if k > 0)
    header = f"{'Feature':<35} {'IC_0':>7} {'HalfL':>6} {'CumIC':>6}  {'Flip':>5}  {lag_hdrs}  Verdict"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    cur_verdict = None
    for r in results:
        if r["verdict"] != cur_verdict:
            if cur_verdict is not None:
                print()
            cur_verdict = r["verdict"]

        flag = "+" if r["verdict"] == "STRONG" else ("~" if r["verdict"] == "MODERATE" else "-")
        ic0  = f"{r['ic_0']:>+7.4f}" if r["ic_0"] is not None else "    N/A"
        hl   = f"{r['half_life']:>6.2f}" if r["half_life"] is not None else "   N/A"
        cum  = f"{r['cum_ic_lag']:>6.4f}" if r["cum_ic_lag"] is not None else "   N/A"
        flip = f"{r['sign_flip_lag']:>5}" if r["sign_flip_lag"] is not None else "    -"

        lag_vals = " ".join(
            f"{r['ic_by_lag'].get(str(k), None):>+6.3f}"
            if r["ic_by_lag"].get(str(k)) is not None else "   N/A"
            for k in lags if k > 0
        )
        print(f"{flag} {r['feature']:<33} {ic0} {hl} {cum}  {flip}  {lag_vals}  {r['verdict']}")
    print(sep)


def save_results(output: dict, report_dir: str = "reports/experiments"):
    os.makedirs(report_dir, exist_ok=True)
    run_id = output["run_id"]

    json_path = os.path.join(report_dir, f"{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    md_path = os.path.join(report_dir, f"{run_id}.md")
    lags = output["lags"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Temporal IC Decay Test — {run_id}\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | lags={lags}*\n\n")
        f.write(f"**Thresholds**: STRONG half-life>={STRONG_HALFLIFE}bar, "
                f"MODERATE half-life>={MODERATE_HALFLIFE}bar, IC_0>={MIN_IC0}\n\n")

        s = output["summary"]
        f.write("## Summary\n\n")
        f.write("| Verdict | Count | Artinya |\n|---------|-------|--------|\n")
        descs = {
            "STRONG":   "Temporal signal kuat — sertakan di LSTM sequence",
            "MODERATE": "Temporal signal moderat — opsional",
            "WEAK":     "Signal hanya di snapshot — skip dari LSTM",
            "NO_IC0":   "Tidak ada IC_0 — sudah di-drop di IC test",
        }
        for k in ["STRONG", "MODERATE", "WEAK", "NO_IC0"]:
            f.write(f"| **{k}** | {s.get(k,0)} | {descs[k]} |\n")
        f.write("\n")

        f.write(f"## Recommended LSTM Features ({len(output['lstm_features'])})\n\n")
        if output["lstm_features"]:
            f.write(", ".join(f"`{ft}`" for ft in output["lstm_features"]) + "\n\n")
        else:
            f.write("*(tidak ada)*\n\n")

        lag_cols = " | ".join(f"lag{k}" for k in lags if k > 0)
        for group in ["STRONG", "MODERATE", "WEAK"]:
            group_r = [r for r in output["results"] if r["verdict"] == group]
            if not group_r:
                continue
            f.write(f"## {group} ({len(group_r)})\n\n")
            f.write(f"| Feature | IC_0 | Half-life | Cum IC lag | Sign Flip | {lag_cols} | Verdict |\n")
            f.write("|---------|-----:|----------:|-----------:|----------:|"
                    + "------:|" * len([k for k in lags if k > 0]) + "--------|\n")
            for r in group_r:
                lag_vals = " | ".join(
                    f"{r['ic_by_lag'].get(str(k), None):+.3f}"
                    if r["ic_by_lag"].get(str(k)) is not None else "N/A"
                    for k in lags if k > 0
                )
                flip = str(r["sign_flip_lag"]) if r["sign_flip_lag"] else "-"
                f.write(
                    f"| `{r['feature']}` | {r['ic_0']:+.4f} | "
                    f"{r['half_life']:.2f} | {r['cum_ic_lag']:.4f} | {flip} | "
                    f"{lag_vals} | **{r['verdict']}** |\n"
                )
            f.write("\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Temporal IC Decay Test untuk seleksi fitur LSTM")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--feature-cols", default="models/feature_cols_ic32.json")
    parser.add_argument("--lags", nargs="+", type=int, default=DEFAULT_LAGS)
    parser.add_argument("--report-dir", default="reports/experiments")
    parser.add_argument("--include-moderate", action="store_true",
                        help="Sertakan MODERATE dalam rekomendasi LSTM (default: STRONG only)")
    args = parser.parse_args()

    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"temporal_ic_{ts}"

    with open(args.feature_cols, encoding="utf-8") as f:
        feature_cols = json.load(f)
    print(f"[{args.feature_cols}] {len(feature_cols)} fitur dimuat")
    print(f"Lags: {args.lags}")

    print(f"\n{'='*65}")
    print(f" TEMPORAL IC DECAY TEST | run_id={args.run_id}")
    print(f"{'='*65}\n")

    print("[1/3] Loading data...")
    df = load_data(feature_cols)
    n  = len(df)
    print(f"  Rows: {n:,} | Effective N per lag: ~{n // AUTOCORR_FACTOR:,}")

    print(f"\n[2/3] Computing IC at {len(args.lags)} lags per feature...")
    results = compute_temporal_ic(df, feature_cols, args.lags)

    print("[3/3] Menyusun hasil...\n")
    print_table(results, args.lags)

    verdicts = [r["verdict"] for r in results]
    summary  = {v: verdicts.count(v) for v in ["STRONG", "MODERATE", "WEAK", "NO_IC0"]}

    include_verdicts = ["STRONG"] + (["MODERATE"] if args.include_moderate else [])
    lstm_features = [r["feature"] for r in results if r["verdict"] in include_verdicts]

    print(f"\nSummary: STRONG={summary['STRONG']} | MODERATE={summary['MODERATE']} | WEAK={summary['WEAK']}")
    print(f"\nRekomendasi LSTM ({len(lstm_features)} fitur — {'STRONG+MODERATE' if args.include_moderate else 'STRONG only'}):")
    for r in results:
        if r["verdict"] in include_verdicts:
            flip = f" [flip@lag{r['sign_flip_lag']}]" if r["sign_flip_lag"] else ""
            print(f"  half_life={r['half_life']:5.2f}  IC_0={r['ic_0']:+.4f}  {r['feature']}{flip}")

    output = {
        "run_id":       args.run_id,
        "feature_cols": args.feature_cols,
        "lags":         args.lags,
        "n_rows":       n,
        "thresholds":   {
            "strong_halflife":   STRONG_HALFLIFE,
            "moderate_halflife": MODERATE_HALFLIFE,
            "min_ic0":           MIN_IC0,
        },
        "summary":       summary,
        "lstm_features": lstm_features,
        "results":       results,
    }

    save_results(output, report_dir=args.report_dir)

    # simpan juga LSTM feature list ke JSON
    lstm_feat_path = "models/feature_cols_lstm_temporal.json"
    with open(lstm_feat_path, "w", encoding="utf-8") as f:
        json.dump(lstm_features, f, indent=2)
    print(f"LSTM feature list saved: {lstm_feat_path} ({len(lstm_features)} fitur)")


if __name__ == "__main__":
    main()
