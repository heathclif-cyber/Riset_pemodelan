"""
pipeline/03c_ic_decay_test.py — IC Decay Test (Simon Methodology Step 2)
Validasi: apakah sinyal stabil lintas waktu?

Membagi data training ke 6 window temporal dan menghitung IC per window.
Fitur dengan IC tidak konsisten (sign flip / IC_IR rendah) → DROP.

IC Information Ratio (IC_IR) = mean(IC) / std(IC)
  IC_IR >= 0.5  → sinyal stabil, layak masuk model
  IC_IR <  0.5  → sinyal tidak stabil, regime-specific

Usage:
    python pipeline/03c_ic_decay_test.py
    python pipeline/03c_ic_decay_test.py --features-from reports/experiments/ic_test_lgbm_allfeats_v1.json
    python pipeline/03c_ic_decay_test.py --run-id ic_decay_v1 --min-ir 0.5
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

LABEL_ORDINAL = {"SHORT": -1, "FLAT": 0, "LONG": 1}
AUTOCORR_FACTOR = 24

# 6 window kalender — tiap window ~1 tahun
WINDOWS = [
    ("2020", "2020-01-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2025-11-01"),
]

MIN_IR = 0.5        # IC_IR threshold — sinyal stabil
MIN_SIGN_CONS = 5   # minimal window dengan sign konsisten (dari 6)


def load_window(feature_cols: list, start: str, end: str) -> pd.DataFrame:
    t_start = pd.Timestamp(start, tz="UTC")
    t_end   = pd.Timestamp(end,   tz="UTC")
    pattern = str(Path(LABEL_DIR) / "*_features_v3.parquet")
    files   = sorted(glob.glob(pattern))

    dfs = []
    for fpath in files:
        df = pd.read_parquet(fpath)
        df = df[(df.index >= t_start) & (df.index < t_end)]
        if df.empty:
            continue
        avail = [c for c in feature_cols if c in df.columns]
        df = df[avail + ["label"]].copy()
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 50:
        return float("nan")
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else float("nan")


def ic_tstat(ic: float, n: int) -> float:
    n_eff = max(n // AUTOCORR_FACTOR, 5)
    denom = np.sqrt(max(1.0 - ic ** 2, 1e-10))
    return ic * np.sqrt(n_eff) / denom


def run_decay_test(
    feature_cols: list,
    run_id: str,
    min_ir: float = MIN_IR,
    min_sign_cons: int = MIN_SIGN_CONS,
) -> dict:
    print(f"\n{'='*65}")
    print(f" IC DECAY TEST | run_id={run_id}")
    print(f" Windows: {len(WINDOWS)} | IC_IR threshold: {min_ir}")
    print(f"{'='*65}\n")

    # kumpulkan IC per window per fitur
    window_data = {}   # {window_label: {feat: ic}}
    window_sizes = {}  # {window_label: n_rows}

    for label, start, end in WINDOWS:
        print(f"  Window {label} ({start} - {end})...", end=" ", flush=True)
        df = load_window(feature_cols, start, end)
        if df.empty:
            print("SKIP (no data)")
            continue

        df = df[df["label"].isin(LABEL_ORDINAL)].copy()
        df["label_ord"] = df["label"].map(LABEL_ORDINAL)
        target = df["label_ord"].values
        n = len(df)
        window_sizes[label] = n

        ics = {}
        for feat in feature_cols:
            if feat in df.columns:
                ics[feat] = compute_ic(df[feat].values, target)
            else:
                ics[feat] = float("nan")

        window_data[label] = ics
        label_dist = dict(df["label"].value_counts())
        print(f"{n:,} rows | {label_dist}")

    windows_used = list(window_data.keys())
    n_windows = len(windows_used)

    # hitung statistik per fitur
    results = []
    for feat in feature_cols:
        ics_per_window = [window_data[w].get(feat, float("nan")) for w in windows_used]
        valid_ics = [ic for ic in ics_per_window if not np.isnan(ic)]

        if len(valid_ics) < 3:
            verdict = "INSUFFICIENT_DATA"
            ic_mean = float("nan")
            ic_std  = float("nan")
            ic_ir   = float("nan")
            sign_cons = 0
        else:
            ic_mean   = float(np.mean(valid_ics))
            ic_std    = float(np.std(valid_ics, ddof=1))
            ic_ir     = abs(ic_mean) / ic_std if ic_std > 1e-10 else float("inf")
            # sign konsistensi: berapa window punya sign yang sama dengan mean
            dominant_sign = np.sign(ic_mean)
            sign_cons = sum(1 for ic in valid_ics if np.sign(ic) == dominant_sign)

            if ic_ir >= min_ir and sign_cons >= min_sign_cons:
                verdict = "STABLE"
            elif sign_cons >= min_sign_cons and ic_ir >= min_ir * 0.6:
                verdict = "MARGINAL"
            else:
                verdict = "UNSTABLE"

        results.append({
            "feature":       feat,
            "ic_mean":       round(ic_mean, 4) if not np.isnan(ic_mean) else None,
            "ic_std":        round(ic_std,  4) if not np.isnan(ic_std)  else None,
            "ic_ir":         round(ic_ir,   3) if not np.isnan(ic_ir) and ic_ir != float("inf") else None,
            "sign_cons":     f"{sign_cons}/{len(valid_ics)}",
            "verdict":       verdict,
            "ic_per_window": {w: round(ic, 4) if not np.isnan(ic) else None
                              for w, ic in zip(windows_used, ics_per_window)},
        })

    # sort: STABLE dulu, lalu by |ic_mean| desc
    order = {"STABLE": 0, "MARGINAL": 1, "UNSTABLE": 2, "INSUFFICIENT_DATA": 3}
    results.sort(key=lambda x: (
        order.get(x["verdict"], 9),
        -abs(x["ic_mean"]) if x["ic_mean"] is not None else 0
    ))

    summary = {v: sum(1 for r in results if r["verdict"] == v)
               for v in ["STABLE", "MARGINAL", "UNSTABLE", "INSUFFICIENT_DATA"]}
    stable_features = [r["feature"] for r in results if r["verdict"] == "STABLE"]

    return {
        "run_id":          run_id,
        "n_windows":       n_windows,
        "windows":         [w[0] for w in WINDOWS],
        "window_sizes":    window_sizes,
        "thresholds":      {"min_ir": min_ir, "min_sign_cons": min_sign_cons},
        "n_features":      len(feature_cols),
        "summary":         summary,
        "stable_features": stable_features,
        "results":         results,
    }


def print_table(results: list, windows_used: list):
    win_hdrs = "  ".join(f"{w:>7}" for w in windows_used)
    header = f"{'Feature':<35} {'Mean':>7} {'Std':>6} {'IR':>5} {'Sign':>6}  {win_hdrs}  Verdict"
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

        flag = "+" if r["verdict"] == "STABLE" else ("~" if r["verdict"] == "MARGINAL" else "-")
        ic_mean = f"{r['ic_mean']:>+7.4f}" if r["ic_mean"] is not None else "    N/A"
        ic_std  = f"{r['ic_std']:>6.4f}"   if r["ic_std"]  is not None else "   N/A"
        ic_ir   = f"{r['ic_ir']:>5.2f}"    if r["ic_ir"]   is not None else "  N/A"

        win_vals = "  ".join(
            f"{r['ic_per_window'].get(w, None):>+7.4f}"
            if r["ic_per_window"].get(w) is not None else "    N/A"
            for w in windows_used
        )
        print(f"{flag} {r['feature']:<33} {ic_mean} {ic_std} {ic_ir} {r['sign_cons']:>6}  {win_vals}  {r['verdict']}")
    print(sep)


def save_results(output: dict, report_dir: str = "reports/experiments"):
    os.makedirs(report_dir, exist_ok=True)
    run_id = output["run_id"]
    windows = output["windows"]

    json_path = os.path.join(report_dir, f"{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    md_path = os.path.join(report_dir, f"{run_id}.md")
    s = output["summary"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# IC Decay Test — {run_id}\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | {output['n_windows']} windows*\n\n")
        thresh = output["thresholds"]
        f.write(f"**Thresholds**: IC_IR >= {thresh['min_ir']}, sign konsisten >= {thresh['min_sign_cons']}/{output['n_windows']} window\n\n")

        f.write("## Summary\n\n")
        f.write("| Verdict | Count | Artinya |\n|---------|-------|--------|\n")
        descs = {
            "STABLE":            "Sinyal konsisten lintas semua regime — masuk model",
            "MARGINAL":          "Sinyal cukup konsisten — masuk dengan catatan",
            "UNSTABLE":          "Sinyal tidak konsisten — regime-specific, hindari",
            "INSUFFICIENT_DATA": "Data tidak cukup di beberapa window",
        }
        for k in ["STABLE", "MARGINAL", "UNSTABLE", "INSUFFICIENT_DATA"]:
            f.write(f"| **{k}** | {s.get(k,0)} | {descs[k]} |\n")
        f.write("\n")

        f.write(f"## STABLE Features ({s.get('STABLE',0)})\n\n")
        if output["stable_features"]:
            f.write(", ".join(f"`{feat}`" for feat in output["stable_features"]) + "\n\n")
        else:
            f.write("*(tidak ada)*\n\n")

        win_cols = " | ".join(f"IC {w}" for w in windows)
        for group in ["STABLE", "MARGINAL", "UNSTABLE"]:
            group_r = [r for r in output["results"] if r["verdict"] == group]
            if not group_r:
                continue
            f.write(f"## {group} ({len(group_r)})\n\n")
            f.write(f"| Feature | Mean IC | Std | IC_IR | Sign | {win_cols} | Verdict |\n")
            f.write("|---------|--------:|----:|------:|-----:|" + "------:|" * len(windows) + "--------|\n")
            for r in group_r:
                win_vals = " | ".join(
                    f"{r['ic_per_window'].get(w):+.4f}" if r["ic_per_window"].get(w) is not None else "N/A"
                    for w in windows
                )
                f.write(
                    f"| `{r['feature']}` | "
                    f"{r['ic_mean']:+.4f} | {r['ic_std']:.4f} | "
                    f"{r['ic_ir']:.2f} | {r['sign_cons']} | "
                    f"{win_vals} | **{r['verdict']}** |\n"
                )
            f.write("\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="IC Decay Test — validasi stabilitas sinyal lintas waktu")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--features-from",
        default="reports/experiments/ic_test_lgbm_allfeats_v1.json",
        help="Ambil KEEP features dari hasil IC test JSON (default: allfeats_v1)",
    )
    parser.add_argument("--verdict-filter", default="KEEP",
                        help="Verdict fitur yang diambil: KEEP, KEEP,REDUNDANT, dll (default: KEEP)")
    parser.add_argument("--min-ir", type=float, default=MIN_IR)
    parser.add_argument("--min-sign-cons", type=int, default=MIN_SIGN_CONS)
    parser.add_argument("--report-dir", default="reports/experiments")
    args = parser.parse_args()

    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"ic_decay_{ts}"

    # load fitur dari IC test JSON
    with open(args.features_from, encoding="utf-8") as f:
        ic_data = json.load(f)

    verdicts_wanted = [v.strip() for v in args.verdict_filter.split(",")]
    feature_cols = [r["feature"] for r in ic_data["results"] if r["verdict"] in verdicts_wanted]
    print(f"Loaded {len(feature_cols)} fitur ({', '.join(verdicts_wanted)}) dari {args.features_from}")

    output = run_decay_test(
        feature_cols=feature_cols,
        run_id=args.run_id,
        min_ir=args.min_ir,
        min_sign_cons=args.min_sign_cons,
    )

    windows_used = output["windows"]
    print()
    print_table(output["results"], windows_used)

    s = output["summary"]
    print(f"\nSummary: STABLE={s.get('STABLE',0)} | MARGINAL={s.get('MARGINAL',0)} | UNSTABLE={s.get('UNSTABLE',0)}")

    if output["stable_features"]:
        print(f"\nSTABLE features ({len(output['stable_features'])}):")
        for feat in output["stable_features"]:
            r = next(x for x in output["results"] if x["feature"] == feat)
            print(f"  IR={r['ic_ir']:.2f}  sign={r['sign_cons']}  {feat}")

    save_results(output, report_dir=args.report_dir)


if __name__ == "__main__":
    main()
