"""
pipeline/03b_ic_test.py — Information Coefficient (IC) Test
Simon Methodology: ukur sinyal sebelum training.

Standalone IC: seberapa kuat fitur memprediksi label secara mandiri.
Marginal IC  : kontribusi unik fitur setelah fitur lain sudah diketahui
               (Gram-Schmidt sequential orthogonalization).

Usage:
    python pipeline/03b_ic_test.py
    python pipeline/03b_ic_test.py --run-id ic_lgbm_v1
    python pipeline/03b_ic_test.py --feature-cols models/feature_cols_v2.json
    python pipeline/03b_ic_test.py --min-standalone 0.02 --min-tstat 2.0 --min-marginal 0.01
    python pipeline/03b_ic_test.py --all-features
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
from config import TRAIN_CUTOFF_DATE, LABEL_DIR, TRAINING_COINS

# --- Encoding label ke ordinal untuk Spearman ---
LABEL_ORDINAL = {"SHORT": -1, "FLAT": 0, "LONG": 1}

# --- Koreksi autocorrelation: data H1, effective N = N / 24 ---
AUTOCORR_FACTOR = 24

# --- Mode ---
# "ordinal"     : target = SHORT=-1, FLAT=0, LONG=+1 (default)
# "flat-binary" : target = FLAT=1, NON-FLAT=0 (untuk mendeteksi fitur FLAT-spesifik)
VALID_MODES = {"ordinal", "flat-binary"}

# --- Kolom metadata yang dikecualikan dari IC test ---
# OHLCV tidak dikecualikan — bisa saja ada di feature_cols_v2.json
META_COLS = {
    "label", "coin", "symbol",
    "h4_swing_high", "h4_swing_low",
    "hmm_regime", "hmm_regime_enc",
}

# --- Default threshold (Simon methodology) ---
MIN_STANDALONE_IC = 0.02
MIN_TSTAT = 2.0
MIN_MARGINAL_IC = 0.01


# ──────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────

def load_training_data(feature_cols: list, cutoff=None) -> pd.DataFrame:
    pattern = str(Path(LABEL_DIR) / "*_features_v3.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Tidak ada file parquet di {LABEL_DIR}")

    dfs = []
    for fpath in files:
        df = pd.read_parquet(fpath)
        if cutoff is not None:
            df = df[df.index < cutoff]
        if df.empty:
            continue
        # ambil hanya fitur yang ada + label
        avail = [c for c in feature_cols if c in df.columns]
        df = df[avail + ["label"]].copy()
        coin = Path(fpath).stem.replace("_features_v3", "")
        df["coin"] = coin
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Semua file kosong setelah filter cutoff.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined):,} rows dari {len(dfs)} koin")
    return combined


# ──────────────────────────────────────────────
#  IC Computation
# ──────────────────────────────────────────────

def standalone_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 100:
        return 0.0
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def tstat(ic: float, n: int) -> float:
    """t-stat dengan koreksi autocorrelation (effective N = N / AUTOCORR_FACTOR)."""
    n_eff = max(n // AUTOCORR_FACTOR, 10)
    denom = np.sqrt(max(1.0 - ic ** 2, 1e-10))
    return ic * np.sqrt(n_eff) / denom


def _rank_norm(x: np.ndarray) -> np.ndarray:
    """Rank-transform + zero-mean + unit-std. NaN sudah dihapus sebelumnya."""
    r = stats.rankdata(x).astype(np.float64)
    r -= r.mean()
    std = r.std()
    return r / std if std > 1e-10 else np.zeros_like(r)


def _project_out(vec: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Hapus komponen vec yang paralel dengan pivot."""
    norm_sq = np.dot(pivot, pivot)
    if norm_sq < 1e-10:
        return vec.copy()
    return vec - (np.dot(vec, pivot) / norm_sq) * pivot


def gram_schmidt_marginal_ic(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Sequential Gram-Schmidt orthogonalization untuk Marginal IC.

    Setiap iterasi:
      1. Cari fitur dengan |corr(x_j_residual, y_residual)| tertinggi.
      2. Catat corr ini sebagai Marginal IC fitur tersebut.
      3. Project-out fitur tersebut dari semua fitur sisa dan target.

    Return: {feature_name: marginal_ic}
    """
    n, p = X_raw.shape

    # isi NaN dengan median kolom sebelum ranking
    X = X_raw.copy().astype(np.float64)
    for j in range(p):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col

    # rank-normalize semua fitur dan target
    X_r = np.column_stack([_rank_norm(X[:, j]) for j in range(p)])
    y_r = _rank_norm(y_raw.astype(np.float64))

    remaining = list(range(p))
    marginal = {}

    for _ in range(p):
        if not remaining:
            break

        # hitung korelasi tiap fitur sisa dengan target residual
        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]
            nx = np.sqrt(np.dot(xj, xj))
            ny = np.sqrt(np.dot(y_r, y_r))
            if nx < 1e-10 or ny < 1e-10:
                corrs[k] = 0.0
            else:
                corrs[k] = np.dot(xj, y_r) / (nx * ny)

        # pilih fitur dengan |corr| tertinggi
        best_k = int(np.argmax(np.abs(corrs)))
        best_j = remaining[best_k]
        marginal[feature_names[best_j]] = float(corrs[best_k])

        # project-out pivot dari semua fitur sisa dan target
        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j:
                X_r[:, j] = _project_out(X_r[:, j], pivot)
        y_r = _project_out(y_r, pivot)

        remaining.remove(best_j)

    return marginal


# ──────────────────────────────────────────────
#  Verdict
# ──────────────────────────────────────────────

def make_verdict(
    ic: float, ts: float, mg: float,
    min_sa: float, min_t: float, min_mg: float,
) -> str:
    sa_pass = abs(ic) >= min_sa and abs(ts) >= min_t
    mg_pass = abs(mg) >= min_mg

    if sa_pass and mg_pass:
        return "KEEP"
    elif sa_pass and not mg_pass:
        return "REDUNDANT"
    elif not sa_pass and mg_pass:
        return "WEAK"          # suppressor variable
    else:
        return "DROP"


# ──────────────────────────────────────────────
#  Main Runner
# ──────────────────────────────────────────────

def run_ic_test(
    feature_cols: list,
    run_id: str,
    cutoff=None,
    min_standalone: float = MIN_STANDALONE_IC,
    min_tstat_val: float = MIN_TSTAT,
    min_marginal: float = MIN_MARGINAL_IC,
    mode: str = "ordinal",
) -> dict:
    if cutoff is None:
        cutoff = TRAIN_CUTOFF_DATE
    if mode not in VALID_MODES:
        raise ValueError(f"mode harus salah satu dari {VALID_MODES}, bukan '{mode}'")

    print(f"\n{'='*65}")
    print(f" IC TEST — LGBM | run_id={run_id}")
    print(f" Mode    : {mode}")
    print(f" Cutoff  : {cutoff.date()}")
    print(f" Threshold: standalone>={min_standalone}, t>={min_tstat_val}, marginal>={min_marginal}")
    print(f"{'='*65}\n")

    # load data
    print("[1/4] Loading training data...")
    df = load_training_data(feature_cols, cutoff=cutoff)

    # filter label valid
    df = df[df["label"].isin(LABEL_ORDINAL)].copy()

    # encode target sesuai mode
    if mode == "flat-binary":
        target = (df["label"] == "FLAT").astype(float).values
        n_flat = int((df["label"] == "FLAT").sum())
    else:
        df["label_ord"] = df["label"].map(LABEL_ORDINAL)
        target = df["label_ord"].values
        n_flat = None

    # align fitur ke yang benar-benar ada
    avail = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} fitur tidak ada di data: {missing}")

    n_total = len(df)
    n_eff = n_total // AUTOCORR_FACTOR

    label_dist = dict(df["label"].value_counts())
    print(f"  Rows: {n_total:,} | Effective N: {n_eff:,}")
    print(f"  Label: {label_dist}")
    if mode == "flat-binary":
        print(f"  Target: FLAT=1 ({n_flat:,}/{n_total:,} = {n_flat/n_total*100:.1f}%), NON-FLAT=0")
    print(f"  Features: {len(avail)}")

    # standalone IC
    print("\n[2/4] Menghitung Standalone IC...")
    standalone = {}
    tstats_dict = {}
    for feat in avail:
        ic = standalone_ic(df[feat].values, target)
        standalone[feat] = ic
        tstats_dict[feat] = tstat(ic, n_total)

    # marginal IC via Gram-Schmidt
    print("[3/4] Menghitung Marginal IC (Gram-Schmidt)...")
    X_mat = df[avail].values
    marginal = gram_schmidt_marginal_ic(X_mat, target, avail)

    # build results
    print("[4/4] Menyusun hasil...")
    results = []
    for feat in avail:
        sa = standalone[feat]
        ts = tstats_dict[feat]
        mg = marginal.get(feat, 0.0)
        v = make_verdict(sa, ts, mg, min_standalone, min_tstat_val, min_marginal)
        results.append({
            "feature": feat,
            "standalone_ic": round(sa, 4),
            "tstat": round(ts, 2),
            "marginal_ic": round(mg, 4),
            "verdict": v,
        })

    # sort: KEEP dulu, lalu by |standalone_ic| desc
    verdict_order = {"KEEP": 0, "REDUNDANT": 1, "WEAK": 2, "DROP": 3}
    results.sort(key=lambda x: (verdict_order[x["verdict"]], -abs(x["standalone_ic"])))

    verdicts = [r["verdict"] for r in results]
    summary = {
        "KEEP": verdicts.count("KEEP"),
        "REDUNDANT": verdicts.count("REDUNDANT"),
        "WEAK": verdicts.count("WEAK"),
        "DROP": verdicts.count("DROP"),
    }
    keep_features = [r["feature"] for r in results if r["verdict"] == "KEEP"]

    return {
        "run_id": run_id,
        "mode": mode,
        "cutoff": str(cutoff.date()),
        "n_rows": n_total,
        "n_eff": n_eff,
        "autocorr_factor": AUTOCORR_FACTOR,
        "n_features_tested": len(avail),
        "thresholds": {
            "min_standalone_ic": min_standalone,
            "min_tstat": min_tstat_val,
            "min_marginal_ic": min_marginal,
        },
        "label_distribution": label_dist,
        "summary": summary,
        "keep_features": keep_features,
        "results": results,
    }


# ──────────────────────────────────────────────
#  Output
# ──────────────────────────────────────────────

def print_results_table(results: list):
    header = f"{'Feature':<38} {'SA_IC':>7} {'t-stat':>7} {'Marg_IC':>9}  Verdict"
    sep = "-" * 75
    print(sep)
    print(header)
    print(sep)

    current_verdict = None
    for r in results:
        if r["verdict"] != current_verdict:
            if current_verdict is not None:
                print()
            current_verdict = r["verdict"]

        flag = "+" if r["verdict"] == "KEEP" else ("~" if r["verdict"] == "REDUNDANT" else "-")
        print(
            f"{flag} {r['feature']:<36} "
            f"{r['standalone_ic']:>+7.4f} "
            f"{r['tstat']:>7.2f} "
            f"{r['marginal_ic']:>+9.4f}  "
            f"{r['verdict']}"
        )
    print(sep)


def save_results(output: dict, report_dir: str = "reports/experiments"):
    os.makedirs(report_dir, exist_ok=True)
    run_id = output["run_id"]

    # JSON
    json_path = os.path.join(report_dir, f"{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # Markdown
    md_path = os.path.join(report_dir, f"{run_id}.md")
    s = output["summary"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# IC Test LGBM — {run_id}\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | cutoff={output['cutoff']}*\n\n")
        f.write(
            f"**Rows**: {output['n_rows']:,} | "
            f"**Effective N**: {output['n_eff']:,} (N/{output['autocorr_factor']}) | "
            f"**Features tested**: {output['n_features_tested']}\n\n"
        )
        thresh = output["thresholds"]
        f.write(
            f"**Thresholds**: standalone>={thresh['min_standalone_ic']}, "
            f"t-stat>={thresh['min_tstat']}, marginal>={thresh['min_marginal_ic']}\n\n"
        )

        f.write("## Summary\n\n")
        f.write("| Verdict | Count | Artinya |\n|---------|-------|--------|\n")
        descs = {
            "KEEP": "Standalone dan marginal IC lolos — masuk model",
            "REDUNDANT": "Standalone lolos tapi marginal kecil — duplikasi sinyal",
            "WEAK": "Standalone gagal tapi marginal lolos — suppressor variable",
            "DROP": "Tidak ada sinyal — buang",
        }
        for k in ["KEEP", "REDUNDANT", "WEAK", "DROP"]:
            f.write(f"| **{k}** | {s[k]} | {descs[k]} |\n")
        f.write("\n")

        f.write(f"## KEEP Features ({s['KEEP']})\n\n")
        if output["keep_features"]:
            f.write(", ".join(f"`{feat}`" for feat in output["keep_features"]) + "\n\n")
        else:
            f.write("*(tidak ada)*\n\n")

        for group in ["KEEP", "REDUNDANT", "WEAK", "DROP"]:
            group_results = [r for r in output["results"] if r["verdict"] == group]
            if not group_results:
                continue
            f.write(f"## {group} ({len(group_results)})\n\n")
            f.write("| Feature | Standalone IC | t-stat | Marginal IC | Verdict |\n")
            f.write("|---------|:------------:|:------:|:-----------:|:-------:|\n")
            for r in group_results:
                f.write(
                    f"| `{r['feature']}` | {r['standalone_ic']:+.4f} | "
                    f"{r['tstat']:.2f} | {r['marginal_ic']:+.4f} | **{r['verdict']}** |\n"
                )
            f.write("\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    return json_path, md_path


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IC Test untuk LGBM features (Simon Methodology)")
    parser.add_argument("--run-id", default=None, help="ID run (default: timestamp)")
    parser.add_argument(
        "--feature-cols",
        default="models/feature_cols_v2.json",
        help="Path ke JSON list fitur (default: models/feature_cols_v2.json)",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Test semua kolom numerik di parquet (bukan hanya feature_cols_v2.json)",
    )
    parser.add_argument("--mode", default="ordinal", choices=list(VALID_MODES),
                        help="'ordinal' (default) atau 'flat-binary' (FLAT=1 vs NON-FLAT=0)")
    parser.add_argument("--min-standalone", type=float, default=None,
                        help="Default: 0.02 (ordinal) atau 0.01 (flat-binary)")
    parser.add_argument("--min-tstat", type=float, default=None,
                        help="Default: 2.0 (ordinal) atau 1.5 (flat-binary)")
    parser.add_argument("--min-marginal", type=float, default=None,
                        help="Default: 0.01 (ordinal) atau 0.005 (flat-binary)")
    parser.add_argument("--report-dir", default="reports/experiments")
    args = parser.parse_args()

    # threshold default berbeda per mode
    if args.mode == "flat-binary":
        min_sa  = args.min_standalone if args.min_standalone is not None else 0.01
        min_ts  = args.min_tstat      if args.min_tstat      is not None else 1.5
        min_mg  = args.min_marginal   if args.min_marginal   is not None else 0.005
    else:
        min_sa  = args.min_standalone if args.min_standalone is not None else MIN_STANDALONE_IC
        min_ts  = args.min_tstat      if args.min_tstat      is not None else MIN_TSTAT
        min_mg  = args.min_marginal   if args.min_marginal   is not None else MIN_MARGINAL_IC

    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"ic_test_lgbm_{ts}"

    # tentukan feature list
    if args.all_features:
        # load satu file contoh, ambil semua kolom numerik minus meta
        sample_file = sorted(glob.glob(str(Path(LABEL_DIR) / "*_features_v3.parquet")))[0]
        sample_df = pd.read_parquet(sample_file, engine="pyarrow")
        feature_cols = [
            c for c in sample_df.select_dtypes(include=[np.number]).columns
            if c not in META_COLS
        ]
        print(f"[--all-features] {len(feature_cols)} kolom numerik ditemukan")
    else:
        feat_path = args.feature_cols
        if not os.path.exists(feat_path):
            print(f"ERROR: file tidak ditemukan: {feat_path}")
            sys.exit(1)
        with open(feat_path, encoding="utf-8") as f:
            feature_cols = json.load(f)
        # hapus ohlcv dan meta jika ada
        feature_cols = [c for c in feature_cols if c not in META_COLS]
        print(f"[{feat_path}] {len(feature_cols)} fitur dimuat")

    output = run_ic_test(
        feature_cols=feature_cols,
        run_id=args.run_id,
        min_standalone=min_sa,
        min_tstat_val=min_ts,
        min_marginal=min_mg,
        mode=args.mode,
    )

    print_results_table(output["results"])

    s = output["summary"]
    print(f"\nSummary: KEEP={s['KEEP']} | REDUNDANT={s['REDUNDANT']} | WEAK={s['WEAK']} | DROP={s['DROP']}")

    if output["keep_features"]:
        print(f"\nKEEP features ({s['KEEP']}):")
        for feat in output["keep_features"]:
            r = next(x for x in output["results"] if x["feature"] == feat)
            print(f"  {r['standalone_ic']:+.4f}  {feat}")

    save_results(output, report_dir=args.report_dir)


if __name__ == "__main__":
    main()
