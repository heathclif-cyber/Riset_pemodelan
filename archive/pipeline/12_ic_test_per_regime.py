"""
pipeline/12_ic_test_per_regime.py — Regime-Aware Information Coefficient (IC) Test
Simon Methodology: ukur kekuatan sinyal khusus untuk model LGBM_TRENDING.

Langkah-langkah:
1. Load data training (cutoff 2026-04-01).
2. Generate continuation labels (Triple Barrier ATR-based) on-the-fly.
3. Pisahkan data ke dalam regime TRENDING_UP (HMM=3) dan TRENDING_DOWN (HMM=0).
4. Hitung Standalone IC, t-stat, dan Marginal IC (Gram-Schmidt) untuk masing-masing regime.
5. Simpan laporan di reports/experiments/.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAIN_CUTOFF_DATE, LABEL_DIR, TRAINING_COINS
from core.features import triple_barrier_labeling
from core.utils import setup_logger

logger = setup_logger("12_ic_test_per_regime")

LABEL_ORDINAL = {"SHORT": -1, "FLAT": 0, "LONG": 1}
AUTOCORR_FACTOR = 24  # H1 data correlation correction

META_COLS = {
    "label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
    "hmm_regime", "hmm_regime_enc", "open", "high", "low", "close", "volume"
}

# ──────────────────────────────────────────────
#  Data Loading & Relabeling
# ──────────────────────────────────────────────

def load_and_relabel_data(feature_cols: list, cutoff=None) -> pd.DataFrame:
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

        coin = Path(fpath).stem.replace("_features_v3", "")
        
        # Load HMM regime labels
        regime_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if regime_path.exists():
            reg = pd.read_parquet(regime_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            if "hmm_regime" in df.columns:
                df = df.drop(columns=["hmm_regime"])
            df = df.join(reg[["hmm_regime_enc", "hmm_regime"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        else:
            logger.warning(f"[{coin}] Regime HMM tidak ditemukan, skip.")
            continue

        # Generate Triple Barrier continuation labels on-the-fly
        # default TP=2.0 * ATR, SL=1.5 * ATR, max_hold=36 bar H1
        tb_labels = triple_barrier_labeling(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr_base=df["atr_14_h1"],
            tp_atr_mult=2.0,
            sl_atr_mult=1.5,
            max_hold=36
        )
        df["trend_label"] = tb_labels
        
        # Ambil hanya fitur yang tersedia
        avail = [c for c in feature_cols if c in df.columns]
        cols_to_keep = avail + ["trend_label", "hmm_regime_enc", "hmm_regime"]
        df = df[cols_to_keep].copy()
        df["coin"] = coin
        dfs.append(df)
        logger.info(f"Loaded {coin}: {len(df):,} rows")

    if not dfs:
        raise RuntimeError("Semua file kosong atau tidak memiliki data HMM.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total Combined Data: {len(combined):,} rows")
    return combined

# ──────────────────────────────────────────────
#  IC Helpers
# ──────────────────────────────────────────────

def standalone_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 100:
        return 0.0
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else 0.0

def tstat(ic: float, n: int) -> float:
    n_eff = max(n // AUTOCORR_FACTOR, 10)
    denom = np.sqrt(max(1.0 - ic ** 2, 1e-10))
    return ic * np.sqrt(n_eff) / denom

def _rank_norm(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x).astype(np.float64)
    r -= r.mean()
    std = r.std()
    return r / std if std > 1e-10 else np.zeros_like(r)

def _project_out(vec: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    norm_sq = np.dot(pivot, pivot)
    if norm_sq < 1e-10:
        return vec.copy()
    return vec - (np.dot(vec, pivot) / norm_sq) * pivot

def gram_schmidt_marginal_ic(X_raw: np.ndarray, y_raw: np.ndarray, feature_names: list) -> dict:
    n, p = X_raw.shape
    X = X_raw.copy().astype(np.float64)
    for j in range(p):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col

    X_r = np.column_stack([_rank_norm(X[:, j]) for j in range(p)])
    y_r = _rank_norm(y_raw.astype(np.float64))

    remaining = list(range(p))
    marginal = {}

    for _ in range(p):
        if not remaining:
            break

        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]
            nx = np.sqrt(np.dot(xj, xj))
            ny = np.sqrt(np.dot(y_r, y_r))
            if nx < 1e-10 or ny < 1e-10:
                corrs[k] = 0.0
            else:
                corrs[k] = np.dot(xj, y_r) / (nx * ny)

        best_k = int(np.argmax(np.abs(corrs)))
        best_j = remaining[best_k]
        marginal[feature_names[best_j]] = float(corrs[best_k])

        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j:
                X_r[:, j] = _project_out(X_r[:, j], pivot)
        y_r = _project_out(y_r, pivot)

        remaining.remove(best_j)

    return marginal

def make_verdict(ic: float, ts: float, mg: float, min_sa: float, min_t: float, min_mg: float) -> str:
    sa_pass = abs(ic) >= min_sa and abs(ts) >= min_t
    mg_pass = abs(mg) >= min_mg
    if sa_pass and mg_pass:
        return "KEEP"
    elif sa_pass and not mg_pass:
        return "REDUNDANT"
    elif not sa_pass and mg_pass:
        return "WEAK"
    else:
        return "DROP"

# ──────────────────────────────────────────────
#  Regime-Specific Runner
# ──────────────────────────────────────────────

def run_regime_ic_test(df: pd.DataFrame, regime_enc: int, regime_name: str, feature_cols: list, args) -> dict:
    # Filter data berdasarkan regime
    df_reg = df[df["hmm_regime_enc"] == regime_enc].copy()
    df_reg = df_reg[df_reg["trend_label"].isin(LABEL_ORDINAL)].copy()
    df_reg["label_ord"] = df_reg["trend_label"].map(LABEL_ORDINAL)
    
    n_total = len(df_reg)
    n_eff = n_total // AUTOCORR_FACTOR
    
    if n_total < 100:
        logger.warning(f"Data tidak cukup untuk regime {regime_name}: {n_total} baris.")
        return {}

    avail = [f for f in feature_cols if f in df_reg.columns]
    target = df_reg["label_ord"].values
    
    logger.info(f"Running IC for regime {regime_name} — Samples: {n_total:,} | Features: {len(avail)}")
    
    # Standalone IC
    standalone = {}
    tstats_dict = {}
    for feat in avail:
        ic = standalone_ic(df_reg[feat].values, target)
        standalone[feat] = ic
        tstats_dict[feat] = tstat(ic, n_total)

    # Marginal IC (Gram-Schmidt)
    X_mat = df_reg[avail].values
    marginal = gram_schmidt_marginal_ic(X_mat, target, avail)

    results = []
    for feat in avail:
        sa = standalone[feat]
        ts = tstats_dict[feat]
        mg = marginal.get(feat, 0.0)
        v = make_verdict(sa, ts, mg, args.min_standalone, args.min_tstat, args.min_marginal)
        results.append({
            "feature": feat,
            "standalone_ic": round(sa, 4),
            "tstat": round(ts, 2),
            "marginal_ic": round(mg, 4),
            "verdict": v,
        })

    # Sort results
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
        "regime_name": regime_name,
        "n_rows": n_total,
        "n_eff": n_eff,
        "summary": summary,
        "keep_features": keep_features,
        "results": results
    }

# ──────────────────────────────────────────────
#  Main Report Generation
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="ic_test_trending_regimes")
    parser.add_argument("--feature-cols", default="models/feature_cols_v2.json")
    parser.add_argument("--min-standalone", type=float, default=0.02)
    parser.add_argument("--min-tstat", type=float, default=2.0)
    parser.add_argument("--min-marginal", type=float, default=0.01)
    parser.add_argument("--report-dir", default="reports/experiments")
    args = parser.parse_args()

    # Load feature list
    with open(args.feature_cols, encoding="utf-8") as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c not in META_COLS]

    logger.info(f"Loaded {len(feature_cols)} features for IC test.")

    # Load data once
    df = load_and_relabel_data(feature_cols, cutoff=TRAIN_CUTOFF_DATE)

    # Run for TRENDING_UP (HMM state 3)
    up_res = run_regime_ic_test(df, 3, "TRENDING_UP", feature_cols, args)
    
    # Run for TRENDING_DOWN (HMM state 0)
    down_res = run_regime_ic_test(df, 0, "TRENDING_DOWN", feature_cols, args)

    # Save Markdown Report
    os.makedirs(args.report_dir, exist_ok=True)
    md_path = Path(args.report_dir) / f"{args.run_id}.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# IC Test Per-Regime — {args.run_id}\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | cutoff={TRAIN_CUTOFF_DATE.date()}*\n\n")
        f.write(f"Laporan ini menguji korelasi fitur terhadap **Triple Barrier Continuation Label** (TP=2.0, SL=1.5) khusus untuk regime pasar trending.\n\n")
        f.write(f"**Thresholds**: Standalone IC >= {args.min_standalone} | t-stat >= {args.min_tstat} | Marginal IC >= {args.min_marginal}\n\n")

        for r in [up_res, down_res]:
            if not r: continue
            f.write(f"## Regime: {r['regime_name']}\n\n")
            f.write(f"- **Total Rows**: {r['n_rows']:,} | **Effective N**: {r['n_eff']:,}\n")
            s = r["summary"]
            f.write(f"- **Summary**: **KEEP**: {s['KEEP']} | **REDUNDANT**: {s['REDUNDANT']} | **WEAK**: {s['WEAK']} | **DROP**: {s['DROP']}\n\n")
            
            f.write("### KEEP Features\n\n")
            if r["keep_features"]:
                f.write(", ".join(f"`{feat}`" for feat in r["keep_features"]) + "\n\n")
            else:
                f.write("*(tidak ada)*\n\n")

            f.write("| Feature | Standalone IC | t-stat | Marginal IC | Verdict |\n")
            f.write("|---------|:------------:|:------:|:-----------:|:-------:|\n")
            for res in r["results"]:
                # Print only KEEP and REDUNDANT to keep the report neat, unless empty
                if res["verdict"] in ["KEEP", "REDUNDANT", "WEAK"]:
                    f.write(f"| `{res['feature']}` | {res['standalone_ic']:+.4f} | {res['tstat']:.2f} | {res['marginal_ic']:+.4f} | **{res['verdict']}** |\n")
            f.write("\n---\n\n")

    print(f"\n{'='*65}")
    print(f" IC TEST PER REGIME COMPLETE")
    print(f" Saved report -> {md_path}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
