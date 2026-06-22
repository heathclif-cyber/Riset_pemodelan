"""
pipeline/03c_fs_shap.py — Feature Selection: SHAP + Gain + Correlation
Simon Methodology: decorrelation lebih penting dari kekuatan individual.

Langkah:
  1. Train lightweight LGBM (1 fold, pre-cutoff) -> gain importance
  2. Compute SHAP mean|values| per feature
  3. Correlation clustering -> tandai fitur redundan
  4. Combined ranking: IC (dari 03b) + SHAP + Gain + Corr

Usage:
    python pipeline/03c_fs_shap.py
    python pipeline/03c_fs_shap.py --ic-json reports/experiments/tb_widyawardhana_fs_v3.json
    python pipeline/03c_fs_shap.py --top-n 30
"""
import argparse, glob, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LABEL_DIR, TRAIN_CUTOFF_DATE, REPORT_DIR
from core.features import triple_barrier_labeling
from core.utils import setup_logger

logger = setup_logger("03c_fs_shap")

LABEL_MAP   = {"LONG": 2, "SHORT": 0, "FLAT": 1}
META_COLS   = {"label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
               "hmm_regime", "hmm_regime_enc"}
TP_ATR      = 2.0
SL_ATR      = 1.5
MAX_HOLD    = 36
CORR_THRESH = 0.80   # pasangan dengan |corr| > ini dianggap redundan


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(cutoff=None) -> pd.DataFrame:
    pattern = str(LABEL_DIR / "*_features_v3.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Tidak ada parquet di {LABEL_DIR}")

    dfs = []
    for fpath in files:
        df = pd.read_parquet(fpath)
        if cutoff is not None:
            df = df[df.index < cutoff]
        if df.empty:
            continue
        coin = Path(fpath).stem.replace("_features_v3", "")
        df["coin"] = coin
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} rows dari {len(dfs)} koin")
    return combined


# ── LGBM Training (lightweight) ───────────────────────────────────────────────

def train_lgbm_for_importance(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Train single-fold LGBM, return model + feature importance."""
    avail = [f for f in feature_cols if f in df.columns]
    logger.info(f"Training LGBM dengan {len(avail)} fitur, {len(df):,} rows...")

    X = df[avail].values.astype(np.float32)
    y = df["label"].map(LABEL_MAP).fillna(1).values.astype(int)

    # Isi NaN dengan median
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = float(np.nanmedian(col))
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col

    params = {
        "objective":         "multiclass",
        "num_class":         3,
        "n_estimators":      300,
        "learning_rate":     0.05,
        "max_depth":         5,
        "num_leaves":        31,
        "min_child_samples": 50,
        "subsample":         0.8,
        "colsample_bytree":  0.7,
        "class_weight":      "balanced",
        "verbose":           -1,
        "n_jobs":            -1,
        "random_state":      42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    gain_raw   = model.booster_.feature_importance(importance_type="gain")
    gain_total = gain_raw.sum()
    gain_norm  = gain_raw / (gain_total + 1e-10)

    gain_dict = {avail[j]: float(gain_norm[j]) for j in range(len(avail))}
    logger.info("LGBM training selesai")
    return model, gain_dict, X, avail


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(model: lgb.LGBMClassifier, X: np.ndarray, feature_cols: list,
                 max_samples: int = 20_000) -> dict:
    """Compute mean |SHAP| per feature (rata-rata semua class)."""
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        X_s = X[idx]
    else:
        X_s = X

    logger.info(f"Computing SHAP pada {len(X_s):,} samples...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_s)

    # Handle both SHAP API versions:
    #   Old (<0.40): list of n_classes arrays each [n_samples, n_features]
    #   New (>=0.40): single array [n_samples, n_features, n_classes]
    if isinstance(shap_vals, list):
        abs_mean = np.mean([np.abs(sv) for sv in shap_vals], axis=0)  # [n, p]
    else:
        sv = np.array(shap_vals)
        if sv.ndim == 3:
            abs_mean = np.abs(sv).mean(axis=2)   # [n, p, c] -> [n, p]
        else:
            abs_mean = np.abs(sv)                 # [n, p]
    shap_importance = abs_mean.mean(axis=0)        # [p]
    total = shap_importance.sum() + 1e-10
    shap_dict = {feature_cols[j]: float(shap_importance[j] / total)
                 for j in range(len(feature_cols))}
    logger.info("SHAP selesai")
    return shap_dict


# ── Correlation Clustering ────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, feature_cols: list,
                         corr_thresh: float = CORR_THRESH) -> dict:
    """
    Hitung correlation matrix dan tandai pasangan redundan.
    Return: {feature: list of correlated partners}
    """
    avail  = [f for f in feature_cols if f in df.columns]
    X_df   = df[avail].fillna(df[avail].median())
    corr   = X_df.corr(method="spearman").abs()

    redundant_pairs = []
    seen = set()
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            if corr.iloc[i, j] >= corr_thresh:
                pair = (avail[i], avail[j])
                if pair not in seen:
                    seen.add(pair)
                    redundant_pairs.append({
                        "feat_a": avail[i],
                        "feat_b": avail[j],
                        "spearman": round(float(corr.iloc[i, j]), 4),
                    })

    # Group by feature: siapa pasangannya
    corr_partners: dict = {f: [] for f in avail}
    for p in redundant_pairs:
        corr_partners[p["feat_a"]].append(p["feat_b"])
        corr_partners[p["feat_b"]].append(p["feat_a"])

    logger.info(f"Correlation: {len(redundant_pairs)} pasangan |rho| >= {corr_thresh}")
    return corr_partners, redundant_pairs


# ── Combined Ranking ──────────────────────────────────────────────────────────

def combined_ranking(
    feature_cols: list,
    gain_dict:    dict,
    shap_dict:    dict,
    corr_partners: dict,
    ic_data:      dict | None,
    top_n:        int = 30,
) -> list:
    """
    Score gabungan (0-1 setelah min-max normalisasi):
      score = 0.35 * gain_norm + 0.35 * shap_norm + 0.30 * ic_norm

    Fitur dengan banyak correlated partners diberi penalti redundancy.
    Prinsip Simon: decorrelation mengutamakan diversifikasi sinyal.
    """
    feats = [f for f in feature_cols if f in gain_dict or f in shap_dict]

    gain_arr  = np.array([gain_dict.get(f, 0.0) for f in feats])
    shap_arr  = np.array([shap_dict.get(f, 0.0) for f in feats])

    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    gain_n = _minmax(gain_arr)
    shap_n = _minmax(shap_arr)

    # IC score dari JSON hasil 03b
    if ic_data:
        ic_map = {r["feature"]: abs(r["standalone_ic"]) for r in ic_data.get("results", [])}
        ic_arr = np.array([ic_map.get(f, 0.0) for f in feats])
        ic_n   = _minmax(ic_arr)
        w_gain, w_shap, w_ic = 0.35, 0.35, 0.30
    else:
        ic_n   = np.zeros(len(feats))
        w_gain, w_shap, w_ic = 0.50, 0.50, 0.00

    score = w_gain * gain_n + w_shap * shap_n + w_ic * ic_n

    # Redundancy penalty: kurangi 0.05 per correlated partner
    n_partners = np.array([len(corr_partners.get(f, [])) for f in feats])
    penalty    = np.clip(n_partners * 0.05, 0.0, 0.30)
    score_adj  = score - penalty

    ic_verdict = {}
    if ic_data:
        ic_verdict = {r["feature"]: r["verdict"] for r in ic_data.get("results", [])}

    rows = []
    for i, f in enumerate(feats):
        rows.append({
            "feature":       f,
            "gain_norm":     round(float(gain_n[i]),  4),
            "shap_norm":     round(float(shap_n[i]),  4),
            "ic_norm":       round(float(ic_n[i]),    4),
            "n_corr_partners": int(n_partners[i]),
            "corr_partners": corr_partners.get(f, []),
            "score":         round(float(score[i]),     4),
            "penalty":       round(float(penalty[i]),   4),
            "score_adj":     round(float(score_adj[i]), 4),
            "ic_verdict":    ic_verdict.get(f, "N/A"),
        })

    rows.sort(key=lambda x: -x["score_adj"])
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature Selection: SHAP + Gain + Corr")
    parser.add_argument("--ic-json", default=None,
                        help="Path JSON hasil 03b_ic_test (opsional, untuk IC score)")
    parser.add_argument("--top-n",   type=int, default=30)
    parser.add_argument("--run-id",  default="tb_fs_shap_v1")
    parser.add_argument("--corr-thresh", type=float, default=CORR_THRESH)
    args = parser.parse_args()

    out_dir = REPORT_DIR / "experiments" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load IC data ──────────────────────────────────────────────────────────
    ic_data = None
    if args.ic_json and Path(args.ic_json).exists():
        with open(args.ic_json, encoding="utf-8") as f:
            ic_data = json.load(f)
        logger.info(f"IC data loaded: {len(ic_data.get('results', []))} fitur")
    else:
        logger.warning("--ic-json tidak ditemukan; IC score tidak dipakai")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data(cutoff=TRAIN_CUTOFF_DATE)
    df = df[df["label"].isin(LABEL_MAP)].copy()

    # Ambil semua fitur numerik minus meta
    sample_file = sorted(glob.glob(str(LABEL_DIR / "*_features_v3.parquet")))[0]
    sample_df   = pd.read_parquet(sample_file)
    all_num_cols = [c for c in sample_df.select_dtypes(include=[np.number]).columns
                    if c not in META_COLS]
    avail_cols   = [c for c in all_num_cols if c in df.columns]
    logger.info(f"Fitur yang akan dianalisis: {len(avail_cols)}")

    # ── LGBM + SHAP ──────────────────────────────────────────────────────────
    model, gain_dict, X_train, feat_list = train_lgbm_for_importance(df, avail_cols)
    shap_dict = compute_shap(model, X_train, feat_list)

    # ── Correlation ───────────────────────────────────────────────────────────
    corr_partners, redundant_pairs = correlation_analysis(
        df, avail_cols, corr_thresh=args.corr_thresh
    )

    # ── Combined Ranking ──────────────────────────────────────────────────────
    ranking = combined_ranking(
        feature_cols   = avail_cols,
        gain_dict      = gain_dict,
        shap_dict      = shap_dict,
        corr_partners  = corr_partners,
        ic_data        = ic_data,
        top_n          = args.top_n,
    )

    # ── Print Top-N ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FEATURE SELECTION — {args.run_id}")
    print(f"  {len(avail_cols)} fitur -> Target {args.top_n} terbaik")
    print(f"  Corr threshold: {args.corr_thresh}")
    print(f"{'='*80}")
    header = f"{'#':>3}  {'Feature':<35} {'Gain':>6} {'SHAP':>6} {'IC':>6} {'Corr':>5} {'Score':>7} {'Adj':>7}  IC_V"
    print(header)
    print("-" * 80)
    for i, r in enumerate(ranking[:args.top_n]):
        corr_flag = f"+{r['n_corr_partners']}" if r["n_corr_partners"] > 0 else "   "
        print(
            f"{i+1:>3}. {r['feature']:<35} "
            f"{r['gain_norm']:>6.3f} {r['shap_norm']:>6.3f} {r['ic_norm']:>6.3f} "
            f"{corr_flag:>5} {r['score']:>7.4f} {r['score_adj']:>7.4f}  {r['ic_verdict']}"
        )

    top_features = [r["feature"] for r in ranking[:args.top_n]]
    print(f"\nTop-{args.top_n} features:\n{top_features}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "run_id":         args.run_id,
        "n_features_in":  len(avail_cols),
        "top_n":          args.top_n,
        "corr_thresh":    args.corr_thresh,
        "top_features":   top_features,
        "ranking":        ranking,
        "redundant_pairs": redundant_pairs[:50],   # simpan 50 teratas saja
    }

    json_path = out_dir / "fs_ranking.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved -> {json_path}")

    # Simpan top features ke JSON (bisa langsung dipakai config)
    feat_path = out_dir / "selected_features.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2)
    logger.info(f"Saved -> {feat_path}")

    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
