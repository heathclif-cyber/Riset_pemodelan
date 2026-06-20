"""
pipeline/03d_feature_selection_v2.py — Comprehensive Non-Linear Feature Selection

Menggantikan pendekatan IC linier murni dengan pipeline 4-stage yang lebih kuat:

Stage 1: Spearman IC + Mutual Information (linear + non-linear statistical)
         → Pass: |IC| >= 0.015 OR MI >= 0.008
Stage 2: Gram-Schmidt Marginal IC (deduplication / redundancy removal)
         → Pass: |Marginal IC| >= 0.008
Stage 3: LGBM OOF Permutation Importance (MDA)
         → Pass: mean F1 drop >= 0.0002 (non-zero non-linear contribution)
Stage 4: SHAP mean |value| ranking — final ranking & validation
         → Semua yang lolos Stage 3 diranking ulang oleh SHAP magnitude

Output: models/feature_cols_v2_nonlinear.json
        reports/experiments/feature_selection_v2_nonlinear.md

Jalankan:
  python pipeline/03d_feature_selection_v2.py
  python pipeline/03d_feature_selection_v2.py --run-id lgbm_v2_nonlinear
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, REPORT_DIR,
    TRAIN_CUTOFF_DATE,
)

LABEL_ORDINAL = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# Kolom yang pasti dieksklusi dari feature candidates
META_COLS = {
    "label", "coin", "symbol",
    "h4_swing_high", "h4_swing_low",
    "hmm_regime",          # raw string regime label
    "hmm_regime_enc",      # akan ditambahkan manual setelah seleksi
    "open", "high", "low", "close",  # raw OHLC — terlalu forward-looking bias
}

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_coins(feature_candidates: list[str]) -> pd.DataFrame:
    frames = []
    for coin in ALL_COINS:
        fpath = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fpath.exists():
            print(f"  [SKIP] {coin} — file tidak ditemukan")
            continue

        df = pd.read_parquet(fpath)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue

        # Merge HMM regime jika ada
        regime_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if regime_path.exists():
            try:
                reg = pd.read_parquet(regime_path)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception:
                pass

        # Filter label valid
        df = df[df["label"].isin(LABEL_ORDINAL)].copy()
        df["label_ord"] = df["label"].map(LABEL_ORDINAL).astype(np.int8)
        df["coin"] = coin

        # Downcast float64 → float32 untuk hemat memori
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype(np.float32)

        avail = [c for c in feature_candidates if c in df.columns]
        keep_cols = avail + ["label", "label_ord", "coin", "hmm_regime_enc"]
        df = df[[c for c in keep_cols if c in df.columns]]
        frames.append(df)
        print(f"  Loaded {coin}: {len(df):,} rows")

    combined = pd.concat(frames).sort_index()
    print(f"  Total: {len(combined):,} rows × {len(combined.columns)} cols")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Stage Helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Spearman IC + Mutual Information
# ──────────────────────────────────────────────────────────────────────────────

def stage1_ic_mi(
    X: pd.DataFrame, y: np.ndarray,
    min_ic: float = 0.015,
    min_mi: float = 0.008,
) -> tuple[list[str], dict, dict]:
    print(f"\n[Stage 1] Spearman IC + Mutual Information ({len(X.columns)} fitur)...")

    # Spearman IC per fitur
    ic_scores = {}
    for col in X.columns:
        x_arr = X[col].values
        mask = ~np.isnan(x_arr)
        if mask.sum() < 100:
            ic_scores[col] = 0.0
            continue
        corr, _ = stats.spearmanr(x_arr[mask], y[mask])
        ic_scores[col] = float(corr) if not np.isnan(corr) else 0.0

    # Mutual Information (subsample 15k untuk kecepatan)
    n_mi = min(len(X), 15000)
    idx_mi = np.random.choice(len(X), n_mi, replace=False)
    X_mi = X.iloc[idx_mi].copy()
    y_mi = y[idx_mi]

    # Drop kolom yang >50% NaN (tidak bisa diisi median dengan baik)
    nan_pct = X_mi.isna().mean()
    cols_ok = nan_pct[nan_pct <= 0.5].index.tolist()
    cols_skip = [c for c in X_mi.columns if c not in cols_ok]
    if cols_skip:
        print(f"  [WARN] {len(cols_skip)} kolom di-skip MI (>50% NaN): {cols_skip}")
    X_mi_ok = X_mi[cols_ok].copy()

    # Fill NaN: median dulu, fallback 0 untuk kolom all-NaN
    col_medians = X_mi_ok.median()
    X_mi_ok = X_mi_ok.fillna(col_medians).fillna(0.0)

    mi_raw = mutual_info_classif(X_mi_ok.values, y_mi, random_state=42, n_neighbors=5)
    mi_scores = {col: float(mi_raw[i]) for i, col in enumerate(cols_ok)}
    # Kolom yang di-skip MI mendapat skor 0
    for col in cols_skip:
        mi_scores[col] = 0.0

    # Gate: lolos jika salah satu threshold terpenuhi
    keep = [
        c for c in X.columns
        if abs(ic_scores[c]) >= min_ic or mi_scores.get(c, 0.0) >= min_mi
    ]
    print(f"  IC threshold={min_ic}, MI threshold={min_mi}")
    print(f"  Stage 1 PASS: {len(keep)} / {len(X.columns)} fitur")
    return keep, ic_scores, mi_scores


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Gram-Schmidt Marginal IC
# ──────────────────────────────────────────────────────────────────────────────

def stage2_marginal_ic(
    X: pd.DataFrame, y: np.ndarray,
    feature_list: list[str],
    min_marginal: float = 0.008,
) -> tuple[list[str], dict]:
    print(f"\n[Stage 2] Gram-Schmidt Marginal IC ({len(feature_list)} fitur)...")

    X_mat = X[feature_list].copy()
    # Isi NaN dengan median
    for col in X_mat.columns:
        med = X_mat[col].median()
        X_mat[col] = X_mat[col].fillna(med if not np.isnan(med) else 0.0)

    X_r = np.column_stack([_rank_norm(X_mat.iloc[:, j].values) for j in range(X_mat.shape[1])])
    y_r = _rank_norm(y.astype(np.float64))

    remaining = list(range(X_mat.shape[1]))
    marginal = {}

    for _ in range(len(feature_list)):
        if not remaining:
            break
        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]
            nx = np.sqrt(np.dot(xj, xj))
            ny = np.sqrt(np.dot(y_r, y_r))
            corrs[k] = np.dot(xj, y_r) / (nx * ny) if nx > 1e-10 and ny > 1e-10 else 0.0

        best_k = int(np.argmax(np.abs(corrs)))
        best_j = remaining[best_k]
        marginal[feature_list[best_j]] = float(corrs[best_k])

        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j:
                X_r[:, j] = _project_out(X_r[:, j], pivot)
        y_r = _project_out(y_r, pivot)
        remaining.remove(best_j)

    keep = [f for f, m in marginal.items() if abs(m) >= min_marginal]
    print(f"  Marginal IC threshold={min_marginal}")
    print(f"  Stage 2 PASS: {len(keep)} / {len(feature_list)} fitur")
    return keep, marginal


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: OOF Permutation Importance (MDA)
# ──────────────────────────────────────────────────────────────────────────────

def stage3_mda(
    X: pd.DataFrame, y: np.ndarray,
    feature_list: list[str],
    min_mda: float = 0.0002,
    n_folds: int = 4,
    sample_size: int = 40000,
) -> tuple[list[str], dict]:
    print(f"\n[Stage 3] OOF Permutation Importance / MDA ({len(feature_list)} fitur, {n_folds} folds)...")

    X_sub = X[feature_list].copy()
    X_sub = X_sub.fillna(X_sub.median())

    # Subsample jika data terlalu besar
    if len(X_sub) > sample_size:
        idx = np.random.choice(len(X_sub), sample_size, replace=False)
        X_sub = X_sub.iloc[idx].reset_index(drop=True)
        y_sub = y[idx]
    else:
        X_sub = X_sub.reset_index(drop=True)
        y_sub = y.copy()

    mda_scores = {col: [] for col in feature_list}
    fold_size = len(X_sub) // n_folds
    idx_perm = np.random.permutation(len(X_sub))

    lgbm_quick = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=60, learning_rate=0.08,
        max_depth=4, num_leaves=15,
        min_child_samples=30, subsample=0.8,
        colsample_bytree=0.8, verbose=-1,
        random_state=42,
    )

    for fold in range(n_folds):
        val_idx = idx_perm[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(idx_perm, val_idx)

        X_tr, X_val = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
        y_tr, y_val = y_sub[train_idx], y_sub[val_idx]

        lgbm_quick.fit(X_tr, y_tr)
        base_f1 = f1_score(y_val, lgbm_quick.predict(X_val), average="macro", zero_division=0)

        for col in feature_list:
            X_perm = X_val.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            drop = base_f1 - f1_score(y_val, lgbm_quick.predict(X_perm), average="macro", zero_division=0)
            mda_scores[col].append(drop)

        print(f"  Fold {fold+1}/{n_folds} done | base F1={base_f1:.4f}")

    avg_mda = {col: float(np.mean(v)) for col, v in mda_scores.items()}
    keep = [c for c in feature_list if avg_mda[c] >= min_mda]
    print(f"  MDA threshold={min_mda}")
    print(f"  Stage 3 PASS: {len(keep)} / {len(feature_list)} fitur")
    return keep, avg_mda


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: SHAP TreeExplainer — Final Ranking
# ──────────────────────────────────────────────────────────────────────────────

def stage4_shap(
    X: pd.DataFrame, y: np.ndarray,
    feature_list: list[str],
    shap_sample: int = 8000,
) -> dict:
    print(f"\n[Stage 4] SHAP TreeExplainer — Ranking {len(feature_list)} fitur...")

    X_sub = X[feature_list].copy().fillna(0.0)
    if len(X_sub) > shap_sample:
        idx = np.random.choice(len(X_sub), shap_sample, replace=False)
        X_shap = X_sub.iloc[idx].reset_index(drop=True)
        y_shap = y[idx]
    else:
        X_shap = X_sub.reset_index(drop=True)
        y_shap = y.copy()

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=80, learning_rate=0.08,
        max_depth=5, num_leaves=20,
        random_state=42, verbose=-1,
    )
    model.fit(X_shap, y_shap)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)

    # Handle kedua SHAP API:
    # - Old (<0.41): list of arrays, satu per kelas, shape (n_samples, n_features)
    # - New (>=0.41): single array shape (n_samples, n_features, n_classes) atau (n_samples, n_features)
    if isinstance(shap_vals, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    else:
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) → mean over samples & classes
            mean_abs_shap = np.abs(arr).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(arr).mean(axis=0)
    shap_dict = {col: float(mean_abs_shap[i]) for i, col in enumerate(feature_list)}

    ranked = sorted(shap_dict.items(), key=lambda x: -x[1])
    print(f"  Top 10 SHAP fitur:")
    for feat, val in ranked[:10]:
        print(f"    {feat:35s}  {val:.6f}")
    return dict(ranked)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Run ID untuk naming output")
    parser.add_argument("--min-ic",       type=float, default=0.015, help="Stage 1 Spearman IC threshold")
    parser.add_argument("--min-mi",       type=float, default=0.008, help="Stage 1 Mutual Info threshold")
    parser.add_argument("--min-marginal", type=float, default=0.008, help="Stage 2 Marginal IC threshold")
    parser.add_argument("--min-mda",      type=float, default=0.0002, help="Stage 3 MDA F1 drop threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or f"lgbm_nonlinear_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\n{'='*65}")
    print(f"  4-STAGE NON-LINEAR FEATURE SELECTION — {run_id}")
    print(f"  Stage 1: Spearman IC (>={args.min_ic}) + Mutual Info (>={args.min_mi})")
    print(f"  Stage 2: Marginal IC (>={args.min_marginal})")
    print(f"  Stage 3: MDA Permutation Importance (>={args.min_mda})")
    print(f"  Stage 4: SHAP TreeExplainer (Final Ranking)")
    print(f"{'='*65}\n")

    # ── Candidate features: semua kolom kecuali META_COLS ────────────────────
    print("[0] Loading data from all coins...")
    # Gunakan semua fitur potensial dari parquet (bukan hanya IC32)
    sample_df = pd.read_parquet(LABEL_DIR / "BTCUSDT_features_v3.parquet")
    all_candidates = [c for c in sample_df.columns if c not in META_COLS]
    print(f"  Kandidat fitur: {len(all_candidates)}")

    df = load_all_coins(all_candidates)

    feature_candidates = [c for c in all_candidates if c in df.columns]
    print(f"  Fitur tersedia di data: {len(feature_candidates)}")

    X_full = df[feature_candidates].copy()
    y_full = df["label_ord"].values

    np.random.seed(42)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    s1_keep, ic_scores, mi_scores = stage1_ic_mi(
        X_full, y_full,
        min_ic=args.min_ic,
        min_mi=args.min_mi,
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    s2_keep, marginal_scores = stage2_marginal_ic(
        X_full, y_full, s1_keep,
        min_marginal=args.min_marginal,
    )

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    s3_keep, mda_scores = stage3_mda(
        X_full, y_full, s2_keep,
        min_mda=args.min_mda,
    )

    # Fallback: jika terlalu sedikit, ambil top 15 MDA
    if len(s3_keep) < 10:
        print(f"  ⚠️ Hanya {len(s3_keep)} fitur lolos Stage 3, fallback ke top-15 MDA")
        s3_keep = sorted(s2_keep, key=lambda c: -mda_scores.get(c, 0))[:15]

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    shap_ranking = stage4_shap(X_full, y_full, s3_keep)

    # Tambahkan hmm_regime_enc sebagai fitur konteks wajib
    final_features = list(shap_ranking.keys())
    if "hmm_regime_enc" not in final_features:
        final_features.append("hmm_regime_enc")

    # ── Simpan hasil ──────────────────────────────────────────────────────────
    out_feat_path = MODEL_DIR / f"feature_cols_{run_id}.json"
    with open(out_feat_path, "w") as f:
        json.dump(final_features, f, indent=2)
    print(f"\n✅ Feature cols ({len(final_features)}) → {out_feat_path}")

    # ── Report Markdown ───────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"feature_selection_{run_id}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🔬 4-Stage Non-Linear Feature Selection — `{run_id}`\n\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | cutoff={TRAIN_CUTOFF_DATE.date()}*\n\n")
        f.write(f"| Stage | Method | Threshold | Input | Output |\n")
        f.write(f"|:---|:---|:---:|:---:|:---:|\n")
        f.write(f"| 1 | Spearman IC + Mutual Info | IC>={args.min_ic} OR MI>={args.min_mi} | {len(feature_candidates)} | {len(s1_keep)} |\n")
        f.write(f"| 2 | Gram-Schmidt Marginal IC | >={args.min_marginal} | {len(s1_keep)} | {len(s2_keep)} |\n")
        f.write(f"| 3 | OOF MDA Permutation | F1 drop>={args.min_mda} | {len(s2_keep)} | {len(s3_keep)} |\n")
        f.write(f"| 4 | SHAP TreeExplainer | (Ranking) | {len(s3_keep)} | {len(final_features)} |\n\n")
        f.write(f"**Final feature count: {len(final_features)}**\n\n---\n\n")

        f.write("## Final Feature List (SHAP-ranked)\n\n")
        f.write("| Rank | Feature | Standalone IC | Mutual Info | Marginal IC | MDA Drop | SHAP |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for rank, (feat, shap_val) in enumerate(shap_ranking.items(), 1):
            f.write(
                f"| {rank} | `{feat}` | {ic_scores.get(feat, 0):+.4f} | "
                f"{mi_scores.get(feat, 0):.4f} | {marginal_scores.get(feat, 0):+.4f} | "
                f"{mda_scores.get(feat, 0):+.5f} | {shap_val:.6f} |\n"
            )
        if "hmm_regime_enc" in final_features and "hmm_regime_enc" not in shap_ranking:
            f.write(f"| - | `hmm_regime_enc` | — | — | — | — | (added manually) |\n")

        f.write("\n---\n\n")
        f.write("## Features Dropped per Stage\n\n")

        s1_drop = [c for c in feature_candidates if c not in s1_keep]
        f.write(f"### Stage 1 Drop ({len(s1_drop)} fitur) — IC+MI too weak\n")
        f.write(", ".join(f"`{c}`" for c in sorted(s1_drop)) + "\n\n")

        s2_drop = [c for c in s1_keep if c not in s2_keep]
        f.write(f"### Stage 2 Drop ({len(s2_drop)} fitur) — Redundant\n")
        f.write(", ".join(f"`{c}`" for c in sorted(s2_drop)) + "\n\n")

        s3_drop = [c for c in s2_keep if c not in s3_keep]
        f.write(f"### Stage 3 Drop ({len(s3_drop)} fitur) — No non-linear contribution\n")
        f.write(", ".join(f"`{c}`" for c in sorted(s3_drop)) + "\n\n")

    print(f"✅ Report → {report_path}")

    print(f"\n{'='*65}")
    print(f"  SELESAI — {run_id}")
    print(f"  Input fitur   : {len(feature_candidates)}")
    print(f"  Stage 1 PASS  : {len(s1_keep)}")
    print(f"  Stage 2 PASS  : {len(s2_keep)}")
    print(f"  Stage 3 PASS  : {len(s3_keep)}")
    print(f"  Final (+ HMM) : {len(final_features)}")
    print(f"  Output        : {out_feat_path}")
    print(f"{'='*65}\n")

    return final_features, run_id


if __name__ == "__main__":
    main()
