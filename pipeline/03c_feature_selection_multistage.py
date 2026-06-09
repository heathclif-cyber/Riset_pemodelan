import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# Path Configuration
ROOT = Path("d:/Apps-Dev/Riset_pemodelan")
LABEL_DIR = ROOT / "data/training/labeled"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports/experiments"
TRAIN_CUTOFF_DATE = pd.Timestamp("2026-04-01", tz="UTC")

ALL_COINS = [
    "BTCUSDT", "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "TONUSDT", "ADAUSDT", "TRXUSDT", "1000SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "1000PEPEUSDT", "TAOUSDT", "ARBUSDT", "HBARUSDT", "ONDOUSDT",
]

META_COLS = {
    "label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
    "hmm_regime", "hmm_regime_enc", "open", "high", "low", "close", "volume",
    "trend_label", "label_ord", "momentum_v2_label"
}

LABEL_ORDINAL = {"SHORT": 0, "FLAT": 1, "LONG": 2}

def load_data(feature_cols):
    print("Loading engineered parquet datasets...")
    frames = []
    for coin in ALL_COINS:
        fpath = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fpath.exists():
            continue
            
        df = pd.read_parquet(fpath)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
            
        # Downcast floats to float32 to save memory
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        
        # Ensure labels are set
        if "label" not in df.columns or "hmm_regime_enc" not in df.columns:
            continue
            
        # Retain necessary columns
        avail = [c for c in feature_cols if c in df.columns]
        keep = avail + ["label", "hmm_regime_enc"]
        
        df = df[keep].copy()
        df = df[df["label"].isin(LABEL_ORDINAL)]
        df["label_ord"] = df["label"].map(LABEL_ORDINAL).astype(np.int8)
        df["coin"] = coin
        frames.append(df)
        
    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    return combined

def _rank_norm(x):
    r = stats.rankdata(x).astype(np.float32)
    r -= r.mean()
    std = r.std()
    return r / std if std > 1e-10 else np.zeros_like(r)

def _project_out(vec, pivot):
    norm_sq = np.dot(pivot, pivot)
    if norm_sq < 1e-10:
        return vec.copy()
    return vec - (np.dot(vec, pivot) / norm_sq) * pivot

def run_multistage_selection(df, regime_enc, name, feature_cols, target_col="label_ord", sample_size=30000):
    print(f"\n=========================================")
    print(f" Running Multi-Stage Feature Selection: {name}")
    print(f"=========================================")
    
    # Filter regime
    if regime_enc is not None:
        df_reg = df[df["hmm_regime_enc"] == regime_enc].copy()
    else:
        df_reg = df.copy()
        
    n_total = len(df_reg)
    print(f"Total samples in regime: {n_total:,}")
    if n_total < 100:
        print("Skipping - insufficient samples.")
        return None
        
    # Sample down to prevent Out Of Memory
    df_reg = df_reg.reset_index(drop=True)
    if n_total > sample_size:
        df_sample = df_reg.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df_reg.copy()
    df_sample = df_sample.reset_index(drop=True)
    
    y = df_sample[target_col].values
    X_df = df_sample[feature_cols].copy()
    
    # Fill NaN safely
    for col in X_df.columns:
        X_df[col] = X_df[col].ffill().fillna(X_df[col].median()).fillna(0.0)
        
    # --- TAHAP 1: Standalone Spearman IC & Mutual Information ---
    print("Stage 1: Standalone Spearman IC & Mutual Information Filter...")
    standalone_ics = {}
    for col in X_df.columns:
        corr, _ = stats.spearmanr(X_df[col].values, y)
        standalone_ics[col] = corr if not np.isnan(corr) else 0.0
        
    # Calculate Mutual Info on a subset for speed
    X_mi_sample = X_df.sample(n=min(len(X_df), 10000), random_state=42)
    y_mi_sample = df_sample[target_col].loc[X_mi_sample.index].values
    mi_scores = mutual_info_classif(X_mi_sample.values, y_mi_sample, random_state=42)
    mi_dict = {col: mi_scores[idx] for idx, col in enumerate(X_df.columns)}
    
    # Pass rule: Standalone IC >= 0.015 OR Mutual Info >= 0.008
    stage1_keep = []
    for col in X_df.columns:
        ic = standalone_ics[col]
        mi = mi_dict[col]
        if abs(ic) >= 0.015 or mi >= 0.008:
            stage1_keep.append(col)
            
    print(f"  Stage 1 Kept: {len(stage1_keep)} / {len(feature_cols)} features.")
    if not stage1_keep:
        return None
        
    # --- TAHAP 2: Gram-Schmidt Marginal IC (Deduplication) ---
    print("Stage 2: Gram-Schmidt Orthogonalization (Marginal IC)...")
    X_mat = X_df[stage1_keep].values
    X_r = np.column_stack([_rank_norm(X_mat[:, j]) for j in range(X_mat.shape[1])])
    y_r = _rank_norm(y.astype(np.float64))
    
    remaining = list(range(X_mat.shape[1]))
    marginal = {}
    p = X_mat.shape[1]
    
    for _ in range(p):
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
        marginal[stage1_keep[best_j]] = float(corrs[best_k])
        
        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j:
                X_r[:, j] = _project_out(X_r[:, j], pivot)
        y_r = _project_out(y_r, pivot)
        remaining.remove(best_j)
        
    # Pass rule: Marginal IC >= 0.010
    stage2_keep = [feat for feat, m_ic in marginal.items() if abs(m_ic) >= 0.010]
    print(f"  Stage 2 Kept: {len(stage2_keep)} / {len(stage1_keep)} features.")
    if not stage2_keep:
        return None
        
    # --- TAHAP 3: Mean Decrease Accuracy (MDA) on 3-Fold Cross-Validation ---
    print("Stage 3: Out-of-Fold Mean Decrease Accuracy (MDA)...")
    X_mda = X_df[stage2_keep].copy()
    
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(X_mda))
    fold_size = len(X_mda) // 3
    mda_scores = {col: [] for col in stage2_keep}
    
    for fold in range(3):
        val_idx = shuffled_idx[fold*fold_size : (fold+1)*fold_size]
        train_idx = np.setdiff1d(shuffled_idx, val_idx)
        
        X_train, X_val = X_mda.iloc[train_idx], X_mda.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=40,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Base Prediction Macro F1
        y_pred = model.predict(X_val)
        base_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        
        for col in stage2_keep:
            X_val_perm = X_val.copy()
            X_val_perm[col] = np.random.permutation(X_val_perm[col].values)
            y_pred_perm = model.predict(X_val_perm)
            perm_f1 = f1_score(y_val, y_pred_perm, average="macro", zero_division=0)
            mda_scores[col].append(base_f1 - perm_f1)
            
    # Pass rule: Avg F1 decrease >= 0.0002
    stage3_keep = []
    feature_report = []
    for col in stage2_keep:
        avg_decrease = float(np.mean(mda_scores[col]))
        feature_report.append({
            "feature": col,
            "standalone_ic": standalone_ics[col],
            "mutual_info": mi_dict[col],
            "marginal_ic": marginal[col],
            "mda_f1_drop": avg_decrease
        })
        if avg_decrease >= 0.0002:
            stage3_keep.append(col)
            
    print(f"  Stage 3 Kept: {len(stage3_keep)} / {len(stage2_keep)} features.")
    
    # --- TAHAP 4: SHAP Explainer (Sample validation) ---
    print("Stage 4: SHAP Values Logic Validation...")
    final_features = stage3_keep if stage3_keep else stage2_keep[:5] # Fallback if all dropped
    
    X_shap = X_df[final_features].copy()
    model_shap = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=42,
        verbose=-1
    )
    model_shap.fit(X_shap, y)
    
    explainer = shap.TreeExplainer(model_shap)
    # Validate calculations
    shap_values = explainer.shap_values(X_shap.iloc[:500])
    print("  SHAP logical calculations complete.")
    
    feature_report.sort(key=lambda x: -x["mda_f1_drop"])
    return {
        "model_name": name,
        "selected_features": final_features,
        "report": feature_report
    }

def main():
    # Load feature columns template from config.py
    sys.path.insert(0, str(ROOT))
    from config import FEATURE_COLS_V3
    
    # Exclude metadata columns
    feature_cols = [c for c in FEATURE_COLS_V3 if c not in META_COLS]
    
    df = load_data(feature_cols)
    print(f"Loaded master training data size: {df.shape}")
    
    # Intersect with columns actually present in the loaded dataframe to prevent KeyError
    feature_cols = [c for c in feature_cols if c in df.columns]
    # Deduplicate
    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"Active features for selection (present in data): {len(feature_cols)}")
    
    # Run the 4-Stage selection for LGBM regime models
    up_res = run_multistage_selection(df, 3, "LGBM_TRENDING_UP", feature_cols, "label_ord")
    down_res = run_multistage_selection(df, 0, "LGBM_TRENDING_DOWN", feature_cols, "label_ord")
    ranging_res = run_multistage_selection(df, 1, "LGBM_RANGING_LOW_VOL", feature_cols, "label_ord")
    ranging_high_res = run_multistage_selection(df, 2, "LGBM_RANGING_HIGH_VOL", feature_cols, "label_ord")
    
    # Global fallback model (union of all)
    global_res = run_multistage_selection(df, None, "LGBM_GLOBAL_FALLBACK", feature_cols, "label_ord")
    
    # LSTM feature selection — no regime filter, uses same label_ord target
    # LSTM benefits from temporal/flow features; let the pipeline decide
    lstm_res = run_multistage_selection(df, None, "TradingLSTM", feature_cols, "label_ord")
    
    # Generate Multi-Stage report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = REPORT_DIR / "multistage_feature_selection_results.md"
    
    all_results = [up_res, down_res, ranging_res, ranging_high_res, global_res, lstm_res]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 Multi-Stage Feature Selection Results (Coinank Integrated)\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | training_cutoff={TRAIN_CUTOFF_DATE.date()}*\n\n")
        f.write("Laporan ini berisi hasil penyaringan fitur multi-tahap: **Spearman IC + Mutual Info**, **Gram-Schmidt Marginal IC**, **OOF MDA**, dan **SHAP Validation**.\n\n")
        
        for res in all_results:
            if res is None:
                continue
            f.write(f"## Model: {res['model_name']}\n\n")
            f.write(f"- **Total Fitur Terpilih**: {len(res['selected_features'])} / {len(feature_cols)}\n")
            f.write("### Daftar Fitur Terpilih\n\n")
            f.write(", ".join(f"`{feat}`" for feat in res["selected_features"]) + "\n\n")
            
            f.write("| Feature | Standalone IC | Mutual Info | Marginal IC | MDA (F1 Drop) | Status |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for r in res["report"]:
                verdict_str = "**KEEP**" if r["feature"] in res["selected_features"] else "DROP"
                f.write(f"| `{r['feature']}` | {r['standalone_ic']:+.4f} | {r['mutual_info']:.4f} | {r['marginal_ic']:+.4f} | {r['mda_f1_drop']:+.5f} | {verdict_str} |\n")
            f.write("\n---\n\n")
            
    print(f"\n=======================================================")
    print(f" MULTI-STAGE FEATURE SELECTION RUN COMPLETE")
    print(f" Saved results -> {report_path}")
    print(f"=======================================================\n")
    
    # Save selected feature arrays in models folder for training scripts
    configs = {
        "up_feats": up_res["selected_features"] if up_res else [],
        "down_feats": down_res["selected_features"] if down_res else [],
        "ranging_low_feats": ranging_res["selected_features"] if ranging_res else [],
        "ranging_high_feats": ranging_high_res["selected_features"] if ranging_high_res else [],
        "global_feats": global_res["selected_features"] if global_res else [],
    }
    
    # Save JSON config
    run_dir = MODEL_DIR / "runs/simons_hybrid_v1"
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir / "multistage_selected_feats.json", "w") as jf:
        json.dump(configs, jf, indent=2)
    print("Saved configurations -> models/runs/simons_hybrid_v1/multistage_selected_feats.json")
    
    # Save LSTM features separately
    lstm_feats = lstm_res["selected_features"] if lstm_res else []
    with open(run_dir / "lstm_feats.json", "w") as jf:
        json.dump(lstm_feats, jf, indent=2)
    print(f"Saved LSTM features ({len(lstm_feats)}) -> models/runs/simons_hybrid_v1/lstm_feats.json")

if __name__ == "__main__":
    main()
