"""
tools/audit_hmm_states.py — Diagnostik HMM: apakah state saat ini informatif?

Cek:
1. Distribusi state per koin (apakah state seimbang atau degenerate?)
2. Transisi antar state (apakah stabil atau loncat-loncat?)
3. WR LGBM OOF per state (apakah state membedakan peluang trade?)
4. Mean fitur per state (apakah cluster masuk akal secara ekonomi?)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM

from config import TRAINING_COINS
from core.regime import _build_hmm_features, _align_states, REGIME_NAMES_4

RUN_DIR  = Path("models/runs/ic32_regime_v2")
PROC_DIR = Path("data/training/processed")

SEP = "=" * 65

# ─── 1. Load OOF predictions (punya hmm_regime_enc per bar) ──────────────────

print(SEP)
print("BAGIAN 1: WR LGBM OOF per HMM state")
print(SEP)

oof_path = RUN_DIR / "oof_predictions.parquet"
if not oof_path.exists():
    print("ERROR: oof_predictions.parquet tidak ditemukan")
    sys.exit(1)

oof = pd.read_parquet(oof_path)
print(f"OOF bars: {len(oof):,} | has_oof: {oof['has_oof'].sum():,}")
print(f"Kolom tersedia: {oof.columns.tolist()}")
print()

# Cek apakah ada kolom regime di OOF
if "hmm_regime_enc" not in oof.columns:
    print("hmm_regime_enc tidak ada di OOF — coba ambil dari labeled parquet")
    # Load dari labeled data
    label_dir = Path("data/training/labeled")
    dfs = []
    for coin in TRAINING_COINS[:3]:  # sample 3 koin dulu
        fp = label_dir / f"{coin}_features_v3.parquet"
        if fp.exists():
            df = pd.read_parquet(fp, columns=["hmm_regime_enc"])
            df["coin"] = coin
            dfs.append(df)
    if dfs:
        regime_df = pd.concat(dfs)
        oof = oof.join(regime_df[["hmm_regime_enc"]], how="left")
        print(f"Joined hmm_regime_enc dari labeled data ({oof['hmm_regime_enc'].notna().sum():,} bar)\n")

# ─── Simulasi trade dari OOF (pake swing_label + proba) ─────────────────────
# OOF punya: p0(short), p1(flat), p2(long), swing_label(0=short,1=flat,2=long)
# Prediksi = argmax proba; benar jika match label

if "hmm_regime_enc" in oof.columns and oof["hmm_regime_enc"].notna().sum() > 100:
    oof_valid = oof[oof["has_oof"]].copy()
    oof_valid["pred"] = oof_valid[["p0", "p1", "p2"]].values.argmax(axis=1)
    oof_valid["correct"] = (oof_valid["pred"] == oof_valid["swing_label"]).astype(int)

    # Hanya bar yang diprediksi LONG atau SHORT (bukan FLAT)
    directional = oof_valid[oof_valid["pred"] != 1].copy()
    directional["state_name"] = directional["hmm_regime_enc"].map({
        0: "TRENDING_DOWN", 1: "RANGING_LOW_VOL",
        2: "RANGING_HIGH_VOL", 3: "TRENDING_UP"
    })

    print("WR per HMM state (prediksi LONG/SHORT, bukan FLAT):")
    state_stats = directional.groupby("state_name").agg(
        n_trades=("correct", "count"),
        wr=("correct", "mean"),
        n_long=("pred", lambda x: (x == 2).sum()),
        n_short=("pred", lambda x: (x == 0).sum()),
    ).sort_index()
    state_stats["wr_pct"] = (state_stats["wr"] * 100).round(1)
    print(state_stats[["n_trades", "n_long", "n_short", "wr_pct"]].to_string())
    print()
    print(f"WR range: {state_stats['wr_pct'].min():.1f}% - {state_stats['wr_pct'].max():.1f}%")
    gap = state_stats["wr_pct"].max() - state_stats["wr_pct"].min()
    print(f"Gap antar state: {gap:.1f}% {'<< KECIL, HMM tidak informatif' if gap < 5 else '>> BAGUS, state berdeda signifikan'}")
else:
    print("Tidak bisa hitung WR per state — hmm_regime_enc tidak tersedia di OOF\n")
    print("Alternatif: load dari labeled parquet langsung\n")

# ─── 2. Load HMM pkl dan inspeksi parameter ──────────────────────────────────

print()
print(SEP)
print("BAGIAN 2: Inspeksi HMM model (means, transition matrix)")
print(SEP)

# Cari pkl HMM yang tersimpan
hmm_dir = Path("data/training/regime")
if hmm_dir.exists():
    pkl_files = list(hmm_dir.glob("**/*.pkl"))
    if not pkl_files:
        pkl_files = list(hmm_dir.glob("*.pkl"))
else:
    pkl_files = []

if not pkl_files:
    # Mungkin di run_dir
    pkl_files = list(RUN_DIR.glob("*hmm*.pkl"))

if pkl_files:
    # Ambil contoh satu koin (BTC atau koin pertama)
    btc_pkl = [p for p in pkl_files if "BTC" in str(p)]
    sample_pkl = btc_pkl[0] if btc_pkl else pkl_files[0]
    print(f"Sample: {sample_pkl.name}")
    try:
        data = joblib.load(sample_pkl)
        if isinstance(data, dict):
            model = data.get("model") or data.get("hmm")
        else:
            model = data
        if hasattr(model, "means_"):
            print(f"\nMeans (return, vol, momentum, vol_ratio) per state:")
            feat_names = ["return_1bar", "volatility_24", "momentum_48", "volume_ratio"]
            means_df = pd.DataFrame(model.means_, columns=feat_names)
            means_df.index.name = "raw_state"
            print(means_df.round(5).to_string())
            print(f"\nTransition matrix:")
            trans_df = pd.DataFrame(model.transmat_.round(3))
            print(trans_df.to_string())
            diag = np.diag(model.transmat_)
            print(f"\nDiagonal (persistence): {diag.round(3)}")
            print(f"Min persistence: {diag.min():.3f} {'<< state tidak stabil (loncat-loncat)' if diag.min() < 0.7 else '>> OK'}")
        else:
            print("Model tidak punya means_ — bukan GaussianHMM")
    except Exception as e:
        print(f"Gagal load pkl: {e}")
else:
    print("Tidak ada HMM pkl yang tersimpan di", hmm_dir)

# ─── 3. Fit ulang HMM dari scratch pada data terbaru, tampilkan diagnostik ───

print()
print(SEP)
print("BAGIAN 3: Fit HMM fresh pada sample koin, cek state distribution")
print(SEP)

sample_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
for coin in sample_coins:
    fp = PROC_DIR / f"{coin}_h4.parquet"
    if not fp.exists():
        fp = PROC_DIR / f"{coin}.parquet"
    if not fp.exists():
        print(f"  {coin}: file tidak ditemukan")
        continue

    try:
        df = pd.read_parquet(fp)
        if "close" not in df.columns:
            print(f"  {coin}: kolom close tidak ada")
            continue

        X = _build_hmm_features(df)
        model = GaussianHMM(n_components=4, covariance_type="diag",
                            n_iter=100, random_state=42, tol=1e-3)
        model.fit(X)
        state_map = _align_states(model, 4)
        raw_labels = model.predict(X)
        labels = [state_map[int(s)] for s in raw_labels]

        dist = pd.Series(labels).value_counts(normalize=True).round(3)
        diag = np.diag(model.transmat_)

        # Means dalam state_map order
        print(f"\n{coin}:")
        print(f"  Distribusi state: {dist.to_dict()}")
        print(f"  Persistence (diagonal): min={diag.min():.3f} mean={diag.mean():.3f}")

        # Print means per canonical state
        means_by_state = {}
        for raw_idx, name in state_map.items():
            means_by_state[name] = model.means_[raw_idx]
        for name in REGIME_NAMES_4:
            if name in means_by_state:
                m = means_by_state[name]
                print(f"  {name:20}: ret={m[0]:+.5f}  vol={m[1]:.5f}  mom={m[2]:+.5f}  vr={m[3]:+.5f}")

    except Exception as e:
        print(f"  {coin}: error — {e}")

print()
print(SEP)
print("SELESAI. Interpretasi:")
print("- Gap WR antar state >= 5% → HMM informatif, worth optimasi")
print("- Gap WR < 5% → HMM hampir tidak berguna, perlu feature overhaul")
print("- Persistence diagonal < 0.7 → state terlalu volatile (noise)")
print("- State distribution terlalu skewed (satu state >70%) → degenerate")
print(SEP)
