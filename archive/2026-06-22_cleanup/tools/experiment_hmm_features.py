"""
tools/experiment_hmm_features.py — HMM Feature Set Sweep

Tujuan: cari feature set HMM yang paling informatif untuk LGBM
TANPA retrain LGBM — pakai OOF predictions yang sudah ada.

Metric: WR gap (max_state_WR - min_state_WR) pada OOF directional bars.
Semakin besar gap = state HMM makin membedakan kualitas sinyal.

Feature sets:
  V0 (baseline) : [ret_1bar, vol_24, mom_48, vol_ratio]
  V1            : V0 + btc_ret_h4
  V2            : V0 + btc_ret_h4 + btc_mom_48
  V3            : Parkinson vol + ret + mom + btc_ret
  V4            : V0 + universe_index
  V5            : V0 + universe_index + corr_dispersion
  V6            : V5 + btc_ret
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from core.regime import _align_states, REGIME_NAMES_4
from config import TRAINING_COINS

SEP = "=" * 65
PROC_DIR  = Path("data/training/processed")
OOF_PATH  = Path("models/runs/ic32_regime_v2/oof_predictions.parquet")
LABEL_DIR = Path("data/training/labeled")

N_STATES    = 4
N_ITER      = 100
RANDOM_SEED = 42


# ─── Load & Cache H4 data untuk semua koin ────────────────────────────────────

def load_h4(coin: str) -> pd.DataFrame | None:
    fp = PROC_DIR / f"{coin}_clean.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp).rename(columns={
        "1h_open": "open", "1h_high": "high",
        "1h_low": "low",   "1h_close": "close",
        "1h_volume": "volume",
    })
    if "close" not in df.columns:
        return None
    h4 = df[["open","high","low","close","volume"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()
    return h4


print("Loading H4 data untuk semua koin...")
h4_cache = {}
for coin in TRAINING_COINS:
    h4 = load_h4(coin)
    if h4 is not None and len(h4) >= 200:
        h4_cache[coin] = h4
print(f"Loaded: {len(h4_cache)} koin\n")

btc_h4 = h4_cache.get("BTCUSDT")
if btc_h4 is None:
    print("ERROR: BTCUSDT tidak tersedia — diperlukan untuk fitur cross-asset")
    sys.exit(1)

# ─── Hitung cross-asset features ─────────────────────────────────────────────

# Universe index: rata-rata H4 return semua koin (equal weight)
print("Hitung universe index (avg H4 return 21 koin)...")
all_rets = []
for coin, h4 in h4_cache.items():
    r = h4["close"].pct_change().rename(coin)
    all_rets.append(r)
ret_panel = pd.concat(all_rets, axis=1).fillna(0)
universe_ret = ret_panel.mean(axis=1)            # equal-weight universe return
universe_mom = (universe_ret.rolling(48, min_periods=8).mean())  # universe momentum

# Correlation dispersion: rolling std of pairwise correlations
print("Hitung correlation dispersion (rolling 20-bar pairwise)...")
CORR_WIN = 20
rolling_corr_std = pd.Series(index=ret_panel.index, dtype=float)
for i in range(CORR_WIN, len(ret_panel)):
    window = ret_panel.iloc[i-CORR_WIN:i]
    corr_mat = window.corr().values
    upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
    upper = upper[~np.isnan(upper)]
    rolling_corr_std.iloc[i] = float(np.std(upper)) if len(upper) > 0 else np.nan
rolling_corr_std = rolling_corr_std.fillna(rolling_corr_std.median())
print(f"  Corr dispersion range: {rolling_corr_std.min():.4f} - {rolling_corr_std.max():.4f}\n")

# BTC features
btc_ret  = btc_h4["close"].pct_change().fillna(0)
btc_mom  = (btc_h4["close"] / btc_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)


# ─── Feature Builders ─────────────────────────────────────────────────────────

def parkinson_vol(h4: pd.DataFrame, window: int = 24) -> pd.Series:
    """Parkinson realized volatility — lebih efisien dari rolling std."""
    log_hl = np.log(h4["high"] / h4["low"])
    pk = (log_hl**2 / (4 * np.log(2))).rolling(window, min_periods=4).mean().apply(np.sqrt)
    return pk.fillna(log_hl.rolling(window).std().fillna(log_hl.std()))


def build_features(coin: str, version: str) -> np.ndarray | None:
    h4 = h4_cache.get(coin)
    if h4 is None:
        return None

    ret = h4["close"].pct_change().fillna(0)
    vol_std = ret.rolling(24, min_periods=4).std().fillna(ret.std())
    vol_pk  = parkinson_vol(h4, 24)
    mom     = (h4["close"] / h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)
    vr      = np.log((h4["volume"] / h4["volume"].rolling(48, min_periods=8).mean()).clip(0.01, 20)).fillna(0)

    # Align cross-asset to coin index
    def align(s: pd.Series) -> pd.Series:
        return s.reindex(h4.index).ffill().fillna(0)

    u_ret  = align(universe_ret)
    u_mom  = align(universe_mom.fillna(0))
    c_disp = align(rolling_corr_std)
    b_ret  = align(btc_ret)
    b_mom  = align(btc_mom)

    if version == "V0":
        cols = [ret, vol_std, mom, vr]
    elif version == "V1":
        cols = [ret, vol_std, mom, vr, b_ret]
    elif version == "V2":
        cols = [ret, vol_std, mom, vr, b_ret, b_mom]
    elif version == "V3":
        cols = [ret, vol_pk, mom, b_ret]
    elif version == "V4":
        cols = [ret, vol_std, mom, vr, u_ret]
    elif version == "V5":
        cols = [ret, vol_std, mom, vr, u_ret, c_disp]
    elif version == "V6":
        cols = [ret, vol_std, mom, vr, u_ret, c_disp, b_ret]
    else:
        raise ValueError(f"Unknown version: {version}")

    X = np.column_stack([c.values for c in cols])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Load OOF + labeled data ──────────────────────────────────────────────────

print("Load OOF predictions...")
oof = pd.read_parquet(OOF_PATH)
oof_valid = oof[oof["has_oof"]].copy()
oof_valid["pred"] = oof_valid[["p0","p1","p2"]].values.argmax(axis=1)
oof_valid["correct"] = (oof_valid["pred"] == oof_valid["swing_label"]).astype(int)
directional = oof_valid[oof_valid["pred"] != 1].copy()
print(f"Directional OOF bars: {len(directional):,}\n")

# Load labeled parquet untuk tiap koin (punya index timestamps per koin)
print("Load labeled parquet (untuk join timestamp -> koin)...")
coin_index = {}
for coin in TRAINING_COINS:
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if fp.exists():
        idx = pd.read_parquet(fp, columns=[]).index
        coin_index[coin] = idx
print(f"  Tersedia: {len(coin_index)} koin\n")


# ─── Evaluasi satu feature set ────────────────────────────────────────────────

def evaluate_version(version: str) -> dict:
    """
    Fit HMM dengan feature set 'version' per koin.
    Join state labels ke OOF directional bars.
    Return WR per state dan WR gap.
    """
    # Kumpulkan mapping timestamp -> state untuk semua koin
    state_records = []

    for coin in TRAINING_COINS:
        if coin not in coin_index or coin not in h4_cache:
            continue

        X = build_features(coin, version)
        if X is None or len(X) < 200:
            continue

        h4 = h4_cache[coin]

        try:
            n_components = min(N_STATES, len(X) // 50)
            if n_components < 2:
                continue
            model = GaussianHMM(n_components=n_components, covariance_type="diag",
                                n_iter=N_ITER, random_state=RANDOM_SEED, tol=1e-3,
                                verbose=False)
            model.fit(X)
            state_map = _align_states(model, n_components)
            raw_labels = model.predict(X)
            labels = [state_map[int(s)] for s in raw_labels]

            # H4 labels -> H1 timestamps via ffill
            h4_regime = pd.Series(labels, index=h4.index, name="state")
            # Align ke H1 coin index
            h1_idx = coin_index[coin]
            h1_regime = h4_regime.reindex(h1_idx).ffill().dropna()

            df_coin = pd.DataFrame({"state": h1_regime, "coin": coin})
            state_records.append(df_coin)

        except Exception:
            continue

    if not state_records:
        return {"version": version, "gap": 0, "states": {}}

    all_states = pd.concat(state_records)

    # Join ke OOF directional bars
    merged = directional.join(all_states["state"], how="left")
    merged = merged.dropna(subset=["state"])

    if len(merged) < 1000:
        return {"version": version, "gap": 0, "states": {}, "n": len(merged)}

    # WR per state
    state_wr = merged.groupby("state")["correct"].agg(["mean","count"])
    state_wr.columns = ["wr", "n"]
    state_wr["wr_pct"] = (state_wr["wr"] * 100).round(2)

    gap = float(state_wr["wr_pct"].max() - state_wr["wr_pct"].min())

    return {
        "version": version,
        "gap": gap,
        "n_matched": len(merged),
        "states": state_wr[["n","wr_pct"]].to_dict(),
    }


# ─── Run sweep ────────────────────────────────────────────────────────────────

VERSIONS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
results = []

print(SEP)
print("SWEEP HMM FEATURE SETS")
print(SEP)

for v in VERSIONS:
    desc = {
        "V0": "baseline (ret, vol_std, mom_48, vol_ratio)",
        "V1": "V0 + btc_ret_h4",
        "V2": "V0 + btc_ret + btc_mom",
        "V3": "Parkinson vol + ret + mom + btc_ret",
        "V4": "V0 + universe_index",
        "V5": "V0 + universe + corr_dispersion",
        "V6": "V5 + btc_ret",
    }[v]
    print(f"\n[{v}] {desc}")
    res = evaluate_version(v)
    results.append(res)

    if res["gap"] > 0:
        print(f"  WR gap : {res['gap']:.2f}%  |  matched bars: {res.get('n_matched',0):,}")
        if "states" in res and res["states"]:
            wr_col = res["states"].get("wr_pct", {})
            n_col  = res["states"].get("n", {})
            for state in sorted(wr_col.keys()):
                print(f"    {state:22}: WR={wr_col[state]:.1f}%  n={n_col.get(state,0):,}")
    else:
        print(f"  GAGAL atau data tidak cukup")


# ─── Rangkuman ────────────────────────────────────────────────────────────────

print()
print(SEP)
print("RANGKUMAN — RANKING FEATURE SETS")
print(SEP)

valid = [(r["version"], r["gap"]) for r in results if r["gap"] > 0]
valid.sort(key=lambda x: x[1], reverse=True)

for rank, (ver, gap) in enumerate(valid, 1):
    tag = " << TERBAIK" if rank == 1 else (" >> BASELINE" if ver == "V0" else "")
    print(f"  #{rank}  {ver}: gap={gap:.2f}%{tag}")

print()
best = valid[0][0] if valid else "V0"
best_gap = valid[0][1] if valid else 0
if best_gap >= 5.0:
    print(f"REKOMENDASI: Pakai {best} (gap {best_gap:.1f}% >= threshold 5%)")
    print("  Langkah: update core/regime.py dengan feature set ini")
    print("  lalu re-run: 03e_regime_hmm -> 04_train_lgbm -> 06_train_guardian")
elif best_gap >= 2.0:
    print(f"MARGINAL: {best} gap {best_gap:.1f}% — ada perbaikan tapi kecil")
    print("  Pertimbangkan sebelum commit ke full retrain")
else:
    print("KESIMPULAN: Semua feature set tidak informatif (<2% gap)")
    print("  Pertimbangkan arsitektur berbeda (global HMM, atau hapus HMM sama sekali)")
