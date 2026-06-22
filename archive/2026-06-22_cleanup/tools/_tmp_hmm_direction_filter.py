"""
Cek apakah HMM state berguna sebagai direction filter.

Hipotesis:
  TRENDING_DOWN  -> WR SHORT > WR LONG di state itu
  TRENDING_UP    -> WR LONG  > WR SHORT di state itu

Jika delta WR cukup besar (>= 3%), HMM bisa dipakai untuk blok
trade yang berlawanan arah regime.

Test semua 7 feature set dari experiment sebelumnya.
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

PROC_DIR  = Path("data/training/processed")
OOF_PATH  = Path("models/runs/ic32_regime_v2/oof_predictions.parquet")
LABEL_DIR = Path("data/training/labeled")
N_STATES  = 4
SEP = "=" * 65

# ─── Load data (sama dengan experiment sebelumnya) ────────────────────────────

def load_h4(coin):
    fp = PROC_DIR / f"{coin}_clean.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp).rename(columns={
        "1h_open":"open","1h_high":"high","1h_low":"low",
        "1h_close":"close","1h_volume":"volume"
    })
    if "close" not in df.columns:
        return None
    return df[["open","high","low","close","volume"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()

print("Loading data...")
h4_cache = {c: h for c in TRAINING_COINS if (h := load_h4(c)) is not None and len(h) >= 200}
btc_h4   = h4_cache["BTCUSDT"]
btc_ret  = btc_h4["close"].pct_change().fillna(0)
btc_mom  = (btc_h4["close"] / btc_h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)

all_rets = pd.concat([h["close"].pct_change().rename(c) for c, h in h4_cache.items()], axis=1).fillna(0)
universe_ret = all_rets.mean(axis=1)

oof = pd.read_parquet(OOF_PATH)
oof_valid = oof[oof["has_oof"]].copy()
oof_valid["pred"] = oof_valid[["p0","p1","p2"]].values.argmax(axis=1)
oof_valid["correct"] = (oof_valid["pred"] == oof_valid["swing_label"]).astype(int)
# pred: 0=SHORT, 1=FLAT, 2=LONG
directional = oof_valid[oof_valid["pred"] != 1].copy()
directional["pred_dir"] = directional["pred"].map({0:"SHORT", 2:"LONG"})

coin_index = {}
for coin in TRAINING_COINS:
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    if fp.exists():
        coin_index[coin] = pd.read_parquet(fp, columns=[]).index

print(f"  H4: {len(h4_cache)} koin | OOF directional: {len(directional):,}\n")


# ─── Feature builders (ringkas) ───────────────────────────────────────────────

def parkinson_vol(h4, window=24):
    log_hl = np.log(h4["high"] / h4["low"])
    pk = (log_hl**2 / (4*np.log(2))).rolling(window, min_periods=4).mean().apply(np.sqrt)
    return pk.fillna(log_hl.rolling(window).std().fillna(log_hl.std()))

def build_X(coin, version):
    h4  = h4_cache.get(coin)
    if h4 is None: return None
    ret = h4["close"].pct_change().fillna(0)
    vol = ret.rolling(24, min_periods=4).std().fillna(ret.std())
    mom = (h4["close"] / h4["close"].rolling(48, min_periods=8).mean() - 1).fillna(0)
    vr  = np.log((h4["volume"]/h4["volume"].rolling(48,min_periods=8).mean()).clip(0.01,20)).fillna(0)
    def align(s): return s.reindex(h4.index).ffill().fillna(0)
    b_r = align(btc_ret); b_m = align(btc_mom); u_r = align(universe_ret)
    pk  = parkinson_vol(h4)

    cols = {"V0":[ret,vol,mom,vr], "V1":[ret,vol,mom,vr,b_r],
            "V2":[ret,vol,mom,vr,b_r,b_m], "V3":[ret,pk,mom,b_r],
            "V4":[ret,vol,mom,vr,u_r], "V5":[ret,vol,mom,vr,u_r],
            "V6":[ret,vol,mom,vr,u_r,b_r]}.get(version)
    if cols is None: return None
    return np.nan_to_num(np.column_stack([c.values for c in cols]), nan=0.0, posinf=0.0, neginf=0.0)


def get_state_labels(version):
    """Fit HMM per koin, return DataFrame index->state."""
    records = []
    for coin in TRAINING_COINS:
        if coin not in coin_index or coin not in h4_cache: continue
        X = build_X(coin, version)
        if X is None or len(X) < 200: continue
        try:
            m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=100, random_state=42, tol=1e-3, verbose=False)
            m.fit(X)
            sm = _align_states(m, N_STATES)
            labels = pd.Series([sm[int(s)] for s in m.predict(X)], index=h4_cache[coin].index)
            h1 = labels.reindex(coin_index[coin]).ffill().dropna()
            records.append(pd.DataFrame({"state": h1, "coin": coin}))
        except Exception:
            continue
    return pd.concat(records)["state"] if records else pd.Series(dtype=str)


# ─── Evaluasi direction filter ────────────────────────────────────────────────

def eval_direction_filter(version):
    states = get_state_labels(version)
    if len(states) == 0:
        return None

    merged = directional.join(states.rename("state"), how="left").dropna(subset=["state"])
    if len(merged) < 500:
        return None

    # WR per (state, direction)
    grp = merged.groupby(["state","pred_dir"])["correct"].agg(["mean","count"]).reset_index()
    grp.columns = ["state","dir","wr","n"]
    grp["wr_pct"] = (grp["wr"]*100).round(2)

    # Untuk tiap state: delta WR = WR_aligned_dir - WR_counter_dir
    # "aligned" = SHORT di TRENDING_DOWN, LONG di TRENDING_UP
    alignment = {
        "TRENDING_DOWN":    ("SHORT","LONG"),
        "TRENDING_UP":      ("LONG","SHORT"),
        "RANGING_LOW_VOL":  (None, None),
        "RANGING_HIGH_VOL": (None, None),
    }

    deltas = {}
    rows = {}
    for state, (favored, opposed) in alignment.items():
        sub = grp[grp["state"]==state]
        if sub.empty: continue
        wr_by_dir = sub.set_index("dir")["wr_pct"]
        rows[state] = wr_by_dir.to_dict()
        if favored and opposed and favored in wr_by_dir and opposed in wr_by_dir:
            deltas[state] = wr_by_dir[favored] - wr_by_dir[opposed]

    return {"version": version, "rows": rows, "deltas": deltas, "grp": grp}


print(SEP)
print("DIRECTION FILTER TEST — WR per (state, direction)")
print(SEP)

VERSIONS = ["V0","V1","V2","V3","V4","V5","V6"]
all_results = []

for v in VERSIONS:
    res = eval_direction_filter(v)
    if res is None:
        print(f"[{v}] SKIP — data tidak cukup\n")
        continue
    all_results.append(res)

    print(f"\n[{v}]")
    print(f"  {'State':22} {'LONG WR':>9} {'SHORT WR':>10} {'Delta (fav-opp)':>16}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*16}")
    for state in REGIME_NAMES_4:
        row = res["rows"].get(state, {})
        wr_l = row.get("LONG",  float("nan"))
        wr_s = row.get("SHORT", float("nan"))
        delta = res["deltas"].get(state, float("nan"))
        delta_str = f"{delta:+.2f}%" if not np.isnan(delta) else "   n/a"
        print(f"  {state:22} {wr_l:>8.1f}% {wr_s:>9.1f}% {delta_str:>16}")

    if res["deltas"]:
        max_delta = max(res["deltas"].values())
        tag = " << FILTER BERGUNA" if max_delta >= 3.0 else (" ~ marginal" if max_delta >= 1.5 else " << tidak ada sinyal")
        print(f"  Max delta: {max_delta:+.2f}%{tag}")


# ─── Rangkuman ────────────────────────────────────────────────────────────────

print()
print(SEP)
print("RANGKUMAN — MAX DELTA WR (favored vs opposed direction)")
print(SEP)

ranked = sorted(
    [(r["version"], max(r["deltas"].values()) if r["deltas"] else 0) for r in all_results],
    key=lambda x: x[1], reverse=True
)
for rank, (ver, delta) in enumerate(ranked, 1):
    tag = " << TERBAIK" if rank == 1 else (" >> BASELINE" if ver == "V0" else "")
    print(f"  #{rank}  {ver}: max_delta={delta:+.2f}%{tag}")

print()
best_ver, best_delta = ranked[0] if ranked else ("V0", 0)
if best_delta >= 3.0:
    print(f"REKOMENDASI: HMM {best_ver} sebagai direction filter LAYAK dipakai")
    print(f"  Implementasi: blok SHORT jika state=TRENDING_UP, blok LONG jika state=TRENDING_DOWN")
elif best_delta >= 1.5:
    print(f"MARGINAL: Delta {best_delta:.2f}% — mungkin worth dicoba tapi dampak kecil")
else:
    print("KESIMPULAN: HMM tidak berguna sebagai direction filter (delta < 1.5%)")
    print("  Rekomendasi: hapus hmm_regime_enc dari LGBM features, ganti dengan")
    print("  fitur cross-asset langsung (btc_ret_h4, universe_index) sebagai raw features")
