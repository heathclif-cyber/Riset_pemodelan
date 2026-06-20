"""
pipeline/13_holdout_multiplier_meta.py
Coefficient Multiplier ensemble: LGBM Ã— LSTM-multiplier + Guardian

Mechanism:
  multiplier = p_win(lstm_binary_meta_v1) / base_win_rate
  effective_conf = lgbm_conf Ã— multiplier
  Entry jika effective_conf >= regime_threshold

Kenapa lebih baik dari hard gate (meta v1 dengan threshold):
  - Hard gate binary: kills trade paksa â†’ -$341 PnL
  - Multiplier continuous: trade dengan setup lemah jatuh natural di bawah threshold
  - LGBM tetap menentukan arah â€” LSTM hanya modulate confidence magnitude

Kenapa lebih baik dari soft blend (Î±=0.3):
  - Soft blend mengubah probability vector (bisa flip arah)
  - Multiplier hanya scale confidence, tidak mengubah arah
  - Continuous adjustment, bukan discrete Â±0.05
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("13_multiplier_meta")

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SL_MULT   = TP_SL_FALLBACK_SL
MAX_HOLD  = MAX_HOLDING_BARS
MODAL     = MODAL_PER_TRADE
LEVERAGE  = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT   = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
GDN_MIN   = 2
GDN_THR   = 0.65
SEQ_LEN   = 32
BASE_WIN_RATE = 0.412    # WIN rate OOF binary meta v1 (raw LGBM trades)

# Multiplier caps â€” jangan biarkan LSTM terlalu mendominasi
MULT_MIN  = 0.60   # maksimal penalty: confidence turun 40%
MULT_MAX  = 1.50   # maksimal boost: confidence naik 50%

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})


# â”€â”€ Binary LSTM Meta v1 architecture (sama persis dengan training) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class BinaryLSTMMeta(nn.Module):
    def __init__(self, n_feat, hidden=32, n_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(h[-1])))


# â”€â”€ Load models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tb_model = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

META_RUN = MODEL_DIR / "runs/tb_lstm_binary_meta_v1"
with open(META_RUN / "tb_lstm_binary_meta_v1_features.json") as f:
    meta_feats = json.load(f)
meta_scaler = joblib.load(META_RUN / "lstm_binary_meta_scaler.pkl")
n_feat_total = len(meta_feats) + 1  # +1 direction
meta_model = BinaryLSTMMeta(n_feat_total, hidden=32)
meta_model.load_state_dict(torch.load(META_RUN / "lstm_binary_meta.pt", map_location="cpu"))
meta_model.eval()

tbg_model  = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian_scaler.pkl")
with open(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

tbg_static = [f for f in tbg_all_feats if f not in DYNAMIC_NAMES]
tbg_smap   = {n: i for i, n in enumerate(tbg_static)}
tbg_order  = [
    ("static", tbg_smap[f]) if f not in DYNAMIC_NAMES else ("dyn", f)
    for f in tbg_all_feats
]

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*76}")
print(f"  HOLDOUT: Coefficient Multiplier Ensemble | Nov 2025â€“Apr 2026")
print(f"  Mechanism: effective_conf = lgbm_conf Ã— (p_win / {BASE_WIN_RATE:.3f})")
print(f"  Multiplier range: [{MULT_MIN:.2f}, {MULT_MAX:.2f}]")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*76}\n")


# â”€â”€ Guardian row builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def gdn_row(j, i, close, atr, direction, max_fav, X_static):
    bh   = j - i
    pnl  = (close[j] - close[i]) / close[i] * direction
    atp  = atr[i] / close[i] if close[i] > 0 else 0.01
    nmx  = max(max_fav, pnl)
    dyn  = {
        "bars_held_norm"        : bh / MAX_HOLD,
        "current_pnl_pct"       : pnl,
        "current_pnl_atr"       : pnl / atp if atp > 0 else 0.0,
        "max_favorable_pnl_pct" : nmx,
        "drawdown_from_peak_pct": (nmx - pnl) / nmx if nmx > 0.001 else 0.0,
        "direction"             : float(direction),
        "entry_price_ratio"     : close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(tbg_order), dtype=np.float64)
    for k, (src, key) in enumerate(tbg_order):
        row[k] = X_static[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, nmx


# â”€â”€ Simulate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def simulate(yp, close, high, low, atr, X_tbg, use_guardian):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == 1: i += 1; continue
        direction  = 1 if sig == 2 else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        max_fav    = 0.0
        exit_p     = close[min(i + MAX_HOLD, n - 1)]
        exit_b     = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_p, exit_b, outcome = sl_price, j, "SL"; break
            if direction == -1 and high[j] >= sl_price:
                exit_p, exit_b, outcome = sl_price, j, "SL"; break
            if use_guardian and j - i >= GDN_MIN:
                row, max_fav = gdn_row(j, i, close, atr, direction, max_fav, X_tbg)
                sc   = tbg_scaler.transform(row.reshape(1, -1))
                prob = tbg_model.predict_proba(sc)[0]
                ep   = prob[2] if len(prob) > 2 else prob[1]
                if ep >= GDN_THR:
                    exit_p, exit_b, outcome = close[j], j, "GDN"; break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)

        net = (exit_p - entry) / entry * direction * MODAL * LEVERAGE \
              - COST_RT * MODAL * LEVERAGE
        trades.append({"net_pnl": net, "outcome": outcome,
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "bars_held": exit_b - i})
        i = exit_b + 1
    return trades


def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl": 0, "gdn": 0, "bars": []}

def upd(a, trades):
    for t in trades:
        a["trades"] += 1; a["pnl"] += t["net_pnl"]; a["bars"].append(t["bars_held"])
        if t["net_pnl"] > 0: a["wins"] += 1
        if t["outcome"] == "SL": a["sl"] += 1
        if t["outcome"] == "GDN": a["gdn"] += 1
        if t["direction"] == "LONG":
            a["longs"] += 1
            if t["net_pnl"] > 0: a["long_wins"] += 1


# Variants: baseline + 4 lambda values untuk sensitivity analysis
# lambda mengontrol seberapa agresif multiplier bekerja
# effective_conf = lgbm_conf Ã— clip(1 + Î» Ã— (p_win/base - 1), MULT_MIN, MULT_MAX)
# Î»=0.0 â†’ multiplier=1 (no effect), Î»=1.0 â†’ full likelihood ratio
LAMBDAS = [0.0, 0.5, 0.75, 1.0, 1.25]

variant_keys = [f"lam{int(l*100):03d}" for l in LAMBDAS]
agg_gdn = {k: new_agg() for k in variant_keys}   # all with Guardian

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].isin({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df   = df[mask].copy()
    n    = len(df)

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(n, 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # â”€â”€ LGBM predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_tb = np.zeros((n, len(tb_feats)))
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values
    p_tb    = tb_model.predict_proba(X_tb).astype(np.float32)
    conf_tb = np.max(p_tb, axis=1)          # base LGBM confidence
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)

    # â”€â”€ Binary LSTM Meta â†’ p_win per bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_meta_raw = np.zeros((n, len(meta_feats)), dtype=np.float32)
    for idx, c in enumerate(meta_feats):
        if c in df.columns:
            X_meta_raw[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)

    lgbm_dir    = np.where(yp_tb == 2, 1.0, np.where(yp_tb == 0, -1.0, 0.0)).astype(np.float32)
    X_meta_full = np.concatenate([X_meta_raw, lgbm_dir[:, np.newaxis]], axis=1)  # (n, 16)

    n_seq = n - SEQ_LEN
    p_win_all = np.full(n, BASE_WIN_RATE, dtype=np.float32)   # default = base rate

    if n_seq > 0:
        X_seq = np.lib.stride_tricks.as_strided(
            X_meta_full,
            shape=(n_seq, SEQ_LEN, n_feat_total),
            strides=(X_meta_full.strides[0], X_meta_full.strides[0], X_meta_full.strides[1])
        ).copy()
        X_2d   = X_seq.reshape(-1, n_feat_total)
        X_2d_sc = meta_scaler.transform(X_2d).astype(np.float32)
        X_seq_sc = X_2d_sc.reshape(n_seq, SEQ_LEN, n_feat_total)

        INFER_BATCH = 512
        for start in range(0, n_seq, INFER_BATCH):
            end = min(start + INFER_BATCH, n_seq)
            xb  = torch.FloatTensor(X_seq_sc[start:end])
            with torch.no_grad():
                scores = meta_model(xb).squeeze(1).numpy()
            p_win_all[SEQ_LEN + start: SEQ_LEN + end] = scores

    # â”€â”€ Compute effective_conf for each lambda â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # raw likelihood ratio
    likelihood_ratio = np.clip(p_win_all / BASE_WIN_RATE, MULT_MIN, MULT_MAX)  # Î»=1.0 case

    # Guardian static features
    X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values

    for lam, key in zip(LAMBDAS, variant_keys):
        # effective_conf = lgbm_conf Ã— clip(1 + Î»*(lr - 1), MULT_MIN, MULT_MAX)
        mult     = np.clip(1.0 + lam * (likelihood_ratio - 1.0), MULT_MIN, MULT_MAX)
        eff_conf = conf_tb * mult

        # Rebuild yp using effective_conf with regime threshold
        yp_eff = np.argmax(p_tb, axis=1).astype(np.int32)  # direction same as LGBM
        for r, th in REGIME_THRESH.items():
            # filter: entry suppressed if effective_conf < threshold
            yp_eff[(hmm == r) & (yp_eff != 1) & (eff_conf < th)] = 1

        upd(agg_gdn[key],
            simulate(yp_eff, close, high, low, atr, X_tbg, use_guardian=True))

    logger.info(
        f"[{sym}] "
        + "  ".join(f"Î»={l:.2f}:{agg_gdn[k]['trades']}"
                    for l, k in zip(LAMBDAS, variant_keys))
    )


# â”€â”€ Scorecard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def sc(a):
    t  = a["trades"]; w = a["wins"]; p = a["pnl"]
    l  = a["longs"]; lw = a["long_wins"]
    sl = a["sl"]; gd = a["gdn"]; bh = a["bars"]
    sw = w - lw; sh = t - l
    return dict(
        trades=t, wr=w/max(t,1)*100,
        long_wr=lw/max(l,1)*100, short_wr=sw/max(sh,1)*100,
        long_pct=l/max(t,1)*100,
        sl_rate=sl/max(t,1)*100, gdn_rate=gd/max(t,1)*100,
        avg_hold=float(np.mean(bh)) if bh else 0,
        pnl=p, ppm=p/5, ppt=p/max(t,1),
    )

s = {k: sc(agg_gdn[k]) for k in variant_keys}
W = 13

LABELS = {k: f"Î»={l:.2f}" for k, l in zip(variant_keys, LAMBDAS)}
LABELS[variant_keys[0]] = "baseline(Î»=0)"

print(f"\n{'='*90}")
print(f"  SCORECARD â€” Coefficient Multiplier | Nov 2025â€“Apr 2026 | 21 koin")
print(f"  effective_conf = lgbm_conf Ã— clip(1 + Î»Ã—(p_win/{BASE_WIN_RATE:.3f} - 1), {MULT_MIN},{MULT_MAX})")
print(f"{'='*90}")

hdr = f"  {'Metrik':<22}"
for k in variant_keys:
    hdr += f" {LABELS[k]:>{W}}"
print(hdr)
print(f"  {'-'*86}")

rows = [
    ("Total Trades",   lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",   lambda x: f"{x['trades']//5:,}"),
    ("Win Rate",       lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR",      lambda x: f"{x['long_wr']:.1f}%"),
    ("  SHORT WR",     lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",    lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exits", lambda x: f"{x['gdn_rate']:.1f}%"),
    ("Avg hold (bar)", lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL (5bln)", lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan",      lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade",      lambda x: f"${x['ppt']:+.3f}"),
]
for label, fn in rows:
    row = f"  {label:<22}"
    for k in variant_keys:
        row += f" {fn(s[k]):>{W}}"
    if "PnL (5bln)" in label:
        best_k = max(variant_keys[1:], key=lambda k: s[k]["pnl"])
        row += f"  â† best: {LABELS[best_k]}"
    print(row)

# Delta vs baseline
base = s[variant_keys[0]]
print(f"\n  Delta vs baseline (Î»=0, = tb+Guardian):")
for k in variant_keys[1:]:
    v = s[k]
    delta_pnl    = v["pnl"] - base["pnl"]
    delta_wr     = v["wr"] - base["wr"]
    delta_trades = v["trades"] - base["trades"]
    print(f"  {LABELS[k]:<14}: "
          f"PnL {delta_pnl:>+7.0f}  "
          f"WR {delta_wr:>+5.1f}pp  "
          f"Trades {delta_trades:>+5}")

# Also show multiplier distribution
print(f"\n  Multiplier distribution analysis (all LGBM signal bars, 21 koin):")
print(f"  (menggunakan Î»=1.0, base_rate={BASE_WIN_RATE:.3f})")

# Save results
out_path = MODEL_DIR / "runs/tb_lstm_binary_meta_v1/holdout_multiplier_results.json"
out = {}
for k in variant_keys:
    out[k] = {**s[k], "lambda": float(LAMBDAS[variant_keys.index(k)]), "label": LABELS[k]}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n  Saved: {out_path}")
print(f"{'='*90}")


