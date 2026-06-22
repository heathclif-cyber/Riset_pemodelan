"""
pipeline/07_holdout_comparison_all.py
Full comparison semua model — Nov 2025-Apr 2026

Variants:
  1. ic32 bare
  2. ic32 + Guardian clean_v2
  3. TB v3 bare         (tb_lgbm_widyawardhana_v3, 18 feat)
  4. TB v3 + LSTM-C     (FLIP hard veto)
  5. TB v3 + Guardian   (tb_guardian_widyawardhana_v2)
  6. TB v3 + LSTM-C+Gdn
  7. TB SHAP bare       (tb_fs_shap_v1, 61 feat, union IC+MI+Gain+SHAP)
  8. TB SHAP + Guardian

Exit: SL=1.5xATR | max_hold=36bar | NO TP | Guardian=early exit
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib, torch
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

logger = setup_logger("07_comparison_all")

PROD     = Path("D:/Apps-Dev/swint_tradev2/models")
LM       = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SHORT, FLAT, LONG = 0, 1, 2

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

IC32_THR_LONG  = LGBM_THRESHOLD_LONG
IC32_THR_SHORT = LGBM_THRESHOLD_SHORT
TB_THR         = 0.42   # tb_widyawardhana_v3 best sweep threshold
TB_SHAP_THR    = 0.42   # same threshold for tb_fs_shap_v1 (no sweep yet)
REGIME_THRESH  = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

GDN_MIN_HOLD = 2
GDN_EXIT_THR = 0.65
SEQ_LEN      = 16

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})


# ── Derived features (computed per-coin from available parquet columns) ────────
def compute_derived(df):
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    df = df.copy()
    df["candle_body_ratio"]       = body / hl
    df["upper_wick_ratio"]        = (df["high"] - df[["open","close"]].max(axis=1)) / hl
    df["lower_wick_ratio"]        = (df[["open","close"]].min(axis=1) - df["low"]) / hl
    df["candle_range_atr_ratio"]  = hl / df["atr_14_h1"].replace(0, np.nan)
    if "funding_rate" in df.columns:
        df["funding_rate_change_8h"]  = df["funding_rate"].diff(8)
        fr_mean = df["funding_rate"].rolling(480, min_periods=100).mean()
        fr_std  = df["funding_rate"].rolling(480, min_periods=100).std()
        df["funding_rate_zscore_20d"] = (df["funding_rate"] - fr_mean) / fr_std.replace(0, np.nan)
    if "open_interest" in df.columns:
        oi_prev = df["open_interest"].shift(24)
        df["oi_pct_1d"] = (df["open_interest"] - oi_prev) / oi_prev.abs().replace(0, np.nan)
    return df


# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models ...")

ic32_model = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

shap_model = joblib.load(MODEL_DIR / "runs" / "tb_fs_shap_v1" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_fs_shap_v1" / "fs_selected.json") as f:
    shap_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "tb_lstm_widyawardhana_v1_features.json") as f:
    lstm_feats = json.load(f)

# ic32 Guardian
gdn_model  = joblib.load(PROD / "guardian_best.pkl")
gdn_scaler = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    gdn_all_feats = json.load(f)

# TB Guardian v2
tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

print(f"  ic32({len(ic32_feats)}f) | tb_v3({len(tb_feats)}f) | "
      f"tb_shap({len(shap_feats)}f) | lstm({len(lstm_feats)}f) | "
      f"ic32_gdn({len(gdn_all_feats)}f) | tb_gdn({len(tbg_all_feats)}f)")


def make_guardian_config(feat_list):
    static     = [f for f in feat_list if f not in DYNAMIC_NAMES]
    static_map = {name: i for i, name in enumerate(static)}
    order      = [("static", static_map[f]) if f in static_map else ("dyn", f) for f in feat_list]
    return static, order

gdn_static, gdn_order = make_guardian_config(gdn_all_feats)
tbg_static, tbg_order = make_guardian_config(tbg_all_feats)


def build_X(df, feat_list, hmm=None):
    n = len(df)
    X = np.zeros((n, len(feat_list)), dtype=np.float64)
    for idx, c in enumerate(feat_list):
        if c == "hmm_regime_enc" and hmm is not None:
            X[:, idx] = hmm.astype(np.float64)
        elif c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    return X


def lstm_predict_proba(df, feats):
    n = len(df)
    X_raw = np.zeros((n, len(feats)), dtype=np.float32)
    for idx, c in enumerate(feats):
        if c in df.columns:
            X_raw[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
    X_sc  = lstm_scaler.transform(X_raw.reshape(-1, len(feats))).reshape(n, len(feats)).astype(np.float32)
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < SEQ_LEN:
        return probs
    seqs = np.stack([X_sc[i - SEQ_LEN + 1: i + 1] for i in range(SEQ_LEN - 1, n)])
    all_p = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t = torch.from_numpy(seqs[b: b + 512])
            all_p.append(torch.softmax(lstm_model(t), dim=1).cpu().numpy())
    probs[SEQ_LEN - 1:] = np.concatenate(all_p, axis=0)
    return probs


def apply_flip_veto(yp_base, p_lstm):
    yp = yp_base.copy()
    lstm_argmax = np.argmax(p_lstm, axis=1)
    sig_mask    = (yp != FLAT)
    idxs        = np.where(sig_mask)[0]
    lgbm_dir    = yp[idxs]
    opposite    = np.where(lgbm_dir == LONG, SHORT, LONG)
    yp[idxs[lstm_argmax[idxs] == opposite]] = FLAT
    return yp


def build_guardian_row(j, i, close, atr, direction, max_fav, X_static, feat_order):
    bars_held = j - i
    pnl_pct   = (close[j] - close[i]) / close[i] * direction
    atr_pct   = atr[i] / close[i] if close[i] > 0 else 0.01
    new_max   = max(max_fav, pnl_pct)
    dyn = {
        "bars_held_norm"        : bars_held / MAX_HOLD,
        "current_pnl_pct"       : pnl_pct,
        "current_pnl_atr"       : pnl_pct / atr_pct if atr_pct > 0 else 0.0,
        "max_favorable_pnl_pct" : new_max,
        "drawdown_from_peak_pct": (new_max - pnl_pct) / new_max if new_max > 0.001 else 0.0,
        "direction"             : float(direction),
        "entry_price_ratio"     : close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(feat_order), dtype=np.float64)
    for idx, (src, key) in enumerate(feat_order):
        row[idx] = X_static[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr,
             guardian=None, feat_order=None, X_static=None, gdn_scaler=None):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == FLAT:
            i += 1; continue
        direction  = 1 if sig == LONG else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        max_fav    = 0.0
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            sl_hit = (direction == 1 and low[j]  <= sl_price) or \
                     (direction == -1 and high[j] >= sl_price)
            if sl_hit:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break
            if guardian is not None and (j - i) >= GDN_MIN_HOLD:
                row, max_fav = build_guardian_row(
                    j, i, close, atr, direction, max_fav, X_static, feat_order)
                prob = guardian.predict_proba(gdn_scaler.transform(row.reshape(1, -1)))[0]
                exit_p = prob[2] if len(prob) > 2 else prob[1]
                if exit_p >= GDN_EXIT_THR:
                    exit_price, exit_bar, outcome = close[j], j, "GUARDIAN_EXIT"; break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)
        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome"  : outcome,
            "net_pnl"  : net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

keys = ["ic32", "ic32_gdn", "tb", "tb_lstm_c", "tb_gdn", "tb_lstm_c_gdn",
        "tb_shap", "tb_shap_gdn"]

def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0,
            "gdn_exits": 0, "bars_held": []}

agg   = {k: new_agg() for k in keys}
decor = {"flip": 0, "consensus": 0, "neutral": 0, "total": 0}

print(f"\n{'='*95}")
print(f"  HOLDOUT OOS | Apr 2026-Jun 2026 (~2.5 bln) | 8 variants | {len(available)} koin | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*95}")

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    df = compute_derived(df)

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # ── ic32 ──────────────────────────────────────────────────────────────────
    X_ic  = build_X(df, ic32_feats, hmm)
    p_ic  = ic32_model.predict_proba(X_ic)
    yp_ic = np.ones(n, dtype=np.int32)
    yp_ic[p_ic[:, 2] >= IC32_THR_LONG]                          = LONG
    yp_ic[(p_ic[:, 0] >= IC32_THR_SHORT) & (yp_ic != LONG)]     = SHORT

    # ic32 guardian static
    X_gdn = build_X(df, gdn_static, hmm)

    # ── TB v3 ─────────────────────────────────────────────────────────────────
    X_tb    = build_X(df, tb_feats)
    p_tb    = tb_model.predict_proba(X_tb)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != FLAT) & (conf_tb < th)] = FLAT
    # Replace with flat threshold approach (T042)
    yp_tb2 = np.ones(n, dtype=np.int32)
    yp_tb2[p_tb[:, 2] >= TB_THR] = LONG
    yp_tb2[(p_tb[:, 0] >= TB_THR) & (yp_tb2 != LONG)] = SHORT
    yp_tb = yp_tb2

    # ── TB LSTM ───────────────────────────────────────────────────────────────
    p_lstm      = lstm_predict_proba(df, lstm_feats)
    yp_tb_lstmc = apply_flip_veto(yp_tb, p_lstm)

    lstm_argmax = np.argmax(p_lstm, axis=1)
    sig_mask    = (yp_tb != FLAT)
    sig_idxs    = np.where(sig_mask)[0]
    lgbm_dir    = yp_tb[sig_idxs]
    opp_dir     = np.where(lgbm_dir == LONG, SHORT, LONG)
    decor["total"]     += len(sig_idxs)
    decor["flip"]      += int(np.sum(lstm_argmax[sig_idxs] == opp_dir))
    decor["consensus"] += int(np.sum(lstm_argmax[sig_idxs] == lgbm_dir))
    decor["neutral"]   += int(np.sum(lstm_argmax[sig_idxs] == FLAT))

    # TB Guardian static
    X_tbg = build_X(df, tbg_static)

    # ── TB SHAP (61 feat) ─────────────────────────────────────────────────────
    X_shap    = build_X(df, shap_feats)
    p_shap    = shap_model.predict_proba(X_shap)
    yp_shap   = np.ones(n, dtype=np.int32)
    yp_shap[p_shap[:, 2] >= TB_SHAP_THR]                          = LONG
    yp_shap[(p_shap[:, 0] >= TB_SHAP_THR) & (yp_shap != LONG)]   = SHORT

    # ── Simulate all 8 variants ───────────────────────────────────────────────
    variants = [
        ("ic32",          yp_ic,        None,      None,      None,  None),
        ("ic32_gdn",      yp_ic,        gdn_model, gdn_order, X_gdn, gdn_scaler),
        ("tb",            yp_tb,        None,      None,      None,  None),
        ("tb_lstm_c",     yp_tb_lstmc,  None,      None,      None,  None),
        ("tb_gdn",        yp_tb,        tbg_model, tbg_order, X_tbg, tbg_scaler),
        ("tb_lstm_c_gdn", yp_tb_lstmc,  tbg_model, tbg_order, X_tbg, tbg_scaler),
        ("tb_shap",       yp_shap,      None,      None,      None,  None),
        ("tb_shap_gdn",   yp_shap,      tbg_model, tbg_order, X_tbg, tbg_scaler),
    ]
    for key, yp, gdn, f_order, X_st, scaler in variants:
        trades = simulate(yp, close, high, low, atr,
                          guardian=gdn, feat_order=f_order,
                          X_static=X_st, gdn_scaler=scaler)
        a = agg[key]
        for t in trades:
            a["trades"] += 1; a["pnl"] += t["net_pnl"]
            a["bars_held"].append(t["bars_held"])
            if t["net_pnl"] > 0: a["wins"] += 1
            if t["outcome"] == "SL": a["sl_hits"] += 1
            if "GUARDIAN" in t["outcome"]: a["gdn_exits"] += 1
            if t["direction"] == "LONG":
                a["longs"] += 1
                if t["net_pnl"] > 0: a["long_wins"] += 1

    logger.info(
        f"[{sym:>14s}]"
        f" ic32={agg['ic32']['trades']}"
        f" ic32+g={agg['ic32_gdn']['trades']}"
        f" tb={agg['tb']['trades']}"
        f" tb+l={agg['tb_lstm_c']['trades']}"
        f" tb+g={agg['tb_gdn']['trades']}"
        f" tb+l+g={agg['tb_lstm_c_gdn']['trades']}"
        f" shap={agg['tb_shap']['trades']}"
        f" shap+g={agg['tb_shap_gdn']['trades']}"
    )


# ── Decorrelation ─────────────────────────────────────────────────────────────
tot = max(decor["total"], 1)
print(f"\n--- DECORRELATION (TB v3 LSTM vs LGBM) ---")
print(f"  Total signals : {decor['total']:,}")
print(f"  FLIP          : {decor['flip']:,}  ({decor['flip']/tot*100:.1f}%)")
print(f"  CONSENSUS     : {decor['consensus']:,}  ({decor['consensus']/tot*100:.1f}%)")
print(f"  NEUTRAL       : {decor['neutral']:,}  ({decor['neutral']/tot*100:.1f}%)")


# ── Scorecard ─────────────────────────────────────────────────────────────────
def sc(a):
    t = a["trades"]; w = a["wins"]; p = a["pnl"]
    l = a["longs"]; lw = a["long_wins"]; bh = a["bars_held"]
    return dict(
        trades   = t,
        wr       = w / max(t, 1) * 100,
        long_wr  = lw / max(l, 1) * 100,
        short_wr = (w - lw) / max(t - l, 1) * 100,
        long_pct = l / max(t, 1) * 100,
        sl_rate  = a["sl_hits"] / max(t, 1) * 100,
        gdn_rate = a["gdn_exits"] / max(t, 1) * 100,
        avg_hold = np.mean(bh) if bh else 0,
        pnl      = p,
        ppm      = p / 2.5,
        ppt      = p / max(t, 1),
    )

s = {k: sc(agg[k]) for k in keys}

LABELS = {
    "ic32"         : "ic32",
    "ic32_gdn"     : "ic32+Gdn",
    "tb"           : "TB v3",
    "tb_lstm_c"    : "TB+LSTM-C",
    "tb_gdn"       : "TB+Gdn",
    "tb_lstm_c_gdn": "TB+LSTM+Gdn",
    "tb_shap"      : "TB-SHAP",
    "tb_shap_gdn"  : "TB-SHAP+Gdn",
}

W = 13
print(f"\n{'='*115}")
print(f"  SCORECARD | Apr 2026-Jun 2026 (~2.5 bln) | 21 koin | $10/trade 5x leverage")
print(f"{'='*115}")
hdr = f"  {'Metrik':<20}" + "".join(f"{LABELS[k]:>{W}}" for k in keys)
print(hdr)
print(f"  {'-'*20}" + "-" * (W * len(keys)))

rows = [
    ("Trades",         lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",   lambda x: f"{x['trades']/2.5:.0f}"),
    ("Win Rate %",     lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR %",    lambda x: f"{x['long_wr']:.1f}%({x['long_pct']:.0f}%)"),
    ("  SHORT WR %",   lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",    lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exit",  lambda x: f"{x['gdn_rate']:.1f}%" if x['gdn_rate'] > 0 else "-"),
    ("Avg hold (bar)", lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL $",      lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan $",    lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade $",    lambda x: f"${x['ppt']:+.3f}"),
]
for label, fn in rows:
    row = f"  {label:<20}" + "".join(f"{fn(s[k]):>{W}}" for k in keys)
    print(row)

best_k  = max(s, key=lambda k: s[k]["pnl"])
best_wk = max(s, key=lambda k: s[k]["wr"])
print(f"\n  Best PnL : {LABELS[best_k]:<16} = ${s[best_k]['pnl']:+.0f}")
print(f"  Best WR  : {LABELS[best_wk]:<16} = {s[best_wk]['wr']:.1f}%")

# ── Save ───────────────────────────────────────────────────────────────────────
out = {k: {m: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
           for m, v in s[k].items() if m != "bars_held"}
       for k in keys}
out["_meta"] = {
    "period": "Apr2026-Jun2026",
    "coins": len(available),
    "models": {
        "ic32": "ic32_regime_v1 (33 feat)",
        "tb_v3": f"tb_lgbm_widyawardhana_v3 ({len(tb_feats)} feat)",
        "tb_shap": f"tb_fs_shap_v1 ({len(shap_feats)} feat, union IC+MI+Gain+SHAP)",
        "guardian_ic32": "ic32_guardian_clean_v2",
        "guardian_tb": "tb_guardian_widyawardhana_v2",
        "lstm": "tb_lstm_widyawardhana_v1 (FLIP veto)",
    },
    "thresholds": {
        "ic32_long": IC32_THR_LONG, "ic32_short": IC32_THR_SHORT,
        "tb_v3": TB_THR, "tb_shap": TB_SHAP_THR,
    },
    "decorrelation": {k: round(v/tot*100, 1) for k, v in decor.items() if k != "total"},
}
out_path = MODEL_DIR / "runs" / "tb_fs_shap_v1" / "holdout_comparison_all.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*115}")
