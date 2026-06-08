"""
Test: LSTM Daily Bias Filter in Cascade
Compares BASELINE vs LSTM DAILY BIAS on holdout Apr-Jun 2026
"""
import sys, json, joblib, torch, torch.nn as nn, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
COINANK_DIR = ROOT / "data" / "coinank"
MACRO_DIR = ROOT / "data" / "macro"

# ── Load LSTM Daily model ────────────────────────────────────────────────
lstm_daily_dir = MODEL_DIR / "runs" / "lstm_daily_v1"
lstm_daily_feats = json.load(open(lstm_daily_dir / "feature_cols.json"))
lstm_daily_scaler = joblib.load(lstm_daily_dir / "scaler.pkl")

class DailyLSTM(nn.Module):
    def __init__(self, nf):
        super().__init__()
        # Directly define BiLSTM here to avoid import issues
        from core.models import _ManualLSTMCell
        class BiLSTM(nn.Module):
            def __init__(self, inp, hid, layers, dropout):
                super().__init__()
                self.hid = hid; self.layers = layers
                self.fwd = nn.ModuleList([_ManualLSTMCell(inp if i==0 else hid, hid) for i in range(layers)])
                self.bwd = nn.ModuleList([_ManualLSTMCell(inp if i==0 else hid, hid) for i in range(layers)])
                self.drop = nn.Dropout(dropout)
                self.ln_f = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
                self.ln_b = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
            def _go(self, x, cells, lns):
                B, T, _ = x.shape; dev = x.device
                h = [torch.zeros(B, self.hid, device=dev) for _ in cells]
                c = [torch.zeros(B, self.hid, device=dev) for _ in cells]
                out = []
                for t in range(T):
                    inp = x[:, t, :]
                    for i, cell in enumerate(cells):
                        h[i], c[i] = cell(inp, (h[i], c[i]))
                        inp = lns[i](h[i])
                        if i < len(cells)-1: inp = self.drop(inp)
                    out.append(inp)
                return torch.stack(out, dim=1)
            def forward(self, x):
                f = self._go(x, self.fwd, self.ln_f)
                b = self._go(torch.flip(x, [1]), self.bwd, self.ln_b)
                return torch.cat([f, torch.flip(b, [1])], dim=-1)

        self.bilstm = BiLSTM(nf, 96, 2, 0.40)
        self.ln = nn.LayerNorm(192)
        self.drop = nn.Dropout(0.40)
        self.fc = nn.Linear(192, 1)

    def forward(self, x):
        out = self.bilstm(x)
        return torch.sigmoid(self.fc(self.drop(self.ln(out[:, -1, :])))).squeeze(-1)

n_daily_feat = len(lstm_daily_feats)
lstm_daily = DailyLSTM(n_daily_feat)
lstm_daily.load_state_dict(torch.load(str(lstm_daily_dir / "model.pt"), map_location="cpu"))
lstm_daily.eval()

# ── Load other models ─────────────────────────────────────────────────────
lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
guard = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
from core.models import load_lstm
lstm_h1 = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
btu.SMART_ENTRY_MODE = "disabled"

# ── Pre-compute LSTM Daily predictions for holdout period ────────────────
def build_daily_data(coins, pos_data, macro):
    """Build daily features for LSTM Daily inference."""
    all_daily = {}

    for coin in coins:
        fp = HOLDOUT_LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp).sort_index()

        daily = df[["open","high","low","close","volume"]].resample("1D").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

        # OHLCV daily features
        daily["ret_1d"] = daily["close"].pct_change(1)
        daily["ret_5d"] = daily["close"].pct_change(5)
        daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"]
        daily["vol_chg"] = daily["volume"].pct_change(1)
        atr = (daily["high"] - daily["low"]).rolling(14).mean()
        daily["atr_14d"] = atr / daily["close"]

        # Positioning
        if coin in pos_data:
            pos = pos_data[coin]
            date_idx = pd.to_datetime(daily.index.date, utc=True)
            for col in ["oi_z20","oi_d7","ls_z20","ls_d7","smart_retail"]:
                if col in pos.columns:
                    s = pos[col].copy()
                    s.index = pd.to_datetime(s.index.date, utc=True)
                    s = s[~s.index.duplicated(keep="last")]
                    daily[col] = s.reindex(date_idx).ffill().fillna(0).values
                    daily[f"{col}_avl"] = (daily[col] != 0).astype(float)
                else:
                    daily[col] = 0.0; daily[f"{col}_avl"] = 0.0
        else:
            for col in ["oi_z20","oi_d7","ls_z20","ls_d7","smart_retail"]:
                daily[col] = 0.0; daily[f"{col}_avl"] = 0.0

        # Global features
        date_idx = pd.to_datetime(daily.index.date, utc=True)
        if "fear_greed" in macro and macro["fear_greed"] is not None:
            fg = macro["fear_greed"].copy()
            fg.index = pd.to_datetime(fg.index.date, utc=True)
            fg = fg[~fg.index.duplicated(keep="last")]
            daily["fear_greed"] = (fg["fear_greed_value"].reindex(date_idx).ffill().fillna(50) - 50) / 25
        else: daily["fear_greed"] = 0.0

        if "etf_btc" in macro and macro["etf_btc"] is not None:
            etf = macro["etf_btc"].copy()
            etf.index = pd.to_datetime(etf.index.date, utc=True)
            etf = etf[~etf.index.duplicated(keep="last")]
            daily["etf_btc_d5"] = etf["btc_etf_volume_usd"].reindex(date_idx).ffill().fillna(0).pct_change(5).fillna(0).values
        else: daily["etf_btc_d5"] = 0.0

        all_daily[coin] = daily

    return all_daily


def predict_lstm_daily(daily_df, feat_cols):
    """Run LSTM Daily inference on daily dataframe."""
    # Build sequences (32 days, sliding window)
    Xc = daily_df[feat_cols].fillna(0).clip(-10, 10).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    seq_len = 32
    if len(Xc) < seq_len: return np.full(len(Xc), 0.5)

    Xs = []
    for i in range(seq_len - 1, len(Xc)):
        Xs.append(Xc[i - seq_len + 1:i + 1])
    Xs = np.stack(Xs)

    # Scale
    n, s, f = Xs.shape
    Xs_s = lstm_daily_scaler.transform(Xs.reshape(-1, f)).reshape(n, s, f).astype(np.float32)

    # Predict
    with torch.no_grad():
        proba = lstm_daily(torch.from_numpy(Xs_s)).cpu().numpy()

    # Align to daily index: first 31 days have no prediction → 0.5
    full = np.full(len(Xc), 0.5)
    full[seq_len - 1:] = proba
    return full


# ── Run holdout ──────────────────────────────────────────────────────────
def run_holdout(coins, all_daily, use_lstm_daily):
    trades = []; n_blocked = 0

    for coin in coins:
        fp = HOLDOUT_LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = HOLDOUT_LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp).sort_index()
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
        if len(df) < 200: continue

        n = len(df)
        X = np.zeros((n, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values

        # Get LSTM Daily predictions for this coin
        lstm_daily_proba = np.full(n, 0.5)
        if use_lstm_daily and coin in all_daily:
            daily = all_daily[coin]
            daily_preds = predict_lstm_daily(daily, lstm_daily_feats)

            # Map daily predictions to hourly bars
            for j in range(n):
                bar_date = pd.Timestamp(df.index[j].date(), tz="UTC")
                # Find closest daily prediction
                daily_idx = daily.index.get_indexer([bar_date], method="ffill")
                if daily_idx[0] >= 0 and daily_idx[0] < len(daily_preds):
                    lstm_daily_proba[j] = daily_preds[daily_idx[0]]

        yp, cf = hierarchical_predict(None, lgbm, lstm_h1, lsc, X, lstm_f, [], df,
                                       trend_alignment_enabled=False, regime_aware_alignment=True)

        # LSTM Daily bias filter
        if use_lstm_daily:
            for j in range(n):
                if yp[j] == 1: continue  # FLAT
                p_bull = lstm_daily_proba[j]
                hmm = int(df["hmm_regime_enc"].iloc[j])

                # Only filter in TRENDING regime
                if hmm in (0, 3):
                    if yp[j] == 2 and p_bull < 0.35:  # LGBM LONG + LSTM BEARISH → block
                        yp[j] = 1; cf[j] = 0.0; n_blocked += 1
                    elif yp[j] == 0 and p_bull > 0.65:  # LGBM SHORT + LSTM BULLISH → block
                        yp[j] = 1; cf[j] = 0.0; n_blocked += 1

        below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1

        Xg = compute_guardian_static_array(df, gst)
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        c = df["close"].values; h = df["high"].values if "high" in df.columns else c
        l = df["low"].values if "low" in df.columns else c
        sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)

        r = simulate_trades_swing(
            y_pred=yp, close=c, high=h, low=l, atr=atr, h4_swing_highs=sh, h4_swing_lows=sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
            min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=cf, guardian_enabled=True,
            guardian_model=guard, guardian_scaler=gs,
            X_guardian=Xg, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=2)
        for t in r.get("trades", []):
            t["coin"] = coin; t["timestamp"] = df.index[t.get("bar_in", 0)]
        trades.extend(r.get("trades", []))

    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl,
                pf=gw/gl if gl else 0, blocked=n_blocked)


# ── Load positioning + macro ─────────────────────────────────────────────
pos_data = {}
for coin in TRAINING_COINS:
    oi_p = COINANK_DIR / f"{coin}_oi.parquet"; lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"
    if not oi_p.exists(): continue
    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None
    daily = pd.DataFrame(index=oi.index)
    oi_c = [c for c in oi.columns if c.startswith("oi_") or "oi" in c.lower()]
    if not oi_c: continue
    oi_t = oi[oi_c[0]]
    om = oi_t.rolling(20).mean(); os = oi_t.rolling(20).std().clip(lower=1e-8)
    daily["oi_z20"] = (oi_t - om) / os; daily["oi_d7"] = oi_t.pct_change(7)
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls = lsp["top_trader_position_ls"]
        lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
        daily["ls_z20"] = (ls - lm) / ls_s; daily["ls_d7"] = ls.diff(7)
        if lsa is not None and "top_trader_account_ls" in lsa.columns:
            daily["smart_retail"] = ls - lsa["top_trader_account_ls"]
    pos_data[coin] = daily

macro = {}
fg_p = MACRO_DIR / "fear_greed.parquet"
if fg_p.exists(): macro["fear_greed"] = pd.read_parquet(fg_p)
etf_p = MACRO_DIR / "etf_btc_combined.parquet"
if etf_p.exists(): macro["etf_btc"] = pd.read_parquet(etf_p)

# ── Run ───────────────────────────────────────────────────────────────────
print("Building daily data for LSTM Daily inference...")
all_daily = build_daily_data(TRAINING_COINS[:5], pos_data, macro)
for coin, daily in all_daily.items():
    print(f"  {coin}: {len(daily)} daily bars | {daily.index[0].date()} -> {daily.index[-1].date()}")

print("\nRunning BASELINE..."); b = run_holdout(TRAINING_COINS[:5], all_daily, False)
print("Running LSTM DAILY BIAS..."); d = run_holdout(TRAINING_COINS[:5], all_daily, True)

print(f"""
{'='*60}
  LSTM DAILY BIAS FILTER — Holdout Apr-Jun 2026
  {'Metric':<20} {'BASELINE':>12} {'LSTM DAILY':>12} {'Delta':>10}
  {'-'*55}""")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),("blocked","Blocked by LSTM")]:
    bv = b[k]; dv = d[k]
    if isinstance(bv, float):
        print(f"  {label:<20} {bv:>12.1f} {dv:>12.1f} {dv-bv:>+10.1f}")
    else:
        print(f"  {label:<20} {bv:>12,} {dv:>12,} {dv-bv:>+10,}")
print(f"{'='*60}")
