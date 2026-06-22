"""
pipeline/12_train_lstm_meta_guardian.py
Binary LSTM Meta v2 — Simon Methodology + Guardian-OOF Labels

Perbaikan dari v1:
  1. Target = Guardian-enhanced WIN/LOSS (bukan raw LGBM outcome)
     - v1 dilatih pada WIN=41%, diterapkan ke Guardian trades (WR=55%) -> mismatch
     - v2 simulasi Guardian pada setiap OOF trade -> label aligned dengan evaluasi
  2. IC test ulang vs Guardian-WIN target (feature yang relevan bisa berbeda)
  3. Hanya fitur yang lulus IC digunakan (Simon Step 2-3)
  4. PATIENCE lebih besar (20 vs 12) + LR lebih rendah

Simon Pipeline:
  Step 1  Signal  : Guardian-adjusted WIN/LOSS per OOF trade
  Step 2  Valid   : IC test tiap fitur vs Guardian-WIN (standalone IC + t-stat)
  Step 3  Indep   : Marginal IC (residualisasi Gram-Schmidt) -> anti-redundancy
  Step 4  Simple  : LSTM hidden=32, 1 layer (justified: ~22K samples, 6K param)
  Step 6  Gate    : Marginal IC(lstm_oof | lgbm_conf) -> PASS / FAIL
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("12_lstm_meta_gdn")

# ── Run config ─────────────────────────────────────────────────────────────────
RUN_NAME  = "tb_lstm_meta_guardian_v1"
RUN_DIR   = MODEL_DIR / "runs" / RUN_NAME

# Simon IC thresholds (trade-level N, bukan bar-level — tidak perlu /24 correction)
IC_MIN   = 0.015   # standalone |IC| minimum
T_MIN    = 1.5     # |t-stat| minimum
MARG_MIN = 0.005   # marginal IC minimum (feature-feature independence)

# LSTM v2 improvements
SEQ_LEN    = 32
HIDDEN     = 32
N_LAYERS   = 1
DROPOUT    = 0.30   # v1=0.50, diperkecil: 22K samples > model size
LR         = 5e-4   # v1=1e-3, lebih lambat untuk stabilitas
EPOCHS     = 100    # v1=80
PATIENCE   = 20     # v1=12, lebih sabar
BATCH      = 128
N_FOLDS    = 8
PURGE_DAYS = 3

# Guardian simulation params (sama dengan evaluasi)
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
GDN_MIN  = 2
GDN_THR  = 0.65
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

# Candidate features — semua 15 dari v1 + beberapa tambahan
# IC test akan memfilter yang tidak relevan terhadap Guardian-WIN
CANDIDATES = [
    "atr_zscore_20d", "atr_percent_h4", "atr_percentile_h1",
    "ofi_h4_delta", "ofi_raw", "cvd_slope_h4", "cvd_momentum_adv",
    "rsi_6", "rsi_14", "rsi_h4",
    "log_ret_1", "log_ret_5", "log_ret_12",
    "swing_momentum", "trend_strength", "ema_50_slope_h4",
    "funding_rate",           # hanya BTC punya data — IC test akan drop jika noise
    "volume_zscore_h1",       # volume spike sebelum entry
    "atr_14_h1",              # raw ATR (bukan z-score)
    "ema_20_slope_h1",        # short-term momentum
]

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Load Guardian model ────────────────────────────────────────────────────────
GDN_RUN    = MODEL_DIR / "runs/tb_guardian_widyawardhana_v2"
gdn_model  = joblib.load(GDN_RUN / "guardian.pkl")
gdn_scaler = joblib.load(GDN_RUN / "guardian_scaler.pkl")
with open(GDN_RUN / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    gdn_all_feats = json.load(f)

gdn_static = [f for f in gdn_all_feats if f not in DYNAMIC_NAMES]
gdn_smap   = {n: i for i, n in enumerate(gdn_static)}
gdn_order  = [
    ("static", gdn_smap[f]) if f not in DYNAMIC_NAMES else ("dyn", f)
    for f in gdn_all_feats
]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Guardian simulation on OOF trades
# ──────────────────────────────────────────────────────────────────────────────
def _gdn_row(j, i, close, atr, direction, max_fav, X_static):
    bh  = j - i
    pnl = (close[j] - close[i]) / close[i] * direction
    atp = atr[i] / close[i] if close[i] > 0 else 0.01
    nmx = max(max_fav, pnl)
    dyn = {
        "bars_held_norm"        : bh / MAX_HOLD,
        "current_pnl_pct"       : pnl,
        "current_pnl_atr"       : pnl / atp if atp > 0 else 0.0,
        "max_favorable_pnl_pct" : nmx,
        "drawdown_from_peak_pct": (nmx - pnl) / nmx if nmx > 0.001 else 0.0,
        "direction"             : float(direction),
        "entry_price_ratio"     : close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(gdn_order), dtype=np.float64)
    for k, (src, key) in enumerate(gdn_order):
        row[k] = X_static[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, nmx


def simulate_with_guardian(pos, direction, close, high, low, atr, X_static):
    """Simulate a single trade starting at bar `pos` through Guardian."""
    n = len(close)
    i = pos
    entry    = close[i]
    sl_price = entry - direction * SL_MULT * atr[i]
    max_fav  = 0.0
    exit_p   = close[min(i + MAX_HOLD, n - 1)]
    exit_b   = min(i + MAX_HOLD, n - 1)
    outcome  = "TIME"

    for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
        if direction == 1 and low[j] <= sl_price:
            exit_p, exit_b, outcome = sl_price, j, "SL"
            break
        if direction == -1 and high[j] >= sl_price:
            exit_p, exit_b, outcome = sl_price, j, "SL"
            break
        if j - i >= GDN_MIN:
            row, max_fav = _gdn_row(j, i, close, atr, direction, max_fav, X_static)
            sc   = gdn_scaler.transform(row.reshape(1, -1))
            prob = gdn_model.predict_proba(sc)[0]
            ep   = prob[2] if len(prob) > 2 else prob[1]
            if ep >= GDN_THR:
                exit_p, exit_b, outcome = close[j], j, "GDN"
                break
        else:
            max_fav = max(max_fav, (close[j] - entry) / entry * direction)

    ret     = (exit_p - entry) / entry * direction
    net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE
    return {
        "win"       : 1 if net_pnl > 0 else 0,
        "net_pnl"   : net_pnl,
        "outcome"   : outcome,
        "bars_held" : exit_b - i,
    }


def generate_guardian_oof(oof_df):
    """
    Untuk setiap OOF trade, simulasikan dengan Guardian menggunakan data
    training (LABEL_DIR). Return DataFrame dengan kolom tambahan:
      guardian_win, guardian_pnl, guardian_outcome, guardian_bars
    """
    logger.info("Generating Guardian-OOF labels...")
    results = []
    total = len(oof_df)
    no_data = 0

    coins = oof_df["coin"].unique()
    for sym in sorted(coins):
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"  [{sym}] feature parquet missing — skip")
            no_data += oof_df[oof_df["coin"] == sym].shape[0]
            continue

        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        # Only training period
        cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        df = df[df.index < cutoff]

        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        atr   = df["atr_14_h1"].values.astype(np.float64)

        # Build static Guardian feature matrix
        X_static = np.zeros((len(df), len(gdn_static)), dtype=np.float64)
        for idx, c in enumerate(gdn_static):
            if c in df.columns:
                X_static[:, idx] = df[c].ffill().fillna(0).values

        # Build index lookup
        ts_idx = df.index  # UTC DatetimeIndex

        sym_oof = oof_df[oof_df["coin"] == sym]
        found = 0
        for ts, row in sym_oof.iterrows():
            ts_utc = ts if getattr(ts, "tzinfo", None) is not None else ts.tz_localize("UTC")
            if ts_utc not in ts_idx:
                results.append({**row.to_dict(), "coin": sym,
                                 "guardian_win": np.nan, "guardian_pnl": np.nan,
                                 "guardian_outcome": "MISSING", "guardian_bars": np.nan})
                continue

            pos = ts_idx.get_loc(ts_utc)
            sim = simulate_with_guardian(
                pos, int(row["direction"]),
                close, high, low, atr, X_static
            )
            results.append({**row.to_dict(), "coin": sym,
                             "guardian_win"     : sim["win"],
                             "guardian_pnl"     : sim["net_pnl"],
                             "guardian_outcome" : sim["outcome"],
                             "guardian_bars"    : sim["bars_held"]})
            found += 1

        logger.info(f"  [{sym}] {found}/{len(sym_oof)} trades simulated")
        no_data += (len(sym_oof) - found)

    gdf = pd.DataFrame(results)
    gdf.index = oof_df.index[:len(gdf)]
    valid = gdf["guardian_win"].notna()
    logger.info(
        f"\n  Total OOF: {total:,} | Valid: {valid.sum():,} | Missing: {no_data:,}"
    )
    raw_wr     = oof_df["win"].mean() * 100
    guardian_wr = gdf.loc[valid, "guardian_win"].mean() * 100
    logger.info(f"  Raw LGBM WIN: {raw_wr:.1f}%  -> Guardian WIN: {guardian_wr:.1f}%")
    return gdf[valid].copy()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2-3: IC test + feature selection (Simon)
# ──────────────────────────────────────────────────────────────────────────────
def spearman_ic(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        return 0.0, 0.0, mask.sum()
    ic, _ = stats.spearmanr(x[mask], y[mask])
    n     = mask.sum()
    t     = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-9)
    return float(ic), float(t), int(n)


def residualize(y, x):
    """Linear residualization: return y - proj(y on x)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return y.copy()
    b, a = np.polyfit(x[mask], y[mask], 1)
    res = y.copy()
    res[mask] = y[mask] - (a + b * x[mask])
    return res


def simon_feature_selection(gdf, df_by_coin, candidates):
    """
    Simon Steps 2-3:
      (a) Standalone IC: setiap fitur vs guardian_win
      (b) Gram-Schmidt marginal IC: drop fitur yang redundan dgn fitur terpilih
    Return: list fitur terpilih, DataFrame hasil IC test
    """
    logger.info("\n" + "="*60)
    logger.info("  SIMON STEP 2: Standalone IC test (features vs guardian_win)")
    logger.info("="*60)

    target_win = gdf["guardian_win"].values.astype(np.float32)

    # Kumpulkan nilai fitur pada ENTRY timestamp per trade
    feat_vals = {c: np.full(len(gdf), np.nan, dtype=np.float32) for c in candidates}

    for i, (ts, row) in enumerate(gdf.iterrows()):
        sym = row["coin"]
        df  = df_by_coin.get(sym)
        if df is None:
            continue
        ts_utc = ts if getattr(ts, "tzinfo", None) is not None else ts.tz_localize("UTC")
        if ts_utc not in df.index:
            continue
        pos = df.index.get_loc(ts_utc)
        for c in candidates:
            if c in df.columns and pos < len(df):
                v = df[c].iloc[pos]
                feat_vals[c][i] = v if np.isfinite(float(v)) else np.nan

    # Standalone IC
    ic_results = []
    for feat in candidates:
        vals = feat_vals[feat]
        ic, t, n = spearman_ic(vals, target_win)
        verdict = ("KEEP" if abs(ic) >= IC_MIN and abs(t) >= T_MIN
                   else ("WEAK" if abs(ic) >= IC_MIN * 0.5 else "DROP"))
        ic_results.append({
            "feature": feat, "IC": round(ic, 4), "t_stat": round(t, 2),
            "n": n, "verdict": verdict
        })

    ic_df = pd.DataFrame(ic_results).sort_values("IC", key=abs, ascending=False)

    print(f"\n  {'Feature':<24} {'IC':>8} {'t-stat':>8} {'n':>7} {'Verdict'}")
    print(f"  {'-'*60}")
    for _, r in ic_df.iterrows():
        star = " *" if r["verdict"] == "KEEP" else ""
        print(f"  {r['feature']:<24} {r['IC']:>+8.4f} {r['t_stat']:>8.2f} "
              f"{r['n']:>7,} {r['verdict']}{star}")

    keep_feats = ic_df[ic_df["verdict"] == "KEEP"]["feature"].tolist()
    logger.info(f"\n  Standalone KEEP: {len(keep_feats)}/{len(candidates)} features")

    if len(keep_feats) == 0:
        logger.warning("  No features passed IC! Using top-5 by |IC|.")
        keep_feats = ic_df.head(5)["feature"].tolist()

    # Gram-Schmidt marginal IC — drop redundant features
    logger.info("\n  SIMON STEP 3: Marginal IC (independence filter)")
    logger.info(f"  Threshold: |marginal IC| >= {MARG_MIN}")

    # Sort by |IC| desc, iteratively add
    keep_sorted = ic_df[ic_df["feature"].isin(keep_feats)].sort_values(
        "IC", key=abs, ascending=False
    )["feature"].tolist()

    selected = []
    selected_vals = []  # list of feature value arrays (for residualization)
    residual_win  = target_win.astype(np.float64).copy()

    for feat in keep_sorted:
        vals = feat_vals[feat].astype(np.float64)
        # Residualize feature against already-selected
        if selected_vals:
            for sv in selected_vals:
                vals = residualize(vals, sv)
        mic, mt, mn = spearman_ic(vals, residual_win)
        verdict = "ADD" if abs(mic) >= MARG_MIN and abs(mt) >= 1.0 else "REDUNDANT"
        print(f"  {feat:<24} marginal IC={mic:+.4f}  t={mt:>5.2f}  -> {verdict}")

        if verdict == "ADD":
            selected.append(feat)
            selected_vals.append(feat_vals[feat].astype(np.float64))
            # Update residual target
            residual_win = residualize(residual_win, feat_vals[feat].astype(np.float64))

    logger.info(f"\n  Final selected: {len(selected)} features: {selected}")
    return selected, ic_df, feat_vals


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Build sequences + train binary LSTM
# ──────────────────────────────────────────────────────────────────────────────
class BinaryLSTMMetaV2(nn.Module):
    def __init__(self, n_feat, hidden=32, n_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(h[-1])))


def build_sequences(gdf, df_by_coin, feat_cols, seq_len):
    X_list, y_list, ts_list, coin_list, conf_list = [], [], [], [], []

    for sym, grp in gdf.groupby("coin"):
        df = df_by_coin.get(sym)
        if df is None:
            continue

        for ts, row in grp.iterrows():
            ts_utc = ts if getattr(ts, "tzinfo", None) is not None else ts.tz_localize("UTC")
            if ts_utc not in df.index:
                continue
            pos = df.index.get_loc(ts_utc)
            if pos < seq_len - 1:
                continue

            seq_raw = df.iloc[pos - seq_len + 1: pos + 1][feat_cols]
            if len(seq_raw) != seq_len:
                continue

            seq = seq_raw.ffill().fillna(0).values.astype(np.float32)
            # Append direction as constant feature (16th)
            direction = np.full((seq_len, 1), float(row["direction"]), dtype=np.float32)
            seq = np.concatenate([seq, direction], axis=1)

            X_list.append(seq)
            y_list.append(float(row["guardian_win"]))
            ts_list.append(ts_utc)
            coin_list.append(sym)
            conf_list.append(float(row["confidence"]))

    X  = np.array(X_list, dtype=np.float32)
    y  = np.array(y_list, dtype=np.float32)
    ts = np.array(ts_list)
    return X, y, ts, np.array(coin_list), np.array(conf_list)


def build_temporal_folds(timestamps, n_folds=8, purge_days=3):
    ts       = pd.DatetimeIndex(timestamps)
    order    = np.argsort(ts)
    ts_s     = ts[order]
    fold_sz  = len(ts_s) // (n_folds + 1)
    folds    = []
    for k in range(1, n_folds + 1):
        v_start  = ts_s[k * fold_sz]
        v_end    = ts_s[min((k + 1) * fold_sz, len(ts_s) - 1)]
        purge_cut = v_start - pd.Timedelta(days=purge_days)
        tr_idx   = np.where(ts < purge_cut)[0]
        val_idx  = np.where((ts >= v_start) & (ts < v_end))[0]
        if len(tr_idx) >= 200 and len(val_idx) >= 50:
            folds.append((tr_idx, val_idx))
    return folds


def train_fold_v2(X_tr, y_tr, X_val, y_val, pos_weight_val):
    n_tr, sl, nf = X_tr.shape

    scaler   = RobustScaler()
    X_tr_2d  = X_tr.reshape(-1, nf)
    X_val_2d = X_val.reshape(-1, nf)
    scaler.fit(X_tr_2d)
    X_tr_sc  = scaler.transform(X_tr_2d).reshape(n_tr, sl, nf).astype(np.float32)
    X_val_sc = scaler.transform(X_val_2d).reshape(len(X_val), sl, nf).astype(np.float32)

    device = "cpu"
    model  = BinaryLSTMMetaV2(nf, HIDDEN, N_LAYERS, DROPOUT).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pw     = torch.tensor([pos_weight_val], dtype=torch.float32)

    tr_ds   = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_tr_sc), torch.FloatTensor(y_tr))
    loader  = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    best_auc, best_ep, best_state = 0.0, 0, None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred    = model(xb).squeeze(1)
            weights = torch.where(yb == 1, pw, torch.ones_like(yb))
            loss    = (weights * nn.functional.binary_cross_entropy(
                pred, yb, reduction="none")).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_sc)).squeeze(1).numpy()
        try:
            auc = roc_auc_score(y_val, val_pred)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc  = auc
            best_ep   = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auc, best_ep, scaler, val_pred


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: Simon Gate — marginal IC(lstm_score | lgbm_conf)
# ──────────────────────────────────────────────────────────────────────────────
def simon_gate(lstm_oof, lgbm_conf, guardian_win):
    mask    = np.isfinite(lstm_oof) & np.isfinite(lgbm_conf) & np.isfinite(guardian_win)
    ls, lc, lw = lstm_oof[mask], lgbm_conf[mask], guardian_win[mask]

    def residuals(y, x):
        b, a = np.polyfit(x, y, 1)
        return y - (a + b * x)

    res_lstm = residuals(ls, lc)
    res_win  = residuals(lw, lc)

    ic, pval = stats.spearmanr(res_lstm, res_win)
    n     = mask.sum()
    t_val = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-9)
    return {
        "ic"    : round(float(ic), 4),
        "t_stat": round(float(t_val), 2),
        "n"     : int(n),
        "pval"  : round(float(pval), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  LSTM Meta v2 — Guardian-OOF + Simon IC | {RUN_NAME}")
    print(f"  LSTM: hidden={HIDDEN} dropout={DROPOUT} lr={LR} patience={PATIENCE}")
    print(f"  IC thresholds: standalone={IC_MIN}/t={T_MIN}  marginal={MARG_MIN}")
    print(f"{'='*68}\n")

    # ── Load OOF dataset (LGBM signals) ───────────────────────────────────────
    oof_path = MODEL_DIR / "runs/tb_meta_v1/oof_meta_dataset.parquet"
    oof = pd.read_parquet(oof_path)
    logger.info(f"Loaded OOF: {len(oof):,} trades | raw WIN={oof['win'].mean()*100:.1f}%")

    # ── Load training feature parquets (per coin) ─────────────────────────────
    logger.info("Loading training feature parquets...")
    df_by_coin = {}
    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")

    for sym in oof["coin"].unique():
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < cutoff]
        df_by_coin[sym] = df
    logger.info(f"  Loaded {len(df_by_coin)} coins")

    # ── Step 1: Guardian-OOF simulation ───────────────────────────────────────
    gdf = generate_guardian_oof(oof)

    gdf_path = RUN_DIR / "guardian_oof_labels.parquet"
    gdf.to_parquet(gdf_path)
    logger.info(f"  Saved Guardian-OOF labels: {gdf_path}")

    # ── Steps 2-3: IC test + feature selection ────────────────────────────────
    # Only keep candidates that exist in at least one coin
    all_cols = set()
    for df in df_by_coin.values():
        all_cols |= set(df.columns)
    cands = [c for c in CANDIDATES if c in all_cols]
    logger.info(f"\n  Candidates available: {len(cands)}/{len(CANDIDATES)}")

    selected_feats, ic_df, feat_vals = simon_feature_selection(gdf, df_by_coin, cands)

    # Save IC results
    ic_df.to_json(RUN_DIR / f"{RUN_NAME}_ic_results.json", orient="records", indent=2)

    if len(selected_feats) == 0:
        logger.error("No features selected! Abort.")
        return

    # Re-slice df_by_coin to selected features only
    for sym in df_by_coin:
        avail = [f for f in selected_feats if f in df_by_coin[sym].columns]
        # Pad missing features with zero column
        for f in selected_feats:
            if f not in df_by_coin[sym].columns:
                df_by_coin[sym][f] = 0.0
        df_by_coin[sym] = df_by_coin[sym][selected_feats]

    # ── Step 4: Build sequences ────────────────────────────────────────────────
    logger.info(f"\n  Building sequences (selected={len(selected_feats)} + direction)...")
    X, y, ts, coins, lgbm_conf = build_sequences(gdf, df_by_coin, selected_feats, SEQ_LEN)
    n_total = len(X)
    n_feat  = X.shape[2]  # selected + direction
    logger.info(f"  Sequences: {n_total:,} | Shape: {X.shape}")

    base_win  = float(y.mean())
    pos_weight = float((1 - base_win) / (base_win + 1e-9))
    logger.info(f"  Guardian WIN: {base_win*100:.1f}%  pos_weight={pos_weight:.2f}")

    # ── Step 4: Temporal CV ────────────────────────────────────────────────────
    logger.info(f"\n  Running {N_FOLDS}-fold temporal CV (purge={PURGE_DAYS}d)...")
    folds = build_temporal_folds(ts, N_FOLDS, PURGE_DAYS)
    logger.info(f"  Valid folds: {len(folds)}")

    oof_scores  = np.full(n_total, np.nan)
    fold_aucs   = []
    best_epochs = []

    for k, (tr_idx, val_idx) in enumerate(folds, 1):
        model, auc, ep, scaler, val_pred = train_fold_v2(
            X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], pos_weight
        )
        thr_mask = val_pred >= 0.55
        wr_sel   = y[val_idx][thr_mask].mean() if thr_mask.sum() > 0 else 0.0
        oof_scores[val_idx] = val_pred
        fold_aucs.append(auc)
        best_epochs.append(ep)
        logger.info(
            f"  Fold {k}/{len(folds)}: n_tr={len(tr_idx):,} n_val={len(val_idx):,} | "
            f"AUC={auc:.4f} ep={ep} | WR@0.55={wr_sel*100:.1f}% (n={thr_mask.sum()})"
        )

    mean_auc = float(np.nanmean(fold_aucs))
    std_auc  = float(np.nanstd(fold_aucs))
    avg_ep   = max(int(np.mean(best_epochs)), 5)
    logger.info(f"\n  CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    logger.info(f"  Avg best epoch: {avg_ep}")

    # ── Step 6: Simon Gate ─────────────────────────────────────────────────────
    valid_mask = np.isfinite(oof_scores)
    mic = simon_gate(oof_scores[valid_mask], lgbm_conf[valid_mask], y[valid_mask])
    gate_pass = abs(mic["ic"]) >= 0.02 and abs(mic["t_stat"]) >= 2.0

    print(f"\n  {'='*60}")
    print(f"  SIMON GATE — Marginal IC(lstm | lgbm_confidence)")
    print(f"  IC = {mic['ic']:+.4f}  t = {mic['t_stat']:+.2f}  p = {mic['pval']:.4f}  n={mic['n']:,}")
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")
    print(f"  {'='*60}")

    # Threshold sweep
    base_wr = y[valid_mask].mean()
    print(f"\n  Threshold sweep (OOF, base_wr={base_wr*100:.1f}%, n={valid_mask.sum():,}):")
    print(f"  {'thr':>6}  {'cover%':>7}  {'WR_sel':>7}  {'lift':>6}")
    for thr in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        sel = oof_scores[valid_mask] >= thr
        if sel.sum() < 10:
            continue
        wr = y[valid_mask][sel].mean()
        print(f"  {thr:>6.2f}  {sel.mean()*100:>6.1f}%  {wr*100:>6.1f}%  "
              f"{(wr - base_wr)*100:>+5.1f}pp")

    # ── Final retrain on ALL data ──────────────────────────────────────────────
    logger.info(f"\n  Final retrain on all data (epochs={avg_ep})...")
    n_all, sl, nf = X.shape
    final_scaler  = RobustScaler()
    X_2d          = X.reshape(-1, nf)
    final_scaler.fit(X_2d)
    X_sc = final_scaler.transform(X_2d).reshape(n_all, sl, nf).astype(np.float32)

    final_model = BinaryLSTMMetaV2(nf, HIDDEN, N_LAYERS, DROPOUT)
    opt  = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    pw   = torch.tensor([pos_weight], dtype=torch.float32)
    ds   = torch.utils.data.TensorDataset(torch.FloatTensor(X_sc), torch.FloatTensor(y))
    ldr  = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    for epoch in range(1, avg_ep + 1):
        final_model.train()
        for xb, yb in ldr:
            opt.zero_grad()
            pred    = final_model(xb).squeeze(1)
            weights = torch.where(yb == 1, pw, torch.ones_like(yb))
            loss    = (weights * nn.functional.binary_cross_entropy(
                pred, yb, reduction="none")).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            opt.step()

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(final_model.state_dict(), RUN_DIR / "lstm_binary_meta.pt")
    joblib.dump(final_scaler, RUN_DIR / "lstm_binary_meta_scaler.pkl")

    with open(RUN_DIR / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(selected_feats, f, indent=2)

    meta = {
        "run_name"       : RUN_NAME,
        "target"         : "binary Guardian-WIN=1/LOSS=0",
        "n_samples"      : n_total,
        "base_win_rate"  : round(base_win, 4),
        "selected_feats" : selected_feats,
        "n_features"     : len(selected_feats),
        "n_feat_total"   : nf,
        "seq_len"        : SEQ_LEN,
        "hidden"         : HIDDEN,
        "n_layers"       : N_LAYERS,
        "dropout"        : DROPOUT,
        "lr"             : LR,
        "patience"       : PATIENCE,
        "cv_mean_auc"    : round(mean_auc, 4),
        "cv_std_auc"     : round(std_auc, 4),
        "avg_best_epoch" : avg_ep,
        "marginal_ic"    : mic,
        "gate_pass"      : gate_pass,
        "fold_aucs"      : [round(a, 4) for a in fold_aucs],
        "pos_weight"     : round(pos_weight, 3),
        "ic_thresholds"  : {"IC_MIN": IC_MIN, "T_MIN": T_MIN, "MARG_MIN": MARG_MIN},
    }
    with open(RUN_DIR / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  {RUN_NAME} COMPLETE")
    print(f"  CV Mean AUC  : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Simon Gate   : {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Marginal IC  : {mic['ic']:+.4f}  t={mic['t_stat']:+.2f}")
    print(f"  Selected feat: {len(selected_feats)} (from {len(CANDIDATES)} candidates)")
    print(f"  Guardian WIN : {base_win*100:.1f}% (vs raw LGBM {oof['win'].mean()*100:.1f}%)")
    print(f"  Model        : {RUN_DIR}/lstm_binary_meta.pt")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
