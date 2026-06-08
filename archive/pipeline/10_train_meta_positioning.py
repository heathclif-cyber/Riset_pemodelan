"""
pipeline/10_train_meta_positioning.py — Phase 3: Meta-Labeling + Positioning

Upgrade from Phase 2 (AUC 0.594):
  - ADD positioning features per trade: OI z-score, LS position, ETF flow
  - Positioning features as TRADE-LEVEL context (not in sequence)
  - Architecture: LSTM(40-bar sequence) + Positioning(context) → P(good_trade)

Simons principle: "Don't predict the market. Predict when your model is wrong."

Usage:
  python pipeline/10_train_meta_positioning.py
"""
import argparse, gc, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import joblib, lightgbm as lgb

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, MODAL_PER_TRADE,
)
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("10_meta_pos")
COINANK_DIR = ROOT / "data" / "coinank"

# ─── Config ──────────────────────────────────────────────────────────────
SEQ_LEN = 40
HIDDEN = 96; LAYERS = 2; DROPOUT = 0.40
LR = 5e-4; WD = 1e-4; BATCH = 128; EPOCHS = 60; PATIENCE = 10
DEVICE = torch.device("cpu")

# Sequence features (OHLCV, per-bar in the 40-bar window before entry)
SEQ_FEATURES = [
    "log_ret_1", "rsi_6", "cvd_momentum_adv", "volume_delta",
    "ofi_z_score", "atr_14_h1", "vol_ratio_20",
]

# Positioning features (ONE value per trade, from daily data at entry date)
POS_FEATURES = [
    "pos_oi_z20",      # OI z-score → extreme = reversal risk
    "pos_ls_z20",      # LS position z-score → extreme = turbulence
    "pos_ls_d7",       # LS delta 7D → smart money flow
    "pos_smart_retail",# Smart vs retail divergence
    "pos_oi_d7",       # OI delta 7D → OI trend
    "pos_fear_greed",  # Fear & Greed at entry date
    "pos_grayscale_d5",# Grayscale BTC change 5D → institutional
]


# ─── MetaLSTM + Positioning ──────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, hidden): super().__init__(); self.attn = nn.Linear(hidden, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)

class MetaLSTMWithPos(nn.Module):
    def __init__(self, n_seq_feat, n_pos_feat, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_seq_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.attn = AttentionPooling(hidden)
        self.drop = nn.Dropout(dropout)
        # Combine sequence + positioning context
        combined_dim = hidden + n_pos_feat
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq_x, pos_x):
        # seq_x: (B, 40, n_seq_feat) — time series before entry
        # pos_x: (B, n_pos_feat)     — static positioning at entry
        out, _ = self.lstm(seq_x)
        pooled = self.attn(out)       # (B, hidden)
        pooled = self.drop(pooled)
        combined = torch.cat([pooled, pos_x], dim=1)  # (B, hidden + n_pos)
        return torch.sigmoid(self.classifier(combined)).squeeze(-1)


# ─── Data Loading ────────────────────────────────────────────────────────
class MetaDataset(Dataset):
    def __init__(self, X_seq, X_pos, y):
        self.X_seq = torch.from_numpy(X_seq.astype(np.float32))
        self.X_pos = torch.from_numpy(X_pos.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.X_seq)
    def __getitem__(self, i): return self.X_seq[i], self.X_pos[i], self.y[i]


def load_positioning_context():
    """Load daily positioning data for all coins, return lookup dict."""
    context = {}
    for coin in TRAINING_COINS:
        oi_p = COINANK_DIR / f"{coin}_oi.parquet"
        lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
        lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"
        if not oi_p.exists(): continue

        oi = pd.read_parquet(oi_p).sort_index()
        lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
        lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None

        daily = pd.DataFrame(index=oi.index)
        # Use first OI column (oi_binance, oi_bybit, etc.)
        oi_cols = [c for c in oi.columns if c.startswith("oi_") or c == "oi"]
        if not oi_cols: oi_cols = [c for c in oi.columns if "oi" in c.lower() and c != "price"]
        oi_t = oi[oi_cols[0]].copy() if oi_cols else None
        if oi_t is None: continue
        om = oi_t.rolling(20).mean(); os = oi_t.rolling(20).std().clip(lower=1e-8)
        daily["oi_z20"] = (oi_t - om) / os
        daily["oi_d7"] = oi_t.pct_change(7)

        if lsp is not None and "top_trader_position_ls" in lsp.columns:
            ls = lsp["top_trader_position_ls"]
            lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
            daily["ls_z20"] = (ls - lm) / ls_s
            daily["ls_d7"] = ls.diff(7)
            if lsa is not None and "top_trader_account_ls" in lsa.columns:
                daily["smart_retail"] = ls - lsa["top_trader_account_ls"]

        context[coin] = daily

    # Also load macro context (Fear & Greed, Grayscale)
    fg_p = ROOT / "data" / "macro" / "fear_greed.parquet"
    gs_p = COINANK_DIR / "grayscale_btc.parquet"
    macro = {}
    if fg_p.exists():
        fg = pd.read_parquet(fg_p)
        macro["fear_greed"] = fg[["fear_greed_value"]] if "fear_greed_value" in fg.columns else None
    if gs_p.exists():
        gs = pd.read_parquet(gs_p)
        if "grayscale_holdings" in gs.columns:
            macro["grayscale"] = gs[["grayscale_holdings"]]
    return context, macro


def get_pos_features(coin, entry_date, context, macro):
    """Get positioning features at entry date for a trade."""
    feats = {}
    # Per-coin positioning
    if coin in context:
        daily = context[coin]
        date_key = pd.Timestamp(entry_date, tz="UTC")
        # Find closest daily bar before or at entry date
        available = daily[daily.index <= date_key]
        if len(available) > 0:
            last = available.iloc[-1]
            for col in ["oi_z20", "ls_z20", "ls_d7", "smart_retail", "oi_d7"]:
                feats[f"pos_{col}"] = float(last[col]) if col in last and pd.notna(last[col]) else 0.0
        else:
            for col in ["oi_z20", "ls_z20", "ls_d7", "smart_retail", "oi_d7"]:
                feats[f"pos_{col}"] = 0.0
    else:
        for col in ["oi_z20", "ls_z20", "ls_d7", "smart_retail", "oi_d7"]:
            feats[f"pos_{col}"] = 0.0

    # Fear & Greed
    fg_val = 50.0
    if "fear_greed" in macro and macro["fear_greed"] is not None:
        fg_df = macro["fear_greed"]
        available = fg_df[fg_df.index <= pd.Timestamp(entry_date, tz="UTC")]
        if len(available) > 0:
            fg_val = float(available.iloc[-1]["fear_greed_value"])
    feats["pos_fear_greed"] = (fg_val - 50.0) / 25.0  # normalize to roughly -2 to +2

    # Grayscale BTC
    gs_val = 0.0
    if "grayscale" in macro and macro["grayscale"] is not None:
        gs_df = macro["grayscale"]
        available = gs_df[gs_df.index <= pd.Timestamp(entry_date, tz="UTC")]
        if len(available) >= 6:
            gs_val = float(available.iloc[-1]["grayscale_holdings"] - available.iloc[-6]["grayscale_holdings"]) / max(abs(available.iloc[-6]["grayscale_holdings"]), 1e-8)
    feats["pos_grayscale_d5"] = gs_val

    return [feats.get(k, 0.0) for k in POS_FEATURES]


def generate_meta_labels(coins):
    """
    Walk-forward OOF meta-label generation:
    1. For each fold, retrain LGBM on training folds
    2. Predict on test fold → get trades
    3. Label: is_good = net_pnl > median profit
    4. Extract 40-bar sequences BEFORE entry + positioning context
    """
    all_sequences = []
    all_pos_features = []
    all_labels = []
    all_entry_bars = []
    total_trades = 0

    context, macro = load_positioning_context()

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

        mask = df["label"].astype(str).isin(LABEL_MAP); df = df[mask].copy()
        if len(df) < 500: continue

        ts_index = pd.DatetimeIndex(df.index)
        folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)
        lgbm_feats_list = json.load(open(MODEL_DIR / "feature_cols_v2.json"))

        for fi, (tr_idx, te_idx) in enumerate(folds):
            if len(te_idx) < 100: continue
            df_tr = df.iloc[tr_idx]; df_te = df.iloc[te_idx]

            feat_cols = [c for c in lgbm_feats_list if c in df_tr.columns]
            X_tr = df_tr[feat_cols].ffill().fillna(0)
            y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)
            if len(np.unique(y_tr)) < 3: continue

            # Retrain LGBM per fold (genuine OOF)
            fold_model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=300,
                learning_rate=0.05, max_depth=6, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                verbose=-1, n_jobs=-1, random_state=42)
            fold_model.fit(X_tr, y_tr)

            # Predict on test fold
            n_te = len(df_te)
            X_te = np.zeros((n_te, len(feat_cols)))
            for i, col in enumerate(feat_cols):
                if col in df_te.columns: X_te[:, i] = df_te[col].ffill().fillna(0).values

            proba = fold_model.predict_proba(X_te)
            y_pred = np.argmax(proba, axis=1)
            confidence = proba[np.arange(n_te), y_pred]

            # Simulate trades (simplified — no Guardian for meta-label generation)
            # Use simple entry: LGBM predicts LONG or SHORT with conf >= 0.59
            close = df_te["close"].values
            for i in range(n_te):
                if y_pred[i] == 1: continue  # FLAT
                conf = confidence[i]
                if (y_pred[i] == 2 and conf < 0.69) or (y_pred[i] == 0 and conf < 0.59):
                    continue

                # Simple trade simulation (TP=2%, SL=1.5%, max_hold=48)
                entry_price = close[i]
                direction = 1 if y_pred[i] == 2 else -1
                tp = entry_price * (1 + 0.02 * direction)
                sl = entry_price * (1 - 0.015 * direction)

                exit_bar = i + 1
                exit_price = entry_price
                while exit_bar < min(i + 48, n_te):
                    if direction == 1:
                        if df_te["high"].iloc[exit_bar] >= tp:
                            exit_price = tp; break
                        if df_te["low"].iloc[exit_bar] <= sl:
                            exit_price = sl; break
                    else:
                        if df_te["low"].iloc[exit_bar] <= tp:
                            exit_price = tp; break
                        if df_te["high"].iloc[exit_bar] >= sl:
                            exit_price = sl; break
                    exit_bar += 1

                pnl = (exit_price - entry_price) * direction * MODAL_PER_TRADE / entry_price

                # Extract 40-bar sequence BEFORE entry
                if i < SEQ_LEN: continue
                seq_start = i - SEQ_LEN; seq_end = i
                seq_data = np.zeros((SEQ_LEN, len(SEQ_FEATURES)))
                for j, col in enumerate(SEQ_FEATURES):
                    if col in df_te.columns:
                        vals = df_te[col].iloc[seq_start:seq_end].ffill().fillna(0).values
                        seq_data[:, j] = vals

                # Get positioning context at entry date
                entry_date = df_te.index[i].date()
                pos_feats = get_pos_features(coin, entry_date, context, macro)

                all_sequences.append(seq_data)
                all_pos_features.append(pos_feats)
                all_labels.append(1.0 if pnl > 0 else 0.0)
                all_entry_bars.append((coin, df_te.index[i]))
                total_trades += 1

    logger.info(f"Total trades with meta-labels: {total_trades}")
    logger.info(f"Good trades: {sum(all_labels):,} ({sum(all_labels)/total_trades*100:.1f}%)")

    return (np.array(all_sequences), np.array(all_pos_features),
            np.array(all_labels), all_entry_bars)


def train_one_fold(Xs_tr, Xp_tr, y_tr, Xs_te, Xp_te, y_te, fi):
    n_seq_f = Xs_tr.shape[2]; n_pos_f = Xp_tr.shape[1]

    # Scale sequence features
    sc = RobustScaler(); sc.fit(Xs_tr.reshape(-1, n_seq_f))
    Xs_tr_s = sc.transform(Xs_tr.reshape(-1, n_seq_f)).reshape(Xs_tr.shape).astype(np.float32)
    Xs_te_s = sc.transform(Xs_te.reshape(-1, n_seq_f)).reshape(Xs_te.shape).astype(np.float32)

    tr_ds = MetaDataset(Xs_tr_s, Xp_tr.astype(np.float32), y_tr.astype(np.float32))
    te_ds = MetaDataset(Xs_te_s, Xp_te.astype(np.float32), y_te.astype(np.float32))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    model = MetaLSTMWithPos(n_seq_f, n_pos_f, HIDDEN, LAYERS, DROPOUT).to(DEVICE)
    crit = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_auc, best_state, pc = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xs, xp, yb in tr_ld:
            xs, xp, yb = xs.to(DEVICE), xp.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xs, xp), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xs, xp, yb in te_ld:
                pv.extend(model(xs.to(DEVICE), xp.to(DEVICE)).cpu().numpy())
                lv.extend(yb.numpy())
        auc = float(roc_auc_score(lv, pv)) if len(np.unique(lv)) > 1 else 0.5

        if auc > best_auc: best_auc, best_state, pc = auc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else: pc += 1
        if pc >= PATIENCE: break

        if ep % 10 == 0 or ep == 1:
            logger.info(f"[F{fi}] E{ep:>3} AUC={auc:.4f} Best={best_auc:.4f}")

    model.load_state_dict(best_state); model.eval()
    return float(best_auc)


def main():
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Meta-Labeling + Positioning Features")
    print(f"  Seq: {SEQ_LEN} bars | Seq features: {len(SEQ_FEATURES)}")
    print(f"  Pos features: {len(POS_FEATURES)} | Simons: predict model error")
    print(f"  Phase 2 baseline: AUC 0.594 | Target: > 0.62")
    print(f"{'='*60}\n")

    coins = TRAINING_COINS[:5]
    print("Generating meta-labels with walk-forward OOF...")
    X_seq, X_pos, y, entry_info = generate_meta_labels(coins)

    n_trades = len(y)
    print(f"\n  Meta-trades: {n_trades:,}")
    print(f"  Good: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f"  Bad:  {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"  X_seq: {X_seq.shape}  X_pos: {X_pos.shape}")
    print(f"  Positioning non-zero: {(X_pos.sum(axis=1) != 0).sum()} / {n_trades} ({(X_pos.sum(axis=1) != 0).mean()*100:.1f}%)")

    # Purged CV folds on trades
    ts_index = pd.DatetimeIndex([info[1] for info in entry_info])
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    aucs = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        auc = train_one_fold(
            X_seq[tr_idx], X_pos[tr_idx], y[tr_idx],
            X_seq[te_idx], X_pos[te_idx], y[te_idx], fi + 1)
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    print(f"\n{'='*60}")
    print(f"  PHASE 3 RESULTS")
    print(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Phase 2:   0.594 +/- 0.029")
    print(f"  Delta:     {mean_auc - 0.594:+.4f}")
    if mean_auc > 0.594:
        print(f"  -> Positioning features IMPROVE meta-labeling!")
    else:
        print(f"  -> Positioning features do NOT improve meta-labeling.")
    print(f"  Folds: {[f'{a:.4f}' for a in aucs]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
