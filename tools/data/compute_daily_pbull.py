"""
tools/compute_daily_pbull.py — Hitung p_bull harian per koin (sisi-RISET) → artefak sync ke live.

ARSITEKTUR PARITY (Opsi B):
  Live TIDAK menjalankan daily LSTM. Live hanya membaca p_bull dari artefak ini.
  Dengan begitu p_bull live == p_bull riset secara by-construction — parity terjamin.

  Daily LSTM (`ic32_daily_lstm_17f_s30`) bergantung pada sumber data yang HANYA ada
  di repo riset: coinank OI/LS, macro cross-asset (SPY/UUP), ETF. Maka p_bull dihitung
  di sini, ditulis ke models/daily_pbull.parquet, lalu di-scp ke VPS via deploy bridge.

  Logika feature + inference IDENTIK dengan pipeline/07b_holdout_parity_guard_v2.py
  (yang menghasilkan holdout tervalidasi WR 68.7%). Jangan ubah tanpa re-validasi.

OUTPUT: models/daily_pbull.parquet
  kolom: [coin, date (UTC midnight), p_bull]
  Live meng-align D-1: p_bull tanggal D-1 dipakai untuk threshold bar H1 di tanggal D.

Pemakaian:
  python tools/compute_daily_pbull.py                            # default: 2026-01-01 .. hari ini
  python tools/compute_daily_pbull.py --start 2026-04-01 --end 2026-06-22
  python tools/compute_daily_pbull.py --use-binance              # pakai Binance-direct L/S
  python tools/compute_daily_pbull.py --use-binance --lstm-run ic32_daily_lstm_17f_s30_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from pipeline._bootstrap import setup_path_from_file
ROOT = setup_path_from_file(__file__)

import joblib
import torch
import torch.nn as nn

from core.models import _ManualLSTMCell
from core.utils import setup_logger, ensure_utc_index
from config import ALL_COINS, MODEL_DIR, HOLDOUT_DIR

logger = setup_logger("compute_daily_pbull")

DEFAULT_LSTM_RUN = "ic32_daily_lstm_17f_s30"
SEQ_LEN  = 30
POSITIONING_DIR  = ROOT / "data" / "positioning"

# Will be overridden per run from its feature_cols.json
LSTM_FEAT_COLS = None


# ─── Daily LSTM (identik 11_train_lstm_daily_v1.py / 07b) ─────────────────────
class DailyLSTM(nn.Module):
    def __init__(self, n_feat, hidden, layers, dropout):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.cells  = nn.ModuleList([
            _ManualLSTMCell(n_feat if i == 0 else hidden, hidden)
            for i in range(layers)
        ])
        self.lns  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        B, T, _ = x.shape
        dev = x.device
        h = [torch.zeros(B, self.hidden, device=dev) for _ in self.cells]
        c = [torch.zeros(B, self.hidden, device=dev) for _ in self.cells]
        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = self.lns[i](h[i])
                if i < len(self.cells) - 1:
                    inp = self.drop(inp)
        return torch.sigmoid(self.fc(self.drop(inp))).squeeze(-1)


def _align_ext(series, idx):
    return series.reindex(idx, method="ffill").shift(1)


def _load_price_h1(coin: str) -> pd.DataFrame | None:
    """Gabung harga H1 dari training labeled (konteks) + holdout processed (terbaru)."""
    parts = []
    train_path = ROOT / "data" / "training" / "labeled" / f"{coin}_features_v3.parquet"
    if train_path.exists():
        t = pd.read_parquet(train_path, columns=["close", "high", "low", "volume"])
        parts.append(ensure_utc_index(t))
    proc_path = HOLDOUT_DIR / "processed" / f"{coin}_clean.parquet"
    if proc_path.exists():
        raw = ensure_utc_index(pd.read_parquet(proc_path))
        cmap = {"close": "1h_close", "high": "1h_high", "low": "1h_low", "volume": "1h_volume"}
        h = pd.DataFrame(
            {k: raw[v] if v in raw.columns else raw.get(k) for k, v in cmap.items()},
            index=raw.index,
        )
        parts.append(h)
    if not parts:
        return None
    h1 = pd.concat(parts)
    h1 = h1[~h1.index.duplicated(keep="last")].sort_index()
    return h1


def _btc_daily_returns():
    btc = _load_price_h1("BTCUSDT")
    if btc is None:
        return None, None, None
    d = btc.resample("1D").last().ffill()
    return d["close"].pct_change(1), d["close"].pct_change(7), d["close"].pct_change(21)


def _macro_daily():
    mc = pd.read_parquet(ROOT / "data/macro/macro_cross_asset.parquet").sort_index()
    cols = ["spy_ret_5d", "uup_ret_5d"]
    if "etf_btc_d5" in mc.columns:  # if precomputed
        cols.append("etf_btc_d5")
    return mc[cols].resample("1D").last().ffill()


def build_daily_features(coin, btc_r1, btc_r7, btc_r21, mc_daily, ctx_start, use_binance: bool = False):
    """Bangun fitur harian 17-kolom. IDENTIK 07b.build_lstm_holdout_features."""
    h1 = _load_price_h1(coin)
    if h1 is None:
        return None
    d = h1.resample("1D").agg(
        {"close": "last", "high": "max", "low": "min", "volume": "sum"}
    ).dropna()
    d = d[d.index >= ctx_start]
    if len(d) < SEQ_LEN + 5:
        return None

    c, idx = d["close"], d.index
    d["ret_1d"] = c.pct_change(1)
    d["ret_5d"]  = c.pct_change(5)
    d["ret_7d"]  = c.pct_change(7)
    d["ret_reversal"] = np.sign(d["ret_1d"]) * np.sign(d["ret_7d"])
    d["vol_z20"] = (d["volume"] - d["volume"].rolling(20).mean()) / d["volume"].rolling(20).std().clip(lower=1e-8)

    if coin != "BTCUSDT":
        d["coin_vs_btc_7d"]  = c.pct_change(7)  - btc_r7.reindex(idx)
        d["coin_vs_btc_21d"] = c.pct_change(21) - btc_r21.reindex(idx)
        cov = c.pct_change(1).rolling(20).cov(btc_r1.reindex(idx))
        var = btc_r1.reindex(idx).rolling(20).var().clip(lower=1e-10)
        d["coin_beta_20d"]   = cov / var
    else:
        d["coin_vs_btc_7d"] = 0.0
        d["coin_vs_btc_21d"] = 0.0
        d["coin_beta_20d"] = 1.0

    d["spy_ret_5d"] = _align_ext(mc_daily["spy_ret_5d"], idx)
    d["uup_ret_5d"] = _align_ext(mc_daily["uup_ret_5d"], idx)
    if "etf_btc_d5" in mc_daily.columns:
        d["etf_btc_d5"] = _align_ext(mc_daily["etf_btc_d5"], idx).fillna(0)
    else:
        d["etf_btc_d5"] = 0.0

    oi_p = ROOT / "data" / "coinank" / f"{coin}_oi.parquet"
    if oi_p.exists():
        oi = pd.read_parquet(oi_p).sort_index()
        oi_col = "oi_binance" if "oi_binance" in oi.columns else oi.columns[0]
        oi_s = oi[oi_col].resample("1D").last().reindex(idx).ffill().shift(1)
        om, os_ = oi_s.rolling(20).mean(), oi_s.rolling(20).std().clip(lower=1e-8)
        d["oi_z20"]       = (oi_s - om) / os_
        d["oi_d7"]        = oi_s.pct_change(7)
        d["oi_price_div"] = np.sign(d["oi_d7"]) * np.sign(c.pct_change(7))
        voi               = d["volume"] / oi_s.clip(lower=1e-8)
        d["vol_oi_ratio"] = voi / voi.rolling(20).mean().clip(lower=1e-8)
        d["oi_avail"]     = (~oi_s.isna()).astype(float)
    else:
        for col in ["oi_z20", "oi_d7", "oi_price_div", "vol_oi_ratio"]:
            d[col] = 0.0
        d["oi_avail"] = 0.0

    if use_binance:
        # Binance-direct: coin-specific, benar (v2 model)
        ls_daily_p = POSITIONING_DIR / f"{coin}_ls_daily.parquet"
        if ls_daily_p.exists():
            ls_df = pd.read_parquet(ls_daily_p).sort_index()
            ls_s  = (ls_df["position_ls"].resample("1D").last().reindex(idx).ffill().shift(1)
                     if "position_ls" in ls_df.columns
                     else pd.Series(np.nan, index=idx))
            if ls_s.notna().any():
                lm, ls_ = ls_s.rolling(20).mean(), ls_s.rolling(20).std().clip(lower=1e-8)
                d["ls_z20"] = (ls_s - lm) / ls_
                d["ls_d7"]  = ls_s.diff(7)
            else:
                d["ls_z20"] = 0.0; d["ls_d7"] = 0.0
            if "account_ls" in ls_df.columns:
                lsa_s = ls_df["account_ls"].resample("1D").last().reindex(idx).ffill().shift(1)
                d["smart_retail"] = ls_s - lsa_s
            else:
                d["smart_retail"] = 0.0
            d["ls_avail"] = ls_s.notna().astype(float)
        else:
            for col in ["ls_z20", "ls_d7", "smart_retail"]:
                d[col] = 0.0
            d["ls_avail"] = 0.0
    else:
        # coinank: sumber lama (dipakai untuk parity dengan holdout tervalidasi)
        lsp_p = ROOT / "data" / "coinank" / f"{coin}_ls_position.parquet"
        lsa_p = ROOT / "data" / "coinank" / f"{coin}_ls_account.parquet"
        if lsp_p.exists():
            lsp  = pd.read_parquet(lsp_p).sort_index()
            ls_s = lsp.iloc[:, 0].resample("1D").last().reindex(idx).ffill().shift(1)
            lm, ls_ = ls_s.rolling(20).mean(), ls_s.rolling(20).std().clip(lower=1e-8)
            d["ls_z20"] = (ls_s - lm) / ls_
            d["ls_d7"]  = ls_s.diff(7)
            if lsa_p.exists():
                lsa   = pd.read_parquet(lsa_p).sort_index()
                lsa_s = lsa.iloc[:, 0].resample("1D").last().reindex(idx).ffill().shift(1)
                d["smart_retail"] = ls_s - lsa_s
            else:
                d["smart_retail"] = 0.0
            d["ls_avail"] = (~ls_s.isna()).astype(float)
        else:
            for col in ["ls_z20", "ls_d7", "smart_retail"]:
                d[col] = 0.0
            d["ls_avail"] = 0.0

    return d


def load_lstm(lstm_run: str):
    global LSTM_FEAT_COLS
    run_dir = MODEL_DIR / "runs" / lstm_run
    cfg = json.load(open(run_dir / "config.json"))
    model = DailyLSTM(cfg["n_feats"], cfg["hidden"], cfg["layers"], cfg["dropout"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False))
    model.eval()
    scaler = joblib.load(run_dir / "scaler.pkl")

    feat_file = run_dir / "feature_cols.json"
    if feat_file.exists():
        LSTM_FEAT_COLS = json.load(open(feat_file))
        logger.info(f"Loaded {len(LSTM_FEAT_COLS)} features from {feat_file.name}")
    else:
        if LSTM_FEAT_COLS is None:
            LSTM_FEAT_COLS = [
                "coin_beta_20d", "spy_ret_5d", "coin_vs_btc_21d",
                "oi_d7", "ls_z20", "coin_vs_btc_7d", "oi_price_div",
                "ls_d7", "vol_oi_ratio", "oi_z20", "smart_retail",
                "vol_z20", "uup_ret_5d", "etf_btc_d5", "ret_5d",
                "oi_avail", "ls_avail",
            ]
        logger.info(f"Using default {len(LSTM_FEAT_COLS)} features (no feature_cols.json)")

    return model, scaler


def infer_pbull(df, model, scaler, start, end):
    X = df[LSTM_FEAT_COLS].fillna(0).clip(-10, 10).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    n, f = X.shape
    Xs = scaler.transform(X.reshape(-1, f)).reshape(n, f)
    out = {}
    dates = df.index
    with torch.no_grad():
        for i in range(SEQ_LEN - 1, n):
            day = dates[i]
            if not (start <= day <= end):
                continue
            seq = Xs[i - SEQ_LEN + 1: i + 1]
            p = float(model(torch.from_numpy(seq[np.newaxis]).float()).item())
            out[day] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",       default="2026-01-01")
    ap.add_argument("--end",         default=None, help="default: hari ini UTC")
    ap.add_argument("--out",         default=str(MODEL_DIR / "daily_pbull.parquet"))
    ap.add_argument("--use-binance", action="store_true",
                    help="Pakai Binance-direct L/S (data/positioning/{coin}_ls_daily.parquet)")
    ap.add_argument("--lstm-run",    default=None,
                    help=f"Override LSTM run ID (default: {DEFAULT_LSTM_RUN})")
    args = ap.parse_args()

    lstm_run    = args.lstm_run or DEFAULT_LSTM_RUN
    use_binance = args.use_binance

    start = pd.Timestamp(args.start, tz="UTC").normalize()
    end   = (pd.Timestamp(args.end, tz="UTC") if args.end
             else pd.Timestamp.utcnow()).normalize()
    ctx_start = start - pd.Timedelta(days=SEQ_LEN + 5)

    logger.info(f"Range p_bull: {start.date()} .. {end.date()}")
    logger.info(f"LSTM run: {lstm_run} | L/S source: {'Binance-direct' if use_binance else 'coinank'}")
    model, scaler = load_lstm(lstm_run)
    btc_r1, btc_r7, btc_r21 = _btc_daily_returns()
    mc_daily = _macro_daily()

    rows = []
    for coin in ALL_COINS:
        df = build_daily_features(coin, btc_r1, btc_r7, btc_r21, mc_daily, ctx_start, use_binance=use_binance)
        if df is None:
            logger.warning(f"  {coin}: data tidak cukup — skip")
            continue
        pb = infer_pbull(df, model, scaler, start, end)
        for day, val in pb.items():
            rows.append({"coin": coin, "date": day, "p_bull": val})
        logger.info(f"  {coin}: {len(pb)} hari p_bull "
                    f"[min={min(pb.values()):.3f} max={max(pb.values()):.3f}]" if pb else f"  {coin}: 0 hari")

    if not rows:
        logger.error("Tidak ada p_bull dihasilkan — cek data sumber.")
        return 1

    out_df = pd.DataFrame(rows).sort_values(["coin", "date"]).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    logger.info(f"OK — {len(out_df)} baris ({out_df['coin'].nunique()} koin) -> {out_path}")
    logger.info(f"Tanggal: {out_df['date'].min().date()} .. {out_df['date'].max().date()}")
    print(out_df.groupby("coin")["p_bull"].agg(["count", "mean", "min", "max"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
