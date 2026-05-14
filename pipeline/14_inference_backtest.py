"""
pipeline/14_inference_backtest.py — Standalone Backtest pakai Inference Config

Menggunakan:
  - inference_config.json (swint_tradev2) sebagai source parameter
  - Model baseline (lgbm_baseline.pkl, lstm_best.pt, lstm_scaler.pkl)
  - Evaluator mandiri — tidak bergantung pada core/evaluator.py training
  - Data holdout/labeled/*_features_v3.parquet

Jalankan:
  python pipeline/14_inference_backtest.py --threshold 0.62
  python pipeline/14_inference_backtest.py --threshold 0.70
  python pipeline/14_inference_backtest.py --compare   # bandingkan 0.62 vs 0.70
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ——— Inference config ————————————————————————————————————————————————
INFERENCE_CONFIG_PATH = Path("D:/Apps-Dev/swint_tradev2/models/inference_config.json")

# ——— Model paths ——————————————————————————————————————————————————————
LGBM_PATH  = ROOT / "models/lgbm_baseline.pkl"
LSTM_PATH  = ROOT / "models/lstm_best.pt"
SCALER_PATH = ROOT / "models/lstm_scaler.pkl"
FEAT_PATH  = ROOT / "models/feature_cols_v2.json"
DATA_DIR   = ROOT / "data/holdout/labeled"

# ——— Import cascade dari pipeline sendiri —————————————————————————————
from pipeline.backtest_utils import hierarchical_predict, get_lstm_proba
from core.models import load_lstm

# =========================================================================
# EVALUATOR MANDIRI (mirip swint_tradev2: simpler, no bumper/hybrid/freshness)
# =========================================================================

def simulate_trades_inference(
    y_pred, close, high, low, atr,
    h4_swing_highs, h4_swing_lows,
    modal, leverage, fee_per_side, slippage,
    min_rr, min_tp_atr, max_sl_atr, max_hold,
    confidence=None,
):
    """Simulasi trade sederhana — sesuai inference_config.json."""
    n = len(close)
    trades = []
    equity = modal   # initial capital
    peak = modal
    max_dd_pct = 0.0  # DD sebagai % dari peak equity

    LONG, SHORT, FLAT = 2, 0, 1

    for i in range(n - 1):
        sig = y_pred[i]
        if sig == FLAT:
            continue

        price = close[i]
        atr_i = atr[i]
        sh_i  = h4_swing_highs[i] if h4_swing_highs is not None else np.nan
        sl_i  = h4_swing_lows[i] if h4_swing_lows is not None else np.nan

        if np.isnan(price) or np.isnan(atr_i) or atr_i == 0:
            continue

        # — TP/SL dari swing H4 (fallback ATR jika NaN) ——————————————
        if not np.isnan(sh_i) and not np.isnan(sl_i):
            if sig == LONG:
                tp_price = sh_i
                sl_price = sl_i
                tp_dist = tp_price - price
                sl_dist = price - sl_price
            else:
                tp_price = sl_i
                sl_price = sh_i
                tp_dist = price - tp_price
                sl_dist = sl_price - price
        else:
            # ATR fallback
            if sig == LONG:
                tp_dist = 2.0 * atr_i
                sl_dist = 1.5 * atr_i
            else:
                tp_dist = 2.0 * atr_i
                sl_dist = 1.5 * atr_i
            tp_price = price + tp_dist if sig == LONG else price - tp_dist
            sl_price = price - sl_dist if sig == LONG else price + sl_dist

        # — RR Gate ————————————————————————————————————————————————————
        if tp_dist <= 0 or sl_dist <= 0:
            continue
        if tp_dist < min_tp_atr * atr_i:
            continue
        if sl_dist > max_sl_atr * atr_i:
            continue
        if tp_dist / sl_dist < min_rr:
            continue

        # — Structural filter (entry dalam swing range) ———————————————
        if not np.isnan(sh_i) and not np.isnan(sl_i):
            if price > sh_i * 1.04 or price < sl_i * 0.96:
                continue

        # — Scan ke depan ——————————————————————————————————————————————
        outcome = "time_exit"
        exit_price = price
        end = min(i + max_hold, n)
        for j in range(i + 1, end):
            if np.isnan(high[j]) or np.isnan(low[j]):
                continue
            if sig == LONG:
                if high[j] >= tp_price:
                    outcome = "tp_hit"; exit_price = tp_price; break
                if low[j] <= sl_price:
                    outcome = "sl_hit"; exit_price = sl_price; break
            else:
                if low[j] <= tp_price:
                    outcome = "tp_hit"; exit_price = tp_price; break
                if high[j] >= sl_price:
                    outcome = "sl_hit"; exit_price = sl_price; break

        if outcome == "time_exit":
            exit_price = close[min(i + max_hold, n - 1)]

        # — PnL ————————————————————————————————————————————————————————
        pct_move = (exit_price - price) / price
        if sig == SHORT:
            pct_move = -pct_move

        # Slippage
        pct_move -= slippage * 2  # entry + exit

        gross_pnl = pct_move * modal * leverage
        fee = 2 * fee_per_side * modal
        net_pnl = gross_pnl - fee

        equity += net_pnl
        peak = max(peak, equity)
        if peak > 0:
            current_dd = (peak - equity) / peak
            max_dd_pct = max(max_dd_pct, current_dd)

        trades.append({
            "direction": "LONG" if sig == LONG else "SHORT",
            "entry": price,
            "exit": exit_price,
            "tp": tp_price,
            "sl": sl_price,
            "outcome": outcome,
            "net_pnl": net_pnl,
            "pct_move": pct_move,
            "hold_bars": min(j - i if outcome != "time_exit" else max_hold, max_hold),
        })

    # — Summary —————————————————————————————————————————————————————————
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    time_exits = [t for t in trades if t["outcome"] == "time_exit"]
    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]

    wr = len(wins) / len(trades) if trades else 0
    lw = len([t for t in wins if t["direction"] == "LONG"])
    lt = len(longs)
    sw = len([t for t in wins if t["direction"] == "SHORT"])
    st = len(shorts)

    total_pnl = sum(t["net_pnl"] for t in trades)
    dd_pct = max_dd_pct  # DD as % of peak equity
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0
    pf = abs(sum(t["net_pnl"] for t in wins) / sum(t["net_pnl"] for t in losses)) if losses else float('inf')

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "time_exits": len(time_exits),
        "winrate": round(wr, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(pf, 4),
        "total_pnl": round(total_pnl, 4),
        "max_drawdown_pct": round(dd_pct, 4),
        "win_by_class": {
            "LONG": round(lw / lt, 4) if lt > 0 else 0.0,
            "SHORT": round(sw / st, 4) if st > 0 else 0.0,
        },
        "n_long_trades": lt,
        "n_short_trades": st,
        "trades": trades,
    }


# =========================================================================
# MAIN
# =========================================================================

def load_config():
    with open(INFERENCE_CONFIG_PATH) as f:
        return json.load(f)


def backtest_symbol(symbol, feat_cols, lgbm, lstm, lstm_scaler, cfg, threshold):
    """Run inference backtest on one symbol."""
    path = DATA_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = df.sort_index()

    # Filter labeled data only
    label_map = {"SHORT": 0, "FLAT": 1, "LONG": 2}
    mask = df["label"].astype(str).isin(label_map)
    df = df[mask].copy()

    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)
    X = df[valid_cols].values.astype(np.float64)

    # Cascade predict
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, lstm_scaler,
        X, valid_cols, [], df[valid_cols],
    )

    # Confidence filter
    below = (y_pred != 1) & (confidence < threshold)
    y_pred_f = y_pred.copy()
    y_pred_f[below] = 1
    n_filt = int(below.sum())

    # Arrays
    close_arr = df["close"].values
    high_arr = df["high"].values if "high" in df.columns else close_arr
    low_arr = df["low"].values if "low" in df.columns else close_arr
    atr_arr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else None

    # Get params from inference config
    risk = cfg.get("risk", {})
    rr_gate = cfg.get("rr_gate", {})
    fallback = cfg.get("fallback_tp_sl", {})

    report = simulate_trades_inference(
        y_pred=y_pred_f,
        close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=risk.get("modal_per_trade", 100.0),
        leverage=5.0,
        fee_per_side=risk.get("fee_per_side", 0.0004),
        slippage=risk.get("slippage_per_side", 0.0005),
        min_rr=rr_gate.get("min_rr", 0.5),
        min_tp_atr=rr_gate.get("min_tp_atr", 1.2),
        max_sl_atr=rr_gate.get("max_sl_atr", 4.0),
        max_hold=cfg.get("inference", {}).get("max_hold_bars", 24),
        confidence=confidence,
    )

    report["n_filtered"] = n_filt
    report["confidence_threshold"] = threshold
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold (default: dari inference config)")
    parser.add_argument("--compare", action="store_true",
                        help="Bandingkan threshold 0.70 vs 0.62")
    parser.add_argument("--coins", nargs="+", default=None,
                        help="Coin yang di-backtest (default: 5 training coins)")
    args = parser.parse_args()

    print("=" * 70)
    print("INFERENCE BACKTEST — pakai inference_config.json + model baseline")
    print("=" * 70)

    # Load config
    cfg = load_config()
    default_threshold = cfg.get("inference", {}).get("confidence_threshold_entry", 0.70)
    print(f"Config version : {cfg.get('model_version')}")
    print(f"Default threshold: {default_threshold}")

    # Load models
    print("\nLoading models...")
    t0 = time.time()
    lgbm = joblib.load(LGBM_PATH)
    lstm = load_lstm(LSTM_PATH)
    lstm_scaler = joblib.load(SCALER_PATH)
    with open(FEAT_PATH) as f:
        feat_cols = json.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s | features: {len(feat_cols)}")

    # Coins
    coins = args.coins or ["SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]

    if args.compare:
        thresholds = [0.70, 0.62]
    else:
        thresholds = [args.threshold or default_threshold]

    all_results = {}
    for thr in thresholds:
        print(f"\n{'='*70}")
        print(f"THRESHOLD = {thr}")
        print(f"{'='*70}")
        print(f"{'Coin':<15s} {'Trades':>6s} {'LONG':>6s} {'SHORT':>6s} {'WR':>8s} {'LONG WR':>8s} {'SHORT WR':>8s} {'PnL':>10s} {'DD':>8s}")

        results = {}
        total_trades = 0
        total_pnl = 0.0
        total_long = 0
        total_short = 0
        total_lw = 0.0
        total_sw = 0.0

        for symbol in coins:
            t1 = time.time()
            report = backtest_symbol(symbol, feat_cols, lgbm, lstm, lstm_scaler, cfg, thr)
            if report is None:
                print(f"{symbol:<15s} SKIP (no data)")
                continue

            n = report["total_trades"]
            lt = report["n_long_trades"]
            st = report["n_short_trades"]
            wr = report["winrate"]
            lw = report["win_by_class"]["LONG"]
            sw = report["win_by_class"]["SHORT"]
            pnl = report["total_pnl"]
            dd = report["max_drawdown_pct"]
            f = report["n_filtered"]
            elapsed = time.time() - t1

            print(f"{symbol:<15s} {n:>6d} {lt:>6d} {st:>6d} {wr:>7.1%} {lw:>7.1%} {sw:>7.1%} {pnl:>+9.0f} {dd:>7.0%}")

            total_trades += n
            total_pnl += pnl
            total_long += lt
            total_short += st
            total_lw += lw * lt
            total_sw += sw * st
            results[symbol] = report

        # Summary
        print("-" * 70)
        avg_lw = total_lw / total_long if total_long > 0 else 0
        avg_sw = total_sw / total_short if total_short > 0 else 0
        avg_wr = (total_lw + total_sw) / (total_long + total_short) if (total_long + total_short) > 0 else 0
        print(f"{'TOTAL':<15s} {total_trades:>6d} {total_long:>6d} {total_short:>6d} {avg_wr:>7.1%} {avg_lw:>7.1%} {avg_sw:>7.1%} {total_pnl:>+9.0f}")
        print(f"\n  SHORT vs LONG delta: {(avg_sw - avg_lw)*100:+.1f}%")
        print(f"  LONG trades : {total_long} ({total_long/(total_long+total_short)*100:.0f}%)" if (total_long+total_short) > 0 else "")
        print(f"  SHORT trades: {total_short} ({total_short/(total_long+total_short)*100:.0f}%)" if (total_long+total_short) > 0 else "")

        all_results[thr] = results

    # Comparison
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("PERBANDINGAN 0.70 vs 0.62")
        print(f"{'='*70}")
        print(f"{'Coin':<15s} {'Trades':>20s} {'WR':>16s} {'PnL':>20s} {'DD':>16s}")
        print(f"{'':15s} {'0.70':>6s} {'0.62':>6s} {'delta':>7s}  {'0.70':>7s} {'0.62':>7s} {'delta':>6s}  {'0.70':>9s} {'0.62':>9s} {'delta':>9s}  {'0.70':>7s} {'0.62':>7s} {'delta':>6s}")
        print("-" * 70)

        r70 = all_results[0.70]
        r62 = all_results[0.62]
        sum70_t = sum70_pnl = sum62_t = sum62_pnl = sum70_long = sum70_short = sum62_long = sum62_short = 0
        sum70_lw = sum70_sw = sum62_lw = sum62_sw = 0.0

        for coin in coins:
            if coin not in r70 or coin not in r62:
                continue
            a = r70[coin]
            b = r62[coin]
            dt = b["total_trades"] - a["total_trades"]
            da_wr = b["winrate"] - a["winrate"]
            da_pnl = b["total_pnl"] - a["total_pnl"]
            da_dd = b["max_drawdown_pct"] - a["max_drawdown_pct"]
            print(f"{coin:<15s} {a['total_trades']:>6d} {b['total_trades']:>6d} {dt:>+7d}  "
                  f"{a['winrate']:>6.1%} {b['winrate']:>6.1%} {da_wr:>+6.1%}  "
                  f"{a['total_pnl']:>+9.0f} {b['total_pnl']:>+9.0f} {da_pnl:>+9.0f}  "
                  f"{a['max_drawdown_pct']:>6.0%} {b['max_drawdown_pct']:>6.0%} {da_dd:>+6.0%}")

            sum70_t += a["total_trades"]
            sum62_t += b["total_trades"]
            sum70_pnl += a["total_pnl"]
            sum62_pnl += b["total_pnl"]
            sum70_long += a["n_long_trades"]
            sum62_long += b["n_long_trades"]
            sum70_short += a["n_short_trades"]
            sum62_short += b["n_short_trades"]
            sum70_lw += a["win_by_class"]["LONG"] * a["n_long_trades"]
            sum70_sw += a["win_by_class"]["SHORT"] * a["n_short_trades"]
            sum62_lw += b["win_by_class"]["LONG"] * b["n_long_trades"]
            sum62_sw += b["win_by_class"]["SHORT"] * b["n_short_trades"]

        print("-" * 70)
        dt = sum62_t - sum70_t
        da_pnl = sum62_pnl - sum70_pnl
        print(f"{'TOTAL':<15s} {sum70_t:>6d} {sum62_t:>6d} {dt:>+7d}  "
              f"{'':>6s} {'':>6s} {'':>6s}  "
              f"{sum70_pnl:>+9.0f} {sum62_pnl:>+9.0f} {da_pnl:>+9.0f}")

        # Weighted WR
        wl70 = sum70_lw / sum70_long if sum70_long else 0
        ws70 = sum70_sw / sum70_short if sum70_short else 0
        wl62 = sum62_lw / sum62_long if sum62_long else 0
        ws62 = sum62_sw / sum62_short if sum62_short else 0

        print(f"\n  Breakout Trades:")
        print(f"    0.70: LONG={sum70_long} SHORT={sum70_short} (ratio S/L={sum70_short/sum70_long:.2f})" if sum70_long else "")
        print(f"    0.62: LONG={sum62_long} SHORT={sum62_short} (ratio S/L={sum62_short/sum62_long:.2f})" if sum62_long else "")
        print(f"  Weighted WR:")
        print(f"    0.70: LONG={wl70:.1%} SHORT={ws70:.1%} (delta={(ws70-wl70)*100:+.1f}%)")
        print(f"    0.62: LONG={wl62:.1%} SHORT={ws62:.1%} (delta={(ws62-wl62)*100:+.1f}%)")
        print(f"  Total PnL: {sum70_pnl:+.0f} -> {sum62_pnl:+.0f} = {sum62_pnl - sum70_pnl:+.0f}")


if __name__ == "__main__":
    main()
