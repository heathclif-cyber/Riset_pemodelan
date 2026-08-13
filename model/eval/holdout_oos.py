"""
OOS holdout eval — predict on holdout-test labeled data (bukan holdout tersegel).

Usage:
  python -m model.eval.holdout_oos
  python pipeline/model/run_holdout_oos.py --end-date 2026-07-02
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    ALL_COINS,
    FEE_PER_SIDE,
    GUARDIAN_EXIT_THRESHOLD,
    GUARDIAN_MOMENTUM_FLOOR_FRAC,
    HOLDOUT_DIR,
    LIVE_DAILY_LOSS_LIMIT,
    LIVE_MAX_OPEN_POSITIONS,
    MAX_HOLDING_BARS,
    MODEL_DIR,
    MODAL_PER_TRADE,
    OOS_START,
    SLIPPAGE_PER_SIDE,
    SWING_LABEL_MAX_SL,
    SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP,
    TP_SL_FALLBACK_SL,
    TP_SL_FALLBACK_TP,
)
from core.evaluator import apply_portfolio_execution_limits, simulate_trades_swing
from core.utils import ensure_utc_index, setup_logger
from model.eval.constants import DYNAMIC_GUARDIAN_FEATS, GUARDIAN_OUTCOMES, LM, LEV
from model.eval.scorecard import compute_scorecard
from model.regime.thresholds import build_regime_thresholds
from model.stacks.load import load_stack
from pipeline.data.core.fetch import wita_end_to_utc

logger = setup_logger("holdout_oos")


def exit_breakdown(trades: list) -> dict:
    out: dict[str, int] = {}
    for t in trades:
        oc = t.get("outcome", "?")
        out[oc] = out.get(oc, 0) + 1
    return out


def evaluate_coin(
    coin: str,
    oos_end: pd.Timestamp,
    regime_thr: dict,
    lgbm_model,
    lgbm_feats: list,
    g_model,
    g_scaler,
    g_feats: list,
    static_names: list,
    *,
    guardian: bool,
    lgbm_trend_model=None,
    lgbm_trend_feats: list | None = None,
    spot_confirm_enabled: bool = False,
    spot_confirm_threshold: float = 0.60,
    spot_confirm_agree_boost: float = 0.08,
    spot_confirm_opposite_pen: float = 0.35,
    regime_routing_trend_states: tuple = (),
    regime_disable_block_long_states: tuple = (),
    entry_m15_dir: Path | None = None,
    live_parity_exit: bool = False,
    floor_frac: float | None = 0.7,
    floor_intrabar: bool = False,
) -> list:
    feat_path = HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
    regime_path = HOLDOUT_DIR / "labeled" / f"{coin}_regime_h1.parquet"
    if not feat_path.exists():
        return []

    df = pd.read_parquet(feat_path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < oos_end)]
    if len(df) < 30:
        return []

    if regime_path.exists():
        reg_df = pd.read_parquet(regime_path)
        reg_df = ensure_utc_index(reg_df).sort_index()
        df["hmm_regime_enc"] = reg_df["hmm_regime_enc"].reindex(df.index).ffill().fillna(1)

    n = len(df)
    X = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X[:, i] = df[col].ffill().fillna(0).values
        else:
            logger.warning(f"[{coin}] missing LGBM feat {col}")

    proba = lgbm_model.predict_proba(X)
    p0, p2 = proba[:, 0].astype(np.float32), proba[:, 2].astype(np.float32)
    reg = df["hmm_regime_enc"].fillna(1).astype(int).values

    # SPOT-CONFIRM DIHAPUS PERMANEN 2026-07-29 (keputusan user; live "polos" sejak
    # 2026-07-14). Parameter spot_confirm_* dipertahankan di signature supaya caller
    # lama tidak error, tapi tidak ada efeknya lagi. Jangan hidupkan lagi.

    # Regime-model-routing: TRENDING_UP pakai LGBM terpisah (label triple-barrier ATR).
    if regime_routing_trend_states and lgbm_trend_model is not None and lgbm_trend_feats:
        X_t = np.zeros((n, len(lgbm_trend_feats)), dtype=np.float64)
        for i, col in enumerate(lgbm_trend_feats):
            if col in df.columns:
                X_t[:, i] = df[col].ffill().fillna(0).values
        proba_t = lgbm_trend_model.predict_proba(X_t)
        p0_t, p2_t = proba_t[:, 0].astype(np.float32), proba_t[:, 2].astype(np.float32)
        used_trend = np.isin(reg, regime_routing_trend_states)
        p0 = np.where(used_trend, p0_t, p0)
        p2 = np.where(used_trend, p2_t, p2)

    y_pred = np.full(n, LM["FLAT"], np.int32)
    lgbm_conf = np.zeros(n, np.float32)
    for i in range(n):
        tl, ts = regime_thr.get(int(reg[i]), regime_thr[1])
        if p2[i] >= tl and int(reg[i]) not in regime_disable_block_long_states:
            y_pred[i] = LM["LONG"]
            lgbm_conf[i] = p2[i]
        elif p0[i] >= ts:
            y_pred[i] = LM["SHORT"]
            lgbm_conf[i] = p0[i]

    if (y_pred != LM["FLAT"]).sum() == 0:
        return []

    X_g = np.zeros((n, len(static_names)), dtype=np.float64)
    for k, col in enumerate(static_names):
        if col in df.columns:
            X_g[:, k] = df[col].ffill().fillna(0).values

    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)

    # Entry M15@HH:15 (opsional) -- parity waktu eksekusi dgn live. Default (None) tetap
    # isi di close H1 (HH:00), yaitu 15 menit LEBIH AWAL dari live: live baru buka posisi
    # ~HH:15 (generate_signals jalan HH:05). Tanpa ini riset dapat keuntungan waktu gratis
    # yang tidak pernah dimiliki live. df.index = waktu OPEN bar H1, jadi bar tutup di
    # +1h dan live mengisi di +1h15m.
    entry_override = None
    if entry_m15_dir is not None:
        m15_path = Path(entry_m15_dir) / f"{coin}_15m.parquet"
        if m15_path.exists():
            m15 = pd.read_parquet(m15_path)
            m15 = ensure_utc_index(m15).sort_index()
            fill_at = df.index + pd.Timedelta(hours=1) + pd.Timedelta(minutes=15)
            entry_override = m15["open"].reindex(fill_at).to_numpy(dtype=np.float64)
            n_miss = int(np.count_nonzero(~np.isfinite(entry_override)))
            if n_miss:
                logger.warning(f"[{coin}] entry M15 tidak ketemu utk {n_miss}/{n} bar — fallback close H1")
        else:
            logger.warning(f"[{coin}] M15 tidak ada: {m15_path} — fallback close H1")

    # live_parity_exit: reproduksi 2 mekanisme exit live yang TIDAK pernah dioper
    # di sini sebelumnya -- floor FIXED 0.7xTP (STOP-LIMIT begitu TP tersentuh,
    # ganti trailing 0.7xMFE) + cooldown profit-only 1 jam. Default False = perilaku
    # LAMA (persis scorecard.holdout_oos yang sudah dilaporkan), tidak breaking.
    live_parity_kwargs = dict(
        guardian_momentum_floor_frac=GUARDIAN_MOMENTUM_FLOOR_FRAC,
        guardian_momentum_floor_tp_frac=0.0,
        guardian_floor_replace_with_tp=False,
        cooldown_enabled=False,
        cooldown_profit_only=False,
        cooldown_profit_bars=1,
    )
    if live_parity_exit:
        live_parity_kwargs.update(
            cooldown_enabled=True,
            cooldown_profit_only=True,
            cooldown_profit_bars=1,
        )
        if floor_frac is None:
            # Floor DIMATIKAN total -- setelah TP tersentuh, trade jalan terus TANPA
            # jaring pengaman, murni sinyal exit Guardian lain / TIMEOUT.
            live_parity_kwargs.update(
                guardian_momentum_floor_tp_frac=0.0,
                guardian_floor_replace_with_tp=False,
                guardian_momentum_floor_frac=0.0,
            )
        else:
            live_parity_kwargs.update(
                guardian_momentum_floor_tp_frac=floor_frac,
                guardian_floor_replace_with_tp=True,
                guardian_floor_intrabar=floor_intrabar,
            )

    res = simulate_trades_swing(
        y_pred=y_pred,
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        atr=df["atr_14_h1"].values,
        h4_swing_highs=h4_sh,
        h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE,
        leverage=LEV,
        fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP,
        sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=guardian,
        guardian_model=g_model if guardian else None,
        guardian_scaler=g_scaler if guardian else None,
        X_guardian=X_g if guardian else None,
        guardian_feat_cols=g_feats if guardian else [],
        guardian_static_names=static_names if guardian else [],
        guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        lgbm_conf_arr=lgbm_conf,
        entry_price_override=entry_override,
        **live_parity_kwargs,
    )
    trades = res.get("trade_log", res.get("trades", []))
    # +1h: df.index adalah waktu OPEN bar H1, tapi keputusan baru diketahui/dieksekusi
    # SETELAH bar itu CLOSE (persis konvensi live -- sinyal dibuat ~15menit setelah
    # bar close). Tanpa +1h, entry_time/exit_time di trade log ini misleading kalau
    # dibandingkan ke signal_time live (bug off-by-one-hour yg sama spt yg sudah
    # difix di compare_oos_live_signals.py, baru diporting ke sini 2026-07-14).
    _hi = df["high"].values
    _lo = df["low"].values
    _cl = df["close"].values
    for t in trades:
        t["coin"] = coin
        bi, bo = t.get("bar_in"), t.get("bar_out")
        t["entry_time"] = (df.index[bi] + pd.Timedelta(hours=1)) if bi is not None and bi < len(df) else None
        t["exit_time"]  = (df.index[bo] + pd.Timedelta(hours=1)) if bo is not None and bo < len(df) else None
        # mae_pct: dibutuhkan scorecard MAE-aware dashboard (tools/model/leverage_mae_aware.py).
        # Ditambahkan 2026-07-29 -- sebelumnya dihitung skrip ad-hoc yang hilang & tak pernah
        # ter-commit, sehingga angka live tidak bisa direproduksi.
        t["mae_pct"] = (
            _mae_pct(_hi, _lo, _cl, bi, bo, t.get("direction"))
            if bi is not None and bo is not None else None
        )
    return trades


def _mae_pct(high, low, close, bar_in: int, bar_out: int, direction: str) -> float | None:
    """Delegasi ke SSOT metodologi MAE-aware -- jangan duplikasi rumusnya di sini."""
    from tools.model.leverage_mae_aware import compute_mae_pct
    if bar_in is None or bar_out is None or bar_in >= len(close):
        return None
    return round(compute_mae_pct(high, low, close, bar_in, bar_out, direction), 6)


def trades_to_df(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        rows.append({
            "coin": t.get("coin"),
            "direction": t.get("direction"),
            "outcome": t.get("outcome"),
            "net_pnl": t.get("net_pnl"),
            "bar_in": t.get("bar_in"),
            "bar_out": t.get("bar_out"),
            "entry": t.get("entry"),
            "exit": t.get("exit"),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "mae_pct": t.get("mae_pct"),
        })
    return pd.DataFrame(rows)


def print_scorecard(label: str, sc: dict | None) -> None:
    print(f"\n{'='*65}\n  {label}\n{'-'*65}")
    if not sc:
        print("  No trades")
        print("=" * 65)
        return
    for k, fmt in [
        ("trades", "{:,}"),
        ("wr", "{:.1%}"),
        ("pf", "{:.3f}"),
        ("pnl", "${:,.2f}"),
        ("pnl_trade", "${:.4f}"),
        ("max_dd", "${:.2f}"),
        ("long_count", "{}"),
        ("short_count", "{}"),
        ("long_pf", "{:.3f}"),
        ("short_pf", "{:.3f}"),
    ]:
        if k in sc:
            print(f"  {k:14s}: {fmt.format(sc[k])}")
    print("=" * 65)


def run(
    stack_name: str = "fs38_28f",
    *,
    end_date_wita: str = "2026-07-02",
    compare_baseline: bool = False,
    lgbm_run: str | None = None,
    guard_run: str | None = None,
    entry_m15_dir: str | None = None,
    live_parity_exit: bool = False,
    hmm_base: float | None = None,
    hmm_delta: float | None = None,
    portfolio_limits: bool = False,
    max_open_positions: int | None = None,
    daily_loss_limit: int | None = None,
    floor_frac: float | None = 0.7,
    floor_intrabar: bool = False,
) -> Path:
    stack = load_stack(stack_name)
    lgbm_id = lgbm_run or stack.lgbm_run
    guard_id = guard_run or stack.guard_run

    eff_base = hmm_base if hmm_base is not None else stack.hmm_base
    eff_delta = hmm_delta if hmm_delta is not None else stack.hmm_delta
    regime_thr = build_regime_thresholds(eff_base, eff_delta)
    oos_end = pd.Timestamp(wita_end_to_utc(end_date_wita))

    sample = HOLDOUT_DIR / "labeled" / "BTCUSDT_features_v3.parquet"
    if sample.exists():
        idx = ensure_utc_index(pd.read_parquet(sample, columns=[])).index
        data_max = idx.max() + pd.Timedelta(hours=1)
        if oos_end > data_max:
            oos_end = data_max
            logger.info(f"Clipped oos_end to data max {data_max}")

    lgbm_dir = MODEL_DIR / "runs" / lgbm_id
    guard_dir = MODEL_DIR / "runs" / guard_id
    lgbm_model = joblib.load(lgbm_dir / "lgbm.pkl")
    lgbm_feats = json.load(open(lgbm_dir / "features.json", encoding="utf-8"))
    g_model = joblib.load(guard_dir / "guardian.pkl")
    g_scaler = joblib.load(guard_dir / "guardian_scaler.pkl")
    g_feats = json.load(open(guard_dir / "guardian_features.json", encoding="utf-8"))
    static_names = [f for f in g_feats if f not in DYNAMIC_GUARDIAN_FEATS]

    lgbm_trend_model, lgbm_trend_feats = None, None
    if stack.regime_routing_enabled and stack.lgbm_trend_run:
        trend_dir = MODEL_DIR / "runs" / stack.lgbm_trend_run
        lgbm_trend_model = joblib.load(trend_dir / "lgbm.pkl")
        lgbm_trend_feats = json.load(open(trend_dir / "features.json", encoding="utf-8"))

    spot_kwargs = dict(
        spot_confirm_enabled=stack.spot_confirm_enabled,
        spot_confirm_threshold=stack.spot_confirm_threshold,
        spot_confirm_agree_boost=stack.spot_confirm_agree_boost,
        spot_confirm_opposite_pen=stack.spot_confirm_opposite_pen,
        regime_routing_trend_states=stack.regime_routing_trend_states if stack.regime_routing_enabled else (),
        regime_disable_block_long_states=stack.regime_disable_block_long_states if stack.regime_disable_enabled else (),
        entry_m15_dir=Path(entry_m15_dir) if entry_m15_dir else None,
        live_parity_exit=live_parity_exit,
        floor_frac=floor_frac,
        floor_intrabar=floor_intrabar,
    )

    print(f"\n{'='*65}")
    print(f"  OOS HOLDOUT — {lgbm_id} + HMM {eff_base}/{eff_delta}"
          f"{' (override dari stack default ' + str(stack.hmm_base) + '/' + str(stack.hmm_delta) + ')' if hmm_base is not None or hmm_delta is not None else ''}"
          f" + {guard_id}")
    print(f"  regime_routing={stack.regime_routing_enabled} regime_disable={stack.regime_disable_enabled} (spot_confirm: DIHAPUS PERMANEN 2026-07-29)")
    print(f"  entry={'M15@HH:15 (parity live)' if entry_m15_dir else 'close H1 @HH:00'}")
    print(f"  exit={'LIVE PARITY (floor FIXED 0.7xTP + cooldown 1h profit-only)' if live_parity_exit else 'LEGACY (trailing floor 0.7xMFE, no cooldown)'}")
    eff_max_pos = max_open_positions if max_open_positions is not None else LIVE_MAX_OPEN_POSITIONS
    eff_daily_lim = daily_loss_limit if daily_loss_limit is not None else LIVE_DAILY_LOSS_LIMIT
    print(f"  portfolio_limits={'ON (max_open_positions=' + str(eff_max_pos) + ', daily_loss_limit=' + str(eff_daily_lim) + ')' if portfolio_limits else 'OFF (unlimited concurrent, no daily-loss circuit breaker)'}")
    print(f"  Window: {OOS_START} .. < {oos_end}")
    print(f"{'='*65}")

    all_no, all_g = [], []
    per_coin = []
    for coin in ALL_COINS:
        t_no = evaluate_coin(
            coin, oos_end, regime_thr, lgbm_model, lgbm_feats,
            g_model, g_scaler, g_feats, static_names, guardian=False,
            lgbm_trend_model=lgbm_trend_model, lgbm_trend_feats=lgbm_trend_feats,
            **spot_kwargs,
        )
        t_g = evaluate_coin(
            coin, oos_end, regime_thr, lgbm_model, lgbm_feats,
            g_model, g_scaler, g_feats, static_names, guardian=True,
            lgbm_trend_model=lgbm_trend_model, lgbm_trend_feats=lgbm_trend_feats,
            **spot_kwargs,
        )
        all_no.extend(t_no)
        all_g.extend(t_g)

    rejected_g: list = []
    if portfolio_limits:
        all_no, _ = apply_portfolio_execution_limits(
            all_no, max_open_positions=eff_max_pos, daily_loss_limit=eff_daily_lim,
        )
        all_g, rejected_g = apply_portfolio_execution_limits(
            all_g, max_open_positions=eff_max_pos, daily_loss_limit=eff_daily_lim,
        )
        n_max_pos = sum(1 for t in rejected_g if t.get("_reject_reason") == "max_open_positions")
        n_daily = sum(1 for t in rejected_g if t.get("_reject_reason") == "daily_loss_limit")
        logger.info(
            f"portfolio_limits: {len(rejected_g)} trade ditolak dari full-stack "
            f"(max_open_positions={n_max_pos}, daily_loss_limit={n_daily})"
        )

    # per_coin dihitung SETELAH portfolio_limits (kalau aktif) supaya konsisten dgn
    # scorecard full-stack yang dilaporkan -- bukan dari kandidat mentah per-koin.
    per_coin_trades: dict = {}
    for t in all_g:
        per_coin_trades.setdefault(t.get("coin"), []).append(t)
    for coin in ALL_COINS:
        tg = per_coin_trades.get(coin, [])
        if tg:
            sc = compute_scorecard(tg)
            per_coin.append({"coin": coin, **sc})
            logger.info(f"[{coin}] trades={sc['trades']} WR={sc['wr']:.1%} PF={sc['pf']:.3f} PnL=${sc['pnl']:.2f}")

    sc_no = compute_scorecard(all_no)
    sc_g = compute_scorecard(all_g)
    print_scorecard("Holdout tanpa Guardian", sc_no)
    print_scorecard("Holdout + Guardian (full stack)", sc_g)
    if portfolio_limits:
        print(f"  (portfolio_limits: {len(rejected_g)} trade ditolak -- max_open_positions={n_max_pos}, daily_loss_limit={n_daily})")

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "stack_name": stack.name,
        "window": f"{OOS_START} s.d. {oos_end}",
        "end_date_wita": end_date_wita,
        "stack": f"{lgbm_id} + HMM {eff_base}/{eff_delta} + {guard_id}",
        "no_guardian": sc_no,
        "with_guardian": sc_g,
        "exit_reasons": exit_breakdown(all_g),
        "per_coin": per_coin,
        "portfolio_limits": {
            "enabled": portfolio_limits,
            "max_open_positions": eff_max_pos,
            "daily_loss_limit": eff_daily_lim,
            "n_rejected": len(rejected_g),
            "n_rejected_max_open_positions": n_max_pos if portfolio_limits else 0,
            "n_rejected_daily_loss_limit": n_daily if portfolio_limits else 0,
        } if portfolio_limits else {"enabled": False},
    }

    sc_path = guard_dir / "oos_holdout_full_scorecard.json"
    with open(sc_path, "w", encoding="utf-8") as f:
        flat = {
            "window": payload["window"],
            **({k: v for k, v in (sc_g or {}).items()}),
            "exit_reasons": payload["exit_reasons"],
            "stack_name": stack.name,
            "created": payload["created"],
        }
        json.dump(flat, f, indent=2)

    trades_to_df(all_g).to_csv(guard_dir / "oos_holdout_full_trades_detail.csv", index=False)
    pd.DataFrame(per_coin).to_csv(guard_dir / "oos_holdout_full_percoin.csv", index=False)
    with open(guard_dir / "oos_holdout_h4closed_full.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if compare_baseline:
        baseline_sc = MODEL_DIR / "runs" / f"{guard_id}_bak_20260703_1505" / "oos_holdout_full_scorecard.json"
        if baseline_sc.exists():
            base = json.load(open(baseline_sc, encoding="utf-8"))
            g = sc_g or {}
            print(f"\n{'='*65}\n  vs Pre-H4 baseline ({baseline_sc.parent.name})\n{'-'*65}")
            for k in ("trades", "wr", "pf", "pnl", "pnl_trade", "max_dd", "long_pf", "short_pf"):
                if k in g and k in base:
                    b, n = base[k], g[k]
                    d = n - b if isinstance(n, (int, float)) else None
                    suffix = f" ({d:+.4f}, {d/b*100:+.1f}%)" if d is not None and b else ""
                    print(f"  {k:12s}: {b} -> {n}{suffix}")

    logger.info(f"Saved -> {sc_path}")
    return sc_path


def main() -> int:
    ap = argparse.ArgumentParser(description="OOS holdout eval")
    ap.add_argument("--stack", default="fs38_28f")
    ap.add_argument("--end-date", default="2026-07-02", help="Akhir periode WITA YYYY-MM-DD")
    ap.add_argument("--compare-baseline", action="store_true")
    ap.add_argument("--lgbm-run", default=None)
    ap.add_argument("--guard-run", default=None)
    ap.add_argument("--entry-m15-dir", default=None,
                    help="Direktori parquet M15 ({coin}_15m.parquet). Kalau diisi, entry pakai "
                         "harga M15 HH:15 (parity waktu eksekusi live), bukan close H1 HH:00.")
    ap.add_argument("--live-parity-exit", action="store_true",
                    help="Pakai mekanisme exit PERSIS live: Guardian floor FIXED 0.7xTP_pnl "
                         "(STOP-LIMIT begitu TP tersentuh, bukan trailing 0.7xMFE) + cooldown "
                         "profit-only 1 jam. Default False = perilaku lama (scorecard.holdout_oos "
                         "yang sudah dilaporkan).")
    ap.add_argument("--hmm-base", type=float, default=None,
                    help="Override base threshold HMM per-state (default: stack.hmm_base).")
    ap.add_argument("--hmm-delta", type=float, default=None,
                    help="Override delta threshold HMM per-state (default: stack.hmm_delta).")
    ap.add_argument("--portfolio-limits", action="store_true",
                    help="Terapkan max_open_positions & daily_loss_limit PERSIS live "
                         "(execution.py::place_entry) -- backtest defaultnya mengasumsikan "
                         "modal tanpa batas & tanpa circuit-breaker rugi harian.")
    ap.add_argument("--max-open-positions", type=int, default=None,
                    help="Override cap posisi konkuren (default: config.LIVE_MAX_OPEN_POSITIONS=10).")
    ap.add_argument("--daily-loss-limit", type=int, default=None,
                    help="Override circuit-breaker rugi harian WITA (default: config.LIVE_DAILY_LOSS_LIMIT=8).")
    ap.add_argument("--floor-frac", default="0.7",
                    help="guardian_momentum_floor_tp_frac saat --live-parity-exit aktif "
                         "(default: 0.7, live skrg). 'off' = matikan floor total.")
    ap.add_argument("--floor-intrabar", action="store_true",
                    help="Model floor seperti eksekusi live (STOP-LIMIT resting: trigger saat "
                         "wick sentuh floor_price, exit DI floor_price).")
    args = ap.parse_args()
    floor_frac = None if args.floor_frac.lower() == "off" else float(args.floor_frac)
    run(
        args.stack,
        end_date_wita=args.end_date,
        compare_baseline=args.compare_baseline,
        lgbm_run=args.lgbm_run,
        guard_run=args.guard_run,
        entry_m15_dir=args.entry_m15_dir,
        live_parity_exit=args.live_parity_exit,
        hmm_base=args.hmm_base,
        hmm_delta=args.hmm_delta,
        portfolio_limits=args.portfolio_limits,
        max_open_positions=args.max_open_positions,
        daily_loss_limit=args.daily_loss_limit,
        floor_frac=floor_frac,
        floor_intrabar=args.floor_intrabar,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())