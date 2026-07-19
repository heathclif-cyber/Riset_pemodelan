"""
pipeline/03_engineer.py — Fase 03: Feature Engineering & Labeling

Jalankan:
  python pipeline/03_engineer.py                    # engineer training coins
  python pipeline/03_engineer.py --new              # engineer new coins
  python pipeline/03_engineer.py --all              # engineer semua koin
  python pipeline/03_engineer.py --coins SOLUSDT    # koin spesifik
"""

import argparse
import json
import multiprocessing
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline._bootstrap import setup_path_from_file
ROOT = setup_path_from_file(__file__)

from config import (
    TRAINING_COINS, NEW_COINS, ALL_COINS, SYMBOL_MAP,
    PROC_DIR, LABEL_DIR, REPORT_DIR, HOLDOUT_DIR,
    SWING_LABEL_MAX_HOLD, SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    FEATURE_COLS_V3,
    VP_WINDOW, VP_BINS, SWING_LOOKBACK, FVG_MIN_GAP_ATR,
    SWING_ROLLING_BARS,
    HMM_N_STATES,
)
from core.regime import encode_regime
from core.utils import setup_logger, ensure_utc_index
from core.features import (
    engineer_features,
    MKT_WIDE_FEATURE_COLS,
    COIN_SYNC_FEATURE_COLS,
    cache_market_panel,
    merge_market_panel,
    augment_coin_sync_features,
)
from core.inference_parity import (
    apply_training_parity_overrides,
    strip_real_positioning,
)

logger = setup_logger("03_engineer")

_IN_DIR  = PROC_DIR
_OUT_DIR = LABEL_DIR
_INFERENCE_PARITY = False
_MKT_PANEL_PATH = PROC_DIR / "_market_panel_h1.parquet"

# Filter thresholds (disesuaikan jika belum ada di config)
LONG_MAX_PRICE_IN_RANGE = 1.0
SHORT_MIN_PRICE_IN_RANGE = 0.0


def validate_features(df: pd.DataFrame) -> list[str]:
    """Memastikan semua fitur V3 ada di DataFrame."""
    missing = [col for col in FEATURE_COLS_V3 if col not in df.columns]
    return missing


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, str(path), compression="snappy")


def engineer_symbol(symbol: str) -> dict[str, Any]:
    report = {"symbol": symbol}
    in_path = _IN_DIR / f"{symbol}_clean.parquet"
    
    if not in_path.exists():
        logger.error(f"[{symbol}] File clean tidak ditemukan: {in_path}")
        report["status"] = "missing_input"
        return report

    logger.info(f"[{symbol}] Starting feature engineering v3...")
    df = pd.read_parquet(in_path)
    df = ensure_utc_index(df)
    if _INFERENCE_PARITY:
        df = strip_real_positioning(df)

    symbol_id = SYMBOL_MAP.get(symbol, -1)

    try:
        # ★ Panggilan fungsi dengan parameter V3 (Base H1)
        feat_df = engineer_features(
            df                       = df,
            symbol                   = symbol,
            symbol_id                = symbol_id,
            max_hold                 = SWING_LABEL_MAX_HOLD,   # ★ BARU
            min_rr                   = SWING_LABEL_MIN_RR,     # ★ BARU
            min_tp_atr               = SWING_LABEL_MIN_TP,     # ★ BARU
            max_sl_atr               = SWING_LABEL_MAX_SL,     # ★ BARU
            vp_window                = VP_WINDOW,
            vp_bins                  = VP_BINS,
            swing_lookback           = SWING_LOOKBACK,
            fvg_min_gap              = FVG_MIN_GAP_ATR,
            long_max_price_in_range  = LONG_MAX_PRICE_IN_RANGE,
            short_min_price_in_range = SHORT_MIN_PRICE_IN_RANGE,
            add_label                = True,
        )
        if _INFERENCE_PARITY:
            feat_df = apply_training_parity_overrides(feat_df)

        if _MKT_PANEL_PATH.exists():
            try:
                mkt_df = pd.read_parquet(_MKT_PANEL_PATH)
                if mkt_df.index.tz is None:
                    mkt_df.index = mkt_df.index.tz_localize("UTC")
                feat_df = merge_market_panel(feat_df, mkt_df)
                # FIX 2026-07-19: coin_mkt_sync_24h dkk (COIN_SYNC_FEATURE_COLS) dulu
                # cuma ditempel via script terpisah (join_sync_features_incremental.py /
                # join_sync_features_holdout.py), TIDAK PERNAH bagian pipeline standar --
                # tiap kali features_v3 di-regenerate dari nol (khususnya holdout-test,
                # sering di-refresh utk extend window OOS), kolom ini hilang total & auto
                # ke-nol-kan di evaluate_coin() (bukan NaN, kolomnya tidak ada) -- regresi
                # bug yg sama 2x (2026-07-13, 2026-07-19). Diintegrasikan di sini biar
                # permanen, tidak bisa lupa lagi. Lihat EXPERIMENTS.md 2026-07-19.
                feat_df = augment_coin_sync_features(feat_df, mkt_df)
                logger.info(
                    f"[{symbol}] Market-wide + sync features merged "
                    f"({len(MKT_WIDE_FEATURE_COLS)}+{len(COIN_SYNC_FEATURE_COLS)} cols)"
                )
            except Exception as me:
                logger.warning(f"[{symbol}] Market panel merge gagal: {me}")

        # ── Merge yfinance ETF & Macro Features (FREE) ─────────────────────────
        try:
            macro_dir = ROOT / "data" / "macro"

            # ETF Netflow (Global Macro) — yfinance, GRATIS
            etf_path = macro_dir / "etf_flow_btc.parquet"
            if etf_path.exists():
                etf_df = pd.read_parquet(etf_path)
                etf_df.index = pd.to_datetime(etf_df.index, utc=True)
                etf_h1 = etf_df.reindex(etf_df.index.union(feat_df.index)).sort_index().ffill().reindex(feat_df.index)
                for c in ["etf_total_change_usd", "etf_gbtc_change_usd"]:
                    if c in etf_h1.columns:
                        feat_df[c] = etf_h1[c].shift(24)  # T-1 lag: 24 H1 bars = 1 trading day

            # Fill missing ETF features safely for historical data before 2024
            for c in ["etf_total_change_usd", "etf_gbtc_change_usd"]:
                if c in feat_df.columns:
                    feat_df[c] = feat_df[c].ffill().fillna(0.0)

            logger.info(f"[{symbol}] yfinance ETF features merged successfully.")
        except Exception as ce:
            logger.warning(f"[{symbol}] yfinance merge failed: {ce}")
        
        # Validasi menggunakan FEATURE_COLS_V3
        missing_cols = validate_features(feat_df)
        if missing_cols:
            logger.warning(f"[{symbol}] Ada {len(missing_cols)} fitur V3 yang hilang: {missing_cols}")
            report["missing_features"] = missing_cols

        # ── Merge HMM regime labels (jika sudah di-generate oleh 03b) ──────
        regime_path = _OUT_DIR / f"{symbol}_regime_h1.parquet"
        if regime_path.exists():
            try:
                reg_df = pd.read_parquet(regime_path)
                if not isinstance(reg_df.index, pd.DatetimeIndex):
                    reg_df.index = pd.to_datetime(reg_df.index, utc=True)
                if reg_df.index.tz is None:
                    reg_df.index = reg_df.index.tz_localize("UTC")
                # Align index ke feat_df
                feat_df = feat_df.join(reg_df[["hmm_regime", "hmm_regime_enc"]], how="left")
                feat_df["hmm_regime_enc"] = feat_df["hmm_regime_enc"].fillna(1).astype("int32")
                feat_df["hmm_regime"]     = feat_df["hmm_regime"].fillna("RANGING_LOW_VOL")
                logger.info(f"[{symbol}] HMM regime merged — dist: {feat_df['hmm_regime'].value_counts().to_dict()}")
            except Exception as e:
                logger.warning(f"[{symbol}] HMM regime merge gagal: {e} — skip")
        else:
            logger.info(f"[{symbol}] Regime HMM tidak dimuat (opsional) — skip")

        # Amankan kolom-kolom V3 yang valid + kolom label + swing levels + regime
        extra_cols = ["label", "h4_swing_high", "h4_swing_low"]
        if "hmm_regime_enc" in feat_df.columns:
            extra_cols += ["hmm_regime", "hmm_regime_enc"]
        mkt_present = [c for c in MKT_WIDE_FEATURE_COLS if c in feat_df.columns]
        sync_present = [c for c in COIN_SYNC_FEATURE_COLS if c in feat_df.columns]
        cols_to_keep = [c for c in FEATURE_COLS_V3 if c in feat_df.columns] + mkt_present + sync_present + extra_cols
        feat_df = feat_df[cols_to_keep]

        # Buang NaN baris awal akibat rolling windows
        initial_len = len(feat_df)
        CRITICAL_COLS = ["open", "high", "low", "close", "volume", "atr_14_h1", "rsi_6", "label"]
        critical_present = [c for c in CRITICAL_COLS if c in feat_df.columns]
        feat_df = feat_df.dropna(subset=critical_present)
        dropped = initial_len - len(feat_df)

        # Output ke _features_v3.parquet
        out_path = _OUT_DIR / f"{symbol}_features_v3.parquet"
        _save(feat_df, out_path)

        n_features = len([c for c in FEATURE_COLS_V3 if c in feat_df.columns])
        n_meta = len(feat_df.columns) - n_features

        report.update({
            "status": "success",
            "rows_output": len(feat_df),
            "rows_dropped_nan": dropped,
            "columns": len(feat_df.columns),
            "output_file": str(out_path)
        })
        logger.info(
            f"[{symbol}] Saved {len(feat_df):,} rows × {len(feat_df.columns)} cols "
            f"({n_features} active features + {n_meta} metadata) to {out_path.name}"
        )

    except Exception as e:
        logger.error(f"[{symbol}] Error during engineering: {e}")
        logger.error(traceback.format_exc())
        report["status"] = "error"
        report["error"] = str(e)

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--training", action="store_true")
    group.add_argument("--new",  action="store_true")
    group.add_argument("--all",  action="store_true")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL")
    parser.add_argument(
        "--rebuild-panel", action="store_true", dest="rebuild_panel",
        help="Paksa rebuild _market_panel_h1.parquet (file bersama, dipakai SEMUA koin) "
             "dari daftar --coins saat ini. WAJIB kalau --coins dipakai dan panel belum "
             "mencakup koin ini -- tanpa flag ini, --coins TIDAK akan menyentuh panel.")
    parser.add_argument("--holdout-test", action="store_true", dest="holdout_test",
                        help="Engineer holdout-test data (data/holdout-test/)")
    return parser.parse_args()


def main():
    global _IN_DIR, _OUT_DIR, _INFERENCE_PARITY, _MKT_PANEL_PATH
    args = parse_args()

    if args.holdout_test:
        _IN_DIR  = HOLDOUT_DIR / "processed"
        _OUT_DIR = HOLDOUT_DIR / "labeled"
        # FIX 2026-07-04: panel market WAJIB dibangun dari data holdout & di-cache
        # di dir holdout. Sebelumnya selalu pakai panel training (< cutoff) yang
        # index-nya tidak overlap dgn holdout -> merge menghasilkan mkt_* NaN semua.
        _MKT_PANEL_PATH = HOLDOUT_DIR / "processed" / "_market_panel_h1.parquet"
        # KOREKSI 2026-07-03: _INFERENCE_PARITY dulu True (strip OI/LSR real, pakai
        # synthetic) dgn asumsi live juga synthetic ("training_parity"). Asumsi ini
        # usang -- training (labeled/labeled_opt2) sudah pakai data asli sejak lama,
        # dan live sekarang positioning_mode="live" (lihat inference_config.json,
        # FEATURE_AUDIT.md). Holdout-test WAJIB ikut kebijakan yang sama: utamakan
        # data asli, agar OOS test apples-to-apples dgn live sungguhan.
        _INFERENCE_PARITY = False
        logger.info(f"Holdout-test mode: in={_IN_DIR}  out={_OUT_DIR} (inference_parity=OFF, data asli)")

    if args.new:
        coins = NEW_COINS
    elif args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Panel pasar (_market_panel_h1.parquet) dipakai BERSAMA oleh semua koin (mkt_breadth,
    # btc_ret_24h, dll). Rebuild dari subset --coins sempit akan merusak fitur cross-coin
    # utk SEMUA koin lain (insiden 2026-07-11/12). --coins TIDAK boleh rebuild diam-diam --
    # wajib --rebuild-panel eksplisit, atau pakai --all/--new/default (list penuh by design).
    is_narrow_subset = bool(args.coins) and not args.all and not args.new
    if is_narrow_subset and not args.rebuild_panel and _MKT_PANEL_PATH.exists():
        logger.info(
            f"--coins dipakai ({len(coins)} koin) -- panel bersama {_MKT_PANEL_PATH.name} "
            f"TIDAK di-rebuild (reuse existing). Pakai --rebuild-panel kalau memang mau "
            f"membangun ulang panel dari daftar koin ini."
        )
    else:
        if is_narrow_subset and args.rebuild_panel:
            logger.warning(
                f"--rebuild-panel dgn --coins ({len(coins)} koin) -- panel bersama akan "
                f"DITULIS ULANG hanya dari {coins}. Pastikan ini daftar koin PENUH yang "
                f"dimaksud, bukan subset -- semua koin lain ikut kepakai panel ini."
            )
        try:
            _panel_kw = {}
            if args.holdout_test:
                _panel_kw = {"proc_dir": HOLDOUT_DIR / "processed", "label_dir": HOLDOUT_DIR / "labeled"}
            cache_market_panel(coins, _MKT_PANEL_PATH, **_panel_kw)
            logger.info(f"Market panel cached -> {_MKT_PANEL_PATH} ({len(coins)} coins)")
        except Exception as e:
            logger.warning(f"Market panel build skipped: {e}")

    # Parallelisasi — tiap koin independen (baca/tulis file terpisah)
    # n_workers: sisakan 1 CPU untuk sistem, maksimal sejumlah koin
    # holdout-test: force n_workers=1 karena globals tidak diwariskan ke child processes
    n_workers = 1 if args.holdout_test else max(1, min(len(coins), multiprocessing.cpu_count() - 1))
    logger.info(f"Processing {len(coins)} coins | workers={n_workers} | cpu_count={multiprocessing.cpu_count()}")

    if n_workers <= 1:
        results = [engineer_symbol(s) for s in coins]
    else:
        # spawn context untuk kompatibilitas Windows
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(engineer_symbol, coins)

    full_report = {"symbols": {s: r for s, r in zip(coins, results)}}
    report_path = REPORT_DIR / "engineering_v3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()