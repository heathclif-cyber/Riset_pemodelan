"""
pipeline/05a_generate_momentum_labels.py — Generate Momentum Labels untuk LSTM H1 (Multi-Head Corrector + Booster)

REKOMENDASI PIPELINE LSTM SAAT INI (2026-06):

**PENTING — Ada dua pendekatan LSTM:**

1. **Untuk cascade_v2.5_hybrid** (revival spirit v2 yang lebih perform di live):
   - LSTM pakai **fitur yang sama dengan LGBM** (FEATURE_COLS_V3).
   - Jangan pakai jalur 05a/05b/05c ini.

2. **Untuk eksperimen Advanced Momentum Detector** (v4 series):
   - Pakai jalur ini (05a → 05b → 05c dengan momentum labels + fitur trajectory khusus).

Lihat `pipeline/archive/README.md` untuk penjelasan lengkap + perbandingan.

JANGAN pakai script lama di folder archive/ kecuali untuk audit.

Tujuan Utama (Update 2026-06):
  Membuat label yang cocok untuk LSTM berperan sebagai **residual corrector + momentum booster**
  bagi LGBM, bukan sebagai standalone classifier.

  Fokus: Membantu memperbaiki dua masalah utama LGBM di live:
  1. Terlalu banyak FLAT padahal ada momentum sequence yang sedang terbentuk (missed opportunity).
  2. Memberi sinyal arah yang salah karena tidak melihat pola temporal dengan baik.

Filosofi Labeling Baru:
  - Head 1 (strong_up_label / momentum_continuation): Label situasi di mana sequence menunjukkan
    momentum naik sedang terbentuk/built-up dengan kualitas baik, dan ke depan cenderung menghasilkan
    pergerakan menguntungkan. Cocok untuk boost/koreksi prob LONG LGBM.
  - Head 2 (upward_exhaustion_label): Label situasi di mana momentum naik mulai kehilangan tenaga
    (exhaustion) dan berisiko reversal/pullback. Cocok untuk boost/koreksi prob SHORT atau
    mengurangi eksposur LONG LGBM.

  Label dibuat agar LSTM belajar mendeteksi "kapan LGBM kemungkinan besar salah atau terlalu konservatif"
  karena kelemahannya dalam melihat sequence.

Referensi:
  - Lopez de Prado — path-aware labeling
  - Hybrid ML models di quant trading (tree model sebagai base + sequence model sebagai corrector/residual learner)

Output:
  LABEL_DIR/{COIN}_momentum_labels.parquet

  Kolom utama untuk Multi-Head LSTM:
    - strong_up_label           : Binary (1 = Momentum continuation ke atas berkualitas)
    - upward_exhaustion_label   : Binary (1 = Exhaustion momentum naik / risiko reversal)

Jalankan:
  python pipeline/05a_generate_momentum_labels.py --all
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, TRAIN_CUTOFF_DATE,
)
from core.utils import setup_logger

logger = setup_logger("05a_momentum_labels")


# ─── Core labeling function ───────────────────────────────────────────────────

def generate_momentum_labels(
    close:        pd.Series,
    high:         pd.Series,
    low:          pd.Series,
    atr:          pd.Series,
    n:            int   = 12,
    min_move:     float = 0.25,
    majority_thr: float = 0.55,
) -> pd.Series:
    """
    Hybrid momentum label: majority direction + minimum magnitude filter.

    Args:
        close:        H1 close price series
        high:         H1 high price series
        low:          H1 low price series
        atr:          H1 ATR-14 series
        n:            jumlah bar H1 ke depan (horizon)
        min_move:     minimum gerak sebagai kelipatan ATR (default 0.25)
        majority_thr: fraksi minimum bar yang harus searah
                      LONG  jika >= majority_thr × N bar naik
                      SHORT jika <= (1 - majority_thr) × N bar naik
                      Default 0.55 → N=12: ≥7/12 LONG, ≤5/12 SHORT
                      Sebelumnya 0.625 → N=12: ≥8/12 — terlalu ketat → FLAT 70%

    Returns:
        Series dengan label LONG/FLAT/SHORT per bar H1
    """
    length    = len(close)
    labels    = np.full(length, "FLAT", dtype=object)
    close_arr = close.values
    atr_arr   = atr.values

    long_bar_thr  = math.ceil(n * majority_thr)           # N=12, 0.55 → 7
    short_bar_thr = math.floor(n * (1.0 - majority_thr))  # N=12, 0.55 → 5

    for i in range(length - n):
        atr_i = atr_arr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        c0 = close_arr[i]
        if np.isnan(c0):
            continue

        # Hitung arah setiap bar ke depan
        bar_rets  = close_arr[i+1 : i+n+1] - close_arr[i : i+n]
        up_count  = int(np.sum(bar_rets > 0))

        # Total return dari entry ke N bar ke depan
        total_ret = close_arr[i+n] - c0
        min_move_abs = min_move * atr_i

        if up_count >= long_bar_thr and total_ret > min_move_abs:
            labels[i] = "LONG"
        elif up_count <= short_bar_thr and total_ret < -min_move_abs:
            labels[i] = "SHORT"
        # else: FLAT (noise, sideways, reversal di tengah/akhir)

    # Tail N bar tidak punya forward window → paksa FLAT
    labels[-n:] = "FLAT"

    return pd.Series(labels, index=close.index, name="momentum_label")


# ─── Strong Momentum Quantile Labeling (Symmetric per-coin) ───────────────────

def generate_strong_momentum_quantile_labels(
    close: pd.Series,
    n: int = 12,
    strong_quantile: float = 0.75,
    weak_quantile: float = 0.25,
    min_majority: float = 0.55,
) -> pd.Series:
    """
    Generate 3-class strong momentum labels using per-coin quantiles.

    Tujuan: Membuat label yang adaptif terhadap karakteristik masing-masing koin
    (volatile coin seperti PEPE butuh move lebih besar daripada BTC untuk disebut "strong").

    Logic (Symmetric):
    - Hitung future_return = close[i+n] - close[i] untuk setiap bar
    - Hitung quantile dari distribusi future_return di coin tersebut (training period)
    - Strong_LONG  : future_return > upper_quantile (misal q75) DAN majority up bars
    - Strong_SHORT : future_return < lower_quantile (misal q25) DAN majority down bars
    - Normal       : sisanya

    Keuntungan vs fixed threshold:
    - Adil antar koin (high-vol vs low-vol)
    - Masih menangkap strong momentum di BOTH direction (sesuai kebutuhan user)

    Args:
        close: H1 close price series
        n: horizon (default 12)
        strong_quantile: quantile untuk upper threshold Strong_LONG (default 0.75 = top 25%)
        weak_quantile: quantile untuk lower threshold Strong_SHORT (default 0.25 = bottom 25%)
        min_majority: minimum fraction of bars in the correct direction

    Returns:
        Series dengan label "Strong_LONG" / "Normal" / "Strong_SHORT"
    """
    length = len(close)
    labels = np.full(length, "Normal", dtype=object)
    close_arr = close.values

    # Hitung future return untuk semua bar yang valid
    future_rets = np.full(length, np.nan)
    up_ratios = np.full(length, np.nan)

    for i in range(length - n):
        c0 = close_arr[i]
        if np.isnan(c0):
            continue

        future_rets[i] = close_arr[i + n] - c0

        # Hitung majority direction
        bar_rets = close_arr[i+1 : i+n+1] - close_arr[i : i+n]
        up_count = np.sum(bar_rets > 0)
        up_ratios[i] = up_count / n

    # Hitung quantile dari future return (hanya bar yang valid)
    valid_mask = ~np.isnan(future_rets)
    valid_rets = future_rets[valid_mask]

    if len(valid_rets) < 100:
        logger.warning("Data terlalu sedikit untuk menghitung quantile. Semua label = Normal.")
        return pd.Series(labels, index=close.index, name="strong_momentum_label")

    lower_thr = np.quantile(valid_rets, weak_quantile)
    upper_thr = np.quantile(valid_rets, strong_quantile)

    long_bar_thr = math.ceil(n * min_majority)
    short_bar_thr = math.floor(n * (1.0 - min_majority))

    for i in range(length - n):
        if np.isnan(future_rets[i]) or np.isnan(up_ratios[i]):
            continue

        ret = future_rets[i]
        up_ratio = up_ratios[i]

        if ret > upper_thr and up_ratio >= min_majority:
            labels[i] = "Strong_LONG"
        elif ret < lower_thr and (1.0 - up_ratio) >= min_majority:
            labels[i] = "Strong_SHORT"
        # else: Normal

    # Tail N bar
    labels[-n:] = "Normal"

    logger.info(
        f"Strong momentum quantile thresholds | lower={lower_thr:.6f} | upper={upper_thr:.6f} "
        f"(q{weak_quantile:.2f} / q{strong_quantile:.2f})"
    )

    return pd.Series(labels, index=close.index, name="strong_momentum_label")


# ─── Multi-Head Labeling for LSTM (2 Tugas Utama) ─────────────────────────────

def generate_strong_up_label(
    close: pd.Series,
    n: int = 12,
    min_future_return_pct: float = 0.007,
    min_majority: float = 0.52,
    lookback_accel: int = 8,
) -> pd.Series:
    """
    Head 1: Momentum Continuation Strength (Corrector + Booster untuk LONG)

    Filosofi (sesuai kebutuhan proyek Juni 2026):
    - Label 1 ketika sequence terbaru menunjukkan momentum naik sedang terbentuk
      dengan kualitas yang layak (ada akselerasi / conviction yang membangun).
    - Dan ke depan menghasilkan pergerakan menguntungkan yang meaningful.
    - Tujuannya: Mendeteksi situasi di mana LGBM cenderung masih FLAT atau kurang yakin,
      padahal sequence H1 menunjukkan momentum yang sedang terbentuk dan berpotensi berlanjut.

    Cocok untuk:
    - Boost prob LONG LGBM
    - Koreksi LGBM ketika ia masih ragu atau salah arah karena kelemahan sequence

    Bukan pure "strong move classifier", melainkan "kapan momentum sequence layak untuk di-boost/koreksi".
    """
    length = len(close)
    labels = np.zeros(length, dtype=np.int8)
    close_arr = close.values

    for i in range(length - n):
        # Future outcome
        future_ret = close_arr[i + n] - close_arr[i]
        if future_ret / close_arr[i] < min_future_return_pct:
            continue

        # Continuation quality ke depan
        bar_rets = close_arr[i+1:i+n+1] - close_arr[i:i+n]
        up_ratio = np.sum(bar_rets > 0) / n
        if up_ratio < min_majority:
            continue

        # Building momentum di sequence sebelum bar i (backward looking)
        if i >= lookback_accel:
            recent_rets = close_arr[i - lookback_accel + 1 : i + 1] - close_arr[i - lookback_accel : i]
            early = np.mean(recent_rets[: lookback_accel // 2])
            late = np.mean(recent_rets[lookback_accel // 2 :])
            if late <= early * 0.5:  # tidak ada akselerasi / conviction
                continue

        labels[i] = 1

    labels[-n:] = 0
    return pd.Series(labels, index=close.index, name="strong_up_label")


def generate_upward_exhaustion_label(
    close: pd.Series,
    high: pd.Series,
    atr: pd.Series,
    n: int = 12,
    exhaustion_lookahead: int = 8,
    pullback_atr: float = 0.9,
) -> pd.Series:
    """
    Head 2: Upward Exhaustion / Reversal Risk (untuk koreksi SHORT atau kurangi LONG)

    Filosofi (sesuai kebutuhan proyek):
    - Label 1 ketika setelah ada pergerakan naik yang cukup, muncul tanda kelelahan
      di sequence terbaru (gagal lanjut naik, divergence, absorption, dll).
    - Dan ke depan cenderung menghasilkan pullback atau pergerakan buruk.
    - Tujuannya: Menangkap situasi di mana LGBM masih condong bullish / over-confident,
      padahal momentum naik sebenarnya sudah habis dan berisiko reversal.

    Sangat berguna untuk mengoreksi kesalahan LGBM yang terlalu lama bertahan di LONG.
    """
    length = len(close)
    labels = np.zeros(length, dtype=np.int8)
    close_arr = close.values
    high_arr = high.values
    atr_arr = atr.values

    for i in range(length - n - exhaustion_lookahead):
        ret_up = close_arr[i + n] - close_arr[i]
        atr_i = atr_arr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # Harus ada kenaikan yang cukup sebagai prasyarat
        if ret_up < 0.5 * atr_i:
            continue

        # Tanda exhaustion di sequence setelah kenaikan
        future_low = np.min(close_arr[i+n : i+n+exhaustion_lookahead])
        pullback = close_arr[i + n] - future_low

        if pullback > pullback_atr * atr_i:
            labels[i] = 1
            continue

        # Gagal membuat high baru yang kuat (lower high)
        future_high = np.max(high_arr[i+n : i+n+exhaustion_lookahead])
        if future_high < high_arr[i + n]:
            labels[i] = 1

    labels[-n - exhaustion_lookahead:] = 0
    return pd.Series(labels, index=close.index, name="upward_exhaustion_label")


# ─── Per-coin processing ──────────────────────────────────────────────────────

def process_coin(sym: str, n: int, min_move: float, majority_thr: float = 0.55) -> dict | None:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"File tidak ditemukan, skip: {path}")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Filter training only — sama dengan LGBM/LSTM training
    df = df[df.index < TRAIN_CUTOFF_DATE].copy()

    if len(df) < n + 100:
        logger.warning(f"{sym}: data terlalu sedikit ({len(df)} rows), skip")
        return None

    # Generate momentum labels (versi klasik - untuk baseline LSTM)
    mom_labels = generate_momentum_labels(
        close        = df["close"],
        high         = df["high"],
        low          = df["low"],
        atr          = df["atr_14_h1"],
        n            = n,
        min_move     = min_move,
        majority_thr = majority_thr,
    )

    # Generate strong momentum labels (Per-coin Quantile, 3-class) - untuk referensi
    strong_labels = generate_strong_momentum_quantile_labels(
        close            = df["close"],
        n                = n,
        strong_quantile  = 0.75,
        weak_quantile    = 0.25,
        min_majority     = majority_thr,
    )

    # === Multi-Head Labels (2 Tugas Utama) ===
    strong_up_labels = generate_strong_up_label(
        close            = df["close"],
        n                = n,
        strong_quantile  = 0.78,
        min_majority     = majority_thr,
    )

    exhaustion_labels = generate_upward_exhaustion_label(
        close                = df["close"],
        high                 = df["high"],
        atr                  = df["atr_14_h1"],
        n                    = n,
        exhaustion_lookahead = 8,
        pullback_atr         = 1.2,
    )

    # === Clean Output for Multi-Head Experiment ===
    # We explicitly build a clean dataframe to avoid carrying over stale columns
    # from previous training runs.
    base_cols = [c for c in df.columns if c not in [
        "momentum_label", "strong_momentum_label", 
        "strong_up_label", "upward_exhaustion_label"
    ]]

    out = df[base_cols].copy()
    out["momentum_label"] = mom_labels
    out["strong_momentum_label"] = strong_labels
    out["strong_up_label"] = strong_up_labels
    out["upward_exhaustion_label"] = exhaustion_labels

    # Simpan (overwrite)
    out_path = LABEL_DIR / f"{sym}_momentum_labels.parquet"
    out.to_parquet(out_path)
    
    logger.info(f"{sym}: Clean label file written with columns: momentum_label, strong_momentum_label, strong_up_label, upward_exhaustion_label")

    # Statistik distribusi (momentum_label biasa)
    dist  = mom_labels.value_counts()
    total = len(mom_labels)
    stats = {
        "symbol":  sym,
        "total":   total,
        "LONG":    int(dist.get("LONG",  0)),
        "FLAT":    int(dist.get("FLAT",  0)),
        "SHORT":   int(dist.get("SHORT", 0)),
        "pct_LONG":  round(dist.get("LONG",  0) / total * 100, 1),
        "pct_FLAT":  round(dist.get("FLAT",  0) / total * 100, 1),
        "pct_SHORT": round(dist.get("SHORT", 0) / total * 100, 1),
    }

    # Statistik strong_momentum_label (3-class quantile)
    strong_dist = strong_labels.value_counts()
    stats["Strong_LONG"] = int(strong_dist.get("Strong_LONG", 0))
    stats["Normal"]      = int(strong_dist.get("Normal", 0))
    stats["Strong_SHORT"] = int(strong_dist.get("Strong_SHORT", 0))
    stats["pct_Strong_LONG"]  = round(stats["Strong_LONG"] / total * 100, 1)
    stats["pct_Normal"]       = round(stats["Normal"] / total * 100, 1)
    stats["pct_Strong_SHORT"] = round(stats["Strong_SHORT"] / total * 100, 1)

    # Statistik Multi-Head labels (untuk LSTM Corrector + Booster)
    stats["strong_up"] = int(strong_up_labels.sum())
    stats["pct_strong_up"] = round(stats["strong_up"] / total * 100, 1)
    stats["exhaustion"] = int(exhaustion_labels.sum())
    stats["pct_exhaustion"] = round(stats["exhaustion"] / total * 100, 1)

    # Bandingkan dengan swing label lama
    if "label" in df.columns:
        old_dist  = df["label"].value_counts()
        old_total = len(df)
        stats["old_pct_FLAT"] = round(old_dist.get("FLAT", 0) / old_total * 100, 1)

    logger.info(
        f"{sym:16}: LONG={stats['pct_LONG']:5.1f}%  "
        f"FLAT={stats['pct_FLAT']:5.1f}%  "
        f"SHORT={stats['pct_SHORT']:5.1f}%  "
        f"(lama FLAT={stats.get('old_pct_FLAT', '?')}%)"
    )
    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Multi-Head momentum labels for LSTM (Strong Up + Exhaustion)")
    p.add_argument("--n",            type=int,   default=12,   help="Horizon bar H1 ke depan (default: 12)")
    p.add_argument("--min-move",     type=float, default=0.25, help="Min move sebagai x ATR untuk label klasik (default: 0.25)")
    p.add_argument("--majority-thr", type=float, default=0.55, help="Fraksi bar searah (default: 0.55)")
    p.add_argument("--all",          action="store_true",      help="Pakai semua koin (ALL_COINS)")
    return p.parse_args()


def main():
    args  = parse_args()
    coins = ALL_COINS if args.all else TRAINING_COINS
    n, mm, mthr = args.n, args.min_move, args.majority_thr

    logger.info("=" * 70)
    logger.info("05a — GENERATE MULTI-HEAD MOMENTUM LABELS (CLEAN RUN)")
    logger.info("=" * 70)
    logger.info("PERINGATAN: Script ini akan overwrite file *_momentum_labels.parquet")
    logger.info("            dengan kolom label terbaru untuk eksperimen Multi-Head LSTM.")
    logger.info("            Data dari training sebelumnya TIDAK akan tercampur.")
    logger.info("-" * 70)

    logger.info(f"Generate labels for Multi-Head LSTM (2 Tugas Utama): N={n} bar")
    logger.info(f"  1. strong_up_label           : Binary (1 = Strong Up momentum)  [HEAD 1 - PRIORITAS]")
    logger.info(f"  2. upward_exhaustion_label   : Binary (1 = Momentum exhaustion / potensi reversal) [HEAD 2]")
    logger.info(f"  (Juga tetap generate momentum_label & strong_momentum_label untuk referensi)")
    logger.info(f"Koin: {len(coins)} | TRAIN_CUTOFF: {TRAIN_CUTOFF_DATE.date()}")
    logger.info(f"Output: {LABEL_DIR}/*_momentum_labels.parquet  (akan di-overwrite)")
    logger.info("-" * 70)

    all_stats = []
    for sym in coins:
        stats = process_coin(sym, n=n, min_move=mm, majority_thr=mthr)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        logger.error("Tidak ada file berhasil diproses.")
        return

    # Ringkasan agregat
    import statistics
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  MOMENTUM LABEL SUMMARY — N={n}, min_move={mm}x ATR")
    print(f"{sep}")
    print(f"  {'Coin':16} {'LONG%':>7} {'FLAT%':>7} {'SHORT%':>8} {'Old FLAT%':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for s in all_stats:
        print(
            f"  {s['symbol']:16} {s['pct_LONG']:>6.1f}% {s['pct_FLAT']:>6.1f}% "
            f"{s['pct_SHORT']:>7.1f}% {s.get('old_pct_FLAT','?'):>9}%"
        )

    pct_longs  = [s["pct_LONG"]  for s in all_stats]
    pct_flats  = [s["pct_FLAT"]  for s in all_stats]
    pct_shorts = [s["pct_SHORT"] for s in all_stats]
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    print(
        f"  {'MEAN':16} {statistics.mean(pct_longs):>6.1f}% "
        f"{statistics.mean(pct_flats):>6.1f}% "
        f"{statistics.mean(pct_shorts):>7.1f}%"
    )
    print(f"{sep}\n")

    # Strong Momentum Quantile Summary (3-class)
    print(f"\n{sep}")
    print(f"  STRONG MOMENTUM (Per-Coin Quantile) — 3-class: Strong_LONG / Normal / Strong_SHORT")
    print(f"{sep}")
    print(f"  {'Coin':16} {'StrLONG%':>9} {'Normal%':>8} {'StrSHORT%':>10}")
    print(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*10}")
    for s in all_stats:
        print(
            f"  {s['symbol']:16} {s.get('pct_Strong_LONG', 0):>8.1f}% "
            f"{s.get('pct_Normal', 0):>7.1f}% {s.get('pct_Strong_SHORT', 0):>9.1f}%"
        )

    pct_strong_long  = [s.get("pct_Strong_LONG", 0)  for s in all_stats]
    pct_normal       = [s.get("pct_Normal", 0)       for s in all_stats]
    pct_strong_short = [s.get("pct_Strong_SHORT", 0) for s in all_stats]
    print(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*10}")
    print(
        f"  {'MEAN':16} {statistics.mean(pct_strong_long):>8.1f}% "
        f"{statistics.mean(pct_normal):>7.1f}% {statistics.mean(pct_strong_short):>9.1f}%"
    )
    print(f"{sep}\n")

    # Multi-Head Summary
    pct_strong_up = [s.get("pct_strong_up", 0) for s in all_stats]
    pct_exhaustion = [s.get("pct_exhaustion", 0) for s in all_stats]
    print(f"\n{sep}")
    print("  MULTI-HEAD LABELS (2 Tugas Utama untuk LSTM)")
    print(f"{sep}")
    print(f"  {'Coin':16} {'StrongUp%':>10} {'Exhaustion%':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*12}")
    for s in all_stats:
        print(f"  {s['symbol']:16} {s.get('pct_strong_up',0):>9.1f}% {s.get('pct_exhaustion',0):>11.1f}%")
    print(f"  {'-'*16} {'-'*10} {'-'*12}")
    print(f"  {'MEAN':16} {statistics.mean(pct_strong_up):>9.1f}% {statistics.mean(pct_exhaustion):>11.1f}%")
    print(f"{sep}\n")

    print(f"  File disimpan di: {LABEL_DIR}")
    print(f"  Pattern: {{COIN}}_momentum_labels.parquet")
    print()
    print("  KOLOM LABEL YANG DIHASILKAN (clean):")
    print("    - momentum_label            (klasik, untuk referensi)")
    print("    - strong_momentum_label     (3-class quantile lama, untuk referensi)")
    print("    - strong_up_label           ← HEAD 1 (Strong Up - prioritas utama)")
    print("    - upward_exhaustion_label   ← HEAD 2 (Exhaustion untuk SHORT)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
