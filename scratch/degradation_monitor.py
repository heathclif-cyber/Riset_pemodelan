"""
scratch/degradation_monitor.py
Statistical Process Control untuk deteksi degradasi LGBM + Guardian.

Run secara periodik (mingguan) terhadap live trade history.
Input: CSV dengan kolom minimal:
  Opened, Exit Reason, PnL ($), Conf, Direction, Hold Bars

Dua monitor independen:
  1. LGBM Entry Quality  — rolling WR (semua trades)
  2. Guardian Exit Quality — rolling WR (guardian_exit trades saja)

Alert threshold: 2-sigma di bawah baseline holdout.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
HOLDOUT_CSV = (
    ROOT / "models" / "runs" / "ic32_guardian_clean_v2" / "holdout_trade_history.csv"
)
LIVE_CSV = Path(r"D:\Apps-Dev\swint_tradev2\hasil_livetrading.csv")

WINDOW = 50            # rolling window trades (bukan waktu)
ALERT_SIGMA = 2.0      # threshold untuk alert
MIN_TRADES_FOR_ALERT = 30  # jangan alert sebelum cukup data

# Baseline dari holdout 5 bulan (hardcoded dari guardian_audit.json)
BASELINE = {
    "entry_wr":   0.675,   # 67.5% semua trades (LGBM + Guardian)
    "guardian_wr": 0.796,  # 79.6% Guardian exits saja
    "entry_wr_std":  0.022,  # monthly std = 2.2pp
    "guardian_wr_std": 0.022,
}
ENTRY_ALERT   = BASELINE["entry_wr"]   - ALERT_SIGMA * BASELINE["entry_wr_std"]
GUARDIAN_ALERT = BASELINE["guardian_wr"] - ALERT_SIGMA * BASELINE["guardian_wr_std"]


def banner(t, w=65):
    print(f"\n{'='*w}\n  {t}\n{'='*w}")


def load_holdout(path):
    df = pd.read_csv(path, parse_dates=["Opened", "Closed"])
    df.columns = df.columns.str.strip()
    df["win"]    = (df["PnL ($)"] > 0).astype(int)
    df["is_gdn"] = (df["Exit Reason"].str.lower().str.contains("guardian")).astype(int)
    df["conf"]   = df["Conf"].astype(float)
    return df.sort_values("Opened").reset_index(drop=True)


def load_live(path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Opened", "Closed"])
        df.columns = df.columns.str.strip()
        df["win"]    = (df["PnL ($)"] > 0).astype(int)
        df["is_gdn"] = (df["Exit Reason"].str.lower().str.contains("guardian")).astype(int)
        df["conf"]   = df["Conf"].astype(float)
        return df.sort_values("Opened").reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Tidak bisa load live CSV: {e}")
        return None


def rolling_wr(series, window):
    return series.rolling(window, min_periods=window // 2).mean()


def spc_report(df, label="DATA", is_live=False):
    banner(f"SPC Report: {label} (N={len(df):,} trades)")

    if len(df) < MIN_TRADES_FOR_ALERT:
        print(f"\n  [SKIP] Terlalu sedikit trades ({len(df)} < {MIN_TRADES_FOR_ALERT})")
        return

    # ── 1. Entry quality ─────────────────────────────────────────────────────
    overall_wr = df["win"].mean()
    gdn_df     = df[df["is_gdn"] == 1]
    guardian_wr = gdn_df["win"].mean() if len(gdn_df) > 10 else np.nan

    print(f"\n  Overall WR         : {overall_wr:.1%}  "
          f"(baseline {BASELINE['entry_wr']:.1%}, alert <{ENTRY_ALERT:.1%})")
    print(f"  Guardian Exit WR   : {guardian_wr:.1%}  "
          f"(baseline {BASELINE['guardian_wr']:.1%}, alert <{GUARDIAN_ALERT:.1%})"
          if not np.isnan(guardian_wr) else "  Guardian Exit WR  : N/A")

    # ── 2. Alert status ───────────────────────────────────────────────────────
    print(f"\n  {'Metrik':<28} {'Status':>10}")
    def alert_line(name, val, threshold):
        if np.isnan(val):
            status = "N/A"
        elif val < threshold:
            status = "!!! ALERT"
        else:
            status = "OK"
        print(f"  {name:<28} {status:>10}  ({val:.1%} vs thr {threshold:.1%})")

    alert_line("LGBM Entry WR",    overall_wr, ENTRY_ALERT)
    alert_line("Guardian Exit WR", guardian_wr if not np.isnan(guardian_wr) else float('nan'),
               GUARDIAN_ALERT)

    # ── 3. Rolling WR per-50-trade window ────────────────────────────────────
    roll_entry = rolling_wr(df["win"].astype(float), WINDOW)
    last_10 = roll_entry.dropna().tail(10)

    print(f"\n  Rolling WR (window={WINDOW}) — last 10 snapshots:")
    for i, (idx, val) in enumerate(last_10.items()):
        flag = " !! ALERT" if val < ENTRY_ALERT else ""
        n_trade = idx + 1
        print(f"    trade #{n_trade:>5}: rolling WR = {val:.1%}{flag}")

    # ── 4. Guardian rolling ───────────────────────────────────────────────────
    if len(gdn_df) >= MIN_TRADES_FOR_ALERT:
        roll_gdn = rolling_wr(gdn_df["win"].astype(float).reset_index(drop=True), WINDOW)
        last_gdn = roll_gdn.dropna().tail(5)
        print(f"\n  Guardian rolling WR (window={WINDOW}) — last 5 snapshots:")
        for i, val in enumerate(last_gdn):
            flag = " !! ALERT" if val < GUARDIAN_ALERT else ""
            print(f"    snapshot {i+1}: {val:.1%}{flag}")

    # ── 5. Monthly breakdown ──────────────────────────────────────────────────
    df["month"] = pd.to_datetime(df["Opened"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("win", "count"),
        wr=("win", "mean"),
        gdn_wr=("win", lambda x: x[df.loc[x.index, "is_gdn"] == 1].mean()
                if (df.loc[x.index, "is_gdn"] == 1).sum() > 0 else np.nan)
    )
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<12} {'Trades':>7} {'WR':>8} {'Gdn WR':>9}")
    for period, row in monthly.iterrows():
        entry_flag = " !" if row["wr"] < ENTRY_ALERT else ""
        gdn_flag   = " !" if (not np.isnan(row["gdn_wr"]) and row["gdn_wr"] < GUARDIAN_ALERT) else ""
        gdn_str    = f"{row['gdn_wr']:.1%}" if not np.isnan(row["gdn_wr"]) else "N/A"
        print(f"  {str(period):<12} {row['trades']:>7} {row['wr']:>8.1%}{entry_flag} {gdn_str:>9}{gdn_flag}")

    # ── 6. Confidence drift ───────────────────────────────────────────────────
    # apakah avg confidence bergeser? ini bisa indikasi market regime shift
    conf_mean = df["conf"].mean()
    conf_std  = df["conf"].std()
    print(f"\n  Avg LGBM conf: {conf_mean:.4f} +/- {conf_std:.4f}")
    print(f"  (baseline range: 0.62-0.72 typical; jika di bawah 0.62 = market lebih uncertain)")

    # ── 7. SL rate monitor ────────────────────────────────────────────────────
    sl_df  = df[df["Exit Reason"].str.lower().str.contains("sl", na=False)]
    sl_rate = len(sl_df) / len(df)
    print(f"  SL hit rate: {sl_rate:.1%}  (baseline 17.3%; jika > 30% = regime shift)")
    if sl_rate > 0.30:
        print(f"  !!! SL rate {sl_rate:.1%} TINGGI - indikasi regime berbeda dari training")


# ── Main ──────────────────────────────────────────────────────────────────────
banner("DEGRADATION MONITOR — ic32_regime_v1 + Guardian clean_v2")
print(f"""
  Alert threshold konfigurasi:
    LGBM Entry WR alert  : < {ENTRY_ALERT:.1%}  (baseline {BASELINE['entry_wr']:.1%} - {ALERT_SIGMA}sigma)
    Guardian Exit WR alert: < {GUARDIAN_ALERT:.1%} (baseline {BASELINE['guardian_wr']:.1%} - {ALERT_SIGMA}sigma)
    Rolling window       : {WINDOW} trades
    Min trades for alert : {MIN_TRADES_FOR_ALERT}
""")

# Report dari holdout data (referensi baseline)
df_hold = load_holdout(HOLDOUT_CSV)
spc_report(df_hold, label="Holdout (Nov 2025 - Apr 2026)")

# Report dari live data (jika ada)
df_live = load_live(LIVE_CSV)
if df_live is not None and len(df_live) > 0:
    spc_report(df_live, label="Live Trading", is_live=True)
else:
    banner("Live Trading Data")
    if not LIVE_CSV.exists():
        print(f"\n  [INFO] Live CSV tidak ditemukan: {LIVE_CSV}")
        print(f"  Script ini dirancang untuk dijalankan setelah ada live trade history.")
    else:
        print(f"\n  [INFO] Live CSV kosong atau belum ada trades.")

# ── Summary: apa yang harus dilakukan jika ALERT ─────────────────────────────
banner("Tindakan jika ALERT")
print("""
  LGBM Entry WR turun di bawah 63.1%:
    1. Cek distribusi fitur terbaru vs training (feature_drift_check.py)
    2. Cek apakah ada regime shift di HMM (hmm_regime_enc distribution)
    3. Pertimbangkan: raise threshold LGBM sementara (0.69 -> 0.72 LONG)
    4. Jangan retrain ic32_regime_v1 tanpa diskusi

  Guardian WR turun di bawah 75.2%:
    1. Cek avg hold bars — apakah Guardian mulai exit terlalu cepat?
    2. Cek per-coin WR spread — ada koin tertentu yang collapse?
    3. Pertimbangkan: naikkan exit_threshold sementara (0.65 -> 0.68)
    4. Ini warning paling serius — Guardian adalah 79% dari trades

  SL rate naik di atas 30%:
    1. Reduce leverage atau trade size sementara
    2. Kemungkinan trending market (ic32 underperform di trending, lihat EXPERIMENTS.md)
    3. Manual review 5 trade terakhir
""")
