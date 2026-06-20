"""
pipeline/10_diag_guardian_labels.py
Diagnosis masalah Guardian labeling:
  1. Label distribution di training samples (label 0/1/2 pada PnL berapa?)
  2. Histogram current_pnl_pct per label -- diambil dari re-running labeling logic
  3. Guardian exit timing di holdout -- exit saat profit atau rugi?
  4. Ringkasan root cause
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import full_trading_report, simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import compute_guardian_static_array
from config import *

logger = setup_logger("10_diag_guardian")

GD_RUN = "ic32_guardian_clean_v2"
FB_RUN = "tb_lgbm_flatboost_v2"
LM     = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# ── Load models ───────────────────────────────────────────────────────────────
gd_model  = joblib.load(MODEL_DIR / "runs" / GD_RUN / "guardian.pkl")
gd_scaler = joblib.load(MODEL_DIR / "runs" / GD_RUN / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / GD_RUN / "guardian_feature_cols.json") as f:
    gd_feats_all = json.load(f)
gd_feats_static = [c for c in gd_feats_all if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json") as f:
    fb_feats = json.load(f)

available = sorted(
    s for s in ALL_COINS
    if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
)

# ── 1. Re-run labeling logic untuk lihat distribusi PnL per label ─────────────
print("=" * 70)
print("DIAGNOSIS 1: Label distribution (PnL saat label ditetapkan)")
print("=" * 70)
print()
print("Labeling rules (06b_train_guardian_clean_v2.py, urutan PRIORITY):")
print()
print("  Rule 1 [HOLD  ]: bars_held < MIN_HOLD_BARS")
print("  Rule 2 [EXIT-2]: current_pnl < -1.0 * atr_pct  <-- MASALAH UTAMA")
print("  Rule 3 [EXIT-2]: mfe > 1.5% DAN current_pnl < mfe * 0.25 (balik 75%)")
print("  Rule 4 [EXIT-2]: current_pnl >= best_future_pnl * 0.95 (di puncak)")
print("  Rule 5 [EXIT-1]: mfe > 1.5% DAN current_pnl < mfe * 0.55 (balik 45%)")
print("  Rule 6 [EXIT-1]: current_pnl > 0.8% DAN upside_ratio < 3%")
print("  Rule 7 [HOLD  ]: best_future_pnl > current_pnl * 1.05")
print("  Rule 8 [SKIP  ]: else")
print()

# Re-run labeling pada 5 koin untuk lihat distribusi
label_samples = {0: [], 1: [], 2: []}
rule_count    = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

for sym in available[:5]:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    n    = len(df)

    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    yp = np.full(n, 1, np.int32)
    yp[proba_fb[:, 2] >= 0.50] = 2
    yp[(proba_fb[:, 0] >= 0.55) & (yp != 2)] = 0

    close_arr = df["close"].values
    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    h_arr     = df["high"].values if "high" in df.columns else close_arr
    l_arr     = df["low"].values  if "low"  in df.columns else close_arr
    sh_arr    = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    sl_arr    = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

    result = simulate_trades_swing(
        y_pred=yp, close=close_arr, high=h_arr, low=l_arr, atr=atr_arr,
        h4_swing_highs=sh_arr, h4_swing_lows=sl_arr,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )
    trades = result.get("trades", [])

    for t in trades:
        bar_in  = t["bar_in"]
        bar_out = t["bar_out"]
        direction  = 2 if t["direction"] == "LONG" else 0
        entry_price = t["entry"]
        atr_entry   = atr_arr[bar_in] if bar_in < n else 0.01
        atr_pct     = atr_entry / entry_price if entry_price > 0 else 0.01

        if bar_out <= bar_in + 1:
            continue

        for j in range(bar_in + 1, min(bar_out, n)):
            cp = close_arr[j]
            if np.isnan(cp): continue

            current_pnl = (cp - entry_price)/entry_price if direction == 2 else (entry_price - cp)/entry_price

            # MFE so far
            mfe_sofar = 0.0
            for k in range(bar_in + 1, j + 1):
                if np.isnan(close_arr[k]): continue
                pnl_k = (close_arr[k] - entry_price)/entry_price if direction == 2 else (entry_price - close_arr[k])/entry_price
                mfe_sofar = max(mfe_sofar, pnl_k)

            # Best future PnL
            best_future_pnl = 0.0
            for k in range(j + 1, min(bar_out, n)):
                if np.isnan(close_arr[k]): continue
                pnl_k = (close_arr[k] - entry_price)/entry_price if direction == 2 else (entry_price - close_arr[k])/entry_price
                best_future_pnl = max(best_future_pnl, pnl_k)

            upside_ratio = (best_future_pnl - current_pnl) / best_future_pnl if best_future_pnl > 0.001 else 0.0

            bars_held = j - bar_in
            label = None

            if bars_held < GUARDIAN_MIN_HOLD_BARS:
                label = 0
            elif current_pnl < -1.0 * atr_pct:
                label = 2; rule_count[2] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.25:
                label = 2; rule_count[3] += 1
            elif current_pnl >= best_future_pnl * 0.95:
                label = 2; rule_count[4] += 1
            elif mfe_sofar > 0.015 and current_pnl < mfe_sofar * 0.55:
                label = 1; rule_count[5] += 1
            elif current_pnl > 0.008 and upside_ratio < 0.03:
                label = 1; rule_count[6] += 1
            elif best_future_pnl > current_pnl * 1.05:
                label = 0; rule_count[7] += 1
            else:
                continue

            if label is not None:
                label_samples[label].append(current_pnl)

# Print label distribution
total_smp = sum(len(v) for v in label_samples.values())
print(f"  Sampel dari 5 koin holdout (total {total_smp:,} samples):")
for lbl, name in [(0, "HOLD   "), (1, "PARTIAL"), (2, "FULL_EXIT")]:
    vals = label_samples[lbl]
    if not vals: continue
    arr = np.array(vals)
    pct = len(arr) / total_smp * 100
    pneg = (arr < 0).mean() * 100
    pos  = (arr > 0).mean() * 100
    print(f"    Label {lbl} [{name}]: {len(arr):>6,} samples ({pct:.1f}%) | "
          f"mean PnL {arr.mean()*100:+.2f}% | neg {pneg:.0f}% | pos {pos:.0f}%")

print()
print("  Rule yang memicu label EXIT-2:")
total_exit2 = sum(rule_count[r] for r in [2, 3, 4])
for r, label in [(2, "Rule 2: current_pnl < -1 ATR  (RUGI)"),
                 (3, "Rule 3: balik 75% dari MFE     (partial loss)"),
                 (4, "Rule 4: current_pnl >= best*0.95 (DI PUNCAK)")]:
    cnt = rule_count[r]
    pct = cnt / max(total_exit2, 1) * 100
    bar = "#" * int(pct / 2)
    print(f"    {label}: {cnt:>6,} ({pct:.0f}%) {bar}")

print()
pnl_at_rule2 = [label_samples[2][i] for i, _ in enumerate(label_samples[2])]
arr2 = np.array(label_samples[2])
pct_neg_exit2 = (arr2 < 0).mean() * 100
pct_pos_exit2 = (arr2 > 0).mean() * 100
print(f"  PnL saat label EXIT-2 ditetapkan: "
      f"mean {arr2.mean()*100:+.2f}% | negatif {pct_neg_exit2:.0f}% | positif {pct_pos_exit2:.0f}%")


# ── 2. Guardian exit timing di holdout ────────────────────────────────────────
print()
print("=" * 70)
print("DIAGNOSIS 2: Guardian exit di holdout -- keluar saat profit atau rugi?")
print("=" * 70)
print()

all_trades = []
for sym in available[:10]:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    n    = len(df)

    X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
    for i, c in enumerate(fb_feats):
        if c in df.columns:
            X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
    proba_fb = fb_model.predict_proba(X_fb)

    yp = np.full(n, 1, np.int32)
    yp[proba_fb[:, 2] >= 0.50] = 2
    yp[(proba_fb[:, 0] >= 0.55) & (yp != 2)] = 0

    X_gd = compute_guardian_static_array(df, gd_feats_static)
    conf = proba_fb.max(axis=1)
    rh4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    rh4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    rh4_tr = df["h4_trend"].values      if "h4_trend"      in df.columns else None

    rep = full_trading_report(
        y_pred=yp, y_actual=df["label"].map(LM).values.astype(np.int32),
        atr=df["atr_14_h1"].values,
        close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=rh4_sh, h4_swing_lows=rh4_sl, h4_trend=rh4_tr,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf,
        guardian_enabled=True,
        guardian_model=gd_model, guardian_scaler=gd_scaler, X_guardian=X_gd,
        guardian_exit_threshold=0.65, guardian_min_hold_bars=2,
        trailing_stop_enabled=False,
    )
    lev = rep.get("lev5x", rep)
    tr  = lev.get("trades", [])
    for t in tr:
        t["symbol"] = sym
    all_trades.extend(tr)

if all_trades:
    df_tr = pd.DataFrame(all_trades)
    df_tr["is_guardian"] = df_tr["outcome"].str.contains("GUARDIAN", na=False)
    df_tr["pnl_pct"]     = df_tr["net_pnl"] / MODAL_PER_TRADE * 100

    gd = df_tr[df_tr["is_guardian"]]
    no = df_tr[~df_tr["is_guardian"]]

    print(f"  Total trades (10 koin): {len(df_tr)} | Guardian exits: {len(gd)} ({len(gd)/len(df_tr)*100:.0f}%)")
    print()
    print(f"  PnL saat Guardian EXIT:")
    print(f"    Mean     : ${gd['net_pnl'].mean():+.3f}  ({gd['pnl_pct'].mean():+.2f}%)")
    print(f"    Median   : ${gd['net_pnl'].median():+.3f}  ({gd['pnl_pct'].median():+.2f}%)")
    print(f"    Positif  : {(gd['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"    Negatif  : {(gd['net_pnl'] < 0).mean()*100:.1f}%")
    print()
    print("  Histogram net_pnl saat GUARDIAN EXIT (per band):")
    bins = [(-99, -5), (-5, -2), (-2, -0.5), (-0.5, 0), (0, 0.5), (0.5, 2), (2, 5), (5, 99)]
    gd_pnl = gd["pnl_pct"].values
    for lo, hi in bins:
        cnt = ((gd_pnl >= lo) & (gd_pnl < hi)).sum()
        bar = "#" * int(cnt / max(len(gd), 1) * 60)
        print(f"    [{lo:+5.1f}% to {hi:+5.1f}%]: {cnt:>4} ({cnt/len(gd)*100:4.1f}%)  {bar}")

    print()
    print("  Per-outcome breakdown:")
    for out, grp in gd.groupby("outcome"):
        wr = (grp["net_pnl"] > 0).mean() * 100
        print(f"    {out:<32}: {len(grp):>4} trades | WR {wr:.0f}% | avg ${grp['net_pnl'].mean():+.3f}")


# ── 3. Root cause summary ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("ROOT CAUSE & FIX")
print("=" * 70)
print("""
ROOT CAUSE (dua masalah di labeling):

  [1] Rule 2 exit saat RUGI (current_pnl < -1 ATR) mendominasi EXIT-2 label.
      Guardian belajar: "ketika RSI jelek + CVD turun = EXIT"
      Tapi kondisi itu baru muncul SETELAH trade -3% sampai -5%.
      Sehingga di live: trade +5% belum ada sinyal balik, Guardian bilang HOLD.
      Baru saat -5%, sinyal balik muncul, Guardian bilang EXIT -- terlambat.

  [2] Saat trade +5%, SEBAGIAN BESAR bar dilabeli HOLD (Rule 7):
      best_future_pnl > current_pnl * 1.05 = masih ada upside lebih jauh.
      Guardian tidak belajar exit dari profit karena training data-nya HOLD.

FIX YANG DIPERLUKAN (retrain Guardian baru):

  HAPUS Rule 2 dari labeling (atau ubah ke label 0 HOLD).
  Alasan: SL di evaluator sudah handle loss-cutting. Guardian harusnya
  HANYA exit untuk proteksi profit, bukan cut loss (itu duplikasi SL).

  Ganti Rule 2 dengan profit-lock rule:
    if mfe_sofar > 0.02 (2% peak) AND drawdown_from_peak > 0.40:
        label = 2   # Exit: trade sudah profit lalu balik 40%

  Tambah rule eksplisit untuk exit profit besar:
    if current_pnl > 0.04 (4%) AND reversal_signal (RSI overbought/oversold):
        label = 2   # Exit: profit signifikan + sinyal balik

  Hasil yang diharapkan:
    - Guardian hanya fire saat trade sudah profit dan mulai berbalik
    - Tidak pernah fire saat trade masih dalam loss (biarkan SL)
    - WR Guardian exit akan LEBIH TINGGI (bukan campuran loss+profit)
""")
