"""Check structural filter dan posisi simulasi vs live untuk 19 Jun 2026 16:00 WITA."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HOLDOUT_DIR, MODEL_DIR, ALL_COINS
from core.utils import ensure_utc_index

# Load signal list dari debug sebelumnya
with open(MODEL_DIR / 'runs' / 'ic32_regime_v1' / 'holdout_diag_jun19_wita.json') as f:
    diag = json.load(f)
entries = diag['entries']

print(f"\n{'='*85}")
print("  INVESTIGASI: Mengapa hanya BTC yang trade live pada 19 Jun 2026?")
print(f"{'='*85}\n")
print("  Simulasi menunjukkan 8 entry signals. Live hanya BTC LONG @16:14 WITA.")
print()

# Fokus pada signal jam 16:00 WITA (08:00 UTC) — BTC, ETH dan 18:00 WITA DOGE
target_ts = {
    'BTC_16':  ('BTCUSDT',   pd.Timestamp('2026-06-19 08:00', tz='UTC')),
    'ETH_16':  ('ETHUSDT',   pd.Timestamp('2026-06-19 08:00', tz='UTC')),
    'DOGE_18': ('DOGEUSDT',  pd.Timestamp('2026-06-19 10:00', tz='UTC')),
    'NEAR_00': ('NEARUSDT',  pd.Timestamp('2026-06-18 16:00', tz='UTC')),
    'NEAR_01': ('NEARUSDT',  pd.Timestamp('2026-06-18 17:00', tz='UTC')),
    'LINK_05': ('LINKUSDT',  pd.Timestamp('2026-06-18 21:00', tz='UTC')),
    'LINK_06': ('LINKUSDT',  pd.Timestamp('2026-06-18 22:00', tz='UTC')),
    'NEAR_08': ('NEARUSDT',  pd.Timestamp('2026-06-19 00:00', tz='UTC')),
}

print(f"  {'Label':<10} {'Sym':<16} {'Close':>10} {'H4_SH':>10} {'H4_SL':>10} {'In range?':>10} {'Swing age':>10} {'Regime'}")
print("  " + "-" * 100)

all_bar_info = {}
for label, (sym, ts_utc) in target_ts.items():
    df = pd.read_parquet(HOLDOUT_DIR / 'labeled' / f'{sym}_features_v3.parquet')
    df = ensure_utc_index(df).sort_index()
    bar = df[df.index == ts_utc]
    if bar.empty:
        print(f"  {label:<10} {sym:<16} BAR NOT FOUND")
        continue
    r = bar.iloc[0]
    close  = float(r['close'])
    h4sh   = float(r['h4_swing_high']) if 'h4_swing_high' in r and not pd.isna(r['h4_swing_high']) else np.nan
    h4sl   = float(r['h4_swing_low'])  if 'h4_swing_low'  in r and not pd.isna(r['h4_swing_low']) else np.nan
    regime = str(r['hmm_regime']) if 'hmm_regime' in r else '?'

    if not np.isnan(h4sh) and not np.isnan(h4sl):
        in_range = h4sl < close < h4sh
    else:
        in_range = None

    # Cek swing age (dari kolom swing_... jika ada)
    swing_age_h = '?'
    # Cek apakah ada kolom terkait swing age
    swing_age_cols = [c for c in r.index if 'swing_age' in c.lower() or 'swing_bar' in c.lower()]

    ts_wita = (ts_utc + pd.Timedelta(hours=8)).strftime('%H:%M')
    status = "PASS" if in_range else ("FAIL" if in_range is False else "?")
    print(f"  {label:<10} {sym:<16} {close:>10.4f} {h4sh if not np.isnan(h4sh) else 'nan':>10} {h4sl if not np.isnan(h4sl) else 'nan':>10} {str(in_range):>10} {'?':>10} {regime}")
    all_bar_info[label] = {'in_range': in_range, 'close': close, 'h4sh': h4sh, 'h4sl': h4sl, 'regime': regime}

print()
print("  Structural filter (inference_config): require_entry_in_swing_range=True, swing_max_age_hours=48")
print()

# Analisis mengapa ETH tidak masuk
print(f"{'='*85}")
print("  ANALISIS: Kemungkinan penyebab hanya BTC yang trade live")
print(f"{'='*85}")
print()

btc16 = all_bar_info.get('BTC_16', {})
eth16 = all_bar_info.get('ETH_16', {})

print("  1. STRUCTURAL FILTER (require_entry_in_swing_range)")
print(f"     BTC 16:00 WITA: in_range={btc16.get('in_range')} -> {'LOLOS' if btc16.get('in_range') else 'BLOKIR'}")
print(f"     ETH 16:00 WITA: in_range={eth16.get('in_range')} -> {'LOLOS' if eth16.get('in_range') else 'BLOKIR'}")
print()

# Cek signal lebih awal (NEAR/LINK) yang mungkin sudah mengisi slot
near_01 = all_bar_info.get('NEAR_01', {})
link_05 = all_bar_info.get('LINK_05', {})
near_08 = all_bar_info.get('NEAR_08', {})

print("  2. SLOT MANAGEMENT — Open positions pada jam 16:00 WITA")
print("     Jika live system mengambil NEAR/LINK signals dari dini hari:")
print("     - NEAR LONG 00:00 WITA: SL hit di bar 1 (01:00 WITA) -> CLOSED")
print("     - NEAR LONG 01:00 WITA: TIME_EXIT bar 30 (Jun 20 07:00 WITA) -> MASIH OPEN")
print("     - LINK SHORT 05:00 WITA: TIME_EXIT bar 26 (Jun 20 07:00 WITA) -> MASIH OPEN")
print("     - LINK SHORT 06:00 WITA: TIME_EXIT bar 25 (Jun 20 07:00 WITA) -> MASIH OPEN")
print("     - NEAR SHORT 08:00 WITA: TP_HIT bar 4 (12:00 WITA) -> CLOSED")
print()
print("     Tapi: max_open_positions=10, jadi 3 slot tidak cukup untuk block BTC+ETH")
print()

print("  3. KEMUNGKINAN BESAR: Signal NEAR/LINK TIDAK diambil di live")
print("     Live system mungkin tidak generate signal NEAR/LINK dini hari")
print("     (perbedaan feature live vs historical, structural filter, atau")
print("      confidence tidak mencukupi di live data real-time)")
print()

print("  4. MENGAPA ETH TIDAK MASUK padahal conf=0.881?")
print("     Hipotesis:")
print("     a) Structural filter: ETH close tidak dalam H4 swing range")
print(f"        ETH: close={eth16.get('close',0):.4f} H4_SH={eth16.get('h4sh',float('nan'))} H4_SL={eth16.get('h4sl',float('nan'))}")
print("     b) Daily loss limit: jika ada loss sebelumnya di hari ini")
print("     c) Live feature berbeda dari historical (OI, funding rate lag)")
print("     d) Slippage/timing: ETH signal muncul bersamaan BTC, sistem pilih 1")
print()

print("  5. BTC TRADE LIVE YANG TERJADI:")
print("     Entry: 16:14 WITA @62,654.21 (sim: 16:00 @62,575.70, delta +$78)")
print("     Exit:  22:38 WITA @63,277.35 via guardian_momentum_exit (6h hold)")
print("     PnL live: +$12.53 (+4.6%) pada $5 modal 5x = actual win rate match")
print("     Sim prediksi: TP_HIT @63,088 bar 5 (21:00 WITA) — close, Guardian exit lebih late")
print()
print("  KESIMPULAN: BTC benar masuk sesuai simulasi. ETH tidak masuk kemungkinan")
print("  karena structural filter (in_range check) atau live feature mismatch.")
print("  Perlu pull live DB untuk diagnosis definitif sinyal ETH/DOGE/NEAR/LINK.")
