"""
tools/patch_features.py — Patch core/features.py untuk menambah 3 fitur volatility baru
"""
import re
from pathlib import Path

features_path = Path("core/features.py")
content = features_path.read_text(encoding="utf-8")

# Marker: akhir dari whale_retail_divergence dan sebelum section 29
old_marker = '    feat["whale_retail_divergence"] = (cvd_z - ls_z).fillna(0.0).clip(-5.0, 5.0)'

assert old_marker in content, f"Marker tidak ditemukan di {features_path}!"

new_block = '''    feat["whale_retail_divergence"] = (cvd_z - ls_z).fillna(0.0).clip(-5.0, 5.0)

    # Feature 4: ATR Z-Score (Volatility Spike Detector)
    # Seberapa jauh ATR H1 saat ini dari rata-rata 20-hari (480 bar)?
    # Nilai tinggi = gejolak/volatility spike. Seluruhnya backward-looking, zero leakage.
    atr_mean_20d = atr_h1.rolling(480, min_periods=48).mean()
    atr_std_20d  = atr_h1.rolling(480, min_periods=48).std().replace(0, np.nan)
    feat["atr_zscore_20d"] = (
        (atr_h1 - atr_mean_20d) / atr_std_20d
    ).fillna(0.0).clip(-5.0, 5.0)

    # Feature 5: ATR Percentile H1 (Relative Volatility Rank)
    # Posisi ATR H1 saat ini dalam distribusi 30-hari (720 bar).
    # 0.0=terendah, 1.0=tertinggi. Memberi konteks "lebih volatile dari biasanya?"
    def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=24).apply(
            lambda x: float((x[-1] > x[:-1]).mean()) if len(x) > 1 else 0.5,
            raw=True,
        ).fillna(0.5)

    feat["atr_percentile_h1"] = _rolling_pct_rank(atr_h1, window=720)

    # Feature 6: Volume Z-Score vs 48-bar baseline
    # Continuous z-score volume sekarang vs distribusi 48 bar terakhir.
    # Spike besar (>3) = kemungkinan kapitulasi, FOMO, atau liquidation cascade.
    vol_mean_48 = v.rolling(48, min_periods=12).mean()
    vol_std_48  = v.rolling(48, min_periods=12).std().replace(0, np.nan)
    feat["vol_spike_zscore"] = (
        (v - vol_mean_48) / vol_std_48
    ).fillna(0.0).clip(-5.0, 5.0)'''

content_new = content.replace(old_marker, new_block, 1)

if content_new == content:
    print("ERROR: Replace gagal - konten tidak berubah!")
else:
    features_path.write_text(content_new, encoding="utf-8")
    added_lines = content_new.count('\n') - content.count('\n')
    print(f"OK: features.py berhasil dipatch (+{added_lines} baris)")
    print(f"   Fitur baru: atr_zscore_20d, atr_percentile_h1, vol_spike_zscore")
