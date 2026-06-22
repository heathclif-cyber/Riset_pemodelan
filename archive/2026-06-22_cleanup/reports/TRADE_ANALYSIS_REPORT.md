# LIVE Trading Analysis — is_live=1 only

**Generated**: 2026-06-17 | **Source**: VPS app.db via `tools/live_db_bridge.py`
**Model**: tb_genuine_v2_dynsize_lstm_cond | **Scope**: 142 closed live trades + 2 open
**Period**: 2026-05-20 → 2026-06-17

> Catatan: ringkasan bridge (+$208, WR 48.8%) mencampur paper `is_live=0`. Report ini
> HANYA uang riil (`is_live=1`). `trade_analyzer.py` tidak dipakai untuk angka final
> karena format CSV bridge bikin kolom PnL NaN (script crash); analisis dari DB langsung.

## A. Scorecard (is_live=1, closed)

| Metrik | Nilai |
|--------|------:|
| Trades closed | 142 |
| Win Rate | **45.8%** |
| Net PnL | **−$31.60** |
| Gross profit / loss | +$28.04 / −$59.65 |
| Profit Factor | **0.47** |
| Expectancy / trade | **−$0.223** |
| Avg win / Avg loss | +$0.43 / **−$0.77** |
| Win:Loss size ratio | **0.56** (butuh >1.18 utk BE @45.8% WR) |
| Max consecutive loss | **8** |
| Best / Worst trade | +$2.16 / −$4.04 |

**Sistem live RUGI dan tidak punya edge.** PF 0.47 = setiap $1 profit diiringi $2.1 loss.
Masalah inti: ukuran loss (−0.77) hampir 2× ukuran win (+0.43).

## B. Temuan Kritis

```
[HIGH] LONG tidak punya edge — sumber utama kerugian
Bukti  : LONG 66 trade WR 40.9% net −$24.85 PF 0.31 | SHORT 76 trade WR 50% net −$6.76 PF 0.71
         LONG = 79% dari total kerugian bersih.
Dampak : Menghapus LONG saja menaikkan PF 0.47 -> 0.71, net −31.6 -> −6.76.

[HIGH] LONG "with-trend" justru paling parah
Bukti  : LONG with-trend (H4 ema50>=ema200) 16 trade WR 25% net −$15.17 PF 0.10 avg −$0.95.
         Satu bucket ini = ~48% total loss. Worst: DOTUSDT LONG sl_hit −$4.04 (conf 0.478).
Dampak : Entry LONG saat H4 uptrend = perangkap (beli di puncak/chop).

[HIGH] Confidence model TIDAK prediktif live
Bukti  : conf>=0.56 (108 trade) WR 48.1% net −$25.53 PF 0.47. conf>=0.52 net −$28.18 PF 0.45.
         Naikkan threshold confidence TIDAK memperbaiki (malah buang trade BE).
Dampak : Gate confidence (0.45) tidak menyaring loss. Sinyal LGBM tidak kalibrasi di live.

[MEDIUM] sl_hit LONG = loss besar
Bukti  : sl_hit LONG 10 trade −$14.87 (avg −$1.49) vs SHORT 13 trade −$8.71 (avg −$0.67).
Dampak : Stop LONG kena di harga jauh (wide-stop hit), bukan noise low-vol.

[MEDIUM] Konsentrasi koin buruk
Bukti  : TAO −9.19 (PF 0.12), LINK −6.46 (PF 0.03, WR 16.7%), DOT −5.51, ADA −2.61, BNB −2.05.
         5 koin ini = −$25.82 (82% net loss). Pemenang: POL +1.79, NEAR +1.65, TRX +0.96, XRP +0.55.
Dampak : Exclude 5 koin terburuk: net −31.6 -> −5.78, PF 0.47 -> 0.81.
```

## C. Per Exit Reason

| exit_reason | n | WR% | net | PF | avg |
|-------------|--:|----:|----:|---:|----:|
| guardian_momentum_exit | 13 | 100 | +8.74 | inf | +0.67 |
| manual_close | 11 | 72.7 | +0.88 | 1.46 | +0.08 |
| reconciled | 7 | 85.7 | −1.13 | 0.54 | −0.16 |
| guardian_exit | 88 | 43.2 | **−16.51** | 0.48 | −0.19 |
| sl_hit | 23 | 0 | **−23.58** | 0.00 | −1.03 |

`guardian_momentum_exit` (exit saat momentum kuat) = satu-satunya yang konsisten profit.
`guardian_exit` biasa (88 trade, mayoritas) WR 43% — Guardian sering keluar di rugi kecil.

## D. Per Arah & Alignment (H4 dari ema50/200 snapshot)

| direction | align | n | WR% | net | PF |
|-----------|-------|--:|----:|----:|---:|
| LONG | with | 16 | 25.0 | **−15.17** | 0.10 |
| LONG | counter | 50 | 46.0 | −9.68 | 0.50 |
| SHORT | with | 63 | 50.8 | −6.87 | 0.66 |
| SHORT | counter | 13 | 46.2 | +0.11 | 1.04 |

## E. Simulasi Dampak Rekomendasi

| Skenario | Trades | WR% | Net PnL | PF | Max Streak | vs Baseline |
|----------|-------:|----:|--------:|---:|-----------:|------------:|
| Baseline | 142 | 45.8 | −$31.60 | 0.47 | 8 | — |
| R1: skip LONG (SHORT only) | 76 | 50.0 | −$6.76 | 0.71 | 6 | +$24.84 |
| R2: conf>=0.52 | 117 | 46.2 | −$28.18 | 0.45 | 8 | +$3.42 |
| R3: exclude TAO/LINK/DOT/ADA/BNB | 112 | 49.1 | −$5.78 | 0.81 | 6 | +$25.82 |
| R1+R3 | 62 | 51.6 | **−$1.17** | **0.93** | 6 | +$30.43 |
| R1+R2+R3 | 54 | 50.0 | −$3.34 | 0.78 | 6 | +$28.26 |

**Kesimpulan rekonstruksi**: lever terbesar = skip LONG (R1) dan exclude koin buruk (R3).
Kombinasi R1+R3 mendekati breakeven (−$1.17, PF 0.93) — tapi **tetap tidak profit**.
R2 (confidence) tidak membantu. **Tidak ada kombinasi filter yang membuat book ini profit** —
masalahnya alpha entry, bukan filter.

## F. Open Positions (risiko)

| Coin | Dir | Entry | SL | Opened | Hold | Conf | Flag |
|------|-----|------:|---:|--------|-----:|-----:|------|
| NEARUSDT | SHORT | 2.3258 | 2.4798 | 06-16 22:05 | 3 | 0.503 | conf rendah; NEAR live +PnL (ok) |
| 1000PEPEUSDT | SHORT | 0.002951 | 0.003012 | 06-17 00:05 | 1 | 0.534 | PEPE live WR 20% net −$2.19 (koin lemah) |

Keduanya SHORT, conf di ambang. Risiko moderat — PEPE historis buruk, pantau.

## Kesimpulan: apa yang perlu diperbaiki

1. **Entry LONG harus disetop/diperketat total** — tidak ada edge (PF 0.31), LONG with-trend PF 0.10.
2. **Confidence gate tidak berfungsi** — LGBM tidak terkalibrasi di live; perlu rekalibrasi / meta-gate, bukan sekadar naikkan threshold.
3. **Blacklist koin** TAO/LINK/DOT/ADA/BNB — konsisten rugi (82% net loss).
4. **Pertahankan guardian_momentum_exit** — satu-satunya exit yang konsisten profit.
5. **Catatan SL floor (deploy hari ini)**: tidak menyentuh masalah utama (loss besar LONG = wide-stop hit, bukan noise low-vol). Pantau agar tidak memperbesar avg loss.
