Jalankan analisis performa trading lengkap: eksekusi script Python, lalu lakukan analisis mendalam terhadap data CSV.

## Konteks Sistem

SwingTrade v2 — paper trading bot Binance Futures. Model cascade_v3: LGBM entry → LSTM confirmation → Guardian v3 dynamic exit. Setiap trade punya kolom: Opened, Closed, Coin, Model, Direction, Conf, Entry, Exit, TP, SL, ATR, H4 Trend, Vol Regime, H4 High, H4 Low, RR, PnL ($), Exit Reason, Hold Bars, Status.

**Vol Regime** = volume candle saat ini dibagi rata-rata rolling volume. Nilai < 0.2 = pasar mati / volume sangat tipis → breakout lebih eksplosif dan tidak terprediksi.

**H4 Swing Deviation** = seberapa jauh harga entry dari level support/resistance H4 (dalam %). Jika > 8%, berarti swing H4 sudah tidak relevan dengan harga saat ini — data struktural basi, TP/SL dihitung dari ATR fallback yang kurang akurat.

**Filter simulasi** yang dijalankan script:
- Filter Vol Regime: skip trade jika Vol Regime < threshold (default 0.05) — hindari pasar mati
- Filter Swing Dev: skip trade jika deviasi H4 swing > threshold (default 8%) — hindari swing data basi

## Langkah 1 — Parse Argumen dari: $ARGUMENTS

- Mengandung `.csv` → gunakan sebagai `--file <path>`
- Mengandung `vol=<angka>` atau `p2=<angka>` → gunakan sebagai `--p2 <angka>` (min Vol Regime)
- Mengandung `dev=<angka>` atau `p4=<angka>` → gunakan sebagai `--p4 <angka>` (max swing deviation)
- Mengandung `no-filter` → tambahkan `--no-filter` (baseline saja)
- Kosong → gunakan semua default

Default:
- `--file livetrade.csv`
- `--p2 0.05` (min Vol Regime)
- `--p4 0.08` (max swing deviation 8%)
- `--output D:\Apps-Dev\Riset_pemodelan\reports\TRADE_ANALYSIS_REPORT.md`

## Langkah 2 — Jalankan Script

```bash
python tools/trade_analyzer.py --file <csv> --p2 <p2> --p4 <p4> [--no-filter]
```

Jika script error → diagnosa dan perbaiki sebelum lanjut.

## Langkah 3 — Baca Data CSV Secara Langsung

Setelah script selesai, baca file CSV yang dianalisis dan lakukan analisis mendalam sendiri.

**A. Pola loss streak terbaru**
- Urutkan trade by `Opened` descending, identifikasi loss berturut-turut terbaru
- Catat: berapa trade, total kerugian, pola gemerisnya (coin apa, arah apa, H4 Trend apa, Vol Regime berapa, exit reason apa)
- Apakah loss streak terjadi saat market sedang dalam kondisi tertentu (semua SHORT di H4 UP, dll)?

**B. Kualitas entry per kondisi market**
- Bandingkan WR dan PnL: SHORT di H4 UP (counter-trend) vs SHORT di H4 DOWN (with-trend), dan LONG di H4 DOWN (counter-trend) vs LONG di H4 UP (with-trend)
- Bandingkan WR dan PnL: Vol Regime < 0.2 vs Vol Regime >= 0.5
- Identifikasi kombinasi paling berbahaya dari data aktual

**C. Perilaku Guardian exit**
- Berapa `guardian_exit` yang loss vs win?
- Rata-rata hold bars saat guardian_exit loss vs win — apakah Guardian keluar terlalu dini?
- Bandingkan avg hold: guardian_exit vs sl_hit vs tp_hit

**D. Anomali coin spesifik**
- Coin mana paling banyak loss? Coin mana yang H4 swing-nya sering basi?
- Ada coin yang sebaiknya dikeluarkan dari daftar monitor?
- Hitung H4 swing deviation tiap coin dari kolom `Entry`, `H4 High`, `H4 Low`

**E. Dampak nyata filter Vol Regime + Swing Dev**
- Dari trade yang diblokir filter: berapa WIN vs LOSS?
- Apakah ada outlier besar yang mendistorsi hasil (contoh: satu coin pump besar dengan data H4 beku)?
- Jika outlier dikeluarkan, apakah filter net positif atau negatif terhadap PnL dan Win Rate?

**F. Kondisi open positions saat ini**
- List semua baris dengan `Status = open`
- Flag tiap posisi: counter-trend, Vol Regime mati (< 0.05), swing deviation > 8%
- Estimasi risiko keseluruhan open book saat ini

## Langkah 4 — Tampilkan ke User

### A. Scorecard

Tabel sebelum vs sesudah filter, ambil angka dari output script:

| Metrik | Sebelum Filter | Sesudah Filter | Delta |
|--------|---------------|----------------|-------|
| Trades | | | |
| Win Rate | | | |
| Net PnL | | | |
| Profit Factor | | | |
| Avg Loss | | | |
| Max loss streak | | | |

Sertakan juga perbandingan **tanpa coin outlier** (jika ada) agar apples-to-apples.

### B. Temuan Kritis

Minimum 3 temuan dari analisis data. Format:
```
[HIGH/MEDIUM/LOW] Judul temuan
Bukti   : angka spesifik dari data (tanggal, coin, nilai)
Dampak  : konsekuensi ke performa sistem
```

HIGH = merugikan langsung (loss streak aktif, counter-trend masif, Vol Regime mati)
MEDIUM = perlu perhatian (Guardian terlalu dini, swing data sering basi)
LOW = bisa dioptimalkan

### C. Analisis Per Model

Untuk setiap model yang muncul di data:
- N trades, WR, Net PnL, PF
- Kondisi market di mana model bekerja baik vs buruk
- Rekomendasi tuning spesifik (confidence threshold, Guardian params, dll)

### D. Rekomendasi Prioritas

```
[R1] Judul singkat
     What : apa yang diubah
     Where: file/parameter spesifik (misal inference_config.json, paper_trading.py)
     Why  : justifikasi dari data (sertakan angka)

[R2] ...
```

### E. Rekonstruksi Dampak Rekomendasi

**WAJIB dijalankan.** Untuk setiap rekomendasi di atas, simulasikan dampaknya terhadap data historis dengan menulis dan menjalankan script Python inline.

Logika simulasi per tipe rekomendasi:

- **Filter entry baru** (misal: skip jika counter-trend, skip jika Vol Regime < X, skip jika confidence < Y):
  Terapkan kondisi filter ke kolom CSV, hitung WR/PnL/PF pada trade yang lolos vs semua trade.

- **Perubahan threshold confidence**:
  Filter trade berdasarkan kolom `Conf` sesuai threshold baru, hitung ulang metrik.

- **Exclude coin tertentu**:
  Filter baris berdasarkan kolom `Coin`, hitung ulang metrik tanpa coin tersebut.

- **Kombinasi beberapa rekomendasi sekaligus**:
  Terapkan semua filter secara kumulatif, tampilkan hasilnya sebagai skenario gabungan.

Setelah menjalankan simulasi, tampilkan tabel proyeksi:

| Skenario | Trades | Win Rate | Net PnL | Profit Factor | Max Streak | vs Baseline |
|----------|--------|----------|---------|---------------|------------|-------------|
| Baseline (saat ini) | | | | | | — |
| Setelah R1 saja | | | | | | +/- |
| Setelah R2 saja | | | | | | +/- |
| Setelah R1 + R2 (combined) | | | | | | +/- |
| Best case (semua rekomendasi) | | | | | | +/- |

Jika ada outlier yang mendistorsi (seperti satu coin dengan pump ekstrem), tampilkan juga versi tanpa outlier tersebut.

Kesimpulan rekonstruksi: rekomendasi mana yang paling signifikan dampaknya, dan apakah kombinasi rekomendasi menghasilkan trade-off yang acceptable (trade count turun tapi WR dan PF naik).

### F. Open Positions Risk

List posisi terbuka dengan flag risikonya. Beri rekomendasi tindakan jika ada posisi berisiko tinggi.

### F. Lokasi Report

Sebutkan path file report yang sudah disimpan.

## Aturan Wajib

- Jalankan script Python terlebih dahulu — angka dari script adalah referensi utama
- Setelah script, baca CSV sendiri untuk analisis kualitatif yang lebih dalam
- Semua klaim harus didukung angka dari data — tidak ada "mungkin" atau "sepertinya"
- Report file selalu ditimpa, bukan append
- Jangan ubah kode `tools/trade_analyzer.py`
- Jika ada open positions, selalu evaluasi risikonya
