"""
File Konfigurasi Khusus: Manual Trading Deployment
Berisi parameter-parameter Take Profit dan Stop Loss (TP/SL) 
yang telah dikalibrasi untuk memitigasi ekor/wick market (stop-hunt) 
saat melakukan order secara manual (Set and Forget) di exchange.
"""

# ─── KONFIGURASI TP/SL (VERSI DEPLOYMENT MANUAL TRADING - AGRESSIVE SL 1.5) ───

TP_SL_HYBRID_MODE       = True       # max(swing, ATR) untuk TP / min(swing, ATR) untuk SL
TP_SL_SWING_FRESHNESS   = True       # tolak trade jika jarak swing > 15% dari entry (batas basi)
TP_SL_STRUCTURAL_FILTER = True       # validasi: entry dilarang jika terlampau jauh di luar range H4
TP_SL_STRUCTURAL_TOLERANCE = 0.04    # (BARU) Toleransi breakout 4%: izinkan entry menembus swing max 4%

# --- Gate & Bumper Khusus SL Manual (Berdasarkan Uji Empiris) ---
TP_SL_RR_GATE_ENABLED   = True       # aktifkan validasi kualitas trade sebelum entry
TP_SL_MIN_RR            = 0.5        # rasio toleransi: trade diizinkan asal TP minimal setengah dari risiko SL
TP_SL_MIN_TP            = 1.2        # jarak TP minimum mutlak adalah 1.2x ATR
TP_SL_MAX_SL            = 3.0        # jarak SL maksimum mutlak adalah 3.0x ATR
TP_SL_SWING_BUMPER      = 0.5        # (BARU) Beri kelonggaran 0.5x ATR di luar swing point untuk cegah Stop-Hunt wick

# --- Tier 2: Parameter Fallback (Saat Swing Point H4 Tidak Tersedia/NaN) ---
TP_SL_FALLBACK_SL       = 1.5        # (UPDATE) SL otomatis diperlebar menjadi 1.5x ATR (Winrate lebih tinggi)
TP_SL_FALLBACK_TP       = 2.0        # TP otomatis disetel 2.0x ATR

# --- Aspek Mekanisme & Eksekusi Tambahan ---
TP_SL_SLIPPAGE_ENABLED  = True       # simulasikan biaya tersembunyi pasar 0.05% per sisi
TP_SL_TRIGGER_MODE      = 'highlow'  # (KUNCI) Wajib "highlow" karena Anda order hard SL di buku exchange, bukan via bot
TP_SL_SIZING_MODE       = 'fixed'    # pukul rata ukuran per posisi (misal $100 per trade) untuk kestabilan portofolio
TP_SL_COOLDOWN_ENABLED  = False      # dimatikan agar sistem tidak kehilangan ratusan momentum trade beruntun

# ─── RIWAYAT PERCOBAAN & EVALUASI ─────────────────────────────────────────────
"""
Berikut adalah rangkuman eksperimen yang mendasari penentuan parameter di atas.
Semua menggunakan skenario Hard SL Manual (highlow trigger) dengan Bumper 0.5 ATR.

Eksperimen 1: Modifikasi RR Gate (Bumper 0.5 ATR)
| Konfigurasi               | Winrate | Trades | Profit  | Max Drawdown |
|---------------------------|---------|--------|---------|--------------|
| Bumper 0.5 + RR 1.0       | 82.2%   | 424    | +$3,612 | -9.4%        |
| Bumper 0.5 + RR 0.8       | 82.2%   | 476    | +$4,072 | -10.5%       |
| Bumper 0.5 + RR 0.5       | 81.9%   | 496    | +$4,259 | -10.4%       |
*Kesimpulan: RR 0.5 adalah sweet spot.*

Eksperimen 2: Modifikasi SL Fallback
| Konfigurasi               | Winrate | Trades | Profit  | Max Drawdown |
|---------------------------|---------|--------|---------|--------------|
| SL Fallback 1.0           | 81.9%   | 496    | +$4,259 | -10.4%       |
| SL Fallback 1.5           | 83.4%   | 496    | +$4,346 | -11.3%       |
*Kesimpulan: SL 1.5 menaikkan WR dan PnL meski Drawdown membengkak 0.9%.*

Eksperimen 3: Toleransi Breakout (Structural Filter)
| Konfigurasi               | Winrate | Trades | Profit  | Max Drawdown |
|---------------------------|---------|--------|---------|--------------|
| Toleransi 0% (Kaku)       | 83.4%   | 496    | +$4,346 | -11.3%       |
| Toleransi 4% (0.04)       | 83.3%   | 542    | +$4,828 | -11.4%       |
*Kesimpulan: Toleransi 4% menambahkan momentum valid (46 extra trades) dan meroketkan profit tanpa merusak risk profile.*
"""
