# Data Pipeline — Fetch, Clean, Features

Domain: persiapan data training. Baca file ini sebelum kerja fetch/clean/engineer/regime/feature selection.

## Sumber Data — Coinank (archive) → data asli live

Train awal: **Coinank trial** sekali → `data/coinank/` (read-only).
Live: **data asli** via Binance + yfinance — `fetch_positioning.py`.

| Coinank file | Pengganti | Endpoint / sumber |
|--------------|-----------|-------------------|
| `{coin}_oi.parquet` | `data/training/open_interest/{coin}_1h.parquet` | Binance `openInterestHist` |
| `{coin}_ls_position.parquet` | `data/positioning/{coin}_top_trader.parquet` | Binance `topLongShortPositionRatio` |
| `{coin}_ls_account.parquet` | `data/positioning/{coin}_top_account.parquet` | Binance `topLongShortAccountRatio` |
| global L/S (clean) | `data/training/long_short_ratio/{coin}_1h.parquet` | Binance `globalLongShortAccountRatio` |
| `{coin}_funding.parquet` | `data/training/funding_rate/{coin}_8h.parquet` | Binance funding (sudah match Coinank) |
| `etf_inflow_*.parquet` | `data/macro/etf_flow_btc.parquet` | yfinance ETF tickers |

**Kebijakan:** utamakan join data asli di `02_clean` → `features.py` pakai kolom real jika ada (`open_interest`, `long_short_ratio`). Synthetic hanya fallback bar NaN/0.

**Catatan irisan:** Coinank OI ≠ Binance OI (unit beda). Coinank L/S = nilai global sama untuk 21 koin — **jangan** dipakai live; ganti top-trader/account Binance per-koin.

**LSTM `p_bull`:** wajib migrasi ke pengganti di atas + retrain. Audit: `tools/features/audit_coinank_replacement.py`.

## Urutan Eksekusi (Core)

```powershell
python pipeline/data/run_fetch.py --all
python pipeline/data/run_fetch_positioning.py
python pipeline/data/run_clean.py
python pipeline/data/run_engineer.py --all
python pipeline/data/run_regime_train.py
python pipeline/data/run_regime_holdout.py   # holdout period only
```

Shim deprecated: `pipeline/01_fetch.py`, `pipeline/02_clean.py`, dll. → redirect ke `pipeline/data/core/*.py`.

## Input / Output

| Tahap | Input | Output |
|-------|-------|--------|
| Fetch | Binance API | `data/raw/` |
| Clean | `data/raw/` | `data/training/processed/` |
| Engineer | processed | `data/training/labeled/` |
| Regime HMM | labeled | regime columns di parquet |

Holdout mirror: `data/holdout/raw/`, `data/holdout/labeled/`

## HMM Regime — SSOT (WAJIB, jangan tertukar)

Ada **dua script** dengan tujuan berbeda. Salah pilih = live ≠ holdout.

| Script | Tujuan | Fit HMM | Predict | BTC cross-asset |
|--------|--------|---------|---------|-----------------|
| `regime_hmm.py` | Label **training** LGBM (pre-cutoff, OOF walk-forward) | Per-fold expanding | OOF val saja | **Ya** |
| `regime_hmm_holdout.py` | **Holdout + live production** | Seluruh H4 `< TRAIN_CUTOFF_DATE` | Seluruh H4 `>= cutoff` | **Tidak** |

**SSOT holdout & live:** `regime_hmm_holdout.py` → `models/hmm/{coin}_hmm.pkl` → merge `{coin}_regime_h1.parquet`.

### Prosedur benar (holdout / live)

1. H4 dari `processed` — kolom `4h_*` sudah **shift(1)** (`clean.py` baris 177–183).
2. Ekstrak bar `hour % 4 == 0` → rename ke `open/high/low/close/volume`.
3. `fit_hmm(df_train)` pada **semua** training H4; simpan `.pkl`.
4. `predict_hmm(model, df_oos, state_map)` pada **semua** H4 OOS `>= TRAIN_CUTOFF_DATE`.
5. `h4_to_h1` (ffill) → `encode_regime` → merge ke `features_v3.parquet`.

### Fitur internal HMM (`core/regime.py` → `_build_hmm_features`)

Bukan fitur LGBM. Input OHLCV H4; praktis pakai **close + volume**:

| # | Fitur | Window |
|---|-------|--------|
| 0 | return_1bar | 1 bar |
| 1 | volatility_24 | 24 bar H4 |
| 2 | momentum_48 | 48 bar H4 |
| 3 | log(volume_ratio) | 48 bar H4 |

### Larangan HMM (pernah menyebabkan mismatch ETH/DOGE Jun 2026)

| ❌ Jangan | ✅ Ganti dengan |
|----------|----------------|
| `predict_hmm(model, df.tail(500))` | Predict **full** OOS sejak cutoff |
| On-the-fly `fit_hmm` di live | Load frozen `models/hmm/{coin}_hmm.pkl` |
| H4 mentah tanpa `shift(1)` | `shift(1)` parity `clean.py` |
| Pakai `regime_hmm.py` OOF untuk inferensi OOS | Pakai `regime_hmm_holdout.py` |

Live mirror: `swint_tradev2/app/services/data_service.py` → `_compute_hmm_regime()`.

Verifikasi: `python tools/model/verify_hmm_feature_parity.py`

## Key Files

| File | Role |
|------|------|
| `core/fetchers.py` | Binance fetch logic |
| `core/binance_client.py` | API client |
| `core/features.py` | Feature engineering + swing labeling (implementasi) |
| `features/pipeline.py` | Facade public API — import dari sini di kode baru |
| `features/CLAUDE.md` | Aturan domain fitur |
| `core/regime.py` | HMM regime utilities |
| `pipeline/data/core/regime_hmm_holdout.py` | **SSOT** HMM holdout + export `.pkl` |
| `config.py` | TRAIN_CUTOFF_DATE, coins, paths |

## Feature Selection Tools

`tools/features/`:
- `ic_feature_selection_daily.py` — IC-based selection
- `audit_daily_features.py` — daily feature audit
- `audit_feature_parity.py` — live vs riset parity
- `diag_feature_drift.py` — drift detection

## Aturan Kausalitas (WAJIB untuk fitur non-H1)

Lihat detail + contoh kode: `METHODOLOGY.md`

1. Setiap `resample()` HARUS diikuti index shift ke akhir window sebelum diff/ffill:
   ```python
   agg = series.resample("4h").sum()
   agg.index = agg.index + pd.Timedelta("4h")   # WAJIB
   feat = agg.diff().reindex(...).ffill()
   ```
2. `ffill()` hanya valid SETELAH shift.
3. IC > 0.05 pada fitur non-H1 baru -> verifikasi kausalitas dulu.
4. Sebelum fitur non-H1 masuk training: catat verifikasi di `EXPERIMENTS.md`.

## Larangan Domain

- Jangan `shift(-N)` pada fitur tanpa lag kompensasi
- Jangan data post-`TRAIN_CUTOFF_DATE` bocor ke training
- Rolling window harus backward-looking

## Experiments

`pipeline/data/experiments/` — script eksperimen data/feature (kosong saat ini). Catat di `EXPERIMENTS.md` sebelum jalankan.