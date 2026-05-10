# Trading ML Pipeline

Pipeline ML untuk prediksi sinyal trading futures leverage. Ensemble model (LightGBM + LSTM + Logistic Regression).

## Struktur Folder

```
trading_ml/
├── config.py                    ← semua parameter di satu tempat
├── run_pipeline.py              ← entry point
├── requirements.txt
│
├── core/                        ← modul reusable
│   ├── binance_client.py        ← Binance API client
│   ├── fetchers.py              ← fetch OHLCV, funding rate, macro
│   ├── features.py              ← feature engineering + labeling
│   ├── models.py                ← TradingLSTM, load/save helpers
│   └── utils.py                 ← logging, parquet I/O, progress tracking
│
├── pipeline/                    ← jalankan berurutan
│   ├── 01_fetch.py              ← fetch data dari Binance
│   ├── 02_clean.py              ← data cleaning + MTF join
│   ├── 03_engineer.py           ← 58 fitur + Triple Barrier labeling
│   ├── 04_train_lgbm.py         ← LightGBM baseline
│   ├── 05_train_lstm.py         ← LSTM sequence model
│   ├── 06_ensemble.py           ← stacking ensemble
│   └── 07_evaluate.py           ← SHAP analysis
│
├── models/
│   ├── lgbm_baseline.pkl
│   ├── lstm_best.pt
│   ├── lstm_scaler.pkl
│   ├── ensemble_meta.pkl
│   └── runs/                    ← output per training run
│       └── run_20260418/
│
├── data/
│   ├── raw/klines/{SYMBOL}/{interval}_all.parquet
│   ├── raw/funding_rate/{SYMBOL}_8h.parquet
│   ├── raw/macro/btc_dominance.parquet
│   ├── processed/{SYMBOL}_clean.parquet
│   └── labeled/{SYMBOL}_features.parquet
│
└── reports/eda/
```

## Cara Pakai

### Install dependencies
```bash
pip install -r requirements.txt
```

### Jalankan pipeline lengkap (training coins)
```bash
python run_pipeline.py --all
```

### Jalankan per fase
```bash
python pipeline/01_fetch.py                   # fetch training coins
python pipeline/01_fetch.py --new             # fetch 15 koin baru
python pipeline/02_clean.py
python pipeline/03_engineer.py
python pipeline/04_train_lgbm.py
python pipeline/05_train_lstm.py              # sebaiknya di Google Colab
python pipeline/06_ensemble.py
python pipeline/07_evaluate.py               # SHAP analysis
```

### Jalankan untuk semua 20 koin
```bash
python run_pipeline.py --fetch --clean --engineer --all-coins
```

## Model Performance (Training 5 Koin)

| Model       | F1-macro | Accuracy |
|-------------|----------|----------|
| LightGBM    | 0.5437   | 70.19%   |
| LSTM        | 0.4953   | 64.20%   |
| Ensemble    | 0.5249   | —        |

## Config

Semua parameter ada di `config.py`:
- Periode data: `TRAIN_START`, `TRAIN_END`, `NEW_COINS_START`
- Triple Barrier: `TP_ATR_MULT`, `SL_ATR_MULT`, `MAX_HOLDING_BARS`
- LSTM: `LSTM_SEQ_LEN`, `LSTM_HIDDEN`, `LSTM_EPOCHS`
- Signal threshold: `CONFIDENCE_FULL` (0.75), `CONFIDENCE_HALF` (0.60)

## Koin

**Training (5 koin):** SOLUSDT, ETHUSDT, BNBUSDT, XRPUSDT, DOGEUSDT

**Koin baru (15 koin):** TONUSDT, ADAUSDT, TRXUSDT, SHIBUSDT, AVAXUSDT, LINKUSDT, DOTUSDT, SUIUSDT, POLUSDT, NEARUSDT, PEPEUSDT, TAOUSDT, APTOSUSDT, ARBUSDT, WLFIUSDT

## Catatan

- OB_price **tidak di-include** di feature set (di-drop berdasarkan EDA — trigger 99.6%)
- Synthetic OI dihitung dari CVD karena OI historis tidak tersedia via API
- LSTM sebaiknya ditraining di Google Colab T4 GPU
- Model tersimpan di `models/` root untuk inference dan di `models/runs/{run_id}/` untuk versioning
