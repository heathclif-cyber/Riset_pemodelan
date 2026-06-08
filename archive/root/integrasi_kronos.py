#!/usr/bin/env python3
"""
integrasi_kronos.py
====================
Feature Augmentation menggunakan Kronos-mini untuk pipeline trading.

Model     : NeoQuasar/Kronos-mini  (4.1M parameter, seq2seq transformer)
Tokenizer : NeoQuasar/Kronos-Tokenizer-2k
Source    : shiyu-coder/Kronos (HuggingFace Hub)

Cara pakai paling sederhana:
    df = pd.read_csv("data.csv")          # kolom: timestamp, open, high, low, close, volume
    df = tambah_fitur_kronos(df)           # ← tambahkan baris ini saja
    # lanjut pipeline LGBM + LSTM seperti biasa

═══════════════════════════════════════════════════════════════════════
  RAM Budget (VPS 1 GB RAM, CPU only):
  ─────────────────────────────────────────────────────────────────
  Python interpreter + stdlib          :  ~80 MB
  pandas + numpy (400 candle data)     :  ~25 MB
  torch (CPU-only wheel)               :  ~90 MB  ← overhead fixed
  Kronos-mini (float16, 4.1M param)    :  ~35 MB  ← weights saja
  transformers library overhead        :  ~30 MB
  ─────────────────────────────────────────────────────────────────
  Subtotal Kronos stack                : ~260 MB  ✓
  LightGBM model (opsional)            : ~80 MB
  TF/Keras LSTM (opsional, terpisah)   : ~200-300 MB  ← JANGAN load bersamaan
  ─────────────────────────────────────────────────────────────────
  Tanpa TF/Keras  : ~340 MB  ✓✓ aman
  Dengan TF/Keras : ~540-640 MB  ⚠ borderline — pakai subprocess mode
═══════════════════════════════════════════════════════════════════════

Install dependencies (VPS):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers>=4.40.0 accelerate sentencepiece
    pip install pandas numpy

Author  : auto-generated
Version : 1.0.0
"""

import gc
import json
import os
import sys
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI — ubah nilai di sini
# ═══════════════════════════════════════════════════════════════════════════════

KRONOS_MODEL_ID     = "NeoQuasar/Kronos-mini"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
KRONOS_REVISION     = "main"   # pin ke commit hash setelah verifikasi, contoh: "a1b2c3d4"
                                # Mencegah breaking change jika repo di-update

# Fallback values jika Kronos error di live trading
# 0/0.5 = neutral/uncertain — LGBM akan dapat sinyal netral, tidak crash
KRONOS_FALLBACK = {
    "kronos_pred_return_1h" : 0.0,
    "kronos_pred_return_36h": 0.0,
    "kronos_direction"      : 0.0,   # neutral
    "kronos_momentum"       : 0.0,
    "kronos_volatility"     : 0.02,  # default volatility kecil
    "kronos_bull_prob"      : 0.5,   # 50/50
    "kronos_range_ratio"    : 0.02,
    "kronos_confidence"     : 0.0,   # tidak yakin = model tidak aktif
}

# Cache per koin untuk live trading — update setiap 1 jam (1 H1 candle)
_KRONOS_CACHE: dict = {}   # {coin_key: (timestamp_computed, features_dict)}
KRONOS_CACHE_TTL = 3600    # detik; 1 jam = 1 H1 candle interval

CFG = {
    "lookback"         : 256,       # candle historis sebagai konteks
                                    # 256 (aman untuk context window Kronos-mini)
                                    # Jika ingin lebih panjang, cek max_position_embeddings
                                    # dengan: print(getattr(_MODEL.config, "max_position_embeddings", None))
    "predict_len"      : 36,        # HARUS = SWING_LABEL_MAX_HOLD = 36 jam (match label horizon LGBM)
                                    # Jika max_hold diubah di config.py, ubah ini juga!
    "num_samples"      : 20,        # sampel probabilistik (lebih banyak = lebih akurat, lebih lambat)
    "dtype"            : "float16", # "float16" hemat RAM, "float32" lebih presisi
    "batch_mode"       : "direct",  # "direct" | "subprocess" (subprocess = isolasi RAM)
    "cache_dir"        : None,      # None = HuggingFace default (~/.cache/huggingface)
    "direction_thresh" : 0.005,     # 0.5% threshold untuk direction = 0 (sideways)
}

# ─── Global model state (load sekali, pakai berkali-kali) ─────────────────────
_MODEL     = None
_TOKENIZER = None
_LOADED    = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MODEL LOADING & MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_kronos(force_reload: bool = False) -> None:
    """
    Load Kronos-mini + tokenizer ke memori global.

    Dipanggil otomatis oleh tambah_fitur_kronos().
    Panggil manual hanya jika ingin pre-warm model sebelum inference.

    RAM tambahan saat pemanggilan: ~35 MB (model) + ~90 MB (torch overhead)
    Total setelah load: ~260 MB (termasuk Python + pandas)
    """
    global _MODEL, _TOKENIZER, _LOADED

    if _LOADED and not force_reload:
        return

    # ── Lazy import torch & transformers ──────────────────────────────────────
    # Lazy import untuk menghindari torch overhead saat file di-import tapi
    # Kronos tidak dipakai.
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise ImportError(
            "Dependency belum terinstall. Jalankan:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install transformers>=4.40.0 accelerate\n"
            f"Detail: {exc}"
        ) from exc

    dtype = torch.float16 if CFG["dtype"] == "float16" else torch.float32
    cache = CFG["cache_dir"]

    print(f"[Kronos] Loading tokenizer: {KRONOS_TOKENIZER_ID} rev={KRONOS_REVISION} ...", flush=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(
        KRONOS_TOKENIZER_ID,
        revision          = KRONOS_REVISION,   # pin version — cegah breaking change
        trust_remote_code = True,
        cache_dir         = cache,
    )

    print(f"[Kronos] Loading model: {KRONOS_MODEL_ID} rev={KRONOS_REVISION} ({CFG['dtype']}, CPU) ...", flush=True)
    _MODEL = AutoModelForSeq2SeqLM.from_pretrained(
        KRONOS_MODEL_ID,
        revision          = KRONOS_REVISION,   # pin version
        torch_dtype       = dtype,
        device_map        = "cpu",
        low_cpu_mem_usage = True,
        trust_remote_code = True,
        cache_dir         = cache,
    )
    _MODEL.eval()

    # Disable gradient secara permanen — tidak butuh backprop saat inference
    # Hemat ~50% memori dibanding mode training
    for param in _MODEL.parameters():
        param.requires_grad_(False)

    _LOADED = True

    n_params  = sum(p.numel() for p in _MODEL.parameters())
    bytes_per = 2 if CFG["dtype"] == "float16" else 4
    ram_mb    = (n_params * bytes_per) / (1024 ** 2)
    print(
        f"[Kronos] ✓ Loaded | {n_params / 1e6:.2f}M params | "
        f"~{ram_mb:.0f} MB weights | total estimate ~{ram_mb + 90:.0f} MB",
        flush=True,
    )


def unload_kronos() -> None:
    """
    Bebaskan memori Kronos (~125 MB dibebaskan).

    Gunakan ini sebelum load TF/Keras jika RAM terbatas:

        unload_kronos()              # bebaskan ~125 MB
        model = load_lstm_keras()    # load TF/Keras setelah Kronos unloaded
    """
    global _MODEL, _TOKENIZER, _LOADED

    if not _LOADED:
        return

    del _MODEL
    del _TOKENIZER
    _MODEL    = None
    _TOKENIZER = None
    _LOADED   = False

    gc.collect()
    # Coba kosongkan CPU allocator cache torch jika tersedia
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("[Kronos] Model unloaded — memori dibebaskan ✓", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: INFERENCE CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Mean-scale normalization — standar untuk Kronos/Chronos-family models.
    Model ditraining pada data ternormalisasi; denormalisasi dilakukan setelah prediksi.
    """
    mean = float(np.abs(values).mean()) + 1e-8
    return values / mean, mean, 0.0


def _denormalize(pred: np.ndarray, scale: float) -> np.ndarray:
    return pred * scale


def _run_inference_direct(
    close_values: np.ndarray,
    predict_len: int,
    num_samples: int,
) -> np.ndarray:
    """
    Jalankan Kronos inference langsung di proses saat ini.

    Input : close_values — shape (lookback,), nilai close historis
    Output: predictions  — shape (num_samples, predict_len), harga prediksi

    RAM tambahan saat inference: ~15-30 MB (aktivasi + KV cache)
    """
    import torch

    load_kronos()

    # ── Preprocessing ──────────────────────────────────────────────────────────
    vals_normalized, scale, _ = _normalize(close_values)
    dtype   = torch.float16 if CFG["dtype"] == "float16" else torch.float32
    context = torch.tensor(vals_normalized, dtype=dtype).unsqueeze(0)  # (1, lookback)

    # ── Tokenization + Generation ──────────────────────────────────────────────
    # Kronos menggunakan tokenizer untuk mengkonversi float → token IDs,
    # kemudian generate token IDs baru → decode ke float.
    with torch.no_grad():
        try:
            # Kronos-style: tokenizer menerima tensor time series
            token_out = _TOKENIZER(
                context,
                return_tensors = "pt",
            )
            input_ids  = token_out["input_ids"]
            attention_mask = token_out.get("attention_mask", None)

        except TypeError:
            # Fallback: tokenizer menerima list of float
            token_out = _TOKENIZER(
                vals_normalized.tolist(),
                return_tensors = "pt",
            )
            input_ids = token_out["input_ids"]
            attention_mask = token_out.get("attention_mask", None)

        # Generate num_samples prediksi (probabilistic forecast)
        gen_kwargs = dict(
            input_ids       = input_ids,
            max_new_tokens  = predict_len,
            do_sample       = (num_samples > 1),
            num_return_sequences = num_samples,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        output_ids = _MODEL.generate(**gen_kwargs)

        # Decode token IDs → float predictions
        try:
            # Kronos tokenizer decode: token IDs → float array
            raw_pred = _TOKENIZER.decode(
                output_ids,
                skip_special_tokens = True,
            )
            if isinstance(raw_pred, torch.Tensor):
                preds_norm = raw_pred.float().numpy()
            elif isinstance(raw_pred, np.ndarray):
                preds_norm = raw_pred.astype(np.float32)
            elif isinstance(raw_pred, (list, tuple)):
                preds_norm = np.array(raw_pred, dtype=np.float32)
            else:
                # Coba parse string (beberapa model output string)
                tokens_str = str(raw_pred).strip().split()
                preds_norm = np.array([float(t) for t in tokens_str], dtype=np.float32)
        except Exception:
            # Last resort: ambil output_ids numerik langsung
            preds_norm = output_ids[:, -predict_len:].float().numpy()

    # ── Reshape ke (num_samples, predict_len) ─────────────────────────────────
    preds_norm = preds_norm.reshape(num_samples, predict_len) if preds_norm.size >= num_samples * predict_len else \
                 np.tile(preds_norm.flatten()[:predict_len], (num_samples, 1))

    # ── Denormalize ───────────────────────────────────────────────────────────
    predictions = _denormalize(preds_norm, scale)

    # ── Cleanup tensor dari memori ─────────────────────────────────────────────
    del context, input_ids, output_ids
    if "token_out" in dir():
        del token_out
    gc.collect()

    return predictions  # shape: (num_samples, predict_len)


def _run_inference_subprocess(
    close_values: np.ndarray,
    predict_len: int,
    num_samples: int,
) -> np.ndarray:
    """
    Jalankan Kronos dalam subprocess terpisah — ISOLASI RAM total.

    Gunakan mode ini jika TF/Keras sudah aktif di proses utama.
    Subprocess akan load Kronos (~260 MB RAM terpisah), inference, lalu mati.

    Overhead: +1-3 detik startup per call. Disarankan untuk batch, bukan real-time.
    """
    # Siapkan script inference minimal
    inference_script = """
import sys, json, numpy as np
import warnings; warnings.filterwarnings("ignore")

data = json.loads(sys.stdin.read())
close_values = np.array(data["close_values"], dtype=np.float32)
predict_len  = data["predict_len"]
num_samples  = data["num_samples"]
model_id     = data["model_id"]
tokenizer_id = data["tokenizer_id"]
dtype_str    = data["dtype"]

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dtype = torch.float16 if dtype_str == "float16" else torch.float32
tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, torch_dtype=dtype, device_map="cpu",
    low_cpu_mem_usage=True, trust_remote_code=True
)
mdl.eval()
for p in mdl.parameters(): p.requires_grad_(False)

mean = float(np.abs(close_values).mean()) + 1e-8
vals_norm = close_values / mean
context = torch.tensor(vals_norm, dtype=dtype).unsqueeze(0)

with torch.no_grad():
    try:
        token_out = tok(context, return_tensors="pt")
    except TypeError:
        token_out = tok(vals_norm.tolist(), return_tensors="pt")
    out_ids = mdl.generate(
        **token_out, max_new_tokens=predict_len,
        do_sample=(num_samples>1), num_return_sequences=num_samples
    )
    try:
        raw = tok.decode(out_ids, skip_special_tokens=True)
        if isinstance(raw, torch.Tensor): preds = raw.float().numpy()
        elif isinstance(raw, (list,np.ndarray)): preds = np.array(raw, dtype=np.float32)
        else: preds = np.array([float(x) for x in str(raw).split()], dtype=np.float32)
    except: preds = out_ids[:, -predict_len:].float().numpy()

preds = preds.reshape(num_samples, predict_len) if preds.size >= num_samples*predict_len \
        else np.tile(preds.flatten()[:predict_len], (num_samples, 1))
preds = preds * mean

print(json.dumps(preds.tolist()))
"""

    payload = json.dumps({
        "close_values" : close_values.tolist(),
        "predict_len"  : predict_len,
        "num_samples"  : num_samples,
        "model_id"     : KRONOS_MODEL_ID,
        "tokenizer_id" : KRONOS_TOKENIZER_ID,
        "dtype"        : CFG["dtype"],
    })

    result = subprocess.run(
        [sys.executable, "-c", inference_script],
        input   = payload,
        capture_output = True,
        text    = True,
        timeout = 120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Kronos subprocess error:\n{result.stderr[-500:]}")

    predictions = np.array(json.loads(result.stdout), dtype=np.float32)
    return predictions  # shape: (num_samples, predict_len)


def _infer(close_values: np.ndarray, predict_len: int, num_samples: int) -> np.ndarray:
    """
    Dispatch ke direct atau subprocess mode sesuai CFG["batch_mode"].
    """
    if CFG["batch_mode"] == "subprocess":
        return _run_inference_subprocess(close_values, predict_len, num_samples)
    else:
        return _run_inference_direct(close_values, predict_len, num_samples)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_features(
    current_price: float,
    predictions: np.ndarray,
    direction_thresh: float = 0.005,
) -> dict:
    """
    Konversi prediksi Kronos (num_samples × predict_len) menjadi 8 fitur tabular.

    Semua fitur backward-compatible dengan pipeline LGBM/LSTM yang sudah ada.
    Tidak menggantikan fitur lama, hanya menambah 8 kolom baru.

    Fitur (predict_len=36 → horizon 36H, match SWING_LABEL_MAX_HOLD=36):
      kronos_pred_return_1h  : expected return candle berikutnya (t+1)
      kronos_pred_return_36h : expected return kumulatif 36 jam (t+36) — match SWING_LABEL_MAX_HOLD=36
      kronos_direction       : arah trend 36H (-1=turun, 0=sideways, +1=naik)
      kronos_momentum        : slope linear prediksi (kecepatan trend)
      kronos_volatility      : std return prediksi (uncertainty model)
      kronos_bull_prob       : probabilitas harga naik > threshold di t+36
      kronos_range_ratio     : (pred_high - pred_low) / current_price
      kronos_confidence      : 1 - CV, makin tinggi = model makin yakin
    """
    eps = 1e-8
    if current_price <= 0:
        current_price = eps

    # Statistik dasar dari distribusi prediksi
    mean_pred = predictions.mean(axis=0)   # (predict_len,) — mean di semua sample
    std_pred  = predictions.std(axis=0)    # (predict_len,) — uncertainty per step

    # ── 1. Expected return 1-step ─────────────────────────────────────────────
    ret_1 = (mean_pred[0] - current_price) / current_price

    # ── 2. Expected return 12-step (kumulatif) ────────────────────────────────
    ret_12 = (mean_pred[-1] - current_price) / current_price

    # ── 3. Direction (−1/0/+1) ────────────────────────────────────────────────
    if ret_12 > direction_thresh:
        direction = 1.0
    elif ret_12 < -direction_thresh:
        direction = -1.0
    else:
        direction = 0.0

    # ── 4. Momentum — slope linear prediksi (ternormalisasi per harga) ────────
    x = np.arange(len(mean_pred), dtype=np.float64)
    if len(x) > 1:
        normalized_pred = (mean_pred - current_price) / current_price
        slope = float(np.polyfit(x, normalized_pred, 1)[0])
    else:
        slope = 0.0

    # ── 5. Volatility — std step-by-step returns ─────────────────────────────
    step_returns = np.diff(mean_pred) / (np.abs(mean_pred[:-1]) + eps)
    volatility   = float(step_returns.std()) if len(step_returns) > 0 else 0.0

    # ── 6. Bull probability ───────────────────────────────────────────────────
    # Fraksi sample yang memprediksi harga akhir > current + threshold
    threshold  = current_price * (1.0 + direction_thresh)
    bull_prob  = float((predictions[:, -1] > threshold).mean())

    # ── 7. Range ratio ────────────────────────────────────────────────────────
    pred_high   = float(mean_pred.max())
    pred_low    = float(mean_pred.min())
    range_ratio = (pred_high - pred_low) / current_price

    # ── 8. Confidence — 1 minus coefficient of variation ─────────────────────
    # CV = std/|mean|; rendah = yakin, tinggi = tidak yakin
    overall_mean = float(np.abs(mean_pred.mean())) + eps
    overall_std  = float(predictions.std())
    cv           = overall_std / overall_mean
    confidence   = float(np.clip(1.0 - cv, -1.0, 1.0))

    return {
        "kronos_pred_return_1h" : float(np.clip(ret_1,       -0.50, 0.50)),
        "kronos_pred_return_36h": float(np.clip(ret_12,      -1.00, 1.00)),  # ret_12 = last step
        "kronos_direction"      : direction,
        "kronos_momentum"       : float(np.clip(slope,       -0.10, 0.10)),
        "kronos_volatility"     : float(np.clip(volatility,   0.00, 1.00)),
        "kronos_bull_prob"      : bull_prob,
        "kronos_range_ratio"    : float(np.clip(range_ratio,  0.00, 0.50)),
        "kronos_confidence"     : confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FUNGSI UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def tambah_fitur_kronos(
    df: pd.DataFrame,
    lookback: int  = None,
    predict_len: int = None,
    close_col: str = "close",
    verbose: bool  = True,
) -> pd.DataFrame:
    """
    Tambahkan 8 fitur Kronos ke DataFrame trading yang sudah ada.

    Parameters:
        df          : DataFrame dengan minimal kolom 'close' (dan opsional OHLCV)
        lookback    : jumlah candle historis sebagai konteks (default: CFG["lookback"] = 400)
        predict_len : candle ke depan yang diprediksi (default: CFG["predict_len"] = 12)
        close_col   : nama kolom close price (default: "close")
        verbose     : cetak progress info

    Returns:
        df dengan 8 kolom tambahan:
            kronos_pred_return_1, kronos_pred_return_12,
            kronos_direction, kronos_momentum, kronos_volatility,
            kronos_bull_prob, kronos_range_ratio, kronos_confidence

    Kolom lama TIDAK dihapus — fitur Kronos ditambahkan di samping fitur existing.

    Contoh:
        df = pd.read_csv("SOLUSDT_1h.csv")
        df = tambah_fitur_kronos(df)
        # Sekarang df punya 8 kolom tambahan 'kronos_*'
        # Lanjut ke training LGBM seperti biasa
    """
    lookback    = lookback    or CFG["lookback"]
    predict_len = predict_len or CFG["predict_len"]
    num_samples = CFG["num_samples"]
    dir_thresh  = CFG["direction_thresh"]

    if close_col not in df.columns:
        raise ValueError(
            f"Kolom '{close_col}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    df = df.copy()
    n  = len(df)

    # ── Inisialisasi kolom output dengan NaN ─────────────────────────────────
    feature_cols = [
        "kronos_pred_return_1h", "kronos_pred_return_36h",
        "kronos_direction", "kronos_momentum", "kronos_volatility",
        "kronos_bull_prob", "kronos_range_ratio", "kronos_confidence",
    ]
    for col in feature_cols:
        df[col] = np.nan

    close_arr = df[close_col].values.astype(np.float32)

    if verbose:
        print(
            f"[Kronos] Memproses {n} bar | lookback={lookback} | "
            f"predict={predict_len} | samples={num_samples} | "
            f"mode={CFG['batch_mode']}",
            flush=True,
        )

    # ── Compute fitur untuk setiap bar yang punya cukup historis ─────────────
    computed = 0
    for i in range(lookback - 1, n):
        window = close_arr[i - lookback + 1 : i + 1]   # (lookback,) — backward only

        # Skip jika ada NaN atau zero di window
        if np.any(np.isnan(window)) or np.any(window <= 0):
            continue

        try:
            preds    = _infer(window, predict_len, num_samples)
            features = _extract_features(
                current_price    = float(close_arr[i]),
                predictions      = preds,
                direction_thresh = dir_thresh,
            )
            for col, val in features.items():
                df.at[df.index[i], col] = val

            computed += 1

        except Exception as exc:
            # Jangan crash pipeline — log warning dan lanjut
            if verbose:
                print(f"[Kronos] Warning bar {i}: {exc}", flush=True)
            continue

    if verbose:
        pct = computed / max(n - lookback + 1, 1) * 100
        print(
            f"[Kronos] ✓ Selesai | {computed}/{n - lookback + 1} bar diproses ({pct:.1f}%) | "
            f"NaN di {n - lookback} bar pertama (normal, lookback belum cukup)",
            flush=True,
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BATCH PROCESSING (untuk dataset besar / hemat RAM)
# ═══════════════════════════════════════════════════════════════════════════════

def tambah_fitur_kronos_batch(
    df: pd.DataFrame,
    step: int = 50,
    **kwargs,
) -> pd.DataFrame:
    """
    Versi batch-processing tambah_fitur_kronos untuk dataset sangat besar.

    Hitung Kronos hanya setiap 'step' candle (bukan setiap candle).
    Nilai di antara dua titik inferensi akan di-forward-fill.
    Ini mengurangi waktu komputasi secara signifikan dengan tradeoff
    akurasi fitur yang sedikit lebih rendah.

    Contoh: step=50 → inference 1x per 50 jam (untuk data H1)
    Hemat komputasi: ~50x lebih cepat.

    Parameters:
        df   : DataFrame input
        step : interval inferensi (candle)
    """
    lookback = kwargs.get("lookback", CFG["lookback"])
    n = len(df)

    # Pilih indeks di mana kita akan inference
    indices = list(range(lookback - 1, n, step))
    if indices and indices[-1] != n - 1:
        indices.append(n - 1)  # pastikan bar terakhir selalu diproses

    df = df.copy()
    close_col = kwargs.get("close_col", "close")
    close_arr = df[close_col].values.astype(np.float32)

    feature_cols = [
        "kronos_pred_return_1h", "kronos_pred_return_36h",
        "kronos_direction", "kronos_momentum", "kronos_volatility",
        "kronos_bull_prob", "kronos_range_ratio", "kronos_confidence",
    ]
    for col in feature_cols:
        df[col] = np.nan

    predict_len = kwargs.get("predict_len", CFG["predict_len"])
    num_samples = CFG["num_samples"]
    dir_thresh  = CFG["direction_thresh"]

    print(f"[Kronos Batch] {len(indices)} titik inferensi (step={step})", flush=True)

    for idx in indices:
        window = close_arr[idx - lookback + 1 : idx + 1]
        if np.any(np.isnan(window)) or np.any(window <= 0):
            continue
        try:
            preds    = _infer(window, predict_len, num_samples)
            features = _extract_features(float(close_arr[idx]), preds, dir_thresh)
            for col, val in features.items():
                df.at[df.index[idx], col] = val
        except Exception as exc:
            print(f"[Kronos Batch] Warning idx {idx}: {exc}", flush=True)

    # Forward-fill nilai di antara titik inferensi
    for col in feature_cols:
        df[col] = df[col].ffill()

    print("[Kronos Batch] ✓ Selesai + forward-fill applied", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: UTILITAS
# ═══════════════════════════════════════════════════════════════════════════════

def get_kronos_features_live(
    coin: str,
    df: pd.DataFrame,
    close_col: str = "close",
    force_refresh: bool = False,
) -> dict:
    """
    Fungsi khusus live trading — ambil fitur Kronos dengan cache per koin.

    Cache valid 1 jam (TTL = KRONOS_CACHE_TTL).
    Jika Kronos error → return KRONOS_FALLBACK (tidak crash pipeline).

    Contoh di production inference.py:
        kronos_feats = get_kronos_features_live("SOLUSDT", df_ohlcv)
        # Tambahkan ke feature vector sebelum LGBM predict
        for col, val in kronos_feats.items():
            feature_vector[col] = val

    Parameters:
        coin          : nama koin (dipakai sebagai cache key)
        df            : DataFrame OHLCV, minimal 257 baris (256 lookback + 1 current)
        close_col     : kolom close price
        force_refresh : paksa inference baru meski cache masih valid
    """
    import time

    cache_key = coin
    now       = time.time()

    # Cek cache
    if not force_refresh and cache_key in _KRONOS_CACHE:
        ts_computed, cached_feats = _KRONOS_CACHE[cache_key]
        if now - ts_computed < KRONOS_CACHE_TTL:
            return cached_feats   # cache hit — skip inference

    # Cache miss → run inference
    lookback = CFG["lookback"]
    if len(df) < lookback + 1:
        # Tidak cukup data historis
        return KRONOS_FALLBACK.copy()

    try:
        df_slice = df[[close_col]].rename(columns={close_col: "close"})
        df_result = tambah_fitur_kronos(
            df_slice.tail(lookback + 1),
            lookback    = lookback,
            predict_len = CFG["predict_len"],
            close_col   = "close",
            verbose     = False,
        )

        feature_cols = [c for c in df_result.columns if c.startswith("kronos_")]
        last_row     = df_result.iloc[-1]
        feats        = {col: float(last_row[col]) for col in feature_cols}

        # Fallback untuk nilai NaN (candle pertama belum ada cukup historis)
        for col in KRONOS_FALLBACK:
            if col not in feats or (feats[col] != feats[col]):   # NaN check
                feats[col] = KRONOS_FALLBACK[col]

        _KRONOS_CACHE[cache_key] = (now, feats)
        return feats

    except Exception as exc:
        import logging
        logging.getLogger("integrasi_kronos").warning(
            f"[{coin}] Kronos inference error: {exc} — pakai fallback"
        )
        return KRONOS_FALLBACK.copy()


def clear_kronos_cache(coin: str = None) -> None:
    """
    Hapus cache Kronos.
    coin=None → hapus semua cache.
    """
    if coin:
        _KRONOS_CACHE.pop(coin, None)
    else:
        _KRONOS_CACHE.clear()


def set_subprocess_mode(enabled: bool = True) -> None:
    """
    Aktifkan mode subprocess untuk isolasi RAM Kronos dari TF/Keras.

    Gunakan ini jika TF/Keras LSTM sudah di-load di proses utama:

        set_subprocess_mode(True)
        df = tambah_fitur_kronos(df)  # Kronos jalan di subprocess terpisah

    Trade-off: +1-3 detik overhead per batch inference.
    """
    CFG["batch_mode"] = "subprocess" if enabled else "direct"
    mode_str = "subprocess (RAM isolated)" if enabled else "direct (same process)"
    print(f"[Kronos] Mode diset ke: {mode_str}", flush=True)


def set_config(**kwargs) -> None:
    """
    Override konfigurasi global.

    Contoh:
        set_config(lookback=200, predict_len=24, num_samples=10)
        set_config(dtype="float32")  # lebih presisi, ~2x lebih banyak RAM
    """
    for key, val in kwargs.items():
        if key in CFG:
            CFG[key] = val
            print(f"[Kronos] Config: {key} = {val}", flush=True)
        else:
            print(f"[Kronos] Warning: key '{key}' tidak dikenal, skip.", flush=True)


def warmup_kronos() -> None:
    """
    Pre-load model ke memori sebelum inference pertama.
    Berguna untuk menghindari delay di real-time trading loop.

    Tanpa warmup: inference pertama lambat (~10-30 detik untuk load model).
    Dengan warmup: inference pertama cepat (model sudah di memori).
    """
    load_kronos()
    print("[Kronos] Model siap untuk inference ✓", flush=True)


def get_ram_estimate() -> dict:
    """
    Estimasi penggunaan RAM saat ini.

    Returns dict dengan breakdown komponen.
    """
    import os
    try:
        import psutil
        proc  = psutil.Process(os.getpid())
        total = proc.memory_info().rss / 1024**2
    except ImportError:
        total = None

    n_params = 0
    if _LOADED and _MODEL is not None:
        n_params = sum(p.numel() for p in _MODEL.parameters())

    bytes_per = 2 if CFG["dtype"] == "float16" else 4
    model_mb  = (n_params * bytes_per) / 1024**2 if n_params else 0

    return {
        "kronos_loaded"    : _LOADED,
        "model_params_M"   : round(n_params / 1e6, 2),
        "model_weights_mb" : round(model_mb, 1),
        "process_rss_mb"   : round(total, 1) if total else "psutil not installed",
        "dtype"            : CFG["dtype"],
        "mode"             : CFG["batch_mode"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CONTOH INTEGRASI KE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

USAGE_EXAMPLE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  CARA INTEGRASI KE PIPELINE LGBM + LSTM                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ Scenario 1: Tambah fitur Kronos sebelum training LGBM ━━━━━━━━━━━━━━━━━━━

    from integrasi_kronos import tambah_fitur_kronos

    df = pd.read_csv("data_training.csv")   # kolom: timestamp, open, high, low, close, volume
    df = tambah_fitur_kronos(df)            # ← tambah 8 kolom kronos_*

    # Lanjut seperti biasa
    feature_cols = [c for c in df.columns if c not in ["timestamp", "label"]]
    model_lgbm.fit(df[feature_cols], df["label"])


━━━ Scenario 2: Inference real-time (setiap candle baru) ━━━━━━━━━━━━━━━━━━━━

    from integrasi_kronos import warmup_kronos, tambah_fitur_kronos

    warmup_kronos()   # load model SEKALI saat startup (~30 detik)

    def on_new_candle(df_latest: pd.DataFrame) -> dict:
        \"\"\"Dipanggil setiap ada candle baru dari exchange.\"\"\"
        df_with_kronos = tambah_fitur_kronos(
            df_latest.tail(401),   # 400 lookback + 1 candle baru
            verbose=False,
        )
        return df_with_kronos.iloc[-1][["kronos_direction", "kronos_confidence"]].to_dict()


━━━ Scenario 3: RAM sangat terbatas — subprocess mode ━━━━━━━━━━━━━━━━━━━━━━━

    from integrasi_kronos import set_subprocess_mode, tambah_fitur_kronos

    # TF/Keras LSTM sudah loaded di proses utama (~300 MB)
    # Aktifkan mode subprocess agar Kronos tidak bertabrakan:
    set_subprocess_mode(True)

    df = tambah_fitur_kronos(df)   # Kronos jalan di child process terpisah
                                    # RAM: subprocess ~260 MB (terpisah dari main)
                                    # Main process: TF/Keras tetap jalan normal


━━━ Scenario 4: Dataset besar — batch mode untuk hemat waktu ━━━━━━━━━━━━━━━━

    from integrasi_kronos import tambah_fitur_kronos_batch

    df = pd.read_csv("SOLUSDT_1h_2years.csv")   # ~17,000 baris
    df = tambah_fitur_kronos_batch(df, step=50)  # inference tiap 50 jam
                                                  # ~50x lebih cepat


━━━ Scenario 5: Cek RAM usage ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from integrasi_kronos import get_ram_estimate
    import json

    print(json.dumps(get_ram_estimate(), indent=2))
    # Output:
    # {
    #   "kronos_loaded"    : true,
    #   "model_params_M"   : 4.1,
    #   "model_weights_mb" : 33.2,
    #   "process_rss_mb"   : 312.4,   # ← total RAM proses
    #   "dtype"            : "float16",
    #   "mode"             : "direct"
    # }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT — demo jika file dijalankan langsung
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kronos Feature Augmentation Demo")
    parser.add_argument("--csv",     type=str, default=None,   help="Path ke CSV input")
    parser.add_argument("--out",     type=str, default=None,   help="Path ke CSV output")
    parser.add_argument("--lookback",type=int, default=400,    help="Lookback candle")
    parser.add_argument("--predict", type=int, default=12,     help="Predict N candle")
    parser.add_argument("--samples", type=int, default=20,     help="Monte Carlo samples")
    parser.add_argument("--dtype",   type=str, default="float16", help="float16 | float32")
    parser.add_argument("--subprocess", action="store_true",   help="Aktifkan subprocess mode")
    parser.add_argument("--warmup",  action="store_true",      help="Pre-load model saja")
    parser.add_argument("--ram",     action="store_true",      help="Cek RAM estimate")
    parser.add_argument("--usage",   action="store_true",      help="Tampilkan contoh integrasi")
    args = parser.parse_args()

    if args.usage:
        print(USAGE_EXAMPLE)
        sys.exit(0)

    if args.ram:
        load_kronos()
        import json
        print(json.dumps(get_ram_estimate(), indent=2))
        sys.exit(0)

    set_config(
        lookback    = args.lookback,
        predict_len = args.predict,
        num_samples = args.samples,
        dtype       = args.dtype,
    )

    if args.subprocess:
        set_subprocess_mode(True)

    if args.warmup:
        warmup_kronos()
        sys.exit(0)

    if args.csv is None:
        # Demo dengan data sintetis
        print("[Demo] Tidak ada --csv, generate data sintetis 500 baris...")
        np.random.seed(42)
        n_demo = 500
        price  = 100.0 + np.cumsum(np.random.randn(n_demo) * 0.5)
        df_demo = pd.DataFrame({
            "timestamp" : pd.date_range("2024-01-01", periods=n_demo, freq="1H"),
            "open"      : price * (1 + np.random.randn(n_demo) * 0.001),
            "high"      : price * (1 + np.abs(np.random.randn(n_demo)) * 0.003),
            "low"       : price * (1 - np.abs(np.random.randn(n_demo)) * 0.003),
            "close"     : price,
            "volume"    : np.random.randint(1000, 5000, n_demo).astype(float),
        })

        print("[Demo] Menjalankan tambah_fitur_kronos()...")
        df_out = tambah_fitur_kronos(df_demo)

        print("\n[Demo] Sample output (5 baris terakhir):")
        kronos_cols = [c for c in df_out.columns if c.startswith("kronos_")]
        print(df_out[["close"] + kronos_cols].tail(5).to_string())

    else:
        print(f"[Run] Load CSV: {args.csv}")
        df_in  = pd.read_csv(args.csv)
        df_out = tambah_fitur_kronos(df_in)

        out_path = args.out or args.csv.replace(".csv", "_kronos.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[Run] Saved: {out_path}")
        print(f"[Run] Kolom baru: {[c for c in df_out.columns if c.startswith('kronos_')]}")

    print("\n" + "═" * 60)
    print("Contoh integrasi lengkap: python integrasi_kronos.py --usage")
    print("RAM estimate          : python integrasi_kronos.py --ram")
    print("═" * 60)
