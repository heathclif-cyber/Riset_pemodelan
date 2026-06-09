"""
core/features.py — Feature Engineering & Labeling v3
Base Timeframe: H1 (bukan M15)
Konteks Swing: H4 (swing high/low, trend, divergence)

Fungsi utama:
  engineer_features()          — hitung semua 100 fitur dari cleaned parquet (H1 base)
  swing_based_labeling()       — labeling v3 berbasis H4 swing high/low
  structural_label_filter()    — filter label berdasarkan posisi harga dalam range
  compute_synthetic_oi()       — synthetic OI dari CVD (H1-adjusted windows)

Perubahan v3 vs v2:
  - Base timeframe: M15 → H1 (noise reduction ~75%)
  - Labeling: Triple Barrier M15 → Swing-Based H4
  - +9 fitur smart money: cvd_div_h4, cvd_slope_h4, vol_efficiency,
    absorption_z, funding_price_div, rsi_h4, rsi_divergence,
    wyckoff_phase, spring_upthrust
  - +16 fitur H4 dynamics & D1 HTF context (fix: sebelumnya hilang dari parquet)
  - Semua rolling windows disesuaikan ke satuan bar H1
  - Parameter names dibersihkan (tidak ada lagi referensi "m15")
"""

import numpy as np
import pandas as pd

from core.utils import setup_logger, ensure_utc_index

try:
    from config import SYNTHETIC_OI_CVD_WINDOW, SYNTHETIC_OI_NORM_WINDOW
except ImportError:
    try:
        from app.services.config_loader import get_feature_engineering_config as _get_fe_cfg
        _fe = _get_fe_cfg()
        SYNTHETIC_OI_CVD_WINDOW  = _fe.get("synthetic_oi_cvd_window",  24)
        SYNTHETIC_OI_NORM_WINDOW = _fe.get("synthetic_oi_norm_window", 168)
    except Exception:
        SYNTHETIC_OI_CVD_WINDOW  = 24
        SYNTHETIC_OI_NORM_WINDOW = 168

logger = setup_logger("features")


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Cari kolom pertama yang cocok (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _get_ohlcv(df: pd.DataFrame, prefix: str = "1h"):
    """Extract OHLCV dari DataFrame dengan prefix timeframe."""
    def col(name):
        full = f"{prefix}_{name}"
        return df[full] if full in df.columns else pd.Series(np.nan, index=df.index)
    return col("open"), col("high"), col("low"), col("close"), col("volume")


# ═══════════════════════════════════════════════════════════════════════════════
# INDIKATOR DASAR
# ═══════════════════════════════════════════════════════════════════════════════

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 6) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_stochrsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    rsi     = calc_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period, min_periods=1).min()
    rsi_max = rsi.rolling(stoch_period, min_periods=1).max()
    k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = k.rolling(k_period, min_periods=1).mean()
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d


def calc_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def calc_cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta — proxy tekanan beli/jual taker."""
    buy_col  = _col(df, "taker_buy_volume", "1h_taker_buy_volume",
                    "taker_ratio_takerBuyVol")
    sell_col = _col(df, "taker_sell_volume", "1h_taker_sell_volume",
                    "taker_ratio_takerSellVol")

    if buy_col and sell_col:
        delta = df[buy_col].fillna(0) - df[sell_col].fillna(0)
    else:
        # Fallback: estimasi dari arah harga × volume
        close  = df.get("close", df.get("1h_close", pd.Series(np.nan, index=df.index)))
        volume = df.get("volume", df.get("1h_volume", pd.Series(np.nan, index=df.index)))
        sign   = np.sign(close.diff().fillna(0))
        delta  = sign * volume.fillna(0)

    return delta.cumsum()


def calc_volume_delta(df: pd.DataFrame) -> pd.Series:
    """Volume Delta per bar (bukan kumulatif)."""
    buy_col  = _col(df, "taker_buy_volume", "1h_taker_buy_volume",
                    "taker_ratio_takerBuyVol")
    sell_col = _col(df, "taker_sell_volume", "1h_taker_sell_volume",
                    "taker_ratio_takerSellVol")

    if buy_col and sell_col:
        return (df[buy_col] - df[sell_col]).fillna(0)

    close  = df.get("close", df.get("1h_close", pd.Series(np.nan, index=df.index)))
    volume = df.get("volume", df.get("1h_volume", pd.Series(np.nan, index=df.index)))
    sign   = np.sign(close.diff().fillna(0))
    return sign * volume.fillna(0)


def compute_synthetic_oi(
    df: pd.DataFrame,
    cvd_window: int = 24,    # 24 bar H1 = 24 jam
    norm_window: int = 168,  # 168 bar H1 = 1 minggu
) -> pd.Series:
    """
    Synthetic Open Interest dari CVD + Volume.
    Dipakai saat data OI real tidak tersedia.
    Window disesuaikan untuk H1 base (bukan M15).
    """
    cvd_col = _col(df, "cvd")
    vol_col = _col(df, "volume")
    if cvd_col is None or vol_col is None:
        raise KeyError("Kolom 'cvd' atau 'volume' tidak ditemukan untuk synthetic OI.")

    cvd    = df[cvd_col].astype(float)
    volume = df[vol_col].astype(float)

    cvd_ma     = cvd.rolling(cvd_window,  min_periods=1).mean()
    vol_ma     = volume.rolling(cvd_window, min_periods=1).mean()
    raw        = cvd_ma + vol_ma * 2
    norm_denom = raw.rolling(norm_window, min_periods=1).mean().replace(0, np.nan)
    return (raw / norm_denom).ffill().fillna(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_volume_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 24,   # 24 bar H1 = 24 jam
    bins: int = 50,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Point of Control, Value Area High, Value Area Low."""
    poc_list, vah_list, val_list = [], [], []
    typical = (high + low + close) / 3
    n = len(close)

    for i in range(n):
        start = max(0, i - window + 1)
        tp_w = typical.iloc[start: i + 1].values
        vo_w = volume.iloc[start:  i + 1].values
        hi_w = high.iloc[start:    i + 1].values
        lo_w = low.iloc[start:     i + 1].values

        if len(tp_w) < 2 or np.nansum(vo_w) == 0:
            poc_list.append(np.nan)
            vah_list.append(np.nan)
            val_list.append(np.nan)
            continue

        price_min = np.nanmin(lo_w)
        price_max = np.nanmax(hi_w)
        if price_max == price_min:
            poc_list.append(price_max)
            vah_list.append(price_max)
            val_list.append(price_min)
            continue

        edges   = np.linspace(price_min, price_max, bins + 1)
        bin_idx = np.digitize(tp_w, edges, right=True).clip(1, bins) - 1
        bin_vol = np.zeros(bins)
        for b, v in zip(bin_idx, vo_w):
            if not np.isnan(v):
                bin_vol[b] += v

        total_vol = bin_vol.sum()
        poc_bin   = int(np.argmax(bin_vol))
        poc_price = (edges[poc_bin] + edges[poc_bin + 1]) / 2

        target = total_vol * 0.70
        lo_ptr, hi_ptr = poc_bin, poc_bin
        acc = bin_vol[poc_bin]
        while acc < target and (lo_ptr > 0 or hi_ptr < bins - 1):
            add_lo = bin_vol[lo_ptr - 1] if lo_ptr > 0         else 0
            add_hi = bin_vol[hi_ptr + 1] if hi_ptr < bins - 1  else 0
            if add_lo >= add_hi and lo_ptr > 0:
                lo_ptr -= 1; acc += bin_vol[lo_ptr]
            elif hi_ptr < bins - 1:
                hi_ptr += 1; acc += bin_vol[hi_ptr]
            else:
                break

        poc_list.append(poc_price)
        vah_list.append((edges[hi_ptr] + edges[hi_ptr + 1]) / 2)
        val_list.append((edges[lo_ptr] + edges[lo_ptr + 1]) / 2)

    idx = close.index
    return (
        pd.Series(poc_list, index=idx, name="POC"),
        pd.Series(vah_list, index=idx, name="VAH"),
        pd.Series(val_list, index=idx, name="VAL"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE & LIQUIDITY
# ═══════════════════════════════════════════════════════════════════════════════

def detect_swing_highs_lows(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Deteksi swing high dan swing low pada timeframe base (H1)."""
    n  = len(high)
    sh = pd.Series(False, index=high.index)
    sl = pd.Series(False, index=low.index)
    for i in range(lookback, n - lookback):
        window_h = high.iloc[i - lookback: i + lookback + 1]
        window_l = low.iloc[i  - lookback: i + lookback + 1]
        if high.iloc[i] == window_h.max():
            sh.iloc[i] = True
        if low.iloc[i] == window_l.min():
            sl.iloc[i] = True
    return sh, sl


def calc_liquidity_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    lookback: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Buy/Sell Liquidity jarak ke swing H/L terakhir + SFP sweep detection."""
    sh, sl   = detect_swing_highs_lows(high, low, lookback)
    # shift(lookback): swing at bar i confirmed only after i+lookback bars pass.
    # Without shift, features use future bars to confirm swings — look-ahead bias.
    swing_hi = high.where(sh).ffill().shift(lookback)
    swing_lo = low.where(sl).ffill().shift(lookback)

    atr_safe = atr.replace(0, np.nan)
    sell_liq = (swing_hi - close) / atr_safe
    buy_liq  = (close - swing_lo)  / atr_safe

    sfp_sweep = (
        ((low < swing_lo) & (close > swing_lo)) |
        ((high > swing_hi) & (close < swing_hi))
    ).astype(int)

    return buy_liq, sell_liq, sfp_sweep


def calc_market_structure(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Market Structure Break (BOS) dan Change of Character (CHoCH)."""
    sh, sl  = detect_swing_highs_lows(high, low, lookback)
    # shift(lookback+1): swing at bar i confirmed at i+lookback; +1 so prev_sh
    # at bar j reflects the last confirmed swing strictly before bar j.
    prev_sh = high.where(sh).ffill().shift(lookback + 1)
    prev_sl = low.where(sl).ffill().shift(lookback + 1)

    bos_bull = (close > prev_sh).astype(int)
    bos_bear = (close < prev_sl).astype(int) * -1
    bos      = bos_bull + bos_bear

    last_bos = bos.replace(0, np.nan).ffill()
    choch    = ((bos != 0) & (last_bos != last_bos.shift(1))).astype(int)

    cum        = (bos != 0).cumsum()
    bars_since = cum.groupby(cum).cumcount()
    bars_since = bars_since.where(cum > 0, other=999)

    return bos, choch, bars_since


def calc_fvg(
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    min_gap_atr: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Fair Value Gap — gap di antara tiga candle berturut-turut."""
    n        = len(high)
    fvg_up   = pd.Series(0.0, index=high.index)
    fvg_down = pd.Series(0.0, index=low.index)

    for i in range(1, n - 1):
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue
        gap_up   = low.iloc[i]  - high.iloc[i - 1]
        gap_down = low.iloc[i - 1]  - high.iloc[i]
        if gap_up   > min_gap_atr * atr_val:
            fvg_up.iloc[i]   = gap_up   / atr_val
        if gap_down > min_gap_atr * atr_val:
            fvg_down.iloc[i] = gap_down / atr_val

    return fvg_up, fvg_down


# ═══════════════════════════════════════════════════════════════════════════════
# KEY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_prev_day_week_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
) -> dict[str, pd.Series]:
    """Previous Day/Week High/Low — ATR-normalized."""
    df_tmp      = pd.DataFrame({"high": high, "low": low})
    daily_high  = df_tmp["high"].resample("1D").max()
    daily_low   = df_tmp["low"].resample("1D").min()
    weekly_high = df_tmp["high"].resample("1W").max()
    weekly_low  = df_tmp["low"].resample("1W").min()

    def shift_ffill(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.reindex(shifted.index.union(high.index)).ffill().reindex(high.index)

    atr_safe = atr.replace(0, np.nan)
    return {
        "PDH": (shift_ffill(daily_high)  - close) / atr_safe,
        "PDL": (shift_ffill(daily_low)   - close) / atr_safe,
        "PWH": (shift_ffill(weekly_high) - close) / atr_safe,
        "PWL": (shift_ffill(weekly_low)  - close) / atr_safe,
    }


def calc_fib_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    window: int = 24,   # 24 bar H1 = 24 jam
) -> dict[str, pd.Series]:
    """Fibonacci retracement 61.8% dan 78.6% dari rolling high/low."""
    roll_high = high.rolling(window, min_periods=5).max()
    roll_low  = low.rolling(window,  min_periods=5).min()
    rng       = roll_high - roll_low
    atr_safe  = atr.replace(0, np.nan)
    return {
        "Fib_618": (roll_high - 0.618 * rng - close) / atr_safe,
        "Fib_786": (roll_high - 0.786 * rng - close) / atr_safe,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TIME FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def calc_market_session(index: pd.DatetimeIndex) -> pd.Series:
    """Session encoding: 1=Asia, 2=London, 3=New York, 0=Off."""
    hour    = index.hour
    session = np.zeros(len(index), dtype=np.int8)
    session[(hour >= 0)  & (hour < 8)]  = 1   # Asia
    session[(hour >= 7)  & (hour < 15)] = 2   # London
    session[(hour >= 13) & (hour < 21)] = 3   # New York
    return pd.Series(session, index=index, name="market_session")


def calc_cyclic_time(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Hour dan day-of-week sebagai fitur siklus (sin/cos)."""
    hour = index.hour + index.minute / 60
    dow  = index.dayofweek
    return {
        "hour_sin": pd.Series(np.sin(2 * np.pi * hour / 24), index=index),
        "hour_cos": pd.Series(np.cos(2 * np.pi * hour / 24), index=index),
        "dow_sin":  pd.Series(np.sin(2 * np.pi * dow  /  7), index=index),
        "dow_cos":  pd.Series(np.cos(2 * np.pi * dow  /  7), index=index),
    }


def calc_time_to_funding(index: pd.DatetimeIndex) -> pd.Series:
    """Jarak ke settlement funding rate berikutnya, dinormalisasi [0,1]."""
    minutes_in_day  = index.hour * 60 + index.minute
    next_settlement = np.ceil(minutes_in_day / 480) * 480
    mins_remaining  = (next_settlement - minutes_in_day) % 480
    return pd.Series(mins_remaining / 480.0, index=index, name="time_to_funding_norm")


# ═══════════════════════════════════════════════════════════════════════════════
# H4 SWING DETECTION (untuk labeling & features)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_h4_swing_points(
    h4_high: pd.Series,
    h4_low:  pd.Series,
    lookback: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Deteksi swing high dan swing low di timeframe H4.
    Lookback 3 = swing yang terkonfirmasi 3 bar kanan dan kiri.

    Return:
        swing_highs : float Series — harga swing high, NaN jika bukan swing
        swing_lows  : float Series — harga swing low,  NaN jika bukan swing
    """
    n  = len(h4_high)
    sh = pd.Series(np.nan, index=h4_high.index, dtype=float)
    sl = pd.Series(np.nan, index=h4_low.index,  dtype=float)

    for i in range(lookback, n - lookback):
        window_h = h4_high.iloc[i - lookback: i + lookback + 1]
        window_l = h4_low.iloc[i  - lookback: i + lookback + 1]
        if h4_high.iloc[i] == window_h.max():
            sh.iloc[i] = h4_high.iloc[i]
        if h4_low.iloc[i] == window_l.min():
            sl.iloc[i] = h4_low.iloc[i]

    return sh, sl


def get_nearest_swing_levels(
    h4_swing_highs: pd.Series,
    h4_swing_lows:  pd.Series,
    base_index:     pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """
    Forward-fill swing levels H4 ke index base (H1).
    Hanya ffill — tidak ada lookahead bias.

    Return:
        sh_base : swing high H4 terakhir, aligned ke H1 index
        sl_base : swing low  H4 terakhir, aligned ke H1 index
    """
    sh_filled = h4_swing_highs.ffill()
    sl_filled = h4_swing_lows.ffill()

    sh_base = sh_filled.reindex(
        sh_filled.index.union(base_index)
    ).ffill().reindex(base_index)

    sl_base = sl_filled.reindex(
        sl_filled.index.union(base_index)
    ).ffill().reindex(base_index)

    return sh_base, sl_base


def calc_rsi_h4(
    h4_close:   pd.Series,
    close_base: pd.Series,   # H1 close untuk alignment index
    period:     int = 14,
) -> pd.Series:
    """RSI dihitung dari H4 close, di-align ke index H1."""
    rsi_h4 = calc_rsi(h4_close, period)
    return rsi_h4.reindex(
        rsi_h4.index.union(close_base.index)
    ).ffill().reindex(close_base.index)


# ═══════════════════════════════════════════════════════════════════════════════
# SMART MONEY FEATURES (v3)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_cvd_divergence(
    h4_close:   pd.Series,
    h4_cvd:     pd.Series,
    base_index: pd.DatetimeIndex,
    window:     int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    CVD Divergence — proxy akumulasi/distribusi smart money di H4.

    cvd_div_h4:
      +1 = bullish divergence (harga LL tapi CVD tidak) → akumulasi
      -1 = bearish divergence (harga HH tapi CVD tidak) → distribusi
       0 = tidak ada divergence

    cvd_slope_h4:
      Rate of change CVD di H4, mengukur momentum smart money.
      Positif = net buying, negatif = net selling.
    """
    # ── Align h4_cvd ke h4_close index ─────────────────────────────────────
    # h4_cvd bisa berasal dari resample ("4h".last()) yang punya frekuensi 4h
    # asli (~8.353 bar), sementara h4_close dari df forward-filled ke H1
    # (33.410 bar). Resample h4_close ke 4h agar keduanya match.
    h4_close_4h = h4_close.resample("4h").last()
    h4_cvd_4h   = h4_cvd.reindex(h4_close_4h.index).ffill()
    # ───────────────────────────────────────────────────────────────────────

    price_chg = h4_close_4h.diff(window)
    cvd_chg   = h4_cvd_4h.diff(window)

    div_raw = np.where(
        (price_chg > 0) & (cvd_chg < 0), -1.0,    # distribusi
        np.where(
            (price_chg < 0) & (cvd_chg > 0), 1.0,  # akumulasi
            0.0
        )
    )
    cvd_div_h4_raw  = pd.Series(div_raw, index=h4_close_4h.index)
    cvd_slope_raw   = h4_cvd_4h.diff(window) / (h4_cvd_4h.abs().rolling(window).mean() + 1e-10)

    # Align ke base index (H1) — ffill saja, tidak ada interpolasi
    cvd_div_h4 = cvd_div_h4_raw.reindex(
        cvd_div_h4_raw.index.union(base_index)
    ).ffill().reindex(base_index).fillna(0.0)

    cvd_slope_h4 = cvd_slope_raw.reindex(
        cvd_slope_raw.index.union(base_index)
    ).ffill().reindex(base_index).fillna(0.0)

    return cvd_div_h4, cvd_slope_h4


def calc_volume_absorption(
    high:   pd.Series,
    low:    pd.Series,
    volume: pd.Series,
    atr:    pd.Series,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Volume Absorption — mendeteksi smart money menyerap order.

    vol_efficiency:
      volume / (range candle / ATR)
      Tinggi = banyak volume relatif terhadap pergerakan harga = absorption

    absorption_z:
      Z-score rolling vol_efficiency.
      Nilai tinggi di dekat swing level = sinyal akumulasi/distribusi kuat.
    """
    candle_range   = (high - low).replace(0, np.nan)
    atr_safe       = atr.replace(0, np.nan)
    vol_efficiency = volume / (candle_range / atr_safe)

    vol_eff_mean = vol_efficiency.rolling(window, min_periods=5).mean()
    vol_eff_std  = vol_efficiency.rolling(window, min_periods=5).std().replace(0, np.nan)
    absorption_z = (vol_efficiency - vol_eff_mean) / vol_eff_std

    return vol_efficiency.fillna(0.0), absorption_z.fillna(0.0)


def calc_funding_price_divergence(
    close:        pd.Series,
    funding_rate: pd.Series,
    window:       int = 8,
) -> pd.Series:
    """
    Funding-Price Divergence — proxy posisi pasar vs pergerakan harga.

    Positif  = akumulasi (funding negatif ekstrem tapi harga tidak turun)
    Negatif  = distribusi (funding positif ekstrem tapi harga tidak naik)
    Magnitude = seberapa kuat sinyal divergence tersebut.
    """
    funding_ffill = funding_rate.ffill().fillna(0.0)
    price_ret     = close.pct_change(window).fillna(0.0)

    fr_mean = funding_ffill.rolling(window * 3, min_periods=window).mean()
    fr_std  = funding_ffill.rolling(window * 3, min_periods=window).std().replace(0, np.nan)
    fr_z    = (funding_ffill - fr_mean) / fr_std

    # Divergence: tanda funding berlawanan dengan tanda return harga
    # Diperkuat oleh magnitude z-score funding
    price_sign    = np.sign(price_ret).replace(0, np.nan).ffill().fillna(0.0)
    divergence    = -np.sign(fr_z) * price_sign * fr_z.abs()

    return divergence.fillna(0.0)


def calc_wyckoff_phase(
    price_in_range: pd.Series,
    vol_regime:     pd.Series,
    h4_trend:       pd.Series,
    cvd_slope_h4:   pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Wyckoff Phase Proxy — identifikasi fase siklus pasar.

    Phase:
      0 = Markup      : trend up,  CVD positif
      1 = Distribution: price di puncak range, volume tinggi, CVD melemah
      2 = Markdown    : trend down, CVD negatif
      3 = Accumulation: price di bawah range, volume tinggi, CVD membalik

    spring_upthrust:
      1 = spring (false breakdown) atau upthrust (false breakout)
      Ciri khas smart money sebelum reversal besar.
    """
    phase = np.zeros(len(price_in_range), dtype=int)

    trend_up   = (h4_trend == 1).values
    trend_down = (h4_trend == -1).values
    price_high = (price_in_range > 0.65).values
    price_low  = (price_in_range < 0.35).values
    vol_high   = (vol_regime > 1.3).values
    cvd_pos    = (cvd_slope_h4 > 0).values
    cvd_neg    = (cvd_slope_h4 < 0).values

    # Prioritas assignment: lebih spesifik override lebih umum
    phase[trend_up   & cvd_pos]              = 0   # Markup
    phase[price_high & vol_high & cvd_neg]   = 1   # Distribution
    phase[trend_down & cvd_neg]              = 2   # Markdown
    phase[price_low  & vol_high & cvd_pos]   = 3   # Accumulation

    phase_s = pd.Series(phase, index=price_in_range.index)

    # Spring: harga tiba-tiba dip ke bawah support, langsung balik naik
    # Upthrust: harga tiba-tiba spike ke atas resistance, langsung balik turun
    pir_shift       = price_in_range.shift(1)
    spring          = (price_in_range < 0.05) & (pir_shift > 0.10)
    upthrust        = (price_in_range > 0.95) & (pir_shift < 0.90)
    spring_upthrust = (spring | upthrust).astype(int)

    return phase_s, spring_upthrust


# ═══════════════════════════════════════════════════════════════════════════════
# SMART MONEY FEATURES v4 (BARU)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_ofi_features(
    taker_buy_vol: pd.Series,
    taker_sell_vol: pd.Series,
    window_z: int = 48,
    window_h4: str = "4h",
) -> dict[str, pd.Series]:
    """
    Order Flow Imbalance (OFI) — jejak smart money di order flow.

    ofi_raw         : OFI per bar (buy - sell taker volume)
    ofi_acceleration: percepatan akumulasi/distribusi (diff 3 bar)
    ofi_z_score     : OFI relatif terhadap rata-rata 48 jam
    ofi_h4_delta    : perubahan OFI agregat per H4
    """
    ofi_raw = (taker_buy_vol - taker_sell_vol).fillna(0)

    ofi_acceleration = ofi_raw.diff(3).fillna(0)

    ofi_mean = ofi_raw.rolling(window_z, min_periods=10).mean()
    ofi_std  = ofi_raw.rolling(window_z, min_periods=10).std().replace(0, np.nan)
    ofi_z_score = ((ofi_raw - ofi_mean) / ofi_std).fillna(0)

    ofi_h4_sum   = ofi_raw.resample(window_h4).sum()
    ofi_h4_delta = ofi_h4_sum.diff()
    ofi_h4_delta = ofi_h4_delta.reindex(
        ofi_h4_delta.index.union(ofi_raw.index)
    ).ffill().reindex(ofi_raw.index).fillna(0)

    return {
        "ofi_raw":          ofi_raw,
        "ofi_acceleration": ofi_acceleration,
        "ofi_z_score":      ofi_z_score,
        "ofi_h4_delta":     ofi_h4_delta,
    }


def calc_vwdp(
    open_: pd.Series,
    high:  pd.Series,
    low:   pd.Series,
    close: pd.Series,
    ofi_raw: pd.Series,
    window: int = 24,
) -> dict[str, pd.Series]:
    """
    Volume-Weighted Directional Pressure (VWDP).
    Mengukur tekanan directional disesuaikan rejection (wick).

    vwdp            : OFI × (1 - wick_ratio), proxy iceberg order detection
    vwdp_smooth     : rolling mean VWDP untuk trend tekanan
    """
    candle_range = (high - low).replace(0, np.nan)
    body         = (close - open_).abs()
    wick_ratio   = ((candle_range - body) / candle_range).fillna(0).clip(0, 1)

    vwdp        = (ofi_raw * (1 - wick_ratio)).fillna(0)
    vwdp_smooth = vwdp.rolling(window, min_periods=5).mean().fillna(0)

    return {
        "vwdp":        vwdp,
        "vwdp_smooth": vwdp_smooth,
    }


def calc_cvd_hidden_divergence(
    close: pd.Series,
    cvd:   pd.Series,
    window: int = 8,
    price_threshold: float = 0.02,
    cvd_threshold:   float = 0.1,
) -> dict[str, pd.Series]:
    """
    CVD Hidden Divergence — deteksi akumulasi/distribusi tersembunyi.

    hidden_bull_div : +1 jika harga LL tapi CVD tidak (akumulasi tersembunyi)
    hidden_bear_div : -1 jika harga HH tapi CVD tidak (distribusi tersembunyi)
    cvd_momentum    : rate of change CVD dinormalisasi
    """
    price_momentum = close.pct_change(window).fillna(0)
    cvd_ma         = cvd.abs().rolling(window, min_periods=3).mean().replace(0, np.nan)
    cvd_momentum   = (cvd.diff(window) / cvd_ma).fillna(0)

    hidden_bull = ((price_momentum < -price_threshold) &
                   (cvd_momentum > cvd_threshold)).astype(float)
    hidden_bear = ((price_momentum >  price_threshold) &
                   (cvd_momentum < -cvd_threshold)).astype(float) * -1

    hidden_divergence = hidden_bull + hidden_bear

    return {
        "hidden_divergence": hidden_divergence,
        "cvd_momentum_adv":  cvd_momentum,
    }


def calc_absorption_at_swing(
    close:          pd.Series,
    absorption_z:   pd.Series,
    h4_swing_highs: pd.Series,
    h4_swing_lows:  pd.Series,
    atr:            pd.Series,
    proximity_atr:  float = 0.5,
) -> pd.Series:
    """
    Absorption Detection di dekat Swing Level.
    Tinggi = smart money menyerap order di level struktural kunci.

    absorption_at_swing: absorption_z × proximity_factor
    Positif kuat = akumulasi di support (swing low)
    Negatif kuat = distribusi di resistance (swing high)
    """
    atr_safe = atr.replace(0, np.nan)

    near_swing_high = (
        (h4_swing_highs - close).abs() / atr_safe
    ).fillna(999) < proximity_atr

    near_swing_low = (
        (close - h4_swing_lows).abs() / atr_safe
    ).fillna(999) < proximity_atr

    # Di dekat swing high = potensi distribusi → negatif
    # Di dekat swing low  = potensi akumulasi  → positif
    proximity_signal = (
        near_swing_low.astype(float) - near_swing_high.astype(float)
    )

    absorption_at_swing = (absorption_z * proximity_signal).fillna(0)
    return absorption_at_swing


def calc_vsa_features(
    open_:  pd.Series,
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    volume: pd.Series,
    window_ultra: int = 48,
    window_avg:   int = 24,
) -> dict[str, pd.Series]:
    """
    Volume Spread Analysis (VSA) — effort vs result.

    spread_to_volume : (high-low) / volume. Rendah = banyak volume, sedikit gerak = absorption
    ultra_high_vol   : 1 jika volume > percentile 95 dalam 48 bar
    no_demand        : spread rendah + close > open (upbar tapi volume lemah = SM tidak mendukung)
    no_supply        : spread rendah + close < open (downbar tapi volume lemah = SM tidak menjual)
    effort_vs_result : divergence antara volume effort dan price result
    """
    candle_range     = (high - low).replace(0, np.nan)
    vol_safe         = volume.replace(0, np.nan)
    spread_to_volume = (candle_range / vol_safe).fillna(0)

    ultra_high_vol = (
        volume > volume.rolling(window_ultra, min_periods=10).quantile(0.95)
    ).astype(int)

    avg_spread = spread_to_volume.rolling(window_avg, min_periods=5).mean()
    low_spread = spread_to_volume < (avg_spread * 0.5)

    no_demand = (low_spread & (close > open_)).astype(int)
    no_supply = (low_spread & (close < open_)).astype(int)

    # Effort vs Result: volume tinggi tapi return kecil = absorption
    abs_return   = (close - open_).abs() / candle_range.fillna(1)
    vol_z        = (volume - volume.rolling(window_avg, min_periods=5).mean()) / \
                   volume.rolling(window_avg, min_periods=5).std().replace(0, np.nan)
    effort_vs_result = (vol_z * (1 - abs_return)).fillna(0)

    return {
        "spread_to_volume":  spread_to_volume.fillna(0),
        "ultra_high_vol":    ultra_high_vol,
        "no_demand":         no_demand,
        "no_supply":         no_supply,
        "effort_vs_result":  effort_vs_result,
    }


def calc_rsi_divergence(
    close:    pd.Series,
    rsi_h4:   pd.Series,
    window:   int = 5,
) -> pd.Series:
    """
    RSI Divergence (Regular dan Hidden) dari RSI H4.

    Regular bearish : harga HH, RSI tidak     → -1.0 (reversal down)
    Regular bullish : harga LL,  RSI tidak     → +1.0 (reversal up)
    Hidden bullish  : harga HL,  RSI HL        → +0.5 (continuation up)
    Hidden bearish  : harga LH,  RSI LH        → -0.5 (continuation down)
    """
    price_chg = close.diff(window)
    rsi_chg   = rsi_h4.diff(window)

    div = np.where(
        (price_chg > 0) & (rsi_chg < 0), -1.0,
        np.where(
            (price_chg < 0) & (rsi_chg > 0),  1.0,
            np.where(
                (price_chg > 0) & (rsi_chg > 0),  0.5,
                np.where(
                    (price_chg < 0) & (rsi_chg < 0), -0.5,
                    0.0
                )
            )
        )
    )
    return pd.Series(div, index=close.index).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TREND QUALITY FEATURES (correction detection)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_trend_acceleration(
    trend_strength: pd.Series,
    atr_h4: pd.Series,
    window: int = 4,
) -> pd.Series:
    """
    Rate of change trend_strength dalam N bar H4 (~16 jam).
    Negatif saat h4_trend masih UP = trend mulai kelelahan (warning LONG).
    Positif saat h4_trend masih DOWN = trend bearish melemah (warning SHORT).
    """
    return trend_strength.diff(window) / atr_h4.replace(0, np.nan)


def calc_volume_price_confirm(
    vol_regime: pd.Series,
    h4_trend: pd.Series,
) -> pd.Series:
    """
    Volume × arah trend H4.
    Nilai rendah (VolR 0.01 × trend DOWN = -0.01) = volume tidak konfirmasi.
    Nilai tinggi positif = volume mendukung uptrend.
    Menggabungkan vol_regime + h4_trend secara eksplisit agar LGBM tidak perlu
    menemukan interaksinya sendiri.
    """
    return vol_regime * h4_trend


def calc_dist_from_recent_high(
    high: pd.Series,
    close: pd.Series,
    atr_h1: pd.Series,
    window: int = 8,
) -> pd.Series:
    """
    Jarak harga dari highest high N bar terakhir (ATR-normalized).
    -0.3 = baru mulai turun dari puncak (koreksi awal, entry SHORT masih valid).
    -2.0 = sudah koreksi 2 ATR (mungkin sudah oversold, jangan entry SHORT).
     0.0 = harga di/dekat puncak (tidak ada koreksi).
    """
    roll_high = high.rolling(window, min_periods=3).max()
    return (close - roll_high) / atr_h1.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# SWING-BASED LABELING v3
# ═══════════════════════════════════════════════════════════════════════════════

def swing_based_labeling(
    close:          pd.Series,
    high:           pd.Series,
    low:            pd.Series,
    atr_base:       pd.Series,
    h4_swing_highs: pd.Series,
    h4_swing_lows:  pd.Series,
    max_hold:       int   = 48,
    min_rr:         float = 1.5,
    min_tp_atr:     float = 1.5,
    max_sl_atr:     float = 3.0,
    # ── Hybrid momentum path ─────────────────────────────────────────────────
    # Saat vol_spike_zscore ≥ momentum_threshold, gunakan ATR-based TP/SL
    # alih-alih swing H4. Menangkap pump/dump yang sering jadi FLAT di swing mode
    # karena swing H4 terlalu jauh saat market sedang trending kuat.
    momentum_vol_spike:   pd.Series | None = None,
    momentum_price_accel: pd.Series | None = None,
    momentum_threshold:   float = 1.5,
    momentum_tp_atr:      float = 2.0,
    momentum_sl_atr:      float = 1.5,
) -> pd.Series:
    """
    Hybrid labeling: swing H4 untuk bar normal, ATR-based untuk bar momentum.

    Normal path (vol_spike < threshold):
      TP = H4 swing high, SL = H4 swing low — structural levels

    Momentum path (vol_spike ≥ threshold):
      TP = close ± momentum_tp_atr × ATR  (2.0× default)
      SL = close ∓ momentum_sl_atr × ATR  (1.5× default)
      RR = 2.0/1.5 = 1.33 — always > min_rr=0.6
      Keduanya (LONG & SHORT) selalu valid, outcome scan menentukan arah.
      Tie-break: price_accel direction jika tersedia.

    Backward-looking guarantee: semua fitur (vol_spike, price_accel, ATR)
    dihitung dari bar i ke belakang — tidak ada lookahead.
    """
    n      = len(close)
    labels = np.full(n, "FLAT", dtype=object)

    c_arr  = close.values
    h_arr  = high.values
    l_arr  = low.values
    a_arr  = atr_base.values
    sh_arr = h4_swing_highs.values
    sl_arr = h4_swing_lows.values

    vs_arr = momentum_vol_spike.values   if momentum_vol_spike   is not None else None
    pa_arr = momentum_price_accel.values if momentum_price_accel is not None else None

    for i in range(n - 1):
        price = c_arr[i]
        atr_i = a_arr[i]

        if np.isnan(price) or np.isnan(atr_i) or atr_i == 0:
            continue

        # ── Tentukan path: momentum atau swing ───────────────────────────────
        is_momentum = (
            vs_arr is not None
            and not np.isnan(vs_arr[i])
            and abs(vs_arr[i]) >= momentum_threshold
        )

        if is_momentum:
            # ── Momentum path: ATR-based TP/SL ───────────────────────────────
            tp_dist = momentum_tp_atr * atr_i
            sl_dist = momentum_sl_atr * atr_i
            rr_mom  = tp_dist / sl_dist if sl_dist > 0 else 0.0

            mom_valid = (
                tp_dist >= min_tp_atr * atr_i
                and sl_dist <= max_sl_atr * atr_i
                and rr_mom >= min_rr
            )
            if not mom_valid:
                continue

            tp_long  = price + tp_dist
            sl_long  = price - sl_dist
            tp_short = price - tp_dist
            sl_short = price + sl_dist
            long_valid  = True
            short_valid = True
            rr_long  = rr_mom
            rr_short = rr_mom

        else:
            # ── Normal path: swing H4 ─────────────────────────────────────────
            swing_hi = sh_arr[i]
            swing_lo = sl_arr[i]
            if np.isnan(swing_hi) or np.isnan(swing_lo):
                continue

            tp_long      = swing_hi
            sl_long      = swing_lo
            tp_dist_long = tp_long - price
            sl_dist_long = price   - sl_long

            long_valid = (
                tp_dist_long > 0
                and sl_dist_long > 0
                and tp_dist_long >= min_tp_atr * atr_i
                and sl_dist_long <= max_sl_atr * atr_i
                and (tp_dist_long / sl_dist_long) >= min_rr
            )
            rr_long = tp_dist_long / sl_dist_long if long_valid and sl_dist_long > 0 else 0.0

            tp_short      = swing_lo
            sl_short      = swing_hi
            tp_dist_short = price    - tp_short
            sl_dist_short = sl_short - price

            short_valid = (
                tp_dist_short > 0
                and sl_dist_short > 0
                and tp_dist_short >= min_tp_atr * atr_i
                and sl_dist_short <= max_sl_atr * atr_i
                and (tp_dist_short / sl_dist_short) >= min_rr
            )
            rr_short = tp_dist_short / sl_dist_short if short_valid and sl_dist_short > 0 else 0.0

        if not long_valid and not short_valid:
            continue

        # ── Scan ke depan (sama untuk kedua path) ─────────────────────────────
        end           = min(i + max_hold, n)
        outcome_long  = "FLAT"
        outcome_short = "FLAT"

        for j in range(i + 1, end):
            if np.isnan(h_arr[j]) or np.isnan(l_arr[j]):
                continue

            if long_valid and outcome_long == "FLAT":
                if h_arr[j] >= tp_long:
                    outcome_long = "LONG"
                elif l_arr[j] <= sl_long:
                    outcome_long = "MISS"

            if short_valid and outcome_short == "FLAT":
                if l_arr[j] <= tp_short:
                    outcome_short = "SHORT"
                elif h_arr[j] >= sl_short:
                    outcome_short = "MISS"

            long_done  = not long_valid  or outcome_long  != "FLAT"
            short_done = not short_valid or outcome_short != "FLAT"
            if long_done and short_done:
                break

        # ── Assign label final ────────────────────────────────────────────────
        if long_valid and short_valid:
            if outcome_long == "LONG" and outcome_short != "SHORT":
                labels[i] = "LONG"
            elif outcome_short == "SHORT" and outcome_long != "LONG":
                labels[i] = "SHORT"
            elif outcome_long == "LONG" and outcome_short == "SHORT":
                # Keduanya menang — tie-break:
                # momentum path: pakai arah price_accel; normal path: pilih RR lebih tinggi
                if is_momentum and pa_arr is not None and not np.isnan(pa_arr[i]):
                    labels[i] = "LONG" if pa_arr[i] > 0 else "SHORT"
                else:
                    labels[i] = "LONG" if rr_long >= rr_short else "SHORT"
        elif long_valid:
            labels[i] = "LONG"  if outcome_long  == "LONG"  else "FLAT"
        elif short_valid:
            labels[i] = "SHORT" if outcome_short == "SHORT" else "FLAT"

    tail = min(max_hold, n)
    labels[-tail:] = "FLAT"

    return pd.Series(labels, index=close.index, name="label")


def triple_barrier_labeling(
    close:       pd.Series,
    high:        pd.Series,
    low:         pd.Series,
    atr_base:    pd.Series,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    max_hold:    int   = 36,
) -> pd.Series:
    """
    Triple Barrier Labeling (Lopez de Prado).

    Setiap bar di-evaluate untuk LONG dan SHORT secara independen:
      LONG:  TP = close + tp_atr × ATR,  SL = close - sl_atr × ATR
      SHORT: TP = close - tp_atr × ATR,  SL = close + sl_atr × ATR

    Scan ke depan max_hold bar — label = barrier pertama yang tercapai.
      TP_LONG hit dulu  → LONG
      TP_SHORT hit dulu → SHORT
      Keduanya → pilih berdasarkan siapa yang lebih cepat
      Tidak ada → FLAT (time barrier)

    RR = tp_atr_mult / sl_atr_mult (konstan untuk semua bar dan koin).
    Label lebih selaras dengan sistem trading nyata (ATR-based TP/SL).
    """
    n       = len(close)
    labels  = np.full(n, "FLAT", dtype=object)
    c_arr   = close.values
    h_arr   = high.values
    l_arr   = low.values
    atr_arr = atr_base.values

    for i in range(n - 1):
        price = c_arr[i]
        atr_i = atr_arr[i]
        if np.isnan(price) or np.isnan(atr_i) or atr_i <= 0:
            continue

        tp_long  = price + tp_atr_mult * atr_i
        sl_long  = price - sl_atr_mult * atr_i
        tp_short = price - tp_atr_mult * atr_i
        sl_short = price + sl_atr_mult * atr_i

        end = min(i + max_hold, n)
        long_bar  = None   # bar index dimana LONG outcome tercapai
        short_bar = None   # bar index dimana SHORT outcome tercapai
        long_out  = "FLAT"
        short_out = "FLAT"

        for j in range(i + 1, end):
            hj, lj = h_arr[j], l_arr[j]
            if np.isnan(hj) or np.isnan(lj):
                continue

            if long_bar is None:
                if hj >= tp_long:
                    long_out, long_bar = "LONG", j
                elif lj <= sl_long:
                    long_out, long_bar = "MISS", j

            if short_bar is None:
                if lj <= tp_short:
                    short_out, short_bar = "SHORT", j
                elif hj >= sl_short:
                    short_out, short_bar = "MISS", j

            if long_bar is not None and short_bar is not None:
                break

        # Tentukan label final
        long_win  = long_out  == "LONG"
        short_win = short_out == "SHORT"

        if long_win and not short_win:
            labels[i] = "LONG"
        elif short_win and not long_win:
            labels[i] = "SHORT"
        elif long_win and short_win:
            # Keduanya menang — pilih yang lebih cepat
            labels[i] = "LONG" if (long_bar or 9999) <= (short_bar or 9999) else "SHORT"
        # else: FLAT (tidak ada yang menang dalam max_hold)

    # Bar terakhir tidak bisa dilabeli (tidak ada data ke depan)
    tail = min(max_hold, n)
    labels[-tail:] = "FLAT"

    return pd.Series(labels, index=close.index, name="label")


def hybrid_labeling(
    close:          pd.Series,
    high:           pd.Series,
    low:            pd.Series,
    atr_base:       pd.Series,
    h4_swing_highs: pd.Series,
    h4_swing_lows:  pd.Series,
    # Swing params
    swing_weight:   float = 0.6,
    max_hold:       int   = 36,
    min_rr:         float = 0.6,
    min_tp_atr:     float = 1.2,
    max_sl_atr:     float = 4.0,
    # Triple Barrier params
    tb_weight:      float = 0.4,
    tp_atr_mult:    float = 3.0,
    sl_atr_mult:    float = 1.5,
    # Threshold untuk final label
    threshold:      float = 0.4,
) -> pd.Series:
    """
    Hybrid labeling: weighted combination swing label + Triple Barrier label.

    score = swing_weight × swing_ordinal + tb_weight × tb_ordinal
    dimana ordinal: LONG=+1, FLAT=0, SHORT=-1

    if score >=  threshold → LONG  (swing dan/atau TB mendukung LONG)
    if score <= -threshold → SHORT
    else                   → FLAT

    Dengan swing_weight=0.6, tb_weight=0.4, threshold=0.4:
      Both LONG (1.0):        → LONG  ✓ (sangat kuat)
      Swing LONG + TB FLAT (0.6): → LONG  ✓ (swing saja cukup)
      Swing FLAT + TB LONG (0.4): → LONG  ✓ (borderline)
      Swing LONG + TB SHORT (0.2): → FLAT  ✓ (konflik = tidak masuk)
      Both FLAT (0.0):         → FLAT  ✓
    """
    ORDINAL = {"LONG": 1, "FLAT": 0, "SHORT": -1}

    # Compute swing labels
    swing = swing_based_labeling(
        close=close, high=high, low=low, atr_base=atr_base,
        h4_swing_highs=h4_swing_highs, h4_swing_lows=h4_swing_lows,
        max_hold=max_hold, min_rr=min_rr,
        min_tp_atr=min_tp_atr, max_sl_atr=max_sl_atr,
    )

    # Compute Triple Barrier labels
    tb = triple_barrier_labeling(
        close=close, high=high, low=low, atr_base=atr_base,
        tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult,
        max_hold=max_hold,
    )

    # Ordinal encoding
    swing_ord = swing.map(ORDINAL).fillna(0).values.astype(float)
    tb_ord    = tb.map(ORDINAL).fillna(0).values.astype(float)

    # Weighted score
    score = swing_weight * swing_ord + tb_weight * tb_ord

    # Final label
    labels = np.where(score >= threshold, "LONG",
             np.where(score <= -threshold, "SHORT", "FLAT"))

    # Anti-lookahead: tail bars force FLAT
    tail = min(max_hold, len(close))
    labels[-tail:] = "FLAT"

    return pd.Series(labels, index=close.index, name="label")


def structural_label_filter(
    labels:                  pd.Series,
    feat_df:                 pd.DataFrame,
    long_max_price_in_range:  float = 1.0,
    short_min_price_in_range: float = 0.0,
) -> pd.Series:
    """
    Filter label berdasarkan posisi harga dalam range swing.

    LONG di-override ke FLAT jika price_in_range > threshold
    (harga sudah terlalu tinggi, bukan zona beli yang baik).

    SHORT di-override ke FLAT jika price_in_range < threshold
    (harga sudah terlalu rendah, bukan zona jual yang baik).
    """
    filtered = labels.copy()
    if "price_in_range" not in feat_df.columns:
        return filtered

    pir = feat_df["price_in_range"]
    filtered[(labels == "LONG")  & (pir > long_max_price_in_range)]  = "FLAT"
    filtered[(labels == "SHORT") & (pir < short_min_price_in_range)] = "FLAT"

    n_long_override  = int(((labels == "LONG")  & (pir > long_max_price_in_range)).sum())
    n_short_override = int(((labels == "SHORT") & (pir < short_min_price_in_range)).sum())
    logger.info(
        f"Structural filter: {n_long_override} LONG → FLAT, "
        f"{n_short_override} SHORT → FLAT"
    )
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE ENGINEERING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    symbol: str,
    symbol_id: int,
    # Labeling parameters
    max_hold:  int   = 24,    # bar H1 = 24 jam
    min_rr:    float = 1.2,
    min_tp_atr: float = 1.2,
    max_sl_atr: float = 3.0,
    long_max_price_in_range:  float = 1.0,
    short_min_price_in_range: float = 0.0,
    # Feature parameters (H1-adjusted)
    vp_window:      int   = 24,    # Volume Profile: 24 bar H1 = 24 jam
    vp_bins:        int   = 50,
    swing_lookback: int   = 5,
    fvg_min_gap:    float = 0.5,
    swing_rolling_bars: int = 24,  # Swing structure: 24 bar H1 = 24 jam
    # Misc
    add_label: bool = True,
    # Legacy parameters (diabaikan, untuk backward compat signature)
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Hitung semua 100 fitur v3 dari cleaned DataFrame (H1 base).

    Input:
        df — output dari 02_clean.py dengan kolom prefixed:
             "1h_open/high/low/close/volume"
             "4h_open/high/low/close" (untuk H4 context)
             "1d_open/high/low/close" (untuk D1 HTF context)
             "funding_rate_*", "macro_*" (optional)

    Output:
        DataFrame dengan 100 fitur + label (jika add_label=True)
    """
    df = ensure_utc_index(df)

    # ── 1. Extract base OHLCV (H1) ───────────────────────────────────────────
    o, h, l, c, v = _get_ohlcv(df, "1h")

    # Fallback jika prefix "1h_" tidak ditemukan (kolom tidak ber-prefix)
    if c.isna().all():
        logger.warning(f"[{symbol}] Kolom 1h_ tidak ditemukan, coba tanpa prefix...")
        o = df.get("open",   pd.Series(np.nan, index=df.index))
        h = df.get("high",   pd.Series(np.nan, index=df.index))
        l = df.get("low",    pd.Series(np.nan, index=df.index))
        c = df.get("close",  pd.Series(np.nan, index=df.index))
        v = df.get("volume", pd.Series(np.nan, index=df.index))

    if c.isna().all():
        raise ValueError(f"[{symbol}] Tidak ada data OHLCV yang valid!")

    # ── 2. ATR H1 dan H4 ─────────────────────────────────────────────────────
    atr_h1   = calc_atr(h, l, c, 14)
    atr_safe = atr_h1.replace(0, np.nan)

    # H4 OHLCV — dua varian:
    #   _closed : dari candle H4 yang sudah tutup (ffilled ke H1) — untuk swing & ATR
    #   partial : expanding max/min dalam periode H4 berjalan — untuk trend & EMA & RSI
    h4_h_closed = df.get("4h_high",  h)
    h4_l_closed = df.get("4h_low",   l)
    h4_c_closed = df.get("4h_close", c)

    _h4_floor = df.index.floor("4h")
    h4_h = h.groupby(_h4_floor).expanding().max().droplevel(0)
    h4_l = l.groupby(_h4_floor).expanding().min().droplevel(0)
    h4_c = c   # partial H4 close = current H1 close

    # ATR H4 dari closed candles — ATR representatif periode penuh, bukan partial
    atr_h4_raw = calc_atr(h4_h_closed, h4_l_closed, h4_c_closed, 14)
    atr_h4 = atr_h4_raw.reindex(
        atr_h4_raw.index.union(df.index)
    ).ffill().reindex(df.index)

    # ── 3. Inisialisasi dict fitur ────────────────────────────────────────────
    feat: dict[str, pd.Series] = {}

    # ── 4. OHLCV base ─────────────────────────────────────────────────────────
    feat["open"]   = o
    feat["high"]   = h
    feat["low"]    = l
    feat["close"]  = c
    feat["volume"] = v

    # ── 5. Volume Flow ────────────────────────────────────────────────────────
    feat["volume_delta"] = calc_volume_delta(df)
    feat["cvd"]          = calc_cvd(df)

    buy_col  = _col(df, "taker_buy_volume",  "1h_taker_buy_volume")
    sell_col = _col(df, "taker_sell_volume", "1h_taker_sell_volume")
    feat["buy_volume"]  = df[buy_col]  if buy_col  else (v * 0.5)
    feat["sell_volume"] = df[sell_col] if sell_col else (v * 0.5)

    # ── 6. H4 CVD (untuk smart money divergence) ──────────────────────────────
    cvd_series  = feat["cvd"]
    h4_cvd_raw  = cvd_series.resample("4h").last()

    # ── 7. H4 Swing Points (untuk labeling) ───────────────────────────────────
    h4_sh_raw, h4_sl_raw = detect_h4_swing_points(h4_h, h4_l, lookback=3)
    h4_swing_highs, h4_swing_lows = get_nearest_swing_levels(
        h4_swing_highs = h4_sh_raw,
        h4_swing_lows  = h4_sl_raw,
        base_index     = df.index,
    )
    # Shift forward to remove 3-bar look-ahead bias from swing detection.
    # detect_h4_swing_points uses i-3..i+3 → the swing at bar i is confirmed
    # by future bars. Shifting forward by lookback bars ensures features and
    # labels only use swing info available at bar i.
    h4_swing_highs = h4_swing_highs.shift(3)
    h4_swing_lows  = h4_swing_lows.shift(3)

    # ── 8. Market Structure ───────────────────────────────────────────────────
    bos, choch, bars_since = calc_market_structure(h, l, c, swing_lookback)
    feat["MSB_BOS"]        = bos
    feat["CHoCH"]          = choch
    feat["bars_since_BOS"] = bars_since

    # ── 9. FVG ────────────────────────────────────────────────────────────────
    fvg_up, fvg_down = calc_fvg(h, l, atr_h1, fvg_min_gap)
    feat["FVG_up"]   = fvg_up
    feat["FVG_down"] = fvg_down

    # ── 10. Liquidity & SFP ───────────────────────────────────────────────────
    buy_liq, sell_liq, sfp = calc_liquidity_levels(h, l, c, atr_h1, swing_lookback)
    feat["Buy_Liq"]   = buy_liq
    feat["Sell_Liq"]  = sell_liq
    feat["SFP_sweep"] = sfp

    # ── 11. Open Interest ─────────────────────────────────────────────────────
    oi_col = _col(df, "open_interest", "open_interest_open_interest", "open_interest_openInterest")
    if oi_col and not df[oi_col].isna().all():
        feat["open_interest"] = df[oi_col]
    else:
        temp_df = pd.DataFrame({"cvd": feat["cvd"], "volume": v}, index=df.index)
        feat["open_interest"] = compute_synthetic_oi(
            temp_df,
            cvd_window  = SYNTHETIC_OI_CVD_WINDOW,
            norm_window = SYNTHETIC_OI_NORM_WINDOW,
        )

    # ── 11b. Dynamic Position Pressure ────────────────────────────────────────
    # Rasio dari delta CVD (tekanan taker) terhadap delta Open Interest (perubahan posisi).
    # Menunjukkan apakah open interest baru didorong oleh taker agresif (buyer / seller).
    cvd_delta = feat["volume_delta"]
    oi_delta  = feat["open_interest"].diff().replace(0, np.nan)
    feat["dynamic_position_pressure"] = (cvd_delta / oi_delta).ffill().fillna(0.0).clip(-10.0, 10.0)

    # ── 12. Funding Rate ──────────────────────────────────────────────────────
    fr_col = _col(df, "funding_rate_funding_rate", "funding_rate_fundingRate", "funding_rate")
    funding_rate = df[fr_col] if fr_col else pd.Series(0.0, index=df.index)
    feat["funding_rate"] = funding_rate

    # ── 13. EMA H1 (ATR-normalized) ───────────────────────────────────────────
    # Format: (EMA - close) / ATR  → negatif = harga di atas EMA
    ema_h1_raw: dict[int, pd.Series] = {}
    for span in (7, 21, 50, 200):
        raw_h1            = calc_ema(c, span)
        ema_h1_raw[span]  = raw_h1
        feat[f"ema_{span}_h1"] = (raw_h1 - c) / atr_safe

    # ── 14. EMA H4 (ATR-normalized, aligned ke H1) ────────────────────────────
    ema_h4_raw: dict[int, pd.Series] = {}
    for span in (7, 21, 50, 200):
        raw_ema       = calc_ema(h4_c, span)
        aligned_ema   = raw_ema.reindex(
            raw_ema.index.union(df.index)
        ).ffill().reindex(df.index)
        ema_h4_raw[span]         = aligned_ema
        feat[f"ema_{span}_h4"]   = (aligned_ema - c) / atr_safe

    # ── 14b. EMA H4 Slopes & Price vs EMA (trend dynamics) ───────────────────
    ema_21_aligned  = ema_h4_raw[21]
    ema_50_aligned  = ema_h4_raw[50]
    ema_200_aligned = ema_h4_raw[200]
    feat["ema_21_slope_h4"]    = (ema_21_aligned - ema_21_aligned.shift(4)) / atr_safe
    feat["ema_50_slope_h4"]    = (ema_50_aligned - ema_50_aligned.shift(4)) / atr_safe
    feat["price_vs_ema_50_h4"] = (c - ema_50_aligned) / atr_safe

    # ── 14c. EMA Distance Features (NEW) ──────────────────────────────────────
    # Format: (close - EMA) / ATR → positif = harga di atas EMA (bullish bias)
    # H1 distances
    feat["price_vs_ema_21_h1"]  = (c - ema_h1_raw[21])  / atr_safe  # jarak ke EMA21 H1 (short-term S/R)
    feat["price_vs_ema_200_h1"] = (c - ema_h1_raw[200]) / atr_safe  # jarak ke MA200 H1 (long-term bias)
    # H4 distances
    feat["price_vs_ema_21_h4"]  = (c - ema_21_aligned)  / atr_safe  # jarak ke EMA21 H4
    feat["price_vs_ema_200_h4"] = (c - ema_200_aligned) / atr_safe  # jarak ke EMA200 H4 (macro trend)
    # EMA cross magnitude: gap antara EMA cepat (7) dan EMA lambat (21) H4
    # Positif = bullish alignment (EMA7 > EMA21), magnitude = kekuatan momentum
    feat["ema_cross_h4"]        = (ema_h4_raw[7] - ema_21_aligned)  / atr_safe

    # ── 15. RSI & StochRSI (H1) ───────────────────────────────────────────────
    feat["rsi_6"]          = calc_rsi(c, 6)
    feat["stochrsi_k"], feat["stochrsi_d"] = calc_stochrsi(c)

    # ── 16. ATR ───────────────────────────────────────────────────────────────
    feat["atr_14_h1"] = atr_h1
    feat["atr_14_h4"] = atr_h4

    # ── 17. Previous Day/Week High/Low ────────────────────────────────────────
    feat.update(calc_prev_day_week_levels(h, l, c, atr_h1))

    # ── 18. Fibonacci Levels ──────────────────────────────────────────────────
    feat.update(calc_fib_levels(h, l, c, atr_h1, window=swing_rolling_bars))

    # ── 19. Volume Profile ────────────────────────────────────────────────────
    poc, vah, val = calc_volume_profile(h, l, c, v, vp_window, vp_bins)
    feat["POC"] = (poc - c) / atr_safe
    feat["VAH"] = (vah - c) / atr_safe
    feat["VAL"] = (val - c) / atr_safe

    # ── 20. Macro ─────────────────────────────────────────────────────────────
    btc_col = _col(df, "macro_btc_dominance_btc_dominance_pct",
                       "macro_btc_dominance_pct", "btc_dominance_pct", "btc_dominance")
    feat["btc_dominance"] = df[btc_col] if btc_col else pd.Series(np.nan, index=df.index)

    fg_col = _col(df, "macro_fear_greed_index_fear_greed",
                      "macro_fear_greed_index_value", "fear_greed")
    feat["fear_greed"] = df[fg_col] if fg_col else pd.Series(np.nan, index=df.index)

    feat["market_session"] = calc_market_session(df.index)

    # ── 21. Log Returns & Volume Ratio ────────────────────────────────────────
    feat["log_ret_1"]  = np.log(c / c.shift(1)).fillna(0)
    feat["log_ret_5"]  = np.log(c / c.shift(5)).fillna(0)
    feat["log_ret_20"] = np.log(c / c.shift(20)).fillna(0)

    vol_ma20 = v.rolling(20, min_periods=5).mean()
    feat["vol_ratio_20"] = v / vol_ma20.replace(0, np.nan)

    # ── 22. Time Cyclical ─────────────────────────────────────────────────────
    feat.update(calc_cyclic_time(df.index))
    feat["time_to_funding_norm"] = calc_time_to_funding(df.index)

    # ── 23. Long/Short Ratio ──────────────────────────────────────────────────
    # Gunakan data real jika tersedia dan memiliki variasi cukup (>10% non-NaN).
    # Fallback ke synthetic proxy jika data tidak ada atau semua 0/NaN —
    # mencegah feature distribution shift antara training dan inference.
    ls_col = _col(df, "long_short_ratio", "long_short_ratio_long_short_ratio", "globalLongShortAccountRatio")
    real_ls = df[ls_col] if ls_col else None
    use_real = (
        real_ls is not None
        and real_ls.notna().mean() > 0.1
        and real_ls.abs().gt(0).mean() > 0.1
    )
    if use_real:
        feat["long_short_ratio"] = real_ls
    else:
        # Synthetic proxy: directional volume pressure, normalized terhadap MA-24.
        # Distribusi: ~0.5–2.0 (mirip real L/S ratio), center di 1.0 = netral.
        vol_delta = v * (c - o) / c.replace(0, np.nan)
        vol_ma24  = v.rolling(24, min_periods=5).mean().replace(0, np.nan)
        feat["long_short_ratio"] = (1.0 + vol_delta / vol_ma24).clip(0.1, 5.0)

    # ── 24. Swing Structure Features (v2, dipertahankan) ──────────────────────
    roll_high = h.rolling(swing_rolling_bars, min_periods=5).max()
    roll_low  = l.rolling(swing_rolling_bars, min_periods=5).min()
    roll_rng  = (roll_high - roll_low).replace(0, np.nan)

    feat["dist_swing_high"] = (c - roll_high) / atr_safe   # ≤ 0 jika di bawah high
    feat["dist_swing_low"]  = (c - roll_low)  / atr_safe   # ≥ 0 jika di atas low
    feat["price_in_range"]  = (c - roll_low)  / roll_rng   # [0, 1]
    feat["swing_momentum"]  = feat["price_in_range"] - feat["price_in_range"].shift(4)

    # ── 26. Market Regime Features (v2, dipertahankan) ────────────────────────
    ema7_h4  = ema_h4_raw[7]
    ema21_h4 = ema_h4_raw[21]
    ema50_h4 = ema_h4_raw[50]

    h4_trend = pd.Series(
        np.where(ema7_h4 > ema21_h4, 1, np.where(ema7_h4 < ema21_h4, -1, 0)),
        index=df.index,
    )
    feat["h4_trend"]       = h4_trend
    feat["trend_strength"] = (ema7_h4 - ema50_h4) / atr_h4.replace(0, np.nan)

    vol_ma_regime = v.rolling(swing_rolling_bars, min_periods=5).mean().replace(0, np.nan)
    feat["vol_regime"] = v / vol_ma_regime

    # ── 27. Smart Money Features (v3 BARU) ────────────────────────────────────

    # CVD Divergence H4
    cvd_div, cvd_slope = calc_cvd_divergence(
        h4_close   = h4_c,
        h4_cvd     = h4_cvd_raw,
        base_index = df.index,
        window     = 5,
    )
    feat["cvd_div_h4"]   = cvd_div
    feat["cvd_slope_h4"] = cvd_slope

    # Volume Absorption
    vol_eff, absorption_z = calc_volume_absorption(h, l, v, atr_h1, window=20)
    feat["vol_efficiency"] = vol_eff
    feat["absorption_z"]   = absorption_z

    # Funding-Price Divergence
    feat["funding_price_div"] = calc_funding_price_divergence(
        close        = c,
        funding_rate = funding_rate,
        window       = 8,
    )

    # RSI H4 dan RSI Divergence
    rsi_h4_series     = calc_rsi_h4(h4_c, c, period=14)
    feat["rsi_h4"]        = rsi_h4_series
    feat["rsi_divergence"] = calc_rsi_divergence(c, rsi_h4_series, window=5)

    # ── 27b. H4 Trend Dynamics (slope-based) ──────────────────────────────────
    # RSI slope — akselerasi momentum (4 bar = ~16 jam)
    feat["rsi_slope_h4"] = rsi_h4_series - rsi_h4_series.shift(4)

    # ATR percent — volatilitas relatif terhadap harga
    feat["atr_percent_h4"] = atr_h4 / c.replace(0, np.nan) * 100

    # Range expansion — deteksi ekspansi/konstraksi H4
    prev_range    = (h4_h - h4_l).shift(1)
    current_range = h4_h - h4_l
    feat["range_expansion_h4"] = current_range / prev_range.replace(0, np.nan)

    # ── Wyckoff Phase (bergantung pada fitur sebelumnya) ────────────────────────
    price_in_range_clean = feat["price_in_range"].fillna(0.5)
    vol_regime_clean     = feat["vol_regime"].fillna(1.0)
    h4_trend_clean       = feat["h4_trend"].fillna(0)
    cvd_slope_clean      = feat["cvd_slope_h4"].fillna(0.0)

    wyckoff_phase, spring_upthrust = calc_wyckoff_phase(
        price_in_range = price_in_range_clean,
        vol_regime     = vol_regime_clean,
        h4_trend       = h4_trend_clean,
        cvd_slope_h4   = cvd_slope_clean,
    )
    feat["wyckoff_phase"]   = wyckoff_phase
    feat["spring_upthrust"] = spring_upthrust

    # ── 28. Smart Money Features v4 (BARU) ───────────────────────────────────

    # Ambil taker volume
    buy_vol_series  = feat.get("buy_volume",  v * 0.5)
    sell_vol_series = feat.get("sell_volume", v * 0.5)

    # OFI Features
    ofi_feats = calc_ofi_features(
        taker_buy_vol  = buy_vol_series,
        taker_sell_vol = sell_vol_series,
        window_z       = 48,
        window_h4      = "4h",
    )
    feat.update(ofi_feats)

    # VWDP
    vwdp_feats = calc_vwdp(
        open_   = o,
        high    = h,
        low     = l,
        close   = c,
        ofi_raw = ofi_feats["ofi_raw"],
        window  = 24,
    )
    feat.update(vwdp_feats)

    # CVD Hidden Divergence
    cvd_div_feats = calc_cvd_hidden_divergence(
        close  = c,
        cvd    = feat["cvd"],
        window = 8,
    )
    feat.update(cvd_div_feats)

    # Absorption at Swing
    feat["absorption_at_swing"] = calc_absorption_at_swing(
        close          = c,
        absorption_z   = feat["absorption_z"],
        h4_swing_highs = h4_swing_highs,
        h4_swing_lows  = h4_swing_lows,
        atr            = atr_h1,
        proximity_atr  = 0.5,
    )

    # VSA Features
    vsa_feats = calc_vsa_features(
        open_  = o,
        high   = h,
        low    = l,
        close  = c,
        volume = v,
        window_ultra = 48,
        window_avg   = 24,
    )
    feat.update(vsa_feats)

    # ── 28b. Trend Quality Features (correction detection) ────────────────────
    feat["trend_accel_4h"]    = calc_trend_acceleration(
        trend_strength = feat["trend_strength"],
        atr_h4         = atr_h4,
        window         = 4,
    )
    feat["vol_price_confirm"] = calc_volume_price_confirm(
        vol_regime = feat["vol_regime"],
        h4_trend   = feat["h4_trend"],
    )
    feat["dist_from_8h_high"] = calc_dist_from_recent_high(
        high   = h,
        close  = c,
        atr_h1 = atr_h1,
        window = 8,
    )

    # ── 28b. Game Changer Features (v4.0) ──────────────────────────────────────
    # Feature 1: Altcoin Relative Strength (RS) & Weakness (RW) vs BTC
    btc_close = df.get("btc_close", c)  # Fallback ke c jika btc_close tidak ada
    alt_btc_ratio = c / btc_close.replace(0, np.nan)
    alt_btc_ratio_sma24 = alt_btc_ratio.rolling(24, min_periods=5).mean()
    alt_btc_ratio_std24 = alt_btc_ratio.rolling(24, min_periods=5).std()
    
    feat["relative_strength_z"] = ((alt_btc_ratio - alt_btc_ratio_sma24) / 
                                   alt_btc_ratio_std24.replace(0, np.nan)).fillna(0.0).clip(-5.0, 5.0)
    feat["relative_strength_momentum"] = feat["relative_strength_z"].diff(4).fillna(0.0)

    # Feature 2: Liquidation Cascade Proxy
    # Estimasi harga likuidasi untuk long leverage 50x dan 20x di bawah swing low
    # Estimasi harga likuidasi untuk short leverage 50x dan 20x di atas swing high
    liq_50x_long = h4_swing_lows * 0.98
    liq_20x_long = h4_swing_lows * 0.95
    liq_50x_short = h4_swing_highs * 1.02
    liq_20x_short = h4_swing_highs * 1.05

    feat["dist_liq_50x_long"]  = ((c - liq_50x_long) / atr_safe).fillna(0.0)
    feat["dist_liq_20x_long"]  = ((c - liq_20x_long) / atr_safe).fillna(0.0)
    feat["dist_liq_50x_short"] = ((liq_50x_short - c) / atr_safe).fillna(0.0)
    feat["dist_liq_20x_short"] = ((liq_20x_short - c) / atr_safe).fillna(0.0)

    # Feature 3: Whale vs Retail Divergence
    # L/S ratio z-score vs CVD z-score divergence
    ls_ratio = feat.get("long_short_ratio", pd.Series(1.0, index=df.index))
    cvd_series = feat.get("cvd", pd.Series(0.0, index=df.index))
    
    ls_sma24 = ls_ratio.rolling(24, min_periods=5).mean()
    ls_std24 = ls_ratio.rolling(24, min_periods=5).std().replace(0, np.nan)
    ls_z = (ls_ratio - ls_sma24) / ls_std24
    
    cvd_sma24 = cvd_series.rolling(24, min_periods=5).mean()
    cvd_std24 = cvd_series.rolling(24, min_periods=5).std().replace(0, np.nan)
    cvd_z = (cvd_series - cvd_sma24) / cvd_std24
    
    feat["whale_retail_divergence"] = (cvd_z - ls_z).fillna(0.0).clip(-5.0, 5.0)

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
    ).fillna(0.0).clip(-5.0, 5.0)

    # ── 28c. Momentum Acceleration Features (v4.2) ───────────────────────────
    # Tujuan: menangkap sinyal pump/dump yang LGBM sering miss karena threshold
    # terlalu tinggi saat momentum baru saja mulai terbentuk.
    # Semua fitur backward-looking, tidak ada leakage.

    # Feature 7: Price Acceleration (2nd derivative log return)
    # Positif = harga mulai akselerasi naik, negatif = akselerasi turun.
    # Berbeda dari log_ret_1 (kecepatan) — ini adalah PERUBAHAN kecepatan.
    log_ret_1_raw = np.log(c / c.shift(1)).fillna(0)
    feat["price_accel_1h"] = (log_ret_1_raw - log_ret_1_raw.shift(1)).fillna(0.0).clip(-0.1, 0.1)

    # Feature 8: OFI Momentum Ratio (short-term vs long-term OFI)
    # Nilai > 1: order flow beli/jual lebih intens dari rata-rata → momentum building
    # Nilai < 1: order flow melemah → momentum fading
    ofi_raw_series  = feat.get("ofi_raw", pd.Series(0.0, index=df.index))
    ofi_short_ma    = ofi_raw_series.rolling(3,  min_periods=1).mean()
    ofi_long_ma     = ofi_raw_series.rolling(24, min_periods=5).mean().replace(0, np.nan)
    feat["ofi_momentum_ratio"] = (ofi_short_ma / ofi_long_ma).fillna(1.0).clip(-10.0, 10.0)

    # Feature 9: Volume Acceleration (perubahan vol_ratio_20 dalam 3 bar)
    # Positif = volume naik lebih cepat dari baseline → surge dimulai
    # Negatif = volume mulai surut → momentum melemah
    vol_ratio_series   = feat.get("vol_ratio_20", pd.Series(1.0, index=df.index))
    feat["vol_accel_3h"] = (vol_ratio_series - vol_ratio_series.shift(3)).fillna(0.0).clip(-5.0, 5.0)

    # ── 29. Build DataFrame ───────────────────────────────────────────────────
    feat_df = pd.DataFrame(feat, index=df.index)
    feat_df = ensure_utc_index(feat_df)

    # ── 30. Labeling (Swing-Based v3) ─────────────────────────────────────────
    if add_label:
        logger.info(
            f"[{symbol}] Swing-Based labeling v3 "
            f"(max_hold={max_hold}h, min_rr={min_rr}, "
            f"min_tp={min_tp_atr}×ATR, max_sl={max_sl_atr}×ATR)..."
        )

        raw_labels = swing_based_labeling(
            close          = c,
            high           = h,
            low            = l,
            atr_base       = atr_h1,
            h4_swing_highs = h4_swing_highs,
            h4_swing_lows  = h4_swing_lows,
            max_hold       = max_hold,
            min_rr         = min_rr,
            min_tp_atr     = min_tp_atr,
            max_sl_atr     = max_sl_atr,
            # Hybrid momentum path: ATR-based TP/SL saat vol_spike tinggi
            momentum_vol_spike   = feat.get("vol_spike_zscore"),
            momentum_price_accel = feat.get("price_accel_1h"),
            momentum_threshold   = 1.5,
            momentum_tp_atr      = 2.0,
            momentum_sl_atr      = 1.5,
        )

        feat_df["label"] = structural_label_filter(
            labels                   = raw_labels,
            feat_df                  = feat_df,
            long_max_price_in_range  = long_max_price_in_range,
            short_min_price_in_range = short_min_price_in_range,
        )

        label_counts = feat_df["label"].value_counts().to_dict()
        total = len(feat_df)
        logger.info(
            f"[{symbol}] Label distribution v3: "
            f"LONG={label_counts.get('LONG', 0)} ({label_counts.get('LONG', 0)/total:.1%}), "
            f"SHORT={label_counts.get('SHORT', 0)} ({label_counts.get('SHORT', 0)/total:.1%}), "
            f"FLAT={label_counts.get('FLAT', 0)} ({label_counts.get('FLAT', 0)/total:.1%})"
        )

    nan_pct = feat_df.isnull().mean().mean()
    logger.info(
        f"[{symbol}] Features v3: {len(feat_df):,} rows × {len(feat_df.columns)} cols "
        f"| NaN: {nan_pct:.1%}"
    )

    # Simpan untuk evaluasi/backtest (TIDAK masuk FEATURE_COLS_V3)
    feat_df["h4_swing_high"] = h4_swing_highs
    feat_df["h4_swing_low"]  = h4_swing_lows

    return feat_df