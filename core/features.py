"""
core/features.py — Feature Engineering & Labeling  (v2)
Gabungan dari feature_engineer.py dan fix_synthetic_oi.py

Fungsi utama:
  engineer_features()        — hitung semua 65 fitur dari cleaned parquet
  add_swing_features()       — ★ BARU: 7 fitur swing + regime
  structural_label_filter()  — ★ BARU: override LONG/SHORT → FLAT berdasarkan konteks H4
  compute_synthetic_oi()     — hitung Synthetic OI dari CVD
  triple_barrier_labeling()  — Triple Barrier label: LONG/SHORT/FLAT

Perubahan v2 vs v1:
  + 4 swing structure features  : dist_swing_high, dist_swing_low, price_in_range, swing_momentum
  + 3 market regime features    : h4_trend, trend_strength, vol_regime
  + structural_label_filter()   : LONG di downtrend H4 → FLAT, SHORT di uptrend H4 → FLAT
  Total fitur: 58 → 65
"""

import numpy as np
import pandas as pd

from core.utils import setup_logger, save_df, ensure_utc_index

logger = setup_logger("features")


# ─── Column helpers ───────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _get_ohlcv(df: pd.DataFrame, prefix: str = "m15"):
    def col(name):
        full = f"{prefix}_{name}"
        return df[full] if full in df.columns else pd.Series(np.nan, index=df.index)
    return col("open"), col("high"), col("low"), col("close"), col("volume")


# ─── ATR ─────────────────────────────────────────────────────────────────────

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean()


# ─── RSI ─────────────────────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 6) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── StochRSI ────────────────────────────────────────────────────────────────

def calc_stochrsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
                  k_period: int = 3, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    rsi     = calc_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period, min_periods=1).min()
    rsi_max = rsi.rolling(stoch_period, min_periods=1).max()
    k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = k.rolling(k_period, min_periods=1).mean()
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d


# ─── EMA ─────────────────────────────────────────────────────────────────────

def calc_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


# ─── CVD ─────────────────────────────────────────────────────────────────────

def calc_cvd(df: pd.DataFrame) -> pd.Series:
    buy_col  = _col(df, "taker_buy_volume", "taker_ratio_takerBuyVol",
                    "m15_taker_buy_base_asset_volume")
    sell_col = _col(df, "taker_sell_volume", "taker_ratio_takerSellVol",
                    "m15_taker_sell_base_asset_volume")
    if buy_col and sell_col:
        delta = df[buy_col].fillna(0) - df[sell_col].fillna(0)
    else:
        close  = df.get("close", df.get("m15_close", pd.Series(np.nan, index=df.index)))
        volume = df.get("volume", df.get("m15_volume", pd.Series(np.nan, index=df.index)))
        sign   = np.sign(close.diff().fillna(0))
        delta  = sign * volume.fillna(0)
    return delta.cumsum()


# ─── Volume Delta ─────────────────────────────────────────────────────────────

def calc_volume_delta(df: pd.DataFrame) -> pd.Series:
    buy_col  = _col(df, "taker_buy_volume", "taker_ratio_takerBuyVol",
                    "m15_taker_buy_base_asset_volume")
    sell_col = _col(df, "taker_sell_volume", "taker_ratio_takerSellVol",
                    "m15_taker_sell_base_asset_volume")
    if buy_col and sell_col:
        return (df[buy_col] - df[sell_col]).fillna(0)
    close  = df.get("close", df.get("m15_close", pd.Series(np.nan, index=df.index)))
    volume = df.get("volume", df.get("m15_volume", pd.Series(np.nan, index=df.index)))
    sign   = np.sign(close.diff().fillna(0))
    return sign * volume.fillna(0)


# ─── Volume Profile ───────────────────────────────────────────────────────────

def calc_volume_profile(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    window: int = 96, bins: int = 50,
) -> tuple[pd.Series, pd.Series, pd.Series]:
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
            poc_list.append(np.nan); vah_list.append(np.nan); val_list.append(np.nan)
            continue

        price_min = np.nanmin(lo_w)
        price_max = np.nanmax(hi_w)
        if price_max == price_min:
            poc_list.append(price_max); vah_list.append(price_max); val_list.append(price_min)
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
            add_lo = bin_vol[lo_ptr - 1] if lo_ptr > 0 else 0
            add_hi = bin_vol[hi_ptr + 1] if hi_ptr < bins - 1 else 0
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


# ─── Swing High / Low ────────────────────────────────────────────────────────

def detect_swing_highs_lows(
    high: pd.Series, low: pd.Series, lookback: int = 5,
) -> tuple[pd.Series, pd.Series]:
    n  = len(high)
    sh = pd.Series(False, index=high.index)
    sl = pd.Series(False, index=low.index)
    for i in range(lookback, n - lookback):
        if high.iloc[i] == high.iloc[i - lookback: i + lookback + 1].max():
            sh.iloc[i] = True
        if low.iloc[i] == low.iloc[i - lookback: i + lookback + 1].min():
            sl.iloc[i] = True
    return sh, sl


# ─── Liquidity & SFP ─────────────────────────────────────────────────────────

def calc_liquidity_levels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    atr: pd.Series, lookback: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sh, sl   = detect_swing_highs_lows(high, low, lookback)
    swing_hi = high.where(sh).ffill()
    swing_lo = low.where(sl).ffill()
    sell_liq = (swing_hi - close) / atr.replace(0, np.nan)
    buy_liq  = (close - swing_lo) / atr.replace(0, np.nan)
    sfp_sweep = ((low < swing_lo) & (close > swing_lo) |
                 (high > swing_hi) & (close < swing_hi)).astype(int)
    return buy_liq, sell_liq, sfp_sweep


# ─── Market Structure ────────────────────────────────────────────────────────

def calc_market_structure(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sh, sl   = detect_swing_highs_lows(high, low, lookback)
    prev_sh  = high.where(sh).ffill().shift(1)
    prev_sl  = low.where(sl).ffill().shift(1)
    bos_bull = (close > prev_sh).astype(int)
    bos_bear = (close < prev_sl).astype(int) * -1
    bos      = bos_bull + bos_bear
    last_bos = bos.replace(0, np.nan).ffill()
    choch    = ((bos != 0) & (last_bos != last_bos.shift(1))).astype(int)
    cum         = (bos != 0).cumsum()
    bars_since  = cum.groupby(cum).cumcount()
    bars_since  = bars_since.where(cum > 0, other=999)
    return bos, choch, bars_since


# ─── FVG ─────────────────────────────────────────────────────────────────────

def calc_fvg(
    high: pd.Series, low: pd.Series, atr: pd.Series, min_gap_atr: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    n        = len(high)
    fvg_up   = pd.Series(0.0, index=high.index)
    fvg_down = pd.Series(0.0, index=low.index)
    for i in range(1, n - 1):
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue
        gap_up   = low.iloc[i + 1] - high.iloc[i - 1]
        gap_down = low.iloc[i - 1] - high.iloc[i + 1]
        if gap_up   > min_gap_atr * atr_val:
            fvg_up.iloc[i]   = gap_up / atr_val
        if gap_down > min_gap_atr * atr_val:
            fvg_down.iloc[i] = gap_down / atr_val
    return fvg_up, fvg_down


# ─── PDH / PDL / PWH / PWL ───────────────────────────────────────────────────

def calc_prev_day_week_levels(
    high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series,
) -> dict[str, pd.Series]:
    df_tmp      = pd.DataFrame({"high": high, "low": low})
    daily_high  = df_tmp["high"].resample("1D").max()
    daily_low   = df_tmp["low"].resample("1D").min()
    weekly_high = df_tmp["high"].resample("1W").max()
    weekly_low  = df_tmp["low"].resample("1W").min()

    def shift_ffill(s):
        shifted = s.shift(1)
        return shifted.reindex(shifted.index.union(high.index)).ffill().reindex(high.index)

    atr_safe = atr.replace(0, np.nan)
    return {
        "PDH": (shift_ffill(daily_high)  - close) / atr_safe,
        "PDL": (shift_ffill(daily_low)   - close) / atr_safe,
        "PWH": (shift_ffill(weekly_high) - close) / atr_safe,
        "PWL": (shift_ffill(weekly_low)  - close) / atr_safe,
    }


# ─── Fibonacci ───────────────────────────────────────────────────────────────

def calc_fib_levels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    atr: pd.Series, window: int = 96,
) -> dict[str, pd.Series]:
    roll_high = high.rolling(window, min_periods=10).max()
    roll_low  = low.rolling(window, min_periods=10).min()
    rng       = roll_high - roll_low
    atr_safe  = atr.replace(0, np.nan)
    return {
        "Fib_618": (roll_high - 0.618 * rng - close) / atr_safe,
        "Fib_786": (roll_high - 0.786 * rng - close) / atr_safe,
    }


# ─── Market Session ──────────────────────────────────────────────────────────

def calc_market_session(index: pd.DatetimeIndex) -> pd.Series:
    hour    = index.hour
    session = np.zeros(len(index), dtype=np.int8)
    session[(hour >= 0)  & (hour < 8)]  = 1
    session[(hour >= 7)  & (hour < 15)] = 2
    session[(hour >= 13) & (hour < 21)] = 3
    return pd.Series(session, index=index, name="market_session")


# ─── Cyclic Time ─────────────────────────────────────────────────────────────

def calc_cyclic_time(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    hour = index.hour + index.minute / 60
    dow  = index.dayofweek
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow  /  7),
        "dow_cos":  np.cos(2 * np.pi * dow  /  7),
    }


# ─── Funding Countdown ────────────────────────────────────────────────────────

def calc_time_to_funding(index: pd.DatetimeIndex) -> pd.Series:
    minutes_in_day  = index.hour * 60 + index.minute
    next_settlement = np.ceil(minutes_in_day / 480) * 480
    mins_remaining  = (next_settlement - minutes_in_day) % 480
    return pd.Series(mins_remaining / 480.0, index=index, name="time_to_funding_norm")


# ─── Synthetic OI ────────────────────────────────────────────────────────────

def compute_synthetic_oi(
    df: pd.DataFrame,
    cvd_window: int = 96,
    norm_window: int = 672,
) -> pd.Series:
    cvd_col = _col(df, "cvd")
    vol_col = _col(df, "volume")
    if cvd_col is None or vol_col is None:
        raise KeyError("Kolom 'cvd' atau 'volume' tidak ditemukan.")
    cvd    = df[cvd_col].astype(float)
    volume = df[vol_col].astype(float)
    cvd_ma       = cvd.rolling(cvd_window,  min_periods=1).mean()
    vol_ma       = volume.rolling(cvd_window, min_periods=1).mean()
    raw          = cvd_ma + vol_ma * 2
    norm_denom   = raw.rolling(norm_window, min_periods=1).mean().replace(0, np.nan)
    synthetic_oi = (raw / norm_denom).ffill().fillna(1.0)
    return synthetic_oi


# ─── ★ BARU: Swing Structure & Regime Features ───────────────────────────────

def add_swing_features(
    feat: dict,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    atr_m15: pd.Series,
    ema_7_h4: pd.Series,
    ema_21_h4: pd.Series,
    ema_50_h4: pd.Series,
    atr_h4: pd.Series,
    rolling_bars: int = 96,
) -> None:
    """
    Tambahkan 7 fitur baru ke dict `feat` (in-place):

    Swing structure (4):
      dist_swing_high  — jarak ke rolling high 24h, ATR-normalized (negatif = di bawah high)
      dist_swing_low   — jarak ke rolling low 24h, ATR-normalized (positif = di atas low)
      price_in_range   — posisi harga dalam 24h range [0=bottom, 1=top]
      swing_momentum   — perubahan price_in_range selama 4 bar terakhir

    Market regime (3):
      h4_trend         — arah trend H4: +1 (up), -1 (down), 0 (sideways)
      trend_strength   — (ema_7_h4 - ema_50_h4) / atr_h4, ATR-normalized
      vol_regime       — volume relatif terhadap rata-rata 96 bar
    """
    atr_safe   = atr_m15.replace(0, np.nan)
    atr_h4_safe = atr_h4.replace(0, np.nan)

    # ── Swing structure ───────────────────────────────────────────────────────
    roll_high = high.rolling(rolling_bars, min_periods=10).max()
    roll_low  = low.rolling(rolling_bars,  min_periods=10).min()
    rng       = (roll_high - roll_low).replace(0, np.nan)

    feat["dist_swing_high"] = (close - roll_high) / atr_safe        # ≤ 0 jika di bawah high
    feat["dist_swing_low"]  = (close - roll_low)  / atr_safe        # ≥ 0 jika di atas low
    feat["price_in_range"]  = (close - roll_low)  / rng             # [0, 1]
    feat["swing_momentum"]  = feat["price_in_range"] - feat["price_in_range"].shift(4)

    # ── Market regime ─────────────────────────────────────────────────────────
    feat["h4_trend"] = np.where(
        ema_7_h4 > ema_21_h4, 1,
        np.where(ema_7_h4 < ema_21_h4, -1, 0)
    )
    feat["h4_trend"] = pd.Series(feat["h4_trend"], index=close.index)

    feat["trend_strength"] = (ema_7_h4 - ema_50_h4) / atr_h4_safe

    vol_ma = volume.rolling(rolling_bars, min_periods=10).mean().replace(0, np.nan)
    feat["vol_regime"] = volume / vol_ma


# ─── ★ BARU: Structural Label Filter ─────────────────────────────────────────

def structural_label_filter(
    labels: pd.Series,
    feat_df: pd.DataFrame,
    long_max_price_in_range:  float = 0.4,
    short_min_price_in_range: float = 0.6,
) -> pd.Series:
    """
    Override label LONG/SHORT → FLAT jika konteks struktural H4 tidak mendukung.

    Aturan:
      LONG  di H4 downtrend DAN price_in_range > long_max_price_in_range  → FLAT
        (tidak dekat swing low, bukan genuine bottom)
      SHORT di H4 uptrend   DAN price_in_range < short_min_price_in_range → FLAT
        (tidak dekat swing high, bukan genuine top)

    Tidak menyentuh label FLAT yang sudah ada.
    """
    labels = labels.copy()

    h4_trend       = feat_df["h4_trend"]
    price_in_range = feat_df["price_in_range"]

    # LONG di downtrend H4 + price tidak di zona bottom
    mask_long_override = (
        (labels == "LONG") &
        (h4_trend == -1) &
        (price_in_range > long_max_price_in_range)
    )
    # SHORT di uptrend H4 + price tidak di zona top
    mask_short_override = (
        (labels == "SHORT") &
        (h4_trend == 1) &
        (price_in_range < short_min_price_in_range)
    )

    n_long_ovr  = mask_long_override.sum()
    n_short_ovr = mask_short_override.sum()

    labels[mask_long_override]  = "FLAT"
    labels[mask_short_override] = "FLAT"

    logger.info(
        f"Structural filter: {n_long_ovr} LONG → FLAT, {n_short_ovr} SHORT → FLAT"
    )
    return labels


# ─── Triple Barrier Labeling ─────────────────────────────────────────────────

def triple_barrier_labeling(
    close: pd.Series,
    atr: pd.Series,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int  = 48,
) -> pd.Series:
    n      = len(close)
    labels = np.full(n, "FLAT", dtype=object)
    c_arr  = close.values
    a_arr  = atr.values

    for i in range(n):
        atr_i = a_arr[i]
        if np.isnan(atr_i) or atr_i == 0 or np.isnan(c_arr[i]):
            continue
        upper = c_arr[i] + tp_mult * atr_i
        lower = c_arr[i] - sl_mult * atr_i
        end   = min(i + max_hold, n)
        for j in range(i + 1, end):
            if np.isnan(c_arr[j]):
                continue
            if c_arr[j] >= upper:
                labels[i] = "LONG"; break
            if c_arr[j] <= lower:
                labels[i] = "SHORT"; break
        if n - i < max_hold // 4:
            labels[i] = "FLAT"

    return pd.Series(labels, index=close.index, name="label")


# ─── Main Feature Engineering Function ───────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    symbol: str,
    symbol_id: int,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int  = 48,
    vp_window: int = 96,
    vp_bins: int   = 50,
    swing_lookback: int = 5,
    fvg_min_gap: float  = 0.5,
    swing_rolling_bars: int = 96,
    long_max_price_in_range: float  = 0.4,
    short_min_price_in_range: float = 0.6,
    add_label: bool = True,
) -> pd.DataFrame:
    """
    Hitung semua 65 fitur dari cleaned DataFrame.
    Input:  cleaned parquet yang sudah memiliki kolom M15 + H4 OHLCV
    Output: DataFrame dengan 65 fitur + label v2 (jika add_label=True)

    v2 vs v1:
      + 7 fitur baru (swing structure + regime)
      + structural_label_filter() setelah triple_barrier_labeling()
    """
    df = ensure_utc_index(df)

    # ── Extract base OHLCV (M15) ──────────────────────────────────────────────
    o, h, l, c, v = _get_ohlcv(df, "m15")
    if c.isna().all():
        o = df.get("open",   pd.Series(np.nan, index=df.index))
        h = df.get("high",   pd.Series(np.nan, index=df.index))
        l = df.get("low",    pd.Series(np.nan, index=df.index))
        c = df.get("close",  pd.Series(np.nan, index=df.index))
        v = df.get("volume", pd.Series(np.nan, index=df.index))

    # ── ATR M15 ───────────────────────────────────────────────────────────────
    atr14    = calc_atr(h, l, c, 14)
    atr_safe = atr14.replace(0, np.nan)

    # ── H4 OHLCV & ATR ────────────────────────────────────────────────────────
    h4_h = df.get("4h_high",  h)
    h4_l = df.get("4h_low",   l)
    h4_c = df.get("4h_close", c)
    atr_h4_raw = calc_atr(h4_h, h4_l, h4_c, 14)
    atr_h4     = atr_h4_raw.reindex(atr_h4_raw.index.union(df.index)).ffill().reindex(df.index)

    feat: dict[str, pd.Series] = {}

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    feat["open"] = o; feat["high"] = h; feat["low"] = l
    feat["close"] = c; feat["volume"] = v

    # ── Volume Flow ───────────────────────────────────────────────────────────
    feat["volume_delta"] = calc_volume_delta(df)
    feat["cvd"]          = calc_cvd(df)

    buy_col  = _col(df, "taker_buy_volume",  "taker_ratio_takerBuyVol",
                    "m15_taker_buy_base_asset_volume")
    sell_col = _col(df, "taker_sell_volume", "taker_ratio_takerSellVol",
                    "m15_taker_sell_base_asset_volume")
    feat["buy_volume"]  = df[buy_col]  if buy_col  else (v * 0.5)
    feat["sell_volume"] = df[sell_col] if sell_col else (v * 0.5)

    # ── Market Structure ──────────────────────────────────────────────────────
    bos, choch, bars_since = calc_market_structure(h, l, c, swing_lookback)
    feat["MSB_BOS"]        = bos
    feat["CHoCH"]          = choch
    feat["bars_since_BOS"] = bars_since

    # ── FVG ───────────────────────────────────────────────────────────────────
    fvg_up, fvg_down = calc_fvg(h, l, atr14, fvg_min_gap)
    feat["FVG_up"]   = fvg_up
    feat["FVG_down"] = fvg_down

    # ── Liquidity & SFP ───────────────────────────────────────────────────────
    buy_liq, sell_liq, sfp = calc_liquidity_levels(h, l, c, atr14, swing_lookback)
    feat["Buy_Liq"]   = buy_liq
    feat["Sell_Liq"]  = sell_liq
    feat["SFP_sweep"] = sfp

    # ── Open Interest ─────────────────────────────────────────────────────────
    oi_col = _col(df, "open_interest")
    if oi_col and not df[oi_col].isna().all():
        feat["open_interest"] = df[oi_col]
    else:
        temp_df = pd.DataFrame({"cvd": feat["cvd"], "volume": v})
        feat["open_interest"] = compute_synthetic_oi(temp_df)

    # ── Funding Rate ──────────────────────────────────────────────────────────
    fr_col = _col(df, "funding_rate_fundingRate", "funding_rate")
    feat["funding_rate"] = df[fr_col] if fr_col else pd.Series(np.nan, index=df.index)

    # ── EMA M15 (ATR-normalized) ──────────────────────────────────────────────
    for span in (7, 21, 50, 200):
        feat[f"ema_{span}_m15"] = (calc_ema(c, span) - c) / atr_safe

    # ── EMA H4 (ATR-normalized) — simpan raw EMA untuk add_swing_features ─────
    ema_h4 = {}
    for span in (7, 21, 50, 200):
        raw     = calc_ema(h4_c, span)
        aligned = raw.reindex(raw.index.union(df.index)).ffill().reindex(df.index)
        ema_h4[span] = aligned
        feat[f"ema_{span}_h4"] = (aligned - c) / atr_safe

    # ── RSI & StochRSI ────────────────────────────────────────────────────────
    feat["rsi_6"] = calc_rsi(c, 6)
    feat["stochrsi_k"], feat["stochrsi_d"] = calc_stochrsi(c)

    # ── ATR ───────────────────────────────────────────────────────────────────
    feat["atr_14_m15"] = atr14
    feat["atr_14_h4"]  = atr_h4

    # ── Key Levels ────────────────────────────────────────────────────────────
    feat.update(calc_prev_day_week_levels(h, l, c, atr14))
    feat.update(calc_fib_levels(h, l, c, atr14))

    # ── Volume Profile ────────────────────────────────────────────────────────
    poc, vah, val = calc_volume_profile(h, l, c, v, vp_window, vp_bins)
    feat["POC"] = (poc - c) / atr_safe
    feat["VAH"] = (vah - c) / atr_safe
    feat["VAL"] = (val - c) / atr_safe

    # ── Macro ─────────────────────────────────────────────────────────────────
    btc_col = _col(df, "macro_btc_dominance", "btc_dominance_pct", "btc_dominance")
    feat["btc_dominance"] = df[btc_col] if btc_col else pd.Series(np.nan, index=df.index)

    fg_col = _col(df, "macro_fear_greed_index_value", "fear_greed", "macro_fear_greed")
    feat["fear_greed"] = df[fg_col] if fg_col else pd.Series(np.nan, index=df.index)

    feat["market_session"] = calc_market_session(df.index)

    # ── Derived ───────────────────────────────────────────────────────────────
    feat["log_ret_1"]  = np.log(c / c.shift(1))
    feat["log_ret_5"]  = np.log(c / c.shift(5))
    feat["log_ret_20"] = np.log(c / c.shift(20))

    vol_ma20 = v.rolling(20, min_periods=5).mean()
    feat["vol_ratio_20"] = v / vol_ma20.replace(0, np.nan)

    feat.update(calc_cyclic_time(df.index))
    feat["time_to_funding_norm"] = calc_time_to_funding(df.index)

    # ── Partial NaN features ──────────────────────────────────────────────────
    ls_col = _col(df, "long_short_ratio")
    la_col = _col(df, "long_account_pct", "longAccount")
    sa_col = _col(df, "short_account_pct", "shortAccount")
    tr_col = _col(df, "taker_buy_sell_ratio", "taker_ratio")
    feat["long_short_ratio"]     = df[ls_col] if ls_col else pd.Series(np.nan, index=df.index)
    feat["long_account_pct"]     = df[la_col] if la_col else pd.Series(np.nan, index=df.index)
    feat["short_account_pct"]    = df[sa_col] if sa_col else pd.Series(np.nan, index=df.index)
    feat["taker_buy_sell_ratio"] = df[tr_col] if tr_col else pd.Series(np.nan, index=df.index)

    # ── Symbol encoding ───────────────────────────────────────────────────────
    feat["symbol"] = symbol_id

    # ── ★ BARU: Swing Structure & Regime Features (65 fitur total) ───────────
    add_swing_features(
        feat       = feat,
        high       = h,
        low        = l,
        close      = c,
        volume     = v,
        atr_m15    = atr14,
        ema_7_h4   = ema_h4[7],
        ema_21_h4  = ema_h4[21],
        ema_50_h4  = ema_h4[50],
        atr_h4     = atr_h4,
        rolling_bars = swing_rolling_bars,
    )

    # ── Build DataFrame ───────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feat, index=df.index)
    feat_df = ensure_utc_index(feat_df)

    # ── Triple Barrier Labeling + Structural Filter ───────────────────────────
    if add_label:
        logger.info(
            f"[{symbol}] Triple Barrier labeling v2 "
            f"(TP={tp_mult}×ATR, SL={sl_mult}×ATR, max={max_hold} bars)..."
        )
        raw_labels = triple_barrier_labeling(c, atr14, tp_mult, sl_mult, max_hold)

        # ★ Structural context filter
        feat_df["label"] = structural_label_filter(
            labels                    = raw_labels,
            feat_df                   = feat_df,
            long_max_price_in_range   = long_max_price_in_range,
            short_min_price_in_range  = short_min_price_in_range,
        )

        label_counts = feat_df["label"].value_counts().to_dict()
        logger.info(f"[{symbol}] Label distribution v2: {label_counts}")

    nan_pct = feat_df.isnull().mean().mean()
    logger.info(
        f"[{symbol}] Features v2: {len(feat_df):,} rows × {len(feat_df.columns)} cols "
        f"| NaN: {nan_pct:.1%}"
    )
    return feat_df