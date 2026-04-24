"""
core/features.py — Feature Engineering & Labeling
Gabungan dari feature_engineer.py dan fix_synthetic_oi.py

Fungsi utama:
  engineer_features()       — hitung semua 58+ fitur dari cleaned parquet
  compute_synthetic_oi()    — hitung Synthetic OI dari CVD (fix_synthetic_oi)
  swing_based_labeling()    — Labeling v3 berbasis H4 Swing Points
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


# ─── ATR & RSI ───────────────────────────────────────────────────────────────

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

def calc_stochrsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
                  k_period: int = 3, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    rsi     = calc_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period, min_periods=1).min()
    rsi_max = rsi.rolling(stoch_period, min_periods=1).max()
    k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = k.rolling(k_period, min_periods=1).mean()
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d

def calc_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


# ─── CVD & Volume ────────────────────────────────────────────────────────────

def calc_cvd(df: pd.DataFrame) -> pd.Series:
    buy_col  = _col(df, "taker_buy_volume", "taker_ratio_takerBuyVol", "m15_taker_buy_base_asset_volume")
    sell_col = _col(df, "taker_sell_volume", "taker_ratio_takerSellVol", "m15_taker_sell_base_asset_volume")
    if buy_col and sell_col:
        delta = df[buy_col].fillna(0) - df[sell_col].fillna(0)
    else:
        close  = df.get("close", df.get("m15_close", pd.Series(np.nan, index=df.index)))
        volume = df.get("volume", df.get("m15_volume", pd.Series(np.nan, index=df.index)))
        sign   = np.sign(close.diff().fillna(0))
        delta  = sign * volume.fillna(0)
    return delta.cumsum()

def calc_volume_delta(df: pd.DataFrame) -> pd.Series:
    buy_col  = _col(df, "taker_buy_volume", "taker_ratio_takerBuyVol", "m15_taker_buy_base_asset_volume")
    sell_col = _col(df, "taker_sell_volume", "taker_ratio_takerSellVol", "m15_taker_sell_base_asset_volume")
    if buy_col and sell_col:
        return (df[buy_col] - df[sell_col]).fillna(0)
    close  = df.get("close", df.get("m15_close", pd.Series(np.nan, index=df.index)))
    volume = df.get("volume", df.get("m15_volume", pd.Series(np.nan, index=df.index)))
    sign   = np.sign(close.diff().fillna(0))
    return sign * volume.fillna(0)

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


# ─── Market Structure & Liquidity ────────────────────────────────────────────

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

def calc_market_session(index: pd.DatetimeIndex) -> pd.Series:
    hour    = index.hour
    session = np.zeros(len(index), dtype=np.int8)
    session[(hour >= 0)  & (hour < 8)]  = 1
    session[(hour >= 7)  & (hour < 15)] = 2
    session[(hour >= 13) & (hour < 21)] = 3
    return pd.Series(session, index=index, name="market_session")

def calc_cyclic_time(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    hour = index.hour + index.minute / 60
    dow  = index.dayofweek
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow  /  7),
        "dow_cos":  np.cos(2 * np.pi * dow  /  7),
    }

def calc_time_to_funding(index: pd.DatetimeIndex) -> pd.Series:
    minutes_in_day  = index.hour * 60 + index.minute
    next_settlement = np.ceil(minutes_in_day / 480) * 480
    mins_remaining  = (next_settlement - minutes_in_day) % 480
    return pd.Series(mins_remaining / 480.0, index=index, name="time_to_funding_norm")

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


# ─── H4 Alignment Tools ──────────────────────────────────────────────────────

def calc_rsi_h4(h4_close: pd.Series, close_m15: pd.Series, period: int = 14) -> pd.Series:
    """RSI dihitung dari H4 close, di-align ke index M15."""
    rsi_h4 = calc_rsi(h4_close, period)
    return rsi_h4.reindex(rsi_h4.index.union(close_m15.index)).ffill().reindex(close_m15.index)

def detect_h4_swing_points(
    h4_high:    pd.Series,
    h4_low:     pd.Series,
    lookback:   int = 3,
) -> tuple[pd.Series, pd.Series]:
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
    m15_index:      pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    sh_filled = h4_swing_highs.ffill()
    sl_filled = h4_swing_lows.ffill()
    sh_m15 = sh_filled.reindex(sh_filled.index.union(m15_index)).ffill().reindex(m15_index)
    sl_m15 = sl_filled.reindex(sl_filled.index.union(m15_index)).ffill().reindex(m15_index)
    return sh_m15, sl_m15


# ─── ★ BARU v3: Smart Money & Divergence Features ────────────────────────────

def calc_cvd_divergence(
    close:     pd.Series,
    cvd:       pd.Series,
    h4_close:  pd.Series,
    h4_cvd:    pd.Series,
    m15_index: pd.DatetimeIndex,
    window:    int = 5,
) -> tuple[pd.Series, pd.Series]:
    """CVD Divergence — proxy akumulasi/distribusi smart money."""
    price_chg = h4_close.diff(window)
    cvd_chg   = h4_cvd.diff(window)

    div_raw = np.where(
        (price_chg > 0) & (cvd_chg < 0), -1.0,   # distribusi
        np.where(
            (price_chg < 0) & (cvd_chg > 0), 1.0,  # akumulasi
            0.0
        )
    )
    cvd_div_h4_raw = pd.Series(div_raw, index=h4_close.index)
    cvd_slope_raw = h4_cvd.diff(window) / (h4_cvd.abs().rolling(window).mean() + 1e-10)

    cvd_div_h4  = cvd_div_h4_raw.reindex(cvd_div_h4_raw.index.union(m15_index)).ffill().reindex(m15_index)
    cvd_slope_h4 = cvd_slope_raw.reindex(cvd_slope_raw.index.union(m15_index)).ffill().reindex(m15_index)

    return cvd_div_h4, cvd_slope_h4


def calc_volume_absorption(
    high:    pd.Series,
    low:     pd.Series,
    volume:  pd.Series,
    atr:     pd.Series,
    window:  int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Volume Absorption — smart money menyerap order di level kunci."""
    candle_range  = (high - low).replace(0, np.nan)
    atr_safe      = atr.replace(0, np.nan)

    vol_efficiency = volume / (candle_range / atr_safe)
    vol_eff_mean   = vol_efficiency.rolling(window, min_periods=5).mean()
    vol_eff_std    = vol_efficiency.rolling(window, min_periods=5).std().replace(0, np.nan)
    absorption_z   = (vol_efficiency - vol_eff_mean) / vol_eff_std

    return vol_efficiency.fillna(0), absorption_z.fillna(0)


def calc_funding_price_divergence(
    close:        pd.Series,
    funding_rate: pd.Series,
    window:       int = 8,
) -> pd.Series:
    """Funding-Price Divergence — proxy distribusi/akumulasi via market positioning."""
    funding_ffill = funding_rate.ffill().fillna(0)
    price_ret     = close.pct_change(window).fillna(0)

    fr_mean = funding_ffill.rolling(window * 3, min_periods=window).mean()
    fr_std  = funding_ffill.rolling(window * 3, min_periods=window).std().replace(0, np.nan)
    fr_z    = (funding_ffill - fr_mean) / fr_std

    divergence = -np.sign(fr_z) * np.sign(price_ret).replace(0, np.nan).ffill().fillna(0)
    divergence = divergence * fr_z.abs()

    return divergence.fillna(0)


def calc_wyckoff_phase(
    price_in_range: pd.Series,
    vol_regime:     pd.Series,
    h4_trend:       pd.Series,
    cvd_slope_h4:   pd.Series,
    window:         int = 10,
) -> tuple[pd.Series, pd.Series]:
    """Wyckoff Phase Proxy — identifikasi fase siklus pasar."""
    phase = np.zeros(len(price_in_range), dtype=int)

    trend_up   = (h4_trend == 1).values
    trend_down = (h4_trend == -1).values
    price_high = (price_in_range > 0.65).values
    price_low  = (price_in_range < 0.35).values
    vol_high   = (vol_regime > 1.3).values
    cvd_pos    = (cvd_slope_h4 > 0).values
    cvd_neg    = (cvd_slope_h4 < 0).values

    phase[trend_up   & cvd_pos] = 0
    phase[price_high & vol_high & cvd_neg] = 1
    phase[trend_down & cvd_neg] = 2
    phase[price_low  & vol_high & cvd_pos] = 3

    phase_s = pd.Series(phase, index=price_in_range.index)

    price_in_range_shift = price_in_range.shift(1)
    spring    = (price_in_range < 0.05) & (price_in_range_shift > 0.10)
    upthrust  = (price_in_range > 0.95) & (price_in_range_shift < 0.90)
    spring_upthrust = (spring | upthrust).astype(int)

    return phase_s, spring_upthrust


def calc_rsi_divergence(
    close:   pd.Series,
    rsi_h4:  pd.Series,
    window:  int = 5,
) -> pd.Series:
    """RSI Divergence (Hidden & Regular) — konfirmasi arah trend."""
    price_chg = close.diff(window)
    rsi_chg   = rsi_h4.diff(window)

    div = np.where(
        (price_chg > 0) & (rsi_chg < 0), -1.0,
        np.where(
            (price_chg < 0) & (rsi_chg > 0),  1.0,
            np.where(
                (price_chg > 0) & (rsi_chg > 0), 0.5,
                np.where(
                    (price_chg < 0) & (rsi_chg < 0), -0.5,
                    0.0
                )
            )
        )
    )
    return pd.Series(div, index=close.index).fillna(0)


# ─── ★ BARU v3: Swing-Based Labeling ─────────────────────────────────────────

def swing_based_labeling(
    close:          pd.Series,
    high:           pd.Series,
    low:            pd.Series,
    atr_m15:        pd.Series,
    h4_swing_highs: pd.Series,
    h4_swing_lows:  pd.Series,
    max_hold:       int   = 192,
    min_rr:         float = 1.5,
    min_tp_atr:     float = 1.5,
    max_sl_atr:     float = 3.0,
) -> pd.Series:
    """Labeling berbasis swing high/low H4 — untuk swing trade sesungguhnya."""
    n      = len(close)
    labels = np.full(n, "FLAT", dtype=object)

    c_arr  = close.values
    h_arr  = high.values
    l_arr  = low.values
    a_arr  = atr_m15.values
    sh_arr = h4_swing_highs.values
    sl_arr = h4_swing_lows.values

    for i in range(n - 1):
        price = c_arr[i]
        atr_i = a_arr[i]

        if np.isnan(price) or np.isnan(atr_i) or atr_i == 0:
            continue

        swing_hi = sh_arr[i]
        swing_lo = sl_arr[i]

        if np.isnan(swing_hi) or np.isnan(swing_lo):
            continue

        # Setup LONG
        tp_long = swing_hi
        sl_long = swing_lo
        tp_dist_long = tp_long - price
        sl_dist_long = price   - sl_long

        long_valid = (
            tp_dist_long > 0 and
            sl_dist_long > 0 and
            tp_dist_long >= min_tp_atr * atr_i and
            sl_dist_long <= max_sl_atr * atr_i and
            (tp_dist_long / sl_dist_long >= min_rr)
        )

        # Setup SHORT
        tp_short = swing_lo
        sl_short = swing_hi
        tp_dist_short = price    - tp_short
        sl_dist_short = sl_short - price

        short_valid = (
            tp_dist_short > 0 and
            sl_dist_short > 0 and
            tp_dist_short >= min_tp_atr * atr_i and
            sl_dist_short <= max_sl_atr * atr_i and
            (tp_dist_short / sl_dist_short >= min_rr)
        )

        if not long_valid and not short_valid:
            continue

        # Scan kedepan
        end = min(i + max_hold, n)
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

            if (not long_valid  or outcome_long  != "FLAT") and \
               (not short_valid or outcome_short != "FLAT"):
                break

        # Assign label
        if long_valid and short_valid:
            rr_long  = tp_dist_long  / sl_dist_long  if sl_dist_long  > 0 else 0
            rr_short = tp_dist_short / sl_dist_short if sl_dist_short > 0 else 0

            if outcome_long == "LONG" and outcome_short != "SHORT":
                labels[i] = "LONG"
            elif outcome_short == "SHORT" and outcome_long != "LONG":
                labels[i] = "SHORT"
            elif outcome_long == "LONG" and outcome_short == "SHORT":
                labels[i] = "LONG" if rr_long >= rr_short else "SHORT"
        elif long_valid:
            labels[i] = "LONG" if outcome_long == "LONG" else "FLAT"
        elif short_valid:
            labels[i] = "SHORT" if outcome_short == "SHORT" else "FLAT"

    tail = min(max_hold // 4, n)
    labels[-tail:] = "FLAT"

    return pd.Series(labels, index=close.index, name="label")


def structural_label_filter(
    labels: pd.Series,
    feat_df: pd.DataFrame,
    long_max_price_in_range: float = 0.8,
    short_min_price_in_range: float = 0.2,
) -> pd.Series:
    """Filter label LONG jika harga sudah terlalu di pucuk, SHORT jika terlalu di bawah."""
    filtered = labels.copy()
    if "price_in_range" in feat_df.columns:
        pir = feat_df["price_in_range"]
        filtered[(labels == "LONG") & (pir > long_max_price_in_range)] = "FLAT"
        filtered[(labels == "SHORT") & (pir < short_min_price_in_range)] = "FLAT"
    return filtered


# ─── Main Feature Engineering Function ───────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    symbol: str,
    symbol_id: int,
    tp_mult: float = 2.0,       # Tetap ada untuk backward compatibility
    sl_mult: float = 1.0,       # Tetap ada untuk backward compatibility
    max_hold: int  = 192,       # Diupdate ke 192 (48 jam) untuk swing v3
    vp_window: int = 96,
    vp_bins: int   = 50,
    swing_lookback: int = 5,
    fvg_min_gap: float  = 0.5,
    add_label: bool     = True,
    # ★ Parameter baru v3:
    min_rr: float = 1.5,
    min_tp_atr: float = 1.5,
    max_sl_atr: float = 3.0,
    long_max_price_in_range: float = 0.8,
    short_min_price_in_range: float = 0.2,
) -> pd.DataFrame:
    """
    Hitung semua 58+ fitur dari cleaned DataFrame + H4 Swing Labeling.
    Input: cleaned parquet yang sudah memiliki kolom M15 OHLCV
    Output: DataFrame dengan fitur + label
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

    # ── ATR M15 & H4 ──────────────────────────────────────────────────────────
    atr14    = calc_atr(h, l, c, 14)
    atr_safe = atr14.replace(0, np.nan)

    h4_h = df.get("4h_high",  h)
    h4_l = df.get("4h_low",   l)
    h4_c = df.get("4h_close", c)
    atr_h4 = calc_atr(h4_h, h4_l, h4_c, 14)

    feat: dict[str, pd.Series] = {}

    feat["open"] = o; feat["high"] = h; feat["low"] = l
    feat["close"] = c; feat["volume"] = v

    # ── Volume Flow ───────────────────────────────────────────────────────────
    feat["volume_delta"] = calc_volume_delta(df)
    feat["cvd"]          = calc_cvd(df)

    # ── ★ BARU v3: H4 CVD (untuk smart money features) ───────────────────────
    cvd_m15_series = feat["cvd"] if isinstance(feat["cvd"], pd.Series) else pd.Series(feat["cvd"], index=df.index)
    h4_cvd_raw = cvd_m15_series.resample("4h").last()

    # ── ★ BARU v3: Swing Highs/Lows H4 (untuk labeling) ──────────────────────
    h4_sh_raw, h4_sl_raw = detect_h4_swing_points(h4_h, h4_l, lookback=3)

    h4_swing_highs_m15, h4_swing_lows_m15 = get_nearest_swing_levels(
        h4_swing_highs = h4_sh_raw,
        h4_swing_lows  = h4_sl_raw,
        m15_index      = df.index,
    )

    buy_col  = _col(df, "taker_buy_volume",  "taker_ratio_takerBuyVol", "m15_taker_buy_base_asset_volume")
    sell_col = _col(df, "taker_sell_volume", "taker_ratio_takerSellVol", "m15_taker_sell_base_asset_volume")
    feat["buy_volume"]  = df[buy_col]  if buy_col  else (v * 0.5)
    feat["sell_volume"] = df[sell_col] if sell_col else (v * 0.5)

    # ── Market Structure & Liquidity ──────────────────────────────────────────
    bos, choch, bars_since = calc_market_structure(h, l, c, swing_lookback)
    feat["MSB_BOS"]        = bos
    feat["CHoCH"]          = choch
    feat["bars_since_BOS"] = bars_since

    fvg_up, fvg_down = calc_fvg(h, l, atr14, fvg_min_gap)
    feat["FVG_up"]   = fvg_up
    feat["FVG_down"] = fvg_down

    buy_liq, sell_liq, sfp = calc_liquidity_levels(h, l, c, atr14, swing_lookback)
    feat["Buy_Liq"]   = buy_liq
    feat["Sell_Liq"]  = sell_liq
    feat["SFP_sweep"] = sfp

    # ── Synthetic OI & Funding ────────────────────────────────────────────────
    oi_col = _col(df, "open_interest")
    if oi_col and not df[oi_col].isna().all():
        feat["open_interest"] = df[oi_col]
    else:
        temp_df = pd.DataFrame({"cvd": feat["cvd"], "volume": v})
        feat["open_interest"] = compute_synthetic_oi(temp_df)

    fr_col = _col(df, "funding_rate_fundingRate", "funding_rate")
    feat["funding_rate"] = df[fr_col] if fr_col else pd.Series(np.nan, index=df.index)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for span in (7, 21, 50, 200):
        feat[f"ema_{span}_m15"] = (calc_ema(c, span) - c) / atr_safe

    for span in (7, 21, 50, 200):
        ema    = calc_ema(h4_c, span)
        ema_m15 = ema.reindex(ema.index.union(df.index)).ffill().reindex(df.index)
        feat[f"ema_{span}_h4"] = (ema_m15 - c) / atr_safe

    # ── RSI & StochRSI ────────────────────────────────────────────────────────
    feat["rsi_6"] = calc_rsi(c, 6)
    feat["stochrsi_k"], feat["stochrsi_d"] = calc_stochrsi(c)
    feat["atr_14_m15"] = atr14
    feat["atr_14_h4"]  = atr_h4.reindex(atr_h4.index.union(df.index)).ffill().reindex(df.index)

    # ── Utilities ─────────────────────────────────────────────────────────────
    feat.update(calc_prev_day_week_levels(h, l, c, atr14))
    feat.update(calc_fib_levels(h, l, c, atr14))

    poc, vah, val = calc_volume_profile(h, l, c, v, vp_window, vp_bins)
    feat["POC"] = (poc - c) / atr_safe
    feat["VAH"] = (vah - c) / atr_safe
    feat["VAL"] = (val - c) / atr_safe

    btc_col = _col(df, "macro_btc_dominance", "btc_dominance_pct", "btc_dominance")
    feat["btc_dominance"] = df[btc_col] if btc_col else pd.Series(np.nan, index=df.index)

    fg_col = _col(df, "macro_fear_greed_index_value", "fear_greed", "macro_fear_greed")
    feat["fear_greed"] = df[fg_col] if fg_col else pd.Series(np.nan, index=df.index)

    feat["market_session"] = calc_market_session(df.index)

    feat["log_ret_1"]  = np.log(c / c.shift(1))
    feat["log_ret_5"]  = np.log(c / c.shift(5))
    feat["log_ret_20"] = np.log(c / c.shift(20))

    vol_ma20 = v.rolling(20, min_periods=5).mean()
    feat["vol_ratio_20"] = v / vol_ma20.replace(0, np.nan)

    feat.update(calc_cyclic_time(df.index))
    feat["time_to_funding_norm"] = calc_time_to_funding(df.index)

    # ── Partial NaN features ──────────────────────────────────────────────────
    ls_col     = _col(df, "long_short_ratio")
    la_col     = _col(df, "long_account_pct", "longAccount")
    sa_col     = _col(df, "short_account_pct", "shortAccount")
    tr_col     = _col(df, "taker_buy_sell_ratio", "taker_ratio")
    feat["long_short_ratio"]   = df[ls_col] if ls_col else pd.Series(np.nan, index=df.index)
    feat["long_account_pct"]   = df[la_col] if la_col else pd.Series(np.nan, index=df.index)
    feat["short_account_pct"]  = df[sa_col] if sa_col else pd.Series(np.nan, index=df.index)
    feat["taker_buy_sell_ratio"] = df[tr_col] if tr_col else pd.Series(np.nan, index=df.index)


    # ── ★ BARU v3: Eksekusi Smart Money Features (Inline Replacement) ─────────
    cvd_div, cvd_slope = calc_cvd_divergence(
        close     = c,
        cvd       = feat["cvd"],
        h4_close  = h4_c,
        h4_cvd    = h4_cvd_raw,
        m15_index = df.index,
        window    = 5,
    )
    feat["cvd_div_h4"]   = cvd_div
    feat["cvd_slope_h4"] = cvd_slope

    vol_eff, absorption_z = calc_volume_absorption(h, l, v, atr14)
    feat["vol_efficiency"] = vol_eff
    feat["absorption_z"]   = absorption_z

    feat["funding_price_div"] = calc_funding_price_divergence(c, feat["funding_rate"])

    rsi_h4_series = calc_rsi_h4(h4_c, c, period=14)
    feat["rsi_h4"] = rsi_h4_series
    feat["rsi_divergence"] = calc_rsi_divergence(c, rsi_h4_series, window=5)

    # Kalkulasi fallback untuk Wyckoff jika belum ada di data sebelumnya
    roll_min = l.rolling(96, min_periods=10).min()
    roll_max = h.rolling(96, min_periods=10).max()
    feat["price_in_range"] = (c - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    
    vol_ma_96 = v.rolling(96, min_periods=10).mean()
    feat["vol_regime"] = v / vol_ma_96.replace(0, np.nan)
    
    # Tren proxy menggunakan EMA
    ema_7_h4_raw  = calc_ema(h4_c, 7).reindex(df.index).ffill()
    ema_21_h4_raw = calc_ema(h4_c, 21).reindex(df.index).ffill()
    feat["h4_trend"] = np.sign(ema_7_h4_raw - ema_21_h4_raw)

    phase, spring_ut = calc_wyckoff_phase(
        price_in_range = feat["price_in_range"].fillna(0.5),
        vol_regime     = feat["vol_regime"].fillna(1.0),
        h4_trend       = feat["h4_trend"].fillna(0),
        cvd_slope_h4   = feat["cvd_slope_h4"].fillna(0),
    )
    feat["wyckoff_phase"]    = phase
    feat["spring_upthrust"]  = spring_ut

    # ── Symbol encoding ───────────────────────────────────────────────────────
    feat["symbol"] = symbol_id

    # ── Build DataFrame ───────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feat, index=df.index)
    feat_df = ensure_utc_index(feat_df)

    # ── ★ BARU v3: Swing-Based Labeling ──────────────────────────────────────
    if add_label:
        logger.info(
            f"[{symbol}] Swing-Based labeling v3 "
            f"(max_hold={max_hold} bar M15, min_rr={min_rr}, "
            f"min_tp={min_tp_atr}×ATR, max_sl={max_sl_atr}×ATR)..."
        )
        raw_labels = swing_based_labeling(
            close          = c,
            high           = h,
            low            = l,
            atr_m15        = atr14,
            h4_swing_highs = h4_swing_highs_m15,
            h4_swing_lows  = h4_swing_lows_m15,
            max_hold       = max_hold,
            min_rr         = min_rr,
            min_tp_atr     = min_tp_atr,
            max_sl_atr     = max_sl_atr,
        )

        feat_df["label"] = structural_label_filter(
            labels                   = raw_labels,
            feat_df                  = feat_df,
            long_max_price_in_range  = long_max_price_in_range,
            short_min_price_in_range = short_min_price_in_range,
        )

        label_counts = feat_df["label"].value_counts().to_dict()
        logger.info(f"[{symbol}] Label distribution v3: {label_counts}")

    nan_pct = feat_df.isnull().mean().mean()
    logger.info(
        f"[{symbol}] Features: {len(feat_df):,} rows × {len(feat_df.columns)} cols "
        f"| NaN: {nan_pct:.1%}"
    )
    
    return feat_df