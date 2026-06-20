# -*- coding: utf-8 -*-
"""Generate per-feature parity table for ic32 33 features."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feats = json.load(open(ROOT / "models/feature_cols_v2.json", encoding="utf-8"))
ic_res = {r["feature"]: r for r in json.load(open(ROOT / "reports/experiments/ic32_hybrid_v1_ic.json", encoding="utf-8"))["results"]}

# Audit deltas (overlap Jun 8-13, live DB vs holdout) — median abs delta where available
audit = json.load(open(ROOT / "reports/experiments/vps_feature_audit.json", encoding="utf-8"))
median_delta = audit.get("compare_summary", {}).get("median_abs_delta", {})
hmm_match = audit.get("compare_summary", {}).get("hmm_match_rate", 0)

# Manual classification based on audit findings
# Categories: MATCH | EXPECTED_DIFF | ERROR_FIXED | INVESTIGATE
CLASS = {
    "long_short_ratio": ("ERROR_FIXED", "LSR=0 di DB Jun17+; holdout ~1.0 synthetic. Fixed training_parity."),
    "hmm_regime_enc": ("ERROR_FIXED", "95% stuck 0 Jun8-13; match 23%. Threshold routing rusak. Post-fix 49%."),
    "whale_retail_divergence": ("ERROR_FIXED", "Derived dari LSR — ikut rusak saat LSR=0."),
    "ofi_h4_delta": ("EXPECTED_DIFF", "Delta skala besar; holdout synthetic path vs live klines/positioning."),
    "cvd": ("EXPECTED_DIFF", "Magnitudo beda; WEAK IC, retained for LGBM interaction."),
    "cvd_slope_h4": ("INVESTIGATE", "Median delta 0.07 relatif kecil tapi top mismatch PEPE skala besar."),
    "ofi_acceleration": ("EXPECTED_DIFF", "Derived dari OFI; skala beda."),
    "cvd_div_h4": ("EXPECTED_DIFF", "Derived CVD; WEAK IC."),
    "rsi_h4": ("INVESTIGATE", "Median delta ~10 poin; timing H4 shift."),
    "stochrsi_d": ("INVESTIGATE", "Median delta ~33 poin; timing/rolling."),
    "h4_trend": ("INVESTIGATE", "Median delta 0 tapi mean 0.54; sering beda di overlap."),
    "log_ret_20": ("MATCH", "Median delta ~0.016; relatif stabil."),
    "relative_strength_z": ("MATCH", "Tidak di audit top-delta; OHLCV-derived."),
    "dist_from_8h_high": ("MATCH", "IC KEEP kuat; OHLCV liquidity."),
    "rsi_6": ("MATCH", "OHLCV oscillator."),
    "swing_momentum": ("MATCH", "IC KEEP; struktur swing."),
    "stochrsi_k": ("MATCH", "IC KEEP."),
    "dist_liq_50x_long": ("MATCH", "IC KEEP kuat."),
    "trend_accel_4h": ("MATCH", "IC KEEP."),
    "rsi_slope_h4": ("MATCH", "IC STRONG temporal."),
    "Fib_786": ("MATCH", "IC KEEP."),
    "Fib_618": ("MATCH", "REDUNDANT IC tapi stabil OHLCV."),
    "dist_liq_50x_short": ("MATCH", "IC KEEP."),
    "Buy_Liq": ("MATCH", "IC KEEP."),
    "dist_liq_20x_long": ("MATCH", "IC STRONG temporal."),
    "cvd_momentum_adv": ("INVESTIGATE", "Order flow; bisa beda skala live."),
    "Sell_Liq": ("MATCH", "IC KEEP."),
    "ema_21_slope_h4": ("MATCH", "IC KEEP."),
    "ema_50_h1": ("MATCH", "IC KEEP."),
    "dist_liq_20x_short": ("MATCH", "WEAK IC; liquidity distance."),
    "vol_price_confirm": ("MATCH", "WEAK IC."),
    "ema_50_slope_h4": ("MATCH", "WEAK IC."),
    "MSB_BOS": ("MATCH", "WEAK IC; struktur."),
}

# hmm IC from regime addition - approximate from docs
HMM_IC = {"standalone_ic": "N/A", "marginal_ic": "N/A", "verdict": "REGIME_ADD"}

print("| # | Fitur | IC Verdict | |IC| marg | Parity | Status | Catatan |")
print("|---|-------|------------|---------|--------|--------|---------|")
for i, f in enumerate(feats, 1):
    ic = ic_res.get(f, HMM_IC if f == "hmm_regime_enc" else {})
    verdict = ic.get("verdict", "?")
    marg = ic.get("marginal_ic", ic.get("standalone_ic", ""))
    if isinstance(marg, float):
        marg_s = f"{abs(marg):.4f}"
    else:
        marg_s = str(marg)
    cat, note = CLASS.get(f, ("MATCH", ""))
    md = median_delta.get(f)
    delta_s = f"{md:.3g}" if md is not None else "—"
    print(f"| {i} | `{f}` | {verdict} | {marg_s} | {delta_s} | **{cat}** | {note[:60]} |")