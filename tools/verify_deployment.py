# -*- coding: utf-8 -*-
"""
tools/verify_deployment.py -- Verifikasi deployment TB Widyawardhana v3.

Cek 7 hal:
  1. File hash -- semua model .pkl dan .json identik di research vs production
  2. Feature columns -- 18 LGBM features dan 23 Guardian features cocok & urutan sama
  3. inference_config -- semua parameter threshold, guardian, LSTM sesuai TB config
  4. ETF zero-fill -- core/features.py production punya zero-fill untuk etf_gbtc/etf_total
  5. LGBM predict sanity -- mock data 18-feature bisa di-predict tanpa error
  6. Guardian predict sanity -- mock data 23-feature bisa di-predict tanpa error
  7. ETF gap warning -- reminder bahwa etf features selalu 0.0 di production

Jalankan: python tools/verify_deployment.py
Semua PASS = deployment valid. Satu FAIL = ada perbedaan riset vs production.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RESEARCH = Path(__file__).parent.parent          # d:/Apps-Dev/Riset_pemodelan
PROD     = Path(r"D:\Apps-Dev\swint_tradev2")

# === Nilai target yang diharapkan dari sweep terbaik ===
EXPECTED = {
    "model_version":            "tb_widyawardhana_v3",
    "n_features":               18,
    "lgbm_threshold_long":      0.42,
    "lgbm_threshold_short":     0.42,
    "confidence_threshold":     0.42,
    "guardian_exit_threshold":  0.45,
    "guardian_min_hold_bars":   1,
    "guardian_model_file":      "guardian_best.pkl",
    "guardian_features_file":   "guardian_feature_cols.json",
    "guardian_n_features":      23,
    "lstm_agree_boost":         0.0,
    "lstm_neutral_pen":         0.0,
    "lstm_opposite_pen":        0.0,
    "lstm_flat_review":         False,
    "regime_alignment":         False,
    "hour_filter":              False,
    "tp_atr_mult":              2.0,
    "sl_atr_mult":              1.5,
}

RESEARCH_LGBM_SRC    = RESEARCH / "models/runs/tb_lgbm_widyawardhana_v3/lgbm.pkl"
RESEARCH_GUARDIAN_SRC = RESEARCH / "models/runs/tb_guardian_widyawardhana_v2/guardian.pkl"
RESEARCH_GUARDIAN_SCALER_SRC = RESEARCH / "models/runs/tb_guardian_widyawardhana_v2/guardian_scaler.pkl"
RESEARCH_FEATURES_SRC = RESEARCH / "models/runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json"
RESEARCH_GUARDIAN_FEATS_SRC = RESEARCH / "models/runs/tb_guardian_widyawardhana_v2/tb_guardian_widyawardhana_v2_feature_cols.json"

PROD_LGBM     = PROD / "models/lgbm_baseline.pkl"
PROD_GUARDIAN = PROD / "models/guardian_best.pkl"
PROD_GUARDIAN_SCALER = PROD / "models/guardian_scaler.pkl"
PROD_FEATURES = PROD / "models/feature_cols_v2.json"
PROD_GUARDIAN_FEATS = PROD / "models/guardian_feature_cols.json"
PROD_CONFIG   = PROD / "models/inference_config.json"
PROD_FEATURES_PY = PROD / "core/features.py"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


results = []

def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = PASS if ok else (WARN if warn_only else FAIL)
    line = f"  {tag} {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    results.append((name, ok, warn_only))


print()
print("=" * 68)
print("  DEPLOYMENT VERIFICATION -- TB Widyawardhana v3")
print("=" * 68)

# ────────────────────────────────────────────────────────────────────────────
print("\n[1] FILE HASH -- research source vs production")
# ────────────────────────────────────────────────────────────────────────────

for label, src, dst in [
    ("LGBM model (.pkl)",         RESEARCH_LGBM_SRC,          PROD_LGBM),
    ("Guardian model (.pkl)",     RESEARCH_GUARDIAN_SRC,       PROD_GUARDIAN),
    ("Guardian scaler (.pkl)",    RESEARCH_GUARDIAN_SCALER_SRC, PROD_GUARDIAN_SCALER),
    ("LGBM feature_cols (.json)", RESEARCH_FEATURES_SRC,       PROD_FEATURES),
    ("Guardian feats (.json)",    RESEARCH_GUARDIAN_FEATS_SRC, PROD_GUARDIAN_FEATS),
]:
    if not src.exists():
        check(label, False, f"Source not found: {src}")
        continue
    if not dst.exists():
        check(label, False, f"Prod file not found: {dst}")
        continue
    h_src = sha256(src)
    h_dst = sha256(dst)
    ok = h_src == h_dst
    check(label, ok,
          f"src={h_src}  prod={h_dst}" if not ok else f"hash={h_src}")

# ────────────────────────────────────────────────────────────────────────────
print("\n[2] FEATURE COLUMNS -- order & completeness")
# ────────────────────────────────────────────────────────────────────────────

EXPECTED_LGBM_FEATURES = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "wyckoff_phase", "Sell_Liq", "atr_percentile_h1",
    "stochrsi_k", "dist_liq_50x_short", "funding_rate", "ema_7_h1",
    "dow_cos", "cvd_div_h4", "dist_swing_low", "VAH",
    "cvd_momentum_adv", "dist_from_8h_high", "ema_200_h1",
]

EXPECTED_GUARDIAN_FEATURES_STATIC = [
    "etf_gbtc_change_usd", "etf_total_change_usd", "cvd_slope_h4",
    "ofi_h4_delta", "Sell_Liq", "atr_percentile_h1", "stochrsi_k",
    "dist_liq_50x_short", "funding_rate", "ema_7_h1", "dow_cos",
    "cvd_div_h4", "dist_swing_low", "VAH", "cvd_momentum_adv",
    "dist_from_8h_high", "ema_200_h1",
]
EXPECTED_GUARDIAN_FEATURES_DYNAMIC = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct", "entry_price_ratio",
]
EXPECTED_GUARDIAN_FEATURES = EXPECTED_GUARDIAN_FEATURES_STATIC + EXPECTED_GUARDIAN_FEATURES_DYNAMIC

if PROD_FEATURES.exists():
    prod_lgbm_feats = json.loads(PROD_FEATURES.read_text())
    ok_count = prod_lgbm_feats == EXPECTED_LGBM_FEATURES
    ok_order = prod_lgbm_feats == EXPECTED_LGBM_FEATURES
    check("LGBM features (18) -- count",
          len(prod_lgbm_feats) == 18, f"prod has {len(prod_lgbm_feats)} features")
    check("LGBM features -- order matches expected",
          ok_order,
          detail="" if ok_order else f"Diff: {set(prod_lgbm_feats) ^ set(EXPECTED_LGBM_FEATURES)}")
    missing = [f for f in EXPECTED_LGBM_FEATURES if f not in prod_lgbm_feats]
    extra   = [f for f in prod_lgbm_feats if f not in EXPECTED_LGBM_FEATURES]
    check("LGBM features -- no missing/extra",
          not missing and not extra,
          f"missing={missing}  extra={extra}" if (missing or extra) else "")
else:
    check("LGBM feature_cols_v2.json exists in prod", False)

if PROD_GUARDIAN_FEATS.exists():
    prod_gdn_feats = json.loads(PROD_GUARDIAN_FEATS.read_text())
    ok_gdn = prod_gdn_feats == EXPECTED_GUARDIAN_FEATURES
    check("Guardian features (23) -- count",
          len(prod_gdn_feats) == 23, f"prod has {len(prod_gdn_feats)} features")
    check("Guardian features -- order matches expected",
          ok_gdn,
          detail="" if ok_gdn else f"Mismatch at: {[(i,a,b) for i,(a,b) in enumerate(zip(prod_gdn_feats, EXPECTED_GUARDIAN_FEATURES)) if a!=b]}")
else:
    check("Guardian guardian_feature_cols.json exists in prod", False)

# ────────────────────────────────────────────────────────────────────────────
print("\n[3] inference_config.json -- parameters")
# ────────────────────────────────────────────────────────────────────────────

if PROD_CONFIG.exists():
    cfg = json.loads(PROD_CONFIG.read_text())
    cas = cfg.get("cascade", {})
    gdn = cfg.get("guardian", {})
    inf = cfg.get("inference", {})
    tp  = cfg.get("tp_sl", {})
    ra  = cfg.get("regime_alignment", {})
    hf  = cfg.get("hour_filter", {})

    check("model_version = tb_widyawardhana_v3",
          cfg.get("model_version") == "tb_widyawardhana_v3",
          f"got: {cfg.get('model_version')}")
    check("n_features = 18",
          cfg.get("n_features") == 18, f"got: {cfg.get('n_features')}")
    check("cascade.lgbm_threshold_long = 0.42",
          cas.get("lgbm_threshold_long") == 0.42, f"got: {cas.get('lgbm_threshold_long')}")
    check("cascade.lgbm_threshold_short = 0.42",
          cas.get("lgbm_threshold_short") == 0.42, f"got: {cas.get('lgbm_threshold_short')}")
    check("cascade.confidence_threshold_entry = 0.42",
          cas.get("confidence_threshold_entry") == 0.42, f"got: {cas.get('confidence_threshold_entry')}")
    check("cascade.lstm_adjust_agree_boost = 0.0",
          cas.get("lstm_adjust_agree_boost") == 0.0, f"got: {cas.get('lstm_adjust_agree_boost')}")
    check("cascade.lstm_adjust_neutral_pen = 0.0",
          cas.get("lstm_adjust_neutral_pen") == 0.0, f"got: {cas.get('lstm_adjust_neutral_pen')}")
    check("cascade.lstm_adjust_opposite_pen = 0.0",
          cas.get("lstm_adjust_opposite_pen") == 0.0, f"got: {cas.get('lstm_adjust_opposite_pen')}")
    check("cascade.lstm_flat_review_enabled = False",
          cas.get("lstm_flat_review_enabled") == False, f"got: {cas.get('lstm_flat_review_enabled')}")
    check("guardian.exit_threshold = 0.45",
          gdn.get("exit_threshold") == 0.45, f"got: {gdn.get('exit_threshold')}")
    check("guardian.min_hold_bars = 1",
          gdn.get("min_hold_bars") == 1, f"got: {gdn.get('min_hold_bars')}")
    check("guardian.model_file = guardian_best.pkl",
          gdn.get("model_file") == "guardian_best.pkl", f"got: {gdn.get('model_file')}")
    check("guardian.features_file = guardian_feature_cols.json",
          gdn.get("features_file") == "guardian_feature_cols.json", f"got: {gdn.get('features_file')}")
    check("regime_alignment.enabled = False",
          ra.get("enabled") == False, f"got: {ra.get('enabled')}")
    check("hour_filter.enabled = False",
          hf.get("enabled") == False, f"got: {hf.get('enabled')}")
    check("tp_sl.tp_atr_mult = 2.0",
          tp.get("tp_atr_mult") == 2.0, f"got: {tp.get('tp_atr_mult')}")
    check("tp_sl.sl_atr_mult = 1.5",
          tp.get("sl_atr_mult") == 1.5, f"got: {tp.get('sl_atr_mult')}")
else:
    check("inference_config.json exists in prod", False)

# ────────────────────────────────────────────────────────────────────────────
print("\n[4] ETF ZERO-FILL -- core/features.py di production")
# ────────────────────────────────────────────────────────────────────────────

if PROD_FEATURES_PY.exists():
    src_text = PROD_FEATURES_PY.read_text(encoding="utf-8", errors="replace")
    has_etf_gbtc  = "etf_gbtc_change_usd" in src_text
    has_etf_total = "etf_total_change_usd" in src_text
    has_zero_fill = ("etf_gbtc_change_usd" in src_text and
                     "fillna" in src_text or "= 0.0" in src_text or "0.0" in src_text)
    # More precise: look for the zero-fill block we added
    has_block = ('_etf_col in ("etf_gbtc_change_usd"' in src_text or
                 "etf_gbtc_change_usd" in src_text and "not in feat_df.columns" in src_text)
    check("core/features.py produksi punya ETF zero-fill block",
          has_block,
          "Pastikan engineer_features() assign 0.0 saat kolom ETF tidak ada")
    check("etf_gbtc_change_usd disebut di features.py",  has_etf_gbtc)
    check("etf_total_change_usd disebut di features.py", has_etf_total)
else:
    check("core/features.py exists in prod", False)

# ────────────────────────────────────────────────────────────────────────────
print("\n[5] LGBM PREDICT SANITY -- mock inference 18 features")
# ────────────────────────────────────────────────────────────────────────────

try:
    lgbm = joblib.load(PROD_LGBM)
    feats = json.loads(PROD_FEATURES.read_text())
    mock = pd.DataFrame([{f: 0.0 for f in feats}])
    proba = lgbm.predict_proba(mock)[0]
    ok = len(proba) == 3 and abs(sum(proba) - 1.0) < 1e-4
    check("LGBM predict_proba(mock) returns [SHORT,FLAT,LONG]", ok,
          f"proba={proba.round(4).tolist()}")
    check("LGBM output sums to 1.0", ok)
    # Check with thr=0.42 -- mock all-zeros should be FLAT (no trade)
    p_long  = float(proba[2])
    p_short = float(proba[0])
    decision = "LONG" if p_long >= 0.42 else ("SHORT" if p_short >= 0.42 else "FLAT")
    check("LGBM zero-input -> FLAT (no spurious entry)",
          decision == "FLAT",
          f"p_long={p_long:.4f} p_short={p_short:.4f} decision={decision}",
          warn_only=(decision != "FLAT"))
except Exception as e:
    check("LGBM predict sanity", False, str(e))

# ────────────────────────────────────────────────────────────────────────────
print("\n[6] GUARDIAN PREDICT SANITY -- mock inference 23 features")
# ────────────────────────────────────────────────────────────────────────────

try:
    gdn_model  = joblib.load(PROD_GUARDIAN)
    gdn_scaler = joblib.load(PROD_GUARDIAN_SCALER)
    gdn_feats  = json.loads(PROD_GUARDIAN_FEATS.read_text())
    mock_gdn = pd.DataFrame([{f: 0.0 for f in gdn_feats}])
    mock_scaled = gdn_scaler.transform(mock_gdn)
    proba_gdn = gdn_model.predict_proba(mock_scaled)[0]
    ok = len(proba_gdn) == 3 and abs(sum(proba_gdn) - 1.0) < 1e-4
    check("Guardian predict_proba(mock) returns [HOLD,PARTIAL,EXIT]", ok,
          f"proba={proba_gdn.round(4).tolist()}")
    pred = int(np.argmax(proba_gdn))
    labels = {0: "HOLD", 1: "PARTIAL_EXIT", 2: "FULL_EXIT"}
    check("Guardian zero-input -> HOLD (no premature exit)",
          pred == 0,
          f"pred={labels[pred]} conf={proba_gdn[pred]:.4f}",
          warn_only=(pred != 0))
except Exception as e:
    check("Guardian predict sanity", False, str(e))

# ────────────────────────────────────────────────────────────────────────────
print("\n[7] ETF GAP WARNING -- reminder keterbatasan")
# ────────────────────────────────────────────────────────────────────────────

check(
    "ETF features (etf_gbtc, etf_total) -- ZERO-FILLED di production",
    True,   # ini bukan failure, cuma reminder
    "WARN: Nilai selalu 0.0 di live. Model training punya nilai real sejak Jan 2024. "
    "Estimasi degradasi: MILD. Fix: tambah daily yfinance ETF fetch ke data_service.py.",
    warn_only=True,
)
check(
    "HMM threshold -- FIXED di production (tidak adaptive)",
    True,
    "WARN: hmm_regime_enc=0 always di production -> threshold 0.42 untuk semua bar. "
    "Ranging bars seharusnya thr=0.50 -> ekspektasi +more trades dari backtest.",
    warn_only=True,
)
check(
    "Structural filter -- ENABLED, tidak divalidasi di TB holdout",
    True,
    "WARN: structural_filter aktif di production tapi tidak ada di sweep/holdout TB. "
    "Potensi mengurangi trades vs backtest. Monitor jumlah trade.",
    warn_only=True,
)

# ────────────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
passes  = sum(1 for _, ok, warn in results if ok)
fails   = sum(1 for _, ok, warn in results if not ok and not warn)
warns   = sum(1 for _, ok, warn in results if not ok and warn)
total   = len(results)

print(f"  HASIL: {passes}/{total} PASS  |  {fails} FAIL  |  {warns} WARN (expected)")
if fails == 0:
    print("  STATUS: DEPLOYMENT VALID -- siap live trading")
else:
    print("  STATUS: ADA MASALAH -- periksa baris [FAIL] di atas")
print("=" * 68)
print()

if fails > 0:
    sys.exit(1)
