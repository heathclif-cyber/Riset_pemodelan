# DEPLOY GUIDE: Hierarchical Cascade Model

**Purpose:** Quick-reference deployment guide after running pipeline training.  
**Target:** [`swint_tradev2/app/services/inference.py`](../swint_tradev2/app/services/inference.py)  
**Prerequisites:** Pipeline training complete (`run_pipeline.py --train` or individual fase 04-06).

---

## 1. Architecture Overview

### Before (Stacked Ensemble — ❌ Removed)

```
LGBM (H1) ──┐
             ├──→ LogisticRegression (meta-learner) → Isotonic Calibrator → Signal
LSTM (H1)  ──┘
```

**Problems solved:**
- Meta-learner compressed 6-D probability → lost variance
- Isotonic Calibrator over-calibrated to FLAT (0.985–0.998) — **confirmed in Railway production logs**
- Calibrator disabled in inference but still used in backtest → misleading metrics

### After (Hierarchical Cascade — ✅ Active)

```
STEP 1: H4 LGBM  ──→ bias direction (LONG/SHORT/FLAT) at ≥0.60
STEP 2: H1 LGBM  ──→ entry signal at ≥0.62 threshold
STEP 3: LSTM     ──→ confirmation vote (agree/reject)
STEP 4: Decision ──→ signal only if STEP 1 + 2 + 3 agree
```

---

## 2. Training Pipeline

| Fase | File | Output |
|:----:|------|--------|
| `04` | [`04_train_lgbm_h4.py`](pipeline/04_train_lgbm_h4.py) | `lgbm_h4.pkl` + `h4_feature_cols.json` |
| `05` | [`05_train_lgbm_h1.py`](pipeline/05_train_lgbm_h1.py) | `lgbm_baseline.pkl` + `feature_cols_v2.json` |
| `06` | [`06_train_lstm.py`](pipeline/06_train_lstm.py) | `lstm.pt` + `lstm_scaler.pkl` |

### Commands

```bash
# Full pipeline (fetch → clean → engineer → train → backtest)
python run_pipeline.py --all

# Train only (H4 + H1 + LSTM)
python run_pipeline.py --train

# Train H4 only
python run_pipeline.py --train-h4

# Train + backtest
python run_pipeline.py --train --backtest

# All 20 coins
python run_pipeline.py --all --all-coins
```

---

## 3. Model Artifacts — What to Deploy

### Training Output (`models/runs/{run_id}/`)

```
models/runs/{run_id}/
├── lgbm_h4.pkl              ← H4 LGBM
├── h4_feature_cols.json     ← H4 features (30 cols)
├── lgbm_baseline.pkl        ← H1 LGBM
├── feature_cols_v2.json     ← H1 features (85 cols)
├── lstm.pt                  ← LSTM state dict
├── lstm_scaler.pkl          ← LSTM StandardScaler
├── h4_lgbm_cv_results.json  ← H4 CV metrics (optional)
├── lgbm_cv_results.json     ← H1 CV metrics (optional)
├── lstm_cv_results.json     ← LSTM CV metrics (optional)
├── backtest_results.json    ← Walk-forward metrics (optional)
└── shap_ranking.json        ← SHAP importance (optional)
```

### Files to Copy to `models/`

| File | Required | Description |
|------|----------|-------------|
| `lgbm_h4.pkl` | ✅ YES | H4 LGBM regime filter |
| `h4_feature_cols.json` | ✅ YES | H4 feature column list |
| `lgbm_baseline.pkl` | ✅ YES | H1 LGBM entry signal |
| `feature_cols_v2.json` | ✅ YES | H1 feature column list |
| `lstm.pt` | ✅ YES | LSTM state dict |
| `lstm_scaler.pkl` | ✅ YES | LSTM StandardScaler |
| `inference_config.json` | ✅ YES | Thresholds & parameters |
| `calibrator.pkl` | ❌ NO (legacy) | Do NOT deploy |
| `ensemble_meta.pkl` | ❌ NO (legacy) | Do NOT deploy |

```bash
# Copy command (adjust {latest_run_id})
cp models/runs/{latest_run_id}/lgbm_h4.pkl          models/lgbm_h4.pkl
cp models/runs/{latest_run_id}/h4_feature_cols.json models/h4_feature_cols.json
cp models/runs/{latest_run_id}/lgbm_baseline.pkl    models/lgbm_baseline.pkl
cp models/runs/{latest_run_id}/feature_cols_v2.json models/feature_cols_v2.json
cp models/runs/{latest_run_id}/lstm.pt              models/lstm.pt
cp models/runs/{latest_run_id}/lstm_scaler.pkl      models/lstm_scaler.pkl
```

---

## 4. `inference_config.json` — Configuration File

This file controls all runtime parameters. Generate via backtest or create manually.

```json
{
  "model_version": "hierarchical_v1",
  "hierarchical_thresholds": {
    "h4_threshold_long": 0.60,
    "h4_threshold_short": 0.60,
    "h1_threshold_long": 0.62,
    "h1_threshold_short": 0.62,
    "lstm_confirmation": true
  },
  "inference": {
    "confidence_threshold_entry": 0.60,
    "confidence_full": 0.75,
    "confidence_half": 0.60,
    "seq_len": 32,
    "num_classes": 3
  },
  "fallback": {
    "skip_trade_on_h4_missing": true,
    "skip_trade_on_h1_missing": true,
    "allow_degrade_on_lstm_fail": false,
    "lstm_timeout_seconds": 5.0,
    "lstm_skip_on_timeout": true
  },
  "position_sizing": {
    "confidence_full": 0.75,
    "size_full": 1.5,
    "confidence_half": 0.65,
    "size_half": 1.0,
    "size_none": 0.0
  },
  "debug": {
    "enabled": false,
    "log_probas": true,
    "log_timing": true,
    "max_log_per_minute": 60
  }
}
```

---

## 5. Inference Service Verification

### Check `inference.py` Before Restart

File: [`../swint_tradev2/app/services/inference.py`](../swint_tradev2/app/services/inference.py)

| What to Check | Expected | Location |
|---------------|----------|----------|
| `model_type` default | `"hierarchical"` | `predict()` method, line ~102 |
| H4 model loading | Log `"[inference] H4 model loaded: 30 features"` | `_load_bundle()`, lines ~196-209 |
| Hierarchical branch | Routes to `_hierarchical_proba()` | `_run_model()`, lines ~245-246 |
| Ensemble path | Marked `DEPRECATED` | `_load_bundle()`, lines ~211-220 |
| Calibrator | Commented out or unused | `_load_bundle()`, lines ~258-292 |

### Decision Flow — `_hierarchical_proba()`

File: [`../swint_tradev2/app/services/inference.py:310-393`](../swint_tradev2/app/services/inference.py:310)

```
Input: df (latest candle row)
Output: np.array([P(SHORT), P(FLAT), P(LONG)])  — default FLAT

STEP 1 — H4 Bias Filter:
  X_h4 = df[h4_feature_cols].fillna(0).iloc[[-1]]
  h4_p = h4_lgbm.predict_proba(X_h4)[0]
  if h4_p[2] ≥ 0.60 → bias = LONG
  elif h4_p[0] ≥ 0.60 → bias = SHORT
  else → return FLAT (no entry)

STEP 2 — H1 Entry Signal:
  h1_p = lgbm.predict_proba(df[feature_cols])[0]
  if bias=LONG and h1_p[2] < 0.62 → return FLAT
  if bias=SHORT and h1_p[0] < 0.62 → return FLAT

STEP 3 — LSTM Confirmation:
  lstm_p = lstm_predict(...)  # [P(SHORT), P(FLAT), P(LONG)]
  lstm_dir = argmax(lstm_p)
  # class_0=SHORT(bearish), class_1=FLAT(→REJECT), class_2=LONG(bullish)
  if bias=LONG and lstm_dir ≠ 2 → return FLAT
  if bias=SHORT and lstm_dir ≠ 0 → return FLAT

STEP 4 — Signal:
  return h1_p  # all 3 models agree
```

### Key Parameter Source

| Parameter | Value | Defined In |
|-----------|-------|------------|
| `H4_THRESHOLD_LONG` | 0.60 | [`config.py:174`](config.py:174) |
| `H4_THRESHOLD_SHORT` | 0.60 | [`config.py:175`](config.py:175) |
| `H1_THRESHOLD_LONG` | 0.62 | [`config.py:177`](config.py:177) |
| `H1_THRESHOLD_SHORT` | 0.62 | [`config.py:178`](config.py:178) |
| `LSTM_CONFIRMATION_ENABLED` | True | [`config.py:179`](config.py:179) |
| `H4_SWING_LABEL_MIN_RR` | 2.0 | [`config.py:165`](config.py:165) |
| `H4_PURGE_GAP_BARS` | 6 | [`config.py:172`](config.py:172) |
| `H4_N_FOLDS` | 8 | [`config.py:171`](config.py:171) |
| H1 class weights | `{0:3.0, 1:1.0, 2:3.0}` | [`05_train_lgbm_h1.py:107`](pipeline/05_train_lgbm_h1.py:107) |
| `LSTM_SEQ_LEN` | 32 | `config.py` |
| `LSTM_HIDDEN` | 128 | `config.py` |

---

## 6. Deployment Steps

### Step 1 — Copy Model Files

Copy 6 required files from `models/runs/{latest_run_id}/` to `models/` (see Section 3).

### Step 2 — Deploy `inference_config.json`

Ensure `models/inference_config.json` exists with correct thresholds (see Section 4).

### Step 3 — Remove Legacy Files (Optional)

```bash
# Safe to remove — not used by hierarchical cascade
del models\calibrator.pkl
del models\ensemble_meta.pkl
```

### Step 4 — Restart Web Application

```bash
# Flask app restart (command depends on deployment)
# InferenceService uses lazy loading + TTL cache
```

### Step 5 — Verify Startup Logs

Check for these log lines:

| Log Message | Meaning |
|-------------|---------|
| `"[inference] H4 model loaded: 30 features"` | H4 LGBM loaded successfully |
| `"model_type = hierarchical"` | Using hierarchical cascade |
| No `"WARNING — fallback"` logs | All models present |

### Step 6 — Monitor Initial Signals

- Signal distribution should show some FLAT (normal, most candles)
- No probability values near 0.99 (legacy calibrator symptom)
- Trade frequency: 0–3 per coin per day (see Section 9)

---

## 7. LSTM Output Interpretation

| Index | Label | Meaning in Cascade |
|:-----:|-------|--------------------|
| `0` | **SHORT** (bearish) | ✅ Confirms SHORT bias |
| `1` | **FLAT** (neutral) | ❌ **REJECT** — always rejects signal |
| `2` | **LONG** (bullish) | ✅ Confirms LONG bias |

**Rule:** LSTM agrees ONLY if `argmax == 2` (for LONG bias) or `argmax == 0` (for SHORT bias). FLAT output always causes REJECT.

---

## 8. Failure Mode Reference

| Scenario | Fallback | Alert |
|----------|----------|-------|
| H4 LGBM missing | **skip_trade** (return FLAT) | ⚠️ Critical — no signals |
| H1 LGBM missing | **skip_trade** (return FLAT) | ⚠️ Critical — no signals |
| LSTM missing | **degrade_mode** if allowed, else skip_trade | ⚠️ High — confirmation off |
| LSTM timeout (>5s) | **skip_confirmation** (H4+H1 only) | ⚠️ High — LSTM bypassed |
| Scaler missing | **skip_trade** (LSTM unusable) | ⚠️ High |
| H4 features missing in df | **skip_trade** (return FLAT) | ⚠️ Critical — data issue |

> **Recommendation:** Keep `allow_degrade_on_lstm_fail = false` initially. Enable only after verifying H4+H1 reliability independently.

---

## 9. Trade Frequency Expectations

| Scope | Normal Range | If Below | If Above |
|-------|-------------|----------|----------|
| Per coin | **0–3 trades/day** | Check FLAT rate >80% | Check threshold too low |
| All 20 coins | **10–30 trades/day** | Possible data alignment issue | Possible overfitting |
| Per week per coin | **5–15 trades** | — | — |

- **0 trades for 72h** → FLAT rate >95%, check H4 model & data alignment
- **>10 trades/day per coin** → thresholds too low or model overfit

---

## 10. Monitoring Checklist (Post-Deploy)

| Check | Frequency | Action if Abnormal |
|-------|-----------|--------------------|
| H4 model loaded in logs | Once at startup | Re-copy `lgbm_h4.pkl` |
| FLAT rate < 80% | Daily | Check data pipeline |
| Trade frequency in expected range | Daily | Check thresholds |
| LSTM confirmation rate | Weekly | Check if LSTM rejecting too many |
| Probability std > 0.05 | Weekly | Possible model collapse if below |
| No "fallback" warnings | Daily | Check model files integrity |

---

## 11. Backtest vs Live — Known Differences

| Aspect | Backtest | Live | Impact |
|--------|----------|------|--------|
| Candle close | Perfect — known at close | 100ms–2s delay | Slippage |
| Feature availability | All pre-computed | Computed real-time | Timing delay |
| H4 alignment | Perfect alignment | Wait 4h for H4 close | No signals initially |
| Bid/Ask spread | Ignored | 0.01%–0.05% | Profit overestimated |
| Order fill | Assumed perfect | Partial/slippage/rejection | Higher drawdown |

**Mitigations:** Conservative thresholds (0.60/0.62), hold-out validation, paper trade 2 weeks first.

---

## 12. Debug Mode (Troubleshooting)

Enable in `inference_config.json`:
```json
{ "debug": { "enabled": true, "log_probas": true } }
```

Expected log output:
```
[DEBUG] _hierarchical_proba() — BTCUSDT
  ├── H4_probs:  [0.21, 0.15, 0.64]  → bias = LONG (thr=0.60 ✓)
  ├── H1_probs:  [0.18, 0.12, 0.70]  → LONG entry (thr=0.62 ✓)
  ├── LSTM_probs: [0.22, 0.08, 0.70] → LSTM setuju LONG (argmax=2)
  ├── Decision:  SIGNAL_LONG (conf=0.70)
  └── Timing:    H4=1.2ms | H1=0.8ms | LSTM=23.4ms | TOTAL=25.4ms
```

Disable debug mode after troubleshooting to reduce log volume.

---

## 13. Data Alignment Requirement

⚠️ **Critical:** H4 candle must be **fully closed** before being used for inference.

- Use only H4 candles where `close_time <= current_h1_timestamp`
- All H4 features (EMA H4, ATR H4, RSI H4) must be computed from closed H4 candles
- If H4 and H1 are misaligned, H4 bias may flip-flop each H1 bar

---

## 14. ⚠️ Known Architecture Limitations (Next Phase Fixes)

These are the **three main bottlenecks** identified in technical review:

| # | Bottleneck | Current Behavior | Impact | Planned Fix (Phase) |
|:-:|------------|-----------------|--------|---------------------|
| **1** | **H4 is a 3-class classifier, not a regime model** | Outputs FLAT too often as a gate filter | Over-filters valid signals, noisy direction | Convert to binary or regression (Phase 3) |
| **2** | **LSTM is a hard veto** | `if lstm_dir != bias → REJECT` | Many valid signals rejected, trade frequency low | Change to soft confidence adjuster (Phase 2) |
| **3** | **Static thresholds (0.60/0.62)** | Same threshold for all market regimes | Performance varies by market condition | Dynamic thresholding (Phase 2) |

### Improvement Roadmap

#### Phase 1 — Production Readiness (NOW)

| # | Area | Action |
|:-:|------|--------|
| 1 | Monitoring | Implement FLAT rate, proba mean/std, trade gap alert |
| 2 | Failure handling | Implement graceful degradation for all 7 scenarios |
| 3 | Debug logging | Activate structured logging per decision step |
| 4 | Data alignment | Enforce H4 candle fully closed before inference |

#### Phase 2 — Performance Boost

| # | Change | Current | Target |
|:-:|--------|---------|--------|
| 1 | LSTM role | Hard veto (reject if disagree) | Soft confidence adjuster (±0.05) |
| 2 | Thresholds | Static (0.60/0.62) | Dynamic (rolling quantile or volatility-adjusted) |

#### Phase 3 — Advanced Architecture

| # | Change | Current | Target |
|:-:|--------|---------|--------|
| 1 | H4 model | 3-class classifier (LONG/FLAT/SHORT) | Binary classifier or trend strength regression |
| 2 | Position sizing | Confidence-only | Confidence × volatility scaling × drawdown control |
| 3 | Market state | None | Regime-dependent inference (trending/ranging/volatile) |

> **Do Phase 1 first.** Without visibility (monitoring + logging), you can't measure whether Phase 2-3 changes actually improve anything.

---

## Appendix: Key Config Values

### H4 LGBM

| Param | Value |
|-------|-------|
| Features | 30 cols (subset of FEATURE_COLS_V3) |
| Label min RR | 2.0 |
| Purge gap | 6 bars (≈24h) |
| N folds | 8 |
| `max_depth` | 4 |
| `num_leaves` | 15 |

### H1 LGBM

| Param | Value |
|-------|-------|
| Class weights | `{0:3.0, 1:1.0, 2:3.0}` (SHORT=3x, FLAT=1x, LONG=3x) |
| Entry threshold | 0.62 (both directions) |

### LSTM

| Param | Value |
|-------|-------|
| `seq_len` | 32 |
| `hidden` | 128 |
| `layers` | 2 |
| `dropout` | 0.3 |

---

*Generated: 2026-05-03 — For redeploying hierarchical cascade model to web trading app.*
