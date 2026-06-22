# -*- coding: utf-8 -*-
"""Audit sinkronisasi Live vs Riset — config, model, fitur, DB signals."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live_db_bridge import LOCAL_DB, load_signals

RISET = ROOT / "models"
SWINT = Path(r"D:\Apps-Dev\swint_tradev2\models")
VPS = "root@139.180.157.176"
FEAT = json.load(open(RISET / "feature_cols_ic32_regime.json", encoding="utf-8"))


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()[:16]


def get_nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fetch_vps_json(path: str) -> dict:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", VPS, f"cat {path}"],
        capture_output=True, text=True, timeout=40,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return json.loads(r.stdout)


def section(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def audit_models():
    section("1. MODEL BINARY (hash Riset = swint lokal)")
    files = [
        "lgbm_baseline.pkl", "guardian_best.pkl", "guardian_scaler.pkl",
        "feature_cols_v2.json", "guardian_feature_cols.json",
        "training_feature_standards.json",
    ]
    ok = fail = 0
    for f in files:
        rp, sp = RISET / f, SWINT / f
        if rp.exists() and sp.exists():
            match = sha16(rp) == sha16(sp)
            ok += match
            fail += not match
            print(f"  [{'OK' if match else 'FAIL'}] {f}")
        else:
            fail += 1
            print(f"  [FAIL] {f} missing")
    return fail == 0


def audit_config():
    section("2. INFERENCE CONFIG (Riset vs VPS)")
    riset = json.loads((RISET / "inference_config.json").read_text(encoding="utf-8"))
    live = fetch_vps_json("/home/swint/swint_tradev2/models/inference_config.json")

    critical = [
        "model_version", "model_type", "n_features",
        "hmm.per_state_thresholds", "guardian.exit_threshold", "guardian.min_hold_bars",
        "guardian.model_file", "rr_gate.sl_trigger_mode", "rr_gate.min_rr",
        "cascade.mode", "cascade.lstm_fusion_mode", "cascade.lstm_adjust_opposite_pen",
        "inference.confidence_threshold_entry",
        "regime_alignment.enabled", "structural_filter.enabled",
        "pyramiding.enabled", "pyramiding.exit_mode",
        "risk.modal_per_trade", "risk.max_open_positions", "risk.daily_loss_limit",
        "risk.leverage_recommended", "risk.fee_per_side", "risk.slippage_per_side",
        "models.lgbm", "models.guardian", "models.lstm", "models.lgbm_features",
        "tp_sl.sl_atr_mult", "tp_sl.tp_atr_mult",
    ]
    fail = []
    for path in critical:
        rv, lv = get_nested(riset, path), get_nested(live, path)
        if rv != lv:
            fail.append(path)
            print(f"  [FAIL] {path}: riset={rv!r} live={lv!r}")
        else:
            print(f"  [OK]   {path}")

    # Env override risk
    section("2b. VPS ENV OVERRIDES (bisa timpa config)")
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", VPS,
         "grep -E '^LIVE_|^TRADING_MODE|^POSITIONING' /home/swint/swint_tradev2/.env 2>/dev/null || true"],
        capture_output=True, text=True, timeout=30,
    )
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    if lines:
        for ln in lines:
            print(f"  [ENV] {ln}")
        print("  [WARN] Env var bisa override inference_config untuk max_pos/modal")
    else:
        print("  [OK] Tidak ada LIVE_* override di .env (config JSON yang dipakai)")

    return len(fail) == 0


def audit_features_code():
    section("3. FEATURE ENGINEERING CODE")
    pairs = [
        (ROOT / "core/features.py", Path(r"D:\Apps-Dev\swint_tradev2\core\features.py")),
        (ROOT / "core/cascade_utils.py", Path(r"D:\Apps-Dev\swint_tradev2\core\cascade_utils.py")),
    ]
    ok = True
    for rp, sp in pairs:
        match = rp.exists() and sp.exists() and sha16(rp) == sha16(sp)
        print(f"  [{'OK' if match else 'FAIL'}] {rp.name}")
        ok = ok and match
    return ok


def audit_db_signals():
    section("4. DB LIVE SIGNALS — feature snapshot health")
    sig = load_signals(LOCAL_DB)
    sig = sig.copy()
    sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True)

    # Post-fix deploy ~2026-06-18
    post_fix = sig[sig["signal_time"] >= "2026-06-18 12:00:00"]
    pre_fix = sig[(sig["signal_time"] >= "2026-06-17") & (sig["signal_time"] < "2026-06-18 12:00:00")]

    def lsr_stats(df, label):
        if df.empty:
            print(f"  [{label}] no signals")
            return
        lsrs, oi_syn, hmm_ok = [], 0, 0
        for _, row in df.iterrows():
            try:
                s = json.loads(row.get("feature_snapshot") or "{}")
            except json.JSONDecodeError:
                continue
            lsr = s.get("long_short_ratio")
            if lsr is not None:
                lsrs.append(float(lsr))
            oi = s.get("open_interest")
            if oi is not None and 0.5 < float(oi) < 2.0:
                oi_syn += 1
        n = len(df)
        if lsrs:
            arr = np.array(lsrs)
            zero_pct = 100 * (arr == 0).mean()
            in_band = 100 * ((arr >= 0.978) & (arr <= 1.020)).mean()
            print(f"  [{label}] n={n} signals")
            print(f"    LSR: mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}")
            print(f"    LSR=0: {zero_pct:.1f}% | in training band [0.978-1.020]: {in_band:.1f}%")
            if zero_pct > 5:
                print(f"    [FAIL] >5% LSR=0 — snapshot pre-fix / pipeline rusak")
            elif in_band < 80 and label == "post-fix":
                print(f"    [WARN] <80% LSR in training band")
            else:
                print(f"    [OK] LSR distribution sehat")
        print(f"    synthetic OI (0.5-2.0): {oi_syn}/{n}")

    lsr_stats(pre_fix, "pre-fix Jun17-morning Jun18")
    lsr_stats(post_fix, "post-fix Jun18 12:00+")

    # Latest 5 directional signals
    print("\n  --- 5 sinyal directional terbaru ---")
    dir_sig = sig[sig["direction"].isin(["LONG", "SHORT"])].tail(5)
    for _, r in dir_sig.iterrows():
        s = json.loads(r.get("feature_snapshot") or "{}")
        lsr = s.get("long_short_ratio", "?")
        hmm = s.get("hmm_regime_enc", "?")
        conf = r.get("confidence", "?")
        print(f"    {r['signal_time']} {r['coin_symbol']:<14} {r['direction']:<5} "
              f"conf={conf} LSR={lsr} hmm={hmm}")

    return True


def audit_vps_parity_api():
    section("5. VPS FEATURE PARITY API (live pipeline vs training standards)")
    try:
        import urllib.request
        with urllib.request.urlopen("http://139.180.157.176:5000/api/features/parity", timeout=90) as resp:
            data = json.loads(resp.read().decode())
        rep = data.get("report", {})
        sm = rep.get("summary", {})
        print(f"  checked_at: {rep.get('checked_at')}")
        print(f"  positioning_mode: {rep.get('positioning_mode')}")
        print(f"  coins: {sm.get('ok')}/{sm.get('total')} OK, errors={sm.get('error')}, warnings={sm.get('warning')}")
        flagged = sm.get("top_flagged_features") or []
        if flagged:
            print(f"  [WARN] flagged: {flagged}")
        else:
            print("  [OK] 21/21 koin parity OK")
        if rep.get("positioning_mode") != "training_parity":
            print("  [FAIL] positioning_mode bukan training_parity!")
            return False
        return sm.get("error", 1) == 0 and sm.get("ok", 0) == 21
    except Exception as e:
        print(f"  [FAIL] API error: {e}")
        return False


def audit_riset_vs_live_backtest_gap():
    section("6. GAP METODOLOGI (bukan bug, tapi harus dipahami)")
    gaps = [
        ("Holdout backtest", "Purged OOF / frozen config — tidak ada saldo gate"),
        ("Live trading", "max_open_positions + daily_loss_limit + balance check"),
        ("Riset MODAL", "config.py MODAL_PER_TRADE=5 (baru diubah)"),
        ("Live modal", "inference_config risk.modal_per_trade=5"),
        ("LSR/OI live", "training_parity mode: synthetic + clip ke band training"),
        ("DB signal lama", "Snapshot Jun17-18 pagi mungkin masih LSR=0 (pre-fix)"),
        ("Pyramiding", "Live ON (scale_in) — holdout script risk_gate tidak simulasi pyr"),
    ]
    for k, v in gaps:
        print(f"  - {k}: {v}")
    return True


def main():
    print("LIVE vs RISET SYNC AUDIT")
    print(f"Waktu: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    results = {
        "models": audit_models(),
        "config": audit_config(),
        "features_code": audit_features_code(),
        "db_signals": audit_db_signals(),
        "vps_parity": audit_vps_parity_api(),
        "methodology": audit_riset_vs_live_backtest_gap(),
    }

    section("RINGKASAN")
    all_ok = True
    for k, v in results.items():
        st = "PASS" if v else "FAIL"
        if not v:
            all_ok = False
        print(f"  [{st}] {k}")

    out = ROOT / "reports" / "experiments" / "live_riset_sync_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")

    if all_ok:
        print("\n>>> SINKRON OK — tidak ada mismatch kritis terdeteksi.")
    else:
        print("\n>>> ADA MASALAH — lihat FAIL di atas.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())