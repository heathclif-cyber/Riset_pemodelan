# -*- coding: utf-8 -*-
"""Quick VPS deploy verification — promoted stack only."""
import json
import subprocess
import sys

HOST = "root@139.180.157.176"
REMOTE = r"""
import json, os, hashlib
from pathlib import Path
os.chdir('/home/swint/swint_tradev2')
ic = json.load(open('models/inference_config.json'))
g = ic.get('guardian', {})
sz = ic.get('sizing', {})
rr = ic.get('rr_gate', {})
hmm = ic.get('hmm', {})
cascade = ic.get('cascade', {})
inf_py = Path('app/services/inference.py').read_text(encoding='utf-8')

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

out = {
    'model_version': ic.get('model_version'),
    'model_type': ic.get('model_type'),
    'guardian_min_hold_bars': g.get('min_hold_bars'),
    'guardian_exit_threshold': g.get('exit_threshold'),
    'guardian_model_file': ic.get('models', {}).get('guardian'),
    'guardian_features_n': len(json.load(open('models/guardian_feature_cols.json'))),
    'lgbm_features_n': len(json.load(open('models/feature_cols_v2.json'))),
    'sl_trigger_mode': rr.get('sl_trigger_mode'),
    'sizing_mode': sz.get('mode', 'fixed'),
    'has_sizing_dynamic_block': 'dynamic' in sz,
    'modal_per_trade': ic.get('risk', {}).get('modal_per_trade'),
    'leverage': ic.get('risk', {}).get('leverage_recommended'),
    'hmm_per_state_thresholds': hmm.get('per_state_thresholds'),
    'cascade_mode': cascade.get('mode'),
    'lstm_fusion_mode': cascade.get('lstm_fusion_mode'),
    'lstm_opposite_pen': cascade.get('lstm_adjust_opposite_pen'),
    'resolve_hmm_thresholds_in_inference': 'resolve_hmm_thresholds' in inf_py,
    'model_files': {
        'guardian_best.pkl': {'size': Path('models/guardian_best.pkl').stat().st_size,
                              'mtime': Path('models/guardian_best.pkl').stat().st_mtime},
        'lgbm_baseline.pkl': {'size': Path('models/lgbm_baseline.pkl').stat().st_size},
    },
    'git_head': Path('.git/refs/heads/main').read_text().strip() if Path('.git/refs/heads/main').exists() else None,
}
print(json.dumps(out))
"""

def main():
    import base64
    b64 = base64.b64encode(REMOTE.encode()).decode()
    remote_cmd = f"python3 -c \"import base64; exec(base64.b64decode('{b64}').decode())\""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", HOST, remote_cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print("SSH error:", r.stderr or r.stdout)
        return 1
    line = r.stdout.strip().splitlines()[-1]
    data = json.loads(line)
    print(json.dumps(data, indent=2))

    checks = []
    checks.append(("model_version", data["model_version"] == "ic32_b_dir_combined"))
    checks.append(("guardian_min_hold=4", data["guardian_min_hold_bars"] == 4))
    checks.append(("guardian_exit=0.65", data["guardian_exit_threshold"] == 0.65))
    checks.append(("sl_trigger=close", data["sl_trigger_mode"] == "close"))
    checks.append(("sizing=fixed", data["sizing_mode"] == "fixed" and not data["has_sizing_dynamic_block"]))
    checks.append(("hard_consensus", data["cascade_mode"] == "hard_consensus"))
    checks.append(("hmm_resolve_patched", data["resolve_hmm_thresholds_in_inference"]))
    checks.append(("guardian_32f", data["guardian_features_n"] == 32))
    checks.append(("lgbm_33f", data["lgbm_features_n"] == 33))

    print("\n--- Deploy checklist ---")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
        if not ok:
            all_ok = False
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())