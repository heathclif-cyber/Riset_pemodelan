# -*- coding: utf-8 -*-
"""Verifikasi model + kode di VPS."""
import json
import subprocess
import sys

REMOTE = r"""
import json, joblib, os
os.chdir('/home/swint/swint_tradev2')
lgbm = joblib.load('models/lgbm_baseline.pkl')
fc = json.load(open('models/feature_cols_v2.json'))
gf = json.load(open('models/guardian_feature_cols.json'))
ic = json.load(open('models/inference_config.json'))
out = {
    'lgbm_n': len(lgbm.feature_name_),
    'fc_n': len(fc),
    'gf_n': len(gf),
    'version': ic.get('model_version'),
    'sl_trigger': ic.get('rr_gate', {}).get('sl_trigger_mode'),
    'hmm_s3': ic.get('hmm', {}).get('per_state_thresholds', {}).get('3'),
    'modal': ic.get('risk', {}).get('modal_per_trade'),
    'cascade_mode': ic.get('cascade', {}).get('mode'),
    'resolve_hmm_in_inference': 'resolve_hmm_thresholds' in open('app/services/inference.py').read(),
}
print(json.dumps(out))
"""

def main():
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "root@139.180.157.176", "python3", "-c", REMOTE],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    data = json.loads(r.stdout.strip().split("\n")[-1])
    print(json.dumps(data, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())