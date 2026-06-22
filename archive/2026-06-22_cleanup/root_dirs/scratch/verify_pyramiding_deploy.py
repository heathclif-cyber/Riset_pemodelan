# -*- coding: utf-8 -*-
import json
import subprocess

HOST = "root@139.180.157.176"
REMOTE = r"""
import json
from pathlib import Path
root = Path('/home/swint/swint_tradev2')
ic = json.load(open(root / 'models/inference_config.json'))
pyr = ic.get('pyramiding', {})
pt = (root / 'app/services/paper_trading.py').read_text(encoding='utf-8')
out = {
    'git_head': (root / '.git/refs/heads/main').read_text().strip(),
    'pyramiding': pyr,
    'guardian_min_hold': ic.get('guardian', {}).get('min_hold_bars'),
    'sl_trigger_mode': ic.get('rr_gate', {}).get('sl_trigger_mode'),
    'scale_in_code': 'scale_in' in pt and '_apply_scale_in' in pt,
    'service_hint': 'check journalctl separately',
}
print(json.dumps(out))
"""


def main():
    import base64
    b64 = base64.b64encode(REMOTE.encode()).decode()
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", HOST,
        f"python3 -c \"import base64; exec(base64.b64decode('{b64}').decode())\"",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    data = json.loads(r.stdout.strip().splitlines()[-1])
    print(json.dumps(data, indent=2))
    ok = (
        data["pyramiding"].get("enabled") is True
        and data["pyramiding"].get("max_positions_per_coin") == 2
        and data["pyramiding"].get("exit_mode") == "scale_in"
        and data.get("scale_in_code") is True
        and data["guardian_min_hold"] == 4
    )
    print("\nDEPLOY OK" if ok else "\nDEPLOY CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())