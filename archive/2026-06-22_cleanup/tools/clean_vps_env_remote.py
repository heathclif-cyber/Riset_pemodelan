#!/usr/bin/env python3
"""Run on VPS: strip LIVE_* risk keys from .env."""
import pathlib

path = pathlib.Path("/home/swint/swint_tradev2/.env")
text = path.read_text(encoding="utf-8")
lines = text.splitlines()
DROP = {"LIVE_MAX_POSITIONS", "LIVE_DAILY_LOSS_LIMIT", "LIVE_MODAL_PER_TRADE", "LIVE_LEVERAGE"}
out = []
replaced_risk_hdr = False
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        if "Risk management live" in line and not replaced_risk_hdr:
            out.append("# -- Risk / sizing: lihat models/inference_config.json (UI /models) --")
            out.append("# JANGAN set LIVE_* di .env -- drift vs riset & UI tidak berlaku.")
            replaced_risk_hdr = True
            continue
        if "Risk management live" in line:
            continue
        out.append(line)
        continue
    key = stripped.split("=", 1)[0].strip()
    if key in DROP:
        continue
    out.append(line)
if not replaced_risk_hdr:
    for i, ln in enumerate(out):
        if "BINANCE_API_SECRET" in ln:
            out.insert(i + 1, "")
            out.insert(i + 2, "# -- Risk / sizing: lihat models/inference_config.json (UI /models) --")
            out.insert(i + 3, "# JANGAN set LIVE_* di .env -- drift vs riset & UI tidak berlaku.")
            break
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("OK: .env cleaned")