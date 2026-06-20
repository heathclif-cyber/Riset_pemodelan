"""Patch semua holdout scripts: tambah apply_training_parity import + call."""
import re
from pathlib import Path

FILES = [
    "07b_holdout_ic32_guardian_variant_diag.py",
    "07c_holdout_ic32_lstm_fusion_diag.py",
    "07d_holdout_ic32_guardian_min_hold_diag.py",
    "07e_holdout_ic32_dual_lstm_diag.py",
    "07f_holdout_ic32_dynsize_diag.py",
    "07g_holdout_ic32_pyramiding_diag.py",
    "07h_holdout_ic32_scale_in_diag.py",
]

PIPELINE_DIR = Path(__file__).parent
IMPORT_OLD = "from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array"
IMPORT_NEW = "from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array, apply_training_parity"

patched = 0

for fname in FILES:
    fp = PIPELINE_DIR / fname
    if not fp.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    text = fp.read_text(encoding="utf-8")

    if "apply_training_parity(df)" in text:
        print(f"  ALREADY PATCHED: {fname}")
        continue

    if IMPORT_OLD not in text:
        print(f"  SKIP (import not found): {fname}")
        continue

    # 1. Patch import
    text = text.replace(IMPORT_OLD, IMPORT_NEW)

    # 2. Find X = np.zeros AFTER first pd.read_parquet(p) call
    # Pattern: "df = pd.read_parquet(p)" sets parquet_seen, then find "X = np.zeros"
    lines = text.split("\n")
    new_lines = []
    parquet_seen = False
    inserted = False

    for line in lines:
        # The features parquet is loaded via `df = pd.read_parquet(p)` where p = f"{sym}_features_v3.parquet"
        # The line itself won't contain "features_v3" — look for `= pd.read_parquet(p)`
        if re.match(r'\s+df\s*=\s*pd\.read_parquet\(p\)', line):
            parquet_seen = True

        if parquet_seen and not inserted and re.match(r'\s+X\s*=\s*np\.zeros', line):
            indent = " " * (len(line) - len(line.lstrip()))
            new_lines.append(f"{indent}# Apply same training parity overrides seperti live inference")
            new_lines.append(f"{indent}df = apply_training_parity(df)")
            new_lines.append("")
            inserted = True

        new_lines.append(line)

    if not inserted:
        print(f"  WARN (insert point not found): {fname}")
        # Print debug info
        for i, line in enumerate(lines):
            if "read_parquet" in line or "X = np" in line:
                print(f"    L{i+1}: {line!r}")
        continue

    fp.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"  PATCHED: {fname}")
    patched += 1

print(f"\nDone: {patched} patched")
