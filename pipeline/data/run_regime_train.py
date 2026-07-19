"""CLI: HMM regime labels for training (OOF walk-forward)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline._wrapper import run_relocated

if __name__ == "__main__":
    run_relocated("pipeline/data/core/regime_hmm.py")