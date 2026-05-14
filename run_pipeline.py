"""
run_pipeline.py — Entry point untuk menjalankan semua fase pipeline

Arsitektur: 2-Model Cascade (LGBM → LSTM)

Urutan fase:
  01_fetch.py          → Fetch klines + funding dari Binance
  02_clean.py          → Clean + alignment multi-TF ke H1 grid
  03_analyze_swing.py  → [OPSIONAL] Grid search parameter swing labeling
  03_engineer.py       → Feature engineering + swing labeling
  05_train_lgbm.py     → LGBM entry signal (primary model)
  06_train_lstm.py     → LSTM confirmation
  07_evaluate.py       → Evaluation + SHAP analysis
  08_backtest.py       → Walk-forward backtest
  09_holdout_backtest  → Hold-out backtest (genuine OOS)

Alur tuning swing (sebelum retrain):
  python run_pipeline.py --clean
  python run_pipeline.py --analyze-swing         # temukan parameter terbaik
  # → update config.py dengan rekomendasi
  python run_pipeline.py --engineer
  python run_pipeline.py --train

Contoh penggunaan:
  python run_pipeline.py --fetch                 # fetch training coins
  python run_pipeline.py --clean                 # clean training coins
  python run_pipeline.py --analyze-swing         # tuning swing parameters
  python run_pipeline.py --engineer              # feature engineering
  python run_pipeline.py --train                 # latih LGBM + LSTM
  python run_pipeline.py --evaluate              # evaluasi + SHAP
  python run_pipeline.py --backtest              # walk-forward backtest
  python run_pipeline.py --all                   # semua fase (01-08, skip 04)
  python run_pipeline.py --fetch --clean --engineer --train --all-coins
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run([sys.executable] + cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"ERROR: command gagal dengan exit code {result.returncode}")
        sys.exit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="Trading ML Pipeline — 2-Model Cascade (LGBM + LSTM)")
    parser.add_argument("--fetch",          action="store_true", help="Fase 01: Fetch data dari Binance")
    parser.add_argument("--clean",          action="store_true", help="Fase 02: Clean + align multi-TF")
    parser.add_argument("--analyze-swing",  action="store_true", help="Fase 04: Grid search swing parameters (opsional, sebelum engineer)")
    parser.add_argument("--engineer",       action="store_true", help="Fase 03: Feature engineering + labeling")
    parser.add_argument("--train",          action="store_true", help="Fase 05-06: Latih LGBM + LSTM")
    parser.add_argument("--evaluate",       action="store_true", help="Fase 07: Evaluation + SHAP")
    parser.add_argument("--backtest",       action="store_true", help="Fase 08: Walk-forward backtest")
    parser.add_argument("--holdout",        action="store_true", help="Fase 09: Hold-out backtest (OOS)")
    parser.add_argument("--guardian",       action="store_true", help="Fase 15: Train exit guardian model")
    parser.add_argument("--all",            action="store_true", help="Semua fase wajib (01-03, 05-08) — skip 04")
    parser.add_argument("--all-coins",      action="store_true", help="Jalankan untuk semua koin")
    parser.add_argument("--run-id",         default=None,        help="Run ID untuk folder output")
    return parser.parse_args()


def main():
    args = parse_args()

    coin_flag = ["--all"] if args.all_coins else []
    run_flag  = ["--run-id", args.run_id] if args.run_id else []
    do_all    = args.all

    # -- Fase 01: Fetch -------------------------------------------------------
    if do_all or args.fetch:
        run(["pipeline/01_fetch.py"] + coin_flag)

    # -- Fase 02: Clean -------------------------------------------------------
    if do_all or args.clean:
        run(["pipeline/02_clean.py"] + coin_flag)

    # -- Fase 04: Analyze Swing (opsional — tidak masuk --all) ----------------
    # Jalankan setelah clean, sebelum engineer untuk tuning parameter labeling.
    # Output: rekomendasi SWING_H4_LOOKBACK, SWING_LABEL_MIN_RR, dll.
    if args.analyze_swing:
        run(["pipeline/03_analyze_swing.py"] + coin_flag)

    # -- Fase 03: Engineer ----------------------------------------------------
    if do_all or args.engineer:
        run(["pipeline/04_engineer.py"] + coin_flag)

    # -- Fase 05: LGBM entry signal (primary model) --------------------------
    if do_all or args.train:
        run(["pipeline/05_train_lgbm.py"] + coin_flag + run_flag)

    # -- Fase 06: LSTM confirmation -------------------------------------------
    if do_all or args.train:
        run(["pipeline/06_train_lstm.py"] + coin_flag + run_flag)

    # -- Fase 07: Evaluate ----------------------------------------------------
    if do_all or args.evaluate:
        run(["pipeline/07_evaluate.py"] + run_flag)

    # -- Fase 08: Backtest ----------------------------------------------------
    if do_all or args.backtest:
        run(["pipeline/08_backtest.py"] + coin_flag + run_flag)

    # -- Fase 09: Hold-out (tidak masuk --all karena butuh data baru) ---------
    if args.holdout:
        run(["pipeline/09_holdout_backtest.py"] + coin_flag + run_flag)

    # -- Fase 15: Guardian Training (tidak masuk --all, experimental) ----------
    if args.guardian:
        run(["pipeline/15_train_guardian.py"] + coin_flag + run_flag)

    if not any([do_all, args.fetch, args.clean, args.analyze_swing,
                args.engineer, args.train, args.evaluate,
                args.backtest, args.holdout, args.guardian]):
        print("Tidak ada fase yang dipilih. Gunakan --help untuk melihat opsi.")


if __name__ == "__main__":
    main()
