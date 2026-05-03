"""
run_pipeline.py — Entry point untuk menjalankan semua fase pipeline

Arsitektur: Hierarchical Cascade (H4 LGBM → H1 LGBM → LSTM)

Urutan fase:
  01_fetch.py         → Fetch klines + funding dari Binance
  02_clean.py         → Clean + alignment multi-TF ke H1 grid
  03_engineer.py      → Feature engineering + swing labeling
  04_train_lgbm_h4.py → H4 LGBM regime filter (bias gate)
  05_train_lgbm_h1.py → H1 LGBM entry signal generator
  06_train_lstm.py    → LSTM momentum confirmation
  07_evaluate.py      → Multi-model evaluation (H4 + H1 + cascade)
  08_backtest.py      → Walk-forward backtest (hierarchical cascade)
  09_holdout_backtest → Hold-out backtest (genuine OOS)

Contoh penggunaan:
  python run_pipeline.py --fetch                 # fetch training coins
  python run_pipeline.py --clean                 # clean training coins
  python run_pipeline.py --engineer              # feature engineering
  python run_pipeline.py --train-h4              # latih H4 LGBM saja
  python run_pipeline.py --train                 # latih H4 + H1 + LSTM
  python run_pipeline.py --evaluate              # evaluasi multi-model
  python run_pipeline.py --backtest              # walk-forward backtest
  python run_pipeline.py --all                   # semua fase (01-08) sekaligus
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
    parser = argparse.ArgumentParser(description="Trading ML Pipeline — Hierarchical Cascade")
    parser.add_argument("--fetch",     action="store_true", help="Fase 01: Fetch data dari Binance")
    parser.add_argument("--clean",     action="store_true", help="Fase 02: Clean + align multi-TF")
    parser.add_argument("--engineer",  action="store_true", help="Fase 03: Feature engineering")
    parser.add_argument("--train-h4",  action="store_true", help="Fase 04: Latih H4 LGBM (regime filter)")
    parser.add_argument("--train",     action="store_true", help="Fase 04-06: Latih semua model (H4+H1+LSTM)")
    parser.add_argument("--evaluate",  action="store_true", help="Fase 07: Multi-model evaluation + SHAP")
    parser.add_argument("--backtest",  action="store_true", help="Fase 08: Walk-forward backtest")
    parser.add_argument("--holdout",   action="store_true", help="Fase 09: Hold-out backtest (OOS)")
    parser.add_argument("--all",       action="store_true", help="Semua fase (01-08)")
    parser.add_argument("--all-coins", action="store_true", help="Jalankan untuk semua 20 koin")
    parser.add_argument("--run-id",    default=None,        help="Run ID untuk folder output")
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

    # -- Fase 03: Engineer ----------------------------------------------------
    if do_all or args.engineer:
        run(["pipeline/03_engineer.py"] + coin_flag)

    # -- Fase 04: H4 LGBM (regime filter) ------------------------------------
    if do_all or args.train or args.train_h4:
        run(["pipeline/04_train_lgbm_h4.py"] + coin_flag + run_flag)

    # -- Fase 05: H1 LGBM (entry signal) -------------------------------------
    if do_all or args.train:
        run(["pipeline/05_train_lgbm_h1.py"] + coin_flag + run_flag)

    # -- Fase 06: LSTM (confirmation) -----------------------------------------
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

    if not any([do_all, args.fetch, args.clean, args.engineer,
                args.train_h4, args.train, args.evaluate,
                args.backtest, args.holdout]):
        print("Tidak ada fase yang dipilih. Gunakan --help untuk melihat opsi.")


if __name__ == "__main__":
    main()
