"""
run_pipeline.py — Entry point untuk menjalankan semua fase pipeline

Contoh penggunaan:
  python run_pipeline.py --fetch                 # fetch training coins
  python run_pipeline.py --fetch --new           # fetch new coins
  python run_pipeline.py --clean                 # clean training coins
  python run_pipeline.py --engineer              # feature engineering
  python run_pipeline.py --train                 # train lgbm + lstm + ensemble
  python run_pipeline.py --all                   # semua fase sekaligus
  python run_pipeline.py --fetch --clean --engineer --all-coins   # pipeline lengkap semua koin
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
    parser = argparse.ArgumentParser(description="Trading ML Pipeline")
    parser.add_argument("--fetch",    action="store_true", help="Fase 1: Fetch data")
    parser.add_argument("--clean",    action="store_true", help="Fase 2: Clean data")
    parser.add_argument("--engineer", action="store_true", help="Fase 3: Feature engineering")
    parser.add_argument("--train",    action="store_true", help="Fase 4-6: Training")
    parser.add_argument("--evaluate", action="store_true", help="Fase 7: SHAP + evaluasi")
    parser.add_argument("--all",      action="store_true", help="Semua fase")
    parser.add_argument("--new-coins",  action="store_true", help="Jalankan untuk new coins (15 koin)")
    parser.add_argument("--all-coins",  action="store_true", help="Jalankan untuk semua 20 koin")
    parser.add_argument("--run-id",   default=None, help="Run ID untuk output folder")
    return parser.parse_args()


def main():
    args = parse_args()

    coin_flag = "--all" if args.all_coins else ("--new" if args.new_coins else "")
    run_flag  = ["--run-id", args.run_id] if args.run_id else []

    do_all = args.all

    if do_all or args.fetch:
        cmd = ["pipeline/01_fetch.py"]
        if coin_flag:
            cmd.append(coin_flag)
        run(cmd)

    if do_all or args.clean:
        cmd = ["pipeline/02_clean.py"]
        if coin_flag:
            cmd.append(coin_flag)
        run(cmd)

    if do_all or args.engineer:
        cmd = ["pipeline/03_engineer.py"]
        if coin_flag:
            cmd.append(coin_flag)
        run(cmd)

    if do_all or args.train:
        for script in ["04_train_lgbm.py", "05_train_lstm.py", "06_ensemble.py"]:
            cmd = [f"pipeline/{script}"] + run_flag
            if args.all_coins:
                cmd.append("--all")
            run(cmd)

    if do_all or args.evaluate:
        run(["pipeline/07_evaluate.py"] + run_flag)

    if not any([do_all, args.fetch, args.clean, args.engineer, args.train, args.evaluate]):
        print("Gunakan --help untuk melihat opsi yang tersedia.")


if __name__ == "__main__":
    main()
