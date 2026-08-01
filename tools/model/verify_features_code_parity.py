"""Gerbang anti-drift: kode fitur RISET vs kode fitur LIVE, dibandingkan per NILAI.

MASALAH YANG DIJAGA
-------------------
`engineer_features` hidup di DUA berkas terpisah yang disinkronkan MANUAL:

    riset : Riset_pemodelan/core/features.py
    live  : live_dualbin_ft/app/core/features.py            (dualbin)

Setiap fitur dihitung dua kali. Ada ~48 tempat di mana keduanya bisa diam-diam berbeda,
dan tidak ada yang memeriksanya. Itu bukan risiko teoretis — biayanya sudah terukur:

    2026-07-12  live diubah agar memakai data posisi ASLI; training tetap isian buatan.
                Tidak ada yang membandingkan. Selama ~3 minggu uang riil berjalan di
                PF 1,220 alih-alih 1,576 — -55,7% PnL, drawdown 52% lebih dalam.
    2026-08-01  saat MENYINKRONKAN kedua berkas secara sadar, saya tetap menyisipkan bug
                (`np.nan` skalar alih-alih Series) yang bikin siklus trading mati dgn
                posisi terbuka. Ditangkap test suite, bukan oleh mata.

Kesimpulannya: kehati-hatian manusia bukan penjaga yang memadai. Ini penjaganya.

CARA KERJA
----------
Memuat KEDUA implementasi sbg modul terpisah dalam satu proses, menjalankan keduanya atas
input clean parquet yang SAMA, lalu membandingkan SETIAP kolom yang ada di keduanya —
bukan hanya fitur yang kebetulan dipakai model saat ini (fitur yang belum dipakai hari ini
bisa dipakai besok, dan drift-nya sudah terlanjur menumpuk).

Yang diperiksa per kolom:
  * pola KOSONG (NaN) harus sama persis — beda pola = beda perilaku walau nilai cocok
  * selisih absolut maksimum harus < toleransi (default 1e-9, praktis identik)
  * kolom yang hanya ada di salah satu sisi dilaporkan sbg TIMPANG

Keluar dgn kode 1 bila ada beda apa pun -> bisa dipasang di CI / pre-deploy.

PAKAI
-----
    python tools/model/verify_features_code_parity.py                 # 18 koin, cepat
    python tools/model/verify_features_code_parity.py --penuh         # seluruh histori
    python tools/model/verify_features_code_parity.py --coins BTCUSDT ETHUSDT
    python tools/model/verify_features_code_parity.py --live-path <path lain>

CATATAN: ini memeriksa KODE vs KODE. Ia TIDAK menjamin data yang diterima live di VPS
sama dgn data riset — untuk itu pakai `tools/model/audit_feature_value_parity.py` dan
`tools/ops/compare_oos_live_signals.py`. Keduanya saling melengkapi, bukan menggantikan.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import PROC_DIR  # noqa: E402

LIVE_DEFAULT = ROOT.parent / "live_dualbin_ft/.worktrees/real-money-execution/app/core/features.py"
COINS_DEFAULT = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
                 "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT", "ATOMUSDT", "UNIUSDT",
                 "NEARUSDT", "FILUSDT", "ETCUSDT", "BCHUSDT"]
BAR_CEPAT = 6000            # cukup panjang utk mengisi seluruh rolling window terpanjang (168)


def muat_live(path: Path):
    if not path.exists():
        raise SystemExit(f"berkas live tidak ditemukan: {path}")
    spec = importlib.util.spec_from_file_location("_live_features", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_live_features"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "engineer_features"):
        raise SystemExit(f"{path} tidak punya engineer_features()")
    return mod.engineer_features


def banding(a: pd.DataFrame, b: pd.DataFrame, tol: float) -> tuple[list[dict], list[str]]:
    beda, timpang = [], []
    for k in sorted(set(a.columns) | set(b.columns)):
        if k not in a.columns or k not in b.columns:
            timpang.append(f"{k} (hanya di {'riset' if k in a.columns else 'live'})")
            continue
        x, y = a[k], b[k]
        if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            if not x.astype(str).equals(y.astype(str)):
                beda.append({"fitur": k, "sebab": "nilai non-numerik beda", "maks": None})
            continue
        x, y = x.astype(float), y.astype(float)
        pola = int((x.isna() != y.isna()).sum())
        ok = x.notna() & y.notna()
        maks = float((x[ok] - y[ok]).abs().max()) if ok.any() else 0.0
        if pola or (maks > tol):
            beda.append({"fitur": k,
                         "sebab": f"pola kosong beda di {pola} bar" if pola else "nilai beda",
                         "maks": maks})
    return beda, timpang


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS_DEFAULT)
    ap.add_argument("--live-path", default=str(LIVE_DEFAULT))
    ap.add_argument("--penuh", action="store_true", help="pakai seluruh histori (lambat)")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    live_path = Path(args.live_path)
    from core.features import engineer_features as riset_fn
    live_fn = muat_live(live_path)

    print(f"riset : {ROOT / 'core/features.py'}")
    print(f"live  : {live_path}")
    print(f"mode  : {'PENUH' if args.penuh else f'CEPAT ({BAR_CEPAT} bar terakhir)'} | "
          f"toleransi {args.tol:g} | {len(args.coins)} koin\n")

    total_beda, total_timpang, gagal_koin = 0, 0, []
    for i, coin in enumerate(args.coins):
        p = Path(PROC_DIR) / f"{coin}_clean.parquet"
        if not p.exists():
            print(f"{coin:10s} SKIP — clean parquet tidak ada")
            continue
        df = pd.read_parquet(p)
        if not args.penuh:
            df = df.tail(BAR_CEPAT)
        a = riset_fn(df, coin, i)
        b = live_fn(df, coin, i)
        beda, timpang = banding(a, b, args.tol)
        total_beda += len(beda)
        total_timpang += len(timpang)
        if beda or timpang:
            gagal_koin.append(coin)
            print(f"{coin:10s} DRIFT — {len(beda)} fitur beda, {len(timpang)} timpang")
            for d in beda[:10]:
                m = "-" if d["maks"] is None else f"{d['maks']:.3e}"
                print(f"           {d['fitur']:34s} {d['sebab']:28s} maks {m}")
            for t in timpang[:10]:
                print(f"           TIMPANG: {t}")
        else:
            print(f"{coin:10s} ok — {len(set(a.columns) & set(b.columns))} fitur identik")

    print()
    if total_beda or total_timpang:
        print(f"GAGAL: drift terdeteksi di {len(gagal_koin)} koin "
              f"({total_beda} fitur beda, {total_timpang} timpang)")
        print("Riset & live berpisah. Cari sebabnya — JANGAN longgarkan toleransi.")
        return 1
    print(f"LULUS: kode riset & live menghasilkan nilai identik di {len(args.coins)} koin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
