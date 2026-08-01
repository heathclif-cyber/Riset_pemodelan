"""Gerbang anti-kebocoran: apakah fitur di bar T memakai informasi SETELAH T?

CARA UJI (tegas, bukan penalaran)
---------------------------------
AWAL data DIKUNCI, hanya UJUNG yang berbeda. Hitung fitur dua kali:

    A. dari `data[0 : T]`        -> yang tersedia di live saat memutuskan di bar T
    B. dari `data[0 : T + k]`    -> sama persis, PLUS k bar masa depan

Lalu bandingkan nilai pada bar-bar < T. Riwayat di belakangnya IDENTIK; satu-satunya beda
adalah keberadaan data setelah T. Kalau ada nilai yang berubah, fitur itu MENGINTIP KE
DEPAN — model belajar dari angka yang tak akan pernah dilihatnya di live.

KENAPA AWALNYA HARUS DIKUNCI (kesalahan rancangan versi pertama, 2026-08-01):
versi pertama membandingkan potongan `[T-N : T]` melawan data PENUH — mengubah awal DAN
ujung sekaligus. Hasilnya 5-7 fitur "bocor" dgn selisih 1e-4..1e-7: `ema_200_h4`, `cvd`,
`relative_strength_*` dan turunannya. Itu BUKAN kebocoran, melainkan ketergantungan pada
TITIK AWAL — EMA punya memori tak terbatas, CVD kumulatif sejak bar pertama. Uji yang
mengubah dua hal sekaligus tidak bisa membedakan "butuh histori lebih panjang" dari
"mengintip masa depan". Dgn awal dikunci, seluruh kelas selisih itu hilang dgn sendirinya.

Satu uji ini menangkap SEMUA bentuk kebocoran ke depan tanpa perlu menebak bentuknya:
  * rolling window yang mengintip ke depan (center=True, shift(-N))
  * salah arah geser saat resample H4/D1
  * normalisasi GLOBAL (z-score / min-max atas seluruh seri) — kesalahan klasik yang
    tidak terlihat di kode karena tampak seperti penskalaan biasa
  * ffill/bfill yang menarik nilai dari bar sesudahnya

CATATAN: `label` SENGAJA melihat ke depan (itu targetnya, bukan fitur) — dikecualikan.

PAKAI
-----
    python tools/model/verify_no_lookahead.py
    python tools/model/verify_no_lookahead.py --coins BTCUSDT ETHUSDT --titik 6
Keluar 1 bila ada fitur bocor -> bisa dipasang di CI / pre-deploy.
"""
from __future__ import annotations

import argparse
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
from core.features import engineer_features  # noqa: E402

COINS_DEFAULT = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "ATOMUSDT"]
# Kolom target & turunannya — melihat ke depan itu memang tugasnya.
KECUALI = {"label", "label_v3", "tp_price", "sl_price", "rr", "hold_bars",
           "exit_reason", "future_ret", "fwd_ret"}
# Panjang data yang dipakai tiap potongan. Harus jauh lebih besar dari rolling terpanjang
# (168 bar) supaya beda yang muncul benar-benar kebocoran, bukan warm-up.
JENDELA = 5000
# Bar warm-up ada di AWAL potongan (belum punya histori sebelumnya), BUKAN di akhir.
# Yang dibandingkan justru bar-bar TERAKHIR potongan — itu yang mewakili keputusan live
# (bar terbaru, dihitung dari data yang tersedia saat itu). Bar awal dibuang.
MASA_DEPAN = 500        # berapa bar "masa depan" ditambahkan di versi B
BANDING_EKOR = 300      # bandingkan sekian bar terakhir SEBELUM titik potong


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS_DEFAULT)
    ap.add_argument("--titik", type=int, default=4, help="berapa titik potong per koin")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--banding-ekor", type=int, default=BANDING_EKOR)
    args = ap.parse_args()

    print("Uji look-ahead: data[0:T] vs data[0:T+k] — AWAL DIKUNCI, hanya ujung berbeda")
    print(f"{args.titik} titik potong/koin | +{MASA_DEPAN} bar masa depan | "
          f"toleransi {args.tol:g}")
    print(f"dibandingkan: {args.banding_ekor} bar terakhir SEBELUM titik potong\n")

    bocor_total: dict[str, float] = {}
    for i, coin in enumerate(args.coins):
        p = Path(PROC_DIR) / f"{coin}_clean.parquet"
        if not p.exists():
            print(f"{coin:10s} SKIP — clean parquet tidak ada")
            continue
        full = pd.read_parquet(p)
        n = len(full)
        if n < JENDELA + 1000:
            print(f"{coin:10s} SKIP — data terlalu pendek ({n})")
            continue

        # Titik potong tersebar merata di paruh akhir data.
        titik = np.linspace(int(n * 0.55), n - MASA_DEPAN - 10, args.titik).astype(int)
        bocor_koin: dict[str, float] = {}
        for t in titik:
            # AWAL DIKUNCI di 0 — hanya ujung yang berbeda. Riwayat identik, jadi
            # ketergantungan pada titik awal (EMA, cumsum) ter-cancel dgn sendirinya.
            f_tanpa = engineer_features(full.iloc[:t], coin, i)
            f_dengan = engineer_features(full.iloc[:t + MASA_DEPAN], coin, i)
            # Bandingkan bar TERAKHIR sebelum T — paling mungkin tercemar kalau ada
            # rolling/resample yang mengintip ke depan.
            idx = f_tanpa.index[-args.banding_ekor:]
            idx = idx.intersection(f_dengan.index)
            f_penuh = f_dengan
            f_potong = f_tanpa
            if len(idx) == 0:
                continue
            for k in sorted(set(f_potong.columns) & set(f_penuh.columns) - KECUALI):
                x, y = f_potong.loc[idx, k], f_penuh.loc[idx, k]
                if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
                    continue
                x, y = x.astype(float), y.astype(float)
                if int((x.isna() != y.isna()).sum()):
                    bocor_koin[k] = max(bocor_koin.get(k, 0.0), float("inf"))
                    continue
                ok = x.notna()
                if ok.any():
                    d = float((x[ok] - y[ok]).abs().max())
                    if d > args.tol:
                        bocor_koin[k] = max(bocor_koin.get(k, 0.0), d)

        if bocor_koin:
            print(f"{coin:10s} BOCOR — {len(bocor_koin)} fitur")
            for k, d in sorted(bocor_koin.items(), key=lambda kv: -kv[1])[:12]:
                m = "pola kosong beda" if d == float("inf") else f"maks {d:.3e}"
                print(f"           {k:34s} {m}")
            for k, d in bocor_koin.items():
                bocor_total[k] = max(bocor_total.get(k, 0.0), d)
        else:
            print(f"{coin:10s} bersih — {len(set(f_penuh.columns) - KECUALI)} fitur causal")

    print()
    if bocor_total:
        print(f"GAGAL: {len(bocor_total)} fitur memakai informasi masa depan.")
        print("Model belajar dari nilai yang TIDAK akan dilihatnya di live.")
        return 1
    print("LULUS: tidak ada fitur yang nilainya berubah ketika data masa depan dibuang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
