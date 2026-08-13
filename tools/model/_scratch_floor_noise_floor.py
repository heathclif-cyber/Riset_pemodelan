"""Review Opus 2026-08-13: ukur LANTAI DERAU utk klaim "floor OFF > floor 0.7" di OOS.

Pertanyaan terbuka #1 dari EXPERIMENTS.md 2026-08-13: selisih OOS PF +0,047
(1,343 -> 1,390) di n=329 -- di atas derau atau bukan? Sesuai disiplin
`feedback-ukur-lantai-derau-sebelum-menafsir` (dualbin: PF +-0,0206, PnL +-8,1%),
angka ini WAJIB diukur sebelum ditafsir.

Dua uji:
  A. BERPASANGAN (paling kuat) -- trade dicocokkan per (coin, entry_time). Entry
     identik, yang beda cuma aturan exit. Bootstrap selisih PnL per-trade.
  B. TIDAK BERPASANGAN -- bootstrap PF tiap varian sendiri-sendiri, lihat overlap
     selang kepercayaan. Ini yang menjawab "berapa lantai derau PF utk n=329".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

A_PATH = Path("data/live_cache/oos_floorfrac07_trades_detail.csv")
B_PATH = Path("data/live_cache/oos_floorOFF_trades_detail.csv")
N_BOOT = 20000
SEED = 42


def _pf(pnl: np.ndarray) -> float:
    win = pnl[pnl > 0].sum()
    loss = -pnl[pnl < 0].sum()
    return float(win / loss) if loss > 0 else float("inf")


def main() -> int:
    rng = np.random.default_rng(SEED)
    a = pd.read_csv(A_PATH)
    b = pd.read_csv(B_PATH)
    for df in (a, b):
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")

    print(f"floor 0.7 : {len(a)} trade | PF {_pf(a['net_pnl'].to_numpy()):.4f} | PnL ${a['net_pnl'].sum():.2f}")
    print(f"floor OFF : {len(b)} trade | PF {_pf(b['net_pnl'].to_numpy()):.4f} | PnL ${b['net_pnl'].sum():.2f}")

    key = ["coin", "entry_time"]
    m = a.merge(b, on=key, how="inner", suffixes=("_a", "_b"))
    only_a = len(a) - len(m)
    only_b = len(b) - len(m)
    print(f"\nCocok berpasangan: {len(m)} | cuma di 0.7: {only_a} | cuma di OFF: {only_b}")

    # ── A. Uji berpasangan ────────────────────────────────────────────────────
    d = (m["net_pnl_b"] - m["net_pnl_a"]).to_numpy(float)
    n_changed = int((np.abs(d) > 1e-9).sum())
    print(f"\n{'='*74}\n  A. BERPASANGAN (n={len(d)}, trade yang exit-nya BERUBAH: {n_changed})\n{'-'*74}")
    print(f"  Total selisih PnL (OFF - 0.7) pada trade cocok : ${d.sum():+.2f}")
    print(f"  Rata-rata per trade                            : ${d.mean():+.4f}")

    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"  CI95 rata-rata selisih : [${lo:+.4f}, ${hi:+.4f}]")
    print(f"  p (dua sisi)           : {p_two:.4f}")
    print(f"  -> {'SIGNIFIKAN (CI tidak melewati nol)' if lo > 0 or hi < 0 else 'TIDAK SIGNIFIKAN (CI melewati nol)'}")

    if n_changed:
        dc = d[np.abs(d) > 1e-9]
        print(f"\n  Hanya pada {n_changed} trade yang exit-nya berubah:")
        print(f"    lebih baik: {(dc > 0).sum()} | lebih buruk: {(dc < 0).sum()}")
        print(f"    total ${dc.sum():+.2f} | median ${np.median(dc):+.4f}")
        print(f"    3 perbaikan terbesar : {np.sort(dc)[::-1][:3].round(2)}")
        print(f"    3 pemburukan terbesar: {np.sort(dc)[:3].round(2)}")

    # ── B. Lantai derau PF (tidak berpasangan) ────────────────────────────────
    print(f"\n{'='*74}\n  B. LANTAI DERAU PF (bootstrap tiap varian, n masing-masing)\n{'-'*74}")
    for name, df in (("floor 0.7", a), ("floor OFF", b)):
        p = df["net_pnl"].to_numpy(float)
        bs = np.array([_pf(p[rng.integers(0, len(p), len(p))]) for _ in range(N_BOOT)])
        bs = bs[np.isfinite(bs)]
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        print(f"  {name}: PF {_pf(p):.4f}  CI95 [{lo_:.4f}, {hi_:.4f}]  (lebar +-{(hi_-lo_)/2:.4f})")

    pa, pb = a["net_pnl"].to_numpy(float), b["net_pnl"].to_numpy(float)
    dpf = np.array([
        _pf(pb[rng.integers(0, len(pb), len(pb))]) - _pf(pa[rng.integers(0, len(pa), len(pa))])
        for _ in range(N_BOOT)
    ])
    dpf = dpf[np.isfinite(dpf)]
    lo2, hi2 = np.percentile(dpf, [2.5, 97.5])
    print(f"\n  Selisih PF (OFF - 0.7) teramati : {_pf(pb) - _pf(pa):+.4f}")
    print(f"  CI95 selisih PF                 : [{lo2:+.4f}, {hi2:+.4f}]")
    print(f"  P(OFF > 0.7)                    : {(dpf > 0).mean():.3f}")
    print(f"  -> LANTAI DERAU PF ~ +-{(hi2-lo2)/2:.3f}. Selisih teramati "
          f"{'DI ATAS' if abs(_pf(pb)-_pf(pa)) > (hi2-lo2)/2 else 'DI BAWAH'} lantai derau.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
