"""
pipeline/analyze_min_hold.py — Analisis Optimal MIN_HOLD_BARS
=============================================================

Logika:
  - Load semua *_features_v3.parquet dari data/labeled/
  - Untuk setiap bar berlabel LONG/SHORT, hitung berapa bar H1 sampai
    harga menyentuh swing target (h4_swing_high untuk LONG, h4_swing_low
    untuk SHORT)
  - Plot distribusi holding time → rekomendasikan MIN_HOLD_BARS

Jalankan:
  python pipeline/analyze_min_hold.py
  python pipeline/analyze_min_hold.py --coins SOLUSDT ETHUSDT
  python pipeline/analyze_min_hold.py --save-plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import LABEL_DIR, MAX_HOLDING_BARS

# ─── Matplotlib setup ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # default: headless-safe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    plt.switch_backend("TkAgg")  # upgrade ke GUI jika ada display
except Exception:
    pass  # tetap pakai Agg (headless / server)


# ─── Helper: hitung holding time per bar ──────────────────────────────────────

def compute_holding_times(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Untuk setiap bar bertanda LONG atau SHORT, simulasikan forward-walk
    dan hitung berapa bar H1 sampai harga menyentuh swing target atau timeout.

    LONG  → target = h4_swing_high  (tunggu high >= target)
    SHORT → target = h4_swing_low   (tunggu low  <= target)

    Return: pd.Series of int (holding bars), index = entry bar timestamps.
    """
    required_cols = {"label", "h4_swing_high", "h4_swing_low", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  [{symbol}] SKIP — kolom tidak ditemukan: {missing}")
        return pd.Series(dtype=int)

    # Filter hanya LONG/SHORT
    mask = df["label"].isin(["LONG", "SHORT"])
    entries = df[mask].copy()
    if entries.empty:
        print(f"  [{symbol}] SKIP — tidak ada bar LONG/SHORT")
        return pd.Series(dtype=int)

    hold_times = {}
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    idx    = df.index

    for ts, row in entries.iterrows():
        pos     = df.index.get_loc(ts)
        label   = row["label"]
        target  = row["h4_swing_high"] if label == "LONG" else row["h4_swing_low"]

        if pd.isna(target):
            continue

        hit = MAX_HOLDING_BARS  # default: timeout
        for k in range(1, MAX_HOLDING_BARS + 1):
            fwd = pos + k
            if fwd >= len(df):
                hit = k
                break
            if label == "LONG"  and highs[fwd] >= target:
                hit = k
                break
            if label == "SHORT" and lows[fwd]  <= target:
                hit = k
                break

        hold_times[ts] = hit

    return pd.Series(hold_times, name="hold_bars")


# ─── Load & analisis ──────────────────────────────────────────────────────────

def load_and_analyze(coins: list[str] | None = None) -> pd.Series:
    """Load semua _features_v3.parquet, kembalikan Series holding_times."""
    parquet_files = sorted(LABEL_DIR.glob("*_features_v3.parquet"))
    if not parquet_files:
        print(f"[ERROR] Tidak ada *_features_v3.parquet di {LABEL_DIR}")
        sys.exit(1)

    if coins:
        coins_upper = [c.upper() for c in coins]
        parquet_files = [p for p in parquet_files
                         if any(coin in p.name for coin in coins_upper)]
        if not parquet_files:
            print(f"[ERROR] Tidak ada file untuk koin: {coins_upper}")
            sys.exit(1)

    all_hold = []
    for pfile in parquet_files:
        symbol = pfile.name.replace("_features_v3.parquet", "")
        print(f"  Loading {symbol}...")
        try:
            df = pd.read_parquet(pfile)
            if df.index.name and pd.api.types.is_datetime64_any_dtype(df.index):
                pass
            else:
                # Coba set index ke kolom waktu jika ada
                for tcol in ("timestamp", "datetime", "time", "open_time"):
                    if tcol in df.columns:
                        df = df.set_index(tcol)
                        break

            print(f"    {len(df):,} rows | cols: {list(df.columns[:8])}...")
            ht = compute_holding_times(df, symbol)
            if not ht.empty:
                all_hold.append(ht)
                print(f"    → {len(ht):,} entries LONG/SHORT | "
                      f"median={ht.median():.0f}h | "
                      f"p10={ht.quantile(0.10):.0f}h")
        except Exception as e:
            print(f"  [{symbol}] ERROR: {e}")

    if not all_hold:
        print("[ERROR] Tidak ada data holding time yang bisa dihitung.")
        sys.exit(1)

    combined = pd.concat(all_hold)
    print(f"\n[OK] Total entries: {len(combined):,} dari {len(parquet_files)} file(s)")
    return combined


# ─── Statistik & Rekomendasi ──────────────────────────────────────────────────

def print_stats(hold: pd.Series) -> dict:
    p10 = hold.quantile(0.10)
    p25 = hold.quantile(0.25)
    p50 = hold.quantile(0.50)
    p75 = hold.quantile(0.75)
    p90 = hold.quantile(0.90)
    mean_ = hold.mean()
    timeout_pct = (hold == MAX_HOLDING_BARS).mean() * 100

    # Rekomendasi: floor(p10), minimum 2, maksimum p25
    recommended = max(2, int(np.floor(p10)))
    recommended = min(recommended, int(p25))  # jangan lebih dari p25

    print("\n" + "═" * 50)
    print("  DISTRIBUSI HOLDING TIME (bar H1)")
    print("═" * 50)
    print(f"  Mean          : {mean_:.1f} bar  ({mean_:.1f} jam)")
    print(f"  Persentil 10  : {p10:.1f} bar  ({p10:.1f} jam)")
    print(f"  Persentil 25  : {p25:.1f} bar  ({p25:.1f} jam)")
    print(f"  Persentil 50  : {p50:.1f} bar  ({p50:.1f} jam)")
    print(f"  Persentil 75  : {p75:.1f} bar  ({p75:.1f} jam)")
    print(f"  Persentil 90  : {p90:.1f} bar  ({p90:.1f} jam)")
    print(f"  Timeout rate  : {timeout_pct:.1f}%  (hit MAX_HOLDING_BARS={MAX_HOLDING_BARS})")
    print("─" * 50)
    print(f"  ✅ REKOMENDASI MIN_HOLD_BARS = {recommended}")
    print(f"     (floor(P10={p10:.1f}) = {int(np.floor(p10))}, "
          f"dibatasi max P25={int(p25)})")
    print("═" * 50)

    return {
        "mean": mean_, "p10": p10, "p25": p25, "p50": p50,
        "p75": p75, "p90": p90, "timeout_pct": timeout_pct,
        "recommended": recommended,
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_distribution(hold: pd.Series, stats: dict, save_plot: bool):
    p10 = stats["p10"]
    p25 = stats["p25"]
    p50 = stats["p50"]
    rec = stats["recommended"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    # ── Panel kiri: Histogram ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1a1d2e")

    bins = range(1, MAX_HOLDING_BARS + 2)
    n, edges, patches = ax1.hist(
        hold, bins=bins, color="#4f8ef7", alpha=0.85, edgecolor="#2a2d3e",
        linewidth=0.5
    )

    # Warnai bar sebelum rekomendasi dengan warna berbeda
    for patch, left in zip(patches, edges[:-1]):
        if left < rec:
            patch.set_facecolor("#e05c5c")
            patch.set_alpha(0.6)

    # Garis persentil
    vline_kw = dict(linewidth=1.8, alpha=0.9)
    ax1.axvline(p10, color="#f5a623", linestyle="--", label=f"P10 = {p10:.0f}h", **vline_kw)
    ax1.axvline(p25, color="#7ed321", linestyle="--", label=f"P25 = {p25:.0f}h", **vline_kw)
    ax1.axvline(p50, color="#50e3c2", linestyle="--", label=f"P50 = {p50:.0f}h", **vline_kw)
    ax1.axvline(rec, color="#ff4757", linestyle="-",  label=f"Rekomendasi MIN_HOLD={rec}h",
                linewidth=2.5, alpha=1.0)

    ax1.set_xlabel("Holding Time (bar H1 = jam)", color="#b0b8d0", fontsize=11)
    ax1.set_ylabel("Jumlah Entry", color="#b0b8d0", fontsize=11)
    ax1.set_title("Distribusi Holding Time — LONG & SHORT", color="#e8eaf6", fontsize=13,
                  pad=12, fontweight="bold")
    ax1.tick_params(colors="#b0b8d0")
    ax1.spines[:].set_color("#2a2d3e")
    ax1.legend(framealpha=0.2, labelcolor="#e8eaf6", facecolor="#1a1d2e", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotation box
    textstr = (f"Total entries: {len(hold):,}\n"
               f"Timeout ({MAX_HOLDING_BARS}h): {stats['timeout_pct']:.1f}%\n"
               f"Mean: {stats['mean']:.1f}h")
    props = dict(boxstyle="round", facecolor="#252836", alpha=0.7, edgecolor="#4f8ef7")
    ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=8.5,
             verticalalignment="top", horizontalalignment="right",
             bbox=props, color="#b0b8d0")

    # ── Panel kanan: CDF ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1d2e")

    sorted_hold = np.sort(hold.values)
    cdf = np.arange(1, len(sorted_hold) + 1) / len(sorted_hold)

    ax2.plot(sorted_hold, cdf * 100, color="#4f8ef7", linewidth=2)
    ax2.fill_between(sorted_hold, cdf * 100, alpha=0.15, color="#4f8ef7")

    # Garis CDF untuk persentil
    for pct, val, color in [(10, p10, "#f5a623"), (25, p25, "#7ed321"), (50, p50, "#50e3c2")]:
        ax2.axhline(pct, color=color, linestyle=":", alpha=0.7, linewidth=1.2)
        ax2.axvline(val, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
        ax2.annotate(f"P{pct}={val:.0f}h", xy=(val, pct),
                     xytext=(val + 0.5, pct + 2),
                     color=color, fontsize=8.5, alpha=0.9)

    ax2.axvline(rec, color="#ff4757", linestyle="-", linewidth=2.5,
                label=f"MIN_HOLD = {rec}h", alpha=1.0)

    # Shaded region: "terlalu cepat exit"
    ax2.axvspan(0, rec, alpha=0.08, color="#ff4757", label="Zone too-early exit")

    ax2.set_xlabel("Holding Time (bar H1 = jam)", color="#b0b8d0", fontsize=11)
    ax2.set_ylabel("Persentase Kumulatif (%)", color="#b0b8d0", fontsize=11)
    ax2.set_title("CDF Holding Time", color="#e8eaf6", fontsize=13, pad=12, fontweight="bold")
    ax2.tick_params(colors="#b0b8d0")
    ax2.spines[:].set_color("#2a2d3e")
    ax2.legend(framealpha=0.2, labelcolor="#e8eaf6", facecolor="#1a1d2e", fontsize=9)
    ax2.set_ylim(0, 102)
    ax2.set_xlim(0, MAX_HOLDING_BARS + 1)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.suptitle(
        f"Analisis Optimal MIN_HOLD_BARS  |  Base TF: H1  |  MAX: {MAX_HOLDING_BARS}h",
        color="#e8eaf6", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if save_plot:
        out_path = ROOT / "reports" / "min_hold_analysis.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n[Plot saved] → {out_path}")
    else:
        try:
            plt.show()
        except Exception:
            out_path = ROOT / "reports" / "min_hold_analysis.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"\n[Tidak ada display] Plot disimpan → {out_path}")

    plt.close()


# ─── Rekomendasi config ───────────────────────────────────────────────────────

def print_config_recommendation(stats: dict):
    rec = stats["recommended"]
    print(f"""
┌─────────────────────────────────────────────┐
│  Salin ke config.py:                        │
│                                             │
│  MIN_HOLD_BARS = {rec:<4}  # bar H1 = {rec} jam    │
│                                             │
│  (berdasarkan P10 holding time aktual)      │
└─────────────────────────────────────────────┘
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisis optimal MIN_HOLD_BARS dari data labeled v3"
    )
    parser.add_argument(
        "--coins", nargs="+", metavar="SYMBOL",
        help="Filter koin tertentu (default: semua)"
    )
    parser.add_argument(
        "--save-plot", action="store_true",
        help="Simpan plot ke reports/min_hold_analysis.png tanpa menampilkan GUI"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Hanya tampilkan nama kolom parquet (debugging)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Inspect mode ─────────────────────────────────────────────────────────
    if args.inspect:
        files = sorted(LABEL_DIR.glob("*_features_v3.parquet"))
        if not files:
            print(f"[ERROR] Tidak ada file di {LABEL_DIR}")
            return
        for f in files[:3]:
            df = pd.read_parquet(f, columns=None)
            print(f"\n{f.name}  ({len(df):,} rows)")
            print("  Columns:", list(df.columns))
            print("  Index  :", df.index[:3].tolist())
            if "label" in df.columns:
                print("  Labels :", df["label"].value_counts().to_dict())
        return

    print(f"[INFO] LABEL_DIR      = {LABEL_DIR}")
    print(f"[INFO] MAX_HOLDING_BARS = {MAX_HOLDING_BARS}")

    # ── Load & hitung holding times ───────────────────────────────────────────
    hold = load_and_analyze(coins=args.coins)

    # ── Statistik ─────────────────────────────────────────────────────────────
    stats = print_stats(hold)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_distribution(hold, stats, save_plot=args.save_plot)

    # ── Rekomendasi ───────────────────────────────────────────────────────────
    print_config_recommendation(stats)


if __name__ == "__main__":
    main()
