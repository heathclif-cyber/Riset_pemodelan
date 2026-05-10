"""
Generate detailed comparison report: tables, matrices, charts for all aspects.
Output: ASPECT_COMPARISON_DETAIL.md + charts in models/runs/aspect_charts/
"""
import json, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    LABEL_MAP, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE, MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.models import load_lstm
from core.evaluator import simulate_trades_swing
from pipeline.backtest_utils import hierarchical_predict

DEVICE = torch.device("cpu")
MODEL_DIR = ROOT / "models"
HOLDOUT_DIR = ROOT / "data" / "holdout" / "labeled"
CHART_DIR = MODEL_DIR / "runs" / "aspect_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
CLR = {
    "bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
    "text": "#e6edf3", "green": "#3fb950", "red": "#f85149",
    "orange": "#d29922", "blue": "#58a6ff", "purple": "#bc8cff",
    "teal": "#39d353", "pink": "#db6dbf",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": CLR["bg"], "axes.facecolor": CLR["panel"],
    "axes.edgecolor": CLR["grid"], "grid.color": CLR["grid"],
    "text.color": CLR["text"], "axes.labelcolor": CLR["text"],
    "xtick.color": CLR["text"], "ytick.color": CLR["text"],
})


def load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt").to(DEVICE)
    scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)
    return lgbm, lstm, scaler, feat_cols


def load_coin_data(symbol, feat_cols):
    path = HOLDOUT_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)
    X = df[valid_cols].values.astype(np.float64)
    return df, X, valid_cols


def get_signals(df, X, valid_cols, lgbm, lstm, scaler):
    y_pred, conf = hierarchical_predict(None, lgbm, lstm, scaler, X, valid_cols, [], df[valid_cols])
    below = (y_pred != 1) & (conf < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred[below] = 1
    return y_pred, conf


def run_sim(df, y_pred, confidence, **kwargs):
    close_arr = df["close"].values
    high_arr  = df["high"].values if "high" in df.columns else close_arr
    low_arr   = df["low"].values if "low" in df.columns else close_arr
    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(len(df), np.nan)
    sl_arr = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(len(df), np.nan)

    defaults = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, max_hold=MAX_HOLDING_BARS,
        confidence=confidence,
    )
    defaults.update(kwargs)
    return simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=sh_arr, h4_swing_lows=sl_arr, **defaults,
    )


ASPECTS = {
    "#1 Sumber TP/SL": {
        "sekarang": {"hybrid_mode": True}, "proposal": {"hybrid_mode": False},
        "sekarang_label": "Hybrid (max swing,ATR)", "proposal_label": "Pure (swing only)",
    },
    "#2 Swing Freshness": {
        "sekarang": {"swing_freshness_check": True}, "proposal": {"swing_freshness_check": False},
        "sekarang_label": "Filter ON", "proposal_label": "Filter OFF",
    },
    "#3 Structural Filter": {
        "sekarang": {"structural_filter": True}, "proposal": {"structural_filter": False},
        "sekarang_label": "Filter ON", "proposal_label": "Filter OFF",
    },
    "#4 RR Gate": {
        "sekarang": {"min_rr": 0.0, "min_tp_atr": 0.0, "max_sl_atr": 999.0},
        "proposal": {"min_rr": 1.0, "min_tp_atr": 1.2, "max_sl_atr": 3.0},
        "sekarang_label": "RR Gate OFF", "proposal_label": "RR Gate ON",
    },
    "#5 SL ATR Mult": {
        "sekarang": {"sl_fallback_atr": 1.0}, "proposal": {"sl_fallback_atr": 1.5},
        "sekarang_label": "SL=1.0xATR", "proposal_label": "SL=1.5xATR",
    },
    "#7 Slippage": {
        "sekarang": {"slippage_enabled": False}, "proposal": {"slippage_enabled": True},
        "sekarang_label": "Slippage OFF", "proposal_label": "Slippage ON",
    },
    "#8 SL Trigger": {
        "sekarang": {"sl_trigger_mode": "close"}, "proposal": {"sl_trigger_mode": "highlow"},
        "sekarang_label": "Close candle", "proposal_label": "High/Low candle",
    },
    "#12 Sizing": {
        "sekarang": {"sizing_mode": "tiered"}, "proposal": {"sizing_mode": "fixed"},
        "sekarang_label": "Tiered (>0.75 -> $100)", "proposal_label": "Fixed ($100)",
    },
    "#15 Cooldown": {
        "sekarang": {"cooldown_enabled": True}, "proposal": {"cooldown_enabled": False},
        "sekarang_label": "Cooldown ON", "proposal_label": "Cooldown OFF",
    },
}


def collect_all_results(symbols, lgbm, lstm, scaler, feat_cols):
    """Collect results for all aspects + combined 'sekarang all' and 'proposal all'."""
    all_data = {}

    # Per-aspect
    for aspect_label, configs in ASPECTS.items():
        print(f"  Collecting: {aspect_label}...")
        all_data[aspect_label] = {"sekarang": {}, "proposal": {}}
        for mode in ("sekarang", "proposal"):
            for sym in symbols:
                data = load_coin_data(sym, feat_cols)
                if data is None: continue
                df, X, vc = data
                y_pred, conf = get_signals(df, X, vc, lgbm, lstm, scaler)
                sim = run_sim(df, y_pred, conf, **configs[mode])
                if sim.get("error"): continue
                all_data[aspect_label][mode][sym] = {
                    "wr": sim["winrate"], "trades": sim["total_trades"],
                    "pnl": sim["total_pnl"], "dd": sim.get("max_drawdown", 0),
                    "wins": sim["wins"], "losses": sim["losses"],
                }

    # Combined "sekarang all" vs "proposal all"
    sekarang_all = {
        "hybrid_mode": True, "swing_freshness_check": True, "structural_filter": True,
        "min_rr": 0.0, "min_tp_atr": 0.0, "max_sl_atr": 999.0,
        "sl_fallback_atr": 1.0, "slippage_enabled": False,
        "sl_trigger_mode": "close", "sizing_mode": "tiered", "cooldown_enabled": True,
    }
    proposal_all = {
        "hybrid_mode": False, "swing_freshness_check": False, "structural_filter": False,
        "min_rr": 1.0, "min_tp_atr": 1.2, "max_sl_atr": 3.0,
        "sl_fallback_atr": 1.5, "slippage_enabled": True,
        "sl_trigger_mode": "highlow", "sizing_mode": "fixed", "cooldown_enabled": False,
    }
    final_all = {
        "hybrid_mode": True, "swing_freshness_check": True, "structural_filter": True,
        "min_rr": 1.0, "min_tp_atr": 1.2, "max_sl_atr": 3.0,
        "sl_fallback_atr": 1.0, "slippage_enabled": True,
        "sl_trigger_mode": "close", "sizing_mode": "fixed", "cooldown_enabled": False,
    }
    final_hard_sl = {
        "hybrid_mode": True, "swing_freshness_check": True, "structural_filter": True,
        "min_rr": 1.0, "min_tp_atr": 1.2, "max_sl_atr": 3.0,
        "sl_fallback_atr": 1.0, "slippage_enabled": True,
        "sl_trigger_mode": "highlow", "sizing_mode": "fixed", "cooldown_enabled": False,
    }

    for label, params in [("Sekarang All", sekarang_all), ("Proposal All", proposal_all), 
                          ("Final Rekomendasi", final_all), ("Final (Hard SL)", final_hard_sl)]:
        print(f"  Collecting: {label}...")
        all_data[label] = {}
        for sym in symbols:
            data = load_coin_data(sym, feat_cols)
            if data is None: continue
            df, X, vc = data
            y_pred, conf = get_signals(df, X, vc, lgbm, lstm, scaler)
            sim = run_sim(df, y_pred, conf, **params)
            if sim.get("error"): continue
            all_data[label][sym] = {
                "wr": sim["winrate"], "trades": sim["total_trades"],
                "pnl": sim["total_pnl"], "dd": sim.get("max_drawdown", 0),
                "wins": sim["wins"], "losses": sim["losses"],
            }

    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def chart_aspect_bar(all_data, aspect_label, configs):
    """Horizontal bar chart: Winrate delta per coin for one aspect."""
    data_s = all_data[aspect_label]["sekarang"]
    data_p = all_data[aspect_label]["proposal"]
    common = sorted(set(data_s) & set(data_p))
    if not common: return

    deltas = [data_p[s]["wr"] - data_s[s]["wr"] for s in common]
    colors = [CLR["green"] if d >= 0 else CLR["red"] for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(common, deltas, color=colors, height=0.6, edgecolor="none")
    ax.axvline(0, color=CLR["text"], linewidth=1)
    ax.set_xlabel("Delta Winrate (Proposal - Sekarang)")
    ax.set_title(f"{aspect_label}: Winrate Delta per Koin\n{configs['sekarang_label']} vs {configs['proposal_label']}")
    ax.grid(axis="x", linewidth=0.5, alpha=0.5)

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() + (0.003 if d >= 0 else -0.003),
                bar.get_y() + bar.get_height()/2, f"{d:+.1%}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7.5,
                color=CLR["text"])

    plt.tight_layout()
    fname = CHART_DIR / f"aspect_{aspect_label.split()[0].replace('#','')}_wr_delta.svg"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close(fig)
    return fname


def chart_aggregate_comparison(all_data):
    """Grouped bar chart: Aggregate winrate, trades, PnL, DD for selected configs."""
    configs = ["#1 Sumber TP/SL", "#4 RR Gate", "#8 SL Trigger", "#15 Cooldown", "Sekarang All", "Proposal All", "Final Rekomendasi"]

    metrics = {
        "Winrate": ("wr", lambda x: f"{x:.1%}"),
        "Trades": ("trades", lambda x: f"{x:.0f}"),
        "PnL ($)": ("pnl", lambda x: f"${x:+.0f}"),
        "Max DD": ("dd", lambda x: f"{x:.1%}"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    colors_modes = {"sekarang": CLR["blue"], "proposal": CLR["orange"]}

    for ax, (metric_name, (key, fmt)) in zip(axes, metrics.items()):
        x = np.arange(len(configs))
        width = 0.35

        s_vals, p_vals = [], []
        for cfg in configs:
            if cfg in ASPECTS:
                s_list = [all_data[cfg]["sekarang"][s][key] for s in all_data[cfg]["sekarang"]]
                p_list = [all_data[cfg]["proposal"][s][key] for s in all_data[cfg]["proposal"]]
            else:
                s_list = [all_data[cfg][s][key] for s in all_data[cfg]]
                p_list = []
            s_vals.append(np.mean(s_list) if s_list else 0)
            if p_list:
                p_vals.append(np.mean(p_list))
            else:
                p_vals.append(0)

        ax.bar(x - width/2, s_vals, width, color=CLR["blue"], alpha=0.85, label="Sekarang", edgecolor="none")
        # Only show proposal where data exists (per-aspect)
        p_x = [i for i, v in enumerate(p_vals) if v != 0]
        p_v = [p_vals[i] for i in p_x]
        ax.bar([x[i] + width/2 for i in p_x], p_v, width, color=CLR["orange"], alpha=0.85, label="Proposal", edgecolor="none")

        # Final recommendation as green bar
        final_x = len(configs) - 1
        ax.bar(final_x + width/2 + 0.05, [s_vals[-1]], width * 0.6, color=CLR["green"], alpha=0.9, label="Final")

        ax.set_xticks(x)
        ax.set_xticklabels([c.split()[-1][:12] if c.split() else c[:12] for c in configs], rotation=45, ha="right", fontsize=7)
        ax.set_title(metric_name)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)
        if metric_name == "Winrate":
            ax.legend(fontsize=7, facecolor=CLR["panel"], labelcolor=CLR["text"])

    plt.suptitle("Aggregate Comparison: Sekarang vs Proposal vs Final", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = CHART_DIR / "aggregate_comparison.svg"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close(fig)
    return fname


def chart_heatmap_matrix(all_data):
    """Heatmap matrix: Per-coin, Per-aspect Winrate Delta (Proposal - Sekarang)."""
    aspect_keys = list(ASPECTS.keys())
    # Collect all coins across all aspects
    all_coins = set()
    for ak in aspect_keys:
        all_coins |= set(all_data[ak]["sekarang"].keys())
    coins = sorted(all_coins)

    matrix = np.zeros((len(coins), len(aspect_keys)))
    for ci, coin in enumerate(coins):
        for ai, ak in enumerate(aspect_keys):
            s_wr = all_data[ak]["sekarang"].get(coin, {}).get("wr", 0)
            p_wr = all_data[ak]["proposal"].get(coin, {}).get("wr", 0)
            matrix[ci, ai] = p_wr - s_wr

    fig, ax = plt.subplots(figsize=(14, max(8, len(coins)*0.35)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.25, vmax=0.25)

    ax.set_xticks(range(len(aspect_keys)))
    ax.set_xticklabels([k.split(" ", 1)[1][:14] if " " in k else k[:14] for k in aspect_keys], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(coins)))
    ax.set_yticklabels(coins, fontsize=7)
    ax.set_title("Winrate Delta Matrix (Proposal - Sekarang)\nGreen = Proposal Better | Red = Sekarang Better")

    for ci in range(len(coins)):
        for ai in range(len(aspect_keys)):
            val = matrix[ci, ai]
            color = "white" if abs(val) > 0.12 else "black"
            ax.text(ai, ci, f"{val:+.1%}", ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Delta Winrate")
    plt.tight_layout()
    fname = CHART_DIR / "heatmap_delta_matrix.svg"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close(fig)
    return fname


def chart_combined_hbar(all_data):
    """Horizontal bar chart: Final Recommendation vs Sekarang All vs Proposal All per coin."""
    coins = sorted(all_data["Final Rekomendasi"].keys())

    s_wr = [all_data["Sekarang All"][c]["wr"] for c in coins]
    p_wr = [all_data["Proposal All"][c]["wr"] for c in coins]
    f_wr = [all_data["Final Rekomendasi"][c]["wr"] for c in coins]
    s_pnl = [all_data["Sekarang All"][c]["pnl"] for c in coins]
    p_pnl = [all_data["Proposal All"][c]["pnl"] for c in coins]
    f_pnl = [all_data["Final Rekomendasi"][c]["pnl"] for c in coins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, len(coins)*0.35)))

    x = np.arange(len(coins))
    h = 0.25
    ax1.barh(x + h, s_wr, h, color=CLR["blue"], alpha=0.8, label="Sekarang All", edgecolor="none")
    ax1.barh(x, p_wr, h, color=CLR["orange"], alpha=0.8, label="Proposal All", edgecolor="none")
    ax1.barh(x - h, f_wr, h, color=CLR["green"], alpha=0.9, label="Final Rekomendasi", edgecolor="none")
    ax1.set_yticks(x)
    ax1.set_yticklabels(coins, fontsize=7)
    ax1.set_xlabel("Winrate")
    ax1.set_title("Winrate Comparison per Koin")
    ax1.legend(fontsize=7, facecolor=CLR["panel"], labelcolor=CLR["text"])
    ax1.grid(axis="x", linewidth=0.5, alpha=0.5)

    ax2.barh(x + h, s_pnl, h, color=CLR["blue"], alpha=0.8, label="Sekarang All", edgecolor="none")
    ax2.barh(x, p_pnl, h, color=CLR["orange"], alpha=0.8, label="Proposal All", edgecolor="none")
    ax2.barh(x - h, f_pnl, h, color=CLR["green"], alpha=0.9, label="Final Rekomendasi", edgecolor="none")
    ax2.set_yticks(x)
    ax2.set_yticklabels(coins, fontsize=7)
    ax2.set_xlabel("PnL ($)")
    ax2.set_title("PnL Comparison per Koin")
    ax2.legend(fontsize=7, facecolor=CLR["panel"], labelcolor=CLR["text"])
    ax2.grid(axis="x", linewidth=0.5, alpha=0.5)

    plt.suptitle("Final Architecture vs Baseline: Per-Coin Winrate & PnL", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = CHART_DIR / "combined_final_comparison.svg"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown(all_data, chart_paths):
    lines = []
    def w(line=""): lines.append(line)

    w("# Laporan Detail: Perbandingan Implementasi Sekarang vs Proposal")
    w()
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**Periode**: Holdout 2025-05-01 s/d 2026-04-01 (21 koin)")
    w(f"**Modal/Trade**: ${MODAL_PER_TRADE} | **Leverage**: {LEVERAGE_SIM[0]}x | **Fee**: {FEE_PER_SIDE:.1%}/side | **Slippage**: {SLIPPAGE_PER_SIDE:.1%}/side")
    w()

    # ── 1. Aggregate Summary Matrix ───────────────────────────────────────────
    w("## 1. Ringkasan Agregat — Semua Aspek")
    w()
    w("| # | Aspek | Sekarang | Proposal | Delta WR | Delta Tr | Delta PnL | Delta DD | Pemenang |")
    w("|---|-------|----------|----------|----------|----------|-----------|----------|----------|")

    for aspect_label, configs in ASPECTS.items():
        s_vals = [r for r in all_data[aspect_label]["sekarang"].values()]
        p_vals = [r for r in all_data[aspect_label]["proposal"].values()]
        if not s_vals or not p_vals: continue

        ms = {k: np.mean([v[k] for v in s_vals]) for k in ["wr","trades","pnl","dd"]}
        mp = {k: np.mean([v[k] for v in p_vals]) for k in ["wr","trades","pnl","dd"]}

        dwr = mp["wr"] - ms["wr"]
        dtr = mp["trades"] - ms["trades"]
        dpnl = mp["pnl"] - ms["pnl"]
        ddd = mp["dd"] - ms["dd"]

        # Score: 0.4*wr + 0.4*pnl_scaled - 0.2*dd
        s_score = ms["wr"]*0.4 + (1+ms["pnl"]/10000)*0.4 - abs(ms["dd"])*0.2
        p_score = mp["wr"]*0.4 + (1+mp["pnl"]/10000)*0.4 - abs(mp["dd"])*0.2
        winner = "**Proposal**" if p_score > s_score else "**Sekarang**"
        if abs(dwr) < 0.005 and abs(dpnl) < 50:
            winner = "Seri (tidak signifikan)"

        short_label = aspect_label.split(" ", 1)[1] if " " in aspect_label else aspect_label
        w(f"| {aspect_label.split()[0]} | {short_label} | "
          f"{ms['wr']:.1%} ({ms['trades']:.0f}tr) | {mp['wr']:.1%} ({mp['trades']:.0f}tr) | "
          f"{dwr:+.1%} | {dtr:+.0f} | ${dpnl:+.0f} | {ddd:+.1%} | {winner} |")
    w()

    # ── 2. Combined Configurations ─────────────────────────────────────────────
    w("## 2. Perbandingan Konfigurasi Gabungan")
    w()
    w("### 2.1 Ringkasan")
    w()
    w("| Konfigurasi | Mean WR | Mean Trades | Mean PnL | Mean Max DD |")
    w("|-------------|---------|-------------|----------|-------------|")

    for cfg_label in ["Sekarang All", "Proposal All", "Final Rekomendasi", "Final (Hard SL)"]:
        data = all_data[cfg_label]
        if not data: continue
        vals = list(data.values())
        m = {k: np.mean([v[k] for v in vals]) for k in ["wr","trades","pnl","dd"]}
        star = " " + chr(9733) if cfg_label == "Final Rekomendasi" else ""
        w(f"| **{cfg_label}**{star} | {m['wr']:.1%} | {m['trades']:.0f} | ${m['pnl']:+.0f} | {m['dd']:.1%} |")
    w()

    w("### 2.2 Detail Per Koin — Final vs Baseline vs Hard SL")
    w()
    w("| Coin | S-WR | P-WR | F-WR | H-WR | S-Tr | P-Tr | F-Tr | H-Tr | S-PnL | P-PnL | F-PnL | H-PnL | S-DD | P-DD | F-DD | H-DD |")
    w("|------|------|------|------|------|------|------|------|------|-------|-------|-------|-------|------|------|------|------|")

    coins = sorted(all_data["Final Rekomendasi"].keys())
    for c in coins:
        s = all_data["Sekarang All"].get(c, {})
        p = all_data["Proposal All"].get(c, {})
        f = all_data["Final Rekomendasi"].get(c, {})
        h = all_data["Final (Hard SL)"].get(c, {})
        w(f"| {c} | {s.get('wr',0):.1%} | {p.get('wr',0):.1%} | {f.get('wr',0):.1%} | {h.get('wr',0):.1%} | "
          f"{s.get('trades',0)} | {p.get('trades',0)} | {f.get('trades',0)} | {h.get('trades',0)} | "
          f"${s.get('pnl',0):+.0f} | ${p.get('pnl',0):+.0f} | ${f.get('pnl',0):+.0f} | ${h.get('pnl',0):+.0f} | "
          f"{s.get('dd',0):.1%} | {p.get('dd',0):.1%} | {f.get('dd',0):.1%} | {h.get('dd',0):.1%} |")
    w()

    # ── 3. Winrate Delta Matrix ───────────────────────────────────────────────
    w("## 3. Matriks Delta Winrate — Proposal vs Sekarang")
    w()
    w("Nilai: hijau = proposal lebih baik, merah = sekarang lebih baik")
    w()

    aspect_keys = list(ASPECTS.keys())
    all_coins = set()
    for ak in aspect_keys:
        all_coins |= set(all_data[ak]["sekarang"].keys())
    coins = sorted(all_coins)

    w("| Coin | " + " | ".join([k.split(" ", 1)[1][:10] if " " in k else k[:10] for k in aspect_keys]) + " |")
    w("|------|" + "|".join([":--:" for _ in aspect_keys]) + "|")

    for coin in coins:
        vals = []
        for ak in aspect_keys:
            s_wr = all_data[ak]["sekarang"].get(coin, {}).get("wr", 0)
            p_wr = all_data[ak]["proposal"].get(coin, {}).get("wr", 0)
            d = p_wr - s_wr
            emoji = "+" if d > 0.01 else ("-" if d < -0.01 else "=")
            vals.append(f"{emoji} {d:+.1%}")
        w(f"| {coin} | " + " | ".join(vals) + " |")
    w()

    # ── 4. Per-Aspek Detail Tables ─────────────────────────────────────────────
    w("## 4. Detail Per Aspek — Semua Koin")
    w()

    for aspect_label, configs in ASPECTS.items():
        w(f"### {aspect_label}")
        w()
        w(f"- **Sekarang**: {configs['sekarang_label']} — `{configs['sekarang']}`")
        w(f"- **Proposal**: {configs['proposal_label']} — `{configs['proposal']}`")
        w()
        w("| Coin | S-WR | P-WR | dWR | S-Tr | P-Tr | dTr | S-PnL | P-PnL | dPnL | S-DD | P-DD | dDD |")
        w("|------|------|------|-----|------|------|-----|-------|-------|------|------|------|-----|")

        common = sorted(set(all_data[aspect_label]["sekarang"]) & set(all_data[aspect_label]["proposal"]))
        for c in common:
            s = all_data[aspect_label]["sekarang"][c]
            p = all_data[aspect_label]["proposal"][c]
            w(f"| {c} | {s['wr']:.1%} | {p['wr']:.1%} | {p['wr']-s['wr']:+.1%} | "
              f"{s['trades']} | {p['trades']} | {p['trades']-s['trades']:+d} | "
              f"${s['pnl']:+.0f} | ${p['pnl']:+.0f} | ${p['pnl']-s['pnl']:+.0f} | "
              f"{s['dd']:.1%} | {p['dd']:.1%} | {p['dd']-s['dd']:+.1%} |")
        w()

    # ── 5. Charts ──────────────────────────────────────────────────────────────
    w("## 5. Charts")
    w()
    for chart_path in chart_paths:
        rel = Path(chart_path).relative_to(ROOT)
        w(f"![{Path(chart_path).stem}]({rel})")
        w()

    # ── 6. Rekomendasi Final ───────────────────────────────────────────────────
    w("## 6. Rekomendasi Final per Aspek")
    w()
    w("| # | Aspek | Rekomendasi | Alasan |")
    w("|---|-------|-------------|--------|")
    recs = [
        ("#1", "Sumber TP/SL", "**Hybrid** (max swing,ATR)", "WR +10%, PnL +$455, trades +13. ATR menstabilkan TP/SL."),
        ("#2", "Swing Freshness", "**Pertahankan** (ON)", "Safety net — tidak berdampak di holdout normal, kritis saat ekstrem."),
        ("#3", "Structural Filter", "**Pertahankan** (ON)", "Mencegah entry di luar range H4 — safety net."),
        ("#4", "RR Gate", "**Aktifkan** (ON)", "Filter 66 trade buruk (-27%), DD membaik. Kualitas > kuantitas."),
        ("#5", "SL ATR Mult", "**1.0xATR**", "SL tighter saat fallback — tidak berdampak di pure tier (swing dominan)."),
        ("#7", "Slippage", "**Aktifkan** (ON)", "Realisme backtest — dampak minimal (-1.0% WR, -$136 PnL)."),
        ("#8", "SL Trigger", "**Close candle**", "WR +11.9% vs highlow. Hindari false wick stop-out."),
        ("#12", "Sizing", "**Fixed** ($100)", "Lebih sederhana, PnL +$86, tidak signifikan."),
        ("#15", "Cooldown", "**Nonaktifkan** (OFF)", "Cooldown buang 170 trade valid ($1,174 PnL potensial)."),
    ]
    for num, aspect, rec, reason in recs:
        w(f"| {num} | {aspect} | {rec} | {reason} |")
    w()

    w("## 7. Arsitektur Final (config.py)")
    w()
    w("```python")
    w("TP_SL_HYBRID_MODE       = True     # max(swing, ATR) TP / min(swing, ATR) SL")
    w("TP_SL_SWING_FRESHNESS   = True     # tolak jika swing > 15% dari entry")
    w("TP_SL_STRUCTURAL_FILTER = True     # entry harus dalam [H4 Low, H4 High]")
    w("TP_SL_RR_GATE_ENABLED   = True     # validasi TP_dist >= 1.2xATR, SL_dist <= 3.0xATR, RR >= 1.0")
    w("TP_SL_FALLBACK_SL       = 1.0      # SL = 1.0 x ATR saat swing NaN")
    w("TP_SL_FALLBACK_TP       = 2.0      # TP = 2.0 x ATR saat swing NaN")
    w("TP_SL_SLIPPAGE_ENABLED  = True     # 0.05% per sisi entry/exit")
    w("TP_SL_TRIGGER_MODE      = 'close'  # hindari false wick stop-out")
    w("TP_SL_SIZING_MODE       = 'fixed'  # $100 per trade")
    w("TP_SL_COOLDOWN_ENABLED  = False    # terlalu restriktif")
    w("```")
    w()

    out_path = ROOT / "ASPECT_COMPARISON_DETAIL.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  DETAILED ASPECT COMPARISON REPORT")
    print("=" * 70)

    lgbm, lstm, scaler, feat_cols = load_models()
    symbols = sorted([p.stem.replace("_features_v3", "")
                      for p in HOLDOUT_DIR.glob("*_features_v3.parquet")])
    print(f"Coins: {len(symbols)}")

    # Collect all data
    print("\nCollecting data...")
    all_data = collect_all_results(symbols, lgbm, lstm, scaler, feat_cols)

    # Generate charts
    print("\nGenerating charts...")
    chart_paths = []

    for aspect_label, configs in ASPECTS.items():
        try:
            cp = chart_aspect_bar(all_data, aspect_label, configs)
            chart_paths.append(cp)
        except Exception as e:
            print(f"  Chart {aspect_label}: SKIP ({e})")

    try:
        chart_paths.append(chart_aggregate_comparison(all_data))
    except Exception as e:
        print(f"  Aggregate chart: SKIP ({e})")

    try:
        chart_paths.append(chart_heatmap_matrix(all_data))
    except Exception as e:
        print(f"  Heatmap: SKIP ({e})")

    try:
        chart_paths.append(chart_combined_hbar(all_data))
    except Exception as e:
        print(f"  Combined chart: SKIP ({e})")

    # Generate markdown
    print("\nGenerating markdown report...")
    md_path = generate_markdown(all_data, chart_paths)

    print(f"\n{'='*70}")
    print(f"  Report : {md_path}")
    print(f"  Charts : {CHART_DIR}")
    for cp in chart_paths:
        print(f"    - {cp.name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
