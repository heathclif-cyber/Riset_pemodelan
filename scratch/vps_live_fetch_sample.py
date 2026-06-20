# -*- coding: utf-8 -*-
"""Sample live feature fetch on VPS — compare key cols."""
import os
import json

os.environ.setdefault("TRADING_MODE", "live")

from app import create_app
from app.services.data_service import InferenceDataService, set_market_breadth
from app.services.config_loader import get_feature_cols, reload_cache

reload_cache()
app = create_app()
cols = get_feature_cols()
KEY = [
    "hmm_regime_enc", "long_short_ratio", "open_interest", "ofi_h4_delta",
    "cvd", "rsi_h4", "h4_trend", "cvd_slope_h4", "stochrsi_d", "log_ret_20",
]

out = {}
with app.app_context():
    set_market_breadth(0.5)
    svc = InferenceDataService()
    for sym in ["SOLUSDT", "ADAUSDT", "BTCUSDT", "LINKUSDT"]:
        df = svc.prepare_latest_features(sym)
        if df is None:
            out[sym] = {"status": "FAIL"}
            continue
        last = df.iloc[-1]
        out[sym] = {
            "status": "OK",
            "ts": str(df.index[-1]),
            "missing_lgbm": [c for c in cols if c not in df.columns],
            "features": {k: float(last.get(k, 0) or 0) for k in KEY if k in df.columns},
        }

print(json.dumps(out, indent=2))