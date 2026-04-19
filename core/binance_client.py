"""
core/binance_client.py — Binance Futures API client
Rate limiting, retry dengan exponential backoff.
"""

import time
from typing import Optional

import requests

from core.utils import setup_logger

logger = setup_logger("binance_client")


class BinanceClient:
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        sleep_between: float = 0.12,
        sleep_rate_limit: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.base_url          = base_url
        self.sleep_between     = sleep_between
        self.sleep_rate_limit  = sleep_rate_limit
        self.max_retries       = max_retries
        self.backoff_base      = backoff_base
        self._last_request_time = 0.0

    def _get(self, endpoint: str, params: dict) -> Optional[list | dict]:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.sleep_between:
                time.sleep(self.sleep_between - elapsed)
            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", self.sleep_rate_limit))
                    logger.warning(f"Rate limit 429 — tunggu {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if resp.status_code == 418:
                    wait = self.sleep_rate_limit * (attempt + 1)
                    logger.warning(f"IP banned 418 — tunggu {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = self.backoff_base ** attempt
                    logger.warning(f"Server error {resp.status_code} [attempt {attempt+1}] — retry {wait:.1f}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                wait = self.backoff_base ** attempt
                logger.warning(f"Timeout [attempt {attempt+1}] — retry {wait:.1f}s")
                time.sleep(wait)
            except requests.exceptions.ConnectionError as e:
                wait = self.backoff_base ** attempt
                logger.warning(f"ConnectionError: {e} — retry {wait:.1f}s")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e} | URL: {url} | params: {params}")
                return None

        logger.error(f"Semua {self.max_retries} retry gagal: {url}")
        return None

    def get_klines(self, symbol, interval, start_time_ms, end_time_ms, limit=1500):
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_time_ms,
            "endTime":   end_time_ms,
            "limit":     limit,
        }
        return self._get("/fapi/v1/klines", params)

    def get_open_interest_hist(self, symbol, period="1h", start_time_ms=None, end_time_ms=None, limit=500):
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        return self._get("/futures/data/openInterestHist", params)

    def get_funding_rate(self, symbol, start_time_ms=None, end_time_ms=None, limit=1000):
        params = {"symbol": symbol, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        return self._get("/fapi/v1/fundingRate", params)

    def get_taker_long_short_ratio(self, symbol, period="1h", start_time_ms=None, end_time_ms=None, limit=500):
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        return self._get("/futures/data/takerlongshortRatio", params)

    def get_global_long_short_ratio(self, symbol, period="1h", start_time_ms=None, end_time_ms=None, limit=500):
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        return self._get("/futures/data/globalLongShortAccountRatio", params)

    def get_server_time(self):
        result = self._get("/fapi/v1/time", {})
        return result.get("serverTime") if result else None

    def test_connection(self) -> bool:
        return self._get("/fapi/v1/ping", {}) is not None
