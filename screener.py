"""
screener.py — Crypto universe + parallel scan.

Universe is dynamic: top-N by 24h quote-volume from the active exchange.
Refresh cadence is throttled (5 min default) so the bot focuses on what's
liquid right now without rebuilding state every tick. Stablecoin pairs are
filtered out.
"""

from __future__ import annotations
import os
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from analysis import full_analysis
from exchange_client import get_client

logger = logging.getLogger(__name__)

UNIVERSE_SIZE = int(os.environ.get("UNIVERSE_SIZE", "30"))
UNIVERSE_REFRESH_SEC = int(os.environ.get("UNIVERSE_REFRESH_SEC", "300"))

# Fallback list if the exchange returns nothing (e.g. andX stub before wiring)
FALLBACK_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT",
    "DOT/USDT", "TRX/USDT", "TON/USDT", "LTC/USDT", "BCH/USDT",
    "ATOM/USDT", "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
    "INJ/USDT", "SUI/USDT", "TIA/USDT", "FIL/USDT", "AAVE/USDT",
    "UNI/USDT", "RNDR/USDT", "FET/USDT", "PEPE/USDT", "WIF/USDT",
]

_universe_cache: list[str] = []
_universe_ts: float = 0.0
_universe_lock = threading.Lock()


def get_universe(force_refresh: bool = False) -> list[str]:
    """Return the active top-volume universe (cached, refreshed periodically)."""
    global _universe_cache, _universe_ts
    now = time.time()
    with _universe_lock:
        if not force_refresh and _universe_cache and (now - _universe_ts) < UNIVERSE_REFRESH_SEC:
            return list(_universe_cache)

    try:
        client = get_client()
        top = client.get_top_volume_symbols(n=UNIVERSE_SIZE, exclude_stables=True)
    except Exception as e:
        logger.warning(f"universe refresh failed: {e}")
        top = []

    if not top:
        top = list(FALLBACK_UNIVERSE)[:UNIVERSE_SIZE]
        logger.warning(f"falling back to static universe ({len(top)} symbols)")

    with _universe_lock:
        _universe_cache = top
        _universe_ts = now
    return list(top)


def scan_single(symbol: str, period: str = "3mo", interval: str = "5m",
                weight_overrides: dict = None):
    try:
        return full_analysis(symbol, period, interval, weight_overrides)
    except Exception as e:
        logger.debug(f"scan_single {symbol} failed: {e}")
        return None


def scan_market(
    symbols: list[str] = None,
    period: str = "3mo",
    interval: str = "5m",
    min_confidence: float = 50.0,
    min_risk_reward: float = 1.0,
    signal_filter: str = "BUY",       # "BUY" | "SELL" | "ALL"
    max_workers: int = 10,
    weight_overrides: dict = None,
    progress_callback=None,
) -> list[dict]:
    """Scan symbols in parallel, filter, rank. Same return shape as SRI MATA."""
    if symbols is None:
        symbols = get_universe()
    results = []
    completed = 0
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=min(max_workers, 16)) as ex:
        futures = {ex.submit(scan_single, s, period, interval, weight_overrides): s for s in symbols}
        for fut in as_completed(futures):
            completed += 1
            sym = futures[fut]
            if progress_callback:
                progress_callback(completed, total, sym)
            if completed % 5 == 0:
                time.sleep(0.05)  # yield GIL so Flask stays responsive
            r = fut.result()
            if r is None:
                continue
            sig = r["signal"]
            if signal_filter == "BUY" and "BUY" not in sig:
                continue
            if signal_filter == "SELL" and "SELL" not in sig:
                continue
            if r["confidence"] < min_confidence:
                continue
            if r["risk_reward"] < min_risk_reward:
                continue
            results.append(r)

    results.sort(key=lambda r: (2 if "STRONG" in r["signal"] else 1,
                                r["confidence"], r["risk_reward"]), reverse=True)
    return results
