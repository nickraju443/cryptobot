"""
alpaca_crypto_client.py — DATA-ONLY adapter for Alpaca's crypto market data.

Same auth pattern as SRI MATA's alpaca_client.py (reuses your existing keys
via env vars or the SRI MATA defaults). Implements BaseExchangeClient so it
can be slotted in via CRYPTO_EXCHANGE=alpaca (data-only, sim fills) or
combined with andX via the hybrid client (data ← Alpaca, orders → andX).

Alpaca's crypto data API is FREE — no separate subscription. The endpoints
live under /v1beta3/crypto/{loc}/ where {loc} is "us" (default) or "global".
Universe is ~25 majors (BTC/USD, ETH/USD, SOL/USD, ...) — not as wide as
Binance/Kraken but it's what Alpaca supports and it covers the highest-
volume names you want to scalp.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Optional
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

from exchange_client import BaseExchangeClient, Balance, OrderResult

logger = logging.getLogger(__name__)

# Alpaca's crypto market data is FREE and needs no API keys. If you have
# keys, set ALPACA_API_KEY / ALPACA_SECRET_KEY in the env for higher rate
# limits — otherwise keyless works fine.
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET  = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_DATA_BASE = os.environ.get("ALPACA_DATA_BASE", "https://data.alpaca.markets")
ALPACA_LOC = os.environ.get("ALPACA_CRYPTO_LOC", "us")

_HEADERS = {}
if ALPACA_API_KEY and ALPACA_SECRET:
    _HEADERS = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

# Alpaca's tradable crypto pairs (as of 2026 — adjust if Alpaca adds/removes).
# Mirrors what the SRI MATA bot would see if it scanned crypto.
ALPACA_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD",
    "LTC/USD", "BCH/USD", "LINK/USD", "MATIC/USD", "SHIB/USD",
    "AAVE/USD", "DOT/USD", "MKR/USD", "UNI/USD", "XRP/USD",
    "GRT/USD", "ADA/USD", "ALGO/USD", "NEAR/USD", "FIL/USD",
    "BAT/USD", "CRV/USD", "SUSHI/USD", "ATOM/USD", "PEPE/USD",
    "ICP/USD", "TRUMP/USD", "YFI/USD",
]

# Alpaca timeframe shorthand. The API accepts "1Min", "5Min", "15Min", "1Hour", "1Day".
_TF_MAP = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1w": "1Week",
}


# ----------------------------------------------------------------------
# Low-level transport
# ----------------------------------------------------------------------

def _get(endpoint: str, params: dict = None, timeout: float = 20, retries: int = 2):
    """REST GET with light retry. Returns parsed JSON or None on failure."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{ALPACA_DATA_BASE}{endpoint}",
                             headers=_HEADERS, params=params or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(1); continue
            logger.warning(f"[ALPACA] timeout after {retries+1} attempts: {endpoint}")
            return None
        except Exception as e:
            logger.warning(f"[ALPACA] {endpoint} failed: {e}")
            return None


# ----------------------------------------------------------------------
# AlpacaCryptoClient
# ----------------------------------------------------------------------

class AlpacaCryptoClient(BaseExchangeClient):
    """Alpaca crypto data feed. DATA-ONLY — place_order/cancel/balance are
    no-ops (use the hybrid client to combine with andX execution)."""

    name = "alpaca-crypto"
    quote_asset = "USD"

    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET:
            logger.info("Alpaca keys not set — using free keyless crypto data (works fine)")
        # Universe cached so we don't refetch snapshots every call
        self._universe_cache: list[str] = []
        self._universe_ts: float = 0.0
        self._universe_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Data (the methods that matter)
    # ------------------------------------------------------------------

    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        """Snapshot all Alpaca crypto pairs, rank by 24h volume.
        Cached for 5 minutes."""
        now = time.time()
        with self._universe_lock:
            if self._universe_cache and (now - self._universe_ts) < 300:
                return list(self._universe_cache[:n])

        all_syms = [s for s in ALPACA_CRYPTO_SYMBOLS
                    if not (exclude_stables and s.split("/")[0] in {"USDT", "USDC", "BUSD", "DAI"})]
        data = _get(f"/v1beta3/crypto/{ALPACA_LOC}/snapshots",
                    params={"symbols": ",".join(all_syms)})
        if not data or "snapshots" not in data:
            # Fall back to ordered hardcoded list
            with self._universe_lock:
                self._universe_cache = all_syms
                self._universe_ts = now
            return list(all_syms[:n])

        rows: list[tuple[str, float]] = []
        for sym, snap in data["snapshots"].items():
            daily = snap.get("dailyBar") or {}
            vol = float(daily.get("v") or 0)  # volume in base units
            price = float(daily.get("c") or snap.get("latestTrade", {}).get("p") or 0)
            quote_vol = vol * price  # approximate USD-quote volume
            if quote_vol > 0:
                rows.append((sym, quote_vol))
        rows.sort(key=lambda x: x[1], reverse=True)
        ranked = [s for s, _ in rows]
        # Pad with un-ranked symbols if Alpaca didn't return them
        for s in all_syms:
            if s not in ranked:
                ranked.append(s)

        with self._universe_lock:
            self._universe_cache = ranked
            self._universe_ts = now
        return ranked[:n]

    def get_price(self, symbol: str) -> Optional[float]:
        data = _get(f"/v1beta3/crypto/{ALPACA_LOC}/latest/trades",
                    params={"symbols": symbol})
        if not data:
            return None
        try:
            trades = data.get("trades", {}) or {}
            t = trades.get(symbol)
            if t and t.get("p"):
                return float(t["p"])
        except Exception:
            pass
        # Snapshot fallback
        snap = _get(f"/v1beta3/crypto/{ALPACA_LOC}/snapshots", params={"symbols": symbol})
        try:
            s = (snap or {}).get("snapshots", {}).get(symbol, {})
            t = s.get("latestTrade") or {}
            q = s.get("latestQuote") or {}
            bar = s.get("minuteBar") or s.get("dailyBar") or {}
            return float(t.get("p") or ((q.get("bp", 0) + q.get("ap", 0)) / 2) or bar.get("c") or 0) or None
        except Exception:
            return None

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        data = _get(f"/v1beta3/crypto/{ALPACA_LOC}/latest/trades",
                    params={"symbols": ",".join(symbols)})
        out: dict[str, float] = {}
        if data:
            for sym, t in (data.get("trades") or {}).items():
                try:
                    p = float(t.get("p") or 0)
                    if p > 0:
                        out[sym] = p
                except Exception:
                    pass
        # Per-symbol fallback for anything missing
        for s in symbols:
            if s not in out:
                p = self.get_price(s)
                if p:
                    out[s] = p
        return out

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        tf = _TF_MAP.get(timeframe, "5Min")
        # Compute an appropriate start time given the timeframe + limit
        seconds_per = {
            "1Min": 60, "5Min": 300, "15Min": 900, "30Min": 1800,
            "1Hour": 3600, "4Hour": 14400, "1Day": 86400, "1Week": 604800,
        }.get(tf, 300)
        start = (datetime.now(timezone.utc) - timedelta(seconds=seconds_per * limit * 2)).isoformat()
        data = _get(f"/v1beta3/crypto/{ALPACA_LOC}/bars",
                    params={"symbols": symbol, "timeframe": tf,
                            "start": start, "limit": int(limit)})
        if not data or "bars" not in data:
            return pd.DataFrame()
        bars = data["bars"].get(symbol) or []
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        # Alpaca crypto bar keys: t (RFC3339 time), o, h, l, c, v, n (trade count), vw (vwap)
        if "t" not in df.columns:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["t"], utc=True)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()

    # ------------------------------------------------------------------
    # Execution (data-only client — never trades, always simulates)
    # ------------------------------------------------------------------

    def place_order(self, symbol, side, qty, order_type="market",
                    limit_price=None, stop_loss=None, take_profit=None) -> OrderResult:
        # No live trading from this client. Caller should route real orders
        # through HybridClient which delegates to andX.
        last = self.get_price(symbol) or limit_price or 0.0
        return OrderResult(
            order_id=None, symbol=symbol, side=side, qty=float(qty),
            filled_price=float(last), status="filled",
            raw={"simulated": True, "reason": "alpaca-crypto is data-only"},
        )

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        return True

    def get_balance(self) -> Balance:
        # Alpaca has a brokerage balance, but for this bot the balance is
        # tracked in portfolio.json (sim) or by the execution client (andX).
        return Balance(quote_asset=self.quote_asset, free=0.0, total=0.0)

    def is_connected(self) -> bool:
        # Cheapest auth-touching call: latest BTC trade.
        try:
            return self.get_price("BTC/USD") is not None
        except Exception:
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    c = AlpacaCryptoClient()
    print("connected:", c.is_connected())
    print("top 5:", c.get_top_volume_symbols(n=5))
    print("BTC/USD:", c.get_price("BTC/USD"))
    df = c.get_candles("BTC/USD", "5m", 10)
    print("candles:", len(df))
    if not df.empty:
        print(df.tail(3))
