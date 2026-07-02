"""
exchange_client.py — Abstract base for the crypto exchange data + execution layer.

The bot talks ONLY to this interface. Plug in any exchange (andX, Binance,
Coinbase, Kraken, CCXT-wrapped, etc.) by subclassing BaseExchangeClient and
filling in the seven methods below.

Default client is selected via env var CRYPTO_EXCHANGE (default: "andx").

  get_top_volume_symbols()  → discover what to trade (24h volume leaders)
  get_price(symbol)         → last trade price (used by portfolio mark-to-market)
  get_prices(symbols)       → batched snapshots (efficient scanning)
  get_candles(symbol, ...)  → OHLCV bars for indicator calculation
  place_order(...)          → market or limit order
  cancel_order(order_id)    → cancel an open order
  get_balance()             → free cash / quote asset balance

Symbols use the format "BTC/USDT" everywhere internally. Adapters translate
to the exchange-specific format (e.g. "BTCUSDT" for Binance, "XBT-USD" for
Kraken) inside the subclass.
"""

from __future__ import annotations

import os
import time
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data shapes
# ----------------------------------------------------------------------

@dataclass
class OrderResult:
    """Returned by place_order. order_id may be None for stub/sim."""
    order_id: Optional[str]
    symbol: str
    side: str               # "buy" or "sell"
    qty: float              # filled quantity (float — crypto is fractional)
    filled_price: float     # average fill price
    status: str             # "filled" | "partial" | "pending" | "rejected"
    raw: Optional[dict] = None


@dataclass
class Balance:
    """Account balance snapshot. `free` is what's available to trade."""
    quote_asset: str        # "USDT", "USD", etc.
    free: float             # available
    total: float            # free + locked-in-orders


# ----------------------------------------------------------------------
# Abstract base
# ----------------------------------------------------------------------

class BaseExchangeClient(ABC):
    """Every exchange adapter implements this contract."""

    name: str = "base"
    quote_asset: str = "USDT"

    @abstractmethod
    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        """Return the top-N symbols by 24h quote-volume, e.g. ['BTC/USDT', 'ETH/USDT', ...].
        Stablecoin pairs (USDC/USDT etc) should be excluded by default."""
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Last-trade price for one symbol, or None if unavailable."""
        ...

    @abstractmethod
    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Batched price snapshot. Missing symbols are simply omitted."""
        ...

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",     # "1m", "5m", "15m", "1h", "4h", "1d"
        limit: int = 200,
    ) -> pd.DataFrame:
        """OHLCV candles, oldest -> newest. Columns: Open, High, Low, Close, Volume.
        Index is a DatetimeIndex (UTC). Empty DataFrame on failure."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,                  # "buy" | "sell"
        qty: float,
        order_type: str = "market", # "market" | "limit"
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """Submit an order. Crypto uses fractional qty (float)."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        ...

    @abstractmethod
    def get_balance(self) -> Balance:
        ...

    # -- Optional, default no-op websocket hook ---------------------------
    def subscribe_prices(self, symbols: list[str], on_tick) -> None:
        """Optional: stream live ticks via websocket. Default falls back to polling.
        Override in subclass for real-time updates."""
        pass

    def is_connected(self) -> bool:
        """Sanity check before starting the trading loop."""
        try:
            _ = self.get_balance()
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------
# Cache wrapper — wraps any client with a 15s price cache (same as SRI MATA)
# ----------------------------------------------------------------------

class CachedExchangeClient(BaseExchangeClient):
    """Delegates to an inner client but caches get_price/get_prices for `ttl_sec`.
    The bot scans many symbols every few seconds and re-reads prices for live PnL,
    so caching at this layer keeps the API call rate down."""

    def __init__(self, inner: BaseExchangeClient, ttl_sec: float = 15.0):
        self.inner = inner
        self.ttl = float(ttl_sec)
        self.name = f"cached-{inner.name}"
        self.quote_asset = inner.quote_asset
        self._cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, fetched_at)
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def _set(self, symbol: str, price: float) -> None:
        with self._lock:
            self._cache[symbol] = (float(price), self._now())

    def _get(self, symbol: str) -> Optional[float]:
        with self._lock:
            v = self._cache.get(symbol)
        if not v:
            return None
        price, ts = v
        if self._now() - ts > self.ttl:
            return None
        return price

    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        return self.inner.get_top_volume_symbols(n=n, exclude_stables=exclude_stables)

    def get_price(self, symbol: str) -> Optional[float]:
        cached = self._get(symbol)
        if cached is not None:
            return cached
        p = self.inner.get_price(symbol)
        if p is not None and p > 0:
            self._set(symbol, p)
        return p

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        # Try cache first
        out: dict[str, float] = {}
        misses: list[str] = []
        for s in symbols:
            c = self._get(s)
            if c is not None:
                out[s] = c
            else:
                misses.append(s)
        if misses:
            fresh = self.inner.get_prices(misses)
            for s, p in fresh.items():
                if p > 0:
                    self._set(s, p)
                    out[s] = p
        return out

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        return self.inner.get_candles(symbol, timeframe=timeframe, limit=limit)

    def place_order(self, symbol, side, qty, order_type="market", limit_price=None, stop_loss=None, take_profit=None):
        # Invalidate cache for this symbol so the next read is fresh
        with self._lock:
            self._cache.pop(symbol, None)
        return self.inner.place_order(
            symbol, side, qty, order_type=order_type,
            limit_price=limit_price, stop_loss=stop_loss, take_profit=take_profit,
        )

    def place_order_universal(self, symbol, side, qty, price_hint):
        """Pass-through to the inner client's auto-routing order method (docs
        API for BTC/ETH/ANDX1 longs, instant_order via session cookies for
        everything else). Falls back to place_order if the inner client doesn't
        implement it (e.g. stub exchange)."""
        with self._lock:
            self._cache.pop(symbol, None)
        if hasattr(self.inner, "place_order_universal"):
            return self.inner.place_order_universal(
                symbol=symbol, side=side, qty=qty, price_hint=price_hint)
        return self.inner.place_order(symbol, side, qty, order_type="market")

    def instant_session_available(self) -> bool:
        """Forward the session-loaded probe to the inner client."""
        if hasattr(self.inner, "instant_session_available"):
            return self.inner.instant_session_available()
        return False

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        return self.inner.cancel_order(order_id, symbol=symbol)

    def get_balance(self) -> Balance:
        return self.inner.get_balance()

    def subscribe_prices(self, symbols, on_tick):
        def _wrapped(symbol, price):
            self._set(symbol, price)
            on_tick(symbol, price)
        return self.inner.subscribe_prices(symbols, _wrapped)


# ----------------------------------------------------------------------
# Factory — pick the active client from env
# ----------------------------------------------------------------------

_client_singleton: Optional[BaseExchangeClient] = None
_client_lock = threading.Lock()


def get_client() -> BaseExchangeClient:
    """Return the active exchange client (cached singleton). Driven by env var
    CRYPTO_EXCHANGE. Defaults to andx."""
    global _client_singleton
    with _client_lock:
        if _client_singleton is not None:
            return _client_singleton

        which = (os.environ.get("CRYPTO_EXCHANGE") or "andx").lower().strip()

        if which == "andx":
            from andx_client import AndxClient
            inner: BaseExchangeClient = AndxClient()
        elif which == "alpaca":
            from alpaca_crypto_client import AlpacaCryptoClient
            inner = AlpacaCryptoClient()
        elif which == "hybrid":
            from hybrid_client import HybridClient
            inner = HybridClient.from_env()
        elif which == "binance":
            from ccxt_client import CcxtClient
            inner = CcxtClient(exchange_id="binance")
        elif which == "kraken":
            from ccxt_client import CcxtClient
            inner = CcxtClient(exchange_id="kraken")
        elif which == "ccxt":
            ex_id = os.environ.get("CCXT_EXCHANGE", "binance")
            from ccxt_client import CcxtClient
            inner = CcxtClient(exchange_id=ex_id)
        else:
            raise RuntimeError(
                f"Unknown CRYPTO_EXCHANGE='{which}'. "
                "Set to andx | alpaca | hybrid | ccxt | binance | kraken."
            )

        ttl = float(os.environ.get("PRICE_CACHE_TTL_SEC", "15"))
        _client_singleton = CachedExchangeClient(inner, ttl_sec=ttl)
        logger.info(f"exchange client active: {_client_singleton.name} (quote={_client_singleton.quote_asset})")
        return _client_singleton


def reset_client() -> None:
    """Test helper — force the next get_client() to re-build."""
    global _client_singleton
    with _client_lock:
        _client_singleton = None
