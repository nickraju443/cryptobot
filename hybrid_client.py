"""
hybrid_client.py — Data from one exchange, execution on another.

Mirrors how SRI MATA splits Alpaca (data/SIP) and IBKR (execution).
Default wiring for CryptoBot:
    data  : alpaca-crypto      (free, no extra subscription)
    exec  : andx               (your platform — fills in andx_client.py)

Selected via env var CRYPTO_EXCHANGE=hybrid. Override the data side with
HYBRID_DATA (alpaca|ccxt|kraken|binance) or the exec side with HYBRID_EXEC
(andx|ccxt|alpaca). The bot is unaware of the split — every call still
goes through `get_client()`.
"""

from __future__ import annotations

import os
import time
import threading
import logging
from typing import Optional

import pandas as pd

from exchange_client import BaseExchangeClient, Balance, OrderResult

logger = logging.getLogger(__name__)


def _build_data_client(name: str) -> BaseExchangeClient:
    name = (name or "alpaca").lower().strip()
    if name in ("alpaca", "alpaca-crypto"):
        from alpaca_crypto_client import AlpacaCryptoClient
        return AlpacaCryptoClient()
    if name == "andx":
        from andx_client import AndxClient
        return AndxClient()
    if name in ("ccxt", "kraken", "binance"):
        from ccxt_client import CcxtClient
        ex_id = name if name in ("kraken", "binance") else os.environ.get("CCXT_EXCHANGE", "kraken")
        return CcxtClient(exchange_id=ex_id)
    raise RuntimeError(f"Unknown HYBRID_DATA='{name}'")


def _build_exec_client(name: str) -> BaseExchangeClient:
    name = (name or "andx").lower().strip()
    if name == "andx":
        from andx_client import AndxClient
        return AndxClient()
    if name in ("andx_browser", "andx-browser"):
        # Browser-driven execution: a persistent Chromium session computes
        # the website's access-sign HMAC for us, unlocking the full
        # ~120-coin universe and sell-to-open shorting. The adapter still
        # delegates BTC/ETH/ANDX1 longs and all read-only calls to the
        # documented REST AndxClient for speed.
        from andx_browser_client import AndxBrowserClient
        return AndxBrowserClient()
    if name in ("alpaca", "alpaca-crypto"):
        from alpaca_crypto_client import AlpacaCryptoClient
        return AlpacaCryptoClient()
    if name in ("ccxt", "kraken", "binance"):
        from ccxt_client import CcxtClient
        ex_id = name if name in ("kraken", "binance") else os.environ.get("CCXT_EXCHANGE", "kraken")
        return CcxtClient(exchange_id=ex_id)
    raise RuntimeError(f"Unknown HYBRID_EXEC='{name}'")


class HybridClient(BaseExchangeClient):
    """Delegates data calls to `data`, order calls to `exec_`. Balance comes
    from the execution side (since that's where the money lives)."""

    def __init__(self, data: BaseExchangeClient, exec_: BaseExchangeClient):
        self.data = data
        self.exec_ = exec_
        self.name = f"hybrid(data={data.name}, exec={exec_.name})"
        # Quote asset follows the DATA side because that's where symbols come
        # from. The execution adapter is expected to translate symbols to its
        # own format internally (e.g. andx_client.to_andx).
        self.quote_asset = data.quote_asset or exec_.quote_asset
        # Cache of "what the exec side can actually trade" — refreshed periodically
        # so the bot never scans a symbol andX/etc. won't accept.
        self._tradable_cache: set[str] | None = None
        self._tradable_ts: float = 0.0
        self._tradable_lock = threading.Lock()
        self._tradable_ttl = 600.0  # 10 min

    @classmethod
    def from_env(cls) -> "HybridClient":
        data_name = os.environ.get("HYBRID_DATA", "alpaca")
        exec_name = os.environ.get("HYBRID_EXEC", "andx")
        data = _build_data_client(data_name)
        exec_ = _build_exec_client(exec_name)
        logger.info(f"HybridClient: data={data.name}, exec={exec_.name}")
        return cls(data, exec_)

    # ----- data ↓ (delegated to `data`) -------------------------------

    def _refresh_tradable_set(self) -> set[str]:
        """Pull the full list of symbols the exec side actually trades.

        Quote-asset alignment: data side may quote in USD, exec side in USDT
        (or vice versa). We normalize on the BASE asset only when matching, so
        BTC/USD (data) and BTC/USDT (exec) count as the same tradable market.
        """
        now = time.time()
        with self._tradable_lock:
            if self._tradable_cache is not None and (now - self._tradable_ts) < self._tradable_ttl:
                return self._tradable_cache
        try:
            # Big n so we get everything the exec side offers
            exec_syms = self.exec_.get_top_volume_symbols(n=500, exclude_stables=False)
        except Exception as e:
            logger.warning(f"hybrid: exec.get_top_volume_symbols failed: {e}")
            exec_syms = []
        bases = {s.split("/")[0] for s in exec_syms if "/" in s}
        with self._tradable_lock:
            self._tradable_cache = bases
            self._tradable_ts = now
        logger.info(f"hybrid: exec ({self.exec_.name}) tradable bases: {sorted(bases)}")
        return bases

    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        """Return the data side's top-by-volume — UNFILTERED.

        We deliberately do NOT prune to only exec-tradable symbols here. The
        brain should scan the full crypto universe so the sim portfolio can
        take every setup; the live side will reject orders for non-exec
        symbols (e.g. andX doesn't list SOL) and the bot logs that path
        clearly. Use `is_exec_tradable(symbol)` to gate live execution.
        """
        return self.data.get_top_volume_symbols(n=n, exclude_stables=exclude_stables)

    def is_exec_tradable(self, symbol: str) -> bool:
        """True iff the exec side has a market for this symbol's BASE asset.
        Bot uses this in _scan_and_buy to decide whether to attempt a LIVE
        order (otherwise: sim-only)."""
        try:
            base = symbol.split("/")[0]
            return base in self._refresh_tradable_set()
        except Exception:
            return False

    def get_price(self, symbol: str) -> Optional[float]:
        return self.data.get_price(symbol)

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        return self.data.get_prices(symbols)

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        return self.data.get_candles(symbol, timeframe=timeframe, limit=limit)

    # ----- execution ↓ (delegated to `exec_`) -------------------------

    def place_order(self, symbol, side, qty, order_type="market",
                    limit_price=None, stop_loss=None, take_profit=None) -> OrderResult:
        return self.exec_.place_order(
            symbol, side, qty, order_type=order_type,
            limit_price=limit_price, stop_loss=stop_loss, take_profit=take_profit,
        )

    def place_order_universal(self, symbol, side, qty, price_hint) -> OrderResult:
        """Auto-routing order: BTC/ETH/ANDX1 longs go through the documented
        API (HMAC), everything else goes through instant_order via session
        cookies. Falls back to plain place_order if the exec adapter is a
        stub that doesn't implement universal routing."""
        if hasattr(self.exec_, "place_order_universal"):
            return self.exec_.place_order_universal(
                symbol=symbol, side=side, qty=qty, price_hint=price_hint)
        return self.exec_.place_order(symbol, side, qty, order_type="market")

    def instant_session_available(self) -> bool:
        if hasattr(self.exec_, "instant_session_available"):
            return self.exec_.instant_session_available()
        return False

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        return self.exec_.cancel_order(order_id, symbol=symbol)

    def get_balance(self) -> Balance:
        return self.exec_.get_balance()

    def is_connected(self) -> bool:
        # Both sides need to be reachable for the bot to function.
        try:
            d = self.data.is_connected()
        except Exception:
            d = False
        try:
            e = self.exec_.is_connected()
        except Exception:
            e = False
        return d and e
