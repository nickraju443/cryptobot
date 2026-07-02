"""
ccxt_client.py — CCXT-backed exchange adapter.

Lets the bot run against any of 100+ exchanges (Binance, Kraken, Coinbase,
OKX, Bybit...) for testing while you wire up the andX adapter. CCXT
normalizes symbol format and the OHLCV/ticker/order shapes, so this file
stays generic.

To use:
  pip install ccxt
  set CRYPTO_EXCHANGE=ccxt
  set CCXT_EXCHANGE=binance              (or kraken, coinbase, okx, bybit, ...)
  set CCXT_API_KEY=...   (only for live orders; read-only data works without)
  set CCXT_API_SECRET=...

Default mode is READ-ONLY. To enable real orders also set CCXT_LIVE_TRADING=1.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import pandas as pd

from exchange_client import BaseExchangeClient, Balance, OrderResult

logger = logging.getLogger(__name__)

_STABLES = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDD", "FRAX",
            # Fiat (so the universe stays crypto-only on exchanges that list FX pairs)
            "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD"}


class CcxtClient(BaseExchangeClient):
    """Thin CCXT wrapper conforming to BaseExchangeClient."""

    def __init__(self, exchange_id: str = "binance"):
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ccxt not installed. pip install ccxt (or set CRYPTO_EXCHANGE=andx)"
            ) from e

        self.name = f"ccxt-{exchange_id}"
        self.quote_asset = os.environ.get("CCXT_QUOTE_ASSET", "USDT")
        self.live_trading = os.environ.get("CCXT_LIVE_TRADING", "0") == "1"

        klass = getattr(ccxt, exchange_id)
        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        api_key = os.environ.get("CCXT_API_KEY", "")
        api_secret = os.environ.get("CCXT_API_SECRET", "")
        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
        self.ex = klass(config)

        # Many CCXT exchanges need markets loaded before symbol translation works
        try:
            self.ex.load_markets()
        except Exception as e:
            logger.warning(f"ccxt {exchange_id} load_markets failed: {e}")

    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        try:
            tickers = self.ex.fetch_tickers()
        except Exception as e:
            logger.warning(f"ccxt fetch_tickers failed: {e}")
            return []
        rows = []
        for sym, t in tickers.items():
            if "/" not in sym:
                continue
            base, quote = sym.split("/", 1)
            if quote != self.quote_asset:
                continue
            if exclude_stables and base in _STABLES:
                continue
            qv = t.get("quoteVolume") or (t.get("baseVolume", 0) * (t.get("last") or 0))
            if not qv or qv <= 0:
                continue
            rows.append((sym, float(qv)))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in rows[:n]]

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.ex.fetch_ticker(symbol)
            p = t.get("last") or t.get("close")
            return float(p) if p else None
        except Exception as e:
            logger.warning(f"ccxt fetch_ticker {symbol} failed: {e}")
            return None

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            tickers = self.ex.fetch_tickers(symbols)
            for s, t in tickers.items():
                p = t.get("last") or t.get("close")
                if p:
                    out[s] = float(p)
        except Exception:
            # Per-symbol fallback
            for s in symbols:
                p = self.get_price(s)
                if p:
                    out[s] = p
        return out

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        try:
            raw = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=int(limit))
        except Exception as e:
            logger.warning(f"ccxt fetch_ohlcv {symbol} {timeframe} failed: {e}")
            return pd.DataFrame()
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        return df

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        if not self.live_trading:
            # Safety: simulated fill at last price
            last = self.get_price(symbol) or limit_price or 0.0
            return OrderResult(
                order_id=None, symbol=symbol, side=side, qty=float(qty),
                filled_price=float(last), status="filled",
                raw={"simulated": True, "reason": "CCXT_LIVE_TRADING != 1"},
            )
        try:
            if order_type == "market":
                resp = self.ex.create_market_order(symbol, side, qty)
            else:
                if limit_price is None:
                    return OrderResult(None, symbol, side, 0.0, 0.0, "rejected",
                                       raw={"error": "limit_price required"})
                resp = self.ex.create_limit_order(symbol, side, qty, limit_price)
            return OrderResult(
                order_id=str(resp.get("id") or ""),
                symbol=symbol, side=side,
                qty=float(resp.get("filled") or qty),
                filled_price=float(resp.get("average") or resp.get("price") or limit_price or 0),
                status=str(resp.get("status") or "filled").lower(),
                raw=resp,
            )
        except Exception as e:
            logger.warning(f"ccxt place_order {symbol} {side} {qty} failed: {e}")
            return OrderResult(None, symbol, side, 0.0, 0.0, "rejected", raw={"error": str(e)})

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        if not self.live_trading:
            return True
        try:
            self.ex.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"ccxt cancel_order failed: {e}")
            return False

    def get_balance(self) -> Balance:
        if not (self.ex.apiKey and self.ex.secret):
            return Balance(quote_asset=self.quote_asset, free=0.0, total=0.0)
        try:
            bal = self.ex.fetch_balance()
            asset = bal.get(self.quote_asset) or {}
            return Balance(
                quote_asset=self.quote_asset,
                free=float(asset.get("free") or 0),
                total=float(asset.get("total") or 0),
            )
        except Exception as e:
            logger.warning(f"ccxt fetch_balance failed: {e}")
            return Balance(quote_asset=self.quote_asset, free=0.0, total=0.0)
