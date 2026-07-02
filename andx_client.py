"""
andx_client.py — Real andX exchange adapter (docs.andx.one spec).

Auth scheme (per https://docs.andx.one):
  HMAC-SHA256(secret, apikey + username + passphrase + timestamp + body_json).hexdigest().upper()

Required headers on every signed request:
  ACCESS-USER          your andX username
  ACCESS-PASSPHRASE    the passphrase you set when creating the API key
  ACCESS-TIMESTAMP     str(time.time())
  ACCESS-SIGN          the signature above
  ACCESS-KEY           your API key

Symbols use andX's format internally: "BTCUSDT" (no slash). The translator
to_andx()/from_andx() converts to/from the bot's canonical "BTC/USDT".

Quote asset defaults to USDT. Override with ANDX_QUOTE_ASSET=USD or TRY.

Endpoint paths MUST be slash-terminated per andX docs (e.g. /ticker/ not /ticker).
"""

from __future__ import annotations

import os
import time
import json
import hmac
import hashlib
import logging
import threading
from typing import Optional
from datetime import datetime, timezone

import pandas as pd
import requests

from exchange_client import BaseExchangeClient, Balance, OrderResult

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config (from env / .env)
# ----------------------------------------------------------------------

ANDX_BASE_URL    = os.environ.get("ANDX_BASE_URL", "https://platform.andx.one/api/v1").rstrip("/")
ANDX_API_KEY     = os.environ.get("ANDX_API_KEY", "")
ANDX_API_SECRET  = os.environ.get("ANDX_API_SECRET", "")
ANDX_USERNAME    = os.environ.get("ANDX_USERNAME", "")
ANDX_PASSPHRASE  = os.environ.get("ANDX_PASSPHRASE", "")
ANDX_QUOTE_ASSET = os.environ.get("ANDX_QUOTE_ASSET", "USDT")
ANDX_ACCOUNT     = os.environ.get("ANDX_ACCOUNT", "Main")

# Fiat + stablecoins to exclude when picking the top-volume universe
_STABLES = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDD", "FRAX",
            "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "TRY"}


# ----------------------------------------------------------------------
# Symbol translation
# ----------------------------------------------------------------------

def to_andx(symbol: str, quote: str = ANDX_QUOTE_ASSET) -> str:
    """'BTC/USDT' -> 'BTCUSDT'. Also bridges quote-asset mismatches:
    when the bot uses 'BTC/USD' (Alpaca) but andX trades 'BTCUSDT', this
    rewrites to the andX-configured quote so orders route correctly.
    USD <-> USDT is nearly 1:1 so the swap is safe."""
    if "/" not in symbol:
        return symbol
    base, sym_quote = symbol.split("/", 1)
    if sym_quote != quote:
        return f"{base}{quote}"
    return f"{base}{quote}"


def from_andx(market_code: str, quote: str = ANDX_QUOTE_ASSET) -> str:
    """'BTCUSDT' -> 'BTC/USDT'. Tries USDT then USD then TRY for the quote split."""
    s = market_code.upper()
    for q in (quote, "USDT", "USD", "TRY"):
        if s.endswith(q):
            return f"{s[:-len(q)]}/{q}"
    return market_code


# ----------------------------------------------------------------------
# AndxClient
# ----------------------------------------------------------------------

class AndxClient(BaseExchangeClient):
    """andX execution adapter. Implements BaseExchangeClient.

    Data methods that map cleanly (price, top-volume) hit /ticker/ endpoints.
    `get_candles` is intentionally a no-op — andX has no OHLCV endpoint, so
    use the HybridClient with HYBRID_DATA=alpaca for candle data."""

    name = "andx"

    def __init__(self):
        # Credentials precedence: andx_credentials.json (set via dashboard)
        # → falls back to env vars from .env. Lets users configure either way.
        try:
            import andx_credentials as _cr
            self.api_key    = _cr.get("api_key")    or ANDX_API_KEY
            self.api_secret = _cr.get("api_secret") or ANDX_API_SECRET
            self.username   = _cr.get("username")   or ANDX_USERNAME
            self.passphrase = _cr.get("passphrase") or ANDX_PASSPHRASE
            self.account    = _cr.get("account_name")  or ANDX_ACCOUNT
            self.quote_asset = _cr.get("quote_asset") or ANDX_QUOTE_ASSET
            self.base_url   = (_cr.get("base_url") or ANDX_BASE_URL).rstrip("/")
        except ImportError:
            # Module not present (e.g. shipped without it) — env-only mode
            self.base_url = ANDX_BASE_URL
            self.api_key = ANDX_API_KEY
            self.api_secret = ANDX_API_SECRET
            self.username = ANDX_USERNAME
            self.passphrase = ANDX_PASSPHRASE
            self.quote_asset = ANDX_QUOTE_ASSET
            self.account = ANDX_ACCOUNT
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "CryptoBot/1.0",
                                       "Accept": "application/json"})
        self._lock = threading.Lock()
        if not (self.api_key and self.api_secret):
            logger.warning("andX credentials not set — paste them via the dashboard "
                           "(andX API panel) or add to .env. signed requests will fail.")
        elif not (self.username and self.passphrase):
            logger.warning("ANDX_USERNAME / ANDX_PASSPHRASE not set — signed endpoints will be rejected by andX")

    # ---------- signing -------------------------------------------------

    def _sign_headers(self, body: str = "{}") -> dict:
        ts = str(time.time())
        message = self.api_key + self.username + self.passphrase + ts + (body or "{}")
        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest().upper()
        return {
            "ACCESS-USER": self.username,
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-SIGN": sig,
            "ACCESS-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def _normalize_path(self, path: str) -> str:
        # andX requires a trailing slash on every path
        if not path.startswith("/"):
            path = "/" + path
        if not path.endswith("/"):
            path += "/"
        return path

    # ---------- low-level transport ------------------------------------

    def _get(self, path: str, signed: bool = False) -> Optional[dict]:
        path = self._normalize_path(path)
        url = f"{self.base_url}{path}"
        headers = self._sign_headers("{}") if signed else {}
        try:
            r = self._session.get(url, headers=headers, timeout=5)
            if r.status_code != 200:
                logger.warning(f"andx GET {path} -> {r.status_code}: {r.text[:200]}")
                return None
            return r.json()
        except Exception as e:
            logger.warning(f"andx GET {path} failed: {e}")
            return None

    def _post(self, path: str, body: dict) -> Optional[dict]:
        path = self._normalize_path(path)
        url = f"{self.base_url}{path}"
        body_str = json.dumps(body, separators=(",", ":"), sort_keys=True)
        headers = self._sign_headers(body_str)
        try:
            r = self._session.post(url, headers=headers, data=body_str, timeout=5)
            if r.status_code not in (200, 201):
                # Redact sensitive fields (account, exact volumes) before logging.
                # The market_code + side are kept because they help diagnose
                # the rejection without exposing position size.
                _safe = {k: ("***" if k in ("account_name", "account_number", "client_id", "volume", "price") else v)
                         for k, v in body.items()}
                logger.warning(f"andx POST {path} body={_safe} -> {r.status_code}: {r.text[:400]}")
                # Return the raw text payload so callers can surface it instead of None
                try:
                    return r.json()
                except Exception:
                    return {"status": "http_error", "http_status": r.status_code, "raw_text": r.text[:500]}
            return r.json()
        except Exception as e:
            logger.warning(f"andx POST {path} failed: {e}")
            return {"status": "exception", "error": str(e)}

    def _delete(self, path: str) -> Optional[dict]:
        path = self._normalize_path(path)
        url = f"{self.base_url}{path}"
        headers = self._sign_headers("{}")
        try:
            r = self._session.delete(url, headers=headers, timeout=10)
            if r.status_code not in (200, 204):
                logger.warning(f"andx DELETE {path} -> {r.status_code}: {r.text[:200]}")
                return None
            return r.json() if r.text else {"status": "success"}
        except Exception as e:
            logger.warning(f"andx DELETE {path} failed: {e}")
            return None

    # ---------- public endpoints ---------------------------------------

    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        """andX /ticker/ returns 24h info for all markets. Rank by quote-volume.

        Response shape per docs.andx.one:
          { "status": "success",
            "data": { "ticker": { "BTCUSDT": {...}, "ETHUSDT": {...}, ... } } }
        """
        data = self._get("/ticker/")
        if not data or data.get("status") != "success":
            return []
        tickers_dict = ((data.get("data") or {}).get("ticker")
                        or (data.get("data") or {}).get("tickers")
                        or {})
        if not isinstance(tickers_dict, dict):
            return []
        rows: list[tuple[str, float]] = []
        for market_code, t in tickers_dict.items():
            if not isinstance(t, dict):
                continue
            mc = (t.get("market") or {}).get("market_code") or market_code
            sym = from_andx(mc, self.quote_asset)
            base, _, quote = sym.partition("/")
            if quote != self.quote_asset:
                continue
            if exclude_stables and base in _STABLES:
                continue
            try:
                price = float(t.get("last_price") or 0)
                vol = float(t.get("volume_24h") or 0)
                qv = price * vol
                rows.append((sym, qv))
            except Exception:
                continue
        rows.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in rows[:n]]

    def get_price(self, symbol: str) -> Optional[float]:
        mc = to_andx(symbol)
        data = self._get(f"/ticker/{mc}/")
        if not data or data.get("status") != "success":
            return None
        inner = (data.get("data") or {}).get("ticker") or data.get("data") or {}
        # The single-symbol endpoint may return the ticker directly OR keyed by market_code
        if isinstance(inner, dict) and mc in inner and isinstance(inner[mc], dict):
            inner = inner[mc]
        try:
            return float(inner.get("last_price") or 0) or None
        except Exception:
            return None

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Single /ticker/ call gets everything; filter to the requested set."""
        out: dict[str, float] = {}
        data = self._get("/ticker/")
        if not data or data.get("status") != "success":
            return out
        tickers_dict = ((data.get("data") or {}).get("ticker")
                        or (data.get("data") or {}).get("tickers")
                        or {})
        if not isinstance(tickers_dict, dict):
            return out
        wanted = set(symbols)
        for market_code, t in tickers_dict.items():
            if not isinstance(t, dict):
                continue
            mc = (t.get("market") or {}).get("market_code") or market_code
            sym = from_andx(mc, self.quote_asset)
            if sym not in wanted:
                continue
            try:
                p = float(t.get("last_price") or 0)
                if p > 0:
                    out[sym] = p
            except Exception:
                continue
        return out

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        """andX has no OHLCV/klines endpoint. Use HybridClient with HYBRID_DATA=alpaca
        for candle data (the bot is already wired this way)."""
        logger.debug(f"andx has no candles endpoint — returning empty for {symbol} {timeframe}")
        return pd.DataFrame()

    # ---------- private endpoints --------------------------------------

    def get_balance(self) -> Balance:
        data = self._get(f"/balance/{self.account}/", signed=True)
        if not data or data.get("status") != "success":
            return Balance(quote_asset=self.quote_asset, free=0.0, total=0.0)
        balances = (data.get("data") or {}).get("balances") or {}
        b = balances.get(self.quote_asset) or {}
        try:
            free = float(b.get("available_balance") or 0)
            total = float(b.get("balance") or 0)
            return Balance(quote_asset=self.quote_asset, free=free, total=total)
        except Exception:
            return Balance(quote_asset=self.quote_asset, free=0.0, total=0.0)

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
        """Submit a spot order on the configured account.

        Per docs.andx.one — POST /orders/ with body:
          {
            "order_type": "limit" | "market",
            "market_code": "BTCUSDT",
            "volume": "0.001",                # qty as string
            "buy_sell": "B" | "S",
            "price": "76500.00",              # required for limit
            "client_id": 0,
            "post_only": false,
            "account_name": "Main"
          }
        Response on success: {"status":"success","data":{"order_number": <int>}}
        """
        # Per docs.andx.one Order Properties table: `price` is REQUIRED on every
        # order. Use "0" for market orders (per docs note); actual limit price
        # for limit orders.
        if order_type.lower() == "limit":
            if limit_price is None:
                return OrderResult(None, symbol, side, 0.0, 0.0, "rejected",
                                   raw={"error": "limit_price required for limit order"})
            price_str = f"{limit_price:.8f}"
        else:
            price_str = "0"

        body: dict = {
            "order_type": order_type.lower(),
            "market_code": to_andx(symbol),
            "volume": f"{qty:.8f}",
            "buy_sell": "B" if side.lower() == "buy" else "S",
            "price": price_str,
            "client_id": 0,
            "post_only": False,
            "account_name": self.account,
        }

        data = self._post("/orders/", body)
        if not data or data.get("status") != "success":
            # Surface every field andX gave back so the trader log has the real reason
            reason = (data or {}).get("reason") or (data or {}).get("error") or "unknown"
            status_code = (data or {}).get("status_code")
            raw_text = (data or {}).get("raw_text")
            http_status = (data or {}).get("http_status")
            return OrderResult(
                None, symbol, side, 0.0, 0.0, "rejected",
                raw={
                    "error": reason,
                    "status_code": status_code,
                    "http_status": http_status,
                    "raw_text": raw_text,
                    "sent_body": body,
                    "response": data,
                },
            )

        d = data.get("data") or {}
        order_number = d.get("order_number") or d.get("order_id")
        # andX returns just the order_number on placement — fill details come from
        # /order_status/<id>/ which we poll separately. Assume submitted at limit_price
        # or last known price for market orders (the SL/TP loop will mark-to-market).
        return OrderResult(
            order_id=str(order_number) if order_number else None,
            symbol=symbol, side=side.lower(),
            qty=float(qty),
            filled_price=float(limit_price) if limit_price else 0.0,
            status="submitted",
            raw=data,
        )

    def instant_order(self, buy_currency: str, sell_currency: str,
                      buy_amount: float, sell_amount: float,
                      visible_price: float) -> dict:
        """Place an order via the platform's INSTANT-TRADE endpoint
        (POST /p/v1/order/instant_order/) — the same one the andX web UI uses.
        This reaches the FULL coin universe (not just the 4 documented markets)
        AND supports shorting (sell-to-open).

        Auth is via the user's browser session cookies (loaded by andx_session.py
        from a paste of "Copy as cURL" in DevTools). API-key auth returns 401 on
        this endpoint, so cookies are the only path.

        Currency-swap semantics:
          long XLM:  buy_currency=XLM, sell_currency=USDT
          short XLM: buy_currency=USDT, sell_currency=XLM
        """
        import andx_session
        return andx_session.place_instant_order(
            buy_currency=buy_currency,
            sell_currency=sell_currency,
            buy_amount=buy_amount,
            sell_amount=sell_amount,
            visible_price=visible_price,
        )

    def instant_session_available(self) -> bool:
        """True if a browser session has been pasted in and is on file."""
        import andx_session
        return andx_session.is_session_available()

    def place_order_universal(
        self,
        symbol: str,
        side: str,
        qty: float,
        price_hint: float,
    ) -> OrderResult:
        """One method that picks the best route automatically:

          * BTC/ETH/ANDX1 + a LONG buy  → documented /orders/ endpoint (fast,
            stable, no cookie expiry to worry about).
          * Everything else (alts, all shorts, sell-to-open) → instant_order
            via session cookies. Reaches the full ~120-coin universe.

        side is 'buy' (open long / close short) or 'sell' (open short / close long).
        qty is in base-currency units. price_hint is used as visible_price for
        instant_order and is ignored for the documented endpoint (market order).
        """
        base, _, quote = symbol.partition("/")
        base = base.upper()
        quote = quote.upper() or self.quote_asset

        # Route 1: documented API for the 4 supported markets.
        # - BUY  → opens a long (always supported when balance available)
        # - SELL → closes a long IF inventory exists. Spot API rejects sell-to-
        #   open with "Balance insufficient" (1103) — that's our fallback signal
        #   to route to instant_order for actual shorting.
        # Either way we TRY docs API first for BTC/ETH/ANDX1 since it's the
        # only auth that works today; only fall through on actual rejection.
        DOCUMENTED = {"BTC", "ETH", "ANDX1", "USDT"}
        use_docs_api = base in DOCUMENTED
        if use_docs_api:
            res = self.place_order(symbol=symbol, side=side, qty=qty,
                                   order_type="market")
            if res.status != "rejected":
                return res
            # Docs API rejected — only fall through to instant_order if the
            # reason is "balance insufficient" (= no inventory to sell-to-open,
            # i.e. real shorting). Other rejections (precision, decimals,
            # market closed) shouldn't be retried via the website endpoint.
            err = (res.raw or {}).get("error", "").lower() if res.raw else ""
            if "insufficient" not in err and side.lower() == "buy":
                return res  # genuine buy rejection — surface it
            logger.info(f"docs API {side} rejected for {symbol}; trying instant_order")

        # Route 2: instant_order via session cookies
        if not self.instant_session_available():
            return OrderResult(
                None, symbol, side, 0.0, 0.0, "rejected",
                raw={"error": "no andX session cookies on file — paste curl via dashboard"},
            )

        # andX's instant_order ONLY trades against USDT — never USD/USDC/etc.
        # The scanner uses Alpaca symbols (BTC/USD), so we coerce the quote to
        # USDT before building the currency-swap body. Without this, the
        # endpoint rejects "USD" as an unknown currency.
        if quote in ("USD", "USDC", ""):
            quote = "USDT"

        # Map (side, base) → currency swap
        if side.lower() == "buy":
            buy_curr, sell_curr = base, quote
            buy_amount = qty
            sell_amount = qty * price_hint
        else:  # sell / short
            buy_curr, sell_curr = quote, base
            buy_amount = qty * price_hint
            sell_amount = qty

        res = self.instant_order(
            buy_currency=buy_curr, sell_currency=sell_curr,
            buy_amount=buy_amount, sell_amount=sell_amount,
            visible_price=price_hint,
        )
        if res.get("ok"):
            j = res.get("json") or {}
            d = j.get("data") or {}
            order_number = d.get("order_number") or d.get("order_id") or d.get("id")
            # Treat "ok=True but no order_number" as a REJECTION, not a success.
            # An order without an ID can't be tracked, cancelled, or reconciled
            # — the upstream needs to know the order didn't actually land.
            if not order_number:
                return OrderResult(
                    None, symbol, side, 0.0, 0.0, "rejected",
                    raw={"error": "instant_order: 200 OK but no order_id returned",
                         "response": res},
                )
            return OrderResult(
                order_id=str(order_number),
                symbol=symbol, side=side.lower(),
                qty=float(qty),
                filled_price=float(price_hint),
                status="submitted",
                raw=res,
            )
        return OrderResult(
            None, symbol, side, 0.0, 0.0, "rejected",
            raw=res,
        )

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """POST /cancel_order/<order_id>/ with empty body."""
        data = self._post(f"/cancel_order/{order_id}/", body={})
        if not data:
            return False
        return data.get("status") == "success"

    def get_order_status(self, order_id: str) -> Optional[dict]:
        return self._get(f"/order_status/{order_id}/", signed=True)

    def get_orders(self, state: str = "A", market_code: Optional[str] = None) -> Optional[dict]:
        """state: A=active, F=filled, C=cancelled (per andX enum)."""
        if market_code:
            return self._get(f"/orders/{self.account}/{state}/{to_andx(market_code)}/", signed=True)
        return self._get(f"/orders/{self.account}/{state}/", signed=True)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line: continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    # Re-read after env load
    ANDX_API_KEY     = os.environ.get("ANDX_API_KEY", "")
    ANDX_USERNAME    = os.environ.get("ANDX_USERNAME", "")
    ANDX_PASSPHRASE  = os.environ.get("ANDX_PASSPHRASE", "")

    logging.basicConfig(level=logging.INFO)
    c = AndxClient()
    print(f"base_url:    {c.base_url}")
    print(f"key set:     {bool(c.api_key)}")
    print(f"user set:    {bool(c.username)}")
    print(f"pass set:    {bool(c.passphrase)}")
    print(f"quote asset: {c.quote_asset}")
    print()
    print("--- PUBLIC ---")
    print(f"top 5 by 24h vol: {c.get_top_volume_symbols(n=5)}")
    print(f"BTC/{c.quote_asset} price: {c.get_price(f'BTC/{c.quote_asset}')}")
    print()
    print("--- PRIVATE (needs username + passphrase) ---")
    bal = c.get_balance()
    print(f"balance: {bal}")
