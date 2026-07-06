"""
andx_browser_client.py — BaseExchangeClient adapter that routes orders
through a persistent Chromium session (playwright_session.py).

Why this exists: andX's /p/v1/order/instant_order/ endpoint requires a
JS-derived access-sign that we cannot compute locally (prior brute-force
across 6,000+ HMAC variants produced zero matches). A real, logged-in
Chromium session can sign for free because it's the same code path the
website uses. This adapter is the bridge between the bot's exchange
interface and that browser session.

Read-only operations (get_top_volume_symbols, get_price, get_candles)
still go through the documented REST AndxClient — those endpoints don't
require the website's access-sign and a real HTTP request is cheaper than
driving a browser. So this client is a *thin shim* over two backends:

    write path  : playwright_session.PlaywrightSession
    read path   : andx_client.AndxClient

Order routing
-------------
place_order_universal(symbol, side, qty, price_hint):

    Translates the bot's (symbol, side, qty) into the currency-swap shape
    that instant_order expects, then calls PlaywrightSession.place_order.
    See andx_session.place_instant_order for the swap semantics — we
    preserve them exactly so callers don't have to special-case routes.

    BTC/ETH/ANDX1 longs still route to the documented REST API via the
    internal AndxClient — it's faster and proven to work for those four
    markets. Shorts and all other coins go through the browser.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from exchange_client import BaseExchangeClient, Balance, OrderResult

logger = logging.getLogger(__name__)

# Symbols that the documented REST API supports. Longs on these go through
# the cheap HMAC path. Everything else (shorts, alts) goes browser.
_DOCUMENTED_BASES = {"BTC", "ETH", "ANDX1", "USDT"}


class AndxBrowserClient(BaseExchangeClient):
    """Hybrid exchange adapter: REST reads + Playwright writes."""

    name = "andx_browser"
    quote_asset = "USDT"

    def __init__(self) -> None:
        # Sibling REST client — owns read-side methods AND the docs-API
        # fast-path for BTC/ETH/ANDX1 longs.
        from andx_client import AndxClient
        self._rest = AndxClient()
        self.quote_asset = self._rest.quote_asset
        # Lazy import so the bot doesn't pay the Playwright import cost
        # until someone actually configures HYBRID_EXEC=andx_browser.
        from playwright_session import get_session
        self._pw = get_session()
        # Auto-start the browser session at adapter construction time so
        # the first order doesn't pay a 5-10s warm-up. Headless flag is
        # taken from ANDX_PW_HEADLESS (default: headed for first-run
        # login UX). Idempotent — safe to call multiple times.
        autostart = os.environ.get("ANDX_PW_AUTOSTART", "1") == "1"
        headless = os.environ.get("ANDX_PW_HEADLESS", "0") == "1"
        if autostart:
            try:
                self._pw.start(headless=headless)
            except Exception as e:
                logger.warning(f"AndxBrowserClient: pw autostart failed: {e}")

    # ----- read-side (delegated to REST) ------------------------------

    def get_top_volume_symbols(self, n: int = 30, exclude_stables: bool = True) -> list[str]:
        return self._rest.get_top_volume_symbols(n=n, exclude_stables=exclude_stables)

    def get_price(self, symbol: str) -> Optional[float]:
        return self._rest.get_price(symbol)

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        return self._rest.get_prices(symbols)

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
        return self._rest.get_candles(symbol, timeframe=timeframe, limit=limit)

    # ----- write-side -------------------------------------------------

    def place_order(
        self, symbol, side, qty,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """Generic place_order — route to place_order_universal for the
        bot's standard buy/sell flow. The browser path needs a price hint
        so we look up the current price if the caller didn't pass one."""
        if limit_price is None:
            price_hint = self._rest.get_price(symbol) or 0.0
        else:
            price_hint = float(limit_price)
        return self.place_order_universal(
            symbol=symbol, side=side, qty=qty, price_hint=price_hint,
        )

    def place_order_universal(
        self, symbol: str, side: str, qty: float, price_hint: float,
    ) -> OrderResult:
        """Auto-route:
          * BTC/ETH/ANDX1 longs (buy) → REST docs API (fast, proven)
          * Everything else (alts, all sells/shorts) → browser
        """
        base, _, quote = symbol.partition("/")
        base = base.upper()
        quote = quote.upper() or self.quote_asset

        # Fast path — docs REST for BTC/ETH/ANDX1 buys
        if base in _DOCUMENTED_BASES and side.lower() == "buy":
            try:
                res = self._rest.place_order(symbol=symbol, side=side, qty=qty,
                                             order_type="market")
                # Fall through to browser only if REST gave a hard rejection
                # *not* due to balance (balance-insufficient on a sell means
                # we're trying to short, which docs API can't do — but for
                # a buy, balance-insufficient is fatal and shouldn't retry).
                if res.status != "rejected":
                    return res
                err = (res.raw or {}).get("error", "").lower() if res.raw else ""
                if "insufficient" not in err:
                    return res  # genuine rejection, surface as-is
                logger.info(f"andx_browser: REST {side} rejected for {symbol}, "
                            f"falling through to browser")
            except Exception as e:
                logger.warning(f"andx_browser: REST path raised ({e}), trying browser")

        # Browser path — every other case
        return self._browser_order(symbol, side, qty, price_hint)

    def _volume_snapshot(self) -> Optional[dict]:
        """Read the user's current leaderboard volume/rank/award. Returns
        None silently if no URL is configured — volume tracking is opt-in
        via env vars (ANDX_LEADERBOARD_URL + ANDX_LEADERBOARD_EMAIL_FRAGMENT)."""
        url = os.environ.get("ANDX_LEADERBOARD_URL", "").strip()
        if not url:
            return None
        frag = os.environ.get("ANDX_LEADERBOARD_EMAIL_FRAGMENT", "nick").strip()
        try:
            return self._pw.snapshot_leaderboard_volume(url, frag, timeout_s=12.0)
        except Exception as e:
            logger.debug(f"volume snapshot failed: {e}")
            return None

    def _browser_order(
        self, symbol: str, side: str, qty: float, price_hint: float,
    ) -> OrderResult:
        """Translate (symbol, side, qty) into instant_order currency swap
        and execute through the Playwright session."""
        base, _, quote = symbol.partition("/")
        base = base.upper()
        quote = quote.upper() or self.quote_asset

        # andX's instant_order ONLY trades against USDT.
        if quote in ("USD", "USDC", ""):
            quote = "USDT"

        # Currency swap, identical to andx_session.place_instant_order:
        #   buy   = open long / close short  → buy=base, sell=quote
        #   sell  = open short / close long  → buy=quote, sell=base
        if side.lower() == "buy":
            buy_curr, sell_curr = base, quote
            buy_amount = float(qty)
            sell_amount = float(qty) * float(price_hint)
        else:
            buy_curr, sell_curr = quote, base
            buy_amount = float(qty) * float(price_hint)
            sell_amount = float(qty)

        # Bracket the order with a leaderboard volume snapshot so the user
        # can see exactly how much volume each fill adds to their rank.
        # Both snapshots are best-effort and never block the order.
        vol_before = self._volume_snapshot()

        try:
            pw_res = self._pw.place_order(
                buy_currency=buy_curr, sell_currency=sell_curr,
                buy_amount=buy_amount, sell_amount=sell_amount,
                visible_price=float(price_hint),
            )
        except Exception as e:
            return OrderResult(
                order_id=None, symbol=symbol, side=side.lower(),
                qty=0.0, filled_price=0.0, status="rejected",
                raw={"error": str(e),
                     "vol_before": (vol_before or {}).get("volume")},
            )

        vol_after = self._volume_snapshot() if (pw_res and pw_res.ok) else None
        vol_delta = None
        if vol_before and vol_after and vol_before.get("ok") and vol_after.get("ok"):
            try:
                vol_delta = float(vol_after["volume"]) - float(vol_before["volume"])
            except (TypeError, ValueError, KeyError):
                vol_delta = None

        # Build a compact volume audit dict that goes into OrderResult.raw
        # so the trader log shows the before/after numbers.
        volume_audit = {}
        if vol_before and vol_before.get("ok"):
            volume_audit["before"] = {
                "volume": vol_before.get("volume"),
                "rank": vol_before.get("rank"),
                "award": vol_before.get("award"),
            }
        if vol_after and vol_after.get("ok"):
            volume_audit["after"] = {
                "volume": vol_after.get("volume"),
                "rank": vol_after.get("rank"),
                "award": vol_after.get("award"),
            }
        if vol_delta is not None:
            volume_audit["delta"] = round(vol_delta, 4)
        # Estimate the notional traded so the user can compare bot-side
        # notional vs the leaderboard delta and see how andX counts it.
        try:
            notional = float(pw_res.filled_qty or qty) * float(price_hint or 0)
            volume_audit["est_notional_usdt"] = round(notional, 4)
        except (TypeError, ValueError):
            pass

        if pw_res.ok:
            return OrderResult(
                order_id=pw_res.order_id,
                symbol=symbol, side=side.lower(),
                qty=float(pw_res.filled_qty or qty),
                filled_price=float(pw_res.filled_price or price_hint),
                status="submitted" if pw_res.status == "filled" else pw_res.status,
                raw={
                    "route": pw_res.route.value,
                    "http_status": pw_res.http_status,
                    "json": pw_res.raw,
                    "volume_audit": volume_audit,
                },
            )

        return OrderResult(
            order_id=None, symbol=symbol, side=side.lower(),
            qty=0.0, filled_price=0.0, status="rejected",
            raw={
                "route": pw_res.route.value,
                "http_status": pw_res.http_status,
                "error": pw_res.error,
                "json": pw_res.raw,
                "volume_audit": volume_audit,
            },
        )

    # ----- session probes --------------------------------------------

    def instant_session_available(self) -> bool:
        """Browser session counts as 'session available' when it's running
        AND we've confirmed login. Bot uses this to decide whether to
        attempt live orders."""
        try:
            from playwright_session import SessionState
            st = self._pw.status()
            return st.state == SessionState.LOGGED_IN
        except Exception:
            return False

    # ----- order management ------------------------------------------

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancellation still goes through the documented REST API — andX
        accepts API-key auth on /cancel_order/ for any order, regardless
        of how it was placed."""
        return self._rest.cancel_order(order_id, symbol=symbol)

    # ----- balance / connectivity ------------------------------------

    def get_balance(self) -> Balance:
        """Authoritative cash. Try the REST balance first (HMAC-signed, fast).
        BUT andX's documented REST balance is a SEPARATE pool from the website
        wallet: for many accounts it returns success with $0 even though the
        platform wallet (what the logged-in browser sees, and what orders
        actually draw from) is funded. So if REST reports nothing, fall back
        to the browser session, which reads the real platform balance."""
        rest_bal = None
        try:
            rest_bal = self._rest.get_balance()
            if rest_bal and (rest_bal.free > 0 or rest_bal.total > 0):
                return rest_bal
        except Exception as e:
            logger.warning(f"andx_browser: REST balance failed ({e}); "
                           "trying browser")
        # REST returned empty (or failed) — the logged-in browser is the
        # source of truth for the platform wallet the bot actually trades.
        try:
            pw_bal = self._pw.get_balance()
            if pw_bal and (pw_bal.free > 0 or pw_bal.total > 0):
                logger.info(f"andx_browser: using browser balance "
                            f"(REST empty): free={pw_bal.free}")
                return Balance(
                    quote_asset=pw_bal.quote_asset,
                    free=pw_bal.free, total=pw_bal.total,
                )
        except Exception as e:
            logger.error(f"andx_browser: browser balance also failed: {e}")
        # Nothing had funds — return REST's answer (usually zeros).
        return rest_bal or Balance(quote_asset=self.quote_asset, free=0.0, total=0.0)

    def is_connected(self) -> bool:
        """REST + Playwright must both be reachable."""
        try:
            rest_ok = self._rest.is_connected()
        except Exception:
            rest_ok = False
        try:
            from playwright_session import SessionState
            pw_ok = self._pw.status().state in (
                SessionState.LOGGED_IN, SessionState.LOGGED_OUT,
            )
        except Exception:
            pw_ok = False
        return rest_ok and pw_ok

    # ----- escape hatch for portfolio_live._read_andx_balances --------

    def _get(self, path: str, signed: bool = True):
        """portfolio_live._andx_client() reaches in and calls andx._get(
        '/balance/Main/', signed=True). Delegating to our internal REST
        client preserves that path verbatim so balance reads keep working
        even with HYBRID_EXEC=andx_browser."""
        return self._rest._get(path, signed=signed)
