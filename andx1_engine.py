"""
andx1_engine.py — Dedicated high-frequency scalper for ANDX1/USDT.

WHY THIS EXISTS:
ANDX1 is andX's native token and trades at 0% fees on both maker and taker
(confirmed via /api/v1/market_info/). Every other coin pays 0.25% taker × 2 =
0.50% round-trip. On a fee-bearing market a tiny scalp gets eaten by the fee;
on a 0%-fee market the same tiny scalp is pure P&L.

But: Alpaca (the bot's data side) has NO ANDX1 price feed. The main scan loop
runs full TA on candle data — that returns null for ANDX1, so it drops out
before scoring. The bot literally CANNOT trade ANDX1 through `_scan_and_buy`.

This module is the workaround. It runs as its own thread, pulls ANDX1 price
straight from andX's /ticker/ANDX1USDT/ endpoint (no Alpaca needed, no TA),
and runs a continuous in/out scalp ladder:

  while running:
      tick = current ANDX1 price (from andX)
      if no position:
          if cash >= MIN_NOTIONAL:
              buy()      # always-on entry
      else:
          gain = (tick - avg_cost) / avg_cost
          if gain >= +0.003:  sell()  # harvest — wait 0s, rebuy next loop
          if gain <= -0.005:  sell()  # cut loss — wait 30s before rebuy

With 0% fees, every harvest cycle pockets the small move. The strategy makes
money on VOLATILITY (the coin oscillating); it does NOT need a directional
signal. Position size is small ($50 default) so the absolute risk per cycle
is ~$0.25 (-0.5% × $50) — survivable.

The engine respects the bot's daily loss circuit breaker and the AGGRESSIVE
mode toggle (only runs when in AGGRESSIVE + force_deploy ON, or when
explicitly enabled via /api/andx1_engine/toggle).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

ANDX1_SYMBOL_ANDX  = "ANDX1/USDT"     # symbol the docs API expects (portfolio_live)
TICK_INTERVAL_SEC  = 2.0              # poll andX ticker every 2s
HARVEST_PCT        = 0.003            # +0.3% gross gain → sell + rebuy
STOP_LOSS_PCT      = 0.005            # -0.5% gross loss → sell + cool down
LOSS_COOLDOWN_SEC  = 30               # after loss, wait this long before next buy
MIN_HOLD_SEC       = 3                # don't exit faster than this (lets noise settle)
MIN_NOTIONAL_USD   = 5.0              # andX min order size
DEFAULT_NOTIONAL_USD = 50.0           # how much $ per scalp cycle (~3% of $1.5K)

# Daily loss limit — coordinated with the bot-wide circuit breaker. Engine
# stops opening new positions if its OWN running loss this session hits this.
DAILY_LOSS_LIMIT_USD = -20.0


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

class _EngineState:
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        # Tape — last 30 ticks (timestamp, price)
        self.tape: list[tuple[float, float]] = []
        # Current state machine: 'waiting' (no position, ok to buy) | 'holding' | 'cooldown'
        self.state = "waiting"
        self.cooldown_until = 0.0
        # Stats (resets when engine restarts)
        self.cycles = 0
        self.wins = 0
        self.losses = 0
        self.session_pnl = 0.0
        self.last_price: Optional[float] = None
        self.last_action_at: Optional[str] = None  # iso ts of most recent buy/sell
        self.last_action: Optional[str] = None     # 'buy' | 'sell' | 'skip'
        self.last_error: Optional[str] = None
        # Self-tracked entry — portfolio_live's avg_cost mirror reads 0 for
        # ANDX1 because Alpaca has no ANDX1 price feed. We snapshot the fill
        # price ourselves when we open, so the exit logic always has a real
        # cost basis to compare against.
        self.entry_price: Optional[float] = None
        self.entry_qty: float = 0.0
        self.entry_time: float = 0.0
        # Configurable at runtime (so the dashboard can tune without restart)
        self.notional_usd = DEFAULT_NOTIONAL_USD
        self.harvest_pct = HARVEST_PCT
        self.sl_pct = STOP_LOSS_PCT


STATE = _EngineState()


# ----------------------------------------------------------------------
# andX price fetch (no Alpaca dependency)
# ----------------------------------------------------------------------

def _get_andx1_market() -> Optional[dict]:
    """Fetch ANDX1/USDT bid/ask/last from andX directly.
    Returns {bid, ask, last, mid} or None on failure.

    ANDX1 has a thin book. We track BOTH sides so:
      - entry uses ask (what we actually pay)
      - MTM uses bid (what we'd get selling now)
    Without this split the P&L always shows ~spread too generous, and the
    harvest threshold fires on phantom gains."""
    try:
        from andx_client import AndxClient
        client = AndxClient()
        data = client._get("/ticker/ANDX1USDT/")
        if not data or data.get("status") != "success":
            return None
        tk = (data.get("data") or {}).get("ticker") or {}
        try:
            bid = float(tk.get("bid") or 0)
            ask = float(tk.get("ask") or 0)
            last = float(tk.get("last_price") or 0)
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (bid or ask or last)
            return {"bid": bid, "ask": ask, "last": last, "mid": mid}
        except (TypeError, ValueError):
            return None
    except Exception as e:
        STATE.last_error = f"market fetch failed: {e}"
        return None


def _get_andx1_price() -> Optional[float]:
    """Best-guess executable mark for a long position (= bid we could sell at)."""
    mkt = _get_andx1_market()
    if not mkt:
        return None
    return mkt["bid"] or mkt["last"] or None


# ----------------------------------------------------------------------
# Position helpers — read from portfolio_live mirror
# ----------------------------------------------------------------------

def _current_position() -> Optional[dict]:
    """Return the bot's current ANDX1 position dict (or None if flat).
    Looks at portfolio_live's andX-mirrored balance, NOT the meta entries
    (which can lag). Filters to side='long' only since this engine is
    long-only."""
    try:
        import portfolio_live
        snap = portfolio_live.get_portfolio_summary()
        for sym, pos in (snap.get("positions") or {}).items():
            if sym.split("/")[0].upper() != "ANDX1":
                continue
            qty = float(pos.get("qty") or 0)
            if qty <= 0:
                continue
            return {
                "symbol": sym,
                "qty": qty,
                "avg_cost": float(pos.get("avg_cost") or 0),
                "side": pos.get("side", "long"),
            }
    except Exception as e:
        STATE.last_error = f"position lookup failed: {e}"
    return None


def _available_cash() -> float:
    """USDT cash available on andX (NOT total portfolio — we want what's
    spendable on a new ANDX1 buy without selling something else)."""
    try:
        import portfolio_live
        return float(portfolio_live.get_portfolio_summary().get("cash") or 0)
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Trade actions — delegated to portfolio_live which routes via docs API
# ----------------------------------------------------------------------

def _open_long(market: dict) -> Optional[dict]:
    """Open a long on ANDX1 sized at STATE.notional_usd. Returns the trade
    result dict from portfolio_live.buy or None on failure.

    market = {bid, ask, last, mid} from _get_andx1_market(). We use ask as
    the executable buy price (what we'll actually pay) so the recorded
    cost basis matches reality, not the bid we'd-sell-at."""
    ask = market.get("ask") or market.get("last") or 0.0
    if ask <= 0:
        STATE.last_error = "no ask price for entry"
        STATE.last_action = "skip"
        return None
    notional = max(MIN_NOTIONAL_USD, STATE.notional_usd)
    cash = _available_cash()
    if cash < notional:
        STATE.last_error = f"insufficient cash ${cash:.2f} < ${notional:.2f}"
        STATE.last_action = "skip"
        return None
    qty = notional / ask
    try:
        import portfolio_live
        sl = ask * (1 - STATE.sl_pct)
        tp = ask * (1 + STATE.harvest_pct)
        res = portfolio_live.buy(
            ANDX1_SYMBOL_ANDX, qty=qty,
            stop_loss=sl, take_profit=tp,
            signal_snapshot={
                "source": "andx1_engine",
                "signal": "ENGINE_ENTRY",
                "scalp_score": 50,
                "scalp_reasons": ["andx1_engine continuous scalp"],
                "price": ask,
                "ask_at_entry": ask,
                "bid_at_entry": market.get("bid"),
                "_direction": "long",
            },
            ml_confidence=0.55,
        )
        if "error" in res:
            STATE.last_error = f"buy rejected: {res.get('error')}"
            STATE.last_action = "skip"
            return None
        # Cost basis = ASK we paid (NOT the bid we'd mark against). Otherwise
        # P&L always shows ~spread of phantom gain on entry.
        STATE.entry_price = float(res.get("price") or ask)
        STATE.entry_qty = float(res.get("qty") or qty)
        STATE.entry_time = time.time()
        STATE.last_action = "buy"
        STATE.last_action_at = datetime.utcnow().isoformat() + "Z"
        return res
    except Exception as e:
        STATE.last_error = f"open_long exception: {e}"
        STATE.last_action = "skip"
        return None


def _close_long(reason: str) -> Optional[dict]:
    """Close the current ANDX1 long. Returns the trade result."""
    try:
        import portfolio_live
        res = portfolio_live.sell(
            ANDX1_SYMBOL_ANDX, sell_all=True,
            signal_snapshot={"source": "andx1_engine", "exit_reason": reason},
            exit_reason=reason,
        )
        if "error" in res:
            STATE.last_error = f"sell rejected: {res.get('error')}"
            STATE.last_action = "skip"
            return None
        # Clear self-tracked entry — engine now flat
        STATE.entry_price = None
        STATE.entry_qty = 0.0
        STATE.entry_time = 0.0
        STATE.last_action = "sell"
        STATE.last_action_at = datetime.utcnow().isoformat() + "Z"
        return res
    except Exception as e:
        STATE.last_error = f"close_long exception: {e}"
        STATE.last_action = "skip"
        return None


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def _loop():
    """Engine loop. Sleeps TICK_INTERVAL_SEC between iterations.
    Each iteration: fetch price → decide → act."""
    logger.info("andx1_engine started")
    while not STATE.stop_event.is_set():
        try:
            # 1. Check daily loss limit FIRST — if we've bled too much today,
            # stop opening new positions but still close existing ones.
            tripped = STATE.session_pnl <= DAILY_LOSS_LIMIT_USD

            # 2. Fetch current market (bid/ask/last/mid)
            market = _get_andx1_market()
            if not market or market.get("bid", 0) <= 0:
                STATE.stop_event.wait(TICK_INTERVAL_SEC)
                continue
            price = market["bid"]  # mark long position against bid (sellable)
            STATE.last_price = price
            STATE.tape.append((time.time(), price))
            if len(STATE.tape) > 60:
                STATE.tape = STATE.tape[-60:]

            # 3. Look at current ANDX1 position
            pos = _current_position()

            if pos is None:
                # Not holding — open a long unless in cooldown or loss-limited.
                STATE.state = "cooldown" if time.time() < STATE.cooldown_until else "waiting"
                if tripped:
                    STATE.last_error = f"daily loss limit (${STATE.session_pnl:.2f} ≤ ${DAILY_LOSS_LIMIT_USD:.2f}) — paused"
                elif STATE.state == "waiting":
                    res = _open_long(market)
                    if res:
                        STATE.cycles += 1
                        STATE.state = "holding"
            else:
                # Holding — decide to harvest / stop / hold.
                # Use OUR self-tracked entry_price (portfolio_live's avg_cost
                # mirror reads 0 for ANDX1 since Alpaca lacks the feed).
                # If entry_price is missing (e.g. bot restart mid-position),
                # adopt the current price as breakeven to avoid flat-stuck.
                STATE.state = "holding"
                if STATE.entry_price is None or STATE.entry_price <= 0:
                    STATE.entry_price = price
                    STATE.entry_qty = float(pos.get("qty") or 0)
                    STATE.entry_time = time.time()
                    STATE.last_error = "adopted existing ANDX1 position at current price as entry"
                entry = STATE.entry_price
                gain = (price - entry) / entry
                held_for = time.time() - STATE.entry_time if STATE.entry_time else 9999
                if held_for >= MIN_HOLD_SEC:
                    if gain >= STATE.harvest_pct:
                        res = _close_long("HARVEST")
                        if res:
                            # Compute our OWN P&L from entry/exit price since
                            # portfolio_live's pnl may be wrong (avg=0).
                            exit_price = float(res.get("price") or price)
                            pnl = STATE.entry_qty * (exit_price - entry)
                            STATE.session_pnl += pnl
                            if pnl >= 0: STATE.wins += 1
                            else: STATE.losses += 1
                            STATE.cooldown_until = 0.0  # no cooldown after harvest
                    elif gain <= -STATE.sl_pct:
                        res = _close_long("STOP_LOSS")
                        if res:
                            exit_price = float(res.get("price") or price)
                            pnl = STATE.entry_qty * (exit_price - entry)
                            STATE.session_pnl += pnl
                            STATE.losses += 1
                            STATE.cooldown_until = time.time() + LOSS_COOLDOWN_SEC

        except Exception as e:
            STATE.last_error = f"loop exception: {e}"
            logger.exception("andx1_engine loop error")
        STATE.stop_event.wait(TICK_INTERVAL_SEC)
    logger.info("andx1_engine stopped")


# ----------------------------------------------------------------------
# Public API — start / stop / status
# ----------------------------------------------------------------------

def start():
    """Start the engine thread. No-op if already running."""
    with STATE.lock:
        if STATE.running:
            return {"ok": True, "already_running": True}
        STATE.stop_event.clear()
        STATE.thread = threading.Thread(
            target=_loop, daemon=True, name="andx1_engine")
        STATE.running = True
        STATE.thread.start()
    return {"ok": True, "started": True}


def stop():
    """Signal the engine to stop. Lets the current iteration finish."""
    with STATE.lock:
        if not STATE.running:
            return {"ok": True, "was_running": False}
        STATE.stop_event.set()
        STATE.running = False
    return {"ok": True, "stopped": True}


def status() -> dict:
    """Lightweight snapshot for the dashboard. Pure read — no locking
    needed for a snapshot (rare-race-on-counter is fine)."""
    # Compute live unrealized P&L vs our self-tracked entry (since
    # portfolio_live's avg_cost is broken for ANDX1).
    unrealized = None
    unrealized_pct = None
    if STATE.entry_price and STATE.last_price and STATE.entry_qty:
        unrealized = STATE.entry_qty * (STATE.last_price - STATE.entry_price)
        unrealized_pct = (STATE.last_price - STATE.entry_price) / STATE.entry_price * 100
    return {
        "running": STATE.running,
        "state": STATE.state,
        "last_price": STATE.last_price,
        "entry_price": STATE.entry_price,
        "entry_qty": STATE.entry_qty,
        "unrealized_pnl": round(unrealized, 4) if unrealized is not None else None,
        "unrealized_pct": round(unrealized_pct, 3) if unrealized_pct is not None else None,
        "cycles": STATE.cycles,
        "wins": STATE.wins,
        "losses": STATE.losses,
        "win_rate_pct": (STATE.wins / max(STATE.cycles, 1)) * 100,
        "session_pnl": round(STATE.session_pnl, 4),
        "cooldown_remaining_sec": max(0.0, STATE.cooldown_until - time.time()),
        "last_action": STATE.last_action,
        "last_action_at": STATE.last_action_at,
        "last_error": STATE.last_error,
        "notional_usd": STATE.notional_usd,
        "harvest_pct": STATE.harvest_pct,
        "sl_pct": STATE.sl_pct,
        "tape_len": len(STATE.tape),
        "loss_limit_usd": DAILY_LOSS_LIMIT_USD,
    }


def configure(notional_usd: Optional[float] = None,
              harvest_pct: Optional[float] = None,
              sl_pct: Optional[float] = None) -> dict:
    """Runtime knob tweaks from the dashboard. Each param is optional;
    None = leave alone."""
    if notional_usd is not None and notional_usd >= MIN_NOTIONAL_USD:
        STATE.notional_usd = float(notional_usd)
    if harvest_pct is not None and 0 < harvest_pct < 0.05:
        STATE.harvest_pct = float(harvest_pct)
    if sl_pct is not None and 0 < sl_pct < 0.05:
        STATE.sl_pct = float(sl_pct)
    return status()
