"""
portfolio_sim.py — Pure simulated portfolio. NEVER touches andX.

Uses live Alpaca crypto prices for mark-to-market and fill simulation, exactly
like SRI MATA's sim portfolio uses Alpaca SIP prices. Independent JSON state
in `portfolio_sim.json` so it can run side-by-side with the live andX mirror.

The bot's "sim trader" thread uses this; the dashboard shows it in a separate
panel from the live andX view.

Public API mirrors portfolio_live.py / portfolio.py so the trading loops can be
backend-agnostic:
  buy / sell
  check_stop_loss_take_profit
  get_portfolio_summary
  get_trade_history
  reset_portfolio
"""

from __future__ import annotations

import os
import json
import threading
from datetime import datetime
from pathlib import Path

# Prices come from Alpaca crypto (the data side). NEVER from andX.
from alpaca_crypto_client import AlpacaCryptoClient

PORTFOLIO_FILE = Path(__file__).parent / "portfolio_sim.json"
DEFAULT_CASH = float(os.environ.get("SIM_STARTING_CASH", os.environ.get("STARTING_CASH", "100000.00")))
QUOTE_ASSET = os.environ.get("SIM_QUOTE_ASSET", "USD")  # Alpaca quotes in USD

_lock = threading.RLock()
_alpaca = AlpacaCryptoClient()


# ---------- persistence ----------

def _load() -> dict:
    with _lock:
        if PORTFOLIO_FILE.exists():
            with open(PORTFOLIO_FILE, "r") as fh:
                return json.load(fh)
        return _new()


def _save(p: dict):
    with _lock:
        with open(PORTFOLIO_FILE, "w") as fh:
            json.dump(p, fh, indent=2)


def _new() -> dict:
    p = {
        "starting_cash": DEFAULT_CASH,
        "cash": DEFAULT_CASH,
        "positions": {},
        "closed_trades": [],
        "history": [],
        "created": datetime.utcnow().isoformat() + "Z",
        "quote_asset": QUOTE_ASSET,
        "mode": "sim",
    }
    _save(p)
    return p


def reset_portfolio(starting_cash: float = None) -> dict:
    global DEFAULT_CASH
    if starting_cash is not None:
        DEFAULT_CASH = float(starting_cash)
    return _new()


# ---------- price (Alpaca only) ----------

def get_live_price(symbol: str) -> float:
    p = _alpaca.get_price(symbol)
    if p and p > 0:
        return float(p)
    raise ValueError(f"Cannot get Alpaca price for {symbol}")


# ---------- trading (simulated fills at live Alpaca price) ----------

def buy(symbol: str, qty: float = 0, dollar_amount: float = 0,
        stop_loss: float = 0, take_profit: float = 0,
        signal_snapshot: dict = None, ml_confidence: float = None) -> dict:
    p = _load()
    try:
        price = get_live_price(symbol)
    except Exception as e:
        return {"error": str(e)}

    if dollar_amount > 0:
        qty = dollar_amount / price
    if qty <= 0:
        return {"error": "Cannot buy 0 qty"}

    cost = qty * price
    if cost > p["cash"]:
        return {"error": f"Not enough cash: need ${cost:,.2f}, have ${p['cash']:,.2f}"}

    p["cash"] -= cost
    entry = {
        "date": datetime.utcnow().isoformat() + "Z",
        "price": price, "qty": qty,
        "stop_loss": stop_loss if stop_loss else None,
        "take_profit": take_profit if take_profit else None,
        "signal_snapshot": signal_snapshot,
        "ml_confidence": ml_confidence,
    }

    if symbol in p["positions"]:
        pos = p["positions"][symbol]
        new_qty = pos["qty"] + qty
        pos["avg_cost"] = (pos["avg_cost"] * pos["qty"] + price * qty) / new_qty if new_qty > 0 else price
        pos["qty"] = new_qty
        pos["entries"].append(entry)
    else:
        p["positions"][symbol] = {"qty": qty, "avg_cost": price, "entries": [entry], "side": "long"}

    p["history"].append({
        "type": "BUY", "symbol": symbol, "qty": qty, "price": price,
        "total": cost, "date": entry["date"],
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
    })
    _save(p)
    return {
        "action": "BUY", "symbol": symbol, "qty": qty, "price": price,
        "total_cost": cost, "remaining_cash": p["cash"],
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
    }


def sell(symbol: str, qty: float = 0, sell_all: bool = False,
         signal_snapshot: dict = None, exit_reason: str = None) -> dict:
    """Close a LONG position. For closing shorts, use cover().
    exit_reason is accepted for API parity with portfolio_live; not stored
    in sim mode today but won't crash if the trader passes it."""
    p = _load()
    if symbol not in p["positions"]:
        return {"error": f"No position in {symbol}"}
    pos = p["positions"][symbol]
    if pos.get("side", "long") != "long":
        return {"error": f"{symbol} is a SHORT position — use cover() to close"}
    try:
        price = get_live_price(symbol)
    except Exception as e:
        return {"error": str(e)}

    if sell_all:
        qty = pos["qty"]
    if qty <= 0:
        return {"error": "Must sell at least some qty"}
    if qty > pos["qty"] + 1e-12:
        return {"error": f"Only hold {pos['qty']:.8g} of {symbol}"}

    revenue = qty * price
    cost_basis = qty * pos["avg_cost"]
    pnl = revenue - cost_basis
    pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0

    p["cash"] += revenue
    closed = {
        "symbol": symbol, "qty": qty,
        "entry_price": pos["avg_cost"], "exit_price": price,
        "pnl": pnl, "pnl_pct": pnl_pct,
        "date_sold": datetime.utcnow().isoformat() + "Z",
        "signal_snapshot": signal_snapshot, "side": "long",
        "exit_reason": exit_reason,
    }
    p["closed_trades"].append(closed)

    pos["qty"] -= qty
    if pos["qty"] <= 1e-12:
        del p["positions"][symbol]
    else:
        p["positions"][symbol] = pos

    p["history"].append({
        "type": "SELL", "symbol": symbol, "qty": qty, "price": price,
        "total": revenue, "pnl": pnl, "date": closed["date_sold"],
    })
    _save(p)
    return {
        "action": "SELL", "symbol": symbol, "qty": qty, "price": price,
        "total_revenue": revenue, "pnl": pnl, "pnl_pct": pnl_pct,
        "remaining_cash": p["cash"],
    }


# ---------- SHORTING (paper-money only — sim has no margin constraints) ----------

def short(symbol: str, qty: float = 0, dollar_amount: float = 0,
          stop_loss: float = 0, take_profit: float = 0,
          signal_snapshot: dict = None, ml_confidence: float = None) -> dict:
    """Open a SHORT position. Cash is locked as collateral at the entry price;
    PnL accrues as price drops. Mirror of buy() but with side='short' and
    SL placed ABOVE entry / TP placed BELOW entry by the caller."""
    p = _load()
    try:
        price = get_live_price(symbol)
    except Exception as e:
        return {"error": str(e)}

    if dollar_amount > 0:
        qty = dollar_amount / price
    if qty <= 0:
        return {"error": "Cannot short 0 qty"}

    collateral = qty * price
    if collateral > p["cash"]:
        return {"error": f"Not enough collateral: need ${collateral:,.2f}, have ${p['cash']:,.2f}"}

    p["cash"] -= collateral  # lock collateral; released + adjusted for PnL on cover()
    entry = {
        "date": datetime.utcnow().isoformat() + "Z",
        "price": price, "qty": qty,
        "stop_loss": stop_loss if stop_loss else None,
        "take_profit": take_profit if take_profit else None,
        "signal_snapshot": signal_snapshot,
        "ml_confidence": ml_confidence,
        "direction": "short",
    }

    if symbol in p["positions"] and p["positions"][symbol].get("side") == "short":
        pos = p["positions"][symbol]
        new_qty = pos["qty"] + qty
        pos["avg_cost"] = (pos["avg_cost"] * pos["qty"] + price * qty) / new_qty if new_qty > 0 else price
        pos["qty"] = new_qty
        pos["entries"].append(entry)
    elif symbol in p["positions"]:
        return {"error": f"Already hold LONG {symbol} — close before opening short"}
    else:
        p["positions"][symbol] = {"qty": qty, "avg_cost": price, "entries": [entry], "side": "short"}

    p["history"].append({
        "type": "SHORT", "symbol": symbol, "qty": qty, "price": price,
        "total": collateral, "date": entry["date"],
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
    })
    _save(p)
    return {
        "action": "SHORT", "symbol": symbol, "qty": qty, "price": price,
        "total_cost": collateral, "remaining_cash": p["cash"],
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
    }


def cover(symbol: str, qty: float = 0, sell_all: bool = False,
          signal_snapshot: dict = None, exit_reason: str = None) -> dict:
    """Close a SHORT position by buying back at current price.
    Returns the locked collateral PLUS the short's PnL (positive if
    current price is below the average entry price)."""
    p = _load()
    if symbol not in p["positions"]:
        return {"error": f"No position in {symbol}"}
    pos = p["positions"][symbol]
    if pos.get("side", "long") != "short":
        return {"error": f"{symbol} is a LONG position — use sell() to close"}
    try:
        price = get_live_price(symbol)
    except Exception as e:
        return {"error": str(e)}

    if sell_all:
        qty = pos["qty"]
    if qty <= 0:
        return {"error": "Must cover at least some qty"}
    if qty > pos["qty"] + 1e-12:
        return {"error": f"Only short {pos['qty']:.8g} of {symbol}"}

    # Short PnL: profit when exit < entry
    pnl = qty * (pos["avg_cost"] - price)
    cost_basis = qty * pos["avg_cost"]
    pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
    # Return the locked collateral + PnL
    p["cash"] += cost_basis + pnl

    closed = {
        "symbol": symbol, "qty": qty,
        "entry_price": pos["avg_cost"], "exit_price": price,
        "pnl": pnl, "pnl_pct": pnl_pct,
        "date_sold": datetime.utcnow().isoformat() + "Z",
        "signal_snapshot": signal_snapshot, "side": "short",
        "exit_reason": exit_reason,
    }
    p["closed_trades"].append(closed)

    pos["qty"] -= qty
    if pos["qty"] <= 1e-12:
        del p["positions"][symbol]
    else:
        p["positions"][symbol] = pos

    p["history"].append({
        "type": "COVER", "symbol": symbol, "qty": qty, "price": price,
        "total": cost_basis + pnl, "pnl": pnl, "date": closed["date_sold"],
    })
    _save(p)
    return {
        "action": "COVER", "symbol": symbol, "qty": qty, "price": price,
        "total_revenue": cost_basis + pnl, "pnl": pnl, "pnl_pct": pnl_pct,
        "remaining_cash": p["cash"],
    }


# ---------- monitoring (live Alpaca prices) ----------

def check_stop_loss_take_profit() -> list[dict]:
    p = _load()
    triggered = []
    if not p["positions"]:
        return triggered
    prices = _alpaca.get_prices(list(p["positions"].keys()))
    for symbol, pos in list(p["positions"].items()):
        price = prices.get(symbol)
        if not price or price <= 0:
            continue
        for entry in pos["entries"]:
            if entry.get("stop_loss") and price <= entry["stop_loss"]:
                triggered.append({"symbol": symbol, "type": "STOP LOSS HIT",
                                  "current_price": price, "stop_loss": entry["stop_loss"],
                                  "entry_price": entry["price"], "qty": pos["qty"]})
                break
            if entry.get("take_profit") and price >= entry["take_profit"]:
                triggered.append({"symbol": symbol, "type": "TAKE PROFIT HIT",
                                  "current_price": price, "take_profit": entry["take_profit"],
                                  "entry_price": entry["price"], "qty": pos["qty"]})
                break
    return triggered


def get_portfolio_summary() -> dict:
    """Live mark-to-market PnL using Alpaca prices — refreshes every call."""
    p = _load()
    positions_out: dict = {}
    total_value = p["cash"]
    total_unrealized = 0.0

    if p["positions"]:
        try:
            prices = _alpaca.get_prices(list(p["positions"].keys()))
        except Exception:
            prices = {}
        for symbol, pos in p["positions"].items():
            price = prices.get(symbol) or pos["avg_cost"]
            qty = pos["qty"]
            side = pos.get("side", "long")
            cb = qty * pos["avg_cost"]
            if side == "short":
                # Short: profit when price falls. Position contributes
                # collateral + unrealized_pnl to total portfolio value.
                unr = qty * (pos["avg_cost"] - price)
                mv = cb + unr  # collateral + PnL = position's contribution
                unr_pct = (unr / cb) * 100 if cb > 0 else 0.0
            else:
                # Long: standard mark-to-market.
                mv = qty * price
                unr = mv - cb
                unr_pct = (unr / cb) * 100 if cb > 0 else 0.0
            total_value += mv
            total_unrealized += unr
            positions_out[symbol] = {
                "qty": qty, "avg_cost": pos["avg_cost"],
                "current_price": price, "market_value": mv,
                "cost_basis": cb,
                "unrealized_pnl": unr, "unrealized_pct": unr_pct,
                "stop_loss": pos["entries"][-1].get("stop_loss") if pos["entries"] else None,
                "take_profit": pos["entries"][-1].get("take_profit") if pos["entries"] else None,
                "side": side,
                "source": "sim",
            }
    realized = sum(t["pnl"] for t in p["closed_trades"])
    total_pnl = total_unrealized + realized
    total_return = ((total_value - p["starting_cash"]) / p["starting_cash"]) * 100 if p["starting_cash"] > 0 else 0
    wins = [t for t in p["closed_trades"] if t["pnl"] > 0]
    losses = [t for t in p["closed_trades"] if t["pnl"] <= 0]
    total = len(p["closed_trades"])
    win_rate = (len(wins) / total * 100) if total > 0 else 0

    return {
        "mode": "sim",
        "cash": p["cash"], "positions": positions_out,
        "total_portfolio_value": total_value,
        "starting_cash": p["starting_cash"],
        "total_return_pct": total_return,
        "unrealized_pnl": total_unrealized,
        "realized_pnl": realized,
        "total_pnl": total_pnl,
        "total_closed_trades": total,
        "win_rate": win_rate,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "closed_trades": p["closed_trades"][-10:],
        "quote_asset": p.get("quote_asset", QUOTE_ASSET),
        "live_trading": False,
    }


def get_trade_history(limit: int = 20) -> list[dict]:
    p = _load()
    return p["history"][-limit:]
