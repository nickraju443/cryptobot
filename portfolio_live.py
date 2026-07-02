"""
portfolio_live.py — REAL portfolio mirrored from andX in real time.

Source of truth = andX `/balance/Main/`. We do NOT store cash or quantities
locally — every status read pulls fresh balances from andX. The local file
`portfolio_live_meta.json` stores ONLY:
  * avg_cost per holding (computed from our fills, since andX doesn't surface it)
  * entries list (timestamp, SL/TP, signal_snapshot, ml_confidence, order_id)
  * closed_trades history (for the learner + PnL accounting)
  * history (audit log of every buy/sell action)

PnL is marked-to-market using LIVE Alpaca prices (USD), bridged to andX's
USDT pairs (USD ≈ USDT). Same approach SRI MATA uses for live equity tracking.

Public API matches portfolio_sim.py so the trading loop is backend-agnostic.

Orders flow through the active exchange client (which is the hybrid → andx
in normal operation). Any non-andX balances (e.g. dormant DOGE acquired via
deposit) appear in the snapshot with `tradable=False` so the bot ignores them.
"""

from __future__ import annotations

import os
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from exchange_client import get_client, OrderResult
from alpaca_crypto_client import AlpacaCryptoClient

logger = logging.getLogger(__name__)

META_FILE = Path(__file__).parent / "portfolio_live_meta.json"
_lock = threading.RLock()

# Use Alpaca for current price quotes — symbols on andX are USDT-quoted but
# Alpaca uses USD. USD ≈ USDT for marking purposes.
_alpaca_for_marks = AlpacaCryptoClient()


def _alpaca_symbol(andx_symbol: str) -> str:
    """Translate 'BTC/USDT' (andX) to 'BTC/USD' (Alpaca) for price marks."""
    base, _, _ = andx_symbol.partition("/")
    return f"{base}/USD"


# ---------- metadata persistence (NOT positions / cash) ----------

def _load_meta() -> dict:
    with _lock:
        if META_FILE.exists():
            with open(META_FILE, "r") as fh:
                return json.load(fh)
        return _new_meta()


def _save_meta(m: dict):
    with _lock:
        with open(META_FILE, "w") as fh:
            json.dump(m, fh, indent=2)


def _new_meta() -> dict:
    m = {
        "version": 1,
        "entries_by_symbol": {},   # symbol -> [entry dict, ...]
        "closed_trades": [],
        "history": [],
        "created": datetime.utcnow().isoformat() + "Z",
        "mode": "live-andx",
    }
    _save_meta(m)
    return m


def reset_portfolio(starting_cash: float = None) -> dict:
    """For a LIVE portfolio, this only resets the LOCAL metadata. The actual
    cash on andX is what it is — withdraw / deposit via andX UI to change it.
    `starting_cash` is ignored (andX balance is the truth)."""
    _new_meta()
    return get_portfolio_summary()


# ---------- helpers ----------

def _andx_client():
    """Return the underlying andx client even when wrapped in hybrid/cache."""
    client = get_client()
    inner = getattr(client, "inner", client)
    if hasattr(inner, "exec_"):
        # HybridClient: execution side is what we want
        return inner.exec_
    # CachedExchangeClient.inner -> direct client
    return inner


def _read_andx_balances() -> dict:
    """Return raw {asset: {balance, available_balance}} from andX /balance/Main/."""
    andx = _andx_client()
    try:
        data = andx._get("/balance/Main/", signed=True)
    except Exception as e:
        logger.warning(f"portfolio_live: could not read andx balance: {e}")
        return {}
    if not data or data.get("status") != "success":
        return {}
    return (data.get("data") or {}).get("balances") or {}


def get_live_price(symbol: str) -> float:
    p = _alpaca_for_marks.get_price(_alpaca_symbol(symbol))
    if p and p > 0:
        return float(p)
    raise ValueError(f"Cannot get Alpaca price for {symbol}")


# ---------- trading: real orders to andX ----------

def buy(symbol: str, qty: float = 0, dollar_amount: float = 0,
        stop_loss: float = 0, take_profit: float = 0,
        signal_snapshot: dict = None, ml_confidence: float = None) -> dict:
    client = get_client()

    ref_price = 0.0
    if dollar_amount > 0:
        try:
            ref_price = get_live_price(symbol)
        except Exception as e:
            return {"error": f"price unavailable: {e}"}
        qty = dollar_amount / ref_price
    if qty <= 0:
        return {"error": "Cannot buy 0 qty"}
    if ref_price <= 0:
        try:
            ref_price = get_live_price(symbol) or 0.0
        except Exception:
            ref_price = 0.0

    # Auto-routes: docs API for BTC/ETH/ANDX1 longs, instant_order (session
    # cookies) for everything else — alts and shorts.
    res: OrderResult = client.place_order_universal(
        symbol=symbol, side="buy", qty=qty, price_hint=ref_price)
    if res.status == "rejected":
        return {"error": f"andX rejected order: {res.raw}"}

    fill_price = res.filled_price
    if not fill_price or fill_price <= 0:
        # Market orders on andX don't return fill price on placement — use live mark
        try:
            fill_price = get_live_price(symbol)
        except Exception:
            fill_price = 0.0

    qty_filled = float(res.qty or qty)
    cost = qty_filled * fill_price

    # Record entry metadata (avg_cost recomputed across all entries for the symbol)
    m = _load_meta()
    entry = {
        "date": datetime.utcnow().isoformat() + "Z",
        "price": fill_price, "qty": qty_filled,
        "stop_loss": stop_loss if stop_loss else None,
        "take_profit": take_profit if take_profit else None,
        "signal_snapshot": signal_snapshot,
        "ml_confidence": ml_confidence,
        "order_id": res.order_id,
        "status": res.status,
    }
    m["entries_by_symbol"].setdefault(symbol, []).append(entry)
    m["history"].append({
        "type": "BUY", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total": cost, "date": entry["date"],
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
        "order_id": res.order_id, "status": res.status,
    })
    _save_meta(m)

    return {
        "action": "BUY", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total_cost": cost,
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
        "order_id": res.order_id, "status": res.status,
    }


def sell(symbol: str, qty: float = 0, sell_all: bool = False,
         signal_snapshot: dict = None, exit_reason: str = None) -> dict:
    client = get_client()
    snap = get_portfolio_summary()
    pos = snap["positions"].get(symbol)
    if not pos or pos["qty"] <= 0:
        return {"error": f"No position in {symbol}"}
    if not pos.get("tradable", True):
        return {"error": f"{symbol} is not tradable on andX (no market exists)"}

    if sell_all:
        qty = pos["qty"]
    if qty <= 0:
        return {"error": "Must sell at least some qty"}
    if qty > pos["qty"] + 1e-9:
        return {"error": f"Only hold {pos['qty']:.8g} of {symbol}"}

    try:
        sell_ref_price = get_live_price(symbol) or 0.0
    except Exception:
        sell_ref_price = 0.0
    res: OrderResult = client.place_order_universal(
        symbol=symbol, side="sell", qty=qty, price_hint=sell_ref_price)
    if res.status == "rejected":
        return {"error": f"andX rejected sell: {res.raw}"}

    fill_price = res.filled_price
    if not fill_price or fill_price <= 0:
        try:
            fill_price = get_live_price(symbol)
        except Exception:
            fill_price = 0.0

    qty_filled = float(res.qty or qty)
    revenue = qty_filled * fill_price
    avg_cost = pos["avg_cost"]
    cost_basis = qty_filled * avg_cost
    pnl = revenue - cost_basis
    pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0

    m = _load_meta()
    closed = {
        "symbol": symbol, "qty": qty_filled,
        "entry_price": avg_cost, "exit_price": fill_price,
        "pnl": pnl, "pnl_pct": pnl_pct,
        "date_sold": datetime.utcnow().isoformat() + "Z",
        "signal_snapshot": signal_snapshot, "side": "long",
        "exit_reason": exit_reason,
        "order_id": res.order_id, "status": res.status,
    }
    m["closed_trades"].append(closed)

    # Remove or reduce entries — quote-alias-aware. The bot writes entries
    # under the SCANNER symbol (BTC/USD) but the SELL is invoked with the
    # ANDX symbol (BTC/USDT). Without alias resolution the entries leak and
    # avg_cost goes stale (ghost positions).
    aliases = [symbol]
    if "/" in symbol:
        base, _, q = symbol.partition("/")
        if q == "USDT": aliases.append(f"{base}/USD")
        elif q == "USD": aliases.append(f"{base}/USDT")
    remaining = qty_filled
    for sym_key in aliases:
        entries = m["entries_by_symbol"].get(sym_key, [])
        while entries and remaining > 1e-12:
            e = entries[0]
            if e["qty"] <= remaining + 1e-12:
                remaining -= e["qty"]
                entries.pop(0)
            else:
                e["qty"] -= remaining
                remaining = 0
        if not entries and sym_key in m["entries_by_symbol"]:
            m["entries_by_symbol"].pop(sym_key, None)
        elif entries:
            m["entries_by_symbol"][sym_key] = entries
        if remaining <= 1e-12:
            break

    m["history"].append({
        "type": "SELL", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total": revenue, "pnl": pnl, "date": closed["date_sold"],
        "exit_reason": exit_reason,
        "order_id": res.order_id,
    })
    _save_meta(m)

    return {
        "action": "SELL", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total_revenue": revenue, "pnl": pnl, "pnl_pct": pnl_pct,
        "order_id": res.order_id, "status": res.status,
    }


# ---------- SHORTING ----------
# andX's documented API is SPOT-ONLY (no margin/perp endpoints), so a SELL
# market order without inventory will be rejected by andX. We still surface
# short attempts so:
#   (a) when andX adds margin endpoints, only client.place_order needs updating
#   (b) the bot's strategy is honestly visible — sim shorts fire and reveal
#       what bidirectional trading would do
# Local meta tracks the synthetic short position; portfolio summary surfaces
# it. If andX rejects the order, the entry is NOT recorded (no fake position).

def short(symbol: str, qty: float = 0, dollar_amount: float = 0,
          stop_loss: float = 0, take_profit: float = 0,
          signal_snapshot: dict = None, ml_confidence: float = None) -> dict:
    client = get_client()

    if dollar_amount > 0:
        try:
            ref_price = get_live_price(symbol)
        except Exception as e:
            return {"error": f"price unavailable: {e}"}
        qty = dollar_amount / ref_price
    if qty <= 0:
        return {"error": "Cannot short 0 qty"}

    # Open short via instant_order (session cookies). The website's SELL
    # button is sell-to-open; that's the only path on andX. Spot /orders/
    # endpoint can't short — place_order_universal automatically picks
    # instant_order for any SELL.
    try:
        short_ref_price = get_live_price(symbol) or 0.0
    except Exception:
        short_ref_price = 0.0
    res: OrderResult = client.place_order_universal(
        symbol=symbol, side="sell", qty=qty, price_hint=short_ref_price)
    if res.status == "rejected":
        return {"error": f"andX rejected short: {res.raw}"}

    fill_price = res.filled_price
    if not fill_price or fill_price <= 0:
        try:
            fill_price = get_live_price(symbol)
        except Exception:
            fill_price = 0.0

    qty_filled = float(res.qty or qty)
    collateral = qty_filled * fill_price

    m = _load_meta()
    entry = {
        "date": datetime.utcnow().isoformat() + "Z",
        "price": fill_price, "qty": qty_filled,
        "stop_loss": stop_loss if stop_loss else None,
        "take_profit": take_profit if take_profit else None,
        "signal_snapshot": signal_snapshot,
        "ml_confidence": ml_confidence,
        "order_id": res.order_id,
        "status": res.status,
        "direction": "short",
    }
    m["entries_by_symbol"].setdefault(symbol, []).append(entry)
    m["history"].append({
        "type": "SHORT", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total": collateral, "date": entry["date"],
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
        "order_id": res.order_id, "status": res.status,
    })
    _save_meta(m)

    return {
        "action": "SHORT", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total_cost": collateral,
        "stop_loss": entry["stop_loss"], "take_profit": entry["take_profit"],
        "order_id": res.order_id, "status": res.status,
    }


def cover(symbol: str, qty: float = 0, sell_all: bool = False,
          signal_snapshot: dict = None, exit_reason: str = None) -> dict:
    """Close a SHORT by sending a BUY market order to andX. PnL is profit
    when fill < entry."""
    client = get_client()
    snap = get_portfolio_summary()
    pos = snap["positions"].get(symbol)
    if not pos:
        return {"error": f"No position in {symbol}"}
    if pos.get("side") != "short":
        return {"error": f"{symbol} is a LONG position — use sell() to close"}

    if sell_all:
        qty = pos["qty"]
    if qty <= 0:
        return {"error": "Must cover at least some qty"}
    if qty > pos["qty"] + 1e-9:
        return {"error": f"Only short {pos['qty']:.8g} of {symbol}"}

    try:
        cover_ref_price = get_live_price(symbol) or 0.0
    except Exception:
        cover_ref_price = 0.0
    res: OrderResult = client.place_order_universal(
        symbol=symbol, side="buy", qty=qty, price_hint=cover_ref_price)
    if res.status == "rejected":
        return {"error": f"andX rejected cover: {res.raw}"}

    fill_price = res.filled_price
    if not fill_price or fill_price <= 0:
        try:
            fill_price = get_live_price(symbol)
        except Exception:
            fill_price = 0.0

    qty_filled = float(res.qty or qty)
    avg_cost = pos["avg_cost"]
    pnl = qty_filled * (avg_cost - fill_price)
    cost_basis = qty_filled * avg_cost
    pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0

    m = _load_meta()
    closed = {
        "symbol": symbol, "qty": qty_filled,
        "entry_price": avg_cost, "exit_price": fill_price,
        "pnl": pnl, "pnl_pct": pnl_pct,
        "date_sold": datetime.utcnow().isoformat() + "Z",
        "signal_snapshot": signal_snapshot, "side": "short",
        "exit_reason": exit_reason,
        "order_id": res.order_id, "status": res.status,
    }
    m["closed_trades"].append(closed)

    # Quote-alias-aware entry removal (BTC/USD entries when sym is BTC/USDT)
    aliases = [symbol]
    if "/" in symbol:
        base, _, q = symbol.partition("/")
        if q == "USDT": aliases.append(f"{base}/USD")
        elif q == "USD": aliases.append(f"{base}/USDT")
    remaining = qty_filled
    for sym_key in aliases:
        entries = m["entries_by_symbol"].get(sym_key, [])
        while entries and remaining > 1e-12:
            e = entries[0]
            if e["qty"] <= remaining + 1e-12:
                remaining -= e["qty"]
                entries.pop(0)
            else:
                e["qty"] -= remaining
                remaining = 0
        if not entries and sym_key in m["entries_by_symbol"]:
            m["entries_by_symbol"].pop(sym_key, None)
        elif entries:
            m["entries_by_symbol"][sym_key] = entries
        if remaining <= 1e-12:
            break

    m["history"].append({
        "type": "COVER", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total": cost_basis + pnl, "pnl": pnl, "date": closed["date_sold"],
        "exit_reason": exit_reason,
        "order_id": res.order_id,
    })
    _save_meta(m)

    return {
        "action": "COVER", "symbol": symbol, "qty": qty_filled, "price": fill_price,
        "total_revenue": cost_basis + pnl, "pnl": pnl, "pnl_pct": pnl_pct,
        "order_id": res.order_id, "status": res.status,
    }


# ---------- monitoring ----------

def check_stop_loss_take_profit() -> list[dict]:
    """Compare live Alpaca prices against the SL/TP we recorded at entry."""
    snap = get_portfolio_summary()
    triggered: list[dict] = []
    for symbol, pos in snap["positions"].items():
        if not pos.get("tradable", True) or pos["qty"] <= 0:
            continue
        price = pos["current_price"]
        sl = pos.get("stop_loss"); tp = pos.get("take_profit")
        if sl and price <= sl:
            triggered.append({"symbol": symbol, "type": "STOP LOSS HIT",
                              "current_price": price, "stop_loss": sl,
                              "entry_price": pos["avg_cost"], "qty": pos["qty"]})
        elif tp and price >= tp:
            triggered.append({"symbol": symbol, "type": "TAKE PROFIT HIT",
                              "current_price": price, "take_profit": tp,
                              "entry_price": pos["avg_cost"], "qty": pos["qty"]})
    return triggered


def get_portfolio_summary() -> dict:
    """Snapshot of LIVE andX state, marked to market with Alpaca prices."""
    andx = _andx_client()
    quote_asset = andx.quote_asset  # "USDT"
    balances = _read_andx_balances()
    m = _load_meta()

    cash = 0.0
    if balances:
        b = balances.get(quote_asset) or {}
        try:
            cash = float(b.get("available_balance") or 0)
        except Exception:
            cash = 0.0

    positions_out: dict = {}
    total_value = cash
    total_unrealized = 0.0

    # Identify non-quote assets with balance > 0
    for asset, info in balances.items():
        if asset == quote_asset:
            continue
        try:
            qty = float(info.get("balance") or 0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue

        # andX symbol form: BASE/USDT (e.g. 'BTC/USDT'). Even if andX doesn't
        # actively trade this market we still surface it so user can see it.
        symbol = f"{asset}/{quote_asset}"

        # Tradable iff andX has a matching market
        try:
            tradable = symbol in andx.get_top_volume_symbols(n=200, exclude_stables=False)
        except Exception:
            tradable = False

        # Live mark via Alpaca (USD pair)
        try:
            mark_price = _alpaca_for_marks.get_price(_alpaca_symbol(symbol))
        except Exception:
            mark_price = None
        if not mark_price or mark_price <= 0:
            mark_price = 0.0

        # Compute avg_cost from local metadata (entries we made via the bot).
        # The bot internally uses Alpaca format (BTC/USD) for symbols while
        # andX surfaces them as BTC/USDT — look up entries under BOTH so we
        # don't lose track of SL/TP just because the quote-asset name differs.
        base = asset
        candidates = [
            symbol,                # e.g. BTC/USDT
            f"{base}/USD",         # Alpaca form (what the bot scans + stores)
            f"{base}/USDT",        # andX form
        ]
        entries = []
        for k in candidates:
            entries = m["entries_by_symbol"].get(k, [])
            if entries:
                break
        if entries:
            tot_qty = sum(e["qty"] for e in entries) or 1.0
            avg_cost = sum(e["price"] * e["qty"] for e in entries) / tot_qty
        else:
            avg_cost = mark_price

        # Direction: andX reports a SHORT as a POSITIVE base-asset balance —
        # identical to a long — so we can't tell from /balance alone. We rely
        # on the bot's own entry metadata: if the entries we recorded for this
        # symbol are shorts, treat the balance as a short position.
        last_entry = entries[-1] if entries else {}
        direction = last_entry.get("direction", "long")

        if direction == "short":
            # Short: profit as price falls. Contribution to equity = collateral
            # (qty*avg_cost) plus unrealized PnL.
            unr = qty * (avg_cost - mark_price)
            cb = qty * avg_cost
            mv = cb + unr
            unr_pct = (unr / cb) * 100 if cb > 0 else 0.0
        else:
            mv = qty * mark_price
            cb = qty * avg_cost
            unr = mv - cb
            unr_pct = (unr / cb) * 100 if cb > 0 else 0.0
        total_value += mv
        total_unrealized += unr

        positions_out[symbol] = {
            "qty": qty, "avg_cost": avg_cost,
            "current_price": mark_price, "market_value": mv,
            "cost_basis": cb,
            "unrealized_pnl": unr, "unrealized_pct": unr_pct,
            "stop_loss": last_entry.get("stop_loss"),
            "take_profit": last_entry.get("take_profit"),
            "side": direction,
            "source": "live-andx",
            "tradable": tradable,
            "pre_existing": len(entries) == 0,
        }

    # Surface SHORT positions tracked only in local meta. andX won't show
    # these in /balance/Main/ (no margin endpoint), so the entries dict is
    # the source of truth. PnL marks against live Alpaca price.
    for sym, entries in m.get("entries_by_symbol", {}).items():
        # Filter to short entries; skip if we've already surfaced this symbol as long
        short_entries = [e for e in entries if e.get("direction") == "short"]
        if not short_entries:
            continue
        if sym in positions_out:
            # andX is also reporting inventory under this symbol — long takes precedence,
            # don't double-surface as a short.
            continue
        sq = sum(e["qty"] for e in short_entries)
        if sq <= 0:
            continue
        sac = sum(e["price"] * e["qty"] for e in short_entries) / sq
        try:
            sp = _alpaca_for_marks.get_price(_alpaca_symbol(sym))
        except Exception:
            sp = sac
        sp = sp or sac
        sunr = sq * (sac - sp)              # short profits as price drops
        scb  = sq * sac
        smv  = scb + sunr                    # contribution to portfolio = collateral + unrealized
        sunr_pct = (sunr / scb) * 100 if scb > 0 else 0.0
        total_value += smv
        total_unrealized += sunr
        last_e = short_entries[-1]
        positions_out[sym] = {
            "qty": sq, "avg_cost": sac,
            "current_price": sp, "market_value": smv,
            "cost_basis": scb,
            "unrealized_pnl": sunr, "unrealized_pct": sunr_pct,
            "stop_loss": last_e.get("stop_loss"),
            "take_profit": last_e.get("take_profit"),
            "side": "short",
            "source": "live-andx",
            "tradable": True,
            "pre_existing": False,
        }

    realized = sum(t["pnl"] for t in m["closed_trades"])
    total_pnl = total_unrealized + realized
    starting_cash = cash + total_value - total_pnl  # informational only
    wins = [t for t in m["closed_trades"] if t["pnl"] > 0]
    losses = [t for t in m["closed_trades"] if t["pnl"] <= 0]
    total = len(m["closed_trades"])
    win_rate = (len(wins) / total * 100) if total > 0 else 0

    return {
        "mode": "live-andx",
        "cash": cash, "positions": positions_out,
        "total_portfolio_value": total_value,
        "starting_cash": starting_cash,
        "total_return_pct": 0.0,  # not meaningful with external deposits/withdrawals
        "unrealized_pnl": total_unrealized,
        "realized_pnl": realized,
        "total_pnl": total_pnl,
        "total_closed_trades": total,
        "win_rate": win_rate,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "closed_trades": m["closed_trades"][-10:],
        "quote_asset": quote_asset,
        "live_trading": True,
    }


def get_trade_history(limit: int = 20) -> list[dict]:
    m = _load_meta()
    return m["history"][-limit:]
