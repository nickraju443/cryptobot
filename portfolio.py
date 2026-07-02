"""
portfolio.py — Facade. Routes to portfolio_live (andX-mirrored) when
LIVE_TRADING=1, otherwise to portfolio_sim (paper, Alpaca prices only).

Both backends expose the same public API so the trading loop in app.py
doesn't care which one is active. app.py can also import portfolio_sim
directly to run a parallel sim trader alongside the live one — that's
how the dual-portfolio dashboard works (mirror SRI MATA's sim + Alpaca
side-by-side view).
"""

from __future__ import annotations

import os

LIVE_TRADING = os.environ.get("LIVE_TRADING", "0") == "1"

if LIVE_TRADING:
    from portfolio_live import (
        buy, sell, short, cover, check_stop_loss_take_profit, reset_portfolio,
        get_live_price, get_portfolio_summary, get_trade_history,
        _load_meta as _load,
        _save_meta as _save,
    )
else:
    from portfolio_sim import (
        buy, sell, short, cover, check_stop_loss_take_profit, reset_portfolio,
        get_live_price, get_portfolio_summary, get_trade_history,
        _load, _save,
    )

__all__ = [
    "buy", "sell", "short", "cover", "check_stop_loss_take_profit",
    "reset_portfolio", "get_live_price", "get_portfolio_summary",
    "get_trade_history", "_load", "_save", "LIVE_TRADING",
]
