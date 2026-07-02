"""
coin_strategies.py — Per-coin scalp tuning.

Each coin has its own personality: BTC is steady & macro-driven, ETH leads
risk-on/risk-off, SOL is volatile + high-beta to BTC, DOGE/PEPE are pure
volatility plays where wider SL/TP is required. A one-size-fits-all
mode config wastes opportunity on the calm coins and gets stopped out on
the volatile ones.

This module loads a per-coin override sheet from `coin_strategies.json` (or
the built-in defaults below) and provides `get(symbol)` returning a dict
the scan loop merges over the active risk mode. Anything the coin doesn't
override falls through to the active mode value.

Override-able keys: scalp_gate, sl_pct_min, sl_pct_max, tp_pct_min,
tp_pct_max, harvest_threshold, min_hold_seconds, max_position_pct, dip_threshold.

Tuning rationale (defaults below):
  - BTC: tight harvests (+0.8%), tight SL, lower dip threshold (it's the
    most-watched coin; small dips fill fast)
  - ETH: similar to BTC but slightly wider since it moves harder
  - Mid-caps (SOL, AVAX, LINK, ADA, DOT): medium harvests + wider SL
  - Meme/volatile (DOGE, SHIB, PEPE, BONK, FLOKI): wider everything,
    higher dip threshold (need real oversold), bigger TP
  - L2s / fundamentals (MATIC, ARB, OP): medium-tight
  - Default: medium settings
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STRATEGIES_FILE = Path(__file__).parent / "coin_strategies.json"

# Tier descriptors — base coins assigned to a tier inherit the tier's settings.
TIER_BLUE_CHIP = {
    # BTC/ETH: 0.25% taker × 2 = 0.50% round-trip. Harvest at 1.0% NET = 1.5%
    # gross — clears Audit-3 edge floor (≥1.5× RT fee). SL clamp 0.5–1.2% caps
    # the loss tail that drove the bot's -$469 historical drawdown.
    "scalp_gate": 14,             # was 10 — every BTC/ETH trade pays 0.5% RT
    "sl_pct_min": 0.005,          # was 0.006
    "sl_pct_max": 0.012,          # was 0.020 — Audit worst loss was -16.4% BTC
    "tp_pct_min": 0.010,          # was 0.008 — must clear edge floor
    "tp_pct_max": 0.025,          # was 0.030
    "harvest_threshold": 0.010,   # was 0.008 — +1.0% NET = +1.5% gross
    "min_hold_seconds": 5,        # was 15 — let small wins out fast
    "dip_threshold": 55,
}

# ANDX1: ZERO fees on andX. This is the bot's frequency engine — most of the
# 200/day are routed here. Tight SL, fast harvest, smallest min_hold.
# At 55% WR with 0.3% TP / 0.5% SL: EV = 0.55×0.003 - 0.45×0.005 ≈ break-even;
# the plan survives by AVOIDING the loss tail rather than via per-trade edge.
TIER_ZERO_FEE = {
    "scalp_gate": 10,
    "sl_pct_min": 0.004,
    "sl_pct_max": 0.005,          # very tight 0.5% stop
    "tp_pct_min": 0.004,
    "tp_pct_max": 0.015,
    "harvest_threshold": 0.003,   # +0.3% gross (no fee) — fast recycler
    "min_hold_seconds": 3,
    "dip_threshold": 50,
}
TIER_MAJOR = {
    "scalp_gate": 12,
    "sl_pct_min": 0.010,
    "sl_pct_max": 0.030,
    "tp_pct_min": 0.012,
    "tp_pct_max": 0.045,
    "harvest_threshold": 0.012,
    "min_hold_seconds": 20,
    "dip_threshold": 60,
}
TIER_ALT = {
    "scalp_gate": 15,
    "sl_pct_min": 0.012,
    "sl_pct_max": 0.040,
    "tp_pct_min": 0.015,
    "tp_pct_max": 0.060,
    "harvest_threshold": 0.018,
    "min_hold_seconds": 25,
    "dip_threshold": 65,
}
TIER_MEME = {
    "scalp_gate": 20,             # tougher gate — meme dips often = down trend
    "sl_pct_min": 0.020,
    "sl_pct_max": 0.060,
    "tp_pct_min": 0.025,
    "tp_pct_max": 0.080,
    "harvest_threshold": 0.030,
    "min_hold_seconds": 30,
    "dip_threshold": 70,          # need a STRONG oversold signal
}
TIER_DEFAULT = TIER_MAJOR  # unknown coins get major treatment


# Coin → tier map. Add to this file or override via coin_strategies.json.
COIN_TIERS = {
    # Blue chip
    "BTC": TIER_BLUE_CHIP, "ETH": TIER_BLUE_CHIP,
    # Major L1s
    "SOL": TIER_MAJOR, "AVAX": TIER_MAJOR, "BNB": TIER_MAJOR,
    "ADA": TIER_MAJOR, "DOT": TIER_MAJOR, "XRP": TIER_MAJOR,
    "LINK": TIER_MAJOR, "LTC": TIER_MAJOR, "BCH": TIER_MAJOR,
    "ATOM": TIER_MAJOR, "NEAR": TIER_MAJOR, "ALGO": TIER_MAJOR,
    "FIL": TIER_MAJOR, "XLM": TIER_MAJOR, "TRX": TIER_MAJOR,
    # L2s + fundamentals
    "MATIC": TIER_ALT, "ARB": TIER_ALT, "OP": TIER_ALT,
    "UNI": TIER_ALT, "AAVE": TIER_ALT, "SUSHI": TIER_ALT,
    "CRV": TIER_ALT, "MKR": TIER_ALT, "GRT": TIER_ALT,
    "SUI": TIER_ALT, "APT": TIER_ALT, "INJ": TIER_ALT,
    "RUNE": TIER_ALT, "FTM": TIER_ALT, "YFI": TIER_ALT,
    # Memes / high-vol
    "DOGE": TIER_MEME, "SHIB": TIER_MEME, "PEPE": TIER_MEME,
    "BONK": TIER_MEME, "FLOKI": TIER_MEME, "WIF": TIER_MEME,
    # andX native — 0% fees, dedicated zero-fee tier (frequency engine)
    "ANDX1": TIER_ZERO_FEE,
}


# ======================================================================
# Transaction fees
# ======================================================================
# andX taker fees (per /market_info/, confirmed 2026-06-08):
#   BTC/USDT, ETH/USDT  -> 0.25% per side
#   ANDX1/USDT          -> 0% per side
#   USDT/USD            -> 0% per side (not traded by the bot anyway)
# Bot uses MARKET orders, so EVERY fill is a taker. Round-trip = 2 × side.
# round_trip_fee(symbol) returns the cost the bot should subtract from gross
# gain before deciding to harvest/trail/stop — so settings like "harvest at
# +0.8%" mean "net 0.8% in your pocket after fees", not "0.8% gross".
DEFAULT_FEE_PER_SIDE = 0.0025  # 0.25% taker, conservative

FEE_PER_SIDE: dict[str, float] = {
    "BTC": 0.0025,
    "ETH": 0.0025,
    "ANDX1": 0.0,
    "USDT": 0.0,
    # All other coins: assume same as BTC/ETH until andX confirms otherwise.
    # Override here when fee schedule changes (volume discounts, promo, etc.).
}


def fee_per_side(symbol: str) -> float:
    """andX taker fee for one side of a trade on this market (0.0025 = 0.25%)."""
    base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
    return FEE_PER_SIDE.get(base, DEFAULT_FEE_PER_SIDE)


def round_trip_fee(symbol: str) -> float:
    """Combined cost of opening + closing one position on this market.
    Subtract this from any gross-gain measurement to get net pocket return."""
    return 2.0 * fee_per_side(symbol)


_user_overrides: Optional[dict] = None


def _load_overrides() -> dict:
    """Load user overrides from coin_strategies.json. Empty dict if missing."""
    global _user_overrides
    if _user_overrides is not None:
        return _user_overrides
    if not STRATEGIES_FILE.exists():
        _user_overrides = {}
        return _user_overrides
    try:
        _user_overrides = json.loads(STRATEGIES_FILE.read_text(encoding="utf-8"))
        if not isinstance(_user_overrides, dict):
            _user_overrides = {}
    except Exception as e:
        logger.warning(f"coin_strategies: load failed ({e}); using defaults only")
        _user_overrides = {}
    return _user_overrides


def reload() -> None:
    """Force re-read of coin_strategies.json on next get()."""
    global _user_overrides
    _user_overrides = None


def get(symbol: str) -> dict:
    """Return the per-coin override dict for this symbol. Merges user
    overrides on top of the built-in tier defaults so users can tune
    individual coins without rewriting the whole tier."""
    base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
    tier_default = dict(COIN_TIERS.get(base, TIER_DEFAULT))
    overrides = _load_overrides() or {}
    if base in overrides and isinstance(overrides[base], dict):
        tier_default.update(overrides[base])
    return tier_default


def list_all() -> dict:
    """Return {coin: effective_strategy} for every coin we know about. Used
    by the dashboard to show 'here's how each coin is being traded'."""
    out = {}
    overrides = _load_overrides() or {}
    for base in COIN_TIERS.keys():
        out[base] = get(base)
    # Also surface any user-only coins not in our tier map
    for base in overrides.keys():
        if base not in out:
            out[base] = get(base)
    return out


def save_override(coin: str, overrides: dict) -> None:
    """Persist a per-coin tweak. Merges with anything already saved for that
    coin so you can update one key without losing the rest."""
    global _user_overrides
    data = _load_overrides() or {}
    base = coin.upper()
    existing = data.get(base) or {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update({k: v for k, v in overrides.items() if v is not None})
    data[base] = existing
    STRATEGIES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _user_overrides = data


# ======================================================================
# Auto-tuner — apply the BTC formula to ANY coin
# ======================================================================
# The BTC strategy was hand-tuned around three numbers:
#   1. round-trip fee (BTC = 0.5%) — sets the floor that TP must clear
#   2. ATR-%-of-price (BTC ~ 0.6-1.0%) — sizes SL/TP windows
#   3. audit-worst-loss cap (BTC = -16.4% historical) — clamps SL max
# Apply that same formula to any coin and you get a coin-specific tuning
# without hand-writing each one.

# Constants from the BTC audit — keep the formula honest
EDGE_FLOOR_MULTIPLIER = 1.5    # TP must clear 1.5× RT fee (Audit-3 rule)
SL_ATR_MIN = 0.5               # SL min = 0.5 × ATR%
SL_ATR_MAX = 1.5               # SL max = 1.5 × ATR%
TP_ATR_MAX = 3.0               # TP max = 3.0 × ATR%
HARVEST_ATR_FACTOR = 1.0       # Harvest target = 1.0 × ATR% (net of fees)
ABSOLUTE_SL_CAP = 0.06         # Never let SL exceed 6% no matter the coin
ABSOLUTE_TP_CAP = 0.10         # Never let TP exceed 10%


def derive_strategy(
    rt_fee: float,
    atr_pct: float,
    audit_max_loss_pct: float = None,
    win_rate: float = None,
) -> dict:
    """Apply the BTC formula to any coin given its own (fees, volatility, audit
    history). Returns a strategy dict in the same shape as the tier defaults.

      rt_fee             : 0.005 for BTC, 0.0 for ANDX1
      atr_pct            : the coin's 5m ATR as a fraction (e.g. 0.008 = 0.8%)
      audit_max_loss_pct : worst historical -% from trade_history (None = skip cap)
      win_rate           : None or 0.45–0.65 from history (None = neutral)
    """
    # 1. Fee floor — TP minimum must net positive after the round trip
    edge_floor = max(0.003, rt_fee * EDGE_FLOOR_MULTIPLIER)

    # 2. SL — sized by ATR, capped by audit-worst-loss + absolute safety
    sl_min = max(0.003, atr_pct * SL_ATR_MIN)
    sl_max = max(sl_min * 1.3, atr_pct * SL_ATR_MAX)  # always wider than sl_min
    if audit_max_loss_pct is not None and audit_max_loss_pct > 0:
        # Audit cap: never let a single loser exceed historical worst (in absolute)
        # But keep at least 1.3× sl_min — otherwise the audit cap collapses
        # the entire SL band into a single point.
        sl_max = max(sl_min * 1.3, min(sl_max, audit_max_loss_pct * 0.8))
    sl_max = min(sl_max, ABSOLUTE_SL_CAP)

    # 3. TP — must clear fee floor, sized by ATR
    tp_min = max(edge_floor, atr_pct * 0.8)
    tp_max = min(atr_pct * TP_ATR_MAX, ABSOLUTE_TP_CAP)
    # Invariant: tp_min ≤ tp_max ≤ ABSOLUTE_TP_CAP.
    # If a huge fee or floor pushes tp_min beyond the absolute cap, collapse
    # tp_min down (refuse to trade with a TP target the formula can't actually
    # reach) instead of letting tp_max exceed the safety cap.
    if tp_min > tp_max:
        tp_min = tp_max
    else:
        tp_max = min(max(tp_max, tp_min * 1.5), ABSOLUTE_TP_CAP)

    # 4. Harvest — net target = 1 ATR of edge after fees
    # Harvest threshold is GROSS (until app.py subtracts fees), so compose
    # both: the move needed = ATR (target) + RT fee (cost).
    harvest = max(edge_floor, atr_pct * HARVEST_ATR_FACTOR + rt_fee)

    # 5. Scalp gate — tougher for fee-paying coins, looser for zero-fee
    scalp_gate = 10 if rt_fee == 0 else 14
    # Bump by win-rate signal: bad recent record → demand stronger setups
    if win_rate is not None and win_rate < 0.45:
        scalp_gate = min(25, scalp_gate + 4)

    # 6. Min hold — short for zero-fee (recycle fast), longer for volatile coins
    if rt_fee == 0:
        min_hold = 3
    elif atr_pct >= 0.03:
        min_hold = 25       # meme territory — let noise settle
    elif atr_pct >= 0.015:
        min_hold = 15
    else:
        min_hold = 5

    # 7. Dip threshold — easier on stable coins, stricter on volatile ones
    if atr_pct >= 0.03:
        dip_threshold = 70    # meme coins: need REAL oversold to risk a dip
    elif atr_pct >= 0.015:
        dip_threshold = 60
    else:
        dip_threshold = 55

    return {
        "scalp_gate": int(scalp_gate),
        "sl_pct_min": round(sl_min, 5),
        "sl_pct_max": round(sl_max, 5),
        "tp_pct_min": round(tp_min, 5),
        "tp_pct_max": round(tp_max, 5),
        "harvest_threshold": round(harvest, 5),
        "min_hold_seconds": int(min_hold),
        "dip_threshold": int(dip_threshold),
    }


def _measure_atr_pct(symbol: str, timeframe: str = "1h", limit: int = 168) -> float:
    """Pull recent candles via the hybrid client and compute a robust ATR-%-of-price.

    Method: 75th-percentile of (high - low)/close over the last 168 1-hour bars
    (= 7 days). Why 75th percentile + 1h:
      - 5-minute bars during quiet hours can underreport real volatility; the
        bot still needs SL/TP wide enough for normal swings.
      - 1h bars over a week capture both quiet AND active sessions, giving
        a more representative volatility number.
      - 75th-pct (vs median) biases toward the "active" moments so live SL
        doesn't get clipped by every micro-spike during normal volatility.
    Floor of 0.4% — even the calmest coin's SL shouldn't be tighter than that
    or noise will stop every position out.
    """
    try:
        from hybrid_client import HybridClient
        hc = HybridClient.from_env()
        df = None
        for sym in (symbol, symbol.replace("/USDT", "/USD"), symbol.replace("/USD", "/USDT")):
            try:
                df = hc.get_candles(sym, timeframe=timeframe, limit=limit)
                if df is not None and len(df) > 10:
                    break
            except Exception:
                continue
        if df is None or len(df) < 10:
            return 0.01
        hi = df["High"] if "High" in df.columns else df["high"]
        lo = df["Low"]  if "Low"  in df.columns else df["low"]
        cl = df["Close"] if "Close" in df.columns else df["close"]
        ranges = (hi - lo) / cl
        # 75th percentile catches the active-session moves, not just the median quiet hour
        p75 = float(ranges.quantile(0.75))
        # Floor: even the calmest coin needs at least 0.4% ATR for sane stops
        return max(0.004, min(0.10, p75))
    except Exception:
        return 0.01


def _measure_audit_loss_pct(symbol: str) -> float:
    """Walk trade_history.json and return the worst historical loss % for THIS
    coin's base asset. Used to cap SL_max so the bot never repeats a 16% bleed.
    Returns None if no losses recorded."""
    try:
        from pathlib import Path
        hist_file = Path(__file__).parent / "trade_history.json"
        if not hist_file.exists():
            return None
        data = json.loads(hist_file.read_text(encoding="utf-8"))
        base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        worst = 0.0
        for t in data if isinstance(data, list) else []:
            sym = (t.get("symbol") or t.get("ticker") or "").upper()
            if sym.split("/")[0] != base:
                continue
            pnl_pct = t.get("pnl_pct") or 0
            if pnl_pct < 0:
                worst = max(worst, abs(pnl_pct) / 100.0)
        return worst if worst > 0 else None
    except Exception:
        return None


def _measure_win_rate(symbol: str, min_trades: int = 20) -> float:
    """Walk trade_history.json and compute the historical win rate for THIS
    coin. Returns None when there aren't enough trades for a meaningful read."""
    try:
        from pathlib import Path
        hist_file = Path(__file__).parent / "trade_history.json"
        if not hist_file.exists():
            return None
        data = json.loads(hist_file.read_text(encoding="utf-8"))
        base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        wins, total = 0, 0
        for t in data if isinstance(data, list) else []:
            sym = (t.get("symbol") or t.get("ticker") or "").upper()
            if sym.split("/")[0] != base:
                continue
            pnl = t.get("pnl") or 0
            total += 1
            if pnl > 0:
                wins += 1
        return (wins / total) if total >= min_trades else None
    except Exception:
        return None


def auto_tune_for_coin(symbol: str) -> dict:
    """Apply the BTC formula to ONE coin, using that coin's actual fee, ATR,
    and historical loss profile. Returns the derived strategy dict and the
    intermediate measurements (for explainability)."""
    rt_fee = round_trip_fee(symbol)
    atr = _measure_atr_pct(symbol)
    audit = _measure_audit_loss_pct(symbol)
    win_rate = _measure_win_rate(symbol)
    strategy = derive_strategy(rt_fee, atr, audit, win_rate)
    return {
        "strategy": strategy,
        "inputs": {
            "rt_fee": round(rt_fee, 5),
            "atr_pct": round(atr, 5),
            "audit_max_loss_pct": round(audit, 5) if audit is not None else None,
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
        },
    }


def auto_tune_universe(persist: bool = True) -> dict:
    """Run the auto-tuner across every coin we know about. When persist=True,
    saves the result to coin_strategies.json so it overrides tier defaults.

    Returns {coin: {"strategy": {...}, "inputs": {...}}}.
    """
    out = {}
    coins = list(COIN_TIERS.keys())
    for base in coins:
        # Use the USDT pair for ATR fetch (matches andX), falls back internally
        result = auto_tune_for_coin(f"{base}/USDT")
        out[base] = result
        if persist:
            save_override(base, result["strategy"])
    # Force the next get() to re-read coin_strategies.json
    reload()
    return out
