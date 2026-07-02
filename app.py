"""
CryptoBot — 24/7 high-volume crypto scalper.
Sibling to SRI MATA. Same brain (signal -> ML -> Kelly -> execution),
crypto-only, no market hours, plugs into andX (or any exchange via
exchange_client.py).

Run:  python app.py
Open: http://localhost:5001
"""

from __future__ import annotations

import sys
import os
import json
import logging
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")

# Auto-load .env (no extra dependency required)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from flask import Flask, render_template, jsonify, request
import numpy as np

from analysis import full_analysis
from portfolio import (
    buy, sell, short, cover, get_portfolio_summary, get_trade_history,
    check_stop_loss_take_profit, reset_portfolio, get_live_price,
    _load as load_portfolio, _save as save_portfolio,
    LIVE_TRADING,
)
# Always import the sim backend too — it runs as a side-car (separate state,
# Alpaca-only prices, never touches andX). The bot's primary loop uses the
# facade above; the sim mirror records the SAME decisions to paper money.
import portfolio_sim
from screener import scan_market, get_universe
from kelly_sizing import (
    kelly_position_size, calculate_win_loss_stats, asset_class, ASSET_CLASS_MAP,
)
from learner import (
    get_weight_overrides, record_trade, get_learning_summary, reset_learning,
    record_ticker_result, record_ticker_pnl, get_ticker_streak,
    get_favorites, get_ticker_memory_summary, _load_ticker_stats,
    record_trade_context, get_strategy_adjustments,
    get_ticker_win_rate, get_optimal_quality_gate,
)
from ml_engine import predict_win_probability, get_market_regime, get_ml_status, train_model
from exchange_client import get_client

logger = logging.getLogger(__name__)

app = Flask(__name__)
PORT = int(os.environ.get("PORT", "5002"))  # 5000 = SRI MATA, 5001 = Oracle, 5002 = CryptoBot

# ==========================================
# PERSISTENT BOT STATS
# ==========================================
BOT_STATS_FILE = Path(__file__).parent / "bot_stats.json"


def _load_stats() -> dict:
    if BOT_STATS_FILE.exists():
        try:
            with open(BOT_STATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return _new_stats()


def _save_stats(stats: dict):
    fd, tmp = tempfile.mkstemp(dir=str(BOT_STATS_FILE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(stats, f, indent=2)
        if BOT_STATS_FILE.exists():
            BOT_STATS_FILE.unlink()
        os.rename(tmp, str(BOT_STATS_FILE))
    except Exception as e:
        logger.warning(f"bot_stats save failed: {e}")
        try: os.unlink(tmp)
        except Exception: pass
    try:
        with open(str(BOT_STATS_FILE.parent / "bot_stats_backup.json"), "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.warning(f"bot_stats backup save failed: {e}")


def _new_stats() -> dict:
    total_pnl = 0.0; hwm = 0.0
    try:
        with open("bot_stats_backup.json", "r") as f:
            backup = json.load(f)
        total_pnl = backup.get("total_pnl", 0.0)
        hwm = backup.get("high_water_mark", 0.0)
    except Exception:
        pass
    stats = {
        "total_pnl": total_pnl, "total_trades": 0, "total_wins": 0, "total_losses": 0,
        "high_water_mark": hwm, "sessions_count": 0,
        "best_trade": 0.0, "worst_trade": 0.0,
        "created": datetime.utcnow().isoformat() + "Z",
    }
    _save_stats(stats)
    return stats


def _record_total_trade(pnl: float, won: bool):
    stats = _load_stats()
    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)
    stats["total_trades"] += 1
    if won: stats["total_wins"] += 1
    else: stats["total_losses"] += 1
    if stats["total_pnl"] > stats["high_water_mark"]:
        stats["high_water_mark"] = round(stats["total_pnl"], 2)
    if pnl > stats["best_trade"]: stats["best_trade"] = round(pnl, 2)
    if pnl < stats["worst_trade"]: stats["worst_trade"] = round(pnl, 2)
    _save_stats(stats)
    return stats


def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)): return obj
    return str(obj)


# ==========================================
# TRADING ENGINE STATE
# ==========================================

trader = {
    "running": False,
    "paused": False,
    "stop_event": threading.Event(),
    "target": 2000,
    "next_milestone": 2000,
    "session_pnl": 0.0,
    "session_peak": 0.0,
    "session_trades": 0,
    "session_wins": 0,
    "session_losses": 0,
    "guard_mode": False,
    "manual_mode": False,
    # Risk mode is the master knob: CONSERVATIVE | REGULAR | AGGRESSIVE.
    # Every entry/exit threshold reads from RISK_MODES[risk_mode].
    "risk_mode": os.environ.get("DEFAULT_RISK_MODE", "REGULAR").upper(),
    # Direction filter: "both" | "long" | "short". "both" lets the brain take
    # bullish AND bearish setups (like SRI MATA on stocks). Sim shorts always
    # work; andX live shorts may be rejected if no margin endpoint exists.
    "trade_mode": os.environ.get("DEFAULT_TRADE_MODE", "both").lower(),
    # Full-deployment: keep ALL andX USDT working at all times by buying
    # andX-tradable coins as longs even without a bullish signal. User wants
    # the live account fully invested. Risk: buys into downtrends too.
    "force_deploy": os.environ.get("FORCE_DEPLOY", "0") == "1",
    # DIP mode: bot ONLY opens longs on quality dip setups (oversold + at
    # support + bullish reversal candle + uptrend on weekly). Cash sits
    # idle until a real dip appears — opposite of force_deploy. When DIP
    # mode is on, force_deploy is auto-disabled (they contradict). A coin's
    # per-tier dip_threshold (coin_strategies.py) decides what counts.
    "dip_mode": os.environ.get("DIP_MODE", "0") == "1",
    "log": [],
    "status": "Idle",
    "last_scan_time": None,
    "auto_tp_enabled": False,
    "auto_tp_threshold": 100.0,
    "auto_tp_baseline": 0.0,
    "session_start": datetime.utcnow().isoformat() + "Z",
}
if trader["risk_mode"] not in ("CONSERVATIVE", "REGULAR", "AGGRESSIVE"):
    trader["risk_mode"] = "REGULAR"
if trader["trade_mode"] not in ("both", "long", "short"):
    trader["trade_mode"] = "both"
trader_lock = threading.Lock()

# Parallel sim trader — runs separately, never touches andX. Records the
# SAME entries/exits to paper money for comparison.
SIM_PARALLEL = os.environ.get("SIM_PARALLEL", "1") == "1"
trader_sim = {
    "enabled": SIM_PARALLEL,
    "session_pnl": 0.0,
    "session_peak": 0.0,
    "session_trades": 0,
    "session_wins": 0,
    "session_losses": 0,
    "log": [],
    "session_start": datetime.utcnow().isoformat() + "Z",
}
trader_sim_lock = threading.Lock()

# Per-position peak gain tracking (fade protection)
_position_peaks: dict[str, float] = {}
_sim_position_peaks: dict[str, float] = {}
# Re-entry cooldown
_sell_cooldowns: dict[str, tuple[datetime, int]] = {}

# Per-symbol cooldown after exit
_ticker_cooldown: dict[str, datetime] = {}


# ==========================================
# TRADING PARAMETERS (crypto-tuned)
# ==========================================
# These are global infrastructure values that don't change per mode.
CHECK_INTERVAL = 1                 # exit check every 1s
ML_HARD_BLOCK = 0.0                # ML kept for retraining, scalp_gate is the real gate
MAX_ENTRIES_PER_SYMBOL = 1
COOLDOWN_LOSS_SECONDS = 90    # KEEP: revenge-trade guard (Audit: bot's failure mode was buying into downtrends)
COOLDOWN_WIN_SECONDS  = 3     # was 20 — at 200 trades/day reclaims ~30 min of idle window
DAILY_TARGET = float(os.environ.get("DAILY_TARGET", "1000"))
PROTECT_THRESHOLD = 0.8

# Does the EXEC venue support real shorting (margin/borrow/futures)?
# andX is spot-only (confirmed: Buy/Sell only, no leverage), so this is False.
# When False, short signals route to SIM-ONLY (paper money) and the live
# account stays long-only. Flip to True only if the exec venue gains a
# margin endpoint (then wire it in andx_client.place_order).
EXEC_SUPPORTS_SHORT = os.environ.get("EXEC_SUPPORTS_SHORT", "0") == "1"

# Live-tunable: dashboard slider overrides the mode's scalp_gate.
# Can only TIGHTEN — never loosen — the active mode's gate.
# None = no override (use mode's gate as-is).
SCALP_GATE_OVERRIDE: int | None = None

NO_SIGNAL_COUNT = 0
_last_scan_results: list = []


# ==========================================
# RISK MODES (CONSERVATIVE / REGULAR / AGGRESSIVE)
# Crypto-tuned port of SRI MATA's mode system. Each mode is one self-contained
# dict — every parameter the trading loops need is here, nothing scattered.
# Edit the dict to change behavior; no other code edits required.
# ==========================================

RISK_MODES: dict[str, dict] = {
    "CONSERVATIVE": {
        "label": "CONSERVATIVE",
        "description": "Picky entries, few positions, slow hunt. Only premium A-grade setups.",
        "scan_interval": 8,            # slower hunt
        "min_confidence": 70,          # only A-grade analysis signals
        "scalp_gate": 35,              # very high entry bar
        "max_positions": 3,
        "max_per_asset_class": 2,
        "max_position_pct": 0.20,      # 20% max per trade
        "max_opens_per_cycle": 1,      # one careful entry per scan
        "min_order_value": 5.0,
        "size_boost": 0.7,
        # SL/TP clamps (% of entry)
        "sl_pct_min": 0.012,
        "sl_pct_max": 0.025,
        "tp_pct_min": 0.010,
        "tp_pct_max": 0.040,
        # Exit management
        "emergency_stop_pct": 0.04,    # hard cut at -4%
        "min_hold_seconds": 90,
        "harvest_threshold": 0.015,    # +1.5% take
        "dump_bleed_pct": -0.018,      # -1.8% bleed cut
        "trail_be_pct": 0.008,         # move SL to BE at +0.8%
        "trail_lock_pct": 0.015,       # lock at +1.5%
        "fade_threshold": 0.010,
        "fade_drop": 0.25,             # exit if dropped 25% from peak gain
        "reentry_cooldown_seconds": 300,
    },
    "REGULAR": {
        "label": "REGULAR",
        "description": "Balanced — moderate position count and size, normal cadence.",
        "scan_interval": 5,
        "min_confidence": 60,
        "scalp_gate": 22,
        "max_positions": 6,
        "max_per_asset_class": 4,
        "max_position_pct": 0.20,      # 20% per trade — fits 5 positions cleanly
        "max_opens_per_cycle": 2,      # up to 2 entries per scan
        "min_order_value": 5.0,
        "size_boost": 1.0,
        "sl_pct_min": 0.015,
        "sl_pct_max": 0.045,
        "tp_pct_min": 0.008,
        "tp_pct_max": 0.060,
        "emergency_stop_pct": 0.06,
        "min_hold_seconds": 60,
        "harvest_threshold": 0.020,
        "dump_bleed_pct": -0.025,
        "trail_be_pct": 0.010,
        "trail_lock_pct": 0.020,
        "fade_threshold": 0.015,
        "fade_drop": 0.30,
        "reentry_cooldown_seconds": 180,
    },
    "SNIPER": {
        "label": "SNIPER",
        # Conviction mode: 1-2 positions, big size, chunky take-profit. Built
        # around "make money quicker per trade" not "trade often." With 40%
        # max_position_pct on a $1.5K account, each position is ~$600. A 3%
        # TP hit nets ~$15 after fees — beats 50 scalps at $0.30 each.
        "description": "Sniper mode: 1-2 high-conviction trades, large size, chunky TPs. Quality over frequency.",
        "scan_interval": 3,
        "min_confidence": 70,              # A-grade only
        "scalp_gate": 60,                  # premium setups only
        "max_positions": 2,                # 1-2 trades at a time
        "max_per_asset_class": 2,
        "max_position_pct": 0.40,          # 40% per position = $600 on $1.5K
        "max_opens_per_cycle": 1,          # one careful entry per scan
        "min_order_value": 10.0,
        "size_boost": 1.0,
        # SL/TP — wider than HFT so winners have room and losers exit cleanly
        "sl_pct_min": 0.015,               # 1.5% min stop
        "sl_pct_max": 0.030,               # 3% max stop
        "tp_pct_min": 0.025,               # 2.5% min take — clears 0.5% RT fee 5x over
        "tp_pct_max": 0.060,               # 6% max take
        # Exit management — let winners breathe, don't panic out
        "emergency_stop_pct": 0.040,       # hard cut at -4%
        "min_hold_seconds": 60,            # don't whipsaw on noise
        "harvest_threshold": 0.030,        # take wins at +3% net
        "dump_bleed_pct": -0.020,          # bleed cut at -2%
        "trail_be_pct": 0.012,             # move SL to breakeven at +1.2%
        "trail_lock_pct": 0.020,           # lock at +2%
        "fade_threshold": 0.015,
        "fade_drop": 0.30,
        "reentry_cooldown_seconds": 300,   # don't immediately re-fire
        # SNIPER-specific flags read by the trading loop
        "force_max_size": True,            # bypass Kelly's risk-shrink, use full pct
        "ignore_coin_tiers": True,         # don't let TIER_BLUE_CHIP cap our TPs
        "skip_andx1_engine": True,         # don't run the 0%-fee HFT scalper
        "skip_final_notional_cap": True,   # the $100 cap is AGGRESSIVE-only
    },
    "AGGRESSIVE": {
        "label": "AGGRESSIVE",
        # 200-trades/day plan: most volume on ANDX1 (0% fees → 0.3% harvest at
        # 3s holds), thin slice on BTC/ETH gated by an edge floor that rejects
        # any TP target not clearing 1.5× round-trip fees.
        # Audit (June 2026): bot had 22% win rate, profit factor 0.147,
        # avg loss 2× avg win. Loss tail was the killer, not frequency. New
        # SL cap 1.2% + $100 notional cap + -$50 daily circuit breaker cap
        # single-trade damage; harvest defaults at 0.5% NET; coin overrides
        # set ANDX1=0.3%, BTC/ETH=1.0% NET so per-coin EV math survives.
        "description": "HFT mode: 200 trades/day on ANDX1 (0% fees) + selective BTC/ETH (edge floor). Bounded loss via tight SL + circuit breaker.",
        "scan_interval": 1,
        "min_confidence": 40,
        "scalp_gate": 12,                  # was 10 — tighter signals
        "max_positions": 30,               # was 20 — distribute across ANDX1 cycles
        "max_per_asset_class": 25,
        "max_position_pct": 0.06,          # was 0.10 — $90/pos on $1.5K
        "max_opens_per_cycle": 12,         # was 6 — remove per-cycle cap
        "min_order_value": 5.0,
        "size_boost": 1.0,                 # was 1.1 — don't amplify on tight account
        "sl_pct_min": 0.005,               # was 0.010 — tight stops
        "sl_pct_max": 0.012,               # was 0.035 — Audit worst loss -16.4%
        "tp_pct_min": 0.008,
        "tp_pct_max": 0.020,               # was 0.050
        "emergency_stop_pct": 0.020,       # was 0.08 — cap catastrophic loss at 2%
        "min_hold_seconds": 3,             # was 10
        "harvest_threshold": 0.005,        # was 0.012 (per-coin override raises BTC/ETH to 0.010, drops ANDX1 to 0.003)
        "dump_bleed_pct": -0.010,          # was -0.025
        "trail_be_pct": 0.004,             # was 0.008
        "trail_lock_pct": 0.008,           # was 0.015
        "fade_threshold": 0.006,           # was 0.012
        "fade_drop": 0.30,
        "reentry_cooldown_seconds": 0,     # was 10 — let market dictate frequency
    },
}

DEFAULT_RISK_MODE = os.environ.get("DEFAULT_RISK_MODE", "REGULAR").upper()
if DEFAULT_RISK_MODE not in RISK_MODES:
    DEFAULT_RISK_MODE = "REGULAR"


def _mode() -> dict:
    """Return the currently active risk mode's parameter dict.
    All trading-loop knobs read from here so changing modes flips behavior
    atomically without touching globals."""
    with trader_lock:
        m = trader.get("risk_mode") or DEFAULT_RISK_MODE
    return RISK_MODES.get(m, RISK_MODES[DEFAULT_RISK_MODE])


def _active_scalp_gate() -> int:
    """Resolve the effective scalp_gate: max(mode's gate, slider override,
    learner's optimal). The slider can only tighten, never loosen."""
    m = _mode()
    gates = [int(m["scalp_gate"])]
    if SCALP_GATE_OVERRIDE is not None:
        gates.append(int(SCALP_GATE_OVERRIDE))
    try:
        learned = get_optimal_quality_gate()
        if learned:
            gates.append(int(learned))
    except Exception:
        pass
    return max(gates)


# ==========================================
# LOGGING
# ==========================================

def _tlog(type_: str, msg: str):
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": type_,
        "message": msg,
    }
    with trader_lock:
        trader["log"].append(entry)
        trader["log"] = trader["log"][-200:]
    print(f"  [{entry['time']}] [{type_}] {msg}")


# ==========================================
# SCALP SCORING (port of SRI MATA _stock_scalp_score)
# ==========================================

def _scalp_to_win_prob(scalp: float) -> float:
    """Convert scalp_score -> calibrated win probability for Kelly sizing."""
    if scalp is None or scalp < 15: return 0.40
    if scalp < 20: return 0.50
    if scalp < 25: return 0.58
    if scalp < 30: return 0.65
    if scalp < 40: return 0.70
    return 0.72


def _scalp_score(r: dict) -> tuple[int, list[str]]:
    """0-100 mean-reversion scalp score. Same shape as SRI MATA's
    _stock_scalp_score but tuned for crypto: tighter RSI thresholds, looser
    rvol expectations (crypto rvol swings huge), VWAP fade bias."""
    ind = r.get("indicators", {})
    score = 0
    reasons: list[str] = []

    rsi = float(ind.get("rsi") or 50)
    macd_hist = float(ind.get("macd_hist") or 0)
    bb_pctb = float(ind.get("bb_pctb") or 0.5)
    cmf = float(ind.get("cmf") or 0)
    stoch_k = float(ind.get("stoch_k") or 50)
    adx = float(ind.get("adx") or 20)
    price = float(r.get("price") or 0)
    vwap = float(ind.get("vwap") or price or 0)
    ema_9 = float(ind.get("ema_9") or price or 0)
    rvol = float(ind.get("rvol") or 1.0)

    # RSI extremes (mean reversion)
    if rsi > 72:
        score += 22; reasons.append("RSI overbought")
    elif rsi > 62:
        score += 15; reasons.append("RSI high")
    elif rsi < 28:
        score += 18; reasons.append("RSI oversold reversal")
    elif rsi < 35:
        score -= 8

    # MACD histogram momentum
    macd_hist_prev = float(ind.get("macd_hist_prev") or macd_hist)
    hist_decel = macd_hist < macd_hist_prev
    if macd_hist < 0:
        if hist_decel:
            score += 18; reasons.append("MACD bearish accelerating")
        else:
            score += 6; reasons.append("MACD bearish fading")
        if adx > 25: score += 5
    elif macd_hist > 0 and hist_decel:
        score += 8; reasons.append("MACD losing steam")

    # Bollinger
    if bb_pctb > 0.85:
        score += 18; reasons.append("near BB upper")
    elif bb_pctb > 0.65:
        score += 8; reasons.append("BB high half")
    elif bb_pctb < 0.15:
        score -= 8; reasons.append("BB bottom")

    # Money flow
    if cmf < -0.1:
        score += 12; reasons.append("money outflow")
    elif cmf < 0:
        score += 5
    elif cmf > 0.15:
        score -= 8

    # Stochastic
    stoch_d = float(ind.get("stoch_d") or stoch_k)
    stoch_cross_dn = stoch_k < stoch_d
    if stoch_k > 78:
        score += (14 if stoch_cross_dn else 4)
        reasons.append("stoch overbought" + (" cross" if stoch_cross_dn else ""))
    elif stoch_k > 65 and stoch_cross_dn:
        score += 6

    # VWAP fade — extension above VWAP is exhaustion in scalp setups
    if price > 0 and vwap > 0:
        vp = (price - vwap) / vwap
        if 0.003 < vp < 0.025:
            score += 10; reasons.append("above VWAP")
        elif vp < -0.025:
            score -= 5

    # EMA9 extension
    if price > 0 and ema_9 > 0:
        ep = (price - ema_9) / ema_9
        if 0.003 < ep < 0.02:
            score += 8; reasons.append("extended above EMA9")

    # Relative volume — crypto rvol swings big, threshold is what matters
    if rvol >= 2.0:
        score += 12; reasons.append(f"rvol {rvol:.1f}x")
    elif rvol >= 1.2:
        score += 6
    elif rvol < 0.4:
        score -= 5

    # Confluence bonus
    cats = r.get("signal_categories", {})
    bearish_cats = sum(1 for v in cats.values() if isinstance(v, (int, float)) and v < 0)
    if bearish_cats >= 3:
        score += 10; reasons.append(f"{bearish_cats} bearish indicators")

    sig = r.get("signal", "")
    if "STRONG SELL" in sig: score += 25
    elif "SELL" in sig: score += 15
    elif "STRONG BUY" in sig: score += 18  # reversal bait — bigger move ahead
    elif "BUY" in sig: score += 10

    # Higher-timeframe alignment
    wt = r.get("weekly_trend", "N/A")
    mtf = r.get("mtf_aligned", False)
    if mtf:
        score += 15; reasons.append("HTF confirms")
    elif wt == "BEARISH":
        score += 8
    elif wt == "BULLISH":
        score -= 6

    # Candlestick
    candle_data = r.get("candlestick_patterns", {})
    cn = candle_data.get("net_score", 0)
    if cn <= -0.5:
        score += 15; reasons.append("bearish candle")
    elif cn <= -0.2:
        score += 8
    elif cn >= 0.5:
        score -= 12; reasons.append("bullish candle (long bias)")
        score += 18  # net: still positive — flip-side score on reversal candle

    # Trend structure
    trend_data = r.get("trend_structure", {})
    trend = trend_data.get("trend", "UNKNOWN")
    tstr = trend_data.get("strength", 0)
    if trend == "DOWNTREND" and tstr >= 2:
        score += 12; reasons.append(f"downtrend x{tstr}")
    elif trend == "UPTREND" and tstr >= 2:
        score -= 10  # reversion-fading uptrend is risky

    # High analysis confidence bonus
    conf = r.get("confidence", 50)
    if conf >= 70:
        score += 8; reasons.append(f"high conf {conf:.0f}")

    return max(0, min(100, score)), reasons


# ==========================================
# SCAN PARAMETERS (mode-aware)
# ==========================================

def _get_scan_params() -> dict:
    """Pull scan cadence + confidence floor from the active risk mode.
    On a losing streak we step AGGRESSIVE one tier tighter (drawdown
    auto-tighten); CONSERVATIVE/REGULAR are never auto-loosened or
    auto-tightened — the user's explicit choice stands."""
    # Single atomic read of all trader state we need.
    with trader_lock:
        sp = trader["session_pnl"]
        spk = trader["session_peak"]
        mode_key = trader.get("risk_mode") or DEFAULT_RISK_MODE
    m = RISK_MODES.get(mode_key, RISK_MODES[DEFAULT_RISK_MODE])
    interval = m["scan_interval"]
    min_conf = m["min_confidence"]
    drawdown = spk - sp if spk > 0 else 0
    if drawdown > 500 and mode_key == "AGGRESSIVE":
        min_conf = max(min_conf, RISK_MODES["REGULAR"]["min_confidence"])
    # DAILY LOSS CIRCUIT BREAKER — small-account ($1.5K) hard floor at -$50
    # (3.3%). If AGGRESSIVE is running and the day is in this hole, demote to
    # REGULAR for the rest of the session. The 200-trades/day plan assumes
    # the SL tail is bounded; if it isn't (WR < 50%), this catches it before
    # the account bleeds out. Manual flip-back required to re-enable.
    if sp <= -50 and mode_key == "AGGRESSIVE":
        with trader_lock:
            if trader.get("risk_mode") == "AGGRESSIVE":
                trader["risk_mode"] = "REGULAR"
        _tlog("system",
              f"DAILY LOSS LIMIT HIT (session P&L ${sp:.2f} ≤ -$50) "
              f"— AGGRESSIVE → REGULAR. Re-enable manually after reviewing.")
    return {"interval": interval, "min_confidence": min_conf}


# ==========================================
# PRICE STREAM CACHE
# ==========================================
# Fast-tick price store used by GET /api/prices/stream so the dashboard can
# poll every ~300ms without hitting Alpaca on every request. A small
# background thread refreshes the cache every PRICE_REFRESH_SECONDS using
# the held + tradable universe.
PRICE_REFRESH_SECONDS = 1.5
_price_cache: dict[str, dict] = {}  # symbol -> {"price": float, "ts": iso_str}
_price_cache_lock = threading.Lock()


def _refresh_price_cache():
    """Pull a snapshot of prices for held + tradable symbols and update the
    in-process cache. Symbols are stored in BOTH Alpaca form (BTC/USD) and
    andX form (BTC/USDT) so the dashboard's lookup is direction-agnostic."""
    try:
        from alpaca_crypto_client import AlpacaCryptoClient
        a = AlpacaCryptoClient()
        # Build the watch list: held positions + active universe (Alpaca form)
        symbols: set[str] = set()
        try:
            snap = get_portfolio_summary()
            for s in (snap.get("positions") or {}).keys():
                base = s.split("/")[0]
                symbols.add(f"{base}/USD")
        except Exception:
            pass
        try:
            for s in get_universe():
                base = s.split("/")[0]
                symbols.add(f"{base}/USD")
        except Exception:
            pass
        if not symbols:
            return
        prices = a.get_prices(list(symbols))
        now = datetime.utcnow().isoformat() + "Z"
        with _price_cache_lock:
            for sym, px in (prices or {}).items():
                if not px or px <= 0:
                    continue
                # Store under both quote forms so dashboard can look up either.
                base = sym.split("/")[0]
                _price_cache[f"{base}/USD"] = {"price": float(px), "ts": now}
                _price_cache[f"{base}/USDT"] = {"price": float(px), "ts": now}
    except Exception as e:
        logger.debug(f"price cache refresh failed: {e}")


def _price_stream_loop():
    """Background thread: keep _price_cache warm so /api/prices/stream is
    a pure dict-read (sub-millisecond) regardless of Alpaca latency."""
    while not trader["stop_event"].is_set():
        _refresh_price_cache()
        trader["stop_event"].wait(PRICE_REFRESH_SECONDS)


# ==========================================
# EXIT MANAGEMENT
# ==========================================

def _do_sell(symbol: str, qty: float = 0, sell_all: bool = False,
             signal_snapshot: dict = None, exit_reason: str = "MANUAL",
             entry_data: dict = None) -> dict:
    """Close a position. Direction-aware: long → sell(), short → cover().
    Direction is read from the entry metadata (defaults to long for back-compat)."""
    direction = (entry_data or {}).get("direction", "long")
    close_fn = cover if direction == "short" else sell
    # Pass exit_reason through to portfolio_live so it's persisted in the
    # closed_trades + history records — the chart popup surfaces it.
    res = close_fn(symbol, qty=qty, sell_all=sell_all,
                   signal_snapshot=signal_snapshot, exit_reason=exit_reason)
    if "error" in res:
        return res

    pnl = res.get("pnl", 0)
    pnl_pct = res.get("pnl_pct", 0)
    won = pnl > 0

    # Update session
    with trader_lock:
        trader["session_pnl"] += pnl
        trader["session_trades"] += 1
        if won: trader["session_wins"] += 1
        else: trader["session_losses"] += 1
        if trader["session_pnl"] > trader["session_peak"]:
            trader["session_peak"] = trader["session_pnl"]

    # Career stats
    _record_total_trade(pnl, won)

    # Learner
    try:
        record_ticker_result(symbol, won)
        record_ticker_pnl(symbol, pnl, pnl_pct)
        if signal_snapshot:
            record_trade(
                {"pnl": pnl, "pnl_pct": pnl_pct, "ticker": symbol,
                 "buy_price": entry_data.get("price") if entry_data else 0,
                 "sell_price": res.get("price")},
                signal_snapshot.get("signal_categories", {}) if signal_snapshot else {},
            )
        # Deep trade context for ML
        feats = _entry_features_snapshot(symbol, entry_data, signal_snapshot)
        record_trade_context(
            ticker=symbol, side="long",
            entry_price=entry_data.get("price") if entry_data else 0,
            exit_price=res.get("price"),
            pnl=pnl, pnl_pct=pnl_pct,
            shares=res.get("qty", 0),
            exit_reason=exit_reason,
            entry_data=entry_data,
            ml_features=feats,
        )
    except Exception as e:
        logger.debug(f"learner record failed: {e}")

    # Cooldown
    cd = COOLDOWN_WIN_SECONDS if won else COOLDOWN_LOSS_SECONDS
    _sell_cooldowns[symbol] = (datetime.now(), cd)
    _ticker_cooldown[symbol] = datetime.now()
    _position_peaks.pop(symbol, None)

    _tlog("sell" if won else "loss",
          f"{symbol} exit @{res.get('price'):.6g} pnl ${pnl:+.2f} ({pnl_pct:+.2f}%) — {exit_reason}")

    # Mirror exit to sim portfolio (direction-aware close)
    if trader_sim["enabled"]:
        try:
            sim_symbol = symbol.replace("/USDT", "/USD")
            sim_close_fn = portfolio_sim.cover if direction == "short" else portfolio_sim.sell
            sim_res = sim_close_fn(sim_symbol, sell_all=True,
                                    signal_snapshot=signal_snapshot)
            if "error" not in sim_res:
                spnl = sim_res.get("pnl", 0); spnl_pct = sim_res.get("pnl_pct", 0)
                swon = spnl > 0
                with trader_sim_lock:
                    trader_sim["session_pnl"] += spnl
                    trader_sim["session_trades"] += 1
                    if swon: trader_sim["session_wins"] += 1
                    else: trader_sim["session_losses"] += 1
                    if trader_sim["session_pnl"] > trader_sim["session_peak"]:
                        trader_sim["session_peak"] = trader_sim["session_pnl"]
                _sim_position_peaks.pop(sim_symbol, None)
                _slog("sell" if swon else "loss",
                      f"SIM {sim_symbol} exit @{sim_res.get('price'):.6g} "
                      f"pnl ${spnl:+.2f} ({spnl_pct:+.2f}%) — {exit_reason}")
        except Exception as e:
            logger.debug(f"sim mirror sell failed: {e}")

    return res


def _slog(type_: str, msg: str):
    """Log to the SIM trader's own log buffer (separate from live trader log)."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": type_, "message": msg,
    }
    with trader_sim_lock:
        trader_sim["log"].append(entry)
        trader_sim["log"] = trader_sim["log"][-200:]
    print(f"  [SIM][{entry['time']}] [{type_}] {msg}")


def _entry_features_snapshot(symbol: str, entry_data: dict, signal_snapshot: dict) -> dict:
    """Slim feature snapshot for ML training (mirrors ml_engine.NUMERICAL_FEATURES)."""
    if not signal_snapshot:
        return {}
    ind = signal_snapshot.get("indicators", {})
    return {
        "rsi": ind.get("rsi"), "macd": ind.get("macd"),
        "macd_signal": ind.get("macd_signal"), "macd_hist": ind.get("macd_hist"),
        "macd_hist_prev": ind.get("macd_hist_prev"),
        "stoch_k": ind.get("stoch_k"), "stoch_d": ind.get("stoch_d"),
        "adx": ind.get("adx"), "bb_pctb": ind.get("bb_pctb"),
        "cmf": ind.get("cmf"), "rvol": ind.get("rvol"),
        "confidence": signal_snapshot.get("confidence"),
        "scalp_score": signal_snapshot.get("scalp_score"),
        "risk_reward": signal_snapshot.get("risk_reward"),
        "weekly_trend": signal_snapshot.get("weekly_trend"),
        "mtf_aligned": signal_snapshot.get("mtf_aligned"),
        "trend": (signal_snapshot.get("trend_structure") or {}).get("trend", "UNKNOWN"),
        "pattern_names": [p["name"] for p in (signal_snapshot.get("candlestick_patterns") or {}).get("patterns", [])],
        "ticker_sector": asset_class(symbol),
        "ticker_win_rate": get_ticker_win_rate(symbol),
        "ticker_streak": get_ticker_streak(symbol),
        "day_of_week": datetime.utcnow().weekday(),
        "minutes_since_open": (datetime.utcnow().hour * 60 + datetime.utcnow().minute),  # UTC minute-of-day
    }


def _rewrite_last_entry_sl_tp(symbol: str, sl: float, tp: float) -> None:
    """Update the most-recent entry's stop_loss + take_profit in whichever
    backend is active. Used by _do_buy to re-anchor SL/TP to the actual fill
    price after the order completes."""
    if LIVE_TRADING:
        import portfolio_live
        m = portfolio_live._load_meta()
        entries = m.get("entries_by_symbol", {}).get(symbol, [])
        if entries:
            entries[-1]["stop_loss"] = sl
            entries[-1]["take_profit"] = tp
        for h in reversed(m.get("history", [])):
            if h.get("type") == "BUY" and h.get("symbol") == symbol:
                h["stop_loss"] = sl
                h["take_profit"] = tp
                break
        portfolio_live._save_meta(m)
    else:
        p = load_portfolio()
        pos = (p.get("positions") or {}).get(symbol) or {}
        entries = pos.get("entries", [])
        if entries:
            entries[-1]["stop_loss"] = sl
            entries[-1]["take_profit"] = tp
        save_portfolio(p)


def _entries_for(symbol: str) -> list:
    """Return ALL entry metadata for a symbol from whichever backend is active.

    Quote-asset bridge: scanner uses Alpaca symbols (BTC/USD), andX balance
    reports BTC/USDT. We persist entries under whatever symbol was passed to
    portfolio.buy(), then look up by ALL plausible aliases so the exit loop
    can find them regardless of which side asks. Without this, SL/TP/harvest
    silently do nothing because the entry lookup misses by quote-asset name."""
    candidates = [symbol]
    if "/" in symbol:
        base, _, quote = symbol.partition("/")
        if quote == "USDT":
            candidates.append(f"{base}/USD")
        elif quote == "USD":
            candidates.append(f"{base}/USDT")
    if LIVE_TRADING:
        import portfolio_live
        m = portfolio_live._load_meta()
        ebs = m.get("entries_by_symbol", {})
        out = []
        for c in candidates:
            out.extend(ebs.get(c, []))
        # Preserve order: oldest first across aliases (Python sort by date if present)
        try:
            out.sort(key=lambda e: e.get("date") or "")
        except Exception:
            pass
        return out
    p = load_portfolio()
    pos = (p.get("positions") or {}).get(symbol) or {}
    if not pos and "/" in symbol:
        for c in candidates[1:]:
            pos = (p.get("positions") or {}).get(c) or {}
            if pos:
                break
    return list(pos.get("entries", []))


def _last_entry_for(symbol: str) -> dict:
    """Return the most-recent entry metadata for a symbol from whichever
    backend is active. Lets _manage_open_positions stay backend-agnostic."""
    entries = _entries_for(symbol)
    return entries[-1] if entries else {}


def _manage_open_positions():
    """Run every CHECK_INTERVAL seconds. Handles SL/TP/trailing/fade/dump.
    Works against either backend (live mirror or sim) via get_portfolio_summary.
    All thresholds come from the active risk mode (_mode())."""
    try:
        snap = get_portfolio_summary()
    except Exception as e:
        logger.debug(f"manage_open_positions: portfolio summary failed: {e}")
        return
    positions = snap.get("positions", {}) or {}
    if not positions:
        return

    # Identify tradable positions (live backend flags dormant assets like DOGE)
    tradable = {sym: pos for sym, pos in positions.items() if pos.get("tradable", True)}
    if not tradable:
        return

    m = _mode()

    for symbol, pos in list(tradable.items()):
        price = pos.get("current_price") or 0
        if not price or price <= 0:
            continue

        avg_cost = pos.get("avg_cost") or 0
        if avg_cost <= 0:
            continue

        # Pull entry metadata from the appropriate backend
        last_entry = _last_entry_for(symbol)
        # Direction-aware gain: long profits as price rises, short as it falls.
        direction = last_entry.get("direction") or pos.get("side") or "long"
        if direction == "short":
            gain = (avg_cost - price) / avg_cost
        else:
            gain = (price - avg_cost) / avg_cost
        entry_date = last_entry.get("date", "")

        # Min hold
        try:
            held = (datetime.utcnow() - datetime.fromisoformat(entry_date.replace("Z", ""))).total_seconds() if entry_date else 99999
        except Exception:
            held = 99999

        # Track peak
        peak = _position_peaks.get(symbol, gain)
        if gain > peak:
            peak = gain
            _position_peaks[symbol] = peak

        snapshot = last_entry.get("signal_snapshot") or {}

        # PER-COIN OVERRIDES: each coin has its own tier-tuned harvest
        # threshold + min_hold (BTC 0.8%, alt 1.8%, meme 3%). Falls through
        # to the active mode when the coin has no override.
        import coin_strategies as _cs
        _coin_cfg = _cs.get(symbol)
        _harvest = float(_coin_cfg.get("harvest_threshold", m["harvest_threshold"]))
        _min_hold = int(_coin_cfg.get("min_hold_seconds", m["min_hold_seconds"]))

        # NET-OF-FEES gain: andX charges 0.25% taker per side on BTC/ETH (free
        # on ANDX1). Bot uses market orders — every fill is a taker. So a
        # +0.8% gross move is +0.3% net pocket on BTC. All exit decisions
        # below should reflect what you'd ACTUALLY take home, not the gross
        # price move. SL/TP are PRICE checks (set with fees in mind), so they
        # use raw price — but harvest/trail/fade/dump/emergency use net_gain.
        fee_rt = _cs.round_trip_fee(symbol)
        net_gain = gain - fee_rt
        net_peak = peak - fee_rt  # for trail/fade comparisons we keep the
                                   # peak in the same "net" frame

        # Emergency stop — always honored, even before min-hold (net-of-fees)
        if net_gain <= -m["emergency_stop_pct"]:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="EMERGENCY STOP", entry_data=last_entry)
            continue

        if held < _min_hold:
            continue

        # Stop loss / take profit — direction-aware PRICE comparison (raw —
        # SL/TP were set at the price level the bot wanted to exit at).
        sl = last_entry.get("stop_loss"); tp = last_entry.get("take_profit")
        if direction == "short":
            sl_hit = sl and price >= sl
            tp_hit = tp and price <= tp
        else:
            sl_hit = sl and price <= sl
            tp_hit = tp and price >= tp
        if sl_hit:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="STOP LOSS HIT", entry_data=last_entry)
            continue
        if tp_hit:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="TAKE PROFIT HIT", entry_data=last_entry)
            continue

        # Harvest gains — fires when NET (post-fee) gain reaches the
        # per-coin threshold. e.g. BTC harvest=0.8% requires gross 1.3%.
        if net_gain >= _harvest:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="HARVEST", entry_data=last_entry)
            continue

        # Trail to breakeven / lock profit (net-of-fees)
        if net_peak >= m["trail_lock_pct"] and net_gain <= net_peak - 0.005:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="TRAIL LOCK", entry_data=last_entry)
            continue
        if net_peak >= m["trail_be_pct"] and net_gain <= 0:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="TRAIL BREAKEVEN", entry_data=last_entry)
            continue

        # Fade protection: peaked but dropped sharply (net-of-fees)
        if net_peak >= m["fade_threshold"] and net_gain < net_peak * (1 - m["fade_drop"]):
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="FADE PROTECT", entry_data=last_entry)
            continue

        # Dump bleed (net-of-fees — fires earlier than gross would)
        if net_gain <= m["dump_bleed_pct"]:
            _do_sell(symbol, sell_all=True, signal_snapshot=snapshot,
                     exit_reason="DUMP BLEED", entry_data=last_entry)
            continue


# ==========================================
# SCAN AND BUY
# ==========================================

# andX's API-key order endpoint (/api/v1/orders/) supports ONLY these markets.
# Confirmed empirically 2026-05-29: BTC/ETH accepted (BTC filled, ETH gave
# 'Balance insufficient' = valid market), while SOL/DOGE/XRP/LTC/BAT/etc. all
# returned 'Undefined market code' (1417). The full ~120-coin universe is only
# reachable via the UI's /p/v1/instant_order/ (session-cookie auth) — not the
# bot's API key. So the bot trades ONLY these live; everything else → sim.
KNOWN_ANDX_BASES = {
    b.strip().upper() for b in
    os.environ.get("ANDX_TRADABLE_BASES", "BTC,ETH,ANDX1").split(",") if b.strip()
}
# Runtime blacklist: if a "known" base ever gets rejected, drop it for a while.
_andx_untradable: dict[str, float] = {}
_ANDX_UNTRADABLE_TTL = 3600.0  # 1 hour


def _exec_tradable(symbol: str) -> bool:
    """True if the bot should attempt a LIVE andX order for this symbol.

    ATTEMPT-AND-LEARN: defaults to True for any symbol not in the rejection
    cache. andX itself becomes the source of truth — the documented ticker
    API only lists 4 markets but the website trades ~120, so a strict
    allowlist over-prunes and leaves live cash idle. We try every symbol;
    `_note_andx_rejection` caches the bases andX actually refuses ("Undefined
    market code") for 1h so we stop spamming. Transient errors (no session,
    collateral, network) do NOT cache — they may succeed next cycle.
    """
    base = symbol.split("/")[0]
    exp = _andx_untradable.get(base)
    if exp is not None:
        if time.time() < exp:
            return False
        _andx_untradable.pop(base, None)
    return True


def _note_andx_rejection(symbol: str, err: str):
    """Classify an andX order rejection.
      * Market-not-found / not-supported → blacklist the base for 1h.
      * Everything else (price drift, minimum amount, collateral, auth,
        network, no session) → TRANSIENT, don't blacklist."""
    e = (err or "").lower()
    transient = (
        # collateral
        "insufficient" in e or "balance" in e or "1103" in e
        # no session
        or "no andx session" in e or "no session" in e
        or "paste curl" in e or "paste cookies" in e
        # auth
        or "csrf" in e or "401" in e or "403" in e
        # network
        or "timeout" in e or "timed out" in e
        or "connection" in e or "network" in e
        # exchange-side transients (price drift, minimum, market closed briefly)
        or "price has changed" in e or "1210" in e
        or "must be greater" in e or "1211" in e or "minimum" in e
        or "1202" in e or "1206" in e
        # browser-route transients
        or "no instant_order post" in e or "could not find" in e
        or "no confirmation toast" in e or "could not click" in e
    )
    if transient:
        return
    base = symbol.split("/")[0]
    _andx_untradable[base] = time.time() + _ANDX_UNTRADABLE_TTL
    _tlog("system", f"{symbol} not tradable on andX (rejected: {err[:60]}) — routing to sim for 1h")


def _do_open(symbol: str, direction: str, signal_snapshot: dict, kelly: dict,
             stop_loss: float, take_profit: float) -> dict:
    """Open a position in the given direction ('long' or 'short').
    Routes to portfolio.buy or portfolio.short, records the entry,
    re-anchors SL/TP to the actual fill price, and mirrors the same
    decision to the parallel sim portfolio.

    Two reasons a position goes SIM-ONLY (skips the live andX order):
      1. Symbol isn't exec-tradable (e.g. andX has no SOL market)
      2. It's a SHORT — andX is spot-only (no margin/borrow), so real
         shorts are impossible there. Shorts still run in sim so the full
         long+short strategy is observable.
    Longs on andX-listed symbols go to BOTH live and sim."""
    # andX confirmed spot-only (Buy/Sell, no leverage) — never attempt live shorts.
    short_not_live = (direction == "short") and not EXEC_SUPPORTS_SHORT
    tradable_live = (_exec_tradable(symbol) and LIVE_TRADING and not short_not_live
                     and not trader.get("paper_only"))
    if not tradable_live:
        # Live skip → sim-only path
        if trader_sim["enabled"]:
            try:
                sim_symbol = symbol.replace("/USDT", "/USD")
                sim_open = portfolio_sim.short if direction == "short" else portfolio_sim.buy
                sim_res = sim_open(sim_symbol, qty=kelly["qty"],
                                   stop_loss=stop_loss, take_profit=take_profit,
                                   signal_snapshot=signal_snapshot,
                                   ml_confidence=signal_snapshot.get("ml_confidence"))
                if "error" not in sim_res:
                    _slog("buy" if direction == "long" else "short",
                          f"SIM-ONLY {sim_symbol} {direction.upper()} {kelly['qty']:.8g} @ {sim_res['price']:.6g} "
                          f"(not exec-tradable)")
                    return sim_res
                else:
                    logger.debug(f"sim-only open failed for {sim_symbol}: {sim_res['error']}")
            except Exception as e:
                logger.debug(f"sim-only open failed: {e}")
        return {"error": "not exec-tradable, sim disabled"}

    open_fn = short if direction == "short" else buy
    res = open_fn(symbol,
                  qty=kelly["qty"],
                  stop_loss=stop_loss, take_profit=take_profit,
                  signal_snapshot=signal_snapshot,
                  ml_confidence=signal_snapshot.get("ml_confidence"))
    if "error" in res:
        _tlog("buy_fail" if direction == "long" else "short_fail",
              f"{symbol} {direction.upper()}: {res['error']}")
        # Learn from the rejection: blacklist missing markets (route to sim),
        # but keep retrying on collateral issues.
        _note_andx_rejection(symbol, str(res.get("error", "")))
        # Even if live rejected, still try sim so the strategy is observable
        if trader_sim["enabled"]:
            try:
                sim_symbol = symbol.replace("/USDT", "/USD")
                sim_open = portfolio_sim.short if direction == "short" else portfolio_sim.buy
                sim_res = sim_open(sim_symbol, qty=kelly["qty"],
                                   stop_loss=stop_loss, take_profit=take_profit,
                                   signal_snapshot=signal_snapshot,
                                   ml_confidence=signal_snapshot.get("ml_confidence"))
                if "error" not in sim_res:
                    _slog("buy" if direction == "long" else "short",
                          f"SIM {sim_symbol} {direction.upper()} {kelly['qty']:.8g} @ {sim_res['price']:.6g} "
                          f"(live rejected — sim only)")
            except Exception as e:
                logger.debug(f"sim {direction} mirror failed: {e}")
        return res

    # Re-anchor SL/TP to ACTUAL fill price on the correct side for direction
    fill_price = float(res.get("price") or 0)
    sl_pct = float(signal_snapshot.get("_sl_pct") or 0)
    tp_pct = float(signal_snapshot.get("_tp_pct") or 0)
    if fill_price > 0 and sl_pct > 0 and tp_pct > 0:
        if direction == "long":
            new_sl = fill_price * (1 - sl_pct); new_tp = fill_price * (1 + tp_pct)
            valid = (new_sl < fill_price < new_tp)
        else:  # short
            new_sl = fill_price * (1 + sl_pct); new_tp = fill_price * (1 - tp_pct)
            valid = (new_tp < fill_price < new_sl)
        if not valid:
            logger.warning(f"re-anchor produced invalid SL/TP for {symbol} {direction}; using mode defaults")
            mm = _mode()
            if direction == "long":
                new_sl = fill_price * (1 - mm["sl_pct_min"])
                new_tp = fill_price * (1 + mm["tp_pct_min"] * 2)
            else:
                new_sl = fill_price * (1 + mm["sl_pct_min"])
                new_tp = fill_price * (1 - mm["tp_pct_min"] * 2)
        _rewrite_last_entry_sl_tp(symbol, new_sl, new_tp)
        stop_loss = new_sl
        take_profit = new_tp

    side_tag = "BUY" if direction == "long" else "SHORT"
    _tlog("buy" if direction == "long" else "short",
          f"{symbol} {side_tag} {kelly['qty']:.8g} @ {res['price']:.6g} "
          f"= ${res['total_cost']:,.2f} | scalp={signal_snapshot.get('scalp_score', 0):.0f} "
          f"SL={stop_loss:.6g} TP={take_profit:.6g}")

    # Mirror to sim portfolio with the same direction
    if trader_sim["enabled"]:
        try:
            sim_symbol = symbol.replace("/USDT", "/USD")
            sim_open = portfolio_sim.short if direction == "short" else portfolio_sim.buy
            sim_res = sim_open(
                sim_symbol, qty=kelly["qty"],
                stop_loss=stop_loss, take_profit=take_profit,
                signal_snapshot=signal_snapshot,
                ml_confidence=signal_snapshot.get("ml_confidence"),
            )
            if "error" not in sim_res:
                _slog("buy" if direction == "long" else "short",
                      f"SIM {sim_symbol} {side_tag} {kelly['qty']:.8g} @ {sim_res['price']:.6g}")
        except Exception as e:
            logger.debug(f"sim mirror open failed: {e}")

    return res


# Back-compat shim: anything that still calls _do_buy gets a long open.
def _do_buy(symbol: str, signal_snapshot: dict, kelly: dict,
            stop_loss: float, take_profit: float) -> dict:
    return _do_open(symbol, "long", signal_snapshot, kelly, stop_loss, take_profit)


def _scan_and_buy():
    """One full scan cycle. Crypto runs 24/7 — no market-hours gating.
    All entry thresholds (gate, max positions, size cap, SL/TP clamps) read
    from the active risk mode."""
    global NO_SIGNAL_COUNT, _last_scan_results

    m = _mode()
    sp = _get_scan_params()
    wo = get_weight_overrides()
    try:
        portfolio = get_portfolio_summary()
    except Exception as e:
        _tlog("err", f"portfolio load failed: {e}")
        return

    # Use the portfolio summary as the source of truth — works the same for
    # both backends (live andX mirror or sim). `tradable_positions` skips
    # dormant balances like the DOGE the bot can't trade.
    tradable_positions = {s: pos for s, pos in portfolio.get("positions", {}).items()
                          if pos.get("tradable", True)}
    cash = portfolio["cash"]
    quote = portfolio.get("quote_asset", "USDT")
    # andX minimum order is 5 USDT; below that no order can clear
    if cash < 5:
        # No live capital. If andX credentials were never configured, keep
        # scanning against the parallel SIM portfolio so paper trading works
        # without any API keys (entries route sim-only via paper_only).
        import andx_credentials
        sim_p = None
        if trader_sim["enabled"] and not andx_credentials.all_required_present():
            try:
                sim_p = portfolio_sim.get_portfolio_summary()
            except Exception:
                sim_p = None
        if not sim_p or float(sim_p.get("cash") or 0) < 5:
            with trader_lock:
                trader["status"] = "Capital deployed — waiting for sells"
                trader["paper_only"] = False
            return
        portfolio = sim_p
        tradable_positions = {s: pos for s, pos in portfolio.get("positions", {}).items()
                              if pos.get("tradable", True)}
        cash = portfolio["cash"]
        quote = portfolio.get("quote_asset", "USD")
        with trader_lock:
            trader["status"] = "PAPER MODE — no andX credentials, trading sim only"
            trader["paper_only"] = True
    else:
        with trader_lock:
            trader["paper_only"] = False

    # Universe = top-volume + favorites + already-held tradable symbols
    universe = list(get_universe())
    favs = get_favorites()
    for s in favs:
        if s not in universe:
            universe.append(s)
    for s in tradable_positions.keys():
        if s not in universe:
            universe.append(s)
    # FORCE ANDX1 into every scan — 0% fees on andX make it the frequency
    # engine for the 200-trades/day plan. Without this it might not crack
    # the top-volume list (small-cap) and would never get considered.
    for forced in ("ANDX1/USD", "ANDX1/USDT"):
        if forced not in universe:
            universe.append(forced)

    with trader_lock:
        trader["status"] = f"[{m['label']}] Scanning {len(universe)} symbols (gate={m['scalp_gate']} | max_pos={m['max_positions']} | size={int(m['max_position_pct']*100)}%)"

    def _progress(done, total, sym):
        with trader_lock:
            trader["status"] = f"Scanning {done}/{total} ({sym})"

    results = scan_market(
        symbols=universe,
        min_confidence=0,
        min_risk_reward=0.0,
        signal_filter="ALL",
        period="1d", interval="5m",
        weight_overrides=wo,
        progress_callback=_progress,
    )
    _last_scan_results = results or []
    with trader_lock:
        trader["last_scan_time"] = datetime.utcnow().isoformat() + "Z"

    if not results:
        NO_SIGNAL_COUNT += 1
        if NO_SIGNAL_COUNT % 5 == 0:
            _tlog("quiet", "No quality setups right now")
        return

    # Score
    scored = []
    scored_all = []
    # DIP MODE: run the dip detector on each candidate. Qualifying dips get
    # a HUGE score boost (+40) so they outrank generic trend signals, AND
    # we stash the dip-tuned SL/TP for later use. In dip-only mode anything
    # that ISN'T a qualifying dip gets dropped from `scored` entirely so
    # the bot ONLY trades dips.
    import dip_detector
    import coin_strategies
    with trader_lock:
        _dip_mode = trader.get("dip_mode", False)
    for r in results:
        try:
            s, reasons = _scalp_score(r)
        except Exception:
            continue
        r["scalp_score"] = s
        r["scalp_reasons"] = reasons
        # Dip detection per-coin (uses its tier's dip_threshold)
        coin = coin_strategies.get(r["ticker"])
        dip_threshold = int(coin.get("dip_threshold", 60))
        try:
            d_score, d_reasons, d_tp, d_sl = dip_detector.score_dip(r)
        except Exception:
            d_score, d_reasons, d_tp, d_sl = 0, [], 0.012, 0.012
        r["dip_score"] = d_score
        r["dip_reasons"] = d_reasons
        r["dip_tp_pct"] = d_tp
        r["dip_sl_pct"] = d_sl
        r["is_dip"] = d_score >= dip_threshold
        # Real dips get +40 to scalp_score so they dominate ranking
        if r["is_dip"]:
            r["scalp_score"] = min(100, r["scalp_score"] + 40)
            r["scalp_reasons"] = [f"DIP({d_score})"] + reasons
        scored_all.append(r)
        # In DIP mode, ONLY dips make the buy list. Otherwise standard 40+ filter.
        if _dip_mode:
            if r["is_dip"]:
                scored.append(r)
        else:
            if r["scalp_score"] >= max(sp["min_confidence"] - 10, 40):
                scored.append(r)
    scored.sort(key=lambda x: x["scalp_score"], reverse=True)
    scored = scored[:10]

    # FULL-DEPLOYMENT: BTC/ETH are the only live-tradable coins and they often
    # rank below the top-10 alts. If force_deploy is on, make sure the
    # andX-tradable coins are ALWAYS in the buy list (appended if missing), so
    # the live account's cash actually gets deployed into them.
    with trader_lock:
        _force_deploy_now = trader.get("force_deploy", False)
    if _force_deploy_now:
        # Pull from ALL results (not just scored_all which had a 40-point
        # pre-filter). When force-deploying we WANT BTC/ETH appended even if
        # they score 10, because the user wants USDT working at all times.
        present = {r["ticker"] for r in scored}
        for r in results:
            if r.get("scalp_score") is None:
                # Compute a score for force-appended coins so downstream code
                # has the field. Fallback to 50 if scoring fails.
                try:
                    s, reasons = _scalp_score(r)
                    r["scalp_score"] = s
                    r["scalp_reasons"] = reasons
                except Exception:
                    r["scalp_score"] = 50
                    r["scalp_reasons"] = ["force-deploy fallback score"]
            if _exec_tradable(r["ticker"]) and r["ticker"] not in present:
                scored.append(r)
                present.add(r["ticker"])

    if not scored:
        NO_SIGNAL_COUNT += 1
        if NO_SIGNAL_COUNT % 5 == 0:
            _tlog("quiet", "No quality scalp setups")
        return

    # MTF confirmation on top 3 (15m)
    for r in scored[:3]:
        try:
            r15 = full_analysis(r["ticker"], period="1d", interval="15m", weight_overrides=wo)
            if r15:
                r["scalp_score"] = r["scalp_score"] * 0.6 + r15.get("confidence", 50) * 0.4
        except Exception:
            pass
    scored.sort(key=lambda x: x["scalp_score"], reverse=True)
    NO_SIGNAL_COUNT = 0

    # Quality gate (mode floor + slider override + learner's optimal)
    quality_gate = _active_scalp_gate()
    if scored[0]["scalp_score"] < quality_gate:
        _tlog("system",
              f"[{m['label']}] Waiting for better setup — gate={quality_gate}, "
              f"best={scored[0]['scalp_score']:.0f}")
        return

    _tlog("system", f"[{m['label']}] Top setup: {scored[0]['ticker']} "
                    f"score {scored[0]['scalp_score']:.0f}")

    # Goal-based throttling
    with trader_lock:
        session_pnl = trader["session_pnl"]
        manual = trader["manual_mode"]
    goal_size_scale = 1.0
    goal_score_boost = 0
    if session_pnl >= DAILY_TARGET:
        goal_score_boost = 25; goal_size_scale = 0.5
        _tlog("goal", f"TARGET HIT (${session_pnl:,.0f}) — protect mode")
    elif session_pnl >= DAILY_TARGET * PROTECT_THRESHOLD:
        goal_score_boost = 10; goal_size_scale = 0.8
        _tlog("goal", f"NEAR TARGET (${session_pnl:,.0f}) — tighter gates")

    if manual:
        _tlog("manual", f"Manual mode — best is {scored[0]['ticker']} ({scored[0]['scalp_score']:.0f})")
        return

    # Learned TP/SL adjustments
    learned = get_strategy_adjustments()
    tp_mult = float(learned.get("tp_multiplier", 1.0))
    sl_mult = float(learned.get("sl_multiplier", 1.0))

    # Regime
    try:
        regime = get_market_regime()
    except Exception:
        regime = {"regime": "NORMAL", "market_bias": "NEUTRAL"}

    bought = 0
    live_attempts = 0   # bound andX order calls per cycle. With the
                        # browser-routed exec layer, each attempt costs
                        # ~3-5s (page nav + UI clicks + response). At
                        # AGGRESSIVE we allow up to max_opens_per_cycle×4
                        # attempts so a burst of rejections can't starve
                        # the order fills.
    MAX_LIVE_ATTEMPTS_PER_CYCLE = max(20, int(m.get("max_opens_per_cycle", 2)) * 4)

    # Count how many exec-tradable coins (andX-listed) appear in `scored` this
    # cycle. Force-deploy will divide available cash evenly across them so
    # ALL the USDT gets used — even if only 1 or 2 coins are tradable.
    with trader_lock:
        _fd_now = trader.get("force_deploy", False)
        _tm_now = trader.get("trade_mode", "both")
    n_exec_scored = sum(1 for r in scored if _exec_tradable(r["ticker"]))

    for r in scored:
        if cash < 5: break
        symbol = r["ticker"]


        if symbol in _sell_cooldowns:
            sold_time, cd_secs = _sell_cooldowns[symbol]
            if (datetime.now() - sold_time).total_seconds() < cd_secs:
                continue
            else:
                del _sell_cooldowns[symbol]

        # Re-entry cooldown after exit (mode-tuned)
        last_exit = _ticker_cooldown.get(symbol)
        if last_exit and (datetime.now() - last_exit).total_seconds() < m["reentry_cooldown_seconds"]:
            continue

        # (combined position cap is applied below after we know whether sim
        # also holds this symbol — see "Total-positions cap" check.)

        # "Already hold?" check must be BACKEND-SPECIFIC. A symbol routes LIVE
        # if it's exec-tradable (BTC/ETH), else SIM. Dedup against the backend
        # it will actually trade in — otherwise a sim BTC holding wrongly
        # blocks a live BTC buy (and vice versa).
        sym_candidates = (symbol, symbol.replace("/USD", "/USDT"), symbol.replace("/USDT", "/USD"))
        going_live = _exec_tradable(symbol) and LIVE_TRADING
        # FORCE-DEPLOY PYRAMID: when force_deploy is on for an andX-tradable
        # coin, we INTENTIONALLY bypass the held-check, the position cap, and
        # the asset-class cap. User's mandate: "use ALL the USDT, even if only
        # a few coins are tradable" — that means averaging into BTC/ETH again
        # is fine (and required) rather than letting cash sit.
        is_force_pyramid = going_live and _fd_now and _tm_now in ("long", "both")
        sim_p = portfolio_sim._load()
        sim_positions = sim_p.get("positions") or {}
        if going_live:
            # Held-check uses the AUTHORITATIVE andX balance (live positions),
            # NOT local meta entries — meta entries can go stale after a close.
            # SKIPPED entirely when force-deploying: we WANT to pyramid in.
            if not is_force_pyramid:
                all_live_positions = portfolio.get("positions") or {}
                live_has = any(c in all_live_positions for c in sym_candidates)
                if live_has:
                    continue
        else:
            if any(s in sim_positions for s in sym_candidates):
                continue

        # Position cap — bypassed for force-pyramid (we're adding to existing
        # positions, not opening new ones, so the cap doesn't apply).
        if not is_force_pyramid:
            if going_live:
                if len(tradable_positions) >= m["max_positions"]:
                    break
            else:
                if len(sim_positions) >= m["max_positions"]:
                    break

        # Asset-class cap — same: skipped when force-pyramiding.
        my_class = asset_class(symbol)
        if not is_force_pyramid:
            if going_live:
                class_count = sum(1 for s in tradable_positions if asset_class(s) == my_class)
            else:
                class_count = sum(1 for s in sim_positions if asset_class(s) == my_class)
            if class_count >= m["max_per_asset_class"]:
                _tlog("skip", f"{symbol} {my_class} class already at {class_count}")
                continue

        # Score gate (with goal boost). BYPASSED for force-deployment on
        # exec-tradable coins — the whole point of full-deployment is to put
        # cash to work regardless of setup quality, so a low score must not
        # block it.
        with trader_lock:
            _fd = trader.get("force_deploy", False)
            _tm = trader.get("trade_mode", "both")
        force_deploy_coin = (_fd and going_live and _tm in ("long", "both"))
        if not force_deploy_coin and r["scalp_score"] < quality_gate + goal_score_boost:
            continue

        # DIRECTION DISPATCH: bullish signal → LONG, bearish → SHORT.
        # HOLD/UNKNOWN setups are skipped (no direction = no edge).
        # trade_mode filter ("both"/"long"/"short") lets the user lock to one side.
        sig = r.get("signal", "")
        is_long_setup = "BUY" in sig
        is_short_setup = "SELL" in sig
        with trader_lock:
            tmode = trader.get("trade_mode", "both")
            force_deploy = trader.get("force_deploy", False)

        # FULL-DEPLOYMENT OVERRIDE: user wants all andX USDT working at all times.
        # When force_deploy is on and we're long-capable, BUY andX-tradable coins
        # as longs even without a bullish signal — just to keep cash deployed.
        # (This deliberately ignores bearish signals; per-position stops cap risk.)
        forced_long = (force_deploy and tmode in ("long", "both")
                       and _exec_tradable(symbol) and LIVE_TRADING)

        if forced_long:
            direction = "long"
        else:
            if not (is_long_setup or is_short_setup):
                continue
            if is_long_setup and tmode not in ("long", "both"):
                continue
            if is_short_setup and tmode not in ("short", "both"):
                continue
            direction = "short" if is_short_setup else "long"

        # Kelly sizing
        stats = calculate_win_loss_stats(_load_trade_history_safe())
        atr = r.get("atr") or 0.0
        price = r.get("price") or 0.0
        atr_pct = (atr / price * 100) if price > 0 else 2.0
        candle_patterns = (r.get("candlestick_patterns") or {}).get("patterns", [])
        bearish_patterns = sum(1 for pat in candle_patterns if pat.get("type") == "bearish")

        kelly = kelly_position_size(
            portfolio_value=portfolio["total_portfolio_value"],
            cash=cash,
            ml_probability=_scalp_to_win_prob(r["scalp_score"]),
            technical_score=r["scalp_score"],
            symbol=symbol,
            price=price,
            atr_pct=atr_pct,
            current_positions=tradable_positions,
            trade_stats=stats,
            max_single_position_pct=m["max_position_pct"],
            min_order_value=m["min_order_value"],
            regime_match=(regime.get("regime") in ("LOW_VOL", "NORMAL")),
            pattern_confluence=bearish_patterns,
        )
        # Full-deployment override: when force_deploy is on for an andX-tradable
        # coin, ALWAYS overwrite Kelly's sizing. Split available cash across
        # ALL exec-tradable coins scored this cycle (whether already held or
        # not — we pyramid). If only 1 or 2 coins are tradable, each gets a
        # huge chunk of cash; that's exactly what "use all the USDT" means.
        # The per-cycle open cap (max_opens_per_cycle) prevents this from
        # dumping everything into a single asset in one tick.
        if forced_long and price > 0:
            n_split = max(n_exec_scored, 1)
            deploy_usd = min(cash * 0.95 / n_split, cash * 0.95)
            if deploy_usd < m["min_order_value"]:
                # Cash too small to subdivide further → dump what's left into
                # this one coin so it doesn't sit idle.
                deploy_usd = cash * 0.95
                if deploy_usd < m["min_order_value"]:
                    continue
            kelly["qty"] = deploy_usd / price
            kelly["reasons"] = [f"forced full-deployment (split across {n_split} tradable)"]
        elif m.get("force_max_size") and price > 0:
            # SNIPER conviction sizing: bypass Kelly's risk-shrink. Conviction
            # mode trades few but BIG — always deploy the mode's full
            # max_position_pct (capped by available cash). Without this Kelly
            # would shrink size based on win-rate history and we'd end up with
            # the same scalp-sized positions the user explicitly rejected.
            target_notional = min(
                portfolio["total_portfolio_value"] * m["max_position_pct"],
                cash * 0.95,
            )
            if target_notional < m["min_order_value"]:
                _tlog("skip", f"{symbol} sniper: ${target_notional:.0f} below min "
                              f"${m['min_order_value']:.0f}")
                continue
            kelly["qty"] = target_notional / price
            kelly["reasons"] = [f"SNIPER conviction sizing ${target_notional:.0f} "
                                f"({m['max_position_pct']*100:.0f}% of portfolio)"]
        elif kelly["qty"] <= 0:
            _tlog("skip", f"{symbol} kelly: {kelly['reasons'][-1] if kelly['reasons'] else 'no edge'}")
            continue

        # Apply mode size_boost AND goal size scale (skip for forced full-deploy
        # so it actually consumes 95% of cash without extra multipliers).
        if not forced_long:
            kelly["qty"] *= m["size_boost"] * goal_size_scale

        # SL / TP — absolute-distance approach, direction-aware placement.
        # PER-COIN OVERRIDES: each coin has its own tier (BLUE_CHIP/MAJOR/ALT/
        # MEME) with tuned SL/TP clamps. BTC tightens (+0.6-2.0% SL, +0.8-3% TP)
        # while meme coins widen (+2-6% SL, +2.5-8% TP). For DIPs we use the
        # detector's tighter scalp targets directly (overrides mode AND coin).
        coin_cfg = coin_strategies.get(symbol)
        # 1) start with mode defaults
        sl_min, sl_max = m["sl_pct_min"], m["sl_pct_max"]
        tp_min, tp_max = m["tp_pct_min"], m["tp_pct_max"]
        # 2) coin tier overrides them — except in SNIPER, where the mode's
        #    chunky TPs (2.5-6%) must NOT be squashed by TIER_BLUE_CHIP's
        #    1.0-2.5% clamps. SNIPER bets are big and need room to pay off.
        if not m.get("ignore_coin_tiers"):
            sl_min = coin_cfg.get("sl_pct_min", sl_min)
            sl_max = coin_cfg.get("sl_pct_max", sl_max)
            tp_min = coin_cfg.get("tp_pct_min", tp_min)
            tp_max = coin_cfg.get("tp_pct_max", tp_max)
        # 3) DIP scalps override both with their own tighter SL/TP from the detector
        if r.get("is_dip"):
            sl_pct = r["dip_sl_pct"] * sl_mult
            tp_pct = r["dip_tp_pct"] * tp_mult
        else:
            analysis_sl_dist = abs(price - r.get("stop_loss", price * (1 - sl_min))) / price
            analysis_tp_dist = abs(r.get("take_profit_1", price * (1 + tp_min)) - price) / price
            sl_pct = max(sl_min, min(sl_max, analysis_sl_dist)) * sl_mult
            tp_pct = max(tp_min, min(tp_max, analysis_tp_dist)) * tp_mult
        if direction == "long":
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        else:  # short: SL above, TP below
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)
        r["_sl_pct"] = sl_pct
        r["_tp_pct"] = tp_pct
        r["_direction"] = direction

        # EDGE FLOOR: Audit-3's death-by-fees fix. If the planned TP target
        # doesn't clear 1.5× the round-trip fee, the trade has negative EV
        # at this fee tier and we MUST refuse to enter. Without this guard,
        # the 0.5% RT cost on BTC/ETH bleeds the account over hundreds of
        # tiny "wins". ANDX1 (0% RT) only enforces a minimum slippage floor.
        _rt_fee = coin_strategies.round_trip_fee(symbol)
        _edge_floor = max(0.002, 1.5 * _rt_fee)
        # Time-of-day guard: outside the 09:00–17:00 UTC EU-US overlap window
        # (peak depth $3.86M @ 11:00 UTC vs trough $2.71M @ 21:00 UTC per
        # Amberdata), thin books eat the edge — raise the floor 2× on BTC/ETH.
        _utc_hr = datetime.utcnow().hour
        # Off-hours fee-floor bump applies to ANY fee-paying coin, not just
        # BTC/ETH. ANDX1 (0% RT fee) skips this branch by definition.
        if not (9 <= _utc_hr < 17) and _rt_fee > 0:
            _edge_floor = max(_edge_floor, 3.0 * _rt_fee)
        if tp_pct < _edge_floor:
            _tlog("skip",
                  f"{symbol} edge-floor reject: TP {tp_pct*100:.2f}% < required "
                  f"{_edge_floor*100:.2f}% (fee {_rt_fee*100:.2f}% RT)")
            continue

        # FINAL $ NOTIONAL CAP: on a small account, one bad fill can dominate
        # the day. Cap absolute notional at $100 in AGGRESSIVE so worst-case
        # single-trade loss is ~$1-2 (1.2% SL × $100). SNIPER opts out via
        # skip_final_notional_cap because conviction mode deliberately puts
        # large size into few trades.
        if m.get("label") == "AGGRESSIVE" and not m.get("skip_final_notional_cap"):
            FINAL_NOTIONAL_CAP_AGGRESSIVE = 100.0
            notional = kelly["qty"] * price
            if notional > FINAL_NOTIONAL_CAP_AGGRESSIVE:
                kelly["qty"] = FINAL_NOTIONAL_CAP_AGGRESSIVE / price

        # Stash some fields into the signal snapshot for the learner
        r["ml_confidence"] = _scalp_to_win_prob(r["scalp_score"])
        r["ml_features"] = _entry_features_snapshot(symbol, {"price": price}, r)

        # Count live attempts (symbols not pre-known-untradable will hit andX).
        if _exec_tradable(symbol) and LIVE_TRADING:
            live_attempts += 1

        res = _do_open(symbol, direction, r, kelly, sl, tp)
        if res and "error" not in res:
            bought += 1
            # Refresh local snapshot
            p = load_portfolio()
            portfolio = get_portfolio_summary()
            cash = portfolio["cash"]
        # Per-cycle open cap is now mode-driven: CONSERVATIVE=1, REGULAR=2,
        # AGGRESSIVE=4. This is what makes AGGRESSIVE truly "constant trading."
        if bought >= int(m.get("max_opens_per_cycle", 2)):
            break
        if live_attempts >= MAX_LIVE_ATTEMPTS_PER_CYCLE:
            # Hit the per-cycle andX call budget — stop here, learn continues
            # next cycle (rejections are cached so they won't be retried).
            break


def _load_trade_history_safe() -> list:
    try:
        f = Path(__file__).parent / "trade_history.json"
        if f.exists():
            with open(f, "r") as fh:
                return json.load(fh)
    except Exception:
        return []
    return []


# ==========================================
# TRADING LOOPS
# ==========================================

def _exit_loop():
    """Tight loop: every CHECK_INTERVAL seconds."""
    while not trader["stop_event"].is_set():
        try:
            if not trader["paused"]:
                _manage_open_positions()
                _check_auto_tp()
                if trader_sim["enabled"]:
                    _manage_sim_positions()
        except Exception as e:
            _tlog("err", f"exit loop: {e}")
        trader["stop_event"].wait(CHECK_INTERVAL)


def _manage_sim_positions():
    """Mirror of _manage_open_positions but operating on portfolio_sim.
    Sim runs independent SL/TP/harvest on Alpaca-priced positions.

    Design choice: sim follows the SAME active risk mode as live, so the
    paper account is a faithful "what-if" of the live strategy. If we let
    them diverge, the sim's numbers would no longer answer the question
    "would this strategy have made money?" — they'd answer "would SOME
    strategy have made money?". Keep coupled."""
    p = portfolio_sim._load()
    if not p["positions"]:
        return
    from alpaca_crypto_client import AlpacaCryptoClient
    a = AlpacaCryptoClient()
    prices = a.get_prices(list(p["positions"].keys()))
    # Hoist mode lookup out of the per-position loop (one lock acquire, not N).
    m = _mode()
    for symbol, pos in list(p["positions"].items()):
        price = prices.get(symbol)
        if not price or price <= 0:
            continue
        avg_cost = pos["avg_cost"]
        if avg_cost <= 0:
            continue
        # Direction-aware gain for sim
        sim_direction = pos.get("side", "long")
        if sim_direction == "short":
            gain = (avg_cost - price) / avg_cost
        else:
            gain = (price - avg_cost) / avg_cost
        last_entry = pos["entries"][-1] if pos["entries"] else {}
        entry_date = last_entry.get("date", "")
        try:
            held = (datetime.utcnow() - datetime.fromisoformat(entry_date.replace("Z", ""))).total_seconds() if entry_date else 99999
        except Exception:
            held = 99999

        peak = _sim_position_peaks.get(symbol, gain)
        if gain > peak:
            peak = gain
            _sim_position_peaks[symbol] = peak

        snapshot = last_entry.get("signal_snapshot") or {}

        def _close(reason, _dir=sim_direction):
            close_fn = portfolio_sim.cover if _dir == "short" else portfolio_sim.sell
            res = close_fn(symbol, sell_all=True, signal_snapshot=snapshot)
            if "error" in res:
                return
            spnl = res.get("pnl", 0); spnl_pct = res.get("pnl_pct", 0)
            swon = spnl > 0
            with trader_sim_lock:
                trader_sim["session_pnl"] += spnl
                trader_sim["session_trades"] += 1
                if swon: trader_sim["session_wins"] += 1
                else: trader_sim["session_losses"] += 1
                if trader_sim["session_pnl"] > trader_sim["session_peak"]:
                    trader_sim["session_peak"] = trader_sim["session_pnl"]
            _sim_position_peaks.pop(symbol, None)
            _slog("sell" if swon else "loss",
                  f"{symbol} exit @{res.get('price'):.6g} pnl ${spnl:+.2f} ({spnl_pct:+.2f}%) — {reason}")

        # Net-of-fees parity with live: sim uses the same per-coin fee map
        # so sim PnL reflects what live would actually pocket. Without this,
        # sim looks artificially profitable vs live.
        import coin_strategies as _cs_sim
        _fee_rt = _cs_sim.round_trip_fee(symbol)
        _net_gain = gain - _fee_rt
        _net_peak = peak - _fee_rt
        _harvest_thr = float(_cs_sim.get(symbol).get("harvest_threshold", m["harvest_threshold"]))
        _min_hold_s = int(_cs_sim.get(symbol).get("min_hold_seconds", m["min_hold_seconds"]))

        if _net_gain <= -m["emergency_stop_pct"]:
            _close("EMERGENCY STOP"); continue
        if held < _min_hold_s:
            continue
        sl = last_entry.get("stop_loss"); tp = last_entry.get("take_profit")
        # Direction-aware SL/TP comparison (price-level — no fee adjustment)
        if sim_direction == "short":
            sim_sl_hit = sl and price >= sl
            sim_tp_hit = tp and price <= tp
        else:
            sim_sl_hit = sl and price <= sl
            sim_tp_hit = tp and price >= tp
        if sim_sl_hit: _close("STOP LOSS HIT"); continue
        if sim_tp_hit: _close("TAKE PROFIT HIT"); continue
        if _net_gain >= _harvest_thr: _close("HARVEST"); continue
        if _net_peak >= m["trail_lock_pct"] and _net_gain <= _net_peak - 0.005: _close("TRAIL LOCK"); continue
        if _net_peak >= m["trail_be_pct"] and _net_gain <= 0: _close("TRAIL BREAKEVEN"); continue
        if _net_peak >= m["fade_threshold"] and _net_gain < _net_peak * (1 - m["fade_drop"]):
            _close("FADE PROTECT"); continue
        if _net_gain <= m["dump_bleed_pct"]: _close("DUMP BLEED"); continue


def _scan_loop():
    """Slower loop: scans every active-mode `scan_interval` seconds."""
    while not trader["stop_event"].is_set():
        try:
            if not trader["paused"]:
                _scan_and_buy()
        except Exception as e:
            _tlog("err", f"scan loop: {e}")
        sp = _get_scan_params()
        trader["stop_event"].wait(sp["interval"])


def _check_auto_tp():
    with trader_lock:
        enabled = trader["auto_tp_enabled"]
        threshold = trader["auto_tp_threshold"]
        baseline = trader["auto_tp_baseline"]
    if not enabled:
        return
    try:
        ps = get_portfolio_summary()
        current = ps["realized_pnl"] + ps["unrealized_pnl"]
    except Exception:
        return
    if (current - baseline) >= threshold:
        _tlog("auto_tp", f"Auto-TP fired: gain ${current - baseline:.2f} >= ${threshold:.2f}")
        _sell_all_positions(exit_reason="AUTO TP")
        with trader_lock:
            trader["auto_tp_enabled"] = False


def _sell_all_positions(exit_reason: str = "MANUAL") -> dict:
    """Attempt to close EVERY open position — real ones, not just the ones the
    docs API flagged tradable. Returns a per-symbol result summary so the
    dashboard can show exactly what worked and what didn't.

    Design notes:
      - We try dormant/tradable=False positions too. The bot's `tradable` flag
        only reflects what the docs-API knows about (BTC/ETH/ANDX1). Positions
        held from earlier browser/UI trades still show as tradable=False but
        the browser CAN sell them if the Chrome debug window is up.
      - Sells are sequential and blocking. That's fine here — the caller is a
        one-shot "Sell all" button, not the hot trading loop.
      - Every result gets recorded whether it succeeded or not, so the UI can
        show a clear "sold X, failed Y" table.
    """
    from datetime import datetime as _dt
    results: list[dict] = []
    snap = get_portfolio_summary()
    positions = snap.get("positions", {}) or {}
    for symbol, pos in list(positions.items()):
        qty = float(pos.get("qty") or 0)
        if qty <= 0:
            continue  # nothing to sell
        started = _dt.utcnow()
        last_entry = _last_entry_for(symbol)
        try:
            res = _do_sell(symbol, sell_all=True,
                           signal_snapshot=last_entry.get("signal_snapshot"),
                           exit_reason=exit_reason,
                           entry_data=last_entry)
        except Exception as e:
            res = {"error": f"exception: {e}"}
        elapsed = (_dt.utcnow() - started).total_seconds()
        entry = {
            "symbol": symbol,
            "qty": qty,
            "elapsed_s": round(elapsed, 1),
            "market_value": float(pos.get("market_value") or 0),
            "tradable_flag": pos.get("tradable"),
        }
        if res and "error" not in res:
            entry.update({
                "ok": True,
                "price": res.get("price"),
                "pnl": res.get("pnl"),
                "pnl_pct": res.get("pnl_pct"),
            })
        else:
            entry.update({
                "ok": False,
                "error": (res or {}).get("error", "unknown") if isinstance(res, dict) else str(res),
            })
        results.append(entry)
    sold = sum(1 for r in results if r.get("ok"))
    failed = sum(1 for r in results if not r.get("ok"))
    total_pnl = sum(float(r.get("pnl") or 0) for r in results if r.get("ok"))
    return {
        "results": results,
        "sold_count": sold,
        "failed_count": failed,
        "total_pnl": round(total_pnl, 2),
    }


# ==========================================
# FLASK ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/api/trader/status")
def api_status():
    with trader_lock:
        t = {k: v for k, v in trader.items() if k != "stop_event"}
    with trader_sim_lock:
        ts = dict(trader_sim)
    # The LIVE panel always shows the andX mirror — even when LIVE_TRADING=0
    # (in that mode the bot won't place real orders, but you still see your
    # actual andX balance + positions for situational awareness).
    try:
        import portfolio_live
        ps = portfolio_live.get_portfolio_summary()
    except Exception as e:
        ps = {"error": str(e), "mode": "live-andx-unreachable"}
    try:
        sim_ps = portfolio_sim.get_portfolio_summary()
    except Exception as e:
        sim_ps = {"error": str(e)}
    try:
        stats = _load_stats()
    except Exception:
        stats = {}
    active_mode = t.get("risk_mode") or DEFAULT_RISK_MODE
    return jsonify(clean({
        "trader": t,
        "trader_sim": ts,
        "portfolio": ps,            # LIVE (andX-mirrored when LIVE_TRADING=1)
        "portfolio_sim": sim_ps,    # SIM (paper, Alpaca prices, never andX)
        "stats": stats,
        "universe": get_universe(),
        # Effective gate = max(mode's gate, slider override, learner's optimum).
        # The slider value alone is reported separately as scalp_gate_override.
        "scalp_gate": _active_scalp_gate(),
        "scalp_gate_override": SCALP_GATE_OVERRIDE,
        "risk_mode": active_mode,
        "risk_mode_params": RISK_MODES.get(active_mode, {}),
        "risk_modes": list(RISK_MODES.keys()),
        "trade_mode": t.get("trade_mode") or "both",
        "trade_modes": ["both", "long", "short"],
        "force_deploy": t.get("force_deploy", False),
        "dip_mode": t.get("dip_mode", False),
        "ml_status": _safe_ml_status(),
        "regime": _safe_regime(),
        "live_trading": LIVE_TRADING,
    }))


@app.route("/api/sim/summary")
def api_sim_summary():
    try:
        return jsonify(clean(portfolio_sim.get_portfolio_summary()))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/sim/log")
def api_sim_log():
    with trader_sim_lock:
        return jsonify({"log": list(trader_sim["log"])})


@app.route("/api/prices/stream")
def api_prices_stream():
    """Fast-tick price feed for the dashboard. Pure dict-read off the
    in-process cache — never blocks on Alpaca. Refreshed in the background
    by _price_stream_loop. Use this for ~300ms polling; use
    /api/trader/status for the heavier 2s poll."""
    with _price_cache_lock:
        snap = dict(_price_cache)
    return jsonify({"prices": snap, "ts": datetime.utcnow().isoformat() + "Z"})


@app.route("/api/sim/start_run", methods=["POST"])
def api_sim_start_run():
    """One-click: reset sim to $100K (or body.starting_cash), enable mirror,
    and ensure the trading engine is running. Returns the fresh sim state."""
    body = request.json or {}
    starting_cash = float(body.get("starting_cash", 100000))
    p = portfolio_sim.reset_portfolio(starting_cash)
    with trader_sim_lock:
        trader_sim["enabled"] = True
        trader_sim["session_pnl"] = 0.0
        trader_sim["session_peak"] = 0.0
        trader_sim["session_trades"] = 0
        trader_sim["session_wins"] = 0
        trader_sim["session_losses"] = 0
        trader_sim["session_start"] = datetime.utcnow().isoformat() + "Z"
    _sim_position_peaks.clear()
    started = False
    with trader_lock:
        if not trader["running"]:
            trader["running"] = True
            trader["paused"] = False
            trader["stop_event"].clear()
            trader["session_start"] = datetime.utcnow().isoformat() + "Z"
            started = True
    if started:
        threading.Thread(target=_exit_loop, daemon=True, name="exit_loop").start()
        threading.Thread(target=_scan_loop, daemon=True, name="scan_loop").start()
        threading.Thread(target=_price_stream_loop, daemon=True, name="price_stream").start()
    _slog("system", f"SIM RUN STARTED — fresh ${starting_cash:,.0f} paper account")
    return jsonify(clean({"sim": p, "engine_started": started}))


@app.route("/api/sim/toggle", methods=["POST"])
def api_sim_toggle():
    body = request.json or {}
    on = bool(body.get("enabled", not trader_sim["enabled"]))
    with trader_sim_lock:
        trader_sim["enabled"] = on
    _slog("system", f"Sim mirror {'ENABLED' if on else 'DISABLED'}")
    return jsonify({"enabled": on})


@app.route("/api/sim/reset", methods=["POST"])
def api_sim_reset():
    body = request.json or {}
    starting_cash = body.get("starting_cash")
    p = portfolio_sim.reset_portfolio(starting_cash)
    with trader_sim_lock:
        trader_sim["session_pnl"] = 0.0
        trader_sim["session_peak"] = 0.0
        trader_sim["session_trades"] = 0
        trader_sim["session_wins"] = 0
        trader_sim["session_losses"] = 0
    return jsonify(clean(p))


def _safe_ml_status():
    try: return get_ml_status()
    except Exception: return {}


def _safe_regime():
    try: return get_market_regime()
    except Exception: return {}


@app.route("/api/trader/start", methods=["POST"])
def api_start():
    with trader_lock:
        if trader["running"]:
            return jsonify({"error": "already running"})
        trader["running"] = True
        trader["paused"] = False
        trader["stop_event"].clear()
        trader["session_start"] = datetime.utcnow().isoformat() + "Z"
    threading.Thread(target=_exit_loop, daemon=True, name="exit_loop").start()
    threading.Thread(target=_scan_loop, daemon=True, name="scan_loop").start()
    threading.Thread(target=_price_stream_loop, daemon=True, name="price_stream").start()
    # Auto-start the ANDX1 0%-fee scalp engine — UNLESS the active mode opts
    # out via skip_andx1_engine. SNIPER opts out because its conviction-trade
    # strategy contradicts the engine's HFT-volume play; running both at once
    # would split capital and obscure attribution.
    started_andx1 = False
    if not _mode().get("skip_andx1_engine"):
        try:
            import andx1_engine
            andx1_engine.start()
            started_andx1 = True
        except Exception as e:
            _tlog("err", f"andx1_engine autostart failed: {e}")
    suffix = " (+ ANDX1 engine)" if started_andx1 else " (ANDX1 engine skipped per mode)"
    _tlog("system", "Trading engine STARTED" + suffix)
    return jsonify({"ok": True})


@app.route("/api/trader/stop", methods=["POST"])
def api_stop():
    with trader_lock:
        trader["running"] = False
        trader["stop_event"].set()
    # Stop the ANDX1 engine too so it doesn't keep scalping after main stop
    try:
        import andx1_engine
        andx1_engine.stop()
    except Exception:
        pass
    _tlog("system", "Trading engine STOPPED (+ ANDX1 engine)")
    return jsonify({"ok": True})


@app.route("/api/trader/pause", methods=["POST"])
def api_pause():
    with trader_lock:
        trader["paused"] = not trader["paused"]
        paused = trader["paused"]
    _tlog("system", "Paused" if paused else "Resumed")
    return jsonify({"paused": paused})


@app.route("/api/trader/manual_mode", methods=["POST"])
def api_manual():
    on = bool(request.json.get("enabled", False)) if request.json else False
    with trader_lock:
        trader["manual_mode"] = on
    _tlog("system", f"Manual mode {'ON' if on else 'OFF'}")
    return jsonify({"manual_mode": on})


@app.route("/api/trader/set_scalp_gate", methods=["POST"])
def api_set_scalp_gate():
    """Manual override on top of the active mode's gate. Can only TIGHTEN —
    sending 0 clears the override so the mode's gate is used as-is."""
    global SCALP_GATE_OVERRIDE
    val = int(float(request.json.get("value", 0)))
    if val <= 0:
        SCALP_GATE_OVERRIDE = None
        _tlog("system", f"Scalp gate override cleared (mode floor = {_mode()['scalp_gate']})")
    else:
        SCALP_GATE_OVERRIDE = max(0, min(100, val))
        _tlog("system", f"Scalp gate override = {SCALP_GATE_OVERRIDE}")
    return jsonify({
        "scalp_gate": _active_scalp_gate(),
        "override": SCALP_GATE_OVERRIDE,
        "mode_gate": _mode()["scalp_gate"],
    })


@app.route("/api/trader/set_risk_mode", methods=["POST"])
def api_set_risk_mode():
    """Switch active risk mode: CONSERVATIVE | REGULAR | AGGRESSIVE.
    All entry/exit thresholds re-read from RISK_MODES[mode] on next loop tick."""
    body = request.json or {}
    mode = str(body.get("mode", "")).upper().strip()
    if mode not in RISK_MODES:
        return jsonify({"error": f"Invalid mode. Use: {list(RISK_MODES.keys())}"}), 400
    # If the incoming mode opts out of the ANDX1 HFT engine, stop it now so
    # we don't keep scalping ANDX1 while SNIPER waits for a conviction setup.
    # If it opts back in, the next /api/trader/start will autostart it.
    new_mode_cfg = RISK_MODES[mode]
    if new_mode_cfg.get("skip_andx1_engine"):
        try:
            import andx1_engine
            andx1_engine.stop()
        except Exception:
            pass
    # SNIPER (force_max_size) is conviction sizing — force_deploy's
    # "deploy all cash regardless of setup" contradicts it. Turn off
    # force_deploy automatically when entering a conviction mode.
    if new_mode_cfg.get("force_max_size"):
        with trader_lock:
            trader["force_deploy"] = False
    with trader_lock:
        prev = trader.get("risk_mode")
        trader["risk_mode"] = mode
    _tlog("system", f"RISK MODE: {prev} -> {mode} ({RISK_MODES[mode]['description']})")
    return jsonify({"risk_mode": mode, "params": RISK_MODES[mode]})


@app.route("/api/trader/risk_modes")
def api_risk_modes():
    """Return full RISK_MODES dict + current selection (for UI dropdown)."""
    with trader_lock:
        current = trader.get("risk_mode") or DEFAULT_RISK_MODE
    return jsonify({"current": current, "modes": RISK_MODES})


@app.route("/api/trader/set_force_deploy", methods=["POST"])
def api_set_force_deploy():
    """Toggle full-deployment mode: when on, the bot buys andX-tradable coins
    as longs even without a bullish signal, to keep all USDT invested."""
    body = request.json or {}
    on = bool(body.get("enabled", True))
    with trader_lock:
        trader["force_deploy"] = on
    _tlog("system", f"FULL DEPLOYMENT {'ON — all USDT kept invested' if on else 'OFF — only buys real setups'}")
    return jsonify({"force_deploy": on})


# ----------------------------------------------------------------------
# andX session-cookie management (unlocks the full ~120-coin universe)
# ----------------------------------------------------------------------

@app.route("/api/andx/session", methods=["GET"])
def api_andx_session_status():
    """Lightweight status: is a browser session loaded? When was it saved?"""
    try:
        import andx_session
        return jsonify(andx_session.session_status())
    except Exception as e:
        return jsonify({"error": str(e), "loaded": False}), 500


@app.route("/api/andx/set_session", methods=["POST"])
def api_andx_set_session():
    """Accept either:
      - {"curl": "<raw curl text from DevTools 'Copy as cURL (bash)'>"}
      - {"cookies": {...}, "account_number": 266} — direct dict form

    Persists to andx_session.json. After this returns the bot can place
    orders on the full andX market list via instant_order.
    """
    try:
        import andx_session
    except Exception as e:
        return jsonify({"error": f"andx_session import failed: {e}"}), 500
    body = request.json or {}
    if "curl" in body and body["curl"]:
        summary = andx_session.save_from_curl(body["curl"])
        if not summary.get("cookies_count"):
            return jsonify({"error": "no cookies found in curl — make sure you copied the FULL request",
                            "parsed": summary}), 400
        _tlog("system", f"andX session updated: {summary['cookies_count']} cookies, "
                        f"account_number={summary.get('account_number')}")
        return jsonify({"ok": True, "summary": summary})
    # Direct dict form
    if "cookies" in body and isinstance(body["cookies"], dict):
        blob = {
            "cookies": body["cookies"],
            "headers": body.get("headers") or {},
            "account_number": body.get("account_number"),
            "saved_at": int(time.time()),
        }
        from pathlib import Path
        andx_session.SESSION_FILE.write_text(
            json.dumps(blob, indent=2), encoding="utf-8")
        _tlog("system", f"andX session updated: {len(body['cookies'])} cookies (direct)")
        return jsonify({"ok": True, "summary": andx_session.session_status()})
    return jsonify({"error": "provide either 'curl' (raw text) or 'cookies' (dict)"}), 400


@app.route("/api/andx/clear_session", methods=["POST"])
def api_andx_clear_session():
    try:
        import andx_session
        andx_session.clear_session()
        _tlog("system", "andX session cleared — bot reverts to BTC/ETH only")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/andx/test_session", methods=["POST"])
def api_andx_test_session():
    """Probe the session by hitting a /p/v1/ endpoint. Confirms whether the
    cookies are still valid (200) or expired (401/403). Run this after pasting
    fresh cookies."""
    try:
        import andx_session
        return jsonify(andx_session.session_self_test())
    except Exception as e:
        return jsonify({"error": str(e), "ok": False}), 500


@app.route("/api/debug/balance", methods=["GET"])
def api_debug_balance():
    """Diagnostic endpoint: hit andX's /balance/<account>/ directly and return
    the raw response so we can see WHY the bot thinks USDT is zero.

    Answers "why does it say insufficient funds when I have $2800?" by
    surfacing:
      - Which account_name the bot is querying
      - Which quote asset it expects
      - The full raw response andX returned (including auth errors)
      - All balances andX reports (not just USDT)
    """
    try:
        from andx_client import AndxClient, ANDX_ACCOUNT, ANDX_QUOTE_ASSET
        import os
        c = AndxClient()
        # Also try a few common alternative account names in case theirs is different
        alt_accounts = ["Main", "Trading", "Spot", "main", "trading", "spot"]
        raw_by_account = {}
        for acct in alt_accounts:
            try:
                data = c._get(f"/balance/{acct}/", signed=True)
                if data:
                    raw_by_account[acct] = data
            except Exception as e:
                raw_by_account[acct] = {"exception": str(e)}
        # Highlight the account the bot is actually configured to use
        configured = c.account
        configured_raw = raw_by_account.get(configured)
        # Extract balances from all responses so we can spot where the money is
        summaries = {}
        for acct, raw in raw_by_account.items():
            if not isinstance(raw, dict):
                continue
            balances = (raw.get("data") or {}).get("balances") or {}
            if isinstance(balances, dict):
                non_zero = {}
                for asset, bal in balances.items():
                    try:
                        total = float((bal or {}).get("balance") or 0)
                        free = float((bal or {}).get("available_balance") or 0)
                        if total > 0 or free > 0:
                            non_zero[asset] = {"free": free, "total": total}
                    except Exception:
                        continue
                summaries[acct] = non_zero or "no non-zero balances"
            else:
                summaries[acct] = f"unexpected format: {type(balances).__name__}"
        # API-key check — is auth even reaching andX?
        api_key = os.environ.get("ANDX_API_KEY", "")
        api_secret = os.environ.get("ANDX_API_SECRET", "")
        api_pass = os.environ.get("ANDX_PASSPHRASE", "")
        api_user = os.environ.get("ANDX_USERNAME", "")
        return jsonify({
            "configured_account": configured,
            "configured_quote_asset": ANDX_QUOTE_ASSET,
            "credentials_set": {
                "api_key": bool(api_key),
                "api_secret": bool(api_secret),
                "passphrase": bool(api_pass),
                "username": bool(api_user),
                "api_key_prefix": api_key[:6] + "..." if len(api_key) >= 6 else api_key,
            },
            "raw_configured_response": configured_raw,
            "all_accounts_summary": summaries,
            "hint": (
                "If your money shows under an account other than "
                f"'{configured}', set ANDX_ACCOUNT=<that name> in .env "
                "and restart. If ALL accounts show empty, your API key "
                "either has no trading permission or the credentials are wrong."
            ),
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()[:800],
        }), 500


# ----------------------------------------------------------------------
# andX API credentials (paste your API key/secret here)
# ----------------------------------------------------------------------

@app.route("/api/andx/credentials", methods=["GET"])
def api_andx_credentials_status():
    """Return whether each credential field is set + masked previews of the
    secret-y ones. Never returns api_secret or passphrase in plaintext."""
    try:
        import andx_credentials
        return jsonify(andx_credentials.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/andx/credentials", methods=["POST"])
def api_andx_credentials_save():
    """Save (or update) the user's andX API credentials.

    Body fields (all optional — only changed fields are sent in):
      api_key, api_secret, username, passphrase,
      account_name, account_number, quote_asset, base_url

    Empty strings are ignored (preserves existing values). Pass null to
    explicitly clear a field. Persists to andx_credentials.json.
    """
    try:
        import andx_credentials
        body = request.json or {}
        new_status = andx_credentials.save(body)
        # Re-instantiate the andX client so the new creds take effect immediately
        # without a restart. The client is rebuilt lazily on next get_client() call.
        try:
            import exchange_client
            with exchange_client._client_lock:
                exchange_client._client_singleton = None
        except Exception:
            pass
        _tlog("system", "andX API credentials updated via dashboard")
        return jsonify({"ok": True, "status": new_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/andx/credentials/test", methods=["POST"])
def api_andx_credentials_test():
    """Hit andX's /balance/ endpoint with the configured creds to confirm
    they work. Returns balance on success, descriptive error on failure."""
    try:
        import andx_credentials
        return jsonify(andx_credentials.test_connection())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/andx/credentials/clear", methods=["POST"])
def api_andx_credentials_clear():
    """Wipe the on-disk credentials file. Bot reverts to .env values."""
    try:
        import andx_credentials
        andx_credentials.clear()
        _tlog("system", "andX API credentials cleared — bot reverts to .env")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------------
# Playwright browser-session endpoints — used when HYBRID_EXEC=andx_browser
# ----------------------------------------------------------------------
# The cookie-paste flow above only works for andX's documented REST API
# (BTC/ETH/ANDX1). To trade the full ~120-coin universe we drive a
# persistent Chromium session that's logged into platform.andx.one. The
# website's JS computes access-sign on every request from inside that
# session, so the bot can place orders via fetch()/UI without re-deriving
# any HMAC. These endpoints expose the lifecycle to the dashboard.

@app.route("/api/andx/pw_status", methods=["GET"])
def api_andx_pw_status():
    """Cheap status snapshot for the dashboard banner — does NOT block."""
    try:
        from playwright_session import get_session
        s = get_session().status()
        return jsonify({
            "state": s.state.value,
            "headless": s.headless,
            "profile_dir": s.profile_dir,
            "last_error": s.last_error,
            "started_at": s.started_at,
            "last_request_at": s.last_request_at,
            "pending": s.pending,
            "fetch_route_dead": s.fetch_route_dead,
        })
    except Exception as e:
        return jsonify({"state": "error", "error": str(e)}), 500


@app.route("/api/andx/pw_start", methods=["POST"])
def api_andx_pw_start():
    """Launch the persistent Chromium context. Body: {"headless": bool}.
    First-time login REQUIRES headless=false so a real window opens and
    you can sign in by hand. Cookies persist into _pw_andx_profile/, so
    subsequent calls can use headless=true."""
    try:
        from playwright_session import get_session
        body = request.json or {}
        headless = bool(body.get("headless", False))
        st = get_session().start(headless=headless)
        _tlog("system", f"andX browser session: start(headless={headless}) -> {st.state.value}")
        return jsonify({"ok": True, "state": st.state.value, "headless": st.headless})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/andx/pw_stop", methods=["POST"])
def api_andx_pw_stop():
    """Tear down the Chromium context. Cookies stay on disk."""
    try:
        from playwright_session import get_session
        get_session().stop()
        _tlog("system", "andX browser session stopped")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/andx/pw_check_login", methods=["POST"])
def api_andx_pw_check_login():
    """Force a fresh probe of the session — useful after the user clicks
    'I'm logged in' in the dashboard. Blocks up to ~30s."""
    try:
        from playwright_session import get_session
        ok = get_session().is_logged_in(force_refresh=True)
        _tlog("system", f"andX browser session: login probe -> {ok}")
        return jsonify({"logged_in": ok})
    except Exception as e:
        return jsonify({"logged_in": False, "error": str(e)}), 500


@app.route("/api/andx/pw_test_order", methods=["POST"])
def api_andx_pw_test_order():
    """Manually fire a single order through the browser session — useful
    for iterating on UI selectors without waiting for the trader to find
    a signal. Body: {"base": "DOGE", "side": "buy", "qty": 100,
    "price_hint": 0.09}. Returns the full PWOrderResult dict."""
    try:
        from playwright_session import get_session
        body = request.json or {}
        base = (body.get("base") or "DOGE").upper()
        side = (body.get("side") or "buy").lower()
        qty = float(body.get("qty") or 1.0)
        price_hint = float(body.get("price_hint") or 0.0)
        if price_hint <= 0:
            from exchange_client import get_client
            price_hint = get_client().get_price(f"{base}/USDT") or 0.0
        # Translate (side, qty) into the currency-swap shape.
        if side == "buy":
            buy_curr, sell_curr = base, "USDT"
            buy_amount = qty
            sell_amount = qty * price_hint
        else:
            buy_curr, sell_curr = "USDT", base
            buy_amount = qty * price_hint
            sell_amount = qty
        s = get_session()
        res = s.place_order(
            buy_currency=buy_curr, sell_currency=sell_curr,
            buy_amount=buy_amount, sell_amount=sell_amount,
            visible_price=price_hint,
        )
        return jsonify({
            "ok": res.ok,
            "route": res.route.value,
            "http_status": res.http_status,
            "order_id": res.order_id,
            "filled_qty": res.filled_qty,
            "filled_price": res.filled_price,
            "status": res.status,
            "error": res.error,
            "raw": res.raw,
            "sent_body": res.sent_body,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/andx/volume", methods=["GET"])
def api_andx_volume():
    """Snapshot the user's leaderboard row (rank/award/volume).
    Reads URL + email-fragment from env (ANDX_LEADERBOARD_URL,
    ANDX_LEADERBOARD_EMAIL_FRAGMENT) so the dashboard can poll it cheaply."""
    try:
        from playwright_session import get_session
        url = os.environ.get("ANDX_LEADERBOARD_URL", "").strip()
        frag = os.environ.get("ANDX_LEADERBOARD_EMAIL_FRAGMENT", "nick").strip()
        if not url:
            return jsonify({"ok": False,
                            "error": "set ANDX_LEADERBOARD_URL in .env"}), 400
        out = get_session().snapshot_leaderboard_volume(url, frag, timeout_s=20.0)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/andx/pw_inspect", methods=["POST"])
def api_andx_pw_inspect():
    """Diagnostic: dump every visible button, input, and tab on the
    /instant/trade/USDT_<base> page so we can write accurate selectors.
    Body: {"base": "SUSHI"}. Returns title/url/buttons/inputs/tabs."""
    try:
        from playwright_session import get_session
        body = request.json or {}
        base = (body.get("base") or "BTC").upper()
        out = get_session().inspect_trade_page(base, timeout_s=20.0)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/andx/pw_wipe", methods=["POST"])
def api_andx_pw_wipe():
    """Delete the persisted profile so the next pw_start launches a fresh
    browser. Only legal when the session is stopped."""
    try:
        from playwright_session import get_session
        get_session().wipe_profile()
        _tlog("system", "andX browser profile wiped")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------------------------------------------------------------
# ANDX1 high-frequency engine — separate thread that scalps ANDX1 (0% fees)
# Reaches the 200-trades/day target via the only fee-free market on andX.
# ----------------------------------------------------------------------

@app.route("/api/andx1_engine/status", methods=["GET"])
def api_andx1_engine_status():
    try:
        import andx1_engine
        return jsonify(andx1_engine.status())
    except Exception as e:
        return jsonify({"error": str(e), "running": False}), 500


@app.route("/api/andx1_engine/start", methods=["POST"])
def api_andx1_engine_start():
    try:
        import andx1_engine
        r = andx1_engine.start()
        _tlog("system", "ANDX1 engine STARTED — continuous 0%-fee scalp")
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/andx1_engine/stop", methods=["POST"])
def api_andx1_engine_stop():
    try:
        import andx1_engine
        r = andx1_engine.stop()
        _tlog("system", "ANDX1 engine STOPPED")
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/andx1_engine/configure", methods=["POST"])
def api_andx1_engine_configure():
    """POST {notional_usd?, harvest_pct?, sl_pct?} — runtime knobs.
    Each field optional; only the provided ones are applied."""
    try:
        import andx1_engine
        body = request.json or {}
        return jsonify(andx1_engine.configure(
            notional_usd=body.get("notional_usd"),
            harvest_pct=body.get("harvest_pct"),
            sl_pct=body.get("sl_pct"),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------------
# DIP mode + per-coin strategies + charts
# ----------------------------------------------------------------------

@app.route("/api/trader/set_dip_mode", methods=["POST"])
def api_set_dip_mode():
    """Toggle DIP-only mode. When on, the bot ONLY buys quality oversold-at-
    support setups (per coin_strategies.py dip_threshold) and DISABLES
    force_deploy automatically since the two contradict."""
    body = request.json or {}
    on = bool(body.get("enabled", True))
    with trader_lock:
        trader["dip_mode"] = on
        if on:
            trader["force_deploy"] = False
    _tlog("system",
          f"DIP MODE {'ON — only quality dips, force_deploy auto-disabled' if on else 'OFF — normal signal trading'}")
    return jsonify({"dip_mode": on, "force_deploy": trader.get("force_deploy", False)})


@app.route("/api/coins/strategies", methods=["GET"])
def api_coin_strategies():
    """Return the effective per-coin strategy table for the dashboard."""
    try:
        import coin_strategies
        return jsonify({"strategies": coin_strategies.list_all()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/coins/strategy/<coin>", methods=["POST"])
def api_set_coin_strategy(coin):
    """Save a user override for a specific coin. Body: any subset of
    {scalp_gate, sl_pct_min, sl_pct_max, tp_pct_min, tp_pct_max,
     harvest_threshold, min_hold_seconds, max_position_pct, dip_threshold}.
    Persists to coin_strategies.json."""
    try:
        import coin_strategies
        body = request.json or {}
        # Coerce numerics
        clean = {}
        for k, v in body.items():
            try:
                clean[k] = float(v) if "_pct" in k or k in ("harvest_threshold",) else int(v) if k in ("scalp_gate", "min_hold_seconds", "dip_threshold") else v
            except Exception:
                pass
        coin_strategies.save_override(coin, clean)
        return jsonify({"ok": True, "coin": coin.upper(), "strategy": coin_strategies.get(coin)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/coins/auto_tune", methods=["POST"])
def api_auto_tune_coins():
    """Apply the BTC formula to EVERY coin: derive per-coin strategy from its
    own fees + ATR + audit-worst-loss. Persists to coin_strategies.json.

    Body: {"persist": true|false} (default true), {"coin": "SOL"} for just one.
    Returns the derived strategies + the inputs (fees, ATR, audit cap) so the
    user can see WHY each coin was tuned the way it was.
    """
    try:
        import coin_strategies
        body = request.json or {}
        persist = bool(body.get("persist", True))
        single = body.get("coin")
        if single:
            res = coin_strategies.auto_tune_for_coin(single)
            if persist:
                coin_strategies.save_override(single, res["strategy"])
                coin_strategies.reload()
            _tlog("system", f"AUTO-TUNED {single}: harvest={res['strategy']['harvest_threshold']*100:.2f}% "
                            f"SL=[{res['strategy']['sl_pct_min']*100:.1f}-{res['strategy']['sl_pct_max']*100:.1f}%] "
                            f"(ATR={res['inputs']['atr_pct']*100:.2f}%)")
            return jsonify({"ok": True, "results": {single.upper(): res}})
        # All coins
        results = coin_strategies.auto_tune_universe(persist=persist)
        _tlog("system", f"AUTO-TUNED {len(results)} coins using BTC formula "
                        f"(fee + ATR + audit-worst-loss)")
        return jsonify({"ok": True, "results": results, "persisted": persist})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[:600]}), 500


@app.route("/api/charts/<path:symbol>", methods=["GET"])
def api_chart(symbol):
    """Return recent OHLC candles + entry/exit markers for ONE symbol.

    Response shape:
      {
        "symbol": "BTC/USD",
        "candles": [{"t": iso, "o": ..., "h": ..., "l": ..., "c": ..., "v": ...}, ...],
        "trades": [
          {"t": iso, "type": "ENTRY"|"EXIT", "side": "long"|"short",
           "price": float, "qty": float, "pnl": float|null, "reason": str|null},
          ...
        ]
      }
    """
    try:
        import portfolio_live
        # Normalize symbol: accept BTC, BTC/USD, BTC/USDT
        sym = symbol.upper()
        if "/" not in sym:
            sym = f"{sym}/USD"
        # Candles from the data side via hybrid_client
        try:
            from hybrid_client import HybridClient
            hc = HybridClient.from_env()
            df = hc.get_candles(sym, timeframe="5m", limit=180)
        except Exception as e:
            df = None
        candles = []
        if df is not None and len(df) > 0:
            # The data adapter may put the timestamp on the index (most common
            # for Alpaca) OR in a column. reset_index gives us both options.
            # The original index name (if any) becomes the new column name;
            # if unnamed, pandas calls it "index".
            orig_index_name = df.index.name
            df2 = df.reset_index()
            # Find the timestamp column by tasting several common names
            time_col = None
            for cand in (orig_index_name, "Date", "timestamp", "time", "Datetime", "datetime", "index"):
                if cand and cand in df2.columns:
                    time_col = cand
                    break
            for _, row in df2.iterrows():
                t = row.get(time_col) if time_col else None
                try:
                    if hasattr(t, "isoformat"):
                        t_iso = t.isoformat()
                    elif t is None or (hasattr(t, "__class__") and t.__class__.__name__ == "NaTType"):
                        continue
                    else:
                        t_iso = str(t)
                except Exception:
                    continue
                candles.append({
                    "t": t_iso,
                    "o": float(row.get("Open", row.get("open", 0)) or 0),
                    "h": float(row.get("High", row.get("high", 0)) or 0),
                    "l": float(row.get("Low", row.get("low", 0)) or 0),
                    "c": float(row.get("Close", row.get("close", 0)) or 0),
                    "v": float(row.get("Volume", row.get("volume", 0)) or 0),
                })
        # Trades: pull from portfolio_live meta — history has the basic record,
        # entries_by_symbol has the full signal snapshot (what indicators fired),
        # closed_trades has exit PnL %.
        try:
            meta = portfolio_live._load_meta()
        except Exception:
            meta = {"history": [], "closed_trades": [], "entries_by_symbol": {}}

        # Index entries (with signal snapshots) by date for quick join
        base = sym.split("/")[0]
        entry_lookup = {}
        for s_key, entries in (meta.get("entries_by_symbol") or {}).items():
            if s_key.split("/")[0] != base:
                continue
            for e in entries:
                # Compose a key from date — used to join history records to
                # their original signal_snapshot
                k = e.get("date")
                if k:
                    entry_lookup[k] = e
        # Index closed_trades by date_sold too (gives us pnl_pct + signal_snapshot at exit)
        exit_lookup = {}
        for c in (meta.get("closed_trades") or []):
            if c.get("symbol", "").split("/")[0] != base:
                continue
            k = c.get("date_sold")
            if k:
                exit_lookup[k] = c

        def _extract_entry_detail(e):
            """Pull the human-readable WHY from a signal snapshot — what made the
            bot enter? Includes scalp score breakdown, dip detector reasons,
            key indicators, the original SL/TP targets, and ML confidence."""
            if not isinstance(e, dict):
                return {}
            snap = e.get("signal_snapshot") or {}
            ind = snap.get("indicators") or {}
            return {
                "signal": snap.get("signal"),
                "scalp_score": snap.get("scalp_score"),
                "scalp_reasons": snap.get("scalp_reasons") or [],
                "dip_score": snap.get("dip_score"),
                "dip_reasons": snap.get("dip_reasons") or [],
                "is_dip": snap.get("is_dip"),
                "ml_confidence": e.get("ml_confidence") or snap.get("ml_confidence"),
                "weekly_trend": snap.get("weekly_trend"),
                "regime": (snap.get("trend_structure") or {}).get("trend"),
                "rsi": ind.get("rsi"),
                "macd_hist": ind.get("macd_hist"),
                "stoch_k": ind.get("stoch_k"),
                "bb_pctb": ind.get("bb_pctb"),
                "cmf": ind.get("cmf"),
                "stop_loss": e.get("stop_loss"),
                "take_profit": e.get("take_profit"),
                "order_id": e.get("order_id"),
            }

        trades = []
        for h in (meta.get("history") or []):
            h_sym = h.get("symbol", "")
            if h_sym.split("/")[0] != base:
                continue
            t_iso = h.get("date") or ""
            ttype = (h.get("type") or "").upper()
            if ttype in ("BUY", "SHORT"):
                detail = _extract_entry_detail(entry_lookup.get(t_iso))
                trades.append({
                    "t": t_iso,
                    "type": "ENTRY",
                    "side": "long" if ttype == "BUY" else "short",
                    "price": h.get("price"),
                    "qty": h.get("qty"),
                    "dollar_amount": (h.get("price") or 0) * (h.get("qty") or 0),
                    "stop_loss": h.get("stop_loss"),
                    "take_profit": h.get("take_profit"),
                    "order_id": h.get("order_id"),
                    "detail": detail,
                })
            elif ttype in ("SELL", "COVER"):
                closed = exit_lookup.get(t_iso) or {}
                # Try matching SELLs to their original BUY for pnl %
                entry_price = closed.get("entry_price")
                pnl = h.get("pnl") if h.get("pnl") is not None else closed.get("pnl")
                pnl_pct = closed.get("pnl_pct")
                if pnl_pct is None and entry_price and h.get("price") and entry_price > 0:
                    p = float(h.get("price"))
                    if ttype == "SELL":
                        pnl_pct = (p - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - p) / entry_price * 100
                # NET P&L: subtract round-trip fee (entry-side + exit-side).
                # Fee is a fraction of NOTIONAL, so we use price × qty for the
                # exit side and approximate the entry-side notional with the
                # same qty × entry_price.
                import coin_strategies as _cs_chart
                fee_rt = _cs_chart.round_trip_fee(sym)
                exit_notional = float(h.get("price") or 0) * float(h.get("qty") or 0)
                entry_notional = float(entry_price or 0) * float(h.get("qty") or 0)
                fee_cost = (entry_notional + exit_notional) / 2.0 * fee_rt
                pnl_net = (pnl - fee_cost) if pnl is not None else None
                pnl_pct_net = (pnl_pct - fee_rt * 100) if pnl_pct is not None else None
                trades.append({
                    "t": t_iso,
                    "type": "EXIT",
                    "side": "long" if ttype == "SELL" else "short",
                    "price": h.get("price"),
                    "qty": h.get("qty"),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "pnl_net": pnl_net,
                    "pnl_pct_net": pnl_pct_net,
                    "fee_cost": fee_cost,
                    "entry_price": entry_price,
                    "reason": h.get("exit_reason") or closed.get("exit_reason"),
                    "order_id": h.get("order_id"),
                })
        # Sort by time
        trades.sort(key=lambda x: x["t"])
        return jsonify({"symbol": sym, "candles": candles, "trades": trades})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[:600]}), 500


@app.route("/api/charts/symbols", methods=["GET"])
def api_chart_symbols():
    """List the symbols worth charting: currently-held + recently-traded."""
    out: set = set()
    try:
        ps = get_portfolio_summary()
        for s in (ps.get("positions") or {}).keys():
            out.add(s)
    except Exception:
        pass
    try:
        import portfolio_live
        meta = portfolio_live._load_meta()
        for h in (meta.get("history") or [])[-40:]:
            s = h.get("symbol")
            if s:
                out.add(s)
    except Exception:
        pass
    return jsonify({"symbols": sorted(out)})


@app.route("/api/trader/set_trade_mode", methods=["POST"])
def api_set_trade_mode():
    """Direction filter: 'both' | 'long' | 'short'.
    both = take bullish AND bearish setups (default — SRI MATA-style).
    long = bullish only (safe spot-only fallback).
    short = bearish only (sim shows what shorting alone would do)."""
    body = request.json or {}
    tmode = str(body.get("mode", "")).lower().strip()
    if tmode not in ("both", "long", "short"):
        return jsonify({"error": "Invalid mode. Use: both, long, short"}), 400
    with trader_lock:
        prev = trader.get("trade_mode")
        trader["trade_mode"] = tmode
    _tlog("system", f"TRADE MODE: {prev} -> {tmode}")
    return jsonify({"trade_mode": tmode})


@app.route("/api/trader/set_auto_tp", methods=["POST"])
def api_set_auto_tp():
    body = request.json or {}
    enabled = bool(body.get("enabled", False))
    threshold = float(body.get("threshold", 100.0))
    try:
        ps = get_portfolio_summary()
        baseline = ps["realized_pnl"] + ps["unrealized_pnl"]
    except Exception:
        baseline = 0.0
    with trader_lock:
        trader["auto_tp_enabled"] = enabled
        trader["auto_tp_threshold"] = threshold
        trader["auto_tp_baseline"] = baseline
    return jsonify({"auto_tp_enabled": enabled, "threshold": threshold, "baseline": baseline})


@app.route("/api/trader/sell_all", methods=["POST"])
def api_sell_all():
    """Close EVERY open position. Runs sequentially (blocking) and returns a
    per-symbol result table. The UI shows this table so the user can see
    which coins actually sold and which failed (and why).

    Client note: this can take 30-60s per non-documented coin (browser UI
    path), so callers should use a long timeout. The dashboard shows a
    "selling..." spinner while it waits."""
    body = request.json or {}
    exit_reason = body.get("reason", "MANUAL SELL ALL")

    # Preflight: warn the caller if the browser sell route is offline. This
    # doesn't block the sell — docs-API sells (BTC/ETH/ANDX1) still work —
    # but alt sells will all fail without Chrome, and the user should know.
    browser_warning = None
    try:
        import socket
        s = socket.socket()
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", 9222))
            s.close()
        except Exception:
            browser_warning = ("Chrome debug window (port 9222) is DOWN. "
                               "Non-BTC/ETH/ANDX1 sells will fail. Launch "
                               "'Login to andX for Bot.bat' first to enable.")
    except Exception:
        pass

    _tlog("system", f"Sell-all requested: reason={exit_reason}")
    summary = _sell_all_positions(exit_reason=exit_reason)
    summary["browser_warning"] = browser_warning
    _tlog("system", f"Sell-all done: {summary['sold_count']} sold, "
                    f"{summary['failed_count']} failed, pnl=${summary['total_pnl']:+.2f}")
    return jsonify(summary)


@app.route("/api/trader/sell_one", methods=["POST"])
def api_sell_one():
    body = request.json or {}
    symbol = body.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"})
    snap = get_portfolio_summary()
    if symbol not in snap.get("positions", {}):
        return jsonify({"error": f"no position in {symbol}"})
    last_entry = _last_entry_for(symbol)
    res = _do_sell(symbol, sell_all=True,
                   signal_snapshot=last_entry.get("signal_snapshot"),
                   exit_reason="MANUAL", entry_data=last_entry)
    return jsonify(clean(res))


@app.route("/api/trader/reset_portfolio", methods=["POST"])
def api_reset_portfolio():
    body = request.json or {}
    starting_cash = body.get("starting_cash")
    p = reset_portfolio(starting_cash)
    return jsonify(clean(p))


@app.route("/api/trader/reset_session", methods=["POST"])
def api_reset_session():
    with trader_lock:
        trader["session_pnl"] = 0.0
        trader["session_peak"] = 0.0
        trader["session_trades"] = 0
        trader["session_wins"] = 0
        trader["session_losses"] = 0
        trader["session_start"] = datetime.utcnow().isoformat() + "Z"
    _tlog("system", "Session counters reset")
    return jsonify({"ok": True})


@app.route("/api/trader/scan_results")
def api_scan_results():
    return jsonify(clean({"results": _last_scan_results[:20]}))


@app.route("/api/trader/log")
def api_log():
    with trader_lock:
        return jsonify({"log": list(trader["log"])})


@app.route("/api/learning")
def api_learning():
    try:
        return jsonify(clean(get_learning_summary()))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/ticker_memory")
def api_ticker_memory():
    try:
        return jsonify(clean(get_ticker_memory_summary()))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/universe/refresh", methods=["POST"])
def api_universe_refresh():
    u = get_universe(force_refresh=True)
    return jsonify({"universe": u, "count": len(u)})


@app.route("/api/exchange/healthcheck")
def api_exchange_healthcheck():
    client = get_client()
    out = {
        "name": client.name,
        "quote_asset": client.quote_asset,
        "connected": False,
        "balance": None,
        "btc_price": None,
    }
    try:
        out["connected"] = client.is_connected()
    except Exception as e:
        out["connect_error"] = str(e)
    try:
        bal = client.get_balance()
        out["balance"] = {"free": bal.free, "total": bal.total, "asset": bal.quote_asset}
    except Exception as e:
        out["balance_error"] = str(e)
    try:
        out["btc_price"] = client.get_price(f"BTC/{client.quote_asset}")
    except Exception as e:
        out["price_error"] = str(e)
    return jsonify(out)


@app.route("/api/portfolio/summary")
def api_portfolio_summary():
    try:
        return jsonify(clean(get_portfolio_summary()))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/portfolio/history")
def api_portfolio_history():
    return jsonify(clean({"history": get_trade_history(50)}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(f"=" * 60)
    print(f" CryptoBot — 24/7 high-volume scalper")
    print(f" Sibling to SRI MATA. Crypto-only. Plug-in andX.")
    print(f"=" * 60)
    client = get_client()
    print(f" Exchange: {client.name}   (quote={client.quote_asset})")
    print(f" Open:     http://localhost:{PORT}")
    print(f"=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
