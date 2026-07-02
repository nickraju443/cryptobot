"""
Self-Learning Module
Tracks which indicators predicted correctly and adjusts weights over time.
Learns from trade context: patterns, trends, time-of-day, hold duration, exit reason.
Tracks candlestick pattern outcomes per trend context.
Records 50+ features per trade for ML model training.
The more trades the bot makes, the smarter it gets.

Saves learning data to learning_data.json.
Saves deep trade history to trade_history.json.
Saves strategy learned data to strategy_learned.json.
"""

import json
import logging
import threading
import tempfile
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LEARNING_FILE = Path(__file__).parent / "learning_data.json"

_write_lock = threading.Lock()

def _atomic_json_write(filepath: Path, data):
    """Write JSON atomically with retry for Windows file locks."""
    with _write_lock:
        dir_path = filepath.parent
        for attempt in range(3):
            try:
                fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                # On Windows, need to remove target first
                if filepath.exists():
                    filepath.unlink()
                os.rename(tmp_path, str(filepath))
                return  # success
            except OSError:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                if attempt < 2:
                    time.sleep(0.1)  # brief wait for lock to release
                else:
                    logger.warning(f"Failed to write {filepath} after 3 attempts")
            except Exception as e:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise e

# All indicator categories tracked
CATEGORIES = [
    "sma", "ema", "rsi", "rsi_divergence", "macd", "stochastic",
    "adx", "bollinger", "ichimoku", "obv", "cmf", "volume", "vwap",
    "price_action", "range_52w",
]


def _load() -> dict:
    if LEARNING_FILE.exists():
        with open(LEARNING_FILE, "r") as f:
            return json.load(f)
    return _new_data()


def _save(data: dict):
    _atomic_json_write(LEARNING_FILE, data)


def _new_data() -> dict:
    data = {
        "version": 1,
        "total_trades_analyzed": 0,
        "categories": {},
        "trade_log": [],  # last N analyzed trades
        "weight_overrides": {},  # current computed weights
        "last_updated": datetime.now().isoformat(),
    }
    for cat in CATEGORIES:
        data["categories"][cat] = {
            "total_signals": 0,
            "correct_signals": 0,
            "incorrect_signals": 0,
            "accuracy": 0.5,  # start neutral
            "weight_multiplier": 1.0,  # start at default
            "recent_correct": 0,  # last 20 trades
            "recent_total": 0,
        }
    _save(data)
    return data


def record_trade(trade: dict, buy_signal_categories: dict):
    """
    Record a completed trade and update indicator accuracy.

    Args:
        trade: dict with at least {pnl, pnl_pct, ticker, buy_price, sell_price}
        buy_signal_categories: dict from analysis result's signal_categories
            Maps category -> score (positive = bullish, negative = bearish)
    """
    data = _load()
    was_profitable = trade.get("pnl", 0) > 0
    pnl_pct = trade.get("pnl_pct", 0)

    if not buy_signal_categories:
        return

    trade_record = {
        "ticker": trade.get("ticker", "?"),
        "pnl": trade.get("pnl", 0),
        "pnl_pct": pnl_pct,
        "profitable": was_profitable,
        "date": datetime.now().isoformat(),
        "categories_at_entry": buy_signal_categories,
    }

    data["trade_log"].append(trade_record)
    # Keep last 200 trades
    if len(data["trade_log"]) > 200:
        data["trade_log"] = data["trade_log"][-200:]

    data["total_trades_analyzed"] += 1

    for category, score in buy_signal_categories.items():
        if category not in data["categories"]:
            data["categories"][category] = {
                "total_signals": 0,
                "correct_signals": 0,
                "incorrect_signals": 0,
                "accuracy": 0.5,
                "weight_multiplier": 1.0,
                "recent_correct": 0,
                "recent_total": 0,
            }

        cat_data = data["categories"][category]

        # A signal was "correct" if:
        # - It was bullish (score > 0) and trade was profitable
        # - It was bearish (score < 0) and trade was unprofitable (correctly warned)
        was_bullish = score > 0
        signal_correct = (was_bullish and was_profitable) or (not was_bullish and not was_profitable)

        cat_data["total_signals"] += 1
        if signal_correct:
            cat_data["correct_signals"] += 1
        else:
            cat_data["incorrect_signals"] += 1

        # Compute accuracy with exponential weighting (recent trades matter more)
        # EMA-style: new_accuracy = alpha * latest + (1 - alpha) * old_accuracy
        alpha = 0.08  # STABLE learning — adapts gradually, prevents flip-flopping
        old_acc = cat_data["accuracy"]
        new_point = 1.0 if signal_correct else 0.0

        # Weight by magnitude of P&L - big wins/losses teach more
        magnitude = min(abs(pnl_pct) / 5.0, 1.5)  # cap at 1.5x
        effective_alpha = alpha * max(magnitude, 0.5)
        effective_alpha = min(effective_alpha, 0.10)  # cap at 10% influence per trade (was 20%)

        cat_data["accuracy"] = round(old_acc * (1 - effective_alpha) + new_point * effective_alpha, 4)

    # Recompute weight multipliers based on accuracy
    _recompute_weights(data)
    data["last_updated"] = datetime.now().isoformat()
    _save(data)


def _recompute_weights(data: dict):
    """Adjust weight multipliers based on indicator accuracy."""
    for cat_name, cat_data in data["categories"].items():
        acc = cat_data["accuracy"]
        total = cat_data["total_signals"]

        if total < 50:
            # Need at least 50 trades before adjusting — prevents early noise
            cat_data["weight_multiplier"] = 1.0
            continue

        # Map accuracy to weight multiplier (tight range to prevent flip-flopping):
        # 0.35 accuracy -> 0.85x weight
        # 0.50 accuracy -> 1.0x weight (neutral)
        # 0.65 accuracy -> 1.15x weight
        # Clamped to [0.85, 1.15] — gentle nudges only, ML model handles the rest
        multiplier = 0.85 + (acc - 0.35) * 1.0
        multiplier = max(0.85, min(1.15, multiplier))

        # Reduce confidence in the adjustment if we have few trades
        # (blend toward 1.0 when trade count is low — needs 200 trades to fully trust)
        confidence = min(total / 200, 1.0)
        multiplier = 1.0 + (multiplier - 1.0) * confidence

        cat_data["weight_multiplier"] = round(multiplier, 3)

    # Update the weight_overrides dict used by analysis engine
    data["weight_overrides"] = {
        cat: data["categories"][cat]["weight_multiplier"]
        for cat in data["categories"]
        if data["categories"][cat]["weight_multiplier"] != 1.0
    }


def get_weight_overrides() -> dict:
    """Get current weight overrides for the analysis engine."""
    data = _load()
    return data.get("weight_overrides", {})


def get_learning_summary() -> dict:
    """Get a summary of what the bot has learned."""
    data = _load()

    categories_summary = {}
    for cat_name, cat_data in data["categories"].items():
        if cat_data["total_signals"] > 0:
            categories_summary[cat_name] = {
                "accuracy": round(cat_data["accuracy"] * 100, 1),
                "weight": cat_data["weight_multiplier"],
                "total_signals": cat_data["total_signals"],
                "correct": cat_data["correct_signals"],
                "incorrect": cat_data["incorrect_signals"],
            }

    # Sort by accuracy (best performing first)
    sorted_cats = dict(sorted(
        categories_summary.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    ))

    # Recent trade performance
    recent = data["trade_log"][-20:] if data["trade_log"] else []
    recent_wins = sum(1 for t in recent if t["profitable"])
    recent_total = len(recent)
    recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 0
    recent_avg_pnl = sum(t["pnl_pct"] for t in recent) / recent_total if recent_total > 0 else 0

    return {
        "total_trades_analyzed": data["total_trades_analyzed"],
        "categories": sorted_cats,
        "weight_overrides": data.get("weight_overrides", {}),
        "recent_20_win_rate": round(recent_win_rate, 1),
        "recent_20_avg_pnl_pct": round(recent_avg_pnl, 2),
        "last_updated": data.get("last_updated", "never"),
    }


def reset_learning():
    """Reset all learning data."""
    _new_data()
    global _ticker_stats
    _ticker_stats = {}
    _save_ticker_stats()


# ==========================================
# PERSISTENT TICKER MEMORY
# Survives restarts — learns which tickers make money
# ==========================================

TICKER_MEMORY_FILE = Path(__file__).parent / "ticker_memory.json"
_ticker_stats = None  # lazy-loaded from disk


def _load_ticker_stats() -> dict:
    """Load ticker stats from disk. Called lazily on first access."""
    global _ticker_stats
    if _ticker_stats is not None:
        return _ticker_stats
    if TICKER_MEMORY_FILE.exists():
        try:
            with open(TICKER_MEMORY_FILE, "r") as f:
                _ticker_stats = json.load(f)
            return _ticker_stats
        except Exception:
            pass
    _ticker_stats = {}
    return _ticker_stats


def _save_ticker_stats():
    """Persist ticker stats to disk."""
    if _ticker_stats is None:
        return
    try:
        with open(TICKER_MEMORY_FILE, "w") as f:
            json.dump(_ticker_stats, f, indent=2)
    except Exception:
        pass


def _get_ticker(ticker: str) -> dict:
    """Get or create a ticker entry."""
    stats = _load_ticker_stats()
    if ticker not in stats:
        stats[ticker] = {
            "wins": 0, "losses": 0, "streak": 0,
            "last_loss": None, "total_pnl": 0.0,
            "avg_pnl_pct": 0.0, "total_trades": 0,
            "last_traded": None,
        }
    return stats[ticker]


def record_ticker_result(ticker: str, won: bool):
    """Track per-ticker win/loss for blacklisting and streak bonuses. Persists to disk."""
    s = _get_ticker(ticker)
    if won:
        s["wins"] += 1
        s["streak"] = max(s["streak"], 0) + 1
    else:
        s["losses"] += 1
        s["streak"] = min(s["streak"], 0) - 1
        s["last_loss"] = datetime.now().isoformat()
    s["last_traded"] = datetime.now().isoformat()
    _save_ticker_stats()


def record_ticker_pnl(ticker: str, pnl: float, pnl_pct: float):
    """Track cumulative P&L per ticker for profitability scoring."""
    s = _get_ticker(ticker)
    s["total_pnl"] = round(s.get("total_pnl", 0) + pnl, 2)
    s["total_trades"] = s.get("total_trades", 0) + 1
    # Running average P&L %
    old_avg = s.get("avg_pnl_pct", 0)
    n = s["total_trades"]
    s["avg_pnl_pct"] = round(old_avg + (pnl_pct - old_avg) / n, 4)
    _save_ticker_stats()


def get_ticker_bonus(ticker: str) -> int:
    """Get confidence bonus/penalty based on historical profitability.
    Positive = historically profitable (favorite), negative = historically bad."""
    stats = _load_ticker_stats()
    s = stats.get(ticker)
    if not s:
        return 0

    total = s.get("wins", 0) + s.get("losses", 0)
    if total < 5:
        return 0  # not enough data

    win_rate = s["wins"] / total

    if win_rate >= 0.70 and total >= 10:
        return 10  # strong favorite
    elif win_rate >= 0.60 and total >= 5:
        return 5   # favorite
    elif win_rate < 0.35 and total >= 5:
        return -10  # historically bad, penalize
    return 0


def get_favorites() -> list[str]:
    """Get tickers with >60% win rate and 5+ trades — prioritized for scanning."""
    stats = _load_ticker_stats()
    favs = []
    for ticker, s in stats.items():
        total = s.get("wins", 0) + s.get("losses", 0)
        if total >= 5 and s["wins"] / total >= 0.60:
            favs.append(ticker)
    return favs


def get_ticker_memory_summary() -> dict:
    """Summary of ticker memory for logging."""
    stats = _load_ticker_stats()
    total_tickers = len(stats)
    traded = [t for t, s in stats.items() if s.get("total_trades", 0) > 0]
    favs = get_favorites()
    blacklisted = [t for t in stats if is_blacklisted(t)]
    best = sorted(stats.items(), key=lambda x: x[1].get("total_pnl", 0), reverse=True)[:5]
    worst = sorted(stats.items(), key=lambda x: x[1].get("total_pnl", 0))[:5]
    return {
        "total_tickers": total_tickers,
        "traded": len(traded),
        "favorites": favs,
        "blacklisted": blacklisted,
        "best_tickers": [(t, round(s.get("total_pnl", 0), 2)) for t, s in best if s.get("total_pnl", 0) != 0],
        "worst_tickers": [(t, round(s.get("total_pnl", 0), 2)) for t, s in worst if s.get("total_pnl", 0) != 0],
    }


def is_blacklisted(ticker: str) -> bool:
    """Check if a ticker should be avoided. More aggressive than before:
    - 2+ consecutive losses (was 3) — catch losers faster
    - >50% loss rate with 3+ trades (was 60% with 5+)
    - Net PnL deeply negative (new) — if we've lost $200+ on a ticker, avoid it
    Blacklist expires after 15 min (was 30) to give second chances sooner."""
    stats = _load_ticker_stats()
    s = stats.get(ticker)
    if not s:
        return False

    def _expired():
        """Check if blacklist has expired (15 min cooldown)."""
        if s["last_loss"]:
            try:
                last = datetime.fromisoformat(s["last_loss"])
                minutes_ago = (datetime.now() - last).total_seconds() / 60
                if minutes_ago > 15:
                    s["streak"] = 0
                    _save_ticker_stats()
                    return True
            except Exception:
                pass
        return False

    # 2+ losses in a row — catch losing streaks faster
    if s["streak"] <= -2:
        return not _expired()

    total = s["wins"] + s["losses"]

    # >50% loss rate with 3+ trades — historically bad
    if total >= 3 and s["losses"] / total > 0.5:
        return not _expired()

    # Net PnL deeply negative — this ticker costs us money
    if s.get("total_pnl", 0) < -200 and total >= 2:
        return not _expired()

    return False


def get_ticker_streak(ticker: str) -> int:
    """Get win/loss streak for a ticker. Positive = win streak, negative = loss streak."""
    stats = _load_ticker_stats()
    s = stats.get(ticker)
    return s["streak"] if s else 0


# ==========================================
# DEEP TRADE HISTORY & FEEDBACK LEARNING
# Learns from full trade context: patterns, trends, time, duration, exit reason
# ==========================================

TRADE_HISTORY_FILE = Path(__file__).parent / "trade_history.json"
STRATEGY_FILE = Path(__file__).parent / "strategy_learned.json"
_history_lock = threading.Lock()


def _load_history() -> list:
    with _history_lock:
        if TRADE_HISTORY_FILE.exists():
            try:
                with open(TRADE_HISTORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []
    return []


def _save_history(history: list):
    with _history_lock:
        # Keep last 2000 trades for ML training
        history = history[-2000:]
        _atomic_json_write(TRADE_HISTORY_FILE, history)


def _load_strategy() -> dict:
    for attempt in range(3):
        if STRATEGY_FILE.exists():
            try:
                with open(STRATEGY_FILE, "r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                if attempt < 2:
                    time.sleep(0.1)
                continue
    return _new_strategy()


def _save_strategy(strategy: dict):
    _atomic_json_write(STRATEGY_FILE, strategy)


def _new_strategy() -> dict:
    return {
        "version": 1,
        "last_analyzed": None,
        "trades_analyzed": 0,
        # Per-ticker learned behavior
        "ticker_profiles": {},
        # Pattern effectiveness
        "pattern_stats": {},
        # Trend context effectiveness
        "trend_stats": {"UPTREND": {"wins": 0, "losses": 0, "avg_pnl": 0},
                        "DOWNTREND": {"wins": 0, "losses": 0, "avg_pnl": 0},
                        "CONSOLIDATION": {"wins": 0, "losses": 0, "avg_pnl": 0}},
        # Time-of-day performance (hour buckets)
        "hour_stats": {},
        # Hold duration buckets (seconds)
        "duration_stats": {"short_0_120": {"wins": 0, "losses": 0, "avg_pnl": 0},
                          "medium_120_600": {"wins": 0, "losses": 0, "avg_pnl": 0},
                          "long_600_plus": {"wins": 0, "losses": 0, "avg_pnl": 0}},
        # Exit reason effectiveness
        "exit_stats": {},
        # TP/SL hit rates
        "tp_hit_rate": 0.0,
        "sl_hit_rate": 0.0,
        "smart_exit_rate": 0.0,
        # Learned adjustments applied to trading
        "adjustments": {
            "tp_multiplier": 1.0,   # scale TP distance
            "sl_multiplier": 1.0,   # scale SL distance
            "avoid_tickers": [],    # tickers to skip
            "prefer_tickers": [],   # tickers to prioritize
            "best_hours": [],       # hours with best win rate
            "worst_hours": [],      # hours with worst win rate
            "best_trends": [],      # trend contexts that work
        },
    }


def record_trade_context(ticker: str, side: str, entry_price: float, exit_price: float,
                         pnl: float, pnl_pct: float, shares: int, exit_reason: str,
                         entry_data: dict = None, ml_features: dict = None):
    """Record full trade context with 50+ features for ML training.

    Args:
        ml_features: dict with full indicator snapshot from entry time.
            Keys: rsi, macd, macd_signal, macd_hist, macd_hist_prev, stoch_k, stoch_d,
            adx, bb_pctb, cmf, rvol, vwap, sma_20, sma_50, sma_200, ema_9, ema_21,
            price_vs_vwap_pct, price_vs_sma20_pct, price_vs_ema9_pct,
            confidence, scalp_score, risk_reward, weekly_trend, mtf_aligned,
            market_bias, vix_level, market_regime, pattern_names,
            consecutive_candle_color, body_atr_ratio, upper_shadow_ratio, lower_shadow_ratio,
            fib_proximity, sr_distance_pct, open_position_count, session_pnl_at_entry,
            ticker_sector, ticker_win_rate, ticker_streak, ichimoku_position, 52w_range_pct,
            ml_confidence
    """
    entry_data = entry_data or {}
    ml_features = ml_features or {}

    record = {
        "ticker": ticker,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 4),
        "shares": shares,
        "won": pnl > 0,
        "exit_reason": exit_reason,
        "date": datetime.now().isoformat(),
        "hour": datetime.now().hour,
        "minute": datetime.now().minute,
        # Entry context (backward compatible)
        "trend": entry_data.get("trend", ml_features.get("trend", "UNKNOWN")),
        "candle_score": entry_data.get("candle_score", 0),
        "atr_pct": entry_data.get("atr_pct", 0),
        "signal_snapshot": entry_data.get("signal_snapshot", {}),
        "stop_loss": entry_data.get("stop_loss", 0),
        "take_profit": entry_data.get("take_profit", 0),
        "entry_date": entry_data.get("date", ""),
        "held_seconds": 0,
        # === ML FEATURES (50+) ===
        # Indicator values at entry
        "rsi": ml_features.get("rsi"),
        "macd": ml_features.get("macd"),
        "macd_signal": ml_features.get("macd_signal"),
        "macd_hist": ml_features.get("macd_hist"),
        "macd_hist_prev": ml_features.get("macd_hist_prev"),
        "stoch_k": ml_features.get("stoch_k"),
        "stoch_d": ml_features.get("stoch_d"),
        "adx": ml_features.get("adx"),
        "bb_pctb": ml_features.get("bb_pctb"),
        "cmf": ml_features.get("cmf"),
        "rvol": ml_features.get("rvol"),
        # Price relative to key levels
        "price_vs_vwap_pct": ml_features.get("price_vs_vwap_pct"),
        "price_vs_sma20_pct": ml_features.get("price_vs_sma20_pct"),
        "price_vs_ema9_pct": ml_features.get("price_vs_ema9_pct"),
        # Entry scores
        "confidence": ml_features.get("confidence"),
        "scalp_score": ml_features.get("scalp_score"),
        "risk_reward": ml_features.get("risk_reward"),
        # Multi-timeframe
        "weekly_trend": ml_features.get("weekly_trend", "N/A"),
        "mtf_aligned": ml_features.get("mtf_aligned"),
        # Market context
        "market_bias": ml_features.get("market_bias", "NEUTRAL"),
        "vix_level": ml_features.get("vix_level"),
        "market_regime": ml_features.get("market_regime", "NORMAL"),
        # Candlestick patterns
        "pattern_names": ml_features.get("pattern_names", []),
        "consecutive_candle_color": ml_features.get("consecutive_candle_color", 0),
        "body_atr_ratio": ml_features.get("body_atr_ratio"),
        "upper_shadow_ratio": ml_features.get("upper_shadow_ratio"),
        "lower_shadow_ratio": ml_features.get("lower_shadow_ratio"),
        # Key levels
        "fib_proximity": ml_features.get("fib_proximity"),
        "sr_distance_pct": ml_features.get("sr_distance_pct"),
        # Portfolio context
        "open_position_count": ml_features.get("open_position_count", 0),
        "session_pnl_at_entry": ml_features.get("session_pnl_at_entry", 0),
        # Time features
        "day_of_week": ml_features.get("day_of_week", datetime.now().weekday()),
        "minutes_since_open": ml_features.get("minutes_since_open", 0),
        # Ticker context
        "ticker_sector": ml_features.get("ticker_sector", "UNKNOWN"),
        "ticker_win_rate": ml_features.get("ticker_win_rate", 0.5),
        "ticker_streak": ml_features.get("ticker_streak", 0),
        # Technical position
        "ichimoku_position": ml_features.get("ichimoku_position", "UNKNOWN"),
        "52w_range_pct": ml_features.get("52w_range_pct"),
        # ML confidence at entry (if model was active)
        "ml_confidence": ml_features.get("ml_confidence"),
    }

    # Calculate hold duration
    if record["entry_date"]:
        try:
            entry_time = datetime.fromisoformat(record["entry_date"])
            record["held_seconds"] = int((datetime.now() - entry_time).total_seconds())
        except Exception:
            pass

    history = _load_history()

    # Dedupe: skip if same ticker+side+pnl+entry_price recorded in last 10 minutes
    dominated = False
    for h in history[-50:]:
        if (h.get("ticker") == record["ticker"]
            and h.get("side") == record["side"]
            and abs(h.get("pnl", 0) - record["pnl"]) < 1
            and abs(h.get("entry_price", 0) - record["entry_price"]) < 0.01):
            dominated = True
            break
    if dominated:
        return

    history.append(record)
    _save_history(history)

    # Record pattern outcomes for pattern learning
    pattern_names = ml_features.get("pattern_names", [])
    trend = record["trend"]
    if pattern_names:
        record_pattern_outcome(pattern_names, trend, record["won"], record["pnl_pct"])

    # Re-analyze every 5 trades (was 10 — faster learning)
    if len(history) % 5 == 0 and len(history) >= 15:
        analyze_and_learn()

    # Trigger ML retraining check
    try:
        from ml_engine import maybe_retrain
        ml_trade_count = sum(1 for t in history if t.get("rsi") is not None)
        maybe_retrain(ml_trade_count)
    except Exception as e:
        logger.debug(f"ML retrain check failed: {e}")


def record_pattern_outcome(pattern_names: list, trend: str, won: bool, pnl_pct: float):
    """Record outcome for each candlestick pattern present at entry.
    Tracks per-pattern and per-pattern+trend win rates."""
    strategy = _load_strategy()
    if "pattern_stats" not in strategy:
        strategy["pattern_stats"] = {}

    for name in pattern_names:
        if name not in strategy["pattern_stats"]:
            strategy["pattern_stats"][name] = {
                "total": 0, "wins": 0, "win_rate": 0.5,
                "avg_pnl_pct": 0.0, "by_trend": {},
            }
        ps = strategy["pattern_stats"][name]
        ps["total"] += 1
        if won:
            ps["wins"] += 1
        ps["win_rate"] = round(ps["wins"] / ps["total"], 3) if ps["total"] > 0 else 0.5
        # Running average PnL
        old_avg = ps.get("avg_pnl_pct", 0)
        ps["avg_pnl_pct"] = round(old_avg + (pnl_pct - old_avg) / ps["total"], 6)

        # Per-trend breakdown
        if trend not in ps["by_trend"]:
            ps["by_trend"][trend] = {"total": 0, "wins": 0, "win_rate": 0.5}
        bt = ps["by_trend"][trend]
        bt["total"] += 1
        if won:
            bt["wins"] += 1
        bt["win_rate"] = round(bt["wins"] / bt["total"], 3) if bt["total"] > 0 else 0.5

    _save_strategy(strategy)


def get_pattern_reliability(pattern_name: str, trend: str = None) -> float:
    """Get learned reliability score for a candlestick pattern.
    Returns 0.0-1.0. Returns 0.5 (neutral) if insufficient data."""
    strategy = _load_strategy()
    ps = strategy.get("pattern_stats", {}).get(pattern_name)
    if not ps or ps.get("total", 0) < 5:
        return 0.5  # Not enough data

    # Use trend-specific rate if available (3+ trades)
    if trend and trend in ps.get("by_trend", {}):
        bt = ps["by_trend"][trend]
        if bt.get("total", 0) >= 3:
            return bt["win_rate"]

    return ps["win_rate"]


def get_ticker_win_rate(ticker: str) -> float:
    """Get historical win rate for a ticker. Returns 0.5 if insufficient data."""
    stats = _load_ticker_stats()
    s = stats.get(ticker)
    if not s:
        return 0.5
    total = s.get("wins", 0) + s.get("losses", 0)
    if total < 3:
        return 0.5
    return s["wins"] / total


def get_optimal_quality_gate() -> int:
    """Return the lowest score bucket with >55% win rate.
    Buckets match real scalp_score distribution (10-30 range, not the old 40-80)."""
    strategy = _load_strategy()
    buckets = strategy.get("score_buckets", {})

    # Check from lowest to highest — real edge kicks in at scalp 20+
    for bucket_name in ["10_15", "15_20", "20_25", "25_30", "30_plus"]:
        b = buckets.get(bucket_name, {})
        if b.get("trades", 0) >= 5 and b.get("win_rate", 0) > 0.55:
            return int(bucket_name.split("_")[0])

    return 20  # Default floor: the 65% WR zone from 2k-trade analysis


def get_setup_profile(pattern_name: str, trend: str) -> dict:
    """Get learned exit timing profile for a pattern+trend combo."""
    strategy = _load_strategy()
    key = f"{pattern_name}|{trend}"
    return strategy.get("setup_profiles", {}).get(key, {})


def analyze_and_learn():
    """Analyze all trade history and update strategy. Runs every 5 trades."""
    history = _load_history()
    if len(history) < 15:
        return

    strategy = _load_strategy()
    strategy["trades_analyzed"] = len(history)
    strategy["last_analyzed"] = datetime.now().isoformat()

    # --- Per-ticker profiles ---
    ticker_trades = {}
    for t in history:
        tk = t["ticker"]
        if tk not in ticker_trades:
            ticker_trades[tk] = []
        ticker_trades[tk].append(t)

    for tk, trades in ticker_trades.items():
        wins = sum(1 for t in trades if t["won"])
        losses = len(trades) - wins
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        avg_held = sum(t.get("held_seconds", 0) for t in trades) / len(trades) if trades else 0
        avg_atr = sum(t.get("atr_pct", 0) for t in trades) / len(trades) if trades else 0

        # Learn optimal hold duration for this ticker
        winning_trades = [t for t in trades if t["won"]]
        avg_winning_hold = sum(t.get("held_seconds", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0

        strategy["ticker_profiles"][tk] = {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(trades), 3) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "avg_held_seconds": round(avg_held),
            "avg_winning_hold": round(avg_winning_hold),
            "avg_atr_pct": round(avg_atr, 5),
        }

    # --- Trend effectiveness ---
    for trend in ["UPTREND", "DOWNTREND", "CONSOLIDATION", "UNKNOWN"]:
        trend_trades = [t for t in history if t.get("trend") == trend]
        if trend_trades:
            wins = sum(1 for t in trend_trades if t["won"])
            avg_pnl = sum(t["pnl_pct"] for t in trend_trades) / len(trend_trades)
            strategy["trend_stats"][trend] = {
                "wins": wins,
                "losses": len(trend_trades) - wins,
                "win_rate": round(wins / len(trend_trades), 3),
                "avg_pnl": round(avg_pnl, 4),
                "count": len(trend_trades),
            }

    # --- Time-of-day performance (crypto trades 24/7) ---
    for hour in range(0, 24):
        hour_trades = [t for t in history if t.get("hour") == hour]
        if hour_trades:
            wins = sum(1 for t in hour_trades if t["won"])
            avg_pnl = sum(t["pnl_pct"] for t in hour_trades) / len(hour_trades)
            strategy["hour_stats"][str(hour)] = {
                "wins": wins,
                "losses": len(hour_trades) - wins,
                "win_rate": round(wins / len(hour_trades), 3),
                "avg_pnl": round(avg_pnl, 4),
                "count": len(hour_trades),
            }

    # --- Hold duration effectiveness ---
    for bucket, (lo, hi) in [("short_0_120", (0, 120)), ("medium_120_600", (120, 600)), ("long_600_plus", (600, 999999))]:
        bucket_trades = [t for t in history if lo <= t.get("held_seconds", 0) < hi]
        if bucket_trades:
            wins = sum(1 for t in bucket_trades if t["won"])
            avg_pnl = sum(t["pnl_pct"] for t in bucket_trades) / len(bucket_trades)
            strategy["duration_stats"][bucket] = {
                "wins": wins,
                "losses": len(bucket_trades) - wins,
                "win_rate": round(wins / len(bucket_trades), 3),
                "avg_pnl": round(avg_pnl, 4),
                "count": len(bucket_trades),
            }

    # --- Exit reason effectiveness ---
    exit_reasons = {}
    for t in history:
        reason = t.get("exit_reason", "UNKNOWN")
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(t)

    for reason, trades in exit_reasons.items():
        wins = sum(1 for t in trades if t["won"])
        avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades)
        strategy["exit_stats"][reason] = {
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": round(wins / len(trades), 3),
            "avg_pnl": round(avg_pnl, 4),
            "count": len(trades),
        }

    # --- TP/SL hit rates ---
    tp_trades = [t for t in history if "TAKE PROFIT" in t.get("exit_reason", "")]
    sl_trades = [t for t in history if "STOP LOSS" in t.get("exit_reason", "") or "EMERGENCY" in t.get("exit_reason", "")]
    smart_trades = [t for t in history if "SMART" in t.get("exit_reason", "")]

    total = len(history)
    strategy["tp_hit_rate"] = round(len(tp_trades) / total, 3) if total else 0
    strategy["sl_hit_rate"] = round(len(sl_trades) / total, 3) if total else 0
    strategy["smart_exit_rate"] = round(len(smart_trades) / total, 3) if total else 0

    # --- Compute adjustments (Step 9: faster, wider ranges) ---
    adj = strategy["adjustments"]

    # TP multiplier: if TP hits often and is profitable, it's well-placed
    # If smart exits dominate with profit, TP is too far -- tighten it
    if smart_trades and tp_trades:
        smart_avg = sum(t["pnl_pct"] for t in smart_trades) / len(smart_trades)
        tp_avg = sum(t["pnl_pct"] for t in tp_trades) / len(tp_trades)
        if len(smart_trades) > len(tp_trades) * 2 and smart_avg > 0:
            adj["tp_multiplier"] = max(0.5, adj["tp_multiplier"] - 0.10)  # tighten TP (was 0.05)
        elif len(tp_trades) > len(smart_trades) * 2:
            adj["tp_multiplier"] = min(2.0, adj["tp_multiplier"] + 0.10)  # widen TP (was 0.05)

    # SL multiplier: if SL hits often, stops are too tight
    if total > 15:
        sl_rate = len(sl_trades) / total
        if sl_rate > 0.35:
            adj["sl_multiplier"] = min(2.0, adj["sl_multiplier"] + 0.10)  # widen SL (was 0.05)
        elif sl_rate < 0.15:
            adj["sl_multiplier"] = max(0.5, adj["sl_multiplier"] - 0.10)  # tighten SL (was 0.05)

    # Ticker preferences (faster: 3 trades instead of 5)
    adj["avoid_tickers"] = [tk for tk, p in strategy["ticker_profiles"].items()
                            if p["trades"] >= 3 and p["win_rate"] < 0.35]
    adj["prefer_tickers"] = [tk for tk, p in strategy["ticker_profiles"].items()
                             if p["trades"] >= 3 and p["win_rate"] >= 0.60 and p["total_pnl"] > 0]

    # Best/worst hours
    hour_data = [(h, d) for h, d in strategy["hour_stats"].items() if d.get("count", 0) >= 5]
    adj["best_hours"] = [int(h) for h, d in hour_data if d["win_rate"] >= 0.60]
    adj["worst_hours"] = [int(h) for h, d in hour_data if d["win_rate"] < 0.40]

    # Best trends
    adj["best_trends"] = [t for t, d in strategy["trend_stats"].items()
                          if d.get("count", 0) >= 5 and d.get("win_rate", 0) >= 0.55]

    # --- Score buckets (Step 8: Entry score calibration) ---
    # Ranges match real scalp_score distribution — edge zone is 20+
    score_buckets = {
        "10_15": {"trades": 0, "wins": 0, "win_rate": 0},
        "15_20": {"trades": 0, "wins": 0, "win_rate": 0},
        "20_25": {"trades": 0, "wins": 0, "win_rate": 0},
        "25_30": {"trades": 0, "wins": 0, "win_rate": 0},
        "30_plus": {"trades": 0, "wins": 0, "win_rate": 0},
    }
    for t in history:
        score = t.get("scalp_score")
        if score is None:
            continue
        if score >= 30:
            bucket = "30_plus"
        elif score >= 25:
            bucket = "25_30"
        elif score >= 20:
            bucket = "20_25"
        elif score >= 15:
            bucket = "15_20"
        elif score >= 10:
            bucket = "10_15"
        else:
            continue
        score_buckets[bucket]["trades"] += 1
        if t.get("won"):
            score_buckets[bucket]["wins"] += 1

    for b in score_buckets.values():
        b["win_rate"] = round(b["wins"] / b["trades"], 3) if b["trades"] > 0 else 0
    strategy["score_buckets"] = score_buckets

    # --- Setup profiles (Step 7: Exit timing optimization) ---
    setup_groups = {}
    for t in history:
        patterns = t.get("pattern_names", [])
        trend = t.get("trend", "UNKNOWN")
        held = t.get("held_seconds", 0)
        if not patterns or held <= 0:
            continue
        for pname in patterns:
            key = f"{pname}|{trend}"
            if key not in setup_groups:
                setup_groups[key] = {"winning_holds": [], "losing_holds": [], "wins": 0, "total": 0}
            sg = setup_groups[key]
            sg["total"] += 1
            if t.get("won"):
                sg["wins"] += 1
                sg["winning_holds"].append(held)
            else:
                sg["losing_holds"].append(held)

    setup_profiles = {}
    for key, sg in setup_groups.items():
        if sg["total"] < 3:
            continue
        avg_win_hold = sum(sg["winning_holds"]) / len(sg["winning_holds"]) if sg["winning_holds"] else 0
        avg_lose_hold = sum(sg["losing_holds"]) / len(sg["losing_holds"]) if sg["losing_holds"] else 0
        # Optimal range: 50% to 150% of average winning hold time
        opt_lo = int(avg_win_hold * 0.5) if avg_win_hold > 0 else 60
        opt_hi = int(avg_win_hold * 1.5) if avg_win_hold > 0 else 600
        setup_profiles[key] = {
            "avg_winning_hold": round(avg_win_hold),
            "avg_losing_hold": round(avg_lose_hold),
            "optimal_hold_range": [opt_lo, opt_hi],
            "win_rate": round(sg["wins"] / sg["total"], 3),
            "trades": sg["total"],
        }
    strategy["setup_profiles"] = setup_profiles

    _save_strategy(strategy)


def get_strategy_adjustments() -> dict:
    """Get learned adjustments for the trading engine."""
    strategy = _load_strategy()
    return strategy.get("adjustments", {})


def get_learning_report() -> str:
    """Generate human-readable learning report with pattern stats and ML status."""
    strategy = _load_strategy()
    if strategy["trades_analyzed"] < 10:
        return "Not enough trades to generate report (need 15+)"

    lines = [f"=== LEARNING REPORT ({strategy['trades_analyzed']} trades analyzed) ==="]

    # Best/worst tickers
    profiles = strategy.get("ticker_profiles", {})
    if profiles:
        sorted_tickers = sorted(profiles.items(), key=lambda x: x[1].get("total_pnl", 0), reverse=True)
        best = [(t, p) for t, p in sorted_tickers[:3] if p["total_pnl"] > 0]
        worst = [(t, p) for t, p in sorted_tickers[-3:] if p["total_pnl"] < 0]
        if best:
            lines.append("Best tickers: " + ", ".join(f"{t} (+${p['total_pnl']}, {p['win_rate']*100:.0f}% WR)" for t, p in best))
        if worst:
            lines.append("Worst tickers: " + ", ".join(f"{t} (${p['total_pnl']}, {p['win_rate']*100:.0f}% WR)" for t, p in worst))

    # Trend stats
    trend_stats = strategy.get("trend_stats", {})
    for trend, data in trend_stats.items():
        if data.get("count", 0) >= 3:
            lines.append(f"{trend}: {data.get('win_rate', 0)*100:.0f}% WR over {data['count']} trades (avg {data.get('avg_pnl', 0)*100:.2f}%)")

    # TP/SL
    lines.append(f"TP hit rate: {strategy.get('tp_hit_rate', 0)*100:.0f}% | SL hit rate: {strategy.get('sl_hit_rate', 0)*100:.0f}% | Smart exits: {strategy.get('smart_exit_rate', 0)*100:.0f}%")

    adj = strategy.get("adjustments", {})
    lines.append(f"TP multiplier: {adj.get('tp_multiplier', 1.0):.2f}x | SL multiplier: {adj.get('sl_multiplier', 1.0):.2f}x")

    if adj.get("avoid_tickers"):
        lines.append(f"Avoiding: {', '.join(adj['avoid_tickers'])}")
    if adj.get("prefer_tickers"):
        lines.append(f"Preferring: {', '.join(adj['prefer_tickers'])}")

    # Pattern stats
    pattern_stats = strategy.get("pattern_stats", {})
    if pattern_stats:
        lines.append("\n--- Candlestick Pattern Performance ---")
        sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1].get("total", 0), reverse=True)
        for name, ps in sorted_patterns[:10]:
            if ps["total"] >= 3:
                lines.append(f"  {name}: {ps['win_rate']*100:.0f}% WR ({ps['wins']}/{ps['total']}) avg {ps['avg_pnl_pct']*100:.2f}%")

    # Score calibration
    score_buckets = strategy.get("score_buckets", {})
    if any(b.get("trades", 0) > 0 for b in score_buckets.values()):
        lines.append("\n--- Score Calibration ---")
        for name, b in sorted(score_buckets.items()):
            if b["trades"] > 0:
                lines.append(f"  Score {name.replace('_', '-')}: {b['win_rate']*100:.0f}% WR ({b['trades']} trades)")
        gate = get_optimal_quality_gate()
        lines.append(f"  Optimal quality gate: {gate}")

    # ML status
    try:
        from ml_engine import get_ml_status
        ml = get_ml_status()
        lines.append("\n--- ML Model Status ---")
        if ml.get("model_exists"):
            lines.append(f"  Trained on {ml.get('trades_used', 0)} trades | CV accuracy: {ml.get('cv_accuracy', 0)*100:.1f}% | AUC: {ml.get('train_auc', 0):.3f}")
            top = ml.get("top_features", [])
            if top:
                lines.append(f"  Top features: {', '.join(f[0] for f in top[:5])}")
        else:
            history = _load_history()
            ml_trades = sum(1 for t in history if t.get("rsi") is not None)
            lines.append(f"  Model not yet trained ({ml_trades}/100 ML-featured trades)")
    except Exception:
        lines.append("\n--- ML Model: not loaded ---")

    return "\n".join(lines)
