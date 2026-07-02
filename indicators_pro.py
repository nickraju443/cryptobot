"""
indicators_pro.py — Three institutional-grade indicators.

Deliberately minimal. Three indicators, nothing else. No bloat.

1. TTM Squeeze — Bollinger inside Keltner. Detects coiling volatility
   before explosive moves. When the squeeze fires, direction is given
   by a linear-regression momentum reading.

2. SuperTrend — ATR-based trend line with clean flips. Better than ADX
   for trend confirmation because it has fewer whipsaws.

3. Anchored VWAP — VWAP anchored to session open. This is what institutional
   desks use as the "fair value" reference price intraday.

Pure numpy/pandas. Zero side effects on input DataFrames.
Functions are safe against short DataFrames — return Nones rather than crash.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ------------------------------------------------------------
# 1. SuperTrend
# ------------------------------------------------------------

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    """ATR-channel trend indicator. Returns current line value, direction, and
    whether the trend just flipped on this bar."""
    if df is None or len(df) < period + 2:
        return {"supertrend": None, "supertrend_direction": None, "supertrend_flip": False}

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.to_numpy(copy=True)
    lower = lower_basic.to_numpy(copy=True)
    c = close.to_numpy(copy=True)

    # Final upper/lower bands
    for i in range(1, len(c)):
        upper[i] = upper_basic.iloc[i] if (upper_basic.iloc[i] < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lower_basic.iloc[i] if (lower_basic.iloc[i] > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]

    # Direction: +1 uptrend, -1 downtrend
    direction = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        if c[i] > upper[i - 1]:
            direction[i] = 1
        elif c[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    line = np.where(direction == 1, lower, upper)
    flip = bool(direction[-1] != direction[-2]) if len(direction) >= 2 else False

    return {
        "supertrend": float(line[-1]),
        "supertrend_direction": "UP" if direction[-1] == 1 else "DOWN",
        "supertrend_flip": flip,
    }


# ------------------------------------------------------------
# 2. TTM Squeeze
# ------------------------------------------------------------

def ttm_squeeze(df: pd.DataFrame, length: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5) -> dict:
    """Bollinger-inside-Keltner squeeze. Fires when Bollinger bands expand
    back outside Keltner — signals volatility release. Momentum sign gives
    the direction of the pending move."""
    if df is None or len(df) < length + 2:
        return {"in_squeeze": None, "squeeze_fired": False, "squeeze_momentum": None, "squeeze_direction": None}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Bollinger
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    bb_upper = mid + bb_mult * std
    bb_lower = mid - bb_mult * std

    # Keltner (EMA mid + ATR bands)
    ema_mid = close.ewm(span=length, adjust=False).mean()
    atr = _atr(high, low, close, length)
    kc_upper = ema_mid + kc_mult * atr
    kc_lower = ema_mid - kc_mult * atr

    in_squeeze_series = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    if len(in_squeeze_series) < 2:
        return {"in_squeeze": None, "squeeze_fired": False, "squeeze_momentum": None, "squeeze_direction": None}

    in_sq_now = bool(in_squeeze_series.iloc[-1])
    in_sq_prev = bool(in_squeeze_series.iloc[-2])
    fired = (not in_sq_now) and in_sq_prev  # was squeezing, just released

    # Momentum via linear regression of (close - donchian_mid) over `length` bars
    donchian_mid = (high.rolling(length).max() + low.rolling(length).min()) / 2.0
    delta = (close - (donchian_mid + ema_mid) / 2.0)
    recent = delta.iloc[-length:].dropna().to_numpy()
    if len(recent) >= 3:
        x = np.arange(len(recent), dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        momentum = float(slope * (len(recent) - 1) + intercept)
    else:
        momentum = 0.0

    direction = "UP" if momentum > 0 else ("DOWN" if momentum < 0 else "NEUTRAL")

    return {
        "in_squeeze": in_sq_now,
        "squeeze_fired": fired,
        "squeeze_momentum": momentum,
        "squeeze_direction": direction,
    }


# ------------------------------------------------------------
# 3. Anchored VWAP (from session open)
# ------------------------------------------------------------

def anchored_vwap(df: pd.DataFrame) -> dict:
    """Session-anchored VWAP. Anchors at the first bar of the DataFrame
    (caller slices to today's session). Returns current AVWAP and
    distance of last close from it in percent."""
    if df is None or len(df) < 2 or "volume" not in df.columns:
        return {"avwap": None, "price_vs_avwap_pct": None, "above_avwap": None}

    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float)

    cum_pv = (typical * vol).cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    avwap_series = cum_pv / cum_v

    last_price = float(df["close"].iloc[-1])
    last_avwap = avwap_series.iloc[-1]
    if pd.isna(last_avwap) or last_avwap == 0:
        return {"avwap": None, "price_vs_avwap_pct": None, "above_avwap": None}

    pct = (last_price - last_avwap) / last_avwap * 100.0

    return {
        "avwap": float(last_avwap),
        "price_vs_avwap_pct": float(pct),
        "above_avwap": bool(last_price > last_avwap),
    }


# ------------------------------------------------------------
# Master
# ------------------------------------------------------------

def compute_all_pro(df: pd.DataFrame) -> dict:
    """Compute all three pro indicators. Per-indicator failures are swallowed
    (logged at WARNING) so a single bad indicator can never crash the caller."""
    out: dict = {}
    for fn in (supertrend, ttm_squeeze, anchored_vwap):
        try:
            out.update(fn(df))
        except Exception as e:
            logger.warning(f"indicators_pro.{fn.__name__} failed: {e}")
    return out


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(7)
    n = 120
    base = 100.0
    returns = np.random.normal(0.0005, 0.01, n)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(100_000, 2_000_000, n).astype(float)

    test_df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    logging.basicConfig(level=logging.INFO)
    result = compute_all_pro(test_df)
    print("indicators_pro self-test:")
    for k, v in result.items():
        print(f"  {k} = {v}")
    print(f"keys returned: {len(result)}")
