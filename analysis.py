"""
analysis.py — Crypto-port of SRI MATA's signal layer.

16 indicator categories + 23 candlestick patterns + trend structure + 3 pro
indicators (TTM Squeeze, SuperTrend, Anchored VWAP). Same brain, same weights.

Differences vs. stock version:
  * Data source is the exchange client (BTC/USDT, ETH/USDT, ...).
  * Default `period` -> timeframe-aware bar count instead of "2y" calendar window.
  * Multi-timeframe weekly confirmation -> 1d candles (since crypto trades 24/7).
  * No premarket/afterhours overlay step; the same MTF logic (5m primary +
    higher-tf confirmation) applies to crypto.
"""

import numpy as np
import pandas as pd

from indicators_pro import compute_all_pro
from exchange_client import get_client


# Map "interval" strings to the exchange's expected timeframe + a sane bar count
_TF_MAP = {
    "1m":  ("1m",  500),
    "5m":  ("5m",  500),
    "15m": ("15m", 400),
    "30m": ("30m", 400),
    "1h":  ("1h",  400),
    "4h":  ("4h",  300),
    "1d":  ("1d",  300),
    "1wk": ("1w",  150),
}


def fetch_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV candles from the active exchange. `period` is ignored — we
    take a sensible bar count for each timeframe instead."""
    tf, limit = _TF_MAP.get(interval, ("1d", 300))
    client = get_client()
    df = client.get_candles(symbol, timeframe=tf, limit=limit)
    min_bars = 14 if interval not in ("1d", "1wk") else 20
    if df is None or df.empty or len(df) < min_bars:
        raise ValueError(f"No data for '{symbol}' ({tf})")
    return df


def fetch_intraday(symbol: str, bar_size: str = "5 mins", duration: str = "2 D") -> pd.DataFrame:
    """Live 5m candles overlay (same role as in SRI MATA)."""
    try:
        client = get_client()
        df = client.get_candles(symbol, timeframe="5m", limit=300)
        if df is not None and not df.empty and len(df) >= 10:
            return df
    except Exception as e:
        print(f"[INTRADAY] Failed for {symbol}: {e}")
    return pd.DataFrame()


def get_symbol_info(symbol: str, df: pd.DataFrame = None) -> dict:
    info = {"name": symbol}
    if df is not None and len(df) > 50:
        info["52w_high"] = float(df["High"].tail(365).max())
        info["52w_low"] = float(df["Low"].tail(365).min())
        info["avg_volume"] = float(df["Volume"].tail(20).mean())
    return info


# -- Indicator calculations (identical to SRI MATA) --------------------------

def calc_sma(s, w): return s.rolling(window=w).mean()
def calc_ema(s, span): return s.ewm(span=span, adjust=False).mean()


def calc_rsi(s, period: int = 14):
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1/period, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = ag / al
    return 100 - (100 / (1 + rs))


def calc_macd(s):
    e12 = calc_ema(s, 12); e26 = calc_ema(s, 26)
    macd = e12 - e26; sig = calc_ema(macd, 9); hist = macd - sig
    return macd, sig, hist


def calc_stochastic(df, k=14, d=3):
    lo = df["Low"].rolling(k).min(); hi = df["High"].rolling(k).max()
    K = 100 * (df["Close"] - lo) / (hi - lo)
    D = K.rolling(d).mean()
    return K, D


def calc_adx(df, period: int = 14):
    high = df["High"]; low = df["Low"]; close = df["Close"]
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low; tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period).mean()


def calc_atr(df, period: int = 14):
    high = df["High"]; low = df["Low"]; close = df["Close"]
    tr1 = high - low; tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


def calc_bollinger(s, window: int = 20, num_std: float = 2.0):
    mid = calc_sma(s, window); std = s.rolling(window).std()
    upper = mid + num_std * std; lower = mid - num_std * std
    pct_b = (s - lower) / (upper - lower)
    return upper, mid, lower, pct_b


def calc_ichimoku(df):
    high = df["High"]; low = df["Low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = df["Close"].shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def calc_vwap(df, window: int = 20):
    typ = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typ * df["Volume"]).rolling(window).sum() / df["Volume"].rolling(window).sum()


def calc_obv(df):
    return (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()


def calc_cmf(df, period: int = 20):
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng * df["Volume"]
    mfv = mfv.fillna(0)
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum()


# -- Candlestick pattern detection (23 patterns — identical) ----------------

def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"patterns": [], "net_score": 0, "bullish_count": 0, "bearish_count": 0}

    patterns = []
    c = df.tail(5)
    o, h, l, cl = c["Open"].values, c["High"].values, c["Low"].values, c["Close"].values

    def body(i): return abs(cl[i] - o[i])
    def rng(i): return h[i] - l[i] if h[i] > l[i] else 0.0001
    def upper_shadow(i): return h[i] - max(o[i], cl[i])
    def lower_shadow(i): return min(o[i], cl[i]) - l[i]
    def is_green(i): return cl[i] > o[i]
    def is_red(i): return cl[i] < o[i]
    def body_pct(i): return body(i) / rng(i) if rng(i) > 0 else 0
    def mid(i): return (o[i] + cl[i]) / 2

    trend_up = cl[-1] > cl[0] and sum(1 for i in range(1, 5) if cl[i] > cl[i-1]) >= 3
    trend_dn = cl[-1] < cl[0] and sum(1 for i in range(1, 5) if cl[i] < cl[i-1]) >= 3

    i, p, p2 = 4, 3, 2
    b = body(i); r = rng(i); ls = lower_shadow(i); us = upper_shadow(i); bp = body_pct(i)

    if bp < 0.1 and r > 0:
        if ls > 2 * b and us < b * 0.5:
            patterns.append({"name": "Dragonfly Doji", "type": "bullish", "strength": 2})
        elif us > 2 * b and ls < b * 0.5:
            patterns.append({"name": "Gravestone Doji", "type": "bearish", "strength": 2})
        else:
            patterns.append({"name": "Doji", "type": "bullish" if trend_dn else "bearish", "strength": 1})
    elif ls >= 2 * b and us <= b * 0.5 and bp < 0.4:
        if trend_dn:
            patterns.append({"name": "Hammer", "type": "bullish", "strength": 2})
        else:
            patterns.append({"name": "Hanging Man", "type": "bearish", "strength": 2})
    elif us >= 2 * b and ls <= b * 0.5 and bp < 0.4:
        if trend_dn:
            patterns.append({"name": "Inverted Hammer", "type": "bullish", "strength": 2})
        else:
            patterns.append({"name": "Shooting Star", "type": "bearish", "strength": 2})
    elif bp < 0.35 and ls > b and us > b:
        patterns.append({"name": "Spinning Top", "type": "bullish" if trend_dn else "bearish", "strength": 1})
    elif bp > 0.85 and us < r * 0.05 and ls < r * 0.05:
        patterns.append({"name": ("Bullish Marubozu" if is_green(i) else "Bearish Marubozu"),
                         "type": ("bullish" if is_green(i) else "bearish"), "strength": 3})

    if is_red(p) and is_green(i) and o[i] <= cl[p] and cl[i] >= o[p] and body(i) > body(p):
        patterns.append({"name": "Bullish Engulfing", "type": "bullish", "strength": 3})
    if is_green(p) and is_red(i) and o[i] >= cl[p] and cl[i] <= o[p] and body(i) > body(p):
        patterns.append({"name": "Bearish Engulfing", "type": "bearish", "strength": 3})
    if is_red(p) and is_green(i) and o[i] < l[p] and cl[i] > mid(p) and cl[i] < o[p]:
        patterns.append({"name": "Piercing Line", "type": "bullish", "strength": 2})
    if is_green(p) and is_red(i) and o[i] > h[p] and cl[i] < mid(p) and cl[i] > o[p]:
        patterns.append({"name": "Dark Cloud Cover", "type": "bearish", "strength": 2})
    if is_red(p) and is_green(i) and body(p) > body(i) * 1.5 and o[i] > cl[p] and cl[i] < o[p]:
        patterns.append({"name": "Bullish Harami", "type": "bullish", "strength": 2})
    if is_green(p) and is_red(i) and body(p) > body(i) * 1.5 and cl[i] > o[p] and o[i] < cl[p]:
        patterns.append({"name": "Bearish Harami", "type": "bearish", "strength": 2})
    if rng(p) > 0 and abs(l[i] - l[p]) / rng(p) < 0.05 and trend_dn:
        patterns.append({"name": "Tweezer Bottom", "type": "bullish", "strength": 2})
    if rng(p) > 0 and abs(h[i] - h[p]) / rng(p) < 0.05 and trend_up:
        patterns.append({"name": "Tweezer Top", "type": "bearish", "strength": 2})
    if is_red(p2) and body_pct(p) < 0.3 and is_green(i) and body(p2) > body(p) * 2 and body(i) > body(p) * 2 and cl[i] > mid(p2):
        patterns.append({"name": "Morning Star", "type": "bullish", "strength": 3})
    if is_green(p2) and body_pct(p) < 0.3 and is_red(i) and body(p2) > body(p) * 2 and body(i) > body(p) * 2 and cl[i] < mid(p2):
        patterns.append({"name": "Evening Star", "type": "bearish", "strength": 3})
    if is_green(p2) and is_green(p) and is_green(i):
        if cl[p] > cl[p2] and cl[i] > cl[p] and o[p] >= o[p2] and o[p] <= cl[p2] and o[i] >= o[p] and o[i] <= cl[p]:
            patterns.append({"name": "Three White Soldiers", "type": "bullish", "strength": 3})
    if is_red(p2) and is_red(p) and is_red(i):
        if cl[p] < cl[p2] and cl[i] < cl[p] and o[p] <= o[p2] and o[p] >= cl[p2] and o[i] <= o[p] and o[i] >= cl[p]:
            patterns.append({"name": "Three Black Crows", "type": "bearish", "strength": 3})
    if is_red(p2) and is_green(p) and is_green(i):
        if o[p] > cl[p2] and cl[p] < o[p2] and body(p2) > body(p) * 1.5 and cl[i] > o[p2]:
            patterns.append({"name": "Three Inside Up", "type": "bullish", "strength": 3})
    if is_green(p2) and is_red(p) and is_red(i):
        if cl[p] > o[p2] and o[p] < cl[p2] and body(p2) > body(p) * 1.5 and cl[i] < o[p2]:
            patterns.append({"name": "Three Inside Down", "type": "bearish", "strength": 3})

    try:
        from learner import get_pattern_reliability
        trend_ctx = "UPTREND" if trend_up else ("DOWNTREND" if trend_dn else "CONSOLIDATION")
        for pat in patterns:
            rel = get_pattern_reliability(pat["name"], trend_ctx)
            if rel < 0.40:
                pat["strength"] = max(1, pat["strength"] - 1)
            elif rel > 0.65:
                pat["strength"] = min(3, pat["strength"] + 1)
            pat["learned_reliability"] = rel
    except Exception:
        pass

    bull = [x for x in patterns if x["type"] == "bullish"]
    bear = [x for x in patterns if x["type"] == "bearish"]
    bull_score = sum(x["strength"] for x in bull) / 9.0
    bear_score = sum(x["strength"] for x in bear) / 9.0
    net = min(1.0, bull_score) - min(1.0, bear_score)

    return {
        "patterns": patterns,
        "net_score": round(net, 3),
        "bullish_count": len(bull),
        "bearish_count": len(bear),
    }


def detect_trend_structure(df: pd.DataFrame, lookback: int = 60) -> dict:
    if len(df) < 10:
        return {"trend": "UNKNOWN", "strength": 0, "swing_highs": [], "swing_lows": [],
                "last_swing_high": 0, "last_swing_low": 0}
    recent = df.tail(min(lookback, len(df)))
    highs = recent["High"].values; lows = recent["Low"].values
    swing_highs, swing_lows = [], []
    for i in range(2, len(recent) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(float(lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "CONSOLIDATION", "strength": 0,
                "swing_highs": swing_highs[-4:], "swing_lows": swing_lows[-4:],
                "last_swing_high": swing_highs[-1] if swing_highs else 0,
                "last_swing_low": swing_lows[-1] if swing_lows else 0}
    hh = sum(1 for i in range(1, min(4, len(swing_highs))) if swing_highs[-i] > swing_highs[-i-1])
    hl = sum(1 for i in range(1, min(4, len(swing_lows))) if swing_lows[-i] > swing_lows[-i-1])
    lh = sum(1 for i in range(1, min(4, len(swing_highs))) if swing_highs[-i] < swing_highs[-i-1])
    ll = sum(1 for i in range(1, min(4, len(swing_lows))) if swing_lows[-i] < swing_lows[-i-1])
    up_strength = min(hh + hl, 3); dn_strength = min(lh + ll, 3)
    if up_strength >= 2 and up_strength > dn_strength:
        trend, strength = "UPTREND", up_strength
    elif dn_strength >= 2 and dn_strength > up_strength:
        trend, strength = "DOWNTREND", dn_strength
    else:
        trend, strength = "CONSOLIDATION", 0
    return {"trend": trend, "strength": strength,
            "swing_highs": swing_highs[-4:], "swing_lows": swing_lows[-4:],
            "last_swing_high": swing_highs[-1] if swing_highs else 0,
            "last_swing_low": swing_lows[-1] if swing_lows else 0}


def quick_momentum_check(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {"exit_signal": False, "bearish_count": 0, "bullish_candle": False, "bearish_candle": False}
    close = df["Close"]
    rsi = calc_rsi(close); _, _, hist = calc_macd(close); ema9 = calc_ema(close, 9)
    latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50
    prev_rsi = float(rsi.iloc[-2]) if pd.notna(rsi.iloc[-2]) else 50
    latest_hist = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else 0
    prev_hist = float(hist.iloc[-2]) if pd.notna(hist.iloc[-2]) else 0
    latest_ema9 = float(ema9.iloc[-1]) if pd.notna(ema9.iloc[-1]) else 0
    price = float(close.iloc[-1])
    candles = detect_candlestick_patterns(df)
    rsi_falling = latest_rsi < prev_rsi
    macd_shrinking = abs(latest_hist) < abs(prev_hist)
    below_ema9 = price < latest_ema9 if latest_ema9 > 0 else False
    bearish = sum([
        rsi_falling and latest_rsi > 60,
        latest_hist < 0 and macd_shrinking,
        below_ema9,
        candles.get("net_score", 0) < -0.3,
    ])
    bullish = sum([
        not rsi_falling and latest_rsi < 40,
        latest_hist > 0 and not macd_shrinking,
        not below_ema9,
        candles.get("net_score", 0) > 0.3,
    ])
    return {"rsi": latest_rsi, "rsi_falling": rsi_falling,
            "macd_hist": latest_hist, "macd_hist_shrinking": macd_shrinking,
            "price_below_ema9": below_ema9,
            "bearish_candle": candles.get("net_score", 0) < -0.3,
            "bullish_candle": candles.get("net_score", 0) > 0.3,
            "bearish_count": bearish, "bullish_count": bullish,
            "exit_signal": bearish >= 2}


def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    recent = df.tail(lookback)
    highs = recent["High"]; lows = recent["Low"]; close = recent["Close"].iloc[-1]
    pivot = (recent["High"].iloc[-1] + recent["Low"].iloc[-1] + recent["Close"].iloc[-1]) / 3
    r1 = 2 * pivot - recent["Low"].iloc[-1]; s1 = 2 * pivot - recent["High"].iloc[-1]
    r2 = pivot + (recent["High"].iloc[-1] - recent["Low"].iloc[-1])
    s2 = pivot - (recent["High"].iloc[-1] - recent["Low"].iloc[-1])
    resistance_levels, support_levels = [], []
    for i in range(2, len(recent) - 2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
           highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
            if highs.iloc[i] > close:
                resistance_levels.append(highs.iloc[i])
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
           lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
            if lows.iloc[i] < close:
                support_levels.append(lows.iloc[i])
    nearest_resistance = min(resistance_levels, key=lambda x: x - close) if resistance_levels else r1
    nearest_support = max(support_levels, key=lambda x: close - x) if support_levels else s1
    return {"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2,
            "nearest_resistance": nearest_resistance, "nearest_support": nearest_support}


def calc_fibonacci_levels(df: pd.DataFrame, lookback: int = 120) -> dict:
    recent = df.tail(lookback)
    high = recent["High"].max(); low = recent["Low"].min(); diff = high - low
    return {"0.0": high, "0.236": high - 0.236 * diff, "0.382": high - 0.382 * diff,
            "0.5": high - 0.5 * diff, "0.618": high - 0.618 * diff,
            "0.786": high - 0.786 * diff, "1.0": low}


# -- Main analysis ------------------------------------------------------------

def full_analysis(symbol: str, period: str = "2y", interval: str = "1d", weight_overrides: dict = None) -> dict:
    """Same return shape as SRI MATA's full_analysis. Keys: ticker (the symbol),
    info, signal, confidence, indicators, risk_reward, etc."""
    wo = weight_overrides or {}
    df = fetch_data(symbol, period, interval)
    info = get_symbol_info(symbol, df)

    df["SMA_10"] = calc_sma(df["Close"], 10)
    df["SMA_20"] = calc_sma(df["Close"], 20)
    df["SMA_50"] = calc_sma(df["Close"], 50)
    df["SMA_100"] = calc_sma(df["Close"], 100)
    df["SMA_200"] = calc_sma(df["Close"], 200)
    df["EMA_9"] = calc_ema(df["Close"], 9)
    df["EMA_21"] = calc_ema(df["Close"], 21)
    df["RSI"] = calc_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calc_macd(df["Close"])
    df["Stoch_K"], df["Stoch_D"] = calc_stochastic(df)
    df["ADX"] = calc_adx(df)
    df["ATR"] = calc_atr(df)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"], df["BB_PctB"] = calc_bollinger(df["Close"])
    tenkan, kijun, span_a, span_b, chikou = calc_ichimoku(df)
    df["Ichimoku_Tenkan"] = tenkan; df["Ichimoku_Kijun"] = kijun
    df["Ichimoku_SpanA"] = span_a; df["Ichimoku_SpanB"] = span_b
    df["VWAP"] = calc_vwap(df); df["OBV"] = calc_obv(df); df["CMF"] = calc_cmf(df)
    df["Vol_SMA20"] = calc_sma(df["Volume"], 20)

    # Live intraday overlay (5m candles)
    _intraday = None
    try:
        idf = fetch_intraday(symbol)
        if idf is not None and not idf.empty and len(idf) >= 20:
            idf["RSI"] = calc_rsi(idf["Close"])
            idf["MACD"], idf["MACD_Signal"], idf["MACD_Hist"] = calc_macd(idf["Close"])
            idf["Stoch_K"], idf["Stoch_D"] = calc_stochastic(idf)
            idf["BB_Upper"], idf["BB_Mid"], idf["BB_Lower"], idf["BB_PctB"] = calc_bollinger(idf["Close"])
            idf["EMA_9"] = calc_ema(idf["Close"], 9); idf["EMA_21"] = calc_ema(idf["Close"], 21)
            idf["OBV"] = calc_obv(idf); idf["CMF"] = calc_cmf(idf)
            idf["VWAP"] = calc_vwap(idf); idf["ATR"] = calc_atr(idf)
            ilast = idf.iloc[-1]
            for col in ["RSI", "MACD", "MACD_Signal", "MACD_Hist", "Stoch_K", "Stoch_D",
                        "BB_Upper", "BB_Mid", "BB_Lower", "BB_PctB",
                        "EMA_9", "EMA_21", "OBV", "CMF", "VWAP", "ATR"]:
                if col in ilast.index and pd.notna(ilast[col]):
                    df.iloc[-1, df.columns.get_loc(col)] = ilast[col]
            if pd.notna(ilast["Close"]) and ilast["Close"] > 0:
                df.iloc[-1, df.columns.get_loc("Close")] = ilast["Close"]
                df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], ilast["Close"])
                df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], ilast["Close"])
            _intraday = idf
    except Exception as e:
        print(f"[ANALYSIS] Intraday overlay failed for {symbol}: {e}")
        _intraday = None

    latest = df.iloc[-1]; prev = df.iloc[-2]
    sr = find_support_resistance(df); fib = calc_fibonacci_levels(df)
    candle_data = detect_candlestick_patterns(_intraday if _intraday is not None and len(_intraday) >= 10 else df)
    trend_data = detect_trend_structure(df)

    try:
        _src = _intraday if _intraday is not None and len(_intraday) >= 25 else df
        _pro_df = _src.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                       "Close": "close", "Volume": "volume"})[
            ["open", "high", "low", "close", "volume"]]
        pro_ind = compute_all_pro(_pro_df)
    except Exception as e:
        print(f"[ANALYSIS] pro indicators failed for {symbol}: {e}")
        pro_ind = {}

    def w(category, base): return base * wo.get(category, 1.0)

    signals = []

    # 1. SMA alignment
    bull = 0
    if pd.notna(latest["SMA_20"]) and pd.notna(latest["SMA_50"]):
        if latest["SMA_20"] > latest["SMA_50"]: bull += 1
    if pd.notna(latest["SMA_50"]) and pd.notna(latest["SMA_100"]):
        if latest["SMA_50"] > latest["SMA_100"]: bull += 1
    if pd.notna(latest["SMA_100"]) and pd.notna(latest["SMA_200"]):
        if latest["SMA_100"] > latest["SMA_200"]: bull += 1
    if bull == 3:
        signals.append((w("sma", 3), 1, "All SMAs aligned bullish", "sma"))
    elif bull == 0:
        signals.append((w("sma", 3), -1, "All SMAs aligned bearish", "sma"))
    elif bull >= 2:
        signals.append((w("sma", 2), 0.5, f"SMA mostly bullish ({bull}/3)", "sma"))
    else:
        signals.append((w("sma", 2), -0.5, f"SMA mostly bearish ({bull}/3)", "sma"))

    # 2. EMA cross
    if pd.notna(latest["EMA_9"]) and pd.notna(latest["EMA_21"]):
        above_both = latest["Close"] > latest["EMA_9"] and latest["Close"] > latest["EMA_21"]
        below_both = latest["Close"] < latest["EMA_9"] and latest["Close"] < latest["EMA_21"]
        ema_up = latest["EMA_9"] > latest["EMA_21"] and prev["EMA_9"] <= prev["EMA_21"]
        ema_dn = latest["EMA_9"] < latest["EMA_21"] and prev["EMA_9"] >= prev["EMA_21"]
        if ema_up:
            signals.append((w("ema", 3), 1, "EMA 9/21 BULLISH CROSSOVER", "ema"))
        elif ema_dn:
            signals.append((w("ema", 3), -1, "EMA 9/21 BEARISH CROSSOVER", "ema"))
        elif above_both:
            signals.append((w("ema", 2), 0.7, "Price above EMA 9/21", "ema"))
        elif below_both:
            signals.append((w("ema", 2), -0.7, "Price below EMA 9/21", "ema"))

    # 3. RSI
    if pd.notna(latest["RSI"]):
        rsi = latest["RSI"]; prev_rsi = prev["RSI"]
        if rsi < 25:
            signals.append((w("rsi", 3), 1, f"RSI {rsi:.1f} - deeply oversold", "rsi"))
        elif rsi < 30:
            signals.append((w("rsi", 2.5), 0.8, f"RSI {rsi:.1f} - oversold", "rsi"))
        elif rsi > 75:
            signals.append((w("rsi", 3), -1, f"RSI {rsi:.1f} - deeply overbought", "rsi"))
        elif rsi > 70:
            signals.append((w("rsi", 2.5), -0.8, f"RSI {rsi:.1f} - overbought", "rsi"))
        elif rsi < 40 and prev_rsi < rsi:
            signals.append((w("rsi", 1.5), 0.5, f"RSI {rsi:.1f} - recovering", "rsi"))
        elif rsi > 60 and prev_rsi > rsi:
            signals.append((w("rsi", 1.5), -0.5, f"RSI {rsi:.1f} - declining", "rsi"))
        else:
            signals.append((w("rsi", 1), 0, f"RSI {rsi:.1f} - neutral", "rsi"))
        if len(df) > 20:
            price_higher = latest["Close"] > df["Close"].iloc[-15]
            rsi_lower = latest["RSI"] < df["RSI"].iloc[-15]
            price_lower = latest["Close"] < df["Close"].iloc[-15]
            rsi_higher = latest["RSI"] > df["RSI"].iloc[-15]
            if price_higher and rsi_lower:
                signals.append((w("rsi_divergence", 2.5), -0.8, "BEARISH RSI DIVERGENCE", "rsi_divergence"))
            elif price_lower and rsi_higher:
                signals.append((w("rsi_divergence", 2.5), 0.8, "BULLISH RSI DIVERGENCE", "rsi_divergence"))

    # 4. MACD
    if pd.notna(latest["MACD"]):
        macd_up = latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]
        macd_dn = latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]
        hist_inc = latest["MACD_Hist"] > prev["MACD_Hist"]
        if macd_up:
            if latest["MACD"] < 0:
                signals.append((w("macd", 3), 1, "MACD bullish crossover below zero", "macd"))
            else:
                signals.append((w("macd", 2.5), 0.8, "MACD bullish crossover above zero", "macd"))
        elif macd_dn:
            if latest["MACD"] > 0:
                signals.append((w("macd", 3), -1, "MACD bearish crossover above zero", "macd"))
            else:
                signals.append((w("macd", 2.5), -0.8, "MACD bearish crossover below zero", "macd"))
        elif latest["MACD"] > latest["MACD_Signal"] and hist_inc:
            signals.append((w("macd", 1.5), 0.5, "MACD bullish momentum", "macd"))
        elif latest["MACD"] < latest["MACD_Signal"] and not hist_inc:
            signals.append((w("macd", 1.5), -0.5, "MACD bearish momentum", "macd"))

    # 5. Stochastic
    if pd.notna(latest["Stoch_K"]):
        k, d = latest["Stoch_K"], latest["Stoch_D"]
        cross_up = k > d and prev["Stoch_K"] <= prev["Stoch_D"]
        cross_dn = k < d and prev["Stoch_K"] >= prev["Stoch_D"]
        if k < 20 and cross_up:
            signals.append((w("stochastic", 2.5), 1, f"Stoch bullish cross oversold ({k:.0f})", "stochastic"))
        elif k > 80 and cross_dn:
            signals.append((w("stochastic", 2.5), -1, f"Stoch bearish cross overbought ({k:.0f})", "stochastic"))
        elif k < 20:
            signals.append((w("stochastic", 1.5), 0.6, f"Stoch oversold ({k:.0f})", "stochastic"))
        elif k > 80:
            signals.append((w("stochastic", 1.5), -0.6, f"Stoch overbought ({k:.0f})", "stochastic"))

    # 6. ADX
    if pd.notna(latest["ADX"]):
        adx = latest["ADX"]
        if adx > 40:
            signals.append((w("adx", 2), 0, f"ADX {adx:.1f} - very strong trend", "adx"))
        elif adx > 25:
            signals.append((w("adx", 1.5), 0, f"ADX {adx:.1f} - trending", "adx"))
        else:
            signals.append((w("adx", 1), 0, f"ADX {adx:.1f} - weak", "adx"))

    # 7. Bollinger
    if pd.notna(latest["BB_PctB"]):
        pctb = latest["BB_PctB"]
        sq = (latest["BB_Upper"] - latest["BB_Lower"]) / latest["BB_Mid"] if latest["BB_Mid"] else 1
        if pctb <= 0:
            signals.append((w("bollinger", 2.5), 0.9, "Below lower BB (extreme oversold)", "bollinger"))
        elif pctb >= 1:
            signals.append((w("bollinger", 2.5), -0.9, "Above upper BB (extreme overbought)", "bollinger"))
        elif pctb < 0.2:
            signals.append((w("bollinger", 1.5), 0.5, f"Near lower BB (%B={pctb:.2f})", "bollinger"))
        elif pctb > 0.8:
            signals.append((w("bollinger", 1.5), -0.5, f"Near upper BB (%B={pctb:.2f})", "bollinger"))
        if sq < 0.04:
            signals.append((w("bollinger", 2), 0, "BB SQUEEZE", "bollinger"))

    # 8. Ichimoku
    if pd.notna(latest.get("Ichimoku_SpanA")) and pd.notna(latest.get("Ichimoku_SpanB")):
        ct = max(latest["Ichimoku_SpanA"], latest["Ichimoku_SpanB"])
        cb = min(latest["Ichimoku_SpanA"], latest["Ichimoku_SpanB"])
        pr = latest["Close"]
        if pr > ct and latest["Ichimoku_Tenkan"] > latest["Ichimoku_Kijun"]:
            signals.append((w("ichimoku", 2.5), 1, "Above cloud + TK bullish", "ichimoku"))
        elif pr > ct:
            signals.append((w("ichimoku", 2), 0.6, "Above cloud", "ichimoku"))
        elif pr < cb and latest["Ichimoku_Tenkan"] < latest["Ichimoku_Kijun"]:
            signals.append((w("ichimoku", 2.5), -1, "Below cloud + TK bearish", "ichimoku"))
        elif pr < cb:
            signals.append((w("ichimoku", 2), -0.6, "Below cloud", "ichimoku"))
        else:
            signals.append((w("ichimoku", 1), 0, "Inside cloud", "ichimoku"))

    # 9. OBV
    if len(df) > 20:
        obv_sma = calc_sma(df["OBV"], 20)
        if pd.notna(obv_sma.iloc[-1]):
            if latest["OBV"] > obv_sma.iloc[-1]:
                signals.append((w("obv", 1.5), 0.5, "OBV above 20-bar avg", "obv"))
            else:
                signals.append((w("obv", 1.5), -0.5, "OBV below 20-bar avg", "obv"))

    # 10. CMF
    if pd.notna(latest["CMF"]):
        cmf = latest["CMF"]
        if cmf > 0.1:
            signals.append((w("cmf", 1.5), 0.6, f"CMF +{cmf:.2f} - strong buying", "cmf"))
        elif cmf > 0:
            signals.append((w("cmf", 1), 0.3, f"CMF +{cmf:.2f}", "cmf"))
        elif cmf < -0.1:
            signals.append((w("cmf", 1.5), -0.6, f"CMF {cmf:.2f} - strong selling", "cmf"))
        elif cmf < 0:
            signals.append((w("cmf", 1), -0.3, f"CMF {cmf:.2f}", "cmf"))

    # 11. Relative volume
    if pd.notna(latest["Vol_SMA20"]) and latest["Vol_SMA20"] > 0:
        rvol = latest["Volume"] / latest["Vol_SMA20"]
        if rvol > 2.0:
            signals.append((w("volume", 2), 0, f"Volume {rvol:.1f}x avg", "volume"))
        elif rvol > 1.3:
            signals.append((w("volume", 1), 0, f"Volume {rvol:.1f}x", "volume"))
        elif rvol < 0.5:
            signals.append((w("volume", 1.5), 0, f"Volume {rvol:.1f}x - LOW", "volume"))

    # 12. VWAP
    if pd.notna(latest["VWAP"]):
        if latest["Close"] > latest["VWAP"] * 1.01:
            signals.append((w("vwap", 1.5), 0.4, "Above VWAP", "vwap"))
        elif latest["Close"] < latest["VWAP"] * 0.99:
            signals.append((w("vwap", 1.5), -0.4, "Below VWAP", "vwap"))

    # 13. Consecutive direction
    up = 0; dn = 0
    for i in range(1, min(8, len(df))):
        if df["Close"].iloc[-i] > df["Close"].iloc[-i-1]:
            up += 1
        else:
            break
    for i in range(1, min(8, len(df))):
        if df["Close"].iloc[-i] < df["Close"].iloc[-i-1]:
            dn += 1
        else:
            break
    if up >= 5:
        signals.append((w("price_action", 1.5), -0.5, f"{up} consecutive green bars (exhaustion)", "price_action"))
    elif dn >= 5:
        signals.append((w("price_action", 1.5), 0.5, f"{dn} consecutive red bars (bounce)", "price_action"))

    # 14. 52w range
    if info.get("52w_high") and info.get("52w_low") and info["52w_high"] > info["52w_low"]:
        range_pct = (latest["Close"] - info["52w_low"]) / (info["52w_high"] - info["52w_low"])
        if range_pct > 0.95:
            signals.append((w("range_52w", 1.5), -0.3, f"At {range_pct:.0%} of 52w range", "range_52w"))
        elif range_pct < 0.1:
            signals.append((w("range_52w", 1.5), 0.3, f"At {range_pct:.0%} of 52w range", "range_52w"))

    # 15. Candlestick
    bc = candle_data["bullish_count"]; brc = candle_data["bearish_count"]
    if bc >= 2:
        names = ", ".join(p["name"] for p in candle_data["patterns"] if p["type"] == "bullish")
        signals.append((w("candlestick", 3), 0.8, f"Bullish patterns: {names}", "candlestick"))
    elif bc == 1:
        nm = candle_data["patterns"][0]["name"] if candle_data["patterns"] else "pattern"
        signals.append((w("candlestick", 2), 0.5, f"Bullish: {nm}", "candlestick"))
    if brc >= 2:
        names = ", ".join(p["name"] for p in candle_data["patterns"] if p["type"] == "bearish")
        signals.append((w("candlestick", 3), -0.8, f"Bearish patterns: {names}", "candlestick"))
    elif brc == 1 and bc == 0:
        nm = [p for p in candle_data["patterns"] if p["type"] == "bearish"][0]["name"]
        signals.append((w("candlestick", 2), -0.5, f"Bearish: {nm}", "candlestick"))

    # 16. Trend structure
    ts = trend_data["trend"]; tstr = trend_data["strength"]
    if ts == "UPTREND" and tstr >= 2:
        signals.append((w("trend_structure", 2.5), 0.8, f"Uptrend HH/HL x{tstr}", "trend_structure"))
    elif ts == "UPTREND":
        signals.append((w("trend_structure", 2), 0.4, "Mild uptrend", "trend_structure"))
    elif ts == "DOWNTREND" and tstr >= 2:
        signals.append((w("trend_structure", 2.5), -0.8, f"Downtrend LH/LL x{tstr}", "trend_structure"))
    elif ts == "DOWNTREND":
        signals.append((w("trend_structure", 2), -0.4, "Mild downtrend", "trend_structure"))
    elif ts == "CONSOLIDATION":
        signals.append((w("trend_structure", 1), 0, "Consolidation", "trend_structure"))

    # Pro indicators
    st_dir = pro_ind.get("supertrend_direction"); st_flip = pro_ind.get("supertrend_flip")
    if st_flip and st_dir == "UP":
        signals.append((w("supertrend", 3), 1, "SuperTrend flipped UP", "supertrend"))
    elif st_flip and st_dir == "DOWN":
        signals.append((w("supertrend", 3), -1, "SuperTrend flipped DOWN", "supertrend"))
    elif st_dir == "UP":
        signals.append((w("supertrend", 2), 0.5, "SuperTrend uptrend", "supertrend"))
    elif st_dir == "DOWN":
        signals.append((w("supertrend", 2), -0.5, "SuperTrend downtrend", "supertrend"))

    sq_fired = pro_ind.get("squeeze_fired"); sq_dir = pro_ind.get("squeeze_direction"); in_sq = pro_ind.get("in_squeeze")
    if sq_fired and sq_dir == "UP":
        signals.append((w("ttm_squeeze", 3.5), 1, "TTM Squeeze FIRED UP", "ttm_squeeze"))
    elif sq_fired and sq_dir == "DOWN":
        signals.append((w("ttm_squeeze", 3.5), -1, "TTM Squeeze FIRED DOWN", "ttm_squeeze"))
    elif in_sq:
        signals.append((w("ttm_squeeze", 1), 0, "In squeeze - coiling", "ttm_squeeze"))

    avwap_pct = pro_ind.get("price_vs_avwap_pct")
    if avwap_pct is not None:
        if avwap_pct > 0.5:
            signals.append((w("anchored_vwap", 2), 0.5, f"Above AVWAP (+{avwap_pct:.2f}%)", "anchored_vwap"))
        elif avwap_pct < -0.5:
            signals.append((w("anchored_vwap", 2), -0.5, f"Below AVWAP ({avwap_pct:.2f}%)", "anchored_vwap"))

    total_weight = sum(wt for wt, s, r, c in signals)
    # NaN guard: any indicator returning NaN (e.g. RSI on empty/all-NaN bars)
    # would propagate through the sum and corrupt the confidence read. Coerce
    # any NaN/None to 0 before summing.
    def _safe(v):
        try:
            x = float(v)
            return x if x == x else 0.0  # NaN check
        except (TypeError, ValueError):
            return 0.0
    weighted_score = sum(wt * _safe(s) for wt, s, r, c in signals)
    max_possible = sum(wt * 1 for wt, s, r, c in signals)
    normalized = weighted_score / max_possible if max_possible > 0 else 0
    confidence = abs(normalized) * 100

    if normalized >= 0.5: signal = "STRONG BUY"
    elif normalized >= 0.25: signal = "BUY"
    elif normalized >= 0.1: signal = "LEAN BUY"
    elif normalized <= -0.5: signal = "STRONG SELL"
    elif normalized <= -0.25: signal = "SELL"
    elif normalized <= -0.1: signal = "LEAN SELL"
    else: signal = "HOLD"

    # Risk targets (TP/SL) from S/R + Fib + ATR
    atr = latest["ATR"] if pd.notna(latest["ATR"]) else 0
    price = latest["Close"]
    fib_above = sorted([v for v in fib.values() if isinstance(v, (int, float)) and v > price * 1.003])
    fib_below = sorted([v for v in fib.values() if isinstance(v, (int, float)) and v < price * 0.997], reverse=True)

    if "BUY" in signal:
        sl_c = [sr["nearest_support"]]
        if fib_below: sl_c.append(fib_below[0])
        if atr > 0: sl_c.append(price - 2 * atr)
        stop_loss = max(c for c in sl_c if c < price) if any(c < price for c in sl_c) else price - 2 * atr
        tp_c = [sr["nearest_resistance"]]
        if fib_above: tp_c.append(fib_above[0])
        if atr > 0: tp_c.append(price + 1.5 * atr)
        tp1 = min(c for c in tp_c if c > price) if any(c > price for c in tp_c) else price + 1.5 * atr
        tp2 = sr["nearest_resistance"]; tp3 = price + 3 * atr
    elif "SELL" in signal:
        sl_c = [sr["nearest_resistance"]]
        if fib_above: sl_c.append(fib_above[0])
        if atr > 0: sl_c.append(price + 2 * atr)
        stop_loss = min(c for c in sl_c if c > price) if any(c > price for c in sl_c) else price + 2 * atr
        tp_c = [sr["nearest_support"]]
        if fib_below: tp_c.append(fib_below[0])
        if atr > 0: tp_c.append(price - 1.5 * atr)
        tp1 = max(c for c in tp_c if c < price) if any(c < price for c in tp_c) else price - 1.5 * atr
        tp2 = sr["nearest_support"]; tp3 = price - 3 * atr
    else:
        stop_loss = price - 2 * atr
        tp1 = price + 1.5 * atr; tp2 = sr["nearest_resistance"]; tp3 = price + 3 * atr

    risk_reward = abs(tp2 - price) / abs(price - stop_loss) if abs(price - stop_loss) > 0 else 0

    # Higher-timeframe confirmation: 1d candles (crypto's "weekly" analog is just a higher TF)
    try:
        higher = fetch_data(symbol, period="1y", interval="1d")
        higher["RSI"] = calc_rsi(higher["Close"])
        higher["SMA_20"] = calc_sma(higher["Close"], 20)
        higher["SMA_50"] = calc_sma(higher["Close"], 50)
        wl = higher.iloc[-1]
        higher_trend = ("BULLISH" if wl["Close"] > wl["SMA_20"] > wl["SMA_50"] else
                        "BEARISH" if wl["Close"] < wl["SMA_20"] < wl["SMA_50"] else "MIXED")
        higher_rsi = wl["RSI"]
        mtf_aligned = (("BUY" in signal and higher_trend == "BULLISH") or
                       ("SELL" in signal and higher_trend == "BEARISH"))
        if mtf_aligned:
            confidence = min(confidence * 1.2, 95)
    except Exception:
        higher_trend = "N/A"; higher_rsi = None; mtf_aligned = None

    return {
        "ticker": symbol,
        "info": info,
        "signal": signal,
        "confidence": round(confidence, 1),
        "normalized_score": round(normalized, 3),
        "weighted_score": round(weighted_score, 2),
        "max_score": round(max_possible, 2),
        "signals": [(r, wt * s) for wt, s, r, c in signals],
        "signal_categories": {c: s for wt, s, r, c in signals if s != 0},
        "price": price,
        "atr": atr,
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(tp2, 8),
        "take_profit_3": round(tp3, 8),
        "risk_reward": round(risk_reward, 2),
        "support_resistance": sr,
        "fibonacci": fib,
        "candlestick_patterns": candle_data,
        "trend_structure": trend_data,
        "pro_indicators": pro_ind,
        "weekly_trend": higher_trend,  # name kept for ML feature parity
        "weekly_rsi": round(higher_rsi, 1) if higher_rsi and pd.notna(higher_rsi) else None,
        "mtf_aligned": mtf_aligned,
        "indicators": {
            "rsi": round(latest["RSI"], 1) if pd.notna(latest["RSI"]) else None,
            "prev_rsi": round(prev["RSI"], 1) if pd.notna(prev["RSI"]) else None,
            "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
            "macd_signal": float(latest["MACD_Signal"]) if pd.notna(latest["MACD_Signal"]) else None,
            "macd_hist": float(latest["MACD_Hist"]) if pd.notna(latest["MACD_Hist"]) else None,
            "macd_hist_prev": float(prev["MACD_Hist"]) if pd.notna(prev["MACD_Hist"]) else None,
            "stoch_k": round(latest["Stoch_K"], 1) if pd.notna(latest["Stoch_K"]) else None,
            "stoch_d": round(latest["Stoch_D"], 1) if pd.notna(latest["Stoch_D"]) else None,
            "adx": round(latest["ADX"], 1) if pd.notna(latest["ADX"]) else None,
            "atr": float(latest["ATR"]) if pd.notna(latest["ATR"]) else None,
            "bb_pctb": round(latest["BB_PctB"], 2) if pd.notna(latest["BB_PctB"]) else None,
            "cmf": round(latest["CMF"], 3) if pd.notna(latest["CMF"]) else None,
            "vwap": float(latest["VWAP"]) if pd.notna(latest["VWAP"]) else None,
            "sma_20": float(latest["SMA_20"]) if pd.notna(latest["SMA_20"]) else None,
            "sma_50": float(latest["SMA_50"]) if pd.notna(latest["SMA_50"]) else None,
            "sma_200": float(latest["SMA_200"]) if pd.notna(latest["SMA_200"]) else None,
            "ema_9": float(latest["EMA_9"]) if pd.notna(latest["EMA_9"]) else None,
            "ema_21": float(latest["EMA_21"]) if pd.notna(latest["EMA_21"]) else None,
            "volume": float(latest["Volume"]),
            "vol_avg": float(latest["Vol_SMA20"]) if pd.notna(latest["Vol_SMA20"]) else None,
            "rvol": round(latest["Volume"] / latest["Vol_SMA20"], 2) if pd.notna(latest["Vol_SMA20"]) and latest["Vol_SMA20"] > 0 else None,
        },
    }


def momentum_check(analysis_result: dict) -> dict:
    ind = analysis_result.get("indicators", {})
    checks = 0; reasons = []
    price = analysis_result.get("price", 0)
    ema9 = ind.get("ema_9")
    if ema9 and price > ema9:
        checks += 1; reasons.append("above EMA9")
    elif ema9:
        reasons.append("below EMA9")
    else:
        checks += 1
    macd_hist = ind.get("macd_hist")
    if macd_hist is not None and macd_hist > 0:
        checks += 1; reasons.append(f"MACD hist +{macd_hist:.4f}")
    elif macd_hist is not None:
        reasons.append(f"MACD hist {macd_hist:.4f}")
    else:
        checks += 1
    rsi = ind.get("rsi")
    if rsi is not None and 40 <= rsi <= 70:
        checks += 1; reasons.append(f"RSI {rsi:.0f}")
    elif rsi is not None:
        reasons.append(f"RSI {rsi:.0f} (out of range)")
    else:
        checks += 1
    candle = analysis_result.get("candlestick_patterns", {})
    if candle.get("net_score", 0) > 0.3:
        checks += 1; reasons.append("bullish candle")
    return {"pass": checks >= 2, "score": int(min(checks / 3, 1) * 100), "reason": ", ".join(reasons)}


def volume_check(analysis_result: dict) -> dict:
    rvol = analysis_result.get("indicators", {}).get("rvol")
    if rvol is None:
        return {"pass": True, "rvol": 0}
    return {"pass": rvol >= 0.5, "rvol": rvol}
