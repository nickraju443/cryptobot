"""
dip_detector.py — Quick-scalp dip detection for long-only crypto.

A "dip" here = price has dropped recently but reversal evidence is showing AND
the asset is still in an overall uptrend (so we're catching a pullback, not
a falling knife). Long-only constraint means we ONLY look for buy-the-dip
setups — no shorts.

A dip earns points across four pillars:
  1. Oversold momentum (RSI low, dropped fast, stochastic oversold)
  2. At support (price within X% of EMA20 / BB lower / S1 pivot / swing low)
  3. Reversal candle (hammer, bullish engulfing, doji at support, RSI div)
  4. Higher-timeframe trend still constructive (weekly above its EMA / not bear)

The dip_score is 0–100. A setup needs ≥60 to qualify as a tradable dip;
the scan loop boosts qualifying setups so they get prioritized over generic
trend signals.

Public API:
    score, reasons, dip_tp_pct, dip_sl_pct = score_dip(analysis_result)
"""

from __future__ import annotations
from typing import Tuple


def _safe(d: dict, *path, default=None):
    """Walk d[path[0]][path[1]]... returning default if any link missing."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def score_dip(analysis: dict) -> Tuple[int, list, float, float]:
    """Score this analysis result as a dip-buy opportunity.

    Returns (dip_score 0-100, reason list, tp_pct, sl_pct).
    tp_pct / sl_pct are dip-tuned: tight (this is a scalp). The caller can
    still clamp them via the active mode's sl_pct_min/max etc.
    """
    score = 0
    reasons: list[str] = []

    price = float(analysis.get("price") or 0)
    if price <= 0:
        return 0, ["no price"], 0.008, 0.012

    inds = analysis.get("indicators") or {}
    rsi = float(inds.get("rsi") or 50)
    prev_rsi = float(inds.get("prev_rsi") or rsi)
    stoch_k = float(inds.get("stoch_k") or 50)
    bb_pctb = float(inds.get("bb_pctb") or 0.5)
    ema9 = float(inds.get("ema_9") or 0)
    ema21 = float(inds.get("ema_21") or 0)
    sma20 = float(inds.get("sma_20") or 0)
    vwap = float(inds.get("vwap") or 0)
    cmf = float(inds.get("cmf") or 0)
    atr = float(inds.get("atr") or 0)

    weekly_trend = (analysis.get("weekly_trend") or "").upper()
    weekly_rsi = float(analysis.get("weekly_rsi") or 50)

    sr = analysis.get("support_resistance") or {}
    nearest_support = float(sr.get("nearest_support") or 0)
    s1 = float(sr.get("s1") or 0)

    candles = (analysis.get("candlestick_patterns") or {}).get("patterns") or []
    bullish_names = {p.get("name") for p in candles if p.get("type") == "bullish"}

    # ====== Pillar 1: Oversold momentum (max 30 points) ======
    if rsi <= 28:
        score += 18; reasons.append(f"RSI deeply oversold ({rsi:.0f})")
    elif rsi <= 35:
        score += 12; reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi <= 42 and prev_rsi - rsi >= 8:
        score += 8; reasons.append(f"RSI dropped {prev_rsi-rsi:.0f}pts fast")

    if stoch_k <= 20:
        score += 8; reasons.append(f"Stoch oversold ({stoch_k:.0f})")

    if bb_pctb <= 0.1:
        score += 6; reasons.append(f"At lower BB (%B={bb_pctb:.2f})")
    elif bb_pctb <= 0.2:
        score += 3; reasons.append(f"Near lower BB (%B={bb_pctb:.2f})")

    # ====== Pillar 2: At a support level (max 25 points) ======
    # Distance to each support — closer = better
    supports = []
    if ema21 > 0: supports.append(("EMA21", ema21))
    if sma20 > 0: supports.append(("SMA20", sma20))
    if nearest_support > 0: supports.append(("S/R", nearest_support))
    if s1 > 0: supports.append(("Pivot S1", s1))
    if vwap > 0: supports.append(("VWAP", vwap))

    best_support_pts = 0
    best_support_name = None
    for name, level in supports:
        if level <= 0:
            continue
        dist_pct = abs(price - level) / price
        # Price needs to be NEAR support and AT OR ABOVE it (we want to buy
        # the bounce). Below support = breakdown = no dip.
        if price < level * 0.995:
            continue  # already broken support
        if dist_pct <= 0.003:
            pts = 14; tag = "right at"
        elif dist_pct <= 0.008:
            pts = 10; tag = "very near"
        elif dist_pct <= 0.015:
            pts = 6;  tag = "near"
        else:
            continue
        if pts > best_support_pts:
            best_support_pts = pts
            best_support_name = f"{tag} {name} (+{dist_pct*100:.2f}%)"
    if best_support_pts:
        score += best_support_pts
        reasons.append(f"At support: {best_support_name}")

    # CMF turning positive = real money buying the dip
    if cmf >= 0.05:
        score += 5; reasons.append(f"CMF buying (+{cmf:.2f})")
    elif cmf >= 0:
        score += 2

    # ====== Pillar 3: Reversal candle / divergence (max 20 points) ======
    high_value_reversals = {"Hammer", "Inverted Hammer", "Bullish Engulfing",
                            "Morning Star", "Piercing", "Dragonfly Doji",
                            "Three White Soldiers"}
    medium_reversals = {"Tweezers Bottom", "Bullish Harami", "Three Inside Up",
                        "Bullish Belt Hold"}
    for n in bullish_names:
        if n in high_value_reversals:
            score += 12; reasons.append(f"Reversal candle: {n}"); break
        elif n in medium_reversals:
            score += 7; reasons.append(f"Reversal candle: {n}"); break

    # Bullish RSI divergence — price made a lower low, RSI made a higher low.
    # The analysis already flags this in its signals list.
    signals_list = analysis.get("signals") or []
    for sig in signals_list:
        if isinstance(sig, (list, tuple)) and len(sig) >= 1:
            label = str(sig[0]).upper()
            if "BULLISH RSI DIVERGENCE" in label:
                score += 8; reasons.append("RSI bull divergence"); break

    # ====== Pillar 4: Higher-timeframe still constructive (max 15 points) ======
    # We're catching pullbacks IN AN UPTREND, not buying capitulation.
    # weekly_trend = BULLISH or MIXED is OK; BEARISH means it's a falling knife.
    if weekly_trend == "BULLISH":
        score += 12; reasons.append("Weekly trend bullish")
    elif weekly_trend == "MIXED":
        score += 6; reasons.append("Weekly trend mixed (ok)")
    else:
        # BEARISH or unknown — heavy penalty, this is likely a falling knife
        score -= 15; reasons.append("⚠ Weekly bearish — risky dip")

    if 40 <= weekly_rsi <= 65:
        score += 3  # weekly RSI in healthy range

    # Also: above EMA9 short-term (so the bounce is starting)
    if ema9 > 0 and price > ema9 * 1.001:
        score += 4; reasons.append("Above EMA9 (bounce starting)")

    # ====== Hard guards: skip outright if these flags are bad ======
    # Price below daily SMA200 AND weekly bearish = bear-market dip, skip
    sma200 = float(inds.get("sma_200") or 0)
    if weekly_trend == "BEARISH" and sma200 > 0 and price < sma200 * 0.97:
        return 0, ["bear-market dip, skip"], 0.008, 0.012

    # ====== Compute dip-tuned TP/SL based on ATR ======
    # SL: just below the support we identified (or 1× ATR if no clear support)
    # TP: tight scalp at +1.2-1.8% (quick mean-reversion to EMA9/VWAP/EMA21)
    if atr > 0 and price > 0:
        atr_pct = atr / price
        sl_pct = min(0.018, max(0.008, atr_pct * 1.2))  # 0.8% to 1.8%
        tp_pct = min(0.025, max(0.012, atr_pct * 1.8))  # 1.2% to 2.5%
    else:
        sl_pct, tp_pct = 0.012, 0.018

    # Clamp + normalize
    score = max(0, min(100, score))
    return score, reasons, tp_pct, sl_pct


def is_dip_tradeable(analysis: dict, threshold: int = 60) -> bool:
    """Quick boolean: is this a tradeable dip at the given threshold?"""
    score, _, _, _ = score_dip(analysis)
    return score >= threshold
