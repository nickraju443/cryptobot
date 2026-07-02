"""
kelly_sizing.py — Crypto port of SRI MATA's position sizing.

Same brain:
  * Kelly criterion on calibrated win probability (fractional-Kelly for safety)
  * Risk-parity volatility adjustment (ATR-based)
  * Concentration soft-brake (sizes DOWN, never blocks)
  * Multi-factor conviction multiplier

The only material change vs. the stock version is: SECTOR -> ASSET-CLASS.
Crypto doesn't have sectors. Instead, we bucket coins by category (Layer-1,
Memes, DeFi, AI, etc.) so the bot doesn't over-concentrate in correlated coins.

Pure functions, never raises — failures become 0 qty + reason string.
All percentages are decimals internally (0.30 = 30%).
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Asset-class map (replaces _SECTOR_MAP). Add coins as the universe grows.
# Unknown bases default to "ALT".
# ----------------------------------------------------------------------

ASSET_CLASS_MAP: dict[str, str] = {
    # Majors (highly correlated — treat as one bucket)
    "BTC": "MAJOR", "ETH": "MAJOR",

    # Layer-1s
    "SOL": "L1", "AVAX": "L1", "ADA": "L1", "DOT": "L1", "ATOM": "L1",
    "NEAR": "L1", "APT": "L1", "SUI": "L1", "SEI": "L1", "INJ": "L1",
    "TIA": "L1", "TON": "L1", "ICP": "L1", "ALGO": "L1", "HBAR": "L1",
    "TRX": "L1",

    # Layer-2s / scaling
    "MATIC": "L2", "ARB": "L2", "OP": "L2", "STRK": "L2", "MNT": "L2",
    "METIS": "L2", "MANTA": "L2",

    # DeFi
    "UNI": "DEFI", "AAVE": "DEFI", "MKR": "DEFI", "SNX": "DEFI", "CRV": "DEFI",
    "COMP": "DEFI", "LDO": "DEFI", "RUNE": "DEFI", "GMX": "DEFI", "DYDX": "DEFI",
    "PENDLE": "DEFI", "JTO": "DEFI",

    # AI
    "FET": "AI", "RNDR": "AI", "TAO": "AI", "AGIX": "AI", "WLD": "AI",
    "AKT": "AI", "OCEAN": "AI",

    # Memes
    "DOGE": "MEME", "SHIB": "MEME", "PEPE": "MEME", "WIF": "MEME", "BONK": "MEME",
    "FLOKI": "MEME", "BRETT": "MEME", "MEW": "MEME", "POPCAT": "MEME",

    # Exchange tokens
    "BNB": "EXCH", "OKB": "EXCH", "CRO": "EXCH", "KCS": "EXCH",

    # Storage / DePIN
    "FIL": "DEPIN", "AR": "DEPIN", "HNT": "DEPIN", "IOTA": "DEPIN",
    "STORJ": "DEPIN",

    # Gaming / metaverse
    "AXS": "GAME", "SAND": "GAME", "MANA": "GAME", "GALA": "GAME",
    "IMX": "GAME", "PIXEL": "GAME", "BEAM": "GAME",

    # Oracles / infra
    "LINK": "ORACLE", "PYTH": "ORACLE", "API3": "ORACLE",

    # Privacy
    "XMR": "PRIVACY", "ZEC": "PRIVACY",

    # Legacy / store-of-value
    "LTC": "LEGACY", "BCH": "LEGACY", "XRP": "LEGACY", "ETC": "LEGACY",
}


def asset_class(symbol: str) -> str:
    """Map BTC/USDT -> 'MAJOR' etc. Unknown bases -> 'ALT'."""
    base = symbol.split("/", 1)[0].upper() if "/" in symbol else symbol.upper()
    return ASSET_CLASS_MAP.get(base, "ALT")


# ----------------------------------------------------------------------
# Kelly fraction
# ----------------------------------------------------------------------

def kelly_fraction(
    win_prob: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    fraction: float = 0.5,
) -> float:
    """f* = (p * b - q) / b, scaled by `fraction`, clamped to [0, 0.5]."""
    try:
        p = float(win_prob); w = float(avg_win_pct); l = float(avg_loss_pct)
        if not (0.0 <= p <= 1.0):
            return 0.0
        if w <= 0.0 or l <= 0.0:
            return 0.0
        q = 1.0 - p
        b = w / l
        # Guard: avoid inf/NaN when l is microscopic (tiny avg loss).
        # b > 100 means we'd be deploying based on a 100:1 reward:risk read
        # from history — not statistically meaningful, default to no-bet.
        if not (b > 0.0) or b > 100.0:
            return 0.0
        f_star = (p * b - q) / b
        f_scaled = f_star * float(fraction)
        if f_scaled <= 0.0:
            return 0.0
        if f_scaled > 0.5:
            return 0.5
        return f_scaled
    except Exception as e:
        logger.warning(f"kelly_fraction failed: {e}")
        return 0.0


def risk_parity_adjustment(atr_pct: float, target_risk_pct: float = 1.0) -> float:
    """Vol-normalize size. High vol -> smaller. Clamped to [0.3, 2.0]."""
    try:
        a = float(atr_pct); t = float(target_risk_pct)
        if a <= 0.0 or t <= 0.0:
            return 1.0
        mult = t / a
        return max(0.3, min(2.0, mult))
    except Exception as e:
        logger.warning(f"risk_parity_adjustment failed: {e}")
        return 1.0


def concentration_penalty(
    symbol: str,
    current_positions: dict,
    asset_class_map: dict = None,
) -> float:
    """Soft-brake for asset-class concentration (replaces sector cap).

    Counts existing positions in the same asset class. Same scale as SRI MATA:
        0 in class  -> 1.00
        1 in class  -> 0.85
        2 in class  -> 0.60
        3 in class  -> 0.40
        4+          -> 0.25
    """
    try:
        if not isinstance(current_positions, dict):
            return 1.0
        amap = asset_class_map if asset_class_map is not None else ASSET_CLASS_MAP
        my_class = asset_class(symbol) if amap is ASSET_CLASS_MAP else amap.get(symbol, "ALT")
        count = 0
        for sym, pos in current_positions.items():
            if sym == symbol:
                continue
            qty = 0.0
            if isinstance(pos, dict):
                qty = float(pos.get("qty", 0) or pos.get("shares", 0) or 0)
            if qty <= 0:
                continue
            sym_class = asset_class(sym) if amap is ASSET_CLASS_MAP else amap.get(sym, "ALT")
            if sym_class == my_class:
                count += 1
        if count <= 0: return 1.0
        if count == 1: return 0.85
        if count == 2: return 0.60
        if count == 3: return 0.40
        return 0.25
    except Exception as e:
        logger.warning(f"concentration_penalty failed: {e}")
        return 1.0


def conviction_multiplier(
    ml_prob: float,
    technical_score: float,
    regime_match: bool = True,
    pattern_confluence: int = 0,
) -> float:
    """Multi-factor agreement -> bigger size.
        Elite:   ml>=0.70 AND tech>=30 AND regime AND patterns>=2 -> 2.00x
        Strong:  ml>=0.65 AND tech>=27 AND regime                 -> 1.60x
        Base:    ml>=0.58 AND tech>=22                            -> 1.00x
        Weak:                                                     -> 0.70x
    """
    try:
        p = float(ml_prob); t = float(technical_score)
        rm = bool(regime_match); pc = int(pattern_confluence or 0)
        if p >= 0.70 and t >= 30 and rm and pc >= 2: return 2.00
        if p >= 0.65 and t >= 27 and rm: return 1.60
        if p >= 0.58 and t >= 22: return 1.00
        return 0.70
    except Exception as e:
        logger.warning(f"conviction_multiplier failed: {e}")
        return 0.85


def calculate_win_loss_stats(trade_history: list, min_trades: int = 30) -> dict:
    """Average win/loss size and win-rate from last 100 closed trades."""
    fallback = {"avg_win_pct": 2.5, "avg_loss_pct": 1.5, "win_rate": 0.55, "trades_analyzed": 0}
    try:
        if not isinstance(trade_history, list) or len(trade_history) == 0:
            return fallback
        recent = trade_history[-100:]
        wins, losses = [], []
        for t in recent:
            if not isinstance(t, dict):
                continue
            pct = None
            try:
                pnl = t.get("pnl"); entry = t.get("entry_price")
                qty = t.get("qty") if t.get("qty") is not None else t.get("shares")
                if pnl is not None and entry is not None and qty is not None:
                    pnl = float(pnl); entry = float(entry); qty = float(qty)
                    notional = abs(entry * qty)
                    if notional > 0:
                        pct = (pnl / notional) * 100.0
            except Exception:
                pct = None
            if pct is None:
                for key in ("pnl_pct", "pct_pnl", "return_pct", "roi_pct"):
                    v = t.get(key)
                    if v is None: continue
                    try:
                        v = float(v)
                        pct = v * 100.0 if abs(v) < 0.10 else v
                        break
                    except Exception:
                        continue
            if pct is None or abs(pct) > 50.0:
                continue
            if pct > 0: wins.append(pct)
            elif pct < 0: losses.append(abs(pct))
        analyzed = len(wins) + len(losses)
        if analyzed < int(min_trades):
            fb = dict(fallback); fb["trades_analyzed"] = analyzed
            return fb
        avg_win = (sum(wins) / len(wins)) if wins else 2.5
        avg_loss = (sum(losses) / len(losses)) if losses else 1.5
        win_rate = len(wins) / analyzed if analyzed > 0 else 0.55
        if avg_win <= 0: avg_win = 2.5
        if avg_loss <= 0: avg_loss = 1.5
        if not (0.0 <= win_rate <= 1.0): win_rate = 0.55
        return {"avg_win_pct": float(avg_win), "avg_loss_pct": float(avg_loss),
                "win_rate": float(win_rate), "trades_analyzed": int(analyzed)}
    except Exception as e:
        logger.warning(f"calculate_win_loss_stats failed: {e}")
        return fallback


def kelly_position_size(
    portfolio_value: float,
    cash: float,
    ml_probability: float,
    technical_score: float,
    symbol: str,
    price: float,
    atr_pct: float,
    current_positions: dict,
    asset_class_map: Optional[dict] = None,
    trade_stats: Optional[dict] = None,
    max_single_position_pct: float = 0.50,
    regime_match: bool = True,
    pattern_confluence: int = 0,
    min_order_value: float = 10.0,
) -> dict:
    """Orchestrate Kelly + risk-parity + class-concentration + conviction.

    Returns:
        {qty, dollar_size, portfolio_pct, kelly_raw, kelly_fractional,
         risk_parity_mult, concentration_mult, conviction_mult, reasons}

    `qty` is a FLOAT (crypto is fractional). `min_order_value` is the
    exchange's minimum (default 10 USDT; many exchanges require ~5-10).
    """
    reasons: list = []
    out = {
        "qty": 0.0, "dollar_size": 0.0, "portfolio_pct": 0.0,
        "kelly_raw": 0.0, "kelly_fractional": 0.0,
        "risk_parity_mult": 1.0, "concentration_mult": 1.0, "conviction_mult": 1.0,
        "reasons": reasons,
    }
    try:
        try:
            pv = float(portfolio_value); cs = float(cash); px = float(price)
            mlp = float(ml_probability); tsc = float(technical_score)
            atr = float(atr_pct); msp = float(max_single_position_pct)
            mov = float(min_order_value)
        except Exception:
            reasons.append("error: non-numeric input"); return out

        if pv <= 0 or px <= 0:
            reasons.append(f"zero size: portfolio_value={pv} price={px}")
            return out

        stats = trade_stats if isinstance(trade_stats, dict) else {}
        avg_win = float(stats.get("avg_win_pct", 2.5))
        avg_loss = float(stats.get("avg_loss_pct", 1.5))
        reasons.append(f"stats: p={mlp:.3f} aw={avg_win:.2f}% al={avg_loss:.2f}%")

        kr = kelly_fraction(mlp, avg_win, avg_loss, fraction=1.0)
        out["kelly_raw"] = float(kr); reasons.append(f"kelly_raw={kr:.4f}")
        if kr <= 0.0:
            reasons.append("no edge: Kelly<=0"); return out

        kf = kelly_fraction(mlp, avg_win, avg_loss, fraction=0.5)
        out["kelly_fractional"] = float(kf); reasons.append(f"kelly_frac(0.5x)={kf:.4f}")

        rp = risk_parity_adjustment(atr, target_risk_pct=1.0)
        out["risk_parity_mult"] = float(rp); reasons.append(f"risk_parity={rp:.3f} (atr={atr:.2f}%)")

        cp = concentration_penalty(symbol, current_positions or {}, asset_class_map)
        out["concentration_mult"] = float(cp)
        cls = asset_class(symbol) if asset_class_map is None else asset_class_map.get(symbol, "ALT")
        reasons.append(f"concentration={cp:.3f} (class={cls})")

        cv = conviction_multiplier(mlp, tsc, regime_match, pattern_confluence)
        out["conviction_mult"] = float(cv)
        reasons.append(f"conviction={cv:.3f} (ml={mlp:.2f} tech={tsc:.1f} regime={regime_match} patterns={pattern_confluence})")

        final_pct = kf * rp * cp * cv
        if final_pct < 0.0: final_pct = 0.0
        if msp <= 0.0: msp = 0.50
        if final_pct > msp:
            reasons.append(f"capped at max_single={msp:.2f}"); final_pct = msp
        else:
            reasons.append(f"combined_pct={final_pct:.4f}")

        dollar_from_pv = pv * final_pct
        cash_cap = max(0.0, cs) * 0.95
        dollar = min(dollar_from_pv, cash_cap)
        if cash_cap < dollar_from_pv:
            reasons.append(f"cash-limited: pv_budget=${dollar_from_pv:,.2f} cash_cap=${cash_cap:,.2f}")
        if dollar < mov:
            reasons.append(f"below min_order_value ${mov:.2f}")
            out["portfolio_pct"] = float(final_pct)
            return out

        qty = dollar / px  # FLOAT — crypto is fractional
        if qty * px < mov:
            reasons.append(f"qty*price below min_order_value ${mov:.2f}")
            return out

        out["qty"] = float(qty)
        out["dollar_size"] = float(qty * px)
        out["portfolio_pct"] = float((qty * px) / pv) if pv > 0 else 0.0
        reasons.append(f"final: {qty:.8g} {symbol.split('/')[0]} @ ${px:.6g} = ${qty*px:,.2f} ({out['portfolio_pct']*100:.2f}% pv)")
        return out
    except Exception as e:
        logger.exception("kelly_position_size failed")
        reasons.append(f"error: {e}")
        return out
