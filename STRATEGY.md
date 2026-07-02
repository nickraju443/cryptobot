# CryptoBot — Trading Strategy

A 24/7 systematic crypto scalper I built around a 4-layer decision brain,
per-coin tuned parameters, and dip-aware entries. Long-only on live (andX),
full long/short on the parallel sim.

---

## Philosophy

This is a **systematic scalper**, not discretionary. Every decision —
when to enter, how much to size, when to exit — comes out of a deterministic
pipeline:

1. Scan the universe every 1–8s (depends on the active risk mode).
2. Score each candidate with 24 indicators + an ML ensemble.
3. Filter through the active mode's gate and the per-coin tier.
4. Size via Kelly + risk-parity, hard-capped by the mode's max position %.
5. Execute through andX (BTC/ETH/ANDX1 via the documented HMAC API) and
   in parallel through sim (full ~30-coin universe on Alpaca data).
6. Manage every open position every 1s — SL / TP / trail / harvest /
   fade / dump / emergency.

The whole system is **fee-aware** and **per-coin tuned**. No one-size-fits-all
parameters. BTC harvest is +1.6%; PEPE harvest is +2.7%. Same formula,
different ATR.

---

## The 4-Layer Brain

### Layer 1 — Signal (analysis.py + indicators_pro.py)
16 classic technical indicators plus 3 pro indicators. Each indicator outputs
a directional score; the weighted sum is the **scalp_score** (0–100).

### Layer 2 — ML brain (ml_engine.py + ml_ensemble.py)
XGBoost + LightGBM ensemble, calibrated to honest win-probability via
isotonic regression. Trained on 2000+ historical trades. Walk-forward
cross-validation, AUC ~0.81. Feeds Kelly sizing.

### Layer 3 — Sizing (kelly_sizing.py)
Kelly criterion on the calibrated probability, multiplied by risk-parity
(inverse-ATR), then by a sector-concentration penalty and a conviction
multiplier. Always respects `max_single_position_pct` (mode hard cap —
e.g. 10% for AGGRESSIVE).

### Layer 4 — Execution (app.py)
Layers a market-regime + goal `size_boost` on top of Kelly. Falls back to
the confidence-ladder if Kelly returns no-edge. Routes to docs API for
BTC/ETH/ANDX1; everything else runs sim-only.

---

## The 24 Indicators

### Trend
- **SMA** — 10 / 20 / 50 / 100 / 200 period alignment
- **EMA** — 9 / 21 crossover (bull/bear cross detection)
- **Ichimoku Cloud** — Tenkan, Kijun, Senkou A/B, price-vs-cloud
- **SuperTrend** — ATR-based trailing line, flip detection
- **Trend structure** — higher-highs / higher-lows over recent swings

### Momentum
- **RSI** — overbought/oversold + bullish/bearish divergence
- **MACD** — line, signal, histogram, crossover, momentum acceleration
- **Stochastic** — %K and %D, overbought/oversold + crosses
- **ADX** — trend strength (weak < 20 < trending < 40 < strong)
- **TTM Squeeze** — BB-inside-Keltner detection + squeeze momentum direction

### Volume / Money Flow
- **OBV** — On-Balance Volume vs 20-bar average
- **CMF** — Chaikin Money Flow (buy vs sell pressure)
- **VWAP** — price above/below
- **Anchored VWAP** — anchored from the last swing low, % distance
- **Relative volume (rvol)** — current bar volume / 20-bar avg. Hard gate at
  rvol ≥ 0.5 (skip thin tape).

### Volatility / Range
- **ATR** — Average True Range, drives SL/TP sizing
- **Bollinger Bands** — %B (position within bands), squeeze detection
- **52-week range** — % distance from high/low
- **BB Squeeze** — low-volatility coiling state

### Structure / Levels
- **Support / Resistance** — pivot, R1/R2, S1/S2
- **Fibonacci** — 0 / 0.236 / 0.382 / 0.5 / 0.618 / 0.786 / 1.0 retracements
- **Swing highs / lows** — last 4 of each

### Candlestick Patterns (24 detected)
**Bullish:** Hammer, Inverted Hammer, Bullish Engulfing, Morning Star, Three
White Soldiers, Bullish Marubozu, Piercing, Bullish Harami, Tweezers Bottom,
Three Inside Up, Bullish Belt Hold, Dragonfly Doji

**Bearish:** Hanging Man, Shooting Star, Bearish Engulfing, Evening Star,
Three Black Crows, Bearish Marubozu, Dark Cloud, Bearish Harami, Tweezers
Top, Three Inside Down, Bearish Belt Hold, Gravestone Doji

### Multi-Timeframe
The top 3 candidates from the 5m scan get re-analyzed on 15m candles. Final
score = 0.6 × 5m + 0.4 × 15m.

---

## Risk Modes

The master knob. Every entry/exit threshold reads from the active mode.
Switching is atomic — flip the dropdown and the next tick uses the new mode.

| Param | CONSERVATIVE | REGULAR | AGGRESSIVE |
|---|---|---|---|
| Scan interval | 8s | 5s | 1s |
| Min confidence | 70 | 60 | 40 |
| Scalp gate | 35 (high) | 22 | 10 (loose) |
| Max positions | 3 | 6 | 20 |
| Max size / position | 20% | 20% | 10% (many small) |
| Entries per scan cycle | 1 | 2 | 4–6 |
| Min hold | 90s | 60s | 10s |
| Harvest target | +1.5% | +2.0% | +1.2% |
| Re-entry cooldown | 5 min | 3 min | 20s |
| Emergency stop | -4% | -6% | -8% |
| Use case | only A-grade setups | balanced default | constant trading across many coins |

---

## DIP Detector (the long-only edge)

When DIP MODE is on, the bot only trades qualifying dips. A dip earns
points across four pillars (total 100):

### Pillar 1 — Oversold momentum (max 30 pts)
- RSI ≤ 28 (deeply oversold): 18 pts
- RSI ≤ 35 (oversold): 12 pts
- Fast RSI drop (8+ pts in a few bars): 8 pts
- Stochastic K ≤ 20: 8 pts
- BB %B ≤ 0.1 (at lower band): 6 pts

### Pillar 2 — At support (max 25 pts)
Distance to EMA21 / SMA20 / S/R / S1 pivot / VWAP:
- Right at support (≤ 0.3% above): 14 pts
- Very near (≤ 0.8%): 10 pts
- Near (≤ 1.5%): 6 pts
- Hard skip: price has already broken below the level

Plus: CMF ≥ 0.05 (real money buying the dip): +5 pts

### Pillar 3 — Reversal candle (max 20 pts)
- High-value reversals (Hammer, Bullish Engulfing, Morning Star, Piercing,
  Dragonfly Doji, Three White Soldiers): 12 pts
- Medium reversals (Tweezers Bottom, Bullish Harami, Three Inside Up): 7 pts
- Bullish RSI divergence: +8 pts

### Pillar 4 — Higher timeframe still constructive (max 15 pts)
- Weekly trend BULLISH: 12 pts
- Weekly trend MIXED: 6 pts
- Hard penalty: Weekly BEARISH → −15 pts (falling-knife guard)
- Weekly RSI in 40–65 range: +3 pts
- Above EMA9 (bounce starting): +4 pts

### Hard guards
- Price below daily SMA200 AND weekly bearish: skip outright (bear-market dip).

### Threshold
Each coin has its own `dip_threshold` (55 for blue chips, 60 for majors, 70
for memes). A dip qualifies when score ≥ threshold. Qualified dips get a
**+40 scalp_score boost** so they outrank generic trend signals.

---

## Per-Coin Strategy Formula

Every coin's SL / TP / harvest is derived from three inputs:

```
INPUTS
  rt_fee             = round_trip_fee(coin)   # 0.5% BTC/ETH, 0% ANDX1
  atr_pct            = 75th percentile of (hi-lo)/close over 168h
  audit_max_loss     = worst historical -% for this coin (from trade history)

FORMULA
  edge_floor         = max(0.3%, 1.5 × rt_fee)
  sl_min             = max(0.3%, 0.5 × atr_pct)
  sl_max             = min(1.5 × atr_pct, audit_max_loss × 0.8, 6%)
  tp_min             = max(edge_floor, 0.8 × atr_pct)
  tp_max             = min(3.0 × atr_pct, 10%)
  harvest_threshold  = atr_pct + rt_fee   # net = 1 ATR after fees
  min_hold_seconds   = 3 (zero-fee) | 5 (calm) | 15 (medium) | 25 (meme)
  dip_threshold      = 55 (atr<1.5%) | 60 (atr<3%) | 70 (meme)
  scalp_gate         = 10 (zero-fee) | 14 (paying fees) +4 if win_rate<45%
```

### Live output (auto-tuned on real ATR)

| Coin | ATR | Audit | SL | TP | Harvest |
|---|---|---|---|---|---|
| BTC | 1.08% | -16.4% | 0.5–1.6% | 0.9–3.2% | +1.58% |
| ETH | 1.46% | -19.8% | 0.7–2.2% | 1.2–4.4% | +1.96% |
| SOL | 1.72% | — | 0.9–2.6% | 1.4–5.2% | +2.22% |
| ADA | 2.42% | — | 1.2–3.6% | 1.9–7.3% | +2.92% |
| SUSHI | 2.57% | — | 1.3–3.9% | 2.1–7.7% | +3.07% |
| PEPE | 2.15% | — | 1.1–3.2% | 1.7–6.4% | +2.65% |
| ANDX1 | 1.00% | — | 0.5–1.5% | 0.8–3.0% | +1.00% |

Re-tuning is one click — the bot pulls fresh ATR + audit each time.

---

## Position Sizing (Kelly + clamps)

Every entry runs through:

```
kelly_qty       = kelly_position_size(
                    ml_probability,         # calibrated win-prob from ensemble
                    technical_score,        # scalp_score
                    atr_pct,                # for risk-parity
                    sector_concentration,   # penalty if too much in one class
                    max_single_position_pct # mode hard cap
                  )
kelly_qty      *= mode.size_boost × goal_size_scale
deployed_usd    = kelly_qty × price
```

### Goal-based throttling
- Daily target hit → size × 0.5, gate +25
- Near target (80%) → size × 0.8, gate +10
- Drawdown 30%+ from session peak → "guard mode" (A-grade signals only)
- Drawdown 40%+ → hard pause until reset

---

## Exit Management (9 mechanisms, checked every 1s)

For every open position, in this order:

1. **Emergency stop** — gain ≤ -emergency_stop_pct (mode-specific). Bypasses min-hold.
2. **Min hold** — skip exit checks until per-coin min_hold_seconds elapses.
3. **Stop loss** — gain ≤ -SL_pct. Direction-aware (long below entry, short above).
4. **Take profit** — gain ≥ TP_pct. Direction-aware.
5. **Harvest** — gain ≥ harvest_threshold (per-coin). Locks small wins fast.
6. **Trail lock** — peak gain ≥ trail_lock_pct AND drop ≥ 0.5% from peak.
7. **Trail breakeven** — peak gain ≥ trail_be_pct AND gain back to 0.
8. **Fade protect** — peak ≥ fade_threshold AND gain dropped fade_drop %
   from peak (e.g. peaked at +3%, now at +1.8%).
9. **Dump bleed** — gain ≤ dump_bleed_pct (slow leak cut).
10. **Smart stop reassess** (sim only) — if SL hit but indicators still
    bullish, brief chance to recover.

---

## Fee Awareness

```
fee_per_side(BTC/ETH) = 0.25%     # andX taker
fee_per_side(ANDX1)   = 0.00%
round_trip(coin)      = 2 × fee_per_side(coin)
```

On a $170 BTC trade: $0.85 round-trip fee. The harvest threshold's
`atr_pct + rt_fee` design means the harvest target already includes the
fee budget — the bot doesn't need explicit fee-subtraction logic if the
formula's gross targets are trusted.

ANDX1 is the **frequency engine**. Zero fees mean tiny moves (+0.3%)
are profitable, so it runs the tightest hold (3s) and gate (10).

---

## Learning System (closes the loop)

After every trade, three records update:

- **`ticker_memory.json`** — per-coin win rate + average gain/loss. Future
  Kelly sizing on the same coin scales 0.4× to 1.1× by this reputation.
- **`learning_data.json`** — per-indicator-category accuracy. Each
  indicator's weight is multiplied by 0.85×–1.15× based on its hit rate
  over the last 50+ trades. Gentle (no flip-flop), only kicks in after 50+
  trades.
- **`trade_history.json`** — full feature vector + outcome. Retrains the
  XGBoost+LightGBM ensemble every 10 trades.

---

## Architectural Choices

- **5m candles** — best signal:noise ratio for scalping. 1m has too much
  noise, 15m too slow for active recycling.
- **Kelly over fixed %** — high-confidence trades get bigger, low-confidence
  trades get smaller. Hard-capped to avoid Kelly's classic "go all-in on +EV"
  failure mode.
- **ML on top of TA** — the 24 indicators have known false-positive patterns
  (e.g. RSI divergence in trending markets). The ensemble learns which
  combinations have predicted wins historically.
- **Long-only live** — andX's documented API supports spot longs only on
  BTC/ETH/ANDX1. Sim runs long+short across ~30 Alpaca coins for comparison.
- **Per-coin strategies** — BTC's 0.5% range gets stopped out instantly by
  SOL's 1.7% noise. One set of SL/TP across all coins either wastes
  opportunities on calm coins or churns on volatile ones.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Main engine, Flask server, trading loops |
| `analysis.py` | 16 classic indicators + scoring |
| `indicators_pro.py` | TTM Squeeze, SuperTrend, Anchored VWAP |
| `dip_detector.py` | 4-pillar dip scoring |
| `coin_strategies.py` | Per-coin tiers + auto-tuner |
| `kelly_sizing.py` | Kelly + risk-parity + sector penalty |
| `ml_engine.py` / `ml_ensemble.py` | ML brain |
| `learner.py` | Adaptive weights + ticker memory |
| `andx_client.py` | andX exchange adapter (HMAC) |
| `portfolio_live.py` | Live portfolio mirror |
| `portfolio_sim.py` | Paper portfolio |
| `hybrid_client.py` | Alpaca data + andX execution |
| `screener.py` | Universe selection |
