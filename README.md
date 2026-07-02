# CryptoBot

24/7 high-volume crypto scalper. Sibling to SRI MATA. **Crypto-only**,
**spot-only**, **separate folder, separate state**. Designed to plug into
your **andX** platform.

> **New here? Read `READ_ME_FIRST.txt` — it has the full setup steps.**
> Quick version: double-click `START_EVERYTHING.bat`, log into andX in the
> Chrome window that opens, then enter your andX API keys in the dashboard's
> credentials panel.

## Brain (same as SRI MATA)

1. **Signal layer** — [analysis.py](analysis.py) + [indicators_pro.py](indicators_pro.py):
   16 indicator categories + 23 candlestick patterns + trend structure +
   TTM Squeeze, SuperTrend, Anchored VWAP.
2. **Scalp score** — [app.py](app.py) `_scalp_score()`: 0-100 mean-reversion
   confluence score. Crypto-tuned thresholds.
3. **Sizing** — [kelly_sizing.py](kelly_sizing.py): fractional Kelly +
   risk-parity (ATR) + asset-class concentration soft-brake (Layer-1, DeFi,
   Meme, AI, etc.) + multi-factor conviction multiplier.
4. **Execution** — [portfolio.py](portfolio.py) + the active exchange client.
   Mirror loop is gone — this bot writes one portfolio, runs one loop.

## Crypto differences vs. SRI MATA

| Stock-only | Crypto-only |
|---|---|
| 9:30am–4pm ET, weekdays | **24/7** |
| Universe = 61 hand-picked tickers | **Top-N by 24h volume**, refreshed every 5 min |
| Long & short | **Spot-only / long-only** (extend `portfolio.py` for futures later) |
| `int` shares | **`float` qty** (fractional crypto) |
| Sector cap (SEMI/TECH/etc.) | **Asset-class cap** (MAJOR/L1/L2/DEFI/MEME/AI/...) |
| SL 3–6%, TP 1–8%, harvest +3.5% | **SL 1.5–4.5%**, **TP 0.8–6%**, **harvest +2.0%** |
| Alpaca data + IBKR execution | **Exchange client adapter** (andX / CCXT / Binance / Kraken) |
| VIX + SPY for regime | **BTC ATR%** for volatility + BTC trend |

## File map

```
exchange_client.py   abstract base for any exchange + 15s price cache wrapper
andx_client.py       andX REST adapter — STUB. Fill in the 7 methods.
ccxt_client.py       CCXT adapter — for testing while andX is being wired up
analysis.py          signal layer (16 indicators + candles + trend + pro)
indicators_pro.py    TTM Squeeze, SuperTrend, Anchored VWAP
screener.py          dynamic top-volume universe + parallel scan
kelly_sizing.py      Kelly + risk-parity + asset-class concentration
portfolio.py         single portfolio (cash + positions + history)
learner.py           indicator weight learning + ticker memory + strategy
ml_engine.py         XGBoost + crypto regime detection (BTC vol/trend)
ml_ensemble.py       XGBoost + LightGBM ensemble (used by ml_engine fallback)
app.py               Flask app + trading loops + scalp scoring
templates/dashboard.html   dashboard at http://localhost:5001
```

## Wiring up andX

`andx_client.py` already conforms to the `BaseExchangeClient` contract — the
rest of the bot runs end-to-end without crashing even when it's stubbed.
To go live with your platform, fill in **seven methods**:

| Method | What to wire |
|---|---|
| `_get(path, params, signed)` | andX REST GET with auth headers |
| `_post(path, body, signed)`  | andX REST POST with body signing |
| `get_top_volume_symbols(n)`  | path that returns 24h-ticker for all markets |
| `get_price(symbol)`           | last-trade price for one symbol |
| `get_prices(symbols)`         | batched price snapshot (or fall back to per-symbol) |
| `get_candles(symbol, tf)`     | OHLCV klines endpoint |
| `place_order / cancel_order / get_balance` | order endpoint + cancel + account |

Adjust `to_andx()` / `from_andx()` if your symbol format isn't `BTCUSDT`.

After wiring it up:
```
python -m andx_client
```
runs a smoke test that prints connection status, top-5 volume, BTC price, a
candle pull, and the account balance.

## Running

1. **Install deps**

   ```powershell
   cd <folder you unzipped this bot into>
   pip install -r requirements.txt
   ```
   (Or skip this — `START_EVERYTHING.bat` installs everything automatically.)

2. **Configure env** — copy `.env.example` to `.env` (or set in your shell).
   The safe default is `CRYPTO_EXCHANGE=ccxt` + `CCXT_EXCHANGE=binance` so the
   bot has real data to chew on while you wire up andX:

   ```powershell
   $env:CRYPTO_EXCHANGE = "ccxt"
   $env:CCXT_EXCHANGE   = "binance"
   $env:LIVE_TRADING    = "0"      # sim fills only
   ```

3. **Start**

   ```powershell
   .\START_BOT.bat
   # or
   python app.py
   ```

   Open <http://localhost:5001>. SRI MATA runs on 5000 — both can run side-by-side.

4. **Go live** — once you trust the sim results:
   - For andX: fill in `andx_client.py`, set `ANDX_API_KEY`/`ANDX_API_SECRET`, then `LIVE_TRADING=1`.
   - For CCXT: set `CCXT_API_KEY` + `CCXT_API_SECRET` + `CCXT_LIVE_TRADING=1`.

## Dashboard

`http://localhost:5001`
- **Scalp gate slider** — same control as SRI MATA. Lower = more trades, lower edge. 22 is the safe default.
- **Auto-TP** — arm a one-shot take-profit on cumulative batch gains.
- **Manual mode** — scan but don't trade. Watch the log to learn the bot's picks.
- **Sell all** — flatten everything now.
- **Healthcheck** — pings the exchange (auth, balance, BTC price).

## State files (all in this folder — separate from SRI MATA)

```
portfolio.json           positions, cash, closed trades
bot_stats.json           career PnL / wins / HWM
bot_stats_backup.json    PnL recovery copy
learning_data.json       indicator weights (per category)
ticker_memory.json       per-symbol win/loss/streak
trade_history.json       50-feature record per trade (used by ML)
strategy_learned.json    pattern stats, score buckets, exit profiles
ml_model.ubj             XGBoost trained model (created after 100+ trades)
ml_calibration.json      isotonic calibration curve
ml_ensemble_meta.json    ensemble metadata
```

## Notes

- **No shorts.** Spot only. Add a `short()` / `cover()` flow on top of the
  exchange client when you point this at perpetual futures.
- **One portfolio.** No SRI MATA-style dual sim+live mirror. `LIVE_TRADING=1`
  swaps the fill simulator for real orders; the same JSON state file tracks both.
- **The ML model is disabled by default** (`ML_HARD_BLOCK=0.0`), same as the
  current production SRI MATA configuration. It's still wired up for retraining —
  scalp_score is the real edge. Once you have ~100 crypto trades, the ML
  background trainer will start adding value.
- **24/7 means 24/7.** Run it under a process supervisor (NSSM on Windows,
  systemd on Linux) so it survives reboots and disconnects.
