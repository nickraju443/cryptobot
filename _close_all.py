"""Attempt to close ALL non-quote-asset positions on andX.
Uses the corrected order endpoint + body schema."""
import os
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from andx_client import AndxClient

c = AndxClient()
balances_resp = c._get("/balance/Main/", signed=True)
if not balances_resp or balances_resp.get("status") != "success":
    print("Could not read balance")
    raise SystemExit(1)

balances = (balances_resp.get("data") or {}).get("balances") or {}

# Get list of tradable markets (only sell things andX actually has a market for)
tickers_resp = c._get("/ticker/")
markets = set()
if tickers_resp and tickers_resp.get("status") == "success":
    tickers = (tickers_resp.get("data") or {}).get("ticker") or {}
    markets = set(tickers.keys())  # e.g. {'BTCUSDT', 'ETHUSDT', 'ANDX1USDT', 'USDTUSD'}

print(f"andX tradable markets: {sorted(markets)}")
print()

quote = c.quote_asset  # USDT
to_close = []
untradable = []

for asset, info in balances.items():
    if asset == quote:
        continue
    try:
        avail = float(info.get("available_balance") or 0)
    except Exception:
        avail = 0
    if avail <= 0:
        continue
    market_code = f"{asset}{quote}"
    if market_code in markets:
        to_close.append((asset, avail, market_code))
    else:
        untradable.append((asset, avail, market_code))

if untradable:
    print("** UNTRADABLE balances (no andX market — can't be sold via API): **")
    for asset, qty, mc in untradable:
        print(f"  - {qty:.6g} {asset}  (would need market {mc} — does not exist on andX)")
    print()

if not to_close:
    print("Nothing to close via API.")
    if untradable:
        print("To convert the untradable balances above, you would need to either:")
        print("  1. Withdraw to a wallet/exchange that supports them (Coinbase, Kraken, etc.)")
        print("  2. Wait for andX to list those markets")
        print("  3. Use andX UI if they offer OTC / convert features in the web UI")
    raise SystemExit(0)

for asset, qty, mc in to_close:
    symbol = f"{asset}/{quote}"
    print(f"Selling {qty:.6g} {asset} as market order...")
    res = c.place_order(symbol, "sell", qty, order_type="market")
    print(f"  status={res.status}  order_id={res.order_id}")
    if res.raw:
        print(f"  response: {res.raw}")
