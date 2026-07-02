"""One-shot: list ALL non-USDT balances on andX and sell each one to market."""
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

from andx_client import AndxClient

c = AndxClient()
data = c._get("/balance/Main/", signed=True)
if not data or data.get("status") != "success":
    print("Could not read balance — aborting")
    raise SystemExit(1)

balances = (data.get("data") or {}).get("balances") or {}
print("=== Full andX Main balances ===")
to_sell = []
for asset, info in balances.items():
    bal = float(info.get("balance") or 0)
    avail = float(info.get("available_balance") or 0)
    if bal > 0:
        print(f"  {asset}: total={bal:.8g}  available={avail:.8g}")
    if asset != c.quote_asset and avail > 0:
        to_sell.append((asset, avail))

if not to_sell:
    print()
    print("No non-USDT positions to close. You're already flat.")
    raise SystemExit(0)

print()
print(f"Would close {len(to_sell)} positions:")
for a, q in to_sell:
    print(f"  - sell {q:.8g} {a} -> USDT (market order)")
print()
# Confirm-then-execute (no prompt because the user already authorized this turn)
print("Executing close orders...")
for asset, qty in to_sell:
    symbol = f"{asset}/{c.quote_asset}"
    res = c.place_order(symbol, "sell", qty, order_type="market")
    print(f"  {symbol}: status={res.status}  filled={res.qty} @ {res.filled_price}  id={res.order_id}")
    if res.raw:
        print(f"     raw: {res.raw}")
