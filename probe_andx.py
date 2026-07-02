"""
probe_andx.py — Discovery script. Tries the most likely andX API base URLs
and endpoint paths with your key, prints what answers and what 404s.

Run:  python probe_andx.py
"""

from __future__ import annotations
import os
import json
import time
import hmac
import hashlib
from urllib.parse import urlencode

import requests

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    # Light manual loader so we don't require dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

KEY = os.environ.get("ANDX_API_KEY", "")
SECRET = os.environ.get("ANDX_API_SECRET", "").encode("utf-8")

if not KEY or not SECRET:
    print("ANDX_API_KEY / ANDX_API_SECRET not set. Aborting.")
    raise SystemExit(1)

CANDIDATES = [
    "https://api.andx.one",
    "https://api.andx.one/v1",
    "https://api.andx.one/api/v1",
    "https://platform.andx.one/api",
    "https://platform.andx.one/api/v1",
    "https://platform.andx.one/v1",
]

# Probe paths that most Binance-style exchanges expose. We'll try unsigned first
# (cheap), then signed for account-protected ones.
PROBES = [
    # path, signed?, description
    ("/ping",          False, "liveness ping"),
    ("/time",          False, "server time"),
    ("/exchangeInfo",  False, "trading rules / symbol list"),
    ("/markets",       False, "market list (alt name)"),
    ("/symbols",       False, "symbol list (alt name)"),
    ("/ticker/24hr",   False, "24h tickers"),
    ("/ticker/price",  False, "all prices"),
    ("/klines",        False, "candles (alt name)"),
    ("/account",       True,  "signed account info"),
    ("/api/v3/account", True, "signed account info v3"),
    ("/balances",      True,  "balances (alt name)"),
]


def sign(payload: str) -> str:
    return hmac.new(SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def try_url(base: str, path: str, signed: bool) -> tuple[int, str]:
    url = f"{base}{path}"
    params = {}
    headers = {"User-Agent": "CryptoBot-probe/1.0"}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        qs = urlencode(params)
        params["signature"] = sign(qs)
        # Try the two most common header conventions:
        headers["X-API-KEY"] = KEY
        headers["X-MBX-APIKEY"] = KEY  # Binance-spec
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        body = r.text[:200].replace("\n", " ")
        return r.status_code, body
    except requests.exceptions.ConnectionError as e:
        return -1, f"DNS/connect failed: {str(e)[:120]}"
    except Exception as e:
        return -2, f"error: {str(e)[:120]}"


def main():
    print(f"Probing andX with key {KEY[:6]}...{KEY[-3:]}\n")

    # First — which base URLs even resolve?
    reachable: list[str] = []
    print("=" * 70)
    print("STEP 1: which base URLs are reachable?")
    print("=" * 70)
    for base in CANDIDATES:
        code, body = try_url(base, "/ping", signed=False)
        marker = "OK " if code > 0 else "x  "
        # Special: 401/403/404 on /ping still means the host resolved
        if code in (200, 201, 401, 403, 404):
            reachable.append(base)
            marker = "OK "
        print(f"  {marker} {base:50s}  [{code}]  {body[:80]}")

    if not reachable:
        print("\nNo base URL resolved. Need the real host from andX dev team.")
        return

    print(f"\nFound reachable bases: {len(reachable)}")
    print("=" * 70)
    print("STEP 2: probe endpoints on each reachable base")
    print("=" * 70)

    for base in reachable:
        print(f"\n--- {base} ---")
        any_ok = False
        for path, signed, desc in PROBES:
            code, body = try_url(base, path, signed)
            if code in (200, 201):
                any_ok = True
                print(f"  OK  [{code}] {path:20s}  ({desc})")
                print(f"       body: {body[:140]}")
            elif code == 401:
                print(f"  AUTH[{code}] {path:20s}  ({desc}) - host has this path but key/sign wrong")
            elif code == 403:
                print(f"  DENY[{code}] {path:20s}  ({desc}) - host has this path but forbidden")
            elif code == 404:
                pass  # not interesting
            else:
                if code < 0:
                    print(f"  ERR  {path:20s}  {body[:80]}")
        if not any_ok:
            print("  (no 200 responses — endpoint paths likely don't match Binance-style)")


if __name__ == "__main__":
    main()
