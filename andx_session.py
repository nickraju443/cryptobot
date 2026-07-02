"""
andx_session.py — Session-cookie adapter for andX's UI-only endpoints.

The documented REST API (`/api/v1/`) only exposes 4 markets — BTC, ETH, ANDX1,
USDT — and zero shorting. The website itself trades ~120 coins and supports
shorts via `/p/v1/order/instant_order/`, but that endpoint rejects API-key auth
(returns 401). The only way for the bot to reach the full universe today is to
ride the user's browser session — i.e. copy the cookies from their logged-in
browser tab and replay them.

How it works:
  1. User opens andX in browser, opens DevTools → Network → finds an instant_order
     POST, copies it as "cURL (bash)".
  2. User pastes the curl into either:
       - the dashboard's "andX Session" panel (POST /api/andx/set_session)
       - or the file andx_session.json (in the bot folder) directly
  3. Bot parses the curl, extracts cookies + the few request headers the server
     actually checks (csrftoken, x-csrftoken, referer, user-agent), and persists
     them. From then on, the bot can place orders on ANY market the website
     supports.

Caveat: session cookies expire when the user logs out, clears cookies, or after
the server's session TTL (typically 2-4 weeks). When that happens, the bot's
order attempts start returning 403; the dashboard surfaces this so the user can
paste a fresh curl and continue.
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Path to the persisted session blob (gitignored).
SESSION_FILE = Path(__file__).parent / "andx_session.json"

# Headers that ANY browser would send and the server expects. We always send
# these regardless of what the user pasted, because most pasted curls won't
# include them (DevTools' "copy as curl" includes them but Postman's doesn't).
# Captured-curl audit (2026-06-15): the bot's request differed from the
# website's in 5 headers: authorization, priority, sec-fetch-{dest,mode,site}.
# All five are now defaults so an unauthenticated 401 cannot be blamed on a
# missing browser fingerprint header.
_DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Authorization": "Bearer null",
    "Origin": "https://platform.andx.one",
    "Priority": "u=1, i",
    "Referer": "https://platform.andx.one/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}

# Endpoint paths we care about.
INSTANT_ORDER_URL = "https://platform.andx.one/p/v1/order/instant_order/"
SESSION_TIMEOUT = 8  # seconds — keep tight; this isn't the hot path

# Per-session lock — POST/GET aren't naturally thread-safe with the same
# session object plus we read/write the same cached cookie dict.
_lock = threading.Lock()


# ----------------------------------------------------------------------
# Curl parser
# ----------------------------------------------------------------------

_CURL_HEADER_RE = re.compile(r"-H\s+'([^']+)'|-H\s+\"([^\"]+)\"|--header\s+'([^']+)'")
_CURL_COOKIE_RE = re.compile(r"-b\s+'([^']+)'|-b\s+\"([^\"]+)\"|--cookie\s+'([^']+)'")
_CURL_DATA_RE   = re.compile(
    r"--data(?:-raw|-binary)?\s+\$?'([^']+)'|--data(?:-raw|-binary)?\s+\$?\"([^\"]+)\"|"
    r"-d\s+\$?'([^']+)'|-d\s+\$?\"([^\"]+)\""
)
_CURL_URL_RE    = re.compile(r"curl\s+(?:--location\s+)?(?:-X\s+\w+\s+)?['\"]?(https?://[^'\"\s]+)")


def _unescape_cmd(s: str) -> str:
    """Strip Windows cmd.exe quoting from a 'Copy as cURL (cmd)' paste.

    cmd escapes by prefixing ^ to literal special chars: ^" → ", ^& → &, etc.
    Inside --data-raw, JSON quotes appear as ^\^" which after our cmd-strip
    becomes \" — that's bash-level JSON-string escaping. The caller's body
    extractor handles that final unescape."""
    # Order matters: collapse line continuations BEFORE the ^ stripping so
    # `^\r\n` and `^\n` don't get half-stripped.
    s = s.replace("^\r\n", " ").replace("^\n", " ")
    # Now strip remaining cmd carets. `^^` → `^` last so we don't accidentally
    # re-escape pairs introduced by other replacements.
    s = (s.replace('^"', '"')
          .replace("^&", "&")
          .replace("^%", "%")
          .replace("^|", "|")
          .replace("^<", "<")
          .replace("^>", ">")
          .replace("^^", "^"))
    return s


def parse_curl(curl_text: str) -> dict:
    """Parse a copy-as-curl string into {url, headers, cookies, body}.
    Tolerates bash line continuations, Windows ^ continuations, and mixed
    quoting. Returns empty dict on failure."""
    if not curl_text or "curl" not in curl_text:
        return {}

    s = curl_text

    # Detect Windows cmd format (presence of ^" or ^& outside the URL). If so,
    # pre-process to bash-equivalent form so the rest of this parser works
    # without two branches.
    if '^"' in s or "^&" in s or "^^" in s:
        s = _unescape_cmd(s)

    # Flatten remaining line continuations from both shells.
    s = s.replace("\\\n", " ").replace("\r\n", "\n").replace("\n", " ")

    out: dict = {"url": None, "headers": {}, "cookies": {}, "body": None}

    # URL
    m = _CURL_URL_RE.search(s)
    if m:
        out["url"] = m.group(1).rstrip("/")

    # Headers
    for m in _CURL_HEADER_RE.finditer(s):
        raw = next((g for g in m.groups() if g), "")
        if ":" in raw:
            k, v = raw.split(":", 1)
            k = k.strip()
            v = v.strip()
            # Cookie header — split into individual cookies
            if k.lower() == "cookie":
                for piece in v.split(";"):
                    piece = piece.strip()
                    if "=" in piece:
                        ck, cv = piece.split("=", 1)
                        out["cookies"][ck.strip()] = cv.strip()
            else:
                out["headers"][k] = v

    # Cookies via -b
    for m in _CURL_COOKIE_RE.finditer(s):
        raw = next((g for g in m.groups() if g), "")
        for piece in raw.split(";"):
            piece = piece.strip()
            if "=" in piece:
                ck, cv = piece.split("=", 1)
                out["cookies"][ck.strip()] = cv.strip()

    # Body
    m = _CURL_DATA_RE.search(s)
    if m:
        body = next((g for g in m.groups() if g), "")
        # If the body still carries backslash-escaped quotes (Windows cmd
        # paste leaves `\"` after our cmd-unescape), bash-unescape them too.
        if '\\"' in body and '"' not in body.replace('\\"', ''):
            body = body.replace('\\"', '"')
        out["body"] = body

    return out


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def _load_blob() -> dict:
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"andx_session: load failed ({e}) — treating as empty")
        return {}


def _save_blob(blob: dict) -> None:
    SESSION_FILE.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def save_from_curl(curl_text: str) -> dict:
    """Parse a curl string and persist the extracted session. Returns a
    summary the dashboard can show: how many cookies, whether csrftoken was
    found, last-saved timestamp."""
    parsed = parse_curl(curl_text)
    cookies = parsed.get("cookies") or {}
    headers = parsed.get("headers") or {}
    body = parsed.get("body") or ""

    # Try to extract account_number from the pasted body (the website always
    # includes it). Falls back to env / 266 default if not present.
    account_number = None
    try:
        body_json = json.loads(body) if body else {}
        if isinstance(body_json, dict) and "account_number" in body_json:
            account_number = int(body_json["account_number"])
    except Exception:
        pass

    blob = {
        "cookies": cookies,
        # Keep ONLY long-lived browser identity headers. We deliberately DROP
        # access-sign and access-timestamp: those are per-request HMAC values
        # tied to the moment the user captured the curl; replaying them past
        # their timestamp triggers 401. The bot recomputes them dynamically
        # per request when ANDX_API_KEY/SECRET/PASSPHRASE are configured (or
        # tries cookies-only when they aren't).
        "headers": {
            k: v for k, v in headers.items()
            if k.lower() in {
                "x-csrftoken", "x-csrf-token",
                "user-agent", "origin", "referer",
                "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
                "device-type",
            }
        },
        "account_number": account_number,
        "saved_at": int(time.time()),
    }
    with _lock:
        _save_blob(blob)
    return {
        "cookies_count": len(cookies),
        "has_csrftoken": "csrftoken" in cookies,
        "has_sessionid": any(k.lower() in ("sessionid", "session_id") for k in cookies),
        "account_number": account_number,
        "saved_at": blob["saved_at"],
    }


def session_status() -> dict:
    """Lightweight status the dashboard can poll."""
    blob = _load_blob()
    if not blob:
        return {"loaded": False}
    cookies = blob.get("cookies") or {}
    age_s = int(time.time()) - int(blob.get("saved_at") or 0)
    return {
        "loaded": True,
        "cookies_count": len(cookies),
        "has_csrftoken": "csrftoken" in cookies,
        "has_sessionid": any(k.lower() in ("sessionid", "session_id") for k in cookies),
        "account_number": blob.get("account_number"),
        "saved_at": blob.get("saved_at"),
        "age_seconds": age_s,
    }


def clear_session() -> None:
    with _lock:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()


# ----------------------------------------------------------------------
# Request helpers
# ----------------------------------------------------------------------

def _build_session() -> Optional[requests.Session]:
    """Build a requests.Session with the persisted cookies + headers loaded."""
    blob = _load_blob()
    if not blob:
        return None
    cookies = blob.get("cookies") or {}
    if not cookies:
        return None
    s = requests.Session()
    s.headers.update(_DEFAULT_HEADERS)
    # Per-blob extra headers (csrftoken, etc.) override defaults — but DROP
    # per-request auth headers if an older save included them. They're
    # recomputed fresh per request in place_instant_order().
    _STALE_PER_REQUEST = {"access-sign", "access-timestamp", "access-user",
                          "content-length", "authorization"}
    for k, v in (blob.get("headers") or {}).items():
        if k.lower() in _STALE_PER_REQUEST:
            continue
        s.headers[k] = v
    # Most Django apps require x-csrftoken to mirror the csrftoken cookie
    if "csrftoken" in cookies and "x-csrftoken" not in {h.lower() for h in s.headers}:
        s.headers["X-CSRFToken"] = cookies["csrftoken"]
    for k, v in cookies.items():
        s.cookies.set(k, v, domain=".andx.one")
    return s


def is_session_available() -> bool:
    """Quick check: can we make session-authenticated requests?"""
    blob = _load_blob()
    return bool(blob and blob.get("cookies"))


def get_account_number() -> int:
    blob = _load_blob()
    n = blob.get("account_number") if blob else None
    if n is not None:
        return int(n)
    return int(os.environ.get("ANDX_ACCOUNT_NUMBER", "266"))


# ----------------------------------------------------------------------
# Instant order — the actual reason this module exists
# ----------------------------------------------------------------------

def place_instant_order(
    buy_currency: str,
    sell_currency: str,
    buy_amount: float,
    sell_amount: float,
    visible_price: float,
) -> dict:
    """POST to /p/v1/order/instant_order/ using the persisted session.

    Currency-swap semantics:
      - To open a LONG on XLM (paid for in USDT):
          buy_currency='XLM', sell_currency='USDT'
      - To open a SHORT on XLM (sell-to-open):
          buy_currency='USDT', sell_currency='XLM'
      - To close a LONG on XLM:
          buy_currency='USDT', sell_currency='XLM'
      - To close a SHORT on XLM:
          buy_currency='XLM', sell_currency='USDT'

    Returns:
      {
        "ok": True/False,
        "http_status": int,
        "json": dict or None,        # parsed response body
        "text": str (when not JSON),
        "error": str (when ok=False),
        "sent_body": dict,
      }
    """
    s = _build_session()
    if s is None:
        return {"ok": False, "error": "no andX session loaded — paste cookies via /api/andx/set_session"}

    body = {
        "buy_currency_code": buy_currency.upper(),
        "sell_currency_code": sell_currency.upper(),
        "visible_price": str(visible_price),
        "buy_amount": str(buy_amount),
        "sell_amount": str(sell_amount),
        "account_number": get_account_number(),
        "depth_order": True,
        "with_bonus": False,
        "with_stake": False,
    }
    # Optional access-sign computation. The website's signing scheme is
    # different from the documented /api/v1/ HMAC, so by default we send
    # cookies-only (csrftoken + AWSALBTG carry the auth). Some servers will
    # accept that; if not, ANDX_SIGN_SCHEME lets us try alternatives:
    #   default ("none")     — no access-* headers, cookies only
    #   "docs"               — HMAC(secret, key+user+pass+ts+body)
    #   "email_body"         — HMAC(secret, email+ts+body) — guess for website
    #   "body_only"          — HMAC(secret, ts+body)
    import hashlib, hmac
    # Default scheme is "none" — empirically (2026-06-15) sending HMAC-signed
    # requests with our api_secret returns 401, meaning the website uses a
    # different signing secret than the documented API. Cookies-only ("none")
    # is the only locally-derivable option that has a chance of succeeding;
    # if the server validates access-sign as MANDATORY (not present-or-valid),
    # we'll still get 401 and need to switch to browser automation.
    scheme = (os.environ.get("ANDX_SIGN_SCHEME") or "none").lower()
    body_str = json.dumps(body, separators=(",", ":"))
    ts = str(int(time.time() * 1000))  # milliseconds, per the captured curl
    if scheme != "none":
        api_key = os.environ.get("ANDX_API_KEY", "")
        api_secret = os.environ.get("ANDX_API_SECRET", "")
        username = os.environ.get("ANDX_USERNAME", "")
        email = os.environ.get("ANDX_LOGIN_EMAIL", username)
        passphrase = os.environ.get("ANDX_PASSPHRASE", "")
        if scheme == "docs":
            msg = (api_key + username + passphrase + ts + body_str).encode()
            sign_user = username
        elif scheme == "email_body":
            msg = (email + ts + body_str).encode()
            sign_user = email
        elif scheme == "body_only":
            msg = (ts + body_str).encode()
            sign_user = email
        else:
            msg = None
        if msg and api_secret:
            sig = hmac.new(api_secret.encode(), msg, hashlib.sha256).hexdigest().upper()
            s.headers["access-user"] = sign_user
            s.headers["access-timestamp"] = ts
            s.headers["access-sign"] = sig
    try:
        # IMPORTANT: send as raw body (not json=) so the signature matches
        # exactly the bytes the server receives.
        s.headers["content-type"] = "application/json"
        r = s.post(INSTANT_ORDER_URL, data=body_str, timeout=SESSION_TIMEOUT)
        out: dict = {"http_status": r.status_code, "sent_body": body}
        try:
            out["json"] = r.json()
        except Exception:
            out["text"] = r.text[:600]
        # Capture diagnostic response headers — on 401, the server may indicate
        # the expected auth scheme via WWW-Authenticate, or surface upstream
        # proxy details via x-amzn-*, server, etc. These are read-only signals
        # we can use to triage the auth failure without re-attempting.
        _diag_headers = {
            k: r.headers[k] for k in (
                "www-authenticate", "server", "x-amzn-errortype",
                "x-amzn-requestid", "x-amzn-trace-id", "x-amz-cf-id",
                "set-cookie", "vary", "x-frame-options",
            ) if k in r.headers
        }
        if _diag_headers:
            out["resp_headers"] = _diag_headers
        # andX uses status in body — successful orders have status="success".
        body_status = (out.get("json") or {}).get("status")
        out["ok"] = r.status_code == 200 and body_status == "success"
        if not out["ok"]:
            j = out.get("json") or {}
            out["error"] = (j.get("reason") or j.get("error")
                            or out.get("text") or f"HTTP {r.status_code}")
        return out
    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "sent_body": body}


def session_self_test() -> dict:
    """Probe the session by hitting a cheap /p/v1/ endpoint. Returns whether
    cookies are still valid. Used by the dashboard's "test session" button."""
    s = _build_session()
    if s is None:
        return {"ok": False, "error": "no session loaded"}
    # GET the user-balance endpoint as a non-destructive probe. If cookies
    # are dead the server returns 403; valid cookies return 200 with our
    # account JSON. Adjust path if andX uses a different one.
    url = "https://platform.andx.one/p/v1/account/balance/"
    try:
        r = s.get(url, timeout=SESSION_TIMEOUT)
        return {
            "ok": r.status_code == 200,
            "http_status": r.status_code,
            "preview": (r.text or "")[:200],
        }
    except requests.RequestException as e:
        return {"ok": False, "error": str(e)}
