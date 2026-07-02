"""
andx_credentials.py — Persistent storage for the user's andX API credentials.

The user pastes API key / secret / username / passphrase / account
into the dashboard. This module persists them to `andx_credentials.json`
(gitignored) so the bot can sign API requests on its own from then on.

Precedence at read time:
  1. andx_credentials.json (if present and has all required fields)
  2. .env / environment variables (ANDX_API_KEY etc.)

That order lets users either edit .env OR use the dashboard — both work.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = Path(__file__).parent / "andx_credentials.json"

# Fields persisted by the dashboard form.
_FIELDS = (
    "api_key", "api_secret", "username", "passphrase",
    "account_name", "account_number", "quote_asset", "base_url",
)

# The two "secret-y" fields. Never returned via the status/list endpoints
# in plaintext — only masked.
_SECRET_FIELDS = ("api_secret", "passphrase")

_lock = threading.Lock()


def _load_blob() -> dict:
    try:
        if not CREDENTIALS_FILE.exists():
            return {}
        return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"andx_credentials: load failed ({e}) — treating as empty")
        return {}


def _save_blob(blob: dict) -> None:
    CREDENTIALS_FILE.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    # Best-effort: tighten file perms on POSIX. On Windows this is a no-op.
    try:
        os.chmod(str(CREDENTIALS_FILE), 0o600)
    except Exception:
        pass


def _env_fallback(field: str) -> str:
    env_key = {
        "api_key": "ANDX_API_KEY",
        "api_secret": "ANDX_API_SECRET",
        "username": "ANDX_USERNAME",
        "passphrase": "ANDX_PASSPHRASE",
        "account_name": "ANDX_ACCOUNT",
        "account_number": "ANDX_ACCOUNT_NUMBER",
        "quote_asset": "ANDX_QUOTE_ASSET",
        "base_url": "ANDX_BASE_URL",
    }.get(field, "")
    if not env_key:
        return ""
    return (os.environ.get(env_key) or "").strip()


def get(field: str) -> str:
    """Return the configured value for ONE field. Checks the on-disk store
    first, then falls back to the corresponding environment variable.
    Returns empty string if neither has a value."""
    with _lock:
        blob = _load_blob()
    v = (blob.get(field) or "").strip()
    if v:
        return v
    return _env_fallback(field)


def all_required_present() -> bool:
    """The bot needs at least: api_key, api_secret, username, passphrase
    to sign requests. account_name defaults to "Main" elsewhere."""
    for f in ("api_key", "api_secret", "username", "passphrase"):
        if not get(f):
            return False
    return True


def status() -> dict:
    """Lightweight read for the dashboard. NEVER returns the secret/passphrase
    in plaintext — only masked previews so the user can confirm they're set."""
    with _lock:
        blob = _load_blob()
    out = {}
    for f in _FIELDS:
        v_store = (blob.get(f) or "").strip()
        v_env = _env_fallback(f)
        v = v_store or v_env
        if f in _SECRET_FIELDS:
            out[f] = {
                "configured": bool(v),
                "preview": _mask(v) if v else "",
                "source": "file" if v_store else ("env" if v_env else "none"),
            }
        else:
            out[f] = {
                "configured": bool(v),
                "value": v,
                "source": "file" if v_store else ("env" if v_env else "none"),
            }
    out["all_required_present"] = all_required_present()
    return out


def _mask(s: str) -> str:
    """Show first 2 + last 2 chars of a secret. Everything else as ***."""
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "***" + s[-2:]


def save(values: dict) -> dict:
    """Persist any subset of {api_key, api_secret, username, passphrase,
    account_name, account_number, quote_asset, base_url}.

    Empty strings are IGNORED (preserves the existing stored value). Pass an
    explicit None to clear a field.

    Returns the new status snapshot.
    """
    with _lock:
        blob = _load_blob()
        for f in _FIELDS:
            if f not in values:
                continue
            v = values[f]
            if v is None:
                # Explicit clear
                blob.pop(f, None)
                continue
            v_str = str(v).strip()
            if not v_str:
                # Empty string = "don't change"
                continue
            blob[f] = v_str
        _save_blob(blob)
    return status()


def clear() -> None:
    """Wipe the credentials file. Bot reverts to env vars (.env)."""
    with _lock:
        if CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink()


def test_connection() -> dict:
    """Sign and send a tiny request to andX's /balance/ endpoint using the
    currently-configured creds. Returns {ok: bool, error?: str, balance?: ...}.

    Used by the dashboard's "Test connection" button to confirm the user's
    credentials actually work before they engage live trading.
    """
    if not all_required_present():
        return {"ok": False, "error": "missing required fields (api_key, api_secret, username, passphrase)"}
    try:
        # Import here so this module can be loaded standalone without
        # pulling in the full client during early-startup config.
        from andx_client import AndxClient
        c = AndxClient()
        bal = c.get_balance()
        return {
            "ok": True,
            "quote_asset": bal.quote_asset,
            "free": float(bal.free),
            "total": float(bal.total),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
