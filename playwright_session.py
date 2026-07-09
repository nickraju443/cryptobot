"""
playwright_session.py — Persistent Chromium driver for andX execution.

Why this exists
---------------
andX's `/p/v1/order/instant_order/` rejects every API-key-signed request
with HTTP 401. The website signs each request with an `access-sign` HMAC
whose secret is derived from the user's login session (it's bundled in the
site's JS / Vuex store and not exposed via the public API). Prior attempts
to brute-force the scheme across 6,000+ HMAC variants produced zero
matches.

Workaround: keep a real, logged-in Chromium session alive in the background
and route every live order through it. The site's own JS computes
access-sign for free because the request originates from inside the page
context.

Threading model
---------------
Playwright's sync API is **not** thread-safe — every Playwright object
must be touched from the thread that created it. So we run a single
dedicated worker thread that owns `sync_playwright()`, the
BrowserContext, and the Page. Public methods enqueue a `_Req` envelope
onto a queue and block on a `threading.Event` until the worker finishes
the work and signals back. This keeps the bot's other threads
(scanner, exit loop) completely independent of the browser lifecycle.

Order routing strategy
----------------------
For each order we try, in order:

  1. **fetch route** — `page.evaluate("fetch('/p/v1/order/instant_order/', …)")`
     from inside the loaded /instant/trade page. If the site patches
     `window.fetch` globally with the access-sign interceptor, this
     just works. Cheapest path: ~150-300 ms per order. If the server
     replies 401 (interceptor not patching bare fetch), we mark the
     fetch route dead for the rest of the process lifetime and fall
     through.

  2. **UI route** — navigate to /instant/trade/USDT_<base>, fill the
     amount input, click BUY/SELL, wait for the result. ~2-3 s per
     order, robust to JS internals.

Persistence
-----------
A persistent `user-data-dir` (PROFILE_DIR) is used so cookies,
localStorage, IndexedDB, and any device-fingerprint state survive bot
restarts. First-time setup launches Chromium HEADED — the user logs in
manually in the visible window. From then on the bot can run headless
and reuse the saved session indefinitely.

Public API (called from the bot's threads)
------------------------------------------
    get_session()                  -> PlaywrightSession (lazy singleton)
    PlaywrightSession.start(headless: bool = False)
    PlaywrightSession.is_logged_in() -> bool
    PlaywrightSession.place_order(buy_currency, sell_currency,
                                  buy_amount, sell_amount, visible_price)
                                  -> PWOrderResult
    PlaywrightSession.get_balance() -> PWBalance
    PlaywrightSession.status()      -> PWStatus
    PlaywrightSession.stop()
    PlaywrightSession.wipe_profile()
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

PROFILE_DIR = Path(__file__).parent / "_pw_andx_profile"
LOGIN_URL = "https://platform.andx.one/login"
HOME_URL = "https://platform.andx.one/"
INSTANT_TRADE_URL_TMPL = "https://platform.andx.one/instant/trade/USDT_{base}"
INSTANT_ORDER_PATH = "/p/v1/order/instant_order/"
BALANCE_PATH = "/p/v1/account/balance/"

DEFAULT_NAV_TIMEOUT_MS = 20_000
DEFAULT_ACTION_TIMEOUT_MS = 8_000
REQUEST_QUEUE_TIMEOUT_S = 45.0  # UI-route order placement (nav → form fill →
                                # confirm modal → server confirmation) takes
                                # 15-30s in practice. 18s was failing every
                                # order; 45s gives them room to actually
                                # land. With AGGRESSIVE this means slower
                                # cadence per coin but real fills instead of
                                # all-timeouts.
WORKER_START_TIMEOUT_S = 30.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/148.0.0.0 Safari/537.36"
)

# Bot quote-asset assumption — every andX market we trade is *_USDT for
# instant_order purposes. Aligns with andx_session.place_instant_order.
QUOTE_ASSET = "USDT"


# ----------------------------------------------------------------------
# Enums + dataclasses
# ----------------------------------------------------------------------

class SessionState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    LOGGED_OUT = "logged_out"
    LOGGED_IN = "logged_in"
    EXPIRED = "expired"
    ERROR = "error"


class OrderRoute(str, Enum):
    FETCH = "fetch"
    UI = "ui"


@dataclass
class PWOrderResult:
    ok: bool
    route: OrderRoute
    http_status: Optional[int]
    order_id: Optional[str]
    filled_qty: float
    filled_price: float
    status: str               # "filled" | "rejected" | "pending"
    error: Optional[str]
    raw: dict = field(default_factory=dict)
    sent_body: dict = field(default_factory=dict)


@dataclass
class PWBalance:
    quote_asset: str = QUOTE_ASSET
    free: float = 0.0
    total: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class PWStatus:
    state: SessionState
    headless: bool
    profile_dir: str
    last_error: Optional[str]
    started_at: Optional[int]
    last_request_at: Optional[int]
    pending: int
    fetch_route_dead: bool


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class PlaywrightSessionError(Exception):
    pass


class NotStartedError(PlaywrightSessionError):
    pass


class LoginRequiredError(PlaywrightSessionError):
    pass


class SessionExpiredError(PlaywrightSessionError):
    pass


class BrowserCrashedError(PlaywrightSessionError):
    pass


class RequestTimeoutError(PlaywrightSessionError):
    pass


# ----------------------------------------------------------------------
# Internal request envelope
# ----------------------------------------------------------------------

@dataclass
class _Req:
    kind: str                          # "probe" | "order" | "balance" | "stop"
    payload: dict = field(default_factory=dict)
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


# ----------------------------------------------------------------------
# Worker — owns Playwright objects, single thread
# ----------------------------------------------------------------------

class PlaywrightSession:
    """Singleton driver. Construct via `get_session()`.

    Lifecycle: STOPPED → STARTING → (LOGGED_OUT | LOGGED_IN) → (ERROR | EXPIRED) → STOPPED
    """

    def __init__(
        self,
        profile_dir: Path = PROFILE_DIR,
        user_agent: str = USER_AGENT,
        nav_timeout_ms: int = DEFAULT_NAV_TIMEOUT_MS,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
        request_timeout_s: float = REQUEST_QUEUE_TIMEOUT_S,
    ) -> None:
        self.profile_dir = profile_dir
        self.user_agent = user_agent
        self.nav_timeout_ms = nav_timeout_ms
        self.action_timeout_ms = action_timeout_ms
        self.request_timeout_s = request_timeout_s

        self._queue: "queue.Queue[_Req]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._worker_ready = threading.Event()

        self._state_lock = threading.Lock()
        self._state: SessionState = SessionState.STOPPED
        self._headless: bool = False
        self._last_error: Optional[str] = None
        self._started_at: Optional[int] = None
        self._last_request_at: Optional[int] = None
        self._fetch_route_dead: bool = False
        self._on_state_change: Optional[Callable[[SessionState], None]] = None

    # --- state helpers ------------------------------------------------

    def _set_state(self, new: SessionState, err: Optional[str] = None) -> None:
        with self._state_lock:
            self._state = new
            if err is not None:
                self._last_error = err
        if self._on_state_change:
            try:
                self._on_state_change(new)
            except Exception:
                pass
        logger.info(f"playwright_session: state -> {new.value}" + (f" ({err})" if err else ""))

    def _get_state(self) -> SessionState:
        with self._state_lock:
            return self._state

    def set_on_state_change(self, cb: Optional[Callable[[SessionState], None]]) -> None:
        self._on_state_change = cb

    # --- public API ---------------------------------------------------

    def start(self, headless: bool = False) -> PWStatus:
        """Launch the worker thread + persistent Chromium. Idempotent."""
        st = self._get_state()
        if st in (SessionState.STARTING, SessionState.LOGGED_IN, SessionState.LOGGED_OUT):
            logger.info(f"playwright_session.start(): already {st.value} — no-op")
            return self.status()
        if st in (SessionState.ERROR, SessionState.STOPPED, SessionState.EXPIRED):
            # spin up
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self._headless = headless
            self._set_state(SessionState.STARTING)
            self._worker_ready.clear()
            self._worker = threading.Thread(
                target=self._worker_main,
                name="pw-worker",
                daemon=True,
            )
            self._worker.start()
            if not self._worker_ready.wait(WORKER_START_TIMEOUT_S):
                self._set_state(SessionState.ERROR, "worker did not start in time")
                raise BrowserCrashedError("worker did not start in time")
            self._started_at = int(time.time())
        return self.status()

    def is_logged_in(self, force_refresh: bool = False) -> bool:
        """Probe the page for login state. Blocks up to request_timeout_s."""
        if self._get_state() == SessionState.STOPPED:
            raise NotStartedError("call start() first")
        if self._get_state() == SessionState.STARTING:
            # Best-effort wait for the worker to finish its first navigation.
            for _ in range(50):
                if self._get_state() != SessionState.STARTING:
                    break
                time.sleep(0.1)
        st = self._get_state()
        if st == SessionState.LOGGED_IN and not force_refresh:
            return True
        if st in (SessionState.ERROR, SessionState.EXPIRED) and not force_refresh:
            return False
        result = self._dispatch("probe", {"force_refresh": force_refresh})
        return bool(result.get("logged_in"))

    def place_order(
        self,
        buy_currency: str,
        sell_currency: str,
        buy_amount: float,
        sell_amount: float,
        visible_price: float,
    ) -> PWOrderResult:
        """Submit an instant_order via the running browser session."""
        st = self._get_state()
        if st == SessionState.STOPPED:
            raise NotStartedError("call start() first")
        if st in (SessionState.LOGGED_OUT, SessionState.EXPIRED):
            raise LoginRequiredError(f"session state is {st.value}")
        body = {
            "buy_currency_code": str(buy_currency).upper(),
            "sell_currency_code": str(sell_currency).upper(),
            "visible_price": _fmt_num(visible_price),
            "buy_amount": _fmt_num(buy_amount),
            "sell_amount": _fmt_num(sell_amount),
            "account_number": _get_account_number(),
            "depth_order": True,
            "with_bonus": False,
            "with_stake": False,
        }
        out = self._dispatch("order", {"body": body})
        return out  # already a PWOrderResult

    def snapshot_leaderboard_volume(self, leaderboard_url: str,
                                    email_fragment: str = "nick",
                                    timeout_s: float = 15.0) -> dict:
        """Open the andX leaderboard URL in a fresh tab on the attached
        Chrome, find the user's row by email fragment, parse the Volume
        cell. Returns {volume, award, rank, raw, ok, error}.

        Used to bracket every order with a before/after snapshot so the
        user can see exactly how much volume each fill adds to their
        competition rank."""
        st = self._get_state()
        if st == SessionState.STOPPED:
            raise NotStartedError("call start() first")
        if st in (SessionState.LOGGED_OUT, SessionState.EXPIRED):
            raise LoginRequiredError(f"session state is {st.value}")
        req = _Req(kind="leaderboard",
                   payload={"url": leaderboard_url,
                            "email_fragment": email_fragment})
        self._queue.put(req)
        if not req.event.wait(max(timeout_s, self.request_timeout_s)):
            raise RequestTimeoutError("leaderboard snapshot timed out")
        if req.error:
            raise req.error
        return req.result

    def inspect_trade_page(self, base: str, timeout_s: float = 10.0) -> dict:
        """Diagnostic: navigate to /instant/trade/USDT_<base> and dump every
        visible button, input, and the page title/URL so we can iterate on
        selectors. Blocks up to timeout_s."""
        st = self._get_state()
        if st == SessionState.STOPPED:
            raise NotStartedError("call start() first")
        if st in (SessionState.LOGGED_OUT, SessionState.EXPIRED):
            raise LoginRequiredError(f"session state is {st.value}")
        # Use the queue with a longer timeout for diagnostics.
        req = _Req(kind="inspect", payload={"base": base})
        self._queue.put(req)
        if not req.event.wait(max(timeout_s, self.request_timeout_s)):
            raise RequestTimeoutError("inspect timed out")
        if req.error:
            raise req.error
        return req.result

    def get_balance(self) -> PWBalance:
        st = self._get_state()
        if st == SessionState.STOPPED:
            raise NotStartedError("call start() first")
        if st in (SessionState.LOGGED_OUT, SessionState.EXPIRED):
            raise LoginRequiredError(f"session state is {st.value}")
        out = self._dispatch("balance", {})
        return out

    def status(self) -> PWStatus:
        with self._state_lock:
            return PWStatus(
                state=self._state,
                headless=self._headless,
                profile_dir=str(self.profile_dir),
                last_error=self._last_error,
                started_at=self._started_at,
                last_request_at=self._last_request_at,
                pending=self._queue.qsize(),
                fetch_route_dead=self._fetch_route_dead,
            )

    def stop(self, timeout_s: float = 8.0) -> None:
        if self._get_state() == SessionState.STOPPED:
            return
        req = _Req(kind="stop")
        self._queue.put(req)
        if self._worker:
            self._worker.join(timeout_s)
        self._set_state(SessionState.STOPPED)
        self._started_at = None

    def wipe_profile(self) -> None:
        if self._get_state() != SessionState.STOPPED:
            raise PlaywrightSessionError("stop() before wiping the profile")
        if self.profile_dir.exists():
            shutil.rmtree(self.profile_dir, ignore_errors=True)

    # --- dispatch helper ---------------------------------------------

    def _dispatch(self, kind: str, payload: dict) -> Any:
        req = _Req(kind=kind, payload=payload)
        self._queue.put(req)
        if not req.event.wait(self.request_timeout_s):
            # Worker is slow or wedged. Don't read req.result — it may still
            # be written by the worker. We just abandon and surface a timeout.
            raise RequestTimeoutError(f"{kind} request timed out after "
                                      f"{self.request_timeout_s}s")
        self._last_request_at = int(time.time())
        if req.error:
            raise req.error
        return req.result

    # --- worker main loop --------------------------------------------

    def _worker_main(self) -> None:
        """Owns the Playwright + Chromium. Single thread, single page."""
        try:
            from playwright.sync_api import sync_playwright, Error as PWError
        except ImportError as e:
            self._set_state(SessionState.ERROR, f"playwright not installed: {e}")
            self._worker_ready.set()
            return

        pw = None
        ctx = None
        page = None
        browser = None

        # CDP-attach mode (ANDX_PW_CDP_URL set) connects to a Chrome the user
        # is already controlling. They launch their REAL Chrome with
        #   chrome.exe --remote-debugging-port=9222 --user-data-dir=...
        # log into andX normally (no bot-detection because it's a real
        # Chrome with no automation flags), and we ride that session.
        # This sidesteps the andX login-loop that Playwright's bundled
        # Chromium triggers via fingerprinting.
        cdp_url = os.environ.get("ANDX_PW_CDP_URL", "").strip()

        try:
            pw = sync_playwright().start()
            if cdp_url:
                logger.info(f"playwright_session: CDP-attaching to {cdp_url}")
                browser = pw.chromium.connect_over_cdp(cdp_url)
                # The user's existing Chrome already has contexts; reuse the
                # first one that has an andX tab if possible, else fall back
                # to the default context.
                ctx = None
                for c in browser.contexts:
                    for p in c.pages:
                        if "andx.one" in (p.url or "").lower():
                            ctx = c
                            page = p
                            break
                    if ctx:
                        break
                if ctx is None:
                    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                    page = ctx.pages[0] if ctx.pages else ctx.new_page()
            else:
                ctx = pw.chromium.launch_persistent_context(
                    user_data_dir=str(self.profile_dir),
                    headless=self._headless,
                    user_agent=self.user_agent,
                    viewport={"width": 1440, "height": 900},
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                    ignore_default_args=["--enable-automation"],
                )
                # Reuse the first page if any (persistent contexts often open one),
                # else create.
                if ctx.pages:
                    page = ctx.pages[0]
                else:
                    page = ctx.new_page()
            ctx.set_default_navigation_timeout(self.nav_timeout_ms)
            ctx.set_default_timeout(self.action_timeout_ms)
            # Navigate to home only if we're not already on an andX page —
            # in CDP-attach mode the user's tab may already be on a useful
            # page (trade form, dashboard) and we shouldn't disrupt it.
            cur_url = (page.url or "").lower()
            if "andx.one" not in cur_url:
                try:
                    page.goto(HOME_URL, wait_until="domcontentloaded")
                except PWError as e:
                    logger.warning(f"pw-worker initial goto failed: {e}")
            # Detect login state from the URL after settle.
            time.sleep(1.2)
            url = (page.url or "").lower()
            if "login" in url or "signin" in url:
                self._set_state(SessionState.LOGGED_OUT)
            else:
                # heuristic — probe the balance endpoint
                logged_in = _probe_logged_in(page)
                self._set_state(
                    SessionState.LOGGED_IN if logged_in else SessionState.LOGGED_OUT
                )
        except Exception as e:
            self._set_state(SessionState.ERROR, f"startup failed: {e}")
            self._worker_ready.set()
            self._teardown(ctx, pw)
            return

        self._worker_ready.set()

        # Idle keepalive: pump a mouse-move every 20s while waiting on the
        # queue so the page's Socket.IO + axios stay warm.
        KEEPALIVE_S = 20.0
        last_keepalive = time.time()

        while True:
            try:
                req: _Req = self._queue.get(timeout=KEEPALIVE_S)
            except queue.Empty:
                # idle tick
                try:
                    if page and not page.is_closed():
                        page.mouse.move(10 + int(time.time()) % 5, 10)
                except Exception:
                    pass
                last_keepalive = time.time()
                continue

            if req.kind == "stop":
                req.event.set()
                break

            try:
                if req.kind == "probe":
                    force = bool((req.payload or {}).get("force_refresh"))
                    result = self._handle_probe(page, force)
                    req.result = result
                elif req.kind == "balance":
                    req.result = self._handle_balance(page)
                elif req.kind == "order":
                    body = (req.payload or {}).get("body") or {}
                    req.result = self._handle_order(page, ctx, body)
                elif req.kind == "inspect":
                    base = (req.payload or {}).get("base", "BTC").upper()
                    req.result = self._handle_inspect(page, base)
                elif req.kind == "leaderboard":
                    p = req.payload or {}
                    req.result = self._handle_leaderboard(
                        page, ctx, p.get("url", ""),
                        p.get("email_fragment", "nick"),
                    )
                else:
                    req.error = PlaywrightSessionError(f"unknown kind: {req.kind}")
            except SessionExpiredError as e:
                self._set_state(SessionState.EXPIRED, str(e))
                req.error = e
            except Exception as e:
                # If the browser died, surface it explicitly.
                msg = str(e)
                if "Target closed" in msg or "Connection closed" in msg or "browser closed" in msg.lower():
                    self._set_state(SessionState.ERROR, f"browser died: {msg}")
                    req.error = BrowserCrashedError(msg)
                else:
                    req.error = e
            finally:
                req.event.set()
                # opportunistic keepalive bookkeeping
                last_keepalive = time.time()

        self._teardown(ctx, pw)

    def _teardown(self, ctx, pw) -> None:
        # In CDP-attach mode we DO NOT close the context — that would close
        # the user's Chrome window. We only close contexts we launched
        # ourselves (launch_persistent_context). Detected via env var.
        cdp_attached = bool(os.environ.get("ANDX_PW_CDP_URL", "").strip())
        if not cdp_attached:
            try:
                if ctx:
                    ctx.close()
            except Exception:
                pass
        try:
            if pw:
                pw.stop()
        except Exception:
            pass

    # --- handlers (run on worker thread only) ------------------------

    def _handle_probe(self, page, force_refresh: bool) -> dict:
        if force_refresh:
            try:
                page.goto(HOME_URL, wait_until="domcontentloaded")
                time.sleep(0.8)
            except Exception:
                pass
        url = (page.url or "").lower()
        if "login" in url or "signin" in url:
            self._set_state(SessionState.LOGGED_OUT)
            return {"logged_in": False, "url": page.url}
        ok = _probe_logged_in(page)
        if ok:
            self._set_state(SessionState.LOGGED_IN)
        else:
            self._set_state(SessionState.LOGGED_OUT)
        return {"logged_in": ok, "url": page.url}

    def _handle_leaderboard(self, page, ctx, url: str,
                            email_fragment: str) -> dict:
        """Open the leaderboard URL in a NEW page (so we don't disturb the
        trade page the order flow is using) and parse the user's row.
        The page is closed before returning so we never accumulate tabs.

        Tolerates a variety of layouts: looks for any table-like row that
        contains the email fragment, then pulls numeric cells from that row
        and labels them by their column header. Returns the volume as the
        last/largest numeric cell (Volume is always the rightmost column
        in the leaderboards observed)."""
        from playwright.sync_api import Error as PWError
        if not url:
            return {"ok": False, "error": "no leaderboard URL configured"}
        new_page = None
        try:
            new_page = ctx.new_page()
            new_page.goto(url, wait_until="domcontentloaded")
            time.sleep(2.5)  # let the leaderboard render rows
            data = new_page.evaluate("""(emailFragment) => {
              const frag = (emailFragment || '').toLowerCase();
              // We're looking for the row that contains the user's masked
              // email (e.g. "ni****@****.io") AND looks like a leaderboard
              // row: contains "USDT" and an "@" plus at least one star-mask
              // ("****") since andX masks emails on its leaderboards.
              const candidates = Array.from(document.querySelectorAll(
                'tr, [role=row], .MuiTableRow-root, .row, .leaderboard-row, div'
              ));
              let best = null;
              for (const el of candidates) {
                const raw = (el.innerText || '');
                const text = raw.toLowerCase();
                if (text.length > 600) continue;
                if (!text.includes(frag)) continue;
                if (!text.includes('@')) continue;     // require an email
                if (!text.includes('****')) continue;  // require email mask
                if (!text.toLowerCase().includes('usdt')) continue;  // require USDT cells
                // Extract every numeric token from this row.
                const tokens = raw.split(/\\s+/);
                const nums = [];
                for (const t of tokens) {
                  const cleaned = t.replace(/[,$]/g, '');
                  const f = parseFloat(cleaned);
                  if (!isNaN(f) && cleaned.length > 0
                      && cleaned.length < 20) {
                    nums.push(f);
                  }
                }
                // Prefer the SHORTEST matching row (the tightest container).
                if (!best || raw.length < best.row_text.length) {
                  best = {row_text: raw.slice(0, 500), nums};
                }
              }
              return best || {row_text: null, nums: []};
            }""", email_fragment)
            row_text = data.get("row_text") or ""
            nums = data.get("nums") or []
            if not row_text:
                return {
                    "ok": False,
                    "error": f"no leaderboard row found for fragment '{email_fragment}'",
                    "url": url,
                }
            # Heuristic: numeric tokens in priority order are typically
            #   rank, award, volume.
            # Volume is always the LARGEST in the row (it's cumulative
            # turnover; award is a fraction). Award is the middle.
            volume = max(nums) if nums else 0.0
            award = sorted(nums)[-2] if len(nums) >= 2 else 0.0
            rank = int(min(nums)) if nums else 0
            return {
                "ok": True,
                "url": url,
                "row_text": row_text,
                "rank": rank,
                "award": award,
                "volume": volume,
                "raw_numbers": nums,
            }
        except PWError as e:
            return {"ok": False, "error": f"nav/eval failed: {e}", "url": url}
        finally:
            try:
                if new_page and not new_page.is_closed():
                    new_page.close()
            except Exception:
                pass

    def _handle_inspect(self, page, base: str) -> dict:
        """Navigate to the trade page and dump every interactable element.
        Used by /api/andx/pw_inspect to discover the real DOM so we can
        write accurate selectors instead of guessing."""
        from playwright.sync_api import Error as PWError
        target_url = INSTANT_TRADE_URL_TMPL.format(base=base)
        try:
            page.goto(target_url, wait_until="domcontentloaded")
            time.sleep(1.5)  # let the SPA paint the order form
        except PWError as e:
            return {"error": f"nav failed: {e}", "url": page.url}
        try:
            dom = page.evaluate("""() => {
              const visible = (el) => {
                const r = el.getBoundingClientRect();
                const s = window.getComputedStyle(el);
                return r.width > 0 && r.height > 0
                       && s.display !== 'none' && s.visibility !== 'hidden';
              };
              const buttons = Array.from(document.querySelectorAll(
                'button, [role=button], a.btn, input[type=submit]'
              )).filter(visible).slice(0, 80).map(el => ({
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role'),
                text: (el.innerText || el.value || '').trim().slice(0, 80),
                cls:  (el.className && el.className.toString) ? el.className.toString().slice(0, 100) : '',
                id:   el.id || '',
                disabled: el.disabled === true,
                aria_label: el.getAttribute('aria-label') || '',
                w: Math.round(el.getBoundingClientRect().width),
                h: Math.round(el.getBoundingClientRect().height),
              }));
              const inputs = Array.from(document.querySelectorAll(
                'input, textarea, select'
              )).filter(visible).slice(0, 40).map(el => ({
                tag: el.tagName.toLowerCase(),
                type: el.type || '',
                inputmode: el.inputMode || el.getAttribute('inputmode') || '',
                name: el.name || '',
                placeholder: el.placeholder || '',
                aria_label: el.getAttribute('aria-label') || '',
                value: (el.value || '').slice(0, 40),
                cls: (el.className && el.className.toString) ? el.className.toString().slice(0, 100) : '',
                id: el.id || '',
              }));
              const tabs = Array.from(document.querySelectorAll(
                '[role=tab], .tab, .tabs > *, nav button, nav a'
              )).filter(visible).slice(0, 20).map(el => ({
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role'),
                text: (el.innerText || '').trim().slice(0, 40),
                cls: (el.className && el.className.toString) ? el.className.toString().slice(0, 100) : '',
              }));
              return {
                title: document.title,
                url: location.href,
                buttons, inputs, tabs,
              };
            }""")
            return {"ok": True, **dom}
        except Exception as e:
            return {"ok": False, "error": str(e), "url": page.url}

    def _handle_balance(self, page) -> PWBalance:
        out = _fetch_json(page, BALANCE_PATH, method="GET", body=None)
        http = out.get("status")
        if http in (401, 403):
            raise SessionExpiredError(f"balance probe got HTTP {http}")
        raw = out.get("json") or {}
        free, total = _parse_usdt_balance(raw)
        return PWBalance(free=free, total=total, raw=raw)

    def _handle_order(self, page, ctx, body: dict) -> PWOrderResult:
        """Try fetch-route first; if 401 (or fetch-route already known
        dead), fall back to UI-route. UI-route auto-retries once on price
        drift (status_code 1210) to convert near-misses into fills."""
        # Route 1 — fetch (cheap path)
        if not self._fetch_route_dead:
            res = self._order_via_fetch(page, body)
            if res.ok or (res.http_status and res.http_status >= 400 and res.http_status != 401):
                return res
            if res.http_status == 401:
                logger.warning("playwright_session: fetch route returned 401; "
                               "marking dead and falling back to UI route")
                self._fetch_route_dead = True
            elif res.http_status is None:
                logger.info(f"playwright_session: fetch route returned no status "
                            f"({res.error}); trying UI fallback")
        # Route 2 — UI with one auto-retry on price-drift.
        res = self._order_via_ui(page, ctx, body)
        if (not res.ok and res.http_status == 200
                and (res.raw or {}).get("status_code") == 1210):
            logger.info("playwright_session: 1210 price-drift; refilling + retry")
            res = self._order_via_ui(page, ctx, body)
        # SAFE retry: the click never fired the order POST (order_submitted
        # False) and it timed out, so NOTHING was placed — re-attempt up to 2x
        # to convert andX's flaky UI into a fill. If a POST DID fire we do NOT
        # retry: the order may have gone through and a retry would double-buy.
        elif (not res.ok and (res.raw or {}).get("timed_out")
              and not (res.raw or {}).get("order_submitted")):
            logger.info(f"playwright_session: order POST never fired — "
                        f"safe retry for {body.get('market_code', '?')}")
            res = self._order_via_ui(page, ctx, body)
        return res

    def _order_via_fetch(self, page, body: dict) -> PWOrderResult:
        """Call /p/v1/order/instant_order/ via page.evaluate(fetch)."""
        out = _fetch_json(page, INSTANT_ORDER_PATH, method="POST", body=body)
        http = out.get("status")
        raw = out.get("json") or {}
        if http == 200 and (raw.get("status") == "success"):
            d = raw.get("data") or {}
            order_id = d.get("order_number") or d.get("order_id") or d.get("id")
            filled_price = float(body.get("visible_price") or 0)
            filled_qty = float(body.get("buy_amount") or 0)
            return PWOrderResult(
                ok=True, route=OrderRoute.FETCH,
                http_status=http,
                order_id=str(order_id) if order_id else None,
                filled_qty=filled_qty, filled_price=filled_price,
                status="filled", error=None, raw=raw, sent_body=body,
            )
        # not ok
        err = (raw.get("reason") or raw.get("error")
               or out.get("error") or f"HTTP {http}")
        return PWOrderResult(
            ok=False, route=OrderRoute.FETCH,
            http_status=http, order_id=None,
            filled_qty=0.0, filled_price=0.0,
            status="rejected", error=str(err), raw=raw, sent_body=body,
        )

    def _order_via_ui(self, page, ctx, body: dict) -> PWOrderResult:
        """Drive the andX trade page, then INTERCEPT the network response
        the page's own JS fires when you click submit. We don't care what
        UI elements/toasts exist — we read the real /p/v1/order/instant_order/
        reply that comes back with a valid access-sign attached by the site.

        Sequence:
          1. Navigate to /instant/trade/USDT_<base> if not already there.
          2. Click BUY/SELL tab toggle (best-effort; site may auto-tab).
          3. Fill the amount input (try several locator heuristics).
          4. Set up a response listener for /p/v1/order/instant_order/.
          5. Click the primary submit button.
          6. If a confirm modal pops up, click its CTA.
          7. Wait up to 15s for the captured response, parse it.
        """
        from playwright.sync_api import Error as PWError
        buy = body.get("buy_currency_code", "").upper()
        sell = body.get("sell_currency_code", "").upper()
        # andX's instant-trade box is the amount you PAY (the "Spent Amount").
        # For a BUY you pay USDT  -> fill sell_amount (the USDT notional, ~$ size).
        # andX's amount box has a swap toggle (img alt="swap icon") that flips
        # it between the COIN and USDT. The DEFAULT side is wrong for us — BUY
        # defaults to the coin, SELL defaults to USDT — so before filling we
        # press the toggle to the side we're going to type, then fill the
        # matching value (which is `sell_amount` in BOTH cases):
        #   BUY  → toggle to USDT,  fill sell_amount (= the USDT you spend)
        #   SELL → toggle to <coin>, fill sell_amount (= the coin qty you sell)
        if buy == QUOTE_ASSET:
            base = sell
            side_word = "sell"
            amount_str = body.get("sell_amount")   # coin qty (toggle set to coin)
        else:
            base = buy
            side_word = "buy"
            amount_str = body.get("sell_amount")   # USDT amount (toggle set to USDT)

        target_url = INSTANT_TRADE_URL_TMPL.format(base=base)
        try:
            # Skip navigation if we're already on this coin's trade page.
            cur = (page.url or "").lower()
            if f"_{base.lower()}" not in cur or "instant/trade" not in cur:
                page.goto(target_url, wait_until="domcontentloaded")
                time.sleep(1.8)  # let the MUI bundle paint the form
        except PWError as e:
            return _ui_reject(body, base, f"nav failed: {e}", page.url)
        # Verify we actually landed on this coin's page — a modal or auth
        # dance can silently swallow goto(), leaving the page on the prior
        # coin. If so, the subsequent form interactions hit the wrong coin.
        cur = (page.url or "").lower()
        if f"_{base.lower()}" not in cur and base.lower() not in cur.rsplit("/", 1)[-1]:
            # Try once more with full reload
            try:
                page.goto(target_url, wait_until="domcontentloaded")
                time.sleep(1.5)
            except PWError:
                pass
            cur = (page.url or "").lower()
            if f"_{base.lower()}" not in cur and base.lower() not in cur.rsplit("/", 1)[-1]:
                return _ui_reject(body, base,
                                  f"navigation didn't reach {base} page (stuck on {page.url})",
                                  page.url)

        # CRITICAL: pull andX's *current* page price and rewrite the body
        # so visible_price matches what the page sees. Without this, the
        # bot's price_hint (sourced from Alpaca) drifts from andX's price
        # by enough that the depth-order check silently blocks the form
        # submit (no POST ever fires, just an 18s timeout).
        page_price = _read_page_price(page)
        if page_price > 0:
            body = dict(body)  # don't mutate caller's body
            body["visible_price"] = _fmt_num(page_price)
            try:
                if side_word == "buy":
                    body["sell_amount"] = _fmt_num(
                        float(body["buy_amount"]) * page_price)
                else:
                    body["buy_amount"] = _fmt_num(
                        float(body["sell_amount"]) * page_price)
            except Exception:
                pass
            # sell_amount is the value that matches the toggled side: for a BUY
            # it's the USDT to spend, for a SELL it's the coin quantity to sell.
            amount_str = body["sell_amount"]

        # andX's tab + button labels use the currency you're RECEIVING under
        # the CURRENTLY-ACTIVE side. So the label for the same tab differs
        # depending on what's active right now:
        #   When BUY is active   → both tabs read "Instant <BASE> *"
        #   When SELL is active  → both tabs read "Instant USDT *"
        # To switch sides reliably we have to try BOTH variants — we don't
        # know which one is currently displayed (depends on the user's
        # previous interaction on this URL within this profile).
        side_label_a = base if side_word == "buy" else QUOTE_ASSET
        side_label_b = QUOTE_ASSET if side_word == "buy" else base
        tab_clicked = False
        for lbl in (side_label_a, side_label_b):
            tab_re = _re_icase(f"instant {lbl} {side_word}")
            try:
                page.get_by_role("tab", name=tab_re).first.click(timeout=1500)
                tab_clicked = True
                break
            except Exception:
                try:
                    page.get_by_role("button", name=tab_re).first.click(timeout=1000)
                    tab_clicked = True
                    break
                except Exception:
                    continue
        if not tab_clicked:
            logger.warning(f"playwright_session: side tab '{side_word}' not "
                           f"clickable on {base} page after both label "
                           f"variants; continuing")

        # Wait for the form to settle after clicking the side tab — MUI
        # re-renders the input fields when switching Buy/Sell.
        time.sleep(1.2)

        # Toggle the amount box to the currency we're about to type. andX
        # defaults to the WRONG side (BUY→coin, SELL→USDT); the swap icon flips
        # it. BUY types USDT, SELL types the coin. Without this the amount is
        # read in the wrong unit and the order silently never fires.
        want_ccy = QUOTE_ASSET if side_word == "buy" else base
        if not _ensure_spent_currency(page, want_ccy):
            logger.warning(f"playwright_session: could not set amount box to "
                           f"{want_ccy} on {base}; filling anyway")
        time.sleep(0.4)

        # Amount input is `<input type=text placeholder="Enter amount">`.
        # On the instant trade page there are TWO such inputs: the active
        # "you pay" field (enabled) and a "you receive" field (Mui-disabled
        # because it's auto-computed). We MUST target the enabled one. MUI
        # controlled inputs also need a click + type sequence to fire the
        # React events that enable the confirm button.
        amount_filled = False
        for locator_fn in (
            lambda: page.locator(
                "input[placeholder*='amount' i]:not([disabled]):not(.Mui-disabled)"
            ).first,
            lambda: page.locator(
                "input[placeholder='Enter amount']:not([disabled])"
            ).first,
            # last resort — any enabled text input
            lambda: page.locator(
                "input[type='text']:not([disabled]):not(.Mui-disabled)"
            ).first,
        ):
            try:
                loc = locator_fn()
                loc.wait_for(state="visible", timeout=2500)
                loc.click(timeout=1500)
                loc.press("Control+a")
                loc.press("Delete")
                loc.type(str(amount_str), delay=15, timeout=2500)
                # Sanity-check the value committed.
                try:
                    val = loc.input_value(timeout=500)
                    if not val:
                        continue
                except Exception:
                    pass
                amount_filled = True
                break
            except Exception:
                continue
        if not amount_filled:
            return _ui_reject(body, base, "could not find amount input", page.url)

        # Submit button uses the receive-currency label of the ACTIVE side:
        #   BUY  → "<BASE> Buy Confirmation"
        #   SELL → "USDT Sell Confirmation"
        # After a successful tab click the button text matches side_label_a.
        # If the tab click failed (tab_clicked=False) the page may still be
        # on the prior side — either label is acceptable; the response
        # interception below will tell us if a real order was submitted.
        confirm_label = side_label_a if tab_clicked else side_label_b
        confirm_re = _re_icase(f"{confirm_label} {side_word} confirmation")
        try:
            confirm_btn = page.get_by_role(
                "button", name=confirm_re
            ).first
            # Wait up to 6s for the button to appear AND become enabled.
            # MUI re-renders the form on tab switch + amount entry, so the
            # button can vanish briefly before reappearing in enabled state.
            confirm_btn.wait_for(state="visible", timeout=6000)
        except Exception as e:
            return _ui_reject(body, base,
                              f"confirm button not found: {e}", page.url)

        # Track whether the order POST actually left the browser. This lets
        # the caller retry SAFELY — only when NO request fired — so a live
        # order whose response was merely slow can never be double-placed.
        submitted = {"v": False}
        def _mark_submitted(req):
            try:
                if INSTANT_ORDER_PATH in req.url and req.method == "POST":
                    submitted["v"] = True
            except Exception:
                pass
        page.on("request", _mark_submitted)
        try:
            with page.expect_response(
                lambda r: INSTANT_ORDER_PATH in r.url and r.request.method == "POST",
                timeout=18_000,
            ) as resp_info:
                try:
                    confirm_btn.click(timeout=3000)
                except Exception as click_err:
                    # If the button was disabled or covered, force-fire via JS.
                    try:
                        confirm_btn.dispatch_event("click")
                    except Exception:
                        return _ui_reject(body, base,
                                          f"could not click confirm: {click_err}",
                                          page.url, order_submitted=submitted["v"])
                # andX may pop a second-stage confirm dialog — click any
                # primary CTA in a visible dialog if one shows up.
                try:
                    for cta in ("confirm", "yes", "ok", side_word, "proceed"):
                        try:
                            page.get_by_role("dialog").get_by_role(
                                "button", name=_re_icase(cta)
                            ).first.click(timeout=1200)
                            break
                        except Exception:
                            continue
                except Exception:
                    pass
            resp = resp_info.value
        except Exception as e:
            # Capture diagnostic state so we can see WHY the click didn't
            # fire a request (disabled button? overlay? wrong page?).
            diag = _ui_diag(page, confirm_re)
            return _ui_reject(
                body, base,
                f"no instant_order POST captured ({e}) — diag: {diag}",
                page.url, order_submitted=submitted["v"],
            )
        finally:
            try:
                page.remove_listener("request", _mark_submitted)
            except Exception:
                pass

        # Parse the captured response.
        http = resp.status
        try:
            raw = resp.json()
        except Exception:
            try:
                raw = {"text": resp.text()[:600]}
            except Exception:
                raw = {}
        if http == 401 or http == 403:
            self._set_state(SessionState.EXPIRED,
                            f"instant_order returned HTTP {http}")
            raise SessionExpiredError(f"instant_order returned HTTP {http}")
        if http == 200 and (raw.get("status") == "success"):
            d = raw.get("data") or {}
            order_id = d.get("order_number") or d.get("order_id") or d.get("id")
            return PWOrderResult(
                ok=True, route=OrderRoute.UI,
                http_status=http,
                order_id=str(order_id) if order_id else None,
                filled_qty=float(amount_str or 0),
                filled_price=float(body.get("visible_price") or 0),
                status="filled", error=None, raw=raw, sent_body=body,
            )
        err = (raw.get("reason") or raw.get("error")
               or raw.get("text") or f"HTTP {http}")
        return PWOrderResult(
            ok=False, route=OrderRoute.UI,
            http_status=http, order_id=None,
            filled_qty=0.0, filled_price=0.0,
            status="rejected", error=str(err), raw=raw, sent_body=body,
        )


# ----------------------------------------------------------------------
# In-page helpers (called by the worker thread)
# ----------------------------------------------------------------------

_FETCH_JS = """
async ({path, method, body}) => {
  const opts = {
    method: method || 'GET',
    credentials: 'include',
    headers: {'Content-Type': 'application/json'},
  };
  if (body !== null && body !== undefined) {
    opts.body = JSON.stringify(body);
  }
  let r;
  try {
    r = await fetch(path, opts);
  } catch (e) {
    return {status: 0, error: String(e), json: null};
  }
  let j = null;
  try { j = await r.json(); } catch (e) {}
  return {status: r.status, json: j};
}
"""


def _fetch_json(page, path: str, method: str = "GET",
                body: Optional[dict] = None) -> dict:
    """Run a same-origin fetch from the page context. Returns {status, json, error}."""
    return page.evaluate(_FETCH_JS, {"path": path, "method": method, "body": body})


def _probe_logged_in(page) -> bool:
    """Login check. Priority order:
      1. If the URL contains /login or /signin → definitely not logged in.
      2. If the document body shows the andX login form heuristically
         (an email input or 'log in'/'sign in' heading) → not logged in.
      3. Otherwise → logged in. We deliberately AVOID hitting a specific
         API path because andX's actual balance/account endpoints are
         not consistently documented; the URL + DOM check is cheap and
         reliable enough to gate first-order placement, and the *real*
         feedback comes from a 401 on the first instant_order.
    """
    try:
        url = (page.url or "").lower()
        if "/login" in url or "/signin" in url or "/auth/" in url:
            return False
        # DOM heuristic — a password input only exists on the login screen
        has_login_form = page.evaluate(
            "() => !!document.querySelector("
            "'input[type=password], input[name*=password i], "
            "form[action*=login i], form[action*=signin i]')"
        )
        if has_login_form:
            return False
        return True
    except Exception:
        # If we can't introspect, fall back to URL-only — better to assume
        # logged-in and let the first order surface 401 than to block the
        # user forever on a probe glitch.
        url = (page.url or "").lower()
        return "/login" not in url and "/signin" not in url


def _parse_usdt_balance(raw: Any) -> tuple[float, float]:
    """Extract (free, total) USDT from a balance response. Tolerant of shapes."""
    if not raw:
        return 0.0, 0.0
    # Common shapes:
    #   {"status":"success","data":[{"currency":"USDT","available":"…","total":"…"}, ...]}
    #   {"balances":[...]}
    #   {"USDT":{"available":...,"total":...}}
    def _walk(obj):
        if isinstance(obj, dict):
            # If the dict itself looks like one balance row, yield it.
            if (("currency" in obj or "currency_code" in obj or "asset" in obj)
                    and ("available" in obj or "free" in obj or "balance" in obj or "total" in obj)):
                yield obj
            for v in obj.values():
                yield from _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from _walk(it)
    free, total = 0.0, 0.0
    for row in _walk(raw):
        ccy = (row.get("currency") or row.get("currency_code") or row.get("asset") or "").upper()
        if ccy != "USDT":
            continue
        f = row.get("available") or row.get("free") or row.get("balance") or 0
        t = row.get("total") or row.get("balance") or row.get("free") or f
        try:
            free = float(f)
            total = float(t)
        except (TypeError, ValueError):
            pass
        break
    return free, total


def _read_spent_currency(page) -> str:
    """The currency the Instant-trade amount box is denominated in right now,
    read from the 'Spent Amount: <CCY>' label. '' if not found."""
    import re
    try:
        m = re.search(r"Spent Amount[:\s]*([A-Za-z0-9]+)",
                      page.inner_text("body", timeout=1500))
        return m.group(1) if m else ""
    except Exception:
        return ""


def _ensure_spent_currency(page, want: str, max_tries: int = 3) -> bool:
    """Press andX's swap-coin toggle until the amount box is denominated in
    `want` (either the coin or 'USDT'). The toggle is the icon button holding
    <img alt="swap icon">. Returns True once the box shows `want`."""
    want = (want or "").upper()
    swap = page.locator("button:has(img[alt='swap icon'])").first
    for _ in range(max_tries):
        cur = _read_spent_currency(page).upper()
        if cur and cur == want:
            return True
        try:
            swap.click(timeout=1500)
        except Exception:
            try:
                swap.dispatch_event("click")
            except Exception:
                return False
        page.wait_for_timeout(700)
    return _read_spent_currency(page).upper() == want


def _read_page_price(page) -> float:
    """Read andX's current displayed price for the loaded trade page.

    The /instant/trade/USDT_<base> page sets its title to:
        "<price> | USDT/<BASE> | <BASE> Buy & Sell | Andx Instant Trade"
    Parsing the first segment is cheap, doesn't need a DOM walk, and
    gives us the exact value the site is depth-checking against. Falls
    back to 0.0 if the title doesn't parse.
    """
    try:
        title = page.title() or ""
        first = title.split("|", 1)[0].strip()
        # Allow comma-thousand separators just in case
        first = first.replace(",", "")
        return float(first)
    except Exception:
        return 0.0


def _ui_diag(page, confirm_re) -> dict:
    """Snapshot the page state when a UI order attempt fails — confirm
    button state, visible alerts, current input value. Read-only."""
    out: dict = {}
    try:
        out["url"] = page.url
        out["title"] = page.title()
    except Exception:
        pass
    try:
        btn = page.get_by_role("button", name=confirm_re).first
        out["confirm_visible"] = btn.is_visible(timeout=600)
        out["confirm_enabled"] = btn.is_enabled(timeout=600)
        out["confirm_text"] = btn.inner_text(timeout=600)[:80]
    except Exception as e:
        out["confirm_lookup_err"] = str(e)[:120]
    try:
        # Any visible alert / error toast text
        alerts = page.evaluate("""() => {
          const sel = '[role=alert], .MuiAlert-root, .Toastify__toast';
          return Array.from(document.querySelectorAll(sel))
            .filter(el => {
              const r = el.getBoundingClientRect();
              const s = getComputedStyle(el);
              return r.width > 0 && r.height > 0
                     && s.display !== 'none' && s.visibility !== 'hidden';
            })
            .slice(0, 5)
            .map(el => (el.innerText || '').trim().slice(0, 200));
        }""")
        if alerts:
            out["alerts"] = alerts
    except Exception:
        pass
    try:
        # Current enabled-amount input value
        val = page.evaluate(
            "() => { const i = document.querySelector("
            "'input[placeholder*=amount i]:not([disabled]):not(.Mui-disabled)'); "
            "return i ? i.value : null; }"
        )
        out["amount_value"] = val
    except Exception:
        pass
    return out


def _ui_reject(body: dict, base: str, error: str, url: str,
               order_submitted: bool = False) -> "PWOrderResult":
    """Build a uniform UI-route rejection result.

    `order_submitted` records whether the instant_order POST actually left
    the browser. When False, the caller may safely retry (nothing was placed);
    when True, it must NOT retry (the order may have gone through)."""
    timed_out = "no instant_order POST captured" in (error or "")
    return PWOrderResult(
        ok=False, route=OrderRoute.UI,
        http_status=None, order_id=None,
        filled_qty=0.0, filled_price=0.0,
        status="rejected", error=error,
        raw={"url": url, "base": base,
             "timed_out": timed_out,
             "order_submitted": order_submitted},
        sent_body=body,
    )


def _read_toast(page) -> tuple[bool, str]:
    """Look for a toast notification. Returns (success?, text)."""
    selectors_success = [
        ".toast-success", ".Toastify__toast--success",
        "[role='status']:has-text('success')",
    ]
    selectors_error = [
        ".toast-error", ".Toastify__toast--error",
        "[role='alert']", "[role='status']:has-text('error')",
    ]
    for sel in selectors_success:
        try:
            t = page.locator(sel).first.inner_text(timeout=800)
            return True, t
        except Exception:
            continue
    for sel in selectors_error:
        try:
            t = page.locator(sel).first.inner_text(timeout=400)
            return False, t
        except Exception:
            continue
    return False, ""


def _re_icase(s: str):
    """Build a case-insensitive regex from a literal for Playwright matchers."""
    import re
    return re.compile(re.escape(s), re.IGNORECASE)


def _fmt_num(n: float) -> str:
    """Format a number the way the website does — plain decimal, no scientific."""
    if abs(float(n)) < 1e-12:
        return "0"
    s = f"{float(n):.18f}".rstrip("0").rstrip(".")
    return s or "0"


def _get_account_number() -> int:
    """Lazy import to avoid a hard dep on andx_session at module load."""
    try:
        import andx_session
        return int(andx_session.get_account_number())
    except Exception:
        return int(os.environ.get("ANDX_ACCOUNT_NUMBER", "266"))


# ----------------------------------------------------------------------
# Singleton accessor
# ----------------------------------------------------------------------

_session_singleton: Optional[PlaywrightSession] = None
_session_lock = threading.Lock()


def get_session() -> PlaywrightSession:
    global _session_singleton
    with _session_lock:
        if _session_singleton is None:
            _session_singleton = PlaywrightSession()
        return _session_singleton


def reset_session() -> None:
    global _session_singleton
    with _session_lock:
        if _session_singleton is not None:
            try:
                _session_singleton.stop()
            except Exception:
                pass
        _session_singleton = None
