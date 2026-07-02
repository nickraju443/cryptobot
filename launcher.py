"""
launcher.py — Single-file entry point for the bundled CryptoBot.exe build.

Opens the dashboard in the user's default browser, then runs the Flask app.
PyInstaller wraps this + the rest of the source into a single executable
so non-technical users can double-click to start the bot.
"""

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _resource_dir() -> Path:
    """When running as a PyInstaller bundle, sys._MEIPASS points to the
    temp folder where the bundled files were unpacked. Templates and other
    data files live there. When running from source, use the script's
    parent directory."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _user_data_dir() -> Path:
    """Where the bot reads/writes user-specific files (credentials,
    portfolio state, trade history, etc.). Same folder as the .exe so the
    user can see them; allows easy backup or wipe."""
    if getattr(sys, "frozen", False):
        # Beside the .exe
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _open_browser_after_delay(url: str, delay_s: float = 2.0):
    def _open():
        time.sleep(delay_s)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


def main():
    # Point the bot at a writable folder beside the .exe for all its state.
    # Without this, PyInstaller's temp _MEIPASS gets wiped each launch
    # and the user would lose their credentials / portfolio every restart.
    user_dir = _user_data_dir()
    os.chdir(str(user_dir))

    # Default to long-only, regular mode, no force-deploy on first start.
    # Pat can change these from the dashboard.
    os.environ.setdefault("LIVE_TRADING", "0")  # safe: paper trading until he flips it
    os.environ.setdefault("DEFAULT_RISK_MODE", "REGULAR")
    os.environ.setdefault("DEFAULT_TRADE_MODE", "long")
    os.environ.setdefault("PORT", "5002")

    # Pull the templates folder into a place Flask can find it. PyInstaller
    # extracts templates/ into sys._MEIPASS; Flask looks for it relative to
    # the app's import location. Adjust sys.path so `import app` resolves
    # to the bundled version.
    res = _resource_dir()
    sys.path.insert(0, str(res))

    port = int(os.environ.get("PORT", "5002"))
    print("=" * 60)
    print(" CryptoBot starting...")
    print(f" Dashboard will open at http://localhost:{port}")
    print(" Keep this window open while the bot runs.")
    print(" Close this window to stop the bot.")
    print("=" * 60)
    print()

    _open_browser_after_delay(f"http://localhost:{port}/", delay_s=3.0)

    # Import + run the Flask app from app.py
    import app  # noqa: F401  — registers all routes
    app.app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to close this window...")
