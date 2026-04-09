"""Portfolio Optimization Engine - Desktop Launcher

PyInstaller entrypoint that boots the Streamlit UI (app.py).

Why this exists
--------------
Running a Streamlit app by `python app.py` is not the supported execution mode.
It often results in a blank / weird UI or non-responsive buttons.

This launcher uses Streamlit's internal bootstrap API to run the UI exactly
as `streamlit run app.py` would, then opens the browser automatically.

This should be the only entrypoint used for EXE builds.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from pathlib import Path


def _resource_base_dir() -> Path:
    """Return base folder where bundled resources exist."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def main() -> None:
    base = _resource_base_dir()
    app_path = base / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at: {app_path}")

    # Streamlit prefers UTF-8
    os.environ.setdefault("PYTHONUTF8", "1")

    # Avoid analytics noise
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    # Use a deterministic port if user wants, otherwise let Streamlit choose
    port = int(os.environ.get("PORTFOLIO_ENGINE_PORT", "8501"))

    # When packaged, streamlit may run from temp. Ensure cwd is base.
    os.chdir(str(base))

    try:
        from streamlit.web import bootstrap
    except Exception as e:
        raise RuntimeError(
            "Streamlit is not available. Ensure requirements are installed and bundled." 
        ) from e

    # Build browser URL
    url = f"http://localhost:{port}"

    # Open browser shortly after boot begins (server spins up)
    def _open_browser_later() -> None:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading

    threading.Thread(target=_open_browser_later, daemon=True).start()

    # Start Streamlit server
    # This mimics: streamlit run app.py --server.port=<port> --server.headless=true
    bootstrap.run(
        str(app_path),
        "streamlit run",
        [
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.serverAddress=localhost",
            "--client.showErrorDetails=true",
        ],
        flag_options={},
    )


if __name__ == "__main__":
    main()
