"""Portfolio Optimization Engine - Desktop Shell

Goal
----
Provide a "real desktop app" feel by embedding the Streamlit UI inside
an Edge WebView2 window (via pywebview).

How it works
------------
- Starts Streamlit app (app.py) in-process using Streamlit's bootstrap API.
- Opens a pywebview window pointing to the local Streamlit server.

Notes
-----
- End users need Microsoft Edge WebView2 Runtime (usually preinstalled on Win10/11).
- This file is the PyInstaller entrypoint.
"""

from __future__ import annotations

import os
import sys
import time
import socket
import threading
from pathlib import Path


def _resource_dir() -> Path:
    """Return base directory for bundled resources."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _find_free_port(preferred: int = 8501) -> int:
    """Find a free localhost port."""
    # Try preferred first
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass

    # Ask OS for a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(port: int, timeout_s: int = 45) -> bool:
    """Wait until Streamlit server accepts connections."""
    start = time.time()
    while time.time() - start < timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except Exception:
                time.sleep(0.25)
    return False


def _run_streamlit(app_path: Path, port: int) -> None:
    """Run Streamlit app using bootstrap API (in a thread)."""
    # Import here so the shell can show helpful errors if streamlit not bundled
    from streamlit.web import bootstrap

    # Ensure working directory matches app resources
    os.chdir(str(app_path.parent))

    # Streamlit flag options (works across recent versions)
    # XSRF is disabled ONLY for local desktop shell where the webview connects to localhost.
    # For cloud deployments, XSRF must be enabled via .streamlit/config.toml.
    flag_options = {
        "server.port": port,
        "server.address": "127.0.0.1",  # Bind to localhost only — never expose to network
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "server.enableCORS": False,
        "server.enableXsrfProtection": False,
    }

    # Command line is informational; args list is empty
    bootstrap.run(str(app_path), "", [], flag_options)


def main() -> int:
    base_dir = _resource_dir()
    app_path = base_dir / "app.py"

    if not app_path.exists():
        # Minimal error message if packaging broke
        print(f"ERROR: app.py not found at {app_path}")
        return 1

    port = _find_free_port(8501)

    # Start Streamlit server
    t = threading.Thread(target=_run_streamlit, args=(app_path, port), daemon=True)
    t.start()

    # Wait for server socket to be ready
    ok = _wait_for_server(port, timeout_s=45)

    # Open webview window
    try:
        import webview
    except Exception as e:
        print("ERROR: pywebview is not available. Install with: pip install pywebview")
        print(str(e))
        return 1

    url = f"http://127.0.0.1:{port}"
    title = "Portfolio Optimization Engine"

    if not ok:
        # Still open; Streamlit may take longer on first run
        pass

    # Create window
    webview.create_window(
        title=title,
        url=url,
        width=1280,
        height=780,
        min_size=(1000, 650),
        confirm_close=True,
    )

    # Use Edge Chromium backend when available
    try:
        webview.start(gui="edgechromium")
    except Exception:
        # Fallback to default
        webview.start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
