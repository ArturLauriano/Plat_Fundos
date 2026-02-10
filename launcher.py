import os
import sys
import time
import webbrowser
from pathlib import Path
import tempfile
import msvcrt
import socket


def _resource_path(relative: str) -> Path:
    """Resolve resource path for PyInstaller onefile/onedir."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative
    return Path(__file__).resolve().parent / relative


def _wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> bool:
    """Best-effort wait until the Streamlit server is accepting connections."""

    start = time.time()
    while time.time() - start < timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def _pick_port(start: int = 8501, attempts: int = 10) -> int:
    """Pick a free localhost port starting from 8501."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _wait_for_any_port(host: str, ports: list[int], timeout_s: float = 30.0) -> int | None:
    """Wait until any of the ports is accepting connections. Returns the port or None."""
    start = time.time()
    while time.time() - start < timeout_s:
        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                try:
                    sock.connect((host, port))
                    return port
                except OSError:
                    continue
        time.sleep(0.5)
    return None


def _open_log(log_path: Path) -> None:
    try:
        os.startfile(str(log_path))
    except Exception:
        pass


def _watch_for_heartbeat(heartbeat_path: Path, idle_grace_s: float = 1800.0, poll_s: float = 2.0) -> None:
    """Exit the app after the Streamlit heartbeat stops for idle_grace_s."""
    had_heartbeat = False
    start_time = time.time()
    while True:
        try:
            mtime = heartbeat_path.stat().st_mtime
            if mtime >= start_time:
                had_heartbeat = True
                if time.time() - mtime > idle_grace_s:
                    os._exit(0)
        except FileNotFoundError:
            # Wait until the app creates the heartbeat.
            pass
        except Exception:
            # Fail open: don't exit if we can't read the heartbeat.
            pass
        time.sleep(poll_s)


def main() -> int:
    app_path = _resource_path("portfolio_app.py")
    if not app_path.exists():
        print(f"App not found: {app_path}")
        return 1

    # Single-instance lock to avoid spawning multiple Streamlit servers/tabs.
    lock_path = Path(tempfile.gettempdir()) / "portfolio_app.lock"
    lock_file = lock_path.open("a+")
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        # Another instance is running; just open the existing app and exit.
        try:
            lock_file.seek(0)
            port_txt = lock_file.read().strip()
            port = int(port_txt) if port_txt else 8501
        except Exception:
            port = 8501
        webbrowser.open(f"http://localhost:{port}")
        return 0

    # Avoid Streamlit telemetry and keep it headless (we open the browser ourselves).
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PORTFOLIO_APP_HEARTBEAT"] = str(
        Path(tempfile.gettempdir()) / f"portfolio_app_{os.getpid()}.heartbeat"
    )

    port = _pick_port(8501, attempts=10)
    log_path = Path(tempfile.gettempdir()) / "portfolio_app.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    log_file.write("PortfolioApp launcher starting...\n")
    log_file.write(f"App path: {app_path}\n")
    log_file.write(f"Python: {sys.executable}\n")
    log_file.write(f"Port: {port}\n")
    log_file.write("Mode: in-process streamlit bootstrap\n")
    log_file.flush()

    # Persist selected port for subsequent runs.
    try:
        lock_file.seek(0)
        lock_file.truncate(0)
        lock_file.write(str(port))
        lock_file.flush()
    except Exception:
        pass

    try:
        import threading
        import contextlib
        import streamlit.config as config
        import streamlit.web.bootstrap as bootstrap

        def _open_browser_when_ready() -> None:
            ports_to_try = list(range(port, port + 10))
            ready_port = _wait_for_any_port("127.0.0.1", ports_to_try, timeout_s=40.0)
            if ready_port is not None:
                webbrowser.open(f"http://localhost:{ready_port}")
                threading.Thread(
                    target=_watch_for_heartbeat,
                    args=(Path(os.environ["PORTFOLIO_APP_HEARTBEAT"]),),
                    daemon=True,
                ).start()
            else:
                _open_log(log_path)

        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

        # Force non-dev mode so Streamlit serves bundled static assets.
        try:
            config.set_option("global.developmentMode", False)
            config.set_option("server.headless", True)
            config.set_option("server.fileWatcherType", "none")
        except Exception:
            pass

        flag_options = {
            "server.headless": True,
            "server.port": port,
            "server.fileWatcherType": "none",
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
        }

        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            bootstrap.run(str(app_path), False, [], flag_options)
        return 0
    finally:
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
