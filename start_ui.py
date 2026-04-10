#!/usr/bin/env python3
"""
S2VMO Digital Twin – UI Launcher
Starts the FastAPI server and opens the browser.

Usage:
    python start_ui.py              # default port 8000
    python start_ui.py --port 9000  # custom port
"""

import argparse
import os
import sys
import webbrowser
import time
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def open_browser(port: int, delay: float = 1.5):
    """Open the browser after a short delay to let the server start."""
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")


def main():
    parser = argparse.ArgumentParser(description="S2VMO Digital Twin UI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("  S2VMO Digital Twin – UI Server")
    print("=" * 60)
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  API:  http://localhost:{args.port}/docs")
    print("=" * 60)
    print()

    if not args.no_browser:
        threading.Thread(
            target=open_browser, args=(args.port,), daemon=True
        ).start()

    try:
        import uvicorn
        uvicorn.run(
            "api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)


if __name__ == "__main__":
    main()
