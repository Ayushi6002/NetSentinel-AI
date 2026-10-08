"""
run.py — NetSentinel AI Application Launcher
=============================================
Run this file from the project root:

    python run.py

This ensures the project root is on sys.path so that all
backend.* imports resolve correctly regardless of IDE or shell.
"""

import sys
from pathlib import Path

# Guarantee project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app, _autoload_sample_data
import os

if __name__ == "__main__":
    _autoload_sample_data()
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
