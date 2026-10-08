"""
NetSentinel AI — Flask Application Entry Point
================================================
Phase 1: Network Traffic Intelligence

Responsibilities:
  - Initialise Flask and CORS
  - Load environment variables
  - Register route Blueprints
  - Auto-load sample traffic data on startup
  - Serve the frontend placeholder

DO NOT add Phase 2 / Phase 3 code here.
"""
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so backend.* imports resolve
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ----------------------------------------------------------------
# Configure logging before anything else
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ----------------------------------------------------------------
# Load .env (safe — does not crash if .env is absent)
# ----------------------------------------------------------------
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ----------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# ----------------------------------------------------------------
# Register Phase 1 Blueprint
# ----------------------------------------------------------------
from backend.routes.traffic_routes import traffic_bp
app.register_blueprint(traffic_bp, url_prefix="/api/traffic")

# ----------------------------------------------------------------
# Core endpoints
# ----------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint — must always return 200."""
    from utils.supabase_client import get_supabase_config
    from backend.collectors.live_collector import LIVE_CAPTURE_AVAILABLE
    import backend.services.traffic_store as store

    supabase_cfg = get_supabase_config()

    return jsonify({
        "status": "ok",
        "project": "NetSentinel AI",
        "phase": "Phase 1 - Network Traffic Intelligence",
        "data_loaded": store.is_loaded(),
        "record_count": store.get_record_count(),
        "supabase_configured": supabase_cfg["ready"],
        "live_capture_available": LIVE_CAPTURE_AVAILABLE,
    })


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve frontend static files. Fall back to index.html for SPAs."""
    target = FRONTEND_DIR / path
    if path and target.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({
        "message": "NetSentinel AI backend is running",
        "endpoints": {
            "health":       "/api/health",
            "load_data":    "/api/traffic/load",
            "stats":        "/api/traffic/stats",
            "protocols":    "/api/traffic/protocols",
            "top_sources":  "/api/traffic/top-sources",
            "top_dests":    "/api/traffic/top-destinations",
            "timeline":     "/api/traffic/timeline",
            "connections":  "/api/traffic/connections",
        }
    })


# ----------------------------------------------------------------
# Auto-load sample data at startup
# ----------------------------------------------------------------

def _autoload_sample_data():
    """
    Load the sample CSV dataset into memory when the server starts.
    This ensures the dashboard works immediately on first visit.
    Skips loading gracefully if the file is not found.
    """
    import backend.services.traffic_store as store
    sample_path = BASE_DIR / "data" / "sample_network_traffic.csv"
    if sample_path.exists():
        ok, msg, count = store.load_from_csv(sample_path)
        if ok:
            logger.info("Auto-loaded %d sample traffic records.", count)
        else:
            logger.warning("Auto-load failed: %s", msg)
    else:
        logger.warning("Sample CSV not found at '%s' — dashboard will show empty state.", sample_path)


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == "__main__":
    _autoload_sample_data()
    port = int(os.getenv("PORT", 5001))
    logger.info("Starting NetSentinel AI on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
