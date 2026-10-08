"""
Traffic Routes — NetSentinel AI Phase 1
========================================
Flask Blueprint containing all Phase 1 /api/traffic/* endpoints.

All endpoints read from the in-memory traffic store (traffic_store.py)
and delegate computation to the analysis engine (traffic_analyzer.py).
No computation happens inside route functions — they only call services.

Endpoints:
  GET  /api/traffic/load          Load the sample CSV dataset
  GET  /api/traffic/status        Current store status
  GET  /api/traffic/stats         Full overview statistics
  GET  /api/traffic/protocols     Protocol distribution
  GET  /api/traffic/top-sources   Top source IP addresses
  GET  /api/traffic/top-destinations  Top destination IP addresses
  GET  /api/traffic/timeline      Traffic volume over time
  GET  /api/traffic/connections   Connection statistics
"""

import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

import backend.services.traffic_store as store
import backend.services.traffic_analyzer as analyzer

logger = logging.getLogger(__name__)

# Default sample data path (relative to project root)
_DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "sample_network_traffic.csv"

traffic_bp = Blueprint("traffic", __name__)


def _error(message: str, status_code: int = 500):
    return jsonify({"error": message}), status_code


def _require_data():
    """Return an error response if no data is loaded, else None."""
    if not store.is_loaded():
        return _error(
            "No traffic data loaded. Call GET /api/traffic/load first.",
            status_code=404,
        )
    return None


# ----------------------------------------------------------------
# Load sample data
# ----------------------------------------------------------------

@traffic_bp.route("/load", methods=["GET", "POST"])
def load_traffic():
    """
    Load (or reload) the sample network traffic CSV dataset into memory.

    Query params:
      path (optional): relative or absolute path to a CSV file.
                       Defaults to data/sample_network_traffic.csv
    """
    csv_path = request.args.get("path", "").strip()
    target = Path(csv_path) if csv_path else _DEFAULT_CSV

    success, message, count = store.load_from_csv(target)

    status = 200 if success else 400
    return jsonify({
        "success": success,
        "message": message,
        "records_loaded": count,
        "source": str(target),
        "note": "SIMULATED/SAMPLE DATA — not real network traffic",
    }), status


# ----------------------------------------------------------------
# Store status
# ----------------------------------------------------------------

@traffic_bp.route("/status", methods=["GET"])
def traffic_status():
    """Return current in-memory store status."""
    return jsonify({
        "loaded": store.is_loaded(),
        "record_count": store.get_record_count(),
        "supabase_configured": _supabase_status(),
        "live_capture_available": _live_capture_status(),
        "data_mode": "CSV/Sample" if store.is_loaded() else "No data loaded",
    })


# ----------------------------------------------------------------
# Full stats (convenience: all analyses in one call)
# ----------------------------------------------------------------

@traffic_bp.route("/stats", methods=["GET"])
def traffic_stats():
    """Return complete traffic overview statistics."""
    err = _require_data()
    if err:
        return err

    df = store.get_dataframe()
    freq = request.args.get("freq", "1min")

    try:
        result = analyzer.analyse_all(df, timeline_freq=freq)
        return jsonify(result)
    except Exception as exc:
        logger.exception("traffic_stats error")
        return _error(f"Analysis failed: {exc}")


# ----------------------------------------------------------------
# Protocol distribution
# ----------------------------------------------------------------

@traffic_bp.route("/protocols", methods=["GET"])
def traffic_protocols():
    """Return per-protocol traffic statistics."""
    err = _require_data()
    if err:
        return err

    df = store.get_dataframe()
    try:
        return jsonify(analyzer.compute_protocol_distribution(df))
    except Exception as exc:
        logger.exception("traffic_protocols error")
        return _error(f"Protocol analysis failed: {exc}")


# ----------------------------------------------------------------
# Top source IPs
# ----------------------------------------------------------------

@traffic_bp.route("/top-sources", methods=["GET"])
def traffic_top_sources():
    """Return the top N source IPs by traffic volume."""
    err = _require_data()
    if err:
        return err

    n = _parse_n(request.args.get("n", "10"))
    df = store.get_dataframe()
    try:
        return jsonify(analyzer.compute_top_sources(df, n=n))
    except Exception as exc:
        logger.exception("traffic_top_sources error")
        return _error(f"Top sources analysis failed: {exc}")


# ----------------------------------------------------------------
# Top destination IPs
# ----------------------------------------------------------------

@traffic_bp.route("/top-destinations", methods=["GET"])
def traffic_top_destinations():
    """Return the top N destination IPs by traffic volume."""
    err = _require_data()
    if err:
        return err

    n = _parse_n(request.args.get("n", "10"))
    df = store.get_dataframe()
    try:
        return jsonify(analyzer.compute_top_destinations(df, n=n))
    except Exception as exc:
        logger.exception("traffic_top_destinations error")
        return _error(f"Top destinations analysis failed: {exc}")


# ----------------------------------------------------------------
# Timeline
# ----------------------------------------------------------------

@traffic_bp.route("/timeline", methods=["GET"])
def traffic_timeline():
    """Return traffic volume aggregated over time buckets."""
    err = _require_data()
    if err:
        return err

    freq = request.args.get("freq", "1min")
    df = store.get_dataframe()
    try:
        return jsonify(analyzer.compute_timeline(df, freq=freq))
    except Exception as exc:
        logger.exception("traffic_timeline error")
        return _error(f"Timeline analysis failed: {exc}")


# ----------------------------------------------------------------
# Connection statistics
# ----------------------------------------------------------------

@traffic_bp.route("/connections", methods=["GET"])
def traffic_connections():
    """Return per-connection aggregate statistics."""
    err = _require_data()
    if err:
        return err

    df = store.get_dataframe()
    try:
        return jsonify(analyzer.compute_connection_stats(df))
    except Exception as exc:
        logger.exception("traffic_connections error")
        return _error(f"Connection stats failed: {exc}")


# ----------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------

def _parse_n(value: str, default: int = 10, max_n: int = 50) -> int:
    try:
        n = int(value)
        return max(1, min(n, max_n))
    except (TypeError, ValueError):
        return default


def _supabase_status() -> bool:
    try:
        from utils.supabase_client import get_supabase_config
        return get_supabase_config()["ready"]
    except Exception:
        return False


def _live_capture_status() -> bool:
    try:
        from backend.collectors.live_collector import LIVE_CAPTURE_AVAILABLE
        return LIVE_CAPTURE_AVAILABLE
    except Exception:
        return False
