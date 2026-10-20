"""
Congestion Prediction Routes — NetSentinel AI Phase 2
=====================================================
Flask Blueprint containing /api/congestion/* endpoints for status and forecasts.

Endpoints:
  GET /api/congestion/status    Model readiness and link capacity status
  GET /api/congestion/forecast  5-minute-ahead forecast and recent 30-min observations
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from backend.services.congestion_service import (
    CONGESTION_THRESHOLD,
    DEFAULT_HORIZON_MINUTES,
    LINK_CAPACITY_MBPS,
    get_congestion_service,
)

logger = logging.getLogger(__name__)

congestion_bp = Blueprint("congestion", __name__)


@congestion_bp.route("/status", methods=["GET"])
def congestion_status():
    """Return model status and configuration."""
    service = get_congestion_service()
    status_data = service.get_status()
    return jsonify(status_data), 200


@congestion_bp.route("/forecast", methods=["GET"])
def congestion_forecast():
    """
    Return 5-minute-ahead congestion forecast and recent 30-minute observed telemetry.

    Query Parameters:
      as_of (optional): Integer row index or timestamp string.
                        Defaults to a representative interval in the Test partition.
    """
    service = get_congestion_service()

    if not service.is_model_available:
        return (
            jsonify(
                {
                    "error": "Congestion prediction model is currently unavailable.",
                    "status": "model_unavailable",
                    "simulated": True,
                    "data_source": "SIMULATED_BOTTLENECK_LINK",
                    "horizon_minutes": DEFAULT_HORIZON_MINUTES,
                    "threshold": CONGESTION_THRESHOLD,
                    "capacity_mbps": LINK_CAPACITY_MBPS,
                }
            ),
            503,
        )

    as_of = request.args.get("as_of", None)

    try:
        forecast_data = service.get_forecast(as_of=as_of)
        return jsonify(forecast_data), 200
    except ValueError as exc:
        logger.warning("Invalid parameter for congestion forecast: %s", exc)
        return (
            jsonify(
                {
                    "error": str(exc),
                    "simulated": True,
                    "data_source": "SIMULATED_BOTTLENECK_LINK",
                    "horizon_minutes": DEFAULT_HORIZON_MINUTES,
                    "threshold": CONGESTION_THRESHOLD,
                    "capacity_mbps": LINK_CAPACITY_MBPS,
                }
            ),
            400,
        )
    except Exception as exc:
        logger.exception("Unexpected error in congestion forecast: %s", exc)
        return (
            jsonify(
                {
                    "error": f"Failed to compute forecast: {exc}",
                    "simulated": True,
                    "data_source": "SIMULATED_BOTTLENECK_LINK",
                    "horizon_minutes": DEFAULT_HORIZON_MINUTES,
                    "threshold": CONGESTION_THRESHOLD,
                    "capacity_mbps": LINK_CAPACITY_MBPS,
                }
            ),
            500,
        )
