"""
Test: Congestion Prediction API — NetSentinel AI Phase 2 Stage 2
================================================================
Integration tests for the /api/congestion/* endpoints and regression checks
for existing Phase 1 endpoints.

All data is strictly SIMULATED on a 100 Mbps bottleneck link.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app import app
import backend.services.congestion_service as cs_module
from backend.services.congestion_service import CongestionService


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_status_endpoint_success(client):
    """GET /api/congestion/status returns 200, ready status, and metadata."""
    resp = client.get("/api/congestion/status")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["status"] == "ready"
    assert data["simulated"] is True
    assert data["data_source"] == "SIMULATED_BOTTLENECK_LINK"
    assert data["horizon_minutes"] == 5
    assert data["threshold"] == 0.80
    assert data["capacity_mbps"] == 100.0
    assert "Ridge" in data["model_name"]
    assert data["features_count"] == 12


def test_forecast_endpoint_default(client):
    """GET /api/congestion/forecast returns 200 with forecast, baselines, and observations."""
    resp = client.get("/api/congestion/forecast")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["simulated"] is True
    assert data["data_source"] == "SIMULATED_BOTTLENECK_LINK"
    assert data["horizon_minutes"] == 5
    assert data["threshold"] == 0.80
    assert data["capacity_mbps"] == 100.0
    assert "as_of" in data
    assert "as_of_index" in data
    assert data["as_of_index"] >= cs_module.MIN_LOOKBACK_MINUTES

    forecast = data["forecast"]
    assert "predicted_utilization" in forecast
    assert 0.0 <= forecast["predicted_utilization"] <= 1.5
    assert "predicted_utilization_pct" in forecast
    assert "persistence_baseline" in forecast
    assert "moving_average_5m" in forecast
    assert "alert" in forecast
    assert forecast["alert"] == (forecast["predicted_utilization"] >= 0.80)

    recent = data["recent_observed"]
    assert len(recent) == 30
    for obs in recent:
        assert "timestamp" in obs
        assert "utilization" in obs
        assert "packets_per_sec" in obs
        assert "bytes_per_sec" in obs


def test_forecast_endpoint_custom_as_of_index(client):
    """GET /api/congestion/forecast?as_of=9200 accepts custom index."""
    resp = client.get("/api/congestion/forecast?as_of=9200")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["as_of_index"] == 9200
    assert len(data["recent_observed"]) == 30


def test_forecast_endpoint_custom_as_of_timestamp(client):
    """GET /api/congestion/forecast?as_of=... accepts valid timestamp in dataset."""
    target_ts = "2024-11-07 14:00:00"
    resp = client.get(f"/api/congestion/forecast?as_of={target_ts}")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["as_of"] == target_ts
    assert data["forecast"]["target_timestamp"] == "2024-11-07 14:05:00"


def test_forecast_bad_params_returns_400(client):
    """GET /api/congestion/forecast returns 400 for invalid indices or timestamps."""
    # Negative index
    resp1 = client.get("/api/congestion/forecast?as_of=-5")
    assert resp1.status_code == 400
    d1 = resp1.get_json()
    assert "error" in d1
    assert d1["simulated"] is True

    # Out of bounds index
    resp2 = client.get("/api/congestion/forecast?as_of=99999999")
    assert resp2.status_code == 400
    d2 = resp2.get_json()
    assert "error" in d2
    assert d2["simulated"] is True

    # Unknown timestamp string
    resp3 = client.get("/api/congestion/forecast?as_of=1999-01-01 00:00:00")
    assert resp3.status_code == 400
    d3 = resp3.get_json()
    assert "error" in d3
    assert d3["simulated"] is True

    # Index with insufficient lookback history (<30)
    resp4 = client.get("/api/congestion/forecast?as_of=10")
    assert resp4.status_code == 400
    d4 = resp4.get_json()
    assert "error" in d4
    assert d4["simulated"] is True


def test_missing_model_returns_unavailable(client):
    """When the model artifact is missing, service returns model_unavailable status without crashing."""
    dummy_service = CongestionService(model_path=Path("models/nonexistent_model.joblib"))
    assert not dummy_service.is_model_available

    status = dummy_service.get_status()
    assert status["status"] == "model_unavailable"
    assert status["simulated"] is True
    assert status["model_name"] is None

    # Test route level handling via mock
    with patch("backend.routes.congestion_routes.get_congestion_service", return_value=dummy_service):
        resp_status = client.get("/api/congestion/status")
        assert resp_status.status_code == 200
        assert resp_status.get_json()["status"] == "model_unavailable"

        resp_forecast = client.get("/api/congestion/forecast")
        assert resp_forecast.status_code == 503
        data = resp_forecast.get_json()
        assert data["status"] == "model_unavailable"
        assert data["simulated"] is True
        assert data["data_source"] == "SIMULATED_BOTTLENECK_LINK"


def test_simulated_flag_present_on_all_responses(client):
    """Every congestion endpoint response must include simulated: True."""
    # 200 status
    r_status = client.get("/api/congestion/status")
    assert r_status.get_json()["simulated"] is True

    # 200 forecast
    r_fc = client.get("/api/congestion/forecast")
    assert r_fc.get_json()["simulated"] is True

    # 400 bad param
    r_bad = client.get("/api/congestion/forecast?as_of=invalid")
    assert r_bad.get_json()["simulated"] is True


def test_existing_phase1_endpoints_unaffected(client):
    """Regression check: all 8 Phase 1 endpoints and dashboard route remain functional."""
    # Ensure sample traffic is loaded for stats/protocols endpoints
    load_resp = client.get("/api/traffic/load")
    assert load_resp.status_code == 200

    # 1. /api/health
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["project"] == "NetSentinel AI"

    # 2. /api/traffic/status
    resp = client.get("/api/traffic/status")
    assert resp.status_code == 200

    # 3. /api/traffic/stats
    resp = client.get("/api/traffic/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "overview" in data

    # 4. /api/traffic/protocols
    resp = client.get("/api/traffic/protocols")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)

    # 5. /api/traffic/top-sources
    resp = client.get("/api/traffic/top-sources")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)

    # 6. /api/traffic/top-destinations
    resp = client.get("/api/traffic/top-destinations")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)

    # 7. /api/traffic/timeline
    resp = client.get("/api/traffic/timeline?freq=1min")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)

    # 8. /api/traffic/connections
    resp = client.get("/api/traffic/connections")
    assert resp.status_code == 200

    # Root dashboard route
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "NetSentinel AI" in html
    assert "Congestion Forecast" in html
    assert "SIMULATED DATA" in html
