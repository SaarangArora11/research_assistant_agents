"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_settings_endpoints():
    # GET settings
    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()
    assert "settings" in data
    assert "hardware" in data
    assert "suggested_directories" in data

    # PUT settings update
    update_res = client.put(
        "/api/settings",
        json={"max_researcher_concurrency": 3, "checkpoint_mode": "plan_only"},
    )
    assert update_res.status_code == 200
    updated_data = update_res.json()
    assert updated_data["settings"]["max_researcher_concurrency"] == 3
    assert updated_data["settings"]["checkpoint_mode"] == "plan_only"


def test_fs_list_endpoint():
    res = client.get("/api/fs/list")
    assert res.status_code == 200
    data = res.json()
    assert "current" in data
    assert "folders" in data


def test_runs_list_endpoint():
    res = client.get("/api/runs")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
