from pathlib import Path

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import app


def test_config_loads():
    settings = load_settings(Path(__file__).parents[1])
    assert settings.port == 8765


def test_health_and_routes():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/cameras").status_code == 200
        assert client.get("/events").status_code == 200
