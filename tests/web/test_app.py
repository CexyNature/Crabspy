from fastapi.testclient import TestClient

from crabspy_web.app import create_app


def test_api_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_contains_title() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "Crabspy web" in response.text


def test_htmx_fragment_status() -> None:
    client = TestClient(create_app())
    response = client.get("/fragments/status")
    assert response.status_code == 200
    assert "fragment" in response.text.lower()
