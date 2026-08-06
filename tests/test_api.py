from __future__ import annotations

from fastapi.testclient import TestClient

from hybrid_rag.api.app import create_app
from hybrid_rag.container import Components


def client_for(components: Components) -> TestClient:
    return TestClient(create_app(component_factory=lambda: components))


def test_health_and_empty_document_list(components: Components) -> None:
    with client_for(components) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["indexed_chunks"] == 0
        assert client.get("/v1/documents").json() == []


def test_ingest_and_ask_api(components: Components) -> None:
    with client_for(components) as client:
        response = client.post(
            "/v1/ingest?strategy=recursive",
            files={
                "file": (
                    "runbook.md",
                    b"# Alerts\n\nOBS-1042 means expired token.",
                    "text/markdown",
                )
            },
        )
        assert response.status_code == 201
        answer = client.post("/v1/ask", json={"question": "What does OBS-1042 mean?"})
        assert answer.status_code == 200
        assert answer.json()["status"] == "answered"
        assert answer.headers["x-request-id"]


def test_rejects_bad_upload_and_invalid_question(components: Components) -> None:
    with client_for(components) as client:
        upload = client.post("/v1/ingest", files={"file": ("data.csv", b"a,b", "text/csv")})
        assert upload.status_code == 415
        assert client.post("/v1/ask", json={"question": " "}).status_code == 422
