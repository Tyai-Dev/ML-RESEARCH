import pytest
from fastapi.testclient import TestClient

import blackboard.server as srv


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "studio" / "draft").mkdir(parents=True)
    (tmp_path / "studio" / "draft" / "main.tex").write_text(
        "\\documentclass{article}", encoding="utf-8"
    )
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    return TestClient(srv.app)


def test_workspace_lists_sections(client):
    sections = {s["name"]: s for s in client.get("/api/workspace").json()["sections"]}
    assert "studio" in sections
    draft = sections["studio"]["children"][0]
    assert draft["dir"] and draft["children"][0]["name"] == "main.tex"


def test_file_roundtrip(client):
    path = "studio/draft/main.tex"
    assert client.get("/api/file", params={"path": path}).json()["content"].startswith("\\document")
    client.post("/api/file", json={"path": path, "content": "\\documentclass{book}"})
    assert "book" in client.get("/api/file", params={"path": path}).json()["content"]


def test_path_escape_rejected(client):
    response = client.get("/api/file", params={"path": "../outside.txt"})
    assert response.status_code == 400


def test_notebook_roundtrip(client):
    path = "studio/draft/main.ipynb"
    cells = [
        {"type": "markdown", "source": "# hello"},
        {"type": "code", "source": "1 + 1"},
    ]
    assert client.post("/api/notebook", json={"path": path, "cells": cells}).json()["cells"] == 2
    loaded = client.get("/api/notebook", params={"path": path}).json()["cells"]
    assert loaded == cells


def test_execute_endpoint(client):
    result = client.post("/api/execute", json={"code": "40 + 2"}).json()
    assert result["ok"] and result["value"] == "42"
