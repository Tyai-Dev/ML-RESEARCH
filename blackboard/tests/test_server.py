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


def test_workspace_lists_tree(client):
    tree = client.get("/api/workspace").json()["tree"]
    studio = next(n for n in tree if n["name"] == "studio")
    draft = studio["children"][0]
    assert draft["dir"] and draft["children"][0]["name"] == "main.tex"


def test_new_board_creates_pair(client):
    created = client.post("/api/board", json={"name": "My First Board"}).json()
    assert created["created"] == "my-first-board"
    nb = client.get("/api/notebook", params={"path": created["notebook"]}).json()
    assert nb["cells"][0]["source"] == "# My First Board"
    tex = client.get("/api/file", params={"path": created["tex"]}).json()["content"]
    assert "\\title{My First Board}" in tex
    assert "papers-common" not in tex  # standalone — no MLR coupling
    assert client.post("/api/board", json={"name": "my first board"}).status_code == 409


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
