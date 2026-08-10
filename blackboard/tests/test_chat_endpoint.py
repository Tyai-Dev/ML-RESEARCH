import pytest
from fastapi.testclient import TestClient

import blackboard.server as srv


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    return TestClient(srv.app)


def test_chat_streams_chunks(client, monkeypatch):
    import blackboard.chat as chat_mod

    def fake_stream(messages, context=None):
        assert messages[-1]["content"] == "hello"
        assert context == "some board"
        yield "chalk "
        yield "reply"

    monkeypatch.setattr(chat_mod, "stream_chat", fake_stream)
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "context": "some board"},
    )
    assert response.status_code == 200
    assert response.text == "chalk reply"


def test_chat_config_error_becomes_503(client, monkeypatch):
    import blackboard.chat as chat_mod

    def fake_stream(messages, context=None):
        raise chat_mod.ChatError("No Claude API credentials.")

    monkeypatch.setattr(chat_mod, "stream_chat", fake_stream)
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 503
    assert "credentials" in response.json()["detail"]


def test_chat_rejects_empty_conversation(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 400


def test_fix_endpoint(client):
    response = client.post("/api/fix", json={"code": "x=1"})
    assert response.status_code == 200
    assert response.json()["code"] == "x = 1"
