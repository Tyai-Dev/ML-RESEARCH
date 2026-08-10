import pytest
from fastapi.testclient import TestClient

import blackboard.server as srv


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    return TestClient(srv.app)


def test_chat_streams_chunks(client, monkeypatch):
    import blackboard.chat as chat_mod

    def fake_stream(messages, context=None, provider="claude", root=None):
        assert messages[-1]["content"] == "hello"
        assert context == "some board"
        assert provider == "gemini"
        assert root is not None  # server passes the workspace for tools
        yield "chalk "
        yield "reply"

    monkeypatch.setattr(chat_mod, "stream_chat", fake_stream)
    response = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "context": "some board",
            "provider": "gemini",
        },
    )
    assert response.status_code == 200
    assert response.text == "chalk reply"


def test_chat_config_error_becomes_503(client, monkeypatch):
    import blackboard.chat as chat_mod

    def fake_stream(messages, context=None, provider="claude", root=None):
        raise chat_mod.ChatError("No claude API credentials.")

    monkeypatch.setattr(chat_mod, "stream_chat", fake_stream)
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 503
    assert "credentials" in response.json()["detail"]


def test_chat_unknown_provider_becomes_503(client):
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "provider": "grok"},
    )
    assert response.status_code == 503
    assert "unknown provider" in response.json()["detail"]


def test_chat_rejects_empty_conversation(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 400


def test_missing_key_message_names_the_env_file(monkeypatch):
    import blackboard.chat as chat_mod

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(chat_mod.ChatError, match="OPENAI_API_KEY"):
        chat_mod.stream_chat([{"role": "user", "content": "hi"}], provider="openai")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(chat_mod.ChatError, match="GEMINI_API_KEY"):
        chat_mod.stream_chat([{"role": "user", "content": "hi"}], provider="gemini")


def test_fix_endpoint(client):
    response = client.post("/api/fix", json={"code": "x=1"})
    assert response.status_code == 200
    assert response.json()["code"] == "x = 1"
