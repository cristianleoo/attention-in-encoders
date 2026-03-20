from fastapi.testclient import TestClient
from app.backend.main import app

client = TestClient(app)


def test_attention_endpoint_returns_tokens_and_attention():
    """Attention endpoint returns correctly shaped token and attention data."""
    response = client.post(
        "/api/attention",
        json={"text": "Hello world", "model_name": "bert-base-uncased"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "tokens" in data
    assert "attention" in data
    assert len(data["tokens"]) > 0
    # attention should be [layer][head][seq][seq]
    attention = data["attention"]
    assert len(attention) > 0
    assert len(attention[0]) > 0                  # at least one head
    seq_len = len(data["tokens"])
    assert len(attention[0][0]) == seq_len        # row count
    assert len(attention[0][0][0]) == seq_len     # col count


def test_limits_endpoint_returns_results():
    """Limits endpoint returns one result per requested seq length."""
    response = client.post(
        "/api/limits",
        json={"model_name": "bert-base-uncased", "seq_lengths": [32, 64]},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model"] == "bert-base-uncased"
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert "seq_len" in r
        assert "latency_ms" in r
        assert "status" in r


def test_invalid_model_returns_400():
    """Requesting a non-existent model returns HTTP 400."""
    response = client.post(
        "/api/attention",
        json={"text": "test", "model_name": "not-a-real-model-xyz"},
    )
    assert response.status_code == 400
