import pytest
from unittest.mock import MagicMock, patch
from app.core.cache import CacheManager


@pytest.fixture
def mock_cache():
    """Create a CacheManager with mocked Redis client."""
    with patch("app.core.cache.Redis") as mock_redis_class, \
         patch("app.core.cache.get_settings") as mock_settings:

        # Setup settings
        mock_settings.return_value.redis_url = "rediss://fake:token@fake.upstash.io:6379"
        mock_settings.return_value.cache_ttl = 3600
        mock_settings.return_value.cache_enabled = True

        # Setup Redis client
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_class.from_url.return_value = mock_client

        cache = CacheManager()
        cache.client = mock_client
        cache.enabled = True

        return cache


def make_result() -> dict:
    """Helper to create a dummy query result."""
    return {
        "query": "What is GPT-2?",
        "rewritten_query": "GPT-2 transformer language model",
        "answer": "GPT-2 is a large language model [Source 1].",
        "sources": [{"file": "gpt2.pdf", "page": 1, "score": 0.9, "rerank_score": 0.95}],
        "model": "llama-3.3-70b-versatile",
        "used_fallback": False,
        "retry_count": 0,
        "retrieval_method": "hybrid",
        "cache_hit": False,
    }


def test_cache_get_hit(mock_cache):
    """Cache should return result on hit."""
    import json
    result = make_result()
    mock_cache.client.get.return_value = json.dumps(result)

    cached = mock_cache.get("What is GPT-2?")
    assert cached is not None
    assert cached["answer"] == result["answer"]


def test_cache_get_miss(mock_cache):
    """Cache should return None on miss."""
    mock_cache.client.get.return_value = None
    cached = mock_cache.get("What is GPT-2?")
    assert cached is None


def test_cache_set_stores_result(mock_cache):
    """Cache set should call Redis setex."""
    result = make_result()
    mock_cache.set("What is GPT-2?", result)
    assert mock_cache.client.setex.called


def test_cache_set_excludes_chunk_texts(mock_cache):
    """Cache set should exclude retrieved_chunks_text to save space."""
    import json
    result = make_result()
    result["retrieved_chunks_text"] = [{"text": "large chunk text", "page": 1}]

    mock_cache.set("What is GPT-2?", result)

    # Get what was stored
    call_args = mock_cache.client.setex.call_args
    stored_json = call_args[0][2]
    stored = json.loads(stored_json)

    assert "retrieved_chunks_text" not in stored


def test_cache_key_is_normalized(mock_cache):
    """Same query with different casing should produce same cache key."""
    key1 = mock_cache._make_key("What is GPT-2?")
    key2 = mock_cache._make_key("what is gpt-2?")
    key3 = mock_cache._make_key("  What is GPT-2?  ")
    assert key1 == key2 == key3


def test_cache_invalidate(mock_cache):
    """Cache invalidate should call Redis delete."""
    mock_cache.invalidate("What is GPT-2?")
    assert mock_cache.client.delete.called


def test_cache_flush(mock_cache):
    """Cache flush should delete all neurorag keys."""
    mock_cache.client.keys.return_value = ["neurorag:query:abc", "neurorag:query:def"]
    mock_cache.client.delete.return_value = 2

    deleted = mock_cache.flush()
    assert deleted == 2


def test_cache_get_fails_gracefully(mock_cache):
    """Cache get should return None on Redis error."""
    mock_cache.client.get.side_effect = Exception("Connection error")
    result = mock_cache.get("What is GPT-2?")
    assert result is None


def test_cache_set_fails_gracefully(mock_cache):
    """Cache set should not raise on Redis error."""
    mock_cache.client.setex.side_effect = Exception("Connection error")
    result = make_result()
    mock_cache.set("What is GPT-2?", result)  # Should not raise


def test_cache_disabled_returns_none(mock_cache):
    """Disabled cache should always return None."""
    mock_cache.enabled = False
    result = mock_cache.get("What is GPT-2?")
    assert result is None
    mock_cache.client.get.assert_not_called()


def test_cache_stats(mock_cache):
    """Cache stats should return correct structure."""
    mock_cache.client.keys.return_value = ["key1", "key2", "key3"]
    stats = mock_cache.stats()

    assert "enabled" in stats
    assert "connected" in stats
    assert "cached_queries" in stats
    assert stats["cached_queries"] == 3