"""
Integration tests for Anthropic SDK redirection.
"""

import httpx
import pytest

# Check if anthropic is installed
try:
    import anthropic

    ANTHROPIC_INSTALLED = True
except ImportError:
    ANTHROPIC_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not ANTHROPIC_INSTALLED, reason="anthropic package not installed"
)


def test_anthropic_redirects_to_mock_server(
    mock_server_url: str, mock_xai_base_url: str
):
    """Test that Anthropic SDK calls are redirected to mock xAI server."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    client = Anthropic(api_key="test-key")

    response = client.messages.create(
        model="claude-3-sonnet",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude!"}],
    )

    # Response should be converted from OpenAI format to Anthropic format
    assert response.content is not None
    assert len(response.content) > 0
    assert hasattr(response.content[0], "text")

    # Verify request was received by mock server in OpenAI format
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    last_request = requests[-1]
    # Should be converted to OpenAI format
    assert "messages" in last_request["body"]
    assert last_request["body"]["messages"][-1]["content"] == "Hello, Claude!"

    just_grok_it.unpatch_all()


def test_anthropic_system_prompt_converted(
    mock_server_url: str, mock_xai_base_url: str
):
    """Test that Anthropic system prompt is converted to OpenAI format."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    client = Anthropic(api_key="test-key")

    response = client.messages.create(
        model="claude-3-sonnet",
        max_tokens=100,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hi!"}],
    )

    # Verify system message was added
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    messages = requests[-1]["body"]["messages"]

    # First message should be system
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    just_grok_it.unpatch_all()


def test_anthropic_uses_default_model(mock_server_url: str, mock_xai_base_url: str):
    """Test that default xAI model is used."""
    import just_grok_it
    from just_grok_it import DEFAULT_XAI_MODEL

    just_grok_it.unpatch_all()
    just_grok_it.all()  # Uses default model

    from anthropic import Anthropic

    client = Anthropic(api_key="test-key")

    response = client.messages.create(
        model="claude-3-sonnet",
        max_tokens=100,
        messages=[{"role": "user", "content": "Test"}],
    )

    # Verify default model was used
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    assert requests[-1]["body"]["model"] == DEFAULT_XAI_MODEL

    just_grok_it.unpatch_all()


def test_anthropic_response_format(mock_server_url: str, mock_xai_base_url: str):
    """Test that response is in proper Anthropic format."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    client = Anthropic(api_key="test-key")

    response = client.messages.create(
        model="claude-3-sonnet",
        max_tokens=100,
        messages=[{"role": "user", "content": "Test"}],
    )

    # Should have Anthropic response structure
    assert hasattr(response, "id")
    assert hasattr(response, "type")
    assert hasattr(response, "role")
    assert hasattr(response, "content")
    assert hasattr(response, "stop_reason")
    assert hasattr(response, "usage")

    assert response.type == "message"
    assert response.role == "assistant"

    just_grok_it.unpatch_all()
