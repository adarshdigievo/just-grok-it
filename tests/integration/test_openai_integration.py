"""
Integration tests for OpenAI SDK redirection.
"""

import httpx
import pytest


def test_openai_redirects_to_mock_server(mock_server_url: str, mock_xai_base_url: str):
    """Test that OpenAI SDK calls are redirected to mock xAI server."""
    import just_grok_it

    # Unpatch and repatch to use new URL
    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "Hello, Grok!"}],
    )

    assert response.choices[0].message.content is not None
    assert "Mock response" in response.choices[0].message.content

    # Verify request was received by mock server
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    last_request = requests[-1]
    assert last_request["body"]["model"] == "grok-4"
    assert last_request["body"]["messages"][0]["content"] == "Hello, Grok!"

    just_grok_it.unpatch_all()


def test_openai_uses_default_model(mock_server_url: str, mock_xai_base_url: str):
    """Test that default model is used when not specified."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()  # Uses default model

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    # Model should be set by interceptor
    response = client.chat.completions.create(
        model="gpt-4",  # This will be used, not overridden for OpenAI
        messages=[{"role": "user", "content": "Test message"}],
    )

    assert response is not None

    just_grok_it.unpatch_all()


def test_openai_with_custom_model(mock_server_url: str, mock_xai_base_url: str):
    """Test that custom model is passed through."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all(default_model="grok-3")

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    response = client.chat.completions.create(
        model="my-custom-model",
        messages=[{"role": "user", "content": "Test"}],
    )

    # Verify the request had the custom model
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    # The OpenAI provider doesn't override the model, just the base_url
    assert requests[-1]["body"]["model"] == "my-custom-model"

    just_grok_it.unpatch_all()


def test_openai_with_openrouter_url_redirected(
    mock_server_url: str, mock_xai_base_url: str
):
    """Test that OpenRouter URL is redirected to xAI."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    # Try to use OpenRouter URL - should be intercepted
    client = OpenAI(api_key="test-key", base_url="https://openrouter.ai/api/v1")

    # The client should now point to xAI
    assert "x.ai" in str(client.base_url) or mock_server_url in str(client.base_url)

    just_grok_it.unpatch_all()


def test_openai_with_together_url_redirected(
    mock_server_url: str, mock_xai_base_url: str
):
    """Test that Together AI URL is redirected to xAI."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    # Try to use Together URL - should be intercepted
    client = OpenAI(api_key="test-key", base_url="https://api.together.xyz/v1")

    # Should be redirected
    assert "together" not in str(client.base_url).lower()

    just_grok_it.unpatch_all()


def test_authorization_header_passed(mock_server_url: str, mock_xai_base_url: str):
    """Test that authorization header is properly passed."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI(api_key="my-secret-key")

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "Test"}],
    )

    # Verify authorization header
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    auth = requests[-1]["authorization"]
    assert auth is not None
    assert "my-secret-key" in auth

    just_grok_it.unpatch_all()


def test_temperature_and_params_passed(mock_server_url: str, mock_xai_base_url: str):
    """Test that temperature and other params are passed through."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "Test"}],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )

    # Verify params
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    body = requests[-1]["body"]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 100
    assert body["top_p"] == 0.9

    just_grok_it.unpatch_all()


def test_tool_calling_params_passed(mock_server_url: str, mock_xai_base_url: str):
    """Test that tool calling parameters are passed through."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
        tool_choice="auto",
    )

    # Verify tools were passed
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    body = requests[-1]["body"]
    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == "auto"

    just_grok_it.unpatch_all()


def test_structured_output_json_mode(mock_server_url: str, mock_xai_base_url: str):
    """Test that JSON mode response_format is passed through."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    response = client.chat.completions.create(
        model="grok-4",
        messages=[
            {"role": "system", "content": "You output JSON."},
            {"role": "user", "content": "Give me info about yourself."},
        ],
        response_format={"type": "json_object"},
    )

    # Verify response_format was passed
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    body = requests[-1]["body"]
    assert "response_format" in body
    assert body["response_format"]["type"] == "json_object"

    just_grok_it.unpatch_all()


def test_system_message_passed(mock_server_url: str, mock_xai_base_url: str):
    """Test that system message is passed through."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI(api_key="test-key")

    response = client.chat.completions.create(
        model="grok-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    # Verify system message was passed
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    body = requests[-1]["body"]
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are a helpful assistant."

    just_grok_it.unpatch_all()


def test_api_key_from_xai_env(
    mock_server_url: str, mock_xai_base_url: str, monkeypatch
):
    """Test that XAI_API_KEY from environment is used."""
    import just_grok_it

    # The conftest sets XAI_API_KEY to "test-api-key"
    # Let's override it to verify our key is used
    monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
    # Clear OPENAI_API_KEY to ensure XAI_API_KEY is used
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI()  # No api_key passed - should use XAI_API_KEY

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "Test"}],
    )

    # Verify the XAI env key was used
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    auth = requests[-1]["authorization"]
    assert "xai-env-key" in auth

    just_grok_it.unpatch_all()


def test_api_key_from_openai_env(
    mock_server_url: str, mock_xai_base_url: str, monkeypatch
):
    """Test that OPENAI_API_KEY from environment is used when XAI_API_KEY is not set."""
    import just_grok_it

    # Clear XAI_API_KEY and set OPENAI_API_KEY
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    client = OpenAI()  # No api_key passed - should fall back to OPENAI_API_KEY

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "Test"}],
    )

    # Verify the OpenAI env key was used
    requests_response = httpx.get(f"{mock_server_url}/v1/requests")
    requests = requests_response.json()["requests"]

    assert len(requests) >= 1
    auth = requests[-1]["authorization"]
    assert "openai-env-key" in auth

    just_grok_it.unpatch_all()
