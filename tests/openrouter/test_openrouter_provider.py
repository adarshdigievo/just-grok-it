"""Tests for the OpenRouter provider."""

import logging

import pytest

# Check if openrouter is installed
try:
    import openrouter

    OPENROUTER_INSTALLED = True
except ImportError:
    OPENROUTER_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not OPENROUTER_INSTALLED, reason="openrouter package not installed"
)


def test_openrouter_provider_is_installed():
    """Test that OpenRouter provider detects installation correctly."""
    from just_grok_it.providers import OpenRouterProvider

    provider = OpenRouterProvider()
    assert provider.is_installed() is True


def test_openrouter_provider_patch_and_unpatch():
    """Test patching and unpatching the OpenRouter provider."""
    from just_grok_it.providers import OpenRouterProvider

    provider = OpenRouterProvider()

    # Initially not patched
    assert provider.is_patched() is False

    # Patch
    provider.patch(default_model="grok-4")
    assert provider.is_patched() is True

    # Unpatch
    provider.unpatch()
    assert provider.is_patched() is False


def test_openrouter_message_conversion():
    """Test conversion of OpenRouter messages to OpenAI format."""
    from just_grok_it.providers import OpenRouterProvider

    provider = OpenRouterProvider()

    # Test dict-based messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = provider._convert_messages(messages)

    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helpful assistant."
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Hello!"
    assert result[2]["role"] == "assistant"
    assert result[2]["content"] == "Hi there!"


def test_openrouter_developer_role_conversion():
    """Test that developer role is converted to system."""
    from just_grok_it.providers import OpenRouterProvider

    provider = OpenRouterProvider()

    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write code."},
    ]

    result = provider._convert_messages(messages)

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a coding assistant."


def test_openrouter_response_conversion():
    """Test conversion of OpenAI response to OpenRouter format."""
    from unittest.mock import MagicMock

    from just_grok_it.providers import OpenRouterProvider

    provider = OpenRouterProvider()

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "Hello! I'm Grok."
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].logprobs = None
    mock_response.created = 1234567890
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    result = provider._convert_response(mock_response, "anthropic/claude-3")

    assert result.id == "chatcmpl-123"
    assert result.model == "anthropic/claude-3"
    assert result.object == "chat.completion"
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello! I'm Grok."
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30


def test_using_all_patches_openrouter():
    """Test that just_grok_it.all() patches OpenRouter."""
    import just_grok_it

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert results.get("openrouter") is True
    assert just_grok_it.is_patched("openrouter") is True

    just_grok_it.unpatch_all()


def test_openrouter_patching_logging(caplog):
    """Test that patching OpenRouter logs correctly."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged about patching
        assert any(
            "openrouter" in record.message.lower()
            and "patched" in record.message.lower()
            for record in caplog.records
        )

    just_grok_it.unpatch_all()


def test_openrouter_in_patched_providers_log(caplog):
    """Test that OpenRouter appears in the patched providers log."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged the patched providers
        assert any(
            "Patched providers:" in record.message and "openrouter" in record.message
            for record in caplog.records
        )

    just_grok_it.unpatch_all()
