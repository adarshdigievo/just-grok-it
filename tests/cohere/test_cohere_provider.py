"""Tests for the Cohere provider."""

import logging

import pytest

# Check if cohere is installed
try:
    import cohere

    COHERE_INSTALLED = True
except ImportError:
    COHERE_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not COHERE_INSTALLED, reason="cohere package not installed"
)


def test_cohere_provider_is_installed():
    """Test that Cohere provider detects installation correctly."""
    from just_grok_it.providers import CohereProvider

    provider = CohereProvider()
    assert provider.is_installed() is True


def test_cohere_provider_patch_and_unpatch():
    """Test patching and unpatching the Cohere provider."""
    from just_grok_it.providers import CohereProvider

    provider = CohereProvider()

    # Initially not patched
    assert provider.is_patched() is False

    # Patch
    provider.patch(default_model="grok-4")
    assert provider.is_patched() is True

    # Unpatch
    provider.unpatch()
    assert provider.is_patched() is False


def test_cohere_to_openai_conversion():
    """Test conversion of Cohere message format to OpenAI format."""
    from just_grok_it._converters.cohere_converter import convert_cohere_to_openai

    chat_history = [
        {"role": "USER", "message": "Hello!"},
        {"role": "CHATBOT", "message": "Hi there!"},
    ]

    result = convert_cohere_to_openai(
        message="Tell me a joke",
        model="grok-4",
        chat_history=chat_history,
        preamble="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=1024,
    )

    assert result["model"] == "grok-4"
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 1024

    # Check messages - should have system message first
    assert len(result["messages"]) == 4
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "You are a helpful assistant."
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][1]["content"] == "Hello!"
    assert result["messages"][2]["role"] == "assistant"
    assert result["messages"][2]["content"] == "Hi there!"
    assert result["messages"][3]["role"] == "user"
    assert result["messages"][3]["content"] == "Tell me a joke"


def test_cohere_simple_message_conversion():
    """Test conversion of simple Cohere message."""
    from just_grok_it._converters.cohere_converter import convert_cohere_to_openai

    result = convert_cohere_to_openai(
        message="Hello, Cohere!",
        model="grok-4",
    )

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Hello, Cohere!"


def test_openai_to_cohere_response_conversion():
    """Test conversion of OpenAI response to Cohere format."""
    from unittest.mock import MagicMock

    from just_grok_it._converters.cohere_converter import (
        CohereChatResponse,
        convert_openai_to_cohere,
    )

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! I'm doing great."
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    result = convert_openai_to_cohere(mock_response, "command")

    assert isinstance(result, CohereChatResponse)
    assert result.text == "Hello! I'm doing great."
    assert result.finish_reason == "COMPLETE"
    assert result.meta.tokens["input_tokens"] == 10
    assert result.meta.tokens["output_tokens"] == 20


def test_cohere_finish_reason_conversion():
    """Test that finish_reason is correctly mapped."""
    from unittest.mock import MagicMock

    from just_grok_it._converters.cohere_converter import convert_openai_to_cohere

    # Test length finish reason
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello..."
    mock_response.choices[0].finish_reason = "length"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    result = convert_openai_to_cohere(mock_response, "command")
    assert result.finish_reason == "MAX_TOKENS"


def test_using_all_patches_cohere():
    """Test that just_grok_it.all() patches Cohere."""
    import just_grok_it

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert results.get("cohere") is True
    assert just_grok_it.is_patched("cohere") is True

    just_grok_it.unpatch_all()


def test_cohere_patching_logging(caplog):
    """Test that patching Cohere logs correctly."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged about patching
        assert any(
            "cohere" in record.message.lower() and "patched" in record.message.lower()
            for record in caplog.records
        )

    just_grok_it.unpatch_all()
