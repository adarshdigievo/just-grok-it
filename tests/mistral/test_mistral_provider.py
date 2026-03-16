"""Tests for the Mistral provider."""

import logging

import pytest

# Check if mistralai is installed
try:
    import mistralai

    MISTRAL_INSTALLED = True
except ImportError:
    MISTRAL_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not MISTRAL_INSTALLED, reason="mistralai package not installed"
)


def test_mistral_provider_is_installed():
    """Test that Mistral provider detects installation correctly."""
    from just_grok_it.providers import MistralProvider

    provider = MistralProvider()
    assert provider.is_installed() is True


def test_mistral_provider_patch_and_unpatch():
    """Test patching and unpatching the Mistral provider."""
    from just_grok_it.providers import MistralProvider

    provider = MistralProvider()

    # Initially not patched
    assert provider.is_patched() is False

    # Patch
    provider.patch(default_model="grok-4")
    assert provider.is_patched() is True

    # Unpatch
    provider.unpatch()
    assert provider.is_patched() is False


def test_mistral_to_openai_conversion():
    """Test conversion of Mistral message format to OpenAI format."""
    from just_grok_it._converters.mistral_converter import convert_mistral_to_openai

    messages = [
        {"role": "user", "content": "Hello, Mistral!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ]

    result = convert_mistral_to_openai(
        messages=messages,
        model="grok-4",
        max_tokens=1024,
        temperature=0.7,
    )

    assert result["model"] == "grok-4"
    assert result["max_tokens"] == 1024
    assert result["temperature"] == 0.7
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Hello, Mistral!"


def test_mistral_content_list_conversion():
    """Test conversion of Mistral content lists to text."""
    from just_grok_it._converters.mistral_converter import convert_mistral_to_openai

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "text", "text": "How are you?"},
            ],
        },
    ]

    result = convert_mistral_to_openai(
        messages=messages,
        model="grok-4",
    )

    # Text parts should be joined
    assert "Hello!" in result["messages"][0]["content"]
    assert "How are you?" in result["messages"][0]["content"]


def test_openai_to_mistral_response_conversion():
    """Test conversion of OpenAI response to Mistral format."""
    from unittest.mock import MagicMock

    from just_grok_it._converters.mistral_converter import (
        MistralChatCompletionResponse,
        convert_openai_to_mistral,
    )

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! I'm doing great."
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    result = convert_openai_to_mistral(mock_response, "mistral-large")

    assert isinstance(result, MistralChatCompletionResponse)
    assert result.object == "chat.completion"
    assert result.model == "mistral-large"
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello! I'm doing great."
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30


def test_using_all_patches_mistral():
    """Test that just_grok_it.all() patches Mistral."""
    import just_grok_it

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert results.get("mistral") is True
    assert just_grok_it.is_patched("mistral") is True

    just_grok_it.unpatch_all()


def test_mistral_patching_logging(caplog):
    """Test that patching Mistral logs correctly."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged about patching
        assert any(
            "mistral" in record.message.lower() and "patched" in record.message.lower()
            for record in caplog.records
        )

    just_grok_it.unpatch_all()
