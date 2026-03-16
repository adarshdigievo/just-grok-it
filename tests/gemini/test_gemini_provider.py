"""Tests for the Gemini provider."""

import logging

import pytest

# Check if google-generativeai is installed
try:
    import google.generativeai

    GEMINI_INSTALLED = True
except ImportError:
    GEMINI_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not GEMINI_INSTALLED, reason="google-generativeai package not installed"
)


def test_gemini_provider_is_installed():
    """Test that Gemini provider detects installation correctly."""
    from just_grok_it.providers import GeminiProvider

    provider = GeminiProvider()
    assert provider.is_installed() is True


def test_gemini_provider_patch_and_unpatch():
    """Test patching and unpatching the Gemini provider."""
    from just_grok_it.providers import GeminiProvider

    provider = GeminiProvider()

    # Initially not patched
    assert provider.is_patched() is False

    # Patch
    provider.patch(default_model="grok-4")
    assert provider.is_patched() is True

    # Unpatch
    provider.unpatch()
    assert provider.is_patched() is False


def test_gemini_string_content_conversion():
    """Test conversion of simple string content to OpenAI format."""
    from just_grok_it._converters.gemini_converter import convert_gemini_to_openai

    result = convert_gemini_to_openai(
        contents="Hello, Gemini!",
        model="grok-4",
    )

    assert result["model"] == "grok-4"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Hello, Gemini!"


def test_gemini_list_content_conversion():
    """Test conversion of list content to OpenAI format."""
    from just_grok_it._converters.gemini_converter import convert_gemini_to_openai

    contents = [
        {"role": "user", "parts": ["Hello!"]},
        {"role": "model", "parts": ["Hi there!"]},
        {"role": "user", "parts": ["Tell me a joke."]},
    ]

    result = convert_gemini_to_openai(
        contents=contents,
        model="grok-4",
    )

    assert result["model"] == "grok-4"
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"  # "model" -> "assistant"
    assert result["messages"][2]["role"] == "user"


def test_gemini_generation_config_conversion():
    """Test conversion of Gemini generation config to OpenAI parameters."""
    from just_grok_it._converters.gemini_converter import convert_gemini_to_openai

    generation_config = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_output_tokens": 2048,
        "stop_sequences": ["END"],
    }

    result = convert_gemini_to_openai(
        contents="Hello!",
        model="grok-4",
        generation_config=generation_config,
    )

    assert result["temperature"] == 0.8
    assert result["top_p"] == 0.9
    assert result["max_tokens"] == 2048
    assert result["stop"] == ["END"]


def test_openai_to_gemini_response_conversion():
    """Test conversion of OpenAI response to Gemini format."""
    from unittest.mock import MagicMock

    from just_grok_it._converters.gemini_converter import (
        GeminiGenerateContentResponse,
        convert_openai_to_gemini,
    )

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! I'm Grok."
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    result = convert_openai_to_gemini(mock_response, "gemini-pro")

    assert isinstance(result, GeminiGenerateContentResponse)
    assert result.text == "Hello! I'm Grok."
    assert len(result.candidates) == 1
    assert result.candidates[0].content.role == "model"
    assert result.candidates[0].finish_reason == 1  # STOP
    assert result.usage_metadata.prompt_token_count == 10
    assert result.usage_metadata.candidates_token_count == 20
    assert result.usage_metadata.total_token_count == 30


def test_using_all_patches_gemini():
    """Test that just_grok_it.all() patches Gemini."""
    import just_grok_it

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert results.get("gemini") is True
    assert just_grok_it.is_patched("gemini") is True

    just_grok_it.unpatch_all()


def test_gemini_patching_logging(caplog):
    """Test that patching Gemini logs correctly."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged about patching
        assert any(
            "gemini" in record.message.lower() and "patched" in record.message.lower()
            for record in caplog.records
        )

    just_grok_it.unpatch_all()
