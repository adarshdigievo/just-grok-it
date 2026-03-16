"""Tests for the Anthropic provider."""

import logging

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


def test_anthropic_provider_is_installed():
    """Test that Anthropic provider detects installation correctly."""
    from just_grok_it.providers import AnthropicProvider

    provider = AnthropicProvider()
    assert provider.is_installed() is True


def test_anthropic_provider_patch_and_unpatch():
    """Test patching and unpatching the Anthropic provider."""
    from just_grok_it.providers import AnthropicProvider

    provider = AnthropicProvider()

    # Initially not patched
    assert provider.is_patched() is False

    # Patch
    provider.patch(default_model="grok-4")
    assert provider.is_patched() is True

    # Unpatch
    provider.unpatch()
    assert provider.is_patched() is False


def test_anthropic_to_openai_conversion():
    """Test conversion of Anthropic message format to OpenAI format."""
    from just_grok_it._converters.anthropic_converter import convert_anthropic_to_openai

    messages = [
        {"role": "user", "content": "Hello, Claude!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ]

    result = convert_anthropic_to_openai(
        messages=messages,
        model="grok-4",
        max_tokens=1024,
        system="You are a helpful assistant.",
        temperature=0.7,
    )

    assert result["model"] == "grok-4"
    assert result["max_tokens"] == 1024
    assert result["temperature"] == 0.7

    # Check messages - should have system message first
    assert len(result["messages"]) == 4
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "You are a helpful assistant."
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][1]["content"] == "Hello, Claude!"


def test_anthropic_content_blocks_conversion():
    """Test conversion of Anthropic content blocks to text."""
    from just_grok_it._converters.anthropic_converter import convert_anthropic_to_openai

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "text", "text": "How are you?"},
            ],
        },
    ]

    result = convert_anthropic_to_openai(
        messages=messages,
        model="grok-4",
        max_tokens=1024,
    )

    # Text blocks should be joined
    assert "Hello!" in result["messages"][0]["content"]
    assert "How are you?" in result["messages"][0]["content"]


def test_openai_to_anthropic_response_conversion():
    """Test conversion of OpenAI response to Anthropic format."""
    from unittest.mock import MagicMock

    from just_grok_it._converters.anthropic_converter import (
        AnthropicMessage,
        convert_openai_to_anthropic,
    )

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! I'm doing great."
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    result = convert_openai_to_anthropic(mock_response, "claude-3-sonnet")

    assert isinstance(result, AnthropicMessage)
    assert result.type == "message"
    assert result.role == "assistant"
    assert result.model == "claude-3-sonnet"
    assert result.stop_reason == "end_turn"
    assert len(result.content) == 1
    assert result.content[0].text == "Hello! I'm doing great."
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20


def test_using_all_patches_anthropic():
    """Test that just_grok_it.all() patches Anthropic."""
    import just_grok_it

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert results.get("anthropic") is True
    assert just_grok_it.is_patched("anthropic") is True

    just_grok_it.unpatch_all()


def test_anthropic_patching_logging(caplog):
    """Test that patching Anthropic logs correctly."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged about patching
        assert any(
            "anthropic" in record.message.lower()
            and "patched" in record.message.lower()
            for record in caplog.records
        )

    just_grok_it.unpatch_all()


def test_anthropic_image_base64_conversion():
    """Test conversion of Anthropic base64 image to OpenAI format."""
    from just_grok_it._converters.anthropic_converter import convert_anthropic_to_openai

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
            ],
        },
    ]

    result = convert_anthropic_to_openai(
        messages=messages,
        model="grok-4",
        max_tokens=1024,
    )

    # Check that image was converted
    assert len(result["messages"]) == 1
    content = result["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2

    # First part should be text
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What's in this image?"

    # Second part should be image_url
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64," in content[1]["image_url"]["url"]


def test_anthropic_image_url_conversion():
    """Test conversion of Anthropic URL image to OpenAI format."""
    from just_grok_it._converters.anthropic_converter import convert_anthropic_to_openai

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image.png"},
                },
            ],
        },
    ]

    result = convert_anthropic_to_openai(
        messages=messages,
        model="grok-4",
        max_tokens=1024,
    )

    # Check that image was converted
    content = result["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "https://example.com/image.png"
