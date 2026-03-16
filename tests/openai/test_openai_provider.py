"""Tests for the OpenAI provider."""

import logging

import pytest


def test_openai_provider_is_installed():
    """Test that OpenAI provider detects installation correctly."""
    from just_grok_it.providers import OpenAIProvider

    provider = OpenAIProvider()
    # OpenAI is a dependency, so it should be installed
    assert provider.is_installed() is True


def test_openai_provider_patch_and_unpatch():
    """Test patching and unpatching the OpenAI provider."""
    from just_grok_it.providers import OpenAIProvider

    provider = OpenAIProvider()

    # Initially not patched
    assert provider.is_patched() is False

    # Patch
    provider.patch(default_model="grok-4")
    assert provider.is_patched() is True

    # Unpatch
    provider.unpatch()
    assert provider.is_patched() is False


def test_openai_base_url_redirect():
    """Test that OpenAI client gets redirected to xAI base URL."""
    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    # Ensure clean state
    just_grok_it.unpatch_all()
    just_grok_it.patch_openai(default_model="grok-4")

    from openai import OpenAI

    # Create a client - it should use xAI base URL
    client = OpenAI(api_key="test-key")

    assert str(client.base_url).rstrip("/") == XAI_BASE_URL

    # Cleanup
    just_grok_it.unpatch_all()


def test_async_openai_base_url_redirect():
    """Test that AsyncOpenAI client gets redirected to xAI base URL."""
    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai(default_model="grok-4")

    from openai import AsyncOpenAI

    # Create an async client - it should use xAI base URL
    client = AsyncOpenAI(api_key="test-key")

    assert str(client.base_url).rstrip("/") == XAI_BASE_URL

    # Cleanup
    just_grok_it.unpatch_all()


def test_explicit_base_url_not_overridden():
    """Test that explicitly provided base_url is not overridden."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai(default_model="grok-4")

    from openai import OpenAI

    # Explicitly set a different base_url
    custom_url = "https://custom.api.com/v1"
    client = OpenAI(api_key="test-key", base_url=custom_url)

    assert str(client.base_url).rstrip("/") == custom_url

    # Cleanup
    just_grok_it.unpatch_all()


def test_double_patch():
    """Test that patching twice doesn't cause issues."""
    import just_grok_it

    just_grok_it.unpatch_all()

    just_grok_it.patch_openai(default_model="grok-4")
    assert just_grok_it.is_patched("openai") is True

    # Patching again should be a no-op
    just_grok_it.patch_openai(default_model="grok-3")
    assert just_grok_it.is_patched("openai") is True

    # Cleanup
    just_grok_it.unpatch_all()


def test_client_initialization_logging(caplog):
    """Test that creating a client logs interception."""
    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai(default_model="grok-4")

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        from openai import OpenAI

        client = OpenAI(api_key="test-key")

        # Should have logged about intercepting
        assert any(
            "Intercepted OpenAI initialization" in record.message
            for record in caplog.records
        )
        assert any(
            "redirecting to xAI API" in record.message for record in caplog.records
        )

    just_grok_it.unpatch_all()


def test_using_all_patches_openai():
    """Test that just_grok_it.all() patches OpenAI."""
    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert results.get("openai") is True

    from openai import OpenAI

    client = OpenAI(api_key="test-key")
    assert str(client.base_url).rstrip("/") == XAI_BASE_URL

    just_grok_it.unpatch_all()


def test_api_key_from_openai_env():
    """Test that client works with API key from OPENAI_API_KEY environment."""
    import os

    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai()

    # Set env var
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-env-key"

    try:
        from openai import OpenAI

        client = OpenAI()  # No api_key passed
        assert str(client.base_url).rstrip("/") == XAI_BASE_URL
    finally:
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    just_grok_it.unpatch_all()


def test_api_key_from_xai_env():
    """Test that client works with API key from XAI_API_KEY environment."""
    import os

    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()

    # Clear OPENAI_API_KEY and set XAI_API_KEY
    original_openai_key = os.environ.pop("OPENAI_API_KEY", None)
    original_xai_key = os.environ.get("XAI_API_KEY")
    os.environ["XAI_API_KEY"] = "test-xai-key"

    try:
        just_grok_it.patch_openai()

        from openai import OpenAI

        client = OpenAI()  # No api_key passed - should use XAI_API_KEY
        assert str(client.base_url).rstrip("/") == XAI_BASE_URL
        assert client.api_key == "test-xai-key"
    finally:
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        if original_xai_key:
            os.environ["XAI_API_KEY"] = original_xai_key
        else:
            os.environ.pop("XAI_API_KEY", None)

    just_grok_it.unpatch_all()


def test_api_key_passed_directly():
    """Test that client works with API key passed directly."""
    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai()

    from openai import OpenAI

    client = OpenAI(api_key="directly-passed-key")
    assert str(client.base_url).rstrip("/") == XAI_BASE_URL

    just_grok_it.unpatch_all()


def test_together_url_redirected():
    """Test that Together AI base URL is redirected."""
    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai()

    from openai import OpenAI

    client = OpenAI(api_key="test-key", base_url="https://api.together.xyz/v1")
    assert str(client.base_url).rstrip("/") == XAI_BASE_URL

    just_grok_it.unpatch_all()


def test_groq_url_redirected():
    """Test that Groq base URL is redirected."""
    import just_grok_it
    from just_grok_it._constants import XAI_BASE_URL

    just_grok_it.unpatch_all()
    just_grok_it.patch_openai()

    from openai import OpenAI

    client = OpenAI(api_key="test-key", base_url="https://api.groq.com/openai/v1")
    assert str(client.base_url).rstrip("/") == XAI_BASE_URL

    just_grok_it.unpatch_all()
