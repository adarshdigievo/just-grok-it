"""Common tests for the just_grok_it library."""

import logging

import pytest


def test_import():
    """Test that the package can be imported."""
    import just_grok_it

    assert hasattr(just_grok_it, "all")
    assert hasattr(just_grok_it, "patch_all")
    assert hasattr(just_grok_it, "unpatch_all")
    assert hasattr(just_grok_it, "patch_openai")
    assert hasattr(just_grok_it, "unpatch_openai")
    assert hasattr(just_grok_it, "is_patched")
    assert hasattr(just_grok_it, "get_installed_providers")
    assert hasattr(just_grok_it, "get_patched_providers")
    assert hasattr(just_grok_it, "XAI_BASE_URL")
    assert hasattr(just_grok_it, "DEFAULT_XAI_MODEL")
    assert hasattr(just_grok_it, "__version__")


def test_version():
    """Test that version is set correctly."""
    import just_grok_it

    assert just_grok_it.__version__ == "0.1.0"


def test_xai_base_url():
    """Test that the xAI base URL is correct."""
    import just_grok_it

    assert just_grok_it.XAI_BASE_URL == "https://api.x.ai/v1"


def test_default_model():
    """Test that the default model is correct."""
    import just_grok_it

    assert just_grok_it.DEFAULT_XAI_MODEL == "grok-4-1-fast-non-reasoning"


def test_get_installed_providers():
    """Test that we can get installed providers."""
    import just_grok_it

    installed = just_grok_it.get_installed_providers()
    assert isinstance(installed, list)
    # OpenAI should always be installed (it's a dependency)
    assert "openai" in installed


def test_get_patched_providers_initially_empty():
    """Test that no providers are patched initially."""
    import just_grok_it

    # Unpatch all first to ensure clean state
    just_grok_it.unpatch_all()

    patched = just_grok_it.get_patched_providers()
    assert isinstance(patched, list)
    assert len(patched) == 0


def test_is_patched_initially_false():
    """Test that is_patched returns False initially."""
    import just_grok_it

    # Unpatch all first to ensure clean state
    just_grok_it.unpatch_all()

    assert just_grok_it.is_patched() is False
    assert just_grok_it.is_patched("openai") is False
    assert just_grok_it.is_patched("openrouter") is False
    assert just_grok_it.is_patched("anthropic") is False
    assert just_grok_it.is_patched("gemini") is False
    assert just_grok_it.is_patched("mistral") is False
    assert just_grok_it.is_patched("cohere") is False


def test_patch_and_unpatch_all():
    """Test patching and unpatching all providers."""
    import just_grok_it

    # Ensure clean state
    just_grok_it.unpatch_all()
    assert just_grok_it.is_patched() is False

    # Patch all
    results = just_grok_it.all(default_model="grok-4")

    # At least OpenAI should be patched
    assert results.get("openai") is True
    assert just_grok_it.is_patched() is True
    assert just_grok_it.is_patched("openai") is True

    # Unpatch all
    just_grok_it.unpatch_all()
    assert just_grok_it.is_patched() is False
    assert just_grok_it.is_patched("openai") is False


def test_all_returns_dict():
    """Test that all() returns a dictionary of results."""
    import just_grok_it

    just_grok_it.unpatch_all()
    results = just_grok_it.all(default_model="grok-4")

    assert isinstance(results, dict)
    assert "openai" in results
    # Other providers may or may not be present depending on installation

    just_grok_it.unpatch_all()


def test_debug_logging(caplog):
    """Test that debug logging works when patching."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged about patching
        assert any("[just_grok_it]" in record.message for record in caplog.records)

    just_grok_it.unpatch_all()


def test_initialization_logs_patched_providers(caplog):
    """Test that initialization logs the patched providers list."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all(default_model="grok-4")

        # Should have logged the patched providers
        assert any("Patched providers:" in record.message for record in caplog.records)

        # Should have logged the default model
        assert any(
            "Initialized with default model:" in record.message
            and "grok-4" in record.message
            for record in caplog.records
        )

    just_grok_it.unpatch_all()


def test_initialization_logs_redirect_url(caplog):
    """Test that initialization logs the redirect URL."""
    import just_grok_it

    just_grok_it.unpatch_all()

    with caplog.at_level(logging.DEBUG, logger="just_grok_it"):
        just_grok_it.all()

        # Should have logged the redirect URL
        assert any(
            "redirected to" in record.message.lower() and "api.x.ai" in record.message
            for record in caplog.records
        )

    just_grok_it.unpatch_all()
