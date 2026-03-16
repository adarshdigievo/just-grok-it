"""
OpenAI SDK provider for xAI API redirection.

This provider handles:
- Direct OpenAI SDK usage
- OpenAI SDK usage with OpenRouter, Together, Groq, and other compatible APIs
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from just_grok_it._base import BaseProvider
from just_grok_it._constants import DEFAULT_XAI_MODEL, KNOWN_PROVIDER_URLS, XAI_BASE_URL

logger = logging.getLogger("just_grok_it")


class OpenAIProvider(BaseProvider):
    """
    Provider for patching the OpenAI Python SDK.

    Redirects all OpenAI API calls (including those using OpenRouter,
    Together, Groq, etc.) to xAI's OpenAI-compatible API.
    """

    name = "openai"
    package_name = "openai"

    def __init__(self) -> None:
        super().__init__()
        self._original_openai_init: Optional[Any] = None
        self._original_async_openai_init: Optional[Any] = None

    def is_installed(self) -> bool:
        """Check if the openai package is installed."""
        try:
            import openai

            return True
        except ImportError:
            return False

    def patch(self, default_model: Optional[str] = None) -> None:
        """Patch OpenAI SDK to redirect to xAI APIs."""
        if self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Already patched, skipping")
            return

        if not self.is_installed():
            logger.debug(f"[just_grok_it] {self.name}: SDK not installed, skipping")
            return

        self._default_model = default_model or DEFAULT_XAI_MODEL

        from openai import AsyncOpenAI, OpenAI

        # Store original __init__ methods
        self._original_openai_init = OpenAI.__init__
        self._original_async_openai_init = AsyncOpenAI.__init__

        # Patch the __init__ methods
        OpenAI.__init__ = self._create_patched_init(
            self._original_openai_init, "OpenAI"
        )
        AsyncOpenAI.__init__ = self._create_patched_init(
            self._original_async_openai_init, "AsyncOpenAI"
        )

        self._patched = True

        logger.debug(
            f"[just_grok_it] {self.name}: Successfully patched to redirect to xAI APIs. "
            f"Default model: {self._default_model}"
        )

    def unpatch(self) -> None:
        """Remove the patch and restore original OpenAI SDK behavior."""
        if not self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Not patched, nothing to unpatch")
            return

        if not self.is_installed():
            return

        from openai import AsyncOpenAI, OpenAI

        # Restore original __init__ methods
        if self._original_openai_init is not None:
            OpenAI.__init__ = self._original_openai_init

        if self._original_async_openai_init is not None:
            AsyncOpenAI.__init__ = self._original_async_openai_init

        self._original_openai_init = None
        self._original_async_openai_init = None
        self._patched = False
        self._default_model = None

        logger.debug(f"[just_grok_it] {self.name}: Successfully unpatched")

    def _detect_provider(self, base_url: Optional[str]) -> Optional[str]:
        """Detect which provider is being used based on base_url."""
        if base_url is None:
            return "openai"

        base_url_str = str(base_url).lower()
        for provider, url in KNOWN_PROVIDER_URLS.items():
            if url.lower() in base_url_str:
                return provider

        return None

    def _create_patched_init(self, original_init: Any, class_name: str) -> Any:
        """Create a patched __init__ that redirects to xAI APIs."""
        import os

        provider = self

        @functools.wraps(original_init)
        def patched_init(client_self: Any, *args: Any, **kwargs: Any) -> None:
            original_base_url = kwargs.get("base_url")
            detected_provider = provider._detect_provider(original_base_url)

            # Always redirect to xAI (unless it's already pointing to xAI)
            if original_base_url is None or (
                detected_provider is not None
                and "x.ai" not in str(original_base_url).lower()
            ):
                kwargs["base_url"] = XAI_BASE_URL

                if detected_provider and original_base_url is not None:
                    logger.debug(
                        f"[just_grok_it] {provider.name}: Intercepted {class_name} "
                        f"(detected: {detected_provider}), redirecting to xAI API: {XAI_BASE_URL}"
                    )
                else:
                    logger.debug(
                        f"[just_grok_it] {provider.name}: Intercepted {class_name} initialization, "
                        f"redirecting to xAI API: {XAI_BASE_URL}"
                    )

            # If no api_key provided, check for XAI_API_KEY environment variable
            # OpenAI SDK only checks OPENAI_API_KEY by default
            if kwargs.get("api_key") is None and len(args) == 0:
                xai_key = os.environ.get("XAI_API_KEY")
                if xai_key:
                    kwargs["api_key"] = xai_key
                    logger.debug(
                        f"[just_grok_it] {provider.name}: Using XAI_API_KEY from environment"
                    )

            # Call the original __init__
            original_init(client_self, *args, **kwargs)

        return patched_init
