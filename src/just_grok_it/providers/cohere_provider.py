"""
Cohere SDK provider for xAI API redirection.

Intercepts Cohere API calls and redirects them to xAI's OpenAI-compatible API.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from just_grok_it._base import BaseProvider
from just_grok_it._constants import DEFAULT_XAI_MODEL, XAI_BASE_URL
from just_grok_it._converters.cohere_converter import (
    convert_cohere_to_openai,
    convert_openai_to_cohere,
)

logger = logging.getLogger("just_grok_it")


class CohereProvider(BaseProvider):
    """
    Provider for patching the Cohere Python SDK.

    Intercepts Cohere chat() calls, converts them to OpenAI format,
    sends to xAI, and converts the response back to Cohere format.
    """

    name = "cohere"
    package_name = "cohere"

    def __init__(self) -> None:
        super().__init__()
        self._original_chat: Optional[Any] = None
        self._original_chat_async: Optional[Any] = None
        self._client_class: Optional[Any] = None
        self._async_client_class: Optional[Any] = None

    def is_installed(self) -> bool:
        """Check if the cohere package is installed."""
        try:
            import cohere

            return True
        except ImportError:
            return False

    def _get_openai_client(self) -> Any:
        """Get an OpenAI client configured for xAI."""
        import os

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to redirect Cohere calls to xAI. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return OpenAI(base_url=XAI_BASE_URL, api_key=api_key)

    def _get_async_openai_client(self) -> Any:
        """Get an async OpenAI client configured for xAI."""
        import os

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to redirect Cohere calls to xAI. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(base_url=XAI_BASE_URL, api_key=api_key)

    def patch(self, default_model: Optional[str] = None) -> None:
        """Patch Cohere SDK to redirect to xAI APIs."""
        if self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Already patched, skipping")
            return

        if not self.is_installed():
            logger.debug(f"[just_grok_it] {self.name}: SDK not installed, skipping")
            return

        self._default_model = default_model or DEFAULT_XAI_MODEL

        try:
            import cohere

            self._client_class = cohere.Client
            self._async_client_class = cohere.AsyncClient

            # Store original chat methods
            self._original_chat = cohere.Client.chat
            self._original_chat_async = cohere.AsyncClient.chat

            # Patch chat methods
            cohere.Client.chat = self._create_patched_chat()
            cohere.AsyncClient.chat = self._create_patched_async_chat()

        except Exception as e:
            logger.warning(f"[just_grok_it] {self.name}: Failed to patch: {e}")
            return

        self._patched = True

        logger.debug(
            f"[just_grok_it] {self.name}: Successfully patched to redirect to xAI APIs. "
            f"Default model: {self._default_model}"
        )

    def unpatch(self) -> None:
        """Remove the patch and restore original Cohere SDK behavior."""
        if not self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Not patched, nothing to unpatch")
            return

        if not self.is_installed():
            return

        try:
            import cohere

            if self._original_chat is not None:
                cohere.Client.chat = self._original_chat

            if self._original_chat_async is not None:
                cohere.AsyncClient.chat = self._original_chat_async

        except Exception:
            pass

        self._original_chat = None
        self._original_chat_async = None
        self._client_class = None
        self._async_client_class = None
        self._patched = False
        self._default_model = None

        logger.debug(f"[just_grok_it] {self.name}: Successfully unpatched")

    def _create_patched_chat(self) -> Any:
        """Create a patched chat method."""
        provider = self

        def patched_chat(
            self_client: Any,
            *,
            message: str = None,
            model: str = None,
            **kwargs: Any,
        ) -> Any:
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted chat request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Cohere request to OpenAI format
            openai_request = convert_cohere_to_openai(
                message=message,
                model=actual_model,
                chat_history=kwargs.get("chat_history"),
                preamble=kwargs.get("preamble"),
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                stop_sequences=kwargs.get("stop_sequences"),
            )

            # Make the call to xAI using OpenAI client
            client = provider._get_openai_client()
            openai_response = client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Cohere format
            return convert_openai_to_cohere(openai_response, model or "command")

        return patched_chat

    def _create_patched_async_chat(self) -> Any:
        """Create a patched async chat method."""
        provider = self

        async def patched_chat(
            self_client: Any,
            *,
            message: str = None,
            model: str = None,
            **kwargs: Any,
        ) -> Any:
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted async chat request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Cohere request to OpenAI format
            openai_request = convert_cohere_to_openai(
                message=message,
                model=actual_model,
                chat_history=kwargs.get("chat_history"),
                preamble=kwargs.get("preamble"),
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                stop_sequences=kwargs.get("stop_sequences"),
            )

            # Make the call to xAI using async OpenAI client
            client = provider._get_async_openai_client()
            openai_response = await client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Cohere format
            return convert_openai_to_cohere(openai_response, model or "command")

        return patched_chat
