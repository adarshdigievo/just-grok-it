"""
Anthropic SDK provider for xAI API redirection.

Intercepts Anthropic API calls and redirects them to xAI's OpenAI-compatible API.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from just_grok_it._base import BaseProvider
from just_grok_it._constants import DEFAULT_XAI_MODEL, XAI_BASE_URL
from just_grok_it._converters.anthropic_converter import (
    convert_anthropic_to_openai,
    convert_openai_to_anthropic,
)

logger = logging.getLogger("just_grok_it")


class AnthropicProvider(BaseProvider):
    """
    Provider for patching the Anthropic Python SDK.

    Intercepts Anthropic messages.create() calls, converts them to OpenAI format,
    sends to xAI, and converts the response back to Anthropic format.
    """

    name = "anthropic"
    package_name = "anthropic"

    def __init__(self) -> None:
        super().__init__()
        self._original_messages_create: Optional[Any] = None
        self._original_async_messages_create: Optional[Any] = None
        self._messages_class: Optional[Any] = None
        self._async_messages_class: Optional[Any] = None

    def is_installed(self) -> bool:
        """Check if the anthropic package is installed."""
        try:
            import anthropic

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
                "The 'openai' package is required to redirect Anthropic calls to xAI. "
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
                "The 'openai' package is required to redirect Anthropic calls to xAI. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(base_url=XAI_BASE_URL, api_key=api_key)

    def patch(self, default_model: Optional[str] = None) -> None:
        """Patch Anthropic SDK to redirect to xAI APIs."""
        if self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Already patched, skipping")
            return

        if not self.is_installed():
            logger.debug(f"[just_grok_it] {self.name}: SDK not installed, skipping")
            return

        self._default_model = default_model or DEFAULT_XAI_MODEL

        import anthropic

        # Get the Messages resource classes
        self._messages_class = anthropic.resources.Messages
        self._async_messages_class = anthropic.resources.AsyncMessages

        # Store original create methods
        self._original_messages_create = self._messages_class.create
        self._original_async_messages_create = self._async_messages_class.create

        # Patch the create methods
        self._messages_class.create = self._create_patched_messages_create()
        self._async_messages_class.create = self._create_patched_async_messages_create()

        self._patched = True

        logger.debug(
            f"[just_grok_it] {self.name}: Successfully patched to redirect to xAI APIs. "
            f"Default model: {default_model or 'None (use model from request)'}"
        )

    def unpatch(self) -> None:
        """Remove the patch and restore original Anthropic SDK behavior."""
        if not self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Not patched, nothing to unpatch")
            return

        if not self.is_installed():
            return

        # Restore original create methods
        if (
            self._original_messages_create is not None
            and self._messages_class is not None
        ):
            self._messages_class.create = self._original_messages_create

        if (
            self._original_async_messages_create is not None
            and self._async_messages_class is not None
        ):
            self._async_messages_class.create = self._original_async_messages_create

        self._original_messages_create = None
        self._original_async_messages_create = None
        self._messages_class = None
        self._async_messages_class = None
        self._patched = False
        self._default_model = None

        logger.debug(f"[just_grok_it] {self.name}: Successfully unpatched")

    def _create_patched_messages_create(self) -> Any:
        """Create a patched messages.create method."""
        provider = self

        def patched_create(
            self_messages: Any,
            *,
            max_tokens: int,
            messages: list,
            model: str,
            **kwargs: Any,
        ) -> Any:
            # Use default model if configured
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted messages.create request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Anthropic request to OpenAI format
            openai_request = convert_anthropic_to_openai(
                messages=messages,
                model=actual_model,
                max_tokens=max_tokens,
                system=kwargs.get("system"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                stop_sequences=kwargs.get("stop_sequences"),
                stream=kwargs.get("stream", False),
            )

            # Make the call to xAI using OpenAI client
            client = provider._get_openai_client()

            if kwargs.get("stream", False):
                # Handle streaming
                return provider._handle_streaming(client, openai_request)

            openai_response = client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Anthropic format
            return convert_openai_to_anthropic(openai_response, model)

        return patched_create

    def _create_patched_async_messages_create(self) -> Any:
        """Create a patched async messages.create method."""
        provider = self

        async def patched_create(
            self_messages: Any,
            *,
            max_tokens: int,
            messages: list,
            model: str,
            **kwargs: Any,
        ) -> Any:
            # Use default model if configured
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted async messages.create request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Anthropic request to OpenAI format
            openai_request = convert_anthropic_to_openai(
                messages=messages,
                model=actual_model,
                max_tokens=max_tokens,
                system=kwargs.get("system"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                stop_sequences=kwargs.get("stop_sequences"),
                stream=kwargs.get("stream", False),
            )

            # Make the call to xAI using async OpenAI client
            client = provider._get_async_openai_client()

            if kwargs.get("stream", False):
                # Handle streaming
                return provider._handle_async_streaming(client, openai_request)

            openai_response = await client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Anthropic format
            return convert_openai_to_anthropic(openai_response, model)

        return patched_create

    def _handle_streaming(self, client: Any, request: dict) -> Any:
        """Handle streaming responses by wrapping the OpenAI stream."""
        from just_grok_it._converters.anthropic_converter import (
            create_anthropic_stream_wrapper,
        )

        openai_stream = client.chat.completions.create(**request)
        return create_anthropic_stream_wrapper(openai_stream, request.get("model", ""))

    def _handle_async_streaming(self, client: Any, request: dict) -> Any:
        """Handle async streaming responses by wrapping the OpenAI stream."""
        from just_grok_it._converters.anthropic_converter import (
            create_async_anthropic_stream_wrapper,
        )

        return create_async_anthropic_stream_wrapper(client, request)
