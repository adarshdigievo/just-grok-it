"""
Mistral AI SDK provider for xAI API redirection.

Intercepts Mistral API calls and redirects them to xAI's OpenAI-compatible API.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from just_grok_it._base import BaseProvider
from just_grok_it._constants import DEFAULT_XAI_MODEL, XAI_BASE_URL
from just_grok_it._converters.mistral_converter import (
    convert_mistral_to_openai,
    convert_openai_to_mistral,
)

logger = logging.getLogger("just_grok_it")


class MistralProvider(BaseProvider):
    """
    Provider for patching the Mistral AI Python SDK.

    Intercepts Mistral chat.complete() calls, converts them to OpenAI format,
    sends to xAI, and converts the response back to Mistral format.
    """

    name = "mistral"
    package_name = "mistralai"

    def __init__(self) -> None:
        super().__init__()
        self._original_chat_complete: Optional[Any] = None
        self._original_chat_complete_async: Optional[Any] = None
        self._chat_class: Optional[Any] = None

    def is_installed(self) -> bool:
        """Check if the mistralai package is installed."""
        try:
            import mistralai

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
                "The 'openai' package is required to redirect Mistral calls to xAI. "
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
                "The 'openai' package is required to redirect Mistral calls to xAI. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(base_url=XAI_BASE_URL, api_key=api_key)

    def patch(self, default_model: Optional[str] = None) -> None:
        """Patch Mistral SDK to redirect to xAI APIs."""
        if self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Already patched, skipping")
            return

        if not self.is_installed():
            logger.debug(f"[just_grok_it] {self.name}: SDK not installed, skipping")
            return

        self._default_model = default_model or DEFAULT_XAI_MODEL

        try:
            from mistralai.chat import Chat

            self._chat_class = Chat

            # Store original complete methods
            self._original_chat_complete = Chat.complete
            self._original_chat_complete_async = Chat.complete_async

            # Patch the complete methods
            Chat.complete = self._create_patched_chat_complete()
            Chat.complete_async = self._create_patched_chat_complete_async()

        except Exception as e:
            logger.warning(f"[just_grok_it] {self.name}: Failed to patch: {e}")
            return

        self._patched = True

        logger.debug(
            f"[just_grok_it] {self.name}: Successfully patched to redirect to xAI APIs. "
            f"Default model: {self._default_model}"
        )

    def unpatch(self) -> None:
        """Remove the patch and restore original Mistral SDK behavior."""
        if not self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Not patched, nothing to unpatch")
            return

        if not self.is_installed():
            return

        try:
            from mistralai.chat import Chat

            if self._original_chat_complete is not None:
                Chat.complete = self._original_chat_complete

            if self._original_chat_complete_async is not None:
                Chat.complete_async = self._original_chat_complete_async

        except Exception:
            pass

        self._original_chat_complete = None
        self._original_chat_complete_async = None
        self._chat_class = None
        self._patched = False
        self._default_model = None

        logger.debug(f"[just_grok_it] {self.name}: Successfully unpatched")

    def _create_patched_chat_complete(self) -> Any:
        """Create a patched chat.complete method."""
        provider = self

        def patched_complete(
            self_chat: Any,
            *,
            model: str,
            messages: list,
            **kwargs: Any,
        ) -> Any:
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted chat.complete request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Mistral request to OpenAI format
            openai_request = convert_mistral_to_openai(
                messages=provider._convert_messages(messages),
                model=actual_model,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                max_tokens=kwargs.get("max_tokens"),
                stream=kwargs.get("stream", False),
                stop=kwargs.get("stop"),
            )

            # Make the call to xAI using OpenAI client
            client = provider._get_openai_client()

            if kwargs.get("stream", False):
                return provider._handle_streaming(client, openai_request, model)

            openai_response = client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Mistral format
            return convert_openai_to_mistral(openai_response, model)

        return patched_complete

    def _create_patched_chat_complete_async(self) -> Any:
        """Create a patched async chat.complete method."""
        provider = self

        async def patched_complete_async(
            self_chat: Any,
            *,
            model: str,
            messages: list,
            **kwargs: Any,
        ) -> Any:
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted async chat.complete request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Mistral request to OpenAI format
            openai_request = convert_mistral_to_openai(
                messages=provider._convert_messages(messages),
                model=actual_model,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                max_tokens=kwargs.get("max_tokens"),
                stream=kwargs.get("stream", False),
                stop=kwargs.get("stop"),
            )

            # Make the call to xAI using async OpenAI client
            client = provider._get_async_openai_client()

            if kwargs.get("stream", False):
                return await provider._handle_async_streaming(
                    client, openai_request, model
                )

            openai_response = await client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Mistral format
            return convert_openai_to_mistral(openai_response, model)

        return patched_complete_async

    def _convert_messages(self, messages: list) -> list:
        """Convert Mistral message objects to dict format."""
        converted = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # It's a message object
                converted.append(
                    {
                        "role": msg.role,
                        "content": msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content),
                    }
                )
            elif isinstance(msg, dict):
                converted.append(msg)
            else:
                # Try to extract role and content
                converted.append(
                    {
                        "role": getattr(msg, "role", "user"),
                        "content": str(getattr(msg, "content", msg)),
                    }
                )
        return converted

    def _handle_streaming(self, client: Any, request: dict, model: str) -> Any:
        """Handle streaming responses."""
        from just_grok_it._converters.mistral_converter import (
            create_mistral_stream_wrapper,
        )

        openai_stream = client.chat.completions.create(**request)
        return create_mistral_stream_wrapper(openai_stream, model)

    async def _handle_async_streaming(
        self, client: Any, request: dict, model: str
    ) -> Any:
        """Handle async streaming responses."""
        from just_grok_it._converters.mistral_converter import (
            create_async_mistral_stream_wrapper,
        )

        return create_async_mistral_stream_wrapper(client, request, model)
