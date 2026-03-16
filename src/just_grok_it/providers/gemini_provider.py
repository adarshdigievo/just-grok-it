"""
Google Gemini SDK provider for xAI API redirection.

Intercepts Google GenAI API calls and redirects them to xAI's OpenAI-compatible API.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from just_grok_it._base import BaseProvider
from just_grok_it._constants import DEFAULT_XAI_MODEL, XAI_BASE_URL
from just_grok_it._converters.gemini_converter import (
    convert_gemini_to_openai,
    convert_openai_to_gemini,
)

logger = logging.getLogger("just_grok_it")


class GeminiProvider(BaseProvider):
    """
    Provider for patching the Google GenAI Python SDK.

    Intercepts GenerativeModel.generate_content() calls, converts them to OpenAI format,
    sends to xAI, and converts the response back to Gemini format.
    """

    name = "gemini"
    package_name = "google-generativeai"

    def __init__(self) -> None:
        super().__init__()
        self._original_generate_content: Optional[Any] = None
        self._original_generate_content_async: Optional[Any] = None
        self._generative_model_class: Optional[Any] = None

    def is_installed(self) -> bool:
        """Check if the google-generativeai package is installed."""
        try:
            import google.generativeai

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
                "The 'openai' package is required to redirect Gemini calls to xAI. "
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
                "The 'openai' package is required to redirect Gemini calls to xAI. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(base_url=XAI_BASE_URL, api_key=api_key)

    def patch(self, default_model: Optional[str] = None) -> None:
        """Patch Gemini SDK to redirect to xAI APIs."""
        if self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Already patched, skipping")
            return

        if not self.is_installed():
            logger.debug(f"[just_grok_it] {self.name}: SDK not installed, skipping")
            return

        self._default_model = default_model or DEFAULT_XAI_MODEL

        import google.generativeai as genai

        # Get the GenerativeModel class
        self._generative_model_class = genai.GenerativeModel

        # Store original generate_content methods
        self._original_generate_content = self._generative_model_class.generate_content
        self._original_generate_content_async = (
            self._generative_model_class.generate_content_async
        )

        # Patch the generate_content methods
        self._generative_model_class.generate_content = (
            self._create_patched_generate_content()
        )
        self._generative_model_class.generate_content_async = (
            self._create_patched_generate_content_async()
        )

        self._patched = True

        logger.debug(
            f"[just_grok_it] {self.name}: Successfully patched to redirect to xAI APIs. "
            f"Default model: {default_model or 'None (use model from request)'}"
        )

    def unpatch(self) -> None:
        """Remove the patch and restore original Gemini SDK behavior."""
        if not self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Not patched, nothing to unpatch")
            return

        if not self.is_installed():
            return

        # Restore original generate_content methods
        if (
            self._original_generate_content is not None
            and self._generative_model_class is not None
        ):
            self._generative_model_class.generate_content = (
                self._original_generate_content
            )

        if (
            self._original_generate_content_async is not None
            and self._generative_model_class is not None
        ):
            self._generative_model_class.generate_content_async = (
                self._original_generate_content_async
            )

        self._original_generate_content = None
        self._original_generate_content_async = None
        self._generative_model_class = None
        self._patched = False
        self._default_model = None

        logger.debug(f"[just_grok_it] {self.name}: Successfully unpatched")

    def _create_patched_generate_content(self) -> Any:
        """Create a patched generate_content method."""
        provider = self

        def patched_generate_content(
            self_model: Any,
            contents: Any,
            *,
            generation_config: Any = None,
            safety_settings: Any = None,
            stream: bool = False,
            tools: Any = None,
            tool_config: Any = None,
            request_options: Any = None,
        ) -> Any:
            # Get the model name from the GenerativeModel instance
            gemini_model = getattr(self_model, "_model_name", "gemini-pro")
            actual_model = provider._default_model or gemini_model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted generate_content request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Gemini request to OpenAI format
            openai_request = convert_gemini_to_openai(
                contents=contents,
                model=actual_model,
                generation_config=generation_config,
                stream=stream,
            )

            # Make the call to xAI using OpenAI client
            client = provider._get_openai_client()

            if stream:
                return provider._handle_streaming(client, openai_request, gemini_model)

            openai_response = client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Gemini format
            return convert_openai_to_gemini(openai_response, gemini_model)

        return patched_generate_content

    def _create_patched_generate_content_async(self) -> Any:
        """Create a patched async generate_content method."""
        provider = self

        async def patched_generate_content_async(
            self_model: Any,
            contents: Any,
            *,
            generation_config: Any = None,
            safety_settings: Any = None,
            stream: bool = False,
            tools: Any = None,
            tool_config: Any = None,
            request_options: Any = None,
        ) -> Any:
            # Get the model name from the GenerativeModel instance
            gemini_model = getattr(self_model, "_model_name", "gemini-pro")
            actual_model = provider._default_model or gemini_model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted async generate_content request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert Gemini request to OpenAI format
            openai_request = convert_gemini_to_openai(
                contents=contents,
                model=actual_model,
                generation_config=generation_config,
                stream=stream,
            )

            # Make the call to xAI using async OpenAI client
            client = provider._get_async_openai_client()

            if stream:
                return provider._handle_async_streaming(
                    client, openai_request, gemini_model
                )

            openai_response = await client.chat.completions.create(**openai_request)

            # Convert OpenAI response back to Gemini format
            return convert_openai_to_gemini(openai_response, gemini_model)

        return patched_generate_content_async

    def _handle_streaming(self, client: Any, request: dict, gemini_model: str) -> Any:
        """Handle streaming responses by wrapping the OpenAI stream."""
        from just_grok_it._converters.gemini_converter import (
            create_gemini_stream_wrapper,
        )

        openai_stream = client.chat.completions.create(**request)
        return create_gemini_stream_wrapper(openai_stream, gemini_model)

    def _handle_async_streaming(
        self, client: Any, request: dict, gemini_model: str
    ) -> Any:
        """Handle async streaming responses by wrapping the OpenAI stream."""
        from just_grok_it._converters.gemini_converter import (
            create_async_gemini_stream_wrapper,
        )

        return create_async_gemini_stream_wrapper(client, request, gemini_model)
