"""
OpenRouter SDK provider for xAI API redirection.

This provider handles the official OpenRouter Python SDK.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from just_grok_it._base import BaseProvider
from just_grok_it._constants import DEFAULT_XAI_MODEL, XAI_BASE_URL

logger = logging.getLogger("just_grok_it")


class OpenRouterProvider(BaseProvider):
    """
    Provider for patching the official OpenRouter Python SDK.

    Redirects OpenRouter API calls to xAI's OpenAI-compatible API.
    """

    name = "openrouter"
    package_name = "openrouter"

    def __init__(self) -> None:
        super().__init__()
        self._original_openrouter_init: Optional[Any] = None
        self._original_chat_send: Optional[Any] = None
        self._original_chat_send_async: Optional[Any] = None
        self._openrouter_class: Optional[Any] = None
        self._chat_class: Optional[Any] = None

    def is_installed(self) -> bool:
        """Check if the openrouter package is installed."""
        try:
            import openrouter

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
                "The 'openai' package is required to redirect OpenRouter calls to xAI. "
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
                "The 'openai' package is required to redirect OpenRouter calls to xAI. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return AsyncOpenAI(base_url=XAI_BASE_URL, api_key=api_key)

    def patch(self, default_model: Optional[str] = None) -> None:
        """Patch OpenRouter SDK to redirect to xAI APIs."""
        if self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Already patched, skipping")
            return

        if not self.is_installed():
            logger.debug(f"[just_grok_it] {self.name}: SDK not installed, skipping")
            return

        self._default_model = default_model or DEFAULT_XAI_MODEL

        try:
            import openrouter
            from openrouter.chat import Chat

            self._openrouter_class = openrouter.OpenRouter
            self._chat_class = Chat

            # Store original methods
            self._original_chat_send = Chat.send
            self._original_chat_send_async = Chat.send_async

            # Patch chat.send and chat.send_async
            Chat.send = self._create_patched_chat_send()
            Chat.send_async = self._create_patched_chat_send_async()

        except Exception as e:
            logger.warning(f"[just_grok_it] {self.name}: Failed to patch: {e}")
            return

        self._patched = True

        logger.debug(
            f"[just_grok_it] {self.name}: Successfully patched to redirect to xAI APIs. "
            f"Default model: {self._default_model}"
        )

    def unpatch(self) -> None:
        """Remove the patch and restore original OpenRouter SDK behavior."""
        if not self._patched:
            logger.debug(f"[just_grok_it] {self.name}: Not patched, nothing to unpatch")
            return

        if not self.is_installed():
            return

        try:
            from openrouter.chat import Chat

            if self._original_chat_send is not None:
                Chat.send = self._original_chat_send

            if self._original_chat_send_async is not None:
                Chat.send_async = self._original_chat_send_async

        except Exception:
            pass

        self._original_openrouter_init = None
        self._original_chat_send = None
        self._original_chat_send_async = None
        self._openrouter_class = None
        self._chat_class = None
        self._patched = False
        self._default_model = None

        logger.debug(f"[just_grok_it] {self.name}: Successfully unpatched")

    def _create_patched_chat_send(self) -> Any:
        """Create a patched chat.send method."""
        provider = self

        def patched_send(
            self_chat: Any,
            *,
            messages: list,
            model: str = None,
            **kwargs: Any,
        ) -> Any:
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted chat.send request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert OpenRouter messages to OpenAI format
            openai_messages = provider._convert_messages(messages)

            # Build OpenAI request
            openai_request = {
                "model": actual_model,
                "messages": openai_messages,
            }

            # Map OpenRouter parameters to OpenAI
            if kwargs.get("temperature") is not None:
                openai_request["temperature"] = kwargs["temperature"]
            if kwargs.get("max_tokens") is not None:
                openai_request["max_tokens"] = kwargs["max_tokens"]
            if kwargs.get("max_completion_tokens") is not None:
                openai_request["max_completion_tokens"] = kwargs[
                    "max_completion_tokens"
                ]
            if kwargs.get("top_p") is not None:
                openai_request["top_p"] = kwargs["top_p"]
            if kwargs.get("frequency_penalty") is not None:
                openai_request["frequency_penalty"] = kwargs["frequency_penalty"]
            if kwargs.get("presence_penalty") is not None:
                openai_request["presence_penalty"] = kwargs["presence_penalty"]
            if kwargs.get("stop") is not None:
                openai_request["stop"] = kwargs["stop"]
            if kwargs.get("stream", False):
                openai_request["stream"] = True

            # Make the call to xAI using OpenAI client
            client = provider._get_openai_client()

            if kwargs.get("stream", False):
                return provider._handle_streaming(client, openai_request)

            openai_response = client.chat.completions.create(**openai_request)

            # Convert response to OpenRouter format
            return provider._convert_response(openai_response, model or actual_model)

        return patched_send

    def _create_patched_chat_send_async(self) -> Any:
        """Create a patched async chat.send method."""
        provider = self

        async def patched_send_async(
            self_chat: Any,
            *,
            messages: list,
            model: str = None,
            **kwargs: Any,
        ) -> Any:
            actual_model = provider._default_model or model

            logger.debug(
                f"[just_grok_it] {provider.name}: Intercepted async chat.send request, "
                f"redirecting to xAI API with model: {actual_model}"
            )

            # Convert OpenRouter messages to OpenAI format
            openai_messages = provider._convert_messages(messages)

            # Build OpenAI request
            openai_request = {
                "model": actual_model,
                "messages": openai_messages,
            }

            # Map OpenRouter parameters to OpenAI
            if kwargs.get("temperature") is not None:
                openai_request["temperature"] = kwargs["temperature"]
            if kwargs.get("max_tokens") is not None:
                openai_request["max_tokens"] = kwargs["max_tokens"]
            if kwargs.get("max_completion_tokens") is not None:
                openai_request["max_completion_tokens"] = kwargs[
                    "max_completion_tokens"
                ]
            if kwargs.get("top_p") is not None:
                openai_request["top_p"] = kwargs["top_p"]
            if kwargs.get("frequency_penalty") is not None:
                openai_request["frequency_penalty"] = kwargs["frequency_penalty"]
            if kwargs.get("presence_penalty") is not None:
                openai_request["presence_penalty"] = kwargs["presence_penalty"]
            if kwargs.get("stop") is not None:
                openai_request["stop"] = kwargs["stop"]
            if kwargs.get("stream", False):
                openai_request["stream"] = True

            # Make the call to xAI using async OpenAI client
            client = provider._get_async_openai_client()

            if kwargs.get("stream", False):
                return await provider._handle_async_streaming(client, openai_request)

            openai_response = await client.chat.completions.create(**openai_request)

            # Convert response to OpenRouter format
            return provider._convert_response(openai_response, model or actual_model)

        return patched_send_async

    def _convert_messages(self, messages: list) -> list:
        """Convert OpenRouter messages to OpenAI format."""
        openai_messages = []

        for msg in messages:
            # Handle message objects with role attribute
            if hasattr(msg, "role"):
                role = msg.role
                content = getattr(msg, "content", "")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                continue

            # Map roles
            if role == "developer":
                role = "system"

            openai_messages.append(
                {
                    "role": role,
                    "content": content if isinstance(content, str) else str(content),
                }
            )

        return openai_messages

    def _convert_response(self, openai_response: Any, original_model: str) -> Any:
        """Convert OpenAI response to OpenRouter format."""
        from dataclasses import dataclass
        from typing import List, Optional

        @dataclass
        class ContentPart:
            type: str
            text: str

        @dataclass
        class Message:
            role: str
            content: Optional[str]
            tool_calls: Optional[list] = None

        @dataclass
        class Choice:
            finish_reason: str
            index: int
            message: Message
            logprobs: Optional[Any] = None

        @dataclass
        class Usage:
            completion_tokens: int
            prompt_tokens: int
            total_tokens: int

        @dataclass
        class OpenRouterResponse:
            id: str
            choices: List[Choice]
            created: int
            model: str
            object: str
            usage: Usage

        # Build OpenRouter-compatible response
        choices = []
        for i, choice in enumerate(openai_response.choices):
            message = Message(
                role=choice.message.role,
                content=choice.message.content,
                tool_calls=getattr(choice.message, "tool_calls", None),
            )
            choices.append(
                Choice(
                    finish_reason=choice.finish_reason or "stop",
                    index=i,
                    message=message,
                    logprobs=getattr(choice, "logprobs", None),
                )
            )

        usage = Usage(
            completion_tokens=openai_response.usage.completion_tokens,
            prompt_tokens=openai_response.usage.prompt_tokens,
            total_tokens=openai_response.usage.total_tokens,
        )

        return OpenRouterResponse(
            id=openai_response.id,
            choices=choices,
            created=openai_response.created,
            model=original_model,
            object="chat.completion",
            usage=usage,
        )

    def _handle_streaming(self, client: Any, request: dict) -> Any:
        """Handle streaming responses."""
        # Return the OpenAI stream directly - it's compatible
        return client.chat.completions.create(**request)

    async def _handle_async_streaming(self, client: Any, request: dict) -> Any:
        """Handle async streaming responses."""
        return await client.chat.completions.create(**request)
