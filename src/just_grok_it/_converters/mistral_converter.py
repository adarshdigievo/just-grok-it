"""
Converter for Mistral AI API format to/from OpenAI API format.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional


def convert_mistral_to_openai(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stop: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert Mistral chat.complete() request to OpenAI chat.completions.create() format.

    Args:
        messages: Mistral-format messages
        model: Model to use
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum tokens
        stream: Whether to stream
        stop: Stop sequences

    Returns:
        OpenAI-format request dictionary
    """
    # Mistral messages are already in OpenAI-compatible format
    openai_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle tool calls if present
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        openai_messages.append({"role": role, "content": content})

    request = {
        "model": model,
        "messages": openai_messages,
        "stream": stream,
    }

    if temperature is not None:
        request["temperature"] = temperature

    if top_p is not None:
        request["top_p"] = top_p

    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    if stop:
        request["stop"] = stop

    return request


@dataclass
class MistralUsage:
    """Represents Mistral usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MistralMessage:
    """Represents a Mistral message."""

    role: str = "assistant"
    content: str = ""


@dataclass
class MistralChoice:
    """Represents a Mistral choice."""

    index: int = 0
    message: MistralMessage = field(default_factory=MistralMessage)
    finish_reason: Optional[str] = None


@dataclass
class MistralChatCompletionResponse:
    """
    Represents a Mistral ChatCompletionResponse.

    This mimics the structure returned by mistral.chat.complete()
    """

    id: str = ""
    object: str = "chat.completion"
    model: str = ""
    choices: List[MistralChoice] = field(default_factory=list)
    usage: MistralUsage = field(default_factory=MistralUsage)


def convert_openai_to_mistral(
    openai_response: Any, original_model: str
) -> MistralChatCompletionResponse:
    """
    Convert OpenAI chat.completions response to Mistral format.

    Args:
        openai_response: OpenAI ChatCompletion response object
        original_model: The original model requested

    Returns:
        MistralChatCompletionResponse object
    """
    choice = openai_response.choices[0] if openai_response.choices else None

    content = ""
    finish_reason = None

    if choice:
        content = choice.message.content or ""
        finish_reason = choice.finish_reason

    usage = MistralUsage(
        prompt_tokens=getattr(openai_response.usage, "prompt_tokens", 0),
        completion_tokens=getattr(openai_response.usage, "completion_tokens", 0),
        total_tokens=getattr(openai_response.usage, "total_tokens", 0),
    )

    return MistralChatCompletionResponse(
        id=openai_response.id or f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        model=original_model,
        choices=[
            MistralChoice(
                index=0,
                message=MistralMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


@dataclass
class MistralStreamDelta:
    """Represents a Mistral stream delta."""

    role: Optional[str] = None
    content: str = ""


@dataclass
class MistralStreamChoice:
    """Represents a Mistral stream choice."""

    index: int = 0
    delta: MistralStreamDelta = field(default_factory=MistralStreamDelta)
    finish_reason: Optional[str] = None


@dataclass
class MistralStreamEvent:
    """Represents a Mistral streaming event."""

    id: str = ""
    model: str = ""
    choices: List[MistralStreamChoice] = field(default_factory=list)


class MistralStreamWrapper:
    """Wraps an OpenAI stream to yield Mistral-format events."""

    def __init__(self, openai_stream: Iterator[Any], model: str):
        self._stream = openai_stream
        self._model = model
        self._id = f"chatcmpl-{uuid.uuid4().hex}"

    def __iter__(self) -> Iterator[MistralStreamEvent]:
        return self

    def __next__(self) -> MistralStreamEvent:
        chunk = next(self._stream)
        if chunk.choices and chunk.choices[0].delta.content:
            return MistralStreamEvent(
                id=self._id,
                model=self._model,
                choices=[
                    MistralStreamChoice(
                        index=0,
                        delta=MistralStreamDelta(
                            content=chunk.choices[0].delta.content
                        ),
                        finish_reason=chunk.choices[0].finish_reason,
                    )
                ],
            )
        return self.__next__()


def create_mistral_stream_wrapper(
    openai_stream: Any, model: str
) -> MistralStreamWrapper:
    """Create a Mistral-compatible stream wrapper."""
    return MistralStreamWrapper(openai_stream, model)


class AsyncMistralStreamWrapper:
    """Wraps an async OpenAI stream to yield Mistral-format events."""

    def __init__(self, client: Any, request: dict, model: str):
        self._client = client
        self._request = request
        self._model = model
        self._stream = None
        self._started = False
        self._id = f"chatcmpl-{uuid.uuid4().hex}"

    def __aiter__(self) -> AsyncIterator[MistralStreamEvent]:
        return self

    async def __anext__(self) -> MistralStreamEvent:
        if not self._started:
            self._started = True
            self._stream = await self._client.chat.completions.create(**self._request)

        chunk = await self._stream.__anext__()
        if chunk.choices and chunk.choices[0].delta.content:
            return MistralStreamEvent(
                id=self._id,
                model=self._model,
                choices=[
                    MistralStreamChoice(
                        index=0,
                        delta=MistralStreamDelta(
                            content=chunk.choices[0].delta.content
                        ),
                        finish_reason=chunk.choices[0].finish_reason,
                    )
                ],
            )
        return await self.__anext__()


def create_async_mistral_stream_wrapper(
    client: Any, request: dict, model: str
) -> AsyncMistralStreamWrapper:
    """Create an async Mistral-compatible stream wrapper."""
    return AsyncMistralStreamWrapper(client, request, model)
