"""
Converter for Cohere API format to/from OpenAI API format.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def convert_cohere_to_openai(
    message: Optional[str],
    model: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    preamble: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert Cohere chat() request to OpenAI chat.completions.create() format.

    Args:
        message: The user message
        model: Model to use
        chat_history: Previous conversation history
        preamble: System preamble/prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        stop_sequences: Stop sequences

    Returns:
        OpenAI-format request dictionary
    """
    openai_messages = []

    # Add preamble as system message
    if preamble:
        openai_messages.append({"role": "system", "content": preamble})

    # Convert chat history
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "USER").upper()
            content = msg.get("message", "")

            # Map Cohere roles to OpenAI roles
            if role == "USER":
                openai_role = "user"
            elif role == "CHATBOT":
                openai_role = "assistant"
            elif role == "SYSTEM":
                openai_role = "system"
            else:
                openai_role = "user"

            openai_messages.append({"role": openai_role, "content": content})

    # Add current message
    if message:
        openai_messages.append({"role": "user", "content": message})

    request = {
        "model": model,
        "messages": openai_messages,
    }

    if temperature is not None:
        request["temperature"] = temperature

    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    if stop_sequences:
        request["stop"] = stop_sequences

    return request


@dataclass
class CohereApiMeta:
    """Represents Cohere API metadata."""

    api_version: Dict[str, str] = field(default_factory=lambda: {"version": "1"})
    billed_units: Dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )
    tokens: Dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )


@dataclass
class CohereChatResponse:
    """
    Represents a Cohere chat response.

    This mimics the structure returned by cohere.Client.chat()
    """

    id: str = ""
    text: str = ""
    generation_id: str = ""
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "COMPLETE"
    meta: CohereApiMeta = field(default_factory=CohereApiMeta)


def convert_openai_to_cohere(
    openai_response: Any, original_model: str
) -> CohereChatResponse:
    """
    Convert OpenAI chat.completions response to Cohere format.

    Args:
        openai_response: OpenAI ChatCompletion response object
        original_model: The original model requested

    Returns:
        CohereChatResponse object
    """
    choice = openai_response.choices[0] if openai_response.choices else None

    text = ""
    finish_reason = "COMPLETE"

    if choice:
        text = choice.message.content or ""
        # Map OpenAI finish_reason to Cohere finish_reason
        fr = choice.finish_reason
        if fr == "stop":
            finish_reason = "COMPLETE"
        elif fr == "length":
            finish_reason = "MAX_TOKENS"
        else:
            finish_reason = "COMPLETE"

    meta = CohereApiMeta(
        api_version={"version": "1"},
        billed_units={
            "input_tokens": getattr(openai_response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(openai_response.usage, "completion_tokens", 0),
        },
        tokens={
            "input_tokens": getattr(openai_response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(openai_response.usage, "completion_tokens", 0),
        },
    )

    return CohereChatResponse(
        id=openai_response.id or f"chatcmpl-{uuid.uuid4().hex}",
        text=text,
        generation_id=f"gen-{uuid.uuid4().hex}",
        chat_history=[],
        finish_reason=finish_reason,
        meta=meta,
    )
