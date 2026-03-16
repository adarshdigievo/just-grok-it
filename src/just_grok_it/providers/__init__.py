"""
LLM SDK providers for just_grok_it.

Each provider handles patching a specific LLM SDK to redirect to xAI APIs.
"""

from just_grok_it.providers.anthropic_provider import AnthropicProvider
from just_grok_it.providers.cohere_provider import CohereProvider
from just_grok_it.providers.gemini_provider import GeminiProvider
from just_grok_it.providers.mistral_provider import MistralProvider
from just_grok_it.providers.openai_provider import OpenAIProvider
from just_grok_it.providers.openrouter_provider import OpenRouterProvider

# Registry of all available providers
# Order matters: OpenAI should be first as it's the most common
PROVIDERS = [
    OpenAIProvider,
    OpenRouterProvider,
    AnthropicProvider,
    GeminiProvider,
    MistralProvider,
    CohereProvider,
]

__all__ = [
    "OpenAIProvider",
    "OpenRouterProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MistralProvider",
    "CohereProvider",
    "PROVIDERS",
]
