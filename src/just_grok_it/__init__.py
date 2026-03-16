"""
Just Grok It - Auto-redirect LLM API requests to xAI APIs.

Usage:
    import just_grok_it
    just_grok_it.all()  # Uses default model: grok-4-1-fast-non-reasoning

    # Now all supported LLM SDK calls will be redirected to xAI APIs

    # OpenAI / Together / Groq / etc. (via openai SDK with custom base_url)
    from openai import OpenAI
    client = OpenAI()  # Uses XAI_API_KEY, connects to xAI
    response = client.chat.completions.create(...)

    # OpenRouter (official SDK)
    from openrouter import OpenRouter
    client = OpenRouter()  # Calls are converted and sent to xAI
    response = client.chat.send(...)

    # Anthropic
    from anthropic import Anthropic
    client = Anthropic()  # Calls are converted and sent to xAI
    response = client.messages.create(...)

    # Google Gemini
    import google.generativeai as genai
    model = genai.GenerativeModel('gemini-pro')  # Uses xAI instead
    response = model.generate_content("Hello!")

    # Mistral AI
    from mistralai import Mistral
    client = Mistral()
    response = client.chat.complete(...)  # Redirected to xAI

    # Cohere
    import cohere
    client = cohere.Client()
    response = client.chat(...)  # Redirected to xAI
"""

from just_grok_it._constants import DEFAULT_XAI_MODEL, XAI_BASE_URL
from just_grok_it._patcher import (
    all,
    get_installed_providers,
    get_patched_providers,
    is_patched,
    patch_all,
    patch_openai,
    unpatch_all,
    unpatch_openai,
)

__version__ = "0.1.0"
__all__ = [
    "all",
    "patch_all",
    "unpatch_all",
    "patch_openai",
    "unpatch_openai",
    "is_patched",
    "get_installed_providers",
    "get_patched_providers",
    "XAI_BASE_URL",
    "DEFAULT_XAI_MODEL",
    "__version__",
]
