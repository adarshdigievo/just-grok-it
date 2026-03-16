"""
Constants used throughout the just_grok_it package.
"""

# xAI API base URL (OpenAI compatible)
XAI_BASE_URL = "https://api.x.ai/v1"

# Default xAI model (fast, non-reasoning variant for efficiency)
DEFAULT_XAI_MODEL = "grok-4-1-fast-non-reasoning"

# Known OpenAI-compatible provider base URLs (for detection)
KNOWN_PROVIDER_URLS = {
    "openai": "https://api.openai.com",
    "openrouter": "https://openrouter.ai/api",
    "together": "https://api.together.xyz",
    "groq": "https://api.groq.com",
    "perplexity": "https://api.perplexity.ai",
    "fireworks": "https://api.fireworks.ai",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai",
}
