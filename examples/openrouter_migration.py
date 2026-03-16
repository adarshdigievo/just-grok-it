#!/usr/bin/env python3
"""
Example: Migrating from OpenRouter to xAI.

This example shows how existing code using OpenRouter (either the official SDK
or via the OpenAI SDK with custom base_url) can be seamlessly redirected to xAI.

Prerequisites:
    pip install just-grok-it openrouter openai
    export XAI_API_KEY="your-xai-api-key"
"""

# Step 1: Import just-grok-it at the top of your entry point
import just_grok_it

just_grok_it.all()

# =============================================================================
# Option 1: Using the Official OpenRouter SDK
# =============================================================================
print("=" * 60)
print("Option 1: Official OpenRouter SDK")
print("=" * 60)

from openrouter import OpenRouter

# Your existing OpenRouter SDK code - NO CHANGES NEEDED!
client = OpenRouter(api_key="your-openrouter-key")

print("OpenRouter client created - calls will be redirected to xAI!\n")

# Your existing API calls work unchanged
response = client.chat.send(
    model="anthropic/claude-3-opus",  # Original model - Grok will be used
    messages=[
        {"role": "user", "content": "Hello! What model am I talking to?"},
    ],
)

print("Response:")
print(response.choices[0].message.content)
print()

# =============================================================================
# Option 2: Using OpenAI SDK with OpenRouter base_url
# =============================================================================
print("=" * 60)
print("Option 2: OpenAI SDK with OpenRouter base_url")
print("=" * 60)

from openai import OpenAI

# This code was originally using OpenRouter via OpenAI SDK
# The base_url will be automatically redirected to xAI
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # Will be redirected!
    api_key="your-openrouter-key",  # Will use XAI_API_KEY instead
)

print(f"Client base URL after patching: {openai_client.base_url}")
print("Note: OpenRouter URL has been redirected to xAI!\n")

# Your existing API calls work unchanged
response = openai_client.chat.completions.create(
    model="anthropic/claude-3-opus",  # Original model - Grok will be used
    messages=[
        {"role": "user", "content": "Hello! What model am I talking to?"},
    ],
)

print("Response:")
print(response.choices[0].message.content)

# This also works with other OpenAI-compatible providers:
# - Together AI (https://api.together.xyz)
# - Groq (https://api.groq.com)
# - Perplexity (https://api.perplexity.ai)
# - And more!
