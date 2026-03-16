#!/usr/bin/env python3
"""
Example: Using multiple providers simultaneously.

This example shows how just-grok-it can redirect multiple
LLM SDKs to xAI at the same time.

Prerequisites:
    pip install just-grok-it openai anthropic google-generativeai
    export XAI_API_KEY="your-xai-api-key"
"""

import just_grok_it

# Patch all installed providers
results = just_grok_it.all()

print("Patching results:")
for provider, success in results.items():
    status = "✓ Patched" if success else "✗ Not installed/failed"
    print(f"  {provider}: {status}")

print(f"\nInstalled providers: {just_grok_it.get_installed_providers()}")
print(f"Patched providers: {just_grok_it.get_patched_providers()}")

# Now you can use any SDK - they all go to xAI!

# OpenAI
print("\n--- Using OpenAI SDK ---")
from openai import OpenAI

openai_client = OpenAI()
response = openai_client.chat.completions.create(
    model="grok-4",
    messages=[{"role": "user", "content": "Say 'Hello from OpenAI SDK'"}],
    max_tokens=50,
)
print(f"OpenAI SDK: {response.choices[0].message.content}")

# Anthropic (if installed)
try:
    print("\n--- Using Anthropic SDK ---")
    from anthropic import Anthropic

    anthropic_client = Anthropic()
    response = anthropic_client.messages.create(
        model="claude-3-sonnet",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'Hello from Anthropic SDK'"}],
    )
    print(f"Anthropic SDK: {response.content[0].text}")
except ImportError:
    print("Anthropic SDK not installed, skipping...")

# Gemini (if installed)
try:
    print("\n--- Using Gemini SDK ---")
    import google.generativeai as genai

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Say 'Hello from Gemini SDK'")
    print(f"Gemini SDK: {response.text}")
except ImportError:
    print("Gemini SDK not installed, skipping...")

print("\n✓ All requests were redirected to xAI!")
