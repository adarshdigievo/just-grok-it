#!/usr/bin/env python3
"""
Example: Debug logging with just-grok-it.

This example shows how to enable debug logging to see
which requests are being intercepted and redirected.

Prerequisites:
    pip install just-grok-it openai
    export XAI_API_KEY="your-xai-api-key"
"""

import logging

# Enable debug logging BEFORE importing just_grok_it
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Now import and use just_grok_it
import just_grok_it

print("=" * 60)
print("Patching providers (watch the DEBUG messages):")
print("=" * 60)

results = just_grok_it.all()

print("\n" + "=" * 60)
print("Creating OpenAI client (watch the DEBUG messages):")
print("=" * 60)

from openai import OpenAI

client = OpenAI(api_key="test-key")  # Will show interception log

print("\n" + "=" * 60)
print("Client info:")
print("=" * 60)
print(f"Base URL: {client.base_url}")
print(f"Expected: https://api.x.ai/v1")

print("\n✓ Debug logging shows all interception points!")
print("\nTip: Use logging.DEBUG level in production to troubleshoot")
print("     redirection issues.")
