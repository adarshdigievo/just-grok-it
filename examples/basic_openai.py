#!/usr/bin/env python3
"""
Basic example: Redirecting OpenAI SDK to xAI.

This example shows how to use just-grok-it to redirect
OpenAI API calls to xAI's Grok models.

Prerequisites:
    pip install just-grok-it openai
    export XAI_API_KEY="your-xai-api-key"
"""

# Step 1: Import and initialize just-grok-it BEFORE importing openai
import just_grok_it

just_grok_it.all()  # Uses default model: grok-4-1-fast-non-reasoning

# Step 2: Use OpenAI SDK as normal - it now connects to xAI!
from openai import OpenAI

client = OpenAI()  # Will use XAI_API_KEY and connect to xAI

# Step 3: Make API calls
response = client.chat.completions.create(
    model="grok-4",  # Optional: specify a Grok model
    messages=[
        {"role": "system", "content": "You are Grok, a helpful AI assistant."},
        {
            "role": "user",
            "content": "What makes you different from other AI assistants?",
        },
    ],
    max_tokens=500,
)

print("Response from Grok:")
print(response.choices[0].message.content)

# Optional: Check what was patched
print(f"\nPatched providers: {just_grok_it.get_patched_providers()}")
