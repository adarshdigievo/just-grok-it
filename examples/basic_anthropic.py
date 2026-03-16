#!/usr/bin/env python3
"""
Basic example: Redirecting Anthropic SDK to xAI.

This example shows how to use just-grok-it to redirect
Anthropic API calls to xAI's Grok models.

Prerequisites:
    pip install just-grok-it anthropic
    export XAI_API_KEY="your-xai-api-key"
"""

# Step 1: Import and initialize just-grok-it BEFORE importing anthropic
import just_grok_it

just_grok_it.all()

# Step 2: Use Anthropic SDK as normal - calls are converted and sent to xAI!
from anthropic import Anthropic

client = Anthropic()  # Will use XAI_API_KEY

# Step 3: Make API calls using Anthropic's interface
response = client.messages.create(
    model="claude-3-5-sonnet",  # This will be replaced with Grok model
    max_tokens=500,
    system="You are Grok, a witty and helpful AI assistant.",
    messages=[
        {"role": "user", "content": "Tell me a short joke about AI."},
    ],
)

print("Response from Grok (via Anthropic SDK):")
print(response.content[0].text)

# The response is automatically converted to Anthropic format
print(f"\nResponse type: {response.type}")
print(f"Stop reason: {response.stop_reason}")
print(f"Usage - Input tokens: {response.usage.input_tokens}")
print(f"Usage - Output tokens: {response.usage.output_tokens}")
