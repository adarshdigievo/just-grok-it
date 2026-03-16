#!/usr/bin/env python3
"""
Example: Streaming responses with just-grok-it.

This example shows how to use streaming with the OpenAI SDK
when redirecting to xAI.

Prerequisites:
    pip install just-grok-it openai
    export XAI_API_KEY="your-xai-api-key"
"""

import just_grok_it

just_grok_it.all()

from openai import OpenAI

client = OpenAI()

print("Streaming response from Grok:")
print("-" * 40)

stream = client.chat.completions.create(
    model="grok-4",
    messages=[
        {"role": "system", "content": "You are a storyteller."},
        {"role": "user", "content": "Tell me a very short story about a robot."},
    ],
    stream=True,
    max_tokens=200,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n" + "-" * 40)
print("Streaming complete!")
