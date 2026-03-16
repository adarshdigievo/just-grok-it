#!/usr/bin/env python3
"""
Basic example: Redirecting Google Gemini SDK to xAI.

This example shows how to use just-grok-it to redirect
Google Gemini API calls to xAI's Grok models.

Prerequisites:
    pip install just-grok-it google-generativeai
    export XAI_API_KEY="your-xai-api-key"
"""

# Step 1: Import and initialize just-grok-it BEFORE importing genai
import just_grok_it

just_grok_it.all()

# Step 2: Use Gemini SDK as normal - calls are converted and sent to xAI!
import google.generativeai as genai

# Configure is optional since we're redirecting
model = genai.GenerativeModel("gemini-pro")

# Step 3: Make API calls using Gemini's interface
response = model.generate_content("Explain quantum computing in simple terms.")

print("Response from Grok (via Gemini SDK):")
print(response.text)

# Multi-turn conversation
print("\n--- Multi-turn conversation ---")

chat = model.start_chat()
response1 = chat.send_message("What's the capital of France?")
print(f"Q: What's the capital of France?")
print(f"A: {response1.text}")

# Note: Conversation history requires additional handling
# The generate_content with history is also supported
