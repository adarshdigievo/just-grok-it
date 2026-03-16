#!/usr/bin/env python3
"""
Example: Async client usage with just-grok-it.

This example shows how to use async clients with just-grok-it.

Prerequisites:
    pip install just-grok-it openai
    export XAI_API_KEY="your-xai-api-key"
"""

import asyncio

import just_grok_it

just_grok_it.all()

from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI()

    print("Making async request to Grok...")

    response = await client.chat.completions.create(
        model="grok-4",
        messages=[
            {"role": "user", "content": "What's 2 + 2? Reply with just the number."},
        ],
        max_tokens=10,
    )

    print(f"Response: {response.choices[0].message.content}")

    # Multiple concurrent requests
    print("\nMaking 3 concurrent requests...")

    tasks = [
        client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": f"What's {i} + {i}? Reply briefly."}],
            max_tokens=20,
        )
        for i in range(1, 4)
    ]

    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses, 1):
        print(f"  {i} + {i} = {response.choices[0].message.content.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
