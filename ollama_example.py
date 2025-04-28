"""Ollama example usage

Run `ollama run phi3:mini` to start the API.
"""

import asyncio
from pprint import pprint

from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

model_client = OllamaChatCompletionClient(model="phi3:mini")

messages = [
    UserMessage(content="What is the value of pi to 10 digits ?", source="user")
]


async def main() -> None:
    response = await model_client.create(messages=messages)
    pprint(response)
    pprint(response.content)
    await model_client.close()


asyncio.run(main())
