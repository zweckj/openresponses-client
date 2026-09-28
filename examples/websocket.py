#!/usr/bin/env python3
"""Hold a multi-turn conversation over one WebSocket.

Usage: python examples/websocket.py
Configure with OPENRESPONSES_BASE_URL, OPENRESPONSES_API_KEY, OPENRESPONSES_MODEL.
"""

import asyncio
import os
import sys

from openresponses_client import (
    APIError,
    InputItem,
    OpenResponsesClient,
    PreviousResponseNotFoundError,
    Response,
    ResponsesWebSocket,
)

MODEL = os.environ.get("OPENRESPONSES_MODEL", "gpt-oss:20b")
QUESTIONS = [
    "Remember the code word: cobalt. Reply with OK.",
    "What is the code word? Reply with only the code word.",
    "Spell the code word backwards.",
]


async def ask(
    ws: ResponsesWebSocket,
    history: list[InputItem],
    previous: Response | None,
    question: str,
) -> Response:
    """Run a turn, resending the full history if the chain was lost."""
    message: InputItem = {"role": "user", "content": question}
    response: Response | None = None
    if previous is not None:
        try:
            response = await ws.create(
                model=MODEL,
                store=False,
                previous_response_id=previous.id,
                input=[message],
            )
        except PreviousResponseNotFoundError:
            print("Connection state was lost, resending the history", file=sys.stderr)
    if response is None:
        response = await ws.create(model=MODEL, store=False, input=[*history, message])
    history.extend([message, *response.output])
    return response


async def run() -> int:
    history: list[InputItem] = []
    previous: Response | None = None
    async with OpenResponsesClient(
        os.environ.get("OPENRESPONSES_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("OPENRESPONSES_API_KEY"),
    ) as client:
        try:
            async with client.websocket() as ws:
                for question in QUESTIONS:
                    previous = await ask(ws, history, previous, question)
                    print(f"> {question}\n{previous.output_text}\n")

                # Turns can be streamed as well.
                async with ws.stream(
                    model=MODEL,
                    store=False,
                    previous_response_id=previous.id if previous else None,
                    input="Now count from 1 to 5.",
                ) as stream:
                    async for delta in stream.text_deltas():
                        print(delta, end="", flush=True)
                print()
        except APIError as err:
            print(f"Request failed: {err}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
