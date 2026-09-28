"""In-process Open Responses server for the tests."""

import asyncio
import json
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

HttpHandler = Callable[[web.Request, dict[str, Any]], Awaitable[web.StreamResponse]]
WsHandler = Callable[["WsConnection"], Awaitable[None]]

SPEC_PATH = Path(__file__).parent / "fixtures" / "openapi.json"


@cache
def spec_validator(schema_name: str) -> Draft202012Validator:
    """Return a validator for a schema of the OpenAPI document."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    resource = Resource.from_contents(spec, default_specification=DRAFT202012)
    registry: Registry[Any] = Registry().with_resource("urn:openresponses", resource)
    return Draft202012Validator(
        {"$ref": f"urn:openresponses#/components/schemas/{schema_name}"},
        registry=registry,
    )


def schema_errors(schema_name: str, instance: Any) -> list[str]:
    """Validate an instance against a schema of the spec."""
    return [
        f"{schema_name}: {'/'.join(map(str, error.absolute_path))}: {error.message}"
        for error in spec_validator(schema_name).iter_errors(instance)
    ]


async def collect(events: AsyncIterable[Any], received: list[Any]) -> None:
    """Append events to `received` one by one."""
    async for event in events:
        received.append(event)  # noqa: PERF401


def make_response(**overrides: Any) -> dict[str, Any]:
    """Return a complete, spec compliant response resource."""
    response: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1_764_967_971,
        "completed_at": 1_764_967_972,
        "status": "completed",
        "incomplete_details": None,
        "model": "test-model",
        "previous_response_id": None,
        "instructions": None,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello!", "annotations": []}
                ],
            }
        ],
        "error": None,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": True,
        "text": {"format": {"type": "text"}},
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "temperature": 1,
        "reasoning": None,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": True,
        "background": False,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }
    response.update(overrides)
    return response


def text_events(
    text: str | list[str],
    *,
    response_id: str = "resp_1",
    item_id: str = "msg_1",
    terminal: str = "response.completed",
    include_output_in_terminal: bool = True,
) -> list[dict[str, Any]]:
    """Return the events of a response with one assistant message."""
    deltas = [text] if isinstance(text, str) else text
    full_text = "".join(deltas)
    created = make_response(
        id=response_id, status="in_progress", output=[], completed_at=None, usage=None
    )
    message = {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": full_text, "annotations": []}],
    }
    status = {
        "response.completed": "completed",
        "response.failed": "failed",
        "response.incomplete": "incomplete",
    }[terminal]
    final = make_response(
        id=response_id,
        status=status,
        output=[message] if include_output_in_terminal else [],
    )
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": created},
        {"type": "response.in_progress", "response": created},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": item_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        *(
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "logprobs": [],
            }
            for delta in deltas
        ),
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        },
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": full_text, "annotations": []},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": terminal, "response": final},
    ]
    for sequence_number, event in enumerate(events):
        event["sequence_number"] = sequence_number
    return events


def encode_sse(
    events: Iterable[Mapping[str, Any] | str], *, done: bool = True, names: bool = True
) -> bytes:
    """Encode events as a `text/event-stream` body."""
    chunks = []
    for event in events:
        if isinstance(event, str):
            chunks.append(event)
            continue
        name = f"event: {event['type']}\n" if names and "type" in event else ""
        chunks.append(f"{name}data: {json.dumps(event)}\n\n")
    if done:
        chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode()


async def sse_response(
    request: web.Request,
    body: bytes,
    *,
    chunk_size: int | None = None,
    content_type: str = "text/event-stream",
) -> web.StreamResponse:
    """Write an SSE body, optionally in small chunks."""
    response = web.StreamResponse(headers={"Content-Type": content_type})
    await response.prepare(request)
    size = chunk_size or len(body) or 1
    for start in range(0, len(body), size):
        await response.write(body[start : start + size])
    await response.write_eof()
    return response


@dataclass
class RecordedRequest:
    """HTTP request received by the fake server."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: Any


@dataclass
class WsConnection:
    """Server side of a WebSocket connection."""

    ws: web.WebSocketResponse
    request: web.Request
    schema_errors: list[str]
    received: list[dict[str, Any]] = field(default_factory=list)

    async def receive_create(self) -> dict[str, Any] | None:
        """Wait for the next client message; `None` once closed."""
        message = await self.ws.receive()
        if message.type is not WSMsgType.TEXT:
            return None
        data: dict[str, Any] = json.loads(message.data)
        self.schema_errors.extend(schema_errors("WebSocketResponseCreateEvent", data))
        self.received.append(data)
        return data

    async def send_events(self, events: Iterable[Mapping[str, Any] | str]) -> None:
        """Send events; strings are sent as they are."""
        for event in events:
            await self.ws.send_str(
                event if isinstance(event, str) else json.dumps(event)
            )


class FakeOpenResponsesServer:
    """Configurable Open Responses server."""

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.responses_handler: HttpHandler = self.default_responses_handler
        self.compact_handler: HttpHandler = self.default_compact_handler
        self.ws_handler: WsHandler = self.echo_ws_handler
        self.ws_connections: list[WsConnection] = []
        self.ws_reject_status: int | None = None
        self.ws_redirect_location: str | None = None
        self.ws_handshake_delay = 0.0
        self.ws_close_timeout = 10.0
        self.ws_heartbeat: float | None = None
        # Requests are validated against the specification unless a test
        # deliberately sends implementation-specific data.
        self.validate_requests = True
        self.schema_errors: list[str] = []
        self.app = web.Application()
        self.app.router.add_post("/v1/responses", self._handle_responses)
        self.app.router.add_post("/v1/responses/compact", self._handle_compact)
        self.app.router.add_get("/v1/responses", self._handle_ws)

    @property
    def last_request(self) -> RecordedRequest:
        """Most recent HTTP request."""
        return self.requests[-1]

    async def _record(self, request: web.Request) -> dict[str, Any]:
        raw = await request.text()
        body = json.loads(raw) if raw else None
        self.requests.append(
            RecordedRequest(request.method, request.path, dict(request.headers), body)
        )
        return body if isinstance(body, dict) else {}

    async def _handle_responses(self, request: web.Request) -> web.StreamResponse:
        body = await self._record(request)
        self.schema_errors.extend(schema_errors("CreateResponseBody", body))
        return await self.responses_handler(request, body)

    async def _handle_compact(self, request: web.Request) -> web.StreamResponse:
        body = await self._record(request)
        self.schema_errors.extend(
            schema_errors("CompactResponseMethodPublicBody", body)
        )
        return await self.compact_handler(request, body)

    async def _handle_ws(self, request: web.Request) -> web.StreamResponse:
        if self.ws_redirect_location and "api-key" not in request.query:
            # Like Azure OpenAI, which redirects the handshake to a wss:// URL.
            raise web.HTTPTemporaryRedirect(self.ws_redirect_location)
        if self.ws_handshake_delay:
            await asyncio.sleep(self.ws_handshake_delay)
        if self.ws_reject_status is not None:
            return web.json_response(
                {"error": {"message": "rejected"}}, status=self.ws_reject_status
            )
        ws = web.WebSocketResponse(
            timeout=self.ws_close_timeout, heartbeat=self.ws_heartbeat
        )
        await ws.prepare(request)
        connection = WsConnection(ws, request, self.schema_errors)
        self.ws_connections.append(connection)
        try:
            await self.ws_handler(connection)
        finally:
            await ws.close()
        return ws

    @staticmethod
    async def default_responses_handler(
        request: web.Request, body: dict[str, Any]
    ) -> web.StreamResponse:
        """Answer with a completed response or a short event stream."""
        if body.get("stream"):
            return await sse_response(request, encode_sse(text_events(["Hel", "lo!"])))
        return web.json_response(make_response(model=body.get("model", "test-model")))

    @staticmethod
    async def default_compact_handler(
        request: web.Request, body: dict[str, Any]
    ) -> web.StreamResponse:
        """Answer with a compacted conversation."""
        return web.json_response(
            {
                "id": "cmp_1",
                "object": "response.compaction",
                "created_at": 1_764_967_971,
                "output": [
                    {
                        "id": "msg_0",
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Earlier input"}],
                    },
                    {
                        "id": "cmp_item_1",
                        "type": "compaction",
                        "encrypted_content": "opaque",
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    async def echo_ws_handler(self, connection: WsConnection) -> None:
        """Emulate WebSocket mode with connection-local `store=false` state."""
        cache: set[str] = set()
        turn = 0
        while (request := await connection.receive_create()) is not None:
            turn += 1
            previous = request.get("previous_response_id")
            if previous is not None and previous not in cache:
                await connection.send_events(
                    [
                        {
                            "type": "error",
                            "status": 400,
                            "error": {
                                "type": "invalid_request_error",
                                "code": "previous_response_not_found",
                                "message": (
                                    f"Previous response with id '{previous}' not found."
                                ),
                                "param": "previous_response_id",
                            },
                        }
                    ]
                )
                continue
            response_id = f"resp_{len(self.ws_connections)}_{turn}"
            cache = {response_id}
            text = f"turn {turn}" + (f" after {previous}" if previous else "")
            await connection.send_events(
                text_events(text, response_id=response_id, item_id=f"msg_{turn}")
            )
            await asyncio.sleep(0)
