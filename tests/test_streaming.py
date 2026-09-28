"""Tests for HTTP streaming."""

import json
from typing import Any

import pytest
from aiohttp import web

from openresponses_client import (
    APIConnectionError,
    APIResponseValidationError,
    BadRequestError,
    ErrorEvent,
    EventType,
    OpenResponsesClient,
    ResponseStreamError,
    UnknownEvent,
)
from openresponses_client.models import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseOutputTextDeltaEvent,
)
from openresponses_client.streaming import decode_event_data

from .fake_server import (
    FakeOpenResponsesServer,
    collect,
    encode_sse,
    make_response,
    sse_response,
    text_events,
)


def serve_sse(
    fake_server: FakeOpenResponsesServer,
    body: bytes,
    *,
    chunk_size: int | None = None,
    content_type: str = "text/event-stream",
) -> None:
    """Answer stream requests with the given SSE body."""

    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return await sse_response(
            request, body, chunk_size=chunk_size, content_type=content_type
        )

    fake_server.responses_handler = handler


async def test_stream_events(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(fake_server, encode_sse(text_events(["Hel", "lo", "!"])), chunk_size=7)

    async with client.stream(model="test-model", input="Hi") as stream:
        events = [event async for event in stream]
        response = await stream.get_final_response()

    assert [event.type for event in events] == [
        EventType.RESPONSE_CREATED,
        EventType.RESPONSE_IN_PROGRESS,
        EventType.OUTPUT_ITEM_ADDED,
        EventType.CONTENT_PART_ADDED,
        EventType.OUTPUT_TEXT_DELTA,
        EventType.OUTPUT_TEXT_DELTA,
        EventType.OUTPUT_TEXT_DELTA,
        EventType.OUTPUT_TEXT_DONE,
        EventType.CONTENT_PART_DONE,
        EventType.OUTPUT_ITEM_DONE,
        EventType.RESPONSE_COMPLETED,
    ]
    assert isinstance(events[0], ResponseCreatedEvent)
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert [event.sequence_number for event in events] == list(range(11))
    assert response.output_text == "Hello!"
    assert stream.final_response is response
    request = fake_server.last_request
    assert request.body == {"model": "test-model", "input": "Hi", "stream": True}
    assert request.headers["Accept"] == "text/event-stream"


async def test_stream_options(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    stream = client.stream(
        model="m", input="Hi", stream_options={"include_obfuscation": False}
    )
    await stream.until_done()

    assert fake_server.last_request.body["stream_options"] == {
        "include_obfuscation": False
    }


async def test_get_final_response_without_iterating(
    client: OpenResponsesClient,
) -> None:
    response = await client.stream(model="m", input="Hi").get_final_response()

    assert response.output_text == "Hello!"


async def test_text_deltas_and_snapshot(client: OpenResponsesClient) -> None:
    async with client.stream(model="m", input="Hi") as stream:
        deltas = [delta async for delta in stream.text_deltas()]
        snapshot = stream.snapshot

    assert deltas == ["Hel", "lo!"]
    assert snapshot is not None
    assert snapshot.output_text == "Hello!"


async def test_stream_without_event_names_or_done(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(fake_server, encode_sse(text_events("Hi"), done=False, names=False))

    response = await client.stream(model="m", input="Hi").get_final_response()

    assert response.output_text == "Hi"


async def test_event_name_is_used_as_type(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(
        fake_server,
        b"event: response.completed\ndata: "
        + json.dumps({"response": make_response()}).encode()
        + b"\n\n",
    )

    response = await client.stream(model="m", input="Hi").get_final_response()

    assert response.output_text == "Hello!"


async def test_extension_events_are_delivered(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    events = text_events("Hi")
    events.insert(2, {"type": "acme:trace_event", "sequence_number": 99, "phase": "a"})
    events.append({"type": "acme:usage_summary", "sequence_number": 100})
    serve_sse(fake_server, encode_sse([": keep-alive\n\n", *events]))

    async with client.stream(model="m", input="Hi") as stream:
        received = [event async for event in stream]

    unknown = [event for event in received if isinstance(event, UnknownEvent)]
    assert [event.type for event in unknown] == [
        "acme:trace_event",
        "acme:usage_summary",
    ]
    assert unknown[0].to_dict()["phase"] == "a"
    assert stream.final_response is not None
    assert stream.final_response.output_text == "Hi"


async def test_events_after_done_are_ignored(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    body = encode_sse(text_events("Hi")) + encode_sse(
        [{"type": "acme:late", "sequence_number": 1}], done=False
    )
    serve_sse(fake_server, body)

    async with client.stream(model="m", input="Hi") as stream:
        received = [event async for event in stream]

    assert received[-1].type == EventType.RESPONSE_COMPLETED


async def test_error_followed_by_failed_response(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    events = text_events("Partial", terminal="response.failed")
    events.insert(
        -1,
        {
            "type": "error",
            "sequence_number": 50,
            "error": {
                "type": "model_error",
                "code": "model_crashed",
                "message": "The model failed",
                "param": None,
            },
        },
    )
    events[-1]["response"]["error"] = {
        "code": "model_crashed",
        "message": "The model failed",
    }
    serve_sse(fake_server, encode_sse(events))

    async with client.stream(model="m", input="Hi") as stream:
        received = [event async for event in stream]
        response = await stream.get_final_response()

    assert isinstance(received[-2], ErrorEvent)
    assert isinstance(received[-1], ResponseFailedEvent)
    assert stream.error_event is received[-2]
    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "model_crashed"


async def test_error_without_terminal_event(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(
        fake_server,
        encode_sse(
            [
                {
                    "type": "error",
                    "sequence_number": 0,
                    "error": {
                        "type": "too_many_requests",
                        "code": "rate_limit_exceeded",
                        "message": "Slow down",
                        "param": None,
                    },
                }
            ]
        ),
    )

    with pytest.raises(ResponseStreamError, match="Slow down") as exc_info:
        await client.stream(model="m", input="Hi").get_final_response()

    assert exc_info.value.code == "rate_limit_exceeded"
    assert exc_info.value.type == "too_many_requests"
    assert exc_info.value.event is not None


async def test_stream_without_terminal_event(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(fake_server, encode_sse(text_events("Hi")[:-1]))

    with pytest.raises(ResponseStreamError, match="without a terminal"):
        await client.stream(model="m", input="Hi").get_final_response()


async def test_status_error_on_open(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response({"error": {"message": "bad", "code": "x"}}, status=400)

    fake_server.responses_handler = handler

    with pytest.raises(BadRequestError):
        async with client.stream(model="m", input="Hi"):
            pass

    with pytest.raises(BadRequestError):
        async for _event in client.stream(model="m", input="Hi"):
            pass


async def test_invalid_event_is_raised_after_valid_events(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    events = text_events("Hi")
    serve_sse(fake_server, encode_sse([events[0], "data: {not json}\n\n", events[1]]))

    received: list[Any] = []
    with pytest.raises(APIResponseValidationError, match="invalid stream event"):
        await collect(client.stream(model="m", input="Hi"), received)

    assert [event.type for event in received] == [EventType.RESPONSE_CREATED]


@pytest.mark.parametrize(
    "data",
    [
        '["not", "an", "object"]',
        json.dumps({"type": "response.output_text.delta", "delta": {"no": "string"}}),
    ],
)
async def test_invalid_event_payloads(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer, data: str
) -> None:
    serve_sse(fake_server, f"data: {data}\n\n".encode())

    with pytest.raises(APIResponseValidationError):
        await client.stream(model="m", input="Hi").until_done()


async def test_large_events(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    text = "x" * 2_000_000
    serve_sse(fake_server, encode_sse(text_events(text)), chunk_size=65536)

    response = await client.stream(model="m", input="Hi").get_final_response()

    assert response.output_text == text


async def test_connection_lost_while_streaming(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        response.content_length = 10_000
        await response.prepare(request)
        await response.write(encode_sse(text_events("Hi")[:1], done=False))
        assert request.transport is not None
        request.transport.close()
        return response

    fake_server.responses_handler = handler

    received: list[Any] = []
    with pytest.raises(APIConnectionError):
        await collect(client.stream(model="m", input="Hi"), received)
    assert len(received) == 1


async def test_close_early(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(fake_server, encode_sse(text_events(["a"] * 50)))

    async with client.stream(model="m", input="Hi") as stream:
        async for event in stream:
            if isinstance(event, ResponseOutputTextDeltaEvent):
                break
    assert stream.http_response is not None
    assert stream.http_response.closed
    assert [event async for event in stream] == []

    # The connection pool is still usable afterwards.
    fake_server.responses_handler = fake_server.default_responses_handler
    response = await client.create(model="m", input="Hi")
    assert response.output_text == "Hello!"


async def test_flat_error_event(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    serve_sse(
        fake_server,
        encode_sse(
            [
                {
                    "type": "error",
                    "sequence_number": 0,
                    "code": "server_error",
                    "message": "Upstream failed",
                    "param": None,
                }
            ]
        ),
    )

    with pytest.raises(ResponseStreamError, match="Upstream failed") as exc_info:
        await client.stream(model="m", input="Hi").get_final_response()

    assert exc_info.value.code == "server_error"


@pytest.mark.parametrize(
    "data",
    [
        "[1]",
        "42",
        '"text"',
        "null",
        "{not json}",
        '{"type": "response.created", "response": 5}',
    ],
)
def test_invalid_event_data(data: str) -> None:
    with pytest.raises(APIResponseValidationError, match="invalid stream event") as exc:
        decode_event_data(data)
    assert exc.value.body == data


def test_event_type_fallbacks() -> None:
    assert decode_event_data(
        '{"delta": "x"}', event_name="response.output_text.delta"
    ) == ResponseOutputTextDeltaEvent(delta="x")
    assert decode_event_data('{"x": 1}') == UnknownEvent(type="", extra={"x": 1})


async def test_stream_without_done_marker(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    # Some servers end the body without `[DONE]`; blank data is skipped.
    body = encode_sse([*text_events("Hi"), "data:  \n\n"], done=False)
    serve_sse(fake_server, body, chunk_size=7)

    async with client.stream(model="m", input="Hi") as stream:
        events = [event async for event in stream]

    assert events[-1].type == EventType.RESPONSE_COMPLETED
    assert stream.final_response is not None
    assert stream.final_response.output_text == "Hi"
