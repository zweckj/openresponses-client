"""Tests for the WebSocket transport."""

import asyncio
import contextlib
import json
import logging
from typing import Any

import pytest
from aiohttp import WSCloseCode
from aiohttp.test_utils import TestServer

from openresponses_client import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AuthenticationError,
    ErrorEvent,
    EventType,
    OpenResponsesClient,
    PreviousResponseNotFoundError,
    ResponseStreamError,
    WebSocketClosedError,
    WebSocketConnectionLimitReachedError,
)
from openresponses_client.models import ResponseOutputTextDeltaEvent

from .conftest import API_KEY
from .fake_server import (
    FakeOpenResponsesServer,
    WsConnection,
    collect,
    make_response,
    text_events,
)

LIMIT_REACHED = {
    "type": "error",
    "status": 400,
    "error": {
        "type": "invalid_request_error",
        "code": "websocket_connection_limit_reached",
        "message": "Responses websocket connection limit reached (60 minutes).",
        "param": None,
    },
}


async def test_single_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async with client.websocket() as ws:
        connected_in_context = ws.connected
        response = await ws.create(
            model="test-model",
            store=False,
            input=[{"role": "user", "content": "Find the bug in fizz_buzz()."}],
            tools=[],
        )

    assert response.output_text == "turn 1"
    assert (connected_in_context, ws.connected, ws.closed) == (True, False, True)
    connection = fake_server.ws_connections[0]
    assert connection.request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert connection.received == [
        {
            "type": "response.create",
            "model": "test-model",
            "store": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Find the bug in fizz_buzz().",
                }
            ],
            "tools": [],
        }
    ]


async def test_stream_events(client: OpenResponsesClient) -> None:
    async with client.websocket() as ws, ws.stream(model="m", input="Hi") as stream:
        events = [event async for event in stream]
        response = await stream.get_final_response()

    assert events[0].type == EventType.RESPONSE_CREATED
    assert events[-1].type == EventType.RESPONSE_COMPLETED
    assert response.output_text == "turn 1"
    assert stream.snapshot is not None
    assert stream.snapshot.output_text == "turn 1"


async def test_sequential_turns_and_continuation(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async with client.websocket() as ws:
        first = await ws.create(model="m", store=False, input="Remember: cobalt.")
        second = await ws.create(
            model="m",
            store=False,
            previous_response_id=first.id,
            input=[{"type": "function_call_output", "call_id": "c", "output": "x"}],
        )
        third = await ws.create(model="m", store=False, input="Unrelated")

    assert first.output_text == "turn 1"
    assert second.output_text == f"turn 2 after {first.id}"
    assert third.output_text == "turn 3"
    assert len(fake_server.ws_connections) == 1
    assert fake_server.ws_connections[0].received[1]["previous_response_id"] == first.id


async def test_previous_response_not_found(client: OpenResponsesClient) -> None:
    async with client.websocket() as ws:
        with pytest.raises(PreviousResponseNotFoundError) as exc_info:
            await ws.create(
                model="m", store=False, previous_response_id="resp_missing", input="Hi"
            )
        # The connection stays usable for a recovery turn.
        recovered = await ws.create(model="m", store=False, input="Start over")

    error = exc_info.value
    assert error.status == 400
    assert error.code == "previous_response_not_found"
    assert error.param == "previous_response_id"
    assert recovered.output_text == "turn 2"


async def test_store_false_chain_does_not_survive_reconnect(
    client: OpenResponsesClient,
) -> None:
    async with client.websocket() as ws:
        first = await ws.create(model="m", store=False, input="Remember: copper.")

    async with client.websocket() as ws:
        with pytest.raises(PreviousResponseNotFoundError):
            await ws.create(
                model="m", store=False, previous_response_id=first.id, input="Continue"
            )
        recovered = await ws.create(
            model="m", store=False, input="The chain was lost. Start again."
        )

    assert recovered.status == "completed"


async def test_concurrent_turns_are_sent_sequentially(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    received_while_busy: list[bool] = []

    async def handler(connection: WsConnection) -> None:
        turn = 0
        while (request := await connection.receive_create()) is not None:
            turn += 1
            events = text_events(
                ["a", "b", "c"],
                response_id=f"resp_{turn}",
                item_id=f"msg_{turn}",
            )
            for event in events:
                await connection.send_events([event])
                await asyncio.sleep(0.001)
            # The next request may only arrive after the terminal event.
            received_while_busy.append(
                len(connection.received) > turn or request["input"] != f"task {turn}"
            )

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        results = await asyncio.gather(
            *(ws.create(model="m", input=f"task {index}") for index in range(1, 4))
        )

    assert [result.id for result in results] == ["resp_1", "resp_2", "resp_3"]
    assert [result.output_text for result in results] == ["abc", "abc", "abc"]
    assert received_while_busy == [False, False, False]


async def test_abandoned_streams(client: OpenResponsesClient) -> None:
    async with client.websocket() as ws:
        closed_stream = ws.stream(model="m", input="closed")
        async with closed_stream:
            async for event in closed_stream:
                if event.type == EventType.OUTPUT_ITEM_ADDED:
                    break

        open_stream = ws.stream(model="m", input="open")
        async for event in open_stream:
            if event.type == EventType.RESPONSE_CREATED:
                break

        third = await ws.create(model="m", input="next")
        # Events of the unclosed stream were buffered while the next turn ran.
        remaining = [event async for event in open_stream]

    assert third.output_text == "turn 3"
    assert remaining[-1].type == EventType.RESPONSE_COMPLETED
    assert (await open_stream.get_final_response()).output_text == "turn 2"
    assert [event async for event in closed_stream] == []


async def test_waiting_turn_can_be_closed_before_it_is_sent(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async with client.websocket() as ws:
        first = ws.stream(model="m", input="first")
        await first.__aenter__()
        queued = ws.stream(model="m", input="queued")
        await queued.__aenter__()
        await queued.close()
        response = await first.get_final_response()
        second = await ws.create(model="m", input="second")

    assert response.output_text == "turn 1"
    assert second.output_text == "turn 2"
    inputs = [message["input"] for message in fake_server.ws_connections[0].received]
    assert inputs == ["first", "second"]


async def test_connection_limit_between_turns(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        request = await connection.receive_create()
        assert request is not None
        index = len(fake_server.ws_connections)
        await connection.send_events(text_events("ok", response_id=f"resp_{index}"))
        await connection.send_events([LIMIT_REACHED])
        await connection.ws.close(code=WSCloseCode.GOING_AWAY)

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        first = await ws.create(model="m", input="one")
        await asyncio.sleep(0.05)
        second = await ws.create(model="m", input="two")

    assert (first.id, second.id) == ("resp_1", "resp_2")
    assert len(fake_server.ws_connections) == 2


async def test_connection_limit_during_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        request = await connection.receive_create()
        assert request is not None
        if len(fake_server.ws_connections) == 1:
            await connection.send_events(text_events("partial")[:3])
            await connection.send_events([LIMIT_REACHED])
            await connection.ws.close()
            return
        await connection.send_events(text_events("fresh"))

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        with pytest.raises(WebSocketConnectionLimitReachedError) as exc_info:
            await ws.create(model="m", input="one")
        response = await ws.create(model="m", input="two")

    assert exc_info.value.code == "websocket_connection_limit_reached"
    assert response.output_text == "fresh"


async def test_unstarted_turn_is_resent_after_close(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        index = len(fake_server.ws_connections)
        await connection.receive_create()
        await connection.send_events(text_events("ok", response_id=f"resp_{index}"))
        if index == 1:
            await connection.ws.close()
            return
        await connection.receive_create()

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        first = await ws.create(model="m", input="one")
        await asyncio.sleep(0.05)
        second = await ws.create(model="m", input="two")

    assert (first.id, second.id) == ("resp_1", "resp_2")
    assert [c.received[0]["input"] for c in fake_server.ws_connections] == [
        "one",
        "two",
    ]


async def test_turn_is_resent_after_failed_send(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_server.ws_close_timeout = 0.05

    async def handler(connection: WsConnection) -> None:
        index = len(fake_server.ws_connections)
        turn = 0
        while await connection.receive_create() is not None:
            turn += 1
            response_id = f"c{index}_t{turn}"
            await connection.send_events(text_events("ok", response_id=response_id))
            if index == 1:
                await connection.send_events([LIMIT_REACHED])
                return

    fake_server.ws_handler = handler
    caplog.set_level(logging.DEBUG, logger="openresponses_client.websocket")

    async with client.websocket() as ws:
        first = await ws.create(model="m", input="one")
        await asyncio.sleep(0.3)
        second = await ws.create(model="m", input="two")
        connected = ws.connected
        third = await ws.create(model="m", input="three")

    assert (first.id, second.id, third.id) == ("c1_t1", "c2_t1", "c2_t2")
    assert connected
    assert len(fake_server.ws_connections) == 2
    assert "Failed to send the request" in caplog.text
    assert caplog.text.count("Resending a turn that the server did not start") == 1


@pytest.mark.parametrize("auto_reconnect", [True, False])
async def test_cancel_while_connecting(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    auto_reconnect: bool,
) -> None:
    fake_server.ws_handshake_delay = 0.2
    ws = client.websocket(auto_reconnect=auto_reconnect)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await ws.create(model="m", input="cancelled")
    second = await ws.create(model="m", input="second")
    await ws.close()

    assert second.status == "completed"
    inputs = [m["input"] for c in fake_server.ws_connections for m in c.received]
    assert inputs == ["second"]


async def test_close_while_connecting(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    fake_server.ws_handshake_delay = 0.1
    ws = client.websocket()

    task = asyncio.create_task(ws.create(model="m", input="never sent"))
    await asyncio.sleep(0.02)
    await ws.close()
    with pytest.raises(WebSocketClosedError, match="closed by the client"):
        await task
    await asyncio.sleep(0.2)

    assert not ws.connected
    assert all(connection.received == [] for connection in fake_server.ws_connections)
    assert all(connection.ws.closed for connection in fake_server.ws_connections)


async def test_turn_is_resent_only_once(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.ws.close()

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        with pytest.raises(WebSocketClosedError):
            await ws.create(model="m", input="one")

    assert len(fake_server.ws_connections) == 2


async def test_auto_reconnect_disabled(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events(text_events("ok"))
        await connection.ws.close()

    fake_server.ws_handler = handler

    async with client.websocket(auto_reconnect=False) as ws:
        await ws.create(model="m", input="one")
        await asyncio.sleep(0.05)
        with pytest.raises(WebSocketClosedError):
            await ws.create(model="m", input="two")
        with pytest.raises(WebSocketClosedError):
            await ws.create(model="m", input="three")

    assert len(fake_server.ws_connections) == 1


async def test_server_closes_during_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events(text_events("partial")[:4])
        await connection.ws.close(code=WSCloseCode.INTERNAL_ERROR)

    fake_server.ws_handler = handler

    received: list[Any] = []
    async with client.websocket() as ws:
        with pytest.raises(WebSocketClosedError) as exc_info:
            await collect(ws.stream(model="m", input="Hi"), received)

    assert len(received) == 4
    assert exc_info.value.close_code == WSCloseCode.INTERNAL_ERROR


MODEL_ERROR = {
    "type": "error",
    "sequence_number": 1,
    "error": {
        "type": "model_error",
        "code": None,
        "message": "The model failed",
        "param": None,
    },
}


def failed_events(response_id: str = "resp_1") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the created and failed events of a failing response."""
    created = make_response(id=response_id, status="in_progress", output=[])
    failed = make_response(
        id=response_id,
        status="failed",
        output=[],
        error={"code": "model_error", "message": "The model failed"},
    )
    return (
        {"type": "response.created", "sequence_number": 0, "response": created},
        {"type": "response.failed", "sequence_number": 3, "response": failed},
    )


async def test_error_is_followed_by_failed_response(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    early_requests: list[Any] = []

    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        created, failed = failed_events()
        partial_item = text_events("partial")[-2]
        await connection.send_events([created, MODEL_ERROR])
        with contextlib.suppress(TimeoutError):
            early_requests.append(await asyncio.wait_for(connection.ws.receive(), 0.2))
        await connection.send_events([partial_item, failed, "[DONE]"])
        await connection.receive_create()
        await connection.send_events(text_events("second", response_id="resp_2"))

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        stream = ws.stream(model="m", input="one")
        events = [event async for event in stream]
        failed_response = await stream.get_final_response()
        second = await ws.create(model="m", input="two")

    assert [event.type for event in events] == [
        EventType.RESPONSE_CREATED,
        EventType.ERROR,
        EventType.OUTPUT_ITEM_DONE,
        EventType.RESPONSE_FAILED,
    ]
    assert early_requests == []
    assert failed_response.status == "failed"
    assert failed_response.error is not None
    assert failed_response.error.message == "The model failed"
    assert stream.error_event is not None
    assert (second.id, second.output_text) == ("resp_2", "second")


async def test_error_without_terminal_event(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openresponses_client.websocket._ERROR_GRACE_PERIOD", 0.1)

    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        created, failed = failed_events()
        await connection.send_events([created, MODEL_ERROR])
        await connection.receive_create()
        # Late messages of the first turn must not end the second one.
        await connection.send_events(
            [failed, "[DONE]", *text_events("second", response_id="resp_2"), "[DONE]"]
        )
        await connection.receive_create()
        await connection.send_events(text_events("third", response_id="resp_3"))

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        stream = ws.stream(model="m", input="one")
        events = [event async for event in stream]
        with pytest.raises(ResponseStreamError, match="The model failed"):
            await stream.get_final_response()
        second = await ws.create(model="m", input="two")
        third = await ws.create(model="m", input="three")

    assert isinstance(events[-1], ErrorEvent)
    assert (second.id, second.output_text) == ("resp_2", "second")
    assert (third.id, third.output_text) == ("resp_3", "third")


@pytest.mark.parametrize("pings", ["client", "server"])
async def test_error_grace_period_with_pings(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    monkeypatch: pytest.MonkeyPatch,
    pings: str,
) -> None:
    monkeypatch.setattr("openresponses_client.websocket._ERROR_GRACE_PERIOD", 0.3)
    if pings == "server":
        fake_server.ws_heartbeat = 0.05

    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        created, _ = failed_events()
        await connection.send_events([created, MODEL_ERROR])
        await connection.receive_create()

    fake_server.ws_handler = handler
    heartbeat = 0.05 if pings == "client" else None

    async with client.websocket(heartbeat=heartbeat) as ws, asyncio.timeout(3):
        with pytest.raises(ResponseStreamError, match="The model failed"):
            await ws.create(model="m", input="one")


async def test_error_before_response_ends_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events([{**MODEL_ERROR, "error": "Invalid request"}])
        await connection.receive_create()
        await connection.send_events(text_events("second", response_id="resp_2"))

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        with pytest.raises(ResponseStreamError, match="Invalid request"):
            await ws.create(model="m", input="one")
        second = await ws.create(model="m", input="two")

    assert second.output_text == "second"


async def test_invalid_messages(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        events = text_events("Hi")
        await connection.send_events(["not json", "[1, 2]", *events[:-1]])
        await connection.ws.send_bytes(json.dumps(events[-1]).encode())
        await connection.receive_create()
        broken = {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": {"not": "a string"},
        }
        second = text_events("unused", response_id="resp_2")
        await connection.send_events([*second[:4], broken, second[-1]])
        await connection.receive_create()
        await connection.send_events(text_events("third", response_id="resp_3"))

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        first = await ws.create(model="m", input="one")
        with pytest.raises(APIResponseValidationError):
            await ws.create(model="m", input="two")
        # The invalid turn was tracked to its end, so this turn is unaffected.
        third = await ws.create(model="m", input="three")

    assert first.output_text == "Hi"
    assert (third.id, third.output_text) == ("resp_3", "third")


async def test_done_message_ends_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events([*text_events("Hi")[:-1], "[DONE]"])

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        with pytest.raises(ResponseStreamError, match="without a terminal"):
            await ws.create(model="m", input="one")


def test_disallowed_parameters(client: OpenResponsesClient) -> None:
    ws = client.websocket()

    with pytest.raises(ValueError, match="stream"):
        ws.stream(model="m", extra_body={"stream": True})
    for name in ("stream", "stream_options", "background"):
        kwargs: dict[str, Any] = {name: True}
        with pytest.raises(ValueError, match=name):
            ws.stream(model="m", **kwargs)


async def test_extra_body_cannot_change_event_type(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async with client.websocket() as ws:
        await ws.create(model="m", input="Hi", extra_body={"type": "x", "acme": 1})

    message = fake_server.ws_connections[0].received[0]
    assert message["type"] == "response.create"
    assert message["acme"] == 1


async def test_handshake_error(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    fake_server.ws_reject_status = 401

    with pytest.raises(AuthenticationError):
        async with client.websocket():
            pass

    ws = client.websocket()
    with pytest.raises(AuthenticationError):
        async with ws.stream(model="m", input="Hi"):
            pass


async def test_connection_refused(unused_tcp_port: int) -> None:
    async with OpenResponsesClient(f"http://127.0.0.1:{unused_tcp_port}/v1") as client:
        with pytest.raises(APIConnectionError, match="Cannot connect"):
            await client.websocket().create(model="m", input="Hi")


async def test_receive_timeout(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await asyncio.sleep(1)

    fake_server.ws_handler = handler

    async with client.websocket(receive_timeout=0.05) as ws:
        with pytest.raises(APITimeoutError):
            await ws.create(model="m", input="Hi")


@pytest.mark.parametrize("pings", ["client", "server"])
async def test_receive_timeout_with_pings(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer, pings: str
) -> None:
    if pings == "server":
        fake_server.ws_heartbeat = 0.05

    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events(text_events("Hi")[:2])
        await connection.receive_create()

    fake_server.ws_handler = handler
    heartbeat = 0.05 if pings == "client" else None

    async with (
        client.websocket(heartbeat=heartbeat, receive_timeout=0.3) as ws,
        asyncio.timeout(3),
    ):
        with pytest.raises(APITimeoutError):
            await ws.create(model="m", input="Hi")


async def test_lazy_connect_and_close_pending_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events(text_events("Hi")[:2])
        await asyncio.sleep(1)

    fake_server.ws_handler = handler
    ws = client.websocket()
    connected_before = ws.connected

    stream = ws.stream(model="m", input="Hi")
    events = []
    async for event in stream:
        events.append(event)
        if len(events) == 2:
            break
    assert (connected_before, ws.connected) == (False, True)

    pending = asyncio.create_task(stream.get_final_response())
    await asyncio.sleep(0.01)
    await ws.close()
    with pytest.raises(WebSocketClosedError, match="closed by the client"):
        await pending
    with pytest.raises(WebSocketClosedError):
        await ws.create(model="m", input="again")
    with pytest.raises(WebSocketClosedError):
        await ws.connect()
    await ws.close()


async def test_close_fails_queued_turns(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await asyncio.sleep(1)

    fake_server.ws_handler = handler

    async with client.websocket() as ws:
        tasks = [
            asyncio.create_task(ws.create(model="m", input=str(index)))
            for index in range(3)
        ]
        await asyncio.sleep(0.05)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(result, WebSocketClosedError) for result in results)
    assert len(fake_server.ws_connections[0].received) == 1


async def test_delta_events(client: OpenResponsesClient) -> None:
    async with client.websocket() as ws:
        deltas = [
            event.delta
            async for event in ws.stream(model="m", input="Hi")
            if isinstance(event, ResponseOutputTextDeltaEvent)
        ]

    assert deltas == ["turn 1"]


async def test_websocket_url(
    server: TestServer, fake_server: FakeOpenResponsesServer
) -> None:
    ws_url = server.make_url("/v1/responses").with_scheme("ws")
    async with OpenResponsesClient(
        "http://invalid.invalid/v1", websocket_url=ws_url
    ) as client:
        assert client.websocket_url == ws_url
        async with client.websocket(extra_headers={"X-Test": "1"}) as ws:
            response = await ws.create(model="m", input="Hi")

    assert response.output_text == "turn 1"
    assert fake_server.ws_connections[0].request.headers["X-Test"] == "1"


def test_server_errors_are_typed(fake_server: FakeOpenResponsesServer) -> None:
    assert set(LIMIT_REACHED) == {"type", "status", "error"}
    assert isinstance(LIMIT_REACHED["error"], dict)
    error: dict[str, Any] = LIMIT_REACHED["error"]
    assert {"code", "message"} <= set(error)


async def test_follows_same_origin_redirect(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    server: TestServer,
) -> None:
    location = server.make_url("/v1/responses").with_scheme("ws")
    fake_server.ws_redirect_location = str(location.with_query({"api-key": "secret"}))

    async with client.websocket() as ws:
        response = await ws.create(model="m", input="Hi")

    assert response.output_text == "turn 1"
    assert fake_server.ws_connections[0].request.query["api-key"] == "secret"


async def test_refuses_foreign_redirect(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    fake_server.ws_redirect_location = "ws://evil.example/v1/responses?api-key=secret"

    with pytest.raises(APIConnectionError, match="Refusing") as exc_info:
        await client.websocket().connect()

    assert "secret" not in str(exc_info.value)
    assert fake_server.ws_connections == []


async def test_too_many_redirects(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    server: TestServer,
) -> None:
    fake_server.ws_redirect_location = str(
        server.make_url("/v1/responses").with_scheme("ws")
    )

    with pytest.raises(APIConnectionError, match="Too many WebSocket redirects"):
        await client.websocket().connect()


async def test_close_reason_in_error(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.ws.close(
            code=WSCloseCode.INTERNAL_ERROR, message=b"WebSocket gateway error"
        )

    fake_server.ws_handler = handler

    with pytest.raises(WebSocketClosedError, match="1011: WebSocket gateway error"):
        await client.websocket().create(model="m", input="Hi")


async def test_flat_error_event_during_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        created, failed = failed_events()
        flat_error = {
            "type": "error",
            "sequence_number": 1,
            "code": "server_error",
            "message": "Upstream failed",
            "param": None,
        }
        await connection.send_events([created, flat_error, failed])

    fake_server.ws_handler = handler

    async with client.websocket() as ws, ws.stream(model="m", input="Hi") as stream:
        response = await stream.get_final_response()

    assert response.status == "failed"
    assert stream.error_event is not None
    assert stream.error_event.error.message == "Upstream failed"


async def test_oversized_message_fails_the_turn(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(connection: WsConnection) -> None:
        await connection.receive_create()
        await connection.send_events(["x" * 1024])

    fake_server.ws_handler = handler

    async with client.websocket(max_msg_size=64, auto_reconnect=False) as ws:
        with pytest.raises(WebSocketClosedError, match=r"WebSocket error: .*1024"):
            await ws.create(model="m", input="one")
