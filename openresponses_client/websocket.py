"""WebSocket transport, one response at a time per connection."""

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, Self, Unpack, override

import aiohttp

from ._serialization import build_body, dumps
from .const import (
    SSE_DONE,
    WS_DISALLOWED_FIELDS,
    WS_RESPONSE_CREATE,
    ErrorCode,
    EventType,
)
from .exceptions import (
    APIError,
    APIResponseValidationError,
    APITimeoutError,
    WebSocketClosedError,
    describe_client_error,
    make_status_error,
)
from .models import Response, ResponseLifecycleEvent, StreamingEvent, parse_event
from .models.base import PARSE_ERRORS
from .params import ResponseParams
from .streaming import BaseResponseStream

__all__ = ["ResponsesWebSocket", "WebSocketResponseStream"]

_LOGGER = logging.getLogger(__name__)

_STALE_RESPONSE_IDS: Final = 32
# Seconds to wait for `response.failed` after an `error` event.
_ERROR_GRACE_PERIOD = 10.0

type WebSocketConnector = Callable[[], Awaitable[aiohttp.ClientWebSocketResponse]]


@dataclass(eq=False)
class Turn:
    """State of one turn, shared by its stream and the connection."""

    payload: str
    events: deque[StreamingEvent] = field(default_factory=deque)
    error: BaseException | None = None
    started: bool = False
    finished: bool = False
    discarded: bool = False
    resent: bool = False
    response_id: str | None = None
    error_deadline: float | None = None

    def push(self, event: StreamingEvent) -> None:
        """Queue an event unless the stream was closed."""
        self.started = True
        match event:
            case ResponseLifecycleEvent(response=Response(id=response_id)) if (
                response_id
            ):
                self.response_id = response_id
        if not self.discarded and self.error is None:
            self.events.append(event)

    def fail(self, error: BaseException) -> None:
        """Record an error to raise after the queued events."""
        self.started = True
        if not self.discarded and self.error is None:
            self.error = error


class WebSocketResponseStream(BaseResponseStream):
    """Events of one WebSocket turn."""

    def __init__(self, connection: ResponsesWebSocket, payload: str) -> None:
        super().__init__()
        self._connection = connection
        self._turn = Turn(payload)

    @override
    async def _open(self) -> None:
        """Queue the turn on the connection."""
        await self._connection.submit(self._turn)

    @override
    async def _receive(self) -> StreamingEvent:
        """Return queued events, reading from the connection as needed."""
        turn = self._turn
        while True:
            if turn.events:
                return turn.events.popleft()
            if turn.error is not None:
                error, turn.error = turn.error, None
                raise error
            if turn.finished:
                raise StopAsyncIteration
            await self._connection.pump(turn)

    @override
    async def _close(self) -> None:
        """Drop further events of the turn."""
        self._turn.discarded = True
        self._turn.events.clear()
        self._connection.abandon(self._turn)


class ResponsesWebSocket:
    """Persistent WebSocket connection to `/responses`."""

    def __init__(
        self,
        connector: WebSocketConnector,
        *,
        auto_reconnect: bool = True,
        receive_timeout: float | None = None,
    ) -> None:
        self._connector = connector
        self._auto_reconnect = auto_reconnect
        self._receive_timeout = receive_timeout
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._inflight: Turn | None = None
        self._waiting: deque[Turn] = deque()
        self._read_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._connected_once = False
        self._expired = False
        self._close_error: APIError | None = None
        self._stale_response_ids: deque[str] = deque(maxlen=_STALE_RESPONSE_IDS)

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def connected(self) -> bool:
        """Whether the connection is open."""
        return self._ws is not None and not self._ws.closed

    @property
    def closed(self) -> bool:
        """Whether `close()` was called."""
        return self._closed

    async def connect(self) -> None:
        """Open the connection if needed."""
        await self._ensure_connected()

    async def close(self) -> None:
        """Close the connection and fail pending turns."""
        if self._closed:
            return
        self._closed = True
        error = WebSocketClosedError(
            "The WebSocket connection was closed by the client"
        )
        if self._inflight is not None:
            self._end_turn(self._inflight, error)
        self._fail_waiting(error)
        if self._ws is not None:
            ws, self._ws = self._ws, None
            await ws.close()

    def stream(
        self,
        *,
        extra_body: Mapping[str, Any] | None = None,
        **params: Unpack[ResponseParams],
    ) -> WebSocketResponseStream:
        """Start a turn and stream its events."""
        body = build_body(params, extra_body)
        if disallowed := WS_DISALLOWED_FIELDS & body.keys():
            raise ValueError(
                f"Not allowed on WebSockets: {', '.join(sorted(disallowed))}"
            )
        return WebSocketResponseStream(
            self, dumps({**body, "type": WS_RESPONSE_CREATE})
        )

    async def create(
        self,
        *,
        extra_body: Mapping[str, Any] | None = None,
        **params: Unpack[ResponseParams],
    ) -> Response:
        """Run a turn and return its final response."""
        async with self.stream(extra_body=extra_body, **params) as stream:
            return await stream.get_final_response()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise WebSocketClosedError(
                "The WebSocket connection was closed by the client"
            )

    async def _ensure_connected(self) -> aiohttp.ClientWebSocketResponse:
        """Return the open connection, reconnecting if allowed."""
        async with self._connect_lock:
            self._raise_if_closed()
            if self._ws is not None and not self._ws.closed and not self._expired:
                return self._ws
            if self._connected_once and not self._auto_reconnect:
                raise self._close_error or WebSocketClosedError(
                    "The WebSocket connection is closed and auto_reconnect is disabled"
                )
            if self._ws is not None:
                # The server announced the end of the connection lifetime.
                ws, self._ws = self._ws, None
                await ws.close()
                self._raise_if_closed()
            ws = await self._connector()
            if self._closed:
                # close() was called while connecting.
                await ws.close()
                self._raise_if_closed()
            self._ws = ws
            self._connected_once = True
            self._expired = False
            self._close_error = None
            self._stale_response_ids.clear()
            return ws

    async def submit(self, turn: Turn) -> None:
        """Queue a turn and start it if the connection is idle."""
        self._raise_if_closed()
        self._waiting.append(turn)
        if self._inflight is None:
            async with self._read_lock:
                await self._start_next()
        if turn.finished and turn.error is not None:
            # Surface connection errors of the first turn when it is started.
            error, turn.error = turn.error, None
            raise error

    async def _start(self, turn: Turn) -> None:
        """Send a turn, resending it once if the connection died while idle."""
        self._inflight = turn
        try:
            ws = await self._ensure_connected()
        except APIError as err:
            self._end_turn(turn, err)
            return
        except BaseException:
            # Cancelled before anything was sent: keep the turn queued. If its
            # own task was cancelled, closing the stream removes it again.
            self._requeue(turn)
            raise
        if self._inflight is not turn or turn.finished:
            return  # The turn ended while connecting, e.g. by close().
        if turn.discarded:
            self._end_turn(turn)  # Abandoned while connecting; never sent.
            return
        try:
            await ws.send_str(turn.payload)
        except (aiohttp.ClientError, OSError, RuntimeError) as err:
            # The connection died while idle; the server never got the turn.
            _LOGGER.debug("Failed to send the request: %s", describe_client_error(err))
            await self._drop_connection(ws)
            if not self._resend(turn):
                self._end_turn(
                    turn,
                    WebSocketClosedError(
                        f"Failed to send the request: {describe_client_error(err)}"
                    ),
                )
        except BaseException:
            # A partially written frame leaves the connection unusable.
            self._expired = True
            self._requeue(turn)
            raise

    async def _start_next(self) -> None:
        """Start the next queued turn if none is in flight."""
        while self._inflight is None and self._waiting and not self._closed:
            turn = self._waiting.popleft()
            if not turn.finished and not turn.discarded:
                await self._start(turn)

    def _requeue(self, turn: Turn) -> None:
        """Put an unsent turn back at the front of the queue."""
        if self._inflight is turn:
            self._inflight = None
        if not turn.finished and not turn.discarded and turn not in self._waiting:
            self._waiting.appendleft(turn)

    async def _drop_connection(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Forget and close a broken connection."""
        if self._ws is ws:
            self._ws = None
        if not ws.closed:
            await ws.close()

    def _end_turn(self, turn: Turn, error: BaseException | None = None) -> None:
        """Finish a turn, optionally with an error."""
        if error is not None:
            turn.fail(error)
        turn.finished = True
        if turn.response_id:
            self._stale_response_ids.append(turn.response_id)
        if self._inflight is turn:
            self._inflight = None

    def _fail_waiting(self, error: BaseException) -> None:
        """Fail all queued turns."""
        while self._waiting:
            self._end_turn(self._waiting.popleft(), error)

    def abandon(self, turn: Turn) -> None:
        """Remove a closed turn from the queue."""
        # In-flight turns stay tracked until they end; their events are dropped.
        if turn in self._waiting:
            self._waiting.remove(turn)
            turn.finished = True

    async def pump(self, requester: Turn) -> None:
        """Receive one message and route it to the turn in flight."""
        async with self._read_lock:
            if requester.events or requester.finished or requester.error is not None:
                return
            if self._inflight is None:
                await self._start_next()
                if self._inflight is None:
                    if not requester.finished:
                        self._end_turn(
                            requester, WebSocketClosedError("The turn was not sent")
                        )
                    return
            ws = self._ws
            if ws is None or ws.closed:
                await self._connection_lost(self._closed_error(), resend=True)
                return
            turn = self._inflight
            # Absolute deadlines: aiohttp restarts its own receive timeout for
            # every ping and pong frame.
            loop = asyncio.get_running_loop()
            deadline = turn.error_deadline
            if self._receive_timeout is not None:
                message_deadline = loop.time() + self._receive_timeout
                deadline = (
                    message_deadline
                    if deadline is None
                    else min(deadline, message_deadline)
                )
            try:
                async with asyncio.timeout_at(deadline):
                    message = await ws.receive()
            except TimeoutError:
                error_deadline = turn.error_deadline
                if error_deadline is not None and loop.time() >= error_deadline:
                    _LOGGER.debug("No terminal event followed an error, ending turn")
                    await self._complete(turn)
                    return
                await self._connection_lost(
                    APITimeoutError("Timed out waiting for the next WebSocket message"),
                    resend=False,
                )
                return
            except (aiohttp.ClientError, OSError) as err:
                await self._connection_lost(
                    WebSocketClosedError(
                        f"WebSocket error: {describe_client_error(err)}"
                    ),
                    resend=True,
                )
                return
            await self._dispatch(message)

    async def _dispatch(self, message: aiohttp.WSMessage) -> None:
        """Handle a WebSocket message by its type."""
        match message.type:
            case aiohttp.WSMsgType.TEXT:
                await self._handle_text(message.data)
            case aiohttp.WSMsgType.BINARY:
                await self._handle_text(message.data.decode("utf-8", errors="replace"))
            case aiohttp.WSMsgType.ERROR:
                error = self._close_error or WebSocketClosedError(
                    f"WebSocket error: {describe_client_error(message.data)}"
                )
                await self._connection_lost(error, resend=True)
            case aiohttp.WSMsgType.CLOSE:
                await self._connection_lost(
                    self._closed_error(message.extra), resend=True
                )
            case aiohttp.WSMsgType.CLOSING | aiohttp.WSMsgType.CLOSED:
                await self._connection_lost(self._closed_error(), resend=True)

    async def _handle_text(self, data: str) -> None:
        """Route a text message: events, error envelopes and `[DONE]`."""
        turn = self._inflight
        if data.strip() == SSE_DONE:
            # [DONE] only ends a turn that already received messages; otherwise
            # it is a late terminator of the previous turn.
            if turn is not None and turn.started:
                await self._complete(turn)
            return
        try:
            decoded: Any = json.loads(data)
        except ValueError:
            decoded = None
        match decoded:
            case {"type": EventType.ERROR, "status": int() as status}:
                await self._handle_error_envelope(turn, status, decoded)
            case {"response": {"id": response_id}} if self._is_stale(turn, response_id):
                _LOGGER.debug(
                    "Ignoring late %s event of a previous turn", decoded.get("type")
                )
            case {} if turn is not None:
                await self._handle_event(turn, decoded)
            case {}:
                _LOGGER.debug(
                    "Ignoring %s event received outside of a turn", decoded.get("type")
                )
            case _:
                _LOGGER.warning("Ignoring invalid WebSocket message: %.200s", data)

    async def _handle_event(self, turn: Turn, data: dict[str, Any]) -> None:
        """Deliver an event to a turn and end the turn when it is done."""
        event_type = data.get("type")
        try:
            turn.push(parse_event(data))
        except PARSE_ERRORS as err:
            # Keep tracking the turn so later events are not misattributed.
            turn.fail(
                APIResponseValidationError(
                    f"Received an invalid {event_type!r} event: {err}", body=data
                )
            )
        match event_type:
            case (
                EventType.RESPONSE_COMPLETED
                | EventType.RESPONSE_FAILED
                | EventType.RESPONSE_INCOMPLETE
            ):
                await self._complete(turn)
            case EventType.ERROR if turn.response_id is None:
                # The request failed before a response was created.
                await self._complete(turn)
            case EventType.ERROR:
                # As on HTTP streams, response.failed follows the error.
                turn.error_deadline = (
                    asyncio.get_running_loop().time() + _ERROR_GRACE_PERIOD
                )

    async def _handle_error_envelope(
        self, turn: Turn | None, status: int, data: dict[str, Any]
    ) -> None:
        """Raise a WebSocket error envelope in the turn in flight."""
        error = make_status_error(status, data, reason="WebSocket error")
        if error.code == ErrorCode.WEBSOCKET_CONNECTION_LIMIT_REACHED:
            self._expired = True
            self._close_error = error
            if turn is not None and self._resend(turn):
                return
        if turn is None:
            _LOGGER.debug("Received WebSocket error outside of a turn: %s", error)
            return
        turn.fail(error)
        await self._complete(turn)

    def _resend(self, turn: Turn) -> bool:
        """Requeue a turn the server never started, once."""
        if (
            not self._auto_reconnect
            or self._closed
            or turn.started
            or turn.resent
            or turn.discarded
            or self._inflight is not turn
        ):
            return False
        _LOGGER.debug("Resending a turn that the server did not start")
        turn.resent = True
        self._inflight = None
        self._waiting.appendleft(turn)
        return True

    def _is_stale(self, turn: Turn | None, response_id: object) -> bool:
        """Tell whether a response id belongs to a previous turn."""
        return response_id in self._stale_response_ids and (
            turn is None or response_id != turn.response_id
        )

    async def _complete(self, turn: Turn) -> None:
        """End a turn and start the next one."""
        self._end_turn(turn)
        await self._start_next()

    def _closed_error(self, reason: object = None) -> APIError:
        """Build the error for a closed connection."""
        close_code = self._ws.close_code if self._ws is not None else None
        detail = f"code {close_code}: {reason}" if reason else f"code {close_code}"
        return self._close_error or WebSocketClosedError(
            f"The WebSocket connection was closed ({detail})", close_code=close_code
        )

    async def _connection_lost(self, error: APIError, *, resend: bool) -> None:
        """Handle a lost connection by failing or resending the turn in flight."""
        ws, self._ws = self._ws, None
        if ws is not None and not ws.closed:
            await ws.close()
        if (turn := self._inflight) is not None and not (resend and self._resend(turn)):
            self._end_turn(turn, error)
        if self._auto_reconnect and not self._closed:
            await self._start_next()
        else:
            self._close_error = error
            self._fail_waiting(error)
