"""Response streams over HTTP server-sent events."""

import json
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from types import TracebackType
from typing import Self, override

import aiohttp

from .accumulator import ResponseAccumulator
from .const import SSE_DONE
from .exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    ResponseStreamError,
    describe_client_error,
)
from .models import (
    ErrorEvent,
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputTextDeltaEvent,
    StreamingEvent,
    parse_event,
)
from .models.base import PARSE_ERRORS
from .sse import ServerSentEvent, SSEDecoder

__all__ = ["BaseResponseStream", "ResponseStream"]


def decode_event_data(data: str, *, event_name: str | None = None) -> StreamingEvent:
    """Decode the JSON of a streaming event."""
    try:
        decoded = json.loads(data)
        decoded.setdefault("type", event_name)
        return parse_event(decoded)
    except PARSE_ERRORS as err:
        raise APIResponseValidationError(
            f"Received an invalid stream event {data[:200]!r}: {err}", body=data
        ) from err


class BaseResponseStream(ABC):
    """Async iterator over the events of one response."""

    def __init__(self) -> None:
        self._accumulator = ResponseAccumulator()
        self._final_response: Response | None = None
        self._error_event: ErrorEvent | None = None
        self._opened = False
        self._closed = False

    @abstractmethod
    async def _open(self) -> None:
        """Start the request."""

    @abstractmethod
    async def _receive(self) -> StreamingEvent:
        """Return the next event; raise `StopAsyncIteration` at the end."""

    @abstractmethod
    async def _close(self) -> None:
        """Release the transport without blocking."""

    async def __aenter__(self) -> Self:
        """Start the request; errors of the initial response are raised here."""
        try:
            await self._ensure_open()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> StreamingEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            await self._ensure_open()
            event = await self._receive()
        except BaseException:
            await self.close()
            raise
        self._accumulator.add(event)
        match event:
            case (
                ResponseCompletedEvent()
                | ResponseFailedEvent()
                | ResponseIncompleteEvent()
            ):
                self._final_response = self._accumulator.response
            case ErrorEvent():
                self._error_event = event
        return event

    async def _ensure_open(self) -> None:
        if not self._opened:
            self._opened = True
            await self._open()

    @property
    def snapshot(self) -> Response | None:
        """Response rebuilt from the events so far."""
        return self._accumulator.response

    @property
    def final_response(self) -> Response | None:
        """Final response, once a terminal event arrived."""
        return self._final_response

    @property
    def error_event(self) -> ErrorEvent | None:
        """Last `error` event, if any."""
        return self._error_event

    async def close(self) -> None:
        """Stop the stream and release the transport."""
        if not self._closed:
            self._closed = True
            await self._close()

    async def until_done(self) -> Self:
        """Consume all remaining events."""
        async for _event in self:
            pass
        return self

    async def get_final_response(self) -> Response:
        """Consume the stream and return the final response."""
        async for _event in self:
            if self._final_response is not None:
                break
        await self.close()
        if self._final_response is not None:
            return self._final_response
        if self._error_event is not None:
            raise ResponseStreamError(
                self._error_event.error.message or "The stream ended with an error",
                event=self._error_event,
                body=self._error_event.to_dict(),
            )
        raise ResponseStreamError("The stream ended without a terminal response event")

    async def text_deltas(self) -> AsyncIterator[str]:
        """Iterate over the output text deltas."""
        async for event in self:
            match event:
                case ResponseOutputTextDeltaEvent(delta=delta):
                    yield delta


class ResponseStream(BaseResponseStream):
    """Response streamed over HTTP as server-sent events."""

    _content: aiohttp.StreamReader

    def __init__(self, opener: Callable[[], Awaitable[aiohttp.ClientResponse]]) -> None:
        super().__init__()
        self._opener = opener
        self._http_response: aiohttp.ClientResponse | None = None
        self._decoder = SSEDecoder()
        self._pending: deque[ServerSentEvent] = deque()
        self._eof = False

    @property
    def http_response(self) -> aiohttp.ClientResponse | None:
        """Underlying HTTP response, once the stream is open."""
        return self._http_response

    @override
    async def _open(self) -> None:
        """Send the request."""
        self._http_response = await self._opener()
        self._content = self._http_response.content

    @override
    async def _receive(self) -> StreamingEvent:
        """Decode the next event, reading more data as needed."""
        while True:
            while not self._pending:
                await self._read_chunk()
            sse = self._pending.popleft()
            data = sse.data.strip()
            if data == SSE_DONE:
                raise StopAsyncIteration
            if data:
                return decode_event_data(sse.data, event_name=sse.event)

    async def _read_chunk(self) -> None:
        """Read a chunk and queue the events it completes."""
        if self._eof:
            raise StopAsyncIteration
        try:
            chunk = await self._content.readany()
        except TimeoutError as err:
            raise APITimeoutError(
                "Timed out while reading the response stream"
            ) from err
        except aiohttp.ClientError as err:
            raise APIConnectionError(
                f"Error while reading the response stream: {describe_client_error(err)}"
            ) from err
        self._eof = not chunk
        self._pending.extend(
            self._decoder.feed(chunk) if chunk else self._decoder.flush()
        )

    @override
    async def _close(self) -> None:
        if self._http_response is not None:
            self._http_response.release()
