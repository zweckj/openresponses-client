"""Incremental decoder for server-sent events."""

import codecs
import re
from dataclasses import dataclass

__all__ = ["SSEDecoder", "ServerSentEvent"]

_LINE_END = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """Dispatched server-sent event."""

    data: str
    event: str | None = None


class SSEDecoder:
    """Turn bytes into server-sent events, following the WHATWG rules."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial_line: list[str] = []
        self._pending_cr = False
        self._at_start = True
        self._data: list[str] = []
        self._event: str | None = None

    def feed(self, chunk: bytes) -> list[ServerSentEvent]:
        """Decode a chunk and return the completed events."""
        return self._process(self._decoder.decode(chunk))

    def flush(self) -> list[ServerSentEvent]:
        """Process the rest of the input at the end of the stream."""
        # End the last line and the event it belongs to.
        return self._process(self._decoder.decode(b"", final=True) + "\n\n")

    def _process(self, text: str) -> list[ServerSentEvent]:
        """Split decoded text into lines, keeping partial lines."""
        if not text:
            return []
        if self._at_start:
            text = text.removeprefix("\ufeff")
        if self._pending_cr:
            # The "\r" ending the previous chunk already ended the line.
            text = text.removeprefix("\n")
        self._at_start = False
        self._pending_cr = text.endswith("\r")
        *lines, rest = _LINE_END.split(text)
        if lines:
            lines[0] = "".join([*self._partial_line, lines[0]])
            self._partial_line.clear()
        self._partial_line.append(rest)
        return [event for line in lines if (event := self._process_line(line))]

    def _process_line(self, line: str) -> ServerSentEvent | None:
        """Apply one line to the event being built; comments match no field."""
        if not line:
            return self._dispatch()
        field, _, value = line.partition(":")
        match field:
            case "data":
                self._data.append(value.removeprefix(" "))
            case "event":
                self._event = value.removeprefix(" ")
        return None

    def _dispatch(self) -> ServerSentEvent | None:
        """Return the event being built and start a new one."""
        data, self._data = self._data, []
        event, self._event = self._event, None
        return ServerSentEvent("\n".join(data), event or None) if data else None
