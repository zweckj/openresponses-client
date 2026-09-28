"""Tests for the server-sent events decoder."""

import json

import pytest

from openresponses_client.sse import ServerSentEvent, SSEDecoder


def decode(*chunks: bytes) -> list[ServerSentEvent]:
    """Decode chunks and flush the decoder."""
    decoder = SSEDecoder()
    events: list[ServerSentEvent] = []
    for chunk in chunks:
        events.extend(decoder.feed(chunk))
    events.extend(decoder.flush())
    return events


def test_basic_events() -> None:
    events = decode(
        b'event: response.created\ndata: {"a": 1}\n\n'
        b'event: response.completed\ndata: {"b": 2}\n\n'
        b"data: [DONE]\n\n"
    )

    assert events == [
        ServerSentEvent(data='{"a": 1}', event="response.created"),
        ServerSentEvent(data='{"b": 2}', event="response.completed"),
        ServerSentEvent(data="[DONE]"),
    ]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_line_endings(newline: bytes) -> None:
    body = newline.join([b"event: a", b"data: 1", b"", b"data: 2", b"", b""])

    assert decode(body) == [
        ServerSentEvent(data="1", event="a"),
        ServerSentEvent(data="2"),
    ]


def test_every_chunk_boundary() -> None:
    body = (
        "\ufeff: comment\r\nevent: response.output_text.delta\r\n"
        'data: {"delta": "caf\u00e9 \u2028 \U0001f600"}\r\n\r\n'
        "data: line one\rdata: line two\r\r"
    ).encode()
    expected = [
        ServerSentEvent(
            data='{"delta": "caf\u00e9 \u2028 \U0001f600"}',
            event="response.output_text.delta",
        ),
        ServerSentEvent(data="line one\nline two"),
    ]

    for split in range(len(body) + 1):
        assert decode(body[:split], body[split:]) == expected, split
    assert decode(*(body[i : i + 1] for i in range(len(body)))) == expected


def test_multiline_data_comments_and_fields() -> None:
    events = decode(
        b": keep-alive\n"
        b"id: 7\n"
        b"retry: 1500\n"
        b"data\n"
        b"data:no-space\n"
        b"data:  two spaces\n"
        b"unknown: ignored\n"
        b"\n"
    )

    assert events == [ServerSentEvent(data="\nno-space\n two spaces")]


def test_empty_events_are_not_dispatched() -> None:
    assert decode(b"event: ping\n\n\n\n") == []


def test_flush_dispatches_unterminated_event() -> None:
    assert decode(b"data: [DONE]") == [ServerSentEvent(data="[DONE]")]
    assert decode(b"data: a\n") == [ServerSentEvent(data="a")]


def test_large_event() -> None:
    payload = json.dumps({"text": "x" * 1_000_000})
    body = f"data: {payload}\n\n".encode()
    chunks = [body[i : i + 65536] for i in range(0, len(body), 65536)]

    events = decode(*chunks)

    assert len(events) == 1
    assert events[0].data == payload


def test_invalid_utf8_is_replaced() -> None:
    assert decode(b"data: \xff\n\n") == [ServerSentEvent(data="\ufffd")]
