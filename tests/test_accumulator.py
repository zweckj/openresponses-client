"""Tests for the response accumulator."""

from typing import Any

from openresponses_client import ResponseAccumulator
from openresponses_client.models import (
    FunctionCall,
    Message,
    OutputTextContent,
    ReasoningItem,
    ReasoningTextContent,
    RefusalContent,
    SummaryTextContent,
    UrlCitation,
    parse_event,
)

from .fake_server import make_response, text_events


def accumulate(events: list[dict[str, Any]]) -> ResponseAccumulator:
    """Feed decoded events into a new accumulator."""
    accumulator = ResponseAccumulator()
    for event in events:
        accumulator.add(parse_event(event))
    return accumulator


def test_text_stream() -> None:
    accumulator = accumulate(text_events(["Hel", "lo", "!"])[:-1])

    snapshot = accumulator.response
    assert snapshot is not None
    assert snapshot.status == "in_progress"
    assert snapshot.output_text == "Hello!"


def test_terminal_event_is_authoritative() -> None:
    events = text_events(["Hel", "lo"])
    events[-1]["response"]["output"][0]["content"][0]["text"] = "Final text"

    snapshot = accumulate(events).response

    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.output_text == "Final text"


def test_terminal_event_without_output_keeps_accumulated_output() -> None:
    events = text_events(["Hel", "lo"], include_output_in_terminal=False)

    snapshot = accumulate(events).response

    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.output_text == "Hello"
    assert snapshot.usage is not None


def test_deltas_without_announced_parts() -> None:
    accumulator = accumulate(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1", "role": "assistant"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 5,
                "delta": "ignored",
            },
        ]
    )

    snapshot = accumulator.response
    assert snapshot is not None
    assert snapshot.output_text == "Hi"


def test_function_call_arguments() -> None:
    accumulator = accumulate(
        [
            {"type": "response.created", "response": make_response(output=[])},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '{"location":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": ' "Paris"}',
            },
        ]
    )
    snapshot = accumulator.response
    assert snapshot is not None
    call = snapshot.output[0]
    assert isinstance(call, FunctionCall)
    assert call.parse_arguments() == {"location": "Paris"}

    accumulator.add(
        parse_event(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": '{"location": "Rome"}',
            }
        )
    )
    assert call.parse_arguments() == {"location": "Rome"}


def test_reasoning_summary_and_content() -> None:
    accumulator = accumulate(
        [
            {"type": "response.created", "response": make_response(output=[])},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {
                "type": "response.reasoning_summary_part.added",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "Plan",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "ning",
            },
            {
                "type": "response.reasoning.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Raw ",
            },
            {
                "type": "response.reasoning.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "thoughts",
            },
        ]
    )
    snapshot = accumulator.response
    assert snapshot is not None
    reasoning = snapshot.output[0]
    assert isinstance(reasoning, ReasoningItem)
    assert reasoning.summary_text == "Planning"
    assert reasoning.reasoning_text == "Raw thoughts"
    assert isinstance(reasoning.content, list)
    assert isinstance(reasoning.content[0], ReasoningTextContent)

    for event in (
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "text": "Final plan",
        },
        {
            "type": "response.reasoning.done",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "text": "Final thoughts",
        },
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 1,
            "part": {"type": "summary_text", "text": "Second"},
        },
    ):
        accumulator.add(parse_event(event))
    assert reasoning.reasoning_text == "Final thoughts"
    assert [
        part.text for part in reasoning.summary if isinstance(part, SummaryTextContent)
    ] == [
        "Final plan",
        "Second",
    ]


def test_refusal_and_annotations() -> None:
    accumulator = accumulate(
        [
            {"type": "response.created", "response": make_response(output=[])},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Source",
                "logprobs": [
                    {
                        "token": "Source",
                        "logprob": -0.1,
                        "bytes": [83],
                        "top_logprobs": [],
                    }
                ],
            },
            {
                "type": "response.output_text.annotation.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "annotation_index": 0,
                "annotation": {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "start_index": 0,
                    "end_index": 6,
                    "title": "Example",
                },
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 1,
                "part": {"type": "refusal", "refusal": ""},
            },
            {
                "type": "response.refusal.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 1,
                "delta": "I can't",
            },
        ]
    )
    snapshot = accumulator.response
    assert snapshot is not None
    message = snapshot.output[0]
    assert isinstance(message, Message)
    text, refusal = message.content
    assert isinstance(text, OutputTextContent)
    assert isinstance(text.annotations[0], UrlCitation)
    assert text.logprobs is not None
    assert text.logprobs[0].token == "Source"
    assert isinstance(refusal, RefusalContent)
    assert refusal.refusal == "I can't"

    accumulator.add(
        parse_event(
            {
                "type": "response.refusal.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 1,
                "refusal": "I can't help with that.",
            }
        )
    )
    accumulator.add(
        parse_event(
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Source.",
                "logprobs": [],
            }
        )
    )
    assert message.refusal == "I can't help with that."
    assert text.text == "Source."
    assert text.logprobs == []


def test_events_are_not_mutated() -> None:
    added = parse_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [],
            },
        }
    )
    part_added = parse_event(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }
    )
    delta = parse_event(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi",
        }
    )
    accumulator = ResponseAccumulator()
    for event in (added, part_added, delta):
        accumulator.add(event)

    assert added.to_dict()["item"]["content"] == []
    assert part_added.to_dict()["part"]["text"] == ""
    assert accumulator.response is not None
    assert accumulator.response.output_text == "Hi"


def test_unknown_and_mismatched_events_are_ignored() -> None:
    accumulator = accumulate(
        [
            {"type": "acme:trace_event", "sequence_number": 0},
            {
                "type": "response.output_text.delta",
                "item_id": "msg_x",
                "output_index": 0,
                "content_index": 0,
                "delta": "lost",
            },
        ]
    )
    assert accumulator.response is None

    accumulator = accumulate(
        [
            *text_events("Hi")[:3],
            {
                "type": "response.output_text.delta",
                "item_id": "msg_unknown",
                "output_index": 7,
                "content_index": 0,
                "delta": "lost",
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "delta": "not a call",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "not reasoning",
            },
        ]
    )
    assert accumulator.response is not None
    assert accumulator.response.output_text == ""


def test_items_are_matched_by_id() -> None:
    accumulator = accumulate(
        [
            {"type": "response.created", "response": make_response(output=[])},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_a", "role": "assistant"},
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "message", "id": "msg_b", "role": "assistant"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_b",
                "output_index": 0,
                "content_index": 0,
                "delta": "B",
            },
        ]
    )

    assert accumulator.response is not None
    first, second = accumulator.response.messages
    assert first.text == ""
    assert second.text == "B"


def test_openai_reasoning_text_events() -> None:
    accumulator = accumulate(
        [
            {"type": "response.created", "response": make_response(output=[])},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "reasoning_text", "text": ""},
            },
            *(
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": "rs_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                }
                for delta in ("Let me ", "think.")
            ),
        ]
    )
    assert accumulator.response is not None
    reasoning = accumulator.response.output[0]
    assert isinstance(reasoning, ReasoningItem)
    assert reasoning.reasoning_text == "Let me think."

    accumulator.add(
        parse_event(
            {
                "type": "response.reasoning_text.done",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Done thinking.",
            }
        )
    )
    assert reasoning.reasoning_text == "Done thinking."
