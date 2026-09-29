"""Tests for the models."""

import json
from copy import deepcopy
from typing import Any

import pytest

from openresponses_client.models import (
    AllowedToolChoice,
    CompactionItem,
    ErrorEvent,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    FunctionToolChoice,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    JsonSchemaResponseFormat,
    Message,
    OutputTextContent,
    ReasoningItem,
    RefusalContent,
    Response,
    ResponseCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseReasoningDeltaEvent,
    ResponseReasoningDoneEvent,
    SummaryTextContent,
    UnknownContent,
    UnknownEvent,
    UnknownItem,
    UnknownTool,
    UrlCitation,
    parse_event,
)

from .fake_server import make_response


def test_parse_complete_response() -> None:
    response = Response.from_dict(make_response())

    assert response.id == "resp_1"
    assert response.status == "completed"
    assert response.output_text == "Hello!"
    assert isinstance(response.output[0], Message)
    assert response.usage is not None
    assert response.usage.total_tokens == 7
    assert response.usage.input_tokens_details is not None
    assert response.text is not None
    assert response.text.format is not None
    assert response.text.format.type == "text"
    assert response.tool_choice == "auto"


def test_response_is_lenient() -> None:
    response = Response.from_dict(
        {"id": "resp_2", "created_at": 1.9, "output": None, "tools": None, "text": {}}
    )

    assert response.created_at == 1
    assert response.output == []
    assert response.tools == []
    assert response.status is None
    assert response.text is not None
    assert response.text.format is None
    assert response.output_text == ""


def test_extra_fields_are_preserved() -> None:
    data = make_response(prompt_cache_retention="24h", acme_trace={"latency_ms": 12})
    response = Response.from_dict(data)

    assert response.extra == {
        "prompt_cache_retention": "24h",
        "acme_trace": {"latency_ms": 12},
    }
    assert response.to_dict()["acme_trace"] == {"latency_ms": 12}


def test_unknown_item_round_trip() -> None:
    item_data = {
        "id": "ws_1",
        "type": "openai:web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "weather: San Francisco, CA"},
    }
    response = Response.from_dict(make_response(output=[item_data]))

    item = response.output[0]
    assert isinstance(item, UnknownItem)
    assert item.type == "openai:web_search_call"
    assert item.id == "ws_1"
    assert item.to_dict() == item_data


def test_all_item_types() -> None:
    response = Response.from_dict(
        make_response(
            output=[
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Thought"}],
                    "content": [{"type": "reasoning_text", "text": "Raw"}],
                    "encrypted_content": "enc",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_1",
                    "output": [{"type": "input_text", "text": "sunny"}],
                    "status": "completed",
                },
                {"type": "compaction", "id": "cmp_1", "encrypted_content": "enc"},
            ]
        )
    )

    reasoning, call, call_output, compaction = response.output
    assert isinstance(reasoning, ReasoningItem)
    assert reasoning.summary_text == "Thought"
    assert reasoning.reasoning_text == "Raw"
    assert isinstance(call, FunctionCall)
    assert call.parse_arguments() == {"location": "Paris"}
    assert response.function_calls == [call]
    assert response.reasoning_items == [reasoning]
    assert isinstance(call_output, FunctionCallOutput)
    assert isinstance(call_output.output, list)
    assert isinstance(call_output.output[0], InputTextContent)
    assert isinstance(compaction, CompactionItem)


def test_function_call_empty_arguments() -> None:
    assert FunctionCall(name="noop").parse_arguments() == {}
    with pytest.raises(json.JSONDecodeError):
        FunctionCall(name="broken", arguments="{").parse_arguments()


@pytest.mark.parametrize("arguments", ["[]", '"text"', "1", "null"])
def test_function_call_non_object_arguments(arguments: str) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        FunctionCall(name="broken", arguments=arguments).parse_arguments()


@pytest.mark.parametrize(
    ("role", "part_type"),
    [("assistant", OutputTextContent), ("user", InputTextContent)],
)
def test_message_string_content(role: str, part_type: type[Any]) -> None:
    message = Message.from_dict({"type": "message", "role": role, "content": "Hi"})

    assert isinstance(message.content[0], part_type)
    assert message.text == "Hi"


def test_message_without_type_is_parsed_as_message() -> None:
    response = Response.from_dict(
        make_response(output=[{"role": "assistant", "content": "Hi there"}])
    )

    assert isinstance(response.output[0], Message)
    assert response.output_text == "Hi there"


def test_message_parts() -> None:
    message = Message.from_dict(
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "See source",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://example.com",
                            "start_index": 0,
                            "end_index": 3,
                            "title": "Example",
                        },
                        {"type": "acme:note", "note": "x"},
                    ],
                },
                {"type": "refusal", "refusal": "No."},
                {"type": "acme:audio", "data": "..."},
            ],
        }
    )

    text, refusal, unknown = message.content
    assert isinstance(text, OutputTextContent)
    assert isinstance(text.annotations[0], UrlCitation)
    assert text.annotations[1].type == "acme:note"
    assert isinstance(refusal, RefusalContent)
    assert isinstance(unknown, UnknownContent)
    assert message.refusal == "No."
    assert message.phase == "final_answer"
    assert Message(content=[]).refusal is None


def test_input_content_models() -> None:
    message = Message.from_dict(
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                {"type": "input_file", "filename": "a.pdf", "file_data": "AA=="},
            ],
        }
    )

    image, file = message.content
    assert isinstance(image, InputImageContent)
    assert image.detail is None
    assert isinstance(file, InputFileContent)
    assert message.to_dict() == {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            {"type": "input_file", "filename": "a.pdf", "file_data": "AA=="},
        ],
    }


def test_tools_and_tool_choice() -> None:
    response = Response.from_dict(
        make_response(
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": None,
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {"type": "acme:search", "index": "docs"},
            ],
            tool_choice={
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        )
    )

    function_tool, hosted_tool = response.tools
    assert isinstance(function_tool, FunctionTool)
    assert isinstance(hosted_tool, UnknownTool)
    assert hosted_tool.to_dict() == {"type": "acme:search", "index": "docs"}
    assert isinstance(response.tool_choice, AllowedToolChoice)
    assert response.tool_choice.tools == [FunctionToolChoice(name="get_weather")]

    forced = Response.from_dict(
        make_response(tool_choice={"type": "function", "name": "get_weather"})
    )
    assert forced.tool_choice == FunctionToolChoice(name="get_weather")


@pytest.mark.parametrize(
    ("model", "name", "value"),
    [
        (Response, "tool_choice", "auto"),
        (Response, "tool_choice", {"type": "function", "name": "get_weather"}),
        (
            Response,
            "tool_choice",
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        ),
        (Response, "tool_choice", {"type": "acme:hosted", "index": "docs"}),
        (FunctionCallOutput, "output", "plain"),
        (FunctionCallOutput, "output", [{"type": "input_text", "text": "hi"}]),
    ],
)
def test_string_or_object_fields_round_trip(
    model: type[Any], name: str, value: Any
) -> None:
    parsed = model.from_dict({name: value})
    dumped = json.loads(parsed.to_json())
    assert dumped[name] == value
    assert model.from_dict(dumped) == parsed


def test_omit_none_false_keeps_nested_none() -> None:
    response = Response.from_dict(
        {"tool_choice": {"type": "allowed_tools", "tools": []}}
    )
    assert response.to_dict(omit_none=False)["tool_choice"]["mode"] is None
    output = FunctionCallOutput.from_dict(
        {"output": [{"type": "input_image", "image_url": "u"}]}
    )
    assert output.to_dict(omit_none=False)["output"][0]["detail"] is None


def test_message_text_ignores_reasoning_parts() -> None:
    message = Message.from_dict(
        {
            "content": [
                {"type": "output_text", "text": "a"},
                {"type": "summary_text", "text": "s"},
                {"type": "reasoning_text", "text": "r"},
                {"type": "text", "text": "b"},
            ]
        }
    )
    assert message.text == "ab"


def test_json_schema_format_alias() -> None:
    response = Response.from_dict(
        make_response(
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "description": None,
                    "schema": {"type": "object"},
                    "strict": True,
                },
                "verbosity": "low",
            }
        )
    )

    assert response.text is not None
    assert isinstance(response.text.format, JsonSchemaResponseFormat)
    assert response.text.format.schema_ == {"type": "object"}
    assert response.text.to_dict() == {
        "format": {
            "type": "json_schema",
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
        },
        "verbosity": "low",
    }


def test_parse_events() -> None:
    delta = parse_event(
        {
            "type": "response.output_text.delta",
            "sequence_number": 10,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": " a",
            "logprobs": [],
            "obfuscation": "Wd6S45xQ7SyQLT",
        }
    )
    assert isinstance(delta, ResponseOutputTextDeltaEvent)
    assert delta.delta == " a"
    assert delta.obfuscation == "Wd6S45xQ7SyQLT"

    added = parse_event(
        {
            "type": "response.output_item.added",
            "sequence_number": 11,
            "output_index": 3,
            "item": {
                "id": "msg_2",
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant",
            },
        }
    )
    assert isinstance(added, ResponseOutputItemAddedEvent)
    assert isinstance(added.item, Message)

    completed = parse_event(
        {
            "type": "response.completed",
            "sequence_number": 12,
            "response": make_response(),
        }
    )
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.output_text == "Hello!"


def test_parse_unknown_event() -> None:
    data = {
        "type": "acme:trace_event",
        "sequence_number": 1,
        "phase": "tool_resolution",
        "latency_ms": 34,
    }
    event = parse_event(data)

    assert isinstance(event, UnknownEvent)
    assert event.sequence_number == 1
    assert event.to_dict() == data


def test_parse_error_events() -> None:
    streaming = parse_event(
        {
            "type": "error",
            "sequence_number": 3,
            "error": {
                "type": "server_error",
                "code": None,
                "message": "Boom",
                "param": None,
            },
        }
    )
    envelope = parse_event(
        {
            "type": "error",
            "status": 400,
            "error": {
                "code": "previous_response_not_found",
                "message": "Previous response with id 'resp_abc' not found.",
                "param": "previous_response_id",
            },
        }
    )

    assert isinstance(streaming, ErrorEvent)
    assert streaming.error.message == "Boom"
    assert streaming.status is None
    assert isinstance(envelope, ErrorEvent)
    assert envelope.status == 400
    assert envelope.error.code == "previous_response_not_found"


def test_summary_content() -> None:
    reasoning = ReasoningItem.from_dict(
        {
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "One"},
                {"type": "summary_text", "text": "Two"},
            ],
            "encrypted_content": None,
        }
    )

    assert all(isinstance(part, SummaryTextContent) for part in reasoning.summary)
    assert reasoning.summary_text == "One\n\nTwo"
    assert reasoning.reasoning_text == ""
    assert reasoning.to_dict() == {
        "type": "reasoning",
        "summary": [
            {"type": "summary_text", "text": "One"},
            {"type": "summary_text", "text": "Two"},
        ],
    }


def test_from_dict_and_to_json() -> None:
    message = Message.from_dict({"type": "message", "role": "user", "content": "Hi"})

    assert json.loads(message.to_json()) == message.to_dict()
    assert message.to_dict(omit_none=False)["id"] is None


def test_flat_error_event() -> None:
    event = parse_event(
        {
            "type": "error",
            "sequence_number": 3,
            "code": "server_error",
            "message": "Upstream failed",
            "param": None,
        }
    )

    assert isinstance(event, ErrorEvent)
    assert (event.error.code, event.error.message) == (
        "server_error",
        "Upstream failed",
    )
    assert event.to_dict() == {
        "type": "error",
        "sequence_number": 3,
        "error": {"code": "server_error", "message": "Upstream failed"},
    }
    assert parse_event({"type": "error", "error": "Plain text"}).to_dict() == {
        "type": "error",
        "error": {"message": "Plain text"},
    }


@pytest.mark.parametrize(
    ("event_type", "event_cls"),
    [
        ("response.reasoning_text.delta", ResponseReasoningDeltaEvent),
        ("response.reasoning_text.done", ResponseReasoningDoneEvent),
    ],
)
def test_openai_reasoning_event_names(event_type: str, event_cls: type[Any]) -> None:
    event = parse_event(
        {
            "type": event_type,
            "sequence_number": 1,
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hmm",
            "text": "Hmm",
        }
    )

    assert type(event) is event_cls
    assert event.type == event_cls.__dataclass_fields__["type"].default


def test_provider_fields_are_attributes() -> None:
    item = UnknownItem.from_dict(
        {
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "completed",
            "result": "aGk=",
            "size": "1024x1024",
        }
    )

    assert (item.result, item.size) == ("aGk=", "1024x1024")
    item.result = None
    assert item.to_dict() == {
        "type": "image_generation_call",
        "id": "ig_1",
        "status": "completed",
        "size": "1024x1024",
    }
    event = parse_event(
        {"type": "response.code_interpreter_call_code.delta", "delta": "print(1)"}
    )
    assert isinstance(event, UnknownEvent)
    assert event.delta == "print(1)"


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (FunctionTool, {"type": "function", "name": "f", "strict": "yes"}),
        (Message, {"type": "message", "role": 5}),
        (Message, {"type": "message", "content": [{"type": "input_text", "text": 1}]}),
        (Response, {"output": [42]}),
        (Response, {"usage": "many"}),
    ],
)
def test_invalid_values_are_rejected(model: type[Any], data: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid value"):
        model.from_dict(data)


def test_null_falls_back_to_default() -> None:
    message = Message.from_dict({"type": "message", "role": None, "content": None})

    assert (message.role, message.content, message.id) == ("assistant", [], None)


def test_provider_field_named_extra() -> None:
    data = {"type": "acme:note", "extra": {"a": 1}, "id": "n_1"}
    item = UnknownItem.from_dict(data)

    assert item.extra == {"extra": {"a": 1}}
    assert item.to_dict() == data


def test_repr_shows_provider_fields() -> None:
    response = Response.from_dict(
        make_response(
            output=[{"type": "acme:search_call", "id": "s_1", "query": "q"}],
            content_filters=[],
        )
    )

    assert "extra" not in repr(response)
    assert repr(response.output[0]) == (
        "UnknownItem(type='acme:search_call', id='s_1', status=None, query='q')"
    )


def test_deepcopy_keeps_provider_fields() -> None:
    item = UnknownItem.from_dict({"type": "acme:note", "tags": ["a"]})

    copied = deepcopy(item)
    copied.tags.append("b")

    assert copied == UnknownItem(type="acme:note", extra={"tags": ["a", "b"]})
    assert item.tags == ["a"]
