"""Streaming events."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast, override

from mashumaro import field_options

from .base import ExtensionFields, OpenResponsesModel, type_registry
from .content import Annotation, ContentPart, LogProb, parse_annotation, parse_content
from .items import Item, parse_item
from .response import Response

__all__ = [
    "BaseStreamingEvent",
    "ErrorEvent",
    "ErrorPayload",
    "ResponseCompletedEvent",
    "ResponseContentPartAddedEvent",
    "ResponseContentPartDoneEvent",
    "ResponseCreatedEvent",
    "ResponseFailedEvent",
    "ResponseFunctionCallArgumentsDeltaEvent",
    "ResponseFunctionCallArgumentsDoneEvent",
    "ResponseInProgressEvent",
    "ResponseIncompleteEvent",
    "ResponseLifecycleEvent",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseOutputTextAnnotationAddedEvent",
    "ResponseOutputTextDeltaEvent",
    "ResponseOutputTextDoneEvent",
    "ResponseQueuedEvent",
    "ResponseReasoningDeltaEvent",
    "ResponseReasoningDoneEvent",
    "ResponseReasoningSummaryPartAddedEvent",
    "ResponseReasoningSummaryPartDoneEvent",
    "ResponseReasoningSummaryTextDeltaEvent",
    "ResponseReasoningSummaryTextDoneEvent",
    "ResponseRefusalDeltaEvent",
    "ResponseRefusalDoneEvent",
    "StreamingEvent",
    "UnknownEvent",
    "parse_event",
]


_FLAT_ERROR_FIELDS: Final = ("code", "message", "param")

# OpenAI names the raw reasoning events differently than the specification.
_EVENT_TYPE_ALIASES: Final = {
    "response.reasoning_text.delta": "response.reasoning.delta",
    "response.reasoning_text.done": "response.reasoning.done",
}


@dataclass
class BaseStreamingEvent(OpenResponsesModel):
    """Fields shared by all streaming events."""

    type: str
    sequence_number: int | None = None


@dataclass
class ResponseLifecycleEvent(BaseStreamingEvent):
    """Base of the events that carry the full `response`."""

    response: Response = field(default_factory=Response)


@dataclass
class ResponseCreatedEvent(ResponseLifecycleEvent):
    """The response was created."""

    type: Literal["response.created"] = "response.created"


@dataclass
class ResponseQueuedEvent(ResponseLifecycleEvent):
    """The response was queued."""

    type: Literal["response.queued"] = "response.queued"


@dataclass
class ResponseInProgressEvent(ResponseLifecycleEvent):
    """The response is being generated."""

    type: Literal["response.in_progress"] = "response.in_progress"


@dataclass
class ResponseCompletedEvent(ResponseLifecycleEvent):
    """The response finished successfully."""

    type: Literal["response.completed"] = "response.completed"


@dataclass
class ResponseFailedEvent(ResponseLifecycleEvent):
    """The response failed; see `response.error`."""

    type: Literal["response.failed"] = "response.failed"


@dataclass
class ResponseIncompleteEvent(ResponseLifecycleEvent):
    """The response stopped early; see `response.incomplete_details`."""

    type: Literal["response.incomplete"] = "response.incomplete"


@dataclass
class ResponseOutputItemAddedEvent(BaseStreamingEvent):
    """An output item started."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int = 0
    item: Item | None = field(
        default=None, metadata=field_options(deserialize=parse_item)
    )


@dataclass
class ResponseOutputItemDoneEvent(BaseStreamingEvent):
    """An output item is complete."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int = 0
    item: Item | None = field(
        default=None, metadata=field_options(deserialize=parse_item)
    )


@dataclass
class ResponseContentPartAddedEvent(BaseStreamingEvent):
    """A content part of an item started."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    part: ContentPart | None = field(
        default=None, metadata=field_options(deserialize=parse_content)
    )


@dataclass
class ResponseContentPartDoneEvent(BaseStreamingEvent):
    """A content part of an item is complete."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    part: ContentPart | None = field(
        default=None, metadata=field_options(deserialize=parse_content)
    )


@dataclass
class ResponseOutputTextDeltaEvent(BaseStreamingEvent):
    """Text was appended to an output text part."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    delta: str = ""
    logprobs: list[LogProb] | None = None
    obfuscation: str | None = None


@dataclass
class ResponseOutputTextDoneEvent(BaseStreamingEvent):
    """Final text of an output text part."""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    text: str = ""
    logprobs: list[LogProb] | None = None


@dataclass
class ResponseOutputTextAnnotationAddedEvent(BaseStreamingEvent):
    """An annotation, such as a citation, was added to output text."""

    type: Literal["response.output_text.annotation.added"] = (
        "response.output_text.annotation.added"
    )
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    annotation_index: int = 0
    annotation: Annotation | None = field(
        default=None, metadata=field_options(deserialize=parse_annotation)
    )


@dataclass
class ResponseRefusalDeltaEvent(BaseStreamingEvent):
    """Refusal text was appended."""

    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


@dataclass
class ResponseRefusalDoneEvent(BaseStreamingEvent):
    """Final refusal text."""

    type: Literal["response.refusal.done"] = "response.refusal.done"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    refusal: str = ""


@dataclass
class ResponseReasoningDeltaEvent(BaseStreamingEvent):
    """Raw reasoning text was appended."""

    type: Literal["response.reasoning.delta"] = "response.reasoning.delta"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    delta: str = ""
    obfuscation: str | None = None


@dataclass
class ResponseReasoningDoneEvent(BaseStreamingEvent):
    """Final raw reasoning text."""

    type: Literal["response.reasoning.done"] = "response.reasoning.done"
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    text: str = ""


@dataclass
class ResponseReasoningSummaryPartAddedEvent(BaseStreamingEvent):
    """A reasoning summary part started."""

    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )
    item_id: str = ""
    output_index: int = 0
    summary_index: int = 0
    part: ContentPart | None = field(
        default=None, metadata=field_options(deserialize=parse_content)
    )


@dataclass
class ResponseReasoningSummaryPartDoneEvent(BaseStreamingEvent):
    """A reasoning summary part is complete."""

    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )
    item_id: str = ""
    output_index: int = 0
    summary_index: int = 0
    part: ContentPart | None = field(
        default=None, metadata=field_options(deserialize=parse_content)
    )


@dataclass
class ResponseReasoningSummaryTextDeltaEvent(BaseStreamingEvent):
    """Reasoning summary text was appended."""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    item_id: str = ""
    output_index: int = 0
    summary_index: int = 0
    delta: str = ""
    obfuscation: str | None = None


@dataclass
class ResponseReasoningSummaryTextDoneEvent(BaseStreamingEvent):
    """Final text of a reasoning summary part."""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    item_id: str = ""
    output_index: int = 0
    summary_index: int = 0
    text: str = ""


@dataclass
class ResponseFunctionCallArgumentsDeltaEvent(BaseStreamingEvent):
    """Function call arguments were appended."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    item_id: str = ""
    output_index: int = 0
    delta: str = ""
    obfuscation: str | None = None


@dataclass
class ResponseFunctionCallArgumentsDoneEvent(BaseStreamingEvent):
    """Final arguments of a function call."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    item_id: str = ""
    output_index: int = 0
    arguments: str = ""


@dataclass
class ErrorPayload(OpenResponsesModel):
    """Details of an `ErrorEvent`."""

    type: str | None = None
    code: str | None = None
    message: str = ""
    param: str | None = None
    headers: dict[str, str] | None = None


@dataclass
class ErrorEvent(BaseStreamingEvent):
    """An error occurred; WebSocket errors also carry a `status`."""

    type: Literal["error"] = "error"
    error: ErrorPayload = field(default_factory=ErrorPayload)
    status: int | None = None

    @classmethod
    @override
    def __pre_deserialize__(cls, d: dict[Any, Any]) -> dict[Any, Any]:
        """Accept OpenAI's flat errors and plain string errors."""
        match d:
            case {"error": str() as message}:
                d = {**d, "error": {"message": message}}
            case {"message": _} if d.get("error") is None:
                d = {
                    **{k: v for k, v in d.items() if k not in _FLAT_ERROR_FIELDS},
                    "error": {k: d[k] for k in _FLAT_ERROR_FIELDS if k in d},
                }
        return super().__pre_deserialize__(d)


@dataclass(repr=False)
class UnknownEvent(ExtensionFields, BaseStreamingEvent):
    """Event type outside the spec, safe to ignore."""

    type: str = ""


type StreamingEvent = (
    ResponseCreatedEvent
    | ResponseQueuedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseOutputTextAnnotationAddedEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseReasoningDeltaEvent
    | ResponseReasoningDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ErrorEvent
    | UnknownEvent
)

EVENT_TYPES: dict[str, type[BaseStreamingEvent]] = type_registry(
    ResponseCreatedEvent,
    ResponseQueuedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseReasoningDeltaEvent,
    ResponseReasoningDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ErrorEvent,
)


def parse_event(data: Mapping[str, Any]) -> StreamingEvent:
    """Parse a decoded event into its model."""
    event_type = data.get("type", "")
    event_type = _EVENT_TYPE_ALIASES.get(event_type, event_type)
    model = EVENT_TYPES.get(event_type, UnknownEvent)
    return cast(StreamingEvent, model.from_dict({**data, "type": event_type}))
