"""Rebuild a response from its streaming events."""

from copy import deepcopy
from typing import Literal, Protocol

from .models import (
    ContentPart,
    FunctionCall,
    Item,
    Message,
    OutputTextContent,
    ReasoningItem,
    ReasoningTextContent,
    RefusalContent,
    Response,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseLifecycleEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReasoningDeltaEvent,
    ResponseReasoningDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    StreamingEvent,
    SummaryTextContent,
)

__all__ = ["ResponseAccumulator"]


class _ItemEvent(Protocol):
    """Event that addresses an output item."""

    item_id: str
    output_index: int


type _TextEvent = (
    ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseReasoningDeltaEvent
    | ResponseReasoningDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
)


def _set_at[T](items: list[T], index: int, value: T) -> None:
    """Replace the item at `index`, or append it."""
    if 0 <= index < len(items):
        items[index] = value
    else:
        items.append(value)


def _updated(text: str, event: _TextEvent) -> str:
    """Append the delta of an event, or take the final text of a done event."""
    match event:
        case (
            ResponseOutputTextDoneEvent(text=final)
            | ResponseRefusalDoneEvent(refusal=final)
            | ResponseReasoningDoneEvent(text=final)
            | ResponseReasoningSummaryTextDoneEvent(text=final)
            | ResponseFunctionCallArgumentsDoneEvent(arguments=final)
        ):
            return final
        case _:
            return text + event.delta


class ResponseAccumulator:
    """Rebuild a response snapshot from streaming events."""

    def __init__(self) -> None:
        self._response = Response(status="in_progress")
        self._started = False

    @property
    def response(self) -> Response | None:
        """The current snapshot, or `None` before the response or an item arrived."""
        return self._response if self._started else None

    def add(self, event: StreamingEvent) -> None:
        """Apply an event to the snapshot."""
        match event:
            case ResponseLifecycleEvent():
                self._apply_response(event.response)
            case (
                ResponseOutputItemAddedEvent(item=item)
                | ResponseOutputItemDoneEvent(item=item)
            ) if item is not None:
                self._started = True
                _set_at(self._response.output, event.output_index, deepcopy(item))
            case (
                ResponseContentPartAddedEvent(part=part)
                | ResponseContentPartDoneEvent(part=part)
            ) if part is not None:
                parts = self._parts(event, "content")
                _set_at(parts, event.content_index, deepcopy(part))
            case (
                ResponseReasoningSummaryPartAddedEvent(part=part)
                | ResponseReasoningSummaryPartDoneEvent(part=part)
            ) if part is not None:
                parts = self._parts(event, "summary")
                _set_at(parts, event.summary_index, deepcopy(part))
            case ResponseOutputTextDeltaEvent() | ResponseOutputTextDoneEvent():
                self._apply_output_text(event)
            case ResponseOutputTextAnnotationAddedEvent(annotation=annotation) if (
                annotation is not None
            ):
                text = self._part(
                    event, "content", event.content_index, OutputTextContent
                )
                _set_at(text.annotations, event.annotation_index, deepcopy(annotation))
            case ResponseRefusalDeltaEvent() | ResponseRefusalDoneEvent():
                refusal = self._part(
                    event, "content", event.content_index, RefusalContent
                )
                refusal.refusal = _updated(refusal.refusal, event)
            case ResponseReasoningDeltaEvent() | ResponseReasoningDoneEvent():
                reasoning = self._part(
                    event, "content", event.content_index, ReasoningTextContent
                )
                reasoning.text = _updated(reasoning.text, event)
            case (
                ResponseReasoningSummaryTextDeltaEvent()
                | ResponseReasoningSummaryTextDoneEvent()
            ):
                summary = self._part(
                    event, "summary", event.summary_index, SummaryTextContent
                )
                summary.text = _updated(summary.text, event)
            case (
                ResponseFunctionCallArgumentsDeltaEvent()
                | ResponseFunctionCallArgumentsDoneEvent()
            ):
                self._apply_arguments(event)

    def _apply_response(self, response: Response) -> None:
        """Take the response of a lifecycle event, keeping streamed output."""
        snapshot = deepcopy(response)
        snapshot.output = snapshot.output or self._response.output
        self._response = snapshot
        self._started = True

    def _apply_output_text(
        self, event: ResponseOutputTextDeltaEvent | ResponseOutputTextDoneEvent
    ) -> None:
        part = self._part(event, "content", event.content_index, OutputTextContent)
        part.text = _updated(part.text, event)
        match event:
            case ResponseOutputTextDeltaEvent(logprobs=logprobs) if logprobs:
                part.logprobs = [*(part.logprobs or []), *deepcopy(logprobs)]
            case ResponseOutputTextDoneEvent(logprobs=logprobs) if logprobs is not None:
                part.logprobs = deepcopy(logprobs)

    def _apply_arguments(
        self,
        event: ResponseFunctionCallArgumentsDeltaEvent
        | ResponseFunctionCallArgumentsDoneEvent,
    ) -> None:
        match self._item(event):
            case FunctionCall() as call:
                call.arguments = _updated(call.arguments, event)

    def _item(self, event: _ItemEvent) -> Item | None:
        """Find the item of an event by output index, falling back to its id."""
        output = self._response.output
        if 0 <= event.output_index < len(output):
            item = output[event.output_index]
            if not event.item_id or item.id in (None, event.item_id):
                return item
        return next(
            (item for item in output if event.item_id and item.id == event.item_id),
            None,
        )

    def _parts(
        self, event: _ItemEvent, attribute: Literal["content", "summary"]
    ) -> list[ContentPart]:
        """Return the parts of the event's item; unknown targets give a new list."""
        match self._item(event), attribute:
            case Message(content=parts), "content":
                return parts
            case ReasoningItem(summary=parts), "summary":
                return parts
            case ReasoningItem() as item, "content":
                item.content = item.content or []
                return item.content
        return []

    def _part[P: ContentPart](
        self,
        event: _ItemEvent,
        attribute: Literal["content", "summary"],
        index: int,
        kind: type[P],
    ) -> P:
        """Return the part of `kind` at `index`; unknown targets give a new part."""
        parts = self._parts(event, attribute)
        if index == len(parts):
            # The part was not announced; create it from the delta.
            parts.append(kind())
        part = parts[index] if 0 <= index < len(parts) else None
        return part if isinstance(part, kind) else kind()
