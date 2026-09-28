"""Content parts of messages and reasoning items."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from mashumaro import field_options

from .base import ExtensionFields, OpenResponsesModel, list_of, one_of, type_registry

__all__ = [
    "ANNOTATION_TYPES",
    "CONTENT_TYPES",
    "Annotation",
    "ContentPart",
    "InputFileContent",
    "InputImageContent",
    "InputTextContent",
    "InputVideoContent",
    "LogProb",
    "OutputTextContent",
    "ReasoningTextContent",
    "RefusalContent",
    "SummaryTextContent",
    "TextContent",
    "TopLogProb",
    "UnknownAnnotation",
    "UnknownContent",
    "UrlCitation",
    "join_text",
    "parse_annotation",
    "parse_content",
]


@dataclass
class UrlCitation(OpenResponsesModel):
    """Citation of a web resource."""

    type: Literal["url_citation"] = "url_citation"
    url: str = ""
    start_index: int = 0
    end_index: int = 0
    title: str = ""


@dataclass(repr=False)
class UnknownAnnotation(ExtensionFields, OpenResponsesModel):
    """Annotation type outside the spec."""

    type: str = ""


type Annotation = UrlCitation | UnknownAnnotation
ANNOTATION_TYPES = type_registry(UrlCitation)
parse_annotation = one_of(ANNOTATION_TYPES, UnknownAnnotation)


@dataclass
class TopLogProb(OpenResponsesModel):
    """One of the most likely tokens at a position."""

    token: str = ""
    logprob: float = 0.0
    bytes: list[int] | None = None


@dataclass
class LogProb(OpenResponsesModel):
    """Log probability of a sampled token."""

    token: str = ""
    logprob: float = 0.0
    bytes: list[int] | None = None
    top_logprobs: list[TopLogProb] = field(default_factory=list)


@dataclass
class InputTextContent(OpenResponsesModel):
    """Text input."""

    type: Literal["input_text"] = "input_text"
    text: str = ""


@dataclass
class InputImageContent(OpenResponsesModel):
    """Image input as URL or data URL."""

    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: str | None = None


@dataclass
class InputFileContent(OpenResponsesModel):
    """File input as URL or base64 data."""

    type: Literal["input_file"] = "input_file"
    filename: str | None = None
    file_data: str | None = None
    file_url: str | None = None


@dataclass
class InputVideoContent(OpenResponsesModel):
    """Video input as URL or data URL."""

    type: Literal["input_video"] = "input_video"
    video_url: str = ""


@dataclass
class OutputTextContent(OpenResponsesModel):
    """Text output of the model."""

    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: list[Annotation] = field(
        default_factory=list,
        metadata=field_options(deserialize=list_of(parse_annotation)),
    )
    logprobs: list[LogProb] | None = None


@dataclass
class TextContent(OpenResponsesModel):
    """Generic text content."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class SummaryTextContent(OpenResponsesModel):
    """Summary of the model's reasoning."""

    type: Literal["summary_text"] = "summary_text"
    text: str = ""


@dataclass
class ReasoningTextContent(OpenResponsesModel):
    """Raw reasoning text."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str = ""


@dataclass
class RefusalContent(OpenResponsesModel):
    """Refusal of the model."""

    type: Literal["refusal"] = "refusal"
    refusal: str = ""


@dataclass(repr=False)
class UnknownContent(ExtensionFields, OpenResponsesModel):
    """Content type outside the spec."""

    type: str = ""


type ContentPart = (
    InputTextContent
    | InputImageContent
    | InputFileContent
    | InputVideoContent
    | OutputTextContent
    | TextContent
    | SummaryTextContent
    | ReasoningTextContent
    | RefusalContent
    | UnknownContent
)
CONTENT_TYPES = type_registry(
    InputTextContent,
    InputImageContent,
    InputFileContent,
    InputVideoContent,
    OutputTextContent,
    TextContent,
    SummaryTextContent,
    ReasoningTextContent,
    RefusalContent,
)
parse_content = one_of(CONTENT_TYPES, UnknownContent)

type TextPart = (
    InputTextContent
    | OutputTextContent
    | TextContent
    | SummaryTextContent
    | ReasoningTextContent
)
MESSAGE_TEXT_PARTS: tuple[type[TextPart], ...] = (
    InputTextContent,
    OutputTextContent,
    TextContent,
)
TEXT_PARTS: tuple[type[TextPart], ...] = (
    *MESSAGE_TEXT_PARTS,
    SummaryTextContent,
    ReasoningTextContent,
)


def join_text(
    parts: Iterable[ContentPart], kinds: tuple[type[TextPart], ...], separator: str = ""
) -> str:
    """Join the text of the parts of the given kinds."""
    return separator.join(part.text for part in parts if isinstance(part, kinds))
