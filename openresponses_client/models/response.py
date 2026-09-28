"""Response objects returned by the server."""

from dataclasses import dataclass, field
from typing import Any, Literal

from mashumaro import field_options

from .base import (
    ExtensionFields,
    OpenResponsesModel,
    list_of,
    one_of,
    or_string,
    type_registry,
)
from .content import OutputTextContent
from .items import FunctionCall, Item, Message, ReasoningItem, parse_item
from .tools import Tool, ToolChoice, parse_tool, parse_tool_choice

__all__ = [
    "RESPONSE_FORMAT_TYPES",
    "CompactResponse",
    "IncompleteDetails",
    "InputTokensDetails",
    "JsonObjectResponseFormat",
    "JsonSchemaResponseFormat",
    "ModelInfo",
    "ModelList",
    "OutputTokensDetails",
    "ReasoningConfig",
    "Response",
    "ResponseError",
    "ResponseFormat",
    "TextConfig",
    "TextResponseFormat",
    "UnknownResponseFormat",
    "Usage",
    "parse_response_format",
]


@dataclass
class ResponseError(OpenResponsesModel):
    """Error that stopped a response."""

    code: str | None = None
    message: str = ""


@dataclass
class IncompleteDetails(OpenResponsesModel):
    """Why a response is incomplete."""

    reason: str | None = None


@dataclass
class InputTokensDetails(OpenResponsesModel):
    """Breakdown of input tokens."""

    cached_tokens: int = 0


@dataclass
class OutputTokensDetails(OpenResponsesModel):
    """Breakdown of output tokens."""

    reasoning_tokens: int = 0


@dataclass
class Usage(OpenResponsesModel):
    """Token usage of a response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails | None = None
    output_tokens_details: OutputTokensDetails | None = None


@dataclass
class ReasoningConfig(OpenResponsesModel):
    """Reasoning settings used for a response."""

    effort: str | None = None
    summary: str | None = None


@dataclass
class TextResponseFormat(OpenResponsesModel):
    """Plain text output."""

    type: Literal["text"] = "text"


@dataclass
class JsonObjectResponseFormat(OpenResponsesModel):
    """Free-form JSON output."""

    type: Literal["json_object"] = "json_object"


@dataclass
class JsonSchemaResponseFormat(OpenResponsesModel):
    """JSON output matching a schema, available as `schema_`."""

    type: Literal["json_schema"] = "json_schema"
    name: str = ""
    description: str | None = None
    schema_: dict[str, Any] | None = field(
        default=None, metadata=field_options(alias="schema")
    )
    strict: bool | None = None


@dataclass(repr=False)
class UnknownResponseFormat(ExtensionFields, OpenResponsesModel):
    """Response format outside the spec."""

    type: str = ""


type ResponseFormat = (
    TextResponseFormat
    | JsonObjectResponseFormat
    | JsonSchemaResponseFormat
    | UnknownResponseFormat
)
RESPONSE_FORMAT_TYPES = type_registry(
    TextResponseFormat, JsonObjectResponseFormat, JsonSchemaResponseFormat
)
parse_response_format = one_of(RESPONSE_FORMAT_TYPES, UnknownResponseFormat)


@dataclass
class TextConfig(OpenResponsesModel):
    """Text output settings used for a response."""

    format: ResponseFormat | None = field(
        default=None, metadata=field_options(deserialize=parse_response_format)
    )
    verbosity: str | None = None


@dataclass
class Response(OpenResponsesModel):
    """Response object returned by `POST /responses`."""

    id: str = ""
    object: str = "response"
    created_at: int = 0
    completed_at: int | None = None
    status: str | None = None
    incomplete_details: IncompleteDetails | None = None
    model: str = ""
    previous_response_id: str | None = None
    instructions: str | list[Any] | None = None
    output: list[Item] = field(
        default_factory=list, metadata=field_options(deserialize=list_of(parse_item))
    )
    error: ResponseError | None = None
    tools: list[Tool] = field(
        default_factory=list, metadata=field_options(deserialize=list_of(parse_tool))
    )
    tool_choice: ToolChoice | None = field(
        default=None,
        metadata=field_options(deserialize=or_string(parse_tool_choice)),
    )
    truncation: str | None = None
    parallel_tool_calls: bool | None = None
    text: TextConfig | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_logprobs: int | None = None
    temperature: float | None = None
    reasoning: ReasoningConfig | None = None
    usage: Usage | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool | None = None
    background: bool | None = None
    service_tier: str | None = None
    metadata: dict[str, Any] | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None

    @property
    def output_text(self) -> str:
        """Text of all assistant messages."""
        return "".join(
            part.text
            for message in self.messages
            if message.role == "assistant"
            for part in message.content
            if isinstance(part, OutputTextContent)
        )

    @property
    def messages(self) -> list[Message]:
        """All message items."""
        return self._items(Message)

    @property
    def function_calls(self) -> list[FunctionCall]:
        """All function calls."""
        return self._items(FunctionCall)

    @property
    def reasoning_items(self) -> list[ReasoningItem]:
        """All reasoning items."""
        return self._items(ReasoningItem)

    def _items[I: Item](self, kind: type[I]) -> list[I]:
        """Return the output items of one kind."""
        return [item for item in self.output if isinstance(item, kind)]


@dataclass
class CompactResponse(OpenResponsesModel):
    """Compacted conversation; use `output` as the next input."""

    id: str = ""
    object: str = "response.compaction"
    created_at: int = 0
    output: list[Item] = field(
        default_factory=list, metadata=field_options(deserialize=list_of(parse_item))
    )
    usage: Usage | None = None


@dataclass
class ModelInfo(OpenResponsesModel):
    """Model listed by `GET /models`."""

    id: str = ""
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None


@dataclass
class ModelList(OpenResponsesModel):
    """Models listed by `GET /models`."""

    data: list[ModelInfo]
    object: str = "list"
