"""Constants and enums of the Open Responses spec."""

from enum import StrEnum
from typing import Final

DEFAULT_TIMEOUT: Final = 600.0
DEFAULT_CONNECT_TIMEOUT: Final = 30.0
DEFAULT_MAX_RETRIES: Final = 2
DEFAULT_WS_MAX_MSG_SIZE: Final = 64 * 1024 * 1024

SSE_DONE: Final = "[DONE]"
WS_RESPONSE_CREATE: Final = "response.create"
WS_DISALLOWED_FIELDS: Final = frozenset({"stream", "stream_options", "background"})


class ResponseStatus(StrEnum):
    """Lifecycle states of a response."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class ItemStatus(StrEnum):
    """Lifecycle states of an item."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class ItemType(StrEnum):
    """Item types defined by the spec."""

    MESSAGE = "message"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    REASONING = "reasoning"
    COMPACTION = "compaction"
    ITEM_REFERENCE = "item_reference"


class ContentType(StrEnum):
    """Content part types defined by the spec."""

    INPUT_TEXT = "input_text"
    INPUT_IMAGE = "input_image"
    INPUT_FILE = "input_file"
    INPUT_VIDEO = "input_video"
    OUTPUT_TEXT = "output_text"
    TEXT = "text"
    SUMMARY_TEXT = "summary_text"
    REASONING_TEXT = "reasoning_text"
    REFUSAL = "refusal"


class MessageRole(StrEnum):
    """Roles of message authors."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"


class MessagePhase(StrEnum):
    """Phases of assistant messages."""

    COMMENTARY = "commentary"
    FINAL_ANSWER = "final_answer"


class ToolChoiceMode(StrEnum):
    """Simple `tool_choice` modes."""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class Truncation(StrEnum):
    """Truncation strategies."""

    AUTO = "auto"
    DISABLED = "disabled"


class ServiceTier(StrEnum):
    """Service tiers listed by the spec."""

    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    PRIORITY = "priority"


class ReasoningEffort(StrEnum):
    """Reasoning effort levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningSummary(StrEnum):
    """Reasoning summary modes."""

    CONCISE = "concise"
    DETAILED = "detailed"
    AUTO = "auto"


class Verbosity(StrEnum):
    """Output verbosity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImageDetail(StrEnum):
    """Detail levels of image inputs."""

    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


class Include(StrEnum):
    """Extra output that can be requested with `include`."""

    REASONING_ENCRYPTED_CONTENT = "reasoning.encrypted_content"
    MESSAGE_OUTPUT_TEXT_LOGPROBS = "message.output_text.logprobs"


class ErrorType(StrEnum):
    """Error types listed by the spec."""

    SERVER_ERROR = "server_error"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    MODEL_ERROR = "model_error"
    TOO_MANY_REQUESTS = "too_many_requests"


class ErrorCode(StrEnum):
    """Error codes that clients should handle."""

    PREVIOUS_RESPONSE_NOT_FOUND = "previous_response_not_found"
    WEBSOCKET_CONNECTION_LIMIT_REACHED = "websocket_connection_limit_reached"


class EventType(StrEnum):
    """Streaming event types defined by the spec."""

    RESPONSE_CREATED = "response.created"
    RESPONSE_QUEUED = "response.queued"
    RESPONSE_IN_PROGRESS = "response.in_progress"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"
    RESPONSE_INCOMPLETE = "response.incomplete"
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_DONE = "response.output_text.done"
    OUTPUT_TEXT_ANNOTATION_ADDED = "response.output_text.annotation.added"
    REFUSAL_DELTA = "response.refusal.delta"
    REFUSAL_DONE = "response.refusal.done"
    REASONING_DELTA = "response.reasoning.delta"
    REASONING_DONE = "response.reasoning.done"
    REASONING_SUMMARY_PART_ADDED = "response.reasoning_summary_part.added"
    REASONING_SUMMARY_PART_DONE = "response.reasoning_summary_part.done"
    REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
    REASONING_SUMMARY_TEXT_DONE = "response.reasoning_summary_text.done"
    FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    ERROR = "error"
