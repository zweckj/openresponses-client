"""Tools and tool choices as reported by the server."""

from dataclasses import dataclass, field
from typing import Any, Literal

from mashumaro import field_options

from .base import ExtensionFields, OpenResponsesModel, list_of, one_of, type_registry

__all__ = [
    "TOOL_CHOICE_TYPES",
    "TOOL_TYPES",
    "AllowedToolChoice",
    "FunctionTool",
    "FunctionToolChoice",
    "Tool",
    "ToolChoice",
    "UnknownTool",
    "UnknownToolChoice",
    "parse_tool",
    "parse_tool_choice",
]


@dataclass
class FunctionTool(OpenResponsesModel):
    """Function the model can call."""

    type: Literal["function"] = "function"
    name: str = ""
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


@dataclass(repr=False)
class UnknownTool(ExtensionFields, OpenResponsesModel):
    """Tool type outside the spec, such as a hosted tool."""

    type: str = ""


type Tool = FunctionTool | UnknownTool
TOOL_TYPES = type_registry(FunctionTool)
parse_tool = one_of(TOOL_TYPES, UnknownTool)


@dataclass
class FunctionToolChoice(OpenResponsesModel):
    """Forces a specific function."""

    type: Literal["function"] = "function"
    name: str | None = None


@dataclass(repr=False)
class UnknownToolChoice(ExtensionFields, OpenResponsesModel):
    """Tool choice type outside the spec."""

    type: str = ""


_parse_allowed_tool = one_of(type_registry(FunctionToolChoice), UnknownToolChoice)


@dataclass
class AllowedToolChoice(OpenResponsesModel):
    """Restricts the tools the model may call."""

    type: Literal["allowed_tools"] = "allowed_tools"
    tools: list[FunctionToolChoice | UnknownToolChoice] = field(
        default_factory=list,
        metadata=field_options(deserialize=list_of(_parse_allowed_tool)),
    )
    mode: str | None = None


type ToolChoice = str | FunctionToolChoice | AllowedToolChoice | UnknownToolChoice
TOOL_CHOICE_TYPES = type_registry(FunctionToolChoice, AllowedToolChoice)
parse_tool_choice = one_of(TOOL_CHOICE_TYPES, UnknownToolChoice)
