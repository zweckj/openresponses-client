"""Check models and params against the official OpenAPI document."""

import inspect
import json
from dataclasses import fields
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, get_args, is_typeddict

import pytest

from openresponses_client import const, params
from openresponses_client.client import OpenResponsesClient
from openresponses_client.models import (
    EVENT_TYPES,
    AllowedToolChoice,
    CompactResponse,
    ContentPart,
    ErrorEvent,
    ErrorPayload,
    FunctionTool,
    FunctionToolChoice,
    IncompleteDetails,
    InputTokensDetails,
    Item,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    LogProb,
    OutputTokensDetails,
    ReasoningConfig,
    Response,
    ResponseError,
    TextConfig,
    TextResponseFormat,
    TopLogProb,
    UnknownContent,
    UnknownItem,
    UrlCitation,
    Usage,
    parse_event,
)
from openresponses_client.models.base import OpenResponsesModel
from openresponses_client.models.content import CONTENT_TYPES
from openresponses_client.models.items import ITEM_TYPES

SPEC_PATH = Path(__file__).parent / "fixtures" / "openapi.json"


@cache
def spec() -> dict[str, Any]:
    """Return the OpenAPI document."""
    result: dict[str, Any] = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return result


def schemas() -> dict[str, Any]:
    """Return the component schemas."""
    result: dict[str, Any] = spec()["components"]["schemas"]
    return result


def resolve(schema: dict[str, Any]) -> dict[str, Any]:
    """Follow `$ref`s to the schema."""
    while "$ref" in schema:
        schema = schemas()[schema["$ref"].rsplit("/", 1)[-1]]
    return schema


def ref_name(schema: dict[str, Any]) -> str:
    """Return the schema name of a `$ref`."""
    return str(schema["$ref"].rsplit("/", 1)[-1])


def type_value(schema: dict[str, Any]) -> str:
    """Return the `type` constant of a schema."""
    type_schema = resolve(schema)["properties"]["type"]
    for option in [type_schema, *type_schema.get("anyOf", [])]:
        if "enum" in option:
            return str(option["enum"][0])
    raise AssertionError(f"No type enum in {schema}")


def field_names(model: type[OpenResponsesModel]) -> set[str]:
    """Return the input keys of a model's fields, using aliases."""
    return {
        item.metadata.get("alias") or item.name
        for item in fields(model)
        if item.name != "extra"
    }


def stream_event_schemas() -> list[str]:
    """Return the schema names of all streaming events."""
    stream = spec()["paths"]["/responses"]["post"]["responses"]["200"]["content"][
        "text/event-stream"
    ]["schema"]
    return [ref_name(option) for option in stream["oneOf"]]


MODEL_SCHEMAS: dict[str, type[OpenResponsesModel]] = {
    "ResponseResource": Response,
    "CompactResource": CompactResponse,
    "Usage": Usage,
    "InputTokensDetails": InputTokensDetails,
    "OutputTokensDetails": OutputTokensDetails,
    "IncompleteDetails": IncompleteDetails,
    "Error": ResponseError,
    "Reasoning": ReasoningConfig,
    "TextField": TextConfig,
    "TextResponseFormat": TextResponseFormat,
    "JsonObjectResponseFormat": JsonObjectResponseFormat,
    "JsonSchemaResponseFormat": JsonSchemaResponseFormat,
    "FunctionTool": FunctionTool,
    "FunctionToolChoice": FunctionToolChoice,
    "AllowedToolChoice": AllowedToolChoice,
    "UrlCitationBody": UrlCitation,
    "LogProb": LogProb,
    "TopLogProb": TopLogProb,
    "ErrorPayload": ErrorPayload,
    "WebSocketErrorEvent": ErrorEvent,
}

ITEM_SCHEMAS = [ref_name(option) for option in schemas()["ItemField"]["oneOf"]]
CONTENT_SCHEMAS = sorted(
    {
        ref_name(option)
        for name in ("Message", "ReasoningBody", "FunctionCallOutput")
        for prop in resolve({"$ref": f"#/components/schemas/{name}"})[
            "properties"
        ].values()
        for candidate in [prop, *prop.get("anyOf", []), prop.get("items", {})]
        for option in [
            *candidate.get("oneOf", []),
            *candidate.get("items", {}).get("oneOf", []),
        ]
        if "$ref" in option
    }
)


def example(schema: dict[str, Any], depth: int = 0) -> Any:  # noqa: PLR0911
    """Generate an instance with every property, following the first variant."""
    schema = resolve(schema)
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for part in schema["allOf"]:
            if "$ref" in part or "properties" in part:
                value = example(part, depth)
                if not isinstance(value, dict):
                    return value
                merged.update(value)
        return merged
    for key in ("oneOf", "anyOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"]
            return example(options[0], depth) if options else None
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next(t for t in schema_type if t != "null")
    if schema_type == "object" or "properties" in schema:
        if depth > 6:
            return {}
        return {
            name: example(prop, depth + 1)
            for name, prop in schema.get("properties", {}).items()
            if not prop.get("x-openresponses-disallowed")
        }
    if schema_type == "array":
        return [example(schema.get("items", {}), depth + 1)] if depth <= 6 else []
    return {
        "string": "x",
        "integer": 1,
        "number": 0.5,
        "boolean": True,
        "null": None,
    }.get(str(schema_type), {})


def test_spec_version() -> None:
    assert spec()["info"] == {"title": "Open Responses", "version": "2026-04-24"}
    assert set(spec()["paths"]) == {"/responses", "/responses/compact"}


@pytest.mark.parametrize("schema_name", stream_event_schemas())
def test_streaming_event_models(schema_name: str) -> None:
    schema = schemas()[schema_name]
    event_type = type_value({"$ref": f"#/components/schemas/{schema_name}"})
    model = EVENT_TYPES[event_type]

    assert set(schema["properties"]) <= field_names(model)
    event = parse_event(example(schema))
    assert type(event) is model
    assert event.to_dict()["type"] == event_type


def test_event_types_match_spec() -> None:
    spec_types = {
        type_value({"$ref": f"#/components/schemas/{name}"})
        for name in stream_event_schemas()
    }

    assert set(EVENT_TYPES) == spec_types
    assert {member.value for member in const.EventType} == spec_types


@pytest.mark.parametrize("schema_name", ITEM_SCHEMAS)
def test_item_models(schema_name: str) -> None:
    ref = {"$ref": f"#/components/schemas/{schema_name}"}
    model = ITEM_TYPES[type_value(ref)]

    assert set(schemas()[schema_name]["properties"]) <= field_names(model)
    item = Response.from_dict({"output": [example(ref)]}).output[0]
    assert type(item) is model


@pytest.mark.parametrize("schema_name", CONTENT_SCHEMAS)
def test_content_models(schema_name: str) -> None:
    ref = {"$ref": f"#/components/schemas/{schema_name}"}
    model = CONTENT_TYPES[type_value(ref)]

    assert set(schemas()[schema_name]["properties"]) <= field_names(model)


def test_content_params_are_covered_by_models() -> None:
    for name, schema in schemas().items():
        if not name.endswith(("ContentParam", "ContentParamAutoParam")):
            continue
        model = CONTENT_TYPES[type_value({"$ref": f"#/components/schemas/{name}"})]
        assert set(schema["properties"]) <= field_names(model), name


@pytest.mark.parametrize(
    ("union", "registry", "unknown"),
    [(Item, ITEM_TYPES, UnknownItem), (ContentPart, CONTENT_TYPES, UnknownContent)],
)
def test_registries_match_unions(
    union: Any, registry: dict[str, type[OpenResponsesModel]], unknown: type[Any]
) -> None:
    assert set(get_args(union.__value__)) == {*registry.values(), unknown}
    for type_name, model in registry.items():
        assert model.__dataclass_fields__["type"].default == type_name


@pytest.mark.parametrize(("schema_name", "model"), MODEL_SCHEMAS.items())
def test_resource_models(schema_name: str, model: type[OpenResponsesModel]) -> None:
    schema = schemas()[schema_name]

    assert set(schema["properties"]) <= field_names(model)
    instance = model.from_dict(example(schema))
    assert set(schema.get("required", [])) <= set(instance.to_dict(omit_none=False))


@pytest.mark.parametrize("schema_name", ["ResponseResource", "CompactResource"])
def test_spec_examples(schema_name: str) -> None:
    data = schemas()[schema_name]["example"]
    model = MODEL_SCHEMAS[schema_name]

    instance = model.from_dict(data)

    assert instance.to_dict(omit_none=False).keys() >= data.keys()


PARAM_SCHEMAS: dict[str, Any] = {
    "UserMessageItemParam": params.UserMessageItemParam,
    "SystemMessageItemParam": params.SystemMessageItemParam,
    "DeveloperMessageItemParam": params.DeveloperMessageItemParam,
    "AssistantMessageItemParam": params.AssistantMessageItemParam,
    "FunctionCallItemParam": params.FunctionCallItemParam,
    "FunctionCallOutputItemParam": params.FunctionCallOutputItemParam,
    "ReasoningItemParam": params.ReasoningItemParam,
    "CompactionSummaryItemParam": params.CompactionItemParam,
    "ItemReferenceParam": params.ItemReferenceParam,
    "InputTextContentParam": params.InputTextContentParam,
    "InputImageContentParamAutoParam": params.InputImageContentParam,
    "InputFileContentParam": params.InputFileContentParam,
    "InputVideoContent": params.InputVideoContentParam,
    "OutputTextContentParam": params.OutputTextContentParam,
    "RefusalContentParam": params.RefusalContentParam,
    "ReasoningSummaryContentParam": params.ReasoningSummaryContentParam,
    "UrlCitationParam": params.UrlCitationParam,
    "FunctionToolParam": params.FunctionToolParam,
    "SpecificFunctionParam": params.SpecificFunctionParam,
    "AllowedToolsParam": params.AllowedToolsParam,
    "TextParam": params.TextParam,
    "TextResponseFormat": params.TextResponseFormatParam,
    "JsonObjectResponseFormat": params.JsonObjectResponseFormatParam,
    "JsonSchemaResponseFormatParam": params.JsonSchemaResponseFormatParam,
    "ReasoningParam": params.ReasoningParam,
    "StreamOptionsParam": params.StreamOptionsParam,
}

# Messages may omit `type`, the client adds it. The JSON schema format needs
# the fields that servers require, as in the stricter resource schema.
OPTIONAL_IN_CLIENT = {
    ("UserMessageItemParam", "type"),
    ("SystemMessageItemParam", "type"),
    ("DeveloperMessageItemParam", "type"),
    ("AssistantMessageItemParam", "type"),
}
REQUIRED_IN_CLIENT = {
    ("JsonSchemaResponseFormatParam", "type"),
    ("JsonSchemaResponseFormatParam", "name"),
    ("JsonSchemaResponseFormatParam", "schema"),
}


@pytest.mark.parametrize(("schema_name", "typed_dict"), PARAM_SCHEMAS.items())
def test_param_typed_dicts(schema_name: str, typed_dict: Any) -> None:
    schema = schemas()[schema_name]
    assert is_typeddict(typed_dict)
    keys = typed_dict.__required_keys__ | typed_dict.__optional_keys__

    assert keys == set(schema["properties"])
    spec_required = set(schema.get("required", []))
    expected = {
        key
        for key in keys
        if (key in spec_required and (schema_name, key) not in OPTIONAL_IN_CLIENT)
        or (schema_name, key) in REQUIRED_IN_CLIENT
    }
    assert typed_dict.__required_keys__ == expected


def test_item_params_cover_spec_union() -> None:
    names = {ref_name(option) for option in schemas()["ItemParam"]["oneOf"]}

    assert names <= set(PARAM_SCHEMAS)
    assert len(get_args(params.InputItemParam.__value__)) == len(names)


def test_create_params_match_request_body() -> None:
    body = set(schemas()["CreateResponseBody"]["properties"])

    assert params.StreamResponseParams.__optional_keys__ == body - {"stream"}
    assert params.CreateResponseParams.__optional_keys__ == body - {
        "stream",
        "stream_options",
    }
    assert params.ResponseParams.__optional_keys__ == body - const.WS_DISALLOWED_FIELDS


def test_websocket_event_schema() -> None:
    websocket = spec()["paths"]["/responses"]["x-openresponses-websocket"]
    create_event = schemas()[ref_name(websocket["clientMessage"])]
    disallowed = {
        name
        for name, prop in create_event["allOf"][0]["properties"].items()
        if prop.get("x-openresponses-disallowed")
    }

    assert create_event["allOf"][0]["properties"]["type"]["enum"] == [
        const.WS_RESPONSE_CREATE
    ]
    assert disallowed == const.WS_DISALLOWED_FIELDS
    error = resolve(websocket["errorMessage"])
    assert set(error["properties"]) <= field_names(ErrorEvent)


def test_compact_signature_matches_request_body() -> None:
    body = set(schemas()["CompactResponseMethodPublicBody"]["properties"])
    parameters = set(inspect.signature(OpenResponsesClient.compact).parameters)

    assert body <= parameters
    assert schemas()["CompactResponseMethodPublicBody"]["required"] == ["model"]
    assert (
        inspect.signature(OpenResponsesClient.compact).parameters["model"].default
        is inspect.Parameter.empty
    )


ENUMS: dict[str, type[StrEnum]] = {
    "IncludeEnum": const.Include,
    "MessageRole": const.MessageRole,
    "MessageStatus": const.ItemStatus,
    "FunctionCallStatus": const.ItemStatus,
    "ReasoningEffortEnum": const.ReasoningEffort,
    "ReasoningSummaryEnum": const.ReasoningSummary,
    "ServiceTierEnum": const.ServiceTier,
    "ToolChoiceValueEnum": const.ToolChoiceMode,
    "TruncationEnum": const.Truncation,
    "VerbosityEnum": const.Verbosity,
    "ImageDetail": const.ImageDetail,
}


@pytest.mark.parametrize(("schema_name", "enum"), ENUMS.items())
def test_enums(schema_name: str, enum: type[StrEnum]) -> None:
    assert {member.value for member in enum} == set(schemas()[schema_name]["enum"])


def test_message_phase_enum() -> None:
    phase = schemas()["Message"]["properties"]["phase"]

    assert {member.value for member in const.MessagePhase} == set(phase["enum"])


def test_item_and_content_type_enums() -> None:
    item_types = {
        type_value({"$ref": f"#/components/schemas/{n}"}) for n in ITEM_SCHEMAS
    }
    content_types = {
        type_value({"$ref": f"#/components/schemas/{n}"}) for n in CONTENT_SCHEMAS
    }

    assert item_types | {"item_reference"} == {
        member.value for member in const.ItemType
    }
    assert content_types == {member.value for member in const.ContentType}
