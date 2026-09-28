"""Build JSON request bodies."""

import json
from collections.abc import Mapping
from typing import Any


def build_body(
    params: Mapping[str, Any], extra_body: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Drop `None` values, add `type: message` to messages and merge `extra_body`."""
    body = {key: value for key, value in params.items() if value is not None}
    match body:
        case {"input": [*items]}:
            body["input"] = [_with_type(item) for item in items]
    return {**body, **(extra_body or {})}


def _with_type(item: Any) -> Any:
    """Add `type: message` to messages given as role-only mappings."""
    match item:
        case {"type": _}:
            return item
        case {"role": _}:
            return {"type": "message", **item}
    return item


def dumps(body: Any) -> str:
    """Encode a body as JSON, serializing models with `to_dict()`."""
    return json.dumps(body, default=_to_dict)


def _to_dict(value: Any) -> Any:
    try:
        return value.to_dict()
    except AttributeError as err:
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        ) from err
    except (TypeError, ValueError) as err:
        # Models check their fields, for example that `str` fields hold strings.
        raise TypeError(f"Invalid {type(value).__name__}: {err}") from err
