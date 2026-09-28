"""Base model and parsing helpers."""

import json
from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, dataclass, field, fields
from functools import cache
from types import NoneType, UnionType
from typing import (
    Any,
    ClassVar,
    Final,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    override,
)

from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig, CodeGenerationOption

# Errors raised by `from_dict` for data that does not match a model.
PARSE_ERRORS: Final = (AttributeError, LookupError, TypeError, ValueError)


def _admits_none(annotation: Any) -> bool:
    """Tell whether an annotation accepts `None`."""
    return annotation in (Any, None, NoneType) or (
        get_origin(annotation) is UnionType and NoneType in get_args(annotation)
    )


def _strict_str(value: Any) -> str:
    """Reject non-strings instead of converting them with `str()`."""
    if isinstance(value, str):
        return value
    raise TypeError(f"expected a string, got {type(value).__name__}")


def _strict_bool(value: Any) -> bool:
    """Reject non-booleans instead of converting them with `bool()`."""
    if isinstance(value, bool):
        return value
    raise TypeError(f"expected a boolean, got {type(value).__name__}")


_CODE_GENERATION_OPTIONS: list[CodeGenerationOption] = ["TO_DICT_ADD_OMIT_NONE_FLAG"]
# Serializing strings strictly also makes unions with `str` pack their models.
_SERIALIZATION_STRATEGY: dict[Any, Any] = {
    str: {"serialize": _strict_str, "deserialize": _strict_str},
    bool: {"deserialize": _strict_bool},
}


@dataclass
class OpenResponsesModel(DataClassDictMixin):
    """Lenient base model that keeps unknown fields in `extra`."""

    extra: dict[str, Any] = field(default_factory=dict, kw_only=True, repr=False)

    class Config(BaseConfig):
        """Serialize by alias, omit `None` and check strings and booleans."""

        serialize_by_alias = True
        omit_none = True
        lazy_compilation = True
        code_generation_options = _CODE_GENERATION_OPTIONS
        serialization_strategy = _SERIALIZATION_STRATEGY

    @classmethod
    @override
    def __pre_deserialize__(cls, d: dict[Any, Any]) -> dict[Any, Any]:
        """Move unknown fields to `extra` and drop `null` from fields with defaults."""
        keys, null_defaults = cls._field_keys()
        return {
            **{
                key: value
                for key, value in d.items()
                if key in keys and (value is not None or key not in null_defaults)
            },
            "extra": {key: value for key, value in d.items() if key not in keys},
        }

    @override
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        """Merge `extra` into the serialized fields."""
        extra = d.pop("extra")
        return d | {
            key: value
            for key, value in extra.items()
            if value is not None and key not in d
        }

    @classmethod
    @cache
    def _field_keys(cls) -> tuple[frozenset[str], frozenset[str]]:
        """Return the input keys of the fields and those that must not be `null`."""
        hints = get_type_hints(cls)
        items = {
            item.metadata.get("alias") or item.name: item
            for item in fields(cls)
            if item.name != "extra"
        }
        null_defaults = {
            key
            for key, item in items.items()
            if (item.default is not MISSING or item.default_factory is not MISSING)
            and not _admits_none(hints[item.name])
        }
        return frozenset(items), frozenset(null_defaults)

    def to_json(self, *, omit_none: bool = True) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(omit_none=omit_none), separators=(",", ":"))


class ExtensionFields:
    """Provider-specific fields as attributes, also for type checkers."""

    extra: dict[str, Any]
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __getattr__(self, name: str) -> Any:
        """Return a provider-specific field."""
        try:
            return self.__dict__["extra"][name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from None

    @override
    def __repr__(self) -> str:
        """Show the fields, including the provider-specific ones."""
        values = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if item.repr
        ]
        values += [f"{key}={value!r}" for key, value in self.extra.items()]
        return f"{type(self).__name__}({', '.join(values)})"

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        """Set a field, storing provider-specific fields in `extra`."""
        if name in self.__dataclass_fields__:
            object.__setattr__(self, name, value)
        else:
            self.extra[name] = value


def type_registry[M: OpenResponsesModel](*models: type[M]) -> dict[str, type[M]]:
    """Map the `type` values of models to the models."""
    return {
        cast(str, model.__dataclass_fields__["type"].default): model for model in models
    }


def one_of[M: OpenResponsesModel](
    registry: Mapping[str, type[M]],
    unknown: type[M],
    *,
    missing_type: str = "",
) -> Callable[[Any], M]:
    """Create a parser that picks the model by `type`, falling back to `unknown`."""

    def parse(value: Any) -> M:
        model_type = value.get("type") or (missing_type if "role" in value else "")
        return registry.get(model_type, unknown).from_dict(value)

    return parse


def list_of[T](parse: Callable[[Any], T]) -> Callable[[Any], list[T]]:
    """Create a parser for a list of values."""
    return lambda values: [parse(value) for value in values]


def or_string[T](parse: Callable[[Any], T]) -> Callable[[Any], T | str]:
    """Create a parser that keeps strings as they are."""
    return lambda value: value if isinstance(value, str) else parse(value)
