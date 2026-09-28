"""Tests that annotations resolve at runtime, where PEP 649 evaluates them lazily."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any, TypeAliasType, get_type_hints

import pytest

import openresponses_client

MODULES = [
    openresponses_client.__name__,
    *(
        module.name
        for module in pkgutil.walk_packages(
            openresponses_client.__path__, "openresponses_client."
        )
    ),
]


def _functions(owner: type) -> list[Callable[..., Any]]:
    """Return the functions defined on a class, unwrapping descriptors."""
    functions: list[Callable[..., Any]] = []
    for member in vars(owner).values():
        match member:
            case staticmethod() | classmethod():
                functions.append(inspect.unwrap(member.__func__))
            case property(fget=getter) if getter is not None:
                functions.append(getter)
            case _ if inspect.isfunction(member):
                functions.append(member)
    return functions


@pytest.mark.parametrize("name", MODULES)
def test_annotations_resolve(name: str) -> None:
    module = importlib.import_module(name)
    # Packages only re-export; their names are checked where they are defined.
    defined = [
        value
        for value in vars(module).values()
        if getattr(value, "__module__", None) == name
    ]
    for value in defined:
        match value:
            case TypeAliasType():
                assert value.__value__ is not None
            case type():
                get_type_hints(value)
                for function in _functions(value):
                    inspect.signature(function)
            case _ if inspect.isfunction(value):
                inspect.signature(value)
