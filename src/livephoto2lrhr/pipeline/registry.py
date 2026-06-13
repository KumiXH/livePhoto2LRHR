from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, Callable[[dict[str, Any]], T]] = {}

    def register(self, name: str, factory: Callable[[dict[str, Any]], T]) -> None:
        if name in self._items:
            raise KeyError(f"algorithm already registered: {name}")
        self._items[name] = factory

    def create(self, name: str, config: dict[str, Any]) -> T:
        try:
            factory = self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown algorithm: {name}; available: {available}") from exc
        return factory(config)
