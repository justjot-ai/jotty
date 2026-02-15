"""
Lazy Component Descriptor
=========================

Enables zero-cost component declaration on Orchestrator.
Components are only initialized when first accessed.

Usage:
    class Orchestrator:
        planner = LazyComponent(lambda self: SwarmPlanner())
        memory  = LazyComponent(lambda self: SwarmMemory(config=self.config))

    sm = Orchestrator()
    # planner/memory not yet created
    sm.planner.plan(...)  # NOW it creates SwarmPlanner
"""

import threading
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class LazyComponent(Generic[T]):
    """
    Descriptor that defers component creation until first access.

    Thread-safe via double-checked locking: fast path (no lock) checks
    cache, slow path acquires per-descriptor lock, re-checks, then creates.
    Stores the created instance on the owning object to avoid
    repeated factory calls.
    """

    def __init__(self, factory: Callable[..., T], attr_name: str = "") -> None:
        self._factory = factory
        self._attr_name = attr_name  # Set by __set_name__
        self._lock = threading.Lock()

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = f"_lazy_{name}"

    def __get__(self, obj: Any, objtype: type = None) -> T:
        if obj is None:
            return self  # type: ignore  # class-level access returns descriptor
        # Fast path: check cache without lock
        cached = obj.__dict__.get(self._attr_name)
        if cached is not None:
            return cached
        # Slow path: acquire lock, re-check, then create
        with self._lock:
            cached = obj.__dict__.get(self._attr_name)
            if cached is not None:
                return cached
            instance = self._factory(obj)
            obj.__dict__[self._attr_name] = instance
            return instance

    def __set__(self, obj: Any, value: T) -> Any:
        """Allow explicit override (e.g., in tests)."""
        obj.__dict__[self._attr_name] = value
