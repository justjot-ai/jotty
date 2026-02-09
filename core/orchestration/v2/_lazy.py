"""
Lazy Component Descriptor
=========================

Enables zero-cost component declaration on SwarmManager.
Components are only initialized when first accessed.

Usage:
    class SwarmManager:
        planner = LazyComponent(lambda self: SwarmPlanner())
        memory  = LazyComponent(lambda self: SwarmMemory(config=self.config))

    sm = SwarmManager()
    # planner/memory not yet created
    sm.planner.plan(...)  # NOW it creates SwarmPlanner
"""

from typing import TypeVar, Generic, Callable, Any

T = TypeVar("T")


class LazyComponent(Generic[T]):
    """
    Descriptor that defers component creation until first access.

    Thread-safe via Python's GIL for single-writer pattern.
    Stores the created instance on the owning object to avoid
    repeated factory calls.
    """

    def __init__(self, factory: Callable[..., T], attr_name: str = ""):
        self._factory = factory
        self._attr_name = attr_name  # Set by __set_name__

    def __set_name__(self, owner: type, name: str):
        self._attr_name = f"_lazy_{name}"

    def __get__(self, obj: Any, objtype: type = None) -> T:
        if obj is None:
            return self  # type: ignore  # class-level access returns descriptor
        # Check if already created on this instance
        cached = obj.__dict__.get(self._attr_name)
        if cached is not None:
            return cached
        # Create, cache, and return
        instance = self._factory(obj)
        obj.__dict__[self._attr_name] = instance
        return instance

    def __set__(self, obj: Any, value: T):
        """Allow explicit override (e.g., in tests)."""
        obj.__dict__[self._attr_name] = value
