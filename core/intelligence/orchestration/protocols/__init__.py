"""
Protocol Mixins for SwarmIntelligence
======================================

Extracted from the monolithic SwarmIntelligence class for modularity.
These are mixed into SwarmIntelligence at class definition time.

- CoordinationMixin: handoff, auction, coalition, gossip, supervisor
- RoutingMixin: smart routing, load balancing, work stealing
- ResilienceMixin: circuit breakers, failures, backpressure, timeouts
- LifecycleMixin: priority queue, decomposition, scaling, caching, parallel
"""

from .coordination import CoordinationMixin
from .lifecycle import LifecycleMixin
from .resilience import ResilienceMixin
from .routing import RoutingMixin

__all__ = [
    "CoordinationMixin",
    "RoutingMixin",
    "ResilienceMixin",
    "LifecycleMixin",
]
