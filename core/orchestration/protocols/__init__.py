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
from .routing import RoutingMixin
from .resilience import ResilienceMixin
from .lifecycle import LifecycleMixin

__all__ = [
    'CoordinationMixin',
    'RoutingMixin',
    'ResilienceMixin',
    'LifecycleMixin',
]
