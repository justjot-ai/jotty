"""
MEMORY Layer - Learning (SwarmMemory, TD-Lambda, Context)
"""
from Jotty.core.memory import SwarmMemory
from Jotty.core.learning import TDLambdaLearner

try:
    from Jotty.core.foundation.types import MemoryLevel, MemoryEntry, GoalValue
except ImportError:
    MemoryLevel = MemoryEntry = GoalValue = None

try:
    from Jotty.core.learning.q_learning import QLearner
except ImportError:
    QLearner = None

try:
    from Jotty.core.context import ContextManager, ContextGuard, GlobalContextGuard
    from Jotty.core.context.chunker import ContextChunker
    from Jotty.core.context.compressor import ContextCompressor
except ImportError:
    ContextManager = ContextGuard = GlobalContextGuard = None
    ContextChunker = ContextCompressor = None

__all__ = [
    "SwarmMemory", "MemoryLevel", "MemoryEntry", "GoalValue",
    "TDLambdaLearner", "QLearner",
    "ContextManager", "ContextGuard", "GlobalContextGuard", "ContextChunker", "ContextCompressor",
]
