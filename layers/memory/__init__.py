"""
MEMORY Layer - Learning (HierarchicalMemory, TD-Lambda, Context)
"""
from Jotty.core.memory import HierarchicalMemory
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
    from Jotty.core.context.chunker import AgenticChunker
    from Jotty.core.context.compressor import ContextCompressor
except ImportError:
    ContextManager = ContextGuard = GlobalContextGuard = None
    AgenticChunker = ContextCompressor = None

__all__ = [
    "HierarchicalMemory", "MemoryLevel", "MemoryEntry", "GoalValue",
    "TDLambdaLearner", "QLearner",
    "ContextManager", "ContextGuard", "GlobalContextGuard", "AgenticChunker", "ContextCompressor",
]
