"""
BRAIN Layer - Coordination (Swarms, Agents, Intelligence)
"""
# Agents
from Jotty.core.agents.base import BaseAgent, DomainAgent, MetaAgent, AutonomousAgent, AgentConfig, AgentResult
from Jotty.core.agents.auto_agent import AutoAgent
from Jotty.core.agents.chat_assistant import ChatAssistant
from Jotty.core.agents.agentic_planner import AgenticPlanner
from Jotty.core.agents.dag_agents import TaskBreakdownAgent, TodoCreatorAgent

# Swarms
from Jotty.core.swarms import BaseSwarm, DomainSwarm, SwarmConfig, SwarmResult

# Specialized swarms
try:
    from Jotty.core.swarms import CodingSwarm, TestingSwarm, ReviewSwarm, DataAnalysisSwarm
    from Jotty.core.swarms import FundamentalSwarm, DevOpsSwarm, IdeaWriterSwarm, LearningSwarm
except ImportError:
    pass

# Intelligence
from Jotty.core.learning import TDLambdaLearner

try:
    from Jotty.core.orchestration.v2.swarm_intelligence import SwarmIntelligence
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager
except ImportError:
    SwarmIntelligence = None
    SwarmManager = None

__all__ = [
    "BaseAgent", "DomainAgent", "MetaAgent", "AutonomousAgent", "AgentConfig", "AgentResult",
    "AutoAgent", "ChatAssistant", "AgenticPlanner", "TaskBreakdownAgent", "TodoCreatorAgent",
    "BaseSwarm", "DomainSwarm", "SwarmConfig", "SwarmResult",
    "CodingSwarm", "TestingSwarm", "ReviewSwarm", "DataAnalysisSwarm",
    "FundamentalSwarm", "DevOpsSwarm", "IdeaWriterSwarm", "LearningSwarm",
    "TDLambdaLearner", "SwarmIntelligence", "SwarmManager",
]
