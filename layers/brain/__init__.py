"""
BRAIN Layer - Coordination (Swarms, Agents, Intelligence)

All 11 specialized swarms + base infrastructure.
"""
# Agents
from Jotty.core.agents.base import BaseAgent, DomainAgent, MetaAgent, AutonomousAgent, AgentConfig, AgentResult
from Jotty.core.agents.auto_agent import AutoAgent
from Jotty.core.agents.chat_assistant import ChatAssistant
from Jotty.core.agents.agentic_planner import AgenticPlanner
from Jotty.core.agents.dag_agents import TaskBreakdownAgent, TodoCreatorAgent

# Swarms - Base
from Jotty.core.swarms import BaseSwarm, DomainSwarm, SwarmConfig, SwarmResult

# Swarms - All 11 specialized swarms
try:
    from Jotty.core.swarms import (
        # Research & Analysis
        ResearchSwarm, ResearchConfig, ResearchResult, research, research_sync,
        FundamentalSwarm, FundamentalConfig, FundamentalResult, analyze_fundamentals,
        DataAnalysisSwarm, DataAnalysisConfig, AnalysisResult, analyze_data,
        # Development
        CodingSwarm, CodingConfig, CodingResult, code, code_sync,
        TestingSwarm, TestingConfig, TestingResult, test, test_sync,
        ReviewSwarm, ReviewConfig, ReviewResult, review_code,
        # Content & Operations
        IdeaWriterSwarm, WriterConfig, WriterResult, write, write_sync,
        DevOpsSwarm, DevOpsConfig, DevOpsResult, deploy,
        # Education
        ArxivLearningSwarm, ArxivLearningConfig, ArxivLearningResult, learn_paper,
        # Meta
        LearningSwarm, LearningConfig, LearningResult, improve_swarm,
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Some swarms not available: {e}")

# Intelligence
from Jotty.core.learning import TDLambdaLearner

try:
    from Jotty.core.orchestration.v2.swarm_intelligence import SwarmIntelligence
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager
except ImportError:
    SwarmIntelligence = None
    SwarmManager = None

__all__ = [
    # Agents
    "BaseAgent", "DomainAgent", "MetaAgent", "AutonomousAgent", "AgentConfig", "AgentResult",
    "AutoAgent", "ChatAssistant", "AgenticPlanner", "TaskBreakdownAgent", "TodoCreatorAgent",
    # Base Swarms
    "BaseSwarm", "DomainSwarm", "SwarmConfig", "SwarmResult",
    # Research & Analysis Swarms
    "ResearchSwarm", "ResearchConfig", "ResearchResult", "research", "research_sync",
    "FundamentalSwarm", "FundamentalConfig", "FundamentalResult", "analyze_fundamentals",
    "DataAnalysisSwarm", "DataAnalysisConfig", "AnalysisResult", "analyze_data",
    # Development Swarms
    "CodingSwarm", "CodingConfig", "CodingResult", "code", "code_sync",
    "TestingSwarm", "TestingConfig", "TestingResult", "test", "test_sync",
    "ReviewSwarm", "ReviewConfig", "ReviewResult", "review_code",
    # Content & Operations Swarms
    "IdeaWriterSwarm", "WriterConfig", "WriterResult", "write", "write_sync",
    "DevOpsSwarm", "DevOpsConfig", "DevOpsResult", "deploy",
    # Education Swarms
    "ArxivLearningSwarm", "ArxivLearningConfig", "ArxivLearningResult", "learn_paper",
    # Meta Swarms
    "LearningSwarm", "LearningConfig", "LearningResult", "improve_swarm",
    # Intelligence
    "TDLambdaLearner", "SwarmIntelligence", "SwarmManager",
]
