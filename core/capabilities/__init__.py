"""
Capability Discovery API
=========================

Programmatic discovery of all Jotty subsystems, execution paths,
and available components.

Usage:
    from Jotty.core.capabilities import capabilities, explain

    # Get structured map of everything Jotty can do
    caps = capabilities()
    print(caps['subsystems'].keys())

    # Human-readable explanation
    print(explain('learning'))
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def capabilities() -> Dict[str, Any]:
    """
    Return a structured map of all Jotty capabilities.

    Returns a dict with:
    - execution_paths: The three ways to run tasks (chat, workflow, swarm)
    - subsystems: Major subsystems with import paths and descriptions
    - swarms: Available domain swarms
    - skills_count: Number of registered skills
    - providers: Skill providers (browser-use, openhands, etc.)
    - utilities: Cross-cutting utilities (budget, circuit breaker, cache, etc.)
    """
    result = {
        "execution_paths": _execution_paths(),
        "subsystems": _subsystems(),
        "swarms": _swarms(),
        "skills_count": _skills_count(),
        "providers": _providers(),
        "utilities": _utilities(),
    }
    return result


def explain(component: str) -> str:
    """
    Human-readable explanation of any Jotty component.

    Args:
        component: Name of a subsystem, execution path, or utility.

    Returns:
        Multi-line explanation string.
    """
    explanations = _explanations()
    key = component.lower().replace("-", "_").replace(" ", "_")
    if key in explanations:
        return explanations[key]
    return f"Unknown component: {component!r}. Use capabilities() to see all available components."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _execution_paths() -> Dict[str, Any]:
    return {
        "chat": {
            "description": "Conversational Q&A — single LLM call, fast responses",
            "import_path": "from Jotty import Jotty; jotty = Jotty(); await jotty.chat('...')",
            "class": "Jotty.core.api.mode_router.ModeRouter",
            "tier": "Tier 1 (DIRECT)",
        },
        "workflow": {
            "description": "Multi-step planning and orchestration with agents",
            "import_path": "from Jotty import Jotty; jotty = Jotty(); await jotty.plan('...')",
            "class": "Jotty.core.execution.executor.TierExecutor",
            "tier": "Tier 2-3 (AGENTIC/LEARNING)",
        },
        "swarm": {
            "description": "Domain swarm execution with multi-agent coordination",
            "import_path": "from Jotty import Jotty; jotty = Jotty(); await jotty.swarm('...', swarm_name='coding')",
            "class": "Jotty.core.orchestration.Orchestrator",
            "tier": "Tier 4-5 (RESEARCH/AUTONOMOUS)",
        },
    }


def _subsystems() -> Dict[str, Any]:
    return {
        "learning": {
            "description": "Reinforcement learning: TD(lambda), credit assignment, cooperative agents",
            "package": "Jotty.core.learning",
            "facade": "Jotty.core.learning.facade",
            "key_classes": [
                "TDLambdaLearner",
                "ReasoningCreditAssigner",
                "OfflineLearner",
                "ShapedRewardManager",
                "PredictiveCooperativeAgent",
                "NashBargainingSolver",
                "LearningManager",
            ],
        },
        "memory": {
            "description": "Hierarchical memory: 5 levels, brain-inspired consolidation, RAG retrieval",
            "package": "Jotty.core.memory",
            "facade": "Jotty.core.memory.facade",
            "key_classes": [
                "MemorySystem",
                "BrainInspiredMemoryManager",
                "SharpWaveRippleConsolidator",
                "LLMRAGRetriever",
                "SwarmMemory",
            ],
        },
        "context": {
            "description": "Context management: auto-chunking, compression, overflow protection",
            "package": "Jotty.core.context",
            "facade": "Jotty.core.context.facade",
            "key_classes": [
                "SmartContextManager",
                "GlobalContextGuard",
                "ContentGate",
                "AgenticCompressor",
            ],
        },
        "orchestration": {
            "description": "Multi-agent coordination: swarm intelligence, paradigms, routing, training",
            "package": "Jotty.core.orchestration",
            "facade": "Jotty.core.orchestration.facade",
            "key_classes": [
                "Orchestrator",
                "SwarmIntelligence",
                "ParadigmExecutor",
                "TrainingDaemon",
                "EnsembleManager",
                "SwarmRouter",
                "ModelTierRouter",
                "ProviderManager",
            ],
        },
        "skills": {
            "description": "Skill registry, providers (browser-use, openhands, etc.), tool management",
            "package": "Jotty.core.registry",
            "facade": "Jotty.core.skills.facade",
            "key_classes": [
                "UnifiedRegistry",
                "SkillsRegistry",
                "ProviderRegistry",
                "SkillProvider",
            ],
        },
        "utils": {
            "description": "Utilities: budget tracking, circuit breaker, LLM cache, tokenizer",
            "package": "Jotty.core.utils",
            "facade": "Jotty.core.utils.facade",
            "key_classes": [
                "BudgetTracker",
                "CircuitBreaker",
                "LLMCallCache",
                "SmartTokenizer",
            ],
        },
    }


def _swarms() -> list:
    """List available domain swarms (safe import)."""
    try:
        from Jotty.core.intelligence.swarms.registry import SwarmRegistry
        return SwarmRegistry.list_all()
    except Exception:
        return []


def _skills_count() -> int:
    """Count registered skills (safe import)."""
    try:
        from Jotty.core.capabilities.registry import get_unified_registry
        registry = get_unified_registry()
        return len(registry.list_skills())
    except Exception:
        return 0


def _providers() -> list:
    """List known skill providers with install status."""
    provider_info = [
        {"name": "browser-use", "module": "browser_use", "description": "Web automation via browser-use library"},
        {"name": "openhands", "module": "openhands", "description": "Terminal/code via OpenHands SDK"},
        {"name": "agent-s", "module": "agent_s", "description": "GUI/computer control via Agent-S"},
        {"name": "open-interpreter", "module": "interpreter", "description": "Local code execution"},
        {"name": "streamlit", "module": "streamlit", "description": "App building (open source)"},
        {"name": "morph", "module": "morph", "description": "Cloud app building"},
        {"name": "n8n", "module": None, "description": "Workflow automation (API-based)"},
        {"name": "activepieces", "module": None, "description": "Flow automation (API-based)"},
    ]
    result = []
    for p in provider_info:
        installed = False
        if p["module"]:
            try:
                __import__(p["module"])
                installed = True
            except ImportError:
                pass
        else:
            installed = None  # API-based, no local module
        result.append({
            "name": p["name"],
            "description": p["description"],
            "installed": installed,
        })
    return result


def _utilities() -> Dict[str, Any]:
    return {
        "BudgetTracker": {
            "import_path": "from Jotty.core.infrastructure.utils.budget_tracker import BudgetTracker",
            "description": "Track and limit LLM spending per scope",
        },
        "CircuitBreaker": {
            "import_path": "from Jotty.core.infrastructure.utils.timeouts import CircuitBreaker",
            "description": "Fail-fast pattern for unreliable services",
        },
        "LLMCallCache": {
            "import_path": "from Jotty.core.infrastructure.utils.llm_cache import LLMCallCache",
            "description": "Semantic caching for LLM calls",
        },
        "SmartTokenizer": {
            "import_path": "from Jotty.core.infrastructure.utils.tokenizer import SmartTokenizer",
            "description": "Token counting with model-aware encoding",
        },
    }


def _explanations() -> Dict[str, str]:
    return {
        "learning": (
            "Learning Subsystem (Jotty.core.learning)\n"
            "==========================================\n"
            "Reinforcement learning components for agent improvement:\n"
            "- TDLambdaLearner: Temporal-difference learning with eligibility traces\n"
            "- ReasoningCreditAssigner: Credit assignment for multi-step reasoning\n"
            "- OfflineLearner: Batch learning from stored episodes\n"
            "- ShapedRewardManager: Reward shaping for faster convergence\n"
            "- PredictiveCooperativeAgent: Multi-agent cooperation with prediction\n"
            "- NashBargainingSolver: Game-theoretic negotiation between agents\n"
            "- LearningManager: Unified coordinator for all learning components\n\n"
            "Access via facade:\n"
            "  from Jotty.core.intelligence.learning.facade import get_learning_system, list_components"
        ),
        "memory": (
            "Memory Subsystem (Jotty.core.memory)\n"
            "======================================\n"
            "Hierarchical memory with brain-inspired consolidation:\n"
            "- MemorySystem: Zero-config unified entry point (recommended)\n"
            "- BrainInspiredMemoryManager: 5-level memory hierarchy\n"
            "- SharpWaveRippleConsolidator: Sleep-like memory consolidation\n"
            "- LLMRAGRetriever: LLM-powered retrieval-augmented generation\n"
            "- SwarmMemory: Low-level 5-level storage (cortex)\n\n"
            "Access via facade:\n"
            "  from Jotty.core.intelligence.memory.facade import get_memory_system, list_components"
        ),
        "context": (
            "Context Subsystem (Jotty.core.context)\n"
            "========================================\n"
            "Token management and context overflow protection:\n"
            "- SmartContextManager: Auto-chunking and compression\n"
            "- GlobalContextGuard: Overflow prevention with budget enforcement\n"
            "- ContentGate: Relevance filtering before context injection\n"
            "- AgenticCompressor: LLM-based context compression\n\n"
            "Access via facade:\n"
            "  from Jotty.core.infrastructure.context.facade import get_context_manager, list_components"
        ),
        "orchestration": (
            "Orchestration Subsystem (Jotty.core.orchestration)\n"
            "===================================================\n"
            "Multi-agent coordination and intelligence:\n"
            "- Orchestrator: Main composable swarm orchestrator\n"
            "- SwarmIntelligence: Emergent specialization, consensus, RL routing\n"
            "- ParadigmExecutor: MALLM paradigms (relay, debate, refinement)\n"
            "- TrainingDaemon: Background self-improvement loop\n"
            "- EnsembleManager: Prompt ensembling for multi-perspective analysis\n"
            "- SwarmRouter: Centralized task routing and agent selection\n"
            "- ModelTierRouter: Complexity-based LLM model selection\n"
            "- ProviderManager: Skill provider registry management\n\n"
            "Access via facade:\n"
            "  from Jotty.core.intelligence.orchestration.facade import get_swarm_intelligence, list_components"
        ),
        "skills": (
            "Skills Subsystem (Jotty.core.registry + Jotty.core.skills)\n"
            "============================================================\n"
            "Skill and provider management:\n"
            "- UnifiedRegistry: Single entry point for all capabilities (126+ skills)\n"
            "- SkillsRegistry: Backend skill definitions and tool metadata\n"
            "- ProviderRegistry: Pluggable providers (browser-use, openhands, etc.)\n"
            "- SkillProvider: Base class for custom providers\n\n"
            "Access via facade:\n"
            "  from Jotty.core.capabilities.skills.facade import get_registry, list_providers"
        ),
        "utils": (
            "Utilities Subsystem (Jotty.core.utils)\n"
            "========================================\n"
            "Cross-cutting utilities:\n"
            "- BudgetTracker: Track and limit LLM spending\n"
            "- CircuitBreaker: Fail-fast for unreliable services\n"
            "- LLMCallCache: Semantic caching for LLM calls\n"
            "- SmartTokenizer: Model-aware token counting\n\n"
            "Access via facade:\n"
            "  from Jotty.core.infrastructure.utils.facade import get_budget_tracker, list_components"
        ),
        "chat": (
            "Chat Execution Path\n"
            "====================\n"
            "Conversational Q&A with single LLM call (Tier 1 DIRECT).\n"
            "Fastest path — no planning, no agents, no memory.\n\n"
            "Usage:\n"
            "  jotty = Jotty()\n"
            "  response = await jotty.chat('What is 2+2?')\n\n"
            "Or via ModeRouter:\n"
            "  from Jotty.core.interface.api.mode_router import ModeRouter\n"
            "  router = ModeRouter()\n"
            "  result = await router.chat(message, context)"
        ),
        "workflow": (
            "Workflow Execution Path\n"
            "========================\n"
            "Multi-step planning and orchestration (Tier 2-3 AGENTIC/LEARNING).\n"
            "Includes planning, step execution, validation, and optional memory.\n\n"
            "Usage:\n"
            "  jotty = Jotty()\n"
            "  result = await jotty.plan('Research AI and create report')\n\n"
            "Or via TierExecutor:\n"
            "  from Jotty.core.modes.execution import TierExecutor, ExecutionConfig\n"
            "  executor = TierExecutor(config=ExecutionConfig())"
        ),
        "swarm": (
            "Swarm Execution Path\n"
            "======================\n"
            "Domain swarm execution with multi-agent coordination (Tier 4-5).\n"
            "Full features: learning, memory, validation, coalition, sandbox.\n\n"
            "Usage:\n"
            "  jotty = Jotty()\n"
            "  result = await jotty.swarm('Build REST API', swarm_name='coding')\n\n"
            "Or via Orchestrator:\n"
            "  from Jotty.core.intelligence.orchestration import Orchestrator\n"
            "  orch = Orchestrator(actors=[...], config=config)\n"
            "  result = await orch.run(goal='...')"
        ),
        "semantic": (
            "Semantic Layer Subsystem (Jotty.core.capabilities.semantic)\n"
            "=============================================================\n"
            "Database schema understanding and intelligent querying:\n"
            "- SemanticLayer: Main interface for schema extraction and querying\n"
            "- SemanticQueryEngine: Natural language to SQL query generation\n"
            "- MongoDBQueryEngine: Natural language to MongoDB aggregation pipelines\n"
            "- VisualizationLayer: LIDA-based data visualization from NL\n"
            "- DatabaseExtractor: Extract schema from live database connections\n"
            "- DDLExtractor: Extract schema from DDL strings\n"
            "- LookMLGenerator: Generate LookML semantic models (like Looker)\n\n"
            "Supported databases: PostgreSQL, MySQL, SQLite, SQL Server, Oracle, MongoDB\n\n"
            "Access via facade:\n"
            "  from Jotty.core.capabilities.semantic.facade import get_semantic_layer\n"
            "  layer = get_semantic_layer(db_type='postgresql', host='localhost', database='sales')\n"
            "  result = layer.query('Show total revenue by region')\n\n"
            "Or use skill wrappers:\n"
            "  - semantic-sql-query: Natural language to SQL queries\n"
            "  - schema-analyzer: Schema extraction and analysis\n"
            "  - data-visualizer: Data visualization from natural language"
        ),
    }
