"""
Workflows - Intent-Based Automation
====================================

Re-exports from core.execution.workflows for backward compatibility.

Three Domain-Specific Workflows:
1. AutoWorkflow - General software development (APIs, apps, systems)
2. ResearchWorkflow - Research and analysis (topics, markets, trends)
3. LearningWorkflow - Educational content (K-12 to Olympiad)
4. AutoMLWorkflow - Machine learning pipelines

⚠️ DEPRECATED: Import from Jotty.core.execution.workflows instead
"""

# Re-export from canonical location
from Jotty.core.execution.workflows.auto_workflow import (
    AutoWorkflow,
    WorkflowIntent,
    build,
    develop,
)
from Jotty.core.execution.workflows.auto_workflow import research as research_stage
from Jotty.core.execution.workflows.learning_workflow import (
    LearningDepth,
    LearningIntent,
    LearningLevel,
    LearningWorkflow,
    Subject,
    learn,
)
from Jotty.core.execution.workflows.research_workflow import (
    ResearchDepth,
    ResearchIntent,
    ResearchType,
    ResearchWorkflow,
    research,
)
from Jotty.core.execution.workflows.smart_swarm_registry import (
    SmartSwarmRegistry,
    StageType,
    SwarmConfig,
    get_smart_registry,
)
from Jotty.skills.document_tools import (
    OutputFormat,
    OutputFormatManager,
    OutputFormatResult,
)
from Jotty.skills.messaging_tools import (
    ChannelDeliveryResult,
    OutputChannel,
    OutputChannelManager,
)

__all__ = [
    # AutoWorkflow (software development)
    "AutoWorkflow",
    "WorkflowIntent",
    "build",
    "develop",
    # ResearchWorkflow (research & analysis)
    "ResearchWorkflow",
    "ResearchIntent",
    "ResearchDepth",
    "ResearchType",
    "research",
    # LearningWorkflow (educational content)
    "LearningWorkflow",
    "LearningIntent",
    "LearningLevel",
    "LearningDepth",
    "Subject",
    "learn",
    # Registry
    "SmartSwarmRegistry",
    "StageType",
    "SwarmConfig",
    "get_smart_registry",
    # Output Formats
    "OutputFormatManager",
    "OutputFormat",
    "OutputFormatResult",
    # Output Channels
    "OutputChannelManager",
    "OutputChannel",
    "ChannelDeliveryResult",
]
