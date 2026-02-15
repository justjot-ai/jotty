"""
Workflows - Intent-Based Automation
====================================

High-level workflows that understand intent and automatically execute.

Three Domain-Specific Workflows:
1. AutoWorkflow - General software development (APIs, apps, systems)
2. ResearchWorkflow - Research and analysis (topics, markets, trends)
3. LearningWorkflow - Educational content (K-12 to Olympiad)
"""

from .auto_workflow import AutoWorkflow, WorkflowIntent, build, develop
from .auto_workflow import research as research_stage  # Rename to avoid conflict
from .learning_workflow import (
    LearningDepth,
    LearningIntent,
    LearningLevel,
    LearningWorkflow,
    Subject,
    learn,
)
from .output_channels import ChannelDeliveryResult, OutputChannel, OutputChannelManager
from .output_formats import OutputFormat, OutputFormatManager, OutputFormatResult
from .research_workflow import research  # This is the actual research workflow
from .research_workflow import ResearchDepth, ResearchIntent, ResearchType, ResearchWorkflow
from .smart_swarm_registry import SmartSwarmRegistry, StageType, SwarmConfig, get_smart_registry

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
