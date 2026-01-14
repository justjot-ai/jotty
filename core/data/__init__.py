"""
Data Layer - Data Management & Parameter Resolution
===================================================

Data processing, parameter resolution, and data discovery.

Modules:
--------
- data_registry: Agentic data discovery and registration
- io_manager: Input/output management
- parameter_resolver: LLM-based parameter matching
- feedback_router: Route feedback to agents
- data_transformer: Data format transformation
- data_extractor: Extract structured data
- information_storage: Information persistence
- agentic_discovery/: Agentic data discovery orchestrator
"""

from .feedback_router import (
    AgenticFeedbackRouter,
    AgenticFeedbackSignature,
)
from .parameter_resolver import (
    AgenticParameterResolver,
    ParameterMatchingSignature,
)
from .data_registry import (
    DataArtifact,
    DataRegistry,
    DataRegistryTool,
)
from .information_storage import (
    InformationTheoreticStorage,
    InformationWeightedMemory,
    SurpriseEstimator,
    SurpriseSignature,
)
from .io_manager import (
    ActorOutput,
    IOManager,
    KnowledgeProvenance,
    SwarmResult,
)
from .data_extractor import (
    SmartDataExtractor,
)
from .data_transformer import (
    FormatTools,
    SmartDataTransformer,
)

__all__ = [
    # feedback_router
    'AgenticFeedbackRouter',
    'AgenticFeedbackSignature',
    # parameter_resolver
    'AgenticParameterResolver',
    'ParameterMatchingSignature',
    # data_registry
    'DataArtifact',
    'DataRegistry',
    'DataRegistryTool',
    # information_storage
    'InformationTheoreticStorage',
    'InformationWeightedMemory',
    'SurpriseEstimator',
    'SurpriseSignature',
    # io_manager
    'ActorOutput',
    'IOManager',
    'KnowledgeProvenance',
    'SwarmResult',
    # data_extractor
    'SmartDataExtractor',
    # data_transformer
    'FormatTools',
    'SmartDataTransformer',
]
