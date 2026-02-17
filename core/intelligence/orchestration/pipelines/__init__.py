"""
Pipelines — Intent-based automation and multi-stage orchestration.

Four domain-specific pipelines:
1. AutoPipeline - General software development (APIs, apps, systems)
2. ResearchPipeline - Research and analysis (topics, markets, trends)
3. LearningPipeline - Educational content (K-12 to Olympiad)
4. AutoMLPipeline - Machine learning pipelines (AutoML)

Plus:
- MultiStagePipeline - Chain multiple swarm executions
- SmartSwarmRegistry - Stage-to-swarm routing
"""

import importlib as _importlib
from typing import Any

_LAZY_IMPORTS: dict[str, str] = {
    # auto_pipeline
    "AutoWorkflow": ".auto_pipeline",
    "WorkflowIntent": ".auto_pipeline",
    "build": ".auto_pipeline",
    "develop": ".auto_pipeline",
    # research_pipeline
    "ResearchWorkflow": ".research_pipeline",
    "ResearchIntent": ".research_pipeline",
    "ResearchDepth": ".research_pipeline",
    "ResearchType": ".research_pipeline",
    "research": ".research_pipeline",
    # learning_pipeline
    "LearningWorkflow": ".learning_pipeline",
    "LearningIntent": ".learning_pipeline",
    "LearningLevel": ".learning_pipeline",
    "LearningDepth": ".learning_pipeline",
    "Subject": ".learning_pipeline",
    "learn": ".learning_pipeline",
    # automl_pipeline
    "AutoMLWorkflow": ".automl_pipeline",
    "ProblemType": ".automl_pipeline",
    "SkillCategory": ".automl_pipeline",
    "get_automl_workflow": ".automl_pipeline",
    "reset_automl_workflow": ".automl_pipeline",
    # registry
    "SmartSwarmRegistry": ".registry",
    "StageType": ".registry",
    "SwarmConfig": ".registry",
    "get_smart_registry": ".registry",
    # multi_stage
    "MultiStagePipeline": ".multi_stage",
    "StageConfig": ".multi_stage",
    "PipelineResult": ".multi_stage",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*_LAZY_IMPORTS.keys()]
