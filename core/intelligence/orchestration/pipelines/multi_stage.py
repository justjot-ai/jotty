"""
Multi-Stage Pipeline — backward compatibility shim.

The canonical pipeline execution is now integrated into the Orchestrator:
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
    result = await orchestrator.run_pipeline(goal, stages=[...])

The standalone MultiStagePipeline is still available for direct use.
For new code, prefer Orchestrator.run_pipeline() which includes
integrated learning, memory, and self-correction.
"""

from Jotty.core.intelligence.orchestration.coordination.multi_stage_pipeline import (
    MultiStagePipeline,
    PipelineResult,
    StageConfig,
    StageResult,
    create_pipeline,
    extract_code_from_markdown,
)

__all__ = [
    "MultiStagePipeline",
    "PipelineResult",
    "StageConfig",
    "StageResult",
    "create_pipeline",
    "extract_code_from_markdown",
]
