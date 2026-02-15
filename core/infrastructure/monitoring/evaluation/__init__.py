"""
Evaluation Framework

Provides standardized evaluation, reproducibility, and empirical validation.
Based on OAgents evaluation approach.
"""

from .ablation_study import (
    DEFAULT_SEARCH_GROUPS,
    AblationResult,
    AblationStudy,
    ComponentContribution,
    ComponentType,
    ConfigSearchGroup,
    ConfigTrialResult,
    ConfigTuner,
    TuningResult,
)
from .benchmark import Benchmark, BenchmarkMetrics, BenchmarkResult, CustomBenchmark
from .eval_store import EvalStore
from .evaluation_protocol import EvaluationProtocol, EvaluationReport, EvaluationRun
from .gaia_adapter import JottyGAIAAdapter
from .gaia_benchmark import GAIABenchmark
from .gaia_signatures import GAIAAnswerExtractSignature, normalize_gaia_answer_with_dspy
from .llm_doc_sources import (
    OPEN_SOURCE_LLM_SOURCES,
    LLMDocSource,
    get_source,
    get_sources_by_provider,
    list_sources,
    load_hf_dataset_info,
    to_context_snippet,
)
from .reproducibility import ReproducibilityConfig, ensure_reproducibility, set_reproducible_seeds

__all__ = [
    "ReproducibilityConfig",
    "set_reproducible_seeds",
    "ensure_reproducibility",
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "CustomBenchmark",
    "GAIABenchmark",
    "JottyGAIAAdapter",
    "GAIAAnswerExtractSignature",
    "normalize_gaia_answer_with_dspy",
    "EvaluationProtocol",
    "EvaluationRun",
    "EvaluationReport",
    "AblationStudy",
    "ComponentContribution",
    "AblationResult",
    "ComponentType",
    "ConfigTuner",
    "ConfigSearchGroup",
    "ConfigTrialResult",
    "TuningResult",
    "DEFAULT_SEARCH_GROUPS",
    "EvalStore",
    "LLMDocSource",
    "OPEN_SOURCE_LLM_SOURCES",
    "list_sources",
    "get_source",
    "get_sources_by_provider",
    "to_context_snippet",
    "load_hf_dataset_info",
]
