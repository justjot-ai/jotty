"""
Evaluation Framework

Provides standardized evaluation, reproducibility, and empirical validation.
Based on OAgents evaluation approach.
"""

from .reproducibility import ReproducibilityConfig, set_reproducible_seeds, ensure_reproducibility
from .benchmark import Benchmark, BenchmarkResult, BenchmarkMetrics, CustomBenchmark
from .evaluation_protocol import EvaluationProtocol, EvaluationRun, EvaluationReport
from .ablation_study import (
    AblationStudy, ComponentContribution, AblationResult, ComponentType,
    ConfigTuner, ConfigSearchGroup, ConfigTrialResult, TuningResult,
    DEFAULT_SEARCH_GROUPS,
)
from .gaia_benchmark import GAIABenchmark
from .gaia_adapter import JottyGAIAAdapter
from .eval_store import EvalStore
from .llm_doc_sources import (
    LLMDocSource,
    OPEN_SOURCE_LLM_SOURCES,
    list_sources,
    get_source,
    get_sources_by_provider,
    to_context_snippet,
    load_hf_dataset_info,
)

__all__ = [
    'ReproducibilityConfig',
    'set_reproducible_seeds',
    'ensure_reproducibility',
    'Benchmark',
    'BenchmarkResult',
    'BenchmarkMetrics',
    'CustomBenchmark',
    'GAIABenchmark',
    'JottyGAIAAdapter',
    'EvaluationProtocol',
    'EvaluationRun',
    'EvaluationReport',
    'AblationStudy',
    'ComponentContribution',
    'AblationResult',
    'ComponentType',
    'ConfigTuner',
    'ConfigSearchGroup',
    'ConfigTrialResult',
    'TuningResult',
    'DEFAULT_SEARCH_GROUPS',
    'EvalStore',
    'LLMDocSource',
    'OPEN_SOURCE_LLM_SOURCES',
    'list_sources',
    'get_source',
    'get_sources_by_provider',
    'to_context_snippet',
    'load_hf_dataset_info',
]
