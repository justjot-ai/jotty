"""
Open-source LLM document and dataset sources (Microsoft, Hugging Face, and others).

Provides a registry of public docs, benchmarks, and guides that can be used for
evaluation context, system-prompt improvements, or future benchmarks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Source definitions: Microsoft, Hugging Face, and other players
# -----------------------------------------------------------------------------

@dataclass
class LLMDocSource:
    """Single open-source LLM doc or dataset source."""
    id: str
    name: str
    provider: str  # e.g. "microsoft", "huggingface"
    url: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_config: Optional[str] = None
    description: str = ""
    kind: str = "doc"  # "doc" | "dataset" | "benchmark"


# Microsoft and other major players' open-source LLM resources
OPEN_SOURCE_LLM_SOURCES: List[LLMDocSource] = [
    # Microsoft
    LLMDocSource(
        id="ms-prompt-engineering",
        name="Prompt engineering for LLMs (Microsoft)",
        provider="microsoft",
        url="https://github.com/microsoft/prompt-engineering-for-llms",
        description="Open-source prompt engineering guide and examples for LLMs.",
        kind="doc",
    ),
    LLMDocSource(
        id="ms-phi-docs",
        name="Phi model family documentation",
        provider="microsoft",
        url="https://github.com/microsoft/Phi",
        description="Microsoft Phi small language models documentation and recipes.",
        kind="doc",
    ),
    LLMDocSource(
        id="ms-orca-math",
        name="Orca-Math (Microsoft)",
        provider="microsoft",
        hf_repo="microsoft/Orca-Math-200K",
        url="https://github.com/microsoft/Orca-Math",
        description="Synthetic math reasoning dataset for training/evaluation.",
        kind="dataset",
    ),
    LLMDocSource(
        id="ms-flan",
        name="FLAN / instruction tuning (Microsoft contributions)",
        provider="microsoft",
        hf_repo="microsoft/FLAN",
        description="Instruction-tuned models and datasets.",
        kind="dataset",
    ),
    # Hugging Face
    LLMDocSource(
        id="hf-llm-course",
        name="Hugging Face LLM course",
        provider="huggingface",
        url="https://huggingface.co/learn/nlp-course",
        description="Free NLP/LLM course and documentation.",
        kind="doc",
    ),
    LLMDocSource(
        id="hf-datasets",
        name="Hugging Face Datasets",
        provider="huggingface",
        url="https://huggingface.co/datasets",
        description="Large catalog of open datasets for training and evaluation.",
        kind="doc",
    ),
    LLMDocSource(
        id="hf-evaluate",
        name="Hugging Face Evaluate",
        provider="huggingface",
        url="https://huggingface.co/docs/evaluate",
        description="Evaluation library and metrics for LLMs.",
        kind="benchmark",
    ),
    # Other players
    LLMDocSource(
        id="anthropic-prompt-engineering",
        name="Anthropic prompt engineering",
        provider="anthropic",
        url="https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering",
        description="Claude prompt engineering and safety docs.",
        kind="doc",
    ),
    LLMDocSource(
        id="openai-cookbook",
        name="OpenAI Cookbook",
        provider="openai",
        url="https://cookbook.openai.com",
        description="OpenAI examples and best practices.",
        kind="doc",
    ),
    LLMDocSource(
        id="mlflow-llm-eval",
        name="MLflow LLM evaluation",
        provider="mlflow",
        url="https://mlflow.org/docs/latest/llms/llm-evaluation/index.html",
        description="MLflow docs for LLM evaluation and benchmarking.",
        kind="benchmark",
    ),
    LLMDocSource(
        id="eleuther-lm-eval",
        name="LM Evaluation Harness (EleutherAI)",
        provider="eleuther",
        url="https://github.com/EleutherAI/lm-evaluation-harness",
        hf_repo="EleutherAI/lm-evaluation-harness",
        description="Standard framework for evaluating LLMs on many benchmarks.",
        kind="benchmark",
    ),
    LLMDocSource(
        id="open-compass",
        name="OpenCompass (Shanghai AI Lab)",
        provider="opencompass",
        url="https://opencompass.org.cn",
        description="Open-source LLM evaluation platform and benchmarks.",
        kind="benchmark",
    ),
]


def list_sources(
    provider: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[LLMDocSource]:
    """List registered open-source LLM doc/dataset sources, optionally filtered."""
    out = list(OPEN_SOURCE_LLM_SOURCES)
    if provider:
        out = [s for s in out if s.provider.lower() == provider.lower()]
    if kind:
        out = [s for s in out if s.kind == kind]
    return out


def get_source(source_id: str) -> Optional[LLMDocSource]:
    """Get a single source by id."""
    for s in OPEN_SOURCE_LLM_SOURCES:
        if s.id == source_id:
            return s
    return None


def get_sources_by_provider(provider: str) -> List[LLMDocSource]:
    """Get all sources for a provider (e.g. 'microsoft', 'huggingface')."""
    return list_sources(provider=provider)


def to_context_snippet(sources: List[LLMDocSource], max_items: int = 5) -> str:
    """
    Build a short text snippet listing these sources for use in a system prompt or context.
    """
    lines = ["Relevant open-source LLM resources you can refer to:"]
    for s in sources[:max_items]:
        ref = s.url or (f"https://huggingface.co/{s.hf_repo}" if s.hf_repo else "")
        if ref:
            lines.append(f"- {s.name} ({s.provider}): {ref}")
    return "\n".join(lines)


def load_hf_dataset_info(repo_id: str, config: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load minimal info about a HuggingFace dataset (e.g. for listing or context).
    Returns None if datasets/huggingface_hub not available or on error.
    """
    try:
        from huggingface_hub import dataset_info
        info = dataset_info(repo_id, token=None)
        return {
            "id": repo_id,
            "config": config,
            "description": getattr(info, "description", "") or "",
            "card_data": getattr(info, "card_data", None),
        }
    except Exception:
        return None
