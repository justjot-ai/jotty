"""Context budget configuration — token allocation and dynamic budgeting."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextBudgetConfig:
    """Token budget allocation and context management."""
    max_context_tokens: int = 100000
    system_prompt_budget: int = 5000
    current_input_budget: int = 15000
    trajectory_budget: int = 20000
    tool_output_budget: int = 15000
    enable_dynamic_budget: bool = True
    min_memory_budget: int = 10000
    max_memory_budget: int = 60000
    token_model_name: Optional[str] = None
