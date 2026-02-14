"""Context budget configuration — token allocation and dynamic budgeting."""

from dataclasses import dataclass
from typing import Optional, Any


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

    def __post_init__(self) -> None:
        # All token budgets must be positive
        _pos_fields = {
            'max_context_tokens': self.max_context_tokens,
            'system_prompt_budget': self.system_prompt_budget,
            'current_input_budget': self.current_input_budget,
            'trajectory_budget': self.trajectory_budget,
            'tool_output_budget': self.tool_output_budget,
            'min_memory_budget': self.min_memory_budget,
            'max_memory_budget': self.max_memory_budget,
        }
        for name, val in _pos_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # min <= max for memory budget
        if self.min_memory_budget > self.max_memory_budget:
            raise ValueError(
                f"min_memory_budget ({self.min_memory_budget}) must be "
                f"<= max_memory_budget ({self.max_memory_budget})"
            )

        # Static budgets should not exceed max context
        static_sum = (self.system_prompt_budget + self.current_input_budget +
                      self.trajectory_budget + self.tool_output_budget)
        if static_sum > self.max_context_tokens:
            raise ValueError(
                f"Sum of static budgets ({static_sum}) exceeds "
                f"max_context_tokens ({self.max_context_tokens})"
            )
