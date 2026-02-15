"""Intelligence configuration — trust tuning, routing, agent communication, local mode."""

from dataclasses import dataclass
from typing import Any


@dataclass
class IntelligenceConfig:
    """Trust tuning, routing, agent communication, and local mode."""

    trust_decrease_on_struggle: float = 0.1
    trust_increase_on_excel: float = 0.05
    trust_min: float = 0.1
    adaptation_interval: int = 5
    adaptation_struggle_threshold: float = 0.3
    adaptation_excel_threshold: float = 0.8
    stigmergy_routing_threshold: float = 0.5
    morph_min_rcs: float = 0.3
    judge_intervention_confidence: float = 0.6
    memory_retrieval_budget: int = 3000
    collective_memory_limit: int = 200
    enable_agent_communication: bool = True
    share_tool_results: bool = True
    share_insights: bool = True
    max_messages_per_episode: int = 20
    local_mode: bool = False
    local_model: str = "ollama/llama3"

    def __post_init__(self) -> None:
        # Unit interval [0, 1] fields
        _unit_fields = {
            "trust_decrease_on_struggle": self.trust_decrease_on_struggle,
            "trust_increase_on_excel": self.trust_increase_on_excel,
            "trust_min": self.trust_min,
            "adaptation_struggle_threshold": self.adaptation_struggle_threshold,
            "adaptation_excel_threshold": self.adaptation_excel_threshold,
            "stigmergy_routing_threshold": self.stigmergy_routing_threshold,
            "morph_min_rcs": self.morph_min_rcs,
            "judge_intervention_confidence": self.judge_intervention_confidence,
        }
        for name, val in _unit_fields.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        # Positive integer fields
        _pos_int_fields = {
            "adaptation_interval": self.adaptation_interval,
            "memory_retrieval_budget": self.memory_retrieval_budget,
            "collective_memory_limit": self.collective_memory_limit,
            "max_messages_per_episode": self.max_messages_per_episode,
        }
        for name, val in _pos_int_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
