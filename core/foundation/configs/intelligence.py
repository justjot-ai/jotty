"""Intelligence configuration — trust tuning, routing, agent communication, local mode."""

from dataclasses import dataclass


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
