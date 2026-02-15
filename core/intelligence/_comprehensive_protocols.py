"""
Comprehensive Protocol definitions for all mixin classes.

These protocols define ALL attributes and methods that mixins expect from parent classes
or other mixins in the composition. This allows mixins to be properly type-checked.

Usage:
    class MyMixin:
        if TYPE_CHECKING:
            # Import the appropriate protocol
            from ._comprehensive_protocols import ReportGeneratorProtocol

            # Then cast self or declare attributes using the protocol
            self: ReportGeneratorProtocol
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Set

if TYPE_CHECKING:
    pass


class ReportGeneratorProtocol(Protocol):
    """
    Comprehensive protocol for report generator mixins.

    Includes ALL attributes from:
    - VisualizationMixin
    - AnalysisSectionsMixin
    - RenderingMixin
    - InterpretabilityMixin
    - And all other report-related mixins
    """

    # === Directories ===
    output_dir: Path
    figures_dir: Path

    # === Configuration ===
    theme: str
    config: Dict[str, Any]
    _llm_narrative_enabled: bool
    _html_enabled: bool
    _report_config: Optional[Dict[str, Any]]
    _telegram_config: Optional[Dict[str, Any]]
    _mlflow_config: Optional[Dict[str, Any]]

    # === Content Tracking ===
    _content: List[Any]
    _figures: List[Any]
    _warnings: List[Any]
    _metadata: Dict[str, Any]
    _raw_data: Dict[str, Any]
    _section_data: List[Any]
    _failed_sections: List[str]
    _failed_charts: List[str]
    figures: List[Any]

    # === Report State ===
    _context: Optional[Any]
    _learned_context: Optional[Any]
    _memory: Optional[Any]

    # === Methods - Chart Creation ===
    def _record_chart_failure(self, chart_name: str, error: Exception) -> None: ...
    def _save_figure(self, fig: Any, name: str) -> Optional[Path]: ...
    def _fig_path_for_markdown(self, fig_path: Path) -> str: ...
    def _chart_context(self, chart_name: str, **kwargs: Any) -> Any: ...

    # === Methods - Section Management ===
    def _add_section(self, title: str, content: str, **kwargs: Any) -> None: ...
    def _record_section_failure(self, section: str, error: Exception) -> None: ...

    # === Methods - Report Building ===
    def _build_report_story(self, **kwargs: Any) -> str: ...
    def _build_telegram_caption(self, **kwargs: Any) -> str: ...
    def generate_report(self, **kwargs: Any) -> Any: ...
    def get_report_health(self) -> Dict[str, Any]: ...

    # === Methods - Telegram ===
    def init_telegram(self, **kwargs: Any) -> None: ...


class OrchestrationProtocol(Protocol):
    """
    Protocol for orchestration and swarm management mixins.

    Includes attributes from:
    - CoordinationMixin
    - LifecycleMixin
    - ResilienceMixin
    - RoutingMixin
    """

    # === Agent Management ===
    agents: List[Any]
    agent_profiles: Dict[str, Any]
    agent_name: Optional[str]

    # === Orchestration State ===
    config: Dict[str, Any]
    name: str
    history: List[Any]
    _swarm_intelligence: Optional[Any]
    _agent_context: Optional[Dict[str, Any]]

    # === Handoffs and Coordination ===
    handoff_history: List[Any]
    pending_handoffs: List[Any]
    active_auctions: Dict[str, Any]
    agent_coalitions: Dict[str, Any]
    coalitions: Dict[str, Any]

    # === Circuit Breakers and Resilience ===
    circuit_breakers: Dict[str, Any]

    # === Gossip Protocol ===
    gossip_inbox: List[Any]
    gossip_seen: Set[str]

    # === Goal Hierarchy ===
    goal_hierarchy: Optional[Any]

    # === Consensus ===
    consensus_history: List[Any]

    # === Morphing ===
    morph_score_history: List[Any]
    morph_scorer: Optional[Any]

    # === Team Config ===
    _team_config: Optional[Dict[str, Any]]

    # === Methods - Agent Operations ===
    def register_agent(self, agent: Any) -> None: ...
    def find_idle_agents(self) -> List[Any]: ...
    def find_overloaded_agents(self) -> List[Any]: ...
    def get_agent_load(self, agent_id: str) -> float: ...
    def get_available_agents(self) -> List[Any]: ...
    def get_best_agent_for_task(self, task: Any) -> Optional[Any]: ...

    # === Methods - Handoffs ===
    def initiate_handoff(self, from_agent: str, to_agent: str, **kwargs: Any) -> None: ...
    def auto_auction(self, task: Any) -> Any: ...
    def close_auction(self, auction_id: str) -> None: ...

    # === Methods - Coalitions ===
    def form_coalition(self, agents: List[str]) -> str: ...

    # === Methods - Circuit Breakers ===
    def check_circuit(self, operation: str) -> bool: ...

    # === Methods - Gossip ===
    def gossip_broadcast(self, message: Any) -> None: ...

    # === Methods - Health ===
    def get_swarm_health(self) -> Dict[str, Any]: ...
    def calculate_backpressure(self) -> float: ...
    def enqueue_task(self, task: Any) -> None: ...

    # === Methods - Supervisor Tree ===
    def build_supervisor_tree(self) -> Any: ...

    # === Methods - Swarm Intelligence ===
    def connect_swarm_intelligence(self, intelligence: Any) -> None: ...


class LearningProtocol(Protocol):
    """
    Protocol for learning mixins.

    Includes attributes from:
    - SwarmLearningMixin
    """

    # === Learning State ===
    learning_enabled: bool
    learning: Optional[Any]
    mas_learning: Optional[Any]
    learning_config: Optional[Any]
    _learning_state: Dict[str, Any]
    _learning_memory: List[Dict[str, Any]]

    # === Learner ===
    _learner: Optional[Any]

    # === Metadata ===
    name: str
    config: Dict[str, Any]

    # === Methods ===
    def _store_learning_memory(self, data: Dict[str, Any]) -> None: ...


class MemoryProtocol(Protocol):
    """
    Protocol for memory mixins.

    Includes attributes from:
    - ConsolidationMixin
    - RetrievalMixin
    """

    # === Memory Storage ===
    _graph: Any
    collective_memory: Optional[Any]
    memories: List[Any]
    memory_persistence: Optional[Any]

    # === Configuration ===
    config: Any

    # === Consolidation State ===
    consolidation_count: int
    _consolidation_module: Optional[Any]

    # === Extraction ===
    causal_extractor: Optional[Any]
    pattern_extractor: Optional[Any]
    meta_extractor: Optional[Any]
    causal_links: List[Any]

    # === Methods ===
    def _consolidate_nodes(self, nodes: List[Any]) -> Any: ...
    def _cluster_episodic_memories(self, memories: List[Any]) -> List[Any]: ...
    def _analyze_import_graph(self, **kwargs: Any) -> Any: ...


class CodingSwarmProtocol(Protocol):
    """
    Protocol for coding swarm mixins.

    Includes attributes from:
    - EditMixin
    - ReviewMixin
    """

    # === Context ===
    _context: Dict[str, Any]

    # === Agents ===
    _actor: Optional[Any]
    _auditor: Optional[Any]
    _arbitrator: Optional[Any]

    # === Modules ===
    _codebase_analyzer: Optional[Any]
    _collab_architect_module: Optional[Any]
    _consolidation_module: Optional[Any]

    # === Improvement ===
    _improvement_agents: List[Any]

    # === Cache ===
    _cache_hits: int
    _cache_misses: int

    # === Event Bus ===
    _bus: Optional[Any]

    # === Methods ===
    def _collect_feedback(self, **kwargs: Any) -> Any: ...
    def _analyze_prior_failures(self, **kwargs: Any) -> List[Any]: ...
    def _init_agents(self) -> None: ...


# === ALL-IN-ONE PROTOCOLS ===


class ComprehensiveMixinProtocol(
    ReportGeneratorProtocol,
    OrchestrationProtocol,
    LearningProtocol,
    MemoryProtocol,
    CodingSwarmProtocol,
    Protocol,
):
    """
    Comprehensive protocol that includes ALL mixin attributes.

    Use this when a class uses multiple types of mixins.
    """

    pass
