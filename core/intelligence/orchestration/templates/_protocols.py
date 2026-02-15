"""
Protocol definitions for mixin classes.

These protocols define the expected attributes and methods that mixins
can assume will be available from the parent class they're mixed with.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    import pandas as pd


class ReportGeneratorProtocol(Protocol):
    """
    Protocol defining the interface that report generator mixins expect.

    This allows mixins to safely access attributes defined in ProfessionalMLReport
    without mypy errors.
    """

    # Directories
    output_dir: Path
    figures_dir: Path

    # Configuration
    theme: str
    config: Dict[str, Any]
    _llm_narrative_enabled: bool
    _html_enabled: bool

    # Content tracking
    _content: List[Any]
    _figures: List[Any]
    _warnings: List[Any]
    _metadata: Dict[str, Any]
    _raw_data: Dict[str, Any]
    _section_data: List[Any]
    _failed_sections: List[str]
    _failed_charts: List[str]

    # Methods that mixins call
    def _record_chart_failure(self, chart_name: str, error: Exception) -> None: ...
    def _save_figure(self, fig: Any, name: str) -> Optional[Path]: ...
    def _fig_path_for_markdown(self, fig_path: Path) -> str: ...
    def _add_section(self, title: str, content: str, **kwargs: Any) -> None: ...
    def _record_section_failure(self, section: str, error: Exception) -> None: ...


class SwarmProtocol(Protocol):
    """Protocol for swarm classes that use learning mixins."""

    # Learning state
    learning_enabled: bool
    learning_config: Optional[Any]
    _learning_memory: List[Dict[str, Any]]

    # Metadata
    name: str

    # Methods
    def _store_learning_memory(self, data: Dict[str, Any]) -> None: ...


class MemoryProtocol(Protocol):
    """Protocol for memory classes that use consolidation mixins."""

    # Memory storage
    _graph: Any
    config: Any

    # Methods
    def _consolidate_nodes(self, nodes: List[Any]) -> Any: ...


class OrchestrationProtocol(Protocol):
    """Protocol for orchestration classes."""

    # State
    state: Dict[str, Any]
    agents: List[Any]

    # Methods
    def coordinate(self) -> Any: ...
    def manage_lifecycle(self) -> Any: ...
