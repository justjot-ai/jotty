"""
Coding Swarm - Types & Configuration
======================================

Enums, configuration dataclasses, and result types for CodingSwarm.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..swarm_types import SwarmConfig, SwarmResult

# =============================================================================
# ENUMS
# =============================================================================


class CodeLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"


class CodeStyle(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    FUNCTIONAL = "functional"
    OOP = "oop"


class EditMode(Enum):
    """Mode for code generation/editing."""

    GENERATE = "generate"  # Greenfield generation (default)
    EDIT = "edit"  # Edit existing code
    REFACTOR = "refactor"  # Refactor without changing behavior
    EXTEND = "extend"  # Add features to existing code


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CodingConfig(SwarmConfig):
    """Configuration for CodingSwarm."""

    language: CodeLanguage = CodeLanguage.PYTHON
    style: CodeStyle = CodeStyle.STANDARD
    include_tests: bool = True
    include_docs: bool = True
    include_types: bool = True
    max_file_size: int = 500  # lines
    frameworks: List[str] = field(default_factory=list)
    lint_rules: str = "standard"
    team: Optional[str] = None  # "fullstack", "datascience", "frontend", or None (auto-detect)
    enable_workspace: bool = True  # Terminal-based validation
    enable_research: bool = True  # Web search research phase
    max_fix_attempts: int = 3  # Max debug-fix iterations in validation
    max_design_iterations: int = 3  # Max iterations for Architect-Researcher collaborative loop
    scope: Optional[str] = None  # "single_tier", "full_stack", or None (auto-detect)
    skip_team_planning: bool = False  # Skip Phase 2 team planning (for trivial tasks)
    skip_team_review: bool = False  # Skip Phase 6 team review (for trivial tasks)
    db_type: str = "sqlite"  # Database type
    backend_framework: str = "fastapi"  # Backend framework
    frontend_framework: str = "react"  # Frontend framework
    enable_arbitrator: bool = True  # Enable review arbitration
    # Edit mode configuration
    mode: EditMode = EditMode.GENERATE  # Operation mode
    target_files: List[str] = field(default_factory=list)  # Files to edit (for EDIT/REFACTOR mode)
    codebase_path: Optional[str] = None  # Root path of existing codebase
    preserve_style: bool = True  # Match existing code style when editing
    # Edit mode enhancements
    auto_discover_files: bool = True  # Auto-discover files from codebase_path if target_files empty
    output_diffs: bool = True  # Output unified diffs instead of full files
    analyze_dependencies: bool = True  # Analyze import graph before editing
    preserve_tests: bool = True  # Keep existing tests intact when editing
    git_integration: bool = False  # Enable git branch/commit for edits
    git_branch_prefix: str = "jotty-edit"  # Prefix for auto-created branches
    # Test-driven iteration (key for SWE-bench performance)
    test_driven: bool = False  # Enable test-driven edit loop
    max_edit_iterations: int = 5  # Max iterations for test-driven refinement
    test_command: Optional[str] = None  # Custom test command (auto-detect if None)
    test_timeout: int = 120  # Timeout for test execution in seconds

    def __post_init__(self) -> None:
        self.name = "CodingSwarm"
        self.domain = "coding"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class CodeOutput:
    """Output from code generation."""

    files: Dict[str, str]  # filename -> content
    main_file: str
    entry_point: str
    dependencies: List[str]
    tests: Dict[str, str]  # test filename -> content
    docs: str
    architecture: str


@dataclass
class ResearchContext:
    """Context gathered from web research before code generation."""

    best_practices: List[str] = field(default_factory=list)
    library_docs: List[str] = field(default_factory=list)
    api_references: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Format findings for injection into developer prompts."""
        sections = []
        if self.best_practices:
            sections.append(
                "BEST PRACTICES:\n" + "\n".join(f"- {bp}" for bp in self.best_practices)
            )
        if self.library_docs:
            sections.append("LIBRARY DOCS:\n" + "\n".join(f"- {ld}" for ld in self.library_docs))
        if self.api_references:
            sections.append(
                "API REFERENCES:\n" + "\n".join(f"- {ar}" for ar in self.api_references)
            )
        if self.warnings:
            sections.append("WARNINGS:\n" + "\n".join(f"- {w}" for w in self.warnings))
        return "\n\n".join(sections)


@dataclass
class FullStackContext:
    """Intermediate artifacts passed between full-stack pipeline phases."""

    data_model: str = ""
    api_contract: str = ""
    component_map: str = ""
    tech_stack: Dict[str, str] = field(
        default_factory=lambda: {"db_type": "sqlite", "backend": "fastapi", "frontend": "react"}
    )
    orm_models: str = ""
    schema_sql: str = ""
    openapi_spec: str = ""
    ui_requirements: str = ""


@dataclass
class CodingResult(SwarmResult):
    """Result from CodingSwarm."""

    code: Optional[CodeOutput] = None
    language: str = "python"
    loc: int = 0  # lines of code
    test_coverage: float = 0.0
    complexity_score: float = 0.0
    quality_score: float = 0.0
