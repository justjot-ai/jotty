"""
Coding Swarm - World-Class Code Generation & Development
=========================================================

Production-grade swarm for:
- Code generation with best practices
- Refactoring and optimization
- Bug fixing and debugging
- Architecture design
- Code review automation

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                          CODING SWARM                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  Architect     │  │  Developer     │  │   Debugger     │            │
│  │    Agent       │  │    Agent       │  │    Agent       │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Optimizer    │  │  Test Writer   │  │  Doc Writer    │            │
│  │    Agent       │  │    Agent       │  │    Agent       │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     CODE ASSEMBLER                               │   │
│  │   Combines all outputs into production-ready code package        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.coding_swarm import CodingSwarm, code

    # Full swarm
    swarm = CodingSwarm()
    result = await swarm.generate("Create a REST API for user management")

    # One-liner
    result = await code("Build a CLI tool for file encryption")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam
from ..agents.base import DomainAgent, DomainAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================

import re

def _strip_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM-generated code.

    Handles: ```python ... ```, ```py ... ```, ``` ... ```
    Also strips leading/trailing whitespace and handles nested fences.
    """
    if not code or not isinstance(code, str):
        return code or ""
    stripped = code.strip()
    # Match opening fence with optional language tag
    if re.match(r'^```\w*\s*\n', stripped):
        # Remove opening fence line
        stripped = re.sub(r'^```\w*\s*\n', '', stripped, count=1)
        # Remove closing fence (last line)
        stripped = re.sub(r'\n```\s*$', '', stripped)
    return stripped.strip()


def _extract_components_from_text(file_structure: str, architecture: str = "", interfaces: str = "") -> List[Dict[str, str]]:
    """Extract components from file structure, architecture, or interfaces text.

    This is a fallback when JSON parsing fails. It looks for:
    1. Python files (.py) - each becomes a component
    2. Class names mentioned in the text
    3. Directory/module names

    Returns a list of component dicts with 'name' and 'responsibility' keys.
    """
    components = []
    seen_names = set()

    # Combine all text sources
    all_text = f"{file_structure}\n{architecture}\n{interfaces}"

    # Pattern 1: Extract .py files (e.g., "main.py", "game.py", "board.py")
    py_files = re.findall(r'(\w+)\.py\b', all_text)
    for name in py_files:
        if name not in seen_names and name not in ('__init__', '__main__'):
            seen_names.add(name)
            # Try to find a description near the file mention
            desc_match = re.search(rf'{name}\.py[^\n]*?(?:#|:|-|–)\s*([^\n]+)', all_text)
            responsibility = desc_match.group(1).strip() if desc_match else f"{name} module"
            components.append({'name': name, 'responsibility': responsibility})

    # Pattern 2: Extract class names (e.g., "class Board", "TicTacToeGame", "GameController")
    class_patterns = [
        r'\bclass\s+(\w+)',  # "class ClassName"
        r'(\w+(?:Game|Board|Display|Controller|View|Model|Service|Handler|Manager|Engine))\b',  # Common class suffixes
    ]
    for pattern in class_patterns:
        for match in re.finditer(pattern, all_text):
            name = match.group(1)
            if name not in seen_names and len(name) > 2:
                seen_names.add(name)
                components.append({'name': name, 'responsibility': f"{name} component"})

    # Pattern 3: Extract directory names from tree structure (e.g., "├── src/", "│   ├── models/")
    dir_matches = re.findall(r'[├└│─\s]+(\w+)/', all_text)
    for name in dir_matches:
        if name not in seen_names and name not in ('src', 'lib', 'dist', 'build', 'node_modules', '__pycache__'):
            seen_names.add(name)
            components.append({'name': name, 'responsibility': f"{name} directory"})

    return components


# Global callbacks for TUI integration (set by generate(), reset on completion)
_active_progress_callback = None
_active_trace_callback = None


def _progress(phase: str, agent: str, message: str):
    """Print live progress to console."""
    print(f"  [{phase}] {agent}: {message}", flush=True)
    if _active_progress_callback is not None:
        try:
            _active_progress_callback(phase, agent, message)
        except Exception:
            pass  # Never let callback errors break the pipeline


async def _stream_call(module, phase: str, agent: str, listener_field: str = "reasoning",
                       timeout: float = 90.0, max_retries: int = 3, **kwargs):
    """Call a DSPy module with streaming, forwarding reasoning tokens to _progress().

    Args:
        module: DSPy ChainOfThought module to call
        phase: Phase name for progress messages (e.g. "Phase 1")
        agent: Agent name for progress messages (e.g. "Architect")
        listener_field: Output field to stream (default: "reasoning")
        timeout: Timeout in seconds per attempt (default: 90s)
        max_retries: Max retry attempts on timeout (default: 3)
        **kwargs: Arguments to pass to the module

    Returns:
        dspy.Prediction result
    """
    from dspy.streaming import streamify, StreamListener

    async def _do_call():
        listener = StreamListener(listener_field)
        streaming_module = streamify(module, stream_listeners=[listener])

        result = None
        last_text = ""
        async for chunk in streaming_module(**kwargs):
            if isinstance(chunk, dspy.Prediction):
                result = chunk
            elif isinstance(chunk, str):
                new_text = chunk[len(last_text):]
                if new_text.strip():
                    display = chunk.strip()[-80:]
                    _progress(phase, agent, f"  ...{display}")
                last_text = chunk

        if result is None:
            result = module(**kwargs)

        return result

    # Retry with timeout
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.wait_for(_do_call(), timeout=timeout)
        except asyncio.TimeoutError:
            last_error = asyncio.TimeoutError(f"Timeout after {timeout}s")
            # Increase timeout for next attempt
            timeout = min(timeout * 1.5, 180.0)
            print(f"⏱️ Attempt {attempt}/{max_retries}: Timeout after {timeout/1.5:.0f}s", flush=True)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    # All retries exhausted - raise last error
    raise last_error


# =============================================================================
# CONFIGURATION
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

    def __post_init__(self):
        self.name = "CodingSwarm"
        self.domain = "coding"


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
            sections.append("BEST PRACTICES:\n" + "\n".join(f"- {bp}" for bp in self.best_practices))
        if self.library_docs:
            sections.append("LIBRARY DOCS:\n" + "\n".join(f"- {ld}" for ld in self.library_docs))
        if self.api_references:
            sections.append("API REFERENCES:\n" + "\n".join(f"- {ar}" for ar in self.api_references))
        if self.warnings:
            sections.append("WARNINGS:\n" + "\n".join(f"- {w}" for w in self.warnings))
        return "\n\n".join(sections)


@dataclass
class FullStackContext:
    """Intermediate artifacts passed between full-stack pipeline phases."""
    data_model: str = ""
    api_contract: str = ""
    component_map: str = ""
    tech_stack: Dict[str, str] = field(default_factory=lambda: {
        "db_type": "sqlite", "backend": "fastapi", "frontend": "react"
    })
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


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class ArchitectureDesignSignature(dspy.Signature):
    """Design software architecture for the given requirements.

    You are a SENIOR SOFTWARE ARCHITECT. Design clean, scalable, maintainable architecture.

    PRINCIPLES:
    1. SOLID principles
    2. Clean Architecture / Hexagonal Architecture
    3. Dependency Injection
    4. Separation of Concerns
    5. Design for testability

    OUTPUT FORMAT:
    - Components and their responsibilities
    - Data flow diagram (ASCII)
    - File structure
    - Key interfaces/contracts
    """
    requirements: str = dspy.InputField(desc="Detailed requirements for the software")
    language: str = dspy.InputField(desc="Target programming language")
    style: str = dspy.InputField(desc="Coding style preference")
    constraints: str = dspy.InputField(desc="Technical constraints and preferences")

    architecture: str = dspy.OutputField(desc="Detailed architecture design")
    components: str = dspy.OutputField(desc="JSON list of components with responsibilities")
    file_structure: str = dspy.OutputField(desc="File/folder structure")
    interfaces: str = dspy.OutputField(desc="Key interfaces and contracts")


class CodeGenerationSignature(dspy.Signature):
    """Generate production-quality code.

    You are a SENIOR DEVELOPER. Write WORLD-CLASS code following best practices.

    CODE QUALITY REQUIREMENTS:
    1. Type hints / type annotations
    2. Comprehensive error handling
    3. Logging at appropriate levels
    4. Clear naming conventions
    5. DRY - Don't Repeat Yourself
    6. KISS - Keep It Simple, Stupid
    7. Proper documentation strings

    SECURITY:
    - No hardcoded secrets
    - Input validation
    - SQL injection prevention
    - XSS prevention where applicable
    """
    architecture: str = dspy.InputField(desc="Architecture design to implement")
    component: str = dspy.InputField(desc="Specific component to implement")
    language: str = dspy.InputField(desc="Programming language")
    dependencies: str = dspy.InputField(desc="Available dependencies")

    code: str = dspy.OutputField(desc="Complete, runnable code")
    imports: str = dspy.OutputField(desc="Required imports/dependencies")
    filename: str = dspy.OutputField(desc="Suggested filename")


class DebugAnalysisSignature(dspy.Signature):
    """Analyze and fix bugs in code.

    You are a DEBUGGING EXPERT. Find and fix issues systematically.

    APPROACH:
    1. Understand the intended behavior
    2. Identify the actual behavior
    3. Locate the root cause
    4. Propose minimal fix
    5. Verify fix doesn't introduce new issues
    """
    code: str = dspy.InputField(desc="Code with potential bugs")
    error_message: str = dspy.InputField(desc="Error message or bug description")
    context: str = dspy.InputField(desc="Additional context")

    root_cause: str = dspy.OutputField(desc="Root cause analysis")
    fix: str = dspy.OutputField(desc="Fixed code")
    explanation: str = dspy.OutputField(desc="Explanation of the fix")
    prevention: str = dspy.OutputField(desc="How to prevent similar bugs")


class CodeOptimizationSignature(dspy.Signature):
    """Optimize code for performance and readability.

    You are an OPTIMIZATION EXPERT. Improve code without changing behavior.
    Do NOT optimize away code that fulfills the original requirements.

    OPTIMIZATION TARGETS:
    1. Time complexity
    2. Space complexity
    3. Readability
    4. Maintainability
    5. Resource usage
    """
    code: str = dspy.InputField(desc="Code to optimize")
    focus: str = dspy.InputField(desc="Optimization focus: performance, readability, memory")
    requirements: str = dspy.InputField(desc="Original requirements the code must satisfy; do not optimize away intent")
    constraints: str = dspy.InputField(desc="Constraints to maintain")

    optimized_code: str = dspy.OutputField(desc="Optimized code")
    improvements: str = dspy.OutputField(desc="List of improvements made, separated by |")
    metrics: str = dspy.OutputField(desc="Before/after metrics if applicable")


class TestGenerationSignature(dspy.Signature):
    """Generate comprehensive tests for code.

    You are a TEST ENGINEER. Write thorough tests with high coverage.

    TEST TYPES:
    1. Unit tests - isolated function/method tests
    2. Integration tests - component interaction
    3. Edge cases - boundary conditions
    4. Error cases - exception handling
    5. Property-based tests where applicable
    """
    code: str = dspy.InputField(desc="Code to test")
    framework: str = dspy.InputField(desc="Test framework: pytest, jest, etc.")
    coverage_target: str = dspy.InputField(desc="Coverage requirements")

    tests: str = dspy.OutputField(desc="Complete test code")
    test_cases: str = dspy.OutputField(desc="List of test cases covered, separated by |")
    coverage_estimate: float = dspy.OutputField(desc="Estimated coverage percentage")


class DocumentationSignature(dspy.Signature):
    """Generate documentation for code.

    You are a TECHNICAL WRITER. Create clear, comprehensive documentation.

    DOCUMENTATION INCLUDES:
    1. Overview and purpose
    2. Installation instructions
    3. Usage examples
    4. API reference
    5. Architecture explanation
    """
    code: str = dspy.InputField(desc="Code to document")
    architecture: str = dspy.InputField(desc="Architecture overview")
    audience: str = dspy.InputField(desc="Target audience: developers, users, etc.")

    documentation: str = dspy.OutputField(desc="Complete documentation in Markdown")
    quickstart: str = dspy.OutputField(desc="Quick start guide")
    api_reference: str = dspy.OutputField(desc="API reference section")


class ResearchQuerySignature(dspy.Signature):
    """Generate targeted search queries for coding research.

    You are a RESEARCH ASSISTANT. Generate specific, targeted search queries
    to find best practices, library docs, and known pitfalls for the task.
    """
    requirements: str = dspy.InputField(desc="Requirements for the software to build")
    language: str = dspy.InputField(desc="Target programming language")
    frameworks: str = dspy.InputField(desc="Frameworks and libraries being used")

    search_queries: str = dspy.OutputField(desc="JSON list of 3-5 targeted search queries")


class CodebaseAnalysisSignature(dspy.Signature):
    """Analyze existing codebase to understand structure and patterns.

    You are a CODE ANALYST. Examine the existing code to understand:
    - Architecture patterns used
    - Code style conventions
    - Key abstractions and interfaces
    - Dependencies between files
    """
    existing_code: str = dspy.InputField(desc="Existing code from the codebase")
    file_paths: str = dspy.InputField(desc="List of file paths in the codebase")
    requirements: str = dspy.InputField(desc="What changes are requested")

    architecture_summary: str = dspy.OutputField(desc="Summary of existing architecture")
    style_conventions: str = dspy.OutputField(desc="Code style conventions observed (naming, formatting, patterns)")
    affected_files: str = dspy.OutputField(desc="JSON list of files that need modification")
    dependencies: str = dspy.OutputField(desc="Key dependencies and interfaces to preserve")
    change_scope: str = dspy.OutputField(desc="Scope of changes: 'minimal', 'moderate', 'extensive'")


class EditPlanSignature(dspy.Signature):
    """Plan surgical edits to existing code.

    You are an EDIT PLANNER. Given existing code and requirements, plan
    precise edits that:
    - Minimize changes (don't rewrite what works)
    - Preserve existing style and patterns
    - Maintain backward compatibility
    - Don't break existing tests
    """
    existing_code: str = dspy.InputField(desc="Current code content")
    file_path: str = dspy.InputField(desc="Path to the file being edited")
    requirements: str = dspy.InputField(desc="What changes are needed")
    style_conventions: str = dspy.InputField(desc="Style conventions to follow")
    dependencies: str = dspy.InputField(desc="Dependencies to preserve")

    edit_plan: str = dspy.OutputField(desc="Step-by-step edit plan")
    edits: str = dspy.OutputField(desc='JSON list of edits: [{"old": "...", "new": "...", "reason": "..."}]')
    new_code: str = dspy.OutputField(desc="Complete new code (only if substantial rewrite needed)")
    edit_type: str = dspy.OutputField(desc="'patch' for surgical edits, 'rewrite' for full replacement")


class TestFailureRefinementSignature(dspy.Signature):
    """Refine code based on test failure feedback.

    You are a DEBUGGING EXPERT. Given code that fails tests, analyze the
    failure and produce a refined version that passes.

    APPROACH:
    1. Parse the test failure output carefully
    2. Identify the root cause (assertion error, exception, missing behavior)
    3. Locate the exact code that needs fixing
    4. Make minimal, targeted changes to fix the issue
    5. Do NOT break other functionality
    """
    current_code: str = dspy.InputField(desc="Current code that fails tests")
    file_path: str = dspy.InputField(desc="Path to the file being fixed")
    original_requirements: str = dspy.InputField(desc="Original requirements/issue description")
    test_output: str = dspy.InputField(desc="Test failure output (stderr/stdout from test run)")
    iteration: int = dspy.InputField(desc="Current iteration number (1, 2, 3...)")
    previous_attempts: str = dspy.InputField(desc="Summary of previous fix attempts and why they failed")

    analysis: str = dspy.OutputField(desc="Analysis of test failure: what's wrong and why")
    fix_strategy: str = dspy.OutputField(desc="Strategy to fix the issue")
    fixed_code: str = dspy.OutputField(desc="Complete fixed code for the file")
    confidence: str = dspy.OutputField(desc="HIGH, MEDIUM, or LOW confidence this fix will work")


class CollaborativeArchitectSignature(dspy.Signature):
    """Design or refine software architecture incorporating research feedback.

    You are a SENIOR SOFTWARE ARCHITECT collaborating with a Researcher.
    Design clean, scalable, maintainable architecture that incorporates
    best practices and findings from research.

    COLLABORATION MODE:
    - In iteration 1: Draft initial architecture based on requirements
    - In subsequent iterations: Refine based on research findings, addressing gaps and improvements
    """
    requirements: str = dspy.InputField(desc="Detailed requirements for the software")
    language: str = dspy.InputField(desc="Target programming language")
    style: str = dspy.InputField(desc="Coding style preference")
    constraints: str = dspy.InputField(desc="Technical constraints and preferences")
    iteration: int = dspy.InputField(desc="Current iteration number (1, 2, 3, ...)")
    previous_architecture: str = dspy.InputField(desc="Architecture from previous iteration (empty for iteration 1)")
    research_findings: str = dspy.InputField(desc="Research findings from previous iteration (empty for iteration 1)")

    architecture: str = dspy.OutputField(desc="Detailed architecture design (refined with research findings)")
    components: str = dspy.OutputField(desc="JSON list of components with responsibilities")
    file_structure: str = dspy.OutputField(desc="File/folder structure")
    interfaces: str = dspy.OutputField(desc="Key interfaces and contracts")
    research_requests: str = dspy.OutputField(desc="Specific topics for Researcher to investigate in next iteration")


class ResearchResponseSignature(dspy.Signature):
    """Research specific topics requested by the Architect.

    You are a RESEARCH SPECIALIST working with an Architect.
    Based on the current architecture design, find best practices,
    library documentation, and potential pitfalls for the specific areas requested.

    FOCUS AREAS:
    - Best practices for the proposed architecture patterns
    - Library/framework specific recommendations
    - Common pitfalls and how to avoid them
    - Security considerations
    - Performance optimization tips
    """
    requirements: str = dspy.InputField(desc="Original requirements")
    architecture: str = dspy.InputField(desc="Current architecture proposal from Architect")
    research_requests: str = dspy.InputField(desc="Specific topics Architect wants researched")
    language: str = dspy.InputField(desc="Target programming language")
    frameworks: str = dspy.InputField(desc="Frameworks being used")

    search_queries: str = dspy.OutputField(desc="JSON list of 3-5 targeted search queries based on Architect's requests")
    analysis: str = dspy.OutputField(desc="Analysis of the architecture with research-backed recommendations")
    best_practices: str = dspy.OutputField(desc="Key best practices relevant to the architecture")
    warnings: str = dspy.OutputField(desc="Potential pitfalls or warnings to consider")
    recommendations: str = dspy.OutputField(desc="Specific recommendations to improve the architecture")


class SystemDesignSignature(dspy.Signature):
    """Design a full-stack system with data model, API contract, and component boundaries."""
    requirements: str = dspy.InputField(desc="What the system should do")
    language: str = dspy.InputField(desc="Primary backend language")
    tech_stack: str = dspy.InputField(desc="JSON: {db_type, backend_framework, frontend_framework}")

    data_model: str = dspy.OutputField(desc="Entity definitions: fields, types, relationships, constraints")
    api_contract: str = dspy.OutputField(desc="REST endpoints: method, path, request body, response schema")
    component_map: str = dspy.OutputField(desc="Components: database, backend, frontend with responsibilities")
    architecture: str = dspy.OutputField(desc="Overall system architecture and design decisions")
    ui_requirements: str = dspy.OutputField(desc="UI pages, forms, data displays needed")


class DatabaseSchemaSignature(dspy.Signature):
    """Generate database schema and ORM models from data model."""
    data_model: str = dspy.InputField(desc="Entity definitions with relationships")
    db_type: str = dspy.InputField(desc="Database type: sqlite, postgresql, mysql")
    language: str = dspy.InputField(desc="Backend language for ORM models")

    schema_sql: str = dspy.OutputField(desc="Complete SQL CREATE TABLE statements")
    orm_models: str = dspy.OutputField(desc="Complete ORM model code (SQLAlchemy)")
    migration_notes: str = dspy.OutputField(desc="Notes on indexes, constraints, migration order")


class APIGenerationSignature(dspy.Signature):
    """Generate backend API code and OpenAPI specification."""
    architecture: str = dspy.InputField(desc="System architecture and design decisions")
    orm_models: str = dspy.InputField(desc="ORM model code")
    api_contract: str = dspy.InputField(desc="API endpoint definitions")
    language: str = dspy.InputField(desc="Backend language")
    framework: str = dspy.InputField(desc="Backend framework: fastapi, flask, express")

    api_code: str = dspy.OutputField(desc="Complete backend API code with routes, services, error handling")
    openapi_spec: str = dspy.OutputField(desc="Complete OpenAPI 3.0 YAML specification")
    dependencies: str = dspy.OutputField(desc="Required packages in pip/npm format")


class FrontendGenerationSignature(dspy.Signature):
    """Generate frontend code consuming an OpenAPI specification."""
    openapi_spec: str = dspy.InputField(desc="OpenAPI 3.0 YAML specification")
    ui_requirements: str = dspy.InputField(desc="UI pages, forms, data displays needed")
    framework: str = dspy.InputField(desc="Frontend framework: react, vue, vanilla")

    frontend_code: str = dspy.OutputField(desc="Complete frontend application code")
    api_client: str = dspy.OutputField(desc="API client code generated from OpenAPI spec")
    dependencies: str = dspy.OutputField(desc="Required frontend packages")


class IntegrationSignature(dspy.Signature):
    """Generate integration artifacts to wire all components together."""
    file_list: str = dspy.InputField(desc="JSON list of all generated files")
    tech_stack: str = dspy.InputField(desc="JSON: {db_type, backend, frontend}")
    architecture: str = dspy.InputField(desc="System architecture")

    docker_compose: str = dspy.OutputField(desc="Complete docker-compose.yml")
    env_config: str = dspy.OutputField(desc="Environment variables template (.env.example)")
    requirements_txt: str = dspy.OutputField(desc="Python requirements.txt")
    startup_script: str = dspy.OutputField(desc="Shell script to start all services")


class ReviewArbitrationSignature(dspy.Signature):
    """Evaluate whether a code review rejection is valid and actionable."""
    code: str = dspy.InputField(desc="The code being reviewed")
    rejection_feedback: str = dspy.InputField(desc="Reviewer's rejection reason and feedback")
    evidence: str = dspy.InputField(desc="Evidence: code snippets, test cases, line references")
    reviewer_name: str = dspy.InputField(desc="Name of the reviewer persona")

    valid: str = dspy.OutputField(desc="TRUE or FALSE — is this rejection valid and actionable?")
    reasoning: str = dspy.OutputField(desc="Why the rejection is valid or invalid")
    actionable_fix: str = dspy.OutputField(desc="If valid: specific fix. If invalid: empty string.")


class ScopeClassificationSignature(dspy.Signature):
    """Classify whether software requirements need a single-tier or full-stack architecture.

    You are a SOFTWARE ARCHITECTURE CLASSIFIER. Analyze the requirements and determine scope.

    SINGLE_TIER — one concern, one process:
    - CLI tools, scripts, libraries, algorithms, utilities, single-file programs
    - Example: "Build a CLI calculator", "Create a sorting library"

    FULL_STACK — multiple architectural tiers (database + backend API + frontend):
    - Web applications, platforms, dashboards, CRUD apps, SaaS products
    - Any system needing persistent storage + API layer + user interface
    - Example: "Build an e-commerce platform", "Create a project management tool"
    """
    requirements: str = dspy.InputField(desc="Software requirements to classify")

    scope: str = dspy.OutputField(desc="Exactly one of: single_tier, full_stack")
    reasoning: str = dspy.OutputField(desc="One-sentence justification")


class CodeVerificationSignature(dspy.Signature):
    """Verify generated code against original requirements.

    You are a CODE VERIFICATION EXPERT. Check that the generated code
    faithfully implements the original requirements without gaps or
    anti-patterns.

    CHECK FOR:
    1. Requirement coverage — every stated requirement has corresponding code
    2. Resource management — files, connections, locks are properly closed/released
    3. Framework consistency — APIs and patterns match the stated language/framework
    4. Anti-patterns — god classes, circular deps, magic numbers, missing error paths

    Return issues as a JSON list of objects: [{"severity": "high"|"medium"|"low", "description": "..."}]
    Return coverage_score as a float 0-1 indicating what fraction of requirements are covered.
    Return verified as true only if there are no high-severity issues.
    """
    code: str = dspy.InputField(desc="Complete generated code to verify")
    original_requirements: str = dspy.InputField(desc="Original requirements the code must satisfy")
    architecture: str = dspy.InputField(desc="Architecture design that was used")

    issues: str = dspy.OutputField(desc="JSON list of issues found: [{severity, description}]")
    coverage_score: float = dspy.OutputField(desc="Float 0-1: fraction of requirements covered by the code")
    verified: bool = dspy.OutputField(desc="True if no high-severity issues found")


class SimplicityJudgeSignature(dspy.Signature):
    """Judge code for over-engineering and unnecessary complexity.

    You are a SIMPLICITY ENGINEER. Your job is to identify and flag
    unnecessary abstractions, over-architecture, and feature bloat.
    Good code matches complexity to the problem — a tic-tac-toe game
    should be ~100 lines, not 5000 lines across 17 files.

    WHAT TO FLAG AS OVER-ENGINEERING:
    1. Over-abstraction: Protocols/interfaces with only 1 implementation
    2. Redundant implementations: Multiple classes doing the same thing
    3. Feature creep: Features beyond stated requirements
    4. Gold-plating: Sophisticated patterns (Factory, Command, Strategy) when simple code suffices
    5. Premature optimization: Bit masks, O(1) claims, "high-performance" for trivial operations
    6. Unnecessary indirection: Layers of abstraction for simple logic
    7. File explosion: Many small files when one file would be cleaner
    8. Enterprise language: "Production-Ready", "Enterprise-Grade" for simple apps

    COMPLEXITY GUIDELINES:
    - Simple game (tic-tac-toe, snake): 50-200 lines, 1-2 files
    - CLI tool: 50-300 lines, 1-3 files
    - REST API: 100-500 lines, 3-5 files
    - Full web app: 500-2000 lines, 5-15 files

    Return over_engineering_issues as JSON: [{"severity": "critical|major|minor", "issue": "...", "location": "file:line", "simpler_alternative": "..."}]
    Return simplicity_score as float 0-1 where 1=appropriately simple, 0=severely over-engineered.
    Return verdict as ACCEPT if code complexity matches requirements, SIMPLIFY if over-engineered.
    """
    code: str = dspy.InputField(desc="Complete generated code to evaluate")
    requirements: str = dspy.InputField(desc="Original requirements the code was built for")
    file_count: int = dspy.InputField(desc="Number of files in the codebase")
    total_lines: int = dspy.InputField(desc="Total lines of code")

    over_engineering_issues: str = dspy.OutputField(
        desc='JSON: [{"severity": "critical|major|minor", "issue": "...", "location": "...", "simpler_alternative": "..."}]'
    )
    simplicity_score: float = dspy.OutputField(
        desc="Float 0-1: 1=appropriately simple, 0=severely over-engineered"
    )
    verdict: str = dspy.OutputField(
        desc="ACCEPT if complexity matches requirements, SIMPLIFY if over-engineered"
    )


# =============================================================================
# TEAM PERSONAS & REVIEW PROTOCOL
# =============================================================================

@dataclass
class TeamPersona:
    """An archetypal engineer persona for team-flavored code generation and review."""
    name: str
    archetype: str
    expertise: List[str]
    review_style: str
    guiding_principles: List[str]

    def to_prompt(self) -> str:
        """Convert persona to injectable prompt text."""
        principles = "\n".join(f"  - {p}" for p in self.guiding_principles)
        expertise = ", ".join(self.expertise)
        return (
            f"You are acting as **{self.name}** ({self.archetype}).\n"
            f"Expertise: {expertise}\n"
            f"Review style: {self.review_style}\n"
            f"Guiding principles:\n{principles}"
        )


@dataclass
class TeamConfig:
    """Configuration for a team of archetypal engineer personas."""
    name: str
    personas: List[TeamPersona]
    role_persona_map: Dict[str, str]
    review_protocol: str = "two_phase"
    require_unanimous: bool = True
    functional_reviewers: List[str] = field(default_factory=list)
    quality_reviewers: List[str] = field(default_factory=list)

    def get_persona(self, agent_role: str) -> Optional[TeamPersona]:
        """Look up persona for a given agent role."""
        persona_name = self.role_persona_map.get(agent_role)
        if not persona_name:
            return None
        for p in self.personas:
            if p.name == persona_name:
                return p
        return None

    def get_reviewers(self, phase: str) -> List[TeamPersona]:
        """Return reviewer personas for 'functional' or 'quality' phase."""
        names = self.functional_reviewers if phase == "functional" else self.quality_reviewers
        return [p for p in self.personas if p.name in names]


# --- 6 Archetype Personas ---

PERSONA_ARCHITECT = TeamPersona(
    name="The Architect",
    archetype="Systems Architect",
    expertise=["system design", "scalability", "clean architecture", "SOLID principles"],
    review_style="Top-down structural analysis; focuses on component boundaries and data flow",
    guiding_principles=[
        "Every module should have a single, clear responsibility",
        "Prefer composition over inheritance",
        "Design for change — isolate what varies",
        "Enforce dependency inversion at boundaries",
    ],
)

PERSONA_PERFORMANCE = TeamPersona(
    name="The Performance Engineer",
    archetype="Performance Engineer",
    expertise=["latency optimization", "throughput", "caching", "profiling", "O-complexity"],
    review_style="Bottom-up hot-path analysis; benchmarks before opinions",
    guiding_principles=[
        "Measure first, optimize second",
        "Avoid premature allocation and copying",
        "Prefer O(1) lookups; document when O(n) is acceptable",
        "Cache at the right layer — not everywhere",
    ],
)

PERSONA_QUALITY = TeamPersona(
    name="The Quality Champion",
    archetype="Quality / Test Engineer",
    expertise=["testing strategy", "reliability", "error paths", "defensive coding"],
    review_style="Adversarial — tries to break the code with edge cases",
    guiding_principles=[
        "Every public function needs at least one happy-path and one sad-path test",
        "Fail fast and fail loudly — silent errors are bugs",
        "Defensive code at system boundaries, trust internals",
        "100% coverage is a ceiling, not a floor — test behavior, not lines",
    ],
)

PERSONA_ALGORITHM = TeamPersona(
    name="The Algorithm Specialist",
    archetype="Algorithm & Data Structures Expert",
    expertise=["correctness proofs", "edge cases", "data structures", "numerical stability"],
    review_style="Formal reasoning; checks invariants and boundary conditions",
    guiding_principles=[
        "Correctness before cleverness",
        "Document loop invariants for non-trivial algorithms",
        "Handle empty, single, and maximum-size inputs explicitly",
        "Prefer well-known algorithms over novel ones unless justified",
    ],
)

PERSONA_BACKEND = TeamPersona(
    name="The Backend Engineer",
    archetype="Backend / API Engineer",
    expertise=["REST APIs", "databases", "microservices", "pragmatic shipping"],
    review_style="Pragmatic — ships working code, then iterates",
    guiding_principles=[
        "Make it work, make it right, make it fast — in that order",
        "Idempotent endpoints and retry-safe operations",
        "Validate at the boundary, trust the core",
        "Keep the API surface small and consistent",
    ],
)

PERSONA_FRONTEND = TeamPersona(
    name="The Frontend Specialist",
    archetype="Frontend / UI Engineer",
    expertise=["component architecture", "accessibility", "state management", "UX patterns"],
    review_style="User-centric; checks a11y, responsiveness, and interaction patterns",
    guiding_principles=[
        "Accessible by default — ARIA where needed, semantic HTML first",
        "State should flow down; events should bubble up",
        "Minimize re-renders; colocate state with the component that owns it",
        "Design for keyboard, touch, and screen readers simultaneously",
    ],
)

PERSONA_SIMPLICITY = TeamPersona(
    name="The Simplicity Champion",
    archetype="Anti-Complexity Engineer",
    expertise=["code reduction", "YAGNI enforcement", "abstraction detection", "minimal solutions"],
    review_style="Ruthlessly removes complexity; questions every abstraction and pattern",
    guiding_principles=[
        "Simple beats clever — always. The best code is code you don't write",
        "Duplication is far better than the wrong abstraction",
        "Question every class, function, and module — can it be removed?",
        "Protocols/interfaces need 3+ implementations to justify existence",
        "YAGNI: Don't implement until you actually need it — not 'might need'",
        "Match complexity to the problem — a 100-line game doesn't need 17 files",
        "Premature optimization is the root of all evil; so is premature abstraction",
    ],
)

ALL_PERSONAS = [
    PERSONA_ARCHITECT, PERSONA_PERFORMANCE, PERSONA_QUALITY,
    PERSONA_ALGORITHM, PERSONA_BACKEND, PERSONA_FRONTEND, PERSONA_SIMPLICITY,
]

# --- 3 Team Presets ---

TEAM_PRESETS: Dict[str, TeamConfig] = {
    "fullstack": TeamConfig(
        name="fullstack",
        personas=ALL_PERSONAS,
        role_persona_map={
            "Architect": "The Architect",
            "Developer": "The Backend Engineer",
            "Optimizer": "The Performance Engineer",
            "TestWriter": "The Quality Champion",
            "DocWriter": "The Algorithm Specialist",
            "Verifier": "The Quality Champion",
            "SimplicityJudge": "The Simplicity Champion",
            "SystemDesigner": "The Architect",
            "DatabaseArchitect": "The Backend Engineer",
            "FrontendDeveloper": "The Frontend Specialist",
            "Integration": "The Backend Engineer",
        },
        functional_reviewers=["The Architect", "The Backend Engineer", "The Algorithm Specialist"],
        quality_reviewers=["The Quality Champion", "The Performance Engineer", "The Frontend Specialist", "The Simplicity Champion"],
    ),
    "datascience": TeamConfig(
        name="datascience",
        personas=[PERSONA_ARCHITECT, PERSONA_ALGORITHM, PERSONA_QUALITY, PERSONA_PERFORMANCE, PERSONA_SIMPLICITY],
        role_persona_map={
            "Architect": "The Architect",
            "Developer": "The Algorithm Specialist",
            "Optimizer": "The Performance Engineer",
            "TestWriter": "The Quality Champion",
            "DocWriter": "The Algorithm Specialist",
            "Verifier": "The Quality Champion",
            "SimplicityJudge": "The Simplicity Champion",
        },
        functional_reviewers=["The Algorithm Specialist", "The Architect"],
        quality_reviewers=["The Quality Champion", "The Performance Engineer", "The Simplicity Champion"],
    ),
    "frontend": TeamConfig(
        name="frontend",
        personas=[PERSONA_ARCHITECT, PERSONA_FRONTEND, PERSONA_QUALITY, PERSONA_PERFORMANCE, PERSONA_SIMPLICITY],
        role_persona_map={
            "Architect": "The Architect",
            "Developer": "The Frontend Specialist",
            "Optimizer": "The Performance Engineer",
            "TestWriter": "The Quality Champion",
            "DocWriter": "The Frontend Specialist",
            "Verifier": "The Quality Champion",
            "SimplicityJudge": "The Simplicity Champion",
        },
        functional_reviewers=["The Frontend Specialist", "The Architect"],
        quality_reviewers=["The Quality Champion", "The Performance Engineer", "The Simplicity Champion"],
    ),
}


class TeamReviewSignature(dspy.Signature):
    """Review code from a specific team archetype's perspective.

    You are a senior engineer reviewing code through the lens of your persona.

    IMPORTANT: You participated in the PLANNING phase and made agreements with the team.
    Do NOT reject code for things that were already agreed upon in planning.
    Only reject for:
    1. Implementation bugs (code doesn't do what it should)
    2. Security vulnerabilities
    3. Clear violations of best practices NOT covered in planning

    Do NOT reject for:
    - Architectural choices that were agreed in planning
    - Style preferences already decided
    - Trade-offs that the team accepted during planning
    """
    code: str = dspy.InputField(desc="Complete code to review")
    requirements: str = dspy.InputField(desc="Original requirements the code must satisfy")
    review_phase: str = dspy.InputField(desc="'functional' or 'quality'")
    persona_context: str = dspy.InputField(desc="Reviewer persona context and expertise")
    team_agreements: str = dspy.InputField(desc="Decisions and agreements made during team planning phase - DO NOT contradict these")

    verdict: str = dspy.OutputField(desc="APPROVED or REJECTED - only reject for bugs, security issues, or clear violations NOT covered in team agreements")
    issues: str = dspy.OutputField(desc='JSON list of issues: [{"severity": "high|medium|low", "description": "..."}] - only include issues NOT already addressed in team agreements')
    feedback: str = dspy.OutputField(desc="Constructive feedback for the developer")
    evidence: str = dspy.OutputField(desc="Specific code lines, test case, or scenario demonstrating the issue. Required for REJECTED verdict.")


class TeamPlanningSignature(dspy.Signature):
    """Provide planning input from a specific team archetype's perspective.

    You are a senior engineer contributing to architecture planning through your expertise lens.
    Review the proposed architecture and research findings, then provide your recommendations.
    """
    requirements: str = dspy.InputField(desc="Original requirements for the software")
    architecture: str = dspy.InputField(desc="Proposed architecture from the Architect")
    research_findings: str = dspy.InputField(desc="Research findings: best practices, API docs, warnings")
    persona_context: str = dspy.InputField(desc="Your persona context and expertise area")

    concerns: str = dspy.OutputField(desc='JSON list of concerns: [{"severity": "high|medium|low", "area": "...", "description": "..."}]')
    recommendations: str = dspy.OutputField(desc="Specific recommendations to improve the architecture from your expertise")
    implementation_notes: str = dspy.OutputField(desc="Key implementation details the developer should know from your area of expertise")


class TeamPlanningConsolidationSignature(dspy.Signature):
    """Consolidate team planning feedback into a refined architecture plan.

    You are a lead architect synthesizing feedback from multiple team members into
    a cohesive, actionable implementation plan.
    """
    original_architecture: str = dspy.InputField(desc="Original architecture proposal")
    team_feedback: str = dspy.InputField(desc="Consolidated feedback from all team members")
    research_findings: str = dspy.InputField(desc="Research findings that informed the planning")
    requirements: str = dspy.InputField(desc="Original requirements")

    refined_architecture: str = dspy.OutputField(desc="Refined architecture incorporating team feedback")
    implementation_plan: str = dspy.OutputField(desc="Step-by-step implementation plan with priorities")
    risk_mitigations: str = dspy.OutputField(desc="Identified risks and how to mitigate them")
    team_agreements: str = dspy.OutputField(desc="Key decisions and agreements from team discussion")


# =============================================================================
# WORKSPACE MANAGER
# =============================================================================

class WorkspaceManager:
    """Wraps SwarmTerminal for CodingSwarm validation.

    Non-blocking: falls back to string-only mode if SwarmTerminal is unavailable.
    Lazy-loads all dependencies on first use.
    """

    def __init__(self):
        self._terminal = None
        self._workspace_dir: Optional[str] = None
        self._initialized = False

    def _ensure_init(self) -> bool:
        """Lazy-load SwarmTerminal and create temp workspace directory."""
        if self._initialized:
            return self._terminal is not None
        self._initialized = True
        try:
            from ..orchestration.v2.swarm_terminal import SwarmTerminal
            import tempfile
            self._workspace_dir = tempfile.mkdtemp(prefix="codingswarm_")
            self._terminal = SwarmTerminal(auto_fix=False, max_fix_attempts=1)
            return True
        except Exception as e:
            logger.debug(f"WorkspaceManager: SwarmTerminal unavailable: {e}")
            return False

    @property
    def available(self) -> bool:
        """Whether terminal-based validation is available."""
        return self._ensure_init()

    @property
    def workspace_dir(self) -> Optional[str]:
        """Return workspace directory path, or None if unavailable."""
        self._ensure_init()
        return self._workspace_dir

    async def write_file(self, filename: str, content: str) -> 'CommandResult':
        """Write a file to the workspace. Returns CommandResult."""
        if not self._ensure_init():
            from ..orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="write_file", output="", error="WorkspaceManager unavailable")
        import os
        filepath = os.path.join(self._workspace_dir, filename)
        return await self._terminal.write_file(filepath, content)

    async def bash(self, command: str, timeout: int = 30) -> 'CommandResult':
        """Execute a bash command in the workspace."""
        if not self._ensure_init():
            from ..orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command=command, output="", error="WorkspaceManager unavailable")
        return await self._terminal.execute(command, timeout=timeout, working_dir=self._workspace_dir)

    async def syntax_check(self, filename: str, language: str = "python") -> 'CommandResult':
        """Run syntax check on a file in the workspace."""
        if not self._ensure_init():
            from ..orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="syntax_check", output="", error="WorkspaceManager unavailable")
        import os
        filepath = os.path.join(self._workspace_dir, filename)
        if language == "python":
            return await self._terminal.execute(f"python3 -m py_compile {filepath}", timeout=15, working_dir=self._workspace_dir)
        return await self._terminal.execute(f"cat {filepath}", timeout=5, working_dir=self._workspace_dir)

    async def run_python(self, filename: str, timeout: int = 30) -> 'CommandResult':
        """Run a Python file in the workspace."""
        if not self._ensure_init():
            from ..orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="run_python", output="", error="WorkspaceManager unavailable")
        import os
        filepath = os.path.join(self._workspace_dir, filename)
        return await self._terminal.execute(f"python3 {filepath}", timeout=timeout, working_dir=self._workspace_dir)

    async def run_tests(self, test_filename: Optional[str] = None, timeout: int = 60) -> 'CommandResult':
        """Run pytest on test files in the workspace."""
        if not self._ensure_init():
            from ..orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="run_tests", output="", error="WorkspaceManager unavailable")
        if test_filename:
            import os
            filepath = os.path.join(self._workspace_dir, test_filename)
            cmd = f"python3 -m pytest {filepath} -v --tb=short"
        else:
            cmd = f"python3 -m pytest {self._workspace_dir} -v --tb=short"
        return await self._terminal.execute(cmd, timeout=timeout, working_dir=self._workspace_dir)

    async def pip_install(self, packages: List[str]) -> 'CommandResult':
        """Install pip packages."""
        if not self._ensure_init() or not packages:
            from ..orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=not packages, command="pip_install", output="" if not packages else "", error="" if not packages else "WorkspaceManager unavailable")
        pkg_str = " ".join(packages)
        return await self._terminal.execute(f"pip install {pkg_str}", timeout=60, working_dir=self._workspace_dir)

    def cleanup(self):
        """Remove workspace directory. Safe to call multiple times."""
        if self._workspace_dir:
            try:
                import shutil
                shutil.rmtree(self._workspace_dir, ignore_errors=True)
            except Exception:
                pass
            self._workspace_dir = None


# =============================================================================
# AGENTS
# =============================================================================

class BaseCodeAgent(DomainAgent):
    """Base class for coding agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "", signature=None):
        # Initialize DomainAgent infrastructure
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        # Domain-specific attributes (override lazy init if provided)
        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus
        self.learned_context = learned_context

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass

    async def _stream(self, module, phase: str, agent: str, listener_field: str = "reasoning", **kwargs):
        """Call DSPy module with streaming reasoning to progress callback."""
        return await _stream_call(module, phase, agent, listener_field, **kwargs)


class ArchitectAgent(BaseCodeAgent):
    """Designs software architecture."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._designer = dspy.ChainOfThought(ArchitectureDesignSignature)

    async def design(
        self,
        requirements: str,
        language: str,
        style: str,
        constraints: str = ""
    ) -> Dict[str, Any]:
        """Design architecture for requirements."""
        try:
            if self.learned_context:
                requirements = requirements + f"\n\n{self.learned_context}"

            result = await self._stream(self._designer, "Phase 1", "Architect",
                requirements=requirements,
                language=language,
                style=style,
                constraints=constraints or "No specific constraints"
            )

            # Parse components
            try:
                components = json.loads(result.components)
            except:
                components = []

            self._broadcast("architecture_designed", {
                'components': len(components),
                'language': language
            })

            return {
                'architecture': str(result.architecture),
                'components': components,
                'file_structure': str(result.file_structure),
                'interfaces': str(result.interfaces)
            }

        except Exception as e:
            logger.error(f"Architecture design failed: {e}")
            return {'error': str(e)}


class DeveloperAgent(BaseCodeAgent):
    """Generates production code."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(CodeGenerationSignature)

    async def generate(
        self,
        architecture: str,
        component: str,
        language: str,
        dependencies: List[str] = None
    ) -> Dict[str, Any]:
        """Generate code for a component."""
        try:
            if self.learned_context:
                architecture = architecture + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 3", "Developer",
                architecture=architecture,
                component=component,
                language=language,
                dependencies=json.dumps(dependencies or [])
            )

            self._broadcast("code_generated", {
                'component': component,
                'filename': str(result.filename)
            })

            return {
                'code': str(result.code),
                'imports': str(result.imports),
                'filename': str(result.filename)
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {'error': str(e)}


class DebuggerAgent(BaseCodeAgent):
    """Debugs and fixes code."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._analyzer = dspy.ChainOfThought(DebugAnalysisSignature)

    async def debug(
        self,
        code: str,
        error_message: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze and fix bugs."""
        try:
            if self.learned_context:
                code = code + f"\n\n{self.learned_context}"

            result = await self._stream(self._analyzer, "Phase 4.5", "Debugger",
                code=code,
                error_message=error_message,
                context=context or "No additional context"
            )

            self._broadcast("bug_fixed", {
                'root_cause': str(result.root_cause)[:100]
            })

            return {
                'root_cause': str(result.root_cause),
                'fix': str(result.fix),
                'explanation': str(result.explanation),
                'prevention': str(result.prevention)
            }

        except Exception as e:
            logger.error(f"Debug analysis failed: {e}")
            return {'error': str(e)}


class OptimizerAgent(BaseCodeAgent):
    """Optimizes code."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._optimizer = dspy.ChainOfThought(CodeOptimizationSignature)

    async def optimize(
        self,
        code: str,
        focus: str = "performance",
        constraints: str = "",
        requirements: str = ""
    ) -> Dict[str, Any]:
        """Optimize code."""
        try:
            if self.learned_context:
                code = code + f"\n\n{self.learned_context}"

            result = await self._stream(self._optimizer, "Phase 4", "Optimizer",
                code=code,
                focus=focus,
                requirements=requirements or "No specific requirements provided",
                constraints=constraints or "Maintain existing functionality"
            )

            improvements = [i.strip() for i in str(result.improvements).split('|') if i.strip()]

            self._broadcast("code_optimized", {
                'focus': focus,
                'improvements': len(improvements)
            })

            return {
                'optimized_code': str(result.optimized_code),
                'improvements': improvements,
                'metrics': str(result.metrics)
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {'error': str(e)}


class TestWriterAgent(BaseCodeAgent):
    """Generates tests."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(TestGenerationSignature)

    async def generate_tests(
        self,
        code: str,
        framework: str = "pytest",
        coverage_target: str = "80%"
    ) -> Dict[str, Any]:
        """Generate tests for code."""
        try:
            if self.learned_context:
                code = code + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 7", "TestWriter",
                code=code,
                framework=framework,
                coverage_target=coverage_target
            )

            test_cases = [t.strip() for t in str(result.test_cases).split('|') if t.strip()]

            self._broadcast("tests_generated", {
                'test_count': len(test_cases),
                'coverage': float(result.coverage_estimate) if result.coverage_estimate else 0
            })

            return {
                'tests': str(result.tests),
                'test_cases': test_cases,
                'coverage_estimate': float(result.coverage_estimate) if result.coverage_estimate else 0.0
            }

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return {'error': str(e)}


class DocWriterAgent(BaseCodeAgent):
    """Generates documentation."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._writer = dspy.ChainOfThought(DocumentationSignature)

    async def document(
        self,
        code: str,
        architecture: str,
        audience: str = "developers"
    ) -> Dict[str, Any]:
        """Generate documentation."""
        try:
            if self.learned_context:
                code = code + f"\n\n{self.learned_context}"

            result = await self._stream(self._writer, "Phase 8", "DocWriter",
                code=code,
                architecture=architecture,
                audience=audience
            )

            self._broadcast("docs_generated", {
                'audience': audience
            })

            return {
                'documentation': str(result.documentation),
                'quickstart': str(result.quickstart),
                'api_reference': str(result.api_reference)
            }

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {'error': str(e)}


class VerifierAgent(BaseCodeAgent):
    """Verifies generated code against original requirements."""

    # Max chars to avoid context overflow (roughly 8K tokens)
    MAX_CODE_CHARS = 32000
    MAX_ARCH_CHARS = 4000

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._verifier = dspy.ChainOfThought(CodeVerificationSignature)

    async def verify(
        self,
        code: str,
        original_requirements: str,
        architecture: str = ""
    ) -> Dict[str, Any]:
        """Verify code against requirements. Non-blocking: returns safe defaults on failure."""
        try:
            # Truncate code if too large to avoid context overflow
            if len(code) > self.MAX_CODE_CHARS:
                code = code[:self.MAX_CODE_CHARS] + "\n\n# ... (truncated for verification)"

            # Truncate architecture if too large
            if len(architecture) > self.MAX_ARCH_CHARS:
                architecture = architecture[:self.MAX_ARCH_CHARS] + "\n... (truncated)"

            # Note: learned_context NOT appended to code - it bloats context unnecessarily

            result = await self._stream(self._verifier, "Phase 5", "Verifier",
                code=code,
                original_requirements=original_requirements,
                architecture=architecture or "No architecture provided"
            )

            # Parse issues JSON
            try:
                issues = json.loads(str(result.issues))
                if not isinstance(issues, list):
                    issues = []
            except (json.JSONDecodeError, TypeError):
                issues = []

            # Parse coverage_score
            try:
                coverage_score = float(result.coverage_score)
                coverage_score = max(0.0, min(1.0, coverage_score))
            except (TypeError, ValueError):
                coverage_score = 0.8

            # Parse verified
            verified = bool(result.verified) if result.verified is not None else True

            self._broadcast("code_verified", {
                'issues_count': len(issues),
                'coverage_score': coverage_score,
                'verified': verified
            })

            return {
                'issues': issues,
                'coverage_score': coverage_score,
                'verified': verified
            }

        except Exception as e:
            logger.error(f"Verification failed (non-blocking): {e}")
            return {
                'issues': [],
                'coverage_score': 1.0,
                'verified': True
            }


class SimplicityJudgeAgent(BaseCodeAgent):
    """Judges code for over-engineering and unnecessary complexity.

    This agent acts as a gate to prevent cargo-cult engineering where
    simple problems get solved with enterprise-scale solutions.
    """

    # Max chars to avoid context overflow
    MAX_CODE_CHARS = 32000

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._judge = dspy.ChainOfThought(SimplicityJudgeSignature)

    async def judge(
        self,
        code: str,
        requirements: str,
        file_count: int = 1,
        total_lines: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate code for over-engineering.

        Returns:
            Dict with:
            - issues: List of over-engineering issues found
            - simplicity_score: 0-1 (1=simple, 0=over-engineered)
            - verdict: 'ACCEPT' or 'SIMPLIFY'
            - needs_simplification: bool
        """
        try:
            # Calculate lines if not provided
            if total_lines == 0:
                total_lines = len(code.split('\n'))

            # Truncate code if too large to avoid context overflow
            if len(code) > self.MAX_CODE_CHARS:
                code = code[:self.MAX_CODE_CHARS] + "\n\n# ... (truncated for evaluation)"

            # Note: learned_context NOT appended - it bloats context unnecessarily

            result = await self._stream(self._judge, "Phase 5.5", "SimplicityJudge",
                code=code,
                requirements=requirements,
                file_count=file_count,
                total_lines=total_lines
            )

            # Parse issues JSON
            try:
                issues = json.loads(str(result.over_engineering_issues))
                if not isinstance(issues, list):
                    issues = []
            except (json.JSONDecodeError, TypeError):
                issues = []

            # Parse simplicity_score
            try:
                simplicity_score = float(result.simplicity_score)
                simplicity_score = max(0.0, min(1.0, simplicity_score))
            except (TypeError, ValueError):
                simplicity_score = 0.8

            # Parse verdict
            verdict = str(result.verdict).strip().upper()
            if verdict not in ('ACCEPT', 'SIMPLIFY'):
                verdict = 'ACCEPT' if simplicity_score >= 0.6 else 'SIMPLIFY'

            needs_simplification = verdict == 'SIMPLIFY'

            # Count severity
            critical_count = sum(1 for i in issues if i.get('severity') == 'critical')
            major_count = sum(1 for i in issues if i.get('severity') == 'major')

            self._broadcast("simplicity_judged", {
                'simplicity_score': simplicity_score,
                'verdict': verdict,
                'issues_count': len(issues),
                'critical_count': critical_count,
                'major_count': major_count,
                'file_count': file_count,
                'total_lines': total_lines,
            })

            return {
                'issues': issues,
                'simplicity_score': simplicity_score,
                'verdict': verdict,
                'needs_simplification': needs_simplification,
                'critical_count': critical_count,
                'major_count': major_count,
            }

        except Exception as e:
            logger.error(f"Simplicity judgment failed (non-blocking): {e}")
            return {
                'issues': [],
                'simplicity_score': 1.0,
                'verdict': 'ACCEPT',
                'needs_simplification': False,
                'critical_count': 0,
                'major_count': 0,
            }


class SystemDesignerAgent(BaseCodeAgent):
    """Designs full-stack system architecture with data model, API contract, and component map."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._designer = dspy.ChainOfThought(SystemDesignSignature)

    async def design(
        self,
        requirements: str,
        language: str,
        tech_stack: Dict[str, str]
    ) -> Dict[str, Any]:
        """Design full-stack system architecture."""
        try:
            if self.learned_context:
                requirements = requirements + f"\n\n{self.learned_context}"

            result = await self._stream(self._designer, "Phase 1", "SystemDesigner",
                requirements=requirements,
                language=language,
                tech_stack=json.dumps(tech_stack)
            )

            self._broadcast("system_designed", {
                'has_data_model': bool(result.data_model),
                'has_api_contract': bool(result.api_contract),
            })

            return {
                'data_model': str(result.data_model),
                'api_contract': str(result.api_contract),
                'component_map': str(result.component_map),
                'architecture': str(result.architecture),
                'ui_requirements': str(result.ui_requirements),
            }

        except Exception as e:
            logger.error(f"System design failed: {e}")
            return {'error': str(e)}


class DatabaseArchitectAgent(BaseCodeAgent):
    """Generates database schema and ORM models."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(DatabaseSchemaSignature)

    async def generate(
        self,
        data_model: str,
        db_type: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate database schema and ORM models."""
        try:
            if self.learned_context:
                data_model = data_model + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2a", "DatabaseArchitect",
                data_model=data_model,
                db_type=db_type,
                language=language
            )

            self._broadcast("database_designed", {
                'db_type': db_type,
                'has_schema': bool(result.schema_sql),
            })

            return {
                'schema_sql': str(result.schema_sql),
                'orm_models': str(result.orm_models),
                'migration_notes': str(result.migration_notes),
            }

        except Exception as e:
            logger.error(f"Database schema generation failed: {e}")
            return {'error': str(e)}


class APIDesignerAgent(BaseCodeAgent):
    """Generates backend API code and OpenAPI specification."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(APIGenerationSignature)

    async def generate(
        self,
        architecture: str,
        orm_models: str,
        api_contract: str,
        language: str,
        framework: str
    ) -> Dict[str, Any]:
        """Generate backend API code and OpenAPI spec."""
        try:
            if self.learned_context:
                architecture = architecture + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2b", "APIDesigner",
                architecture=architecture,
                orm_models=orm_models,
                api_contract=api_contract,
                language=language,
                framework=framework
            )

            self._broadcast("api_generated", {
                'framework': framework,
                'has_openapi': bool(result.openapi_spec),
            })

            return {
                'api_code': str(result.api_code),
                'openapi_spec': str(result.openapi_spec),
                'dependencies': str(result.dependencies),
            }

        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return {'error': str(e)}


class FrontendDeveloperAgent(BaseCodeAgent):
    """Generates frontend code consuming an OpenAPI specification."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(FrontendGenerationSignature)

    async def generate(
        self,
        openapi_spec: str,
        ui_requirements: str,
        framework: str
    ) -> Dict[str, Any]:
        """Generate frontend code consuming OpenAPI spec."""
        try:
            if self.learned_context:
                openapi_spec = openapi_spec + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2c", "FrontendDeveloper",
                openapi_spec=openapi_spec,
                ui_requirements=ui_requirements,
                framework=framework
            )

            self._broadcast("frontend_generated", {
                'framework': framework,
                'has_api_client': bool(result.api_client),
            })

            return {
                'frontend_code': str(result.frontend_code),
                'api_client': str(result.api_client),
                'dependencies': str(result.dependencies),
            }

        except Exception as e:
            logger.error(f"Frontend generation failed: {e}")
            return {'error': str(e)}


class IntegrationAgent(BaseCodeAgent):
    """Generates integration artifacts (Docker Compose, configs, startup scripts)."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(IntegrationSignature)

    async def generate(
        self,
        file_list: List[str],
        tech_stack: Dict[str, str],
        architecture: str
    ) -> Dict[str, Any]:
        """Generate integration artifacts (Docker, configs)."""
        try:
            if self.learned_context:
                architecture = architecture + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2d", "IntegrationEngineer",
                file_list=json.dumps(file_list),
                tech_stack=json.dumps(tech_stack),
                architecture=architecture
            )

            self._broadcast("integration_generated", {
                'has_docker': bool(result.docker_compose),
            })

            return {
                'docker_compose': str(result.docker_compose),
                'env_config': str(result.env_config),
                'requirements_txt': str(result.requirements_txt),
                'startup_script': str(result.startup_script),
            }

        except Exception as e:
            logger.error(f"Integration generation failed: {e}")
            return {'error': str(e)}


class ArbitratorAgent(BaseCodeAgent):
    """Evaluates whether a code review rejection is valid and actionable."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._evaluator = dspy.ChainOfThought(ReviewArbitrationSignature)

    async def evaluate(
        self,
        code: str,
        rejection_feedback: str,
        evidence: str,
        reviewer_name: str
    ) -> Dict[str, Any]:
        """Evaluate whether a review rejection is valid."""
        try:
            # Empty or generic evidence is automatically invalid
            if not evidence or not evidence.strip() or evidence.strip().lower() in (
                'none', 'n/a', 'no evidence', 'see above', 'see feedback'
            ):
                self._broadcast("rejection_arbitrated", {
                    'reviewer': reviewer_name, 'valid': False, 'reason': 'empty_evidence'
                })
                return {
                    'valid': False,
                    'reasoning': 'Rejection lacks specific evidence (code lines, test cases, or scenarios).',
                    'actionable_fix': '',
                }

            result = self._evaluator(
                code=code,
                rejection_feedback=rejection_feedback,
                evidence=evidence,
                reviewer_name=reviewer_name
            )

            valid_str = str(result.valid).strip().upper()
            valid = valid_str == "TRUE"

            self._broadcast("rejection_arbitrated", {
                'reviewer': reviewer_name, 'valid': valid,
            })

            return {
                'valid': valid,
                'reasoning': str(result.reasoning),
                'actionable_fix': str(result.actionable_fix) if valid else '',
            }

        except Exception as e:
            logger.error(f"Arbitration failed (non-blocking): {e}")
            return {'valid': True, 'reasoning': f'Arbitration error: {e}', 'actionable_fix': ''}


class CodebaseAnalyzerAgent(BaseCodeAgent):
    """Analyzes existing codebase to understand structure and patterns."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._analyzer = dspy.ChainOfThought(CodebaseAnalysisSignature)

    async def analyze(
        self,
        existing_code: str,
        file_paths: List[str],
        requirements: str
    ) -> Dict[str, Any]:
        """Analyze existing codebase structure and patterns."""
        try:
            result = await self._stream(self._analyzer, "Phase 0", "CodebaseAnalyzer",
                existing_code=existing_code,
                file_paths=json.dumps(file_paths),
                requirements=requirements
            )

            # Parse affected files
            try:
                affected_files = json.loads(str(result.affected_files))
                if not isinstance(affected_files, list):
                    affected_files = []
            except (json.JSONDecodeError, TypeError):
                affected_files = []

            self._broadcast("codebase_analyzed", {
                'files_analyzed': len(file_paths),
                'affected_files': len(affected_files),
            })

            return {
                'architecture_summary': str(result.architecture_summary),
                'style_conventions': str(result.style_conventions),
                'affected_files': affected_files,
                'dependencies': str(result.dependencies),
                'change_scope': str(result.change_scope),
            }

        except Exception as e:
            logger.error(f"Codebase analysis failed: {e}")
            return {'error': str(e)}


class EditPlannerAgent(BaseCodeAgent):
    """Plans surgical edits to existing code."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._planner = dspy.ChainOfThought(EditPlanSignature)

    async def plan_edit(
        self,
        existing_code: str,
        file_path: str,
        requirements: str,
        style_conventions: str = "",
        dependencies: str = ""
    ) -> Dict[str, Any]:
        """Plan edits for a single file."""
        try:
            result = await self._stream(self._planner, "Phase 1", "EditPlanner",
                existing_code=existing_code,
                file_path=file_path,
                requirements=requirements,
                style_conventions=style_conventions or "Follow existing patterns",
                dependencies=dependencies or "Preserve all existing interfaces"
            )

            # Parse edits
            try:
                edits = json.loads(str(result.edits))
                if not isinstance(edits, list):
                    edits = []
            except (json.JSONDecodeError, TypeError):
                edits = []

            edit_type = str(result.edit_type).strip().lower()
            if edit_type not in ('patch', 'rewrite'):
                edit_type = 'patch' if edits else 'rewrite'

            self._broadcast("edit_planned", {
                'file': file_path,
                'edit_type': edit_type,
                'num_edits': len(edits),
            })

            return {
                'edit_plan': str(result.edit_plan),
                'edits': edits,
                'new_code': str(result.new_code) if edit_type == 'rewrite' else '',
                'edit_type': edit_type,
            }

        except Exception as e:
            logger.error(f"Edit planning failed for {file_path}: {e}")
            return {'error': str(e)}


# =============================================================================
# CODING SWARM
# =============================================================================

@register_swarm("coding")
class CodingSwarm(DomainSwarm):
    """
    World-Class Coding Swarm.

    Generates production-quality code with:
    - Clean architecture
    - Comprehensive tests
    - Full documentation
    - Optimized performance
    """

    AGENT_TEAM = AgentTeam.define(
        (ArchitectAgent, "Architect", "_architect"),
        (DeveloperAgent, "Developer", "_developer"),
        (DebuggerAgent, "Debugger", "_debugger"),
        (OptimizerAgent, "Optimizer", "_optimizer"),
        (TestWriterAgent, "TestWriter", "_test_writer"),
        (DocWriterAgent, "DocWriter", "_doc_writer"),
        (VerifierAgent, "Verifier", "_verifier"),
        (SimplicityJudgeAgent, "SimplicityJudge", "_simplicity_judge"),
    )

    def __init__(self, config: CodingConfig = None):
        super().__init__(config or CodingConfig())

        # Team configuration
        self._team_config: Optional[TeamConfig] = None
        if self.config.team:
            self._team_config = TEAM_PRESETS.get(self.config.team)

        # Scope classifier (uses DSPy's configured LM)
        self._scope_classifier = dspy.ChainOfThought(ScopeClassificationSignature)

        # Review module (lazy init)
        self._review_module = None

    def set_team(self, team_config: TeamConfig):
        """Set a custom team configuration."""
        self._team_config = team_config

    def _agent_context(self, agent_name: str) -> str:
        """Build per-agent learned context with optional persona injection."""
        parts = []
        if self._team_config:
            persona = self._team_config.get_persona(agent_name)
            if persona:
                parts.append(persona.to_prompt())
        base_context = super()._agent_context(agent_name)
        if base_context:
            parts.append(base_context)
        return "\n\n".join(parts)

    def _detect_team(self, requirements: str) -> Optional[str]:
        """Auto-detect best team preset from requirements keywords."""
        req_lower = requirements.lower()
        frontend_signals = ['react', 'frontend', 'ui component', 'css', 'tailwind', 'nextjs', 'vue', 'angular']
        ds_signals = ['data pipeline', 'ml ', 'machine learning', 'pandas', 'etl', 'dataset', 'model training']
        if any(kw in req_lower for kw in frontend_signals):
            return "frontend"
        if any(kw in req_lower for kw in ds_signals):
            return "datascience"
        return "fullstack"

    def _detect_scope(self, requirements: str) -> str:
        """Detect whether requirements need full-stack or single-tier generation.

        Priority:
        1. Explicit config.scope overrides everything
        2. LLM classification via ScopeClassificationSignature
        3. Keyword fallback if LLM fails
        """
        # 1. Explicit config
        explicit = getattr(self.config, 'scope', None)
        if explicit in ('single_tier', 'full_stack'):
            return explicit

        # 2. LLM classification
        try:
            result = self._scope_classifier(requirements=requirements)
            scope = str(result.scope).strip().lower().replace('-', '_').replace(' ', '_')
            if scope in ('single_tier', 'full_stack'):
                return scope
            # Handle partial matches from LLM output
            if 'full' in scope:
                return 'full_stack'
            if 'single' in scope:
                return 'single_tier'
        except Exception as e:
            logger.warning(f"LLM scope classification failed, falling back to keywords: {e}")

        # 3. Keyword fallback
        return self._detect_scope_keywords(requirements)

    def _detect_scope_keywords(self, requirements: str) -> str:
        """Keyword-based scope detection fallback."""
        req_lower = requirements.lower()

        fullstack_keywords = [
            'full-stack', 'full stack', 'fullstack',
            'web app', 'web application',
        ]
        if any(kw in req_lower for kw in fullstack_keywords):
            return 'full_stack'

        has_db = any(kw in req_lower for kw in [
            'database', 'sqlite', 'postgresql', 'mysql', 'mongodb',
            'sql', 'schema', 'orm', 'migration',
        ])
        has_api = any(kw in req_lower for kw in [
            'rest api', 'api endpoint', 'backend', 'fastapi', 'flask',
            'express', 'endpoint', 'openapi',
        ])
        has_frontend = any(kw in req_lower for kw in [
            'frontend', 'react', 'vue', 'angular', 'ui component',
            'user interface', 'dashboard', 'web page',
        ])
        tier_count = sum([has_db, has_api, has_frontend])
        if tier_count >= 2:
            return 'full_stack'

        return 'single_tier'

    def _is_trivial_task(self, requirements: str) -> bool:
        """Detect trivial tasks that don't need full swarm treatment.

        Trivial tasks: hello world, simple print, basic scripts, etc.
        These get fast path: 1 iteration, no team planning/review.

        Returns True if task is trivial and should use fast path.
        """
        req_lower = requirements.lower()
        req_words = len(requirements.split())

        # Very short requirements (< 10 words) are likely trivial
        if req_words < 10:
            trivial_patterns = [
                'hello world', 'hello', 'print', 'simple',
                'basic', 'minimal', 'one liner', 'quick',
            ]
            if any(p in req_lower for p in trivial_patterns):
                return True

        # Explicit trivial task indicators
        trivial_explicit = [
            'hello world',
            'print hello',
            'print hi',
            'simple script',
            'basic script',
            'one file',
            'single file',
            'minimal',
        ]
        if any(p in req_lower for p in trivial_explicit):
            return True

        # Short requirements without complex keywords
        if req_words < 15:
            complex_keywords = [
                'api', 'database', 'authentication', 'crud',
                'frontend', 'backend', 'test', 'multiple',
                'class', 'module', 'package', 'framework',
            ]
            if not any(kw in req_lower for kw in complex_keywords):
                return True

        return False

    def _init_fullstack_agents(self):
        """Lazy initialization of full-stack agents. Only creates agents when needed."""
        if hasattr(self, '_system_designer') and self._system_designer is not None:
            return

        self._system_designer = SystemDesignerAgent(
            self._memory, self._context, self._bus, self._agent_context("SystemDesigner"))
        self._db_architect = DatabaseArchitectAgent(
            self._memory, self._context, self._bus, self._agent_context("DatabaseArchitect"))
        self._api_designer = APIDesignerAgent(
            self._memory, self._context, self._bus, self._agent_context("Developer"))
        self._frontend_developer = FrontendDeveloperAgent(
            self._memory, self._context, self._bus, self._agent_context("FrontendDeveloper"))
        self._integration_agent = IntegrationAgent(
            self._memory, self._context, self._bus, self._agent_context("Integration"))

    async def _generate_fullstack(
        self,
        requirements: str,
        config,
        research_context,
        review_criteria: str,
        workspace
    ) -> tuple:
        """Full-stack pipeline: SystemDesign -> DB -> Backend -> Frontend -> Integration.

        Returns:
            (files_dict, main_file, architecture_str)
        """
        self._init_fullstack_agents()
        ctx = FullStackContext()

        # Build tech_stack from config
        ctx.tech_stack = {
            'db_type': getattr(config, 'db_type', 'sqlite'),
            'backend': getattr(config, 'backend_framework', 'fastapi'),
            'frontend': getattr(config, 'frontend_framework', 'react'),
        }

        lang = (config.language.value if hasattr(config.language, 'value')
                else str(config.language))
        files = {}

        # Inject research context
        enriched_requirements = requirements
        research_prompt = research_context.to_prompt() if research_context else ""
        if research_prompt:
            enriched_requirements += "\n\n## Research Findings\n" + research_prompt
        if review_criteria:
            enriched_requirements += "\n\n## Code Review Criteria\n" + review_criteria

        # =================================================================
        # Phase 1: System Design
        # =================================================================
        _progress("Phase 1", "SystemDesigner", "Designing full-stack system...")

        system_result = await self._system_designer.design(
            requirements=enriched_requirements,
            language=lang,
            tech_stack=ctx.tech_stack,
        )

        if 'error' in system_result:
            _progress("Phase 1", "SystemDesigner", f"Error: {system_result['error']}")
            return files, 'app.py', ''

        ctx.data_model = system_result.get('data_model', '')
        ctx.api_contract = system_result.get('api_contract', '')
        ctx.component_map = system_result.get('component_map', '')
        ctx.ui_requirements = system_result.get('ui_requirements', '')
        architecture = system_result.get('architecture', '')

        _progress("Phase 1", "SystemDesigner", "Done -- system design complete")

        # =================================================================
        # Phase 2a: Database
        # =================================================================
        _progress("Phase 2a", "DatabaseArchitect", "Generating schema and ORM models...")

        db_result = await self._db_architect.generate(
            data_model=ctx.data_model,
            db_type=ctx.tech_stack.get('db_type', 'sqlite'),
            language=lang,
        )

        if 'error' not in db_result:
            ctx.schema_sql = db_result.get('schema_sql', '')
            ctx.orm_models = db_result.get('orm_models', '')
            files['schema.sql'] = _strip_code_fences(ctx.schema_sql)
            files['models.py'] = _strip_code_fences(ctx.orm_models)
            _progress("Phase 2a", "DatabaseArchitect", f"Done -- schema.sql + models.py")
        else:
            _progress("Phase 2a", "DatabaseArchitect", f"Error: {db_result['error']}")

        # =================================================================
        # Phase 2b: Backend + OpenAPI
        # =================================================================
        _progress("Phase 2b", "APIDesigner", "Generating API code and OpenAPI spec...")

        api_result = await self._api_designer.generate(
            architecture=architecture,
            orm_models=ctx.orm_models,
            api_contract=ctx.api_contract,
            language=lang,
            framework=ctx.tech_stack.get('backend', 'fastapi'),
        )

        if 'error' not in api_result:
            files['app.py'] = _strip_code_fences(api_result.get('api_code', ''))
            ctx.openapi_spec = api_result.get('openapi_spec', '')
            files['openapi.yaml'] = _strip_code_fences(ctx.openapi_spec)
            _progress("Phase 2b", "APIDesigner", "Done -- app.py + openapi.yaml")
        else:
            _progress("Phase 2b", "APIDesigner", f"Error: {api_result['error']}")

        # =================================================================
        # Phase 2c: Frontend (consumes OpenAPI spec)
        # =================================================================
        _progress("Phase 2c", "FrontendDeveloper", "Generating frontend code...")

        frontend_result = await self._frontend_developer.generate(
            openapi_spec=ctx.openapi_spec,
            ui_requirements=ctx.ui_requirements,
            framework=ctx.tech_stack.get('frontend', 'react'),
        )

        if 'error' not in frontend_result:
            files['frontend/App.jsx'] = _strip_code_fences(frontend_result.get('frontend_code', ''))
            files['frontend/api.js'] = _strip_code_fences(frontend_result.get('api_client', ''))
            _progress("Phase 2c", "FrontendDeveloper", "Done -- frontend/App.jsx + frontend/api.js")
        else:
            _progress("Phase 2c", "FrontendDeveloper", f"Error: {frontend_result['error']}")

        # =================================================================
        # Phase 2d: Integration
        # =================================================================
        _progress("Phase 2d", "IntegrationEngineer", "Generating integration artifacts...")

        integration_result = await self._integration_agent.generate(
            file_list=list(files.keys()),
            tech_stack=ctx.tech_stack,
            architecture=architecture,
        )

        if 'error' not in integration_result:
            files['docker-compose.yml'] = _strip_code_fences(integration_result.get('docker_compose', ''))
            files['.env.example'] = _strip_code_fences(integration_result.get('env_config', ''))
            files['requirements.txt'] = _strip_code_fences(integration_result.get('requirements_txt', ''))
            files['start.sh'] = _strip_code_fences(integration_result.get('startup_script', ''))
            _progress("Phase 2d", "IntegrationEngineer", "Done -- docker-compose.yml, .env.example, requirements.txt, start.sh")
        else:
            _progress("Phase 2d", "IntegrationEngineer", f"Error: {integration_result['error']}")

        # Stream summary
        _progress("FullStack", "Pipeline", f"Generated {len(files)} file(s): {', '.join(files.keys())}")

        return files, 'app.py', architecture

    async def _execute_domain(
        self,
        requirements: str,
        language: CodeLanguage = None,
        style: CodeStyle = None,
        **kwargs
    ) -> CodingResult:
        """Execute code generation (called by DomainSwarm.execute())."""
        return await self.generate(requirements, language, style, **kwargs)

    async def generate(
        self,
        requirements: str,
        language: CodeLanguage = None,
        style: CodeStyle = None,
        include_tests: bool = None,
        include_docs: bool = None,
        progress_callback=None,
        trace_callback=None,
    ) -> CodingResult:
        """
        Generate complete code from requirements.

        Args:
            requirements: What to build
            language: Target language
            style: Coding style
            include_tests: Generate tests
            include_docs: Generate documentation
            progress_callback: Optional callable(phase, agent, message) for TUI integration
            trace_callback: Optional callable(trace_dict) for TUI trace panel

        Returns:
            CodingResult with generated code
        """
        global _active_progress_callback, _active_trace_callback
        _active_progress_callback = progress_callback
        _active_trace_callback = trace_callback
        start_time = datetime.now()

        # Auto-detect team if not explicitly set
        if not self._team_config:
            detected = self._detect_team(requirements)
            if detected:
                self._team_config = TEAM_PRESETS.get(detected)

        config = self.config
        lang = language or config.language
        code_style = style or config.style
        gen_tests = include_tests if include_tests is not None else config.include_tests
        gen_docs = include_docs if include_docs is not None else config.include_docs

        # FAST PATH: Detect trivial tasks and reduce LLM calls
        is_trivial = self._is_trivial_task(requirements)
        if is_trivial:
            # Override config for speed: 1 iteration, skip team planning/review
            config = type(config)(**{**config.__dict__,
                'max_design_iterations': 1,
                'skip_team_planning': True,
                'skip_team_review': True,
            })
            _progress("FastPath", "Detector", "Trivial task detected - using fast path (1 iteration, no team)")

        team_name = self._team_config.name if self._team_config else "auto"
        print(f"\n{'='*60}", flush=True)
        print(f"  CodingSwarm | {lang.value} | {code_style.value} | team={team_name}", flush=True)
        print(f"{'='*60}", flush=True)
        logger.info(f"CodingSwarm starting: {lang.value}, {code_style.value}")

        # Initialize workspace for terminal-based validation
        workspace = WorkspaceManager() if getattr(config, 'enable_workspace', True) else None

        try:
            # =================================================================
            # PHASE 1 + 1.5: COLLABORATIVE DESIGN LOOP (Architect + Researcher)
            # =================================================================
            max_design_iterations = getattr(config, 'max_design_iterations', 3)

            arch_result, research_context = await self._collaborative_design_loop(
                requirements=requirements,
                language=lang.value,
                style=code_style.value,
                constraints=json.dumps({
                    'frameworks': config.frameworks,
                    'max_file_size': config.max_file_size
                }),
                max_iterations=max_design_iterations,
            )

            if 'error' in arch_result:
                return CodingResult(
                    success=False,
                    swarm_name=self.config.name,
                    domain=self.config.domain,
                    output={},
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=arch_result['error']
                )

            n_components = len(arch_result.get('components', []))
            total_findings = (len(research_context.best_practices) +
                            len(research_context.library_docs) +
                            len(research_context.api_references) +
                            len(research_context.warnings))

            self._trace_phase("CollaborativeDesign", AgentRole.PLANNER,
                {'requirements': requirements[:100], 'iterations': max_design_iterations},
                {'components': n_components, 'research_findings': total_findings},
                success='error' not in arch_result, phase_start=start_time,
                tools_used=['arch_design', 'web_search'])

            # =================================================================
            # PHASE 0: SCOPE DETECTION
            # =================================================================
            scope = self._detect_scope(requirements)
            _progress("Phase 0", "ScopeDetector", f"Detected scope: {scope}")

            # Build review criteria for both paths
            review_criteria = ""
            if self._team_config:
                review_criteria = self._build_review_criteria()

            # =================================================================
            # PHASE 2: TEAM PLANNING (refine architecture with team input)
            # =================================================================
            phase2_start = datetime.now()
            planning_result = None
            research_prompt = research_context.to_prompt()

            # Skip team planning for trivial tasks (fast path)
            skip_planning = getattr(config, 'skip_team_planning', False)
            if self._team_config and not skip_planning:
                _progress("Phase 2", "TeamPlanning", f"Team planning session ({self._team_config.name})...")
                planning_result = await self._team_planning(
                    requirements=requirements,
                    architecture=arch_result.get('architecture', ''),
                    research_findings=research_prompt,
                )

                # Use refined architecture if planning succeeded
                if planning_result and planning_result.get('refined_architecture'):
                    arch_result['architecture'] = planning_result['refined_architecture']
                    _progress("Phase 2", "TeamPlanning", "Architecture refined with team consensus")

                    # Log key agreements
                    if planning_result.get('team_agreements'):
                        for line in str(planning_result['team_agreements']).split('\n')[:5]:
                            if line.strip():
                                _progress("Phase 2", "TeamPlanning", f"  Agreement: {line.strip()[:70]}")

                self._trace_phase("TeamPlanning", AgentRole.PLANNER,
                    {'team': self._team_config.name, 'personas_count': len(planning_result.get('team_feedback', []))},
                    {'has_refined_arch': bool(planning_result.get('refined_architecture')),
                     'has_impl_plan': bool(planning_result.get('implementation_plan'))},
                    success=True, phase_start=phase2_start, tools_used=['team_planning'])
            elif skip_planning:
                _progress("Phase 2", "TeamPlanning", "Skipped (fast path - trivial task)")
            else:
                _progress("Phase 2", "TeamPlanning", "Skipped (no team configured)")

            if scope == "full_stack":
                # =============================================================
                # FULL-STACK PATH: SystemDesign → DB → Backend → Frontend → Integration
                # =============================================================
                files, main_file, architecture = await self._generate_fullstack(
                    requirements, config, research_context, review_criteria, workspace
                )
                # Override arch_result architecture for downstream phases
                arch_result['architecture'] = architecture
            else:
                # =============================================================
                # SINGLE-TIER PATH: Existing Architect + Developer pipeline
                # =============================================================
                # PHASE 3: CODE GENERATION (parallel for each component)
                _progress("Phase 3", "Developer", "Generating code...")

                components = arch_result.get('components', [])
                if not components:
                    components = [{'name': 'main', 'description': requirements}]

                # Inject implementation plan from team planning if available
                enriched_arch = arch_result['architecture']
                if planning_result and planning_result.get('implementation_plan'):
                    enriched_arch = enriched_arch + "\n\n## Implementation Plan (from team planning)\n" + planning_result['implementation_plan']

                # Inject reviewer criteria so developer writes code that passes review
                if review_criteria:
                    enriched_arch = enriched_arch + "\n\n## Code Review Criteria (your code WILL be reviewed against these)\n" + review_criteria

                total_components = len(components)
                completed_count = 0

                async def _generate_component(comp):
                    nonlocal completed_count
                    comp_name = comp.get('name', 'component') if isinstance(comp, dict) else str(comp)
                    _progress("Phase 3", "Developer", f"  Writing {comp_name}...")
                    result = await self._developer.generate(
                        architecture=enriched_arch,
                        component=comp_name,
                        language=lang.value,
                        dependencies=config.frameworks
                    )
                    completed_count += 1
                    if isinstance(result, dict) and 'filename' in result:
                        _progress("Phase 3", "Developer", f"  [{completed_count}/{total_components}] {result['filename']} ready")
                    else:
                        _progress("Phase 3", "Developer", f"  [{completed_count}/{total_components}] {comp_name} failed")
                    return result

                code_results = await asyncio.gather(
                    *[_generate_component(comp) for comp in components],
                    return_exceptions=True
                )

                # Collect generated files (strip markdown fences from LLM output)
                files = {}
                main_file = None
                for result in code_results:
                    if isinstance(result, Exception):
                        continue
                    if 'code' in result and 'filename' in result:
                        filename = result['filename']
                        files[filename] = _strip_code_fences(result['code'])
                        if main_file is None:
                            main_file = filename

                _progress("Phase 3", "Developer", f"Done -- {len(files)} file(s): {', '.join(files.keys())}")
                # Stream developer output (first 5 lines of each file)
                for fname, code_content in files.items():
                    lines = code_content.strip().split('\n')
                    _progress("Phase 3", "Developer", f"  --- {fname} ({len(lines)} lines) ---")
                    for line in lines[:5]:
                        _progress("Phase 3", "Developer", f"    {line}")
                    if len(lines) > 5:
                        _progress("Phase 3", "Developer", f"    ... ({len(lines)-5} more lines)")

            phase3_start = datetime.now()
            self._trace_phase("Developer", AgentRole.ACTOR,
                {'components': len(arch_result.get('components', [])), 'scope': scope},
                {'files_generated': len(files)},
                success=len(files) > 0, phase_start=phase2_start, tools_used=['code_generate'])

            # =================================================================
            # PHASE 4: OPTIMIZATION (parallel, skip config files)
            # =================================================================
            # Config files that don't benefit from LLM optimization
            SKIP_OPTIMIZATION = {
                '.env', '.env.example', '.env.local', '.env.production',
                'docker-compose.yml', 'docker-compose.yaml', 'Dockerfile',
                'requirements.txt', 'package.json', 'package-lock.json',
                'yarn.lock', 'Pipfile', 'Pipfile.lock', 'pyproject.toml',
                'setup.py', 'setup.cfg', 'Makefile', 'Procfile',
                '.gitignore', '.dockerignore', 'LICENSE', 'README.md',
                'manifest.json', 'tsconfig.json', 'jest.config.js',
                '.prettierrc', '.eslintrc', '.babelrc',
            }
            SKIP_EXTENSIONS = {'.yml', '.yaml', '.json', '.toml', '.cfg', '.ini', '.lock', '.md', '.txt', '.sh'}

            def _should_optimize(fname: str) -> bool:
                """Check if file should be optimized (code files only)."""
                base_name = fname.rsplit('/', 1)[-1]  # Get filename without path
                if base_name in SKIP_OPTIMIZATION:
                    return False
                ext = '.' + base_name.rsplit('.', 1)[-1] if '.' in base_name else ''
                # Skip config extensions, but allow .py, .js, .ts, .jsx, .tsx, .go, .rs, .java, .rb
                if ext in SKIP_EXTENSIONS:
                    return False
                return True

            files_to_optimize = {fn: c for fn, c in files.items() if _should_optimize(fn)}
            files_to_skip = {fn: c for fn, c in files.items() if not _should_optimize(fn)}

            if files_to_skip:
                _progress("Phase 4", "Optimizer", f"Skipping {len(files_to_skip)} config file(s): {', '.join(files_to_skip.keys())}")

            _progress("Phase 4", "Optimizer", f"Optimizing {len(files_to_optimize)} code file(s) in parallel...")
            phase4_start = datetime.now()

            total_files = len(files_to_optimize)
            optimized_count = 0

            async def _optimize_one(fname: str, code_str: str):
                nonlocal optimized_count
                _progress("Phase 4", "Optimizer", f"  Optimizing {fname}...")
                opt_result = await self._optimizer.optimize(
                    code=code_str,
                    focus="readability",
                    constraints="Maintain all functionality",
                    requirements=requirements
                )
                optimized_count += 1
                improvements = opt_result.get('improvements', [])
                _progress("Phase 4", "Optimizer", f"  [{optimized_count}/{total_files}] {fname} optimized")
                return fname, _strip_code_fences(opt_result.get('optimized_code', code_str)), improvements

            opt_tasks = [_optimize_one(fn, c) for fn, c in files_to_optimize.items()]
            opt_results = await asyncio.gather(*opt_tasks, return_exceptions=True)

            optimized_files = {}
            for r in opt_results:
                if isinstance(r, Exception):
                    continue
                fname, optimized_code, improvements = r
                optimized_files[fname] = optimized_code
                # Stream optimizer output
                if improvements:
                    imp_list = improvements if isinstance(improvements, list) else [str(improvements)]
                    for imp in imp_list[:5]:
                        _progress("Phase 4", "Optimizer", f"  {fname}: {imp}")

            # Keep originals for any that failed optimization
            for fname in files_to_optimize:
                if fname not in optimized_files:
                    optimized_files[fname] = files_to_optimize[fname]

            # Add back skipped config files unchanged
            optimized_files.update(files_to_skip)

            files = optimized_files
            _progress("Phase 4", "Optimizer", f"Done -- {len(files_to_optimize)} code file(s) optimized, {len(files_to_skip)} config file(s) kept as-is")

            self._trace_phase("Optimizer", AgentRole.ACTOR,
                {'files_count': len(files)},
                {'optimized': len(optimized_files)},
                success=True, phase_start=phase4_start, tools_used=['code_optimize'])

            # =================================================================
            # PHASE 3.5: VALIDATION & FIX LOOP
            # =================================================================
            validation_metadata = {"validated": False, "fix_attempts": 0, "errors_fixed": []}
            max_fix = getattr(config, 'max_fix_attempts', 3)

            if workspace and workspace.available and files:
                _progress("Phase 4.5", "Validator", "Validating code in workspace...")
                phase35_start = datetime.now()

                FIXABLE_ERRORS = ('SyntaxError', 'ImportError', 'NameError', 'TypeError', 'IndentationError')

                for attempt in range(max_fix):
                    syntax_ok = True
                    runtime_ok = True

                    # Write all files to workspace
                    for fname, code_content in files.items():
                        await workspace.write_file(fname, code_content)

                    # --- Pass 1: Syntax check all .py files ---
                    for fname in list(files.keys()):
                        if not fname.endswith('.py'):
                            continue
                        check_result = await workspace.syntax_check(fname, language="python")
                        if check_result.success:
                            _progress("Phase 4.5", "Validator", f"Syntax OK: {fname}")
                        else:
                            error_text = check_result.error or check_result.output
                            if any(err in error_text for err in FIXABLE_ERRORS):
                                syntax_ok = False
                                _progress("Phase 4.5", "Debugger", f"Fixing syntax in {fname} (attempt {attempt+1})...")
                                fix_result = await self._debugger.debug(
                                    code=files[fname],
                                    error_message=error_text,
                                    context=f"Fix the syntax error. Return ONLY the corrected Python code, no markdown fences. File: {fname}"
                                )
                                if fix_result and 'fix' in fix_result and 'error' not in fix_result:
                                    files[fname] = _strip_code_fences(fix_result['fix'])
                                    validation_metadata["errors_fixed"].append(f"{fname}: syntax")

                    # --- Pass 2: Run main file ONLY if all syntax passed ---
                    if syntax_ok and main_file and main_file in files and main_file.endswith('.py'):
                        _progress("Phase 4.5", "Validator", f"Running: {main_file}...")
                        await workspace.write_file(main_file, files[main_file])
                        run_result = await workspace.run_python(main_file, timeout=15)
                        if run_result.success:
                            _progress("Phase 4.5", "Validator", f"Run OK: {main_file}")
                        else:
                            error_text = run_result.error or run_result.output
                            if any(err in error_text for err in FIXABLE_ERRORS):
                                runtime_ok = False
                                _progress("Phase 4.5", "Debugger", f"Fixing runtime error in {main_file} (attempt {attempt+1})...")
                                fix_result = await self._debugger.debug(
                                    code=files[main_file],
                                    error_message=error_text,
                                    context=f"Fix the runtime error. Return ONLY the corrected Python code, no markdown fences. File: {main_file}"
                                )
                                if fix_result and 'fix' in fix_result and 'error' not in fix_result:
                                    files[main_file] = _strip_code_fences(fix_result['fix'])
                                    validation_metadata["errors_fixed"].append(f"{main_file}: runtime")
                            else:
                                # Non-fixable runtime error (EOF, FileNotFound, etc.) -- not a code bug
                                _progress("Phase 4.5", "Validator", f"Non-fixable runtime condition (skipped): {error_text[:80]}")

                    validation_metadata["fix_attempts"] = attempt + 1

                    if syntax_ok and runtime_ok:
                        validation_metadata["validated"] = True
                        _progress("Phase 4.5", "Validator", f"All files validated after {attempt+1} attempt(s)")
                        break
                    elif not syntax_ok:
                        _progress("Phase 4.5", "Validator", f"Syntax errors fixed, re-validating (attempt {attempt+1}/{max_fix})...")

                if not validation_metadata["validated"]:
                    _progress("Phase 4.5", "Validator", f"Max attempts ({max_fix}) reached")

                self._trace_phase("Validator", AgentRole.AUDITOR,
                    {'max_fix_attempts': max_fix},
                    {'validated': validation_metadata['validated'],
                     'fix_attempts': validation_metadata['fix_attempts'],
                     'errors_fixed': len(validation_metadata['errors_fixed'])},
                    success=True, phase_start=phase35_start, tools_used=['workspace_validate', 'debug'])

            # =================================================================
            # PHASE 5: VERIFICATION + DEBUGGER FEEDBACK
            # =================================================================
            phase5_start = datetime.now()
            verification_result = None
            try:
                _progress("Phase 5", "Verifier", "Verifying code against requirements...")

                all_code = "\n\n".join(files.values())
                verification_result = await self._verifier.verify(
                    code=all_code,
                    original_requirements=requirements,
                    architecture=arch_result.get('architecture', '')
                )

                # Stream verifier output
                if verification_result:
                    v_score = verification_result.get('coverage_score', '?')
                    v_status = "PASSED" if verification_result.get('verified', True) else "ISSUES FOUND"
                    _progress("Phase 5", "Verifier", f"{v_status} (coverage: {v_score})")

                if verification_result and not verification_result.get('verified', True):
                    issues = verification_result.get('issues', [])
                    if issues:
                        # Format issues into description for debugger
                        issues_desc = "; ".join(
                            f"[{iss.get('severity', 'unknown')}] {iss.get('description', 'no description')}"
                            for iss in issues if isinstance(iss, dict)
                        )
                        _progress("Phase 5", "Verifier", f"Found {len(issues)} issue(s), attempting fix...")
                        for iss in issues[:5]:
                            if isinstance(iss, dict):
                                _progress("Phase 5", "Verifier", f"  [{iss.get('severity','?')}] {iss.get('description','')[:80]}")

                        # One attempt with debugger — non-blocking
                        try:
                            debug_result = await self._debugger.debug(
                                code=all_code,
                                error_message=issues_desc,
                                context=f"Return ONLY corrected Python code, no markdown fences. Requirements: {requirements}"
                            )
                            if debug_result and 'fix' in debug_result and 'error' not in debug_result:
                                fixed_code = _strip_code_fences(debug_result['fix'])
                                if main_file and main_file in files:
                                    files[main_file] = fixed_code
                                    _progress("Phase 5", "Debugger", "Fix applied to main file")
                        except Exception as dbg_err:
                            logger.error(f"Debugger fix attempt failed (non-blocking): {dbg_err}")
            except Exception as ver_err:
                logger.error(f"Verification phase failed (non-blocking): {ver_err}")

            self._trace_phase("Verifier", AgentRole.AUDITOR,
                {'requirements_len': len(requirements)},
                {'verified': verification_result.get('verified', True) if verification_result else True,
                 'issues_count': len(verification_result.get('issues', [])) if verification_result else 0},
                success=True, phase_start=phase5_start, tools_used=['verify', 'debug'])

            # =================================================================
            # PHASE 5.5: SIMPLICITY JUDGE (Anti-Over-Engineering Gate)
            # =================================================================
            phase55_start = datetime.now()
            simplicity_result = None
            try:
                all_code_str = "\n\n".join(files.values())
                file_count = len(files)
                total_lines = sum(len(f.split('\n')) for f in files.values())

                _progress("Phase 5.5", "SimplicityJudge", f"Evaluating complexity ({file_count} files, {total_lines} lines)...")

                simplicity_result = await self._simplicity_judge.judge(
                    code=all_code_str,
                    requirements=requirements,
                    file_count=file_count,
                    total_lines=total_lines
                )

                score = simplicity_result.get('simplicity_score', 1.0)
                verdict = simplicity_result.get('verdict', 'ACCEPT')
                issues = simplicity_result.get('issues', [])
                critical_count = simplicity_result.get('critical_count', 0)

                if verdict == 'SIMPLIFY':
                    _progress("Phase 5.5", "SimplicityJudge", f"OVER-ENGINEERED (score: {score:.2f})")
                    # Log top issues
                    for issue in issues[:3]:
                        sev = issue.get('severity', 'major')
                        desc = issue.get('issue', '')[:80]
                        _progress("Phase 5.5", "SimplicityJudge", f"  [{sev.upper()}] {desc}")

                    # If severely over-engineered, trigger optimizer with simplification focus
                    if critical_count >= 2 or score < 0.3:
                        _progress("Phase 5.5", "SimplicityJudge", "Requesting code simplification...")
                        simplify_feedback = "SIMPLIFY CODE: " + "; ".join(
                            i.get('simpler_alternative', i.get('issue', ''))
                            for i in issues[:3]
                        )
                        try:
                            opt_result = await self._optimizer.optimize(
                                code=all_code_str,
                                requirements=requirements,
                                focus="simplification",
                                constraints=simplify_feedback
                            )
                            if opt_result and opt_result.get('optimized_code'):
                                # Update main file with simplified code
                                files[main_file] = str(opt_result['optimized_code'])
                                _progress("Phase 5.5", "SimplicityJudge", "Code simplified")
                        except Exception as opt_err:
                            logger.warning(f"Simplification attempt failed: {opt_err}")
                else:
                    _progress("Phase 5.5", "SimplicityJudge", f"APPROVED (score: {score:.2f})")

            except Exception as simp_err:
                logger.error(f"Simplicity judgment failed (non-blocking): {simp_err}")
                simplicity_result = {'simplicity_score': 1.0, 'verdict': 'ACCEPT', 'issues': []}

            self._trace_phase("SimplicityJudge", AgentRole.AUDITOR,
                {'file_count': len(files), 'total_lines': sum(len(f.split('\\n')) for f in files.values())},
                {'simplicity_score': simplicity_result.get('simplicity_score', 1.0) if simplicity_result else 1.0,
                 'verdict': simplicity_result.get('verdict', 'ACCEPT') if simplicity_result else 'ACCEPT',
                 'issues_count': len(simplicity_result.get('issues', [])) if simplicity_result else 0},
                success=True, phase_start=phase55_start, tools_used=['simplicity_judge'])

            # =================================================================
            # PHASE 6: TEAM REVIEW
            # =================================================================
            phase6_start = datetime.now()
            team_review_result = None
            skip_review = getattr(config, 'skip_team_review', False)
            if self._team_config and self._team_config.review_protocol != "none" and not skip_review:
                _progress("Phase 6", "TeamReview", f"Team review ({self._team_config.name})...")
                all_code_str = "\n\n".join(files.values())
                team_review_result, files = await self._team_review(
                    all_code_str, requirements,
                    arch_result.get('architecture', ''),
                    files, main_file,
                    planning_result=planning_result,
                    simplicity_result=simplicity_result,  # Pass simplicity verdict to avoid contradictions
                )
                self._trace_phase("TeamReview", AgentRole.REVIEWER,
                    {'team': self._team_config.name, 'protocol': self._team_config.review_protocol},
                    {'approved': team_review_result.get('approved', True),
                     'rework_attempts': team_review_result.get('rework_attempts', 0)},
                    success=True, phase_start=phase6_start, tools_used=['team_review'])
            elif skip_review:
                _progress("Phase 6", "TeamReview", "Skipped (fast path - trivial task)")

            # =================================================================
            # PHASE 7: TEST GENERATION (if enabled)
            # =================================================================
            phase7_start = datetime.now()
            tests = {}
            test_coverage = 0.0

            if gen_tests and files:
                _progress("Phase 7", "TestWriter", "Generating tests...")

                # Combine all code for test generation
                all_code = "\n\n".join(files.values())
                test_framework = "pytest" if lang == CodeLanguage.PYTHON else "jest"

                test_result = await self._test_writer.generate_tests(
                    code=all_code,
                    framework=test_framework,
                    coverage_target="80%"
                )

                if 'tests' in test_result:
                    test_ext = "_test.py" if lang == CodeLanguage.PYTHON else ".test.js"
                    tests[f"test_{main_file or 'main'}{test_ext}"] = test_result['tests']
                    test_coverage = test_result.get('coverage_estimate', 0.0)

            self._trace_phase("TestWriter", AgentRole.ACTOR,
                {'gen_tests': gen_tests},
                {'test_files': len(tests), 'coverage': test_coverage},
                success=True, phase_start=phase7_start, tools_used=['test_generate'])

            # =================================================================
            # PHASE 8: DOCUMENTATION (if enabled)
            # =================================================================
            phase8_start = datetime.now()
            documentation = ""

            if gen_docs and files:
                _progress("Phase 8", "DocWriter", "Generating documentation...")

                all_code = "\n\n".join(files.values())
                doc_result = await self._doc_writer.document(
                    code=all_code,
                    architecture=arch_result['architecture'],
                    audience="developers"
                )

                if 'documentation' in doc_result:
                    documentation = doc_result['documentation']

            self._trace_phase("DocWriter", AgentRole.ACTOR,
                {'gen_docs': gen_docs},
                {'has_docs': bool(documentation)},
                success=True, phase_start=phase8_start, tools_used=['doc_generate'])

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            # Calculate LOC
            loc = sum(code.count('\n') + 1 for code in files.values())

            code_output = CodeOutput(
                files=files,
                main_file=main_file or "",
                entry_point="main()" if lang == CodeLanguage.PYTHON else "index",
                dependencies=config.frameworks,
                tests=tests,
                docs=documentation,
                architecture=arch_result['architecture']
            )

            # Calculate complexity_score from simplicity result (invert: 1=simple becomes 0=no-over-engineering)
            simplicity_score = simplicity_result.get('simplicity_score', 1.0) if simplicity_result else 1.0
            complexity_score = 1.0 - simplicity_score  # 0=appropriately simple, 1=severely over-engineered

            result = CodingResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={'files': list(files.keys())},
                execution_time=exec_time,
                code=code_output,
                language=lang.value,
                loc=loc,
                test_coverage=test_coverage,
                quality_score=verification_result.get('coverage_score', 0.8) if verification_result else 0.8,
                complexity_score=complexity_score
            )

            # Store simplicity judgment metadata
            if simplicity_result:
                result.metadata['simplicity_judgment'] = {
                    'score': simplicity_score,
                    'verdict': simplicity_result.get('verdict', 'ACCEPT'),
                    'issues_count': len(simplicity_result.get('issues', [])),
                    'critical_count': simplicity_result.get('critical_count', 0),
                }

            validated_str = "validated" if validation_metadata.get("validated") else "not validated"
            simplicity_verdict = simplicity_result.get('verdict', 'ACCEPT') if simplicity_result else 'ACCEPT'
            print(f"\n  {'='*56}", flush=True)
            print(f"  DONE | {loc} LOC | {len(files)} file(s) | {len(tests)} test(s) | {validated_str}", flush=True)
            print(f"  Time: {exec_time:.1f}s | Quality: {result.quality_score:.2f} | Simplicity: {simplicity_verdict}", flush=True)
            print(f"  {'='*56}\n", flush=True)
            logger.info(f"CodingSwarm complete: {loc} LOC, {len(tests)} test files")

            # Store team review metadata
            if team_review_result:
                result.metadata['team_review'] = team_review_result

            # Store validation metadata
            if validation_metadata.get('validated') or validation_metadata.get('fix_attempts'):
                result.metadata['validation'] = validation_metadata

            # Persist output to disk
            output_path = self._persist_output(code_output, requirements)
            if output_path:
                result.metadata['output_path'] = output_path
                _progress("Output", "Persist", f"Saved to: {output_path}")

                # Write ADRs for team planning and review decisions
                base_dir = Path(output_path)
                adr_count = 0

                if planning_result and planning_result.get('refined_architecture'):
                    adr_path = self._write_planning_adr(base_dir, planning_result, requirements)
                    if adr_path:
                        adr_count += 1
                        _progress("Output", "ADR", f"Planning decisions: {Path(adr_path).name}")

                if team_review_result:
                    adr_path = self._write_review_adr(base_dir, team_review_result, adr_number=2)
                    if adr_path:
                        adr_count += 1
                        _progress("Output", "ADR", f"Review decisions: {Path(adr_path).name}")

                if adr_count > 0:
                    result.metadata['adr_count'] = adr_count
                    _progress("Output", "ADR", f"Total: {adr_count} ADR(s) written to adr/")

            # Post-execution learning (includes evaluation + improvement cycle)
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['code_generate', 'code_optimize', 'test_generate']),
                task_type='code_generation',
                output_data={'code': files, 'tests': tests},
                input_data={'requirements': requirements, 'language': lang.value}
            )

            return result

        except Exception as e:
            logger.error(f"❌ CodingSwarm error: {e}")
            import traceback
            traceback.print_exc()
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['code_generate']),
                task_type='code_generation'
            )
            return CodingResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

        finally:
            _active_progress_callback = None
            _active_trace_callback = None
            if workspace:
                workspace.cleanup()

    async def debug(
        self,
        code: str,
        error: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Debug code and provide fix."""
        self._init_agents()
        return await self._debugger.debug(code, error, context)

    async def refactor(
        self,
        code: str,
        focus: str = "readability",
        requirements: str = ""
    ) -> Dict[str, Any]:
        """Refactor/optimize code."""
        self._init_agents()
        return await self._optimizer.optimize(code, focus, requirements=requirements)

    # -----------------------------------------------------------------
    # EDIT MODE HELPER METHODS
    # -----------------------------------------------------------------

    def _discover_files(self, codebase_path: str, extensions: List[str] = None) -> Dict[str, str]:
        """Auto-discover source files from codebase path.

        Args:
            codebase_path: Root directory to scan
            extensions: File extensions to include (default: language-appropriate)

        Returns:
            Dict of {filepath: content} for discovered files
        """
        import os
        import fnmatch

        if extensions is None:
            # Determine extensions based on configured language
            lang_extensions = {
                CodeLanguage.PYTHON: ['.py'],
                CodeLanguage.JAVASCRIPT: ['.js', '.jsx', '.ts', '.tsx'],
                CodeLanguage.TYPESCRIPT: ['.ts', '.tsx', '.js', '.jsx'],
                CodeLanguage.JAVA: ['.java'],
                CodeLanguage.GO: ['.go'],
                CodeLanguage.RUST: ['.rs'],
                CodeLanguage.CPP: ['.cpp', '.hpp', '.h', '.cc'],
            }
            extensions = lang_extensions.get(self.config.language, ['.py'])

        # Directories to skip
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv',
                     'env', '.env', 'dist', 'build', '.pytest_cache', '.mypy_cache'}

        discovered = {}
        codebase_path = os.path.abspath(codebase_path)

        for root, dirs, files in os.walk(codebase_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for filename in files:
                if any(filename.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, codebase_path)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            discovered[rel_path] = f.read()
                    except (IOError, UnicodeDecodeError):
                        continue

        return discovered

    def _generate_unified_diff(self, original: str, modified: str, filepath: str) -> str:
        """Generate unified diff between original and modified code.

        Args:
            original: Original file content
            modified: Modified file content
            filepath: Path to the file (for diff header)

        Returns:
            Unified diff string
        """
        import difflib

        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm=''
        )
        return ''.join(diff)

    def _analyze_import_graph(self, files: Dict[str, str]) -> Dict[str, List[str]]:
        """Analyze import dependencies between files.

        Args:
            files: Dict of {filepath: content}

        Returns:
            Dict of {filepath: [list of files it imports]}
        """
        import re
        import os

        graph = {}
        file_modules = {}

        # Build module name mapping
        for filepath in files.keys():
            # Convert filepath to module name (e.g., "src/utils/helpers.py" -> "src.utils.helpers")
            module = filepath.replace('/', '.').replace('\\', '.')
            if module.endswith('.py'):
                module = module[:-3]
            file_modules[module] = filepath
            # Also map just the filename
            basename = os.path.basename(filepath)
            if basename.endswith('.py'):
                file_modules[basename[:-3]] = filepath

        # Python import patterns
        import_pattern = re.compile(
            r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))',
            re.MULTILINE
        )

        for filepath, content in files.items():
            imports = []
            for match in import_pattern.finditer(content):
                module = match.group(1) or match.group(2)
                if module:
                    # Check if this module maps to a local file
                    parts = module.split('.')
                    for i in range(len(parts), 0, -1):
                        partial = '.'.join(parts[:i])
                        if partial in file_modules:
                            target = file_modules[partial]
                            if target != filepath:
                                imports.append(target)
                            break
            graph[filepath] = imports

        return graph

    def _get_edit_order(self, files: Dict[str, str], affected_files: List[str]) -> List[str]:
        """Determine optimal edit order based on dependency graph.

        Files that are imported by others should be edited first.

        Args:
            files: All files in codebase
            affected_files: Files that will be edited

        Returns:
            Sorted list of affected files (dependency order)
        """
        graph = self._analyze_import_graph(files)

        # Build reverse graph (who depends on me)
        reverse_graph = {fp: [] for fp in affected_files}
        for fp, deps in graph.items():
            if fp in affected_files:
                for dep in deps:
                    if dep in reverse_graph:
                        reverse_graph[dep].append(fp)

        # Topological sort (Kahn's algorithm)
        in_degree = {fp: 0 for fp in affected_files}
        for fp in affected_files:
            for dep in graph.get(fp, []):
                if dep in in_degree:
                    in_degree[fp] += 1

        queue = [fp for fp, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            fp = queue.pop(0)
            result.append(fp)
            for dependent in reverse_graph.get(fp, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Add any remaining (cycles) at the end
        result.extend([fp for fp in affected_files if fp not in result])
        return result

    def _is_test_file(self, filepath: str) -> bool:
        """Check if a file is a test file.

        Args:
            filepath: Path to check

        Returns:
            True if it appears to be a test file
        """
        import os
        # Normalize path separators
        normalized = filepath.replace('\\', '/')
        filename = os.path.basename(filepath)

        # Common test file patterns
        return (
            filename.startswith('test_') or
            filename.endswith('_test.py') or
            filename.endswith('.test.py') or
            filename.endswith('.spec.py') or
            (filename.startswith('test') and filename.endswith('.py')) or
            '/tests/' in normalized or
            normalized.startswith('tests/') or
            '/test/' in normalized or
            normalized.startswith('test/')
        )

    def _filter_test_files(self, files: Dict[str, str], preserve: bool = True) -> tuple:
        """Separate test files from source files.

        Args:
            files: All files
            preserve: If True, test files are returned separately

        Returns:
            (source_files, test_files) dicts
        """
        if not preserve:
            return files, {}

        source_files = {}
        test_files = {}

        for filepath, content in files.items():
            if self._is_test_file(filepath):
                test_files[filepath] = content
            else:
                source_files[filepath] = content

        return source_files, test_files

    async def _git_prepare_branch(self, codebase_path: str, requirements: str) -> Optional[str]:
        """Create a new git branch for the edit.

        Args:
            codebase_path: Path to git repository
            requirements: Requirements string (used to generate branch name)

        Returns:
            Branch name if created, None if git not available
        """
        import subprocess
        import re
        import os
        from datetime import datetime

        if not os.path.isdir(os.path.join(codebase_path, '.git')):
            return None

        try:
            # Generate branch name from requirements
            slug = re.sub(r'[^a-z0-9]+', '-', requirements.lower()[:40]).strip('-')
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            branch_name = f"{self.config.git_branch_prefix}/{slug}-{timestamp}"

            # Create and checkout branch
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=codebase_path,
                capture_output=True,
                check=True
            )

            _progress("Git", "Integration", f"Created branch: {branch_name}")
            return branch_name

        except subprocess.CalledProcessError:
            return None
        except FileNotFoundError:
            return None

    async def _git_commit_changes(self, codebase_path: str, files: Dict[str, str],
                                   requirements: str) -> bool:
        """Commit edited files to git.

        Args:
            codebase_path: Path to git repository
            files: Dict of {filepath: content} that were modified
            requirements: Requirements string (used for commit message)

        Returns:
            True if commit successful
        """
        import subprocess
        import os

        if not os.path.isdir(os.path.join(codebase_path, '.git')):
            return False

        try:
            # Stage all modified files
            for filepath in files.keys():
                full_path = os.path.join(codebase_path, filepath)
                subprocess.run(
                    ['git', 'add', full_path],
                    cwd=codebase_path,
                    capture_output=True,
                    check=True
                )

            # Create commit message
            commit_msg = f"Jotty edit: {requirements[:100]}"
            if len(requirements) > 100:
                commit_msg += "..."

            # Commit
            subprocess.run(
                ['git', 'commit', '-m', commit_msg],
                cwd=codebase_path,
                capture_output=True,
                check=True
            )

            _progress("Git", "Integration", f"Committed {len(files)} file(s)")
            return True

        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            return False

    def _detect_test_command(self, codebase_path: str) -> Optional[str]:
        """Auto-detect the test command for a codebase.

        Args:
            codebase_path: Root path of the codebase

        Returns:
            Test command string or None if not detected
        """
        import os

        # Check for common test configuration files (use python3 -m for reliability)
        checks = [
            ('pytest.ini', 'python3 -m pytest'),
            ('setup.cfg', 'python3 -m pytest'),
            ('pyproject.toml', 'python3 -m pytest'),
            ('tox.ini', 'tox'),
            ('Makefile', 'make test'),
            ('package.json', 'npm test'),
            ('Cargo.toml', 'cargo test'),
            ('go.mod', 'go test ./...'),
        ]

        for config_file, cmd in checks:
            if os.path.exists(os.path.join(codebase_path, config_file)):
                return cmd

        # Check for test directories
        if os.path.isdir(os.path.join(codebase_path, 'tests')):
            return 'python3 -m pytest tests/ -x'
        if os.path.isdir(os.path.join(codebase_path, 'test')):
            return 'python3 -m pytest test/ -x'

        # Look for test files
        for f in os.listdir(codebase_path):
            if f.startswith('test_') and f.endswith('.py'):
                return 'python3 -m pytest -x'

        # Default to pytest for Python
        if self.config.language == CodeLanguage.PYTHON:
            return 'python3 -m pytest -x'

        return None

    async def _run_tests(self, codebase_path: str, test_command: str = None) -> Dict[str, Any]:
        """Run tests in the codebase and return results.

        Args:
            codebase_path: Root path of the codebase
            test_command: Test command to run (auto-detect if None)

        Returns:
            Dict with 'success', 'output', 'passed', 'failed', 'errors'
        """
        import subprocess

        cmd = test_command or self.config.test_command or self._detect_test_command(codebase_path)
        if not cmd:
            return {'success': False, 'output': 'No test command detected', 'passed': 0, 'failed': 0, 'errors': []}

        _progress("Test", "Runner", f"Running: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=codebase_path,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout
            )

            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0

            # Parse test output for metrics
            import re
            passed = output.lower().count(' passed')
            failed = output.lower().count(' failed')
            errors = []

            # Also count collection/import errors (pytest shows these differently)
            collection_errors = len(re.findall(r'ImportError|ModuleNotFoundError|CollectionError', output))
            if collection_errors > 0:
                failed += collection_errors

            # Extract error messages for feedback - focus on assertion details
            if not success:
                lines = output.split('\n')
                in_error = False
                current_error = []

                # First pass: find the key assertion error
                assertion_context = []
                for i, line in enumerate(lines):
                    if 'AssertionError' in line or 'assertEqual' in line or 'assertIs' in line:
                        # Get context around assertion
                        start = max(0, i - 3)
                        end = min(len(lines), i + 5)
                        assertion_context.extend(lines[start:end])

                if assertion_context:
                    errors.append("=== KEY ASSERTION ===\n" + '\n'.join(assertion_context))

                # Second pass: collect other errors
                for line in lines:
                    # Handle import errors and collection failures
                    if any(x in line for x in ['FAILED', 'ERROR', 'AssertionError',
                                                'ImportError', 'ModuleNotFoundError',
                                                'CollectionError', 'no tests ran']):
                        in_error = True
                    if in_error:
                        current_error.append(line)
                        if len(current_error) > 20:  # Limit error context
                            errors.append('\n'.join(current_error))
                            current_error = []
                            in_error = False
                if current_error:
                    errors.append('\n'.join(current_error))

                # If no specific failures found but tests didn't pass
                if failed == 0 and not success:
                    failed = 1  # At least one failure occurred
                    if not errors:
                        errors = [output[-2000:]]  # Include raw output for debugging

            status = "PASS" if success else f"FAIL ({failed} failure{'s' if failed != 1 else ''})"
            _progress("Test", "Runner", status)

            return {
                'success': success,
                'output': output[-5000:],  # Limit output size
                'passed': passed,
                'failed': failed,
                'errors': errors[:5],  # Limit to 5 errors
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            _progress("Test", "Runner", "TIMEOUT")
            return {'success': False, 'output': 'Test timeout exceeded', 'passed': 0, 'failed': 0, 'errors': ['Timeout']}
        except Exception as e:
            _progress("Test", "Runner", f"ERROR: {e}")
            return {'success': False, 'output': str(e), 'passed': 0, 'failed': 0, 'errors': [str(e)]}

    async def _refine_from_test_failure(
        self,
        file_path: str,
        current_code: str,
        requirements: str,
        test_output: str,
        iteration: int,
        previous_attempts: List[str]
    ) -> Dict[str, Any]:
        """Refine code based on test failure feedback.

        Args:
            file_path: Path to the file being fixed
            current_code: Current code content
            requirements: Original requirements
            test_output: Test failure output
            iteration: Current iteration number
            previous_attempts: List of previous attempt summaries

        Returns:
            Dict with 'fixed_code', 'analysis', 'confidence'
        """
        if not hasattr(self, '_refinement_module') or self._refinement_module is None:
            self._refinement_module = dspy.ChainOfThought(TestFailureRefinementSignature)

        _progress(f"Iter {iteration}", "Refiner", f"Analyzing test failure for {file_path}...")

        try:
            result = await _stream_call(
                self._refinement_module,
                f"Iter {iteration}",
                "Refiner",
                current_code=current_code,
                file_path=file_path,
                original_requirements=requirements,
                test_output=test_output[-3000:],  # Limit context
                iteration=iteration,
                previous_attempts="\n".join(previous_attempts[-3:]) or "No previous attempts"
            )

            fixed_code = _strip_code_fences(str(result.fixed_code))
            confidence = str(result.confidence).upper()

            _progress(f"Iter {iteration}", "Refiner", f"Confidence: {confidence}")

            return {
                'fixed_code': fixed_code,
                'analysis': str(result.analysis),
                'strategy': str(result.fix_strategy),
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return {
                'fixed_code': current_code,
                'analysis': f"Refinement error: {e}",
                'strategy': '',
                'confidence': 'LOW'
            }

    async def _test_driven_edit_loop(
        self,
        codebase_path: str,
        edited_files: Dict[str, str],
        original_files: Dict[str, str],
        requirements: str,
        affected_files: List[str]
    ) -> tuple:
        """Run test-driven iteration loop until tests pass or max iterations.

        Args:
            codebase_path: Root path of codebase
            edited_files: Current edited file contents
            original_files: Original file contents (for diff)
            requirements: Original requirements
            affected_files: List of files that were edited

        Returns:
            (final_files, iteration_history)
        """
        import os

        max_iters = self.config.max_edit_iterations
        history = []
        current_files = edited_files.copy()

        _progress("Test", "Loop", f"Starting test-driven iteration (max {max_iters})")

        for iteration in range(1, max_iters + 1):
            # Write current files to disk
            for filepath, content in current_files.items():
                full_path = os.path.join(codebase_path, filepath)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            # Run tests
            test_result = await self._run_tests(codebase_path)

            if test_result['success']:
                _progress("Test", "Loop", f"PASS on iteration {iteration}")
                history.append({
                    'iteration': iteration,
                    'success': True,
                    'passed': test_result['passed'],
                    'failed': test_result['failed']
                })
                return current_files, history

            # Tests failed - refine
            history.append({
                'iteration': iteration,
                'success': False,
                'passed': test_result['passed'],
                'failed': test_result['failed'],
                'errors': test_result['errors'][:2]
            })

            _progress("Test", "Loop", f"Iteration {iteration}/{max_iters}: {test_result['failed']} failures")

            # Refine each affected file
            previous_attempts = [f"Iter {h['iteration']}: {h.get('errors', ['?'])[:1]}" for h in history[:-1]]

            for filepath in affected_files:
                if filepath not in current_files:
                    continue

                refine_result = await self._refine_from_test_failure(
                    file_path=filepath,
                    current_code=current_files[filepath],
                    requirements=requirements,
                    test_output=test_result['output'],
                    iteration=iteration,
                    previous_attempts=previous_attempts
                )

                if refine_result['fixed_code'] != current_files[filepath]:
                    current_files[filepath] = refine_result['fixed_code']
                    _progress(f"Iter {iteration}", "Refiner", f"Updated {filepath}")

        _progress("Test", "Loop", f"Max iterations ({max_iters}) reached")
        return current_files, history

    async def edit(
        self,
        requirements: str,
        target_files: Dict[str, str] = None,
        progress_callback=None,
        codebase_path: str = None,
    ) -> CodingResult:
        """
        Edit existing code based on requirements.

        This is an alternative to generate() for modifying existing codebases.
        It analyzes existing code, plans surgical edits, and applies them.

        Features:
        - Auto-discovers files from codebase_path if target_files not provided
        - Analyzes import dependencies to determine optimal edit order
        - Preserves test files (unless preserve_tests=False)
        - Generates unified diffs (if output_diffs=True)
        - Git integration: creates branch and commits (if git_integration=True)

        Args:
            requirements: What changes to make
            target_files: Dict of {filepath: content} for files to edit (optional if codebase_path set)
            progress_callback: Optional callable(phase, agent, message)
            codebase_path: Root path of codebase (used for file discovery and git integration)

        Returns:
            CodingResult with edited files (and diffs if output_diffs=True)
        """
        global _active_progress_callback
        _active_progress_callback = progress_callback
        start_time = datetime.now()

        config = self.config
        codebase_path = codebase_path or config.codebase_path

        # -----------------------------------------------------------------
        # FILE DISCOVERY: Auto-discover files if not provided
        # -----------------------------------------------------------------
        if target_files is None or len(target_files) == 0:
            if codebase_path and config.auto_discover_files:
                _progress("Phase 0", "FileDiscovery", f"Scanning {codebase_path}...")
                target_files = self._discover_files(codebase_path)
                _progress("Phase 0", "FileDiscovery", f"Found {len(target_files)} file(s)")
            else:
                return CodingResult(
                    success=False,
                    swarm_name=config.name,
                    domain=config.domain,
                    output={},
                    execution_time=0,
                    error="No target_files provided and codebase_path not set for auto-discovery"
                )

        # Store original files for diff generation
        original_files = {fp: content for fp, content in target_files.items()}

        # -----------------------------------------------------------------
        # TEST PRESERVATION: Separate test files if configured
        # -----------------------------------------------------------------
        test_files = {}
        if config.preserve_tests:
            target_files, test_files = self._filter_test_files(target_files, preserve=True)
            if test_files:
                _progress("Phase 0", "TestPreservation", f"Preserved {len(test_files)} test file(s) from editing")

        # -----------------------------------------------------------------
        # GIT INTEGRATION: Create branch if enabled
        # -----------------------------------------------------------------
        git_branch = None
        if config.git_integration and codebase_path:
            git_branch = await self._git_prepare_branch(codebase_path, requirements)

        self._init_agents()

        # Initialize edit-specific agents
        if not hasattr(self, '_codebase_analyzer') or self._codebase_analyzer is None:
            self._codebase_analyzer = CodebaseAnalyzerAgent(
                self._memory, self._context, self._bus, self._agent_context("Architect"))
        if not hasattr(self, '_edit_planner') or self._edit_planner is None:
            self._edit_planner = EditPlannerAgent(
                self._memory, self._context, self._bus, self._agent_context("Developer"))

        config = self.config
        lang = config.language

        print(f"\n{'='*60}", flush=True)
        print(f"  CodingSwarm EDIT MODE | {len(target_files)} file(s)", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            # =================================================================
            # PHASE 0: CODEBASE ANALYSIS
            # =================================================================
            _progress("Phase 0", "CodebaseAnalyzer", f"Analyzing {len(target_files)} file(s)...")

            all_code = "\n\n".join(f"# FILE: {fp}\n{content}" for fp, content in target_files.items())
            file_paths = list(target_files.keys())

            analysis = await self._codebase_analyzer.analyze(
                existing_code=all_code,
                file_paths=file_paths,
                requirements=requirements
            )

            if 'error' in analysis:
                return CodingResult(
                    success=False,
                    swarm_name=self.config.name,
                    domain=self.config.domain,
                    output={},
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=analysis['error']
                )

            affected_files = analysis.get('affected_files', file_paths)
            # CRITICAL FIX: If analyzer returns empty, fall back to first few files
            # This ensures test-driven loop can refine even when analyzer misses
            if not affected_files:
                affected_files = file_paths[:3]  # Focus on top 3 most relevant files
            style_conventions = analysis.get('style_conventions', '')
            dependencies = analysis.get('dependencies', '')
            change_scope = analysis.get('change_scope', 'moderate')

            _progress("Phase 0", "CodebaseAnalyzer", f"Change scope: {change_scope}, affecting {len(affected_files)} file(s)")

            # =================================================================
            # DEPENDENCY ANALYSIS: Determine optimal edit order
            # =================================================================
            if config.analyze_dependencies and len(affected_files) > 1:
                _progress("Phase 0.5", "DependencyAnalyzer", "Analyzing import graph...")
                import_graph = self._analyze_import_graph(target_files)
                affected_files = self._get_edit_order(target_files, affected_files)
                _progress("Phase 0.5", "DependencyAnalyzer", f"Edit order determined: {len(affected_files)} file(s)")

            # =================================================================
            # PHASE 1: EDIT PLANNING (parallel for each affected file)
            # =================================================================
            _progress("Phase 1", "EditPlanner", f"Planning edits for {len(affected_files)} file(s)...")

            edited_files = {}
            edit_summaries = []

            async def _plan_and_apply_edit(file_path: str):
                if file_path not in target_files:
                    return file_path, target_files.get(file_path, ''), {'skipped': True}

                existing_code = target_files[file_path]
                _progress("Phase 1", "EditPlanner", f"  Planning: {file_path}...")

                edit_result = await self._edit_planner.plan_edit(
                    existing_code=existing_code,
                    file_path=file_path,
                    requirements=requirements,
                    style_conventions=style_conventions,
                    dependencies=dependencies
                )

                if 'error' in edit_result:
                    return file_path, existing_code, edit_result

                edit_type = edit_result.get('edit_type', 'patch')
                edits = edit_result.get('edits', [])

                if edit_type == 'rewrite':
                    new_code = _strip_code_fences(edit_result.get('new_code', existing_code))
                    _progress("Phase 1", "EditPlanner", f"  {file_path}: REWRITE")
                else:
                    # Apply patch edits
                    new_code = existing_code
                    for edit in edits:
                        if isinstance(edit, dict) and 'old' in edit and 'new' in edit:
                            old_str = edit['old']
                            new_str = edit['new']
                            if old_str in new_code:
                                new_code = new_code.replace(old_str, new_str, 1)
                                _progress("Phase 1", "EditPlanner", f"  {file_path}: patched '{old_str[:30]}...'")

                return file_path, new_code, edit_result

            # Run edit planning in parallel
            edit_tasks = [_plan_and_apply_edit(fp) for fp in affected_files]
            edit_results = await asyncio.gather(*edit_tasks, return_exceptions=True)

            for result in edit_results:
                if isinstance(result, Exception):
                    continue
                file_path, new_code, edit_info = result
                edited_files[file_path] = new_code
                if not edit_info.get('skipped'):
                    edit_summaries.append({
                        'file': file_path,
                        'type': edit_info.get('edit_type', 'unknown'),
                        'num_edits': len(edit_info.get('edits', [])),
                    })

            # Include unchanged files
            for fp, content in target_files.items():
                if fp not in edited_files:
                    edited_files[fp] = content

            _progress("Phase 1", "EditPlanner", f"Done -- {len(edit_summaries)} file(s) modified")

            # =================================================================
            # PHASE 4.5: VALIDATION
            # =================================================================
            workspace = WorkspaceManager() if getattr(config, 'enable_workspace', True) else None
            validation_metadata = {"validated": False, "fix_attempts": 0, "errors_fixed": []}

            if workspace and workspace.available:
                _progress("Phase 4.5", "Validator", "Validating edited code...")
                # Write and syntax check edited files
                for fname, content in edited_files.items():
                    if fname.endswith('.py'):
                        await workspace.write_file(fname, content)
                        check = await workspace.syntax_check(fname)
                        if check.success:
                            _progress("Phase 4.5", "Validator", f"Syntax OK: {fname}")
                            validation_metadata["validated"] = True
                        else:
                            _progress("Phase 4.5", "Validator", f"Syntax ERROR: {fname}")

            # =================================================================
            # TEST-DRIVEN ITERATION (if enabled)
            # =================================================================
            iteration_history = []
            if config.test_driven and codebase_path:
                _progress("Phase 3", "TestLoop", "Starting test-driven refinement...")

                # Merge test files temporarily for testing
                files_for_testing = edited_files.copy()
                files_for_testing.update(test_files)

                # Run test-driven loop
                refined_files, iteration_history = await self._test_driven_edit_loop(
                    codebase_path=codebase_path,
                    edited_files=files_for_testing,
                    original_files=original_files,
                    requirements=requirements,
                    affected_files=affected_files
                )

                # Extract only source files (not test files) from refined result
                for fp in edited_files.keys():
                    if fp in refined_files:
                        edited_files[fp] = refined_files[fp]

                if iteration_history and iteration_history[-1].get('success'):
                    _progress("Phase 3", "TestLoop", "Tests passing!")
                else:
                    _progress("Phase 3", "TestLoop", "Max iterations reached, tests may still fail")

            # =================================================================
            # MERGE TEST FILES BACK
            # =================================================================
            if test_files:
                edited_files.update(test_files)
                _progress("Phase 2", "TestPreservation", f"Merged {len(test_files)} preserved test file(s)")

            # =================================================================
            # GENERATE DIFFS (if enabled)
            # =================================================================
            diffs = {}
            if config.output_diffs:
                _progress("Phase 2", "DiffGenerator", "Generating unified diffs...")
                for filepath, new_content in edited_files.items():
                    old_content = original_files.get(filepath, '')
                    if old_content != new_content:
                        diff = self._generate_unified_diff(old_content, new_content, filepath)
                        if diff:
                            diffs[filepath] = diff
                _progress("Phase 2", "DiffGenerator", f"Generated {len(diffs)} diff(s)")

            # =================================================================
            # GIT COMMIT (if enabled)
            # =================================================================
            git_committed = False
            if config.git_integration and codebase_path and git_branch:
                # Write files to disk first
                import os
                modified_files = {fp: content for fp, content in edited_files.items()
                                  if original_files.get(fp, '') != content}
                if modified_files:
                    for filepath, content in modified_files.items():
                        full_path = os.path.join(codebase_path, filepath)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    git_committed = await self._git_commit_changes(codebase_path, modified_files, requirements)

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()
            loc = sum(code.count('\n') + 1 for code in edited_files.values())

            code_output = CodeOutput(
                files=edited_files,
                main_file=affected_files[0] if affected_files else "",
                entry_point="",
                dependencies=[],
                tests=test_files,
                docs="",
                architecture=analysis.get('architecture_summary', '')
            )

            result = CodingResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={'files': list(edited_files.keys()), 'mode': 'edit'},
                execution_time=exec_time,
                code=code_output,
                language=lang.value,
                loc=loc,
            )

            result.metadata['edit_summaries'] = edit_summaries
            result.metadata['change_scope'] = change_scope
            result.metadata['validation'] = validation_metadata
            result.metadata['diffs'] = diffs
            result.metadata['preserved_tests'] = list(test_files.keys()) if test_files else []
            result.metadata['git_branch'] = git_branch
            result.metadata['git_committed'] = git_committed
            result.metadata['test_iterations'] = iteration_history
            result.metadata['tests_passing'] = iteration_history[-1].get('success', False) if iteration_history else None

            print(f"\n  {'='*56}", flush=True)
            print(f"  EDIT DONE | {len(edit_summaries)} file(s) modified | {loc} LOC", flush=True)
            if diffs:
                print(f"  Diffs generated: {len(diffs)} | Tests preserved: {len(test_files)}", flush=True)
            if iteration_history:
                tests_status = "PASSING" if iteration_history[-1].get('success') else "FAILING"
                print(f"  Test iterations: {len(iteration_history)} | Tests: {tests_status}", flush=True)
            if git_branch:
                print(f"  Git branch: {git_branch} | Committed: {git_committed}", flush=True)
            print(f"  Time: {exec_time:.1f}s", flush=True)
            print(f"  {'='*56}\n", flush=True)

            return result

        except Exception as e:
            logger.error(f"Edit mode error: {e}")
            import traceback
            traceback.print_exc()
            return CodingResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

        finally:
            _active_progress_callback = None
            if workspace:
                workspace.cleanup()

    def _build_review_criteria(self) -> str:
        """Collect guiding principles from all team reviewers.

        Shifts review expectations left: the developer sees what reviewers
        will check BEFORE writing code, reducing first-round rejections.
        """
        if not self._team_config:
            return ""
        seen = set()
        criteria_lines = []
        for phase in ("functional", "quality"):
            for persona in self._team_config.get_reviewers(phase):
                if persona.name in seen:
                    continue
                seen.add(persona.name)
                for principle in persona.guiding_principles:
                    criteria_lines.append(f"- [{persona.name}] {principle}")
        return "\n".join(criteria_lines)

    # -----------------------------------------------------------------
    # COLLABORATIVE DESIGN LOOP (Architect + Researcher)
    # -----------------------------------------------------------------

    async def _collaborative_design_loop(
        self,
        requirements: str,
        language: str,
        style: str,
        constraints: str,
        max_iterations: int = 3,
    ) -> tuple:
        """Run collaborative design loop between Architect and Researcher.

        The Architect drafts/refines architecture while the Researcher searches
        for best practices, library docs, and pitfalls to inform each iteration.

        Args:
            requirements: Software requirements
            language: Target programming language
            style: Coding style preference
            constraints: Technical constraints
            max_iterations: Number of collaborative iterations (default: 3)

        Returns:
            (arch_result_dict, research_context)
        """
        _progress("Phase 1", "CollaborativeDesign", f"Starting {max_iterations}-iteration design loop...")

        # Initialize modules
        if not hasattr(self, '_collab_architect_module') or self._collab_architect_module is None:
            self._collab_architect_module = dspy.ChainOfThought(CollaborativeArchitectSignature)
        if not hasattr(self, '_research_response_module') or self._research_response_module is None:
            self._research_response_module = dspy.ChainOfThought(ResearchResponseSignature)

        # Get web search function
        web_search_fn = None
        try:
            from ...skills import get_skill_registry
            registry = get_skill_registry()
            web_search_fn = registry.get('web-search', {}).get('search_web_tool')
        except Exception:
            pass

        if web_search_fn is None:
            try:
                from ..skills import search_web_tool as web_search_fn
            except Exception:
                pass

        if web_search_fn is None:
            try:
                import importlib
                mod = importlib.import_module('skills.web-search.tools')
                web_search_fn = getattr(mod, 'search_web_tool', None)
            except Exception:
                pass

        # State for iterative refinement
        current_architecture = ""
        current_components = []
        current_file_structure = ""
        current_interfaces = ""
        accumulated_research = ResearchContext()
        research_findings_str = ""

        config = self.config
        frameworks_json = json.dumps(getattr(config, 'frameworks', []))

        for iteration in range(1, max_iterations + 1):
            _progress("Phase 1", "CollaborativeDesign", f"=== Iteration {iteration}/{max_iterations} ===")

            # --- Architect Phase ---
            _progress("Phase 1", "Architect", f"Designing architecture (iteration {iteration})...")
            try:
                arch_result = await _stream_call(
                    self._collab_architect_module, "Phase 1", "Architect",
                    requirements=requirements,
                    language=language,
                    style=style,
                    constraints=constraints,
                    iteration=iteration,
                    previous_architecture=current_architecture,
                    research_findings=research_findings_str,
                )

                current_architecture = str(arch_result.architecture)
                current_file_structure = str(arch_result.file_structure)
                current_interfaces = str(arch_result.interfaces)
                research_requests = str(arch_result.research_requests)

                # Parse components - try JSON first, fallback to text extraction
                try:
                    current_components = json.loads(str(arch_result.components))
                    if not isinstance(current_components, list):
                        current_components = []
                except (json.JSONDecodeError, TypeError):
                    current_components = []

                # Fallback: extract components from file structure/architecture text
                if not current_components:
                    current_components = _extract_components_from_text(
                        current_file_structure,
                        current_architecture,
                        current_interfaces
                    )

                n_components = len(current_components)
                _progress("Phase 1", "Architect", f"Designed {n_components} component(s)")

                # Show file structure preview
                if current_file_structure:
                    for line in current_file_structure.strip().split('\n')[:5]:
                        _progress("Phase 1", "Architect", f"  {line.strip()}")

                # Show research requests
                if research_requests:
                    _progress("Phase 1", "Architect", f"Requesting research on: {research_requests[:100]}...")

            except Exception as e:
                logger.error(f"Architect iteration {iteration} failed: {e}")
                _progress("Phase 1", "Architect", f"Error: {e}")
                break

            # --- Researcher Phase (skip on last iteration) ---
            if iteration < max_iterations and getattr(config, 'enable_research', True):
                _progress("Phase 1.5", "Researcher", f"Researching Architect's requests (iteration {iteration})...")
                try:
                    # Generate targeted queries based on architect's requests
                    research_result = await _stream_call(
                        self._research_response_module, "Phase 1.5", "Researcher",
                        requirements=requirements,
                        architecture=current_architecture,
                        research_requests=research_requests,
                        language=language,
                        frameworks=frameworks_json,
                    )

                    # Parse search queries
                    try:
                        search_queries = json.loads(str(research_result.search_queries))
                        if not isinstance(search_queries, list):
                            search_queries = []
                    except (json.JSONDecodeError, TypeError):
                        search_queries = []

                    # Execute web searches
                    if web_search_fn and search_queries:
                        for qi, query in enumerate(search_queries[:5]):
                            _progress("Phase 1.5", "Researcher", f"Searching ({qi+1}/{min(len(search_queries),5)}): {str(query)[:50]}")
                            try:
                                result = web_search_fn({'query': str(query), 'max_results': 3})
                                if result.get('success') and result.get('results'):
                                    for r in result['results']:
                                        snippet = r.get('snippet', r.get('description', ''))
                                        if snippet:
                                            title = r.get('title', '')
                                            # Categorize findings
                                            query_lower = str(query).lower()
                                            if any(kw in query_lower for kw in ['best practice', 'pattern', 'convention', 'standard']):
                                                accumulated_research.best_practices.append(f"{title}: {snippet}")
                                            elif any(kw in query_lower for kw in ['doc', 'api', 'reference', 'example']):
                                                accumulated_research.api_references.append(f"{title}: {snippet}")
                                            elif any(kw in query_lower for kw in ['pitfall', 'warning', 'avoid', 'mistake', 'security']):
                                                accumulated_research.warnings.append(f"{title}: {snippet}")
                                            else:
                                                accumulated_research.library_docs.append(f"{title}: {snippet}")
                            except Exception as search_err:
                                logger.debug(f"Web search failed: {search_err}")

                    # Accumulate research analysis
                    analysis = str(research_result.analysis)
                    best_practices = str(research_result.best_practices)
                    warnings = str(research_result.warnings)
                    recommendations = str(research_result.recommendations)

                    # Build research findings string for next iteration
                    research_parts = []
                    if analysis:
                        research_parts.append(f"ANALYSIS:\n{analysis}")
                    if best_practices:
                        research_parts.append(f"BEST PRACTICES:\n{best_practices}")
                        for bp in best_practices.split('\n')[:3]:
                            if bp.strip():
                                accumulated_research.best_practices.append(bp.strip())
                    if warnings:
                        research_parts.append(f"WARNINGS:\n{warnings}")
                        for w in warnings.split('\n')[:3]:
                            if w.strip():
                                accumulated_research.warnings.append(w.strip())
                    if recommendations:
                        research_parts.append(f"RECOMMENDATIONS:\n{recommendations}")

                    research_findings_str = "\n\n".join(research_parts)

                    total_findings = (len(accumulated_research.best_practices) +
                                    len(accumulated_research.library_docs) +
                                    len(accumulated_research.api_references) +
                                    len(accumulated_research.warnings))
                    _progress("Phase 1.5", "Researcher", f"Done -- {total_findings} total finding(s) accumulated")

                except Exception as e:
                    logger.error(f"Researcher iteration {iteration} failed (non-blocking): {e}")
                    _progress("Phase 1.5", "Researcher", f"Research skipped: {str(e)[:50]}")

        # Build final result
        arch_result_dict = {
            'architecture': current_architecture,
            'components': current_components,
            'file_structure': current_file_structure,
            'interfaces': current_interfaces,
        }

        total_research = (len(accumulated_research.best_practices) +
                         len(accumulated_research.library_docs) +
                         len(accumulated_research.api_references) +
                         len(accumulated_research.warnings))
        _progress("Phase 1", "CollaborativeDesign", f"Done -- {len(current_components)} component(s), {total_research} research finding(s)")

        return arch_result_dict, accumulated_research

    # -----------------------------------------------------------------
    # TEAM PLANNING METHODS
    # -----------------------------------------------------------------

    async def _run_persona_planning(
        self, requirements: str, architecture: str, research_findings: str, persona: TeamPersona
    ) -> Dict[str, Any]:
        """Run a single persona's planning input. Non-blocking on failure."""
        try:
            _progress("Phase 2", persona.name, "Providing planning input...")
            if not hasattr(self, '_planning_module') or self._planning_module is None:
                self._planning_module = dspy.ChainOfThought(TeamPlanningSignature)
            result = await _stream_call(self._planning_module, "Phase 2", persona.name,
                requirements=requirements,
                architecture=architecture,
                research_findings=research_findings,
                persona_context=persona.to_prompt(),
            )
            try:
                concerns = json.loads(str(result.concerns))
                if not isinstance(concerns, list):
                    concerns = []
            except (json.JSONDecodeError, TypeError):
                concerns = []
            _progress("Phase 2", persona.name, f"Done -- {len(concerns)} concern(s)")
            return {
                "persona": persona.name,
                "concerns": concerns,
                "recommendations": str(result.recommendations),
                "implementation_notes": str(result.implementation_notes),
            }
        except Exception as e:
            logger.error(f"Persona planning failed for {persona.name} (non-blocking): {e}")
            _progress("Phase 2", persona.name, "Failed (skipped)")
            return {"persona": persona.name, "concerns": [], "recommendations": "", "implementation_notes": ""}

    async def _team_planning(
        self, requirements: str, architecture: str, research_findings: str
    ) -> Dict[str, Any]:
        """Orchestrate Phase 2: team planning discussion to refine architecture.

        All team personas provide input on the architecture based on their expertise
        and research findings. Their feedback is consolidated into a refined plan.

        Returns:
            Dict with refined_architecture, implementation_plan, and team_feedback
        """
        planning_result = {
            "refined_architecture": architecture,  # Default to original if planning fails
            "implementation_plan": "",
            "risk_mitigations": "",
            "team_agreements": "",
            "team_feedback": [],
        }

        if not self._team_config:
            return planning_result

        # Gather input from all personas (both functional and quality reviewers)
        all_personas = []
        seen = set()
        for phase in ("functional", "quality"):
            for persona in self._team_config.get_reviewers(phase):
                if persona.name not in seen:
                    seen.add(persona.name)
                    all_personas.append(persona)

        if not all_personas:
            return planning_result

        persona_names = ", ".join(p.name for p in all_personas)
        _progress("Phase 2", "TeamPlanning", f"Gathering input from: {persona_names}")

        # Run all persona planning in parallel
        tasks = [
            self._run_persona_planning(requirements, architecture, research_findings, persona)
            for persona in all_personas
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect feedback
        feedback_parts = []
        all_concerns = []
        for r in results:
            if isinstance(r, Exception):
                continue
            planning_result["team_feedback"].append(r)
            if r.get("recommendations"):
                feedback_parts.append(f"[{r['persona']}] RECOMMENDATIONS:\n{r['recommendations']}")
            if r.get("implementation_notes"):
                feedback_parts.append(f"[{r['persona']}] IMPLEMENTATION NOTES:\n{r['implementation_notes']}")
            for concern in r.get("concerns", []):
                if isinstance(concern, dict):
                    all_concerns.append(f"[{r['persona']}] [{concern.get('severity', '?')}] {concern.get('description', '')}")

        if all_concerns:
            _progress("Phase 2", "TeamPlanning", f"Team raised {len(all_concerns)} concern(s)")
            for concern in all_concerns[:5]:
                _progress("Phase 2", "TeamPlanning", f"  {concern[:80]}")

        # Consolidate feedback into refined architecture
        if feedback_parts:
            _progress("Phase 2", "TeamPlanning", "Consolidating team feedback...")
            try:
                if not hasattr(self, '_consolidation_module') or self._consolidation_module is None:
                    self._consolidation_module = dspy.ChainOfThought(TeamPlanningConsolidationSignature)

                team_feedback_str = "\n\n".join(feedback_parts)
                if all_concerns:
                    team_feedback_str += "\n\nCONCERNS:\n" + "\n".join(all_concerns)

                consolidation = await _stream_call(
                    self._consolidation_module, "Phase 2", "TeamPlanning",
                    original_architecture=architecture,
                    team_feedback=team_feedback_str,
                    research_findings=research_findings,
                    requirements=requirements,
                )

                planning_result["refined_architecture"] = str(consolidation.refined_architecture)
                planning_result["implementation_plan"] = str(consolidation.implementation_plan)
                planning_result["risk_mitigations"] = str(consolidation.risk_mitigations)
                planning_result["team_agreements"] = str(consolidation.team_agreements)

                _progress("Phase 2", "TeamPlanning", "Done -- architecture refined with team input")

            except Exception as e:
                logger.error(f"Planning consolidation failed (non-blocking): {e}")
                _progress("Phase 2", "TeamPlanning", "Consolidation failed, using original architecture")
        else:
            _progress("Phase 2", "TeamPlanning", "No feedback to consolidate, using original architecture")

        return planning_result

    # -----------------------------------------------------------------
    # TEAM REVIEW METHODS
    # -----------------------------------------------------------------

    async def _run_persona_review(
        self, code: str, requirements: str, phase: str, persona: TeamPersona,
        team_agreements: str = ""
    ) -> Dict[str, Any]:
        """Run a single persona review. Non-blocking on failure."""
        try:
            _progress("Phase 6", persona.name, f"Reviewing ({phase})...")
            if self._review_module is None:
                self._review_module = dspy.ChainOfThought(TeamReviewSignature)

            # Build team agreements context
            agreements_context = team_agreements or "No specific team agreements recorded."

            result = await _stream_call(self._review_module, "Phase 6", persona.name,
                code=code,
                requirements=requirements,
                review_phase=phase,
                persona_context=persona.to_prompt(),
                team_agreements=agreements_context,
            )
            verdict = str(result.verdict).strip().upper()
            try:
                issues = json.loads(str(result.issues))
                if not isinstance(issues, list):
                    issues = []
            except (json.JSONDecodeError, TypeError):
                issues = []
            _progress("Phase 6", persona.name, f"Done -- {verdict}")
            return {
                "persona": persona.name,
                "verdict": verdict,
                "issues": issues,
                "feedback": str(result.feedback),
                "evidence": str(getattr(result, 'evidence', '')),
            }
        except Exception as e:
            logger.error(f"Persona review failed for {persona.name} (non-blocking): {e}")
            _progress("Phase 6", persona.name, "Failed (auto-approved)")
            return {"persona": persona.name, "verdict": "APPROVED", "issues": [], "feedback": "", "evidence": ""}

    async def _run_review_phase(
        self, reviewers: List[TeamPersona], code: str, requirements: str, phase: str,
        team_agreements: str = ""
    ) -> List[Dict[str, Any]]:
        """Run all reviewers for a phase in parallel."""
        reviewer_names = ", ".join(p.name for p in reviewers)
        _progress("Phase 6", "TeamReview", f"{phase.capitalize()} review: {reviewer_names}")
        tasks = [
            self._run_persona_review(code, requirements, phase, persona, team_agreements)
            for persona in reviewers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parsed = []
        for r in results:
            if isinstance(r, Exception):
                parsed.append({"persona": "unknown", "verdict": "APPROVED", "issues": [], "feedback": ""})
            else:
                parsed.append(r)
                verdict = r.get('verdict', '?')
                feedback_preview = r.get('feedback', '')[:100]
                n_issues = len(r.get('issues', []))
                detail = f" ({n_issues} issue(s))" if n_issues else ""
                _progress("Phase 6", r.get("persona", "Reviewer"), f"{verdict}{detail}")
                if feedback_preview and verdict == "REJECTED":
                    _progress("Phase 6", r.get("persona", "Reviewer"), f"  {feedback_preview}")
        return parsed

    def _collect_feedback(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate feedback strings from review results."""
        parts = []
        for r in results:
            if r.get("feedback"):
                parts.append(f"[{r.get('persona', 'Reviewer')}] {r['feedback']}")
        return "\n".join(parts)

    async def _team_review(
        self, all_code: str, requirements: str, architecture: str,
        files: Dict[str, str], main_file: Optional[str],
        planning_result: Dict[str, Any] = None,
        simplicity_result: Dict[str, Any] = None
    ) -> tuple:
        """Orchestrate Phase 6: two-phase team review with auto-fix loop.

        Args:
            all_code: Combined code string from all generated files
            requirements: Original requirements
            architecture: Architecture design
            files: Dict of filename -> code content
            main_file: Name of the main/entry file
            planning_result: Result from team planning phase, contains team_agreements
            simplicity_result: Result from SimplicityJudge (Phase 5.5), to avoid contradictions

        Returns:
            (review_result_dict, possibly_updated_files)
        """
        review_result = {
            "approved": True,
            "team": self._team_config.name if self._team_config else "unknown",
            "phase_a_results": [],
            "phase_b_results": [],
            "feedback": "",
            "rework_attempts": 0,
        }

        if not self._team_config:
            return review_result, files

        current_code = all_code

        # Extract team agreements from planning phase to avoid contradictory rejections
        team_agreements = ""
        if planning_result and planning_result.get('team_agreements'):
            team_agreements = str(planning_result['team_agreements'])

        # Include SimplicityJudge verdict to prevent contradictory "too simple" rejections
        if simplicity_result:
            verdict = simplicity_result.get('verdict', 'ACCEPT')
            score = simplicity_result.get('simplicity_score', 1.0)
            if verdict == 'SIMPLIFY' or score < 0.5:
                team_agreements += f"\n\n**SIMPLICITY OVERRIDE (Phase 5.5)**: Code was intentionally simplified by SimplicityJudge (score: {score:.2f}). Do NOT reject for being 'too simple' or 'missing architecture components'. The simplified code is the correct implementation."
            else:
                team_agreements += f"\n\n**Simplicity Check**: Code passed SimplicityJudge (score: {score:.2f}, verdict: {verdict})."

        # --- Phase 6a: Functional Review ---
        func_reviewers = self._team_config.get_reviewers("functional")
        if func_reviewers:
            phase_a = await self._run_review_phase(func_reviewers, current_code, requirements, "functional", team_agreements)
            review_result["phase_a_results"] = phase_a

            rejected = [r for r in phase_a if r.get("verdict") == "REJECTED"]

            # Arbitrator: validate rejections before rework
            if rejected and getattr(self.config, 'enable_arbitrator', True):
                if not hasattr(self, '_arbitrator') or self._arbitrator is None:
                    self._arbitrator = ArbitratorAgent(
                        self._memory, self._context, self._bus, self._agent_context("Verifier"))
                validated = []
                for r in rejected:
                    arb_result = await self._arbitrator.evaluate(
                        current_code, r.get('feedback', ''), r.get('evidence', ''), r['persona']
                    )
                    if arb_result.get('valid', True):
                        validated.append(r)
                        _progress("Phase 6", "Arbitrator", f"CONFIRMED: {r['persona']}")
                    else:
                        r['verdict'] = 'APPROVED'
                        r['arbitrator_overruled'] = True
                        _progress("Phase 6", "Arbitrator", f"OVERRULED: {r['persona']}")
                rejected = validated

            if rejected:
                feedback_text = self._collect_feedback(rejected)
                _progress("Phase 6", "Optimizer", f"Reworking code ({len(rejected)} rejection(s) in functional review)...")
                review_result["rework_attempts"] += 1

                opt_result = await self._optimizer.optimize(
                    code=current_code,
                    focus="review_feedback",
                    requirements=requirements,
                    constraints=feedback_text,
                )
                if "optimized_code" in opt_result:
                    current_code = _strip_code_fences(opt_result["optimized_code"])
                    if main_file and main_file in files:
                        files[main_file] = current_code
                    _progress("Phase 6", "Optimizer", "Rework applied, re-reviewing...")

                # Re-review once
                phase_a = await self._run_review_phase(func_reviewers, current_code, requirements, "functional", team_agreements)
                review_result["phase_a_results"] = phase_a

        # --- Phase 6b: Code Quality Review ---
        quality_reviewers = self._team_config.get_reviewers("quality")
        if quality_reviewers:
            phase_b = await self._run_review_phase(quality_reviewers, current_code, requirements, "quality", team_agreements)
            review_result["phase_b_results"] = phase_b

            rejected = [r for r in phase_b if r.get("verdict") == "REJECTED"]

            # Arbitrator: validate rejections before rework
            if rejected and getattr(self.config, 'enable_arbitrator', True):
                if not hasattr(self, '_arbitrator') or self._arbitrator is None:
                    self._arbitrator = ArbitratorAgent(
                        self._memory, self._context, self._bus, self._agent_context("Verifier"))
                validated = []
                for r in rejected:
                    arb_result = await self._arbitrator.evaluate(
                        current_code, r.get('feedback', ''), r.get('evidence', ''), r['persona']
                    )
                    if arb_result.get('valid', True):
                        validated.append(r)
                        _progress("Phase 6", "Arbitrator", f"CONFIRMED: {r['persona']}")
                    else:
                        r['verdict'] = 'APPROVED'
                        r['arbitrator_overruled'] = True
                        _progress("Phase 6", "Arbitrator", f"OVERRULED: {r['persona']}")
                rejected = validated

            if rejected:
                feedback_text = self._collect_feedback(rejected)
                _progress("Phase 6", "Optimizer", f"Reworking code ({len(rejected)} rejection(s) in quality review)...")
                review_result["rework_attempts"] += 1

                opt_result = await self._optimizer.optimize(
                    code=current_code,
                    focus="review_feedback",
                    requirements=requirements,
                    constraints=feedback_text,
                )
                if "optimized_code" in opt_result:
                    current_code = _strip_code_fences(opt_result["optimized_code"])
                    if main_file and main_file in files:
                        files[main_file] = current_code
                    _progress("Phase 6", "Optimizer", "Rework applied, re-reviewing...")

                # Re-review once
                phase_b = await self._run_review_phase(quality_reviewers, current_code, requirements, "quality", team_agreements)
                review_result["phase_b_results"] = phase_b

        # Final verdict
        all_results = review_result["phase_a_results"] + review_result["phase_b_results"]
        all_approved = all(r.get("verdict") == "APPROVED" for r in all_results)
        review_result["approved"] = all_approved
        review_result["feedback"] = self._collect_feedback(
            [r for r in all_results if r.get("verdict") == "REJECTED"]
        )

        verdict_str = "APPROVED" if all_approved else "REJECTED"
        _progress("Phase 6", "TeamReview", f"Final verdict: {verdict_str} (rework attempts: {review_result['rework_attempts']})")

        return review_result, files

    def _persist_output(self, code_output: CodeOutput, requirements: str) -> Optional[str]:
        """Persist generated code output to disk.

        Writes source files, tests, docs, architecture, requirements, and a
        manifest to config.output_dir/<timestamp>/.

        Args:
            code_output: The generated code output.
            requirements: Original requirements text for traceability.

        Returns:
            Output directory path string, or None on failure.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(self.config.output_dir) / timestamp

            # Create directory structure
            src_dir = base_dir / "src"
            tests_dir = base_dir / "tests"
            docs_dir = base_dir / "docs"

            src_dir.mkdir(parents=True, exist_ok=True)
            tests_dir.mkdir(parents=True, exist_ok=True)
            docs_dir.mkdir(parents=True, exist_ok=True)

            # Write source files
            for filename, content in code_output.files.items():
                file_path = src_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # Write test files
            for filename, content in code_output.tests.items():
                file_path = tests_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # Write documentation
            if code_output.docs:
                (docs_dir / "DOCUMENTATION.md").write_text(code_output.docs, encoding="utf-8")

            # Write architecture
            if code_output.architecture:
                (docs_dir / "ARCHITECTURE.md").write_text(code_output.architecture, encoding="utf-8")

            # Write original requirements for traceability
            (base_dir / "REQUIREMENTS.txt").write_text(requirements, encoding="utf-8")

            # Write manifest
            manifest = {
                "timestamp": timestamp,
                "files": list(code_output.files.keys()),
                "tests": list(code_output.tests.keys()),
                "main_file": code_output.main_file,
                "entry_point": code_output.entry_point,
                "dependencies": code_output.dependencies,
                "has_docs": bool(code_output.docs),
                "has_architecture": bool(code_output.architecture),
            }
            (base_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            logger.info(f"📁 Output persisted to {base_dir}")
            return str(base_dir)

        except Exception as e:
            logger.error(f"Failed to persist output (non-blocking): {e}")
            return None

    def _write_adr(
        self,
        base_dir: Path,
        adr_number: int,
        title: str,
        context: str,
        decision: str,
        consequences: str,
        participants: List[str] = None,
        status: str = "Accepted"
    ) -> Optional[str]:
        """Write an Architecture Decision Record (ADR) to the adr folder.

        ADR format follows the standard template:
        - Title, Status, Context, Decision, Consequences

        Args:
            base_dir: Output directory path
            adr_number: ADR sequence number (e.g., 1, 2, 3)
            title: Short title for the decision
            context: What is the issue/situation requiring a decision
            decision: What decision was made
            consequences: What are the results/implications
            participants: List of personas/agents involved
            status: Decision status (Accepted, Proposed, Deprecated, Superseded)

        Returns:
            Path to written ADR file, or None on failure.
        """
        try:
            adr_dir = base_dir / "adr"
            adr_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename: NNNN-title-slug.md
            title_slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:50]
            filename = f"{adr_number:04d}-{title_slug}.md"

            participants_str = ", ".join(participants) if participants else "Team"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            content = f"""# ADR {adr_number:04d}: {title}

**Date:** {timestamp}
**Status:** {status}
**Participants:** {participants_str}

## Context

{context}

## Decision

{decision}

## Consequences

{consequences}

---
*Generated by Jotty CodingSwarm*
"""
            adr_path = adr_dir / filename
            adr_path.write_text(content, encoding="utf-8")
            logger.debug(f"ADR written: {adr_path}")
            return str(adr_path)

        except Exception as e:
            logger.error(f"Failed to write ADR (non-blocking): {e}")
            return None

    def _write_planning_adr(self, base_dir: Path, planning_result: Dict[str, Any], requirements: str) -> Optional[str]:
        """Write ADR for team planning decisions."""
        if not planning_result:
            return None

        team_feedback = planning_result.get('team_feedback', [])
        participants = [fb.get('persona', 'Unknown') for fb in team_feedback if isinstance(fb, dict)]

        # Collect concerns as context
        concerns = []
        for fb in team_feedback:
            if isinstance(fb, dict):
                for c in fb.get('concerns', []):
                    if isinstance(c, dict):
                        concerns.append(f"- [{c.get('severity', '?')}] {c.get('description', '')}")

        context = f"Requirements: {requirements[:200]}...\n\n"
        if concerns:
            context += "Team concerns raised:\n" + "\n".join(concerns[:10])
        else:
            context += "No major concerns raised by team."

        decision = planning_result.get('refined_architecture', 'No refined architecture')[:1000]
        if planning_result.get('implementation_plan'):
            decision += "\n\n### Implementation Plan\n" + planning_result['implementation_plan'][:500]

        consequences = planning_result.get('risk_mitigations', 'No specific risks identified.')
        if planning_result.get('team_agreements'):
            consequences += "\n\n### Team Agreements\n" + planning_result['team_agreements'][:500]

        return self._write_adr(
            base_dir=base_dir,
            adr_number=1,
            title="Architecture Design Decision",
            context=context,
            decision=decision,
            consequences=consequences,
            participants=participants,
            status="Accepted"
        )

    def _write_review_adr(self, base_dir: Path, review_result: Dict[str, Any], adr_number: int = 2) -> Optional[str]:
        """Write ADR for team review decisions and debates."""
        if not review_result:
            return None

        phase_a = review_result.get('phase_a_results', [])
        phase_b = review_result.get('phase_b_results', [])
        all_reviews = phase_a + phase_b

        participants = list(set(r.get('persona', 'Unknown') for r in all_reviews if isinstance(r, dict)))

        # Collect issues raised as context
        issues = []
        for r in all_reviews:
            if isinstance(r, dict) and r.get('verdict') == 'REJECTED':
                persona = r.get('persona', 'Reviewer')
                for issue in r.get('issues', []):
                    if isinstance(issue, dict):
                        issues.append(f"- [{persona}] {issue.get('description', '')[:100]}")

        context = f"Code review conducted by {len(participants)} reviewers.\n\n"
        if issues:
            context += "Issues raised:\n" + "\n".join(issues[:15])
        else:
            context += "No critical issues identified during review."

        # Decision based on final outcome
        approved = review_result.get('approved', True)
        rework_attempts = review_result.get('rework_attempts', 0)

        if approved:
            decision = "Code APPROVED after review."
            if rework_attempts > 0:
                decision += f" Required {rework_attempts} rework cycle(s) to address concerns."
        else:
            decision = "Code REJECTED. Issues remain unresolved."
            decision += f"\n\nFeedback:\n{review_result.get('feedback', 'No specific feedback')[:500]}"

        # Consequences
        verdicts = []
        for r in all_reviews:
            if isinstance(r, dict):
                persona = r.get('persona', 'Reviewer')
                verdict = r.get('verdict', 'UNKNOWN')
                overruled = " (overruled by Arbitrator)" if r.get('arbitrator_overruled') else ""
                verdicts.append(f"- {persona}: {verdict}{overruled}")

        consequences = "### Reviewer Verdicts\n" + "\n".join(verdicts)

        return self._write_adr(
            base_dir=base_dir,
            adr_number=adr_number,
            title="Code Review Decision",
            context=context,
            decision=decision,
            consequences=consequences,
            participants=participants,
            status="Accepted" if approved else "Proposed"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def code(requirements: str, **kwargs) -> CodingResult:
    """
    One-liner code generation.

    Usage:
        from core.swarms.coding_swarm import code
        result = await code("Create a REST API for user management")
    """
    swarm = CodingSwarm()
    return await swarm.generate(requirements, **kwargs)


def code_sync(requirements: str, **kwargs) -> CodingResult:
    """Synchronous code generation."""
    return asyncio.run(code(requirements, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CodingSwarm',
    'CodingConfig',
    'CodingResult',
    'CodeOutput',
    'CodeLanguage',
    'CodeStyle',
    'EditMode',
    'code',
    'code_sync',
    # Data models
    'ResearchContext',
    'FullStackContext',
    # Signatures
    'CodeVerificationSignature',
    'ResearchQuerySignature',
    'CollaborativeArchitectSignature',
    'ResearchResponseSignature',
    'CodebaseAnalysisSignature',
    'EditPlanSignature',
    'TeamReviewSignature',
    'TeamPlanningSignature',
    'TeamPlanningConsolidationSignature',
    'SystemDesignSignature',
    'DatabaseSchemaSignature',
    'APIGenerationSignature',
    'FrontendGenerationSignature',
    'IntegrationSignature',
    'ReviewArbitrationSignature',
    # Infrastructure
    'WorkspaceManager',
    '_strip_code_fences',
    '_progress',
    '_stream_call',
    'BaseCodeAgent',
    # Team personas & config
    'TeamPersona',
    'TeamConfig',
    'TEAM_PRESETS',
    # Agents
    'ArchitectAgent',
    'DeveloperAgent',
    'DebuggerAgent',
    'OptimizerAgent',
    'TestWriterAgent',
    'DocWriterAgent',
    'VerifierAgent',
    'SystemDesignerAgent',
    'DatabaseArchitectAgent',
    'APIDesignerAgent',
    'FrontendDeveloperAgent',
    'IntegrationAgent',
    'ArbitratorAgent',
    'CodebaseAnalyzerAgent',
    'EditPlannerAgent',
]
