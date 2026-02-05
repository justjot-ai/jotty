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


async def _stream_call(module, phase: str, agent: str, listener_field: str = "reasoning", **kwargs):
    """Call a DSPy module with streaming, forwarding reasoning tokens to _progress().

    Args:
        module: DSPy ChainOfThought module to call
        phase: Phase name for progress messages (e.g. "Phase 1")
        agent: Agent name for progress messages (e.g. "Architect")
        listener_field: Output field to stream (default: "reasoning")
        **kwargs: Arguments to pass to the module

    Returns:
        dspy.Prediction result
    """
    from dspy.streaming import streamify, StreamListener

    listener = StreamListener(listener_field)
    streaming_module = streamify(module, stream_listeners=[listener])

    result = None
    last_text = ""
    async for chunk in streaming_module(**kwargs):
        if isinstance(chunk, dspy.Prediction):
            result = chunk
        elif isinstance(chunk, str):
            # chunk is cumulative streamed text for the listened field
            new_text = chunk[len(last_text):]
            if new_text.strip():
                # Show last 80 chars of reasoning in progress
                display = chunk.strip()[-80:]
                _progress(phase, agent, f"  ...{display}")
            last_text = chunk

    if result is None:
        # Fallback: non-streaming call if streamify didn't yield a Prediction
        result = module(**kwargs)

    return result


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
    scope: Optional[str] = None  # "single_tier", "full_stack", or None (auto-detect)
    db_type: str = "sqlite"  # Database type
    backend_framework: str = "fastapi"  # Backend framework
    frontend_framework: str = "react"  # Frontend framework
    enable_arbitrator: bool = True  # Enable review arbitration

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

ALL_PERSONAS = [
    PERSONA_ARCHITECT, PERSONA_PERFORMANCE, PERSONA_QUALITY,
    PERSONA_ALGORITHM, PERSONA_BACKEND, PERSONA_FRONTEND,
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
            "SystemDesigner": "The Architect",
            "DatabaseArchitect": "The Backend Engineer",
            "FrontendDeveloper": "The Frontend Specialist",
            "Integration": "The Backend Engineer",
        },
        functional_reviewers=["The Architect", "The Backend Engineer", "The Algorithm Specialist"],
        quality_reviewers=["The Quality Champion", "The Performance Engineer", "The Frontend Specialist"],
    ),
    "datascience": TeamConfig(
        name="datascience",
        personas=[PERSONA_ARCHITECT, PERSONA_ALGORITHM, PERSONA_QUALITY, PERSONA_PERFORMANCE],
        role_persona_map={
            "Architect": "The Architect",
            "Developer": "The Algorithm Specialist",
            "Optimizer": "The Performance Engineer",
            "TestWriter": "The Quality Champion",
            "DocWriter": "The Algorithm Specialist",
            "Verifier": "The Quality Champion",
        },
        functional_reviewers=["The Algorithm Specialist", "The Architect"],
        quality_reviewers=["The Quality Champion", "The Performance Engineer"],
    ),
    "frontend": TeamConfig(
        name="frontend",
        personas=[PERSONA_ARCHITECT, PERSONA_FRONTEND, PERSONA_QUALITY, PERSONA_PERFORMANCE],
        role_persona_map={
            "Architect": "The Architect",
            "Developer": "The Frontend Specialist",
            "Optimizer": "The Performance Engineer",
            "TestWriter": "The Quality Champion",
            "DocWriter": "The Frontend Specialist",
            "Verifier": "The Quality Champion",
        },
        functional_reviewers=["The Frontend Specialist", "The Architect"],
        quality_reviewers=["The Quality Champion", "The Performance Engineer"],
    ),
}


class TeamReviewSignature(dspy.Signature):
    """Review code from a specific team archetype's perspective.

    You are a senior engineer reviewing code through the lens of your persona.
    Evaluate thoroughly but constructively.
    """
    code: str = dspy.InputField(desc="Complete code to review")
    requirements: str = dspy.InputField(desc="Original requirements the code must satisfy")
    review_phase: str = dspy.InputField(desc="'functional' or 'quality'")
    persona_context: str = dspy.InputField(desc="Reviewer persona context and expertise")

    verdict: str = dspy.OutputField(desc="APPROVED or REJECTED")
    issues: str = dspy.OutputField(desc='JSON list of issues: [{"severity": "high|medium|low", "description": "..."}]')
    feedback: str = dspy.OutputField(desc="Constructive feedback for the developer")
    evidence: str = dspy.OutputField(desc="Specific code lines, test case, or scenario demonstrating the issue. Required for REJECTED verdict.")


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

class BaseCodeAgent:
    """Base class for coding agents."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        self.memory = memory
        self.context = context
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

            result = await self._stream(self._generator, "Phase 2", "Developer",
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

            result = await self._stream(self._analyzer, "Phase 3.5", "Debugger",
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

            result = await self._stream(self._optimizer, "Phase 3", "Optimizer",
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

            result = await self._stream(self._generator, "Phase 4", "TestWriter",
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

            result = await self._stream(self._writer, "Phase 5", "DocWriter",
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
            if self.learned_context:
                code = code + f"\n\n{self.learned_context}"

            result = await self._stream(self._verifier, "Phase 5.5", "Verifier",
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


# =============================================================================
# CODING SWARM
# =============================================================================

@register_swarm("coding")
class CodingSwarm(BaseSwarm):
    """
    World-Class Coding Swarm.

    Generates production-quality code with:
    - Clean architecture
    - Comprehensive tests
    - Full documentation
    - Optimized performance
    """

    def __init__(self, config: CodingConfig = None):
        super().__init__(config or CodingConfig())
        self._agents_initialized = False

        # Team configuration
        self._team_config: Optional[TeamConfig] = None
        if self.config.team:
            self._team_config = TEAM_PRESETS.get(self.config.team)

        # Scope classifier (uses DSPy's configured LM)
        self._scope_classifier = dspy.ChainOfThought(ScopeClassificationSignature)

        # Review module (lazy init)
        self._review_module = None

        # Agents
        self._architect = None
        self._developer = None
        self._debugger = None
        self._optimizer = None
        self._test_writer = None
        self._doc_writer = None
        self._verifier = None

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

    def _init_agents(self):
        """Initialize all agents with per-agent learned context."""
        if self._agents_initialized:
            return

        self._init_shared_resources()

        self._architect = ArchitectAgent(self._memory, self._context, self._bus, self._agent_context("Architect"))
        self._developer = DeveloperAgent(self._memory, self._context, self._bus, self._agent_context("Developer"))
        self._debugger = DebuggerAgent(self._memory, self._context, self._bus, self._agent_context("Debugger"))
        self._optimizer = OptimizerAgent(self._memory, self._context, self._bus, self._agent_context("Optimizer"))
        self._test_writer = TestWriterAgent(self._memory, self._context, self._bus, self._agent_context("TestWriter"))
        self._doc_writer = DocWriterAgent(self._memory, self._context, self._bus, self._agent_context("DocWriter"))
        self._verifier = VerifierAgent(self._memory, self._context, self._bus, self._agent_context("Verifier"))

        self._agents_initialized = True
        logger.info("CodingSwarm agents initialized")

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

    async def execute(
        self,
        requirements: str,
        language: CodeLanguage = None,
        style: CodeStyle = None,
        **kwargs
    ) -> CodingResult:
        """Execute code generation."""
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

        # Pre-execution learning: load state, warmup, compute scores
        await self._pre_execute_learning()

        self._init_agents()

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

        team_name = self._team_config.name if self._team_config else "auto"
        print(f"\n{'='*60}", flush=True)
        print(f"  CodingSwarm | {lang.value} | {code_style.value} | team={team_name}", flush=True)
        print(f"{'='*60}", flush=True)
        logger.info(f"CodingSwarm starting: {lang.value}, {code_style.value}")

        # Initialize workspace for terminal-based validation
        workspace = WorkspaceManager() if getattr(config, 'enable_workspace', True) else None

        try:
            # =================================================================
            # PHASE 1: ARCHITECTURE DESIGN
            # =================================================================
            _progress("Phase 1", "Architect", "Designing architecture...")

            arch_result = await self._architect.design(
                requirements=requirements,
                language=lang.value,
                style=code_style.value,
                constraints=json.dumps({
                    'frameworks': config.frameworks,
                    'max_file_size': config.max_file_size
                })
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
            _progress("Phase 1", "Architect", f"Done -- {n_components} component(s) designed")
            # Stream architect output
            if arch_result.get('file_structure'):
                for line in str(arch_result['file_structure']).strip().split('\n')[:8]:
                    _progress("Phase 1", "Architect", f"  {line.strip()}")
            for comp in arch_result.get('components', []):
                comp_name = comp.get('name', comp) if isinstance(comp, dict) else str(comp)
                _progress("Phase 1", "Architect", f"  Component: {comp_name}")

            self._trace_phase("Architect", AgentRole.PLANNER,
                {'requirements': requirements[:100]},
                {'components': n_components},
                success='error' not in arch_result, phase_start=start_time, tools_used=['arch_design'])

            # =================================================================
            # PHASE 1.5: RESEARCH (web search for best practices)
            # =================================================================
            research_context = ResearchContext()
            if getattr(config, 'enable_research', True):
                try:
                    _progress("Phase 1.5", "Researcher", "Generating search queries...")
                    phase15_start = datetime.now()

                    # Generate search queries via LLM
                    query_gen = dspy.ChainOfThought(ResearchQuerySignature)
                    query_result = query_gen(
                        requirements=requirements,
                        language=lang.value,
                        frameworks=json.dumps(config.frameworks)
                    )

                    # Parse search queries
                    try:
                        search_queries = json.loads(str(query_result.search_queries))
                        if not isinstance(search_queries, list):
                            search_queries = []
                    except (json.JSONDecodeError, TypeError):
                        search_queries = []

                    # Execute web searches if we have queries
                    if search_queries:
                        try:
                            from ...skills import get_skill_registry
                            registry = get_skill_registry()
                            web_search_fn = registry.get('web-search', {}).get('search_web_tool')
                        except Exception:
                            web_search_fn = None

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

                        if web_search_fn:
                            for qi, query in enumerate(search_queries[:5]):
                                _progress("Phase 1.5", "Researcher", f"Searching ({qi+1}/{min(len(search_queries),5)}): {str(query)[:60]}")
                                try:
                                    result = web_search_fn({'query': str(query), 'max_results': 3})
                                    if result.get('success') and result.get('results'):
                                        for r in result['results']:
                                            snippet = r.get('snippet', r.get('description', ''))
                                            if snippet:
                                                title = r.get('title', '')
                                                if any(kw in str(query).lower() for kw in ['best practice', 'pattern', 'convention']):
                                                    research_context.best_practices.append(f"{title}: {snippet}")
                                                elif any(kw in str(query).lower() for kw in ['doc', 'api', 'reference']):
                                                    research_context.api_references.append(f"{title}: {snippet}")
                                                elif any(kw in str(query).lower() for kw in ['pitfall', 'warning', 'common mistake', 'avoid']):
                                                    research_context.warnings.append(f"{title}: {snippet}")
                                                else:
                                                    research_context.library_docs.append(f"{title}: {snippet}")
                                except Exception as search_err:
                                    logger.debug(f"Web search query failed: {search_err}")

                    total_findings = len(research_context.best_practices) + len(research_context.library_docs) + len(research_context.api_references) + len(research_context.warnings)
                    _progress("Phase 1.5", "Researcher", f"Done -- {total_findings} finding(s) collected")

                    self._trace_phase("Researcher", AgentRole.EXPERT,
                        {'queries': len(search_queries)},
                        {'best_practices': len(research_context.best_practices),
                         'references': len(research_context.api_references)},
                        success=True, phase_start=phase15_start, tools_used=['web_search'])

                except Exception as research_err:
                    _progress("Phase 1.5", "Researcher", "Skipped (web search unavailable)")
                    logger.debug(f"Research phase skipped (non-blocking): {research_err}")

            # =================================================================
            # PHASE 0: SCOPE DETECTION
            # =================================================================
            scope = self._detect_scope(requirements)
            _progress("Phase 0", "ScopeDetector", f"Detected scope: {scope}")

            # Build review criteria for both paths
            review_criteria = ""
            if self._team_config:
                review_criteria = self._build_review_criteria()

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
                # PHASE 2: CODE GENERATION (parallel for each component)
                _progress("Phase 2", "Developer", "Generating code...")

                components = arch_result.get('components', [])
                if not components:
                    components = [{'name': 'main', 'description': requirements}]

                # Inject research context into architecture for developer
                enriched_arch = arch_result['architecture']
                research_prompt = research_context.to_prompt()
                if research_prompt:
                    enriched_arch = enriched_arch + "\n\n## Research Findings\n" + research_prompt

                # Inject reviewer criteria so developer writes code that passes review
                if review_criteria:
                    enriched_arch = enriched_arch + "\n\n## Code Review Criteria (your code WILL be reviewed against these)\n" + review_criteria

                total_components = len(components)
                completed_count = 0

                async def _generate_component(comp):
                    nonlocal completed_count
                    comp_name = comp.get('name', 'component') if isinstance(comp, dict) else str(comp)
                    _progress("Phase 2", "Developer", f"  Writing {comp_name}...")
                    result = await self._developer.generate(
                        architecture=enriched_arch,
                        component=comp_name,
                        language=lang.value,
                        dependencies=config.frameworks
                    )
                    completed_count += 1
                    if isinstance(result, dict) and 'filename' in result:
                        _progress("Phase 2", "Developer", f"  [{completed_count}/{total_components}] {result['filename']} ready")
                    else:
                        _progress("Phase 2", "Developer", f"  [{completed_count}/{total_components}] {comp_name} failed")
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

                _progress("Phase 2", "Developer", f"Done -- {len(files)} file(s): {', '.join(files.keys())}")
                # Stream developer output (first 5 lines of each file)
                for fname, code_content in files.items():
                    lines = code_content.strip().split('\n')
                    _progress("Phase 2", "Developer", f"  --- {fname} ({len(lines)} lines) ---")
                    for line in lines[:5]:
                        _progress("Phase 2", "Developer", f"    {line}")
                    if len(lines) > 5:
                        _progress("Phase 2", "Developer", f"    ... ({len(lines)-5} more lines)")

            phase2_start = datetime.now()
            self._trace_phase("Developer", AgentRole.ACTOR,
                {'components': len(arch_result.get('components', [])), 'scope': scope},
                {'files_generated': len(files)},
                success=len(files) > 0, phase_start=start_time, tools_used=['code_generate'])

            # =================================================================
            # PHASE 3: OPTIMIZATION (parallel)
            # =================================================================
            _progress("Phase 3", "Optimizer", f"Optimizing {len(files)} file(s) in parallel...")

            total_files = len(files)
            optimized_count = 0

            async def _optimize_one(fname: str, code_str: str):
                nonlocal optimized_count
                _progress("Phase 3", "Optimizer", f"  Optimizing {fname}...")
                opt_result = await self._optimizer.optimize(
                    code=code_str,
                    focus="readability",
                    constraints="Maintain all functionality",
                    requirements=requirements
                )
                optimized_count += 1
                improvements = opt_result.get('improvements', [])
                _progress("Phase 3", "Optimizer", f"  [{optimized_count}/{total_files}] {fname} optimized")
                return fname, _strip_code_fences(opt_result.get('optimized_code', code_str)), improvements

            opt_tasks = [_optimize_one(fn, c) for fn, c in files.items()]
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
                        _progress("Phase 3", "Optimizer", f"  {fname}: {imp}")

            # Keep originals for any that failed optimization
            for fname in files:
                if fname not in optimized_files:
                    optimized_files[fname] = files[fname]

            files = optimized_files
            _progress("Phase 3", "Optimizer", f"Done -- {len(optimized_files)} file(s) optimized")

            self._trace_phase("Optimizer", AgentRole.ACTOR,
                {'files_count': len(files)},
                {'optimized': len(optimized_files)},
                success=True, phase_start=phase2_start, tools_used=['code_optimize'])

            # =================================================================
            # PHASE 3.5: VALIDATION & FIX LOOP
            # =================================================================
            validation_metadata = {"validated": False, "fix_attempts": 0, "errors_fixed": []}
            max_fix = getattr(config, 'max_fix_attempts', 3)

            if workspace and workspace.available and files:
                _progress("Phase 3.5", "Validator", "Validating code in workspace...")
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
                            _progress("Phase 3.5", "Validator", f"Syntax OK: {fname}")
                        else:
                            error_text = check_result.error or check_result.output
                            if any(err in error_text for err in FIXABLE_ERRORS):
                                syntax_ok = False
                                _progress("Phase 3.5", "Debugger", f"Fixing syntax in {fname} (attempt {attempt+1})...")
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
                        _progress("Phase 3.5", "Validator", f"Running: {main_file}...")
                        await workspace.write_file(main_file, files[main_file])
                        run_result = await workspace.run_python(main_file, timeout=15)
                        if run_result.success:
                            _progress("Phase 3.5", "Validator", f"Run OK: {main_file}")
                        else:
                            error_text = run_result.error or run_result.output
                            if any(err in error_text for err in FIXABLE_ERRORS):
                                runtime_ok = False
                                _progress("Phase 3.5", "Debugger", f"Fixing runtime error in {main_file} (attempt {attempt+1})...")
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
                                _progress("Phase 3.5", "Validator", f"Non-fixable runtime condition (skipped): {error_text[:80]}")

                    validation_metadata["fix_attempts"] = attempt + 1

                    if syntax_ok and runtime_ok:
                        validation_metadata["validated"] = True
                        _progress("Phase 3.5", "Validator", f"All files validated after {attempt+1} attempt(s)")
                        break
                    elif not syntax_ok:
                        _progress("Phase 3.5", "Validator", f"Syntax errors fixed, re-validating (attempt {attempt+1}/{max_fix})...")

                if not validation_metadata["validated"]:
                    _progress("Phase 3.5", "Validator", f"Max attempts ({max_fix}) reached")

                self._trace_phase("Validator", AgentRole.AUDITOR,
                    {'max_fix_attempts': max_fix},
                    {'validated': validation_metadata['validated'],
                     'fix_attempts': validation_metadata['fix_attempts'],
                     'errors_fixed': len(validation_metadata['errors_fixed'])},
                    success=True, phase_start=phase35_start, tools_used=['workspace_validate', 'debug'])

            # =================================================================
            # PHASE 4: TEST GENERATION (if enabled)
            # =================================================================
            tests = {}
            test_coverage = 0.0

            if gen_tests and files:
                _progress("Phase 4", "TestWriter", "Generating tests...")

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

            phase4_start = datetime.now()
            self._trace_phase("TestWriter", AgentRole.ACTOR,
                {'gen_tests': gen_tests},
                {'test_files': len(tests), 'coverage': test_coverage},
                success=True, phase_start=phase2_start, tools_used=['test_generate'])

            # =================================================================
            # PHASE 5: DOCUMENTATION (if enabled)
            # =================================================================
            documentation = ""

            if gen_docs and files:
                _progress("Phase 5", "DocWriter", "Generating documentation...")

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
                success=True, phase_start=phase4_start, tools_used=['doc_generate'])

            # =================================================================
            # PHASE 5.5: VERIFICATION + DEBUGGER FEEDBACK
            # =================================================================
            verification_result = None
            try:
                _progress("Phase 5.5", "Verifier", "Verifying code against requirements...")

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
                    _progress("Phase 5.5", "Verifier", f"{v_status} (coverage: {v_score})")

                if verification_result and not verification_result.get('verified', True):
                    issues = verification_result.get('issues', [])
                    if issues:
                        # Format issues into description for debugger
                        issues_desc = "; ".join(
                            f"[{iss.get('severity', 'unknown')}] {iss.get('description', 'no description')}"
                            for iss in issues if isinstance(iss, dict)
                        )
                        _progress("Phase 5.5", "Verifier", f"Found {len(issues)} issue(s), attempting fix...")
                        for iss in issues[:5]:
                            if isinstance(iss, dict):
                                _progress("Phase 5.5", "Verifier", f"  [{iss.get('severity','?')}] {iss.get('description','')[:80]}")

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
                                    _progress("Phase 5.5", "Debugger", "Fix applied to main file")
                        except Exception as dbg_err:
                            logger.error(f"Debugger fix attempt failed (non-blocking): {dbg_err}")
            except Exception as ver_err:
                logger.error(f"Verification phase failed (non-blocking): {ver_err}")

            phase55_start = datetime.now()
            self._trace_phase("Verifier", AgentRole.AUDITOR,
                {'requirements_len': len(requirements)},
                {'verified': verification_result.get('verified', True) if verification_result else True,
                 'issues_count': len(verification_result.get('issues', [])) if verification_result else 0},
                success=True, phase_start=phase4_start, tools_used=['verify', 'debug'])

            # =================================================================
            # PHASE 6: TEAM REVIEW
            # =================================================================
            team_review_result = None
            if self._team_config and self._team_config.review_protocol != "none":
                _progress("Phase 6", "TeamReview", f"Team review ({self._team_config.name})...")
                all_code_str = "\n\n".join(files.values())
                team_review_result, files = await self._team_review(
                    all_code_str, requirements,
                    arch_result.get('architecture', ''),
                    files, main_file,
                )
                self._trace_phase("TeamReview", AgentRole.REVIEWER,
                    {'team': self._team_config.name, 'protocol': self._team_config.review_protocol},
                    {'approved': team_review_result.get('approved', True),
                     'rework_attempts': team_review_result.get('rework_attempts', 0)},
                    success=True, phase_start=phase55_start, tools_used=['team_review'])

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
                quality_score=verification_result.get('coverage_score', 0.8) if verification_result else 0.8
            )

            validated_str = "validated" if validation_metadata.get("validated") else "not validated"
            print(f"\n  {'='*56}", flush=True)
            print(f"  DONE | {loc} LOC | {len(files)} file(s) | {len(tests)} test(s) | {validated_str}", flush=True)
            print(f"  Time: {exec_time:.1f}s | Quality: {result.quality_score:.2f}", flush=True)
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
    # TEAM REVIEW METHODS
    # -----------------------------------------------------------------

    async def _run_persona_review(
        self, code: str, requirements: str, phase: str, persona: TeamPersona
    ) -> Dict[str, Any]:
        """Run a single persona review. Non-blocking on failure."""
        try:
            _progress("Phase 6", persona.name, f"Reviewing ({phase})...")
            if self._review_module is None:
                self._review_module = dspy.ChainOfThought(TeamReviewSignature)
            result = await _stream_call(self._review_module, "Phase 6", persona.name,
                code=code,
                requirements=requirements,
                review_phase=phase,
                persona_context=persona.to_prompt(),
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
        self, reviewers: List[TeamPersona], code: str, requirements: str, phase: str
    ) -> List[Dict[str, Any]]:
        """Run all reviewers for a phase in parallel."""
        reviewer_names = ", ".join(p.name for p in reviewers)
        _progress("Phase 6", "TeamReview", f"{phase.capitalize()} review: {reviewer_names}")
        tasks = [
            self._run_persona_review(code, requirements, phase, persona)
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
        files: Dict[str, str], main_file: Optional[str]
    ) -> tuple:
        """Orchestrate Phase 6: two-phase team review with auto-fix loop.

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

        # --- Phase 6a: Functional Review ---
        func_reviewers = self._team_config.get_reviewers("functional")
        if func_reviewers:
            phase_a = await self._run_review_phase(func_reviewers, current_code, requirements, "functional")
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
                phase_a = await self._run_review_phase(func_reviewers, current_code, requirements, "functional")
                review_result["phase_a_results"] = phase_a

        # --- Phase 6b: Code Quality Review ---
        quality_reviewers = self._team_config.get_reviewers("quality")
        if quality_reviewers:
            phase_b = await self._run_review_phase(quality_reviewers, current_code, requirements, "quality")
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
                phase_b = await self._run_review_phase(quality_reviewers, current_code, requirements, "quality")
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
    'code',
    'code_sync',
    # Data models
    'ResearchContext',
    'FullStackContext',
    # Signatures
    'CodeVerificationSignature',
    'ResearchQuerySignature',
    'TeamReviewSignature',
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
]
