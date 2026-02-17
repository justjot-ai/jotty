"""
Coding Swarm - DSPy Signatures
================================

All DSPy Signature definitions used by CodingSwarm agents.
"""

import dspy

# =============================================================================
# CORE SIGNATURES
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
    requirements: str = dspy.InputField(
        desc="Original requirements the code must satisfy; do not optimize away intent"
    )
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


# =============================================================================
# EDIT MODE SIGNATURES
# =============================================================================


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
    style_conventions: str = dspy.OutputField(
        desc="Code style conventions observed (naming, formatting, patterns)"
    )
    affected_files: str = dspy.OutputField(desc="JSON list of files that need modification")
    dependencies: str = dspy.OutputField(desc="Key dependencies and interfaces to preserve")
    change_scope: str = dspy.OutputField(
        desc="Scope of changes: 'minimal', 'moderate', 'extensive'"
    )


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
    edits: str = dspy.OutputField(
        desc='JSON list of edits: [{"old": "...", "new": "...", "reason": "..."}]'
    )
    new_code: str = dspy.OutputField(desc="Complete new code (only if substantial rewrite needed)")
    edit_type: str = dspy.OutputField(
        desc="'patch' for surgical edits, 'rewrite' for full replacement"
    )


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
    previous_attempts: str = dspy.InputField(
        desc="Summary of previous fix attempts and why they failed"
    )

    analysis: str = dspy.OutputField(desc="Analysis of test failure: what's wrong and why")
    fix_strategy: str = dspy.OutputField(desc="Strategy to fix the issue")
    fixed_code: str = dspy.OutputField(desc="Complete fixed code for the file")
    confidence: str = dspy.OutputField(desc="HIGH, MEDIUM, or LOW confidence this fix will work")


# =============================================================================
# COLLABORATIVE SIGNATURES
# =============================================================================


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
    previous_architecture: str = dspy.InputField(
        desc="Architecture from previous iteration (empty for iteration 1)"
    )
    research_findings: str = dspy.InputField(
        desc="Research findings from previous iteration (empty for iteration 1)"
    )

    architecture: str = dspy.OutputField(
        desc="Detailed architecture design (refined with research findings)"
    )
    components: str = dspy.OutputField(desc="JSON list of components with responsibilities")
    file_structure: str = dspy.OutputField(desc="File/folder structure")
    interfaces: str = dspy.OutputField(desc="Key interfaces and contracts")
    research_requests: str = dspy.OutputField(
        desc="Specific topics for Researcher to investigate in next iteration"
    )


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

    search_queries: str = dspy.OutputField(
        desc="JSON list of 3-5 targeted search queries based on Architect's requests"
    )
    analysis: str = dspy.OutputField(
        desc="Analysis of the architecture with research-backed recommendations"
    )
    best_practices: str = dspy.OutputField(desc="Key best practices relevant to the architecture")
    warnings: str = dspy.OutputField(desc="Potential pitfalls or warnings to consider")
    recommendations: str = dspy.OutputField(
        desc="Specific recommendations to improve the architecture"
    )


# =============================================================================
# FULL-STACK SIGNATURES
# =============================================================================


class SystemDesignSignature(dspy.Signature):
    """Design a full-stack system with data model, API contract, and component boundaries."""

    requirements: str = dspy.InputField(desc="What the system should do")
    language: str = dspy.InputField(desc="Primary backend language")
    tech_stack: str = dspy.InputField(desc="JSON: {db_type, backend_framework, frontend_framework}")

    data_model: str = dspy.OutputField(
        desc="Entity definitions: fields, types, relationships, constraints"
    )
    api_contract: str = dspy.OutputField(
        desc="REST endpoints: method, path, request body, response schema"
    )
    component_map: str = dspy.OutputField(
        desc="Components: database, backend, frontend with responsibilities"
    )
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

    api_code: str = dspy.OutputField(
        desc="Complete backend API code with routes, services, error handling"
    )
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


# =============================================================================
# REVIEW / VERIFICATION SIGNATURES
# =============================================================================


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
    coverage_score: float = dspy.OutputField(
        desc="Float 0-1: fraction of requirements covered by the code"
    )
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
# TEAM SIGNATURES
# =============================================================================


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
    team_agreements: str = dspy.InputField(
        desc="Decisions and agreements made during team planning phase - DO NOT contradict these"
    )

    verdict: str = dspy.OutputField(
        desc="APPROVED or REJECTED - only reject for bugs, security issues, or clear violations NOT covered in team agreements"
    )
    issues: str = dspy.OutputField(
        desc='JSON list of issues: [{"severity": "high|medium|low", "description": "..."}] - only include issues NOT already addressed in team agreements'
    )
    feedback: str = dspy.OutputField(desc="Constructive feedback for the developer")
    evidence: str = dspy.OutputField(
        desc="Specific code lines, test case, or scenario demonstrating the issue. Required for REJECTED verdict."
    )


class TeamPlanningSignature(dspy.Signature):
    """Provide planning input from a specific team archetype's perspective.

    You are a senior engineer contributing to architecture planning through your expertise lens.
    Review the proposed architecture and research findings, then provide your recommendations.
    """

    requirements: str = dspy.InputField(desc="Original requirements for the software")
    architecture: str = dspy.InputField(desc="Proposed architecture from the Architect")
    research_findings: str = dspy.InputField(
        desc="Research findings: best practices, API docs, warnings"
    )
    persona_context: str = dspy.InputField(desc="Your persona context and expertise area")

    concerns: str = dspy.OutputField(
        desc='JSON list of concerns: [{"severity": "high|medium|low", "area": "...", "description": "..."}]'
    )
    recommendations: str = dspy.OutputField(
        desc="Specific recommendations to improve the architecture from your expertise"
    )
    implementation_notes: str = dspy.OutputField(
        desc="Key implementation details the developer should know from your area of expertise"
    )


class TeamPlanningConsolidationSignature(dspy.Signature):
    """Consolidate team planning feedback into a refined architecture plan.

    You are a lead architect synthesizing feedback from multiple team members into
    a cohesive, actionable implementation plan.
    """

    original_architecture: str = dspy.InputField(desc="Original architecture proposal")
    team_feedback: str = dspy.InputField(desc="Consolidated feedback from all team members")
    research_findings: str = dspy.InputField(desc="Research findings that informed the planning")
    requirements: str = dspy.InputField(desc="Original requirements")

    refined_architecture: str = dspy.OutputField(
        desc="Refined architecture incorporating team feedback"
    )
    implementation_plan: str = dspy.OutputField(
        desc="Step-by-step implementation plan with priorities"
    )
    risk_mitigations: str = dspy.OutputField(desc="Identified risks and how to mitigate them")
    team_agreements: str = dspy.OutputField(
        desc="Key decisions and agreements from team discussion"
    )
