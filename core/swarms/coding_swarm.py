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

    OPTIMIZATION TARGETS:
    1. Time complexity
    2. Space complexity
    3. Readability
    4. Maintainability
    5. Resource usage
    """
    code: str = dspy.InputField(desc="Code to optimize")
    focus: str = dspy.InputField(desc="Optimization focus: performance, readability, memory")
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

            result = self._designer(
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

            result = self._generator(
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

            result = self._analyzer(
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
        constraints: str = ""
    ) -> Dict[str, Any]:
        """Optimize code."""
        try:
            if self.learned_context:
                code = code + f"\n\n{self.learned_context}"

            result = self._optimizer(
                code=code,
                focus=focus,
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

            result = self._generator(
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

            result = self._writer(
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

        # Agents
        self._architect = None
        self._developer = None
        self._debugger = None
        self._optimizer = None
        self._test_writer = None
        self._doc_writer = None

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

        self._agents_initialized = True
        logger.info("CodingSwarm agents initialized")

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
        include_docs: bool = None
    ) -> CodingResult:
        """
        Generate complete code from requirements.

        Args:
            requirements: What to build
            language: Target language
            style: Coding style
            include_tests: Generate tests
            include_docs: Generate documentation

        Returns:
            CodingResult with generated code
        """
        start_time = datetime.now()

        # Pre-execution learning: load state, warmup, compute scores
        await self._pre_execute_learning()

        self._init_agents()

        config = self.config
        lang = language or config.language
        code_style = style or config.style
        gen_tests = include_tests if include_tests is not None else config.include_tests
        gen_docs = include_docs if include_docs is not None else config.include_docs

        logger.info(f"🚀 CodingSwarm starting: {lang.value}, {code_style.value}")

        try:
            # =================================================================
            # PHASE 1: ARCHITECTURE DESIGN
            # =================================================================
            logger.info("📐 Phase 1: Designing architecture...")

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

            self._trace_phase("Architect", AgentRole.PLANNER,
                {'requirements': requirements[:100]},
                {'components': len(arch_result.get('components', []))},
                success='error' not in arch_result, phase_start=start_time, tools_used=['arch_design'])

            # =================================================================
            # PHASE 2: CODE GENERATION (parallel for each component)
            # =================================================================
            logger.info("💻 Phase 2: Generating code...")

            components = arch_result.get('components', [])
            if not components:
                components = [{'name': 'main', 'description': requirements}]

            code_tasks = []
            for comp in components:
                comp_name = comp.get('name', 'component') if isinstance(comp, dict) else str(comp)
                code_tasks.append(
                    self._developer.generate(
                        architecture=arch_result['architecture'],
                        component=comp_name,
                        language=lang.value,
                        dependencies=config.frameworks
                    )
                )

            code_results = await asyncio.gather(*code_tasks, return_exceptions=True)

            # Collect generated files
            files = {}
            main_file = None
            for result in code_results:
                if isinstance(result, Exception):
                    continue
                if 'code' in result and 'filename' in result:
                    filename = result['filename']
                    files[filename] = result['code']
                    if main_file is None:
                        main_file = filename

            phase2_start = datetime.now()
            self._trace_phase("Developer", AgentRole.ACTOR,
                {'components': len(components)},
                {'files_generated': len(files)},
                success=len(files) > 0, phase_start=start_time, tools_used=['code_generate'])

            # =================================================================
            # PHASE 3: OPTIMIZATION
            # =================================================================
            logger.info("⚡ Phase 3: Optimizing code...")

            optimized_files = {}
            for filename, code in files.items():
                opt_result = await self._optimizer.optimize(
                    code=code,
                    focus="readability",
                    constraints="Maintain all functionality"
                )
                if 'optimized_code' in opt_result:
                    optimized_files[filename] = opt_result['optimized_code']
                else:
                    optimized_files[filename] = code

            files = optimized_files

            self._trace_phase("Optimizer", AgentRole.ACTOR,
                {'files_count': len(files)},
                {'optimized': len(optimized_files)},
                success=True, phase_start=phase2_start, tools_used=['code_optimize'])

            # =================================================================
            # PHASE 4: TEST GENERATION (if enabled)
            # =================================================================
            tests = {}
            test_coverage = 0.0

            if gen_tests and files:
                logger.info("🧪 Phase 4: Generating tests...")

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
                logger.info("📝 Phase 5: Generating documentation...")

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
                quality_score=0.8  # Could be calculated from analysis
            )

            logger.info(f"✅ CodingSwarm complete: {loc} LOC, {len(tests)} test files")

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
        focus: str = "readability"
    ) -> Dict[str, Any]:
        """Refactor/optimize code."""
        self._init_agents()
        return await self._optimizer.optimize(code, focus)


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
    # Agents
    'ArchitectAgent',
    'DeveloperAgent',
    'DebuggerAgent',
    'OptimizerAgent',
    'TestWriterAgent',
    'DocWriterAgent',
]
