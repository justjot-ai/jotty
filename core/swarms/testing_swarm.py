"""
Testing Swarm - World-Class Test Automation & Quality Assurance
================================================================

Production-grade swarm for:
- Comprehensive test generation (unit, integration, e2e)
- Test coverage analysis and gap identification
- Test quality assessment
- Mutation testing
- Flaky test detection
- Test optimization

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                          TESTING SWARM                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  Analyzer      │  │  Unit Test     │  │  Integration   │            │
│  │    Agent       │  │    Agent       │  │  Test Agent    │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   E2E Test     │  │   Coverage     │  │   Quality      │            │
│  │    Agent       │  │    Agent       │  │    Agent       │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     TEST ORCHESTRATOR                            │   │
│  │   Coordinates test strategy and generates comprehensive suite    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.testing_swarm import TestingSwarm, test

    # Full swarm
    swarm = TestingSwarm()
    result = await swarm.generate_tests(code="...", language="python")

    # One-liner
    result = await test(code, language="python")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmBaseConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam, _split_field
from .swarm_signatures import TestingSwarmSignature
from ..agents.base import DomainAgent, DomainAgentConfig, BaseSwarmAgent

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PROPERTY = "property"
    MUTATION = "mutation"
    PERFORMANCE = "performance"


class TestFramework(Enum):
    PYTEST = "pytest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    RSPEC = "rspec"
    GO_TEST = "go_test"


class CoverageTarget(Enum):
    MINIMUM = 60
    STANDARD = 80
    COMPREHENSIVE = 90
    FULL = 95


@dataclass
class TestingConfig(SwarmBaseConfig):
    """Configuration for TestingSwarm."""
    test_types: List[TestType] = field(default_factory=lambda: [TestType.UNIT, TestType.INTEGRATION])
    framework: TestFramework = TestFramework.PYTEST
    coverage_target: CoverageTarget = CoverageTarget.STANDARD
    include_mocks: bool = True
    include_fixtures: bool = True
    include_property_tests: bool = False
    include_mutation_tests: bool = False
    parallel_tests: bool = True
    language: str = "python"

    def __post_init__(self):
        self.name = "TestingSwarm"
        self.domain = "testing"


@dataclass
class TestCase:
    """Individual test case."""
    name: str
    test_type: TestType
    description: str
    code: str
    expected_coverage: float = 0.0
    priority: int = 1


@dataclass
class TestSuite:
    """Complete test suite."""
    tests: List[TestCase]
    fixtures: Dict[str, str]
    mocks: Dict[str, str]
    conftest: str = ""
    setup_teardown: str = ""


@dataclass
class CoverageReport:
    """Coverage analysis report."""
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    uncovered_lines: List[int]
    uncovered_branches: List[str]
    gaps: List[str]


@dataclass
class TestingResult(SwarmResult):
    """Result from TestingSwarm."""
    test_suite: Optional[TestSuite] = None
    coverage: Optional[CoverageReport] = None
    test_count: int = 0
    estimated_coverage: float = 0.0
    quality_score: float = 0.0
    gaps_found: List[str] = field(default_factory=list)


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class CodeAnalysisSignature(dspy.Signature):
    """Analyze code for testability.

    You are a TEST ARCHITECT. Analyze code to identify:
    1. Testable units (functions, classes, methods)
    2. Dependencies to mock
    3. Edge cases to cover
    4. Integration points
    5. Potential failure modes

    Be thorough - find ALL testable components.
    """
    code: str = dspy.InputField(desc="Source code to analyze")
    language: str = dspy.InputField(desc="Programming language")
    context: str = dspy.InputField(desc="Additional context about the codebase")

    testable_units: str = dspy.OutputField(desc="JSON list of testable units with names and descriptions")
    dependencies: str = dspy.OutputField(desc="JSON list of dependencies to mock")
    edge_cases: str = dspy.OutputField(desc="Edge cases to test, separated by |")
    integration_points: str = dspy.OutputField(desc="Integration points requiring tests, separated by |")


class UnitTestSignature(dspy.Signature):
    """Generate comprehensive unit tests.

    You are a UNIT TEST EXPERT. Write tests that:
    1. Test one thing at a time
    2. Are isolated (no external dependencies)
    3. Are deterministic
    4. Use clear naming: test_<function>_<scenario>_<expected>
    5. Follow AAA pattern: Arrange, Act, Assert

    Include:
    - Happy path tests
    - Error cases
    - Edge cases
    - Boundary conditions
    """
    code: str = dspy.InputField(desc="Code to test")
    unit: str = dspy.InputField(desc="Specific unit to test")
    framework: str = dspy.InputField(desc="Test framework")
    mocks: str = dspy.InputField(desc="Available mocks/stubs")

    test_code: str = dspy.OutputField(desc="Complete test code")
    test_cases: str = dspy.OutputField(desc="Test cases covered, separated by |")
    coverage_areas: str = dspy.OutputField(desc="Code paths covered, separated by |")


class IntegrationTestSignature(dspy.Signature):
    """Generate integration tests.

    You are an INTEGRATION TEST EXPERT. Write tests that verify:
    1. Component interactions
    2. Data flow between modules
    3. API contracts
    4. Database operations
    5. External service integration

    Tests should be:
    - Reproducible
    - Idempotent
    - Clean up after themselves
    """
    code: str = dspy.InputField(desc="Code to test")
    integration_point: str = dspy.InputField(desc="Integration point to test")
    framework: str = dspy.InputField(desc="Test framework")
    setup_info: str = dspy.InputField(desc="Setup/teardown requirements")

    test_code: str = dspy.OutputField(desc="Integration test code")
    test_scenarios: str = dspy.OutputField(desc="Scenarios covered, separated by |")
    fixtures_needed: str = dspy.OutputField(desc="Fixtures required, separated by |")


class E2ETestSignature(dspy.Signature):
    """Generate end-to-end tests.

    You are an E2E TEST EXPERT. Write tests that verify:
    1. Complete user flows
    2. System-wide behavior
    3. Real-world scenarios
    4. Error recovery paths

    E2E tests should:
    - Simulate real user behavior
    - Test critical paths first
    - Be resilient to minor UI changes
    """
    code: str = dspy.InputField(desc="Code/system to test")
    user_flow: str = dspy.InputField(desc="User flow to test")
    framework: str = dspy.InputField(desc="Test framework (playwright, cypress, selenium)")
    config: str = dspy.InputField(desc="E2E configuration")

    test_code: str = dspy.OutputField(desc="E2E test code")
    steps: str = dspy.OutputField(desc="Test steps, separated by |")
    assertions: str = dspy.OutputField(desc="Key assertions, separated by |")


class CoverageAnalysisSignature(dspy.Signature):
    """Analyze test coverage gaps.

    You are a COVERAGE ANALYST. Identify:
    1. Uncovered lines
    2. Untested branches
    3. Missing edge cases
    4. Weak test areas
    5. Recommendations for improvement
    """
    code: str = dspy.InputField(desc="Source code")
    tests: str = dspy.InputField(desc="Existing test code")
    coverage_data: str = dspy.InputField(desc="Current coverage data if available")

    estimated_coverage: float = dspy.OutputField(desc="Estimated coverage percentage")
    gaps: str = dspy.OutputField(desc="Coverage gaps, separated by |")
    recommendations: str = dspy.OutputField(desc="Recommendations to improve coverage, separated by |")
    priority_areas: str = dspy.OutputField(desc="Priority areas to test, separated by |")


class TestQualitySignature(dspy.Signature):
    """Assess test quality.

    You are a TEST QUALITY EXPERT. Evaluate tests for:
    1. Clarity and readability
    2. Maintainability
    3. Completeness
    4. Isolation
    5. Speed
    6. Determinism

    Score each aspect and provide improvement suggestions.
    """
    tests: str = dspy.InputField(desc="Test code to evaluate")
    code: str = dspy.InputField(desc="Code being tested")

    quality_score: float = dspy.OutputField(desc="Overall quality score 0.0-1.0")
    clarity_score: float = dspy.OutputField(desc="Clarity score 0.0-1.0")
    completeness_score: float = dspy.OutputField(desc="Completeness score 0.0-1.0")
    issues: str = dspy.OutputField(desc="Quality issues found, separated by |")
    improvements: str = dspy.OutputField(desc="Improvement suggestions, separated by |")


# =============================================================================
# AGENTS
# =============================================================================



class CodeAnalyzerAgent(BaseSwarmAgent):
    """Analyzes code for testability."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=CodeAnalysisSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(CodeAnalysisSignature)

    async def analyze(
        self,
        code: str,
        language: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze code for testability."""
        try:
            enriched_context = (context or "No additional context") + ("\n\n" + self.learned_context if self.learned_context else "")
            result = self._analyzer(
                code=code,
                language=language,
                context=enriched_context
            )

            # Parse results
            try:
                testable_units = json.loads(result.testable_units)
            except Exception:
                testable_units = []

            try:
                dependencies = json.loads(result.dependencies)
            except Exception:
                dependencies = []

            edge_cases = _split_field(result.edge_cases)
            integration_points = _split_field(result.integration_points)

            self._broadcast("code_analyzed", {
                'units': len(testable_units),
                'dependencies': len(dependencies)
            })

            return {
                'testable_units': testable_units,
                'dependencies': dependencies,
                'edge_cases': edge_cases,
                'integration_points': integration_points
            }

        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {'error': str(e)}


class UnitTestAgent(BaseSwarmAgent):
    """Generates unit tests."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=UnitTestSignature)
        self.learned_context = learned_context
        self._generator = dspy.ChainOfThought(UnitTestSignature)

    async def generate(
        self,
        code: str,
        unit: str,
        framework: str,
        mocks: List[str] = None
    ) -> Dict[str, Any]:
        """Generate unit tests for a specific unit."""
        try:
            enriched_code = code + ("\n\n# Learned Context:\n" + self.learned_context if self.learned_context else "")
            result = self._generator(
                code=enriched_code,
                unit=unit,
                framework=framework,
                mocks=json.dumps(mocks or [])
            )

            test_cases = _split_field(result.test_cases)
            coverage_areas = _split_field(result.coverage_areas)

            self._broadcast("unit_tests_generated", {
                'unit': unit,
                'test_count': len(test_cases)
            })

            return {
                'test_code': str(result.test_code),
                'test_cases': test_cases,
                'coverage_areas': coverage_areas
            }

        except Exception as e:
            logger.error(f"Unit test generation failed: {e}")
            return {'error': str(e)}


class IntegrationTestAgent(BaseSwarmAgent):
    """Generates integration tests."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=IntegrationTestSignature)
        self.learned_context = learned_context
        self._generator = dspy.ChainOfThought(IntegrationTestSignature)

    async def generate(
        self,
        code: str,
        integration_point: str,
        framework: str,
        setup_info: str = ""
    ) -> Dict[str, Any]:
        """Generate integration tests."""
        try:
            enriched_setup = (setup_info or "Standard setup") + ("\n\n" + self.learned_context if self.learned_context else "")
            result = self._generator(
                code=code,
                integration_point=integration_point,
                framework=framework,
                setup_info=enriched_setup
            )

            scenarios = _split_field(result.test_scenarios)
            fixtures = _split_field(result.fixtures_needed)

            self._broadcast("integration_tests_generated", {
                'integration_point': integration_point,
                'scenarios': len(scenarios)
            })

            return {
                'test_code': str(result.test_code),
                'test_scenarios': scenarios,
                'fixtures_needed': fixtures
            }

        except Exception as e:
            logger.error(f"Integration test generation failed: {e}")
            return {'error': str(e)}


class E2ETestAgent(BaseSwarmAgent):
    """Generates end-to-end tests."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=E2ETestSignature)
        self.learned_context = learned_context
        self._generator = dspy.ChainOfThought(E2ETestSignature)

    async def generate(
        self,
        code: str,
        user_flow: str,
        framework: str = "playwright",
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate E2E tests."""
        try:
            enriched_config = json.dumps(config or {}) + ("\n\n" + self.learned_context if self.learned_context else "")
            result = self._generator(
                code=code,
                user_flow=user_flow,
                framework=framework,
                config=enriched_config
            )

            steps = _split_field(result.steps)
            assertions = _split_field(result.assertions)

            self._broadcast("e2e_tests_generated", {
                'user_flow': user_flow,
                'steps': len(steps)
            })

            return {
                'test_code': str(result.test_code),
                'steps': steps,
                'assertions': assertions
            }

        except Exception as e:
            logger.error(f"E2E test generation failed: {e}")
            return {'error': str(e)}


class CoverageAgent(BaseSwarmAgent):
    """Analyzes test coverage."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=CoverageAnalysisSignature)
        self.learned_context = learned_context
        self._analyzer = dspy.ChainOfThought(CoverageAnalysisSignature)

    async def analyze(
        self,
        code: str,
        tests: str,
        coverage_data: str = ""
    ) -> Dict[str, Any]:
        """Analyze coverage gaps."""
        try:
            enriched_coverage_data = (coverage_data or "No coverage data available") + ("\n\n" + self.learned_context if self.learned_context else "")
            result = self._analyzer(
                code=code,
                tests=tests,
                coverage_data=enriched_coverage_data
            )

            gaps = _split_field(result.gaps)
            recommendations = _split_field(result.recommendations)
            priority_areas = _split_field(result.priority_areas)

            self._broadcast("coverage_analyzed", {
                'estimated_coverage': float(result.estimated_coverage) if result.estimated_coverage else 0,
                'gaps_found': len(gaps)
            })

            return {
                'estimated_coverage': float(result.estimated_coverage) if result.estimated_coverage else 0.0,
                'gaps': gaps,
                'recommendations': recommendations,
                'priority_areas': priority_areas
            }

        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return {'error': str(e)}


class QualityAgent(BaseSwarmAgent):
    """Assesses test quality."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=TestQualitySignature)
        self.learned_context = learned_context
        self._assessor = dspy.ChainOfThought(TestQualitySignature)

    async def assess(
        self,
        tests: str,
        code: str
    ) -> Dict[str, Any]:
        """Assess test quality."""
        try:
            enriched_code = code + ("\n\n# Learned Context:\n" + self.learned_context if self.learned_context else "")
            result = self._assessor(
                tests=tests,
                code=enriched_code
            )

            issues = _split_field(result.issues)
            improvements = _split_field(result.improvements)

            self._broadcast("quality_assessed", {
                'quality_score': float(result.quality_score) if result.quality_score else 0
            })

            return {
                'quality_score': float(result.quality_score) if result.quality_score else 0.0,
                'clarity_score': float(result.clarity_score) if result.clarity_score else 0.0,
                'completeness_score': float(result.completeness_score) if result.completeness_score else 0.0,
                'issues': issues,
                'improvements': improvements
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {'error': str(e)}


# =============================================================================
# TESTING SWARM
# =============================================================================

@register_swarm("testing")
class TestingSwarm(DomainSwarm):
    """
    World-Class Testing Swarm.

    Generates comprehensive test suites with:
    - Unit tests
    - Integration tests
    - E2E tests
    - Coverage analysis
    - Quality assessment
    """

    # Declarative agent team - auto-initialized by DomainSwarm
    AGENT_TEAM = AgentTeam.define(
        (CodeAnalyzerAgent, "CodeAnalyzer", "_analyzer"),
        (UnitTestAgent, "UnitTest", "_unit_tester"),
        (IntegrationTestAgent, "IntegrationTest", "_integration_tester"),
        (E2ETestAgent, "E2ETest", "_e2e_tester"),
        (CoverageAgent, "Coverage", "_coverage_agent"),
        (QualityAgent, "Quality", "_quality_agent"),
    )
    SWARM_SIGNATURE = TestingSwarmSignature

    def __init__(self, config: TestingConfig = None):
        super().__init__(config or TestingConfig())

    async def _execute_domain(
        self,
        code: str,
        language: str = None,
        **kwargs
    ) -> TestingResult:
        """Execute test generation (called by DomainSwarm.execute())."""
        return await self.generate_tests(code, language, **kwargs)

    async def generate_tests(
        self,
        code: str,
        language: str = None,
        test_types: List[TestType] = None,
        framework: TestFramework = None,
        **kwargs
    ) -> TestingResult:
        """
        Generate comprehensive test suite.

        Args:
            code: Source code to test
            language: Programming language
            test_types: Types of tests to generate
            framework: Test framework to use

        Returns:
            TestingResult with test suite
        """
        config = self.config
        lang = language or config.language
        types = test_types or config.test_types
        fw = framework or config.framework

        logger.info(f"TestingSwarm starting: {lang}, {fw.value}")

        return await self._safe_execute_domain(
            task_type='test_generation',
            default_tools=['code_analyze', 'unit_test_generate', 'integration_test_generate'],
            result_class=TestingResult,
            execute_fn=lambda executor: self._execute_phases(executor, code, lang, types, fw),
            output_data_fn=lambda result: {
                'test_count': result.test_count,
                'estimated_coverage': result.estimated_coverage,
                'quality_score': result.quality_score,
                'gaps_found': len(result.gaps_found),
                'unit_tests': len([t for t in (result.test_suite.tests if result.test_suite else []) if t.test_type == TestType.UNIT]),
                'integration_tests': len([t for t in (result.test_suite.tests if result.test_suite else []) if t.test_type == TestType.INTEGRATION]),
                'e2e_tests': len([t for t in (result.test_suite.tests if result.test_suite else []) if t.test_type == TestType.E2E]),
            },
            input_data_fn=lambda: {
                'code_length': len(code),
                'language': lang,
                'test_types': [t.value for t in types],
                'framework': fw.value,
            },
        )

    async def _execute_phases(
        self,
        executor,
        code: str,
        lang: str,
        types: List[TestType],
        fw: TestFramework
    ) -> TestingResult:
        """Execute all testing phases using PhaseExecutor.

        Args:
            executor: PhaseExecutor instance from _safe_execute_domain
            code: Source code to test
            lang: Programming language
            types: Types of tests to generate
            fw: Test framework to use

        Returns:
            TestingResult with complete test suite
        """
        # =================================================================
        # PHASE 1: CODE ANALYSIS
        # =================================================================
        analysis = await executor.run_phase(
            1, "Code Analysis", "CodeAnalyzer", AgentRole.EXPERT,
            self._analyzer.analyze(code, lang),
            input_data={'code_length': len(code)},
            tools_used=['code_analyze'],
        )

        if 'error' in analysis:
            return executor.build_error_result(
                TestingResult, Exception(analysis['error']),
                self.config.name, self.config.domain,
            )

        testable_units = analysis.get('testable_units', [])
        raw_deps = analysis.get('dependencies', [])
        # LLM may return deps as dicts (e.g. {'name': 'os', 'type': 'stdlib'}) — coerce to strings
        dependencies = [
            d.get('name', str(d)) if isinstance(d, dict) else str(d)
            for d in raw_deps
        ] if isinstance(raw_deps, list) else []
        raw_points = analysis.get('integration_points', [])
        integration_points = [
            p.get('name', str(p)) if isinstance(p, dict) else str(p)
            for p in raw_points
        ] if isinstance(raw_points, list) else []

        # =================================================================
        # PHASE 2: UNIT TEST GENERATION (parallel)
        # =================================================================
        all_tests = []
        all_test_code = []

        if TestType.UNIT in types and testable_units:
            unit_parallel_tasks = []
            for unit in testable_units[:10]:  # Limit to 10 units
                unit_name = unit.get('name', str(unit)) if isinstance(unit, dict) else str(unit)
                unit_parallel_tasks.append((
                    f"UnitTest({unit_name})", AgentRole.ACTOR,
                    self._unit_tester.generate(
                        code=code,
                        unit=unit_name,
                        framework=fw.value,
                        mocks=dependencies,
                    ),
                    ['unit_test_generate'],
                ))

            unit_results = await executor.run_parallel(
                2, "Unit Test Generation", unit_parallel_tasks,
            )

            for i, result in enumerate(unit_results):
                if isinstance(result, dict) and 'error' in result:
                    continue
                if isinstance(result, dict) and 'test_code' in result:
                    unit_name = testable_units[i].get('name', f'unit_{i}') if isinstance(testable_units[i], dict) else str(testable_units[i])
                    all_tests.append(TestCase(
                        name=f"test_{unit_name}",
                        test_type=TestType.UNIT,
                        description=f"Unit tests for {unit_name}",
                        code=result['test_code'],
                    ))
                    all_test_code.append(result['test_code'])

        # =================================================================
        # PHASE 3: INTEGRATION TEST GENERATION (parallel)
        # =================================================================
        if TestType.INTEGRATION in types and integration_points:
            int_parallel_tasks = []
            for point in integration_points[:5]:  # Limit to 5 points
                int_parallel_tasks.append((
                    f"IntegrationTest({point})", AgentRole.ACTOR,
                    self._integration_tester.generate(
                        code=code,
                        integration_point=point,
                        framework=fw.value,
                    ),
                    ['integration_test_generate'],
                ))

            int_results = await executor.run_parallel(
                3, "Integration Test Generation", int_parallel_tasks,
            )

            for i, result in enumerate(int_results):
                if isinstance(result, dict) and 'error' in result:
                    continue
                if isinstance(result, dict) and 'test_code' in result:
                    all_tests.append(TestCase(
                        name=f"test_integration_{i}",
                        test_type=TestType.INTEGRATION,
                        description=f"Integration tests for {integration_points[i]}",
                        code=result['test_code'],
                    ))
                    all_test_code.append(result['test_code'])

        # =================================================================
        # PHASE 4: E2E TEST GENERATION
        # =================================================================
        if TestType.E2E in types:
            e2e_result = await executor.run_phase(
                4, "E2E Test Generation", "E2ETest", AgentRole.ACTOR,
                self._e2e_tester.generate(
                    code=code,
                    user_flow="Main user journey",
                    framework="playwright",
                ),
                input_data={'e2e_enabled': True},
                tools_used=['e2e_test_generate'],
            )

            if isinstance(e2e_result, dict) and 'test_code' in e2e_result:
                all_tests.append(TestCase(
                    name="test_e2e_main_flow",
                    test_type=TestType.E2E,
                    description="E2E test for main user journey",
                    code=e2e_result['test_code'],
                ))
                all_test_code.append(e2e_result['test_code'])

        # =================================================================
        # PHASE 5: COVERAGE ANALYSIS
        # =================================================================
        combined_tests = "\n\n".join(all_test_code)

        coverage_result = await executor.run_phase(
            5, "Coverage Analysis", "Coverage", AgentRole.EXPERT,
            self._coverage_agent.analyze(code, combined_tests),
            input_data={'test_count': len(all_tests)},
            tools_used=['coverage_analyze'],
        )

        estimated_coverage = coverage_result.get('estimated_coverage', 0.0)
        gaps = coverage_result.get('gaps', [])

        # =================================================================
        # PHASE 6: QUALITY ASSESSMENT
        # =================================================================
        quality_result = await executor.run_phase(
            6, "Quality Assessment", "Quality", AgentRole.REVIEWER,
            self._quality_agent.assess(combined_tests, code),
            input_data={'test_count': len(all_tests)},
            tools_used=['quality_assess'],
        )

        quality_score = quality_result.get('quality_score', 0.0)

        # =================================================================
        # BUILD RESULT
        # =================================================================
        test_suite = TestSuite(
            tests=all_tests,
            fixtures={},
            mocks={dep: f"mock_{dep}" for dep in dependencies},
            conftest="# Pytest configuration\nimport pytest\n",
            setup_teardown="",
        )

        coverage_report = CoverageReport(
            line_coverage=estimated_coverage,
            branch_coverage=estimated_coverage * 0.9,  # Estimate
            function_coverage=estimated_coverage * 0.95,
            uncovered_lines=[],
            uncovered_branches=[],
            gaps=gaps,
        )

        logger.info(f"TestingSwarm complete: {len(all_tests)} tests, {estimated_coverage:.1f}% coverage")

        return TestingResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                'tests': combined_tests,
                'test_count': len(all_tests),
                'estimated_coverage': estimated_coverage,
                'quality_score': quality_score,
                'gaps': gaps,
            },
            execution_time=executor.elapsed(),
            test_suite=test_suite,
            coverage=coverage_report,
            test_count=len(all_tests),
            estimated_coverage=estimated_coverage,
            quality_score=quality_score,
            gaps_found=gaps,
        )

    async def analyze_coverage(
        self,
        code: str,
        existing_tests: str
    ) -> CoverageReport:
        """Analyze coverage of existing tests."""
        self._init_agents()
        result = await self._coverage_agent.analyze(code, existing_tests)
        return CoverageReport(
            line_coverage=result.get('estimated_coverage', 0.0),
            branch_coverage=result.get('estimated_coverage', 0.0) * 0.9,
            function_coverage=result.get('estimated_coverage', 0.0) * 0.95,
            uncovered_lines=[],
            uncovered_branches=[],
            gaps=result.get('gaps', [])
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def test(code: str, **kwargs) -> TestingResult:
    """
    One-liner test generation.

    Usage:
        from core.swarms.testing_swarm import test
        result = await test(my_code, language="python")
    """
    swarm = TestingSwarm()
    return await swarm.generate_tests(code, **kwargs)


def test_sync(code: str, **kwargs) -> TestingResult:
    """Synchronous test generation."""
    return asyncio.run(test(code, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TestingSwarm',
    'TestingConfig',
    'TestingResult',
    'TestSuite',
    'TestCase',
    'CoverageReport',
    'TestType',
    'TestFramework',
    'CoverageTarget',
    'test',
    'test_sync',
    # Agents
    'CodeAnalyzerAgent',
    'UnitTestAgent',
    'IntegrationTestAgent',
    'E2ETestAgent',
    'CoverageAgent',
    'QualityAgent',
]
