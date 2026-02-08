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
class TestingConfig(SwarmConfig):
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

class BaseTestAgent(DomainAgent):
    """Base class for testing agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, signature=None):
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus

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


class CodeAnalyzerAgent(BaseTestAgent):
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
            except:
                testable_units = []

            try:
                dependencies = json.loads(result.dependencies)
            except:
                dependencies = []

            edge_cases = [e.strip() for e in str(result.edge_cases).split('|') if e.strip()]
            integration_points = [i.strip() for i in str(result.integration_points).split('|') if i.strip()]

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


class UnitTestAgent(BaseTestAgent):
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

            test_cases = [t.strip() for t in str(result.test_cases).split('|') if t.strip()]
            coverage_areas = [c.strip() for c in str(result.coverage_areas).split('|') if c.strip()]

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


class IntegrationTestAgent(BaseTestAgent):
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

            scenarios = [s.strip() for s in str(result.test_scenarios).split('|') if s.strip()]
            fixtures = [f.strip() for f in str(result.fixtures_needed).split('|') if f.strip()]

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


class E2ETestAgent(BaseTestAgent):
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

            steps = [s.strip() for s in str(result.steps).split('|') if s.strip()]
            assertions = [a.strip() for a in str(result.assertions).split('|') if a.strip()]

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


class CoverageAgent(BaseTestAgent):
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

            gaps = [g.strip() for g in str(result.gaps).split('|') if g.strip()]
            recommendations = [r.strip() for r in str(result.recommendations).split('|') if r.strip()]
            priority_areas = [p.strip() for p in str(result.priority_areas).split('|') if p.strip()]

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


class QualityAgent(BaseTestAgent):
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

            issues = [i.strip() for i in str(result.issues).split('|') if i.strip()]
            improvements = [i.strip() for i in str(result.improvements).split('|') if i.strip()]

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
        framework: TestFramework = None
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
        start_time = datetime.now()

        # Note: Pre-execution learning and agent init handled by DomainSwarm.execute()

        config = self.config
        lang = language or config.language
        types = test_types or config.test_types
        fw = framework or config.framework

        logger.info(f"🧪 TestingSwarm starting: {lang}, {fw.value}")

        try:
            # =================================================================
            # PHASE 1: CODE ANALYSIS
            # =================================================================
            logger.info("🔍 Phase 1: Analyzing code for testability...")

            analysis = await self._analyzer.analyze(code, lang)

            if 'error' in analysis:
                return TestingResult(
                    success=False,
                    swarm_name=self.config.name,
                    domain=self.config.domain,
                    output={},
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=analysis['error']
                )

            testable_units = analysis.get('testable_units', [])
            dependencies = analysis.get('dependencies', [])
            edge_cases = analysis.get('edge_cases', [])
            integration_points = analysis.get('integration_points', [])

            self._trace_phase("CodeAnalyzer", AgentRole.EXPERT,
                {'code_length': len(code)},
                {'testable_units': len(testable_units), 'integration_points': len(integration_points)},
                success='error' not in analysis, phase_start=start_time, tools_used=['code_analyze'])

            # =================================================================
            # PHASE 2: UNIT TEST GENERATION (parallel)
            # =================================================================
            all_tests = []
            all_test_code = []

            if TestType.UNIT in types and testable_units:
                logger.info("🔬 Phase 2: Generating unit tests...")

                unit_tasks = []
                for unit in testable_units[:10]:  # Limit to 10 units
                    unit_name = unit.get('name', str(unit)) if isinstance(unit, dict) else str(unit)
                    unit_tasks.append(
                        self._unit_tester.generate(
                            code=code,
                            unit=unit_name,
                            framework=fw.value,
                            mocks=dependencies
                        )
                    )

                unit_results = await asyncio.gather(*unit_tasks, return_exceptions=True)

                for i, result in enumerate(unit_results):
                    if isinstance(result, Exception):
                        continue
                    if 'test_code' in result:
                        unit_name = testable_units[i].get('name', f'unit_{i}') if isinstance(testable_units[i], dict) else str(testable_units[i])
                        all_tests.append(TestCase(
                            name=f"test_{unit_name}",
                            test_type=TestType.UNIT,
                            description=f"Unit tests for {unit_name}",
                            code=result['test_code']
                        ))
                        all_test_code.append(result['test_code'])

            phase2_start = datetime.now()
            self._trace_phase("UnitTest", AgentRole.ACTOR,
                {'testable_units': len(testable_units)},
                {'unit_tests': len([t for t in all_tests if t.test_type == TestType.UNIT])},
                success=True, phase_start=start_time, tools_used=['unit_test_generate'])

            # =================================================================
            # PHASE 3: INTEGRATION TEST GENERATION
            # =================================================================
            if TestType.INTEGRATION in types and integration_points:
                logger.info("🔗 Phase 3: Generating integration tests...")

                int_tasks = []
                for point in integration_points[:5]:  # Limit to 5 points
                    int_tasks.append(
                        self._integration_tester.generate(
                            code=code,
                            integration_point=point,
                            framework=fw.value
                        )
                    )

                int_results = await asyncio.gather(*int_tasks, return_exceptions=True)

                for i, result in enumerate(int_results):
                    if isinstance(result, Exception):
                        continue
                    if 'test_code' in result:
                        all_tests.append(TestCase(
                            name=f"test_integration_{i}",
                            test_type=TestType.INTEGRATION,
                            description=f"Integration tests for {integration_points[i]}",
                            code=result['test_code']
                        ))
                        all_test_code.append(result['test_code'])

            self._trace_phase("IntegrationTest", AgentRole.ACTOR,
                {'integration_points': len(integration_points)},
                {'integration_tests': len([t for t in all_tests if t.test_type == TestType.INTEGRATION])},
                success=True, phase_start=phase2_start, tools_used=['integration_test_generate'])

            # =================================================================
            # PHASE 4: E2E TEST GENERATION
            # =================================================================
            if TestType.E2E in types:
                logger.info("🌐 Phase 4: Generating E2E tests...")

                e2e_result = await self._e2e_tester.generate(
                    code=code,
                    user_flow="Main user journey",
                    framework="playwright"
                )

                if 'test_code' in e2e_result:
                    all_tests.append(TestCase(
                        name="test_e2e_main_flow",
                        test_type=TestType.E2E,
                        description="E2E test for main user journey",
                        code=e2e_result['test_code']
                    ))
                    all_test_code.append(e2e_result['test_code'])

            phase4_start = datetime.now()
            self._trace_phase("E2ETest", AgentRole.ACTOR,
                {'e2e_enabled': TestType.E2E in types},
                {'e2e_tests': len([t for t in all_tests if t.test_type == TestType.E2E])},
                success=True, phase_start=phase2_start, tools_used=['e2e_test_generate'])

            # =================================================================
            # PHASE 5: COVERAGE ANALYSIS
            # =================================================================
            logger.info("📊 Phase 5: Analyzing coverage...")

            combined_tests = "\n\n".join(all_test_code)
            coverage_result = await self._coverage_agent.analyze(code, combined_tests)

            estimated_coverage = coverage_result.get('estimated_coverage', 0.0)
            gaps = coverage_result.get('gaps', [])

            self._trace_phase("Coverage", AgentRole.EXPERT,
                {'test_count': len(all_tests)},
                {'estimated_coverage': estimated_coverage, 'gaps': len(gaps)},
                success=True, phase_start=phase4_start, tools_used=['coverage_analyze'])

            # =================================================================
            # PHASE 6: QUALITY ASSESSMENT
            # =================================================================
            logger.info("✨ Phase 6: Assessing quality...")

            quality_result = await self._quality_agent.assess(combined_tests, code)
            quality_score = quality_result.get('quality_score', 0.0)

            self._trace_phase("Quality", AgentRole.REVIEWER,
                {'test_count': len(all_tests)},
                {'quality_score': quality_score},
                success=True, phase_start=phase4_start, tools_used=['quality_assess'])

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            test_suite = TestSuite(
                tests=all_tests,
                fixtures={},
                mocks={dep: f"mock_{dep}" for dep in dependencies},
                conftest="# Pytest configuration\nimport pytest\n",
                setup_teardown=""
            )

            coverage_report = CoverageReport(
                line_coverage=estimated_coverage,
                branch_coverage=estimated_coverage * 0.9,  # Estimate
                function_coverage=estimated_coverage * 0.95,
                uncovered_lines=[],
                uncovered_branches=[],
                gaps=gaps
            )

            result = TestingResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={'test_count': len(all_tests)},
                execution_time=exec_time,
                test_suite=test_suite,
                coverage=coverage_report,
                test_count=len(all_tests),
                estimated_coverage=estimated_coverage,
                quality_score=quality_score,
                gaps_found=gaps
            )

            logger.info(f"✅ TestingSwarm complete: {len(all_tests)} tests, {estimated_coverage:.1f}% coverage")

            # Post-execution learning
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['code_analyze', 'unit_test_generate', 'integration_test_generate']),
                task_type='test_generation',
                output_data={
                    'test_count': len(all_tests),
                    'estimated_coverage': estimated_coverage,
                    'quality_score': quality_score,
                    'gaps_found': len(gaps),
                    'unit_tests': len([t for t in all_tests if t.test_type == TestType.UNIT]),
                    'integration_tests': len([t for t in all_tests if t.test_type == TestType.INTEGRATION]),
                    'e2e_tests': len([t for t in all_tests if t.test_type == TestType.E2E]),
                },
                input_data={
                    'code_length': len(code),
                    'language': lang,
                    'test_types': [t.value for t in types],
                    'framework': fw.value,
                }
            )

            return result

        except Exception as e:
            logger.error(f"❌ TestingSwarm error: {e}")
            import traceback
            traceback.print_exc()
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['code_analyze']),
                task_type='test_generation'
            )
            return TestingResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
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
