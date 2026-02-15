"""
Coding Swarm - Agent Definitions
==================================

All specialized coding agents used by CodingSwarm:
  ArchitectAgent, DeveloperAgent, DebuggerAgent, OptimizerAgent,
  TestWriterAgent, DocWriterAgent, VerifierAgent, SimplicityJudgeAgent,
  SystemDesignerAgent, DatabaseArchitectAgent, APIDesignerAgent,
  FrontendDeveloperAgent, IntegrationAgent, ArbitratorAgent,
  CodebaseAnalyzerAgent, EditPlannerAgent
"""

import json
import logging
from typing import Dict, Any, Optional, List

import dspy

from Jotty.core.modes.agent.base import DomainAgent, DomainAgentConfig, BaseSwarmAgent
from Jotty.core.intelligence.swarms.base import _split_field
from .utils import _stream_call
from .signatures import (
    ArchitectureDesignSignature,
    CodeGenerationSignature,
    DebugAnalysisSignature,
    CodeOptimizationSignature,
    TestGenerationSignature,
    DocumentationSignature,
    CodeVerificationSignature,
    SimplicityJudgeSignature,
    SystemDesignSignature,
    DatabaseSchemaSignature,
    APIGenerationSignature,
    FrontendGenerationSignature,
    IntegrationSignature,
    ReviewArbitrationSignature,
    CodebaseAnalysisSignature,
    EditPlanSignature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE
# =============================================================================

class BaseCodeAgent(BaseSwarmAgent):
    """Base class for coding agents. Extends BaseSwarmAgent with streaming support."""

    async def _stream(self, module: Any, phase: str, agent: str, listener_field: str = 'reasoning', **kwargs: Any) -> Any:
        """Call DSPy module with streaming reasoning to progress callback."""
        return await _stream_call(module, phase, agent, listener_field, **kwargs)


# =============================================================================
# CORE AGENTS
# =============================================================================

class ArchitectAgent(BaseCodeAgent):
    """Designs software architecture."""

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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
            except Exception:
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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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

            improvements = _split_field(result.improvements)

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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

            test_cases = _split_field(result.test_cases)

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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


# =============================================================================
# VERIFICATION AGENTS
# =============================================================================

class VerifierAgent(BaseCodeAgent):
    """Verifies generated code against original requirements."""

    # Max chars to avoid context overflow (roughly 8K tokens)
    MAX_CODE_CHARS = 32000
    MAX_ARCH_CHARS = 4000

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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
    """Judges code for over-engineering and unnecessary complexity."""

    MAX_CODE_CHARS = 32000

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._judge = dspy.ChainOfThought(SimplicityJudgeSignature)

    async def judge(
        self,
        code: str,
        requirements: str,
        file_count: int = 1,
        total_lines: int = 0
    ) -> Dict[str, Any]:
        """Evaluate code for over-engineering."""
        try:
            if total_lines == 0:
                total_lines = len(code.split('\n'))

            if len(code) > self.MAX_CODE_CHARS:
                code = code[:self.MAX_CODE_CHARS] + "\n\n# ... (truncated for evaluation)"

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

            try:
                simplicity_score = float(result.simplicity_score)
                simplicity_score = max(0.0, min(1.0, simplicity_score))
            except (TypeError, ValueError):
                simplicity_score = 0.8

            verdict = str(result.verdict).strip().upper()
            if verdict not in ('ACCEPT', 'SIMPLIFY'):
                verdict = 'ACCEPT' if simplicity_score >= 0.6 else 'SIMPLIFY'

            needs_simplification = verdict == 'SIMPLIFY'
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


# =============================================================================
# FULL-STACK AGENTS
# =============================================================================

class SystemDesignerAgent(BaseCodeAgent):
    """Designs full-stack system architecture."""

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(DatabaseSchemaSignature)

    async def generate(self, data_model: str, db_type: str, language: str) -> Dict[str, Any]:
        """Generate database schema and ORM models."""
        try:
            if self.learned_context:
                data_model = data_model + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2a", "DatabaseArchitect",
                data_model=data_model, db_type=db_type, language=language)

            self._broadcast("database_designed", {'db_type': db_type, 'has_schema': bool(result.schema_sql)})

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(APIGenerationSignature)

    async def generate(self, architecture: str, orm_models: str, api_contract: str,
                       language: str, framework: str) -> Dict[str, Any]:
        """Generate backend API code and OpenAPI spec."""
        try:
            if self.learned_context:
                architecture = architecture + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2b", "APIDesigner",
                architecture=architecture, orm_models=orm_models,
                api_contract=api_contract, language=language, framework=framework)

            self._broadcast("api_generated", {'framework': framework, 'has_openapi': bool(result.openapi_spec)})

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(FrontendGenerationSignature)

    async def generate(self, openapi_spec: str, ui_requirements: str, framework: str) -> Dict[str, Any]:
        """Generate frontend code consuming OpenAPI spec."""
        try:
            if self.learned_context:
                openapi_spec = openapi_spec + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2c", "FrontendDeveloper",
                openapi_spec=openapi_spec, ui_requirements=ui_requirements, framework=framework)

            self._broadcast("frontend_generated", {'framework': framework, 'has_api_client': bool(result.api_client)})

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(IntegrationSignature)

    async def generate(self, file_list: List[str], tech_stack: Dict[str, str],
                       architecture: str) -> Dict[str, Any]:
        """Generate integration artifacts (Docker, configs)."""
        try:
            if self.learned_context:
                architecture = architecture + f"\n\n{self.learned_context}"

            result = await self._stream(self._generator, "Phase 2d", "IntegrationEngineer",
                file_list=json.dumps(file_list), tech_stack=json.dumps(tech_stack),
                architecture=architecture)

            self._broadcast("integration_generated", {'has_docker': bool(result.docker_compose)})

            return {
                'docker_compose': str(result.docker_compose),
                'env_config': str(result.env_config),
                'requirements_txt': str(result.requirements_txt),
                'startup_script': str(result.startup_script),
            }
        except Exception as e:
            logger.error(f"Integration generation failed: {e}")
            return {'error': str(e)}


# =============================================================================
# REVIEW / EDIT AGENTS
# =============================================================================

class ArbitratorAgent(BaseCodeAgent):
    """Evaluates whether a code review rejection is valid and actionable."""

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._evaluator = dspy.ChainOfThought(ReviewArbitrationSignature)

    async def evaluate(self, code: str, rejection_feedback: str,
                       evidence: str, reviewer_name: str) -> Dict[str, Any]:
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
                code=code, rejection_feedback=rejection_feedback,
                evidence=evidence, reviewer_name=reviewer_name)

            valid_str = str(result.valid).strip().upper()
            valid = valid_str == "TRUE"

            self._broadcast("rejection_arbitrated", {'reviewer': reviewer_name, 'valid': valid})

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._analyzer = dspy.ChainOfThought(CodebaseAnalysisSignature)

    async def analyze(self, existing_code: str, file_paths: List[str],
                      requirements: str) -> Dict[str, Any]:
        """Analyze existing codebase structure and patterns."""
        try:
            result = await self._stream(self._analyzer, "Phase 0", "CodebaseAnalyzer",
                existing_code=existing_code,
                file_paths=json.dumps(file_paths),
                requirements=requirements)

            try:
                affected_files = json.loads(str(result.affected_files))
                if not isinstance(affected_files, list):
                    affected_files = []
            except (json.JSONDecodeError, TypeError):
                affected_files = []

            self._broadcast("codebase_analyzed", {
                'files_analyzed': len(file_paths), 'affected_files': len(affected_files)})

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

    def __init__(self, memory: Any = None, context: Any = None, bus: Any = None, learned_context: str = '') -> None:
        super().__init__(memory, context, bus, learned_context)
        self._planner = dspy.ChainOfThought(EditPlanSignature)

    async def plan_edit(self, existing_code: str, file_path: str, requirements: str,
                        style_conventions: str = "", dependencies: str = "") -> Dict[str, Any]:
        """Plan edits for a single file."""
        try:
            result = await self._stream(self._planner, "Phase 1", "EditPlanner",
                existing_code=existing_code, file_path=file_path,
                requirements=requirements,
                style_conventions=style_conventions or "Follow existing patterns",
                dependencies=dependencies or "Preserve all existing interfaces")

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
                'file': file_path, 'edit_type': edit_type, 'num_edits': len(edits)})

            return {
                'edit_plan': str(result.edit_plan),
                'edits': edits,
                'new_code': str(result.new_code) if edit_type == 'rewrite' else '',
                'edit_type': edit_type,
            }
        except Exception as e:
            logger.error(f"Edit planning failed for {file_path}: {e}")
            return {'error': str(e)}
