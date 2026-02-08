"""
Coding Swarm - Main CodingSwarm Class
=======================================

The CodingSwarm orchestrator: generates production-quality code through
collaborative multi-agent pipelines (architecture, development, optimization,
testing, review, documentation).
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

import dspy

from ..base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from ..base import DomainSwarm, AgentTeam

from .types import (
    CodeLanguage, CodeStyle, EditMode, CodingConfig,
    CodeOutput, ResearchContext, FullStackContext, CodingResult,
)
from .signatures import ScopeClassificationSignature
from .teams import TeamConfig, TEAM_PRESETS
from .agents import (
    ArchitectAgent, DeveloperAgent, DebuggerAgent, OptimizerAgent,
    TestWriterAgent, DocWriterAgent, VerifierAgent, SimplicityJudgeAgent,
    SystemDesignerAgent, DatabaseArchitectAgent, APIDesignerAgent,
    FrontendDeveloperAgent, IntegrationAgent,
)
from . import utils as _coding_utils
from .utils import (
    _strip_code_fences, _extract_components_from_text,
    _progress, _stream_call,
)
from ._codebase_mixin import CodebaseMixin
from ._edit_mixin import EditMixin
from ._review_mixin import ReviewMixin
from ._persistence_mixin import PersistenceMixin

logger = logging.getLogger(__name__)


@register_swarm("coding")
class CodingSwarm(CodebaseMixin, EditMixin, ReviewMixin, PersistenceMixin, DomainSwarm):
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
        _coding_utils._active_progress_callback = progress_callback
        _coding_utils._active_trace_callback = trace_callback
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
            _coding_utils._active_progress_callback = None
            _coding_utils._active_trace_callback = None
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

