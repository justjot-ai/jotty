"""
Composite Skill Providers for Jotty V2
=======================================

Combines multiple providers for complex multi-step tasks.
Uses swarm intelligence to coordinate and learn.
"""

import time
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import SkillProvider, SkillCategory, ProviderCapability, ProviderResult

logger = logging.getLogger(__name__)


@dataclass
class CompositeStep:
    """A step in a composite task."""
    provider_name: str
    category: SkillCategory
    task: str
    depends_on: Optional[List[str]] = None  # Step IDs this depends on
    step_id: str = ""
    optional: bool = False
    result: Optional[ProviderResult] = None


class ResearchAndAnalyzeProvider(SkillProvider):
    """
    Composite: Browser + Code Execution for research tasks.

    Pipeline:
    1. browser-use: Search and gather information
    2. open-interpreter: Analyze and process data
    3. jotty: Synthesize and format output
    """

    name = "research-analyze"
    version = "1.0.0"
    description = "Composite provider for research and analysis tasks"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.WEB_SEARCH,
                actions=["research", "analyze", "summarize", "report"],
                estimated_latency_ms=30000,  # Complex tasks take time
            ),
            ProviderCapability(
                category=SkillCategory.DATA_EXTRACTION,
                actions=["gather", "process", "analyze", "visualize"],
                estimated_latency_ms=30000,
            ),
        ]

        # Sub-providers (lazy loaded)
        self._browser_provider = None
        self._code_provider = None
        self._registry = None

    def set_registry(self, registry) -> None:
        """Set the provider registry for accessing other providers."""
        self._registry = registry

    async def initialize(self) -> bool:
        """Initialize composite provider."""
        # Providers will be accessed via registry
        self.is_initialized = True
        self.is_available = True
        logger.info(f" {self.name} composite provider initialized")
        return True

    def get_categories(self) -> List[SkillCategory]:
        return [SkillCategory.WEB_SEARCH, SkillCategory.DATA_EXTRACTION]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute composite research task."""
        start_time = time.time()
        context = context or {}

        try:
            results = {}

            # Step 1: Web research
            logger.info(f" Step 1: Web research for '{task[:30]}...'")
            if self._registry:
                browser_result = await self._registry.execute(
                    SkillCategory.BROWSER,
                    f"Search and gather information about: {task}",
                    context,
                )
                results['research'] = browser_result.output
            else:
                results['research'] = {'status': 'skipped', 'reason': 'no registry'}

            # Step 2: Data analysis
            logger.info(f" Step 2: Analyzing gathered data...")
            if self._registry and results.get('research'):
                analysis_task = f"Analyze this data and extract insights: {results['research']}"
                code_result = await self._registry.execute(
                    SkillCategory.CODE_EXECUTION,
                    analysis_task,
                    {'data': results['research']},
                )
                results['analysis'] = code_result.output
            else:
                results['analysis'] = {'status': 'skipped'}

            # Step 3: Synthesis
            logger.info(f" Step 3: Synthesizing results...")
            results['synthesis'] = {
                'task': task,
                'research_summary': str(results.get('research', ''))[:500],
                'analysis_summary': str(results.get('analysis', ''))[:500],
            }

            execution_time = time.time() - start_time

            result = ProviderResult(
                success=True,
                output=results,
                execution_time=execution_time,
                provider_name=self.name,
                category=SkillCategory.WEB_SEARCH,
                confidence=0.8,
            )

            self.record_execution(result)
            return result

        except Exception as e:
            logger.error(f"Composite research error: {e}")
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
            )


class AutomateWorkflowProvider(SkillProvider):
    """
    Composite: Terminal + Browser + Computer Use for automation.

    For tasks like:
    - "Download file from URL, extract data, and commit to git"
    - "Open browser, fill form, take screenshot, and save locally"
    """

    name = "automate-workflow"
    version = "1.0.0"
    description = "Composite provider for multi-step automation"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.TERMINAL,
                actions=["workflow", "automate", "pipeline"],
                estimated_latency_ms=60000,
            ),
            ProviderCapability(
                category=SkillCategory.BROWSER,
                actions=["workflow", "automate", "multi-step"],
                estimated_latency_ms=60000,
            ),
        ]

        self._registry = None

    def set_registry(self, registry) -> None:
        self._registry = registry

    async def initialize(self) -> bool:
        self.is_initialized = True
        self.is_available = True
        logger.info(f" {self.name} composite provider initialized")
        return True

    def get_categories(self) -> List[SkillCategory]:
        return [SkillCategory.TERMINAL, SkillCategory.BROWSER]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute multi-step automation workflow with dependency awareness."""
        start_time = time.time()
        context = context or {}

        try:
            # Parse task into CompositeStep objects
            steps = self._parse_workflow(task)
            results: Dict[str, Dict] = {}  # step_id -> result dict

            for i, step in enumerate(steps):
                logger.info(f" Step {i+1}/{len(steps)}: {step.task[:30]}...")

                # Check dependency satisfaction
                if step.depends_on:
                    deps_met = all(
                        dep_id in results and results[dep_id].get('success', False)
                        for dep_id in step.depends_on
                    )
                    if not deps_met:
                        logger.warning(f"Skipping step '{step.step_id}': unmet dependencies {step.depends_on}")
                        if not step.optional:
                            break
                        continue

                if self._registry:
                    step_result = await self._registry.execute(
                        step.category,
                        step.task,
                        {**context, 'step': i, 'previous_results': list(results.values())},
                    )
                    result_dict = {
                        'step': i + 1,
                        'step_id': step.step_id,
                        'action': step.task,
                        'success': step_result.success,
                        'output': step_result.output,
                    }
                    results[step.step_id] = result_dict

                    # Stop on failure unless marked as optional
                    if not step_result.success and not step.optional:
                        break

            execution_time = time.time() - start_time

            all_success = all(r['success'] for r in results.values()) if results else False

            return ProviderResult(
                success=all_success,
                output={
                    'workflow': task,
                    'steps': list(results.values()),
                    'total_steps': len(steps),
                    'completed_steps': len(results),
                },
                execution_time=execution_time,
                provider_name=self.name,
                category=SkillCategory.TERMINAL,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
            )

    def _parse_workflow(self, task: str) -> List[CompositeStep]:
        """Parse task into CompositeStep objects with dependency tracking."""
        steps: List[CompositeStep] = []

        # Split on ' then ' first (preserves order/dependency), then ', '
        parts = task.replace(' then ', '\x00').split('\x00')
        expanded = []
        for part in parts:
            expanded.extend(p.strip() for p in part.split(',') if p.strip())

        prev_step_id: Optional[str] = None
        for idx, part in enumerate(expanded):
            part_lower = part.lower()

            # Determine category
            if any(kw in part_lower for kw in ['browse', 'open url', 'website', 'search']):
                category = SkillCategory.BROWSER
            elif any(kw in part_lower for kw in ['click', 'type', 'screenshot', 'gui']):
                category = SkillCategory.COMPUTER_USE
            elif any(kw in part_lower for kw in ['run', 'execute', 'command', 'git', 'pip']):
                category = SkillCategory.TERMINAL
            elif any(kw in part_lower for kw in ['analyze', 'process', 'code']):
                category = SkillCategory.CODE_EXECUTION
            else:
                category = SkillCategory.TERMINAL  # Default

            step_id = f"step_{idx}"
            step = CompositeStep(
                provider_name=self.name,
                category=category,
                task=part,
                step_id=step_id,
                depends_on=[prev_step_id] if prev_step_id else None,
                optional='optional' in part_lower,
            )
            steps.append(step)
            prev_step_id = step_id

        if not steps:
            steps = [CompositeStep(
                provider_name=self.name,
                category=SkillCategory.TERMINAL,
                task=task,
                step_id="step_0",
            )]

        return steps


class FullStackAgentProvider(SkillProvider):
    """
    Composite: All providers combined for complex agent tasks.

    The most powerful composite - uses ALL available providers
    and learns which combination works best.
    """

    name = "fullstack-agent"
    version = "1.0.0"
    description = "Full-stack agent using all available providers"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=cat,
                actions=["auto", "smart", "adaptive"],
                estimated_latency_ms=120000,  # Complex tasks
            )
            for cat in SkillCategory
        ]

        self._registry = None
        self._swarm_intelligence = None

    def set_registry(self, registry) -> None:
        self._registry = registry

    def set_swarm_intelligence(self, si) -> None:
        self._swarm_intelligence = si

    async def initialize(self) -> bool:
        self.is_initialized = True
        self.is_available = True
        logger.info(f" {self.name} fullstack agent initialized")
        return True

    def get_categories(self) -> List[SkillCategory]:
        return list(SkillCategory)

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute using best available providers with learning."""
        start_time = time.time()
        context = context or {}

        try:
            # Analyze task to determine required capabilities
            required_categories = self._analyze_task(task)
            logger.info(f" Fullstack agent: Task requires {[c.value for c in required_categories]}")

            results = {}

            for category in required_categories:
                if self._registry:
                    # Registry will select best provider for this category
                    result = await self._registry.execute(category, task, context)
                    results[category.value] = {
                        'success': result.success,
                        'output': result.output,
                        'provider': result.provider_name,
                    }

                    # Update context with results for next step
                    context[f'{category.value}_result'] = result.output

                    # Record in swarm intelligence
                    if self._swarm_intelligence and result.success:
                        self._swarm_intelligence.deposit_success_signal(
                            agent=result.provider_name,
                            task_type=category.value,
                            execution_time=result.execution_time,
                        )

            execution_time = time.time() - start_time

            # Determine overall success
            all_success = all(r.get('success', False) for r in results.values())

            return ProviderResult(
                success=all_success,
                output={
                    'task': task,
                    'categories_used': [c.value for c in required_categories],
                    'results': results,
                },
                execution_time=execution_time,
                provider_name=self.name,
                confidence=0.9 if all_success else 0.3,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
            )

    def _analyze_task(self, task: str) -> List[SkillCategory]:
        """Analyze task to determine required categories (deduplicated)."""
        task_lower = task.lower()
        seen: set = set()
        categories: List[SkillCategory] = []

        # Check for category keywords
        category_keywords = {
            SkillCategory.BROWSER: ['browse', 'website', 'url', 'web', 'search online'],
            SkillCategory.TERMINAL: ['command', 'shell', 'terminal', 'git', 'run'],
            SkillCategory.CODE_EXECUTION: ['code', 'python', 'analyze', 'calculate'],
            SkillCategory.COMPUTER_USE: ['click', 'type', 'gui', 'desktop', 'application'],
            SkillCategory.FILE_OPERATIONS: ['file', 'read', 'write', 'save', 'load'],
            SkillCategory.DATA_EXTRACTION: ['extract', 'scrape', 'parse', 'data'],
            SkillCategory.WEB_SEARCH: ['search', 'find', 'lookup', 'research'],
        }

        for category, keywords in category_keywords.items():
            if category not in seen and any(kw in task_lower for kw in keywords):
                seen.add(category)
                categories.append(category)

        # Default to web search + code if nothing detected
        if not categories:
            categories = [SkillCategory.WEB_SEARCH, SkillCategory.CODE_EXECUTION]

        return categories
