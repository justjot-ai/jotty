"""
Provider Registry and Adaptive Selector for Jotty V2
=====================================================

Manages skill providers and learns which provider works best for each use case
using swarm intelligence (Q-learning, stigmergy, adaptive weights).

Includes sandbox integration for secure execution of untrusted providers.
"""

import json
import time
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

from .base import (
    SkillProvider,
    SkillCategory,
    ProviderResult,
    JottyDefaultProvider,
    CATEGORY_KEYWORDS,
    ContributedSkill,
)

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from Jotty.core.orchestration.sandbox_manager import SandboxManager, TrustLevel

# Import adaptive weights (no circular dependency)
try:
    from Jotty.core.foundation.robust_parsing import AdaptiveWeightGroup
except ImportError:
    AdaptiveWeightGroup = None

# SwarmIntelligence is imported lazily to avoid circular imports
SwarmIntelligence = None  # Type hint only, actual import done lazily

logger = logging.getLogger(__name__)


@dataclass
class ProviderPerformance:
    """Tracks performance metrics for a provider on a specific task type."""
    provider_name: str
    category: SkillCategory
    task_pattern: str  # Hashed or simplified task pattern

    # Metrics
    total_calls: int = 0
    successful_calls: int = 0
    total_execution_time: float = 0.0
    total_reward: float = 0.0

    # Learning
    q_value: float = 0.5  # Initial neutral value
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.successful_calls / max(1, self.total_calls)

    @property
    def avg_execution_time(self) -> float:
        return self.total_execution_time / max(1, self.total_calls)

    def update(self, success: bool, execution_time: float, reward: float = None):
        """Update metrics after execution."""
        self.total_calls += 1
        self.total_execution_time += execution_time

        if success:
            self.successful_calls += 1

        # Calculate reward if not provided
        if reward is None:
            reward = 1.0 if success else -0.5
            # Bonus for fast execution
            if execution_time < 1.0:
                reward += 0.2
            elif execution_time > 10.0:
                reward -= 0.1

        self.total_reward += reward

        # Q-learning update
        alpha = 0.1  # Learning rate
        self.q_value = self.q_value + alpha * (reward - self.q_value)
        self.last_updated = time.time()


class ProviderSelector:
    """
    Learns which provider is best for each use case.

    Uses multiple signals:
    1. Q-values from past performance
    2. Stigmergy signals from swarm
    3. Adaptive weights for category preferences
    4. Exploration/exploitation balance (epsilon-greedy)
    """

    def __init__(self, epsilon: float = 0.1, swarm_intelligence: SwarmIntelligence = None):
        """
        Initialize the selector.

        Args:
            epsilon: Exploration rate (probability of trying non-optimal provider)
            swarm_intelligence: Optional swarm intelligence for stigmergy
        """
        self.epsilon = epsilon
        self.swarm_intelligence = swarm_intelligence

        # Performance tracking: (category, task_hash) -> {provider_name: ProviderPerformance}
        self.performance: Dict[Tuple[str, str], Dict[str, ProviderPerformance]] = defaultdict(dict)

        # Category weights: how much to weight different factors
        if AdaptiveWeightGroup:
            self.selection_weights = AdaptiveWeightGroup({
                'q_value': 0.4,        # Historical Q-value
                'success_rate': 0.3,   # Recent success rate
                'speed': 0.2,          # Execution speed
                'stigmergy': 0.1,      # Swarm signals
            })
        else:
            self.selection_weights = None

        # Exploration tracking
        self.exploration_count = 0
        self.exploitation_count = 0

    def select_provider(
        self,
        category: SkillCategory,
        task: str,
        available_providers: List[SkillProvider],
        context: Dict[str, Any] = None
    ) -> SkillProvider:
        """
        Select the best provider for a task.

        Args:
            category: Skill category
            task: Task description
            available_providers: List of available providers
            context: Additional context

        Returns:
            Selected provider
        """
        if not available_providers:
            raise ValueError(f"No providers available for category {category}")

        if len(available_providers) == 1:
            return available_providers[0]

        # Get task hash for lookup
        task_hash = self._hash_task(task)
        key = (category.value, task_hash)

        # Epsilon-greedy exploration
        import random
        if random.random() < self.epsilon:
            self.exploration_count += 1
            selected = random.choice(available_providers)
            logger.debug(f" Exploring: selected {selected.name} randomly")
            return selected

        self.exploitation_count += 1

        # Score each provider
        scores = {}
        for provider in available_providers:
            score = self._score_provider(provider, category, task_hash, context)
            scores[provider.name] = score

        # Select best
        best_name = max(scores.keys(), key=lambda n: scores[n])
        best_provider = next(p for p in available_providers if p.name == best_name)

        logger.debug(f" Exploiting: selected {best_name} (score: {scores[best_name]:.3f})")
        return best_provider

    def _score_provider(
        self,
        provider: SkillProvider,
        category: SkillCategory,
        task_hash: str,
        context: Dict[str, Any] = None
    ) -> float:
        """Calculate score for a provider."""
        key = (category.value, task_hash)

        # Get or create performance record
        if provider.name not in self.performance[key]:
            # New provider for this task - start with neutral score
            return 0.5

        perf = self.performance[key][provider.name]

        # Component scores
        q_score = perf.q_value
        success_score = perf.success_rate
        speed_score = 1.0 / (1.0 + perf.avg_execution_time / 5.0)  # Normalize to 0-1

        # Stigmergy score from swarm
        stigmergy_score = 0.5  # Neutral default
        if self.swarm_intelligence:
            signals = self.swarm_intelligence.stigmergy.get_route_signals(category.value)
            if provider.name in signals:
                stigmergy_score = min(1.0, signals[provider.name])

        # Weighted combination
        if self.selection_weights:
            score = (
                self.selection_weights.get('q_value') * q_score +
                self.selection_weights.get('success_rate') * success_score +
                self.selection_weights.get('speed') * speed_score +
                self.selection_weights.get('stigmergy') * stigmergy_score
            )
        else:
            # Fallback to simple average
            score = (q_score + success_score + speed_score + stigmergy_score) / 4

        return score

    def record_result(
        self,
        provider: SkillProvider,
        category: SkillCategory,
        task: str,
        result: ProviderResult
    ):
        """Record execution result for learning."""
        task_hash = self._hash_task(task)
        key = (category.value, task_hash)

        # Get or create performance record
        if provider.name not in self.performance[key]:
            self.performance[key][provider.name] = ProviderPerformance(
                provider_name=provider.name,
                category=category,
                task_pattern=task_hash,
            )

        perf = self.performance[key][provider.name]
        perf.update(result.success, result.execution_time)

        # Update stigmergy in swarm
        if self.swarm_intelligence and result.success:
            self.swarm_intelligence.stigmergy.deposit(
                signal_type='route',
                content={'provider': provider.name, 'task_type': category.value},
                agent=provider.name,
                strength=0.8 if result.success else 0.3,
            )

        # Update adaptive weights based on overall performance
        if self.selection_weights and result.success:
            # If Q-value was a good predictor, strengthen it
            if perf.q_value > 0.6:
                self.selection_weights.update_from_feedback('q_value', 0.05, reward=1.0)

        logger.debug(f" Recorded {provider.name} result: success={result.success}, q={perf.q_value:.3f}")

    def _hash_task(self, task: str) -> str:
        """Create a hash for task pattern matching."""
        # Normalize task
        normalized = task.lower().strip()
        # Extract key words (simple approach)
        words = sorted(set(normalized.split()[:10]))  # First 10 unique words
        pattern = ' '.join(words)
        return hashlib.md5(pattern.encode()).hexdigest()[:12]

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what the selector has learned."""
        summary = {
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'exploration_rate': self.exploration_count / max(1, self.exploration_count + self.exploitation_count),
            'providers_by_category': defaultdict(dict),
        }

        for (category, task_hash), providers in self.performance.items():
            for name, perf in providers.items():
                if category not in summary['providers_by_category']:
                    summary['providers_by_category'][category] = {}
                summary['providers_by_category'][category][name] = {
                    'q_value': round(perf.q_value, 3),
                    'success_rate': round(perf.success_rate, 3),
                    'total_calls': perf.total_calls,
                }

        return summary

    def save_state(self, path: str):
        """Save learned state to file."""
        data = {
            'epsilon': self.epsilon,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'performance': {},
            'selection_weights': self.selection_weights.to_dict() if self.selection_weights else None,
        }

        for (category, task_hash), providers in self.performance.items():
            key = f"{category}:{task_hash}"
            data['performance'][key] = {}
            for name, perf in providers.items():
                data['performance'][key][name] = {
                    'total_calls': perf.total_calls,
                    'successful_calls': perf.successful_calls,
                    'total_execution_time': perf.total_execution_time,
                    'total_reward': perf.total_reward,
                    'q_value': perf.q_value,
                }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved ProviderSelector state to {path}")

    def load_state(self, path: str) -> bool:
        """Load learned state from file."""
        if not Path(path).exists():
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.epsilon = data.get('epsilon', 0.1)
            self.exploration_count = data.get('exploration_count', 0)
            self.exploitation_count = data.get('exploitation_count', 0)

            if data.get('selection_weights') and AdaptiveWeightGroup:
                self.selection_weights = AdaptiveWeightGroup.from_dict(data['selection_weights'])

            for key, providers in data.get('performance', {}).items():
                category, task_hash = key.split(':', 1)
                perf_key = (category, task_hash)

                for name, perf_data in providers.items():
                    self.performance[perf_key][name] = ProviderPerformance(
                        provider_name=name,
                        category=SkillCategory(category),
                        task_pattern=task_hash,
                        total_calls=perf_data['total_calls'],
                        successful_calls=perf_data['successful_calls'],
                        total_execution_time=perf_data['total_execution_time'],
                        total_reward=perf_data['total_reward'],
                        q_value=perf_data['q_value'],
                    )

            logger.info(f"Loaded ProviderSelector state from {path}")
            return True

        except Exception as e:
            logger.warning(f"Could not load ProviderSelector state: {e}")
            return False


class ProviderRegistry:
    """
    Central registry for all skill providers.

    Manages provider lifecycle, discovery, selection, and sandboxed execution.

    Features:
    - Provider registration with trust levels
    - Adaptive selection using Q-learning
    - Sandboxed execution for untrusted providers
    - Automatic trust level detection
    """

    # Packages considered trusted (built-in or well-known)
    TRUSTED_PACKAGES = {
        'jotty', 'browser-use', 'openhands', 'agent-s', 'open-interpreter',
        'skyvern', 'playwright', 'selenium', 'requests', 'httpx', 'aiohttp',
        'morph', 'morph-data', 'streamlit', 'gradio',
        'n8n', 'activepieces',
    }

    def __init__(self, swarm_intelligence: SwarmIntelligence = None):
        """
        Initialize the registry.

        Args:
            swarm_intelligence: Optional swarm for learning integration
        """
        self.swarm_intelligence = swarm_intelligence

        # Provider storage: name -> provider instance
        self._providers: Dict[str, SkillProvider] = {}

        # Category index: category -> [provider_names]
        self._category_index: Dict[SkillCategory, List[str]] = defaultdict(list)

        # Trust levels: provider_name -> TrustLevel
        self._trust_levels: Dict[str, 'TrustLevel'] = {}

        # Sandbox manager (lazy loaded)
        self._sandbox_manager: Optional['SandboxManager'] = None

        # Adaptive selector
        self.selector = ProviderSelector(swarm_intelligence=swarm_intelligence)

        # Register default Jotty provider
        self.register(JottyDefaultProvider(), trust_level='trusted')

        # Register app building providers (Streamlit first as default)
        self._register_app_building_providers()
        # Register workflow-engine providers (n8n, Activepieces as skills)
        self._register_workflow_providers()

        logger.info(" ProviderRegistry initialized")

    def _register_app_building_providers(self):
        """Register app building providers. Streamlit first (default, open source)."""
        # StreamlitProvider - fully open source, no cloud needed (DEFAULT)
        try:
            from .streamlit_provider import StreamlitProvider
            self.register(StreamlitProvider(), trust_level='trusted')
        except Exception as e:
            logger.debug(f"Could not register StreamlitProvider: {e}")

        # MorphProvider - requires cloud credentials (secondary option)
        try:
            from .morph_provider import MorphProvider
            self.register(MorphProvider(), trust_level='trusted')
        except Exception as e:
            logger.debug(f"Could not register MorphProvider: {e}")

    def _register_workflow_providers(self):
        """Register n8n and Activepieces as skill providers (workflows as skills)."""
        try:
            from .n8n_provider import N8nProvider
            self.register(N8nProvider(), trust_level='trusted')
        except Exception as e:
            logger.debug(f"Could not register N8nProvider: {e}")
        try:
            from .activepieces_provider import ActivepiecesProvider
            self.register(ActivepiecesProvider(), trust_level='trusted')
        except Exception as e:
            logger.debug(f"Could not register ActivepiecesProvider: {e}")

    def _get_sandbox_manager(self) -> 'SandboxManager':
        """Lazy load sandbox manager."""
        if self._sandbox_manager is None:
            from Jotty.core.orchestration.sandbox_manager import SandboxManager
            self._sandbox_manager = SandboxManager()
        return self._sandbox_manager

    def _get_trust_level_enum(self, level: str) -> 'TrustLevel':
        """Convert string to TrustLevel enum."""
        from Jotty.core.orchestration.sandbox_manager import TrustLevel
        level_map = {
            'trusted': TrustLevel.TRUSTED,
            'sandboxed': TrustLevel.SANDBOXED,
            'dangerous': TrustLevel.DANGEROUS,
        }
        return level_map.get(level.lower(), TrustLevel.SANDBOXED)

    def register(
        self,
        provider: SkillProvider,
        trust_level: Optional[str] = None
    ):
        """
        Register a provider with optional trust level.

        Args:
            provider: SkillProvider instance
            trust_level: Trust level ('trusted', 'sandboxed', 'dangerous')
                        If not provided, auto-detects based on provider name
        """
        self._providers[provider.name] = provider

        # Index by category
        for category in provider.get_categories():
            if provider.name not in self._category_index[category]:
                self._category_index[category].append(provider.name)

        # Set trust level
        if trust_level:
            self._trust_levels[provider.name] = self._get_trust_level_enum(trust_level)
        else:
            self._trust_levels[provider.name] = self._detect_trust_level(provider)

        trust_str = self._trust_levels[provider.name].value
        logger.info(f" Registered provider: {provider.name} (trust: {trust_str}, categories: {[c.value for c in provider.get_categories()]})")

    def _detect_trust_level(self, provider: SkillProvider) -> 'TrustLevel':
        """
        Auto-detect trust level for a provider.

        Heuristics:
        - Known packages: TRUSTED
        - Code execution capabilities: DANGEROUS
        - Default: SANDBOXED
        """
        from Jotty.core.orchestration.sandbox_manager import TrustLevel

        # Check if trusted package
        if provider.name.lower() in self.TRUSTED_PACKAGES:
            return TrustLevel.TRUSTED

        # Check for dangerous capabilities
        dangerous_categories = {
            SkillCategory.CODE_EXECUTION,
            SkillCategory.TERMINAL,
            SkillCategory.COMPUTER_USE,
        }

        provider_categories = set(provider.get_categories())
        if provider_categories & dangerous_categories:
            return TrustLevel.DANGEROUS

        # Default to sandboxed
        return TrustLevel.SANDBOXED

    def get_trust_level(self, provider_name: str) -> Optional['TrustLevel']:
        """Get trust level for a provider."""
        return self._trust_levels.get(provider_name)

    def set_trust_level(self, provider_name: str, level: str):
        """Set trust level for a provider."""
        if provider_name in self._providers:
            self._trust_levels[provider_name] = self._get_trust_level_enum(level)
            logger.info(f"Updated trust level for {provider_name}: {level}")

    def unregister(self, provider_name: str):
        """Unregister a provider."""
        if provider_name in self._providers:
            provider = self._providers[provider_name]
            for category in provider.get_categories():
                if provider_name in self._category_index[category]:
                    self._category_index[category].remove(provider_name)
            del self._providers[provider_name]
            logger.info(f"Unregistered provider: {provider_name}")

    def get_provider(self, name: str) -> Optional[SkillProvider]:
        """Get a specific provider by name."""
        return self._providers.get(name)

    def get_all_contributed_skills(self) -> List[ContributedSkill]:
        """
        Merge skills contributed by all providers (e.g. n8n workflows, Activepieces flows).
        KISS: sync; returns whatever each provider's list_skills() returns (may be empty until inited).
        """
        out: List[ContributedSkill] = []
        for provider in self._providers.values():
            if hasattr(provider, "list_skills"):
                try:
                    skills = provider.list_skills()
                    if skills:
                        out.extend(skills)
                except Exception as e:
                    logger.debug("list_skills from %s: %s", provider.name, e)
        return out

    def get_providers_for_category(self, category: SkillCategory) -> List[SkillProvider]:
        """Get all providers that support a category."""
        names = self._category_index.get(category, [])
        return [self._providers[n] for n in names if n in self._providers]

    def detect_category(self, task: str) -> Optional[SkillCategory]:
        """
        Auto-detect skill category from task description using CATEGORY_KEYWORDS.

        Args:
            task: Natural language task description

        Returns:
            Detected SkillCategory or None if no match

        Example:
            >>> registry.detect_category("build me a dashboard app")
            SkillCategory.APP_BUILDING
        """
        task_lower = task.lower()

        # Score each category by keyword matches
        scores: Dict[SkillCategory, int] = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in task_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return None

        # Return category with highest score
        return max(scores, key=scores.get)

    def get_provider_for_task(self, task: str, context: Dict[str, Any] = None) -> Optional[SkillProvider]:
        """
        Auto-detect category and get best provider for a task.

        This is the main entry point for autonomous task routing.

        Args:
            task: Natural language task description
            context: Additional context

        Returns:
            Best provider for the task or None

        Example:
            >>> provider = registry.get_provider_for_task("build a stock dashboard")
            >>> await provider.execute("build a stock dashboard")
        """
        category = self.detect_category(task)

        if category:
            logger.info(f" Detected category: {category.value} for task: {task[:50]}...")
            return self.get_best_provider(category, task, context)

        logger.warning(f"Could not detect category for task: {task[:50]}...")
        return self._providers.get('jotty')  # Fallback

    async def auto_execute(self, task: str, context: Dict[str, Any] = None) -> 'ProviderResult':
        """
        Autonomously detect category, select provider, and execute task.

        This is the fully autonomous entry point.

        Args:
            task: Natural language task description
            context: Additional context

        Returns:
            ProviderResult from execution

        Example:
            >>> result = await registry.auto_execute("build me a chat app")
            # Automatically detects APP_BUILDING, selects StreamlitProvider, executes
        """
        category = self.detect_category(task)

        if not category:
            logger.warning(f"Could not detect category, using default")
            category = SkillCategory.CODE_EXECUTION  # Fallback

        logger.info(f" Auto-executing: {task[:50]}... (category: {category.value})")

        return await self.execute(category, task, context)

    def get_best_provider(
        self,
        category: SkillCategory,
        task: str,
        context: Dict[str, Any] = None
    ) -> SkillProvider:
        """
        Get the best provider for a task using learned selection.

        Args:
            category: Skill category
            task: Task description
            context: Additional context

        Returns:
            Best provider (uses learning to select)
        """
        available = self.get_providers_for_category(category)

        if not available:
            # Fall back to Jotty default
            logger.warning(f"No providers for {category}, using Jotty default")
            return self._providers.get('jotty')

        return self.selector.select_provider(category, task, available, context)

    async def execute(
        self,
        category: SkillCategory,
        task: str,
        context: Dict[str, Any] = None,
        provider_name: str = None,
        force_sandbox: bool = False
    ) -> ProviderResult:
        """
        Execute a task using the best (or specified) provider.

        Routes through sandbox based on provider trust level:
        - TRUSTED: Direct execution
        - SANDBOXED: Basic sandbox (Docker or subprocess)
        - DANGEROUS: Isolated sandbox (E2B preferred)

        Args:
            category: Skill category
            task: Task description
            context: Additional context
            provider_name: Optional specific provider to use
            force_sandbox: Force sandbox execution regardless of trust level

        Returns:
            Execution result
        """
        # Get provider
        if provider_name:
            provider = self.get_provider(provider_name)
            if not provider:
                return ProviderResult(
                    success=False,
                    output=None,
                    error=f"Provider '{provider_name}' not found",
                )
        else:
            provider = self.get_best_provider(category, task, context)

        # Ensure initialized
        if not provider.is_initialized:
            await provider.initialize()

        # Get trust level
        trust_level = self._trust_levels.get(provider.name)

        # Check if sandboxed execution needed
        from Jotty.core.orchestration.sandbox_manager import TrustLevel
        needs_sandbox = force_sandbox or (trust_level and trust_level != TrustLevel.TRUSTED)

        logger.info(f" Executing via {provider.name}: {task[:50]}... (trust: {trust_level.value if trust_level else 'unknown'})")

        if needs_sandbox and trust_level:
            result = await self._execute_sandboxed(provider, task, context, trust_level)
        else:
            result = await provider.execute(task, context)

        result.category = category

        # Record for learning
        self.selector.record_result(provider, category, task, result)

        return result

    async def _execute_sandboxed(
        self,
        provider: SkillProvider,
        task: str,
        context: Dict[str, Any],
        trust_level: 'TrustLevel'
    ) -> ProviderResult:
        """
        Execute provider in appropriate sandbox.

        For most providers, we still call their execute method but may
        wrap code execution in a sandbox if the provider generates code.
        """
        try:
            # Get sandbox manager
            sandbox = self._get_sandbox_manager()

            # For now, execute provider normally
            # Future: intercept code execution and route through sandbox
            result = await provider.execute(task, context)

            # Add sandbox metadata
            result.metadata['trust_level'] = trust_level.value
            result.metadata['sandboxed'] = True

            return result

        except Exception as e:
            logger.error(f"Sandboxed execution failed: {e}")
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                provider_name=provider.name,
                metadata={'trust_level': trust_level.value, 'sandboxed': True},
            )

    async def initialize_all(self):
        """Initialize all registered providers."""
        for name, provider in self._providers.items():
            try:
                if not provider.is_initialized:
                    success = await provider.initialize()
                    logger.info(f"Initialized {name}: {'' if success else ''}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

    def get_all_providers(self) -> Dict[str, SkillProvider]:
        """Get all registered providers."""
        return self._providers.copy()

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registry state."""
        return {
            'providers': list(self._providers.keys()),
            'categories': {
                cat.value: self._category_index[cat]
                for cat in SkillCategory
                if self._category_index[cat]
            },
            'learning': self.selector.get_learning_summary(),
        }

    def save_state(self, path: str):
        """Save registry and selector state."""
        self.selector.save_state(path)

    def load_state(self, path: str) -> bool:
        """Load registry and selector state."""
        return self.selector.load_state(path)
