"""
LearningCoordinator - Unified Learning Management for Jotty
============================================================

Consolidates all learning functionality:
- Q-learning (LLMQPredictor) with tiered memory
- TD(λ) temporal difference learning
- Per-agent learners and memories
- Session registry with domain-based retrieval
- Integration with TransferableLearningStore

This replaces:
- core.learning.learning_manager.LearningManager (DEPRECATED)
- core.orchestration.v2.rl_learning_manager.LearningManager

Usage:
    from core.learning.learning_coordinator import LearningCoordinator

    coordinator = LearningCoordinator(config)
    coordinator.initialize()  # Auto-loads latest learning

    # Per-agent learning
    q_learner = coordinator.get_agent_learner("Planner")
    memory = coordinator.get_agent_memory("Planner")

    # Record experiences
    coordinator.record_experience("Planner", state, action, reward)

    # Get learned context for prompts
    context = coordinator.get_learned_context(state, action)

    # Save all learning
    coordinator.save_all(episode_count=10, avg_reward=0.75)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LearningSession:
    """Metadata for a learning session."""
    session_id: str
    created_at: float
    updated_at: float
    episode_count: int
    total_experiences: int
    domains: List[str]
    agents: List[str]
    avg_reward: float
    path: str


@dataclass
class LearningUpdate:
    """Result of a learning update."""
    actor: str
    reward: float
    q_value: Optional[float] = None
    td_error: Optional[float] = None


class LearningCoordinator:
    """
    Unified learning coordinator for Jotty MAS.

    Combines:
    - Q-learning with tiered memory (working, semantic clusters, long-term)
    - TD(λ) temporal difference learning
    - Per-agent Q-learners and memories
    - Session registry for persistence
    - Domain-based transfer learning

    Responsibilities:
    - Manage Q-learner and TD(λ) instances
    - Coordinate per-agent learning state
    - Handle session persistence and registry
    - Provide learned context for agent prompts
    - Support domain-based learning retrieval
    """

    def __init__(self, config, base_dir: str = None):
        """
        Initialize Learning Coordinator.

        Args:
            config: JottyConfig or SwarmConfig with learning settings
            base_dir: Base directory for learning storage
        """
        self.config = config
        self.base_dir = Path(base_dir or getattr(config, 'output_base_dir', './outputs'))
        self.learning_dir = self.base_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Session registry
        self.registry_path = self.learning_dir / "registry.json"
        self.registry: Dict[str, LearningSession] = {}

        # Current session
        self.session_id = f"session_{int(time.time())}"
        self.session_dir = self.learning_dir / self.session_id

        # Per-agent learners and memories
        self._agent_q_learners: Dict[str, Any] = {}  # agent_name -> LLMQPredictor
        self._agent_memories: Dict[str, Any] = {}    # agent_name -> SimpleFallbackMemory

        # Shared learners (cross-agent)
        self._shared_q_learner = None
        self._td_lambda_learner = None

        # Domain index for transfer learning
        self._domain_index: Dict[str, List[str]] = {}  # domain -> [session_ids]
        self._current_domains: List[str] = []

        # Initialize core learners
        self._init_core_learners()

        # Load registry
        self._load_registry()

        logger.info(f"LearningCoordinator initialized: {self.learning_dir}")

    def _init_core_learners(self):
        """Initialize Q-learner and TD(λ) learner."""
        # Initialize shared Q-learner
        try:
            from .q_learning import LLMQPredictor
            self._shared_q_learner = LLMQPredictor(self.config)
            logger.info("Q-Learning initialized (LearningCoordinator)")
        except ImportError as e:
            logger.warning(f"Q-Learning not available: {e}")

        # Initialize TD(λ) learner if RL enabled
        if getattr(self.config, 'enable_rl', False):
            try:
                from .learning import TDLambdaLearner
                self._td_lambda_learner = TDLambdaLearner(self.config)
                logger.info("TD(λ) Learner initialized (LearningCoordinator)")
            except ImportError as e:
                logger.warning(f"TD(λ) Learning not available: {e}")

    # =========================================================================
    # Session Registry Management
    # =========================================================================

    def _load_registry(self):
        """Load learning session registry from disk."""
        if not self.registry_path.exists():
            return

        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)

            for sid, sdata in data.get('sessions', {}).items():
                self.registry[sid] = LearningSession(
                    session_id=sid,
                    created_at=sdata.get('created_at', 0),
                    updated_at=sdata.get('updated_at', 0),
                    episode_count=sdata.get('episode_count', 0),
                    total_experiences=sdata.get('total_experiences', 0),
                    domains=sdata.get('domains', []),
                    agents=sdata.get('agents', []),
                    avg_reward=sdata.get('avg_reward', 0),
                    path=sdata.get('path', '')
                )

            self._domain_index = data.get('domain_index', {})
            logger.info(f"Loaded registry: {len(self.registry)} sessions")
        except Exception as e:
            logger.warning(f"Could not load registry: {e}")

    def _save_registry(self):
        """Save learning session registry to disk."""
        data = {
            'sessions': {
                sid: {
                    'session_id': s.session_id,
                    'created_at': s.created_at,
                    'updated_at': s.updated_at,
                    'episode_count': s.episode_count,
                    'total_experiences': s.total_experiences,
                    'domains': s.domains,
                    'agents': s.agents,
                    'avg_reward': s.avg_reward,
                    'path': s.path
                }
                for sid, s in self.registry.items()
            },
            'domain_index': self._domain_index,
            'last_updated': time.time()
        }

        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # Initialization and Loading
    # =========================================================================

    def initialize(self, auto_load: bool = True) -> bool:
        """
        Initialize coordinator, optionally loading previous learning.

        Args:
            auto_load: If True, auto-load latest learning session

        Returns:
            True if previous learning was loaded
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create 'latest' symlink
        latest_link = self.learning_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(self.session_id)
        except OSError:
            pass  # Symlink creation may fail on some systems

        if auto_load and self.registry:
            return self.load_latest()

        return False

    def load_latest(self) -> bool:
        """Load the most recent learning session."""
        if not self.registry:
            logger.info("No previous learning sessions found")
            return False

        latest = max(self.registry.values(), key=lambda s: s.updated_at)
        return self.load_session(latest.session_id)

    def load_session(self, session_id: str) -> bool:
        """
        Load a specific learning session.

        Args:
            session_id: Session ID to load

        Returns:
            True if loaded successfully
        """
        if session_id not in self.registry:
            logger.warning(f"Session {session_id} not found in registry")
            return False

        session = self.registry[session_id]
        session_path = Path(session.path)

        if not session_path.exists():
            logger.warning(f"Session path does not exist: {session_path}")
            return False

        loaded = False

        # Load shared Q-learner
        shared_q_path = session_path / "shared_q_learning.json"
        if shared_q_path.exists() and self._shared_q_learner:
            try:
                self._shared_q_learner.load_state(str(shared_q_path))
                loaded = True
            except Exception as e:
                logger.warning(f"Could not load shared Q-learner: {e}")

        # Load per-agent learners
        agents_dir = session_path / "agents"
        if agents_dir.exists():
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    agent_name = agent_dir.name

                    # Load agent Q-learner
                    q_path = agent_dir / "q_learning.json"
                    if q_path.exists():
                        learner = self.get_agent_learner(agent_name)
                        try:
                            learner.load_state(str(q_path))
                            loaded = True
                        except Exception as e:
                            logger.debug(f"Could not load Q-learner for {agent_name}: {e}")

                    # Load agent memory
                    mem_path = agent_dir / "memory.json"
                    if mem_path.exists():
                        memory = self.get_agent_memory(agent_name)
                        try:
                            memory.load(str(mem_path))
                            loaded = True
                        except Exception as e:
                            logger.debug(f"Could not load memory for {agent_name}: {e}")

        if loaded:
            logger.info(f"Loaded learning from session: {session_id}")

        return loaded

    def load_domain_learning(self, domain: str, top_k: int = 3) -> bool:
        """
        Load learning from sessions that worked on similar domain.

        Args:
            domain: Domain to search for (e.g., "microservices", "ml")
            top_k: Number of best sessions to load from

        Returns:
            True if any learning was loaded
        """
        matching_sessions = []

        for session_id, session in self.registry.items():
            for d in session.domains:
                if domain.lower() in d.lower() or d.lower() in domain.lower():
                    matching_sessions.append(session)
                    break

        if not matching_sessions:
            logger.info(f"No sessions found for domain: {domain}")
            return False

        # Sort by avg_reward (best first)
        matching_sessions.sort(key=lambda s: s.avg_reward, reverse=True)

        loaded = False
        for session in matching_sessions[:top_k]:
            if self.load_session(session.session_id):
                loaded = True
                logger.info(f"Loaded domain learning: {session.session_id} (reward: {session.avg_reward:.3f})")

        return loaded

    # =========================================================================
    # Per-Agent Learner Access
    # =========================================================================

    def get_agent_learner(self, agent_name: str) -> Any:
        """
        Get or create Q-learner for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            LLMQPredictor instance for this agent
        """
        if agent_name not in self._agent_q_learners:
            try:
                from .q_learning import LLMQPredictor
                self._agent_q_learners[agent_name] = LLMQPredictor(self.config)
                logger.debug(f"Created Q-learner for agent: {agent_name}")
            except ImportError:
                # Return a no-op learner
                self._agent_q_learners[agent_name] = _NoOpLearner()

        return self._agent_q_learners[agent_name]

    def get_agent_memory(self, agent_name: str) -> Any:
        """
        Get or create memory for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            SimpleFallbackMemory instance for this agent
        """
        if agent_name not in self._agent_memories:
            try:
                from ..memory.fallback_memory import SimpleFallbackMemory
                self._agent_memories[agent_name] = SimpleFallbackMemory()
                logger.debug(f"Created memory for agent: {agent_name}")
            except ImportError:
                # Return a no-op memory
                self._agent_memories[agent_name] = _NoOpMemory()

        return self._agent_memories[agent_name]

    def get_shared_learner(self) -> Any:
        """Get shared Q-learner (cross-agent)."""
        if self._shared_q_learner is None:
            try:
                from .q_learning import LLMQPredictor
                self._shared_q_learner = LLMQPredictor(self.config)
            except ImportError:
                self._shared_q_learner = _NoOpLearner()

        return self._shared_q_learner

    # =========================================================================
    # Q-Value Prediction and Learning
    # =========================================================================

    def predict_q_value(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        goal: str = ""
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Predict Q-value for a state-action pair.

        Args:
            state: Current state dict
            action: Action dict
            goal: Goal description

        Returns:
            (q_value, confidence, alternative_suggestion)
        """
        if not self._shared_q_learner:
            return 0.5, 0.1, None

        try:
            return self._shared_q_learner.predict_q_value(state, action, goal)
        except Exception as e:
            logger.warning(f"Q-value prediction failed: {e}")
            return 0.5, 0.1, None

    def record_experience(
        self,
        agent_name: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False,
        domain: str = None
    ) -> LearningUpdate:
        """
        Record an experience for an agent.

        Args:
            agent_name: Name of the agent
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode is complete
            domain: Optional domain tag for indexing

        Returns:
            LearningUpdate with results
        """
        # Record in agent's Q-learner
        learner = self.get_agent_learner(agent_name)
        try:
            learner.add_experience(state, action, reward)
        except Exception as e:
            logger.debug(f"Could not add experience to agent learner: {e}")

        # Also record in shared learner with agent context
        if self._shared_q_learner:
            shared_state = {**state, 'agent': agent_name}
            try:
                self._shared_q_learner.record_outcome(shared_state, action, reward, next_state, done)
            except Exception as e:
                logger.debug(f"Could not record in shared learner: {e}")

        # Track domain
        if domain and domain not in self._current_domains:
            self._current_domains.append(domain)

        # Get Q-value for update result
        q_value = None
        if self._shared_q_learner:
            try:
                q_value, _, _ = self._shared_q_learner.predict_q_value(state, action)
            except Exception:
                pass

        return LearningUpdate(
            actor=agent_name,
            reward=reward,
            q_value=q_value
        )

    def record_outcome(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False
    ) -> LearningUpdate:
        """
        Record outcome (alias for anonymous agent).

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode is complete

        Returns:
            LearningUpdate with results
        """
        actor = action.get('actor', 'unknown')
        return self.record_experience(actor, state, action, reward, next_state, done)

    def update_td_lambda(
        self,
        trajectory: list,
        final_reward: float,
        gamma: float = 0.99,
        lambda_trace: float = 0.95
    ) -> None:
        """
        Perform TD(λ) update on trajectory.

        Args:
            trajectory: List of (state, action, reward) tuples
            final_reward: Final episode reward
            gamma: Discount factor
            lambda_trace: Eligibility trace decay
        """
        if not self._td_lambda_learner:
            return

        try:
            self._td_lambda_learner.update(trajectory, final_reward, gamma, lambda_trace)
        except Exception as e:
            logger.error(f"TD(λ) update failed: {e}")

    # =========================================================================
    # Context and Summaries
    # =========================================================================

    def get_learned_context(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get learned context to inject into agent prompts.

        This is how learning manifests in LLM agents.

        Args:
            state: Current state
            action: Optional action being considered

        Returns:
            Natural language lessons learned
        """
        if not self._shared_q_learner:
            return ""

        try:
            return self._shared_q_learner.get_learned_context(state, action)
        except Exception as e:
            logger.warning(f"Failed to get learned context: {e}")
            return ""

    def get_q_table_summary(self) -> str:
        """Get summary of Q-table for logging."""
        if not self._shared_q_learner:
            return "Q-learner not available"

        try:
            if hasattr(self._shared_q_learner, 'get_q_table_summary'):
                return self._shared_q_learner.get_q_table_summary()
            elif hasattr(self._shared_q_learner, 'experience_buffer'):
                return f"Q-learner active: {len(self._shared_q_learner.experience_buffer)} experiences"
            return "Q-learner active"
        except Exception as e:
            return f"Q-table summary failed: {e}"

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of current learning state."""
        summary = {
            'session_id': self.session_id,
            'session_dir': str(self.session_dir),
            'total_sessions': len(self.registry),
            'agents': list(self._agent_q_learners.keys()),
            'domains_indexed': list(self._domain_index.keys()),
        }

        if self._shared_q_learner and hasattr(self._shared_q_learner, 'get_q_table_stats'):
            summary['shared_q_stats'] = self._shared_q_learner.get_q_table_stats()

        summary['per_agent_stats'] = {}
        for agent_name, learner in self._agent_q_learners.items():
            if hasattr(learner, 'get_q_table_stats'):
                summary['per_agent_stats'][agent_name] = learner.get_q_table_stats()

        return summary

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available learning sessions."""
        from datetime import datetime
        sessions = []
        for session in sorted(self.registry.values(), key=lambda s: s.updated_at, reverse=True):
            sessions.append({
                'session_id': session.session_id,
                'created': datetime.fromtimestamp(session.created_at).isoformat(),
                'episodes': session.episode_count,
                'experiences': session.total_experiences,
                'domains': session.domains,
                'agents': session.agents,
                'avg_reward': session.avg_reward
            })
        return sessions

    # =========================================================================
    # Memory Management
    # =========================================================================

    def promote_demote_memories(self, episode_reward: float) -> None:
        """Promote/demote memories based on episode performance."""
        if not self._shared_q_learner:
            return

        try:
            if hasattr(self._shared_q_learner, '_promote_demote_memories'):
                self._shared_q_learner._promote_demote_memories(episode_reward=episode_reward)
        except Exception as e:
            logger.warning(f"Memory promotion/demotion failed: {e}")

    def prune_tier3(self, sample_rate: float = 0.1) -> None:
        """Prune Tier 3 memories by causal impact."""
        if not self._shared_q_learner:
            return

        try:
            if hasattr(self._shared_q_learner, 'prune_tier3_by_causal_impact'):
                self._shared_q_learner.prune_tier3_by_causal_impact(sample_rate=sample_rate)
        except Exception as e:
            logger.warning(f"Tier 3 pruning failed: {e}")

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_all(
        self,
        episode_count: int = 0,
        avg_reward: float = 0.0,
        domains: List[str] = None
    ):
        """
        Save all learning state.

        Args:
            episode_count: Number of episodes completed
            avg_reward: Average reward achieved
            domains: List of domains worked on
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Save shared Q-learner
        if self._shared_q_learner and hasattr(self._shared_q_learner, 'save_state'):
            shared_q_path = self.session_dir / "shared_q_learning.json"
            try:
                self._shared_q_learner.save_state(str(shared_q_path))
            except Exception as e:
                logger.warning(f"Could not save shared Q-learner: {e}")

        # Save per-agent learners
        agents_dir = self.session_dir / "agents"
        agents_dir.mkdir(exist_ok=True)

        for agent_name, learner in self._agent_q_learners.items():
            agent_dir = agents_dir / agent_name
            agent_dir.mkdir(exist_ok=True)
            if hasattr(learner, 'save_state'):
                try:
                    learner.save_state(str(agent_dir / "q_learning.json"))
                except Exception as e:
                    logger.debug(f"Could not save Q-learner for {agent_name}: {e}")

        for agent_name, memory in self._agent_memories.items():
            agent_dir = agents_dir / agent_name
            agent_dir.mkdir(exist_ok=True)
            if hasattr(memory, 'save'):
                try:
                    memory.save(str(agent_dir / "memory.json"))
                except Exception as e:
                    logger.debug(f"Could not save memory for {agent_name}: {e}")

        # Calculate total experiences
        total_exp = 0
        if self._shared_q_learner and hasattr(self._shared_q_learner, 'experience_buffer'):
            total_exp += len(self._shared_q_learner.experience_buffer)
        for learner in self._agent_q_learners.values():
            if hasattr(learner, 'experience_buffer'):
                total_exp += len(learner.experience_buffer)

        # Merge current domains
        all_domains = list(set((domains or []) + self._current_domains))

        # Update registry
        self.registry[self.session_id] = LearningSession(
            session_id=self.session_id,
            created_at=self.registry[self.session_id].created_at if self.session_id in self.registry else time.time(),
            updated_at=time.time(),
            episode_count=episode_count,
            total_experiences=total_exp,
            domains=all_domains,
            agents=list(self._agent_q_learners.keys()),
            avg_reward=avg_reward,
            path=str(self.session_dir)
        )

        # Update domain index
        for domain in all_domains:
            if domain not in self._domain_index:
                self._domain_index[domain] = []
            if self.session_id not in self._domain_index[domain]:
                self._domain_index[domain].append(self.session_id)

        self._save_registry()

        logger.info(f"Saved all learning to: {self.session_dir}")


# =============================================================================
# NO-OP FALLBACKS
# =============================================================================

class _NoOpLearner:
    """No-op learner for when Q-learning is unavailable."""

    def add_experience(self, *args, **kwargs):
        pass

    def record_outcome(self, *args, **kwargs):
        pass

    def predict_q_value(self, *args, **kwargs):
        return 0.5, 0.1, None

    def get_learned_context(self, *args, **kwargs):
        return ""

    def get_q_table_stats(self):
        return {'size': 0, 'avg_q_value': 0}

    def save_state(self, path):
        pass

    def load_state(self, path):
        pass


class _NoOpMemory:
    """No-op memory for when memory system is unavailable."""

    def store(self, *args, **kwargs):
        pass

    def retrieve(self, *args, **kwargs):
        return []

    def get_statistics(self):
        return {'total_entries': 0}

    def save(self, path):
        pass

    def load(self, path):
        pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_coordinator: Optional[LearningCoordinator] = None


def get_learning_coordinator(config=None, base_dir: str = None) -> LearningCoordinator:
    """Get or create global learning coordinator."""
    global _global_coordinator

    if _global_coordinator is None:
        if config is None:
            @dataclass
            class MinimalConfig:
                output_base_dir: str = "./outputs"
                alpha: float = 0.3
                gamma: float = 0.9
                epsilon: float = 0.1
                enable_rl: bool = False
                max_q_table_size: int = 10000
                tier1_max_size: int = 50
                tier2_max_clusters: int = 10
                tier3_max_size: int = 500
                max_experience_buffer: int = 200

            config = MinimalConfig()

        _global_coordinator = LearningCoordinator(config, base_dir)

    return _global_coordinator


def reset_learning_coordinator():
    """Reset global learning coordinator."""
    global _global_coordinator
    _global_coordinator = None


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for backward compatibility with old LearningManager
LearningManager = LearningCoordinator
get_learning_manager = get_learning_coordinator
reset_learning_manager = reset_learning_coordinator
