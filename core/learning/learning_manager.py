"""
Learning Manager - Centralized Learning State Management for MAS

Handles:
- Automatic loading of previous learning on startup
- Per-agent learning state (Q-tables, memories)
- Domain-based learning retrieval
- Learning registry and discovery
- Version management and rollback

Usage:
    from core.learning.learning_manager import LearningManager

    # Initialize with config
    manager = LearningManager(config)

    # Auto-loads latest learning if exists
    manager.initialize()

    # Get agent-specific learner
    planner_learner = manager.get_agent_learner("Planner")
    executor_learner = manager.get_agent_learner("Executor")

    # Save all learning
    manager.save_all()

    # Load specific domain learning
    manager.load_domain_learning("microservices")
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

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


class LearningManager:
    """
    Centralized manager for all learning state in MAS.

    Features:
    - Auto-discovery of previous learning sessions
    - Per-agent Q-tables and memories
    - Domain-indexed learning for transfer
    - Automatic loading on initialization
    - Session versioning and registry
    """

    def __init__(self, config, base_dir: str = None):
        """
        Initialize Learning Manager.

        Args:
            config: SwarmConfig or similar with learning settings
            base_dir: Base directory for learning storage (default: from config)
        """
        self.config = config
        self.base_dir = Path(base_dir or getattr(config, 'output_base_dir', './outputs'))
        self.learning_dir = self.base_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Registry of learning sessions
        self.registry_path = self.learning_dir / "registry.json"
        self.registry: Dict[str, LearningSession] = {}

        # Current session
        self.session_id = f"session_{int(time.time())}"
        self.session_dir = self.learning_dir / self.session_id

        # Per-agent learners
        self._agent_q_learners: Dict[str, 'LLMQPredictor'] = {}
        self._agent_memories: Dict[str, 'SimpleFallbackMemory'] = {}

        # Shared learning (cross-agent)
        self._shared_q_learner = None
        self._shared_memory = None

        # Domain index for transfer learning
        self._domain_index: Dict[str, List[str]] = {}  # domain -> [session_ids]

        # Current session domains
        self._current_domains: List[str] = []

        # Load registry
        self._load_registry()

        logger.info(f"📚 LearningManager initialized: {self.learning_dir}")

    def _load_registry(self):
        """Load learning session registry."""
        if self.registry_path.exists():
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
                logger.info(f"📂 Loaded registry: {len(self.registry)} sessions")
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")

    def _save_registry(self):
        """Save learning session registry."""
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

    def initialize(self, auto_load: bool = True) -> bool:
        """
        Initialize learning manager, optionally loading previous learning.

        Args:
            auto_load: If True, automatically load latest learning session

        Returns:
            True if previous learning was loaded
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create 'latest' symlink
        latest_link = self.learning_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(self.session_id)

        if auto_load and self.registry:
            return self.load_latest()

        return False

    def load_latest(self) -> bool:
        """Load the most recent learning session."""
        if not self.registry:
            logger.info("No previous learning sessions found")
            return False

        # Find latest by updated_at
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

        # Load shared Q-learner (create if needed)
        shared_q_path = session_path / "shared_q_learning.json"
        if shared_q_path.exists():
            shared_learner = self.get_shared_learner()  # Creates if needed
            shared_learner.load_state(str(shared_q_path))
            loaded = True

        # Load shared memory (create if needed)
        shared_mem_path = session_path / "shared_memory.json"
        if shared_mem_path.exists():
            shared_memory = self.get_shared_memory()  # Creates if needed
            shared_memory.load(str(shared_mem_path))
            loaded = True

        # Load per-agent learners (create agents if they don't exist)
        agents_dir = session_path / "agents"
        if agents_dir.exists():
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    agent_name = agent_dir.name

                    # Load agent Q-learner (create if needed)
                    q_path = agent_dir / "q_learning.json"
                    if q_path.exists():
                        learner = self.get_agent_learner(agent_name)  # Creates if needed
                        learner.load_state(str(q_path))
                        loaded = True
                        logger.debug(f"Loaded Q-learner for agent: {agent_name}")

                    # Load agent memory (create if needed)
                    mem_path = agent_dir / "memory.json"
                    if mem_path.exists():
                        memory = self.get_agent_memory(agent_name)  # Creates if needed
                        memory.load(str(mem_path))
                        loaded = True
                        logger.debug(f"Loaded memory for agent: {agent_name}")

        if loaded:
            logger.info(f"✅ Loaded learning from session: {session_id}")

        return loaded

    def load_domain_learning(self, domain: str, top_k: int = 3) -> bool:
        """
        Load learning from sessions that worked on similar domain.

        Args:
            domain: Domain to search for (e.g., "microservices", "ml", "database")
            top_k: Number of best sessions to load from

        Returns:
            True if any learning was loaded
        """
        # Find sessions with this domain
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

        # Load from top sessions
        loaded = False
        for session in matching_sessions[:top_k]:
            if self.load_session(session.session_id):
                loaded = True
                logger.info(f"Loaded domain learning from: {session.session_id} (reward: {session.avg_reward:.3f})")

        return loaded

    def get_agent_learner(self, agent_name: str) -> 'LLMQPredictor':
        """
        Get or create Q-learner for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            LLMQPredictor instance for this agent
        """
        if agent_name not in self._agent_q_learners:
            from core.learning.q_learning import LLMQPredictor
            self._agent_q_learners[agent_name] = LLMQPredictor(self.config)
            logger.debug(f"Created Q-learner for agent: {agent_name}")

        return self._agent_q_learners[agent_name]

    def get_agent_memory(self, agent_name: str) -> 'SimpleFallbackMemory':
        """
        Get or create memory for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            SimpleFallbackMemory instance for this agent
        """
        if agent_name not in self._agent_memories:
            from core.memory.fallback_memory import SimpleFallbackMemory
            self._agent_memories[agent_name] = SimpleFallbackMemory()
            logger.debug(f"Created memory for agent: {agent_name}")

        return self._agent_memories[agent_name]

    def get_shared_learner(self) -> 'LLMQPredictor':
        """Get shared Q-learner (cross-agent)."""
        if self._shared_q_learner is None:
            from core.learning.q_learning import LLMQPredictor
            self._shared_q_learner = LLMQPredictor(self.config)

        return self._shared_q_learner

    def get_shared_memory(self) -> 'SimpleFallbackMemory':
        """Get shared memory (cross-agent)."""
        if self._shared_memory is None:
            from core.memory.fallback_memory import SimpleFallbackMemory
            self._shared_memory = SimpleFallbackMemory()

        return self._shared_memory

    def record_experience(
        self,
        agent_name: str,
        state: Dict,
        action: Dict,
        reward: float,
        domain: str = None
    ):
        """
        Record an experience for an agent.

        Args:
            agent_name: Name of the agent
            state: Current state
            action: Action taken
            reward: Reward received
            domain: Optional domain tag for indexing
        """
        # Record in agent's Q-learner
        learner = self.get_agent_learner(agent_name)
        learner.add_experience(state, action, reward)

        # Also record in shared learner with agent context
        shared_state = {**state, 'agent': agent_name}
        self.get_shared_learner().add_experience(shared_state, action, reward)

        # Track domain
        if domain and domain not in self._current_domains:
            self._current_domains.append(domain)

    def save_all(self, episode_count: int = 0, avg_reward: float = 0.0, domains: List[str] = None):
        """
        Save all learning state.

        Args:
            episode_count: Number of episodes completed
            avg_reward: Average reward achieved
            domains: List of domains worked on
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Save shared Q-learner
        if self._shared_q_learner:
            shared_q_path = self.session_dir / "shared_q_learning.json"
            self._shared_q_learner.save_state(str(shared_q_path))

        # Save shared memory
        if self._shared_memory:
            shared_mem_path = self.session_dir / "shared_memory.json"
            self._shared_memory.save(str(shared_mem_path))

        # Save per-agent learners
        agents_dir = self.session_dir / "agents"
        agents_dir.mkdir(exist_ok=True)

        for agent_name, learner in self._agent_q_learners.items():
            agent_dir = agents_dir / agent_name
            agent_dir.mkdir(exist_ok=True)
            learner.save_state(str(agent_dir / "q_learning.json"))

        for agent_name, memory in self._agent_memories.items():
            agent_dir = agents_dir / agent_name
            agent_dir.mkdir(exist_ok=True)
            memory.save(str(agent_dir / "memory.json"))

        # Calculate total experiences
        total_exp = 0
        if self._shared_q_learner:
            total_exp += len(self._shared_q_learner.experience_buffer)
        for learner in self._agent_q_learners.values():
            total_exp += len(learner.experience_buffer)

        # Update registry
        self.registry[self.session_id] = LearningSession(
            session_id=self.session_id,
            created_at=time.time() if self.session_id not in self.registry else self.registry[self.session_id].created_at,
            updated_at=time.time(),
            episode_count=episode_count,
            total_experiences=total_exp,
            domains=domains or [],
            agents=list(self._agent_q_learners.keys()),
            avg_reward=avg_reward,
            path=str(self.session_dir)
        )

        # Update domain index
        for domain in (domains or []):
            if domain not in self._domain_index:
                self._domain_index[domain] = []
            if self.session_id not in self._domain_index[domain]:
                self._domain_index[domain].append(self.session_id)

        self._save_registry()

        logger.info(f"💾 Saved all learning to: {self.session_dir}")

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of current learning state."""
        summary = {
            'session_id': self.session_id,
            'session_dir': str(self.session_dir),
            'total_sessions': len(self.registry),
            'agents': list(self._agent_q_learners.keys()),
            'domains_indexed': list(self._domain_index.keys()),
        }

        # Add Q-table stats
        if self._shared_q_learner:
            summary['shared_q_stats'] = self._shared_q_learner.get_q_table_stats()

        summary['per_agent_stats'] = {}
        for agent_name, learner in self._agent_q_learners.items():
            summary['per_agent_stats'][agent_name] = learner.get_q_table_stats()

        return summary

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available learning sessions."""
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


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_learning_manager: Optional[LearningManager] = None


def get_learning_manager(config=None, base_dir: str = None) -> LearningManager:
    """Get or create global learning manager."""
    global _global_learning_manager

    if _global_learning_manager is None:
        if config is None:
            # Create minimal config
            from dataclasses import dataclass
            @dataclass
            class MinimalConfig:
                output_base_dir: str = "./outputs"
                alpha: float = 0.3
                gamma: float = 0.9
                epsilon: float = 0.1
                max_q_table_size: int = 10000

            config = MinimalConfig()

        _global_learning_manager = LearningManager(config, base_dir)

    return _global_learning_manager


def reset_learning_manager():
    """Reset global learning manager."""
    global _global_learning_manager
    _global_learning_manager = None
