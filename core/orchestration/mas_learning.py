"""
MAS Learning - Multi-Agent System Learning & Persistence (DRY Version)
======================================================================

Provides UNIQUE learning functionality not covered by other components:

1. Fix Database - Persistent error→solution mappings (UNIQUE)
2. Session History - Task history with topic-based relevance matching (UNIQUE)
3. Execution Strategy - Combines learnings for smart execution (UNIQUE)

DELEGATES TO (does not duplicate):
- SwarmIntelligence: Agent performance, specialization, best agent selection
- LearningManager: Q-learning, RL operations
- TransferableLearningStore: Cross-task pattern transfer

Usage:
    mas_learning = MASLearning(config, swarm_intelligence=si)
    mas_learning.load_relevant_learnings(task_description)
    # ... run MAS ...
    mas_learning.save_all()
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Fix Database (UNIQUE to MAS Learning)
# =============================================================================

@dataclass
class FixRecord:
    """Record of a successful fix."""
    error_pattern: str
    error_hash: str
    solution_commands: List[str]
    solution_description: str
    source: str  # 'pattern', 'web', 'llm', 'user'
    success_count: int = 1
    fail_count: int = 0
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0


# =============================================================================
# Session History (UNIQUE to MAS Learning)
# =============================================================================

@dataclass
class SessionLearning:
    """Learning from a single session."""
    session_id: str
    timestamp: str
    task_description: str
    task_topics: List[str]
    agents_used: List[str]
    stigmergy_signals: int
    total_time: float
    success: bool
    workspace: str
    # Lightweight metrics (no duplication with SwarmIntelligence)
    agent_count: int = 0
    output_quality: float = 0.0

    def get_relevance_score(self, query_topics: List[str], query_agents: List[str] = None) -> float:
        """Calculate relevance of this session to a query."""
        topic_overlap = len(set(self.task_topics) & set(query_topics))
        topic_score = topic_overlap / max(len(query_topics), 1)

        agent_score = 0.0
        if query_agents:
            agent_overlap = len(set(self.agents_used) & set(query_agents))
            agent_score = agent_overlap / len(query_agents)

        recency_days = (datetime.now() - datetime.fromisoformat(self.timestamp)).days
        recency_score = max(0, 1 - recency_days / 30)  # Decay over 30 days

        success_bonus = 0.2 if self.success else 0

        return 0.4 * topic_score + 0.2 * agent_score + 0.2 * recency_score + 0.2 + success_bonus


# =============================================================================
# MAS Learning (Coordinator - DRY)
# =============================================================================

class MASLearning:
    """
    Multi-Agent System Learning Manager (DRY Version).

    Provides UNIQUE functionality:
    - Fix database (error→solution mappings)
    - Session history with task relevance matching
    - Execution strategy recommendations

    DELEGATES to existing components (no duplication):
    - SwarmIntelligence for agent performance/specialization
    - LearningManager for Q-learning
    - TransferableLearningStore for pattern transfer
    """

    def __init__(
        self,
        config: Any = None,
        workspace_path: Optional[Path] = None,
        learning_dir: Optional[Path] = None,
        swarm_intelligence: Any = None,  # SwarmIntelligence instance
        learning_manager: Any = None,    # LearningManager instance
        transfer_learning: Any = None    # TransferableLearningStore instance
    ):
        """
        Initialize MAS Learning.

        Args:
            config: SwarmConfig (optional)
            workspace_path: Current workspace/project path
            learning_dir: Directory for learning files
            swarm_intelligence: SwarmIntelligence for agent tracking (DELEGATE)
            learning_manager: LearningManager for Q-learning (DELEGATE)
            transfer_learning: TransferableLearningStore for patterns (DELEGATE)
        """
        self.config = config
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()

        # Delegate to existing components (DRY)
        self.swarm_intelligence = swarm_intelligence
        self.learning_manager = learning_manager
        self.transfer_learning = transfer_learning

        # Learning directory
        if learning_dir:
            self.learning_dir = Path(learning_dir)
        else:
            base = getattr(config, 'base_path', None) if config else None
            if base:
                self.learning_dir = Path(base) / 'mas_learning'
            else:
                self.learning_dir = Path.home() / '.jotty' / 'mas_learning'

        self.learning_dir.mkdir(parents=True, exist_ok=True)

        # Project-specific learning directory
        workspace_hash = hashlib.md5(str(self.workspace_path).encode()).hexdigest()[:8]
        self.project_learning_dir = self.learning_dir / 'projects' / workspace_hash
        self.project_learning_dir.mkdir(parents=True, exist_ok=True)

        # UNIQUE data stores (not in other components)
        self.fix_database: Dict[str, FixRecord] = {}
        self.session_learnings: List[SessionLearning] = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Topic extraction cache
        self._topic_cache: Dict[str, List[str]] = {}

        # Load existing learnings
        self._load_all()

        logger.info(f" MASLearning initialized (DRY): {len(self.fix_database)} fixes, "
                   f"{len(self.session_learnings)} sessions")

    # =========================================================================
    # Fix Database (UNIQUE - error→solution mappings)
    # =========================================================================

    def _error_hash(self, error: str) -> str:
        """Create hash of error for lookup."""
        import re
        normalized = error.lower()
        normalized = re.sub(r'\d+', 'N', normalized)
        normalized = re.sub(r'/[\w/.-]+', '/PATH', normalized)
        normalized = re.sub(r'0x[0-9a-f]+', 'ADDR', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()

    def find_fix(self, error: str) -> Optional[FixRecord]:
        """Find a fix for an error from the database."""
        error_hash = self._error_hash(error)

        if error_hash in self.fix_database:
            fix = self.fix_database[error_hash]
            if fix.success_rate >= 0.5:
                return fix

        for fix_hash, fix in self.fix_database.items():
            if fix.error_pattern in error or error in fix.error_pattern:
                if fix.success_rate >= 0.6:
                    return fix

        return None

    def record_fix(
        self,
        error: str,
        solution_commands: List[str],
        solution_description: str,
        source: str,
        success: bool,
        context: Dict[str, Any] = None
    ):
        """Record a fix attempt (success or failure)."""
        error_hash = self._error_hash(error)

        if error_hash in self.fix_database:
            fix = self.fix_database[error_hash]
            if success:
                fix.success_count += 1
            else:
                fix.fail_count += 1
            fix.last_used = datetime.now().isoformat()
        elif success:
            self.fix_database[error_hash] = FixRecord(
                error_pattern=error[:500],
                error_hash=error_hash,
                solution_commands=solution_commands,
                solution_description=solution_description,
                source=source,
                context=context or {}
            )

        if len(self.fix_database) % 10 == 0:
            self._save_fix_database()

    # =========================================================================
    # Session History (UNIQUE - task relevance matching)
    # =========================================================================

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text for similarity matching."""
        if text in self._topic_cache:
            return self._topic_cache[text]

        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        stopwords = {'this', 'that', 'with', 'from', 'have', 'will', 'would', 'could',
                    'should', 'being', 'been', 'were', 'what', 'when', 'where', 'which',
                    'their', 'them', 'then', 'than', 'these', 'those', 'some', 'such',
                    'only', 'other', 'into', 'over', 'also', 'more', 'most', 'very'}

        topics = [w for w in words if w not in stopwords]

        from collections import Counter
        topic_counts = Counter(topics)
        top_topics = [t for t, _ in topic_counts.most_common(10)]

        self._topic_cache[text] = top_topics
        return top_topics

    def record_session(
        self,
        task_description: str,
        agents_used: List[str],
        total_time: float,
        success: bool,
        stigmergy_signals: int = 0,
        output_quality: float = 0.0
    ):
        """Record learning from a completed session."""
        session = SessionLearning(
            session_id=self.current_session_id,
            timestamp=datetime.now().isoformat(),
            task_description=task_description,
            task_topics=self._extract_topics(task_description),
            agents_used=agents_used,
            stigmergy_signals=stigmergy_signals,
            total_time=total_time,
            success=success,
            workspace=str(self.workspace_path),
            agent_count=len(agents_used),
            output_quality=output_quality
        )

        self.session_learnings.append(session)

        if len(self.session_learnings) > 100:
            self.session_learnings = self.session_learnings[-100:]

        self._save_sessions()

    def load_relevant_learnings(
        self,
        task_description: str,
        agent_types: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Load learnings relevant to the current task.

        Combines:
        - Session history (unique to MAS Learning)
        - SwarmIntelligence agent recommendations (delegated)
        - Fix database (unique to MAS Learning)
        """
        query_topics = self._extract_topics(task_description)

        # Score and rank past sessions
        scored_sessions = [
            (session, session.get_relevance_score(query_topics, agent_types))
            for session in self.session_learnings
        ]
        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        relevant_sessions = [s for s, score in scored_sessions[:top_k] if score > 0.3]

        # Extract insights from sessions
        past_strategies = []
        for session in relevant_sessions:
            if session.success:
                past_strategies.append({
                    'task': session.task_description[:100],
                    'agents_used': session.agents_used,
                    'time': session.total_time,
                    'stigmergy_signals': session.stigmergy_signals
                })

        # Get agent recommendations from SwarmIntelligence (DELEGATE - DRY)
        suggested_agents = {}
        underperformers = {}
        if self.swarm_intelligence:
            try:
                # Use SwarmIntelligence's existing agent tracking
                for agent_name in (agent_types or []):
                    profile = self.swarm_intelligence.get_agent_profile(agent_name)
                    if profile:
                        suggested_agents[agent_name] = {
                            'success_rate': profile.success_rate,
                            'specialization': profile.specialization.value if hasattr(profile, 'specialization') else 'unknown',
                            'total_tasks': profile.total_tasks
                        }
                        if profile.success_rate < 0.6 and profile.total_tasks >= 2:
                            underperformers[agent_name] = profile.success_rate
            except Exception as e:
                logger.debug(f"Could not get SwarmIntelligence data: {e}")

        # Performance hints from session history
        avg_time = sum(s.total_time for s in relevant_sessions) / len(relevant_sessions) if relevant_sessions else 60
        success_rate = sum(1 for s in relevant_sessions if s.success) / len(relevant_sessions) if relevant_sessions else 0.5

        performance_hints = {
            'expected_time': avg_time,
            'expected_success_rate': success_rate,
            'similar_task_count': len(relevant_sessions),
        }

        # Get relevant fixes
        relevant_fixes = list(self.fix_database.values())[:10]

        logger.info(f" Loaded learnings: {len(relevant_sessions)} sessions, "
                   f"{len(suggested_agents)} agents, {len(relevant_fixes)} fixes")

        return {
            'suggested_agents': suggested_agents,
            'underperformers': underperformers,
            'relevant_fixes': [{'pattern': f.error_pattern[:50], 'solution': f.solution_description} for f in relevant_fixes],
            'performance_hints': performance_hints,
            'past_strategies': past_strategies[:5],
            'query_topics': query_topics
        }

    # =========================================================================
    # Execution Strategy (Combines learnings - UNIQUE)
    # =========================================================================

    def get_execution_strategy(self, task_description: str, available_agents: List[str]) -> Dict[str, Any]:
        """
        Get recommended execution strategy based on all learnings.

        Combines:
        - Session history for expected performance
        - SwarmIntelligence for agent selection (delegated)
        - Fix database for potential issues
        """
        learnings = self.load_relevant_learnings(task_description, available_agents)

        # Get underperformers from SwarmIntelligence (DELEGATE)
        skip_agents = []
        retry_agents = []

        for agent in available_agents:
            if agent in learnings.get('underperformers', {}):
                success_rate = learnings['underperformers'][agent]
                if success_rate < 0.4:
                    skip_agents.append({'agent': agent, 'reason': f'{success_rate*100:.0f}% success rate'})
                else:
                    retry_agents.append({'agent': agent, 'success_rate': success_rate})

        # Order by success rate (use SwarmIntelligence data)
        agent_scores = {}
        for agent in available_agents:
            if agent in learnings.get('suggested_agents', {}):
                agent_scores[agent] = learnings['suggested_agents'][agent].get('success_rate', 0.5)
            else:
                agent_scores[agent] = 0.5

        recommended_order = sorted(
            [a for a in available_agents if a not in [s['agent'] for s in skip_agents]],
            key=lambda a: agent_scores.get(a, 0.5),
            reverse=True
        )

        hints = learnings.get('performance_hints', {})

        return {
            'recommended_order': recommended_order,
            'skip_agents': skip_agents,
            'retry_agents': retry_agents,
            'expected_time': hints.get('expected_time', 60.0),
            'confidence': min(1.0, hints.get('similar_task_count', 0) / 5),
            'relevant_fixes': learnings.get('relevant_fixes', [])
        }

    # =========================================================================
    # Persistence (Fix Database + Sessions only - others delegated)
    # =========================================================================

    def _get_fix_db_path(self) -> Path:
        return self.learning_dir / 'fix_database.json'

    def _get_sessions_path(self) -> Path:
        return self.learning_dir / 'session_learnings.json'

    def _save_fix_database(self):
        """Save fix database to disk."""
        try:
            data = {
                hash_: {
                    'error_pattern': fix.error_pattern,
                    'error_hash': fix.error_hash,
                    'solution_commands': fix.solution_commands,
                    'solution_description': fix.solution_description,
                    'source': fix.source,
                    'success_count': fix.success_count,
                    'fail_count': fix.fail_count,
                    'last_used': fix.last_used,
                    'context': fix.context
                }
                for hash_, fix in self.fix_database.items()
            }

            with open(self._get_fix_db_path(), 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save fix database: {e}")

    def _load_fix_database(self) -> None:
        """Load fix database from disk."""
        path = self._get_fix_db_path()
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for hash_, fix_data in data.items():
                self.fix_database[hash_] = FixRecord(
                    error_pattern=fix_data['error_pattern'],
                    error_hash=fix_data['error_hash'],
                    solution_commands=fix_data['solution_commands'],
                    solution_description=fix_data['solution_description'],
                    source=fix_data['source'],
                    success_count=fix_data.get('success_count', 1),
                    fail_count=fix_data.get('fail_count', 0),
                    last_used=fix_data.get('last_used', datetime.now().isoformat()),
                    context=fix_data.get('context', {})
                )

            logger.info(f"Loaded {len(self.fix_database)} fixes from database")

        except Exception as e:
            logger.warning(f"Could not load fix database: {e}")

    def _save_sessions(self):
        """Save session learnings to disk."""
        try:
            data = [asdict(s) for s in self.session_learnings]
            with open(self._get_sessions_path(), 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Could not save sessions: {e}")

    def _load_sessions(self) -> None:
        """Load session learnings from disk."""
        path = self._get_sessions_path()
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for session_data in data:
                self.session_learnings.append(SessionLearning(
                    session_id=session_data['session_id'],
                    timestamp=session_data['timestamp'],
                    task_description=session_data['task_description'],
                    task_topics=session_data.get('task_topics', []),
                    agents_used=session_data.get('agents_used', []),
                    stigmergy_signals=session_data.get('stigmergy_signals', 0),
                    total_time=session_data.get('total_time', 0),
                    success=session_data.get('success', False),
                    workspace=session_data.get('workspace', ''),
                    agent_count=session_data.get('agent_count', 0),
                    output_quality=session_data.get('output_quality', 0.0)
                ))

            logger.info(f"Loaded {len(self.session_learnings)} sessions")

        except Exception as e:
            logger.warning(f"Could not load sessions: {e}")

    def _load_all(self):
        """Load all learning data from disk."""
        self._load_fix_database()
        self._load_sessions()

    def save_all(self) -> None:
        """Save all learning data to disk."""
        self._save_fix_database()
        self._save_sessions()
        logger.info(" MAS learnings saved (fix database + sessions)")

    # =========================================================================
    # Integration with SwarmTerminal (UNIQUE - fix persistence)
    # =========================================================================

    def integrate_with_terminal(self, swarm_terminal) -> None:
        """Integrate with SwarmTerminal to persist fix learnings."""
        if not swarm_terminal:
            return

        # Load existing fixes into terminal's cache
        for fix in self.fix_database.values():
            if fix.success_rate >= 0.5:
                swarm_terminal._fix_cache[fix.error_hash] = type(
                    'ErrorSolution', (), {
                        'error_pattern': fix.error_pattern,
                        'solution': fix.solution_description,
                        'source': 'database',
                        'confidence': fix.success_rate,
                        'commands': fix.solution_commands
                    }
                )()

        logger.info(f"Integrated {len(self.fix_database)} fixes with SwarmTerminal")

    def sync_from_terminal(self, swarm_terminal) -> int:
        """Sync fix learnings from SwarmTerminal to database."""
        if not swarm_terminal:
            return 0

        new_fixes = 0

        for fix_entry in getattr(swarm_terminal, '_fix_history', []):
            error = fix_entry.get('error', '')
            if not error:
                continue

            self.record_fix(
                error=error,
                solution_commands=fix_entry.get('commands', []),
                solution_description=fix_entry.get('description', ''),
                source=fix_entry.get('source', 'terminal'),
                success=fix_entry.get('success', True),
                context=fix_entry.get('context', {})
            )
            new_fixes += 1

        if new_fixes:
            self._save_fix_database()
            logger.info(f"Synced {new_fixes} fixes from SwarmTerminal")

        return new_fixes

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'fix_database': {
                'total_fixes': len(self.fix_database),
                'avg_success_rate': sum(f.success_rate for f in self.fix_database.values()) / max(len(self.fix_database), 1),
            },
            'sessions': {
                'total_sessions': len(self.session_learnings),
                'successful_sessions': sum(1 for s in self.session_learnings if s.success),
                'avg_time': sum(s.total_time for s in self.session_learnings) / max(len(self.session_learnings), 1)
            },
            'delegates_to': {
                'swarm_intelligence': self.swarm_intelligence is not None,
                'learning_manager': self.learning_manager is not None,
                'transfer_learning': self.transfer_learning is not None
            }
        }


# Convenience function
def get_mas_learning(config=None, workspace_path=None, **delegates) -> MASLearning:
    """Get or create MAS Learning instance."""
    return MASLearning(config=config, workspace_path=workspace_path, **delegates)
