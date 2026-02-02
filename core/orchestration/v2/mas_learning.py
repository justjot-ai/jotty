"""
MAS Learning - Multi-Agent System Learning & Persistence
=========================================================

Provides persistent learning across sessions for the multi-agent swarm:

1. Memory Persistence - HierarchicalMemory saved/loaded from disk
2. Fix Database - Successful error→solution mappings persisted
3. Performance Metrics - Track agent/strategy effectiveness over time
4. Agent Performance - Which agents excel at which task types
5. Smart Loading - Match current task to relevant past learnings

Usage:
    mas_learning = MASLearning(config, workspace_path)
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
from collections import defaultdict

logger = logging.getLogger(__name__)


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


@dataclass
class AgentPerformance:
    """Track performance metrics for an agent type."""
    agent_type: str
    task_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_tasks: int = 0
    total_success: int = 0
    total_time: float = 0.0
    specializations: List[str] = field(default_factory=list)

    def record_task(self, task_type: str, success: bool, time_taken: float, output_quality: float = 0.0):
        """Record a task execution."""
        if task_type not in self.task_types:
            self.task_types[task_type] = {
                'count': 0, 'success': 0, 'total_time': 0.0,
                'avg_quality': 0.0, 'quality_samples': 0
            }

        stats = self.task_types[task_type]
        stats['count'] += 1
        stats['total_time'] += time_taken
        if success:
            stats['success'] += 1
        if output_quality > 0:
            n = stats['quality_samples']
            stats['avg_quality'] = (stats['avg_quality'] * n + output_quality) / (n + 1)
            stats['quality_samples'] += 1

        self.total_tasks += 1
        if success:
            self.total_success += 1
        self.total_time += time_taken

        # Update specializations (top 3 task types by success rate)
        self._update_specializations()

    def _update_specializations(self):
        """Update agent's specializations based on performance."""
        task_scores = []
        for task_type, stats in self.task_types.items():
            if stats['count'] >= 2:  # Need at least 2 samples
                success_rate = stats['success'] / stats['count']
                quality = stats.get('avg_quality', 0.5)
                score = 0.7 * success_rate + 0.3 * quality
                task_scores.append((task_type, score))

        task_scores.sort(key=lambda x: x[1], reverse=True)
        self.specializations = [t[0] for t in task_scores[:3]]

    @property
    def success_rate(self) -> float:
        return self.total_success / self.total_tasks if self.total_tasks > 0 else 0.0


@dataclass
class SessionLearning:
    """Learning from a single session."""
    session_id: str
    timestamp: str
    task_description: str
    task_topics: List[str]
    agent_performances: Dict[str, Dict[str, Any]]
    fixes_applied: List[Dict[str, Any]]
    stigmergy_signals: int
    total_time: float
    success: bool
    workspace: str

    def get_relevance_score(self, query_topics: List[str], query_agents: List[str] = None) -> float:
        """Calculate relevance of this session to a query."""
        topic_overlap = len(set(self.task_topics) & set(query_topics))
        topic_score = topic_overlap / max(len(query_topics), 1)

        agent_score = 0.0
        if query_agents:
            agent_overlap = len(set(self.agent_performances.keys()) & set(query_agents))
            agent_score = agent_overlap / len(query_agents)

        recency_days = (datetime.now() - datetime.fromisoformat(self.timestamp)).days
        recency_score = max(0, 1 - recency_days / 30)  # Decay over 30 days

        success_bonus = 0.2 if self.success else 0

        return 0.4 * topic_score + 0.2 * agent_score + 0.2 * recency_score + 0.2 + success_bonus


class MASLearning:
    """
    Multi-Agent System Learning Manager.

    Handles all persistent learning across sessions:
    - Fix database (error→solution mappings)
    - Agent performance metrics
    - Session learnings with smart retrieval
    - Memory persistence integration
    """

    def __init__(
        self,
        config: Any = None,
        workspace_path: Optional[Path] = None,
        learning_dir: Optional[Path] = None
    ):
        """
        Initialize MAS Learning.

        Args:
            config: JottyConfig (optional)
            workspace_path: Current workspace/project path (for project-specific learning)
            learning_dir: Directory for learning files (default: ~/.jotty/mas_learning/)
        """
        self.config = config
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()

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

        # Core data stores
        self.fix_database: Dict[str, FixRecord] = {}
        self.agent_performances: Dict[str, AgentPerformance] = {}
        self.session_learnings: List[SessionLearning] = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Topic extraction cache
        self._topic_cache: Dict[str, List[str]] = {}

        # Load existing learnings
        self._load_all()

        logger.info(f"🧠 MASLearning initialized: {len(self.fix_database)} fixes, "
                   f"{len(self.agent_performances)} agents, {len(self.session_learnings)} sessions")

    # =========================================================================
    # Fix Database (Error → Solution Mappings)
    # =========================================================================

    def _error_hash(self, error: str) -> str:
        """Create hash of error for lookup."""
        # Normalize: remove timestamps, paths, specific values
        import re
        normalized = error.lower()
        normalized = re.sub(r'\d+', 'N', normalized)  # Replace numbers
        normalized = re.sub(r'/[\w/.-]+', '/PATH', normalized)  # Replace paths
        normalized = re.sub(r'0x[0-9a-f]+', 'ADDR', normalized)  # Replace addresses
        return hashlib.md5(normalized.encode()).hexdigest()

    def find_fix(self, error: str) -> Optional[FixRecord]:
        """Find a fix for an error from the database."""
        error_hash = self._error_hash(error)

        if error_hash in self.fix_database:
            fix = self.fix_database[error_hash]
            if fix.success_rate >= 0.5:  # Only return if >50% success rate
                return fix

        # Try partial matching for similar errors
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
            # Only create new record if successful
            self.fix_database[error_hash] = FixRecord(
                error_pattern=error[:500],  # Truncate long errors
                error_hash=error_hash,
                solution_commands=solution_commands,
                solution_description=solution_description,
                source=source,
                context=context or {}
            )

        # Auto-save periodically
        if len(self.fix_database) % 10 == 0:
            self._save_fix_database()

    # =========================================================================
    # Agent Performance Tracking
    # =========================================================================

    def record_agent_task(
        self,
        agent_type: str,
        task_type: str,
        success: bool,
        time_taken: float,
        output_quality: float = 0.0
    ):
        """Record an agent's task execution for learning."""
        if agent_type not in self.agent_performances:
            self.agent_performances[agent_type] = AgentPerformance(agent_type=agent_type)

        self.agent_performances[agent_type].record_task(
            task_type, success, time_taken, output_quality
        )

    def get_best_agent_for_task(self, task_type: str) -> Optional[str]:
        """Get the best performing agent for a task type."""
        candidates = []

        for agent_type, perf in self.agent_performances.items():
            if task_type in perf.task_types:
                stats = perf.task_types[task_type]
                if stats['count'] >= 2:
                    success_rate = stats['success'] / stats['count']
                    quality = stats.get('avg_quality', 0.5)
                    score = 0.6 * success_rate + 0.4 * quality
                    candidates.append((agent_type, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def get_agent_specializations(self) -> Dict[str, List[str]]:
        """Get specializations for all agents."""
        return {
            agent_type: perf.specializations
            for agent_type, perf in self.agent_performances.items()
            if perf.specializations
        }

    def get_underperforming_agents(self, threshold: float = 0.6) -> Dict[str, float]:
        """
        Get agents with success rate below threshold.

        Args:
            threshold: Success rate threshold (default 60%)

        Returns:
            Dict of agent_name → success_rate for underperformers
        """
        underperformers = {}
        for agent_type, perf in self.agent_performances.items():
            if perf.total_tasks >= 2 and perf.success_rate < threshold:
                underperformers[agent_type] = perf.success_rate
        return underperformers

    def should_use_agent(self, agent_type: str, task_type: str = None, min_success_rate: float = 0.5) -> Tuple[bool, str]:
        """
        Determine if an agent should be used based on learning history.

        Args:
            agent_type: The agent to check
            task_type: Specific task type (optional)
            min_success_rate: Minimum acceptable success rate

        Returns:
            (should_use: bool, reason: str)
        """
        if agent_type not in self.agent_performances:
            return True, "No performance history - allow first use"

        perf = self.agent_performances[agent_type]

        if perf.total_tasks < 2:
            return True, "Insufficient data - allow more attempts"

        # Check overall performance
        if perf.success_rate < min_success_rate:
            return False, f"Low success rate: {perf.success_rate*100:.0f}% (threshold: {min_success_rate*100:.0f}%)"

        # Check task-specific performance if requested
        if task_type and task_type in perf.task_types:
            stats = perf.task_types[task_type]
            if stats['count'] >= 2:
                task_success_rate = stats['success'] / stats['count']
                if task_success_rate < min_success_rate:
                    return False, f"Low success for {task_type}: {task_success_rate*100:.0f}%"

        return True, f"Good performance: {perf.success_rate*100:.0f}% success"

    def get_execution_strategy(self, task_description: str, available_agents: List[str]) -> Dict[str, Any]:
        """
        Get recommended execution strategy based on learnings.

        Args:
            task_description: The task to execute
            available_agents: List of available agent names

        Returns:
            Strategy dict with:
            - recommended_order: Suggested agent execution order
            - skip_agents: Agents to skip (low performance)
            - retry_agents: Agents that might need retries
            - expected_time: Estimated total time
            - confidence: Strategy confidence (0-1)
        """
        learnings = self.load_relevant_learnings(task_description, available_agents)
        underperformers = self.get_underperforming_agents()

        # Determine which agents to skip or retry
        skip_agents = []
        retry_agents = []
        recommended_agents = []

        for agent in available_agents:
            should_use, reason = self.should_use_agent(agent)
            if not should_use:
                skip_agents.append({'agent': agent, 'reason': reason})
            elif agent in underperformers:
                retry_agents.append({'agent': agent, 'success_rate': underperformers[agent]})
                recommended_agents.append(agent)  # Still use, but mark for retry
            else:
                recommended_agents.append(agent)

        # Order agents by performance (best first for critical tasks)
        agent_scores = {}
        for agent in recommended_agents:
            if agent in self.agent_performances:
                agent_scores[agent] = self.agent_performances[agent].success_rate
            else:
                agent_scores[agent] = 0.5  # Unknown agents get neutral score

        recommended_order = sorted(recommended_agents, key=lambda a: agent_scores.get(a, 0.5), reverse=True)

        # Calculate expected time and confidence
        hints = learnings.get('performance_hints', {})
        expected_time = hints.get('expected_time', 60.0)
        similar_count = hints.get('similar_task_count', 0)
        confidence = min(1.0, similar_count / 5)  # More similar tasks = higher confidence

        return {
            'recommended_order': recommended_order,
            'skip_agents': skip_agents,
            'retry_agents': retry_agents,
            'expected_time': expected_time,
            'confidence': confidence,
            'underperformers': underperformers,
            'suggested_agents': learnings.get('suggested_agents', {})
        }

    # =========================================================================
    # Session Learning & Smart Loading
    # =========================================================================

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text for similarity matching."""
        if text in self._topic_cache:
            return self._topic_cache[text]

        # Simple keyword extraction (can be enhanced with NLP)
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Filter common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'will', 'would', 'could',
                    'should', 'being', 'been', 'were', 'what', 'when', 'where', 'which',
                    'their', 'them', 'then', 'than', 'these', 'those', 'some', 'such',
                    'only', 'other', 'into', 'over', 'also', 'more', 'most', 'very'}

        topics = [w for w in words if w not in stopwords]

        # Get unique topics by frequency
        from collections import Counter
        topic_counts = Counter(topics)
        top_topics = [t for t, _ in topic_counts.most_common(10)]

        self._topic_cache[text] = top_topics
        return top_topics

    def record_session(
        self,
        task_description: str,
        agent_performances: Dict[str, Dict[str, Any]],
        fixes_applied: List[Dict[str, Any]],
        stigmergy_signals: int,
        total_time: float,
        success: bool
    ):
        """Record learning from a completed session."""
        session = SessionLearning(
            session_id=self.current_session_id,
            timestamp=datetime.now().isoformat(),
            task_description=task_description,
            task_topics=self._extract_topics(task_description),
            agent_performances=agent_performances,
            fixes_applied=fixes_applied,
            stigmergy_signals=stigmergy_signals,
            total_time=total_time,
            success=success,
            workspace=str(self.workspace_path)
        )

        self.session_learnings.append(session)

        # Keep only last 100 sessions
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

        This is the key method for "which MAS learning should be loaded".

        Args:
            task_description: Description of the current task
            agent_types: Agent types that will be used (optional)
            top_k: Number of relevant sessions to consider

        Returns:
            Dict with relevant learnings:
            - suggested_agents: Best agents for this task type
            - relevant_fixes: Fixes likely to be needed
            - performance_hints: Expected performance based on history
            - past_strategies: Strategies that worked before
        """
        query_topics = self._extract_topics(task_description)

        # Score and rank past sessions by relevance
        scored_sessions = [
            (session, session.get_relevance_score(query_topics, agent_types))
            for session in self.session_learnings
        ]
        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        relevant_sessions = [s for s, score in scored_sessions[:top_k] if score > 0.3]

        # Extract insights from relevant sessions
        suggested_agents = {}
        relevant_fixes = []
        performance_hints = {}
        past_strategies = []

        for session in relevant_sessions:
            # Aggregate agent performance from similar past tasks
            for agent_type, perf in session.agent_performances.items():
                if agent_type not in suggested_agents:
                    suggested_agents[agent_type] = {'score': 0, 'count': 0, 'avg_time': 0}
                sa = suggested_agents[agent_type]
                sa['count'] += 1
                sa['score'] += perf.get('success_rate', 0.5)
                sa['avg_time'] += perf.get('avg_time', 0)

            # Collect relevant fixes
            for fix in session.fixes_applied:
                if fix not in relevant_fixes:
                    relevant_fixes.append(fix)

            # Extract strategies
            if session.success:
                past_strategies.append({
                    'task': session.task_description[:100],
                    'agents_used': list(session.agent_performances.keys()),
                    'time': session.total_time,
                    'stigmergy_signals': session.stigmergy_signals
                })

        # Finalize suggested agents
        for agent_type, stats in suggested_agents.items():
            if stats['count'] > 0:
                stats['score'] /= stats['count']
                stats['avg_time'] /= stats['count']

        # Add global agent specializations
        specializations = self.get_agent_specializations()
        for agent_type, specs in specializations.items():
            for topic in query_topics:
                if any(topic in spec.lower() for spec in specs):
                    if agent_type not in suggested_agents:
                        suggested_agents[agent_type] = {'score': 0.7, 'count': 0, 'specialized': True}
                    else:
                        suggested_agents[agent_type]['specialized'] = True

        # Performance hints based on task complexity
        avg_time = sum(s.total_time for s in relevant_sessions) / len(relevant_sessions) if relevant_sessions else 60
        success_rate = sum(1 for s in relevant_sessions if s.success) / len(relevant_sessions) if relevant_sessions else 0.5

        performance_hints = {
            'expected_time': avg_time,
            'expected_success_rate': success_rate,
            'similar_task_count': len(relevant_sessions),
            'recommended_agents': sorted(
                suggested_agents.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )[:3]
        }

        logger.info(f"📚 Loaded learnings: {len(relevant_sessions)} relevant sessions, "
                   f"{len(suggested_agents)} suggested agents, {len(relevant_fixes)} fixes")

        return {
            'suggested_agents': suggested_agents,
            'relevant_fixes': relevant_fixes[:10],  # Top 10 fixes
            'performance_hints': performance_hints,
            'past_strategies': past_strategies[:5],  # Top 5 strategies
            'query_topics': query_topics
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _get_fix_db_path(self) -> Path:
        return self.learning_dir / 'fix_database.json'

    def _get_agent_perf_path(self) -> Path:
        return self.learning_dir / 'agent_performance.json'

    def _get_sessions_path(self) -> Path:
        return self.learning_dir / 'session_learnings.json'

    def _get_project_sessions_path(self) -> Path:
        return self.project_learning_dir / 'sessions.json'

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

    def _load_fix_database(self):
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

    def _save_agent_performance(self):
        """Save agent performance to disk."""
        try:
            data = {}
            for agent_type, perf in self.agent_performances.items():
                data[agent_type] = {
                    'agent_type': perf.agent_type,
                    'task_types': perf.task_types,
                    'total_tasks': perf.total_tasks,
                    'total_success': perf.total_success,
                    'total_time': perf.total_time,
                    'specializations': perf.specializations
                }

            with open(self._get_agent_perf_path(), 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save agent performance: {e}")

    def _load_agent_performance(self):
        """Load agent performance from disk."""
        path = self._get_agent_perf_path()
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for agent_type, perf_data in data.items():
                self.agent_performances[agent_type] = AgentPerformance(
                    agent_type=perf_data['agent_type'],
                    task_types=perf_data.get('task_types', {}),
                    total_tasks=perf_data.get('total_tasks', 0),
                    total_success=perf_data.get('total_success', 0),
                    total_time=perf_data.get('total_time', 0.0),
                    specializations=perf_data.get('specializations', [])
                )

            logger.info(f"Loaded performance for {len(self.agent_performances)} agents")

        except Exception as e:
            logger.warning(f"Could not load agent performance: {e}")

    def _save_sessions(self):
        """Save session learnings to disk."""
        try:
            # Global sessions
            data = [asdict(s) for s in self.session_learnings]
            with open(self._get_sessions_path(), 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Project-specific sessions
            project_sessions = [
                s for s in self.session_learnings
                if s.workspace == str(self.workspace_path)
            ]
            project_data = [asdict(s) for s in project_sessions]
            with open(self._get_project_sessions_path(), 'w') as f:
                json.dump(project_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Could not save sessions: {e}")

    def _load_sessions(self):
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
                    agent_performances=session_data.get('agent_performances', {}),
                    fixes_applied=session_data.get('fixes_applied', []),
                    stigmergy_signals=session_data.get('stigmergy_signals', 0),
                    total_time=session_data.get('total_time', 0),
                    success=session_data.get('success', False),
                    workspace=session_data.get('workspace', '')
                ))

            logger.info(f"Loaded {len(self.session_learnings)} sessions")

        except Exception as e:
            logger.warning(f"Could not load sessions: {e}")

    def _load_all(self):
        """Load all learning data from disk."""
        self._load_fix_database()
        self._load_agent_performance()
        self._load_sessions()

    def save_all(self):
        """Save all learning data to disk."""
        self._save_fix_database()
        self._save_agent_performance()
        self._save_sessions()
        logger.info("💾 All MAS learnings saved")

    # =========================================================================
    # Integration with SwarmTerminal
    # =========================================================================

    def integrate_with_terminal(self, swarm_terminal) -> None:
        """
        Integrate with SwarmTerminal to persist fix learnings.

        Args:
            swarm_terminal: SwarmTerminal instance
        """
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
        """
        Sync fix learnings from SwarmTerminal to database.

        Args:
            swarm_terminal: SwarmTerminal instance

        Returns:
            Number of new fixes synced
        """
        if not swarm_terminal:
            return 0

        new_fixes = 0

        # Sync from fix history
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
    # Integration with HierarchicalMemory
    # =========================================================================

    def enable_memory_persistence(self, swarm_memory, agent_name: str = "SwarmShared"):
        """
        Enable persistence for HierarchicalMemory.

        Args:
            swarm_memory: HierarchicalMemory instance
            agent_name: Name for the memory persistence
        """
        try:
            from ...memory.memory_persistence import enable_memory_persistence

            persistence_dir = self.learning_dir / 'memories' / agent_name
            persistence = enable_memory_persistence(swarm_memory, persistence_dir)

            logger.info(f"Enabled memory persistence for {agent_name} at {persistence_dir}")
            return persistence

        except Exception as e:
            logger.warning(f"Could not enable memory persistence: {e}")
            return None

    # =========================================================================
    # Statistics & Reporting
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'fix_database': {
                'total_fixes': len(self.fix_database),
                'avg_success_rate': sum(f.success_rate for f in self.fix_database.values()) / max(len(self.fix_database), 1),
                'by_source': dict(defaultdict(int, {
                    fix.source: sum(1 for f in self.fix_database.values() if f.source == fix.source)
                    for fix in self.fix_database.values()
                }))
            },
            'agent_performance': {
                'total_agents': len(self.agent_performances),
                'total_tasks': sum(p.total_tasks for p in self.agent_performances.values()),
                'overall_success_rate': sum(p.total_success for p in self.agent_performances.values()) /
                                        max(sum(p.total_tasks for p in self.agent_performances.values()), 1),
                'specializations': self.get_agent_specializations()
            },
            'sessions': {
                'total_sessions': len(self.session_learnings),
                'successful_sessions': sum(1 for s in self.session_learnings if s.success),
                'avg_time': sum(s.total_time for s in self.session_learnings) / max(len(self.session_learnings), 1)
            }
        }


# Convenience function for quick integration
def get_mas_learning(config=None, workspace_path=None) -> MASLearning:
    """Get or create MAS Learning instance."""
    return MASLearning(config=config, workspace_path=workspace_path)
