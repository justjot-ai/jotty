"""
Transferable Learning for Swarm Systems
========================================

Implements learning that transfers across:
- Different agent combinations
- Different goals/queries
- Different domains

Key Components:
1. SEMANTIC EMBEDDINGS: Vector similarity for matching
2. ABSTRACT PATTERNS: Task-type, error-type, role-based learning
3. META-LEARNING: Learn how to learn
4. HIERARCHICAL MEMORY: Specific → Pattern → Meta levels
"""

import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Lazy imports: sentence-transformers (5s) and numpy are loaded on first use,
# not at module import time. This saves ~5s on every `import transfer_learning`.
import os
EMBEDDINGS_DISABLED = os.environ.get('JOTTY_DISABLE_EMBEDDINGS', '').lower() in ('1', 'true', 'yes')

# These are set True when lazy import succeeds (in SemanticEmbedder._ensure_model)
EMBEDDINGS_AVAILABLE = not EMBEDDINGS_DISABLED  # Assume available; checked on first use
NUMPY_AVAILABLE = True  # Assume available; checked on first use
SentenceTransformer = None  # Loaded lazily


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AbstractPattern:
    """An abstracted, transferable pattern."""
    pattern_id: str
    level: str  # 'task', 'error', 'workflow', 'meta'
    pattern_type: str  # e.g., 'COUNT_QUERY', 'COLUMN_NOT_FOUND', 'RETRY_STRATEGY'
    description: str
    success_count: int = 0
    failure_count: int = 0
    total_reward: float = 0.0
    contexts: List[Dict] = field(default_factory=list)  # Where this pattern worked
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def avg_reward(self) -> float:
        total = self.success_count + self.failure_count
        return self.total_reward / total if total > 0 else 0.0


@dataclass
class RoleProfile:
    """Learned profile for an agent ROLE (not specific agent name)."""
    role: str  # e.g., 'sql_generator', 'validator', 'planner'
    strengths: List[str] = field(default_factory=list)  # Task types it excels at
    weaknesses: List[str] = field(default_factory=list)  # Task types it struggles with
    success_by_task_type: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # task_type -> (success, total)
    avg_execution_time: float = 0.0
    cooperation_score: float = 0.5


@dataclass
class MetaPattern:
    """Meta-learning pattern: learning about learning."""
    pattern_id: str
    trigger: str  # When to apply this meta-pattern
    strategy: str  # What to do
    success_rate: float = 0.5
    applications: int = 0


# =============================================================================
# SEMANTIC EMBEDDING ENGINE
# =============================================================================

class SemanticEmbedder:
    """
    Provides semantic similarity using embeddings.

    Falls back to enhanced keyword matching if sentence-transformers unavailable.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_embeddings: bool = True):
        self.model = None
        self._model_name = model_name
        self._use_embeddings = use_embeddings
        self._model_loaded = False  # Lazy: don't load 9s model until first embed()
        self.cache: Dict[str, Any] = {}  # text -> embedding
        self.cache_max_size = 10000
        logger.info(f"SemanticEmbedder initialized with {model_name} (lazy)")

    def _ensure_model(self):
        """Load sentence-transformers + model on first use (saves ~14s on startup)."""
        if self._model_loaded:
            return
        self._model_loaded = True

        global EMBEDDINGS_AVAILABLE, NUMPY_AVAILABLE, SentenceTransformer

        if EMBEDDINGS_DISABLED or not self._use_embeddings:
            EMBEDDINGS_AVAILABLE = False
            return

        # Lazy import: sentence_transformers takes ~5s, model load ~3s
        try:
            import sys, io, os
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['TQDM_DISABLE'] = '1'

            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            import logging as _logging
            for _name in ['safetensors', 'sentence_transformers', 'transformers', 'huggingface_hub', 'tqdm']:
                _logging.getLogger(_name).setLevel(_logging.ERROR)

            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST

            _real_stdout, _real_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                self.model = SentenceTransformer(self._model_name, trust_remote_code=False)
            finally:
                sys.stdout = _real_stdout
                sys.stderr = _real_stderr

            logger.info(f"SemanticEmbedder model loaded: {self._model_name}")
        except ImportError:
            EMBEDDINGS_AVAILABLE = False
            logger.info("sentence-transformers not available, using fallback similarity")
        except Exception as e:
            EMBEDDINGS_AVAILABLE = False
            logger.debug(f"Could not load embedding model: {e}, using fallback")

        # Lazy numpy check
        try:
            import numpy as _np
            NUMPY_AVAILABLE = True
        except ImportError:
            NUMPY_AVAILABLE = False

    def embed(self, text: str) -> Any:
        """Get embedding for text (cached). Loads model on first call."""
        if text in self.cache:
            return self.cache[text]

        self._ensure_model()  # Lazy load

        if self.model and NUMPY_AVAILABLE:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: create pseudo-embedding from word frequencies
            embedding = self._fallback_embed(text)

        # Cache management
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest 10%
            keys_to_remove = list(self.cache.keys())[:self.cache_max_size // 10]
            for k in keys_to_remove:
                del self.cache[k]

        self.cache[text] = embedding
        return embedding

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if text1 == text2:
            return 1.0

        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        if NUMPY_AVAILABLE and self.model:
            # Cosine similarity
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                return float(dot / (norm1 * norm2))
            return 0.0
        else:
            # Fallback similarity
            return self._fallback_similarity(emb1, emb2)

    def find_similar(self, query: str, candidates: List[str], threshold: float = 0.7, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar texts from candidates."""
        results = []
        for candidate in candidates:
            sim = self.similarity(query, candidate)
            if sim >= threshold:
                results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _fallback_embed(self, text: str) -> Dict[str, float]:
        """Fallback: word frequency vector."""
        words = text.lower().split()
        freq = defaultdict(float)
        for w in words:
            if len(w) > 2:  # Skip very short words
                freq[w] += 1.0
        # Normalize
        total = sum(freq.values())
        if total > 0:
            for k in freq:
                freq[k] /= total
        return dict(freq)

    def _fallback_similarity(self, emb1: Dict, emb2: Dict) -> float:
        """Fallback: Jaccard-like similarity on word vectors."""
        if not emb1 or not emb2:
            return 0.0

        all_keys = set(emb1.keys()) | set(emb2.keys())
        if not all_keys:
            return 0.0

        dot = sum(emb1.get(k, 0) * emb2.get(k, 0) for k in all_keys)
        norm1 = sum(v**2 for v in emb1.values()) ** 0.5
        norm2 = sum(v**2 for v in emb2.values()) ** 0.5

        if norm1 > 0 and norm2 > 0:
            return dot / (norm1 * norm2)
        return 0.0


# =============================================================================
# PATTERN EXTRACTOR
# =============================================================================

class PatternExtractor:
    """
    Extracts abstract, transferable patterns from experiences.

    Converts specific experiences into general patterns:
    - "count P2P users yesterday" → TASK_TYPE: aggregation, TIME: relative
    - "sql_agent failed on join" → ROLE: sql_generator, WEAKNESS: complex_joins
    """

    # Task type keywords (domain-agnostic)
    # Order matters: more specific types checked first.
    # Uses word-boundary matching (see extract_task_type) to prevent
    # "sum" matching "summarize", "search" matching "research", etc.
    #
    # NOTE: The LLM inference mixin uses a separate TaskType enum with
    # 'creation' instead of 'generation'. We normalize via TASK_TYPE_ALIASES
    # so stigmergy/transfer learning lookups match regardless of source.
    TASK_TYPES = {
        'comparison': ['compare', 'comparing', 'diff', 'contrast', 'versus', 'vs', 'pros and cons'],
        'analysis': ['analyze', 'analyse', 'examining', 'investigate', 'explore', 'research'],
        'aggregation': ['count', 'sum up', 'avg', 'average', 'total', 'aggregate', 'tally'],
        'filtering': ['filter', 'where', 'find', 'select'],
        'transformation': ['transform', 'convert', 'map', 'process', 'clean'],
        'prediction': ['predict', 'forecast', 'estimate', 'project'],
        'validation': ['validate', 'verify', 'check', 'confirm', 'audit'],
        'generation': ['generate', 'create', 'produce', 'build', 'make', 'write', 'summarize', 'summarise', 'scrape', 'script'],
    }

    # Normalize aliases between different classifiers (LLM vs keyword)
    TASK_TYPE_ALIASES = {
        'creation': 'generation',   # LLM inference says 'creation', we store as 'generation'
        'research': 'analysis',     # LLM inference says 'research', we store as 'analysis'
        'automation': 'generation', # LLM inference says 'automation', we store as 'generation'
    }

    # Time pattern keywords
    TIME_PATTERNS = {
        'relative_past': ['yesterday', 'last', 'previous', 'ago', 'recent'],
        'relative_future': ['tomorrow', 'next', 'upcoming', 'future'],
        'absolute': ['2024', '2023', 'january', 'february', 'q1', 'q2', 'q3', 'q4'],
        'range': ['between', 'from', 'to', 'through', 'mtd', 'ytd', 'wtd'],
        'current': ['today', 'now', 'current', 'this'],
    }

    # Role inference from agent behavior/name
    ROLE_KEYWORDS = {
        'sql_generator': ['sql', 'query', 'database', 'db'],
        'validator': ['valid', 'check', 'audit', 'verify', 'inspect'],
        'planner': ['plan', 'architect', 'design', 'strategy'],
        'executor': ['exec', 'run', 'process', 'worker'],
        'analyzer': ['analy', 'insight', 'report'],
        'transformer': ['transform', 'convert', 'etl', 'pipeline'],
    }

    @classmethod
    def normalize_task_type(cls, task_type: str) -> str:
        """Normalize task type aliases to canonical form.
        
        The LLM inference mixin returns 'creation'/'research'/'automation'
        but stigmergy/transfer learning stores 'generation'/'analysis'.
        This ensures consistent lookups regardless of source.
        """
        if not task_type:
            return 'general'
        canonical = task_type.lower().strip()
        return cls.TASK_TYPE_ALIASES.get(canonical, canonical)

    def extract_task_type(self, query: str) -> str:
        """Extract abstract task type from query using word-boundary matching.
        
        Prevents false positives like 'sum' matching 'summarize' or
        'search' matching 'research'. Multi-word keywords (like 'pros and cons')
        use plain substring matching since they're already specific enough.
        """
        import re
        query_lower = query.lower()
        for task_type, keywords in self.TASK_TYPES.items():
            for kw in keywords:
                if ' ' in kw:
                    # Multi-word: substring match is fine (already specific)
                    if kw in query_lower:
                        return task_type
                else:
                    # Single-word: use word boundary to prevent partial matches
                    if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                        return task_type
        return 'general'

    def extract_time_pattern(self, query: str) -> str:
        """Extract time pattern type."""
        query_lower = query.lower()
        for pattern, keywords in self.TIME_PATTERNS.items():
            if any(kw in query_lower for kw in keywords):
                return pattern
        return 'none'

    def extract_role(self, agent_name: str, task_types_handled: List[str] = None) -> str:
        """Infer role from agent name or behavior."""
        name_lower = agent_name.lower()

        # Try name-based inference
        for role, keywords in self.ROLE_KEYWORDS.items():
            if any(kw in name_lower for kw in keywords):
                return role

        # Try behavior-based inference
        if task_types_handled:
            if 'validation' in task_types_handled:
                return 'validator'
            if 'analysis' in task_types_handled:
                return 'analyzer'
            if 'transformation' in task_types_handled:
                return 'transformer'

        return 'general'

    def extract_error_type(self, error: str) -> str:
        """Extract abstract error type."""
        error_lower = error.lower()

        if 'column' in error_lower and ('not found' in error_lower or 'missing' in error_lower):
            return 'COLUMN_NOT_FOUND'
        if 'timeout' in error_lower or 'timed out' in error_lower:
            return 'TIMEOUT'
        if 'permission' in error_lower or 'access denied' in error_lower:
            return 'PERMISSION_DENIED'
        if 'connection' in error_lower or 'network' in error_lower:
            return 'CONNECTION_ERROR'
        if 'syntax' in error_lower or 'parse' in error_lower:
            return 'SYNTAX_ERROR'
        if 'memory' in error_lower or 'oom' in error_lower:
            return 'MEMORY_ERROR'

        return 'UNKNOWN_ERROR'

    def abstract_state(self, state: Dict) -> Dict:
        """Convert specific state to abstract representation."""
        query = state.get('query', '')
        agent = state.get('agent', '')
        error = state.get('error', '')

        abstract = {
            'task_type': self.extract_task_type(query),
            'time_pattern': self.extract_time_pattern(query),
            'role': self.extract_role(agent),
            'has_error': bool(error),
            'error_type': self.extract_error_type(error) if error else None,
            'success': state.get('success', None),
        }

        return abstract


# =============================================================================
# TRANSFERABLE LEARNING STORE
# =============================================================================

class TransferableLearningStore:
    """
    Stores learnings that transfer across swarm configurations.

    Hierarchical storage:
    - Level 1: Specific experiences (exact matches)
    - Level 2: Abstract patterns (task types, error types)
    - Level 3: Meta patterns (learning strategies)
    """

    def __init__(self, config=None):
        self.config = config
        self.embedder = SemanticEmbedder()
        self.extractor = PatternExtractor()

        # Level 1: Specific experiences (with embeddings)
        self.experiences: List[Dict] = []
        self.experience_embeddings: Dict[str, Any] = {}  # exp_id -> embedding

        # Level 2: Abstract patterns
        self.task_patterns: Dict[str, AbstractPattern] = {}  # task_type -> pattern
        self.error_patterns: Dict[str, AbstractPattern] = {}  # error_type -> pattern
        self.workflow_patterns: Dict[str, AbstractPattern] = {}  # workflow_id -> pattern

        # Level 3: Meta patterns
        self.meta_patterns: Dict[str, MetaPattern] = {}

        # Role profiles (transferable across agent names)
        self.role_profiles: Dict[str, RoleProfile] = {}

        logger.info("TransferableLearningStore initialized")

    def record_experience(
        self,
        query: str,
        agent: str,
        action: str,
        reward: float,
        success: bool,
        error: str = None,
        context: Dict = None
    ):
        """Record an experience and extract transferable patterns."""
        # Create experience record
        exp_id = hashlib.md5(f"{query}:{agent}:{time.time()}".encode()).hexdigest()[:12]
        experience = {
            'id': exp_id,
            'query': query,
            'agent': agent,
            'action': action,
            'reward': reward,
            'success': success,
            'error': error,
            'context': context or {},
            'timestamp': time.time()
        }

        # Store specific experience
        self.experiences.append(experience)
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]  # Keep recent

        # Embed for similarity search
        self.experience_embeddings[exp_id] = self.embedder.embed(query)

        # Extract and update abstract patterns
        self._update_task_pattern(query, success, reward)
        if error:
            self._update_error_pattern(error, action, success, reward)
        self._update_role_profile(agent, query, success, reward)

        # Check for meta-patterns
        self._check_meta_patterns(experience)

    def _update_task_pattern(self, query: str, success: bool, reward: float):
        """Update task-type pattern from experience."""
        task_type = self.extractor.extract_task_type(query)

        if task_type not in self.task_patterns:
            self.task_patterns[task_type] = AbstractPattern(
                pattern_id=f"task_{task_type}",
                level='task',
                pattern_type=task_type,
                description=f"Pattern for {task_type} tasks"
            )

        pattern = self.task_patterns[task_type]
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1
        pattern.total_reward += reward
        pattern.last_used = time.time()

    def _update_error_pattern(self, error: str, recovery_action: str, success: bool, reward: float):
        """Update error recovery pattern."""
        error_type = self.extractor.extract_error_type(error)

        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = AbstractPattern(
                pattern_id=f"error_{error_type}",
                level='error',
                pattern_type=error_type,
                description=f"Recovery pattern for {error_type}"
            )

        pattern = self.error_patterns[error_type]
        if success:
            pattern.success_count += 1
            # Record successful recovery
            pattern.contexts.append({
                'recovery_action': recovery_action,
                'success': True,
                'timestamp': time.time()
            })
        else:
            pattern.failure_count += 1
        pattern.total_reward += reward
        pattern.last_used = time.time()

        # Keep only recent contexts
        if len(pattern.contexts) > 20:
            pattern.contexts = pattern.contexts[-20:]

    def _update_role_profile(self, agent: str, query: str, success: bool, reward: float):
        """Update role profile based on agent performance."""
        role = self.extractor.extract_role(agent)
        task_type = self.extractor.extract_task_type(query)

        if role not in self.role_profiles:
            self.role_profiles[role] = RoleProfile(role=role)

        profile = self.role_profiles[role]

        # Update success by task type
        if task_type not in profile.success_by_task_type:
            profile.success_by_task_type[task_type] = (0, 0)

        succ, total = profile.success_by_task_type[task_type]
        profile.success_by_task_type[task_type] = (succ + (1 if success else 0), total + 1)

        # Update strengths/weaknesses
        self._update_role_strengths_weaknesses(profile)

    def _update_role_strengths_weaknesses(self, profile: RoleProfile):
        """Determine role strengths and weaknesses from history."""
        strengths = []
        weaknesses = []

        for task_type, (succ, total) in profile.success_by_task_type.items():
            if total >= 3:  # Need enough data
                rate = succ / total
                if rate >= 0.7:
                    strengths.append(task_type)
                elif rate <= 0.3:
                    weaknesses.append(task_type)

        profile.strengths = strengths
        profile.weaknesses = weaknesses

    def _check_meta_patterns(self, experience: Dict):
        """Check if experience triggers or creates meta-patterns."""
        # Meta-pattern: Repeated failures suggest strategy change
        recent_failures = sum(1 for exp in self.experiences[-5:] if not exp.get('success'))

        if recent_failures >= 3:
            meta_id = "retry_strategy_change"
            if meta_id not in self.meta_patterns:
                self.meta_patterns[meta_id] = MetaPattern(
                    pattern_id=meta_id,
                    trigger="3+ consecutive failures",
                    strategy="Change approach: try different agent role or decompose task"
                )
            self.meta_patterns[meta_id].applications += 1

        # Meta-pattern: Low confidence should trigger more context gathering
        confidence = experience.get('context', {}).get('confidence', 1.0)
        if confidence < 0.5:
            meta_id = "low_confidence_gather"
            if meta_id not in self.meta_patterns:
                self.meta_patterns[meta_id] = MetaPattern(
                    pattern_id=meta_id,
                    trigger="confidence < 0.5",
                    strategy="Gather more context before proceeding"
                )

    def get_relevant_learnings(self, query: str, agent: str = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Get relevant transferable learnings for a query.

        Returns learnings at all levels:
        - Similar specific experiences
        - Relevant task patterns
        - Applicable error patterns
        - Role recommendations
        - Meta-pattern advice
        """
        result = {
            'similar_experiences': [],
            'task_pattern': None,
            'error_patterns': [],
            'role_advice': None,
            'meta_advice': [],
        }

        # 1. Find similar experiences using embeddings
        if self.experiences:
            query_embedding = self.embedder.embed(query)
            similarities = []

            for exp in self.experiences:
                exp_emb = self.experience_embeddings.get(exp['id'])
                if exp_emb is not None:
                    sim = self.embedder.similarity(query, exp['query'])
                    if sim > 0.5:  # Threshold
                        similarities.append((exp, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            result['similar_experiences'] = [
                {
                    'query': exp['query'],
                    'success': exp['success'],
                    'reward': exp['reward'],
                    'similarity': sim,
                    'lesson': f"{'SUCCESS' if exp['success'] else 'FAILED'}: {exp['query'][:50]}..."
                }
                for exp, sim in similarities[:top_k]
            ]

        # 2. Get task pattern
        task_type = self.extractor.extract_task_type(query)
        if task_type in self.task_patterns:
            pattern = self.task_patterns[task_type]
            result['task_pattern'] = {
                'task_type': task_type,
                'success_rate': pattern.success_rate,
                'avg_reward': pattern.avg_reward,
                'advice': f"{(task_type or 'UNKNOWN').upper()} queries have {pattern.success_rate*100:.0f}% success rate"
            }

        # 3. Get role advice
        if agent:
            role = self.extractor.extract_role(agent)
            if role in self.role_profiles:
                profile = self.role_profiles[role]
                result['role_advice'] = {
                    'role': role,
                    'strengths': profile.strengths,
                    'weaknesses': profile.weaknesses,
                    'advice': f"Role '{role}' excels at: {', '.join(profile.strengths) or 'unknown'}"
                }

                # Warn if task type is a weakness
                if task_type in profile.weaknesses:
                    result['role_advice']['warning'] = f"This role struggles with {task_type} tasks"

        # 4. Get meta-pattern advice
        for meta in self.meta_patterns.values():
            result['meta_advice'].append({
                'trigger': meta.trigger,
                'strategy': meta.strategy,
                'applications': meta.applications
            })

        return result

    def get_best_role_for_task(self, task_type: str) -> Optional[str]:
        """Recommend best role for a task type based on learned profiles."""
        best_role = None
        best_rate = 0.0

        for role, profile in self.role_profiles.items():
            if task_type in profile.success_by_task_type:
                succ, total = profile.success_by_task_type[task_type]
                if total >= 2:
                    rate = succ / total
                    if rate > best_rate:
                        best_rate = rate
                        best_role = role

        return best_role

    def format_context_for_agent(self, query: str, agent: str = None) -> str:
        """Format transferable learnings as context for agent prompt."""
        learnings = self.get_relevant_learnings(query, agent)

        lines = ["# Transferable Learnings (from past swarm executions):\n"]

        # Similar experiences
        if learnings['similar_experiences']:
            lines.append("## Similar Past Queries:")
            for exp in learnings['similar_experiences'][:3]:
                status = "OK" if exp['success'] else "FAIL"
                lines.append(f"  {status} {exp['lesson']} (similarity: {exp['similarity']:.0%})")

        # Task pattern
        if learnings['task_pattern']:
            tp = learnings['task_pattern']
            lines.append(f"\n## Task Type Pattern ({tp['task_type']}):")
            lines.append(f"  - Success rate: {tp['success_rate']*100:.0f}%")
            lines.append(f"  - {tp['advice']}")

        # Role advice
        if learnings['role_advice']:
            ra = learnings['role_advice']
            lines.append(f"\n## Role Advice ({ra['role']}):")
            lines.append(f"  - {ra['advice']}")
            if 'warning' in ra:
                lines.append(f" - {ra['warning']}")

        # Meta advice
        if learnings['meta_advice']:
            lines.append("\n## Meta-Learning Advice:")
            for ma in learnings['meta_advice'][:2]:
                lines.append(f"  - When {ma['trigger']}: {ma['strategy']}")

        return "\n".join(lines)

    # =========================================================================
    # Session History & Task Scoring (consolidated from MASLearning)
    # =========================================================================

    def record_session(
        self,
        task_description: str,
        agents_used: List[str],
        total_time: float,
        success: bool,
        stigmergy_signals: int = 0,
        output_quality: float = 0.0
    ):
        """Record a session for task relevance matching."""
        if not hasattr(self, 'sessions'):
            self.sessions = []

        # Extract topics for matching
        topics = self._extract_topics(task_description)

        session = {
            'session_id': hashlib.md5(f"{task_description}:{time.time()}".encode()).hexdigest()[:12],
            'timestamp': time.time(),
            'task_description': task_description,
            'task_topics': topics,
            'agents_used': agents_used,
            'stigmergy_signals': stigmergy_signals,
            'total_time': total_time,
            'success': success,
            'output_quality': output_quality
        }

        self.sessions.append(session)

        # Keep only last 100 sessions
        if len(self.sessions) > 100:
            self.sessions = self.sessions[-100:]

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text for similarity matching."""
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        stopwords = {'this', 'that', 'with', 'from', 'have', 'will', 'would', 'could',
                    'should', 'being', 'been', 'were', 'what', 'when', 'where', 'which',
                    'their', 'them', 'then', 'than', 'these', 'those', 'some', 'such',
                    'only', 'other', 'into', 'over', 'also', 'more', 'most', 'very'}

        topics = [w for w in words if w not in stopwords]

        from collections import Counter
        topic_counts = Counter(topics)
        return [t for t, _ in topic_counts.most_common(10)]

    def score_task_relevance(self, task_description: str, session: Dict) -> float:
        """
        Score relevance for EXECUTION - find best strategy to copy.

        Optimized for: recent successful similar tasks.
        - Prefers recent sessions (recency decay)
        - Prefers successful sessions (success bonus)
        - Topic similarity matters
        """
        query_topics = self._extract_topics(task_description)
        session_topics = session.get('task_topics', [])

        # Topic overlap score
        topic_overlap = len(set(query_topics) & set(session_topics))
        topic_score = topic_overlap / max(len(query_topics), 1)

        # Recency score (decay over 30 days)
        age_seconds = time.time() - session.get('timestamp', 0)
        age_days = age_seconds / 86400
        recency_score = max(0, 1 - age_days / 30)

        # Success bonus
        success_bonus = 0.2 if session.get('success', False) else 0

        return 0.5 * topic_score + 0.3 * recency_score + 0.2 + success_bonus

    def score_learning_relevance(self, task_description: str, session: Dict) -> float:
        """
        Score relevance for LEARNING - find lessons to apply.

        Optimized for: learning from both successes and failures.
        - No recency decay (old lessons still valuable)
        - Weights failures higher (learn from mistakes)
        - Higher weight on topic similarity
        - Considers output quality
        """
        query_topics = self._extract_topics(task_description)
        session_topics = session.get('task_topics', [])

        # Topic overlap score (higher weight for learning)
        topic_overlap = len(set(query_topics) & set(session_topics))
        topic_score = topic_overlap / max(len(query_topics), 1)

        # Failure bonus - learn more from mistakes
        success = session.get('success', False)
        outcome_score = 0.3 if not success else 0.1

        # Quality score - high quality outputs are better lessons
        quality = session.get('output_quality', 0.5)
        quality_score = quality * 0.2

        # Diversity bonus - sessions with more agents provide richer lessons
        agents_used = session.get('agents_used', [])
        diversity_score = min(len(agents_used) / 5, 1.0) * 0.1

        return 0.5 * topic_score + outcome_score + quality_score + diversity_score

    def get_learning_sessions(self, task_description: str, top_k: int = 10) -> List[Dict]:
        """
        Get sessions most relevant for LEARNING about a task type.

        Returns both successes and failures to learn from.
        Uses score_learning_relevance() for scoring.
        """
        if not hasattr(self, 'sessions') or not self.sessions:
            return []

        scored = [
            (session, self.score_learning_relevance(task_description, session))
            for session in self.sessions
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return sessions above threshold, ensuring mix of outcomes
        results = []
        successes = 0
        failures = 0

        for session, score in scored:
            if score < 0.2:
                continue
            if len(results) >= top_k:
                break

            is_success = session.get('success', False)
            # Ensure we get at least some of each outcome type
            if is_success:
                if successes < top_k // 2 or failures >= top_k // 2:
                    results.append(session)
                    successes += 1
            else:
                if failures < top_k // 2 or successes >= top_k // 2:
                    results.append(session)
                    failures += 1

        return results

    def get_relevant_sessions(self, task_description: str, top_k: int = 5) -> List[Dict]:
        """
        Get sessions most relevant for EXECUTION strategy.

        Returns recent successful similar tasks to inform execution.
        Uses score_task_relevance() for scoring.
        """
        if not hasattr(self, 'sessions') or not self.sessions:
            return []

        scored = [
            (session, self.score_task_relevance(task_description, session))
            for session in self.sessions
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s for s, score in scored[:top_k] if score > 0.3]

    def get_execution_strategy(self, task_description: str, available_agents: List[str]) -> Dict:
        """Get recommended execution strategy based on learnings."""
        relevant = self.get_relevant_sessions(task_description, top_k=5)

        # Aggregate agent performance from relevant sessions
        agent_success = {}
        for session in relevant:
            for agent in session.get('agents_used', []):
                if agent not in agent_success:
                    agent_success[agent] = {'success': 0, 'total': 0, 'time': 0}
                agent_success[agent]['total'] += 1
                if session.get('success', False):
                    agent_success[agent]['success'] += 1
                agent_success[agent]['time'] += session.get('total_time', 0) / max(len(session.get('agents_used', [])), 1)

        # Calculate success rates
        agent_scores = {}
        for agent in available_agents:
            if agent in agent_success and agent_success[agent]['total'] >= 2:
                agent_scores[agent] = agent_success[agent]['success'] / agent_success[agent]['total']
            else:
                agent_scores[agent] = 0.5  # Unknown agents get neutral score

        # Identify underperformers
        skip_agents = []
        retry_agents = []
        for agent, score in agent_scores.items():
            if score < 0.4:
                skip_agents.append({'agent': agent, 'reason': f'{score*100:.0f}% success'})
            elif score < 0.6:
                retry_agents.append({'agent': agent, 'success_rate': score})

        # Order by success rate
        recommended_order = sorted(
            [a for a in available_agents if a not in [s['agent'] for s in skip_agents]],
            key=lambda a: agent_scores.get(a, 0.5),
            reverse=True
        )

        # Expected time from similar tasks
        expected_time = 60.0
        if relevant:
            expected_time = sum(s.get('total_time', 60) for s in relevant) / len(relevant)

        return {
            'recommended_order': recommended_order,
            'skip_agents': skip_agents,
            'retry_agents': retry_agents,
            'expected_time': expected_time,
            'confidence': min(1.0, len(relevant) / 5),
            'similar_sessions': len(relevant)
        }

    def save(self, path: str):
        """Save transferable learnings to disk."""
        data = {
            'experiences': self.experiences[-500:],  # Keep recent
            'sessions': getattr(self, 'sessions', [])[-100:],  # Session history
            'task_patterns': {
                k: {
                    'pattern_id': v.pattern_id,
                    'level': v.level,
                    'pattern_type': v.pattern_type,
                    'description': v.description,
                    'success_count': v.success_count,
                    'failure_count': v.failure_count,
                    'total_reward': v.total_reward,
                    'contexts': v.contexts[-10:],
                }
                for k, v in self.task_patterns.items()
            },
            'error_patterns': {
                k: {
                    'pattern_id': v.pattern_id,
                    'level': v.level,
                    'pattern_type': v.pattern_type,
                    'description': v.description,
                    'success_count': v.success_count,
                    'failure_count': v.failure_count,
                    'total_reward': v.total_reward,
                    'contexts': v.contexts[-10:],
                }
                for k, v in self.error_patterns.items()
            },
            'role_profiles': {
                k: {
                    'role': v.role,
                    'strengths': v.strengths,
                    'weaknesses': v.weaknesses,
                    'success_by_task_type': v.success_by_task_type,
                }
                for k, v in self.role_profiles.items()
            },
            'meta_patterns': {
                k: {
                    'pattern_id': v.pattern_id,
                    'trigger': v.trigger,
                    'strategy': v.strategy,
                    'success_rate': v.success_rate,
                    'applications': v.applications,
                }
                for k, v in self.meta_patterns.items()
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved transferable learnings: {len(self.task_patterns)} task patterns, "
                   f"{len(self.role_profiles)} role profiles")

    def load(self, path: str) -> bool:
        """Load transferable learnings from disk."""
        if not Path(path).exists():
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Restore experiences
            self.experiences = data.get('experiences', [])

            # Restore sessions (for task relevance scoring)
            self.sessions = data.get('sessions', [])

            # Re-embed experiences
            for exp in self.experiences:
                self.experience_embeddings[exp['id']] = self.embedder.embed(exp['query'])

            # Restore task patterns
            for k, v in data.get('task_patterns', {}).items():
                self.task_patterns[k] = AbstractPattern(**v)

            # Restore error patterns
            for k, v in data.get('error_patterns', {}).items():
                self.error_patterns[k] = AbstractPattern(**v)

            # Restore role profiles
            for k, v in data.get('role_profiles', {}).items():
                self.role_profiles[k] = RoleProfile(**v)

            # Restore meta patterns
            for k, v in data.get('meta_patterns', {}).items():
                self.meta_patterns[k] = MetaPattern(**v)

            logger.info(f"Loaded transferable learnings: {len(self.task_patterns)} task patterns, "
                       f"{len(self.role_profiles)} role profiles, {len(self.sessions)} sessions")
            return True

        except Exception as e:
            logger.warning(f"Could not load transferable learnings: {e}")
            return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SemanticEmbedder',
    'PatternExtractor',
    'TransferableLearningStore',
    'AbstractPattern',
    'RoleProfile',
    'MetaPattern',
]
