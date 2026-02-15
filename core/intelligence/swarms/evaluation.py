"""
Evaluation Infrastructure
==========================

Persistent storage and tracking for gold standards, improvements, and evaluations:
- GoldStandardDB: Database for storing and retrieving gold standards
- ImprovementHistory: Tracks improvement suggestions and their outcomes
- EvaluationHistory: Persistent evaluation tracking across sessions

Extracted from base_swarm.py for modularity.
"""

import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from .swarm_types import GoldStandard, ImprovementSuggestion, AgentRole

logger = logging.getLogger(__name__)


# =============================================================================
# GOLD STANDARD DATABASE
# =============================================================================

class GoldStandardDB:
    """
    Database for storing and retrieving gold standards.

    Supports:
    - JSON file storage
    - In-memory caching
    - Version tracking
    - Domain filtering
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else Path.home() / "jotty" / "gold_standards"
        self.path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, GoldStandard] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all gold standards into cache."""
        for file in self.path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    gs = GoldStandard(
                        id=data['id'],
                        domain=data['domain'],
                        task_type=data['task_type'],
                        input_data=data['input_data'],
                        expected_output=data['expected_output'],
                        evaluation_criteria=data['evaluation_criteria'],
                        created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
                        version=data.get('version', 1)
                    )
                    self._cache[gs.id] = gs
            except Exception as e:
                logger.warning(f"Failed to load gold standard from {file}: {e}")

    def add(self, gold_standard: GoldStandard) -> str:
        """Add a gold standard to the database."""
        # Generate ID if not provided
        if not gold_standard.id:
            content = json.dumps({
                'domain': gold_standard.domain,
                'task_type': gold_standard.task_type,
                'input_data': gold_standard.input_data
            }, sort_keys=True)
            gold_standard.id = hashlib.md5(content.encode()).hexdigest()[:12]

        # Save to file
        file_path = self.path / f"{gold_standard.id}.json"
        with open(file_path, 'w') as f:
            json.dump({
                'id': gold_standard.id,
                'domain': gold_standard.domain,
                'task_type': gold_standard.task_type,
                'input_data': gold_standard.input_data,
                'expected_output': gold_standard.expected_output,
                'evaluation_criteria': gold_standard.evaluation_criteria,
                'created_at': gold_standard.created_at.isoformat(),
                'version': gold_standard.version
            }, f, indent=2)

        self._cache[gold_standard.id] = gold_standard
        return gold_standard.id

    def get(self, id: str) -> Optional[GoldStandard]:
        """Get a gold standard by ID."""
        return self._cache.get(id)

    def find_by_domain(self, domain: str) -> List[GoldStandard]:
        """Find all gold standards for a domain."""
        return [gs for gs in self._cache.values() if gs.domain == domain]

    def find_similar(self, task_type: str, input_data: Dict[str, Any]) -> Optional[GoldStandard]:
        """Find the most similar gold standard for evaluation."""
        candidates = [gs for gs in self._cache.values() if gs.task_type == task_type]
        if not candidates:
            return None
        # Simple similarity: prefer exact task_type match
        return candidates[0]

    def list_all(self) -> List[GoldStandard]:
        """List all gold standards."""
        return list(self._cache.values())


# =============================================================================
# IMPROVEMENT HISTORY
# =============================================================================

class ImprovementHistory:
    """
    Tracks improvement suggestions and their outcomes.

    Used for:
    - Learning what improvements work
    - Avoiding repeated failed improvements
    - Measuring improvement velocity
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else Path.home() / "jotty" / "improvements"
        self.path.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load improvement history."""
        history_file = self.path / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.history = json.load(f)

    def _save_history(self) -> None:
        """Save improvement history."""
        history_file = self.path / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def record_suggestion(self, suggestion: ImprovementSuggestion) -> str:
        """Record an improvement suggestion."""
        entry = {
            'id': hashlib.md5(f"{suggestion.agent_role.value}:{suggestion.description}:{datetime.now()}".encode()).hexdigest()[:12],
            'suggestion': asdict(suggestion),
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'applied_at': None,
            'outcome': None,
            'impact_measured': None
        }
        self.history.append(entry)
        self._save_history()
        return entry['id']

    def mark_applied(self, suggestion_id: str) -> None:
        """Mark a suggestion as applied."""
        for entry in self.history:
            if entry['id'] == suggestion_id:
                entry['status'] = 'applied'
                entry['applied_at'] = datetime.now().isoformat()
                break
        self._save_history()

    def record_outcome(self, suggestion_id: str, success: bool, impact: float, notes: str = "") -> None:
        """Record the outcome of an applied improvement."""
        for entry in self.history:
            if entry['id'] == suggestion_id:
                entry['status'] = 'completed'
                entry['outcome'] = 'success' if success else 'failure'
                entry['impact_measured'] = impact
                entry['notes'] = notes
                break
        self._save_history()

    def get_successful_improvements(self, agent_role: Optional[AgentRole] = None) -> List[Dict]:
        """Get successful improvements, optionally filtered by role."""
        successful = [e for e in self.history if e.get('outcome') == 'success']
        if agent_role:
            successful = [e for e in successful if e['suggestion']['agent_role'] == agent_role.value]
        return successful

    def get_pending_suggestions(self) -> List[Dict]:
        """Get pending suggestions."""
        return [e for e in self.history if e['status'] == 'pending']


class EvaluationHistory:
    """Persistent evaluation tracking across sessions.
    Follows same pattern as ImprovementHistory."""

    def __init__(self, path: Any = None) -> None:
        self.path = Path(path) if path else Path.home() / "jotty" / "evaluations"
        self.path.mkdir(parents=True, exist_ok=True)
        self.evaluations: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        history_file = self.path / "evaluations.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.evaluations = json.load(f)

    def _save(self) -> None:
        history_file = self.path / "evaluations.json"
        with open(history_file, 'w') as f:
            json.dump(self.evaluations[-200:], f, indent=2, default=str)

    def record(self, evaluation: Any) -> None:
        entry = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': evaluation.overall_score if hasattr(evaluation, 'overall_score') else 0,
            'status': evaluation.status if hasattr(evaluation, 'status') else 'unknown',
            'scores': evaluation.dimension_scores if hasattr(evaluation, 'dimension_scores') else {},
            'feedback': evaluation.feedback if hasattr(evaluation, 'feedback') else '',
        }
        self.evaluations.append(entry)
        self._save()

    def get_recent(self, n: Any = 10) -> List[Dict]:
        return self.evaluations[-n:]

    def get_average_score(self, n: Any = 10) -> float:
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return sum(e.get('overall_score', 0) for e in recent) / len(recent)

    def get_failures(self, n: Any = 20) -> List[Dict]:
        """Get recent failures for failure recovery analysis."""
        return [e for e in self.evaluations[-n:] if e.get('overall_score', 1.0) < 0.5]


__all__ = [
    'GoldStandardDB',
    'ImprovementHistory',
    'EvaluationHistory',
]
