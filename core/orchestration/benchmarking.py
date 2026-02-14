"""
Swarm Benchmarks
=================

Performance benchmarking for swarm intelligence:
- SwarmMetrics: Metrics container for swarm evaluation
- SwarmBenchmarks: Benchmark suite for comparing performance

Tracks single-agent vs multi-agent speedup, communication overhead,
specialization emergence, and cooperation effectiveness.

Extracted from swarm_intelligence.py for modularity.
"""

import time
import math
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .swarm_data_structures import AgentProfile, AgentSpecialization

logger = logging.getLogger(__name__)


@dataclass
class SwarmMetrics:
    """Metrics for evaluating swarm performance."""
    communication_overhead: float = 0.0   # Time spent in inter-agent communication
    specialization_diversity: float = 0.0  # How diverse are agent specializations
    single_vs_multi_ratio: float = 1.0     # Speedup from multi-agent vs single
    cooperation_index: float = 0.0         # How well agents cooperate
    task_distribution_entropy: float = 0.0 # How evenly work is distributed


class SwarmBenchmarks:
    """
    Benchmark suite for comparing swarm performance.

    Tracks:
    - Single-agent vs multi-agent speedup
    - Communication overhead
    - Specialization emergence
    - Cooperation effectiveness
    """

    def __init__(self):
        # Single-agent baselines: task_type -> [(time, success)]
        self.single_agent_runs: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

        # Multi-agent runs: task_type -> [(time, agents_count, success)]
        self.multi_agent_runs: Dict[str, List[Tuple[float, int, bool]]] = defaultdict(list)

        # Communication overhead tracking
        self.communication_events: List[Dict] = []

        # Cooperation events
        self.cooperation_events: List[Dict] = []

        # Iteration history for self-improvement tracking
        self.iteration_history: List[Dict] = []

    def record_iteration(self, iteration_id: str, task_type: str,
                         score: float, execution_time: float, success: bool):
        """Record a self-improvement iteration."""
        self.iteration_history.append({
            'iteration_id': iteration_id, 'task_type': task_type,
            'score': score, 'execution_time': execution_time,
            'success': success, 'timestamp': time.time()
        })
        if len(self.iteration_history) > 200:
            self.iteration_history = self.iteration_history[-200:]

    def get_improvement_trend(self, task_type: str = None, window: int = 10) -> Dict[str, Any]:
        """Split-half comparison with stddev-based noise-aware threshold."""
        history = getattr(self, 'iteration_history', [])
        if task_type:
            history = [h for h in history if h['task_type'] == task_type]
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'delta': 0.0, 'iterations': len(history)}
        recent = history[-window:]
        mid = len(recent) // 2
        avg_first = sum(h['score'] for h in recent[:mid]) / mid
        avg_second = sum(h['score'] for h in recent[mid:]) / len(recent[mid:])
        delta = avg_second - avg_first

        # Compute stddev of the full recent window for noise-aware threshold
        scores = [h['score'] for h in recent]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        stddev = math.sqrt(variance)
        threshold = max(0.5 * stddev, 0.01)

        trend = 'improving' if delta > threshold else ('declining' if delta < -threshold else 'stable')
        return {'trend': trend, 'delta': delta, 'avg_recent': avg_second,
                'avg_earlier': avg_first, 'iterations': len(recent),
                'stddev': stddev, 'threshold': threshold}

    def record_single_agent_run(self, task_type: str, execution_time: float, success: bool = True):
        """Record single-agent baseline run."""
        self.single_agent_runs[task_type].append((execution_time, success))

        # Keep bounded
        if len(self.single_agent_runs[task_type]) > 100:
            self.single_agent_runs[task_type] = self.single_agent_runs[task_type][-100:]

    def record_multi_agent_run(self, task_type: str, execution_time: float,
                               agents_count: int, success: bool = True):
        """Record multi-agent run."""
        self.multi_agent_runs[task_type].append((execution_time, agents_count, success))

        # Keep bounded
        if len(self.multi_agent_runs[task_type]) > 100:
            self.multi_agent_runs[task_type] = self.multi_agent_runs[task_type][-100:]

    def record_communication(self, from_agent: str, to_agent: str, message_size: int = 0):
        """Record inter-agent communication event."""
        self.communication_events.append({
            'from': from_agent,
            'to': to_agent,
            'size': message_size,
            'timestamp': time.time()
        })

        # Keep bounded
        if len(self.communication_events) > 1000:
            self.communication_events = self.communication_events[-1000:]

    def record_cooperation(self, helper: str, helped: str, task_type: str, success: bool):
        """Record cooperation event between agents."""
        self.cooperation_events.append({
            'helper': helper,
            'helped': helped,
            'task_type': task_type,
            'success': success,
            'timestamp': time.time()
        })

        # Keep bounded
        if len(self.cooperation_events) > 500:
            self.cooperation_events = self.cooperation_events[-500:]

    def compute_metrics(self, agent_profiles: Dict[str, 'AgentProfile'] = None) -> SwarmMetrics:
        """Compute current swarm metrics."""
        metrics = SwarmMetrics()

        # 1. Single vs Multi speedup ratio
        speedups = []
        for task_type in self.multi_agent_runs:
            single_times = [t for t, s in self.single_agent_runs.get(task_type, []) if s]
            multi_times = [t for t, n, s in self.multi_agent_runs.get(task_type, []) if s]

            if single_times and multi_times:
                avg_single = sum(single_times) / len(single_times)
                avg_multi = sum(multi_times) / len(multi_times)
                if avg_multi > 0:
                    speedups.append(avg_single / avg_multi)

        if speedups:
            metrics.single_vs_multi_ratio = sum(speedups) / len(speedups)

        # 2. Communication overhead (messages per hour)
        recent_comms = [c for c in self.communication_events
                       if time.time() - c['timestamp'] < 3600]
        metrics.communication_overhead = len(recent_comms)

        # 3. Specialization diversity (entropy of specializations)
        if agent_profiles:
            spec_counts = defaultdict(int)
            for profile in agent_profiles.values():
                spec_counts[profile.specialization.value] += 1

            total = sum(spec_counts.values())
            if total > 0:
                entropy = 0.0
                for count in spec_counts.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * math.log2(p)
                # Normalize by max entropy (log2(num_specializations))
                max_entropy = math.log2(len(AgentSpecialization))
                metrics.specialization_diversity = entropy / max_entropy if max_entropy > 0 else 0

        # 4. Cooperation index
        recent_coop = [c for c in self.cooperation_events
                      if time.time() - c['timestamp'] < 3600]
        if recent_coop:
            successful = sum(1 for c in recent_coop if c['success'])
            metrics.cooperation_index = successful / len(recent_coop)

        return metrics

    def format_benchmark_report(self, agent_profiles: Dict[str, 'AgentProfile'] = None) -> str:
        """Generate human-readable benchmark report."""
        metrics = self.compute_metrics(agent_profiles)

        lines = [
            "# Swarm Benchmark Report",
            "=" * 40,
            "",
            f"## Performance Metrics",
            f"  - Multi-agent speedup ratio: {metrics.single_vs_multi_ratio:.2f}x",
            f"  - Communication overhead: {metrics.communication_overhead:.0f} msgs/hour",
            f"  - Cooperation index: {metrics.cooperation_index:.2%}",
            f"  - Specialization diversity: {metrics.specialization_diversity:.2%}",
            "",
        ]

        # Task-specific breakdown
        if self.multi_agent_runs:
            lines.append("## Task Performance")
            for task_type in sorted(self.multi_agent_runs.keys()):
                multi = self.multi_agent_runs[task_type]
                single = self.single_agent_runs.get(task_type, [])

                multi_success = sum(1 for _, _, s in multi if s) / len(multi) if multi else 0
                single_success = sum(1 for _, s in single if s) / len(single) if single else 0

                lines.append(f"  {task_type}:")
                lines.append(f"    - Multi-agent success: {multi_success:.1%} ({len(multi)} runs)")
                lines.append(f"    - Single-agent success: {single_success:.1%} ({len(single)} runs)")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            'single_agent_runs': dict(self.single_agent_runs),
            'multi_agent_runs': dict(self.multi_agent_runs),
            'communication_events': self.communication_events[-200:],
            'cooperation_events': self.cooperation_events[-100:],
            'iteration_history': getattr(self, 'iteration_history', [])[-200:],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SwarmBenchmarks':
        """Deserialize from persistence."""
        instance = cls()
        instance.single_agent_runs = defaultdict(list, data.get('single_agent_runs', {}))
        instance.multi_agent_runs = defaultdict(list, data.get('multi_agent_runs', {}))
        instance.communication_events = data.get('communication_events', [])
        instance.cooperation_events = data.get('cooperation_events', [])
        instance.iteration_history = data.get('iteration_history', [])
        return instance


# =============================================================================
# MAS-BENCH EVALUATION HARNESS
# =============================================================================
# Measures MAS-Bench metrics: SR, SSR, MS, MET, MToC, GSAR
# for hybrid mobile GUI + API shortcut agents.


@dataclass
class MASBenchResult:
    """Result of a single MAS-Bench task execution.

    Captures all metrics needed for MAS-Bench evaluation:
    success, steps, time, tokens, and action type breakdown.
    """
    task_id: str
    task_description: str = ""
    success: bool = False

    # Step counts
    total_steps: int = 0
    optimal_steps: int = 1      # ground truth optimal steps
    gui_steps: int = 0          # steps using GUI actions
    shortcut_steps: int = 0     # steps using API/deeplink/RPA

    # Shortcut metrics
    shortcut_calls: int = 0     # total shortcut invocations
    shortcut_successes: int = 0 # successful shortcut executions

    # Cost/time
    execution_time_sec: float = 0.0
    token_cost_k: float = 0.0   # kilo-tokens consumed

    # Metadata
    app_package: str = ""
    difficulty_level: int = 1   # 1, 2, or 3
    is_cross_app: bool = False
    error: str = ""

    @property
    def step_ratio(self) -> float:
        """Mean Step Ratio (MSR): actual steps / optimal steps."""
        return self.total_steps / max(self.optimal_steps, 1)

    @property
    def shortcut_success_rate(self) -> float:
        """Shortcut Success Rate (SSR): successful / total shortcut calls."""
        return self.shortcut_successes / max(self.shortcut_calls, 1)

    @property
    def gui_shortcut_ratio(self) -> float:
        """GUI-to-Shortcut Action Ratio (GSAR)."""
        return self.shortcut_steps / max(self.gui_steps, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'total_steps': self.total_steps,
            'optimal_steps': self.optimal_steps,
            'gui_steps': self.gui_steps,
            'shortcut_steps': self.shortcut_steps,
            'shortcut_calls': self.shortcut_calls,
            'shortcut_successes': self.shortcut_successes,
            'step_ratio': self.step_ratio,
            'shortcut_success_rate': self.shortcut_success_rate,
            'gui_shortcut_ratio': self.gui_shortcut_ratio,
            'execution_time_sec': self.execution_time_sec,
            'token_cost_k': self.token_cost_k,
            'difficulty_level': self.difficulty_level,
            'is_cross_app': self.is_cross_app,
            'error': self.error,
        }


class MASBenchRunner:
    """Evaluation harness for MAS-Bench hybrid mobile agent benchmark.

    Runs tasks through Jotty's SkillPlanExecutor and measures the 7 MAS-Bench
    metrics: SR, SSR, MS, MSR, MET, MToC, GSAR.

    Usage::

        runner = MASBenchRunner()
        runner.add_task('task_1', 'Search for "laptop" on Amazon and add to cart',
                        app='com.amazon.mShop.android.shopping', optimal_steps=5)
        results = runner.run_all()
        metrics = runner.compute_aggregate_metrics(results)
    """

    # Skills classified as GUI (slow, fragile)
    GUI_SKILLS = frozenset({
        'browser-automation', 'android-automation', 'webapp-testing',
    })

    # Skills classified as API shortcuts (fast, reliable)
    API_SKILLS = frozenset({
        'http-client', 'web-search', 'web-scraper', 'oauth-automation',
        'telegram-sender', 'slack-integration', 'pmi-market-data',
        'pmi-portfolio', 'pmi-trading', 'pmi-watchlist', 'pmi-alerts',
        'pmi-strategies', 'pmi-broker', 'shell-exec', 'database-tools',
    })

    def __init__(self):
        self.tasks: List[Dict[str, Any]] = []
        self.results: List[MASBenchResult] = []

    def add_task(self, task_id: str, description: str,
                 app: str = "", optimal_steps: int = 1,
                 difficulty: int = 1, cross_app: bool = False):
        """Register a task for evaluation.

        Args:
            task_id: Unique task identifier.
            description: Natural language task description.
            app: Android package name (for app launch).
            optimal_steps: Ground truth optimal step count.
            difficulty: Difficulty level (1-3).
            cross_app: Whether task spans multiple apps.
        """
        self.tasks.append({
            'task_id': task_id,
            'description': description,
            'app': app,
            'optimal_steps': optimal_steps,
            'difficulty': difficulty,
            'cross_app': cross_app,
        })

    def classify_step(self, skill_name: str) -> str:
        """Classify a step as 'gui' or 'shortcut' based on skill name.

        Args:
            skill_name: The skill used for this step.

        Returns:
            'gui' or 'shortcut'
        """
        if skill_name in self.GUI_SKILLS:
            return 'gui'
        return 'shortcut'

    def evaluate_execution(self, task: Dict[str, Any],
                           steps_executed: List[Dict[str, Any]],
                           success: bool,
                           execution_time: float,
                           token_cost: float) -> MASBenchResult:
        """Evaluate a single task execution and compute metrics.

        Args:
            task: Task definition dict from add_task().
            steps_executed: List of executed step dicts with 'skill_name', 'status'.
            success: Whether the task completed successfully.
            execution_time: Total execution time in seconds.
            token_cost: Total tokens consumed (in thousands).

        Returns:
            MASBenchResult with all metrics computed.
        """
        result = MASBenchResult(
            task_id=task['task_id'],
            task_description=task['description'],
            success=success,
            total_steps=len(steps_executed),
            optimal_steps=task.get('optimal_steps', 1),
            execution_time_sec=execution_time,
            token_cost_k=token_cost,
            app_package=task.get('app', ''),
            difficulty_level=task.get('difficulty', 1),
            is_cross_app=task.get('cross_app', False),
        )

        for step in steps_executed:
            skill = step.get('skill_name', '')
            step_type = self.classify_step(skill)
            step_status = step.get('status', 'completed')

            if step_type == 'gui':
                result.gui_steps += 1
            else:
                result.shortcut_steps += 1
                result.shortcut_calls += 1
                if step_status == 'completed':
                    result.shortcut_successes += 1

        return result

    @staticmethod
    def compute_aggregate_metrics(results: List[MASBenchResult]) -> Dict[str, Any]:
        """Compute aggregate MAS-Bench metrics across all tasks.

        Returns the 7 MAS-Bench metrics:
        - SR: Success Rate
        - SSR: Shortcut Success Rate
        - MS: Mean Steps
        - MSR: Mean Step Ratio
        - MET: Mean Execution Time
        - MToC: Mean Token Cost (kilo-tokens)
        - GSAR: GUI-to-Shortcut Action Ratio

        Args:
            results: List of MASBenchResult from individual tasks.

        Returns:
            Dict with aggregate metrics and breakdowns.
        """
        if not results:
            return {'SR': 0, 'SSR': 0, 'MS': 0, 'MSR': 0, 'MET': 0, 'MToC': 0, 'GSAR': 0}

        n = len(results)
        successes = sum(1 for r in results if r.success)
        total_shortcut_calls = sum(r.shortcut_calls for r in results)
        total_shortcut_successes = sum(r.shortcut_successes for r in results)

        metrics = {
            'SR': successes / n,
            'SSR': total_shortcut_successes / max(total_shortcut_calls, 1),
            'MS': sum(r.total_steps for r in results) / n,
            'MSR': sum(r.step_ratio for r in results) / n,
            'MET': sum(r.execution_time_sec for r in results) / n,
            'MToC': sum(r.token_cost_k for r in results) / n,
            'GSAR': sum(r.gui_shortcut_ratio for r in results) / n,
            'total_tasks': n,
            'total_successes': successes,
        }

        # Breakdown by difficulty
        for level in [1, 2, 3]:
            level_results = [r for r in results if r.difficulty_level == level]
            if level_results:
                level_successes = sum(1 for r in level_results if r.success)
                metrics[f'SR_L{level}'] = level_successes / len(level_results)

        # Breakdown by task type (single-app vs cross-app)
        single = [r for r in results if not r.is_cross_app]
        cross = [r for r in results if r.is_cross_app]
        if single:
            metrics['SR_single_app'] = sum(1 for r in single if r.success) / len(single)
        if cross:
            metrics['SR_cross_app'] = sum(1 for r in cross if r.success) / len(cross)

        return metrics

    def summary(self, results: List[MASBenchResult] = None) -> str:
        """Generate human-readable summary of benchmark results.

        Args:
            results: Results to summarize (default: self.results).

        Returns:
            Formatted summary string.
        """
        results = results or self.results
        metrics = self.compute_aggregate_metrics(results)

        lines = [
            "MAS-Bench Results",
            "=" * 50,
            f"Tasks: {metrics.get('total_tasks', 0)} "
            f"({metrics.get('total_successes', 0)} passed)",
            f"Success Rate (SR): {metrics['SR']:.1%}",
            f"Shortcut Success Rate (SSR): {metrics['SSR']:.1%}",
            f"Mean Steps (MS): {metrics['MS']:.1f}",
            f"Mean Step Ratio (MSR): {metrics['MSR']:.2f}",
            f"Mean Execution Time (MET): {metrics['MET']:.1f}s",
            f"Mean Token Cost (MToC): {metrics['MToC']:.1f}k",
            f"GUI-to-Shortcut Ratio (GSAR): {metrics['GSAR']:.2f}",
        ]

        if 'SR_single_app' in metrics:
            lines.append(f"Single-app SR: {metrics['SR_single_app']:.1%}")
        if 'SR_cross_app' in metrics:
            lines.append(f"Cross-app SR: {metrics['SR_cross_app']:.1%}")

        for level in [1, 2, 3]:
            key = f'SR_L{level}'
            if key in metrics:
                lines.append(f"Level {level} SR: {metrics[key]:.1%}")

        return "\n".join(lines)


__all__ = [
    'SwarmMetrics',
    'SwarmBenchmarks',
    'MASBenchResult',
    'MASBenchRunner',
]
