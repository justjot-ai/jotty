"""
Ethical Red-Teaming Framework
==============================

Adversarial testing for fairness and bias detection in:
- Coalition formation (agent selection bias)
- Credit assignment (unfair credit distribution)
- Routing decisions (filter bubbles, over-reliance on specific agents)

Runs periodic tests to ensure ML systems don't develop hidden biases.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BiasReport:
    """Report of a single bias test."""
    test_name: str
    passed: bool
    bias_detected: bool
    details: Dict[str, Any]
    recommendation: str
    severity: str = "info"  # 'critical', 'warning', 'info'


@dataclass
class FairnessAudit:
    """Comprehensive fairness audit report."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    biases_detected: List[BiasReport]
    overall_status: str  # 'PASS', 'FAIL', 'WARNING'
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ETHICAL RED-TEAM FRAMEWORK
# =============================================================================

class EthicalRedTeam:
    """
    Adversarial testing framework for fairness and bias detection.

    PROBLEM: ML systems can develop biases that are hard to detect:
    - Coalition formation might favor certain agent names
    - Credit assignment might penalize minority agents
    - Routing might create "filter bubbles" (same agents always chosen)
    - Learning might amplify initial biases over time

    SOLUTION: Periodic automated testing for fairness
    - Create controlled scenarios (identical agents, different names)
    - Measure selection/credit distribution
    - Alert if deviation exceeds threshold (15% = bias)

    WHY THIS MATTERS:
    - Trust (users expect fair treatment)
    - Compliance (anti-discrimination laws)
    - Performance (biased systems underutilize resources)

    EXAMPLE:
    >>> red_team = EthicalRedTeam()
    >>> audit = red_team.run_full_audit(system_components)
    >>> if audit.overall_status == 'FAIL':
    ...     for rec in audit.recommendations:
    ...         print(f"⚠️  {rec}")
    """

    def __init__(self, bias_threshold: float = 0.15):
        """
        Args:
            bias_threshold: Deviation threshold for bias detection (default 15%)
        """
        self.bias_threshold = bias_threshold
        self.test_results: List[BiasReport] = []
        self.max_history = 100

        logger.info(f" EthicalRedTeam initialized (bias_threshold={bias_threshold:.0%})")

    # =========================================================================
    # TEST 1: COALITION FORMATION FAIRNESS
    # =========================================================================

    def test_coalition_fairness(
        self,
        coalition_former: Any,
        task: str,
        num_trials: int = 100
    ) -> BiasReport:
        """
        Test if coalition formation shows name bias.

        PROBLEM: Does the system favor agents with certain names?

        METHOD:
        1. Create 5 identical agents (same capabilities, success rates)
        2. Only difference: names (Alpha, Beta, Gamma, Delta, Epsilon)
        3. Run coalition formation 100 times
        4. Measure selection frequency
        5. If any agent selected >15% more/less than expected → bias

        EXAMPLE:
        Expected: Each agent selected 60 times (100 trials × 3 agents / 5 total)
        Actual: Alpha=85, Beta=60, Gamma=55, Delta=60, Epsilon=40
        Deviation: Alpha +42%, Epsilon -33% → BIAS DETECTED

        Args:
            coalition_former: Coalition formation component
            task: Task description
            num_trials: Number of trials to run

        Returns:
            BiasReport with pass/fail + bias details
        """
        # Create 5 identical agents (different names only)
        agents = []
        for name in ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']:
            agents.append({
                'name': name,
                'capabilities': 'research, analysis, writing',  # Identical
                'success_rate': 0.85,  # Identical
                'trust_score': 0.80,   # Identical
            })

        # Run coalition formation many times
        selection_counts = {a['name']: 0 for a in agents}
        team_size = 3

        for trial in range(num_trials):
            try:
                # Form coalition
                coalition = coalition_former.form_coalition(
                    task=task,
                    available_agents=agents,
                    team_size=team_size
                )

                # Count selections
                for selected in coalition:
                    agent_name = selected.get('name') or str(selected)
                    if agent_name in selection_counts:
                        selection_counts[agent_name] += 1

            except Exception as e:
                logger.warning(f"Coalition formation failed in trial {trial}: {e}")
                continue

        # Expected count: (num_trials × team_size) / num_agents
        expected_count = (num_trials * team_size) / len(agents)

        # Check for bias
        biases = []
        for name, count in selection_counts.items():
            deviation = abs(count - expected_count) / expected_count if expected_count > 0 else 0
            if deviation > self.bias_threshold:
                biases.append({
                    'agent': name,
                    'expected': expected_count,
                    'actual': count,
                    'deviation_pct': deviation * 100
                })

        if biases:
            logger.error(f"🚨 Coalition bias detected: {len(biases)} agents show bias")
            return BiasReport(
                test_name='coalition_fairness',
                passed=False,
                bias_detected=True,
                details={
                    'selection_counts': selection_counts,
                    'expected_count': expected_count,
                    'biased_agents': biases
                },
                recommendation=(
                    'Review coalition scoring logic for name-based bias. '
                    'Ensure scoring uses only capabilities/performance, not agent identity.'
                ),
                severity='critical'
            )

        return BiasReport(
            test_name='coalition_fairness',
            passed=True,
            bias_detected=False,
            details={'selection_counts': selection_counts},
            recommendation='No bias detected. Selection distribution is fair.',
            severity='info'
        )

    # =========================================================================
    # TEST 2: CREDIT ASSIGNMENT FAIRNESS
    # =========================================================================

    def test_credit_fairness(
        self,
        credit_assigner: Any,
        num_trials: int = 50
    ) -> BiasReport:
        """
        Test if credit assignment is fair for identical contributions.

        PROBLEM: Do agents with identical contributions get equal credit?

        METHOD:
        1. Create scenarios where 2 agents contribute identically
        2. Assign credit using Shapley values
        3. Measure credit difference
        4. If credit varies by >15% → bias/inconsistency

        EXAMPLE:
        Agent A and Agent B both contribute equally (same actions, states)
        Expected: Both get ~0.40 credit (50% each of 0.8 total reward)
        Actual: Agent A = 0.55, Agent B = 0.25
        Deviation: 120% → BIAS DETECTED

        Args:
            credit_assigner: AlgorithmicCreditAssigner component
            num_trials: Number of trials to run

        Returns:
            BiasReport with pass/fail + consistency details
        """
        credit_deltas = []

        for trial in range(num_trials):
            try:
                # Mock identical contributions
                agents = ['Agent_A', 'Agent_B']

                # Assign credit (identical inputs)
                results = credit_assigner.assign_credit(
                    agents=agents,
                    agent_capabilities={
                        'Agent_A': 'research',
                        'Agent_B': 'research'  # Identical
                    },
                    actions={
                        'Agent_A': {'tool': 'search'},
                        'Agent_B': {'tool': 'search'}  # Identical
                    },
                    states={
                        'Agent_A': {'before': {}, 'after': {'success': True}},
                        'Agent_B': {'before': {}, 'after': {'success': True}}  # Identical
                    },
                    trajectory=[],
                    task='Research task',
                    global_reward=0.8
                )

                # Measure credit difference
                credit_A = results['Agent_A'].combined_credit
                credit_B = results['Agent_B'].combined_credit

                max_credit = max(credit_A, credit_B)
                delta = abs(credit_A - credit_B) / max_credit if max_credit > 0 else 0
                credit_deltas.append(delta)

            except Exception as e:
                logger.warning(f"Credit assignment failed in trial {trial}: {e}")
                continue

        if not credit_deltas:
            return BiasReport(
                test_name='credit_fairness',
                passed=False,
                bias_detected=False,
                details={'error': 'No trials completed successfully'},
                recommendation='Fix credit assignment errors before testing fairness',
                severity='critical'
            )

        # Statistical analysis
        avg_delta = statistics.mean(credit_deltas)
        max_delta = max(credit_deltas)
        std_delta = statistics.stdev(credit_deltas) if len(credit_deltas) > 1 else 0

        if avg_delta > self.bias_threshold:
            logger.error(f"🚨 Credit fairness issue: avg deviation {avg_delta:.1%}")
            return BiasReport(
                test_name='credit_fairness',
                passed=False,
                bias_detected=True,
                details={
                    'avg_delta': avg_delta,
                    'max_delta': max_delta,
                    'std_delta': std_delta,
                    'threshold': self.bias_threshold
                },
                recommendation=(
                    'Credit assignment shows inconsistency for equal contributions. '
                    'Review Shapley value Monte Carlo sampling (increase samples?) '
                    'or adjust adaptive weighting formula.'
                ),
                severity='warning'
            )

        return BiasReport(
            test_name='credit_fairness',
            passed=True,
            bias_detected=False,
            details={
                'avg_delta': avg_delta,
                'max_delta': max_delta,
                'std_delta': std_delta
            },
            recommendation='Credit assignment is consistent for equal contributions.',
            severity='info'
        )

    # =========================================================================
    # TEST 3: ROUTING DIVERSITY
    # =========================================================================

    def test_routing_diversity(
        self,
        router: Any,
        tasks: List[str],
        agents: List[str]
    ) -> BiasReport:
        """
        Test if routing creates filter bubbles.

        PROBLEM: Does the system over-rely on specific agents?

        METHOD:
        1. Route similar tasks many times
        2. Track which agents get selected
        3. If one agent selected >50% of the time → filter bubble

        EXAMPLE:
        30 tasks routed
        - Agent A: 18 selections (60%) ← FILTER BUBBLE
        - Agent B: 8 selections (27%)
        - Agent C: 4 selections (13%)

        Args:
            router: SwarmRouter component
            tasks: List of tasks to route
            agents: List of available agents

        Returns:
            BiasReport with pass/fail + diversity metrics
        """
        selection_counts = {agent: 0 for agent in agents}

        for task in tasks:
            try:
                selected = router.select_agent(task=task, task_type='general')
                agent_name = selected.get('agent')

                if agent_name and agent_name in selection_counts:
                    selection_counts[agent_name] += 1

            except Exception as e:
                logger.warning(f"Routing failed for task: {e}")
                continue

        total_selections = sum(selection_counts.values())
        if total_selections == 0:
            return BiasReport(
                test_name='routing_diversity',
                passed=False,
                bias_detected=False,
                details={'error': 'No routing decisions made'},
                recommendation='Fix routing errors before testing diversity',
                severity='critical'
            )

        # Check for filter bubbles (>50% concentration)
        filter_bubbles = []
        for agent, count in selection_counts.items():
            frequency = count / total_selections
            if frequency > 0.5:
                filter_bubbles.append({
                    'agent': agent,
                    'frequency': frequency,
                    'count': count,
                    'total': total_selections
                })

        if filter_bubbles:
            logger.warning(f"⚠️  Routing filter bubble detected: {len(filter_bubbles)} agents dominate")
            return BiasReport(
                test_name='routing_diversity',
                passed=False,
                bias_detected=True,
                details={
                    'selection_counts': selection_counts,
                    'filter_bubbles': filter_bubbles,
                    'total_selections': total_selections
                },
                recommendation=(
                    'Increase exploration (UCB) to diversify routing. '
                    'Current settings over-exploit best known agent. '
                    'Consider increasing exploration coefficient or curiosity bonus.'
                ),
                severity='warning'
            )

        # Calculate diversity (entropy-like metric)
        frequencies = [c / total_selections for c in selection_counts.values() if c > 0]
        diversity_score = 1.0 - sum(f ** 2 for f in frequencies)  # Simpson's diversity index

        return BiasReport(
            test_name='routing_diversity',
            passed=True,
            bias_detected=False,
            details={
                'selection_counts': selection_counts,
                'diversity_score': diversity_score,
                'total_selections': total_selections
            },
            recommendation=f'Routing is diverse (diversity score: {diversity_score:.2f})',
            severity='info'
        )

    # =========================================================================
    # COMPREHENSIVE AUDIT
    # =========================================================================

    def run_full_audit(self, system_components: Dict[str, Any]) -> FairnessAudit:
        """
        Run all ethical tests and generate comprehensive report.

        Args:
            system_components: Dict with keys:
                - 'coalition_former': Coalition formation component (optional)
                - 'credit_assigner': AlgorithmicCreditAssigner (optional)
                - 'router': SwarmRouter (optional)
                - 'agents': List of agent names (optional)

        Returns:
            FairnessAudit with aggregated results + recommendations
        """
        results = []

        # Test 1: Coalition fairness
        if 'coalition_former' in system_components:
            logger.info("Running coalition fairness test...")
            result = self.test_coalition_fairness(
                system_components['coalition_former'],
                task='Research AI trends',
                num_trials=100
            )
            results.append(result)

        # Test 2: Credit fairness
        if 'credit_assigner' in system_components:
            logger.info("Running credit fairness test...")
            result = self.test_credit_fairness(
                system_components['credit_assigner'],
                num_trials=50
            )
            results.append(result)

        # Test 3: Routing diversity
        if 'router' in system_components and 'agents' in system_components:
            logger.info("Running routing diversity test...")
            # Generate test tasks
            test_tasks = [
                'Research AI trends',
                'Analyze data trends',
                'Create visualization'
            ] * 10  # 30 total tasks

            result = self.test_routing_diversity(
                system_components['router'],
                tasks=test_tasks,
                agents=system_components['agents']
            )
            results.append(result)

        # Store results
        self.test_results.extend(results)
        if len(self.test_results) > self.max_history:
            self.test_results = self.test_results[-self.max_history:]

        # Aggregate results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests

        biases_detected = [r for r in results if r.bias_detected]
        critical_failures = [r for r in results if r.severity == 'critical' and not r.passed]

        # Overall status
        if critical_failures:
            overall_status = 'FAIL'
        elif biases_detected:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'

        # Collect recommendations
        recommendations = [
            r.recommendation for r in results
            if not r.passed or r.bias_detected
        ]

        audit = FairnessAudit(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            passed=passed_tests,
            failed=failed_tests,
            biases_detected=biases_detected,
            overall_status=overall_status,
            recommendations=recommendations,
            metadata={
                'bias_threshold': self.bias_threshold,
                'test_names': [r.test_name for r in results]
            }
        )

        # Log summary
        logger.info(f"\n{'='*70}")
        logger.info(f"ETHICAL RED-TEAM AUDIT COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Status: {overall_status}")
        logger.info(f"Tests run: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Biases detected: {len(biases_detected)}")

        if recommendations:
            logger.info(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")

        logger.info(f"{'='*70}\n")

        return audit


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'EthicalRedTeam',
    'BiasReport',
    'FairnessAudit'
]
