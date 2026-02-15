"""
RL Components for ReVal.

STATUS: UNUSED — RLComponents class has 0 external imports.
Only re-exported via __init__.py. Candidate for archival.
See MODULE_STATUS.md for details.

Provides reinforcement learning functionality including:
- Semantic experience retrieval
- TD learning with explanations
- Q-divergence calculation
- Pattern extraction
- Counterfactual credit assignment
- Theory of Mind modeling

GENERIC: No domain-specific logic.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RLComponents:
    """
    Unified RL functionality for ReVal swarm learning.

    Implements A-Team consensus design:
    - Semantic similarity for experience retrieval
    - TD_error + WHY for better learning
    - Q-divergence for reward shaping
    - Counterfactual credit assignment
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.compression_agent = None  # Lazy init

    def get_similar_experiences_semantic(
        self, experience_buffer: List[Dict], state: Dict, action: Dict, top_k: int = 10
    ) -> List[Dict]:
        """
        # TASK 1: Semantic experience retrieval with similarity scoring.

                A-TEAM CONSENSUS: Use semantic similarity, not just recency.

                Similarity based on:
                1. Same actor (exact match bonus)
                2. Similar state properties (Task List structure, pending tasks)
                3. Similar context (what came before)
                4. TD-error (prioritize surprising experiences)

                Args:
                    experience_buffer: List of past experiences
                    state: Current state dictionary
                    action: Action dictionary or string
                    top_k: Number of similar experiences to return

                Returns:
                    List of most similar experiences, sorted by similarity
        """
        actor = action.get("actor", "") if isinstance(action, dict) else str(action)

        if not experience_buffer:
            return []

        # Extract state features for comparison
        current_pending = state.get("todo", "").count("pending") if "todo" in state else 0
        current_iteration = state.get("iteration", 0)

        scored_experiences = []
        for exp in experience_buffer:
            similarity_score = 0.0

            # 1. Exact actor match (weight: 0.4)
            exp_action = exp.get("action", "")
            if isinstance(exp_action, dict):
                exp_actor = exp_action.get("actor", "")
            else:
                exp_actor = str(exp_action)

            if exp_actor == actor:
                similarity_score += 0.4

            # 2. Similar state structure (weight: 0.3)
            exp_state = exp.get("state", {})
            if isinstance(exp_state, dict):
                exp_pending = exp_state.get("todo", "").count("pending")
                # Closer pending count = more similar
                if current_pending > 0:
                    pending_sim = 1.0 - abs(exp_pending - current_pending) / max(
                        current_pending, exp_pending, 1
                    )
                    similarity_score += 0.3 * pending_sim

            # 3. TD-error (weight: 0.2) - prioritize surprising experiences
            td_error = exp.get("td_error", 0.0)
            if td_error > 0.3:  # High TD error = surprising = valuable
                similarity_score += 0.2 * min(td_error, 1.0)

            # 4. Recency bonus (weight: 0.1) - slight preference for recent
            age = (time.time() - exp.get("timestamp", 0)) / 3600  # hours
            recency_bonus = 0.1 * max(0, 1.0 - age / 24)  # Decay over 24 hours
            similarity_score += recency_bonus

            scored_experiences.append((similarity_score, exp))

        # Sort by similarity score (descending) and return top_k
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for score, exp in scored_experiences[:top_k]]

    async def add_td_error_and_why(
        self,
        experience: Dict[str, Any],
        td_error: float,
        state: Dict[str, Any],
        action: str,
        reward: float,
        v_current: float,
    ) -> Dict[str, Any]:
        """
        # TASK 2: Add TD_error and WHY to experience.

                A-TEAM CONSENSUS: Causal explanation critical for learning.

                Args:
                    experience: Experience dictionary to enhance
                    td_error: Temporal difference error
                    state: Current state
                    action: Action taken
                    reward: Actual reward
                    v_current: Predicted value

                Returns:
                    Enhanced experience with td_error and why fields
        """
        experience["td_error"] = abs(td_error)

        # Generate WHY reflection if error is significant
        if abs(td_error) > 0.2:
            why = await self._generate_why_reflection(
                state=state,
                action=action,
                reward=reward,
                td_error=td_error,
                context=f"Expected value ~{v_current:.2f}, but got reward {reward:.2f}",
            )
            experience["why"] = why
            logger.debug(f" WHY: {why}...")
        else:
            experience["why"] = ""

        return experience

    async def _generate_why_reflection(
        self, state: Dict[str, Any], action: str, reward: float, td_error: float, context: str
    ) -> str:
        """
        # TASK 6: LLM reflection for learning.

                Generate WHY reflection using LLM to understand prediction errors.

                Args:
                    state: Current state
                    action: Action taken
                    reward: Actual reward received
                    td_error: Temporal difference error
                    context: Additional context about prediction

                Returns:
                    Causal explanation of why prediction was wrong
        """
        try:
            # Use compression agent for intelligent reflection
            if not self.compression_agent:
                from ..integration.compression_agent import CompressionAgent

                self.compression_agent = CompressionAgent()

            reflection_prompt = f"""Analyze this RL experience and explain WHY the prediction was incorrect:

**State**: {str(state)}
**Action**: {action}
**Actual Reward**: {reward:.3f}
**TD Error**: {td_error:.3f}
**Context**: {context}

Provide concise causal explanation (2-3 sentences):
1. What made this different from expectations?
2. What pattern was missed?
3. What to learn for future?"""

            # Generate reflection (limit to 200 tokens)
            reflection = await self.compression_agent.compress(
                content=reflection_prompt, purpose="learning_reflection", token_budget=200
            )

            return reflection.strip()

        except Exception as e:
            logger.debug(f"WHY reflection generation failed: {e}")
            # Fallback template
            if td_error > 0:
                return f"Reward ({reward:.2f}) higher than predicted - action more valuable than expected."
            else:
                return f"Reward ({reward:.2f}) lower than predicted - context mismatch likely."

    def calculate_q_divergence_bonus(
        self, q_predicted: Optional[float], q_actual: float, predictability_bonus: float
    ) -> Tuple[float, float]:
        """
        # TASK 4: Q-divergence bonus calculation.

                A-TEAM CONSENSUS: Reward accurate predictions to improve learning.

                Args:
                    q_predicted: Predicted Q-value (None if no prediction)
                    q_actual: Actual Q-value (reward + discounted future)
                    predictability_bonus: Bonus multiplier from config

                Returns:
                    (q_divergence, bonus)
        """
        if q_predicted is None:
            return 0.0, 0.0

        # Calculate divergence
        q_divergence = abs(q_actual - q_predicted)

        # Bonus inversely proportional to divergence
        # Perfect prediction (divergence=0) → full bonus
        # Large error (divergence≥1) → no bonus
        bonus = (1.0 - min(q_divergence, 1.0)) * predictability_bonus

        logger.debug(
            f" Q-prediction: predicted={q_predicted:.3f}, actual={q_actual:.3f}, divergence={q_divergence:.3f}, bonus={bonus:.3f}"
        )

        return q_divergence, bonus

    def extract_patterns(
        self, experiences: List[Dict], min_frequency: int = 3
    ) -> List[Dict[str, Any]]:
        """
        # TASK 3: Pattern extraction from experiences.

                A-TEAM CONSENSUS: Identify recurring success/failure patterns.

                Args:
                    experiences: List of experiences to analyze
                    min_frequency: Minimum occurrences to be considered a pattern

                Returns:
                    List of identified patterns
        """
        if not experiences:
            return []

        # Group by (actor, outcome)
        pattern_counts: Dict[str, List[Dict]] = {}

        for exp in experiences:
            actor = exp.get("action", {}).get("actor", "unknown")
            reward = exp.get("reward", 0.0)
            outcome = "success" if reward > 0.7 else ("failure" if reward < 0.3 else "neutral")

            key = f"{actor}_{outcome}"
            if key not in pattern_counts:
                pattern_counts[key] = []
            pattern_counts[key].append(exp)

        # Extract frequent patterns
        patterns = []
        for key, exps in pattern_counts.items():
            if len(exps) >= min_frequency:
                actor, outcome = key.rsplit("_", 1)
                avg_reward = sum(e.get("reward", 0.0) for e in exps) / len(exps)

                patterns.append(
                    {
                        "actor": actor,
                        "outcome": outcome,
                        "frequency": len(exps),
                        "avg_reward": avg_reward,
                        "sample_states": [str(e.get("state", {})) for e in exps],
                    }
                )

        return sorted(patterns, key=lambda x: x["frequency"], reverse=True)

    def counterfactual_credit_assignment(
        self, trajectory: List[Dict], final_reward: float, gamma: float = 0.99
    ) -> Dict[str, float]:
        """
        # TASK 8: Counterfactual credit assignment.

                A-TEAM CONSENSUS: Use "what if we didn't take this action?" logic.

                Assigns credit to each actor in trajectory based on:
                1. Temporal proximity to outcome (recent = more credit)
                2. Counterfactual reasoning (how critical was this step?)
                3. Cooperative contribution (enabled others' success?)

                Args:
                    trajectory: List of (actor, action, intermediate_reward) tuples
                    final_reward: Final outcome reward
                    gamma: Discount factor

                Returns:
                    Dict mapping actor name to credit score
        """
        if not trajectory:
            return {}

        credits: Dict[str, float] = {}
        total_steps = len(trajectory)

        for i, step in enumerate(trajectory):
            actor = step.get("actor", "unknown")
            step_reward = step.get("reward", 0.0)

            # 1. Temporal proximity credit (closer to end = more impact)
            temporal_factor = gamma ** (total_steps - i - 1)

            # 2. Direct contribution (step's own reward)
            direct_credit = step_reward * temporal_factor

            # 3. Enabling credit (portion of final reward)
            # Each step gets credit for enabling the final outcome
            enabling_credit = (final_reward * temporal_factor) / total_steps

            # 4. Cooperative bonus (if step helped others)
            coop_bonus = 0.0
            if i < total_steps - 1:  # Not the last step
                # Check if next steps succeeded (this step enabled them)
                next_rewards = [s.get("reward", 0.0) for s in trajectory[i + 1 :]]
                if next_rewards and sum(next_rewards) > 0:
                    coop_bonus = 0.1 * temporal_factor

            # Total credit
            total_credit = direct_credit + enabling_credit + coop_bonus

            credits[actor] = credits.get(actor, 0.0) + total_credit

        return credits

    def theory_of_mind_predict(
        self, actor_name: str, other_actors: List[str], trajectory: List[Dict], current_state: Dict
    ) -> Dict[str, float]:
        """
        # TASK 9: Theory of Mind modeling.

                A-TEAM CONSENSUS: Predict what other actors will do to coordinate better.

                Predicts likelihood of each actor being chosen next based on:
                1. Recent activation patterns
                2. State compatibility (who can handle current state?)
                3. Success history

                Args:
                    actor_name: Current actor
                    other_actors: List of other available actors
                    trajectory: Recent trajectory
                    current_state: Current swarm state

                Returns:
                    Dict mapping actor name to predicted probability of being next
        """
        if not other_actors:
            return {}

        predictions: Dict[str, float] = {}

        # Analyze recent trajectory for patterns
        recent_actors = [step.get("actor", "") for step in trajectory]
        recent_success = [step.get("reward", 0.0) for step in trajectory]

        for actor in other_actors:
            score = 0.0

            # 1. Recency (was this actor used recently?)
            if actor in recent_actors:
                score += 0.3

            # 2. Success rate (did this actor succeed recently?)
            actor_successes = [r for a, r in zip(recent_actors, recent_success) if a == actor]
            if actor_successes:
                avg_success = sum(actor_successes) / len(actor_successes)
                score += 0.4 * avg_success

            # 3. Turn-taking pattern (actors often alternate)
            if recent_actors and recent_actors[-1] != actor:
                score += 0.2

            # 4. State compatibility (simple heuristic: pending tasks)
            pending = current_state.get("todo", "").count("pending")
            if pending > 0:
                score += 0.1  # All actors equally likely when tasks remain

            predictions[actor] = min(score, 1.0)

        # Normalize to probabilities
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v / total for k, v in predictions.items()}

        return predictions
