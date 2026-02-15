"""
 CONTEXT GRADIENT - Learning Mechanism for LLM-based RL
==========================================================

Since we can't update model weights (models are frozen), context becomes our
learning mechanism - our "gradient"!

Traditional ML:
    Loss = f(prediction, truth)
    Gradient = ∂Loss/∂weights
    weights -= learning_rate * gradient

LLM-based RL (our approach):
    Experience = (state, action, reward, next_state, reasoning)
    Context_update = g(experience, past_context)
    context += context_update  # Our "gradient"!

Author: A-Team (Von Neumann, Nash, Anthropic, Cursor engineers)
Date: Dec 29, 2025
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT GRADIENT COMPONENTS
# =============================================================================


class MemoryGradientSignature(dspy.Signature):
    """
    Extract key lessons from experience for memory update.

    What should we remember from this experience?
    Focus on actionable insights, not just facts.
    """

    experience_summary = dspy.InputField(desc="Summary of what happened (state, action, outcome)")
    reward = dspy.InputField(desc="Reward received (0.0-1.0)")
    agent_name = dspy.InputField(desc="Which agent had this experience")
    past_similar_experiences = dspy.InputField(
        desc="Similar experiences from memory (for comparison)"
    )

    key_lesson = dspy.OutputField(desc="The key lesson to remember (actionable insight)")
    lesson_confidence = dspy.OutputField(desc="Confidence in this lesson (0.0-1.0)")
    when_to_apply = dspy.OutputField(desc="When should this lesson be applied? (conditions)")
    related_lessons = dspy.OutputField(desc="How does this relate to past lessons? (connections)")


class CooperationGradientSignature(dspy.Signature):
    """
    Extract cooperation insights from experience.

    How should agents cooperate better based on this experience?
    """

    cooperation_event = dspy.InputField(desc="Description of cooperation (or lack thereof)")
    outcome = dspy.InputField(desc="What happened as a result")
    agents_involved = dspy.InputField(desc="Which agents were involved")
    system_reward = dspy.InputField(desc="System reward achieved")

    cooperation_insight = dspy.OutputField(
        desc="Insight about cooperation (what worked/didn't work)"
    )
    recommendation = dspy.OutputField(desc="How should agents cooperate differently next time?")
    confidence = dspy.OutputField(desc="Confidence in this insight (0.0-1.0)")


# =============================================================================
# CONTEXT GRADIENT COMPUTER
# =============================================================================


@dataclass
class ContextUpdate:
    """A single context update (our "gradient")."""

    component: str  # 'memory', 'q_table', 'dqn', 'cooperation'
    update_type: str  # 'add', 'modify', 'strengthen', 'weaken'
    content: str  # The actual update (natural language)
    confidence: float  # How confident we are
    priority: float  # How important is this update
    metadata: Dict = field(default_factory=dict)


class ContextGradient:
    """
    Compute context updates (our "gradient") from experiences.

    Components:
    1. Memory Gradient: What to remember
    2. Q-table Gradient: Update Q-values (TD learning)
    3. DQN Gradient: Update predictions about other agents
    4. Cooperation Gradient: Update cooperation strategies
    """

    def __init__(self) -> None:
        self.memory_extractor = dspy.ChainOfThought(MemoryGradientSignature)
        self.cooperation_extractor = dspy.ChainOfThought(CooperationGradientSignature)

        logger.info(" ContextGradient initialized")

    def compute_gradient(self, experience: Dict, past_context: Dict) -> List[ContextUpdate]:
        """
        Compute context updates from experience.

        Args:
            experience: {
                'state': current_state,
                'action': action_taken,
                'reward': reward_received,
                'next_state': resulting_state,
                'reasoning': why_this_action,
                'agent': agent_name,
                'predictions': {other_agent: predicted_actions},
                'actual_others': {other_agent: actual_actions},
                'cooperation_events': [events]
            }
            past_context: {
                'memory': past_memories,
                'q_table': current_q_values,
                'dqn_predictions': past_predictions,
                'cooperation_insights': past_insights
            }

        Returns:
            List of ContextUpdate objects
        """
        logger.info(f" Computing context gradient for {experience.get('agent', 'unknown')}...")

        updates = []

        # 1. Memory Gradient
        memory_update = self._compute_memory_gradient(experience, past_context)
        if memory_update:
            updates.append(memory_update)

        # 2. Q-table Gradient (TD learning)
        q_update = self._compute_q_gradient(experience, past_context)
        if q_update:
            updates.append(q_update)

        # 3. DQN Gradient (prediction correction)
        dqn_update = self._compute_dqn_gradient(experience, past_context)
        if dqn_update:
            updates.append(dqn_update)

        # 4. Cooperation Gradient
        coop_updates = self._compute_cooperation_gradient(experience, past_context)
        updates.extend(coop_updates)

        logger.info(f" Computed {len(updates)} context updates")
        for update in updates:
            logger.info(
                f"   - {update.component}: {update.update_type} (confidence={update.confidence:.2f})"
            )

        return updates

    def _compute_memory_gradient(
        self, experience: Dict, past_context: Dict
    ) -> Optional[ContextUpdate]:
        """Extract key lesson for memory."""
        try:
            # Format experience
            exp_summary = self._format_experience(experience)

            # Get similar past experiences
            past_memories = past_context.get("memory", [])
            similar = self._find_similar_experiences(experience, past_memories)
            similar_desc = self._format_similar(similar)

            # LLM extraction
            result = self.memory_extractor(
                experience_summary=exp_summary,
                reward=str(experience.get("reward", 0.0)),
                agent_name=experience.get("agent", "unknown"),
                past_similar_experiences=similar_desc,
            )

            # Create update
            update = ContextUpdate(
                component="memory",
                update_type="add",
                content=result.key_lesson,
                confidence=float(result.lesson_confidence) if result.lesson_confidence else 0.5,
                priority=experience.get("reward", 0.5),
                metadata={
                    "when_to_apply": result.when_to_apply,
                    "related_lessons": result.related_lessons,
                    "agent": experience.get("agent"),
                },
            )

            return update

        except Exception as e:
            logger.error(f" Memory gradient failed: {e}")
            return None

    def _compute_q_gradient(self, experience: Dict, past_context: Dict) -> Optional[ContextUpdate]:
        """Compute Q-table update using TD(λ)."""
        try:
            state = experience.get("state", {})
            action = experience.get("action", "")
            reward = experience.get("reward", 0.0)
            next_state = experience.get("next_state", {})

            # Get current Q-value
            q_table = past_context.get("q_table", {})
            state_key = self._state_to_key(state)
            current_q = q_table.get((state_key, action), 0.5)

            # Estimate next Q-value (max over next actions)
            next_q = 0.5  # Default
            if next_state:
                next_key = self._state_to_key(next_state)
                next_actions = [k[1] for k in q_table.keys() if k[0] == next_key]
                if next_actions:
                    next_q = max(q_table.get((next_key, a), 0.5) for a in next_actions)

            # TD error
            gamma = 0.95  # Discount factor
            td_error = reward + gamma * next_q - current_q

            # New Q-value
            alpha = 0.1  # Learning rate
            new_q = current_q + alpha * td_error

            # Create update
            update = ContextUpdate(
                component="q_table",
                update_type="modify",
                content=f"Q({state_key}, {action}) = {new_q:.3f} (was {current_q:.3f}, TD error={td_error:.3f})",
                confidence=0.9,  # High confidence in math
                priority=abs(td_error),  # Larger errors = higher priority
                metadata={
                    "state_key": state_key,
                    "action": action,
                    "old_q": current_q,
                    "new_q": new_q,
                    "td_error": td_error,
                },
            )

            return update

        except Exception as e:
            logger.error(f" Q-gradient failed: {e}")
            return None

    def _compute_dqn_gradient(
        self, experience: Dict, past_context: Dict
    ) -> Optional[ContextUpdate]:
        """
        Update predictions about other agents.

         A-TEAM ENHANCEMENT: Use divergence as TD error for context updates.

        Traditional RL: θ ← θ - α * TD_error
        Agentic RL: context ← context + f(divergence)
        """
        try:
            predicted = experience.get("predictions", {})
            actual = experience.get("actual_others", {})

            if not predicted or not actual:
                return None

            # Compute divergence (acts as TD error)
            divergences = {}
            td_errors = []
            for agent, pred_actions in predicted.items():
                actual_action = actual.get(agent)
                if actual_action:
                    # Check if prediction was correct
                    was_correct = actual_action in pred_actions
                    div = 0.0 if was_correct else 1.0
                    divergences[agent] = div

                    # COMPUTE TD ERROR: δ = |actual_reward - predicted_reward|
                    pred_reward = experience.get("predicted_reward", 0.5)
                    actual_reward = experience.get("reward", 0.5)
                    td_error = abs(actual_reward - pred_reward)
                    td_errors.append(td_error)

            if not divergences:
                return None

            avg_divergence = sum(divergences.values()) / len(divergences)
            avg_td_error = sum(td_errors) / len(td_errors) if td_errors else 0.0

            # LEARNING RATE: Scale updates by TD error magnitude
            # Higher TD error = more aggressive update
            learning_rate = min(1.0, 0.3 + 0.7 * avg_td_error)

            # Create update with TD error context
            corrections = []
            for agent, div in divergences.items():
                if div > 0.5:  # Significant error
                    corrections.append(
                        f"{agent}: predicted {predicted[agent]}, actually did {actual[agent]}"
                    )

            # ACTIONABLE LESSON based on divergence pattern
            if avg_divergence > 0.7:
                lesson_type = "MAJOR_CORRECTION"
                lesson = (
                    f" Model of agents is significantly wrong. Re-learn: {'; '.join(corrections)}"
                )
            elif avg_divergence > 0.3:
                lesson_type = "MINOR_ADJUSTMENT"
                lesson = f" Adjust predictions: {'; '.join(corrections)}"
            else:
                lesson_type = "CONFIRMATION"
                lesson = " Predictions were mostly accurate - patterns confirmed"

            update = ContextUpdate(
                component="dqn",
                update_type="modify",
                content=lesson,
                confidence=1.0 - avg_divergence,  # Lower divergence = higher confidence
                priority=avg_divergence * learning_rate,  # Scale by learning rate
                metadata={
                    "divergences": divergences,
                    "avg_divergence": avg_divergence,
                    "td_error": avg_td_error,
                    "learning_rate": learning_rate,
                    "lesson_type": lesson_type,
                },
            )

            logger.debug(
                f" DQN gradient: divergence={avg_divergence:.2f}, TD_error={avg_td_error:.2f}, lr={learning_rate:.2f}"
            )

            return update

        except Exception as e:
            logger.error(f" DQN gradient failed: {e}")
            return None

    def _compute_cooperation_gradient(
        self, experience: Dict, past_context: Dict
    ) -> List[ContextUpdate]:
        """Extract cooperation insights."""
        updates = []

        try:
            coop_events = experience.get("cooperation_events", [])

            for event in coop_events:
                # LLM extraction
                result = self.cooperation_extractor(
                    cooperation_event=str(event),
                    outcome=str(experience.get("reward", 0.0)),
                    agents_involved=str(event.get("agents", [])),
                    system_reward=str(experience.get("reward", 0.0)),
                )

                # Create update
                update = ContextUpdate(
                    component="cooperation",
                    update_type="add",
                    content=result.cooperation_insight,
                    confidence=float(result.confidence) if result.confidence else 0.5,
                    priority=experience.get("reward", 0.5),
                    metadata={"recommendation": result.recommendation, "event": event},
                )

                updates.append(update)

        except Exception as e:
            logger.error(f" Cooperation gradient failed: {e}")

        return updates

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _format_experience(self, experience: Dict) -> str:
        """Format experience for LLM."""
        lines = []
        lines.append(f"Agent: {experience.get('agent', 'unknown')}")
        lines.append(f"State: {self._format_dict(experience.get('state', {}))}")
        lines.append(f"Action: {experience.get('action', 'unknown')}")
        lines.append(f"Outcome: {self._format_dict(experience.get('next_state', {}))}")
        lines.append(f"Reward: {experience.get('reward', 0.0):.2f}")
        lines.append(f"Reasoning: {experience.get('reasoning', 'N/A')}")
        return "\n".join(lines)

    def _format_dict(self, d: Dict) -> str:
        """Format dict for LLM."""
        if not d:
            return "N/A"
        items = [f"{k}={v}" for k, v in list(d.items())[:5]]  # Top 5
        return ", ".join(items)

    def _find_similar_experiences(self, experience: Dict, past_memories: List[Dict]) -> List[Dict]:
        """Find similar past experiences (simple heuristic).

        NOTE: Currently using recency-based heuristic (last 3 memories).
        Semantic similarity could be added in the future with embeddings,
        but the current approach provides good performance without
        additional LLM calls or embedding computation.
        """
        return past_memories[-3:] if past_memories else []

    def _format_similar(self, similar: List[Dict]) -> str:
        """Format similar experiences for LLM."""
        if not similar:
            return "No similar past experiences"

        lines = []
        for exp in similar:
            lines.append(f"- {exp.get('summary', 'Unknown')}")
        return "\n".join(lines)

    def _state_to_key(self, state: Dict) -> str:
        """Convert state dict to string key."""
        # Simple heuristic: use sorted keys and values
        items = sorted(state.items())
        return str(items)


# =============================================================================
# CONTEXT APPLIER
# =============================================================================


class ContextApplier:
    """
    Apply context updates to the actual context.

    This is where the "gradient" gets applied!
    """

    def __init__(self) -> None:
        logger.info(" ContextApplier initialized")

    def apply_updates(self, updates: List[ContextUpdate], current_context: Dict) -> Dict:
        """
        Apply context updates to current context.

        Args:
            updates: List of ContextUpdate objects
            current_context: Current context dict

        Returns:
            Updated context dict
        """
        logger.info(f" Applying {len(updates)} context updates...")

        updated_context = current_context.copy()

        for update in updates:
            if update.component == "memory":
                self._apply_memory_update(update, updated_context)
            elif update.component == "q_table":
                self._apply_q_update(update, updated_context)
            elif update.component == "dqn":
                self._apply_dqn_update(update, updated_context)
            elif update.component == "cooperation":
                self._apply_cooperation_update(update, updated_context)

        logger.info(" Context updates applied")
        return updated_context

    def _apply_memory_update(self, update: ContextUpdate, context: Dict) -> Any:
        """Apply memory update."""
        if "memory" not in context:
            context["memory"] = []

        context["memory"].append(
            {
                "lesson": update.content,
                "confidence": update.confidence,
                "when_to_apply": update.metadata.get("when_to_apply"),
                "agent": update.metadata.get("agent"),
                "timestamp": __import__("time").time(),
            }
        )

        logger.info(f" Memory: Added lesson (confidence={update.confidence:.2f})")

    def _apply_q_update(self, update: ContextUpdate, context: Dict) -> Any:
        """Apply Q-table update."""
        if "q_table" not in context:
            context["q_table"] = {}

        state_key = update.metadata.get("state_key")
        action = update.metadata.get("action")
        new_q = update.metadata.get("new_q")

        if state_key and action:
            context["q_table"][(state_key, action)] = new_q
            logger.info(f" Q-table: Updated Q({state_key}, {action}) = {new_q:.3f}")

    def _apply_dqn_update(self, update: ContextUpdate, context: Dict) -> Any:
        """Apply DQN update."""
        if "dqn_corrections" not in context:
            context["dqn_corrections"] = []

        context["dqn_corrections"].append(
            {
                "corrections": update.content,
                "divergences": update.metadata.get("divergences"),
                "timestamp": __import__("time").time(),
            }
        )

        logger.info(f" DQN: Added prediction corrections")

    def _apply_cooperation_update(self, update: ContextUpdate, context: Dict) -> Any:
        """Apply cooperation update."""
        if "cooperation_insights" not in context:
            context["cooperation_insights"] = []

        context["cooperation_insights"].append(
            {
                "insight": update.content,
                "recommendation": update.metadata.get("recommendation"),
                "confidence": update.confidence,
                "timestamp": __import__("time").time(),
            }
        )

        logger.info(f" Cooperation: Added insight (confidence={update.confidence:.2f})")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ContextGradient",
    "ContextApplier",
    "ContextUpdate",
    "MemoryGradientSignature",
    "CooperationGradientSignature",
]
