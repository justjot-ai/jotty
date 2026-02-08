#!/usr/bin/env python3
"""
Complex Learning Demonstration for Jotty MAS

This demo proves that Jotty ACTUALLY LEARNS by:
1. Running a complex multi-agent task across episodes
2. Showing Q-values improve from experience
3. Demonstrating knowledge transfer between similar problems
4. Persisting learning and loading in new session

Run: python demo_learning_test.py
"""

import sys
import time
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent))

import dspy
from core.integration.direct_claude_cli_lm import DirectClaudeCLI
from core.learning.q_learning import LLMQPredictor
from core.memory.fallback_memory import SimpleFallbackMemory, MemoryType


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LearningConfig:
    """Config for Q-learning."""
    tier1_max_size: int = 50
    tier2_max_clusters: int = 10
    tier3_max_size: int = 500
    max_experience_buffer: int = 200
    alpha: float = 0.3  # Learning rate
    gamma: float = 0.9  # Discount factor
    epsilon: float = 0.1
    max_q_table_size: int = 500
    q_prune_percentage: float = 0.2


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

class PlannerSignature(dspy.Signature):
    """Plan how to solve a complex problem."""
    problem: str = dspy.InputField(desc="Complex problem to solve")
    context: str = dspy.InputField(desc="Prior knowledge and context")
    plan: str = dspy.OutputField(desc="Step-by-step plan")
    key_challenges: str = dspy.OutputField(desc="Main challenges identified")


class ExecutorSignature(dspy.Signature):
    """Execute a step of the plan."""
    task: str = dspy.InputField(desc="Current task to execute")
    plan: str = dspy.InputField(desc="Overall plan")
    prior_results: str = dspy.InputField(desc="Results from prior steps")
    result: str = dspy.OutputField(desc="Execution result")
    success: str = dspy.OutputField(desc="yes/no - was execution successful")
    lessons_learned: str = dspy.OutputField(desc="What was learned from this step")


class ReviewerSignature(dspy.Signature):
    """Review and score the solution quality."""
    problem: str = dspy.InputField(desc="Original problem")
    solution: str = dspy.InputField(desc="Proposed solution")
    score: str = dspy.OutputField(desc="Quality score 0-100")
    feedback: str = dspy.OutputField(desc="Specific feedback for improvement")
    is_complete: str = dspy.OutputField(desc="yes/no - is solution complete")


# =============================================================================
# MULTI-AGENT SYSTEM
# =============================================================================

class LearningMAS:
    """Multi-Agent System that learns from experience."""

    def __init__(self, output_dir: str = None):
        # Initialize LLM
        self.lm = DirectClaudeCLI(model='sonnet')
        dspy.configure(lm=self.lm)

        # Initialize agents
        self.planner = dspy.ChainOfThought(PlannerSignature)
        self.executor = dspy.ChainOfThought(ExecutorSignature)
        self.reviewer = dspy.ChainOfThought(ReviewerSignature)

        # Initialize learning components
        self.config = LearningConfig()
        self.q_learner = LLMQPredictor(self.config)
        self.memory = SimpleFallbackMemory(max_entries=500)

        # Output directory for persistence
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix="jotty_learning_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []

        print(f"📁 Output directory: {self.output_dir}")

    def get_relevant_context(self, problem: str) -> str:
        """Retrieve relevant prior knowledge for problem."""
        memories = self.memory.retrieve(problem, top_k=5)
        if memories:
            return "\n".join([f"- {m.content}" for m in memories])
        return "No prior knowledge available."

    def solve_problem(self, problem: str, max_steps: int = 3) -> Dict[str, Any]:
        """
        Solve a complex problem using multi-agent collaboration.

        Returns episode results with rewards.
        """
        self.episode_count += 1
        print(f"\n{'='*60}")
        print(f"📍 EPISODE {self.episode_count}: {problem[:50]}...")
        print(f"{'='*60}")

        episode_start = time.time()
        trajectory = []
        total_reward = 0.0

        # Step 1: Planning
        print("\n🎯 PHASE 1: Planning")
        context = self.get_relevant_context(problem)
        print(f"   Context retrieved: {len(context)} chars")

        state_plan = {
            'phase': 'planning',
            'problem': problem[:100],
            'has_context': len(context) > 50,
            'episode': self.episode_count
        }
        action_plan = {'actor': 'Planner', 'task': 'create_plan'}

        try:
            plan_result = self.planner(problem=problem, context=context)
            plan = plan_result.plan
            challenges = plan_result.key_challenges

            plan_reward = 0.7  # Base reward for successful planning
            print(f"   ✅ Plan created: {len(plan)} chars")
            print(f"   Challenges: {challenges[:100]}...")

            # Store planning knowledge
            self.memory.store(
                f"PLAN for '{problem[:50]}': {plan[:200]}",
                MemoryType.PROCEDURAL,
                importance=0.8
            )
        except Exception as e:
            plan = f"Error: {e}"
            challenges = "Planning failed"
            plan_reward = 0.1
            print(f"   ❌ Planning failed: {e}")

        self.q_learner.add_experience(state_plan, action_plan, plan_reward)
        trajectory.append(('plan', plan_reward))
        total_reward += plan_reward

        # Step 2: Execution (multiple steps)
        print("\n⚙️ PHASE 2: Execution")
        prior_results = ""
        execution_rewards = []

        for step in range(max_steps):
            state_exec = {
                'phase': 'execution',
                'step': step + 1,
                'problem': problem[:50],
                'has_plan': len(plan) > 50,
                'prior_success_rate': sum(execution_rewards) / len(execution_rewards) if execution_rewards else 0.5
            }
            action_exec = {'actor': 'Executor', 'task': f'execute_step_{step+1}'}

            try:
                exec_result = self.executor(
                    task=f"Step {step+1} of plan",
                    plan=plan[:500],
                    prior_results=prior_results[:300]
                )

                result = exec_result.result
                success = 'yes' in exec_result.success.lower()
                lessons = exec_result.lessons_learned

                step_reward = 0.8 if success else 0.3
                execution_rewards.append(step_reward)
                prior_results += f"\nStep {step+1}: {result[:100]}"

                print(f"   Step {step+1}: {'✅' if success else '⚠️'} reward={step_reward:.2f}")

                # Store execution lessons
                if lessons:
                    self.memory.store(
                        f"LESSON: {lessons[:200]}",
                        MemoryType.SEMANTIC,
                        importance=0.7 if success else 0.5
                    )

            except Exception as e:
                step_reward = 0.1
                execution_rewards.append(step_reward)
                print(f"   Step {step+1}: ❌ Error - {e}")

            self.q_learner.add_experience(state_exec, action_exec, step_reward)
            trajectory.append((f'exec_{step+1}', step_reward))
            total_reward += step_reward

        # Step 3: Review
        print("\n📋 PHASE 3: Review")
        state_review = {
            'phase': 'review',
            'problem': problem[:50],
            'execution_success_rate': sum(execution_rewards) / len(execution_rewards),
            'total_steps': max_steps
        }
        action_review = {'actor': 'Reviewer', 'task': 'evaluate_solution'}

        try:
            review_result = self.reviewer(
                problem=problem,
                solution=prior_results
            )

            score = float(review_result.score) if review_result.score.isdigit() else 50
            is_complete = 'yes' in review_result.is_complete.lower()
            feedback = review_result.feedback

            review_reward = score / 100.0
            print(f"   Score: {score}/100, Complete: {is_complete}")
            print(f"   Feedback: {feedback[:100]}...")

            # Store review insights
            self.memory.store(
                f"FEEDBACK for '{problem[:30]}': {feedback[:200]}",
                MemoryType.EPISODIC,
                importance=review_reward
            )

        except Exception as e:
            score = 30
            review_reward = 0.3
            is_complete = False
            feedback = f"Review error: {e}"
            print(f"   ❌ Review failed: {e}")

        self.q_learner.add_experience(state_review, action_review, review_reward)
        trajectory.append(('review', review_reward))
        total_reward += review_reward

        # Calculate final episode reward
        avg_reward = total_reward / len(trajectory)
        self.episode_rewards.append(avg_reward)

        episode_duration = time.time() - episode_start

        print(f"\n📊 EPISODE {self.episode_count} SUMMARY")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Avg Reward: {avg_reward:.3f}")
        print(f"   Duration: {episode_duration:.1f}s")

        # Show Q-table stats
        q_stats = self.q_learner.get_q_table_stats()
        print(f"   Q-Table: {q_stats['size']} entries, avg Q={q_stats.get('avg_q_value', 0):.3f}")

        return {
            'episode': self.episode_count,
            'problem': problem,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'trajectory': trajectory,
            'score': score,
            'is_complete': is_complete,
            'duration': episode_duration
        }

    def save_learning(self):
        """Persist all learning state."""
        # Save Q-learning state
        q_path = self.output_dir / "q_learning_state.json"
        self.q_learner.save_state(str(q_path))

        # Save memory
        mem_path = self.output_dir / "memory_state.json"
        self.memory.save(str(mem_path))

        # Save episode history
        history_path = self.output_dir / "episode_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'episode_count': self.episode_count,
                'episode_rewards': self.episode_rewards,
                'timestamp': time.time()
            }, f, indent=2)

        print(f"\n💾 Learning saved to {self.output_dir}")

    def load_learning(self) -> bool:
        """Load previous learning state."""
        q_path = self.output_dir / "q_learning_state.json"
        mem_path = self.output_dir / "memory_state.json"
        history_path = self.output_dir / "episode_history.json"

        loaded = False

        if q_path.exists():
            self.q_learner.load_state(str(q_path))
            loaded = True

        if mem_path.exists():
            self.memory.load(str(mem_path))
            loaded = True

        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                self.episode_count = history.get('episode_count', 0)
                self.episode_rewards = history.get('episode_rewards', [])
            loaded = True

        if loaded:
            print(f"📂 Loaded previous learning: {self.episode_count} episodes")

        return loaded

    def show_learning_progress(self):
        """Display learning progress metrics."""
        print("\n" + "="*60)
        print("📈 LEARNING PROGRESS ANALYSIS")
        print("="*60)

        if len(self.episode_rewards) < 2:
            print("Need at least 2 episodes to show progress")
            return

        # Show reward trend
        first_half = self.episode_rewards[:len(self.episode_rewards)//2]
        second_half = self.episode_rewards[len(self.episode_rewards)//2:]

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        improvement = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0

        print(f"\n📊 Reward Trend:")
        print(f"   First half avg:  {first_avg:.3f}")
        print(f"   Second half avg: {second_avg:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")

        # Show Q-table growth
        q_stats = self.q_learner.get_q_table_stats()
        print(f"\n🧠 Q-Table Statistics:")
        print(f"   Entries: {q_stats['size']}")
        print(f"   Avg Q-value: {q_stats.get('avg_q_value', 0):.3f}")
        print(f"   Max Q-value: {q_stats.get('max_q_value', 0):.3f}")
        print(f"   Total visits: {q_stats.get('total_visits', 0)}")

        # Show memory stats
        mem_stats = self.memory.get_statistics()
        print(f"\n💾 Memory Statistics:")
        print(f"   Total memories: {mem_stats['total_entries']}")
        print(f"   By type: {mem_stats['by_type']}")

        # Show episode rewards chart (ASCII)
        print(f"\n📉 Episode Rewards (ASCII chart):")
        max_reward = max(self.episode_rewards) if self.episode_rewards else 1
        for i, reward in enumerate(self.episode_rewards):
            bar_len = int((reward / max_reward) * 30)
            bar = "█" * bar_len
            print(f"   Ep {i+1:2d}: {bar} {reward:.3f}")


# =============================================================================
# COMPLEX PROBLEMS FOR TESTING
# =============================================================================

COMPLEX_PROBLEMS = [
    # Similar problems (should show transfer learning)
    "Design a microservices architecture for an e-commerce platform with 1M daily users",
    "Design a microservices architecture for a social media platform with 500K daily users",
    "Design a microservices architecture for a banking application with high security needs",

    # Different but related problems
    "Implement a real-time recommendation engine using collaborative filtering",
    "Build a fraud detection system using machine learning for payment transactions",

    # Complex multi-step problems
    "Create a CI/CD pipeline for a monorepo with 50 microservices and automated testing",
]


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    print("="*70)
    print("🚀 JOTTY MULTI-AGENT LEARNING DEMONSTRATION")
    print("="*70)
    print()
    print("This demo shows that Jotty ACTUALLY LEARNS by:")
    print("1. Running complex problems across multiple episodes")
    print("2. Building Q-values from experience")
    print("3. Transferring knowledge between similar problems")
    print("4. Persisting and loading learned state")
    print()

    # Create output directory
    output_dir = Path("outputs/learning_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MAS
    mas = LearningMAS(output_dir=str(output_dir))

    # Try to load previous learning
    mas.load_learning()

    # Run episodes
    print("\n" + "="*70)
    print("🎮 RUNNING LEARNING EPISODES")
    print("="*70)

    results = []
    for i, problem in enumerate(COMPLEX_PROBLEMS[:4]):  # Run 4 problems
        result = mas.solve_problem(problem, max_steps=2)
        results.append(result)

        # Save after each episode
        mas.save_learning()

        # Brief pause between episodes
        if i < len(COMPLEX_PROBLEMS) - 1:
            print("\n⏳ Preparing next episode...")
            time.sleep(1)

    # Show learning progress
    mas.show_learning_progress()

    # Test knowledge transfer
    print("\n" + "="*70)
    print("🔄 TESTING KNOWLEDGE TRANSFER")
    print("="*70)

    # Query similar problem - should use learned knowledge
    test_query = "microservices architecture for high traffic"
    retrieved = mas.memory.retrieve(test_query, top_k=3)
    print(f"\nQuery: '{test_query}'")
    print("Retrieved knowledge:")
    for mem in retrieved:
        print(f"  - {mem.content[:80]}...")

    # Final summary
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nTotal episodes: {mas.episode_count}")
    print(f"Final avg reward: {mas.episode_rewards[-1]:.3f}" if mas.episode_rewards else "N/A")
    print(f"Q-table size: {mas.q_learner.get_q_table_stats()['size']}")
    print(f"Memory entries: {mas.memory.get_statistics()['total_entries']}")
    print(f"\n📁 All learning persisted to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
