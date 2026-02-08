#!/usr/bin/env python3
"""
Proper Learning Demo - Shows how agents should load/save learning

This demonstrates the correct way to:
1. Initialize LearningManager with config
2. Auto-load previous learning on startup
3. Per-agent Q-tables and memories
4. Save all learning at end
5. Domain-based transfer learning

Run: python demo_learning_proper.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import dspy
from core.integration.direct_claude_cli_lm import DirectClaudeCLI
from core.learning.learning_coordinator import LearningCoordinator as LearningManager
from core.memory.fallback_memory import MemoryType


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SwarmConfig:
    """Configuration for the swarm."""
    output_base_dir: str = "./outputs"

    # Learning settings
    auto_load_learning: bool = True
    per_agent_learning: bool = True
    shared_learning: bool = True
    learning_alpha: float = 0.3
    learning_gamma: float = 0.9
    learning_epsilon: float = 0.1
    max_q_table_size: int = 10000
    q_prune_percentage: float = 0.2
    enable_domain_transfer: bool = True

    # For Q-learner compatibility
    alpha: float = 0.3
    gamma: float = 0.9
    epsilon: float = 0.1
    tier1_max_size: int = 50
    tier2_max_clusters: int = 10
    tier3_max_size: int = 500
    max_experience_buffer: int = 200


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

class PlannerAgent:
    """Planner agent with its own learning."""

    def __init__(self, learning_manager: LearningManager):
        self.name = "Planner"
        self.q_learner = learning_manager.get_agent_learner(self.name)
        self.memory = learning_manager.get_agent_memory(self.name)

        class PlanSignature(dspy.Signature):
            """Create a plan for solving a problem."""
            problem: str = dspy.InputField()
            context: str = dspy.InputField()
            plan: str = dspy.OutputField()

        self.predictor = dspy.ChainOfThought(PlanSignature)

    def plan(self, problem: str) -> dict:
        """Create a plan, using learned knowledge."""
        # Get relevant context from memory
        memories = self.memory.retrieve(problem, top_k=3)
        context = "\n".join([m.content for m in memories]) if memories else "No prior knowledge"

        # Create state for Q-learning
        state = {
            'problem_type': problem[:50],
            'has_context': len(context) > 20,
            'memory_count': len(memories)
        }
        action = {'agent': self.name, 'task': 'planning'}

        try:
            result = self.predictor(problem=problem, context=context)
            plan = result.plan
            reward = 0.8  # Successful planning

            # Store lesson
            self.memory.store(
                f"PLAN for {problem[:30]}: {plan[:100]}",
                MemoryType.PROCEDURAL,
                importance=0.8
            )
        except Exception as e:
            plan = f"Error: {e}"
            reward = 0.2

        # Record experience
        self.q_learner.add_experience(state, action, reward)

        return {'plan': plan, 'reward': reward}


class ExecutorAgent:
    """Executor agent with its own learning."""

    def __init__(self, learning_manager: LearningManager):
        self.name = "Executor"
        self.q_learner = learning_manager.get_agent_learner(self.name)
        self.memory = learning_manager.get_agent_memory(self.name)

        class ExecuteSignature(dspy.Signature):
            """Execute a step of the plan."""
            task: str = dspy.InputField()
            plan: str = dspy.InputField()
            result: str = dspy.OutputField()
            success: str = dspy.OutputField()

        self.predictor = dspy.ChainOfThought(ExecuteSignature)

    def execute(self, task: str, plan: str) -> dict:
        """Execute task, learning from outcome."""
        state = {
            'task_type': task[:30],
            'has_plan': len(plan) > 20
        }
        action = {'agent': self.name, 'task': 'execution'}

        try:
            result = self.predictor(task=task, plan=plan)
            success = 'yes' in result.success.lower()
            reward = 0.9 if success else 0.3

            # Store lesson
            self.memory.store(
                f"RESULT for {task[:30]}: {'SUCCESS' if success else 'FAILED'}",
                MemoryType.EPISODIC,
                importance=reward
            )
        except Exception as e:
            result = type('obj', (object,), {'result': f"Error: {e}", 'success': 'no'})()
            success = False
            reward = 0.1

        self.q_learner.add_experience(state, action, reward)

        return {'result': result.result, 'success': success, 'reward': reward}


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print("="*70)
    print("🎓 PROPER LEARNING FLOW DEMONSTRATION")
    print("="*70)

    # Step 1: Create config
    config = SwarmConfig()
    print(f"\n📋 Config: auto_load={config.auto_load_learning}, per_agent={config.per_agent_learning}")

    # Step 2: Initialize LearningManager
    learning_manager = LearningManager(config, base_dir="outputs")

    # Step 3: Auto-load previous learning
    print("\n🔄 Step 1: Auto-loading previous learning...")
    loaded = learning_manager.initialize(auto_load=config.auto_load_learning)

    if loaded:
        print("   ✅ Loaded previous learning!")
        sessions = learning_manager.list_sessions()
        print(f"   📚 Available sessions: {len(sessions)}")
        for s in sessions[:3]:
            print(f"      - {s['session_id']}: {s['episodes']} episodes, reward={s['avg_reward']:.3f}")
    else:
        print("   📭 No previous learning found (starting fresh)")

    # Step 4: Initialize LLM
    print("\n⚡ Step 2: Initializing LLM...")
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)
    print("   ✅ Claude CLI configured")

    # Step 5: Create agents with per-agent learning
    print("\n🤖 Step 3: Creating agents with per-agent learning...")
    planner = PlannerAgent(learning_manager)
    executor = ExecutorAgent(learning_manager)

    planner_stats = planner.q_learner.get_q_table_stats()
    executor_stats = executor.q_learner.get_q_table_stats()
    print(f"   Planner Q-table: {planner_stats['size']} entries")
    print(f"   Executor Q-table: {executor_stats['size']} entries")
    print(f"   Planner memories: {planner.memory.get_statistics()['total_entries']}")
    print(f"   Executor memories: {executor.memory.get_statistics()['total_entries']}")

    # Step 6: Run a problem
    print("\n🎯 Step 4: Running a complex problem...")
    problem = "Design a real-time analytics pipeline for processing IoT sensor data"
    print(f"   Problem: {problem[:50]}...")

    # Planner creates plan
    plan_result = planner.plan(problem)
    print(f"   Planner: reward={plan_result['reward']:.2f}, plan={len(plan_result['plan'])} chars")

    # Executor executes
    exec_result = executor.execute("Implement data ingestion layer", plan_result['plan'])
    print(f"   Executor: reward={exec_result['reward']:.2f}, success={exec_result['success']}")

    # Step 7: Show learning accumulation
    print("\n📊 Step 5: Learning accumulated...")
    planner_stats = planner.q_learner.get_q_table_stats()
    executor_stats = executor.q_learner.get_q_table_stats()
    shared_stats = learning_manager.get_shared_learner().get_q_table_stats()

    print(f"   Planner Q-table: {planner_stats['size']} entries (avg Q={planner_stats.get('avg_q_value', 0):.3f})")
    print(f"   Executor Q-table: {executor_stats['size']} entries (avg Q={executor_stats.get('avg_q_value', 0):.3f})")
    print(f"   Shared Q-table: {shared_stats['size']} entries")

    # Step 8: Save all learning
    print("\n💾 Step 6: Saving all learning...")
    total_reward = plan_result['reward'] + exec_result['reward']
    learning_manager.save_all(
        episode_count=1,
        avg_reward=total_reward / 2,
        domains=['iot', 'analytics', 'data-pipeline']
    )

    summary = learning_manager.get_learning_summary()
    print(f"   Session: {summary['session_id']}")
    print(f"   Session dir: {summary['session_dir']}")
    print(f"   Total sessions in registry: {summary['total_sessions']}")

    # Step 9: Show how next session would work
    print("\n" + "="*70)
    print("📖 HOW NEXT SESSION LOADS LEARNING")
    print("="*70)
    print("""
    # In your next session, just do:

    config = SwarmConfig(auto_load_learning=True)
    learning_manager = LearningManager(config)
    learning_manager.initialize()  # <-- Auto-loads latest!

    # Or load domain-specific learning:
    learning_manager.load_domain_learning("iot")

    # Agents automatically get previous knowledge:
    planner = PlannerAgent(learning_manager)  # Has previous Q-values!
    """)

    print("\n✅ Demo complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
