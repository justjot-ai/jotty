"""
Demo: DAG Agents - Task Breakdown and Actor Assignment

This demo shows how to use the integrated TaskBreakdownAgent and TodoCreatorAgent
with Jotty's memory and learning systems.

Usage:
    python -m examples.demo_dag_agents
"""

import dspy
import logging
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agents.dag_agents import (
    TaskBreakdownAgent,
    TodoCreatorAgent,
    create_task_breakdown_agent,
    create_todo_creator_agent,
)
from core.foundation.data_structures import JottyConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample implementation plan
SAMPLE_PLAN = """
# User Authentication System Implementation Plan

## Phase 1: Database Setup
1. Create User model with fields: id, email, password_hash, created_at
2. Create migrations for users table
3. Add indexes on email field

## Phase 2: Authentication Logic
1. Implement password hashing using bcrypt
2. Create login endpoint with email/password validation
3. Create registration endpoint with email validation
4. Implement JWT token generation and validation

## Phase 3: Testing
1. Write unit tests for User model
2. Write integration tests for login endpoint
3. Write integration tests for registration endpoint
4. Add test fixtures for user data

## Phase 4: Security
1. Add rate limiting to login endpoint
2. Implement password strength validation
3. Add input sanitization

## Success Criteria
- All tests pass
- Login/register endpoints return correct status codes
- JWT tokens are properly validated
"""


# Available actors/agents
AVAILABLE_ACTORS = [
    {
        "name": "CodeAgent",
        "capabilities": ["coding", "implementation", "debugging", "git"],
        "description": "Writes and debugs code",
        "max_concurrent_tasks": 2
    },
    {
        "name": "TestAgent",
        "capabilities": ["testing", "validation", "test_fixtures", "coverage"],
        "description": "Writes and runs tests",
        "max_concurrent_tasks": 1
    },
    {
        "name": "SecurityAgent",
        "capabilities": ["security", "validation", "review", "analysis"],
        "description": "Reviews security and validates implementations",
        "max_concurrent_tasks": 1
    },
    {
        "name": "DatabaseAgent",
        "capabilities": ["database", "migrations", "sql", "setup"],
        "description": "Handles database operations and migrations",
        "max_concurrent_tasks": 1
    }
]


def demo_basic_usage():
    """Demonstrate basic usage of DAG agents."""
    print("\n" + "=" * 80)
    print("DEMO: Basic DAG Agent Usage")
    print("=" * 80)

    config = JottyConfig()

    # Step 1: Break down the plan into tasks
    print("\n📋 Step 1: Breaking down implementation plan...")
    breakdown_agent = create_task_breakdown_agent(config=config)
    markovian_todo = breakdown_agent.forward(SAMPLE_PLAN)

    print(f"\n✅ Created {len(markovian_todo.subtasks)} tasks:")
    for task_id, task in markovian_todo.subtasks.items():
        deps = f" (depends on: {', '.join(task.depends_on)})" if task.depends_on else ""
        print(f"   • {task_id}: {task.description[:60]}...{deps}")

    # Step 2: Assign actors and create executable DAG
    print("\n\n👥 Step 2: Assigning actors to tasks...")
    todo_agent = create_todo_creator_agent(config=config)
    executable_dag = todo_agent.create_executable_dag(
        markovian_todo=markovian_todo,
        available_actors=AVAILABLE_ACTORS
    )

    # Visualize assignments
    print(todo_agent.visualize_assignments(executable_dag))

    # Step 3: Show execution stages
    print("\n📊 Step 3: Execution stages (parallel groups):")
    stages = executable_dag.get_execution_stages()
    for i, stage in enumerate(stages, 1):
        tasks = [executable_dag.markovian_todo.subtasks[tid] for tid in stage]
        task_info = ", ".join([
            f"{t.task_id} ({executable_dag.assignments.get(t.task_id, 'unassigned').name if t.task_id in executable_dag.assignments else 'unassigned'})"
            for t in tasks
        ])
        print(f"   Stage {i}: {task_info}")

    # Step 4: Validation results
    print(f"\n✓ Validation passed: {executable_dag.validation_passed}")
    if executable_dag.validation_issues:
        print("⚠ Issues found:")
        for issue in executable_dag.validation_issues:
            print(f"   - {issue}")

    return executable_dag


def demo_learning_from_execution():
    """Demonstrate learning from execution outcomes."""
    print("\n" + "=" * 80)
    print("DEMO: Learning from Execution Outcomes")
    print("=" * 80)

    config = JottyConfig()

    # Create agents
    breakdown_agent = create_task_breakdown_agent(config=config)
    todo_agent = create_todo_creator_agent(config=config)

    # Break down plan
    markovian_todo = breakdown_agent.forward(SAMPLE_PLAN)

    # Create executable DAG
    executable_dag = todo_agent.create_executable_dag(
        markovian_todo=markovian_todo,
        available_actors=AVAILABLE_ACTORS
    )

    # Simulate execution outcomes
    print("\n🎲 Simulating execution outcomes...")
    outcomes = {}
    for task_id in executable_dag.markovian_todo.subtasks:
        # Simulate: 80% success rate
        import random
        success = random.random() < 0.8
        outcomes[task_id] = success
        status = "✅" if success else "❌"
        print(f"   {status} {task_id}")

    # Learn from outcomes
    print("\n📚 Learning from outcomes...")
    todo_agent.update_from_execution(executable_dag, outcomes)

    print("\n✓ Learning update complete")
    print("   - TD learning updated Q-values for actor-task_type pairs")
    print("   - Memory stored episodic/procedural patterns")

    return executable_dag, outcomes


def demo_serialization():
    """Demonstrate DAG serialization for persistence."""
    print("\n" + "=" * 80)
    print("DEMO: DAG Serialization")
    print("=" * 80)

    config = JottyConfig()

    # Create and execute
    breakdown_agent = create_task_breakdown_agent(config=config)
    todo_agent = create_todo_creator_agent(config=config)

    markovian_todo = breakdown_agent.forward(SAMPLE_PLAN)
    executable_dag = todo_agent.create_executable_dag(
        markovian_todo=markovian_todo,
        available_actors=AVAILABLE_ACTORS
    )

    # Serialize
    print("\n💾 Serializing ExecutableDAG...")
    dag_dict = executable_dag.to_dict()

    import json
    serialized = json.dumps(dag_dict, indent=2, default=str)
    print(f"\n✓ Serialized to {len(serialized)} bytes")
    print("\n📄 Preview (first 500 chars):")
    print(serialized[:500] + "...")

    return dag_dict


def main():
    """Run all demos."""
    print("\n" + "#" * 80)
    print("# JOTTY DAG AGENTS DEMO")
    print("#" * 80)

    # Configure DSPy (use a mock LM for demo)
    try:
        from core.foundation.unified_lm_provider import get_lm
        lm = get_lm()
        dspy.configure(lm=lm)
        print("\n✓ Configured DSPy with LLM")
    except Exception as e:
        print(f"\n⚠ Could not configure LLM: {e}")
        print("  Using mock responses for demo...")
        # For demo without LLM, we'd need mock responses
        return

    # Run demos
    try:
        demo_basic_usage()
        demo_learning_from_execution()
        demo_serialization()

        print("\n" + "#" * 80)
        print("# DEMO COMPLETE")
        print("#" * 80)

    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
