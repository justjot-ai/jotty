"""
Jotty v2 - Kaggle Swarm Demo
============================

Minimal code to demonstrate Jotty solving a Kaggle-style problem.
Uses DrZero + MorphAgent scoring + full ML pipeline.

Usage:
    python demo_kaggle_swarm.py
"""

import asyncio
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Jotty imports
from core.orchestration.v2.swarm import Swarm
from core.orchestration.v2.swarm_intelligence import SwarmIntelligence


async def main():
    print("=" * 70)
    print("JOTTY v2 - KAGGLE SWARM DEMO")
    print("DrZero Curriculum + MorphAgent Scoring + Auto ML Pipeline")
    print("=" * 70)

    # 1. Load dataset (Breast Cancer - fast, real problem)
    print("\n📥 Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    print(f"   Shape: {X.shape}")
    print(f"   Target: binary classification")
    print(f"   Features: {list(X.columns)[:3]}...")

    # 2. Initialize Swarm Intelligence with DrZero + MorphAgent
    print("\n🔧 Initializing Swarm Intelligence...")
    si = SwarmIntelligence()

    # Simulate agent task history
    print("   Recording agent task history...")
    task_history = [
        ('eda_agent', 'analysis', True), ('eda_agent', 'analysis', True),
        ('eda_agent', 'analysis', False),
        ('feature_agent', 'transformation', True), ('feature_agent', 'transformation', True),
        ('model_agent', 'validation', True), ('model_agent', 'validation', True),
        ('model_agent', 'analysis', True),
    ]
    for agent, task_type, success in task_history:
        si.record_task_result(agent, task_type, success, execution_time=1.0)

    # 3. MorphAgent Health Check
    print("\n📊 MorphAgent Swarm Health:")
    health = si.get_swarm_health()
    print(f"   RCS (Role Clarity): {health['avg_rcs']:.2f}")
    print(f"   RDS (Role Differentiation): {health['rds']:.2f}")
    print(f"   Trust Score: {health['avg_trust']:.2f}")

    # 4. DrZero Curriculum Generator
    print("\n🔥 DrZero Curriculum Generator:")
    for i in range(3):
        task = si.curriculum_generator.generate_training_task(si.agent_profiles)
        print(f"   Task {i+1}: {task.task_type} (difficulty: {task.difficulty:.2f})")
        si.curriculum_generator.update_from_result(task, success=(i != 1), execution_time=1.0)

    # 5. MorphAgent Task Routing
    print("\n🎯 MorphAgent TRAS Routing:")
    for task, task_type in [("Analyze feature distributions", "analysis"),
                            ("Engineer interaction features", "transformation")]:
        best = si.get_best_agent_for_task(
            task_type=task_type,
            available_agents=list(si.agent_profiles.keys()),
            task_description=task,
            use_morph_scoring=True
        )
        print(f"   '{task[:30]}...' -> {best}")

    # 6. Solve with Jotty Swarm
    print("\n🚀 Solving with Jotty Swarm ML Pipeline...")
    result = await Swarm.solve(
        template="ml",
        X=X,
        y=y,
        time_budget=60,
        context="Predict breast cancer diagnosis",
        feedback_iterations=1,
        show_progress=True
    )

    # 7. Final Results
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"   Success: {result.success}")
    print(f"   Score: {result.score:.4f}")
    print(f"   Features: {result.feature_count}")
    print(f"   Time: {result.execution_time:.1f}s")

    # Handle model which might be DataFrame or model object
    model = result.model
    if model is not None:
        if hasattr(model, '__class__'):
            print(f"   Model: {type(model).__name__}")
        else:
            print(f"   Model: {model}")
    else:
        print("   Model: None")

    # 8. MorphAgent Final Report
    print("\n📈 MorphAgent Report:")
    scores = si.compute_morph_scores()
    for name, score in scores.items():
        print(f"   {name}: RCS={score.rcs:.2f}, RDS={score.rds:.2f}")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
