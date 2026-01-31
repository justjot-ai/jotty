"""
Skill Orchestrator Demo
=======================

Demonstrates how the SkillOrchestrator auto-chains skills
to solve ANY ML problem with a single call.

Usage:
    result = await orchestrator.solve(X, y)

That's it! The orchestrator handles everything else.
"""

import asyncio
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


async def test_house_prices():
    """Test on California Housing (Regression)."""
    from sklearn.datasets import fetch_california_housing
    from core.orchestration.v2.skill_orchestrator import get_skill_orchestrator

    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: HOUSE PRICES (Regression)")
    logger.info("=" * 70)

    # Load data
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='price')

    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

    # ONE LINE TO SOLVE!
    orchestrator = get_skill_orchestrator()
    result = await orchestrator.solve(X, y, time_budget=60)

    logger.info(f"\n🏆 RESULT: R² = {result.best_score:.4f}")
    logger.info(f"   Features: {X.shape[1]} → {result.feature_count}")

    return result


async def test_classification():
    """Test on Digits (Classification)."""
    from sklearn.datasets import load_digits
    from core.orchestration.v2.skill_orchestrator import get_skill_orchestrator

    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: IMAGE CLASSIFICATION (Digits)")
    logger.info("=" * 70)

    # Load data
    digits = load_digits()
    X = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
    y = pd.Series(digits.target, name='digit')

    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes")

    # ONE LINE TO SOLVE!
    orchestrator = get_skill_orchestrator()
    result = await orchestrator.solve(X, y, time_budget=60)

    logger.info(f"\n🏆 RESULT: Accuracy = {result.best_score:.4f}")
    logger.info(f"   Features: {X.shape[1]} → {result.feature_count}")

    return result


async def test_fraud_detection():
    """Test on imbalanced fraud data (Classification)."""
    from sklearn.datasets import make_classification
    from core.orchestration.v2.skill_orchestrator import get_skill_orchestrator

    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: FRAUD DETECTION (Imbalanced)")
    logger.info("=" * 70)

    # Generate imbalanced data
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=15,
        weights=[0.97, 0.03], random_state=42
    )
    X = pd.DataFrame(X, columns=[f'V{i}' for i in range(20)])
    y = pd.Series(y, name='fraud')

    logger.info(f"Data: {len(y)} samples, {y.sum()} frauds ({y.mean()*100:.1f}%)")

    # ONE LINE TO SOLVE!
    orchestrator = get_skill_orchestrator()
    result = await orchestrator.solve(X, y, time_budget=60)

    logger.info(f"\n🏆 RESULT: Accuracy = {result.best_score:.4f}")
    logger.info(f"   Features: {X.shape[1]} → {result.feature_count}")

    return result


async def test_titanic():
    """Test on Titanic (our benchmark)."""
    from core.orchestration.v2.skill_orchestrator import get_skill_orchestrator

    print("\n" + "=" * 70)
    print("TEST 4: TITANIC (Benchmark)")
    print("=" * 70)

    # Load Titanic data - pass RAW columns, let Jotty figure out features
    data_path = Path(__file__).parent / "train.csv"
    df = pd.read_csv(data_path)

    # MINIMAL preprocessing - only handle missing values
    # Let Jotty's LLM reason about feature extraction from raw columns
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Cabin'] = df['Cabin'].fillna('Unknown')

    # Pass RAW columns - Jotty should learn to extract Title, Deck, Ticket features, etc.
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Cabin', 'Ticket']
    X = df[feature_cols]
    y = df['Survived']

    print(f"Data: {len(y)} samples, {y.mean()*100:.1f}% survived")
    print(f"Features (RAW): {list(X.columns)}")
    print(f"Sample Name: {X['Name'].iloc[0]}")
    print(f"Sample Cabin: {X['Cabin'].iloc[0]}")
    print(f"Sample Ticket: {X['Ticket'].iloc[0]}")

    # Business context - minimal hints, let Jotty discover patterns
    business_context = """
    Titanic survival prediction dataset.
    Analyze the text/string columns to discover extractable patterns.
    Look at the sample values to understand what information can be extracted.
    """

    # ONE LINE TO SOLVE - Jotty should figure out the rest!
    orchestrator = get_skill_orchestrator()
    result = await orchestrator.solve(X, y, time_budget=90, business_context=business_context)

    logger.info(f"\n🏆 RESULT: Accuracy = {result.best_score:.4f}")
    logger.info(f"   Features: {X.shape[1]} → {result.feature_count}")

    # Show top features
    if result.feature_importance:
        logger.info("\n📊 Top Features:")
        for feat, imp in list(result.feature_importance.items())[:5]:
            logger.info(f"   {feat}: {imp:.4f}")

    return result


async def main_quick():
    """Quick test - just Titanic."""
    print("=" * 70)
    print("SKILL ORCHESTRATOR - QUICK TEST (Titanic)")
    print("=" * 70)

    result = await test_titanic()

    print("\n" + "=" * 70)
    print(f"TITANIC RESULT: {result.best_score:.4f} accuracy")
    print("=" * 70)

    return {'Titanic': result}


async def main_full():
    """Run all tests."""
    print("=" * 70)
    print("SKILL ORCHESTRATOR - FULL TEST SUITE")
    print("=" * 70)
    print("One line to solve: orchestrator.solve(X, y)")
    print("=" * 70)

    results = {}

    # Run tests
    results['House Prices'] = await test_house_prices()
    results['Classification'] = await test_classification()
    results['Fraud Detection'] = await test_fraud_detection()
    results['Titanic'] = await test_titanic()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n{'Use Case':<25} {'Problem':<15} {'Score':<12} {'Features'}")
    print("-" * 65)

    for name, result in results.items():
        prob = result.problem_type.value
        score = result.best_score
        feats = f"{result.skill_results[0].metrics.get('n_features', '?')} → {result.feature_count}"
        print(f"{name:<25} {prob:<15} {score:<12.4f} {feats}")

    print("\n" + "=" * 70)
    print("ALL PROBLEMS SOLVED WITH: orchestrator.solve(X, y)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    import sys

    # Use --full for all tests, otherwise quick Titanic test
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        results = asyncio.run(main_full())
    else:
        results = asyncio.run(main_quick())
