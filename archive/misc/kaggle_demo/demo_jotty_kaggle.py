"""
Jotty V2 Kaggle Demo
=====================

This demo shows how Jotty's skill system handles a Kaggle competition.
Uses the SkillsRegistry to access automl, hyperopt, and feature-engineer skills.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 60)
    logger.info("JOTTY V2 KAGGLE COMPETITION DEMO")
    logger.info("=" * 60)

    # Initialize Jotty's SkillsRegistry
    from core.registry.skills_registry import get_skills_registry
    registry = get_skills_registry()
    registry.init()

    logger.info(f"Loaded {len(registry.loaded_skills)} skills")

    # Check our ML skills are available
    ml_skills = ['automl', 'hyperopt', 'feature-engineer']
    for skill_name in ml_skills:
        skill = registry.get_skill(skill_name)
        if skill:
            logger.info(f"  ✅ {skill_name}: {list(skill.tools.keys())}")
        else:
            logger.warning(f"  ❌ {skill_name}: not found")

    # Load data
    data_path = Path(__file__).parent / "train.csv"
    import pandas as pd
    df = pd.read_csv(data_path)
    logger.info(f"\nLoaded data: {df.shape}")

    # ================================================================
    # STEP 1: Feature Engineering using Jotty skill
    # ================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: Feature Engineering (Jotty Skill)")
    logger.info("=" * 40)

    fe_skill = registry.get_skill('feature-engineer')
    if fe_skill and 'feature_engineer_tool' in fe_skill.tools:
        fe_tool = fe_skill.tools['feature_engineer_tool']
        fe_result = await fe_tool({
            'data': df,
            'target': 'Survived',
            'domain': 'titanic'
        })

        if fe_result['success']:
            df_engineered = fe_result['data']
            logger.info(f"  New features: {fe_result['new_features']}")
            logger.info(f"  Shape: {fe_result['original_shape']} -> {fe_result['new_shape']}")
        else:
            df_engineered = df
            logger.warning("  Feature engineering failed, using original data")
    else:
        df_engineered = df
        logger.warning("  feature-engineer skill not available")

    # ================================================================
    # STEP 2: AutoML using Jotty skill
    # ================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: AutoML Model Selection (Jotty Skill)")
    logger.info("=" * 40)

    automl_skill = registry.get_skill('automl')
    if automl_skill and 'automl_classify_tool' in automl_skill.tools:
        automl_tool = automl_skill.tools['automl_classify_tool']

        # Prepare features
        drop_cols = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
        feature_cols = [c for c in df_engineered.columns if c not in drop_cols]

        automl_result = await automl_tool({
            'data': df_engineered,
            'target': 'Survived',
            'features': feature_cols,
            'cv_folds': 5
        })

        if automl_result['success']:
            logger.info(f"  Best model: {automl_result['best_model']}")
            logger.info(f"  Best score: {automl_result['best_score']:.4f}")
            baseline_score = automl_result['best_score']
            models = automl_result['models']
            all_scores = automl_result['all_scores']
        else:
            logger.warning("  AutoML failed")
            baseline_score = 0
            models = {}
            all_scores = {}
    else:
        logger.warning("  automl skill not available")
        baseline_score = 0
        models = {}
        all_scores = {}

    # ================================================================
    # STEP 3: Hyperparameter Optimization using Jotty skill
    # ================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Hyperparameter Optimization (Jotty Skill)")
    logger.info("=" * 40)

    hyperopt_skill = registry.get_skill('hyperopt')
    if hyperopt_skill and 'hyperopt_optimize_tool' in hyperopt_skill.tools:
        hyperopt_tool = hyperopt_skill.tools['hyperopt_optimize_tool']

        hyperopt_result = await hyperopt_tool({
            'data': df_engineered[feature_cols + ['Survived']],
            'target': 'Survived',
            'model_type': 'xgboost',
            'n_trials': 50
        })

        if hyperopt_result['success']:
            logger.info(f"  Optimized score: {hyperopt_result['best_score']:.4f}")
            logger.info(f"  Improvement: {(hyperopt_result['best_score'] - baseline_score)*100:+.2f}%")
            optimized_score = hyperopt_result['best_score']
        else:
            logger.warning("  Hyperopt failed")
            optimized_score = baseline_score
    else:
        logger.warning("  hyperopt skill not available")
        optimized_score = baseline_score

    # ================================================================
    # STEP 4: Ensemble using Jotty skill
    # ================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: Ensemble Creation (Jotty Skill)")
    logger.info("=" * 40)

    if automl_skill and 'automl_ensemble_tool' in automl_skill.tools and models:
        ensemble_tool = automl_skill.tools['automl_ensemble_tool']

        ensemble_result = await ensemble_tool({
            'models': models,
            'all_scores': all_scores,
            'data': df_engineered[feature_cols + ['Survived']],
            'target': 'Survived',
            'top_k': 5,
            'method': 'voting'
        })

        if ensemble_result['success']:
            logger.info(f"  Ensemble score: {ensemble_result['ensemble_score']:.4f}")
            ensemble_score = ensemble_result['ensemble_score']
        else:
            logger.warning("  Ensemble failed")
            ensemble_score = optimized_score
    else:
        ensemble_score = optimized_score

    # ================================================================
    # FINAL RESULTS
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    final_score = max(baseline_score, optimized_score, ensemble_score)

    logger.info(f"  Baseline (AutoML):     {baseline_score:.4f}")
    logger.info(f"  Optimized (Hyperopt):  {optimized_score:.4f}")
    logger.info(f"  Ensemble:              {ensemble_score:.4f}")
    logger.info(f"  FINAL SCORE:           {final_score:.4f} ({final_score*100:.2f}%)")

    # Leaderboard comparison
    logger.info("\n" + "-" * 40)
    logger.info("TITANIC LEADERBOARD POSITION")
    logger.info("-" * 40)
    if final_score >= 0.86:
        logger.info(">> TOP 1% - EXPERT!")
    elif final_score >= 0.84:
        logger.info(">> TOP 5%")
    elif final_score >= 0.82:
        logger.info(">> TOP 10%")
    elif final_score >= 0.80:
        logger.info(">> TOP 25%")

    return {'final_score': final_score}


if __name__ == "__main__":
    result = asyncio.run(main())
