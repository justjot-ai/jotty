"""
Jotty V2 Autonomous Kaggle Agent
=================================

A TRULY self-improving multi-agent system that:
1. Researches what techniques win competitions
2. Auto-discovers and installs tools it needs
3. Learns from results and adapts strategy
4. Continues until target performance is reached

This is NOT a static pipeline - it's an autonomous learning loop.
"""

import asyncio
import logging
import sys
import time
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class SwarmState:
    """Current state of the swarm's knowledge and progress."""
    current_score: float = 0.0
    target_score: float = 0.87  # Top legitimate Titanic scores
    iteration: int = 0
    max_iterations: int = 10

    # What we've learned
    techniques_tried: List[str] = field(default_factory=list)
    techniques_to_try: List[str] = field(default_factory=list)
    tools_installed: List[str] = field(default_factory=list)

    # Research findings
    research_findings: Dict[str, Any] = field(default_factory=dict)

    # Best model so far
    best_model: Any = None
    best_features: List[str] = field(default_factory=list)

    # History
    score_history: List[float] = field(default_factory=list)

    def is_goal_reached(self) -> bool:
        return self.current_score >= self.target_score

    def should_continue(self) -> bool:
        return (
            not self.is_goal_reached() and
            self.iteration < self.max_iterations and
            (len(self.techniques_to_try) > 0 or self.iteration == 0)
        )


@dataclass
class ResearchInsight:
    """An insight from research."""
    technique: str
    description: str
    expected_gain: str
    tools_needed: List[str]
    priority: int  # 1 = highest
    source: str


# =============================================================================
# Meta Agent - The Brain
# =============================================================================

class MetaAgent:
    """
    The orchestrator that decides WHAT to do next.

    Unlike static pipelines, this agent:
    - Evaluates current performance
    - Decides if more research is needed
    - Prioritizes which techniques to try
    - Knows when to stop
    """

    def __init__(self, state: SwarmState):
        self.state = state

    async def decide_next_action(self) -> str:
        """Decide what the swarm should do next."""

        if self.state.iteration == 0:
            return "research"  # Always start with research

        if self.state.is_goal_reached():
            return "done"

        # Check if we're improving
        if len(self.state.score_history) >= 2:
            recent_improvement = self.state.score_history[-1] - self.state.score_history[-2]
            if recent_improvement < 0.001 and len(self.state.techniques_to_try) == 0:
                return "research_more"  # Stuck, need new ideas

        if len(self.state.techniques_to_try) > 0:
            return "try_technique"

        if self.state.iteration >= self.state.max_iterations:
            return "done"

        return "research_more"

    def prioritize_techniques(self, insights: List[ResearchInsight]) -> List[str]:
        """Prioritize techniques based on expected gain and feasibility."""
        # Sort by priority, filter out already tried
        sorted_insights = sorted(insights, key=lambda x: x.priority)
        return [
            i.technique for i in sorted_insights
            if i.technique not in self.state.techniques_tried
        ]


# =============================================================================
# Research Agent - Uses SwarmResearcher
# =============================================================================

class ResearchAgent:
    """
    Researches what techniques win Kaggle competitions.

    Uses:
    - Web search for Kaggle winning solutions
    - LLM to analyze and extract techniques
    - SwarmResearcher for tool discovery
    """

    def __init__(self, state: SwarmState):
        self.state = state
        self._researcher = None

    def _init_researcher(self):
        """Lazy init SwarmResearcher."""
        if self._researcher is None:
            try:
                from core.orchestration.v2.swarm_researcher import SwarmResearcher
                self._researcher = SwarmResearcher()
            except ImportError:
                logger.warning("SwarmResearcher not available, using fallback")

    async def research_winning_techniques(self, competition: str) -> List[ResearchInsight]:
        """Research what techniques win this type of competition."""
        logger.info(f"[ResearchAgent] Researching winning techniques for: {competition}")

        self._init_researcher()

        insights = []

        # Knowledge base of Kaggle winning techniques (would normally come from web search)
        kaggle_knowledge = {
            "titanic": [
                ResearchInsight(
                    technique="optuna_tuning",
                    description="Use Optuna for Bayesian hyperparameter optimization instead of grid search",
                    expected_gain="+1-2% accuracy",
                    tools_needed=["optuna"],
                    priority=1,
                    source="Top Kaggle solutions"
                ),
                ResearchInsight(
                    technique="target_encoding",
                    description="Use target encoding for high-cardinality categoricals (Ticket, Cabin)",
                    expected_gain="+0.5-1% accuracy",
                    tools_needed=["category_encoders"],
                    priority=2,
                    source="Feature engineering best practices"
                ),
                ResearchInsight(
                    technique="ticket_grouping",
                    description="Extract ticket prefix and group passengers by shared tickets",
                    expected_gain="+0.5% accuracy",
                    tools_needed=[],
                    priority=3,
                    source="Top 1% Titanic solutions"
                ),
                ResearchInsight(
                    technique="name_features",
                    description="Extract more from names: surname frequency, name length, married women",
                    expected_gain="+0.3% accuracy",
                    tools_needed=[],
                    priority=4,
                    source="Feature engineering analysis"
                ),
                ResearchInsight(
                    technique="catboost",
                    description="CatBoost handles categoricals natively, often beats XGBoost",
                    expected_gain="+0.5-1% accuracy",
                    tools_needed=["catboost"],
                    priority=2,
                    source="Kaggle grandmaster tips"
                ),
                ResearchInsight(
                    technique="pseudo_labeling",
                    description="Use confident predictions on test set as additional training data",
                    expected_gain="+0.5% accuracy",
                    tools_needed=[],
                    priority=5,
                    source="Semi-supervised learning"
                ),
                ResearchInsight(
                    technique="stacking_meta_features",
                    description="Use out-of-fold predictions as features for meta-learner",
                    expected_gain="+0.5-1% accuracy",
                    tools_needed=[],
                    priority=3,
                    source="Ensemble techniques"
                ),
                ResearchInsight(
                    technique="feature_selection",
                    description="Use SHAP or permutation importance to remove noisy features",
                    expected_gain="+0.2-0.5% accuracy",
                    tools_needed=["shap"],
                    priority=4,
                    source="Model interpretability"
                ),
                ResearchInsight(
                    technique="cross_validation_averaging",
                    description="Train multiple models on different CV folds, average predictions",
                    expected_gain="+0.3% accuracy",
                    tools_needed=[],
                    priority=3,
                    source="Kaggle best practices"
                ),
            ]
        }

        # Get insights for this competition type
        comp_lower = competition.lower()
        for key, value in kaggle_knowledge.items():
            if key in comp_lower:
                insights.extend(value)

        # Use SwarmResearcher to discover additional tools
        if self._researcher:
            try:
                research_result = await self._researcher.research(
                    f"best machine learning techniques for {competition} kaggle",
                    research_type="tool"
                )
                logger.info(f"[ResearchAgent] Found {len(research_result.tools_found)} additional tools")
                self.state.research_findings['tools'] = research_result.tools_found
            except Exception as e:
                logger.debug(f"SwarmResearcher failed: {e}")

        logger.info(f"[ResearchAgent] Found {len(insights)} techniques to try")
        for insight in insights[:5]:
            logger.info(f"  - {insight.technique}: {insight.expected_gain}")

        return insights


# =============================================================================
# Tool Discovery Agent - Uses AutoProviderDiscovery
# =============================================================================

class ToolDiscoveryAgent:
    """
    Discovers and installs tools needed for techniques.

    Uses AutoProviderDiscovery and SwarmInstaller.
    """

    def __init__(self, state: SwarmState):
        self.state = state
        self._installer = None

    def _init_installer(self):
        if self._installer is None:
            try:
                from core.orchestration.v2.swarm_installer import SwarmInstaller
                self._installer = SwarmInstaller()
            except ImportError:
                logger.warning("SwarmInstaller not available, using pip directly")

    async def ensure_tools(self, tools: List[str]) -> Dict[str, bool]:
        """Ensure required tools are installed."""
        results = {}

        for tool in tools:
            if tool in self.state.tools_installed:
                results[tool] = True
                continue

            logger.info(f"[ToolDiscovery] Installing: {tool}")

            try:
                # Try using SwarmInstaller first
                self._init_installer()
                if self._installer:
                    result = await self._installer.install(tool)
                    success = result.success
                else:
                    # Fallback to pip
                    proc = subprocess.run(
                        [sys.executable, "-m", "pip", "install", tool, "-q"],
                        capture_output=True, timeout=120
                    )
                    success = proc.returncode == 0

                if success:
                    self.state.tools_installed.append(tool)
                    logger.info(f"[ToolDiscovery] ✅ Installed: {tool}")
                else:
                    logger.warning(f"[ToolDiscovery] ❌ Failed to install: {tool}")

                results[tool] = success

            except Exception as e:
                logger.warning(f"[ToolDiscovery] Error installing {tool}: {e}")
                results[tool] = False

        return results


# =============================================================================
# Technique Execution Agents
# =============================================================================

class TechniqueExecutor:
    """Base class for technique executors."""

    def __init__(self, state: SwarmState):
        self.state = state

    @abstractmethod
    async def execute(self, X, y, current_best_score: float) -> Dict[str, Any]:
        pass


class OptunaTuningExecutor(TechniqueExecutor):
    """Executes Optuna hyperparameter optimization."""

    async def execute(self, X, y, current_best_score: float) -> Dict[str, Any]:
        logger.info("[OptunaTuning] Starting Bayesian hyperparameter optimization...")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import xgboost as xgb
            import numpy as np

            # Preprocess
            X_processed = X.copy()
            for col in X_processed.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }

                model = xgb.XGBClassifier(**params)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                return scores.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, show_progress_bar=False)

            best_score = study.best_value
            best_params = study.best_params

            logger.info(f"[OptunaTuning] Best score: {best_score:.4f}")
            logger.info(f"[OptunaTuning] Improvement: {(best_score - current_best_score)*100:+.2f}%")

            # Train final model
            best_model = xgb.XGBClassifier(**best_params, random_state=42,
                                           use_label_encoder=False, eval_metric='logloss')
            best_model.fit(X_scaled, y)

            return {
                'success': True,
                'score': best_score,
                'model': best_model,
                'params': best_params,
                'technique': 'optuna_tuning'
            }

        except Exception as e:
            logger.error(f"[OptunaTuning] Failed: {e}")
            return {'success': False, 'score': current_best_score, 'error': str(e)}


class TargetEncodingExecutor(TechniqueExecutor):
    """Executes target encoding for categorical features."""

    async def execute(self, X, y, current_best_score: float) -> Dict[str, Any]:
        logger.info("[TargetEncoding] Applying target encoding...")

        try:
            import category_encoders as ce
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import xgboost as xgb

            X_processed = X.copy()

            # Find high-cardinality categoricals
            cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            high_card_cols = [c for c in cat_cols if X_processed[c].nunique() > 5]

            if high_card_cols:
                # Apply target encoding to high cardinality columns
                encoder = ce.TargetEncoder(cols=high_card_cols, smoothing=0.3)
                X_encoded = encoder.fit_transform(X_processed, y)
            else:
                X_encoded = X_processed

            # Encode remaining categoricals
            for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)

            # Test with XGBoost
            model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       random_state=42, use_label_encoder=False, eval_metric='logloss')

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            score = scores.mean()

            logger.info(f"[TargetEncoding] Score: {score:.4f}")
            logger.info(f"[TargetEncoding] Improvement: {(score - current_best_score)*100:+.2f}%")

            return {
                'success': True,
                'score': score,
                'technique': 'target_encoding',
                'encoded_cols': high_card_cols
            }

        except Exception as e:
            logger.error(f"[TargetEncoding] Failed: {e}")
            return {'success': False, 'score': current_best_score, 'error': str(e)}


class CatBoostExecutor(TechniqueExecutor):
    """Executes CatBoost which handles categoricals natively."""

    async def execute(self, X, y, current_best_score: float) -> Dict[str, Any]:
        logger.info("[CatBoost] Training CatBoost model...")

        try:
            from catboost import CatBoostClassifier
            from sklearn.model_selection import cross_val_score, StratifiedKFold

            X_processed = X.copy()

            # Find categorical columns
            cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()

            # Fill NaN in categoricals
            for col in cat_cols:
                X_processed[col] = X_processed[col].fillna('Unknown').astype(str)

            # Fill NaN in numericals
            for col in X_processed.select_dtypes(include=['number']).columns:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())

            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                cat_features=cat_cols,
                random_state=42,
                verbose=False
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
            score = scores.mean()

            logger.info(f"[CatBoost] Score: {score:.4f}")
            logger.info(f"[CatBoost] Improvement: {(score - current_best_score)*100:+.2f}%")

            # Train final model
            model.fit(X_processed, y, verbose=False)

            return {
                'success': True,
                'score': score,
                'model': model,
                'technique': 'catboost'
            }

        except Exception as e:
            logger.error(f"[CatBoost] Failed: {e}")
            return {'success': False, 'score': current_best_score, 'error': str(e)}


class AdvancedFeatureExecutor(TechniqueExecutor):
    """Creates advanced features based on research insights."""

    async def execute(self, X, y, current_best_score: float) -> Dict[str, Any]:
        logger.info("[AdvancedFeatures] Engineering research-based features...")

        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import xgboost as xgb

            X_processed = X.copy()
            new_features = []

            # 1. Ticket grouping (from top solutions)
            if 'Ticket' in X_processed.columns:
                # Extract ticket prefix
                X_processed['TicketPrefix'] = X_processed['Ticket'].apply(
                    lambda x: ''.join(filter(str.isalpha, str(x).split()[0])) or 'None'
                )
                # Count passengers per ticket
                ticket_counts = X_processed['Ticket'].value_counts()
                X_processed['TicketGroupSize'] = X_processed['Ticket'].map(ticket_counts)
                X_processed['IsSharedTicket'] = (X_processed['TicketGroupSize'] > 1).astype(int)
                new_features.extend(['TicketPrefix', 'TicketGroupSize', 'IsSharedTicket'])

            # 2. Name features (from research)
            if 'Name' in X_processed.columns:
                # Surname frequency (family groups)
                X_processed['Surname'] = X_processed['Name'].apply(
                    lambda x: str(x).split(',')[0].strip()
                )
                surname_counts = X_processed['Surname'].value_counts()
                X_processed['SurnameFreq'] = X_processed['Surname'].map(surname_counts)

                # Name length (proxy for social status)
                X_processed['NameLength'] = X_processed['Name'].apply(len)

                # Has parentheses (maiden name = married woman)
                X_processed['HasMaidenName'] = X_processed['Name'].str.contains(r'\(').astype(int)

                new_features.extend(['SurnameFreq', 'NameLength', 'HasMaidenName'])

            # 3. Cabin deck detailed
            if 'Cabin' in X_processed.columns:
                # Multiple cabins = wealth
                X_processed['NumCabins'] = X_processed['Cabin'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                )
                new_features.append('NumCabins')

            # 4. Age * Class interaction (children in higher class survived more)
            if 'Age' in X_processed.columns and 'Pclass' in X_processed.columns:
                X_processed['Age'].fillna(X_processed['Age'].median(), inplace=True)
                X_processed['AgeClass'] = X_processed['Age'] / X_processed['Pclass']
                X_processed['ChildInFirstClass'] = (
                    (X_processed['Age'] < 15) & (X_processed['Pclass'] == 1)
                ).astype(int)
                new_features.extend(['AgeClass', 'ChildInFirstClass'])

            # 5. Family survival rate (if we had this info)
            if 'SurnameFreq' in X_processed.columns:
                X_processed['LargeFamily'] = (X_processed['SurnameFreq'] > 3).astype(int)
                new_features.append('LargeFamily')

            logger.info(f"[AdvancedFeatures] Created {len(new_features)} new features: {new_features}")

            # Prepare for modeling
            drop_cols = ['Name', 'Ticket', 'Cabin', 'Surname']
            feature_cols = [c for c in X_processed.columns if c not in drop_cols]
            X_model = X_processed[feature_cols]

            # Encode categoricals
            for col in X_model.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_model[col] = le.fit_transform(X_model[col].astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_model)

            # Test
            model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       random_state=42, use_label_encoder=False, eval_metric='logloss')
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            score = scores.mean()

            logger.info(f"[AdvancedFeatures] Score: {score:.4f}")
            logger.info(f"[AdvancedFeatures] Improvement: {(score - current_best_score)*100:+.2f}%")

            return {
                'success': True,
                'score': score,
                'technique': 'advanced_features',
                'new_features': new_features,
                'X_enhanced': X_processed
            }

        except Exception as e:
            logger.error(f"[AdvancedFeatures] Failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'score': current_best_score, 'error': str(e)}


class StackingMetaFeaturesExecutor(TechniqueExecutor):
    """Creates stacking with out-of-fold meta features."""

    async def execute(self, X, y, current_best_score: float) -> Dict[str, Any]:
        logger.info("[StackingMeta] Creating OOF meta-features...")

        try:
            import numpy as np
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            import xgboost as xgb
            import lightgbm as lgb

            # Preprocess
            X_processed = X.copy()
            for col in X_processed.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            y_arr = y.values

            # Base models
            base_models = {
                'xgb': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                          random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'lgb': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                           random_state=42, verbose=-1),
                'rf': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
            }

            # Generate OOF predictions
            n_folds = 5
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            meta_features = np.zeros((len(X_scaled), len(base_models)))

            for i, (name, model) in enumerate(base_models.items()):
                oof_preds = np.zeros(len(X_scaled))

                for train_idx, val_idx in cv.split(X_scaled, y_arr):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_train, y_train)

                    if hasattr(model_clone, 'predict_proba'):
                        oof_preds[val_idx] = model_clone.predict_proba(X_val)[:, 1]
                    else:
                        oof_preds[val_idx] = model_clone.predict(X_val)

                meta_features[:, i] = oof_preds
                logger.info(f"[StackingMeta] Generated OOF for {name}")

            # Combine original features with meta features
            X_stacked = np.hstack([X_scaled, meta_features])

            # Train meta-learner
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(meta_learner, X_stacked, y_arr, cv=5, scoring='accuracy')
            score = scores.mean()

            logger.info(f"[StackingMeta] Score: {score:.4f}")
            logger.info(f"[StackingMeta] Improvement: {(score - current_best_score)*100:+.2f}%")

            return {
                'success': True,
                'score': score,
                'technique': 'stacking_meta_features'
            }

        except Exception as e:
            logger.error(f"[StackingMeta] Failed: {e}")
            return {'success': False, 'score': current_best_score, 'error': str(e)}


# =============================================================================
# Main Autonomous Swarm
# =============================================================================

class AutonomousKaggleSwarm:
    """
    Self-improving Kaggle competition swarm.

    This is the main orchestrator that:
    1. Uses MetaAgent to decide what to do
    2. Uses ResearchAgent to find techniques
    3. Uses ToolDiscoveryAgent to install tools
    4. Executes techniques and learns from results
    5. Loops until target is reached or max iterations
    """

    def __init__(self, target_score: float = 0.87):
        self.state = SwarmState(target_score=target_score)
        self.meta_agent = MetaAgent(self.state)
        self.research_agent = ResearchAgent(self.state)
        self.tool_agent = ToolDiscoveryAgent(self.state)

        # Technique executors
        self.executors = {
            'optuna_tuning': OptunaTuningExecutor(self.state),
            'target_encoding': TargetEncodingExecutor(self.state),
            'catboost': CatBoostExecutor(self.state),
            'ticket_grouping': AdvancedFeatureExecutor(self.state),  # Combined
            'name_features': AdvancedFeatureExecutor(self.state),
            'stacking_meta_features': StackingMetaFeaturesExecutor(self.state),
        }

        # Data
        self.X = None
        self.y = None
        self.X_enhanced = None

    async def run(self, data_path: str, target: str = 'Survived') -> Dict[str, Any]:
        """Run the autonomous improvement loop."""

        logger.info("=" * 70)
        logger.info("JOTTY V2 AUTONOMOUS KAGGLE SWARM")
        logger.info("=" * 70)
        logger.info(f"Target score: {self.state.target_score:.2%}")
        logger.info("")

        # Load and prepare data
        import pandas as pd
        df = pd.read_csv(data_path)

        # Basic feature engineering (baseline)
        df = await self._baseline_features(df)

        # Prepare X and y
        drop_cols = [target, 'PassengerId']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        self.X = df[feature_cols]
        self.y = df[target]

        # Get baseline score
        baseline_score = await self._get_baseline_score()
        self.state.current_score = baseline_score
        self.state.score_history.append(baseline_score)

        logger.info(f"Baseline score: {baseline_score:.4f} ({baseline_score:.2%})")
        logger.info("")

        # Main autonomous loop
        while self.state.should_continue():
            self.state.iteration += 1

            logger.info("=" * 50)
            logger.info(f"ITERATION {self.state.iteration}")
            logger.info(f"Current: {self.state.current_score:.4f} | Target: {self.state.target_score:.4f}")
            logger.info("=" * 50)

            # 1. Decide next action
            action = await self.meta_agent.decide_next_action()
            logger.info(f"[MetaAgent] Decision: {action}")

            if action == "done":
                break

            # 2. Research if needed
            if action in ["research", "research_more"]:
                insights = await self.research_agent.research_winning_techniques("titanic")
                self.state.techniques_to_try = self.meta_agent.prioritize_techniques(insights)

                # Collect tools needed
                tools_needed = []
                for insight in insights:
                    if insight.technique in self.state.techniques_to_try[:3]:
                        tools_needed.extend(insight.tools_needed)

                # Install tools
                if tools_needed:
                    await self.tool_agent.ensure_tools(list(set(tools_needed)))

            # 3. Try next technique
            if len(self.state.techniques_to_try) > 0:
                technique = self.state.techniques_to_try.pop(0)
                logger.info(f"[Swarm] Trying technique: {technique}")

                if technique in self.executors:
                    X_to_use = self.X_enhanced if self.X_enhanced is not None else self.X
                    result = await self.executors[technique].execute(
                        X_to_use, self.y, self.state.current_score
                    )

                    self.state.techniques_tried.append(technique)

                    if result['success'] and result['score'] > self.state.current_score:
                        improvement = (result['score'] - self.state.current_score) * 100
                        logger.info(f"[Swarm] ✅ Improvement! +{improvement:.2f}%")
                        self.state.current_score = result['score']

                        if 'model' in result:
                            self.state.best_model = result['model']
                        if 'X_enhanced' in result:
                            self.X_enhanced = result['X_enhanced']
                    else:
                        logger.info(f"[Swarm] ❌ No improvement from {technique}")

                    self.state.score_history.append(self.state.current_score)

            logger.info("")

        # Final summary
        return self._generate_summary()

    async def _baseline_features(self, df):
        """Create baseline features."""
        import pandas as pd

        # Title extraction
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
            df['Title'] = df['Title'].replace(title_mapping)
            rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
            df['Title'] = df['Title'].replace(rare_titles, 'Rare')

        # Family features
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Age
        if 'Age' in df.columns:
            if 'Title' in df.columns:
                age_by_title = df.groupby('Title')['Age'].median()
                for title in df['Title'].unique():
                    mask = (df['Age'].isnull()) & (df['Title'] == title)
                    if mask.any() and title in age_by_title.index:
                        df.loc[mask, 'Age'] = age_by_title[title]
            df['Age'].fillna(df['Age'].median(), inplace=True)
            df['IsChild'] = (df['Age'] < 12).astype(int)

        # Fare
        if 'Fare' in df.columns:
            df['Fare'].fillna(df['Fare'].median(), inplace=True)

        # Cabin
        if 'Cabin' in df.columns:
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            df['CabinDeck'] = df['Cabin'].str[0].fillna('U')

        # Embarked
        if 'Embarked' in df.columns:
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        return df

    async def _get_baseline_score(self):
        """Get baseline XGBoost score."""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        import xgboost as xgb

        X_processed = self.X.copy()
        for col in X_processed.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)

        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                   random_state=42, use_label_encoder=False, eval_metric='logloss')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, self.y, cv=cv, scoring='accuracy')
        return scores.mean()

    def _generate_summary(self):
        """Generate final summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("AUTONOMOUS SWARM COMPLETE")
        logger.info("=" * 70)

        logger.info(f"Iterations: {self.state.iteration}")
        logger.info(f"Techniques tried: {len(self.state.techniques_tried)}")
        logger.info(f"Tools installed: {self.state.tools_installed}")
        logger.info("")
        logger.info("Score progression:")
        for i, score in enumerate(self.state.score_history):
            marker = "← START" if i == 0 else ("← BEST" if score == max(self.state.score_history) else "")
            logger.info(f"  {i}: {score:.4f} ({score:.2%}) {marker}")

        logger.info("")
        logger.info(f"FINAL SCORE: {self.state.current_score:.4f} ({self.state.current_score:.2%})")

        if self.state.is_goal_reached():
            logger.info(f"🎉 TARGET REACHED! ({self.state.target_score:.2%})")
        else:
            gap = (self.state.target_score - self.state.current_score) * 100
            logger.info(f"Gap to target: {gap:.2f}%")

        # Benchmark
        logger.info("")
        logger.info("-" * 40)
        logger.info("TITANIC LEADERBOARD COMPARISON")
        logger.info("-" * 40)
        score = self.state.current_score
        if score >= 0.87:
            logger.info(">> TOP 0.1% - GRANDMASTER LEVEL!")
        elif score >= 0.85:
            logger.info(">> TOP 1% - EXPERT LEVEL!")
        elif score >= 0.82:
            logger.info(">> TOP 10%")
        elif score >= 0.80:
            logger.info(">> TOP 25%")

        return {
            'final_score': self.state.current_score,
            'iterations': self.state.iteration,
            'techniques_tried': self.state.techniques_tried,
            'tools_installed': self.state.tools_installed,
            'score_history': self.state.score_history,
            'target_reached': self.state.is_goal_reached(),
        }


async def main():
    """Run the autonomous Kaggle swarm."""
    swarm = AutonomousKaggleSwarm(target_score=0.87)

    data_path = Path(__file__).parent / "train.csv"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    results = await swarm.run(str(data_path), target='Survived')
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
