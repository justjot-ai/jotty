"""
LLM Feature Reasoner Skill
==========================

Use LLM to reason about features from multiple perspectives.

This is the KEY skill for achieving 10/10 ML performance.
It uses chain-of-thought prompting to generate intelligent features.

Perspectives:
1. Text Engineer - Extract features from text/string columns
2. Domain Expert - Business/domain-specific features
3. Data Science Head - Statistical patterns, interactions, transformations
4. Feedback Analyst - Learn from feature importance and improve

The skill supports iterative feedback loops where the LLM learns
from model performance to generate improved features.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import MLSkill, SkillCategory, SkillResult

logger = logging.getLogger(__name__)


class LLMFeatureReasonerSkill(MLSkill):
    """
    LLM-powered feature reasoning skill.

    Uses chain-of-thought prompting to generate features from multiple perspectives.
    Supports feedback loops for iterative improvement.
    """

    name = "llm_feature_reasoning"
    version = "2.0.0"
    description = "Chain-of-thought LLM feature generation from multiple perspectives"
    category = SkillCategory.LLM_REASONING

    required_inputs = ["X"]
    optional_inputs = ["y", "eda_insights", "business_context", "feature_importance"]
    outputs = ["suggestions", "X_enhanced"]

    requires_llm = True

    # Chain-of-thought prompts (10/10 quality)
    PROMPTS = {
        "text_engineer": """You are a **Senior ML Engineer** specializing in text/string feature extraction.

## Analysis Phase (Think Step-by-Step)

**Step 1: Data Understanding**
String columns with samples: {string_samples}

**Step 2: Pattern Discovery**
Analyze each string column and identify:
- Embedded categories (titles, prefixes, suffixes)
- Hierarchical structure (A/B/C patterns, nested info)
- Length/character patterns that correlate with outcome
- Repeating values that form natural groups

**Step 3: Feature Design**
For each pattern found, determine:
- Best extraction method (regex, split, slice)
- How to handle edge cases (missing, malformed)
- Expected predictive value (high/medium/low)

## Code Generation Phase
Target: {problem_type} to predict {target}
Features: {features}

Generate ONLY executable Python code:
X['new_feature'] = X['column'].str.extract(r'pattern', expand=False)

Code only:""",
        "domain": """You are a **Principal Data Scientist** with domain expertise.

## Chain-of-Thought Analysis

**Step 1: Business Context Understanding**
Context: {context}
Problem: {problem_type} to predict {target}

**Step 2: Domain Knowledge Application**
Think about what domain expert would know:
- Which feature COMBINATIONS have business meaning?
- What real-world constraints exist?
- What derived metrics matter? (per-capita, ratios, rates)
- What segments/cohorts are meaningful?

**Step 3: Hypothesis Generation**
For each potential feature, ask:
- Does this have CAUSAL relationship with target?
- Is this capturing something NOT already in data?
- Will this generalize to new data?

## Code Generation
Features available: {features}

Generate 3-5 HIGH-VALUE domain features:

Return ONLY executable Python code:
X['family_size'] = X['SibSp'] + X['Parch'] + 1

Code only:""",
        "ds": """You are a **Kaggle Grandmaster** with expertise in statistical feature engineering.

## Structured Analysis

**Step 1: Distribution Analysis**
Features: {features}
For each numeric feature, consider:
- Is it skewed? → log/sqrt transform
- Has outliers? → clip or indicator
- Bimodal? → binary split

**Step 2: Correlation Hypothesis**
Think about which features might interact:
- Multiplicative relationships (area = length × width)
- Ratio relationships (rate = count / time)
- Difference relationships (profit = revenue - cost)

**Step 3: Feature Prioritization**
Rank potential features by expected information gain.

## Code Generation
Problem: {problem_type} to predict {target}

Generate 5-8 STATISTICALLY-MOTIVATED features:

Return ONLY executable Python code:
X['Age_log'] = np.log1p(X['Age'].clip(lower=0))

Code only:""",
        "feedback": """You are a **Kaggle Grandmaster** analyzing feature importance from a trained model.

## FEEDBACK FROM TRAINED MODEL (Iteration {iteration})

### Top Performing Features (These WORKED - create MORE like these):
{top_features}

### Weak Features (These did NOT help - avoid similar patterns):
{bottom_features}

## Analysis Task

**Step 1: Pattern Recognition**
Look at the TOP features. What patterns make them effective?

**Step 2: Generate IMPROVED Features**
Create NEW features that:
1. Extend the successful patterns to OTHER columns
2. Create variations of top features
3. Combine multiple successful features
4. AVOID patterns similar to the weak features

## Context
Problem: {problem_type}
Business context: {context}
Available columns: {columns}

Generate 5-8 NEW features:

Return ONLY executable Python code:
X['new_feature'] = ...

Code only:""",
        "group_analyst": """You are a Group/Aggregation Feature Engineer for {problem_type} to predict {target}.

Features: {features}
String columns: {string_samples}

Find columns where rows share SAME VALUES (groups).
For each groupable column, create:
1. GROUP SIZE - count of rows with same value
2. GROUP AGGREGATIONS - mean/std of numeric columns per group

Return ONLY executable Python code:
X['column_group_size'] = X.groupby('column')['column'].transform('count')

Code only:""",
        "interaction_analyst": """You are a Feature Interaction Analyst for {problem_type} to predict {target}.

Features: {features}

Find MEANINGFUL interactions:
1. PRODUCTS - multiply related features
2. RATIOS - divide related features
3. DIFFERENCES - subtract related features

Return ONLY executable Python code:
X['feat1_x_feat2'] = X['feat1'] * X['feat2']

Code only:""",
        "binning_analyst": """You are a Binning Expert for {problem_type} to predict {target}.

Features: {features}

Create MEANINGFUL bins with NUMERIC labels (0,1,2,3...):

Return ONLY executable Python code:
X['age_group'] = pd.cut(X['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])

Code only:""",
    }

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self._llm = None
        self._llm_available = None

    async def init(self) -> Any:
        """Initialize LLM client."""
        await super().init()
        self._init_llm()

    def _init_llm(self) -> Any:
        """Initialize LLM client using core.llm module."""
        if self._llm_available is None:
            try:
                from core.llm import generate_text

                self._llm = generate_text
                self._llm_available = True
                logger.info("LLM Feature Reasoner: Using Claude CLI")
            except Exception as e:
                logger.warning(f"LLM not available: {e}")
                self._llm_available = False
        return self._llm_available

    async def execute(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **context: Any
    ) -> SkillResult:
        """
        Execute LLM feature reasoning.

        Args:
            X: Input features
            y: Target variable
            **context: eda_insights, business_context, feature_importance, etc.

        Returns:
            SkillResult with feature suggestions and enhanced X
        """
        start_time = time.time()

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        self._init_llm()

        suggestions = []
        problem_type = context.get("problem_type", "classification")
        business_context = context.get("business_context", "")
        eda_insights = context.get("eda_insights", {})
        feature_importance = context.get("feature_importance", {})
        iteration = context.get("iteration", 0)

        # Build context for prompts
        features = list(X.columns)
        target = y.name if y is not None and hasattr(y, "name") else "target"

        # Get string column samples
        string_samples = self._get_string_samples(X)

        # Determine which perspectives to use
        if iteration > 0 and feature_importance:
            # Feedback loop mode
            perspectives = ["feedback"]
        else:
            # Initial mode - use all perspectives
            perspectives = [
                "text_engineer",
                "domain",
                "ds",
                "group_analyst",
                "interaction_analyst",
                "binning_analyst",
            ]

        # Generate features from each perspective
        for perspective in perspectives:
            prompt_template = self.PROMPTS.get(perspective, "")
            if not prompt_template:
                continue

            try:
                # Format prompt
                prompt = self._format_prompt(
                    prompt_template,
                    features=features,
                    target=target,
                    problem_type=problem_type,
                    context=business_context,
                    string_samples=string_samples,
                    eda_insights=eda_insights,
                    feature_importance=feature_importance,
                    iteration=iteration,
                )

                # Call LLM
                if self._llm_available and self._llm:
                    response = self._llm(prompt, provider="claude-cli", timeout=45)
                    parsed = self._parse_response(response, perspective)
                    suggestions.extend(parsed)

                    if parsed:
                        logger.info(f"LLM ({perspective}): {len(parsed)} features")

            except Exception as e:
                logger.debug(f"LLM reasoning failed for {perspective}: {e}")

        # Apply suggestions to X
        X_enhanced = self._apply_suggestions(X, suggestions)

        execution_time = time.time() - start_time

        return self._create_result(
            success=True,
            data=X_enhanced,
            metrics={
                "n_suggestions": len(suggestions),
                "n_applied": X_enhanced.shape[1] - X.shape[1],
                "n_perspectives": len(perspectives),
            },
            metadata={
                "suggestions": suggestions,
                "perspectives_used": perspectives,
            },
            execution_time=execution_time,
        )

    def _get_string_samples(self, X: pd.DataFrame) -> str:
        """Get string column samples for prompts."""
        string_cols = X.select_dtypes(include=["object"]).columns.tolist()
        samples = {}
        for col in string_cols[:5]:
            samples[col] = X[col].dropna().head(3).tolist()

        if not samples:
            return "No string columns"

        return "\n".join([f"  {col}: {vals}" for col, vals in samples.items()])

    def _format_prompt(self, template: str, **kwargs: Any) -> str:
        """Format prompt template with context."""
        # Handle feature_importance for feedback loop
        if "feature_importance" in kwargs and kwargs["feature_importance"]:
            fi = kwargs["feature_importance"]
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

            kwargs["top_features"] = "\n".join(
                [f"  - {f}: importance={imp:.4f}" for f, imp in sorted_fi[:10]]
            )
            kwargs["bottom_features"] = "\n".join(
                [f"  - {f}: importance={imp:.4f}" for f, imp in sorted_fi[-10:]]
            )
            kwargs["columns"] = list(fi.keys())[:30]

        # Format with available kwargs
        try:
            return template.format(**{k: v for k, v in kwargs.items() if v is not None})
        except KeyError:
            return template

    def _parse_response(self, response: str, perspective: str) -> List[Dict]:
        """Parse LLM response into feature suggestions."""
        suggestions = []
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if "=" in line and ("X[" in line or "df[" in line):
                suggestions.append({"perspective": perspective, "code": line, "source": "llm"})

        return suggestions

    def _apply_suggestions(self, X: pd.DataFrame, suggestions: List[Dict]) -> pd.DataFrame:
        """Apply feature suggestions to dataframe."""
        X_new = X.copy()
        applied = 0

        # Deduplicate
        created_features = set()
        dedup_suggestions = []
        for suggestion in suggestions:
            code = suggestion.get("code", "")
            match = re.search(r"X\['([^']+)'\]\s*=", code)
            if match:
                feat_name = match.group(1)
                if feat_name not in created_features:
                    created_features.add(feat_name)
                    dedup_suggestions.append(suggestion)
            else:
                dedup_suggestions.append(suggestion)

        for suggestion in dedup_suggestions:
            try:
                code = suggestion.get("code", "")
                local_vars = {"X": X_new, "np": np, "pd": pd}
                exec(code, {"__builtins__": {}}, local_vars)
                X_new = local_vars.get("X", X_new)
                applied += 1

                # Convert Categorical to numeric
                for col in X_new.columns:
                    if hasattr(X_new[col], "cat") or X_new[col].dtype.name == "category":
                        X_new[col] = X_new[col].cat.codes

            except Exception as e:
                logger.debug(f"Could not apply suggestion: {e}")

        logger.info(f"Applied {applied}/{len(dedup_suggestions)} suggestions")
        return X_new

    async def feedback_loop(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importance: Dict[str, float],
        iteration: int = 1,
        **context: Any,
    ) -> SkillResult:
        """
        Run feedback loop to generate improved features.

        This is called AFTER model training to learn from feature importance.

        Args:
            X: Current features
            y: Target
            feature_importance: Dict of feature_name -> importance
            iteration: Iteration number (1, 2, 3...)
            **context: Additional context

        Returns:
            SkillResult with improved features
        """
        return await self.execute(
            X, y, feature_importance=feature_importance, iteration=iteration, **context
        )

    # ========================================================================
    # CONVENIENCE METHODS (for direct use by orchestrator)
    # These provide simpler access patterns without the SkillResult wrapper
    # ========================================================================

    async def reason_features(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str, context: str = ""
    ) -> List[Dict]:
        """
        Generate feature suggestions using LLM reasoning.

        Convenience method for orchestrator use.
        Automatically runs EDA first to provide insights to LLM.

        Args:
            X: Input features
            y: Target variable
            problem_type: 'classification' or 'regression'
            context: Business context string

        Returns:
            List of suggestion dicts with 'code', 'perspective', 'source' keys
        """
        # Run EDA first to get insights for LLM prompts
        eda_insights = {}
        try:
            from .eda import EDASkill

            eda = EDASkill()
            eda_insights = eda.analyze(X, y, problem_type)

            # Also get EDA-based recommendations as suggestions
            eda_suggestions = []
            for rec in eda_insights.get("recommendations", []):
                if rec.get("code") and not rec["code"].startswith("#"):
                    eda_suggestions.append(
                        {
                            "perspective": "eda_rule",
                            "code": rec["code"],
                            "source": "eda",
                            "reason": rec.get("reason", ""),
                        }
                    )

            logger.info(f"EDA generated {len(eda_suggestions)} rule-based suggestions")
        except Exception as e:
            logger.debug(f"EDA analysis skipped: {e}")
            eda_suggestions = []

        # Run LLM reasoning with EDA insights
        result = await self.execute(
            X, y, problem_type=problem_type, business_context=context, eda_insights=eda_insights
        )

        if result.success:
            llm_suggestions = result.metadata.get("suggestions", [])
            # Combine EDA and LLM suggestions
            return eda_suggestions + llm_suggestions
        return eda_suggestions

    def apply_suggestions(
        self, X: pd.DataFrame, suggestions: List[Dict], drop_text_cols: bool = True
    ) -> pd.DataFrame:
        """
        Apply feature suggestions to dataframe.

        Public wrapper for _apply_suggestions with optional text column dropping.

        Args:
            X: Input dataframe
            suggestions: List of suggestion dicts with 'code' key
            drop_text_cols: Whether to drop high-cardinality text columns

        Returns:
            Enhanced dataframe with new features
        """
        X_new = self._apply_suggestions(X, suggestions)

        # Drop high-cardinality text columns to prevent leakage
        if drop_text_cols:
            original_text_cols = X.select_dtypes(include=["object"]).columns.tolist()
            n_rows = len(X_new)
            threshold = max(50, int(n_rows * 0.1))

            cols_to_drop = []
            for col in original_text_cols:
                if col in X_new.columns:
                    n_unique = X_new[col].nunique()
                    if n_unique > threshold:
                        cols_to_drop.append(col)

            if cols_to_drop:
                X_new = X_new.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} high-cardinality text columns")

        return X_new

    async def reason_with_feedback(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        context: str,
        feature_importance: Dict[str, float],
        iteration: int = 1,
    ) -> List[Dict]:
        """
        Generate improved features based on model feedback.

        Convenience method for orchestrator use.

        Args:
            X: Input features
            y: Target variable
            problem_type: 'classification' or 'regression'
            context: Business context string
            feature_importance: Dict of feature_name -> importance score
            iteration: Feedback loop iteration number

        Returns:
            List of suggestion dicts
        """
        result = await self.execute(
            X,
            y,
            problem_type=problem_type,
            business_context=context,
            feature_importance=feature_importance,
            iteration=iteration,
        )

        if result.success:
            return result.metadata.get("suggestions", [])
        return []
