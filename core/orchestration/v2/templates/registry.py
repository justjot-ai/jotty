"""
Template Registry
=================

Central registry for all swarm templates.

Features:
- Register and discover templates
- Auto-detect best template for data
- Template versioning
- Custom template support

Usage:
    from jotty.templates import TemplateRegistry

    # Get a template
    template = TemplateRegistry.get("ml")

    # Auto-detect
    template = TemplateRegistry.auto_detect(X, y)

    # Register custom template
    TemplateRegistry.register(MyCustomTemplate)
"""

from typing import Dict, List, Type, Optional, Any
import numpy as np
import pandas as pd
import logging

from .base import SwarmTemplate

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Registry for swarm templates.

    The registry provides:
    - Template discovery by name
    - Auto-detection based on data characteristics
    - Custom template registration
    - Template metadata and versioning
    """

    _templates: Dict[str, Type[SwarmTemplate]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, template_class: Type[SwarmTemplate], name: str = None):
        """
        Register a template class.

        Args:
            template_class: The template class to register
            name: Optional name override (defaults to template.name)
        """
        if name is None:
            name = template_class.name

        cls._templates[name.lower()] = template_class
        logger.debug(f"Registered template: {name}")

    @classmethod
    def get(cls, name: str) -> SwarmTemplate:
        """
        Get a template instance by name.

        Args:
            name: Template name (case-insensitive)

        Returns:
            Template instance

        Raises:
            KeyError: If template not found
        """
        cls._ensure_initialized()

        name_lower = name.lower()
        if name_lower not in cls._templates:
            available = list(cls._templates.keys())
            raise KeyError(f"Template '{name}' not found. Available: {available}")

        return cls._templates[name_lower]()

    @classmethod
    def auto_detect(cls, X, y=None, **kwargs) -> SwarmTemplate:
        """
        Auto-detect the best template for the given data.

        Detection logic:
        1. Check data types (numeric, text, image)
        2. Check target variable (classification, regression, none)
        3. Check temporal patterns (time series)
        4. Match to best template

        Args:
            X: Input features (DataFrame, ndarray, or path)
            y: Target variable (optional)
            **kwargs: Additional hints

        Returns:
            Best matching template instance
        """
        cls._ensure_initialized()

        # Analyze data characteristics
        data_type = cls._detect_data_type(X)
        problem_type = cls._detect_problem_type(y)
        has_temporal = cls._detect_temporal(X)

        logger.info(f"Auto-detect: data_type={data_type}, problem_type={problem_type}, temporal={has_temporal}")

        # Decision tree for template selection
        if data_type == "image":
            return cls.get("cv") if "cv" in cls._templates else cls.get("ml")

        if data_type == "text":
            return cls.get("nlp") if "nlp" in cls._templates else cls.get("ml")

        if has_temporal:
            return cls.get("timeseries") if "timeseries" in cls._templates else cls.get("ml")

        # Default to ML
        return cls.get("ml")

    @classmethod
    def list_templates(cls) -> List[Dict[str, Any]]:
        """
        List all registered templates with metadata.

        Returns:
            List of template info dictionaries
        """
        cls._ensure_initialized()

        templates = []
        for name, template_class in cls._templates.items():
            instance = template_class()
            templates.append({
                'name': name,
                'version': instance.version,
                'description': instance.description,
                'supported_problems': instance.supported_problem_types,
                'agent_count': len(instance.agents),
                'stage_count': len(instance.pipeline),
            })

        return templates

    @classmethod
    def _ensure_initialized(cls):
        """Ensure built-in templates are registered."""
        if cls._initialized:
            return

        # Import and register built-in templates
        try:
            from .swarm_ml import SwarmML
            cls.register(SwarmML, "ml")
        except ImportError as e:
            logger.warning(f"Could not load SwarmML: {e}")

        # Future templates
        # from .swarm_nlp import SwarmNLP
        # cls.register(SwarmNLP, "nlp")

        # from .swarm_cv import SwarmCV
        # cls.register(SwarmCV, "cv")

        # from .swarm_timeseries import SwarmTimeSeries
        # cls.register(SwarmTimeSeries, "timeseries")

        cls._initialized = True

    @classmethod
    def _detect_data_type(cls, X) -> str:
        """Detect the type of input data."""
        if X is None:
            return "unknown"

        # Check if it's a path to images
        if isinstance(X, str):
            if any(ext in X.lower() for ext in ['.jpg', '.png', '.jpeg', 'image']):
                return "image"
            return "tabular"

        # Check if DataFrame
        if isinstance(X, pd.DataFrame):
            # Check for text columns
            text_cols = X.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                # Check if text columns have long strings (likely NLP)
                for col in text_cols:
                    avg_len = X[col].astype(str).str.len().mean()
                    if avg_len > 100:  # Long text
                        return "text"

            return "tabular"

        # Check if numpy array
        if isinstance(X, np.ndarray):
            # Check shape for images (4D: batch, height, width, channels)
            if len(X.shape) == 4:
                return "image"
            # Check shape for sequences
            if len(X.shape) == 3:
                return "sequence"
            return "tabular"

        return "unknown"

    @classmethod
    def _detect_problem_type(cls, y) -> str:
        """Detect the type of ML problem from target variable."""
        if y is None:
            return "unsupervised"

        if isinstance(y, pd.Series):
            y_values = y
        elif isinstance(y, np.ndarray):
            y_values = pd.Series(y)
        else:
            return "unknown"

        n_unique = y_values.nunique()
        n_samples = len(y_values)

        # Classification if few unique values
        if n_unique <= 20 and n_unique / n_samples < 0.05:
            return "classification"

        # Regression otherwise
        return "regression"

    @classmethod
    def _detect_temporal(cls, X) -> bool:
        """Detect if data has temporal patterns."""
        if isinstance(X, pd.DataFrame):
            # Check for datetime columns
            datetime_cols = X.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                return True

            # Check for common time column names
            time_keywords = ['date', 'time', 'timestamp', 'datetime', 'day', 'month', 'year']
            for col in X.columns:
                if any(kw in col.lower() for kw in time_keywords):
                    return True

        return False

    @classmethod
    def reset(cls):
        """Reset the registry (mainly for testing)."""
        cls._templates.clear()
        cls._initialized = False
