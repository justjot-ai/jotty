"""
Feature Tools Skill for Jotty
=============================

Automated deep feature synthesis using Featuretools.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("feature-tools")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def featuretools_dfs_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep Feature Synthesis using Featuretools.

    Args:
        params: Dict with keys:
            - data: DataFrame or dict of DataFrames for relational data
            - target_entity: Name of target entity (default 'data')
            - index_col: Index column name
            - time_index: Optional time index column
            - max_depth: Max depth for DFS (default 2)
            - agg_primitives: List of aggregation primitives
            - trans_primitives: List of transform primitives

    Returns:
        Dict with engineered features DataFrame
    """
    status.set_callback(params.pop('_status_callback', None))

    import featuretools as ft

    logger.info("[FeatureTools] Starting Deep Feature Synthesis...")

    data = params.get('data')
    target_entity = params.get('target_entity', 'data')
    index_col = params.get('index_col')
    time_index = params.get('time_index')
    max_depth = params.get('max_depth', 2)
    agg_primitives = params.get('agg_primitives', ['mean', 'sum', 'std', 'max', 'min', 'count'])
    trans_primitives = params.get('trans_primitives', ['day', 'month', 'year', 'weekday'])

    if isinstance(data, str):
        data = pd.read_csv(data)

    # Create EntitySet
    es = ft.EntitySet(id='main')

    if isinstance(data, dict):
        # Multiple dataframes (relational)
        for name, df in data.items():
            df = df.copy()
            if index_col and index_col in df.columns:
                es = es.add_dataframe(
                    dataframe_name=name,
                    dataframe=df,
                    index=index_col,
                    time_index=time_index if time_index and time_index in df.columns else None
                )
            else:
                df['_auto_index'] = range(len(df))
                es = es.add_dataframe(
                    dataframe_name=name,
                    dataframe=df,
                    index='_auto_index',
                    time_index=time_index if time_index and time_index in df.columns else None
                )
    else:
        # Single dataframe
        df = data.copy()
        if index_col and index_col in df.columns:
            es = es.add_dataframe(
                dataframe_name=target_entity,
                dataframe=df,
                index=index_col,
                time_index=time_index if time_index and time_index in df.columns else None
            )
        else:
            df['_auto_index'] = range(len(df))
            es = es.add_dataframe(
                dataframe_name=target_entity,
                dataframe=df,
                index='_auto_index',
                time_index=time_index if time_index and time_index in df.columns else None
            )

    # Run DFS
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name=target_entity,
        max_depth=max_depth,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        verbose=False
    )

    # Get feature names
    new_features = [f.get_name() for f in feature_defs]
    original_cols = df.columns.tolist()
    generated_features = [f for f in new_features if f not in original_cols]

    logger.info(f"[FeatureTools] Generated {len(generated_features)} new features")

    return {
        'success': True,
        'feature_matrix': feature_matrix.reset_index(),
        'feature_names': new_features,
        'generated_features': generated_features,
        'num_features': len(new_features),
    }


@async_tool_wrapper()
async def featuretools_primitives_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply specific feature primitives to data.

    Args:
        params: Dict with keys:
            - data: DataFrame
            - primitives: List of primitive names to apply
            - columns: Optional specific columns to apply primitives to

    Returns:
        Dict with transformed DataFrame
    """
    status.set_callback(params.pop('_status_callback', None))

    import featuretools as ft
    from featuretools.primitives import get_transform_primitives, get_aggregation_primitives

    logger.info("[FeatureTools] Applying primitives...")

    data = params.get('data')
    primitives = params.get('primitives', ['mean', 'std'])
    columns = params.get('columns')

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    if columns:
        df = df[columns]

    # Get available primitives
    trans_prims = get_transform_primitives()
    agg_prims = get_aggregation_primitives()

    # Create EntitySet
    df['_idx'] = range(len(df))
    es = ft.EntitySet(id='main')
    es = es.add_dataframe(
        dataframe_name='data',
        dataframe=df,
        index='_idx'
    )

    # Filter primitives
    trans_to_use = [p for p in primitives if p in trans_prims]
    agg_to_use = [p for p in primitives if p in agg_prims]

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='data',
        max_depth=1,
        agg_primitives=agg_to_use,
        trans_primitives=trans_to_use,
        verbose=False
    )

    logger.info(f"[FeatureTools] Applied {len(primitives)} primitives")

    return {
        'success': True,
        'feature_matrix': feature_matrix.reset_index(drop=True),
        'applied_primitives': primitives,
        'new_features': [f.get_name() for f in feature_defs],
    }


@async_tool_wrapper()
async def featuretools_normalize_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a denormalized table into entities.

    Args:
        params: Dict with keys:
            - data: DataFrame
            - base_entity: Name for base entity
            - normalize_config: Dict mapping new entity names to their columns

    Returns:
        Dict with EntitySet and normalized dataframes
    """
    status.set_callback(params.pop('_status_callback', None))

    import featuretools as ft

    logger.info("[FeatureTools] Normalizing data...")

    data = params.get('data')
    base_entity = params.get('base_entity', 'main')
    normalize_config = params.get('normalize_config', {})

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    df['_idx'] = range(len(df))

    es = ft.EntitySet(id='normalized')
    es = es.add_dataframe(
        dataframe_name=base_entity,
        dataframe=df,
        index='_idx'
    )

    # Normalize based on config
    for new_entity, config in normalize_config.items():
        index_col = config.get('index')
        additional_cols = config.get('columns', [])

        if index_col and index_col in df.columns:
            es = es.normalize_dataframe(
                base_dataframe_name=base_entity,
                new_dataframe_name=new_entity,
                index=index_col,
                additional_columns=additional_cols
            )

    # Get all dataframes
    dataframes = {}
    for df_name in es.dataframe_dict:
        dataframes[df_name] = es[df_name].copy()

    logger.info(f"[FeatureTools] Created {len(dataframes)} normalized entities")

    return {
        'success': True,
        'entity_set': es,
        'dataframes': dataframes,
        'entity_names': list(dataframes.keys()),
    }


@async_tool_wrapper()
async def featuretools_interesting_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find interesting features based on correlation with target.

    Args:
        params: Dict with keys:
            - feature_matrix: DataFrame from DFS
            - target: Target column or series
            - threshold: Correlation threshold (default 0.1)

    Returns:
        Dict with interesting features ranked by correlation
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[FeatureTools] Finding interesting features...")

    feature_matrix = params.get('feature_matrix')
    target = params.get('target')
    threshold = params.get('threshold', 0.1)

    if isinstance(feature_matrix, str):
        feature_matrix = pd.read_csv(feature_matrix)

    fm = feature_matrix.copy()

    # Get target series
    if isinstance(target, str) and target in fm.columns:
        y = fm[target]
        fm = fm.drop(columns=[target])
    elif isinstance(target, (pd.Series, np.ndarray, list)):
        y = pd.Series(target)
    else:
        return {'success': False, 'error': 'Invalid target specification'}

    # Calculate correlations
    numeric_cols = fm.select_dtypes(include=[np.number]).columns
    correlations = fm[numeric_cols].corrwith(y).abs()

    # Filter by threshold
    interesting = correlations[correlations >= threshold].sort_values(ascending=False)

    logger.info(f"[FeatureTools] Found {len(interesting)} interesting features")

    return {
        'success': True,
        'interesting_features': interesting.to_dict(),
        'feature_names': interesting.index.tolist(),
        'correlations': interesting.values.tolist(),
        'num_interesting': len(interesting),
    }
