"""
Dimensionality Reduction Skill for Jotty
========================================

Dimensionality reduction for visualization and feature extraction.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


async def pca_reduce_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Principal Component Analysis for dimensionality reduction.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - n_components: Number of components (or variance ratio like 0.95)
            - scale: Whether to scale features (default True)

    Returns:
        Dict with transformed data and explained variance
    """
    logger.info("[DimReduction] Running PCA...")

    data = params.get('data')
    n_components = params.get('n_components', 0.95)
    scale = params.get('scale', True)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_transformed = pca.fit_transform(X_scaled)

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=numeric_cols
    )

    # Feature importance per component
    feature_importance = {}
    for i in range(min(5, pca.n_components_)):
        top_features = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
        feature_importance[f'PC{i+1}'] = top_features.to_dict()

    logger.info(f"[DimReduction] PCA: {len(numeric_cols)} -> {pca.n_components_} components")

    return {
        'success': True,
        'transformed': X_transformed.tolist(),
        'n_components': int(pca.n_components_),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'total_variance_explained': float(sum(pca.explained_variance_ratio_)),
        'loadings': loadings.to_dict(),
        'feature_importance': feature_importance,
    }


async def tsne_reduce_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    t-SNE for visualization (2D/3D embedding).

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - n_components: 2 or 3 (default 2)
            - perplexity: Perplexity parameter (default 30)
            - learning_rate: Learning rate (default 'auto')
            - n_iter: Number of iterations (default 1000)

    Returns:
        Dict with 2D/3D embedding
    """
    from sklearn.manifold import TSNE

    logger.info("[DimReduction] Running t-SNE...")

    data = params.get('data')
    n_components = params.get('n_components', 2)
    perplexity = params.get('perplexity', 30)
    learning_rate = params.get('learning_rate', 'auto')
    n_iter = params.get('n_iter', 1000)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Adjust perplexity if needed
    perplexity = min(perplexity, len(X) - 1)

    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
        init='pca'
    )
    X_embedded = tsne.fit_transform(X_scaled)

    logger.info(f"[DimReduction] t-SNE: {len(numeric_cols)} -> {n_components}D")

    return {
        'success': True,
        'embedding': X_embedded.tolist(),
        'n_components': n_components,
        'perplexity': perplexity,
        'kl_divergence': float(tsne.kl_divergence_),
        'n_iter_final': tsne.n_iter_,
    }


async def umap_reduce_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    UMAP for dimensionality reduction and visualization.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - n_components: Number of dimensions (default 2)
            - n_neighbors: Number of neighbors (default 15)
            - min_dist: Minimum distance (default 0.1)
            - metric: Distance metric (default 'euclidean')

    Returns:
        Dict with UMAP embedding
    """
    try:
        import umap
    except ImportError:
        return {
            'success': False,
            'error': 'UMAP not installed. Install with: pip install umap-learn'
        }

    logger.info("[DimReduction] Running UMAP...")

    data = params.get('data')
    n_components = params.get('n_components', 2)
    n_neighbors = params.get('n_neighbors', 15)
    min_dist = params.get('min_dist', 0.1)
    metric = params.get('metric', 'euclidean')

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    X_embedded = reducer.fit_transform(X_scaled)

    logger.info(f"[DimReduction] UMAP: {len(numeric_cols)} -> {n_components}D")

    return {
        'success': True,
        'embedding': X_embedded.tolist(),
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'metric': metric,
    }


async def variance_analysis_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze feature variance and determine optimal components.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - variance_threshold: Cumulative variance threshold (default 0.95)

    Returns:
        Dict with variance analysis
    """
    logger.info("[DimReduction] Analyzing variance...")

    data = params.get('data')
    variance_threshold = params.get('variance_threshold', 0.95)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    # Find optimal components
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_components = int(np.argmax(cumulative_var >= variance_threshold) + 1)

    # Feature variance
    feature_variance = pd.Series(X.var(), index=numeric_cols).sort_values(ascending=False)

    # Low variance features
    low_var_threshold = 0.01
    low_variance_features = feature_variance[feature_variance < low_var_threshold].index.tolist()

    logger.info(f"[DimReduction] Optimal components for {variance_threshold*100}% variance: {optimal_components}")

    return {
        'success': True,
        'optimal_components': optimal_components,
        'variance_threshold': variance_threshold,
        'explained_variance_per_component': pca_full.explained_variance_ratio_.tolist(),
        'cumulative_variance': cumulative_var.tolist(),
        'feature_variance': feature_variance.to_dict(),
        'low_variance_features': low_variance_features,
        'total_features': len(numeric_cols),
    }


async def svd_reduce_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Truncated SVD for dimensionality reduction (works with sparse data).

    Args:
        params: Dict with keys:
            - data: DataFrame or sparse matrix
            - n_components: Number of components
            - algorithm: 'arpack' or 'randomized' (default 'randomized')

    Returns:
        Dict with SVD results
    """
    from sklearn.decomposition import TruncatedSVD

    logger.info("[DimReduction] Running Truncated SVD...")

    data = params.get('data')
    n_components = params.get('n_components', 50)
    algorithm = params.get('algorithm', 'randomized')

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        X = data[numeric_cols].fillna(0).values
    else:
        X = data

    n_components = min(n_components, X.shape[1] - 1)

    svd = TruncatedSVD(n_components=n_components, algorithm=algorithm, random_state=42)
    X_transformed = svd.fit_transform(X)

    logger.info(f"[DimReduction] SVD: {X.shape[1]} -> {n_components}")

    return {
        'success': True,
        'transformed': X_transformed.tolist(),
        'n_components': n_components,
        'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(sum(svd.explained_variance_ratio_)),
        'singular_values': svd.singular_values_.tolist(),
    }
