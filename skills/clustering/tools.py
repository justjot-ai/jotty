"""
Clustering Skill for Jotty
==========================

Unsupervised clustering algorithms and analysis.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("clustering")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def cluster_kmeans_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    K-Means clustering with automatic k selection.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - n_clusters: Number of clusters (or 'auto' for elbow method)
            - max_clusters: Max clusters for auto selection (default 10)
            - scale: Whether to scale features (default True)

    Returns:
        Dict with cluster labels and metrics
    """
    status.set_callback(params.pop("_status_callback", None))

    logger.info("[Clustering] Running K-Means...")

    data = params.get("data")
    n_clusters = params.get("n_clusters", "auto")
    max_clusters = params.get("max_clusters", 10)
    scale = params.get("scale", True)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Auto-select k using elbow method
    if n_clusters == "auto":
        inertias = []
        silhouettes = []
        K_range = range(2, min(max_clusters + 1, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        # Find elbow (max silhouette)
        best_k = K_range[np.argmax(silhouettes)]
        n_clusters = best_k
        elbow_data = {"k": list(K_range), "inertia": inertias, "silhouette": silhouettes}
    else:
        elbow_data = None

    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Metrics
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    # Cluster centers
    centers = kmeans.cluster_centers_
    if scale:
        centers = scaler.inverse_transform(centers)

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    logger.info(f"[Clustering] K-Means with k={n_clusters}, silhouette={silhouette:.4f}")

    return {
        "success": True,
        "labels": labels.tolist(),
        "n_clusters": int(n_clusters),
        "metrics": {
            "silhouette": float(silhouette),
            "calinski_harabasz": float(calinski),
            "davies_bouldin": float(davies_bouldin),
            "inertia": float(kmeans.inertia_),
        },
        "cluster_centers": centers.tolist(),
        "cluster_sizes": cluster_sizes,
        "elbow_data": elbow_data,
    }


@async_tool_wrapper()
async def cluster_dbscan_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    DBSCAN clustering (density-based).

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - eps: Maximum distance between samples (or 'auto')
            - min_samples: Minimum samples per cluster (default 5)
            - scale: Whether to scale features (default True)

    Returns:
        Dict with cluster labels and metrics
    """
    status.set_callback(params.pop("_status_callback", None))

    from sklearn.neighbors import NearestNeighbors

    logger.info("[Clustering] Running DBSCAN...")

    data = params.get("data")
    eps = params.get("eps", "auto")
    min_samples = params.get("min_samples", 5)
    scale = params.get("scale", True)

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

    # Auto-select eps using k-distance graph
    if eps == "auto":
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])
        # Use knee point (simplified)
        eps = float(np.percentile(k_distances, 90))

    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Count clusters (excluding noise = -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    # Metrics (only if we have clusters)
    metrics = {}
    if n_clusters > 1:
        # Exclude noise points for metrics
        mask = labels != -1
        if mask.sum() > n_clusters:
            metrics["silhouette"] = float(silhouette_score(X_scaled[mask], labels[mask]))
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(X_scaled[mask], labels[mask])
            )

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    logger.info(f"[Clustering] DBSCAN found {n_clusters} clusters, {n_noise} noise points")

    return {
        "success": True,
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise_points": int(n_noise),
        "eps": float(eps),
        "min_samples": min_samples,
        "metrics": metrics,
        "cluster_sizes": cluster_sizes,
    }


@async_tool_wrapper()
async def cluster_hierarchical_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hierarchical/Agglomerative clustering.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - n_clusters: Number of clusters (or None for dendrogram)
            - linkage: 'ward', 'complete', 'average', 'single' (default 'ward')
            - scale: Whether to scale features (default True)

    Returns:
        Dict with cluster labels and dendrogram data
    """
    status.set_callback(params.pop("_status_callback", None))

    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    logger.info("[Clustering] Running Hierarchical clustering...")

    data = params.get("data")
    n_clusters = params.get("n_clusters", 3)
    linkage_method = params.get("linkage", "ward")
    scale = params.get("scale", True)

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

    # Get linkage matrix for dendrogram
    Z = scipy_linkage(X_scaled, method=linkage_method)

    # Cluster
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)

    # Metrics
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    logger.info(
        f"[Clustering] Hierarchical with {n_clusters} clusters, silhouette={silhouette:.4f}"
    )

    return {
        "success": True,
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "linkage": linkage_method,
        "metrics": {
            "silhouette": float(silhouette),
            "calinski_harabasz": float(calinski),
            "davies_bouldin": float(davies_bouldin),
        },
        "cluster_sizes": cluster_sizes,
        "linkage_matrix": Z.tolist()[:50],  # First 50 for brevity
    }


@async_tool_wrapper()
async def cluster_gmm_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gaussian Mixture Model clustering.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - n_components: Number of mixture components (or 'auto')
            - max_components: Max components for auto selection (default 10)
            - covariance_type: 'full', 'tied', 'diag', 'spherical' (default 'full')

    Returns:
        Dict with cluster labels and probabilities
    """
    status.set_callback(params.pop("_status_callback", None))

    logger.info("[Clustering] Running Gaussian Mixture Model...")

    data = params.get("data")
    n_components = params.get("n_components", "auto")
    max_components = params.get("max_components", 10)
    covariance_type = params.get("covariance_type", "full")

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-select n_components using BIC
    if n_components == "auto":
        bics = []
        aics = []
        for n in range(2, min(max_components + 1, len(X))):
            gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=42)
            gmm.fit(X_scaled)
            bics.append(gmm.bic(X_scaled))
            aics.append(gmm.aic(X_scaled))

        n_components = 2 + np.argmin(bics)
        selection_data = {
            "n": list(range(2, min(max_components + 1, len(X)))),
            "bic": bics,
            "aic": aics,
        }
    else:
        selection_data = None

    # Final model
    gmm = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type, random_state=42
    )
    labels = gmm.fit_predict(X_scaled)
    proba = gmm.predict_proba(X_scaled)

    # Metrics
    silhouette = silhouette_score(X_scaled, labels)

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    logger.info(f"[Clustering] GMM with {n_components} components, silhouette={silhouette:.4f}")

    return {
        "success": True,
        "labels": labels.tolist(),
        "probabilities": proba.tolist(),
        "n_components": int(n_components),
        "metrics": {
            "silhouette": float(silhouette),
            "bic": float(gmm.bic(X_scaled)),
            "aic": float(gmm.aic(X_scaled)),
            "log_likelihood": float(gmm.score(X_scaled)),
        },
        "cluster_sizes": cluster_sizes,
        "selection_data": selection_data,
    }


@async_tool_wrapper()
async def cluster_evaluate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate clustering quality with multiple metrics.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - labels: Cluster labels
            - true_labels: Optional true labels for external validation

    Returns:
        Dict with clustering evaluation metrics
    """
    status.set_callback(params.pop("_status_callback", None))

    from sklearn.metrics import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score

    logger.info("[Clustering] Evaluating clustering quality...")

    data = params.get("data")
    labels = np.array(params.get("labels"))
    true_labels = params.get("true_labels")

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics = {}

    # Internal validation
    n_clusters = len(np.unique(labels[labels >= 0]))
    if n_clusters > 1:
        mask = labels >= 0
        if mask.sum() > n_clusters:
            metrics["silhouette"] = float(silhouette_score(X_scaled[mask], labels[mask]))
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(X_scaled[mask], labels[mask])
            )
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_scaled[mask], labels[mask]))

    # External validation (if true labels provided)
    if true_labels is not None:
        true_labels = np.array(true_labels)
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(true_labels, labels))
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(true_labels, labels))
        metrics["homogeneity"] = float(homogeneity_score(true_labels, labels))

    # Cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    # Balance metric
    if n_clusters > 0:
        size_std = np.std(counts[unique >= 0])
        size_mean = np.mean(counts[unique >= 0])
        metrics["size_balance"] = float(1 - (size_std / size_mean)) if size_mean > 0 else 0

    logger.info(f"[Clustering] Evaluation complete for {n_clusters} clusters")

    return {
        "success": True,
        "metrics": metrics,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "n_noise": int((labels == -1).sum()) if -1 in labels else 0,
    }
