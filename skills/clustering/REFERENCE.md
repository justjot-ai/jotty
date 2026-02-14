# Clustering Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`cluster_kmeans_tool`](#cluster_kmeans_tool) | K-Means clustering with automatic k selection. |
| [`cluster_dbscan_tool`](#cluster_dbscan_tool) | DBSCAN clustering (density-based). |
| [`cluster_hierarchical_tool`](#cluster_hierarchical_tool) | Hierarchical/Agglomerative clustering. |
| [`cluster_gmm_tool`](#cluster_gmm_tool) | Gaussian Mixture Model clustering. |
| [`cluster_evaluate_tool`](#cluster_evaluate_tool) | Evaluate clustering quality with multiple metrics. |

---

## `cluster_kmeans_tool`

K-Means clustering with automatic k selection.

**Parameters:**

- **data**: DataFrame with features
- **n_clusters**: Number of clusters (or 'auto' for elbow method)
- **max_clusters**: Max clusters for auto selection (default 10)
- **scale**: Whether to scale features (default True)

**Returns:** Dict with cluster labels and metrics

---

## `cluster_dbscan_tool`

DBSCAN clustering (density-based).

**Parameters:**

- **data**: DataFrame with features
- **eps**: Maximum distance between samples (or 'auto')
- **min_samples**: Minimum samples per cluster (default 5)
- **scale**: Whether to scale features (default True)

**Returns:** Dict with cluster labels and metrics

---

## `cluster_hierarchical_tool`

Hierarchical/Agglomerative clustering.

**Parameters:**

- **data**: DataFrame with features
- **n_clusters**: Number of clusters (or None for dendrogram)
- **linkage**: 'ward', 'complete', 'average', 'single' (default 'ward')
- **scale**: Whether to scale features (default True)

**Returns:** Dict with cluster labels and dendrogram data

---

## `cluster_gmm_tool`

Gaussian Mixture Model clustering.

**Parameters:**

- **data**: DataFrame with features
- **n_components**: Number of mixture components (or 'auto')
- **max_components**: Max components for auto selection (default 10)
- **covariance_type**: 'full', 'tied', 'diag', 'spherical' (default 'full')

**Returns:** Dict with cluster labels and probabilities

---

## `cluster_evaluate_tool`

Evaluate clustering quality with multiple metrics.

**Parameters:**

- **data**: DataFrame with features
- **labels**: Cluster labels
- **true_labels**: Optional true labels for external validation

**Returns:** Dict with clustering evaluation metrics
