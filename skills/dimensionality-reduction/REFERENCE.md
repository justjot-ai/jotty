# Dimensionality Reduction Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`pca_reduce_tool`](#pca_reduce_tool) | Principal Component Analysis for dimensionality reduction. |
| [`tsne_reduce_tool`](#tsne_reduce_tool) | t-SNE for visualization (2D/3D embedding). |
| [`umap_reduce_tool`](#umap_reduce_tool) | UMAP for dimensionality reduction and visualization. |
| [`variance_analysis_tool`](#variance_analysis_tool) | Analyze feature variance and determine optimal components. |
| [`svd_reduce_tool`](#svd_reduce_tool) | Truncated SVD for dimensionality reduction (works with sparse data). |

---

## `pca_reduce_tool`

Principal Component Analysis for dimensionality reduction.

**Parameters:**

- **data**: DataFrame with features
- **n_components**: Number of components (or variance ratio like 0.95)
- **scale**: Whether to scale features (default True)

**Returns:** Dict with transformed data and explained variance

---

## `tsne_reduce_tool`

t-SNE for visualization (2D/3D embedding).

**Parameters:**

- **data**: DataFrame with features
- **n_components**: 2 or 3 (default 2)
- **perplexity**: Perplexity parameter (default 30)
- **learning_rate**: Learning rate (default 'auto')
- **n_iter**: Number of iterations (default 1000)

**Returns:** Dict with 2D/3D embedding

---

## `umap_reduce_tool`

UMAP for dimensionality reduction and visualization.

**Parameters:**

- **data**: DataFrame with features
- **n_components**: Number of dimensions (default 2)
- **n_neighbors**: Number of neighbors (default 15)
- **min_dist**: Minimum distance (default 0.1)
- **metric**: Distance metric (default 'euclidean')

**Returns:** Dict with UMAP embedding

---

## `variance_analysis_tool`

Analyze feature variance and determine optimal components.

**Parameters:**

- **data**: DataFrame with features
- **variance_threshold**: Cumulative variance threshold (default 0.95)

**Returns:** Dict with variance analysis

---

## `svd_reduce_tool`

Truncated SVD for dimensionality reduction (works with sparse data).

**Parameters:**

- **data**: DataFrame or sparse matrix
- **n_components**: Number of components
- **algorithm**: 'arpack' or 'randomized' (default 'randomized')

**Returns:** Dict with SVD results
