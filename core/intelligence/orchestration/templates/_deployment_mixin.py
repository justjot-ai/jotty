"""
Deployment readiness mixin for ProfessionalMLReport.

Provides methods for assessing model deployment readiness including
prediction latency benchmarking, model serialization size checks,
memory footprint estimation, deployment checklists, input schema
generation, ONNX exportability checks, dependency extraction, and
model signature analysis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .ml_report_generator import ReportContext

logger = logging.getLogger(__name__)


class DeploymentMixin:
    """Mixin class that provides deployment readiness assessment capabilities for ProfessionalMLReport.

    This mixin is intended to be used alongside ProfessionalMLReport (or a compatible base)
    that supplies attributes such as self._content, self.theme, self.figures_dir, and
    helper methods like self._maybe_add_narrative and self._store_section_data.
    """

    def add_deployment_readiness(
        self,
        model: Any,
        X_sample: Any,
        batch_sizes: List[int] = None,
        latency_threshold_ms: float = 10.0,
        size_threshold_mb: float = 100.0,
    ) -> Any:
        """
        Add deployment readiness assessment:
        - Prediction latency at various batch sizes (configurable threshold)
        - Model serialization size (configurable threshold)
        - Memory footprint estimation
        - Deployment checklist
        - Input schema generation (feature names, dtypes, value ranges)
        - ONNX exportability check
        - Dependency extraction
        - Model signature (input/output shape and dtype)

        Args:
            model: Trained model with predict method
            X_sample: Sample data for benchmarking
            batch_sizes: List of batch sizes to test
            latency_threshold_ms: Max acceptable single-sample latency in ms
            size_threshold_mb: Max acceptable model size in MB
        """
        try:
            import importlib
            import pickle
            import sys
            import time as _time

            X_arr = (
                X_sample
                if isinstance(X_sample, np.ndarray)
                else (X_sample.values if hasattr(X_sample, "values") else np.array(X_sample))
            )

            if batch_sizes is None:
                batch_sizes = [1, 10, 100, 1000]
            batch_sizes = [b for b in batch_sizes if b <= len(X_arr)]
            if not batch_sizes:
                batch_sizes = [min(len(X_arr), 1)]

            # Measure latency
            latency_results = []
            n_runs = 10

            for batch_size in batch_sizes:
                X_batch = X_arr[:batch_size]
                times = []
                for _ in range(n_runs):
                    start = _time.perf_counter()
                    model.predict(X_batch)
                    elapsed = (_time.perf_counter() - start) * 1000  # ms
                    times.append(elapsed)

                times_arr = np.array(times)
                latency_results.append(
                    {
                        "batch_size": batch_size,
                        "mean_ms": float(np.mean(times_arr)),
                        "p95_ms": float(np.percentile(times_arr, 95)),
                        "throughput": (
                            float(batch_size / (np.mean(times_arr) / 1000))
                            if np.mean(times_arr) > 0
                            else 0
                        ),
                    }
                )

            # Model serialization size
            model_size_mb = None
            try:
                serialized = pickle.dumps(model)
                model_size_mb = len(serialized) / (1024 * 1024)
            except Exception as e:
                self._record_internal_warning(
                    "ModelSerialization", "Failed to serialize model with pickle", e
                )
                pass

            # Memory footprint
            memory_bytes = sys.getsizeof(model)
            memory_mb = memory_bytes / (1024 * 1024)

            # Input schema generation
            input_schema = {}
            if hasattr(X_sample, "columns"):
                for col in X_sample.columns:
                    col_data = X_sample[col]
                    input_schema[col] = {
                        "dtype": str(col_data.dtype),
                        "min": (
                            float(col_data.min())
                            if np.issubdtype(col_data.dtype, np.number)
                            else None
                        ),
                        "max": (
                            float(col_data.max())
                            if np.issubdtype(col_data.dtype, np.number)
                            else None
                        ),
                    }
            else:
                for i in range(X_arr.shape[1]):
                    input_schema[f"feature_{i}"] = {
                        "dtype": str(X_arr[:, i].dtype),
                        "min": float(np.nanmin(X_arr[:, i])),
                        "max": float(np.nanmax(X_arr[:, i])),
                    }

            # Model signature
            model_signature = {
                "input_shape": list(X_arr[:1].shape),
                "input_dtype": str(X_arr.dtype),
            }
            try:
                out = model.predict(X_arr[:1])
                model_signature["output_shape"] = list(np.asarray(out).shape)
                model_signature["output_dtype"] = str(np.asarray(out).dtype)
            except Exception as e:
                self._record_internal_warning(
                    "ModelSignature", "Failed to determine model output signature", e
                )
                pass

            # ONNX exportability check
            onnx_available = False
            try:
                importlib.import_module("onnxruntime")
                importlib.import_module("skl2onnx")
                onnx_available = True
            except ImportError as e:
                self._record_internal_warning(
                    "ONNXImport", "ONNX runtime or skl2onnx not available for export", e
                )
                pass

            # Dependency extraction
            model_module = type(model).__module__
            model_package = model_module.split(".")[0] if model_module else "unknown"
            dependencies = {"model_module": model_module, "model_package": model_package}
            try:
                pkg = importlib.import_module(model_package)
                dependencies["package_version"] = getattr(pkg, "__version__", "unknown")
            except Exception as e:
                self._record_internal_warning(
                    "PackageVersion", "Failed to extract package version for model dependency", e
                )
                dependencies["package_version"] = "unknown"

            # Deployment checklist with configurable thresholds
            checklist = {
                "serializable": model_size_mb is not None,
                "has_predict": hasattr(model, "predict"),
                "has_predict_proba": hasattr(model, "predict_proba"),
                "deterministic": True,
                "latency_ok": (
                    latency_results[0]["mean_ms"] < latency_threshold_ms
                    if latency_results
                    else False
                ),
                "size_ok": model_size_mb < size_threshold_mb if model_size_mb else False,
            }

            # Test determinism
            if len(X_arr) > 0:
                pred1 = model.predict(X_arr[:1])
                pred2 = model.predict(X_arr[:1])
                checklist["deterministic"] = np.array_equal(pred1, pred2)

            passed = sum(1 for v in checklist.values() if v)
            total = len(checklist)

            # Create latency chart
            fig_path = self._create_latency_chart(latency_results)

            # Build content
            content = f"""
# Deployment Readiness Assessment

Evaluating model readiness for production deployment.

## Prediction Latency

| Batch Size | Mean Latency (ms) | P95 Latency (ms) | Throughput (samples/sec) |
|-----------|-------------------|-------------------|-------------------------|
"""
            for lr in latency_results:
                content += f"| {lr['batch_size']:,} | {lr['mean_ms']:.2f} | {lr['p95_ms']:.2f} | {lr['throughput']:,.0f} |\n"

            content += f"""
## Model Size

| Property | Value |
|----------|-------|
| Serialized Size | {f'{model_size_mb:.2f} MB' if model_size_mb else 'Not serializable'} |
| Memory Footprint | {memory_mb:.4f} MB |
| Model Type | {type(model).__name__} |

## Model Signature

| Property | Value |
|----------|-------|
| Input Shape | {model_signature.get('input_shape', 'N/A')} |
| Input Dtype | {model_signature.get('input_dtype', 'N/A')} |
| Output Shape | {model_signature.get('output_shape', 'N/A')} |
| Output Dtype | {model_signature.get('output_dtype', 'N/A')} |

## Dependencies

| Property | Value |
|----------|-------|
| Model Package | {dependencies['model_package']} |
| Package Version | {dependencies['package_version']} |
| ONNX Export Available | {'Yes' if onnx_available else 'No'} |

## Input Schema (Top 10 Features)

| Feature | Dtype | Min | Max |
|---------|-------|-----|-----|
"""
            for feat_name, schema in list(input_schema.items())[:10]:
                min_val = f"{schema['min']:.4f}" if schema["min"] is not None else "N/A"
                max_val = f"{schema['max']:.4f}" if schema["max"] is not None else "N/A"
                content += f"| {feat_name[:25]} | {schema['dtype']} | {min_val} | {max_val} |\n"

            content += f"""
## Deployment Checklist

| Check | Status |
|-------|--------|
| Serializable (pickle) | {'PASS' if checklist['serializable'] else 'FAIL'} |
| Has predict() | {'PASS' if checklist['has_predict'] else 'FAIL'} |
| Has predict_proba() | {'PASS' if checklist['has_predict_proba'] else 'FAIL'} |
| Deterministic | {'PASS' if checklist['deterministic'] else 'FAIL'} |
| Latency < {latency_threshold_ms}ms (single) | {'PASS' if checklist['latency_ok'] else 'FAIL'} |
| Size < {size_threshold_mb}MB | {'PASS' if checklist['size_ok'] else 'FAIL'} |

**Overall: {passed}/{total} checks passed**

"""
            if fig_path:
                content += f"""## Latency Profile

![Deployment Latency]({fig_path})

"""

            narrative = self._maybe_add_narrative(
                "Deployment Readiness",
                f'Passed: {passed}/{total}, Latency: {latency_results[0]["mean_ms"]:.2f}ms, Size: {model_size_mb}MB',
                section_type="deployment_readiness",
            )
            content += f"""{narrative}

---
"""
            self._content.append(content)
            self._store_section_data(
                "deployment_readiness",
                "Deployment Readiness",
                {
                    "latency_results": latency_results,
                    "model_size_mb": model_size_mb,
                    "checklist": checklist,
                    "input_schema": input_schema,
                    "model_signature": model_signature,
                    "dependencies": dependencies,
                    "onnx_available": onnx_available,
                },
                [{"type": "line"}],
            )

        except Exception as e:
            self._record_section_failure("Deployment Readiness", e)

    def _create_latency_chart(self, latency_results: Any) -> str:
        """Create 2-panel latency and throughput chart."""
        try:
            if not latency_results:
                return ""

            with self._chart_context("deployment_latency", figsize=(12, 5), nrows=1, ncols=2) as (
                fig,
                (ax1, ax2),
            ):
                batch_sizes = [r["batch_size"] for r in latency_results]
                mean_latencies = [r["mean_ms"] for r in latency_results]
                p95_latencies = [r["p95_ms"] for r in latency_results]
                throughputs = [r["throughput"] for r in latency_results]

                # Latency panel
                ax1.plot(
                    batch_sizes,
                    mean_latencies,
                    "o-",
                    color=self.theme["accent"],
                    linewidth=2,
                    markersize=8,
                    label="Mean",
                )
                ax1.plot(
                    batch_sizes,
                    p95_latencies,
                    "s--",
                    color=self.theme["warning"],
                    linewidth=2,
                    markersize=6,
                    label="P95",
                )
                ax1.axhline(
                    y=10,
                    color=self.theme["danger"],
                    linestyle=":",
                    alpha=0.7,
                    label="10ms threshold",
                )
                ax1.set_xlabel("Batch Size", fontsize=11)
                ax1.set_ylabel("Latency (ms)", fontsize=11)
                ax1.set_title(
                    "Prediction Latency",
                    fontsize=14,
                    fontweight="bold",
                    color=self.theme["primary"],
                )
                ax1.legend(fontsize=9)
                if len(batch_sizes) > 1:
                    ax1.set_xscale("log")

                # Throughput panel
                ax2.bar(
                    range(len(batch_sizes)),
                    throughputs,
                    color=self.theme["accent"],
                    alpha=0.85,
                    edgecolor="white",
                )
                ax2.set_xticks(range(len(batch_sizes)))
                ax2.set_xticklabels([str(b) for b in batch_sizes])
                ax2.set_xlabel("Batch Size", fontsize=11)
                ax2.set_ylabel("Throughput (samples/sec)", fontsize=11)
                ax2.set_title(
                    "Prediction Throughput",
                    fontsize=14,
                    fontweight="bold",
                    color=self.theme["primary"],
                )

                for i, tp in enumerate(throughputs):
                    ax2.text(
                        i,
                        tp + max(throughputs) * 0.02,
                        f"{tp:,.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="medium",
                    )

            return "figures/deployment_latency.png"
        except Exception as e:
            logger.debug(f"Failed to create latency chart: {e}")
            return ""
