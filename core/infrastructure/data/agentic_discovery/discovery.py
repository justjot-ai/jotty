from typing import Any

"""
Agentic Discovery Module - LLM-Based Artifact Analysis and Registration
========================================================================

A-TEAM DESIGN CONSENSUS:
------------------------
- Turing: Validates computability of semantic analysis operations
- Shannon: Designs information quantification for artifact descriptions
- DSPy Author: Uses proper DSPy signatures for LLM interactions
- Nash/von Neumann: Ensures discovery is a cooperative game (not competitive)
- Alex Chen (MIT): Named for clarity and intuitiveness

PURPOSE:
--------
This module enables agents to autonomously discover, analyze, and register
data artifacts produced by other agents in the swarm. It uses LLM-based
semantic analysis to generate rich metadata for each artifact.

KEY COMPONENTS:
---------------
1. ArtifactAnalyzer - LLM-based artifact content analysis
2. SemanticTagger - Generates semantic tags for discovery
3. InformationExtractor - Extracts statistics and structure
4. ArtifactValidator - Validates artifact integrity
5. RegistrationOrchestrator - Coordinates the registration pipeline

GENERIC DESIGN:
---------------
This module is COMPLETELY GENERIC - no domain-specific (SQL, custom domains, etc.)
logic is embedded. It works with ANY artifact type:
- DataFrames, HTML, JSON, Files, Predictions, Custom objects
- Marketing content, code, reports, visualizations, etc.

INFORMATION THEORY (Shannon):
-----------------------------
Each artifact is characterized by:
- Entropy: Measure of information content
- Structure: Schema and type information
- Semantic Density: Tags per unit of content
- Provenance: Where the artifact came from
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import DSPy for LLM-based analysis
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using heuristic analysis")


# ============================================================================
# DATA STRUCTURES (Von Neumann: Clean, typed, extensible)
# ============================================================================


class ArtifactType(Enum):
    """Universal artifact types - NOT domain specific."""

    DATAFRAME = "dataframe"
    DICT = "dict"
    LIST = "list"
    STRING = "string"
    HTML = "html"
    JSON = "json"
    FILE = "file"
    PREDICTION = "prediction"  # DSPy Prediction
    CUSTOM = "custom"


@dataclass
class AnalysisResult:
    """Result of LLM-based artifact analysis."""

    artifact_type: ArtifactType
    semantic_description: str = ""
    purpose: str = ""
    key_fields: List[str] = field(default_factory=list)
    structure_summary: str = ""
    confidence: float = 0.0
    analysis_time: float = 0.0


@dataclass
class TaggingResult:
    """Result of semantic tagging."""

    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result of information extraction."""

    statistics: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, str] = field(default_factory=dict)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    size_bytes: int = 0
    preview: str = ""
    entropy_estimate: float = 0.0  # Shannon's information content


@dataclass
class DiscoveryValidationResult:
    """Result of artifact validation."""

    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class RegistrationResult:
    """Complete result of artifact registration."""

    artifact_id: str
    success: bool = True
    analysis: Optional[AnalysisResult] = None
    tagging: Optional[TaggingResult] = None
    extraction: Optional[ExtractionResult] = None
    validation: Optional[DiscoveryValidationResult] = None
    registration_time: float = 0.0
    error: Optional[str] = None


# ============================================================================
# ARTIFACT ANALYZER (Aristotle: Sound reasoning about content)
# ============================================================================


class ArtifactAnalyzer:
    """
    LLM-based artifact content analyzer.

    Uses DSPy to understand what an artifact contains and its purpose.
    Falls back to heuristic analysis if DSPy unavailable.
    """

    def __init__(self, lm: Optional[Any] = None) -> None:
        self.lm = lm

        if DSPY_AVAILABLE:
            # DSPy signature for artifact analysis
            class AnalyzeArtifactSignature(dspy.Signature):
                """Analyze a data artifact to understand its content and purpose."""

                artifact_preview: str = dspy.InputField(
                    desc="Preview of the artifact content (first 2000 chars)"
                )
                artifact_type: str = dspy.InputField(
                    desc="Detected type of artifact (dataframe, dict, list, etc.)"
                )
                context: str = dspy.InputField(desc="Context about where this artifact came from")

                description: str = dspy.OutputField(
                    desc="Clear, concise description of what this artifact contains"
                )
                purpose: str = dspy.OutputField(
                    desc="The likely purpose or use case for this artifact"
                )
                key_fields: str = dspy.OutputField(
                    desc="Comma-separated list of key fields/attributes"
                )
                structure: str = dspy.OutputField(
                    desc="Brief description of the artifact's structure"
                )

            self.analyzer = dspy.ChainOfThought(AnalyzeArtifactSignature)
        else:
            self.analyzer = None

        logger.info(" ArtifactAnalyzer initialized")

    def analyze(
        self, artifact: Any, artifact_type: ArtifactType, context: str = ""
    ) -> AnalysisResult:
        """Analyze an artifact to extract semantic information."""
        start_time = time.time()

        # Generate preview
        preview = self._get_preview(artifact, artifact_type)

        if self.analyzer is not None:
            try:
                result = self.analyzer(
                    artifact_preview=preview[:2000],
                    artifact_type=artifact_type.value,
                    context=context or "No additional context",
                )

                return AnalysisResult(
                    artifact_type=artifact_type,
                    semantic_description=result.description,
                    purpose=result.purpose,
                    key_fields=[f.strip() for f in result.key_fields.split(",") if f.strip()],
                    structure_summary=result.structure,
                    confidence=0.9,
                    analysis_time=time.time() - start_time,
                )
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}, using heuristic")

        # Fallback heuristic analysis
        return self._heuristic_analyze(artifact, artifact_type, preview, start_time)

    def _get_preview(self, artifact: Any, artifact_type: ArtifactType) -> str:
        """Generate a preview of the artifact."""
        try:
            if artifact_type == ArtifactType.DATAFRAME:
                if hasattr(artifact, "head"):
                    return str(artifact.head(10))
                return str(artifact)[:2000]
            elif artifact_type == ArtifactType.DICT:
                return json.dumps(artifact, indent=2, default=str)[:2000]
            elif artifact_type == ArtifactType.LIST:
                return json.dumps(artifact[:10] if len(artifact) > 10 else artifact, default=str)[
                    :2000
                ]
            elif artifact_type == ArtifactType.PREDICTION:
                if hasattr(artifact, "_store"):
                    return json.dumps(dict(artifact._store), indent=2, default=str)[:2000]
                return str(artifact)[:2000]
            else:
                return str(artifact)[:2000]
        except Exception as e:
            return f"<Preview unavailable: {e}>"

    def _heuristic_analyze(
        self, artifact: Any, artifact_type: ArtifactType, preview: str, start_time: float
    ) -> AnalysisResult:
        """Fallback heuristic analysis when LLM unavailable."""
        key_fields = []

        if artifact_type == ArtifactType.DICT and isinstance(artifact, dict):
            key_fields = list(artifact.keys())[:10]
        elif artifact_type == ArtifactType.DATAFRAME and hasattr(artifact, "columns"):
            key_fields = list(artifact.columns)[:10]
        elif artifact_type == ArtifactType.PREDICTION and hasattr(artifact, "_store"):
            key_fields = list(artifact._store.keys())[:10]

        return AnalysisResult(
            artifact_type=artifact_type,
            semantic_description=f"A {artifact_type.value} artifact",
            purpose="General data storage",
            key_fields=key_fields,
            structure_summary=f"Type: {artifact_type.value}, Fields: {len(key_fields)}",
            confidence=0.5,
            analysis_time=time.time() - start_time,
        )


# ============================================================================
# SEMANTIC TAGGER (Shannon: Information tagging for retrieval)
# ============================================================================


class SemanticTagger:
    """
    Generates semantic tags for artifact discovery.

    Tags are designed to maximize retrievability:
    - Content-based tags (what the artifact contains)
    - Purpose-based tags (what it's used for)
    - Structure-based tags (its format/type)
    """

    def __init__(self, lm: Optional[Any] = None) -> None:
        self.lm = lm

        if DSPY_AVAILABLE:

            class GenerateTagsSignature(dspy.Signature):
                """Generate semantic tags for artifact discovery."""

                description: str = dspy.InputField(desc="Description of the artifact")
                key_fields: str = dspy.InputField(desc="Key fields in the artifact")
                artifact_type: str = dspy.InputField(desc="Type of artifact")

                tags: str = dspy.OutputField(
                    desc="Comma-separated semantic tags for discovery (5-10 tags)"
                )
                categories: str = dspy.OutputField(
                    desc="Comma-separated category classifications (2-3 categories)"
                )

            self.tagger = dspy.ChainOfThought(GenerateTagsSignature)
        else:
            self.tagger = None

        logger.info(" SemanticTagger initialized")

    def tag(self, analysis: AnalysisResult) -> TaggingResult:
        """Generate tags from analysis result."""
        if self.tagger is not None:
            try:
                result = self.tagger(
                    description=analysis.semantic_description,
                    key_fields=", ".join(analysis.key_fields),
                    artifact_type=analysis.artifact_type.value,
                )

                tags = [t.strip() for t in result.tags.split(",") if t.strip()]
                categories = [c.strip() for c in result.categories.split(",") if c.strip()]

                return TaggingResult(
                    tags=tags,
                    categories=categories,
                    relevance_scores={tag: 0.8 for tag in tags},
                    confidence=0.85,
                )
            except Exception as e:
                logger.warning(f"LLM tagging failed: {e}, using heuristic")

        # Fallback heuristic tagging
        tags = [analysis.artifact_type.value, *analysis.key_fields[:5]]

        return TaggingResult(
            tags=tags,
            categories=[analysis.artifact_type.value],
            relevance_scores={tag: 0.5 for tag in tags},
            confidence=0.5,
        )


# ============================================================================
# INFORMATION EXTRACTOR (Shannon: Quantify information content)
# ============================================================================


class InformationExtractor:
    """
    Extracts quantitative information from artifacts.

    Shannon-inspired: Measures information content and structure.
    """

    def __init__(self) -> None:
        logger.info(" InformationExtractor initialized")

    def extract(self, artifact: Any, artifact_type: ArtifactType) -> ExtractionResult:
        """Extract statistics and schema from artifact."""
        result = ExtractionResult()

        try:
            # Size estimation
            result.size_bytes = len(str(artifact).encode("utf-8"))

            # Preview
            result.preview = str(artifact)[:200]

            # Type-specific extraction
            if artifact_type == ArtifactType.DATAFRAME:
                result = self._extract_dataframe(artifact, result)
            elif artifact_type == ArtifactType.DICT:
                result = self._extract_dict(artifact, result)
            elif artifact_type == ArtifactType.LIST:
                result = self._extract_list(artifact, result)
            elif artifact_type == ArtifactType.PREDICTION:
                result = self._extract_prediction(artifact, result)

            # Entropy estimate (simplified Shannon entropy)
            result.entropy_estimate = self._estimate_entropy(str(artifact)[:10000])

        except Exception as e:
            logger.warning(f"Extraction error: {e}")

        return result

    def _extract_dataframe(self, df: Any, result: ExtractionResult) -> ExtractionResult:
        """Extract DataFrame statistics."""
        if hasattr(df, "shape"):
            result.row_count = df.shape[0]
            result.column_count = df.shape[1]

        if hasattr(df, "columns"):
            result.schema = {col: str(df[col].dtype) for col in df.columns}

        if hasattr(df, "describe"):
            try:
                desc = df.describe().to_dict()
                result.statistics = {"summary": desc}
            except Exception:
                pass

        return result

    def _extract_dict(self, d: dict, result: ExtractionResult) -> ExtractionResult:
        """Extract dictionary statistics."""
        result.column_count = len(d)
        result.schema = {k: type(v).__name__ for k, v in list(d.items())[:50]}
        result.statistics = {"key_count": len(d), "nested_depth": self._get_nested_depth(d)}
        return result

    def _extract_list(self, lst: list, result: ExtractionResult) -> ExtractionResult:
        """Extract list statistics."""
        result.row_count = len(lst)
        if lst and isinstance(lst[0], dict):
            result.schema = {k: type(v).__name__ for k, v in lst[0].items()}
        result.statistics = {
            "length": len(lst),
            "element_type": type(lst[0]).__name__ if lst else "empty",
        }
        return result

    def _extract_prediction(self, pred: Any, result: ExtractionResult) -> ExtractionResult:
        """Extract DSPy Prediction statistics."""
        if hasattr(pred, "_store"):
            store = pred._store
            result.schema = {k: type(v).__name__ for k, v in store.items()}
            result.column_count = len(store)
            result.statistics = {"fields": list(store.keys())}
        return result

    def _get_nested_depth(self, d: Any, depth: int = 0) -> int:
        """Calculate nested depth of structure."""
        if not isinstance(d, dict) or depth > 10:
            return depth
        if not d:
            return depth
        return max(self._get_nested_depth(v, depth + 1) for v in d.values())

    def _estimate_entropy(self, text: str) -> float:
        """Estimate Shannon entropy of text."""
        if not text:
            return 0.0

        import math
        from collections import Counter

        freq = Counter(text)
        total = len(text)
        entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())

        return round(entropy, 3)


# ============================================================================
# ARTIFACT VALIDATOR (Gödel: Verify consistency and completeness)
# ============================================================================


class ArtifactValidator:
    """
    Validates artifact integrity before registration.

    Checks:
    - Non-empty content
    - Valid structure
    - Required fields present
    - No circular references
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        logger.info(" ArtifactValidator initialized")

    def validate(
        self,
        artifact: Any,
        artifact_type: ArtifactType,
        required_fields: Optional[List[str]] = None,
    ) -> DiscoveryValidationResult:
        """Validate an artifact."""
        issues = []
        warnings = []

        # Check non-empty
        if artifact is None:
            issues.append("Artifact is None")
            return DiscoveryValidationResult(is_valid=False, issues=issues, confidence=1.0)

        # Check empty content
        if self._is_empty(artifact, artifact_type):
            if self.strict:
                issues.append("Artifact is empty")
            else:
                warnings.append("Artifact appears to be empty")

        # Check required fields
        if required_fields:
            missing = self._check_required_fields(artifact, artifact_type, required_fields)
            if missing:
                issues.append(f"Missing required fields: {missing}")

        # Type-specific validation
        type_issues, type_warnings = self._validate_type(artifact, artifact_type)
        issues.extend(type_issues)
        warnings.extend(type_warnings)

        is_valid = len(issues) == 0
        confidence = 1.0 if is_valid else 0.5

        return DiscoveryValidationResult(
            is_valid=is_valid, issues=issues, warnings=warnings, confidence=confidence
        )

    def _is_empty(self, artifact: Any, artifact_type: ArtifactType) -> bool:
        """Check if artifact is empty."""
        if artifact_type == ArtifactType.DATAFRAME:
            return hasattr(artifact, "empty") and artifact.empty
        elif artifact_type == ArtifactType.DICT:
            return len(artifact) == 0
        elif artifact_type == ArtifactType.LIST:
            return len(artifact) == 0
        elif artifact_type == ArtifactType.STRING:
            return len(str(artifact).strip()) == 0
        return False

    def _check_required_fields(
        self, artifact: Any, artifact_type: ArtifactType, required: List[str]
    ) -> List[str]:
        """Check for required fields."""
        missing = []

        if artifact_type == ArtifactType.DICT and isinstance(artifact, dict):
            for field in required:
                if field not in artifact:
                    missing.append(field)
        elif artifact_type == ArtifactType.DATAFRAME and hasattr(artifact, "columns"):
            for field in required:
                if field not in artifact.columns:
                    missing.append(field)

        return missing

    def _validate_type(self, artifact: Any, artifact_type: ArtifactType) -> tuple:
        """Type-specific validation."""
        issues: list[Any] = []
        warnings = []

        # Add type-specific checks as needed
        if artifact_type == ArtifactType.PREDICTION:
            if not hasattr(artifact, "_store"):
                warnings.append("Prediction missing _store attribute")

        return issues, warnings


# ============================================================================
# REGISTRATION ORCHESTRATOR (Conductor: Coordinates the pipeline)
# ============================================================================


class RegistrationOrchestrator:
    """
    Orchestrates the complete artifact registration pipeline.

    Pipeline:
    1. Type Detection → Identify artifact type
    2. Analysis → LLM-based content analysis
    3. Tagging → Generate semantic tags
    4. Extraction → Extract statistics/schema
    5. Validation → Validate integrity
    6. Registration → Register with DataRegistry

    GENERIC DESIGN:
    - No domain-specific logic
    - Works with any artifact type
    - Pluggable components
    """

    def __init__(
        self,
        data_registry: Any = None,
        lm: Optional[Any] = None,
        model: Optional[str] = None,
        config: Any = None,
        enable_validation: bool = True,
        enable_llm_analysis: bool = True,
    ) -> None:
        """
        Initialize the RegistrationOrchestrator.

        Args:
            data_registry: DataRegistry instance for storing artifacts
            lm: DSPy language model (uses dspy.settings.lm if None)
            model: Model name for token counting
            config: Configuration object
            enable_validation: Whether to validate before registration
            enable_llm_analysis: Whether to use LLM for analysis (vs heuristic)
        """
        self.data_registry = data_registry
        self.lm = lm
        self.model = model
        self.config = config
        self.enable_validation = enable_validation
        self.enable_llm_analysis = enable_llm_analysis

        # Initialize components
        self.analyzer = ArtifactAnalyzer(lm) if enable_llm_analysis else None
        self.tagger = SemanticTagger(lm) if enable_llm_analysis else None
        self.extractor = InformationExtractor()
        self.validator = ArtifactValidator(strict=False)

        # Metrics
        self._registrations = 0
        self._failures = 0

        logger.info(" RegistrationOrchestrator initialized")
        logger.info(f"   LLM Analysis: {'enabled' if enable_llm_analysis else 'disabled'}")
        logger.info(f"   Validation: {'enabled' if enable_validation else 'disabled'}")

    def register(
        self,
        artifact: Any,
        artifact_id: str,
        source_actor: str,
        context: str = "",
        tags: Optional[List[str]] = None,
        skip_analysis: bool = False,
    ) -> RegistrationResult:
        """
        Register an artifact with the registry.

        Args:
            artifact: The artifact to register
            artifact_id: Unique identifier
            source_actor: Which actor produced this
            context: Additional context
            tags: Pre-defined tags (skips LLM tagging if provided)
            skip_analysis: Skip LLM analysis (use for performance)

        Returns:
            RegistrationResult with all metadata
        """
        start_time = time.time()

        try:
            # Step 1: Detect type
            artifact_type = self._detect_type(artifact)

            # Step 2: Validate (if enabled)
            validation = None
            if self.enable_validation:
                validation = self.validator.validate(artifact, artifact_type)
                if not validation.is_valid:
                    self._failures += 1
                    return RegistrationResult(
                        artifact_id=artifact_id,
                        success=False,
                        validation=validation,
                        error=f"Validation failed: {validation.issues}",
                        registration_time=time.time() - start_time,
                    )

            # Step 3: Extract information
            extraction = self.extractor.extract(artifact, artifact_type)

            # Step 4: Analyze (if enabled and not skipped)
            analysis = None
            if self.analyzer and not skip_analysis:
                analysis = self.analyzer.analyze(artifact, artifact_type, context)
            else:
                # Minimal analysis
                analysis = AnalysisResult(
                    artifact_type=artifact_type,
                    semantic_description=f"Artifact from {source_actor}",
                    purpose="Data storage",
                    key_fields=list(extraction.schema.keys())[:10],
                    confidence=0.5,
                )

            # Step 5: Generate tags
            tagging = None
            if tags:
                # Use provided tags
                tagging = TaggingResult(tags=tags, confidence=1.0)
            elif self.tagger and not skip_analysis:
                tagging = self.tagger.tag(analysis)
            else:
                # Minimal tagging
                tagging = TaggingResult(tags=[artifact_type.value, source_actor], confidence=0.5)

            # Step 6: Register with DataRegistry
            if self.data_registry:
                self._register_with_registry(
                    artifact_id=artifact_id,
                    artifact=artifact,
                    source_actor=source_actor,
                    analysis=analysis,
                    tagging=tagging,
                    extraction=extraction,
                )

            self._registrations += 1

            return RegistrationResult(
                artifact_id=artifact_id,
                success=True,
                analysis=analysis,
                tagging=tagging,
                extraction=extraction,
                validation=validation,
                registration_time=time.time() - start_time,
            )

        except Exception as e:
            self._failures += 1
            logger.error(f"Registration failed for {artifact_id}: {e}")
            return RegistrationResult(
                artifact_id=artifact_id,
                success=False,
                error=str(e),
                registration_time=time.time() - start_time,
            )

    def _detect_type(self, artifact: Any) -> ArtifactType:
        """Detect artifact type."""
        # Check for DataFrame
        if hasattr(artifact, "to_dict") and hasattr(artifact, "columns"):
            return ArtifactType.DATAFRAME

        # Check for DSPy Prediction
        if hasattr(artifact, "_store") and hasattr(artifact, "_completions"):
            return ArtifactType.PREDICTION

        # Check for HTML
        if isinstance(artifact, str):
            if artifact.strip().startswith("<") and artifact.strip().endswith(">"):
                return ArtifactType.HTML
            if artifact.strip().startswith("{") or artifact.strip().startswith("["):
                return ArtifactType.JSON
            return ArtifactType.STRING

        # Check for dict/list
        if isinstance(artifact, dict):
            return ArtifactType.DICT
        if isinstance(artifact, (list, tuple)):
            return ArtifactType.LIST

        return ArtifactType.CUSTOM

    def _register_with_registry(
        self,
        artifact_id: str,
        artifact: Any,
        source_actor: str,
        analysis: AnalysisResult,
        tagging: TaggingResult,
        extraction: ExtractionResult,
    ) -> Any:
        """Register artifact with DataRegistry."""
        try:
            # Import DataArtifact from parent package
            from ..data_registry import DataArtifact

            data_artifact = DataArtifact(
                id=artifact_id,
                name=artifact_id,
                source_actor=source_actor,
                data=artifact,
                data_type=analysis.artifact_type.value,
                schema=extraction.schema,
                tags=tagging.tags,
                description=analysis.semantic_description,
                timestamp=time.time(),
                size=extraction.size_bytes,
                preview=extraction.preview,
                semantic_description=analysis.semantic_description,
                semantic_tags=tagging.tags,
                purpose=analysis.purpose,
                confidence=analysis.confidence,
                validated=True,
                validation_confidence=1.0,
                statistics=extraction.statistics,
            )

            self.data_registry.register(data_artifact)

        except ImportError:
            # Fallback if DataArtifact not available
            logger.debug(f"Registered {artifact_id} (DataArtifact not imported)")
        except Exception as e:
            logger.warning(f"Registry registration failed: {e}")

    async def register_output(self, actor_name: str, output: Any, context: str = "") -> List[str]:
        """
        Async wrapper for registering actor output.

        This is the method called by Conductor for artifact registration.

        Args:
            actor_name: Name of the actor that produced the output
            output: The output to register
            context: Additional context

        Returns:
            List of artifact IDs that were registered
        """
        artifact_id = f"{actor_name}_output_{int(time.time())}"

        result = self.register(
            artifact=output, artifact_id=artifact_id, source_actor=actor_name, context=context
        )

        if result.success:
            return [result.artifact_id]
        else:
            logger.warning(f"Registration failed for {actor_name}: {result.error}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get registration statistics."""
        return {
            "total_registrations": self._registrations,
            "failures": self._failures,
            "success_rate": self._registrations / max(1, self._registrations + self._failures),
        }
