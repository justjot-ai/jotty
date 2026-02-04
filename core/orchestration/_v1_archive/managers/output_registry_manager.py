"""
OutputRegistryManager - Manages output type detection, schema extraction, and registration.

Extracted from conductor.py to improve maintainability.
Handles output detection, preview generation, tag generation, and registry management.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OutputRegistryManager:
    """
    Centralized output registry management.

    Responsibilities:
    - Output type detection and classification
    - Schema extraction from various output formats
    - Preview generation for debugging
    - Tag generation for semantic search
    - Output registration in Data Registry
    - Output retrieval from trajectory
    """

    def __init__(self, config, data_registry=None, registration_orchestrator=None):
        """
        Initialize output registry manager.

        Args:
            config: JottyConfig
            data_registry: Optional DataRegistry instance
            registration_orchestrator: Optional RegistrationOrchestrator instance
        """
        self.config = config
        self.data_registry = data_registry
        self.registration_orchestrator = registration_orchestrator
        self.registration_count = 0

        logger.info("📦 OutputRegistryManager initialized")

    def detect_output_type(self, output: Any) -> str:
        """
        Auto-detect output type.

        Args:
            output: Output data to detect type of

        Returns:
            Type string (dataframe, text, json, etc.)
        """
        if hasattr(output, 'to_dict'):  # DataFrame
            return 'dataframe'
        elif isinstance(output, str):
            if len(output) > 100:
                if '<html' in output[:100].lower():
                    return 'html'
                elif '#' in output[:100]:
                    return 'markdown'
            return 'text'
        elif isinstance(output, bytes):
            return 'binary'
        elif isinstance(output, dict):
            return 'json'
        elif hasattr(output, 'output'):  # EpisodeResult
            return 'episode_result'
        elif hasattr(output, '__dict__'):
            return 'prediction'
        return 'unknown'

    def extract_schema(self, output: Any) -> Dict[str, str]:
        """
        Extract schema from output.

        Args:
            output: Output data to extract schema from

        Returns:
            Dict mapping field names to types
        """
        schema = {}

        # Handle EpisodeResult
        if hasattr(output, 'output') and hasattr(output, 'success'):
            if output.output is not None:
                return self.extract_schema(output.output)
            return {}

        if hasattr(output, '__dict__'):
            for field_name, field_value in vars(output).items():
                if not field_name.startswith('_'):
                    schema[field_name] = type(field_value).__name__

        elif isinstance(output, dict):
            for key, value in output.items():
                schema[key] = type(value).__name__

        elif hasattr(output, 'columns'):  # DataFrame
            schema = {col: 'column' for col in output.columns}

        return schema

    def generate_preview(self, output: Any) -> str:
        """
        Generate preview of output for debugging.

        Args:
            output: Output data to preview

        Returns:
            Preview string (truncated)
        """
        try:
            if isinstance(output, str):
                return output[:200]
            elif hasattr(output, '__str__'):
                return str(output)[:200]
            elif hasattr(output, 'head'):  # DataFrame
                return str(output.head(3))[:200]
            return f"<{type(output).__name__}>"
        except (AttributeError, TypeError, ValueError, Exception) as e:
            logger.debug(f"Preview generation failed: {e}")
            return "<preview unavailable>"

    def generate_tags(self, actor_name: str, output: Any, output_type: str) -> List[str]:
        """
        Generate semantic tags for output.

        Args:
            actor_name: Name of the actor that produced output
            output: Output data
            output_type: Detected output type

        Returns:
            List of semantic tags
        """
        tags = [output_type, actor_name.lower()]

        # Handle EpisodeResult
        if hasattr(output, 'output') and hasattr(output, 'success'):
            if output.output is not None:
                return self.generate_tags(actor_name, output.output, output_type)
            return tags

        # Add field names as tags
        if hasattr(output, '__dict__'):
            field_names = [f for f in vars(output).keys() if not f.startswith('_')]
            tags.extend(field_names[:5])  # Top 5 fields

        elif isinstance(output, dict):
            tags.extend(list(output.keys())[:5])

        return tags

    def register_output(self, actor_name: str, output: Any):
        """
        Register output in Data Registry.

        Args:
            actor_name: Name of the actor that produced output
            output: Output data to register
        """
        if not self.data_registry:
            logger.debug("No data_registry available, skipping registration")
            return

        try:
            # Import DataArtifact here to avoid circular import
            from ...data.data_registry import DataArtifact

            # Detect type
            output_type = self.detect_output_type(output)

            # Extract schema
            schema = self.extract_schema(output)

            # Generate tags
            tags = self.generate_tags(actor_name, output, output_type)

            # Generate preview
            preview = self.generate_preview(output)

            # Calculate size
            try:
                size = len(str(output))
            except (TypeError, AttributeError):
                size = 0

            # Create artifact
            artifact = DataArtifact(
                id=f"{actor_name}_{int(time.time() * 1000)}",
                name=actor_name,
                source_actor=actor_name,
                data=output,
                data_type=output_type,
                schema=schema,
                tags=tags,
                description=f"Output from {actor_name}",
                timestamp=time.time(),
                depends_on=[],
                size=size,
                preview=preview
            )

            # Register
            self.data_registry.register(artifact)
            self.registration_count += 1

            logger.debug(f"✅ Registered output from {actor_name} ({output_type}, {size} chars)")

        except Exception as e:
            logger.warning(f"⚠️  Failed to register output in registry: {e}")

    def should_inject_registry_tool(self, actor_name: str, actor_signatures: Dict) -> bool:
        """
        Check if actor signature requests data_registry.

        Args:
            actor_name: Name of the actor
            actor_signatures: Dict of actor signatures

        Returns:
            True if data_registry should be injected
        """
        signature = actor_signatures.get(actor_name, {})
        return 'data_registry' in signature

    def get_actor_outputs(self, trajectory: List[Dict]) -> Dict[str, Any]:
        """
        Extract all actor outputs from trajectory.

        Args:
            trajectory: List of execution steps

        Returns:
            Dict mapping actor_name -> latest output
        """
        outputs = {}
        for step in trajectory:
            actor = step.get('actor')
            if actor and 'actor_output' in step:
                outputs[actor] = step['actor_output']
        return outputs

    def get_output_from_actor(
        self,
        actor_name: str,
        trajectory: List[Dict],
        field: Optional[str] = None
    ) -> Any:
        """
        Get specific output from an actor.

        Args:
            actor_name: Name of the actor
            trajectory: List of execution steps
            field: Optional field to extract from output dict

        Returns:
            Actor output or specific field value
        """
        # Search from most recent to oldest
        for step in reversed(trajectory):
            if step.get('actor') == actor_name and 'actor_output' in step:
                output = step['actor_output']
                if field and isinstance(output, dict):
                    return output.get(field)
                return output
        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get output registry statistics.

        Returns:
            Dict with registration metrics
        """
        return {
            "total_registrations": self.registration_count,
            "has_data_registry": self.data_registry is not None,
            "has_registration_orchestrator": self.registration_orchestrator is not None
        }

    def reset_stats(self):
        """Reset registration statistics."""
        self.registration_count = 0
        logger.debug("OutputRegistryManager stats reset")
