"""
Artifacts Module
================

Handles extraction and management of generative UI artifacts from LLM responses.
Supports code blocks, HTML, Mermaid diagrams, charts, and executable code.
"""

import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of artifacts that can be extracted from LLM responses."""
    CODE = "code"
    HTML = "html"
    MERMAID = "mermaid"
    CHART = "chart"
    SVG = "svg"
    JSON = "json"
    MARKDOWN = "markdown"
    LATEX = "latex"


@dataclass
class Artifact:
    """
    Represents an extractable artifact from LLM response.

    Artifacts are renderable content blocks that can be displayed
    in special UI components (code editors, diagram viewers, etc.)
    """
    artifact_id: str
    artifact_type: str
    content: str
    language: Optional[str] = None
    title: Optional[str] = None
    executable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary for JSON serialization."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "content": self.content,
            "language": self.language,
            "title": self.title,
            "executable": self.executable,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create artifact from dictionary."""
        return cls(
            artifact_id=data.get("artifact_id", str(uuid.uuid4())[:12]),
            artifact_type=data.get("artifact_type", "code"),
            content=data.get("content", ""),
            language=data.get("language"),
            title=data.get("title"),
            executable=data.get("executable", False),
            metadata=data.get("metadata", {})
        )


class ArtifactExtractor:
    """
    Extracts artifacts from LLM response text.

    Detects code blocks, HTML, Mermaid diagrams, and other
    structured content that can be rendered in special UI.
    """

    # Languages that are executable in browser or server
    EXECUTABLE_LANGUAGES = {
        "python", "py", "python3",
        "javascript", "js", "node",
        "typescript", "ts",
        "bash", "sh", "shell",
        "sql"
    }

    # Languages that render as diagrams
    DIAGRAM_LANGUAGES = {
        "mermaid", "mmd",
        "plantuml", "uml",
        "dot", "graphviz"
    }

    # Languages that render as markup
    MARKUP_LANGUAGES = {
        "html", "htm",
        "svg",
        "markdown", "md",
        "latex", "tex"
    }

    def __init__(self):
        self._code_block_pattern = re.compile(
            r'```(\w+)?\n([\s\S]*?)```',
            re.MULTILINE
        )
        self._inline_html_pattern = re.compile(
            r'<(!DOCTYPE|html|body|div|span|p|table|form)[^>]*>[\s\S]*?</\1>',
            re.IGNORECASE
        )

    def extract_artifacts(self, text: str) -> List[Artifact]:
        """
        Extract all artifacts from LLM response text.

        Args:
            text: Raw LLM response text

        Returns:
            List of extracted Artifact objects
        """
        artifacts = []

        # Extract code blocks
        for match in self._code_block_pattern.finditer(text):
            language = match.group(1) or ""
            content = match.group(2).strip()

            if not content:
                continue

            artifact = self._create_artifact_from_code_block(language, content)
            if artifact:
                artifacts.append(artifact)

        return artifacts

    def _create_artifact_from_code_block(
        self,
        language: str,
        content: str
    ) -> Optional[Artifact]:
        """Create an artifact from a code block."""
        language_lower = language.lower() if language else ""

        # Determine artifact type
        if language_lower in self.DIAGRAM_LANGUAGES:
            artifact_type = ArtifactType.MERMAID.value if language_lower in ("mermaid", "mmd") else "diagram"
            return Artifact(
                artifact_id=str(uuid.uuid4())[:12],
                artifact_type=artifact_type,
                content=content,
                language=language_lower,
                title=f"{language.capitalize()} Diagram",
                executable=False
            )

        if language_lower in ("html", "htm"):
            return Artifact(
                artifact_id=str(uuid.uuid4())[:12],
                artifact_type=ArtifactType.HTML.value,
                content=content,
                language="html",
                title="HTML Preview",
                executable=True  # Can be previewed in iframe
            )

        if language_lower == "svg":
            return Artifact(
                artifact_id=str(uuid.uuid4())[:12],
                artifact_type=ArtifactType.SVG.value,
                content=content,
                language="svg",
                title="SVG Image",
                executable=False
            )

        if language_lower == "json":
            return Artifact(
                artifact_id=str(uuid.uuid4())[:12],
                artifact_type=ArtifactType.JSON.value,
                content=content,
                language="json",
                title="JSON Data",
                executable=False
            )

        if language_lower in ("latex", "tex"):
            return Artifact(
                artifact_id=str(uuid.uuid4())[:12],
                artifact_type=ArtifactType.LATEX.value,
                content=content,
                language="latex",
                title="LaTeX Math",
                executable=False
            )

        # Default: code block
        executable = language_lower in self.EXECUTABLE_LANGUAGES
        return Artifact(
            artifact_id=str(uuid.uuid4())[:12],
            artifact_type=ArtifactType.CODE.value,
            content=content,
            language=language_lower or None,
            title=f"{language.capitalize()} Code" if language else "Code",
            executable=executable
        )

    def extract_with_positions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract artifacts with their positions in the text.

        Useful for rendering mixed content with artifacts inline.

        Returns:
            List of dicts with artifact and position info
        """
        results = []

        for match in self._code_block_pattern.finditer(text):
            language = match.group(1) or ""
            content = match.group(2).strip()

            if not content:
                continue

            artifact = self._create_artifact_from_code_block(language, content)
            if artifact:
                results.append({
                    "artifact": artifact.to_dict(),
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(0)
                })

        return results


class SearchResult:
    """
    Represents a web search result for citation display.
    """

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str = "",
        source: str = "",
        favicon: Optional[str] = None
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source or self._extract_domain(url)
        self.favicon = favicon

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return url

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "favicon": self.favicon
        }


class ToolCallResult:
    """
    Represents a tool call result for display in UI.
    """

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any = None,
        duration_ms: int = 0,
        success: bool = True,
        error: Optional[str] = None
    ):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.duration_ms = duration_ms
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error
        }


# Singleton extractor instance
_extractor = None

def get_artifact_extractor() -> ArtifactExtractor:
    """Get singleton artifact extractor."""
    global _extractor
    if _extractor is None:
        _extractor = ArtifactExtractor()
    return _extractor


def extract_artifacts(text: str) -> List[Dict[str, Any]]:
    """
    Convenience function to extract artifacts from text.

    Returns list of artifact dictionaries.
    """
    extractor = get_artifact_extractor()
    artifacts = extractor.extract_artifacts(text)
    return [a.to_dict() for a in artifacts]
