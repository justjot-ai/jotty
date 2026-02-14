#!/usr/bin/env python3
"""
Document Model for Jotty Content Generation
Simplified from JustJot.ai Document model
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class SectionType(Enum):
    """Types of sections in a document"""
    TEXT = "text"
    CODE = "code"
    MERMAID = "mermaid"
    MATH = "math"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class Section:
    """A section in a document"""
    type: SectionType
    content: str
    title: Optional[str] = None
    language: Optional[str] = None  # For code sections
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert section to markdown"""
        parts = []

        if self.title:
            parts.append(f"## {self.title}\n")

        if self.type == SectionType.TEXT:
            parts.append(self.content)

        elif self.type == SectionType.CODE:
            lang = self.language or "text"
            parts.append(f"```{lang}\n{self.content}\n```")

        elif self.type == SectionType.MERMAID:
            parts.append(f"```mermaid\n{self.content}\n```")

        elif self.type == SectionType.MATH:
            parts.append(f"$$\n{self.content}\n$$")

        elif self.type == SectionType.IMAGE:
            parts.append(f"![{self.title or 'Image'}]({self.content})")

        elif self.type == SectionType.TABLE:
            parts.append(self.content)

        return "\n".join(parts)


@dataclass
class Document:
    """
    Document model for content generation

    Used by PDF, HTML, and Markdown generators
    """

    # Core content
    title: str
    content: str = ""  # Flat markdown content
    sections: List[Section] = field(default_factory=list)  # Structured sections

    # Metadata
    author: Optional[str] = None
    topic: Optional[str] = None
    created: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source information
    source_type: str = "jotty"

    @property
    def full_content(self) -> str:
        """Get complete content (flat or from sections)"""
        if self.content:
            return self.content

        if self.sections:
            return self.sections_to_markdown()

        return ""

    def sections_to_markdown(self) -> str:
        """Convert sections to flat markdown"""
        parts = []

        for section in self.sections:
            parts.append(section.to_markdown())
            parts.append("\n")

        return "\n".join(parts)

    def add_section(self, section_type: SectionType, content: str, title: Optional[str] = None, language: Optional[str] = None, **metadata: Any) -> 'Document':
        """Add a section to the document"""
        section = Section(
            type=section_type,
            content=content,
            title=title,
            language=language,
            metadata=metadata
        )
        self.sections.append(section)
        return self

    def __repr__(self) -> str:
        section_info = f", {len(self.sections)} sections" if self.sections else ""
        content_len = len(self.content) if self.content else 0
        return f"Document('{self.title}', {content_len} chars{section_info})"
