"""
Data Registry - Agentic Data Discovery and Retrieval
=====================================================

Enables agents to autonomously discover, search, and retrieve data
artifacts produced by other agents in the swarm.

Features:
- Auto-registration of all actor outputs
- Type detection and schema extraction
- Semantic tagging and search
- Multi-index for fast lookup
- Non-breaking integration

A-Team Design: Enables true agentic autonomy!
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataArtifact:
    """Universal data artifact with rich metadata."""
    
    # Identity
    id: str  # Unique ID
    name: str  # Human-readable name
    source_actor: str  # Which actor produced it
    
    # Content
    data: Any  # The actual data (HTML, DataFrame, Prediction, etc.)
    data_type: str  # 'html', 'dataframe', 'prediction', 'file', 'json'
    
    # Metadata
    schema: Dict[str, Any] = field(default_factory=dict)  # Field names and types
    tags: List[str] = field(default_factory=list)  # Semantic tags for discovery
    description: str = ""  # Human-readable description
    
    # Provenance
    timestamp: float = 0.0
    depends_on: List[str] = field(default_factory=list)  # Dependencies on other artifacts
    
    # Access info
    size: int = 0  # Size in bytes
    preview: str = ""  # Preview of data (first 200 chars)
    
    # 🆕 AGENTIC DISCOVERY: Rich semantic metadata (from LLM analysis)
    semantic_description: str = ""  # LLM-generated semantic description
    semantic_tags: List[str] = field(default_factory=list)  # LLM-generated semantic tags
    purpose: str = ""  # Why artifact was created
    confidence: float = 1.0  # Confidence in classification
    
    # 🆕 AGENTIC DISCOVERY: Provenance tracking
    derived_from: List[str] = field(default_factory=list)  # Parent artifacts
    enables: List[str] = field(default_factory=list)  # What this artifact enables
    
    # 🆕 AGENTIC DISCOVERY: Validation
    validated: bool = False  # Was artifact validated?
    validation_confidence: float = 0.0  # Validation confidence
    validation_issues: List[str] = field(default_factory=list)  # Any validation issues
    
    # 🆕 AGENTIC DISCOVERY: Rich metadata
    statistics: Dict[str, Any] = field(default_factory=dict)  # Statistics from InformationExtractor
    
    def matches_query(self, query: Dict) -> float:
        """Score how well this artifact matches a query."""
        score = 0.0
        
        # Type match
        if 'type' in query and query['type'] == self.data_type:
            score += 0.5
        
        # Tag match
        if 'tags' in query:
            query_tags = query['tags'] if isinstance(query['tags'], list) else [query['tags']]
            matching_tags = set(query_tags) & set(self.tags)
            if query_tags:
                score += 0.3 * (len(matching_tags) / len(query_tags))
        
        # Single tag match
        if 'tag' in query and query['tag'] in self.tags:
            score += 0.3
        
        # Actor match
        if 'actor' in query and query['actor'] == self.source_actor:
            score += 0.4
        
        # Field match
        if 'fields' in query:
            query_fields = query['fields'] if isinstance(query['fields'], list) else [query['fields']]
            matching_fields = set(query_fields) & set(self.schema.keys())
            if query_fields:
                score += 0.2 * (len(matching_fields) / len(query_fields))
        
        return score


class DataRegistry:
    """
    Central registry of all data artifacts produced by agents.
    Enables agent self-discovery and retrieval.
    """
    
    def __init__(self) -> None:
        self.artifacts: Dict[str, DataArtifact] = {}
        self.index_by_type: Dict[str, List[str]] = {}
        self.index_by_tag: Dict[str, List[str]] = {}
        self.index_by_actor: Dict[str, List[str]] = {}
        
        logger.info(" DataRegistry initialized")
    
    def register(self, artifact: DataArtifact) -> None:
        """Register a new data artifact."""
        self.artifacts[artifact.id] = artifact
        
        # Update type index
        if artifact.data_type not in self.index_by_type:
            self.index_by_type[artifact.data_type] = []
        self.index_by_type[artifact.data_type].append(artifact.id)
        
        # Update tag indices
        for tag in artifact.tags:
            if tag not in self.index_by_tag:
                self.index_by_tag[tag] = []
            self.index_by_tag[tag].append(artifact.id)
        
        # Update actor index
        if artifact.source_actor not in self.index_by_actor:
            self.index_by_actor[artifact.source_actor] = []
        self.index_by_actor[artifact.source_actor].append(artifact.id)
        
        logger.info(
            f" Registered artifact '{artifact.name}' "
            f"(type={artifact.data_type}, tags={artifact.tags}, size={artifact.size})"
        )
    
    def discover(self) -> Dict[str, Any]:
        """Return overview of all available data."""
        return {
            'total_artifacts': len(self.artifacts),
            'types_available': list(self.index_by_type.keys()),
            'tags_available': list(self.index_by_tag.keys()),
            'actors': list(self.index_by_actor.keys()),
            'artifacts': [
                {
                    'id': a.id,
                    'name': a.name,
                    'type': a.data_type,
                    'source': a.source_actor,
                    'tags': a.tags,
                    'schema': list(a.schema.keys()),
                    'preview': a.preview
                }
                for a in self.artifacts.values()
            ]
        }
    
    def search(self, query: Dict) -> List[DataArtifact]:
        """Search for artifacts matching query."""
        candidates = []
        
        # Fast path: filter by type
        if 'type' in query:
            artifact_ids = self.index_by_type.get(query['type'], [])
            candidates = [self.artifacts[aid] for aid in artifact_ids]
        
        # Fast path: filter by tag
        elif 'tag' in query:
            artifact_ids = self.index_by_tag.get(query['tag'], [])
            candidates = [self.artifacts[aid] for aid in artifact_ids]
        
        # Fast path: filter by actor
        elif 'actor' in query:
            artifact_ids = self.index_by_actor.get(query['actor'], [])
            candidates = [self.artifacts[aid] for aid in artifact_ids]
        
        else:
            # Full search
            candidates = list(self.artifacts.values())
        
        # Score and rank
        scored = [(a, a.matches_query(query)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [a for a, score in scored if score > 0]
    
    def get(self, artifact_id: str) -> Optional[DataArtifact]:
        """Get artifact by ID."""
        return self.artifacts.get(artifact_id)
    
    def get_by_name(self, name: str) -> Optional[DataArtifact]:
        """Get artifact by name (returns most recent if multiple)."""
        matches = [a for a in self.artifacts.values() if a.name == name]
        if matches:
            # Return most recent
            return max(matches, key=lambda a: a.timestamp)
        return None
    
    def list_types(self) -> List[str]:
        """List all available data types."""
        return list(self.index_by_type.keys())
    
    def list_tags(self) -> List[str]:
        """List all available semantic tags."""
        return list(self.index_by_tag.keys())
    
    def list_actors(self) -> List[str]:
        """List all actors that have produced data."""
        return list(self.index_by_actor.keys())


class DataRegistryTool:
    """Tool that agents can use to discover and retrieve data."""
    
    def __init__(self, registry: DataRegistry) -> None:
        self.registry = registry
    
    def __call__(self, action: str, **kwargs: Any) -> Any:
        """
        Main entry point for agents.
        
        Actions:
        - 'discover': Get overview of available data
        - 'search': Search for specific data
        - 'get': Retrieve specific artifact data
        - 'list_types': List available data types
        - 'list_tags': List available semantic tags
        - 'list_actors': List actors that produced data
        - 'metadata': Get metadata about an artifact
        """
        if action == 'discover':
            return self.registry.discover()
        
        elif action == 'search':
            artifacts = self.registry.search(kwargs)
            # Return list of dicts with data
            return [
                {
                    'id': a.id,
                    'name': a.name,
                    'type': a.data_type,
                    'data': a.data,
                    'schema': a.schema,
                    'tags': a.tags
                }
                for a in artifacts
            ]
        
        elif action == 'get':
            artifact_id = kwargs.get('id')
            name = kwargs.get('name')
            
            if artifact_id:
                artifact = self.registry.get(artifact_id)
            elif name:
                artifact = self.registry.get_by_name(name)
            else:
                return None
            
            if artifact:
                return artifact.data
            return None
        
        elif action == 'list_types':
            return self.registry.list_types()
        
        elif action == 'list_tags':
            return self.registry.list_tags()
        
        elif action == 'list_actors':
            return self.registry.list_actors()
        
        elif action == 'metadata':
            artifact_id = kwargs.get('id')
            name = kwargs.get('name')
            
            if artifact_id:
                artifact = self.registry.get(artifact_id)
            elif name:
                artifact = self.registry.get_by_name(name)
            else:
                return None
            
            if artifact:
                return {
                    'id': artifact.id,
                    'name': artifact.name,
                    'type': artifact.data_type,
                    'schema': artifact.schema,
                    'tags': artifact.tags,
                    'description': artifact.description,
                    'size': artifact.size,
                    'source': artifact.source_actor,
                    'timestamp': artifact.timestamp,
                    'preview': artifact.preview
                }
            return None
        
        return None

