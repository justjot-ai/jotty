"""
Intent Parser - Thin layer for natural language → TaskGraph

Uses AgenticPlanner for task type inference (no hardcoded keyword matching).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Reuse TaskType from AutoAgent (avoid duplication)
from ..agents.auto_agent import TaskType
from ..agents.agentic_planner import AgenticPlanner


@dataclass
class TaskGraph:
    """
    Structured representation of user intent.
    
    Maps to AutoAgent's task type system for seamless integration.
    """
    task_type: TaskType  # Reuse from AutoAgent
    workflow: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    schedule: Optional[str] = None
    operations: List[str] = field(default_factory=list)
    integrations: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentParser:
    """
    Parses natural language requests into structured task graphs.
    
    DRY Principle: Thin layer that converts natural language to TaskGraph.
    Actual task execution reuses AutoAgent.
    """
    
    def __init__(self, planner: Optional[AgenticPlanner] = None, auto_agent=None):
        """
        Initialize intent parser.
        
        Args:
            planner: Optional AgenticPlanner instance (creates new if None)
            auto_agent: DEPRECATED - ignored, kept for backward compatibility
        """
        # Use AgenticPlanner for task type inference (no hardcoded logic)
        self.planner = planner or AgenticPlanner()
        
        if auto_agent is not None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "auto_agent parameter is deprecated. "
                "IntentParser now uses AgenticPlanner for task type inference."
            )
        
        # Pattern matching for extraction (still needed for other fields)
        self.patterns = {
            'schedule': [
                r'daily', r'weekly', r'monthly', r'every\s+\w+day',
                r'when\s+', r'whenever', r'automatically', r'auto'
            ],
            'data_pipeline': [
                r'scrape', r'extract', r'pipeline', r'flow',
                r'from\s+\w+\s+to\s+\w+', r'send\s+to', r'upload\s+to'
            ],
            'software_setup': [
                r'install', r'setup', r'configure', r'set\s+up',
                r'initialize', r'prepare'
            ],
            'integration': [
                r'integrate', r'connect', r'link', r'sync',
                r'integrate\s+\w+\s+with\s+\w+'
            ]
        }
    
    def parse(self, user_request: str) -> TaskGraph:
        """
        Parse natural language request into task graph.
        
        Uses AgenticPlanner for task type inference (semantic understanding).
        
        Args:
            user_request: Natural language request from user
            
        Returns:
            TaskGraph with structured intent
        """
        # Use AgenticPlanner for semantic task type inference (no keyword matching)
        task_type, reasoning, confidence = self.planner.infer_task_type(user_request)
        
        # Extract components
        schedule = self._extract_schedule(user_request)
        source, destination = self._extract_source_destination(user_request)
        operations = self._extract_operations(user_request)
        integrations = self._extract_integrations(user_request)
        requirements = self._extract_requirements(user_request)
        
        return TaskGraph(
            task_type=task_type,
            workflow=self._infer_workflow(task_type, operations),
            source=source,
            destination=destination,
            schedule=schedule,
            operations=operations,
            integrations=integrations,
            requirements=requirements,
            metadata={'original_request': user_request}
        )
    
    # Removed _infer_task_type_fallback() - now using AgenticPlanner for semantic inference
    # No hardcoded keyword matching - fully agentic
    
    def _extract_schedule(self, request: str) -> Optional[str]:
        """Extract scheduling information."""
        import re
        request_lower = request.lower()
        
        schedule_patterns = {
            'daily': r'daily|every\s+day',
            'weekly': r'weekly|every\s+week',
            'monthly': r'monthly|every\s+month',
            'hourly': r'hourly|every\s+hour',
        }
        
        for schedule, pattern in schedule_patterns.items():
            if re.search(pattern, request_lower):
                return schedule
        
        return None
    
    def _extract_source_destination(self, request: str) -> tuple[Optional[str], Optional[str]]:
        """Extract source and destination from request."""
        import re
        request_lower = request.lower()
        
        # Pattern: "from X to Y" or "X to Y"
        from_match = re.search(r'from\s+(\w+)', request_lower)
        to_match = re.search(r'to\s+(\w+)', request_lower)
        
        source = from_match.group(1) if from_match else None
        destination = to_match.group(1) if to_match else None
        
        # Also check for common services
        services = ['reddit', 'twitter', 'slack', 'notion', 'github', 'email', 'telegram']
        if not source:
            for service in services:
                if service in request_lower:
                    pos = request_lower.index(service)
                    if 'to' not in request_lower or pos < request_lower.index('to'):
                        source = service
                        break
        
        if not destination:
            for service in services:
                if service in request_lower:
                    pos = request_lower.index(service)
                    if 'to' in request_lower and pos > request_lower.index('to'):
                        destination = service
                        break
        
        return source, destination
    
    def _extract_operations(self, request: str) -> List[str]:
        """Extract operations from request."""
        operations = []
        request_lower = request.lower()
        
        operation_keywords = {
            'scrape': ['scrape', 'extract', 'fetch'],
            'summarize': ['summarize', 'summary'],
            'send': ['send', 'post', 'upload'],
            'analyze': ['analyze', 'analyse', 'examine'],
            'transform': ['transform', 'convert'],
            'store': ['store', 'save', 'archive'],
        }
        
        for op, keywords in operation_keywords.items():
            if any(kw in request_lower for kw in keywords):
                operations.append(op)
        
        return operations
    
    def _extract_integrations(self, request: str) -> List[str]:
        """Extract integration services mentioned."""
        services = ['reddit', 'twitter', 'slack', 'notion', 'github', 'email', 
                   'telegram', 'discord', 'google', 'aws', 'azure']
        
        integrations = []
        request_lower = request.lower()
        for service in services:
            if service in request_lower:
                integrations.append(service)
        
        return integrations
    
    def _extract_requirements(self, request: str) -> List[str]:
        """Extract specific requirements or constraints."""
        import re
        requirements = []
        
        requirement_patterns = [
            r'using\s+(\w+)',
            r'with\s+(\w+)',
            r'that\s+(\w+)',
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, request)
            requirements.extend(matches)
        
        return requirements
    
    def _infer_workflow(self, task_type: TaskType, operations: List[str]) -> Optional[str]:
        """Infer workflow type from task type and operations."""
        if task_type == TaskType.AUTOMATION:
            return 'automation_workflow'
        elif task_type == TaskType.RESEARCH:
            return 'research_workflow'
        elif operations:
            return f"{'_'.join(operations)}_workflow"
        else:
            return None
