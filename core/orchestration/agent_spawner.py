"""
Agent Spawner Interface - Abstract agent spawning
Project-specific implementations via plugins
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from ..queue.task import Task


class AgentSpawner(ABC):
    """
    Abstract interface for spawning agents.
    
    Projects implement this interface to provide their own agent spawning logic.
    This allows Jotty's TaskOrchestrator to work with any project's agent setup.
    """
    
    @abstractmethod
    async def spawn(self, task: Task) -> Tuple[int, str]:
        """
        Spawn agent for task.
        
        Args:
            task: Task to execute
            
        Returns:
            Tuple of (process_id, log_file_path)
        """
        pass
    
    @abstractmethod
    async def get_agent_type(self, task: Task) -> str:
        """
        Get agent type for task.
        
        Args:
            task: Task to check
            
        Returns:
            Agent type string (e.g., 'claude', 'cursor', 'opencode')
        """
        pass
    
    @abstractmethod
    async def check_credentials(self, agent_type: str) -> bool:
        """
        Check if credentials are available for agent type.
        
        Args:
            agent_type: Agent type to check
            
        Returns:
            True if credentials available, False otherwise
        """
        pass
