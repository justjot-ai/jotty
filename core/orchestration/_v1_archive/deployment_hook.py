"""
Deployment Hook Interface - Generic deployment trigger
Project-specific implementations via callbacks
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..queue.task import Task


class DeploymentHook(ABC):
    """
    Abstract interface for deployment triggers.
    
    Projects implement this interface to provide their own deployment logic.
    This allows Jotty's TaskOrchestrator to trigger deployments when tasks complete.
    """
    
    @abstractmethod
    async def trigger(self, task: Task) -> bool:
        """
        Trigger deployment for completed task.
        
        Args:
            task: Completed task that should trigger deployment
            
        Returns:
            True if deployment successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def should_deploy(self, task: Task) -> bool:
        """
        Check if task should trigger deployment.
        
        Args:
            task: Task to check
            
        Returns:
            True if deployment should be triggered, False otherwise
        """
        pass
