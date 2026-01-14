"""
Expert Agent Registry

Registry for managing and accessing expert agents.
Provides easy access to pre-trained expert agents.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path

from .expert_agent import ExpertAgent, ExpertAgentConfig
from .mermaid_expert import MermaidExpertAgent
from .pipeline_expert import PipelineExpertAgent

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """
    Registry for expert agents.
    
    Provides centralized access to expert agents and manages their lifecycle.
    """
    
    def __init__(self):
        self._experts: Dict[str, ExpertAgent] = {}
        self._initialized = False
    
    def register(self, name: str, expert: ExpertAgent):
        """Register an expert agent."""
        self._experts[name] = expert
        logger.info(f"Registered expert agent: {name}")
    
    def get(self, name: str, auto_train: bool = False) -> Optional[ExpertAgent]:
        """
        Get an expert agent by name.
        
        Args:
            name: Expert agent name
            auto_train: If True, automatically train if not already trained
                       (Note: This requires async context. Use await get_async() in async code)
        
        Returns:
            ExpertAgent instance or None if not found
        """
        return self._experts.get(name)
    
    async def get_async(self, name: str, auto_train: bool = False) -> Optional[ExpertAgent]:
        """
        Get an expert agent by name (async version).
        
        Args:
            name: Expert agent name
            auto_train: If True, automatically train if not already trained
        
        Returns:
            ExpertAgent instance or None if not found
        """
        expert = self._experts.get(name)
        
        if expert and auto_train and not expert.trained:
            logger.info(f"Auto-training expert: {name}")
            await expert.train()
        
        return expert
    
    def list_experts(self) -> list:
        """List all registered expert agents."""
        return list(self._experts.keys())
    
    def get_mermaid_expert(self, auto_train: bool = False) -> MermaidExpertAgent:
        """Get the Mermaid expert agent (sync version - no auto-train)."""
        name = "mermaid"
        if name not in self._experts:
            self._experts[name] = MermaidExpertAgent()
        return self._experts[name]
    
    async def get_mermaid_expert_async(self, auto_train: bool = True) -> MermaidExpertAgent:
        """Get the Mermaid expert agent (async version with auto-train)."""
        name = "mermaid"
        if name not in self._experts:
            self._experts[name] = MermaidExpertAgent()
        
        expert = self._experts[name]
        
        if auto_train and not expert.trained:
            await expert.train()
        
        return expert
    
    def get_pipeline_expert(self, output_format: str = "mermaid", auto_train: bool = False) -> PipelineExpertAgent:
        """Get the Pipeline expert agent (sync version - no auto-train)."""
        name = f"pipeline_{output_format}"
        if name not in self._experts:
            self._experts[name] = PipelineExpertAgent(output_format=output_format)
        return self._experts[name]
    
    async def get_pipeline_expert_async(self, output_format: str = "mermaid", auto_train: bool = True) -> PipelineExpertAgent:
        """Get the Pipeline expert agent (async version with auto-train)."""
        name = f"pipeline_{output_format}"
        if name not in self._experts:
            self._experts[name] = PipelineExpertAgent(output_format=output_format)
        
        expert = self._experts[name]
        
        if auto_train and not expert.trained:
            await expert.train()
        
        return expert
    
    def ensure_trained(self, name: str):
        """Ensure an expert agent is trained."""
        expert = self.get(name)
        if expert and not expert.trained:
            import asyncio
            asyncio.run(expert.train())


# Global registry instance
_registry = ExpertRegistry()


def get_expert_registry() -> ExpertRegistry:
    """Get the global expert registry."""
    return _registry


def get_mermaid_expert(auto_train: bool = False) -> MermaidExpertAgent:
    """Convenience function to get Mermaid expert (sync - no auto-train)."""
    return _registry.get_mermaid_expert(auto_train=auto_train)


async def get_mermaid_expert_async(auto_train: bool = True) -> MermaidExpertAgent:
    """Convenience function to get Mermaid expert (async with auto-train)."""
    return await _registry.get_mermaid_expert_async(auto_train=auto_train)


def get_pipeline_expert(output_format: str = "mermaid", auto_train: bool = False) -> PipelineExpertAgent:
    """Convenience function to get Pipeline expert (sync - no auto-train)."""
    return _registry.get_pipeline_expert(output_format=output_format, auto_train=auto_train)


async def get_pipeline_expert_async(output_format: str = "mermaid", auto_train: bool = True) -> PipelineExpertAgent:
    """Convenience function to get Pipeline expert (async with auto-train)."""
    return await _registry.get_pipeline_expert_async(output_format=output_format, auto_train=auto_train)
