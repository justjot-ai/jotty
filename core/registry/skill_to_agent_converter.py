"""
Skill to Agent Converter - Auto-create AgentConfig from Skills

Converts skills to agents automatically using LLM analysis.
No manual AgentConfig creation needed.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.agent_config import AgentConfig
from .skills_registry import SkillDefinition, get_skills_registry

logger = logging.getLogger(__name__)


# =============================================================================
# DSPy Signatures for Skill Analysis
# =============================================================================

class SkillAnalysisSignature(dspy.Signature):
    """Analyze skill capabilities and create agent signature."""
    skill_name: str = dspy.InputField(desc="Name of the skill")
    skill_description: str = dspy.InputField(desc="Description from SKILL.md")
    tool_names: str = dspy.InputField(desc="JSON list of available tool names")
    tool_descriptions: str = dspy.InputField(desc="JSON dict of tool_name: description")
    
    agent_name: str = dspy.OutputField(desc="Suggested agent name (e.g., 'WebSearchAgent')")
    agent_description: str = dspy.OutputField(desc="What this agent does")
    dspy_signature: str = dspy.OutputField(
        desc="DSPy signature definition. Format: 'input_field: type -> output_field: type'. "
        "Example: 'query: str -> results: str'"
    )
    capabilities: str = dspy.OutputField(desc="JSON list of agent capabilities")
    input_field_names: str = dspy.OutputField(desc="JSON list of input field names")
    output_field_names: str = dspy.OutputField(desc="JSON list of output field names")


class SkillToAgentConverter:
    """
    Converts skills to agents automatically.
    
    Uses LLM to:
    1. Analyze skill capabilities
    2. Generate DSPy signature
    3. Create DSPy module
    4. Create AgentConfig
    """
    
    def __init__(self) -> None:
        """Initialize converter."""
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy required for SkillToAgentConverter")
        
        self.analyzer = dspy.ChainOfThought(SkillAnalysisSignature)
        self._signature_cache: Dict[str, type] = {}  # Cache generated signatures
        
        logger.info(" SkillToAgentConverter initialized")
    
    async def convert_skill_to_agent(
        self,
        skill: SkillDefinition,
        agent_name: Optional[str] = None
    ) -> AgentConfig:
        """
        Convert skill to AgentConfig.
        
        Args:
            skill: SkillDefinition to convert
            agent_name: Optional custom agent name
            
        Returns:
            AgentConfig ready for Conductor
        """
        try:
            # Step 1: Analyze skill capabilities (LLM)
            analysis = await self._analyze_skill(skill)
            
            # Step 2: Generate DSPy signature class
            signature_class = self._create_signature_class(
                skill_name=skill.name,
                analysis=analysis
            )
            
            # Step 3: Create DSPy module that uses AutoAgent
            # Use SkillBasedAgent which wraps AutoAgent execution
            from ..agents.skill_based_agent import SkillBasedAgent
            dspy_module = SkillBasedAgent(
                skill_name=skill.name,
                tool_name=list(skill.tools.keys())[0] if skill.tools else None
            )
            
            # Step 4: Create AgentConfig
            agent_config = AgentConfig(
                name=agent_name or analysis['agent_name'],
                agent=dspy_module,
                architect_prompts=None,  # Can be added later
                auditor_prompts=None,    # Can be added later
                capabilities=json.loads(analysis.get('capabilities', '[]')),
                metadata={
                    'source': 'skill',
                    'skill_name': skill.name,
                    'skill_description': skill.description,
                    'tools': list(skill.tools.keys()),
                    'converted_at': str(Path(__file__).stat().st_mtime)
                },
                outputs=json.loads(analysis.get('output_fields', '[]')),
                provides=json.loads(analysis.get('input_fields', '[]')),
                is_executor=True  # Skills execute tools
            )
            
            logger.info(f" Converted skill '{skill.name}' to agent '{agent_config.name}'")
            return agent_config
            
        except Exception as e:
            logger.error(f"Failed to convert skill '{skill.name}' to agent: {e}", exc_info=True)
            raise
    
    async def create_agent_from_skill_name(
        self,
        skill_name: str,
        agent_name: Optional[str] = None
    ) -> AgentConfig:
        """
        Load skill and convert to agent.
        
        Args:
            skill_name: Name of skill to convert
            agent_name: Optional custom agent name
            
        Returns:
            AgentConfig
        """
        registry = get_skills_registry()
        registry.init()
        
        skill = registry.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")
        
        return await self.convert_skill_to_agent(skill, agent_name)
    
    async def convert_skills_to_agents(
        self,
        skills: List[SkillDefinition],
        agent_name_prefix: Optional[str] = None
    ) -> List[AgentConfig]:
        """
        Convert multiple skills to agents.
        
        Args:
            skills: List of skills to convert
            agent_name_prefix: Optional prefix for agent names
            
        Returns:
            List of AgentConfig
        """
        agents = []
        for skill in skills:
            try:
                agent_name = None
                if agent_name_prefix:
                    agent_name = f"{agent_name_prefix}_{skill.name.replace('-', '_')}"
                
                agent = await self.convert_skill_to_agent(skill, agent_name)
                agents.append(agent)
            except Exception as e:
                logger.warning(f"Skipping skill '{skill.name}': {e}")
                continue
        
        logger.info(f" Converted {len(agents)}/{len(skills)} skills to agents")
        return agents
    
    async def _analyze_skill(self, skill: SkillDefinition) -> Dict[str, Any]:
        """Analyze skill capabilities using LLM."""
        # Extract tool descriptions
        tool_names = list(skill.tools.keys())
        tool_descriptions = {}
        
        # Try to get tool docstrings
        for tool_name, tool_func in skill.tools.items():
            if hasattr(tool_func, '__doc__') and tool_func.__doc__:
                tool_descriptions[tool_name] = tool_func.__doc__.strip()[:200]
            else:
                tool_descriptions[tool_name] = f"Tool: {tool_name}"
        
        try:
            result = self.analyzer(
                skill_name=skill.name,
                skill_description=skill.description or f"Skill: {skill.name}",
                tool_names=json.dumps(tool_names),
                tool_descriptions=json.dumps(tool_descriptions)
            )
            
            # Parse results
            analysis = {
                'agent_name': result.agent_name or f"{skill.name.replace('-', '_').title()}Agent",
                'agent_description': result.agent_description or skill.description,
                'dspy_signature': result.dspy_signature,
                'capabilities': result.capabilities or '[]',
                'input_fields': result.input_field_names or '[]',
                'output_fields': result.output_field_names or '[]'
            }
            
            logger.debug(f"Analyzed skill '{skill.name}': {analysis['agent_name']}")
            return analysis
            
        except Exception as e:
            logger.warning(f"Skill analysis failed for '{skill.name}': {e}")
            # Fallback: create basic signature
            return self._create_fallback_analysis(skill)
    
    def _create_fallback_analysis(self, skill: SkillDefinition) -> Dict[str, Any]:
        """Create fallback analysis when LLM fails."""
        tool_names = list(skill.tools.keys())
        agent_name = f"{skill.name.replace('-', '_').title()}Agent"
        
        # Simple signature: task -> result
        signature = "task: str -> result: str"
        
        return {
            'agent_name': agent_name,
            'agent_description': skill.description or f"Agent for {skill.name}",
            'dspy_signature': signature,
            'capabilities': json.dumps([f"Execute {tn}" for tn in tool_names]),
            'input_fields': json.dumps(['task']),
            'output_fields': json.dumps(['result'])
        }
    
    def _create_signature_class(
        self,
        skill_name: str,
        analysis: Dict[str, Any]
    ) -> type:
        """Create DSPy signature class from analysis."""
        # Check cache
        cache_key = f"{skill_name}_{analysis['dspy_signature']}"
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]
        
        # Parse signature string
        signature_str = analysis['dspy_signature']
        
        # Simple parser for signature format: "input: type -> output: type"
        # Or more complex: "field1: type, field2: type -> output1: type, output2: type"
        try:
            if '->' in signature_str:
                input_part, output_part = signature_str.split('->', 1)
                
                # Parse input fields
                input_fields = self._parse_fields(input_part.strip())
                
                # Parse output fields
                output_fields = self._parse_fields(output_part.strip())
                
                # Create signature class dynamically
                signature_dict = {}
                signature_dict.update(input_fields)
                signature_dict.update(output_fields)
                
                # Create class
                signature_class = type(
                    f"{analysis['agent_name']}Signature",
                    (dspy.Signature,),
                    signature_dict
                )
                
                # Cache it
                self._signature_cache[cache_key] = signature_class
                
                return signature_class
            else:
                # Fallback: simple signature
                return self._create_simple_signature(analysis['agent_name'])
                
        except Exception as e:
            logger.warning(f"Failed to parse signature '{signature_str}': {e}")
            return self._create_simple_signature(analysis['agent_name'])
    
    def _parse_fields(self, fields_str: str) -> Dict[str, Any]:
        """Parse field definitions from string."""
        fields = {}
        
        # Split by comma
        parts = [p.strip() for p in fields_str.split(',')]
        
        for part in parts:
            if ':' in part:
                field_name, field_type = part.split(':', 1)
                field_name = field_name.strip()
                field_type = field_type.strip()
                
                # Map type strings to dspy fields
                if field_type.lower() in ['str', 'string']:
                    fields[field_name] = dspy.InputField(desc=f"{field_name}") if 'input' in field_name.lower() else dspy.OutputField(desc=f"{field_name}")
                elif field_type.lower() in ['int', 'integer']:
                    fields[field_name] = dspy.InputField(desc=f"{field_name}") if 'input' in field_name.lower() else dspy.OutputField(desc=f"{field_name}")
                elif field_type.lower() in ['list', 'array']:
                    fields[field_name] = dspy.InputField(desc=f"{field_name}") if 'input' in field_name.lower() else dspy.OutputField(desc=f"{field_name}")
                else:
                    # Default to string
                    fields[field_name] = dspy.InputField(desc=f"{field_name}") if 'input' in field_name.lower() else dspy.OutputField(desc=f"{field_name}")
        
        return fields
    
    def _create_simple_signature(self, agent_name: str) -> type:
        """Create a simple fallback signature."""
        class SimpleSignature(dspy.Signature):
            task: str = dspy.InputField(desc="Task to execute")
            result: str = dspy.OutputField(desc="Execution result")
        
        SimpleSignature.__name__ = f"{agent_name}Signature"
        return SimpleSignature


# =============================================================================
# Convenience Functions
# =============================================================================

def create_skill_converter() -> SkillToAgentConverter:
    """Create a new skill converter instance."""
    return SkillToAgentConverter()
