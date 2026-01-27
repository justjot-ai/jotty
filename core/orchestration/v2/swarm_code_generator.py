"""
SwarmCodeGenerator - Glue Code and Integration Code Generation

Generates code to connect tools and handle integrations.
Follows DRY: Reuses existing code templates and patterns.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Generated code result."""
    code: str
    language: str
    dependencies: List[str]
    description: str
    usage_example: Optional[str] = None


class SwarmCodeGenerator:
    """
    Generates glue code and integration code.
    
    DRY Principle: Reuses existing code templates and patterns from skills.
    """
    
    def __init__(self, config=None):
        """
        Initialize SwarmCodeGenerator.
        
        Args:
            config: Optional JottyConfig
        """
        self.config = config
        self._planner = None
    
    def _init_dependencies(self):
        """Lazy load dependencies (DRY: avoid circular imports)."""
        if self._planner is None:
            from ...agents.agentic_planner import AgenticPlanner
            self._planner = AgenticPlanner()
    
    def generate_glue_code(
        self,
        source_tool: str,
        destination_tool: str,
        transformation: Optional[str] = None
    ) -> GeneratedCode:
        """
        Generate glue code to connect two tools.
        
        Args:
            source_tool: Source tool name (e.g., "reddit_scraper")
            destination_tool: Destination tool name (e.g., "notion_client")
            transformation: Optional transformation description
            
        Returns:
            GeneratedCode
        """
        self._init_dependencies()
        
        logger.info(f"🔧 SwarmCodeGenerator: Generating glue code {source_tool} → {destination_tool}")
        
        # Use LLM to generate code (DRY: reuse AgenticPlanner's LLM)
        prompt = f"""
Generate Python code to connect:
- Source: {source_tool}
- Destination: {destination_tool}
- Transformation: {transformation or 'pass data through'}

Requirements:
1. Handle errors gracefully
2. Add logging
3. Include type hints
4. Add docstrings
5. Make it reusable

Generate complete, runnable code.
"""
        
        try:
            import dspy
            if hasattr(dspy, 'settings') and dspy.settings.lm:
                lm = dspy.settings.lm
                
                # Set timeout on LM if it supports it (30 seconds for code generation)
                original_timeout = None
                if hasattr(lm, 'timeout'):
                    original_timeout = lm.timeout
                    lm.timeout = 30
                elif hasattr(lm, '__call__'):
                    # Try to pass timeout via kwargs
                    try:
                        response = lm(prompt, timeout=30)
                        # Handle response
                        if isinstance(response, list):
                            response = ' '.join(str(r) for r in response)
                        elif not isinstance(response, str):
                            response = str(response)
                        
                        code = self._extract_code_from_response(response)
                        dependencies = self._extract_dependencies(code)
                        
                        return GeneratedCode(
                            code=code,
                            language="python",
                            dependencies=dependencies,
                            description=f"Glue code connecting {source_tool} to {destination_tool}",
                            usage_example=self._generate_usage_example(source_tool, destination_tool)
                        )
                    except Exception as e:
                        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                            logger.warning(f"⚠️  Code generation timed out: {e}, using template")
                        else:
                            logger.warning(f"⚠️  Code generation LLM call failed: {e}, using template")
                        raise
                
                # Use DSPy ChainOfThought with timeout
                try:
                    from dspy.predict import ChainOfThought
                    cot = ChainOfThought("input -> output")
                    cot.lm = lm
                    
                    result = cot(input=prompt)
                    
                    # Restore timeout
                    if original_timeout is not None:
                        lm.timeout = original_timeout
                    
                    # Extract response
                    response = result.output if hasattr(result, 'output') else str(result)
                    if isinstance(response, list):
                        response = ' '.join(str(r) for r in response)
                    
                    code = self._extract_code_from_response(response)
                    dependencies = self._extract_dependencies(code)
                    
                    return GeneratedCode(
                        code=code,
                        language="python",
                        dependencies=dependencies,
                        description=f"Glue code connecting {source_tool} to {destination_tool}",
                        usage_example=self._generate_usage_example(source_tool, destination_tool)
                    )
                except (TimeoutError, Exception) as e:
                    # Restore timeout
                    if original_timeout is not None:
                        lm.timeout = original_timeout
                    
                    if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                        logger.warning(f"⚠️  Code generation timed out: {e}, using template")
                    else:
                        logger.warning(f"⚠️  Code generation LLM call failed: {e}, using template")
                    raise
            else:
                # No LM available, use template
                raise Exception("No LM available for code generation")
        except (TimeoutError, Exception) as e:
            logger.debug(f"Code generation LLM failed: {e}, falling back to template")
        
        # Fallback to template-based generation
        return self._generate_from_template(source_tool, destination_tool, transformation)
    
    def generate_integration_code(
        self,
        service: str,
        operation: str,
        config: Optional[Dict[str, Any]] = None
    ) -> GeneratedCode:
        """
        Generate integration code for a service.
        
        Args:
            service: Service name (e.g., "reddit", "notion")
            operation: Operation (e.g., "scrape", "send", "store")
            config: Optional configuration
            
        Returns:
            GeneratedCode
        """
        self._init_dependencies()
        
        logger.info(f"🔧 SwarmCodeGenerator: Generating {operation} code for {service}")
        
        # Use LLM to generate integration code (DRY: reuse AgenticPlanner)
        prompt = f"""
Generate Python code for {service} integration:
- Operation: {operation}
- Config: {config or 'use environment variables'}

Requirements:
1. Use standard library or common packages
2. Handle authentication
3. Handle errors gracefully
4. Add logging
5. Include type hints
6. Add docstrings

Generate complete, runnable code.
"""
        
        try:
            import dspy
            if hasattr(dspy, 'settings') and dspy.settings.lm:
                # Try with shorter timeout
                try:
                    lm = dspy.settings.lm
                    if hasattr(lm, '__call__'):
                        response = lm(prompt, timeout=30)
                    else:
                        response = lm(prompt)
                    
                    # Handle list responses
                    if isinstance(response, list):
                        response = ' '.join(str(r) for r in response)
                    elif not isinstance(response, str):
                        response = str(response)
                    
                    code = self._extract_code_from_response(response)
                    dependencies = self._extract_dependencies(code)
                    
                    return GeneratedCode(
                        code=code,
                        language="python",
                        dependencies=dependencies,
                        description=f"{operation} integration for {service}",
                        usage_example=self._generate_usage_example(service, operation)
                    )
                except TimeoutError:
                    logger.warning("⚠️  Integration code generation timed out, using template")
                    raise
                except Exception as e:
                    logger.warning(f"⚠️  Integration code generation LLM call failed: {e}, using template")
                    raise
        except (TimeoutError, Exception) as e:
            logger.debug(f"Integration code generation LLM failed: {e}, falling back to template")
        
        # Fallback to template
        return self._generate_integration_template(service, operation, config)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code block from LLM response."""
        # Look for code blocks
        import re
        
        # Try to find Python code block
        code_block_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try generic code block
        code_block_pattern = r'```\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Return response as-is if no code blocks found
        return response.strip()
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import statements to determine dependencies."""
        import re
        
        dependencies = []
        
        # Find import statements
        import_pattern = r'^(?:from|import)\s+(\w+)'
        matches = re.findall(import_pattern, code, re.MULTILINE)
        dependencies.extend(matches)
        
        # Remove standard library imports
        stdlib = {'os', 'sys', 'json', 'logging', 'typing', 'dataclasses', 'pathlib', 'asyncio'}
        dependencies = [d for d in dependencies if d not in stdlib]
        
        return list(set(dependencies))  # Remove duplicates
    
    def _generate_usage_example(self, source: str, destination: str) -> str:
        """Generate usage example."""
        return f"""
# Usage example:
from {source.lower().replace(' ', '_')} import {source}
from {destination.lower().replace(' ', '_')} import {destination}

# Use the generated glue code
result = connect_{source}_to_{destination}(data)
"""
    
    def _generate_from_template(
        self,
        source_tool: str,
        destination_tool: str,
        transformation: Optional[str]
    ) -> GeneratedCode:
        """Fallback template-based generation."""
        code = f'''
"""
Glue code connecting {source_tool} to {destination_tool}
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def connect_{source_tool.replace("-", "_")}_to_{destination_tool.replace("-", "_")}(data: Any) -> Dict[str, Any]:
    """
    Connect {source_tool} to {destination_tool}.
    
    Args:
        data: Data from {source_tool}
        
    Returns:
        Transformed data for {destination_tool}
    """
    try:
        # TODO: Implement transformation
        # Transformation: {transformation or 'pass through'}
        
        # Placeholder transformation
        result = {{"data": data}}
        
        logger.info(f"✅ Connected {source_tool} to {destination_tool}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {{e}}")
        raise
'''
        
        return GeneratedCode(
            code=code.strip(),
            language="python",
            dependencies=[],
            description=f"Template glue code for {source_tool} → {destination_tool}",
            usage_example=None
        )
    
    def _generate_integration_template(
        self,
        service: str,
        operation: str,
        config: Optional[Dict[str, Any]]
    ) -> GeneratedCode:
        """Fallback template-based integration code."""
        code = f'''
"""
{operation.title()} integration for {service}
"""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class {service.title()}Client:
    """Client for {service} API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize {service} client.
        
        Args:
            config: Configuration dict or None to use environment variables
        """
        self.config = config or {{}}
        # Load from environment if not provided
        # TODO: Implement actual {service} API client
        
    def {operation}(self, *args, **kwargs) -> Any:
        """
        {operation.title()} operation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Operation result
        """
        try:
            # TODO: Implement {operation} for {service}
            logger.info(f"Executing {{operation}} on {{service}}")
            return {{"success": True}}
        except Exception as e:
            logger.error(f"❌ {{operation}} failed: {{e}}")
            raise
'''
        
        return GeneratedCode(
            code=code.strip(),
            language="python",
            dependencies=[],
            description=f"{operation} integration for {service}",
            usage_example=None
        )
