"""
SwarmCodeGenerator - Glue Code and Integration Code Generation

Generates code to connect tools and handle integrations.
Includes provider adapter generation for auto-discovered packages.
Follows DRY: Reuses existing code templates and patterns.
"""
import logging
import re
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .swarm_researcher import ProviderCandidate

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Generated code result."""
    code: str
    language: str
    dependencies: List[str]
    description: str
    usage_example: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmCodeGenerator:
    """
    Generates glue code and integration code.
    
    DRY Principle: Reuses existing code templates and patterns from skills.
    """
    
    def __init__(self, config=None):
        """
        Initialize SwarmCodeGenerator.
        
        Args:
            config: Optional SwarmConfig
        """
        self.config = config
        self._planner = None
    
    def _init_dependencies(self):
        """Lazy load dependencies (DRY: avoid circular imports)."""
        if self._planner is None:
            from Jotty.core.agents.agentic_planner import TaskPlanner
            self._planner = TaskPlanner()
    
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
        
        logger.info(f" SwarmCodeGenerator: Generating glue code {source_tool} → {destination_tool}")
        
        # Use LLM to generate code (DRY: reuse TaskPlanner's LLM)
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
                            logger.warning(f" Code generation timed out: {e}, using template")
                        else:
                            logger.warning(f" Code generation LLM call failed: {e}, using template")
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
                        logger.warning(f" Code generation timed out: {e}, using template")
                    else:
                        logger.warning(f" Code generation LLM call failed: {e}, using template")
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
        
        logger.info(f" SwarmCodeGenerator: Generating {operation} code for {service}")
        
        # Use LLM to generate integration code (DRY: reuse TaskPlanner)
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
                    logger.warning(" Integration code generation timed out, using template")
                    raise
                except Exception as e:
                    logger.warning(f" Integration code generation LLM call failed: {e}, using template")
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
        
        logger.info(f" Connected {source_tool} to {destination_tool}")
        return result
        
    except Exception as e:
        logger.error(f" Connection failed: {{e}}")
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
            logger.error(f" {{operation}} failed: {{e}}")
            raise
'''

        return GeneratedCode(
            code=code.strip(),
            language="python",
            dependencies=[],
            description=f"{operation} integration for {service}",
            usage_example=None
        )

    # =========================================================================
    # Provider Adapter Generation
    # =========================================================================

    def generate_provider_adapter(
        self,
        package_name: str,
        package_info: Dict[str, Any],
        categories: List[str]
    ) -> GeneratedCode:
        """
        Generate a SkillProvider adapter for an external package.

        Creates a wrapper that integrates the package into Jotty's provider system.

        Args:
            package_name: Name of the package (e.g., "pytesseract")
            package_info: Package metadata (from PyPI, GitHub, etc.)
            categories: List of SkillCategory values this provider supports

        Returns:
            GeneratedCode with the provider adapter

        Example:
            code = generator.generate_provider_adapter(
                "pytesseract",
                {"description": "OCR tool", "version": "0.3.10"},
                ["document"]
            )
        """
        self._init_dependencies()

        logger.info(f" Generating provider adapter for: {package_name}")

        # Try LLM-based generation first
        try:
            return self._generate_provider_with_llm(package_name, package_info, categories)
        except Exception as e:
            logger.debug(f"LLM generation failed: {e}, using template")

        # Fallback to template
        return self._generate_provider_template(package_name, package_info, categories)

    def _generate_provider_with_llm(
        self,
        package_name: str,
        package_info: Dict[str, Any],
        categories: List[str]
    ) -> GeneratedCode:
        """Generate provider adapter using LLM."""
        import dspy

        if not (hasattr(dspy, 'settings') and dspy.settings.lm):
            raise Exception("No LM available")

        # Build prompt
        categories_str = ', '.join(categories)
        prompt = f"""
Generate a Python SkillProvider class that wraps the '{package_name}' package.

Package info:
- Name: {package_name}
- Description: {package_info.get('description', 'N/A')}
- Version: {package_info.get('version', 'N/A')}

Requirements:
1. Create a class that inherits from SkillProvider
2. Set name = "{package_name.replace('-', '_')}_provider"
3. Set categories to: [{categories_str}]
4. Implement initialize() - check package availability
5. Implement execute(task, context) - route to package functions
6. Add proper error handling and logging
7. Include example usage in docstring

The class should:
- Import the package
- Provide wrapper methods for common operations
- Handle package-specific errors gracefully

Generate complete, runnable Python code.
"""

        lm = dspy.settings.lm
        response = lm(prompt, timeout=45)

        if isinstance(response, list):
            response = ' '.join(str(r) for r in response)
        elif not isinstance(response, str):
            response = str(response)

        code = self._extract_code_from_response(response)
        dependencies = self._extract_dependencies(code)
        dependencies.append(package_name)

        return GeneratedCode(
            code=code,
            language="python",
            dependencies=list(set(dependencies)),
            description=f"SkillProvider adapter for {package_name}",
            file_path=f"providers/{package_name.replace('-', '_')}_provider.py",
            usage_example=self._generate_provider_usage(package_name, categories),
            metadata={
                'package_name': package_name,
                'categories': categories,
                'generated_by': 'llm',
            }
        )

    def _generate_provider_template(
        self,
        package_name: str,
        package_info: Dict[str, Any],
        categories: List[str]
    ) -> GeneratedCode:
        """Generate provider adapter using template (fallback)."""
        # Normalize names
        class_name = ''.join(word.title() for word in package_name.replace('-', '_').split('_'))
        provider_name = package_name.replace('-', '_')
        categories_enum = ', '.join(f"SkillCategory.{c.upper()}" for c in categories)

        code = f'''"""
SkillProvider adapter for {package_name}
========================================

Auto-generated adapter to integrate {package_name} into Jotty's provider system.

Package: {package_name}
Description: {package_info.get('description', 'N/A')}
Categories: {', '.join(categories)}
"""

import time
import logging
from typing import Any, Dict, List, Optional

from core.skills.providers.base import (
    SkillProvider,
    SkillCategory,
    ProviderCapability,
    ProviderResult,
)

logger = logging.getLogger(__name__)


class {class_name}Provider(SkillProvider):
    """
    SkillProvider adapter for {package_name}.

    This adapter wraps the {package_name} package and exposes its
    functionality through the standard SkillProvider interface.

    Usage:
        from providers.{provider_name}_provider import {class_name}Provider

        provider = {class_name}Provider()
        await provider.initialize()
        result = await provider.execute("your task here")
    """

    name = "{provider_name}"
    version = "{package_info.get('version', '1.0.0')}"
    description = "{package_info.get('description', f'{package_name} integration')}"

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize {class_name}Provider.

        Args:
            config: Provider-specific configuration
        """
        super().__init__(config)

        # Package reference (lazy loaded)
        self._package = None

        # Define capabilities
        self.capabilities = [
            ProviderCapability(
                category=cat,
                actions=["execute", "process"],
                estimated_latency_ms=1000,
            )
            for cat in [{categories_enum}]
        ]

    async def initialize(self) -> bool:
        """
        Initialize the provider by importing the package.

        Returns:
            True if package is available and initialized
        """
        try:
            import {package_name.replace('-', '_')} as pkg
            self._package = pkg
            self.is_initialized = True
            self.is_available = True
            logger.info(f" {{self.name}} provider initialized")
            return True
        except ImportError as e:
            logger.warning(f" {{self.name}} not available: {{e}}")
            self.is_initialized = True
            self.is_available = False
            return False
        except Exception as e:
            logger.error(f" {{self.name}} initialization failed: {{e}}")
            self.is_initialized = True
            self.is_available = False
            return False

    def get_categories(self) -> List[SkillCategory]:
        """Get supported categories."""
        return [{categories_enum}]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """
        Execute a task using {package_name}.

        Args:
            task: Natural language task description
            context: Additional context (files, URLs, etc.)

        Returns:
            ProviderResult with execution output
        """
        start_time = time.time()
        context = context or {{}}

        if not self.is_available or not self._package:
            return ProviderResult(
                success=False,
                output=None,
                error=f"{{self.name}} is not available",
                provider_name=self.name,
            )

        try:
            # TODO: Implement task routing to package functions
            # This is a placeholder - customize based on package API
            result = self._route_task(task, context)

            return ProviderResult(
                success=True,
                output=result,
                execution_time=time.time() - start_time,
                provider_name=self.name,
            )

        except Exception as e:
            logger.error(f"{{self.name}} execution error: {{e}}")
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
                retryable=True,
            )

    def _route_task(self, task: str, context: Dict[str, Any]) -> Any:
        """
        Route task to appropriate package function.

        Override this method to implement package-specific routing.
        """
        # Placeholder implementation
        # TODO: Add package-specific logic
        return {{
            "message": f"Executed via {{self.name}}",
            "task": task,
            "package": "{package_name}",
        }}


# Factory function for registration
def create_provider(config: Dict[str, Any] = None) -> {class_name}Provider:
    """Create and return a {class_name}Provider instance."""
    return {class_name}Provider(config)
'''

        return GeneratedCode(
            code=code.strip(),
            language="python",
            dependencies=[package_name],
            description=f"SkillProvider adapter for {package_name}",
            file_path=f"providers/{provider_name}_provider.py",
            usage_example=self._generate_provider_usage(package_name, categories),
            metadata={
                'package_name': package_name,
                'categories': categories,
                'class_name': f"{class_name}Provider",
                'generated_by': 'template',
            }
        )

    def _generate_provider_usage(self, package_name: str, categories: List[str]) -> str:
        """Generate usage example for provider."""
        class_name = ''.join(word.title() for word in package_name.replace('-', '_').split('_'))
        provider_name = package_name.replace('-', '_')

        return f'''
# Usage example:
from providers.{provider_name}_provider import {class_name}Provider

# Create and initialize provider
provider = {class_name}Provider()
await provider.initialize()

# Execute task
result = await provider.execute("process document.pdf")
if result.success:
    print(result.output)
else:
    print(f"Error: {{result.error}}")

# Register with ProviderRegistry
from core.skills.providers import ProviderRegistry
registry = ProviderRegistry()
registry.register(provider, trust_level="sandboxed")
'''

    # =========================================================================
    # Morph App Code Generation
    # =========================================================================

    def generate_morph_workflow(
        self,
        workflow_name: str,
        workflow_spec: Dict[str, Any],
        streaming: bool = True
    ) -> GeneratedCode:
        """
        Generate a Python workflow file for Morph.

        Creates a workflow with @morph.func decorator that can be called
        from MDX pages.

        Args:
            workflow_name: Name of the workflow function
            workflow_spec: Workflow specification
                - description: What the workflow does
                - inputs: List of input parameter names
                - outputs: List of output field names
                - body: Optional custom body code
            streaming: Whether to use streaming response

        Returns:
            GeneratedCode with Morph workflow

        Example:
            code = gen.generate_morph_workflow(
                "analyze_stock",
                {
                    "description": "Analyze stock data",
                    "inputs": ["ticker", "period"],
                    "outputs": ["chart", "summary"],
                },
                streaming=True
            )
        """
        description = workflow_spec.get('description', f'{workflow_name} workflow')
        inputs = workflow_spec.get('inputs', [])
        outputs = workflow_spec.get('outputs', ['result'])
        custom_body = workflow_spec.get('body')

        # Build imports
        imports = ['import morph', 'from morph import MorphGlobalContext']
        if streaming:
            imports.append('from morph_lib.stream import stream_chat')
        imports.append('from typing import Dict, Any')

        # Build parameter annotations
        params_str = ''
        param_docs = []
        if inputs:
            params = [f'{inp}: str = None' for inp in inputs]
            params_str = ', ' + ', '.join(params)
            param_docs = [f'        {inp}: Input parameter' for inp in inputs]

        # Build body
        if custom_body:
            body = custom_body
        elif streaming:
            body = self._generate_streaming_workflow_body(workflow_name, inputs, outputs)
        else:
            body = self._generate_sync_workflow_body(workflow_name, inputs, outputs)

        # Build decorators
        decorators = '@morph.func'
        if streaming:
            decorators = '@morph.func\n@morph.streaming'

        # Assemble code
        code = f'''"""
{description}

Auto-generated Morph workflow by Jotty V2 SwarmCodeGenerator.
"""
{chr(10).join(imports)}


{decorators}
def {workflow_name}(context: MorphGlobalContext{params_str}):
    """
    {description}

    Args:
        context: Morph global context with request data
{chr(10).join(param_docs) if param_docs else "        None"}

    {'Yields' if streaming else 'Returns'}:
        {'Streamed response chunks' if streaming else 'Result dictionary'}
    """
{body}
'''

        return GeneratedCode(
            code=code.strip(),
            language="python",
            dependencies=['morph-data'],
            description=f"Morph workflow: {workflow_name}",
            file_path=f"python/{workflow_name}.py",
            usage_example=self._generate_morph_workflow_usage(workflow_name, inputs),
            metadata={
                'workflow_name': workflow_name,
                'streaming': streaming,
                'inputs': inputs,
                'outputs': outputs,
            }
        )

    def _generate_streaming_workflow_body(
        self,
        workflow_name: str,
        inputs: List[str],
        outputs: List[str]
    ) -> str:
        """Generate streaming workflow body."""
        indent = '    '

        # Input extraction
        input_lines = []
        for inp in inputs:
            input_lines.append(f'{indent}{inp} = context.vars.get("{inp}")')

        # Build body
        body_lines = input_lines or [f'{indent}# Get input from context', f'{indent}user_input = context.vars.get("message", "")']

        body_lines.extend([
            '',
            f'{indent}# Stream response using Morph stream_chat',
            f'{indent}# TODO: Integrate with your LLM (OpenAI, Anthropic, etc.)',
            f'{indent}response = f"Processing: {{str(context.vars)}}"',
            f'{indent}for word in response.split():',
            f'{indent}    yield stream_chat(word + " ")',
        ])

        return '\n'.join(body_lines)

    def _generate_sync_workflow_body(
        self,
        workflow_name: str,
        inputs: List[str],
        outputs: List[str]
    ) -> str:
        """Generate synchronous workflow body."""
        indent = '    '

        # Input extraction
        input_lines = []
        for inp in inputs:
            input_lines.append(f'{indent}{inp} = context.vars.get("{inp}")')

        # Output construction
        output_fields = ', '.join([f'"{o}": None' for o in outputs])

        body_lines = input_lines or [f'{indent}# Get input from context']

        body_lines.extend([
            '',
            f'{indent}# Process and return result',
            f'{indent}result = {{{output_fields}}}',
            '',
            f'{indent}# TODO: Implement {workflow_name} logic',
            '',
            f'{indent}return result',
        ])

        return '\n'.join(body_lines)

    def _generate_morph_workflow_usage(self, workflow_name: str, inputs: List[str]) -> str:
        """Generate usage example for Morph workflow."""
        inputs_example = ', '.join([f'{inp}="value"' for inp in inputs]) if inputs else ''

        return f'''
# Usage in MDX page:
# <Chat postData="{workflow_name}" />

# Or programmatically:
from python.{workflow_name} import {workflow_name}
from morph import MorphGlobalContext

context = MorphGlobalContext()
context.vars = {{{', '.join([f'"{inp}": "value"' for inp in inputs]) if inputs else '"message": "Hello"'}}}

# For streaming workflows:
for chunk in {workflow_name}(context{', ' + inputs_example if inputs_example else ''}):
    print(chunk, end="")

# For sync workflows:
result = {workflow_name}(context{', ' + inputs_example if inputs_example else ''})
print(result)
'''

    def generate_morph_page(
        self,
        page_name: str,
        workflows: List[str],
        components: List[Dict[str, Any]],
        title: str = None,
        description: str = None
    ) -> GeneratedCode:
        """
        Generate an MDX page for Morph.

        Creates a page that connects to Python workflows using
        Morph's pre-built components.

        Args:
            page_name: Name of the page file (without extension)
            workflows: List of workflow names this page uses
            components: List of component specifications
                - type: Component type (chat, form, table, chart, input, button)
                - workflow: Workflow to connect (optional)
                - Additional component-specific props
            title: Page title (default: page_name)
            description: Page description

        Returns:
            GeneratedCode with MDX page

        Example:
            code = gen.generate_morph_page(
                "dashboard",
                workflows=["get_metrics", "analyze_data"],
                components=[
                    {"type": "chart", "workflow": "get_metrics", "chart_type": "line"},
                    {"type": "table", "workflow": "analyze_data"},
                ],
                title="Analytics Dashboard"
            )
        """
        title = title or page_name.replace('_', ' ').title()
        description = description or f"{title} - Auto-generated by Jotty V2"

        # Build component MDX
        component_mdx = []
        for comp in components:
            mdx = self._generate_component_mdx(comp, workflows)
            if mdx:
                component_mdx.append(mdx)

        # Build content section
        content = self._generate_page_content(title, components)

        # Assemble MDX
        mdx = f'''---
title: {title}
description: {description}
---

# {title}

{content}

{chr(10).join(component_mdx)}
'''

        return GeneratedCode(
            code=mdx.strip(),
            language="mdx",
            dependencies=['morph-data'],
            description=f"Morph page: {page_name}",
            file_path=f"pages/{page_name}.mdx",
            usage_example=f'# This page is served at: http://localhost:8080/{page_name}',
            metadata={
                'page_name': page_name,
                'workflows': workflows,
                'components': [c.get('type') for c in components],
            }
        )

    def _generate_component_mdx(
        self,
        component: Dict[str, Any],
        workflows: List[str]
    ) -> str:
        """Generate MDX for a single component."""
        comp_type = component.get('type', 'chat')
        workflow = component.get('workflow', workflows[0] if workflows else 'main')

        templates = {
            'chat': f'<Chat postData="{workflow}" height={{{{300}}}} />',
            'form': f'<Form postData="{workflow}" />',
            'table': f'<DataTable postData="{workflow}" />',
            'chart': f'<Chart postData="{workflow}" type="{component.get("chart_type", "line")}" />',
            'input': f'<Input name="{component.get("name", "input")}" label="{component.get("label", "Input")}" />',
            'button': f'<Button onClick={{{{() => run("{workflow}")}}}}>{component.get("children", "Submit")}</Button>',
            'markdown': f'<Markdown>{component.get("content", "")}</Markdown>',
        }

        template = templates.get(comp_type)
        if template:
            return template

        # Unknown component type - return as custom
        props = ' '.join([f'{k}="{v}"' for k, v in component.items() if k != 'type'])
        return f'<{comp_type.title()} {props} />'

    def _generate_page_content(self, title: str, components: List[Dict[str, Any]]) -> str:
        """Generate descriptive content for the page."""
        comp_types = [c.get('type') for c in components]

        sections = []
        if 'chat' in comp_types:
            sections.append("Start a conversation with the AI assistant below.")
        if 'form' in comp_types:
            sections.append("Fill out the form to submit your data.")
        if 'chart' in comp_types or 'table' in comp_types:
            sections.append("View your data visualizations and insights.")

        return '\n\n'.join(sections) if sections else f"Welcome to {title}!"

    def generate_provider_from_candidate(
        self,
        candidate: 'ProviderCandidate'
    ) -> GeneratedCode:
        """
        Generate provider adapter from a ProviderCandidate.

        Convenience method that extracts info from candidate.

        Args:
            candidate: ProviderCandidate from SwarmResearcher

        Returns:
            GeneratedCode with the provider adapter
        """
        package_info = {
            'description': candidate.description,
            'version': candidate.metadata.get('version', '1.0.0'),
            'url': candidate.url,
            'source': candidate.source,
        }

        categories = candidate.categories or ['api_calls']

        return self.generate_provider_adapter(
            candidate.package_name,
            package_info,
            categories
        )
