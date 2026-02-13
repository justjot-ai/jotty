"""
Skill Generator - AI-Powered Skill Creation

Clawd.bot can generate skills on-demand using AI. This module provides
the same capability for Jotty.

Key Features:
- Generate skills from natural language descriptions
- Auto-create SKILL.md and tools.py files
- Validate generated skills before registration
- Hot-reload generated skills

Uses Jotty's unified LLM interface (DSPy BaseLM) for generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json
import dspy
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


class SkillGenerator:
    """
    AI-powered skill generator for Jotty.
    
    Generates skills from natural language descriptions,
    similar to Clawd.bot's capability.
    """
    
    def __init__(self, skills_dir: Optional[str] = None, lm: Optional[BaseLM] = None, skills_registry=None):
        """
        Initialize skill generator.
        
        Args:
            skills_dir: Directory where skills are stored
            lm: DSPy BaseLM instance (uses Jotty's unified LLM interface)
                If None, uses dspy.configure() default LM
            skills_registry: Optional SkillsRegistry instance for auto-reload after generation
        """
        if skills_dir is None:
            # Priority: env var > repo-relative > user home
            skills_dir = os.getenv("JOTTY_SKILLS_DIR")
            
            if not skills_dir:
                # Try repo-relative (for development)
                # __file__ is core/registry/skill_generator.py
                # Go up: core/registry -> core -> Jotty -> skills
                current_file = Path(__file__).resolve()
                repo_root = current_file.parent.parent.parent  # core/registry -> core -> Jotty
                repo_skills = repo_root / "skills"
                if repo_skills.exists() or repo_root.name == "Jotty":
                    # Create if doesn't exist (we're in repo)
                    repo_skills.mkdir(exist_ok=True)
                    skills_dir = str(repo_skills)
                else:
                    # Fallback to user home (for installed packages)
                    home = os.path.expanduser("~")
                    skills_dir = os.path.join(home, "jotty", "skills")
        
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.skills_registry = skills_registry  # For auto-reload after generation
        
        # Use provided LM or get from DSPy configuration (same approach as supervisor)
        if lm:
            self.lm = lm
        else:
            # Try to get from DSPy configuration (may already be configured)
            try:
                self.lm = dspy.LM()
            except Exception:
                self.lm = None
            
            # If not configured, use UnifiedLMProvider (same as supervisor)
            if self.lm is None:
                try:
                    from ..foundation.unified_lm_provider import UnifiedLMProvider
                    self.lm = UnifiedLMProvider.configure_default_lm()
                    logger.info(" SkillGenerator using unified LLM provider (auto-detected)")
                except Exception as e:
                    logger.warning(f" Could not configure unified LM provider: {e}")
                    raise ValueError(
                        "No LLM available. Provide 'lm' parameter or ensure DSPy is configured with "
                        "UnifiedLMProvider.configure_default_lm()"
                    )
    
    def generate_skill(
        self,
        skill_name: str,
        description: str,
        requirements: Optional[str] = None,
        examples: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate a skill from natural language description.
        
        Args:
            skill_name: Name of the skill (e.g., "email-checker")
            description: What the skill should do (natural language)
            requirements: Optional requirements/constraints
            examples: Optional examples of expected behavior
            
        Returns:
            Dict with skill metadata and file paths
            
        Example:
            await generator.generate_skill(
                skill_name="todoist-automation",
                description="Automate Todoist tasks - create, complete, and list tasks",
                requirements="Use Todoist API, handle authentication",
                examples=["Create task 'Buy milk'", "List all tasks"]
            )
        """
        skill_dir = self.skills_dir / skill_name
        skill_dir.mkdir(exist_ok=True)
        
        # Generate SKILL.md
        skill_md_content = self._generate_skill_md(
            skill_name, description, requirements, examples
        )
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_md_content)
        
        # Generate tools.py
        tools_py_content = self._generate_tools_py(
            skill_name, description, requirements, examples
        )
        tools_py_path = skill_dir / "tools.py"
        tools_py_path.write_text(tools_py_content)
        
        logger.info(f" Generated skill: {skill_name}")
        
        # Auto-reload skill if registry is available
        reloaded = False
        tool_tested = False
        
        if self.skills_registry:
            try:
                # Reload skills to pick up new one
                self.skills_registry.load_all_skills()
                reloaded = True
                logger.info(f" Skill '{skill_name}' auto-reloaded")
                
                # Test tool execution if possible
                skill = self.skills_registry.get_skill(skill_name)
                if skill and skill.tools:
                    # Try to test first tool with empty params (safe test)
                    first_tool_name = list(skill.tools.keys())[0]
                    first_tool = skill.tools[first_tool_name]
                    
                    try:
                        # Test with empty params (tool should handle gracefully)
                        test_result = first_tool({})
                        tool_tested = True
                        if test_result.get('success'):
                            logger.info(f" Tool '{first_tool_name}' test passed")
                        else:
                            logger.info(f" Tool '{first_tool_name}' test returned: {test_result.get('error', 'unknown')}")
                    except Exception as e:
                        logger.warning(f" Tool '{first_tool_name}' test failed: {e}")
            except Exception as e:
                logger.warning(f" Auto-reload failed: {e}")
        else:
            logger.info(f" Skill '{skill_name}' generated. Pass skills_registry to generator for auto-reload.")
        
        return {
            "name": skill_name,
            "description": description,
            "path": str(skill_dir),
            "skill_md": str(skill_md_path),
            "tools_py": str(tools_py_path),
            "reloaded": reloaded,
            "tool_tested": tool_tested,
        }
    
    def _generate_skill_md(
        self,
        skill_name: str,
        description: str,
        requirements: Optional[str],
        examples: Optional[list]
    ) -> str:
        """Generate SKILL.md content using Jotty's unified LLM interface."""
        prompt = f"""Create SKILL.md for "{skill_name}" skill.

Description: {description}
{f'Requirements: {requirements}' if requirements else ''}

Output ONLY the markdown file content, no explanations:

# {skill_name}

## Description
{description}

## Tools
[List tools this skill provides]

## Usage
[Usage examples]

## Requirements
{requirements or 'No external dependencies'}"""
        
        # Use DSPy LM interface
        try:
            response = self.lm(prompt=prompt, timeout=180)
        except Exception as e:
            logger.warning(f"LLM call failed, using fallback: {e}")
            response = None
        
        # Extract text from DSPy response (Claude CLI returns list)
        if response is None:
            # Fallback
            text = f"""# {skill_name}

## Description
{description}

## Tools
- {skill_name.replace("-", "_")}_tool: {description}

## Usage
Use the {skill_name} tool to {description.lower()}.

## Requirements
{requirements or 'No external dependencies'}"""
        elif isinstance(response, list) and len(response) > 0:
            text = response[0]
        elif hasattr(response, 'completions') and response.completions:
            text = response.completions[0].text.strip()
        elif hasattr(response, 'text'):
            text = response.text.strip()
        elif isinstance(response, str):
            text = response.strip()
        else:
            text = str(response).strip()
        
        # Handle JSON-wrapped responses
        import json
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) > 0:
                text = parsed[0]
            elif isinstance(parsed, str):
                text = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Extract markdown if wrapped in explanations
        if "```markdown" in text:
            start = text.find("```markdown") + 11
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        return text.strip()
    
    def _generate_tools_py(
        self,
        skill_name: str,
        description: str,
        requirements: Optional[str],
        examples: Optional[list]
    ) -> str:
        """Generate tools.py content using Jotty's unified LLM interface."""
        # Direct prompt for code generation
        prompt = f"""Write Python code for tools.py file. Skill: "{skill_name}".

Description: {description}
{f'Requirements: {requirements}' if requirements else ''}

Create Python functions that:
- Accept params: dict parameter
- Return dict with success/error info
- Function names end with _tool

Output ONLY Python code, no explanations, no markdown:"""
        
        # Use DSPy LM interface with increased timeout for code generation
        try:
            response = self.lm(prompt=prompt, timeout=300)  # 5 minutes for code generation
        except Exception as e:
            logger.warning(f"LLM call failed, using fallback: {e}")
            response = None
        
        # Extract text from DSPy response (Claude CLI returns list)
        if response is None:
            # Fallback stub
            text = f'''"""
{skill_name} Skill Tools
"""

def {skill_name.replace("-", "_")}_tool(params: dict) -> dict:
    """{description}"""
    return {{"success": True, "message": "Tool stub - implement me"}}
'''
        elif isinstance(response, list) and len(response) > 0:
            text = response[0]
        elif hasattr(response, 'completions') and response.completions:
            text = response.completions[0].text.strip()
        elif hasattr(response, 'text'):
            text = response.text.strip()
        elif isinstance(response, str):
            text = response.strip()
        else:
            text = str(response).strip()
        
        # Handle JSON-wrapped responses
        import json
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) > 0:
                text = parsed[0]
            elif isinstance(parsed, str):
                text = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Extract Python code if wrapped in explanations or code blocks
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Remove leading/trailing code block markers
        if text.startswith('```python'):
            text = text[9:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        return text.strip()
    
    def improve_skill(
        self,
        skill_name: str,
        feedback: str,
        changes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Improve an existing skill based on feedback.
        
        Args:
            skill_name: Name of skill to improve
            feedback: What's wrong or what needs improvement
            changes: Specific changes requested
            
        Returns:
            Updated skill metadata
        """
        skill_dir = self.skills_dir / skill_name
        
        if not skill_dir.exists():
            raise ValueError(f"Skill {skill_name} not found")
        
        # Read existing files
        skill_md_path = skill_dir / "SKILL.md"
        tools_py_path = skill_dir / "tools.py"
        
        existing_md = skill_md_path.read_text() if skill_md_path.exists() else ""
        existing_py = tools_py_path.read_text() if tools_py_path.exists() else ""
        
        # Generate improvements
        if skill_md_path.exists():
            improved_md = self._improve_skill_md(existing_md, feedback, changes)
            skill_md_path.write_text(improved_md)
        
        if tools_py_path.exists():
            improved_py = self._improve_tools_py(existing_py, feedback, changes)
            tools_py_path.write_text(improved_py)
        
        logger.info(f" Improved skill: {skill_name}")
        
        return {
            "name": skill_name,
            "improved": True,
            "feedback": feedback,
        }
    
    def _improve_skill_md(
        self,
        existing_content: str,
        feedback: str,
        changes: Optional[str]
    ) -> str:
        """Improve SKILL.md based on feedback using Jotty's unified LLM interface."""
        prompt = f"""Improve this SKILL.md file based on feedback.

Existing content:
{existing_content}

Feedback: {feedback}
{f'Requested changes: {changes}' if changes else ''}

Generate the improved SKILL.md:"""
        
        # Use DSPy LM interface
        response = self.lm(prompt=prompt)
        
        # Extract text from DSPy response
        if hasattr(response, 'completions') and response.completions:
            return response.completions[0].text.strip()
        elif hasattr(response, 'text'):
            return response.text.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response).strip()
    
    def _improve_tools_py(
        self,
        existing_content: str,
        feedback: str,
        changes: Optional[str]
    ) -> str:
        """Improve tools.py based on feedback using Jotty's unified LLM interface."""
        prompt = f"""Improve this tools.py file based on feedback.

Existing code:
{existing_content}

Feedback: {feedback}
{f'Requested changes: {changes}' if changes else ''}

Generate the improved, runnable Python code:"""
        
        # Use DSPy LM interface
        response = self.lm(prompt=prompt)
        
        # Extract text from DSPy response
        if hasattr(response, 'completions') and response.completions:
            return response.completions[0].text.strip()
        elif hasattr(response, 'text'):
            return response.text.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response).strip()
    
    def validate_generated_skill(self, skill_name: str) -> Dict[str, Any]:
        """
        Validate a generated skill before registration.
        
        Args:
            skill_name: Name of skill to validate
            
        Returns:
            Validation results
        """
        skill_dir = self.skills_dir / skill_name
        
        if not skill_dir.exists():
            return {
                "valid": False,
                "errors": [f"Skill directory not found: {skill_dir}"],
            }
        
        errors = []
        warnings = []
        
        # Check SKILL.md
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            errors.append("Missing SKILL.md")
        else:
            content = skill_md.read_text()
            if "description" not in content.lower():
                warnings.append("SKILL.md missing description")
        
        # Check tools.py
        tools_py = skill_dir / "tools.py"
        if not tools_py.exists():
            errors.append("Missing tools.py")
        else:
            content = tools_py.read_text()
            # Check for function definitions (more lenient)
            if "def " not in content:
                errors.append("tools.py missing function definitions")
            # Note: "tool" in name is optional, just need functions
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


# Singleton instance
_generator_instance: Optional[SkillGenerator] = None


def get_skill_generator(skills_dir: Optional[str] = None, lm: Optional[BaseLM] = None, skills_registry=None) -> SkillGenerator:
    """
    Get singleton skill generator instance.
    
    Args:
        skills_dir: Directory where skills are stored
        lm: Optional DSPy BaseLM instance (uses unified LLM provider if None)
        skills_registry: Optional SkillsRegistry for auto-reload after generation
    
    Returns:
        SkillGenerator instance using Jotty's unified LLM interface
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SkillGenerator(skills_dir, lm, skills_registry)
    elif skills_registry and not _generator_instance.skills_registry:
        # Update registry if provided
        _generator_instance.skills_registry = skills_registry
    return _generator_instance
