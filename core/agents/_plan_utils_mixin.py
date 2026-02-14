"""TaskPlanner Plan Utils Mixin - Tool schema, param building, fallbacks, metadata."""

import json
import logging
import re
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from ._execution_types import ExecutionStep, TaskType, ToolSchema
from Jotty.core.utils.context_utils import strip_enrichment_context

logger = logging.getLogger(__name__)


class PlanUtilsMixin:
    def _extract_tool_schema(self, tool_func: Any, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameter schema from a tool function.

        Uses ``ToolSchema.from_tool_function()`` which inspects:
        1. ``@tool_wrapper(required_params=[...])`` decorator (most reliable)
        2. Docstring parsing for types, descriptions, and extra params

        Returns a backward-compatible dict with 'name', 'parameters', 'description'
        while also caching the typed ``ToolSchema`` on the function for downstream use.

        Args:
            tool_func: The tool function
            tool_name: Name of the tool

        Returns:
            Dictionary with tool name, parameters, and description
        """
        if not tool_func:
            return {'name': tool_name, 'parameters': [], 'description': ''}

        # Build typed schema (cached on function for execute_step to reuse)
        schema = ToolSchema.from_tool_function(tool_func, tool_name)
        if not hasattr(tool_func, '_tool_schema'):
            tool_func._tool_schema = schema

        # Return backward-compatible dict for existing callers
        return schema.to_dict()
    
    def _abstract_task_for_planning(self, task: str) -> str:
        """
        Prepare task description for LLM planning calls.

        CRITICAL: Strips out context pollution markers that can corrupt:
        - Search queries (causing massive URLs)
        - PDF topic parameters
        - Skill tool inputs

        Context pollution markers include:
        - Q-Learning lessons
        - Transfer learning context
        - Multi-perspective analysis
        - Previous run metadata

        Truncates to 500 chars to stay within reasonable prompt limits.
        """
        # Clean the task first, then truncate
        cleaned = self._clean_task_for_query(task)
        return cleaned[:500].strip()

    def _clean_task_for_query(self, task: str) -> str:
        """
        Clean task description for use in search queries and skill inputs.

        CRITICAL: This prevents query pollution where enrichment context
        (Q-learning, transfer learning, etc.) gets passed to web search
        causing massive URLs and timeouts.

        Uses the centralised ``strip_enrichment_context()`` utility so
        all marker lists are maintained in one place.
        """
        return strip_enrichment_context(task)

    def _build_skill_params(self, skill_name: str, task: str, prev_output_key: Optional[str] = None, tool_name: str = None) -> Dict[str, Any]:
        """Build params for a skill by looking up its tool schema from registry."""
        params = {}
        prev_ref = f"${{{prev_output_key}}}" if prev_output_key else task

        # Special handling for write_file_tool - generate actual content
        # Only short-circuit if LLM generation succeeds AND no previous step output
        # to wire. When prev_output_key exists, prefer wiring step output as content.
        if skill_name == 'file-operations' and tool_name == 'write_file_tool':
            if not prev_output_key:
                params = self._generate_file_content(task)
                if params.get('path') and params.get('content'):
                    return params

        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            skill = registry.get_skill(skill_name)

            if skill and hasattr(skill, 'tools') and skill.tools:
                # Get the specific tool or first available
                tool_func = skill.tools.get(tool_name) if tool_name else list(skill.tools.values())[0]

                if tool_func:
                    # Extract parameter schema from docstring
                    schema = self._extract_tool_schema(tool_func, tool_name or list(skill.tools.keys())[0])
                    tool_params = schema.get('parameters', [])

                    # CRITICAL: Clean task for query params (strip enrichment context)
                    # The task may contain appended context like:
                    # - "[Multi-Perspective Analysis..."
                    # - "Learned Insights:..."
                    # - "# Transferable Learnings..."
                    # This context pollutes search queries causing timeouts
                    clean_task = self._clean_task_for_query(task)

                    # Build params based on actual schema
                    for param in tool_params:
                        param_name = param.get('name', '')
                        param_required = param.get('required', False)
                        param_desc = param.get('description', '').lower()

                        # Intelligently fill params based on description and name
                        # Use CLEAN task for search queries to avoid polluting URLs
                        if param_name in ['query', 'topic', 'search_query', 'q']:
                            params[param_name] = clean_task
                        elif param_name == 'expression':
                            # Extract math expression from task text
                            import re as _re
                            math_match = _re.search(
                                r'([\d\.\+\-\*/\(\)\^\s%]+(?:[\d\.\)])\s*)', clean_task
                            )
                            params[param_name] = math_match.group(1).strip() if math_match else clean_task
                        elif param_name in ['message', 'text', 'content', 'body']:
                            # For write_file_tool, try LLM content gen first, then
                            # fall back to previous step output (prev_ref), never TODO stub
                            if tool_name == 'write_file_tool':
                                file_content = self._generate_file_content(task)
                                if file_content and file_content.get('content'):
                                    params[param_name] = file_content['content']
                                else:
                                    # Wire previous step output — ParameterResolver will
                                    # substitute ${step_key} at execution time
                                    params[param_name] = prev_ref
                            else:
                                params[param_name] = prev_ref
                        elif param_name in ['file_path', 'pdf_path', 'path', 'input_path']:
                            import re as _re_path
                            # Extract actual file/directory path from task for ALL file-operations tools
                            _path_patterns = [
                                # Preposition + optional words + relative path: "in the current Jotty/core/agents/"
                                r'(?:in|from|at|of)\s+[\w\s]{0,30}?([\w\-\.]+(?:/[\w\-\.]+)+/?)(?:\s|$)',
                                # "save as/to" patterns
                                r'save\s+(?:as|to|it\s+(?:as|to))\s+["\']?([\w\-\./]+)["\']?',
                                # "called/named" patterns: file called foo.txt
                                r'(?:called|named)\s+["\']?([\w\-]+\.[\w]{1,5})["\']?',
                                # Quoted paths
                                r'["\']([\w\-\.]+(?:/[\w\-\.]+)*/?)["\']',
                                # Absolute paths: /tmp/dir/, /home/user/file.py
                                r'(/(?:[\w\-\.]+/)+[\w\-\.]*)',
                                # Relative path with 2+ segments: Jotty/core/agents/
                                r'([\w\-\.]+(?:/[\w\-\.]+){2,}/?)',
                                # Any dir/file: src/file.py
                                r'([\w\-]+/[\w\-]+\.[\w]{1,5})',
                                # Simple filename: output.txt
                                r'(\w+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))',
                            ]
                            extracted_path = None
                            for pattern in _path_patterns:
                                match = _re_path.search(pattern, task, _re_path.IGNORECASE)
                                if match:
                                    extracted_path = match.group(1).rstrip()
                                    break
                            if extracted_path:
                                params[param_name] = extracted_path
                                logger.debug(f"Extracted path: '{extracted_path}' from task")
                            else:
                                params[param_name] = prev_ref
                        elif param_name in ['max_results', 'limit', 'count']:
                            params[param_name] = 10
                        elif param_name in ['title', 'name']:
                            params[param_name] = clean_task[:50]
                        elif param_name in ['format', 'output_format', 'type']:
                            # Pick a sensible default from the description if it lists options
                            import re as _re_fmt
                            options = _re_fmt.findall(r"'(\w+)'", param_desc)
                            params[param_name] = options[0] if options else 'plain'
                        elif param_name in ['encoding', 'charset']:
                            params[param_name] = 'utf-8'
                        elif param_required:
                            # For other required params, use task or prev_ref based on description
                            if any(word in param_desc for word in ['input', 'content', 'data', 'result']):
                                params[param_name] = prev_ref
                            else:
                                params[param_name] = clean_task

                    if params:
                        logger.debug(f"Built params from schema for {skill_name}: {list(params.keys())}")
                        return params

        except Exception as e:
            logger.warning(f"Could not build params from registry for {skill_name}: {e}")

        # Fallback: generic params covering common required fields
        # Use CLEAN task for query/topic to avoid polluting search URLs
        clean_task = self._clean_task_for_query(task)

        # Extract math expression for calculator-like skills
        import re as _re
        math_match = _re.search(r'([\d\.\+\-\*/\(\)\^\s%]+(?:[\d\.\)]))\s*', clean_task)
        expression_val = math_match.group(1).strip() if math_match else clean_task

        return {
            'task': clean_task,
            'query': clean_task,
            'topic': clean_task,  # For research skills
            'expression': expression_val,  # For calculator skills
            'input': prev_ref,
            'content': prev_ref,
            'text': prev_ref,
            'message': prev_ref,  # For notification skills
        }

    def _generate_file_content(self, task: str) -> Dict[str, Any]:
        """Generate actual file content using LLM for write_file_tool."""
        import re

        # Extract filename/path from task (supports directory and absolute paths)
        filepath_patterns = [
            # Match absolute paths: /tmp/file.html, /home/user/file.py
            r'(/(?:[\w\-\.]+/)*[\w\-\.]+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))',
            # Match "save as /path/file.ext" or "save to /path/file.ext"
            r'save\s+(?:as|to|it\s+as|it\s+to)\s+["\']?(/[\w\-\./]+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))["\']?',
            # Match paths with directories: pkg/subdir/file.py
            r'(?:create|write|make|generate)\s+(?:a\s+)?(?:file\s+)?(?:called\s+)?["\']?([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))["\']?',
            # Match quoted paths with directories
            r'["\']([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))["\']',
            # Match any path-like pattern: dir/file.ext
            r'([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))',
            # Fallback: simple filename
            r'(\w+\.(?:py|js|ts|html|css|json|md|txt))',
        ]

        filename = None
        for pattern in filepath_patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                filename = match.group(1)
                break

        # If no explicit filename, infer from task content
        if not filename:
            filename = self._infer_filename_from_task(task)

        if not filename:
            return {}

        # Use LLM to generate the actual content
        try:
            import dspy
            lm = dspy.settings.lm
            if not lm:
                return {}

            # Determine file type for appropriate code generation
            ext = filename.split('.')[-1].lower()
            lang_hint = {
                'py': 'Python',
                'js': 'JavaScript',
                'ts': 'TypeScript',
                'html': 'HTML',
                'css': 'CSS',
                'json': 'JSON',
                'md': 'Markdown',
            }.get(ext, 'code')

            if lang_hint == 'Markdown':
                prompt = f"""Generate a professional Markdown document for this task.
Use proper headings, bullet points, and tables where appropriate.
Use ONLY real data from the task context — do NOT fabricate or simulate data.
Return ONLY the Markdown content, no explanations.

Task: {task}
Filename: {filename}

Markdown:"""
            else:
                prompt = f"""Generate the complete {lang_hint} code for this task. Return ONLY the code, no explanations.

Task: {task}
Filename: {filename}

{lang_hint} code:"""

            response = lm(prompt=prompt)
            content = response[0] if isinstance(response, list) else str(response)

            # Clean up response (remove markdown code blocks if present)
            content = content.strip()
            if content.startswith('```'):
                lines = content.split('\n')
                # Remove first line (```python) and last line (```)
                if lines[-1].strip() == '```':
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                content = '\n'.join(lines)

            logger.debug(f"Generated {len(content)} chars of content for {filename}")
            return {'path': filename, 'content': content}

        except Exception as e:
            logger.warning(f"Failed to generate file content: {e}")
            return {}

    def _infer_filename_from_task(self, task: str) -> Optional[str]:
        """Infer appropriate filename from task description when not explicitly provided."""
        import re
        task_lower = task.lower()

        # Detect file type from task keywords
        # Priority: Reports/synthesis → explicit code types → default
        file_ext = '.py'  # Default to Python
        _is_report_or_synthesis = any(w in task_lower for w in [
            'report', 'recommendation', 'summary', 'comparison', 'analysis',
            'overview', 'ranking', 'ranked', 'findings', 'synthesis',
            'strengths and weaknesses', 'pros and cons',
        ])
        _is_python_backend = any(w in task_lower for w in [
            'python', 'fastapi', 'flask', 'django', 'api', 'endpoint', 'server',
            'backend', 'script', 'class', 'function', 'module', '.py',
        ])
        if _is_report_or_synthesis and not any(w in task_lower for w in ['script', '.py', 'python code']):
            file_ext = '.md'
        elif _is_python_backend:
            file_ext = '.py'  # Explicit Python — don't let 'web' override
        elif any(w in task_lower for w in ['html page', 'webpage', 'web page', 'html file', '.html']):
            file_ext = '.html'
        elif any(w in task_lower for w in ['javascript', 'react', 'node', '.js']):
            file_ext = '.js'
        elif any(w in task_lower for w in ['typescript', '.ts']):
            file_ext = '.ts'
        elif any(w in task_lower for w in ['css', 'stylesheet']):
            file_ext = '.css'
        elif any(w in task_lower for w in ['json config', '.json']):
            file_ext = '.json'
        elif any(w in task_lower for w in ['markdown', 'readme', 'documentation', '.md']):
            file_ext = '.md'

        # Try to extract a meaningful name from the task
        # Pattern 1: "Create/Build/Make X model/class/service/component"
        patterns = [
            r'(?:create|build|make|implement|write|add)\s+(?:a\s+)?(\w+)\s+(?:model|class|service|component|module|handler|controller|view)',
            r'(?:create|build|make|implement|write)\s+(?:a\s+)?(\w+)(?:Service|Model|Controller|Handler|Manager|Component)',
            r'(\w+)\s+(?:model|class|service|component|module)\s+with',
            r'(?:unit\s+)?tests?\s+for\s+(?:the\s+)?(\w+)',
            r'(\w+)\s+(?:implementation|functionality)',
        ]

        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                name = match.group(1).lower()
                # Clean up common prefixes
                name = re.sub(r'^(the|a|an)\s*', '', name)
                if name and len(name) > 1:
                    # Format filename based on type
                    if 'test' in task_lower:
                        return f"test_{name}{file_ext}"
                    elif 'model' in task_lower:
                        return f"models/{name}{file_ext}"
                    elif 'service' in task_lower:
                        return f"services/{name}_service{file_ext}"
                    elif 'api' in task_lower or 'endpoint' in task_lower:
                        return f"api/{name}{file_ext}"
                    else:
                        return f"{name}{file_ext}"

        # Fallback: Generate name from first meaningful word
        words = re.findall(r'\b([a-z]{3,})\b', task_lower)
        skip_words = {'create', 'build', 'make', 'implement', 'write', 'add', 'the', 'with', 'for', 'and', 'that', 'this'}
        for word in words:
            if word not in skip_words:
                if 'test' in task_lower:
                    return f"test_{word}{file_ext}"
                return f"{word}{file_ext}"

        # Last resort: app.py or index.html
        if file_ext == '.html':
            return 'index.html'
        return 'app.py'

    def _create_fallback_plan(self, task: str, task_type: Any, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a minimal fallback plan when LLM fails.

        Prioritizes skills by relevance to task type:
        - RESEARCH: web-search, http-client, web-scraper, claude-cli-llm
        - CREATION: claude-cli-llm, file-operations, docx-tools
        - ANALYSIS: calculator, claude-cli-llm
        - Default: claude-cli-llm, web-search
        """
        if not skills:
            return []

        # TaskType imported at module level from _execution_types

        # Priority order by task type
        priority_map = {
            TaskType.RESEARCH: ['web-search', 'http-client', 'web-scraper', 'claude-cli-llm', 'arxiv-downloader'],
            TaskType.COMPARISON: ['web-search', 'claude-cli-llm', 'calculator'],
            TaskType.CREATION: ['claude-cli-llm', 'file-operations', 'docx-tools', 'text-utils'],
            TaskType.ANALYSIS: ['claude-cli-llm', 'calculator', 'web-search'],  # claude-cli-llm first for general analysis
            TaskType.COMMUNICATION: ['claude-cli-llm', 'http-client'],
            TaskType.AUTOMATION: ['shell-exec', 'file-operations', 'process-manager'],
            TaskType.UNKNOWN: ['claude-cli-llm', 'web-search', 'calculator'],  # Default fallback for unknown
        }

        # Get priority list for this task type (claude-cli-llm is always a safe fallback)
        priority_skills = priority_map.get(task_type, ['claude-cli-llm', 'web-search', 'calculator'])

        # Sort skills: originally selected skills FIRST, then priority skills
        skill_names = {s.get('name', ''): s for s in skills}
        sorted_skills = []

        # Add originally selected skills first (respect LLM's selection)
        for s in skills:
            if s not in sorted_skills:
                sorted_skills.append(s)

        # Then add priority skills that weren't already selected
        for ps in priority_skills:
            if ps in skill_names and skill_names[ps] not in sorted_skills:
                sorted_skills.append(skill_names[ps])

        plan = []

        for skill in sorted_skills:
            skill_name = skill.get('name', '')
            tools = skill.get('tools', [])
            if isinstance(tools, dict):
                tools = list(tools.keys())
            if not tools:
                # Try loading from registry
                try:
                    from ..registry.skills_registry import get_skills_registry
                    registry = get_skills_registry()
                    skill_obj = registry.get_skill(skill_name)
                    if skill_obj and skill_obj.tools:
                        tools = list(skill_obj.tools.keys())
                except Exception:
                    pass

            if not tools:
                continue

            # Get tool names list
            tool_names = [t if isinstance(t, str) else t.get('name', '') for t in tools]
            tool_names = [t for t in tool_names if t]  # Filter empty

            # Select best tool based on task keywords
            task_lower = task.lower()
            tool_name = tool_names[0] if tool_names else ''  # Default to first

            # Smart tool selection for common skills
            if skill_name == 'file-operations':
                if any(w in task_lower for w in ['create', 'write', 'generate', 'make', 'save']):
                    if 'write_file_tool' in tool_names:
                        tool_name = 'write_file_tool'
                elif any(w in task_lower for w in ['read', 'load', 'get', 'open']):
                    if 'read_file_tool' in tool_names:
                        tool_name = 'read_file_tool'
                elif any(w in task_lower for w in ['delete', 'remove']):
                    if 'delete_file_tool' in tool_names:
                        tool_name = 'delete_file_tool'
                elif any(w in task_lower for w in ['directory', 'folder', 'mkdir']):
                    if 'create_directory_tool' in tool_names:
                        tool_name = 'create_directory_tool'
            elif skill_name == 'claude-cli-llm':
                if any(w in task_lower for w in ['generate', 'create', 'write', 'code']):
                    if 'generate_text_tool' in tool_names:
                        tool_name = 'generate_text_tool'
            elif skill_name == 'shell-exec':
                if 'run_command_tool' in tool_names:
                    tool_name = 'run_command_tool'

            logger.debug(f"Fallback selected tool '{tool_name}' for skill '{skill_name}'")

            prev_output_key = f'result_{len(plan) - 1}' if plan else None
            params = self._build_skill_params(skill_name, task, prev_output_key, tool_name)

            plan.append({
                'skill_name': skill_name,
                'tool_name': tool_name,
                'params': params,
                'description': f'Execute {skill_name}: {task}',
                'depends_on': [len(plan) - 1] if plan else [],  # Chain steps
                'output_key': f'result_{len(plan)}',
                'optional': len(plan) > 0  # First step required, rest optional
            })

            # For simple tasks with few selected skills, limit fallback steps
            max_fallback = min(3, max(1, len(skills)))
            if len(plan) >= max_fallback:
                break

        logger.info(f"Fallback plan: {len(plan)} steps from selected skills")
        return plan
    
    def _extract_skill_names_from_text(self, text: str) -> List[str]:
        """Extract skill names from LLM text output."""
        import re
        skill_names = []

        # Try to find JSON array pattern: ["skill1", "skill2"]
        json_array_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if json_array_match:
            array_content = json_array_match.group(1)
            # Extract quoted strings
            matches = re.findall(r'"([^"]+)"', array_content)
            skill_names.extend(matches)

        # Try markdown list format: - skill-name or * skill-name
        if not skill_names:
            md_matches = re.findall(r'^[\-\*]\s*([a-z][a-z0-9\-_]+)', text, re.MULTILINE)
            skill_names.extend(md_matches)

        # Try numbered list: 1. skill-name
        if not skill_names:
            num_matches = re.findall(r'^\d+\.\s*([a-z][a-z0-9\-_]+)', text, re.MULTILINE)
            skill_names.extend(num_matches)

        # Also look for standalone quoted strings that might be skill names
        if not skill_names:
            matches = re.findall(r'"([^"]+)"', text)
            # Filter to likely skill names (lowercase, hyphens, common skill patterns)
            skill_names = [m for m in matches if ('-' in m or '_' in m) and m.islower()]

        # Last resort: find any word that looks like a skill name (has hyphen)
        if not skill_names:
            word_matches = re.findall(r'\b([a-z][a-z0-9]*-[a-z0-9\-]+)\b', text.lower())
            skill_names.extend(word_matches)

        # Remove duplicates and limit
        return list(dict.fromkeys(skill_names))[:10]
    
    def _extract_plan_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract execution plan from LLM text output when JSON parsing fails.

        Returns empty list if extraction fails - fallback plan will be used.
        """
        import re
        steps = []

        # Try to find quoted skill names like "web-search" or 'web-search'
        skill_pattern = r'["\']([a-z][a-z0-9\-]+)["\']'
        skill_mentions = re.findall(skill_pattern, text.lower())

        # Filter to valid-looking skill names (must have hyphen or underscore, typical of skill names)
        valid_skills = [s for s in skill_mentions if '-' in s and len(s) > 4]

        if not valid_skills:
            # No valid skills found - return empty so fallback plan is used
            logger.debug("Could not extract valid skill names from text, using fallback plan")
            return []

        # Create a simple step for each mentioned skill
        for i, skill_name in enumerate(dict.fromkeys(valid_skills)):  # Deduplicate
            steps.append({
                'skill_name': skill_name,
                'tool_name': '',  # Will be resolved by validation
                'params': {},
                'description': f'Execute {skill_name}',
                'depends_on': [i-1] if i > 0 else [],
                'output_key': f'step_{i}',
                'optional': i > 0  # First step required, rest optional
            })

        return steps[:5]  # Limit to 5 steps
    
    # =============================================================================
    # Enhanced Planning with TaskGraph and Metadata
    # =============================================================================
    
    async def plan_with_metadata(
        self,
        task: Union[str, 'TaskGraph'],
        available_skills: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 15,
        convert_to_agents: bool = False
    ) -> 'TaskPlan':
        """
        Plan execution with enhanced metadata (TaskPlan).
        
        Works with both raw strings and TaskGraph.
        
        Args:
            task: Task description (str) or TaskGraph
            available_skills: Available skills (auto-discovers if None)
            max_steps: Maximum execution steps
            convert_to_agents: If True, also convert skills to AgentConfig for Conductor
            
        Returns:
            TaskPlan with steps and metadata
        """
        # Handle TaskGraph or raw string
        if TASK_GRAPH_AVAILABLE and isinstance(task, TaskGraph):
            task_string = task.metadata.get('original_request', '')
            task_type = task.task_type
            integrations = task.integrations
        else:
            task_string = str(task)
            task_type, _, _ = self.infer_task_type(task_string)
            integrations = []
        
        # Get TaskType for type checking
        # TaskType imported at module level from _execution_types
        
        # Discover skills if not provided
        if available_skills is None:
            try:
                from ..registry.skills_registry import get_skills_registry
                registry = get_skills_registry()
                registry.init()
                available_skills = registry.list_skills()
            except Exception as e:
                logger.warning(f"Failed to discover skills: {e}")
                available_skills = []
        
        # Select best skills
        selected_skills, selection_reasoning = self.select_skills(
            task=task_string,
            available_skills=available_skills,
            max_skills=10
        )
        
        # Optionally convert skills to agents for Conductor
        agents = None
        if convert_to_agents:
            agents = await self._convert_skills_to_agents(selected_skills)
        
        # Plan execution
        steps, planning_reasoning = self.plan_execution(
            task=task_string,
            task_type=task_type,
            skills=selected_skills,
            max_steps=max_steps
        )
        
        # Extract metadata
        skill_names = [s.get('name') for s in selected_skills]
        required_tools = self._extract_required_tools(steps, skill_names)
        required_credentials = self._extract_required_credentials(integrations)
        estimated_time = self._estimate_time(steps)
        
        # Create TaskPlan
        if TASK_GRAPH_AVAILABLE and isinstance(task, TaskGraph):
            task_graph = task
        else:
            # Create minimal TaskGraph from raw string
            task_graph = self._create_task_graph_from_string(task_string, task_type)
        
        metadata = {
            'skills_discovered': skill_names,
            'selection_reasoning': selection_reasoning,
            'planning_reasoning': planning_reasoning
        }
        
        if agents:
            metadata['agents_created'] = [a.name for a in agents]
        
        return TaskPlan(
            task_graph=task_graph,
            steps=steps,
            estimated_time=estimated_time,
            required_tools=required_tools,
            required_credentials=required_credentials,
            metadata=metadata
        )
    
    async def _convert_skills_to_agents(self, skills: List[Dict[str, Any]]) -> Any:
        """Convert skills to AgentConfig for Conductor."""
        try:
            from ..registry.skill_to_agent_converter import SkillToAgentConverter
            from ..registry.skills_registry import get_skills_registry
            
            converter = SkillToAgentConverter()
            registry = get_skills_registry()
            
            # Get SkillDefinition objects
            skill_defs = []
            for skill_dict in skills:
                skill_name = skill_dict.get('name')
                if skill_name:
                    skill_def = registry.get_skill(skill_name)
                    if skill_def:
                        skill_defs.append(skill_def)
            
            # Convert to agents
            agents = await converter.convert_skills_to_agents(skill_defs)
            return agents
            
        except Exception as e:
            logger.warning(f"Failed to convert skills to agents: {e}")
            return None
    
    def _extract_required_tools(self, steps: List[Any], skill_names: List[str]) -> List[str]:
        """Extract all required tools from steps."""
        tools = set(skill_names)
        for step in steps:
            if step.skill_name:
                tools.add(step.skill_name)
        return list(tools)
    
    def _extract_required_credentials(self, integrations: List[str]) -> List[str]:
        """Extract required credentials from integrations (any integration may need credentials)."""
        credentials = []
        for integration in integrations:
            # Assume any integration might need an API key
            credentials.append(f"{integration.lower()}_api_key")
        return credentials
    
    def _estimate_time(self, steps: List[Any]) -> str:
        """Estimate total execution time."""
        time_per_step = 2  # minutes average
        total_minutes = len(steps) * time_per_step
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"
    
    def _create_task_graph_from_string(self, task_string: str, task_type: Any) -> Optional[Any]:
        """Create minimal TaskGraph from raw string."""
        if not TASK_GRAPH_AVAILABLE:
            return None

        from ..autonomous.intent_parser import TaskGraph
        return TaskGraph(
            task_type=task_type,
            workflow=None,
            metadata={'original_request': task_string}
        )

    # =========================================================================
    # Plan normalization + parsing (moved from agentic_planner.py)
    # =========================================================================

    def _normalize_raw_plan(self, raw_plan: Any, skills: Optional[List[Dict[str, Any]]] = None, task: str = '', task_type: Any = None) -> list:
        """
        Single, robust normalizer: convert any LLM plan output to a list of dicts.

        Handles all known LLM output formats in a single pipeline:
        1. Already a list (JSONAdapter working correctly)
        2. Direct JSON string starting with '['
        3. JSON inside markdown code block
        4. JSON array embedded in prose text
        5. Skill-name extraction from unstructured text
        6. Direct LLM retry with explicit JSON-only prompt (last resort)

        Args:
            raw_plan: Raw plan from DSPy (list, Pydantic model, string, or None)
            skills: Available skills (needed for Method 6 LLM retry)
            task: Original task description (needed for Method 6)
            task_type: Task type (needed for Method 6)

        Returns:
            List of dicts (may be empty if all parsing fails)
        """
        # Already a list — nothing to normalize
        if isinstance(raw_plan, list):
            logger.info(f"   Plan data is list: {len(raw_plan)} steps")
            return raw_plan

        if not raw_plan:
            return []

        plan_str = str(raw_plan).strip()

        # Method 1: Direct JSON parse (LLM returned clean JSON string)
        if plan_str.startswith('['):
            try:
                plan_data = json.loads(plan_str)
                logger.info(f"   Direct JSON parse successful: {len(plan_data)} steps")
                return plan_data if isinstance(plan_data, list) else [plan_data]
            except json.JSONDecodeError:
                pass

        # Method 2: Extract from markdown code block
        if '```' in plan_str:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', plan_str, re.DOTALL)
            if json_match:
                try:
                    plan_data = json.loads(json_match.group(1).strip())
                    logger.info(f"   Extracted from code block: {len(plan_data)} steps")
                    return plan_data if isinstance(plan_data, list) else [plan_data]
                except json.JSONDecodeError:
                    pass

        # Method 3: Find JSON array anywhere in text
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', plan_str, re.DOTALL)
        if array_match:
            try:
                plan_data = json.loads(array_match.group(0))
                logger.info(f"   Extracted JSON array from text: {len(plan_data)} steps")
                return plan_data if isinstance(plan_data, list) else [plan_data]
            except json.JSONDecodeError:
                pass

        # Method 4: Extract skill names from unstructured text
        plan_data = self._extract_plan_from_text(plan_str)
        if plan_data:
            logger.info(f"   Extracted via skill-name helper: {len(plan_data)} steps")
            return plan_data

        # Method 5: Direct LLM retry with explicit JSON-only prompt (last resort)
        if skills:
            try:
                import dspy as _dspy
                lm = _dspy.settings.lm
                if lm:
                    skill_info = []
                    for s in skills:
                        tools = s.get('tools', [])
                        if isinstance(tools, list) and tools:
                            tool_name = tools[0].get('name', tools[0]) if isinstance(tools[0], dict) else tools[0]
                            skill_info.append(f"{s.get('name')}/{tool_name}")
                        else:
                            skill_info.append(s.get('name', ''))

                    task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type or 'general')
                    direct_prompt = (
                        f"Return ONLY a JSON array with 2-3 steps. Select ONLY the most relevant skills.\n\n"
                        f"Task: {task}\nTask type: {task_type_str}\n"
                        f"Available skills: {skill_info}\n\n"
                        f'Select 2-3 most relevant skills. Return JSON array:\n'
                        f'[{{"skill_name": "skill-name", "tool_name": "tool-name", "params": {{}}, '
                        f'"description": "what it does", "depends_on": [], "output_key": "step_0", "optional": false}}]\n\n'
                        f'JSON:'
                    )

                    response = lm(prompt=direct_prompt)
                    response_text = (response[0] if isinstance(response, list) else str(response)).strip()
                    logger.debug(f"   Direct LLM response (first 200): {response_text[:200]}")

                    if response_text.startswith('['):
                        plan_data = json.loads(response_text)
                        logger.info(f"   Direct LLM retry successful: {len(plan_data)} steps")
                        return plan_data if isinstance(plan_data, list) else [plan_data]
                    elif '[' in response_text:
                        start = response_text.find('[')
                        end = response_text.rfind(']') + 1
                        if end > start:
                            plan_data = json.loads(response_text[start:end])
                            logger.info(f"   Extracted from direct LLM: {len(plan_data)} steps")
                            return plan_data if isinstance(plan_data, list) else [plan_data]
            except Exception as e:
                logger.warning(f"   Direct LLM retry failed: {e}")

        logger.error(f"Could not parse plan. Raw (first 300 chars): {plan_str[:300]}")
        return []

    def _parse_plan_to_steps(self, raw_plan: Any, skills: List[Dict[str, Any]], task: str, task_type: Any = None, max_steps: int = 10) -> list:
        """
        Parse raw plan data into ExecutionStep objects.

        Phase 1 (normalization) is delegated to _normalize_raw_plan().
        Phase 2 converts the normalized list of dicts into ExecutionStep objects
        with fuzzy skill matching, tool inference, and param building.

        Args:
            raw_plan: Raw plan from DSPy (list of dicts/Pydantic models, or string, or None)
            skills: Available skills list
            task: Original task description
            task_type: Task type (for fallback plan, optional)
            max_steps: Maximum steps to parse

        Returns:
            List of ExecutionStep objects
        """
        # --- Phase 1: Normalize to list of dicts ---
        plan_data = self._normalize_raw_plan(raw_plan, skills=skills, task=task, task_type=task_type)

        if not plan_data:
            return []

        # --- Phase 2: Convert plan_data to ExecutionStep objects ---
        steps = []

        # Build tool-to-skill mapping
        tool_to_skill = {}
        for s in skills:
            skill_name_map = s.get('name', '')
            for t in s.get('tools', []):
                t_name = t.get('name') if isinstance(t, dict) else t
                if t_name:
                    tool_to_skill[t_name] = skill_name_map

        available_skill_names = {s.get('name', '') for s in skills if s.get('name')}
        logger.info(f" Available skills for validation: {sorted(available_skill_names)}")

        def get_val(obj: Union[Dict[str, Any], object], key: str, default: Any = '') -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def find_matching_skill(name: str) -> str:
            """Find matching skill using exact, contains, or word-overlap match."""
            if not name:
                return ''
            name_lower = name.lower().strip()
            if name_lower in {s.lower() for s in available_skill_names}:
                for s in available_skill_names:
                    if s.lower() == name_lower:
                        return s
            for s in available_skill_names:
                if name_lower in s.lower() or s.lower() in name_lower:
                    logger.debug(f"Fuzzy matched '{name}' -> '{s}'")
                    return s
            name_words = set(name_lower.replace('-', '_').replace(' ', '_').split('_'))
            for s in available_skill_names:
                skill_words = set(s.lower().replace('-', '_').split('_'))
                if name_words & skill_words:
                    logger.debug(f"Word overlap matched '{name}' -> '{s}'")
                    return s
            return ''

        for i, step_data in enumerate(plan_data[:max_steps]):
            try:
                logger.debug(f"Processing step {i+1}: {step_data}")

                skill_name = get_val(step_data, 'skill_name') or get_val(step_data, 'skill', '')
                tool_name = get_val(step_data, 'tool_name') or get_val(step_data, 'tool', '') or get_val(step_data, 'action', '')

                # Infer skill from tool if skill is empty
                if not skill_name and tool_name:
                    skill_name = tool_to_skill.get(tool_name, '')

                # Infer skill from available skills if only one selected
                if not skill_name and len(available_skill_names) == 1:
                    skill_name = list(available_skill_names)[0]
                    logger.info(f"Auto-inferred skill_name='{skill_name}' (only one skill available)")
                elif not skill_name and len(available_skill_names) <= 3:
                    desc = str(get_val(step_data, 'description', task)).lower()
                    for candidate in available_skill_names:
                        if candidate.replace('-', ' ') in desc or any(w in desc for w in candidate.split('-')):
                            skill_name = candidate
                            logger.info(f"Inferred skill_name='{skill_name}' from description match")
                            break

                # Try fuzzy matching if exact skill not found
                if skill_name and skill_name not in available_skill_names:
                    matched = find_matching_skill(skill_name)
                    if matched:
                        logger.info(f"Skill name normalized: '{skill_name}' -> '{matched}'")
                        skill_name = matched

                # FALLBACK: Infer from description or use default
                description = get_val(step_data, 'description', f'Step {i+1}')
                if not skill_name or skill_name not in available_skill_names:
                    desc_lower = str(description).lower()

                    inferred_skill = None
                    if any(w in desc_lower for w in ['search', 'find', 'lookup', 'research', 'web', 'news', 'fetch data']):
                        inferred_skill = 'web-search'
                    elif any(w in desc_lower for w in ['create', 'write', 'generate', 'save', 'file', 'report']):
                        inferred_skill = 'file-operations'
                    elif any(w in desc_lower for w in ['chart', 'graph', 'plot', 'visualiz']):
                        inferred_skill = 'chart-creator'
                    elif any(w in desc_lower for w in ['mindmap', 'diagram', 'map']):
                        inferred_skill = 'mindmap-generator'
                    elif any(w in desc_lower for w in ['analyz', 'compar', 'evaluat']):
                        inferred_skill = 'web-search'

                    if inferred_skill and inferred_skill in available_skill_names:
                        logger.info(f"Step {i+1}: Inferred skill '{inferred_skill}' from description")
                        skill_name = inferred_skill
                    elif 'web-search' in available_skill_names:
                        skill_name = 'web-search'
                    elif 'file-operations' in available_skill_names:
                        skill_name = 'file-operations'
                    elif available_skill_names:
                        skill_name = list(available_skill_names)[0]
                    else:
                        logger.warning(f"Skipping step {i+1}: '{str(description)[:50]}' - no skills available")
                        continue

                # Infer tool_name from skill if empty
                if not tool_name:
                    for s in skills:
                        if s.get('name') == skill_name:
                            skill_tools = s.get('tools', [])
                            if skill_tools:
                                tool_names_list = [t.get('name') if isinstance(t, dict) else t for t in skill_tools]
                                desc_lower = description.lower()
                                task_lower = task.lower()

                                if skill_name == 'file-operations':
                                    if any(w in desc_lower for w in ['directory', 'folder', 'mkdir']):
                                        if 'create_directory_tool' in tool_names_list:
                                            tool_name = 'create_directory_tool'
                                    elif any(w in task_lower or w in desc_lower for w in ['create', 'write', 'generate', 'make']):
                                        if any(ext in desc_lower for ext in ['.py', '.js', '.ts', '.json', '.md', '.txt', '.html', '.css', 'file']):
                                            if 'write_file_tool' in tool_names_list:
                                                tool_name = 'write_file_tool'
                                        elif 'write_file_tool' in tool_names_list:
                                            tool_name = 'write_file_tool'
                                    elif any(w in task_lower or w in desc_lower for w in ['read', 'load', 'get']):
                                        if 'read_file_tool' in tool_names_list:
                                            tool_name = 'read_file_tool'

                                if not tool_name:
                                    first_tool = skill_tools[0]
                                    tool_name = first_tool.get('name') if isinstance(first_tool, dict) else first_tool

                                logger.debug(f"Inferred tool_name='{tool_name}' from skill '{skill_name}'")
                            break

                # Handle params
                step_params = (
                    get_val(step_data, 'params') or
                    get_val(step_data, 'parameters') or
                    get_val(step_data, 'tool_parameters') or
                    get_val(step_data, 'inputs') or
                    get_val(step_data, 'tool_input') or
                    {}
                )
                if isinstance(step_params, str):
                    step_params = {}

                # Build fallback params, then merge: LLM params override fallback,
                # but fallback fills in any required params the LLM missed.
                prev_output = f'step_{i-1}' if i > 0 else None
                param_source = task if tool_name in ['write_file_tool', 'read_file_tool'] else task
                fallback_params = self._build_skill_params(skill_name, param_source, prev_output, tool_name)

                if not step_params:
                    step_params = fallback_params
                    logger.debug(f"Built params for step {i+1}: {list(step_params.keys())}")
                else:
                    # Merge: fill missing required params from fallback
                    merged = dict(fallback_params)
                    merged.update(step_params)  # LLM params take priority
                    step_params = merged

                # Extract verification and fallback_skill (research-backed fields)
                verification = get_val(step_data, 'verification', '')
                fallback_skill = get_val(step_data, 'fallback_skill', '')

                # Parse I/O contracts (inputs_needed, outputs_produced)
                raw_inputs = get_val(step_data, 'inputs_needed', {})
                inputs_needed = raw_inputs if isinstance(raw_inputs, dict) else {}
                raw_outputs = get_val(step_data, 'outputs_produced', [])
                outputs_produced = raw_outputs if isinstance(raw_outputs, list) else []

                step = ExecutionStep(
                    skill_name=skill_name,
                    tool_name=tool_name,
                    params=step_params,
                    description=description,
                    depends_on=get_val(step_data, 'depends_on', []),
                    output_key=get_val(step_data, 'output_key', f'step_{i}'),
                    optional=get_val(step_data, 'optional', False),
                    verification=verification,
                    fallback_skill=fallback_skill,
                    inputs_needed=inputs_needed,
                    outputs_produced=outputs_produced,
                )
                steps.append(step)
            except Exception as e:
                logger.warning(f"Failed to create step {i+1}: {e}")
                continue

        return steps

    def _enrich_io_contracts(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        """Post-process LLM plan to auto-populate I/O contracts from tool schemas.

        The fast LLM (Haiku/Gemini Flash) often omits ``inputs_needed`` and
        ``outputs_produced``, and uses bare ``${step_0}`` instead of field-level
        ``${step_0.holdings}``.  This method fixes that by:

        1. Auto-populating ``outputs_produced`` from each tool's ``returns`` schema.
        2. Auto-populating ``inputs_needed`` by scanning params for template refs.
        3. Upgrading bare ``${output_key}`` to ``${output_key.best_field}`` when
           the receiving param name matches a declared output field.

        Args:
            steps: Parsed ExecutionStep objects from ``_parse_plan_to_steps()``.

        Returns:
            Same list, mutated in-place with enriched I/O contracts.
        """
        if not steps:
            return steps

        # Build output_key -> ToolSchema mapping
        key_to_schema: Dict[str, ToolSchema] = {}
        key_to_step_idx: Dict[str, int] = {}

        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
        except Exception:
            registry = None

        for i, step in enumerate(steps):
            output_key = step.output_key or f'step_{i}'

            # --- Phase 1: Auto-populate outputs_produced from tool schema ---
            schema = self._get_tool_schema_for_step(step, registry)
            if schema and schema.outputs:
                declared_names = schema.output_field_names
                if not step.outputs_produced:
                    step.outputs_produced = declared_names
                    logger.debug(f"Step {i} ({output_key}): auto-set outputs_produced={declared_names}")
                key_to_schema[output_key] = schema

            key_to_step_idx[output_key] = i

        # --- Phase 2 & 3: Enrich params with field-level refs + inputs_needed ---
        template_re = re.compile(r'\$\{([^}]+)\}')

        for i, step in enumerate(steps):
            if not step.params:
                continue

            updated_params = {}
            inferred_inputs: Dict[str, str] = dict(step.inputs_needed)  # preserve existing

            for param_name, param_value in step.params.items():
                if not isinstance(param_value, str):
                    updated_params[param_name] = param_value
                    continue

                new_value = param_value

                # Find all template refs in this param value
                for match in template_re.finditer(param_value):
                    ref = match.group(1)  # e.g. "step_0" or "step_0.results"

                    if '.' in ref:
                        # Already field-level — record in inputs_needed
                        base_key = ref.split('.')[0]
                        if param_name not in inferred_inputs:
                            inferred_inputs[param_name] = ref
                        continue

                    # Bare ref like ${step_0} or ${market_breadth}
                    if ref not in key_to_schema:
                        continue

                    schema = key_to_schema[ref]
                    best_field = self._match_output_field(
                        param_name, schema.output_field_names
                    )
                    if best_field:
                        # Upgrade: ${step_0} -> ${step_0.holdings}
                        old_ref = f'${{{ref}}}'
                        new_ref = f'${{{ref}.{best_field}}}'
                        new_value = new_value.replace(old_ref, new_ref)
                        logger.info(
                            f"Step {i}: upgraded {old_ref} -> {new_ref} "
                            f"(param '{param_name}' matched field '{best_field}')"
                        )
                        if param_name not in inferred_inputs:
                            inferred_inputs[param_name] = f'{ref}.{best_field}'

                updated_params[param_name] = new_value

            step.params = updated_params
            if inferred_inputs and not step.inputs_needed:
                step.inputs_needed = inferred_inputs

        return steps

    @staticmethod
    def _get_tool_schema_for_step(step: ExecutionStep, registry: Any) -> Optional[ToolSchema]:
        """Look up ToolSchema for a step's tool from the registry."""
        if not registry:
            return None
        try:
            skill_obj = registry.get_skill(step.skill_name)
            if not skill_obj or not hasattr(skill_obj, 'tools'):
                return None
            tool_func = skill_obj.tools.get(step.tool_name)
            if not tool_func:
                return None
            # Check cached schema first
            if hasattr(tool_func, '_tool_schema'):
                return tool_func._tool_schema
            return ToolSchema.from_tool_function(tool_func, step.tool_name)
        except Exception:
            return None

    @staticmethod
    def _match_output_field(param_name: str, output_fields: List[str]) -> Optional[str]:
        """Match a receiving param name to the best output field.

        Matching priority:
        1. Exact name match (param='holdings', field='holdings')
        2. Semantic match (param='content'/'text'/'body' -> largest text field)
        3. Single-field schemas (only one output field — unambiguous)

        Args:
            param_name: The receiving parameter name (e.g. 'content', 'text', 'data')
            output_fields: Declared output field names from the producer tool

        Returns:
            Best matching field name, or None if no confident match.
        """
        if not output_fields:
            return None

        # 1. Exact match
        if param_name in output_fields:
            return param_name

        # 2. Semantic match for content-like params
        _CONTENT_PARAMS = {'content', 'text', 'body', 'message', 'data', 'input', 'prompt'}
        _CONTENT_FIELDS = {'content', 'text', 'results', 'data', 'output', 'summary', 'response'}
        _IGNORE_FIELDS = {'success', 'error', 'status', 'count', 'query', 'provider'}

        if param_name in _CONTENT_PARAMS:
            # Prefer content-like fields from the output
            for field_name in output_fields:
                if field_name in _CONTENT_FIELDS:
                    return field_name
            # If no content field, pick first non-meta field
            for field_name in output_fields:
                if field_name not in _IGNORE_FIELDS:
                    return field_name

        # 3. Single non-meta field — unambiguous
        non_meta = [f for f in output_fields if f not in _IGNORE_FIELDS]
        if len(non_meta) == 1:
            return non_meta[0]

        return None

    def _maybe_decompose_plan(self, steps: list, skills: List[Dict[str, Any]], task: str, task_type: Any) -> Optional[list]:
        """
        Check if plan quality can be improved by decomposing composite skills.

        For comparison/research tasks with 1-step composite plans, decompose
        into granular steps for better quality (separate searches per entity,
        dedicated synthesis, dedicated formatting).

        Returns:
            Decomposed steps list, or None if no decomposition needed.
        """
        if not steps or len(steps) > 2:
            return None

        clean_task = self._clean_task_for_query(task) if hasattr(self, '_clean_task_for_query') else task
        task_lower = clean_task.lower()

        comparison_markers = ['vs', 'versus', 'compare', 'comparison', 'difference between', 'vs.']
        is_comparison = any(m in task_lower for m in comparison_markers)

        research_markers = ['research on', 'deep dive', 'comprehensive', 'detailed analysis']
        is_deep_research = any(m in task_lower for m in research_markers)

        if not is_comparison and not is_deep_research:
            return None

        composite_skills = {'search-summarize-pdf-telegram', 'search-summarize-pdf-telegram-v2',
                           'content-research-writer', 'content-pipeline'}
        uses_composite = any(s.skill_name in composite_skills for s in steps)

        if not uses_composite and len(steps) >= 2:
            return None

        entities = self._extract_comparison_entities(clean_task)
        if not entities:
            entities = [task]

        available = {s.get('name', ''): s for s in skills}
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if registry:
                for sname, sdef in registry.loaded_skills.items():
                    if sname not in available:
                        available[sname] = {'name': sname}
        except Exception:
            pass

        decomposed = []

        task_wants_pdf = any(w in task_lower for w in ['pdf', 'report', 'document'])
        task_wants_telegram = 'telegram' in task_lower
        task_wants_slack = 'slack' in task_lower

        stop_words = {
            'vs', 'vs.', 'versus', 'compare', 'comparison', 'research', 'create',
            'generate', 'send', 'via', 'telegram', 'slack', 'pdf', 'report',
            'and', 'the', 'a', 'an', 'on', 'for', 'with', 'difference', 'between',
            'it', 'its', 'to', 'of', 'in', 'is', 'be', 'document', 'make', 'build',
        }
        entity_words = set()
        for e in entities:
            for w in e.lower().split():
                entity_words.add(w)

        topic_words = []
        for w in clean_task.lower().split():
            w_clean = w.strip('.,;!?')
            if len(w_clean) <= 2:
                continue
            if w_clean in stop_words:
                continue
            if w_clean in entity_words:
                continue
            topic_words.append(w_clean)
        topic_context = ' '.join(topic_words[:3])

        domain_context = ''
        if len(entities) >= 2:
            longest = max(entities, key=len)
            shortest = min(entities, key=len)
            if len(longest.split()) > len(shortest.split()):
                shortest_words = {w.lower() for w in shortest.split()}
                domain_words = [w for w in longest.split() if w.lower() not in shortest_words
                               and w.lower() not in {e.split()[0].lower() for e in entities}]
                domain_context = ' '.join(domain_words)

        if not domain_context:
            domain_context = topic_context

        for i, entity in enumerate(entities[:4]):
            entity_clean = entity.strip()
            if domain_context and domain_context.lower() not in entity_clean.lower():
                search_query = f'{entity_clean} {domain_context}'.strip()
            else:
                search_query = entity_clean
            search_params = {
                'query': search_query,
                'max_results': 5,
            }
            decomposed.append(ExecutionStep(
                skill_name='web-search',
                tool_name='search_web_tool',
                params=search_params,
                description=f'Research: {search_query}',
                output_key=f'research_{i}',
                depends_on=[],
            ))

        research_refs = ' '.join(
            f'${{research_{i}.results}}' for i in range(len(entities[:4]))
        )
        entity_names = ' vs '.join(e.strip() for e in entities[:4])

        if 'claude-cli-llm' in available:
            synth_skill = 'claude-cli-llm'
            synth_tool = 'generate_text_tool'
        elif 'summarize' in available:
            synth_skill = 'summarize'
            synth_tool = 'summarize_text_tool'
        else:
            synth_skill = 'claude-cli-llm'
            synth_tool = 'generate_text_tool'

        decomposed.append(ExecutionStep(
            skill_name=synth_skill,
            tool_name=synth_tool,
            params={
                'prompt': (
                    f'Create a detailed, structured comparison of {entity_names}. '
                    f'Format as a professional markdown report with these sections:\n'
                    f'# {entity_names} Comparison Report\n'
                    f'## Executive Summary\n'
                    f'## Feature Comparison\n'
                    f'| Feature | {" | ".join(e.strip() for e in entities[:4])} |\n'
                    f'## Pricing & Plans\n'
                    f'## Pros and Cons\n'
                    f'## Recommendation\n\n'
                    f'Use the following research data:\n{research_refs}'
                ),
            },
            description=f'Synthesize structured comparison: {entity_names}',
            output_key='synthesis',
            depends_on=list(range(len(entities[:4]))),
        ))

        if task_wants_pdf or task_wants_telegram:
            pdf_skill = 'simple-pdf-generator' if 'simple-pdf-generator' in available else 'document-converter'
            pdf_tool = 'generate_pdf_tool' if pdf_skill == 'simple-pdf-generator' else 'convert_to_pdf_tool'
            content_ref = '${synthesis.text}' if synth_skill == 'claude-cli-llm' else '${synthesis.summary}'
            decomposed.append(ExecutionStep(
                skill_name=pdf_skill,
                tool_name=pdf_tool,
                params={
                    'content': content_ref,
                    'title': f'{entity_names} Comparison Report',
                    'topic': entity_names,
                },
                description=f'Generate PDF report: {entity_names}',
                output_key='pdf_output',
                depends_on=[len(entities[:4])],
            ))

        if task_wants_telegram and 'telegram-sender' in available:
            decomposed.append(ExecutionStep(
                skill_name='telegram-sender',
                tool_name='send_telegram_file_tool',
                params={
                    'file_path': '${pdf_output.pdf_path}',
                    'caption': f' {entity_names} Comparison Report',
                },
                description=f'Send comparison report via Telegram',
                output_key='telegram_send',
                depends_on=[len(decomposed) - 1],
                optional=True,
            ))

        if task_wants_slack and 'slack' in available:
            decomposed.append(ExecutionStep(
                skill_name='slack',
                tool_name='send_slack_message_tool',
                params={
                    'file_path': '${pdf_output.pdf_path}',
                    'message': f' {entity_names} Comparison Report',
                },
                description=f'Send comparison report via Slack',
                output_key='slack_send',
                depends_on=[len(decomposed) - 1],
                optional=True,
            ))

        logger.info(f" Decomposed {len(steps)}-step composite plan → {len(decomposed)} granular steps")
        for i, step in enumerate(decomposed):
            logger.info(f"   Step {i+1}: {step.skill_name}/{step.tool_name} → {step.output_key}")

        return decomposed

    def _extract_comparison_entities(self, task: str) -> List[str]:
        """
        Extract entities being compared from task description.

        Examples:
            "Compare Paytm vs PhonePe" -> ["Paytm", "PhonePe"]
            "Research Paytm vs PhonePe vs Razorpay" -> ["Paytm", "PhonePe", "Razorpay"]
            "difference between React and Vue" -> ["React", "Vue"]
        """
        # Pattern 1: "X vs Y vs Z" or "X vs. Y"
        vs_match = re.split(r'\b(?:vs\.?|versus)\b', task, flags=re.IGNORECASE)
        if len(vs_match) >= 2:
            entities = []
            for part in vs_match:
                part = re.sub(r'^\s*(compare|research|analyze|research on|create|generate|send|make|build)\s+', '', part, flags=re.IGNORECASE)
                part = re.sub(r'\s*(comparison|compare|report|pdf|document|analysis|review|overview)\b.*$', '', part, flags=re.IGNORECASE)
                part = re.sub(r'\s*(via telegram|via slack|and send|and create|,.*$)\s*$', '', part, flags=re.IGNORECASE)
                part = part.strip().rstrip(',. ')
                if part and len(part) > 1:
                    entities.append(part)
            if len(entities) >= 2:
                return entities

        # Pattern 2: "difference between X and Y"
        between_match = re.search(
            r'(?:difference|comparison)\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*[,.]|\s+(?:and|create|generate|send|via))',
            task, flags=re.IGNORECASE
        )
        if between_match:
            return [between_match.group(1).strip(), between_match.group(2).strip()]

        # Pattern 3: "compare X and Y"
        compare_match = re.search(
            r'compare\s+(.+?)\s+and\s+(.+?)(?:\s*[,.]|\s+(?:create|generate|send|via)|$)',
            task, flags=re.IGNORECASE
        )
        if compare_match:
            return [compare_match.group(1).strip(), compare_match.group(2).strip()]

        return []


# =============================================================================
# TaskPlan (moved here for unified planning)
# =============================================================================

