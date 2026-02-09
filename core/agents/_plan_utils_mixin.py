"""AgenticPlanner Plan Utils Mixin - Tool schema, param building, fallbacks, metadata."""

import json
import logging
import re
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from .agentic_planner import _get_task_type

logger = logging.getLogger(__name__)


class PlanUtilsMixin:
    def _extract_tool_schema(self, tool_func, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameter schema from tool function docstring.

        Handles multiple docstring formats:
        1. Google style with 'containing' pattern: params: Dictionary containing 'location' key
        2. Dash-prefixed: - location (str, required): The location
        3. Indented Google style: location: The location to check

        Args:
            tool_func: The tool function
            tool_name: Name of the tool

        Returns:
            Dictionary with tool name, parameters, and description
        """
        import re

        schema = {
            'name': tool_name,
            'parameters': [],
            'description': ''
        }

        if not tool_func or not hasattr(tool_func, '__doc__') or not tool_func.__doc__:
            return schema

        docstring = tool_func.__doc__
        lines = docstring.split('\n')

        # Extract description (first non-empty line)
        for line in lines:
            stripped = line.strip()
            if stripped:
                schema['description'] = stripped
                break

        # Pattern 1: Extract from "Dictionary containing 'X' key" pattern
        # This is common in Jotty skills: params: Dictionary containing 'location' key
        containing_patterns = [
            r"containing\s+'(\w+)'\s+key",  # containing 'location' key
            r"containing\s+'(\w+)'",         # containing 'location'
            r"with\s+'(\w+)'\s+key",         # with 'location' key
            r'"(\w+)"\s+key',                # "location" key
        ]

        for pattern in containing_patterns:
            matches = re.findall(pattern, docstring, re.IGNORECASE)
            for param_name in matches:
                if param_name not in [p['name'] for p in schema['parameters']]:
                    schema['parameters'].append({
                        'name': param_name,
                        'type': 'str',
                        'required': True,
                        'description': f'The {param_name} parameter'
                    })

        # Pattern 2: Parse Args section (Google style and dash-prefixed)
        in_args = False
        current_param = None

        for line in lines:
            stripped = line.strip()

            if 'Args:' in line or 'Parameters:' in line:
                in_args = True
                continue

            if in_args:
                # End of Args section
                if stripped.startswith('Returns:') or stripped.startswith('Raises:') or (stripped.startswith('##') or stripped == ''):
                    if stripped.startswith('Returns:') or stripped.startswith('Raises:'):
                        break
                    if stripped == '' and current_param is None:
                        continue  # Empty line before parameters

                # Dash-prefixed: - location (str, required): Description
                if stripped.startswith('-'):
                    parts = stripped[1:].strip().split(':', 1)
                    if len(parts) == 2:
                        param_def = parts[0].strip()
                        desc = parts[1].strip()

                        param_name = param_def.split('(')[0].strip() if '(' in param_def else param_def.strip()

                        if param_name and param_name not in ['params', 'kwargs', 'args']:
                            if '(' in param_def:
                                type_info = param_def.split('(')[1].split(')')[0]
                                param_type = type_info.split(',')[0].strip()
                                required = 'required' in type_info.lower() or 'optional' not in type_info.lower()
                            else:
                                param_type = 'str'
                                required = True

                            if param_name not in [p['name'] for p in schema['parameters']]:
                                schema['parameters'].append({
                                    'name': param_name,
                                    'type': param_type,
                                    'required': required,
                                    'description': desc
                                })

                # Indented Google style: location (str): The location to check
                # Or: location: The location to check
                elif ':' in stripped and not stripped.startswith('#'):
                    parts = stripped.split(':', 1)
                    if len(parts) == 2:
                        param_def = parts[0].strip()
                        desc = parts[1].strip()

                        # Extract parameter name (before any type annotation in parentheses)
                        param_name = param_def.split('(')[0].strip() if '(' in param_def else param_def.strip()

                        # Skip generic 'params' dictionary entry - we handle it via containing pattern
                        if param_name and param_name not in ['params', 'kwargs', 'args', 'self']:
                            if '(' in param_def:
                                type_info = param_def.split('(')[1].split(')')[0]
                                param_type = type_info.split(',')[0].strip()
                                required = 'optional' not in type_info.lower()
                            else:
                                param_type = 'str'
                                required = True

                            if param_name not in [p['name'] for p in schema['parameters']]:
                                schema['parameters'].append({
                                    'name': param_name,
                                    'type': param_type,
                                    'required': required,
                                    'description': desc
                                })

        # Pattern 3: Look for common parameter patterns in the description
        # e.g., "requires a 'topic' parameter" or "the location parameter"
        param_mention_patterns = [
            r"'(\w+)'\s+(?:parameter|argument|key)",
            r"(\w+)\s+(?:parameter|argument|key)\s+(?:is|must be|should be)",
        ]

        for pattern in param_mention_patterns:
            matches = re.findall(pattern, docstring, re.IGNORECASE)
            for param_name in matches:
                if param_name.lower() not in ['the', 'a', 'an', 'this'] and param_name not in [p['name'] for p in schema['parameters']]:
                    schema['parameters'].append({
                        'name': param_name,
                        'type': 'str',
                        'required': True,
                        'description': f'The {param_name} parameter'
                    })

        return schema
    
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

        Args:
            task: Task description (may contain enrichment context)

        Returns:
            Clean task description (original request only)
        """
        # Markers that indicate enrichment context (to be stripped)
        context_markers = [
            '\n[Learning Context',         # From autonomous_agent learning injection
            '\n\n[Learning Context',        # With double newline prefix
            '\n[Multi-Perspective Analysis',
            '\n[Multi-Perspective',
            '\nLearned Insights:',
            '\n# Transferable Learnings',
            '\n# Q-Learning Lessons',
            '\n## Task Type Pattern',
            '\n## Role Advice',
            '\n## Meta-Learning Advice',
            '\n\n---\n',  # Common separator before context
            '\nBased on previous learnings:',
            '\nRecommended approach:',
            '\nPrevious success patterns:',
            '\n[Analysis]:',
            '\n[Consensus]:',
            '\n[Tensions]:',
            '\n[Blind Spots]:',
            '\n[Learned]',                  # From SwarmManager intelligence hints
        ]

        cleaned = task
        for marker in context_markers:
            if marker in cleaned:
                # Keep only the part before the marker
                cleaned = cleaned.split(marker)[0]

        return cleaned.strip()

    def _build_skill_params(self, skill_name: str, task: str, prev_output_key: Optional[str] = None, tool_name: str = None) -> Dict[str, Any]:
        """Build params for a skill by looking up its tool schema from registry."""
        params = {}
        prev_ref = f"${{{prev_output_key}}}" if prev_output_key else task

        # Special handling for write_file_tool - generate actual content
        if skill_name == 'file-operations' and tool_name == 'write_file_tool':
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
                            # For write_file_tool, try to get generated content
                            if tool_name == 'write_file_tool':
                                file_content = self._generate_file_content(task)
                                if file_content and file_content.get('content'):
                                    params[param_name] = file_content['content']
                                else:
                                    params[param_name] = f"# TODO: Generated content for: {clean_task}"
                            else:
                                params[param_name] = prev_ref
                        elif param_name in ['file_path', 'pdf_path', 'path', 'input_path']:
                            # For file creation, extract actual filename/path from task
                            if tool_name == 'write_file_tool':
                                import re
                                # Try to extract file path from task (supports directory and absolute paths)
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
                                extracted_filepath = None
                                for pattern in filepath_patterns:
                                    match = re.search(pattern, task, re.IGNORECASE)
                                    if match:
                                        extracted_filepath = match.group(1)
                                        break
                                params[param_name] = extracted_filepath or prev_ref
                                logger.debug(f"Extracted filepath: '{extracted_filepath}' from task")
                            else:
                                params[param_name] = prev_ref
                        elif param_name in ['max_results', 'limit', 'count']:
                            params[param_name] = 10
                        elif param_name in ['title', 'name']:
                            params[param_name] = clean_task[:50]
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
        # Priority: Python backends > HTML frontends (most code tasks are Python)
        # Only use .html when the task is EXPLICITLY about HTML/frontend
        file_ext = '.py'  # Default to Python
        _is_python_backend = any(w in task_lower for w in [
            'python', 'fastapi', 'flask', 'django', 'api', 'endpoint', 'server',
            'backend', 'script', 'class', 'function', 'module', '.py',
        ])
        if _is_python_backend:
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

    def _create_fallback_plan(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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

        TaskType = _get_task_type()

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
    ) -> 'ExecutionPlan':
        """
        Plan execution with enhanced metadata (ExecutionPlan).
        
        Works with both raw strings and TaskGraph.
        
        Args:
            task: Task description (str) or TaskGraph
            available_skills: Available skills (auto-discovers if None)
            max_steps: Maximum execution steps
            convert_to_agents: If True, also convert skills to AgentConfig for Conductor
            
        Returns:
            ExecutionPlan with steps and metadata
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
        TaskType = _get_task_type()
        
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
        
        # Create ExecutionPlan
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
        
        return ExecutionPlan(
            task_graph=task_graph,
            steps=steps,
            estimated_time=estimated_time,
            required_tools=required_tools,
            required_credentials=required_credentials,
            metadata=metadata
        )
    
    async def _convert_skills_to_agents(self, skills: List[Dict[str, Any]]):
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
    
    def _create_task_graph_from_string(self, task_string: str, task_type) -> Optional[Any]:
        """Create minimal TaskGraph from raw string."""
        if not TASK_GRAPH_AVAILABLE:
            return None
        
        from ..autonomous.intent_parser import TaskGraph
        return TaskGraph(
            task_type=task_type,
            workflow=None,
            metadata={'original_request': task_string}
        )


# =============================================================================
# ExecutionPlan (moved here for unified planning)
# =============================================================================

