"""
Step Processors — Parameter Resolution & Tool Output Processing
================================================================

Extracted from skill_plan_executor.py for decomposition.

ParameterResolver: Resolves template variables and applies sanity checks
to step parameters. Supports optional ToolSchema for type-aware auto-wiring.

ToolResultProcessor: Sanitizes and truncates tool outputs before they
enter the outputs dict. Uses JSON-aware truncation.

Usage::

    resolver = ParameterResolver(outputs)
    resolved = resolver.resolve(params, step, tool_schema=schema)

    processor = ToolResultProcessor()
    clean = processor.process(raw_result, elapsed_seconds=1.2)

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ParameterResolver:
    """
    Resolves template variables and applies sanity checks to step parameters.

    Supports optional ``ToolSchema`` for type-aware auto-wiring: when a schema
    is provided, missing required params are filled from previous step outputs
    by name/alias/type match, and aliases are resolved to canonical names.

    Usage::

        resolver = ParameterResolver(outputs)
        resolved = resolver.resolve(params, step)

        # With schema (preferred — enables auto-wiring + validation):
        resolved = resolver.resolve(params, step, tool_schema=schema)
    """

    # Content-like field names to look for in step output dicts
    _CONTENT_FIELDS = ('response', 'text', 'content', 'output', 'stdout', 'result')

    # Instruction-like prefixes that indicate bad content
    _INSTRUCTION_PREFIXES = (
        'filename:', 'parse ', "i'll help", "i'll create", "let me ",
        'create a ', 'here is', "here's", 'this is', 'the following',
        'i will ', 'we will ', 'step ', 'save ', 'write ',
    )

    _MAX_RESOLVE_DEPTH = 10

    def __init__(self, outputs: Dict[str, Any]):
        self._outputs = outputs

    @staticmethod
    def _is_template(value) -> bool:
        """Check if a value contains unresolved template references."""
        if not isinstance(value, str):
            return False
        return '${' in value or ('{' in value and '}' in value)

    def resolve(
        self,
        params: Dict[str, Any],
        step: Any = None,
        _depth: int = 0,
        tool_schema: Any = None,
    ) -> Dict[str, Any]:
        """Resolve template variables in parameters with sanity checks.

        When *tool_schema* (a ``ToolSchema`` instance) is provided:
        1. Aliases are resolved to canonical names first
        2. Templates and bare keys are substituted (existing behavior)
        3. Missing required params are auto-wired from outputs
        4. Validation errors are logged as warnings
        """
        if _depth > self._MAX_RESOLVE_DEPTH:
            logger.warning(f"Parameter resolution exceeded max depth ({self._MAX_RESOLVE_DEPTH}), returning as-is")
            return params

        # Phase 0: resolve aliases via schema (before template substitution)
        if tool_schema is not None and _depth == 0:
            params = tool_schema.resolve_aliases(params)

        # Phase -1: Direct resolution from inputs_needed (I/O contract)
        if step and _depth == 0 and hasattr(step, 'inputs_needed') and step.inputs_needed:
            for param_name, source in step.inputs_needed.items():
                if param_name in params and self._is_template(params[param_name]):
                    continue  # Has template — will be resolved by Phase 1
                if source.startswith('literal:'):
                    params[param_name] = source[8:]  # Strip "literal:" prefix
                else:
                    val = self.resolve_path(source)
                    if val != source:  # Successfully resolved
                        params[param_name] = val

        resolved = {}

        for key, value in params.items():
            if isinstance(value, str):
                value = self._substitute_templates(key, value)
                value = self._resolve_bare_keys(key, value)
                value = self._resolve_placeholder_strings(key, value)
                # Schema-driven coercion when available, legacy fallback otherwise
                if tool_schema is not None:
                    value = self._schema_coerce_param(key, value, tool_schema, step)
                else:
                    value = self._sanitize_command_param(key, value, step)
                    value = self._sanitize_path_param(key, value, step)
                    value = self._sanitize_content_param(key, value)
                resolved[key] = value
            elif isinstance(value, dict):
                resolved[key] = self.resolve(value, step, _depth + 1)
            elif isinstance(value, list):
                resolved[key] = [
                    self.resolve(item, step, _depth + 1) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value

        # Phase 2: auto-wire missing required params from outputs (scoped to dependencies)
        if tool_schema is not None and _depth == 0 and self._outputs:
            scoped_keys = None
            if step and hasattr(step, 'depends_on') and step.depends_on:
                scoped_keys = []
                for dep_idx in step.depends_on:
                    scoped_keys.append(f'step_{dep_idx}')
                    for k in self._outputs:
                        if k.startswith(f'step_{dep_idx}') or k == f'step_{dep_idx}':
                            scoped_keys.append(k)
                    # Also include custom output_key-based keys
                    if step and hasattr(step, 'inputs_needed') and step.inputs_needed:
                        for source in step.inputs_needed.values():
                            base_key = source.split('.')[0]
                            if base_key in self._outputs and base_key not in scoped_keys:
                                scoped_keys.append(base_key)
            resolved = tool_schema.auto_wire(resolved, self._outputs, scoped_keys=scoped_keys)

        # Phase 3: validate with coercion, apply coerced values
        if tool_schema is not None and _depth == 0:
            validation = tool_schema.validate(resolved, coerce=True)
            resolved.update(validation.coerced_params)
            for err in validation.errors:
                logger.warning(f"Param validation ({tool_schema.name}): {err}")

        return resolved

    def _substitute_templates(self, param_name: str, value: str) -> str:
        """Replace ${ref} and {ref} template variables."""
        def replacer(match, _param_name=param_name):
            ref_path = match.group(1)
            raw = self.resolve_path(ref_path)
            if raw.startswith('{'):
                extracted = self._smart_extract(raw, _param_name)
                if extracted is not None:
                    return extracted.replace('\\', '\\\\')
            return raw.replace('\\', '\\\\')

        # Always substitute ${ref} patterns (explicit template references)
        value = re.sub(r'\$\{([^}]+)\}', replacer, value)

        # Skip bare {ref} substitution for code content to protect f-strings
        _CODE_MARKERS = ('def ', 'class ', 'import ', 'from ', 'f"', "f'", 'async def ', 'if __name__')
        if not any(marker in value for marker in _CODE_MARKERS):
            value = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\}', replacer, value)

        # Aggregate unresolved research references
        if '${research_' in value or '{research_' in value:
            aggregated = self._aggregate_research_outputs()
            if aggregated:
                value = re.sub(r'\$\{research_\d+\.results\}', aggregated, value)

        return value

    def _smart_extract(self, json_str: str, param_name: str) -> Optional[str]:
        """Extract the right field from a JSON dict based on param context."""
        try:
            obj = json.loads(json_str)
            if not isinstance(obj, dict):
                return None

            # Direct match: param name exists as key
            if param_name in obj:
                val = obj[param_name]
                # Format list results (e.g. search results) into readable text
                if isinstance(val, list) and param_name in ('content', 'text', 'body'):
                    return self._format_list_results(val)
                return str(val)

            # Content-like params: prefer rich text fields
            if param_name in ('content', 'text', 'body'):
                for fk in self._CONTENT_FIELDS:
                    if fk in obj:
                        val = obj[fk]
                        if isinstance(val, list):
                            return self._format_list_results(val)
                        val_str = str(val)
                        if len(val_str) > 50 and not val_str.strip().startswith('{"success"'):
                            return val_str

                # Fallback for content: format 'results' list (search results, etc.)
                if 'results' in obj and isinstance(obj['results'], list):
                    formatted = self._format_list_results(obj['results'])
                    if formatted and len(formatted) > 50:
                        return formatted

            # URL param: extract first URL from nested search results
            if param_name in ('url', 'link', 'webpage_url', 'page_url'):
                # Direct match
                if 'url' in obj:
                    return str(obj['url'])
                if 'link' in obj:
                    return str(obj['link'])
                # Nested in results list (search results)
                if 'results' in obj and isinstance(obj['results'], list):
                    for item in obj['results']:
                        if isinstance(item, dict):
                            url = item.get('link') or item.get('url') or item.get('href', '')
                            if url and url.startswith('http'):
                                return str(url)

            # Common field mappings
            for fk in ('path', 'output', 'content', 'stdout', 'result', 'response'):
                if param_name.lower() in (fk, f'file_{fk}', f'{fk}_path') and fk in obj:
                    return str(obj[fk])

            # Path param: scan all outputs for most recent path
            if param_name == 'path':
                for k in reversed(list(self._outputs.keys())):
                    sd = self._outputs[k]
                    if isinstance(sd, dict) and 'path' in sd:
                        return str(sd['path'])

        except (ValueError, KeyError):
            pass
        return None

    @staticmethod
    def _format_list_results(items: list) -> str:
        """Format a list of result dicts into human-readable text.

        Handles search results, API responses, and other structured lists.
        Each item is formatted as a section with title, snippet, and URL.
        """
        if not items:
            return ""
        lines = []
        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                title = item.get('title', item.get('name', ''))
                snippet = item.get('snippet', item.get('description', item.get('text', '')))
                url = item.get('link', item.get('url', ''))
                parts = []
                if title:
                    parts.append(f"{i}. {title}")
                if snippet:
                    parts.append(f"   {snippet}")
                if url:
                    parts.append(f"   Source: {url}")
                if parts:
                    lines.append("\n".join(parts))
                else:
                    lines.append(f"{i}. {json.dumps(item, default=str)[:200]}")
            else:
                lines.append(f"{i}. {str(item)[:200]}")
        return "\n\n".join(lines)

    def _resolve_bare_keys(self, key: str, value: str) -> str:
        """Handle bare output keys (e.g. "step_2" without ${} wrapping)."""
        if value not in self._outputs or not isinstance(self._outputs[value], dict):
            return value

        step_data = self._outputs[value]

        if key in step_data:
            logger.info(f"Resolved bare output key '{key}={value}' → {str(step_data[key])[:100]}")
            return str(step_data[key])

        if key == 'path' and 'path' in step_data:
            logger.info(f"Resolved bare output key 'path={value}' → {step_data['path']}")
            return str(step_data['path'])

        if key in ('content', 'text'):
            for fk in self._CONTENT_FIELDS:
                if fk in step_data:
                    val = str(step_data[fk])[:50000]
                    logger.info(f"Resolved bare output key '{key}={value}' → {fk} ({len(val)} chars)")
                    return val

        if key == 'path':
            for k in reversed(list(self._outputs.keys())):
                sd = self._outputs[k]
                if isinstance(sd, dict) and 'path' in sd:
                    path_val = str(sd['path'])
                    logger.info(f"Resolved bare output key 'path={value}' → path from '{k}': {path_val[:100]}")
                    return path_val

        # Serialize whole dict as fallback
        serialized = json.dumps(step_data, default=str)[:8000]
        logger.info(f"Resolved bare output key '{key}={value}' → full dict ({len(serialized)} chars)")
        return serialized

    def _resolve_placeholder_strings(self, key: str, value: str) -> str:
        """Replace uppercase placeholder patterns like {CONTENT_FROM_STEP_1}."""
        if not re.search(r'\{[A-Z_]+\}', value) or not self._outputs:
            return value

        last_output = list(self._outputs.values())[-1]
        if isinstance(last_output, dict):
            replacement = json.dumps(last_output, default=str)[:8000]
        else:
            replacement = str(last_output)[:8000]
        replacement = replacement.replace('\\', '\\\\')
        value = re.sub(r'\{[A-Z_]+\}', replacement, value)
        logger.info(f"Resolved unrecognised placeholder in param '{key}' with last step output")
        return value

    def _sanitize_command_param(self, key: str, value: str, step: Any) -> str:
        """Detect 'command' params that are LLM output instead of shell commands."""
        if key != 'command' or len(value) <= 150:
            return value

        stripped = value.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and 'text' in obj:
                    for k in reversed(list(self._outputs.keys())):
                        sd = self._outputs[k]
                        if isinstance(sd, dict) and 'path' in sd and sd['path'].endswith('.py'):
                            value = f'python {sd["path"]}'
                            logger.info(f"Auto-fixed 'command' from LLM output → '{value}'")
                            return value
            except (ValueError, KeyError):
                pass
        elif value.count(' ') > 15:
            for k in reversed(list(self._outputs.keys())):
                sd = self._outputs[k]
                if isinstance(sd, dict) and 'path' in sd and sd['path'].endswith('.py'):
                    value = f'python {sd["path"]}'
                    logger.info(f"Auto-fixed 'command' from task description → '{value}'")
                    return value
        return value

    def _sanitize_path_param(self, key: str, value: str, step: Any) -> str:
        """Detect 'path' params that are content instead of filenames."""
        if key != 'path' or len(value) <= 200:
            return value

        found = None

        # Extract filename from the content itself
        m = re.search(
            r'(?:saved?|wrot?e?|created?|output)\s+(?:to|as|in)\s+["\']?'
            r'([a-zA-Z0-9_.-]+\.\w{1,5})', value, re.IGNORECASE,
        )
        if m:
            found = m.group(1)

        # Extract from step description
        if not found and step and getattr(step, 'description', None):
            m2 = re.search(
                r'(?:read|verify|check|open)\s+(?:the\s+)?["\']?'
                r'([a-zA-Z0-9_.-]+\.\w{1,5})', step.description, re.IGNORECASE,
            )
            if m2:
                found = m2.group(1)

        # Fallback: most recent file path from outputs
        if not found:
            for k in reversed(list(self._outputs.keys())):
                sd = self._outputs[k]
                if isinstance(sd, dict) and 'path' in sd:
                    p = str(sd['path'])
                    if len(p) < 200 and '.' in p:
                        found = p
                        break

        if found:
            logger.info(f"Auto-fixed 'path' from content → '{found}'")
            return found
        return value

    def _sanitize_content_param(self, key: str, value: str) -> str:
        """Detect 'content' params that resolved to wrong values."""
        if key != 'content':
            return value

        vs = value.strip()
        if not self._is_bad_content(vs):
            return value

        replacement = self._find_best_content()
        if replacement:
            logger.info(f"Auto-fixed 'content' from bad/short value → real content ({len(replacement)} chars)")
            return replacement
        return value

    def _schema_coerce_param(self, key: str, value: str, tool_schema, step) -> str:
        """Schema-driven parameter coercion using ToolParam type hints.

        Delegates to TypeCoercer for type-aware coercion when a matching
        ToolParam exists, falling back to legacy sanitizers for specific
        param categories (command, path, content).
        """
        from Jotty.core.agents._execution_types import TypeCoercer

        tp = tool_schema.get_param(key)

        # Path params: validate path-like, fallback to output search
        if tp and tp.type_hint in ('path', 'file_path'):
            coerced, error = TypeCoercer.coerce(value, 'path')
            if error:
                # Path looks like content — try to find real path from outputs
                found = self._find_path_from_outputs(step)
                if found:
                    logger.info(f"Schema coercion: replaced bad path with '{found}' from outputs")
                    return found
                # Still fall back to legacy sanitizer
                return self._sanitize_path_param(key, value, step)
            return coerced

        # Command params with suspiciously long values
        if key == 'command' and len(value) > 150:
            return self._sanitize_command_param(key, value, step)

        # Content params: check for bad content
        if key == 'content' and self._is_bad_content(value.strip()):
            return self._sanitize_content_param(key, value)

        # All other typed params: coerce via TypeCoercer
        if tp and tp.type_hint and tp.type_hint.lower() not in ('str', 'string', ''):
            coerced, error = TypeCoercer.coerce(value, tp.type_hint)
            if error:
                logger.debug(f"Schema coercion warning for '{key}': {error}")
                return value  # Keep original on failure
            return coerced

        return value

    def _find_path_from_outputs(self, step=None) -> Optional[str]:
        """Find the most recent valid file path from previous step outputs.

        Searches outputs in reverse order for 'path' keys containing
        short, extension-bearing strings (actual file paths).
        """
        for k in reversed(list(self._outputs.keys())):
            sd = self._outputs[k]
            if isinstance(sd, dict) and 'path' in sd:
                p = str(sd['path'])
                if len(p) < 300 and '.' in p and '\n' not in p:
                    return p
        return None

    def _is_bad_content(self, s: str) -> bool:
        """Check if a string looks like bad/wrong content for a file write."""
        if len(s) < 80:
            return True
        if s.startswith('{"success"') and '"bytes_written"' in s and len(s) < 300:
            return True
        lower = s.lower()[:80]
        if len(s) < 300 and any(lower.startswith(p) or lower.strip().startswith(p) for p in self._INSTRUCTION_PREFIXES):
            return True
        return False

    def _find_best_content(self) -> Optional[str]:
        """Find the best real content from previous outputs."""
        best, best_len = None, 0
        for k in reversed(list(self._outputs.keys())):
            sd = self._outputs[k]
            if isinstance(sd, dict):
                for cf in self._CONTENT_FIELDS:
                    if cf in sd:
                        c = str(sd[cf])
                        if len(c) > best_len and not c.startswith('{"success"') and not self._is_bad_content(c):
                            best, best_len = c, len(c)
            elif isinstance(sd, str) and len(sd) > best_len and not self._is_bad_content(sd):
                best, best_len = sd, len(sd)
        return best if best and best_len > 100 else None

    def resolve_path(self, path: str) -> str:
        """Resolve a dotted path like 'step_key.field' from outputs."""
        parts = path.split('.')
        value = self._outputs

        for part in parts:
            if value is None:
                break
            if '[' in part:
                arr_key = part.split('[')[0]
                try:
                    idx = int(part.split('[')[1].split(']')[0])
                    if isinstance(value, dict):
                        value = value.get(arr_key, [])
                    if isinstance(value, list) and idx < len(value):
                        value = value[idx]
                    else:
                        value = None
                        break
                except (ValueError, IndexError):
                    value = None
                    break
            else:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break

        if value is None:
            return self._resolve_missing_path(path)

        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        return str(value)

    def _resolve_missing_path(self, path: str) -> str:
        """Step-scoped fallback resolution for missing output keys.

        Resolution priority:
        1. Exact step key match (step_N in outputs)
        2. Adjacent step (N-1) for field lookup
        3. Return unresolved path (not random content) so semantic fallback can handle it
        """
        step_match = re.match(r'^step_(\d+)(?:\.(.+))?$', path)
        if not step_match or not self._outputs:
            return path

        step_idx = int(step_match.group(1))
        field = step_match.group(2)

        # Strategy 1: exact step key match
        exact_key = f'step_{step_idx}'
        if exact_key in self._outputs:
            v = self._outputs[exact_key]
            if isinstance(v, dict):
                if field and field in v:
                    logger.info(f"Resolved missing '{path}' -> '{exact_key}'.{field}")
                    return str(v[field])
                if not field:
                    return json.dumps(v, default=str)
            elif not field:
                return str(v)

        # Strategy 2: try adjacent step (N-1) only
        adj_key = f'step_{step_idx - 1}' if step_idx > 0 else None
        if adj_key and adj_key in self._outputs:
            v = self._outputs[adj_key]
            if isinstance(v, dict) and field and field in v:
                cand = str(v[field])
                if cand.strip():
                    logger.info(f"Resolved missing '{path}' -> adjacent '{adj_key}'.{field}")
                    return cand

        # Strategy 3: return unresolved (no broad scan)
        logger.warning(f"Could not resolve '{path}' — no matching step output found")
        return path

    def _aggregate_research_outputs(self) -> str:
        """Aggregate all research_* outputs into a formatted string."""
        parts = []
        for key in sorted(self._outputs.keys()):
            if key.startswith('research_') and isinstance(self._outputs[key], dict):
                result = self._outputs[key]
                query = result.get('query', key)
                results_list = result.get('results', [])
                if results_list:
                    part = f"\n## Research: {query}\n"
                    for i, r in enumerate(results_list[:5], 1):
                        title = r.get('title', 'Untitled') if isinstance(r, dict) else str(r)
                        snippet = r.get('snippet', '') if isinstance(r, dict) else ''
                        url = r.get('url', '') if isinstance(r, dict) else ''
                        part += f"\n### {i}. {title}\n{snippet}\n"
                        if url:
                            part += f"Source: {url}\n"
                    parts.append(part)
        return '\n'.join(parts) if parts else ''


class ToolResultProcessor:
    """
    Sanitizes and truncates tool outputs before they enter the outputs dict.

    Uses JSON-aware truncation: preserves ALL keys (paths, status, metadata)
    and only truncates large VALUES, preventing broken downstream references.

    Usage::

        processor = ToolResultProcessor()
        clean = processor.process(raw_result, elapsed_seconds=1.2)
    """

    _DEFAULT_MAX_SIZE = 50_000  # character budget
    # Key names that typically contain binary/base64 data
    _BINARY_KEY_PATTERNS = ('screenshot', 'image', 'base64', 'binary', 'png', 'jpeg', 'pdf_data')

    def process(self, result: dict, max_size: int = 0, elapsed: float = 0.0) -> dict:
        """Main entry — sanitize a tool result dict."""
        if not isinstance(result, dict):
            result = {'output': result}
        budget = max_size or self._DEFAULT_MAX_SIZE
        result = self._convert_sets(result)
        result = self._strip_binary(result)
        result = self._truncate_preserving_keys(result, budget)
        if elapsed > 0:
            result['_execution_time_ms'] = round(elapsed * 1000, 1)
        return result

    def _truncate_preserving_keys(self, data: dict, max_chars: int) -> dict:
        """
        JSON-aware truncation: keep ALL keys, truncate only large VALUES.

        Naive char-slicing (result[:10000]) breaks JSON and loses critical
        metadata (paths, exit codes). This preserves structure.
        """
        # Pass 1: separate small values (keep intact) from large values
        small_items = {}
        large_items = []
        small_total = 0

        for key, value in data.items():
            if isinstance(value, str):
                size = len(value)
            elif isinstance(value, dict):
                size = len(json.dumps(value, default=str))
            else:
                size = len(str(value))

            if size <= 500:
                small_items[key] = value
                small_total += size
            else:
                large_items.append((key, value, size))

        # If no large items, or everything fits, return as-is
        total = small_total + sum(s for _, _, s in large_items)
        if total <= max_chars or not large_items:
            return data

        # Pass 2: distribute remaining budget across large values proportionally
        remaining = max(1000, max_chars - small_total - 200)
        result = dict(small_items)
        total_large = sum(s for _, _, s in large_items)

        for key, value, size in large_items:
            share = max(200, int(remaining * (size / total_large)))
            value_str = value if isinstance(value, str) else json.dumps(value, default=str)
            if len(value_str) > share:
                result[key] = value_str[:share] + f"\n... [truncated {len(value_str) - share} chars]"
            else:
                result[key] = value
        return result

    def _strip_binary(self, result: dict) -> dict:
        """Replace base64 screenshot data with a size placeholder."""
        for key in list(result.keys()):
            value = result[key]
            if isinstance(value, str) and len(value) > 1000:
                key_lower = key.lower()
                # Fast path: key name suggests binary data
                if any(pat in key_lower for pat in self._BINARY_KEY_PATTERNS):
                    result[key] = f"[binary data: {len(value)} chars]"
                    continue
                # Slow path: high-diversity alphanumeric = likely base64
                sample = value[:200]
                if re.match(r'^[A-Za-z0-9+/=]{200}$', sample) and len(set(sample)) > 20:
                    result[key] = f"[binary data: {len(value)} chars]"
            elif isinstance(value, dict):
                result[key] = self._strip_binary(value)
        return result

    @staticmethod
    def _format_search_results(results: list, query: str = "") -> str:
        """Format search result dicts into a readable report.

        Used by the content-from-previous-steps wiring in execute_step()
        to turn raw search output into file-writable content.
        """
        lines = []
        if query:
            lines.append(f"# Search Results: {query}")
            lines.append(f"# {len(results)} results found\n")
        for i, item in enumerate(results, 1):
            if isinstance(item, dict):
                title = item.get('title', item.get('name', f'Result {i}'))
                snippet = item.get('snippet', item.get('description', ''))
                url = item.get('link', item.get('url', ''))
                lines.append(f"## {i}. {title}")
                if snippet:
                    lines.append(f"{snippet}")
                if url:
                    lines.append(f"Source: {url}")
                lines.append("")
            else:
                lines.append(f"{i}. {str(item)[:500]}\n")
        return "\n".join(lines)

    def _convert_sets(self, result: dict) -> dict:
        """Recursively convert set -> list for JSON safety."""
        cleaned = {}
        for key, value in result.items():
            if isinstance(value, set):
                cleaned[key] = sorted(str(v) for v in value)
            elif isinstance(value, dict):
                cleaned[key] = self._convert_sets(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._convert_sets(item) if isinstance(item, dict)
                    else sorted(str(v) for v in item) if isinstance(item, set)
                    else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned


# =============================================================================
# SEMANTIC PARAMETER RESOLVER (DSPy-based fallback)
# =============================================================================

try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


if _DSPY_AVAILABLE:
    class ParameterMatchSignature(dspy.Signature):
        """Match a missing parameter to the best available data source by meaning."""
        parameter_name: str = dspy.InputField(desc="Name of missing parameter")
        parameter_purpose: str = dspy.InputField(desc="What the tool needs this for")
        available_outputs: str = dspy.InputField(desc="JSON of available step outputs with previews")

        best_source: str = dspy.OutputField(desc="Best source key (e.g., 'step_0.text') or 'NO_MATCH'")
        confidence: float = dspy.OutputField(desc="0.0-1.0 confidence in the match")
else:
    ParameterMatchSignature = None  # type: ignore[assignment,misc]


class SemanticParamResolver:
    """DSPy-based fallback for when I/O contracts and templates fail.

    Only called as a last-resort for high-stakes params (content/text/body).
    Uses fast LM (Haiku) to minimize latency.
    """

    _CONFIDENCE_THRESHOLD = 0.7
    _HIGH_STAKES_PARAMS = ('content', 'text', 'body', 'message')

    def __init__(self):
        self._matcher = None

    def _ensure_matcher(self):
        """Lazily initialize DSPy matcher to avoid import-time cost."""
        if self._matcher is None and _DSPY_AVAILABLE and ParameterMatchSignature is not None:
            self._matcher = dspy.Predict(ParameterMatchSignature)

    def resolve(self, param_name: str, step_desc: str, outputs: Dict[str, Any]) -> Optional[Any]:
        """Attempt semantic match. Returns value if confidence > threshold, else None.

        Args:
            param_name: Name of missing parameter
            step_desc: Description of what the step does
            outputs: All available step outputs

        Returns:
            Resolved value if confident match found, else None
        """
        if not _DSPY_AVAILABLE or param_name not in self._HIGH_STAKES_PARAMS:
            return None

        self._ensure_matcher()
        if self._matcher is None:
            return None

        try:
            previews = {}
            for k, v in outputs.items():
                if isinstance(v, dict):
                    previews[k] = {
                        sk: str(sv)[:200] for sk, sv in v.items()
                        if isinstance(sv, (str, int, float, bool))
                    }

            if not previews:
                return None

            result = self._matcher(
                parameter_name=param_name,
                parameter_purpose=step_desc,
                available_outputs=json.dumps(previews, default=str)
            )

            confidence = float(result.confidence) if result.confidence else 0.0
            best_source = str(result.best_source).strip()

            if confidence >= self._CONFIDENCE_THRESHOLD and best_source != 'NO_MATCH':
                # Resolve the path from outputs
                resolver = ParameterResolver(outputs)
                val = resolver.resolve_path(best_source)
                if val != best_source:  # Successfully resolved
                    logger.info(f"Semantic resolver matched '{param_name}' -> '{best_source}' "
                               f"(confidence={confidence:.2f})")
                    return val
        except Exception as e:
            logger.debug(f"Semantic param resolver failed for '{param_name}': {e}")

        return None


__all__ = [
    'ParameterResolver',
    'ToolResultProcessor',
    'SemanticParamResolver',
    'ParameterMatchSignature',
]
