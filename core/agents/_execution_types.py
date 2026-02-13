"""
Shared execution types - breaks circular dependencies.

This module provides ExecutionStep, ExecutionStepSchema, TaskType, and
ExecutionResult in a dependency-free location so that both agentic_planner.py
and auto_agent.py can import them without circular imports.

Also provides:
- FileReference: lazy handle for large outputs spilled to disk
- SwarmArtifactStore: tag-queryable artifact registry (replaces flat outputs dict)
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# TASK TYPE (Enum)
# =============================================================================


class TaskType(Enum):
    """Inferred task types."""
    RESEARCH = "research"
    COMPARISON = "comparison"
    CREATION = "creation"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    UNKNOWN = "unknown"


# =============================================================================
# EXECUTION STEP (Dataclass)
# =============================================================================


@dataclass
class ExecutionStep:
    """A step in the execution plan."""
    skill_name: str
    tool_name: str
    params: Dict[str, Any]
    description: str
    depends_on: List[int] = field(default_factory=list)
    output_key: str = ""
    optional: bool = False
    verification: str = ""
    fallback_skill: str = ""


# =============================================================================
# TOOL SCHEMA (Typed parameter definitions for tools)
# =============================================================================


@dataclass
class ToolParam:
    """Typed parameter definition for a tool function.

    Built from multiple sources (decorator, docstring, metadata).
    Used by ToolSchema for validation and auto-wiring.
    """
    name: str
    type_hint: str = "str"
    required: bool = True
    description: str = ""
    default: Any = None
    aliases: List[str] = field(default_factory=list)
    reserved: bool = False


class TypeCoercer:
    """Schema-driven type coercion. Replaces heuristic sanitizers.

    Converts string values to their declared types based on ToolParam.type_hint.
    Each coercion method returns (coerced_value, error_or_None).
    """

    _COERCERS: Dict[str, str] = {
        'int': '_coerce_int', 'integer': '_coerce_int',
        'float': '_coerce_float', 'number': '_coerce_float',
        'bool': '_coerce_bool', 'boolean': '_coerce_bool',
        'list': '_coerce_list', 'array': '_coerce_list',
        'dict': '_coerce_dict', 'object': '_coerce_dict',
        'path': '_coerce_path', 'file_path': '_coerce_path',
    }

    @classmethod
    def coerce(cls, value: Any, type_hint: str) -> Tuple[Any, Optional[str]]:
        """Coerce *value* to the type described by *type_hint*.

        Returns (coerced_value, error_or_None).  For unknown or 'str'
        type hints the value is returned unchanged.
        """
        if type_hint is None or type_hint.lower() in ('str', 'string', ''):
            return value, None

        method_name = cls._COERCERS.get(type_hint.lower())
        if method_name is None:
            return value, None  # Unknown type — pass through

        method = getattr(cls, method_name)
        try:
            return method(value)
        except Exception as exc:
            return value, f"Cannot coerce {repr(value)[:80]} to {type_hint}: {exc}"

    # -- individual coercers --------------------------------------------------

    @staticmethod
    def _coerce_int(value: Any) -> Tuple[Any, Optional[str]]:
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        if isinstance(value, float) and value == int(value):
            return int(value), None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit() or (stripped.startswith('-') and stripped[1:].isdigit()):
                return int(stripped), None
            # Try float-then-int for "42.0"
            try:
                f = float(stripped)
                if f == int(f):
                    return int(f), None
            except ValueError:
                pass
            return value, f"Cannot coerce {repr(value)[:60]} to int: not numeric"
        return value, f"Cannot coerce type {type(value).__name__} to int"

    @staticmethod
    def _coerce_float(value: Any) -> Tuple[Any, Optional[str]]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
        if isinstance(value, str):
            try:
                return float(value.strip()), None
            except ValueError:
                return value, f"Cannot coerce {repr(value)[:60]} to float"
        return value, f"Cannot coerce type {type(value).__name__} to float"

    @staticmethod
    def _coerce_bool(value: Any) -> Tuple[Any, Optional[str]]:
        if isinstance(value, bool):
            return value, None
        if isinstance(value, (int, float)):
            return bool(value), None
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ('true', 'yes', '1', 'on'):
                return True, None
            if lower in ('false', 'no', '0', 'off'):
                return False, None
            return value, f"Cannot coerce {repr(value)[:60]} to bool"
        return value, f"Cannot coerce type {type(value).__name__} to bool"

    @staticmethod
    def _coerce_list(value: Any) -> Tuple[Any, Optional[str]]:
        if isinstance(value, list):
            return value, None
        if isinstance(value, str):
            stripped = value.strip()
            # JSON array
            if stripped.startswith('['):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed, None
                except (json.JSONDecodeError, ValueError):
                    pass
            # Comma-separated
            if ',' in stripped:
                return [item.strip() for item in stripped.split(',') if item.strip()], None
            return [stripped] if stripped else [], None
        if isinstance(value, (tuple, set)):
            return list(value), None
        return value, f"Cannot coerce type {type(value).__name__} to list"

    @staticmethod
    def _coerce_dict(value: Any) -> Tuple[Any, Optional[str]]:
        if isinstance(value, dict):
            return value, None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith('{'):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        return parsed, None
                except (json.JSONDecodeError, ValueError):
                    pass
            return value, f"Cannot coerce {repr(value)[:60]} to dict: not valid JSON object"
        return value, f"Cannot coerce type {type(value).__name__} to dict"

    @staticmethod
    def _coerce_path(value: Any) -> Tuple[Any, Optional[str]]:
        """Validate that value looks like a file path (not content).

        Rejects strings that are too long, contain newlines, or lack
        path separators/extensions — indicators of content being
        mistakenly placed in a path parameter.
        """
        if not isinstance(value, str):
            return str(value), None
        stripped = value.strip()
        if not stripped:
            return value, "Empty path"
        if '\n' in stripped:
            return value, "Path contains newlines — likely content, not a path"
        if len(stripped) > 500:
            return value, f"Path too long ({len(stripped)} chars) — likely content, not a path"
        # Must have a separator or extension to look path-like
        if '/' not in stripped and '\\' not in stripped and '.' not in stripped:
            # Single word with no extension — could be a bare name, allow it
            if ' ' in stripped:
                return value, "Path contains spaces but no separator/extension — likely content"
        return stripped, None


@dataclass
class ToolValidationResult:
    """Structured result from ToolSchema.validate().

    Carries coerced parameter values alongside validation errors,
    so callers can apply fixes in one pass.
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    coerced_params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def error_summary(self) -> str:
        """One-line summary of all errors for logging."""
        if not self.errors:
            return "OK"
        return "; ".join(self.errors)


class ToolSchema:
    """Typed schema for a tool function.

    Replaces fragile docstring-regex parsing with structured param
    definitions.  Built from multiple sources (highest priority first):

    1. ``@tool_wrapper(required_params=[...])`` decorator inspection
    2. ``ToolMetadata.parameters`` (if defined on SkillDefinition)
    3. Docstring parsing (regex fallback)

    Usage::

        schema = ToolSchema.from_tool_function(my_tool, 'my_tool')
        errors = schema.validate({'query': 'test'})
        wired  = schema.auto_wire({'query': 'test'}, previous_outputs)
    """

    # Centralized alias map (single source of truth, shared with tool_helpers)
    try:
        from Jotty.core.foundation.data_structures import DEFAULT_PARAM_ALIASES
    except ImportError:
        DEFAULT_PARAM_ALIASES = {}

    _BUILTIN_ALIASES: Dict[str, List[str]] = {
        'path': ['file_path', 'filepath', 'filename', 'file_name', 'file'],
        'content': ['text', 'data', 'body', 'file_content'],
        'command': ['cmd', 'shell_command', 'shell_cmd'],
        'script': ['script_content', 'script_code', 'code', 'python_code'],
        'query': ['search_query', 'q', 'search', 'question'],
        'url': ['link', 'href', 'website', 'page_url'],
        'message': ['msg', 'text_message'],
        'timeout': ['time_limit', 'max_time'],
    }
    _ALIASES: Dict[str, List[str]] = {**DEFAULT_PARAM_ALIASES, **_BUILTIN_ALIASES}

    # Content-like fields used by auto_wire to find rich text in outputs
    _CONTENT_FIELDS = ('response', 'text', 'content', 'output', 'stdout', 'result')

    def __init__(
        self,
        name: str,
        description: str = "",
        params: Optional[List[ToolParam]] = None,
    ):
        self.name = name
        self.description = description
        self.params: List[ToolParam] = params or []

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_tool_function(cls, func: Callable, tool_name: str) -> ToolSchema:
        """Build schema by inspecting a tool function.

        Priority:
        1. ``func._required_params`` set by ``@tool_wrapper``
        2. Docstring parsing (type + description extraction)
        3. Alias injection from ``_ALIASES``
        """
        schema = cls(name=tool_name)

        # Extract description from docstring first line
        doc = inspect.getdoc(func) or ""
        if doc:
            schema.description = doc.split("\n")[0].strip()

        # Source 1: decorator-stashed required_params (most reliable)
        required_names: List[str] = getattr(func, '_required_params', []) or []

        # Source 2: parse docstring for param details
        doc_params = cls._parse_docstring_params(doc)

        # Merge: decorator params are required; docstring adds descriptions/types
        seen: Set[str] = set()
        for pname in required_names:
            dp = doc_params.get(pname, {})
            schema.params.append(ToolParam(
                name=pname,
                type_hint=dp.get('type', 'str'),
                required=True,
                description=dp.get('description', f'The {pname} parameter'),
                aliases=cls._ALIASES.get(pname, []),
            ))
            seen.add(pname)

        # Add optional params from docstring that weren't in required list
        for pname, info in doc_params.items():
            if pname not in seen:
                schema.params.append(ToolParam(
                    name=pname,
                    type_hint=info.get('type', 'str'),
                    required=info.get('required', False),
                    description=info.get('description', ''),
                    aliases=cls._ALIASES.get(pname, []),
                ))

        return schema

    @classmethod
    def from_metadata(cls, tool_name: str, metadata_dict: Dict[str, Any]) -> ToolSchema:
        """Build from a ToolMetadata.parameters JSON-Schema dict."""
        schema = cls(
            name=tool_name,
            description=metadata_dict.get('description', ''),
        )
        properties = metadata_dict.get('parameters', {}).get('properties', {})
        required_list = metadata_dict.get('parameters', {}).get('required', [])

        for pname, pdef in properties.items():
            schema.params.append(ToolParam(
                name=pname,
                type_hint=pdef.get('type', 'str'),
                required=pname in required_list,
                description=pdef.get('description', ''),
                aliases=cls._ALIASES.get(pname, []),
            ))
        return schema

    # Reserved params injected by the executor, hidden from LLM prompts
    _RESERVED_PARAMS: frozenset = frozenset({
        '_status_callback', '_task_context', '_outputs',
    })

    # -- validation -----------------------------------------------------------

    def validate(self, params: Dict[str, Any], coerce: bool = False) -> ToolValidationResult:
        """Validate params against schema with optional type coercion.

        Args:
            params: Parameter dict to validate.
            coerce: When True, attempts TypeCoercer conversion for each
                    param with a non-str type_hint. Successful coercions
                    are placed in ``result.coerced_params``.

        Returns:
            ToolValidationResult with errors, coerced values, and warnings.
            Backward-compatible: ``bool(result.errors)`` works like the
            old ``List[str]`` return.
        """
        result = ToolValidationResult()

        for tp in self.params:
            if tp.reserved:
                continue
            # Check presence (canonical name + aliases)
            found_key = None
            if tp.name in params:
                found_key = tp.name
            else:
                for alias in tp.aliases:
                    if alias in params:
                        found_key = alias
                        break

            if found_key is None:
                if tp.required:
                    result.errors.append(f"Missing required parameter: {tp.name}")
                continue

            # Type coercion when requested
            if coerce and tp.type_hint and tp.type_hint.lower() not in ('str', 'string', ''):
                value = params[found_key]
                coerced, error = TypeCoercer.coerce(value, tp.type_hint)
                if error:
                    result.errors.append(f"Type error for '{tp.name}': {error}")
                elif coerced is not value:
                    result.coerced_params[tp.name] = coerced

        result.valid = len(result.errors) == 0
        return result

    def get_param(self, name: str) -> Optional[ToolParam]:
        """Look up a ToolParam by canonical name or alias."""
        for tp in self.params:
            if tp.name == name:
                return tp
            if name in tp.aliases:
                return tp
        return None

    def get_llm_visible_params(self) -> List[ToolParam]:
        """Return params visible to LLM planners (excludes reserved)."""
        return [p for p in self.params if not p.reserved and p.name not in self._RESERVED_PARAMS]

    def resolve_aliases(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter aliases to canonical names (in-place + return)."""
        for tp in self.params:
            if tp.name not in params:
                for alias in tp.aliases:
                    if alias in params:
                        params[tp.name] = params.pop(alias)
                        break
        return params

    # -- auto-wiring ----------------------------------------------------------

    def auto_wire(
        self,
        params: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fill missing required params from previous step outputs.

        Strategy for each missing param:
        1. Exact name match in most recent output dict
        2. Alias match in most recent output dict
        3. Content-field match for content/text params
        4. Most recent path for path params
        """
        wired = dict(params)
        reversed_keys = list(reversed(list(outputs.keys())))

        for tp in self.params:
            if not tp.required or tp.name in wired:
                continue

            # Check aliases too
            alias_present = any(a in wired for a in tp.aliases)
            if alias_present:
                continue

            value = self._find_in_outputs(tp, outputs, reversed_keys)
            if value is not None:
                wired[tp.name] = value
                logger.debug(f"Auto-wired '{tp.name}' from outputs ({str(value)[:60]})")

        return wired

    def _find_in_outputs(
        self,
        tp: ToolParam,
        outputs: Dict[str, Any],
        reversed_keys: List[str],
    ) -> Optional[Any]:
        """Find a matching value for *tp* in previous step outputs."""
        names_to_check = [tp.name] + tp.aliases

        # Strategy 1+2: exact name or alias in any output (most recent first)
        for key in reversed_keys:
            out = outputs[key]
            if not isinstance(out, dict):
                continue
            for candidate_name in names_to_check:
                if candidate_name in out:
                    val = out[candidate_name]
                    if val is not None and str(val).strip():
                        return val

        # Strategy 3: content-field match for content/text/body params
        if tp.name in ('content', 'text', 'body', 'message'):
            for key in reversed_keys:
                out = outputs[key]
                if not isinstance(out, dict):
                    continue
                for cf in self._CONTENT_FIELDS:
                    if cf in out:
                        val = str(out[cf])
                        if len(val) > 50:
                            return val

        # Strategy 4: path fallback
        if tp.name in ('path', 'file_path', 'input_path'):
            for key in reversed_keys:
                out = outputs[key]
                if isinstance(out, dict) and 'path' in out:
                    return out['path']

        return None

    # -- introspection --------------------------------------------------------

    @property
    def required_param_names(self) -> List[str]:
        """Return names of all required parameters."""
        return [p.name for p in self.params if p.required]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM planner prompts (excludes reserved params)."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': [
                {
                    'name': p.name,
                    'type': p.type_hint,
                    'required': p.required,
                    'description': p.description,
                }
                for p in self.get_llm_visible_params()
            ],
        }

    def __repr__(self) -> str:
        req = ", ".join(self.required_param_names)
        return f"ToolSchema({self.name}, required=[{req}])"

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _parse_docstring_params(docstring: str) -> Dict[str, Dict[str, Any]]:
        """Parse parameter info from a Google-style docstring.

        Returns ``{param_name: {'type': ..., 'description': ..., 'required': bool}}``.
        """
        params: Dict[str, Dict[str, Any]] = {}
        if not docstring:
            return params

        lines = docstring.split('\n')
        in_args = False

        for line in lines:
            stripped = line.strip()
            if stripped in ('Args:', 'Parameters:'):
                in_args = True
                continue
            if in_args and stripped and (stripped.startswith('Returns:') or stripped.startswith('Raises:')):
                break

            if not in_args:
                continue

            # "Dictionary containing 'X' key" patterns
            for m in re.finditer(r"containing\s+'(\w+)'", stripped, re.I):
                pname = m.group(1)
                if pname not in params:
                    params[pname] = {'type': 'str', 'required': True, 'description': f'The {pname} parameter'}

            # Dash-prefixed: - location (str, required): Description
            if stripped.startswith('-'):
                parts = stripped[1:].strip().split(':', 1)
                if len(parts) == 2:
                    param_def = parts[0].strip()
                    desc = parts[1].strip()
                    pname = param_def.split('(')[0].strip()
                    if pname and pname not in ('params', 'kwargs', 'args', 'self'):
                        ptype = 'str'
                        required = True
                        if '(' in param_def:
                            type_info = param_def.split('(')[1].split(')')[0]
                            ptype = type_info.split(',')[0].strip()
                            required = 'optional' not in type_info.lower()
                        params[pname] = {'type': ptype, 'required': required, 'description': desc}

            # Indented Google style: location (str): Description
            elif ':' in stripped and not stripped.startswith('#') and not stripped.startswith('Returns'):
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    param_def = parts[0].strip()
                    desc = parts[1].strip()
                    pname = param_def.split('(')[0].strip()
                    if pname and pname not in ('params', 'kwargs', 'args', 'self') and len(pname) < 30:
                        ptype = 'str'
                        required = True
                        if '(' in param_def:
                            type_info = param_def.split('(')[1].split(')')[0]
                            ptype = type_info.split(',')[0].strip()
                            required = 'optional' not in type_info.lower()
                        params[pname] = {'type': ptype, 'required': required, 'description': desc}

        return params


# =============================================================================
# AGENT I/O SCHEMA (Typed input/output definitions for agents)
# =============================================================================


class AgentIOSchema:
    """Typed input/output contract for an agent.

    Built from a DomainAgent's DSPy Signature, this exposes what fields
    an agent accepts as input and what it produces as output.  Used by
    orchestration layers (relay, pipeline) to auto-wire agent outputs
    to the next agent's inputs.

    Usage::

        schema_a = agent_a.get_io_schema()
        schema_b = agent_b.get_io_schema()

        # Check compatibility
        mapping = schema_a.wire_to(schema_b)
        # {'analysis': 'text', 'sources': 'context'}

        # Apply wiring
        next_kwargs = schema_a.map_outputs(result_a, schema_b)
    """

    def __init__(
        self,
        agent_name: str,
        inputs: Optional[List[ToolParam]] = None,
        outputs: Optional[List[ToolParam]] = None,
        description: str = "",
    ):
        self.agent_name = agent_name
        self.inputs: List[ToolParam] = inputs or []
        self.outputs: List[ToolParam] = outputs or []
        self.description = description

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_dspy_signature(cls, agent_name: str, signature_cls) -> AgentIOSchema:
        """Build from a DSPy Signature class.

        Extracts InputField/OutputField with their ``desc`` annotations
        to produce typed field definitions.
        """
        inputs: List[ToolParam] = []
        outputs: List[ToolParam] = []
        description = ""

        try:
            import dspy

            # Extract description from signature docstring
            description = (signature_cls.__doc__ or "").strip().split("\n")[0]

            # Pydantic-style (DSPy 2.x+): model_fields
            if hasattr(signature_cls, 'model_fields'):
                for name, field_info in signature_cls.model_fields.items():
                    extra = getattr(field_info, 'json_schema_extra', None) or {}
                    field_type = extra.get('__dspy_field_type', '')
                    desc = extra.get('desc', '') or field_info.description or ''
                    type_hint = 'str'  # DSPy fields are strings by default

                    # Check annotation for type hints
                    annotation = field_info.annotation
                    if annotation is not None:
                        type_hint = getattr(annotation, '__name__', str(annotation))

                    param = ToolParam(
                        name=name,
                        type_hint=type_hint,
                        required=field_info.is_required(),
                        description=desc,
                    )
                    if field_type == 'input':
                        inputs.append(param)
                    elif field_type == 'output':
                        outputs.append(param)

            # Fallback: class attribute inspection
            if not inputs and not outputs:
                for name in dir(signature_cls):
                    if name.startswith('_'):
                        continue
                    attr = getattr(signature_cls, name, None)
                    desc = getattr(attr, 'desc', '') or ''
                    param = ToolParam(name=name, description=desc)
                    if isinstance(attr, dspy.InputField):
                        inputs.append(param)
                    elif isinstance(attr, dspy.OutputField):
                        outputs.append(param)

            # Last resort: use signature's field dicts
            if not inputs and hasattr(signature_cls, 'input_fields'):
                for name, info in signature_cls.input_fields.items():
                    desc = getattr(info, 'desc', '') or ''
                    inputs.append(ToolParam(name=name, description=desc))
            if not outputs and hasattr(signature_cls, 'output_fields'):
                for name, info in signature_cls.output_fields.items():
                    desc = getattr(info, 'desc', '') or ''
                    outputs.append(ToolParam(name=name, description=desc))

        except Exception as e:
            logger.warning(f"Failed to extract IO schema from signature: {e}")

        return cls(
            agent_name=agent_name,
            inputs=inputs,
            outputs=outputs,
            description=description,
        )

    # -- wiring ---------------------------------------------------------------

    # Generic "content receiver" input names — accept any string output
    _CONTENT_RECEIVERS = {'input', 'text', 'content', 'body', 'message', 'data'}

    # Generic "content producer" output names — provide string content
    _CONTENT_PRODUCERS = {
        'output', 'result', 'response', 'answer', 'analysis',
        'summary', 'findings', 'report', 'text', 'content',
    }

    # Semantic equivalences for agent I/O wiring
    _SEMANTIC_GROUPS = [
        {'text', 'content', 'body', 'input', 'message', 'data'},
        {'result', 'output', 'response', 'answer'},
        {'analysis', 'summary', 'findings', 'report'},
        {'query', 'question', 'task', 'instruction', 'prompt'},
        {'context', 'background', 'history'},
        {'sources', 'references', 'citations', 'urls'},
    ]

    def wire_to(self, target: AgentIOSchema) -> Dict[str, str]:
        """Determine how our outputs map to *target*'s inputs.

        Returns ``{target_input_name: our_output_name}`` for compatible pairs.

        Matching strategy (priority order):
        1. Exact name match (our output ``name`` == their input ``name``)
        2. Semantic group match (``analysis`` in same group as ``summary``)
        3. Content fallback — generic receivers (``input``, ``text``) accept
           the first content-producing output (``analysis``, ``result``, etc.)
        """
        mapping: Dict[str, str] = {}
        our_output_names = {p.name for p in self.outputs}
        used_outputs: Set[str] = set()

        for target_input in target.inputs:
            # Strategy 1: exact name match
            if target_input.name in our_output_names:
                mapping[target_input.name] = target_input.name
                used_outputs.add(target_input.name)
                continue

            # Strategy 2: semantic group match
            matched = False
            for group in self._SEMANTIC_GROUPS:
                if target_input.name in group:
                    for out_name in our_output_names:
                        if out_name in group and out_name not in used_outputs:
                            mapping[target_input.name] = out_name
                            used_outputs.add(out_name)
                            matched = True
                            break
                if matched:
                    break
            if matched:
                continue

            # Strategy 3: content fallback — generic receivers accept any content output
            if target_input.name in self._CONTENT_RECEIVERS:
                for out in self.outputs:
                    if out.name in self._CONTENT_PRODUCERS and out.name not in used_outputs:
                        mapping[target_input.name] = out.name
                        used_outputs.add(out.name)
                        break
                    # If no named match, use the first string output
                if target_input.name not in mapping:
                    for out in self.outputs:
                        if out.type_hint == 'str' and out.name not in used_outputs:
                            mapping[target_input.name] = out.name
                            used_outputs.add(out.name)
                            break

        return mapping

    def map_outputs(
        self,
        output_dict: Dict[str, Any],
        target: AgentIOSchema,
    ) -> Dict[str, Any]:
        """Map our output dict to kwargs suitable for *target* agent.

        Uses :meth:`wire_to` for field mapping, then extracts values from
        *output_dict*. Missing fields are skipped.
        """
        wiring = self.wire_to(target)
        kwargs: Dict[str, Any] = {}
        for target_input, our_output in wiring.items():
            if our_output in output_dict:
                kwargs[target_input] = output_dict[our_output]
        return kwargs

    # -- introspection --------------------------------------------------------

    @property
    def input_names(self) -> List[str]:
        return [p.name for p in self.inputs]

    @property
    def output_names(self) -> List[str]:
        return [p.name for p in self.outputs]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'description': self.description,
            'inputs': [{'name': p.name, 'type': p.type_hint, 'description': p.description} for p in self.inputs],
            'outputs': [{'name': p.name, 'type': p.type_hint, 'description': p.description} for p in self.outputs],
        }

    def __repr__(self) -> str:
        ins = ", ".join(self.input_names)
        outs = ", ".join(self.output_names)
        return f"AgentIOSchema({self.agent_name}, in=[{ins}], out=[{outs}])"


# =============================================================================
# FILE REFERENCE (Lazy handle for large outputs)
# =============================================================================


@dataclass
class FileReference:
    """Lazy handle for large outputs spilled to disk.

    Instead of keeping >500KB strings in memory, callers write the data
    to *path* and store a lightweight FileReference in the outputs dict.
    Consumers call :meth:`load` when they actually need the content.
    """
    path: str
    content_type: str = "text/plain"
    size_bytes: int = 0
    checksum: str = ""
    step_key: str = ""
    description: str = ""

    def exists(self) -> bool:
        """Check if the backing file still exists on disk."""
        return os.path.isfile(self.path)

    def load(self) -> str:
        """Read and return the file contents."""
        return Path(self.path).read_text(encoding="utf-8")


# =============================================================================
# SWARM ARTIFACT STORE (Tag-queryable artifact registry)
# =============================================================================


class SwarmArtifactStore:
    """Tag-queryable artifact registry.

    Drop-in replacement for the flat ``outputs: Dict[str, Any]`` dict used
    during plan execution.  Each artifact can carry semantic *tags* and a
    human-readable *description* so that downstream consumers can query by
    meaning rather than relying on ``step_0``, ``step_1`` keys.

    Backward-compatible: supports ``keys()``, ``values()``, ``items()``,
    ``__getitem__``, ``__len__``, ``__bool__``, and ``to_outputs_dict()``.
    """

    @dataclass
    class _Entry:
        data: Any
        tags: Set[str] = field(default_factory=set)
        description: str = ""

    def __init__(self) -> None:
        self._store: Dict[str, SwarmArtifactStore._Entry] = {}

    # -- mutators -------------------------------------------------------------

    def register(
        self,
        key: str,
        data: Any,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """Store an artifact with optional semantic tags."""
        self._store[key] = self._Entry(
            data=data,
            tags=set(tags) if tags else set(),
            description=description,
        )

    # -- queries --------------------------------------------------------------

    def query_by_tag(self, tag: str) -> Dict[str, Any]:
        """Return ``{key: data}`` for every artifact carrying *tag*."""
        return {
            k: entry.data
            for k, entry in self._store.items()
            if tag in entry.tags
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get — returns raw data for *key*."""
        entry = self._store.get(key)
        return entry.data if entry is not None else default

    # -- dict-like interface --------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._store[key].data

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow ``store[key] = value`` for backward compat (no tags)."""
        self.register(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        return bool(self._store)

    def keys(self) -> Iterator[str]:
        return iter(self._store.keys())

    def values(self) -> Iterator[Any]:
        return (e.data for e in self._store.values())

    def items(self) -> Iterator[Tuple[str, Any]]:
        return ((k, e.data) for k, e in self._store.items())

    def to_outputs_dict(self) -> Dict[str, Any]:
        """Convert back to a plain dict for legacy callers."""
        return {k: e.data for k, e in self._store.items()}


# =============================================================================
# EXECUTION STEP SCHEMA (Pydantic, optional)
# =============================================================================

try:
    from pydantic import BaseModel, Field, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:
    class ExecutionStepSchema(BaseModel):
        """Schema for execution plan steps - accepts common LLM field name variations."""
        skill_name: str = Field(default="", description="Skill name from available_skills")
        tool_name: str = Field(default="", description="Tool name from that skill's tools list")
        params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
        description: str = Field(default="", description="What this step does")
        depends_on: List[int] = Field(default_factory=list, description="Indices of steps this depends on")
        output_key: str = Field(default="", description="Key to store output under")
        optional: bool = Field(default=False, description="Whether step is optional")
        verification: str = Field(default="", description="How to confirm this step succeeded")
        fallback_skill: str = Field(default="", description="Alternative skill if this one fails")

        model_config = {"extra": "allow"}

        @model_validator(mode='before')
        @classmethod
        def normalize_field_names(cls, data: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize common LLM field name variations to expected names."""
            if not isinstance(data, dict):
                return data

            if 'skill_name' not in data or not data.get('skill_name'):
                skill = data.get('skill', '')
                if not skill:
                    skills_used = data.get('skills_used', [])
                    if skills_used and isinstance(skills_used, list):
                        skill = skills_used[0]
                data['skill_name'] = skill

            if 'tool_name' not in data or not data.get('tool_name'):
                tool = data.get('tool', '')
                if not tool:
                    tools_used = data.get('tools_used', [])
                    if tools_used and isinstance(tools_used, list):
                        tool = tools_used[0]
                if not tool:
                    action = data.get('action', '')
                    if action and isinstance(action, str):
                        tool_match = re.search(r'\b([a-z_]+_tool)\b', action)
                        if tool_match:
                            tool = tool_match.group(1)
                data['tool_name'] = tool

            if 'params' not in data or not data.get('params'):
                data['params'] = (
                    data.get('parameters') or
                    data.get('tool_input') or
                    data.get('tool_params') or
                    data.get('inputs') or
                    data.get('input') or
                    {}
                )

            return data
else:
    ExecutionStepSchema = None  # type: ignore[assignment,misc]


# =============================================================================
# EXECUTION RESULT (Dataclass)
# =============================================================================

try:
    from Jotty.core.utils.context_utils import strip_enrichment_context
except ImportError:
    def strip_enrichment_context(text: str) -> str:
        return text


def _clean_for_display(text: str) -> str:
    """Remove internal enrichment context from text for user-facing display."""
    return strip_enrichment_context(text) if text else text


@dataclass
class AgenticExecutionResult:
    """Result of task execution."""
    success: bool
    task: str
    task_type: TaskType
    skills_used: List[str]
    steps_executed: int
    outputs: Dict[str, Any]
    final_output: Any
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    stopped_early: bool = False

    @property
    def artifacts(self) -> List[Dict[str, Any]]:
        """Extract all created files/artifacts from execution outputs.

        Scans outputs for file paths (from file-operations, shell-exec, etc.)
        Returns list of {path, type, size_bytes, step} dicts.
        """
        found = []
        for step_name, step_data in (self.outputs or {}).items():
            if not isinstance(step_data, dict):
                continue
            # file-operations returns {path, bytes_written}
            if 'path' in step_data and step_data.get('success', True):
                found.append({
                    'path': step_data['path'],
                    'type': 'file',
                    'size_bytes': step_data.get('bytes_written', 0),
                    'step': step_name,
                })
            # shell-exec may create files (check stdout for file paths)
            if 'stdout' in step_data:
                stdout = str(step_data.get('stdout', ''))
                for m in re.finditer(
                    r'(?:saved?|wrot?e?|created?|output)\s+(?:to\s+)?["\']?([^\s"\']+\.\w{1,5})',
                    stdout, re.I,
                ):
                    fpath = m.group(1)
                    if len(fpath) > 3 and ('/' in fpath or '.' in fpath):
                        found.append({
                            'path': fpath, 'type': 'file',
                            'size_bytes': 0, 'step': step_name,
                        })
        return found

    @property
    def summary(self) -> str:
        """Human-readable summary of what was done and what was produced."""
        parts = []
        status = "completed successfully" if self.success else "failed"
        parts.append(f"Task {status} in {self.execution_time:.1f}s ({self.steps_executed} steps)")

        if self.skills_used:
            parts.append(f"Skills used: {', '.join(self.skills_used)}")

        artifacts = self.artifacts
        if artifacts:
            parts.append("Files created:")
            for a in artifacts:
                size = f" ({a['size_bytes']} bytes)" if a.get('size_bytes') else ""
                parts.append(f"  → {a['path']}{size}")

        if self.errors:
            parts.append(f"Errors: {'; '.join(self.errors[:3])}")

        if self.final_output and isinstance(self.final_output, str):
            clean = self.final_output.strip()[:300]
            if clean:
                parts.append(f"Output: {clean}")

        return '\n'.join(parts)


__all__ = [
    'TaskType',
    'ExecutionStep',
    'ToolParam',
    'TypeCoercer',
    'ToolValidationResult',
    'ToolSchema',
    'AgentIOSchema',
    'FileReference',
    'SwarmArtifactStore',
    'ExecutionStepSchema',
    'AgenticExecutionResult',
    '_clean_for_display',
]
