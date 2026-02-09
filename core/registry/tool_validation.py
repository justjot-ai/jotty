"""
Tool Validation Framework

Validates tools before registration to catch errors early.
Based on OAgents tool validation approach.
"""
import inspect
import ast
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Authorized types for tool inputs/outputs
AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "dict",
    "list",
    "any",
    "null",
]

# Type conversion mapping
TYPE_CONVERSION = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "dict",
    "list": "list",
}


@dataclass
class ValidationResult:
    """Result of tool validation."""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class MethodChecker(ast.NodeVisitor):
    """
    AST visitor to check for dangerous method calls.
    
    Based on OAgents MethodChecker.
    """
    
    DANGEROUS_METHODS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input', 'raw_input',
        'execfile', 'reload', 'exit', 'quit', 'file'
    }
    
    DANGEROUS_ATTRIBUTES = {
        '__builtins__', '__globals__', '__locals__', '__code__', '__dict__'
    }
    
    def __init__(self, allowed_imports: Set[str]):
        """
        Initialize method checker.
        
        Args:
            allowed_imports: Set of allowed import names
        """
        self.allowed_imports = allowed_imports
        self.errors: List[str] = []
    
    def visit_Call(self, node):
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_METHODS:
                self.errors.append(
                    f"Dangerous method call: {node.func.id}()"
                )
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.DANGEROUS_METHODS:
                self.errors.append(
                    f"Dangerous method call: {node.func.attr}()"
                )
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Check imports."""
        for alias in node.names:
            if alias.name not in self.allowed_imports:
                self.errors.append(
                    f"Unauthorized import: {alias.name}"
                )
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check from imports."""
        if node.module not in self.allowed_imports:
            self.errors.append(
                f"Unauthorized import from: {node.module}"
            )
        self.generic_visit(node)


class ToolValidator:
    """
    Validates tools before registration.
    
    Checks:
    - Signature matches metadata
    - Types are authorized
    - Code safety (no dangerous calls)
    - Required attributes present
    
    Usage:
        validator = ToolValidator()
        result = validator.validate_tool(tool_func, tool_metadata)
        if not result.valid:
            print(f"Validation errors: {result.errors}")
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize tool validator.
        
        Args:
            strict: Whether to use strict validation (default: True)
        """
        self.strict = strict
    
    def validate_tool(
        self,
        tool_func: Callable,
        tool_metadata: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate a tool function.
        
        Args:
            tool_func: Tool function to validate
            tool_metadata: Tool metadata with:
                - name: Tool name
                - description: Tool description
                - inputs: Input schema (dict)
                - output_type: Output type
                
        Returns:
            ValidationResult
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check required metadata
        required_fields = ['name', 'description', 'inputs', 'output_type']
        for field in required_fields:
            if field not in tool_metadata:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validate signature
        sig_result = self._validate_signature(tool_func, tool_metadata)
        errors.extend(sig_result.errors)
        warnings.extend(sig_result.warnings)
        
        # Validate types
        type_result = self._validate_types(tool_metadata)
        errors.extend(type_result.errors)
        warnings.extend(type_result.warnings)
        
        # Validate code safety (if strict)
        if self.strict:
            safety_result = self._validate_code_safety(tool_func)
            errors.extend(safety_result.errors)
            warnings.extend(safety_result.warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_signature(
        self,
        tool_func: Callable,
        tool_metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate function signature matches metadata."""
        errors: List[str] = []
        warnings: List[str] = []
        
        try:
            signature = inspect.signature(tool_func)
            param_names = set(signature.parameters.keys())
            
            # Remove 'self' if present
            if 'self' in param_names:
                param_names.remove('self')
            
            # Get expected inputs
            expected_inputs = set(tool_metadata.get('inputs', {}).keys())
            
            # Check match
            if param_names != expected_inputs:
                missing = expected_inputs - param_names
                extra = param_names - expected_inputs
                
                if missing:
                    errors.append(
                        f"Signature missing parameters: {missing}"
                    )
                if extra:
                    warnings.append(
                        f"Signature has extra parameters: {extra}"
                    )
        
        except Exception as e:
            errors.append(f"Failed to inspect signature: {e}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_types(self, tool_metadata: Dict[str, Any]) -> ValidationResult:
        """Validate input/output types."""
        errors: List[str] = []
        warnings: List[str] = []
        
        # Validate input types
        inputs = tool_metadata.get('inputs', {})
        for input_name, input_spec in inputs.items():
            if isinstance(input_spec, dict):
                input_type = input_spec.get('type', 'any')
                if input_type not in AUTHORIZED_TYPES:
                    errors.append(
                        f"Input '{input_name}': type '{input_type}' not authorized. "
                        f"Allowed: {AUTHORIZED_TYPES}"
                    )
        
        # Validate output type
        output_type = tool_metadata.get('output_type', 'any')
        if output_type not in AUTHORIZED_TYPES:
            errors.append(
                f"Output type '{output_type}' not authorized. "
                f"Allowed: {AUTHORIZED_TYPES}"
            )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_code_safety(self, tool_func: Callable) -> ValidationResult:
        """Validate code safety (no dangerous calls)."""
        errors: List[str] = []
        warnings: List[str] = []
        
        try:
            # Get source code
            source = inspect.getsource(tool_func)
            
            # Parse AST
            tree = ast.parse(source)
            
            # Check for dangerous calls
            # Get allowed imports from function's imports
            allowed_imports = self._get_allowed_imports(tree)
            
            checker = MethodChecker(allowed_imports)
            checker.visit(tree)
            
            if checker.errors:
                errors.extend(checker.errors)
        
        except Exception as e:
            warnings.append(f"Could not validate code safety: {e}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _get_allowed_imports(self, tree: ast.AST) -> Set[str]:
        """Extract allowed imports from AST."""
        allowed = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    allowed.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    allowed.add(node.module)
        
        # Add standard library
        import sys
        allowed.update(sys.stdlib_module_names)
        
        return allowed


def validate_tool_attributes(tool_class: type) -> ValidationResult:
    """
    Validate tool class attributes (OAgents style).
    
    Checks:
    - Required attributes present (name, description, inputs, output_type)
    - Types are correct
    - Values are valid
    
    Args:
        tool_class: Tool class to validate
        
    Returns:
        ValidationResult
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # Check required attributes
    required_attrs = {
        'name': str,
        'description': str,
        'inputs': dict,
        'output_type': str,
    }
    
    for attr_name, expected_type in required_attrs.items():
        if not hasattr(tool_class, attr_name):
            errors.append(f"Missing required attribute: {attr_name}")
            continue
        
        attr_value = getattr(tool_class, attr_name)
        if not isinstance(attr_value, expected_type):
            errors.append(
                f"Attribute {attr_name} should be {expected_type.__name__}, "
                f"got {type(attr_value).__name__}"
            )
    
    # Validate inputs structure
    if hasattr(tool_class, 'inputs'):
        inputs = tool_class.inputs
        if isinstance(inputs, dict):
            for input_name, input_spec in inputs.items():
                if not isinstance(input_spec, dict):
                    errors.append(f"Input '{input_name}' should be a dictionary")
                    continue
                
                if 'type' not in input_spec or 'description' not in input_spec:
                    errors.append(
                        f"Input '{input_name}' should have 'type' and 'description' keys"
                    )
                    continue
                
                input_type = input_spec['type']
                if input_type not in AUTHORIZED_TYPES:
                    errors.append(
                        f"Input '{input_name}': type '{input_type}' not authorized. "
                        f"Allowed: {AUTHORIZED_TYPES}"
                    )
    
    # Validate output type
    if hasattr(tool_class, 'output_type'):
        output_type = tool_class.output_type
        if output_type not in AUTHORIZED_TYPES:
            errors.append(
                f"Output type '{output_type}' not authorized. "
                f"Allowed: {AUTHORIZED_TYPES}"
            )
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


# =============================================================================
# RUNTIME TOOL GUARD (Cline ToolExecutor patterns)
# =============================================================================

class ToolGuard:
    """
    Runtime tool execution guard — enforces safety policies BEFORE tool runs.

    Inspired by Cline's ToolExecutor:
    1. Plan/Act mode: planning phase only allows SAFE tools
    2. One-side-effect-per-turn: at most one SIDE_EFFECT/DESTRUCTIVE tool per turn
    3. Path-based access control: block tools on sensitive paths
    4. Cooldown: prevent rapid-fire destructive calls

    Usage:
        guard = ToolGuard()
        ok, reason = guard.check("send-telegram", "side_effect", mode="plan")
        if not ok:
            logger.warning(f"Blocked: {reason}")

        # After successful tool execution:
        guard.record_use("send-telegram", "side_effect")

        # Next side-effect in same turn is blocked:
        ok, reason = guard.check("write-file", "side_effect")
        # ok=False, reason="One side-effect tool per turn..."

        # Reset at turn boundary:
        guard.reset_turn()
    """

    # Paths that tools should never touch
    DEFAULT_BLOCKED_PATHS = {
        '.env', '.git', 'credentials', 'secrets',
        'id_rsa', 'id_ed25519', '.ssh', '.aws',
        'token', 'password', '.key', '.pem',
    }

    def __init__(
        self,
        blocked_path_patterns: Optional[Set[str]] = None,
        max_destructive_per_session: int = 3,
    ):
        self._blocked_patterns = blocked_path_patterns or self.DEFAULT_BLOCKED_PATHS
        self._max_destructive = max_destructive_per_session
        self._side_effect_used_this_turn = False
        self._destructive_count_session = 0
        self._last_tool: Optional[str] = None

    def check(
        self,
        tool_name: str,
        trust_level: str = "safe",
        mode: str = "act",
        target_path: Optional[str] = None,
    ) -> tuple:
        """
        Check if a tool call should be allowed.

        Args:
            tool_name: Name of the tool
            trust_level: "safe", "side_effect", or "destructive"
            mode: "plan" or "act"
            target_path: Optional file path the tool will operate on

        Returns:
            (allowed: bool, reason: str)
        """
        # 1. Plan mode: only SAFE tools
        if mode == "plan" and trust_level != "safe":
            return (False,
                f"Tool '{tool_name}' ({trust_level}) blocked in plan mode. "
                f"Only safe/read-only tools allowed during planning."
            )

        # 2. One side-effect per turn
        if trust_level in ("side_effect", "destructive") and self._side_effect_used_this_turn:
            return (False,
                f"Tool '{tool_name}' blocked: one side-effect tool per turn. "
                f"Previous: '{self._last_tool}'. Reset turn boundary first."
            )

        # 3. Session-level destructive limit
        if trust_level == "destructive":
            if self._destructive_count_session >= self._max_destructive:
                return (False,
                    f"Tool '{tool_name}' blocked: destructive tool limit "
                    f"({self._max_destructive}) reached for this session."
                )

        # 4. Path-based access control
        if target_path:
            path_lower = target_path.lower()
            for pattern in self._blocked_patterns:
                if pattern in path_lower:
                    return (False,
                        f"Tool '{tool_name}' blocked: path '{target_path}' "
                        f"matches blocked pattern '{pattern}'."
                    )

        return (True, "")

    def record_use(self, tool_name: str, trust_level: str = "safe"):
        """Record a successful tool use (call after execution)."""
        if trust_level in ("side_effect", "destructive"):
            self._side_effect_used_this_turn = True
            self._last_tool = tool_name
        if trust_level == "destructive":
            self._destructive_count_session += 1

    def reset_turn(self):
        """Reset per-turn state. Call at the start of each agent turn."""
        self._side_effect_used_this_turn = False
        self._last_tool = None

    def reset_session(self):
        """Full reset (new task/session)."""
        self.reset_turn()
        self._destructive_count_session = 0

    def stats(self) -> Dict[str, Any]:
        """Current guard state for debugging."""
        return {
            'side_effect_used_this_turn': self._side_effect_used_this_turn,
            'last_tool': self._last_tool,
            'destructive_count_session': self._destructive_count_session,
            'max_destructive': self._max_destructive,
        }
