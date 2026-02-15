"""Calculate cyclomatic complexity, LOC, function count from Python source."""

import ast
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("code-complexity-analyzer")


def _cyclomatic(node: ast.AST) -> int:
    """Count decision points in an AST node."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, (ast.Assert, ast.With)):
            complexity += 1
        elif isinstance(child, ast.comprehension):
            complexity += 1 + len(child.ifs)
    return complexity


@tool_wrapper(required_params=["source"])
def analyze_complexity(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Python source code complexity.

    Params:
        source: Python source code string
        threshold: complexity warning threshold (default 10)
    """
    status.set_callback(params.pop("_status_callback", None))
    source = params["source"]
    threshold = int(params.get("threshold", 10))

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return tool_error(f"Syntax error: {e}")

    lines = source.splitlines()
    total_loc = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    code_lines = total_loc - blank_lines - comment_lines

    functions: List[Dict[str, Any]] = []
    classes: List[str] = []
    imports = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _cyclomatic(node)
            end = getattr(node, "end_lineno", node.lineno)
            func_lines = end - node.lineno + 1
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": cc,
                    "lines": func_lines,
                    "high_complexity": cc > threshold,
                }
            )
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1

    overall_cc = _cyclomatic(tree)
    high = [f for f in functions if f["high_complexity"]]

    return tool_response(
        total_loc=total_loc,
        code_lines=code_lines,
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        function_count=len(functions),
        class_count=len(classes),
        import_count=imports,
        overall_complexity=overall_cc,
        functions=functions,
        high_complexity_functions=high,
        threshold=threshold,
    )


__all__ = ["analyze_complexity"]
