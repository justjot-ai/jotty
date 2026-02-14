#!/usr/bin/env python3
"""
Analyze functions missing type hints and suggest proper types.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def analyze_return_statements(func_node: ast.FunctionDef) -> List[str]:
    """Analyze all return statements to understand return type."""
    returns = []

    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            if node.value is None:
                returns.append('None')
            elif isinstance(node.value, ast.Constant):
                val = node.value.value
                if val is None:
                    returns.append('None')
                elif isinstance(val, bool):
                    returns.append('bool')
                elif isinstance(val, int):
                    returns.append('int')
                elif isinstance(val, str):
                    returns.append('str')
                elif isinstance(val, float):
                    returns.append('float')
            elif isinstance(node.value, ast.Dict):
                returns.append('Dict')
            elif isinstance(node.value, ast.List):
                returns.append('List')
            elif isinstance(node.value, ast.Tuple):
                returns.append('Tuple')
            elif isinstance(node.value, ast.Call):
                # Function call - try to infer from function name
                if isinstance(node.value.func, ast.Name):
                    fname = node.value.func.id
                    if 'dict' in fname.lower():
                        returns.append('Dict')
                    elif 'list' in fname.lower():
                        returns.append('List')
                    else:
                        returns.append(f'{fname}_return')

    return returns


def suggest_param_type(param_name: str, default_value, func_name: str, file_content: str) -> str:
    """Suggest parameter type based on context."""
    # Check default value first
    if default_value is not None:
        if isinstance(default_value, ast.Constant):
            val = default_value.value
            if val is None:
                # Look for usage in function to infer
                if 'config' in param_name.lower():
                    return 'Optional[SwarmConfig]'
                return 'Optional[Any]'  # fallback
            elif isinstance(val, bool):
                return 'bool'
            elif isinstance(val, int):
                return 'int'
            elif isinstance(val, str):
                return 'str'
            elif isinstance(val, float):
                return 'float'
        elif isinstance(default_value, ast.List):
            return 'List'
        elif isinstance(default_value, ast.Dict):
            return 'Dict[str, Any]'

    # Analyze parameter name patterns
    name_lower = param_name.lower()

    # Config/options patterns
    if 'config' in name_lower:
        # Check file imports for config types
        if 'SwarmConfig' in file_content or 'SwarmBaseConfig' in file_content:
            return 'Optional[SwarmConfig]'
        return 'Dict[str, Any]'

    if param_name in ('kwargs', 'kw'):
        return 'Any'  # **kwargs
    if param_name == 'args':
        return 'Any'  # *args

    # Common patterns
    if name_lower.endswith(('_id', '_name', '_path', '_key')):
        return 'str'
    if name_lower.startswith(('is_', 'has_', 'should_', 'enable_', 'disable_')):
        return 'bool'
    if name_lower in ('count', 'limit', 'max', 'min', 'size', 'index'):
        return 'int'
    if name_lower.endswith(('_dict', '_params', '_options', '_metadata')):
        return 'Dict[str, Any]'
    if name_lower.endswith(('_list', '_items', '_values')):
        return 'List'

    # Context-based inference
    if 'agent' in name_lower:
        return 'BaseAgent'
    if 'swarm' in name_lower:
        return 'BaseSwarm'
    if 'model' in name_lower:
        return 'str'  # model name

    return None  # Can't infer


def analyze_file(filepath: Path) -> List[Dict]:
    """Analyze a file and return functions needing hints with suggestions."""
    try:
        with open(filepath) as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))
    except:
        return []

    suggestions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Collect missing hints
            missing_return = node.returns is None
            missing_params = []

            for arg in node.args.args:
                if arg.arg not in ('self', 'cls') and arg.annotation is None:
                    default_val = None
                    # Find default value
                    arg_index = node.args.args.index(arg)
                    defaults_offset = len(node.args.args) - len(node.args.defaults)
                    if arg_index >= defaults_offset:
                        default_val = node.args.defaults[arg_index - defaults_offset]

                    suggested_type = suggest_param_type(arg.arg, default_val, node.name, content)
                    if suggested_type:
                        missing_params.append((arg.arg, suggested_type))

            if missing_return or missing_params:
                # Suggest return type
                return_suggestion = None
                if missing_return:
                    returns = analyze_return_statements(node)
                    if returns:
                        unique_returns = set(returns)
                        if len(unique_returns) == 1:
                            return_suggestion = unique_returns.pop()
                        elif 'None' in unique_returns and len(unique_returns) == 2:
                            other = [r for r in unique_returns if r != 'None'][0]
                            return_suggestion = f'Optional[{other}]'
                        else:
                            return_suggestion = 'Union[' + ', '.join(sorted(unique_returns)) + ']'

                suggestions.append({
                    'file': str(filepath),
                    'line': node.lineno,
                    'function': node.name,
                    'is_private': node.name.startswith('_'),
                    'missing_return': missing_return,
                    'return_suggestion': return_suggestion,
                    'missing_params': missing_params
                })

    return suggestions


def main():
    import sys

    # Analyze core/ directory
    all_suggestions = []

    for py_file in Path('Jotty/core').rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue

        suggestions = analyze_file(py_file)
        all_suggestions.extend(suggestions)

    # Filter to public methods only
    public_suggestions = [s for s in all_suggestions if not s['is_private']]

    print(f"📊 Missing Type Hints Analysis\n")
    print(f"Total functions missing hints: {len(all_suggestions)}")
    print(f"Public functions: {len(public_suggestions)}")
    print(f"Private functions: {len(all_suggestions) - len(public_suggestions)}\n")

    # Group by file
    by_file = defaultdict(list)
    for sug in public_suggestions:
        short_file = sug['file'].replace('/var/www/sites/personal/stock_market/Jotty/', '')
        by_file[short_file].append(sug)

    # Show top files
    sorted_files = sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)

    print("Top 20 files with public methods needing hints:\n")
    for file, suggestions in sorted_files[:20]:
        print(f"\n📄 {file} ({len(suggestions)} methods)")
        for sug in suggestions[:3]:
            print(f"   Line {sug['line']}: {sug['function']}")
            if sug['missing_params']:
                for param, ptype in sug['missing_params']:
                    print(f"      Param: {param} -> {ptype}")
            if sug['return_suggestion']:
                print(f"      Return: -> {sug['return_suggestion']}")

    print(f"\n💡 Use this analysis to add proper type hints manually")
    print(f"   Focus on top files for maximum impact")


if __name__ == '__main__':
    main()
