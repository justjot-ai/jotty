"""
CLI Result Display - Output formatting and presentation
=======================================================

Extracted from app.py. Handles displaying execution results:
- File path extraction and display
- Summary panel rendering
- Markdown content rendering
- Export option prompts
"""

import re
import logging
from typing import Any, List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


def extract_file_paths(result) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Extract file paths and summary info from execution result.

    Returns:
        (file_paths, summary) where file_paths is [(label, path), ...]
    """
    file_paths = []
    summary = {}

    output = result.output if hasattr(result, 'output') else result

    # LeanResult with direct output_path
    if hasattr(result, 'output_path') and result.output_path:
        fmt = getattr(result, 'output_format', 'file')
        file_paths.append((fmt.upper(), result.output_path))

    # ExecutionResult from AutoAgent
    elif hasattr(output, 'outputs') and hasattr(output, 'final_output'):
        outputs_dict = output.outputs or {}
        seen_paths = set()
        for step_key, step_result in outputs_dict.items():
            if isinstance(step_result, dict):
                for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
                    if key in step_result and step_result[key]:
                        path = step_result[key]
                        if path not in seen_paths:
                            file_paths.append((key.replace('_', ' ').title(), path))
                            seen_paths.add(path)
                for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                    if key in step_result and step_result[key]:
                        summary[key] = step_result[key]

    elif isinstance(output, dict):
        for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
            if key in output and output[key]:
                file_paths.append((key.replace('_', ' ').title(), output[key]))
        for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
            if key in output and output[key]:
                summary[key] = output[key]

    elif isinstance(output, str) and not file_paths:
        path_matches = re.findall(
            r'(/[\w/\-_.]+\.(pdf|md|txt|html|json|csv|png|jpg|docx))', output
        )
        for match in path_matches:
            file_paths.append(('Output', match[0]))

    return file_paths, summary


def display_result(renderer, result, elapsed: float, output_history: list):
    """
    Display execution result with rich formatting.

    Args:
        renderer: RichRenderer instance
        result: Execution result object
        elapsed: Execution time in seconds
        output_history: Mutable list to append output content to
    """
    renderer.newline()

    if result.success:
        # Show completion status
        steps = getattr(result, 'steps_taken', None)
        if steps:
            arrow = ' \u2192 '
            renderer.success(f"Completed in {elapsed:.1f}s ({len(steps)} steps: {arrow.join(steps)})")
        else:
            renderer.success(f"Completed in {elapsed:.1f}s")

        file_paths, summary = extract_file_paths(result)

        # File paths
        if file_paths:
            renderer.newline()
            renderer.print("[bold green]\U0001f4c1 Generated Files:[/bold green]")
            for label, path in file_paths:
                renderer.print(f"   {label}: [cyan]{path}[/cyan]")

        # Summary
        if summary:
            renderer.newline()
            renderer.panel(
                "\n".join([f"\u2022 {k}: {v}" for k, v in summary.items()]),
                title="Summary",
                style="green"
            )
        elif not file_paths:
            full_content = str(result.output if hasattr(result, 'output') else result)
            was_streamed = getattr(result, 'was_streamed', False)

            if not was_streamed:
                renderer.newline()
                renderer.markdown(full_content)

            # Store in history
            output_history.append(full_content)
            if len(output_history) > 20:
                del output_history[:-20]

        return file_paths
    else:
        renderer.error(f"Failed after {elapsed:.1f}s")
        error_msg = getattr(result, 'error', None)
        if not error_msg and hasattr(result, 'alerts') and result.alerts:
            error_msg = "; ".join(result.alerts[:3])
        if error_msg:
            renderer.panel(error_msg, title="Error Details", style="red")
        return []
