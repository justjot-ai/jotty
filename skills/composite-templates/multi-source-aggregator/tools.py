"""
Multi-Source Aggregator Template

Common use case: Collect data from multiple sources → Combine → Output.

Workflow:
1. Parallel data collection from multiple sources
2. Combine/merge results
3. Process/transform data (optional)
4. Output to file/format

Customizable for:
- Data sources (web, APIs, files)
- Combination strategy (merge, append, aggregate)
- Output format (JSON, CSV, PDF, etc.)
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def aggregate_sources_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate data from multiple sources.

    Template for: Parallel Sources → Combine → Output workflow

    Args:
        params: Dictionary containing:
            - sources (list, required): List of source configurations
                Each source: {'type': 'web-search'|'file'|'api', 'params': {...}}
            - combine_strategy (str, optional): 'merge', 'append', 'aggregate' (default: 'merge')
            - output_format (str, optional): 'json', 'csv', 'pdf', 'markdown' (default: 'json')
            - output_file (str, optional): Output file path

    Returns:
        Dictionary with aggregated data and output file path
    """
    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from core.registry.skills_registry import get_skills_registry

        sources = params.get("sources", [])
        if not sources:
            return {"success": False, "error": "sources parameter is required"}

        combine_strategy = params.get("combine_strategy", "merge")
        output_format = params.get("output_format", "json")
        output_file = params.get("output_file")

        registry = get_skills_registry()
        registry.init()

        composer_skill = registry.get_skill("skill-composer")
        if not composer_skill:
            return {"success": False, "error": "skill-composer not available"}

        compose_tool = composer_skill.tools.get("compose_skills_tool")

        # Build parallel collection step
        parallel_skills = []
        for i, source in enumerate(sources):
            source_type = source.get("type", "web-search")
            source_params = source.get("params", {})

            if source_type == "web-search":
                parallel_skills.append(
                    {"skill": "web-search", "tool": "search_web_tool", "params": source_params}
                )
            elif source_type == "file":
                parallel_skills.append(
                    {"skill": "file-operations", "tool": "read_file_tool", "params": source_params}
                )
            # Add more source types as needed

        workflow_steps = [{"type": "parallel", "name": "collect_data", "skills": parallel_skills}]

        # Add combination/processing step
        if combine_strategy == "merge":
            # Use text-utils to combine
            workflow_steps.append(
                {
                    "type": "single",
                    "name": "combine",
                    "skill": "text-utils",
                    "tool": "combine_texts_tool",
                    "params": {"texts": ["${collect_data.output.results}"]},
                }
            )

        # Add output step based on format
        if output_format == "json":
            import tempfile

            output_file = output_file or tempfile.mktemp(suffix=".json")
            workflow_steps.append(
                {
                    "type": "single",
                    "name": "write_output",
                    "skill": "file-operations",
                    "tool": "write_file_tool",
                    "params": {
                        "path": output_file,
                        "content": "${combine.output.text}",  # Would need JSON serialization
                    },
                }
            )
        elif output_format == "pdf":
            import tempfile

            temp_md = tempfile.mktemp(suffix=".md")
            workflow_steps.extend(
                [
                    {
                        "type": "single",
                        "name": "write_md",
                        "skill": "file-operations",
                        "tool": "write_file_tool",
                        "params": {"path": temp_md, "content": "${combine.output.text}"},
                    },
                    {
                        "type": "single",
                        "name": "convert_pdf",
                        "skill": "document-converter",
                        "tool": "convert_to_pdf_tool",
                        "params": {
                            "input_file": "${write_md.path}",
                            "output_file": output_file or tempfile.mktemp(suffix=".pdf"),
                        },
                    },
                ]
            )

        workflow = {"workflow": workflow_steps}

        # Execute workflow
        result = await compose_tool(workflow)

        if not result.get("success"):
            return {"success": False, "error": f'Aggregation failed: {result.get("error")}'}

        import os

        return {
            "success": True,
            "output_file": output_file,
            "file_size": (
                os.path.getsize(output_file) if output_file and os.path.exists(output_file) else 0
            ),
            "sources_count": len(sources),
            "format": output_format,
        }

    except Exception as e:
        logger.error(f"Aggregation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
