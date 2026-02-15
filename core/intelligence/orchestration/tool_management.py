"""
Tool Management (Agent0 Dynamic Tool Management)
==================================================

Dynamic tool management based on Agent0 learned performance:
- ToolManager: Bridges tool success rates with runtime tool selection

Tracks per-swarm tool additions/removals and suggests replacements for
consistently failing tools.

Extracted from swarm_intelligence.py for modularity.
"""

import logging
from typing import Any, Dict, List, Tuple

try:
    from Jotty.core.infrastructure.metadata.tool_shed import ToolShedSchema as ToolSchema
except ImportError:
    ToolSchema = None

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Dynamic tool management based on Agent0 learned performance.

    Bridges CurriculumGenerator._tool_success_rates with runtime tool selection.
    Tracks per-swarm tool additions/removals and suggests replacements for
    consistently failing tools.
    """

    FAILURE_THRESHOLD = 0.6  # Below 60% success = failing tool
    MIN_SAMPLES = 3  # Need 3+ uses before judging

    def __init__(self) -> None:
        self._tool_assignments: Dict[str, List[str]] = {}  # swarm_name -> [added tools]
        self._deactivated_tools: Dict[str, List[str]] = {}  # swarm_name -> [removed tools]
        self._tool_registry: Dict[str, Any] = {}  # name -> ToolSchema

    def register_tool_schema(self, schema: Any) -> None:
        """Register a tool schema for capability-based replacement."""
        self._tool_registry[schema.name] = schema

    def analyze_tools(
        self, tool_success_rates: Dict[str, Tuple[int, int]], swarm_name: str
    ) -> Dict[str, Any]:
        """
        Classify tools as weak/strong based on Agent0 success rates.

        Args:
            tool_success_rates: Dict of tool_name -> (successes, total_uses)
            swarm_name: Name of the swarm being analyzed

        Returns:
            Dict with weak_tools, strong_tools, suggested_removals, replacements
        """
        weak_tools = []
        strong_tools = []
        suggested_removals = []
        replacements = {}

        for tool, rate_data in tool_success_rates.items():
            if isinstance(rate_data, (list, tuple)) and len(rate_data) == 2:
                successes, total = rate_data
            else:
                continue

            if total < self.MIN_SAMPLES:
                continue

            success_rate = successes / total if total > 0 else 0.0

            if success_rate < self.FAILURE_THRESHOLD:
                weak_tools.append(
                    {
                        "tool": tool,
                        "success_rate": success_rate,
                        "successes": successes,
                        "total": total,
                    }
                )
                suggested_removals.append(tool)
                tool_replacements = self.find_replacements(tool)
                if tool_replacements:
                    replacements[tool] = tool_replacements
            else:
                strong_tools.append(
                    {
                        "tool": tool,
                        "success_rate": success_rate,
                        "successes": successes,
                        "total": total,
                    }
                )

        return {
            "weak_tools": weak_tools,
            "strong_tools": strong_tools,
            "suggested_removals": suggested_removals,
            "replacements": replacements,
            "swarm_name": swarm_name,
        }

    def find_replacements(self, failing_tool: str) -> List[Dict[str, str]]:
        """
        Search for replacement tools by capability matching (schema-based)
        with keyword fallback.

        Args:
            failing_tool: Name of the tool that is performing poorly

        Returns:
            List of dicts with name, description, reason for each candidate
        """
        # 1. Try schema-based matching
        if self._tool_registry:
            failing_schema = self._tool_registry.get(failing_tool)
            if failing_schema:
                target_capabilities = set(
                    getattr(failing_schema, "producer_of", [])
                    + getattr(failing_schema, "consumer_of", [])
                )
                replacements = []
                for name, schema in self._tool_registry.items():
                    if name == failing_tool:
                        continue
                    candidate_caps = set(
                        getattr(schema, "producer_of", []) + getattr(schema, "consumer_of", [])
                    )
                    overlap = target_capabilities & candidate_caps
                    rate = getattr(schema, "success_rate", 0.5)
                    if overlap and rate > self.FAILURE_THRESHOLD:
                        replacements.append(
                            {
                                "name": name,
                                "description": getattr(schema, "description", ""),
                                "reason": f"Shares capabilities: {', '.join(overlap)}, "
                                f"success_rate={rate:.0%}",
                                "score": rate,
                            }
                        )
                replacements.sort(key=lambda r: r.get("score", 0.0), reverse=True)
                if replacements:
                    return replacements

        # 2. Fallback: keyword-based
        replacements = []
        keywords = failing_tool.lower().replace("_", " ").split()

        tool_alternatives = {
            "fetch": ["web_search", "api_query", "scrape"],
            "search": ["grep", "web_search", "semantic_search"],
            "generate": ["create", "synthesize", "compose"],
            "analyze": ["evaluate", "inspect", "assess"],
            "extract": ["parse", "mine", "collect"],
            "validate": ["verify", "check", "confirm"],
            "transform": ["convert", "normalize", "reshape"],
        }

        for keyword in keywords:
            if keyword in tool_alternatives:
                for alt in tool_alternatives[keyword]:
                    replacements.append(
                        {
                            "name": alt,
                            "description": f"Alternative for {failing_tool} (keyword: {keyword})",
                            "reason": f"{failing_tool} below {self.FAILURE_THRESHOLD*100:.0f}% success",
                            "score": 0.0,
                        }
                    )

        return replacements

    def get_active_tools(self, swarm_name: str, defaults: List[str] = None) -> List[str]:
        """
        Return merged tool list: defaults + dynamic additions - deactivated.

        Args:
            swarm_name: Name of the swarm
            defaults: Default tool list for this swarm

        Returns:
            Active tool list after applying additions and removals
        """
        active = set(defaults or [])
        active.update(self._tool_assignments.get(swarm_name, []))
        active -= set(self._deactivated_tools.get(swarm_name, []))
        return list(active)

    def update_assignments(
        self, swarm_name: str, add: List[str] = None, remove: List[str] = None
    ) -> None:
        """
        Track tool additions/removals per swarm.

        Args:
            swarm_name: Name of the swarm
            add: Tools to add to this swarm's active set
            remove: Tools to deactivate for this swarm
        """
        if add:
            if swarm_name not in self._tool_assignments:
                self._tool_assignments[swarm_name] = []
            for tool in add:
                if tool not in self._tool_assignments[swarm_name]:
                    self._tool_assignments[swarm_name].append(tool)

        if remove:
            if swarm_name not in self._deactivated_tools:
                self._deactivated_tools[swarm_name] = []
            for tool in remove:
                if tool not in self._deactivated_tools[swarm_name]:
                    self._deactivated_tools[swarm_name].append(tool)

    def auto_register_from_rates(self, tool_rates: Dict[str, Any]) -> None:
        """
        Auto-populate registry from tracked tool success rates.

        Creates ToolSchema entries for tools not yet in the registry,
        infers producer_of/consumer_of from name keywords, and computes
        success_rate from rate tuples. Updates existing entries' success_rate.

        Args:
            tool_rates: Dict of tool_name -> (successes, total_uses)
        """
        keyword_capabilities = {
            "fetch": (["raw_html", "content"], ["url"]),
            "search": (["search_results"], ["query"]),
            "generate": (["generated_content"], ["prompt"]),
            "analyze": (["analysis_result"], ["data"]),
            "extract": (["extracted_data"], ["source"]),
            "validate": (["validation_result"], ["data"]),
            "transform": (["transformed_data"], ["input_data"]),
            "parse": (["parsed_data"], ["raw_input"]),
            "scrape": (["raw_html", "structured_data"], ["url"]),
        }

        for tool_name, rate_data in tool_rates.items():
            if not isinstance(rate_data, (list, tuple)) or len(rate_data) != 2:
                continue

            successes, total = rate_data
            success_rate = successes / total if total > 0 else 0.5

            if tool_name in self._tool_registry:
                # Update existing entry's success_rate
                existing = self._tool_registry[tool_name]
                existing.success_rate = success_rate
            else:
                # Infer capabilities from name keywords
                producer_of = []
                consumer_of = []
                keywords = tool_name.lower().replace("_", " ").split()
                for keyword in keywords:
                    if keyword in keyword_capabilities:
                        produces, consumes = keyword_capabilities[keyword]
                        producer_of.extend(produces)
                        consumer_of.extend(consumes)

                # Create schema (prefer ToolSchema when available)
                desc = f"Auto-registered from usage tracking ({total} uses)"
                p_of = list(set(producer_of))
                c_of = list(set(consumer_of))
                if ToolSchema is not None:
                    schema = ToolSchema(
                        name=tool_name,
                        description=desc,
                        producer_of=p_of,
                        consumer_of=c_of,
                        success_rate=success_rate,
                    )
                else:
                    from types import SimpleNamespace

                    schema = SimpleNamespace(
                        name=tool_name,
                        description=desc,
                        producer_of=p_of,
                        consumer_of=c_of,
                        success_rate=success_rate,
                    )
                self._tool_registry[tool_name] = schema

    def to_dict(self) -> Dict:
        """Serialize tool assignments, deactivations, and registry for persistence."""
        registry_data = {}
        for name, schema in self._tool_registry.items():
            registry_data[name] = {
                "description": getattr(schema, "description", ""),
                "producer_of": getattr(schema, "producer_of", []),
                "consumer_of": getattr(schema, "consumer_of", []),
                "success_rate": getattr(schema, "success_rate", 1.0),
            }
        return {
            "tool_assignments": dict(self._tool_assignments),
            "deactivated_tools": dict(self._deactivated_tools),
            "tool_registry": registry_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolManager":
        """Restore ToolManager from persisted data."""
        instance = cls()
        instance._tool_assignments = data.get("tool_assignments", {})
        instance._deactivated_tools = data.get("deactivated_tools", {})
        # Restore registry as lightweight namespace objects if ToolSchema unavailable
        registry_data = data.get("tool_registry", {})
        for name, schema_data in registry_data.items():
            if ToolSchema is not None:
                schema = ToolSchema(
                    name=name,
                    description=schema_data.get("description", ""),
                    producer_of=schema_data.get("producer_of", []),
                    consumer_of=schema_data.get("consumer_of", []),
                    success_rate=schema_data.get("success_rate", 1.0),
                )
            else:
                # Lightweight fallback when ToolSchema class not available
                from types import SimpleNamespace

                schema = SimpleNamespace(
                    name=name,
                    description=schema_data.get("description", ""),
                    producer_of=schema_data.get("producer_of", []),
                    consumer_of=schema_data.get("consumer_of", []),
                    success_rate=schema_data.get("success_rate", 1.0),
                )
            instance._tool_registry[name] = schema
        return instance


__all__ = [
    "ToolManager",
]
