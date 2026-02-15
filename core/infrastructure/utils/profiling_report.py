"""
Profiling Report Generator with Gantt Chart Visualization
==========================================================

Generates comprehensive profiling reports with:
- Gantt chart timeline visualization
- Component-level timing breakdown
- Optimization recommendations
- JSON data for further analysis
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TimingEntry:
    """Single timing entry for an operation."""

    operation: str
    component: str  # e.g., "Agent", "ParameterResolution", "LLMCall", "StatePersistence"
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any]

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


class ProfilingReport:
    """Collects and generates profiling reports."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.entries: List[TimingEntry] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def record_timing(
        self, operation: str, component: str, start_time: float, end_time: float, **metadata: Any
    ) -> Any:
        """Record a timing entry."""
        entry = TimingEntry(
            operation=operation,
            component=component,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            metadata=metadata,
        )
        self.entries.append(entry)

    def set_overall_timing(self, start_time: float, end_time: float) -> None:
        """Set overall execution timing."""
        self.start_time = start_time
        self.end_time = end_time

    def generate_gantt_chart(self) -> str:
        """Generate Mermaid Gantt chart syntax."""
        if not self.entries:
            return "gantt\n    title No profiling data available"

        # Sort by start time
        sorted_entries = sorted(self.entries, key=lambda x: x.start_time)

        # Build Gantt chart
        lines = [
            "gantt",
            "    title Agent Execution Timeline",
            "    dateFormat x",  # Unix timestamp in milliseconds
            "    axisFormat %M:%S.%L",  # Minutes:Seconds.Milliseconds
            "",
        ]

        # Group by component
        components = {}
        for entry in sorted_entries:
            if entry.component not in components:
                components[entry.component] = []
            components[entry.component].append(entry)

        # Add sections for each component
        for component, entries in components.items():
            lines.append(f"    section {component}")
            for i, entry in enumerate(entries, 1):
                # Convert to milliseconds for Mermaid
                start_ms = int(entry.start_time * 1000)
                end_ms = int(entry.end_time * 1000)
                task_name = entry.metadata.get("name", entry.operation)
                lines.append(f"    {task_name} :{start_ms}, {end_ms}")

        return "\n".join(lines)

    def generate_text_gantt(self) -> str:
        """Generate ASCII-based Gantt chart for text files."""
        if not self.entries or not self.start_time or not self.end_time:
            return "No profiling data available"

        sorted_entries = sorted(self.entries, key=lambda x: x.start_time)
        total_duration = self.end_time - self.start_time

        # Use 100 characters for the timeline
        chart_width = 100

        lines = []
        lines.append("=" * 120)
        lines.append("⏱️  EXECUTION TIMELINE (Gantt Chart)")
        lines.append("=" * 120)
        lines.append(f"Total Duration: {total_duration:.2f}s")
        lines.append("")

        # Timeline header
        timeline = "Timeline: |" + "-" * chart_width + "|"
        lines.append(timeline)

        time_markers = "          0s"
        for i in range(1, 5):
            pos = int(chart_width * i / 4)
            marker_time = f"{total_duration * i / 4:.1f}s"
            time_markers += " " * (pos - len(time_markers) + 10) + marker_time
        lines.append(time_markers)
        lines.append("")

        # Group by component
        components = {}
        for entry in sorted_entries:
            if entry.component not in components:
                components[entry.component] = []
            components[entry.component].append(entry)

        # Draw each component's timeline
        for component, entries in components.items():
            lines.append(f"{component}:")
            for entry in entries:
                # Calculate position and width
                start_offset = (entry.start_time - self.start_time) / total_duration
                duration_ratio = entry.duration / total_duration

                start_pos = int(start_offset * chart_width)
                bar_width = max(1, int(duration_ratio * chart_width))

                # Create the bar
                bar = " " * 10 + "|"
                bar += " " * start_pos
                bar += "█" * bar_width

                # Add timing info
                name = entry.metadata.get("name", entry.operation)[:30]
                timing_info = f" {entry.duration:.2f}s"
                bar += timing_info

                lines.append(f"  {name:30s} {bar}")
            lines.append("")

        lines.append("=" * 120)
        return "\n".join(lines)

    def generate_component_breakdown(self) -> str:
        """Generate detailed component timing breakdown."""
        if not self.entries:
            return "No profiling data available"

        # Group by component
        component_stats = {}
        for entry in self.entries:
            if entry.component not in component_stats:
                component_stats[entry.component] = {"count": 0, "total": 0.0, "entries": []}
            component_stats[entry.component]["count"] += 1
            component_stats[entry.component]["total"] += entry.duration
            component_stats[entry.component]["entries"].append(entry)

        lines = []
        lines.append("=" * 120)
        lines.append("📊 COMPONENT BREAKDOWN")
        lines.append("=" * 120)

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            lines.append(f"Total Execution Time: {total_duration:.2f}s")
            lines.append("")

        # Sort by total time descending
        sorted_components = sorted(
            component_stats.items(), key=lambda x: x[1]["total"], reverse=True
        )

        for component, stats in sorted_components:
            avg_time = stats["total"] / stats["count"]
            lines.append(f"📦 {component}")
            lines.append(f"   Count:     {stats['count']}")
            lines.append(f"   Total:     {stats['total']:.3f}s")
            lines.append(f"   Average:   {avg_time:.3f}s")
            lines.append(f"   Min:       {min(e.duration for e in stats['entries']):.3f}s")
            lines.append(f"   Max:       {max(e.duration for e in stats['entries']):.3f}s")

            if self.start_time and self.end_time:
                percentage = (stats["total"] / total_duration) * 100
                lines.append(f"   % of Total: {percentage:.1f}%")

            # Show individual operations
            lines.append("   Operations:")
            for entry in stats["entries"]:
                name = entry.metadata.get("name", entry.operation)
                lines.append(f"      - {name}: {entry.duration:.3f}s ({entry.duration_ms:.0f}ms)")
            lines.append("")

        lines.append("=" * 120)
        return "\n".join(lines)

    def generate_optimization_recommendations(self) -> str:
        """Generate optimization recommendations based on profiling data."""
        if not self.entries:
            return "No profiling data available"

        lines = []
        lines.append("=" * 120)
        lines.append("🎯 OPTIMIZATION RECOMMENDATIONS")
        lines.append("=" * 120)
        lines.append("")

        # Analyze agent execution times
        agent_entries = [e for e in self.entries if e.component == "Agent"]
        if agent_entries:
            avg_agent_time = sum(e.duration for e in agent_entries) / len(agent_entries)
            lines.append("🤖 Agent Execution:")
            lines.append(f"   Average time: {avg_agent_time:.2f}s")

            if avg_agent_time > 5.0:
                lines.append("   ⚠️  Agents are taking >5s on average")
                lines.append("   💡 Consider using Haiku model for faster responses (1-2s vs 3-4s)")
                lines.append("   💡 Review agent signatures - simpler outputs = faster responses")
            elif avg_agent_time > 3.0:
                lines.append("   ✅ Normal range for Sonnet model (3-4s per agent)")
            else:
                lines.append("   ✅ Good performance (likely using Haiku model)")
            lines.append("")

        # Analyze parameter resolution
        param_entries = [e for e in self.entries if e.component == "ParameterResolution"]
        if param_entries:
            total_param_time = sum(e.duration for e in param_entries)
            lines.append("🔧 Parameter Resolution:")
            lines.append(f"   Total time: {total_param_time:.3f}s")
            lines.append(f"   Operations: {len(param_entries)}")

            if total_param_time > 0.5:
                lines.append("   ⚠️  Parameter resolution taking >500ms total")
                lines.append("   💡 Consider caching resolved parameters")
                lines.append("   💡 Reduce number of parameter lookups")
            else:
                lines.append("   ✅ Parameter resolution is efficient (<500ms)")
            lines.append("")

        # Analyze state persistence
        state_entries = [e for e in self.entries if e.component == "StatePersistence"]
        if state_entries:
            total_state_time = sum(e.duration for e in state_entries)
            lines.append("💾 State Persistence:")
            lines.append(f"   Total time: {total_state_time:.3f}s")
            lines.append(f"   Operations: {len(state_entries)}")

            if total_state_time > 1.0:
                lines.append("   ⚠️  State persistence taking >1s total")
                lines.append("   💡 State may be too large - consider reducing what's persisted")
                lines.append("   💡 Use async file I/O for better performance")
            else:
                lines.append("   ✅ State persistence is efficient (<1s)")
            lines.append("")

        # Overall recommendations
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            agent_time = sum(e.duration for e in agent_entries) if agent_entries else 0
            overhead = total_duration - agent_time

            lines.append("📈 Overall Analysis:")
            lines.append(f"   Total execution: {total_duration:.2f}s")
            lines.append(f"   Agent time: {agent_time:.2f}s ({agent_time/total_duration*100:.1f}%)")
            lines.append(f"   Overhead: {overhead:.2f}s ({overhead/total_duration*100:.1f}%)")
            lines.append("")

            if overhead > total_duration * 0.2:
                lines.append("   ⚠️  Orchestration overhead is >20% of total time")
                lines.append("   💡 Review conductor logic for optimization opportunities")
            else:
                lines.append("   ✅ Orchestration overhead is reasonable (<20%)")

        lines.append("")
        lines.append("=" * 120)
        return "\n".join(lines)

    def save_reports(self) -> Dict:
        """Save all profiling reports to files."""
        profiling_dir = self.output_dir / "profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save JSON data for programmatic analysis
        json_path = profiling_dir / "profiling_data.json"
        data = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": (
                (self.end_time - self.start_time) if (self.start_time and self.end_time) else None
            ),
            "entries": [asdict(entry) for entry in self.entries],
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # 2. Save Mermaid Gantt chart
        gantt_path = profiling_dir / "gantt_chart.mmd"
        with open(gantt_path, "w") as f:
            f.write(self.generate_gantt_chart())

        # 3. Save text-based timeline report
        timeline_path = profiling_dir / "execution_timeline.txt"
        with open(timeline_path, "w") as f:
            f.write(self.generate_text_gantt())
            f.write("\n\n")
            f.write(self.generate_component_breakdown())
            f.write("\n\n")
            f.write(self.generate_optimization_recommendations())

        # 4. Save markdown report
        md_path = profiling_dir / "profiling_report.md"
        with open(md_path, "w") as f:
            f.write("# Profiling Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Gantt Chart\n\n")
            f.write("```mermaid\n")
            f.write(self.generate_gantt_chart())
            f.write("\n```\n\n")

            f.write("## Component Breakdown\n\n")
            f.write("```\n")
            f.write(self.generate_component_breakdown())
            f.write("\n```\n\n")

            f.write("## Optimization Recommendations\n\n")
            f.write("```\n")
            f.write(self.generate_optimization_recommendations())
            f.write("\n```\n")

        return {
            "json": str(json_path),
            "gantt": str(gantt_path),
            "timeline": str(timeline_path),
            "markdown": str(md_path),
        }
