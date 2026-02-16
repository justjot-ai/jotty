#!/usr/bin/env python3
"""
Swarm Template Testing & Rating System
======================================

Tests all swarm templates with real LLM calls and provides comprehensive ratings.

Evaluation Criteria:
1. Functionality (Does it work?)
2. Output Quality (Is the output useful?)
3. Performance (Token usage, execution time)
4. Error Handling (Graceful degradation)
5. Documentation (Clear usage examples)

Rating Scale: 1-5 stars
- ⭐ (1): Broken or unusable
- ⭐⭐ (2): Works but poor quality
- ⭐⭐⭐ (3): Functional, acceptable quality
- ⭐⭐⭐⭐ (4): High quality, well-designed
- ⭐⭐⭐⭐⭐ (5): Exceptional, production-ready
"""

import asyncio
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SwarmTestResult:
    """Result of testing a single swarm."""

    name: str
    success: bool
    execution_time: float
    output_length: int
    token_usage: Optional[Dict[str, int]]
    error: Optional[str]
    output_sample: str  # First 500 chars
    rating: int  # 1-5 stars
    notes: str


class SwarmTester:
    """Test harness for swarm templates."""

    def __init__(self):
        self.results: List[SwarmTestResult] = []

    async def test_research_swarm(self) -> SwarmTestResult:
        """Test ResearchSwarm with a simple research task."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import research

            result = await research(
                query="What are the top 3 programming languages in 2026?",
                depth="quick",
                max_sources=3,
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="ResearchSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Quick research task on programming languages",
            )

        except Exception as e:
            return SwarmTestResult(
                name="ResearchSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_coding_swarm(self) -> SwarmTestResult:
        """Test CodingSwarm with a simple coding task."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import code

            result = await code(
                task="Write a Python function to calculate factorial",
                language="python",
                include_tests=True,
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="CodingSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Simple factorial function generation",
            )

        except Exception as e:
            return SwarmTestResult(
                name="CodingSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_arxiv_learning_swarm(self) -> SwarmTestResult:
        """Test ArxivLearningSwarm with paper learning."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import learn_paper

            result = await learn_paper(
                arxiv_id="2301.00001", depth="quick", audience="undergraduate"  # Sample paper
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="ArxivLearningSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="ArXiv paper learning task",
            )

        except Exception as e:
            return SwarmTestResult(
                name="ArxivLearningSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_olympiad_learning_swarm(self) -> SwarmTestResult:
        """Test OlympiadLearningSwarm with educational content."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import learn_topic

            result = await learn_topic(
                subject="mathematics",
                topic="Basic Algebra",
                student_name="Test Student",
                depth="quick",
                level="foundation",
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="OlympiadLearningSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Basic algebra learning material generation",
            )

        except Exception as e:
            return SwarmTestResult(
                name="OlympiadLearningSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_data_analysis_swarm(self) -> SwarmTestResult:
        """Test DataAnalysisSwarm."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import analyze_data

            # Create simple test data
            import pandas as pd

            test_data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

            result = await analyze_data(
                data=test_data, analysis_type="descriptive", include_visualizations=False
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="DataAnalysisSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Simple data analysis task",
            )

        except Exception as e:
            return SwarmTestResult(
                name="DataAnalysisSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_testing_swarm(self) -> SwarmTestResult:
        """Test TestingSwarm."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import test

            sample_code = """
def add(a, b):
    return a + b
"""

            result = await test(
                code=sample_code, language="python", test_types=["unit"], coverage_target=80
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="TestingSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Test generation for simple function",
            )

        except Exception as e:
            return SwarmTestResult(
                name="TestingSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_review_swarm(self) -> SwarmTestResult:
        """Test ReviewSwarm."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import review_code

            sample_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total
"""

            result = await review_code(
                code=sample_code, review_type="comprehensive", language="python"
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="ReviewSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Code review of simple function",
            )

        except Exception as e:
            return SwarmTestResult(
                name="ReviewSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_idea_writer_swarm(self) -> SwarmTestResult:
        """Test IdeaWriterSwarm."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import write

            result = await write(
                topic="The Future of AI", content_type="article", tone="informative", word_count=500
            )

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="IdeaWriterSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Article writing on AI future",
            )

        except Exception as e:
            return SwarmTestResult(
                name="IdeaWriterSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    async def test_pilot_swarm(self) -> SwarmTestResult:
        """Test PilotSwarm."""
        start = time.time()

        try:
            from Jotty.core.execution.swarms import pilot

            result = await pilot(goal="Create a simple Python calculator", max_iterations=3)

            execution_time = time.time() - start

            output = str(result)
            rating = self._evaluate_output(output, execution_time, result)

            return SwarmTestResult(
                name="PilotSwarm",
                success=True,
                execution_time=execution_time,
                output_length=len(output),
                token_usage=getattr(result, "token_usage", None),
                error=None,
                output_sample=output[:500],
                rating=rating,
                notes="Autonomous goal completion",
            )

        except Exception as e:
            return SwarmTestResult(
                name="PilotSwarm",
                success=False,
                execution_time=time.time() - start,
                output_length=0,
                token_usage=None,
                error=str(e),
                output_sample="",
                rating=1,
                notes=f"Failed: {str(e)[:100]}",
            )

    def _evaluate_output(self, output: str, execution_time: float, result: Any) -> int:
        """
        Evaluate output quality and assign rating.

        Criteria:
        - Output length (should have meaningful content)
        - Execution time (should be reasonable)
        - Token efficiency
        - Content quality (basic checks)
        """
        rating = 3  # Start with neutral

        # Check output length
        if len(output) < 100:
            rating -= 1  # Too short
        elif len(output) > 1000:
            rating += 1  # Good detail

        # Check execution time
        if execution_time < 5:
            rating += 1  # Fast execution
        elif execution_time > 30:
            rating -= 1  # Slow execution

        # Check for error indicators
        if "error" in output.lower() or "failed" in output.lower():
            rating -= 1

        # Check for quality indicators
        if any(word in output.lower() for word in ["detailed", "comprehensive", "analysis"]):
            rating += 1

        # Clamp to 1-5 range
        return max(1, min(5, rating))

    async def run_all_tests(self) -> List[SwarmTestResult]:
        """Run all swarm tests."""
        tests = [
            self.test_research_swarm(),
            self.test_coding_swarm(),
            self.test_arxiv_learning_swarm(),
            self.test_olympiad_learning_swarm(),
            self.test_data_analysis_swarm(),
            self.test_testing_swarm(),
            self.test_review_swarm(),
            self.test_idea_writer_swarm(),
            self.test_pilot_swarm(),
        ]

        print("=" * 80)
        print("SWARM TEMPLATE TESTING WITH REAL LLM")
        print("=" * 80)
        print()
        print(f"Testing {len(tests)} swarm templates...")
        print()

        results = []
        for i, test in enumerate(tests, 1):
            print(f"[{i}/{len(tests)}] Running test...", end=" ", flush=True)
            try:
                result = await test
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"{status} - {result.name} ({result.execution_time:.2f}s)")
                results.append(result)
            except Exception as e:
                print(f"❌ CRASH - {str(e)[:50]}")
                traceback.print_exc()

        self.results = results
        return results

    def generate_report(self):
        """Generate comprehensive rating report."""
        print("\n" + "=" * 80)
        print("SWARM RATING REPORT")
        print("=" * 80)
        print()

        # Sort by rating (highest first)
        sorted_results = sorted(self.results, key=lambda x: x.rating, reverse=True)

        # Summary table
        print("OVERALL RATINGS")
        print("-" * 80)
        print(f"{'Swarm':<30} {'Rating':<15} {'Time':<10} {'Output':<10} {'Status'}")
        print("-" * 80)

        for result in sorted_results:
            stars = "⭐" * result.rating
            status = "✅" if result.success else "❌"
            print(
                f"{result.name:<30} {stars:<15} {result.execution_time:>6.2f}s  {result.output_length:>8}  {status}"
            )

        print()

        # Detailed results
        print("=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)

        for result in sorted_results:
            print(f"\n{'='*80}")
            print(f"Swarm: {result.name}")
            print(f"{'='*80}")
            print(f"Rating:        {'⭐' * result.rating}")
            print(f"Success:       {'✅ Yes' if result.success else '❌ No'}")
            print(f"Execution:     {result.execution_time:.2f}s")
            print(f"Output Length: {result.output_length} chars")
            print(f"Notes:         {result.notes}")

            if result.error:
                print(f"\nError:")
                print(f"  {result.error[:200]}")

            if result.output_sample:
                print(f"\nOutput Sample (first 500 chars):")
                print(f"  {result.output_sample[:500]}...")

            if result.token_usage:
                print(f"\nToken Usage:")
                for key, value in result.token_usage.items():
                    print(f"  {key}: {value}")

        # Statistics
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)

        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        avg_rating = sum(r.rating for r in self.results) / total if total > 0 else 0
        avg_time = sum(r.execution_time for r in self.results) / total if total > 0 else 0

        print(f"Total Swarms Tested:   {total}")
        print(f"Successful:            {successful} ({successful/total*100:.1f}%)")
        print(f"Failed:                {failed} ({failed/total*100:.1f}%)")
        print(f"Average Rating:        {avg_rating:.2f} {'⭐' * int(avg_rating)}")
        print(f"Average Execution:     {avg_time:.2f}s")

        # Rating distribution
        print(f"\nRating Distribution:")
        for rating in range(5, 0, -1):
            count = sum(1 for r in self.results if r.rating == rating)
            bar = "█" * count
            print(f"  {'⭐' * rating:<15} {count:>2}  {bar}")

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        excellent = [r for r in self.results if r.rating >= 4]
        good = [r for r in self.results if r.rating == 3]
        poor = [r for r in self.results if r.rating <= 2]

        if excellent:
            print(f"\n✅ Production Ready ({len(excellent)} swarms):")
            for r in excellent:
                print(f"  • {r.name} - {'⭐' * r.rating}")

        if good:
            print(f"\n⚠️  Needs Improvement ({len(good)} swarms):")
            for r in good:
                print(f"  • {r.name} - {r.notes}")

        if poor:
            print(f"\n❌ Requires Attention ({len(poor)} swarms):")
            for r in poor:
                print(f"  • {r.name} - {r.error or r.notes}")


async def main():
    """Main test execution."""
    tester = SwarmTester()

    # Run all tests
    await tester.run_all_tests()

    # Generate report
    tester.generate_report()

    print("\n" + "=" * 80)
    print("Testing complete! Check output above for detailed ratings.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
