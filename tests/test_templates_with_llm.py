"""
Test templates with real LLM execution.

This tests the simple stub templates to verify they work end-to-end.
"""

import asyncio
import os
from core.intelligence.swarms.templates import (
    ReviewTemplate,
    MLTemplate,
    TestingTemplate,
    DataAnalysisTemplate,
)

# Requires ANTHROPIC_API_KEY env var
assert os.getenv("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY to run real LLM tests"


async def test_review_template():
    """Test ReviewTemplate with simple code review."""
    print("\n" + "=" * 80)
    print("TEST 1: ReviewTemplate - Code Review")
    print("=" * 80)

    template = ReviewTemplate()

    code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
"""

    try:
        result = await template.execute(
            query="Review this code",
            code=code,
            language="Python",
            focus_areas=["quality", "security", "performance"],
        )

        print(f"✅ Success: {result.success}")
        print(f"✅ Domain: {result.domain}")
        print(f"✅ Swarm: {result.swarm_name}")
        print(f"✅ Execution time: {result.execution_time:.2f}s")
        print(f"\nOutput keys: {list(result.output.keys())}")
        print(f"Code analyzed: {result.output.get('code_analyzed')}")
        print(f"Language: {result.output.get('language')}")
        print(f"Focus areas: {result.output.get('focus_areas')}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_ml_template():
    """Test MLTemplate with simple ML task."""
    print("\n" + "=" * 80)
    print("TEST 2: MLTemplate - Machine Learning")
    print("=" * 80)

    template = MLTemplate()

    try:
        result = await template.execute(
            query="Train a model to predict customer churn",
            data="customer_data.csv",
            target="churn",
            model_type="xgboost",
        )

        print(f"✅ Success: {result.success}")
        print(f"✅ Domain: {result.domain}")
        print(f"✅ Swarm: {result.swarm_name}")
        print(f"✅ Execution time: {result.execution_time:.2f}s")
        print(f"\nOutput keys: {list(result.output.keys())}")
        print(f"Model type: {result.output.get('model_type')}")
        print(f"Target: {result.output.get('target')}")
        print(f"Status: {result.output.get('status')}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_testing_template():
    """Test TestingTemplate with test generation."""
    print("\n" + "=" * 80)
    print("TEST 3: TestingTemplate - Test Generation")
    print("=" * 80)

    template = TestingTemplate()

    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    try:
        result = await template.execute(
            query="Generate tests for this code", code=code, framework="pytest", language="Python"
        )

        print(f"✅ Success: {result.success}")
        print(f"✅ Domain: {result.domain}")
        print(f"✅ Swarm: {result.swarm_name}")
        print(f"✅ Execution time: {result.execution_time:.2f}s")
        print(f"\nOutput keys: {list(result.output.keys())}")
        print(f"Code tested: {result.output.get('code_tested')}")
        print(f"Framework: {result.output.get('framework')}")
        print(f"Test code preview: {result.output.get('test_code', '')[:100]}...")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_data_analysis_template():
    """Test DataAnalysisTemplate."""
    print("\n" + "=" * 80)
    print("TEST 4: DataAnalysisTemplate - Data Analysis")
    print("=" * 80)

    template = DataAnalysisTemplate()

    try:
        result = await template.execute(
            query="Analyze sales trends", data="sales_data.csv", analysis_type="trend_analysis"
        )

        print(f"✅ Success: {result.success}")
        print(f"✅ Domain: {result.domain}")
        print(f"✅ Swarm: {result.swarm_name}")
        print(f"✅ Execution time: {result.execution_time:.2f}s")
        print(f"\nOutput keys: {list(result.output.keys())}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING TEMPLATES WITH REAL LLM")
    print("=" * 80)
    print(f"API Key: {'sk-ant-api03-...' + os.environ.get('ANTHROPIC_API_KEY', '')[-20:]}")

    results = []

    # Test simple templates (these use placeholder implementations, no LLM needed)
    results.append(("ReviewTemplate", await test_review_template()))
    results.append(("MLTemplate", await test_ml_template()))
    results.append(("TestingTemplate", await test_testing_template()))
    results.append(("DataAnalysisTemplate", await test_data_analysis_template()))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
