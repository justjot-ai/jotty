"""
Test LangGraph Integration
===========================

Tests both dynamic and static modes with simple agents.
"""

import asyncio
import sys
import os

# Add Jotty to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available. Install with: pip install dspy-ai")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not available. Install with: pip install langgraph langchain-core")
    sys.exit(1)

from Jotty.core.orchestration.conductor import Conductor
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.data_structures import JottyConfig


# Simple DSPy agents for testing
class ResearchAgent(dspy.Module):
    """Simple research agent."""
    
    def __init__(self):
        super().__init__()
        self.signature = "query -> research_result"
    
    def forward(self, query: str) -> str:
        """Research the query."""
        return f"Research results for: {query}"


class AnalyzeAgent(dspy.Module):
    """Simple analysis agent."""
    
    def __init__(self):
        super().__init__()
        self.signature = "research_data -> analysis"
    
    def forward(self, research_data: str) -> str:
        """Analyze research data."""
        return f"Analysis of: {research_data}"


class ReportAgent(dspy.Module):
    """Simple report agent."""
    
    def __init__(self):
        super().__init__()
        self.signature = "analysis -> report"
    
    def forward(self, analysis: str) -> str:
        """Generate report from analysis."""
        return f"Report: {analysis}"


async def test_dynamic_mode():
    """Test dynamic mode with dependencies."""
    print("\n" + "="*60)
    print("TEST 1: Dynamic Mode (Dependency-Based)")
    print("="*60)
    
    # Configure DSPy (use a simple LM for testing)
    try:
        # Try to use default LM if configured
        lm = dspy.settings.lm
        if not lm:
            print("⚠️  No DSPy LM configured. Using mock.")
            # Create a simple mock LM
            class MockLM:
                def __call__(self, *args, **kwargs):
                    return "Mock response"
            dspy.configure(lm=MockLM())
    except:
        print("⚠️  Using mock LM for testing")
        class MockLM:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        dspy.configure(lm=MockLM())
    
    # Create agents with dependencies
    agents = [
        AgentConfig(
            name="ResearchAgent",
            agent=ResearchAgent(),
            dependencies=["AnalyzeAgent"],  # AnalyzeAgent depends on ResearchAgent
        ),
        AgentConfig(
            name="AnalyzeAgent",
            agent=AnalyzeAgent(),
            dependencies=["ReportAgent"],  # ReportAgent depends on AnalyzeAgent
        ),
        AgentConfig(
            name="ReportAgent",
            agent=ReportAgent(),
        ),
    ]
    
    # Create conductor with dynamic mode
    try:
        conductor = Conductor(
            actors=agents,
            metadata_provider=None,
            config=JottyConfig(),
            use_langgraph=True,
            langgraph_mode="dynamic"
        )
        
        print("✅ Conductor created with dynamic LangGraph mode")
        print(f"   Agents: {[a.name for a in agents]}")
        print(f"   Dependencies: ResearchAgent → AnalyzeAgent → ReportAgent")
        
        # Run
        print("\n🚀 Running workflow...")
        result = await conductor.run(
            goal="Research and analyze AI trends",
            query="What are the latest AI trends?"
        )
        
        print("\n✅ Dynamic mode test completed!")
        print(f"   Success: {result.success}")
        print(f"   Completed agents: {result.metadata.get('completed_agents', [])}")
        print(f"   Mode: {result.metadata.get('langgraph_mode', 'unknown')}")
        print(f"   Output preview: {str(result.final_output)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Dynamic mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_static_mode():
    """Test static mode with explicit order."""
    print("\n" + "="*60)
    print("TEST 2: Static Mode (Explicit Order)")
    print("="*60)
    
    # Configure DSPy
    try:
        lm = dspy.settings.lm
        if not lm:
            class MockLM:
                def __call__(self, *args, **kwargs):
                    return "Mock response"
            dspy.configure(lm=MockLM())
    except:
        class MockLM:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        dspy.configure(lm=MockLM())
    
    # Create agents (no dependencies needed for static mode)
    agents = [
        AgentConfig(name="ResearchAgent", agent=ResearchAgent()),
        AgentConfig(name="AnalyzeAgent", agent=AnalyzeAgent()),
        AgentConfig(name="ReportAgent", agent=ReportAgent()),
    ]
    
    # Create conductor with static mode
    try:
        conductor = Conductor(
            actors=agents,
            metadata_provider=None,
            config=JottyConfig(),
            use_langgraph=True,
            langgraph_mode="static",
            agent_order=["ResearchAgent", "AnalyzeAgent", "ReportAgent"]
        )
        
        print("✅ Conductor created with static LangGraph mode")
        print(f"   Agents: {[a.name for a in agents]}")
        print(f"   Order: ResearchAgent → AnalyzeAgent → ReportAgent")
        
        # Run
        print("\n🚀 Running workflow...")
        result = await conductor.run(
            goal="Research, analyze, and report on AI trends",
            query="What are the latest AI trends?"
        )
        
        print("\n✅ Static mode test completed!")
        print(f"   Success: {result.success}")
        print(f"   Completed agents: {result.metadata.get('completed_agents', [])}")
        print(f"   Mode: {result.metadata.get('langgraph_mode', 'unknown')}")
        print(f"   Output preview: {str(result.final_output)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Static mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_runtime_override():
    """Test runtime mode override."""
    print("\n" + "="*60)
    print("TEST 3: Runtime Mode Override")
    print("="*60)
    
    # Configure DSPy
    try:
        lm = dspy.settings.lm
        if not lm:
            class MockLM:
                def __call__(self, *args, **kwargs):
                    return "Mock response"
            dspy.configure(lm=MockLM())
    except:
        class MockLM:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        dspy.configure(lm=MockLM())
    
    agents = [
        AgentConfig(name="ResearchAgent", agent=ResearchAgent()),
        AgentConfig(name="AnalyzeAgent", agent=AnalyzeAgent()),
        AgentConfig(name="ReportAgent", agent=ReportAgent()),
    ]
    
    try:
        # Create with dynamic mode
        conductor = Conductor(
            actors=agents,
            metadata_provider=None,
            config=JottyConfig(),
            use_langgraph=True,
            langgraph_mode="dynamic"
        )
        
        print("✅ Conductor created with dynamic mode (default)")
        
        # Override to static at runtime
        print("\n🔄 Overriding to static mode at runtime...")
        result = await conductor.run(
            goal="Task with runtime override",
            mode="static",
            agent_order=["ResearchAgent", "AnalyzeAgent", "ReportAgent"]
        )
        
        print("\n✅ Runtime override test completed!")
        print(f"   Success: {result.success}")
        print(f"   Mode: {result.metadata.get('langgraph_mode', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Runtime override test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LANGGRAPH INTEGRATION TESTS")
    print("="*60)
    
    if not DSPY_AVAILABLE:
        print("❌ DSPy not available")
        return
    
    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph not available")
        return
    
    results = []
    
    # Test 1: Dynamic mode
    try:
        result1 = await test_dynamic_mode()
        results.append(("Dynamic Mode", result1))
    except Exception as e:
        print(f"❌ Dynamic mode test crashed: {e}")
        results.append(("Dynamic Mode", False))
    
    # Test 2: Static mode
    try:
        result2 = await test_static_mode()
        results.append(("Static Mode", result2))
    except Exception as e:
        print(f"❌ Static mode test crashed: {e}")
        results.append(("Static Mode", False))
    
    # Test 3: Runtime override
    try:
        result3 = await test_runtime_override()
        results.append(("Runtime Override", result3))
    except Exception as e:
        print(f"❌ Runtime override test crashed: {e}")
        results.append(("Runtime Override", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
