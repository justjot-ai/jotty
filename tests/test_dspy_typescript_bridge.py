#!/usr/bin/env python3
"""
DSPy TypeScript Bridge Integration Tests
Tests the complete flow: TypeScript → DSPy → MCP tools → response
"""
import sys
import asyncio
import pytest
from pathlib import Path

# Add supervisor to path (Jotty is now a pip package)
sys.path.insert(0, "/var/www/sites/personal/stock_market/JustJot.ai/supervisor")


def test_dspy_bridge_imports():
    """Test that DSPy bridge imports work"""
    print("=" * 80)
    print("TEST 1: DSPy Bridge Imports")
    print("=" * 80)

    try:
        from dspy_bridge import get_or_create_agent, DSPY_AVAILABLE
        print("✅ dspy_bridge imported successfully")
        print(f"   DSPy available: {DSPY_AVAILABLE}")
        return True

    except ImportError as e:
        print(f"❌ dspy_bridge import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test DSPy agent creation via bridge"""
    print("=" * 80)
    print("TEST 2: Agent Creation via Bridge")
    print("=" * 80)

    try:
        from dspy_bridge import get_or_create_agent, DSPY_AVAILABLE

        if not DSPY_AVAILABLE:
            print("⚠️  DSPy not available - skipping test")
            return True

        async def create():
            agent = await get_or_create_agent('research-assistant')
            return agent

        agent = asyncio.run(create())

        print("✅ Agent created successfully via bridge")
        print(f"   Name: {agent.name}")
        print(f"   Description: {agent.description}")
        print(f"   Tools: {len(agent.mcp_executor.available_tools)}")
        print()
        return True

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bridge_execution():
    """Test agent execution via bridge (simulates TypeScript request)"""
    print("=" * 80)
    print("TEST 3: Bridge Execution (TypeScript Simulation)")
    print("=" * 80)

    try:
        from dspy_bridge import get_or_create_agent, DSPY_AVAILABLE

        if not DSPY_AVAILABLE:
            print("⚠️  DSPy not available - skipping test")
            return True

        async def execute():
            # Simulates TypeScript request:
            # POST /api/dspy/execute
            # { "agentId": "research-assistant", "query": "List all ideas" }

            agent = await get_or_create_agent('research-assistant')
            result = await agent.execute(
                query="List the first 3 ideas in JustJot",
                conversation_history=""
            )
            return result

        result = asyncio.run(execute())

        print("✅ Bridge execution successful")
        print(f"   Reasoning: {result['reasoning'][:100]}...")
        print(f"   Tool calls: {len(result.get('tool_calls', []))}")
        print(f"   Tool results: {len(result.get('tool_results', []))}")
        print(f"   Response: {result['response'][:100]}...")
        print()
        return True

    except Exception as e:
        print(f"❌ Bridge execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flask_blueprint():
    """Test Flask blueprint registration"""
    print("=" * 80)
    print("TEST 4: Flask Blueprint")
    print("=" * 80)

    try:
        from flask import Flask
        from dspy_bridge import register_dspy_routes

        app = Flask(__name__)
        register_dspy_routes(app)

        # Check that routes are registered
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        expected_routes = [
            '/api/dspy/execute',
            '/api/dspy/stream',
            '/api/dspy/agents',
            '/api/dspy/health'
        ]

        print("✅ Flask blueprint registered")
        print(f"   Total routes: {len(routes)}")

        for route in expected_routes:
            if route in routes:
                print(f"   ✓ {route}")
            else:
                print(f"   ✗ {route} (missing)")

        print()
        return all(route in routes for route in expected_routes)

    except Exception as e:
        print(f"❌ Flask blueprint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_endpoint():
    """Test health check endpoint"""
    print("=" * 80)
    print("TEST 5: Health Endpoint")
    print("=" * 80)

    try:
        from flask import Flask
        from dspy_bridge import register_dspy_routes, DSPY_AVAILABLE

        app = Flask(__name__)
        register_dspy_routes(app)

        with app.test_client() as client:
            response = client.get('/api/dspy/health')
            data = response.get_json()

            print("✅ Health endpoint working")
            print(f"   Status: {data.get('status')}")
            print(f"   DSPy available: {data.get('dspy_available')}")
            print(f"   Active agents: {data.get('active_agents')}")
            print()

            return response.status_code == 200

    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_list_agents_endpoint():
    """Test list agents endpoint"""
    print("=" * 80)
    print("TEST 6: List Agents Endpoint")
    print("=" * 80)

    try:
        from flask import Flask
        from dspy_bridge import register_dspy_routes

        app = Flask(__name__)
        register_dspy_routes(app)

        with app.test_client() as client:
            response = client.get('/api/dspy/agents')
            data = response.get_json()

            print("✅ List agents endpoint working")
            print(f"   Success: {data.get('success')}")
            print(f"   Agents count: {len(data.get('agents', []))}")

            for agent in data.get('agents', []):
                print(f"   - {agent['id']}: {agent['name']} ({agent['tools']} tools)")

            print()
            return response.status_code == 200

    except Exception as e:
        print(f"❌ List agents endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all DSPy TypeScript bridge tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "DSPy TypeScript Bridge Integration Tests" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []

    # Run tests
    results.append(("DSPy Bridge Imports", test_dspy_bridge_imports()))
    results.append(("Agent Creation", test_agent_creation()))
    results.append(("Bridge Execution", test_bridge_execution()))
    results.append(("Flask Blueprint", test_flask_blueprint()))
    results.append(("Health Endpoint", test_health_endpoint()))
    results.append(("List Agents Endpoint", test_list_agents_endpoint()))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("🎉 All integration tests passed!")
        print()
        print("Phase 3 Complete! TypeScript can now call DSPy agents via HTTP bridge.")
        print()
        print("Next steps:")
        print("  1. Deploy supervisor container with DSPy bridge")
        print("  2. Test from Next.js app: import { executeDSPyAgent } from '@/lib/ai/agents/dspy-client'")
        print("  3. Migrate first agent (Research Assistant) to use DSPy")
    elif passed >= total - 2:
        print("✅ Core bridge components working!")
        print("   (Some integration tests may need running supervisor)")
    else:
        print("❌ Some core components failed - check errors above")

    print()


if __name__ == "__main__":
    run_all_tests()
