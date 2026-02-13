"""
Professional Mermaid Expert Test - 10 Complex Scenarios

Tests if Mermaid expert has learned to generate complex, professional diagrams.
"""

import asyncio
import sys
from pathlib import Path
import re
import os
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent, ExpertAgentConfig
from core.experts.memory_integration import sync_improvements_to_memory
from core.experts.mermaid_renderer import validate_mermaid_syntax
from core.memory.cortex import SwarmMemory
from core.foundation.data_structures import SwarmConfig, MemoryLevel


def configure_llm():
    """Configure DSPy with LLM."""
    try:
        import dspy
    except ImportError:
        return False
    
    try:
        from examples.claude_cli_wrapper import ClaudeCLILM
        import subprocess
        result = subprocess.run(["claude", "--version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            lm = ClaudeCLILM(model="sonnet")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude CLI")
            return True
    except:
        pass
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude API")
            return True
        except:
            pass
    
    return False


# validate_mermaid_syntax is now imported from mermaid_renderer module


# 10 Professional Mermaid Scenarios
PROFESSIONAL_SCENARIOS = [
    {
        "name": "Microservices Architecture",
        "description": "Create a graph TD diagram with multiple subgraphs representing different VPCs, Load Balancers, and Auth services. Show a microservices architecture with at least 3 VPCs, load balancers, and authentication services.",
        "diagram_type": "flowchart",
        "expected_elements": ["VPC", "Load Balancer", "Auth", "subgraph"],
        "complexity": "high"
    },
    {
        "name": "Global CI/CD Pipeline",
        "description": "Create a stateDiagram-v2 showing parallel processing, error handling loops, and deployment gates. Include states for Build, Test, Deploy, and error handling states with transitions.",
        "diagram_type": "stateDiagram-v2",
        "expected_elements": ["Build", "Test", "Deploy", "Error", "parallel"],
        "complexity": "high"
    },
    {
        "name": "E-commerce Order Lifecycle",
        "description": "Create a detailed sequenceDiagram involving a User, Frontend, API Gateway, Payment Processor, Inventory DB, and Shipping Provider with alt/else logic for payment success/failure and inventory availability.",
        "diagram_type": "sequenceDiagram",
        "expected_elements": ["User", "Frontend", "API Gateway", "Payment Processor", "Inventory DB", "Shipping Provider", "alt", "else"],
        "complexity": "high"
    },
    {
        "name": "Project Management Gantt Chart",
        "description": "Create a gantt chart with multiple sections, dependencies (after), and milestones for a 6-month software launch project. Include phases like Planning, Development, Testing, and Launch with dependencies.",
        "diagram_type": "gantt",
        "expected_elements": ["Planning", "Development", "Testing", "Launch", "after", "milestone"],
        "complexity": "high"
    },
    {
        "name": "Database Entity Relationship (ERD)",
        "description": "Create an erDiagram for a complex SaaS platform involving Users, Subscriptions, Permissions, and Multi-tenancy relationships. Show one-to-many and many-to-many relationships.",
        "diagram_type": "erDiagram",
        "expected_elements": ["Users", "Subscriptions", "Permissions", "Multi-tenancy", "||--o{", "}o--||"],
        "complexity": "high"
    },
    {
        "name": "Git Flow Strategy",
        "description": "Create a gitGraph showing main, develop, feature branches, hotfixes, and complex merges/rebases. Show branching strategy with feature branches merging to develop, and hotfixes merging to main.",
        "diagram_type": "gitGraph",
        "expected_elements": ["main", "develop", "feature", "hotfix", "merge", "commit"],
        "complexity": "high"
    },
    {
        "name": "Customer Support Decision Tree",
        "description": "Create a flowchart LR with nested logic for an automated AI support bot. Include decision nodes for different customer issues and routing logic.",
        "diagram_type": "flowchart",
        "expected_elements": ["decision", "routing", "AI support", "customer issue"],
        "complexity": "medium"
    },
    {
        "name": "Network Topology",
        "description": "Create a graph showing a hybrid cloud setup with On-Prem servers, VPN tunnels, and Cloud Resources. Show connections between on-premise infrastructure and cloud resources via VPN.",
        "diagram_type": "graph",
        "expected_elements": ["On-Prem", "VPN", "Cloud", "servers"],
        "complexity": "medium"
    },
    {
        "name": "User Journey Map",
        "description": "Create a journey diagram highlighting friction points in a multi-step financial onboarding process. Include stages like Sign Up, Verification, KYC, Account Setup, and First Transaction.",
        "diagram_type": "journey",
        "expected_elements": ["Sign Up", "Verification", "KYC", "Account Setup", "friction"],
        "complexity": "high"
    },
    {
        "name": "Class Diagram for Design Patterns",
        "description": "Create a classDiagram illustrating a complex pattern like 'Observer' or 'Strategy' with inheritance and member visibility. Show Subject, Observer classes with +public and -private members, and inheritance relationships.",
        "diagram_type": "classDiagram",
        "expected_elements": ["Observer", "Subject", "Strategy", "+", "-", "<|--"],
        "complexity": "high"
    }
]


async def test_professional_scenarios(use_renderer: bool = True, max_scenarios: int = 10):
    """
    Test Mermaid expert with professional scenarios.
    
    Args:
        use_renderer: If True, use renderer API for validation (slower but accurate)
        max_scenarios: Maximum number of scenarios to test (default: 10)
    """
    print("=" * 80)
    print(f"PROFESSIONAL MERMAID EXPERT TEST - {min(max_scenarios, len(PROFESSIONAL_SCENARIOS))} SCENARIOS")
    print(f"Renderer validation: {'ENABLED' if use_renderer else 'DISABLED (basic checks)'}")
    print("=" * 80)
    print()
    
    if not configure_llm():
        print("❌ Cannot proceed without LLM")
        return []
    print()
    
    # Create memory system
    print("1. Setting up Memory System")
    print("-" * 80)
    memory_config = SwarmConfig()
    memory = SwarmMemory(
        agent_name="mermaid_professional_test",
        config=memory_config
    )
    print("✅ Memory system created")
    print()
    
    # Create expert with memory
    print("2. Creating Mermaid Expert Agent")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="mermaid_professional",
        domain="mermaid",
        description="Mermaid expert for professional diagrams",
        use_memory_storage=True,
        use_memory_synthesis=False,  # Use raw improvements for now
        expert_data_dir="./test_outputs/mermaid_professional"
    )
    
    expert = MermaidExpertAgent(config=config, memory=memory)
    
    # Load existing improvements from file and sync to memory
    print("   Loading improvements from file...")
    improvements_file = Path("./test_outputs/mermaid_complex_memory/improvements.json")
    if improvements_file.exists():
        try:
            with open(improvements_file) as f:
                file_improvements = json.load(f)
            
            # Sync to memory
            synced = sync_improvements_to_memory(
                memory=memory,
                improvements=file_improvements,
                expert_name="mermaid_professional",
                domain="mermaid"
            )
            print(f"   ✅ Synced {synced}/{len(file_improvements)} improvements to memory")
            
            # Reload improvements from memory
            expert.improvements = expert._load_improvements()
            print(f"   ✅ Loaded {len(expert.improvements)} improvements from memory")
        except Exception as e:
            print(f"   ⚠️  Failed to sync improvements: {e}")
            expert.improvements = expert._load_improvements()
    else:
        print(f"   ⚠️  No improvements file found at {improvements_file}")
        expert.improvements = expert._load_improvements()
    
    # Mark as trained for testing
    expert.trained = True
    
    print(f"✅ Expert created")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    print(f"   Memory storage: {expert.use_memory_storage}")
    
    # Show what improvements are being used
    if expert.improvements:
        print(f"\n   Sample Learned Patterns:")
        for i, imp in enumerate(expert.improvements[:5], 1):
            pattern = imp.get('learned_pattern', '')
            if pattern:
                print(f"     {i}. {pattern[:120]}...")
    else:
        print(f"\n   ⚠️  No improvements loaded - expert will generate from scratch")
    print()
    
    # Test scenarios
    scenarios_to_test = PROFESSIONAL_SCENARIOS[:max_scenarios]
    print(f"3. Testing {len(scenarios_to_test)} Professional Scenarios")
    print("-" * 80)
    print()
    
    results = []
    for i, scenario in enumerate(scenarios_to_test, 1):
        print(f"   Scenario {i}/10: {scenario['name']}")
        print(f"   Type: {scenario['diagram_type']}, Complexity: {scenario['complexity']}")
        print(f"   Description: {scenario['description'][:80]}...")
        
        try:
            diagram = await asyncio.wait_for(
                expert.generate_mermaid(
                    description=scenario['description'],
                    diagram_type=scenario['diagram_type']
                ),
                timeout=120  # Increased timeout for complex diagrams
            )
            
            # Validate syntax (renderer or basic)
            if use_renderer:
                print(f"      Validating via renderer...", end="", flush=True)
                try:
                    is_valid, error_msg, metadata = validate_mermaid_syntax(diagram, use_renderer=True)
                    print(f" ✓")
                except Exception as renderer_error:
                    # Fallback to basic validation if renderer fails
                    print(f" ✗ (fallback)")
                    is_valid, error_msg, metadata = validate_mermaid_syntax(diagram, use_renderer=False)
            else:
                is_valid, error_msg, metadata = validate_mermaid_syntax(diagram, use_renderer=False)
            
            # Check for expected elements
            found_elements = []
            for element in scenario['expected_elements']:
                if element.lower() in diagram.lower():
                    found_elements.append(element)
            
            element_coverage = len(found_elements) / len(scenario['expected_elements'])
            
            # Check diagram type match
            type_match = metadata['diagram_type'] == scenario['diagram_type'] or \
                        (scenario['diagram_type'] == 'flowchart' and metadata['diagram_type'] in ['graph', 'flowchart'])
            
            # Check complexity features
            has_complexity = False
            if scenario['complexity'] == 'high':
                has_complexity = metadata.get('has_subgraphs', False) or \
                               metadata.get('has_alt_else', False) or \
                               metadata.get('has_parallel', False) or \
                               metadata['lines'] > 10
            
            result = {
                "scenario": scenario['name'],
                "valid": is_valid,
                "error": error_msg if not is_valid else None,
                "diagram_type": metadata['diagram_type'],
                "type_match": type_match,
                "elements_found": found_elements,
                "element_coverage": element_coverage,
                "has_complexity": has_complexity,
                "lines": metadata['lines'],
                "success": is_valid and type_match and element_coverage >= 0.6 and \
                          (scenario['complexity'] == 'medium' or has_complexity),
                "diagram_preview": diagram[:200] + "..." if len(diagram) > 200 else diagram
            }
            results.append(result)
            
            # Print result
            status = "✅" if result['success'] else "⚠️" if result['valid'] else "❌"
            print(f"   {status} Valid: {is_valid}, Type: {metadata['diagram_type']} ({'✓' if type_match else '✗'}), Elements: {len(found_elements)}/{len(scenario['expected_elements'])} ({element_coverage:.0%})")
            if result['elements_found']:
                print(f"      Found: {', '.join(result['elements_found'][:5])}")
            if not is_valid:
                print(f"      Error: {error_msg}")
            if result['lines']:
                print(f"      Lines: {result['lines']}, Complexity features: {has_complexity}")
            
        except asyncio.TimeoutError:
            print(f"   ⏱️  Timeout")
            results.append({
                "scenario": scenario['name'],
                "valid": False,
                "error": "Timeout",
                "success": False
            })
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            results.append({
                "scenario": scenario['name'],
                "valid": False,
                "error": str(e)[:100],
                "success": False
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r.get('success', False))
    valid_syntax = sum(1 for r in results if r.get('valid', False))
    type_matches = sum(1 for r in results if r.get('type_match', False))
    avg_coverage = sum(r.get('element_coverage', 0) for r in results) / len(results) if results else 0
    avg_lines = sum(r.get('lines', 0) for r in results) / len(results) if results else 0
    
    total_scenarios = len(scenarios_to_test)
    print(f"✅ Successful scenarios: {successful}/{total_scenarios}")
    print(f"✅ Valid syntax: {valid_syntax}/{total_scenarios}")
    print(f"✅ Correct diagram type: {type_matches}/{total_scenarios}")
    print(f"✅ Average element coverage: {avg_coverage:.1%}")
    print(f"✅ Average diagram size: {avg_lines:.1f} lines")
    print(f"✅ Improvements used: {len(expert.improvements)}")
    print()
    
    print("Detailed Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "✅" if result.get('success') else "⚠️" if result.get('valid') else "❌"
        print(f"{i:2d}. {status} {result['scenario']}")
        print(f"     Type: {result.get('diagram_type', 'unknown')} ({'✓' if result.get('type_match') else '✗'})")
        if result.get('elements_found'):
            print(f"     Elements: {len(result['elements_found'])}/{len(PROFESSIONAL_SCENARIOS[i-1]['expected_elements'])} found")
            print(f"       {', '.join(result['elements_found'][:5])}")
        if result.get('error'):
            print(f"     Error: {result['error'][:60]}")
        if result.get('lines'):
            print(f"     Size: {result['lines']} lines")
    print()
    
    # Show learned patterns
    print("Learned Patterns Being Used:")
    print("-" * 80)
    if expert.improvements:
        for i, imp in enumerate(expert.improvements[:5], 1):
            pattern = imp.get('learned_pattern', '')
            if pattern:
                print(f"{i}. {pattern[:150]}...")
    else:
        print("No improvements loaded")
    print()
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Mermaid Expert Professional Scenarios')
    parser.add_argument('--no-renderer', action='store_true', help='Disable renderer validation (faster)')
    parser.add_argument('--max-scenarios', type=int, default=10, help='Maximum scenarios to test')
    args = parser.parse_args()
    
    results = asyncio.run(test_professional_scenarios(
        use_renderer=not args.no_renderer,
        max_scenarios=args.max_scenarios
    ))
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    threshold = int(total * 0.8)
    print(f"\n{'✅ Expert PASSED' if successful >= threshold else '⚠️  Expert PARTIAL' if successful >= int(total * 0.6) else '❌ Expert FAILED'}: {successful}/{total} professional scenarios successful")
    sys.exit(0 if successful >= threshold else 1)
