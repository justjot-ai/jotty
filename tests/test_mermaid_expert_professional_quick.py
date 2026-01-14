"""
Quick Professional Mermaid Test - 3 Scenarios with Renderer Validation

Tests if improvements are loaded and renderer validation works.
"""

import asyncio
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent, ExpertAgentConfig
from core.experts.memory_integration import sync_improvements_to_memory
from core.experts.mermaid_renderer import validate_mermaid_syntax
from core.memory.cortex import HierarchicalMemory
from core.foundation.data_structures import JottyConfig


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


QUICK_SCENARIOS = [
    {
        "name": "Microservices Architecture",
        "description": "Create a graph TD diagram with multiple subgraphs representing different VPCs, Load Balancers, and Auth services.",
        "diagram_type": "flowchart",
        "expected_elements": ["VPC", "Load Balancer", "Auth", "subgraph"]
    },
    {
        "name": "E-commerce Order Lifecycle",
        "description": "Create a detailed sequenceDiagram involving a User, Frontend, API Gateway, Payment Processor, Inventory DB, and Shipping Provider with alt/else logic.",
        "diagram_type": "sequenceDiagram",
        "expected_elements": ["User", "Frontend", "API Gateway", "Payment Processor", "alt", "else"]
    },
    {
        "name": "User Journey Map",
        "description": "Create a journey diagram highlighting friction points in a multi-step financial onboarding process.",
        "diagram_type": "journey",
        "expected_elements": ["Sign Up", "Verification", "KYC", "friction"]
    }
]


async def test_quick():
    """Quick test with 3 scenarios."""
    print("=" * 80)
    print("QUICK MERMAID PROFESSIONAL TEST - 3 SCENARIOS WITH RENDERER")
    print("=" * 80)
    print()
    
    import os
    if not configure_llm():
        print("❌ Cannot proceed without LLM")
        return []
    print()
    
    # Create memory system
    print("1. Setting up Memory System")
    print("-" * 80)
    memory_config = JottyConfig()
    memory = HierarchicalMemory(
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
        use_memory_synthesis=False,
        expert_data_dir="./test_outputs/mermaid_professional"
    )
    
    expert = MermaidExpertAgent(config=config, memory=memory)
    
    # Load and sync improvements
    print("   Loading improvements from file...")
    improvements_file = Path("./test_outputs/mermaid_complex_memory/improvements.json")
    if improvements_file.exists():
        try:
            with open(improvements_file) as f:
                file_improvements = json.load(f)
            
            synced = sync_improvements_to_memory(
                memory=memory,
                improvements=file_improvements,
                expert_name="mermaid_professional",
                domain="mermaid"
            )
            print(f"   ✅ Synced {synced}/{len(file_improvements)} improvements to memory")
            
            expert.improvements = expert._load_improvements()
            print(f"   ✅ Loaded {len(expert.improvements)} improvements from memory")
        except Exception as e:
            print(f"   ⚠️  Failed to sync improvements: {e}")
            expert.improvements = expert._load_improvements()
    else:
        print(f"   ⚠️  No improvements file found at {improvements_file}")
        expert.improvements = expert._load_improvements()
    
    expert.trained = True
    
    print(f"✅ Expert created")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    
    if expert.improvements:
        print(f"\n   Sample Learned Patterns:")
        for i, imp in enumerate(expert.improvements[:3], 1):
            pattern = imp.get('learned_pattern', '')
            if pattern:
                print(f"     {i}. {pattern[:100]}...")
    print()
    
    # Test scenarios
    print("3. Testing 3 Scenarios with Renderer Validation")
    print("-" * 80)
    print()
    
    results = []
    for i, scenario in enumerate(QUICK_SCENARIOS, 1):
        print(f"   Scenario {i}/3: {scenario['name']}")
        print(f"   Type: {scenario['diagram_type']}")
        
        try:
            diagram = await asyncio.wait_for(
                expert.generate_mermaid(
                    description=scenario['description'],
                    diagram_type=scenario['diagram_type']
                ),
                timeout=60
            )
            
            # Validate via renderer
            print(f"   Validating via renderer...")
            is_valid, error_msg, metadata = validate_mermaid_syntax(diagram, use_renderer=True)
            
            # Check elements
            found_elements = [e for e in scenario['expected_elements'] if e.lower() in diagram.lower()]
            element_coverage = len(found_elements) / len(scenario['expected_elements'])
            type_match = metadata['diagram_type'] == scenario['diagram_type'] or \
                        (scenario['diagram_type'] == 'flowchart' and metadata['diagram_type'] in ['graph', 'flowchart'])
            
            result = {
                "scenario": scenario['name'],
                "valid": is_valid,
                "error": error_msg if not is_valid else None,
                "diagram_type": metadata['diagram_type'],
                "type_match": type_match,
                "elements_found": found_elements,
                "element_coverage": element_coverage,
                "success": is_valid and type_match and element_coverage >= 0.6
            }
            results.append(result)
            
            status = "✅" if result['success'] else "⚠️" if result['valid'] else "❌"
            print(f"   {status} Valid: {is_valid}, Type: {metadata['diagram_type']} ({'✓' if type_match else '✗'}), Elements: {len(found_elements)}/{len(scenario['expected_elements'])} ({element_coverage:.0%})")
            if result['elements_found']:
                print(f"      Found: {', '.join(result['elements_found'][:5])}")
            if not is_valid:
                print(f"      Error: {error_msg}")
            
        except asyncio.TimeoutError:
            print(f"   ⏱️  Timeout")
            results.append({"scenario": scenario['name'], "valid": False, "error": "Timeout", "success": False})
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            results.append({"scenario": scenario['name'], "valid": False, "error": str(e)[:100], "success": False})
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r.get('success', False))
    valid_syntax = sum(1 for r in results if r.get('valid', False))
    print(f"✅ Successful: {successful}/3")
    print(f"✅ Valid syntax (via renderer): {valid_syntax}/3")
    print(f"✅ Improvements used: {len(expert.improvements)}")
    print()
    
    return results


if __name__ == "__main__":
    results = asyncio.run(test_quick())
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n{'✅ PASSED' if successful >= 2 else '⚠️  PARTIAL' if successful >= 1 else '❌ FAILED'}: {successful}/3 scenarios")
    sys.exit(0 if successful >= 2 else 1)
