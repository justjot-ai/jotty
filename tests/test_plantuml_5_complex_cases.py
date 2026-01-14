"""
Test PlantUML Expert with 5 Complex Use Cases

Tests PlantUML expert generation including HTTP 414 (URI Too Long) scenarios.
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import PlantUMLExpertAgent, ExpertAgentConfig
from core.experts.plantuml_renderer import validate_plantuml_syntax


# 5 Complex Test Cases (including large diagrams for 414 testing)
TEST_CASES = [
    {
        "name": "1. Microservices Architecture (Large - 414 test)",
        "type": "component",
        "complexity": "high",
        "description": """Create a comprehensive component diagram showing a microservices architecture with:
- API Gateway, Auth Service, User Service, Order Service, Payment Service, Inventory Service
- Multiple databases (User DB, Order DB, Payment DB, Inventory DB)
- Message queues (RabbitMQ, Kafka)
- Service mesh (Istio)
- Monitoring (Prometheus, Grafana)
- Load balancers and reverse proxies
- Include all connections, dependencies, and relationships
- Use detailed component descriptions and ports""",
        "expected_elements": ["API Gateway", "Auth Service", "User Service", "Order Service", "Database", "Message Queue"],
        "expected_size": "large"  # Will trigger 414
    },
    {
        "name": "2. Complex State Machine (Medium)",
        "type": "state",
        "complexity": "high",
        "description": """Create a detailed state diagram for an e-commerce order processing system:
- States: Cart, Payment Processing, Inventory Check, Shipping, Delivered, Cancelled, Refunded
- Include entry/exit actions
- Show concurrent states (Payment Processing + Inventory Check)
- Include error states and recovery paths
- Show transitions with guards and actions
- Include history states for cart recovery""",
        "expected_elements": ["Cart", "Payment", "Shipping", "Delivered", "Cancelled"],
        "expected_size": "medium"
    },
    {
        "name": "3. Enterprise Class Diagram (Large - 414 test)",
        "type": "class",
        "complexity": "high",
        "description": """Create a comprehensive class diagram for an enterprise SaaS platform:
- User management: User, Role, Permission, Organization, Team, Membership
- Subscription: Plan, Subscription, Billing, Invoice, PaymentMethod
- Content: Document, Folder, Version, Comment, Tag, Share
- Analytics: Event, Metric, Dashboard, Report, Widget
- Include all attributes, methods, relationships (inheritance, composition, aggregation)
- Show visibility modifiers (+, -, #, ~)
- Include interfaces and abstract classes
- Show multiplicities and constraints""",
        "expected_elements": ["User", "Role", "Subscription", "Document", "Organization"],
        "expected_size": "large"  # Will trigger 414
    },
    {
        "name": "4. CI/CD Pipeline Sequence (Large - 414 test)",
        "type": "sequence",
        "complexity": "high",
        "description": """Create a detailed sequence diagram for a CI/CD pipeline:
- Participants: Developer, Git Repository, CI Server, Build System, Test Runner, Security Scanner, Artifact Registry, Container Registry, Kubernetes, Monitoring
- Show complete flow: Push → Build → Test → Security Scan → Package → Deploy → Monitor
- Include alt blocks for success/failure paths
- Show parallel processing (unit tests, integration tests, security scans)
- Include retry logic and error handling
- Show notifications and rollback procedures
- Include detailed messages with parameters""",
        "expected_elements": ["Developer", "CI Server", "Build", "Test", "Deploy", "Kubernetes"],
        "expected_size": "large"  # Will trigger 414
    },
    {
        "name": "5. Network Topology (Very Large - 414 test)",
        "type": "deployment",
        "complexity": "high",
        "description": """Create a comprehensive deployment diagram showing hybrid cloud infrastructure:
- On-premises: Corporate network, Firewall, VPN Gateway, Internal Load Balancer, Application Servers (10+), Database Cluster, Backup Systems
- Cloud (AWS): VPC with multiple subnets (Public, Private, Database), Internet Gateway, NAT Gateway, Application Load Balancer, Auto Scaling Groups, RDS Multi-AZ, S3, CloudFront, Route53
- Cloud (Azure): Virtual Network, Application Gateway, App Services, SQL Database, Blob Storage, CDN
- Include all network connections, security groups, routing tables
- Show data flow and replication
- Include monitoring and logging infrastructure
- Show disaster recovery sites and backup procedures""",
        "expected_elements": ["VPC", "Load Balancer", "Database", "Firewall", "VPN", "Cloud"],
        "expected_size": "very_large"  # Will definitely trigger 414
    }
]


async def test_plantuml_expert():
    """Test PlantUML expert with 5 complex cases."""
    print("=" * 80)
    print("PLANTUML EXPERT - 5 COMPLEX USE CASES TEST")
    print("=" * 80)
    print()
    
    # Check if expert was trained
    print("Checking Training Status")
    print("-" * 80)
    
    # Check for improvements/memory
    memory_file = Path("./expert_data/plantuml_expert/memory.json")
    improvements_file = Path("./expert_data/plantuml_expert/improvements.json")
    
    trained = False
    if memory_file.exists() or improvements_file.exists():
        print("✅ Found training artifacts - Expert appears to be trained")
        trained = True
    else:
        print("⚠️  No training artifacts found - Expert may not be trained")
        print("   Will test with current knowledge")
    
    print()
    
    # Create expert
    print("Creating PlantUML Expert Agent")
    print("-" * 80)
    
    try:
        from examples.claude_cli_wrapper import ClaudeCLILM
        
        # Initialize Claude CLI
        lm = ClaudeCLILM(model="sonnet")
        
        config = ExpertAgentConfig(
            name="PlantUML Expert",
            description="Expert for generating PlantUML diagrams",
            domain="plantuml"
        )
        
        expert = PlantUMLExpertAgent(config=config)
        print("✅ Expert agent created")
        
        # Quick training with default cases (if not already trained)
        print("Training expert with default cases...")
        try:
            default_gold_standards = expert._get_default_training_cases()
            if default_gold_standards:
                print(f"   Using {len(default_gold_standards)} default training cases")
                # Quick training - just pre-training to mark as trained
                await expert.train(
                    gold_standards=default_gold_standards,
                    enable_pre_training=True,
                    training_mode="pattern_extraction",  # Quick pre-training only
                    max_iterations=1  # Minimal iterations
                )
                print("✅ Expert trained (quick mode)")
            else:
                print("⚠️  No default training cases available")
        except Exception as train_error:
            print(f"⚠️  Training skipped: {train_error}")
            print("   Will attempt generation anyway")
        
        print()
    except Exception as e:
        print(f"❌ Failed to create expert: {e}")
        print("   Claude CLI may not be installed")
        print("   Install from: https://github.com/anthropics/claude-code")
        return
    
    # Test each case
    print("Testing 5 Complex Use Cases")
    print("=" * 80)
    print()
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Case {i}/5: {test_case['name']}")
        print("-" * 80)
        print(f"Type: {test_case['type']}, Complexity: {test_case['complexity']}")
        print(f"Expected Size: {test_case['expected_size']}")
        print(f"Description: {test_case['description'][:100]}...")
        print()
        
        try:
            # Generate PlantUML diagram
            print("Generating PlantUML diagram...")
            output = await expert.generate_plantuml(
                description=test_case['description'],
                diagram_type=test_case['type'],
                context={
                    "description": test_case['description'],
                    "diagram_type": test_case['type'],
                    "required_elements": test_case['expected_elements']
                },
                timeout=120  # Longer timeout for complex diagrams
            )
            
            if not output:
                print("❌ No output generated")
                results.append({
                    "case": test_case['name'],
                    "status": "failed",
                    "error": "No output"
                })
                continue
            
            # Check output size
            output_length = len(output)
            print(f"✅ Generated output: {output_length} characters, {len(output.split(chr(10)))} lines")
            
            # Validate syntax
            print("Validating syntax...")
            is_valid, error_msg, metadata = validate_plantuml_syntax(
                output,
                use_renderer=True
            )
            
            validation_method = metadata.get("validation_method", metadata.get("method", "renderer"))
            status_code = metadata.get("status_code", "N/A")
            
            if is_valid:
                print(f"✅ Valid: True, Method: {validation_method}, Status: {status_code}")
            else:
                print(f"❌ Valid: False, Method: {validation_method}, Status: {status_code}")
                print(f"   Error: {error_msg}")
            
            # Check for 414 error
            has_414 = "414" in str(status_code) or "414" in error_msg
            if has_414:
                print("⚠️  HTTP 414 detected - Large diagram (expected for this case)")
                print("   Using fallback validation (structure-based)")
            
            # Check expected elements
            found_elements = []
            for element in test_case['expected_elements']:
                if element.lower() in output.lower():
                    found_elements.append(element)
            
            element_coverage = len(found_elements) / len(test_case['expected_elements']) * 100
            print(f"📊 Elements: {len(found_elements)}/{len(test_case['expected_elements'])} ({element_coverage:.0f}%)")
            if found_elements:
                print(f"   Found: {', '.join(found_elements)}")
            
            # Check diagram type
            output_lower = output.lower()
            has_startuml = "@startuml" in output_lower or "@start" in output_lower
            has_enduml = "@enduml" in output_lower or "@end" in output_lower
            
            type_match = test_case['type'].lower() in output_lower or \
                        (test_case['type'] == "sequence" and ("sequence" in output_lower or "participant" in output_lower)) or \
                        (test_case['type'] == "class" and ("class" in output_lower)) or \
                        (test_case['type'] == "state" and ("state" in output_lower)) or \
                        (test_case['type'] == "component" and ("component" in output_lower)) or \
                        (test_case['type'] == "deployment" and ("deployment" in output_lower or "node" in output_lower))
            
            print(f"📋 Type: {test_case['type']} ({'✓' if type_match else '✗'})")
            print(f"📋 Tags: @startuml ({'✓' if has_startuml else '✗'}), @enduml ({'✓' if has_enduml else '✗'})")
            
            # Store result
            results.append({
                "case": test_case['name'],
                "status": "success" if is_valid else "validation_failed",
                "output_length": output_length,
                "lines": len(output.split("\n")),
                "is_valid": is_valid,
                "validation_method": validation_method,
                "status_code": status_code,
                "has_414": has_414,
                "element_coverage": element_coverage,
                "type_match": type_match,
                "has_tags": has_startuml and has_enduml,
                "error": error_msg if not is_valid else None
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "case": test_case['name'],
                "status": "error",
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    total = len(results)
    successful = sum(1 for r in results if r.get("is_valid", False))
    has_414_cases = sum(1 for r in results if r.get("has_414", False))
    
    print(f"Total Cases: {total}")
    print(f"✅ Valid: {successful}/{total}")
    print(f"⚠️  HTTP 414 Cases: {has_414_cases}/{total}")
    print()
    
    print("Detailed Results:")
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result.get("is_valid") else "❌"
        print(f"{status_icon} Case {i}: {result['case']}")
        print(f"   Status: {result['status']}")
        if result.get("output_length"):
            print(f"   Size: {result['output_length']} chars, {result.get('lines', 0)} lines")
        if result.get("has_414"):
            print(f"   ⚠️  HTTP 414: Handled with fallback validation")
        if result.get("element_coverage"):
            print(f"   Elements: {result['element_coverage']:.0f}%")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        print()
    
    # Save results
    results_file = Path("./test_outputs/plantuml_5_cases_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "trained": trained,
            "total_cases": total,
            "successful": successful,
            "has_414_cases": has_414_cases,
            "results": results
        }, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")
    print()
    
    # Final verdict
    print("=" * 80)
    if successful == total:
        print("✅ ALL TESTS PASSED!")
    elif successful >= total * 0.8:
        print("⚠️  MOSTLY PASSED (80%+)")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)


if __name__ == "__main__":
    print("Testing PlantUML Expert with 5 Complex Cases (Including 414 Tests)")
    print()
    asyncio.run(test_plantuml_expert())
