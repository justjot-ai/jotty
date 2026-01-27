"""
Test Tool Collection Functionality

Tests ToolCollection loading from various sources and integration with SkillsRegistry.
"""
import sys
import tempfile
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.registry import ToolCollection, get_skills_registry


def test_local_collection():
    """Test loading collection from local directory."""
    print("=== Test 1: Local Collection ===\n")
    
    try:
        # Load from local skills directory
        collection = ToolCollection.from_local("./skills")
        
        assert len(collection) > 0, "Collection should have tools"
        assert collection.source == "local", f"Expected source='local', got '{collection.source}'"
        
        print(f"✅ Loaded {len(collection)} tools from local")
        print(f"✅ Source: {collection.source}")
        print(f"✅ Metadata: {collection.metadata}")
        
        # List tools
        tools = collection.list_tools()
        assert len(tools) > 0, "Should have tools"
        print(f"✅ Listed {len(tools)} tools")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collection_to_skill_definitions():
    """Test converting collection to SkillDefinitions."""
    print("\n=== Test 2: Collection to SkillDefinitions ===\n")
    
    try:
        # Create a simple collection
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "forward": lambda x: f"Result: {x}"
            }
        ]
        
        collection = ToolCollection(tools=tools, source="test")
        
        # Convert to SkillDefinitions
        skill_defs = collection.to_skill_definitions()
        
        assert len(skill_defs) > 0, "Should have skill definitions"
        print(f"✅ Converted {len(skill_defs)} tools to SkillDefinitions")
        
        # Check first skill
        skill_def = skill_defs[0]
        assert hasattr(skill_def, 'name'), "SkillDefinition should have name"
        assert hasattr(skill_def, 'tools'), "SkillDefinition should have tools"
        print(f"✅ First skill: {skill_def.name}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry_integration():
    """Test loading collection into SkillsRegistry."""
    print("\n=== Test 3: Registry Integration ===\n")
    
    try:
        registry = get_skills_registry()
        registry.init()
        
        # Load local collection
        collection = ToolCollection.from_local("./skills")
        
        # Load into registry
        tools = registry.load_collection(collection, collection_name="test_collection")
        
        assert len(tools) > 0, "Should have loaded tools"
        print(f"✅ Loaded {len(tools)} tools into registry")
        
        # List collections
        collections = registry.list_collections()
        assert len(collections) > 0, "Should have collections"
        print(f"✅ Found {len(collections)} collections in registry")
        
        # Get collection
        coll_info = registry.get_collection("test_collection")
        assert coll_info is not None, "Should find collection"
        print(f"✅ Retrieved collection: {coll_info['source']}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_and_load():
    """Test saving and loading collection."""
    print("\n=== Test 4: Save and Load ===\n")
    
    try:
        # Create a simple collection
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "source": "test"
            }
        ]
        
        collection = ToolCollection(tools=tools, source="test")
        
        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            collection.save_to_local(tmpdir)
            
            # Check file exists
            collection_json = Path(tmpdir) / "collection.json"
            assert collection_json.exists(), "collection.json should exist"
            print(f"✅ Saved collection to {tmpdir}")
            
            # Load it back
            loaded = ToolCollection.from_local(tmpdir)
            assert len(loaded) > 0, "Should load tools"
            print(f"✅ Loaded {len(loaded)} tools from saved collection")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hub_integration_availability():
    """Test Hub integration availability."""
    print("\n=== Test 5: Hub Integration Availability ===\n")
    
    try:
        # Check if Hub integration is available
        from core.registry.tool_collection import HUGGINGFACE_HUB_AVAILABLE
        
        if HUGGINGFACE_HUB_AVAILABLE:
            print("✅ HuggingFace Hub integration available")
            print("   Install with: pip install huggingface_hub")
        else:
            print("⚠️  HuggingFace Hub integration not available")
            print("   Install with: pip install huggingface_hub")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_mcp_integration_availability():
    """Test MCP integration availability."""
    print("\n=== Test 6: MCP Integration Availability ===\n")
    
    try:
        # Check if MCP integration is available
        from core.registry.tool_collection import MCP_AVAILABLE
        
        if MCP_AVAILABLE:
            print("✅ MCP integration available")
            print("   Install with: pip install mcp")
        else:
            print("⚠️  MCP integration not available")
            print("   Install with: pip install mcp")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Tool Collection Tests")
    print("=" * 60)
    
    tests = [
        test_local_collection,
        test_collection_to_skill_definitions,
        test_registry_integration,
        test_save_and_load,
        test_hub_integration_availability,
        test_mcp_integration_availability,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Tool Collections working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
