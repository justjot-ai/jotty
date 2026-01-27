"""
Tool Collection Examples

Demonstrates how to use ToolCollection to load tools from:
- HuggingFace Hub collections
- MCP servers
- Local collections
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.registry import ToolCollection, get_skills_registry


def example_hub_collection():
    """Example: Load tools from HuggingFace Hub collection."""
    print("=== Example 1: Load from HuggingFace Hub ===\n")
    
    try:
        # Load collection from Hub
        # Note: Requires trust_remote_code=True and huggingface_hub installed
        collection = ToolCollection.from_hub(
            collection_slug="huggingface-tools/diffusion-tools",
            trust_remote_code=True  # Always inspect tools before loading!
        )
        
        print(f"Loaded collection: {collection.source}")
        print(f"Number of tools: {len(collection)}")
        print(f"Metadata: {collection.metadata}")
        
        # List tools
        tools = collection.list_tools()
        print(f"\nTools in collection:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool['name']}: {tool['description'][:50]}...")
        
        # Load into registry
        registry = get_skills_registry()
        registry.init()
        registry.load_collection(collection, collection_name="hub_diffusion_tools")
        
        print(f"\n✅ Collection loaded into registry")
        
    except ImportError as e:
        print(f"⚠️  Hub integration not available: {e}")
        print("   Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ Failed to load from Hub: {e}")


def example_mcp_collection():
    """Example: Load tools from MCP server."""
    print("\n\n=== Example 2: Load from MCP Server ===\n")
    
    try:
        # Note: Requires mcp package and trust_remote_code=True
        from mcp import StdioServerParameters
        
        server_params = StdioServerParameters(
            command="uv",
            args=["--quiet", "pubmedmcp@0.1.3"],
            env={"UV_PYTHON": "3.12", **os.environ}
        )
        
        # Load collection from MCP
        with ToolCollection.from_mcp(server_params, trust_remote_code=True) as collection:
            print(f"Loaded collection: {collection.source}")
            print(f"Number of tools: {len(collection)}")
            
            # List tools
            tools = collection.list_tools()
            print(f"\nTools from MCP:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description'][:50]}...")
            
            # Load into registry
            registry = get_skills_registry()
            registry.init()
            registry.load_collection(collection, collection_name="mcp_pubmed")
            
            print(f"\n✅ Collection loaded into registry")
            
    except ImportError as e:
        print(f"⚠️  MCP integration not available: {e}")
        print("   Install with: pip install mcp")
    except Exception as e:
        print(f"❌ Failed to load from MCP: {e}")


def example_local_collection():
    """Example: Load tools from local collection."""
    print("\n\n=== Example 3: Load from Local Collection ===\n")
    
    try:
        # Load from local directory (uses existing skills)
        collection = ToolCollection.from_local("./skills")
        
        print(f"Loaded collection: {collection.source}")
        print(f"Number of tools: {len(collection)}")
        print(f"Metadata: {collection.metadata}")
        
        # List tools
        tools = collection.list_tools()
        print(f"\nTools in local collection:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool['name']}: {tool['description'][:50]}...")
        
        # Load into registry
        registry = get_skills_registry()
        registry.init()
        registry.load_collection(collection, collection_name="local_skills")
        
        print(f"\n✅ Collection loaded into registry")
        
    except Exception as e:
        print(f"❌ Failed to load local collection: {e}")


def example_collection_usage():
    """Example: Using tools from collection."""
    print("\n\n=== Example 4: Using Tools from Collection ===\n")
    
    try:
        registry = get_skills_registry()
        registry.init()
        
        # List all collections
        collections = registry.list_collections()
        print(f"Loaded collections: {len(collections)}")
        for coll in collections:
            print(f"  - {coll['name']}: {coll['source']} ({coll['tool_count']} tools)")
        
        # Get tools from registry
        all_tools = registry.get_registered_tools()
        print(f"\nTotal tools in registry: {len(all_tools)}")
        
        # List some tools
        print(f"\nSample tools:")
        for tool_name in list(all_tools.keys())[:5]:
            print(f"  - {tool_name}")
        
    except Exception as e:
        print(f"❌ Failed to use collection: {e}")


def example_save_collection():
    """Example: Save collection to local."""
    print("\n\n=== Example 5: Save Collection ===\n")
    
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
        
        # Save to local
        output_path = "./test_collection"
        collection.save_to_local(output_path)
        
        print(f"✅ Saved collection to {output_path}")
        
        # Load it back
        loaded = ToolCollection.from_local(output_path)
        print(f"✅ Loaded collection: {len(loaded)} tools")
        
    except Exception as e:
        print(f"❌ Failed to save collection: {e}")


if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("Tool Collection Examples")
    print("=" * 60)
    
    # Run examples
    example_local_collection()
    example_collection_usage()
    example_save_collection()
    
    # These require external dependencies
    # example_hub_collection()
    # example_mcp_collection()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nNote: Hub and MCP examples require:")
    print("  - pip install huggingface_hub  (for Hub)")
    print("  - pip install mcp  (for MCP)")
    print("  - trust_remote_code=True  (always inspect tools!)")
