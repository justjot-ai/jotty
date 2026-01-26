#!/usr/bin/env python3
"""Test MCP client on cmd.dev with direct MongoDB access"""

import asyncio
import os
import sys
import json
sys.path.insert(0, '/var/www/sites/personal/stock_market/Jotty')

from core.integration.mcp_client import MCPClient

async def test_mcp_on_cmd_dev():
    print("🧪 Testing MCP Client on cmd.dev...")
    print()
    
    # Get MongoDB URI from environment or use default
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/justjot")
    clerk_secret_key = os.getenv("CLERK_SECRET_KEY", "")
    server_path = "/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js"
    
    print(f"   MongoDB URI: {mongodb_uri}")
    print(f"   MCP Server: {server_path}")
    print(f"   Server exists: {os.path.exists(server_path)}")
    print()
    
    if not os.path.exists(server_path):
        print("❌ MCP server not found!")
        print(f"   Please compile: cd JustJot.ai && npm run build:mcp")
        return
    
    # Prepare environment
    env = {}
    if mongodb_uri:
        env['MONGODB_URI'] = mongodb_uri
    if clerk_secret_key:
        env['CLERK_SECRET_KEY'] = clerk_secret_key
    
    client = MCPClient(
        server_path=server_path,
        mongodb_uri=mongodb_uri,
        env=env
    )
    
    try:
        print("🔌 Connecting to MCP server...")
        await client.connect()
        print("✅ Connected!")
        print()
        
        print("📋 Listing tools...")
        tools = await client.list_tools()
        print(f"✅ Found {len(tools)} tools")
        print(f"   Tools: {', '.join([t.get('name', 'N/A') for t in tools[:5]])}...")
        print()
        
        print("🚀 Creating test idea...")
        result = await client.call_tool("create_idea", {
            "title": "Test Idea from MCP on cmd.dev",
            "description": "Testing MCP client with direct MongoDB access",
            "tags": ["test", "mcp", "cmd.dev"]
        })
        
        print("\n📊 Result:")
        print(f"   Type: {type(result)}")
        print(f"   Value: {result}")
        
        if isinstance(result, dict):
            # Check if result has content array (MCP format)
            if 'content' in result:
                content = result['content']
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', '')
                    print(f"   Content text: {text_content[:200]}...")
                    try:
                        parsed = json.loads(text_content)
                        print(f"   Parsed: {parsed}")
                        if parsed.get('success'):
                            idea_id = parsed.get('id')
                            print(f"   ✅ Idea created successfully!")
                            print(f"   Idea ID: {idea_id}")
                            
                            # Test getting the idea
                            print("\n📖 Getting idea...")
                            get_result = await client.call_tool("get_idea", {"id": idea_id})
                            print(f"   Get result: {get_result}")
                        else:
                            print(f"   ❌ Failed: {parsed.get('error', 'Unknown error')}")
                    except json.JSONDecodeError:
                        print(f"   Raw text: {text_content}")
            elif result.get('success'):
                idea_id = result.get('id')
                print(f"   ✅ Idea created successfully!")
                print(f"   Idea ID: {idea_id}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   Raw result: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 Disconnecting...")
        try:
            await client.disconnect()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_mcp_on_cmd_dev())
