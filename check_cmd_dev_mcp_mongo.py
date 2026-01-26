#!/usr/bin/env python3
"""Check MongoDB and MCP Server status on cmd.dev"""

import os
import subprocess
import sys
import time
import json

print("🔍 Checking MongoDB and MCP Server on cmd.dev...")
print("=" * 60)
print()

# Check MongoDB
print("1️⃣  Checking MongoDB...")
print("-" * 60)

mongodb_uris = [
    ("Local", "mongodb://localhost:27017/justjot"),
    ("Remote (justjot)", "mongodb://planmyinvesting:aRpOVx2HYl6jS9LO@planmyinvesting.com:27017/justjot"),
    ("Remote (planmyinvesting)", "mongodb://planmyinvesting:aRpOVx2HYl6jS9LO@planmyinvesting.com:27017/planmyinvesting"),
]

if os.getenv("MONGODB_URI"):
    mongodb_uris.insert(0, ("Environment", os.getenv("MONGODB_URI")))

mongodb_accessible = False
working_uri = None

for name, uri in mongodb_uris:
    print(f"   Testing {name}: {uri.split('@')[-1] if '@' in uri else uri}")
    try:
        import pymongo
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client.get_database()
        collections = db.list_collection_names()
        print(f"   ✅ MongoDB accessible!")
        print(f"      Database: {db.name}")
        print(f"      Collections: {len(collections)} found")
        if collections:
            print(f"      Sample: {', '.join(collections[:5])}")
        if 'ideas' in collections:
            count = db.ideas.count_documents({})
            print(f"      Ideas count: {count}")
        mongodb_accessible = True
        working_uri = uri
        client.close()
        break
    except ImportError:
        print("   ⚠️  pymongo not installed - install with: pip install pymongo")
        break
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")

if not mongodb_accessible:
    print("   ⚠️  MongoDB not accessible with any URI")

print()

# Check MCP Server file
print("2️⃣  Checking MCP Server File...")
print("-" * 60)

mcp_server_path = "/var/www/sites/personal/stock_market/JustJot.ai/dist/mcp/server.js"
if os.path.exists(mcp_server_path):
    print(f"   ✅ MCP server file exists")
    print(f"      Path: {mcp_server_path}")
    size = os.path.getsize(mcp_server_path)
    mtime = os.path.getmtime(mcp_server_path)
    import datetime
    mod_time = datetime.datetime.fromtimestamp(mtime)
    print(f"      Size: {size / 1024:.1f} KB")
    print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if it's a valid JavaScript file
    try:
        with open(mcp_server_path, 'r') as f:
            first_line = f.readline()
            if 'require' in first_line or 'import' in first_line or 'function' in first_line:
                print(f"      ✅ Valid JavaScript file")
            else:
                print(f"      ⚠️  May not be valid JavaScript")
    except:
        pass
else:
    print(f"   ❌ MCP server file not found")
    print(f"      Expected: {mcp_server_path}")
    print(f"      💡 Compile with: cd JustJot.ai && npm run build:mcp")

print()

# Check Node.js
print("3️⃣  Checking Node.js...")
print("-" * 60)

try:
    node_version = subprocess.check_output(["node", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    node_path = subprocess.check_output(["which", "node"], text=True).strip()
    print(f"   ✅ Node.js available")
    print(f"      Version: {node_version}")
    print(f"      Path: {node_path}")
except subprocess.CalledProcessError:
    print("   ❌ Node.js not available")
except FileNotFoundError:
    print("   ❌ Node.js not found in PATH")

print()

# Check if MCP server can start
print("4️⃣  Testing MCP Server Startup...")
print("-" * 60)

if os.path.exists(mcp_server_path):
    try:
        # Prepare environment
        env = os.environ.copy()
        if working_uri:
            env['MONGODB_URI'] = working_uri
        else:
            env['MONGODB_URI'] = mongodb_uris[1][1]  # Use remote URI
        
        print(f"   Starting MCP server with MongoDB URI: {env['MONGODB_URI'].split('@')[-1] if '@' in env['MONGODB_URI'] else env['MONGODB_URI']}")
        
        # Start MCP server
        process = subprocess.Popen(
            ["node", mcp_server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=0
        )
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        request_json = json.dumps(init_request) + "\n"
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Wait for response
        time.sleep(1)
        
        # Check if process is still running
        if process.poll() is None:
            print("   ✅ MCP server started successfully")
            print("   ✅ Process is running")
            
            # Try to read response
            try:
                import select
                if select.select([process.stdout], [], [], 0.5)[0]:
                    response = process.stdout.readline()
                    if response:
                        print(f"   ✅ Received response: {response[:100]}...")
            except:
                pass
            
            # Cleanup
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
            stdout, stderr = process.communicate()
            print(f"   ❌ MCP server exited immediately")
            if stderr:
                print(f"      Error: {stderr[:300]}")
            if stdout:
                print(f"      Output: {stdout[:300]}")
                
    except Exception as e:
        print(f"   ⚠️  Could not test MCP server: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⚠️  Cannot test - MCP server file not found")

print()
print("=" * 60)
print("📊 Summary")
print("=" * 60)
print(f"MongoDB: {'✅ Accessible' if mongodb_accessible else '❌ Not accessible'}")
print(f"MCP Server File: {'✅ Exists' if os.path.exists(mcp_server_path) else '❌ Not found'}")
print(f"Node.js: {'✅ Available' if 'node_version' in locals() else '❌ Not available'}")

if mongodb_accessible and os.path.exists(mcp_server_path) and 'node_version' in locals():
    print()
    print("✅ All prerequisites met! MCP client should work on cmd.dev")
    print(f"   Use MongoDB URI: {working_uri}")
else:
    print()
    print("⚠️  Some prerequisites missing - check above for details")
