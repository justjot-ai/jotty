# Notion Authentication Guide

## Overview

Notion uses **API key authentication** (not traditional login). Jotty's Notion skills authenticate using a Notion Integration Token (API key) that you create in your Notion workspace.

## How It Works

### Authentication Method
- **Type**: Bearer Token (API Key)
- **No OAuth/Login Flow**: Notion uses integration tokens, not user login
- **Token Format**: `secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Authentication Flow

```
1. User creates Notion Integration in Notion workspace
2. User gets Integration Token (API key)
3. User configures token in Jotty (env var or config file)
4. Jotty uses token in Authorization header for all API requests
```

## Step-by-Step Setup

### 1. Create Notion Integration

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Click **"+ New integration"**
3. Fill in:
   - **Name**: e.g., "Jotty Integration"
   - **Logo**: (optional)
   - **Associated workspace**: Select your workspace
4. Click **"Submit"**
5. Copy the **"Internal Integration Token"** (starts with `secret_`)

### 2. Configure Integration Permissions

1. In your Notion workspace, open any page you want Jotty to access
2. Click **"..."** (three dots) → **"Connections"**
3. Find your integration and **connect it** to the page
4. Repeat for any pages/databases you want Jotty to access

**Important**: Notion integrations can only access pages/databases they're connected to!

### 3. Configure API Key in Jotty

Jotty checks for the API key in this order:

#### Option 1: Environment Variable (Recommended)
```bash
export NOTION_API_KEY="secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Or add to your shell config (`~/.bashrc`, `~/.zshrc`):
```bash
echo 'export NOTION_API_KEY="secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

#### Option 2: Config File
```bash
mkdir -p ~/.config/notion
echo "secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > ~/.config/notion/api_key
chmod 600 ~/.config/notion/api_key  # Secure the file
```

### 4. Verify Configuration

Test that authentication works:

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

notion_skill = registry.get_skill('notion')
result = notion_skill.tools['search_pages_tool']({
    'query': 'test',
    'page_size': 1
})

if result.get('success'):
    print("✅ Notion authentication working!")
else:
    print(f"❌ Error: {result.get('error')}")
```

## How Jotty Uses Authentication

### Code Flow

```python
# In skills/notion/tools.py

class NotionClient:
    def _get_api_key(self) -> str:
        # 1. Check environment variable
        api_key = os.environ.get('NOTION_API_KEY')
        if api_key:
            return api_key
        
        # 2. Check config file
        config_path = os.path.expanduser('~/.config/notion/api_key')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return f.read().strip()
        
        # 3. Raise error if not found
        raise ValueError("Notion API key not found...")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_key()}",  # Bearer token
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
```

### API Request Example

```python
# All Notion API requests include:
headers = {
    "Authorization": "Bearer secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://api.notion.com/v1/pages",
    headers=headers,
    json=data
)
```

## Composite Skills That Use Notion

These composite skills require Notion authentication:

1. **research-to-notion** - Documents research in Notion
2. **meeting-intelligence-pipeline** - Creates meeting materials in Notion
3. **notion-knowledge-pipeline** - Manages knowledge in Notion

### Individual Notion Skills

- `notion` - Core Notion API operations
- `notion-knowledge-capture` - Captures knowledge to Notion
- `notion-meeting-intelligence` - Prepares meeting materials
- `notion-research-documentation` - Documents research
- `notion-spec-to-implementation` - Creates implementation plans

## Troubleshooting

### Error: "Notion API key not found"

**Solution**: Set the API key using one of the methods above.

```bash
# Check if env var is set
echo $NOTION_API_KEY

# Check if config file exists
cat ~/.config/notion/api_key
```

### Error: "Notion API error (401): Unauthorized"

**Causes**:
1. Invalid API key
2. API key not connected to the page/database
3. Integration not connected to workspace

**Solutions**:
1. Verify API key is correct (starts with `secret_`)
2. Connect integration to the page/database in Notion
3. Check integration is enabled in workspace settings

### Error: "Notion API error (404): Not found"

**Cause**: Page/database ID is incorrect or integration doesn't have access

**Solution**: 
1. Verify page/database ID is correct
2. Connect integration to the page/database in Notion

### Error: "Notion API error (403): Forbidden"

**Cause**: Integration doesn't have permission to perform the action

**Solution**: 
1. Check integration capabilities in Notion settings
2. Ensure integration is connected to the page/database

## Security Best Practices

1. **Never commit API keys to git**
   ```bash
   # Add to .gitignore
   echo ".config/notion/api_key" >> .gitignore
   ```

2. **Use environment variables in production**
   ```bash
   # In production, use environment variables
   export NOTION_API_KEY="secret_..."
   ```

3. **Restrict file permissions**
   ```bash
   chmod 600 ~/.config/notion/api_key
   ```

4. **Use separate integrations for different environments**
   - Development integration
   - Production integration
   - Testing integration

## Example: Using Notion Skills

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

# Get Notion skill
notion_skill = registry.get_skill('notion')

# Search pages (requires authentication)
result = await notion_skill.tools['search_pages_tool']({
    'query': 'project notes',
    'page_size': 10
})

# Create page (requires authentication + parent page connection)
result = await notion_skill.tools['create_page_tool']({
    'parent_id': 'your-page-id',
    'title': 'New Page',
    'content': [
        {
            'object': 'block',
            'type': 'paragraph',
            'paragraph': {
                'rich_text': [{'type': 'text', 'text': {'content': 'Hello!'}}]
            }
        }
    ]
})
```

## Summary

- **Authentication Type**: API Key (Bearer Token)
- **No Login Required**: Just configure the API key
- **Configuration**: Environment variable or config file
- **Access Control**: Connect integration to pages/databases in Notion
- **Security**: Keep API keys secret, never commit to git

---

**Need Help?**
- [Notion API Documentation](https://developers.notion.com/)
- [Notion Integrations Guide](https://developers.notion.com/docs/getting-started)
