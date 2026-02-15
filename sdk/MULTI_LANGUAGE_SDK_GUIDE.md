# Multi-Language SDK Guide for Jotty

## 🎯 Problem Statement

**Challenge**: Maintaining SDKs for multiple languages manually is:
- ❌ Time-consuming (duplicate code across languages)
- ❌ Error-prone (inconsistencies between languages)
- ❌ Hard to keep in sync (API changes require manual updates everywhere)
- ❌ Not scalable (adding new languages multiplies maintenance burden)

**Solution**: Automatically generate SDKs from a single OpenAPI specification.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Flask Server (Source of Truth)             │
│         core/server/http_server.py                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         OpenAPI Specification Generator                  │
│         sdk/openapi_generator.py                        │
│                                                         │
│  • Analyzes Flask routes                               │
│  • Defines request/response schemas                    │
│  • Generates openapi.json                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              OpenAPI Specification                      │
│              sdk/openapi.json                           │
│                                                         │
│  Single source of truth for API contract                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Multi-Language SDK Generator                    │
│         sdk/generate_sdks.py                            │
│                                                         │
│  Uses OpenAPI Generator to create:                     │
│  • TypeScript (Node.js & Browser)                      │
│  • Python                                              │
│  • Go                                                   │
│  • Java                                                 │
│  • Ruby, PHP, Swift, Kotlin, Rust, C#, Dart            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Generated SDKs                             │
│              sdk/generated/                             │
│                                                         │
│  • typescript-node/  • python/                         │
│  • typescript-fetch/ • go/                             │
│  • java/             • ruby/                            │
│  • ... (12+ languages)                                 │
└─────────────────────────────────────────────────────────┘
```

## 🔄 Workflow

### 1. Development Workflow

```bash
# 1. Make API changes in Flask server
vim core/server/http_server.py

# 2. Update OpenAPI spec generator
vim sdk/openapi_generator.py

# 3. Regenerate OpenAPI spec
python sdk/openapi_generator.py

# 4. Generate all SDKs
python sdk/generate_sdks.py

# 5. Test with your preferred language
cd sdk/generated/typescript-node && npm test
```

### 2. CI/CD Workflow

The GitHub Actions workflow (`.github/workflows/generate-sdks.yml`) automatically:

1. **Detects Changes**: Monitors `core/server/` and `sdk/` directories
2. **Generates Spec**: Runs `openapi_generator.py`
3. **Validates Spec**: Ensures OpenAPI spec is valid
4. **Generates SDKs**: Creates SDKs for all languages
5. **Commits Changes**: Auto-commits generated SDKs (on main branch)
6. **Publishes**: Optionally publishes to package registries

### 3. Adding a New Endpoint

**Step 1**: Add route to Flask server
```python
@self.app.route('/api/new-endpoint', methods=['POST'])
def new_endpoint():
    # Implementation
    pass
```

**Step 2**: Update OpenAPI spec generator
```python
# In sdk/openapi_generator.py, add to paths:
"/api/new-endpoint": {
    "post": {
        "tags": ["NewFeature"],
        "summary": "New endpoint",
        "requestBody": {...},
        "responses": {...}
    }
}
```

**Step 3**: Regenerate
```bash
python sdk/openapi_generator.py
python sdk/generate_sdks.py
```

All SDKs are now updated! 🎉

## 📋 Best Practices

### 1. Keep OpenAPI Spec Up-to-Date

- ✅ Update spec immediately when adding/changing endpoints
- ✅ Include comprehensive descriptions and examples
- ✅ Define all request/response schemas explicitly
- ❌ Don't skip updating the spec (breaks SDK generation)

### 2. Version Management

```python
# In openapi_generator.py
spec = generate_openapi_spec(
    version="1.2.3"  # Increment on breaking changes
)
```

- **Major** (1.0.0 → 2.0.0): Breaking API changes
- **Minor** (1.0.0 → 1.1.0): New endpoints, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, no API changes

### 3. Testing Generated SDKs

```bash
# Test TypeScript SDK
cd sdk/generated/typescript-node
npm install
npm test

# Test Python SDK
cd sdk/generated/python
pip install -e .
pytest

# Test Go SDK
cd sdk/generated/go
go test ./...
```

### 4. Customization

**Option A: Templates** (for major customization)
```bash
# Copy templates
cp -r sdk/templates/typescript-node/ sdk/custom-templates/

# Modify templates
vim sdk/custom-templates/typescript-node/api.mustache

# Use custom templates
openapi-generator-cli generate \
  -i sdk/openapi.json \
  -g typescript-node \
  -t sdk/custom-templates/typescript-node \
  -o sdk/generated/typescript-node
```

**Option B: Post-processing** (for minor tweaks)
```bash
# Add post-processing script
sdk/scripts/post-process/typescript-node.sh

# Script runs after generation
# Can modify generated files, add custom code, etc.
```

## 🚀 Advanced Features

### 1. Streaming Support

SSE (Server-Sent Events) endpoints are handled specially:

```typescript
// TypeScript SDK automatically handles SSE
const stream = await chatApi.chatStream({...});
for await (const event of stream) {
  console.log(event);
}
```

### 2. Authentication

All SDKs support Bearer token authentication:

```typescript
// TypeScript
const config = new Configuration({
  accessToken: 'your-token'
});

// Python
config = Configuration(access_token='your-token')

// Go
cfg.DefaultHeader["Authorization"] = "Bearer your-token"
```

### 3. Error Handling

Consistent error handling across all SDKs:

```typescript
try {
  const result = await chatApi.chatExecute({...});
} catch (error) {
  if (error instanceof ApiError) {
    console.error(`API Error: ${error.status} - ${error.message}`);
  }
}
```

## 📊 Comparison: Manual vs Automated

| Aspect | Manual SDKs | Automated SDKs |
|--------|------------|----------------|
| **Setup Time** | Days/weeks per language | Minutes (one-time) |
| **Maintenance** | High (update each language) | Low (update spec once) |
| **Consistency** | Low (human error) | High (generated) |
| **Coverage** | Limited (few languages) | Extensive (40+ languages) |
| **Type Safety** | Manual (error-prone) | Automatic (from spec) |
| **Documentation** | Manual | Auto-generated |
| **Testing** | Per-language | Shared spec validation |

## 🔧 Troubleshooting

### Issue: OpenAPI Generator not found

```bash
# Solution 1: Install via npm
npm install -g @openapitools/openapi-generator-cli

# Solution 2: Use Docker
docker pull openapitools/openapi-generator-cli

# Solution 3: Use Homebrew (macOS)
brew install openapi-generator
```

### Issue: Generated SDK has errors

1. **Validate OpenAPI spec**:
   ```bash
   npx @apidevtools/swagger-cli validate sdk/openapi.json
   ```

2. **Check generator version**:
   ```bash
   openapi-generator-cli version
   # Update if outdated
   ```

3. **Review spec for issues**:
   - Missing required fields
   - Invalid schema references
   - Inconsistent types

### Issue: SDK doesn't match API behavior

- Ensure OpenAPI spec accurately reflects Flask routes
- Check that request/response schemas match actual data
- Verify authentication is correctly configured

## 📚 Resources

- **OpenAPI Generator**: https://openapi-generator.tech/
- **OpenAPI Spec**: https://swagger.io/specification/
- **Generator Templates**: https://github.com/OpenAPITools/openapi-generator/tree/master
- **Jotty API Docs**: `docs/API_REFERENCE.md`

## ✅ Checklist for New API Changes

- [ ] Update Flask route in `core/server/http_server.py`
- [ ] Update OpenAPI spec in `sdk/openapi_generator.py`
- [ ] Regenerate spec: `python sdk/openapi_generator.py`
- [ ] Validate spec: `swagger-cli validate sdk/openapi.json`
- [ ] Generate SDKs: `python sdk/generate_sdks.py`
- [ ] Test at least one SDK (TypeScript/Python)
- [ ] Update API documentation
- [ ] Commit changes (CI will auto-commit generated SDKs)

## 🎉 Benefits

✅ **Single Source of Truth**: OpenAPI spec defines API contract
✅ **Automatic Sync**: CI/CD keeps SDKs in sync with API
✅ **Multi-Language**: Support 12+ languages with minimal effort
✅ **Type Safety**: Generated types match API exactly
✅ **Consistency**: Same API surface across all languages
✅ **Maintainability**: Update spec once, all SDKs update
✅ **Scalability**: Add new languages easily

---

**Result**: No more manual SDK maintenance! 🚀
