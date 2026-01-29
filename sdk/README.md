# Jotty Multi-Language SDK

Automatically generated SDKs for Jotty API in multiple programming languages.

## 🎯 Overview

This directory contains SDKs (Software Development Kits) for interacting with the Jotty API from different programming languages. All SDKs are **automatically generated** from the OpenAPI specification, ensuring consistency and eliminating manual maintenance.

## 📦 Available SDKs

| Language | Package Name | Status | Location |
|----------|-------------|--------|----------|
| TypeScript (Node.js) | `@jotty/sdk-node` | ✅ | `generated/typescript-node/` |
| TypeScript (Browser) | `@jotty/sdk-browser` | ✅ | `generated/typescript-fetch/` |
| Python | `jotty-sdk` | ✅ | `generated/python/` |
| Go | `github.com/jotty/jotty-sdk-go` | ✅ | `generated/go/` |
| Java | `com.jotty.sdk` | ✅ | `generated/java/` |
| Ruby | `jotty-sdk` | ✅ | `generated/ruby/` |
| PHP | `jotty/sdk` | ✅ | `generated/php/` |
| Swift | `JottySDK` | ✅ | `generated/swift/` |
| Kotlin | `com.jotty.sdk` | ✅ | `generated/kotlin/` |
| Rust | `jotty-sdk` | ✅ | `generated/rust/` |
| C# | `Jotty.SDK` | ✅ | `generated/csharp/` |
| Dart | `jotty_sdk` | ✅ | `generated/dart/` |

## 🚀 Quick Start

### Generate OpenAPI Specification

```bash
# Generate OpenAPI spec from Flask server
python sdk/openapi_generator.py

# Output: sdk/openapi.json
```

### Generate All SDKs

```bash
# Install OpenAPI Generator CLI (if not installed)
npm install -g @openapitools/openapi-generator-cli

# Generate all SDKs
python sdk/generate_sdks.py

# Or generate specific languages
python sdk/generate_sdks.py --languages typescript-node python go
```

### Using Docker (Alternative)

If you prefer Docker:

```bash
# Generate TypeScript SDK
docker run --rm \
  -v ${PWD}/sdk:/local \
  openapitools/openapi-generator-cli generate \
  -i /local/openapi.json \
  -g typescript-node \
  -o /local/generated/typescript-node
```

## 📖 Usage Examples

### TypeScript (Node.js)

```typescript
import { Configuration, ChatApi } from '@jotty/sdk-node';

const config = new Configuration({
  basePath: 'http://localhost:8080',
  accessToken: 'your-api-key'
});

const chatApi = new ChatApi(config);

// Execute chat
const result = await chatApi.chatExecute({
  message: 'Hello, how can you help?',
  history: []
});

console.log(result.finalOutput);
```

### Python

```python
from jotty_sdk import Configuration, ChatApi

config = Configuration(
    host="http://localhost:8080",
    access_token="your-api-key"
)

chat_api = ChatApi(config)

# Execute chat
result = chat_api.chat_execute(
    body={
        "message": "Hello, how can you help?",
        "history": []
    }
)

print(result.final_output)
```

### Go

```go
package main

import (
    "fmt"
    "github.com/jotty/jotty-sdk-go/jotty"
)

func main() {
    cfg := jotty.NewConfiguration()
    cfg.BasePath = "http://localhost:8080"
    cfg.DefaultHeader["Authorization"] = "Bearer your-api-key"
    
    client := jotty.NewAPIClient(cfg)
    chatApi := client.ChatApi
    
    result, _, err := chatApi.ChatExecute(context.Background(), jotty.ChatExecuteRequest{
        Message: "Hello, how can you help?",
    })
    
    if err != nil {
        panic(err)
    }
    
    fmt.Println(result.FinalOutput)
}
```

## 🔄 Automatic Sync Workflow

### Local Development

1. **Update API** - Make changes to Flask server routes in `core/server/http_server.py`
2. **Update OpenAPI Spec** - Modify `sdk/openapi_generator.py` to reflect changes
3. **Regenerate Spec** - Run `python sdk/openapi_generator.py`
4. **Generate SDKs** - Run `python sdk/generate_sdks.py`
5. **Test SDKs** - Use generated SDKs in your projects

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/generate-sdks.yml`) automatically:

1. Detects changes to API routes or OpenAPI spec
2. Regenerates OpenAPI specification
3. Generates all SDKs
4. Publishes SDKs to package registries (npm, PyPI, etc.)
5. Creates pull request with updated SDKs

## 🛠️ Maintenance

### Adding a New Endpoint

1. Add route to Flask server (`core/server/http_server.py`)
2. Update OpenAPI spec in `sdk/openapi_generator.py`:
   - Add path to `paths` section
   - Add request/response schemas to `components.schemas`
3. Regenerate: `python sdk/openapi_generator.py`
4. Regenerate SDKs: `python sdk/generate_sdks.py`

### Adding a New Language

1. Add configuration to `SDK_CONFIGS` in `sdk/generate_sdks.py`
2. Run: `python sdk/generate_sdks.py --languages <new-language>`
3. Test the generated SDK
4. Update this README

### Customizing Generated Code

OpenAPI Generator supports customization via:

- **Templates**: Override default templates in `sdk/templates/<language>/`
- **Configuration**: Modify `additional_properties` in `SDK_CONFIGS`
- **Post-processing**: Add scripts in `sdk/scripts/post-process/<language>.sh`

## 📝 OpenAPI Specification

The OpenAPI spec (`sdk/openapi.json`) is the single source of truth for all SDKs. It defines:

- **Endpoints**: All API routes and methods
- **Schemas**: Request/response data structures
- **Authentication**: Security schemes
- **Examples**: Sample requests/responses

View the spec:
```bash
cat sdk/openapi.json | jq
```

Validate the spec:
```bash
npx @apidevtools/swagger-cli validate sdk/openapi.json
```

## 🧪 Testing

Each generated SDK includes:

- **Type definitions** - Strongly typed interfaces
- **Client classes** - Pre-configured API clients
- **Examples** - Usage examples in `examples/` directory
- **Tests** - Unit tests (if generator supports)

Run SDK tests:
```bash
# TypeScript
cd generated/typescript-node && npm test

# Python
cd generated/python && pytest

# Go
cd generated/go && go test ./...
```

## 📚 Documentation

- **API Reference**: See `docs/API_REFERENCE.md`
- **OpenAPI Spec**: `sdk/openapi.json`
- **Generator Docs**: https://openapi-generator.tech/docs/generators/

## 🤝 Contributing

When contributing API changes:

1. Update the OpenAPI spec first
2. Regenerate SDKs locally
3. Test with at least one language
4. Submit PR with both API changes and updated SDKs

## 🔗 Related Files

- `sdk/openapi_generator.py` - OpenAPI spec generator
- `sdk/generate_sdks.py` - SDK generation script
- `.github/workflows/generate-sdks.yml` - CI/CD automation
- `core/server/http_server.py` - Flask server (source of truth)

## ❓ FAQ

**Q: Do I need to manually update SDKs when the API changes?**  
A: No! The CI/CD pipeline automatically regenerates SDKs when the API changes.

**Q: Can I customize the generated code?**  
A: Yes, use templates or post-processing scripts. See "Customizing Generated Code" above.

**Q: Which language SDK should I use?**  
A: Use the SDK for your project's language. All SDKs have the same functionality.

**Q: How do I publish SDKs to package registries?**  
A: The CI/CD pipeline handles publishing. See `.github/workflows/publish-sdks.yml`.

**Q: Can I generate SDKs for a custom language?**  
A: Yes! OpenAPI Generator supports 40+ languages. Add configuration to `generate_sdks.py`.
