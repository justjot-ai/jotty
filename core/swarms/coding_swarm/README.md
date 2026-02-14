# Coding Swarm

Full-stack code generation, review, and testing with specialized developer agents.

## 🎯 Purpose

Generates production-quality code with:
- Multi-file projects (backend, frontend, database)
- Automated testing and validation
- Code review and quality checks
- Support for 7+ programming languages
- Multiple architectural styles

## ✨ Features

- **Full-stack capable**: Database → Backend → Frontend → Tests
- **Language support**: Python, JavaScript, TypeScript, Go, Rust, Java, C#
- **Code styles**: Minimal, Standard, Enterprise, Functional, OOP
- **Team-based**: Specialized agents (Architect, Backend, Frontend, QA, etc.)
- **Validation**: Terminal-based code execution and testing
- **Research**: Web search for best practices and libraries

## 🚀 Quick Start

```python
from Jotty.core.swarms.coding_swarm import CodingSwarm
from Jotty.core.swarms.coding_swarm.types import CodingConfig, CodeLanguage

# Simple Python function
swarm = CodingSwarm()
result = await swarm.execute(
    "Create a function to check if a number is prime"
)

print(result.code.files)  # Generated code
print(result.code.tests)  # Test files
```

## 📋 Configuration

### Basic Configuration

```python
from Jotty.core.swarms.coding_swarm.types import (
    CodingConfig,
    CodeLanguage,
    CodeStyle
)

config = CodingConfig(
    language=CodeLanguage.PYTHON,
    style=CodeStyle.STANDARD,
    include_tests=True,
    include_docs=True,
    include_types=True,
)

swarm = CodingSwarm(config)
```

### Full-Stack Configuration

```python
config = CodingConfig(
    # Tech stack
    language=CodeLanguage.PYTHON,
    backend_framework="fastapi",
    frontend_framework="react",
    db_type="sqlite",

    # Team & scope
    team="fullstack",        # fullstack/datascience/frontend
    scope="full_stack",      # single_tier/full_stack

    # Quality
    include_tests=True,
    include_docs=True,
    enable_workspace=True,   # Run tests
    enable_research=True,    # Search best practices
)
```

## 🎓 Language & Style Options

### Languages

```python
CodeLanguage.PYTHON      # Python 3.11+
CodeLanguage.JAVASCRIPT  # JavaScript ES6+
CodeLanguage.TYPESCRIPT  # TypeScript 5+
CodeLanguage.GO          # Go 1.20+
CodeLanguage.RUST        # Rust latest
CodeLanguage.JAVA        # Java 17+
CodeLanguage.CSHARP      # C# .NET 7+
```

### Code Styles

```python
CodeStyle.MINIMAL     # Compact, no comments
CodeStyle.STANDARD    # Balanced, clear (default)
CodeStyle.ENTERPRISE  # Verbose, fully documented
CodeStyle.FUNCTIONAL  # Functional programming paradigm
CodeStyle.OOP         # Object-oriented design
```

## 💼 Team Presets

### Frontend Team
```python
config = CodingConfig(team="frontend")
# Agents: UI Designer, Component Builder, Stylist
# Focus: React/Vue components, styling, responsiveness
```

### Full-Stack Team
```python
config = CodingConfig(team="fullstack")
# Agents: System Designer, DB Architect, Backend Dev, Frontend Dev, QA
# Focus: Complete applications with database
```

### Data Science Team
```python
config = CodingConfig(team="datascience")
# Agents: Data Engineer, ML Engineer, Visualizer, Statistician
# Focus: Data pipelines, ML models, analysis
```

## 📝 Usage Examples

### 1. REST API
```python
swarm = CodingSwarm(CodingConfig(
    language=CodeLanguage.PYTHON,
    backend_framework="fastapi",
    include_tests=True,
))

result = await swarm.execute(
    "Create a REST API for user management with CRUD operations"
)
```

### 2. React Component
```python
swarm = CodingSwarm(CodingConfig(
    language=CodeLanguage.TYPESCRIPT,
    team="frontend",
    frontend_framework="react",
))

result = await swarm.execute(
    "Create a reusable data table component with sorting and filtering"
)
```

### 3. Data Pipeline
```python
swarm = CodingSwarm(CodingConfig(
    language=CodeLanguage.PYTHON,
    team="datascience",
    include_tests=True,
))

result = await swarm.execute(
    "Build a pipeline to process CSV files and generate summary statistics"
)
```

### 4. Full-Stack App
```python
swarm = CodingSwarm(CodingConfig(
    team="fullstack",
    backend_framework="fastapi",
    frontend_framework="react",
    db_type="sqlite",
    enable_workspace=True,  # Test execution
))

result = await swarm.execute(
    "Create a todo list app with user authentication"
)
```

## 🔧 Edit Mode (Code Modification)

```python
from Jotty.core.swarms.coding_swarm.types import EditMode

config = CodingConfig(
    mode=EditMode.EDIT,
    target_files=["app.py", "utils.py"],
    codebase_path="/path/to/project",
    preserve_style=True,
)

swarm = CodingSwarm(config)
result = await swarm.execute("Add logging to all API endpoints")
```

### Edit Modes

```python
EditMode.GENERATE   # Generate new code (default)
EditMode.EDIT       # Modify existing files
EditMode.REFACTOR   # Restructure without changing behavior
EditMode.EXTEND     # Add features to existing code
```

## 🧪 Testing & Validation

### Automatic Testing
```python
config = CodingConfig(
    include_tests=True,
    enable_workspace=True,  # Actually run tests
    max_fix_attempts=3,     # Auto-fix if tests fail
)
```

### Test-Driven Development
```python
config = CodingConfig(
    test_driven=True,        # Generate tests first
    max_edit_iterations=5,   # Refine until tests pass
    test_command="pytest",   # Custom test command
)
```

## 📊 Output Structure

```python
result.code.files          # Dict[filename, content]
result.code.main_file      # Entry point filename
result.code.tests          # Dict[test_file, content]
result.code.dependencies   # List of required packages
result.code.docs           # Documentation/README
result.code.architecture   # Architecture description

result.loc                 # Lines of code
result.quality_score       # 0-1 quality rating
```

## 🏗️ Architecture

### Phase-Based Execution

**Phase 1: Research** (if enabled)
- Web search for best practices
- Library documentation lookup
- Recent code examples

**Phase 2: Design**
- Architect plans system
- Researcher validates approach
- Iterate until consensus

**Phase 3: Implementation**
- Backend developers write API
- Frontend developers build UI
- Database architect creates schema

**Phase 4: Testing**
- QA writes comprehensive tests
- Workspace executor runs tests
- Auto-fix failures (up to max_fix_attempts)

**Phase 5: Review**
- Code reviewer checks quality
- Arbitrator resolves conflicts
- Final validation

**Phase 6: Delivery**
- Package code and tests
- Generate documentation
- Return CodeOutput

### Specialized Agents

**Planning:**
- Architect
- Researcher
- Scope Classifier

**Development:**
- Backend Developer
- Frontend Developer
- Database Architect

**Quality:**
- QA Engineer
- Code Reviewer
- Arbitrator

## 💡 Tips & Best Practices

**For Best Results:**
1. Be specific: "FastAPI user auth with JWT" vs "user login"
2. Mention tech stack: "React TypeScript" vs just "frontend"
3. Enable research for unfamiliar libraries
4. Use enable_workspace to catch bugs early

**Performance:**
- `skip_team_planning=True` for trivial tasks (faster)
- `skip_team_review=True` to skip review phase
- Use CodeStyle.MINIMAL for quick prototypes

**Quality:**
- Always enable `include_tests` for production code
- Use `test_driven=True` for TDD workflow
- Enable `enable_research` for new frameworks

## 🎯 Common Patterns

### Microservice
```python
result = await swarm.execute(
    "Create a FastAPI microservice for email notifications "
    "with retry logic and dead letter queue"
)
```

### CLI Tool
```python
config = CodingConfig(
    language=CodeLanguage.PYTHON,
    style=CodeStyle.MINIMAL,
)
result = await swarm.execute(
    "Create a CLI tool to convert CSV to JSON with --verbose flag"
)
```

### Database Schema
```python
config = CodingConfig(db_type="postgresql")
result = await swarm.execute(
    "Design a database schema for an e-commerce platform"
)
```

## 🐛 Troubleshooting

**Problem**: Code doesn't run
- **Solution**: Enable `enable_workspace=True` for auto-testing

**Problem**: Too many files generated
- **Solution**: Add `max_file_size` limit or be more specific

**Problem**: Wrong tech stack
- **Solution**: Explicitly set `backend_framework`, `frontend_framework`

**Problem**: Tests fail
- **Solution**: Increase `max_fix_attempts` or disable `enable_workspace`

## 📚 Related

- **TestingSwarm**: Dedicated testing/QA
- **ReviewSwarm**: Code review only
- **ResearchSwarm**: Technical research

## 📄 License

Part of Jotty AI Framework - See main LICENSE file
