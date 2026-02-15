

# 🏗️ Base Skill Architecture

**Unified base classes for all Jotty skills to eliminate code duplication.**

## 📊 **Impact**

- **Before:** 341 skills with ~114K lines of code (~334 lines/skill)
- **After:** Reduces to ~100-150 lines/skill on average
- **Savings:** ~60-100K lines of code eliminated!
- **Benefit:** Unified LLM access, consistent error handling, easier testing

---

## 🎯 **Quick Start**

### **Option 1: Class-Based (Recommended for Complex Skills)**

```python
from skills._base import BaseLLMSkill

class SummarizerSkill(BaseLLMSkill):
    """Summarize text using LLM."""

    def execute(self, params):
        text = params["text"]
        max_length = params.get("max_length", 200)

        # Use unified LLM (auto-configured!)
        summary = self.call_lm(
            f"Summarize this in {max_length} words: {text}",
            max_tokens=500
        )

        return self.success(summary=summary)

# Create instance and use
summarizer = SummarizerSkill("summarizer")
result = summarizer({"text": "Long text here..."})
```

### **Option 2: Decorator-Based (Quick & Easy)**

```python
from skills._base import skill_tool

@skill_tool(
    name="translator",
    required_params=["text", "target_language"],
    use_llm=True
)
def translate(skill, params):
    """Translate text to target language."""
    text = params["text"]
    lang = params["target_language"]

    translation = skill.call_lm(f"Translate to {lang}: {text}")

    return {"translation": translation}

# Use directly
result = translate({"text": "Hello", "target_language": "Spanish"})
```

### **Option 3: Function-Based (Programmatic)**

```python
from skills._base import create_llm_skill

summarizer = create_llm_skill(
    name="summarizer",
    execute_fn=lambda skill, params: {
        "summary": skill.call_lm(f"Summarize: {params['text']}")
    }
)

result = summarizer({"text": "Long text..."})
```

---

## 🧱 **Base Classes**

### **BaseSkill** (Abstract Base)

Core functionality for all skills:
- ✅ Status callback handling
- ✅ Configuration management
- ✅ Consistent error handling
- ✅ Tool response formatting

```python
class MySkill(BaseSkill):
    def execute(self, params):
        # Your logic here
        return self.success(result="done")
```

### **BaseToolSkill** (For Utilities)

For skills that don't need LLM (calculator, file ops, etc.):

```python
from skills._base import BaseToolSkill

class CalculatorSkill(BaseToolSkill):
    def execute(self, params):
        expression = params["expression"]
        result = eval(expression)  # Use safe eval in production!
        return self.success(result=result)
```

### **BaseLLMSkill** (For AI-Powered Skills)

For skills that need LLM access:

```python
from skills._base import BaseLLMSkill

class CodeGeneratorSkill(BaseLLMSkill):
    def execute(self, params):
        prompt = params["prompt"]
        language = params.get("language", "python")

        # Option 1: Simple call
        code = self.call_lm(f"Generate {language} code: {prompt}")

        # Option 2: With system prompt
        code = self.generate_text(
            prompt=prompt,
            system_prompt=f"You are an expert {language} developer",
            temperature=0.3
        )

        # Option 3: Structured output
        result = self.generate_structured(
            prompt=prompt,
            schema={"code": "string", "explanation": "string"}
        )

        return self.success(code=code)
```

**Built-in LLM Methods:**
- `self.lm` - DSPy LM instance (lazy-initialized)
- `self.call_lm(prompt, **kwargs)` - Simple LLM call
- `self.generate_text(prompt, system_prompt, **kwargs)` - Text generation
- `self.generate_structured(prompt, schema, **kwargs)` - JSON output

**Supported Providers:**
- `anthropic` (Claude - default)
- `openai` (GPT-4, GPT-3.5)
- `google` (Gemini)
- `groq` (Fast inference)
- `openrouter` (Multi-provider)
- `claude-cli`, `cursor-cli`

---

## 🎨 **Decorators**

### **@skill_tool** - Turn Functions into Skills

```python
from skills._base import skill_tool

@skill_tool(
    name="my_skill",
    required_params=["input"],
    optional_params=["max_length"],
    use_llm=True,
    provider="anthropic",
    temperature=0.7
)
def my_skill_fn(skill, params):
    """My skill logic."""
    input_text = params["input"]
    max_len = params.get("max_length", 1000)

    result = skill.call_lm(f"Process: {input_text}")

    return {"output": result}

# Use it
result = my_skill_fn({"input": "test"})
```

### **@validate_params** - Parameter Validation

```python
from skills._base import validate_params, BaseLLMSkill

class MySkill(BaseLLMSkill):
    @validate_params(
        required=["text", "language"],
        optional=["max_length", "temperature"]
    )
    def execute(self, params):
        # params are guaranteed to be valid
        text = params["text"]
        language = params["language"]
        max_length = params.get("max_length", 1000)

        return self.success(result="done")
```

---

## 🔧 **Helper Functions**

### **create_tool_skill** - Quick Utility Skills

```python
from skills._base import create_tool_skill

# Simple calculator
calculator = create_tool_skill(
    name="calculator",
    execute_fn=lambda params: {
        "result": eval(params["expression"])
    }
)

result = calculator({"expression": "2 + 2"})
# {'success': True, 'result': 4}
```

### **create_llm_skill** - Quick LLM Skills

```python
from skills._base import create_llm_skill

# Simple summarizer
summarizer = create_llm_skill(
    name="summarizer",
    execute_fn=lambda skill, params: {
        "summary": skill.call_lm(f"Summarize: {params['text']}")
    },
    provider="anthropic"
)

result = summarizer({"text": "Long text..."})
# {'success': True, 'summary': '...'}
```

---

## 📝 **Migration Guide**

### **Before (Old Way - ~150 lines)**

```python
import anthropic
import os
from typing import Dict, Any

def my_skill_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """My skill that uses Anthropic."""

    # Create client (duplicated across 10+ skills!)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API key"}

    client = anthropic.Anthropic(api_key=api_key)

    # Error handling boilerplate
    try:
        if "input" not in params:
            return {"success": False, "error": "Missing input"}

        input_text = params["input"]

        # Make LLM call
        response = client.messages.create(
            model="claude-sonnet-3-5-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": input_text}]
        )

        result = response.content[0].text

        return {"success": True, "result": result}

    except anthropic.APIError as e:
        return {"success": False, "error": str(e)}
    except KeyError as e:
        return {"success": False, "error": f"Missing param: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### **After (New Way - ~20 lines!)**

```python
from skills._base import BaseLLMSkill

class MySkill(BaseLLMSkill):
    """My skill that uses unified LLM."""

    def execute(self, params):
        input_text = params["input"]  # Auto-validated

        # LLM auto-configured with unified provider!
        result = self.call_lm(input_text)

        return self.success(result=result)

my_skill_tool = MySkill("my_skill")
```

**Savings:**
- ❌ No more API client boilerplate (15-30 lines)
- ❌ No more error handling boilerplate (20-40 lines)
- ❌ No more parameter validation (10-20 lines)
- ❌ No more response formatting (5-10 lines)
- ✅ **Total: 50-100 lines saved per skill!**

---

## 🧪 **Testing**

```python
import pytest
from skills._base import BaseLLMSkill

class MockLLM:
    """Mock LLM for testing."""
    def __call__(self, prompt, **kwargs):
        return type('Response', (), {'completions': [type('C', (), {'text': 'mocked'})]})()

def test_my_skill():
    skill = MySkill("test")

    # Mock the LLM
    skill._lm = MockLLM()

    result = skill({"input": "test"})

    assert result["success"] is True
    assert "result" in result
```

---

## 🎯 **Best Practices**

1. **Use BaseLLMSkill for AI tasks** - Don't create custom Anthropic/OpenAI clients
2. **Use BaseToolSkill for utilities** - Calculator, file ops, web scraping, etc.
3. **Validate params** - Use `@validate_params` or check in `execute()`
4. **Log appropriately** - Use `self.logger` (auto-configured)
5. **Update status** - Use `self.update_status()` for long-running tasks
6. **Return proper responses** - Use `self.success()` and `self.error()`

---

## 📚 **Examples**

### **Text Summarizer**
```python
from skills._base import BaseLLMSkill

class SummarizerSkill(BaseLLMSkill):
    def execute(self, params):
        return self.success(
            summary=self.call_lm(f"Summarize: {params['text']}")
        )
```

### **Code Generator**
```python
from skills._base import BaseLLMSkill, validate_params

class CodeGenSkill(BaseLLMSkill):
    @validate_params(required=["prompt"], optional=["language"])
    def execute(self, params):
        lang = params.get("language", "python")
        result = self.generate_structured(
            prompt=params["prompt"],
            schema={"code": "string", "explanation": "string"}
        )
        return self.success(**result)
```

### **Calculator (No LLM)**
```python
from skills._base import BaseToolSkill

class CalcSkill(BaseToolSkill):
    def execute(self, params):
        result = eval(params["expression"])
        return self.success(result=result)
```

---

## 🚀 **What's Next?**

1. **Migrate existing skills** - Start with LLM-heavy skills first
2. **Remove duplicate code** - Eliminate custom API clients
3. **Standardize patterns** - Use unified base classes everywhere
4. **Add tests** - Easier to test with mocked LLM
5. **Document skills** - Add docstrings with param descriptions

**Target:** Reduce 341 skills from ~114K lines to ~35-50K lines (~60-70% reduction!)
