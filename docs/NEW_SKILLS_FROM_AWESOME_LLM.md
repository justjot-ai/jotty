# New Skills from Awesome LLM Skills Repository

We've integrated several high-value skills from the [awesome-llm-skills](https://github.com/Prat011/awesome-llm-skills) repository into Jotty.

## ✅ Completed Skills

### 1. **Changelog Generator** ✅
**Location:** `skills/changelog-generator/`

**What it does:**
- Transforms technical git commits into user-friendly changelogs
- Categorizes changes (features, improvements, fixes, security, breaking)
- Filters out internal commits (refactoring, tests, etc.)
- Generates professional release notes

**Usage:**
```python
result = await generate_changelog_tool({
    'since': 'last-release',
    'version': '2.5.0',
    'output_file': 'CHANGELOG.md'
})
```

**Dependencies:** `gitpython`

---

### 2. **Lead Research Assistant** ✅
**Location:** `skills/lead-research-assistant/`

**What it does:**
- Identifies high-quality leads for your product/service
- Analyzes your product and ideal customer profile
- Searches for target companies matching criteria
- Provides fit scores, decision makers, and outreach strategies
- Exports to markdown, CSV, or JSON

**Usage:**
```python
result = await research_leads_tool({
    'product_description': 'AI-powered code review tool',
    'industry': 'Technology',
    'company_size': '50-500 employees',
    'max_leads': 10,
    'output_format': 'csv'
})
```

**Dependencies:** `web-search`, `claude-cli-llm`

---

### 3. **Domain Name Brainstormer** ✅
**Location:** `skills/domain-name-brainstormer/`

**What it does:**
- Generates creative domain name suggestions
- Checks availability across multiple TLDs (.com, .io, .dev, .ai)
- Provides brandability scores and recommendations
- Suggests why each name works

**Usage:**
```python
result = await brainstorm_domains_tool({
    'project_description': 'AI-powered code review tool',
    'keywords': ['code', 'review'],
    'preferred_tlds': ['.com', '.io', '.dev'],
    'max_suggestions': 15
})
```

**Dependencies:** `claude-cli-llm`

---

## 📋 Planned Skills (From Repository)

### 4. **Meeting Insights Analyzer** (Planned)
- Analyzes meeting transcripts for communication patterns
- Identifies conflict avoidance, filler words, speaking ratios
- Provides actionable feedback for improvement

### 5. **Competitive Ads Extractor** (Planned)
- Extracts and analyzes competitors' ads
- Understands messaging and creative approaches
- Helps with competitive intelligence

### 6. **Content Research Writer** (Planned)
- Assists in writing high-quality content
- Conducts research and adds citations
- Improves hooks and provides section-by-section feedback

### 7. **File Organizer** (Planned)
- Intelligently organizes files and folders
- Finds duplicates and suggests better structures

### 8. **Invoice Organizer** (Planned)
- Organizes invoices and receipts for tax preparation
- Extracts information and renames consistently

---

## 🚀 How to Use

### Load Skills
Skills are automatically discovered from `~/jotty/skills/` directory.

### Use in Code
```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

# Get skill
skill = registry.get_skill('changelog-generator')
tool = skill.tools['generate_changelog_tool']

# Use tool
result = await tool({
    'since': 'last-release',
    'version': '2.5.0'
})
```

### Install Dependencies
```bash
# For changelog-generator
pip install gitpython

# Skills will auto-install dependencies via SkillDependencyManager
```

---

## 📊 Skill Statistics

- **Total Skills Created:** 3
- **Total Tools:** 3
- **Status:** All loaded and ready to use ✅

---

## 🔗 Source Repository

All skills adapted from: https://github.com/Prat011/awesome-llm-skills

Original skills are Claude Skills format (SKILL.md only), converted to Jotty format (SKILL.md + tools.py).

---

## 💡 Next Steps

1. **Test the skills** with real use cases
2. **Add more skills** from the repository
3. **Enhance existing skills** with additional features
4. **Create composite skills** combining multiple skills

---

## 📝 Notes

- Skills follow Jotty's skill structure (SKILL.md + tools.py)
- All skills integrate with existing Jotty infrastructure
- Skills use Jotty's registry system for dependencies
- AI-powered features use Claude CLI via `claude-cli-llm` skill
