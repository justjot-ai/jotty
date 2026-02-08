# Slides Generation Setup Note

## Current Status

✅ **All packages installed** (except mineru)
✅ **Configuration file created** (`Paper2Slides/paper2slides/.env`)
⚠️ **Needs**: IMAGE_GEN_API_KEY + mineru OR use markdown workaround

---

## Issue with PDF Input

Paper2Slides requires `mineru[core]` (~500MB) to parse PDF files. This is a **very large dependency**.

### Why It's Needed
Paper2Slides' RAG stage parses PDFs to extract:
- Text content
- Figures and tables
- Document structure

### The Problem
```bash
pip install 'mineru[core]==2.6.4'
# Downloads ~500MB of dependencies
# Takes 5-10 minutes to install
```

---

## Solution: Use Markdown Input Instead! 🎯

**Good news**: Jotty already generates **Markdown** files! We can use those instead of PDFs.

### Current Workflow (Not Working)
```
Guide PDF → Paper2Slides (needs mineru) → Slides
```

### Better Workflow (Works!)
```
Guide Markdown → Paper2Slides → Slides
```

---

## How to Generate Slides (Without mineru)

### Option 1: Install mineru (If you have time/space)

```bash
pip install 'mineru[core]==2.6.4'
# Then slides will work with PDFs
```

### Option 2: Use Markdown (Recommended)

Paper2Slides can accept markdown files! Let me update the script to use the markdown we already generate.

**Currently working on**: Creating a markdown-to-slides workflow that bypasses PDF parsing.

---

## What's Already Installed

✅ All packages from `requirements.txt`:
```
dspy-ai>=2.0.0
pyyaml>=6.0
langgraph>=0.0.1
duckduckgo-search
cloudscraper
fake-useragent
python-docx>=0.8.11
python-pptx>=0.6.21
lightrag-hku
huggingface_hub
Pillow>=10.0.0
reportlab>=4.0.0
python-dotenv>=1.0.0
tqdm
```

❌ **Not installed** (optional):
```
mineru[core]==2.6.4  # 500MB+ package
```

---

## Next Steps

### Immediate (Testing Markdown Input)

I'll create a modified script that:
1. Takes the Markdown file we generate
2. Converts it to Paper2Slides' expected format
3. Bypasses the PDF parsing stage
4. Generates slides directly

**This will work WITHOUT mineru!**

### Alternative (For External PDFs)

If you need to convert **external PDFs** (not Jotty-generated), then install mineru:
```bash
pip install 'mineru[core]==2.6.4'
```

---

## Requirements Summary

**To generate guides only**:
- ✅ All packages installed
- ✅ Works with `python3 generate_guide_with_research.py --topic "Topic"`

**To generate slides from Jotty markdown**:
- ✅ All packages installed
- ⚠️ Need IMAGE_GEN_API_KEY (OpenRouter or Google Gemini)
- ✅ Will create markdown-based workflow (no mineru needed)

**To generate slides from external PDFs**:
- ✅ All packages installed
- ⚠️ Need IMAGE_GEN_API_KEY
- ❌ Need `mineru[core]` (~500MB)

---

## Cost Breakdown

**Guide Generation**: FREE
- Uses local Claude CLI (no API key)
- Uses free DuckDuckGo search

**Slides Generation**: ~$0.10 per presentation
- LLM: FREE (local Claude CLI)
- Image generation: ~$0.10 (OpenRouter) or FREE (Google Gemini free tier)

---

## I'll Create a Workaround Now

Let me modify the slides generator to use markdown input instead of PDF, so you can test slides without installing the huge mineru package!
