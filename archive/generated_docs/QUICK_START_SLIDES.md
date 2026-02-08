# Quick Start: Free Slides Generation with Google Gemini

All packages are now installed! Just need your FREE Google Gemini API key.

---

## Step 1: Get Your FREE Google Gemini API Key

### Visit Google AI Studio
🔗 **https://aistudio.google.com/app/apikey**

### What to Do:
1. Sign in with your Google account
2. Click **"Get API key"** or **"Create API key"**
3. Select **"Create API key in new project"**
4. **Copy the API key** (looks like: `AIzaSyXXXXXXXXXXXXXXXX`)

### Free Tier Benefits:
- ✅ 60 requests per minute
- ✅ 1,500 requests per day
- ✅ **FREE forever** for moderate use
- ✅ Perfect for slide generation!

---

## Step 2: Add Your API Key to Configuration

Open the config file:
```bash
nano Paper2Slides/paper2slides/.env
```

Find this line:
```bash
IMAGE_GEN_API_KEY=""
```

Replace with your key:
```bash
IMAGE_GEN_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

**Save**: Ctrl+X, then Y, then Enter

---

## Step 3: Test with Poodles! 🐩

### Option A: Generate NEW Guide + Slides
```bash
python3 generate_guide_with_slides.py --topic "Poodles"
```

**What happens**:
1. Multi-agent research (2-4 min)
2. Guide generation (10 sec)
3. Slides generation (5-10 min)

**Output**:
```
outputs/poodles_guide/
├── Poodles_Guide_a4.pdf         ← Research guide
├── Poodles_Guide.md
└── Poodles_Guide.html

outputs/slides/poodles_guide/
└── academic_<timestamp>/
    ├── slides.pdf               ← Presentation deck!
    ├── slide_001.png
    ├── slide_002.png
    └── slide_003.png (10-15 slides total)
```

### Option B: Use EXISTING Poodles PDF
```bash
python3 core/tools/content_generation/slides_generator.py \
    --input outputs/poodles_guide/Poodles_for_Dummies_A_Comprehensive_Guide_a4.pdf \
    --style academic \
    --length medium
```

**What happens**:
- Skips research/guide generation
- Just creates slides from existing PDF
- Faster: 5-7 minutes

---

## Expected Output

### Terminal Output:
```
🎨 Generating slides from: Poodles_Guide.pdf
   Style: academic
   Length: medium

================================================================================
  PAPER2SLIDES PIPELINE
================================================================================

────────────────────────────────────────────────────────────
STAGE: RAG
────────────────────────────────────────────────────────────
Parsing file: Poodles_Guide.pdf
✅ Parsing completed: 1 successful

────────────────────────────────────────────────────────────
STAGE: SUMMARY
────────────────────────────────────────────────────────────
Analyzing content...
✅ Content analysis complete

────────────────────────────────────────────────────────────
STAGE: PLAN
────────────────────────────────────────────────────────────
Planning slide layout...
✅ Planned 12 slides

────────────────────────────────────────────────────────────
STAGE: GENERATE
────────────────────────────────────────────────────────────
Generating slide 1/12...
Generating slide 2/12...
...
✅ All slides generated!

================================================================================
  SLIDES GENERATION COMPLETE!
================================================================================

✅ Generated 12 slides
✅ PDF: slides.pdf (2.3 MB)
📁 Output: outputs/slides/poodles_guide/...
```

---

## Different Styles

### Academic (Professional)
```bash
--style academic
```
Clean, formal, suitable for presentations

### Doraemon (Colorful)
```bash
--style doraemon
```
Illustrated, engaging, suitable for education

### Custom
```bash
--style "minimalist with blue theme and modern fonts"
```
Describe what you want!

---

## Different Lengths

```bash
--length short    # 5-8 slides
--length medium   # 10-15 slides (default)
--length long     # 20+ slides
```

---

## Troubleshooting

### Error: "IMAGE_GEN_API_KEY not set"
✅ **Fix**: Add your Gemini API key to `.env` file

### Error: "Rate limit exceeded"
✅ **Fix**: Wait 60 seconds, then re-run (checkpoints preserved!)

### Error: "mineru command not found"
✅ **Fix**: Already installed! Should work now.

### Want to change style?
✅ Just re-run with different `--style` - it reuses RAG/Analysis (faster)

---

## Cost

**Everything is FREE!**
- ✅ LLM: FREE (local Claude CLI)
- ✅ Image generation: FREE (Google Gemini free tier)
- ✅ Research: FREE (DuckDuckGo)
- ✅ Storage: FREE (local files)

**Total cost**: $0.00 🎉

---

## Summary Checklist

- [ ] Get Google Gemini API key from https://aistudio.google.com/app/apikey
- [ ] Add key to `Paper2Slides/paper2slides/.env`
- [ ] Run test command
- [ ] Wait 5-10 minutes for slides
- [ ] Check output directory

**Ready to generate your first presentation!** 🚀
