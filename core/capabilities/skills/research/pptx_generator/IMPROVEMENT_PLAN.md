# PPTX Generator Improvement Plan
## Based on Genspark Analysis

---

## Current Architecture
```
Paper Data → LIDA Planner (LLM) → JSON Specs → PptxGenJS (shapes) → PPTX
```

## Target Architecture (Genspark-inspired)
```
Paper Data → Outline Planner (LLM) → Slide Structure
                                          ↓
                              → Image Generator → Diagram Images
                              → Content Generator → Text/Bullets
                                          ↓
                                    PptxGenJS → PPTX
```

---

## Key Improvements

### 1. AI-Generated Diagram Images (HIGH IMPACT)

**Current**: PptxGenJS draws shapes (boxes, arrows, lines)
- Prone to overflow
- Limited visual appeal
- Complex positioning code

**Target**: AI generates diagram images
- Professional-looking diagrams
- No overflow issues (pre-sized)
- Much simpler code

**Implementation**:
```python
# New DiagramImageGenerator class
class DiagramImageGenerator:
    async def generate_architecture_image(self, spec: dict) -> str:
        """Generate architecture diagram image using AI."""
        prompt = f"""
        Create a professional technical diagram showing:
        - Title: {spec['title']}
        - Components: {spec['nodes']}
        - Connections: {spec['connections']}

        Style: Clean, minimal, blue color scheme, white background
        Format: 1920x1080, suitable for presentation slide
        """
        image_path = await self.image_api.generate(prompt)
        return image_path
```

**Options for Image Generation**:
1. OpenAI DALL-E 3 (best quality, paid)
2. Stable Diffusion (local, free)
3. Replicate API (various models)
4. Claude's image generation (if available)

### 2. Deck Outline Planning (MEDIUM IMPACT)

**Current**: Fixed slide order based on content
**Target**: LLM plans optimal deck structure

```python
class DeckOutlinePlanner(dspy.Signature):
    """Plan the optimal deck structure for this paper."""

    paper_summary: str = dspy.InputField()
    concepts: list = dspy.InputField()

    outline: list = dspy.OutputField(desc="""
        List of slides with:
        - slide_type: title|content|diagram|comparison|metrics|quote
        - title: slide title
        - purpose: what this slide achieves
        - key_points: main content points
    """)
```

### 3. Simplified Slide Templates (MEDIUM IMPACT)

Reduce from 10+ complex templates to 5 clean ones:

| Template | Use Case | Design |
|----------|----------|--------|
| **Hero** | Title, key quotes | Full-bleed background, large text |
| **Content** | Text + bullets | Clean white, accent bar |
| **Diagram** | Architecture, flow | Full-width image |
| **Comparison** | Before/after | Two-column layout |
| **Metrics** | Stats, KPIs | Card grid |

### 4. Research Integration (LOW PRIORITY)

Add web search before generation:
- Search for paper citations
- Find related diagrams for inspiration
- Get latest context

---

## Implementation Order

### Phase 1: Image Generation (1-2 days)
1. Add `DiagramImageGenerator` class
2. Integrate with existing LIDA specs
3. Generate images for: architecture, flow, concept_map
4. Insert images into slides instead of shapes

### Phase 2: Outline Planning (1 day)
1. Add `DeckOutlinePlanner` DSPy signature
2. Generate outline before slides
3. Use outline to guide slide generation

### Phase 3: Template Cleanup (1 day)
1. Simplify to 5 core templates
2. Remove complex shape-based diagrams
3. Cleaner, more consistent styling

---

## Image Generation Prompt Templates

### Architecture Diagram
```
Create a professional technical architecture diagram for a presentation slide.

Topic: {title}
Components:
{formatted_nodes}

Style requirements:
- Clean, minimal design
- Navy blue (#1e3a5f) primary color
- White background
- Clear labels with readable fonts
- Arrows showing data flow
- Subtle shadows for depth
- 16:9 aspect ratio (1920x1080)
- No text outside the diagram
- Professional, suitable for business/academic presentation
```

### Comparison Diagram
```
Create a professional comparison diagram for a presentation slide.

Left side: {left_title}
- {left_points}

Right side: {right_title}
- {right_points}

Style requirements:
- Split layout with clear divider
- Left side: warm colors (orange/red accent)
- Right side: cool colors (green/blue accent)
- Check/X icons for benefits/drawbacks
- Clean, minimal design
- 16:9 aspect ratio
```

### Concept Map
```
Create a professional concept relationship diagram for a presentation slide.

Central concept: {center}
Related concepts: {related}

Style requirements:
- Hub and spoke layout
- Central node larger and prominent
- Clean connecting lines
- Color-coded by category
- Minimal, professional design
- 16:9 aspect ratio
```

---

## Files to Modify

1. **NEW**: `diagram_image_generator.py` - AI image generation
2. **MODIFY**: `visualization_planner.py` - Add outline planning
3. **MODIFY**: `generate_pptx.js` - Use images instead of shapes
4. **MODIFY**: `__init__.py` - Integrate new components

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Diagram overflow issues | ~20% | 0% |
| Visual quality (subjective) | 6/10 | 9/10 |
| Generation time | ~30s | ~60s (image gen adds latency) |
| Code complexity | High | Medium |
