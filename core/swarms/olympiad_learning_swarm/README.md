# Olympiad Learning Swarm

Generate comprehensive educational content for K-12 students with professional PDF and interactive HTML output.

## 🎯 Purpose

Creates engaging learning materials that:
- Build from fundamentals to advanced concepts
- Include worked examples and practice problems
- Generate both PDF (A4 format) and interactive HTML
- Optionally send to Telegram for easy sharing
- Use 12 specialized agents for content generation

## ✨ Features

- **Multi-tier difficulty**: Foundation → Intermediate → Advanced → Olympiad
- **Multiple subjects**: Math, Physics, Chemistry, Biology, Computer Science, General
- **Adaptive depth**: Quick (5 min) → Standard → Deep → Marathon
- **Professional output**: A4 PDF + responsive HTML with A4 preview
- **Rich content**: Concepts, patterns, problems, examples, real-life scenarios
- **Optimization modes**: Parallel deep (fast, 30+ pages) or sequential (detailed)

## 🚀 Quick Start

```python
from Jotty.core.swarms.olympiad_learning_swarm import learn_topic

# Generate 5th grade economics content
result = await learn_topic(
    subject="general",
    topic="Economics for 5th Grade",
    student_name="Alex",
    depth="deep",              # quick/standard/deep/marathon
    target="foundation",       # foundation/intermediate/advanced/olympiad
    send_telegram=True
)

print(f"Generated: {result.pdf_path}")
print(f"HTML: {result.html_path}")
```

## 📋 Configuration

### Subject Options

```python
from Jotty.core.swarms.olympiad_learning_swarm.types import Subject

Subject.MATHEMATICS       # Math problems and concepts
Subject.PHYSICS          # Physics with equations
Subject.CHEMISTRY        # Chemistry with reactions
Subject.BIOLOGY          # Biology with diagrams
Subject.COMPUTER_SCIENCE # Programming and algorithms
Subject.GENERAL          # Any topic (economics, history, etc.)
```

### Lesson Depth

```python
from Jotty.core.swarms.olympiad_learning_swarm.types import LessonDepth

LessonDepth.QUICK     # 5-minute overview
LessonDepth.STANDARD  # Full understanding (default)
LessonDepth.DEEP      # Expert-level with all details
LessonDepth.MARATHON  # Comprehensive, competition-ready
```

### Difficulty Tiers

```python
from Jotty.core.swarms.olympiad_learning_swarm.types import DifficultyTier

DifficultyTier.FOUNDATION    # Basics, build from scratch
DifficultyTier.INTERMEDIATE  # Standard curriculum
DifficultyTier.ADVANCED      # Challenging problems
DifficultyTier.OLYMPIAD      # Competition-level
```

## 🎓 Full Configuration Example

```python
from Jotty.core.swarms.olympiad_learning_swarm import OlympiadLearningSwarm
from Jotty.core.swarms.olympiad_learning_swarm.types import (
    OlympiadLearningConfig,
    Subject,
    LessonDepth,
    DifficultyTier
)

config = OlympiadLearningConfig(
    subject=Subject.MATHEMATICS,
    student_name="Jordan",
    depth=LessonDepth.DEEP,
    target_tier=DifficultyTier.ADVANCED,

    # Problem counts per tier
    foundation_problems=3,
    intermediate_problems=5,
    advanced_problems=5,
    olympiad_problems=2,

    # Output options
    generate_pdf=True,
    generate_html=True,
    send_telegram=True,
    celebration_word="Excellent!",

    # Optimization
    optimization_mode="parallel_deep",  # Fast, 30+ pages in ~90s
    max_concurrent_llm=5,
    llm_model="haiku",  # Fast model
)

swarm = OlympiadLearningSwarm(config)
result = await swarm.execute(topic="Calculus - Limits and Continuity")
```

## 🎨 Output Format

### PDF Output
- **Format**: A4 (210mm × 297mm)
- **Style**: Professional, clean typography
- **Sections**:
  - Title page with student name
  - Table of contents
  - Concepts and patterns
  - Worked examples
  - Practice problems with difficulty markers
  - Real-life applications
- **Size**: Typically 20-50 pages depending on depth

### HTML Output
- **Responsive**: Works on mobile, tablet, desktop
- **A4 Preview**: Screen shows A4 page layout
- **Interactive**: Clickable TOC, expandable solutions
- **Print-ready**: Matches PDF when printed

## 💡 Use Cases

### 1. Homework Help
```python
result = await learn_topic(
    subject="mathematics",
    topic="Quadratic Equations",
    student_name="Student",
    depth="standard",
    target="foundation"
)
```

### 2. Competition Prep
```python
config = OlympiadLearningConfig(
    subject=Subject.PHYSICS,
    depth=LessonDepth.MARATHON,
    target_tier=DifficultyTier.OLYMPIAD,
    olympiad_problems=10
)
swarm = OlympiadLearningSwarm(config)
result = await swarm.execute(topic="Thermodynamics")
```

### 3. Tutoring Materials
```python
# Generate multiple lessons
topics = ["Fractions", "Decimals", "Percentages"]
for topic in topics:
    result = await learn_topic(
        subject="mathematics",
        topic=topic,
        student_name="Class 6A",
        depth="deep",
        target="foundation",
        send_telegram=True
    )
```

## 🔧 Optimization Modes

### parallel_deep (Recommended)
- **Speed**: ~90 seconds
- **Quality**: High (30+ pages)
- **Method**: Parallel per-concept generation
- **Use**: Default for most cases

### unified
- **Speed**: ~60 seconds
- **Quality**: Good (shorter)
- **Method**: Single LLM call
- **Use**: Quick overviews

### sequential
- **Speed**: ~5 minutes
- **Quality**: Highest (most detailed)
- **Method**: Step-by-step generation
- **Use**: Maximum quality needed

## 📊 Output Statistics

Typical output for `depth=deep, target=foundation`:
- **Pages**: 25-35
- **Words**: 6,000-8,000
- **Problems**: 8-12 total
- **Examples**: 5-8 worked examples
- **Time**: 90 seconds (parallel_deep)
- **Cost**: $0.10-$0.20

## 🎯 Tips & Best Practices

**For Best Results:**
1. Be specific in topic description: "Linear Equations with Word Problems" vs "Algebra"
2. Match depth to student level: Quick for overview, Deep for thorough understanding
3. Use parallel_deep mode for speed without sacrificing quality
4. Enable send_telegram for easy sharing with students/parents

**Common Pitfalls:**
- Topic too broad: "Mathematics" → Better: "Pythagorean Theorem"
- Wrong tier: Olympiad-level for beginners → Start with Foundation
- Too many problems: 20+ problems → Reduces quality, stick to 10-15 total

## 🏗️ Architecture

**12 Specialized Agents:**
1. **ConceptExplorer** - Identifies key concepts
2. **PatternHunter** - Finds problem-solving patterns
3. **ExampleCreator** - Generates worked examples
4. **ProblemDesigner** - Creates practice problems
5. **RealLifeLinker** - Connects to real-world applications
6. **DifficultyCalibrator** - Ensures appropriate challenge
7. **ExplanationEnhancer** - Improves clarity
8. **VisualizationSuggester** - Recommends diagrams
9. **MistakeAnalyzer** - Identifies common errors
10. **SolutionValidator** - Verifies correctness
11. **ProgressionPlanner** - Orders content logically
12. **NarrativeEditor** - Ensures cohesive flow

## 📚 Related

- **ArxivLearningSwarm**: For research paper learning
- **ResearchSwarm**: For general topic research
- **CodingSwarm**: For programming lessons

## 🐛 Troubleshooting

**Problem**: PDF/HTML don't match
- **Solution**: Regenerate with latest version (A4 styling added)

**Problem**: Timeout errors
- **Solution**: Increase `llm_timeout` in config or use faster model

**Problem**: Content too basic/advanced
- **Solution**: Adjust `target_tier` parameter

**Problem**: Too short/long
- **Solution**: Change `depth` parameter

## 📄 License

Part of Jotty AI Framework - See main LICENSE file
