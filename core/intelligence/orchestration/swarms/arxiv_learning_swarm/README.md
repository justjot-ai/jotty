# ArXiv Learning Swarm

Transform academic papers into engaging learning materials.

## 🎯 Purpose

Creates educational content from ArXiv papers:
- Breaks down complex research
- Builds from fundamentals
- Includes code examples and visualizations
- Generates PDF + PowerPoint + HTML
- Optimized for understanding

## 🚀 Quick Start

```python
from Jotty.core.swarms.arxiv_learning_swarm import learn_paper

# By ArXiv ID
result = await learn_paper("2301.00001")

# By topic search
result = await learn_paper_by_topic("transformer attention mechanisms")
```

## 📋 Configuration

```python
from Jotty.core.swarms.arxiv_learning_swarm.types import (
    ArxivLearningConfig,
    LearningDepth,
    ContentStyle,
    AudienceLevel
)

config = ArxivLearningConfig(
    depth=LearningDepth.DEEP,
    style=ContentStyle.ENGAGING,
    audience=AudienceLevel.INTERMEDIATE,
    include_code_examples=True,
    include_visualizations=True,
    generate_pptx=True,
    send_telegram=True,
    optimization_mode="parallel_deep",
)
```

## 🎓 Output Formats

- **PDF**: Professional learning guide
- **PowerPoint**: Presentation slides
- **HTML**: Interactive web version
- All formats optimized for teaching

## 💡 Use Cases

- Course material creation
- Research paper study groups
- Technical presentations
- Self-paced learning

## 📄 License

Part of Jotty AI Framework
