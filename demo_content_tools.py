#!/usr/bin/env python3
"""
DEMO: Content Generation Tools
================================

Quick demonstration of ported JustJot.ai content generation tools:
- Document creation
- Markdown export
- HTML generation
- PDF generation

Creates a sample Transformer paper to show all features.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, Section, SectionType, ContentGenerators


def create_sample_transformer_document() -> Document:
    """
    Create a comprehensive Transformer research paper

    Demonstrates:
    - Text sections
    - Mathematical formulas (LaTeX)
    - Mermaid diagrams
    - Code blocks
    - Professional structure
    """

    doc = Document(
        title="Attention is All You Need: Understanding Transformer Architecture",
        author="Jotty Research Team",
        topic="Deep Learning",
        source_type="jotty"
    )

    # Abstract
    doc.add_section(
        SectionType.TEXT,
        """
The Transformer architecture revolutionized deep learning by introducing self-attention mechanisms
that eliminate the need for recurrent or convolutional layers. This paper provides a comprehensive
explanation of the Transformer model, from intuitive concepts to mathematical foundations.
We explore why Transformers succeeded where previous architectures struggled, examine their
mathematical formulation, and analyze their impact across NLP, computer vision, and beyond.
        """.strip(),
        title="Abstract"
    )

    # Introduction
    doc.add_section(
        SectionType.TEXT,
        """
Before Transformers, sequence modeling relied heavily on Recurrent Neural Networks (RNNs) and
their variants (LSTMs, GRUs). These architectures processed sequences sequentially, inherently
limiting parallelization. Long-range dependencies remained challenging due to vanishing gradients.

**The Central Question**: Can we build a sequence model that:
- Processes input in parallel (not sequentially)?
- Handles long-range dependencies without gradient issues?
- Remains interpretable?

The Transformer architecture, introduced by Vaswani et al. in "Attention is All You Need" (2017),
answered "yes" to all three questions. The key innovation: **self-attention** - a mechanism that
weighs the importance of different parts of the input when processing each element.

**Analogy**: Think of reading a research paper. When you encounter a pronoun ("it"), you
automatically "attend" to relevant nouns mentioned earlier. You don't re-read the entire paper
sequentially - you jump directly to relevant context. This is precisely what self-attention does.
        """.strip(),
        title="Introduction: The Problem with Sequential Processing"
    )

    # Architecture Diagram
    doc.add_section(
        SectionType.MERMAID,
        """
graph TB
    Input[Input Embedding] --> PE[Positional Encoding]
    PE --> Encoder[Encoder Stack<br/>6 Layers]

    Encoder --> E1[Multi-Head<br/>Self-Attention]
    E1 --> E2[Add & Norm]
    E2 --> E3[Feed-Forward]
    E3 --> E4[Add & Norm]

    E4 --> Decoder[Decoder Stack<br/>6 Layers]

    Decoder --> D1[Masked Multi-Head<br/>Self-Attention]
    D1 --> D2[Add & Norm]
    D2 --> D3[Multi-Head<br/>Cross-Attention]
    D3 --> D4[Add & Norm]
    D4 --> D5[Feed-Forward]
    D5 --> D6[Add & Norm]

    D6 --> Linear[Linear]
    Linear --> Softmax[Softmax]
    Softmax --> Output[Output Probabilities]

    style Encoder fill:#e1f5ff
    style Decoder fill:#ffe1e1
        """.strip(),
        title="Figure 1: Transformer Architecture"
    )

    # Self-Attention Explained
    doc.add_section(
        SectionType.TEXT,
        """
**Self-Attention: The Core Innovation**

Self-attention allows each position in a sequence to attend to all positions in the previous layer.
Here's how it works:

1. **Create three vectors** for each input token:
   - Query (Q): "What am I looking for?"
   - Key (K): "What do I contain?"
   - Value (V): "What do I output?"

2. **Compute attention scores**: Compare each query with all keys (dot product)

3. **Normalize scores**: Apply softmax to get attention weights (sum to 1)

4. **Weighted sum**: Multiply weights by values

**Why it works**: Tokens that are semantically related will have high dot products between
their Q and K vectors, receiving high attention weights.

**Multi-Head Attention**: Instead of one attention mechanism, use 8 parallel attention "heads".
Each head learns different aspects of relationships (syntax, semantics, coreference, etc.).
        """.strip(),
        title="Self-Attention Mechanism"
    )

    # Mathematics
    doc.add_section(
        SectionType.MATH,
        r"""
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

\text{where:}
\begin{align*}
Q &= \text{queries matrix} \in \mathbb{R}^{n \times d_k} \\
K &= \text{keys matrix} \in \mathbb{R}^{n \times d_k} \\
V &= \text{values matrix} \in \mathbb{R}^{n \times d_v} \\
d_k &= \text{dimension of keys (typically 64)} \\
n &= \text{sequence length}
\end{align*}

\text{Multi-Head Attention:}

\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O

\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
        """.strip(),
        title="Mathematical Formulation"
    )

    # Python implementation
    doc.add_section(
        SectionType.CODE,
        """
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
        """.strip(),
        title="PyTorch Implementation",
        language="python"
    )

    # Complexity Analysis
    doc.add_section(
        SectionType.TEXT,
        """
**Computational Complexity**

| Layer Type | Complexity | Sequential Operations | Max Path Length |
|------------|------------|----------------------|-----------------|
| Self-Attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |

Where:
- n = sequence length
- d = representation dimension
- k = kernel size

**Key Advantages**:
- **Parallelization**: O(1) sequential operations vs RNN's O(n)
- **Long-range dependencies**: O(1) path length vs RNN's O(n)
- **Trade-off**: O(n²) complexity in sequence length
        """.strip(),
        title="Complexity Analysis"
    )

    # Applications
    doc.add_section(
        SectionType.TEXT,
        """
**Transformers Everywhere: Applications**

The Transformer architecture has become the foundation for state-of-the-art models across domains:

**Natural Language Processing**:
- GPT family (GPT-3, GPT-4): Autoregressive language models
- BERT: Bidirectional encoder representations
- T5: Text-to-text transfer transformer

**Computer Vision**:
- Vision Transformer (ViT): Image classification
- DETR: Object detection
- Swin Transformer: Hierarchical vision transformer

**Multi-Modal**:
- CLIP: Image-text understanding
- DALL-E: Text-to-image generation
- Flamingo: Few-shot visual understanding

**Scientific Applications**:
- AlphaFold: Protein structure prediction
- MusicGen: Audio generation
- Galactica: Scientific knowledge
        """.strip(),
        title="Applications and Impact"
    )

    # Conclusion
    doc.add_section(
        SectionType.TEXT,
        """
The Transformer architecture represents a paradigm shift in sequence modeling. By replacing
recurrence with self-attention, it unlocked:

1. **Massive parallelization** during training
2. **Better long-range dependencies** without vanishing gradients
3. **Interpretable attention patterns**
4. **Transfer learning** capabilities across domains

While the O(n²) complexity poses challenges for very long sequences, variants like Linformer,
Reformer, and Longformer address these limitations.

The impact extends far beyond NLP - Transformers are now the default architecture for most
deep learning applications, from vision to biology to reinforcement learning.

**Future Directions**:
- Efficient attention mechanisms (linear complexity)
- Longer context windows (100K+ tokens)
- Multi-modal foundation models
- Sparse and mixture-of-experts architectures
        """.strip(),
        title="Conclusion"
    )

    # References
    doc.add_section(
        SectionType.TEXT,
        """
1. Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.

2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers
   for Language Understanding." NAACL 2019.

3. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers
   for Image Recognition at Scale." ICLR 2021.

4. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.

5. Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold."
   Nature 596, 583–589.

6. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural
   Language Supervision." ICML 2021.
        """.strip(),
        title="References"
    )

    return doc


def main():
    """Generate all output formats"""

    print("\n" + "=" * 100)
    print("  CONTENT GENERATION TOOLS DEMO")
    print("  Ported from JustJot.ai adapters/sinks/")
    print("=" * 100 + "\n")

    # Create output directory
    output_dir = Path("./outputs/research/transformer_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create document
    print("📄 Creating Transformer research paper...")
    doc = create_sample_transformer_document()
    print(f"✅ Document created: {len(doc.full_content):,} characters")
    print(f"   - {len(doc.sections)} sections")
    print(f"   - ~{len(doc.full_content.split()):,} words\n")

    # Initialize generators
    generators = ContentGenerators()

    print("=" * 100)
    print("  GENERATING OUTPUT FILES")
    print("=" * 100 + "\n")

    # 1. Markdown
    print("1️⃣  Exporting Markdown...")
    md_path = generators.export_markdown(doc, output_path=output_dir)
    print(f"   ✅ MD: {md_path.name}")
    print(f"   📊 Size: {md_path.stat().st_size:,} bytes\n")

    # 2. HTML
    print("2️⃣  Generating HTML...")
    try:
        html_path = generators.generate_html(
            doc,
            output_path=output_dir,
            standalone=True,
            include_toc=True
        )
        print(f"   ✅ HTML: {html_path.name}")
        print(f"   📊 Size: {html_path.stat().st_size:,} bytes\n")
    except Exception as e:
        print(f"   ⚠️  HTML generation skipped: {e}\n")

    # 3. PDF
    print("3️⃣  Generating PDF...")
    try:
        pdf_path = generators.generate_pdf(
            doc,
            output_path=output_dir,
            format='a4'
        )
        print(f"   ✅ PDF: {pdf_path.name}")
        print(f"   📊 Size: {pdf_path.stat().st_size:,} bytes\n")
    except Exception as e:
        print(f"   ⚠️  PDF generation skipped: {e}\n")

    print("=" * 100)
    print("  SUCCESS")
    print("=" * 100 + "\n")

    print(f"📁 Output Directory: {output_dir}")
    print(f"📄 Files Generated:")

    for file in sorted(output_dir.glob("*")):
        size_kb = file.stat().st_size / 1024
        print(f"   - {file.name} ({size_kb:.1f} KB)")

    print("\n🎯 Features Demonstrated:")
    print("   ✅ Document model (ported from JustJot.ai)")
    print("   ✅ Section types (text, math, mermaid, code)")
    print("   ✅ Markdown export")
    print("   ✅ HTML generation (pandoc)")
    print("   ✅ PDF generation (pandoc + XeLaTeX)")
    print("   ✅ Professional formatting")
    print("   ✅ Mathematical formulas (LaTeX)")
    print("   ✅ Diagrams (Mermaid)")
    print("   ✅ Code blocks (Python)")
    print()

    print("🚀 Content generation tools successfully ported to Jotty!")
    print("   Ready for use by research agents in multi-agent workflows.")
    print()


if __name__ == "__main__":
    main()
