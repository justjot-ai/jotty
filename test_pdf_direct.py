#!/usr/bin/env python3
"""
Direct PDF Generation Test (No LLM)
====================================

Tests PDF generation with properly formatted LaTeX math to ensure it works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, Section, SectionType, ContentGenerators


def test_pdf_generation():
    """Test PDF generation with proper LaTeX syntax"""

    print("\n" + "=" * 80)
    print("  PDF GENERATION TEST (Direct, No LLM)")
    print("=" * 80 + "\n")

    # Create simple document with working LaTeX
    doc = Document(
        title="Transformer Architecture - Quick Reference",
        author="Jotty Test",
        topic="Deep Learning"
    )

    # Introduction
    doc.add_section(
        SectionType.TEXT,
        """
The Transformer architecture uses self-attention mechanisms to process sequences
in parallel, achieving state-of-the-art results in NLP and beyond.

**Key Innovation**: Self-attention replaces recurrence, enabling:
- Parallel processing (no sequential bottleneck)
- Better long-range dependencies
- Interpretable attention patterns
        """.strip(),
        title="Introduction"
    )

    # Math section with PROPER LaTeX delimiters for pandoc
    doc.add_section(
        SectionType.TEXT,
        r"""
**Attention Formula**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = queries matrix
- $K$ = keys matrix
- $V$ = values matrix
- $d_k$ = dimension of keys (typically 64)

**Multi-Head Attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
        """.strip(),
        title="Mathematics"
    )

    # Architecture
    doc.add_section(
        SectionType.TEXT,
        """
**Encoder**: 6 identical layers
- Multi-head self-attention
- Feed-forward network
- Residual connections + layer normalization

**Decoder**: 6 identical layers
- Masked multi-head self-attention
- Encoder-decoder attention
- Feed-forward network
- Residual connections + layer normalization

**Complexity**: $O(n^2 \cdot d)$ for self-attention vs $O(n \cdot d^2)$ for RNN
        """.strip(),
        title="Architecture"
    )

    # Code example
    doc.add_section(
        SectionType.CODE,
        """
import torch.nn as nn

class SelfAttention(nn.Module):
    def forward(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
        """.strip(),
        title="Implementation",
        language="python"
    )

    print(f"📄 Document created: {len(doc.full_content)} characters\n")

    # Test PDF generation
    output_dir = Path("./outputs/pdf_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = ContentGenerators()

    print("Testing PDF generation...")
    print("-" * 80)

    try:
        pdf_path = generators.generate_pdf(
            doc,
            output_path=output_dir,
            format='a4'
        )

        file_size = pdf_path.stat().st_size
        print(f"\n✅ SUCCESS!")
        print(f"   PDF: {pdf_path.name}")
        print(f"   Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"   Path: {pdf_path}")
        print()

        # Verify PDF has content
        if file_size < 5000:
            print("⚠️  Warning: PDF seems small, might have rendering issues")
        else:
            print("✅ PDF has substantial content")

        return True

    except Exception as e:
        print(f"\n❌ PDF generation failed:")
        print(f"   Error: {e}")
        print()
        print("Possible issues:")
        print("   - pandoc not installed")
        print("   - xelatex not installed")
        print("   - LaTeX packages missing")
        print()
        print("Install with:")
        print("   sudo yum install pandoc texlive-xetex texlive-latex-recommended")
        return False


if __name__ == "__main__":
    success = test_pdf_generation()

    print("=" * 80)
    if success:
        print("  PDF GENERATION: ✅ WORKING")
    else:
        print("  PDF GENERATION: ⚠️  NEEDS DEPENDENCIES")
    print("=" * 80)
    print()

    sys.exit(0 if success else 1)
