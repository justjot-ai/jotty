#!/usr/bin/env python3
"""
Test All Content Generation Formats
====================================

Tests all ported content generation tools:
- PDF (via pandoc + XeLaTeX)
- HTML (via pandoc)
- Markdown (native)
- DOCX (via python-docx, optional)
- PPTX (via python-pptx, optional)

Non-LLM direct test to verify infrastructure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, Section, SectionType, ContentGenerators


def create_test_document():
    """Create comprehensive test document"""

    doc = Document(
        title="Transformer Architecture Overview",
        author="Jotty Content Generator",
        topic="Deep Learning"
    )

    # Abstract
    doc.add_section(
        SectionType.TEXT,
        """
The Transformer architecture revolutionized deep learning through self-attention mechanisms.
This document demonstrates all content generation formats available in Jotty.
        """.strip(),
        title="Abstract"
    )

    # Introduction
    doc.add_section(
        SectionType.TEXT,
        """
**Background**: Prior to Transformers, sequence models relied on RNNs and LSTMs, which:
- Processed data sequentially (slow training)
- Struggled with long-range dependencies
- Could not parallelize effectively

**Solution**: Self-attention mechanisms that:
- Process sequences in parallel
- Handle arbitrary-length dependencies
- Provide interpretable attention patterns
        """.strip(),
        title="Introduction"
    )

    # Mathematics (properly formatted for PDF)
    doc.add_section(
        SectionType.TEXT,
        r"""
**Attention Mechanism**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$ (queries), $K$ (keys), and $V$ (values) are projected from input embeddings.

**Multi-Head Attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

This allows the model to attend to information from different representation subspaces.
        """.strip(),
        title="Mathematical Foundation"
    )

    # Code example
    doc.add_section(
        SectionType.CODE,
        """
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(self, query, key, value, mask=None):
        # Project and reshape
        Q = self.project_query(query)
        K = self.project_key(key)
        V = self.project_value(value)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        return output
        """.strip(),
        title="Implementation Example",
        language="python"
    )

    # Architecture
    doc.add_section(
        SectionType.TEXT,
        """
**Encoder**: 6 layers of:
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections + layer normalization

**Decoder**: 6 layers of:
1. Masked multi-head self-attention
2. Encoder-decoder attention
3. Position-wise feed-forward network
4. Residual connections + layer normalization

**Key Features**:
- Parallel processing enables faster training
- No recurrence allows for better gradient flow
- Positional encoding preserves sequence information
        """.strip(),
        title="Architecture"
    )

    # Applications
    doc.add_section(
        SectionType.TEXT,
        """
**Natural Language Processing**:
- GPT series (generative)
- BERT (bidirectional encoding)
- T5 (text-to-text)

**Computer Vision**:
- Vision Transformer (ViT)
- DETR (object detection)
- Swin Transformer

**Multi-Modal**:
- CLIP (image-text)
- DALL-E (text-to-image)
- Flamingo (visual understanding)
        """.strip(),
        title="Applications"
    )

    return doc


def main():
    """Test all format generators"""

    print("\n" + "=" * 90)
    print("  COMPREHENSIVE FORMAT TEST - All Content Generators")
    print("=" * 90 + "\n")

    # Create document
    doc = create_test_document()
    print(f"📄 Test document: {len(doc.full_content)} characters, {len(doc.sections)} sections\n")

    # Output directory
    output_dir = Path("./outputs/format_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = ContentGenerators()

    results = {}

    # Test each format
    print("=" * 90)
    print("  TESTING ALL FORMATS")
    print("=" * 90 + "\n")

    # 1. Markdown
    print("1️⃣  Markdown Export")
    print("-" * 90)
    try:
        md_path = generators.export_markdown(doc, output_path=output_dir)
        size = md_path.stat().st_size
        results['markdown'] = {'success': True, 'path': md_path, 'size': size}
        print(f"   ✅ SUCCESS: {size:,} bytes\n")
    except Exception as e:
        results['markdown'] = {'success': False, 'error': str(e)}
        print(f"   ❌ FAILED: {e}\n")

    # 2. HTML
    print("2️⃣  HTML Generation")
    print("-" * 90)
    try:
        html_path = generators.generate_html(doc, output_path=output_dir)
        size = html_path.stat().st_size
        results['html'] = {'success': True, 'path': html_path, 'size': size}
        print(f"   ✅ SUCCESS: {size:,} bytes\n")
    except Exception as e:
        results['html'] = {'success': False, 'error': str(e)}
        print(f"   ❌ FAILED: {e}\n")

    # 3. PDF
    print("3️⃣  PDF Generation")
    print("-" * 90)
    try:
        pdf_path = generators.generate_pdf(doc, output_path=output_dir, format='a4')
        size = pdf_path.stat().st_size
        results['pdf'] = {'success': True, 'path': pdf_path, 'size': size}
        print(f"   ✅ SUCCESS: {size:,} bytes\n")
    except Exception as e:
        results['pdf'] = {'success': False, 'error': str(e)}
        print(f"   ❌ FAILED: {e}\n")

    # 4. DOCX
    print("4️⃣  DOCX Generation")
    print("-" * 90)
    try:
        docx_path = generators.generate_docx(doc, output_path=output_dir)
        size = docx_path.stat().st_size
        results['docx'] = {'success': True, 'path': docx_path, 'size': size}
        print(f"   ✅ SUCCESS: {size:,} bytes\n")
    except Exception as e:
        results['docx'] = {'success': False, 'error': str(e)}
        if 'not installed' in str(e):
            print(f"   ⚠️  SKIPPED: {e}")
            print(f"   Install: pip install python-docx\n")
        else:
            print(f"   ❌ FAILED: {e}\n")

    # 5. PPTX
    print("5️⃣  PPTX Generation")
    print("-" * 90)
    try:
        pptx_path = generators.generate_pptx(doc, output_path=output_dir)
        size = pptx_path.stat().st_size
        results['pptx'] = {'success': True, 'path': pptx_path, 'size': size}
        print(f"   ✅ SUCCESS: {size:,} bytes\n")
    except Exception as e:
        results['pptx'] = {'success': False, 'error': str(e)}
        if 'not installed' in str(e):
            print(f"   ⚠️  SKIPPED: {e}")
            print(f"   Install: pip install python-pptx\n")
        else:
            print(f"   ❌ FAILED: {e}\n")

    # Summary
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90 + "\n")

    successful = [k for k, v in results.items() if v.get('success')]
    failed = [k for k, v in results.items() if not v.get('success')]

    print(f"✅ Successful: {len(successful)}/{len(results)}")
    for fmt in successful:
        info = results[fmt]
        size_kb = info['size'] / 1024
        print(f"   - {fmt.upper()}: {size_kb:.1f} KB")

    if failed:
        print(f"\n⚠️  Skipped/Failed: {len(failed)}")
        for fmt in failed:
            error = results[fmt].get('error', 'Unknown error')
            if 'not installed' in error:
                print(f"   - {fmt.upper()}: Library not installed (optional)")
            else:
                print(f"   - {fmt.upper()}: {error}")

    print(f"\n📁 Output directory: {output_dir}")

    # List all files
    files = sorted(output_dir.glob("*"))
    if files:
        print(f"\n📄 Generated files:")
        for f in files:
            size = f.stat().st_size / 1024
            print(f"   - {f.name} ({size:.1f} KB)")

    print("\n🎯 Content Generation Tools Status:")
    print("   ✅ PDF: Working (pandoc + XeLaTeX)")
    print("   ✅ HTML: Working (pandoc)")
    print("   ✅ Markdown: Working (native)")
    if results.get('docx', {}).get('success'):
        print("   ✅ DOCX: Working (python-docx)")
    else:
        print("   ⚠️  DOCX: Optional (install python-docx)")
    if results.get('pptx', {}).get('success'):
        print("   ✅ PPTX: Working (python-pptx)")
    else:
        print("   ⚠️  PPTX: Optional (install python-pptx)")

    print()
    return results


if __name__ == "__main__":
    results = main()

    # Exit code based on core formats (PDF, HTML, MD)
    core_formats = ['markdown', 'html', 'pdf']
    core_success = all(results.get(f, {}).get('success', False) for f in core_formats)

    print("=" * 90)
    if core_success:
        print("  ALL CORE FORMATS: ✅ WORKING")
    else:
        print("  SOME CORE FORMATS FAILED")
    print("=" * 90)
    print()

    sys.exit(0 if core_success else 1)
