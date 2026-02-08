# Content Pipeline Skill

## Description
Content pipeline toolkit that orchestrates document processing flows: Source -> Processors -> Sinks.
Wraps JustJot.ai's ContentPipeline for integration with Jotty's skill system.

Supports converting content from various sources (markdown, arxiv, html, pdf, youtube) through
processors (diagram rendering, latex handling, image downloading) to multiple output formats
(pdf, epub, markdown, docx, html, remarkable).


## Type
composite

## Base Skills
- web-search
- document-converter

## Execution
sequential


## Capabilities
- document
- research

## Tools

### run_pipeline_tool
Run a complete content pipeline: Source -> Processors -> Sinks.

**Parameters:**
- `source_type` (str, required): Source adapter type: "markdown", "arxiv", "html", "pdf", "youtube"
- `source_params` (dict, required): Parameters for the source adapter
- `processors` (list, optional): List of processor configs: [{"type": "diagram_renderer"}, {"type": "latex_handler"}]
- `sinks` (list, optional): List of sink configs: [{"type": "pdf", "format": "remarkable"}]
- `output_dir` (str, optional): Output directory path

**Source Types:**
- `markdown`: Content from markdown text or file
  - `content` (str): Direct markdown content
  - `file_path` (str): Path to markdown file
- `arxiv`: ArXiv paper source
  - `arxiv_id` (str): ArXiv paper ID (e.g., "1706.03762")
- `html`: HTML page source
  - `url` (str): URL to fetch
  - `content` (str): Direct HTML content
- `pdf`: PDF document source
  - `file_path` (str): Path to PDF file
- `youtube`: YouTube video transcript
  - `url` (str): YouTube video URL

**Processor Types:**
- `diagram_renderer`: Render Mermaid, PlantUML, Graphviz diagrams to images
- `latex_handler`: Process LaTeX math expressions
- `image_downloader`: Download remote images to local files
- `syntax_fixer`: Fix common markdown syntax issues

**Sink Types:**
- `pdf`: Generate PDF (supports "remarkable", "a4", "letter", "kindle" formats)
- `epub`: Generate EPUB ebook
- `markdown`: Write processed markdown
- `docx`: Generate Word document
- `html`: Generate HTML page
- `remarkable`: Send directly to reMarkable device

### run_source_tool
Run only the source stage to generate a Document.

**Parameters:**
- `source_type` (str, required): Source adapter type
- `source_params` (dict, required): Parameters for the source adapter

**Returns:**
- `success` (bool): Whether source generation succeeded
- `document` (dict): Generated document (serialized)
- `error` (str, optional): Error message if failed

### process_document_tool
Process an existing document through processors.

**Parameters:**
- `document` (dict, required): Document dictionary (from run_source_tool)
- `processors` (list, required): List of processor configs

**Returns:**
- `success` (bool): Whether processing succeeded
- `document` (dict): Processed document (serialized)
- `error` (str, optional): Error message if failed

### sink_document_tool
Write a document to one or more sinks.

**Parameters:**
- `document` (dict, required): Document dictionary (from process_document_tool)
- `sinks` (list, required): List of sink configs
- `output_dir` (str, optional): Output directory path

**Returns:**
- `success` (bool): Whether sink writing succeeded
- `output_paths` (list): List of generated file paths
- `error` (str, optional): Error message if failed

## Examples

### Convert ArXiv paper to reMarkable PDF
```python
result = run_pipeline_tool({
    'source_type': 'arxiv',
    'source_params': {'arxiv_id': '1706.03762'},
    'processors': [{'type': 'latex_handler'}],
    'sinks': [{'type': 'pdf', 'format': 'remarkable'}]
})
```

### Convert YouTube video to EPUB
```python
result = run_pipeline_tool({
    'source_type': 'youtube',
    'source_params': {'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'},
    'sinks': [{'type': 'epub'}]
})
```

### Process markdown with diagrams
```python
result = run_pipeline_tool({
    'source_type': 'markdown',
    'source_params': {'content': '# Test\n\n```mermaid\ngraph LR\nA-->B\n```'},
    'processors': [{'type': 'diagram_renderer'}],
    'sinks': [{'type': 'pdf'}]
})
```

## Dependencies
- JustJot.ai core modules (auto-imported if available)
- Optional: pandoc, latex, mermaid-cli for full functionality
