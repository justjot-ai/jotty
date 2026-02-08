# ArXiv Downloader Skill

## Description
Downloads and extracts content from arXiv papers. Supports 40+ conference/journal templates and can extract text from LaTeX source or compile to PDF.


## Type
base

## Tools

### download_arxiv_paper_tool
Downloads an arXiv paper and extracts its content.

**Parameters:**
- `arxiv_id` (str, required): arXiv ID (e.g., '2010.11929') or URL
- `extract_mode` (str, optional): 'text' (fast, from LaTeX) or 'pdf' (slow, compile then extract), default: 'text'
- `output_dir` (str, optional): Output directory for downloads, default: './output/arxiv'
- `clean_latex` (bool, optional): Remove LaTeX commands from text, default: True
- `include_bibliography` (bool, optional): Include bibliography section, default: True

**Returns:**
- `success` (bool): Whether download succeeded
- `arxiv_id` (str): Extracted arXiv ID
- `title` (str): Paper title
- `authors` (list): List of authors
- `content` (str): Extracted text content
- `output_path` (str): Path to downloaded files
- `error` (str, optional): Error message if failed

### search_arxiv_tool
Searches arXiv for papers matching a query.

**Parameters:**
- `query` (str, required): Search query
- `max_results` (int, optional): Maximum number of results, default: 10
- `sort_by` (str, optional): Sort order - 'relevance', 'submittedDate', 'lastUpdatedDate', default: 'relevance'

**Returns:**
- `success` (bool): Whether search succeeded
- `results` (list): List of papers with id, title, authors, abstract, url
- `count` (int): Number of results
- `error` (str, optional): Error message if failed
