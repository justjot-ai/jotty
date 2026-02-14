---
name: managing-archives
description: "Create and extract ZIP, TAR, and GZIP archives using Python stdlib. Use when the user wants to zip, unzip, tar, extract archive, compress files."
---

# Archive Manager Skill

Create and extract ZIP, TAR, and GZIP archives using Python stdlib. Use when the user wants to zip, unzip, tar, extract archive, compress files.

## Type
base

## Capabilities
- code

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "zip"
- "unzip"
- "tar"
- "archive"
- "compress"
- "extract"
- "gzip"

## Category
development

## Tools

### create_archive_tool
Create an archive from files.

**Parameters:**
- `files` (list, required): List of file paths to archive
- `output` (str, required): Output archive path
- `format` (str, optional): zip, tar, tar.gz, tar.bz2 (default: zip)

**Returns:**
- `success` (bool)
- `output_path` (str): Path to created archive
- `file_count` (int): Number of files archived
- `size_bytes` (int): Archive size in bytes

### extract_archive_tool
Extract an archive.

**Parameters:**
- `archive_path` (str, required): Path to archive
- `output_dir` (str, optional): Extraction directory (default: current dir)

**Returns:**
- `success` (bool)
- `extracted_files` (list): List of extracted file paths
- `output_dir` (str): Extraction directory

### list_archive_tool
List contents of an archive.

**Parameters:**
- `archive_path` (str, required): Path to archive

**Returns:**
- `success` (bool)
- `files` (list): List of files in archive
- `total_size` (int): Total uncompressed size

## Dependencies
None
