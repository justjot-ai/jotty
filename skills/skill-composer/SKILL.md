# Skill Composer

Compose multiple skills into sequential, parallel, or conditional workflows.


## Type
derived

## Base Skills
- file-operations


## Capabilities
- code

## Tools

### compose_skills_tool

Compose multiple skills into a workflow.

**Parameters:**
- `workflow` (list, required): List of workflow steps

**Step Types:**

1. **Sequential**: Execute skills one after another
   ```json
   {
     "type": "sequential",
     "name": "create_and_convert",
     "skills": [
       {"skill": "file-operations", "tool": "write_file_tool", "params": {...}},
       {"skill": "document-converter", "tool": "convert_to_pdf_tool", "params": {...}}
     ]
   }
   ```

2. **Parallel**: Execute skills simultaneously
   ```json
   {
     "type": "parallel",
     "name": "parallel_searches",
     "skills": [
       {"skill": "web-search", "params": {"query": "Python"}},
       {"skill": "web-search", "params": {"query": "JavaScript"}}
     ]
   }
   ```

3. **Conditional**: Execute based on conditions
   ```json
   {
     "type": "conditional",
     "condition": {"type": "exists", "key": "search_results"},
     "then": [{"skill": "summarize", "params": {...}}],
     "else": [{"skill": "web-search", "params": {...}}]
   }
   ```

4. **Loop**: Repeat a skill
   ```json
   {
     "type": "loop",
     "count": 3,
     "skill": {"skill": "web-search", "params": {"query": "AI"}}
   }
   ```

5. **Single**: Execute one skill
   ```json
   {
     "type": "single",
     "skill": "file-operations",
     "tool": "write_file_tool",
     "params": {"path": "test.md", "content": "Hello"}
   }
   ```

**Parameter References:**
- Use `${step_name}` to reference previous outputs
- Use `${step_name.field}` to reference specific fields

**Example:**
```json
{
  "workflow": [
    {
      "type": "single",
      "name": "write_file",
      "skill": "file-operations",
      "tool": "write_file_tool",
      "params": {"path": "test.md", "content": "# Test"}
    },
    {
      "type": "single",
      "name": "convert_pdf",
      "skill": "document-converter",
      "tool": "convert_to_pdf_tool",
      "params": {"input_file": "${write_file.output.file_path}"}
    }
  ]
}
```

## Use Cases

- PDF generation workflows
- Research pipelines
- Data processing chains
- Multi-step automation
- Parallel data collection

## Triggers
- "skill composer"

## Category
workflow-automation
