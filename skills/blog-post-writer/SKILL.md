---
name: writing-blog-posts
description: "Generate blog post outlines with SEO metadata, headings, and section templates. Use when the user wants to write blog post, blog outline, article structure."
---

# Blog Post Writer Skill

Generate blog post outlines with SEO metadata, headings, and section templates. Use when the user wants to write blog post, blog outline, article structure.

## Type
base

## Capabilities
- generate
- document

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
- "blog post"
- "blog outline"
- "article"
- "blog writing"
- "content outline"

## Category
content-creation

## Tools

### generate_blog_outline_tool
Generate a blog post outline with SEO metadata.

**Parameters:**
- `title` (str, required): Blog post title
- `keywords` (list, optional): Target SEO keywords
- `sections` (int, optional): Number of sections (default: 5)
- `tone` (str, optional): professional, casual, academic (default: professional)
- `word_target` (int, optional): Target word count (default: 1500)

**Returns:**
- `success` (bool)
- `outline` (dict): Structured blog outline
- `seo_meta` (dict): SEO metadata suggestions

## Dependencies
None
