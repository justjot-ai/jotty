# Content Repurposer Skill

## Description
Repurposes long-form content into multiple platform-specific formats: Twitter threads, LinkedIn posts, blog excerpts, email newsletters, and more.


## Type
derived

## Base Skills
- claude-cli-llm


## Capabilities
- document

## Tools

### repurpose_content_tool
Repurposes content for different platforms.

**Parameters:**
- `content` (str, required): Source content to repurpose
- `outputs` (list, required): List of desired output formats:
    - 'twitter_thread': Multi-tweet thread (8-10 tweets)
    - 'linkedin_post': Professional LinkedIn post (~1500 chars)
    - 'linkedin_carousel': 8-slide carousel outline
    - 'blog_excerpt': Compelling excerpt with hook
    - 'email_newsletter': Newsletter format with sections
- `title` (str, optional): Content title
- `custom_settings` (dict, optional): Platform-specific settings

**Returns:**
- `success` (bool): Whether repurposing succeeded
- `outputs` (dict): Repurposed content for each requested format
- `error` (str, optional): Error message if failed

## Triggers
- "content repurposer"

## Category
content-creation
