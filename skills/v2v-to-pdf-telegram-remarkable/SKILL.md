# V2V to PDF + Telegram + reMarkable Skill

Complete workflow: Search V2V trending topics → Generate PDF → Send to Telegram and reMarkable.

## Description

This composite skill combines:
1. V2V trending search
2. PDF generation
3. Telegram sending
4. reMarkable uploading

## Usage

```python
from skills.v2v_to_pdf_telegram_remarkable.tools import v2v_to_pdf_and_send_tool

result = await v2v_to_pdf_and_send_tool({
    'query': 'multi agent systems',
    'send_telegram': True,
    'send_remarkable': True
})
```

## Parameters

- `query` (str, optional): Search query (default: 'trending topics')
- `title` (str, optional): Report title
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `send_remarkable` (bool, optional): Send to reMarkable (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID
- `remarkable_folder` (str, optional): reMarkable folder (default: '/')
