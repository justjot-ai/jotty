# Trello Skill

## Description
Integrates with Trello API to manage boards, lists, and cards. Supports listing boards, viewing board details with lists, managing cards (create, update, move), and adding comments.


## Type
base


## Capabilities
- data-fetch

## Configuration
Credentials are loaded from:
1. Environment variables: `TRELLO_API_KEY` and `TRELLO_TOKEN`
2. Config files: `~/.config/trello/api_key` and `~/.config/trello/token`

To get your API key and token:
1. Visit https://trello.com/power-ups/admin to get your API key
2. Generate a token by visiting: `https://trello.com/1/authorize?expiration=never&scope=read,write&response_type=token&key=YOUR_API_KEY`

## Tools

### list_boards_tool
List all boards accessible to the authenticated user.

**Parameters:**
- `filter` (str, optional): Filter boards - `all`, `open`, `closed`, `members`, `organization`, `public`, `starred` (default: `all`)
- `fields` (str, optional): Comma-separated fields to include

### get_board_tool
Get board details including its lists.

**Parameters:**
- `board_id` (str, required): The Trello board ID or shortLink
- `include_lists` (bool, optional): Include lists in response (default: True)
- `include_members` (bool, optional): Include board members (default: False)
- `include_labels` (bool, optional): Include board labels (default: False)
- `list_filter` (str, optional): Filter lists - `all`, `open`, `closed` (default: `open`)

### list_cards_tool
List cards in a list or board.

**Parameters:**
- `list_id` (str, optional): The Trello list ID (use this OR board_id)
- `board_id` (str, optional): The Trello board ID (use this OR list_id)
- `filter` (str, optional): Filter cards - `all`, `open`, `closed`, `visible` (default: `open`)
- `fields` (str, optional): Comma-separated fields to include
- `limit` (int, optional): Maximum number of cards (default: 100, max: 1000)

### create_card_tool
Create a new card in a Trello list.

**Parameters:**
- `list_id` (str, required): The ID of the list to add the card to
- `name` (str, required): The name/title of the card
- `desc` (str, optional): The description of the card
- `pos` (str/float, optional): Position - `top`, `bottom`, or a positive float (default: `bottom`)
- `due` (str, optional): Due date in ISO 8601 format (e.g., `2024-12-31T23:59:59.000Z`)
- `due_complete` (bool, optional): Whether the due date is complete (default: False)
- `id_labels` (list, optional): List of label IDs to add
- `id_members` (list, optional): List of member IDs to assign
- `url_source` (str, optional): URL to attach to the card

### update_card_tool
Update an existing card (move, rename, change description, etc.).

**Parameters:**
- `card_id` (str, required): The ID of the card to update
- `name` (str, optional): New name/title for the card
- `desc` (str, optional): New description for the card
- `closed` (bool, optional): Archive (True) or unarchive (False) the card
- `id_list` (str, optional): ID of the list to move the card to
- `id_board` (str, optional): ID of the board to move the card to
- `pos` (str/float, optional): New position - `top`, `bottom`, or a positive float
- `due` (str, optional): New due date in ISO 8601 format (null to remove)
- `due_complete` (bool, optional): Whether the due date is complete
- `id_labels` (list, optional): List of label IDs (replaces existing)
- `id_members` (list, optional): List of member IDs (replaces existing)

### add_comment_tool
Add a comment to a card.

**Parameters:**
- `card_id` (str, required): The ID of the card to comment on
- `text` (str, required): The comment text

## Requirements
- `requests` library
- Trello API key and token with read/write permissions

## Usage Examples

**List all boards:**
```python
result = list_boards_tool({'filter': 'open'})
```

**Get board with lists:**
```python
result = get_board_tool({
    'board_id': 'abc123',
    'include_lists': True,
    'include_labels': True
})
```

**List cards in a list:**
```python
result = list_cards_tool({'list_id': 'list123', 'filter': 'open'})
```

**List all cards on a board:**
```python
result = list_cards_tool({'board_id': 'board123'})
```

**Create a card:**
```python
result = create_card_tool({
    'list_id': 'list123',
    'name': 'New Task',
    'desc': 'Task description here',
    'due': '2024-12-31T23:59:59.000Z',
    'pos': 'top'
})
```

**Move a card to another list:**
```python
result = update_card_tool({
    'card_id': 'card123',
    'id_list': 'newlist456'
})
```

**Update card name and description:**
```python
result = update_card_tool({
    'card_id': 'card123',
    'name': 'Updated Task Name',
    'desc': 'Updated description'
})
```

**Archive a card:**
```python
result = update_card_tool({
    'card_id': 'card123',
    'closed': True
})
```

**Add a comment:**
```python
result = add_comment_tool({
    'card_id': 'card123',
    'text': 'This is a comment on the card'
})
```

## Triggers
- "trello"
- "trello board"
- "trello card"
- "create"

## Category
workflow-automation
