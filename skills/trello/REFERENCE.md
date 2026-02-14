# Trello Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_boards_tool`](#list_boards_tool) | List all boards accessible to the authenticated user. |
| [`get_board_tool`](#get_board_tool) | Get board details including its lists. |
| [`list_cards_tool`](#list_cards_tool) | List cards in a list or board. |
| [`create_card_tool`](#create_card_tool) | Create a new card in a Trello list. |
| [`update_card_tool`](#update_card_tool) | Update an existing card (move, rename, change description, etc. |
| [`add_comment_tool`](#add_comment_tool) | Add a comment to a card. |

---

## `list_boards_tool`

List all boards accessible to the authenticated user.

**Parameters:**

- **filter** (`str, optional`): Filter boards - 'all', 'open', 'closed', 'members', 'organization', 'public', 'starred' (default: 'all')
- **fields** (`str, optional`): Comma-separated fields to include (default: 'name,desc,url,closed,starred')
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether request succeeded - boards (list): List of board objects - board_count (int): Number of boards returned - error (str, optional): Error message if failed

---

## `get_board_tool`

Get board details including its lists.

**Parameters:**

- **board_id** (`str, required`): The Trello board ID or shortLink
- **include_lists** (`bool, optional`): Include lists in response (default: True)
- **include_members** (`bool, optional`): Include board members (default: False)
- **include_labels** (`bool, optional`): Include board labels (default: False)
- **list_filter** (`str, optional`): Filter lists - 'all', 'open', 'closed' (default: 'open')
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether request succeeded - board (dict): Board details - lists (list, optional): List of lists on the board - members (list, optional): List of board members - labels (list, optional): List of board labels - error (str, optional): Error message if failed

---

## `list_cards_tool`

List cards in a list or board.

**Parameters:**

- **list_id** (`str, optional`): The Trello list ID (use this OR board_id)
- **board_id** (`str, optional`): The Trello board ID (use this OR list_id)
- **filter** (`str, optional`): Filter cards - 'all', 'open', 'closed', 'visible' (default: 'open')
- **fields** (`str, optional`): Comma-separated fields to include
- **limit** (`int, optional`): Maximum number of cards (default: 100, max: 1000)
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether request succeeded - cards (list): List of card objects - card_count (int): Number of cards returned - error (str, optional): Error message if failed

---

## `create_card_tool`

Create a new card in a Trello list.

**Parameters:**

- **list_id** (`str, required`): The ID of the list to add the card to
- **name** (`str, required`): The name/title of the card
- **desc** (`str, optional`): The description of the card
- **pos** (`str/float, optional`): Position - 'top', 'bottom', or a positive float (default: 'bottom')
- **due** (`str, optional`): Due date in ISO 8601 format (e.g., '2024-12-31T23:59:59.000Z')
- **due_complete** (`bool, optional`): Whether the due date is complete (default: False)
- **id_labels** (`list, optional`): List of label IDs to add
- **id_members** (`list, optional`): List of member IDs to assign
- **url_source** (`str, optional`): URL to attach to the card
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether creation succeeded - card_id (str): ID of created card - card (dict): Full card object - url (str): URL of the created card - error (str, optional): Error message if failed

---

## `update_card_tool`

Update an existing card (move, rename, change description, etc.).

**Parameters:**

- **card_id** (`str, required`): The ID of the card to update
- **name** (`str, optional`): New name/title for the card
- **desc** (`str, optional`): New description for the card
- **closed** (`bool, optional`): Archive (True) or unarchive (False) the card
- **id_list** (`str, optional`): ID of the list to move the card to
- **id_board** (`str, optional`): ID of the board to move the card to
- **pos** (`str/float, optional`): New position - 'top', 'bottom', or a positive float
- **due** (`str, optional`): New due date in ISO 8601 format (null to remove)
- **due_complete** (`bool, optional`): Whether the due date is complete
- **id_labels** (`list, optional`): List of label IDs (replaces existing)
- **id_members** (`list, optional`): List of member IDs (replaces existing)
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether update succeeded - card (dict): Updated card object - error (str, optional): Error message if failed

---

## `add_comment_tool`

Add a comment to a card.

**Parameters:**

- **card_id** (`str, required`): The ID of the card to comment on
- **text** (`str, required`): The comment text
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether comment was added - comment_id (str): ID of the created comment - comment (dict): Full comment object - error (str, optional): Error message if failed
