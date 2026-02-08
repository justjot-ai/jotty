# Raffle Winner Picker Skill

Picks random winners from lists, spreadsheets, or files for giveaways, raffles, and contests with fair, unbiased selection.

## Description

This skill randomly selects winners from lists, spreadsheets, or CSV files for giveaways and contests. Uses cryptographically secure random selection and provides transparent results.


## Type
derived

## Base Skills
- calculator

## Tools

### `pick_raffle_winner_tool`

Pick random winner(s) from a list or file.

**Parameters:**
- `source` (str, required): Source - file path, Google Sheets URL, or list of entries
- `num_winners` (int, optional): Number of winners to pick (default: 1)
- `exclude` (list, optional): List of entries to exclude
- `weighted_column` (str, optional): Column name for weighted selection
- `output_file` (str, optional): Path to save results

**Returns:**
- `success` (bool): Whether selection succeeded
- `winners` (list): List of selected winners with details
- `total_entries` (int): Total number of entries
- `selection_method` (str): Method used for selection
- `timestamp` (str): Selection timestamp
- `output_file` (str, optional): Path to saved results
- `error` (str, optional): Error message if failed

## Usage Examples

### From CSV File

```python
result = await pick_raffle_winner_tool({
    'source': 'entries.csv',
    'num_winners': 3
})
```

### From List

```python
result = await pick_raffle_winner_tool({
    'source': ['Alice', 'Bob', 'Carol', 'David'],
    'num_winners': 1
})
```

## Dependencies

- `pandas`: For reading CSV/Excel files
- `secrets`: For cryptographically secure random selection
